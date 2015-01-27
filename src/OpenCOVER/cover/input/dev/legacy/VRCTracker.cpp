/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// CLASS VRCTracker
//
// Initial version: 2001-07-02 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "coRawMouse.h"
#include <util/common.h>
#include <errno.h>
#include <signal.h>
#ifndef _WIN32
//#include <termio.h>
#include <sys/socket.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#ifndef __APPLE__
#include <sys/prctl.h>
#endif
#ifdef __linux__
#define sigset signal
#endif
#ifndef _WIN32
#define closesocket close
#endif

#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <netinet/in.h>
#include <arpa/inet.h>
//For closing a socket you need
//closesocket under windows
//but close under linux
#else
typedef unsigned long in_addr_t;
#include <windows.h>
#define ioctl ioctlsocket
#endif

#include <sysdep/net.h>
#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>

#include "cover/input/coMousePointer.h"
#include "VRCTracker.h"

// 1st shm key we try
#define SHMKEY ((key_t)127)

//
#define PERMS 0666
using namespace covise;

// currently only print 8 buttons
inline const char *prBut(unsigned int val)
{
    static char buffer[9];
    buffer[8] = '\0';
    buffer[7] = (val & 0x01) ? '+' : '.';
    buffer[6] = (val & 0x02) ? '+' : '.';
    buffer[5] = (val & 0x04) ? '+' : '.';
    buffer[4] = (val & 0x08) ? '+' : '.';
    buffer[3] = (val & 0x10) ? '+' : '.';
    buffer[2] = (val & 0x20) ? '+' : '.';
    buffer[1] = (val & 0x40) ? '+' : '.';
    buffer[0] = (val & 0x80) ? '+' : '.';
    return buffer;
}

/////////////////////////////////////////////////////////////////////////////

// last argument: no options defined so far
VRCTracker::VRCTracker(int portnumber, int debugLevel, float scale, const char *)
{
    d_scale = scale;
    unit = 10.0;

    // define different debug states
    d_debugLevel = debugLevel;
    d_rawDump = NULL;
    if (d_debugLevel > 0)
    {
        fprintf(stderr, "+VRC+ =========================================\n");
        fprintf(stderr, "+VRC+ Starting Receiver: Port=%d\n", portnumber);
    }
    if (d_debugLevel > 1)
        fprintf(stderr, "+VRC+ Dumping Tracking values\n");
    if (d_debugLevel > 2)
    {
        char filename[64];
        sprintf(filename, "VRCTracker.%d", getpid());
        d_rawDump = fopen(filename, "w");
        if (!d_rawDump)
        {
            sprintf(filename, "/var/tmp/VRCTracker.%d", getpid());
            d_rawDump = fopen(filename, "w");
        }
        if (d_rawDump)
            fprintf(stderr, "+VRC+ Dumping packets to %s\n", filename);
        else
            fprintf(stderr, "+VRC+ Binary dump requested, but could not write\n");
    }

    // allocate a shared memory segment between COVER and tracking receiver
    d_stationData = allocSharedMemoryData();

    // open the socket
    d_socket = 0;
    d_socket = openUDPPort(portnumber);
    if (d_socket < 0)
        return;

    std::string mcastaddr = coCoviseConfig::getEntry("COVER.Input.VRC.Multicast");
    if (!mcastaddr.empty())
    {
        in_addr_t addr = inet_addr(mcastaddr.c_str());

        if (addr == INADDR_NONE)
        {
            fprintf(stderr, "+VRC+ failed to convert multicast address %s, exiting\n", mcastaddr.c_str());
            if (d_socket)
            {
                closesocket(d_socket);
                d_socket = 0;
            }
        }
        else
        {
            struct ip_mreq mreq = {
                { addr },
                { INADDR_ANY }
            };

            int enable = 1;
#ifdef WIN32
            if (-1 == setsockopt(d_socket, IPPROTO_IP, SO_REUSEADDR,
                                 (char *)&enable, sizeof(enable)))
#else
            if (-1 == setsockopt(d_socket, IPPROTO_IP, SO_REUSEADDR,
                                 &enable, sizeof(enable)))
#endif
            {
                fprintf(stderr, "+VRC+ failed to enable SO_REUSEADDR: %s",
                        strerror(errno));
            }

#ifdef WIN32
            if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                                 (char *)&mreq, sizeof(mreq)))
#else
            if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                                 &mreq, sizeof(mreq)))
#endif
            {
                fprintf(stderr, "+VRC+ failed to join multicast group %s: %s, exiting\n",
                        mcastaddr.c_str(), strerror(errno));
                if (d_socket)
                    closesocket(d_socket);
                d_socket = 0;
            }
        }
    }

    // we send binary - check sizes
    if (sizeof(float) != 4 || sizeof(int) != 4)
    {
        fprintf(stderr, "+VRC+ VRCTracker requires 4-byte float & int ... exiting\n");
        if (d_socket)
            closesocket(d_socket);
        d_socket = 0;
    }

    d_debugTracking = false;
    d_debugButtons = false;
    d_debugStation = -1;
    std::string entry;
    unit = coCoviseConfig::getFloat("COVER.Input.VRC.Unit", 10.0);
    entry = coCoviseConfig::getEntry("COVER.Input.DebugTracking");
    if (strcasecmp(entry.c_str(), "RAW") == 0)
        d_debugTracking = true;
    d_debugButtons = coCoviseConfig::isOn("COVER.Input.DebugButtons", false);
    coCoviseConfig::isOn("COVER.DebugButtons", false);

    d_debugStation = coCoviseConfig::getInt("COVER.Input.DebugStation", 0);
}

/////////////////////////////////////////////////////////////////////////////

// if we ever leane, we do it here...
VRCTracker::~VRCTracker()
{
    closesocket(d_socket);
}

/////////////////////////////////////////////////////////////////////////////

// get a UDP port to receive binary dara
int VRCTracker::openUDPPort(int portnumber)
{

    // CREATING UDP SOCKET
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket < 0)
    {
        fprintf(stderr, "+VRC+ socket creation failed\n");
        return -1;
    }

    // FILL SOCKET ADRESS STRUCTURE
    sockaddr_in any_adr;

    memset((char *)&any_adr, 0, sizeof(any_adr));
    any_adr.sin_family = AF_INET;
    any_adr.sin_addr.s_addr = INADDR_ANY;
    any_adr.sin_port = htons(portnumber);

    // BIND TO A LOCAL PROTOCOL PORT
    if (bind(sock, (sockaddr *)&any_adr, sizeof(any_adr)) < 0)
    {
        fprintf(stderr, "+VRC+ could not bind to port %d\n", portnumber);
        return -1;
    }
    return sock;
}

/////////////////////////////////////////////////////////////////////////////

void VRCTracker::receiveData()
{
    sockaddr remote_adr;
    socklen_t rlen;

    // read into a buffer first, copy only vaild parts afterwards
    const int BUFSIZE = 2048;
    char rawdata[BUFSIZE];

// check wether we already received package
#ifdef WIN32
    u_long bytes = 0;
#else //WIN32
    size_t bytes = 0;
#endif //WIN32
    // warum sollte man das hier berhaupt machen?
    // das select reicht doch...
    int retVal = ioctl(d_socket, FIONREAD, &bytes);
    if (retVal < 0)
    {
        perror("+VRC+");
        return;
    }

    // if no data: print message after 5 sec.
    if (bytes <= 0)
    {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(d_socket, &readfds);
        struct timeval timeout = { 5, 0 };
        if (0 == select(
                     d_socket + 1, // param nfds: specifies the range of descriptors to be tested;
                     // descriptors in [0;nfds-1] are examined
                     &readfds, NULL, NULL, &timeout))
        {
            cerr << "+VRC+ No data received" << endl;
            return;
        }
    }

    // receive a package
    rlen = sizeof(remote_adr);
    int numbytes = recvfrom(
        d_socket, // specifies the socket file descriptor
        rawdata, // container for received data
        BUFSIZE - 1, // one char reserved for trailing '\0' (see below)
        0, // flags
        &remote_adr, // actual sending socket address
        &rlen); // size of remote_adr
#ifdef WIN32
    int err = WSAGetLastError();
    if (numbytes < 0)
    {
        fprintf(stderr, "+VRC+ !! error: recvfrom failed w/ %d vvvvvvvvvvvvvv\n", err);
        fprintf(stderr, "+VRC+ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    } // fi
#endif //WIN32
    if (d_debugLevel > 3 && d_rawDump)
    {
        int iread = fwrite(rawdata, 1, numbytes, d_rawDump);
        if (iread == 0)
            fprintf(stderr, "error in fwrite of rawdata \n");
    }

    if (numbytes == BUFSIZE)
        fprintf(stderr, "+VRC+ Received long message, ignoring rest\n");

    // terminate string
    rawdata[numbytes] = '\0';

    StationOutput d;
    int numRead;
    char magic[BUFSIZE];
    numRead = sscanf(rawdata, "%s %d %d [%f %f %f] - [%f %f %f %f %f %f %f %f %f] - [%f %f]",
                     magic, &d.stationID, &d.buttonVal, &d.x, &d.y, &d.z,
                     &d.mat[0], &d.mat[1], &d.mat[2], &d.mat[3], &d.mat[4],
                     &d.mat[5], &d.mat[6], &d.mat[7], &d.mat[8],
                     &d.analog[0], &d.analog[1]);

    // read correctly: copy data to correct field if station number ok
    if (numRead == 17 && d.stationID <= MAX_STATIONS)
    {
        memcpy(&(d_stationData[d.stationID]), &d, sizeof(StationOutput));
        if (d_debugLevel > 1)
        {
            fprintf(stderr, "+VRC+ %s\n", rawdata);
            fprintf(stderr, "+VRC+ %s %d %d [%f %f %f] - [%f %f %f %f %f %f %f %f %f] - [%f %f]\n",
                    magic, d_stationData[d.stationID].stationID, d_stationData[d.stationID].buttonVal, d_stationData[d.stationID].x, d_stationData[d.stationID].y, d_stationData[d.stationID].z,
                    d_stationData[d.stationID].mat[0], d_stationData[d.stationID].mat[1], d_stationData[d.stationID].mat[2], d_stationData[d.stationID].mat[3], d_stationData[d.stationID].mat[4],
                    d_stationData[d.stationID].mat[5], d_stationData[d.stationID].mat[6], d_stationData[d.stationID].mat[7], d_stationData[d.stationID].mat[8],
                    d.analog[0], d.analog[1]);
        }
    }
    else
    {
        fprintf(stderr, "+VRC+ Received illegal data  vvvvvvvvvvvvvv\n");
        fprintf(stderr, "+VRC+ !! %s, stationid=%d, numbytes=%d\n", rawdata, d.stationID, numbytes);
        fprintf(stderr, "+VRC+ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    }
}

void receiveLoop(void *userdata)
{

    VRCTracker *tr = (VRCTracker *)userdata;

    while (1)
    {
        tr->receiveData();
#ifndef WIN32
        if (getppid() == 1)
        {
            exit(1);
        }
#endif
    }
}

void
VRCTracker::mainLoop()
{
#ifndef WIN32
    int ret;
    ret = fork();
    if (ret == -1)
    {
        fprintf(stderr, "+VRC+ fork of VRCTracker server failed\n");
    }
    else if (ret == 0) // child process
    {
        // child code
        // read serial port and write data to shared memory
        if (d_debugLevel > 0)
            fprintf(stderr, "+VRC+ VRCTracker server running\n");
#if !defined(__linux__) && !defined(__APPLE__)
        prctl(PR_TERMCHILD); // Exit when parent does
#endif
        sigset(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
#endif

#ifdef WIN32
        // Not used: "uintptr_t thread = "
        _beginthread(receiveLoop, 0, this);
#else

    receiveLoop(this);
}
else
{
    if (d_debugLevel > 0)
        fprintf(stderr, "+VRC+ VRCTracker server forked, pid=%d\n", ret);
}
#endif
    }

    VRCTracker::StationOutput *
    VRCTracker::allocSharedMemoryData()
    {
#ifndef WIN32
        // get shared memory segment for tracker output data ds1, ds2. ds3, ds4
        int shmid;
        key_t shmkey = SHMKEY;

        // allocate 4 int as header : numStations + MAGIC + 2 spares (alignment)
        int size = MAX_STATIONS * sizeof(StationOutput);

        while ((shmid = shmget(shmkey, size, PERMS | IPC_CREAT)) < 0)
        {
            if (d_debugLevel > 0)
                fprintf(stderr, "+VRC+ ShmGet failed\n");
            shmkey++;
        }

        if (d_debugLevel > 0)
            fprintf(stderr, "+VRC+ SHM area of %d allocated under key %d\n",
                    size, shmkey);

        d_stationData = (StationOutput *)shmat(shmid, (char *)0, 0);
#else

    d_stationData = new StationOutput[MAX_STATIONS];
#endif
        memset(d_stationData, 0, MAX_STATIONS * sizeof(StationOutput));
        return d_stationData;
    }

    void
    VRCTracker::getPositionMatrix(int station, float &x, float &y, float &z, float &m00, float &m01, float &m02, float &m10, float &m11, float &m12, float &m20, float &m21, float &m22)
    {
        if (station >= 0 && station < MAX_STATIONS)
        {
            x = d_stationData[station].x * d_scale;
            y = d_stationData[station].y * d_scale;
            z = d_stationData[station].z * d_scale;

            m00 = d_stationData[station].mat[0];
            m01 = d_stationData[station].mat[1];
            m02 = d_stationData[station].mat[2];

            m10 = d_stationData[station].mat[3];
            m11 = d_stationData[station].mat[4];
            m12 = d_stationData[station].mat[5];

            m20 = d_stationData[station].mat[6];
            m21 = d_stationData[station].mat[7];
            m22 = d_stationData[station].mat[8];
        }
        else
            fprintf(stderr, "+VRC+ Error: getPositionMatrix(station=%d)\n", station);
    }

    void
    VRCTracker::getMatrix(int station, osg::Matrix &mat)
    {
        if (station >= 0 && station < MAX_STATIONS)
        {
            mat.makeIdentity();
            float m30, m31, m32, m00, m01, m02, m10, m11, m12, m20, m21, m22;
            getPositionMatrix(station, m30, m31, m32,
                              m00, m01, m02,
                              m10, m11, m12,
                              m20, m21, m22);
            mat.set(m00, m01, m02, 0.0,
                    m10, m11, m12, 0.0,
                    m20, m21, m22, 0.0,
                    m30, m31, m32, 1.0);

            if (d_debugTracking && (station == d_debugStation))
                fprintf(stderr, "VRC motion device [%2d]: raw position  [X: %3.3f Y: %3.3f Z: %3.3f]\n", station, m30, m31, m32);
        }
        else
            fprintf(stderr, "+VRC+ Error: getPositionMatrix(station=%d)\n", station);
    }

    unsigned int
    VRCTracker::getButtons(int station)
    {
#ifdef CO_tiger
        // just report mouse buttons
        return cover->getMouseButtons();
#endif
        if (station >= 0 && station < MAX_STATIONS)
        {
            return d_stationData[station].buttonVal;
        }
        else
        {
            fprintf(stderr, "+VRC+ Error: getButtons(station=%d)\n", station);
            return 0;
        }
    }

    unsigned int VRCTracker::getButton(int station)
    {
        unsigned int button;
        button = getButtons(station);

        if (d_debugButtons && (station == d_debugStation))
            fprintf(stderr, "VRC button device [%2d]: unmapped button [B: %d]\n", station, button);

        return (button);
    }

    void
    VRCTracker::getAnalog(int station, float &d1, float &d2)
    {
        if (station >= 0 && station < MAX_STATIONS)
        {
            d1 = d_stationData[station].analog[0];
            d2 = d_stationData[station].analog[1];
        }
        else
        {
            fprintf(stderr, "+VRC+ Error: getAnalog(station=%d)\n", station);
            return;
        }
    }
