/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			polhemusdrvr.C 				*
 *									*
 *	Description		polhemus driver class			*
 *				extra process for reading the		*
 *				serial port				*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			January 5th '97				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#ifndef STANDALONE
#include <config/CoviseConfig.h>
#endif

#include <util/common.h>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef _WIN32
#include <termios.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#endif

#include <sys/types.h>

#include "polhemusdrvr.h"

#ifndef false
#define false 0
#endif

#ifndef true
#define true 1
#endif

#define DOWN 0
#define UP 1

#define SHMKEY ((key_t)123)

#define PERMS 0666

//#define   DEBUG
//#undef	DEBUG
//#undef    SIM

fastrak::fastrak(const char *portname, int baudrate, int numStations, int btdev)
{
    fprintf(stderr, "---- in fastrak::fastrak\n");
    fprintf(stderr, "------ %s %d %d\n", portname, baudrate, numStations);

    strcpy(serialPortName, portname);
    this->baudrate = baudrate;
    numActiveStations = numStations;
    buttonDevice = btdev;
    fprintf(stderr, "ButtonDevice: %d\n", buttonDevice);
    if (buttonDevice == BUTTONDEVICE_STYLUS)
        bufLen = 56;
    else
        bufLen = 59;
    // initial values for hemisphere
    hx = 1;
    hy = 0;
    hz = 0;
    // init ds1, ds2, ds3, ds4 (tracker data)
    allocSharedMemoryData();

    if (openSerialPort() == -1)
    {
        cout << "FATAL: aborting polhemus driver" << endl;
        exit(-1);
    }
    else
        cout << "INFO: serial port " << serialPortName << " open" << endl;

    // set all stations to "DOWN"
    s1 = s2 = s3 = s4 = DOWN;
    //numActiveStations = 0;

    // configuration
    //reinitialize();
    disableContinousOutput();
    setUnitCentimeters();
    setAsciiFormat();
}

fastrak::~fastrak()
{
#ifdef WIN32
    CloseHandle(desc);
#else
    close(desc);
#endif
}

int
fastrak::testConnection()
{
    fprintf(stderr, "fastrak::testConnection\n");
    if (desc < 0)
        return (false);
    else
        return (true);
}

void
fastrak::setHemisphere(int station, float x, float y, float z)
{
    hx = x;
    hy = y;
    hz = z;

#ifdef DEBUG
    cout << "INFO: Setting hemisphere: ";
#endif
    char s[500];
    sprintf(s, "H%d,%10f,%10f,%10f", station, hx, hy, hz);

    sendFastrakCmd(s);
}

void
fastrak::setPositionFilter(float f, float flow, float fhigh, float factor)
{
#ifdef DEBUG
    cout << "CONFIG: Setting position filter: ";
#endif
    char s[500];

    sprintf(s, "x,%10f,%10f,%10f,%10f", f, flow, fhigh, factor);
    sendFastrakCmd(s);
}

void
fastrak::setAttitudeFilter(float f, float flow, float fhigh, float factor)
{
#ifdef DEBUG
    cout << "CONFIG: Setting attitude filter: ";
#endif
    char s[500];

    sprintf(s, "v,%10f,%10f,%10f,%10f", f, flow, fhigh, factor);
    sendFastrakCmd(s);
}

void
fastrak::setStation(int station)
{
    switch (station)
    {
    case 1:
        // test if already active
        if (s1 == DOWN)
        {
            s1 = UP;
            setStationActive(1);
            resetReferenceFrame(station);
            setOutput(station);
        }
        break;
    case 2:
        if (s2 == DOWN)
        {
            s2 = UP;
            setStationActive(2);
            resetReferenceFrame(station);
            setOutput(station);
        }
        break;
    case 3:
        if (s3 == DOWN)
        {
            s3 = UP;
            setStationActive(3);
            resetReferenceFrame(station);
            setOutput(station);
        }
        break;

    case 4:
        if (s4 == DOWN)
        {
            s4 = UP;
            setStationActive(4);
            resetReferenceFrame(station);
            setOutput(station);
        }
        break;

    default:
        cout << "Invalid station number" << endl;
        break;
    }

#ifdef DEBUG
    cout << "INFO: station status: " << s1 << s2 << s3 << s4 << endl;
#endif
}

void
fastrak::unsetStation(int station)
{

    switch (station)
    {
    case 1:
        if (s1 == UP)
        {
            s1 = DOWN;
            setStationPassive(1);
        }
        break;
    case 2:
        if (s2 == UP)
        {
            s2 = DOWN;
            setStationPassive(2);
        }
        break;
    case 3:
        if (s3 == UP)
        {
            s3 = DOWN;
            setStationPassive(3);
        }
        break;

    case 4:
        if (s4 == UP)
        {
            s4 = DOWN;
            setStationPassive(4);
        }
        break;

    default:
        cout << "Invalid station number" << endl;
        break;
    }
}

void
fastrak::calibrateStation(int station, float Ox, float Oy, float Oz,
                          float Xx, float Xy, float Xz, float Yx, float Yy, float Yz)
{

    resetReferenceFrame(station);

    setReferenceFrame(station, Ox, Oy, Oz, Xx, Xy, Xz, Yx, Yy, Yz);
}

void fastrak::continuousThread(void *data)
{
    fastrak *bt = (fastrak *)data;
    while (1)
    {
        bt->readActiveStations();
    }
}

void
fastrak::start()
{
//fprintf(stderr,"---- in fastrak::start\n");

#ifdef _WIN32
    _beginthread(continuousThread, 0, this);
#else
    int ret;
    ret = fork();
    if (ret == -1)
    {
        //cout << "fork failed" << endl;
    }

    else if (ret == 0) // child process
    {
        // child code
        // read serial port and write data to shared memory

        //cout << "INFO: server forked" << endl;

        while (1)
        {
            //fprintf(stderr,"------ fastrak server processes is running....\n");

            readActiveStations();
            if (getppid() == 1)
            {
                //fprintf(stderr, "SERVER: exit\n");
                exit(1);
            }
        }
    }
#endif
    //else
    //cout << "INFO: client is running" << endl;
}

void
fastrak::getAbsPositions(int station, float *x, float *y, float *z)
{
    switch (station)
    {
    case 1:
        *x = ds1->x;
        *y = ds1->y;
        *z = ds1->z;
        break;

    case 2:
        *x = ds2->x;
        *y = ds2->y;
        *z = ds2->z;
        //printf("... CLIENT got %f %f %f\n", ds2->x, ds2->y, ds2->z);
        break;

    case 3:
        *x = ds3->x;
        *y = ds3->y;
        *z = ds3->z;
        break;

    case 4:
        *x = ds4->x;
        *y = ds4->y;
        *z = ds4->z;
        break;

    default:
        printf("... fastrak::getAbsPositions Invalid Station Number\n");
        break;
    }
}

void
fastrak::printAbsPositions(int station)
{
    //system("clear");

    switch (station)
    {
    case 1:
        //cout << "	x	y	z" << endl;
        cout << "	" << ds1->x << "	" << ds1->y << "	" << ds1->z << endl;
        break;
    case 2:
        cout << "Polyhemus: not implemented" << endl;
        break;
    case 3:
        cout << "Polyhemus: not implemented" << endl;
        break;
    case 4:
        cout << "Polyhemus: not implemented" << endl;
        break;

    default:
        cout << "INFO: unknown station" << endl;
    }
}

void
    /*station*/
    fastrak::getRelPositions(int, float * /*dx*/, float * /*dy*/, float * /*dz*/)
{
    cout << "get rel pos not implemented " << endl;
    //(station, dx, dy, dz);
}

void
    /*station*/
    fastrak::getEulerAngles(int, float * /*az*/, float * /*el*/, float * /*roll*/)
{
    cout << "get euler angles not implemented \n";
    //(station, az, el, roll);
}

void
fastrak::getQuaternions(int station, float *w, float *q1, float *q2, float *q3)
{
    switch (station)
    {
    case 1:
        *w = ds1->w;
        *q1 = ds1->q1;
        *q2 = ds1->q2;
        *q3 = ds1->q3;
        break;

    case 2:
        *w = ds2->w;
        *q1 = ds2->q1;
        *q2 = ds2->q2;
        *q3 = ds2->q3;
        //printf("... CLIENT got %f %f %f\n", ds2->x, ds2->y, ds2->z);
        break;

    case 3:
        *w = ds3->w;
        *q1 = ds3->q1;
        *q2 = ds3->q2;
        *q3 = ds3->q3;
        break;

    case 4:
        *w = ds4->w;
        *q1 = ds4->q1;
        *q2 = ds4->q2;
        *q3 = ds4->q3;
        break;

    default:
        printf("... fastrak::getAbsPositions Invalid Station Number\n");
        break;
    }
}

void
    /*station*/
    fastrak::getXDirCosines(int, float * /*xdircos[3]*/)
{
    cout << "get xdircos not implemented \n";
    //(station, xdircos);
}

void
    /*station*/
    fastrak::getYDirCosines(int, float * /*ydircos[3]*/)
{
    cout << "get ydircos not implemented \n";
    //(station, ydircos);
}

void
    /*station*/
    fastrak::getZDirCosines(int, float * /*zdircos[3]*/)
{
    cout << "get zdircos not implemented \n";
    //(station, zdircos);
}

void
fastrak::getStylusSwitchStatus(int station, unsigned int *status)
{
    switch (station)
    {
    case 1:
        *status = ds1->button;
        break;

    case 2:
        *status = ds2->button;
        //printf("... CLIENT got %f %f %f\n", ds2->x, ds2->y, ds2->z);
        break;

    case 3:
        *status = ds3->button;
        break;

    case 4:
        *status = ds4->button;
        break;

    default:
        printf("... Invalid Station Number\n");
        break;
    }
}

/*************************** private *******************************/

void
fastrak::allocSharedMemoryData()
{
#ifdef WIN32

    stationOutputData *stationsMem = new stationOutputData[4];
    ds1 = &stationsMem[0];

    ds2 = &stationsMem[1];

    ds3 = &stationsMem[2];

    ds4 = &stationsMem[3];

    resetFlag = new int;
#else
    // get shared memory segment for tracker output data ds1, ds2. ds3, ds4

    int shmid;
    key_t shmkey = SHMKEY;
    char *shm_start_addr;

    while ((shmid = shmget(shmkey, 4 * sizeof(stationOutputData) + 1 * sizeof(int) + 1, PERMS | IPC_CREAT)) < 0)
    {
        //cout << "shmget failed" << endl;
        shmkey++;
    }
#ifdef DEBUG
    cout << "INFO: shmid: " << shmid << " shmkey: " << shmkey << endl;
#endif

    // attach ds1, ds2, d3, ds4 to shared memory segment

    shm_start_addr = (char *)shmat(shmid, (char *)0, 0);
#ifdef DEBUG
    printf("INFO: shm_start_addr: %x\n", shm_start_addr);
#endif

    ds1 = (stationOutputData *)shm_start_addr;

    ds2 = (stationOutputData *)(shm_start_addr + sizeof(stationOutputData));

    ds3 = (stationOutputData *)(shm_start_addr + 2 * sizeof(stationOutputData));

    ds4 = (stationOutputData *)(shm_start_addr + 3 * sizeof(stationOutputData));

    resetFlag = (int *)(shm_start_addr + 4 * sizeof(stationOutputData));

// init ds2-ds4
#endif
    memset(ds1, 0, 4 * sizeof(stationOutputData));
    *resetFlag = 0;
}

//
// !!! 	if you compile polhemusdrvr.cpp on IRIX 6.2	!!!
// !!! 	use compiler flag -D_OLD_TERMIOS 		!!!
//
int
fastrak::openSerialPort()
{

    fprintf(stderr, "fastrak::openSerialPort %s\n", serialPortName);
#ifdef WIN32
    DCB dcb;
    BOOL fSuccess;
    //device ==  "COM2";

    desc = CreateFile(serialPortName,
                      GENERIC_READ | GENERIC_WRITE,
                      0, // must be opened with exclusive-access
                      NULL, // no security attributes
                      OPEN_EXISTING, // must use OPEN_EXISTING
                      0, // not overlapped I/O
                      NULL // hTemplate must be NULL for comm devices
                      );

    if (desc == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", serialPortName, GetLastError());
        return (false);
    }

    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    fSuccess = GetCommState(desc, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

    /* First, get the parameters which can be set in the Preferences */
    switch (baudrate)
    {
    case 1200:
        dcb.BaudRate = CBR_1200;
        break;
    case 2400:
        dcb.BaudRate = CBR_2400;
        break;
    case 4800:
        dcb.BaudRate = CBR_4800;
        break;

    case 9600:
        dcb.BaudRate = CBR_9600;
        break;

    case 19200:
        dcb.BaudRate = CBR_19200;
        break;

    case 38400:
        dcb.BaudRate = CBR_38400;
        break;
    case 57600:
        dcb.BaudRate = CBR_57600;
        break;

    default:
        dcb.BaudRate = CBR_19200;
        break;
    }
    // Fill in rest of DCB:  8 data bits, no parity, and 1 stop bit.

    dcb.ByteSize = 8; // data size, xmit, and rcv
    dcb.Parity = NOPARITY; // no parity bit
    dcb.StopBits = ONESTOPBIT; // one stop bit

    fSuccess = SetCommState(desc, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

    return (true);
#else
#ifdef __linux__

    desc = open(serialPortName, O_RDWR | O_NOCTTY | O_NDELAY);
    fprintf(stderr, "fastrak::desc=%d\n", desc);

    if (desc >= 0)
    {

        switch (baudrate)
        {
        case (9600):
            this->baudrate = B9600;
            break;
        case (19200):
            this->baudrate = B19200;
            break;
        case (38400):
            this->baudrate = B38400;
            break;
        default:
            printf("Error: Unsupported baudrate: %d\n", baudrate);
        }

        struct termios options;

        // Get the current options for the port...
        tcgetattr(desc, &options);

        // Set the baud rates
        cfsetispeed(&options, this->baudrate);
        cfsetospeed(&options, this->baudrate);

        // Enable the receiver and set local mode...
        options.c_cflag |= (CLOCAL | CREAD);

        // Set the new options for the port...
        tcsetattr(desc, TCSANOW, &options);
    }

#elif defined _OLD_TERMIOS

    static struct termios termconf;
    desc = open(serialPortName, O_RDWR);
    fprintf(stderr, "fastrak::desc=%d\n", desc);

    if (desc >= 0)
    {

        // compile for IRIX62
        switch (baudrate)
        {

        case (9600):
            this->baudrate = B9600;
            break;

        case (19200):
            this->baudrate = B19200;
            break;

        case (38400):
            this->baudrate = B38400;
            break;
        default:
            printf("Error: Unsupported baudrate: %d\n", baudrate);
        }
        termconf.c_iflag = 0;
        termconf.c_oflag = 0;
        termconf.c_cflag = baudrate | CS8 | CREAD;
        termconf.c_lflag = 0;
#ifndef __APPLE__
        termconf.c_line = N_TTY;
#endif
        termconf.c_cc[VTIME] = 50;
        termconf.c_cc[VMIN] = bufLen * numActiveStations;

        if (ioctl(desc, TIOCSETAW, &termconf) == -1)
        {
            cout << "ERROR: Tracker-Term Setup failed" << endl;
            desc = -1;
        }
    }

#else

    static struct termios termconf;
    desc = open(serialPortName, O_RDWR);
    fprintf(stderr, "fastrak::desc=%d\n", desc);

    if (desc >= 0)
    {
        // then we compile for IRIX64
        termconf.c_iflag = 0;
        termconf.c_oflag = 0;
        termconf.c_ospeed = baudrate;
        termconf.c_ispeed = baudrate;
        termconf.c_cflag = CS8 | CREAD;
        termconf.c_lflag = 0;
#ifndef __APPLE__
        termconf.c_line = N_TTY;
#endif
        termconf.c_cc[VTIME] = 10;
        termconf.c_cc[VMIN] = bufLen * numActiveStations;

        if (ioctl(desc, TIOCSETAW, &termconf) == -1)
        {
            cout << "ERROR: Tracker-Term Setup failed" << endl;
            desc = -1;
        }
    }
#endif
    else
        cout << "ERROR: Opening serial port " << serialPortName << " failed" << endl;
    return (desc);
#endif
}

void
fastrak::setOutput(int station)
{

#ifdef DEBUG
    cout << "CONFIG: setOutput: ";
#endif
    char s[1000];
    // 1: carriage return, line feed]
    // 2: abs pos
    // 3: rel pos
    // 4: euler angles
    // 5: x direction cosines
    // 6: y direction cosines
    // 7: z direction cosines
    // 11: quaternions
    // 16: button
    // 22: intersense wand

    if (buttonDevice == BUTTONDEVICE_STYLUS)
        sprintf(s, "O%d,2,11,16,1", station);
    else
        sprintf(s, "O%d,2,11,22,1", station);

    sendFastrakCmd(s);
}

void
fastrak::reset()
{
    // write reset flag into shared memory to inform tracker process
    *resetFlag = 1;
}

void
fastrak::reinitialize()
{

#ifdef DEBUG
    cout << "STANDARD: reinitialize: ";
#endif

    char s[10];

    // first try to get rid of continous mode
    disableContinousOutput();

#ifndef STANDALONE
    if (covise::coCoviseConfig::isOn("COVER.Input.PolhemusConfig.ResetFactory", false))
    {
        // reset to factory defaults
        fprintf(stderr, "\nFASTRAK RESET:  resetting to factory defaults\n");
        sprintf(s, "W");
        sendFastrakCmd(s);
    }
    else
#endif
    {
        sprintf(s, "%c", 25);
        sendFastrakCmd(s);
        fprintf(stderr, "\nFASTRAK RESET:  reinitialize ");
        for (int i = 0; i < 20; i++)
        {
            sleep(1);
            fprintf(stderr, ".");
        }
        fprintf(stderr, "\n");
    }

    // configure again after reset
    disableContinousOutput();
    setUnitCentimeters();
    setAsciiFormat();
    setStylusMouseMode();
    if (s1 == UP)
    {
        resetReferenceFrame(1);
        setOutput(1);
        setHemisphere(1, hx, hy, hz);
    }
    if (s2 == UP)
    {
        resetReferenceFrame(2);
        setOutput(2);
        setHemisphere(2, hx, hy, hz);
    }
    if (s3 == UP)
    {
        resetReferenceFrame(3);
        setOutput(3);
        setHemisphere(3, hx, hy, hz);
    }
    if (s4 == UP)
    {
        resetReferenceFrame(3);
        setOutput(4);
        setHemisphere(4, hx, hy, hz);
    }
}

void
fastrak::resetReferenceFrame(int station)
{
#ifdef DEBUG
    cout << "STANDARD: resetReferenceFrame: ";
#endif
    char s[10];

    sprintf(s, "R%d", station);
    sendFastrakCmd(s);
}

void
fastrak::setReferenceFrame(int station, float Ox, float Oy, float Oz,
                           float Xx, float Xy, float Xz, float Yx, float Yy, float Yz)
{
#ifdef DEBUG
    cout << "CONFIG: setReferenceFrame: ";
#endif
    char s[1000];

    sprintf(s, "A%d,%1f,%1f,%1f,%1f,%1f,%1f,%1f,%1f,%1f", station,
            Ox, Oy, Oz, Xx, Xy, Xz, Yx, Yy, Yz);

    sendFastrakCmd(s);
}

void
fastrak::setStationActive(int station)
{
#ifdef DEBUG
    cout << "CONFIG: setStationActive: ";
#endif
    char s[20];

    sprintf(s, "l%d,1", station);
    sendFastrakCmd(s);
}

void
fastrak::setStationPassive(int station)
{
    char s[20];
#ifdef DEBUG
    cout << "CONFIG: setStationPassive: ";
#endif
    sprintf(s, "l%d,0", station);
    sendFastrakCmd(s);
}

void
fastrak::setAsciiFormat()
{
#ifdef DEBUG
    cout << "CONFIG: setAsciiFormat: ";
#endif
    sendFastrakCmd((char *)"F");
}

void
fastrak::disableContinousOutput()
{
#ifdef DEBUG
    cout << "CONFIG: disableContinuousOutput: ";
#endif
    sendFastrakCmd((char *)"\n");
    sendFastrakCmd((char *)"c");
}

void
fastrak::setUnitCentimeters()
{
#ifdef DEBUG
    cout << "CONFIG: setUnitCentimeters: ";
#endif
    sendFastrakCmd((char *)"u");
}

void
fastrak::setUnitInches()
{
#ifdef DEBUG
    cout << "CONFIG: setUnitInches: ";
#endif
    sendFastrakCmd((char *)"U");
}

void
fastrak::setBoresight(int) /*station*/
{
    // I don't get the boresight command working
    cout << "Polyhemus: not implemented\n";
    //(station);
}

void
fastrak::unsetBoresight(int) /*station*/
{
    // I don't get the boresight command working
    cout << "Polyhemus: not implemented\n";
    //(station);
}

void
fastrak::setStylusMouseMode()
{
    sendFastrakCmd((char *)"e1,0");
}

void
fastrak::sendFastrakCmd(char *cmd_buf)
{

    //cerr << "Sending command: " << cmd_buf << endl;
    static char crlf[] = "\r\n";

/* code to add a CR-LF pair to the end of a command */
/* if it's needed */
#ifdef WIN32
    DWORD nBytesWritten = 0;
    BOOL bResult;
    bResult = WriteFile(desc, cmd_buf, strlen(cmd_buf), &nBytesWritten, NULL);

    if (strlen(cmd_buf) > 1)
    {
        nBytesWritten = 0;
        bResult = WriteFile(desc, crlf, 2, &nBytesWritten, NULL);
    }
#else
    if (write(desc, cmd_buf, strlen(cmd_buf)) != strlen(cmd_buf))
    {
        cerr << "fasttrak::sendFastrakCmd: short write" << endl;
    }

    if (strlen(cmd_buf) > 1)
    {
        if (write(desc, crlf, 2) != 2)
        {
            cerr << "fasttrak::sendFastrakCmd: short write2" << endl;
        }
    }
#endif
}

/************************************************************************/
/*									*/
/*	Description	server reads all active stations		*/
/*									*/
/************************************************************************/

int
fastrak::readActiveStations()
{
    int ret;
    char buf[300];
    char oc, er;
    int st;
    int i;
    int ok = 0;
    static int misalignment_count = 0;
    int bytes_offset = 0;
    static int timeoutCount = 0;
    //fprintf(stderr,"------ in fastrak::readActiveStations\n");

    // write "P" to get data for all active stations
    sendFastrakCmd((char *)"P");

    // read for all active stations
    for (i = 0; i < numActiveStations; i++)

    {

        // Clean the buffer
        for (int si = 0; si < 256; si++)
            buf[si] = 0;

        // read a full record of station no i
        int numRead = 0;
        while (numRead < bufLen)
        {
#ifdef WIN32
            DWORD numReadChunk = 0;
            BOOL result = ReadFile(desc, buf + numRead, bufLen - numRead, &numReadChunk, NULL);
            if (!result)
            {
                cerr << "Polhemus: read error" << endl;
            }
            numRead += numReadChunk;
#else
            // prevent busy loops
            struct timeval tv = { 0, 1000 };
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(desc, &fds);
            select(desc + 1, &fds, NULL, NULL, &tv);

            int numReadChunk = read(desc, buf + numRead, bufLen - numRead);
            if (numReadChunk < 0)
            {
                if (errno != EAGAIN)
                    perror("polhemus read");
            }
            else if (numReadChunk == 0)
            {
                cerr << "Polhemus:null read " << endl;
            }
            else
            {
                numRead += numReadChunk;
            }
#endif
        }
        ret = bufLen;

        // now we are aligned but have more than bufLen byte
        if (ret > bufLen)
        {
            // Have been looping looking for a return, and buf[0] is not the start
            memmove(&buf[0], &buf[ret - bufLen], bufLen);
            //printf ("Had to fix buffer, is now  = %s\n", buf);
        }

        if (buf[bufLen - 1] != '\n')
        {
            fprintf(stderr, "WARNING: we are misaligned\n");
            misalignment_count++;
            for (bytes_offset = 0; bytes_offset < ret && buf[bytes_offset] != '\n'; bytes_offset++)
                ;
            if (bytes_offset == ret)
            {
                fprintf(stderr, "WARNING: reading another %d bytes\n", 6 - bytes_offset);

#ifdef WIN32
                DWORD diff_ret = 0;
                BOOL result = ReadFile(desc, &buf[ret], bufLen - bytes_offset, &diff_ret, NULL);
                if (!result)
                {
                    cerr << "Polhemus: read error" << endl;
                }
#else
                int diff_ret = read(desc, &buf[ret], bufLen - bytes_offset);
#endif
                fprintf(stderr, "WARNING: read %d bytes\n", diff_ret);
                for (int k = 0; k < bufLen; k++)
                    buf[k] = buf[bytes_offset + k];
                buf[bufLen - 1] = '\n';
                fprintf(stderr, "WARNING: now buffer is %s\n", buf);
            }
        }

#ifdef DBGPRINT
//printf(" nach read\n");
#endif

        // check data string

        if ((buf[0] == '0') && (buf[bufLen - 1] == '\n'))
        {
            // data looks ok but we had many timeouts
            if (timeoutCount > numActiveStations * bufLen * 10)
            {
                timeoutCount = 0;
                fprintf(stderr, "\n");
                fprintf(stderr, "FASTRAK WARNING: needed several reads to get a complete dataset\n");
                fprintf(stderr, "                Problem may be caused by a different configuration\n");
                fprintf(stderr, "                Reset FASTRAK at runtime by pressing 'r' on the keyboard\n");
                fprintf(stderr, "                if this doesn't help quit COVER ...\n");
                fprintf(stderr, "                ... and switch FASTRAK off and on again\n");
            }
#ifdef DBGPRINT1
            printf("... SERVER: valid data\n");
#endif

            // data is from station 1
            if (buf[1] == '1')
            {
                ok++;
                int ret = sscanf(buf, "%c%i%c%f%f%f%f%f%f%f%d",
                                 &oc, &st, &er, &(ds1->x), &(ds1->y), &(ds1->z),
                                 &(ds1->w), &(ds1->q1), &(ds1->q2), &(ds1->q3), &(ds1->button));
                if (ret != 11)
                {
                    cerr << "fastrak::readActiveStations: sscanf1 failed" << endl;
                }
                //printf("... SERVER station 2: %f %f %f\n", ds1->x, ds1->y, ds1->z);

                //if (ds1->button)
                //printf("\a\n");fflush(stdout);
            }

            else if (buf[1] == '2')
            {

                ok++;
                int ret = sscanf(buf, "%c%i%c%f%f%f%f%f%f%f%d",
                                 &oc, &st, &er, &(ds2->x), &(ds2->y), &(ds2->z),
                                 &(ds2->w), &(ds2->q1), &(ds2->q2), &(ds2->q3), &(ds2->button));
                if (ret != 11)
                {
                    cerr << "fastrak::readActiveStations: sscanf2 failed" << endl;
                }
#ifdef DBGPRINT1
                printf("... SERVER station 2: %f %f %f %f %f %f %f %d\n",
                       ds2->x, ds2->y, ds2->z,
                       ds2->w, ds2->q1, ds2->q2, ds2->q3, ds2->button);
#endif
            }

            else if (buf[1] == '3')
            {
                ok++;
                int ret = sscanf(buf, "%c%i%c%f%f%f%f%f%f%f%d",
                                 &oc, &st, &er, &(ds3->x), &(ds3->y), &(ds3->z),
                                 &(ds3->w), &(ds3->q1), &(ds3->q2), &(ds3->q3), &(ds3->button));
                if (ret != 11)
                {
                    cerr << "fastrak::readActiveStations: sscanf3 failed" << endl;
                }
                //printf("... SERVER station 2: %f %f %f\n", ds3->x, ds3->y, ds3->z);
            }

            else if (buf[1] == '4')
            {

                ok++;
                int ret = sscanf(buf, "%c%i%c%f%f%f%f%f%f%f%d",
                                 &oc, &st, &er, &(ds4->x), &(ds4->y), &(ds4->z),
                                 &(ds4->w), &(ds4->q1), &(ds4->q2), &(ds4->q3), &(ds4->button));
                if (ret != 11)
                {
                    cerr << "fastrak::readActiveStations: sscanf4 failed" << endl;
                }
#ifdef DBGPRINT1
                printf("... SERVER station 4: %f %f %f %f %f %f %f %d\n",
                       ds4->x, ds4->y, ds4->z,
                       ds4->w, ds4->q1, ds4->q2, ds4->q3, ds4->button);
#endif
            }

            // not a valid station
            else
                ok = 0;
        }
        else
        {
            // data string not ok

            ok = 0;
            timeoutCount++;
            if (timeoutCount > numActiveStations * bufLen * 10)
            {
                timeoutCount = 0;
                fprintf(stderr, "\n");
                fprintf(stderr, "FASTRAK WARNING: couldn't get a complete dataset\n");
                fprintf(stderr, "                Problem may be caused by a different configuration\n");
                fprintf(stderr, "                Reset FASTRAK at runtime by pressing 'r' on the keyboard\n");
                fprintf(stderr, "                if this doesn't help quit COVER ...\n");
                fprintf(stderr, "                ... and switch FASTRAK off and on again\n");
            }
        }
    }

    if (*resetFlag == 1)
    {
        reinitialize();
        *resetFlag = 0;
    }

    return (ok);
}

void
fastrak::serverDummyRead()
{
    static float i = 0.0;
    ds1->x = i;
    ds2->x = i;
    //cout << "SERVER data: " << ds1->x << endl;
    //printf("server reads : ds1->x = %f\n", ds1->x);
    //printf("server reads : ds2->x = %f\n", ds2->x);

    i += 0.0001f;
}
