/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#include <WS2tcpip.h>
#endif
#include <util/common.h>
#include <util/unixcompat.h>

#include <config/CoviseConfig.h>
#include <config/coConfigConstants.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <net/covise_host.h>
#include <net/udpMessage.h>
#include "vvSlave.h"
#include "vvPluginSupport.h"
//#include "vvCommunication.h"
//#include "vvNavigationManager.h"
//#include "vvFileManager.h"
#include "vvViewer.h"
#include "vvVIVE.h"
//#include "vvHud.h"
//#include "coClusterStat.h"
//#include "vvConfig.h"
#include <vrb/client/VRBClient.h>

#ifdef HAS_MPI
#include <mpi.h>
#define MPI_BCAST
#define MPI_BARRIER
#ifndef CO_MPI_SEND
#define CO_MPI_SEND MPI_Ssend
#endif
#endif

using namespace covise;
using namespace vive;

#define RINGBUFLEN 200
#ifdef __linux__
#include <linux/ppdev.h>
#endif

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define NOMCAST
#else
#include <sys/ioctl.h>
#endif

#undef DOTIMING
#ifdef DOTIMING
#include <util/coTimer.h>
#else
#define MARK0(a)
#define MARK1(a, b)
#endif

#if !defined(NOMCAST) && defined(HAVE_NORM)
#include "rel_mcast.h"
#endif

#include <errno.h>
#include "vvMSController.h"

#ifdef DEBUG_MESSAGES
int debugMessageCounter;
bool debugMessagesCheck;
#endif

vvMSController *vvMSController::s_singleton = NULL;

vvMSController::SlaveData::SlaveData(int n)
    : data(vvMSController::instance()->numSlaves)
    , n(n)
{
    for (int ctr = 0; ctr < vvMSController::instance()->numSlaves; ++ctr)
    {
        this->data[ctr] = malloc(n);
        memset(this->data[ctr], 0, 1);
    }
}

vvMSController::SlaveData::~SlaveData()
{
    for (size_t ctr = 0; ctr < this->data.size(); ++ctr)
    {
        free(this->data[ctr]);
    }
}

vvMSController *vvMSController::instance()
{
    if(s_singleton == NULL)
    {
        s_singleton = new vvMSController();
    }
    return s_singleton;
}

void vvMSController::destroy()
{
    delete s_singleton;
    s_singleton = nullptr;
}


vvMSController::vvMSController(int AmyID, const char *addr, int port)
    : m_debugLevel(2)
    , master(true)
    , slave(false)
    , myID(0)
    , socket(0)
    , socketDraw(0)
#ifdef HAS_MPI
    , appComm(MPI_COMM_WORLD)
    , drawComm(MPI_COMM_WORLD)
#endif
    , heartBeatCounter(0)
    , heartBeatCounterDraw(0)

{
    assert(!s_singleton);
    s_singleton = this;
#ifdef _WIN32
    int err;
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(2, 2);
    err = WSAStartup(wVersionRequested, &wsaData);
#endif

#ifdef HAS_MPI
    int mpiInit = 0;
    MPI_Initialized(&mpiInit);
    if (mpiInit)
    {
        drawRank.resize(numSlaves + 1, -1);
        MPI_Comm_split(appComm, vvViewer::mustDraw() ? 1 : MPI_UNDEFINED, AmyID, &drawComm);
        int dr = -1;
        if (vvViewer::mustDraw())
            MPI_Comm_rank(drawComm, &dr);
        MPI_Gather(&dr, 1, MPI_INT, drawRank.data(), 1, MPI_INT, 0, appComm);
    }
#endif

    MARK0("vvMSController::vvMSController");
    if (AmyID >= 0)
    {
        myID = AmyID;
    }

    m_debugLevel = covise::coCoviseConfig::getInt("VIVE.DebugLevel", m_debugLevel);

    if (debugLevel(2))
        fprintf(stderr, "\nnew vvMSController\n");
#ifdef DEBUG_MESSAGES
    debugMessageCounter = 0;
    debugMessagesCheck = true;
#endif
    syncMode = SYNC_TCP;

    m_drawStatistics = coCoviseConfig::isOn("VIVE.MultiPC.Statistics", false);
    //   vvPluginSupport::instance()->setBuiltInFunctionState("CLUSTER_STATISTICS",m_drawStatistics);

    // Multicast settings
    multicastDebugLevel = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.debugLevel", 0);
    multicastAddress = coCoviseConfig::getEntry("VIVE.MultiPC.Multicast.mcastAddr");
    multicastPort = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.mcastPort", 23232);
    multicastInterface = coCoviseConfig::getEntry("VIVE.MultiPC.Multicast.mcastIface");
    multicastMTU = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.mtu", 1500);
    multicastTTL = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.ttl", 1);
    multicastLoop = coCoviseConfig::isOn("VIVE.MultiPC.Multicast.lback", false);
    multicastBufferSpace = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.bufferSpace", 1000000);
    multicastBlockSize = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.blockSize", 4);
    multicastNumParity = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.numParity", 0);
    multicastTxCacheSize = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txCacheSize", 100000000);
    multicastTxCacheMin = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txCacheMin", 1);
    multicastTxCacheMax = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txCacheMax", 128);
    multicastTxRate = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txRate", 1000);
    multicastBackoffFactor = (double)coCoviseConfig::getFloat("VIVE.MultiPC.Multicast.backoffFactor", 0.0);
    multicastSockBuffer = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.sockBufferSize", 512000);
    multicastClientTimeout = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.readTimeoutSec", 30);
    multicastServerTimeout = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.writeTimeoutMsec", 500);
    multicastRetryTimeout = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.retryTimeout", 100);
    multicastMaxLength = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.maxLength", 1000000);

    string sm = coCoviseConfig::getEntry("VIVE.MultiPC.SyncMode");
    numSlaves = coCoviseConfig::getInt("VIVE.MultiPC.NumSlaves", 0);

    if (numSlaves==0 && myID>0)
    {
        fprintf(stderr, "vvMSController: id=%d>0 but numSlaves=0\n", myID);
        exit(1);
    }
    assert(myID==0 || numSlaves>0);

    if (strcasecmp(sm.c_str(), "TCP") == 0)
    {
        MARK0("\tsyncMode: TCP");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: TCP\n");

        syncMode = SYNC_TCP;
    }
    else if (strcasecmp(sm.c_str(), "SERIAL") == 0)
    {
        MARK0("\tsyncMode: SERIAL");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: SERIAL\n");

        syncMode = SYNC_SERIAL;
    }
    else if (strcasecmp(sm.c_str(), "MAGIC") == 0)
    {
        MARK0("\tsyncMode: MAGIC");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: MAGIC\n");

        syncMode = SYNC_MAGIC;
    }
    else if (strcasecmp(sm.c_str(), "TCP_SERIAL") == 0)
    {
        MARK0("\tsyncMode: TCP_SERIAL");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: TCP_SERIAL\n");

        syncMode = SYNC_TCP_SERIAL;
    }
    else if (strcasecmp(sm.c_str(), "PARALLEL") == 0)
    {
        MARK0("\tsyncMode: PARALLEL");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: PARALLEL\n");

        syncMode = SYNC_PARA;
    }
    else if (strcasecmp(sm.c_str(), "UDP") == 0)
    {
        MARK0("\tsyncMode: UDP");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: UDP\n");

        syncMode = SYNC_UDP;
    }
    else if (strcasecmp(sm.c_str(), "MULTICAST") == 0 && numSlaves > 0)
    {
        MARK0("\tsyncMode: MULTICAST");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: MULTICAST\n");

#if defined(NOMCAST) || !defined(HAVE_NORM)
        fprintf(stderr, "This vvVIVE does not have MULTICAST support\n");
#else
        syncMode = SYNC_MULTICAST;
#endif
    }
    else if (strcasecmp(sm.c_str(), "MPI") == 0)
    {
        MARK0("\tsyncMode: MPI");
        if (debugLevel(3))
            fprintf(stderr, "syncMode: MPI\n");

#if !defined(HAS_MPI)
        fprintf(stderr, "This vvVIVE does not have MPI support\n");
#else
        syncMode = SYNC_MPI;
        MPI_Comm_size(appComm, &numSlaves);
        --numSlaves;
#endif
    }
    else
    {
        if (debugLevel(3))
            fprintf(stderr, "syncMode: TCP\n");
        MARK0("\tsyncMode TCP");
    }

    barrierProcess = SYNC_DRAW;
    sm = coCoviseConfig::getEntry("VIVE.MultiPC.SyncProcess");
    if (strcasecmp(sm.c_str(), "APP") == 0)
    {
        if (debugLevel(3))
            fprintf(stderr, "barrierProcess: APP\n");
        barrierProcess = SYNC_APP;

        MARK0("\tsyncProcess: APP");
    }
    else
    {
        MARK0("\tsyncProcess: DRAW");
        if (debugLevel(3))
            fprintf(stderr, "barrierProcess: DRAW\n");
    }

    if ((syncMode == SYNC_SERIAL) || (syncMode == SYNC_TCP_SERIAL))
    {
#ifndef _WIN32
        std::string deviceFile = coCoviseConfig::getEntry("value", "VIVE.MultiPC.SerialDevice", "/dev/ttyd1");
        serial = open(deviceFile.c_str(), O_RDWR);
        if (serial == -1)
        {
            perror("ERROR: Could not open Serial port");
            cerr << "ERROR: deviceFile: " << deviceFile << endl;
            syncMode = SYNC_TCP;
            MARK1("\topening serial port %s: failed", deviceFile.c_str());
        }
        MARK1("\topening serial port %s: successful", deviceFile.c_str());

        fcntl(serial, F_SETFL, O_APPEND | O_NONBLOCK | FNDELAY);
        int statusByte;
        ioctl(serial, TIOCMGET, &statusByte);
        statusByte &= ~(TIOCM_RTS);
        if (ioctl(serial, TIOCMSET, &statusByte) == -1)
            cerr << "RTS=0 ERROR" << endl;
#endif
    }
    else if (syncMode == SYNC_MAGIC)
    {
#ifndef _WIN32
        // open 'wired' device and set to false
        std::string deviceFile = coCoviseConfig::getEntry("VIVE.MultiPC.SerialDevice");
        magicFd = open(deviceFile.c_str(), O_RDWR);
        MARK1("\tMAGIC: opening port %s\n", deviceFile.c_str());

        // set my state to 'busy'
        char magicBuf = 0;
        if (write(magicFd, &magicBuf, 1) != 1)
        {
            cerr << "vvMSController::vvMSController: short write: " << strerror(errno) << endl;
        }
        MARK0("\tMAGIC: send BUSY");
#endif
    }
    else if (syncMode == SYNC_PARA)
    {
#ifdef __linux__
        std::string deviceFile = coCoviseConfig::getEntry("value", "VIVE.MultiPC.ParallelDevice", "/dev/parport0");
        parallel = open(deviceFile.c_str(), O_RDWR);
        if (parallel == -1)
        {
            perror("ERROR: Could not open Parallel port");
            cerr << "ERROR: deviceFile: " << deviceFile << endl;
            syncMode = SYNC_TCP;
            MARK1("\tsyncMode: PARALLEL: port %s open failed", deviceFile.c_str());
        }
        MARK1("\tsyncMode: PARALLEL: port %s opened successful", deviceFile.c_str());

        ioctl(parallel, PPCLAIM);
        unsigned char statusByte = 0xff;
        statusByte = 0x0;
        ioctl(parallel, PPWDATA, &statusByte);
        allChildren = 0;
        for (int i = 0; i < numSlaves; i++)
            allChildren |= 1 << (i + 3);
#endif
    }

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        MPI_Comm_rank(appComm, &myID);
        master = myID == 0;
        slave = !master;
    }
    else
#endif
    {
        /// This is a slave
        if (myID > 0)
        {
            master = false;
            slave = true;

            MARK1("VIVE starting as slave %d", myID);
            if (debugLevel(3))
                fprintf(stderr, "VIVE starting as slave id=%d\n", myID);

#if !defined(NOMCAST) && defined(HAVE_NORM)
            if (syncMode == SYNC_MULTICAST)
            {
                if (!multicastAddress.empty())
                {
                    // Call client constructor with address/port
                    multicast = new Rel_Mcast(myID, multicastAddress.c_str(), multicastPort);
                }
                else
                {
                    // Call client constructor without address/port
                    multicast = new Rel_Mcast(myID);
                }

                if (!multicastInterface.empty())
                {
                    // Set an alternate interface (e.g. eth1)
                    multicast->setInterface(multicastInterface.c_str());
                }

                // Various settings from coconfig
                multicast->setDebugLevel(multicastDebugLevel);
                multicast->setBufferSpace(multicastBufferSpace);
                multicast->setSockBufferSize(multicastSockBuffer);
                multicast->setTimeout(multicastClientTimeout);
                multicast->setMaxLength(multicastMaxLength);

                if (multicast->init() != Rel_Mcast::RM_OK)
                {
                    delete multicast;
                    exit(0);
                }
            }
#endif

            connectToMaster(addr, port);
        }
        else
        {
            MARK0("VIVE starting as master");
#if !defined(NOMCAST) && defined(HAVE_NORM)
            if (syncMode == SYNC_MULTICAST)
            {
                if (!multicastAddress.empty())
                {
                    multicast = new Rel_Mcast(true, numSlaves, multicastAddress.c_str(), multicastPort);
                }
                else
                {
                    multicast = new Rel_Mcast(true, numSlaves);
                }
                if (!multicastInterface.empty())
                {
                    // Set an alternate interface (e.g. eth1)
                    multicast->setInterface(multicastInterface.c_str());
                }

                // Various settings from coconfig
                multicast->setDebugLevel(multicastDebugLevel);
                multicast->setMTU(multicastMTU);
                multicast->setTTL(multicastTTL);
                multicast->setLoopback(multicastLoop);
                multicast->setBufferSpace(multicastBufferSpace);
                multicast->setBlocksAndParity(multicastBlockSize, multicastNumParity);
                multicast->setTxCacheBounds(multicastTxCacheSize, multicastTxCacheMin, multicastTxCacheMax);
                multicast->setTxRate(multicastTxRate);
                multicast->setBackoffFactor(multicastBackoffFactor);
                multicast->setSockBufferSize(multicastSockBuffer);
                multicast->setTimeout(multicastServerTimeout);
                multicast->setRetryTimeout(multicastRetryTimeout);
                multicast->setMaxLength(multicastMaxLength);

                if (multicast->init() != Rel_Mcast::RM_OK)
                {
                    delete multicast;
                    exit(0);
                }
            }
#endif
            if (debugLevel(3))
                fprintf(stderr, "VIVE starting as master\n");
        }
        stats[0] = NULL;
    }
	
    if (covise::coConfigConstants::getRank() != myID) {
        std::cerr << "vvMSController: coConfigConstants::getRank()=" << covise::coConfigConstants::getRank() << ", myID=" << myID << std::endl;
    }
    if(AmyID != -1)
    {
    assert(covise::coConfigConstants::getRank() == myID);
    }
}

#ifdef HAS_MPI
vvMSController::vvMSController(const MPI_Comm *comm, pthread_barrier_t *shmBarrier)
    : m_debugLevel(2)
    , master(true)
    , slave(false)
    , myID(0)
    , socket(0)
    , socketDraw(0)
    , appComm(*comm)
    , drawComm(*comm)
    , pthreadShmBarrier(shmBarrier)
    , heartBeatCounter(0)
    , heartBeatCounterDraw(0)

{
    assert(!s_singleton);
    s_singleton = this;

    MARK0("vvMSController::vvMSController");
    m_debugLevel = covise::coCoviseConfig::getInt("VIVE.DebugLevel", m_debugLevel);
    if (debugLevel(2))
        fprintf(stderr, "\nnew vvMSController\n");

    int mpiInit = 0;
    MPI_Initialized(&mpiInit);
    assert(mpiInit);
    MPI_Comm_rank(appComm, &myID);
    MPI_Comm_size(appComm, &numSlaves);
    --numSlaves;

    drawRank.resize(numSlaves+1, -1);
     MPI_Comm_split(appComm, vvViewer::mustDraw() ? 1 : MPI_UNDEFINED, myID, &drawComm);
    int dr = -1;
    if (vvViewer::mustDraw())
        MPI_Comm_rank(drawComm, &dr);
    MPI_Gather(&dr, 1, MPI_INT, drawRank.data(), 1, MPI_INT, 0, appComm);

#ifdef DEBUG_MESSAGES
    debugMessageCounter = 0;
    debugMessagesCheck = true;
#endif

    m_drawStatistics = coCoviseConfig::isOn("VIVE.MultiPC.Statistics", false);
    //   vvPluginSupport::instance()->setBuiltInFunctionState("CLUSTER_STATISTICS",m_drawStatistics);

    // Multicast settings
    multicastDebugLevel = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.debugLevel", 0);
    multicastAddress = coCoviseConfig::getEntry("VIVE.MultiPC.Multicast.mcastAddr");
    multicastPort = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.mcastPort", 23232);
    multicastInterface = coCoviseConfig::getEntry("VIVE.MultiPC.Multicast.mcastIface");
    multicastMTU = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.mtu", 1500);
    multicastTTL = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.ttl", 1);
    multicastLoop = coCoviseConfig::isOn("VIVE.MultiPC.Multicast.lback", false);
    multicastBufferSpace = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.bufferSpace", 1000000);
    multicastBlockSize = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.blockSize", 4);
    multicastNumParity = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.numParity", 0);
    multicastTxCacheSize = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txCacheSize", 100000000);
    multicastTxCacheMin = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txCacheMin", 1);
    multicastTxCacheMax = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txCacheMax", 128);
    multicastTxRate = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.txRate", 1000);
    multicastBackoffFactor = (double)coCoviseConfig::getFloat("VIVE.MultiPC.Multicast.backoffFactor", 0.0);
    multicastSockBuffer = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.sockBufferSize", 512000);
    multicastClientTimeout = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.readTimeoutSec", 30);
    multicastServerTimeout = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.writeTimeoutMsec", 500);
    multicastRetryTimeout = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.retryTimeout", 100);
    multicastMaxLength = coCoviseConfig::getInt("VIVE.MultiPC.Multicast.maxLength", 1000000);

    barrierProcess = SYNC_DRAW;
    string sm = coCoviseConfig::getEntry("VIVE.MultiPC.SyncProcess");
    if (strcasecmp(sm.c_str(), "APP") == 0)
    {
        if (debugLevel(3))
            fprintf(stderr, "barrierProcess: APP\n");
        barrierProcess = SYNC_APP;

        MARK0("\tsyncProcess: APP");
    }
    else
    {
        MARK0("\tsyncProcess: DRAW");
        if (debugLevel(3))
            fprintf(stderr, "barrierProcess: DRAW\n");
    }

    syncMode = SYNC_MPI;
    MARK0("\tsyncMode: forced MPI");
    if (debugLevel(3))
        fprintf(stderr, "syncMode: MPI\n");

    master = myID == 0;
    slave = !master;

    if (covise::coConfigConstants::getRank() != myID) {
        std::cerr << "vvMSController: coConfigConstants::getRank()=" << covise::coConfigConstants::getRank() << ", myID=" << myID << std::endl;
    }
    assert(covise::coConfigConstants::getRank() == myID);
}
#endif


vvMSController::~vvMSController()
{
    delete socket;
    delete socketDraw;
    if ((syncMode == SYNC_SERIAL) || (syncMode == SYNC_TCP_SERIAL))
    {
        close(serial);
    }
    else if (syncMode == SYNC_PARA)
    {
#ifdef __linux__
        ioctl(parallel, PPRELEASE);
        close(parallel);
#endif
    }
    else if (syncMode == SYNC_MAGIC)
    {
        close(magicFd);
    }
    else if (syncMode == SYNC_MULTICAST)
    {
#if !defined(NOMCAST) && defined(HAVE_NORM)
        delete multicast;
#endif
    }

    s_singleton = nullptr;
}

bool
vvMSController::debugLevel(int l) const
{
    return m_debugLevel >= l;
}

void
vvMSController::killClients()
{
    if (syncMode == SYNC_MULTICAST)
    {
#if !defined(NOMCAST) && defined(HAVE_NORM)
        delete multicast;
#endif
    }
}

void vvMSController::heartBeat(const std::string &name, bool draw)
{
    if (draw)
    {
        ++heartBeatCounterDraw;
    }
    else
    {
        ++heartBeatCounter;
    }
    int localCounter = draw ? heartBeatCounterDraw : heartBeatCounter;

    if (isMaster())
    {
        std::cerr << "vvMSController: heart beat \"" << name << "\", count=" << localCounter << std::endl;
        if (draw)
            sendSlavesDraw(&localCounter, sizeof(localCounter));
        else
            sendSlaves(&localCounter, sizeof(localCounter));
    }
    else
    {
        int masterCount = 0;
        if (draw)
            readMasterDraw(&masterCount, sizeof(masterCount));
        else
            readMaster(&masterCount, sizeof(masterCount));
        if (localCounter != masterCount)
        {
            std::cerr << "vvMSController: missed heart beat \"" << name << "\", master is " << masterCount << ", on " << getID() << " is " << localCounter << std::endl;
            exit(0);
        }
    }
}

bool vvMSController::drawStatistics() const
{
    return m_drawStatistics;
}

void vvMSController::setDrawStatistics(bool enable)
{
    m_drawStatistics = enable;
}

void vvMSController::checkMark(const char *file, int line)
{
    cerr << file << line << endl;
    std::stringstream str;
    str << file << ":" << line;
    heartBeat(str.str());
}

void vvMSController::connectToMaster(const char *addr, int port)
{
    Host h(addr);
    socket = new Socket(&h, port, 200,10);

    int port2;
    readMaster(&port2, sizeof(port2), true);
    socketDraw = new Socket(&h, port2, 200, 0);
    int sendbuf = 64 * 1024;
    int recvbuf = 64 * 1024;
    if (setsockopt(socket->get_id(), SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
    {
        cerr << "could not set socket buff to " << sendbuf << endl;
    }
    if (setsockopt(socket->get_id(), SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
    {
        cerr << "could not set socket buff to " << sendbuf << endl;
    }
    if (setsockopt(socketDraw->get_id(), SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
    {
        cerr << "could not set socket buff to " << sendbuf << endl;
    }
    if (setsockopt(socketDraw->get_id(), SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
    {
        cerr << "could not set socket buff to " << sendbuf << endl;
    }
}

void vvMSController::sendSlaves(const Message *msg)
{
    assert(isMaster());

    if (syncMode == SYNC_MULTICAST)
    {
#if !defined(NOMCAST) && defined(HAVE_NORM)

        // Prepare header
        int headerSize = 4 * sizeof(int);
        char header[headerSize];
        int *header_int;
        header_int = (int *)header;
        header_int[0] = msg->sender;
        header_int[1] = msg->send_type;
        header_int[2] = msg->type;
        header_int[3] = msg->data.length();

        // Write header via multicast
        if (multicast->write_mcast(header, headerSize) != Rel_Mcast::RM_OK)
        {
            delete multicast;
            exit(0);
        }

        // Write data via multicast (split up into pieces if necessary)
        int numMsg = msg->data.length() / multicastMaxLength;
        if (msg->data.length() % multicastMaxLength != 0)
            numMsg++;
        int curMsg;

        // Write first numMsg-1 messages
        for (curMsg = 1; curMsg < numMsg; curMsg++)
        {
            if (multicast->write_mcast(msg->data.data() + multicastMaxLength * (curMsg - 1), multicastMaxLength) != Rel_Mcast::RM_OK)
            {
                delete multicast;
                exit(0);
            }
        }
        // Write numMsg message
        if (multicast->write_mcast(msg->data.data() + multicastMaxLength * (curMsg - 1), msg->data.length() % multicastMaxLength) != Rel_Mcast::RM_OK)
        {
            delete multicast;
            exit(0);
        }
#endif
    }
    else

    {
        for (int i = 0; i < numSlaves; i++)
        {
            slaves[i]->sendMessage(msg);
        }
#ifdef DEBUG_MESSAGES
        debugMessageCounter++;
#endif
    }
}
void vvMSController::sendSlaves(const UdpMessage* msg)
{
	for (int i = 0; i < numSlaves; i++)
	{
		slaves[i]->sendMessage(msg);
	}
}
int vvMSController::readMaster(Message *msg)
{
    assert(isSlave());

#if !defined(NOMCAST) && defined(HAVE_NORM)
    if (syncMode == SYNC_MULTICAST)
    {
        // Prepare header
        int headerSize = 4 * sizeof(int);
        char header[headerSize];
        int *header_int;

        // Receive header
        if (multicast->read_mcast(header, headerSize) != Rel_Mcast::RM_OK)
        {
            delete multicast;
            exit(0);
        }

        // Parse header, prepare data
        header_int = (int *)header;
        msg->sender = header_int[0];
        msg->send_type = (sender_type)header_int[1];
        msg->type = header_int[2];
        msg->data = DataHandle{ header_int[3] };

        // Read data via multicast (Read piece-by-piece if necessary)
        int numMsg = msg->data.length() / multicastMaxLength;
        if (msg->data.length() % multicastMaxLength != 0)
            numMsg++;
        int curMsg;

        // Read first numMsg-1 messages
        for (curMsg = 1; curMsg < numMsg; curMsg++)
        {
            if (multicast->read_mcast(msg->data.accessData() + multicastMaxLength * (curMsg - 1), multicastMaxLength) != Rel_Mcast::RM_OK)
            {
                delete multicast;
                exit(0);
            }
        }
        // Read numMsg message
        if (multicast->read_mcast(msg->data.accessData() + multicastMaxLength * (curMsg - 1), msg->data.length() % multicastMaxLength) != Rel_Mcast::RM_OK)
        {
            delete multicast;
            exit(0);
        }

        // Return size
        return msg->data.length() + headerSize;
    }
    else
#endif
#ifdef HAS_MPI
        if (syncMode == SYNC_MPI)
    {
        const int headerSize = 4 * sizeof(int);

        int buffer[headerSize];

        int received = readMaster(buffer, headerSize);

        if (received < headerSize)
            return received;

        int *bufferInt = (int *)buffer;
        msg->sender = bufferInt[0];
        msg->send_type = bufferInt[1];
        msg->type = bufferInt[2];

        msg->data = DataHandle(bufferInt[3]);

        return received + readMaster(msg->data.accessData(), msg->data.length());
    }
    else
#endif
    {
        char read_buf[4 * sizeof(int)];
        int *read_buf_int;
        int headerSize = 4 * sizeof(int);
        int toRead;
        int bytesRead = 0;
        int ret = readMaster(read_buf, headerSize);
        if (ret < headerSize)
            return ret;
        read_buf_int = (int *)read_buf;
        msg->sender = read_buf_int[0];
        msg->send_type = read_buf_int[1];
        msg->type = read_buf_int[2];
        msg->data = DataHandle(read_buf_int[3]);
#ifdef DEBUG_MESSAGES
        debugMessagesCheck = false;
#endif
        while (bytesRead < msg->data.length())
        {
            toRead = msg->data.length() - bytesRead;
            if (toRead > READ_BUFFER_SIZE)
                toRead = READ_BUFFER_SIZE;
            int ret = readMaster(msg->data.accessData() + bytesRead, toRead);
            if (ret < toRead)
            {
                //cerr << "Short Message" << ret << endl;
                if (ret < 0)
                    return ret;
            }
            bytesRead += ret;
        }
#ifdef DEBUG_MESSAGES
        debugMessagesCheck = true;
#endif
        return bytesRead;
    }
}

void vvMSController::sendMaster(const Message *msg)
{
    assert(isSlave());

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        int header[4];

        header[0] = msg->sender;
        header[1] = msg->send_type;
        header[2] = msg->type;
        header[3] = msg->data.length();

        sendMaster(reinterpret_cast<char *>(&header[0]), 4 * sizeof(int));
        sendMaster(msg->data.data(), msg->data.length());
    }
    else
#endif
    {
        char write_buf[WRITE_BUFFER_SIZE];
        int *write_buf_int;
        int headerSize = 4 * sizeof(int);
        int len = msg->data.length() + headerSize;
        int toWrite;
        int written = 0;
        toWrite = len;
        if (toWrite > WRITE_BUFFER_SIZE)
            toWrite = WRITE_BUFFER_SIZE;
        write_buf_int = (int *)write_buf;
        write_buf_int[0] = msg->sender;
        write_buf_int[1] = msg->send_type;
        write_buf_int[2] = msg->type;
        write_buf_int[3] = msg->data.length();
        if (toWrite > WRITE_BUFFER_SIZE)
            toWrite = WRITE_BUFFER_SIZE;
        memcpy(write_buf + headerSize, msg->data.data(), toWrite - headerSize);
        sendMaster(write_buf, toWrite);
        written += toWrite;
        while (written < len)
        {
            toWrite = len - written;
            if (toWrite > WRITE_BUFFER_SIZE)
                toWrite = WRITE_BUFFER_SIZE;
            sendMaster(msg->data.data() + written - headerSize, toWrite);
            written += toWrite;
        }
    }
}

int vvMSController::readMaster(UdpMessage* msg)
{
	char read_buf[UDP_MESSAGE_HEADER_SIZE];
	int ret = readMaster(read_buf, UDP_MESSAGE_HEADER_SIZE);
	if (ret < UDP_MESSAGE_HEADER_SIZE)
		return -1;
	int* read_buf_int = (int*)read_buf;
	msg->type = (udp_msg_type)read_buf_int[0];
	msg->sender = read_buf_int[1];
	//cerr << "reading master, type = " << read_buf_int[0] << " sender = " << read_buf_int[1] << " length = " << read_buf_int[2] << endl;
	if (read_buf_int[2] >  WRITE_BUFFER_SIZE - UDP_MESSAGE_HEADER_SIZE)
	{
		cerr << "udp message of type " << msg->type << " was too long to read;" << endl;
		return 0;
	}
    msg->data = DataHandle(read_buf_int[2]);
	ret = readMaster(msg->data.accessData(), msg->data.length());
	return ret;
}
void vvMSController::sendMaster(const std::string &s)
{
    int sz = (int)s.size();
    sendMaster(&sz, sizeof(sz));
    sendMaster(s.c_str(), sz);
}

void vvMSController::readSlave(int i, std::string &s)
{
    int sz = 0;
    readSlave(i, &sz, sizeof(sz));
    std::vector<char> d(sz);
    readSlave(i, d.data(), sz);
    std::string result(d.data(), sz);
    s = result;
}

// Default for readMaster: if multicast is set, do not send over TCP
int vvMSController::readMaster(void *c, int n)
{
    assert(isSlave());

    return readMaster(c, n, false);
}
// bool mcastOverTCP: if using multicast, control whether to read via TCP socket
// Needs to be set true in calling function when:
//   SyncMode is Multicast AND (connecting OR syncing)
int vvMSController::readMaster(void *c, int n, bool mcastOverTCP)
{
    assert(isSlave());

    int ret, read = 0;
    double startTime = 0.0;
#if defined(NOMCAST) || !defined(HAVE_NORM)
    (void)mcastOverTCP;
#endif
    if (m_drawStatistics)
    {
        startTime = vvPluginSupport::instance()->currentTime();
    }
#if !defined(NOMCAST) && defined(HAVE_NORM)
    if (syncMode == SYNC_MULTICAST && !mcastOverTCP)
    {
        if (multicast->read_mcast(c, n) != Rel_Mcast::RM_OK)
        {
            cerr << "multicast read failed" << endl;
            delete multicast;
            exit(0);
        }
        else
            return n;
    }
    else
#endif
#ifdef HAS_MPI
        if (syncMode == SYNC_MPI)
    {
        MPI_Status status;
        MPI_Recv(c, n, MPI_BYTE, 0, AppTag, appComm, &status);

        if (m_drawStatistics)
        {
            networkRecv += vvPluginSupport::instance()->currentTime() - startTime;
        }

        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        return count;
    }
    else
#endif
    {
#ifdef DEBUG_MESSAGES
        int checkRead = 0;
        int num;
        if (debugMessagesCheck)
        {
            while (checkRead < sizeof(n))
            {
                do
                {
                    ret = socket->Read((char *)(&(num)) + checkRead, sizeof(n) - checkRead);

                } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
                if (ret < 0)
                    return ret;
                checkRead += ret;
                if (num != n)
                {
                    cerr << "tried to read " << n << " but received " << num << endl;
                    sleep(1000);
                }
            }
            checkRead = 0;
            while (checkRead < sizeof(debugMessageCounter))
            {
                do
                {
                    ret = socket->Read((char *)(&(num)) + checkRead, sizeof(debugMessageCounter) - checkRead);

                } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
                if (ret < 0)
                    return ret;
                checkRead += ret;
                if (num != debugMessageCounter)
                {
                    cerr << "tried to read message number " << debugMessageCounter << " but received message number " << num << endl;
                    sleep(1000);
                }
            }
            debugMessageCounter++;
            sendMaster(&n, sizeof(n));
        }

#endif
        while (read < n)
        {
            do
            {
                ret = socket->Read((char *)c + read, n - read);

            } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
            if (m_drawStatistics)
            {
                networkRecv += vvPluginSupport::instance()->currentTime() - startTime;
            }
            if (ret < 0)
                return ret;
            read += ret;
        }
    }
    return read;
}

void vvMSController::sendMaster(const void *c, int n)
{
    assert(isSlave());

    int ret;
    double startTime = 0.0;
    if (m_drawStatistics)
    {
        startTime = vvPluginSupport::instance()->currentTime();
    }

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        CO_MPI_SEND(const_cast<void *>(c), n, MPI_BYTE, 0, AppTag, appComm);
    }
    else
#endif
    {
        do
        {
            ret = socket->write(c, n);
            if (ret <= 0)
            {
                cerr << "Return Value = " << ret;
                perror("testit2:");
            }

        } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    }

    if (m_drawStatistics)
    {
        networkSend += vvPluginSupport::instance()->currentTime() - startTime;
    }
}

int vvMSController::readSlave(int slaveNum, void *data, int num)
{
    assert(isMaster());
    assert(slaveNum >= 0);
    assert(slaveNum < getNumSlaves());

    return slaves[slaveNum]->read(data, num);
}

int vvMSController::readSlaves(SlaveData *c)
{
    assert(isMaster());

    int i;
    int ret = 0;
    double startTime = 0.0;
    if (m_drawStatistics)
    {
        startTime = vvPluginSupport::instance()->currentTime();
    }
    for (i = 0; i < numSlaves; i++)
    {
        ret = slaves[i]->read(c->data.at(i), c->size());
        (void)ret;
        //       if(ret<c->size())
        //       {
        //          cerr << "vvMSController::readSlaves err: slave " << i << ", error = " << ret << endl;
        //          perror("readSlaves error:");
        //          return -1;
        //       }
    }
    if (m_drawStatistics)
    {
        networkRecv += vvPluginSupport::instance()->currentTime() - startTime;
    }
    return c->size();
}

std::vector<std::string> vvMSController::readSlaves(const std::string &s)
{
    assert(isMaster());
    std::vector<std::string> retval(numSlaves);
    for (int i = 0; i < numSlaves; i++)
        readSlave(i, retval[i]);
    return retval;
}

// Default for readMasterDraw: if multicast is set, do not send over TCP
int vvMSController::readMasterDraw(void *c, int n)
{
    assert(isSlave());

    return readMasterDraw(c, n, false);
}
// bool mcastOverTCP: if using multicast, control whether to read via TCP socket
// Needs to be set true in calling function when:
//   SyncMode is Multicast AND (connecting OR syncing)
int vvMSController::readMasterDraw(void *c, int n, bool mcastOverTCP)
{
    assert(isSlave());

#if defined(NOMCAST) || !defined(HAVE_NORM)
    (void)mcastOverTCP;
#else
    if (syncMode == SYNC_MULTICAST && !mcastOverTCP)
    {
        if (multicast->read_mcast(c, n) != Rel_Mcast::RM_OK)
        {
            cerr << "multicast read failed" << endl;
            delete multicast;
            exit(0);
        }
        else
            return n;
    }
    else
#endif
#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        MPI_Status status;
        MPI_Recv(const_cast<void *>(c), n, MPI_BYTE, 0, DrawTag, drawComm, &status);
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        return count;
    }
    else
#endif
    {
        int ret, read = 0;
        while (read < n)
        {
            do
            {
                ret = socketDraw->Read((char *)c + read, n - read);
            } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
            if (ret < 0)
                return ret;
            read += ret;
        }
        return read;
    }

    return 0;
}

void vvMSController::sendMasterDraw(const void *c, int n)
{
    assert(isSlave());

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        CO_MPI_SEND(const_cast<void *>(c), n, MPI_BYTE, 0, DrawTag, drawComm);
    }
    else
#endif
    {
        int ret;
        do
        {
            ret = socketDraw->write(c, n);
            if (ret <= 0)
            {
                cerr << "sendMasterDraw: Return Value = " << ret;
                perror("testit2:");
            }

        } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    }
}

int vvMSController::readSlavesDraw(void *c, int n)
{
    assert(isMaster());

    int i;
    int ret;
    for (i = 0; i < numSlaves; i++)
    {
        ret = slaves[i]->readDraw(c, n);
        if (ret < n)
        {
            if (ret == -1)
            {
                cerr << "vvMSController::readSlavesDraw err: slave " << i << ", error = " << ret << endl;
                cerr << "Network error: " << errno << ", " << strerror(errno) << endl;
                perror("readSlavesDraw error:");
            }
            else
            {
                cerr << "vvMSController::readSlavesDraw short read: slave " << i << ", expect=" << n << ", got=" << ret << endl;
                abort();
            }
            return -1;
        }
    }
    return n;
}

void vvMSController::sendSlavesDraw(const void *c, int n)
{
    assert(isMaster());

#if !defined(NOMCAST) && defined(HAVE_NORM)
    if (syncMode == SYNC_MULTICAST)
    {
        if (multicast->write_mcast(c, n) != Rel_Mcast::RM_OK)
        {
            delete multicast;
            exit(0);
        }
    }
#endif
    int i;
    for (i = 0; i < numSlaves; i++)
    {
        slaves[i]->sendDraw(c, n);
    }
}

void vvMSController::waitForSlavesDraw()
{
    char buf[100];
    if (master)
    {
        MARK0("VIVE cluster master waiting for slaves to send sync");
        if (vvPluginSupport::instance()->debugLevel(5))
            fprintf(stderr, "VIVE cluster master waiting for slaves to send sync");

        if (readSlavesDraw(buf, 1) < 0) // wait for all slaves
        {
            cerr << "sync_exit1 myID=" << myID << endl;
            exit(0);
        }
        MARK0("done");
    }
    else
    {
        MARK0("VIVE cluster slave sending ID as sync");
        if (vvPluginSupport::instance()->debugLevel(5))
            fprintf(stderr, "VIVE cluster slave sending ID as sync");
        *buf = (char)myID;
        sendMasterDraw(buf, 1);
    }
    MARK0("done");
    if (vvPluginSupport::instance()->debugLevel(5))
        fprintf(stderr, "done\n");
}

void vvMSController::sendGoDraw()
{
    char buf[100];
    if (master)
    {
        MARK0("VIVE cluster master send GO");
        *buf = 'g';
        //send go to all slaves
        sendSlavesDraw(buf, 1);
        MARK0("done");
    }
    else
    {
        MARK0("VIVE cluster slave receive GO");

        if (readMasterDraw(buf, 1) < 1)
        {
            cerr << "sync_exit2 myID=" << myID << endl;
            exit(0);
        }
        MARK0("done");
    }
}

void vvMSController::barrierDraw()
{
    if (numSlaves == 0)
        return;
    MARK0("vvMSController::barrierDraw");
    if (vvPluginSupport::instance()->debugLevel(5))
        fprintf(stderr, "\nvvMSController::barrierDraw\n");

    if (syncMode == SYNC_TCP)
    {
        waitForSlavesDraw();
        sendGoDraw();
    }
    else if (syncMode == SYNC_UDP)
    {
    }
    else if (syncMode == SYNC_TCP_SERIAL)
    {
        waitForSlavesDraw();
        if (master)
        {
            sendSerialGo();
        }
        else
        {
            waitForSerialGo();
        }
    }
    else if (syncMode == SYNC_SERIAL)
    {
        if (master)
        {
            waitForSerialGo();
            sendSerialGo();
        }
        else
        {
            sendSerialGo();
            waitForSerialGo();
        }
    }
    else if (syncMode == SYNC_MAGIC)
    {

        char magicBuf = 1;
        int wrbytes;
        // I am ready
        wrbytes = write(magicFd, &magicBuf, 1);
        if (wrbytes != 1)
        {
            cerr << "vvMSController::barrier: short write" << endl;
        }
        MARK0("\tMAGIC: send READY go WAITING");

        // wait till all are ready
        int status;
        wrbytes = read(magicFd, &magicBuf, 1);
        if (wrbytes != 1)
        {
            cerr << "vvMSController::barrier: short read" << endl;
        }
        status = magicBuf & 0x20;

        while (status == 0)
        {
            if (read(magicFd, &magicBuf, 1) != 1)
            {
                cerr << "vvMSController::barrier: short read2" << endl;
            }
            status = magicBuf & 0x20;
        }
        MARK0("\tMAGIC: received GO");
    }
    else if (syncMode == SYNC_PARA)
    {
        if (master)
        {
            waitForParallelJoin();
            sendParallelGo();
        }
        else
        {
            sendParallelGo();
            waitForParallelGo();
        }
    }
#if !defined(NOMCAST) && defined(HAVE_NORM)
    else if (syncMode == SYNC_MULTICAST)
    {
        waitForSlavesDraw(); // over TCP
        sendGoDraw(); // over Multicast
    }
#endif
#ifdef HAS_MPI
    else if (syncMode == SYNC_MPI)
    {
#ifdef MPI_BARRIER
        if (vvViewer::mustDraw())
            MPI_Barrier(drawComm);
#else
        waitForSlavesDraw();
        sendGoDraw();
#endif
    }
#endif
}

void vvMSController::sendSlave(int i, const void *c, int n)
{
    assert(isMaster());
    assert(i >= 0);
    assert(i < getNumSlaves());

    double startTime = 0.0;
    if (m_drawStatistics)
    {
        startTime = vvPluginSupport::instance()->currentTime();
    }
#ifdef DEBUG_MESSAGES
    slaves[i]->send(&n, sizeof(n));
    slaves[i]->send(&debugMessageCounter, sizeof(debugMessageCounter));
    slaves[i]->read(&n, sizeof(n));
    debugMessageCounter++;
#endif
    slaves[i]->send(c, n);
    //std::cerr << i << " : " << (char*) data.data[i] << std::endl;
    //std::cerr << i << " : " << data.size() << std::endl;

    if (m_drawStatistics)
    {
        networkSend += vvPluginSupport::instance()->currentTime() - startTime;
    }
}

void vvMSController::sendSlaves(const SlaveData &data)
{
    assert(isMaster());

    int i;
    double startTime = 0.0;
    if (m_drawStatistics)
    {
        startTime = vvPluginSupport::instance()->currentTime();
    }
#ifdef DEBUG_MESSAGES
    int n = data.size();
    for (i = 0; i < numSlaves; i++)
    {
        slaves[i]->send(&n, sizeof(n));
        slaves[i]->send(&debugMessageCounter, sizeof(debugMessageCounter));
        slaves[i]->read(&n, sizeof(n));
    }
    debugMessageCounter++;
#endif
    for (i = 0; i < numSlaves; i++)
    {
        slaves[i]->send(data.data[i], data.size());
        //std::cerr << i << " : " << (char*) data.data[i] << std::endl;
        //std::cerr << i << " : " << data.size() << std::endl;
    }

    if (m_drawStatistics)
    {
        networkSend += vvPluginSupport::instance()->currentTime() - startTime;
    }
}

void vvMSController::sendSlaves(const void *c, int n)
{
    assert(isMaster());

    int i;
    double startTime = 0.0;
    if (m_drawStatistics)
    {
        startTime = vvPluginSupport::instance()->currentTime();
    }
#if !defined(NOMCAST) && defined(HAVE_NORM)
    if (syncMode == SYNC_MULTICAST)
    {
        if (multicast->write_mcast(c, n) != Rel_Mcast::RM_OK)
        {
            delete multicast;
            exit(0);
        }
    }
    else
#endif
    {
#ifdef DEBUG_MESSAGES
        for (i = 0; i < numSlaves; i++)
        {
            slaves[i]->send(&n, sizeof(n));
            slaves[i]->send(&debugMessageCounter, sizeof(debugMessageCounter));

            slaves[i]->read(&n, sizeof(n));
        }
        debugMessageCounter++;
#endif
        for (i = 0; i < numSlaves; i++)
        {
            slaves[i]->send(c, n);
        }
    }
    if (m_drawStatistics)
    {
        networkSend += vvPluginSupport::instance()->currentTime() - startTime;
    }
}

void vvMSController::startSlaves()
{
    if (numSlaves == 0)
    {
        return;
    }
    if (master)
    {
        for (int i = 0; i < numSlaves; i++)
        {
//cerr << "new vvSlave(" << i+1 << ")" << endl;
#ifdef HAS_MPI
            if (syncMode == SYNC_MPI)
            {
                slaves[i] = new vvMpiSlave(i + 1, appComm, drawRank[i+1], drawComm);
            }
            else
#endif
            {
                slaves[i] = new vvTcpSlave(i + 1);
            }
        }
        for (int i = 0; i < numSlaves; i++)
        {
            //cerr << "slaves[" << i << "]->start(" << endl;
            slaves[i]->start();
        }
        for (int i = 0; i < numSlaves; i++)
        {
            slaves[i]->accept();
        }
#ifdef DEBUG_MESSAGES
        debugMessageCounter++;
#endif
    }
}

void vvMSController::waitForSlaves()
{
    char buf[100];
    if (master)
    {

        static SlaveData result(1);
        MARK0("VIVE cluster master waiting for slaves to send sync");
        if (vvPluginSupport::instance()->debugLevel(5))
            fprintf(stderr, "VIVE cluster master waiting for slaves to send sync");

        if (readSlaves(&result) < 0) // wait for all slaves
        {
            cerr << "sync_exit1 myID=" << myID << endl;
            exit(0);
        }
        MARK0("done");
    }
    else
    {
        MARK0("VIVE cluster slave sending ID as sync");
        if (vvPluginSupport::instance()->debugLevel(5))
            fprintf(stderr, "VIVE cluster slave sending ID as sync");
        *buf = (char)myID;
        sendMaster(buf, 1);
    }
    MARK0("done");
    if (vvPluginSupport::instance()->debugLevel(5))
        fprintf(stderr, "done\n");
}

void vvMSController::waitForMaster()
{
    char buf[100];
    if (master)
    {
        MARK0("VIVE cluster master waiting for master to send sync");
        if (vvPluginSupport::instance()->debugLevel(5))
            cerr << "VIVE cluster master waiting for master to send sync" << endl;
        sendSlaves(buf, 1);
        MARK0("done");
    }
    else
    {
        MARK0("VIVE cluster slave sending ID as sync");
        if (vvPluginSupport::instance()->debugLevel(5))
            cerr << "VIVE cluster slave sending ID as sync" << endl;
        *buf = (char)myID;
        readMaster(buf, 1);
    }
    MARK0("done");
    if (vvPluginSupport::instance()->debugLevel(5))
        cerr << "waitforslaves done" << endl;
}

void vvMSController::sendGo()
{
    char buf[100];
    if (master)
    {
        MARK0("VIVE cluster master send GO");
        *buf = 'g';
        //send go to all slaves
        sendSlaves(buf, 1);
        MARK0("done");
    }
    else
    {
        MARK0("VIVE cluster slave receive GO");

        if (readMaster(buf, 1) < 1)
        {
            cerr << "sync_exit2 myID=" << myID << endl;
            exit(0);
        }
        MARK0("done");
    }
}

void vvMSController::startupBarrier()
{
    if (debugLevel(3))
        fprintf(stderr, "\nvvMSController::startupBarrier: numSlaves=%d\n", numSlaves);

    std::string masterName;
    if (isMaster())
    {
        Host local;
        masterName = local.getName();
    }
    masterName = syncString(masterName);
    covise::coConfigConstants::setMaster(masterName.c_str());

    if (numSlaves == 0)
        return;

    if ((syncMode == SYNC_SERIAL) || (syncMode == SYNC_MAGIC) || (syncMode == SYNC_PARA))
    {
        waitForSlaves();
        sendGo();
    }
    else
    {
        barrier();
    }
}

void vvMSController::shmBarrier()
{
#ifdef HAS_MPI
#ifndef __APPLE__
    if (pthreadShmBarrier)
        pthread_barrier_wait(pthreadShmBarrier);
#endif
#endif
}

void vvMSController::barrier()
{
    if (numSlaves == 0)
        return;
    MARK0("vvMSController::barrier");
    if (vvPluginSupport::instance()->debugLevel(5))
        fprintf(stderr, "\nvvMSController::barrier\n");

    if (syncMode == SYNC_TCP)
    {
        waitForSlaves();
        sendGo();
    }
    else if (syncMode == SYNC_UDP)
    {
    }
    else if (syncMode == SYNC_TCP_SERIAL)
    {
        waitForSlaves();
        if (master)
        {
            sendSerialGo();
        }
        else
        {
            waitForSerialGo();
        }
    }
    else if (syncMode == SYNC_SERIAL)
    {
        if (master)
        {
            waitForSerialGo();
            sendSerialGo();
        }
        else
        {
            sendSerialGo();
            waitForSerialGo();
        }
    }
    else if (syncMode == SYNC_MAGIC)
    {

        char magicBuf = 1;
        int wrbytes;
        // I am ready
        wrbytes = write(magicFd, &magicBuf, 1);
        if (wrbytes != 1)
        {
            cerr << "vvMSController::barrier: short write" << endl;
        }
        MARK0("\tMAGIC: send READY go WAITING");

        // wait till all are ready
        int status;
        wrbytes = read(magicFd, &magicBuf, 1);
        if (wrbytes != 1)
        {
            cerr << "vvMSController::barrier: short read" << endl;
        }
        status = magicBuf & 0x20;

        while (status == 0)
        {
            if (read(magicFd, &magicBuf, 1) != 1)
            {
                cerr << "vvMSController::barrier: short read2" << endl;
            }
            status = magicBuf & 0x20;
        }
        MARK0("\tMAGIC: received GO");
    }
    else if (syncMode == SYNC_PARA)
    {
        if (master)
        {
            waitForParallelJoin();
            sendParallelGo();
        }
        else
        {
            sendParallelGo();
            waitForParallelGo();
        }
    }
#if !defined(NOMCAST) && defined(HAVE_NORM)
    else if (syncMode == SYNC_MULTICAST)
    {
        waitForSlaves(); // over TCP
        sendGo(); // over Multicast
    }
#endif
#ifdef HAS_MPI
    else if (syncMode == SYNC_MPI)
    {
        shmBarrier();
#ifdef MPI_BARRIER
        MPI_Barrier(appComm);
#else
        waitForSlaves();
        sendGo();
#endif
    }
#endif
}

void vvMSController::sendSerialGo()
{
    MARK0("vvMSController::sendSerialGo");
#ifndef _WIN32
    int statusByte;
    static bool state = true;
    if (state)
    {
        ioctl(serial, TIOCMGET, &statusByte);
        statusByte |= TIOCM_RTS;
        if (ioctl(serial, TIOCMSET, &statusByte) == -1)
            cerr << "RTS=1 ERROR" << endl;
        state = false;
    }
    else
    {
        ioctl(serial, TIOCMGET, &statusByte);
        statusByte &= ~(TIOCM_RTS);
        if (ioctl(serial, TIOCMSET, &statusByte) == -1)
            cerr << "RTS=0 ERROR" << endl;
        state = true;
    }
#endif
}

void vvMSController::waitForSerialGo()
{
    MARK0("vvMSController::waitForSerialGo");
#ifndef _WIN32
    int statusByte = 0;
    static bool state = false;
    do
    {
        ioctl(serial, TIOCMGET, &statusByte);
    } while ((statusByte & TIOCM_CTS) == state);
    state = !state;
#endif
}

void vvMSController::sendParallelGo()
{
#ifdef __linux__
    static bool state = false;
    unsigned char statusByte = 0xff;
    if (state)
        statusByte = 0x0;
    ioctl(parallel, PPWDATA, &statusByte);
    /*if(master)
   fprintf(stderr,"Master");
   fprintf(stderr,"Go\n");*/
    state = !state;
#endif
}

void vvMSController::waitForParallelGo()
{
    int myBit;
    myBit = 1 << (myID + 2);
    myBit = 1 << 3;
/* int i;
       for(i=0;i<8;i++)
       {
        if(myBit & (1<<i))
        {
           fprintf(stderr,"1");
      }
       else
       {
          fprintf(stderr,"0");
          }
   }
   fprintf(stderr,"myBit\n");*/

#ifdef __linux__
    static bool state = false;
    //fprintf(stderr,"myID: %d s=%d\n",myID,state);
    /*if(master)
   fprintf(stderr,"Master");
   fprintf(stderr,"wait\n");*/
    unsigned char statusByte = 0x0;
    if (state)
    {
        do
        {
            ioctl(parallel, PPRSTATUS, &statusByte);
            /*	 int i;
                for(i=0;i<8;i++)
                {
                 if(statusByte & (1<<i))
                 {
                    fprintf(stderr,"1");
               }
                else
                {
                   fprintf(stderr,"0");
                   }
         }
         fprintf(stderr,"wait\n");*/

        } while (statusByte & myBit);
    }
    else
    {
        do
        {
            ioctl(parallel, PPRSTATUS, &statusByte);
            /*int i;
            for(i=0;i<8;i++)
            {
             if(statusByte & (1<<i))
             {
                fprintf(stderr,"1");
           }
            else
            {
               fprintf(stderr,"0");
               }
         }
         fprintf(stderr,"!wait\n");
         for(i=0;i<8;i++)
         {
         if(myBit & (1<<i))
         {
         fprintf(stderr,"1");
         }
         else
         {
         fprintf(stderr,"0");
         }
         }
         fprintf(stderr,"myBit\n");*/

        } while (!(statusByte & myBit));
    }
    //fprintf(stderr,"finished myID: %d s=%d\n",myID,state);
    /*if(master)
   fprintf(stderr,"Master");
   fprintf(stderr,"waitFinished\n");*/
    state = !state;
#endif
}

void vvMSController::waitForParallelJoin()
{
#ifdef __linux__
    /*int i;
      for(i=0;i<8;i++)
      {
       if(allChildren & (1<<i))
       {
          fprintf(stderr,"1");
          }
           else
      {
         fprintf(stderr,"0");
         }
   }
   fprintf(stderr,"AllChildren\n");

   if(master)
   fprintf(stderr,"Master");
   fprintf(stderr,"join\n"); */

    static bool state = true;
    //fprintf(stderr,"join myID: %d s=%d\n",myID,state);
    unsigned char statusByte = 0x0;
    if (state)
    {
        do
        {
            ioctl(parallel, PPRSTATUS, &statusByte);
            /*	 int i;
                for(i=0;i<8;i++)
                {
                 if(statusByte & (1<<i))
                 {
                    fprintf(stderr,"1");
               }
                else
                {
                   fprintf(stderr,"0");
                   }
         }
         fprintf(stderr,"join\n");
         for(i=0;i<8;i++)
         {
         if(allChildren & (1<<i))
         {
         fprintf(stderr,"1");
         }
         else
         {
         fprintf(stderr,"0");
         }
         }
         fprintf(stderr,"AllChildren\n");
         fprintf(stderr,"s&a%d\n",(statusByte&allChildren));*/

        } while (!((statusByte & allChildren) == allChildren));
    }
    else
    {
        do
        {
            /*         ioctl(parallel, PPRSTATUS, &statusByte);
             int i;
                for(i=0;i<8;i++)
                {
                 if(statusByte & (1<<i))
                 {
                    fprintf(stderr,"1");
               }
                else
                {
                   fprintf(stderr,"0");
         }
         }
         fprintf(stderr,"!join\n");*/

        } while (!((statusByte & allChildren) == 0));
    }
    //fprintf(stderr,"join finished myID: %d s=%d\n",myID,state);
    state = !state;
/* if(master)
    fprintf(stderr,"Master");
    fprintf(stderr,"joinFinished\n");*/
#endif
}

void vvMSController::barrierApp(int frameNum)
{
    if (numSlaves == 0)
        return;
    if (master)
    {
        sendSlaves(&frameNum, sizeof(frameNum));
    }
    else
    {
        int masterFrameNum = 0;
        if (readMaster(&masterFrameNum, sizeof(masterFrameNum)) < 0)
        {
            cerr << "bcould not read message from Master" << endl;
            cerr << "sync_exit15a myID=" << myID << endl;
            exit(0);
        }
        if (masterFrameNum != frameNum)
        {
            cerr << "myId=" << myID << ": frame numbers differ: master=" << masterFrameNum << ", me=" << frameNum << std::endl;
            exit(0);
        }
    }
    if (barrierProcess != SYNC_APP)
        return;

    //double sTime=0.0;
    //sTime = vvPluginSupport::instance()->currentTime();
    //cerr << "id: " << myID << " time: " << vvPluginSupport::instance()->currentTime()-sTime << endl;

    MARK0("VIVE barrierApp");
    if (vvPluginSupport::instance()->debugLevel(5))
        fprintf(stderr, "\nvvMSController::barrierApp\n");

    barrier();
    MARK0("VIVE barrierApp done");
}

void vvMSController::agreeInt(int value)
{
    if (numSlaves == 0)
        return;
    if (master)
    {
        sendSlaves(&value, sizeof(value));
    }
    else
    {
        int masterValue = 0;
        if (readMaster(&masterValue, sizeof(masterValue)) < 0)
        {
            cerr << "bcould not read message from Master" << endl;
            cerr << "agreeInt_exit15a myID=" << myID << endl;
            exit(0);
        }
        if (masterValue != value)
        {
            cerr << "values differ master:" << masterValue << "me:" << value <<endl;
            cerr << "myID=" << myID << endl;
            while(true)
{
 // loop forever so that we can attach a debugger
}
        }
    }
    barrier();
}

void vvMSController::agreeFloat(float value)
{
    if (numSlaves == 0)
        return;
    if (master)
    {
        sendSlaves(&value, sizeof(value));
    }
    else
    {
        float masterValue = 0;
        if (readMaster(&masterValue, sizeof(masterValue)) < 0)
        {
            cerr << "bcould not read message from Master" << endl;
            cerr << "agreeInt_exit15a myID=" << myID << endl;
            exit(0);
        }
        if (masterValue != value)
        {
            cerr << "values differ master:" << masterValue << "me:" << value <<endl;
            cerr << "myID=" << myID << endl;
            while(true)
{
 // loop forever so that we can attach a debugger
}
        }
    }
    barrier();
}
void vvMSController::agreeString(std::string s)
{
    if (numSlaves == 0)
        return;
    if (s.length() == 0)
        return;
    if (master)
    {
        int len = (int)s.length();
	const char *buf = s.c_str();
        sendSlaves(&len, sizeof(len));
        sendSlaves(buf, len+1);
    }
    else
    {
        int masterValue = 0;
	std::string str;
        if (readMaster(&masterValue, sizeof(masterValue)) < 0)
        {
            cerr << "bcould not read message from Master" << endl;
            cerr << "agreeInt_exit15a myID=" << myID << endl;
            exit(0);
        }
        if (masterValue != s.length())
        {
            cerr << "values differ master:" << masterValue << "me:" << s.length() <<endl;
            cerr << "myID=" << myID << endl;
            while(true)
	    {
	     // loop forever so that we can attach a debugger
	    }
	 }
char *buf = new char[masterValue+1];
        if (readMaster(buf, masterValue+1) < 0)
        {
            cerr << "bcould not read message from Master" << endl;
            cerr << "agreeInt_exit15a myID=" << myID << endl;
            exit(0);
        }
	str = std::string(buf);
	
        if (str != s)
        {
            cerr << "values differ master:" << str << "me:" << s <<endl;
            cerr << "myID=" << myID << endl;
            while(true)
	    {
	     // loop forever so that we can attach a debugger
	    }
	 }
    }
    barrier();
}
//sync Time and handle Cluster statistics

void vvMSController::syncTime()
{

    if (numSlaves == 0)
        return;
    int i;
    static bool oldStat = false;
    if ((oldStat != m_drawStatistics) && (master) && vvPluginSupport::instance()->getScene() != 0)
    {
       
        oldStat = m_drawStatistics;
    }
    if (m_drawStatistics && vvPluginSupport::instance()->getScene() != 0)
    {
        static double lastTime = 0;
        double currentTime = vvPluginSupport::instance()->currentTime();
        if (master)
        {
            unsigned int ret;
            char buf[3 * sizeof(double)];
            for (i = 0; i < numSlaves; i++)
            {
                ret = slaves[i]->read(buf, 3 * sizeof(double));
                if (ret < 3 * sizeof(double))
                {
                    cerr << "Return Value = " << ret << "slave" << i << endl;
                    perror("readSlaves error:");
                    return;
                }

                double frameTime;
                memcpy(&frameTime, buf, sizeof(double));
                memcpy(&networkSend, buf + sizeof(double), sizeof(double));
                memcpy(&networkRecv, buf + 2 * sizeof(double), sizeof(double));
                fprintf(stderr, "slave: % 2d frameTime: %-10.5lf networkRecv: %-10.5lf networkSend: %-10.5lf\n", i, frameTime, networkRecv, networkSend);
                //cerr << "slave: " << i << " frameTime: " << frameTime << " networkRecv: "<< networkRecv << " networkSend: "<< networkSend<< endl;
            }
            //get global min/max to be able to compare graphs
            float globalMax = 0;
            float globalSendMax = 0;
            float globalRecvMax = 0;
        }
        else
        {
            char buf[3 * sizeof(double)];
            // sendTime Diff to master
            int len = 0;
            *((double *)(buf + len)) = currentTime - lastTime;
            len += sizeof(double);
            *((double *)(buf + len)) = networkSend;
            len += sizeof(double);
            *((double *)(buf + len)) = networkRecv;
            len += sizeof(double);
            sendMaster(buf, len);
        }
        networkRecv = 0;
        networkSend = 0;
        lastTime = currentTime;
    }

    if (vvPluginSupport::instance()->debugLevel(4))
        fprintf(stderr, "\nvvMSController::syncTime\n");

    double frameTime, frameRealTime;
    if (master)
    {
        frameTime = vvPluginSupport::instance()->frameTime();
        frameRealTime = vvPluginSupport::instance()->frameRealTime();
        sendSlaves(&frameTime, sizeof(double));
        sendSlaves(&frameRealTime, sizeof(double));
    }
    else
    {
        if (readMaster(&frameTime, sizeof(double)) < 0
            || readMaster(&frameRealTime, sizeof(double)) < 0)
        {
            cerr << "ccould not read message from Master" << endl;
            cerr << "sync_exit14 myID=" << myID << endl;
            exit(0);
        }
        vvPluginSupport::instance()->setFrameTime(frameTime);
        vvPluginSupport::instance()->setFrameRealTime(frameRealTime);
    }

    if (syncMode == SYNC_MAGIC)
    {
        waitForSlaves();
        waitForMaster();
        // I am busy again
        char magicBuf = 0;
        if (write(magicFd, &magicBuf, 1) != 1)
        {
            cerr << "vvMSController::syncTime: short write" << endl;
        }
        MARK0("\tMAGIC: send BUSY (after tcp sync with acknowledge\n");
    }
}

int vvMSController::syncData(void *data, int size)
{
#if defined(HAS_MPI) && defined(MPI_BCAST)
    if (syncMode == SYNC_MPI)
    {
        shmBarrier();
        MPI_Bcast(data, size, MPI_BYTE, 0, appComm);
        return size;
    }
#endif

    if (isMaster())
    {
        sendSlaves(data, size);
    }
    else
    {
        if (readMaster(data, size) < 0)
        {
            cerr << "dcould not read message from Master" << endl;
            cerr << "sync_exit15b myID=" << myID << endl;
            exit(0);
        }
    }
    return size;
}

int vvMSController::syncMessage(covise::Message *msg)
{
    const int headerSize = 4;
    int buffer[headerSize];

    if (!vvMSController::instance()->isCluster())
        return sizeof(buffer)+msg->data.length();

    if (vvMSController::instance()->isMaster())
    {
        buffer[0] = msg->sender;
        buffer[1] = msg->send_type;
        buffer[2] = msg->type;
        buffer[3] = msg->data.length();
    }
    int ret = syncData(&buffer[0], sizeof(buffer));
    if (ret >= 0)
    {
        if (vvMSController::instance()->isSlave())
        {
            msg->sender = buffer[0];
            msg->send_type = buffer[1];
            msg->type = buffer[2];
            msg->data = DataHandle(buffer[3]);
        }
        int n = syncData(msg->data.accessData(), msg->data.length());
        if (n >= 0)
            return ret + n;
    }
    return -1;
}

bool vvMSController::syncBool(bool state)
{
    char c = state ? 1 : 0;
    syncData(&c, 1);
    return (c != 0);
}



bool vvMSController::reduceOr(bool val)
{
    if (numSlaves == 0)
        return val;

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        shmBarrier();
        int in = val ? 1 : 0, out = in;
        MPI_Reduce(&in, &out, 1, MPI_INT, MPI_LOR, 0, appComm);
        return out != 0;
    }
#endif

    if (isMaster())
    {
        SlaveData sd(sizeof(bool));
        instance()->readSlaves(&sd);
        for (int s=0; s<getNumSlaves(); ++s) {
            bool sval = *static_cast<bool *>(sd.data[s]);
            val = val || sval;
         }
    }
    else
    {
        sendMaster(&val, sizeof(val));
    }
    return val;
}

bool vvMSController::reduceAnd(bool val)
{
    if (numSlaves == 0)
        return val;

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        shmBarrier();
        int in = val ? 1 : 0, out = in;
        MPI_Reduce(&in, &out, 1, MPI_INT, MPI_LAND, 0, appComm);
        return out != 0;
    }
#endif

    if (isMaster())
    {
        SlaveData sd(sizeof(bool));
        instance()->readSlaves(&sd);
        for (int s=0; s<getNumSlaves(); ++s) {
            bool sval = *static_cast<bool *>(sd.data[s]);
            val = val && sval;
         }
    }
    else
    {
        sendMaster(&val, sizeof(val));
    }
    return val;
}

bool vvMSController::allReduceOr(bool val)
{
    if (numSlaves == 0)
        return val;

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        shmBarrier();
        int in = val ? 1 : 0, out = in;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LOR, appComm);
        return out != 0;
    }
#endif

    return syncBool(reduceOr(val));
}

bool vvMSController::allReduceAnd(bool val)
{
    if (numSlaves == 0)
        return val;

#ifdef HAS_MPI
    if (syncMode == SYNC_MPI)
    {
        shmBarrier();
        int in = val ? 1 : 0, out = in;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LAND, appComm);
        return out != 0;
    }
#endif

    return syncBool(reduceAnd(val));
}

std::string vvMSController::syncString(const std::string &s)
{
    if (numSlaves == 0)
        return s;

    int sz = 0;
    if (isMaster())
    {
        sz = (int)s.size();
        sendSlaves(&sz, sizeof(sz));
        if (sz > 0)
            sendSlaves(s.c_str(), sz);
        return s;
    }
    else
    {
        readMaster(&sz, sizeof(sz));
        if (sz > 0)
        {
            std::vector<char> v(sz);
            readMaster(&v[0], sz);
            std::string r(&v[0], sz);
            return r;
        }
        else
        {
            return std::string();
        }
    }
}

template<typename T> 
typename std::enable_if<std::is_pod<T>::value, std::vector<T>>::type vvMSController::syncVector(const std::vector<T> &vec)
{
    std::vector<T> retval = vec;
    auto s = retval.size();
    syncData(&s, sizeof(typename std::vector<T>::size_type));
    retval.resize(s);
    if(s == 0)
        return retval;
    syncData(retval.data(), (int)s * sizeof(T));
    return retval; 
}

#define INSTANTIATE_SYNCVECTOR(type)\
template std::vector<type> vvMSController::syncVector(const std::vector<type> &vec);

INSTANTIATE_SYNCVECTOR(signed short)
INSTANTIATE_SYNCVECTOR(unsigned short)
INSTANTIATE_SYNCVECTOR(signed)
INSTANTIATE_SYNCVECTOR(unsigned)
INSTANTIATE_SYNCVECTOR(signed long)
INSTANTIATE_SYNCVECTOR(unsigned long)
INSTANTIATE_SYNCVECTOR(signed long long)
INSTANTIATE_SYNCVECTOR(unsigned long long)
INSTANTIATE_SYNCVECTOR(float)
INSTANTIATE_SYNCVECTOR(double)

std::vector<std::string> vvMSController::syncVector(const std::vector<std::string> &vec)
{
    std::vector<std::string> retval = vec;
    auto s = retval.size();
    syncData(&s, sizeof(typename std::vector<std::string>::size_type));
    retval.resize(s);
    if(s == 0)
        return retval;
    for (size_t i = 0; i < s; i++)
    {
        retval[i] = syncString(retval[i]);
    }
    return retval; 
}

bool vvMSController::syncVRBMessages()
{
#define MAX_VRB_MESSAGES 500
    Message *vrbMsgs[MAX_VRB_MESSAGES];
	UdpMessage* udpMsgs[MAX_VRB_MESSAGES];
    int numVrbMessages = 0;
	int numUdpMessages = 0;

    if (vvPluginSupport::instance()->debugLevel(5))
        fprintf(stderr, "\nvvMSController::syncVRBMessages\n");

    Message *vrbMsg = new Message;
	UdpMessage* udpMsg = new UdpMessage;
	if (master)
	{
		if (vvVIVE::instance()->isVRBconnected())
		{
			//poll tcp messages
			while (vvVIVE::instance()->vrbc()->poll(vrbMsg))
			{
				vrbMsgs[numVrbMessages] = vrbMsg;
				numVrbMessages++;
                if (vrbMsg->type == COVISE_MESSAGE_SOCKET_CLOSED)
                {
                    vvVIVE::instance()->restartVrbc();
                    delete udpMsg;
                    return false;
                }

                vrbMsg = new Message;
				if (numVrbMessages >= MAX_VRB_MESSAGES)
				{
					cerr << "too many VRB Messages!!" << endl;
					break;
				}
				if (!vvVIVE::instance()->isVRBconnected())
					break;
			}
			//poll udp messages
			while (vvVIVE::instance()->vrbc()->pollUdp(udpMsg))
			{
				udpMsgs[numUdpMessages] = udpMsg;
				numUdpMessages++;
				udpMsg = new UdpMessage;
				if (numUdpMessages >= MAX_VRB_MESSAGES)
				{
					cerr << "too many UDP Messages!!" << endl;
					break;
				}
                cerr << "received udp msg from client " << udpMsg->sender << ": " << udpMsg->type << "" << endl;
				if (!vvVIVE::instance()->isVRBconnected())
					break;
			}
		}
		else
		{
			static double oldSec = 0;
			double curSec;
			curSec = vvPluginSupport::instance()->frameTime();

			// try to reconnect
			if ((curSec - oldSec) > 2.0)
			{
				if (vvPluginSupport::instance()->debugLevel(3))
				{
					fprintf(stderr, "trying to establish VRB connection\n");
				}
                vvVIVE::instance()->startVrbc();
				oldSec = curSec;
			}
		}
		sendSlaves(&numVrbMessages, sizeof(int));
		//cerr << "numMasterMSGS " <<  numVrbMessages << endl;
        for (int i = 0; i < numVrbMessages; i++)
        {
            sendSlaves(vrbMsgs[i]);
			//vvCommunication::instance()->handleVRB(*vrbMsgs[i]);
			delete vrbMsgs[i];
        }
        sendSlaves(&numUdpMessages, sizeof(int));
        for (int i = 0; i < numUdpMessages; i++)
        {
            sendSlaves(udpMsgs[i]);
			//vvCommunication::instance()->handleUdp(udpMsgs[i]);
			delete udpMsgs[i];
        }
    }
	else
	{
		//get number of Messages
		if (readMaster(&numVrbMessages, sizeof(int)) < 0)
		{
			cerr << "sync_exit16 myID=" << myID << endl;
			exit(0);
		}
        for (int i = 0; i < numVrbMessages; i++)
        {
            if (readMaster(vrbMsg) < 0)
			{
				cerr << "sync_exit17 myID=" << myID << endl;
				exit(0);
			}
			//vvCommunication::instance()->handleVRB(*vrbMsg);
        }
        if (readMaster(&numUdpMessages, sizeof(int)) < 0)
		{
			cerr << "sync_exit160 myID=" << myID << endl;
			exit(0);
		}
        for (int i = 0; i < numUdpMessages; i++)
        {
            if (readMaster(udpMsg) < 0)
			{
				cerr << "sync_exit170 myID=" << myID << endl;
				exit(0);
			}
			//vvCommunication::instance()->handleUdp(udpMsg);
        }
    }
    delete vrbMsg;
	delete udpMsg;
    return numVrbMessages > 0 || numUdpMessages > 0;
}

void vvMSController::loadFile(const char *filename)
{
    if (filename)
    {
        char buf[1000];
        snprintf(buf, 1000, "loading %s", filename);
        //vvVIVE::instance()->hud->setText3(buf);
    }
    if (numSlaves == 0)
    {
        if (filename != NULL)
        {
            //vvFileManager::instance()->loadFile(filename);
        }
        return;
    }

    if (vvPluginSupport::instance()->debugLevel(3))
        fprintf(stderr, "\nvvMSController::loadFile\n");
    int len = 0;
    if (master)
    {
        if (filename)
            len = (int)strlen(filename) + 1;
        sendSlaves(&len, sizeof(int));
        if (len > 0)
            sendSlaves(filename, len);

        if (filename != NULL)
        {
            //vvFileManager::instance()->loadFile(filename);
        }
    }
    else
    {
        int numcs;
        if (readMaster(&numcs, sizeof(int)) < 0)
        {
            cerr << "bcould not read message from Master" << endl;
        }
        cerr << "numcs" << numcs << endl;
        if (numcs)
        {
            char *buf = new char[numcs];
            if (readMaster(buf, numcs) < 0)
            {
                cerr << "ccould not read message from Master" << endl;
            }
            //vvFileManager::instance()->loadFile(buf);
            delete[] buf;
        }
    }
}
