/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VRMSController_H
#define CO_VRMSController_H

/*! \file
 \brief  cluster master

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#ifdef HAS_MPI
#include <mpi.h>
#endif

#include <string>
#include <vector>
#include <util/coTypes.h>

//#define DEBUG_MESSAGES
#ifdef DEBUG_MESSAGES
#define CLUSTER_MARK() \
    ;                  \
    coVRMSController::msController->checkMark(__FILE__, __LINE__);
#else
#define CLUSTER_MARK() ;
#endif
#define UDP_MESSAGE_HEADER_SIZE  3 *sizeof(int)

namespace covise
{
class Socket;
class Message;
class UdpMessage;
}
#define MAX_NUMBER_OF_SLAVES 256

namespace opencover
{
class coVRSlave;
class Rel_Mcast;
class coClusterStat;
class COVEREXPORT coVRMSController
{
public:
    enum
    {
        SYNC_TCP = 0,
        SYNC_UDP,
        SYNC_SERIAL,
        SYNC_TCP_SERIAL,
        SYNC_PARA,
        SYNC_MULTICAST,
        SYNC_MPI,
        SYNC_MAGIC
    };
    enum
    {
        SYNC_APP = 0,
        SYNC_DRAW
    };

    enum
    {
        AppTag,
        DrawTag
    };

    class COVEREXPORT SlaveData
    {
    public:
        /**
          * Structure that holds the results of a readSlaves call
          * @param n Maximum size of a result entry.
          */
        SlaveData(int n);
        virtual ~SlaveData();
        // Result vector
        std::vector<void *> data;

        // Maximum size of a data element. NOT the size of the data array.
        inline int size() const
        {
            return this->n;
        }

    private:
        SlaveData(const SlaveData &)
        {
        }
        int n;
    };

    static coVRMSController *instance();
    coVRMSController(int AmyID = -1, const char *addr = NULL, int port = 0);
#ifdef HAS_MPI
    coVRMSController(const MPI_Comm *comm);
#endif
    ~coVRMSController();
    void startSlaves();
    void checkMark(const char *file, int line);
    void connectToMaster(const char *addr, int port);
    bool isMaster() const
    {
        return master;
    };
    bool isSlave() const
    {
        return !master;
    };
    bool isCluster() const
    {
        return (numSlaves > 0);
    };
	void setStartSession(const std::string& sessionName);
    int readMaster(void *c, int n, bool mcastOverTCP);
    int readMaster(void *c, int n);
    int readMasterDraw(void *c, int n, bool mcastOverTCP);
    int readMasterDraw(void *c, int n);
    void sendMaster(const void *c, int n);
    void sendMasterDraw(const void *c, int n);
    int readSlave(int slaveNum, void *data, int num);
    int readSlaves(SlaveData *c);
    int readSlavesDraw(void *c, int n);
    void sendSlave(int i, const void *c, int n);
    void sendSlaves(const void *c, int n);
    void sendSlaves(const SlaveData &c);
    void sendSlavesDraw(const void *c, int n);
    void sendSlaves(const covise::Message *msg);
	void sendSlaves(const covise::UdpMessage* msg);
    int readMaster(covise::Message *msg);
    void sendMaster(const covise::Message *msg);
	int readMaster(covise::UdpMessage* msg);
	void sendMaster(const std::string &s);
    void readSlave(int i, std::string &s);
    int getID()
    {
        return myID;
    };
    void sync();
    void startupSync();
    void syncApp(int frameNum);
    void syncInt(int value);
    void syncFloat(float value);
    void syncStringStop(std::string name);
    void syncDraw();
    void syncTime();
    int syncData(void *data, int size);
    int syncMessage(covise::Message *msg);
    bool syncBool(bool);
    bool reduceOr(bool); // master will receive logical or of all inputs
    bool reduceAnd(bool);
    bool allReduceOr(bool); // master and slaves will receive logical or of all inputs
    bool allReduceAnd(bool);
    std::string syncString(const std::string &s);
    bool syncVRBMessages();
    void waitForSlaves();
    void waitForSlavesDraw();
    void waitForMaster();
    void sendGo();
    void sendGoDraw();
    void sendSerialGo();
    void waitForSerialGo();
    void sendParallelGo();
    void waitForParallelGo();
    void waitForParallelJoin();
    void loadFile(const char *filename);
    int clusterSize()
    {
        return numSlaves + 1;
    }
    void killClients();
    int getNumSlaves() const
    {
        return this->numSlaves;
    }

    void heartBeat(const std::string &name = "unnamed", bool draw = false);

    bool drawStatistics() const;
    void setDrawStatistics(bool enable);

#ifdef HAS_MPI
    MPI_Comm getAppCommunicator() const
    {
        return this->appComm;
    }
#endif

private:
	std::string startSession;
    bool debugLevel(int l) const;
    int m_debugLevel;
    bool master;
    bool slave;
    int allChildren; // one bit for each child
    int serial;
    int parallel;
    int myID;
    int numSlaves;
    int syncMode;
    int syncProcess;
    bool m_drawStatistics;
    Rel_Mcast *multicast;
    int multicastDebugLevel;
    std::string multicastAddress;
    int multicastPort;
    std::string multicastInterface;
    int multicastMTU;
    int multicastTTL;
    bool multicastLoop;
    unsigned int multicastBufferSpace;
    int multicastBlockSize;
    int multicastNumParity;
    unsigned int multicastTxCacheSize;
    int multicastTxCacheMin;
    int multicastTxCacheMax;
    int multicastTxRate;
    double multicastBackoffFactor;
    int multicastSockBuffer;
    int multicastClientTimeout;
    int multicastServerTimeout;
    int multicastRetryTimeout;
    int multicastMaxLength;

    int magicFd; // filedescriptor for magic sync

    covise::Socket *socket;
    covise::Socket *socketDraw;
    coVRSlave *slaves[MAX_NUMBER_OF_SLAVES];
    coClusterStat *stats[MAX_NUMBER_OF_SLAVES];
    double networkSend;
    double networkRecv;

#ifdef HAS_MPI
    MPI_Comm appComm;
    MPI_Comm drawComm;
#endif

    int heartBeatCounter, heartBeatCounterDraw;
    static coVRMSController *s_singleton;
};
}
#endif
