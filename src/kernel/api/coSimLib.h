/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_SIM_LIB_H_
#define _CO_SIM_LIB_H_

// 11.06.99

#include <sys/types.h>
#include <set>
#include <list>
#include <util/coviseCompat.h>
#ifndef YAC
#include <api/coModule.h>
#include <covise/covise.h>
#else
#include <comm/logic/coSocketListener.h>
#endif
#include <net/covise_connect.h>

/**
 * Base Class for simulation applications
 *
 */
namespace covise
{

class command_object
{

public:
    command_object(int type, char *port = 0, char *name = 0, char *data = 0, int length = 0, int numComp = 0, int actNode = 0)
        : _type(type)
        , _port(port)
        , _name(name)
        , _data(data)
        , _length(length)
        , _numComp(numComp)
        , _actNode(actNode)
    {
    }

    ~command_object()
    {
        if (_port)
            delete[] _port;
        if (_name)
            delete[] _name;
        if (_data)
            delete[] _data;
    }

    // private:
    int _type;
    char *_port, *_name, *_data;
    int _length, _numComp, _actNode;
};

#ifndef YAC
class APIEXPORT coSimLib : public coModule
#else
class APIEXPORT coSimLib : public coSimpleModule, public coSocketListener
#endif
{
public:
    typedef int int32;

// execute commands received from the simulation in sockData()
#ifndef YAC
    void executeCommands();
#else
    void executeCommands(std::set<coOutputPort *> *);
#endif

    // overload coModule sockData routine and make it private
    virtual void sockData(int sockNo);

private:
    // ---------- Internal member functions ----------------------------

    /// lookup nameserver and return IP, 0 on failure
    uint32_t nslookup(const char *name);

    // Open a TCP server
    int openServer();

    // Open a TCP client
    int openClient();

    // Accept a TCP server
    int acceptServer(float wait);

    // handle all Commands
    int handleCommand(/*int fromWhere*/);

    // ---------- Class data -------------------------------------------

    // number of user-defined arguments in call and contents
    char *d_userArg[10];

    // IP number of local and target machine, min/max port number
    uint32_t d_localIP, d_targetIP, d_minPort, d_maxPort, d_usePort;

    // the socket the simulation connects to if COVISE is the server
    int server_socket;

    // Socket number =1 on error or not-yet-open
    int d_socket;

    // Verbose level
    int d_verbose;

    // do we have to byteswap incoming data ?
    bool d_byteswap;

    // the name of the simulation
    char *d_name;

    // active command: if a command is in the 'queue': sent for server, recv in main_loop
    int32 d_command;

    // mapping for parallel data : only ONE mapping right now
    // make better it with list and per-port mapping later!
    int32 **d_cellMap, d_numCells;
    int32 **d_vertMap, d_numVert;
    int d_numNodes;
    struct PortListElem
    {
        const char *name; // name of the port
        int ***map; // mapping for this port: either cell- or vertex
        int numParts; // number of parts collected so far
        coDistributedObject *openObj; // open Object to write into

        PortListElem *next; // chain...

    } *d_portList;

    // when going parallel: active node number. If not: -1;
    int32 d_actNode;

    // all the startup lines we have
    char **d_startup_label;
    const char **d_startup_line;

    // and how many we have
    int d_numStartup;

    // Flag: The simulation requested an EXEC. Cleared by
    //       - C'tor
    //       - startSim
    //       - serverMode
    //       - resetSim
    int d_simExec;

    // list of command objects that were sent during sockData
    // the two pointers are swapped when the simulation sent COMM_QUIT
    std::list<command_object *> *command_objects;
    std::list<command_object *> *tmp_objects;

    // called from sockData when COMM_QUIT command is received
    virtual int endIteration();

    void closeSocket(int socket);

#ifdef YAC
    coCommunicator *comm;
    std::set<coOutputPort *> portList;
#endif
protected:
#include "coSimLibComm.h"

    /// Copy-Constructor: NOT  IMPLEMENTED
    coSimLib(const coSimLib &);

    /// Assignment operator: NOT  IMPLEMENTED
    coSimLib &operator=(const coSimLib &);

    /// Default constructor: NOT  IMPLEMENTED
    coSimLib();

    // this allows to choose between the different start-ups
    coChoiceParam *p_StartupSwitch;

    int reAccept();

public:
    /// Constructor: give module name, so we can read config file
    coSimLib(int argc, char *argv[], const char *moduleName, const char *desc);

    /// Destructor
    virtual ~coSimLib();

    /// set target/local host: return -1 on error (e.g. not found in nameserver)
    int setTargetHost(const char *hostname);
    int setLocalHost(const char *hostname);

    /// set a user startup argument: 0 <= num < 63
    int setUserArg(int num, const char *data);

    /// start the user's application and connect to it
    int startSim(int reattach = 0);

    /// start the remote control server: 0=normal termination, -1 on error
    //int serverMode();

    /// reset everything that didn't come from covise.config
    /// User must take care that the simulation is killed someway!
    void resetSimLib();

    /// are we connected to a simulation ?
    int isConnected();

    /// receive and send data binary: do only if you are sure...
    int recvData(void *buffer, size_t length);
    int sendData(const void *buffer, size_t length);

    /// receive and send data binary: do the same, but byte-swap if necessary
    /// length is still in bytes, %4 must be 0
    int recvBS_Data(void *buffer, size_t length);
    int sendBS_Data(void *buffer, size_t length);

    /// initialize Parallelism: local COPAIN
    int coParallelInit(int numParts, int numPorts);

    /// initialize Parallel ports: local COPAPO
    int coParallelPort(const char *portName, int isCellData);

    /// set the mapping from the module: takes over map, do not delete!!
    // i.e. local COPCM or COPAVM
    int setParaMap(int isCell, int isFortran, int nodeNo, int length,
                   int32 *nodeMap);

    /// set verbose level
    void setVerbose(int level)
    {
        d_verbose = level;
    }

    /// request verbose level
    int getVerboseLevel()
    {
        return d_verbose;
    }

    /// get the startup choice
    coChoiceParam *getStartupChoice()
    {
        return p_StartupSwitch;
    }

    /// check whether simulation requested exec
    int simRequestExec()
    {
        return d_simExec;
    }

    void setPorts(int min, int max);
};
}
#endif
