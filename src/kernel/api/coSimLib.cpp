/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WIN32
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#else
#include <winsock2.h>
#include <WS2tcpip.h>
#include <io.h>
#endif
#include <util/common.h>
#include <assert.h>
#include <sys/types.h>

#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>

#ifdef __APPLE__
#include <crt_externs.h>
#endif

#include "coSimLib.h"
#include <config/CoviseConfig.h>
#include <do/coDoData.h>

#include <covise/covise_process.h>

#ifdef __ia64
extern char **environ;
#endif

using namespace covise;

/// execute all commands that were sent by the simulation during
/// sockdata. this method may only be called during compute()
void coSimLib::executeCommands()
{

    while (command_objects->size() > 0)
    {
        command_object *o = command_objects->front();
        command_objects->pop_front();

        PortListElem *port = NULL;
        if (d_numNodes)
        {
            // find the port's record
            port = d_portList->next; // skip dummy
            while (port && (strcmp(port->name, o->_port)))
                port = port->next;
        }

        coUifElem *portElem = findElem(o->_port);
        coOutputPort *outPort = dynamic_cast<coOutputPort *>(portElem);

        if (!outPort)
        {
            fprintf(stderr, "port [%s] is not an output port\n", o->_port);
            delete o;
            continue;
        }

        switch (o->_type)
        {

        case ATTRIBUTE:
        {
            coDistributedObject *obj = outPort->getCurrentObject();

            if (obj)
                obj->addAttribute(o->_name, o->_data);
            else
            {
                sendWarning("Simulation sent attribs for '%s': no object at port", o->_port);
                break;
            }
            break;
        }

        case SEND_1DATA:
        case SEND_3DATA:
        {

            int numComp = (o->_type == SEND_1DATA) ? 1 : 3;

            if (port)
            {
                int *global = (*(port->map))[o->_actNode];
                if (!global)
                {
                    sendError("No translation map for node found");
                    delete o;
                    continue;
                }
                int globLen = (port->map == &d_cellMap) ? d_numCells : d_numVert;
                float *dataPtr[3];

                coObjInfo name;
                if (!port->openObj)
                    name = outPort->getNewObjectInfo();

                if (numComp == 1)
                {

                    coDoFloat *data;
                    if (!port->openObj)
                    {
                        data = new coDoFloat(name, globLen);
                    }
                    else
                        data = (coDoFloat *)port->openObj;

                    data->getAddress(&dataPtr[0]);
                    port->openObj = data;
                    outPort->setCurrentObject(data);
                }
                else if (numComp == 3)
                {

                    coDoVec3 *data;
                    if (!port->openObj)
                    {
                        data = new coDoVec3(name, globLen);
                    }
                    else
                        data = (coDoVec3 *)port->openObj;

                    data->getAddresses(&dataPtr[0], &dataPtr[1], &dataPtr[2]);
                    port->openObj = data;
                    outPort->setCurrentObject(data);
                }

                float *dPtr = (float *)o->_data;
                int fieldNo, local;

                // sort into global array
                for (fieldNo = 0; fieldNo < numComp; fieldNo++)
                {
                    for (local = 0; local < o->_length; local++)
                    {
                        // just make sure indexing table is ok...
                        if (global[local] >= globLen || global[local] < 0)
                        {
                            cerr << "@@@@@@@@ illegal: accessing field "
                                 << global[local] << " on field size "
                                 << globLen << endl;
                        }
                        else
                            dataPtr[fieldNo][global[local]] = *dPtr;
                        dPtr++;
                    }
                }
            }
            else
            {
                coDistributedObject *distrObj;
                float *dataPtr[3];
                // create new data object
                coObjInfo name = outPort->getNewObjectInfo();

                if (numComp == 1)
                {
                    coDoFloat *data
                        = new coDoFloat(name, o->_length);
                    data->getAddress(&dataPtr[0]);
                    distrObj = data;
                }
                else
                {
                    coDoVec3 *data
                        = new coDoVec3(name, o->_length);
                    data->getAddresses(&dataPtr[0], &dataPtr[1], &dataPtr[2]);
                    distrObj = data;
                }
                for (int i = 0; i < numComp; i++)
                    memcpy(dataPtr[i], &o->_data[o->_length * sizeof(float) * i], o->_length * sizeof(float));

                outPort->setCurrentObject(distrObj);
            }
        }
        break;

        default:
            cerr << "wrong command in executeCommands()" << endl;
        }

        delete o;
    }
}

////////////////////////////////////////////////////////////////////
/// reset all member fields for startup and re-start
void coSimLib::resetSimLib()
{

    if (d_socket > 1)
    {
        closeSocket(d_socket);
    }
    d_socket = -1;

    // if we had user args : erase it
    int i;
    for (i = 0; i < 10; i++)
    {
        delete[] d_userArg[i];
        d_userArg[i] = NULL;
    }

    // no command pending
    d_command = 0;

    // Parallel distribution Maps
    if (d_cellMap)
    {
        for (i = 0; i < d_numNodes; i++)
            delete[] d_cellMap[i];
        delete[] d_cellMap;
        d_cellMap = NULL;
    }
    d_numCells = 0;

    if (d_vertMap)
    {
        for (i = 0; i < d_numNodes; i++)
            delete[] d_vertMap[i];
        delete[] d_vertMap;
        d_vertMap = NULL;
    }
    d_numVert = 0;

    // we are not parallel ... ANY MORE
    d_numNodes = 0;

    // kill all members of the port list after the dummy
    PortListElem *portPtr = d_portList->next;
    while (portPtr)
    {
        PortListElem *nextPort = portPtr->next;
        delete portPtr;
        portPtr = nextPort;
    }
    d_portList->next = NULL;

    // not yet parallel - no active number
    d_actNode = -1;

    // sim hasn't requested exec yet
    d_simExec = 0;
}

////////////////////////////////////////////////////////////////////
/// constructor
coSimLib::coSimLib(int argc, char *argv[], const char *name, const char *desc)
    : coModule(argc, argv, desc)
{
    // not yet connected
    d_socket = -1;

    command_objects = new list<command_object *>;
    tmp_objects = new list<command_object *>;

    // save then name
    d_name = strcpy(new char[strlen(name) + 1], name);
    std::string configName = std::string("Module.") + name;

    int i;

    // no user-defined arguments so far
    for (i = 0; i < 10; i++)
        d_userArg[i] = NULL;

    // typically we use the default interface of the machine,
    // but user may specify a different, e.g. for routing reasons
    char buffer[128];
    d_localIP = 0;

    bool exists;
    std::string confData = coCoviseConfig::getEntry(configName + ".Local", &exists);

    if (exists)
    {
        printf("------------ using [%s]\n", confData.c_str());
        d_localIP = nslookup(confData.c_str());
        if (!d_localIP)
            cerr << "did not find host '" << confData
                 << "'specified in covise.config, using hostname instead"
                 << endl;
    }

    // either no LOCAL string or host not found
    if (!d_localIP)
    {
        gethostname(buffer, 128);
        buffer[127] = '\0'; // just in case... ;-)
        confData = buffer;
        d_localIP = nslookup(buffer);
    }

    // still nothing fond -> misconfigured system
    if (!d_localIP)
    {
        cerr << "Mis-configured system, could not find IP of '"
             << buffer << "', the configured hostname,"
             << endl;
    }

    // and, if user doesn't change it, we start the simulation locally
    d_targetIP = d_localIP;

    if (d_localIP == 0)
    {
        d_socket = 1;
        return;
    }

    // get port numbers

    std::string ports = coCoviseConfig::getEntry(configName + ".Ports", &exists);

    if (!exists)
    {
        d_minPort = 31500;
        d_maxPort = 31600;
    }
    else
    {
        size_t retval = sscanf(ports.c_str(), "%d %d", &d_minPort, &d_maxPort);
        if (retval != 2)
        {
            std::cerr << "coSimLib::coSimLib: sscanf failed " << std::endl;
            d_minPort = 31500; // default covise ports
            d_maxPort = 31600;
        }
    }

    //fprintf(stderr, "coSimLib::coSimLib: ports [%d %d]\n", d_minPort, d_maxPort);

    // no command pending
    d_command = 0;

    // we are not parallel ... YET
    d_numNodes = 0;

    // but we declare a dummy port for easier searching
    static const char *empty = "";
    d_portList = new PortListElem;
    d_portList->name = empty;
    d_portList->map = NULL;
    d_portList->numParts = 0;
    d_portList->openObj = NULL;
    d_portList->next = NULL;

    // we do not know yet, how big the fields will become
    d_numCells = d_numVert = 0;

    // analyse all STARTUP lines
    const char **entry;
    const char **ePtr;

    coCoviseConfig::ScopeEntries e = coCoviseConfig::getScopeEntries(configName, "Startup");
    entry = e.getValue();

    ePtr = entry;
    d_numStartup = 0;
    while (ePtr && *ePtr)
    {
        d_numStartup++;
        ePtr++;
    }
    d_numStartup /= 2;

    // copy all STARTUP lines into buffers
    if (d_numStartup)
    {
        d_startup_line = new const char *[d_numStartup];
        d_startup_label = new char *[d_numStartup];
        for (i = 0; i < d_numStartup; i++)
        {
            // skip leading blanks in field
            const char *actEntry = entry[2 * i + 1];
            while (*actEntry && isspace(*actEntry))
                actEntry++;

            // copy complete string to names field
            d_startup_label[i] = new char[strlen(actEntry) + 1];
            strcpy(d_startup_label[i], actEntry);
            char *cPtr = d_startup_label[i];

            // find next space
            while (*cPtr && !isspace(*cPtr))
                cPtr++;

            // terminate first string here: if this is only a label,
            // correctly terminate the 'line' string
            if (*cPtr)
            {
                *cPtr = '\0';
                cPtr++;
                while (*cPtr && isspace(*cPtr)) // skip spaces
                    cPtr++;
                d_startup_line[i] = cPtr;
            }
            else
            {
                cerr << "only label, but no text in startup line:\n    \""
                     << entry[2 * i + 1] << '"' << endl;
                d_startup_line[i] = actEntry;
            }
        }
    }
    else
    {
        static const char *dummy[1] = { "echo \"startup sequence not specified\"" };

        d_startup_label = new char *[1];
        d_startup_label[0] = strcpy(new char[9], "no label");
        d_startup_line = dummy;
        d_numStartup = 1;
    }

    p_StartupSwitch = addChoiceParam("Startup", "Switch startup messages");
    p_StartupSwitch->setValue(d_numStartup, d_startup_label, 0);

    // sim hasn't requested exec yet
    d_simExec = 0;
}

// destructor
coSimLib::~coSimLib()
{
    int i;
    for (i = 0; i < 10; i++)
        delete[] d_userArg[i];

    delete d_name;
}

// start the user's application
int coSimLib::startSim(int reattach)
{
    // sim hasn't requested exec yet
    d_simExec = 0;

    // Check, who is server: default is Module
    bool exists;
    std::string confData = coCoviseConfig::getEntry("Module." + std::string(d_name) + ".Server", &exists);
    int modIsServer = (!exists || *confData.c_str() == 'M' || *confData.c_str() == 'm');

    // now get the timeout, default = 1min
    float timeout = 60.0;
    confData = coCoviseConfig::getEntry("Module." + std::string(d_name) + ".Timeout");
    sendInfo("Simlib timeout is %s seconds. Can be increased by setting Module.%s.Timeout in your Covise config.", confData.c_str(), d_name);
    if (confData != "")
    {
        size_t retval;
        retval = sscanf(confData.c_str(), "%f", &timeout);
        if (retval != 1)
        {
            std::cerr << "coSimLib::startSim: sscanf failed" << std::endl;
            return -1;
        }
        //cerr << " Timeout: " << timeout << endl;
    }

    // get the verbose level, default = 0
    d_verbose = 0;
    confData = coCoviseConfig::getEntry("Module." + std::string(d_name) + ".Verbose");

    if (confData != "")
    {
        size_t retval;
        retval = sscanf(confData.c_str(), "%d", &d_verbose);
        if (retval != 1)
        {
            std::cerr << "coSimLib::startSim: sscanf failed" << std::endl;
            return -1;
        }
        cerr << " Verbose: level=" << d_verbose << endl;
        switch (d_verbose)
        {
        case 4:
            cerr << "VERBOSE  - Log all binary read/write" << endl;
        case 3:
            cerr << "VERBOSE  - Log data creation details" << endl;
        case 2:
            cerr << "VERBOSE  - Write Mapping files" << endl;
        case 1:
            cerr << "VERBOSE  - Protocol Byteswapping, Object creations, Port requests" << endl;
        }
    }

    // if we are a server, start the server
    if (modIsServer)
        if (openServer())
        {
            d_socket = -1;
            return -1;
        }

    // if we don't have the start line, forget it...
    // const char *configLine=d_config->get_scope_entry(d_name,"STARTUP");
    const char *configLine = d_startup_line[p_StartupSwitch->getValue()];
    if (!configLine)
    {
        sendError("Could not find section %s with STARTUP line in covise.config",
                  d_name);
        d_socket = -1;
        return -1;
    }

    // we need an additional '\0' after the termination: makes treating
    // a '%' as the last char easier
    char *startLine = new char[strlen(configLine) + 2];
    strcpy(startLine, configLine);
    startLine[strlen(configLine) + 1] = '\0';

    if (d_verbose > 0)
        cerr << "Startup Line: " << startLine << endl;

    // build the CO_SIMLIB_CONN variable
    char envVar[128];
    char buf[100];
    if (modIsServer)
    {
        inet_ntop(AF_INET, &d_localIP, buf, 100);
        sprintf(envVar, "C:%s/%d_%f_%d", buf,d_usePort, timeout, d_verbose);
    }
    else
    {
        sprintf(envVar, "S:%d-%d_%f_%d", d_minPort, d_maxPort, timeout, d_verbose);
    }
    
    printf("CO_SIMLIB_CONN: [%s]\n", envVar);

    // now: build the command line
    char command[4096];
    memset(command, 0, sizeof(command));
#ifndef _WIN32
    strcpy(command, "( ");
#endif
    char *startPtr = startLine;

    char *nextTok = strchr(startLine, '%');
    while (nextTok)
    {
        //cerr << "NextTok = " << nextTok << endl;
        // copy everything before the '%'
        *nextTok = '\0';
        strcat(command, startPtr);
        nextTok++;
        switch (*nextTok)
        {
        case '%':
            strcat(command, "%");
            break;
        case 'e':
            strcat(command, envVar);
            break;
        case 'h':
		    {
			    char buf[100];
			    inet_ntop(AF_INET, &d_targetIP, buf, 100);
			    strcat(command, buf);
		    }
            break;
        case '0':
            if (d_userArg[0])
                strcat(command, d_userArg[0]);
            break;
        case '1':
            if (d_userArg[1])
                strcat(command, d_userArg[1]);
            break;
        case '2':
            if (d_userArg[2])
                strcat(command, d_userArg[2]);
            break;
        case '3':
            if (d_userArg[3])
                strcat(command, d_userArg[3]);
            break;
        case '4':
            if (d_userArg[4])
                strcat(command, d_userArg[4]);
            break;
        case '5':
            if (d_userArg[5])
                strcat(command, d_userArg[5]);
            break;
        case '6':
            if (d_userArg[6])
                strcat(command, d_userArg[6]);
            break;
        case '7':
            if (d_userArg[7])
                strcat(command, d_userArg[7]);
            break;
        case '8':
            if (d_userArg[8])
                strcat(command, d_userArg[8]);
            break;
        case '9':
            if (d_userArg[9])
                strcat(command, d_userArg[9]);
            break;
        }

        startPtr = nextTok + 1;
        nextTok = strchr(startPtr, '%');
    }

    // copy the rest
    strcat(command, startPtr);

// the line is nearly ready, just make sure we start into background
#ifndef _WIN32
    strcat(command, " ) &");
#endif

    sendInfo("Starting simulation: '%s'", command);
    delete[] startLine;

    printf("command: [%s]\n", command);

    if (!reattach &&
#ifndef _WIN32
        ((!strncmp(command, "( rdaemon", 9)) ||
#else
        ((!strncmp(command, "rdaemon", 7)) ||
#endif
         (!strncmp(command, "WMI", 3))))
    {
        // Remote Daemon and WMI startup
        char *execcommand;

        char *rd_command;
        char *dir;
        char *hostname;
        char *user;
        char *passwd = (char *)"NONE";
        char *connstring;

        char *exec_string = new char[strlen(command) + 1];
        strcpy(exec_string, command);

        //fprintf(stderr,"exec_string=%s\n",exec_string);

        //DEBUG
        cerr << "******** exec_string=[" << exec_string << "]" << endl;

        (void)strtok(NULL, "\"");
        rd_command = strtok(NULL, "\"");
        (void)strtok(NULL, "\"");
        dir = strtok(NULL, "\"");
        (void)strtok(NULL, "\"");
        hostname = strtok(NULL, "\"");
        (void)strtok(NULL, "\"");
        user = strtok(NULL, "\"");
        connstring = strtok(NULL, " ");

        if (connstring == NULL)
        {
            fprintf(stderr, "ERROR! connstring is NULL!\n");
            return (-1);
        }

        execcommand = new char[strlen(rd_command) + strlen(connstring) + 6];
        strcpy(execcommand, rd_command);

        strcat(execcommand, " \"");
        strcat(execcommand, connstring);
        strcat(execcommand, "\" ");

        fprintf(stderr, "\nDEBUG output!\n");
        fprintf(stderr, "rd_command='%s'\n", rd_command);
        fprintf(stderr, "dir='%s'\n", dir);
        fprintf(stderr, "hostname='%s'\n", hostname);
        fprintf(stderr, "user='%s'\n", user);
        fprintf(stderr, "passwd='%s'\n", passwd);
        fprintf(stderr, "connstring='%s'\n", connstring);

#ifndef _WIN32
        if (!strncmp(command, "( rdaemon", 9))
#else
        if (!strncmp(command, "rdaemon", 7))
#endif
        {
            Host *rdHost = new Host((const char *)hostname);
            int remport = coCoviseConfig::getInt("port", "RemoteDaemon.Server", 31090);
            // verify myID value
            SimpleClientConnection *clientConn = new SimpleClientConnection(rdHost, remport);
            if (!clientConn)
            {
                cerr << "Creation of ClientConnection failed!" << endl;
                return -1;
            }
            else
            {
                cerr << "ClientConnection created!" << endl;
            }
            if (!(clientConn->is_connected()))
            {
                cerr << "Connection to RemoteDaemon on " << hostname << " failed!" << endl;
                return -1;
            }
            else
            {
                cerr << "Connection to RemoteDaemon on " << hostname << " established!" << endl;
            }

            // create command to send to remote daemon
            //startFEN + Parameter
            char rdcommand[1000];
            sprintf(rdcommand, "startFEN %s\n", execcommand);

            cerr << "Sending RemoteDaemon the message: " << rdcommand << endl;

            clientConn->getSocket()->write(rdcommand, (int)strlen(rdcommand));

            cerr << "Message sent!" << endl;
            cerr << "Closing connection objects!" << endl;

            delete rdHost;
            delete clientConn;

            cerr << "Leaving Start-Method of coVRSlave " << endl;
        }

#ifdef _WIN32
        else if (!strncmp(command, "WMI", 3))
        {
            // execProcessWMI: command, workingdirectory, host, user, password
            int ret = execProcessWMI(execcommand, NULL, hostname, user, NULL);
            if (ret < 0)
            {
                return ret;
            }
        }
#endif
    }

#ifndef _WIN32
    else if (!reattach)
    {

        int pid = fork();

        if (!pid)
        {
            // child process
            char *comm[4];
            comm[0] = (char *)"/bin/sh";
            comm[1] = (char *)"-c";
            comm[2] = command;
            comm[3] = 0;

            if (server_socket)
                close(server_socket);
            if (d_socket)
                close(d_socket);

#ifdef __APPLE__
            execve("/bin/sh", comm, *_NSGetEnviron());
#else
            execve("/bin/sh", comm, environ);
#endif
        }
    }
#else
    if (!reattach)
    {
        if (!strncmp(command, "local ", 6))
        {
            char *tmpstr = new char[strlen(command) + 1];
            strcpy(tmpstr, command + 6);
            strcpy(command, tmpstr);
        }

        unsigned int retval = WinExec(command, SW_SHOWNORMAL);

        if (retval > 31)
        {
            printf("WinExec(%s) - Done!\n", command);
        }
        else
        {
            switch (retval)
            {
            case ERROR_BAD_FORMAT:
                printf("WinExec() - bad format!\n");
                break;
            case ERROR_FILE_NOT_FOUND:
                printf("WinExec() - file not found!\n");
                break;
            case ERROR_PATH_NOT_FOUND:
                printf("WinExec() - path not found!\n");
                break;
            default:
                printf("WinExec() - failed!\n");
            }
        }
    }
#endif

    // if we are a server, accept, else start the client connection
    if (modIsServer)
    {
        if (acceptServer(timeout))
        {
            d_socket = -1;
            return -1;
        }
    }
    else
    {
        if (openClient())
        {
            d_socket = 1;
            return -1;
        }
    }

    // nothing works in coSimlib so far if this is not correct
    assert(sizeof(float) == 4);

    // now handshake: the other side will send us an integer 12345
    // it must be either 12345 or byte-swapped 12345
    int32 handshake, size;
    size = recvData(&handshake, sizeof(handshake));
    if (size != sizeof(handshake))
    {
        sendError("Simulation socket closed");
#ifdef _WIN32
        closesocket(d_socket);
#else
        ::close(d_socket);
#endif
        d_command = 0;
        d_socket = -1;
        return -1;
    }
    if (handshake == 12345)
    {
        d_byteswap = false;
        if (d_verbose > 0)
            cerr << "NOT perfoming byte-swapping" << endl;
    }
    else
    {
        byteSwap(handshake);
        if (handshake == 12345)
        {
            d_byteswap = true;
            if (d_verbose > 0)
                cerr << "BYTE-SWAP necessary" << endl;
        }
        else
        {
            sendError("Startup handshake failed - using old version of SimLib?");
#ifdef _WIN32
            closesocket(d_socket);
#else
            ::close(d_socket);
#endif
            d_socket = -1;
            return -1;
        }
    }

    // now: this port at coModule
    addSocket(d_socket);
    return 0;
}

/// set target host: return -1 on error
int coSimLib::setTargetHost(const char *hostname)
{
    d_targetIP = nslookup(hostname);
    if (d_targetIP)
        return 0;
    else
        return -1;
}

/// set local host: return -1 on error
int coSimLib::setLocalHost(const char *hostname)
{
    d_localIP = nslookup(hostname);
    if (d_localIP)
        return 0;
    else
        return -1;
}

/// set a user startup argument
int coSimLib::setUserArg(int num, const char *data)
{
    // max. 10 user arguments
    if ((num < 0) || (num > 9) || (!data))
        return -1;

    // delete old and set new arg
    delete[] d_userArg[num];
    d_userArg[num] = strcpy(new char[strlen(data) + 1], data);

    if (d_verbose > 0)
        cerr << "Set user[" << num << "] = '" << data << "'" << endl;

    return 0;
}

////////////////////////////////////////////////////////////////////////
//// Utility functions
////////////////////////////////////////////////////////////////////////

///// Nameserver request

#ifdef _CRAY
#define CONV (char *)
#else
#define CONV
#endif

uint32_t coSimLib::nslookup(const char *name)
{
    // try whether this is already ###.###.###.### IP address
    unsigned long addr = 0;
	inet_pton(AF_INET, name, &addr);

    if (addr && addr != INADDR_NONE)
        return (uint32_t)addr;

    // not yet a numerical address, try a nameserver look-up

	struct addrinfo hints, *result = NULL;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family = AF_INET;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = 0; /* any type of socket */
	//hints.ai_flags = AI_ADDRCONFIG; // this prevents localhost from being resolved if no network is connected on windows
	hints.ai_protocol = 0;          /* Any protocol */
	int s = getaddrinfo(name, NULL /* service */, &hints, &result);
	if (s != 0)
	{
		fprintf(stderr, "Host::HostSymbolic: getaddrinfo failed for %s: %s\n", name, gai_strerror(s));
		sendError("Could find IP address for hostname '%s'", name);
		return 0;
	}
	else
	{

		for (struct addrinfo *rp = result; rp != NULL; rp = rp->ai_next)
		{
			if (rp->ai_family != AF_INET)
				continue;

			struct sockaddr_in *saddr = reinterpret_cast<struct sockaddr_in *>(rp->ai_addr);

			freeaddrinfo(result);           /* No longer needed */
			return ntohl(saddr->sin_addr.s_addr);
		}

		freeaddrinfo(result);           /* No longer needed */
	}

	sendError("Could find IP address for hostname '%s'", name);

    return 0;
}

///// Open a TCP server

int coSimLib::openServer()
{
    unsigned int port;

    covise::Socket::initialize();

    // open the socket: if not possible, return -1
    server_socket = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0)
    {
        server_socket = -1;
        return -1;
    }

    // Find a port to start with
    port = d_minPort;

    // Assign an address to this socket
    struct sockaddr_in addr_in;
    memset((char *)&addr_in, 0, sizeof(addr_in));
    addr_in.sin_family = AF_INET;

    addr_in.sin_addr.s_addr = INADDR_ANY;
    addr_in.sin_port = htons(port);

    fprintf(stderr, "trying to connect (%d %d)\n", d_minPort, d_maxPort);

    // bind with changing port# until unused port found

    int success = 0, numTries = 0;

    while (!success && numTries < 10)
    {

        for (port = d_minPort; port <= d_maxPort; port++)
        {
            addr_in.sin_port = htons(port);

            if (!bind(server_socket, (sockaddr *)(void *)&addr_in, sizeof(addr_in)))
            {
                success = 1;
                break;
            }

#ifndef _WIN32
            if (errno == EADDRINUSE) // if port is used (UNIX)
#else
            if (GetLastError() == WSAEADDRINUSE) //                 (WIN32)
#endif
                fprintf(stderr, "port %d already in use\n", port);
        }
        if (!success)
        {
            numTries++;
            sleep(1);
        }
    }

    if (success)
    {

        fprintf(stderr, "coSimLib::openServer listening on port %d\n", port);
        ::listen(server_socket, 1); // start listening
        d_usePort = port;
        return 0;
    }
    else
    {
        fprintf(stderr, "coSimLib::openServer failed\n");
#ifdef _WIN32
        closesocket(server_socket);
#else
        ::close(server_socket);
#endif
        server_socket = -1;
        return -1;
    }
}

///// Open a TCP client ////////////////////////////////////////////////////

int coSimLib::openClient()
{
    //sleep(7);
    int connectStatus = 0;
    int numConnectTries = 0;
    unsigned int port = d_minPort;
	struct in_addr ip;
	inet_pton(AF_INET, "127.0.0.1", &ip.s_addr);
    do
    {
        fprintf(stderr, "coSimLib: trying port %d\n", port);
        // open the socket: if not possible, return -1
        d_socket = (int)socket(AF_INET, SOCK_STREAM, 0);
        if (d_socket < 0)
        {
            d_socket = -1;
            return -1;
        }

        // set s_addr structure
        struct sockaddr_in s_addr_in;
        s_addr_in.sin_addr.s_addr = htonl(ip.s_addr);
        s_addr_in.sin_port = htons(port); // network byte order
        s_addr_in.sin_family = AF_INET;

        // Try connecting
        connectStatus = connect(d_socket, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in));

        // didn't connect
        if (connectStatus < 0) // -> next port
        {
            port++;
            if (port > d_maxPort) // last Port failed -> wait & start over
            {
                port = d_minPort;
                numConnectTries++;
                sleep(2);
            }
        }
    }
    // try 10 rounds
    while ((connectStatus < 0) && (numConnectTries <= 10));

    if (connectStatus == 0)
    {
        return 0;
    }
    else
    {
        d_socket = 1;
        return -1;
    }
}

int coSimLib::acceptServer(float wait)
{
	int tmp_soc;
	struct timeval timeout;
	fd_set fdread;
	int i;

	// prepare for select(2) call and wait for incoming connection
	timeout.tv_sec = (int)wait;
	timeout.tv_usec = (int)((wait - timeout.tv_sec) * 1000000);
	FD_ZERO(&fdread);
	FD_SET(server_socket, &fdread);
	if (wait >= 0) // wait period was specified
		i = select(server_socket + 1, &fdread, NULL, NULL, &timeout);
	else // wait infinitly
		i = select(server_socket + 1, &fdread, NULL, NULL, NULL);

	if (i == 0) // nothing happened: return -1
		return -1;

	// now accepting the connection
	struct sockaddr_in s_addr_in;

#ifdef _WIN32
	tmp_soc = (int)accept(server_socket, (sockaddr *)(void *)&s_addr_in, NULL);
#else
	socklen_t length = sizeof(s_addr_in);
	tmp_soc = accept(server_socket, (sockaddr *)(void *)&s_addr_in, &length);
#endif

	if (tmp_soc < 0)
		return -1;

	// use the socket 'accept' delivered
	d_socket = tmp_soc;

	if (d_verbose > 0)
	{
		char buf[100];
		inet_ntop(AF_INET, &s_addr_in.sin_addr, buf, 100);
		cerr << "Accepted connection from "
			<< buf
			<< " to socket " << d_socket
			<< endl;
	}
    return 0;
}

/***************************************************
 * Send a certain amount of data to the simulation *
 ***************************************************/
int coSimLib::sendData(const void *buffer, size_t _length)
{
    unsigned long length = (unsigned long)_length; // make 64-bit proof printf
    char *bptr = (char *)buffer;
    int written;
    int nbytes = length;
    if (d_verbose > 3)
        fprintf(stderr, "coSimLib sending %ld Bytes to Socket %d\n",
                length, d_socket);

    while (nbytes > 0)
    {
        do
        {
#ifdef _WIN32
            written = ::send(d_socket, (char *)bptr, nbytes, 0);
#else
            written = write(d_socket, (void *)bptr, nbytes);
#endif
#ifdef _WIN32
        } while ((written <= 0) && ((WSAGetLastError() == WSAEINPROGRESS) || (WSAGetLastError() == WSAEINTR) || (WSAGetLastError() == WSAEWOULDBLOCK)));
#else
        } while ((written <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
#endif
        if (written < 0)
        {
#ifdef _WIN32
            fprintf(stderr, "coSimLib error: write returned %d, %d\n", written, WSAGetLastError());
#else
            fprintf(stderr, "coSimLib error: write returned %d: %s\n", written, strerror(errno));
#endif
            return -1;
        }
        nbytes -= written;
        bptr += written;
        if (written == 0)
            return -2;
    }
    if (d_verbose > 3)
        fprintf(stderr, "coSimLib sent %ld Bytes\n", length);
    return length;
}

/********************************************************
 * Receive a certain amount of data from the simulation *
 ********************************************************/

int coSimLib::recvData(void *buffer, size_t _length)
{
    unsigned long length = (unsigned long)_length; // make 64-bit proof printf
    char *bptr = (char *)buffer;
    int nread;
    int nbytes = length;

    if (d_verbose > 3)
        fprintf(stderr, " coSimLib waiting for %ld Bytes from Socket %d\n",
                length, d_socket);

    while (nbytes > 0)
    {
        do
        {
#ifdef _WIN32
            nread = ::recv(d_socket, (char *)bptr, nbytes, 0);
#else
            nread = read(d_socket, (void *)bptr, nbytes);
#endif
#ifdef _WIN32
        } while ((nread <= 0) && ((WSAGetLastError() == WSAEINPROGRESS) || (WSAGetLastError() == WSAEINTR) || (WSAGetLastError() == WSAEWOULDBLOCK)));
#else
        } while ((nread <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
#endif
        if (nread < 0)
        {
#ifdef _WIN32
            fprintf(stderr, "coSimLib error: received  %d Bytes, %d\n", nread, WSAGetLastError());
#else
            fprintf(stderr, "coSimLib error: received %d Bytes: %s\n", nread, strerror(errno));
#endif
            return -1;
        }
        nbytes -= nread;
        bptr += nread;
        if (nread == 0)
            break;
    }
    if (nbytes)
    {
        fprintf(stderr, "coSimLib error: socket closed while %d left\n", nbytes);
#ifdef _WIN32
//Sleep(1);
#else
//sleep(1);
#endif
        return -2;
    }
    else
    {
        if (d_verbose > 3)
            fprintf(stderr, "coSimLib received %ld Bytes\n", length);
        return length;
    }
}

///// receive with swapping if necessary
int coSimLib::recvBS_Data(void *buffer, size_t _length)
{
    int res = recvData(buffer, _length);
    _length = res >> 2;
    if (d_byteswap)
        byteSwap((int32_t *)buffer, (int)_length);
    return res;
}

///// send with swapping if necessary
/// we swap back and forth here - expensive for large data
int coSimLib::sendBS_Data(void *buffer, size_t _length)
{
    int numInts = (int)(_length >> 2);
    if (d_byteswap)
        byteSwap((int32_t *)buffer, numInts);
    int res = sendData(buffer, _length);
    if (d_byteswap)
        byteSwap((int32_t *)buffer, numInts);
    return res;
}

///// handle events sent outside ServerMode

void coSimLib::sockData(int sockNo)
{
    // if we have another command in the queue: just return and do nothing
    if (d_command) // handle controller messages instead
    {
        Covise::check_and_handle_event(1.0f); // wait for a longer time here because we are usually waiting for an execute message from the controller otherwise we are in a busy loop until we get the controller message which kills windows
        return;
    }

    // if this is not our socket: throw message and return
    if (sockNo != d_socket)
    {
        fprintf(stderr, " our Socket: %d incoming socket: %d \n", d_socket, sockNo);
        sendError("Overloading of sockData() not allowed in coSimLib");
#ifdef WIN32
        Sleep(1000);
#endif
        return;
    }

    // receive the command ID from the socket
    int recvSize = recvBS_Data(&d_command, sizeof(d_command));

    // if we get -1 here, something went wrong
    if (recvSize < 0)
    {
        sendError("Simulation socket crashed: closed connection");
        closeSocket(d_socket);
        d_command = 0;
        d_socket = -1;
    }

    if (recvSize < sizeof(d_command))
    {
        return;
    }

    handleCommand();

    return;
}

///// handle all kinds of commands from the simulation
///// caller must check, whether we are allowed to do this NOW

///// return last command, set active command to 0
int coSimLib::handleCommand()
{
    // memorise the last command, we have to return it...
    CommandType actComm = (CommandType)d_command;
    d_command = COMM_NONE;

    /// we re-use this buffer frequently...
    char buffer[128];

    // common segments for some Command types
    switch (actComm)
    {
    // all these send a portname first...
    case GET_SLI_PARA:
    case GET_SC_PARA_FLO:
    case GET_SC_PARA_INT:
    case GET_CHOICE_PARA:
    case GET_BOOL_PARA:
    case GET_TEXT_PARA:
    case SEND_USG:
    case SEND_1DATA:
    case SEND_3DATA:
    case PARA_PORT:
    case ATTRIBUTE:
    {
        // receive Parameter name: binary data
        if (recvData((void *)buffer, 64) != 64)
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }
        break;
    }
    default:
        break;
    }

    switch (actComm)
    {
    // ###########################################################
    // COMM_TEST is always simply ignored
    // ###########################################################
    case COMM_TEST:
        if (d_verbose > 1)
        {
            cerr << "coSimLib Client called COMM_TEST" << endl;
        }
        break;

    // ###########################################################
    // EXEC sends a callback message : only set self-exec flag
    // ###########################################################
    case EXEC_COVISE:
    {
        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called EXEC" << endl;
        }
        selfExec();
        break;
    }

    case ATTRIBUTE:
    {
        char *name = new char[1024];
        char *value = new char[1024];
        //char name[1024],value[1024];

        // receive the content: binary data
        if (recvData(name, 1024) != 1024)
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            delete[] value;
            delete[] name;
            return -1;
        }
        if (recvData(value, 1024) != 1024)
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            delete[] value;
            delete[] name;
            return -1;
        }

        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called ATTRIBUTE" << endl;
            cerr << "   port: '" << buffer << "'" << endl;
            cerr << "   name: '" << name << "'" << endl;
            cerr << "   val : '" << value << "'" << endl;
        }

        command_object *o = new command_object(actComm,
                                               strdup(buffer),
                                               name,
                                               value);
        tmp_objects->push_back(o);

        break;
    }

    // ###########################################################
    // Text Parameter request
    // ###########################################################
    case GET_TEXT_PARA:
    {
        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called GET_TEXT_PARA" << endl;
        }
        // get this parameter
        char res[256];

        coUifElem *para = findElem(buffer);
        coStringParam *param = dynamic_cast<coStringParam *>(para);

        if (param)
        {
            const char *val = param->getValue();
            strncpy(res, val, 255);
            res[255] = '\0';
            if (strlen(val) > 255)
                sendWarning("coSimLib: Truncated parameter %s when sending", buffer);
        }
        else
        {
            sendWarning("coSimLib: could not find requested parameter '%s'", buffer);
        }

        // send the answer back to the client
        if (sendData(res, 256) != 256)
            return -1;
        break;
    }

    // ###########################################################
    // Slider Parameter request
    // ###########################################################
    case GET_SLI_PARA:
    {
        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called GET_SLI_PARA" << endl;
        }

        // get this parameter
        struct
        {
            float min, max, value;
            int32 error;
        } ret;

        coUifElem *para = findElem(buffer);
        coFloatSliderParam *param = dynamic_cast<coFloatSliderParam *>(para);
        if (param)
        {
            param->getValue(ret.min, ret.max, ret.value);
            ret.error = 0;
        }
        else
            ret.error = -1;

        // send the answer back to the client
        if (sendBS_Data((void *)&ret, sizeof(ret)) != sizeof(ret))
            return -1;
        break;
    }
    // ###########################################################
    // Float scalar Parameter request
    // ###########################################################
    case GET_SC_PARA_FLO:
    {
        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called GET_SC_PARA_FLO" << endl;
        }

        // get this parameter
        struct
        {
            float val;
            int32 error;
        } ret;

        coUifElem *para = findElem(buffer);
        coFloatParam *param = dynamic_cast<coFloatParam *>(para);
        if (param)
        {
            ret.val = param->getValue();
            ret.error = 0;
        }
        else
            ret.error = -1;

        // send the answer back to the client
        if (sendBS_Data((void *)&ret, sizeof(ret)) != sizeof(ret))
            return -1;
        break;
    }

    // ###########################################################
    // All these get exactly one integer, just the appearance is different
    // ###########################################################
    case GET_SC_PARA_INT:
    case GET_CHOICE_PARA:
    case GET_BOOL_PARA:
    {
        // get this parameter
        struct
        {
            int32 val;
            int32 error;
        } ret;
        coUifElem *para = findElem(buffer);
        coIntScalarParam *paraInt = dynamic_cast<coIntScalarParam *>(para);
        coChoiceParam *paraChoice = dynamic_cast<coChoiceParam *>(para);
        coBooleanParam *paraBool = dynamic_cast<coBooleanParam *>(para);

        if (paraInt && actComm == GET_SC_PARA_INT)
        {
            if (d_verbose > 0)
            {
                cerr << "coSimLib Client called GET_SC_PARA_INT" << endl;
            }
            ret.val = paraInt->getValue();
            ret.error = 0;
        }
        else if (paraChoice && actComm == GET_CHOICE_PARA)
        {
            if (d_verbose > 0)
            {
                cerr << "coSimLib Client called GET_CHOICE_PARA" << endl;
            }
            ret.val = paraChoice->getValue();
            ret.error = 0;
        }
        else if (paraBool && actComm == GET_BOOL_PARA)
        {
            if (d_verbose > 0)
            {
                cerr << "coSimLib Client called GET_BOOL_PARA " << buffer << ":" << ((coBooleanParam *)para)->getValue() << endl;
            }
            ret.val = paraBool->getValue();
            ret.error = 0;
        }
        else
            ret.error = -1;

        // send the answer back to the client
        if (sendBS_Data((void *)&ret, sizeof(ret)) != sizeof(ret))
            return -1;
        break;
    }

    // ###########################################################
    //  Client creates 1D or 3D data field
    // ###########################################################
    case SEND_1DATA:
    case SEND_3DATA:
    {
        // number of components in the data set
        int numComp = (actComm == SEND_1DATA) ? 1 : 3;

        if (d_verbose > 0)
            cerr << "coSimLib Client called SEND_" << numComp << "DATA" << endl;

        int32 length;

        // receive length
        if (recvBS_Data(&length, sizeof(int32)) != sizeof(int32))
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }

        // ###########################################################
        // parallel applications may have non-parallel data...

        PortListElem *port = NULL;
        if (d_numNodes)
        {
            // find the port's record
            port = d_portList->next; // skip dummy
            while (port && (strcmp(port->name, buffer)))
                port = port->next;
        }

        // ###########################################################
        // ok, this is really parallel : distributed data collection
        if (port)
        {

            // set our translation table to an easier pointer and check
            int *global = (*(port->map))[d_actNode];
            if (!global)
            {
                sendError("No translation map for node found");
                return -1;
            }

            if (d_verbose > 2)
            {
                cerr << "Recv Data from node " << d_actNode << endl;
                cerr << "Conversion table starts : "
                     << global[0] << "," << global[1] << "," << global[2] << endl;
            }

            // receive to dummy field
            float *dummy = new float[length * numComp];

            // read node data into temporary field
            int local = length * numComp * sizeof(float);
            if (recvBS_Data(dummy, local) != local)
            {

                sendError("Simulation socket closed");
                closeSocket(d_socket);
                d_command = 0;
                d_socket = -1;
                return -1;
            }

            if (d_verbose > 2)
                cerr << "received data: " << length << " values"
                     << dummy[0] << "," << dummy[1] << "," << dummy[2] << endl;

            command_object *o = new command_object(actComm,
                                                   strdup(buffer),
                                                   0,
                                                   (char *)dummy,
                                                   length,
                                                   numComp,
                                                   d_actNode);
            tmp_objects->push_back(o);
        }

        // ###########################################################
        // if we are not parallel, it is easy
        else
        {
            float *dummy = new float[length * numComp];

            // receive contents
            if (recvBS_Data(dummy, length * numComp * sizeof(float)) != length * numComp * sizeof(float))
            {
                sendError("Simulation socket closed");
                closeSocket(d_socket);
                d_command = 0;
                d_socket = -1;
                return -1;
            }

            command_object *o = new command_object(actComm,
                                                   strdup(buffer),
                                                   0,
                                                   (char *)dummy,
                                                   length,
                                                   numComp,
                                                   d_actNode);
            tmp_objects->push_back(o);
        }
        break;
    }

    // ###########################################################
    // Initialisation of a parallel data
    // ###########################################################
    case PARA_INIT:
    {
        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called PARA_INIT" << endl;
        }

        // the information, but type has been read !!
        struct
        {
            int32 partitions, ports;
        } data;
        if (recvBS_Data((int *)&data, sizeof(data)) != sizeof(data))
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }

        // we have got another PARA_INIT call before
        if (d_numNodes)
            sendWarning("Multi-parallelism not implemented YET: Expect coredump");

        coParallelInit(data.partitions, data.ports);

        break;
    }

    // ###########################################################
    // Definition of a parallel data port
    // ###########################################################
    case PARA_PORT: // portname is read before
    {
        if (d_verbose > 0)
        {
            cerr << "coSimLib Client called PARA_PORT" << endl;
        }

        int32 data;
        if (recvBS_Data(&data, sizeof(data)) != sizeof(data))
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }

        // buffer holds the port name
        coParallelPort(buffer, data);

        break;
    }

    // ###########################################################
    // We get a mapping: either vertex or cell
    // ###########################################################
    case PARA_CELL_MAP:
    case PARA_VERTEX_MAP:
    {
        if (d_verbose > 0)
        {
            if (actComm == PARA_CELL_MAP)
                cerr << "coSimLib Client PARA_CELL_MAP" << endl;
        }
        else
            cerr << "coSimLib Client PARA_VERTEX_MAP" << endl;

        // the information, but type has been read !!
        struct
        {
            int32 fortran, node, length;
        } data;

        if (recvBS_Data(&data, sizeof(data)) != sizeof(data))
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }

        // whether this is a valid node number
        if (data.node > d_numNodes || data.node < 0)
        {
            sendError("%s: illegal node number: %d", d_name, data.node);
            return -1;
        }

        // read binary -> client has to convert if necessary
        int32 *actMap = new int32[data.length];
        int bytes = sizeof(int32) * data.length;
        if (recvBS_Data(actMap, bytes) != bytes)
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }

        // and now set the map
        if (setParaMap((actComm == PARA_CELL_MAP), data.fortran, data.node,
                       data.length, actMap) < 0)
            return -1;

        break;
    }

    // ###########################################################
    // We get a new active node number
    // ###########################################################
    case PARA_NODE:
    {
        if (d_verbose > 0)
            cerr << "coSimLib Client PARA_NODE :";

        // recv node number
        if (recvBS_Data(&d_actNode, sizeof(d_actNode)) != sizeof(d_actNode))
        {
            sendError("Simulation socket closed");
            closeSocket(d_socket);
            d_command = 0;
            d_socket = -1;
            return -1;
        }

        if (d_verbose > 0)
            cerr << " Node = " << d_actNode << endl;

        // check
        if (d_actNode < 0 || d_actNode >= d_numNodes)
        {
            sendError("%s: illegal node number: %d", d_name, d_actNode);
            return -1;
        }

        break;
    }

    // ###########################################################
    // QUIT : exit from server mode - close all open object
    // ###########################################################
    case COMM_QUIT:
    {
        if (d_verbose > 0)
            cerr << "coSimLib Client QUIT" << endl;

        PortListElem *port = d_portList->next;
        while (port)
        {
            if (port->openObj)
            {
                port->openObj = NULL; // we don't delete it -> the port will do
            }

            port = port->next;
        }

        endIteration();

        list<command_object *> *tmp = command_objects;
        command_objects = tmp_objects;
        tmp_objects = tmp;

        break;
    }

    case COMM_EXIT:

    case COMM_DETACH:

        if (server_socket > 0)
        {
#ifdef _WIN32
            closesocket(server_socket);
#else
            ::close(server_socket);
#endif
        }
        if (d_socket > 0)
        {
            closeSocket(d_socket);
        }
        d_socket = -1;
        server_socket = -1;
        break;

    ////// default: kill the command from the Queue
    default:
    {
        d_command = COMM_NONE;
        break;
    }
    }

    // command was handled
    return actComm;
}

int coSimLib::endIteration()
{

    cerr << "Warning: you might want to overload coSimLib::endIteration()" << endl;
    return 1;
}

int coSimLib::isConnected()
{
    return (d_socket > 1);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
/////
/////   Module using client commands

///////////////////////////////////////////////////////////////////////////
/// set the mapping from the module

int coSimLib::coParallelInit(int numNodes, int numPorts)
{
    if (d_verbose > 0)
    {
        cerr << "coParallelInit( numNodes=" << numNodes << ", numPorts="
             << numPorts << " )" << endl;
    }

    d_numNodes = numNodes;

    // prepate the lists
    d_cellMap = new int *[d_numNodes];
    d_vertMap = new int *[d_numNodes];

    // set zero
    int i;
    for (i = 0; i < d_numNodes; i++)
        d_cellMap[i] = d_vertMap[i] = NULL;
    return 0;
}

///////////////////////////////////////////////////////////////////////////
/// initialize Parallel ports: local COPAPO

int coSimLib::coParallelPort(const char *portName, int isCellData)
{
    if (d_verbose > 0)
    {
        cerr << "coParallelPort( portName='" << portName << "', isCellData="
             << isCellData << " )" << endl;
    }

    // create port at the end of the list: we have a dummy at the start!
    PortListElem *port = d_portList;
    while (port->next)
        port = port->next;
    port->next = new PortListElem;
    port = port->next;

    port->name = strcpy(new char[strlen(portName) + 1], portName);
    port->openObj = NULL;
    port->numParts = 0;
    port->next = NULL;
    if (isCellData)
        port->map = &d_cellMap;
    else
        port->map = &d_vertMap;
    return 0;
}

///////////////////////////////////////////////////////////////////////////
/// set the mapping from the module

int coSimLib::setParaMap(int isCell, int isFortran, int nodeNo, int length,
                         int32 *nodeMap)
{
    if (d_verbose > 0)
    {
        cerr << "setParaMap( isCell=" << isCell
             << ", isFortran=" << isFortran
             << ", nodeNo=" << nodeNo
             << ", length=" << length
             << ", Map=(" << nodeMap[0] << "," << nodeMap[1] << "..."
             << nodeMap[length - 2] << "," << nodeMap[length - 1]
             << " )" << endl;
    }
    int i;
    char buffer[128];
    int **map = (isCell) ? d_cellMap : d_vertMap;

    // write Mappings to files
    if (d_verbose > 1)
    {
        sprintf(buffer, "coSimlib.%d.Map", nodeNo);
        FILE *outFile = fopen(buffer, "w");

        fprintf(outFile, "Node mapping file for %s(%d) for Proc #%d\n\n",
                d_name, getpid(), nodeNo);
        fprintf(outFile, "%6s ; %6s\n\n", "node", "proc");
        if (outFile)
        {
            for (i = 0; i < length; i++)
                fprintf(outFile, "%6d ; %6d\n", nodeMap[i], nodeNo);
            fclose(outFile);
        }
        else
            cerr << "ERROR: coSimLib requested output file '"
                 << buffer << "' could not be created." << endl;
    }

    // whether this is a valid node number
    if (nodeNo > d_numNodes || nodeNo < 0)
    {
        sendError("%s: illegal node number: %d", d_name, nodeNo);
        return -1;
    }

    map[nodeNo] = nodeMap;

    // decrement 1 for fortran counting
    if (isFortran)
        for (i = 0; i < length; i++)
            (nodeMap[i])--;

    // now find the max. Index
    if (isCell)
    {
        for (i = 0; i < length; i++)
            if (nodeMap[i] >= d_numCells)
                d_numCells = nodeMap[i] + 1;
    }
    else
    {
        for (i = 0; i < length; i++)
            if (nodeMap[i] >= d_numVert)
                d_numVert = nodeMap[i] + 1;
    }

    return 0;
}

void coSimLib::setPorts(int min, int max)
{

    printf("ports: %d-%d\n", min, max);
    d_minPort = min;
    d_maxPort = max;
}

int coSimLib::reAccept()
{

    closeSocket(d_socket);
    float timeout = coCoviseConfig::getFloat("Module." + std::string(d_name) + ".Timeout", 60.);

    int res = acceptServer(timeout);

    if (res == 0)
    {

        int32 handshake, size;
        size = recvData(&handshake, sizeof(handshake));
        if (size != sizeof(handshake))
        {
            sendError("Simulation socket closed");
#ifdef _WIN32
            closesocket(d_socket);
#else
            ::close(d_socket);
#endif
            d_command = 0;
            d_socket = -1;
            return -1;
        }
        if (handshake == 12345)
        {
            d_byteswap = false;
            if (d_verbose > 0)
                cerr << "NOT perfoming byte-swapping" << endl;
        }
        else
        {
            byteSwap(handshake);
            if (handshake == 12345)
            {
                d_byteswap = true;
                if (d_verbose > 0)
                    cerr << "BYTE-SWAP necessary" << endl;
            }
            else
            {
                sendError("Startup handshake failed - using old version of SimLib?");
#ifdef _WIN32
                closesocket(d_socket);
#else
                ::close(d_socket);
#endif
                d_socket = -1;
                return -1;
            }
        }
    }
    addSocket(d_socket);
    return 0;
}

void coSimLib::closeSocket(int socket)
{
#ifdef _WIN32
    closesocket(socket);
#else
    ::close(socket);
#endif
    removeSocket(socket);
}
