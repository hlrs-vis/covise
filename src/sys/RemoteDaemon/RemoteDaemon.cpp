/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**
 * @Desc: Remote daemon providing services for remote starting of covise and crb.
 * @author: Michael Braitmaier
 * @Date: 2004-04-20
 *
 */
#ifdef _WIN32
#include <winsock2.h>
#include <process.h>
#endif
#include <util/unixcompat.h>
#include <iostream>


#include <qsocketnotifier.h>
#define IOMANIPH
// don't include iomanip.h becaus it interferes with qt


#include "RemoteDaemon.h"
#include <stdio.h>
#include <string.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <covise/covise_msg.h>
#include <signal.h>

#include <stdlib.h>

#include <sysdep/net.h>
#include <sys/stat.h>

#include <config/CoviseConfig.h>

using namespace covise;
using std::cerr;
using std::endl;
using std::ios;
using std::ofstream;

static void sigChild(int)
{
    exit(0);
}

/**
 * @Desc: Main function starting the server and instanciating the server object
 *
 *
 */
int main(int argc, char **argv)
{

    if (argc > 2) // RemoteDaemon host command args
    {
        int remPort = 0;
        remPort = coCoviseConfig::getInt("port", "System.RemoteDaemon.Server", 31090);

        Host *objHost = new Host(argv[1]);
        SimpleClientConnection *clientConn = new SimpleClientConnection(objHost, remPort);
        if (!clientConn)
        {
            cerr << "Creation of ClientConnection failed!" << endl;
            return 1;
        }
        else
        {
            cerr << "ClientConnection created!" << endl;
        }
        if (!(clientConn->is_connected()))
        {
            cerr << "Connection to RemoteDaemon on " << argv[1] << " failed!" << endl;
            return 1;
        }
        else
        {
            cerr << "Connection to RemoteDaemon on " << argv[1] << " established!" << endl;
        }

        char msg[100000];
        msg[0] = '\0';
        for (int i = 2; i < argc; i++)
        {
            strcat(msg, argv[i]);
            if (i != argc - 1)
                strcat(msg, " ");
        }
        strcat(msg, "\n");
        // create command to send to remote daemon
        cerr << "Sending RemoteDaemon the message: " << msg << endl;

        clientConn->getSocket()->write(msg, (int)strlen(msg));

        cerr << "Message sent!" << endl;
        cerr << "Closing connection objects!" << endl;

        delete objHost;
        delete clientConn;

        cerr << "Leaving Start-Method of coVRSlave " << endl;
    }
    else
    {

#ifndef _WIN32
        // Daemonize me using command line arg -d
        signal(SIGHUP, SIG_IGN);
        int t;
        if ((argc > 1) && strcmp(argv[1], "-d") == 0)
        {
            t = fork();
            if (t == 0)
            {
                // Workaround to fool Covise Connection (which dies if no parent process is running)
                t = fork();
                if (t == 0)
                {
                    // Silence all console I/O
                    for (int i = 0; i < 3; ++i)
                        close(i);
                }
                else
                {
                    signal(SIGCHLD, sigChild);
                    while (true)
                        usleep(10000);
                    return 0;
                }
            }
            else
            {
                exit(0);
            }
        }
#endif
        RemoteDaemon *rd;
        cerr << "Creating server object..." << endl;
        rd = new RemoteDaemon();
        cerr << "Opening server..." << endl;
        if (rd->openServer() >= 0)
        {
            rd->loop();
        }
        cerr << "Delete server object..." << endl;
        delete rd;
    }

    return 0;
}

/**
 * @Desc: Constructor for server class
 *
 *
 */
RemoteDaemon::RemoteDaemon()
{
    m_debugFile = NULL;
    m_file = NULL;
    m_sbuf = NULL;

    if (coCoviseConfig::isOn("System.RemoteDaemon.Debug", false))
    {
        if (coCoviseConfig::isOn("System.RemoteDaemon.EnableFileDebug", false))
        {
            std::string line = coCoviseConfig::getEntry("System.RemoteDaemon.DebugFile");
            if (!line.empty())
            {
                cerr << "Value of debug-file: " << line << endl;
            }
            else
            {
                cerr << "No config entry for debugFile. Using hardcoded!" << endl;
                line = "RemoteDaemon.log";
            }
            cerr << "Used debug-file: " << line << endl;

            m_file = new ofstream();
            m_file->open(line.c_str(), ios::out);
            m_sbuf = std::cerr.rdbuf();
            std::cerr.rdbuf(m_file->rdbuf());
        }
    }
    else
    {
        cerr << "Dump to NULL stream!" << endl;
        cerr << "Use config file to enable debugging to console or file!" << endl;
        m_file = new ofstream();
#ifdef WIN32
        m_file->open("nul", ios::out);
#else
        m_file->open("/dev/null", ios::out);
#endif
        m_sbuf = std::cerr.rdbuf();
        std::cerr.rdbuf(m_file->rdbuf());
    }

    cerr << "Initialize Controller to NULL..." << endl;
    toController = NULL;

    cerr << "Initialize RemoteConnection to NULL..." << endl;
    toAG = NULL;

    port = 31090;
    cerr << "Default port " << port << "..." << endl;
    cerr << "Determine TCP-Port from Covise Config..." << endl;
    port = coCoviseConfig::getInt("port", "System.RemoteDaemon", port);
    receivedMesssages = 0;

    cerr << "TCP-Port is " << port << endl;
    cerr << "Finished constructor!" << endl;

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
}

/**
 * @Desc: Destructor for server class
 *
 *
 */
RemoteDaemon::~RemoteDaemon()
{
    printf("Delete server connection...");
    if (m_sbuf)
    {
        std::cerr.rdbuf(m_sbuf);
    }
    cerr << "Clearing memory from objects!" << endl;
    delete m_file;
    cerr << "Deleted debugFile name variable!" << endl;
    delete sConn;
    cerr << "closed Server connection" << endl;
    cerr << "Done!" << endl;
}

/**
 * @Desc: Main loop of the server class doing the message loop.
 * @return: no return value
 */
void RemoteDaemon::loop()
{
    while (processMessages())
    {
    }
}

/**
 * @Desc: Opens the server, by opening the server connection at the port
 *        defined in the constructor. Adds connection to the connection list.
 * @return: 0 per default
 */
int RemoteDaemon::openServer()
{
    cerr << "Create new server connection..." << endl;
    sConn = new SimpleServerConnection(port, 0, (sender_type)0);

    //check for valid SimpleServerConnection object

    if (!sConn->getSocket())
    {
        cerr << "Creation of server failed!" << endl;
        cerr << "Port-Binding failed! Port already bound?" << endl;
        return (-1);
    }

    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    cerr << "Set socket options..." << endl;
    setsockopt(sConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

    cerr << "Set server to listen mode..." << endl;
    sConn->listen();
    if (!sConn->is_connected()) // could not open server port
    {
        fprintf(stderr, "Could not open server port %d\n", port);
        delete sConn;
        sConn = NULL;
        return (-1);
    }
    cerr << "Add server connection to connection list..." << endl;
    connections = new ConnectionList();
    connections->add(sConn);
    cerr << "adding" << sConn << endl;
    msg = new Message;

    cerr << "Server opened!" << endl;

    return 0;
}

/**
 * @Desc: Closes the server and therefore closes the server connection
 *
 * @return: no return value
 */
void RemoteDaemon::closeServer()
{
    printf("Closing server connection...");
    connections->remove(sConn);
    cerr << "removing" << sConn;
    // tut sonst nicht (close_socket kommt nicht an)
    // das will ich doch mal probieren
    printf("Delete server connection...");
    delete sConn;
    sConn = NULL;
}

/**
 * @Desc: Processes the incoming data on the registered connections and
 *        performs appropriate message processing.
 * @return: 1 if successfull, else 0
 */
int RemoteDaemon::processMessages()
{
    Connection *conn;
    Connection *clientConn;
    const char *line;
    while ((conn = connections->check_for_input(1)))
    {
        cerr << "Input received!" << endl;
        cerr << "data on conn " << conn << endl;
        receivedMesssages++;
        if (conn == sConn) // connection to server port
        {
            cerr << "Connected to server port, spawn new connection and add to list..." << endl;
            clientConn = sConn->spawnSimpleConnection();
            //clientConn->getSocket()->write("0", 1); //confirming client that no dataformat is used
            connections->add(clientConn); //add new connection;
            cerr << "add " << clientConn << endl;
            cerr << "new connection " << endl;
        }
        else
        {
            if (conn == toController)
            {
                cerr << "Received message for existing controller!" << endl;
                if (conn->recv_msg(msg))
                {
                    if (msg)
                    {
                        handleClient(msg);
                    }
                }
            }
            else
            {
                cerr << "Received message from Client!" << endl;
                if ((line = ((SimpleServerConnection *)conn)->readLine()) != NULL)
                {
                    cerr << " Received end of line --> readLine()" << endl;
                    cerr << " Message to process: " << line << endl;
                    if (!handleClient(line, conn))
                    {
                        cerr << "Processing of the message failed!" << endl;
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}

/**
 * @Desc: Handles messages sent by the remote client application.
 * @param msg - Message as sent by the client
 * @return: no return value
 */
void RemoteDaemon::handleClient(Message *msg)
{
    TokenBuffer tb(msg);
    switch (msg->type)
    {
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    {
    }
    break;
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    {
        if (msg->conn == toController)
        {
            toController = NULL;
        }
        connections->remove(msg->conn);
        cerr << "remove" << msg->conn;
        delete msg->conn;
        if (toAG)
        {
            toAG->getSocket()->write("masterLeft", (int)strlen("masterLeft") + 1);
        }
        cerr << "controller left" << endl;
    }
    break;
    default:
        cerr << "unknown message in vrb" << endl;
        break;
    }
}

/**
 * @Desc: Prepares and performs application start of covise
 * @return: no return value
 */
void RemoteDaemon::startCovise()
{
    char cport[200];
    int sPort;
    toController = new SimpleServerConnection(&sPort, 0, (sender_type)0);
    toController->listen();
    sprintf(cport, "%d", sPort);

#ifdef _WIN32
    spawnlp(P_NOWAIT, "covise", "covise", "-a", cport, NULL);
#else
    int pid = fork();
    if (pid == 0)
    {
        execlp("covise", "covise", "-a", cport, NULL);
    }
    else
    {
        // Needed to prevent zombies
        // if childs terminate
        signal(SIGCHLD, SIG_IGN);
    }
#endif
    toController->acceptOne();
    connections->add(toController); //add new connection;
    cerr << "add" << toController;
}

/**
 * @Desc: Handles messages sent by the remote client application which do
 *		  not fit into predefined message patterns.
 * @param line - data sent to the server appliction
 * @param conn - connection on which the data was delivered
 * @return: 1 if successful, 0 else
 */
int RemoteDaemon::handleClient(const char *line, Connection *conn)
{
    cerr << line << endl;

    if (strnicmp(line, "ConnectionClosed", 16) == 0)
    {
        if (conn == toAG)
        {
            cerr << " Connection is toAG conn" << endl;
            toAG = NULL;
        }
        connections->remove(conn);
        cerr << "Received data packages: " << receivedMesssages << endl;
        receivedMesssages = 0;
        cerr << "remove" << conn << endl;
        delete conn;
    }
    else if (strnicmp(line, "quit", 4) == 0)
    {
        return 0;
    }

    else if (strnicmp(line, "check", 5) == 0)
    {
        if (toController == NULL)
        {
            conn->getSocket()->write("masterLeft", (int)strlen("masterLeft") + 1);
        }
        else
        {
            conn->getSocket()->write("masterRunning\n", (int)strlen("masterRunning\n") + 1);
        }
    }

    else if (strnicmp(line, "join", 4) == 0)
    {
        //removing special character at end of message line
        const size_t len = strlen(line);
        char *client = new char[len];
        strncpy(client, line, len-1);
        client[len-1] = '\0';

        cerr << "Line is: " << client << "!" << endl;
        cerr << "Length of line is : " << strlen(client) << endl;

        if (toController == NULL)
        {
            toAG = conn;
            startCovise();
        }
        Message msg;
        msg.type = COVISE_MESSAGE_ACCESSGRID_DAEMON;
        msg.data = DataHandle(client, (int)strlen(client) + 1);
        toController->sendMessage(&msg);
    }

    else if (strnicmp(line, "startCRB", 8) == 0)
    {
        char *args[1000];
        int i = 0, n = 0;
        char *buf = new char[strlen(line) + 1];
        strcpy(buf, line);
        while (buf[i])
        {
            if (buf[i] == ' ')
            {
                buf[i] = '\0';
                args[n] = buf + i + 1;
                //printf(stderr,"args %d:%s\n",n,args[n]);
                n++;
            }
            i++;
        }
        args[n] = NULL;
#ifdef _WIN32
        spawnvp(P_NOWAIT, args[0], (char *const *)args);
#else
        int pid = fork();
        if (pid == 0)
        {
            //printf(stderr,"args0:%s\n",args[n]);
            execvp(args[0], args);
        }
        else
        {
            // Needed to prevent zombies
            // if childs terminate
            signal(SIGCHLD, SIG_IGN);
        }
#endif
        delete[] buf;
    }
    else if (strnicmp(line, "rebootClient", 8) == 0)
    {
        char *args[4];
        char *executable = (char *)"RemoteRebootSlave";
        int i = 0, n = 1;
        char *buf = new char[strlen(line) + 1];
        strcpy(buf, line);
        while (buf[i] && n < 4)
        {
            if (buf[i] == ' ')
            {
                buf[i] = '\0';
                args[n] = buf + i + 1;
                n++;
            }
            i++;
        }
        args[0] = executable;
        args[3] = NULL;
        fprintf(stderr, "Calling");
        for (int ctr = 0; ctr < 3; ++ctr)
            fprintf(stderr, " %s", args[ctr]);
        fprintf(stderr, "\n");
#ifdef _WIN32
        spawnvp(P_NOWAIT, executable, (char *const *)args);
#else
        int pid = fork();
        if (pid == 0)
        {
            //printf(stderr,"args0:%s\n",args[n]);
            execvp(executable, args);
        }
        else
        {
            // Needed to prevent zombies
            // if childs terminate
            signal(SIGCHLD, SIG_IGN);
        }
#endif
        delete[] buf;
    }

    else if (strnicmp(line, "cleanCovise", 11) == 0)
    {
        char *args[1000];
        args[0] = (char *)"clean_covise";
        args[1] = NULL;
#ifdef _WIN32
        spawnvp(P_NOWAIT, args[0], (char *const *)args);
#else
        int pid = fork();
        if (pid == 0)
        {
            //printf(stderr,"args0:%s\n",args[n]);
            execvp(args[0], args);
        }
        else
        {
            // Needed to prevent zombies
            // if childs terminate
            signal(SIGCHLD, SIG_IGN);
        }
#endif
    }

    if (strnicmp(line, "startCovise", 11) == 0)
    {
        cerr << " Command to start covise detected!" << endl;
        toAG = conn;
        startCovise();
    }

    if (strnicmp(line, "startCover", 10) == 0)
    {

        cerr << "Command to start cover detected!" << endl;
        cerr << "Command to call is: " << line << endl;
#ifndef _WIN32
        int pid = fork();
        if (pid == 0)
#endif
        {
            cerr << "DBG: line check: " << line + 12 << endl;

            //parse string
            char *args[100];
            char *tmp = (char *)line + 11;
            int argc = 2;
            args[0] = (char *)"cover";
            args[1] = tmp;
            while (*tmp)
            {
                if (*tmp == ' ')
                {
                    *tmp = '\0';
                    if (*(tmp + 1))
                    {
                        args[argc] = tmp + 1;
                        argc++;
                    }
                }
                tmp++;
            }
            args[argc] = NULL;

            for (int i = 0; i < argc; i++)
            {
                cerr << "Param"
                     << " is " << args[i] << endl;
            }

#ifdef _WIN32
            spawnvp(P_NOWAIT, "cover", args);
#else
            execvp("cover", args);
#endif
        }
    }

    if (strnicmp(line, "startOpenCover", 14) == 0)
    {

        cerr << "Command to start cover detected!" << endl;
        cerr << "Command to call is: " << line << endl;
#ifndef _WIN32
        int pid = fork();
        if (pid == 0)
#endif
        {
            cerr << "DBG: line check: " << line + 12 << endl;

            //parse string
            char *args[100];
            char *tmp = (char *)line + 15;
            int argc = 2;
            args[0] = (char *)"opencover";
            args[1] = tmp;
            while (*tmp)
            {
                if (*tmp == ' ')
                {
                    *tmp = '\0';
                    if (*(tmp + 1))
                    {
                        args[argc] = tmp + 1;
                        argc++;
                    }
                }
                tmp++;
            }
            args[argc] = NULL;

            for (int i = 0; i < argc; i++)
            {
                cerr << "Param"
                     << " is " << args[i] << endl;
            }

#ifdef _WIN32
            spawnvp(P_NOWAIT, "opencover", args);
#else
            execvp("opencover", args);
#endif
        }
    }

    if (strnicmp(line, "startFEN", 8) == 0)
    {

        cerr << "Command to start FEN detected!" << endl;
        cerr << "Command to call is: " << line << endl;
#ifndef _WIN32
        int pid = fork();
        if (pid == 0)
#endif
        {
            cerr << "DBG: line check: " << line + 11 << endl;

            //parse string
            char *args[100];
            char *tmp = (char *)line + 9;
            int argc = 1;
            args[0] = tmp;
            while (*tmp)
            {

                if (*tmp == ' ')
                {
                    *tmp = '\0';
                    if (*(tmp + 1))
                    {
                        args[argc] = tmp + 1;
                        argc++;
                    }
                }
                tmp++;

                if (*tmp == '"')
                {
                    tmp++;
                    args[argc] = tmp;
                    while (*tmp && *tmp != '"')
                    {
                        tmp++;
                    }
                    if (*++tmp == ' ')
                        *tmp = '\0';
                    else
                        cerr << "possible error: no space after doublequotes" << endl;
                }
            }
            args[argc] = NULL;

            for (int i = 0; i < argc; i++)
            {
                cerr << "Param"
                     << " is " << args[i] << endl;
            }

#ifdef _WIN32
            spawnvp(P_NOWAIT, args[0], args);
            cerr << " FenFloss started!" << endl;
        }
#else
            execvp(args[0], args);
            cerr << " FenFloss started!" << endl;
        }
        else
        {
            // Needed to prevent zombies
            // if childs terminate
            signal(SIGCHLD, SIG_IGN);
        }
#endif
    }

    return 1;
}

char *RemoteDaemon::GetParameters(const char *pLine, int param)
{
    char *result = (char *)malloc(sizeof(char) * 15);

    int j = 0;
    for (unsigned int i = 0; i <= strlen(pLine); i++)
    {
        if (isspace(pLine[i]))
        {
            j++;
            cerr << endl;
        }
        else if (j == param)
        {
            strcat(result, &pLine[i]);
            cerr << pLine[i];
        }
    }

    for (unsigned int i = 0; i < (unsigned int)param; i++)
    {
        cerr << "Fnt:Param" << param << " = " << result << endl;
    }

    return result;
}

int RemoteDaemon::GetParameterCount(const char *pLine)
{
    int j = 0;
    unsigned int i;

    j = 0;
    for (i = 0; i <= strlen(pLine); i++)
    {
        if (isspace(pLine[i]))
        {
            j++;
            cerr << endl;
        }
    }

    return j;
}
