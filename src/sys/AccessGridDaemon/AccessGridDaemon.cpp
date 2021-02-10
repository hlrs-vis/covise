/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <qsocketnotifier.h>
#define IOMANIPH
// don't include iomanip.h becaus it interferes with qt

#ifdef _WIN32
#include <winsock2.h>
#include <process.h>
#endif
#include <util/unixcompat.h>
#include <iostream>
#include <sysdep/net.h>

#include "AccessGridDaemon.h"
#include <stdio.h>
#include <string.h>
#include <signal.h>

#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <covise/covise_msg.h>

using namespace covise;
using std::cerr;
using std::endl;

int main()
{
    AccessGridDaemon *agd;
    agd = new AccessGridDaemon();
    if (agd->openServer() >= 0)
    {
        agd->loop();
    }
    delete agd;
}

AccessGridDaemon::AccessGridDaemon()
    :connections(new ConnectionList())
{
    toController = NULL;
    toAG = NULL;
    port = coCoviseConfig::getInt("port", "System.AccessGridDaemon.Server", 31098);

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
}

AccessGridDaemon::~AccessGridDaemon()
{
    delete sConn;
    delete connections;
    //cerr << "closed Server connection" << endl;
}

void AccessGridDaemon::closeServer()
{
    connections->remove(sConn);
    cerr << "removing" << sConn;
    // tut sonst nicht (close_socket kommt nicht an)
    // das will ich doch mal probieren
    delete sConn;
    sConn = NULL;
}

int AccessGridDaemon::openServer()
{
    sConn = connections->tryAddNewListeningConn<ServerConnection>(port, 0, 0);
    if (!sConn)
    {
        fprintf(stderr, "Could not open server port %d\n", port);
        return (-1);
    }
    msg = new Message;
    return 0;
}

void AccessGridDaemon::loop()
{
    while (processMessages())
    {
    }
}

int AccessGridDaemon::processMessages()
{
    const Connection *conn;
    const char *line;
    while ((conn = connections->check_for_input(1)))
    {

        cerr << "data on conn " << conn;
        if (conn == sConn) // connection to server port
        {
            connections->add(sConn->spawnSimpleConnection()); //add new connection;
        }
        else
        {
            if (conn == toController)
            {
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

                if ((line = ((SimpleServerConnection *)conn)->readLine()) != NULL)
                {

                    if (!handleClient(line, conn))
                        return 0;
                }
            }
        }
    }
    return 1;
}

void AccessGridDaemon::handleClient(Message *msg)
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
            toController = NULL;
        connections->remove(msg->conn);
        cerr << "remove" << msg->conn;
        if (toAG)
            toAG->getSocket()->write("masterLeft", (unsigned int)strlen("masterLeft") + 1);
        cerr << "controller left" << endl;
    }
    break;
    default:
        cerr << "unknown message in vrb" << endl;
        break;
    }
}

void AccessGridDaemon::startCovise()
{
    char cport[200];
    int sPort;
    auto conn = std::unique_ptr<ServerConnection>(new ServerConnection(&sPort, 0, (sender_type)0));
    conn->listen();
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
    conn->acceptOne();
    toController = dynamic_cast<const ServerConnection*>(connections->add(std::move(conn))); //add new connection;
    cerr << "add" << toController;
}

int AccessGridDaemon::handleClient(const char *line, const Connection *conn)
{
    cerr << line << endl;
#ifdef _WIN32
    if (strnicmp(line, "ConnectionClosed", 16) == 0)
#else
    if (strncasecmp(line, "ConnectionClosed", 16) == 0)
#endif
    {
        if (conn == toAG)
            toAG = NULL;
        connections->remove(conn);
        cerr << "remove" << conn;
        delete conn;
    }
#ifdef _WIN32
    else if (strnicmp(line, "quit", 4) == 0)
#else
    else if (strncasecmp(line, "quit", 4) == 0)
#endif
    {
        return 0;
    }
#ifdef _WIN32
    else if (strnicmp(line, "check", 5) == 0)
#else
    else if (strncasecmp(line, "check", 5) == 0)
#endif
    {
        if (toController == NULL)
            conn->getSocket()->write("masterLeft", (unsigned int)strlen("masterLeft") + 1);
        else
            conn->getSocket()->write("masterRunning\n", (unsigned int)strlen("masterRunning\n") + 1);
    }
#ifdef _WIN32
    else if (strnicmp(line, "join", 4) == 0)
#else
    else if (strncasecmp(line, "join", 4) == 0)
#endif
    {
        char *client = new char[strlen(line) + 1];
        int port = 31098;
        int retval;
        retval = sscanf(line + 5, "%s:%d", client, &port);
        if (retval != 1)
        {
            std::cerr << "AccessGridDaemon::handleClient: sscanf failed" << std::endl;
            return 0;
        }
        if (toController == NULL)
        {
            toAG = conn;
            startCovise();
        }
        Message *msg = new Message;
        msg->type = COVISE_MESSAGE_ACCESSGRID_DAEMON;
        msg->data = DataHandle((char *)line, strlen(line) + 1);
        toController->sendMessage(msg);
    }
#ifdef _WIN32
    else if (strnicmp(line, "startCRB", 8) == 0)
#else
    else if (strncasecmp(line, "startCRB", 8) == 0)
#endif
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
                //fprintf(stderr,"args %d:%s\n",n,args[n]);
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
            //fprintf(stderr,"args0:%s\n",args[n]);
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
#ifdef _WIN32
    else if (strnicmp(line, "cleanCovise", 11) == 0)
#else
    else if (strncasecmp(line, "cleanCovise", 11) == 0)
#endif
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
            //fprintf(stderr,"args0:%s\n",args[n]);
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
#ifdef _WIN32
    if (strnicmp(line, "startCovise", 11) == 0)
#else
    if (strncasecmp(line, "startCovise", 11) == 0)
#endif
    {
        toAG = conn;
        startCovise();
    }
#ifdef _WIN32
    if (strnicmp(line, "startCover", 10) == 0)
#else
    if (strncasecmp(line, "startCover", 10) == 0)
#endif
    {

#ifdef _WIN32
        spawnlp(P_NOWAIT, "cover", "cover", line + 11, NULL);
#else
        int pid = fork();
        if (pid == 0)
        {
            execlp("cover", "cover", line + 11, NULL);
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
