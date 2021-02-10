/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIProxy.h"
#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <util/unixcompat.h>
#include <iostream>
#include <cstdlib>
using std::cerr;
using std::endl;

#include <signal.h>

using namespace covise;

bool debugOn = false;

int main(int argc, char **argv)
{

    TUIProxy *proxy = new TUIProxy(argc, argv);
    proxy->openServer();
    proxy->handleMessages();
    delete proxy;
    return 0;
}

TUIProxy::TUIProxy(int argc, char **argv)
{
    port = coCoviseConfig::getInt("port", "COVER.TabletUI", 31802);
    int i;
    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            // options
            switch (argv[i][1])
            {
            case 'd':
            case 'D': // daemon infos
            {
                debugOn = true;
                break;
            }
#ifndef YAC
            case 'v':
                printf("TUIProxy 1.0, Covise: %s\n", CoviseVersion::shortVersion());
                exit(0);
                break;
            case 'V':
                printf("TUIProxy 1.0, Covise: %s\n", CoviseVersion::longVersion());
                exit(0);
                break;
#endif
            default:
            {
                cerr << "Unrecognized Option -" << argv[i][1] << " \n";
#ifndef YAC
                cerr << "usage: tuiProxy [port] [-v] [-V] [-d]\n   port = TCP port to listen for incomping tabletUI connections,\n     default: 31802, or TabletPC.TCPPort\n   -v : Version\n    -V : Version Long\n    -d : debug Output\n";
#else
                cerr << "usage: tuiProxy [port] [-d]\n   port = TCP port to listen for incomping tabletUI connections,\n     default: 31802, or TabletPC.TCPPort\n    -d : debug Output\n";
#endif
                exit(-1);
                break;
            }
            } // end options switch
        }
        else
        {
            int retval;
            retval = sscanf(argv[i], "%d", &port);
            if (retval != 1)
            {
                std::cerr << "TUIProxy::TUIProxy: sscanf failed" << std::endl;
                return;
            }
            if (port < 100)
            {
                cerr << "usage: tuiProxy [port] [-v] [-V] [-d]\n   port = TCP port to listen for incomping tabletUI connections,\n     default: 31802, or TabletPC.TCPPort\n   -v : Version\n    -V : Version Long\n    -d : debug Output\n";
                exit(-1);
            }
        }
    }
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
    if (debugOn)
        cerr << "opening Server on port" << port << endl;
}

TUIProxy::~TUIProxy()
{
}

void TUIProxy::handleMessages()
{
    Message *msg = new Message;
    while (1)
    {
        const Connection *conn;
        if ((conn = connections.check_for_input(1)))
        {
            if (conn == sConn) // connection to server port
            {
                auto newConn = sConn->spawn_connection();
                struct linger linger;
                linger.l_onoff = 0;
                linger.l_linger = 0;
                if (debugOn)
                    cerr << "new Connection" << endl;
                setsockopt(newConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));
                toCOVER = dynamic_cast<const ServerConnection*>(connections.add(std::move(newConn)));
                std::string line = coCoviseConfig::getEntry("host","COVER.TabletUI");
                Host *serverHost = NULL;
                if (!line.empty())
                {
                    if (strcasecmp(line.c_str(), "NONE") == 0)
                        serverHost = NULL;
                    else
                        serverHost = new Host(line.c_str());
                }
                else
                {
                    serverHost = NULL;
                }

                if (serverHost)
                {
                    toTUI = connections.tryAddNewConnectedConn<ClientConnection>(serverHost, port, 0, 0, 0);
                    toTUI = new ClientConnection(serverHost, port, 0, 0, 0);
                    if (!toTUI)
                    {
                        connections.remove(toCOVER);
                        fprintf(stderr, "Could not connect to server on %s; port %d\n", line.c_str(), port);
                    }
                }
            }
            else if (conn == toCOVER)
            {
                if (conn->recv_msg(msg))
                {
                    if (msg)
                    {

                        switch (msg->type)
                        {

                        case Message::SOCKET_CLOSED:
                        case Message::CLOSE_SOCKET:
                        {

                            connections.remove(toCOVER);
                            connections.remove(toTUI);
                            delete toCOVER;
                            delete toTUI;
                            cerr << "connections closed" << endl;
                        }
                        break;
                        default:
                            toTUI->sendMessage(msg);
                            break;
                        }
                        //delete msg;
                        //msg=NULL;
                    }
                }
            }
            else if (conn == toTUI)
            {
                if (conn->recv_msg(msg))
                {
                    if (msg)
                    {

                        switch (msg->type)
                        {

                        case Message::SOCKET_CLOSED:
                        case Message::CLOSE_SOCKET:
                        {

                            connections.remove(toCOVER);
                            connections.remove(toTUI);
                            delete toCOVER;
                            delete toTUI;
                            cerr << "connections closed" << endl;
                        }
                        break;
                        default:
                            toCOVER->sendMessage(msg);
                            break;
                        }
                        //delete msg;
                        //msg=NULL;
                    }
                }
            }
        }
    }
}

int TUIProxy::openServer()
{
    auto conn = new ServerConnection(port, 0, 0);
    sConn = dynamic_cast<const ServerConnection *>(connections.tryAddNewListeningConn<ServerConnection>(port, 0, 0));
    if (sConn)
    {
        fprintf(stderr, "Could not open server port %d\n", port);
        return -1;
    }
    return 0;
}

void TUIProxy::closeServer()
{
    connections.remove(sConn);
    // tut sonst nicht (close_socket kommt nicht an)
    sConn = nullptr;
}
