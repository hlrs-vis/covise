/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRBProxy.h"
#include <config/CoviseConfig.h>
#include <util/covise_version.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <util/unixcompat.h>

#include <iostream>
#include <signal.h>

using namespace covise;
using std::cerr;
using std::endl;

bool debugOn = false;

VRBPClientList clients;
int main(int argc, char **argv)
{

    VRBProxy *proxy = new VRBProxy(argc, argv);
    proxy->openServer();
    proxy->handleMessages();
    delete proxy;
    return 0;
}

VRBProxy::VRBProxy(int argc, char **argv)
{
    port = coCoviseConfig::getInt("port", "System.VRB.Proxy", 31900);
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
            case 'v':
                printf("VRBProxy 1.0, Covise: %s\n", CoviseVersion::shortVersion());
                exit(0);
                break;
            case 'V':
                printf("VRBProxy 1.0, Covise: %s\n", CoviseVersion::longVersion());
                exit(0);
                break;
            default:
            {
                cerr << "Unrecognized Option -" << argv[i][1] << " \n";
#ifndef YAC
                cerr << "usage: vrbProxy [port] [-v] [-V] [-d]\n   port = TCP port to listen for incomping VRB connections,\n     default: 31900, or VRB.ProxyPort\n   -v : Version\n    -V : Version Long\n    -d : debug Output\n";
#else
                cerr << "usage: vrbProxy [port] [-d]\n   port = TCP port to listen for incomping VRB connections,\n     default: 31900, or VRB.ProxyPort\n    -d : debug Output\n";
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
                std::cerr << "VRBProxy::VRBProxy: sscanf failed" << std::endl;
                return;
            }
            if (port < 100)
            {
                cerr << "usage: vrbProxy [port] [-v] [-V] [-d]\n   port = TCP port to listen for incomping VRB connections,\n     default: 31900, or VRB.ProxyPort\n   -v : Version\n    -V : Version Long\n    -d : debug Output\n";
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

VRBProxy::~VRBProxy()
{
}

void VRBProxy::handleMessages()
{
    while (1)
    {
        Connection *conn;
        Connection *clientConn;
        if ((conn = connections->check_for_input(1)))
        {
            if (conn == sConn) // connection to server port
            {
                clientConn = sConn->spawn_connection();
                struct linger linger;
                linger.l_onoff = 0;
                linger.l_linger = 0;
                if (debugOn)
                    cerr << "new Connection" << endl;
                setsockopt(clientConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

                clients.push_back(new VRBPClient(clientConn, this));
            }
            else
            {
                conn->recv_msg(msg);
                if (msg)
                {
                    clients.sendMessage(msg);
                }
            }
        }
    }
}

void VRBProxy::closeServer()
{
    connections->remove(sConn);
    // tut sonst nicht (close_socket kommt nicht an)
    delete sConn;
    sConn = NULL;
}

void VRBPClientList::sendMessage(Message *msg)
{
    VRBPClient *cl;
    cl = get(msg->conn);
    if (cl)
    {
        if ((msg->type == Message::SOCKET_CLOSED) || (msg->type == Message::CLOSE_SOCKET))
        {
			remove(cl);
            delete cl;
        }
        else
            cl->sendMessage(msg);
    }
}

void VRBPClientList::deleteAll()
{
	clear();
}

VRBPClient *VRBPClientList::get(Connection *c)
{
	for (const auto &it : *this)
	{
		if ((it->toClient == c) || (it->toVRB == c))
			return it;
	}
    return NULL;
}

VRBPClient::VRBPClient(Connection *c, VRBProxy *p)
{
    prox = p;
    toClient = c;
    prox->connections->add(c);

    if (debugOn)
        std::cerr << "new Client" << std::endl;
    Host *serverHost = NULL;
    int tcp_p = coCoviseConfig::getInt("tcpPort", "System.VRB.Server", 31800);
    std::string line = coCoviseConfig::getEntry("System.VRB.Server");
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
    toVRB = new ClientConnection(serverHost, tcp_p, 0, 0, 0);
    if (toVRB)
    {
        if (!toVRB->is_connected()) // could not open server port
        {
            fprintf(stderr, "Could not connect to server on %s; port %d\n", serverHost->getAddress(), tcp_p);
            delete toVRB;
            toVRB = NULL;
        }
        prox->connections->add(toVRB);

        struct linger linger;
        linger.l_onoff = 0;
        linger.l_linger = 1;
        setsockopt(toVRB->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

        if (debugOn)
            cerr << "new Client" << serverHost << endl;
    }
}

VRBPClient::~VRBPClient()
{
    if (debugOn)
        cerr << "closing vrb connection" << endl;
    prox->connections->remove(toVRB);
    prox->connections->remove(toClient);
    delete toVRB;
    toVRB = NULL;
    delete toClient;
    toClient = NULL;
}

void VRBPClient::sendMessage(Message *msg)
{
    if (msg->conn == toVRB)
    {
        toClient->sendMessage(msg);
    }
    else
    {
        toVRB->sendMessage(msg);
    }
}

int VRBProxy::openServer()
{
    sConn = new ServerConnection(port, 0, 0);

    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    setsockopt(sConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

    sConn->listen();
    if (!sConn->is_connected()) // could not open server port
    {
        fprintf(stderr, "Could not open server port %d\n", port);
        delete sConn;
        sConn = NULL;
        return (-1);
    }
    connections = new ConnectionList();
    connections->add(sConn);
    msg = new Message;

    return 0;
}
