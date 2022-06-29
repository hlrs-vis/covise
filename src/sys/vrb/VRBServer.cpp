/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */





#include <qsocketnotifier.h>
#define IOMANIPH
// don't include iomanip.h becaus it interferes with qt


#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
//#include <windows.h>
#else
#include <unistd.h>
#include <dirent.h>
#include <signal.h>
#endif

#include "VRBServer.h"
#include <vrb/server/VrbClientList.h>
using std::cerr;
using std::endl;
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <sys/socket.h>
#endif

#include <QDir>
#include <QHostInfo>
#include <QList>
#include <QHostAddress>

#include "gui/VRBapplication.h"
#include "gui/coRegister.h"
#include "VrbUiClientList.h"
#include <vrb/server/VrbClientList.h>
#include <VrbUiMessageHandler.h>
extern ApplicationWindow *mw;


#include <config/CoviseConfig.h>
#include <net/covise_socket.h>
#include <net/covise_connect.h>
#include <net/message_types.h>
#include <net/udpMessage.h>
#include <util/unixcompat.h>

#include <qtutil/NetHelp.h>
#include <qtutil/FileSysAccess.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qdir.h>
#include <QtNetwork/qhostinfo.h>
#include <QSocketNotifier>
//#include <QTreeWidget>

#include <vrb/server/VrbClientList.h>

#include "gui/VRBapplication.h"

#ifndef MAX_PATH
#define MAX_PATH 1024
#endif
//#define MB_DEBUG

vrb::VRBClientList *vrbClients;

using namespace covise;
using namespace vrb;

VRBServer::VRBServer(bool gui)
    :m_gui(gui)
{
    covise::Socket::initialize();

    m_tcpPort = coCoviseConfig::getInt("tcpPort", "System.VRB.Server", 31800);
	m_udpPort = coCoviseConfig::getInt("udpPort", "System.VRB.Server", m_tcpPort +1);
    requestToQuit = false;
    if (gui)
    {
        vrbClients = &uiClients;
        handler = new VrbUiMessageHandler(this);
    }
    else
    {
        vrbClients = &clients;
        handler = new VrbMessageHandler(this);
    }
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
}

VRBServer::~VRBServer()
{
	delete[] ip;
    delete handler;
    //cerr << "closed Server connection" << endl;
}

void VRBServer::closeServer()
{
    if (m_gui)
    {
    delete serverSN;
    serverSN = nullptr;
    }


    connections.remove(sConn);
    sConn = nullptr;
    // tut sonst nicht (close_socket kommt nicht an)delete sConn;
    //sConn = NULL;

    requestToQuit = true;
    handler->closeConnection();
}

void VRBServer::removeConnection(const covise::Connection * conn)
{
    connections.remove(conn);
    if (requestToQuit && handler->numberOfClients() == 0)
    {
        exit(0);
    }
}

int VRBServer::getPort()
{
    return m_tcpPort;
}

int VRBServer::getUdpPort()
{
    return m_udpPort;
}

int VRBServer::openServer(bool printport)
{
    if (printport) {
        for (int port = 1024; port <= 0xffff; ++port) {
            sConn = connections.tryAddNewListeningConn<ServerConnection>(port, 0, 0);
            if (sConn) {
                m_tcpPort = port;
                std::cout << port << std::endl << std::endl << std::flush;
                break;
            }
        }

        if (!sConn) {
            fprintf(stderr, "Could not open server on any port\n");
            return (-1);
        }
    }
    else
    {
        sConn = connections.tryAddNewListeningConn<ServerConnection>(m_tcpPort,
                                                                     0, 0);
        if (!sConn) {
            fprintf(stderr, "Could not open server port %d\n", m_tcpPort);
            return (-1);
        }
    }
    if (!msg)
        msg = new Message;
    if(!udpMsg)
	    udpMsg = new UdpMessage;
    if (m_gui)
    {
        QSocketNotifier *serverSN = new QSocketNotifier(sConn->get_id(NULL), QSocketNotifier::Read);
        QObject::connect(serverSN, SIGNAL(activated(int)),
            this, SLOT(processMessages()));
    }
    return 0;
}

void VRBServer::loop()
{
    while (1)
    {
		processMessages(10.f);
    }
}

void VRBServer::processMessages(float waitTime)
{
	while (const Connection *conn = connections.check_for_input(waitTime))
    {
#ifdef MB_DEBUG
        std::cerr << "VRB: have input" << std::endl;
#endif
        if (conn == udpConn) // udp connection
        {
			processUdpMessages();
			return;
		}
        else if (conn == sConn) //tcp connection to server port
        {
#ifdef MB_DEBUG
            std::cerr << "accepting new client connection" << std::endl;
#endif
            std::unique_ptr<Connection> clientConn;
            clientConn = sConn->spawn_connection();
#ifdef MB_DEBUG
            std::cerr << "spawned new client connection" << std::endl;
#endif

            struct linger linger;
            linger.l_onoff = 0;
            linger.l_linger = 0;
            setsockopt(clientConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));
            vrb::ConnectionDetails::ptr cd;
            if (m_gui)
            {
                QSocketNotifier *sn = new QSocketNotifier(clientConn->get_id(NULL), QSocketNotifier::Read);
                QObject::connect(sn, SIGNAL(activated(int)),
                    this, SLOT(processMessages()));

                auto uicd = UiConnectionDetails::ptr{new UiConnectionDetails{}};
                uicd->notifier.reset(sn);
                cd = std::move(uicd);
            }
			else
			{
                cd.reset(new vrb::ConnectionDetails{});
			}
            cd->tcpConn = connections.add(std::move(clientConn));
            cd->udpConn = udpConn;
            handler->addClient(std::move(cd));
            std::cerr << "VRB new client request" << std::endl;
        }
        else
        {
#ifdef MB_DEBUG
            std::cerr << "Receive Message!" << std::endl;
#endif
            if (m_gui)
            {
                static_cast<VrbUiMessageHandler*>(handler)->setClientNotifier(conn, false);
            }

            if (conn->recv_msg(msg))
            {
#ifdef MB_DEBUG
                std::cerr << "Message found!" << std::endl;
#endif
                if (msg)
                {
                    handler->handleMessage(msg);
#ifdef MB_DEBUG
                    std::cerr << "Left handleClient!" << std::endl;
#endif
                }
                else
                {
#ifdef MB_DEBUG
                    std::cerr << "Message is NULL!" << std::endl;
#endif
                }
            }

            if (m_gui)
            {
                static_cast<VrbUiMessageHandler*>(handler)->setClientNotifier(conn, true);
            }


        }
    }
}
void VRBServer::processUdpMessages()
{
	while (udpConn->recv_udp_msg(udpMsg))
	{
		handler->handleUdpMessage(udpMsg);
	}
}
bool VRBServer::startUdpServer() 
{
    auto conn = std::unique_ptr<UDPConnection>(new UDPConnection{0, 0, m_udpPort, nullptr});
    if (conn->getSocket()->get_id() < 0)
    {
		return false;
	}
	struct linger linger;
	linger.l_onoff = 0;
	linger.l_linger = 0;
        udpConn = dynamic_cast<const UDPConnection*>(connections.add(std::move(conn)));
	setsockopt(udpConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char*)& linger, sizeof(linger));
	if (m_gui)
	{
		QSocketNotifier* sn = new QSocketNotifier(udpConn->get_id(NULL), QSocketNotifier::Read);
		QObject::connect(sn, SIGNAL(activated(int)),
			this, SLOT(processUdpMessages()));
	}
	return true;
}
