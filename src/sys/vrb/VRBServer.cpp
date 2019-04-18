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
#include <vrbserver/VrbClientList.h>
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
#include <vrbserver/VrbClientList.h>
#include <VrbUiMessageHandler.h>
extern ApplicationWindow *mw;


#include <config/CoviseConfig.h>
#include <net/covise_socket.h>
#include <net/covise_connect.h>
#include <net/message_types.h>
#include <util/unixcompat.h>

#include <qtutil/NetHelp.h>
#include <qtutil/FileSysAccess.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qdir.h>
#include <QtNetwork/qhostinfo.h>
#include <QSocketNotifier>
//#include <QTreeWidget>

#include <vrbserver/VrbClientList.h>

#include "gui/VRBapplication.h"

#ifndef MAX_PATH
#define MAX_PATH 1024
#endif
vrb::VRBClientList *vrbClients;
//#define MB_DEBUG

using namespace covise;
using namespace vrb;
VRBServer::VRBServer(bool gui)
    :m_gui(gui)
{
    covise::Socket::initialize();

    port = coCoviseConfig::getInt("port", "System.VRB.Server", 31800);
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
    delete sConn;
    delete handler;
    //cerr << "closed Server connection" << endl;
}

void VRBServer::closeServer()
{
    if (m_gui)
    {
    delete serverSN;
    }


    connections->remove(sConn);
    // tut sonst nicht (close_socket kommt nicht an)delete sConn;
    //sConn = NULL;

    requestToQuit = true;
    handler->closeConnection();
}

void VRBServer::removeConnection(covise::Connection * conn)
{
    connections->remove(conn);
    if (requestToQuit && handler->numberOfClients() == 0)
    {
        exit(0);
    }
}

int VRBServer::openServer()
{
    sConn = new ServerConnection(port, 0, (sender_type)0);

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
        processMessages();
    }
}

void VRBServer::processMessages()
{
    while (Connection *conn = connections->check_for_input(0.0001f))
    {
        if (conn == sConn) // connection to server port
        {
            Connection *clientConn = sConn->spawn_connection();
            struct linger linger;
            linger.l_onoff = 0;
            linger.l_linger = 0;
            setsockopt(clientConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

            if (m_gui)
            {
                QSocketNotifier *sn = new QSocketNotifier(clientConn->get_id(NULL), QSocketNotifier::Read);
                QObject::connect(sn, SIGNAL(activated(int)),
                    this, SLOT(processMessages()));
                VrbUiClient *cl = new VrbUiClient(clientConn, sn);
                handler->addClient(cl);
                std::cerr << "VRB new client: Numclients=" << handler->numberOfClients() << std::endl;
            }
            connections->add(clientConn); //add new connection;
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



