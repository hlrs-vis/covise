/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef GUI
#include <QSocketNotifier>
#define IOMANIPH
// don't include iomanip.h becaus it interferes with qt
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
//#include <windows.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif

#include "VRBServer.h"
#include "VRBClientList.h"
#include <net/covise_connect.h>
#include <net/tokenbuffer.h>
#include <util/unixcompat.h>
#include <qtutil/FileSysAccess.h>
#include <qtutil/NetHelp.h>
#include <util/coTabletUIMessages.h>

#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>

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

#ifdef GUI
#include "gui/VRBapplication.h"
#include "gui/coRegister.h"
extern ApplicationWindow *mw;
#endif

#include <config/CoviseConfig.h>

using namespace covise;

VRBServer::VRBServer()
{
    port = coCoviseConfig::getInt("port", "System.VRB.Server", 31800);
    requestToQuit = false;
    currentFileClient = NULL;
    currentFile = NULL;
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
}

VRBServer::~VRBServer()
{
    delete sConn;
    //cerr << "closed Server connection" << endl;
}

void VRBServer::closeServer()
{
#ifdef GUI
    delete serverSN;
#endif
    connections->remove(sConn);
    // tut sonst nicht (close_socket kommt nicht an)delete sConn;
    //sConn = NULL;
    TokenBuffer tb;
    requestToQuit = true;
    clients.sendMessage(tb, -2, COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION);
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

#ifdef GUI
    QSocketNotifier *serverSN;

    serverSN = new QSocketNotifier(sConn->get_id(NULL), QSocketNotifier::Read);
    QObject::connect(serverSN, SIGNAL(activated(int)),
                     this, SLOT(processMessages()));
#endif
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
    Connection *conn;
    Connection *clientConn;
    while ((conn = connections->check_for_input(0.0001f)))
    {
        if (conn == sConn) // connection to server port
        {
            clientConn = sConn->spawn_connection();
            struct linger linger;
            linger.l_onoff = 0;
            linger.l_linger = 0;
            setsockopt(clientConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

#ifdef GUI

            QSocketNotifier *sn;

            sn = new QSocketNotifier(clientConn->get_id(NULL), QSocketNotifier::Read);
            QObject::connect(sn, SIGNAL(activated(int)),
                             this, SLOT(processMessages()));

            clients.append(new VRBSClient(clientConn, sn));
#endif
            connections->add(clientConn); //add new connection;
        }
        else
        {
#ifdef MB_DEBUG
            std::cerr << "Receive Message!" << std::endl;
#endif
            clients.reset();

#ifdef GUI
            VRBSClient *cl = clients.get(conn);
            if (cl != NULL)
            {
                cl->getSN()->setEnabled(false);
            }
#endif

            if (conn->recv_msg(msg))
            {
#ifdef MB_DEBUG
                std::cerr << "Message found!" << std::endl;
#endif

                if (msg)
                {
                    handleClient(msg);
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

#ifdef GUI
            cl = clients.get(conn);
            if (cl != NULL)
            {
                QSocketNotifier *sn = cl->getSN();
                sn->setEnabled(true);
            }
#endif
        }
    }
}

void VRBServer::handleClient(Message *msg)
{
    char *Class;
    int classID, senderID;
    char *name;
    char *value;
    int fd;
    TokenBuffer tb(msg);

    switch (msg->type)
    {
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Request file message!" << std::endl;
#endif
        struct stat statbuf;
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            c->addBytesSent(msg->length);
        }
        char *filename;
        tb >> filename;
        int requestorsID;
        tb >> requestorsID;
        TokenBuffer rtb;
        if (stat(filename, &statbuf) >= 0)
        {
            rtb << requestorsID;
            if ((fd = open(filename, O_RDONLY)) > 0)
            {
                rtb << (int)statbuf.st_size;
                const char *buf = rtb.allocBinary(statbuf.st_size);
                ssize_t retval;
                retval = read(fd, (void *)buf, statbuf.st_size);
                if (retval == -1)
                {
                    cerr << "VRBServer::handleClient: read failed" << std::endl;
                    return;
                }
            }
            else
            {
                cerr << " file access error, could not open " << filename << endl;
                rtb << 0;
            }

            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SEND_FILE;
            if (c)
            {
                c->addBytesReceived(m.length);
            }
            msg->conn->send_msg(&m);
        }
        else
        {
            // forward this request to the COVER that loaded the current file
            if ((currentFileClient) && (currentFileClient != c) && (currentFileClient->getID() != requestorsID))
            {
                currentFileClient->conn->send_msg(msg);
                currentFileClient->addBytesReceived(msg->length);
            }
            else
            {
                rtb << requestorsID;
                rtb << 0; // file not found
                Message m(rtb);
                m.type = COVISE_MESSAGE_VRB_SEND_FILE;
                if (c)
                {
                    c->addBytesReceived(m.length);
                }
                msg->conn->send_msg(&m);
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_SEND_FILE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Send file message!" << std::endl;
#endif
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            c->addBytesSent(msg->length);
        }
        int destinationID;
        tb >> destinationID;
        // forward this File to the one that requested it
        c = clients.get(destinationID);
        if (c)
        {
            c->conn->send_msg(msg);
            c->addBytesReceived(msg->length);
        }
    }
    break;
    case COVISE_MESSAGE_VRB_CURRENT_FILE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Current file message!" << std::endl;
#endif
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
#ifdef MB_DEBUG
            std::cerr << "====> Client valid!" << std::endl;
#endif
            c->addBytesSent(msg->length);
#ifdef MB_DEBUG
            std::cerr << "====> Added length of msg to bytes sent!" << std::endl;
#endif
            currentFileClient = c;
            delete[] currentFile;
            char *buf;
            tb >> buf;
            currentFile = new char[strlen(buf) + 1];
            strcpy(currentFile, buf);

// send current file name to all clients
#ifdef MB_DEBUG
            std::cerr << "====> Sending current file to all clients!" << std::endl;
#endif

            if (currentFile && currentFileClient)
            {
                TokenBuffer rtb3;
                rtb3 << currentFile;
                rtb3 << currentFileClient->getID();
                Message m(rtb3);
                m.type = COVISE_MESSAGE_VRB_CURRENT_FILE;
                c->conn->send_msg(&m);
#ifdef MB_DEBUG
                std::cerr << "====> Send done!" << std::endl;
#endif
                clients.reset();
                while ((c = clients.current()))
                {
                    if (c->conn != msg->conn)
                    {
                        c->conn->send_msg(&m);
                        c->addBytesReceived(m.length);
                    }
                    clients.next();
                }
#ifdef MB_DEBUG
                std::cerr << "====> Iterate all vrb clients for current file xmit!" << std::endl;
#endif
                //delete[] currentFile;
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE: // Set Registry value
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set registry value!" << std::endl;
#endif
        tb >> Class;
        tb >> classID;
        tb >> name;
        tb >> value;
        registry.setVar(Class, classID, name, value);
#ifdef GUI
        mw->registry->updateEntry(Class, classID, name, value);
#endif
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry subscribe class!" << std::endl;
#endif
        tb >> Class;
        tb >> classID;
        tb >> senderID;
        registry.observe(Class, classID, senderID);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry subscribe variable!" << std::endl;
#endif
        tb >> Class;
        tb >> classID;
        tb >> name;
        tb >> senderID;
        registry.observe(Class, classID, senderID, name);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry unsubscribe class!" << std::endl;
#endif
        tb >> Class;
        tb >> classID;
        tb >> senderID;
        registry.unObserve(Class, classID, senderID);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry unsubscribe variable!" << std::endl;
#endif
        tb >> Class;
        tb >> classID;
        tb >> name;
        tb >> senderID;
        registry.unObserve(Class, classID, senderID, name);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry create entry!" << std::endl;
#endif
        int staticVar;
        tb >> Class;
        tb >> classID;
        tb >> name;
        tb >> staticVar;
        registry.create(Class, classID, name, staticVar);
#ifdef GUI
        mw->registry->updateEntry(Class, classID, name, "");
#endif
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry delete entry!" << std::endl;
#endif
        tb >> Class;
        tb >> classID;
        tb >> name;
        registry.deleteEntry(Class, classID, name);
#ifdef GUI
        mw->registry->removeEntry(Class, classID, name);
#endif
    }
    break;
    case COVISE_MESSAGE_VRB_CONTACT:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Contact!" << std::endl;
#endif
        char *name;
        char *ip;
        tb >> name;
        tb >> ip;
        if (strcasecmp(name, "CRB") == 0)
        {
        }
        else
        {
#ifdef GUI
            VRBSClient *c = clients.get(msg->conn);
            if (c)
            {
                c->setContactInfo(ip, name);
            }
#else
            clients.append(new VRBSClient(msg->conn, ip, name));
#endif
        }
        //cerr << name << " connected from " << ip << endl;
        // send a list of all participants to all clients
        TokenBuffer rtb;
        rtb << clients.num();
        clients.reset();
        VRBSClient *c;
        while ((c = clients.current()))
        {
            if (strcmp(c->getIP(), ip) == 0)
            {
                rtb << c->getName();
                rtb << c->getIP();
                break;
            }
            clients.next();
        }
    }
    break;
    case COVISE_MESSAGE_VRB_CONNECT_TO_COVISE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Connect Covise!" << std::endl;
#endif
        char *ip;
        tb >> ip;
        clients.reset();
        VRBSClient *c;
        while ((c = clients.current()))
        {
            if (strcmp(c->getIP(), ip) == 0)
            {
                c->conn->send_msg(msg);
                break;
            }
            clients.next();
        }
    }
    break;
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set UserInfo!" << std::endl;
#endif
        char *ui;
        tb >> ui;
        clients.reset();
        VRBSClient *c2;
        VRBSClient *c = clients.get(msg->conn);
        TokenBuffer rtb;
        TokenBuffer rtb2;
        if (c)
        {
            rtb << 1;
            c->setUserInfo(ui);
            c->getInfo(rtb);
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
            clients.reset();
            while ((c2 = clients.current()))
            {
                if (c2 != c)
                    c2->conn->send_msg(&m);
                clients.next();
            }
        }
        rtb2 << clients.num();
        clients.reset();
        while ((c2 = clients.current()))
        {
            c2->getInfo(rtb2);
            clients.next();
        }
        Message m(rtb2);
        m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
        c->conn->send_msg(&m);

        // send current file name to new client
        if (currentFile && currentFileClient)
        {
            TokenBuffer rtb3;
            rtb3 << currentFile;
            rtb3 << currentFileClient->getID();
            Message m(rtb3);
            m.type = COVISE_MESSAGE_VRB_CURRENT_FILE;
            c->conn->send_msg(&m);
        }
    }
    break;
    case COVISE_MESSAGE_RENDER:
    case COVISE_MESSAGE_RENDER_MODULE: // send Message to all others in same group
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Render/Render Module!" << std::endl;
#endif
        int toGroup = -2;
#ifdef MB_DEBUG
        std::cerr << "====> Get senders vrbc connection!" << std::endl;
#endif

        VRBSClient *c = clients.get(msg->conn);

        if (c)
        {
#ifdef MB_DEBUG
            std::cerr << "====> Sender found!" << std::endl;
            std::cerr << "====> Sender: " << c->getIP() << std::endl;
#endif
            toGroup = c->getGroup();
            c->addBytesSent(msg->length);
#ifdef MB_DEBUG
            std::cerr << "====> Increased amount of sent bytes!" << std::endl;
#endif
        }
#ifdef MB_DEBUG
        std::cerr << "====> Reset clients!" << std::endl;
#endif

        clients.reset();
#ifdef MB_DEBUG
        std::cerr << "====> Iterate through all vrbcs!" << std::endl;
#endif
        while ((c = clients.current()))
        {
            if ((c->getGroup() == toGroup) && (c->conn != msg->conn))
            {
#ifdef MB_DEBUG
                std::cerr << "====> Distribute render message!" << std::endl;
#endif
                c->conn->send_msg(msg);
                c->addBytesReceived(msg->length);
#ifdef MB_DEBUG
                std::cerr << "====> Increased amount of sent bytes!" << std::endl;
#endif
            }
            clients.next();
        }
    }
    break;
    case COVISE_MESSAGE_VRB_CHECK_COVER:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Check Cover!" << std::endl;
#endif
        char *ip;
        TokenBuffer rtb;
        tb >> ip;
        clients.reset();
        VRBSClient *c;
        while ((c = clients.current()))
        {
            if (strcmp(c->getIP(), ip) == 0)
            {
                rtb << 1;
                break;
            }
            clients.next();
        }
        rtb << 0;
        Message m(rtb);
        m.type = COVISE_MESSAGE_VRB_CHECK_COVER;
        msg->conn->send_msg(&m);
    }
    break;
    case COVISE_MESSAGE_VRB_GET_ID:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Get ID!" << std::endl;
#endif
        TokenBuffer rtb;
        clients.reset();
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            rtb << c->getID();
        }
        else
            rtb << -1;

        Message m(rtb);
        m.type = COVISE_MESSAGE_VRB_GET_ID;
        msg->conn->send_msg(&m);
    }
    break;
    case COVISE_MESSAGE_VRB_SET_GROUP:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set Group!" << std::endl;
#endif
        int newGroup;
        tb >> newGroup;
        VRBSClient *c, *c2;
        c = clients.get(msg->conn);
        int numInGroup = clients.numInGroup(newGroup);
        if (numInGroup == 1)
        {
            // is there  a master in this group?
            clients.reset();
            while ((c2 = clients.current()))
            {
                if (c2->getGroup() == newGroup)
                {
                    if (c2->getMaster())
                    {
                        if (c->getMaster())
                        {
                            TokenBuffer rtb;
                            rtb << c->getID();
                            rtb << 0;
                            Message m(rtb);
                            m.type = COVISE_MESSAGE_VRB_SET_MASTER;
                            c->setMaster(0);
                            c->conn->send_msg(&m);
                        }
                    }
                    else
                    {
                        if (!c->getMaster())
                        {
                            TokenBuffer rtb;
                            rtb << c->getID();
                            rtb << 1;
                            Message m(rtb);
                            m.type = COVISE_MESSAGE_VRB_SET_MASTER;
                            c->setMaster(1);
                            c->conn->send_msg(&m);
                        }
                    }
                }
                clients.next();
            }
        }
        else
        {
            // there is already a master in this group
            if (c->getMaster())
            {
                TokenBuffer rtb;
                rtb << c->getID();
                rtb << 0;
                Message m(rtb);
                m.type = COVISE_MESSAGE_VRB_SET_MASTER;
                c->setMaster(0);
                c->conn->send_msg(&m);
            }
        }
        if (c)
        {
            c->setGroup(newGroup);
            TokenBuffer rtb;
            rtb << c->getID();
            rtb << newGroup;
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SET_GROUP;
            VRBSClient *c2;
            clients.reset();
            while ((c2 = clients.current()))
            {
                if (c2 != c)
                    c2->conn->send_msg(&m);
                clients.next();
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_SET_MASTER:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set Master!" << std::endl;
#endif
        int masterState;
        tb >> masterState;
        if (!masterState)
            break;
        clients.reset();
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            c->setMaster(masterState);
            TokenBuffer rtb;
            rtb << c->getID();
            rtb << masterState;
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SET_MASTER;
            VRBSClient *c2;
            clients.reset();
            while ((c2 = clients.current()))
            {
                if (c2->getGroup() == c->getGroup())
                {
                    if (c2 == c)
                    {
                        c2->setMaster(masterState);
                    }
                    else
                    {
                        c2->setMaster(0);
                    }
                    c2->conn->send_msg(&m);
                }
                clients.next();
            }
        }
    }
    break;
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Close Socket!" << std::endl;
#endif
        connections->remove(msg->conn);
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            if (currentFileClient == c)
                currentFileClient = NULL;
            bool wasMaster = false;
            int MasterGroup = 0;
            TokenBuffer rtb;
            cerr << c->getName() << " (host " << c->getIP() << " ID: " << c->getID() << ") left" << endl;
            rtb << c->getID();
            if (c->getMaster())
            {
                MasterGroup = c->getGroup();
                cerr << "Master Left" << endl;
                wasMaster = true;
            }
#ifdef GUI
            mw->registry->removeEntries(c->getID());
#endif
            clients.remove();

            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_QUIT;
            clients.reset();
            VRBSClient *cc;
            while ((cc = clients.current()))
            {
                cc->conn->send_msg(&m);
                clients.next();
            }
            if (wasMaster)
            {
                clients.reset();
                while ((cc = clients.current()))
                {
                    if (cc->getGroup() == MasterGroup)
                    {
                        break;
                    }
                }
                if (cc)
                {
                    TokenBuffer rtb;
                    rtb << cc->getID();
                    rtb << 1;
                    Message m(rtb);
                    m.type = COVISE_MESSAGE_VRB_SET_MASTER;
                    VRBSClient *c2;
                    clients.reset();
                    while ((c2 = clients.current()))
                    {
                        if (c2->getGroup() == MasterGroup)
                        {
                            c2->setMaster(1);
                        }
                        c2->conn->send_msg(&m);
                        clients.next();
                    }
                }
            }
        }
        else
            cerr << "CRB left" << endl;

        cerr << "Numclients: " << clients.num() << endl;
        if (clients.num() == 0 && requestToQuit)
        {
            exit(0);
        }
    }
    break;
    case COVISE_MESSAGE_VRB_FB_RQ:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB FileBrowser Request!" << std::endl;
#endif
        int id = 0;
        int recvId = 0;
        int type = 0;
        char *filter = NULL;
        char *path = NULL;
        char *location = NULL;
        QStringList list;

        //Message for tabletui filebrowser request
        TokenBuffer tb(msg);
        tb >> id;
        tb >> recvId;
        tb >> type;

        VRBSClient *receiver = clients.get(msg->conn);

        switch (type)
        {
        case TABLET_REQ_DIRLIST:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request DirList!" << std::endl;
#endif
            tb >> filter;
            tb >> path;
            tb >> location;

            QString qFilter = filter;
            QString qPath = path;

            NetHelp networkHelper;
            QString localIP = networkHelper.getLocalIP();
            if (strcmp(location, localIP.toStdString().c_str()) == 0)
            {
                //Server-side dir request
                FileSysAccess fileSys;
                list = fileSys.getFileSysList(qFilter, qPath, false);

                //Create return message
                TokenBuffer rtb;
                rtb << TABLET_SET_DIRLIST;
                rtb << id;
                rtb << list.size();
                for (int i = 0; i < list.size(); i++)
                {
                    std::string tempStr = list.at(i).toStdString();
                    rtb << tempStr.c_str();
                }

                Message m(rtb);
                m.type = COVISE_MESSAGE_VRB_FB_SET;

                //Send message
                receiver->conn->send_msg(&m);
            }
            else
            {
                //Determine destination's VRBSClient object and
                //route the request to the appropriate client
                RerouteRequest(location, TABLET_SET_DIRLIST, id, recvId, qFilter, qPath);
            }
        }
        break;
        case TABLET_REQ_FILELIST:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request FileList!" << std::endl;
#endif
            tb >> filter;
            tb >> path;
            tb >> location;

            QString qFilter = filter;
            QString qPath = path;

            // not used bool sendDir = false;
            NetHelp networkHelper;
            QString localIP = networkHelper.getLocalIP();
            if (strcmp(location, localIP.toStdString().c_str()) == 0)
            {

                QString qoldPath = qPath;

                FileSysAccess fileSys;
                list = fileSys.getFileSysList(qFilter, qPath, true);

                if (!(qoldPath == qPath))
                {
                    TokenBuffer rtb;
                    rtb << TABLET_SET_CURDIR;
                    rtb << id;
                    std::string tempStr = qPath.toStdString();
                    char *pathTemp = (char *)tempStr.c_str();
                    rtb << pathTemp;

                    Message m(rtb);
                    m.type = COVISE_MESSAGE_VRB_FB_SET;

                    //Send message
                    receiver->conn->send_msg(&m);
                }

                TokenBuffer tb2;
                //Create new message
                tb2 << TABLET_SET_FILELIST;
                tb2 << id;
                int d1 = list.size();
                tb2 << d1;
                for (int i = 0; i < d1; i++)
                {
                    std::string tempStr = list.at(i).toStdString();
                    char *temp = (char *)tempStr.c_str();
                    tb2 << temp;
                }

                Message m2(tb2);
                m2.type = COVISE_MESSAGE_VRB_FB_SET;

                //Send message
                receiver->conn->send_msg(&m2);
            }
            else
            {
                //Determine destination's VRBSClient object and
                //route the request to the appropriate client
                RerouteRequest(location, TABLET_SET_FILELIST, id, recvId, qFilter, qPath);
            }
        }
        break;
        case TABLET_REQ_DRIVES:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request Drives!" << std::endl;
#endif
            tb >> filter;
            tb >> path;
            tb >> location;

            QString qFilter = filter;
            QString qPath = path;

            // not used bool sendDir = false;
            NetHelp networkHelper;
            QString localIP = networkHelper.getLocalIP();
            if (strcmp(location, localIP.toStdString().c_str()) == 0)
            {

                //Determine available drives
                QFileInfoList list = QDir::drives();

                TokenBuffer tb2;
                //Create new message
                tb2 << TABLET_SET_DRIVES;
                tb2 << id;
                tb2 << list.size();
                for (int i = 0; i < list.size(); i++)
                {
                    QFileInfo info = list.at(i);
                    QString dir = info.absolutePath();
                    std::string sdir = dir.toStdString();
                    char *temp = (char *)sdir.c_str();
                    tb2 << temp;
                }

                Message m2(tb2);
                m2.type = COVISE_MESSAGE_VRB_FB_SET;

                //Send message
                receiver->conn->send_msg(&m2);
            }
            else
            {
                //Determine destination's VRBSClient object and
                //route the request to the appropriate client
                RerouteRequest(location, TABLET_SET_DRIVES, id, recvId, qFilter, qPath);
            }
        }
        break;
        case TABLET_REQ_CLIENTS:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request Clients!" << std::endl;
#endif
            QStringList tuiClientList;
            QString locClient;
            QString locClientName;

            //Determine client list connected to VRB
            for (int i = clients.num(); i > 0;)
            {
                VRBSClient *locConn = clients.item(--i);
                locClientName = locConn->getName();
                locClient = locConn->getIP();
                tuiClientList.append(locClient);
            }

            //Now send gathered client list to requesting client
            TokenBuffer tb2;
            //Create new message
            tb2 << TABLET_SET_CLIENTS;
            tb2 << id;
            tb2 << tuiClientList.size() + 1;
            for (int i = 0; i < tuiClientList.size(); i++)
            {
                std::string tempStr = tuiClientList.at(i).toStdString();
                char *temp = (char *)tempStr.c_str();
                tb2 << temp;
            }

            NetHelp networkHelper;
            QString serverIP = networkHelper.getLocalIP();
            tb2 << serverIP.toStdString().c_str();

            QHostInfo host;
            QString hostName;
            std::string strHostName;

            for (int i = 0; i < tuiClientList.size(); i++)
            {
                QString address(tuiClientList.at(i));
                hostName = networkHelper.GetNamefromAddr(&address);
                strHostName = hostName.toStdString();
                char *temp = (char *)strHostName.c_str();
                tb2 << temp;
            }

            //tb2 << (gethostbyaddr(this->getLocalIP().toAscii().data(),4,AF_INET))->h_name;
            hostName = networkHelper.GetNamefromAddr(&serverIP);
            hostName = hostName + "(VRB-Server)";
            strHostName = hostName.toStdString();
            tb2 << strHostName.c_str();

            Message m2(tb2);
            m2.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            receiver->conn->send_msg(&m2);

            break;
        }
        case TABLET_FB_FILE_SEL:
        {
            tb >> location;
            tb >> path;

#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request File!" << std::endl;
            std::cerr << "File: " << path << std::endl;
#endif

            // not used bool sendDir = false;
            NetHelp networkHelper;
            QString localIP = networkHelper.getLocalIP();
            if (strcmp(location, localIP.toStdString().c_str()) == 0)
            {

                //Open local file and put in binary buffer
                QString qpath(path);

                QDir dir;

                QStringList list = qpath.split(dir.separator());
                QString file = list.last();

                int bytes = 0;
                std::ifstream vrbFile;
                //Currently opens files in
                vrbFile.open(path, std::ifstream::binary);
                if (vrbFile.fail())
                {
                    //TODO: Evaluate and send NoSuccess message
                    TokenBuffer tb2;
                    tb2 << TABLET_SET_FILE_NOSUCCESS;
                    tb2 << id;
                    tb2 << "Open failed!";
                    std::cerr << "Opening of file for submission failed!" << std::endl;
                    Message m(tb2);
                    m.type = COVISE_MESSAGE_VRB_FB_SET;
                    //Send message
                    receiver->conn->send_msg(&m);
                    return;
                }
                vrbFile.seekg(0, std::ios::end);
                bytes = vrbFile.tellg();
                char *data = new char[bytes];
                vrbFile.seekg(std::ios::beg);
                vrbFile.read(data, bytes);
                if (vrbFile.fail())
                {
                    //TODO: Evaluate and send NoSuccess message
                    TokenBuffer tb2;
                    tb2 << TABLET_SET_FILE_NOSUCCESS;
                    tb2 << id;
                    tb2 << "Read failed!";
                    std::cerr << "Reading of file for submission failed!" << std::endl;
                    Message m(tb2);
                    m.type = COVISE_MESSAGE_VRB_FB_SET;
                    //Send message
                    receiver->conn->send_msg(&m);
                    return;
                }
                vrbFile.close();

#ifdef MB_DEBUG
                std::cerr << "Start file send!" << std::endl;
#endif

                TokenBuffer tb2; //Create new message
                tb2 << TABLET_SET_FILE;
                tb2 << id;
                std::string sfile = file.toStdString();
                tb2 << sfile.c_str();
                tb2 << bytes;
                tb2.addBinary(data, bytes);

                delete[] data;

                Message m(tb2);
                m.type = COVISE_MESSAGE_VRB_FB_SET;

                //Send message
                receiver->conn->send_msg(&m);
#ifdef MB_DEBUG
                std::cerr << "End file send!" << std::endl;
#endif
            }
            else
            {
//Determine destination's VRBSClient object and
//route the request to the appropriate client
#ifdef MB_DEBUG
                std::cerr << "Entered reroute of request to OC Client!" << std::endl;
#endif

                QString qFilter = filter;
                QString qPath = path;
                RerouteRequest(location, TABLET_FB_FILE_SEL, id, recvId, qFilter, QString(path));
#ifdef MB_DEBUG
                std::cerr << "Finished reroute of request to OC Client!" << std::endl;
#endif
            }
            break;
        }
        case TABLET_REQ_GLOBALLOAD:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request GlobalLoad!" << std::endl;
#endif
            char *curl = NULL;
            tb >> curl;

            TokenBuffer tb2;

            //Create new message
            tb2 << TABLET_SET_GLOBALLOAD;
            tb2 << id;
            tb2 << curl;

            Message m2(tb2);
            m2.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            for (int i = clients.num(); i > 0;)
            {
                VRBSClient *locConn = clients.item(--i);
                if (locConn->conn != msg->conn)
                {
                    locConn->conn->send_msg(&m2);
                }
            }
            break;
        }
        case TABLET_REQ_HOMEDIR:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request HOMEDIR!" << std::endl;
#endif
            tb >> location;

            // not used bool sendDir = false;
            NetHelp networkHelper;
            QString localIP = networkHelper.getLocalIP();
            if (strcmp(location, localIP.toStdString().c_str()) == 0)
            {

                FileSysAccess file;
                QStringList dirList = file.getLocalHomeDir();

                TokenBuffer ltb;
                ltb << TABLET_SET_DIRLIST;
                ltb << id;
                ltb << dirList.size();

                for (int i = 0; i < dirList.size(); i++)
                {
                    QString qentry = dirList.at(i);
                    std::string sentry = qentry.toStdString();
                    ltb << sentry.c_str();
                }

                Message m2(ltb);
                m2.type = COVISE_MESSAGE_VRB_FB_SET;

                //Send message
                receiver->conn->send_msg(&m2);

                ltb.delete_data();
                ltb << TABLET_SET_CURDIR;
                ltb << id;
                std::string shome = file.getLocalHomeDirStr().toStdString();
                ltb << shome.c_str();

                Message m3(ltb);
                m3.type = COVISE_MESSAGE_VRB_FB_SET;
                receiver->conn->send_msg(&m3);
            }
            else
            {
                // TODO: HomeDir retrieval from OC client
                //Determine destination's VRBSClient object and
                //route the request to the appropriate client
                RerouteRequest(location, TABLET_REQ_HOMEDIR, id, recvId, QString(filter), QString(path));
            }
            break;
        }

        case TABLET_REQ_HOMEFILES:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Request HOMEFILES!" << std::endl;
#endif
            tb >> filter;
            tb >> location;

            // not used bool sendDir = false;
            NetHelp networkHelper;
            QString localIP = networkHelper.getLocalIP();
            if (strcmp(location, localIP.toStdString().c_str()) == 0)
            {

                FileSysAccess file;
                QStringList fileList = file.getLocalHomeFiles(filter);

                TokenBuffer ltb;
                ltb << TABLET_SET_FILELIST;
                ltb << id;
                ltb << fileList.size();

                for (int i = 0; i < fileList.size(); i++)
                {
                    QString qentry = fileList.at(i);
                    std::string sentry = qentry.toStdString();
                    ltb << sentry.c_str();
                }

                Message m2(ltb);
                m2.type = COVISE_MESSAGE_VRB_FB_SET;

                //Send message
                receiver->conn->send_msg(&m2);

                //Send message
                receiver->conn->send_msg(&m2);
            }
            else
            {
                // TODO: HomeFiles retrieval from OC client
                //Determine destination's VRBSClient object and
                //route the request to the appropriate client
                RerouteRequest(location, TABLET_REQ_HOMEFILES, id, recvId, QString(filter), QString(path));
            }
            break;
        }
        default:
            cerr << "Unknown FileBrowser request!" << endl;
            break;
        }
    }
    break;
    case COVISE_MESSAGE_VRB_FB_REMREQ:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest!" << std::endl;
#endif
        int id = 0;
        int receiverId = 0;
        int type = 0;

        //Message for tabletui filebrowser request
        TokenBuffer tb(msg);
        tb >> type;
        tb >> id;
        tb >> receiverId;

        VRBSClient *receiver = clients.get(receiverId);

        switch (type)
        {
        case TABLET_REMSET_FILELIST:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest FileList!" << std::endl;
#endif
            int size = 0;
            char *entry = NULL;

            tb >> size;

            TokenBuffer tb2;
            //Create new message
            tb2 << TABLET_SET_FILELIST;
            tb2 << id;
            tb2 << size;

            for (int i = 0; i < size; i++)
            {
                tb >> entry;
                tb2 << entry;
            }

            Message m2(tb2);
            m2.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            receiver->conn->send_msg(&m2);
        }
        break;
        case TABLET_REMSET_FILE_NOSUCCESS:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest NoSuccess!" << std::endl;
#endif
            char *comment = NULL;

            tb >> comment;

            TokenBuffer tb2;
            //Create new message
            tb2 << TABLET_SET_FILE_NOSUCCESS;
            tb2 << id;
            tb2 << comment;

            Message m2(tb2);
            m2.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            receiver->conn->send_msg(&m2);
        }
        break;
        case TABLET_REMSET_DIRLIST:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest DirList!" << std::endl;
#endif
            int size = 0;
            char *entry = NULL;

            tb >> size;

            TokenBuffer tb2;
            //Create new message
            tb2 << TABLET_SET_DIRLIST;
            tb2 << id;
            tb2 << size;

            for (int i = 0; i < size; i++)
            {
                tb >> entry;
                tb2 << entry;
            }

            Message m2(tb2);
            m2.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            receiver->conn->send_msg(&m2);
        }
        break;
        case TABLET_REMSET_FILE:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest FileRequest!" << std::endl;
#endif
            char *filename = NULL;
            int size = 0;
            tb >> filename;
            tb >> size;

            TokenBuffer rt;
            rt << TABLET_SET_FILE;
            rt << id;
            rt << filename;
            rt << size;
            rt.addBinary(tb.getBinary(size), size);

            Message m(rt);
            m.type = COVISE_MESSAGE_VRB_FB_SET;
            receiver->conn->send_msg(&m);
        }
        break;
        case TABLET_REMSET_DIRCHANGE:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest DirChange!" << std::endl;
#endif
            char *cpath = NULL;
            tb >> cpath;
            TokenBuffer rtb;
            rtb << TABLET_SET_CURDIR;
            rtb << id;
            rtb << cpath;

            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            receiver->conn->send_msg(&m);
        }
        break;
        case TABLET_REMSET_DRIVES:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest DriveList!" << std::endl;
#endif
            int size = 0;
            char *entry = NULL;

            tb >> size;

            TokenBuffer tb2;
            //Create new message
            tb2 << TABLET_SET_DRIVES;
            tb2 << id;
            tb2 << size;

            for (int i = 0; i < size; i++)
            {
                tb >> entry;
                tb2 << entry;
            }

            Message m2(tb2);
            m2.type = COVISE_MESSAGE_VRB_FB_SET;

            //Send message
            receiver->conn->send_msg(&m2);
        }
        break;
        default:
        {
#ifdef MB_DEBUG
            std::cerr << "::HANDLECLIENT VRB FileBrowser Unknown RemoteRequest!" << std::endl;
#endif
            cerr << "unknown VRB FileBrowser RemoteRequest message in vrb" << endl;
        }
        }
    }
    break;
    default:
        cerr << "unknown message in vrb: type=" << msg->type << endl;
        break;
    }
}

void VRBServer::RerouteRequest(const char *location, int type, int senderId, int recvVRBId, QString filter, QString path)
{
    VRBSClient *locClient = NULL;
    char *host = NULL;

    // not used int t = clients.num();
    for (int i = 0; i < clients.num(); i++)
    {
        host = (char *)(clients.item(i))->getIP();
        if (strcmp(location, host) == 0)
        {
            locClient = clients.item(i);
        }
    }

    TokenBuffer tb;
    tb << type;
    tb << senderId;
    tb << recvVRBId;
    std::string spath = path.toStdString();
    tb << spath.c_str();
    std::string sfilter = filter.toStdString();
    tb << sfilter.c_str();

    Message msg(tb);
    msg.type = COVISE_MESSAGE_VRB_FB_REMREQ;

    if (locClient != NULL)
    {
        locClient->conn->send_msg(&msg);
    }
}
