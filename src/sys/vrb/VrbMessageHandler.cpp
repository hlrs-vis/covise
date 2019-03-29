/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include"VrbMessageHandler.h"
#include "VRBClientList.h"
#include "VrbServerRegistry.h"

#include <vrbclient/SessionID.h>
#include <vrbclient/SharedStateSerializer.h>

#include <util/coTabletUIMessages.h>
#include <util/unixcompat.h>

#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <net/covise_connect.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>

#include <sys/types.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <qtutil/NetHelp.h>
#include <qtutil/FileSysAccess.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qdir.h>
#include <QtNetwork/qhostinfo.h>

#ifdef GUI
#include "gui/VRBapplication.h"
#include "gui/coRegister.h"
#endif



using namespace std;
using namespace covise;
namespace vrb
{
VrbMessageHandler::VrbMessageHandler(ServerInterface * s)
    :m_server(s)
{
}
void VrbMessageHandler::handleMessage(Message *msg)
{
    char *Class;
    vrb::SessionID sessionID;
    int senderID;
    int modeID; // =senderID for looseCoupling, =0 for observe all, else =-1  
    char *variable;
    char *value;
    int fd;
    TokenBuffer tb(msg);
    TokenBuffer tb_value;

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
            tb >> senderID;
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
                rtb3 << currentFileClient->getID();
                rtb3 << currentFile;
                Message m(rtb3);
                m.conn = msg->conn;
                m.type = COVISE_MESSAGE_VRB_CURRENT_FILE;
                c->conn->send_msg(&m);
#ifdef MB_DEBUG
                std::cerr << "====> Send done!" << std::endl;
#endif
                clients.passOnMessage(&m);

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
        tb >> sessionID;
        tb >> senderID;
        tb >> Class;
        tb >> variable;
        tb >> tb_value;

        sessions[sessionID]->setVar(senderID, Class, variable, tb_value); //maybe handle no existing registry for the given session ID

#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set registry value, class=" << Class << ", name=" << variable << std::endl;
#endif


#ifdef GUI
        if (std::strcmp(Class, "SharedState") != 0)
        {
            tb_value >> value;
            m_server->getAW()->registry->updateEntry(Class, senderID, variable, value);
        }
        else
        {
            m_server->getAW()->registry->updateEntry(Class, senderID, variable, vrb::tokenBufferToString(std::move(tb_value)).c_str());
        }
#endif
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS:
    {
        tb >> sessionID;
        tb >> senderID;
        tb >> Class;
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry subscribe class=" << Class << std::endl;
#endif
        sessions[sessionID]->observeClass(senderID, Class);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE:
    {
        tb >> sessionID;
        tb >> senderID;
        tb >> Class;
        tb >> variable;
        tb >> tb_value;
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry subscribe variable=" << variable << ", class=" << Class << std::endl;
#endif
        createSessionIfnotExists(sessionID, senderID)->observeVar(senderID, Class, variable, tb_value);
#ifdef GUI
        if (std::strcmp(Class, "SharedState") != 0)
        {
            tb_value >> value;
            m_server->getAW()->registry->updateEntry(Class, senderID, variable, value);
        }
        else
        {
            m_server->getAW()->registry->updateEntry(Class, senderID, variable, vrb::tokenBufferToString(std::move(tb_value)).c_str());
        }
#endif
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS:
    {
        tb >> sessionID;
        tb >> senderID;
        tb >> Class;
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry unsubscribe class=" << Class << std::endl;
#endif
        sessions[sessionID]->unObserveClass(senderID, Class);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE:
    {
        tb >> senderID;
        tb >> Class;
        tb >> variable;

#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry unsubscribe variable=" << name << ", class=" << Class << std::endl;
#endif
        auto it = sessions.begin();
        while (it != sessions.end())
        {
            it->second->unObserveVar(senderID, Class, variable);
            ++it;
        }
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry create entry!" << std::endl;
#endif
        bool isStatic;
        tb >> sessionID;
        tb >> senderID;
        tb >> Class;
        tb >> variable;
        tb >> tb_value;
        tb >> isStatic;
        createSessionIfnotExists(sessionID, senderID)->create(senderID, Class, variable, tb_value, isStatic);
#ifdef GUI
        m_server->getAW()->registry->updateEntry(Class, senderID, variable, "");
#endif
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry delete entry!" << std::endl;
#endif
        tb >> sessionID;
        tb >> Class;
        tb >> variable;
        sessions[sessionID]->deleteEntry(Class, variable);
#ifdef GUI
        m_server->getAW()->registry->removeEntry(Class, senderID, variable);
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
                vrb::SessionID sid(c->getID(), "none");
                c->setContactInfo(ip, name, createSession(sid));
            }
#else
            clients.addClient(new VRBSClient(msg->conn, ip, name));
            std::cerr << "VRB new client: Numclients=" << clients.numberOfClients() << std::endl;
#endif
        }
        //cerr << name << " connected from " << ip << endl;
        // send a list of all participants to all clients

        sendSessions();
    }
    break;
    case COVISE_MESSAGE_VRB_CONNECT_TO_COVISE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Connect Covise!" << std::endl;
#endif
        char *ip;
        tb >> ip;
        if (VRBSClient *c = clients.get(ip))
        {
            c->conn->send_msg(msg);
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
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            TokenBuffer rtb;
            rtb << 1;
            c->setUserInfo(ui);
            c->getInfo(rtb);
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
            m.conn = c->conn;
            clients.passOnMessage(&m);
        }
        TokenBuffer rtb2;
        clients.collectClientInfo(rtb2);
        Message m(rtb2);
        m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
        c->conn->send_msg(&m);

        // send current file name to new client
        if (currentFile && currentFileClient)
        {
            TokenBuffer rtb3;
            rtb3 << currentFileClient->getID();
            rtb3 << currentFile;
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
        std::cerr << "::HANDLECLIENT VRB Render/Render Module of length " << msg->length << "!" << std::endl;
#endif
        vrb::SessionID toGroup;
#ifdef MB_DEBUG
        //std::cerr << "====> Get senders vrbc connection!" << std::endl;
#endif

        VRBSClient *c = clients.get(msg->conn);

        if (c)
        {
#ifdef MB_DEBUG
            //std::cerr << "====> Sender found!" << std::endl;
            //std::cerr << "====> Sender: " << c->getIP() << std::endl;
#endif
            toGroup = c->getSession();
            c->addBytesSent(msg->length);
#ifdef MB_DEBUG
            //std::cerr << "====> Increased amount of sent bytes!" << std::endl;
#endif
        }
#ifdef MB_DEBUG
        //std::cerr << "====> Reset clients!" << std::endl;
#endif

#ifdef MB_DEBUG
        //std::cerr << "====> Iterate through all vrbcs!" << std::endl;
#endif
        clients.passOnMessage(msg, toGroup);
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
        VRBSClient *c = clients.get(ip);
        if (c)
        {
            rtb << 1;
        }
        else
        {
            rtb << 0;
        }
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
        VRBSClient *c = clients.get(msg->conn);
        int clId = -1;
        if (c)
        {
            clId = c->getID();
        }
        rtb << clId;
        vrb::SessionID sid(clId, "anon");
        rtb << createSession(sid);
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
        vrb::SessionID newSession;
        tb >> newSession;
        if (newSession.isPrivate())
        {
            return;
        }
        VRBSClient *c, *c2;
        c = clients.get(msg->conn);
        bool becomeMaster;
        if (c)
        {
            if ((c2 = clients.getMaster(newSession)) && c->getMaster()) //there is already a master
            {
                becomeMaster = false;
            }
            else
            {
                becomeMaster = true;
            }
            TokenBuffer mtb;
            mtb << c->getID();
            mtb << becomeMaster;
            Message m(mtb);
            m.type = COVISE_MESSAGE_VRB_SET_MASTER;
            c->setMaster(becomeMaster);
            c->conn->send_msg(&m);
            //inform others about new session
            c->setSession(newSession);
            TokenBuffer rtb;
            rtb << c->getID();
            rtb << newSession;
            Message mg(rtb);
            mg.type = COVISE_MESSAGE_VRB_SET_GROUP;
            mg.conn = c->conn;
            clients.passOnMessage(&mg);
        }
    }
    break;
    case COVISE_MESSAGE_VRB_SET_MASTER:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set Master!" << std::endl;
#endif
        bool masterState;
        tb >> masterState;
        if (!masterState)
            break;
        if (VRBSClient *c = clients.get(msg->conn))
        {
            clients.setMaster(c);
            TokenBuffer rtb;
            rtb << c->getID();
            rtb << masterState;
            Message  rmsg(rtb);
            rmsg.type = COVISE_MESSAGE_VRB_SET_MASTER;
            rmsg.conn = c->conn;
            clients.passOnMessage(&rmsg, c->getSession());
        }
    }
    break;
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Close Socket!" << std::endl;
#endif
        m_server->removeConnection(msg->conn);
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            disconectClientFromSessions(c);
            if (currentFileClient == c)
                currentFileClient = NULL;
            bool wasMaster = false;
            vrb::SessionID MasterSession;
            TokenBuffer rtb;
            cerr << c->getName() << " (host " << c->getIP() << " ID: " << c->getID() << ") left" << endl;
            rtb << c->getID();
            if (c->getMaster())
            {
                MasterSession = c->getSession();
                cerr << "Master Left" << endl;
                wasMaster = true;
            }
#ifdef GUI
            m_server->getAW()->registry->removeEntries(c->getID());
#endif
            clients.removeClient(c);
            delete c;
            clients.sendMessageToAll(rtb, COVISE_MESSAGE_VRB_QUIT);
            if (wasMaster)
            {
                VRBSClient *newMaster = clients.getNextInGroup(MasterSession);
                if (newMaster)
                {
                    clients.setMaster(newMaster);
                    TokenBuffer rtb;
                    rtb << newMaster->getID();
                    rtb << true;
                    clients.sendMessage(rtb, MasterSession, COVISE_MESSAGE_VRB_SET_MASTER);
                }
                else
                {
                    //save & delete Masersession
                }
            }
        }
        else
            cerr << "CRB left" << endl;

        cerr << "Numclients: " << clients.numberOfClients() << endl;
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
            for (int i = clients.numberOfClients(); i > 0;)
            {
                VRBSClient *locConn = clients.getNthClient(--i);
                locClientName = QString::fromStdString(locConn->getName());
                locClient = QString::fromStdString(locConn->getIP());
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
                const char *temp = strHostName.c_str();
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
            for (int i = clients.numberOfClients(); i > 0;)
            {
                VRBSClient *locConn = clients.getNthClient(--i);
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
    case COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION:
    {
        tb >> sessionID;
        vrb::SessionID newSessionID = createSession(sessionID);
        sessions[newSessionID]->setOwner(sessionID.owner());
        sendSessions();
        setSession(newSessionID);


    }
    break;
    case COVISE_MESSAGE_VRBC_UNOBSERVE_SESSION:
    {
        tb >> sessionID;
        tb >> senderID;
        auto session = sessions.find(sessionID);
        if (session != sessions.end())
        {
            session->second->unObserve(senderID);
        }
    }
    break;
    case COVISE_MESSAGE_VRBC_SET_SESSION:
    {
        tb >> sessionID;
        tb >> senderID;

        VRBSClient *c = clients.get(senderID);
        if (!c)
        {
            cerr << "client " << senderID << " not found" << endl;
            return;
        }
        c->setSession(sessionID);

        TokenBuffer rtb;
        rtb << c->getID();
        rtb << sessionID;
        Message mg(rtb);
        mg.type = COVISE_MESSAGE_VRB_SET_GROUP;
        mg.conn = c->conn;
        clients.passOnMessage(&mg);

    }
    break;
    case COVISE_MESSAGE_VRB_REQUEST_SAVED_SESSION:
    {
        tb >> senderID;
        std::set<std::string> files = getFilesInDir(home() + "/Sessions", ".vrbreg");
        TokenBuffer ftb;
        ftb << int(files.size());
        for (const auto file : files)
        {
            ftb << file;
        }
        clients.sendMessageToID(ftb, senderID, COVISE_MESSAGE_VRB_REQUEST_SAVED_SESSION);
    }
    break;
    case COVISE_MESSAGE_VRB_SAVE_SESSION:
    {
        tb >> sessionID;
        auto ses = sessions.find(sessionID);
        if (ses == sessions.end())
        {
            std::cout << "can't save a session that not exists" << std::endl;
        }
        else
        {
            ses->second->saveFile(std::string(home() + "/Sessions"));
        }
        std::set<std::string> files = getFilesInDir(home() + "/Sessions", ".vrbreg");
        TokenBuffer ftb;
        int size = files.size();
        ftb << size;
        for (std::string file : files)
        {
            ftb << file;
        }
        clients.sendMessageToAll(ftb, COVISE_MESSAGE_VRB_REQUEST_SAVED_SESSION);
    }
    break;
    case COVISE_MESSAGE_VRB_LOAD_SESSION:
    {
        std::string filename;
        tb >> senderID;
        tb >> sessionID;
        tb >> filename;
        sessions[sessionID]->loadFile(std::string(home() + "/Sessions/") + filename);
        TokenBuffer stb;
        stb << sessionID;
        if (sessionID.isPrivate())//inform the owner of the private session 
        {
            clients.sendMessageToID(stb, senderID, COVISE_MESSAGE_VRB_LOAD_SESSION);
        }
        else //inform all session members
        {
            clients.sendMessage(stb, sessionID, COVISE_MESSAGE_VRB_LOAD_SESSION);
        }

    }
    break;
    case COVISE_MESSAGE_VRB_MESSAGE:
    {
        vrb::SessionID toGroup;
        VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            toGroup = c->getSession();
            c->addBytesSent(msg->length);
        }
        clients.passOnMessage(msg, toGroup);
    }
    break;
    default:
        cerr << "unknown message in vrb: type=" << msg->type << endl;
        break;


    }
}
void VrbMessageHandler::closeConnection()
{
    TokenBuffer tb;
    //requestToQuit = true;
    clients.sendMessageToAll(tb, COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION);
}
vrb::SessionID & VrbMessageHandler::createSession(vrb::SessionID & id)
{
    if (id.name() == std::string()) //unspecific name -> create generic name here
    {
        int genericName = 1;
        id.setName("1");
        while (sessions.find(id) != sessions.end())
        {
            ++genericName;
            id.setName(std::to_string(genericName));

        }

    }
    if (sessions.find(id) == sessions.end())
    {
        sessions[id].reset(new VrbServerRegistry(id));
    }
    return id;
}
std::shared_ptr<vrb::VrbServerRegistry> VrbMessageHandler::createSessionIfnotExists(vrb::SessionID & sessionID, int senderID)
{
    auto ses = sessions.find(sessionID);
    if (ses == sessions.end())
    {
        VRBSClient *cl = clients.get(senderID);
        if (cl && !sessionID.isPrivate())
        {
            cl->setSession(sessionID);
        }
        sessions[sessionID].reset(new VrbServerRegistry(sessionID));
    }
    return sessions[sessionID];
}
void VrbMessageHandler::sendSessions()
{
    TokenBuffer mtb;
    std::set<vrb::SessionID> publicSessions;
    for (const auto session : sessions)
    {
        if (session.first != vrb::SessionID(0, std::string(), false))
        {
            publicSessions.insert(session.first);
        }
    }
    int size = publicSessions.size();
    mtb << size;
    for (const auto session : publicSessions)
    {
        mtb << session;
    }
    if (size > 0)
    {
        clients.sendMessageToAll(mtb, COVISE_MESSAGE_VRBC_SEND_SESSIONS);
    }
}
void VrbMessageHandler::RerouteRequest(const char * location, int type, int senderId, int recvVRBId, QString filter, QString path)
{
    VRBSClient *locClient = NULL;

    for (int i = 0; i < clients.numberOfClients(); i++)
    {
        std::string host = clients.getNthClient(i)->getIP();
        if (host == location)
        {
            locClient = clients.getNthClient(i);
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
std::string VrbMessageHandler::home()
{
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string::size_type pos = std::string(buffer).find_last_of("\\/");
    return std::string(buffer).substr(0, pos);
#else
    if (auto h = getenv("HOME"))
        return h;

    return "/";
#endif
}
std::set<std::string> VrbMessageHandler::getFilesInDir(const std::string & path, const std::string & fileEnding) const
{
    boost::filesystem::path p(path);
    std::set<std::string> files;
    if (is_directory(p))
    {
        for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
            if (fileEnding == "" || entry.path().filename().generic_string().find(fileEnding) != std::string::npos)
            {
                files.insert(entry.path().filename().generic_string());
            }
        }
    }
    else
    {
        std::cerr << "Directory not found: " << path << std::endl;
    }
    return files;
}
void VrbMessageHandler::disconectClientFromSessions(VRBSClient * cl)
{
    if (!cl)
    {
        return;
    }
    auto it = sessions.begin();
    while (it != sessions.end())
    {
        if (it->first.owner() == cl->getID())
        {
            if (it->first.isPrivate())
            {
                it = sessions.erase(it);
            }
            else
            {
                vrb::SessionID newId(it->first);
                VRBSClient *newOwner = clients.getNextInGroup(it->first);
                if (newOwner)
                {
                    newId.setOwner(newOwner->getID());
                    sessions[newId] = it->second;
                }
                //int newOwner;
                bool first = false;
                sendSessions();
                TokenBuffer stb;
                stb << newId;
                clients.sendMessage(stb, it->first, COVISE_MESSAGE_VRBC_SET_SESSION);
                it = sessions.erase(it);
            }
        }
        else
        {
            ++it;
        }


    }
    sendSessions();
}
void VrbMessageHandler::setSession(vrb::SessionID & sessionId)
{
    TokenBuffer stb;
    stb << sessionId;
    clients.sendMessageToID(stb, sessionId.owner(), COVISE_MESSAGE_VRBC_SET_SESSION);
}

}

