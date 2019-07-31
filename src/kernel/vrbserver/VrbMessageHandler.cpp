/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VrbMessageHandler.h"
#include "VrbClientList.h"
#include "VrbServerRegistry.h"

#include <vrbclient/SessionID.h>
#include <vrbclient/SharedStateSerializer.h>

#include <util/coTabletUIMessages.h>
#include <util/unixcompat.h>

#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <net/udp_message_types.h>
#include <net/udpMessage.h>
#include <net/covise_connect.h>
#include <net/dataHandle.h>

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
#include <boost/algorithm/string.hpp>

#include <QDir>
#include <qtutil/NetHelp.h> 
#include <qtutil/FileSysAccess.h>
#include <QtNetwork/qhostinfo.h>


using namespace std;
using namespace covise;
namespace vrb
{
VrbMessageHandler::VrbMessageHandler(ServerInterface *server)
    :m_server(server)
{
    SessionID id(0, std::string(), false);
    sessions[id].reset(new VrbServerRegistry(id));
}
void VrbMessageHandler::handleMessage(Message *msg)
{
    char *Class;
    vrb::SessionID sessionID;
    int senderID;
    char *variable;
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
        vrb::VRBSClient *c = clients.get(msg->conn);
        if (c)
        {
            c->addBytesSent(msg->data.length());
        }
        std::string filename;
        tb >> filename;
        int requestorsID, fileOwner;
        tb >> requestorsID;
		tb >> fileOwner;
        TokenBuffer rtb;
		//file found local on the vrb server
        if (stat(filename.c_str(), &statbuf) >= 0)
        {
            rtb << requestorsID;
			rtb << std::string(filename);
            if ((fd = open(filename.c_str(), O_RDONLY)) > 0)
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
                c->addBytesReceived(m.data.length());
            }
            msg->conn->send_msg(&m);
        }
        else
        {
            // forward this request to the COVER that loaded the requested file
            VRBSClient* currentFileClient = clients.get(fileOwner);
            if ((currentFileClient) && (currentFileClient != c) && (currentFileClient->getID() != requestorsID))
            {
                currentFileClient->conn->send_msg(msg);
                currentFileClient->addBytesReceived(msg->data.length());
            }
            else
            {
                rtb << requestorsID;
				rtb << filename;
                rtb << 0; // file not found
                Message m(rtb);
                m.type = COVISE_MESSAGE_VRB_SEND_FILE;
                if (c)
                {
                    c->addBytesReceived(m.data.length());
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
            c->addBytesSent(msg->data.length());
        }
        int destinationID, buffersize;
		std::string fileName;
        tb >> destinationID;
		tb >> fileName;
		tb >> buffersize;
		if (buffersize == 0) //send request to next client in session
		{
			c->addUnknownFile(fileName);
			VRBSClient* nextCl = clients.getNextPossibleFileOwner(fileName, c->getSession());
			if (nextCl)
			{
				TokenBuffer rtb;
				rtb << fileName;
				rtb << destinationID;
				Message rmsg(rtb);
				rmsg.type = COVISE_MESSAGE_VRB_REQUEST_FILE;
				nextCl->conn->send_msg(&rmsg);
				break;
			}
		}
        // forward this File to the one that requested it
        c = clients.get(destinationID);
        if (c)
        {
            c->conn->send_msg(msg);
            c->addBytesReceived(msg->data.length());
        }
    }
    break;
    case COVISE_MESSAGE_VRB_CURRENT_FILE:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Current file message!" << std::endl;
#endif
		assert(true);
		//        VRBSClient *c = clients.get(msg->conn);
//        if (c)
//        {
//#ifdef MB_DEBUG
//            std::cerr << "====> Client valid!" << std::endl;
//#endif
//            c->addBytesSent(msg->data.length());
//#ifdef MB_DEBUG
//            std::cerr << "====> Added length of msg to bytes sent!" << std::endl;
//#endif
//
//            delete[] currentFile;
//            char *buf;
//            tb >> senderID;
//            tb >> buf;
//            currentFile = new char[strlen(buf) + 1];
//
//            // send current file name to all clients
//#ifdef MB_DEBUG
//            std::cerr << "====> Sending current file to all clients!" << std::endl;
//#endif
//
//            if (currentFile)
//            {
//                TokenBuffer rtb3;
//                rtb3 << c->getID();
//                rtb3 << currentFile;
//                Message m(rtb3);
//                m.conn = msg->conn;
//                m.type = COVISE_MESSAGE_VRB_CURRENT_FILE;
//                c->conn->send_msg(&m);
//#ifdef MB_DEBUG
//                std::cerr << "====> Send done!" << std::endl;
//#endif
//                clients.passOnMessage(&m);
//
//#ifdef MB_DEBUG
//                std::cerr << "====> Iterate all vrb clients for current file xmit!" << std::endl;
//#endif
//                //delete[] currentFile;
//            }
//        }
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE: // Set Registry value
    {
        tb >> sessionID;
        tb >> senderID;
        tb >> Class;
        tb >> variable;
		DataHandle  valueData;
		deserialize(tb, valueData);


        sessions[sessionID]->setVar(senderID, Class, variable, valueData); //maybe handle no existing registry for the given session ID

#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Set registry value, class=" << Class << ", name=" << variable << std::endl;
#endif

        updateApplicationWindow(Class, senderID, variable, valueData);
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
        if (Class == "SharedMap")
        {
            cerr << "map" << endl;
        }
		DataHandle  valueData;
		tb >> valueData;
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Registry subscribe variable=" << variable << ", class=" << Class << std::endl;
#endif
        createSessionIfnotExists(sessionID, senderID)->observeVar(senderID, Class, variable, valueData);

		updateApplicationWindow(Class, senderID, variable, valueData);
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
		DataHandle  valueData;
		deserialize(tb, valueData);
        tb >> isStatic;
        createSessionIfnotExists(sessionID, senderID)->create(senderID, Class, variable, valueData, isStatic);
        updateApplicationWindow(Class, senderID, variable, valueData);
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

        removeEntryFromApplicationWindow(Class, senderID, variable);

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

        VRBSClient *c = clients.get(msg->conn);
        if (!c)
        {
            c = new VRBSClient(msg->conn, nullptr, ip, name, false,false);
            clients.addClient(c);
        }
        //create unique private session for the client
        vrb::SessionID sid = vrb::SessionID(c->getID(), "private");
        createSession(sid);

        c->setContactInfo(ip, name, sid);
        std::cerr << "VRB new client: Numclients=" << clients.numberOfClients() << std::endl;
        //if the client is started by covise (CRB) it has a muduleID that can be translated to sessionID
		//if OpenCOVER got a session from commandline (-s) the client will also send this given session name
        if (strcasecmp(name, "CRB") == 0 || (strcasecmp(name, "-g") == 0))
        {
            char *coviseModuleID; 
            tb >> coviseModuleID;
            //set session owner to the client that created the session
			vrb::SessionID sid = vrb::SessionID(c->getID(), std::string(coviseModuleID), false);
			auto it = sessions.find(sid);
			if (it != sessions.end())
			{
				sid.setOwner(it->first.owner());
			}
            createSessionIfnotExists(sid, c->getID());
            sendSessions();
            c->setSession(sid);
			m_sessionsToSet.insert(c->getID());
        }
        else
        {
            sendSessions();
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
		//if the client chose a starting session set it now 
		auto it = m_sessionsToSet.find(c->getID());
		if (it != m_sessionsToSet.end())
		{
			setSession(c->getSession(), c->getID());
			m_sessionsToSet.erase(it);
		}
    }
    break;
    case COVISE_MESSAGE_RENDER:
    case COVISE_MESSAGE_RENDER_MODULE: // send Message to all others in same group
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Render/Render Module of length " << msg->data.length() << "!" << std::endl;
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
            c->addBytesSent(msg->data.length());
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
        vrb::SessionID sid = vrb::SessionID(c->getID(), "private");
        createSession(sid);
        int clId = -1;
        if (c)
        {
            clId = c->getID();
            c->setPrivateSession(sid);
        }
        rtb << clId;

        rtb << sid;
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
	case COVISE_MESSAGE_QUIT:
    {
#ifdef MB_DEBUG
        std::cerr << "::HANDLECLIENT VRB Close Socket!" << std::endl;
#endif
		remove(msg->conn);
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

                ltb.reset();
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
        createSession(sessionID);
        sessions[sessionID]->setOwner(sessionID.owner());
        sendSessions();
        setSession(sessionID, sessionID.owner());


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
        std::set<std::string> files = getFilesInDir(home() + "/Sessions");
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
        saveSession(sessionID);

        //send information about all existing session files to clients
        std::set<std::string> files = getFilesInDir(home() + "/Sessions");
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
        loadSesion(filename, sessionID);

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
            c->addBytesSent(msg->data.length());
        }
        clients.passOnMessage(msg, toGroup);
    }
    break;
    default:

		cerr << "unknown message in vrb: type=" << msg->type;
		if (VRBSClient * c = clients.get(msg->conn))
		{
			cerr << " from " << c->getID();
		}
		cerr << endl;
        break;


    }
}
void VrbMessageHandler::closeConnection()
{
    TokenBuffer tb;
    clients.sendMessageToAll(tb, COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION);
}
int VrbMessageHandler::numberOfClients()
{
    return clients.numberOfClients();
}

void VrbMessageHandler::addClient(VRBSClient *client)
{
    clients.addClient(client);
}

/// remove client with connection c
void VrbMessageHandler::remove(Connection *conn)
{
	m_server->removeConnection(conn);
	VRBSClient *c = clients.get(conn);
	if (c)
	{
		int clID = c->getID();
		cerr << c->getName() << " (host " << c->getIP() << " ID: " << c->getID() << ") left" << endl;
		vrb::SessionID MasterSession;
		bool wasMaster = false;
		TokenBuffer rtb;

		rtb << c->getID();
		if (c->getMaster())
		{
			MasterSession = c->getSession();
			cerr << "Master Left" << endl;
			wasMaster = true;
		}
		clients.removeClient(c);
		delete c;
		disconectClientFromSessions(clID);

        removeEntriesFromApplicationWindow(clID);
		sendSessions();
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
}
void VrbMessageHandler::handleUdpMessage(vrb::UdpMessage* msg)
{
	VRBSClient* sender = clients.get(msg->sender);
	if (!sender)
	{
		cerr << "received udp message from unknown client wit ip = " << msg->m_ip << endl;
		return;
	}
	msg->conn = sender->conn;
	covise::TokenBuffer tb(msg);
	switch (msg->type)
	{
	case vrb::EMPTY:
		cerr << "received empty udp message" << endl;
		break;
	case vrb::AVATAR_HMD_POSITION:
	case vrb::AVATAR_CONTROLLER_POSITION:
	{
		//
	std:string m;
		tb >> m;
		cerr << "received udp message: " << m << endl;
		clients.passOnMessage(msg, sender->getSession());
	}
	break;
	default:
		break;
	}
}
void VrbMessageHandler::updateApplicationWindow(const char * cl, int sender, const char * var, const covise::DataHandle &value)
{
    //std::cerr <<"userinterface not implemented" << std:endl;
}
void VrbMessageHandler::removeEntryFromApplicationWindow(const char * cl, int sender, const char * var)
{
    //std::cerr <<"userinterface not implemented" << std:endl;
}
void VrbMessageHandler::removeEntriesFromApplicationWindow(int sender)
{
    //std::cerr <<"userinterface not implemented" << std:endl;
}


void VrbMessageHandler::createSession(vrb::SessionID & id)
{
    int genericName = 0;
    std::string name = id.name();
    if (name == std::string()) //unspecific name or already existing session -> create generic name here
    {
        ++genericName;
        id.setName("1");
    }
    while (sessions.find(id) != sessions.end())
    {
        ++genericName;
        id.setName(name+ std::to_string(genericName));

    }
    sessions[id].reset(new VrbServerRegistry(id));
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
void VrbMessageHandler::disconectClientFromSessions(int clID)
{
    auto it = sessions.begin();
    while (it != sessions.end())
    {
        if (it->first.owner() == clID)
        {
            if (it->first.isPrivate())
            {
                it = sessions.erase(it);
            }
            else
            {
                VRBSClient *newOwner = clients.getNextInGroup(it->first);
                if (newOwner)
                {
                    it->first.setOwner(newOwner->getID());
                }
				else
				{
					it = sessions.erase(it); //detele session if there are no more clients in it
				}
            }
        }
        else
        {
            ++it;
        }


    }
	sendSessions();
}
void VrbMessageHandler::setSession(vrb::SessionID & sessionId, int clID)
{
    TokenBuffer stb;
    stb << sessionId;
    clients.sendMessageToID(stb, clID, COVISE_MESSAGE_VRBC_SET_SESSION);
}
void VrbMessageHandler::saveSession(const vrb::SessionID &id)
{
    //get all participants of the sesision
    std::vector<VRBSClient *> participants;
    for (size_t i = 0; i < clients.numberOfClients(); i++)
    {
        if (clients.getNthClient(i)->getSession() == id)
        {
            participants.push_back(clients.getNthClient(i));
        }
    }
    //create directory archsuffix/bin/sessions/sessionName
    std::string path = home() + "/Sessions/" + id.name();
    if (boost::filesystem::create_directory(home() + "/Sessions"))
    {
        std::cerr << "Directory created: " << home() << "/Sessions" << std::endl;
    }
    int i = 2;
    std::string tryPath = path;
    while(!boost::filesystem::create_directory(tryPath))
    {
        tryPath = path + "(" + std::to_string(i) + ")";
        ++i;
    }
    std::cerr << "Directory created: " << path.c_str() << std::endl;
    std::ofstream outFile;
    //write private registries
    std::set<std::string> participantNames;
    i = 2;
    for (auto cl : participants)
    {
        std::string name = cl->getUserName();
        while (!participantNames.insert(name).second)
        {
            name += "_" + std::to_string(i);
            ++i;
        }
        outFile.open(tryPath + "/" + name + suffix, std::ios_base::binary);
        outFile << getTime() << "\n";
        sessions[cl->getPrivateSession()]->saveRegistry(outFile);
        outFile.close();
    }
    //write shared registry
    outFile.open(tryPath + "/" + id.name() + suffix, std::ios_base::binary);
    outFile << getTime() << "\n";
    outFile << "Number of participants : " << participants.size() << "\n";
    for (auto cl : participantNames)
    {
        outFile << cl << "\n";
    }
    sessions[id]->saveRegistry(outFile);
    outFile.close();
}
void VrbMessageHandler::loadSesion(const std::string &name, const vrb::SessionID &currentSession)
{
    //open file of shared session
    std::vector<std::string> fragments;
    boost::split(fragments, name, [](char c) {return c == '('; });
    std::string sessionName = fragments[0];
    std::string path = home() + "/Sessions/" + name + "/" + sessionName + suffix;
    std::ifstream inFile;
    inFile.open(path, std::ios_base::binary);
    if (inFile.fail())
    {
        std::cerr << "can not load file: file does not exist" << std::endl;
        return;
    }
    //read participants 
    std::string trash;
    int numberOfParticipants;
    std::vector<VRBSClient *> participants;
    std::map <std::string, std::vector<std::string>> paricipantFileNames;
    std::getline(inFile, trash, ':');
    inFile >> numberOfParticipants;
    for (size_t i = 0; i < numberOfParticipants; i++)
    {
        std::string fullName;
        inFile >> fullName;
        fragments.clear();
        boost::split(fragments, fullName, [](char c) {return c == '_'; });
        paricipantFileNames[fragments[0]].push_back(fullName);
    }
    inFile.close();
    //read and assign private registries
    for (const auto par : paricipantFileNames)
    {
        for (size_t i = 0; i < std::min(par.second.size(), clients.getClientsWithUserName(par.first).size()); i++)
        {
            path = home() + "/Sessions/" + name + "/" +par.second[i] +suffix;
            inFile.open(path, std::ios_base::binary);
            inFile >> trash; //get rid of date
            sessions[clients.getClientsWithUserName(par.first)[i]->getPrivateSession()]->loadRegistry(inFile);
            inFile.close();
        }
    }
    //read shared session (after private sessions to ensure that sessions get merged correctly in case of currensession.isPrivate())
    path = home() + "/Sessions/" + name + "/" + sessionName + suffix;
    inFile.open(path, std::ios_base::binary);
    std::getline(inFile, trash, ':');
    inFile >> numberOfParticipants;
    for (size_t i = 0; i < numberOfParticipants; i++)
    {
        inFile >> trash;
    }
    sessions[currentSession]->loadRegistry(inFile);
    inFile.close();
}
std::string VrbMessageHandler::getTime() const {
    time_t rawtime;
    time(&rawtime);
    struct tm *timeinfo = localtime(&rawtime);

    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeinfo);
    std::string str(buffer);
    return str;
}

}

