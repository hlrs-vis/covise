/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VrbClientList.h"
#include "VrbFileBrowserHandler.h"
#include "VrbMessageHandler.h"
#include "VrbServerRegistry.h"
#include "VrbSessionList.h"

#include <net/covise_connect.h>
#include <net/dataHandle.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>
#include <net/udpMessage.h>
#include <net/udp_message_types.h>
#include <vrb/SessionID.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace covise;

namespace covise{
CREATE_TB_CLASS(std::string, Class);
CREATE_TB_CLASS(vrb::SessionID, sessionID);
CREATE_TB_CLASS(int, senderID);
CREATE_TB_CLASS(std::string, variable);
CREATE_TB_CLASS(char*, value);
CREATE_TB_CLASS(DataHandle, valueData)

std::ostream& operator<<(std::ostream& s, const DataHandle& dh) {
	s << "DataHandle with size: " << dh.length() << " ";
	return s;
}
}
 
#ifdef MB_DEBUG 
	#define PRINT_DEBUG	std::cerr << res << std::endl;
#else 
	#define PRINT_DEBUG
#endif

namespace vrb
{

VrbMessageHandler::VrbMessageHandler(ServerInterface* server)
	:m_server(server)
{
}
void VrbMessageHandler::handleMessage(Message* msg)
{
	TokenBuffer tb(msg);
#ifdef MB_DEBUG
	std::cerr << "::handleMessage of type " << covise_msg_types_array[int(msg->type)] << " with length " << msg->data.length() << "!" << std::endl;
#endif
	switch (msg->type)
	{
	case COVISE_MESSAGE_VRB_REQUEST_FILE:
	{
		handleFileRequest(msg, tb);
	}
	break;
	case COVISE_MESSAGE_VRB_SEND_FILE:
	{
		sendFile(msg, tb);
	}
	break;
	case COVISE_MESSAGE_VRB_CURRENT_FILE:
	{
		assert(false);
		//deprecated, use COVISE_MESSAGE_VRB_REQUEST_FILE and sharedState coVRFileManager_filePaths
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE: // Set Registry value
	{
		changeRegVar(tb);
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class>(tb);
		PRINT_DEBUG
		sessions[res.get<sessionID>()].observeClass(res.get<senderID>(), res.get<Class>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class, variable, valueData>(tb);
		PRINT_DEBUG
		createSessionIfnotExists(res.get<sessionID>(), res.get<senderID>())
			.observeVar(res.get<senderID>(), res.get<Class>(), res.get < variable>(), res.get < valueData>());

		updateApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>(), res.get<valueData>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class>(tb);
		PRINT_DEBUG
		sessions[res.get<sessionID>()].unObserveClass(res.get<senderID>(), res.get<Class>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE:
	{
		auto res = readTokenBuffer<senderID, Class, variable>(tb);
		PRINT_DEBUG
		sessions.unobserveFromAll(res.get<senderID>(), res.get<Class>(), res.get<variable>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY:
	{
		bool isStatic;
		auto res = readTokenBuffer<sessionID, senderID, Class, variable, valueData>(tb);
		PRINT_DEBUG
		tb >> isStatic;
		createSessionIfnotExists(res.get<sessionID>(), res.get<senderID>())
			.create(res.get<senderID>(), res.get<Class>(), res.get<variable>(), res.get<valueData>(), isStatic);
		updateApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>(), res.get<valueData>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class, variable>(tb);
		PRINT_DEBUG
		sessions[res.get<sessionID>()].deleteEntry(res.get<Class>(), res.get<variable>());
		removeEntryFromApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>());
	}
	break;
	case COVISE_MESSAGE_VRB_CONTACT:
	{
		handleNewClient(tb, msg);
	}
	break;
	case COVISE_MESSAGE_VRB_CONNECT_TO_COVISE:
	{
		char* ip;
		tb >> ip;
		if (VRBSClient* c = clients.get(ip))
		{
			c->conn->send_msg(msg);
		}
	}
	break;
	case COVISE_MESSAGE_VRB_SET_USERINFO:
	{
		setUserInfo(tb, msg);
	}
	break;
	case COVISE_MESSAGE_RENDER:
	case COVISE_MESSAGE_RENDER_MODULE: 
	case COVISE_MESSAGE_VRB_MESSAGE:
	{
		passMessageToParticipants(msg);
	}
	break;
	case COVISE_MESSAGE_VRB_CHECK_COVER:
	{
		returnWhetherClientExists(tb, msg);
	}
	break;
	case COVISE_MESSAGE_VRB_GET_ID:
	{
		returnStartSession(msg);
	}
	break;
	case COVISE_MESSAGE_VRB_SET_GROUP:
	{
		VRBSClient* c = clients.get(msg->conn);
		vrb::SessionID newSession;
		tb >> newSession;
		if (c && !newSession.isPrivate())
		{
			determineNewMaster(newSession, c);
			informParticipantsAboutNewParticipant(c);
		}
	}
	break;
	case COVISE_MESSAGE_VRB_SET_MASTER:
	{
		VRBSClient* c = clients.get(msg->conn);
		bool masterState;
		tb >> masterState;
		if (c && masterState)
		{
			clients.setMaster(c);
			informParticipantsAboutNewMaster(c);
		}
	}
	break;
	case COVISE_MESSAGE_SOCKET_CLOSED:
	case COVISE_MESSAGE_CLOSE_SOCKET:
	case COVISE_MESSAGE_QUIT:
	{
		remove(msg->conn);
		cerr << "Numclients: " << clients.numberOfClients() << endl;
	}
	break;
	case COVISE_MESSAGE_VRB_FB_RQ:
	{
		handleFileBrouwserRequest(msg);
	}
	break;
	case COVISE_MESSAGE_VRB_FB_REMREQ:
	{
		handleFileBrowserRemoteRequest(msg);
	}
	break;
	case COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION:
	{
		SessionID sid;
		tb >> sid;
		sid = sessions.forceCreateSession(sid).sessionID();
		sendSessions();
		setSession(sid, sid.owner());
	}
	break;
	case COVISE_MESSAGE_VRBC_UNOBSERVE_SESSION:
	{
		auto res = readTokenBuffer<sessionID, senderID>(tb);
		PRINT_DEBUG
		sessions.unobserveSession(res.get<senderID>(), res.get<sessionID>());
	}
	break;
	case COVISE_MESSAGE_VRBC_SET_SESSION:
	{
		auto res = readTokenBuffer<sessionID, senderID>(tb);
		VRBSClient* c = clients.get(res.get<senderID>());
		if (c)
		{
			c->setSession(res.get<sessionID>());
			informParticipantsAboutNewParticipant(c);
		}
		cerr << "client " << res.get<senderID>() << " not found" << endl;
	}
	break;
	case COVISE_MESSAGE_VRB_SAVE_SESSION:
	{
		returnSerializedSessionInfo(tb);
	}
	break;
	case COVISE_MESSAGE_VRB_LOAD_SESSION:
	{
		auto res = readTokenBuffer<senderID, sessionID, valueData>(tb);
		PRINT_DEBUG
		loadSession(res.get<valueData>(), res.get<sessionID>());
		informParticipantsAboutLoadedSession(res.get<senderID>(), res.get<sessionID>());
	}
	break;
	case COVISE_MESSAGE_UI:
		//ignore
		break;
	default:
		cerr << "unknown message in vrb: type=" << msg->type;
		if (VRBSClient* c = clients.get(msg->conn))
		{
			cerr << " from client" << c->getID();
		}
		cerr << endl;
		break;


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
	clients.passOnMessage(msg, sender->getSession());
}
int VrbMessageHandler::numberOfClients()
{
	return clients.numberOfClients();
}
void VrbMessageHandler::addClient(VRBSClient* client)
{
	clients.addClient(client);
}
void VrbMessageHandler::remove(Connection* conn)
{
	m_server->removeConnection(conn);
	VRBSClient* c = clients.get(conn);
	if (c)
	{
		int clID = c->getID();
		cerr << c->getName() << " (host " << c->getIP() << " ID: " << c->getID() << ") left" << endl;
		vrb::SessionID MasterSession;
		bool wasMaster = false;
		TokenBuffer rtb;

		rtb << c->getID();
		if (c->isMaster())
		{
			MasterSession = c->getSession();
			cerr << "Master Left" << endl;
			wasMaster = true;
		}
		clients.removeClient(c);
		delete c;
		sessions.disconectClientFromSessions(clID);


		removeEntriesFromApplicationWindow(clID);
		sendSessions();
		clients.sendMessageToAll(rtb, COVISE_MESSAGE_VRB_QUIT);
		if (wasMaster)
		{
			VRBSClient* newMaster = clients.getNextInGroup(MasterSession);
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
void VrbMessageHandler::closeConnection()
{
	TokenBuffer tb;
	clients.sendMessageToAll(tb, COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION);
}
//protected->implementation in derived classes
void VrbMessageHandler::updateApplicationWindow(const std::string& cl, int sender, const std::string& var, const covise::DataHandle& value)
{
	//std::cerr <<"userinterface not implemented" << std:endl;
}
void VrbMessageHandler::removeEntryFromApplicationWindow(const std::string& cl, int sender, const std::string& var)
{
	//std::cerr <<"userinterface not implemented" << std:endl;
}
void VrbMessageHandler::removeEntriesFromApplicationWindow(int sender)
{
	//std::cerr <<"userinterface not implemented" << std:endl;
}
//private
void VrbMessageHandler::handleFileRequest(covise::Message* msg, covise::TokenBuffer& tb)
{
	struct stat statbuf;
	vrb::VRBSClient* c = clients.get(msg->conn);
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
		int fd;
		if ((fd = open(filename.c_str(), O_RDONLY)) > 0)
		{
			rtb << (int)statbuf.st_size;
			const char* buf = rtb.allocBinary(statbuf.st_size);
			ssize_t retval;
			retval = read(fd, (void*)buf, statbuf.st_size);
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
void VrbMessageHandler::sendFile(covise::Message* msg, covise::TokenBuffer& tb)
{
	VRBSClient* c = clients.get(msg->conn);
	if (!c)
	{
		return;
	}
	c->addBytesSent(msg->data.length());

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
			return;
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
void VrbMessageHandler::changeRegVar(covise::TokenBuffer& tb)
{
	auto res = readTokenBuffer<sessionID, senderID, Class, variable, valueData>(tb);
	PRINT_DEBUG
	sessions[res.get<sessionID>()].setVar(res.get<senderID>(), res.get<Class>(), res.get<variable>(), res.get<valueData>()); //maybe handle no existing registry for the given session ID
	updateApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>(), res.get<valueData>());
}
VrbServerRegistry& VrbMessageHandler::createSessionIfnotExists(const vrb::SessionID& sessionID, int senderID)
{
	VRBSClient* cl = clients.get(senderID);
	return sessions.createSessionIfnotExists(sessionID, cl);
}
void VrbMessageHandler::handleNewClient(covise::TokenBuffer& tb, covise::Message* msg)
{

	bool hasStartupSession;
	tb >> hasStartupSession; //== coviseModuleID
	char* startupSession, *name, *hostaddress;
	if (hasStartupSession)
	{
		tb >> startupSession;
	}
	tb >> name >> hostaddress;
	VRBSClient *c = clients.get(msg->conn);
	if (!c)
	{
		c = new VRBSClient(msg->conn, nullptr, hostaddress, name, false, false);
		clients.addClient(c);
	}
	//create unique private session for the client
	vrb::SessionID sid = vrb::SessionID(c->getID(), "private");
	sid = sessions.forceCreateSession(sid).sessionID();

	c->setContactInfo(hostaddress, name, sid);
	std::cerr << "VRB new client: Numclients=" << clients.numberOfClients() << std::endl;
	//if the client is started by covise (CRB) it has a moduleID that can be translated to sessionID
	//if OpenCOVER got a session from commandline (-s) the client will also send this given session name
	if (hasStartupSession)
	{
		sid = vrb::SessionID(c->getID(), std::string(startupSession), false);
		sid.setOwner(sessions.getSessionOwner(sid));

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
void VrbMessageHandler::setUserInfo(covise::TokenBuffer& tb, covise::Message* msg)
{
	UserInfo ui;
	tb >> ui;
	VRBSClient* c = clients.get(msg->conn);
	if (!c)
	{
		std::cerr << "failed to set user info: client not found!" << std::endl;
		return;
	}
	c->setUserInfo(ui);
	informClientsAboutNewClient(c);
	informNewClientAboutClients(c);
	setNewClientsStartingSession(c);

}
void VrbMessageHandler::informClientsAboutNewClient(vrb::VRBSClient* c)
{
	TokenBuffer rtb;
	rtb << 1;
	c->getInfo(rtb);
	Message m(rtb);
	m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
	m.conn = c->conn;
	clients.passOnMessage(&m);
}
void VrbMessageHandler::informNewClientAboutClients(vrb::VRBSClient* c)
{
	TokenBuffer rtb2;
	clients.collectClientInfo(rtb2);
	Message m(rtb2);
	m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
	c->conn->send_msg(&m);
}
void VrbMessageHandler::setNewClientsStartingSession(vrb::VRBSClient* c)
{
	auto it = m_sessionsToSet.find(c->getID());
	if (it != m_sessionsToSet.end())
	{
		setSession(c->getSession(), c->getID());
		m_sessionsToSet.erase(it);
	}
}
void VrbMessageHandler::passMessageToParticipants(covise::Message* msg)
{

	vrb::SessionID toGroup;

	VRBSClient* c = clients.get(msg->conn);

	if (c)
	{
		toGroup = c->getSession();
		c->addBytesSent(msg->data.length());
	}
	clients.passOnMessage(msg, toGroup);
}
void VrbMessageHandler::returnWhetherClientExists(covise::TokenBuffer& tb, covise::Message* msg)
{
	char* ip;
	tb >> ip;
	VRBSClient* c = clients.get(ip);
	TokenBuffer rtb;
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
void VrbMessageHandler::returnStartSession(covise::Message* msg)
{
	int clId = -1;
	vrb::SessionID sid;
	VRBSClient* c = clients.get(msg->conn);
	if (c)
	{
		sid = sessions.forceCreateSession(vrb::SessionID(c->getID(), "private")).sessionID();
		clId = c->getID();
		c->setPrivateSession(sid);
	}
	TokenBuffer rtb;
	rtb << clId;
	rtb << sid;
	Message m(rtb);
	m.type = COVISE_MESSAGE_VRB_GET_ID;
	msg->conn->send_msg(&m);
}
void VrbMessageHandler::determineNewMaster(vrb::SessionID& newSession, vrb::VRBSClient* c)
{
	bool becomeMaster;
	if (clients.getMaster(newSession) && c->isMaster()) //there is already a master
	{
		becomeMaster = false;
	}
	else
	{
		becomeMaster = true;
	}
	c->setMaster(becomeMaster);
	TokenBuffer mtb;
	mtb << c->getID();
	mtb << becomeMaster;
	Message m(mtb);
	m.type = COVISE_MESSAGE_VRB_SET_MASTER;
	c->conn->send_msg(&m);
}
void VrbMessageHandler::informParticipantsAboutNewMaster(vrb::VRBSClient* c)
{
	TokenBuffer rtb;
	rtb << c->getID();
	rtb << c->isMaster();
	Message  rmsg(rtb);
	rmsg.type = COVISE_MESSAGE_VRB_SET_MASTER;
	rmsg.conn = c->conn;
	clients.passOnMessage(&rmsg, c->getSession());
}
void VrbMessageHandler::informParticipantsAboutNewParticipant(vrb::VRBSClient* c)
{
	TokenBuffer rtb;
	rtb << c->getID();
	rtb << c->getSession();
	Message mg(rtb);
	mg.type = COVISE_MESSAGE_VRB_SET_GROUP;
	mg.conn = c->conn;
	clients.passOnMessage(&mg);
}
void VrbMessageHandler::sendSessions()
{
	TokenBuffer mtb;

	if (sessions.serializeSessions(mtb))
	{
		clients.sendMessageToAll(mtb, COVISE_MESSAGE_VRBC_SEND_SESSIONS);
	}
}
void VrbMessageHandler::setSession(const vrb::SessionID& sessionId, int clID)
{
	TokenBuffer stb;
	stb << sessionId;
	clients.sendMessageToID(stb, clID, COVISE_MESSAGE_VRBC_SET_SESSION);
}
void VrbMessageHandler::returnSerializedSessionInfo(covise::TokenBuffer& tb)
{
	auto res = readTokenBuffer<senderID, sessionID>(tb);
	PRINT_DEBUG
	std::string fileName;
	tb >> fileName;
	auto s = sessions.serializeSession(res.get<sessionID>());
	TokenBuffer out;
	out << fileName;
	out << s;
	clients.sendMessageToID(out, res.get<senderID>(), COVISE_MESSAGE_VRB_SAVE_SESSION);
}
void VrbMessageHandler::loadSession(const covise::DataHandle& file, const vrb::SessionID& currentSession)
{
	TokenBuffer tb{ file };
	auto registry = sessions.deserializeSession(tb, currentSession);
	//update application window 
	for (const auto& cl : registry)
	{
		for (const auto& var : *cl)
		{
			updateApplicationWindow(cl->name().c_str(), currentSession.owner(), var.first.c_str(), var.second->value());
		}
	}
}
void VrbMessageHandler::informParticipantsAboutLoadedSession(int senderID, SessionID sid)
{
	TokenBuffer stb;
	stb << sid;
	if (sid.isPrivate())//inform the owner of the private session 
	{
		clients.sendMessageToID(stb, senderID, COVISE_MESSAGE_VRB_LOAD_SESSION);
	}
	else //inform all session members
	{
		clients.sendMessage(stb, sid, COVISE_MESSAGE_VRB_LOAD_SESSION);
	}
}


}

