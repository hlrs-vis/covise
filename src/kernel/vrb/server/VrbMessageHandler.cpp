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
		m_sessions[res.get<sessionID>()].observeClass(res.get<senderID>(), res.get<Class>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class, variable, valueData>(tb);
		PRINT_DEBUG
		putClientInSession(res.get<sessionID>(), res.get<senderID>())
			.observeVar(res.get<senderID>(), res.get<Class>(), res.get < variable>(), res.get < valueData>());

		updateApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>(), res.get<valueData>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class>(tb);
		PRINT_DEBUG
		m_sessions[res.get<sessionID>()].unObserveClass(res.get<senderID>(), res.get<Class>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE:
	{
		auto res = readTokenBuffer<senderID, Class, variable>(tb);
		PRINT_DEBUG
		m_sessions.unobserveFromAll(res.get<senderID>(), res.get<Class>(), res.get<variable>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY:
	{
		bool isStatic;
		auto res = readTokenBuffer<sessionID, senderID, Class, variable, valueData>(tb);
		PRINT_DEBUG
		tb >> isStatic;
		putClientInSession(res.get<sessionID>(), res.get<senderID>())
			.create(res.get<senderID>(), res.get<Class>(), res.get<variable>(), res.get<valueData>(), isStatic);
		updateApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>(), res.get<valueData>());
	}
	break;
	case COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY:
	{
		auto res = readTokenBuffer<sessionID, senderID, Class, variable>(tb);
		PRINT_DEBUG
		m_sessions[res.get<sessionID>()].deleteEntry(res.get<Class>(), res.get<variable>());
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
			c->conn->sendMessage(msg);
		}
	}
	break;
	case COVISE_MESSAGE_VRB_SET_USERINFO:
	{
		//deprecated
	}
	break;
	case COVISE_MESSAGE_RENDER:
	case COVISE_MESSAGE_RENDER_MODULE: 
	case COVISE_MESSAGE_VRB_MESSAGE:
	{
		passMessageToParticipants(msg);
	}
	break;
	case COVISE_MESSAGE_BROADCAST_TO_PROGRAM:
	{
		vrb::Program p;
		int messageType;
		covise::DataHandle broadcastDh;
		tb >> p >> messageType >> broadcastDh;
		covise::Message broadcastMsg{messageType, broadcastDh};
		clients.broadcastMessageToProgramm(p, &broadcastMsg);
	}
	break;
	case COVISE_MESSAGE_VRB_GET_ID:
	{
		returnStartSession(msg);
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
			informOtherParticipantsAboutNewMaster(c);
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
		sid = m_sessions.forceCreateSession(sid).sessionID();
		sendSessions();
		setSession(sid, sid.owner());
	}
	break;
	case COVISE_MESSAGE_VRBC_UNOBSERVE_SESSION:
	{
		auto res = readTokenBuffer<sessionID, senderID>(tb);
		PRINT_DEBUG
		m_sessions.unobserveSession(res.get<senderID>(), res.get<sessionID>());
	}
	break;
	case COVISE_MESSAGE_VRBC_SET_SESSION:
	{
		auto res = readTokenBuffer<sessionID, senderID>(tb);
		PRINT_DEBUG
		VRBSClient* c = clients.get(res.get<senderID>());
		if (c)
		{
			bool wasMaster = c->isMaster();
			c->setSession(res.get<sessionID>());
			if (wasMaster)
			{
				determineNewMaster(c->sessionID());
			}

			informOtherParticipantsAboutNewParticipant(c);
		}
		else
		{
			cerr << "client " << res.get<senderID>() << " not found" << endl;
		}
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
			cerr << " from client" << c->ID();
		}
		cerr << endl;
		break;


	}
}
void VrbMessageHandler::handleUdpMessage(covise::UdpMessage* msg)
{
	VRBSClient* sender = clients.get(msg->sender);
	if (!sender)
	{
		cerr << "received udp message from unknown client wit ip = " << msg->m_ip << endl;
		return;
	}
	msg->conn = sender->conn;
	clients.passOnMessage(msg, sender->sessionID());
}
int VrbMessageHandler::numberOfClients()
{
	return clients.numberOfClients();
}
void VrbMessageHandler::addClient(ConnectionDetails::ptr &&clientCon)
{
	m_unregisteredClients.push_back(std::move(clientCon));
}
void VrbMessageHandler::remove(Connection* conn)
{
	m_server->removeConnection(conn);
	VRBSClient* c = clients.get(conn);
	if (c)
	{
		int clID = c->ID();
		cerr << c->userInfo() << "----> left" << endl;;
		vrb::SessionID sid = c->sessionID();
		bool master = c->isMaster();

		clients.removeClient(c);
		removeEntriesFromApplicationWindow(clID);
		m_sessions.disconectClientFromSessions(clID);
		sendSessions();

		TokenBuffer rtb;
		rtb << clID;
		clients.sendMessageToAll(rtb, COVISE_MESSAGE_VRB_QUIT);

		if (master)
		{
			cerr << ": was master of session " << sid << endl;
			determineNewMaster(sid);
		}
		else
		{
			cerr << endl;
		}
	}
	else
	{
		removeUnregisteredClient(conn);
	}
}

void VrbMessageHandler::removeUnregisteredClient(Connection* conn) {
	auto unregisteredClient = std::find_if(m_unregisteredClients.begin(), m_unregisteredClients.end(), [conn](const ConnectionDetails::ptr& cd) {
		return cd->tcpConn.get() == conn;
		});
	if (unregisteredClient != m_unregisteredClients.end())
	{
		m_unregisteredClients.erase(unregisteredClient);
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

VRBSClient *VrbMessageHandler::createNewClient(ConnectionDetails::ptr &&cd, covise::TokenBuffer &tb, bool deleteTcpCon){
	return new VRBSClient(cd->tcpConn.release(), cd->udpConn, tb, deleteTcpCon);
}

VRBSClient *VrbMessageHandler::createNewClient(covise::TokenBuffer &tb, covise::Message *msg){
	auto it = std::find_if(m_unregisteredClients.begin(), m_unregisteredClients.end(), [msg](const ConnectionDetails::ptr &cd) {
		return cd->tcpConn.get() == msg->conn;
	});
	if (it == m_unregisteredClients.end()) //this happens if the server is covise
	{
		ConnectionDetails::ptr cd{new ConnectionDetails{}};
		cd->tcpConn.reset(msg->conn);
		return createNewClient(std::move(cd), tb, false);
	}
	else
	{
		auto cd = std::move(*it);
		m_unregisteredClients.erase(it);
		return createNewClient(std::move(cd), tb);
	}
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
		msg->conn->sendMessage(&m);
	}
	else
	{
		// forward this request to the COVER that loaded the requested file
		VRBSClient* currentFileClient = clients.get(fileOwner);
		if ((currentFileClient) && (currentFileClient != c) && (currentFileClient->ID() != requestorsID))
		{
			currentFileClient->conn->sendMessage(msg);
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
			msg->conn->sendMessage(&m);
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
		VRBSClient* nextCl = clients.getNextPossibleFileOwner(fileName, c->sessionID());
		if (nextCl)
		{
			TokenBuffer rtb;
			rtb << fileName;
			rtb << destinationID;
			Message rmsg(rtb);
			rmsg.type = COVISE_MESSAGE_VRB_REQUEST_FILE;
			nextCl->conn->sendMessage(&rmsg);
			return;
		}
	}
	// forward this File to the one that requested it
	c = clients.get(destinationID);
	if (c)
	{
		c->conn->sendMessage(msg);
		c->addBytesReceived(msg->data.length());
	}
}
void VrbMessageHandler::changeRegVar(covise::TokenBuffer& tb)
{
	auto res = readTokenBuffer<sessionID, senderID, Class, variable, valueData>(tb);
	PRINT_DEBUG
	m_sessions[res.get<sessionID>()].setVar(res.get<senderID>(), res.get<Class>(), res.get<variable>(), res.get<valueData>()); //maybe handle no existing registry for the given session ID
	updateApplicationWindow(res.get<Class>(), res.get<senderID>(), res.get<variable>(), res.get<valueData>());
}
VrbServerRegistry& VrbMessageHandler::putClientInSession(const vrb::SessionID& sessionID, int senderID)
{
	VRBSClient* cl = clients.get(senderID);
	return putClientInSession(sessionID, cl);
}

VrbServerRegistry& VrbMessageHandler::putClientInSession(const vrb::SessionID& sessionID, VRBSClient* cl)
{
	SessionID sid(cl->ID(), sessionID.name(), sessionID.isPrivate());
	VrbServerRegistry &reg = m_sessions[sid];
	cl->setSession(reg.sessionID());
	return reg;
}

void VrbMessageHandler::handleNewClient(covise::TokenBuffer& tb, covise::Message* msg)
{

	assert(!clients.get(msg->conn));
	VRBSClient *c = createNewClient(tb, msg);
	clients.addClient(c);
	//create unique private session for the client
	vrb::SessionID privateSid{c->ID()};
	privateSid = m_sessions.forceCreateSession(privateSid).sessionID();
	c->setPrivateSession(privateSid);
	std::cerr << "VRB new client: Numclients=" << clients.numberOfClients() << std::endl;
	//if the client is started by covise (CRB) it has a moduleID that can be translated to sessionID
	//if OpenCOVER got a session from commandline (-s) the client will also send this given session name
	if (!c->sessionID().name().empty())
	{
		SessionID sid = vrb::SessionID(c->ID(), c->sessionID().name(), false);
		sid = putClientInSession(sid, c).sessionID();
		c->setSession(sid);
	}
	else
	{
		c->setSession(privateSid);
	}
	sendSessions();
	informClientsAboutNewClient(c);
	informNewClientAboutClients(c);
	//re-sent the sessions after the the new client is informed about himself
	TokenBuffer mtb;
	if (m_sessions.serializeSessions(mtb))
	{
		clients.sendMessageToClient(c, mtb, COVISE_MESSAGE_VRBC_SEND_SESSIONS);
	}

	
}

void VrbMessageHandler::informClientsAboutNewClient(vrb::VRBSClient* c)
{
	TokenBuffer rtb;
	rtb << 1;
	rtb << -1; //recipient info is not included
	rtb << *c;
	Message m(rtb);
	m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
	m.conn = c->conn;
	clients.passOnMessage(&m);
}
void VrbMessageHandler::informNewClientAboutClients(vrb::VRBSClient* c)
{
	TokenBuffer rtb2;
	clients.collectClientInfo(rtb2, c);
	Message m(rtb2);
	m.type = COVISE_MESSAGE_VRB_SET_USERINFO;
	c->conn->sendMessage(&m);
}
void VrbMessageHandler::setNewClientsStartingSession(vrb::VRBSClient* c)
{
	auto it = m_sessionsToSet.find(c->ID());
	if (it != m_sessionsToSet.end())
	{
		setSession(c->sessionID(), c->ID());
		m_sessionsToSet.erase(it);
	}
}
void VrbMessageHandler::passMessageToParticipants(covise::Message* msg)
{

	vrb::SessionID toGroup;

	VRBSClient* c = clients.get(msg->conn);

	if (c)
	{
		toGroup = c->sessionID();
		c->addBytesSent(msg->data.length());
	}
	clients.passOnMessage(msg, toGroup);
}

void VrbMessageHandler::returnStartSession(covise::Message* msg)
{
	int clId = -1;
	vrb::SessionID sid;
	VRBSClient* c = clients.get(msg->conn);
	if (c)
	{
		sid = m_sessions.forceCreateSession(vrb::SessionID(c->ID(), "private")).sessionID();
		clId = c->ID();
		c->setPrivateSession(sid);
	}
	TokenBuffer rtb;
	rtb << clId;
	rtb << sid;
	Message m(rtb);
	m.type = COVISE_MESSAGE_VRB_GET_ID;
	msg->conn->sendMessage(&m);
}
void VrbMessageHandler::determineNewMaster(const vrb::SessionID& sid)
{
	VRBSClient* newMasterofOldSession = clients.getNextInGroup(sid);
	if (newMasterofOldSession)
	{
		clients.setMaster(newMasterofOldSession);
		TokenBuffer rtb;
		rtb << newMasterofOldSession->ID();
		rtb << true;
		Message  rmsg(rtb);
		rmsg.type = COVISE_MESSAGE_VRB_SET_MASTER;
		clients.passOnMessage(&rmsg, newMasterofOldSession->sessionID()); //informOtherParticipantsAboutNewMaster + msg to the master
	}
}
void VrbMessageHandler::informOtherParticipantsAboutNewMaster(vrb::VRBSClient* c)
{
	TokenBuffer rtb;
	rtb << c->ID();
	rtb << c->isMaster();
	Message  rmsg(rtb);
	rmsg.type = COVISE_MESSAGE_VRB_SET_MASTER;
	rmsg.conn = c->conn;
	clients.passOnMessage(&rmsg, c->sessionID());
}
void VrbMessageHandler::informOtherParticipantsAboutNewParticipant(vrb::VRBSClient* c)
{
	TokenBuffer rtb;
	rtb << c->ID();
	rtb << c->sessionID();
	Message mg(rtb);
	mg.type = COVISE_MESSAGE_VRBC_SET_SESSION;
	mg.conn = c->conn;
	clients.passOnMessage(&mg);
}
void VrbMessageHandler::sendSessions()
{
	TokenBuffer mtb;

	if (m_sessions.serializeSessions(mtb))
	{
		clients.sendMessageToAll(mtb, COVISE_MESSAGE_VRBC_SEND_SESSIONS);
	}
}
void VrbMessageHandler::setSession(const vrb::SessionID& sessionId, int clID)
{
	TokenBuffer stb;
	stb << clID;
	stb << sessionId;
	clients.sendMessageToClient(clID, stb, COVISE_MESSAGE_VRBC_SET_SESSION);
}
void VrbMessageHandler::returnSerializedSessionInfo(covise::TokenBuffer& tb)
{
	auto res = readTokenBuffer<senderID, sessionID>(tb);
	PRINT_DEBUG
	std::string fileName;
	tb >> fileName;
	auto s = m_sessions.serializeSession(res.get<sessionID>());
	TokenBuffer out;
	out << fileName;
	out << s;
	clients.sendMessageToClient(res.get<senderID>(), out, COVISE_MESSAGE_VRB_SAVE_SESSION);
}
void VrbMessageHandler::loadSession(const covise::DataHandle& file, const vrb::SessionID& currentSession)
{
	TokenBuffer tb{ file };
	auto &registry = m_sessions.deserializeSession(tb, currentSession);
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
		clients.sendMessageToClient(senderID, stb, COVISE_MESSAGE_VRB_LOAD_SESSION);
	}
	else //inform all session members
	{
		clients.sendMessage(stb, sid, COVISE_MESSAGE_VRB_LOAD_SESSION);
	}
}


}

