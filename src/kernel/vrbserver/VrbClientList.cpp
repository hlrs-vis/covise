/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbClientList.h"
#include "VrbMessageHandler.h"
#include <net/tokenbuffer.h>
#include <net/message.h>
#include <net/message_types.h>
#include <net/covise_connect.h>
#include <net/udpMessage.h>
#include <boost/algorithm/string.hpp>

using namespace covise;
using namespace std;
namespace vrb
{
VRBClientList clients;

VRBSClient::VRBSClient(Connection * c, UDPConnection* udpc, const char * ip, const char * n, bool send, bool dc)
	:deleteClient(dc)
	,address(ip)
	,m_name(n)
	,myID(clients.getNextFreeClientID())
	,m_publicSession(vrb::SessionID())
	,conn(c)
	,udpConn(udpc)
	,m_master(0)
	,lastRecTime(0.0)
,lastSendTime(0.0)
{
    if (send)
    {
        TokenBuffer rtb;
        rtb << myID;
        rtb << m_publicSession;
        Message m(rtb);
        m.type = COVISE_MESSAGE_VRB_GET_ID;
        conn->send_msg(&m);
    }

}

VRBSClient::~VRBSClient()
{
	if(deleteClient)
       delete conn;
    cerr << "closed connection to client " << myID << endl;
}

void VRBSClient::setContactInfo(const char *ip, const char *n, vrb::SessionID &session)
{
    address = ip;
    m_name = n;
    m_privateSession = session;
    TokenBuffer rtb;
    rtb << myID;
    rtb << session;
    Message m(rtb);
    m.type = COVISE_MESSAGE_VRB_GET_ID;
    conn->send_msg(&m);
}

void VRBSClient::setMaster(bool m)
{
    m_master = m;

}

std::string VRBSClient::getUserInfo()
{
    return userInfo;
}

std::string VRBSClient::getUserName()
{
    std::vector<std::string> info;
    boost::split(info, userInfo, [](char c) {return c == ',' || c == '"' || c == '\\'; });
    return info[1];
}

void VRBSClient::setSession(const vrb::SessionID &g)
{
    m_publicSession = g;
}

vrb::SessionID & VRBSClient::getPrivateSession()
{
    return m_privateSession;
}

void VRBSClient::setPrivateSession(vrb::SessionID & g)
{
    m_privateSession = g;
}

int VRBSClient::getMaster()
{
    return m_master;
}

void VRBSClient::setInterval(float i)
{
    interval = i;
}

void VRBSClient::getInfo(TokenBuffer &rtb)
{
    rtb << myID;
    rtb << address;
	cerr << "my ip is " << address << endl;
    rtb << m_name;
    rtb << userInfo;
    rtb << m_publicSession;
    rtb << m_master;
}

bool VRBSClient::doesNotKnowFile(const std::string& fileName)
{
	for (auto f : m_unknownFiles)
	{
		if (f == fileName)
		{
			return true;
		}
	}
	return false;
}

void VRBSClient::addUnknownFile(const std::string& fileName)
{
	m_unknownFiles.insert(fileName);
}

void VRBSClient::setUserInfo(const char *ui)
{
    userInfo = ui;
}

void VRBSClient::sendMsg(covise::MessageBase* msg)
{
	if (covise::Message * tcp = dynamic_cast<covise::Message*>(msg))
	{
		conn->send_msg(tcp);
		return;
	}
	if (UdpMessage * udp = dynamic_cast<UdpMessage*>(msg))
	{
		if (udpConn)
		{
            udpConn->send_udp_msg(udp, "188.40.97.72");
			udpConn->send_udp_msg(udp, address.c_str());
		}
		else if (firstTryUdp)
		{
			firstTryUdp = false;
			cerr << "trying to send udp message to a client without udp connection: type = " << dynamic_cast<UdpMessage*>(msg)->type << endl;
		}
	}
}

std::string VRBSClient::getName() const
{
    return m_name;
}

std::string VRBSClient::getIP() const
{
    return address;
}

int VRBSClient::getID() const
{
    return myID;
}

vrb::SessionID &VRBSClient::getSession()
{
    return m_publicSession;
}

int VRBSClient::getSentBPS()
{
    double currentTime = time();
    if ((currentTime - lastSendTime) > interval)
    {
        bytesSentPerSecond = (int)(bytesSentPerInterval / (currentTime - lastSendTime));
        lastSendTime = currentTime;
        bytesSentPerInterval = 0;
    }
    return bytesSentPerSecond;
}

int VRBSClient::getReceivedBPS()
{
    double currentTime = time();
    if ((currentTime - lastSendTime) > interval)
    {
        bytesReceivedPerSecond = (int)(bytesReceivedPerInterval / (currentTime - lastSendTime));
        lastRecTime = currentTime;
        bytesReceivedPerInterval = 0;
    }
    return bytesReceivedPerSecond;
}

void VRBSClient::addBytesSent(int b)
{
    double currentTime = time();
    bytesSent += b;
    bytesSentPerInterval += b;
    if ((currentTime - lastSendTime) > interval)
    {
        bytesSentPerSecond = (int)(bytesSentPerInterval / (currentTime - lastSendTime));
        lastSendTime = currentTime;
        bytesSentPerInterval = 0;
    }
}

void VRBSClient::addBytesReceived(int b)
{
    double currentTime = time();
    bytesReceived += b;
    bytesReceivedPerInterval += b;
    if ((currentTime - lastSendTime) > interval)
    {
        bytesReceivedPerSecond = (int)(bytesReceivedPerInterval / (currentTime - lastSendTime));
        lastRecTime = currentTime;
        bytesReceivedPerInterval = 0;
    }
}
#ifdef _WIN32

// Windows

#include <sys/timeb.h>
#include <time.h>

double VRBSClient::time()
{
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    return (double)timebuffer.time + 1.e-3 * (double)timebuffer.millitm;
}

#else

// Unix/Linux

#include <sys/time.h>
#include <unistd.h>

double VRBSClient::time()
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);

    return (double)tv.tv_sec + 1.e-6 * (double)tv.tv_usec;
}
#endif // _WIN32


///////////////////////////////////////////////////////////
//___________________________________________________________
//////////////////////////////////////////////////////////////
VRBSClient *VRBClientList::get(Connection *c)
{
    for (VRBSClient *cl : m_clients)
    {
        if (cl->conn == c)
        {
            return cl;
        }
    }
    return NULL;
}

int VRBClientList::numInSession(vrb::SessionID &session)
{
    int num = 0;
    for (auto cl : m_clients)
    {
        if (cl->getSession() == session)
        {
            ++num;
        }
    }
    return num;
}

int VRBClientList::numberOfClients()
{
    return m_clients.size();
}

void VRBClientList::addClient(VRBSClient * cl)
{
    m_clients.insert(cl);
}

void VRBClientList::removeClient(VRBSClient * cl)
{
    m_clients.erase(cl);
}

void VRBClientList::remove(Connection * c)
{
	for (auto cl : m_clients)
	{
		if (cl->conn == c)
		{
			m_clients.erase(cl);
			return;
		}
	}
}

void VRBClientList::passOnMessage(covise::MessageBase* msg, const vrb::SessionID &session)
{
    if (session.isPrivate())
    {
        return;
    }
    for (auto cl : m_clients)
    {
        if (cl->conn != msg->conn && (cl->getSession() == session || session == vrb::SessionID(0, "", false)))
        {
            cl->sendMsg(msg);
        }
    }
}

void VRBClientList::collectClientInfo(covise::TokenBuffer & tb)
{
    tb << (int)m_clients.size();
    for (auto cl : m_clients)
    {
        cl->getInfo(tb);
    }
}

VRBSClient* VRBClientList::getNextPossibleFileOwner(const std::string& fileName, const vrb::SessionID& id)
{
	for (auto cl : m_clients)
	{
		if (cl->getSession() == id && !cl->doesNotKnowFile(fileName))
		{
			return cl;
		}
	}
	return nullptr;
}

VRBSClient *VRBClientList::get(const char *ip)
{
    for (VRBSClient *cl : m_clients)
    {
        if (cl->getIP() == ip)
        {
            return cl;
        }
    }
    return NULL;
}

VRBSClient *VRBClientList::get(int id)
{
    for (VRBSClient *cl : m_clients)
    {
        if (cl->getID() == id)
        {
            return cl;
        }
    }
    return NULL;
}

VRBSClient * VRBClientList::getMaster(const vrb::SessionID &session)
{
    for (auto cl : m_clients)
    {
        if (cl->getSession() == session && cl->getMaster())
        {
            return cl;
        }
    }
    return nullptr;
}

VRBSClient * VRBClientList::getNextInGroup(const vrb::SessionID & id)
{
    for (auto cl : m_clients)
    {
        if (cl->getSession() == id)
        {
            return cl;
        }
    }
    return nullptr;
}

VRBSClient * VRBClientList::getNthClient(int N)
{
    if (N > m_clients.size())
    {
        return nullptr;
    }
    auto it = m_clients.begin();
    std::advance(it, N);
    return *it;

}

std::vector<VRBSClient*> VRBClientList::getClientsWithUserName(const std::string & name)
{
    std::vector<VRBSClient *> clients;
    for (const auto cl : m_clients)
    {
        if (cl->getUserName() == name)
        {
            clients.push_back(cl);
        }
    }
    return clients;
}

int VRBClientList::getNextFreeClientID()
{
    int id = 1;
    while (get(id))
    {
        id++;
    }
    return id;
}

void VRBClientList::setMaster(VRBSClient * client)
{
    for (auto cl : m_clients)
    {
        if (cl == client)
        {
            cl->setMaster(true);
        }
        else if (cl->getSession() == client->getSession())
        {
            cl->setMaster(false);
        }
    }
}

void VRBClientList::setInterval(float i)
{
    for (VRBSClient *cl : m_clients)
    {
        cl->setInterval(i);
    }

}

void VRBClientList::deleteAll()
{
    m_clients.clear();
}

void VRBClientList::sendMessage(TokenBuffer &stb, const vrb::SessionID &group, covise_msg_type type)
{
    Message m(stb);
    m.type = type;

    for (VRBSClient *cl : m_clients)
    {
        if (group == vrb::SessionID(0, std::string(), false) || group == cl->getSession())
        {
            cl->sendMsg(&m);
        }
    }
}

void VRBClientList::sendMessageToID(TokenBuffer &stb, int ID, covise_msg_type type)
{
    Message m(stb);
    m.type = type;
    VRBSClient *cl = get(ID);
    if (cl)
    {
		cl->sendMsg(&m);
    }
}

void VRBClientList::sendMessageToAll(covise::TokenBuffer &stb, covise::covise_msg_type type)
{
    Message m(stb);
    m.type = type;

    for (VRBSClient *cl : m_clients)
    {
		cl->sendMsg(&m);
    }
}
std::string VRBClientList::cutFileName(const std::string& fileName)
{
	std::string name = fileName;
	for (size_t i = fileName.length() - 1; i > 0; --i)
	{
		name.pop_back();
		if (fileName[i] == '/' || fileName[i] == '\\')
		{
			return name;
		}

	}
	cerr << "invalid file path : " << fileName << endl;
	return "";
}
}