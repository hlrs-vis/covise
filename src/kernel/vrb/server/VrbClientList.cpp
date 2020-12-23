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

VRBSClient::VRBSClient(Connection * c, UDPConnection* udpc, covise::TokenBuffer &tb, bool dc)
	: RemoteClient(tb)
    , conn(c)
	, udpConn(udpc)
	, m_deleteClient(dc)
{
    setID(clients.getNextFreeClientID());
	cerr << "my ip is " << userInfo().ipAdress << endl;
}

VRBSClient::~VRBSClient()
{
    if (m_deleteClient)
    {
        delete conn;
        conn = nullptr;
        //udpConn is owned by server
    }
    cerr << "closed connection to client " << ID() << endl;
}

const vrb::SessionID & VRBSClient::getPrivateSession() const
{
    return m_privateSession;
}

void VRBSClient::setPrivateSession(vrb::SessionID & g)
{
    m_privateSession = g;
}


void VRBSClient::setInterval(float i)
{
    m_interval = i;
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

bool VRBSClient::sendMessage(const covise::Message* msg)
{
	if (conn)
    {
        return conn->sendMessage(msg);
    }
    return false;
    
}

bool VRBSClient::sendMessage(const covise::UdpMessage *msg){

    if (udpConn)
    {
        udpConn->send_udp_msg(msg, "188.40.97.72");
        return udpConn->send_udp_msg(msg, userInfo().ipAdress.c_str());
    }
    else if (m_firstTryUdp)
    {
        m_firstTryUdp = false;
        cerr << "trying to send udp message to a client without udp connection: type = " << msg->type << endl;
    }
    return false;
}

int VRBSClient::getSentBPS()
{
    double currentTime = time();
    if ((currentTime - m_lastSendTime) > m_interval)
    {
        m_bytesSentPerSecond = (int)(m_bytesSentPerInterval / (currentTime - m_lastSendTime));
        m_lastSendTime = currentTime;
        m_bytesSentPerInterval = 0;
    }
    return m_bytesSentPerSecond;
}

int VRBSClient::getReceivedBPS()
{
    double currentTime = time();
    if ((currentTime - m_lastSendTime) > m_interval)
    {
        m_bytesReceivedPerSecond = (int)(m_bytesReceivedPerInterval / (currentTime - m_lastSendTime));
        m_lastRecTime = currentTime;
        m_bytesReceivedPerInterval = 0;
    }
    return m_bytesReceivedPerSecond;
}

void VRBSClient::addBytesSent(int b)
{
    double currentTime = time();
    m_bytesSent += b;
    m_bytesSentPerInterval += b;
    if ((currentTime - m_lastSendTime) > m_interval)
    {
        m_bytesSentPerSecond = (int)(m_bytesSentPerInterval / (currentTime - m_lastSendTime));
        m_lastSendTime = currentTime;
        m_bytesSentPerInterval = 0;
    }
}

void VRBSClient::addBytesReceived(int b)
{
    double currentTime = time();
    m_bytesReceived += b;
    m_bytesReceivedPerInterval += b;
    if ((currentTime - m_lastSendTime) > m_interval)
    {
        m_bytesReceivedPerSecond = (int)(m_bytesReceivedPerInterval / (currentTime - m_lastSendTime));
        m_lastRecTime = currentTime;
        m_bytesReceivedPerInterval = 0;
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
//VRBClientList_______________________________________________
//////////////////////////////////////////////////////////////
VRBSClient *VRBClientList::get(Connection *c)
{
    for (auto &cl : m_clients)
    {
        if (cl->conn == c)
        {
            return cl.get();
        }
    }
    return nullptr;
}

int VRBClientList::numInSession(vrb::SessionID &session)
{
    int num = 0;
    for (auto &cl : m_clients)
    {
        if (cl->sessionID() == session)
        {
            ++num;
        }
    }
    return num;
}

size_t VRBClientList::numberOfClients()
{
    return m_clients.size();
}

void VRBClientList::addClient(VRBSClient * cl)
{
    m_clients.emplace_back(cl);
}

void VRBClientList::removeClient(VRBSClient * cl)
{
    auto it = std::find_if(m_clients.begin(), m_clients.end(), [cl](const std::unique_ptr<VRBSClient>& c) {
        return cl == c.get();
    });
    m_clients.erase(it);
}

void VRBClientList::remove(Connection * c)
{
    for (auto cl = m_clients.begin(); cl != m_clients.end();)
    {
        if (cl->get()->conn == c)
        {
            cl = m_clients.erase(cl);
        }
        else
        {
            ++cl;
        }
    }
}

void VRBClientList::passOnMessage(const covise::MessageBase* msg, const vrb::SessionID &session)
{
    if (session.isPrivate())
    {
        return;
    }
    for (auto &cl : m_clients)
    {
        if (cl->conn != msg->conn && (cl->sessionID() == session || session == vrb::SessionID(0, "", false)))
        {
            cl->send(msg);
        }
    }
}

void VRBClientList::broadcastMessageToProgramm(vrb::Program program, covise::MessageBase *msg)
{
    for (auto &cl : m_clients)
    {
        if (cl->userInfo().userType == program)
        {
            cl->send(msg);
        }
    }
}

void VRBClientList::collectClientInfo(covise::TokenBuffer & tb, const VRBSClient *recipient) const
{
    tb << (int)m_clients.size();
    auto recipientPos = tb.addPlaceHolder<int>();
    int i = 0;
    for (auto &cl : m_clients)
    {
        if (cl.get() == recipient)
        {
            recipientPos.replace(i);
            tb << recipient->ID() << recipient->sessionID() << recipient->getPrivateSession();
        }
        else
        {
            ++i;
            tb << *cl;
        }
        
    }
    assert(i == m_clients.size() - 1);
}

VRBSClient* VRBClientList::getNextPossibleFileOwner(const std::string& fileName, const vrb::SessionID& id)
{
	for (auto &cl : m_clients)
	{
		if (cl->sessionID() == id && !cl->doesNotKnowFile(fileName))
		{
			return cl.get();
		}
	}
	return nullptr;
}

VRBSClient *VRBClientList::get(const char *ip)
{
    auto it = std::find_if(m_clients.begin(), m_clients.end(), [ip](const std::unique_ptr<VRBSClient>& cl) {
        return cl->userInfo().ipAdress == ip;
        });
    return (it == m_clients.end() ? nullptr : (*it).get());
}

VRBSClient *VRBClientList::get(int id)
{
    auto it = std::find_if(m_clients.begin(), m_clients.end(), [id](const std::unique_ptr<VRBSClient>& cl) {
        return cl->ID() == id;
        });
    return (it == m_clients.end() ? nullptr : (*it).get());
}

VRBSClient * VRBClientList::getMaster(const vrb::SessionID &session)
{
    auto cl = get(session.master());
    if (cl && cl->sessionID() == session)
    {
        return cl;
    }
    return nullptr;
}

VRBSClient * VRBClientList::getNextInGroup(const vrb::SessionID & id)
{
    for (auto &cl : m_clients)
    {
        if (cl->sessionID() == id)
        {
            return cl.get();
        }
    }
    return nullptr;
}

VRBSClient * VRBClientList::getNthClient(size_t N)
{
    if (N >= m_clients.size())
    {
        return nullptr;
    }
    return m_clients[N].get();
}

std::vector<VRBSClient*> VRBClientList::getClientsWithUserName(const std::string & name)
{
    std::vector<VRBSClient *> clients;
    for (auto &cl : m_clients)
    {
        if (cl->userInfo().name == name)
        {
            clients.push_back(cl.get());
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
    for (auto &cl : m_clients)
    {
        if (cl->sessionID() == client->sessionID())
        {
            cl->setMaster(client->ID());
        }
    }
}

void VRBClientList::setInterval(float i)
{
    for (auto &cl : m_clients)
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

    for (auto &cl : m_clients)
    {
        if (group == vrb::SessionID(0, std::string(), false) || group == cl->sessionID())
        {
            cl->send(&m);
        }
    }
}

void VRBClientList::sendMessageToClient(VRBSClient* cl, TokenBuffer &tb, covise_msg_type type){
    Message m(tb);
    m.type = type;
    if (cl)
    {
		cl->send(&m);
    }
}

void VRBClientList::sendMessageToClient(int clientID, TokenBuffer &tb, covise_msg_type type)
{
    VRBSClient *cl = get(clientID);
    sendMessageToClient(cl, tb, type);
}

void VRBClientList::sendMessageToAll(covise::TokenBuffer &stb, covise::covise_msg_type type)
{
    Message m(stb);
    m.type = type;

    for (auto &cl : m_clients)
    {
		cl->send(&m);
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