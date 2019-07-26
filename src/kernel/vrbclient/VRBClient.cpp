/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRBClient.h"
#include <stdio.h>
#include <string.h>
#include <util/unixcompat.h>

#ifdef _WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

#include <util/common.h>
#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/tokenbuffer.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/udpMessage.h>
#include <net/udp_message_types.h>
#include <net/message_types.h>

using namespace covise;

VRBClient::VRBClient(const char *n, const char *collaborativeConfigurationFile, bool slave)
    : sendDelay(0.1f)
	,isSlave(slave)
{
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    if (collaborativeConfigurationFile != NULL)
    {
        FILE *fp = fopen(collaborativeConfigurationFile, "r");
        if (fp)
        {
            char buf[5000];
            while (!feof(fp))
            {
                char *retval_fgets;
                retval_fgets = fgets(buf, 5000, fp);
                if (retval_fgets == NULL)
                {
                    std::cerr << "VRBClient::VRBClient: fgets failed" << std::endl;
                    return;
                }
                if (strncmp(buf, "server:", 7) == 0)
                {
                    char *hostName = new char[strlen(buf) + 1];
                    size_t retval;
                    retval = sscanf(buf, "server:%s", hostName);
                    if (retval != 1)
                    {
                        std::cerr << "VRBClient::VRBClient: sscanf failed" << std::endl;
                        delete[] hostName;
                        return;
                    }
                    serverHost = new Host(hostName);
                    delete[] hostName;
                }
                else if (strncmp(buf, "port:", 5) == 0)
                {
                    size_t retval;
                    retval = sscanf(buf, "port:%d", &m_tcpPort);
                    if (retval != 1)
                    {
                        std::cerr << ": sscanf failed" << std::endl;
                        return;
                    }
					m_udpPort = m_tcpPort + 1;
                }
            }
            fclose(fp);
        }
    }
    if (serverHost == NULL)
    {
		m_tcpPort = coCoviseConfig::getInt("tcpPort", "System.VRB.Server", 31800);
		m_udpPort = coCoviseConfig::getInt("udpPort", "System.VRB.Server", m_tcpPort + 1);
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
    }
}

VRBClient::VRBClient(const char *n, const char *host, int tcp_p, int udp_p, bool slave)
    : sendDelay(0.1f)
	, isSlave(slave)
{
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    serverHost = NULL;
    //port = 31800;
    m_tcpPort = tcp_p;
	m_udpPort = udp_p;
    if (host && strcmp(host, ""))
    {
        serverHost = new Host(host);
    }
    if (serverHost == NULL)
    {
        std::string line = coCoviseConfig::getEntry("VRB.TCPPort");
        if (!line.empty())
        {
            size_t retval;
            retval = sscanf(line.c_str(), "%d", &m_tcpPort);
            if (retval != 1)
            {
                std::cerr << "VRBClient::VRBClient: sscanf failed" << std::endl;
                return;
            }
        }
        line = coCoviseConfig::getEntry("VRB.Server");
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
    }
    sConn = NULL;
    ID = -1;
}

VRBClient::~VRBClient()
{
    delete[] name;
    delete sConn;

	delete udpConn;

}

int VRBClient::getID()
{
    return ID;
}

int VRBClient::poll(Message *m)
{
    if (isSlave)
        return 0;
    if (sConn->check_for_input())
    {
        sConn->recv_msg(m);
        return 1;
    }
	return 0;


}
bool VRBClient::pollUdp(vrb::UdpMessage* m)
{
	if (!udpConn)
	{
		return false;
	}
	return udpConn->recv_udp_msg(m);
}
int VRBClient::wait(Message *m)
{
#ifdef MB_DEBUG
    std::cerr << "VRBCLIENT::MESSAGEWAIT: Message: " << m->type << std::endl;
    std::cerr << "VRBCLIENT::MESSAGEWAIT: Data: " << m->data << std::endl;
#endif
    if (isSlave)
        return 0;
    if (messageQueue.num())
    {
        messageQueue.reset();
        m = messageQueue.current();
        messageQueue.remove();
        return 1;
    }

    return sConn->recv_msg(m);
}

int VRBClient::wait(Message *m, int messageType)
{
#ifdef MB_DEBUG
    if (m->type != COVISE_MESSAGE_EMPTY)
    {
        std::cerr << "VRBCLIENT::MESSAGEWAIT: Message: " << m->type << std::endl;
        std::cerr << "VRBCLIENT::MESSAGEWAIT: Data: " << m->data << std::endl;
        std::cerr << "VRBCLIENT::MESSAGEWAIT: Type: " << messageType << std::endl;
    }
#endif
    int ret = 1;
    while (ret > 0)
    {
        ret = sConn->recv_msg(m);
        if (m->type == messageType)
        {
            return ret;
        }
        else
        {
            Message *msg = new Message(*m);
            messageQueue.append(msg);
        }
        if(udpConn)
        {
		ret = udpConn->recv_msg(m);
		if (m->type == messageType)
		{
			return ret;
		}
		else
		{
			Message* msg = new Message(*m);
			messageQueue.append(msg);
		}
        }


    }
    return ret;
}

int VRBClient::setUserInfo(const char *userInfo)
{
    if (serverHost == NULL)
        return 0;
    if (!sConn || (!sConn->is_connected())) // not connected to a server
    {
        return 0;
    }
    TokenBuffer tb;
    tb << userInfo;
    Message msg(tb);
    msg.type = COVISE_MESSAGE_VRB_SET_USERINFO;
    sConn->send_msg(&msg);
    return 1;
}

int VRBClient::sendMessage(const Message *m)
{
	if (!sConn || (!sConn->is_connected())) // not connected to a server
	{
		return 0;
	}
	return sendMessage(m, sConn);
}

void covise::VRBClient::sendMessage(TokenBuffer & tb, int type)
{
    Message m(tb);
    m.type = type;
    sendMessage(&m);
}

int VRBClient::sendUdpMessage(const vrb::UdpMessage* m)
{
	if (!udpConn) // not connected to a server
	{
		return 0;
	}
    m->sender = ID;
	return udpConn->send_udp_msg(m);

}

void covise::VRBClient::sendUdpMessage(TokenBuffer& tb, vrb::udp_msg_type type, int sender)
{
	vrb::UdpMessage m(tb);
	m.type = type;
	m.sender = sender;
	sendUdpMessage(&m);
}


int covise::VRBClient::sendMessage(const Message* m, Connection* conn)
{
#ifdef MB_DEBUG
	std::cerr << "VRBCLIENT::SENDMESSAGE: Sending Message: " << m->type << std::endl;
	std::cerr << "VRBCLIENT::SENDMESSAGE: Data: " << m->data << std::endl;
#endif

	if (serverHost == NULL)
		return 0;
	//cerr << "sendMessage " << conn << endl;

#ifdef _MSC_VER
	struct __timeb64 start;
	_ftime64(&start);
#else
	struct timeval start, end;
	gettimeofday(&start, NULL);
#endif

	conn->send_msg(m);
#ifdef _MSC_VER
	struct __timeb64 end;
	_ftime64(&end);
#else
	gettimeofday(&end, NULL);
#endif

#ifdef _MSC_VER
	float delay = (end.millitm - start.millitm) * 1e-6f + end.time - start.time;
#else
	float delay = (end.tv_usec - start.tv_usec) * 1e-6f + end.tv_sec - start.tv_sec;
#endif

	if (m->data.length() > 1000)
		delay /= m->data.length() / 1000;
	sendDelay = 0.99f * sendDelay + delay * 0.01f;
	return 1;
}

int VRBClient::isConnected()
{
    if (isSlave)
        return 1;
    if (sConn == NULL)
        return 0;
    return sConn->is_connected();
}

int VRBClient::connectToServer(const std::string& sessionName)
{
	if (!udpConn && !isSlave)
	{
		setupUdpConn();
	}
	if (isSlave || serverHost == NULL|| sConn != nullptr)
        return 0;

    connMutex.lock();
    if (!connFuture.valid())
    {
        connFuture = std::async(std::launch::async, [this]() -> ClientConnection *
        {
            ClientConnection *myConn = new ClientConnection(serverHost, m_tcpPort, 0, (sender_type)0,0, 1.0);
            if (!myConn->is_connected()) // could not open server port
            {
                if (firstVrbConnection)
                {
                    fprintf(stderr, "Could not connect to server on %s; port %d\n", serverHost->getAddress(), m_tcpPort);
                    firstVrbConnection = false;
                }

                delete myConn;
                myConn = NULL;
                sleep(1);
                return nullptr;
            }
            struct linger linger;
            linger.l_onoff = 0;
            linger.l_linger = 1;
            setsockopt(myConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

            return myConn;
        });
    }
    connMutex.unlock();
    if (connFuture.valid())
    {
        connMutex.lock();
        auto status = connFuture.wait_for(std::chrono::seconds(0));
        if (status == std::future_status::ready)
        {
            sConn = connFuture.get();
            if(sConn != nullptr)
            {
                Host host;

                TokenBuffer tb;
				if (sessionName == "")
				{
					tb << name;
					tb << host.getAddress();
				}
				else
				{
					tb << "-g";
					tb << host.getAddress();
					tb << sessionName.c_str();
				}


                Message msg(tb);
                msg.type = COVISE_MESSAGE_VRB_CONTACT;
                sConn->send_msg(&msg);
            }
        }
        connMutex.unlock();
    }


    if (!sConn)
    {
        return false;
    }
    return true;
}


void VRBClient::setupUdpConn() 
{
    if(serverHost)
    {
	udpConn = new UDPConnection(0, 0, m_udpPort, serverHost->getAddress());
    }
}

void VRBClient::setID(int i)
{
    ID = i;
}

int VRBClient::isCOVERRunning()
{
    if (serverHost == NULL)
        return 0;
    if (!sConn || (!sConn->is_connected())) // could not open server port
    {
        return 0;
    }
    int i;
    Host host;
    TokenBuffer tb;
    tb << host.getAddress();
    Message msg(tb);
    msg.type = COVISE_MESSAGE_VRB_CHECK_COVER;
    sConn->send_msg(&msg);
    if (sConn->check_for_input(5))
    {
        sConn->recv_msg(&msg);
        if (msg.type == COVISE_MESSAGE_VRB_CHECK_COVER)
        {
            TokenBuffer rtb(&msg);
            rtb >> i;
            return i;
        }
    }
    return 0;
}

void VRBClient::connectToCOVISE(int argc, const char **argv)
{
    if (serverHost == NULL)
    {
        std::cerr << "can't send Message, because not connected to Server" << std::endl;
        return;
    }
    if (!sConn || (!sConn->is_connected())) // could not open server port
    {
        std::cerr << "can't send Message, because not connected to Server" << std::endl;
        return;
    }
    int i;

    Host host;
    TokenBuffer tb;
    tb << host.getAddress();
    tb << argc;
    for (i = 0; i < argc; i++)
        tb << argv[i];
    Message msg(tb);
    msg.type = COVISE_MESSAGE_VRB_CONNECT_TO_COVISE;
    sConn->send_msg(&msg);
}

float VRBClient::getSendDelay()
{
    return sendDelay;
}
