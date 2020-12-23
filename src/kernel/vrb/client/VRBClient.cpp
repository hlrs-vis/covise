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

#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/udpMessage.h>
#include <net/udp_message_types.h>
#include <util/common.h>

using namespace vrb;
using namespace covise;

VRBClient::VRBClient(vrb::Program p, const char *collaborativeConfigurationFile, bool slave)
    : VRBClient(p, readcollaborativeConfigurationFile(collaborativeConfigurationFile), slave)
    {}

VRBClient::VRBClient(vrb::Program p, const vrb::VrbCredentials &credentials, bool slave)
    : vrb::RemoteClient(p)
    , m_credentials(credentials)
	, isSlave(slave)
{
    if (!credentials.ipAddress.empty())
    {
        serverHost = new Host(credentials.ipAddress.c_str());
    }
}

VRBClient::~VRBClient()
{
    delete sConn;
	delete udpConn;
}

bool VRBClient::poll(Message *m)
{
    if (isSlave)
        return false;
    if (sConn->check_for_input())
    {
        sConn->recv_msg(m);
        return true;
    }
	return false;


}
bool VRBClient::pollUdp(covise::UdpMessage* m)
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
    if (messageQueue.size())
    {
        *m = *(messageQueue.front()); // copy message
        messageQueue.remove(messageQueue.front());
        delete m;
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
            messageQueue.push_back(msg);
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
			messageQueue.push_back(msg);
		}
        }


    }
    return ret;
}


bool VRBClient::sendMessage(const covise::UdpMessage* m)
{
	if (!udpConn) // not connected to a server
	{
		return 0;
	}
    m->sender = ID();
	return udpConn->send_udp_msg(m);

}

bool VRBClient::sendMessage(const Message* m)
{
    if (!sConn)
    {
        return false;
    }
    
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

	sConn->sendMessage(m);
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

bool VRBClient::isConnected()
{
    if (isSlave)
        return 1;
    if (sConn == NULL)
    {
        return completeConnection();
    }
    return sConn->is_connected();
}

bool VRBClient::connectToServer(std::string sessionName)
{
    setSession(vrb::SessionID{ID(), sessionName});
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
            ClientConnection *myConn = new ClientConnection(serverHost, m_credentials.tcpPort, 0, (sender_type)0,0, 1.0);
            if (!myConn->is_connected()) // could not open server port
            {
                if (firstVrbConnection)
                {
                    fprintf(stderr, "Could not connect to server on %s; port %d\n", serverHost->getAddress(), m_credentials.tcpPort);
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
    completeConnection();


    if (!sConn)
    {
        return false;
    }
    return true;
}

bool VRBClient::completeConnection(){
    if (connFuture.valid())
    {
        std::lock_guard<std::mutex> g(connMutex);
        auto status = connFuture.wait_for(std::chrono::seconds(0));
        if (status == std::future_status::ready)
        {
            sConn = connFuture.get();
            if(sConn != nullptr)
            {
                m_isConnected = true;
                TokenBuffer tb;
                tb << *this;
                Message msg(tb);
                msg.type = COVISE_MESSAGE_VRB_CONTACT;
                sConn->sendMessage(&msg);
                return true;
            }
        }
    }
    return false;
}

void VRBClient::setupUdpConn() 
{
    if(serverHost)
    {
	udpConn = new UDPConnection(0, 0, m_credentials.udpPort, serverHost->getAddress());
    }
}

float VRBClient::getSendDelay()
{
    return sendDelay;
}

void VRBClient::shutdown(){
    if(m_isConnected)
    {
        int id = sConn->getSocket()->get_id();
        ::shutdown(id, 2); //2 stands for SHUT_RDWR/SD_BOTH 
    }
}

const vrb::VrbCredentials &VRBClient::getCredentials(){
    return m_credentials;
}

vrb::VrbCredentials vrb::readcollaborativeConfigurationFile(const char *collaborativeConfigurationFile) {
    if (collaborativeConfigurationFile != NULL)
    {
        FILE *fp = fopen(collaborativeConfigurationFile, "r");
        if (fp)
        {
            std::string ipAddress;
            char buf[5000];
            int tcpPort = 0;
            while (!feof(fp))
            {
                char *retval_fgets;
                retval_fgets = fgets(buf, 5000, fp);
                if (retval_fgets == NULL)
                {
                    std::cerr << "VRBClient::VRBClient: fgets failed" << std::endl;
                    return vrb::VrbCredentials{};
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
                        return vrb::VrbCredentials{};
                    }
                    if (strcasecmp(hostName, "NONE"))
                    {
                        ipAddress = hostName;
                    }
                    
                    delete[] hostName;
                }
                else if (strncmp(buf, "port:", 5) == 0)
                {
                    size_t retval;
                    retval = sscanf(buf, "port:%d", &tcpPort);
                    if (retval != 1)
                    {
                        std::cerr << ": sscanf failed" << std::endl;
                        return vrb::VrbCredentials{};
                    }
                }
            }
            fclose(fp);
            return vrb::VrbCredentials{ipAddress, static_cast<unsigned int>(tcpPort), static_cast<unsigned int>(tcpPort + 1)};
        }
    }
    return vrb::VrbCredentials{};
}
