/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_H
#define _VRB_CLIENT_H
#include "VrbCredentials.h"

#include <vrb/RemoteClient.h>

#include <mutex>
#include <string>
#include <list>
#include <util/coTypes.h>
#include <stdio.h>
#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>
#endif
namespace vrb
{
	class UdpMessage;
	enum udp_msg_type : int;
}
namespace covise
{

class Host;
class CoviseConfig;
class Connection;
class ClientConnection;
class UDPConnection;
class Message;
class TokenBuffer;


class VRBCLIENTEXPORT VRBClient : public vrb::RemoteClient
{

public:
    VRBClient(vrb::Program p, const char *collaborativeConfigurationFile = NULL, bool isSlave = false);
    VRBClient(vrb::Program p, const vrb::VrbCredentials &credentials, bool isSlave = false);
    ~VRBClient();
    bool connectToServer(std::string sessionName = ""); 
    bool completeConnection();

    void connectToCOVISE(int argc, const char **argv);
    bool isCOVERRunning();
    bool isConnected();
    bool poll(Message *m);
	bool pollUdp(vrb::UdpMessage* m);
    int wait(Message *m);
    int wait(Message *m, int messageType);
    //bool sendUserInfo(const char *userInfo);
    bool sendMessage(const Message *m);
    bool sendMessage(TokenBuffer &tb, int type);
	bool sendUdpMessage(const vrb::UdpMessage* m);
	bool sendUdpMessage(TokenBuffer& tb, vrb::udp_msg_type type, int sender);

	void setupUdpConn();
    std::list<Message *> messageQueue;
    float getSendDelay();
    void shutdown(); //threadsafe, shuts down the tcp socked, don't use the client after a call to this function
    const vrb::VrbCredentials &getCredentials();

private:
    ClientConnection *sConn = nullptr; // tcp connection to Server

	UDPConnection* udpConn = nullptr; //udp connection to server

    vrb::VrbCredentials m_credentials;
    Host *serverHost = nullptr;
    bool isSlave = false; // it true, we are a slave in a multiPC config, so do not actually connect to server
    float sendDelay = 0.1f; // low-pass filtered time for sending one packet of 1000 bytes
    std::mutex connMutex;
    std::atomic_bool m_isConnected{false};
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::future<ClientConnection *> connFuture;
	std::future<UDPConnection*> udpConnFuture;
#endif
    bool firstVrbConnection = true;
	std::mutex udpConnMutex;
    bool firstUdpVrbConnection = true;

    bool sendMessage(const Message* m, Connection* conn);
};
vrb::VrbCredentials readcollaborativeConfigurationFile(const char *collaborativeConfigurationFile);
}
#endif
