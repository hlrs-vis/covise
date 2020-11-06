/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_H
#define _VRB_CLIENT_H
#include <stdio.h>
#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>
#endif
#include <mutex>
#include <string>
#include <list>
#include <util/coTypes.h>


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

//
//
//

class VRBEXPORT VRBClient
{

public:
    VRBClient(const char *name, const char *collaborativeConfigurationFile = NULL, bool isSlave = false);
    VRBClient(const char *name, const char *host, int tcp_p, int udp_p, bool isSlave = false);
    ~VRBClient();
    int connectToServer(std::string sessionName = ""); // returns -1, if Connection to Server fails
    bool completeConnection();

    void connectToCOVISE(int argc, const char **argv);
    int isCOVERRunning();
    int isConnected();
    int poll(Message *m);
	bool pollUdp(vrb::UdpMessage* m);
    int wait(Message *m);
    int wait(Message *m, int messageType);
    int setUserInfo(const char *userInfo);
    int sendMessage(const Message *m);
    void sendMessage(TokenBuffer &tb, int type);

	void setupUdpConn();
	int sendUdpMessage(const vrb::UdpMessage* m);
	void sendUdpMessage(TokenBuffer& tb, vrb::udp_msg_type type, int sender);

    int getID();
    void setID(int ID);
    std::list<Message *> messageQueue;
    float getSendDelay();

private:
    ClientConnection *sConn = nullptr; // tcp connection to Server

	UDPConnection* udpConn = nullptr; //udp connection to server

    char *name;
    int m_tcpPort = 31800;
	int m_udpPort = 31801; //port + 1 for up (fix me: define up port in config)
    int ID = -1;
    Host *serverHost = nullptr;
    bool isSlave; // it true, we are a slave in a multiPC config, so do not actually connect to server
    float sendDelay; // low-pass filtered time for sending one packet of 1000 bytes
    std::mutex connMutex;
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::future<ClientConnection *> connFuture;
	std::future<UDPConnection*> udpConnFuture;
#endif
    bool firstVrbConnection = true;
	std::mutex udpConnMutex;
    std::string startupSession;
    bool firstUdpVrbConnection = true;

    int sendMessage(const Message* m, Connection* conn);
};
}
#endif
