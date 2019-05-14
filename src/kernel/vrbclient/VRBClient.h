/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_H
#define _VRB_CLIENT_H
#include <stdio.h>
#include <future>
#include <mutex>
#include <util/DLinkList.h>
#include <util/coTypes.h>



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
    VRBClient(const char *name, const char *host, int pPort, bool isSlave = false);
    ~VRBClient();
    int connectToServer(); // returns -1, if Connection to Server fails
    void connectToCOVISE(int argc, const char **argv);
    int isCOVERRunning();
    int isConnected();
    int poll(Message *m);
    int wait(Message *m);
    int wait(Message *m, int messageType);
    int setUserInfo(const char *userInfo);
    int sendMessage(const Message *m);
    void sendMessage(TokenBuffer &tb, int type);

	int connectToUdpServer(); // returns -1, if Connection to Server fails, uses port +1 (fix me: define udp port in config)
	void setupUdpConn();
	int sendUdpMessage(const Message* m);
	void sendUdpMessage(TokenBuffer& tb, int type);

    int getID();
    void setID(int ID);
    DLinkList<Message *> messageQueue;
    float getSendDelay();

private:
    ClientConnection *sConn = nullptr; // tcp connection to Server

	UDPConnection* udpConn = nullptr; //udp connection to server

    char *name;
    int port = 31800;
	int udpPort = 31801; //port + 1 for up (fix me: define up port in config)
    int ID = -1;
    Host *serverHost = nullptr;
    bool isSlave; // it true, we are a slave in a multiPC config, so do not actually connect to server
    float sendDelay; // low-pass filtered time for sending one packet of 1000 bytes
    std::mutex connMutex;
    std::future<ClientConnection *> connFuture;
    bool firstVrbConnection = true;

	std::mutex udpConnMutex;
	std::future<UDPConnection*> udpConnFuture;
	bool firstUdpVrbConnection = true;

	int sendMessage(const Message* m, Connection* conn);
};
}
#endif
