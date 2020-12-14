/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_H
#define _VRB_CLIENT_H
#include "VrbCredentials.h"

#include <vrb/RemoteClient.h>
#include <net/message_sender_interface.h>

#include <mutex>
#include <string>
#include <list>
#include <util/coTypes.h>
#include <stdio.h>
#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>
#endif
namespace covise
{
class Host;
class CoviseConfig;
class Connection;
class ClientConnection;
class UDPConnection;
class Message;
class TokenBuffer;
class UdpMessage;
enum udp_msg_type : int;
}
namespace vrb{

class VRBCLIENTEXPORT VRBClient : public vrb::RemoteClient, public covise::MessageSenderInterface
{

public:
    VRBClient(Program p, const char *collaborativeConfigurationFile = NULL, bool isSlave = false);
    VRBClient(Program p, const VrbCredentials &credentials, bool isSlave = false);
    ~VRBClient();
    bool connectToServer(std::string sessionName = ""); 
    bool completeConnection();

    bool isConnected();
    bool poll(covise::Message *m);
	bool pollUdp(covise::UdpMessage* m);
    int wait(covise::Message *m);
    int wait(covise::Message *m, int messageType);


	void setupUdpConn();
    std::list<covise::Message *> messageQueue;
    float getSendDelay();
    void shutdown(); //threadsafe, shuts down the tcp socked, don't use the client after a call to this function
    const VrbCredentials &getCredentials();

private:
    covise::ClientConnection *sConn = nullptr; // tcp connection to Server

	covise::UDPConnection* udpConn = nullptr; //udp connection to server

    VrbCredentials m_credentials;
    covise::Host *serverHost = nullptr;
    bool isSlave = false; // it true, we are a slave in a multiPC config, so do not actually connect to server
    float sendDelay = 0.1f; // low-pass filtered time for sending one packet of 1000 bytes
    std::mutex connMutex;
    std::atomic_bool m_isConnected{false};
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::future<covise::ClientConnection *> connFuture;
	std::future<covise::UDPConnection*> udpConnFuture;
#endif
    bool firstVrbConnection = true;
	std::mutex udpConnMutex;
    bool firstUdpVrbConnection = true;

    bool sendMessage(const covise::Message* m) override;
    bool sendMessage(const covise::UdpMessage *m) override;
};
VrbCredentials readcollaborativeConfigurationFile(const char *collaborativeConfigurationFile);
}
#endif
