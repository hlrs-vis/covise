/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_H
#define _VRB_CLIENT_H
#include "VrbCredentials.h"

#include <vrb/RemoteClient.h>
#include <net/message_sender_interface.h>
#include <net/covise_connect.h>

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
class Message;
class TokenBuffer;
class UdpMessage;
enum udp_msg_type : int;
}
namespace vrb{

class VRBCLIENTEXPORT VRBClient : public vrb::RemoteClient, public covise::MessageSenderInterface
{

public:
    VRBClient(covise::Program p, const char *collaborativeConfigurationFile = NULL, bool isSlave = false, bool useUDP=false);
    VRBClient(covise::Program p, const VrbCredentials &credentials, bool isSlave = false, bool useUDP=false);
    VRBClient(covise::Program p, covise::MessageSenderInterface *sender, bool isSlave = false, bool useUDP=false);
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
    const VrbCredentials &getCredentials() const;

private:
    std::unique_ptr<covise::ClientConnection> sConn; // tcp connection to Server
    std::unique_ptr<covise::UDPConnection> udpConn; //udp connection to server

    covise::MessageSenderInterface *m_sender = nullptr;
    VrbCredentials m_credentials;
    covise::Host *serverHost = nullptr;
    bool isSlave = false; // it true, we are a slave in a multiPC config, so do not actually connect to server
    bool useUDP = false; // only setup a udp connection if this is true, this should be true only in OpenCOVER (and only one instance per computer :-(
    mutable float sendDelay = 0.1f; // low-pass filtered time for sending one packet of 1000 bytes
    std::mutex connMutex;
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::future<std::unique_ptr<covise::ClientConnection>> connFuture;
	std::future<std::unique_ptr<covise::UDPConnection>> udpConnFuture;
#endif
    std::atomic_bool m_shutdown{false};

    bool sendMessage(const covise::Message* m) const override;
    bool sendMessage(const covise::UdpMessage *m) const override;
};
VrbCredentials readcollaborativeConfigurationFile(const char *collaborativeConfigurationFile);
}
#endif
