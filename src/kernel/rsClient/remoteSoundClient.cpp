#include "remoteSoundClient.h"
#include "remoteSound.h"
#include "../../sys/carSound/remoteSoundMessages.h"

#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message_types.h>
#include <util/UDP_Sender.h>

#include <QtDebug>

using namespace remoteSound;
Client::Client(std::string hostname, int port, std::string a, std::string u)
{
    Application = a;
    user = u;
    udpSender = new covise::UDP_Sender(hostname.c_str(), 31804);
    if (user == "")
    {
        // get local user name
#ifdef _WIN32
        user = getenv("USERNAME");
#else
        char* login = getlogin();
        if (!login)
        {
            qWarning() << "Getting user name failed";
            login = getenv("USER");
        }
        if (login)
            user = login;
        else
            user = "unknown";
#endif
    }

    connectToServer(hostname,port);
}

Client::~Client()
{
    delete udpSender;
}

bool remoteSound::Client::connectToServer(std::string hostname, int port)
{
    serverHost = new covise::Host(hostname.c_str());
    sConn = std::unique_ptr<covise::ClientConnection>{ new covise::ClientConnection(serverHost, port, 0, (covise::sender_type)0, 0, 1.0) };
    if (!sConn->is_connected()) // could not open server port
    {
            fprintf(stderr, "Could not connect to remote sound server on %s; port %d\n", serverHost->getAddress(), port);
            return false;
    }
    else
    {
        struct linger linger;
        linger.l_onoff = 0;
        linger.l_linger = 1;
        setsockopt(sConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char*)&linger, sizeof(linger));
    }
    covise::Host localhost;
    covise::TokenBuffer tb;
    tb << (int)SoundMessages::SOUND_CLIENT_INFO;
    tb << Application;
    tb << user;
    tb << localhost.getHostname();
    tb << localhost.getAddress();
    covise::Message* m = new covise::Message(tb);
    m->type = covise::COVISE_MESSAGE_SOUND;
    sendMessage(m);
    return true;
}

Sound* remoteSound::Client::getSound(const std::string filename)
{
    return new Sound(this,filename);
}

bool Client::isConnected()
{
    if (sConn.get() == nullptr)
        return false;
    return sConn->is_connected();
}

int Client::send(const void *buf, size_t size)
{
    return udpSender->send(buf,(int)size);
}
bool Client::send(covise::TokenBuffer& tb)
{
    covise::Message* m = new covise::Message(tb);
    m->type = covise::COVISE_MESSAGE_SOUND;
    return sendMessage(m);
}

covise::Message* remoteSound::Client::receiveMessage()
{
    covise::Message* m=new covise::Message();
    m->type = covise::COVISE_MESSAGE_QUIT;
    int res = sConn->recv_msg(m);
    return m;
    
}

bool Client::sendMessage(const covise::Message* m) const
{
    return sConn->sendMessage(m);
}
