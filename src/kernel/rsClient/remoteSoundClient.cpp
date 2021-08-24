#include "remoteSoundClient.h"
#include "remoteSound.h"
#include "../../sys/carSound/remoteSoundMessages.h"

#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message_types.h>
#include <util/UDP_Sender.h>

#include <QtDebug>

using namespace remoteSound;
Client::Client(std::string h, int p, std::string a, std::string u)
{
    Application = a;
    user = u;
    hostname = h;
    port = p;
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

    connectToServer();
}

Client::~Client()
{
    delete udpSender;
}

bool remoteSound::Client::connectToServer()
{
    serverHost = new covise::Host(hostname.c_str());
    if (sConn || !serverHost)
        return false;
    connMutex.lock();
    if (!connFuture.valid())
    {
            connFuture = std::async(std::launch::async, [this]() -> std::unique_ptr<covise::ClientConnection>
            {
                bool firstConnectionAttempt = true;
                m_shutdown = false;
                while (!m_shutdown)
                {
                    auto myConn = std::unique_ptr<covise::ClientConnection>{ new covise::ClientConnection(serverHost, port, 0, (covise::sender_type)0, 0, 1.0) };
                    if (!myConn->is_connected()) // could not open server port
                    {
                            fprintf(stderr, "Could not connect to remote sound server on %s; port %d\n", serverHost->getAddress(), port);
                            firstConnectionAttempt = false;
                    }
                    else
                    {
                        struct linger linger;
                        linger.l_onoff = 0;
                        linger.l_linger = 1;
                        setsockopt(myConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char*)&linger, sizeof(linger));
                        return myConn;
                    }
                }
                return nullptr;
            });
    }
    connMutex.unlock();
    completeConnection();

    return true;
}


bool remoteSound::Client::completeConnection() {
    if (connFuture.valid())
    {
        std::lock_guard<std::mutex> g(connMutex);
        auto status = connFuture.wait_for(std::chrono::seconds(0));
        if (status == std::future_status::ready)
        {
            sConn = connFuture.get();
            if (sConn != nullptr)
            {
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
                for (const auto &s : sounds)
                {
                    s->resend();
                }
                return true;
            }
        }
    }
    return false;
}

Sound* remoteSound::Client::getSound(const std::string filename)
{
    return new Sound(this,filename);
}


void Client::update()
{
    if (!isConnected())
    {
        connectToServer();
    }
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
