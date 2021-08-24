#include <util/coExport.h>
#include <string>
#include <memory>
#include "remoteSound.h"
#include <net/tokenbuffer.h>
#include <util/UDP_Sender.h>

#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>
#endif

namespace covise
{
    class ClientConnection;
    class Host;
    class Message;
}


namespace remoteSound
{

    class RSEXPORT Client
    {

    public:
        Client(std::string hostname, int port, std::string Application, std::string user="");
        ~Client();
        bool connectToServer();
        bool completeConnection();
        Sound* getSound(const std::string filename);
        void update();

        bool isConnected();
        bool send(covise::TokenBuffer& tb);
        int send(const void* buf, size_t size);
        covise::Message *receiveMessage();
        covise::UDP_Sender *udpSender;


        std::list<remoteSound::Sound*> sounds;
    private:
        std::unique_ptr<covise::ClientConnection> sConn=nullptr; // tcp connection to Server
        covise::Host* serverHost = nullptr;
        std::string hostname;
        int port;
        std::string Application;
        std::string user;
        bool sendMessage(const covise::Message* m) const;

        std::mutex connMutex;
#ifndef _M_CEE //no future in Managed OpenCOVER
        std::future<std::unique_ptr<covise::ClientConnection>> connFuture;
#endif
        std::atomic_bool m_shutdown{ false };
    };
}