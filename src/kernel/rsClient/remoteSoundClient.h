#include <util/coExport.h>
#include <string>
#include <memory>
#include "remoteSound.h"
#include <net/tokenbuffer.h>
#include <util/UDP_Sender.h>

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
        bool connectToServer(std::string hostname, int port);
        Sound* getSound(const std::string filename);

        bool isConnected();
        bool send(covise::TokenBuffer& tb);
        int send(const void* buf, size_t size);
        covise::Message *receiveMessage();
        covise::UDP_Sender *udpSender;


    private:
        std::unique_ptr<covise::ClientConnection> sConn; // tcp connection to Server
        covise::Host* serverHost = nullptr;
        std::string Application;
        std::string user;
        bool sendMessage(const covise::Message* m) const;
    };
}