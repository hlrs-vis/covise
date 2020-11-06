#ifndef VRB_REMOTE_LAUNCHER_MESSAGE_HANDLER_H
#define VRB_REMOTE_LAUNCHER_MESSAGE_HANDLER_H

#include "MessageTypes.h"
#include "export.h"

#include <vrb/client/RemoteClient.h>
#include <vrb/SessionID.h>
#include <vrb/client/VrbCredentials.h>
#include <vrb/client/VRBClient.h>

#include <QObject>
#include <QString>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
namespace covise
{
    class VRBClient;
}
namespace vrb{
namespace launcher
{
    class REMOTELAUNCHER_EXPORT VrbRemoteLauncher : public QObject
    {
        Q_OBJECT
    public:
        ~VrbRemoteLauncher();
        void connect(const vrb::VrbCredentials &credentials = vrb::VrbCredentials{});
        void disconnect();
        void printClientInfo();
    public slots:
        void sendLaunchRequest(Program p, int lientID, const std::vector<std::string> &args = std::vector<std::string>{});
    signals:
        void launchSignal(vrb::launcher::Program programID, std::vector<std::string> startOptions); //namespace must be explicitly stated for qRegisterMetaType
        void connectedSignal();
        void disconnectedSignal();
        void updateClient(int clientID, QString clientInfo);
        void removeClient(int id);

    private:
        typedef std::lock_guard<std::mutex> Guard;
        std::atomic<bool> m_terminate{false};
        std::atomic<bool> m_shouldBeConnected{false};
        std::unique_ptr<vrb::VrbCredentials> m_credentials;
        bool m_newCredentials = false;
        std::unique_ptr<covise::VRBClient> m_client = nullptr;
        std::unique_ptr<std::thread> m_thread;
        std::mutex m_mutex;

        vrb::SessionID m_sessionID;
        vrb::RemoteClient m_me;
        std::vector<std::unique_ptr<vrb::RemoteClient>> m_clientList; //default constructor is only for me
        void loop();
        bool handleVRB();
        bool removeOtherClient(covise::TokenBuffer &tb);
        bool setOtherClientInfo(covise::TokenBuffer &tb);
        void handleVrbLauncherMessage(covise::TokenBuffer &tb);
        void setMyIDs(covise::TokenBuffer &tb);
        std::vector<std::unique_ptr<vrb::RemoteClient>>::iterator findClient(int id);

    };

    QString getClientInfo(const vrb::RemoteClient &cl);

} // namespace launcher
} //vrb

Q_DECLARE_METATYPE(std::vector<std::string>);


#endif