#ifndef VRB_REMOTE_LAUNCHER_MESSAGE_HANDLER_H
#define VRB_REMOTE_LAUNCHER_MESSAGE_HANDLER_H

#include "metaTypes.h"

#include <vrb/ProgramType.h>
#include <vrb/client/VrbCredentials.h>
#include <vrb/client/VRBClient.h>

#include <QObject>
#include <QString>
#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

class VrbRemoteLauncher : public QObject
{
    Q_OBJECT
public:
    ~VrbRemoteLauncher();
    void connect(const vrb::VrbCredentials &credentials = vrb::VrbCredentials{});
    void disconnect();
    void printClientInfo();
public slots:
    void sendLaunchRequest(vrb::Program p, int lientID, const std::vector<std::string> &args = std::vector<std::string>{});
signals:
    void launchSignal(vrb::Program programID, std::vector<std::string> startOptions); //namespace must be explicitly stated for qRegisterMetaType
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
    std::unique_ptr<vrb::VRBClient> m_client = nullptr;
    std::unique_ptr<std::thread> m_thread;
    std::mutex m_mutex;

    vrb::SessionID m_sessionID;
    std::set<vrb::RemoteClient> m_clientList;
    void loop();
    bool handleVRB();
    bool removeOtherClient(covise::TokenBuffer &tb);
    void handleVrbLauncherMessage(covise::Message &msg);
    std::set<vrb::RemoteClient>::iterator findClient(int id);
};
QString getClientInfo(const vrb::RemoteClient &cl);
void spawnProgram(vrb::Program p, const std::vector<std::string> &args);


#endif