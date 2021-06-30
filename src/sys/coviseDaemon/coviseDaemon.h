/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_REMOTE_LAUNCHER_MESSAGE_HANDLER_H
#define VRB_REMOTE_LAUNCHER_MESSAGE_HANDLER_H

#include "childProcess.h"
#include "metaTypes.h"

#include <vrb/ProgramType.h>
#include <vrb/client/VrbCredentials.h>
#include <vrb/client/VRBClient.h>
#include <vrb/client/LaunchRequest.h>

#include <QObject>
#include <QSocketNotifier>
#include <QString>

#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

//The CoviseDaemon that listens to launch request submitted via VRB
class CoviseDaemon : public QObject
{
    Q_OBJECT
public:
    ~CoviseDaemon();
    void connect(const vrb::VrbCredentials &credentials = vrb::VrbCredentials{});
    void disconnect();
    void printClientInfo();
public slots:
    void sendLaunchRequest(vrb::Program p, int clientID, const std::vector<std::string> &args = std::vector<std::string>{});
    void answerPermissionRequest(vrb::Program p, int clientID, bool answer); 
signals:
    void connectedSignal();
    void disconnectedSignal();
    void updateClient(int clientID, QString clientInfo);
    void removeClient(int id);
    void childProgramOutput(const QString &child, const QString &output);
    void childTerminated(const QString &child);
    void receivedVrbMsg(const covise::Message &msg);
    void askForPermission(vrb::Program p, int clientID, const QString &description); //must call answerPermissionRequest

private slots:
    bool handleVRB(const covise::Message& msg);

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
    std::set<ChildProcess> m_children;
    std::vector<std::unique_ptr<vrb::VRB_MESSAGE>> m_sentLaunchRequests;
    std::unique_ptr<vrb::VRB_MESSAGE> m_receivedLaunchRequest;
    void loop();
    void spawnProgram(vrb::Program p, const std::vector<std::string> &args);
    bool removeOtherClient(covise::TokenBuffer &tb);
    void handleVrbLauncherMessage(const covise::Message &msg);
    void ask(vrb::Program p, int clientID);
    std::set<vrb::RemoteClient>::iterator findClient(int id);

    struct ProgramToLaunch
    {

        ProgramToLaunch(vrb::Program p, int requestorId);
        ProgramToLaunch(vrb::Program p, int requestorId, int code);

        bool operator==(const ProgramToLaunch &other) const;
        int code() const;

    private:
        vrb::Program m_p;
        int m_requestorId;
        int m_code;
    };

    std::vector<ProgramToLaunch> m_allowedProgramsToLaunch;
};
QString getClientInfo(const vrb::RemoteClient &cl);

#endif