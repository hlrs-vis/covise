/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_HANDLER_H
#define CTRL_HANDLER_H

#include <string>

#include <QStringList>
#include <QMap>

#include <vrb/client/VRBClient.h>

#include "hostManager.h"
#include "userinterface.h"
#include "config.h"


namespace covise
{

class Message;
class ControlConfig;
struct NEW_UI;

// == == == == == == == == == == == == == == == == == == == == == == == ==
namespace controller{

struct NumRunning
{
    int apps = 0;
    int renderer = 0;
};
class CTRLHandler
// == == == == == == == == == == == == == == == == == == == == == == == ==
{
public:
    enum readMode
    {
        NETWORKMAP,
        CLIPBOARD,
        UNDO
    };

    CTRLHandler(int argc, char **argv);
    ~CTRLHandler();
    static CTRLHandler *instance();

    NumRunning &numRunning();
    const UIOptions &uiOptions();
    string handleBrowserPath(const string &name, const string &nr, const string &host, const string &oldhost,
                             const string &parameterName, const string &parameterValue);
    void handleMsg(const std::unique_ptr<Message>& msg);
    bool slaveUpdate = false;
    bool m_miniGUI = false, m_rgbTextOpen = false;

    ControlConfig Config;

    bool recreate(const string &content, readMode mode);
    const std::string &globalFile() const;
    void finishExecuteIfLastRunning(const NetModule &app);
    void sendCollaborativeState();

private:
    bool m_exit = false; //flag to exit main loop and terminate the controller
    static CTRLHandler *singleton;
    NumRunning m_numRunning;
    FILE *fp;
    string m_globalFilename, m_clipboardBuffer, m_collaborationRoom;
struct CommandLineOptions{
    std::string netFile;
    std::string crbLaunchScript;
    std::string localUserName;
    bool quit = false; //m_quitAfterExececute
    bool executeOnLoad = false; 
    bool isLoaded = true;
    UIOptions uiOptions;

} m_options;
bool m_globalLoadReady = true, m_clipboardReady = true;
HostManager m_hostManager;

void addHostIfnotAlreadyAdded(const std::string &ipAddress, const std::string &userName, LaunchStyle style);
void lookupSiblings();
void loop();
void parseCommandLine(int argc, char **argv);
void startCrbUiAndDatamanager();
void loadNetworkFile();
void handleQuit(const std::unique_ptr<Message> &msg);
void handleUI(Message *msg, string data);
void handleNewUi(const NEW_UI &msg);
void handleFinall(const std::unique_ptr<Message>& msg, string data);
void delModuleNode(const vector<NetModule *> &liste);
const NetModule * initModuleNode(const string &name, const string &nr, const string &host, int, int, const string &, int, ExecFlag flags);
NetModule *findApplication(const std::string &ipAddress, const std::string &name, int instance);
void makeConnection(const string &from_mod, const string &from_nr, const string &from_host, const string &from_port,
                    const string &to_mod, const string &to_nr, const string &to_host, const string &to_port);

obj_conn * connectPorts(const string &from_name, const string &from_nr, const string &from_host,
                               const string &from_port,
                               const string &to_name, const string &to_nr, const string &to_host,
                               const string &to_port);

void sendNewParam(const string &name, const string &nr, const string &hostAddress,
                  const string &parname, const string &partype, const string &parvalue, const string &apptype,
                  const string &oldhost, bool init = false);
bool checkIfModuleAvailable(const string &modname, const string &modhost);
bool waitForVrbRegistration();
bool vrbUpdate();
void vrbClientsUpdate(const Message &userInfoMessage);
void getAllConnections();
std::string createApplicationsAndConnectionsData();
void resetLists(); //delete all application modules
string writeClipboard(const string &keyword, const vector<NetModule *> &liste, bool all = false);
void addBuffer(const QString &text);
void addHost(const vrb::RemoteClient &rc);
void addPartner(const vrb::RemoteClient &rc);
void removeClient(const vrb::RemoteClient &rc);
void sendCurrentNetToUI(const MessageSenderInterface &ui);

void sendGenericInfoToRenderer(const std::string& prefix, const Message &msg);
void broadcastMsgToOtherUis(const Message &msg);
void saveCurrentNetworkFile(const std::string &filename);
bool loadNetworkFile(const std::string &filename);

std::string m_autosavefile;

bool m_writeUndoBuffer = true;
QStringList m_qbuffer;
QStringList m_undoBuffer;

int m_nconn;
vector<string> from_name, from_inst, from_host, from_port;
vector<string> to_name, to_inst, to_host, to_port;
typedef std::list<std::pair<std::string, std::string>> slist;

slist siblings;
};
} //namespace controller
} //namespace covise
#endif
