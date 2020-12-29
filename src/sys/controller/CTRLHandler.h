/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_CTRL_H
#define CTRL_CTRL_H

#include <QMap>
#include <QStringList>
#include "CTRLGlobal.h"
#include <vrb/client/VRBClient.h>
#include <string>
class QString;

namespace covise
{

class net_module;
class Message;
class ControlConfig;
class AccessGridDaemon;
class AppModule;
class SSLClient;

// == == == == == == == == == == == == == == == == == == == == == == == ==
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

    bool m_miniGUI, m_rgbTextOpen;
    int m_numRunning, m_numRendererRunning;

    ControlConfig *Config;
    AccessGridDaemon *m_accessGridDaemon;

    static CTRLHandler *instance();
    void handleClosedMsg(Message *msg);
    void handleAndDeleteMsg(Message *msg);
    string handleBrowserPath(const string &name, const string &nr, const string &host, const string &oldhost,
                             const string &parameterName, const string &parameterValue);

    vector<string> splitString(string text, const string &sep);
    bool recreate(string buffer, readMode mode);
    const std::string &vrbSessionName() const;
private:
    static CTRLHandler *singleton;
    FILE *fp;
    string m_globalFilename, m_netfile, m_clipboardBuffer, m_collaborationRoom;
    string m_localUser, m_scriptName, m_filename, m_filePartnerHost, m_pyFile;

    bool m_globalLoadReady, m_clipboardReady;
    bool m_readConfig; /* all Mapeditors started */
    bool m_useGUI; /* use an user interface */
    bool m_isLoaded, m_executeOnLoad, m_iconify, m_maximize;
    int m_quitAfterExececute;
    int m_daemonPort, m_xuif, m_startScript, m_accessGridDaemonPort;
    int m_SSLDaemonPort;
    SSLClient *m_SSLClient;
    vrb::VRBClient m_client;
    mutable std::string m_sessionName;
    int parseCommandLine(int argc, char **argv);
    void startCrbUiDm();
    void loadNetworkFile();
    void handleAccessGridDaemon(Message *msg);
    void handleSSLDaemon(Message *msg);
    void handleQuit(Message *msg);
    void handleUI(Message *msg, string data);
    void handleFinall(Message *msg, string data);
    void delModuleNode(vector<net_module *> liste);
    int initModuleNode(const string &name, const string &nr, const string &host, int, int, const string &, int, ExecFlag flags);
    void makeConnection(const string &from_mod, const string &from_nr, const string &from_host, const string &from_port,
                        const string &to_mod, const string &to_nr, const string &to_host, const string &to_port);
    void sendNewParam(const string &name, const string &nr, const string &host,
                      const string &parname, const string &partype, const string &parvalue, const string &apptype,
                      const string &oldhost, bool init = false);
    bool checkModule(const string &modname, const string &modhost);
    void getAllConnections();
    void resetLists();
    string writeClipboard(const string &keyword, vector<net_module *> liste, bool all = false);
    void addBuffer(const QString &text);
    void sendCollaborativeState();

    std::string m_autosavefile;

    bool m_writeUndoBuffer;
    QStringList m_qbuffer;
    QStringList m_undoBuffer;

    int m_nconn;
    vector<string> from_name, from_inst, from_host, from_port;
    vector<string> to_name, to_inst, to_host, to_port;
    typedef std::list<std::pair<std::string, std::string> > slist;

    slist siblings;
};
}
#endif
