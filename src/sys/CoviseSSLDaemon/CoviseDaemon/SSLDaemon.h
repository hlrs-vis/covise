/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SSLDAEMON_H_
#define SSLDAEMON_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include "frmMainWindow.h"
#include "frmRequestDialog.h"
#include <net/covise_connect.h>
#include <QSocketNotifier>
#include <QObject>
#include <config/coConfig.h>

class frmMainWindow;
class frmRequestDialog;

using namespace ::covise;

class SSLDaemon : QObject
{
    Q_OBJECT
public:
    SSLDaemon(frmMainWindow *wnd);
    ~SSLDaemon(void);

    bool openServer();
    void closeServer();
    bool run();
    void stop();
    void setWindow(frmMainWindow *window);
    void allowPermanent(bool storeGlobal);
    void allow();
    void deny();

    static int sslPasswdCallback(char *buf, int size, int rwflag, void *userData);

protected:
    std::vector<std::string> mSubjectList;
    std::vector<std::string> mSubjectNameList;
    std::string mSubjectUID;
    bool isClientValid(SSLServerConnection *server);

protected slots:
    void processMessagesLegacy(); // Legacy method with higher systemload

private:
    frmRequestDialog *mRequest; // Request dialog for user interaction when unauthorized access
    std::streambuf *mSBuf; // storage for original cerr streaming buffer to reset at end of execution
    std::stringstream *mLogInternal; // Internal logging for debug messages
    std::ofstream *mFile; // file streaming buffer
    char *mDebugFile; // filename of logfile
    SSLServerConnection *mSSLConn; // SSLConnection object
    SSLServerConnection *mController; // Connection to COVISE controller
    SSLServerConnection *mAG; // Connection to AccessGrid
    int mPort; // Port of the CoviseDaemon to listen on
    ConnectionList *mConnections; // Spawned connections
    bool mIsRunning; // Flag indicating whether the server is running
    frmMainWindow *mWindow; // GUI Window
    std::map<SSLServerConnection *, QSocketNotifier *> mNotifier; // Network Notifier
    std::vector<std::string> mHostList; // List of allowed hosts
    bool mIsAllowed; //Flag for indication of allowed remote access
    bool mbConfirmed; //Flag indicating termination of Dialog
    bool mbCertCheck; //Indicates whether Client certificate checking
    //for remote access is enabled
    coConfigGroup *mConfig; //Config object to store personal settings
    std::string mCurrentPeer; //Current peer requesting connection
    float mNetworkpollIntervall; //Intervall between network polls
    std::string mPassword; // Password use for key-encryption

    // Conversion of int to std::string
    std::string ToString(int value);
    void updateLog(); // Method to update internal log-widget
    void handleMessage(Message &msg); // handles COVISE specific requests
    void parseCommands(SSLConnection *conn); // parse string-based commands
    void startCovise(); // Covise starting code through Controller

    // Method for implementing generic process spawn
    void spawnProcesses(std::string cmd, std::vector<std::string> opt);

    //Helper function to split a string into a vector of strings
    int SplitString(const std::string &input,
                    const std::string &delimiter,
                    std::vector<std::string> &results,
                    bool includeEmpties);
};
#endif
