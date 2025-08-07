/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MainWindow_H
#define MainWindow_H

#include "childOutput.h"
#include "coviseDaemon.h"
#include "permissionRequest.h"

#include <QMainWindow>
#include <QTimer>

#include <memory>

#include "config/access.h"
#include "config/value.h"
#include "config/file.h"

namespace Ui
{
    class MainWindow;
}
namespace covise{
    class NonBlockingDialogue;
}
class ClientWidgetList;
class QStackedWidget;
class QSystemTrayIcon;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const vrb::VrbCredentials &credentials, QWidget *parent = nullptr);
    ~MainWindow();
private slots:
    void on_actionSideMenuAction_triggered();
    void on_timeoutSlider_sliderMoved(int val);
    void on_autostartCheckBox_clicked();
    void on_autoconnectCheckBox_clicked();
    void on_backgroundCheckBox_clicked();
    void on_minimizedCheckBox_clicked();
    void on_cmdArgsInput_textChanged();
    
    void onConnectBtnClicked();
    void onCancelBtnClicked();
    void onDisconnectBtnClicked();
    void updateStatusBar();
    void setStateDisconnected();
    void setStateConnecting();
    void setStateConnected();
    void updateClient(int clientID, QString clientInfo);
    void removeClient(int clientID);
    void closeEvent(QCloseEvent* event) override;
    void removePermissionRequest(covise::Program p, int clientID);

    void openDemoClient();
    void showDemoClientContextMenu();
    void copyDemoLinkToClipboard();

signals:
    void updateStatusBarSignal();

private:
    typedef std::lock_guard<std::mutex> Guard;
    Ui::MainWindow *ui = nullptr;
    QSystemTrayIcon* m_tray = nullptr;
    std::mutex m_mutex;
    std::atomic_bool m_isConnecting{false};
    QTimer m_progressBarTimer;
    ClientWidgetList *m_clientList;
    CoviseDaemon m_remoteLauncher;
    covise::config::Access m_access;
    std::unique_ptr<covise::config::File> m_configFile;


    // covise::coConfigGroup *cdConfig;
    std::unique_ptr<covise::ConfigInt> cfgTimeout;
    std::unique_ptr<covise::ConfigBool> cfgAutostart;
    std::unique_ptr<covise::ConfigBool> cfgAutoConnect;
    std::unique_ptr<covise::ConfigBool> cfgBackground;
    std::unique_ptr<covise::ConfigBool> cfgMinimized;
    std::unique_ptr<covise::ConfigString> cfgArguments;
    std::unique_ptr<covise::ConfigString> cfgOutputMode;
    std::unique_ptr<covise::ConfigString> cfgOutputModeFile;
    std::vector<ChildOutput> m_childOutputs;
    std::vector<std::pair<QString, std::unique_ptr<std::fstream>>> m_childOutputFiles;
    std::vector<std::unique_ptr<PermissionRequest>> m_permissionRequests;

    void initConfigSettings();
    void initOutputModes();

    void initUi(const vrb::VrbCredentials &credentials);
    void setRemoteLauncherCallbacks();
    void reconnectOutPut();

    void readOptions();
    void initClientList();
    void setHotkeys();
    void handleAutoconnect();
    void setStartupWindowStyle();

    void showThis();
    void hideThis();
    void createTrayIcon();

    void showConnectionProgressBar(int seconds);
    void askForPermission(covise::Program p, int clientID, const QString &description);
    void saveOptions();

    QString getDemoClientUrl();
    void setupDemoClientAction();

    std::vector<std::string> parseCmdArgsInput();
};
void setStackedWidget(QStackedWidget *stack, int index);

#endif // MainWindow_H
