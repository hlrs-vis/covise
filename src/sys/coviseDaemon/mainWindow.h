/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MainWindow_H
#define MainWindow_H

#include "coviseDaemon.h"

#include <QMainWindow>

#include <memory>
#include <future>
namespace Ui
{
    class MainWindow;
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
    void onConnectBtnClicked();
    void onCancelBtnClicked();
    void onDisconnectBtnClicked();
    void updateStatusBar();
    void setStateDisconnected();
    void setStateConnecting();
    void setStateConnected();
    void updateClient(int clientID, QString clientInfo);
    void removeClient(int clientID);
    void launchProgram(int senderID, const QString& senderDescription, vrb::Program programID, const std::vector<std::string> &args);
    void closeEvent(QCloseEvent* event) override;

signals:
    void updateStatusBarSignal();

private:
    typedef std::lock_guard<std::mutex> Guard;
    Ui::MainWindow *ui = nullptr;
    QSystemTrayIcon* m_tray = nullptr;
    std::mutex m_mutex;
    std::atomic_bool m_isConnecting{false};
    std::future<void> m_waitFuture;
    ClientWidgetList *m_clientList;
    CoviseDaemon m_remoteLauncher;

    void initUi(const vrb::VrbCredentials &credentials);
    void setRemoteLauncherCallbacks();
    void readOptions();
    void initClientList();
    void setHotkeys();
    void handleAutoconnect();
    void setStartupWindowStyle();

    void showThis();
    void hideThis();
    void createTrayIcon();

    void showConnectionProgressBar(int seconds);
    bool askForPermission(const QString &senderDescription, vrb::Program programID);
    
    void saveOptions();

    std::vector<std::string> parseCmdArgsInput();
};
void setStackedWidget(QStackedWidget *stack, int index);

#endif // MainWindow_H
