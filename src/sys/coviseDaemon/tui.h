#ifndef VRB_REMOTE_LAUNCHER_TUI_H
#define VRB_REMOTE_LAUNCHER_TUI_H

#include "metaTypes.h"
#include "tuiCommands.h"
#include "coviseDaemon.h"

#include <vrb/client/VrbCredentials.h>

#include <QObject>
#include <QMainWindow>
#include <QTimer>

#include <string>
#include <atomic>
#include <mutex>
#include <memory>
#include <queue>

class CommandLineUi : public QObject
{
    Q_OBJECT
public:
    CommandLineUi(const vrb::VrbCredentials &credentials, bool autostart);
    ~CommandLineUi();
   
    private:
        CoviseDaemon m_launcher;
        std::mutex m_mutex;
        bool m_autostart = false;
        std::vector<std::unique_ptr<CommandInterface>> m_commands;
        QTimer m_inputTimer;
        std::thread m_inputThread;
        std::atomic<bool> m_running{true};
        std::queue<std::string> m_commandQueue;
        std::mutex m_queueMutex;
        void handleCommand(const std::string &command);
        void createCommands();
    private slots:
        void processQueuedCommands();

};

#endif // !VRB_REMOTE_LAUNCHER_TUI_H