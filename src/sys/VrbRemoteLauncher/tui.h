#ifndef VRB_REMOTE_LAUNCHER_TUI_H
#define VRB_REMOTE_LAUNCHER_TUI_H

#include "metaTypes.h"
#include "tuiCommands.h"
#include "vrbRemoteLauncher.h"

#include <vrb/client/VrbCredentials.h>

#include <QObject>
#include <QMainWindow>

#include <string>
#include <atomic>
#include <mutex>
#include <memory>

class Tui : public QMainWindow
{
    Q_OBJECT
public:
    Tui(const vrb::VrbCredentials &credentials, bool autostart);
    void run();
    
    private:
        VrbRemoteLauncher m_launcher;
        std::atomic_bool m_terminate{false};
        std::atomic_bool m_launchDialog{false};
        std::mutex m_mutex;
        std::vector<std::string> m_args;
        int m_senderId;
        vrb::Program m_program;
        bool m_autostart = false;
        std::vector<std::unique_ptr<CommandInterface>> m_commands;
        void handleCommand(const std::string &command);
        void createCommands();

};

#endif // !VRB_REMOTE_LAUNCHER_TUI_H