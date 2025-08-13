#include "tui.h"
#include "coviseDaemon.h"

#include <QApplication>

#include <iostream>

CommandLineUi::CommandLineUi(const vrb::VrbCredentials &credentials, bool autostart)
    : m_autostart(autostart)
{
    qRegisterMetaType<covise::Program>();
    qRegisterMetaType<std::vector<std::string>>();

    std::cerr << "connecting to VRB on " << credentials.ipAddress() << ", TCP-Port: " << credentials.tcpPort() << ", UDP-Port: " << credentials.udpPort() << std::endl;
    connect(&m_launcher, &CoviseDaemon::connectedSignal, this, []()
            { std::cerr << "connected!" << std::endl; });
    connect(&m_launcher, &CoviseDaemon::disconnectedSignal, this, [this]()
            {
                std::cerr << "disconnected!" << std::endl;
                m_launcher.connect();
            });
    QObject::connect(&m_launcher, &CoviseDaemon::askForPermission, this, [this](covise::Program p, int clientID, const QString &description)
            {
                std::lock_guard<std::mutex> g(m_mutex);
                if (m_autostart)
                {
                    m_launcher.answerPermissionRequest(p, clientID, true);
                }
                else
                {
                    std::cerr << description.toStdString() << std::endl;
                    std::cerr << "Do you want to execute that program?" << std::endl;
                    while (true)
                    {
                        std::string arg;
                        std::cin >> arg;
                        if (arg == "y" || arg == "yes")
                        {
                            m_launcher.answerPermissionRequest(p, clientID, true);
                            return;
                        }
                        else if (arg == "n" || arg == "no")
                        {
                            m_launcher.answerPermissionRequest(p, clientID, false);
                            return;
                        }
                        else
                            std::cerr << "Please enter yes or no:" << std::endl;
                    }
                }
            });
    connect(&m_launcher, &CoviseDaemon::childProgramOutput, this, [](const QString &child, const QString &msg)
            { std::cerr << msg.toStdString() << std::endl; });

    connect(&m_launcher, &CoviseDaemon::childTerminated, this, [](const QString &child)
            { std::cerr << "child " << child.toStdString() << " terminated!" << std::endl; });

    m_launcher.connect(credentials);
    createCommands();

    m_inputThread = std::thread([this]() {
        std::string command;
        while (m_running) {
            std::cout << "> " << std::flush;
            if (std::getline(std::cin, command)) {
                if (!command.empty()) {
                    // Thread-safe queue the command
                    {
                        std::lock_guard<std::mutex> lock(m_queueMutex);
                        m_commandQueue.push(command);
                    }
                    
                    // Process commands in main thread using Qt's signal system
                    QMetaObject::invokeMethod(this, &CommandLineUi::processQueuedCommands, Qt::QueuedConnection);
                }
            }
            
            if (std::cin.eof()) {
                // Handle Ctrl+D or EOF
                m_running = false;
                QMetaObject::invokeMethod(QCoreApplication::instance(), &QCoreApplication::quit, Qt::QueuedConnection);
                break;
            }
        }
    });
    
    std::cerr << "Type 'help' for available commands." << std::endl;
}

void CommandLineUi::processQueuedCommands()
{
    std::lock_guard<std::mutex> lock(m_queueMutex);
    while (!m_commandQueue.empty()) {
        std::string command = m_commandQueue.front();
        m_commandQueue.pop();
        handleCommand(command);
    }
}

// Add destructor to clean up thread
CommandLineUi::~CommandLineUi()
{
    m_running = false;
    if (m_inputThread.joinable()) {
        m_inputThread.join();
    }
}

void CommandLineUi::handleCommand(const std::string &command)
{
    for (auto &c : m_commands)
        c->execute(command);
}

void CommandLineUi::createCommands()
{
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"help"}, "show this message", [this]()
                                                                       {
                                                                           for (const auto &command : m_commands)
                                                                               command->print();
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"close", "quit", "terminate", "exit"}, "terminate this application", [this]()
                                                                       {
                                                                           QCoreApplication::quit();
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"print", "client", "clients", "info", "partner"}, "print list of available clients", [this]()
                                                                       {
                                                                           m_launcher.printClientInfo();
                                                                       }}));

    for (int i = 0; i < static_cast<int>(covise::Program::LAST_DUMMY); i++)
    {
        if (i != static_cast<int>(covise::Program::coviseDaemon))
        {
            m_commands.push_back(std::unique_ptr<CommandInterface>(new LaunchCommand{static_cast<covise::Program>(i), m_launcher}));
        }
    }
}
