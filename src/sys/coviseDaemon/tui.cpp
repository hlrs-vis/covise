#include "tui.h"
#include "coviseDaemon.h"

#include <QApplication>

#include <iostream>

CommandLineUi::CommandLineUi(const vrb::VrbCredentials &credentials, bool autostart)
    : m_autostart(autostart), m_cinNotifier(0, QSocketNotifier::Type::Read) //on std in
{
    qRegisterMetaType<vrb::Program>();
    qRegisterMetaType<std::vector<std::string>>();

    std::cerr << "connecting to VRB on " << credentials.ipAddress << ", TCP-Port: " << credentials.tcpPort << ", UDP-Port: " << credentials.udpPort << std::endl;
    connect(&m_launcher, &CoviseDaemon::connectedSignal, this, []() {
        std::cerr << "connected!" << std::endl;
    });
    connect(&m_launcher, &CoviseDaemon::disconnectedSignal, this, [this]() {
        std::cerr << "disconnected!" << std::endl;
        m_launcher.connect();
    });
    connect(&m_launcher, &CoviseDaemon::launchSignal, this, [this](int senderId, QString senderDescription, vrb::Program id, std::vector<std::string> args) {
        std::lock_guard<std::mutex> g(m_mutex);
        m_program = id;
        m_senderId = senderId;
        m_args = args;
        if (m_autostart)
        {
            m_launcher.spawnProgram(m_program, m_args, std::function<void(const QString &)>{}, std::function<void(void)>{});
        }
        else
        {
            m_launchDialog = true;
            std::cerr << senderDescription.toStdString() << "requests start of " << vrb::programNames[id] << std::endl;
            std::cerr << "Do you want to execute that program?" << std::endl;
        }
    });
    m_launcher.connect(credentials);
    createCommands();

    connect(&m_cinNotifier, &QSocketNotifier::activated, this, [this]() {
        std::string command;
        std::cin >> command;
        handleCommand(command);
    });
}

void CommandLineUi::handleCommand(const std::string &command)
{
    for (auto &c : m_commands)
        c->execute(command);
    if (m_launchDialog && (command == "y" || command == "yes"))
    {
        std::lock_guard<std::mutex> g(m_mutex);
        m_launchDialog = false;
        m_launcher.sendPermission(m_senderId, true);
        m_launcher.spawnProgram(m_program, m_args, [](const QString &s) { std::cerr << s.toStdString() << std::endl; }, []() { std::cerr << "child process termintated!" << std::endl; });
    }
    else if (m_launchDialog && (command == "n" || command == "no"))
    {
        m_launchDialog = false;
        m_launcher.sendPermission(m_senderId, false);
    }
    else if (m_launchDialog)
    {
        std::cerr << "please enter yes or no" << std::endl;
    }
}

void CommandLineUi::createCommands()
{

    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"help"}, "show this message", [this]() {
                                                                           for (const auto &command : m_commands)
                                                                               command->print();
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"close", "quit", "terminate", "exit"}, "terminate this application", [this]() {
                                                                           QApplication::quit();
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"print", "client", "clients", "info", "partner"}, "print list of available clients", [this]() {
                                                                           m_launcher.printClientInfo();
                                                                       }}));

    for (int i = 0; i < static_cast<int>(vrb::Program::LAST_DUMMY); i++)
    {
        if (i != static_cast<int>(vrb::Program::coviseDaemon))
        {
            m_commands.push_back(std::unique_ptr<CommandInterface>(new LaunchCommand{static_cast<vrb::Program>(i), m_launcher}));
        }
    }
}
