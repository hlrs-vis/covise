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
    connect(&m_launcher, &CoviseDaemon::connectedSignal, this, []()
            { std::cerr << "connected!" << std::endl; });
    connect(&m_launcher, &CoviseDaemon::disconnectedSignal, this, [this]()
            {
                std::cerr << "disconnected!" << std::endl;
                m_launcher.connect();
            });
    m_launcher.setLaunchRequestCallback([this](QString launchDescription)
                                        {
                                            std::lock_guard<std::mutex> g(m_mutex);
                                            if (m_autostart)
                                            {
                                                return true;
                                            }
                                            else
                                            {
                                                std::cerr << launchDescription.toStdString() << std::endl;
                                                std::cerr << "Do you want to execute that program?" << std::endl;
                                                while (true)
                                                {
                                                    std::string arg;
                                                    std::cin >> arg;
                                                    if (arg == "y" || arg == "yes")
                                                        return true;
                                                    else if (arg == "n" || arg == "no")
                                                        return false;
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

    connect(&m_cinNotifier, &QSocketNotifier::activated, this, [this]()
            {
        std::string command;
        std::cin >> command;
        handleCommand(command);
            });
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
                                                                           QApplication::quit();
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"print", "client", "clients", "info", "partner"}, "print list of available clients", [this]()
                                                                       {
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
