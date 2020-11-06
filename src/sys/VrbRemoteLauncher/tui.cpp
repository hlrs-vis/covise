#include "tui.h"

#include <vrb/remoteLauncher/VrbRemoteLauncher.h>
#include <vrb/remoteLauncher/MessageTypes.h>

#include <QApplication>

#include <iostream>
using namespace vrb::launcher;

Tui::Tui(const vrb::VrbCredentials &credentials, bool autostart)
    : m_autostart(autostart)
{
    qRegisterMetaType<Program>();
    qRegisterMetaType<std::vector<std::string>>();

    std::cerr << "connecting to VRB on " << credentials.ipAddress << ", TCP-Port: " << credentials.tcpPort << ", UDP-Port: " << credentials.udpPort << std::endl;
    connect(&m_launcher, &VrbRemoteLauncher::connectedSignal, this, []() {
        std::cerr << "connected!" << std::endl;
    });
    connect(&m_launcher, &VrbRemoteLauncher::disconnectedSignal, this, [this]() {
        std::cerr << "disconnected!" << std::endl;
        m_launcher.connect();
    });
    connect(&m_launcher, &VrbRemoteLauncher::launchSignal, this, [this](Program id, std::vector<std::string> args) {
        std::lock_guard<std::mutex> g(m_mutex);
        m_program = id;
        m_args = args;
        if (m_autostart)
        {
            spawnProgram(m_program, m_args);
        }
        else
        {
            m_launchDialog = true;
            std::cerr << "do you want to start " << programNames[id] << "?" << std::endl;
        }
    });
    m_launcher.connect(credentials);
    createCommands();
}

void Tui::run()
{
    while (!m_terminate)
    {
        std::string command;
        std::cin >> command;
        handleCommand(command);
    }
    QApplication::quit();
}


void Tui::handleCommand(const std::string &command)
{
    for (auto &c : m_commands)
        c->execute(command);
    if (m_launchDialog && (command == "y" || command == "yes"))
    {
        std::lock_guard<std::mutex> g(m_mutex);
        spawnProgram(m_program, m_args);
        m_launchDialog = false;
    }
    else if (m_launchDialog && (command == "n" || command == "no"))
    {
        m_launchDialog = false;
    }
    else if (m_launchDialog)
    {
        std::cerr << "please enter yes or no" << std::endl;
    }
}

void Tui::createCommands()
{

    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"help"}, "show this message", [this]() {
                                                                           for (const auto &command : m_commands)
                                                                               command->print();
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"close", "quit", "terminate", "exit"}, "terminate this application", [this]() {
                                                                           m_terminate = true;
                                                                       }}));
    m_commands.push_back(std::unique_ptr<CommandInterface>(new Command{{"print", "client", "clients", "info", "partner"}, "print list of available clients", [this]() {
                                                                           m_launcher.printClientInfo();
                                                                       }}));

    for (int i = 0; i < static_cast<int>(Program::DUMMY); i++)
    {
        m_commands.push_back(std::unique_ptr<CommandInterface>(new LaunchCommand{static_cast<Program>(i), m_launcher}));
    }
}
