#include "tuiCommands.h"
#include <iostream>

using namespace vrb::launcher;
Command::Command(const std::vector<std::string> &conditions, const std::string &toolTip, std::function<void(void)> function)
    : m_conditions(conditions), m_toolTip(toolTip), m_function(function) {}

void Command::print() const
{
    std::cerr << "Command: ";
    for (size_t i = 0; i < m_conditions.size() - 1; i++)
    {
        std::cerr << m_conditions[i] << ", ";
    }
    std::cerr << m_conditions[m_conditions.size() - 1] << " : ";
    std::cerr << m_toolTip << std::endl;
}

void Command::execute(const std::string &command)
{
    for (const auto condition : m_conditions)
    {
        if (condition == command)
        {
            m_function();
        }
    }
}

LaunchCommand::LaunchCommand(Program program, VrbRemoteLauncher &launcher)
    : m_program(program), m_launcher(launcher) {}

void LaunchCommand::print() const
{
    std::cerr << "Command: " << programNames[m_program] << " + client ID + args: launch " << programNames[m_program] << " on the specified client" << std::endl;
}

void LaunchCommand::execute(const std::string &command)
{
    if (command.substr(0, strlen(programNames[m_program])) == programNames[m_program])
    {
        size_t l;
        int clID = 0;
        std::vector<std::string> args;
        try
        {
            std::cin >> clID;
            while (std::cin.peek() != '\n')
            {
                auto arg = args.emplace(args.end(), std::string{});
                std::cin >> *arg;
            }
            for(const auto & arg : args)
                std::cerr << arg << " ";
            std::cerr << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Command requiers int after " << programNames[m_program] << ", " << command.substr(strlen(programNames[m_program]) + 1) << "found:" << e.what() << '\n';
            return;
        }
        m_launcher.sendLaunchRequest(m_program, clID, args);
    }
}
