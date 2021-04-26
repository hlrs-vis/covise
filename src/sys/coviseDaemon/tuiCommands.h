#ifndef VRB_REMOTE_LAUNCHER_TUI_COMMMANDS_H
#define VRB_REMOTE_LAUNCHER_TUI_COMMMANDS_H

#include "coviseDaemon.h"

#include <vector>
#include <string>
#include <functional>


struct CommandInterface
{
    virtual void print() const = 0;
    virtual void execute(const std::string &command) = 0;
    virtual ~CommandInterface() = default;
};

struct Command : CommandInterface
{
    Command(const std::vector<std::string> &conditions, const std::string &toolTip, std::function<void(void)> function);

    void print() const override;

    void execute(const std::string &command) override;


private:
    const std::vector<std::string> m_conditions;
    const std::string m_toolTip;
    const std::function<void(void)> m_function;
};


struct LaunchCommand : CommandInterface{

    LaunchCommand(vrb::Program program, CoviseDaemon &launcher);
    void print() const override;

    void execute(const std::string &command) override;
    private:
        vrb::Program m_program;
        CoviseDaemon &m_launcher;
};

#endif // !VRB_REMOTE_LAUNCHER_TUI_COMMMANDS_H