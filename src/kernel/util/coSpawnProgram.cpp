#include "coSpawnProgram.h"
#ifdef _WIN32
#include <stdio.h>
#include <process.h>
#else
#include <unistd.h>
#endif
#include <signal.h>

using namespace covise;

void covise::spawnProgram(const std::vector<const char *> &args)
{
    if (!args[0] || args[args.size() - 1])
    {
        return;
    }

#ifdef _WIN32
    _spawnvp(P_NOWAIT, args[0], const_cast<char *const *>(args.data()));
#else
    int pid = fork();
    if (pid == 0)
    {
        execvp(args[0], const_cast<char *const *>(args.data()));
    }
    else
    {
        // Needed to prevent zombies
        // if childs terminate
        signal(SIGCHLD, SIG_IGN);
    }
#endif
}

void covise::spawnProgram(const std::string &name, const std::vector<std::string> &args)
{

    std::vector<const char *> argV{args.size() + 2};
    argV[0] = name.c_str();
    for (size_t i = 0; i < args.size(); i++)
    {
        argV[i + 1] = args[i].c_str();
    }
    argV[args.size() + 1] = nullptr;
    spawnProgram(argV);
}

std::vector<const char*> covise::parseCmdArgString(const std::string &commandLine)
{
    std::vector<const char*> args;

    std::string delim = " ";

    size_t start = 0;
    auto end = commandLine.find(delim);
    while (end != std::string::npos)
    {
        args.emplace_back(commandLine.substr(start, end - start).c_str());
        start = end + delim.length();
        end = commandLine.find(delim, start);
    }
    return args;
}
