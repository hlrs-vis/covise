#include "SpawnProgram.h"
#ifdef _WIN32
#include <stdio.h>
#include <process.h>
#else
#include <unistd.h>
#endif
#include <signal.h>

using namespace vrb::launcher;
void vrb::launcher::spawnProgram(Program p, const std::vector<std::string> &args)
{
    spawnProgram(programNames[p], args);
}

void vrb::launcher::spawnProgram(const char *name, const std::vector<std::string> &args)
{

    std::vector<const char *> argV{args.size() + 2};
    argV[0] = name;
    for (size_t i = 0; i < args.size(); i++)
    {
        argV[i + 1] = args[i].c_str();
    }
    argV[args.size() + 1] = nullptr;

#ifdef _WIN32
    _spawnvp(P_NOWAIT, name, const_cast<char *const *>(argV.data()));
#else
    int pid = fork();
    if (pid == 0)
    {
        execvp(name, const_cast<char *const *>(argV.data()));
    }
    else
    {
        // Needed to prevent zombies
        // if childs terminate
        signal(SIGCHLD, SIG_IGN);
    }
#endif
}
