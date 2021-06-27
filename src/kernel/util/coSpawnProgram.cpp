#include "coSpawnProgram.h"
#include "coLog.h"
#include <string_util.h>
#ifdef _WIN32
#include <stdio.h>
#include <process.h>
#include <stdlib.h>
#include <Windows.h>
#else
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <sys/errno.h>
#endif
#include <signal.h>
#include <algorithm>
#include <array>
using namespace covise;

void covise::spawnProgram(const char *execPath, const std::vector<const char *> &args)
{
    if (!execPath || args[args.size() - 1])
    {
        print_error(__LINE__, __FILE__, "spawnProgram called with invalid args");
        return;
    }

#ifdef _WIN32
    //_spawnvp(P_NOWAIT, execPath, const_cast<char *const *>(args.data()));

    std::string win_cmd_line;
    for (const auto &arg : args)
    {
        if (arg != nullptr)
        {
            win_cmd_line.append(arg);
            win_cmd_line.append(" ");
        }
    }

    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Start the child process.
    if (!CreateProcess(NULL,                        // No module name (use command line)
                       (LPSTR)win_cmd_line.c_str(), // Command line
                       NULL,                        // Process handle not inheritable
                       NULL,                        // Thread handle not inheritable
                       FALSE,                       // Set handle inheritance to FALSE
                       0,                           // No creation flags
                       NULL,                        // Use parent's environment block
                       NULL,                        // Use parent's starting directory
                       &si,                         // Pointer to STARTUPINFO structure
                       &pi)                         // Pointer to PROCESS_INFORMATION structure
    )
    {
        printf("Could not launch %s !\nCreateProcess failed (%d).\n", win_cmd_line.c_str(), GetLastError());
    }
    else
    {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

#else
    int pid = fork();
    if (pid == 0)
    {
        execvp(execPath, const_cast<char *const *>(args.data()));
        print_error(__LINE__, __FILE__, " exec of \"%s\" failed %s", execPath, strerror(errno));
        exit(1);
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
    spawnProgram(name.c_str(), argV);
}

std::vector<std::string> createOsasciptArgs()
{
    std::vector<std::string> args;
    args.push_back("osascript");
    args.push_back("-e");
    args.push_back("tell application \"Terminal\"");
    args.push_back("-e");
    args.push_back("activate");
    args.push_back("-e");
    return args;
}

std::string createDebugEnvironmentCommandLineForApple()
{
    std::string covisedir;
    if (getenv("COVISEDIR"))
        covisedir = getenv("COVISEDIR");

    std::array<const char *, 3> env;
    env[0] = "COVISEDIR";
    env[1] = "ARCHSUFFIX";
    env[2] = "COCONFIG";

    std::string arg = "do script with command \"";
    for (const auto e : env)
    {
        const char *val = getenv(e);
        if (val)
            arg += std::string("export ") + e + "='" + val + "'; ";
    }
    arg += "source '" + covisedir + "/.covise.sh'; ";
    arg += "source '" + covisedir + "/scripts/covise-env.sh'; ";
    return arg;
}

struct Spawn
{
    std::string execPath;
    std::vector<std::string> args;
};

#ifndef WIN32

void trySpawn(const std::string &execName, const std::vector<Spawn> &spawns)
{
    int pid = fork();
    if (pid == 0)
    {
        for (const auto &spawn : spawns)
        {
            auto args = cmdArgsToCharVec(spawn.args);
            execvp(spawn.execPath.c_str(), const_cast<char *const *>(args.data()));
        }
        print_error(__LINE__, __FILE__, " exec of \"%s\" failed %s", execName.c_str(), strerror(errno));
        exit(1);
    }
    else
    {
        // Needed to prevent zombies
        // if childs terminate
        signal(SIGCHLD, SIG_IGN);
    }
}
#endif // !WIN32

#ifdef __APPLE__

#endif
std::vector<std::string> appleArgs(const std::string &debugPath, const std::string &execPath, const std::vector<std::string> &args, bool memcheck)
{
    auto command = createOsasciptArgs();
    std::string arg = createDebugEnvironmentCommandLineForApple();

    //arg += "gdb --args ";
    arg += debugPath;
    arg += execPath;
    if (!memcheck)
        arg += " -- ";

    for (const auto a : args)
        arg += " " + a;
    if (!memcheck)
        arg += "; exit";
    arg += "\"";
    command.push_back(arg);
    command.push_back("-e");
    command.push_back("end tell");
    return command;
}

void covise::spawnProgramWithDebugger(const std::string &execPath, const std::string &debugCommands, const std::vector<std::string> &args)
{
#ifdef WIN32
    std::string win_cmd_line;
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    for (const auto arg : args)
    {
        win_cmd_line.append(arg);
        win_cmd_line.append(" ");
    }

    // launch & debug
    // launch the child process "suspended" then attach debugger and resume child's main thread
    if (!CreateProcess(execPath.c_str(),                    // No module name (use command line)
                       (LPSTR)win_cmd_line.c_str(), // Command line
                       NULL,                        // Process handle not inheritable
                       NULL,                        // Thread handle not inheritable
                       FALSE,                       // Set handle inheritance to FALSE
                       CREATE_SUSPENDED,            // create in suspended state
                       NULL,                        // Use parent's environment block
                       NULL,                        // Use parent's starting directory
                       &si,                         // Pointer to STARTUPINFO structure
                       &pi)                         // Pointer to PROCESS_INFORMATION structure
    )
    {
        fprintf(stderr, "Could not launch %s !\nCreateProcess failed (%d).\n", win_cmd_line.c_str(), (int)GetLastError());
    }
    else
    {
        // success now launch/attach debugger
        STARTUPINFO dbg_si;
        PROCESS_INFORMATION dbg_pi;
        ZeroMemory(&dbg_si, sizeof(dbg_si));
        dbg_si.cb = sizeof(dbg_si);
        ZeroMemory(&dbg_pi, sizeof(dbg_pi));

#ifdef __MINGW32__
        std::string debug_cmd_line("qtcreator -debug ");
#else
        std::string debug_cmd_line("vsjitdebugger -p ");
#endif
        debug_cmd_line += std::to_string(pi.dwProcessId);

        if (!CreateProcess(NULL, (LPSTR)debug_cmd_line.c_str(), NULL, NULL, FALSE, 0, NULL, NULL, &dbg_si, &dbg_pi))
        {
            fprintf(stderr, "Could not launch debugger %s !\nCreateProcess failed (%d).\n", debug_cmd_line.c_str(), (int)GetLastError());
        }
        else
        {
            fprintf(stderr, "Launched debugger with %s\n", debug_cmd_line.c_str());
            DWORD wait_ret = WaitForInputIdle(dbg_pi.hProcess, INFINITE);
            if (wait_ret == 0)
            {
                fprintf(stderr, "Wait for debugger successful!\n");
            }
            else if (wait_ret == WAIT_TIMEOUT)
            {
                fprintf(stderr, "Wait for debugger timed out!\n");
            }
            else
            {
                fprintf(stderr, "Wait for debugger failed! GetLastError() -> %d\n", (int)GetLastError());
            }
            //Sleep(10000);
            CloseHandle(dbg_pi.hProcess);
            CloseHandle(dbg_pi.hThread);
        }
        // resume process' main thread
        if (ResumeThread(pi.hThread) == -1)
        {
            fprintf(stderr, "ResumeThread() failed on %s !\nGetLastError() -> %d\n", win_cmd_line.c_str(), (int)GetLastError());
        }
        // Wait until child process exits.
        //WaitForSingleObject( pi.hProcess, INFINITE );
        // Close process and thread handles.
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
#else // !WIN32
    std::vector<Spawn> spawns;
#ifdef __APPLE__
    {
        Spawn &s = *spawns.emplace(spawns.end());
        s.execPath = "/Applications/Xcode.app/Contents/Developer/usr/bin/lldb ";
        s.args = appleArgs(s.execPath, execPath, args, false);
        std::string arg = createDebugEnvironmentCommandLineForApple();
    }
#endif
    bool defaultCommand = debugCommands.empty();
    Spawn &s = *spawns.emplace(spawns.end());
    s.execPath = "Konsole";
    s.args = defaultCommand ? std::vector<std::string>{"Konsole", "-e", "gdb", "--args"} : parseCmdArgString(debugCommands);
    s.args.emplace_back(execPath);
    s.args.insert(s.args.end(), args.begin(), args.end());

    if (defaultCommand)
    {
        auto &s2 = *spawns.emplace(spawns.end(), s);
        s2.execPath = s2.args[0] = "xterm";
    }
    trySpawn(execPath, spawns);
#endif // !WIN32
}

void covise::spawnProgramWithMemCheck(const std::string &execPath, const std::string &debugCommands, const std::vector<std::string> &args)
{

#ifdef WIN32
    spawnProgramWithDebugger(execPath, debugCommands, args);
#else //!WIN32
    std::vector<Spawn> spawns;

#ifdef __APPLE__
    {
        Spawn &s = *spawns.emplace(spawns.end());
        s.execPath = "valgrind --trace-children=no --dsymutil=yes ";
        s.args = appleArgs(s.execPath, execPath, args, false);
        std::string arg = createDebugEnvironmentCommandLineForApple();
    }
#endif
    bool defaultCommand = debugCommands.empty();
    Spawn &s = *spawns.emplace(spawns.end());

    s.args = defaultCommand ?
#ifdef __APPLE__
    std::vector<std::string>{"konsole", "--noclose", "-e", "valgrind", "--trace-children=no", "--dsymutil=yes"}
#else
    std::vector<std::string>{"konsole", "-hold", "-e", "valgrind", "--trace-children=no"}
#endif
    : parseCmdArgString(debugCommands);
    s.args.emplace_back(execPath);
    s.args.insert(s.args.end(), args.begin(), args.end());
    s.execPath = s.args[0];

    if (defaultCommand)
    {
        auto &s2 = *spawns.emplace(spawns.end(), s);
        s2.execPath = s2.args[0] = "xterm";
    }
    trySpawn(execPath, spawns);

#endif //!WIN32
}

std::vector<std::string> covise::parseCmdArgString(const std::string &commandLine)
{
    return split(commandLine, ' ');
}

std::vector<const char *> covise::cmdArgsToCharVec(const std::vector<std::string> &args)
{
    std::vector<const char *> v(args.size() + 1);
    std::transform(args.begin(), args.end(), v.begin(), [](const std::string &s)
                   { return s.c_str(); });
    v[args.size()] = nullptr;
    return v;
}