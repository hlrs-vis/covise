#include "coSpawnProgram.h"
#include "coLog.h"

#ifdef _WIN32
#include <stdio.h>
#include <process.h>
#include <stdlib.h>
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <signal.h>

#include <array>

using namespace covise;

void covise::spawnProgram(const char* execPath, const std::vector<const char *> &args)
{
    if (!execPath || args[args.size() - 1])
    {
        print_error(__LINE__, __FILE__, "spawnProgram called with invalid args");
        return;
    }

#ifdef _WIN32
    _spawnvp(P_NOWAIT, execPath, const_cast<char *const *>(args.data()));
#else
    int pid = fork();
    if (pid == 0)
    {
        if(execvp(execPath, const_cast<char *const *>(args.data())) == -1){

            print_error(__LINE__, __FILE__, "%s%s%s", "exec of ", execPath, " failed");
            exit(1);
        }
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

std::vector<const char*> createOsasciptArgs() {
    std::vector<const char*> args;
    args.push_back("osascript");
    args.push_back("-e");
    args.push_back("tell application \"Terminal\"");
    args.push_back("-e");
    args.push_back("activate");
    args.push_back("-e");
    return args;
}

std::string createDebugEnvironmentCommandLineForApple() {
    std::string covisedir;
    if (getenv("COVISEDIR"))
        covisedir = getenv("COVISEDIR");


    std::array<const char*, 3> env;
    env[0] = "COVISEDIR";
    env[1] = "ARCHSUFFIX";
    env[2] = "COCONFIG";

    std::string arg = "do script with command \"";
    for (const auto e : env)
    {
        const char* val = getenv(e);
        if (val)
            arg += std::string("export ") + e + "='" + val + "'; ";
    }
    arg += "export CO_MODULE_BACKEND=covise; ";
    arg += "source '" + covisedir + "/.covise.sh'; ";
    arg += "source '" + covisedir + "/scripts/covise-env.sh'; ";
    return arg;
}


void covise::spawnProgramWithDebugger(const char* execPath, const std::string &debugCommands, const std::vector<const char*>& args) {
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
    if (!CreateProcess(execPath, // No module name (use command line)
        (LPSTR)win_cmd_line.c_str(), // Command line
        NULL, // Process handle not inheritable
        NULL, // Thread handle not inheritable
        FALSE, // Set handle inheritance to FALSE
        CREATE_SUSPENDED, // create in suspended state
        NULL, // Use parent's environment block
        NULL, // Use parent's starting directory
        &si, // Pointer to STARTUPINFO structure
        &pi) // Pointer to PROCESS_INFORMATION structure
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
    std::vector<const char*> debugArgs;
#ifdef __APPLE__
    debugArgs = createOsasciptArgs();
    std::string arg = createDebugEnvironmentCommandLineForApple();

    //arg += "gdb --args ";
    arg += "/Applications/Xcode.app/Contents/Developer/usr/bin/lldb ";
    arg += execPath;
    arg += " -- ";
    for (int i = 1; args[i]; ++i)
    {
        arg += " ";
        arg += args[i];
    }
    arg += "; exit";
    arg += "\"";
    debugArgs.push_back(arg.c_str());
    debugArgs.push_back("-e");
    debugArgs.push_back("end tell");
    debugArgs.push_back(nullptr);
    spawnProgram(execPath, debugArgs);
#endif
    bool defaultCommand = false;
    std::string command;
    command = debugCommands;
    if (command.empty())
    {
        command = "konsole -e gdb --args";
        defaultCommand = true;
    }
    debugArgs = parseCmdArgString(command);
    debugArgs.push_back(execPath);
    debugArgs.insert(debugArgs.end(), args.begin(), args.end());
    spawnProgram(execPath, debugArgs);

    if (defaultCommand)
    {
        // try again with xterm
        debugArgs[0] = "xterm";
        spawnProgram(execPath, debugArgs);
    }

#endif // WIN32
}


void covise::spawnProgramWithMemCheck(const char* execPath, const std::string& debugCommands, const std::vector<const char*>& args) {
   
#ifdef WIN32
    spawnProgramWithDebugger(execPath, "", args);
#else //!WIN32
    std::vector<const char*> debugArgs;
#ifdef __APPLE__

    debugArgs = createOsasciptArgs();
    std::string arg = createDebugEnvironmentCommandLineForApple();

    arg += "valgrind --trace-children=no --dsymutil=yes ";
    arg += execPath;

    for (int i = 1; args[i]; ++i)
    {
        arg += " ";
        arg += args[i];
    }
    arg += "\"";
    debugArgs.push_back(arg.c_str());
    debugArgs.push_back("-e");
    debugArgs.push_back("end tell");
    debugArgs.push_back(nullptr);
    spawnProgram(execPath, debugArgs);
#endif

    bool defaultCommand = false;
    std::string command = debugCommands;
    if (command.empty())
    {
#ifdef __APPLE__
        command = "konsole --noclose -e valgrind --trace-children=no --dsymutil=yes";
#else
        command = "xterm -hold -e valgrind --trace-children=no";
#endif
        defaultCommand = true;
    }
    debugArgs = parseCmdArgString(command);
    debugArgs.push_back(execPath);
    debugArgs.insert(debugArgs.end(), args.begin(), args.end());
    spawnProgram(execPath, debugArgs);

    if (defaultCommand)
    {

        debugArgs[1] = const_cast<char*>("xterm");
        debugArgs.erase(args.begin());
        spawnProgram(execPath, debugArgs);
    }
#endif //!WIN32

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


