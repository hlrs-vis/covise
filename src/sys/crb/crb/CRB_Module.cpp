/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <util/coFileUtil.h>
#include <dmgr/dmgr.h>
#include <util/unixcompat.h>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <direct.h>
#include <stdlib.h>

//#define PURIFY
#define PURIFY_COMMAND "C:\\Progra~1\\Rational\\Common\\purify.exe"

#else
#include <sys/wait.h>
#include <signal.h>
#endif

#include "CRB_Module.h"
#include <covise/Covise_Util.h>

#ifndef NO_VRB
#include <vrbclient/VRBClient.h>
#endif

#ifdef __APPLE__
extern char **environ;
#endif

using namespace covise;

extern bool rendererIsPossible;
extern bool rendererIsActive;
extern DataManagerProcess *datamgr;

// this is our own C++ conformant strcpy routine
inline char *STRDUP(const char *old)
{
    return strcpy(new char[strlen(old) + 1], old);
}

module::module()
{
    name = NULL;
    execpath = NULL;
    category = NULL;
}

module::module(const char *na, const char *ex, const char *ca)
{
    name = new char[strlen(na) + 1];
    strcpy(name, na);
    execpath = new char[strlen(ex) + 1];
    strcpy(execpath, ex);
    category = new char[strlen(ca) + 1];
    strcpy(category, ca);
}

module::~module()
{
    if (NULL != name)
    {
        delete[] name;
    }
    if (NULL != execpath)
    {
        delete[] execpath;
    }
    if (NULL != category)
    {
        delete[] category;
    }
}

void module::set_name(const char *str)
{
    if (NULL != name)
    {
        delete[] name;
    }
    name = new char[strlen(str) + 1];
    strcpy(name, str);
}

void module::set_execpath(const char *str)
{
    if (NULL != execpath)
    {
        delete[] execpath;
    }
    execpath = new char[strlen(str) + 1];
    strcpy(execpath, str);
}

void module::set_category(const char *str)
{
    if (NULL != category)
    {
        delete[] category;
    }
    category = new char[strlen(str) + 1];
    strcpy(category, str);
}

void module::start(char *parameter, Start::Flags flags)
{
    char *argv[100];
    char *tmp = parameter;
    int argc = 2;
    argv[0] = name;
    argv[1] = tmp;
    while (*tmp)
    {
        if (*tmp == ' ')
        {
            *tmp = '\0';
            if (*(tmp + 1))
            {
                argv[argc] = tmp + 1;
                argc++;
            }
        }
        tmp++;
    }
    argv[argc] = NULL;
// Special Hack for COVER, OPENCOVER
// prior to starting COVER look for VRB and connect to an already running one
//
#ifndef NO_VRB
    int sl = (int)strlen(name);
    if (sl > 5)
    {
        if (strncmp(name + sl - 5, "COVER", 5) == 0 || strcmp(name, "VRRenderer") == 0)
        {
            VRBClient vrbc("CRB");
            if (vrbc.connectToServer() >= 0)
            {
                if (vrbc.isCOVERRunning())
                {
                    cerr << "connecting to COVER" << endl;
                    vrbc.connectToCOVISE(argc, (const char **)argv);
                    return;
                }
            }
        }
    }
#endif

#ifdef _WIN32
    string win_cmd_line;
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
#ifdef PURIFY
    for (int ctr = argc; ctr > 0; --ctr)
    {
        argv[ctr + 1] = argv[ctr];
    }
    argv[1] = execpath;
    argv[0] = "/run";
    argc += 2;
    /*printf("Running '%s", PURIFY_COMMAND);
   for (int ctr = 0; ctr < argc; ++ctr) {
     printf(" %s", argv[ctr]);
   }
   printf("'\n");*/

    spawnv(P_NOWAIT, PURIFY_COMMAND, (const char *const *)argv);
#else
    /* printf("Running '%s", execpath);
    for (int ctr = 0; ctr < argc; ++ctr) {
      printf(" %s", argv[ctr]);
    }
    printf("'\n");*/

    //spawnv(P_NOWAIT,execpath, (const char *const *)argv);
    win_cmd_line = argv[0];
    for (int i = 1; i < argc; i++)
    {
        win_cmd_line.append(" ");
        win_cmd_line.append(argv[i]);
    }

    if (flags == Start::Normal)
    {
        // Start the child process.
        if (!CreateProcess(execpath, // No module name (use command line)
                           (LPSTR)win_cmd_line.c_str(), // Command line
                           NULL, // Process handle not inheritable
                           NULL, // Thread handle not inheritable
                           FALSE, // Set handle inheritance to FALSE
                           0, // No creation flags
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
            // Wait until child process exits.
            //WaitForSingleObject( pi.hProcess, INFINITE );
            // Close process and thread handles.
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
    }
    else
    {
        // launch & debug
        // launch the child process "suspended" then attach debugger and resume child's main thread
        if (!CreateProcess(execpath, // No module name (use command line)
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
            string debug_cmd_line("qtcreator -debug ");
#else
            string debug_cmd_line("vsjitdebugger -p ");
#endif
            char pid_str[16];
            itoa(pi.dwProcessId, pid_str, 10);
            debug_cmd_line.append(pid_str);

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
    }
#endif

#else
    int pid = fork();
    if (0 == pid)
    {
        if (flags == Start::Normal)
        {
            execv(execpath, argv);
        }
        else
        {
            char *nargv[120];
#ifdef __APPLE__
            std::string tool;

            std::string covisedir;
            if (getenv("COVISEDIR"))
                covisedir = getenv("COVISEDIR");

            std::vector<std::string> args;
            args.push_back("osascript");
            args.push_back("-e");
            args.push_back("tell application \"Terminal\"");
            args.push_back("-e");
            args.push_back("activate");
            args.push_back("-e");

            std::vector<std::string> env;
            env.push_back("COVISEDIR");
            env.push_back("ARCHSUFFIX");
            env.push_back("COCONFIG");

            std::string arg = "do script with command \"";
            for (std::vector<std::string>::iterator it = env.begin();
                    it != env.end();
                    ++it)
            {
                if (getenv(it->c_str()))
                    arg += "export " + *it + "='" + getenv(it->c_str()) + "'; ";
            }
            arg += "export CO_MODULE_BACKEND=covise; ";
            arg += "source '" + covisedir + "/.covise.sh'; ";
            arg += "source '" + covisedir + "/scripts/covise-env.sh'; ";
            if (flags == Start::Memcheck)
            {
                arg += "valgrind --trace-children=no --dsymutil=yes ";
                arg += execpath;
            }
            else if (flags == Start::Debug)
            {
                //arg += "gdb --args ";
                arg += "/Applications/Xcode.app/Contents/Developer/usr/bin/lldb ";
                arg += execpath;
                arg += " -- ";
            }
            for (int i = 1; argv[i]; ++i)
            {
                arg += " ";
                arg += argv[i];
            }
            if (flags == Start::Debug)
                arg += "; exit";
            arg += "\"";
            args.push_back(arg);
            args.push_back("-e");
            args.push_back("end tell");

            for (int i = 0; i < args.size(); ++i)
            {
                nargv[i] = const_cast<char *>(args[i].c_str());
            }
            nargv[args.size()] = NULL;
            execvp("osascript", nargv);
#endif
            bool defaultCommand = false;
            std::string command;
            if (flags == Start::Debug)
            {
                command = coCoviseConfig::getEntry("System.CRB.DebugCommand");
                if (command.empty())
                {
                    command = "konsole -e gdb --args";
                    defaultCommand = true;
                }
            }
            else if (flags == Start::Memcheck)
            {
                command = coCoviseConfig::getEntry("System.CRB.MemcheckCommand");
                if (command.empty())
                {
#ifdef __APPLE__
                    command = "konsole --noclose -e valgrind --trace-children=no --dsymutil=yes";
#else
                    command = "xterm -hold -e valgrind --trace-children=no";
#endif
                    defaultCommand = true;
                }
            }
            char *buf = new char[command.length() + 1];
            strcpy(buf, command.c_str());

            char *help;
            nargv[0] = strtok_r(buf, " ", &help);
            int i = 1;
            while (char *p = strtok_r(NULL, " ", &help))
            {
                nargv[i] = p;
                ++i;
            }
            nargv[i] = execpath;
            for (int j = 1; j < sizeof(argv); ++j)
            {
                nargv[i + j] = argv[j];
                if (!argv[j])
                    break;
            }

            execvp(nargv[0], nargv);

            if (defaultCommand)
            {
                // try again with xterm
                if (flags == Start::Debug)
                {
                    nargv[0] = const_cast<char *>("xterm");
                    execvp(nargv[0], nargv);
                }
                else
                {
                    nargv[1] = const_cast<char *>("xterm");
                    execvp(nargv[1], &nargv[1]);
                }
            }
        }

        fprintf(stderr, "executing %s failed: %s\n", execpath, strerror(errno));
        _exit(1);
    }
    if (-1 == pid)
    {
        fprintf(stderr, "forking for executing %s failed: %s\n", execpath, strerror(errno));
        exit(1);
    }
    else
    {
        //Needed to prevent zombies
        //if childs terminate
        signal(SIGCHLD, SIG_IGN);
    }
#endif
}

moduleList::moduleList()
    : DLinkList<module *>()
{
    // search for Modules in Covise_dir/bin

    char *tmpp, *dirname, buf[500];

    const char *covisepath = getenv("COVISE_PATH");
    if (covisepath == NULL)
    {
        //print_comment(__LINE__, __FILE__, "ERROR: COVISE_PATH not defined!\n");
        //print_exit(__LINE__, __FILE__, 1);
        cerr << "*                                                             *" << endl;
        cerr << "* COVISE_PATH variable not set !!!                            *" << endl;
        cerr << "*                                                             *" << endl;
        covisepath = "";
    }

    const char *archsuffix = getenv("ARCHSUFFIX");
    if (archsuffix == NULL)
    {
        //print_comment(__LINE__, __FILE__, "ERROR: ARCHSUFFIX not defined!\n");
        //print_exit(__LINE__, __FILE__, 1);
        cerr << "*                                                             *" << endl;
        cerr << "* ARCHSUFFIX variable not set !!!                             *" << endl;
        cerr << "*                                                             *" << endl;
        archsuffix = "";
    }

    coCoviseConfig::ScopeEntries mae = coCoviseConfig::getScopeEntries("System.CRB", "ModuleAlias");
    const char **moduleAliases = mae.getValue();
    for (int i = 0; moduleAliases && moduleAliases[i] != NULL && moduleAliases[i+1] != NULL; i = i + 2)
    {
        //fprintf(stderr, "___ %s___%s\n", moduleAliases[i], moduleAliases[i+1]);
        char *line = new char[strlen(moduleAliases[i]) + 1];
        strcpy(line, moduleAliases[i]);
        strtok(line, ":");
        char *newName = strtok(NULL, ":");

        char *oldName = new char[strlen(moduleAliases[i + 1]) + 1];
        strcpy(oldName, moduleAliases[i + 1]);

        //fprintf(stderr, "module alias: %s -> %s\n", newName, oldName);
        aliasMap.insert(Alias(oldName, newName));
        aliasedSet.insert(newName);
    }

    tmpp = STRDUP(covisepath);

#ifdef _WIN32
    dirname = strtok(tmpp, ";");
#else
    dirname = strtok(tmpp, ":");
#endif

    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s/bin", dirname, archsuffix);
        coDirectory *dir = coDirectory::open(buf);
        if (dir)
        {
            for (int i = 0; i < dir->count(); i++) // skip . and ..
            {
                if (dir->is_directory(i) && strcmp(dir->name(i), ".") && strcmp(dir->name(i), ".."))
                {
                    char *tmp = dir->full_name(i);

#ifdef _WIN32
                    int len = (int)strlen(tmp);
                    if (len >= 4 && !strcmp(&tmp[len - 4], ".exe"))
                    {
                        delete[] tmp;
                        continue;
                    }
#endif

#ifdef __APPLE__
                    int len = strlen(tmp);
                    if (len >= 4 && !strcmp(&tmp[len - 4], ".app"))
                    {
                        delete[] tmp;
                        continue;
                    }
#endif
                    search_dir(tmp, (char *)dir->name(i));
                    delete[] tmp;
                }
            }
            delete dir;
        }

#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }

    if (num() == 0)
    {
        //print_comment(__LINE__, __FILE__, "No Modules Found\n");
        cerr << "*                                                             *" << endl;
        cerr << "************ ERROR     ERROR      ERROR      ERROR ************" << endl;
        cerr << "*   No Modules Found                                          *" << endl;

        cerr << "*    Perhaps COVISE_PATH = \"" << covisepath << "\" is not correct " << endl;
        cerr << "***************************************************************" << endl;
        //print_exit(__LINE__, __FILE__, 1);
        exit(0);
    }
}

void moduleList::appendModule(const char *name, const char *execpath, const char *category)
{
    char *search = new char[strlen(name) + strlen(category) + 2];
    sprintf(search, "%s/%s", category, name);
    if (aliasedSet.find(search) != aliasedSet.end())
    {
        //fprintf(stderr, "%s is object of an alias, ignored\n", search);
        delete[] search;
        return;
    }

    append(new module(name, execpath, category));
    AliasMap::iterator end = aliasMap.upper_bound(search);
    for (AliasMap::iterator it = aliasMap.lower_bound(search);
         it != end;
         it++)
    {
        //fprintf(stderr, "alias %s -> %s\n", it->first, it->second);
        char *newCategory = new char[strlen(it->second) + 1];
        strcpy(newCategory, it->second);
        char *strtok_data = NULL;
        strtok_r(newCategory, "/", &strtok_data);
        char *newName = strtok_r(NULL, "/", &strtok_data);
        append(new module(newName, execpath, newCategory));
        //printf("appending category %s\n", newCategory);
        delete[] newCategory;
    }

    delete[] search;
}

void moduleList::search_dir(char *path, char *subdir)
{
    coDirectory *dir = coDirectory::open(path);
    if (NULL != dir)
    {
        for (int i = 0; i < dir->count(); i++)
        {
            if (!strcmp(dir->name(i), ".") || !strcmp(dir->name(i), ".."))
                continue;

            char *tmp = dir->full_name(i);

#ifdef _WIN32
            int len2 = (int)strlen(dir->name(i));
            char *modname = new char[len2 + 1];
            strcpy(modname, dir->name(i));
            if (len2 >= 4 && !strcmp(&modname[len2 - 4], ".exe"))
            {
                modname[len2 - 4] = '\0';
                appendModule(modname, tmp, subdir);
                delete[] modname;
                delete[] tmp;
            }
#endif
#ifdef __APPLE__
            int len = strlen(tmp);
            //fprintf(stderr, "tmp=%s\n", tmp);
            if (len >= 4 && !strcmp(&tmp[len - 4], ".app"))
            {
                char *execpath = new char[2 * len + strlen("/Contents/MacOS/") + 1];
                strcpy(execpath, tmp);
                strcat(execpath, "/Contents/MacOS/");
                int len2 = strlen(dir->name(i));
                char *modname = new char[len2 + 1];
                strcpy(modname, dir->name(i));
                if (len2 >= 4 && !strcmp(&modname[len2 - 4], ".app"))
                {
                    modname[len2 - 4] = '\0';
                }
                strcat(execpath, modname);
                appendModule(modname, execpath, subdir);
                delete[] modname;
                delete[] execpath;
                delete[] tmp;
                continue;
            }
#endif
#ifndef _WIN32
            if (dir->is_exe(i))
            {
                if (!find((char *)dir->name(i), subdir))
                    appendModule((char *)dir->name(i), tmp, subdir);
                delete[] tmp;
            }

#endif
        }
        delete dir;
    }
}

int moduleList::find(char *name, char *category)
{
    // Find a specific module return true if found and set current position

    reset();
    while (current())
    {
        if ((strcmp(current()->get_name(), name) == 0) && (strcmp(current()->get_category(), category) == 0))
            return (1);
        next();
    }
    return (0);
}

void moduleList::startRenderer(char *name, char *category)
{
    CharBuffer buf(200);
    if (find(name, category))
    {
        if (rendererIsPossible && !rendererIsActive && strstr(current()->get_execpath(), "ViNCE") != 0)
        {
            buf += "YES";
        }
        else
        {
            buf += "NO";
        }
    }
    char* d = buf.return_data();
    Message retmsg{ COVISE_MESSAGE_UI, DataHandle{d, strlen(d) + 1} };
    datamgr->send_ctl_msg(&retmsg);
}

char *moduleList::get_list_message()
{
    CharBuffer buf(num() * 20);
    buf += "LIST\n";
    buf += num();
    buf += '\n';
    reset();
    while (current())
    {
        buf += current()->get_name();
        buf += '\n';
        buf += current()->get_category();
        buf += '\n';
        next();
    }
    return (buf.return_data());
}

int moduleList::start(char *name, char *category, char *parameter, Start::Flags flags)
{
    if (find(name, category))
    {
        current()->start(parameter, flags);
        return (1);
    }

    else
    {
        return (0);
    }
}
