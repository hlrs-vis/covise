/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <util/coFileUtil.h>
#include <dmgr/dmgr.h>
#include <util/unixcompat.h>
#include <util/coSpawnProgram.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <direct.h>
#include <stdlib.h>


#else
#include <sys/wait.h>
#include <signal.h>
#endif

#include "CRB_Module.h"
#include <covise/Covise_Util.h>
#include <comsg/CRB_EXEC.h>

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


module::module(const char *na, const char *ex, const char *ca)
: name(na)
, execpath(ex)
, category(ca)
{
}

void module::set_name(const char *str)
{
    name = str;
}

void module::set_execpath(const char *str)
{
    execpath = str;
}

void module::set_category(const char *str)
{
    category = str;
}

const char *module::get_name() const
{
    return name.c_str();
};
const char *module::get_execpath() const
{
    return execpath.c_str();
};
const char *module::get_category() const
{
    return category.c_str();
};

void module::start(const CRB_EXEC & exec)
{
    auto a = getCmdArgs(exec);
    auto args = cmdArgsToCharVec(a);
    args[0] = execpath.c_str();
    switch (exec.flag)
    {
    case ExecFlag::Normal:
        spawnProgram(execpath.c_str(), args);
        break;
    case ExecFlag::Debug:
        spawnProgramWithDebugger(execpath.c_str(), coCoviseConfig::getEntry("System.CRB.DebugCommand"), args);
        break;
    case ExecFlag::Memcheck:
        spawnProgramWithMemCheck(execpath.c_str(), coCoviseConfig::getEntry("System.CRB.MemcheckCommand"), args);
        break;
    }
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

int moduleList::find(const char *name, const char *category)
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

bool moduleList::start(const CRB_EXEC& exec)
{
    if (find(exec.name, exec.category))
    {
        current()->start(exec);
        return true;;
    }
    return false;
}
