/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifdef _WIN32
#include <ws2tcpip.h>
#endif
#include <covise/covise.h>
#include <util/unixcompat.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#else
#include <arpa/inet.h>
#include <dirent.h>
#endif

#include <net/covise_host.h>
#include "control_coviseconfig.h"
#include "control_process.h"
#include "covise_module.h"

using namespace covise;

#define DEFAULT_TIMEOUT 30

//----------------------------------------------------------------------------
void ControlConfig::addhostinfo(const std::string &name, int s_mode, int e_mode, int t)
//----------------------------------------------------------------------------
{
    //std::cerr << "adding host info for " << name << std::endl;
    if (name.length()>0)
    {
        hostMap[name].exectype = e_mode;
        hostMap[name].shminfo = s_mode;
        hostMap[name].display = NULL;
        hostMap[name].timeout = t;
    }
}

//----------------------------------------------------------------------------
char *ControlConfig::getDisplayIP(const covise::Host &h)
//----------------------------------------------------------------------------
{
    getOrCreateHostInfo(h.getName());

    return hostMap[h.getName()].display;
}


ControlConfig::HostMap::iterator ControlConfig::getOrCreateHostInfo(const std::string &name)
{
    HostMap::iterator it = hostMap.find(name);
    if (it != hostMap.end())
    {
        return it;
    }

    hostMap[name].timeout = DEFAULT_TIMEOUT;
#ifdef WIN32
    hostMap[name].exectype = COVISE_REMOTE_DAEMON;
#else
    hostMap[name].exectype = COVISE_SSH;
#endif
    hostMap[name].display = NULL;
    hostMap[name].shminfo = COVISE_SHM;

    addhostinfo_from_config(name);

    return hostMap.find(name);
}


//----------------------------------------------------------------------------
int ControlConfig::set_shminfo(const std::string &n, const char *shm_info)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(n);

    int s_info = COVISE_SHM;
    int retval = sscanf(shm_info, "%d", &s_info);
    if (retval != 1)
    {
        std::cerr << "ControlConfig::set_shminfo_ip: sscanf failed" << std::endl;
    }

    it->second.shminfo = s_info;

    return (s_info);
}

//----------------------------------------------------------------------------
int ControlConfig::set_timeout(const std::string &n, const char *t)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(n);

    int ti = DEFAULT_TIMEOUT;
    int retval = sscanf(t, "%d", &ti);
    if (retval != 1)
    {
        std::cerr << "ControlConfig::set_timeout: sscanf failed" << std::endl;
    }

    it->second.timeout = ti;

    return (ti);
}

//----------------------------------------------------------------------------
int ControlConfig::set_exectype(const std::string &n, const char *exec_mode)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(n);

    int e_mode = COVISE_SSH;
    int retval = sscanf(exec_mode, "%d", &e_mode);
    if (retval != 1)
    {
        std::cerr << "ControlConfig::set_exectype_ip: sscanf failed" << std::endl;
    }

    it->second.exectype = e_mode;

    return (e_mode);
}

//----------------------------------------------------------------------------
char *ControlConfig::set_display(const std::string &n, const char *dp)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(n);

    char *dsp = NULL;
    if (dp && strlen(dp) > 1)
    {
        dsp = new char[strlen(dp) + 1];
        strcpy(dsp, dp);
    }

    it->second.display = dsp;

    return (dsp);
}

//----------------------------------------------------------------------------
int ControlConfig::getexectype(const std::string &n)
//----------------------------------------------------------------------------
{
	getOrCreateHostInfo(n);
	return hostMap[n].exectype;
}

//----------------------------------------------------------------------------
int ControlConfig::gettimeout(const std::string &n)
//----------------------------------------------------------------------------
{
	getOrCreateHostInfo(n);

	return hostMap[n].timeout;
}

int covise::ControlConfig::gettimeout(const covise::Host & h)
{
	getOrCreateHostInfo(h.getName());

	return hostMap[h.getName()].timeout;
}

//----------------------------------------------------------------------------
int ControlConfig::getshminfo(const std::string &n)
//----------------------------------------------------------------------------
{
    getOrCreateHostInfo(n);

    return hostMap[n].shminfo;
}

void ControlConfig::addhostinfo_from_config(const std::string &name)
{
	std::string shm_mode, exec_mode;
    int s_mode, e_mode, tim;

    /// default values

    char key[1024];
    snprintf(key, sizeof(key), "System.HostConfig.Host:%s", name.c_str());
    for (int i = (int)strlen("System.HostConfig.Host:"); key[i]; i++)
    {
        if (key[i] == '.')
            key[i] = '_';
    }
	tim = coCoviseConfig::getInt("timeout", key, DEFAULT_TIMEOUT);
	shm_mode = coCoviseConfig::getEntry("memory", key, "shm");
	exec_mode = coCoviseConfig::getEntry("method", key, "ssh");

    s_mode = COVISE_SHM;
    if (strcasecmp(shm_mode.c_str(), "shm") == 0 || strcasecmp(shm_mode.c_str(), "sysv") == 0)
    {
        s_mode = COVISE_SHM;
    }
    else if (strcasecmp(shm_mode.c_str(), "posix") == 0)
    {
        s_mode = COVISE_POSIX;
    }
    else if (strcasecmp(shm_mode.c_str(), "mmap") == 0)
    {
        s_mode = COVISE_MMAP;
    }
    else if (strcasecmp(shm_mode.c_str(), "none") == 0)
    {
        s_mode = COVISE_NOSHM;
    }
    else if (strcasecmp(shm_mode.c_str(), "noshm") == 0)
    {
        s_mode = COVISE_NOSHM;
    }
    else if (strcasecmp(shm_mode.c_str(), "cray") == 0)
    {
        s_mode = COVISE_NOSHM;
    }
    else if (strcasecmp(shm_mode.c_str(), "proxie") == 0)
    {
        s_mode = COVISE_PROXIE;
    }
    else
    {
        print_error(__LINE__, __FILE__, "Wrong memory mode %s for %s, should be shm, mmap, or none (covise.config)! Using default shm", shm_mode.c_str(), name.c_str());
        fflush(stderr);
    }
    e_mode = COVISE_REXEC;
    if (strcasecmp(exec_mode.c_str(), "rexec") == 0)
    {
        e_mode = COVISE_REXEC;
    }
    else if (strcasecmp(exec_mode.c_str(), "rsh") == 0)
    {
        e_mode = COVISE_RSH;
    }
    else if (strcasecmp(exec_mode.c_str(), "ssh") == 0)
    {
        e_mode = COVISE_SSH;
    }
    else if (strcasecmp(exec_mode.c_str(), "accessGrid") == 0)
    {
        e_mode = COVISE_ACCESSGRID;
    }
    else if (strcasecmp(exec_mode.c_str(), "manual") == 0)
    {
        e_mode = COVISE_MANUAL;
    }
    else if (strcasecmp(exec_mode.c_str(), "SSLDaemon") == 0)
    {
        e_mode = COVISE_SSLDAEMON;
    }
    else if (strcasecmp(exec_mode.c_str(), "nqs") == 0)
    {
        e_mode = COVISE_NQS;
    }
    else if (strcasecmp(exec_mode.c_str(), "remoteDaemon") == 0)
    {
        e_mode = COVISE_REMOTE_DAEMON;
    }
    else if (strcasecmp(exec_mode.c_str(), "globus_gram") == 0)
    {
        e_mode = COVISE_GLOBUS_GRAM;
    }
    else
    {
        print_error(__LINE__, __FILE__, "Wrong exec mode %s for %s, should be rexec, rsh or ssh (covise.config)! Using default rexec", shm_mode.c_str(), name.c_str());
        fflush(stderr);
    }
    if (tim == 0)
        tim = DEFAULT_TIMEOUT;

    addhostinfo(name, s_mode, e_mode, tim);
}
