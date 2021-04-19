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
#include "config.h"
#include "controlProcess.h"
#include "module.h"

using namespace covise;
using namespace covise::controller;

//----------------------------------------------------------------------------
void ControlConfig::addhostinfo(const HostMap::iterator &host, ShmMode s_mode, ExecType e_mode, int t)
//----------------------------------------------------------------------------
{
    host->second.exectype = e_mode;
    host->second.shmMode = s_mode;
    host->second.timeout = t;
}

ControlConfig::HostMap::iterator ControlConfig::getOrCreateHostInfo(int clientId)
{
    HostMap::iterator it = hostMap.find(clientId);
    if (it == hostMap.end())
    {
        it = hostMap.insert(HostMap::value_type{clientId, HostMap::value_type::second_type{}}).first;
        addhostinfo_from_config(it);
    }
    return it;
}

//----------------------------------------------------------------------------
ShmMode ControlConfig::set_shminfo(int clientId, const char *shm_info)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(clientId);

    int s_info;
    int retval = sscanf(shm_info, "%d", &s_info);
    if (retval != 1)
    {
        std::cerr << "ControlConfig::set_shminfo_ip: sscanf failed" << std::endl;
    }

    it->second.shmMode = static_cast<ShmMode>(s_info);

    return it->second.shmMode;
}

//----------------------------------------------------------------------------
int ControlConfig::set_timeout(int clientId, const char *t)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(clientId);

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
controller::ExecType ControlConfig::set_exectype(int clientId, const char *exec_mode)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(clientId);

    int e_mode = static_cast<int>(ExecType::VRB);
    int retval = sscanf(exec_mode, "%d", &e_mode);
    if (retval != 1)
    {
        std::cerr << "ControlConfig::set_exectype_ip: sscanf failed" << std::endl;
    }
    it->second.exectype = static_cast<ExecType>(e_mode);

    return (it->second.exectype);
}

//----------------------------------------------------------------------------
controller::ExecType ControlConfig::getexectype(int clientId)
//----------------------------------------------------------------------------
{
    return getOrCreateHostInfo(clientId)->second.exectype;
}

//----------------------------------------------------------------------------
int ControlConfig::gettimeout(int clientId)
//----------------------------------------------------------------------------
{
    return getOrCreateHostInfo(clientId)->second.timeout;
}

//----------------------------------------------------------------------------
ShmMode ControlConfig::getshmMode(int clientId)
//----------------------------------------------------------------------------
{
    return getOrCreateHostInfo(clientId)->second.shmMode;
}

void ControlConfig::addhostinfo_from_config(const HostMap::iterator &host)
{
    /// default values

    std::string key = "System.HostConfig.Host:" + std::to_string(host->first);
    host->second.timeout = coCoviseConfig::getInt("timeout", key, DEFAULT_TIMEOUT);
    host->second.timeout = host->second.timeout == 0 ? DEFAULT_TIMEOUT : host->second.timeout;
    std ::string shm_mode = coCoviseConfig::getEntry("memory", key, "shm");
    std ::string exec_mode = coCoviseConfig::getEntry("method", key, "vrb");

    if (strcasecmp(shm_mode.c_str(), "shm") == 0 || strcasecmp(shm_mode.c_str(), "sysv") == 0)
    {
        host->second.shmMode = ShmMode::Default;
    }
    else if (strcasecmp(shm_mode.c_str(), "posix") == 0)
    {
        host->second.shmMode = ShmMode::Posix;
    }
    else if (strcasecmp(shm_mode.c_str(), "mmap") == 0)
    {
        host->second.shmMode = ShmMode::MMap;
    }
    else if (strcasecmp(shm_mode.c_str(), "none") == 0 || strcasecmp(shm_mode.c_str(), "noshm") == 0 || strcasecmp(shm_mode.c_str(), "cray") == 0)
    {
        host->second.shmMode = ShmMode::NoShm;
    }
    else if (strcasecmp(shm_mode.c_str(), "proxie") == 0)
    {
        host->second.shmMode = ShmMode::Proxie;
    }
    else
    {
        print_error(__LINE__, __FILE__, "Wrong memory mode %s for %d, should be shm, mmap, or none (covise.config)! Using default shm", shm_mode.c_str(), host->first);
        fflush(stderr);
    }
    if (strcasecmp(exec_mode.c_str(), "rexec") == 0 ||
        strcasecmp(exec_mode.c_str(), "rsh") == 0 ||
        strcasecmp(exec_mode.c_str(), "ssh") == 0 ||
        strcasecmp(exec_mode.c_str(), "accessGrid") == 0 ||
        strcasecmp(exec_mode.c_str(), "SSLDaemon") == 0 ||
        strcasecmp(exec_mode.c_str(), "nqs") == 0 ||
        strcasecmp(exec_mode.c_str(), "remoteDaemon") == 0 ||
        strcasecmp(exec_mode.c_str(), "globus_gram") == 0)
    {
        std::cerr << "exec mode " << exec_mode << " is no longer supported" << std::endl
                  << "exec mode is set to default(VRB)" << std::endl;
        host->second.exectype = ExecType::VRB;
    }
    else if (strcasecmp(exec_mode.c_str(), "manual") == 0)
    {
        host->second.exectype = ExecType::Manual;
    }
    else if (strcasecmp(exec_mode.c_str(), "vrb") == 0)
    {
        host->second.exectype = ExecType::VRB;
    }
    else if (strcasecmp(exec_mode.c_str(), "script") == 0)
    {
        host->second.exectype = ExecType::Script;
    }
    else
    {
        print_error(__LINE__, __FILE__, "Wrong exec mode %s for %d, should be vrb, manual or script (covise.config)! Using default vrb", shm_mode.c_str(), host->first);
        fflush(stderr);
    }
}
