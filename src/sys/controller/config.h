/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_COVISECONFIG_H_
#define CTRL_COVISECONFIG_H_
#include <covise/covise.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#endif
#include <config/CoviseConfig.h>

#define COVISE_MAXHOSTCONFIG 2000



namespace covise
{
namespace controller{
    
constexpr int DEFAULT_TIMEOUT = 30;

enum class ExecType
{
    Local,
    VRB,
    Manual,
    Script
};

enum class ShmMode {
    Default = 1, //COVISE_SHM 
    MMap, //COVISE_MMAP,
    NoShm, //COVISE_NOSHM, 
    Proxie, //COVISE_PROXIE, 
    Posix //COVISE_POSIX, 
 };

template<typename T>
T &operator<<(T &stream, ExecType execType){
    stream << static_cast<int>(execType);
    return stream;
}

class ControlConfig
{
private:
    struct HostInfo
    {
        ShmMode shmMode = ShmMode::Default;
        controller::ExecType exectype = controller::ExecType::VRB;
        int timeout = DEFAULT_TIMEOUT;
    };
    typedef std::map<std::string, HostInfo> HostMap;
    HostMap hostMap;

    HostMap::iterator getOrCreateHostInfo(const std::string &hostName);
    void addhostinfo(const HostMap::iterator &host, ShmMode s_mode, ExecType e_mode, int t);
    void addhostinfo_from_config(const HostMap::iterator &host);

public:

    ShmMode getshmMode(const std::string &hostName);
    controller::ExecType getexectype(const std::string &hostName);
	int gettimeout(const std::string &hostName);
    ShmMode set_shminfo(const std::string &hostName, const char *shm_mode);
    int set_timeout(const std::string &hostName, const char *t);
    ExecType set_exectype(const std::string &hostName, const char *e);
    char *set_display(const std::string &hostName, const char *e);

};
} //namespace controller 
} //namespace covise

#endif
