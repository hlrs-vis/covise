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

#define COVISE_POSIX 5
#define COVISE_PROXIE 4
#define COVISE_NOSHM 3
#define COVISE_MMAP 2
#define COVISE_SHM 1

namespace covise
{

enum exec_type
{
    COVISE_REXEC = 1,
    COVISE_RSH,
    COVISE_SSH,
    COVISE_NQS,
    COVISE_MANUAL,
    COVISE_REMOTE_DAEMON,
    COVISE_SSLDAEMON,
    COVISE_SCRIPT,
    COVISE_LOCAL,
    COVISE_ACCESSGRID,
    COVISE_GLOBUS_GRAM,
    COVISE_REATTACH
};

class ControlConfig
{
private:
    struct HostInfo
    {
        int shminfo;
        int exectype;
        int timeout;
        char *display;
    };
    typedef std::map<uint32_t, HostInfo> HostMap;
    HostMap hostMap;

    HostMap::iterator getOrCreateHostInfo(uint32_t ip);
    uint32_t genip(const char *n);
    void addhostinfo(const char *name, int s_mode, int e_mode, int t);
    void addhostinfo_from_config(const char *name);

public:
    ControlConfig()
    {
    }
    ~ControlConfig()
    {
    }

    int getshminfo(const char *n);
    int getexectype(const char *n);
    int gettimeout(const char *n);

    int gettimeout_ip(uint32_t ip);
    int getshminfo_ip(uint32_t ip);
    int getexectype_ip(uint32_t ip);
    char *getDisplayIP(uint32_t ip);

    int set_shminfo(const char *n, const char *shm_mode);
    int set_timeout(const char *n, const char *t);
    int set_exectype(const char *n, const char *e);
    char *set_display(const char *n, const char *e);

    int set_shminfo_ip(uint32_t ip, const char *shm_mode);
    int set_timeout_ip(uint32_t ip, const char *t);
    int set_exectype_ip(uint32_t ip, const char *e);
    char *set_display_ip(uint32_t ip, const char *e);
};
}
#endif
