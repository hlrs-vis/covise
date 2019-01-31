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
    typedef std::map<std::string, HostInfo> HostMap;
    HostMap hostMap;

    HostMap::iterator getOrCreateHostInfo(const std::string &);
    void addhostinfo(const std::string &name, int s_mode, int e_mode, int t);
    void addhostinfo_from_config(const std::string &name);

public:
    ControlConfig()
    {
    }
    ~ControlConfig()
    {
    }

    int getshminfo(const std::string &n);
    int getexectype(const std::string &n);
	int gettimeout(const std::string &n);
	int gettimeout(const covise::Host &h);

	char *getDisplayIP(const covise::Host &h);

    int set_shminfo(const std::string &n, const char *shm_mode);
    int set_timeout(const std::string &n, const char *t);
    int set_exectype(const std::string &n, const char *e);
    char *set_display(const std::string &n, const char *e);

};
}
#endif
