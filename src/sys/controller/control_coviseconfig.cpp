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
uint32_t ControlConfig::genip(const char *n)
//----------------------------------------------------------------------------
{
    //std::cerr << "genip: n=" << n << std::endl;
    unsigned addr[4] = {0, 0, 0, 0};

    int no_of_no = sscanf(n, "%u.%u.%u.%u", &addr[0],
                          &addr[1], &addr[2], &addr[3]);

    if (no_of_no == 4)
    {
        return (addr[0] << 24) | (addr[1] << 16) | (addr[2] << 8) | addr[3];
    }

	struct addrinfo hints, *result = NULL;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family = AF_INET;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = 0; /* any type of socket */
	hints.ai_flags = AI_ADDRCONFIG;
	hints.ai_protocol = 0;          /* Any protocol */

	int s = getaddrinfo(n, NULL /* service */, &hints, &result);
	if (s != 0)
	{
		fprintf(stderr, "ControlConfig::genip: getaddrinfo failed for %s: %s\n", n, gai_strerror(s));
		return INADDR_NONE;
	}
	else
	{
		/* getaddrinfo() returns a list of address structures.
		Try each address until we successfully connect(2).
		If socket(2) (or connect(2)) fails, we (close the socket
		and) try the next address. */

		for (struct addrinfo *rp = result; rp != NULL; rp = rp->ai_next)
		{
			if (rp->ai_family != AF_INET)
				continue;

			char address[1000];
			struct sockaddr_in *saddr = reinterpret_cast<struct sockaddr_in *>(rp->ai_addr);
			memcpy(addr, &saddr->sin_addr, sizeof(saddr->sin_addr));
			if (!inet_ntop(rp->ai_family, &saddr->sin_addr, address, sizeof(address)))
			{
				std::cerr << "could not convert address of " << n << " to printable format: " << strerror(errno) << std::endl;
				continue;
			}
			else
			{
				memcpy(addr, &saddr->sin_addr, sizeof(saddr->sin_addr));
				break;
			}
		}

		freeaddrinfo(result);           /* No longer needed */
	}

    return htonl(addr[0]);
}

//----------------------------------------------------------------------------
void ControlConfig::addhostinfo(const char *name, int s_mode, int e_mode, int t)
//----------------------------------------------------------------------------
{
    std::cerr << "adding host info for " << name << std::endl;
    uint32_t ip = genip(name);
    if (ip)
    {
        hostMap[ip].exectype = e_mode;
        hostMap[ip].shminfo = s_mode;
        hostMap[ip].display = NULL;
        hostMap[ip].timeout = t;
    }
}

//----------------------------------------------------------------------------
int ControlConfig::getshminfo_ip(uint32_t ip)
//----------------------------------------------------------------------------
{
    getOrCreateHostInfo(ip);

    return hostMap[ip].shminfo;
}

//----------------------------------------------------------------------------
int ControlConfig::getexectype_ip(uint32_t ip)
//----------------------------------------------------------------------------
{
    getOrCreateHostInfo(ip);

    return hostMap[ip].exectype;
}

//----------------------------------------------------------------------------
char *ControlConfig::getDisplayIP(uint32_t ip)
//----------------------------------------------------------------------------
{
    getOrCreateHostInfo(ip);

    return hostMap[ip].display;
}

//----------------------------------------------------------------------------
int ControlConfig::gettimeout_ip(uint32_t ip)
//----------------------------------------------------------------------------
{
    getOrCreateHostInfo(ip);

    return hostMap[ip].timeout;
}

ControlConfig::HostMap::iterator ControlConfig::getOrCreateHostInfo(uint32_t ip)
{
    HostMap::iterator it = hostMap.find(ip);
    if (it != hostMap.end())
    {
        return it;
    }

    hostMap[ip].timeout = DEFAULT_TIMEOUT;
#ifdef WIN32
    hostMap[ip].exectype = COVISE_REMOTE_DAEMON;
#else
    hostMap[ip].exectype = COVISE_SSH;
#endif
    hostMap[ip].display = NULL;
    hostMap[ip].shminfo = COVISE_SHM;

    char num[128];
    sprintf(num, "%d.%d.%d.%d", (ip >> 24) & 0xff, (ip >> 16) & 0xff, (ip >> 8) & 0xff, ip & 0xff);
    std::string hostname = Host::lookupHostname(num);
    addhostinfo_from_config(hostname.c_str());

    return hostMap.find(ip);
}

//----------------------------------------------------------------------------
int ControlConfig::set_shminfo_ip(uint32_t ip, const char *shm_info)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(ip);

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
int ControlConfig::set_timeout_ip(uint32_t ip, const char *t)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(ip);

    int ti = DEFAULT_TIMEOUT;
    int retval = sscanf(t, "%d", &ti);
    if (retval != 1)
    {
        std::cerr << "ControlConfig::set_timeout_ip: sscanf failed" << std::endl;
    }

    it->second.timeout = ti;

    return (ti);
}

//----------------------------------------------------------------------------
int ControlConfig::set_exectype_ip(uint32_t ip, const char *exec_mode)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(ip);

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
char *ControlConfig::set_display_ip(uint32_t ip, const char *dp)
//----------------------------------------------------------------------------
{
    HostMap::iterator it = getOrCreateHostInfo(ip);

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
int ControlConfig::getexectype(const char *n)
//----------------------------------------------------------------------------
{
    return (getexectype_ip(genip(n)));
}

//----------------------------------------------------------------------------
int ControlConfig::gettimeout(const char *n)
//----------------------------------------------------------------------------
{
    return (gettimeout_ip(genip(n)));
}

//----------------------------------------------------------------------------
int ControlConfig::set_exectype(const char *n, const char *e)
//----------------------------------------------------------------------------
{
    return (set_exectype_ip(genip(n), e));
}

//----------------------------------------------------------------------------
char *ControlConfig::set_display(const char *n, const char *e)
//----------------------------------------------------------------------------
{
    return (set_display_ip(genip(n), e));
}

//----------------------------------------------------------------------------
int ControlConfig::set_shminfo(const char *n, const char *s)
//----------------------------------------------------------------------------
{
    return (set_shminfo_ip(genip(n), s));
}

//----------------------------------------------------------------------------
int ControlConfig::set_timeout(const char *n, const char *t)
//----------------------------------------------------------------------------
{
    return (set_timeout_ip(genip(n), t));
}

//----------------------------------------------------------------------------
int ControlConfig::getshminfo(const char *n)
//----------------------------------------------------------------------------
{
    return (getshminfo_ip(genip(n)));
}

void ControlConfig::addhostinfo_from_config(const char *name)
{
    char shm_mode[100], exec_mode[100], tmp_str[800];
    int s_mode, e_mode, tim;

    /// default values
    strcpy(shm_mode, "shm");
    strcpy(exec_mode, "ssh");
    tim = 60;

    char key[1024];
    snprintf(key, sizeof(key), "HostConfig.%s", name);
    for (int i = (int)strlen("HostConfig."); key[i]; i++)
    {
        if (key[i] == '.')
            key[i] = '_';
    }
    // TODO coconfig
    string entry = coCoviseConfig::getEntry(key);
    if (entry.empty())
    {
        char *p = strchr(key, '_');
        if (p)
        {
            *p = '\0';
            entry = coCoviseConfig::getEntry(key);
        }
    }
    if (!entry.empty())
    {
        int retval = sscanf(entry.c_str(), "%s%s%d", shm_mode, exec_mode, &tim);
        if (retval != 3)
        {
            std::cerr << "ControlConfig::readconfig: sscanf failed" << std::endl;
        }
    }

    s_mode = COVISE_SHM;
    if (strcasecmp(shm_mode, "shm") == 0 || strcasecmp(shm_mode, "sysv") == 0)
    {
        s_mode = COVISE_SHM;
    }
    else if (strcasecmp(shm_mode, "posix") == 0)
    {
        s_mode = COVISE_POSIX;
    }
    else if (strcasecmp(shm_mode, "mmap") == 0)
    {
        s_mode = COVISE_MMAP;
    }
    else if (strcasecmp(shm_mode, "none") == 0)
    {
        s_mode = COVISE_NOSHM;
    }
    else if (strcasecmp(shm_mode, "noshm") == 0)
    {
        s_mode = COVISE_NOSHM;
    }
    else if (strcasecmp(shm_mode, "cray") == 0)
    {
        s_mode = COVISE_NOSHM;
    }
    else if (strcasecmp(shm_mode, "proxie") == 0)
    {
        s_mode = COVISE_PROXIE;
    }
    else
    {
        print_error(__LINE__, __FILE__, "Wrong memory mode %s for %s, should be shm, mmap, or none (covise.config)! Using default shm", shm_mode, name);
        fprintf(stderr, "\n\n%s\n\n", tmp_str);
        fflush(stderr);
    }
    e_mode = COVISE_REXEC;
    if (strcasecmp(exec_mode, "rexec") == 0)
    {
        e_mode = COVISE_REXEC;
    }
    else if (strcasecmp(exec_mode, "rsh") == 0)
    {
        e_mode = COVISE_RSH;
    }
    else if (strcasecmp(exec_mode, "ssh") == 0)
    {
        e_mode = COVISE_SSH;
    }
    else if (strcasecmp(exec_mode, "accessGrid") == 0)
    {
        e_mode = COVISE_ACCESSGRID;
    }
    else if (strcasecmp(exec_mode, "manual") == 0)
    {
        e_mode = COVISE_MANUAL;
    }
    else if (strcasecmp(exec_mode, "SSLDaemon") == 0)
    {
        e_mode = COVISE_SSLDAEMON;
    }
    else if (strcasecmp(exec_mode, "nqs") == 0)
    {
        e_mode = COVISE_NQS;
    }
    else if (strcasecmp(exec_mode, "remoteDaemon") == 0)
    {
        e_mode = COVISE_REMOTE_DAEMON;
    }
    else if (strcasecmp(exec_mode, "globus_gram") == 0)
    {
        e_mode = COVISE_GLOBUS_GRAM;
    }
    else
    {
        print_error(__LINE__, __FILE__, "Wrong exec mode %s for %s, should be rexec, rsh or ssh (covise.config)! Using default rexec", shm_mode, name);
        fprintf(stderr, "\n\n%s\n\n", tmp_str);
        fflush(stderr);
    }
    if (tim == 0)
        tim = DEFAULT_TIMEOUT;

    addhostinfo(name, s_mode, e_mode, tim);
}
