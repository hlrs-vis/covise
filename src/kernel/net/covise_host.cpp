/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>

#include <sys/types.h>

#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#ifdef _WIN32
#include <WS2tcpip.h>
#define inet_pton InetPton
#else
#include <errno.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#endif

#include <util/unixcompat.h>

#include <util/coWristWatch.h>

#include "covise_host.h"

#undef DEBUG
using namespace covise;

std::string Host::lookupIpAddress(const char *hostname)
{
#ifdef DEBUG
    coWristWatch watch;
#endif
    Host ch(hostname);
    const char* c = ch.getAddress();
    if (!c)
    {
        std::cerr << "Host: lookupIpAddress failed: hostname = " << hostname << std::endl;
        return std::string();
    }
    std::string retVal(c);

#ifdef DEBUG
    fprintf(stderr, "lookup result for %s: %s (%f s)\n", hostname, retVal.c_str(), watch.elapsed());
#endif
    return retVal;
}

std::string Host::lookupHostname(const char *numericIP)
{
#ifdef DEBUG
    coWristWatch watch;
#endif
    std::string retVal;
    static bool onlyNumeric = false;
    if (!onlyNumeric)
    {

		struct sockaddr_in sa;
		char hostname[NI_MAXHOST];
		memset(&sa, 0, sizeof sa);
		sa.sin_family = AF_INET;
        int err = inet_pton(AF_INET, numericIP, &sa.sin_addr);
        if (err == 1)
        {

			int res = getnameinfo((struct sockaddr*)&sa, sizeof(sa),
				hostname, sizeof(hostname),
				NULL, 0, NI_NAMEREQD);

			if (!res) 
			{
				retVal = hostname;
				return retVal;
			}
        }
#if 0
        if (err == 0)
        {
            struct in6_addr v6;
            err = inet_pton(AF_INET6, numericIP, &v6);
            if (err == 1)
                he = gethostbyaddr(&v6, sizeof(v6), AF_INET6);
        }
#endif
        
		retVal = numericIP;
		//TODO coConfig - das muss wieder richtig geparst werden
		coCoviseConfig::ScopeEntries ipe = coCoviseConfig::getScopeEntries("System.IpTable");
		const char **ipEntries = ipe.getValue();
		const char *last;
		if (NULL != ipEntries)
		{
			bool gotAll = false;
			bool found = false;
			do
			{
				//An IpTable Entry has the form
				//<symbolic> <numeric>
				//The CoviseConfig::getScopeEntries
				//method gets them word by word
				//so we have to parse two of them
				last = *ipEntries;
				fprintf(stderr, "IPTABLE:%s ", last);
				ipEntries++;
				if (NULL != *ipEntries)
				{
					fprintf(stderr, "IPTABLE:%s \n", *ipEntries);
					if (0 == strcmp(numericIP, *ipEntries))
					{
						//We found the entry
						retVal = last;
						found = true;
					}
					else
					{
						//There is an entry, but it does not match
						ipEntries++;
						if (NULL == *ipEntries)
						{
							onlyNumeric = true;
							gotAll = true;
							retVal = numericIP;
						}
					}
				}
				else
				{
					//We got all entries, the last of which is incomplete
					onlyNumeric = true;
					gotAll = true;
					retVal = numericIP;
				}
			} while ((!gotAll) && (!found));
		}
		else
		{
			onlyNumeric = true;
		}
	}
#ifdef DEBUG
    fprintf(stderr, "lookup result for %s: %s (%f s)\n", numericIP, retVal.c_str(), watch.elapsed());
#endif
    return retVal;
}

void Host::setAddress(const char *n)
{
    m_address.clear();
    if (n)
    {
        m_address = n;
        m_addressValid = true;
    }
    else
    {
        m_addressValid = false;
    }
}

void Host::setName(const char *n)
{
    m_name.clear();
    if (n)
    {
        m_name = n;
        m_nameValid = true;
    }
    else
    {
        m_nameValid = false;
    }
}

void Host::HostNumeric(const char *n)
{
    struct in_addr v4;
    int err = inet_pton(AF_INET, n, &v4);
    memcpy(char_address, &v4, sizeof(char_address));
#if 0
    if (err == 0)
    {
        struct in6_addr v6;
        err = inet_pton(AF_INET6, n, &v6);
    }
#endif
    if (err == 1)
    {
        setAddress(n);
        setName(lookupHostname(n).c_str());
    }
    else
    {
        if (err == 0)
        {
            std::cerr << "Host::HostNumeric: unsupported address format" << std::endl;
        }
        else
        {
            std::cerr << "Host::HostNumeric: error: " << strerror(errno) << std::endl;
        }
        setAddress("Invalid IP address");
        setName(NULL);
        memset(char_address, 0, sizeof(char_address));
    }
#ifdef DEBUG
    LOGINFO(address);
#endif
}

void Host::HostSymbolic(const char *n)
{
    //std::cerr << "HostSymbolic: n=" << (n?n:"(null)") << std::endl;
    //The address is not numeric
    //and we try to convert the
    //symbolic address into a numeric one
    //I) By searching an entry in covise.config
    //II) By  gethostbyname
    //III) If this fails we get "unresolvable IP address"

    //TODO coConfig - richtig parsen
    std::string addr = coCoviseConfig::getEntry(std::string("System.IpTable.") + n);
    if (!addr.empty())
    {
        HostNumeric(addr.c_str());
        return;
    }

    struct in_addr v4;
    int err = inet_pton(AF_INET, n, &v4);
    if (err == 1)
    {
        memcpy(char_address, &v4, sizeof(char_address));
        setAddress(n);
        auto name = lookupHostname(n);
        if (name.empty())
            setName(n);
        else
            setName(name.c_str());
        return;
    }
    if (err == 0)
    {
        struct in6_addr v6;
        err = inet_pton(AF_INET6, n, &v6);
    }
    if (err == 1)
    {
        setAddress(n);
        auto name = lookupHostname(n);
        if (name.empty())
            setName(n);
        else
            setName(name.c_str());
        return;
    }
    memset(char_address, 0, sizeof(char_address));

#if 0
    struct hostent *hent = gethostbyname(n);
    if (NULL == hent)
    {
        fprintf(stderr, "lookup for %s failed\n", n);
        if (strchr(n, ' '))
            abort();
        setAddress(NULL);
        return;
    }
    if (hent->h_addrtype == AF_INET6)
    {
        fprintf(stderr, "cannot handle inet6 address: lookup for %s failed\n", n);
    }
    else if (hent->h_addrtype == AF_INET)
    {
        char_address[0] = *hent->h_addr_list[0];
        char_address[1] = *(hent->h_addr_list[0] + 1);
        char_address[2] = *(hent->h_addr_list[0] + 2);
        char_address[3] = *(hent->h_addr_list[0] + 3);
        char buf[1024];
        sprintf(buf, "%d.%d.%d.%d",
                char_address[0],
                char_address[1],
                char_address[2],
                char_address[3]);
        setAddress(buf);
    }
#else

    struct addrinfo hints, *result = NULL;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
    hints.ai_socktype = 0; /* any type of socket */
    hints.ai_flags = AI_ADDRCONFIG;
    hints.ai_protocol = 0;          /* Any protocol */

    //std::cerr << "HostSymbolic: calling getaddrinfo for n=" << n << std::endl;
    int s = getaddrinfo(n, NULL /* service */, &hints, &result);
    //std::cerr << "HostSymbolic: after calling getaddrinfo for n=" << n << std::endl;
    if (s != 0)
    {
        //std::cerr << "Host::HostSymbolic: getaddrinfo failed for " << n << ": " << s << " " << gai_strerror(s) << std::endl;
        fprintf(stderr, "Host::HostSymbolic: getaddrinfo failed for %s: %s\n", n, gai_strerror(s));
        return;
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
            memcpy(char_address, &saddr->sin_addr, sizeof(char_address));
            if(!inet_ntop(rp->ai_family, &saddr->sin_addr, address, sizeof(address)))
            {
                std::cerr << "could not convert address of " << n << " to printable format: " << strerror(errno) << std::endl;
                continue;
            }
            else
            {
                memcpy(char_address, &saddr->sin_addr, sizeof(char_address));
                setAddress(address);
                break;
            }
        }

        freeaddrinfo(result);           /* No longer needed */
    }
#endif
    setName(n);
}

Host::Host(const char *n, bool numeric)
{
    //std::cerr << "Host: n=" << (n?n:"(null)") << ", numeric=" << numeric << std::endl;
    memset(char_address, '\0', sizeof(char_address));
    m_addressValid = false;
    m_nameValid = false;

    setName(n);

    if ((numeric) || (NULL == n))
    {
        HostNumeric(n);
    }
    else
    {
        HostSymbolic(n);
    }
}

Host::Host(const Host &h)
{
    memset(char_address, '\0', sizeof(char_address));
    memcpy(this->char_address, h.char_address, sizeof(char_address));

    m_addressValid = h.m_addressValid;
    m_nameValid = h.m_nameValid;
    m_address = h.m_address;
    m_name = h.m_name;
}

Host::Host(unsigned long a)
{
    memset(char_address, '\0', sizeof(char_address));
    m_addressValid = false;
    m_nameValid = false;

    unsigned char *tmpaddr = (unsigned char *)&a;
    char_address[0] = tmpaddr[0];
    char_address[1] = tmpaddr[1];
    char_address[2] = tmpaddr[2];
    char_address[3] = tmpaddr[3];

    char tmpName[256];
    sprintf(tmpName, "%d.%d.%d.%d",
            char_address[0],
            char_address[1],
            char_address[2],
            char_address[3]);
    setAddress(tmpName);
    if (a == 0)
        setName(tmpName);
    else
        setName(Host::lookupHostname(tmpName).c_str());
}

Host::~Host()
{
}

Host &Host::operator=(const Host &h)
{
    memset(char_address, '\0', sizeof(char_address));
    memcpy(this->char_address, h.char_address, sizeof(char_address));

    m_addressValid = h.m_addressValid;
    m_nameValid = h.m_nameValid;
    m_address = h.m_address;
    m_name = h.m_name;

    return *this;
}

void Host::get_char_address(unsigned char *c) const
{
    c[0] = char_address[0];
    c[1] = char_address[1];
    c[2] = char_address[2];
    c[3] = char_address[3];
}

const unsigned char *Host::get_char_address() const
{
    return char_address;
}

uint32_t Host::get_ipv4() const
{
    return (char_address[0] << 24) | (char_address[1] << 16) | (char_address[2] << 8) | char_address[3];
}

const char *Host::getName() const
{
    if (!m_nameValid)
        return NULL;
    return m_name.c_str();
}

const char *Host::getAddress() const
{
    if (!m_addressValid)
        return NULL;
    return m_address.c_str();
}

static bool isLoopbackAddress(const unsigned char address[4])
{
    if (address[0] == 127)
        return true;

    return false;
}

static bool isPrivateAddress(const unsigned char address[4])
{
    // RFC 1918
    if (address[0] == 10)
        return true;
    if (address[0] == 192 && address[1] == 168)
        return true;
    if (address[0] == 172 && ((address[1] & 0xf0) == 16))
        return true;

    return false;
}

static bool isMulticastAddress(const unsigned char address[4])
{
    if ((address[0] & 0xf0) == 224)
        return true;

    return false;
}

static bool isRoutableAddress(const unsigned char address[4])
{
    if (address[0] == 0 && address[1] == 1 && address[2] == 0 && address[3] == 0)
        return false;

    if (address[0] == 255 && address[1] == 255 && address[2] == 255 && address[3] == 255)
        return false;

    if (isMulticastAddress(address))
        return false;

    if (isPrivateAddress(address))
        return false;

    if (isLoopbackAddress(address))
        return false;

    return true;
}

#ifndef _WIN32
std::vector<ifreq *> getNetworkInterfaces(char *buf, size_t buflen)
{
    std::vector<ifreq *> result;

    memset(buf, 0, buflen);

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd == -1)
        return result;

    struct ifconf ifconfig;
    ifconfig.ifc_buf = buf;
    ifconfig.ifc_len = buflen;

    if (ioctl(fd, SIOCGIFCONF, &ifconfig))
    {
        close(fd);
        return result;
    }

    int remaining = ifconfig.ifc_len;
    struct ifreq *ifrp = ifconfig.ifc_req;
    std::vector<ifreq *> ifs;
    while (remaining > 0)
    {
#ifdef __APPLE__
        int current = ifrp->ifr_addr.sa_len + IFNAMSIZ;
#else
        int current = sizeof(struct ifreq);
#endif
        ifrp = (struct ifreq *)(((char *)ifrp) + current);
        remaining -= current;
        ifs.push_back(ifrp);
    }

    for (int i = 0; i < ifs.size(); ++i)
    {
        if (ioctl(fd, SIOCGIFADDR, ifs[i]))
            continue;

        if (ioctl(fd, SIOCGIFFLAGS, ifs[i]) != 0)
            continue;

        if ((ifs[i]->ifr_flags & IFF_UP) == 0)
            continue;

        result.push_back(ifs[i]);
    }

    close(fd);
    return result;
}
#endif

static bool isAddressConfigured(unsigned char address[4])
{
#ifndef _WIN32
    char buf[10240];
    std::vector<ifreq *> ifs = getNetworkInterfaces(buf, sizeof(buf));

    int n = ifs.size();
    // try to find something that is not a VMware interface
    for (int i = n - 1; i >= 0; --i)
    {
        struct in_addr ipaddr = (*(struct sockaddr_in *)&ifs[i]->ifr_addr).sin_addr;
        in_addr_t addr = ipaddr.s_addr;
        unsigned char a[4] = {
            static_cast<unsigned char>(addr & 0xff),
            static_cast<unsigned char>((addr >> 8) & 0xff),
            static_cast<unsigned char>((addr >> 16) & 0xff),
            static_cast<unsigned char>((addr >> 24) & 0xff)
        };

        bool equal = true;
        for (int j = 0; j < 4; ++j)
        {
            if (a[j] != address[j])
            {
                equal = false;
                break;
            }
        }

        if (equal)
            return true;
    }
    return false;
#else
    return true;
#endif
}

// inspired by Samba 3.0.28a's _get_interfaces from lib/interfaces.c
static bool findPrimaryIpAddress(unsigned char address[4])
{
#ifndef _WIN32
    char buf[10240];
    std::vector<ifreq *> ifrequest = getNetworkInterfaces(buf, sizeof(buf));

    int n = ifrequest.size();
    // try to find something that is not a VMware interface
    for (int i = n - 1; i >= 0; --i)
    {
        if (!strncmp(ifrequest[i]->ifr_name, "vmnet", 5))
            continue;

        struct in_addr ipaddr = (*(struct sockaddr_in *)&ifrequest[i]->ifr_addr).sin_addr;
        in_addr_t addr = ipaddr.s_addr;
        address[0] = addr & 0xff;
        address[1] = (addr >> 8) & 0xff;
        address[2] = (addr >> 16) & 0xff;
        address[3] = (addr >> 24) & 0xff;

        if (isRoutableAddress(address))
            return true;
    }

    // use anything except vmnet interfaces
    for (int i = n - 1; i >= 0; --i)
    {
        if (!strncmp(ifrequest[i]->ifr_name, "vmnet", 5))
            continue;

        struct in_addr ipaddr = (*(struct sockaddr_in *)&ifrequest[i]->ifr_addr).sin_addr;
        in_addr_t addr = ipaddr.s_addr;
        address[0] = addr & 0xff;
        address[1] = (addr >> 8) & 0xff;
        address[2] = (addr >> 16) & 0xff;
        address[3] = (addr >> 24) & 0xff;

        if (isPrivateAddress(address))
            return true;
    }

    // use anything
    for (int i = n - 1; i >= 0; --i)
    {
        struct in_addr ipaddr = (*(struct sockaddr_in *)&ifrequest[i]->ifr_addr).sin_addr;
        in_addr_t addr = ipaddr.s_addr;
        address[0] = addr & 0xff;
        address[1] = (addr >> 8) & 0xff;
        address[2] = (addr >> 16) & 0xff;
        address[3] = (addr >> 24) & 0xff;

        if (isPrivateAddress(address))
            return true;
    }

#endif
    return false;
}

bool Host::hasRoutableAddress() const
{
    return isRoutableAddress(char_address);
}

Host::Host()
{
    //std::cerr << "Host()" << std::endl;
    m_addressValid = false;
    m_nameValid = false;

    char *hostname = getenv("COVISE_HOST");
    if (!hostname)
        hostname = getenv("COVISEHOST");
    if (hostname)
    {
        HostSymbolic(hostname);
        if (hasValidAddress())
            return;

        std::cerr << "DNS lookup for hostname " << hostname << " set via COVISE_HOST failed" << std::endl;
    }

    char buf[4096];
    gethostname(buf, 4096);
    buf[sizeof(buf) - 1] = '\0';
    HostSymbolic(buf);
    if (!getAddress() || !hasRoutableAddress() || !isAddressConfigured(char_address))
    {
        if (findPrimaryIpAddress(char_address))
        {
            char numeric[100];
            snprintf(numeric, sizeof(numeric), "%d.%d.%d.%d",
                     (int)char_address[0],
                     (int)char_address[1],
                     (int)char_address[2],
                     (int)char_address[3]);
            setAddress(numeric);
#ifdef DEBUG
            std::cerr << "primary address determined to be " << numeric << std::endl;
#endif
        }
    }
    if (!getAddress())
        HostSymbolic("localhost");
    if (!getAddress())
        HostSymbolic("127.0.0.1");
    if (!getAddress())
    {
        std::cerr << "failed to find a useful local hostname" << std::endl;
        exit(1);
    }
    setName(buf);
}

bool Host::hasValidName() const
{
    return m_nameValid;
}

bool Host::hasValidAddress() const
{
    return m_addressValid;
}
