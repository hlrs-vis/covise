/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS UDP_Sender
//
// This class @@@
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#ifdef _WIN32
#include <ws2tcpip.h>
#endif
#include "UDP_Sender.h"

#ifdef WIN32
#include <windows.h>
typedef unsigned long in_addr_t;
#else
// network stuff
#include <sys/socket.h>
#include <netdb.h>

#include <netinet/in.h>
#include <arpa/inet.h>

#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#endif
#ifdef __sgi
#include <strings.h>
#else
#include <string.h>
#endif

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

UDP_Sender::UDP_Sender(const char *hostname, int port, const char *mcastif, int mcastttl)
{
    // call common set-up
    setup(hostname, port, mcastif, mcastttl);
}

UDP_Sender::UDP_Sender(const char *hostPort, const char *mcastif, int mcastttl)
{
    // copy to a fixed bufer - prevent possible overrun
    char hostname[1024];
    hostname[1023] = '\0';
    strncpy(hostname, hostPort, 1023);

    // grep out port part
    char *dpoint = strchr(hostname, ':');
    if (!dpoint)
    {
        strcpy(d_error, "Target must be hostname:port");
        return;
    }

    // convert to int
    int port;
    int retval;
    retval = sscanf(dpoint + 1, "%d", &port);
    if (retval != 1)
    {
        std::cerr << "UDP_Sender::UDP_Sender: sscanf failed" << std::endl;
        return;
    }

    // mask away port from copied string - result is host name
    *dpoint = '\0';

    // call common set-up
    setup(hostname, port, mcastif, mcastttl);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Do the complete set-up

void UDP_Sender::setup(const char *hostname, int port,
                       const char *mcastif, int mcastttl)
{
    // no errors yet
    d_error[0] = '\0';

    // port numbers - must be correct
    if (port <= 0 || port > 65535) //ports range extended, otherwise A.R.T default port 50105 not possible
    {
        strcpy(d_error, "Port number out of range [0..65535]");
        return;
    }
#ifdef _WIN32
    int err;
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(1, 1);
    err = WSAStartup(wVersionRequested, &wsaData);
#endif
    // Create the socket
    d_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (d_socket < 0)
    {
        sprintf(d_error, "Error opening UDP port: %s", strerror(errno));
        return;
    }

    // convert hostname to IP number
    unsigned long toAddr = getIP(hostname);

    // build up adress field
    memset((void *)&d_address, 0, sizeof(d_address));
    d_address.sin_family = AF_INET;
    d_address.sin_port = htons(port);
    d_address.sin_addr.s_addr = toAddr;

    if (mcastttl != -1)
    {
        u_char ttl = mcastttl;
#ifdef _WIN32
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_TTL, (const char *)&ttl, sizeof(ttl)))
#else
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)))
#endif
        {
            sprintf(d_error, "Error setting multicast TTL: %s", strerror(errno));
            return;
        }
    }

    if (mcastif)
    {
		struct in_addr v4;
		int err = inet_pton(AF_INET, mcastif, &v4);

        if (v4.s_addr == INADDR_NONE)
        {
            sprintf(d_error, "Error converting multicast interface address %s: %s",
                    mcastif, strerror(errno));
            return;
        }
        
#ifdef _WIN32
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&v4, sizeof(v4)))
#else
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_IF, &v4, sizeof(v4)))
#endif
        {
            sprintf(d_error, "Error setting multicast interface %s: %s",
                    mcastif, strerror(errno));
            return;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

UDP_Sender::~UDP_Sender()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// send a \0-terminated string
int UDP_Sender::send(const char *string)
{
    // send string including terminating \0
    if (string)
        return send(string, (int)strlen(string) + 1);
    else
        return (0);
}

// send a UDP packet
int UDP_Sender::send(const void *buffer, int numBytes)
{
    int nbytes;
#ifdef __linux__
    socklen_t toLen;
#else
    int toLen;
#endif
    toLen = sizeof(struct sockaddr_in);
    nbytes = sendto(d_socket, (const char *)buffer, numBytes, 0,
                    (struct sockaddr *)(void *)&d_address, toLen);
    if (nbytes < 0)
    {
        sprintf(d_error, "Error sending UDP package: %s", strerror(errno));
    }
    return nbytes;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// is this serner ok?
bool UDP_Sender::isBad()
{
    return d_error[0] != '\0'; // ok, if no error messages
}

// get error string of last error
const char *UDP_Sender::errorMessage()
{
    return d_error;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
unsigned long
UDP_Sender::getIP(const char *hostname)
{
    // try dot notation

	struct in_addr v4;
	int err = inet_pton(AF_INET, hostname, &v4);
    if (v4.s_addr != INADDR_NONE)
        return (v4.s_addr);

	// try nameserver
	struct addrinfo hints, *result = NULL;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family = AF_INET;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = 0; /* any type of socket */
	hints.ai_flags = AI_ADDRCONFIG;
	hints.ai_protocol = 0;          /* Any protocol */

	char *cPtr = (char *)&v4.s_addr;

	int s = getaddrinfo(hostname, NULL /* service */, &hints, &result);
	if (s != 0)
	{
		fprintf(stderr, "Host::HostSymbolic: getaddrinfo failed for %s: %s\n", hostname, gai_strerror(s));
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
			memcpy(cPtr, &saddr->sin_addr, sizeof(cPtr));
			if (!inet_ntop(rp->ai_family, &saddr->sin_addr, address, sizeof(address)))
			{
				std::cerr << "could not convert address of " << hostname << " to printable format: " << strerror(errno) << std::endl;
				continue;
			}
			else
			{
				memcpy(cPtr, &saddr->sin_addr, sizeof(cPtr));
				return v4.s_addr;
			}
		}

		freeaddrinfo(result);           /* No longer needed */
	}
    return INADDR_NONE;
}
