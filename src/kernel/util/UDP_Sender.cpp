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
        in_addr_t ifaddr = inet_addr(mcastif);

        if (ifaddr == INADDR_NONE)
        {
            sprintf(d_error, "Error converting multicast interface address %s: %s",
                    mcastif, strerror(errno));
            return;
        }
        struct in_addr inaddr;
        inaddr.s_addr = ifaddr;
#ifdef _WIN32
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&inaddr, sizeof(inaddr)))
#else
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_IF, &inaddr, sizeof(inaddr)))
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
    unsigned int binAddr;
    binAddr = inet_addr(hostname);
    if (binAddr != INADDR_NONE)
        return binAddr;

    // try nameserver
    struct hostent *hostRecord = gethostbyname(hostname);
    if (hostRecord == NULL)
        return INADDR_NONE;

    // analyse 1st result
    if (hostRecord->h_addrtype == AF_INET)
    {
        char *cPtr = (char *)&binAddr;
        cPtr[0] = *hostRecord->h_addr_list[0];
        cPtr[1] = *(hostRecord->h_addr_list[0] + 1);
        cPtr[2] = *(hostRecord->h_addr_list[0] + 2);
        cPtr[3] = *(hostRecord->h_addr_list[0] + 3);
        return binAddr;
    }
    else
        return INADDR_NONE;
}
