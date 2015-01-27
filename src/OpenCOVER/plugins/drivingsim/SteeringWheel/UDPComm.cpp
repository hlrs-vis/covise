/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS UDPComm
//
// This class @@@
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#include "UDPComm.h"

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
#ifndef _WIN32
#define closesocket close
#endif

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bool UDPComm::error_SW = true;
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructorsq
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

UDPComm::UDPComm(const char *hostname, int port, int localPort, const char *mcastif, int mcastttl)
{
    msgSize = 0;
    currMsgPtr = msgBuf;
    // call common set-up
    setup(hostname, port, localPort, mcastif, mcastttl);
}

UDPComm::UDPComm(int localPort)
{
    msgSize = 0;
    currMsgPtr = msgBuf;
    d_socket = 0;
#ifdef _WIN32
    int err;
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(1, 1);
    err = WSAStartup(wVersionRequested, &wsaData);
#endif
    openReceivePort(localPort);
}

UDPComm::UDPComm(int port, const char *hostname)
{
    msgSize = 0;
    currMsgPtr = msgBuf;
    // no errors yet
    d_error[0] = '\0';

    // port numbers - must be correct
    if (port <= 0 || port > 332767)
    {
        fprintf(stderr, "Port number out of range [0..32767]");
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
    d_socket = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (d_socket < 0)
    {
        fprintf(stderr, "Error creating socket port: %s", strerror(errno));
        return;
    }
    // convert hostname to IP number
    unsigned long toAddr = getIP(hostname);

    // build up adress field
    memset((void *)&d_address, 0, sizeof(d_address));
    d_address.sin_family = AF_INET;
    d_address.sin_port = htons(port);
    d_address.sin_addr.s_addr = toAddr;
}

UDPComm::UDPComm(const char *hostPort, int localPort, const char *mcastif, int mcastttl)
{
    msgSize = 0;
    currMsgPtr = msgBuf;
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
        std::cerr << "UDPComm::UDPComm: sscanf failed" << std::endl;
        return;
    }

    // mask away port from copied string - result is host name
    *dpoint = '\0';

    // call common set-up
    setup(hostname, port, localPort, mcastif, mcastttl);
}

// get a UDP port to receive binary dara
int UDPComm::openReceivePort(int portnumber)
{

    // CREATING UDP SOCKET
    d_rsocket = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (d_rsocket < 0)
    {
        fprintf(stderr, "+VRC+ socket creation failed\n");
        return -1;
    }

    fprintf(stderr, "portnumber: %d\n", portnumber);

    // FILL SOCKET ADRESS STRUCTURE
    sockaddr_in any_adr;

    memset((char *)&any_adr, 0, sizeof(any_adr));
    any_adr.sin_family = AF_INET;
    any_adr.sin_port = htons(portnumber);
    any_adr.sin_addr.s_addr = htonl(INADDR_ANY); // accept data from anyone

    // BIND TO A LOCAL PROTOCOL PORT
    fprintf(stderr, "UDPComm:: d_rsocket %d\n", d_rsocket);
    if (bind(d_rsocket, (sockaddr *)&any_adr, sizeof(any_adr)) < 0)
    {
        fprintf(stderr, "UDPComm:: Could not bind to port %d\n", portnumber);
        return -1;
    }

    return d_rsocket;
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Do the complete set-up

void UDPComm::setup(const char *hostname, int port, int localPort,
                    const char *mcastif, int mcastttl)
{
    // no errors yet
    d_error[0] = '\0';

    // port numbers - must be correct
    if (port <= 0 || port > 332767)
    {
        fprintf(stderr, "Port number out of range [0..32767]");
        return;
    }
#ifdef _WIN32
    int err;
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(1, 1);
    err = WSAStartup(wVersionRequested, &wsaData);
#endif
    if (openReceivePort(localPort) < 0)
    {
        fprintf(stderr, "Error opening local port: %d %s", localPort, strerror(errno));
        return;
    }
    // Create the socket
    d_socket = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (d_socket < 0)
    {
        fprintf(stderr, "Error creating socket port: %s", strerror(errno));
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
#ifdef WIN32
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_TTL, (const char *)&ttl, sizeof(ttl)))
#else
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)))
#endif
        {
            fprintf(stderr, "Error setting multicast TTL: %s", strerror(errno));
            return;
        }
    }

    if (mcastif)
    {
        in_addr_t ifaddr = inet_addr(mcastif);

        if (ifaddr == INADDR_NONE)
        {
            fprintf(stderr, "Error converting multicast interface address %s: %s",
                    mcastif, strerror(errno));
            return;
        }
        struct in_addr inaddr = { ifaddr };
#ifdef WIN32
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&inaddr, sizeof(inaddr)))
#else
        if (-1 == setsockopt(d_socket, IPPROTO_IP, IP_MULTICAST_IF, &inaddr, sizeof(inaddr)))
#endif
        {
            fprintf(stderr, "Error setting multicast interface %s: %s",
                    mcastif, strerror(errno));
            return;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

UDPComm::~UDPComm()
{
    if (d_socket > 0)
    {
        closesocket(d_socket);
    }
    if (d_rsocket > 0)
    {
        closesocket(d_rsocket);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// send a \0-terminated string
int UDPComm::send(const char *string)
{
    // send string including terminating \0
    if (string)
        return send(string, (int)strlen(string) + 1);
    else
        return (0);
}

// send a UDP packet
int UDPComm::send(const void *buffer, int numBytes)
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

int UDPComm::receive(void *buffer, int numBytes)
{
    /*sockaddr remote_adr;
   socklen_t rlen;*/

    // check wether we already received package
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(d_rsocket, &readfds);
    struct timeval timeout = { 2, 0 };
    if (0 == select(d_rsocket + 1, &readfds, NULL, NULL, &timeout))
    {
        if (UDPComm::getError_SW())
            cerr << "No data received from Realtime Engine (receive - SW)" << endl;
        return -1;
    }

    //fprintf(stderr,"UDPComm: reading %d  socket: %d\n",numBytes,d_rsocket);
    // receive a package
    int numbytes = recvfrom(d_rsocket, (char *)buffer, numBytes, 0, 0, 0);

    if (numbytes != numBytes)
    {
        fprintf(stderr, "UDPComm: Message to short, expected %d but got %d, socket: %d\n", numBytes, numbytes, d_rsocket);
#ifdef WIN32
        if (numbytes < 0)
        {
            int numToRead = 0;
            numToRead = recvfrom(d_rsocket, (char *)buffer, numBytes, MSG_PEEK, 0, 0);

            fprintf(stderr, "Errno: %d MSG_PEER:%d\n", WSAGetLastError(), numToRead);
            //DebugBreak();
        }
#endif
        return numbytes;
    }
    return numbytes;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// is this serner ok?
bool UDPComm::isBad()
{
    return d_error[0] != '\0'; // ok, if no error messages
}

// get error string of last error
const char *UDPComm::errorMessage()
{
    return d_error;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
unsigned long
UDPComm::getIP(const char *hostname)
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

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
UDPComm::UDPComm(const UDPComm &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
UDPComm &UDPComm::operator=(const UDPComm &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
UDPComm::UDPComm()
{
    assert(0);
}

int UDPComm::readMessage()
{

    // check wether we already received package
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(d_rsocket, &readfds);
    struct timeval timeout = { 2, 0 };
    if (0 == select(d_rsocket + 1, &readfds, NULL, NULL, &timeout))
    {
        if (UDPComm::getError_SW())
            cerr << "No data received from Realtime Engine (read - SWs)" << endl;
        return -1;
    }

    //sockaddr remote_adr;
    //fprintf(stderr,"UDPComm: reading %d  socket: %d\n",numBytes,d_rsocket);
    // receive a package
    msgSize = recvfrom(d_rsocket, msgBuf, UDP_COMM_MAX_SIZE, 0, 0, 0);
    currMsgPtr = msgBuf;
    if (msgSize < 0)
    {
        fprintf(stderr, "UDPComm: Message to shortgot %d, socket: %d\n", msgSize, d_rsocket);
        perror("errno");
#ifdef WIN32
        //sockaddr remote_adr;
        if (msgSize < 0)
        {
            // numToRead not used: generates C4189; "int numToRead = "
            recvfrom(d_rsocket, msgBuf, UDP_COMM_MAX_SIZE, MSG_PEEK, 0, 0);

            fprintf(stderr, "Errno: %d MSG_PEER:%d\n", WSAGetLastError(), msgSize);
        }
#endif
        fprintf(stderr, "UDPComm: Message %d, socket: %d\n", msgSize, d_rsocket);
        return msgSize;
    }
    return msgSize;
}

int UDPComm::getMessagePart(void *buf, int size)
{
    if (currMsgPtr + size < msgBuf + msgSize)
    {
        memcpy(buf, currMsgPtr, size);
        currMsgPtr += size;
        return size;
    }
    else
    {
        if (currMsgPtr < msgBuf + msgSize)
        {
            int left = (int)(msgBuf + msgSize - currMsgPtr);
            memcpy(buf, currMsgPtr, left);
            currMsgPtr += left;
            return left;
        }
    }
    return -1;
}

void UDPComm::errorStatus_SW()
{
    if (UDPComm::getError_SW())
        UDPComm::setError_SW(false);
    else
        UDPComm::setError_SW(true);
}

bool UDPComm::getError_SW()
{
    return UDPComm::error_SW;
}

void UDPComm::setError_SW(bool status)
{
    UDPComm::error_SW = status;
}
