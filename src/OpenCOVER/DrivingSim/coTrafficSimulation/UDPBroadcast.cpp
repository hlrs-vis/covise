/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS UDPBroadcast
//
// This class @@@
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "UDPBroadcast.h"

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
bool UDPBroadcast::error_TS = true;
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructorsq
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

UDPBroadcast::UDPBroadcast(const char *hostname, int port, int localPort, const char *mcastif, int mcastttl, bool broadcast)
{
    setError_TS(true);
    msgSize = 0;
    currMsgPtr = msgBuf;
    // call common set-up
    setup(hostname, port, localPort, mcastif, mcastttl, broadcast);
}

UDPBroadcast::UDPBroadcast(const char *hostPort, int localPort, const char *mcastif, int mcastttl, bool broadcast)
{
    setError_TS(true);
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
        std::cerr << "UDPBroadcast::UDPBroadcast: sscanf failed" << std::endl;
        return;
    }

    // mask away port from copied string - result is host name
    *dpoint = '\0';

    // call common set-up
    setup(hostname, port, localPort, mcastif, mcastttl, broadcast);
}

// get a UDP port to receive binary dara
int UDPBroadcast::openReceivePort(int portnumber)
{
    std::cout << "openReceivePort" << std::endl;

    // CREATING UDP SOCKET
    d_rsocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (d_rsocket < 0)
    {
        std::cerr << "+VRC+ socket creation failed" << std::endl;
        //       fprintf(stderr, "+VRC+ socket creation failed\n");
        return -1;
    }

    //    fprintf(stderr,"portnumber: %d\n",portnumber);

    // FILL SOCKET ADRESS STRUCTURE
    sockaddr_in any_adr;

    memset((char *)&any_adr, 0, sizeof(any_adr));
    any_adr.sin_family = AF_INET;
    any_adr.sin_port = htons(portnumber);
    any_adr.sin_addr.s_addr = htonl(INADDR_ANY); // accept data from anyone

    // BIND TO A LOCAL PROTOCOL PORT
    std::cout << "UDPBroadcast:: d_rsocket " << d_rsocket << std::endl;
    //       fprintf(stderr, "UDPBroadcast:: d_rsocket %d\n",d_rsocket);
    if (bind(d_rsocket, (sockaddr *)&any_adr, sizeof(any_adr)) < 0)
    {
        std::cerr << "UDPBroadcast:: Could not bind to port " << portnumber << std::endl;
        //       fprintf(stderr, "UDPBroadcast:: Could not bind to port %d\n",portnumber);
        return -1;
    }

    return d_rsocket;
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Do the complete set-up

void UDPBroadcast::setup(const char *hostname, int port, int localPort,
                         const char *mcastif, int mcastttl, bool broadcast)
{
    // no errors yet
    d_error[0] = '\0';

    // port numbers - must be correct
    if (port <= 0 || port > 332767)
    {
        std::cerr << "Port number out of range [0..332767]" << std::endl;
        //       fprintf(stderr,"Port number out of range [0..32767]");
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
        std::cerr << "Error opening local port: " << localPort << " " << strerror(errno) << std::endl;
        //       fprintf(stderr,"Error opening local port: %d %s",localPort,strerror(errno));
        return;
    }
    // Create the socket
    d_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (d_socket < 0)
    {
        std::cerr << "Error creating socket port: " << localPort << " " << strerror(errno) << std::endl;
        //       fprintf(stderr,"Error creating socket port: %s",strerror(errno));
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

    if (broadcast)
    {
        int wahr = 1;
        if (-1 == setsockopt(d_socket, SOL_SOCKET, SO_BROADCAST, (char *)&wahr, sizeof(int)))
            std::cout << "Error setting broadcast: " << strerror(errno) << std::endl;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

UDPBroadcast::~UDPBroadcast()
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
int UDPBroadcast::send(const char *string)
{
    // send string including terminating \0
    if (string)
        return send(string, strlen(string) + 1);
    else
        return (0);
}

// send a UDP packet
int UDPBroadcast::send(const void *buffer, int numBytes)
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
        //		std::cout << "Error sending UDP package: " << strerror(errno) << std::endl;
        //       sprintf(d_error,"Error sending UDP package: %s",strerror(errno));
    }
    return nbytes;
}

int UDPBroadcast::receive(void *buffer, int numBytes)
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
        if (UDPBroadcast::getError_TS())
            cerr << "No data received from Realtime Engine (receive - TSP)" << endl;
        return -1;
    }

    //fprintf(stderr,"UDPBroadcast: reading %d  socket: %d\n",numBytes,d_rsocket);
    // receive a package
    int numbytes = recvfrom(d_rsocket, (char *)buffer, numBytes, 0, 0, 0);

    if (numbytes != numBytes)
    {
        fprintf(stderr, "UDPBroadcast: Message to short, expected %d but got %d, socket: %d\n", numBytes, numbytes, d_rsocket);
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
bool UDPBroadcast::isBad()
{
    return d_error[0] != '\0'; // ok, if no error messages
}

// get error string of last error
const char *UDPBroadcast::errorMessage()
{
    return d_error;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
unsigned long
UDPBroadcast::getIP(const char *hostname)
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
UDPBroadcast::UDPBroadcast(const UDPBroadcast &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
UDPBroadcast &UDPBroadcast::operator=(const UDPBroadcast &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
UDPBroadcast::UDPBroadcast()
{
    assert(0);
}

int UDPBroadcast::readMessage()
{

    // check wether we already received package
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(d_rsocket, &readfds);
    struct timeval timeout = { 2, 0 };
    if (0 == select(d_rsocket + 1, &readfds, NULL, NULL, &timeout))
    {
        if (UDPBroadcast::getError_TS())
            cerr << "No radar cones received from Realtime Engine (read - TSP)" << endl;
        return -1;
    }

    //sockaddr remote_adr;
    //fprintf(stderr,"UDPBroadcast: reading %d  socket: %d\n",numBytes,d_rsocket);
    // receive a package
    msgSize = recvfrom(d_rsocket, msgBuf, UDP_COMM_MAX_SIZE, 0, 0, 0);
    currMsgPtr = msgBuf;
    if (msgSize < 0)
    {
        fprintf(stderr, "UDPBroadcast: Message to shortgot %d, socket: %d\n", msgSize, d_rsocket);
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
        fprintf(stderr, "UDPBroadcast: Message %d, socket: %d\n", msgSize, d_rsocket);
        return msgSize;
    }
    return msgSize;
}

int UDPBroadcast::getMessagePart(void *buf, int size)
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
            int left = msgBuf + msgSize - currMsgPtr;
            memcpy(buf, currMsgPtr, left);
            currMsgPtr += left;
            return left;
        }
    }
    return -1;
}

void UDPBroadcast::errorStatus_TS()
{
    if (UDPBroadcast::getError_TS())
        UDPBroadcast::setError_TS(false);
    else
        UDPBroadcast::setError_TS(true);
}

bool UDPBroadcast::getError_TS()
{
    return UDPBroadcast::error_TS;
}

void UDPBroadcast::setError_TS(bool status)
{
    UDPBroadcast::error_TS = status;
}
