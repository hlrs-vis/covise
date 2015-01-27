/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WCE_SOCKET_H
#define WCE_SOCKET_H

#include <iostream>

#include <math.h>
#include <netinet/in.h>
#include <errno.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <util/coExport.h>
#ifdef _WIN32
#include <windows.h>
#include <time.h>
#endif
#define LOGINFO(x)
#define LOGERROR(x)
#include <wce_connect.h>

namespace covise
{
extern void coPerror(const char *prefix);
#define HAVEMULTICAST

class Host;

/*
 $Log: covise_socket.h,v $
 * Revision 1.3  1993/10/08  19:10:08  zrhk0125
 * modifications for correct conversion handling
 *
 * Revision 1.2  93/09/30  17:08:08  zrhk0125
 * basic modifications for CRAY
 *
 * Revision 1.1  93/09/25  20:51:06  zrhk0125
 * Initial revision
 *
*/

/***********************************************************************\
 **                                                                     **
 **   Socket Classes                              Version: 1.0          **
 **                                                                     **
 **                                                                     **
 **   Description  : The Socket class handles the operating system      **
 **		    part of the socket handling.                       **
 **                                                                     **
 **   Classes      : Socket                                             **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

const char DF_NONE = 0;
const char DF_IEEE = 1;
const char DF_CRAY = 2;
const int COVISE_SOCKET_INVALID = -2;

#if defined(CRAY) && !defined(_WIN32)
const char df_local_machine = DF_CRAY;
#else
const char df_local_machine = DF_IEEE;
#endif

class FirewallConfig
{
    static FirewallConfig *theFirewallConfig;
    FirewallConfig();
    ~FirewallConfig();

public:
    int sourcePort;
    int destinationPort;
    bool setSourcePort;
    static FirewallConfig *the();
};

class ServerConnection;
class SimpleServerConnection;
class SSLServerConnection;

class Socket
{
protected:
    static int stport;
    static char **ip_alias_list;
    static Host **host_alias_list;
    static bool bInitialised;
    struct sockaddr_in s_addr_in;
    Host *host; // host on the other end of the socket
    int sock_id;
    int port;
    int setTCPOptions();
    bool connected;

public:
    // connect as client
    Socket(Host *h, int p, int retries = 20, double timeout = 0.0);
    Socket(int p); // initiate as server
    Socket(int *p); // initiate as server and use free port
    Socket() // initialize with descriptor
    {
        sock_id = -1;
        host = NULL;
    };
    Socket(int, int sfd); // initialize with descriptor
    Socket(const Socket &); // initiate as server and use free port
    Socket(int socket_id, sockaddr_in *sockaddr);
    virtual ~Socket(); // NIL
    static void initialize();
    static void uninitialize();
    static void set_start_port(int stp)
    {
        stport = stp;
    };
    int get_start_port()
    {
        return stport;
    };
    ServerConnection *copy_and_accept(); // after select/accept
    // after select/accept
    SimpleServerConnection *copySimpleAndAccept();
    int listen(); // listen for actual connection (server)
    virtual int accept(); // wait for and accept a connection (server)
    int accept(int); // wait for and accept a connection (server) wait max secondes
    // returns -1 when no accept, 0 otherwise
    virtual int read(void *buf, unsigned nbyte);
    virtual int Read(void *buf, unsigned nbyte); // does not exit when read failes but returns -1

    int setNonBlocking(bool on);
    //int read_non_blocking(void *buf, unsigned nbyte);
    virtual int write(const void *buf, unsigned nbyte);
#ifdef CRAY
    int writea(const void *buf, unsigned nbyte);
#endif
    int get_id()
    {
        return sock_id;
    };
    int get_port()
    {
        return port;
    };
    Host *get_ip_alias(Host *);
    Host *get_host()
    {
        return host;
    };
    const char *get_hostname();
    void print();
#ifdef WIN32
    static int getErrno()
    {
        return WSAGetLastError();
    };
#else
    static int getErrno()
    {
        return errno;
    };
#endif

    int available(void);
    bool isConnected();
    static const char *coStrerror(int err);
};

class UDPSocket : public Socket
{
public:
    UDPSocket(char *address, int p);
    ~UDPSocket();
    int read(void *buf, unsigned nbyte);
    int write(const void *buf, unsigned nbyte);
};

#ifdef HAVEMULTICAST
class MulticastSocket : public Socket
{
    int ttl;

public:
    MulticastSocket(char *MulticastGroup, int p, int ttl);
    ~MulticastSocket();
    int read(void *buf, unsigned nbyte);
    int write(const void *buf, unsigned nbyte);
    int get_ttl()
    {
        return ttl;
    };
};

#endif

void resolveError();
}

#endif
