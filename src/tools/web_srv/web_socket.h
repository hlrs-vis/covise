/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_SOCKET_H
#define EC_SOCKET_H

#undef _POSIX_SOURCE

#define MULTICAST

#include <sys/types.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/socket.h>
#ifndef _SX
extern "C" {
#include <netinet/in.h>
}

#else
#include <sys/socket.h>
#include <sys/select.h>
#endif
#include <netdb.h>
#endif
#include <string.h>
#include <covise/covise_host.h>
#include <config/CoviseConfig.h>

#ifdef MULTICAST
#include <iostream>
#include <stdio.h>
#if !defined _WIN32 && !defined __GNUC__ && !defined __hpux && !defined __sun
#include <bstring.h>
#include <unistd.h>
#endif
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <netdb.h>
#include <arpa/inet.h>
#endif
#endif
/*
 $Log:  $
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

class ServerConnection;

class Socket
{
protected:
    static int stport;
    static char **ip_alias_list;
    static Host **host_alias_list;
    struct sockaddr_in s_addr_in;
    Host *host; // host on the other end of the socket
    int sock_id;
    int port;

public:
    Socket(Host *h, int p, int retries = 20); // connect as client
    Socket(int p); // initiate as server
    Socket(int *p); // initiate as server and use free port
    Socket() // initialize with descriptor
    {
        sock_id = -1;
        host = NULL;
    };
    Socket(int, int sfd); // initialize with descriptor
    Socket(const Socket &); // initiate as server and use free port
    virtual ~Socket(); // NIL
    static void set_start_port(int stp)
    {
        stport = stp;
    };
    int get_start_port()
    {
        return stport;
    };
    Socket *copy_and_accept(void);
    void listen(); // listen for actual connection (server)
    void accept(); // wait for and accept a connection (server)
    int accept(int); // wait for and accept a connection (server) wait max secondes
    // returns -1 when no accept, 0 otherwise
    virtual int read(void *buf, unsigned nbyte);
    int Read(void *buf, unsigned nbyte); // does not exit when read failes but returns -1
    int read_non_blocking(void *buf, unsigned nbyte);
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
    char *get_hostname()
    {
        return (char *)host->get_name();
    };
    void print()
    {
#ifdef DEBUG
        cerr << "Socket id: " << sock_id << "  Port: " << port << endl;
        host->print();
#endif
    }
};

#ifdef MULTICAST
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
#endif
