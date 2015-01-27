/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef __alpha
#include "externc_alpha.h"
#endif

#include "web_socket.h"
#include <covise/covise_global.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#define sleep(x)
#include <iostream.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <iostream>
#include <sys/time.h>
#endif

#ifdef CRAY
#include <sys/iosw.h>
#include <sys/param.h>
#include <signal.h>
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#ifdef PARAGON
#include "externc.h"
#endif
#ifdef _AIX
#include <sys/select.h>
#include <strings.h>
#include "externc_aix.h"
#endif

#ifdef _WIN32
#undef CRAY
#endif
/*
 $Log:  $
Revision 1.6  1994/03/23  18:07:06  zrf30125
Modifications for multiple Shared Memory segments have been finished
(not yet for Cray)

Revision 1.5  93/11/25  16:44:21  zrfg0125
TCP_NODELAY for ALL sockets added

Revision 1.5  93/11/25  16:42:59  zrhk0125
TCP_NODELAY added for all sockets

Revision 1.4  93/10/20  15:14:18  zrhk0125
socket cleanup improved

Revision 1.2  93/09/30  17:08:17  zrhk0125
basic modifications for CRAY

Revision 1.1  93/09/25  20:50:47  zrhk0125
Initial revision

*/
#undef DEBUG
/***********************************************************************\ 
 **                                                                     **
 **   Socket Class Routines                       Version: 1.0          **
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
 **                  26.05.93      TCP_NODELAY inserted                 **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#ifdef __hpux
#define MAX_SOCK_BUF 58254
#else
#define MAX_SOCK_BUF 65536
#endif
#ifndef CRAY
extern "C" void herror(const char *string);
#endif

#if defined(CRAY)
extern "C" int write(int fildes, const void *buf, unsigned nbyte);
extern "C" int read(int fildes, void *buf, unsigned nbyte);
extern "C" void (*signal(int sig, void (*func)(int)))(int);
extern "C" int recalla(long mask[RECALL_SIZEOF]);
extern "C" int writea(int fildes, char *buf, unsigned nbyte, struct iosw *status, int signo);
#endif

#ifndef _WIN32
extern "C" int shutdown(int s, int how);
extern "C" char *inet_ntoa(struct in_addr in);
#endif
extern "C" int close(int fildes);

#undef DEBUG

Socket::Socket(int, int sfd)
{
    sock_id = sfd;
    host = 0L;
    ip_alias_list = 0L;
    host_alias_list = 0L;
    port = 0;
}

Socket::Socket(Host *h, int p, int retries)
{
    int on;
    struct linger linger;
    int sendbuf, recvbuf;
#ifdef CRAY
    char *hostname;
    int len, winshift;
#endif
    port = p;
    host = get_ip_alias(h);
    //    cerr << "connecting to host " << host->get_name() << " on port " << port << endl;

    sock_id = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        perror("error with socket");
        print_comment(__LINE__, __FILE__, "error with socket");
    }

    on = 1;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_NODELAY,
                   (char *)&on, sizeof(on)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_NODELAY");
    linger.l_onoff = 1;
    linger.l_linger = 60;
    if (setsockopt(sock_id, SOL_SOCKET, SO_LINGER,
                   (char *)&linger, sizeof(linger)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_LINGER");
#ifdef CRAY
    hostname = (unsigned char *)h->get_name();
    len = strlen(hostname);
    if (strncmp(&hostname[len - 3], "-hi", 3))
    {
        winshift = 4;
        if (setsockopt(sock_id, IPPROTO_TCP, TCP_WINSHIFT,
                       &winshift, sizeof(winshift)) < 0)
            print_comment(__LINE__, __FILE__, "setsockopt TCP_RFC1323");
        sendbuf = 6 * 65536;
        recvbuf = 6 * 65536;
        print_comment(__LINE__, __FILE__, "HIPPI interface");
    }
    else
    {
#endif
        sendbuf = MAX_SOCK_BUF;
        recvbuf = MAX_SOCK_BUF;
#ifdef CRAY
        print_comment(__LINE__, __FILE__, "no HIPPI interface");
    }
#endif
    if (setsockopt(sock_id, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_SNDBUF");
    if (setsockopt(sock_id, SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_RCVBUF");

#if defined CRAY || defined PARAGON || defined __alpha || defined _AIX
    //        memcpy ((char *) &s_addr_in.sin_addr, hp->h_addr, hp->h_length);
    host->get_char_address((unsigned char *)&s_addr_in.sin_addr);
#else
    //        memcpy (&(s_addr_in.sin_addr.s_addr), hp->h_addr, hp->h_length);
    host->get_char_address((unsigned char *)&(s_addr_in.sin_addr.s_addr));
#endif

    s_addr_in.sin_port = htons(port);
    s_addr_in.sin_family = AF_INET;

    //    if (connect(sock_id, (sockaddr *)&s_addr_in,sizeof(s_addr_in)) < 0) {
    //        close(sock_id);
    //        perror("Connect to server");
    //    	print_comment(__LINE__, __FILE__, "Connect to server error");
    //    	print_comment(__LINE__, __FILE__, host->get_name());
    //        print_exit(__LINE__, __FILE__, 1);
    //    }
    int numconn = 0;
    while (connect(sock_id, (sockaddr *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
        perror("connect failed");
        close(sock_id);
        if (numconn >= retries)
        {
            sock_id = -1;
            return;
        }
        numconn++;
        sleep(1);
        fprintf(stderr, "Try to reconnect %d\n", numconn);
        sock_id = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_id < 0)
        {
            perror("error 2 with socket");
            print_comment(__LINE__, __FILE__, "error 2 with socket");
        }

        on = 1;
        if (setsockopt(sock_id, IPPROTO_TCP, TCP_NODELAY,
                       (char *)&on, sizeof(on)) < 0)
            cerr << "setsockopt2 error\n";
        linger.l_onoff = 1;
        linger.l_linger = 60;
        if (setsockopt(sock_id, SOL_SOCKET, SO_LINGER,
                       (char *)&linger, sizeof(linger)) < 0)
            print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_LINGER");
#ifdef CRAY
        hostname = (unsigned char *)h->get_name();
        len = strlen(hostname);
        if (strncmp(&hostname[len - 3], "-hi", 3))
        {
            winshift = 4;
            if (setsockopt(sock_id, IPPROTO_TCP, TCP_WINSHIFT,
                           &winshift, sizeof(winshift)) < 0)
                print_comment(__LINE__, __FILE__, "setsockopt IPPROTO_TCP TCP_WINSHIFT");
            sendbuf = 6 * 65536;
            recvbuf = 6 * 65536;
            print_comment(__LINE__, __FILE__, "HIPPI interface");
        }
        else
        {
#endif
            sendbuf = MAX_SOCK_BUF;
            recvbuf = MAX_SOCK_BUF;
#ifdef CRAY
            print_comment(__LINE__, __FILE__, "no HIPPI interface");
        }
#endif
        if (setsockopt(sock_id, SOL_SOCKET, SO_SNDBUF,
                       (char *)&sendbuf, sizeof(sendbuf)) < 0)
            print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_SNDBUF");
        if (setsockopt(sock_id, SOL_SOCKET, SO_RCVBUF,
                       (char *)&recvbuf, sizeof(recvbuf)) < 0)
            print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_RCVBUF");

#if defined CRAY || defined PARAGON || defined __alpha || defined _AIX
        //        memcpy ((char *) &s_addr_in.sin_addr, hp->h_addr, hp->h_length);
        host->get_char_address((unsigned char *)&s_addr_in.sin_addr);
#else
        //        memcpy (&(s_addr_in.sin_addr.s_addr), hp->h_addr, hp->h_length);
        host->get_char_address((unsigned char *)&(s_addr_in.sin_addr.s_addr));
#endif

        s_addr_in.sin_port = htons(port);
        s_addr_in.sin_family = AF_INET;

        //perror("Connect to server");
        //print_comment(__LINE__, __FILE__, "Connect to server error");
        //print_comment(__LINE__, __FILE__, host->get_name());
    }

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "connected");
#endif
}

Socket::Socket(int p)
{
    int on;
    struct linger linger;
    int sendbuf, recvbuf;
    port = p;
#ifdef CRAY
    char *hostname;
    int len, winshift;
#endif
    host = NULL;
    sock_id = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        perror("socket");
        sock_id = -1;
        return;
        //print_exit(__LINE__, __FILE__, 1);
    }

    on = 1;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_NODELAY,
                   (char *)&on, sizeof(on)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_NODELAY");
    linger.l_onoff = 1;
    linger.l_linger = 60;
#ifdef CRAY
    winshift = 4;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_WINSHIFT,
                   &winshift, sizeof(winshift)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_WINSHIFT");
    sendbuf = 6 * 65536;
    recvbuf = 6 * 65536;
    print_comment(__LINE__, __FILE__, "HIPPI interface");
#else
    sendbuf = MAX_SOCK_BUF;
    recvbuf = MAX_SOCK_BUF;
#endif

    if (setsockopt(sock_id, SOL_SOCKET, SO_LINGER,
                   (char *)&linger, sizeof(linger)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_LINGER");
    if (setsockopt(sock_id, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_SNDBUF");
    if (setsockopt(sock_id, SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_RCVBUF");

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(port);

    /* Assign an address to this socket */

    if (bind(sock_id, (sockaddr *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
        cerr << endl << "ERROR: bind to port : " << s_addr_in.sin_port << " ";
        perror("bind");
        close(sock_id);
        sock_id = -1;
        return;
        //print_exit(__LINE__, __FILE__, 1);
    }
    host = new Host(s_addr_in.sin_addr.s_addr);

#ifdef DEBUG
    print();
#endif
}

Socket::Socket(int *p)
{
    struct linger linger;
    int sendbuf, recvbuf;
    int on;
    host = NULL;
#ifdef CRAY
    char *hostname;
    int len, winshift;
#endif

    sock_id = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        perror("socket");
        return;
        //print_exit(__LINE__, __FILE__, 1);
    }

    on = 1;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_NODELAY,
                   (char *)&on, sizeof(on)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_NODELAY");
    linger.l_onoff = 1;
    linger.l_linger = 60;
    if (setsockopt(sock_id, SOL_SOCKET, SO_LINGER,
                   (char *)&linger, sizeof(linger)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_LINGER");
#ifdef CRAY
    winshift = 4;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_WINSHIFT,
                   &winshift, sizeof(winshift)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_WINSHIFT");
    sendbuf = 6 * 65536;
    recvbuf = 6 * 65536;
    print_comment(__LINE__, __FILE__, "HIPPI interface");
#else
    sendbuf = MAX_SOCK_BUF;
    recvbuf = MAX_SOCK_BUF;
#endif
    if (setsockopt(sock_id, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_SNDBUF");
    if (setsockopt(sock_id, SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
        print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_RCVBUF");

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(stport);

    /* Assign an address to this socket and finds an unused port */

    while (bind(sock_id, (sockaddr *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
#ifndef _WIN32
        if (errno == EADDRINUSE)
        {
#else
        if (GetLastError() == WSAEADDRINUSE)
        {
#endif
            stport++;
            s_addr_in.sin_port = htons(stport);
            //	    cerr << "neuer port: " << stport << "\n";
        }
        else
        {
            perror("bind");
            close(sock_id);
            sock_id = -1;
            return;
            //print_exit(__LINE__, __FILE__, 1);
        }
    }
    *p = port = ntohs(s_addr_in.sin_port);
    stport++;
    host = new Host(s_addr_in.sin_addr.s_addr);
}

Socket::Socket(const Socket &s)
{
    s_addr_in = s.s_addr_in;
    port = s.port;
    sock_id = s.sock_id;
    host = new Host(*(s.host));
}

void Socket::accept()
{
    int tmp_sock_id;
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    ::listen(sock_id, 20);

#if defined(_AIX)
    size_t length;
#elif defined(__linux__)
    socklen_t length;
#else
    int length;
#endif

    length = sizeof(s_addr_in);
    tmp_sock_id = ::accept(sock_id, (sockaddr *)&s_addr_in, &length);

    port = ntohs(s_addr_in.sin_port);

    if (tmp_sock_id < 0)
    {
        perror("accept");
        //print_exit(__LINE__, __FILE__, 1);
    }

#ifdef DEBUG
    printf("Connection from host %s, port %u\n",
           inet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

    //    shutdown(s, 2);
    close(sock_id);
    sock_id = tmp_sock_id;
    host = new Host(s_addr_in.sin_addr.s_addr);
#ifdef DEBUG
    print();
#endif
}

Socket *Socket::copy_and_accept(void)
{
    Socket *tmp_sock;
    tmp_sock = new Socket(*this);

#if defined(_AIX)
    size_t length;
#elif defined(__linux__)
    socklen_t length;
#else
    int length;
#endif

    length = sizeof(s_addr_in);
    tmp_sock->sock_id = ::accept(sock_id, (sockaddr *)&s_addr_in, &length);

    return tmp_sock;
}

void Socket::listen()
{
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    ::listen(sock_id, 20);
}

int Socket::accept(int wait)
{
    int tmp_sock_id;
    struct timeval timeout;
    fd_set fdread;
    int i;
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    ::listen(sock_id, 20);

    timeout.tv_sec = wait;
    timeout.tv_usec = 0;
    FD_ZERO(&fdread);
    FD_SET(sock_id, &fdread);
    if (wait > 0)
    {
#ifdef __hpux9
        i = select(sock_id + 1, (int *)&fdread, 0L, 0L, &timeout);
#else
        i = select(sock_id + 1, &fdread, 0L, 0L, &timeout);
#endif
    }
    else
    {
#ifdef __hpux9
        i = select(sock_id + 1, (int *)&fdread, 0L, 0L, NULL);
#else
        i = select(sock_id + 1, &fdread, 0L, 0L, NULL);
#endif
    }
    if (i == 0)
    {
        return -1;
    }

    socklen_t length = sizeof(s_addr_in);
    tmp_sock_id = ::accept(sock_id, (sockaddr *)&s_addr_in, &length);

    port = ntohs(s_addr_in.sin_port);

    if (tmp_sock_id < 0)
    {
        perror("accept");
        //print_exit(__LINE__, __FILE__, 1);
    }

#ifdef DEBUG
    printf("Connection from host %s, port %u\n",
           inet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

    //    shutdown(s, 2);
    close(sock_id);
    sock_id = tmp_sock_id;
    host = new Host(s_addr_in.sin_addr.s_addr);
#ifdef DEBUG
    print();
#endif

    return 0;
}

Socket::~Socket()
{
    //    shutdown(sock_id, 2);
    //    char tmpstr[100];

    //    sprintf(tmpstr, "delete socket %d", port);
    //    print_comment(__LINE__, __FILE__, tmpstr);
    close(sock_id);
    delete host;
}

//extern CoviseTime *covise_time;

int Socket::write(const void *buf, unsigned nbyte)
{
    int no_written;
    char tmp_str[255];
    char *ptr;
    int no_left;

    no_left = nbyte;
    ptr = (char *)buf;

    while (no_left > 0)
    {
        no_written = ::write(sock_id, ptr, no_left);
        if (no_written <= 0)
        {
            sprintf(tmp_str, "Socket write error = %d, no_of_bytes = %d", errno, no_written);
            print_error(__LINE__, __FILE__, tmp_str);
            if (errno == EINTR)
            {
                no_written = 0; // call ::write again
            }
            else
            {
                if (errno == EPIPE)
                    return COVISE_SOCKET_INVALID;
                print_error(__LINE__, __FILE__, "write returns <= 0: close socket.");
                return COVISE_SOCKET_INVALID;
            }
        }
        no_left -= no_written;
        ptr += no_written;
    }
    return nbyte;
}

#ifdef CRAY
struct iosw wrstat;

int Socket::writea(const void *buf, unsigned nbyte)
{
    long mask[RECALL_SIZEOF];
    void wrhdlr(int signo);
    static int first = 1;
    int ret;

    if (first)
    {
        signal(SIGUSR1, wrhdlr);
        first = 0;
    }

    RECALL_SET(mask, sock_id); /* set bit for fd in mask */
    //    covise_time->mark(__LINE__, "vor Socket::writea");
    ::writea(sock_id, (char *)buf, nbyte, &wrstat, SIGUSR1);
    //    covise_time->mark(__LINE__, "nach Socket::writea");
    ret = recalla(mask);
    //    covise_time->mark(__LINE__, "nach Socket::recalla");
    if (ret == 0)
        return nbyte;
    else
        return 0;
}

void wrhdlr(int signo)
{
    signal(signo, wrhdlr);
    printf("writea wrote %d bytes\n", wrstat.sw_count);
    wrstat.sw_flag = 0;
}
#endif

int Socket::read(void *buf, unsigned nbyte)
{
    int no_of_bytes;
    char tmp_str[255];

#ifdef _WIN32
    no_of_bytes = ::recv(sock_id, (char *)buf, nbyte, 0);
#else
    no_of_bytes = ::read(sock_id, buf, nbyte);
#endif
    if (no_of_bytes <= 0)
    {
#ifdef _WIN32
        sprintf(tmp_str, "Socket recv error = %d", WSAGetLastError());
        print_comment(__LINE__, __FILE__, tmp_str);
#else
        sprintf(tmp_str, "Socket read error = %d", errno);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        //    perror("Socket read error");
        if (errno != EAGAIN) // sollte spaeter nochmal ueberprueft werden
        {
            print_comment(__LINE__, __FILE__, "read returns <= 0: close socket. Terminating process");
#ifdef __hpux
            if (errno == ENOENT)
                return 0;
#endif
#ifndef _WIN32
            if (errno == EADDRINUSE)
#else
            if (GetLastError() == WSAEADDRINUSE)
#endif
                return 0;
            print_exit(__LINE__, __FILE__, 1);
        }
    }

    return no_of_bytes;
}

int Socket::Read(void *buf, unsigned nbyte)
{
    int no_of_bytes;
    int end = 0;

    while (!end)
    {

        no_of_bytes = ::read(sock_id, buf, nbyte);
        end = 1;
        if (no_of_bytes < 0)
        {
            perror("Error reading socket");
            if (errno == EINTR)
            {
                end = 0; // repeat read call
            }
        }
    }

    return no_of_bytes;
}

Host *Socket::get_ip_alias(Host *test_host)
{
    int i, j, no, count;
    char *hostname, *aliasname;
    const char *th_address, **alias_list;

    if (ip_alias_list == 0L)
    {
        // get Network entries IP_ALIAS
        alias_list = coCoviseConfig::getScopeEntries("Network", "IpAlias");
        if (alias_list == 0L)
        {
            ip_alias_list = new char *[1];
            host_alias_list = new Host *[1];
            count = 0;
        }
        else
        {
            // clean up
            // find no of entries
            for (no = 0; alias_list[no] != 0L; no++)
            {
            }

            // allocat arrays
            ip_alias_list = new char *[no + 1];
            host_alias_list = new Host *[no + 1];
            // process entries
            count = 0;
            for (i = 0; i < no; i++)
            {
                // isolate hostname
                j = 0;
                hostname = new char[strlen(alias_list[i]) + 1];
                strcpy(hostname, alias_list[i]);
                while (hostname[j] != '\t' && hostname[j] != '\0' && hostname[j] != '\n' && hostname[j] != ' ')
                    j++;
                if (hostname[j] == '\0')
                    continue;
                hostname[j] = '\0';
                // isolate alias
                aliasname = &hostname[j + 1];
                while (*aliasname == ' ' || *aliasname == '\t')
                    aliasname++;
                j = 0;
                while (aliasname[j] != '\t' && aliasname[j] != '\0' && aliasname[j] != '\n' && aliasname[j] != ' ')
                    j++;
                aliasname[j] = '\0';
                //	    cerr << "Alias: " << hostname << ", " << aliasname << endl;
                // enter name into list
                ip_alias_list[count] = new char[strlen(hostname) + 1];
                strcpy(ip_alias_list[count], hostname);
                delete[] hostname;
                // enter host into list
                host_alias_list[count] = new Host(aliasname);
                count++;
            }
        }
        // init last entry
        ip_alias_list[count] = 0L;
        host_alias_list[count] = 0L;
        delete[] alias_list;
    }
    th_address = test_host->getAddress();
    j = 0;
    //    cerr << "looking for matches\n";
    while (ip_alias_list[j] != 0L)
    {
        if (strcmp(ip_alias_list[j], th_address) == 0)
        {
            //	    cerr << "found: " << ip_alias_list[j] << endl;
            return host_alias_list[j];
        }
        //	cerr << ip_alias_list[j] << " != " << th_address << endl;
        j++;
    }
    //    cerr << "no match\n";
    return (new Host(*test_host));
}

#ifdef MULTICAST
MulticastSocket::MulticastSocket(char *MulticastGroup, int p, int ttl_value)
{
    struct sockaddr_in addr;
    struct in_addr grpaddr;
    int i;
    for (i = 0; i < sizeof(addr); i++)
        ((char *)&addr)[i] = 0;

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;

    grpaddr.s_addr = inet_addr(MulticastGroup);

    addr.sin_port = htons(p);

    sock_id = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_id < 0)
    {
        sock_id = -1;
        return;
    }
#ifndef _WIN32
// Make the sock_id non-blocking so it can read from the net faster
//    if (fcntl(sock_id, F_SETFL, FNDELAY) < 0) {
//       print_comment(__LINE__, __FILE__, "Could not set socket for nonblocking reads");
//    }
// Allow multiple instances of this program to listen on the same
// port on the same host. By default only 1 program can bind
// to the port on a host.
#if !defined __linux__ && !defined __sun
    int on = 1;
    if (setsockopt(sock_id, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on)) < 0)
    {
        sock_id = -1;
        return;
    }
#endif
#endif

    // Assign a name to the socket.
    if (bind(sock_id, (sockaddr *)&addr, sizeof(addr)) < 0)
    {
        sock_id = -1;
        return;
    }

    // Indicate our interest in receiving packets sent to the mcast address.
    struct ip_mreq mreq;
    mreq.imr_multiaddr = grpaddr;
    mreq.imr_interface.s_addr = INADDR_ANY;
    if (setsockopt(sock_id, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                   (char *)&mreq, sizeof(mreq)) < 0)
    {
        print_comment(__LINE__, __FILE__, "Could not add multicast address to socket");
        sock_id = -1;
        return;
    }

    // Put a limit on how far the packets can travel.
    u_char ttl = ttl_value;

    if (setsockopt(sock_id, IPPROTO_IP, IP_MULTICAST_TTL,
                   (char *)&ttl, sizeof(ttl)) < 0)
    {
        print_comment(__LINE__, __FILE__, "Could not initialize ttl on socket");
        //sock_id= -1;
        //return;
    }
    // Turn looping off so packets will not go back to the same host.
    // This means that multiple instances of this program will not receive
    // packets from each other.
    /*    u_char loop = 0;
          if (setsockopt(sock_id, IPPROTO_IP, IP_MULTICAST_LOOP,
            &loop, sizeof(loop)) < 0) {
              print_comment(__LINE__, __FILE__, "Could not initialize socket");
              sock_id= -1;
              return;
          }
      */
    // Set up the destination address for packets we send out.
    s_addr_in = addr;
    s_addr_in.sin_addr = grpaddr;
}

int MulticastSocket::write(const void *buf, unsigned nbyte)
{
    return (sendto(sock_id, (char *)buf, nbyte, 0, (sockaddr *)&s_addr_in, sizeof(struct sockaddr_in)));
}
int MulticastSocket::read(void *buf, unsigned nbyte)
{
    return (recvfrom(sock_id, (char *)buf, nbyte, 0, NULL, 0));
}
MulticastSocket::~MulticastSocket()
{
}
#endif
