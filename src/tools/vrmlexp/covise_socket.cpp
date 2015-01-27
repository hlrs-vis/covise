/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTypes.h"
#ifdef __alpha
#include "externc_alpha.h"
#endif

#include "covise_socket.h"
//#include <config/CoviseConfig.h>

#ifdef _WIN32
#include <WS2tcpip.h>
#define sleep(x)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/time.h>
#endif

#ifdef CRAY
#include <sys/iosw.h>
#include <sys/param.h>
#endif

#ifdef _WIN32
#undef CRAY
#endif

#define DEBUG
#undef DEBUG

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
#include <netinet/in.h>
#else
#define MAX_SOCK_BUF 65536
#endif

#ifdef PROTOTYPES_FOR_FUNCTIONS_FROM_SYSTEM_HEADERS
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
#endif

FirewallConfig *FirewallConfig::theFirewallConfig = NULL;

FirewallConfig::FirewallConfig()
{
    sourcePort = 31000;
    destinationPort = 31000;
    setSourcePort = false;
    /* const char *cport = coCoviseConfig::getEntry("Network.SourcePort");
   if(cport)
   {
      sourcePort = atoi(cport);
   setSourcePort = coCoviseConfig::isOn("Network.SetSourcePort",false);
   cport = coCoviseConfig::getEntry("Network.CovisePort");
   if(cport)
   {
      destinationPort = atoi(cport);
   }
   }*/
}

FirewallConfig::~FirewallConfig()
{
}

FirewallConfig *FirewallConfig::the()
{
    if (!theFirewallConfig)
        theFirewallConfig = new FirewallConfig();

    return theFirewallConfig;
}

int Socket::stport = 31000;
char **Socket::ip_alias_list = NULL;
Host **Socket::host_alias_list = NULL;
void Socket::initialize()
{
#ifdef _WIN32
    int err;
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(1, 1);
    err = WSAStartup(wVersionRequested, &wsaData);
#endif
}

Socket::Socket(int, int sfd)
{
    sock_id = sfd;
    host = 0L;
    ip_alias_list = 0L;
    host_alias_list = 0L;
    port = 0;
}

Socket::Socket(Host *h, int p, int retries, double timeout)
{
    port = p;
    host = get_ip_alias(h);
    //    cerr << "connecting to host " << host->get_name() << " on port " << port << endl;

    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        fprintf(stderr, "error with socket to %s:%d: %s\n", host->get_name(), p, strerror(errno));
        //print_comment(__LINE__, __FILE__,"error with socket");
    }

    setTCPOptions();

    if (FirewallConfig::the()->setSourcePort)
    {
        memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
        s_addr_in.sin_family = AF_INET;
        s_addr_in.sin_addr.s_addr = INADDR_ANY;
        s_addr_in.sin_port = htons(FirewallConfig::the()->sourcePort);

        /* Assign an address to this socket and finds an unused port */

        while (bind(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
        {
#ifndef _WIN32
            if (errno == EADDRINUSE)
#else
            if (GetLastError() == WSAEADDRINUSE)
#endif
            {
                FirewallConfig::the()->sourcePort++;
                s_addr_in.sin_port = htons(FirewallConfig::the()->sourcePort);
            }
            else
            {
                fprintf(stderr, "error with bind (socket to %s:%d): %s\n", host->get_name(), p, strerror(errno));
#ifdef _WIN32
                closesocket(sock_id);
#else
                close(sock_id);
#endif
                sock_id = -1;
                return;
            }
        }
        FirewallConfig::the()->sourcePort++;
    }

#if defined CRAY || defined __alpha || defined _AIX
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
    //    	//print_comment(__LINE__, __FILE__, "Connect to server error");
    //    	//print_comment(__LINE__, __FILE__, host->get_name());
    //        print_exit(__LINE__, __FILE__, 1);
    //    }
    if (timeout > 0.0)
    {
        if (setNonBlocking(true) < 0)
        {
            timeout = 0.0;
        }
    }

    int numconn = 0;
    while (connect(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
#ifndef _WIN32
        if (timeout > 0.0 && errno == EINPROGRESS)
        {
            double remain = timeout;
            bool writable = false;
            do
            {
                fd_set writefds, readfds;
                FD_ZERO(&writefds);
                FD_ZERO(&readfds);
                FD_SET(sock_id, &writefds);
                FD_SET(sock_id, &readfds);
                struct timeval tv = { (long)remain, (long)(fmod(remain, 1.0) * 1.0e6) };
                int numfds = select(sock_id + 1, &readfds, &writefds, NULL, &tv);
                if (numfds < 0)
                {
                    if (errno == EAGAIN || errno == EINTR)
                    {
                        fprintf(stderr, "select again after socket connect\n");
                        remain -= 1.0;
                    }
                    else
                    {
                        fprintf(stderr, "select after socket connect to %s:%d failed: %s\n", host->get_name(), p, strerror(errno));
                        remain = timeout + 1.0;
                        break;
                    }
                }
                else if (numfds > 0 && (FD_ISSET(sock_id, &readfds) || FD_ISSET(sock_id, &writefds)))
                {
                    writable = true;
                    break;
                }
                else
                {
                    // timed out
                    remain = 0.0;
                }
            } while (remain > 0.0);

            if (writable)
            {
                int optval = 0;
                socklen_t optlen = sizeof(optval);
                if (getsockopt(sock_id, SOL_SOCKET, SO_ERROR, &optval, &optlen) < 0)
                {
                    fprintf(stderr, "getsockopt after select on socket to %s:%d: %s\n", host->get_name(), p, strerror(errno));
                }
                else
                {
                    if (optval == 0)
                    {
                        break;
                    }
                    else
                    {
                        fprintf(stderr, "connect to %s:%d failed according to getsockopt: %s\n", host->get_name(), p, strerror(errno));
                    }
                }
            }
        }
#else
        if (WSAGetLastError() == WSAETIMEDOUT)
        {
            fprintf(stderr, "connect to %s:%d timed out, setting retries to 0\n", host->get_name(), p);
            retries = 0;
        }
#endif
        // always allow two unsuccessful tries before complaining - looks better
        if (numconn > 1)
        {
            fprintf(stderr, "connect to %s:%d failed: %s\n", host->get_name(), p, strerror(errno));
        }
#ifdef _WIN32
        closesocket(sock_id);
#else
        close(sock_id);
#endif
        if (numconn >= retries)
        {
            sock_id = -1;
            return;
        }

        sleep(1);
        if (numconn > 1)
            fprintf(stderr, "Try to reconnect %d\n", numconn);
        numconn++;
        sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
        if (sock_id < 0)
        {
            fprintf(stderr, "error 2 with socket to %s:%d: %s\n", host->get_name(), p, strerror(errno));
            //print_comment(__LINE__, __FILE__,"error 2 with socket");
        }

        setTCPOptions();

#if defined CRAY || defined __alpha || defined _AIX
        //        memcpy ((char *) &s_addr_in.sin_addr, hp->h_addr, hp->h_length);
        host->get_char_address((unsigned char *)&s_addr_in.sin_addr);
#else
        //        memcpy (&(s_addr_in.sin_addr.s_addr), hp->h_addr, hp->h_length);
        host->get_char_address((unsigned char *)&(s_addr_in.sin_addr.s_addr));
#endif

        s_addr_in.sin_port = htons(port);
        s_addr_in.sin_family = AF_INET;

        if (timeout > 0.0 && setNonBlocking(true) < 0)
        {
            timeout = 0.0;
        }

        //perror("Connect to server");
        ////print_comment(__LINE__, __FILE__, "Connect to server error");
        ////print_comment(__LINE__, __FILE__, host->get_name());
    }
    if (timeout > 0.0 && setNonBlocking(false) < 0)
    {
        timeout = 0.0;
    }

#ifdef DEBUG
//print_comment(__LINE__, __FILE__,"connected");
#endif
}

Socket::Socket(int p)
{
    port = p;
    host = NULL;
    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        fprintf(stderr, "creating socket to %s:%d failed: %s\n", host->get_name(), p, strerror(errno));
        sock_id = -1;
        return;
        //print_exit(__LINE__, __FILE__, 1);
    }

    setTCPOptions();

#if defined(SO_REUSEADDR)
    int on = 1;
    if (setsockopt(sock_id, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_REUSEADDR");
    }
#endif

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(port);

    /* Assign an address to this socket */

    if (bind(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
        fprintf(stderr, "error with binding socket to %s:%d: %s\n", (host && host->get_name() ? host->get_name() : "*NULL*-host"), p, strerror(errno));
#ifdef _WIN32
        closesocket(sock_id);
#else
        close(sock_id);
#endif
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
    host = NULL;

    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        fprintf(stderr, "error creating socket to %s: %s\n", host->get_name(), strerror(errno));
        return;
        //print_exit(__LINE__, __FILE__, 1);
    }

    setTCPOptions();

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(stport);

    /* Assign an address to this socket and finds an unused port */

    while (bind(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
#ifdef DEBUG
        fprintf(stderr, "bind to port %d failed: %s\n", stport, strerror(errno));
#endif
#ifndef _WIN32
        if (errno == EADDRINUSE)
#else
        if (GetLastError() == WSAEADDRINUSE)
#endif
        {
            stport++;
            s_addr_in.sin_port = htons(stport);
            //	    cerr << "neuer port: " << stport << "\n";
        }
        else
        {
            fprintf(stderr, "error with bind (socket to %s): %s\n", host->get_name(), strerror(errno));
#ifdef _WIN32
            closesocket(sock_id);
#else
            close(sock_id);
#endif
            sock_id = -1;
            return;
            //print_exit(__LINE__, __FILE__, 1);
        }
    }
    *p = port = ntohs(s_addr_in.sin_port);
    stport++;
#ifdef DEBUG
    fprintf(stderr, "bind succeeded: port=%d, addr=0x%08x\n", *p, s_addr_in.sin_addr.s_addr);
#endif
    host = new Host(s_addr_in.sin_addr.s_addr);

#ifdef DEBUG
    print();
#endif
}

Socket::Socket(const Socket &s)
{
    s_addr_in = s.s_addr_in;
    port = s.port;
    sock_id = s.sock_id;
    host = new Host(*(s.host));
}

int Socket::setTCPOptions()
{
#ifdef CRAY
    char *hostname;
    int len, winshift;
#endif

    int ret = 0;
    int on = 1;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_NODELAY,
                   (char *)&on, sizeof(on)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_NODELAY");
        ret |= -1;
    }
#if 0
   struct linger linger;
   linger.l_onoff = 1;
   linger.l_linger = 60;
   if(setsockopt(sock_id, SOL_SOCKET, SO_LINGER,
      (char *) &linger, sizeof(linger)) < 0)
   {
      //print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET SO_LINGER");
      ret |= -1;
   }
#endif
    int sendbuf = MAX_SOCK_BUF;
    int recvbuf = MAX_SOCK_BUF;
#ifdef CRAY
    winshift = 4;
    if (setsockopt(sock_id, IPPROTO_TCP, TCP_WINSHIFT,
                   &winshift, sizeof(winshift)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "setsockopt error IPPROTO_TCP TCP_WINSHIFT");
        ret |= -1;
    }
    sendbuf = 6 * 65536;
    recvbuf = 6 * 65536;
//print_comment(__LINE__, __FILE__, "HIPPI interface");
#endif
    if (setsockopt(sock_id, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_SNDBUF");
        ret |= -1;
    }
    if (setsockopt(sock_id, SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_RCVBUF");
        ret |= -1;
    }

#if 0
   on = 1;
   if (setsockopt(sock_id, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)) < 0)
   {
      //print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_REUSEADDR");
      ret |= -1;
   }
#endif

#if /* defined(_WIN32) ||*/ defined(SO_EXCLUSIVEADDRUSE)
    on = 1;
    if (setsockopt(sock_id, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (char *)&on, sizeof(on)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "setsockopt error SOL_SOCKET, SO_EXCLUSIVEREUSEADDR");
        ret |= -1;
    }
#endif

    return ret;
}

int Socket::accept()
{
    int tmp_sock_id;
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    int err = ::listen(sock_id, 20);
    if (err == -1)
    {
        fprintf(stderr, "error with listen (socket on %s:%d): %s\n",
                host->get_name(), port, strerror(errno));
        return -1;
    }

    socklen_t length = sizeof(s_addr_in);
    tmp_sock_id = (int)::accept(sock_id, (sockaddr *)(void *)&s_addr_in, &length);

    port = ntohs(s_addr_in.sin_port);

    if (tmp_sock_id < 0)
    {
        fprintf(stderr, "error with accept (socket to %s:%d): %s\n", host->get_name(), port, strerror(errno));
        //print_exit(__LINE__, __FILE__, 1);
        return -1;
    }

#ifdef DEBUG
    printf("Connection from host %s, port %u\n",
           inet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

//    shutdown(s, 2);
#ifdef _WIN32
    closesocket(sock_id);
#else
    close(sock_id);
#endif
    sock_id = tmp_sock_id;
    host = new Host(s_addr_in.sin_addr.s_addr);
#ifdef DEBUG
    print();
#endif

    return 0;
}

int Socket::listen()
{
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    int err = ::listen(sock_id, 20);
    if (err == -1)
    {
        fprintf(stderr, "error with listen (socket on %s:%d): %s\n",
                host->get_name(), port, strerror(errno));
        return -1;
    }

    return 0;
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
    int err = ::listen(sock_id, 20);
    if (err == -1)
    {
        fprintf(stderr, "error with listen (socket on %s:%d): %s\n",
                host->get_name(), port, strerror(errno));
        return -1;
    }

    do
    {
        timeout.tv_sec = wait;
        timeout.tv_usec = 0;
        FD_ZERO(&fdread);
        FD_SET(sock_id, &fdread);
        if (wait > 0)
        {
            i = select(sock_id + 1, &fdread, 0L, 0L, &timeout);
        }
        else
        {
            i = select(sock_id + 1, &fdread, 0L, 0L, NULL);
        }
        if (i == 0)
        {
            return -1;
        }
        else if (i < 0)
        {
            if (errno != EINTR)
            {
                fprintf(stderr, "error with select (socket listening on %s:%d): %s\n", host->get_name(), port, strerror(errno));
                return -1;
            }
        }
        //fprintf(stderr, "retrying select, listening on %s:%d, timeout was %ds, pid=%d\n", host->get_name(), port, wait, getpid());
    } while (i < 0);
    socklen_t length = sizeof(s_addr_in);
    tmp_sock_id = (int)::accept(sock_id, (sockaddr *)(void *)&s_addr_in, &length);

    port = ntohs(s_addr_in.sin_port);

    if (tmp_sock_id < 0)
    {
        fprintf(stderr, "error with accept (socket listening on %s:%d): %s\n", host->get_name(), port, strerror(errno));
        //print_exit(__LINE__, __FILE__, 1);
    }

#ifdef DEBUG
    printf("Connection from host %s, port %u\n",
           inet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

//    shutdown(s, 2);
#ifdef _WIN32
    closesocket(sock_id);
#else
    close(sock_id);
#endif
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
//    //print_comment(__LINE__, __FILE__, tmpstr);
#ifdef _WIN32
    closesocket(sock_id);
#else
    if (sock_id != -1)
    {
        close(sock_id);
    }
#endif
    delete host;
}

//extern CoviseTime *covise_time;

int Socket::write(const void *buf, unsigned nbyte)
{
    int no_of_bytes;
    char tmp_str[255];
    do
    {
//    covise_time->mark(__LINE__, "vor Socket::write");
#ifdef _WIN32
        no_of_bytes = ::send(sock_id, (char *)buf, nbyte, 0);
#else
        no_of_bytes = ::write(sock_id, buf, nbyte);
#endif
    } while ((no_of_bytes <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    if (no_of_bytes <= 0)
    {
#ifdef _WIN32
        sprintf(tmp_str, "Socket send error = %d", WSAGetLastError());
//print_error(__LINE__, __FILE__, tmp_str);
#else
        if (errno == EPIPE)
            return COVISE_SOCKET_INVALID;
        if (errno == ECONNRESET)
            return COVISE_SOCKET_INVALID;
        sprintf(tmp_str, "Socket write error = %d: %s, no_of_bytes = %d", errno, strerror(errno), no_of_bytes);
//print_error(__LINE__, __FILE__, tmp_str);
#endif
        fprintf(stderr, "error writing on socket to %s:%d: %s\n", host->get_name(), port, strerror(errno));
        //print_error(__LINE__, __FILE__, "write returns <= 0: close socket.");
        //	print_exit(__LINE__, __FILE__, 1);
        return COVISE_SOCKET_INVALID;
    }

    return no_of_bytes;
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
    errno = 0;
    do
    {
#ifdef _WIN32
        no_of_bytes = ::recv(sock_id, (char *)buf, nbyte, 0);
#else
        no_of_bytes = ::read(sock_id, buf, nbyte);
#endif
    } while ((no_of_bytes <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    if (no_of_bytes <= 0)
    {
#ifdef _WIN32
        sprintf(tmp_str, "Socket recv error = %d", WSAGetLastError());
//print_comment(__LINE__, __FILE__, tmp_str);
#else
        sprintf(tmp_str, "Socket read error = %d", errno);
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
        //    perror("Socket read error");
        // sollte spaeter nochmal ueberprueft werden
        if ((errno != EAGAIN) && (errno != EINTR))
        {
//print_comment(__LINE__, __FILE__, "read returns <= 0: close socket. Terminating process");
#ifdef __hpux
            if (errno == ENOENT)
                return 0;
#endif
#ifndef _WIN32
            if (errno == EADDRINUSE)
#else
            int errorNumber = GetLastError();
            if (errorNumber == WSAEADDRINUSE || errorNumber == WSAEWOULDBLOCK)
#endif
                return 0;
            //print_exit(__LINE__, __FILE__, 1);
            return -1;
        }
    }

    return no_of_bytes;
}

int Socket::setNonBlocking(bool on)
{
#ifndef _WIN32
    int flags = fcntl(sock_id, F_GETFL, 0);
    if (flags < 0)
    {
        fprintf(stderr, "getting flags on socket to %s:%d failed: %s\n", host->get_name(), port, strerror(errno));
        return -1;
    }
#endif
    if (on)
    {
#ifdef _WIN32
        unsigned long tru = 1;
        ioctlsocket(sock_id, FIONBIO, &tru);
#else
        if (fcntl(sock_id, F_SETFL, flags | O_NONBLOCK))
        {
            fprintf(stderr, "setting O_NONBLOCK on socket to %s:%d failed: %s\n", host->get_name(), port, strerror(errno));
            return -1;
        }
#endif
    }
    else
    {
#ifdef _WIN32
        unsigned long tru = 0;
        ioctlsocket(sock_id, FIONBIO, &tru);
#else
        if (fcntl(sock_id, F_SETFL, flags & (~O_NONBLOCK)))
        {
            fprintf(stderr, "removing O_NONBLOCK from socket to %s:%d failed: %s\n", host->get_name(), port, strerror(errno));
            return -1;
        }
#endif
    }

    return 0;
}

int Socket::Read(void *buf, unsigned nbyte)
{
    int no_of_bytes;

    do
    {
#ifdef _WIN32
        no_of_bytes = ::recv(sock_id, (char *)buf, nbyte, 0);
#else
        no_of_bytes = ::read(sock_id, buf, nbyte);
#endif
    } while ((no_of_bytes <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    if (no_of_bytes <= 0)
    {
        return (-1); // error
        /*
      char tmp_str[255];

      #ifdef _WIN32
      sprintf(tmp_str,"Socket recv error = %d", WSAGetLastError());
      //print_comment(__LINE__, __FILE__, tmp_str);
      #else
      sprintf(tmp_str,"Socket read error = %d", errno);
      //print_comment(__LINE__, __FILE__, tmp_str);
      #endif
      if(errno != EAGAIN) { // sollte spaeter nochmal ueberprueft werden
      //print_comment(__LINE__, __FILE__, "read returns <= 0: close socket. Terminating process");
      #ifndef _WIN32
      if(errno == EADDRINUSE)
      #else
      if(GetLastError() == WSAEADDRINUSE)
      #endif
      return 0;
      }*/
    }

    return no_of_bytes;
}

Host *Socket::get_ip_alias(Host *test_host)
{
    int j, count;
    const char **alias_list = NULL;
    const char *th_address;

    if (ip_alias_list == 0L)
    {
        // get Network entries IpAlias
        /*alias_list = coCoviseConfig::getScopeEntries("Network", "IpAlias");
      if(alias_list == 0L)
      {*/
        ip_alias_list = new char *[1];
        host_alias_list = new Host *[1];
        count = 0;
        /* }
      else
      {
         // clean up
         // find no of entries
         for(no = 0;alias_list[no] != 0L;no++)
            {}

            // allocat arrays
            ip_alias_list = new char*[no+1];
         host_alias_list = new Host*[no+1];
         // process entries
         count = 0;
         for(i = 0;i < no;i++)
         {
            // isolate hostname
            j = 0;
            hostname = alias_list[i];
            while(hostname[j] != '\t' && hostname[j] != '\0' &&
               hostname[j] != '\n' && hostname[j] != ' ')
               j++;
            if(hostname[j] == '\0')
               continue;
            char *hostName = new char[j+1];
            strncpy(hostName,hostname,j);
            hostName[j] = '\0';
            // isolate alias
            aliasname = &hostname[j+1];
            while(*aliasname == ' ' || *aliasname == '\t')
               aliasname++;
            j = 0;
            while(aliasname[j] != '\t' && aliasname[j] != '\0' &&
               aliasname[j] != '\n' && aliasname[j] != ' ')
               j++;

            char *aliasName = new char[j+1];
            strncpy(aliasName,aliasname,j);
            aliasName[j] = '\0';
            //	    cerr << "Alias: " << hostname << ", " << aliasname << endl;
            // enter name into list
            ip_alias_list[count] = hostName;
            // enter host into list
            host_alias_list[count] = new Host(aliasName);
            delete[] aliasName;
            count++;
         }
      }*/
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

UDPSocket::UDPSocket(char *address, int p)
{
    struct sockaddr_in addr;
    struct in_addr grpaddr;
    int i;
    for (i = 0; i < sizeof(addr); i++)
        ((char *)&addr)[i] = 0;

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;

    grpaddr.s_addr = inet_addr(address);

    addr.sin_port = htons(p);

    sock_id = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_id < 0)
    {
        sock_id = -1;
        return;
    }
#ifndef _WIN32
// Make the sock_id non-blocking so it can read from the net faster
//    if (fcntl(sock_id, F_SETFL, O_NDELAY) < 0) {
//       //print_comment(__LINE__, __FILE__, "Could not set socket for nonblocking reads");
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
    if (bind(sock_id, (sockaddr *)(void *)&addr, sizeof(addr)) < 0)
    {
        fprintf(stderr, "binding of udp socket to port %d failed: %s\n",
                p, strerror(errno));
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
        //print_comment(__LINE__, __FILE__, "Could not add multicast address to socket");
        sock_id = -1;
        return;
    }
    // Set up the destination address for packets we send out.
    s_addr_in = addr;
    s_addr_in.sin_addr = grpaddr;
}

int UDPSocket::write(const void *buf, unsigned nbyte)
{
    return (sendto(sock_id, (char *)buf, nbyte, 0, (sockaddr *)(void *)&s_addr_in, sizeof(struct sockaddr_in)));
}
int UDPSocket::read(void *buf, unsigned nbyte)
{
    return (recvfrom(sock_id, (char *)buf, nbyte, 0, NULL, 0));
}
UDPSocket::~UDPSocket()
{
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

    sock_id = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_id < 0)
    {
        sock_id = -1;
        return;
    }
#ifndef _WIN32
// Make the sock_id non-blocking so it can read from the net faster
//    if (fcntl(sock_id, F_SETFL, O_NDELAY) < 0) {
//       //print_comment(__LINE__, __FILE__, "Could not set socket for nonblocking reads");
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
    if (bind(sock_id, (sockaddr *)(void *)&addr, sizeof(addr)) < 0)
    {
        fprintf(stderr, "binding of multicast socket to port %d failed: %s\n",
                p, strerror(errno));
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
        //print_comment(__LINE__, __FILE__, "Could not add multicast address to socket");
        sock_id = -1;
        return;
    }

    // Put a limit on how far the packets can travel.
    unsigned char ttl = ttl_value;

    if (setsockopt(sock_id, IPPROTO_IP, IP_MULTICAST_TTL,
                   (char *)&ttl, sizeof(ttl)) < 0)
    {
        //print_comment(__LINE__, __FILE__, "Could not initialize ttl on socket");
        //sock_id= -1;
        //return;
    }
    // Turn looping off so packets will not go back to the same host.
    // This means that multiple instances of this program will not receive
    // packets from each other.
    /*    unsigned char loop = 0;
       if (setsockopt(sock_id, IPPROTO_IP, IP_MULTICAST_LOOP,
         &loop, sizeof(loop)) < 0) {
           //print_comment(__LINE__, __FILE__, "Could not initialize socket");
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
    return (sendto(sock_id, (char *)buf, nbyte, 0, (sockaddr *)(void *)&s_addr_in, sizeof(struct sockaddr_in)));
}
int MulticastSocket::read(void *buf, unsigned nbyte)
{
    return (recvfrom(sock_id, (char *)buf, nbyte, 0, NULL, 0));
}
MulticastSocket::~MulticastSocket()
{
}
#endif
