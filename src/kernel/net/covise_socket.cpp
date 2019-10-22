/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _WIN32
#include <ws2tcpip.h>
#endif

#ifdef HAVE_OPENSSL
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include <iostream>

#include <sys/types.h>
#include <signal.h>
#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#endif

#ifdef CRAY
#include <sys/iosw.h>
#include <sys/param.h>
#endif

#ifdef _WIN32
#undef CRAY
#endif

#include <util/coErr.h>
#include "covise_socket.h"
#include "covise_host.h"

using std::cerr;
using std::endl;

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

using namespace covise;

int Socket::stport = 31000;
char **Socket::ip_alias_list = NULL;
Host **Socket::host_alias_list = NULL;
bool Socket::bInitialised = false;

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

#ifndef _WIN32
#define closesocket close
#endif

FirewallConfig *FirewallConfig::theFirewallConfig = NULL;

FirewallConfig::FirewallConfig()
{
    sourcePort = coCoviseConfig::getInt("sourcePort", "System.Network", 31000);
    setSourcePort = coCoviseConfig::isOn("setSourcePort", "System.Network", false, 0);
    destinationPort = coCoviseConfig::getInt("covisePort", "System.Network", 31000);
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

void Socket::initialize()
{
#ifdef _WIN32
    if (!bInitialised)
    {
        int err;
        unsigned short wVersionRequested;
        struct WSAData wsaData;
        wVersionRequested = MAKEWORD(2, 2);
        err = WSAStartup(wVersionRequested, &wsaData);
        bInitialised = true;
    }
#else
    signal(SIGPIPE, SIG_IGN);
#endif
}

void Socket::uninitialize()
{
#ifdef _WIN32
    if (bInitialised)
    {
        int err;
        err = WSACleanup();
        bInitialised = false;
    }
#endif
}

Socket::Socket(int, int sfd)
{
    sock_id = sfd;
    host = NULL;
    ip_alias_list = NULL;
    host_alias_list = NULL;
    port = 0;
    this->connected = true;
}

Socket::Socket(int socket_id, sockaddr_in *sockaddr)
{
    sock_id = socket_id;
	char buf[100];
	if (!inet_ntop(AF_INET, &sockaddr->sin_addr, buf, sizeof(buf)))
	{
		cerr << "Socket::Socket: address    = <unknown>" << endl;
		buf[0] = '\0';
	}
	host = new Host(buf);
    ip_alias_list = NULL;
    host_alias_list = NULL;
    port = ntohs(sockaddr->sin_port);
    this->connected = true;
}

Socket::Socket(const Host *h, int p, int retries, double timeout)
{
    port = p;
    this->connected = false;

    host = get_ip_alias(h);
    //    cerr << "connecting to host " << host->getAddress() << " on port " << port << endl;

    errno = 0;
    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
    // needed otherwise socket is sometimes closed (win 7 64bit)
    //fprintf(stderr, "**** timeout %f\n", timeout);
    if (timeout == 0.0)
        timeout = .1;
#endif

    if (sock_id < 0)
    {
        fprintf(stderr, "error with socket to %s:%d: %s\n", host->getAddress(), p, coStrerror(getErrno()));
        coPerror("Socket::Socket");
        LOGINFO("error with socket");
    }

    setTCPOptions();

    if (FirewallConfig::the()->setSourcePort)
    {
        memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
        s_addr_in.sin_family = AF_INET;
        s_addr_in.sin_addr.s_addr = INADDR_ANY;
        s_addr_in.sin_port = htons(FirewallConfig::the()->sourcePort);

        /* Assign an address to this socket and finds an unused port */

        errno = 0;
        while (bind(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
        {
#ifndef _WIN32
            if (errno == EADDRINUSE)
#else
            if (getErrno() == WSAEADDRINUSE)
#endif
            {
                FirewallConfig::the()->sourcePort++;
                s_addr_in.sin_port = htons(FirewallConfig::the()->sourcePort);
            }
            else
            {
                fprintf(stderr, "error with bind (socket to %s:%d): %s\n", host->getAddress(), p, coStrerror(getErrno()));
                coPerror("Socket_bind");
                closesocket(sock_id);
                sock_id = -1;
                return;
            }
            errno = 0;
        }
        FirewallConfig::the()->sourcePort++;
    }

    //        memcpy (&(s_addr_in.sin_addr.s_addr), hp->h_addr, hp->h_length);
    host->get_char_address((unsigned char *)&(s_addr_in.sin_addr.s_addr));

    s_addr_in.sin_port = htons(port);
    s_addr_in.sin_family = AF_INET;

    if (timeout > 0.0)
    {
        if (setNonBlocking(true) < 0)
        {
            timeout = 0.0;
        }
    }

    int numconn = 0;
    errno = 0;
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
                struct timeval tv = { (int)remain, (int)(fmod(remain, 1.0) * 1.0e6) };
                errno = 0;
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
                        fprintf(stderr, "select after socket connect to %s:%d failed: %s\n", host->getAddress(), p, coStrerror(errno));
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
                errno = 0;
                if (getsockopt(sock_id, SOL_SOCKET, SO_ERROR, &optval, &optlen) < 0)
                {
                    fprintf(stderr, "getsockopt after select on socket to %s:%d: %s\n", host->getAddress(), p, coStrerror(errno));
                }
                else
                {
                    if (optval == 0)
                    {
                        break;
                    }
                    else
                    {
                        errno = optval;
                        if (optval != ECONNREFUSED)
                            fprintf(stderr, "connect to %s:%d failed according to getsockopt: %s\n", host->getAddress(), p, coStrerror(optval));
                    }
                }
            }
        }
#else

        if (WSAGetLastError() == WSAEWOULDBLOCK) // connection not yet established, have to wait in a select
        {
            double remain = 1.0; // wait a second, otherwise it won't work on windows
            if (timeout > 0.0)
                remain = timeout;
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

                    fprintf(stderr, "select after socket connect to %s:%d failed: %d\n", host->getAddress(), p, WSAGetLastError());
                    coPerror("select after socket connect");
                    remain = timeout + 1.0;
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
                break;
        }
        else if (WSAGetLastError() == WSAETIMEDOUT)
        {
            //fprintf(stderr, "connect to %s:%d timed out, setting retries to 0\n", host->getAddress(), p);
            //retries=0;
        }
        else
        {
            fprintf(stderr, "connect to %s:%d failed: %s %d\n", host->getAddress(), p, coStrerror(WSAGetLastError()), WSAGetLastError());
        }

#endif
// always allow two unsuccessful tries before complaining - looks better
#ifndef WIN32
        if (numconn > 1)
        {
            fprintf(stderr, "connect to %s:%d failed: %s %d\n", host->getAddress(), p, coStrerror(getErrno()), getErrno());
        }
#endif

        closesocket(sock_id);
        if (numconn >= retries)
        {
            sock_id = -1;
            return;
        }

        if (numconn > 1)
        {
            fprintf(stderr, "Try to reconnect %d\n", numconn);
            sleep(1);
        }
        numconn++;
        errno = 0;
        sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
        if (sock_id < 0)
        {
            fprintf(stderr, "error 2 with socket to %s:%d: %s\n", host->getAddress(), p, coStrerror(getErrno()));
            LOGINFO("error 2 with socket");
        }

        setTCPOptions();

        //        memcpy (&(s_addr_in.sin_addr.s_addr), hp->h_addr, hp->h_length);
        host->get_char_address((unsigned char *)&(s_addr_in.sin_addr.s_addr));

        s_addr_in.sin_port = htons(port);
        s_addr_in.sin_family = AF_INET;

        if (timeout > 0.0 && setNonBlocking(true) < 0)
        {
            timeout = 0.0;
        }

        //perror("Connect to server");
        //LOGINFO( "Connect to server error");
        //LOGINFO( host->getAddress());
        errno = 0;
    }
    if (timeout > 0.0 && setNonBlocking(false) < 0)
    {
        timeout = 0.0;
    }

    this->connected = true;

#ifdef DEBUG
    LOGINFO("connected");
#endif
}

Socket::Socket(int p)
{
    port = p;
    this->connected = false;

    host = NULL;
    errno = 0;
    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        fprintf(stderr, "creating socket for port %d failed: %s\n", p, coStrerror(getErrno()));
        sock_id = -1;
        return;
    }

    setTCPOptions(); //TODO: Re-enable setTCPOptions

#if defined(SO_REUSEADDR)
    int on = 1;
    if (setsockopt(sock_id, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)) < 0)
    {
        LOGINFO("setsockopt error SOL_SOCKET, SO_REUSEADDR");
    }
#endif

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(port);

    /* Assign an address to this socket */

    errno = 0;
    if (bind(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
        fprintf(stderr, "error with binding socket to %s:%d: %s\n", (host && host->getAddress() ? host->getAddress() : "*NULL*-host"), p, coStrerror(getErrno()));
        closesocket(sock_id);
        sock_id = -1;
        return;
    }
    host = new Host(s_addr_in.sin_addr.s_addr);

#ifdef DEBUG
    print();
#endif
}

Socket::Socket(int *p)
{
    host = NULL;
    this->connected = false;

    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        fprintf(stderr, "error creating listening socket: %s\n", coStrerror(getErrno()));
        return;
    }

    setTCPOptions();

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(stport);

    /* Assign an address to this socket and finds an unused port */

    errno = 0;
    while (bind(sock_id, (sockaddr *)(void *)&s_addr_in, sizeof(s_addr_in)) < 0)
    {
#ifdef DEBUG
        fprintf(stderr, "bind to port %d failed: %s\n", stport, coStrerror(getErrno()));
#endif
#ifndef _WIN32
        if (errno == EADDRINUSE)
#else
        if (WSAGetLastError() == WSAEADDRINUSE)
#endif
        {
            stport++;
            s_addr_in.sin_port = htons(stport);
//	    cerr << "neuer port: " << stport << "\n";
#ifdef _WIN32
            /* after an unsuccessfull bind we need to close the socket and reopen it on windows, or at least this was the case here on vista on a surface box */
            closesocket(sock_id);
            sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
            if (sock_id < 0)
            {
                fprintf(stderr, "error creating listening socket: %s\n", coStrerror(getErrno()));
                return;
            }

            setTCPOptions();

            memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
            s_addr_in.sin_family = AF_INET;
            s_addr_in.sin_addr.s_addr = INADDR_ANY;
            s_addr_in.sin_port = htons(stport);
#endif
        }
        else
        {
            fprintf(stderr, "error with bind (listening socket): %s\n", coStrerror(getErrno()));
            closesocket(sock_id);
            sock_id = -1;
            return;
        }
        errno = 0;
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
    this->connected = true;

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
        LOGINFO("setsockopt error IPPROTO_TCP TCP_NODELAY");
        ret |= -1;
    }
#if 0
   struct linger linger;
   linger.l_onoff = 1;
   linger.l_linger = 60;
   if(setsockopt(sock_id, SOL_SOCKET, SO_LINGER,
      (char *) &linger, sizeof(linger)) < 0)
   {
      LOGINFO( "setsockopt error SOL_SOCKET SO_LINGER");
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
        LOGINFO("setsockopt error IPPROTO_TCP TCP_WINSHIFT");
        ret |= -1;
    }
    sendbuf = 6 * 65536;
    recvbuf = 6 * 65536;
    LOGINFO("HIPPI interface");
#endif
    if (setsockopt(sock_id, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sendbuf, sizeof(sendbuf)) < 0)
    {
        LOGINFO("setsockopt error SOL_SOCKET, SO_SNDBUF");
        ret |= -1;
    }
    if (setsockopt(sock_id, SOL_SOCKET, SO_RCVBUF,
                   (char *)&recvbuf, sizeof(recvbuf)) < 0)
    {
        LOGINFO("setsockopt error SOL_SOCKET, SO_RCVBUF");
        ret |= -1;
    }

#if 0
   on = 1;
   if (setsockopt(sock_id, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)) < 0)
   {
      LOGINFO( "setsockopt error SOL_SOCKET, SO_REUSEADDR");
      ret |= -1;
   }
#endif

#if /* defined(_WIN32) ||*/ defined(SO_EXCLUSIVEADDRUSE)
    on = 1;
    if (setsockopt(sock_id, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (char *)&on, sizeof(on)) < 0)
    {
        LOGINFO("setsockopt error SOL_SOCKET, SO_EXCLUSIVEREUSEADDR");
        ret |= -1;
    }
#endif

    return ret;
}

int Socket::accept()
{
    errno = 0;
    int err = ::listen(sock_id, 20);
    if (err == -1)
    {
        fprintf(stderr, "error with listen (socket on %s:%d): %s\n",
                host->getAddress(), port, coStrerror(getErrno()));
        return -1;
    }
    return acceptOnly();
}
int Socket::acceptOnly()
{
    int tmp_sock_id;
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    errno = 0;

    socklen_t length = sizeof(s_addr_in);
    tmp_sock_id = (int)::accept(sock_id, (sockaddr *)(void *)&s_addr_in, &length);

    port = ntohs(s_addr_in.sin_port);

    if (tmp_sock_id < 0)
    {
        fprintf(stderr, "error with accept (socket to %s:%d): %s\n", host->getAddress(), port, coStrerror(getErrno()));
        return -1;
    }

#ifdef DEBUG
    printf("Connection from host %s, port %u\n",
           inet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

    //    shutdown(s, 2);
    closesocket(sock_id);
    sock_id = tmp_sock_id;
    host = new Host(s_addr_in.sin_addr.s_addr);
#ifdef DEBUG
    print();
#endif
    this->connected = true;

    return 0;
}

int Socket::listen()
{
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    errno = 0;
    int err = ::listen(sock_id, 20);
    if (err == -1)
    {
        fprintf(stderr, "error with listen (socket on %s:%d): %s\n",
                host->getAddress(), port, coStrerror(getErrno()));
        return -1;
    }

    return 0;
}

int Socket::accept(int wait)
{
    errno = 0;
    int err = ::listen(sock_id, 20);
    if (err == -1)
    {
        fprintf(stderr, "error with listen (socket on %s:%d): %s\n",
                host->getAddress(), port, coStrerror(getErrno()));
        return -1;
    }
   return acceptOnly(wait);
   }
int Socket::acceptOnly(int wait)
{
    int tmp_sock_id;
    struct timeval timeout;
    fd_set fdread;
    int i;
#ifdef DEBUG
    printf("Listening for connect requests on port %d\n", port);
#endif
    errno = 0;

    do
    {
        timeout.tv_sec = wait;
        timeout.tv_usec = 0;
        FD_ZERO(&fdread);
        FD_SET(sock_id, &fdread);
        errno = 0;
        if (wait > 0)
        {
            i = select(sock_id + 1, &fdread, NULL, NULL, &timeout);
        }
        else
        {
            i = select(sock_id + 1, &fdread, NULL, NULL, NULL);
        }
        if (i == 0)
        {
            return -1;
        }
        else if (i < 0)
        {
#ifdef WIN32
            if ((getErrno() != WSAEINTR) && (getErrno() != WSAEINPROGRESS))
#else
            if (getErrno() != EINTR)
#endif
            {
                fprintf(stderr, "error with select (socket listening on %s:%d): %s\n", host->getAddress(), port, coStrerror(getErrno()));
                return -1;
            }
        }
        //fprintf(stderr, "retrying select, listening on %s:%d, timeout was %ds, pid=%d\n", host->getAddress(), port, wait, getpid());
    } while (i < 0);

    socklen_t length = sizeof(s_addr_in);
    errno = 0;
    tmp_sock_id = (int)::accept(sock_id, (sockaddr *)(void *)&s_addr_in, &length);

    port = ntohs(s_addr_in.sin_port);

    if (tmp_sock_id < 0)
    {
        fprintf(stderr, "error with accept (socket listening on %s:%d): %s\n", host->getAddress(), port, coStrerror(getErrno()));
    }

#ifdef DEBUG
    printf("Connection from host %s, port %u\n",
           inet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

    //    shutdown(s, 2);
    closesocket(sock_id);
    sock_id = tmp_sock_id;
    host = new Host(s_addr_in.sin_addr.s_addr);
#ifdef DEBUG
    print();
#endif
    this->connected = true;

    return 0;
}


Socket::~Socket()
{
    //    shutdown(sock_id, 2);
    //    char tmpstr[100];

    //    sprintf(tmpstr, "delete socket %d", port);
    //    LOGINFO( tmpstr);
    this->connected = false;

    if (sock_id != -1)
    {
        closesocket(sock_id);
    }
    delete host;
}

//extern CoviseTime *covise_time;

int Socket::write(const void *buf, unsigned nbyte)
{
    int no_of_bytes = 0;
    char tmp_str[255];
    do
    {
        errno = 0;
//    covise_time->mark(__LINE__, "vor Socket::write");
#ifdef _WIN32
        no_of_bytes = ::send(sock_id, (char *)buf, nbyte, 0);
#else
        no_of_bytes = ::write(sock_id, buf, nbyte);
#endif
#ifdef WIN32
    } while ((no_of_bytes < 0) && ((getErrno() == WSAEINPROGRESS) || (getErrno() == WSAEINTR) || (getErrno() == WSAEWOULDBLOCK)));
#else
    } while ((no_of_bytes < 0) && ((errno == EAGAIN) || (errno == EINTR)));
#endif
    if (no_of_bytes < 0)
    {
#ifdef _WIN32
        sprintf(tmp_str, "Socket send error = %d", WSAGetLastError());
        LOGERROR(tmp_str);
#else
        if (errno == EPIPE)
            return COVISE_SOCKET_INVALID;
        if (errno == ECONNRESET)
            return COVISE_SOCKET_INVALID;
        sprintf(tmp_str, "Socket write error = %d: %s, no_of_bytes = %d", errno, coStrerror(errno), no_of_bytes);
        LOGERROR(tmp_str);
#endif
        fprintf(stderr, "error writing on socket to %s:%d: %s\n", host->getAddress(), port, coStrerror(getErrno()));
        LOGERROR("write returns <= 0: close socket.");
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
    ::write::writea(sock_id, (char *)buf, nbyte, &wrstat, SIGUSR1);
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
    do
    {
        errno = 0;
#ifdef _WIN32
        no_of_bytes = ::recv(sock_id, (char *)buf, nbyte, 0);
#else
        no_of_bytes = ::read(sock_id, buf, nbyte);
#endif
    }
#ifdef WIN32
    while ((no_of_bytes < 0) && ((getErrno() == WSAEINPROGRESS) || (getErrno() == WSAEINTR) || (getErrno() == WSAEWOULDBLOCK)));
#else
    while ((no_of_bytes < 0) && ((errno == EAGAIN) || (errno == EINTR)));
#endif
    if (no_of_bytes < 0)
    {
        sprintf(tmp_str, "Socket read error = %d", getErrno());
        LOGINFO(tmp_str);
        //    perror("Socket read error");
        LOGINFO("read returns <= 0: close socket.");
#ifdef __hpux
        if (errno == ENOENT)
            return 0;
#endif
#ifndef _WIN32
        if (errno == EADDRINUSE)
            return 0;
#endif

        //perror("Socket read error");
        //fprintf(stderr,"errno:%d",errorNumber);
        return -1;
    }

    return no_of_bytes;
}

int Socket::available(void)
{
    int retval = 0;
#if defined(_WIN32)
    int rc = ioctlsocket(sock_id, FIONREAD, reinterpret_cast<unsigned long *>(&retval));
#else
    int rc = ::ioctl(sock_id, FIONREAD, &retval);
#endif
    if (rc == -1)
    {
        return -1;
    }
    return retval;
}

int Socket::Read(void *buf, unsigned nbyte, char* ip)
{
    int no_of_bytes = 0;

    do
    {
        errno = 0;
#ifdef _WIN32
        no_of_bytes = ::recv(sock_id, (char *)buf, nbyte, 0);
#else
        no_of_bytes = ::read(sock_id, buf, nbyte);
#endif
    }
#ifdef _WIN32
    while (no_of_bytes == -1 && ((getErrno() == WSAEINPROGRESS) || (getErrno() == WSAEINTR) || (getErrno() == WSAEWOULDBLOCK)));
#else
    while (no_of_bytes == -1 && (errno == EAGAIN || errno == EINTR));
#endif
    if (no_of_bytes <= 0)
    {
        return (-1); // error
    }

    return no_of_bytes;
}

int Socket::setNonBlocking(bool on)
{
#ifndef _WIN32
    errno = 0;
    int flags = fcntl(sock_id, F_GETFL, 0);
    if (flags < 0)
    {
        fprintf(stderr, "getting flags on socket to %s:%d failed: %s\n", host->getAddress(), port, coStrerror(getErrno()));
        return -1;
    }
#endif
    if (on)
    {
#ifdef _WIN32
        unsigned long tru = 1;
        ioctlsocket(sock_id, FIONBIO, &tru);
#else
        errno = 0;
        if (fcntl(sock_id, F_SETFL, flags | O_NONBLOCK))
        {
            fprintf(stderr, "setting O_NONBLOCK on socket to %s:%d failed: %s\n", host->getAddress(), port, coStrerror(getErrno()));
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
        errno = 0;
        if (fcntl(sock_id, F_SETFL, flags & (~O_NONBLOCK)))
        {
            fprintf(stderr, "removing O_NONBLOCK from socket to %s:%d failed: %s\n", host->getAddress(), port, coStrerror(getErrno()));
            return -1;
        }
#endif
    }

    return 0;
}

Host *Socket::get_ip_alias(const Host *test_host)
{
    int i, j, no, count;
    const char **alias_list;
    const char *hostname, *aliasname;
    const char *th_address;

    if (ip_alias_list == NULL)
    {
        // get Network entries IpAlias
        coCoviseConfig::ScopeEntries aliasEntries = coCoviseConfig::getScopeEntries("System.Network", "IpAlias");
        alias_list = aliasEntries.getValue();
        if (alias_list == NULL)
        {
            ip_alias_list = new char *[1];
            host_alias_list = new Host *[1];
            count = 0;
        }
        else
        {
            // clean up
            // find no of entries
            for (no = 0; alias_list[no] != NULL; no++)
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
                hostname = alias_list[i];
                while (hostname[j] != '\t' && hostname[j] != '\0' && hostname[j] != '\n' && hostname[j] != ' ')
                    j++;
                if (hostname[j] == '\0')
                    continue;
                char *hostName = new char[j + 1];
                strncpy(hostName, hostname, j);
                hostName[j] = '\0';
                // isolate alias
                aliasname = &hostname[j + 1];
                while (*aliasname == ' ' || *aliasname == '\t')
                    aliasname++;
                j = 0;
                while (aliasname[j] != '\t' && aliasname[j] != '\0' && aliasname[j] != '\n' && aliasname[j] != ' ')
                    j++;

                char *aliasName = new char[j + 1];
                strncpy(aliasName, aliasname, j);
                aliasName[j] = '\0';
                //	    cerr << "Alias: " << hostname << ", " << aliasname << endl;
                // enter name into list
                ip_alias_list[count] = hostName;
                // enter host into list
                host_alias_list[count] = new Host(aliasName);
                delete[] aliasName;
                count++;
            }
        }
        // init last entry
        ip_alias_list[count] = NULL;
        host_alias_list[count] = NULL;
        //delete[] alias_list;
    }
    th_address = test_host->getAddress();
    j = 0;
    //    cerr << "looking for matches\n";
    while (ip_alias_list[j] != NULL)
    {
        if (strcmp(ip_alias_list[j], th_address) == 0)
        {
            //	    cerr << "found: " << ip_alias_list[j] << endl;
            return (new Host(*(host_alias_list[j])));
        }
        //	cerr << ip_alias_list[j] << " != " << th_address << endl;
        j++;
    }
    //    cerr << "no match\n";
    return (new Host(*test_host));
}

const char *Socket::get_hostname()
{
    return host->getAddress();
}

bool Socket::isConnected()
{
    return this->connected;
}

void Socket::print()
{
#ifdef DEBUG
    std::cerr << "Socket id: " << sock_id << "  Port: " << port << std::endl;
    host->print();
#endif
}

UDPSocket::UDPSocket(int p,const char *address)
{
	port = p;
    this->connected = false;	
    sock_id = (int)socket(AF_INET, SOCK_DGRAM, 0);	
    if (sock_id < 0)
    {
        sock_id = -1;
        return;
    }
	memset((char*)& s_addr_in, 0, sizeof(s_addr_in));
	s_addr_in.sin_family = AF_INET;
	s_addr_in.sin_addr.s_addr = INADDR_ANY;
	s_addr_in.sin_port = htons(port);
	sockaddr_in c_addr_in = s_addr_in; //save adress in s_addr_in for sending, but bind without it
	if (address)
	{
		int err = inet_pton(AF_INET, address, &s_addr_in.sin_addr.s_addr);
		if (err != 1)
		{
			sock_id = -1;
			return;
		}
	}


#ifndef _WIN32
// Make the sock_id non-blocking so it can read from the net faster
//    if (fcntl(sock_id, F_SETFL, O_NDELAY) < 0) {
//       LOGINFO( "Could not set socket for nonblocking reads");
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
    errno = 0;
	if (bind(sock_id, (sockaddr*)(void*)& c_addr_in, sizeof(c_addr_in)) < 0)
    {
        fprintf(stderr, "binding of udp socket to port %d failed: %s\n",
                p, coStrerror(getErrno()));
        sock_id = -1;
        return;
    }
	
	if (!address)
	{
		host = new Host(s_addr_in.sin_addr.s_addr);
	}
}


int UDPSocket::write(const void *buf, unsigned nbyte)
{
    return (sendto(sock_id, (char *)buf, nbyte, 0, (sockaddr *)(void *)&s_addr_in, sizeof(struct sockaddr_in)));
}
int UDPSocket::writeTo(const void* buf, unsigned nbyte, const char*addr)
{
	struct sockaddr_in target = s_addr_in;
	if (addr)
	{
		int err = inet_pton(AF_INET, addr, &target.sin_addr.s_addr);
		if (err != 1)
		{
			return -1;
		}
	}
	return (sendto(sock_id, (char*)buf, nbyte, 0, (sockaddr*)(void*)& target, sizeof(struct sockaddr_in)));
}
int UDPSocket::read(void *buf, unsigned nbyte)
{
    return (recvfrom(sock_id, (char *)buf, nbyte, 0, NULL, 0));
}
int UDPSocket::Read(void* buf, unsigned nbyte, char* ip)
{
	struct timeval stTimeOut;

	fd_set stReadFDS;

	FD_ZERO(&stReadFDS);

	// Timeout of one second
	stTimeOut.tv_sec = 0;
	stTimeOut.tv_usec = 0;

	FD_SET(sock_id, &stReadFDS);
	int retval = select(sock_id  + 1, &stReadFDS, 0, 0, &stTimeOut);
	static bool first = true;
	if (retval < 0)
	{
		if (first)
		{
			cerr << "select() failed on socket " << sock_id << endl;
			first = false;
		}
		return 0;
	}
	else if(retval)
	{
		struct sockaddr_in  cliaddr;
		memset(&cliaddr, 0, sizeof(cliaddr));
		socklen_t len = sizeof(cliaddr);
		int l = recvfrom(sock_id, (char*)buf, nbyte, 0, (struct sockaddr*) & cliaddr, &len);
		if (l <= 0)
		{
			return 0;
		}
		if (ip)
		{
			inet_ntop(AF_INET, &cliaddr.sin_addr.s_addr, ip, 16);
		}

		return l;
	}
	else return 0;
}

UDPSocket::~UDPSocket()
{
}
#ifdef HAVEMULTICAST
MulticastSocket::MulticastSocket(char *MulticastGroup, int p, int ttl_value)
{
    struct sockaddr_in addr;
    struct in_addr grpaddr;
    int i;
    this->connected = false;

    for (i = 0; i < sizeof(addr); i++)
        ((char *)&addr)[i] = 0;

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;

    //grpaddr.s_addr = inet_addr(MulticastGroup);
	int err = inet_pton(AF_INET, MulticastGroup, &grpaddr.s_addr);
	if (err != 1)
	{
		sock_id = -1;
		return;
	}

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
//       LOGINFO( "Could not set socket for nonblocking reads");
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
    errno = 0;
    if (bind(sock_id, (sockaddr *)(void *)&addr, sizeof(addr)) < 0)
    {
        fprintf(stderr, "binding of multicast socket to port %d failed: %s\n",
                p, coStrerror(getErrno()));
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
        LOGINFO("Could not add multicast address to socket");
        sock_id = -1;
        return;
    }

    // Put a limit on how far the packets can travel.
    unsigned char ttl = ttl_value;

    if (setsockopt(sock_id, IPPROTO_IP, IP_MULTICAST_TTL,
                   (char *)&ttl, sizeof(ttl)) < 0)
    {
        LOGINFO("Could not initialize ttl on socket");
        //sock_id= -1;
        //return;
    }
    // Turn looping off so packets will not go back to the same host.
    // This means that multiple instances of this program will not receive
    // packets from each other.
    /*    unsigned char loop = 0;
       if (setsockopt(sock_id, IPPROTO_IP, IP_MULTICAST_LOOP,
         &loop, sizeof(loop)) < 0) {
           LOGINFO( "Could not initialize socket");
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

#ifndef WIN32
const char *Socket::coStrerror(int err)
{
    return strerror(err);
}
#else
const char *Socket::coStrerror(int err)
{
    static char errorString[1000];

    LPVOID lpMsgBuf;
    if (FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            GetLastError(),
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
            (LPTSTR)&lpMsgBuf,
            0,
            NULL))
    {
        strcpy(errorString, reinterpret_cast<const char *>(lpMsgBuf));
    }
    else
    {
        strcpy(errorString, "FormatMessage API failed");
    }
    LocalFree(lpMsgBuf);
    return errorString;
}
#endif

#ifdef HAVE_OPENSSL
SSLSocket::SSLSocket()
{
    port = 0;
    host = NULL;
    this->connected = false;

    sock_id = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock_id < 0)
    {
        fprintf(stderr, "creation of socket failed!");
        sock_id = -1;
        return;
    }

    setTCPOptions();

#if defined(SO_REUSEADDR)
    int on = 1;
    if (setsockopt(sock_id, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)) < 0)
    {
        LOGINFO("setsockopt error SOL_SOCKET, SO_REUSEADDR");
    }
#endif

#ifdef DEBUG
    print();
#endif

    memset((char *)&s_addr_in, 0, sizeof(s_addr_in));
    s_addr_in.sin_family = AF_INET;
    s_addr_in.sin_addr.s_addr = INADDR_ANY;
    s_addr_in.sin_port = htons(FirewallConfig::the()->sourcePort);

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
            fprintf(stderr, "error with bind (socket to %s:%d): %s\n", host->getAddress(), s_addr_in.sin_port, strerror(errno));
            closesocket(sock_id);
            sock_id = -1;
            return;
        }
    }
    FirewallConfig::the()->sourcePort++;
}

SSLSocket::SSLSocket(int p, SSL *ssl)
    : Socket(p)
{
    mSSLObject = ssl;
}

SSLSocket::SSLSocket(int *p, SSL *ssl)
    : Socket(p)
{
    mSSLObject = ssl;
}

SSLSocket::SSLSocket(Host *h, int p, int retries, double timeout, SSL *ssl)
    : Socket(h, p, retries, timeout)
{
    mSSLObject = ssl;
}

SSLSocket::SSLSocket(int socket_id, sockaddr_in *sockaddr, SSL *ssl)
    : Socket(socket_id, sockaddr)
{
    mPeer = *sockaddr;
    mSSLObject = ssl;
}

SSLSocket::~SSLSocket()
{
    closesocket(sock_id);
}

std::string SSLSocket::getPeerAddress()
{

	char buf[100];
	if (!inet_ntop(AF_INET, &mPeer.sin_addr, buf, sizeof(buf)))
	{
		cerr << "SSLSocket::getPeerAddress(): address    = <unknown>" << endl;
	}
    return std::string(buf);
}

int SSLSocket::read(void *buf, unsigned int nbyte)
{
    int no_of_bytes = 0;
    cerr << "SSLSocket::read(): entered..." << endl;

    do
    {
        errno = 0;
        cerr << "SSLSocket::read(): Read data" << endl;
        no_of_bytes = SSL_read(mSSLObject, buf, nbyte);
    } while (no_of_bytes == -1 && (errno == EAGAIN || errno == EINTR));

    if (no_of_bytes <= 0)
    {
        if (SSL_get_shutdown(mSSLObject) == SSL_RECEIVED_SHUTDOWN)
        {
            cerr << "SSLSocket::read(): Peer notified shutdown!" << endl;
            return no_of_bytes;
        }
        cerr << "SSLSocket::read(): error!" << endl;
        checkSSLError(mSSLObject, no_of_bytes);
        cerr << "SSLSocket::read(): ...left" << endl;
        return (-1); // error
    }

    cerr << "SSLSocket::read(): ...left!" << endl;
    return no_of_bytes;
}
int SSLSocket::Read(void *buf, unsigned int nbyte, char *ip)
{
    return this->read(buf, nbyte);
}

int SSLSocket::write(const void *buf, unsigned int nbyte)
{
    int no_of_bytes = 0;
    char tmp_str[255];
    do
    {
        //    covise_time->mark(__LINE__, "vor Socket::write");

        no_of_bytes = SSL_write(mSSLObject, buf, nbyte);

    } while ((no_of_bytes <= 0) && ((errno == EAGAIN) || (errno == EINTR) || (errno == 2 /* no error */)));

    if (no_of_bytes <= 0)
    {
        checkSSLError(mSSLObject, no_of_bytes);

#ifdef _WIN32
        sprintf(tmp_str, "Socket send error = %d", WSAGetLastError());
        LOGERROR(tmp_str);
#else
        if (errno == EPIPE)
            return COVISE_SOCKET_INVALID;
        if (errno == ECONNRESET)
            return COVISE_SOCKET_INVALID;
        sprintf(tmp_str, "Socket write error = %d: %s, no_of_bytes = %d", errno, strerror(errno), no_of_bytes);
        LOGERROR(tmp_str);
#endif
        fprintf(stderr, "error writing on socket to %s:%d: %s\n", host->getAddress(), port, strerror(errno));
        LOGERROR("write returns <= 0: close socket.");
        return COVISE_SOCKET_INVALID;
    }

    return no_of_bytes;
}

int SSLSocket::connect(sockaddr_in addr /*, int retries, double timeout*/)
{
    try
    {
        int ret = ::connect(sock_id, (const sockaddr *)&addr, sizeof(addr));
        if (ret != 0)
        {
            coStrerror(errno);
        }
    }
    catch (std::exception &ex)
    {
        cerr << "Connect failed!" << endl;
        cerr << "Exception Detail: " << endl;
        cerr << ex.what() << endl;
        return -1;
    }

    return 0;
}

void SSLSocket::setSSL(SSL *pSSL)
{
    mSSLObject = pSSL;
}

SSLServerConnection *SSLSocket::spawnConnection(SSLConnection::PasswordCallback *cb, void *userData)
{
    cerr << "SSLSocket::spawnConnection(): Spawn new connection!" << endl;
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    int new_sock_id = (int)::accept(sock_id, (struct sockaddr *)&client_addr, &client_addr_len);

    //unsigned long state=0;
    //int err = 0;
    /*if ((err = ioctlsocket(new_sock_id, FIONBIO, &state))!=0)
   {
      resolveError();
      cerr << "SSLSocket::spawnConnection(): Making socket blocking failed" << endl;
      cerr << "SSLSocket::spawnConnection(): Error-Number: " << err << endl;
   }*/

    cerr << "SSLSocket::spawnConnection(): New socket:" << endl;
	char buf[100];
	if (!inet_ntop(AF_INET, &client_addr.sin_addr, buf, sizeof(buf)))
	{
		cerr << "SSLSocket::spawnConnection(): address    = <unknown>" << endl;
	}
    cerr << "SSLSocket::spawnConnection(): address    = " << buf << endl;
    cerr << "SSLSocket::spawnConnection(): port       = " << ntohs(client_addr.sin_port) << endl;
    cerr << "SSLSocket::spawnConnection(): Socket-ID  = " << new_sock_id << endl;
    cerr << "SSLSocket::spawnConnection(): ServSockID = " << sock_id << endl;

#ifdef WIN32
    if (new_sock_id == INVALID_SOCKET)
#else
    if (new_sock_id == -1) //-1 might be the same as Windows's INVALID_SOCKET needs to be checked
#endif
    {
        coPerror("Accept for incoming connection failed:\n");
        resolveError();
        return NULL;
    }

    //Need to retrieve connect data from addr struct
    cerr << "SSLSocket::spawnConnection(): Spawn new Socket based on spawned socket id" << endl;
    SSLSocket *sock = new SSLSocket(new_sock_id, &client_addr, mSSLObject);
    cerr << "SSLSocket::spawnConnection(): Spawn new Connection based on spawned socket" << endl;
    SSLServerConnection *locConn = new SSLServerConnection(sock, cb, userData);

    return locConn;
}
#endif

//extern functions have to be defined in the namespace
namespace covise
{

#ifdef HAVE_OPENSSL
bool checkSSLError(SSL *ssl, int error)
{
    std::string sError = "";

    switch (SSL_get_error(ssl, error))
    {
    case SSL_ERROR_ZERO_RETURN:
    {
        sError = "The TLS/SSL connection has been closed. If the protocol version is SSL 3.0 or TLS 1.0, this result code is returned only if a closure alert has occurred in the protocol, i.e. if the connection has been closed cleanly. Note that in this case SSL_ERROR_ZERO_RETURN does not necessarily indicate that the underlying transport has been closed.";
    }
    break;
    case SSL_ERROR_SYSCALL:
    {
        sError = "Some I/O error occurred. The OpenSSL error queue may contain more information on the error. If the error queue is empty (i.e. ERR_get_error() returns 0), ret can be used to find out more about the error: If ret == 0, an EOF was observed that violates the protocol. If ret == -1, the underlying BIO reported an I/O error (for socket I/O on Unix systems, consult errno for details).";
    }
    break;
    case SSL_ERROR_SSL:
    {
        sError = "A failure in the SSL library occurred, usually a protocol error. The OpenSSL error queue contains more information on the error.";
        cerr << "checkSSLError(): " << sError.c_str() << endl;
        unsigned long result = 0;
        char cmessage[240];
        while ((result = ERR_get_error()) != 0)
        {
            ERR_error_string_n(result, cmessage, 240);
            cerr << "checkSSLError(): " << cmessage << endl;
        }
    }
    break;
    case SSL_ERROR_WANT_CONNECT:
#ifdef SSL_ERROR_WANT_ACCEPT
    case SSL_ERROR_WANT_ACCEPT:
#endif
    {
        sError = "The operation did not complete; the same TLS/SSL I/O function should be called again later. The underlying BIO was not connected yet to the peer and the call would block in connect()/accept(). The SSL function should be called again when the connection is established. These messages can only appear with a BIO_s_connect() or BIO_s_accept() BIO, respectively. In order to find out, when the connection has been successfully established, on many platforms select() or poll() for writing on the socket file descriptor can be used.";
    }
    break;
    case SSL_ERROR_WANT_X509_LOOKUP:
    {
        sError = "The operation did not complete because an application callback set by SSL_CTX_set_client_cert_cb() has asked to be called again. The TLS/SSL I/O function should be called again later. Details depend on the application.";
    }
    break;
    case SSL_ERROR_WANT_READ:
    case SSL_ERROR_WANT_WRITE:
    {
        sError = "The operation did not complete; the same TLS/SSL I/O function should be called again later. If, by then, the underlying BIO has data available for reading (if the result code is SSL_ERROR_WANT_READ) or allows writing data (SSL_ERROR_WANT_WRITE), then some TLS/SSL protocol progress will take place, i.e. at least part of an TLS/SSL record will be read or written. Note that the retry may again lead to a SSL_ERROR_WANT_READ or SSL_ERROR_WANT_WRITE condition. There is no fixed upper limit for the number of iterations that may be necessary until progress becomes visible at application protocol level.\n";
        sError += "For socket BIOs (e.g. when SSL_set_fd() was used), select() or poll() on the underlying socket can be used to find out when the TLS/SSL I/O function should be retried. \n";
        sError += "Caveat: Any TLS/SSL I/O function can lead to either of SSL_ERROR_WANT_READ and SSL_ERROR_WANT_WRITE. In particular, SSL_read() or SSL_peek() may want to write data and SSL_write() may want to read data. This is mainly because TLS/SSL handshakes may occur at any time during the protocol (initiated by either the client or the server); SSL_read(), SSL_peek(), and SSL_write() will handle any pending handshakes.\n";
    }
    break;
    case SSL_ERROR_NONE:
    {
        sError = "The TLS/SSL I/O operation completed. This result code is returned if and only if ret > 0.";
    }
    break;
    }
    cerr << "checkSSLError(): " << sError.c_str() << endl;
    return true;
}
#endif

void resolveError()
{
    std::string sError = "";

#ifdef WIN32
    int error = WSAGetLastError();

    cerr << "resolveError(): Error-Code: " << error << endl;

    switch (error)
    {
    case WSANOTINITIALISED:
    {
        sError = "A successful WSAStartup call must occur before using this function.";
    }
    break;
    case WSAENETDOWN:
    {
        sError = "The network subsystem has failed.";
    }
    break;
    case WSAEADDRINUSE:
    {
        sError = "The socket's local address is already in use and the socket was not marked to allow address reuse with SO_REUSEADDR. This error usually occurs when executing bind, but could be delayed until this function if the bind was to a partially wildcard address (involving ADDR_ANY) and if a specific address needs to be committed at the time of this function.";
    }
    break;
    case WSAEINTR:
    {
        sError = "The blocking Windows Socket 1.1 call was canceled through WSACancelBlockingCall.";
    }
    break;
    case WSAEINPROGRESS:
    {
        sError = "A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function.";
    }
    break;
    case WSAEALREADY:
    {
        sError = "A nonblocking connect call is in progress on the specified socket. \nNote:\n  In order to preserve backward compatibility, this error is reported as WSAEINVAL to Windows Sockets 1.1 applications that link to either Winsock.dll or Wsock32.dll.";
    }
    break;
    case WSAEADDRNOTAVAIL:
    {
        sError = "The remote address is not a valid address (such as ADDR_ANY).";
    }
    break;
    case WSAEAFNOSUPPORT:
    {
        sError = "Addresses in the specified family cannot be used with this socket.";
    }
    break;
    case WSAECONNREFUSED:
    {
        sError = "The attempt to connect was forcefully rejected.";
    }
    break;
    case WSAEFAULT:
    {
        sError = "The name or the namelen parameter is not a valid part of the user address space, the namelen parameter is too small, or the name parameter contains incorrect address format for the associated address family.";
    }
    break;
    case WSAEINVAL:
    {
        sError = "The parameter s is a listening socket.";
    }
    break;
    case WSAEISCONN:
    {
        sError = "The socket is already connected (connection-oriented sockets only).";
    }
    break;
    case WSAENETUNREACH:
    {
        sError = "The network cannot be reached from this host at this time.";
    }
    break;
    case WSAEHOSTUNREACH:
    {
        sError = "A socket operation was attempted to an unreachable host.";
    }
    break;
    case WSAENOBUFS:
    {
        sError = "No buffer space is available. The socket cannot be connected.";
    }
    break;
    case WSAENOTSOCK:
    {
        sError = "The descriptor is not a socket.";
    }
    break;
    case WSAETIMEDOUT:
    {
        sError = "Attempt to connect timed out without establishing a connection.";
    }
    break;
    case WSAEWOULDBLOCK:
    {
        sError = "The socket is marked as nonblocking and the connection cannot be completed immediately.";
    }
    break;
    case WSAEACCES:
    {
        sError = "Attempt to connect datagram socket to broadcast address failed because setsockopt option SO_BROADCAST is not enabled.";
    }
    break;
    case WSAECONNRESET:
    {
        sError = "An existing connection was forcibly closed by the remote host. This normally results if the peer application on the remote host is suddenly stopped, the host is rebooted, the host or remote network interface is disabled, or the remote host uses a hard close (see setsockopt for more information on the SO_LINGER option on the remote socket). This error may also result if a connection was broken due to keep-alive activity detecting a failure while one or more operations are in progress. Operations that were in progress fail with WSAENETRESET. Subsequent operations fail with WSAECONNRESET.";
    }
    case WSAENOTCONN:
    {
        sError = "A request to send or receive data was disallowed because the socket is not connected and (when sending on a datagram socket using sendto) no address was supplied. Any other type of operation might also return this errorfor example, setsockopt setting SO_KEEPALIVE if the connection has been reset.";
    }
    default:
    {
        sError = "Error unkown!";
    }
    break;
    }
#endif
    cerr << sError.c_str() << endl;
}

#ifdef WIN32
/*
// List of Winsock error constants mapped to an interpretation string.
// Note that this list must remain sorted by the error constants'
// values, because we do a binary search on the list when looking up
// items.
static struct ErrorEntry {
    int nID;
    const char* pcMessage;

    ErrorEntry(int id, const char* pc = 0) : 
    nID(id), 
    pcMessage(pc) 
    { 
    }

    bool operator<(const ErrorEntry& rhs) 
    {
        return nID < rhs.nID;
    }
} gaErrorList[] = {
    ErrorEntry(0,                  "No error"),
    ErrorEntry(WSAEINTR,           "Interrupted system call"),
    ErrorEntry(WSAEBADF,           "Bad file number"),
    ErrorEntry(WSAEACCES,          "Permission denied"),
    ErrorEntry(WSAEFAULT,          "Bad address"),
    ErrorEntry(WSAEINVAL,          "Invalid argument"),
    ErrorEntry(WSAEMFILE,          "Too many open sockets"),
    ErrorEntry(WSAEWOULDBLOCK,     "Operation would block"),
    ErrorEntry(WSAEINPROGRESS,     "Operation now in progress"),
    ErrorEntry(WSAEALREADY,        "Operation already in progress"),
    ErrorEntry(WSAENOTSOCK,        "Socket operation on non-socket"),
    ErrorEntry(WSAEDESTADDRREQ,    "Destination address required"),
    ErrorEntry(WSAEMSGSIZE,        "Message too long"),
    ErrorEntry(WSAEPROTOTYPE,      "Protocol wrong type for socket"),
    ErrorEntry(WSAENOPROTOOPT,     "Bad protocol option"),
    ErrorEntry(WSAEPROTONOSUPPORT, "Protocol not supported"),
    ErrorEntry(WSAESOCKTNOSUPPORT, "Socket type not supported"),
    ErrorEntry(WSAEOPNOTSUPP,      "Operation not supported on socket"),
    ErrorEntry(WSAEPFNOSUPPORT,    "Protocol family not supported"),
    ErrorEntry(WSAEAFNOSUPPORT,    "Address family not supported"),
    ErrorEntry(WSAEADDRINUSE,      "Address already in use"),
    ErrorEntry(WSAEADDRNOTAVAIL,   "Can't assign requested address"),
    ErrorEntry(WSAENETDOWN,        "Network is down"),
    ErrorEntry(WSAENETUNREACH,     "Network is unreachable"),
    ErrorEntry(WSAENETRESET,       "Net connection reset"),
    ErrorEntry(WSAECONNABORTED,    "Software caused connection abort"),
    ErrorEntry(WSAECONNRESET,      "Connection reset by peer"),
    ErrorEntry(WSAENOBUFS,         "No buffer space available"),
    ErrorEntry(WSAEISCONN,         "Socket is already connected"),
    ErrorEntry(WSAENOTCONN,        "Socket is not connected"),
    ErrorEntry(WSAESHUTDOWN,       "Can't send after socket shutdown"),
    ErrorEntry(WSAETOOMANYREFS,    "Too many references, can't splice"),
    ErrorEntry(WSAETIMEDOUT,       "Connection timed out"),
    ErrorEntry(WSAECONNREFUSED,    "Connection refused"),
    ErrorEntry(WSAELOOP,           "Too many levels of symbolic links"),
    ErrorEntry(WSAENAMETOOLONG,    "File name too long"),
    ErrorEntry(WSAEHOSTDOWN,       "Host is down"),
    ErrorEntry(WSAEHOSTUNREACH,    "No route to host"),
    ErrorEntry(WSAENOTEMPTY,       "Directory not empty"),
    ErrorEntry(WSAEPROCLIM,        "Too many processes"),
    ErrorEntry(WSAEUSERS,          "Too many users"),
    ErrorEntry(WSAEDQUOT,          "Disc quota exceeded"),
    ErrorEntry(WSAESTALE,          "Stale NFS file handle"),
    ErrorEntry(WSAEREMOTE,         "Too many levels of remote in path"),
    ErrorEntry(WSASYSNOTREADY,     "Network system is unavailable"),
    ErrorEntry(WSAVERNOTSUPPORTED, "Winsock version out of range"),
    ErrorEntry(WSANOTINITIALISED,  "WSAStartup not yet called"),
    ErrorEntry(WSAEDISCON,         "Graceful shutdown in progress"),
    ErrorEntry(WSAHOST_NOT_FOUND,  "Host not found"),
    ErrorEntry(WSANO_DATA,         "No host data of that type was found")
};
const int kNumMessages = sizeof(gaErrorList) / sizeof(ErrorEntry);


//// WSAGetLastErrorMessage ////////////////////////////////////////////
// A function similar in spirit to Unix's perror() that tacks a canned 
// interpretation of the value of WSAGetLastError() onto the end of a
// passed string, separated by a ": ".  Generally, you should implement
// smarter error handling than this, but for default cases and simple
// programs, this function is sufficient.
//
// This function returns a pointer to an internal static buffer, so you
// must copy the data from this function before you call it again.  It
// follows that this function is also not thread-safe.

const char* WSAGetLastErrorMessage(const char* pcMessagePrefix)
{
    int nErrorID = WSAGetLastError();
    // Build basic error string
    static char acErrorBuffer[256];
    ostrstream outs(acErrorBuffer, sizeof(acErrorBuffer));
    outs << pcMessagePrefix << ": ";

    // Tack appropriate canned message onto end of supplied message 
    // prefix. Note that we do a binary search here: gaErrorList must be
	// sorted by the error constant's value.
	ErrorEntry* pEnd = gaErrorList + kNumMessages;
    ErrorEntry Target(nErrorID ? nErrorID : WSAGetLastError());
    ErrorEntry* it = lower_bound(gaErrorList, pEnd, Target);
    if ((it != pEnd) && (it->nID == Target.nID)) {
        outs << it->pcMessage;
    }
    else {
        // Didn't find error in list, so make up a generic one
        outs << "unknown error";
    }
    outs << " (" << Target.nID << ")";

    // Finish error message off and return it.
    outs << ends;
    acErrorBuffer[sizeof(acErrorBuffer) - 1] = '\0';
    return acErrorBuffer;
}*/

void coPerror(const char *prefix)
{
    fprintf(stderr, "%s (%d): %s\n", prefix, WSAGetLastError(), Socket::coStrerror(WSAGetLastError()));
}
#else

void coPerror(const char *prefix)
{
    perror(prefix);
}
#endif
}
#endif
