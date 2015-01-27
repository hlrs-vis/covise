/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "tcpsock.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <errno.h>
#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

int disable_nagle(int fd, const char *desc)
{
    errno = 0;
    int optval = 1;
#if defined __MINGW32__ || defined WIN32
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&optval, sizeof(optval)) < 0)
#else
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) < 0)
#endif
    {
        std::string msg = "disable_nagle: setsockopt(TCP_NODELAY) - ";
        msg += desc;
        perror(msg.c_str());
        return 1;
    }

    return 0;
}
