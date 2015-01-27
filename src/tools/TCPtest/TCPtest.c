/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <strings.h>

typedef unsigned short uint16;

/*************** Nameserver request ****************/

#ifdef _CRAY
#define CONV (char *)
#else
#define CONV
#endif

static unsigned int getIP(const char *name)
{
    struct hostent *hostinfo;
    hostinfo = gethostbyname(CONV name); /* Hack for Cray */
    if (hostinfo)

#ifndef _CRAY
        return *((unsigned int *)*hostinfo->h_addr_list);
#else
    {
        unsigned char *x = (unsigned char *)*hostinfo->h_addr_list;
        return ((*x) << 24) | (*(x + 1) << 16) | (*(x + 2) << 8) | *(x + 3);
    }
#endif

    else
    {
        fprintf(stderr, "Nameserver didn't find '%s'", name);
        return INADDR_ANY;
    }
}

/*************** Inet SERVER ****************/

static int inetserver(uint16 port)
{
    struct sockaddr_in ServAddr, CliAddr;
    int xsoc, CliAddrLen, soc;
    fprintf(stderr, "Creating socket ...\n");
    xsoc = socket(AF_INET, SOCK_STREAM, 0);
    if (xsoc == -1)
    {
        perror("InetServer");
        return -1;
    }
    bzero((char *)&ServAddr, sizeof(ServAddr));
    ServAddr.sin_family = AF_INET;
    ServAddr.sin_addr.s_addr = INADDR_ANY;
    ServAddr.sin_port = htons(port);

    fprintf(stderr, "Bind address %s to socket ...\n", inet_ntoa(ServAddr.sin_addr));
    if (bind(xsoc, (struct sockaddr *)&ServAddr, sizeof(ServAddr)))
    {
        close(xsoc);
        perror("InetServer");
        return -1;
    }

    fprintf(stderr, "Listen ...\n");
    if (listen(xsoc, 5))
    {
        perror("InetServer");
        return -1;
    }

    bzero((char *)&CliAddr, sizeof(CliAddr));
    CliAddrLen = sizeof(CliAddr);

    fprintf(stderr, "Accept ...\n");
#ifdef _IBM_
    size_t CliAddrLen_ibm = CliAddrLen;
    if (-1 == (soc = accept(xsoc, (struct sockaddr *)&CliAddr, &CliAddrLen_ibm)))
#endif

#ifndef _IBM_
        if (-1 == (soc = accept(xsoc, (struct sockaddr *)&CliAddr, &CliAddrLen)))
#endif
        {
            perror("InetServer");
            return -1;
        }
    close(xsoc);

    fprintf(stderr, "Accepted Connection from %s\n", inet_ntoa(CliAddr.sin_addr));

    return soc;
}

/*************** Inet CLIENT ****************/

static int inetclient(uint16 port, const char *host, int clientPort)
{
    struct sockaddr_in ServAddr, CliAddr;
    int soc, cval;

    fprintf(stderr, "Creating socket ...\n");
    soc = socket(AF_INET, SOCK_STREAM, 0);
    if (soc == -1)
    {
        perror("InetClient");
        return -1;
    }

    if (clientPort != -1)
    {
        bzero((char *)&ServAddr, sizeof(ServAddr));
        CliAddr.sin_family = AF_INET;
        CliAddr.sin_addr.s_addr = INADDR_ANY;
        CliAddr.sin_port = htons(clientPort);

        fprintf(stderr, "Bind address %s to socket ...\n", inet_ntoa(ServAddr.sin_addr));
        if (bind(soc, (struct sockaddr *)&CliAddr, sizeof(ServAddr)))
        {
            close(soc);
            perror("InetClient");
            return -1;
        }
    }

    bzero((char *)&ServAddr, sizeof(ServAddr));
    ServAddr.sin_family = AF_INET;
    ServAddr.sin_port = htons(port);
    ServAddr.sin_addr.s_addr = getIP(host);

    fprintf(stderr, "Connect to %s Port %d ...\n", inet_ntoa(ServAddr.sin_addr), ntohs(ServAddr.sin_port));

    cval = connect(soc, (struct sockaddr *)&ServAddr, sizeof(ServAddr));
    if (cval == -1)
    {
        close(soc);
        perror("InetClient");
        return -1;
    }
    else
        return soc;
}

void usageExit(char *argv[])
{
    fprintf(stderr, "Call:  %s client <hostname> <port #> [ <local Port> ]\n", argv[0]);
    fprintf(stderr, "       %s server <port #>\n", argv[0]);
    exit(-1);
}

int main(int argc, char *argv[])
{

    if (argc < 3)
        usageExit(argv);

    /********************* Client test *********************/

    if (*argv[1] == 'C' || *argv[1] == 'c')
    {
        int soc;
        int port;
        int clientPort;
        char host[256];

        /* get and check input parameters */
        if (argc > 5 || argc < 4)
            usageExit(argv);

        sscanf(argv[2], "%s", host);
        sscanf(argv[3], "%d", &port);

        if (argc == 5)
        {
            sscanf(argv[4], "%d", &clientPort);
            if (port < 0 || port > 65535 || clientPort < 0 || clientPort > 65535)
            {
                fprintf(stderr, "Port must be 0...65535\n");
                exit(0);
            }
        }
        else
            clientPort = -1;

        if (port < 1024)
            fprintf(stderr, "Warning: using priviledged port!\n");

        /* open connection */
        soc = inetclient(port, host, clientPort);
        if (soc < 0)
            exit(-1);

        /* write test */
        if (write(soc, "C", 1) < 1)
        {
            fprintf(stderr, "Could not send 1 Bytes!\n");
            exit(-1);
        }

        /* read test */
        if (read(soc, host, 1) < 1)
        {
            fprintf(stderr, "Did not receive 1 Bytes!\n");
            exit(-1);
        }

        fprintf(stderr, "Seems OK, press Return to quit!\n");
        gets(host);

        close(soc);
        exit(0);
    }

    /********************* Server test *********************/

    else if (*argv[1] == 'S' || *argv[1] == 's')
    {
        int soc;
        int port;
        char buf[2];

        /* get and check input parameters */
        if (argc != 3)
            usageExit(argv);
        sscanf(argv[2], "%d", &port);
        if (port < 0 || port > 65535)
        {
            fprintf(stderr, "Port must be 0...65535\n");
            exit(0);
        }
        if (port < 1024)
            fprintf(stderr, "Warning: using priviledged port!\n");

        /* open connection */
        soc = inetserver(port);
        if (soc < 0)
            exit(-1);

        /* read test */
        if (read(soc, buf, 1) < 1)
        {
            fprintf(stderr, "Did not receive 1 Bytes!\n");
            exit(-1);
        }
        /* write test */
        if (write(soc, "S", 1) < 1)
        {
            fprintf(stderr, "Could not send 1 Bytes!\n");
            exit(-1);
        }

        fprintf(stderr, "Seems OK, press Return to quit!\n");
        gets(buf);

        close(soc);
        exit(0);
    }

    /* Haeee ? */
    else
    {
        fprintf(stderr, "Call:  %s client <hostname> <port #>\n", argv[0]);
        fprintf(stderr, " %s server <port #>\n", argv[0]);
    }

    return 0;
}
