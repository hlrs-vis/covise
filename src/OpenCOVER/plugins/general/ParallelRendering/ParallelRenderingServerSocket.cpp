/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <getopt.h>
#include <arpa/inet.h>
#endif
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <util/byteswap.h>
#include <time.h>
#include <errno.h>

#include <cover/coVRPluginSupport.h>
#include <ParallelRenderingServerSocket.h>
#include "ParallelRenderingOGLTexQuadCompositor.h"

int ParallelRenderingServerSocket::server_connect(int port)
{

    struct addrinfo *res, *t;
    struct addrinfo hints;

    memset(&hints, 0, sizeof(hints));

    hints.ai_flags = AI_PASSIVE;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char *service;
    int sockfd = -1;
    int n, connfd;
#ifndef WIN32
    asprintf(&service, "%d", port);
#else
    service = new char[100];
    sprintf(service, "%d", port);
#endif
    n = getaddrinfo(NULL, service, &hints, &res);
    fprintf(stderr, "server_connect %d\n", port);
    if (n < 0)
    {
        fprintf(stderr, "%s for port %d\n", gai_strerror(n), port);
        return n;
    }

    for (t = res; t; t = t->ai_next)
    {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd >= 0)
        {
            n = 1;

            setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&n, sizeof n);

            if (!bind(sockfd, t->ai_addr, t->ai_addrlen))
                break;

            close(sockfd);
            sockfd = -1;
        }
    }

    freeaddrinfo(res);

    if (sockfd < 0)
    {
        fprintf(stderr, "Couldn't listen to port %d\n", port);
        return sockfd;
    }

    fprintf(stderr, "server accept\n");
    listen(sockfd, 1);
    connfd = accept(sockfd, NULL, 0);
    if (connfd < 0)
    {
        perror("error: server accept");
        fprintf(stderr, "accept() failed\n");
        close(sockfd);
        return connfd;
    }

    fprintf(stderr, "server accepted\n");

    close(sockfd);
    return connfd;
}

ParallelRenderingServerSocket::ParallelRenderingServerSocket(int numClients, bool compositorRenders)
    : ParallelRenderingServer(numClients, compositorRenders)
{

    fd = new int[numClients];
    pixels = new unsigned char *[numClients];
    dimension = new ParallelRenderingDimension[numClients];

    for (int index = startClient; index < numClients; index++)
    {
        fd[index] = 0;
        pixels[index] = 0;
        dimension[index].width = 0;
        dimension[index].height = 0;
    }

    connected = false;
    lock.lock();
}

ParallelRenderingServerSocket::~ParallelRenderingServerSocket()
{

    for (int index = startClient; index < numClients; index++)
    {
        delete pixels[index];
    }

    delete[] fd;
    delete[] dimension;
}

void ParallelRenderingServerSocket::acceptConnection()
{

    for (int index = startClient; index < numClients; index++)
    {

        int port = 18515 + index;
        if (!compositorRenders)
            port++;

        fd[index] = server_connect(port);
        if (fd[index] < 0)
        {
            cerr << "ParallelRenderingServerSocket::connect err: connection to client " << index << " failed" << endl;
            return;
        }
        /*
      int opts = fcntl(fd[index] ,F_GETFL);
      if (opts >= 0) {
         fcntl(fd[index], F_SETFL, opts | O_NONBLOCK);
      }
*/
    }

    connected = true;
}

void ParallelRenderingServerSocket::run()
{

    while (keepRunning)
    {

        lock.lock();
        receive();
    }
}

void ParallelRenderingServerSocket::receive()
{

    int *received = new int[numClients];
    for (int index = startClient; index < numClients; index++)
        received[index] = -1;

    fd_set socks;
    FD_ZERO(&socks);
    bool done = false;

    while (!done)
    {

        done = true;
        for (int index = startClient; index < numClients; index++)
        {
            if (received[index] == -1 || received[index] != dimension[index].width * dimension[index].height * 4)
            {
                FD_SET(fd[index], &socks);
                done = false;
            }
        }

        struct timeval timeout = { 0, 1000 };
        int n = select(fd[numClients - 1] + 1, &socks, NULL, NULL, &timeout);

        if (n > 0)
        {
            for (int index = startClient; index < numClients; index++)
            {
                if (FD_ISSET(fd[index], &socks))
                {
                    if (received[index] == -1)
                    {
                        int width, height;
                        read(fd[index], &width, sizeof(int));
                        read(fd[index], &height, sizeof(int));
                        if (dimension[index].width != width || dimension[index].height != height)
                        {
                            dimension[index].width = width;
                            dimension[index].height = height;

                            delete pixels[index];
                            pixels[index] = new unsigned char[width * height * 4];
                        }
                        received[index] = 0;
                    }
                    else
                    {
                        int r = read(fd[index], ((char *)pixels[index]) + received[index],
                                     dimension[index].width * dimension[index].height * 4 - received[index]);
                        received[index] += r;
                    }
                }
            }
        }
        else if (n < 0)
            perror("select");
    }

    delete received;

    renderLock.unlock();
}

void ParallelRenderingServerSocket::render()
{

    renderLock.lock();

    if (compositorRenders)
    {
        compositors[0]->setTexture(width, height, (unsigned char *)image);
        readBackImage();
    }

    for (int index = startClient; index < numClients; index++)
    {
        if (compositors[index])
        {

            if (!compositorRenders || index != 0)
                if (pixels[index])
                    compositors[index]->setTexture(dimension[index].width, dimension[index].height, pixels[index]);

            compositors[index]->render();
        }
    }

    lock.unlock();
}
