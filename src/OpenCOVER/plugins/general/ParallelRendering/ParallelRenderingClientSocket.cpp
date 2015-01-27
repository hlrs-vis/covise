/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/unixcompat.h>
#include <cover/coVRPluginSupport.h>
#include <ParallelRenderingClientSocket.h>
#ifdef WIN32
#include <Ws2tcpip.h>
#endif

int ParallelRenderingClientSocket::client_connect(const char *servername, int port)
{

    struct addrinfo *res, *t;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char *service;
    int n;
    int sockfd = -1;
#ifndef WIN32
    asprintf(&service, "%d", port);
#else
    service = new char[100];
    sprintf(service, "%d", port);
#endif
    n = getaddrinfo(servername, service, &hints, &res);

    if (n < 0)
    {
        fprintf(stderr, "%s for %s:%d\n", gai_strerror(n), servername, port);
        return n;
    }

    while (!connected)
    {
        fprintf(stderr, "connecting to %s:%d\n", servername, port);

        for (t = res; t; t = t->ai_next)
        {
            sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
            if (sockfd >= 0)
            {
                if (!connect(sockfd, t->ai_addr, t->ai_addrlen) == 0)
                {
                    close(sockfd);
                    sockfd = -1;
                }
            }
        }

        if (sockfd < 0)
        {
            fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
            usleep(1000);
        }
        else
        {
            fprintf(stderr, "connected to %s:%d\n", servername, port);
            connected = true;
        }
    }

    freeaddrinfo(res);

    return sockfd;
}

ParallelRenderingClientSocket::ParallelRenderingClientSocket(int number, const std::string &compositor)
    : ParallelRenderingClient(number, compositor)
{

    cerr << "ParallelRenderingClientSocket::<init> info: creating client " << number
         << " for compositor " << compositor << endl;

    this->externalPixelFormat = GL_BGRA;

    lock.lock();
}

ParallelRenderingClientSocket::~ParallelRenderingClientSocket()
{
}

void ParallelRenderingClientSocket::connectToServer()
{

    cerr << "ParallelRenderingClientSocket::connect info: connecting to compositor '"
         << compositor << "' as server " << number << endl;
    fd = client_connect(const_cast<char *>(compositor.c_str()), 18515 + number);
}

void ParallelRenderingClientSocket::run()
{

    cerr << "ParallelRenderingClientSocket::run info: starting client " << number << endl;

    while (keepRunning)
    {
        lock.lock();
        int bytesWritten = 0;

        int w = width;
        int h = height;

        write(fd, &w, sizeof(int));
        write(fd, &h, sizeof(int));

        while (bytesWritten != w * h * 4)
        {
            int n = write(fd, image + bytesWritten, w * h * 4 - bytesWritten);
            bytesWritten += n;
            //printf(" wrote %d bytes\n", bytesWritten);
        }
        //printf(" wrote %d bytes\n", bytesWritten);
        sendLock.unlock();
    }
}

void ParallelRenderingClientSocket::send()
{

    sendLock.lock();
    lock.unlock();
}
