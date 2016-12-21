/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Copyright (C) 2000-2013 Clemens Fuchslocher <clemens@vakuumverpackt.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 * USA
 *
 */

#include <errno.h>
#ifdef _WIN32
#include <util/XGetOpt.h>
#else
#include <getopt.h>
#endif
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#ifdef WIN32
#include <winsock2.h>
#else
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include "rTcpClient.h"
#include "tcpsock.h"

struct struct_rc rc;
struct struct_options options;
struct struct_settings settings = { 0, 0, 0, 0, 0, 0, 0, 0 };

static struct option long_options[] = {
    { "tunnel-port", required_argument, NULL, TUNNEL_PORT_OPTION },
    { "tunnel-host", required_argument, NULL, TUNNEL_HOST_OPTION },
    { "remote-port", required_argument, NULL, REMOTE_PORT_OPTION },
    { "remote-host", required_argument, NULL, REMOTE_HOST_OPTION },
    { "buffer-size", required_argument, NULL, BUFFER_SIZE_OPTION },
#ifndef __MINGW32__
    { "fork", no_argument, NULL, FORK_OPTION },
#endif
    { "log", no_argument, NULL, LOG_OPTION },
    { "stay-alive", no_argument, NULL, STAY_ALIVE_OPTION },
    { "help", no_argument, NULL, HELP_OPTION },
    { "version", no_argument, NULL, VERSION_OPTION },
    { 0, 0, 0, 0 }
};

int main(int argc, char *argv[])
{
#ifdef __MINGW32__
    WSADATA info;
    if (WSAStartup(MAKEWORD(1, 1), &info) != 0)
    {
        perror("main: WSAStartup()");
        exit(1);
    }
#endif

    name = argv[0];

    set_options(argc, argv);
    do
    {

        if (build_tunnel() == 1)
        {
            exit(1);
        }

        char buffer[100];
        for (int i = 0; i < 100; i++)
            buffer[i] = '\0';
        bool connected = false;
        do
        {
            if (read(rc.tunnel_socket, buffer, 7) != 7)
                perror("handle_client: read failed");
            fprintf(stderr, "%s\n", buffer);

#ifndef __MINGW32__
            signal(SIGCHLD, SIG_IGN);
#endif

            if (connectRemote() == 0)
            {
                if (write(rc.tunnel_socket, "success", 7) != 7)
                    perror("handle_client: write failed");
                use_tunnel();
                connected = true;
            }
            else
            {
                if (write(rc.tunnel_socket, "fail   ", 7) != 7)
                    perror("handle_client: write failed");
            }
        } while (!connected);

        //close(rc.remote_socket);
    } while (settings.stay_alive);

    close(rc.tunnel_socket);

    return 0;
}

void set_options(int argc, char *argv[])
{
    int opt;
    int index;

    options.buffer_size = 4096;

    do
    {
        opt = getopt_long(argc, argv, "", long_options, &index);
        switch (opt)
        {
        case TUNNEL_PORT_OPTION:
        {
            options.tunnel_port = optarg;
            settings.tunnel_port = 1;
            break;
        }
        case TUNNEL_HOST_OPTION:
        {
            options.tunnel_host = optarg;
            settings.tunnel_host = 1;
            break;
        }

        case REMOTE_PORT_OPTION:
        {
            options.remote_port = optarg;
            settings.remote_port = 1;
            break;
        }

        case REMOTE_HOST_OPTION:
        {
            options.remote_host = optarg;
            settings.remote_host = 1;
            break;
        }

        case BUFFER_SIZE_OPTION:
        {
            options.buffer_size = atoi(optarg);
            settings.buffer_size = 1;
            break;
        }

        case FORK_OPTION:
        {
            settings.fork = 1;
            settings.stay_alive = 1;
            break;
        }

        case LOG_OPTION:
        {
            settings.log = 1;
            break;
        }

        case STAY_ALIVE_OPTION:
        {
            settings.stay_alive = 1;
            break;
        }

        case HELP_OPTION:
        {
            print_usage();
            print_help();
            exit(0);
        }

        case VERSION_OPTION:
        {
            print_version();
            exit(0);
        }

        case '?':
        {
            print_usage();
            print_helpinfo();
            exit(0);
        }
        }
    } while (opt != -1);

    if (!settings.tunnel_port)
    {
        print_missing("missing '--tunnel-port=' option.");
        exit(1);
    }

    if (!settings.tunnel_host)
    {
        print_missing("missing '--tunnel-host=' option.");
        exit(1);
    }

    if (!settings.remote_port)
    {
        print_missing("missing '--remote-port=' option.");
        exit(1);
    }

    if (!settings.remote_host)
    {
        print_missing("missing '--remote-host=' option.");
        exit(1);
    }
}

int build_tunnel(void)
{
    rc.tunnel_host = gethostbyname(options.tunnel_host);
    if (rc.tunnel_host == NULL)
    {
        perror("build_tunnel: gethostbyname()");
        return 1;
    }

    memset(&rc.tunnel_addr, 0, sizeof(rc.tunnel_addr));

    rc.tunnel_addr.sin_family = AF_INET;
    rc.tunnel_addr.sin_port = htons(atoi(options.tunnel_port));

    memcpy(&rc.tunnel_addr.sin_addr.s_addr, rc.tunnel_host->h_addr, rc.tunnel_host->h_length);

    rc.tunnel_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (rc.tunnel_socket < 0)
    {
        perror("build_tunnel: socket()");
        return 1;
    }

    if (connect(rc.tunnel_socket, (struct sockaddr *)&rc.tunnel_addr, sizeof(rc.tunnel_addr)) < 0)
    {
        perror("build_tunnel: connect()");
        return 1;
    }

    disable_nagle(rc.tunnel_socket, "tunnel socket");

    return 0;
}
int connectRemote(void)
{
    const double timeout = 6;

    rc.remote_host = gethostbyname(options.remote_host);
    if (rc.remote_host == NULL)
    {
        perror("connectRemote: gethostbyname()");
        return 1;
    }

    memset(&rc.remote_addr, 0, sizeof(rc.remote_addr));

    rc.remote_addr.sin_family = AF_INET;
    rc.remote_addr.sin_port = htons(atoi(options.remote_port));

    memcpy(&rc.remote_addr.sin_addr.s_addr, rc.remote_host->h_addr, rc.remote_host->h_length);

    rc.remote_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (rc.remote_socket < 0)
    {
        perror("connectRemote: socket()");
        return 1;
    }

    struct timeval tv_start;
    if (gettimeofday(&tv_start, NULL) < 0)
    {
        perror("connectRemote: gettimeofday start");
        close(rc.remote_socket);
        return 1;
    }
    double start = tv_start.tv_sec + tv_start.tv_usec * 1e-6;

    while (connect(rc.remote_socket, (struct sockaddr *)&rc.remote_addr, sizeof(rc.remote_addr)) < 0)
    {
        if (errno == ECONNREFUSED || errno == EADDRINUSE || errno == ETIMEDOUT)
        {
            struct timeval tv_now;
            if (gettimeofday(&tv_now, NULL) < 0)
            {
                perror("connectRemote: gettimeofday now");
                close(rc.remote_socket);
                return 1;
            }
            double now = tv_now.tv_sec + tv_now.tv_usec * 1e-6;
            if (now - start > timeout)
            {
                perror("connectRemote: timeout");
                close(rc.remote_socket);
                return 1;
            }
            fprintf(stdout, ".");
            fflush(stdout);
            usleep(10000);
        }
        else
        {
            perror("connectRemote: connect()");
            close(rc.remote_socket);
            return 1;
        }
    }

    disable_nagle(rc.remote_socket, "remote socket");

    return 0;
}

int use_tunnel(void)
{
    fd_set io;
    char buffer[options.buffer_size];

    for (;;)
    {
        FD_ZERO(&io);
        FD_SET(rc.tunnel_socket, &io);
        FD_SET(rc.remote_socket, &io);

        memset(buffer, 0, sizeof(buffer));

        if (select(fd(), &io, NULL, NULL, NULL) < 0)
        {
            perror("use_tunnel: select()");
            break;
        }

        if (FD_ISSET(rc.tunnel_socket, &io))
        {
            int count = recv(rc.tunnel_socket, buffer, sizeof(buffer), 0);
            if (count < 0)
            {
                perror("use_tunnel: recv(rc.tunnel_socket)");
                close(rc.tunnel_socket);
                close(rc.remote_socket);
                return 1;
            }

            if (count == 0)
            {
                close(rc.tunnel_socket);
                close(rc.remote_socket);
                return 0;
            }

            send(rc.remote_socket, buffer, count, 0);

            if (settings.log)
            {
                printf("> %s > ", get_current_timestamp());
                fwrite(buffer, sizeof(char), count, stdout);
                fflush(stdout);
            }
        }

        if (FD_ISSET(rc.remote_socket, &io))
        {
            int count = recv(rc.remote_socket, buffer, sizeof(buffer), 0);
            if (count < 0)
            {
                perror("use_tunnel: recv(rc.remote_socket)");
                close(rc.tunnel_socket);
                close(rc.remote_socket);
                return 1;
            }

            if (count == 0)
            {
                close(rc.tunnel_socket);
                close(rc.remote_socket);
                return 0;
            }

            send(rc.tunnel_socket, buffer, count, 0);

            if (settings.log)
            {
                fwrite(buffer, sizeof(char), count, stdout);
                fflush(stdout);
            }
        }
    }

    return 0;
}

int fd(void)
{
    unsigned int fd = rc.tunnel_socket;
    if (fd < rc.remote_socket)
    {
        fd = rc.remote_socket;
    }
    return fd + 1;
}

char *get_current_timestamp(void)
{
    static char date_str[20];
    time_t date;

    time(&date);
    strftime(date_str, sizeof(date_str), "%Y-%m-%d %H:%M:%S", localtime(&date));
    return date_str;
}

void print_usage(void)
{
    fprintf(stderr, "Usage: %s [options]\n\n", name);
}

void print_helpinfo(void)
{
    fprintf(stderr, "Try `%s --help' for more options\n", name);
}

void print_help(void)
{
    fprintf(stderr, "\
         Options:\n\
         --version\n\
         --help\n\n\
         --tunnel-port=PORT    local port\n\
         --tunnel-host=PORT    local host\n\
         --remote-port=PORT   remote port\n\
         --remote-host=HOST   remote host\n\
         --buffer-size=BYTES  buffer size\n"
#ifndef __MINGW32__
                    "  --fork               fork-based concurrency\n"
#endif
                    "  --log\n\
         --stay-alive\n\n\
         \n");
}

void print_version(void)
{
    fprintf(stderr, "\
         tcptunnel v" VERSION " Copyright (C) 2000-2013 Clemens Fuchslocher\n\n\
         This program is distributed in the hope that it will be useful,\n\
         but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
         MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\
         GNU General Public License for more details.\n\n\
         Written by Clemens Fuchslocher <clemens@vakuumverpackt.de>\n\
         ");
}

void print_missing(const char *message)
{
    print_usage();
    fprintf(stderr, "%s: %s\n", name, message);
    print_helpinfo();
}
