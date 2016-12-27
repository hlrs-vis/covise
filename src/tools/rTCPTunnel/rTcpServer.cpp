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
#include <unistd.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include "rTcpServer.h"
#include "tcpsock.h"

struct struct_rc rc;
struct struct_options options;
struct struct_settings settings = { 0, 0, 0, 0, 0, 0, 0 };

static struct option long_options[] = {
    { "local-port", required_argument, NULL, LOCAL_PORT_OPTION },
    { "remote-port", required_argument, NULL, REMOTE_PORT_OPTION },
    { "bind-address", required_argument, NULL, BIND_ADDRESS_OPTION },
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

    if (build_remote_server() == 1)
    {
        exit(1);
    }
    rc.server_socket = 0;
    do
    {

#ifndef __MINGW32__
        signal(SIGCHLD, SIG_IGN);
#endif

        fprintf(stderr, "wrc\n");
        if (wait_for_remote_clients() == 0)
        {
            if (rc.server_socket == 0)
            {
                fprintf(stderr, "bs\n");
                if (build_server() == 1)
                {
                    exit(1);
                }
            }
            fprintf(stderr, "wc\n");
            while (wait_for_clients() == 0 && handle_client() < 0)
                ;
        }
    } while (settings.stay_alive);

    close(rc.server_socket);

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
        case LOCAL_PORT_OPTION:
        {
            options.local_port = optarg;
            settings.local_port = 1;
            break;
        }

        case REMOTE_PORT_OPTION:
        {
            options.remote_port = optarg;
            settings.remote_port = 1;
            break;
        }

        case BIND_ADDRESS_OPTION:
        {
            options.bind_address = optarg;
            settings.bind_address = 1;
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

    if (!settings.local_port)
    {
        print_missing("missing '--local-port=' option.");
        exit(1);
    }

    if (!settings.remote_port)
    {
        print_missing("missing '--remote-port=' option.");
        exit(1);
    }
}

int build_server(void)
{
    memset(&rc.server_addr, 0, sizeof(rc.server_addr));

    rc.server_addr.sin_port = htons(atoi(options.local_port));
    rc.server_addr.sin_family = AF_INET;
    rc.server_addr.sin_addr.s_addr = INADDR_ANY;

    rc.server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (rc.server_socket < 0)
    {
        perror("build_remote_server: socket()");
        return 1;
    }

    int optval = 1;
#ifdef __MINGW32__
    if (setsockopt(rc.server_socket, SOL_SOCKET, SO_REUSEADDR, (const char *)&optval, sizeof(optval)) < 0)
#else
    if (setsockopt(rc.server_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0)
#endif
    {
        perror("build_server: setsockopt(SO_REUSEADDR)");
        return 1;
    }

    if (settings.bind_address)
    {
        rc.server_addr.sin_addr.s_addr = inet_addr(options.bind_address);
    }

    if (bind(rc.server_socket, (struct sockaddr *)&rc.server_addr, sizeof(rc.server_addr)) < 0)
    {
        perror("build_server: bind()");
        return 1;
    }

    if (listen(rc.server_socket, 1) < 0)
    {
        perror("build_server: listen()");
        return 1;
    }

    return 0;
}

int build_remote_server(void)
{
    memset(&rc.remote_server_addr, 0, sizeof(rc.remote_server_addr));

    rc.remote_server_addr.sin_port = htons(atoi(options.remote_port));
    rc.remote_server_addr.sin_family = AF_INET;
    rc.remote_server_addr.sin_addr.s_addr = INADDR_ANY;

    rc.remote_server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (rc.remote_server_socket < 0)
    {
        perror("build_remote_server: socket()");
        return 1;
    }

    int optval = 1;
#ifdef __MINGW32__
    if (setsockopt(rc.remote_server_socket, SOL_SOCKET, SO_REUSEADDR, (const char *)&optval, sizeof(optval)) < 0)
#else
    if (setsockopt(rc.remote_server_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0)
#endif
    {
        perror("build_sremote_erver: setsockopt(SO_REUSEADDR)");
        return 1;
    }

    if (settings.bind_address)
    {
        rc.remote_server_addr.sin_addr.s_addr = inet_addr(options.bind_address);
    }

    if (bind(rc.remote_server_socket, (struct sockaddr *)&rc.remote_server_addr, sizeof(rc.remote_server_addr)) < 0)
    {
        perror("build_remote_server: bind()");
        return 1;
    }

    if (listen(rc.remote_server_socket, 1) < 0)
    {
        perror("build_remote_server: listen()");
        return 1;
    }

    return 0;
}

int wait_for_remote_clients(void)
{
#ifndef __MINGW32__
    unsigned int client_addr_size;
#else
    int client_addr_size;
#endif

    client_addr_size = sizeof(struct sockaddr_in);

    rc.remote_client_socket = accept(rc.remote_server_socket, (struct sockaddr *)&rc.remote_client_addr, &client_addr_size);
    if (rc.remote_client_socket < 0)
    {
        if (errno != EINTR)
        {
            perror("wait_for_remote_clients: accept()");
        }
        return 1;
    }

    if (settings.log)
    {
        printf("> %s tcptunnel: request from %s\n", get_current_timestamp(), inet_ntoa(rc.remote_client_addr.sin_addr));
    }

    disable_nagle(rc.remote_client_socket, "remote client socket");

    return 0;
}
int wait_for_clients(void)
{
#ifndef __MINGW32__
    unsigned int client_addr_size;
#else
    int client_addr_size;
#endif

    client_addr_size = sizeof(struct sockaddr_in);

    rc.client_socket = accept(rc.server_socket, (struct sockaddr *)&rc.client_addr, &client_addr_size);
    if (rc.client_socket < 0)
    {
        if (errno != EINTR)
        {
            perror("wait_for_clients: accept()");
        }
        return 1;
    }

    if (settings.log)
    {
        printf("> %s tcptunnel: request from %s\n", get_current_timestamp(), inet_ntoa(rc.client_addr.sin_addr));
    }

    disable_nagle(rc.client_socket, "client socket");

    return 0;
}

int handle_client(void)
{
    if (write(rc.remote_client_socket, "Connect", 7) != 7)
        perror("handle_client: write failed");
    char buffer[100];
    for (int i = 0; i < 100; i++)
        buffer[i] = '\0';
    if (read(rc.remote_client_socket, buffer, 7) != 7)
        perror("handle_client: read failed");
    if (strcmp(buffer, "success") == 0)
    {
        handle_tunnel();
        return 0;
    }
    else
    {
        close(rc.client_socket);
        return -1;
    }
}

void handle_tunnel(void)
{
    //if (build_tunnel() == 0)
    //{
    use_tunnel();
    //}
}
/*
   int build_tunnel(void)
   {
   rc.remote_host = gethostbyname(options.remote_host);
   if (rc.remote_host == NULL)
   {
   perror("build_tunnel: gethostbyname()");
   return 1;
   }

   memset(&rc.remote_addr, 0, sizeof(rc.remote_addr));

   rc.remote_addr.sin_family = AF_INET;
   rc.remote_addr.sin_port = htons(atoi(options.remote_port));

   memcpy(&rc.remote_addr.sin_addr.s_addr, rc.remote_host->h_addr, rc.remote_host->h_length);

   rc.remote_socket = socket(AF_INET, SOCK_STREAM, 0);
   if (rc.remote_socket < 0)
   {
   perror("build_tunnel: socket()");
   return 1;
   }

   if (connect(rc.remote_socket, (struct sockaddr *) &rc.remote_addr, sizeof(rc.remote_addr)) < 0)
   {
   perror("build_tunnel: connect()");
   return 1;
   }

   return 0;
   }*/

int use_tunnel(void)
{
    fd_set io;
    char buffer[options.buffer_size];

    for (;;)
    {
        FD_ZERO(&io);
        FD_SET(rc.client_socket, &io);
        FD_SET(rc.remote_client_socket, &io);

        memset(buffer, 0, sizeof(buffer));
        bool gotData = false;
        if (select(fd(), &io, NULL, NULL, NULL) < 0)
        {
            perror("use_tunnel: select()");
            break;
        }
        //fprintf(stderr, "select returned\n");

        if (FD_ISSET(rc.client_socket, &io))
        {
            int count = recv(rc.client_socket, buffer, sizeof(buffer), 0);
            if (count < 0)
            {
                perror("use_tunnel: recv(rc.client_socket)");
                close(rc.client_socket);
                close(rc.remote_client_socket);
                return 1;
            }

            if (count == 0)
            {
                close(rc.client_socket);
                close(rc.remote_client_socket);
                //fprintf(stderr, "use_returned 0_server\n");
                return 0;
            }

            send(rc.remote_client_socket, buffer, count, 0);
            gotData = true;

            if (settings.log)
            {
                printf("> %s > ", get_current_timestamp());
                fwrite(buffer, sizeof(char), count, stdout);
                fflush(stdout);
            }
        }

        if (FD_ISSET(rc.remote_client_socket, &io))
        {
            int count = recv(rc.remote_client_socket, buffer, sizeof(buffer), 0);
            if (count < 0)
            {
                perror("use_tunnel: recv(rc.remote_client_socket)");
                close(rc.client_socket);
                close(rc.remote_client_socket);
                return 1;
            }

            if (count == 0)
            {
                close(rc.client_socket);
                close(rc.remote_client_socket);
                //fprintf(stderr, "use_returned 0_client\n");
                return 0;
            }

            send(rc.client_socket, buffer, count, 0);

            gotData = true;
            if (settings.log)
            {
                fwrite(buffer, sizeof(char), count, stdout);
                fflush(stdout);
            }
        }
        if (!gotData)
        {
            // we have a connection to a new tunnel client, return and do another accept
            return 0;
        }
    }

    return 0;
}

int fd(void)
{
    unsigned int fd = rc.client_socket;
    if (fd < rc.remote_client_socket)
    {
        fd = rc.remote_client_socket;
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
    fprintf(stderr, "rTcpServer - rTcp tunnel entry point\n");
    fprintf(stderr, "Usage: %s [options]\n\n", name);
}

void print_helpinfo(void)
{
    fprintf(stderr, "Try `%s --help' for more options\n", name);
}

void print_help(void)
{
    fprintf(stderr,
"        Options:\n"
"        --version\n"
"        --help\n\n"
"        --local-port=PORT    local port (where tunnel users connect to)\n"
"        --remote-port=PORT   remote port (where rTcpClients connect to)\n"
"        --bind-address=IP    bind address (for both tunnel and user clients)\n"
"        --buffer-size=BYTES  buffer size\n"
#ifndef __MINGW32__
"        --fork               fork-based concurrency\n"
#endif
"        --log\n"
"        --stay-alive\n"
"\n");
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
