/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include "gui_command.h"

gui_command_t *
gui_command_new()
{
    gui_command_t *rv = NULL;

    rv = (gui_command_t *)malloc(sizeof(gui_command_t));
    if (rv != NULL)
        gui_command_init(rv);
    return rv;
}

void
gui_command_delete(gui_command_t **gc)
{
    if (*gc)
    {

        free(*gc);
        *gc = NULL;
    }
}

void
gui_command_init(gui_command_t *gc)
{
    if (gc != NULL)
    {
        gc->command = APPL_COMMAND_INVALID;
        gc->ID = 0;
        gc->protocol_version = -1;
        gc->TAN = 0;
        gc->seq = -1;
        gc->data = NULL;
    }
}

int
gui_command_header_write(gui_command_t *gc, sock_t sock)
{
    int rv;

    if (gc != NULL)
    {
        sock_write_int(sock, &gc->command);
        sock_write_int(sock, &gc->ID);
        sock_write_int(sock, &gc->protocol_version);
        sock_write_int(sock, &gc->TAN);
        sock_write_int(sock, &gc->seq);
    }

    return rv;
}

gui_command_t *
gui_command_header_read(sock_t sock)
{
    gui_command_t *rv = NULL;

    rv = gui_command_new();

    if (rv != NULL)
    {
        if (gui_command_header_read_ip(rv, sock) != 0)
        {
            gui_command_delete(&rv);
        }
    }

    return rv;
}

int
gui_command_header_read_ip(gui_command_t *gc, sock_t sock)
{
    int rv = 0;
    char bytes[10];

    if (gc != NULL)
    {
        if (recv(sock, bytes, 10, MSG_PEEK) > 0)
        {
            rv |= sock_read_int(sock, &gc->command);
            rv |= sock_read_int(sock, &gc->ID);
            rv |= sock_read_int(sock, &gc->protocol_version);
            rv |= sock_read_int(sock, &gc->TAN);
            rv |= sock_read_int(sock, &gc->seq);
        }
        else
            rv = -1;
    }

    return rv;
}

int
gui_command_interpet_buffer(gui_t *gui, char *buf, int size)
{
    int rv = 0;
    int internal_state;

    return rv;
}

gui_command_t *
gui_command_wait(gui_t *gui, int milli)
{
    gui_command_t *rv = NULL;
    sock_t sock;
    fd_set wait_set;
    struct timeval time_out;
    char bytes[10];
    int num_descr;

    char key_buffer[1024];
    int bytes_read;
    int i;

    char waiting;

    sock = gui->sock;

    waiting = 1;
    while (waiting)
    {

        time_out.tv_sec = milli / 1000;
        time_out.tv_usec = (milli - time_out.tv_sec * 1000) * 1000;

        FD_ZERO(&wait_set);
        FD_SET(sock, &wait_set);
        /* FD_SET(0, &wait_set); */

        num_descr = select(sock + 1, &wait_set, NULL, NULL, &time_out);
        if (num_descr >= 0 || (num_descr < 0 && errno != EINTR))
            waiting = 0;
    }

    if (num_descr > 0)
    {

        printf("Waiting is over...\n");

        if (FD_ISSET(sock, &wait_set))
        {
            rv = gui_command_header_read(sock);
            if (rv == NULL)
            {
                printf("NO command header.\n");
                sock_close(gui->sock);
                gui->state = GUI_STATE_DISCONNECTED;
            }
            else
            {
                printf("Found a command.\n");
            }
        }

        if (FD_ISSET(0, &wait_set))
        {
            printf("Reading from stdin.\n");
            bytes_read = read(0, key_buffer, 1024);
            if (bytes_read < 0)
                perror("Read");
            else if (bytes_read == 0)
                printf("Done.\n");
            else
            { /* gui_buffer_interpret(gui, key_buffer, bytes_read); */
            }
        }
    }
    else if (num_descr < 0)
    {
        perror("select");
        sock_close(gui->sock);
        rv = NULL;
        gui->state = GUI_STATE_DISCONNECTED;
    }

    return rv;
}

int
gui_command_stop_appl(sock_t sock, int applID)
{
    int rv = 0;
    gui_command_t newComm;

    newComm.command = APPL_COMMAND_END;
    gui_command_header_write(&newComm, sock);

    return rv;
}

int
gui_command_restart_appl(sock_t sock, int applID)
{
    int rv = 0;
    gui_command_t newComm;

    gui_command_init(&newComm);
    newComm.command = APPL_COMMAND_HALT;
    gui_command_header_write(&newComm, sock);

    gui_command_init(&newComm);
    newComm.command = APPL_COMMAND_START;
    gui_command_header_write(&newComm, sock);

    return rv;
}

int
gui_command_read_tree(sock_t sock)
{
    int rv = 0;
    gui_command_t newCommand;

    return rv;
}

int
gui_command_write_tree(sock_t sock)
{
    int rv = 0;

    return rv;
}

int
gui_command_printf(gui_command_t *gc)
{
    printf("Command:     %x\n", gc->command);
    printf("ID:          %d\n", gc->ID);
    printf("Seq:         %d\n", gc->seq);

    return 0;
}
