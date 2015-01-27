/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GUI_COMMAND_H
#define _GUI_COMMAND_H

#include <appl/appl.h>
#include <network/network.h>

#include "gui.h"

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

struct _gui_t;

struct _gui_command_t
{

    int command;
    int ID;
    int protocol_version;

    int TAN;
    int seq;

    void *data;
};
typedef struct _gui_command_t gui_command_t;

EXTERN gui_command_t *gui_command_new();
EXTERN void gui_command_delete(gui_command_t **gc);
EXTERN void gui_command_init(gui_command_t *gc);

/* General networking functions */

EXTERN int gui_command_header_write(gui_command_t *gc, sock_t sock);
EXTERN gui_command_t *gui_command_header_read(sock_t sock);
EXTERN int gui_command_header_read_ip(gui_command_t *gc, sock_t sock);

EXTERN gui_command_t *gui_command_wait(struct _gui_t *gui, int milli);

/* Specific commands. */

EXTERN int gui_command_stop_appl(sock_t sock, int applID);
EXTERN int gui_command_restart_appl(sock_t sock, int applID);

EXTERN int gui_command_read_tree(sock_t sock);
EXTERN int gui_command_write_tree(sock_t sock);

EXTERN int gui_command_printf(gui_command_t *gc);

#endif
