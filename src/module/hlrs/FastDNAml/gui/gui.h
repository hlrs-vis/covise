/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GUI_H
#define _GUI_H

#include <network/network.h>

#include "gui_command.h"
#include "gui_appl_list.h"

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define GUI_STATE_INITED 0x1000
#define GUI_STATE_DISCONNECTED 0x2000
#define GUI_STATE_CONNECTED 0x3000

struct _gui_command_t;

struct _gui_t
{
    sock_t sock;
    int state;
    gui_appl_list_t *appl_list;
};
typedef struct _gui_t gui_t;

EXTERN gui_t *gui_new();
EXTERN void gui_delete(gui_t **g);

EXTERN void gui_init(gui_t *g);

/* Not yet implemented: extern int gui_register_appl(gui_t *gui, appl_t *appl); */
EXTERN int gui_handle_command(gui_t *gui, struct _gui_command_t *comm);

EXTERN int gui_get_tree(gui_t *gui);

EXTERN int gui_main_loop(gui_t *gui);

#endif
