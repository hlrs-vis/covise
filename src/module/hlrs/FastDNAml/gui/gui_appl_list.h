/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GUI_APPL_LIST_H
#define _GUI_APPL_LIST_H

#include <appl/appl.h>
#include <network/network.h>

#include "gui_appl.h"

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

struct _gui_appl_list_t
{
    int max_num;
    int num;
    gui_appl_t **appls;
};
typedef struct _gui_appl_list_t gui_appl_list_t;

/* Constructors, destructors... */

EXTERN gui_appl_list_t *gui_appl_list_new();
EXTERN void gui_appl_list_delete(gui_appl_list_t **gal);

EXTERN void gui_appl_list_init(gui_appl_list_t *gal);

/* Searching in that list */

EXTERN gui_appl_t *gui_appl_list_find(gui_appl_list_t *appl, int ID);

/* Datatype functions */

EXTERN void gui_appl_list_add(gui_appl_list_t *gal, appl_t *appl, int applID);

/* Network stuff */

EXTERN int gui_appl_list_write(gui_appl_list_t *gal, sock_t sock);
EXTERN gui_appl_list_t *gui_appl_list_read(sock_t sock);

/* Misc. functions */

EXTERN int gui_appl_list_printf(gui_appl_list_t *gal);

#endif
