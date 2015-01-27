/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GUI_APPL_H
#define _GUI_APPL_H

#include <appl/appl.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

struct _gui_appl_t
{
    appl_t *appl;
    int applID;
};
typedef struct _gui_appl_t gui_appl_t;

EXTERN gui_appl_t *gui_appl_new();
EXTERN void gui_appl_delete(gui_appl_t **ga);

#endif
