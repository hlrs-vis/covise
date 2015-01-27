/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>

#include "gui_appl.h"

gui_appl_t *
gui_appl_new()
{
    gui_appl_t *rv = NULL;

    rv = (gui_appl_t *)malloc(sizeof(gui_appl_t));

    if (rv != NULL)
    {
        rv->appl = appl_new();
        if (rv->appl != NULL)
        {
            rv->applID = -1;
        }
        else
            gui_appl_delete(&rv);
    }

    return rv;
}

void
gui_appl_delete(gui_appl_t **ga)
{
    if (*ga)
    {
        if ((*ga)->appl)
            appl_delete(&(*ga)->appl);
        free(*ga);
        *ga = NULL;
    }
}
