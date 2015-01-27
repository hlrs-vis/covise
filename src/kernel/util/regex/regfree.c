/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>

#include "regex.h"
#include "utils.h"
#include "regex2.h"

/*
 - regfree - free everything
 = void UTILEXPORT regfree(regex_t *);
 */
void
    UTILEXPORT
        regfree(preg)
            regex_t *preg;
{
    register struct re_guts *g;

    if (preg->re_magic != MAGIC1) /* oops */
        return; /* nice to complain, but hard */

    g = preg->re_g;
    if (g == NULL || g->magic != MAGIC2) /* oops again */
        return;
    preg->re_magic = 0; /* mark it invalid */
    g->magic = 0; /* mark it invalid */

    if (g->strip != NULL)
        free((char *)g->strip);
    if (g->sets != NULL)
        free((char *)g->sets);
    if (g->setbits != NULL)
        free((char *)g->setbits);
    if (g->must != NULL)
        free(g->must);
    free((char *)g);
}
