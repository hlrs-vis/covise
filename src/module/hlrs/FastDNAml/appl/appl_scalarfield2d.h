/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_SCALARFIELD2D_H
#define _LIBAPPL_APPL_SCALARFIELD2D_H

#include <network/network.h>

struct _Scalarfield2D
{
    int size_x, size_y;
    double *data;
};
typedef struct _Scalarfield2D Scalarfield2D;

extern Scalarfield2D *Scalarfield2D_new(int dim_x, int dim_y);
extern void Scalarfield2D_delete(Scalarfield2D **sf2d);

extern int Scalarfield2D_write(Scalarfield2D *sf2d, sock_t sock);
extern int Scalarfield2D_read(Scalarfield2D *sfd, sock_t sock);

#endif
