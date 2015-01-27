/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COC_TYPES_H
#define COC_TYPES_H

struct _CoC_UNSGRD
{
    int numElem;
    int numCoords;
    int numConn;
    int *elem_l;
    int *type_l;
    int *conn_l;
    float *x_coord;
    float *y_coord;
    float *z_coord;
};

typedef struct _CoC_UNSGRD CoC_UNSGRD;

struct _CoC_Polyed_UNSGRD
{
    int numTetra;
    int numHexa;
    int numPyra;

    int *hexaIdx;
    int *tetraIdx;
    int *pyraIdx;
};
typedef struct _CoC_Polyed_UNSGRD CoC_Polyed_UNSGRD;

struct _CoC_Polyed_POLYGN
{
    int numTri;
    int numQuad;

    int *triIdx;
    int *quadIdx;
};
typedef struct _CoC_Polyed_POLYGN CoC_Polyed_POLYGN;
#endif
