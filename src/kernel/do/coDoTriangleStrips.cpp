/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoTriangleStrips.h"

/*
 $Log: covise_unstr.C,v $
Revision 1.1  1993/09/25  20:52:52  zrhk0125
Initial revision

*/

/***********************************************************************\ 
 **                                                                     **
 **   Structured classes Routines                   Version: 1.1        **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of structured grids       **
 **                  in a distributed manner.                           **
 **                                                                     **
 **   Classes      : coDoPoints, coDoLines, coDoPolygons **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93  Ver 1.1 shm-access restructured,         **
 **                                    recursive data-objects (simple   **
 **                                    version),                        **
 **                                    some new types added             **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

using namespace covise;

coDistributedObject *coDoTriangleStrips::virtualCtor(coShmArray *arr)
{
    return new coDoTriangleStrips(coObjInfo(), arr);
}

int coDoTriangleStrips::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 1)
    {
        (*il)[0].description = "Lines";
        return 1;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoTriangleStrips::coDoTriangleStrips(const coObjInfo &info, coShmArray *arr)
    : coDoGrid(info, "TRIANG")
{
    lines = new coDoLines(coObjInfo());
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoTriangleStrips::coDoTriangleStrips(const coObjInfo &info,
                                       int no_p, int no_v, int no_pol)
    : coDoGrid(info, "TRIANG")
{
    char *l_name;
    covise_data_list dl[5];

    l_name = new char[strlen(info.getName()) + 3];
    strcpy(l_name, info.getName());
    strcat(l_name, "_L");
    lines = new coDoLines(coObjInfo(l_name), no_p, no_v, no_pol);
    delete[] l_name;
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)lines;
    new_ok = store_shared_dl(1, dl) != 0;
    if (!new_ok)
        return;
}

coDoTriangleStrips::coDoTriangleStrips(const coObjInfo &info, int no_p,
                                       float *x_c, float *y_c, float *z_c, int no_v, int *v_l,
                                       int no_pol, int *pol_l)
    : coDoGrid(info, "TRIANG")
{
    char *l_name;
    covise_data_list dl[5];

    l_name = new char[strlen(info.getName()) + 3];
    strcpy(l_name, info.getName());
    strcat(l_name, "_L");
    lines = new coDoLines(coObjInfo(l_name), no_p, x_c, y_c, z_c,
                          no_v, v_l, no_pol, pol_l);
    delete[] l_name;
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)lines;
    new_ok = store_shared_dl(1, dl) != 0;
    if (!new_ok)
        return;
}

coDoTriangleStrips *coDoTriangleStrips::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    int *v_l, *s_l;
    getAddresses(&c[0], &c[1], &c[2], &v_l, &s_l);
    return new coDoTriangleStrips(newinfo, getNumPoints(), c[0], c[1], c[2], getNumVertices(), v_l,
                                  getNumStrips(), s_l);
}

int coDoTriangleStrips::rebuildFromShm()
{
    covise_data_list dl[5];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)lines;

    return restore_shared_dl(1, dl);
}
