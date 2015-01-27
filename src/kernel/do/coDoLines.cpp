/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoLines.h"

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

coDistributedObject *coDoLines::virtualCtor(coShmArray *arr)
{
    return new coDoLines(coObjInfo(), arr);
}

int coDoLines::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 5)
    {
        (*il)[0].description = "Points";
        (*il)[1].description = "Number of Vertices";
        (*il)[2].description = "Vertex List";
        (*il)[3].description = "Number of Lines";
        (*il)[4].description = "Line List";
        return 5;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

int coDoLines::setNumVertices(int numElem)
{
    if (numElem > no_of_vertices)
        return -1;

    no_of_vertices = numElem;
    return 0;
}

int coDoLines::setNumLines(int numElem)
{
    if (numElem > no_of_lines)
        return -1;

    no_of_lines = numElem;
    return 0;
}

coDoLines::coDoLines(const coObjInfo &info, coShmArray *arr)
    : coDoGrid(info, "LINES")
{
    points = new coDoPoints(coObjInfo());
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoLines::coDoLines(const coObjInfo &info,
                     int no_p, int no_v, int no_l)
    : coDoGrid(info, "LINES")
{
    char *p_name;
    covise_data_list dl[5];

    p_name = new char[strlen(info.getName()) + 3];
    strcpy(p_name, info.getName());
    strcat(p_name, "_P");
    points = new coDoPoints(coObjInfo(p_name), no_p);
    delete[] p_name;
    vertex_list.set_length(no_v);
    line_list.set_length(no_l);
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    dl[3].type = INTSHM;
    dl[3].ptr = (void *)&no_of_lines;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&line_list;
    new_ok = store_shared_dl(5, dl) != 0;
    if (!new_ok)
        return;
    no_of_vertices = no_v;
    no_of_lines = no_l;
}

coDoLines::coDoLines(const coObjInfo &info, int no_p,
                     float *x_c, float *y_c, float *z_c, int no_v, int *v_l,
                     int no_l, int *l_l)
    : coDoGrid(info, "LINES")
{
    char *p_name;
    int i;
    covise_data_list dl[5];

    p_name = new char[strlen(info.getName()) + 3];
    strcpy(p_name, info.getName());
    strcat(p_name, "_P");
    points = new coDoPoints(coObjInfo(p_name), no_p, x_c, y_c, z_c);
    delete[] p_name;
    vertex_list.set_length(no_v);
    line_list.set_length(no_l);
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    dl[3].type = INTSHM;
    dl[3].ptr = (void *)&no_of_lines;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&line_list;
    new_ok = store_shared_dl(5, dl) != 0;
    if (!new_ok)
        return;
    no_of_vertices = no_v;
    no_of_lines = no_l;

    int *tmpv, *tmpl;
    float *tmpf;
    getAddresses(&tmpf, &tmpf, &tmpf, &tmpv, &tmpl);
    i = no_v * sizeof(int);
    memcpy(tmpv, v_l, i);
    i = no_l * sizeof(int);
    memcpy(tmpl, l_l, i);

    /*
       for(i = 0;i < no_v;i++)
      vertex_list[i] = v_l[i];
       for(i = 0;i < no_l;i++)
      line_list[i] = l_l[i];
   */
}

coDoLines *coDoLines::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    int *v_l, *l_l;
    getAddresses(&c[0], &c[1], &c[2], &v_l, &l_l);
    return new coDoLines(newinfo, getNumPoints(), c[0], c[1], c[2], getNumVertices(), v_l,
                         getNumLines(), l_l);
}

int coDoLines::rebuildFromShm()
{
    covise_data_list dl[5];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&no_of_vertices;
    dl[2].type = INTSHMARRAY;
    dl[2].ptr = (void *)&vertex_list;
    dl[3].type = INTSHM;
    dl[3].ptr = (void *)&no_of_lines;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&line_list;
    return restore_shared_dl(5, dl);
}
