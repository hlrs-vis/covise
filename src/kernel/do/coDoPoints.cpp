/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoPoints.h"
#include "coDoData.h"
#include "coDoOctTreeP.h"
#include <covise/covise.h>

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

coDistributedObject *coDoPoints::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoPoints(coObjInfo(), arr);
    return ret;
}

int coDoPoints::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 4)
    {
        (*il)[0].description = "Number of Points";
        (*il)[1].description = "X Coordinates";
        (*il)[2].description = "Y Coordinates";
        (*il)[3].description = "Z Coordinates";
        return 4;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

int coDoPoints::setSize(int numElem)
{
    if (numElem > no_of_points)
        return -1;

    no_of_points = numElem;
    return 0;
}

coDoPoints::coDoPoints(const coObjInfo &info, int no)
    : coDoCoordinates(info, "POINTS")
{
    covise_data_list dl[4];

    vx.set_length(no);
    vy.set_length(no);
    vz.set_length(no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&vx;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&vy;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&vz;
    new_ok = store_shared_dl(4, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
}

coDoPoints::coDoPoints(const coObjInfo &info, int no,
                       float *x, float *y, float *z)
    : coDoCoordinates(info, "POINTS")
{
    covise_data_list dl[4];

    vx.set_length(no);
    vy.set_length(no);
    vz.set_length(no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&vx;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&vy;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&vz;
    new_ok = store_shared_dl(4, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    float *tmpvx, *tmpvy, *tmpvz;
    getAddresses(&tmpvx, &tmpvy, &tmpvz);
    int i = no * sizeof(float);
    memcpy(tmpvx, x, i);
    memcpy(tmpvy, y, i);
    memcpy(tmpvz, z, i);
    /*
       for(int i = 0;i < no;i++) {
      vx[i] = x[i];
      vy[i] = y[i];
      vz[i] = z[i];
       }
   */
}

coDoPoints::coDoPoints(const coObjInfo &info, coShmArray *arr)
    : coDoCoordinates(info, "POINTS")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoPoints *coDoPoints::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    getAddresses(&c[0], &c[1], &c[2]);
    return new coDoPoints(newinfo, getNumPoints(), c[0], c[1], c[2]);
}

int coDoPoints::rebuildFromShm()
{
    covise_data_list dl[4];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&vx;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&vy;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&vz;
    return restore_shared_dl(4, dl);
}
