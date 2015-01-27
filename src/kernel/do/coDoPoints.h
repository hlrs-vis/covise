/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_POINTS_H
#define CO_DO_POINTS_H

#include "coDoCoordinates.h"

/*
 $Log: covise_unstr.h,v $
 * Revision 1.1  1993/09/25  20:52:42  zrhk0125
 * Initial revision
 *
*/

/***********************************************************************\ 
 **                                                                     **
 **   Untructured class                              Version: 1.0       **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of an unstructured grid   **
 **                  and the data on it in a distributed manner.        **
 **                                                                     **
 **   Classes      :                                                    **
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
 **                  23.06.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise
{

class DOEXPORT coDoPoints : public coDoCoordinates
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm no_of_points;
    coFloatShmArray vx;
    coFloatShmArray vy;
    coFloatShmArray vz;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoPoints *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoPoints(const coObjInfo &info)
        : coDoCoordinates(info, "POINTS")
    {
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };
    coDoPoints(const coObjInfo &info, coShmArray *arr);
    coDoPoints(const coObjInfo &info, int no);
    coDoPoints(const coObjInfo &info, int no,
               float *x, float *y, float *z);
    int getNumPoints() const
    {
        return no_of_points;
    }
    int getNumElements() const
    {
        return no_of_points;
    }
    void getPointCoordinates(int no, float *xc, float *yc, float *zc) const
    {
        *xc = vx[no];
        *yc = vy[no];
        *zc = vz[no];
    };
    void getAddresses(float **x_c, float **y_c, float **z_c) const
    {
        *x_c = (float *)vx.getDataPtr();
        *y_c = (float *)vy.getDataPtr();
        *z_c = (float *)vz.getDataPtr();
    };

    int setSize(int numElem);
};
}
#endif
