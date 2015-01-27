/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_TRIANGLESTRIPS_H
#define CO_DO_TRIANGLESTRIPS_H

#include "coDoGrid.h"
#include "coDoLines.h"

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

class DOEXPORT coDoTriangleStrips : public coDoGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

protected:
    coDoLines *lines;
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoTriangleStrips *cloneObject(const coObjInfo &info) const;

public:
    coDoTriangleStrips(const coObjInfo &info)
        : coDoGrid(info, "TRIANG")
    {
        lines = new coDoLines(coObjInfo());
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
    coDoTriangleStrips(const coObjInfo &info, coShmArray *arr);
    coDoTriangleStrips(const coObjInfo &info, int no_p, int no_v, int no_l);
    coDoTriangleStrips(const coObjInfo &info, int no_p, float *x_c,
                       float *y_c, float *z_c, int no_v, int *v_l, int no_pol, int *pol_l);
    int getNumStrips() const
    {
        return lines->getNumLines();
    }
    int setNumStrips(int num)
    {
        return lines->setNumLines(num);
    }
    int getNumVertices() const
    {
        return lines->getNumVertices();
    }
    int setNumVertices(int num)
    {
        return lines->setNumVertices(num);
    }
    int getNumPoints() const
    {
        return lines->getNumPoints();
    }
    int setNumPoints(int num)
    {
        return lines->setNumPoints(num);
    }
    void getAddresses(float **x_c, float **y_c, float **z_c, int **v_l, int **l_l) const
    {
        lines->getAddresses(x_c, y_c, z_c, v_l, l_l);
    }
};
}
#endif
