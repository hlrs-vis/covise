/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_LINES_H
#define CO_DO_LINES_H

#include "coDoGrid.h"
#include "coDoPoints.h"

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

class DOEXPORT coDoLines : public coDoGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    coIntShm no_of_vertices;
    coIntShmArray vertex_list;
    coIntShm no_of_lines;
    coIntShmArray line_list;

protected:
    coDoPoints *points;
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoLines *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoLines(const coObjInfo &info)
        : coDoGrid(info, "LINES")
    {
        points = new coDoPoints(coObjInfo());
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
    coDoLines(const coObjInfo &info, coShmArray *arr);
    coDoLines(const coObjInfo &info, int no_p, int no_v, int no_l);
    coDoLines(const coObjInfo &info, int no_p,
              float *x_c, float *y_c, float *z_c, int no_v, int *v_l,
              int no_l, int *l_l);
    int getNumLines() const
    {
        return no_of_lines;
    }
    int setNumLines(int num_elem);
    int getNumVertices() const
    {
        return no_of_vertices;
    }
    int setNumVertices(int num_elem);
    int getNumPoints() const
    {
        return points->getNumPoints();
    }
    int setNumPoints(int num)
    {
        return points->setSize(num);
    };
    void getAddresses(float **x_c, float **y_c, float **z_c, int **v_l, int **l_l) const
    {
        points->getAddresses(x_c, y_c, z_c);
        *v_l = (int *)vertex_list.getDataPtr();
        *l_l = (int *)line_list.getDataPtr();
    };
};
}
#endif
