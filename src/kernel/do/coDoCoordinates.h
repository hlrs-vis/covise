/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_COORDINATES_H
#define CO_DO_COORDINATES_H

#include "coDoGrid.h"

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

class DOEXPORT coDoCoordinates : public coDoGrid
{
public:
    coDoCoordinates(const coObjInfo &info, const char *t)
        : coDoGrid(info, t)
    {
    }

    virtual void getPointCoordinates(int no, float *xc, float *yc, float *zc) const = 0;
    virtual int getNumElements() const = 0;
};
}
#endif
