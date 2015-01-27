/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_INTARR_H
#define CO_DO_INTARR_H

#include "coDistributedObject.h"

/****************************************************************
 **                                                              **
 **   Integer Array Class                        Version: 1.1    **
 **                                                              **
 **                                                              **
 **   Description  : Classes for multi-dimensional Integer       **
 **                  Arrays                                      **
 **                                                              **
 **   Classes      : coDoIntArr                                   **
 **                                                              **
 **   Copyright (C) 1997     by University of Stuttgart          **
 **                             Computer Center (RUS)            **
 **                             Allmandring 30a                  **
 **                             70550 Stuttgart                  **
 **                                                              **
 **                                                              **
 **   Author       : A. Werner   (RUS)                           **
 **                                                              **
 **   History      : 26.06.97  Implementation                    **
 **                                                              **
\****************************************************************/
namespace covise
{

class DOEXPORT coDoIntArr : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm numDim; // number of dimensions
    coIntShmArray dimension; // size in each dimension
    coIntShmArray data; // data

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoIntArr *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoIntArr(const coObjInfo &info)
        : coDistributedObject(info, "INTARR")
    {
        if (info.getName())
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

    coDoIntArr(const coObjInfo &info, coShmArray *arr);

    coDoIntArr(const coObjInfo &info,
               int numDim,
               const int *dimArray,
               const int *initdata = NULL);

    int getNumDimensions() const
    {
        return numDim;
    }
    int getDimension(int i) const
    {
        if ((i >= 0) && (i < numDim))
            return dimension[i];
        else
            return -1;
    }

    int getSize() const
    {
        return (data.get_length());
    }

    int *getAddress() const
    {
        return (int *)(data.getDataPtr());
    }
    int *getDimensionPtr() const
    {
        return (int *)(dimension.getDataPtr());
    }

    void getAddress(int **res) const
    {
        *res = (int *)data.getDataPtr();
    }

    ~coDoIntArr(){};
};
}
#endif
