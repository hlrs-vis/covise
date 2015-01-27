/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_DBLARR_H
#define CO_DO_DBLARR_H

#include "coDistributedObject.h"

namespace covise
{

class DOEXPORT coDoDoubleArr : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm numDim; // number of dimensions
    coIntShmArray dimension; // size in each dimension
    coDoubleShmArray data; // data

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoDoubleArr *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoDoubleArr(const coObjInfo &info)
        : coDistributedObject(info, "DBLARR")
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

    coDoDoubleArr(const coObjInfo &info, coShmArray *arr);

    coDoDoubleArr(const coObjInfo &info,
                  int numDim,
                  const int *dimArray,
                  const double *initdata = NULL);

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

    double *getAddress() const
    {
        return (double *)(data.getDataPtr());
    }
    int *getDimensionPtr() const
    {
        return (int *)(dimension.getDataPtr());
    }

    void getAddress(double **res) const
    {
        *res = (double *)data.getDataPtr();
    }

    ~coDoDoubleArr(){};
};
}
#endif
