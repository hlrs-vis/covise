/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoDoubleArr.h"

using namespace covise;

//////////////////////////////////////////////////////////////////

coDoDoubleArr::coDoDoubleArr(const coObjInfo &info, coShmArray *arr)
    : coDistributedObject(info, "DBLARR")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

//////////////////////////////////////////////////////////////////

coDoDoubleArr::coDoDoubleArr(const coObjInfo &info,
                             int ndim,
                             const int *dimArray,
                             const double *initdata)
    : coDistributedObject(info, "DBLARR")
{
    //////// set the dimension array length
    dimension.set_length(ndim);

    //////// set the data array length
    int i, len = 1;
    for (i = 0; i < ndim; i++)
        len *= dimArray[i];
    data.set_length(len);

//////// store in shared memory

#ifdef DEBUG
    cerr << "vor store_shared coDoRectilinearGrid\n";
#endif

    covise_data_list dl[] = {
        {
          INTSHM, &numDim,
        },
        {
          INTSHMARRAY, &dimension,
        },
        { DOUBLESHMARRAY, &data }
    };
    new_ok = store_shared_dl(3, dl) != 0;

    if (!new_ok)
        return;

    //////// copy the variables

    numDim = ndim;
    memcpy(dimension.getDataPtr(), dimArray, ndim * sizeof(int));
    if (initdata)
    {
        memcpy(data.getDataPtr(), (const void *)initdata,
               len * sizeof(double));
    }
}

coDoDoubleArr *coDoDoubleArr::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoDoubleArr(newinfo, getNumDimensions(), getDimensionPtr(), getAddress());
}

//////////////////////////////////////////////////////////////////

int coDoDoubleArr::rebuildFromShm()
{
    if (shmarr == 0L)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }

    covise_data_list dl[] = {
        {
          INTSHM, &numDim,
        },
        {
          INTSHMARRAY, &dimension,
        },
        { DOUBLESHMARRAY, &data }
    };
    return restore_shared_dl(3, dl);
}

//////////////////////////////////////////////////////////////////

coDistributedObject *coDoDoubleArr::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoDoubleArr(coObjInfo(), arr);
    return ret;
}

//////////////////////////////////////////////////////////////////

int coDoDoubleArr::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 3)
    {
        (*il)[0].description = "Number of Dimensions";
        (*il)[1].description = "Size per Dimension";
        (*il)[2].description = "Data";
        return 3;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}
