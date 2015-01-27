/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoIntArr.h"

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

using namespace covise;

//////////////////////////////////////////////////////////////////

coDoIntArr::coDoIntArr(const coObjInfo &info, coShmArray *arr)
    : coDistributedObject(info, "INTARR")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

//////////////////////////////////////////////////////////////////

coDoIntArr::coDoIntArr(const coObjInfo &info,
                       int ndim,
                       const int *dimArray,
                       const int *initdata)
    : coDistributedObject(info, "INTARR")
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
        { INTSHMARRAY, &data }
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
               len * sizeof(int));
    }
}

coDoIntArr *coDoIntArr::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoIntArr(newinfo, getNumDimensions(), getDimensionPtr(), getAddress());
}

//////////////////////////////////////////////////////////////////

int coDoIntArr::rebuildFromShm()
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
        { INTSHMARRAY, &data }
    };
    return restore_shared_dl(3, dl);
}

//////////////////////////////////////////////////////////////////

coDistributedObject *coDoIntArr::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoIntArr(coObjInfo(), arr);
    return ret;
}

//////////////////////////////////////////////////////////////////

int coDoIntArr::getObjInfo(int no, coDoInfo **il) const
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
