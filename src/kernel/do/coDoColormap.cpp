/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoColormap.h"

/****************************************************************\ 
 **                                                              **
 **   Colormap Object Class                        Version: 1.o  **
 **                                                              **
 **                                                              **
 **   Description  : for Colormap objects                        **
 **                                                              **
 **   Classes      : coDoColormap                                 **
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

coDoColormap::coDoColormap(const coObjInfo &info, coShmArray *arr)
    : coDistributedObject(info, "COLMAP")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

//////////////////////////////////////////////////////////////////

coDoColormap::coDoColormap(const coObjInfo &info,
                           int numSteps,
                           float minVal,
                           float maxVal,
                           const float *map,
                           const char *mapName)
    : coDistributedObject(info, "COLMAP")
{
    // No name ? -> null name
    if (!mapName)
        mapName = "";

    //////// set the array dimensions
    shm_colors.set_length(5 * numSteps);
    shm_name.set_length((int)strlen(mapName) + 1);

    //////// store in shared memory
    covise_data_list dl[] = {
        { INTSHM, &shm_numSteps },
        { FLOATSHMARRAY, &shm_colors },
        { FLOATSHM, &shm_min },
        { FLOATSHM, &shm_max },
        { CHARSHMARRAY, &shm_name }
    };
    new_ok = store_shared_dl(5, dl) != 0;

    if (!new_ok)
        return;

    //////// copy the fields and set the values

    shm_numSteps = numSteps;
    memcpy(shm_colors.getDataPtr(), map, 5 * numSteps * sizeof(float));
    shm_min = minVal;
    shm_max = maxVal;
    strcpy((char *)shm_name.getDataPtr(), mapName);
}

coDoColormap *coDoColormap::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoColormap(newinfo, getNumSteps(), getMin(), getMax(), getAddress(), getMapName());
}

//////////////////////////////////////////////////////////////////

int coDoColormap::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
        return 0;
    }

    covise_data_list dl[] = {
        { INTSHM, &shm_numSteps },
        { FLOATSHMARRAY, &shm_colors },
        { FLOATSHM, &shm_min },
        { FLOATSHM, &shm_max },
        { CHARSHMARRAY, &shm_name }
    };
    return restore_shared_dl(5, dl);
}

//////////////////////////////////////////////////////////////////

coDistributedObject *coDoColormap::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = (coDistributedObject *)new coDoColormap(coObjInfo(), arr);
    return ret;
}

//////////////////////////////////////////////////////////////////

int coDoColormap::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 5)
    {
        (*il)[0].description = "Number Steps in Map";
        (*il)[1].description = "Packed Colors";
        (*il)[2].description = "Minimum Value";
        (*il)[3].description = "Maximum Value";
        (*il)[4].description = "Colormap name";
        return 5;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}
