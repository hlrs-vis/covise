/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_COLORMAP_H
#define CO_DO_COLORMAP_H

#include "coDistributedObject.h"

/****************************************************************
 **                                                              **
 **   Integer Array Class                        Version: 1.1    **
 **                                                              **
 **                                                              **
 **   Description  : Classes for multi-dimensional Integer       **
 **                  Arrays                                      **
 **                                                              **
 **   Classes      : coDoColormap                                   **
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

class DOEXPORT coDoColormap : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm shm_numSteps; // number values in Map
    coFloatShmArray shm_colors; // RGBAX values 5*float
    coFloatShm shm_min; // minimum value
    coFloatShm shm_max; // maximum value
    coCharShmArray shm_name; // Colormap name

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoColormap(const coObjInfo &info, coShmArray *arr);
    coDoColormap *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoColormap(const coObjInfo &info)
        : coDistributedObject(info, "COLMAP")
    {
        getObjectFromShm();
    };

    coDoColormap(const coObjInfo &info,
                 int numSteps,
                 float min,
                 float max,
                 const float *map,
                 const char *name);

    float *getAddress() const
    {
        return (float *)shm_colors.getDataPtr();
    }
    float getMin() const
    {
        return shm_min;
    }
    float getMax() const
    {
        return shm_max;
    }
    int getNumSteps() const
    {
        return shm_numSteps;
    }
    const char *getMapName() const
    {
        return (const char *)shm_name.getDataPtr();
    }

    virtual ~coDoColormap()
    {
    }
};
}
#endif
