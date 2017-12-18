/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SAMPLE_H
#define _SAMPLE_H
/**************************************************************************\ 
 **                                                (C)1999-2000 VirCinity  **
 **   Sample module                                                        **
 **                                                                        **
 ** Authors: Ralph Bruckschen (Vircinity)                                  **
 **          D. Rainer (RUS)                                               **
 **          S. Leseduarte (Vircinity)                                     **
 **                                                                        **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include "unstruct.h"
#include <vector>

class Sample : public coModule
{

private:
    //////////  member functions

    virtual int compute(const char *port);
    virtual void quit();
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    ////////// the data in- and output ports

    coInputPort *Grid_In_Port, *Data_In_Port, *Reference_Grid_In_Port;
    coOutputPort *Grid_Out_Port, *Data_Out_Port;

    // treatment of sets...
    int TimeSteps; // flag if we have time steps
    const coDistributedObject *const *gridTimeSteps;
    const coDistributedObject *const *dataTimeSteps;
    int num_set_ele;
    int num_blocks;

    typedef std::vector<std::vector<const coDistributedObject *> > ias;
    int Diagnose(ias &grids, ias &data, unstruct_grid::vecFlag typeFlag, bool isStrGrid);

    ////////// parameters

    //coChoiceParam *algorithmParam;
    coChoiceParam *iSizeChoiceParam;
    coChoiceParam *jSizeChoiceParam;
    coChoiceParam *kSizeChoiceParam;
    coIntScalarParam *iSizeParam;
    coIntScalarParam *jSizeParam;
    coIntScalarParam *kSizeParam;
    coChoiceParam *outsideChoiceParam; //  select between MAX_FLT and number

    coChoiceParam *p_algorithm; //
    coChoiceParam *p_pointSampling; //
    coChoiceParam *p_bounding_box; //
    coFloatVectorParam *p_P1bound_manual; //
    coFloatVectorParam *p_P2bound_manual; //
    coFloatVectorParam *p_rangeMin, *p_rangeMax;

    coFloatParam *fillValueParam; // if outside is number
    coFloatParam *epsParam; // eps covers numerical problems

    enum
    {
        SAMPLE_ACCURATE = 3,
        SAMPLE_HOLES = 0,
        SAMPLE_NO_HOLES = 2,
        SAMPLE_NO_HOLES_BETTER = 1
    };
    enum
    {
        OUTSIDE_MAX_FLT = 0,
        OUTSIDE_NUMBER = 1
    };
    float eps;

public:
    enum
    {
        POINTS_LINEAR = 0,
        POINTS_LOGARITHMIC = 1,
        POINTS_LINEAR_NORMALIZED = 2,
        POINTS_LOGARITHMIC_NORMALIZED = 3
    };

    Sample(int argc, char *argv[]);
};

template <class T>
void
compress(std::vector<T> &iarray, int item)
{
    if (item < 0 || item >= iarray.size())
    {
        return;
    }
    if (iarray.size() == 1)
    {
        iarray.clear();
        return;
    }
    size_t length = iarray.size();
    T *buffer = new T[length - 1];
    for (int count = 0; count < item; ++count)
    {
        buffer[count] = iarray[count];
    }
    for (int count = item + 1; count < length; ++count)
    {
        buffer[count - 1] = iarray[count];
    }
    iarray.clear();
    for (int count = 0; count < length - 1; ++count)
    {
        iarray.push_back(buffer[count]);
    }
    delete[] buffer;
}
#endif // _SAMPLE_H
