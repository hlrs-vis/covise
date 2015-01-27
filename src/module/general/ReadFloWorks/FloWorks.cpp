/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                            ++
// ++             Implementation of class FloWorks                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "FloWorks.h"
#include <util/coviseCompat.h>

//
// Constructor
//
FloWorks::FloWorks(const char *path)
    : numVert_(0)
    , numConn_(0)
    , numCells_(0)
{
    is_ok_ = fw_.openDirectory(path);
    numSets_ = fw_.getNumTimeSteps();

    float rt;
    fw_.activateTimeStep(0, rt);
    numSpecies_ = fw_.getNumSpecies();
    //fprintf(stderr, "%s:%d,%d\n", path, numSets_, numSpecies_ );

    int i;
    DataItem *item;
    CFloWorksData::DataType dummy;
    const char *species;

    for (i = 0; i < numSpecies_; i++)
    {
        fw_.getSpecies(i, dummy, species);
        string file(path);
        string desc(species);

        item = new DataItem(DataItem::scalar, file, desc);
        dataIts_.push_back(*item);
        //delete item;
    }
}

//
// Method:
//
int
FloWorks::getGridConn(int *cellList, int *typeList, int *connList)
{
    int ret = fw_.getGridConn(cellList, typeList, connList);

    // correct connectivity
    correctConn(numConn_, connList);

    return ret;
}

void
FloWorks::correctConn(int numConn, int *connList)
{
    int i, j;
    int store[8];
    for (i = 0; i < numConn; i += 8)
    {
        for (j = 0; j < 8; j++)
        {
            store[j] = connList[i + j];
        }
        connList[i] = store[0];
        connList[i + 1] = store[2];
        connList[i + 2] = store[3];
        connList[i + 3] = store[1];
        connList[i + 4] = store[4];
        connList[i + 5] = store[6];
        connList[i + 6] = store[7];
        connList[i + 7] = store[5];
    }
}

//
// Destructor
//
FloWorks::~FloWorks()
{
}
