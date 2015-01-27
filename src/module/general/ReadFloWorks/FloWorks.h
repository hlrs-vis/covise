/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    FloWorks
//
// Description: Class to access FloWorks file library
//
// Initial version: 2002-
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef FloWorks_H
#define FloWorks_H

#include <util/coviseCompat.h>

#include "DataItem.h"
#include "CFloWorksData.h"

class TimeSet;

typedef vector<DataItem> DataList;
typedef vector<TimeSet *> TimeSets;

class FloWorks
{
public:
    /// default CONSTRUCTOR
    FloWorks(const char *path);

    /// DESTRUCTOR
    ~FloWorks();

    DataList getDataIts() const
    {
        return dataIts_;
    };
    int getNumSteps() const
    {
        return numSets_;
    };
    int getNumVert() const
    {
        return numVert_;
    };
    int getNumCells() const
    {
        return numCells_;
    };

    bool activateTimeStep(int stepNo, float &realtime)
    {
        return fw_.activateTimeStep(stepNo, realtime);
    };

    bool getGridSize(int &numCell, int &numConn, int &numVert)
    {
        bool ret = fw_.getGridSize(numCell, numConn, numVert);
        numCells_ = numCell;
        numVert_ = numVert;
        numConn_ = numConn;
        return ret;
    };

    int getGridConn(int *cellList, int *typeList, int *connList);

    int getGridCoord(float *x, float *y, float *z)
    {
        return fw_.getGridCoord(x, y, z);
    };

    bool getData(int species, int comp, float *dest)
    {
        return fw_.getData(species, comp, dest);
    };

    bool empty()
    {
        return false;
    }

private:
    DataList dataIts_;
    int numSets_;
    int numSpecies_;
    int numCells_, numVert_, numConn_;
    bool is_ok_;
    CFloWorksData fw_;

    void correctConn(int numConn, int *connList);
};
#endif
