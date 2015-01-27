/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DTF_TIMESTEP_H_
#define __DTF_TIMESTEP_H_

#include "Data.h"
#include "grid.h"

#include "coModule.h"

namespace DTF
{
class ClassInfo_DTFTimeStep;

class TimeStep : public Tools::BaseObject
{
    friend class ClassInfo_DTFTimeStep;

private:
    TimeStep();
    TimeStep(string className, int objectID);

    static ClassInfo_DTFTimeStep classInfo;

    map<int, int> typeWrapper;
    DTF::Grid *grid;
    map<string, vector<double> > dblData;

    bool fillTypeList(Zone *zone, Grid *gridObj);
    bool fillElementList(Zone *zone, Grid *gridObj);
    bool fillCornerList(Zone *zone, Grid *gridObj);
    bool fillCoordList(Nodes *nodes, Grid *gridObj);

    bool extractGrid(Zone *zone);
    bool extractData(Zone *zone);

    bool convertType(int dtfType, int &coType);

public:
    virtual ~TimeStep();

    bool setData(Sim *sim, int zoneNr);

    coDoUnstructuredGrid *getGrid(string name);
    coDoFloat *getData(string name, string dataName);
    bool hasData();
    virtual void clear();
    virtual void print();
};

CLASSINFO(ClassInfo_DTFTimeStep, TimeStep);
};
#endif
