/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CoolEmAllToDatabase_H
#define _CoolEmAllToDatabase_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "CoolEmAllToDatabase, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include <api/coModule.h>
#include <do/coDoUnstructuredGrid.h>

#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <osg/Vec3f>

using namespace covise;
class BoundaryPatch;
class BoundaryQuad;

class CoolEmAllToDatabase : public coModule
{
private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

public:
    CoolEmAllToDatabase(int argc, char *argv[]);
    coInputPort *p_boco, *p_temp, *p_grid, *p_p, *p_velo;
    coOutputPort *p_gridOut;

    covise::coStringParam *p_databasePrefix;
    covise::coFileBrowserParam *p_csvPath;
    int numObj;
    int *elem;
    int *diricletIndex;
    int *balance;
    int *conn;
    int *inTypeList;
    int *wall, *wall_machine;

    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *diricletVal;
    float *temp;
    float *press;

    int numCoord;
    int numConn;
    int numElem;
    int numBalance;
    int numDiriclet;
    int numWall;
    int colDiriclet, colWall, colBalance;
    int colDiricletVals; // u,v,w,k,e
    int numberinletnodes;
    std::list<BoundaryPatch *> patches;
};

class BoundaryPatch
{
public:
    BoundaryPatch(CoolEmAllToDatabase *c, int type, int balanceNumber);
    ~BoundaryPatch();
    CoolEmAllToDatabase *cd;
    std::string name;
    int instance;
    int type;
    std::list<BoundaryQuad *> quads;
    float area;
    float averageTemperature;
    float averagePressure;
    float averageVelo[3];
};
class BoundaryQuad
{
public:
    void computeArea();
    osg::Vec3f corners[4];
    float area;
    float averageTemperature;
    float averagePressure;
    float averageVelo[3];
};
#endif
