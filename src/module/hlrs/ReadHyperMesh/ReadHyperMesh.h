/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ReadHyperMesh_H
#define _ReadHyperMesh_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadHyperMesh module                                      ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  6.2006                                                      ++
// ++**********************************************************************/
#include <math.h>
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <vector>

#define NUMRES 8
#define LINE_SIZE 1024
#define MAXTIMESTEPS 2048

class ReadHyperMesh : public coSimpleModule
{
public:
    ReadHyperMesh(int argc, char **argv);

private:
    //////////  inherited member functions
    virtual int compute(const char *port);
    void getLine();
    void pushBack(); // push back this line (don´t read a new on next time)
    bool pushedBack;
    coDistributedObject *readVec(const char *name);
    coDistributedObject *readScal(const char *name);

    char bfr[2048];
    char line[LINE_SIZE];

    std::vector<float> xPoints;
    std::vector<float> yPoints;
    std::vector<float> zPoints;
    std::vector<float> xData;
    std::vector<float> yData;
    std::vector<float> zData;
    std::vector<int> vertices;
    std::vector<int> types;
    std::vector<int> elements;
    char *names[NUMRES];
    std::vector<coDistributedObject *> dataObjects[NUMRES];

    FILE *file;
    char *c;
    ////////// ports
    coOutputPort *mesh;
    coOutputPort *dataPort[NUMRES];

    ///////// params
    coFileBrowserParam *fileName;
    coFileBrowserParam *resultsFileName;

    coBooleanParam *subdivideParam;
    coIntScalarParam *p_numt;
    coIntScalarParam *p_skip;
    coStringParam *p_Selection;

    ////////// member variables;
    int numberOfTimesteps;
    int numPoints;
    int numVert;
    int numElem;

    int skipped;
    bool doSkip;

    int numTimeSteps;
    int numData;
    int displacementData;
    float *dx, *dy, *dz;
    bool haveDisplacements;
    bool subdivide;
};
#endif
