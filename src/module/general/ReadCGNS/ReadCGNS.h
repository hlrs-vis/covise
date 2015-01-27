/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_CGNS_H
#define _READ_CGNS_H
/**************************************************************************\
**                                   (C)2010 Stellba Hydro GmbH & Co. KG  **
**                                                                        **
** Description: READ CGNS CFD format                                      **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cgnslib.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <api/coModule.h>
using namespace covise;

class ReadCGNS : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);

    //  member data
    coOutputPort *p_mesh;
    coOutputPort *p_scalar_3D;
    coOutputPort *p_vector_3D;
    coOutputPort *p_boundaries;
    coOutputPort *p_scalar_2D;
    coOutputPort *p_vector_2D;

    coFileBrowserParam *p_cgnsMeshFile;
    coIntScalarParam *p_meshFileBaseNr;

    coFileBrowserParam *p_cgnsDataFile;
    coIntScalarParam *p_dataFileBaseNr;

    coStringParam *p_boundary;

    coChoiceParam *p_scalar;
    coChoiceParam *p_vector;
    coChoiceParam *p_boundScalar;
    coChoiceParam *p_boundVector;

    char **VectChoiceVal, **ScalChoiceVal, **RegionChoiceVal, **ZoneChoiceVal;
    int *VectIndex, *ScalIndex;
    std::vector<char *> FieldName; // stores original field name

    inline float sqrdist(float x1, float x2, float y1, float y2, float z1, float z2)
    {
        return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    }

    float findSmallestEdge(float *xCoords, float *yCoords, float *zCoords, int nElems, int *elemList, int *connList, int *typeList);
    int mergeNodes(float *x, float *y, float *z, int numNodes, float mergeTolerance, int *mapLocalToGlobal, int &nodesMerged);

    int createMeshDataZoneMapping(int meshFile, int dataFile, int meshBase, int dataBase, std::vector<char *> *meshZoneNames, int *zoneMapping);

public:
    ReadCGNS(int argc, char *argv[]);
    virtual ~ReadCGNS();
};

#endif
