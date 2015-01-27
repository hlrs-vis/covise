/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Unification Library for Modular Visualization Systems
//
// Geometry
//
// CGL ETH Zuerich
// Filip Sadlo 2006 - 2008

// Usage: define AVS or COVISE or VTK but not more than one
//        define also COVISE5 for Covise 5.x

#ifndef _UNIGEOM_H_
#define _UNIGEOM_H_

#define UNIGEOM_VERSION "0.01"

#include <vector>

#include "linalg.h"

#ifdef AVS
#include <avs/avs.h>
#include <avs/geom.h>
#endif

#ifdef COVISE
#ifdef COVISE5
#include <coModule.h>
#else
#include <api/coModule.h>
#endif
#endif

#ifdef VTK
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#endif

using namespace std;

#ifdef COVISE
using namespace covise;
#endif

class UniGeom
{

public:
    typedef enum
    {
        GT_LINE,
        GT_POLYHEDRON
    } geomTypeEnum;

private:
#ifdef AVS
    GEOMedit_list *avsGeomEditList;
    GEOMobj *avsGeomObj;
#endif

#ifdef COVISE
    int geomType;
#ifdef COVISE5
    coOutPort *covGeom;
    coOutPort *covNormals;
#else
    coInputPort *covGeomIn;
    coOutputPort *covGeom;
    coOutputPort *covNormals;
#endif

    // DO_Lines
    // interleaved 3-space float positions
    std::vector<std::vector<float> > lines;

    // vertices e.g. for DO_Polygons
    // interleaved 3-space float positions
    std::vector<float> vertices;

    // normals e.g. for DO_Polygons
    // interleaved 3-space float vectors
    std::vector<float> normals;

    // DO_Polygons
    std::vector<std::vector<int> > polygons;
#endif

#ifdef VTK
    int geomType;
    vtkPolyData *vtkPolyDat;
    vtkPoints *outputPoints;
    vtkCellArray *outputCells;
    vtkFloatArray *outputNormals;
    int numPoints, numCells;
#endif

public:
#ifdef AVS // AVS constructor
    UniGeom(GEOMedit_list *geom);
#endif

#ifdef COVISE // Covise constructor
#ifdef COVISE5
    UniGeom(coOutPort *geom, coOutPort *normals = NULL);
#else
    UniGeom(coOutputPort *geom, coOutputPort *normals = NULL);
    UniGeom(coInputPort *geom);
#endif
#endif

#ifdef VTK // VTK constructor
    UniGeom(vtkPolyData *vtkPolyDat);
#endif

    ~UniGeom();

    // create object
    void createObj(int geomType);

    // add polyline to object
    void addPolyline(float *vertices, float *colors, int nvertices);

    // add vertices
    void addVertices(float *vertices, int nvertices);

    void getVertex(int vertex, vec3 pos);
    int getVertexCnt(void);

    // here, vertex indices are zero-based!
    // note that AVS uses 1-based indices!
    void addPolygon(int nvertices, int *indices);

    // generate normals
    void generateNormals(void);

    // assign and destroy object
    void assignObj(const char *name);
};

inline void UniGeom::getVertex(int vertex, vec3 pos)
{
#ifdef AVS
    // ###### TODO: HACK: fixed to polyhedra
    pos[0] = PH(avsGeomObj).verts.l[vertex * 3 + 0];
    pos[1] = PH(avsGeomObj).verts.l[vertex * 3 + 1];
    pos[2] = PH(avsGeomObj).verts.l[vertex * 3 + 2];
#endif

#ifdef COVISE
    if (covGeom)
    {
        pos[0] = vertices[(vertex)*3 + 0];
        pos[1] = vertices[(vertex)*3 + 1];
        pos[2] = vertices[(vertex)*3 + 2];
    }
    else if (covGeomIn)
    {
        if (geomType == GT_LINE)
        {
            coDoLines *lin = ((coDoLines *)covGeomIn->getCurrentObject());
            float *x, *y, *z;
            int *corL, *linL;
            lin->getAddresses(&x, &y, &z, &corL, &linL);
            pos[0] = x[vertex];
            pos[1] = y[vertex];
            pos[2] = z[vertex];
        }
        else if (geomType == GT_POLYHEDRON)
        {
            printf("UniGeom::getVertex not yet implemented for polygon type\n");
        }
    }
#endif

#ifdef VTK
    outputPoints->GetPoint(vertex, pos);
#endif
}

inline int UniGeom::getVertexCnt(void)
{
#ifdef AVS
    printf("UniGeom::getVertexCnt: not yet implemented\n");
#endif
#ifdef COVISE
    if (covGeom)
    {
        printf("UniGeom::getVertexCnt: not yet implemented for output object\n");
    }
    else if (covGeomIn)
    {
        if (geomType == GT_LINE)
        {
            coDoLines *lin = ((coDoLines *)covGeomIn->getCurrentObject());
            return lin->getNumVertices();
        }
        else if (geomType == GT_POLYHEDRON)
        {
            printf("UniGeom::getVertexCnt: not yet implemented for polygon type\n");
        }
    }
#endif
#ifdef VTK
    printf("UniGeom::getVertexCnt: not yet implemented\n");
#endif
    return 0;
}

#endif // _UNIGEOM_H_
