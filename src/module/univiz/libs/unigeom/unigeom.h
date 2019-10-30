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

#ifdef VISTLE
#include <module/module.h>
#include <core/port.h>
#include <core/lines.h>
#include <core/polygons.h>
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

#ifdef VISTLE
#include "../vistle_ext/export.h"
class V_UNIVIZEXPORT UniGeom
#else
class UniGeom
#endif
{

public:
    typedef enum
    {
        GT_INVALID,
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

#ifdef VISTLE
    int geomType = GT_INVALID;
    vistle::Module *mod = nullptr;
    vistle::PortTask *task = nullptr;
    vistle::Object::const_ptr sourceObject;
    vistle::Port *inputPort = nullptr;
    vistle::Port *outputPort = nullptr;
    vistle::Lines::ptr outLine;
    vistle::Polygons::ptr outPoly;

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

#ifdef VISTLE
    UniGeom(vistle::Module *mod, vistle::Port *outport, vistle::Object::const_ptr src=vistle::Object::const_ptr());
    UniGeom(vistle::PortTask *task, vistle::Port *outport, vistle::Object::const_ptr src=vistle::Object::const_ptr());
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





#endif // _UNIGEOM_H_
