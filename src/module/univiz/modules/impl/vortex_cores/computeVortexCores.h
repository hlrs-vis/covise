/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Compute Vortex Core Lines according to Parallel Vectors
//
//  Ronald Peikert, Martin Roth, Dirk Bauer <=2005 and Filip Sadlo >=2006
//  Computer Graphics Laboratory, ETH Zurich

#include "linalg.h"
#include "unstructured.h"
#include "unigeom.h"
#include "unisys.h"

#define BIGINT 999999
#define BIGFLOAT 1e19f

#define VC_VOID -1
#define ALL_INTERSECTED 0x7

#define NEW(type, n) (type *) malloc((n) * sizeof(type))
#define NEW0(type, n) (type *) calloc((n), sizeof(type))

/* Algorithmic variants */
#define TRIANGLE 1
#define QUAD_NEWTON 2

typedef int *intPtr;

typedef struct
{
    float mat00, mat01, mat02, mat11, mat12, mat22;
    fvec3 rhs_u, rhs_v, rhs_w;
} LsfSums;

typedef struct
{
    fvec3 xyz; /* Position of the vertex                     */
    fvec3 velo; /* Direction of the two vector fields v and w */
    float lambda; /* Ratio of the two vectors v and w           */
    float strength; /* Vortex strength (see Martin Roth pp 57-59) */
    float quality; /* Feature quality (see Martin Roth pp 59-60) */
    float angle; /* Angle(tangent,velocity) in range [0,90]    */
    int sign; /* Sign of angle dot product v.t              */
    int linkF, linkB; /* Forward, backward neighbor vertices        */
    int cell1, cell2; /* Cells incident to face of the vertex       */
    bool used; /* Has the vertex already been traversed?     */
} Vertex;

typedef struct
{
    int nElems;
    Vertex *elem; /* Dynamic array of vertices */
} VertexList;

typedef struct
{
    int nSegments;
    int start;
} VC_Polyline;

typedef struct
{
    int nElems;
    VC_Polyline *elem;
} PolylineList;

typedef struct
{
    int nEdges; /* Number of edges in the grid */
    int *firstEdge; /* Array of length ucd->nnodes+1 */
    int *otherNode; /* Array of length nEdges */

    int nFaces; /* Number of faces in the grid */
    int *firstFace; /* Array of length ucd->nnodes+1 */
    int *oppositeNode; /* Array of length nFaces */
} UCD_connectivity;

//extern Visun vu;

extern UCD_connectivity *computeConnectivity(Unstructured *unst);

extern float *computeGradient(Unstructured *unst, int compV, UCD_connectivity *ucdc);
extern void deleteUcdConnectivity(UCD_connectivity *ucdc);

extern float *computeVorticity(float *gradient, int nNodes);
extern float *computeAcceleration(float *gradient, Unstructured *unst, int compV);
extern float computeExtent(Unstructured *unst);

extern void deleteVertexList(VertexList *vertexList);

extern PolylineList *generatePolylines(VertexList *vertexList);
extern PolylineList *prunePolylines(PolylineList *polylineList, VertexList *vertexList);
extern void deletePolylineList(PolylineList *polylineList);

extern void computeFeatureQuality(VertexList *vertexList);

extern void generateOutputGeometry(UniSys *vu,
                                   VertexList *vertexList, PolylineList *polylineList,
                                   int min_vertices, int max_exceptions,
                                   float min_strength, float max_angle,
                                   UniGeom *ugeom);

extern VertexList *findParallel(
    UniSys *vu,
    Unstructured *unst, UCD_connectivity *ucdc,
    int compV, fvec3 *w, fmat3 *gradient,
    int variant, float extent);
