/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Unification Library for Modular Visualization Systems
//
// Unstructured Field
//
// CGL ETH Zurich
// Ronald Peikert and Filip Sadlo 2006 - 2008

// Usage: define at least AVS, COVISE or VTK, or two, or all
//        define also COVISE5 for Covise 5.x
//        define UNST_DATA_DICT for access to transient data using DataDict

// TODO:
// - 2007-11-20: VTK: change all cell/node ID types to vtkIdType.
//   do by defining an own ID-type and define it to vtkIdType in case of VTK
// - suport 2D cells, actually AVS constructor does not skip them: BAD HACK ###
// - time-dependent data:
//   - probably allow multiple formats: DataDict, Ensight, (CFX? slow, caching)
// - support "double type" data channels and coordinates (postponed)

#ifndef _UNSTRUCTURED_H_
#define _UNSTRUCTURED_H_

#define UNST_VERSION "0.12"

#include <vector>
#include <string>
#include "linalg.h"
#ifdef UNST_DATA_DICT
#include "dataDict.h"
#endif

#ifdef AVS
// AVS
#include <avs/avs.h>
#include <avs/ucd_defs.h>
#include "avs_ext.h"
#endif

#ifdef COVISE
// Covise
#ifdef COVISE5
#include <coModule.h>
#else
#include <api/coModule.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#endif
#endif

#ifdef VTK
// ### TODO: clean up, some are probably not needed and others can go to .cpp
#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkCellArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkFloatArray.h"
#include "vtkIdList.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkUnstructuredGrid.h"
#endif

using namespace std;

//#define CELL_VERTICES 0x1
//#define CELL_DATA     0x2
//#define CELL_TETS     0x10

#define VECLEN_MAX 128

extern int nVertices[8];
extern int nFaces[8];
extern int nQuads[8];
extern int nEdges[8];

extern int faces[8][6][4];

extern int faceNodeCnt[8][6];

extern int edges[8][12][2];

// connected order means that the following invariant holds:
// the visited nodes always are inside one connected component when
// the edges are visited in order
// additionally, node zero is contained (visited) in edge zero
extern int edgesConnectedOrder[8][12][2];

// Tet decomposition by splitting quads into 4 tris
extern int nTets[8];
extern int tets[8][24][4];

struct SearchGrid
{
    ivec3 dim; // Number of boxes in each dimension
    fvec3 min, max, step;
    vector<int> *box; // Array of vectors of cell indices
    int findBox(vec3 xyz); // -1 if nonexisting
    int findBoxes(vec3 xyz, int *boxes); // returns number of boxes
};

class DataDesc
{
    // TODO: actually only supporting interleaved vector data (stride = veclen)
public:
    int operationId; // see operationIdEnum
    int dataType; // see dataTypeEnum
    int veclen;
#if 0
  union p {
    float *f;
    vec3 *v3;
    fmat3 *fm3;
  } p;
#else
    union p
    {
        float *f;
    } p;
#endif

    DataDesc(int opId, int dataT, int vecl, float *data)
    {
        operationId = opId;
        dataType = dataT;
        veclen = vecl;
        p.f = data;
    }
};

struct CellInfo
{
    int index;
    int type; // gives access to nNodes, nFaces, nEdges, nTets etc.
    int tet; // index of current tet in tet decomposition (0..23)
    int node[8]; // index of vertices in vertex list
    double radiusSqr;
    vec3 vertex[15]; // max 8+1+6 original and interpolated vertices (for hex type)
    double wallDist[15]; // effective wall distance or binary value 0=boundary 1=interior
    bool computeWeights(vec3 xyz, double wt[8]);
    void computeGradientWeights(vec3 wt[8]); // for current tet in current cell
    bool atWall;

    vec3 vec[15];
    double scal[15];
    int currCell_scalarComponent; // scal was set from this component
    int currCell_vectorComponent; // vec was set from this component
    int currCell_wallDistComponent; // wallDist  was set from this component
    DataDesc *currCell_scalarExtraData; // scal was set from this extraData (NULL if from component)
    DataDesc *currCell_vectorExtraData; // vec was set from this extraData (NULL if from component)
    double currCell_time;
    bool dataLoaded;
};

struct CriticalPoint
{
    vec3 coord;
    double wallDist;
    int cell;
    int type;
    bool allReal;
    double eigenvalue[3];
    vec3 eigenvector[3];
};

typedef struct NodeCompDataPtr
{
    std::vector<float *> ptrs; // pointer to each component of vectorial UCD component
    int stride;
} NodeCompDataPtr;

class Unstructured
{

public:
    typedef enum
    {
        TP_FLOAT
    } dataTypeEnum;

    typedef enum
    {
        OP_GRADIENT,
        OP_EIGENVECTOR0,
        OP_EIGENVECTOR1,
        OP_EIGENVECTOR2
    } operationIdEnum;

    typedef enum
    {
        CONN_FACE,
        CONN_NODE
    } connectivityEnum;

    // consistent with AVS and Covise, except for point (TODO actually unsupported) and Hex (is 7 in AVS and ...)
    typedef enum
    {
        CELL_POINT = 0,
        CELL_LINE = 1,
        CELL_TRI = 2,
        CELL_QUAD = 3,
        CELL_TET = 4,
        CELL_PYR = 5,
        CELL_PRISM = 6,
        CELL_HEX = 7
    } cellTypeEnum;

    typedef enum
    {
        GRAD_LS,
        GRAD_MEYER
    } gradientMethodEnum;

    char *name;
    int nCells;
    int nNodes;

private:
    CellInfo currCell;

    int *nodeComponents;
    char **nodeComponentLabels;
    std::vector<float *> allocatedNodeDataComponents;
    std::vector<std::vector<DataDesc *> > nodeCompExtraData;
    bool isCopy;
    bool nameAllocated;
    bool nodeListAllocated;
    bool nodeListOffsetsAllocated;
    bool cellTypesAllocated;
    bool coordinatesAllocated;

    bool transient;
#ifdef UNST_DATA_DICT
    DataDict *transientDataDict;
#else
    void *transientDataDict;
#endif
    std::vector<string> transientFiles;
    std::vector<std::vector<float> > transientFilesTimeSteps;
    int transientFileIdx;
    float *transientFile;
    int transientFileSize; // in bytes, for mmap() and munmap()
    // TODO: move to own class
    int lastTimeStepL, lastTimeStepU;
    double lastTimeL, lastTimeU;
    double lastWeightL, lastWeightU;
    int transientFileVerbose;
    int transientFileMapNb;
    int transientZoneRotating; // the zone id of rotating zone, otherwise -1
    vec3 transientZoneRotationCenter;
    vec3 transientZoneRotationVector; // includes angular speed in rad / sec (includes angle in radiants rotated within 1 second)

public:
    std::vector<NodeCompDataPtr> nodeComponentDataPtrs;

private:
    fvec3 *cellCentroid;
    float *cellRadiusSqr;

    float *x;
    float *y;
    float *z;

    int vectorComponent;
    DataDesc *vectorComponentExtraData; // points to selected extra data, NULL otherwise
    int scalarComponent;
    DataDesc *scalarComponentExtraData; // points to selected extra data, NULL otherwise
    int wallDistComponent;
    int zoneComponent;
    int nodeComponentNb;
    bool strictNodeCompExtraDeletion;
    int nodeComponentVecLenTot;
    std::vector<int> nodeComponentVecLenCumulated;

    //void (*vector3CB)(vec3 pos, vec3 out, double time=0.0); // default not allowed?
    void (*vector3CB)(vec3 pos, vec3 out, double time);
    vec3 vector3CBPosition;
    int vector3CBBackupComp;
    bool vector3CBRestrictToGrid;

    float *u;
    float *v;
    float *w;
    float *p;
    float *wallDist;

    int vStride; // stride for vector data
    int sStride; // stride for scalar data
    int cStride; // stride for coordinate data
    bool vectorExists;
    bool scalarExists;
    bool wallDistExists;

    fvec3 boundingBoxMin, boundingBoxMax;
    SearchGrid *searchGrid;
    int *cellType;
    int *nodeList;
    int *nodeListOffset;

    int edgeNb;
    int faceNb;

public:
    bool divideVelocityByWalldist;
    bool extrapolateToNoSlipBoundary;

private:
    vector<int> *nodeNeighbors; // Neighbor nodes (for derivatives, filters etc.)
    vector<int> *cellNeighbors; // Neighbor cells
    std::vector<int> cellFaceNeighborsCnt; // number of neighbor cells (cellNeighbors) that share a face with given cell

public:
    vector<CriticalPoint> criticalPoints;
    vector<CriticalPoint> criticalBoundaryPoints;

private:
    double weight[8];
    vec3 gradientWeight[8];

    void setupUnstructured(Unstructured *templ, DataDesc *dd);
    void setupUnstructured(Unstructured *templ, std::vector<DataDesc *> *dv);

public:
#ifdef AVS
    // AVS constructor
    Unstructured(UCD_structure *ucd);
#endif

#ifdef COVISE
// Covise constructor
#ifdef COVISE5
    Unstructured(DO_UnstructuredGrid *grid,
                 std::vector<DO_Unstructured_S3D_Data *> *scal,
                 std::vector<DO_Unstructured_V3D_Data *> *vect);
#else
    Unstructured(covise::coDoUnstructuredGrid *grid,
                 std::vector<covise::coDoFloat *> *scal,
                 std::vector<covise::coDoVec3 *> *vect);
#endif
#endif

#ifdef VTK
    // VTK (Paraview) constructor
    Unstructured(vtkUnstructuredGrid *grid,
                 std::vector<vtkFloatArray *> *scal,
                 std::vector<vtkFloatArray *> *vect);
#endif

    // system independent constructors
    Unstructured(Unstructured *templ, DataDesc *dd);
    Unstructured(Unstructured *templ, int veclen);
    // ######### TODO: this method is buggy, see ucd_ridge_surface !!
    Unstructured(Unstructured *templ, int componentCnt, int *components);
    Unstructured(const char *fileName);
    ~Unstructured();

    void printInfo(FILE *fp);
    int getNodeListSize();
    bool saveAs(const char *fileName);
#ifdef UNST_DATA_DICT
    void setTransient(char *dataDictDir, int cacheSize = 3, bool verbose = true);
#endif
    // this version does not use DataDict
    void setTransientFile(const char *dataInfoFile, int verbose = 2);
    void unsetTransientFile(); // used to "clear" mmap cache for saving space
    int getTransientFileMapNb(void);

    // sets callback for vector3, in this case this overlays all components
    // (not only vector components !!) -> components are not supported in
    // callback mode
    // restrictToGrid: if true, positions outside grid are not found,
    // SETTING TO FALSE IS DANGEROUS !!! (findCells() returns that cell has been
    // found although it is not found (only vector3CBPosition is set)
    void setVector3CB(void (*vec3CB)(vec3 pos, vec3 out, double time), bool restrictToGrid = true);
    void unsetVector3CB(void);

private:
    // private because indices may change after removal of extra data
    // (use of indices into extra data is to be avoided)
    int findNodeCompExtraDataIdx(int comp, int operationId);
    int findNodeCompExtraDataCompIdx(DataDesc *dd, int &comp, int &idx);

    void computeNodeNeighborsRek(int node, int range, int *neighborsSorted, int *neighborsSortedCnt);
    void getCellNeighborsN_rek(int cell, int ncell, int range, int connectivity, std::vector<int> *neighbors, int *cellSizes = NULL);
    int getDataDictFloatIdx(int node, int unstComp, int vectorComp);
    int getTimeSteps(double time, int &step1, int &step2, double &weight1, double &weight2);
    float getTransientFileEntry(int node, int ucdComponent, int vectorComponent, int timeStep);
    void mapTransientFile(int fileIdx); // don't access, use setTransientFile
    void rotateWithTransientZone(vec3 pnt, double time, vec3 out);
    void computeGradientMeyer(Unstructured *unst, int compIn, int range, Unstructured *unst_out, int compOut);

public:
    void getCellCentroid(int cell, vec3 centroid, double time = 0.0);
    void getCellCentroid(int cell, fvec3 centroid, double time = 0.0);

public:
    static int findInSortedIntArray(int key, int *arr, int arr_len);
    static int insertInSortedIntArray(int key, int ascending, int *arr, int *arr_len);

    void computeBoundingBox();
    void computeCentroids();
    void computeMinBoxSize(fvec3 minSize);
    void setupSearchGrid();

    void computeNodeNeighbors();
    int computeNodeNeighborsN(int node, int range, int *neighborsN);
    void deleteNodeNeighbors();

    void computeCellNeighbors();
    void deleteCellNeighbors();
    std::vector<int> *getCellNeighbors(int);
    int cellNeighborCnt(int cell, int connectivity);
    bool domainBoundaryCell(int cell);
    bool domainBoundaryCellNode(int node);
    std::vector<int> getCellNeighborsN(int cell, int range, int connectivity, int *cellSizes = NULL);

    bool computeWallNormal(vec3 nml);

    void computeCriticalPoints();
    void deleteCriticalPoints();

    void computeCriticalBoundaryPoints();
    void deleteCriticalBoundaryPoints();

    //void computeVelocityGradient();
    //void deleteVelocityGradient();
    void gradient(int comp, vec3 position, float *output, double time = 0.0, int range = 1, bool *omitNodes = NULL, float *gradDefault = NULL, std::vector<int> *lonelyNodes = NULL, bool forceSymmetric = false);
    DataDesc *gradient(int comp, bool onlyGetID = false, int range = 1, bool *omitNodes = NULL, float *gradDefault = NULL, std::vector<int> *lonelyNodes = NULL, bool forceSymmetric = false, std::vector<int> *nodes = NULL, double time = 0.0);
    void gradient(int comp, Unstructured *out, int outComp, int range = 1, bool *omitNodes = NULL, float *gradDefault = NULL, std::vector<int> *lonelyNodes = NULL, bool forceSymmetric = false, std::vector<int> *nodes = NULL, double time = 0.0, int method = GRAD_LS);
    void transpose(int matComp, Unstructured *out, int outComp);
    void matrixInvert(int matComp, Unstructured *out, int outComp, bool mode2D = false);
    //DataDesc *gradient(int comp, bool onlyGetID=false, int *id=NULL);

    char *getName(void);
    int getEdgeNb(void);
    int getFaceNb(void);
    int getCellType(int cell)
    {
        return cellType[cell];
    }
    int getCellIndex();
    double getCellRadius(void);
    double getCellRadius(int cell);
    double getCellRadiusSqr(void);
    double getCellRadiusSqr(int cell);

    int getNodeCompNb(void);
    int getNodeVecLenTot(void);
    char *getNodeCompLabel(int comp);

    float getScalar(int node);
    float getScalar(int node, double time);
    float getScalar(int node, int comp, double time);
    float getScalar(int node, int comp);
    float getScalar(int node, int comp, int opId);
    void setScalar(int node, float scal);
    void setScalar(int node, int comp, float scal);
    void setScalar(int node, int comp, int opId, float scal);
    void getVector3(int node, vec3 vec);
    void getVector3(int node, double time, vec3 vec);
    void getVector3(int node, int comp, double time, vec3 vec);
    void getVector3(int node, fvec3 vec);
    void getVector3(int node, int comp, vec3 vec);
    void getVector3(int node, int comp, fvec3 vec);
    void getVector3(int node, int comp, int opId, vec3 vec);
    // TODO:  void getVelocity(int node, int comp, DataDesc *extDat, vec3 vec);
    void setVector3(int node, vec3 vec);
    void setVector3(int node, int comp, vec3 vec);
    void setVector3(int node, int comp, int opId, vec3 vec);
    void getVectorN(int node, int comp, double *vec);
    void getMatrix3(int node, fmat3 mat);
    void getMatrix3(int node, int comp, fmat3 mat);
    void getMatrix3(int node, int comp, int opId, fmat3 mat);
    void setMatrix3(int node, fmat3 mat);
    void setMatrix3(int node, int comp, fmat3 mat);
    void setMatrix3(int node, int comp, int opId, fmat3 mat);
    void getCoords(int node, vec3 xyz, double time = 0.0);
    void getCoords(int node, fvec3 xyz, double time = 0.0);
    double getWallDist(int node);
    int getCellZone(int cell, double time);
    int getNodeCompVecLen(int comp);
    float *getNodeMatrixCompPtr(int comp);

    DataDesc *newNodeCompExtraData(int comp, int dataType, int veclen, int operationId);
    void deleteNodeCompExtraData(DataDesc *dd);
    void deleteNodeCompExtraData(int comp, int operationId);
    void deleteNodeCompExtraData(void);
    DataDesc *findNodeCompExtraData(int comp, int operationId);

    int *getCellNodesAVS(void);
    int *getCellNodesAVS(int cell);
    void getCellEdgeNodesConnOrderAVS(int cell, int locEdgeIdx, int &v1, int &v2);
    void getCellEdgeLocNodesConnOrderAVS(int cell, int locEdgeIdx, int &v1, int &v2);
    static void nodeOrderAVStoCovise(int cellType, int *nodesIn, int *nodesOut);
    static void nodeOrderAVStoVTK(int cellType, int *nodesIn, int *nodesOut);
    CellInfo &loadCell(int i, bool complete = false, double time = 0.0); // time ignored if !complete || !transient

    std::vector<int> getNodeCells(int node);

    void loadCellData(double time = 0.0); // time ignored if !transient
    void assureCellData(double time = 0.0); // time ignored if !transient

    bool findCell(vec3 xyz, double time = 0.0); // Set current cell and interpolation weights, time is ignored if !transient
    std::vector<int> findCells(vec3 bboxMin, vec3 bboxMax);
    std::vector<int> findCellsInZ(vec3 xyz);
    void keepOffWall(vec3 xyz, double frac);

    void selectVectorNodeData(int comp); // Select node data component for vector quantity
    void selectVectorNodeData(int comp, int opId);
    void selectScalarNodeData(int comp); // Select node data component for scalar quantity
    void selectScalarNodeData(int comp, int opId);
    void selectWallDistNodeData(int comp); // Select node data compont for wall distance
    void selectWallDistNodeData(int comp, int opId);

    int getVectorNodeDataComponent(void); // Get selected vector component
    int getScalarNodeDataComponent(void); // Get selected scalar component

    // TODO: interpolateVectorN
    void interpolateVector3(vec3 vec, double time = 0.0, bool orientateVectors = false, vec3 direction = NULL); // Interpolate at current position, time ignored if !transient
    void interpolateMatrix3(mat3 mat); // Interpolate at current position
    void interpolateVectorGradient(mat3 grad);
    double interpolateScalar();
    double interpolateWallDist();
    void interpolatedMatrixEigenvector3(int evIdx, vec3 ev, vec3 direction = NULL, bool absoluteSorted = false);
    bool interpolatedMatrixEigenvaluesDesc3(vec3 evals);

    // First arg is read/write.  Returns integration time
    double integrate(vec3 xyz, bool forward, double maxTime, double maxDist, int maxSteps, int order = 4, bool keepOffWall = false, double startTime = 0.0, double *distDone = NULL, bool tensorLine = false, vec3 direction = NULL, bool interpolateTensor = false, int evIdx = 0);

    void matVec(int matComp, Unstructured *uvec, int vecComp,
                Unstructured *out, int outComp);
    void matVec(int matComp, int opId, Unstructured *uvec, int vecComp,
                Unstructured *out, int outComp);
    void matMat(int matComp1, Unstructured *umat, int matComp2,
                Unstructured *out, int outComp);
    void det(int matComp, Unstructured *out, int outComp);
    void realEigenvaluesSortedDesc(int matComp, bool absoluteSorted, Unstructured *out, int outComp);
    DataDesc *realEigenvectorSortedDesc(int matComp, bool absoluteSorted, int index, bool suppressWarnings);
    void realEigenvectorSortedDesc(int matComp, bool absoluteSorted, int index, Unstructured *out, int outComp, bool suppressWarnings = false);
    void realEigenvectorSortedAsc(int matComp, bool absoluteSorted, int index, Unstructured *out, int outComp, bool suppressWarnings = false);
    void gradMulVec(int quantComp, Unstructured *uvec, int vecComp,
                    Unstructured *out, int outComp);
};

// --- inlined accessors -------------------------------------------------------

inline char *Unstructured::getName(void)
{
    return name;
}

inline int Unstructured::getNodeCompNb(void)
{
    // TODO: replace when introducing e.g. scalarCB, e.g. by fake components:
    //       first component stores vector3CB, second stores scalarCB
    if (vector3CB)
    {
        return 1;
    }

    return nodeComponentNb;
}

inline int Unstructured::getNodeVecLenTot(void)
{
    // TODO: replace when introducing e.g. scalarCB, e.g. by fake components:
    //       first component stores vector3CB, second stores scalarCB
    if (vector3CB)
    {
        return 3;
    }

    return nodeComponentVecLenTot;
}

inline char *Unstructured::getNodeCompLabel(int comp)
{
    // TODO: replace when introducing e.g. scalarCB, e.g. by fake components:
    //       first component stores vector3CB, second stores scalarCB
    if (vector3CB)
    {
        return NULL; // ### HACK
    }

    if (nodeComponentLabels)
        return nodeComponentLabels[comp];
    else
        return NULL;
}

inline int Unstructured::cellNeighborCnt(int cell, int connectivity)
{
    if (!cellNeighbors)
        computeCellNeighbors();

    switch (connectivity)
    {
    // DELETEME case CONN_FACE: return nFaces[getCellType(cell)]; break;
    case CONN_FACE:
        return cellFaceNeighborsCnt[cell];
        break;
    case CONN_NODE:
        return getCellNeighbors(cell)->size();
        break;
    default:
        printf("Unstructured::cellNeighborCnt: unsupported connectivity\n");
    }

    return 0;
}

inline bool Unstructured::domainBoundaryCell(int cell)
{ // returns true if cell is at domain boundary

    int cellNeighNb = cellNeighborCnt(cell, CONN_FACE);

    if (cellNeighNb != nFaces[getCellType(cell)])
    {
        // cell at domain boundary
        return true;
    }
    return false;
}

inline std::vector<int> *Unstructured::getCellNeighbors(int cell)
{ // returns all cells that share at least one node with given cell
    // 'cellFaceNeighborsCnt[cell]' cells sharing a face come first

    if (!cellNeighbors)
        computeCellNeighbors();

    //printf("cell=%d neighbors=", cell);
    //for (int i=0; i<(int)cellNeighbors[cell].size(); i++) {
    //  printf(" %d", cellNeighbors[cell][i]);
    //}
    //printf("\n");

    return &cellNeighbors[cell];
}

inline void Unstructured::getCoords(int node, vec3 xyz, double time)
{
    xyz[0] = x[node * cStride];
    xyz[1] = y[node * cStride];
    xyz[2] = z[node * cStride];
    if (time == 0.0 || transientZoneRotating < 0)
    {
        return;
    }
    else
    {
        int zone = (int)getScalar(node, zoneComponent);
        if (zone != transientZoneRotating)
        {
            // node is not rotating
            return;
        }
        else
        {
            // node is rotating
            rotateWithTransientZone(xyz, time, xyz);
        }
    }
}

inline void Unstructured::getCoords(int node, fvec3 xyz, double time)
{
    vec3 xyzd;
    getCoords(node, xyzd, time);
    vec3tofvec3(xyzd, xyz);
}

inline int Unstructured::getDataDictFloatIdx(int node, int unstComp, int vectorComp)
{ // TODO: assumes dataDict identical to unst (all comp, same order)
    //       assumes interleaved dataDict

    int idx = 0;
    for (int c = 0; c < unstComp; c++)
    {
        idx += getNodeCompVecLen(c) * nNodes;
    }
    idx += node * getNodeCompVecLen(unstComp) + vectorComp;

    return idx;
}

inline int Unstructured::getTransientFileMapNb(void)
{
    return transientFileMapNb;
}

inline float Unstructured::getScalar(int node)
{
    if (transient)
        printf("Unstructured::getScalar error: no transient data, returning reference time step\n");

    return p[sStride * node];
}

inline float Unstructured::getScalar(int node, double time)
{
    if (!transient || (!transientDataDict && !transientFile))
    {
#if 0 // DELETEME
    printf("Unstructured::getScalar: error: no transient DataDict\n");
    return 0.0;
#else
        return getScalar(node);
#endif
    }

    if (transientDataDict)
    {
#ifdef UNST_DATA_DICT
        return transientDataDict->interpolate(time, getDataDictFloatIdx(node, scalarComponent, 0));
#endif
    }
#if 0
  //else { // transientFile
    printf("Unstructured::getScalar: error: transientFile mode not yet implemented\n");
    return 0.0;
    //}
#else // ### TODO:
    else
    { // transientFile

        int step1, step2;
        double weight1, weight2;
        int res = getTimeSteps(time, step1, step2, weight1, weight2);
        if (res < 1)
        {
            printf("Unstructured::getScalar: error: time step not found\n");
            return 0.0;
        }
        int scalComp = getScalarNodeDataComponent();

        // not ok (actually only single component in file)
        //###
        return weight1 * getTransientFileEntry(node, scalComp, 0, step1) + weight2 * getTransientFileEntry(node, scalComp, 0, step2);
    }
#endif

    return 0.0;
}

inline float Unstructured::getScalar(int node, int comp, double time)
{
    if (!transient || (!transientDataDict && !transientFile))
    {
#if 0 // DELETEME
    printf("Unstructured::getScalar: error: no transient DataDict\n");
    return 0.0;
#else
        return getScalar(node, comp);
#endif
    }

    if (transientDataDict)
    {
#ifdef UNST_DATA_DICT
        // ### not tested
        return transientDataDict->interpolate(time, getDataDictFloatIdx(node, comp, 0));
#endif
    }
#if 0
  //else { // transientFile
    printf("Unstructured::getScalar: error: transientFile mode not yet implemented\n");
    return 0.0;
    //}
#else // ### TODO
    else
    { // transientFile
        int step1, step2;
        double weight1, weight2;
        int res = getTimeSteps(time, step1, step2, weight1, weight2);
        if (res < 1)
        {
            printf("Unstructured::getScalar: error: time step not found\n");
            return 0.0;
        }

        // not ok (actually only single component in file)
        //###############
        return weight1 * getTransientFileEntry(node, comp, 0, step1) + weight2 * getTransientFileEntry(node, comp, 0, step2);
    }
#endif

    return 0.0;
}

inline float Unstructured::getScalar(int node, int comp)
{
    if (transient)
        printf("Unstructured::getScalar error: no transient data, returning reference time step\n");

    float *ptr = nodeComponentDataPtrs[comp].ptrs[0];
    int sStride = nodeComponentDataPtrs[comp].stride;
    return ptr[sStride * node];
}

inline float Unstructured::getScalar(int node, int comp, int opId)
{
    if (transient)
        printf("Unstructured::getScalar error: no transient data, returning reference time step\n");

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    float *ptr = dd->p.f;
    int sStride = dd->veclen; // dd must be actually interleaved
    return ptr[sStride * node];
}

inline void Unstructured::setScalar(int node, float scal)
{
    p[sStride * node] = scal;
}

inline void Unstructured::setScalar(int node, int comp, float scal)
{
    float *ptr = nodeComponentDataPtrs[comp].ptrs[0];
    int sStride = nodeComponentDataPtrs[comp].stride;
    ptr[sStride * node] = scal;
}

inline void Unstructured::setScalar(int node, int comp, int opId, float scal)
{
    DataDesc *dd = findNodeCompExtraData(comp, opId);
    float *ptr = dd->p.f;
    int sStride = dd->veclen; // dd must be actually interleaved
    ptr[sStride * node] = scal;
}

inline void Unstructured::getVector3(int node, vec3 vec)
{
    if (transient)
        printf("Unstructured::getVector3 error: no transient data, returning reference time step\n");

    if (vector3CB)
    {
        vec3 pos;
        getCoords(node, pos); // TODO: no support for rotating zones
        //vector3CB(pos, vec); // not using time (time = 0.0)
        vector3CB(pos, vec, 0.0); // not using time (time = 0.0)
        return;
    }

    vec[0] = u[vStride * node];
    vec[1] = v[vStride * node];
    vec[2] = w[vStride * node];

    if (divideVelocityByWalldist && wallDistExists)
    {
        double wd = getWallDist(node);
        if (wd == 0)
            wd = 1e-9;
        vec3scal(vec, 1. / wd, vec);
    }
}

inline void Unstructured::getVector3(int node, fvec3 vec)
{
    vec3 v;
    getVector3(node, v);
    vec[0] = v[0];
    vec[1] = v[1];
    vec[2] = v[2];
}

inline void Unstructured::getVector3(int node, double time, vec3 vec)
{
    if (!transient || (!transientDataDict && !transientFile))
    {
#if 0 // DELETEME
    printf("Unstructured::getVector3: error: no transient DataDict\n");
    return;
#else
        vec3 v;
        getVector3(node, v);
        vec3copy(v, vec);
        return;
#endif
    }

    if (vector3CB)
    {
        vec3 pos;
        getCoords(node, pos); // TODO: no support for rotating zones
        vector3CB(pos, vec, time);
        return;
    }

    if (transientDataDict)
    {
#ifdef UNST_DATA_DICT
        vec[0] = transientDataDict->interpolate(time, getDataDictFloatIdx(node, vectorComponent, 0));
        vec[1] = transientDataDict->interpolate(time, getDataDictFloatIdx(node, vectorComponent, 1));
        vec[2] = transientDataDict->interpolate(time, getDataDictFloatIdx(node, vectorComponent, 2));
#endif
    }
    else
    { // transientFile
        int step1, step2;
        double weight1, weight2;
        int res = getTimeSteps(time, step1, step2, weight1, weight2);
        if (res < 1)
        {
            printf("Unstructured::getVector3: error: time step not found\n");
            return;
        }
        int vecComp = getVectorNodeDataComponent();

        vec[0] = weight1 * getTransientFileEntry(node, vecComp, 0, step1) + weight2 * getTransientFileEntry(node, vecComp, 0, step2);
        vec[1] = weight1 * getTransientFileEntry(node, vecComp, 1, step1) + weight2 * getTransientFileEntry(node, vecComp, 1, step2);
        vec[2] = weight1 * getTransientFileEntry(node, vecComp, 2, step1) + weight2 * getTransientFileEntry(node, vecComp, 2, step2);
    }

    if (divideVelocityByWalldist && wallDistExists)
    {
        double wd = getWallDist(node);
        if (wd == 0)
            wd = 1e-9;
        vec3scal(vec, 1. / wd, vec);
    }
}

inline void Unstructured::getVector3(int node, int comp, double time, vec3 vec)
{
    if (!transient || (!transientDataDict && !transientFile))
    {
#if 0 // DELETEME
    printf("Unstructured::getVector3: error: no transient DataDict\n");
    return;
#else
        vec3 v;
        getVector3(node, comp, v);
        vec3copy(v, vec);
        return;
#endif
    }

    if (vector3CB)
    {
        vec3 pos;
        getCoords(node, pos); // TODO: no support for rotating zones
        vector3CB(pos, vec, time);
        return;
    }

    if (transientDataDict)
    {
#ifdef UNST_DATA_DICT
        vec[0] = transientDataDict->interpolate(time, getDataDictFloatIdx(node, comp, 0));
        vec[1] = transientDataDict->interpolate(time, getDataDictFloatIdx(node, comp, 1));
        vec[2] = transientDataDict->interpolate(time, getDataDictFloatIdx(node, comp, 2));
#endif
    }
    else
    { // transientFile
        int step1, step2;
        double weight1, weight2;
        int res = getTimeSteps(time, step1, step2, weight1, weight2);
        if (res < 1)
        {
            printf("Unstructured::getVector3: error: time step not found\n");
            return;
        }
        //int vecComp = getVectorNodeDataComponent();
        // not ok (actually only single component in file)
        //###############
        int vecComp = comp;

        vec[0] = weight1 * getTransientFileEntry(node, vecComp, 0, step1) + weight2 * getTransientFileEntry(node, vecComp, 0, step2);
        vec[1] = weight1 * getTransientFileEntry(node, vecComp, 1, step1) + weight2 * getTransientFileEntry(node, vecComp, 1, step2);
        vec[2] = weight1 * getTransientFileEntry(node, vecComp, 2, step1) + weight2 * getTransientFileEntry(node, vecComp, 2, step2);
    }

    if (divideVelocityByWalldist && wallDistExists)
    {
        double wd = getWallDist(node);
        if (wd == 0)
            wd = 1e-9;
        vec3scal(vec, 1. / wd, vec);
    }
}

inline void Unstructured::getVector3(int node, int comp, vec3 vec)
{
    if (transient)
        printf("Unstructured::getVector3 error: no transient data, returning reference time step\n");

    if (vector3CB && comp != 0)
    {
        printf("Unstructured: error: specified component!=0 but in vector3CB mode\n");
        return;
    }

    int vStride = nodeComponentDataPtrs[comp].stride;
    vec[0] = nodeComponentDataPtrs[comp].ptrs[0][vStride * node];
    vec[1] = nodeComponentDataPtrs[comp].ptrs[1][vStride * node];
    vec[2] = nodeComponentDataPtrs[comp].ptrs[2][vStride * node];

    if (divideVelocityByWalldist && wallDistExists)
    {
        double wd = getWallDist(node);
        if (wd == 0)
            wd = 1e-9;
        vec3scal(vec, 1. / wd, vec);
    }
}

inline void Unstructured::getVector3(int node, int comp, fvec3 vec)
{
    vec3 v;
    getVector3(node, comp, v);
    vec[0] = v[0];
    vec[1] = v[1];
    vec[2] = v[2];
}

inline void Unstructured::getVector3(int node, int comp, int opId, vec3 vec)
{
    if (transient)
        printf("Unstructured::getVector3 error: no transient data, returning reference time step\n");

    if (vector3CB && comp != 0)
    {
        printf("Unstructured: error: specified component!=0 but in vector3CB mode\n");
        return;
    }

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    float *ptrU = dd->p.f;
    int vStride = dd->veclen; // dd must be actually interleaved
    vec[0] = (ptrU + 0)[vStride * node];
    vec[1] = (ptrU + 1)[vStride * node];
    vec[2] = (ptrU + 2)[vStride * node];

    if (divideVelocityByWalldist && wallDistExists)
    {
        double wd = getWallDist(node);
        if (wd == 0)
            wd = 1e-9;
        vec3scal(vec, 1. / wd, vec);
    }
}

inline void Unstructured::setVector3(int node, vec3 vec)
{
    if (vector3CB)
    {
        printf("Unstructured: error: can not write to vector3 callback\n");
        return;
    }

    u[vStride * node] = vec[0];
    v[vStride * node] = vec[1];
    w[vStride * node] = vec[2];
}

inline void Unstructured::setVector3(int node, int comp, vec3 vec)
{
    if (vector3CB)
    {
        printf("Unstructured: error: can not write to vector3 callback\n");
        return;
    }

    int vStride = nodeComponentDataPtrs[comp].stride;
    nodeComponentDataPtrs[comp].ptrs[0][vStride * node] = vec[0];
    nodeComponentDataPtrs[comp].ptrs[1][vStride * node] = vec[1];
    nodeComponentDataPtrs[comp].ptrs[2][vStride * node] = vec[2];
}

inline void Unstructured::setVector3(int node, int comp, int opId, vec3 vec)
{
    if (vector3CB)
    {
        printf("Unstructured: error: can not write to vector3 callback\n");
        return;
    }

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    float *ptrU = dd->p.f;
    int vStride = dd->veclen; // dd must be actually interleaved
    (ptrU + 0)[vStride * node] = vec[0];
    (ptrU + 1)[vStride * node] = vec[1];
    (ptrU + 2)[vStride * node] = vec[2];
}

inline void Unstructured::getVectorN(int node, int comp, double *vec)
{
    if (transient)
        printf("Unstructured::getVectorN error: no transient data, returning reference time step\n");

    if (getNodeCompVecLen(comp) == 1)
    {
        *vec = getScalar(node, comp);
    }
    else if (getNodeCompVecLen(comp) == 3)
    {
        getVector3(node, comp, vec);
    }
    else if (getNodeCompVecLen(comp) == 9)
    {
        fmat3 fm;
        getMatrix3(node, comp, fm);
        fmat3tovecN(fm, vec);
    }
    else
    {
        printf("Unstructured::getVectorN ERROR: unsuported vector length\n");
        return;
    }
}

/* TODO: read n-vect: inline void Unstructured::getVector(int node, double *vec)
   inline void Unstructured::getVector(int node, double *vec)
   {
   for (int v=0; v<getNodeCompVeclen(vectorComponent); v++) {
   vec[v] = u[];
   }
   }*/

// NOTE (####): tensor data must be actually interleaved !!
// this is true for AVS
// and also true for Covise's DO_Unstructured_T3D_Data of F3D type (only supported type)

inline void Unstructured::getMatrix3(int node, fmat3 mat)
{
    if (transient)
        printf("Unstructured::getMatrix3 error: no transient data, returning reference time step\n");

    memcpy(mat, u + vStride * node, 3 * 3 * sizeof(float));
}

inline void Unstructured::getMatrix3(int node, int comp, fmat3 mat)
{
    if (transient)
        printf("Unstructured::getMatrix3 error: no transient data, returning reference time step\n");

    float *ptrU = nodeComponentDataPtrs[comp].ptrs[0];
    int vStride = nodeComponentDataPtrs[comp].stride;
    memcpy(mat, ptrU + vStride * node, 3 * 3 * sizeof(float));
}

inline void Unstructured::getMatrix3(int node, int comp, int opId, fmat3 mat)
{
    if (transient)
        printf("Unstructured::getMatrix3 error: no transient data, returning reference time step\n");

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    float *ptrU = dd->p.f;
    int vStride = dd->veclen;
    memcpy(mat, ptrU + vStride * node, 3 * 3 * sizeof(float));
}

inline void Unstructured::setMatrix3(int node, fmat3 mat)
{
    memcpy(u + vStride * node, mat, 3 * 3 * sizeof(float));
}

inline void Unstructured::setMatrix3(int node, int comp, fmat3 mat)
{
    float *ptrU = nodeComponentDataPtrs[comp].ptrs[0];
    int vStride = nodeComponentDataPtrs[comp].stride;
    memcpy(ptrU + vStride * node, mat, 3 * 3 * sizeof(float));
}

inline void Unstructured::setMatrix3(int node, int comp, int opId, fmat3 mat)
{
    DataDesc *dd = findNodeCompExtraData(comp, opId);
    float *ptrU = dd->p.f;
    int vStride = dd->veclen;
    memcpy(ptrU + vStride * node, mat, 3 * 3 * sizeof(float));
}

/* unused
inline void Unstructured::setMatrix3(int node, int i, int j, float f)
{
  u[vStride*node + i*3 + j] = f;
  }*/

inline double Unstructured::getWallDist(int node)
{
    return wallDist[sStride * node];
}

inline void Unstructured::rotateWithTransientZone(vec3 xyz, double time, vec3 out)
{
    vec3 rotVect;
    vec3copy(transientZoneRotationVector, rotVect);
    vec3scal(rotVect, time, rotVect);
    mat3 rotMat;
    rotVectTomat3(rotVect, rotMat);
    vec3sub(xyz, transientZoneRotationCenter, out);
    mat3vec(rotMat, out, out);
    vec3add(out, transientZoneRotationCenter, out);
}

inline int Unstructured::getCellZone(int cell, double time)
{
    int *cellNodes = getCellNodesAVS(cell);
    int zone;
    if (transient)
        //zone = (int) getScalar(cellNodes[0], zoneComponent, time);
        // ### TODO: actually taking scalars not from unsteady data
        zone = (int)getScalar(cellNodes[0], zoneComponent);
    else
        zone = (int)getScalar(cellNodes[0], zoneComponent);
    for (int n = 1; n < nVertices[getCellType(cell)]; n++)
    {
        int wz;
        if (transient)
            //wz = (int) getScalar(cellNodes[n], zoneComponent, time);
            // ### TODO: actually taking scalars not from unsteady data
            wz = (int)getScalar(cellNodes[n], zoneComponent);
        else
            wz = (int)getScalar(cellNodes[n], zoneComponent);
        if (zone != wz)
        {
            printf("Unstructured::getCellZone: inconsistent cell zones: at=%d before=%d now=%g\n", n, zone, getScalar(cellNodes[n], zoneComponent, time));
            printf("node zones:\n");
            for (int n = 0; n < nVertices[getCellType(cell)]; n++)
            {
                printf(" %d:%g\n", n, getScalar(cellNodes[n], zoneComponent, time));
            }
            return -1;
        }
    }
    return zone;
}

inline void Unstructured::getCellCentroid(int cell, vec3 centroid, double time)
{
    if (time == 0.0 || transientZoneRotating < 0)
    {
        fvec3tovec3(cellCentroid[cell], centroid);
        return;
    }

    int zone = getCellZone(cell, time);
    if (zone != transientZoneRotating)
    {
        // cell is not rotating
        fvec3tovec3(cellCentroid[cell], centroid);
        return;
    }
    else
    {
        // cell is rotating
        fvec3tovec3(cellCentroid[cell], centroid);
        rotateWithTransientZone(centroid, time, centroid);
    }
}

inline void Unstructured::getCellCentroid(int cell, fvec3 centroid, double time)
{
    vec3 cen;
    getCellCentroid(cell, cen, time);
    vec3tofvec3(cen, centroid);
}

inline int Unstructured::getNodeCompVecLen(int comp)
{
    // TODO: replace when introducing e.g. scalarCB, e.g. by fake components:
    //       first component stores vector3CB, second stores scalarCB
    if (vector3CB)
    {
        return 3;
    }

    return nodeComponents[comp];
}

#if 0 // ### delete or replace
inline float *Unstructured::getNodeDataCompPtr(int comp)
{
  return nodeComponentDataPtrs[comp];
}
#else
inline float *Unstructured::getNodeMatrixCompPtr(int comp)
{ // matrix data is actually interleaved ###
    return nodeComponentDataPtrs[comp].ptrs[0];
}
#endif

inline void Unstructured::selectVectorNodeData(int comp)
{
    if (vector3CB && comp != 0)
    {
        printf("Unstructured::selectVectorNodeData: error: selecting component!=0 with vector3CB\n");
        return;
    }

    u = nodeComponentDataPtrs[comp].ptrs[0];
    v = nodeComponentDataPtrs[comp].ptrs[1];
    w = nodeComponentDataPtrs[comp].ptrs[2];
    vStride = nodeComponentDataPtrs[comp].stride;
    vectorComponent = comp;
    vectorComponentExtraData = NULL;
}

inline void Unstructured::selectVectorNodeData(int comp, int opId)
{
    if (vector3CB && comp != 0)
    {
        printf("Unstructured::selectVectorNodeData: error: selecting component!=0 with vector3CB\n");
        return;
    }

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    if (!dd)
    {
        fprintf(stderr, "node component %d has no vector extra data for operation ID %d\n", comp, opId);
        exit(1); // ###
    }
    // dd data must be actually interleaved
    u = dd->p.f;
    v = u + 1;
    w = v + 1;
    vStride = dd->veclen;
    vectorComponent = comp;
    vectorComponentExtraData = dd;
}

inline void Unstructured::selectScalarNodeData(int comp)
{
    if (vector3CB)
    {
        printf("Unstructured::selectScalarNodeData: not allowed with vector3CB\n");
        return;
    }

    p = nodeComponentDataPtrs[comp].ptrs[0];
    sStride = nodeComponentDataPtrs[comp].stride;
    scalarComponent = comp;
    scalarComponentExtraData = NULL;
}

inline void Unstructured::selectScalarNodeData(int comp, int opId)
{
    if (vector3CB)
    {
        printf("Unstructured::selectScalarNodeData: not allowed with vector3CB\n");
        return;
    }

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    if (!dd)
    {
        fprintf(stderr, "node component %d has no scalar extra data for operation ID %d\n", comp, opId);
        exit(1); // ###
    }
    p = dd->p.f;
    sStride = dd->veclen; // dd data must be actually interleaved
    scalarComponent = comp;
    scalarComponentExtraData = dd;
}

inline void Unstructured::selectWallDistNodeData(int comp)
{
    if (vector3CB)
    {
        printf("Unstructured::selectWallDistNodeData: warning: selecting walldist data component with vector3CB\n");
    }

    wallDist = nodeComponentDataPtrs[comp].ptrs[0];
    sStride = nodeComponentDataPtrs[comp].stride;
    wallDistComponent = comp;
}

inline void Unstructured::selectWallDistNodeData(int comp, int opId)
{
    if (vector3CB)
    {
        printf("Unstructured::selectWallDistNodeData: warning: selecting walldist data component with vector3CB\n");
    }

    DataDesc *dd = findNodeCompExtraData(comp, opId);
    if (!dd)
    {
        fprintf(stderr, "node component %d has no scalar extra data for operation ID %d\n", comp, opId);
        exit(1); // ###
    }
    wallDist = dd->p.f;
    sStride = dd->veclen; // dd data must be actually interleaved
    wallDistComponent = comp;
}

inline int Unstructured::getVectorNodeDataComponent(void)
{
    // TODO: replace when introducing e.g. scalarCB, e.g. by fake components:
    //       first component stores vector3CB, second stores scalarCB
    if (vector3CB)
    {
        return 0;
    }

    return vectorComponent;
}

inline int Unstructured::getScalarNodeDataComponent(void)
{
    return scalarComponent;
}

inline void Unstructured::assureCellData(double time)
{
    if ((!currCell.dataLoaded) || (scalarComponent != currCell.currCell_scalarComponent) || (vectorComponent != currCell.currCell_vectorComponent) || (wallDistComponent != currCell.currCell_wallDistComponent) || (scalarComponentExtraData != currCell.currCell_scalarExtraData) || (vectorComponentExtraData != currCell.currCell_vectorExtraData) || (transient && (time != currCell.currCell_time)))
    {
        loadCellData(time);
    }
}

inline int Unstructured::getCellIndex()
{
    return currCell.index;
}

inline double Unstructured::getCellRadiusSqr()
{
    return (double)cellRadiusSqr[getCellIndex()];
}

inline double Unstructured::getCellRadiusSqr(int cell)
{
    return (double)cellRadiusSqr[cell];
}

inline double Unstructured::getCellRadius()
{
    return (double)sqrt(cellRadiusSqr[getCellIndex()]);
}

inline double Unstructured::getCellRadius(int cell)
{
    return (double)sqrt(cellRadiusSqr[cell]);
}

inline void Unstructured::getCellEdgeNodesConnOrderAVS(int cell, int locEdgeIdx, int &v1, int &v2)
{
    int *cellNodes = getCellNodesAVS(cell);
    v1 = cellNodes[edgesConnectedOrder[getCellType(cell)][locEdgeIdx][0]];
    v2 = cellNodes[edgesConnectedOrder[getCellType(cell)][locEdgeIdx][1]];
}

inline void Unstructured::getCellEdgeLocNodesConnOrderAVS(int cell, int locEdgeIdx, int &v1, int &v2)
{
    v1 = edgesConnectedOrder[getCellType(cell)][locEdgeIdx][0];
    v2 = edgesConnectedOrder[getCellType(cell)][locEdgeIdx][1];
}

inline double Unstructured::interpolateScalar()
{
#if CHECK_VARIABLES
    if (!scalarExists)
    {
        fprintf(stderr, "Unstructured::interpolateScalar: no scalar\n");
        return 0.0;
    }
#endif

    // no call to loadCell or findCell, -> assure data
    assureCellData();

    CellInfo &c = currCell;
    double s = 0;
    for (int k = 0; k < nVertices[c.type]; k++)
    {
        s += c.scal[k] * weight[k];
    }
    return s;
}

// --- private ----------------------------------------------------------------

inline float Unstructured::getTransientFileEntry(int node, int ucdComponent,
                                                 int vectorComponent,
                                                 int timeStep)
{ // TODO: ### actually only supporting single UCD-component

    // TODO: check if in transientFile mode

    int timeStepNb = transientFilesTimeSteps[transientFileIdx].size();

    // ######### hard coded 3-vector !!!!
    //return transientFile[(node * timeStepNb + timeStep) * 3 + vectorComponent];
    return transientFile[(node * timeStepNb + timeStep) * nodeComponentVecLenTot + nodeComponentVecLenCumulated[ucdComponent] + vectorComponent];
}

#endif // _UNSTRUCTURED_H_
