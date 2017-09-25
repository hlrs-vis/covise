/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Unification Library for Modular Visualization Systems
//
// Unstructured Field
//
// CGL ETH Zuerich
// Ronald Peikert and Filip Sadlo 2006 - 2008

// TODO: make Windows-portable
#ifndef WIN32
#include <libgen.h> // for dirname()
#include <sys/mman.h>
#include <unistd.h>
#else
#include <io.h>
#define munmap(transientFile, transientFileSize) printf("fixme")
#define mmap(a, fileSize, PROT_READ, MAP_SHARED, fd, b) printf("fixme")
#define madvise(transientFile, transientFileSize, MADV_SEQUENTIAL) printf("fixme")
#define open _open
#define close _close
float dummy;
#define MAP_FAILED &dummy
#define dirname(path) printf("fixme")
#endif

#include "unstructured.h"

#include <climits>
#include <vector>
#include <map>
#include <algorithm>
#include <float.h>

// for mmap for transientFile
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#define CHECK_VARIABLES 1
#define DEBUG_OUTPUT 0

#ifdef COVISE
using namespace covise;
#endif

// ### sadlof 2006-06-14
//#define DEBUGGING

//int testNode = 15725; // uniform // HACK RP
int testNode = 5583; // adaptive // HACK RP

int nVertices[8] = { 1, 2, 3, 4, 4, 5, 6, 8 };
int nFaces[8] = { 0, 0, 1, 1, 4, 5, 5, 6 };
int nQuads[8] = { 0, 0, 0, 1, 0, 1, 3, 6 };
int nEdges[8] = { 0, 1, 3, 4, 6, 8, 9, 12 };

int faces[8][6][4] = //  from "cell_topo" in ucd_topo.h, sorting: quads followed by tris
    {
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      { { 0, 1, 2, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      { { 0, 1, 2, 3 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      { { 1, 2, 3, -1 }, { 0, 2, 1, -1 }, { 0, 3, 2, -1 }, { 0, 1, 3, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      { { 1, 2, 3, 4 }, { 0, 2, 1, -1 }, { 0, 3, 2, -1 }, { 0, 4, 3, -1 }, { 0, 1, 4, -1 }, { -1, -1, -1, -1 } },
      { { 1, 4, 5, 2 }, { 1, 0, 3, 4 }, { 0, 2, 5, 3 }, { 5, 4, 3, -1 }, { 0, 1, 2, -1 }, { -1, -1, -1, -1 } },
      { { 0, 1, 2, 3 }, { 1, 5, 6, 2 }, { 3, 2, 6, 7 }, { 0, 3, 7, 4 }, { 0, 4, 5, 1 }, { 4, 7, 6, 5 } }
    };

int faceNodeCnt[8][6] = {
    { 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0 },
    { 3, 0, 0, 0, 0, 0 },
    { 4, 0, 0, 0, 0, 0 },
    { 3, 3, 3, 3, 0, 0 },
    { 4, 3, 3, 3, 3, 0 },
    { 4, 4, 4, 3, 3, 0 },
    { 4, 4, 4, 4, 4, 4 }
};

// NOTE: some of these (prism, hex) are not in AVS order
int edges[8][12][2] = {
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // point
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // line
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // tri
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // quad
    { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 2, 3 }, { 3, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // tet
    { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // pyr
    { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 4 }, { 2, 5 }, { 3, 4 }, { 3, 5 }, { 4, 5 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // prism
    { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 0, 3 }, { 1, 2 }, { 4, 7 }, { 5, 6 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } } // hex
};

// connected order means that the following invariant holds:
// the visited nodes always are inside one connected component when
// the edges are visited in order
// additionally, node zero is contained (visited) in edge zero
int edgesConnectedOrder[8][12][2] = {
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // point
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // line
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // tri
    { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // quad
    { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 2, 3 }, { 3, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // tet
    { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // pyr
    { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 4 }, { 2, 5 }, { 3, 4 }, { 3, 5 }, { 4, 5 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, // prism
    { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 } } // hex
};

// Tet decomposition by splitting quads into 4 tris
int nTets[8] = { 0, 0, 0, 0, 1, 4, 14, 24 };
int tets[8][24][4];

int ct0 = 0, ct1 = 0, ct2 = 0;

inline float sqr(float x) { return x * x; }

class Keys3
{
public:
    Keys3(int k1, int k2, int k3)
        : key1(k1)
        , key2(k2)
        , key3(k3)
    {
    }

    bool operator<(const Keys3 &right) const
    {
        return (key1 < right.key1 || (key1 == right.key1 && (key2 < right.key2 || (key2 == right.key2 && (key3 < right.key3)))));
    }

    int key1, key2, key3;

    static void sortInts3(int &i1, int &i2, int &i3)
    {
        int w[3] = { i1, i2, i3 };
        sort(w, w + 3);
        i1 = w[0];
        i2 = w[1];
        i3 = w[2];
    }
};

class Keys4
{
public:
    Keys4(int k1, int k2, int k3, int k4)
        : key1(k1)
        , key2(k2)
        , key3(k3)
        , key4(k4)
    {
    }

    bool operator<(const Keys4 &right) const
    {
        return (key1 < right.key1 || (key1 == right.key1 && (key2 < right.key2 || (key2 == right.key2 && (key3 < right.key3 || (key3 == right.key3 && (key4 < right.key4)))))));
    }

    int key1, key2, key3, key4;

    static void sortInts4(int &i1, int &i2, int &i3, int &i4)
    {
        int w[4] = { i1, i2, i3, i4 };
        sort(w, w + 4);
        i1 = w[0];
        i2 = w[1];
        i3 = w[2];
        i4 = w[3];
    }
};

void Unstructured::nodeOrderAVStoCovise(int cellType,
                                        int *nodesIn,
                                        int *nodesOut)
{
    switch (cellType)
    {
    case CELL_TET:
    {
        nodesOut[0] = nodesIn[1];
        nodesOut[1] = nodesIn[2];
        nodesOut[2] = nodesIn[3];
        nodesOut[3] = nodesIn[0];
    }
    break;
    case CELL_PYR:
    {
        nodesOut[0] = nodesIn[1];
        nodesOut[1] = nodesIn[2];
        nodesOut[2] = nodesIn[3];
        nodesOut[3] = nodesIn[4];
        nodesOut[4] = nodesIn[0];
    }
    break;
    case CELL_PRISM:
    {
        nodesOut[0] = nodesIn[3];
        nodesOut[1] = nodesIn[4];
        nodesOut[2] = nodesIn[5];
        nodesOut[3] = nodesIn[0];
        nodesOut[4] = nodesIn[1];
        nodesOut[5] = nodesIn[2];
    }
    break;
    case CELL_HEX:
    {
        nodesOut[0] = nodesIn[4];
        nodesOut[1] = nodesIn[5];
        nodesOut[2] = nodesIn[6];
        nodesOut[3] = nodesIn[7];
        nodesOut[4] = nodesIn[0];
        nodesOut[5] = nodesIn[1];
        nodesOut[6] = nodesIn[2];
        nodesOut[7] = nodesIn[3];
    }
    break;
    default:
    {
        printf("nodeOrderAVStoCovise: ERROR: unsupported cell type: %d\n", cellType);
    }
    }
}

void Unstructured::nodeOrderAVStoVTK(int cellType,
                                     int *nodesIn,
                                     int *nodesOut)
{
    switch (cellType)
    {
    case CELL_TET:
    {
        nodesOut[0] = nodesIn[1];
        nodesOut[1] = nodesIn[2];
        nodesOut[2] = nodesIn[3];
        nodesOut[3] = nodesIn[0];
    }
    break;
    case CELL_PYR:
    {
        nodesOut[0] = nodesIn[1];
        nodesOut[1] = nodesIn[2];
        nodesOut[2] = nodesIn[3];
        nodesOut[3] = nodesIn[4];
        nodesOut[4] = nodesIn[0];
    }
    break;
    case CELL_PRISM:
    {
        nodesOut[0] = nodesIn[0];
        nodesOut[1] = nodesIn[1];
        nodesOut[2] = nodesIn[2];
        nodesOut[3] = nodesIn[3];
        nodesOut[4] = nodesIn[4];
        nodesOut[5] = nodesIn[5];
    }
    break;
    case CELL_HEX:
    {
        nodesOut[0] = nodesIn[4];
        nodesOut[1] = nodesIn[5];
        nodesOut[2] = nodesIn[6];
        nodesOut[3] = nodesIn[7];
        nodesOut[4] = nodesIn[0];
        nodesOut[5] = nodesIn[1];
        nodesOut[6] = nodesIn[2];
        nodesOut[7] = nodesIn[3];
    }
    break;
    default:
    {
        printf("nodeOrderAVStoVTK: ERROR: unsupported cell type: %d\n", cellType);
    }
    }
}

void computeTetDecomposition()
{
    tets[Unstructured::CELL_TET][0][0] = 0;
    tets[Unstructured::CELL_TET][0][1] = 1;
    tets[Unstructured::CELL_TET][0][2] = 2;
    tets[Unstructured::CELL_TET][0][3] = 3;

    for (int type = Unstructured::CELL_PYR; type <= Unstructured::CELL_HEX; type++)
    {
        int ct = 0;
        int centroid = nVertices[type] + nQuads[type];
        if (type == Unstructured::CELL_PYR)
            centroid = 0;

        if (type == Unstructured::CELL_PRISM)
        {
            tets[type][0][0] = centroid;
            tets[type][0][1] = 5;
            tets[type][0][2] = 4;
            tets[type][0][3] = 3;

            tets[type][1][0] = centroid;
            tets[type][1][1] = 0;
            tets[type][1][2] = 1;
            tets[type][1][3] = 2;
            ct = 2;
        }

        for (int i = 0; i < nFaces[type]; i++)
        {
            int *nr = faces[type][i];
            if (nr[3] != -1)
            { // quad face
                int faceCentroid = nVertices[type] + i;
                for (int j = 0; j < 4; j++)
                {
                    tets[type][ct][0] = centroid;
                    tets[type][ct][1] = faceCentroid;
                    tets[type][ct][2] = nr[(j + 1) % 4];
                    tets[type][ct][3] = nr[j];
                    ct++;
                }
            }
        }
    }

#ifdef DEBUG
    int type = Unstructured::CELL_HEX;
    for (int i = 0; i < 24; i++)
    {
        printf("%3d %3d %3d %3d\n",
               tets[type][i][0],
               tets[type][i][1],
               tets[type][i][2],
               tets[type][i][3]);
    }
    exit(0);
#endif
}

bool insideTet(vec3 xyz, vec3 t0, vec3 t1, vec3 t2, vec3 t3, vec4 bary)
{
    // Make all relative to xyz
    vec3 r0, r1, r2, r3;

    vec3sub(t0, xyz, r0);
    vec3sub(t1, xyz, r1);
    vec3sub(t2, xyz, r2);
    vec3sub(t3, xyz, r3);

    double det0 = vec3det(r1, r2, r3);
    double det1 = vec3det(r0, r3, r2);
    double det2 = vec3det(r0, r1, r3);
    double det3 = vec3det(r0, r2, r1);

    double sum = det0 + det1 + det2 + det3;

    //printf("bary = %9.6f %9.6f %9.6f %9.6f\n", det1/sum, det2/sum, det3/sum, det0/sum);

    bary[0] = det0 / sum;
    if (bary[0] < -1e-5)
        return false;
    bary[1] = det1 / sum;
    if (bary[1] < -1e-5)
        return false;
    bary[2] = det2 / sum;
    if (bary[2] < -1e-5)
        return false;
    bary[3] = det3 / sum;
    if (bary[3] < -1e-5)
        return false;

    return true;
}

// ---------------------------------

#ifdef AVS
// AVS constructor
// TODO #### check if planar cell types are supported by whole library
//           or if they need to be excluded (skipped)
Unstructured::Unstructured(UCD_structure *ucd)
{
#if DEBUG_OUTPUT
    printf("this is Unstructured %s\n", UNST_VERSION);
#endif

    isCopy = false;
    nameAllocated = false;
    nodeListAllocated = false;
    nodeListOffsetsAllocated = false;
    cellTypesAllocated = false;
    coordinatesAllocated = false;
    allocatedNodeDataComponents.clear();
    strictNodeCompExtraDeletion = false;
    transientDataDict = NULL;
    transientFile = NULL;
    transient = false;
    vector3CB = NULL;

    name = ucd->name;

    // rotating zones (experimental)
    if (strncmp(name, "rotating", strlen("rotating")) == 0)
    {
        // there is a rotating zone
        double angularSpeed;
        sscanf(name, "%*s%d%lg,%lg,%lg%lg,%lg,%lg%lg",
               &transientZoneRotating,
               &transientZoneRotationCenter[0],
               &transientZoneRotationCenter[1],
               &transientZoneRotationCenter[2],
               &transientZoneRotationVector[0],
               &transientZoneRotationVector[1],
               &transientZoneRotationVector[2],
               &angularSpeed);
        vec3nrm(transientZoneRotationVector, transientZoneRotationVector);
        vec3scal(transientZoneRotationVector, angularSpeed, transientZoneRotationVector);
        printf("rotating zone:\n id=%d\n center=%g,%g,%g\n axis=%g,%g,%g\n",
               transientZoneRotating,
               transientZoneRotationCenter[0],
               transientZoneRotationCenter[1],
               transientZoneRotationCenter[2],
               transientZoneRotationVector[0],
               transientZoneRotationVector[1],
               transientZoneRotationVector[2]);
        zoneComponent = 1; // ############## HACK !!!!!!!!! TODO TODO TODO
    }
    else
    {
        transientZoneRotating = -1;
    }

    nCells = ucd->ncells;
    nNodes = ucd->nnodes;

    edgeNb = -1;
    faceNb = -1;

    // Coordinates
    x = ucd->x;
    y = ucd->y;
    z = ucd->z;
    cStride = 1; // non-interleaved coordinate data

    // Components
    //nodeComponents = ucd->node_components;
    nodeComponents = new int[ucd_nodeCompNb(ucd)];
    memcpy(nodeComponents, ucd->node_components, ucd_nodeCompNb(ucd) * sizeof(int));
    nodeComponentNb = ucd_nodeCompNb(ucd);
    nodeComponentLabels = new char *[nodeComponentNb];
    // ### TODO: reference instead of copy
    for (int c = 0; c < nodeComponentNb; c++)
    {
        nodeComponentLabels[c] = new char[256]; // ###
        // ### TODO: test for non-existant (NULL) labels and generate fake labels
        if (ucd_nodeCompLabel(ucd, c, nodeComponentLabels[c]))
        {
            // generate fake label
            sprintf(nodeComponentLabels[c], "%d", c);
        }
    }

    // sample pointers to node components data
    {
        float *dptr = ucd->node_data;
        int n = 0;
        for (int comp = 0; n < ucd->node_veclen; comp++)
        {

            int comp_veclen = ucd->node_components[comp];

            NodeCompDataPtr dp;
            for (int vv = 0; vv < comp_veclen; vv++)
            {
                dp.ptrs.push_back(dptr + vv); // avs data is interleaved
            }
            dp.stride = comp_veclen; // avs data is interleaved
            nodeComponentDataPtrs.push_back(dp);

            n += comp_veclen;
            dptr += comp_veclen * nNodes;
        }
    }

    // Vector quantity   (first component with veclen > 1)
    vectorComponent = ucd_findNodeCompByVeclen(ucd, 1, 0, 1);
    vectorExists = vectorComponent >= 0;
    if (vectorExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: vector quantity initially at ucd component %d\n",
               name, vectorComponent);
#endif
        selectVectorNodeData(vectorComponent);
    }

    // Scalar
    scalarComponent = ucd_findNodeCompByVeclen(ucd, 1, 0);
    scalarExists = scalarComponent >= 0;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: scalar quantity initially at ucd component %d\n",
               name, scalarComponent);
#endif
        selectScalarNodeData(scalarComponent);
    }

    // Wall Distance
    wallDistExists = false;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: warning: wall distance initially at scalar quantity ucd component\n", name);
#endif
        wallDist = p; // TODO: general case
        wallDistExists = true;
        wallDistComponent = scalarComponent;
    }

    // Cells
    cellType = ucd->cell_type;
    nodeList = ucd->node_list;
    nodeListOffset = ucd->node_list_ptr;

    divideVelocityByWalldist = false;
    extrapolateToNoSlipBoundary = false;

    searchGrid = 0;
    cellRadiusSqr = 0;
    cellCentroid = NULL;
    nodeNeighbors = NULL;
    cellNeighbors = NULL;
    //vGradient = NULL;

    computeBoundingBox();
    computeCentroids();
    computeTetDecomposition();

    for (int i = 0; i < nodeComponentNb; i++)
    {
        std::vector<DataDesc *> w;
        nodeCompExtraData.push_back(w);
    }

    nodeComponentVecLenTot = 0;
    nodeComponentVecLenCumulated.clear();
    for (int i = 0; i < nodeComponentNb; i++)
    {
        nodeComponentVecLenCumulated.push_back(nodeComponentVecLenTot);
        nodeComponentVecLenTot += getNodeCompVecLen(i);
    }

    if (nCells > 0)
        loadCell(0, true);
    // ################## HACK (WORKAROUND FOR NOT FOUND FIRST CALL TO findCell()
    {
        vec3 cen;
        getCellCentroid(0, cen);
        findCell(cen);
    }
}
#endif

#ifdef COVISE
// Covise constructors
// TODO #### check if planar cell types are supported by whole library
//           or if they need to be excluded (skipped)
#ifdef COVISE5
Unstructured::Unstructured(DO_UnstructuredGrid *grid,
                           std::vector<DO_Unstructured_S3D_Data *> *scal,
                           std::vector<DO_Unstructured_V3D_Data *> *vect)
#else
Unstructured::Unstructured(coDoUnstructuredGrid *grid,
                           std::vector<coDoFloat *> *scal,
                           std::vector<coDoVec3 *> *vect)
#endif
{
// Covise and AVS have zero-based vertex IDs
// and element IDs (cell's offset into nodelist) hence still
// no adaptation (kind of: bool zeroBased) of code necessary

// TODO:
// - support tensors DO_Unstructured_T3D_Data of type F3D:
//    tensors must be (and are) interleaved in current version of Unstructured
// - (support sets of DO_Unstructured_S3D_Data etc.)

#if DEBUG_OUTPUT
    printf("this is Unstructured %s\n", UNST_VERSION);
#endif

    if (scal)
    {
        bool bad = false;
        if (scal->size() < 1)
            bad = true;
        for (int i = 0; i < scal->size(); i++)
        {
            if (!(*scal)[i])
                bad = true;
        }
        if (bad)
        {
            printf("Unstructured: got NULL pointer(s) in scalar data, aborting\n");
            exit(1); // ###
        }
    }

    if (vect)
    {
        bool bad = false;
        if (vect->size() < 1)
            bad = true;
        for (int i = 0; i < vect->size(); i++)
        {
            if (!(*vect)[i])
                bad = true;
        }
        if (bad)
        {
            printf("Unstructured: got NULL pointer(s) in vector data, aborting\n");
            exit(1); // ###
        }
    }

    isCopy = false;
    nameAllocated = false;
    nodeListAllocated = true;
    nodeListOffsetsAllocated = false;
    cellTypesAllocated = false;
    coordinatesAllocated = false;
    allocatedNodeDataComponents.clear();
    strictNodeCompExtraDeletion = false;
    transientDataDict = NULL;
    transientFile = NULL;
    transient = false;
    vector3CB = NULL;

#ifdef COVISE5
    name = grid->get_name(); // ### ok? (accesses DistributedObject base class)
#else
    name = grid->getName(); // ### ok? (accesses DistributedObject base class)
#endif

    //printf("name of unstructured grid: %s\n", name);

    // rotating zones (experimental)
    if (strncmp(name, "rotating", strlen("rotating")) == 0)
    {
        // there is a rotating zone
        double angularSpeed;
        sscanf(name, "%*s%d%lg,%lg,%lg%lg,%lg,%lg%lg",
               &transientZoneRotating,
               &transientZoneRotationCenter[0],
               &transientZoneRotationCenter[1],
               &transientZoneRotationCenter[2],
               &transientZoneRotationVector[0],
               &transientZoneRotationVector[1],
               &transientZoneRotationVector[2],
               &angularSpeed);
        vec3nrm(transientZoneRotationVector, transientZoneRotationVector);
        vec3scal(transientZoneRotationVector, angularSpeed, transientZoneRotationVector);
        printf("rotating zone:\n id=%d\n center=%g,%g,%g\n axis=%g,%g,%g\n",
               transientZoneRotating,
               transientZoneRotationCenter[0],
               transientZoneRotationCenter[1],
               transientZoneRotationCenter[2],
               transientZoneRotationVector[0],
               transientZoneRotationVector[1],
               transientZoneRotationVector[2]);
        zoneComponent = 1; // ############## HACK !!!!!!!!! TODO TODO TODO
    }
    else
    {
        transientZoneRotating = -1;
    }

    int nConn;
    //grid->get_grid_size(&nCells, &nConn, &nNodes);
    grid->getGridSize(&nCells, &nConn, &nNodes);

    edgeNb = -1;
    faceNb = -1;

//printf("### nConn=%d\n", nConn);

// Cells and Coordinates
#ifdef COVISE5
    grid->get_adresses(&nodeListOffset, &nodeList, &x, &y, &z);
#else
    // #### there seems to be a bug inside Covise6, switching temporally back
    grid->getAddresses(&nodeListOffset, &nodeList, &x, &y, &z);
//grid->get_addresses(&nodeListOffset, &nodeList, &x, &y, &z);
#endif
    cStride = 1; // non-interleaved coordinate data

// type definitions for tet, pyr, prism, and hex are ACTUALLY (### HACK TODO)
// identical between AVS and Covise
#ifdef COVISE5
    if (!grid->has_type_list())
    {
        fprintf(stderr, "error: no type list!\n");
        exit(1); // ###
    }
    else
#endif
    {
#ifdef COVISE5
        grid->get_type_list(&cellType);
#else
        grid->getTypeList(&cellType);
#endif
    }

    // convert node order from Covise to AVS
    // the node list buffer is allocated ###,
    // this violates the philosophy to keep memory usage of Unstructured very small
    // TODO: replace all direct accesses to nodelist by accessors and change
    //       the node ordering inside the accessors instead of memory allocation
    {
        int *nodeListCov = nodeList;

        nodeList = new int[nConn];
        nodeListAllocated = true;

        int nodeIdx = 0;
        for (int c = 0; c < nCells; c++)
        {

            switch (cellType[c])
            {
            case CELL_TET:
            {
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 3];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 0];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 1];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 2];
            }
            break;
            case CELL_PYR:
            {
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 4];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 0];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 1];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 2];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 3];
            }
            break;
            case CELL_PRISM:
            {
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 3];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 4];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 5];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 0];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 1];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 2];
            }
            break;
            case CELL_HEX:
            {
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 4];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 5];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 6];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 7];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 0];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 1];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 2];
                nodeList[nodeIdx++] = nodeListCov[nodeListOffset[c] + 3];
            }
            break;
            default:
            {
                printf("Unstructured-Covise: ERROR: unsupported cell type: %d\n", cellType[c]);
                // planar types have opposite orientation, but this should not matter
                // TODO: copy nodes
            }
            }
        }
    }

    // fake components: scalars come first
    // TODO: support sets of DO_Unstructured_S3D_Data etc. ? (may interfere with Covise scheme to use sets for time-dependent data only
    {
        int numNodeComp = 0;
        if (scal)
            numNodeComp += int(scal->size());
        if (vect)
            numNodeComp += int(vect->size());

        nodeComponentNb = numNodeComp;
        nodeComponents = new int[nodeComponentNb];

        nodeComponentLabels = new char *[nodeComponentNb];
        // ### TODO: reference instead of copy
        for (int c = 0; c < nodeComponentNb; c++)
        {
            nodeComponentLabels[c] = new char[256]; // ###
            strcpy(nodeComponentLabels[c], "TODO");
        }

        // setup
        numNodeComp = 0;
        vectorComponent = -1;
        scalarComponent = -1;
        if (scal)
        {
            for (int co = 0; co < scal->size(); co++)
            {
                nodeComponents[numNodeComp] = 1;

                float *wp;
#ifdef COVISE5
                (*scal)[co]->get_adress(&wp);
#else
                // #### there seems to be a bug inside Covise6, switching temporally back
                (*scal)[co]->getAddress(&wp);
//(*scal)[co]->get_address(&wp);
#endif

                NodeCompDataPtr dp;
                dp.ptrs.push_back(wp);
                dp.stride = 1;

                nodeComponentDataPtrs.push_back(dp);

                //delete:(*scal)[co]->get_adress(&nodeComponentDataPtrs[numNodeComp]);
                if (scalarComponent < 0)
                    scalarComponent = numNodeComp;
                numNodeComp++;
            }
        }
        if (vect)
        {
            const int vveclen = 3;
            for (int co = 0; co < vect->size(); co++)
            {
                nodeComponents[numNodeComp] = vveclen;

                NodeCompDataPtr dp;

                float *up, *vp, *wp;
#ifdef COVISE5
                (*vect)[co]->get_adresses(&up, &vp, &wp);
#else
                // #### there seems to be a bug inside Covise6, switching temporally back
                (*vect)[co]->getAddresses(&up, &vp, &wp);
//(*vect)[co]->get_addresses(&up, &vp, &wp);
#endif
                dp.ptrs.push_back(up);
                dp.ptrs.push_back(vp);
                dp.ptrs.push_back(wp);
                dp.stride = 1; // Covise data is not interleaved

                nodeComponentDataPtrs.push_back(dp);

                //delete: (*vect)[co]->get_adresses(&nodeComponentDataPtrs[numNodeComp]);
                if (vectorComponent < 0)
                    vectorComponent = numNodeComp;
                numNodeComp++;
            }
        }
    }

    // Vector quantity   (first component with veclen > 1)
    vectorExists = vectorComponent >= 0;
    if (vectorExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: vector quantity initially at ucd component %d\n",
               name, vectorComponent);
#endif
        selectVectorNodeData(vectorComponent);
    }

    // Scalar
    scalarExists = scalarComponent >= 0;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: scalar quantity initially at ucd component %d\n",
               name, scalarComponent);
#endif
        selectScalarNodeData(scalarComponent);
    }

    // Wall Distance
    wallDistExists = false;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: warning: wall distance initially at scalar quantity ucd component\n", name);
#endif
        wallDist = p; // TODO: general case
        wallDistExists = true;
        wallDistComponent = scalarComponent;
    }

    // setup
    divideVelocityByWalldist = false;
    extrapolateToNoSlipBoundary = false;

    // setup grid
    searchGrid = 0;
    cellRadiusSqr = 0;
    cellCentroid = NULL;
    nodeNeighbors = NULL;
    cellNeighbors = NULL;
    //vGradient = NULL;

    computeBoundingBox();
    computeCentroids();
    computeTetDecomposition();

    // initialize extra data arrays
    for (int i = 0; i < nodeComponentNb; i++)
    {
        std::vector<DataDesc *> w;
        nodeCompExtraData.push_back(w);
    }

    nodeComponentVecLenTot = 0;
    nodeComponentVecLenCumulated.clear();
    for (int i = 0; i < nodeComponentNb; i++)
    {
        nodeComponentVecLenCumulated.push_back(nodeComponentVecLenTot);
        nodeComponentVecLenTot += getNodeCompVecLen(i);
    }

    if (nCells > 0)
        loadCell(0, true);
}
#endif

#ifdef VTK
// VTK (Paraview) constructors
#define VTK_EXCLUDE_PLANAR_CELLS // exclude 1D and 2D cell types
Unstructured::Unstructured(vtkUnstructuredGrid *grid,
                           std::vector<vtkFloatArray *> *scal,
                           std::vector<vtkFloatArray *> *vect)
{
// Covise, AVS and VTK have zero-based vertex IDs
// and element IDs (cell's offset into nodelist) hence still
// no adaptation (kind of: bool zeroBased) of code necessary

#if DEBUG_OUTPUT
    printf("this is Unstructured %s\n", UNST_VERSION);
#endif

    nNodes = 0;

    edgeNb = -1;
    faceNb = -1;

    if (scal)
    {
        bool bad = false;
        if (scal->size() < 1)
            bad = true;
        for (int i = 0; i < scal->size(); i++)
        {
            if (!(*scal)[i])
                bad = true;
        }
        if (bad)
        {
            printf("Unstructured: got NULL pointer(s) in scalar data, aborting\n");
            exit(1); // ###
        }
        else
        {
            nNodes = (*scal)[0]->GetNumberOfTuples();
        }
    }

    if (vect)
    {
        bool bad = false;
        if (vect->size() < 1)
            bad = true;
        for (int i = 0; i < vect->size(); i++)
        {
            if (!(*vect)[i])
                bad = true;
        }
        if (bad)
        {
            printf("Unstructured: got NULL pointer(s) in vector data, aborting\n");
            exit(1); // ###
        }
        else
        {
            nNodes = (*vect)[0]->GetNumberOfTuples();
        }
    }

    isCopy = false;
    nameAllocated = false;
    nodeListAllocated = true;
    nodeListOffsetsAllocated = true; // ####
    cellTypesAllocated = true;
    coordinatesAllocated = false;
    allocatedNodeDataComponents.clear();
    strictNodeCompExtraDeletion = false;
    transientDataDict = NULL;
    transientFile = NULL;
    transient = false;
    vector3CB = NULL;

    name = (char *)grid->GetClassName(); // ### ok?

    //printf("name of unstructured grid: %s\n", name);

    // rotating zones (experimental)
    if (strncmp(name, "rotating", strlen("rotating")) == 0)
    {
        // there is a rotating zone
        double angularSpeed;
        sscanf(name, "%*s%d%lg,%lg,%lg%lg,%lg,%lg%lg",
               &transientZoneRotating,
               &transientZoneRotationCenter[0],
               &transientZoneRotationCenter[1],
               &transientZoneRotationCenter[2],
               &transientZoneRotationVector[0],
               &transientZoneRotationVector[1],
               &transientZoneRotationVector[2],
               &angularSpeed);
        vec3nrm(transientZoneRotationVector, transientZoneRotationVector);
        vec3scal(transientZoneRotationVector, angularSpeed, transientZoneRotationVector);
        printf("rotating zone:\n id=%d\n center=%g,%g,%g\n axis=%g,%g,%g\n",
               transientZoneRotating,
               transientZoneRotationCenter[0],
               transientZoneRotationCenter[1],
               transientZoneRotationCenter[2],
               transientZoneRotationVector[0],
               transientZoneRotationVector[1],
               transientZoneRotationVector[2]);
        zoneComponent = 1; // ############## HACK !!!!!!!!! TODO TODO TODO
    }
    else
    {
        transientZoneRotating = -1;
    }

    nCells = grid->GetNumberOfCells();
    // VTK nodelist includes for each cell the number of points, therefore subtract
    //delete int nConn = grid->GetCells()->GetNumberOfConnectivityEntries() - nCells;
    int nConn = grid->GetCells()->GetNumberOfConnectivityEntries();
//printf("#### nConn = %d\n", nConn);

// Cells and Coordinates
#if 0 // does not work with some data (CSCS CFX reader) 2007-08-24
  x = (float *) ((vtkDataArray *) grid->GetPoints()->GetData())->GetVoidPointer(0); // #### float or double? Gives pointer to single entry or to whole interleaved data? TODO
  y = x + 1;
  z = x + 2;
  cStride = 3; // interleaved coordinate data
#elif 0 // similar problem to previous
    x = (float *)grid->GetPoints()->GetVoidPointer(0); // #### float or double? Gives pointer to single entry or to whole interleaved data? TODO
    y = x + 1;
    z = x + 2;
    cStride = 3; // interleaved coordinate data
#else // ############# wastes memory: TODO: do it inside accessor
    coordinatesAllocated = true;
    x = new float[nNodes];
    y = new float[nNodes];
    z = new float[nNodes];
    for (int n = 0; n < nNodes; n++)
    {
        double pnt[3];
        grid->GetPoint(n, pnt);
        x[n] = pnt[0];
        y[n] = pnt[1];
        z[n] = pnt[2];
    }
    cStride = 1; // non-interleaved coordinate data
#endif

    // cell types
    unsigned char *vtkCellTypes = grid->GetCellTypesArray()->GetPointer(0); // ##### Gives pointer to single entry or to whole interleaved data?

#ifdef VTK_EXCLUDE_PLANAR_CELLS
    int newNCells = 0;
    int newNConn = 0;
    int *newCellType = NULL;
#endif

    if (!vtkCellTypes)
    {
        fprintf(stderr, "error: no type list!\n");
        exit(1); // ###
    }
    else
    {

#ifdef VTK_EXCLUDE_PLANAR_CELLS
        //  get number of cells, excluding planar cells ###
        int skippedCells = 0;
        for (int c = 0; c < nCells; c++)
        {
            if (vtkCellTypes[c] >= 10)
            { // HACK
                newNCells++;
            }
            else
            {
                skippedCells++;
            }
        }
        //if (skippedCells) printf("Unstructured-VTK: skipped %d non-3D cells\n", skippedCells);

        newCellType = new int[newNCells];

        int newC = 0;
#endif

        // ### actually duplicating -> future: do it inside accessors
        cellType = new int[nCells];

        for (int c = 0; c < nCells; c++)
        {
            int val;

            // ### TODO: get id's from vtkCellType.h
            switch (vtkCellTypes[c])
            {
            case 1:
            { // vertex
                val = CELL_POINT;
            }
            break;
            //case 2: { // poly vertex ### actually not supported (using CELL_POINT)
            //val = CELL_POINT;
            //} break;
            case 3:
            { // line
                val = CELL_LINE;
            }
            break;
            //case 4: { // polyline ### actually not supported (using CELL_LINE)
            //val = CELL_LINE;
            //} break;
            case 5:
            { // triangle
                val = CELL_TRI;
            }
            break;
            //case 6: { // triangle strip ### actually not supported (using CELL_TRI)
            //val = CELL_TRI;
            //} break;
            //case 7: { // polygon ### ACTUALLY NOT SUPPORTED
            //val = 0; // ###
            //} break;
            //case 8: { // pixel ### actually not supported (using CELL_QUAD)
            //val = CELL_QUAD;
            //} break;
            case 9:
            { // quad
                val = CELL_QUAD;
            }
            break;
            case 10:
            { // terahedron
                val = CELL_TET;
            }
            break;
            //case 11: { // voxel ### actually not supported (using CELL_HEX)
            //val = CELL_HEX;
            //} break;
            case 12:
            { // hexahedron
                val = CELL_HEX;
            }
            break;
            case 13:
            { // wedge
                val = CELL_PRISM;
            }
            break;
            case 14:
            { // pyramid
                val = CELL_PYR;
            }
            break;
            default:
            {
                printf("Unstructured-VTK: ERROR1: unsupported cell type: %d\n", vtkCellTypes[c]);
                // planar types have opposite orientation, but this should not matter
                // TODO: copy nodes
            }
            }
#ifdef VTK_EXCLUDE_PLANAR_CELLS
            if (vtkCellTypes[c] >= 10)
            { // HACK
                newCellType[newC++] = val;
                newNConn += nVertices[val];
            }
#endif
            cellType[c] = val;
        }
    }

    // convert node order from VTK to AVS
    // the node list buffer is allocated ###,
    // this violates the philosophy to keep memory usage of Unstructured very small
    // TODO: replace all direct accesses to nodelist by accessors and change
    //       the node ordering inside the accessors instead of memory allocation
    {
// DELETEME int *nodeListVTK = (int *) grid->GetCells()->GetPointer();

// alloc nodeList
#ifdef VTK_EXCLUDE_PLANAR_CELLS
        nodeList = new int[newNConn];
#else
        nodeList = new int[nConn];
#endif
        nodeListAllocated = true;

// alloc nodeListOffset
// ### actually allocating and transforming, should do in accessors
// ### no check
#ifdef VTK_EXCLUDE_PLANAR_CELLS
        nodeListOffset = new int[newNCells];
#else
        nodeListOffset = new int[nCells];
#endif
        nodeListOffsetsAllocated = true;

        int nodeIdx = 0;
        int listOffset = 0;
#ifdef VTK_EXCLUDE_PLANAR_CELLS
        int newC = 0;
#endif
        for (int c = 0; c < nCells; c++)
        {

            //int listOffset = nodeListOffset[c];
            // skip int that gives number of points per cell
            listOffset++;
#ifdef VTK_EXCLUDE_PLANAR_CELLS
            nodeListOffset[newC] = nodeIdx;
#else
            nodeListOffset[c] = nodeIdx;
#endif

            switch (cellType[c])
            {
            case CELL_TET:
            { // identical in Covise and VTK
#if 0 // DELETEME
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 3];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 0];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 1];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 2];
#else
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(3);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(0);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(1);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(2);
#endif
                listOffset += 4;
#ifdef VTK_EXCLUDE_PLANAR_CELLS
                newC++;
#endif
            }
            break;
            case CELL_PYR:
            { // identical in Covise and VTK
#if 0 // DELETEME
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 4];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 0];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 1];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 2];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 3];
#else
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(4);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(0);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(1);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(2);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(3);
#endif
                listOffset += 5;
#ifdef VTK_EXCLUDE_PLANAR_CELLS
                newC++;
#endif
            }
            break;
            case CELL_PRISM:
            { // identical in AVS and VTK
#if 0 // DELETEME
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 0];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 1];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 2];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 3];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 4];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 5];
#else
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(0);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(1);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(2);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(3);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(4);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(5);
#endif
                listOffset += 6;
#ifdef VTK_EXCLUDE_PLANAR_CELLS
                newC++;
#endif
            }
            break;
            case CELL_HEX:
            { // identical in Covise and VTK
#if 0 // DELETEME
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 4];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 5];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 6];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 7];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 0];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 1];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 2];
	nodeList[nodeIdx++] = nodeListVTK[listOffset + 3];
#else
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(4);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(5);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(6);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(7);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(0);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(1);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(2);
                nodeList[nodeIdx++] = grid->GetCell(c)->GetPointId(3);
#endif
                listOffset += 8;
#ifdef VTK_EXCLUDE_PLANAR_CELLS
                newC++;
#endif
            }
            break;
            default:
            {
#ifdef VTK_EXCLUDE_PLANAR_CELLS
                listOffset += nVertices[cellType[c]];
#else
                printf("Unstructured-VTK: ERROR2: unsupported cell type: %d\n", cellType[c]);
                // planar types have opposite orientation, but this should not matter
                // TODO: copy nodes
                listOffset++; // ####  TODO: is this ok? (seems not)!!  ######
#endif
            }
            }
        }
    }

#ifdef VTK_EXCLUDE_PLANAR_CELLS
    delete[] cellType;
    cellType = newCellType;
    nCells = newNCells;
    nConn = newNConn;
#endif

    // fake components: scalars come first
    {
        int numNodeComp = 0;
        if (scal)
            numNodeComp += scal->size();
        if (vect)
            numNodeComp += vect->size();

        nodeComponentNb = numNodeComp;
        nodeComponents = new int[nodeComponentNb];
        nodeComponentLabels = new char *[nodeComponentNb];

        // setup
        numNodeComp = 0;
        vectorComponent = -1;
        scalarComponent = -1;
        if (scal)
        {
            for (int co = 0; co < scal->size(); co++)
            {
                nodeComponents[numNodeComp] = 1;

                // ### TODO: reference instead of copy
                nodeComponentLabels[numNodeComp] = new char[256]; // ###
                if ((*scal)[co]->GetName())
                {
                    strcpy(nodeComponentLabels[numNodeComp], (*scal)[co]->GetName());
                }
                else
                { // generate fake label
                    char label[256];
                    sprintf(label, "%p", (*scal)[co]->GetPointer(0));
                    strcpy(nodeComponentLabels[numNodeComp], label);
                }
#if DEBUG_OUTPUT
                printf("VTK-Unstructured[%d]: wrapping scalar: %s\n", numNodeComp, nodeComponentLabels[numNodeComp]);
#endif

                float *wp;
                wp = (*scal)[co]->GetPointer(0); // #### Gives pointer to single entry or to whole interleaved data?

                NodeCompDataPtr dp;
                dp.ptrs.push_back(wp);
                dp.stride = 1;

                nodeComponentDataPtrs.push_back(dp);

                //delete:(*scal)[co]->get_adress(&nodeComponentDataPtrs[numNodeComp]);
                if (scalarComponent < 0)
                    scalarComponent = numNodeComp;
                numNodeComp++;
            }
        }
        if (vect)
        {
            const int vveclen = 3; //#### fixed to veclen 3, TODO: support tuples
            for (int co = 0; co < vect->size(); co++)
            {
                nodeComponents[numNodeComp] = vveclen;

                // ### TODO: reference instead of copy
                nodeComponentLabels[numNodeComp] = new char[256]; // ###
                if ((*vect)[co]->GetName())
                {
                    strcpy(nodeComponentLabels[numNodeComp], (*vect)[co]->GetName());
                }
                else
                {
                    char label[256];
                    sprintf(label, "%p", (*vect)[co]->GetPointer(0));
                    strcpy(nodeComponentLabels[numNodeComp], label);
                }
#if DEBUG_OUTPUT
                printf("VTK-Unstructured[%d]: wrapping vector: %s\n", numNodeComp, nodeComponentLabels[numNodeComp]);
#endif

                NodeCompDataPtr dp;

                float *up, *vp, *wp;
                up = (*vect)[co]->GetPointer(0); // #### Gives pointer to single entry or to whole interleaved data?
                vp = up + 1;
                wp = up + 2;

                dp.ptrs.push_back(up);
                dp.ptrs.push_back(vp);
                dp.ptrs.push_back(wp);
                dp.stride = 3; // VTK data is interleaved (?)

                nodeComponentDataPtrs.push_back(dp);

                //delete: (*vect)[co]->get_adresses(&nodeComponentDataPtrs[numNodeComp]);
                if (vectorComponent < 0)
                    vectorComponent = numNodeComp;
                numNodeComp++;
            }
        }
    }

    // Vector quantity   (first component with veclen > 1)
    vectorExists = vectorComponent >= 0;
    if (vectorExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: vector quantity initially at ucd component %d\n",
               name, vectorComponent);
#endif
        selectVectorNodeData(vectorComponent);
    }

    // Scalar
    scalarExists = scalarComponent >= 0;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: scalar quantity initially at ucd component %d\n",
               name, scalarComponent);
#endif
        selectScalarNodeData(scalarComponent);
    }

    // Wall Distance
    wallDistExists = false;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: warning: wall distance initially at scalar quantity ucd component\n", name);
#endif
        wallDist = p; // TODO: general case
        wallDistExists = true;
        wallDistComponent = scalarComponent;
    }

    // setup
    divideVelocityByWalldist = false;
    extrapolateToNoSlipBoundary = false;

    // setup grid
    searchGrid = 0;
    cellRadiusSqr = 0;
    cellCentroid = NULL;
    nodeNeighbors = NULL;
    cellNeighbors = NULL;
    //vGradient = NULL;

    computeBoundingBox();
    computeCentroids();
    computeTetDecomposition();

    // initialize extra data arrays
    for (int i = 0; i < nodeComponentNb; i++)
    {
        std::vector<DataDesc *> w;
        nodeCompExtraData.push_back(w);
    }

    nodeComponentVecLenTot = 0;
    nodeComponentVecLenCumulated.clear();
    for (int i = 0; i < nodeComponentNb; i++)
    {
        nodeComponentVecLenCumulated.push_back(nodeComponentVecLenTot);
        nodeComponentVecLenTot += getNodeCompVecLen(i);
    }

    if (nCells > 0)
        loadCell(0, true);
}
#endif

#if 0 // DELETEME:

void Unstructured::setupUnstructured(Unstructured *templ, DataDesc *dd)
{
#if DEBUG_OUTPUT
  printf("this is Unstructured %s\n", UNST_VERSION);
#endif

  *this = *templ;
  isCopy = true;
  nameAllocated = false; // ##### ok?
  nodeListAllocated = false; // ### ok?
  nodeListOffsetsAllocated = false; // ### ok?
  cellTypesAllocated = false; // ### ok?
  coordinatesAllocated = false; // #### ok?
  allocatedNodeDataComponents.clear();
  vector3CB = NULL;
  
  // Components
  nodeComponents = new int[1];
  nodeComponents[0] = dd->veclen;
  nodeComponentNb = 1;
  
  // sample pointers to node components data
  nodeComponentDataPtrs.clear();
  NodeCompDataPtr dp;
  // actually dd must point to interleaved data
  for (int vv=0; vv<dd->veclen; vv++) {
    dp.ptrs.push_back(dd->p.f + vv);
  }
  dp.stride = dd->veclen; 
  nodeComponentDataPtrs.push_back(dp);
  
  scalarExists = false;
  vectorExists = false;
  wallDistExists = false; 

  if (dd->veclen > 1) {
    // Vector quantity
    vectorComponent = 0;
    vectorExists = true;
#if DEBUG_OUTPUT
    printf("unstructured[%s]: vector quantity initially at ucd component %d\n",
           name, vectorComponent);
#endif
    selectVectorNodeData(vectorComponent);
  }
  else {
    // Scalar
    scalarComponent = 0;
    scalarExists = true;
#if DEBUG_OUTPUT
    printf("unstructured[%s]: scalar quantity initially at ucd component %d\n",
           name, scalarComponent);
#endif
    selectScalarNodeData(scalarComponent);
  }
  
  // Wall Distance : TODO ####

  for (int i=0; i<nodeComponentNb; i++) {
    nodeCompExtraData[i].clear();
  }

  nodeComponentVecLenTot = 0;
  nodeComponentVecLenCumulated.clear();
  for (int i=0; i<nodeComponentNb; i++) {
    nodeComponentVecLenCumulated.push_back(nodeComponentVecLenTot);
    nodeComponentVecLenTot += getNodeCompVecLen(i);
  }

}

#else

void Unstructured::setupUnstructured(Unstructured *templ, DataDesc *dd)
{
    std::vector<DataDesc *> dv;
    dv.push_back(dd);
    setupUnstructured(templ, &dv);
}

#endif

void Unstructured::setupUnstructured(Unstructured *templ, std::vector<DataDesc *> *dv)
{
#if DEBUG_OUTPUT
    printf("this is Unstructured %s\n", UNST_VERSION);
#endif

    *this = *templ;
    isCopy = true;
    nameAllocated = false; // ##### ok?
    nodeListAllocated = false; // ### ok?
    nodeListOffsetsAllocated = false; // ### ok?
    cellTypesAllocated = false; // ### ok?
    coordinatesAllocated = false; // #### ok?
    allocatedNodeDataComponents.clear();
    vector3CB = NULL;

    // Components
    nodeComponents = new int[dv->size()];
    for (int c = 0; c < (int)dv->size(); c++)
    {
        nodeComponents[c] = (*dv)[c]->veclen;
    }
    nodeComponentNb = int(dv->size());
    nodeComponentLabels = NULL; // #### HACK

    // sample pointers to node components data
    nodeComponentDataPtrs.clear();
    for (int c = 0; c < (int)dv->size(); c++)
    {
        NodeCompDataPtr dp;
        // actually dd must point to interleaved data
        for (int vv = 0; vv < (*dv)[c]->veclen; vv++)
        {
            dp.ptrs.push_back((*dv)[c]->p.f + vv);
        }
        dp.stride = (*dv)[c]->veclen;
        nodeComponentDataPtrs.push_back(dp);
    }

    scalarExists = false;
    vectorExists = false;
    wallDistExists = false;

    for (int c = 0; c < (int)dv->size(); c++)
    {
        if ((!vectorExists) && ((*dv)[c]->veclen > 1))
        {
            // Vector quantity
            vectorComponent = c;
            vectorExists = true;
#if DEBUG_OUTPUT
            printf("unstructured[%s]: vector quantity initially at ucd component %d\n",
                   name, vectorComponent);
#endif
            selectVectorNodeData(vectorComponent);
        }
        else if ((!scalarExists) && ((*dv)[c]->veclen == 1))
        {
            // Scalar
            scalarComponent = c;
            scalarExists = true;
#if DEBUG_OUTPUT
            printf("unstructured[%s]: scalar quantity initially at ucd component %d\n",
                   name, scalarComponent);
#endif
            selectScalarNodeData(scalarComponent);
        }
    }

// Wall Distance : TODO ####

#if 0 // DELETEME: is a bug
  for (int i=0; i<nodeComponentNb; i++) {
    nodeCompExtraData[i].clear();
  }
#else
    nodeCompExtraData.clear();
    for (int i = 0; i < nodeComponentNb; i++)
    {
        std::vector<DataDesc *> w;
        nodeCompExtraData.push_back(w);
    }
#endif

    nodeComponentVecLenTot = 0;
    nodeComponentVecLenCumulated.clear();
    for (int i = 0; i < nodeComponentNb; i++)
    {
        nodeComponentVecLenCumulated.push_back(nodeComponentVecLenTot);
        nodeComponentVecLenTot += getNodeCompVecLen(i);
    }
}

Unstructured::Unstructured(Unstructured *templ, DataDesc *dd)
{
    setupUnstructured(templ, dd);
    if (nCells > 0)
        loadCell(0, true);
}

#if 1
Unstructured::Unstructured(Unstructured *templ, int veclen)
{ // allocates single component of veclen

    float *dat = new float[veclen * templ->nNodes];
    DataDesc dd = DataDesc(0, TP_FLOAT, veclen, dat);

    setupUnstructured(templ, &dd);

    allocatedNodeDataComponents.push_back(dd.p.f);
}

#else // DELETEME

Unstructured::Unstructured(Unstructured *templ, int veclen)
{ // allocates single component of veclen

    int components[1] = { veclen };
    Unstructured(templ, 1, components);
    if (nCells > 0)
        loadCell(0, true);
}

#endif

// ######### TODO: this method is buggy, see ucd_ridge_surface !!
Unstructured::Unstructured(Unstructured *templ, int componentCnt, int *components)
{ // allocates components

    std::vector<DataDesc *> dv;

    for (int c = 0; c < componentCnt; c++)
    {
        float *dat = new float[components[c] * templ->nNodes];
        DataDesc *dd = new DataDesc(0, TP_FLOAT, components[c], dat);
        dv.push_back(dd);
    }

    setupUnstructured(templ, &dv);

    for (int c = 0; c < componentCnt; c++)
    {
        allocatedNodeDataComponents.push_back(dv[c]->p.f);
        delete dv[c];
    }
    if (nCells > 0)
        loadCell(0, true);
}

Unstructured::Unstructured(const char *fileName)
{
    nNodes = 0;
    nCells = 0;
    nodeComponentNb = 0;

    edgeNb = -1;
    faceNb = -1;

    FILE *fp = fopen(fileName, "rb");
    if (!fp)
    {
        fprintf(stderr, "Cannot read %s\n", fileName);
        return;
    }

    isCopy = false;
    allocatedNodeDataComponents.clear();
    strictNodeCompExtraDeletion = false;
    transientDataDict = NULL;
    transientFile = NULL;
    transient = false;
    vector3CB = NULL;

    name = new char[160];
    nameAllocated = true;
    fread(name, sizeof(char), 160, fp);
    fread(&nCells, sizeof(int), 1, fp);
    fread(&nNodes, sizeof(int), 1, fp);

    // Components
    fread(&nodeComponentNb, sizeof(int), 1, fp);
    nodeComponents = new int[nodeComponentNb];
    fread(nodeComponents, sizeof(int), nodeComponentNb, fp);
#if 0
  nodeComponentLabels = NULL; // #### HACK
#else
    nodeComponentLabels = new char *[nodeComponentNb];
    // ### TODO: reference instead of copy
    for (int c = 0; c < nodeComponentNb; c++)
    {
        nodeComponentLabels[c] = new char[256]; // ###
        fread(nodeComponentLabels[c], sizeof(char), 256, fp);
    }
#endif

    // Coordinates
    x = new float[nNodes];
    y = new float[nNodes];
    z = new float[nNodes];
    coordinatesAllocated = true;
    fread(x, sizeof(float), nNodes, fp);
    fread(y, sizeof(float), nNodes, fp);
    fread(z, sizeof(float), nNodes, fp);
    cStride = 1; // non-interleaved coordinate data

    // Data
    vectorComponent = -1;
    scalarComponent = -1;
    for (int i = 0; i < nodeComponentNb; i++)
    {
        int veclen = nodeComponents[i];
        if (veclen == 1 && scalarComponent == -1)
            scalarComponent = i;
        if (veclen > 1 && vectorComponent == -1)
            vectorComponent = i;
        float *dptr = new float[nNodes * veclen]; // data is interleaved ###
        fread(dptr, sizeof(float), nNodes * veclen, fp);

        NodeCompDataPtr dp;
        for (int vv = 0; vv < veclen; vv++)
        {
            dp.ptrs.push_back(dptr + vv); // data is interleaved ###
        }
        dp.stride = veclen; // data is interleaved ###
        nodeComponentDataPtrs.push_back(dp);
        allocatedNodeDataComponents.push_back(dptr);
    }

    // Vector quantity   (first component with veclen > 1)
    //vectorComponent = ucd_findNodeCompByVeclen(ucd, 1, 0, 1);
    vectorExists = vectorComponent >= 0;
    if (vectorExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: vector quantity initially at ucd component %d\n",
               name, vectorComponent);
#endif
        selectVectorNodeData(vectorComponent);
    }

    // Scalar
    //scalarComponent = ucd_findNodeCompByVeclen(ucd, 1, 0);
    scalarExists = scalarComponent >= 0;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: scalar quantity initially at ucd component %d\n",
               name, scalarComponent);
#endif
        selectScalarNodeData(scalarComponent);
    }

    // Wall Distance
    wallDistExists = false;
    if (scalarExists)
    {
#if DEBUG_OUTPUT
        printf("unstructured[%s]: warning: wall distance initially at scalar quantity ucd component\n", name);
#endif
        wallDist = p; // TODO: general case
        wallDistExists = true;
    }

    // Cells
    cellType = new int[nCells];
    cellTypesAllocated = true;
    fread(cellType, sizeof(int), nCells, fp);

    int nodeListSize = getNodeListSize();
    nodeList = new int[nodeListSize];
    nodeListAllocated = true;
    fread(nodeList, sizeof(int), nodeListSize, fp);

    nodeListOffset = new int[nCells];
    nodeListOffsetsAllocated = true;
    fread(nodeListOffset, sizeof(int), nCells, fp);

    divideVelocityByWalldist = false;
    extrapolateToNoSlipBoundary = false;

    searchGrid = 0;
    cellRadiusSqr = 0;
    cellCentroid = NULL;
    nodeNeighbors = NULL;
    cellNeighbors = NULL;
    //vGradient = NULL;

    computeBoundingBox();
    computeCentroids();
    computeTetDecomposition();

    for (int i = 0; i < nodeComponentNb; i++)
    {
        std::vector<DataDesc *> w;
        nodeCompExtraData.push_back(w);
    }

    nodeComponentVecLenTot = 0;
    nodeComponentVecLenCumulated.clear();
    for (int i = 0; i < nodeComponentNb; i++)
    {
        nodeComponentVecLenCumulated.push_back(nodeComponentVecLenTot);
        nodeComponentVecLenTot += getNodeCompVecLen(i);
    }

    fclose(fp);
    if (nCells > 0)
        loadCell(0, true);
}

Unstructured::~Unstructured()
{
    if (vector3CB)
        unsetVector3CB();

    if (!isCopy)
    {
#if DEBUG_OUTPUT
        printf("Cell search statistics:\n");
        printf("%30d found in same cell\n", ct1);
        printf("%30d found in new cell\n", ct2);
        printf("%30d not found\n", ct0);
#endif

        if (searchGrid)
        {
            delete[] searchGrid -> box;
            //delete[] cellCentroid;
            //delete[] cellRadiusSqr;
            delete searchGrid;
        }
        if (cellCentroid)
            delete[] cellCentroid;
        if (cellRadiusSqr)
            delete[] cellRadiusSqr;

        deleteNodeNeighbors();
        deleteCellNeighbors();
    }

    if (nameAllocated)
        delete[] name;

    if (coordinatesAllocated)
    {
        delete[] x;
        delete[] y;
        delete[] z;
    }

    for (int p = 0; p < (signed)allocatedNodeDataComponents.size(); p++)
    {
        delete[] allocatedNodeDataComponents[p];
    }

    deleteNodeCompExtraData();
    delete[] nodeComponents;
    if (nodeComponentLabels)
    {
        for (int c = 0; c < this->getNodeCompNb(); c++)
            delete[] nodeComponentLabels[c];
        delete[] nodeComponentLabels;
    }
    //free(nodeComponentDataPtrs);
    if (nodeListAllocated)
        delete[] nodeList;
    if (nodeListOffsetsAllocated)
        delete[] nodeListOffset;
    if (cellTypesAllocated)
        delete[] cellType;
#ifdef UNST_DATA_DICT
    if (transientDataDict)
        delete transientDataDict;
#endif
    if (transientFile)
    {
        munmap(transientFile, transientFileSize);
    }
}

#ifdef UNST_DATA_DICT
void Unstructured::setTransient(char *dataDictDir, int cacheSize, bool verbose)
{

    // disable transientFile method
    if (transientFile)
    {
        munmap(transientFile, transientFileSize);
        transientFile = NULL;
    }

    if (transientDataDict)
        delete transientDataDict;
    transientDataDict = new DataDict();
    transientDataDict->setDataDict(dataDictDir, cacheSize, verbose);

    // compute size of data
    // TODO: - actually datadict must be identical to unstructured ! #####
    //         (same components in same order)
    //       - future: Allow any nonzero intersection of dict and unst.
    //                 This is done by storing node positions, component
    //                 labels, and veclens in dict for checking
    int vecLenTot = 0;
    for (int i = 0; i < nodeComponentNb; i++)
    {
        vecLenTot += getNodeCompVecLen(i);
    }

    if (!transientDataDict->setByteSize(nNodes * vecLenTot * sizeof(float)))
    {
        // TODO: react
    }
    printf("setTransient: new data dict created\n");

    transient = true;
}
#endif

void Unstructured::mapTransientFile(int fileIdx)
{ // unmaps old, maps new
    transientFileIdx = fileIdx;

    // get size of binary file
    int fileSize;
    {
        int vecLenTot = 0;
        for (int i = 0; i < nodeComponentNb; i++)
        {
            vecLenTot += getNodeCompVecLen(i);
        }
        fileSize = transientFilesTimeSteps[int(transientFileIdx].size() * nNodes * vecLenTot * sizeof(float));
    }

    if (transientFileVerbose == 2)
        printf("mapping %s\n", transientFiles[transientFileIdx].c_str());

    // open file
    int fd = open(transientFiles[transientFileIdx].c_str(), O_RDONLY);
    if (fd < 1)
    {
        printf("could not open transient file %s\n", transientFiles[transientFileIdx].c_str());
        exit(1); // ###
    }

    if (transientFile)
        munmap(transientFile, transientFileSize);

    transientFile = (float *)mmap(0, int(fileSize), PROT_READ, MAP_SHARED, fd, 0);

    transientFileSize = fileSize;

    // tell the virtual memory system about expected accesses
    //madvise(transientFile, transientFileSize, MADV_RANDOM);
    madvise(transientFile, transientFileSize, MADV_SEQUENTIAL); // to be used
    //madvise(transientFile, transientFileSize, MADV_WILLNEED);
    //madvise(transientFile, transientFileSize, MADV_DONTNEED);

    close(fd); // file does not need to be open for mmap

    if (transientFile == MAP_FAILED)
    {
        fprintf(stderr, "could not map file %s\n  fileSize=%d, fd=%d\n error:  %s\n",
                transientFiles[transientFileIdx].c_str(),
                fileSize, fd,
                strerror(errno));
        printf("  one possible cause is that the Unstructured object has more channels (UCD components) than the transient file\n  another possible cause is exhausted address space -> use smaller map files\n");
        transient = false;
        transientFile = NULL;
        //return; // TODO: return false ###
        exit(1); // ####
    }
    else
    {
        //fprintf(stderr, "successfully mapped file %s\n  fileSize=%d, fd=%d\n",
        //      transientFiles[transientFileIdx].c_str(),
        //      fileSize, fd);
    }

    transient = true;

    lastTimeStepL = -1;
    lastTimeStepU = -1;

    transientFileMapNb++;
}

void Unstructured::setTransientFile(const char *dataInfoFile, int verbose)
{
    transientFileVerbose = verbose;
    transientFileMapNb = 0;

#ifdef UNST_DATA_DICT
    // disable DataDict method
    if (transientDataDict)
    {
        delete transientDataDict;
        transientDataDict = NULL;
    }
#endif

    // open info file
    FILE *fp = fopen(dataInfoFile, "r");
    if (!fp)
    {
        printf("could not open info file %s\n", dataInfoFile);
        exit(1); // ###
    }
    if (transientFileVerbose > 0)
        printf("info file: %s\n", dataInfoFile);

    // get number of components
    int compNb;
    fscanf(fp, "%d", &compNb);

    // get labels
    // TODO
    // #### actually assuming one component and skipping
    fscanf(fp, "%*s");

    // get number of files of current component
    int fileNb;
    fscanf(fp, "%d", &fileNb);

    transientFiles.clear();
    transientFilesTimeSteps.clear();
    for (int file = 0; file < fileNb; file++)
    {

        // get name of binary file
        {
            char fileName[1024];
            char name[1024];
            fscanf(fp, "%s", name);
            char path[1024];
            strcpy(path, dataInfoFile);
            sprintf(fileName, "%s/%s", dirname(path), name);
            transientFiles.push_back(string(fileName));
        }

        // get number of time steps
        int timeStepNb;
        {
            char buf[1024];
            fscanf(fp, "%s", buf);
            timeStepNb = atoi(buf);
        }

        // read in time step info
        std::vector<float> vec;
        transientFilesTimeSteps.push_back(vec);
        for (int t = 0; t < timeStepNb; t++)
        {
            // get time
            char time[1024];
            fscanf(fp, "%s", time);
            transientFilesTimeSteps[file].push_back(atof(time));
        }

        if (transientFileVerbose > 0)
            printf("transient file %d: name=%s steps=%d\n", file,
                   transientFiles[file].c_str(), (int)transientFilesTimeSteps[file].size());
    }

    fclose(fp);

    // mmap first file
    mapTransientFile(0);
}

void Unstructured::unsetTransientFile(void)
{
    if (transientFile)
    {
        munmap(transientFile, transientFileSize);
        transientFile = NULL;
        transient = false;
    }
}

// TODO: move to another class
int Unstructured::getTimeSteps(double time, int &step1, int &step2,
                               double &weight1, double &weight2)
{ // returns 1 if time is exactly on one time step
// returns 2 if time is between two time steps
// returns 0 if time step not found

// TODO: check if in transientFile mode ###
#if 0 // linear search
  int lower=0, upper=0;
  for (int t=0; t<(int)transientFileTimeSteps.size() - 1; t++) {

    double t1 = transientFileTimeSteps[t];
    double t2 = transientFileTimeSteps[t + 1];

    if (time == t1) {
      lower = t;
      upper = t;
      step1 = lower;
      step2 = upper;
      weight1 = 1.0;
      weight2 = 0.0;
      return 1;
    }

    if (time == t2) {
      lower = t + 1;
      upper = t + 1;
      step1 = lower;
      step2 = upper;
      weight1 = 1.0;
      weight2 = 0.0;
      return 1;
    }

    bool lowerFound=false, upperFound=false;
    if (time > t1) {
      lower = t;
      lowerFound = true;
    }
    if (time < t2) {
      upper = t + 1;
      upperFound = true;
    }
    if (lowerFound && upperFound) {
      step1 = lower;
      step2 = upper;

      // compute weights
      double delta = t2 - t1;
      weight1 = (t2 - time) / delta;
      weight2 = (time - t1) / delta;
      return 2;
    }
  }
  return 0;

#else // binary search

    if (lastTimeStepL == -2)
    { // ################ TODO (lastTimeStepL != -1)?
        if ((time >= lastTimeL) && (time <= lastTimeU))
        {
            step1 = lastTimeStepL;
            step2 = lastTimeStepU;
            weight1 = lastWeightL;
            weight2 = lastWeightU;
            return 2; // ###
        }
    }

    // ### heuristic
    double epsilon = (transientFilesTimeSteps[transientFileIdx].back() - transientFilesTimeSteps[transientFileIdx].front()) * 1e-6;

    int fileNb = int(transientFilesTimeSteps.size());

    // map corresponding file, if necessary
    // epsilon applies at first step of first file and last step of last file
    if ((time + (transientFileIdx == 0 ? epsilon : 0.0) < transientFilesTimeSteps[transientFileIdx].front()) || (time - (transientFileIdx == fileNb - 1 ? epsilon : 0.0) > transientFilesTimeSteps[transientFileIdx].back()))
    {

        //printf("NEW MAP NOT FOUND, searching time=%.15f transientFileIdx=%d\n min=%.15f max=%.15f\n", time, transientFileIdx, transientFilesTimeSteps[transientFileIdx].front(), transientFilesTimeSteps[transientFileIdx].back());

        // find transient file
        for (int file = 0; file < (int)transientFiles.size(); file++)
        {

            if ((time + (transientFileIdx == 0 ? epsilon : 0.0) >= transientFilesTimeSteps[file].front()) && (time - (transientFileIdx == fileNb - 1 ? epsilon : 0.0) <= transientFilesTimeSteps[file].back()))
            {
                // found

                // unmap old and map new file
                mapTransientFile(file);

                break;
            }
        }
    }

    // ADAPTING TIME WITH RESPECT TO EPSILON
    if (time < transientFilesTimeSteps[transientFileIdx].front())
    {
        time += epsilon;
    }
    else if (time > transientFilesTimeSteps[transientFileIdx].back())
    {
        time -= epsilon;
    }

    int timeStepNb = transientFilesTimeSteps[transientFileIdx].size();
    int lower = 0;
    int upper = timeStepNb - 1;

    double tl = transientFilesTimeSteps[transientFileIdx][lower];
    double tu = transientFilesTimeSteps[transientFileIdx][upper];

    if ((time < tl) || (time > tu))
    {
        fprintf(stderr, "Unstructured::getTimeSteps: error: time=%g out of range at epsilon=%g\n", time, epsilon);
        return 0;
    }

    if (time <= tl)
    {
        step1 = lower;
        step2 = lower;
        weight1 = 1.0;
        weight2 = 0.0;
        lastTimeStepL = step1;
        lastTimeStepU = step2;
        lastWeightL = weight1;
        lastWeightU = weight2;
        return 1;
    }

    if (time >= tu)
    {
        step1 = upper;
        step2 = upper;
        weight1 = 1.0;
        weight2 = 0.0;
        lastTimeStepL = step1;
        lastTimeStepU = step2;
        lastWeightL = weight1;
        lastWeightU = weight2;
        return 1;
    }

    while (true)
    {

        if (upper - lower <= 1)
            break;

        int mid = (lower + upper) / 2;
        double tm = transientFilesTimeSteps[transientFileIdx][mid];

        if (time <= tm)
        {
            upper = mid;
            tu = tm;
        }
        else
        {
            lower = mid;
            tl = tm;
        }
    }

    step1 = lower;
    step2 = upper;

    if (lower == upper)
    {
        step1 = lower;
        step2 = upper;
        weight1 = 1.0;
        weight2 = 0.0;
        lastTimeStepL = step1;
        lastTimeStepU = step2;
        lastWeightL = weight1;
        lastWeightU = weight2;
        return 1;
    }

    // compute weights
    double delta = tu - tl;
    weight1 = (tu - time) / delta;
    weight2 = (time - tl) / delta;

    lastTimeStepL = step1;
    lastTimeStepU = step2;
    lastWeightL = weight1;
    lastWeightU = weight2;

    return 2;
#endif
}

void Unstructured::setVector3CB(void (*vec3CB)(vec3 pos, vec3 out, double time),
                                bool restrictToGrid)
{
    vector3CBBackupComp = vectorComponent;
    vector3CB = vec3CB;
    vector3CBRestrictToGrid = restrictToGrid;
}

void Unstructured::unsetVector3CB(void)
{
    vector3CB = NULL;
    selectVectorNodeData(vector3CBBackupComp);
}

int Unstructured::getEdgeNb(void)
{

    if (edgeNb >= 0)
        return edgeNb;

    edgeNb = 0;

    // get edges
    std::map<pair<int, int>, bool> alledges;
    for (int c = 0; c < nCells; c++)
    {

        for (int e = 0; e < nEdges[getCellType(c)]; e++)
        {
            int v1, v2;
            getCellEdgeNodesConnOrderAVS(c, e, v1, v2);

            // order indices
            if (v2 < v1)
            {
                int w = v1;
                v1 = v2;
                v2 = w;
            }

            // store inside map
            std::pair<int, int> key;
            key.first = v1;
            key.second = v2;
            if (alledges.find(key) == alledges.end())
            {
                // new edge
                alledges[key] = true;
                edgeNb++;
            }
            else
            {
                // existing edge
            }
        }
    }

    //printf("ucd has %d edges\n", nedges);

    return edgeNb;
}

// ### move
void sortInts(int &i1, int &i2, int &i3, int &i4)
{
    int w[4] = { i1, i2, i3, i4 };
    sort(w, w + 4);
    i1 = w[0];
    i2 = w[1];
    i3 = w[2];
    i4 = w[3];
}

// ### move
#if 0
class Keys4 {
public:
  Keys4(int k1, int k2, int k3, int k4) : key1(k1), key2(k2), key3(k3), key4(k4) { }
  bool operator<(const Keys4 &right) const {
    return (key1 < right.key1 ||
            (key1 == right.key1 &&
             (key2 < right.key2 || 
              (key2 == right.key2 &&
               (key3 < right.key3 ||
                (key3 == right.key3 &&
                 (key4 < right.key4))))))
            );
  }

  int key1, key2, key3, key4;
};
#endif

int Unstructured::getFaceNb(void)
{
    if (faceNb >= 0)
        return faceNb;

    faceNb = 0;

    // get faces
    std::map<Keys4, bool> allfaces;
    for (int c = 0; c < nCells; c++)
    {

        int *cellNodes = getCellNodesAVS(c);

        for (int f = 0; f < nFaces[getCellType(c)]; f++)
        {
            int v1, v2, v3, v4;
            v1 = cellNodes[faces[getCellType(c)][f][0]];
            v2 = cellNodes[faces[getCellType(c)][f][1]];
            v3 = cellNodes[faces[getCellType(c)][f][2]];
            v4 = cellNodes[faces[getCellType(c)][f][3]];

            // order indices
            sortInts(v1, v2, v3, v4);

            // store inside map
            Keys4 key(v1, v2, v3, v4);
            if (allfaces.find(key) == allfaces.end())
            {
                // new face
                allfaces[key] = true;
                faceNb++;
            }
            else
            {
                // existing face
            }
        }
    }

    //printf("ucd has %d faces\n", nfaces);

    return faceNb;
}

int *Unstructured::getCellNodesAVS(void)
{
    return this->getCellNodesAVS(this->getCellIndex());
}

int *Unstructured::getCellNodesAVS(int cell)
{ // get zero-based node indices of cell, in AVS order

    // TODO ###: generalize, actual order is AVS

    return &nodeList[nodeListOffset[cell]];
}

DataDesc *Unstructured::findNodeCompExtraData(int comp, int operationId)
{ // TODO: make more efficient and inline (used by e.g. selectVectorNodeData)
    for (int i = 0; i < (signed)nodeCompExtraData[comp].size(); i++)
    {
        if (nodeCompExtraData[comp][i]->operationId == operationId)
        {
            return nodeCompExtraData[comp][i];
        }
    }
    return NULL;
}

int Unstructured::findNodeCompExtraDataIdx(int comp, int operationId)
{
    for (int i = 0; i < (signed)nodeCompExtraData[comp].size(); i++)
    {
        if (nodeCompExtraData[comp][i]->operationId == operationId)
        {
            return i;
        }
    }
    return -1;
}

int Unstructured::findNodeCompExtraDataCompIdx(DataDesc *dd, int &comp, int &idx)
{ // returns 0 if not found
    for (int c = 0; c < nodeComponentNb; c++)
    {
        for (int i = 0; i < (signed)nodeCompExtraData[c].size(); i++)
        {
            if (nodeCompExtraData[c][i] == dd)
            {
                comp = c;
                idx = i;
                return 1;
            }
        }
    }
    return 0;
}

void Unstructured::computeMinBoxSize(fvec3 minSize)
{
    minSize[0] = 0;
    minSize[1] = 0;
    minSize[2] = 0;

    for (int i = 0; i < nCells; i++)
    {
        CellInfo &c = loadCell(i);

        fvec3 min, max, minBox;
        min[0] = 1e19f;
        min[1] = 1e19f;
        min[2] = 1e19f;
        max[0] = -1e19f;
        max[1] = -1e19f;
        max[2] = -1e19f;

        int n = nVertices[c.type];

        for (int j = 0; j < n; j++)
        {
            if (c.vertex[j][0] < min[0])
                min[0] = c.vertex[j][0];
            if (c.vertex[j][1] < min[1])
                min[1] = c.vertex[j][1];
            if (c.vertex[j][2] < min[2])
                min[2] = c.vertex[j][2];
            if (c.vertex[j][0] > max[0])
                max[0] = c.vertex[j][0];
            if (c.vertex[j][1] > max[1])
                max[1] = c.vertex[j][1];
            if (c.vertex[j][2] > max[2])
                max[2] = c.vertex[j][2];
        }
        minBox[0] = 2.001 * (max[0] - min[0]); // more than twice the cell extent
        minBox[1] = 2.001 * (max[1] - min[1]);
        minBox[2] = 2.001 * (max[2] - min[2]);

        if (minSize[0] < minBox[0])
            minSize[0] = minBox[0];
        if (minSize[1] < minBox[1])
            minSize[1] = minBox[1];
        if (minSize[2] < minBox[2])
            minSize[2] = minBox[2];
    }
}

void Unstructured::computeCentroids()
{
    cellCentroid = new fvec3[nCells];
    cellRadiusSqr = new float[nCells];

    for (int i = 0; i < nCells; i++)
    {
        // this call to loadCell uses default time=0, this is important for
        // rotating zones: the centroid is the position at time=0 (not rotated) !!
        CellInfo &c = loadCell(i);

        int n = nVertices[c.type];

        // Compute centroid
        vec3 v = { 0 };
        for (int j = 0; j < n; j++)
        {
            vec3add(v, c.vertex[j], v);
        }
        vec3scal(v, 1. / (double)n, v);
        vec3tofvec3(v, cellCentroid[i]);

        // Compute square of radius
        double sqrMax = 0;
        for (int j = 0; j < n; j++)
        {
            double r2 = vec3distSqr(v, c.vertex[j]);
            if (r2 > sqrMax)
                sqrMax = r2;
        }
        cellRadiusSqr[i] = 1.00001 * sqrMax;
    }
}

CellInfo &Unstructured::loadCell(int i, bool complete, double time)
{
    CellInfo &c = currCell;
    c.index = i;
    c.type = cellType[i];
    c.radiusSqr = cellRadiusSqr[i];
    c.atWall = false;
    c.dataLoaded = false;
    int n = nVertices[c.type];

    for (int j = 0; j < n; j++)
    {
        int k = nodeList[nodeListOffset[i] + j]; // Get in AVS order!
        c.node[j] = k;
        getCoords(k, c.vertex[j], time);
    }

    if (!complete)
        return c;

    // Load data
    loadCellData(time);

    return c;
}

void Unstructured::loadCellData(double time)
{
    CellInfo &c = currCell;
    int n = nVertices[c.type];

    for (int j = 0; j < n; j++)
    {
        int k = nodeList[nodeListOffset[c.index] + j]; // Get in AVS order!!

        if (!transient)
        {
            if (vector3CB || vectorExists)
                getVector3(k, c.vec[j]);
            if (scalarExists)
                c.scal[j] = getScalar(k);
        }
        else
        {
            if (vector3CB || vectorExists)
                getVector3(k, time, c.vec[j]);
            if (scalarExists)
                c.scal[j] = getScalar(k, time);
        }
        if (wallDistExists)
        {
            c.wallDist[j] = wallDist[sStride * k]; // TODO: also handle "solid" bool attribute
            if (c.wallDist[j] == 0)
                c.atWall = true;
        }
    }

    if (scalarExists)
    {
        c.currCell_scalarComponent = scalarComponent;
        c.currCell_scalarExtraData = scalarComponentExtraData;
    }
    if (vectorExists)
    {
        c.currCell_vectorComponent = vectorComponent;
        c.currCell_vectorExtraData = vectorComponentExtraData;
    }
    if (wallDistExists)
    {
        c.currCell_wallDistComponent = wallDistComponent;
    }

    if (transient)
    {
        c.currCell_time = time;
    }

    // compute derived data
    // moved from loadCell()
    // TODO ###: support time (gradients ...)
    if (c.atWall && extrapolateToNoSlipBoundary && (vector3CB || vectorExists) && wallDistExists)
    {
        //if (!vGradient) computeVelocityGradient();
        DataDesc *gradD = findNodeCompExtraData(vectorComponent,
                                                // TODO: find a non-nasty solution for:
                                                //(int) gradient(0, true)
                                                OP_GRADIENT);
        if (!gradD)
            gradD = gradient(vectorComponent);
        fmat3 *grad = (fmat3 *)gradD->p.f;
        for (int j = 0; j < nEdges[c.type]; j++)
        {
            int a = edges[c.type][j][0];
            int b = edges[c.type][j][1];
            vec3 dx, dv;
            mat3 m;

            if (c.wallDist[a] == 0 && c.wallDist[b] != 0)
            {
                vec3sub(c.vertex[a], c.vertex[b], dx);
                //fmat3tomat3(vGradient[c.node[b]], m);
                fmat3tomat3(grad[c.node[b]], m);
                mat3vec(m, dx, dv);
                vec3add(c.vec[b], dv, c.vec[a]);
            }
            else if (c.wallDist[b] == 0 && c.wallDist[a] != 0)
            {
                vec3sub(c.vertex[b], c.vertex[a], dx);
                // fmat3tomat3(vGradient[c.node[a]], m);
                fmat3tomat3(grad[c.node[a]], m);
                mat3vec(m, dx, dv);
                vec3add(c.vec[a], dv, c.vec[b]);
            }
        }
    }

    // Get face centroids and cell centroid, interpolate coords and data
    for (int j = 0; j < nFaces[c.type]; j++)
    {
        int *nr = faces[c.type][j];
        if (nr[3] != -1)
        { // quad face
            vec3avg4(c.vertex[nr[0]], c.vertex[nr[1]], c.vertex[nr[2]], c.vertex[nr[3]], c.vertex[n]); // overhead, TODO
            if (vector3CB || vectorExists)
                vec3avg4(c.vec[nr[0]], c.vec[nr[1]], c.vec[nr[2]], c.vec[nr[3]], c.vec[n]);
            if (scalarExists)
                c.scal[n] = (c.scal[nr[0]] + c.scal[nr[1]] + c.scal[nr[2]] + c.scal[nr[3]]) / 4.;
            if (wallDistExists)
                c.wallDist[n] = (c.wallDist[nr[0]] + c.wallDist[nr[1]] + c.wallDist[nr[2]] + c.wallDist[nr[3]]) / 4.; // overhead, TODO
            n++;
        }
    }
    if (c.type == CELL_PRISM)
    {
        //fvec3tovec3(cellCentroid[c.index], c.vertex[n]);  // overhead, TODO
        getCellCentroid(c.index, c.vertex[n], time);
        if (vector3CB || vectorExists)
            vec3avg3(c.vec[6], c.vec[7], c.vec[8], c.vec[n]);
        if (scalarExists)
            c.scal[n] = (c.scal[6] + c.scal[7] + c.scal[8]) / 3.;
        if (wallDistExists)
            c.wallDist[n] = (c.wallDist[6] + c.wallDist[7] + c.wallDist[8]) / 3.; // overhead, TODO
    }
    else if (c.type == CELL_HEX)
    {
        //fvec3tovec3(cellCentroid[c.index], c.vertex[n]);  // overhead, TODO
        getCellCentroid(c.index, c.vertex[n], time);
        if (vector3CB || vectorExists)
            vec3avg(c.vec[8], c.vec[13], c.vec[n]);
        if (scalarExists)
            c.scal[n] = (c.scal[8] + c.scal[13]) / 2.;
        if (wallDistExists)
            c.wallDist[n] = (c.wallDist[8] + c.wallDist[13]) / 2.; // overhead, TODO
    }

    c.dataLoaded = true;
}

void Unstructured::computeBoundingBox()
{
    boundingBoxMin[0] = 1e19f;
    boundingBoxMin[1] = 1e19f;
    boundingBoxMin[2] = 1e19f;
    boundingBoxMax[0] = -1e19f;
    boundingBoxMax[1] = -1e19f;
    boundingBoxMax[2] = -1e19f;

    for (int i = 0; i < nNodes; i++)
    {
        vec3 xyz;
        getCoords(i, xyz); // TODO: no support for rotating zones
        if (xyz[0] < boundingBoxMin[0])
            boundingBoxMin[0] = xyz[0];
        if (xyz[1] < boundingBoxMin[1])
            boundingBoxMin[1] = xyz[1];
        if (xyz[2] < boundingBoxMin[2])
            boundingBoxMin[2] = xyz[2];
        if (xyz[0] > boundingBoxMax[0])
            boundingBoxMax[0] = xyz[0];
        if (xyz[1] > boundingBoxMax[1])
            boundingBoxMax[1] = xyz[1];
        if (xyz[2] > boundingBoxMax[2])
            boundingBoxMax[2] = xyz[2];
    }
}

void Unstructured::setupSearchGrid()
{
    fvec3 minSize;
    computeMinBoxSize(minSize);

    searchGrid = new SearchGrid;
    fvec3copy(boundingBoxMin, searchGrid->min);
    fvec3copy(boundingBoxMax, searchGrid->max);

    // slightly enlarge the search grid (give tolerance)
    double h = fvec3dist(searchGrid->min, searchGrid->max) / 1e6;
    searchGrid->min[0] -= h;
    searchGrid->min[1] -= h;
    searchGrid->min[2] -= h;
    searchGrid->max[0] += h;
    searchGrid->max[1] += h;
    searchGrid->max[2] += h;

    fvec3 ext;
    fvec3sub(searchGrid->max, searchGrid->min, ext);
    int nx = (int)ceil(ext[0] / minSize[0]);
    int ny = (int)ceil(ext[1] / minSize[1]);
    int nz = (int)ceil(ext[2] / minSize[2]);
    int nBoxes = nx * ny * nz;
    if (nBoxes > nCells / 3)
    {
        float factor = nCells / 3. / (float)nBoxes;
        factor = pow((double)factor, .33333);
        nx = (int)ceil(factor * nx);
        ny = (int)ceil(factor * ny);
        nz = (int)ceil(factor * nz);
        // int nBoxes = nx * ny * nz; ### replaced, sadlo 2006-07-16
        nBoxes = nx * ny * nz;
    }

    searchGrid->dim[0] = nx;
    searchGrid->dim[1] = ny;
    searchGrid->dim[2] = nz;

#if DEBUG_OUTPUT
    printf("%dx%dx%d search grid (%d boxes)\n", nx, ny, nz, nBoxes);
#endif

    searchGrid->box = new vector<int>[nBoxes];

    searchGrid->step[0] = ext[0] / (float)searchGrid->dim[0];
    searchGrid->step[1] = ext[1] / (float)searchGrid->dim[1];
    searchGrid->step[2] = ext[2] / (float)searchGrid->dim[2];

    // Put cell centers in boxes
    for (int i = 0; i < nCells; i++)
    {
        vec3 v;
        //fvec3tovec3(cellCentroid[i], v);
        // time = 0.0 means zone is not rotated
        getCellCentroid(i, v, 0.0);
        int b = searchGrid->findBox(v);
        searchGrid->box[b].push_back(i);
    }

#ifdef DEBUGGING
    for (int i = 0; i < nBoxes; i++)
    {
        printf("%4d: %6d\n", i, searchGrid->box[i].size());
    }
#endif
}

int SearchGrid::findBox(vec3 xyz) // returns box index, -1 if outside
{
    if (xyz[0] < min[0])
        return -1;
    if (xyz[0] > max[0])
        return -1;
    if (xyz[1] < min[1])
        return -1;
    if (xyz[1] > max[1])
        return -1;
    if (xyz[2] < min[2])
        return -1;
    if (xyz[2] > max[2])
        return -1;

    int nx = (int)floor((xyz[0] - min[0]) / step[0]);
    int ny = (int)floor((xyz[1] - min[1]) / step[1]);
    int nz = (int)floor((xyz[2] - min[2]) / step[2]);

    if (nx >= dim[0])
        nx = dim[0] - 1;
    if (ny >= dim[1])
        ny = dim[1] - 1;
    if (nz >= dim[2])
        nz = dim[2] - 1;

    int index = nx + dim[0] * (ny + dim[1] * nz);

    return index;
}

int SearchGrid::findBoxes(vec3 xyz, int *boxes) // returns number of boxes
{
    if (xyz[0] < min[0])
        return 0;
    if (xyz[0] > max[0])
        return 0;
    if (xyz[1] < min[1])
        return 0;
    if (xyz[1] > max[1])
        return 0;
    if (xyz[2] < min[2])
        return 0;
    if (xyz[2] > max[2])
        return 0;

    float x = (xyz[0] - min[0]) / step[0];
    float y = (xyz[1] - min[1]) / step[1];
    float z = (xyz[2] - min[2]) / step[2];

    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int z0 = (int)floor(z);

    if (x0 >= dim[0])
        x0 = dim[0] - 1;
    if (y0 >= dim[1])
        y0 = dim[1] - 1;
    if (z0 >= dim[2])
        z0 = dim[2] - 1;

    int ct = 0;
    int index = x0 + dim[0] * (y0 + dim[1] * z0);
    boxes[ct++] = index;

    double xf = x - x0;
    double yf = y - y0;
    double zf = z - z0;

    int dx = xf < .5 ? (x0 > 0 ? -1 : 0) : (x0 < dim[0] - 1 ? 1 : 0);
    int dy = yf < .5 ? (y0 > 0 ? -dim[0] : 0) : (y0 < dim[1] - 1 ? dim[0] : 0);
    int dz = zf < .5 ? (z0 > 0 ? -dim[0] * dim[1] : 0) : (z0 < dim[2] - 1 ? dim[0] * dim[1] : 0);

    if (dx)
    {
        boxes[ct++] = index + dx;
        if (dy)
        {
            boxes[ct++] = index + dx + dy;
            if (dz)
                boxes[ct++] = index + dx + dy + dz;
        }
        if (dz)
            boxes[ct++] = index + dx + dz;
    }
    if (dy)
    {
        boxes[ct++] = index + dy;
        if (dz)
            boxes[ct++] = index + dy + dz;
    }
    if (dz)
        boxes[ct++] = index + dz;

    return ct;
}
#if 0 // TODO DELETEME
bool Unstructured::findCell(vec3 xyz, double time)
{
  
  if (vector3CB) {
    vec3copy(xyz, vector3CBPosition);
  }
  
  if (!searchGrid) setupSearchGrid();
  
  if (currCell.computeWeights(xyz, weight)) {
    ct1++;
    assureCellData(time); // ### 2006-11-16
    return true;			// found in current cell
  }
  // TODO: Try first with a ring buffer of old cells

  assureCellData(time); // ### 2006-11-16

  int boxes[8];
  int n = searchGrid->findBoxes(xyz, boxes);
        
  fvec3 xyzf;
  vec3tofvec3(xyz, xyzf);

  for (int r = 0; r < n; r++) {
    int b = boxes[r];

    vector<int>& box = searchGrid->box[b];

    for (int i = 0; i < (signed) box.size(); i++) {
      int cell = box[i];
      double f = fvec3distSqr(cellCentroid[cell], xyzf);
      double r = cellRadiusSqr[cell];
      if (f > r) continue; // trivial reject

      //printf("point(%g, %g, %g) is in circumsphere of cell %d\n", xyz[0], xyz[1], xyz[2], cell);

      CellInfo& c = loadCell(cell, true, time);
      if (c.computeWeights(xyz, weight)) {
        ct2++;
        return true;
      }
    }
  }
  ct0++;
  if (!vector3CB) {
    return false;
  }
  else {
    if (vector3CBRestrictToGrid) return false;
    else return true; // ##### TODO: this is dangerous!
  }
}
#else // allows rotating zones
// the rotating zone must not translate and must be circular
// (to allow a pointsearch with the unrotated zone for the determination of the
//  zone the point is in)
bool Unstructured::findCell(vec3 xyz, double time)
{
    if (vector3CB)
    {
        vec3copy(xyz, vector3CBPosition);
    }

    if (!searchGrid)
        setupSearchGrid();

    if (currCell.computeWeights(xyz, weight))
    {
        // cell is either inside not rotating zone or inside rotating zone, but
        // then the cell is already loaded in rotated position
        ct1++;
        assureCellData(time); // ### 2006-11-16
        return true; // found in current cell
    }
    // TODO: Try first with a ring buffer of old cells

    assureCellData(time); // ### 2006-11-16

    // search for uncompensated position
    {
        bool found = false;
        int boxes[8];
        int n = searchGrid->findBoxes(xyz, boxes);

        fvec3 xyzf;
        vec3tofvec3(xyz, xyzf);

        for (int r = 0; r < n; r++)
        {
            int b = boxes[r];

            vector<int> &box = searchGrid->box[b];

            for (int i = 0; i < (signed)box.size(); i++)
            {
                int cell = box[i];
                //double f = fvec3distSqr(cellCentroid[cell], xyzf);
                fvec3 cen;
                if (transientZoneRotating < 0)
                {
                    getCellCentroid(cell, cen, time);
                }
                else
                {
                    getCellCentroid(cell, cen, 0.0); // time = 0.0 : uncompensated centroid
                }
                double f = fvec3distSqr(cen, xyzf);
                double r = cellRadiusSqr[cell];
                if (f > r)
                    continue; // trivial reject

                //printf("point(%g, %g, %g) is in circumsphere of cell %d\n", xyz[0], xyz[1], xyz[2], cell);

                //CellInfo& c = loadCell(cell, true, time);
                CellInfo &c = loadCell(cell, true, time);
                if (transientZoneRotating < 0)
                {
                    c = loadCell(cell, true, time);
                }
                else
                {
                    c = loadCell(cell, true, 0.0); // time = 0.0 : uncompensated
                }
                if (c.computeWeights(xyz, weight))
                {
                    ct2++;
                    //return true;
                    found = true;
                    goto Out;
                }
            }
        }
        ct0++;
        if (!vector3CB)
        {
            //return false;
        }
        else
        {
            if (vector3CBRestrictToGrid)
            {
                //return false;
                found = false;
            }
            else
            {
                //return true; // ##### TODO: this is dangerous!
                found = true;
            }
        }
    Out:

        if (transientZoneRotating < 0)
        {
            // no rotating zone -> everything ok
            return found;
        }
        else
        {
            if (found)
            {
                int cellZone = getCellZone(getCellIndex(), time);
                if (cellZone == -1)
                {
                    printf("Unstructured: inconsistent cell zone\n");
                    return false; // ###
                }
                else if (cellZone != transientZoneRotating)
                {
                    // inside not rotating zone -> everything ok
                    return found;
                }
                else
                {
                    // inside rotating zone
                }
            }
            else
            {
                // #### this is not ok, since we could have hit a hole in the
                // rotating domain and would still need to search at the compensated
                // position TODO TODO TODO
                return false;
            }
        }
    }

    // inside rotating zone

    // compensate rotation
    vec3 xyzc;
    rotateWithTransientZone(xyz, -time, xyzc);

    // search

    int boxes[8];
    int n = searchGrid->findBoxes(xyzc, boxes);

    fvec3 xyzcf;
    vec3tofvec3(xyzc, xyzcf);

    for (int r = 0; r < n; r++)
    {
        int b = boxes[r];

        vector<int> &box = searchGrid->box[b];

        for (int i = 0; i < (signed)box.size(); i++)
        {
            int cell = box[i];
            //double f = fvec3distSqr(cellCentroid[cell], xyzcf);
            fvec3 cen;
            getCellCentroid(cell, cen, 0.0); // compensated centroid
            double f = fvec3distSqr(cen, xyzcf);
            double r = cellRadiusSqr[cell];
            if (f > r)
                continue; // trivial reject

            //printf("point(%g, %g, %g) is in circumsphere of cell %d\n", xyz[0], xyz[1], xyz[2], cell);

            CellInfo &c = loadCell(cell, true, time);
            if (c.computeWeights(xyz, weight))
            { // not compensated pos
                ct2++;
                int cellZone = getCellZone(getCellIndex(), time);
                if (cellZone != transientZoneRotating)
                {
                    printf("Unstructured: error: wrong cell zone: %d instead of %d at time %g\n",
                           cellZone, transientZoneRotating, time);
                    return false; // ###
                }
                else
                {
                    return true;
                }
            }
        }
    }

    ct0++;
    if (!vector3CB)
    {
        return false;
    }
    else
    {
        if (vector3CBRestrictToGrid)
            return false;
        else
            return true; // ##### TODO: this is dangerous!
    }
}
#endif

// ################## untested !!!
// #### not aware of rotating zones
std::vector<int> Unstructured::findCells(vec3 bboxMin, vec3 bboxMax)
{ // returns vector of cells in bounding box
    // ########## actually no rejection of cells that are at boundary?

    vector<int> cells;

    if (vector3CB)
    {
        printf("Unstructured::findCells: vector3CB not yet supported\n");

        return cells;
    }

    if (!searchGrid)
        setupSearchGrid();

    int boxesMin[8];
    int nMin = searchGrid->findBoxes(bboxMin, boxesMin);
    if (nMin == 0)
        return cells;

    int boxesMax[8];
    int nMax = searchGrid->findBoxes(bboxMax, boxesMax);
    if (nMax == 0)
        return cells;

    // get minimum indices
    int minI = INT_MAX;
    int minJ = INT_MAX;
    int minK = INT_MAX;
    {
        for (int r = 0; r < nMin; r++)
        {
            int idx = boxesMin[r];

            int k = idx / (searchGrid->dim[0] * searchGrid->dim[1]);
            int j = (idx - k * searchGrid->dim[0] * searchGrid->dim[1]) / searchGrid->dim[0];
            int i = idx - k * searchGrid->dim[0] * searchGrid->dim[1] - j * searchGrid->dim[0];

            if (i < minI)
                minI = i;
            if (j < minJ)
                minJ = j;
            if (k < minK)
                minK = k;
        }
    }

    // get maximum indices
    int maxI = -INT_MAX;
    int maxJ = -INT_MAX;
    int maxK = -INT_MAX;
    {
        for (int r = 0; r < nMax; r++)
        {
            int idx = boxesMax[r];

            int k = idx / (searchGrid->dim[0] * searchGrid->dim[1]);
            int j = (idx - k * searchGrid->dim[0] * searchGrid->dim[1]) / searchGrid->dim[0];
            int i = idx - k * searchGrid->dim[0] * searchGrid->dim[1] - j * searchGrid->dim[0];

            if (i > maxI)
                maxI = i;
            if (j > maxJ)
                maxJ = j;
            if (k > maxK)
                maxK = k;
        }
    }

    // collect cells
    // ########## actually no rejection of cells that are at boundary?
    {
        std::map<int, bool> cellsMap;
        for (int k = minK; k < maxK + 1; k++)
        {
            for (int j = minJ; j < maxJ + 1; j++)
            {
                for (int i = minI; i < maxI + 1; i++)
                {

                    int index = i + searchGrid->dim[0] * (j + searchGrid->dim[1] * k);

                    vector<int> &box = searchGrid->box[index];

                    for (int c = 0; c < (int)box.size(); c++)
                    {

                        int cell = box[c];

                        cellsMap[cell] = true;
                    }
                }
            }
        }

        map<int, bool>::iterator it = cellsMap.begin();
        while (it != cellsMap.end())
        {
            cells.push_back((*it).first);
            it++;
        }

        return cells;
    }
}

std::vector<int> Unstructured::findCellsInZ(vec3 xyz)
{ // returns vector of cells in bounding box
    // ########## actually no rejection of cells that are at boundary?

    vector<int> cells;

    if (vector3CB)
    {
        printf("Unstructured::findCells: vector3CB not yet supported\n");

        return cells;
    }

    if (!searchGrid)
        setupSearchGrid();

    int boxes[8];
    int n = searchGrid->findBoxes(xyz, boxes);
    if (n == 0)
        return cells;

    // collect cells
    // ########## actually no rejection of cells that are at boundary?
    {
        std::map<int, bool> cellsMap;
        for (int r = 0; r < n; r++)
        {

            int baseIndex = boxes[r] - (boxes[r] / (searchGrid->dim[0] * searchGrid->dim[1])) * searchGrid->dim[0] * searchGrid->dim[1];

            for (int zi = 0; zi < searchGrid->dim[2]; zi++)
            {
                int index = baseIndex + zi * searchGrid->dim[0] * searchGrid->dim[1];

                vector<int> &box = searchGrid->box[index];

                for (int c = 0; c < (int)box.size(); c++)
                {

                    int cell = box[c];

                    cellsMap[cell] = true;
                }
            }
        }
        map<int, bool>::iterator it = cellsMap.begin();
        while (it != cellsMap.end())
        {
            cells.push_back((*it).first);
            it++;
        }

        return cells;
    }
}

// TODO: Test alternative: compute normal vector to plane fitted through wall vertices,
// 	                       compute max (positive) offset, and frac of it, add to xyz
void Unstructured::keepOffWall(vec3 xyz, double frac)
{
    if (!wallDistExists)
    {
        fprintf(stderr, "Unstructured::keepOffWall: warning: no wall distance available\n");
        return;
    }

    CellInfo &c = currCell;
    double sum = 0;
    for (int k = 0; k < nVertices[c.type]; k++)
    {
        if (c.wallDist[k] == 0)
            sum += weight[k];
    }
    if (sum < 1 - frac)
        return; // all fine

    // Apply correction
    vec3zero(xyz);
    for (int k = 0; k < nVertices[c.type]; k++)
    {
        double wt = weight[k];

        if (c.wallDist[k] == 0)
            wt *= (1 - frac) / sum;
        else
            wt *= frac / (1 - sum);

        vec3 v;
        vec3scal(c.vertex[k], wt, v);
        vec3add(xyz, v, xyz);
    }
}

#if 0 // moved to header
double Unstructured::interpolateScalar()
{
#if CHECK_VARIABLES
  if (!scalarExists) {
    fprintf(stderr, "Unstructured::interpolateScalar: no scalar\n");
    return 0.0;
  }
#endif

  // no call to loadCell or findCell, -> assure data
  assureCellData();

  CellInfo& c = currCell;
  double s = 0;
  for (int k = 0; k < nVertices[c.type]; k++) {
    s += c.scal[k] * weight[k];
  }
  return s;
}
#endif

void Unstructured::interpolateVector3(vec3 vec, double time, bool orientate, vec3 direction)
{
#if CHECK_VARIABLES
    if (!vector3CB && !vectorExists)
    {
        fprintf(stderr, "Unstructured::interpolateVector: no vector\n");
        return;
    }
#endif

    if (vector3CB)
    {
        //vector3CB(vector3CBPosition, vec);
        vector3CB(vector3CBPosition, vec, time);
        return;
    }

    // no call to loadCell or findCell, -> assure cell data
    assureCellData(time);

    CellInfo &c = currCell;
    vec3zero(vec);
    if (orientate)
    {
        for (int k = 0; k < nVertices[c.type]; k++)
        {
            vec3 v;
            vec3 orientated;
            if (vec3dot(c.vec[k], direction) < 0)
                vec3scal(c.vec[k], -1.0, orientated);
            else
                vec3copy(c.vec[k], orientated);
            vec3scal(orientated, weight[k], v);
            vec3add(vec, v, vec);
        }
    }
    else
    {
        for (int k = 0; k < nVertices[c.type]; k++)
        {
            vec3 v;
            vec3scal(c.vec[k], weight[k], v);
            vec3add(vec, v, vec);
        }
    }
}

void Unstructured::interpolateMatrix3(mat3 mat)
{
#if CHECK_VARIABLES
    if (!vectorExists)
    {
        fprintf(stderr, "Unstructured::interpolateVector: no vector\n");
        return;
    }
#endif

    // no call to loadCell or findCell, -> assure cell data
    assureCellData();

    CellInfo &c = currCell;
    mat3zero(mat);
    for (int k = 0; k < nVertices[c.type]; k++)
    {

        // TODO: instead of getting matrices this way, should incorporate
        //       matrices in loadCellData etc.
        fmat3 nmatf;
        getMatrix3(c.node[k], nmatf);
        mat3 nmat;
        fmat3tomat3(nmatf, nmat);

        mat3 m;
        mat3scal(nmat, weight[k], m);
        mat3add(mat, m, mat);
    }
}

#if 0
void Unstructured::interpolatedMatrixEigenvector3(int evIdx, vec3 ev, vec3 direction)
{ // evIdx: 0 for major, 1 for intermediate, 2 for minor eigenvector
  // direction: for orientation of eigenvector, may be NULL

  mat3 m;
  interpolateMatrix3(m);

  vec3 eigenvalues;
  bool allReal = (mat3eigenvalues(m, eigenvalues) == 3);
  if (!allReal) {
    printf("Unstructured::interpolatedMatrixEigenvector: complex eigenvalues (real1=%g real2=%g imag=%g), returning zero\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
    mat3dump(m, stdout);
    vec3set(ev, 0, 0, 0);
    return;
  }

  // sort eigenvalues in descending order
  vec3sortd(eigenvalues, eigenvalues);
  
  if (!mat3realEigenvector(m, eigenvalues[evIdx], ev)) {
    printf("Unstructured::interpolatedMatrixEigenvector: error computing eigenvector\n");
    vec3set(ev, 0, 0, 0);
    return;
  }

  //if (fabs(eigenvalues[evIdx]) < 1e-6) {
  //if (m[2][0] == 0.0 && m[2][1] == 0.0 && m[2][2] == 0.0 && ev[0] == 0.0 && ev[1] == 0.0) {
  //  printf("interpolatedMatrixEigenvector3: eigenvalue=%g eigenvector=%g,%g,%g\n", eigenvalues[evIdx], ev[0], ev[1], ev[2]);
  //  mat3dump(m, stdout);
  //}

  // handle "2D" matrices
  if (m[2][0] == 0.0 && m[2][1] == 0.0 && m[2][2] == 0.0 && ev[0] == 0.0 && ev[1] == 0.0 && (evIdx == 0 || evIdx == 2)) {
    evIdx = 1; // ### OK?
    if (!mat3realEigenvector(m, eigenvalues[evIdx], ev)) {
      printf("Unstructured::interpolatedMatrixEigenvector: error computing eigenvector 2\n");
      vec3set(ev, 0, 0, 0);
      return;
    }
    printf("interpolatedMatrixEigenvector3: eigenvalues=%g,%g,%g eigenvector=%g,%g,%g\n", eigenvalues[0], eigenvalues[1], eigenvalues[2], ev[0], ev[1], ev[2]);
    mat3dump(m, stdout);
  }

  vec3nrm(ev, ev);

  if (direction && vec3dot(direction, ev) < 0) {
    vec3scal(ev, -1.0, ev);
  }
}
#else
void Unstructured::interpolatedMatrixEigenvector3(int evIdx, vec3 ev, vec3 direction, bool absoluteSorted)
{ // evIdx: 0 for major, 1 for intermediate, 2 for minor eigenvector
    // direction: for orientation of eigenvector, may be NULL

    // ########### this seems to assume symmetric matrices!!!

    mat3 m;
    interpolateMatrix3(m);

    vec3 eigenvalues;
    bool allReal = (mat3eigenvalues(m, eigenvalues) == 3);
    if (!allReal)
    {
        printf("Unstructured::interpolatedMatrixEigenvector: complex eigenvalues (real1=%g real2=%g imag=%g), returning zero\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
        mat3dump(m, stdout);
        vec3set(ev, 0, 0, 0);
        return; // ### TODO: support complex eigenvalues
    }

    // sort eigenvalues in descending order
    if (absoluteSorted)
        vec3sortdAbs(eigenvalues, eigenvalues);
    else
        vec3sortd(eigenvalues, eigenvalues);

    //vec3 ev1;
    //if (!mat3realEigenvector(m, eigenvalues[evIdx], ev1)) {
    //  printf("Unstructured::interpolatedMatrixEigenvector: error computing eigenvector\n");
    //  vec3set(ev, 0, 0, 0);
    //  return;
    //}

    if (!mat3realOrthogonalEigenvector(m, eigenvalues, evIdx, ev))
    {
        printf("Unstructured::interpolatedMatrixEigenvector: error computing eigenvector for eigenvalue=%g (eigenvalues=%g,%g,%g)\n",
               eigenvalues[evIdx], eigenvalues[0], eigenvalues[1], eigenvalues[2]);
        vec3set(ev, 0, 0, 0);
        return;
    }

    if (m[2][0] == 0.0 && m[2][1] == 0.0 && m[2][2] == 0.0 && ev[0] == 0.0 && ev[1] == 0 && (evIdx == 0 || evIdx == 2))
    {
        //printf("interpolatedMatrixEigenvector3: old: eigenvalues=%g,%g,%g eigenvector=%g,%g,%g\n", eigenvalues[0], eigenvalues[1], eigenvalues[2], ev1[0], ev1[1], ev1[2]);
        //printf("interpolatedMatrixEigenvector3: new: eigenvector=%g,%g,%g\n", ev[0], ev[1], ev[2]);

        //printf("interpolatedMatrixEigenvector3: eigenvalues=%g,%g,%g eigenvector=%g,%g,%g\n", eigenvalues[0], eigenvalues[1], eigenvalues[2], ev[0], ev[1], ev[2]);
        //mat3dump(m, stdout);

        if (!mat3realOrthogonalEigenvector(m, eigenvalues, 1, ev))
        {
            printf("Unstructured::interpolatedMatrixEigenvector: error computing eigenvector2\n");
            vec3set(ev, 0, 0, 0);
            return;
        }
        //printf("interpolatedMatrixEigenvector3: new: eigenvector=%g,%g,%g\n", ev[0], ev[1], ev[2]);
    }

    vec3nrm(ev, ev);

    if (direction && vec3dot(direction, ev) < 0)
    {
        vec3scal(ev, -1.0, ev);
    }
}
#endif

bool Unstructured::interpolatedMatrixEigenvaluesDesc3(vec3 evals)
{
    // ########### this seems to assume symmetric matrices!!!

    mat3 m;
    interpolateMatrix3(m);

    vec3 eigenvalues;
    bool allReal = (mat3eigenvalues(m, eigenvalues) == 3);
    if (!allReal)
    {
        printf("Unstructured::interpolatedMatrixEigenvaluesDesc3: complex eigenvalues (real1=%g real2=%g imag=%g), returning zero\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
        mat3dump(m, stdout);
        vec3set(evals, 0, 0, 0);
        return false; // ### TODO: support complex eigenvalues
    }

    // sort eigenvalues in descending order
    vec3sortd(eigenvalues, evals);

    return true;
}

double Unstructured::interpolateWallDist()
{
    CellInfo &c = currCell;
    double s = 0;
    for (int k = 0; k < nVertices[c.type]; k++)
    {
        s += c.wallDist[k] * weight[k];
    }
    return s;
}

// NOTE: this does not use LS but uses linear field in current tetrahedron
void Unstructured::interpolateVectorGradient(mat3 grad)
{
    CellInfo &c = currCell;
    mat3zero(grad);

    // no call to loadCell or findCell, -> assure data
    assureCellData();

//#define STANDARD
#ifdef STANDARD
    for (int k = 0; k < nVertices[c.type]; k++)
    {
        mat3 m;
        fmat3tomat3(vGradient[c.node[k]], m);
        mat3scal(m, weight[k], m);
        mat3add(grad, m, grad);
    }
#else
    vec3 wt[8];
    c.computeGradientWeights(wt);
    for (int i = 0; i < nVertices[c.type]; i++)
    {
        mat3 m;
        vec3scal(wt[i], c.vec[i][0], m[0]);
        vec3scal(wt[i], c.vec[i][1], m[1]);
        vec3scal(wt[i], c.vec[i][2], m[2]);
        mat3add(grad, m, grad);

#ifdef DEBUGGING // ### tmp sadlof 2006-06-14
        printf("[%d]grad:\n", i);
        mat3dump(grad, stdout);
#endif
    }
#endif
}

#if 1
double Unstructured::integrate(vec3 x, bool forward, double maxTime, double maxDist, int maxSteps, int order, bool keepOffWal, double startTime, double *distDone, bool tensorLine, vec3 direction, bool interpolateTensor, int evIdx)
{
    // if transient=true pathlines are integrated instead of streamlines

    // TODO: transient support for tensor lines

    //order = 1; // HACK

    if (!vector3CB && !vectorExists)
    {
        fprintf(stderr, "Unstructured::integrate: no velocity\n");
        if (distDone)
            *distDone = 0.0;
        return 0.0; // ###
    }

    double timeLeft = maxTime;
    double distLeft = maxDist;

    int i;
    double time = startTime; // only for transient mode
    for (i = 0; i < maxSteps; i++)
    {
        //printf("unstrucutred integrate: time=%g\n", time);
        if (!findCell(x, time))
        {
            if (distDone)
                *distDone = maxDist - distLeft;
            return maxTime - timeLeft;
        }

        double maxStep = sqrt(currCell.radiusSqr) / 10; // Heuristic
        if (maxStep > distLeft)
            maxStep = distLeft;

        vec3 v;
        if (tensorLine && interpolateTensor)
            interpolatedMatrixEigenvector3(evIdx, v, direction);
        else
            interpolateVector3(v, time, tensorLine, direction);
        double speed = vec3mag(v);
        if (speed == 0)
            speed = 1e-9; // TODO: relative to typicalSpeed

        double dt = maxStep / speed;
        if (dt > timeLeft)
            dt = timeLeft;

        bool valid = false;
        double h = forward ? dt : -dt;

        for (int j = 0; j < 10; j++)
        { // Adaptation levels
            vec3 y;

            if (order == 1)
            {
                vec3 k1;

                vec3scal(v, h, k1);
                vec3add(x, k1, y);
            }
            else if (order == 2)
            {
            }
            else if (order == 4)
            {
                vec3 k1half, k2, k2half, k3, k4half, sum;

                vec3scal(v, h / 2., k1half);

                // @ Ronny: please explain to Filip and maybe document more e.g. 15deg

                vec3add(x, k1half, y);
                if (!findCell(y, time + h / 2.0))
                {
                    h /= 2;
                    continue;
                } // TODO Ronny: is time ok?
                if (tensorLine && interpolateTensor)
                    interpolatedMatrixEigenvector3(evIdx, v, direction);
                else
                    interpolateVector3(v, time + h / 2.0, tensorLine, direction); // TODO Ronny: is time ok below too?
                vec3scal(v, h, k2);
                vec3scal(k2, .5, k2half);

                vec3add(x, k2half, y);
                if (!findCell(y, time + h / 2.0))
                {
                    h /= 2;
                    continue;
                }
                if (tensorLine && interpolateTensor)
                    interpolatedMatrixEigenvector3(evIdx, v, direction);
                else
                    interpolateVector3(v, time + h / 2.0, tensorLine, direction);
                vec3scal(v, h, k3);

                vec3add(x, k3, y);
                if (!findCell(y, time + h))
                {
                    h /= 2;
                    continue;
                }
                if (tensorLine && interpolateTensor)
                    interpolatedMatrixEigenvector3(evIdx, v, direction);
                else
                    interpolateVector3(v, time + h, tensorLine, direction);
                vec3scal(v, h / 2., k4half);

                // Check if angle between k1half and k4half is more than 15 degrees
                double dot = vec3dot(k1half, k4half);
                double cos2 = sqr(dot) / (vec3sqr(k1half) * vec3sqr(k4half));
                if (dot < 0 || cos2 < .933)
                {
                    h /= 2;
                    continue;
                }

                vec3add(k1half, k2, sum);
                vec3add(sum, k3, sum);
                vec3add(sum, k4half, sum);
                vec3scal(sum, 1. / 3., sum);

                vec3add(x, sum, y);
            }

            // Point within grid?
            if (!findCell(y, time + h))
            {
                h /= 2;
                continue;
            }

            // Need shortening?
            double ds = vec3dist(x, y);
            if (ds > 1.01 * distLeft)
            {
                h *= distLeft / ds;
                continue; // Try again
            }
            if (ds > 1.5 * maxStep)
            {
                h *= maxStep / ds;
                continue; // Try again
            }

            // Accept the step
            //if (keepOffWal) keepOffWall(y, .5);	// TRYING
            if (keepOffWal)
                keepOffWall(y, .05);
            if (tensorLine)
            {
                if (forward)
                    vec3sub(y, x, direction);
                else
                    vec3sub(x, y, direction);
            }
            vec3copy(y, x);
            timeLeft -= forward ? h : -h;
            //time += h;
            time = startTime + (forward ? 1.0 : -1.0) * (maxTime - timeLeft);
            distLeft -= ds;
            valid = true;

            break;
        }
        if (!valid)
            break;

        // Are we done?
        if (timeLeft == 0)
            break;
        if (distLeft < sqrt(currCell.radiusSqr) / 1000)
            break;
    }
    //	printf("%12.8f %12.8f %12.8f (%3d %12.8f %12.8f)\n",
    //			x[0], x[1], x[2], i, maxTime-timeLeft, maxDist-distLeft);

    findCell(x, time); // Set current position
    if (distDone)
        *distDone = maxDist - distLeft;
    return maxTime - timeLeft;
}
#else // DELETEME: this was first approach, callback is now set using
// setVector3CB(), and "everything" else is transparent
double Unstructured::integrate(vec3 x, bool forward, double maxTime, double maxDist, int maxSteps, int order, bool keepOffWal, double startTime, double *distDone, void (*veloCB)(vec3 pos, double time, vec3 out), double veloCB_maxStep)
{
    // if transient=true pathlines are integrated instead of streamlines
    // veloCB: if not NULL, velocity is taken from this callback instead

    //order = 1; // HACK

    if (!veloCB && !vectorExists)
    {
        fprintf(stderr, "Unstructured::integrate: no velocity\n");
        if (distDone)
            *distDone = 0.0;
        return 0.0; // ###
    }

    double timeLeft = maxTime;
    double distLeft = maxDist;

    int i;
    double time = startTime; // only for transient mode
    for (i = 0; i < maxSteps; i++)
    {
        //printf("unstrucutred integrate: time=%g\n", time);
        if (!veloCB && !findCell(x, time))
        {
            if (distDone)
                *distDone = maxDist - distLeft;
            return maxTime - timeLeft;
        }

        double maxStep;
        if (!veloCB)
            maxStep = sqrt(currCell.radiusSqr) / 10; // Heuristic
        else
            maxStep = veloCB_maxStep;
        if (maxStep > distLeft)
            maxStep = distLeft;

        vec3 v;
        if (!veloCB)
            interpolateVector3(v, time);
        else
            veloCB(x, time, v);
        double speed = vec3mag(v);
        if (speed == 0)
            speed = 1e-9; // TODO: relative to typicalSpeed

        double dt = maxStep / speed;
        if (dt > timeLeft)
            dt = timeLeft;

        bool valid = false;
        double h = forward ? dt : -dt;

        for (int j = 0; j < 10; j++)
        { // Adaptation levels
            vec3 y;

            if (order == 1)
            {
                vec3 k1;

                vec3scal(v, h, k1);
                vec3add(x, k1, y);
            }
            else if (order == 2)
            {
            }
            else if (order == 4)
            {
                vec3 k1half, k2, k2half, k3, k4half, sum;

                vec3scal(v, h / 2., k1half);

                // @ Ronny: please explain to Filip and maybe document more e.g. 15deg

                vec3add(x, k1half, y);
                if (!veloCB)
                {
                    if (!findCell(y, time + h / 2.0))
                    {
                        h /= 2;
                        continue;
                    } // TODO Ronny: is time ok?
                    interpolateVector3(v, time + h / 2.0); // TODO Ronny: is time ok below too?
                }
                else
                {
                    veloCB(y, time + h / 2.0, v);
                }
                vec3scal(v, h, k2);
                vec3scal(k2, .5, k2half);

                vec3add(x, k2half, y);
                if (!veloCB)
                {
                    if (!findCell(y, time + h / 2.0))
                    {
                        h /= 2;
                        continue;
                    }
                    interpolateVector3(v, time + h / 2.0);
                }
                else
                {
                    veloCB(y, time + h / 2.0, v);
                }
                vec3scal(v, h, k3);

                vec3add(x, k3, y);
                if (!veloCB)
                {
                    if (!findCell(y, time + h))
                    {
                        h /= 2;
                        continue;
                    }
                    interpolateVector3(v, time + h);
                }
                else
                {
                    veloCB(y, time + h, v);
                }
                vec3scal(v, h / 2., k4half);

                // Check if angle between k1half and k4half is more than 15 degrees
                double dot = vec3dot(k1half, k4half);
                double cos2 = sqr(dot) / (vec3sqr(k1half) * vec3sqr(k4half));
                if (dot < 0 || cos2 < .933)
                {
                    h /= 2;
                    continue;
                }

                vec3add(k1half, k2, sum);
                vec3add(sum, k3, sum);
                vec3add(sum, k4half, sum);
                vec3scal(sum, 1. / 3., sum);

                vec3add(x, sum, y);
            }

            // Point within grid?
            if (!veloCB && !findCell(y, time + h))
            {
                h /= 2;
                continue;
            }

            // Need shortening?
            double ds = vec3dist(x, y);
            if (ds > 1.01 * distLeft)
            {
                h *= distLeft / ds;
                continue; // Try again
            }
            if (ds > 1.5 * maxStep)
            {
                h *= maxStep / ds;
                continue; // Try again
            }

            // Accept the step
            //if (keepOffWal) keepOffWall(y, .5);	// TRYING
            if (!veloCB && keepOffWal)
                keepOffWall(y, .05);
            vec3copy(y, x);
            timeLeft -= forward ? h : -h;
            //time += h;
            time = startTime + (forward ? 1.0 : -1.0) * (maxTime - timeLeft);
            distLeft -= ds;
            valid = true;

            break;
        }
        if (!valid)
            break;

        // Are we done?
        if (timeLeft == 0)
            break;
        if (!veloCB && distLeft < sqrt(currCell.radiusSqr) / 1000)
            break;
    }
    //	printf("%12.8f %12.8f %12.8f (%3d %12.8f %12.8f)\n",
    //			x[0], x[1], x[2], i, maxTime-timeLeft, maxDist-distLeft);

    if (!veloCB)
        findCell(x, time); // Set current position
    if (distDone)
        *distDone = maxDist - distLeft;
    return maxTime - timeLeft;
}
#endif

bool CellInfo::computeWeights(vec3 xyz, double wt[8])
{
    for (int i = 0; i < 8; i++)
        wt[i] = 0;

    for (int j = 0; j < nTets[type]; j++)
    {
        int *t = tets[type][j];
        vec4 bary;
        if (insideTet(xyz, vertex[t[0]], vertex[t[1]], vertex[t[2]], vertex[t[3]], bary))
        {
            for (int k = 0; k < 4; k++)
            {
                if (t[k] < nVertices[type])
                { // regular vertex
                    int vert = t[k];
                    wt[vert] += bary[k];
                }
                else if (k == 0)
                { // centroid
                    for (int vert = 0; vert < nVertices[type]; vert++)
                    {
                        wt[vert] += bary[k] / (float)nVertices[type];
                    }
                }
                else /* if (k ==1) */
                { // face centroid
                    int face = t[k] - nVertices[type];
                    for (int l = 0; l < 4; l++)
                    {
                        int vert = faces[type][face][l];
                        wt[vert] += bary[k] * .25;
                    }
                }
            }
            tet = j; // store current tet in CellInfo
            return true;
        }
    }
    return false;
}

void CellInfo::computeGradientWeights(vec3 wt[8])
{
    for (int i = 0; i < 8; i++)
        vec3zero(wt[i]);

    int *t = tets[type][tet];

    // Compute inverse grid Jacobian (local axes = mi * global axes)
    mat3 m, mi;
    vec3sub(vertex[t[1]], vertex[t[0]], m[0]);
    vec3sub(vertex[t[2]], vertex[t[0]], m[1]);
    vec3sub(vertex[t[3]], vertex[t[0]], m[2]);
    mat3inv(m, mi);

#ifdef DEBUGGING // ### tmp sadlof 2006-06-14
    for (int i = 0; i < 4; i++)
    {
        printf("vec[%d]: (%g,%g,%g)\n", i,
               vertex[t[i]][0], vertex[t[i]][1], vertex[t[i]][2]);
    }

    printf("mi:\n");
    mat3dump(mi, stdout);
#endif

    // For single tet: grad f = mi * df/(dxi, deta, dzeta)
    //   df/(dxi, ...) = ( f(t[1])-f(t[0]), ...)

    // Compute weights for single tet, use columns of mi
    mat3 mit;
    mat3trp(mi, mit);
    vec3 wtTet[4];
    vec3copy(mit[0], wtTet[1]);
    vec3copy(mit[1], wtTet[2]);
    vec3copy(mit[2], wtTet[3]);

    vec3add(wtTet[1], wtTet[2], wtTet[0]);
    vec3add(wtTet[0], wtTet[3], wtTet[0]);
    vec3scal(wtTet[0], -1., wtTet[0]);

    // Distribute weights to cell vertices
    for (int k = 0; k < 4; k++)
    {
        if (t[k] < nVertices[type])
        { // regular vertex
            int vert = t[k];
            vec3add(wt[vert], wtTet[k], wt[vert]);
        }
        else if (k == 0)
        { // centroid
            for (int vert = 0; vert < nVertices[type]; vert++)
            {
                vec3 v;
                vec3scal(wtTet[k], 1. / (float)nVertices[type], v);
                vec3add(wt[vert], v, wt[vert]);
            }
        }
        else /* if (k ==1) */
        { // face centroid
            int face = t[k] - nVertices[type];
            for (int l = 0; l < 4; l++)
            {
                int vert = faces[type][face][l];
                vec3 v;
                vec3scal(wtTet[k], .25, v);
                vec3add(wt[vert], v, wt[vert]);
            }
        }
    }

#ifdef DEBUGGING // ### tmp sadlof 2006-06-14
    for (int i = 0; i < 8; i++)
    {
        printf("wt: (%g,%g,%g)\n", wt[i][0], wt[i][1], wt[i][2]);
    }
#endif
}

void Unstructured::computeNodeNeighbors()
{
    nodeNeighbors = new vector<int>[nNodes];

    for (int i = 0; i < nCells; i++)
    {
        CellInfo &c = loadCell(i);

        for (int j = 0; j < nEdges[c.type]; j++)
        {
            int a = edges[c.type][j][0];
            int b = edges[c.type][j][1];
            int node0 = c.node[a];
            int node1 = c.node[b];

            bool found = false;
            for (int k = 0; k < (signed)nodeNeighbors[node0].size(); k++)
            {
                if (nodeNeighbors[node0][k] == node1)
                    found = true;
            }
            if (!found)
            {
                nodeNeighbors[node0].push_back(node1);
                nodeNeighbors[node1].push_back(node0);
            }
        }
    }
}

void Unstructured::computeCellNeighbors()
{ // computes neighboring cells that share at least one node with given cell
    /* #### TODO:
     - actually ignoring T-nodes
     - neighbor cells sharing edge should come before cells sharing only node
   */

    cellNeighbors = new vector<int>[nCells];
    cellFaceNeighborsCnt.clear();
    for (int i = 0; i < nCells; i++)
    {
        cellFaceNeighborsCnt.push_back(0);
    }

    // first get cells that share a face ----------------------------------------

    // store cells by faces (helping data structure)
    std::map<Keys3, std::vector<int> > face3Nodes;
    std::map<Keys4, std::vector<int> > face4Nodes;
    for (int i = 0; i < nCells; i++)
    {
        CellInfo &ci = loadCell(i);
        for (int f = 0; f < nFaces[ci.type]; f++)
        {
            if (faceNodeCnt[ci.type][f] == 3)
            {
                int a = faces[ci.type][f][0];
                int b = faces[ci.type][f][1];
                int c = faces[ci.type][f][2];
                int node0 = ci.node[a];
                int node1 = ci.node[b];
                int node2 = ci.node[c];
                Keys3::sortInts3(node0, node1, node2);
                Keys3 key(node0, node1, node2);

                if (face3Nodes.find(key) == face3Nodes.end())
                {
                    // not found
                    std::vector<int> vec;
                    vec.push_back(i);
                    face3Nodes[key] = vec;
                }
                else
                {
                    face3Nodes[key].push_back(i);
                }
            }
            else
            { // quadrangle
                int a = faces[ci.type][f][0];
                int b = faces[ci.type][f][1];
                int c = faces[ci.type][f][2];
                int d = faces[ci.type][f][3];
                int node0 = ci.node[a];
                int node1 = ci.node[b];
                int node2 = ci.node[c];
                int node3 = ci.node[d];
                Keys4::sortInts4(node0, node1, node2, node3);
                Keys4 key(node0, node1, node2, node3);

                if (face4Nodes.find(key) == face4Nodes.end())
                {
                    // not found
                    std::vector<int> vec;
                    vec.push_back(i);
                    face4Nodes[key] = vec;
                }
                else
                {
                    face4Nodes[key].push_back(i);
                }
            }
        }
    }

    // get cell neighbors
    std::map<Keys3, std::vector<int> >::iterator it3;
    for (it3 = face3Nodes.begin(); it3 != face3Nodes.end(); it3++)
    {

        if (it3->second.size() < 2)
            continue;

        int cell0 = it3->second[0];
        int cell1 = it3->second[1];

        bool found = false;
        for (int k = 0; k < (signed)cellNeighbors[cell0].size(); k++)
        {
            if (cellNeighbors[cell0][k] == cell1)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            cellNeighbors[cell0].push_back(cell1);
            cellNeighbors[cell1].push_back(cell0);
            cellFaceNeighborsCnt[cell0]++;
            cellFaceNeighborsCnt[cell1]++;
        }
    }
    std::map<Keys4, std::vector<int> >::iterator it4;
    for (it4 = face4Nodes.begin(); it4 != face4Nodes.end(); it4++)
    {

        if (it4->second.size() < 2)
            continue;

        int cell0 = it4->second[0];
        int cell1 = it4->second[1];

        bool found = false;
        for (int k = 0; k < (signed)cellNeighbors[cell0].size(); k++)
        {
            if (cellNeighbors[cell0][k] == cell1)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            cellNeighbors[cell0].push_back(cell1);
            cellNeighbors[cell1].push_back(cell0);
            cellFaceNeighborsCnt[cell0]++;
            cellFaceNeighborsCnt[cell1]++;
        }
    }

    // then append cells that share at least one node ---------------------------

    // store cells by nodes (helping data structure)
    std::vector<int> *nodeCells = new std::vector<int>[nNodes];

    for (int i = 0; i < nCells; i++)
    {
        CellInfo &ci = loadCell(i);
        for (int ni = 0; ni < nVertices[ci.type]; ni++)
        {
            nodeCells[ci.node[ni]].push_back(i);
        }
    }

    // append cells
    for (int n = 0; n < nNodes; n++)
    {

        for (int ci1 = 0; ci1 < (int)nodeCells[n].size(); ci1++)
        {

            int c1 = nodeCells[n][ci1];

            for (int ci2 = 0; ci2 < (int)nodeCells[n].size(); ci2++)
            {

                int c2 = nodeCells[n][ci2];

                if (c1 == c2)
                    continue;

                // test if c2 already collected for c1
                bool found = false;
                for (int k = 0; k < (signed)cellNeighbors[c1].size(); k++)
                {
                    if (cellNeighbors[c1][k] == c2)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    cellNeighbors[c1].push_back(c2);

                // test if c1 already collected for c2
                found = false;
                for (int k = 0; k < (signed)cellNeighbors[c2].size(); k++)
                {
                    if (cellNeighbors[c2][k] == c1)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    cellNeighbors[c2].push_back(c1);
            }
        }
    }

    delete[] nodeCells;
}

inline void Unstructured::getCellNeighborsN_rek(int cell, int ncell, int range,
                                                int connectivity,
                                                std::vector<int> *neighbors,
                                                int *cellSizes)
{ // cellSizes: may be NULL
    if (range < 1)
        return;

    //for (int cn=0; cn<(int)cellNeighbors[ncell].size(); cn++) {
    for (int cn = 0; cn < cellNeighborCnt(ncell, connectivity); cn++)
    {

        int neigh = cellNeighbors[ncell][cn];

        // test if neighbor already collected
        bool found = false;
        for (int n = 0; n < (int)neighbors->size(); n++)
        {

            if ((neigh == (*neighbors)[n]) || (neigh == cell))
            {
                found = true;
                break;
            }
        }

        if (!found)
            neighbors->push_back(neigh);

        // recursion
        // getCellNeighborsN_rek(cell, neigh, range - 1, connectivity, neighbors);
        if (cellSizes)
        {
            // THIS WAS A BUG: 2007-07-26
            //getCellNeighborsN_rek(cell, neigh, range - cellSizes[cell], connectivity, neighbors, cellSizes);
            getCellNeighborsN_rek(cell, neigh, range - cellSizes[neigh], connectivity, neighbors, cellSizes);
        }
        else
        {
            getCellNeighborsN_rek(cell, neigh, range - 1, connectivity, neighbors, cellSizes);
        }
    }
}

std::vector<int> Unstructured::getCellNeighborsN(int cell, int range, int connectivity, int *cellSizes)
{ // gets neighboring cells inside range 'range'
    // connectivity: CONN_FACE: only sharing faces ('N6')
    //               CONN_NODE: sharing nodes ('N26')

    if (!cellNeighbors)
        computeCellNeighbors();

    std::vector<int> neighbors;

    getCellNeighborsN_rek(cell, cell, range, connectivity, &neighbors, cellSizes);

    return neighbors;
}

bool Unstructured::domainBoundaryCellNode(int node)
{ // returns true if node belongs to a cell that is at domain boundary

    // get all cells that contain node
    std::vector<int> nodeCells;
    {
        int saveCell = getCellIndex();

        vec3 nodeCoord;
        getCoords(node, nodeCoord); // no support for moving zones
        findCell(nodeCoord); // should not fail

        // get all cells that share at least one node with current cell
        // TODO: make more efficient (directly get cells that share node)
        std::vector<int> *cellNeigh = getCellNeighbors(getCellIndex());

        // now get the cells that contain the node
        for (int c = 0; c < (int)cellNeigh->size(); c++)
        {
            int *nodes = getCellNodesAVS((*cellNeigh)[c]);
            for (int n = 0; n < nVertices[getCellType((*cellNeigh)[c])]; n++)
            {
                if (nodes[n] == node)
                {
                    // found
                    nodeCells.push_back((*cellNeigh)[c]);
                }
            }
        }

        // add current cell
        nodeCells.push_back(getCellIndex());

        // restore selected cell
        loadCell(saveCell);
    }

    for (int c = 0; c < (int)nodeCells.size(); c++)
    {
        if (domainBoundaryCell(nodeCells[c]))
        {
            return true;
        }
    }

    return false;
}

// ### temp, should use STL map
int Unstructured::findInSortedIntArray(int key, int *arr, int arr_len)
{
    if (!arr || (arr_len < 1))
        return -1; // not found
    //if (!arr || (arr_len < 1)) return -INT_MAX; // not found // ###

    int ascending = (arr[0] <= arr[arr_len - 1]);

    int left = 0;
    int right = arr_len - 1;
    int mid = (left + right) / 2;

    while (1)
    {
        if (key == arr[mid])
        {
            // found
            return mid;
        }
        else if (left >= right)
        {
            return -mid - 1; // not found #### val
        }
        else if ((ascending ? key < arr[mid] : key > arr[mid]))
        {
            // in 'left' half
            right = mid - 1;
        }
        else
        {
            // in 'right' half
            left = mid + 1;
        }
        mid = (left + right) / 2;
    }
}

// ### temp, should use STL map
int Unstructured::insertInSortedIntArray(int key, int ascending, int *arr, int *arr_len)
{ // arr must be large enough
    // changes array (indices may invalidate)

    // binary search

    if (!arr || *arr_len < 1)
    {
        // no entry -> first
        arr[0] = key;
        (*arr_len)++;
        return 0;
    }

    int w = findInSortedIntArray(key, arr, *arr_len);
    if (w >= 0)
    {
        // found -> ### no multiple
        return w;
    }

    int w2 = -(w + 1);
    if ((ascending ? key < arr[w2] : key > arr[w2]))
    {
        // insert before
        memmove(&arr[w2 + 1], &arr[w2], (*arr_len - w2) * sizeof(int));
        arr[w2] = key;
        (*arr_len)++;
        return w2;
    }
    else
    {
        if (w2 + 1 > *arr_len - 1)
        {
            // no next
            // insert after
            memmove(&arr[w2 + 1 + 1], &arr[w2 + 1], (*arr_len - (w2 + 1)) * sizeof(int));
            arr[w2 + 1] = key;
            (*arr_len)++;
            return w2 + 1;
        }
        else
        {
            // ##### seems that this branch is not needed
            if ((ascending ? key < arr[w2 + 1] : key > arr[w2 + 1]))
            {
                // insert after current
                memmove(&arr[w2 + 1 + 1], &arr[w2 + 1], (*arr_len - (w2 + 1)) * sizeof(int));
                arr[w2 + 1] = key;
                (*arr_len)++;
                return w2 + 1;
            }
            else
            {
                // insert after next
                memmove(&arr[w2 + 2 + 1], &arr[w2 + 2], (*arr_len - (w2 + 2)) * sizeof(int));
                arr[w2 + 2] = key;
                (*arr_len)++;
                return w2 + 2;
            }
        }
    }
}

void Unstructured::computeNodeNeighborsRek(int node, int range, int *neighborsSorted, int *neighborsSortedCnt)
{

    if (range < 1)
        return;

    for (int n = 0; n < (signed)nodeNeighbors[node].size(); n++)
    {
        insertInSortedIntArray(nodeNeighbors[node][n], 1, neighborsSorted, neighborsSortedCnt);

        computeNodeNeighborsRek(nodeNeighbors[node][n], range - 1, neighborsSorted, neighborsSortedCnt);
    }
}

int Unstructured::computeNodeNeighborsN(int node, int range, int *neighborsN)
{ // compute all neighbors inside neighbot level 'range'
    // (range=2 -> second-level neighborhood)
    // returns number of neighbors written to 'neighborsN'

    // compute direct neighbors
    if (nodeNeighbors == 0)
        computeNodeNeighbors();

    // simple cases
    if (range < 1)
    {
        return 0;
    }
    else if (range == 1)
    {
        for (int n = 0; n < (signed)nodeNeighbors[node].size(); n++)
        {
            neighborsN[n] = nodeNeighbors[node][n];
        }
        return int(nodeNeighbors[node].size());
    }

    // compute neighbors of neighbors
    //###int neighborsSortedCnt = 0;
    int neighCnt = 0;

    computeNodeNeighborsRek(node, range, neighborsN, &neighCnt);

    return neighCnt;
}

DataDesc *Unstructured::newNodeCompExtraData(int comp, int dataType,
                                             int veclen, int operationId)
{ // alloc extra data at node component 'comp', labeled by operationId

    DataDesc *d = findNodeCompExtraData(comp, operationId);
    if (d)
    {
        // exists already
        printf("Unstructured::newNodeCompExtraData: exists already\n");
        return d;
    }

    d = new DataDesc(operationId, dataType, veclen, NULL);

    switch (dataType)
    {
    //case TP_MATRIX3:
    //d->p.fm3 = new fmat3[nNodes];
    //if (!d->p.fm3) return NULL;
    case TP_FLOAT:
        d->p.f = new float[veclen * nNodes];
        if (!d->p.f)
            return NULL;
        break;
    default:
        printf("Unstructured::newNodeCompExtraData: data type not implemented\n");
        break;
    }

    nodeCompExtraData[comp].push_back(d);
    return d;
}

void Unstructured::deleteNodeCompExtraData(DataDesc *dd)
{
    int comp, idx;
    if (!findNodeCompExtraDataCompIdx(dd, comp, idx))
        return;

    // ### ugly, change?
    switch (dd->dataType)
    {
#if 0
  case TP_SCALAR: delete [] dd->p.f;
    break;
  case TP_VECTOR3: delete [] dd->p.v3;
    break;
  case TP_MATRIX3: delete [] dd->p.fm3;
    break;
#else
    case TP_FLOAT:
        delete[] dd -> p.f;
#endif
    }
    delete dd;

    nodeCompExtraData[comp][idx] = nodeCompExtraData[comp][nodeCompExtraData[comp].size() - 1];
    nodeCompExtraData[comp].pop_back();

    if (scalarComponentExtraData == dd)
        scalarComponentExtraData = NULL;
    if (vectorComponentExtraData == dd)
        vectorComponentExtraData = NULL;
}

void Unstructured::deleteNodeCompExtraData(int comp, int operationId)
{ // delete extra data at node component 'comp', generated by operationId

    int idx = findNodeCompExtraDataIdx(comp, operationId);
    if (idx < 0)
        return;

    DataDesc *dd = nodeCompExtraData[comp][idx];
    deleteNodeCompExtraData(dd);

    //nodeCompExtraData[comp][idx] = nodeCompExtraData[comp][nodeCompExtraData[comp].size()-1];
    //nodeCompExtraData[comp].pop_back();
}

void Unstructured::deleteNodeCompExtraData(void)
{ // delete all extra data

    for (int comp = 0; comp < nodeComponentNb; comp++)
    {
        for (int i = 0; i < (signed)nodeCompExtraData[comp].size(); i++)
        {
            deleteNodeCompExtraData(nodeCompExtraData[comp][i]);
        }
        nodeCompExtraData[comp].clear();
    }
}

void Unstructured::deleteNodeNeighbors()
{
    if (nodeNeighbors)
        delete[] nodeNeighbors;
}

void Unstructured::deleteCellNeighbors()
{
    if (cellNeighbors)
        delete[] cellNeighbors;
}

bool Unstructured::computeWallNormal(vec3 nml)
{
    CellInfo &c = currCell;

    if (!c.atWall)
    {
        fprintf(stderr, "Error in CellInfo::computewallNormal -- cell is not at a wall\n");
        return false;
    }

    // no call to loadCell or findCell, -> assure data
    assureCellData();

    int n = 0;
    int a[8];
    for (int k = 0; k < nVertices[c.type]; k++)
    {
        if (c.wallDist[k] == 0)
            a[n++] = k;
    }

    if (n < 3)
    {
        return false;
        fprintf(stderr, "Error in CellInfo::computewallNormal -- current cell has less than 3 wall vertices\n");
    }

    vec3 edge1, edge2;
    vec3sub(c.vertex[a[1]], c.vertex[a[0]], edge1);
    vec3sub(c.vertex[a[2]], c.vertex[a[0]], edge2);
    vec3cross(edge1, edge2, nml);
    vec3nrm(nml, nml);
    return true;
}

void Unstructured::deleteCriticalPoints()
{
    criticalPoints.resize(0);
}

void Unstructured::deleteCriticalBoundaryPoints()
{
    criticalBoundaryPoints.resize(0);
}

void Unstructured::computeCriticalPoints()
{
    if (!vector3CB && !vectorExists)
    {
        fprintf(stderr, "Unstructured::computeCriticalPoints: error: no velocity\n");
        return;
    }

    if (divideVelocityByWalldist && !wallDistExists)
    {
        fprintf(stderr, "Unstructured::computeCriticalPoints: error: no wall distance\n");
        return;
    }

    deleteCriticalPoints();
    for (int i = 0; i < nCells; i++)
    {
        // ### TODO: attention, using time = 0.0 (rotating zone would be not rot.)
        CellInfo &c = loadCell(i, true);

        if (c.atWall && !divideVelocityByWalldist)
            continue;

        // Trivial reject test
        int signs = 0;
        for (int j = 0; j < nVertices[c.type]; j++)
        {
            if (c.vec[j][0] < 0)
                signs |= 0x1;
            if (c.vec[j][0] > 0)
                signs |= 0x2;
            if (c.vec[j][1] < 0)
                signs |= 0x4;
            if (c.vec[j][1] > 0)
                signs |= 0x8;
            if (c.vec[j][2] < 0)
                signs |= 0x10;
            if (c.vec[j][2] > 0)
                signs |= 0x20;
        }
        if (signs != 0x3f)
            continue;

        // Loop over tetrahedra
        for (int j = 0; j < nTets[c.type]; j++)
        {
            int *t = tets[c.type][j];
            CriticalPoint cp;

            // Trivial reject test again, now for the tet
            int signs = 0;
            for (int k = 0; k < 4; k++)
            {
                if (c.vec[t[k]][0] < 0)
                    signs |= 0x1;
                if (c.vec[t[k]][0] > 0)
                    signs |= 0x2;
                if (c.vec[t[k]][1] < 0)
                    signs |= 0x4;
                if (c.vec[t[k]][1] > 0)
                    signs |= 0x8;
                if (c.vec[t[k]][2] < 0)
                    signs |= 0x10;
                if (c.vec[t[k]][2] > 0)
                    signs |= 0x20;
            }
            if (signs != 0x3f)
                continue;

            // Find the point in local (barycentric) coords by solving:
            //   velo0 + matrix(velo1-velo0, velo2-velo0, velo3-velo0) * (x,y,z) = (0,0,0)
            vec3 dv1, dv2, dv3, rhs;
            vec3sub(c.vec[t[1]], c.vec[t[0]], dv1);
            vec3sub(c.vec[t[2]], c.vec[t[0]], dv2);
            vec3sub(c.vec[t[3]], c.vec[t[0]], dv3);
            vec3scal(c.vec[t[0]], -1, rhs);
            double denom = vec3det(dv1, dv2, dv3);
            if (denom == 0)
                continue;
            double x = vec3det(rhs, dv2, dv3) / denom;
            double y = vec3det(dv1, rhs, dv3) / denom;
            double z = vec3det(dv1, dv2, rhs) / denom;

            // Check if inside tet
            if (x < 0)
                continue;
            if (y < 0)
                continue;
            if (z < 0)
                continue;
            if (x + y + z > 1)
                continue;

            // Interpolate wall distance
            if (wallDistExists)
            {
                cp.wallDist = c.wallDist[t[0]] + x * (c.wallDist[t[1]] - c.wallDist[t[0]]) + y * (c.wallDist[t[2]] - c.wallDist[t[0]]) + z * (c.wallDist[t[3]] - c.wallDist[t[0]]);
            }
            else
            {
                cp.wallDist = 1.0; // ### ok?
            }

            // Interpolate coords
            vec3 dx1, dx2, dx3;
            vec3sub(c.vertex[t[1]], c.vertex[t[0]], dx1);
            vec3sub(c.vertex[t[2]], c.vertex[t[0]], dx2);
            vec3sub(c.vertex[t[3]], c.vertex[t[0]], dx3);

            vec3scal(dx1, x, dx1);
            vec3scal(dx2, y, dx2);
            vec3scal(dx3, z, dx3);

            vec3add(c.vertex[t[0]], dx1, cp.coord);
            vec3add(cp.coord, dx2, cp.coord);
            vec3add(cp.coord, dx3, cp.coord);

            cp.cell = i;

            //CellInfo& c = loadCell(i, true);
            c.computeWeights(cp.coord, weight);

            c.tet = j; // TODO: This is ugly!  Add a "setTet" method

            // DEUBUGGING:
            vec3 u;
            interpolateVector3(u);
#ifdef DEBUGGING
            printf("interpolated velo: %g %g %g\n", u[0], u[1], u[2]);
#endif

            mat3 m;
            interpolateVectorGradient(m);
            cp.allReal = (mat3eigenvalues(m, cp.eigenvalue) == 3);

#ifdef DEBUGGING
            if (cp.allReal)
                printf("eigenvalues: %g, %g, %g\n", cp.eigenvalue[0], cp.eigenvalue[1], cp.eigenvalue[2]);
            else
                printf("eigenvalues: %g, %g +-I %g\n", cp.eigenvalue[0], cp.eigenvalue[1], cp.eigenvalue[2]);
#endif

            if (cp.allReal)
            {
                mat3realEigenvector(m, cp.eigenvalue[0], cp.eigenvector[0]);
                mat3realEigenvector(m, cp.eigenvalue[1], cp.eigenvector[1]);
                mat3realEigenvector(m, cp.eigenvalue[2], cp.eigenvector[2]);
            }
            else
            {
                mat3realEigenvector(m, cp.eigenvalue[0], cp.eigenvector[0]);
                mat3complexEigenplane(m, cp.eigenvalue[1], cp.eigenvalue[2], cp.eigenvector[1]);
            }

            criticalPoints.push_back(cp);
        }
    }
}

void Unstructured::computeCriticalBoundaryPoints() // Assume velocity is divided by wallDist
{
    if (!vectorExists || !wallDistExists)
    {
        fprintf(stderr, "Unstructured::computeCriticalBoundaryPoints: no velocity or no wall-distance\n");
        return;
    }

    deleteCriticalBoundaryPoints();

    for (int i = 0; i < nCells; i++)
    {
        // ### TODO: attention, using time = 0.0 (rotating zone would be not rot.)
        CellInfo &c = loadCell(i, true);

        if (!c.atWall)
            continue;

        // Loop over tetrahedra
        for (int j = 0; j < nTets[c.type]; j++)
        {
            int *t = tets[c.type][j];
            CriticalPoint cp;

            // Reject if # wall vertices != 3
            int a[4];
            int n = 0;
            if (c.wallDist[t[0]] == 0)
                a[n++] = t[0];
            if (c.wallDist[t[1]] == 0)
                a[n++] = t[1];
            if (c.wallDist[t[2]] == 0)
                a[n++] = t[2];
            if (c.wallDist[t[3]] == 0)
                a[n++] = t[3];
            if (n != 3)
                continue;

            // Reject if zero vector at node
            if (vec3iszero(c.vec[a[0]]))
                continue;
            if (vec3iszero(c.vec[a[1]]))
                continue;
            if (vec3iszero(c.vec[a[2]]))
                continue;

            // Get critical point in triangle
            vec3 edge1, edge2, basis[3];
            vec3sub(c.vertex[a[1]], c.vertex[a[0]], edge1);
            vec3sub(c.vertex[a[2]], c.vertex[a[0]], edge2);
            vec3nrm(edge1, basis[0]);
            vec3cross(edge1, edge2, basis[2]);
            vec3nrm(basis[2], basis[2]);
            vec3cross(basis[2], basis[0], basis[1]);

            vec2 v0, v1, v2;
            v0[0] = vec3dot(c.vec[a[0]], basis[0]);
            v0[1] = vec3dot(c.vec[a[0]], basis[1]);
            v1[0] = vec3dot(c.vec[a[1]], basis[0]);
            v1[1] = vec3dot(c.vec[a[1]], basis[1]);
            v2[0] = vec3dot(c.vec[a[2]], basis[0]);
            v2[1] = vec3dot(c.vec[a[2]], basis[1]);

            vec2 dv1, dv2, rhs;
            vec2sub(v1, v0, dv1);
            vec2sub(v2, v0, dv2);
            vec2scal(v0, -1, rhs);
            double denom = vec2det(dv1, dv2);
            if (denom == 0)
                continue;

            double x = vec2det(rhs, dv2) / denom;
            double y = vec2det(dv1, rhs) / denom;

            // Check if inside triangle
            if (x < 0)
                continue;
            if (y < 0)
                continue;
            if (x + y > 1)
                continue;

            // Interpolate coords
            vec3 dx1, dx2;
            vec3scal(edge1, x, dx1);
            vec3scal(edge2, y, dx2);

            vec3add(c.vertex[a[0]], dx1, cp.coord);
            vec3add(cp.coord, dx2, cp.coord);

            cp.cell = i;

            // TODO: compute cp.allReal, cp.type, cp.eigenvalues[0], cp.eigenvectors[0]

            criticalBoundaryPoints.push_back(cp);
        }
    }
}

// --- basic math --------------------------------------------------------------
// TODO: move to separate source files

void Unstructured::gradient(int comp, vec3 position,
                            float *output,
                            double time,
                            int range,
                            bool *omitNodes, float *gradDefault,
                            std::vector<int> *lonelyNodes,
                            bool forceSymmetric)
{
    if (!findCell(position, time))
    {
        fprintf(stderr, "Unstructured::gradient: position outside domain\n");
        return;
    }

    // get nodes of current cell
    std::vector<int> cellNodesVec;
    {
        int *cellNodes = getCellNodesAVS(getCellIndex());
        for (int v = 0; v < nVertices[getCellType(getCellIndex())]; v++)
        {
            cellNodesVec.push_back(cellNodes[v]);
        }
    }

    // compute gradient at nodes of current cell
    int vecLenRes;
    if (getNodeCompVecLen(comp) == 1)
        vecLenRes = 3;
    else if (getNodeCompVecLen(comp) == 3)
        vecLenRes = 9;
    else
    {
        fprintf(stderr, "Unstructured::gradient: unsupported veclen\n");
        return;
    }
    Unstructured *unst_tmp = new Unstructured(this, vecLenRes);
    gradient(comp, unst_tmp, 0, range, omitNodes, gradDefault, lonelyNodes, forceSymmetric, &cellNodesVec);

    // interpolate
    unst_tmp->findCell(position);
    if (vecLenRes == 3)
    {
        vec3 v;
        unst_tmp->interpolateVector3(v);
        output[0] = v[0];
        output[1] = v[1];
        output[2] = v[2];
    }
    else
    {
        mat3 m;
        unst_tmp->interpolateMatrix3(m);
        output[0] = m[0][0];
        output[1] = m[1][0];
        output[2] = m[2][0];
        output[3] = m[0][1];
        output[4] = m[1][1];
        output[5] = m[2][1];
        output[6] = m[0][2];
        output[7] = m[1][2];
        output[8] = m[2][2];
    }

    delete unst_tmp;
}

// --- unary operations (allow output to extraData inside same component) ------

DataDesc *Unstructured::gradient(int comp, bool onlyGetID, int range,
                                 bool *omitNodes, float *gradDefault,
                                 std::vector<int> *lonelyNodes,
                                 bool forceSymmetric,
                                 std::vector<int> *nodes,
                                 double time)
{ // onlyGetID: if true, operation ID is returned (as pointer however .. )
    //            this is very ugly and produces compiler errors on 64bit systems
    // omitNodes: if not NULL, these nodes are neither included in neighborhood
    //            of a node nor is the gradient computed at these nodes
    //            (the gradient is set to gradDefault at these nodes)
    // lonelyNodes: if not NULL (and omitNodes not NULL), stores nodes that have
    //              not enough neighbors
    // forceSymmetric: if resulting gradient is matrix, it is forced to symmetric
    // nodes: if not NULL, gradient is computed only at these nodes

    if (onlyGetID)
    {
        return (DataDesc *)OP_GRADIENT;
    }

    deleteNodeCompExtraData(comp, OP_GRADIENT);
    DataDesc *dd = newNodeCompExtraData(comp, TP_FLOAT, 3 * 3, OP_GRADIENT);
    if (!dd)
    {
        fprintf(stderr, "Unstructured::gradient: error: out of memory\n");
        exit(1); // ###
    }

    Unstructured out = Unstructured(this, dd);

    gradient(comp, &out, 0, range, omitNodes, gradDefault, lonelyNodes, forceSymmetric, nodes, time);

    return dd;
}

void Unstructured::gradient(int comp, Unstructured *out, int outComp,
                            int range, bool *omitNodes,
                            float *gradDefault,
                            std::vector<int> *lonelyNodes,
                            bool forceSymmetric,
                            std::vector<int> *nodes,
                            double time,
                            int method)
{ // range >= 1
    // TODO: test if output component != input component

    if (method == GRAD_MEYER)
    {
        // ############# TODO: support time, outcomp, ......
        computeGradientMeyer(this, comp, range, out, outComp);
        return;
    }

    //printf("\nHACK\n"); // HACK RP

    if (nodeNeighbors == 0)
        computeNodeNeighbors();

    int *neighborsN = NULL;
    if (range > 1)
    {
        neighborsN = new int[nNodes];
    }

    if (transient)
    {
        printf("Unstructured::gradient: error: only per-cell transient update but gradient needs support range\n");
        return;
    }

#if 0 // future work
  bool *omitNodesArr = NULL;
  if (omitNodes) {
    omitNodesArr = new bool[nNodes];
    for (int n=0; n<nNodes; n++) {
      omitNodesArr[n] = false;
    }
    for (int ni=0; ni<(int)omitNodes->size(); ni++) {
      omitNodesArr[omitNodes[ni]] = true;
    }
  }
#endif

    if (getNodeCompVecLen(comp) == 1)
    {

        if (out->getNodeCompVecLen(outComp) != 3)
        {
            fprintf(stderr, "Unstructured::gradient: error: output component must have veclen %d\n", 3);
            exit(1); // ###
        }

        vec3 defaultG;
        if (gradDefault)
            vec3set(defaultG, gradDefault[0], gradDefault[1], gradDefault[2]);
        else
            vec3set(defaultG, 0.0, 0.0, 0.0);

        out->selectVectorNodeData(outComp);

        //for (int i = 0; i < nNodes; i++) {
        for (int ii = 0; (nodes ? ii < (int)nodes->size() : ii < nNodes); ii++)
        {
            int i;
            if (nodes)
                i = (*nodes)[ii];
            else
                i = ii;

            if (!(i % 1000))
                printf("%d%% done       \r", (int)((i * 100.0) / nNodes));

            if (omitNodes && omitNodes[i])
            {
                if (gradDefault)
                    out->setVector3(i, defaultG);
                continue;
            }

            mat3 m = { { 0 }, { 0 }, { 0 } };
            vec3 rhs = { 0 };

            vec3 xyzi; // ###, vi;
            getCoords(i, xyzi); // no support for rotating zones (not necessary? TODO)
            double fi = getScalar(i, comp, time);

            int neighCnt;
            if (range > 1)
            {
                // compute all neighbors inside level 'range'
                neighCnt = computeNodeNeighborsN(i, range, neighborsN);
            }
            else
            {
                neighCnt = int(nodeNeighbors[i].size());
            }

            // ### added 2007-08-16 for treating planar (skipped) cells
            if (neighCnt <= 0)
            {
                out->setVector3(i, defaultG);
                continue;
            }

            int effNeighCnt = 0;
            for (int k = 0; k < neighCnt; k++)
            {

                int j;
                if (range <= 1)
                {
                    j = nodeNeighbors[i][k];
                }
                else
                {
                    j = neighborsN[k];
                }

                if (omitNodes && omitNodes[j])
                    continue;
                else
                    effNeighCnt++;

                // Relative coordinates of neighbors
                vec3 xyzj, XYZ;
                getCoords(j, xyzj); // no support for rotating zones (not necessary? TODO)
                vec3sub(xyzj, xyzi, XYZ);
                double X = XYZ[0];
                double Y = XYZ[1];
                double Z = XYZ[2];

                m[0][0] += X * X;
                m[0][1] += X * Y;
                m[0][2] += X * Z;
                m[1][0] += Y * X;
                m[1][1] += Y * Y;
                m[1][2] += Y * Z;
                m[2][0] += Z * X;
                m[2][1] += Z * Y;
                m[2][2] += Z * Z;

                // Relative function values of neighbors
                double f = getScalar(j, comp, time) - fi;

                rhs[0] += f * X;
                rhs[1] += f * Y;
                rhs[2] += f * Z;
            }

            if (effNeighCnt < 3)
            { // TODO: how many neighbors needed?
                //if (effNeighCnt < 4) { // TODO: how many neighbors needed? think 4 but still can get singular ...
                if (gradDefault)
                    out->setVector3(i, defaultG);
                if (lonelyNodes)
                    lonelyNodes->push_back(i);
                continue;
            }

            double det = mat3det(m);
#if 0 // DELETEME?  ################### discuss with Ronny
      if (det == 0) det = 1;
#else
            if (lonelyNodes && (det == 0))
            {
                if (gradDefault)
                    out->setVector3(i, defaultG);
                if (lonelyNodes)
                    lonelyNodes->push_back(i);
                continue;
            }
            else
            {
                if (det == 0)
                    det = 1;
            }
#endif

            vec3 grad;

            grad[0] = vec3det(rhs, m[1], m[2]) / det; // dV0/dx
            grad[1] = vec3det(m[0], rhs, m[2]) / det; // dV0/dy
            grad[2] = vec3det(m[0], m[1], rhs) / det; // dV0/dz

            for (int v = 0; v < 3; v++)
            {
                if (grad[v] > FLT_MAX)
                {
                    printf("component %d at node %d is >FLT_MAX -> limiting to FLT_MAX\n", v, i);
                    grad[v] = FLT_MAX;
                }
                else if (grad[v] < -FLT_MAX)
                {
                    printf("component %d at node %d is < -FLT_MAX -> limiting to -FLT_MAX\n", v, i);
                    grad[v] = -FLT_MAX;
                }
            }

            out->setVector3(i, grad);
        }
    }
    else if (getNodeCompVecLen(comp) == 3)
    {

        if (out->getNodeCompVecLen(outComp) != 3 * 3)
        {
            fprintf(stderr, "Unstructured::gradient: error: output component must have veclen %d\n", 3 * 3);
            exit(1); // ###
        }

        fmat3 defaultG;
        if (gradDefault)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    defaultG[i][j] = gradDefault[i + j * 3];
                }
            }
        }
        else
        {
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    defaultG[i][j] = 0.0;
                }
            }
        }

        out->selectVectorNodeData(outComp);

        //for (int i = 0; i < nNodes; i++) {
        for (int ii = 0; (nodes ? ii < (int)nodes->size() : ii < nNodes); ii++)
        {
            int i;
            if (nodes)
                i = (*nodes)[ii];
            else
                i = ii;

            if (!(i % 100))
                printf("%d%% done       \r", (int)((i * 100.0) / nNodes));

            if (omitNodes && omitNodes[i])
            {
                if (gradDefault)
                    out->setMatrix3(i, defaultG);
                continue;
            }

            mat3 m = { { 0 }, { 0 }, { 0 } };
            vec3 rhs0 = { 0 };
            vec3 rhs1 = { 0 };
            vec3 rhs2 = { 0 };

            vec3 xyzi, vi;
            getCoords(i, xyzi); // no support for rotating zones (not necessary? TODO)
            getVector3(i, comp, time, vi);

            int neighCnt;
            if (range > 1)
            {
                // compute all neighbors inside level 'range'
                neighCnt = computeNodeNeighborsN(i, range, neighborsN);
            }
            else
            {
                neighCnt = int(nodeNeighbors[i].size());
            }

            // ### added 2007-08-16 for treating planar (skipped) cells
            if (neighCnt <= 0)
            {
                out->setMatrix3(i, defaultG);
                continue;
            }

            int effNeighCnt = 0;
            for (int k = 0; k < neighCnt; k++)
            {

                int j;
                if (range <= 1)
                {
                    j = nodeNeighbors[i][k];
                }
                else
                {
                    j = neighborsN[k];
                }

                if (omitNodes && omitNodes[j])
                    continue;
                else
                    effNeighCnt++;

                // Relative coordinates of neighbors
                vec3 xyzj, XYZ;
                getCoords(j, xyzj); // no support for rotating zones (not necessary? TODO)
                vec3sub(xyzj, xyzi, XYZ);
                double X = XYZ[0];
                double Y = XYZ[1];
                double Z = XYZ[2];

                m[0][0] += X * X;
                m[0][1] += X * Y;
                m[0][2] += X * Z;
                m[1][0] += Y * X;
                m[1][1] += Y * Y;
                m[1][2] += Y * Z;
                m[2][0] += Z * X;
                m[2][1] += Z * Y;
                m[2][2] += Z * Z;

                // Relative velocities of neighbors
                vec3 vj, V;
                getVector3(j, comp, time, vj);
                vec3sub(vj, vi, V);

                rhs0[0] += V[0] * X;
                rhs0[1] += V[0] * Y;
                rhs0[2] += V[0] * Z;
                rhs1[0] += V[1] * X;
                rhs1[1] += V[1] * Y;
                rhs1[2] += V[1] * Z;
                rhs2[0] += V[2] * X;
                rhs2[1] += V[2] * Y;
                rhs2[2] += V[2] * Z;

                // HACK RP
                //if (i == testNode) {
                //printf("nb #%2d of node %d: node %d\n", k, i, j);
                //	printf("  xyz: %10.6f %10.6f %10.6f", xyzj[0], xyzj[1], xyzj[2]);
                //	printf("  map: %10.6f %10.6f %10.6f\n", vj[0], vj[1], vj[2]);
                //}
            }

            if (effNeighCnt < 3)
            { // TODO: how many neighbors needed?
                if (gradDefault)
                    out->setMatrix3(i, defaultG);
                if (lonelyNodes)
                    lonelyNodes->push_back(i);
                continue;
            }

            double det = mat3det(m);
#if 0 // DELETEME?  ################### discuss with Ronny
      if (det == 0) det = 1;
#else
            if (lonelyNodes && (det == 0))
            {
                if (gradDefault)
                    out->setMatrix3(i, defaultG);
                if (lonelyNodes)
                    lonelyNodes->push_back(i);
                continue;
            }
            else
            {
                if (det == 0)
                    det = 1;
            }
#endif

            fmat3 grad;

            grad[0][0] = vec3det(rhs0, m[1], m[2]) / det; // dV0/dx
            grad[0][1] = vec3det(m[0], rhs0, m[2]) / det; // dV0/dy
            grad[0][2] = vec3det(m[0], m[1], rhs0) / det; // dV0/dz

            grad[1][0] = vec3det(rhs1, m[1], m[2]) / det; // dV1/dx
            grad[1][1] = vec3det(m[0], rhs1, m[2]) / det; // dV1/dy
            grad[1][2] = vec3det(m[0], m[1], rhs1) / det; // dV1/dz

            grad[2][0] = vec3det(rhs2, m[1], m[2]) / det; // dV2/dx
            grad[2][1] = vec3det(m[0], rhs2, m[2]) / det; // dV2/dy
            grad[2][2] = vec3det(m[0], m[1], rhs2) / det; // dV2/dz

            for (int w = 0; w < 3; w++)
            {
                for (int v = 0; v < 3; v++)
                {
                    if (grad[v][w] > FLT_MAX)
                    {
                        printf("element %d,%d at node %d is >FLT_MAX -> limiting to FLT_MAX\n", v, w, i);
                        grad[v][w] = FLT_MAX;
                    }
                    else if (grad[v][w] < -FLT_MAX)
                    {
                        printf("element %d,%d at node %d is < -FLT_MAX -> limiting to -FLT_MAX\n", v, w, i);
                        grad[v][w] = -FLT_MAX;
                    }
                }
            }

            // if (i == testNode) {	// HACK RP
            //	  printf("%10.6f %10.6f %10.6f\n", grad[0][0], grad[0][1], grad[0][2]);
            //	  printf("%10.6f %10.6f %10.6f\n", grad[1][0], grad[1][1], grad[1][2]);
            //	  printf("%10.6f %10.6f %10.6f\n", grad[2][0], grad[2][1], grad[2][2]);
            //  }

            // untested 2007-08-15:
            if (forceSymmetric)
                fmat3symm(grad, grad);

#if 1 // TESTING
            {
                bool ok = true;
                for (int jj = 0; jj < 3; jj++)
                {
                    for (int ii = 0; ii < 3; ii++)
                    {
                        if (isnan(grad[ii][jj]))
                        {
                            printf("Unstructured::gradient: NAN at node %d from vec (%g,%g,%g)\n", i, vi[0], vi[1], vi[2]);
                            ok = false;
                            break;
                        }
                    }
                }
                if (!ok)
                { // added 2007-08-22
                    if (gradDefault)
                        out->setMatrix3(i, defaultG);
                    if (lonelyNodes)
                        lonelyNodes->push_back(i);

                    for (int k = 0; k < neighCnt; k++)
                    {

                        int j;
                        if (range <= 1)
                        {
                            j = nodeNeighbors[i][k];
                        }
                        else
                        {
                            j = neighborsN[k];
                        }

                        // Relative coordinates of neighbors
                        vec3 xyzj, XYZ;
                        getCoords(j, xyzj); // no support for rotating zones (not necessary? TODO)
                        vec3sub(xyzj, xyzi, XYZ);
                        double X = XYZ[0];
                        double Y = XYZ[1];
                        double Z = XYZ[2];

                        // Relative velocities of neighbors
                        vec3 vj, V;
                        getVector3(j, comp, time, vj);
                        vec3sub(vj, vi, V);

                        printf("neigh=%d dxyz=(%g,%g,%g) dv=(%g,%g,%g)\n",
                               k, X, Y, Z, V[0], V[1], V[2]);
                    }

                    continue;
                }
            }
#endif

            out->setMatrix3(i, grad);
        }
    }
    else
    {
        printf("Unstructured::gradient: veclen=%d not yet supported\n", getNodeCompVecLen(comp));
    }

    if (neighborsN)
        delete[] neighborsN;
    //if (omitNodesArr) delete [] omitNodesArr;
}

// ######## TODO: support time, lonely nodes, etc. ....
//... use univiz_tmp version !
void Unstructured::computeGradientMeyer(Unstructured *unst, int compIn, int range, Unstructured *unst_out, int compOut)
{
    // according to Meyer, Eriksson, Maggio: "Gradient Estimation from Irregularly Spaced Data Sets"

    int neighborsN[1000]; // ### HACK

    for (int n = 0; n < unst->nNodes; n++)
    {

        if (!(n % 1000))
        {
            //char buf[256];
            //sprintf(buf, "processing node %d", n);
            //AVSmodule_status(buf, (int) ((((double) n) / unst->nNodes) * 100));
            printf("%d%% done       \r", (int)((n * 100.0) / unst->nNodes));
        }

        int neighCnt;
        // compute all neighbors inside level 'range'
        neighCnt = unst->computeNodeNeighborsN(n, range, neighborsN);

        mat3 VTV;
        vec3 t;
        vec3 pos;
        vec3zero(t);
        mat3zero(VTV);
        unst->getCoords(n, pos);
        for (int nn = 0; nn < neighCnt; nn++)
        {

            if (neighborsN[nn] == n)
                continue;

            vec3 npos, v, vn;
            unst->getCoords(neighborsN[nn], npos);
            vec3sub(npos, pos, v);
            vec3nrm(v, vn);

            VTV[0][0] += vn[0] * vn[0];
            VTV[0][1] += vn[0] * vn[1];
            VTV[0][2] += vn[0] * vn[2];

            VTV[1][0] += vn[1] * vn[0];
            VTV[1][1] += vn[1] * vn[1];
            VTV[1][2] += vn[1] * vn[2];

            VTV[2][0] += vn[2] * vn[0];
            VTV[2][1] += vn[2] * vn[1];
            VTV[2][2] += vn[2] * vn[2];
        }

        if (unst->getNodeCompVecLen(compIn) == 1)
        {
            mat3 VTVI;
            mat3inv(VTV, VTVI);
            double valN = unst->getScalar(n);
            for (int nn = 0; nn < neighCnt; nn++)
            {

                if (neighborsN[nn] == n)
                    continue;

                double valNN = unst->getScalar(neighborsN[nn]);
                vec3 npos, v, vn;
                unst->getCoords(neighborsN[nn], npos);
                vec3sub(npos, pos, v);
                vec3nrm(v, vn);
                double vmag = vec3mag(v);
                double scal = (valNN - valN) / vmag;
                t[0] += scal * (vn[0] * VTVI[0][0] + vn[1] * VTVI[0][1] + vn[2] * VTVI[0][2]);
                t[1] += scal * (vn[0] * VTVI[1][0] + vn[1] * VTVI[1][1] + vn[2] * VTVI[1][2]);
                t[2] += scal * (vn[0] * VTVI[2][0] + vn[1] * VTVI[2][1] + vn[2] * VTVI[2][2]);
            }

            unst_out->setVector3(n, compOut, t);
        }
        else if (unst->getNodeCompVecLen(compIn) == 3)
        {

            vec3 vecN;
            unst->getVector3(n, vecN);

            mat3 m;

            for (int c = 0; c < 3; c++)
            {

                mat3 VTVI;
                mat3inv(VTV, VTVI);
                double valN = vecN[c];
                for (int nn = 0; nn < neighCnt; nn++)
                {

                    if (neighborsN[nn] == n)
                        continue;

                    vec3 vecNN;
                    unst->getVector3(neighborsN[nn], vecNN);
                    double valNN = vecNN[c];
                    vec3 npos, v, vn;
                    unst->getCoords(neighborsN[nn], npos);
                    vec3sub(npos, pos, v);
                    vec3nrm(v, vn);
                    double vmag = vec3mag(v);
                    double scal = (valNN - valN) / vmag;
                    t[0] += scal * (vn[0] * VTVI[0][0] + vn[1] * VTVI[0][1] + vn[2] * VTVI[0][2]);
                    t[1] += scal * (vn[0] * VTVI[1][0] + vn[1] * VTVI[1][1] + vn[2] * VTVI[1][2]);
                    t[2] += scal * (vn[0] * VTVI[2][0] + vn[1] * VTVI[2][1] + vn[2] * VTVI[2][2]);
                }

                m[c][0] = t[0];
                m[c][1] = t[1];
                m[c][2] = t[2];
            }

            fmat3 mf;
            mat3tofmat3(m, mf);
            unst_out->setMatrix3(n, compOut, mf);
        }
    }
}

// TODO:DataDesc *Unstructured::transpose(...)

void Unstructured::transpose(int matComp, Unstructured *out, int outComp)
{ // transpose matrix
    // TODO: test if output component != input component

    if (getNodeCompVecLen(matComp) != 3 * 3)
    {
        printf("Unstructured::transpose: veclen %d not yet implemented\n",
               getNodeCompVecLen(matComp));
        return; // TODO
    }
    else
    {
        // 3x3 matrix quantity

        fmat3 *matIn = (fmat3 *)getNodeMatrixCompPtr(matComp);
        out->selectVectorNodeData(outComp);

        for (int n = 0; n < nNodes; n++)
        {
            fmat3 trp;
            fmat3trp(matIn[n], trp);
            out->setMatrix3(n, trp);
        }
    }
}

void Unstructured::matrixInvert(int matComp, Unstructured *out, int outComp, bool mode2D)
{ // invert matrix
    // TODO: test if output component != input component

    if (getNodeCompVecLen(matComp) != 3 * 3)
    {
        printf("Unstructured::matrixInvert: veclen %d not yet implemented\n",
               getNodeCompVecLen(matComp));
        return; // TODO
    }
    else
    {
        // 3x3 matrix quantity

        fmat3 *matIn = (fmat3 *)getNodeMatrixCompPtr(matComp);
        out->selectVectorNodeData(outComp);

        for (int n = 0; n < nNodes; n++)
        {
            fmat3 inv;

            if (mode2D)
            {
                fmat2 mIn, mOut;
                mIn[0][0] = matIn[n][0][0];
                mIn[1][0] = matIn[n][1][0];
                mIn[0][1] = matIn[n][0][1];
                mIn[1][1] = matIn[n][1][1];
                if (fmat2det(mIn) != 0)
                {
                    fmat2inv(mIn, mOut);
                }
                else
                {
                    mIn[0][0] = 0;
                    mIn[1][0] = 0;
                    mIn[0][1] = 0;
                    mIn[1][1] = 0;
                }
                inv[0][0] = mOut[0][0];
                inv[1][0] = mOut[1][0];
                inv[0][1] = mOut[0][1];
                inv[1][1] = mOut[1][1];
                inv[2][0] = 0.0;
                inv[2][1] = 0.0;
                inv[2][2] = 0.0;
                inv[0][2] = 0.0;
                inv[1][2] = 0.0;
            }
            else
            {
                fmat3inv(matIn[n], inv);
            }
            out->setMatrix3(n, inv);
        }
    }
}

void Unstructured::det(int matComp, Unstructured *out, int outComp)
{ // determinant
    // TODO: test if output component != input component

    if (getNodeCompVecLen(matComp) != 3 * 3)
    {
        printf("Unstructured::det: veclen %d not yet implemented\n",
               getNodeCompVecLen(matComp));
        return; // TODO
    }
    else
    {

        fmat3 *mat = (fmat3 *)getNodeMatrixCompPtr(matComp);

        out->selectScalarNodeData(outComp);

        for (int n = 0; n < nNodes; n++)
        {

            double det = fmat3det(mat[n]);
            out->setScalar(n, det);
        }
    }
}

void Unstructured::realEigenvaluesSortedDesc(int matComp, bool absoluteSorted, Unstructured *out, int outComp)
{ // compute real eigenvalues, sorted in descending order
    // absoluteSorted: if true, sorted according absolute value of eigenvalue
    // TODO: test if output component != input component

    if (getNodeCompVecLen(matComp) != 3 * 3)
    {
        printf("Unstructured::realEigenvalue: veclen %d not yet implemented\n",
               getNodeCompVecLen(matComp));
        return; // TODO
    }
    else
    {

        fmat3 *mat = (fmat3 *)getNodeMatrixCompPtr(matComp);

        out->selectVectorNodeData(outComp);

        int complexEVCnt = 0;
        double complexEV_imagMin = FLT_MAX;
        double complexEV_imagMax = 0.0;

        for (int n = 0; n < nNodes; n++)
        {

            // compute eigenvalues
            mat3 dmat;
            fmat3tomat3(mat[n], dmat);
            // FIX 2007-08-15: force symmetric matrix (untested) ###
            mat3symm(dmat, dmat);
            vec3 eigenvalues;
            bool allReal = (mat3eigenvalues(dmat, eigenvalues) == 3);

            if (!allReal)
            {
                //printf("Unstructured::realEigenvalue: got complex eigenvalue, skipping node %d\n", n);
                complexEVCnt++;
                {
                    double imagAbs = fabs(eigenvalues[2]);
                    if (imagAbs > complexEV_imagMax)
                    {
                        complexEV_imagMax = imagAbs;
                    }
                    if (imagAbs < complexEV_imagMin)
                    {
                        complexEV_imagMin = imagAbs;
                    }
                }
                vec3 vec;
                vec3zero(vec);
                out->setVector3(n, vec);
                continue;
            }

            if (absoluteSorted)
            {
                if (fabs(eigenvalues[0]) < fabs(eigenvalues[1]))
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;
                }
                if (fabs(eigenvalues[1]) < fabs(eigenvalues[2]))
                {
                    double w = eigenvalues[1];
                    eigenvalues[1] = eigenvalues[2];
                    eigenvalues[2] = w;
                }
                if (fabs(eigenvalues[0]) < fabs(eigenvalues[1]))
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;
                }
            }
            else
            {
                // ### ugly replication of code
                if (eigenvalues[0] < eigenvalues[1])
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;
                }
                if (eigenvalues[1] < eigenvalues[2])
                {
                    double w = eigenvalues[1];
                    eigenvalues[1] = eigenvalues[2];
                    eigenvalues[2] = w;
                }
                if (eigenvalues[0] < eigenvalues[1])
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;
                }
            }

            out->setVector3(n, eigenvalues);
        }

        //if (complexEVCnt) printf("Unstructured::realEigenvalue: got complex eigenvalue at %d nodes\n", complexEVCnt);
        if (complexEVCnt)
            printf("Unstructured::realEigenvalue: got complex eigenvalue at %d nodes, minImag=%g maxImag=%g\n", complexEVCnt, complexEV_imagMin, complexEV_imagMax);
    }
}

DataDesc *Unstructured::realEigenvectorSortedDesc(int matComp, bool absoluteSorted, int index, bool suppressWarnings)
{ // compute real eigenvector, sorted in order of descending eigenvalues
    // absoluteSorted: if true, sorted according absolute value of eigenvalue
    // index: index of corresponding sorted eigenvalue
    // TODO: test if output component != input component

    // TODO: ATTENTION do not use ...SortedDesc and ...SortedAsc in same unst.
    int id = -1;
    if (index == 0)
        id = OP_EIGENVECTOR0;
    else if (index == 1)
        id = OP_EIGENVECTOR1;
    else if (index == 2)
        id = OP_EIGENVECTOR2;

    deleteNodeCompExtraData(matComp, id);
    DataDesc *dd = newNodeCompExtraData(matComp, TP_FLOAT, 3, id);
    if (!dd)
    {
        fprintf(stderr, "Unstructured::realEigenvectorSortedDesc: error: out of memory\n");
        exit(1); // ###
    }

    Unstructured out = Unstructured(this, dd);

    realEigenvectorSortedDesc(matComp, absoluteSorted, index, &out, 0, suppressWarnings);

    return dd;
}

void Unstructured::realEigenvectorSortedDesc(int matComp, bool absoluteSorted, int index, Unstructured *out, int outComp, bool suppressWarnings)
{ // compute real eigenvector, sorted in order of descending eigenvalues
    // absoluteSorted: if true, sorted according absolute value of eigenvalue
    // index: index of corresponding sorted eigenvalue
    // TODO: test if output component != input component

    if (getNodeCompVecLen(matComp) != 3 * 3)
    {
        printf("Unstructured::realEigenvector: veclen %d not yet implemented\n",
               getNodeCompVecLen(matComp));
        return; // TODO
    }
    else
    {

        fmat3 *mat = (fmat3 *)getNodeMatrixCompPtr(matComp);

        out->selectVectorNodeData(outComp);

        int complexEVCnt = 0;

        for (int n = 0; n < nNodes; n++)
        {

            // compute eigenvalues
            mat3 dmat;
            fmat3tomat3(mat[n], dmat);
            vec3 eigenvalues;
            bool allReal = (mat3eigenvalues(dmat, eigenvalues) == 3);

            if (!allReal)
            {
                //if (!suppressWarnings) printf("Unstructured::realEigenvector: got complex eigenvalue, skipping node %d\n", n);
                complexEVCnt++;
                vec3 vec;
                vec3zero(vec);
                out->setVector3(n, vec);
                continue;
            }

            vec3 eigenvectors[3];

            mat3realEigenvector(dmat, eigenvalues[0], eigenvectors[0]);

            // real eigenvalues -> compute eigenvectors
            mat3realEigenvector(dmat, eigenvalues[1], eigenvectors[1]);
            mat3realEigenvector(dmat, eigenvalues[2], eigenvectors[2]);

            if (absoluteSorted)
            {
                if (fabs(eigenvalues[0]) < fabs(eigenvalues[1]))
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
                if (fabs(eigenvalues[1]) < fabs(eigenvalues[2]))
                {
                    double w = eigenvalues[1];
                    eigenvalues[1] = eigenvalues[2];
                    eigenvalues[2] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[1], wv);
                    vec3copy(eigenvectors[2], eigenvectors[1]);
                    vec3copy(wv, eigenvectors[2]);
                }
                if (fabs(eigenvalues[0]) < fabs(eigenvalues[1]))
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
            }
            else
            {
                // ### ugly replication of code

                if (eigenvalues[0] < eigenvalues[1])
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
                if (eigenvalues[1] < eigenvalues[2])
                {
                    double w = eigenvalues[1];
                    eigenvalues[1] = eigenvalues[2];
                    eigenvalues[2] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[1], wv);
                    vec3copy(eigenvectors[2], eigenvectors[1]);
                    vec3copy(wv, eigenvectors[2]);
                }
                if (eigenvalues[0] < eigenvalues[1])
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
            }

            out->setVector3(n, eigenvectors[index]);
        }

        if (!suppressWarnings && complexEVCnt)
            printf("Unstructured::realEigenvector: got complex eigenvalue at %d nodes\n", complexEVCnt);
    }
}

void Unstructured::realEigenvectorSortedAsc(int matComp, bool absoluteSorted, int index, Unstructured *out, int outComp, bool suppressWarnings)
{ // compute real eigenvector, sorted in order of ascending eigenvalues
    // absoluteSorted: if true, sorted according absolute value of eigenvalue
    // index: index of corresponding sorted eigenvalue
    // TODO: test if output component != input component

    if (getNodeCompVecLen(matComp) != 3 * 3)
    {
        printf("Unstructured::realEigenvector: veclen %d not yet implemented\n",
               getNodeCompVecLen(matComp));
        return; // TODO
    }
    else
    {

        fmat3 *mat = (fmat3 *)getNodeMatrixCompPtr(matComp);

        out->selectVectorNodeData(outComp);

        int complexEVCnt = 0;

        for (int n = 0; n < nNodes; n++)
        {

            // compute eigenvalues
            mat3 dmat;
            fmat3tomat3(mat[n], dmat);
            vec3 eigenvalues;
            bool allReal = (mat3eigenvalues(dmat, eigenvalues) == 3);

            if (!allReal)
            {
                //if (!suppressWarnings) printf("Unstructured::realEigenvector: got complex eigenvalue, skipping node %d\n", n);
                complexEVCnt++;
                vec3 vec;
                vec3zero(vec);
                out->setVector3(n, vec);
                continue;
            }

            vec3 eigenvectors[3];

            mat3realEigenvector(dmat, eigenvalues[0], eigenvectors[0]);

            // real eigenvalues -> compute eigenvectors
            mat3realEigenvector(dmat, eigenvalues[1], eigenvectors[1]);
            mat3realEigenvector(dmat, eigenvalues[2], eigenvectors[2]);

            if (absoluteSorted)
            {
                if (fabs(eigenvalues[0]) > fabs(eigenvalues[1]))
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
                if (fabs(eigenvalues[1]) > fabs(eigenvalues[2]))
                {
                    double w = eigenvalues[1];
                    eigenvalues[1] = eigenvalues[2];
                    eigenvalues[2] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[1], wv);
                    vec3copy(eigenvectors[2], eigenvectors[1]);
                    vec3copy(wv, eigenvectors[2]);
                }
                if (fabs(eigenvalues[0]) > fabs(eigenvalues[1]))
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
            }
            else
            {
                // ### ugly replication of code

                if (eigenvalues[0] > eigenvalues[1])
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
                if (eigenvalues[1] > eigenvalues[2])
                {
                    double w = eigenvalues[1];
                    eigenvalues[1] = eigenvalues[2];
                    eigenvalues[2] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[1], wv);
                    vec3copy(eigenvectors[2], eigenvectors[1]);
                    vec3copy(wv, eigenvectors[2]);
                }
                if (eigenvalues[0] > eigenvalues[1])
                {
                    double w = eigenvalues[0];
                    eigenvalues[0] = eigenvalues[1];
                    eigenvalues[1] = w;

                    vec3 wv;
                    vec3copy(eigenvectors[0], wv);
                    vec3copy(eigenvectors[1], eigenvectors[0]);
                    vec3copy(wv, eigenvectors[1]);
                }
            }

            out->setVector3(n, eigenvectors[index]);
        }

        if (!suppressWarnings && complexEVCnt)
            printf("Unstructured::realEigenvector: got complex eigenvalue at %d nodes\n", complexEVCnt);
    }
}

// --- (n-ary) operations (no output to extraData) ----------------------------

void Unstructured::matVec(int matComp, Unstructured *uvec, int vecComp,
                          Unstructured *out, int outComp)
{ // matrix vector multiplication
    // TODO: test if output component != input component

    if ((getNodeCompVecLen(matComp) != 3 * 3) || (uvec->getNodeCompVecLen(vecComp) != 3))
    {
        printf("Unstructured::matVec: veclens %d x %d not yet implemented\n",
               getNodeCompVecLen(matComp), uvec->getNodeCompVecLen(vecComp));
        return; // TODO
    }
    else
    {
        // 3x3 matrix quantity

        if (out->getNodeCompVecLen(outComp) != 3)
        {
            printf("Unstructured::matVec: output vector length must be 3 but is %d\n",
                   out->getNodeCompVecLen(outComp));
            return; // TODO
        }

        fmat3 *grad = (fmat3 *)getNodeMatrixCompPtr(matComp);

        uvec->selectVectorNodeData(vecComp);
        out->selectVectorNodeData(outComp);

        for (int n = 0; n < nNodes; n++)
        {
            vec3 v;
            uvec->getVector3(n, v);

            mat3 j;
            fmat3tomat3(grad[n], j);

            vec3 res;
            mat3vec(j, v, res);
            out->setVector3(n, res);
        }
    }
}

void Unstructured::matVec(int matComp, int opId,
                          Unstructured *uvec, int vecComp,
                          Unstructured *out, int outComp)
{ // matrix vector multiplication (matrix as extra data inside component)

    DataDesc *dd = findNodeCompExtraData(matComp, opId);
    Unstructured wrap = Unstructured(this, dd);
    wrap.matVec(0, uvec, vecComp, out, outComp);
}

void Unstructured::matMat(int matComp1, Unstructured *umat, int matComp2,
                          Unstructured *out, int outComp)
{ // matrix matrix multiplication
    // TODO: test if output component != input component

    if ((getNodeCompVecLen(matComp1) != 3 * 3) || (umat->getNodeCompVecLen(matComp2) != 3 * 3))
    {
        printf("Unstructured::matMat: veclens %d x %d not yet implemented\n",
               getNodeCompVecLen(matComp1), umat->getNodeCompVecLen(matComp2));
        return; // TODO
    }
    else
    {
        // 3x3 times 3x3 matrix quantity

        fmat3 *mat1 = (fmat3 *)getNodeMatrixCompPtr(matComp1);
        fmat3 *mat2 = (fmat3 *)getNodeMatrixCompPtr(matComp2);

        out->selectVectorNodeData(outComp);

        for (int n = 0; n < nNodes; n++)
        {

            mat3 m1, m2;
            fmat3tomat3(mat1[n], m1);
            fmat3tomat3(mat2[n], m2);

            fmat3 res;
            fmat3mul(mat1[n], mat2[n], res);
            out->setMatrix3(n, res);
        }
    }
}

int Unstructured::getNodeListSize()
{
    int n = 0;
    for (int i = 0; i < nCells; i++)
    {
        n += nVertices[cellType[i]];
    }
    return n;
}

void Unstructured::printInfo(FILE *fp)
{
    if (nNodes == 0)
    {
        fprintf(fp, "Invalid 'Unstructured'.\n");
        return;
    }

    fprintf(fp, "\t\t ---------------------------------------\n");
    fprintf(fp, "\n");

    fprintf(fp, "\t\t 'Unstructured' named: %s\n", name);
    fprintf(fp, "\t\t %d nodes\n", nNodes);
    fprintf(fp, "\t\t %d cells\n", nCells);
    fprintf(fp, "\t\t %d components: ", nodeComponentNb);
    for (int i = 0; i < nodeComponentNb; i++)
    {
        fprintf(fp, "%d ", nodeComponents[i]);
    }
    fprintf(fp, "\n");

    for (int i = 0; i < nodeComponentNb; i++)
    {
        int veclen = nodeComponents[i];
        //delete: float* dptr = nodeComponentDataPtrs[i];
        for (int k = 0; k < veclen; k++)
        {

            //delete: fprintf(fp, "\t\t channel %d[%d]: %g %g ... %g\n",
            //	      i, k,
            //      dptr[0*veclen+k],
            //      dptr[1*veclen+k],
            //      dptr[(nNodes-1)*veclen+k]);
            fprintf(fp, "\t\t channel %d[%d]: %g %g ... %g\n",
                    i, k,
                    //delete: dptr[0*veclen+k],
                    //delete: dptr[1*veclen+k],
                    //delete: dptr[(nNodes-1)*veclen+k]
                    *(nodeComponentDataPtrs[i].ptrs[k] + 0 * nodeComponentDataPtrs[i].stride),
                    *(nodeComponentDataPtrs[i].ptrs[k] + 1 * nodeComponentDataPtrs[i].stride),
                    *(nodeComponentDataPtrs[i].ptrs[k] + (nNodes - 1) * nodeComponentDataPtrs[i].stride));
        }
    }

    vec3 xyz0, xyz1, xyzLast;
    getCoords(0, xyz0); // no support for rotating zones (not necessary? TODO)
    getCoords(1, xyz1); // no support for rotating zones (not necessary? TODO)
    getCoords(nNodes - 1, xyzLast); // no support for rotating zones (not necessary? TODO)

    //delete: fprintf(fp, "\t\t x coords: %g %g ... %g\n", x[0], x[1], x[nNodes-1]);
    //delete: fprintf(fp, "\t\t y coords: %g %g ... %g\n", y[0], y[1], y[nNodes-1]);
    //delete: fprintf(fp, "\t\t z coords: %g %g ... %g\n", z[0], z[1], x[nNodes-1]);
    fprintf(fp, "\t\t x coords: %g %g ... %g\n", xyz0[0], xyz1[0], xyzLast[0]);
    fprintf(fp, "\t\t y coords: %g %g ... %g\n", xyz0[1], xyz1[1], xyzLast[1]);
    fprintf(fp, "\t\t z coords: %g %g ... %g\n", xyz0[2], xyz1[2], xyzLast[2]);

    fprintf(fp, "\t\t cell types: %d %d ... %d\n",
            cellType[0],
            cellType[1],
            cellType[nCells - 1]);

    int nodeListSize = getNodeListSize();
    fprintf(fp, "\t\t node list: %d %d ... %d\n",
            nodeList[0],
            nodeList[1],
            nodeList[nodeListSize - 1]);

    fprintf(fp, "\t\t node list offsets: %d %d ... %d\n",
            nodeListOffset[0],
            nodeListOffset[1],
            nodeListOffset[nCells - 1]);

    fprintf(fp, "\n");
    fprintf(fp, "\t\t ---------------------------------------\n");
}

bool Unstructured::saveAs(const char *fileName)
{
    FILE *fp = fopen(fileName, "wb");
    if (!fp)
    {
        fprintf(stderr, "Cannot write %s\n", fileName);
        return false;
    }

    fwrite(name, sizeof(char), 160, fp);
    fwrite(&nCells, sizeof(int), 1, fp);
    fwrite(&nNodes, sizeof(int), 1, fp);

    // Components
    fwrite(&nodeComponentNb, sizeof(int), 1, fp);
    fwrite(nodeComponents, sizeof(int), nodeComponentNb, fp);
    for (int l = 0; l < nodeComponentNb; l++)
    {
        if (nodeComponentLabels)
        {
            fwrite(nodeComponentLabels[l], sizeof(char), 256, fp); // 256: identical to constructor! TODO: constant
        }
        else
        {
            char buf[256];
            sprintf(buf, "%d", l);
            fwrite(buf, sizeof(char), 256, fp); // 256: identical to constructor! TODO: constant
        }
    }

    // Coordinates (write in non-interleaved format)
    //delete: fwrite(x, sizeof(float), nNodes, fp);
    //delete: fwrite(y, sizeof(float), nNodes, fp);
    //delete: fwrite(z, sizeof(float), nNodes, fp);
    for (int comp = 0; comp < 3; comp++)
    {
        for (int n = 0; n < nNodes; n++)
        {
            vec3 xyz;
            getCoords(n, xyz); // rotating zones are saved in position at time=0
            fvec3 fxyz;
            vec3tofvec3(xyz, fxyz);
            fwrite(&fxyz[comp], sizeof(float), 1, fp);
        }
    }

    // Data (write in interleaved format)
    for (int i = 0; i < nodeComponentNb; i++)
    {
        int veclen = nodeComponents[i];
        //delete: float* dptr = nodeComponentDataPtrs[i];
        //delete: fwrite(dptr, sizeof(float), nNodes*veclen, fp);
        if (veclen <= 1)
        {
            // scalar -> no interleaving possible
            fwrite(nodeComponentDataPtrs[i].ptrs[0], sizeof(float), nNodes, fp);
        }
        else
        {
            if (nodeComponentDataPtrs[i].stride == 1)
            {
                // not interleaved -> write interleaved
                for (int nd = 0; nd < nNodes; nd++)
                {
                    for (int vv = 0; vv < veclen; vv++)
                    {
                        fwrite(nodeComponentDataPtrs[i].ptrs[vv] + nd * nodeComponentDataPtrs[i].stride,
                               sizeof(float), 1, fp);
                    }
                }
            }
            else
            {
                // already interleaved TODO: is this assumption correct?
                fwrite(nodeComponentDataPtrs[i].ptrs[0], sizeof(float), nNodes * veclen, fp);
            }
        }
    }

    // Cells
    fwrite(cellType, sizeof(int), nCells, fp);
    int nodeListSize = getNodeListSize();
    fwrite(nodeList, sizeof(int), nodeListSize, fp);
    fwrite(nodeListOffset, sizeof(int), nCells, fp);

    // ##### BUG?
    divideVelocityByWalldist = false;
    extrapolateToNoSlipBoundary = false;

    fclose(fp);

    return true;
}

// obsolete ... but still an example of how to use Unstructured
void Unstructured::gradMulVec(int quantComp, Unstructured *uvec, int vecComp,
                              Unstructured *out, int outComp)
{ // gradient of quantity component times vector component, into out component
    // TODO:
    // - test if output component != input component

    if (getNodeCompVecLen(quantComp) == 1)
    {
        // scalar quantity

        // unfinished: TODO: compute calar gradient
        printf("Unstructured::gradMulVec: scalar gradient not yet implemented\n");
    }
    else
    {
        // vector quantity

        DataDesc *gradD = findNodeCompExtraData(quantComp,
                                                // TODO: find a non-nasty solution for:
                                                //(int) gradient(0, true)
                                                OP_GRADIENT);
        if (!gradD)
            gradD = gradient(quantComp);

        fmat3 *grad = (fmat3 *)gradD->p.f;

        uvec->selectVectorNodeData(vecComp);
        out->selectVectorNodeData(outComp);

        for (int n = 0; n < nNodes; n++)
        {
            vec3 v;
            uvec->getVector3(n, v);

            mat3 j;
            fmat3tomat3(grad[n], j);

            vec3 res;
            mat3vec(j, v, res);
            out->setVector3(n, res);
        }

        if (strictNodeCompExtraDeletion)
            deleteNodeCompExtraData(gradD);
    }
}
