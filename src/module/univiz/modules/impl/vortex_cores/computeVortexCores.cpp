/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Compute Vortex Core Lines according to Parallel Vectors
//
//  Ronald Peikert, Martin Roth, Dirk Bauer <=2005 and Filip Sadlo >=2006
//  Computer Graphics Laboratory, ETH Zurich

#include <string.h>
#include <stdlib.h>
#include "linalg.h"
#include "computeVortexCores.h"

static int cellNNodes[4] = { 4, 5, 6, 8 }; // tet, prism, pyr, hex
static int cellNEdges[4] = { 6, 8, 9, 12 }; // tet, prism, pyr, hex
static int cellNFaces[4] = { 4, 5, 5, 6 }; // tet, prism, pyr, hex

static int cellFace[4][6][4] = { // [cellTypeIdx][face][node]
    // cellTypeIdx: 0-tet 1-prism 2-pyr 3-hex
    // nodeId -1 if node not existing (in case of triangle face)

    // tet (in AVS order)
    { { 1, 2, 3, -1 },
      { 0, 2, 1, -1 },
      { 0, 3, 2, -1 },
      { 0, 1, 3, -1 } },

    // pyramid (in AVS order)
    { { 1, 2, 3, 4 },
      { 0, 2, 1, -1 },
      { 0, 3, 2, -1 },
      { 0, 4, 3, -1 },
      { 0, 1, 4, -1 } },

    // prism (in AVS order)
    { { 3, 4, 5, -1 },
      { 0, 2, 1, -1 },
      { 0, 1, 4, 3 },
      { 1, 2, 5, 4 },
      { 2, 0, 3, 5 } },

    // hex (not in AVS order)
    {
      { 0, 1, 3, 2 },
      { 7, 6, 4, 5 }, // changed orientation, this was orig
      //{ 4, 6, 7, 5 },
      { 0, 2, 6, 4 },
      { 7, 5, 1, 3 }, // changed orientation, this was orig
      //{ 1, 5, 7, 3 },
      { 0, 4, 5, 1 },
      { 7, 3, 2, 6 } // changed orientation, this was orig
      //{ 2, 3, 7, 6 }
    }
};

static int cellEdge[4][12][2] = { // [cellTypeIdx][edge][node]
    // cellTypeIdx: 0-tet 1-prism 2-pyr 3-hex
    // nodeId -1 if edge not existing

    // tet (in AVS order)
    { { 0, 1 },
      { 0, 2 },
      { 0, 3 },
      { 1, 2 },
      { 2, 3 },
      { 1, 3 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 } },

    // pryramid (in AVS order)
    { { 0, 1 },
      { 0, 2 },
      { 0, 3 },
      { 0, 4 },
      { 1, 2 },
      { 2, 3 },
      { 3, 4 },
      { 1, 4 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 } },

    // prism (in AVS order)
    { { 0, 1 },
      { 1, 2 },
      { 0, 2 },
      { 3, 4 },
      { 4, 5 },
      { 3, 5 },
      { 0, 3 },
      { 1, 4 },
      { 2, 5 },
      { -1, -1 },
      { -1, -1 },
      { -1, -1 } },

    // hex (not in AVS order)
    { { 0, 1 },
      { 0, 2 },
      { 0, 4 },
      { 2, 3 },
      { 1, 3 },
      { 1, 5 },
      { 4, 5 },
      { 4, 6 },
      { 2, 6 },
      { 6, 7 },
      { 5, 7 },
      { 3, 7 } }
};

#define PRINT_WARNINGS 0

inline double ABS(double x)
{
    return x < 0 ? -x : x;
}

inline double SGN(double x) { return x < 0 ? -1 : (x > 0 ? 1 : 0); }

int getCellTypeIdx(Unstructured *unst, int cell)
{
    switch (unst->getCellType(cell))
    {
    case Unstructured::CELL_TET:
        return 0;
    case Unstructured::CELL_PYR:
        return 1;
    case Unstructured::CELL_PRISM:
        return 2;
    case Unstructured::CELL_HEX:
        return 3;
    default:
    {
        fprintf(stderr, "cellTypeIdx: ERROR: unsupported cell type (returning index -1)\n");
        return -1;
    }
    }
}

bool getCellNodes(Unstructured *unst, int cell, int globalNode[8])
{
    int *cellNodes = unst->getCellNodesAVS(cell);

    switch (unst->getCellType(cell))
    {
    case Unstructured::CELL_TET:
    { // AVS order
        globalNode[0] = cellNodes[0];
        globalNode[1] = cellNodes[1];
        globalNode[2] = cellNodes[2];
        globalNode[3] = cellNodes[3];
    }
    break;
    case Unstructured::CELL_PYR:
    { // AVS order
        globalNode[0] = cellNodes[0];
        globalNode[1] = cellNodes[1];
        globalNode[2] = cellNodes[2];
        globalNode[3] = cellNodes[3];
        globalNode[4] = cellNodes[4];
    }
    break;
    case Unstructured::CELL_PRISM:
    { // AVS order
        globalNode[0] = cellNodes[0];
        globalNode[1] = cellNodes[1];
        globalNode[2] = cellNodes[2];
        globalNode[3] = cellNodes[3];
        globalNode[4] = cellNodes[4];
        globalNode[5] = cellNodes[5];
    }
    break;
    case Unstructured::CELL_HEX:
    {
        // TODO: change hex to AVS order too (adapt cellFace etc.)
        globalNode[4] = cellNodes[0];
        globalNode[5] = cellNodes[1];
        globalNode[7] = cellNodes[2];
        globalNode[6] = cellNodes[3];
        globalNode[0] = cellNodes[4];
        globalNode[1] = cellNodes[5];
        globalNode[3] = cellNodes[6];
        globalNode[2] = cellNodes[7];
    }
    break;
    default:
    {
        return false;
    }
    }
    return true;
}

UCD_connectivity *computeConnectivity(Unstructured *unst)
{
    int i, j;

    /* Initialize UCD connectivity structure */
    UCD_connectivity *ucdc = NEW0(UCD_connectivity, 1);
    ucdc->nEdges = 0;
    ucdc->firstEdge = NEW0(int, unst->nNodes + 1);
    ucdc->nFaces = 0;
    ucdc->firstFace = NEW0(int, unst->nNodes + 1);

    /* Allocate temporary storage */
    intPtr *otherNode = NEW0(intPtr, unst->nNodes + 1);
    intPtr *oppositeNode = NEW0(intPtr, unst->nNodes + 1);
    int *nEdges = NEW0(int, unst->nNodes + 1);
    int *nFaces = NEW0(int, unst->nNodes + 1);

    /* Loop over cells */
    for (int cell = 0; cell < unst->nCells; cell++)
    {
        int cellTypeIdx;
        int globalNode[8];
        int i, j;
        int nCellEdges;
        int nCellFaces;

        /* Collect global node numbers */
        cellTypeIdx = getCellTypeIdx(unst, cell);
        nCellEdges = cellNEdges[cellTypeIdx];
        nCellFaces = cellNFaces[cellTypeIdx];
        if (!getCellNodes(unst, cell, globalNode))
        {
            fprintf(stderr, "skipping unsupported cell type at cell %d\n", cell);
            continue;
        }

        /* Loop over edges in cell */
        for (i = 0; i < nCellEdges; i++)
        {
            int node[2], node0, node1;
            bool alreadyStored;

            node[0] = globalNode[cellEdge[cellTypeIdx][i][0]];
            node[1] = globalNode[cellEdge[cellTypeIdx][i][1]];

            if (node[0] < node[1])
            {
                node0 = node[0];
                node1 = node[1];
            }
            else
            {
                node0 = node[1];
                node1 = node[0];
            }

            /* Try to find in stored edges */
            alreadyStored = false;
            for (j = 0; j < nEdges[node0]; j++)
            {
                if (otherNode[node0][j] == node1)
                {
                    alreadyStored = true;
                }
            }
            /* Store the edge */
            if (!alreadyStored)
            {
                int n;

                ucdc->nEdges++;
                n = nEdges[node0]++;
                otherNode[node0] = (int *)realloc(otherNode[node0], (n + 1) * sizeof(int));
                otherNode[node0][n] = node1;
            }
        }

        /* Loop over faces in cell */
        for (i = 0; i < nCellFaces; i++)
        {
            int node[4], node0, node1, node2 = -1;
            int min;
            bool alreadyStored;

            bool triangle = cellFace[cellTypeIdx][i][3] < 0;

            node[0] = globalNode[cellFace[cellTypeIdx][i][0]];
            node[1] = globalNode[cellFace[cellTypeIdx][i][1]];
            node[2] = globalNode[cellFace[cellTypeIdx][i][2]];
            if (!triangle)
                node[3] = globalNode[cellFace[cellTypeIdx][i][3]];

            /* For addressing the face find its smallest node number */
            min = 0;
            if (node[1] < node[min])
                min = 1;
            if (node[2] < node[min])
                min = 2;
            if ((!triangle) && (node[3] < node[min]))
                min = 3;
            node0 = node[min];
            if (triangle)
            {
                // triangle -> use all three ordered nodes for identification
                int max = 0;
                if (node[1] > node[max])
                    max = 1;
                if (node[2] > node[max])
                    max = 2;

                int med = 3 - (min + max);

                node1 = node[med];
                node2 = node[max];
            }
            else
            {
                // quandrangle -> use also opposite node for identification
                node1 = node[min ^ 2];
            }

            /* Try to find in stored faces */
            alreadyStored = false;
            for (j = 0; j < nFaces[node0]; j++)
            {
                if (triangle)
                {
                    if ((oppositeNode[node0][j * 2 + 0] == node1) && (oppositeNode[node0][j * 2 + 1] == node2))
                    {
                        alreadyStored = true;
                    }
                }
                else
                {
                    if (oppositeNode[node0][j * 2 + 0] == node1)
                    {
                        alreadyStored = true;
                    }
                }
            }

            /* Store the face */
            if (!alreadyStored)
            {
                int n;

                ucdc->nFaces++;
                n = nFaces[node0]++;

                if (triangle)
                {
                    oppositeNode[node0] = (int *)realloc(oppositeNode[node0], (n + 1) * 2 * sizeof(int));
                    oppositeNode[node0][n * 2 + 0] = node1;
                    oppositeNode[node0][n * 2 + 1] = node2;
                }
                else
                {
                    oppositeNode[node0] = (int *)realloc(oppositeNode[node0], (n + 1) * 2 * sizeof(int));
                    oppositeNode[node0][n * 2 + 0] = node1;
                    // second entry in oppositeNode -1 in case of quadrangle
                    oppositeNode[node0][n * 2 + 1] = -1;
                }
            }
        }
    }

    /* Convert temporary edge info to final edge info */
    ucdc->otherNode = NEW(int, ucdc->nEdges);
    int edge = 0;
    for (i = 0; i <= unst->nNodes; i++)
    {
        ucdc->firstEdge[i] = edge;
        for (j = 0; j < nEdges[i]; j++)
        {
            ucdc->otherNode[edge++] = otherNode[i][j];
        }
        if (otherNode[i])
            free(otherNode[i]);
    }

    /* Convert temporary face info to final face info */
    ucdc->oppositeNode = NEW(int, 2 * ucdc->nFaces);
    int face = 0;
    for (i = 0; i <= unst->nNodes; i++)
    {
        ucdc->firstFace[i] = face;
        for (j = 0; j < nFaces[i]; j++)
        {
            if (oppositeNode[i][j * 2 + 1] >= 0)
            { // triangle
                ucdc->oppositeNode[face * 2 + 0] = oppositeNode[i][j * 2 + 0];
                ucdc->oppositeNode[face * 2 + 1] = oppositeNode[i][j * 2 + 1];
                face++;
            }
            else
            { // quad
                ucdc->oppositeNode[face * 2 + 0] = oppositeNode[i][j * 2 + 0];
                ucdc->oppositeNode[face * 2 + 1] = -1; // unused
                face++;
            }
        }
        if (oppositeNode[i])
            free(oppositeNode[i]);
    }

    return ucdc;
}

/* Delete an UCD connectivity info */
void deleteUcdConnectivity(UCD_connectivity *ucdc)
{
    free(ucdc->firstEdge);
    free(ucdc->otherNode);
    free(ucdc->firstFace);
    free(ucdc->oppositeNode);
    free(ucdc);
}

/* Delete a vertex list */
void deleteVertexList(VertexList *vertexList)
{
    free(vertexList->elem);
    free(vertexList);
}

/* Delete a VC Polyline list */
void deletePolylineList(PolylineList *polylineList)
{
    free(polylineList->elem);
    free(polylineList);
}

/* Compute the gradient field for a (1-component, 3-vector) ucd */
float *computeGradient(Unstructured *unst, int compV, UCD_connectivity *ucdc)
{
    /* Gradient computation, repeat for each velocity component

       LSF per node: ansatz for relative coords & relative data:

          ax + by + cz = u
          err^2 = sum(ax[i] + by[i] + cz[i] - u[i])^2

          d/da err^2 = (..) * x[i] == 0
          d/db err^2 = (..) * y[i] == 0
          d/dc err^2 = (..) * z[i] == 0

    [ sum_xx sum_xy sum_xz ] [a]    [ sum_ux ]
    [ sum_xy sum_yy sum_yz ] [b] =  [ sum_uy ]
    [ sum_xz sum_yz sum_zz ] [c]    [ sum_uz ]

    Solve for a, b, c:   (a,b,c) = gradient of u.
    Repeat for v and w.

    Algorithm:
    (1) Accumulate for each node: sum_xx, sum_xy, ... , sum_wx
    (15 floats).
    (2) For each node solve 3 LSF problems -> grad(u), grad(v), grad(w).
    */

    LsfSums *lsfSums = NEW0(LsfSums, unst->nNodes);
    float *gradient = NEW(float, 9 * unst->nNodes);

    /* Loop over all edges of the ucd */
    int edge = 0;
    for (int node = 0; node < unst->nNodes; node++)
    {
        while (edge < ucdc->firstEdge[node + 1])
        {
            float x, y, z, u, v, w; /* Relative values */
            LsfSums *s0, *s1;

            int node1 = ucdc->otherNode[edge];

            vec3 n1Pos, nPos;
            unst->getCoords(node1, n1Pos);
            unst->getCoords(node, nPos);

            x = n1Pos[0] - nPos[0];
            y = n1Pos[1] - nPos[1];
            z = n1Pos[2] - nPos[2];

            vec3 n1Dat, nDat;
            unst->getVector3(node1, compV, n1Dat);
            unst->getVector3(node, compV, nDat);

            u = n1Dat[0] - nDat[0];
            v = n1Dat[1] - nDat[1];
            w = n1Dat[2] - nDat[2];

            s0 = &(lsfSums[node]);
            s1 = &(lsfSums[node1]);

            s0->mat00 += x * x;
            s1->mat00 += x * x;
            s0->mat01 += x * y;
            s1->mat01 += x * y;
            s0->mat02 += x * z;
            s1->mat02 += x * z;
            s0->mat11 += y * y;
            s1->mat11 += y * y;
            s0->mat12 += y * z;
            s1->mat12 += y * z;
            s0->mat22 += z * z;
            s1->mat22 += z * z;
            s0->rhs_u[0] += x * u;
            s1->rhs_u[0] += x * u;
            s0->rhs_u[1] += y * u;
            s1->rhs_u[1] += y * u;
            s0->rhs_u[2] += z * u;
            s1->rhs_u[2] += z * u;
            s0->rhs_v[0] += x * v;
            s1->rhs_v[0] += x * v;
            s0->rhs_v[1] += y * v;
            s1->rhs_v[1] += y * v;
            s0->rhs_v[2] += z * v;
            s1->rhs_v[2] += z * v;
            s0->rhs_w[0] += x * w;
            s1->rhs_w[0] += x * w;
            s0->rhs_w[1] += y * w;
            s1->rhs_w[1] += y * w;
            s0->rhs_w[2] += z * w;
            s1->rhs_w[2] += z * w;

            edge++;
        }
    }
    /* Compute gradients at nodes */
    float *gradPtr = gradient;
    for (int node = 0; node < unst->nNodes; node++)
    {
        LsfSums *s = &(lsfSums[node]);
        fmat3 a;
        a[0][0] = s->mat00;
        a[0][1] = s->mat01;
        a[0][2] = s->mat02;
        a[1][0] = s->mat01;
        a[1][1] = s->mat11;
        a[1][2] = s->mat12;
        a[2][0] = s->mat02;
        a[2][1] = s->mat12;
        a[2][2] = s->mat22;
        fvec3solve(a, s->rhs_u, gradPtr);
        gradPtr += 3;
        fvec3solve(a, s->rhs_v, gradPtr);
        gradPtr += 3;
        fvec3solve(a, s->rhs_w, gradPtr);
        gradPtr += 3;
    }
    free(lsfSums);

    return gradient;
}

#ifdef BACKUP
float *computeVorticity(float *gradient, int nNodes)
{
    float *vorticity = NEW(float, 3 * nNodes);
    float *rptr = gradient;
    float *wptr = vorticity;

    for (int i = 0; i < nNodes; i++)
    {
        *wptr++ = rptr[2 * 3 + 1] - rptr[1 * 3 + 2];
        *wptr++ = rptr[0 * 3 + 2] - rptr[2 * 3 + 0];
        *wptr++ = rptr[1 * 3 + 0] - rptr[0 * 3 + 1];
        rptr += 9;
    }

    return vorticity;
}
#endif

float *computeVorticity(float *gradient, int nNodes)
{
    fvec3 *v = NEW(fvec3, nNodes);
    fmat3 *g = (fmat3 *)gradient;

    for (int i = 0; i < nNodes; i++)
    {
        fmat3omega(g[i], v[i]);
    }

    return (float *)v;
}

float *computeAcceleration(float *gradient, Unstructured *unst, int compV)
{
    fvec3 *a = NEW(fvec3, unst->nNodes);
    fmat3 *g = (fmat3 *)gradient;

    for (int i = 0; i < unst->nNodes; i++)
    {
        fvec3 v;
        unst->getVector3(i, compV, v);
        fmat3vec(g[i], v, a[i]);
    }

    return (float *)a;
}

float computeExtent(Unstructured *unst)
{
    fvec3 min, max;
    min[0] = min[1] = min[2] = BIGFLOAT;
    max[0] = max[1] = max[2] = -BIGFLOAT;

    for (int i = 0; i < unst->nNodes; i++)
    {

        vec3 pos;
        unst->getCoords(i, pos);

        if (pos[0] < min[0])
            min[0] = pos[0];
        if (pos[1] < min[1])
            min[1] = pos[1];
        if (pos[2] < min[2])
            min[2] = pos[2];
        if (pos[0] > max[0])
            max[0] = pos[0];
        if (pos[1] > max[1])
            max[1] = pos[1];
        if (pos[2] > max[2])
            max[2] = pos[2];
    }

    // disabled because this changes input port data, sadlo 2006
    //UCDstructure_set_extent(ucd, min, max);

    float extent = 0.0;
    if (max[0] - min[0] > extent)
        extent = max[0] - min[0];
    if (max[1] - min[1] > extent)
        extent = max[1] - min[1];
    if (max[2] - min[2] > extent)
        extent = max[2] - min[2];

    return extent;
}

void markIntersectedEdges(Unstructured *unst, UCD_connectivity *ucdc,
                          int compV /*fvec3 *v*/, fvec3 *w, unsigned char *flags)
{
    int edge = 0;
    for (int node = 0; node < unst->nNodes; node++)
    {
        while (edge < ucdc->firstEdge[node + 1])
        {
            fvec3 dv, dw /*, c0, c1*/;
            fvec3 v0, w0;
            fvec3 p, q, r; /* coefficients of quadratic form */
            /* c = p t^2 + q t + r            */
            fvec3 tmp;

            int node1 = ucdc->otherNode[edge];

            fvec3 v1, v;
            unst->getVector3(node1, compV, v1);
            unst->getVector3(node, compV, v);

            fvec3sub(v1, v, dv);
            fvec3sub(w[node1], w[node], dw);

            fvec3copy(v, v0);
            fvec3copy(w[node], w0);

            fvec3cross(dv, dw, p);
            fvec3cross(dv, w0, q);
            fvec3cross(v0, dw, tmp);
            fvec3add(q, tmp, q);
            fvec3cross(v0, w0, r);

            /* Test for single intesections: opposite signs of c(0), c(1) */
            if (r[0] * (p[0] + q[0] + r[0]) < 0)
                flags[edge] |= 1;
            if (r[1] * (p[1] + q[1] + r[1]) < 0)
                flags[edge] |= 2;
            if (r[2] * (p[2] + q[2] + r[2]) < 0)
                flags[edge] |= 4;

            /* Test for double intersections:               */
            /* opposite signs of c(0) and c(t)              */
            /* where t is an extremum of c with 0 <= t <= 1 */

            if (p[0] != 0)
            {
                double t = -q[0] / (2 * p[0]);
                if (t >= 0 && t <= 1 && r[0] * ((p[0] * t + q[0]) * t + r[0]) < 0)
                    flags[edge] |= 1;
            }

            if (p[1] != 0)
            {
                double t = -q[1] / (2 * p[1]);
                if (t >= 0 && t <= 1 && r[1] * ((p[1] * t + q[1]) * t + r[1]) < 0)
                    flags[edge] |= 2;
            }

            if (p[2] != 0)
            {
                double t = -q[2] / (2 * p[2]);
                if (t >= 0 && t <= 1 && r[2] * ((p[2] * t + q[2]) * t + r[2]) < 0)
                    flags[edge] |= 4;
            }

            edge++;
        }
    }
}

#ifdef CHEAP
void markIntersectedEdges(Unstructured *unst, UCD_connectivity *ucdc,
                          fvec3 *v, fvec3 *w, unsigned char *flags)
{
    int edge;
    int node, node1;

    edge = 0;
    for (node = 0; node < unst->nNodes; node++)
    {
        while (edge < ucdc->firstEdge[node + 1])
        {
            vec3 c0, c1;

            node1 = ucdc->otherNode[edge];

            vec3cross(v[node], w[node], c0);
            vec3cross(v[node1], w[node1], c1);

            if (c0[0] * c1[0] < 0)
                flags[edge] |= 1;
            if (c0[1] * c1[1] < 0)
                flags[edge] |= 2;
            if (c0[2] * c1[2] < 0)
                flags[edge] |= 4;

            edge++;
        }
    }
}
#endif

int findParallelInTriangle(vec3 v0, vec3 v1, vec3 v2, vec3 w0, vec3 w1, vec3 w2,
                           double *s, double *t, double *lambda)
{
    vec3 v01, v02;
    vec3 w01, w02;
    mat3 V, Vinv;
    mat3 W, Winv;
    mat3 X;

    double eigenvalues[3];
    vec3 eigenvectors[3];

    double detV, detW;
    double absdetV, absdetW, absdetmax;
    double nx, ny, nz;
    double ss, tt;
    int numParal;
    int numEigen;
    int i, ok, take;

    /* The vectors v0->v1 and v0->v2 span the triangle.        */
    /* The vectors v0,v01,v02 are the columns of the V matrix. */

    vec3sub(v1, v0, v01);
    vec3sub(v2, v0, v02);
    vec3sub(w1, w0, w01);
    vec3sub(w2, w0, w02);

    mat3setcols(V, v0, v01, v02);
    mat3setcols(W, w0, w01, w02);

    detW = mat3det(W);
    detV = mat3det(V);

    absdetW = fabs(detW);
    absdetV = fabs(detV);

    /* Take the matrix of larger determinant.  */
    /* At least one matrix must be invertible! */

    take = 0;
    absdetmax = 0.0;

    if (absdetW > absdetmax)
        take = 1;
    if (absdetV > absdetmax)
        take = 2;

    switch (take)
    {
    case 0:
        return 0;
    case 1:
        mat3invdet(W, detW, Winv);
        mat3mul(Winv, V, X);
        break;
    case 2:
        mat3invdet(V, detV, Vinv);
        mat3mul(Vinv, W, X);
        break;
    }

    numParal = 0;
    numEigen = mat3eigenvalues(X, eigenvalues);

    for (i = 0; i < numEigen; i++)
    {
        ok = mat3realEigenvector(X, eigenvalues[i], eigenvectors[i]);

        if (take == 2)
        {
            if (eigenvalues[i] == 0.0)
                ok = 0;
            else
                eigenvalues[i] = 1.0 / eigenvalues[i];
        }

        if (ok)
        {
            /* Scale the normed eigenvector (nx,ny,nz) to length (1,s,t) */

            nx = eigenvectors[i][0];
            ny = eigenvectors[i][1];
            nz = eigenvectors[i][2];

            if (nx != 0.0)
            {
                ss = ny / nx;
                tt = nz / nx;

                /* The parallel point must be inside the triangle */

                if (ss >= 0 && tt >= 0 && ss + tt <= 1)
                {
                    s[numParal] = ss;
                    t[numParal] = tt;
                    lambda[numParal] = eigenvalues[i];
                    numParal++;
                }
            }
        } /* ok */
    } /* next i */

    return numParal;
}

int findParallelInQuadN(vec3 vCorner[4], vec3 wCorner[4], double s[12], double t[12], vec3 dir[12],
                        double lambda[12])
{
    /* Start searching from several seed positions */
    for (int i = 0; i < 1; i++)
    { // Find vertices in a triangle
        // TODO: i < 5
        double S = .5;
        double T = .5;

        vec2 corr;
        vec3 v, w, c;

        for (int iter = 0; iter < 5; iter++)
        {
            vec3 tmp;
            vec3 vs, ws, cs, vt, wt, ct;

            /* Evaluate v(s,t) and w(s,t) */
            vec3bilint(vCorner[0], vCorner[1], vCorner[3], vCorner[2], S, T, v);
            vec3bilint(wCorner[0], wCorner[1], wCorner[3], wCorner[2], S, T, w);

            /* Evaluate v x w */
            vec3cross(v, w, c);

            /* Evaluate d/ds v(s,t) */
            vec3lint(vCorner[1], vCorner[2], T, vs);
            vec3lint(vCorner[0], vCorner[3], T, tmp);
            vec3sub(vs, tmp, vs);

            /* Evaluate d/ds w(s,t) */
            vec3lint(wCorner[1], wCorner[2], T, ws);
            vec3lint(wCorner[0], wCorner[3], T, tmp);
            vec3sub(ws, tmp, ws);

            /* Evaluate d/ds (v x w) */
            vec3cross(vs, w, cs);
            vec3cross(v, ws, tmp);
            vec3add(cs, tmp, cs);

            /* Evaluate d/dt v(s,t) */
            vec3lint(vCorner[3], vCorner[2], S, vt);
            vec3lint(vCorner[0], vCorner[1], S, tmp);
            vec3sub(vt, tmp, vt);

            /* Evaluate d/dt w(s,t) */
            vec3lint(wCorner[3], wCorner[2], S, wt);
            vec3lint(wCorner[0], wCorner[1], S, tmp);
            vec3sub(wt, tmp, wt);

            /* Evaluate d/dt (v x w) */
            vec3cross(vt, w, ct);
            vec3cross(v, wt, tmp);
            vec3add(ct, tmp, ct);

            /* Set up transpose Jacobian */
            mat3 JcT;
            vec3copy(cs, JcT[0]);
            vec3copy(ct, JcT[1]);

            /* Newton method:

                (s_new, t_new) = (s,t) - inv(Jc) * c(s,t)

                Jc * (s_new - s, t_new - t) = -c(s,t)

                Jc * corr = -c(s,t)

                tp(Jc) * Jc * corr = - tp(Jc) * c(s,t)
            */

            /* Multiply with its transpose */
            mat2 J2, J2i;
            mat23MMT(JcT, J2);
            double det = mat2det(J2);
            if (det == 0)
                break;
            mat2invdet(J2, det, J2i);

            /* Multiply rhs with transpose */
            vec2 p;
            mat23vec(JcT, c, p);
            mat2vec(J2i, p, corr);

            S -= corr[0];
            T -= corr[1];

            /* if (S < 0 || S > 1 || T < 0 || T > 1) break; */
        }

        /* Verify that solution is inside the face */
        if (S < 0 || S > 1 || T < 0 || T > 1)
            continue;
        /* Cell has been left */

        /* Verify parallelism */
        if (corr[0] * corr[0] + corr[1] * corr[1] > 1e-8)
            continue;
        /* No convergence */

        /* Compute square of sine of angle between the two vectors */
        if (vec3sqr(c) / (vec3sqr(v) * vec3sqr(w)) > 1e-6)
            continue;
        /* Not a solution */

        s[0] = S;
        t[0] = T;
        vec3copy(v, dir[0]);

        /* Calculate lambda */

        double w0 = ABS(w[0]);
        double w1 = ABS(w[1]);
        double w2 = ABS(w[2]);
        int index = 0;
        double wmax = w0;
        if (w1 > wmax)
        {
            index = 1;
            wmax = w1;
        }
        if (w2 > wmax)
        {
            index = 2;
            wmax = w2;
        }
        if (wmax == 0.)
        {
            lambda[0] = (v[0] > 0) ? 1e19 : -1e19;
        }
        else
            lambda[0] = v[index] / w[index];
        return 1; // TODO: multiple vertices
    }
    return 0;
}

int findParallelInQuadT(vec3 vCorner[4], vec3 wCorner[4],
                        double *s, double *t, vec3 *dir, double *lambda)
{
    vec3 vMid, wMid;
    int i;
    int n;

    vec3avg4(vCorner[0], vCorner[1], vCorner[2], vCorner[3], vMid);
    vec3avg4(wCorner[0], wCorner[1], wCorner[2], wCorner[3], wMid);

    n = 0;
    for (i = 0; i < 4; i++)
    { /* Find vertices in a triangle */
        int j, ni;
        double S[3], T[3];
        int k;

        j = (i + 1) % 4;

        ni = findParallelInTriangle(vMid, vCorner[i], vCorner[j],
                                    wMid, wCorner[i], wCorner[j], S, T, lambda);

        for (k = 0; k < ni; k++)
        {
            switch (i)
            {
            case 0:
                *s = (1. - S[k] + T[k]) / 2.;
                *t = (1. - S[k] - T[k]) / 2.;
                break;
            case 1:
                *s = (1. + S[k] + T[k]) / 2.;
                *t = (1. - S[k] + T[k]) / 2.;
                break;
            case 2:
                *s = (1. + S[k] - T[k]) / 2.;
                *t = (1. + S[k] + T[k]) / 2.;
                break;
            case 3:
                *s = (1. - S[k] - T[k]) / 2.;
                *t = (1. + S[k] - T[k]) / 2.;
                break;
            }

            vec3bilint(vCorner[0], vCorner[1], vCorner[3], vCorner[2],
                       *s, *t, dir[k]);
            s++;
            t++;
        }

        n += ni;
        dir += ni;
        lambda += ni;
    }

    return n;
}

int findParallelInTria(vec3 vCorner[3], vec3 wCorner[3],
                       double *s, double *t, vec3 *dir, double *lambda)
{
    int ni;
    double S[3], T[3];
    int k;

    ni = findParallelInTriangle(vCorner[0], vCorner[1], vCorner[2],
                                wCorner[0], wCorner[1], wCorner[2], S, T, lambda);

    for (k = 0; k < ni; k++)
    {
        // S and T are local coordinates in triangle 0,1,2 with origin at 0
        *s = S[k];
        *t = T[k];

        vec3lerp3(vCorner[0], vCorner[1], vCorner[2], *s, *t, dir[k]);
        s++;
        t++;
    }

    return ni;
}

bool degenerateCell(int globalNode[8], int cellTypeIdx,
                    Unstructured *unst, int compV, fvec3 *w)
{
    int i;
    int nDegenerateV;
    int nDegenerateW;

    nDegenerateV = 0;
    nDegenerateW = 0;

    for (i = 0; i < cellNNodes[cellTypeIdx]; i++)
    {
        int node;
        node = globalNode[i];

        fvec3 v;
        unst->getVector3(node, compV, v);

        if (fvec3iszero(v))
            nDegenerateV++;
        if (fvec3iszero(w[node]))
            nDegenerateW++;
    }

    return (nDegenerateV >= 1 || nDegenerateW >= 1);
}

void dumpCell(int globalNode[8], int cellTypeIdx, Unstructured *unst, int compV, fvec3 *w)
{
    int i;

    for (i = 0; i < cellNNodes[cellTypeIdx]; i++)
    {
        int node;
        node = globalNode[i];

        fvec3 v;
        unst->getVector3(node, compV, v);

        printf("%12g%12g%12g %12g%12g%12g\n",
               v[0], v[1], v[2],
               w[node][0], w[node][1], w[node][2]);
    }
}

int intersectCell(int globalNode[8], int cellTypeIdx,
                  Unstructured *unst, UCD_connectivity *ucdc,
                  int compV /*fvec3 *v*/, fvec3 *w, fmat3 *grad,
                  signed char *nVerticesOnFace, int *firstVertex, unsigned char *flags,
                  bool quickMode,
                  VertexList *vertexList, int *verticesInCell, int variant, float extent)
{
    int nVerticesInCell;
    int i;

    nVerticesInCell = 0;

    /* Loop over faces: Find intersection points */
    for (i = 0; i < cellNFaces[cellTypeIdx]; i++)
    {
        int node[4];
        int face;
        //int vertexNr;
        int k;
        int min, minNode, oppNode1, oppNode2 = -1;

        bool triangle = cellFace[cellTypeIdx][i][3] < 0;

        node[0] = globalNode[cellFace[cellTypeIdx][i][0]];
        node[1] = globalNode[cellFace[cellTypeIdx][i][1]];
        node[2] = globalNode[cellFace[cellTypeIdx][i][2]];
        if (!triangle)
            node[3] = globalNode[cellFace[cellTypeIdx][i][3]];

        /* printf("\nface #%d: [%d %d %d %d]\n",
        i, node[0], node[1], node[2], node[3]); */

        /* Lookup face containing node[0..3] */
        min = 0;
        if (node[1] < node[min])
            min = 1;
        if (node[2] < node[min])
            min = 2;
        if ((!triangle) && (node[3] < node[min]))
            min = 3;
        minNode = node[min];
        if (triangle)
        {
            // triangle -> use all three ordered nodes for identification
            int max = 0;
            if (node[1] > node[max])
                max = 1;
            if (node[2] > node[max])
                max = 2;

            int med = 3 - (min + max);

            oppNode1 = node[med];
            oppNode2 = node[max];
        }
        else
        {
            // quandrangle -> use also opposite node for identification
            oppNode1 = node[min ^ 2];
        }

        face = ucdc->firstFace[minNode];
        if (triangle)
        { // tria
            while ((ucdc->oppositeNode[face * 2 + 0] != oppNode1) || (ucdc->oppositeNode[face * 2 + 1] != oppNode2))
                face++;
        }
        else
        { // quad
            while (ucdc->oppositeNode[face * 2 + 0] != oppNode1)
                face++;
        }

        /* Face not treated before or           */
        /* Quickmode did not find intersections */
        if (nVerticesOnFace[face] == -1 || (nVerticesOnFace[face] == 0 && !quickMode))
        {
            int j;
            int edge;
            int n;
            double s[12];
            double t[12];
            double lambda[12];
            vec3 dir[12];
            vec3 xyz[4];
            vec3 vCorner[4];
            vec3 wCorner[4];
            vec3 ex, ey, ez;
            vec3 cx, cy, cz;
            vec3 mSpan, mUnit;
            vec3 nSpan, nUnit;
            vec3 Jm, Jn;
            mat3 g[4];
            mat3 Jacobian;
            double mag, cxmag, cymag, czmag;
            double a, b, c, d;
            double trace, det, discr;
            int take;

            firstVertex[face] = vertexList->nElems;

            /* Check all edges for intersections */
            if (quickMode)
            {
                int edgeFlags;

                edgeFlags = 0;
                for (j = 0; j < (triangle ? 3 : 4); j++)
                {
                    int node0, node1;

                    /* Lookup edge (node0, node1) */
                    node0 = node[j];
                    if (triangle)
                        node1 = node[(j + 1) % 3];
                    else
                        node1 = node[(j + 1) % 4];

                    if (node0 > node1)
                    {
                        int h;
                        h = node0;
                        node0 = node1;
                        node1 = h;
                    }

                    edge = ucdc->firstEdge[node0];
                    while (ucdc->otherNode[edge] != node1)
                        edge++;
                    edgeFlags |= flags[edge];
                }

                if (edgeFlags != ALL_INTERSECTED)
                    continue;
                /* reject face */
            }

            /* Collect corner data and convert to double */
            for (j = 0; j < (triangle ? 3 : 4); j++)
            {
                //fvec3tovec3(v[node[j]], vCorner[j]);
                fvec3 v;
                unst->getVector3(node[j], compV, v);
                fvec3tovec3(v, vCorner[j]);
                fvec3tovec3(w[node[j]], wCorner[j]);
            }

            /* Find vertices on the face */
            if (triangle)
            {
                n = findParallelInTria(vCorner, wCorner, s, t, dir, lambda);
            }
            else
            { // quad
                if (variant == QUAD_NEWTON)
                    n = findParallelInQuadN(vCorner, wCorner, s, t, dir, lambda);
                else
                    n = findParallelInQuadT(vCorner, wCorner, s, t, dir, lambda);
            }
            /* printf("%d vertices.\n", n); */

            if (n > 0)
            {
                //vec3 tmp;

                /* Collect corner coords */
                vec3 n0, n1, n2, n3;
                vec3zero(n3);
                unst->getCoords(node[0], n0);
                unst->getCoords(node[1], n1);
                unst->getCoords(node[2], n2);
                if (!triangle)
                    unst->getCoords(node[3], n3);
                vec3set(xyz[0], n0[0], n0[1], n0[2]);
                vec3set(xyz[1], n1[0], n1[1], n1[2]);
                vec3set(xyz[2], n2[0], n2[1], n2[2]);
                if (!triangle)
                    vec3set(xyz[3], n3[0], n3[1], n3[2]);
            }

            for (k = 0; k < n; k++)
            {
                Vertex *vert;
                vec3 tmp;

                /* Create a new vertex */
                vertexList->nElems++;
                vertexList->elem = (Vertex *)realloc(vertexList->elem, vertexList->nElems * sizeof(Vertex));
                vert = &(vertexList->elem[vertexList->nElems - 1]);

                /* Compute coords of the vertex */
                /* Convert to floats */

                if (triangle)
                    vec3lerp3(xyz[0], xyz[1], xyz[2], s[k], t[k], tmp);
                else
                    vec3bilint(xyz[0], xyz[1], xyz[3], xyz[2], s[k], t[k], tmp);
                vec3tofvec3(tmp, vert->xyz);

                /* Compute attributes of the vertex */

                vec3tofvec3(dir[k], vert->velo);
                vert->lambda = lambda[k];
                vert->linkF = VC_VOID;
                vert->linkB = VC_VOID;
                vert->cell1 = VC_VOID;
                vert->cell2 = VC_VOID;
                vert->used = false;
                vert->strength = 0.0;

                /* Project the flow on 2D, using J         */
                /* then compute eigenvalues:               */
                /* Get the Jacobian at the corner points   */
                /* and interpolate it at position (s,t)    */

                fmat3tomat3(grad[node[0]], g[0]);
                fmat3tomat3(grad[node[1]], g[1]);
                fmat3tomat3(grad[node[2]], g[2]);
                if (!triangle)
                    fmat3tomat3(grad[node[3]], g[3]);

                if (triangle)
                    mat3lerp3(g[0], g[1], g[2], s[k], t[k], Jacobian);
                else
                    mat3bilint(g[0], g[1], g[2], g[3], s[k], t[k], Jacobian);

                /* Create the m and n span vectors                   */
                /* to span up a plane perpendicular to the core line */
                /* dir[k] is the normal vector of this plane         */

                vec3set(ex, 1, 0, 0);
                vec3set(ey, 0, 1, 0);
                vec3set(ez, 0, 0, 1);

                vec3cross(dir[k], ex, cx);
                vec3cross(dir[k], ey, cy);
                vec3cross(dir[k], ez, cz);

                cxmag = vec3sqr(cx);
                cymag = vec3sqr(cy);
                czmag = vec3sqr(cz);

                /* Take the cross product of largest value */
                /* to avoid that it is a null vector       */

                mag = 0.0;
                take = 0;
                if (cxmag > mag)
                {
                    mag = cxmag;
                    take = 1;
                }
                if (cymag > mag)
                {
                    mag = cymag;
                    take = 2;
                }
                if (czmag > mag)
                {
                    mag = czmag;
                    take = 3;
                }

                switch (take)
                {
                case 0:
                    vec3set(mSpan, 0, 0, 0);
                    break;
                case 1:
                    vec3copy(cx, mSpan);
                    break;
                case 2:
                    vec3copy(cy, mSpan);
                    break;
                case 3:
                    vec3copy(cz, mSpan);
                    break;
                }

                vec3cross(dir[k], mSpan, nSpan);

                /* Normalize the m and n span vectors            */
                /* Compute the directional derivatives Jm and Jn */
                /* Setup the projection matrix [a b][c d]        */

                vec3nrm(mSpan, mUnit);
                vec3nrm(nSpan, nUnit);

                mat3vec(Jacobian, mUnit, Jm);
                mat3vec(Jacobian, nUnit, Jn);

                a = vec3dot(mUnit, Jm);
                b = vec3dot(mUnit, Jn);
                c = vec3dot(nUnit, Jm);
                d = vec3dot(nUnit, Jn);

                /* Compute the eigenvalues of the projection matrix   */
                /* vortex strength = absolute value of imaginary part */

                trace = a + d;
                det = a * d - b * c;
                discr = trace * trace / 4 - det;

                if (discr < 0.0)
                    vert->strength = sqrt(-discr);

                /* Make vortex strength relative to speed and grid size */
                /* Its meaninig is now: revolutions per grid extent */

                vert->strength *= extent / (2 * M_PI * fvec3mag(vert->velo));
                /*
                               printf("\n");
                               mat3printf("Jacobian", Jacobian);
                               printf("\n");
                               vec3printf("cx", cx, "\t");  printf("cxmag = %f\n", cxmag);
                               vec3printf("cy", cy, "\t");  printf("cymag = %f\n", cymag);
                               vec3printf("cz", cz, "\t");  printf("czmag = %f\n", czmag);
                               printf("\n");
                               printf("take = %d \t mag = %f\n", take, mag);
                               printf("\n");
                               vec3printf("dir  ", dir[k], "\n");
                vec3printf("mSpan", mSpan,  "\t");
                vec3printf("mUnit", mUnit,  "\n");
                vec3printf("nSpan", nSpan,  "\t");
                vec3printf("nUnit", nUnit,  "\n");
                printf("\n");
                vec3printf("Jm", Jm, "\n");
                vec3printf("Jn", Jn, "\n");
                printf("\n");
                printf("a = %f \t b = %f\n", a, b);
                printf("c = %f \t d = %f\n", c, d);
                printf("\n");
                printf("trace = %f \t det = %f\n", trace, det);
                printf("discr = %f\n", discr);
                printf("\n");
                printf("vortex strength = %f\n", vert->strength);
                */
            } /* next k */

            nVerticesOnFace[face] = vertexList->nElems - firstVertex[face];

        } /* end if */

        /* Add all face vertices to the current hex */
        for (k = 0; k < nVerticesOnFace[face]; k++)
            verticesInCell[nVerticesInCell++] = firstVertex[face] + k;

    } /* next i */

    return nVerticesInCell;
}

/* Find the places in a UCD where two vectors are parallel */
VertexList *findParallel(UniSys *us, Unstructured *unst, UCD_connectivity *ucdc,
                         int compV /*fvec3 *v*/, fvec3 *w, fmat3 *grad,
                         int variant, float extent)
{
    unsigned char *flags;
    VertexList *vertexList;
    //int *nodes;
    int cell;
    //int edge;
    int *firstVertex;
    signed char *nVerticesOnFace;
    int i;
    int msgCount;

    msgCount = 0;
    vertexList = NEW0(VertexList, 1);

    /* Initialize an index into the vertex list, given a face number */
    firstVertex = NEW(int, ucdc->nFaces);
    nVerticesOnFace = NEW(signed char, ucdc->nFaces);
    /* Not set */
    for (i = 0; i < ucdc->nFaces; i++)
        nVerticesOnFace[i] = -1;

    flags = NEW0(unsigned char, ucdc->nEdges);

    markIntersectedEdges(unst, ucdc, compV, w, flags);

    /* Loop over cells: find segments of core lines */
    for (cell = 0; cell < unst->nCells; cell++)
    {
        int cellTypeIdx;
        int globalNode[8];
        //int i, j;
        int nVerticesInCell;
        int verticesInCell[72];

        /* Collect global node numbers */
        cellTypeIdx = getCellTypeIdx(unst, cell);
        if (!getCellNodes(unst, cell, globalNode))
        {
            fprintf(stderr, "skipping unsupported cell type at cell %d\n", cell);
            continue;
        }

        nVerticesInCell = intersectCell(
            globalNode, cellTypeIdx, unst, ucdc, compV, w, grad,
            nVerticesOnFace, firstVertex, flags, true,
            vertexList, verticesInCell, variant, extent);

        /* If number of vertices is odd, retry without reject test */
        if (nVerticesInCell & 1)
        {
            nVerticesInCell = intersectCell(
                globalNode, cellTypeIdx, unst, ucdc, compV, w, grad,
                nVerticesOnFace, firstVertex, flags, false,
                vertexList, verticesInCell, variant, extent);
        }

        /* If still an odd number, print a warning */
        if (nVerticesInCell & 1)
        {
            if (!degenerateCell(globalNode, cellTypeIdx, unst, compV, w) && msgCount <= 20)
            {
#if PRINT_WARNINGS
                printf("cell %d has %d vertices %s\n",
                       cell, nVerticesInCell,
                       (msgCount == 20) ? "(suppressing further warnings)" : "");
                dumpCell(globalNode, cellTypeIdx, unst, compV, w);
#endif
                msgCount++;
            }
        }

        /* Sort the list 'verticesInCell' by ascending lambda */
        {
            int i, j, buf;

            for (i = 0; i < nVerticesInCell - 1; i++)
            {
                for (j = i + 1; j < nVerticesInCell; j++)
                {
                    if (vertexList->elem[j].lambda < vertexList->elem[i].lambda)
                    {
                        buf = verticesInCell[i];
                        verticesInCell[i] = verticesInCell[j];
                        verticesInCell[j] = buf;
                    }
                }
            }
        }

        /* Now take pairs in sequence */
        while (nVerticesInCell >= 2)
        {
            int vert0, vert1;
            fvec3 velo, segment;
            Vertex *v0, *v1;

            vert0 = verticesInCell[nVerticesInCell - 1];
            vert1 = verticesInCell[nVerticesInCell - 2];
            nVerticesInCell -= 2;

            v0 = &(vertexList->elem[vert0]);
            v1 = &(vertexList->elem[vert1]);

            /* Orient the line segment (vert0, vert1)
            consistently with its mean velocity */

            fvec3add(v0->velo, v1->velo, velo);
            fvec3sub(v1->xyz, v0->xyz, segment);

#ifdef CONNECTING_THE_OLD_WAY
            // Orient the segment according to the velocity
            if (fvec3dot(velo, segment) < 0)
            {
                Vertex *vTmp;
                int vertTmp;

                vTmp = v0;
                v0 = v1;
                v1 = vTmp;
                vertTmp = vert0;
                vert0 = vert1;
                vert1 = vertTmp;
            }

            if (v1->linkF == VC_VOID && v0->linkB == VC_VOID)
            {
                v1->linkF = vert0;
                v0->linkB = vert1;
            }
#else
            // Orient the segment such that it fits
            if ((v0->linkF != VC_VOID || v1->linkB != VC_VOID) // v0-->v1 does not work
                && (v0->linkB != VC_VOID || v1->linkF != VC_VOID))
            { // v1-->v0 neither

                // Flip the sequence attached to v0
                Vertex *start = v0;
                bool fwd = start->linkF != VC_VOID;
                Vertex *curr = start;

                while (true)
                {
                    // Swap forward and backward links
                    int save = curr->linkF;
                    curr->linkF = curr->linkB;
                    curr->linkB = save;

                    // Step along non-void link (note: they have been swapped!)
                    int nextVert = fwd ? curr->linkB : curr->linkF;
                    if (nextVert == VC_VOID)
                        break;
                    curr = &(vertexList->elem[nextVert]);
                    if (curr == start)
                        break;
                }
            }
            if (v0->linkF == VC_VOID && v1->linkB == VC_VOID)
            {
                v0->linkF = vert1;
                v1->linkB = vert0;
            }
            //else if (v0->linkB == VC_VOID && v1->linkF == VC_VOID) {
            else
            {
                v0->linkB = vert1;
                v1->linkF = vert0;
            }
#endif
        }

        /* Update AVS status */
        if (cell % 10000 == 0)
        {
            char str[100];
            sprintf(str, "cell %4dk of %4dk", cell / 1000, unst->nCells / 1000);
            us->moduleStatus(str, (100 * cell) / unst->nCells);
        }
    }

    printf("%d cells had odd number of vertices                 \n", msgCount);

    free(flags);
    free(firstVertex);
    free(nVerticesOnFace);

    return vertexList;
}

/***************************************/
/* Connect vertices to VC_Polylines    */
/***************************************/

PolylineList *generatePolylines(VertexList *vertexList)
{
    int i, k;
    //int nPolylines;
    //int *spoints;
    //int *nsegs;
    PolylineList *polylineList;

    polylineList = NEW0(PolylineList, 1);

    for (i = 0; i < vertexList->nElems; i++)
    {
        int i0, i1, i2;
        //int n;
        VC_Polyline *pl;

        /* Ignore if vertex has been used before */
        if (vertexList->elem[i].used)
            continue;

        /* Find the start vertex */
        i0 = i;
        while (true)
        {
            i1 = vertexList->elem[i0].linkB;
            if (i1 == i || i1 == VC_VOID)
                break;
            i0 = i1;
        }

        i1 = vertexList->elem[i0].linkF;
        if (i1 == VC_VOID)
            continue;

        /* Generate a new VC_Polyline */
        polylineList->nElems++;
        polylineList->elem = (VC_Polyline *)realloc(polylineList->elem, polylineList->nElems * sizeof(VC_Polyline));
        pl = &(polylineList->elem[polylineList->nElems - 1]);
        pl->start = i0;

        /* Traverse the VC_Polyline, mark all used vertices */
        vertexList->elem[i0].used = true;

        pl->nSegments = 1;

        do
        {
            i2 = vertexList->elem[i1].linkF;
            vertexList->elem[i1].used = true;
            if (i2 == VC_VOID)
                break;
            pl->nSegments++;
            i0 = i1;
            i1 = i2;
        } while (!(vertexList->elem[i2].used));
    }

    /* Test for VC_Polylines starting/ending in the grid interior */
    /* For debugging purposes only, can be omitted later */

    for (k = 0; k < polylineList->nElems; k++)
    {
        int i0, i1, n;
        int startCell1, startCell2;
        int endCell1, endCell2;
        Vertex *v = NULL;
        VC_Polyline *pl;

        pl = &(polylineList->elem[k]);
        i1 = pl->start;

        startCell1 = vertexList->elem[i1].cell1;
        startCell2 = vertexList->elem[i1].cell2;

        for (n = 0; n <= pl->nSegments; n++)
        {
            i0 = i1;
            v = &(vertexList->elem[i0]);
            i1 = v->linkF;
        }

        endCell1 = v->cell1;
        endCell2 = v->cell2;

        if ((startCell1 != endCell1) || (startCell2 != endCell2))
        {
            if (((startCell1 != VC_VOID) && (startCell2 != VC_VOID)) || ((endCell1 != VC_VOID) && (endCell2 != VC_VOID)))
            {
                printf("VC_Polyline of length %4d from cells (%6d,%6d)"
                       "to cells (%6d, %6d)\n",
                       n, startCell1, startCell2, endCell1, endCell2);
            }
        }
    }

    return polylineList;
}

/***********************************************/
/* Compute the feature quality for each vertex */
/***********************************************/

void computeFeatureQuality(VertexList *vertexList)
{
    for (int i = 0; i < vertexList->nElems; i++)
    {
        int i1, i2;
        vec3 position;
        vec3 position1;
        vec3 position2;
        vec3 velocity;
        vec3 tangent;
        vec3 v_normed;
        vec3 t_normed;
        double cos_orig;
        double cos_sign;
        double cos_value;
        double angle;

        vertexList->elem[i].sign = 0;
        vertexList->elem[i].quality = 0.0;
        vertexList->elem[i].angle = 90.0;

        i1 = vertexList->elem[i].linkF;
        i2 = vertexList->elem[i].linkB;

        if (i1 == VC_VOID)
            i1 = i;
        if (i2 == VC_VOID)
            i2 = i;
        if (i1 == i2)
            continue;

        fvec3tovec3(vertexList->elem[i].velo, velocity);
        fvec3tovec3(vertexList->elem[i].xyz, position);
        fvec3tovec3(vertexList->elem[i1].xyz, position1);
        fvec3tovec3(vertexList->elem[i2].xyz, position2);

        vec3sub(position1, position2, tangent);
        vec3nrm(velocity, v_normed);
        vec3nrm(tangent, t_normed);

        cos_orig = vec3dot(v_normed, t_normed);
        cos_sign = SGN(cos_orig);
        cos_value = ABS(cos_orig);
        angle = acos(cos_value);
        angle = angle / M_PI * 180.0;

        vertexList->elem[i].sign = (int)cos_sign;
        vertexList->elem[i].quality = (float)cos_value;
        vertexList->elem[i].angle = (float)angle;
        /*
               printf("\n");
               printf("i1 i i2 = [%4d %4d %4d]   ", i1,i,i2);
               printf("orig  = % 8.6f   ", cos_orig);
               printf("sign  = % d   ", (int) cos_sign);
               printf("value = %8.6f   ",  cos_value);
               printf("angle = %9.6f   ",  angle);
               printf("\n");
               vec3printf("position1", position1, "\n");
               vec3printf("position ", position,  "\n");
               vec3printf("position2", position2, "\n");
        vec3printf("velocity ", velocity,  "\n");
        vec3printf("tangent  ", tangent,   "\n");
        printf("\n");
        */
    }

    return;
}

/********************************/
/* Generate the output geometry */
/********************************/

void generateOutputGeometry(UniSys *us, VertexList *vertexList, PolylineList *polylineList,
                            int min_vertices, int max_exceptions, float min_strength, float max_angle, UniGeom *ugeom)
{
    ugeom->createObj(UniGeom::GT_LINE);

    fvec3 *verts = NEW(fvec3, vertexList->nElems);
    signed char *signs = NEW(signed char, vertexList->nElems);

    /* Draw the VC_Polylines */
    int lineCnt = 0;
    for (int k = 0; k < polylineList->nElems; k++)
    {
        int i0, i1, n;
        int length, nExceptions;
        Vertex *v;
        VC_Polyline *pl;

        pl = &(polylineList->elem[k]);
        i1 = pl->start;

        length = 0;
        nExceptions = BIGINT;
        /* Do not tolerate exceptions at the start */

        for (n = 0; n <= pl->nSegments; n++)
        {
            i0 = i1;
            v = &(vertexList->elem[i0]);

            if ((v->strength < min_strength) || (v->angle > max_angle))
                nExceptions++;
            else
                nExceptions = 0;

            fvec3copy(v->xyz, verts[length]);
            length++;

            if (nExceptions > max_exceptions || n == pl->nSegments)
            {

                if (length - nExceptions >= min_vertices)
                {

                    /* Cut off exceptions at the end */
                    length -= nExceptions;

                    ugeom->addPolyline((float *)verts, NULL, length);
                    lineCnt++;
                }
                length = 0;
            }

            i1 = v->linkF;

        } /* Next n (next segment) */
    } /* next VC_Polyline */

    free(verts);
    free(signs);

    ugeom->assignObj("vortex core");

    us->info("%d core lines", lineCnt);

    return;
}
