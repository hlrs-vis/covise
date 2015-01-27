/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UCD_RIDGE_LIB_CPP
#define UCD_RIDGE_LIB_CPP

#include <vector>
#include <map>

typedef int Boolean;
#define FALSE 0
#define TRUE 1

#define MAXCOLORS 256
#define MAXLABELSIZE 10000

// ### sadlo hacks (respective to isosurface module which this is based on):
#define NORMALS_FROM_GRAD 0
#define COMPUTE_RIDGE 1

#define EXTERNAL_DATA 0

#define FILTER_BY_EXCEPTION_CNT 1

#include "unstructured.h"

#include "linalg.h"
//typedef float vec3[3];

#include "ridge_surface_impl.h"

//#define MO_SEP    ":"
//#define MO_CELL_PCA  "cell nodes PCA"
//#define MO_EDGE_PCA  "edge nodes PCA"
//#define MO_ALL  MO_CELL_PCA MO_SEP MO_EDGE_PCA
#define MONR_CELL_PCA 1
#define MONR_EDGE_PCA 2

//#define EX_SEP    ":"
//#define EX_MAX  "maximum ridges"
//#define EX_MIN  "minimum ridges"
//#define EX_ALL  EX_MAX EX_SEP EX_MIN
#define EXNR_MAX 1
#define EXNR_MIN 2

struct IEdge
{
    int partner;
    int vertexnr;
};

struct Nodeinfo
{
    Boolean marked;
    int nedges;
    int *edge;
    int niedges;
    IEdge *iedge;
    fvec3 grad;
    float scalar2;
};

//static int nvertices[8] = {
//  1, 2, 3, 4, 4, 5, 6, 8
//};
static int edge[12][2] = { // hex
    { 0, 1 },
    { 2, 3 },
    { 4, 5 },
    { 6, 7 },
    { 0, 2 },
    { 1, 3 },
    { 4, 6 },
    { 5, 7 },
    { 0, 4 },
    { 1, 5 },
    { 2, 6 },
    { 3, 7 }
};
static int tedge[6][2] = { // tet
    { 0, 1 },
    { 0, 2 },
    { 0, 3 },
    { 1, 2 },
    { 2, 3 },
    { 3, 1 }
};

#define signBin(a) ((a) < 0.0 ? (-1.0) : (1.0))

inline int nodeOrderAVStoMC(int cellTypeUnst, int navs)
{
    if (cellTypeUnst == Unstructured::CELL_HEX)
    {
        switch (navs)
        {
        case 0:
            return 4;
            break;
        case 1:
            return 5;
            break;
        case 2:
            return 7;
            break;
        case 3:
            return 6;
            break;
        case 4:
            return 0;
            break;
        case 5:
            return 1;
            break;
        case 6:
            return 3;
            break;
        case 7:
            return 2;
            break;
        }
    }
    else if (cellTypeUnst == Unstructured::CELL_TET)
    {
        switch (navs)
        {
        case 0:
            return 0;
            break;
        case 1:
            return 1;
            break;
        case 2:
            return 2;
            break;
        case 3:
            return 3;
            break;
        }
    }
    else
        printf("nodeOrderAVStoMC: error: unsupported cell type\n");
    return -1;
}

inline int nodeOrderMCtoAVS(int cellTypeUnst, int nmc)
{
    if (cellTypeUnst == Unstructured::CELL_HEX)
    {
        switch (nmc)
        {
        case 4:
            return 0;
            break;
        case 5:
            return 1;
            break;
        case 7:
            return 2;
            break;
        case 6:
            return 3;
            break;
        case 0:
            return 4;
            break;
        case 1:
            return 5;
            break;
        case 3:
            return 6;
            break;
        case 2:
            return 7;
            break;
        }
    }
    else if (cellTypeUnst == Unstructured::CELL_TET)
    {
        switch (nmc)
        {
        case 0:
            return 0;
            break;
        case 1:
            return 1;
            break;
        case 2:
            return 2;
            break;
        case 3:
            return 3;
            break;
        }
    }
    else
        printf("nodeOrderMCtoAVS: error: unsupported cell type\n");
    return -1;
}

#if 0 // replaced 2007-10-01
void computeConsistentNodeValuesByPCA(Unstructured *unst,
                                      int compEV, int compGrad,
                                      int *nodes, int nodeCnt,
                                      float *nodeValues,
                                      double *eigenValuesDesc)
{ // compEV: Unstructured component for eigenvectors
  // length of eigenvectors is normalized for uniform weights
  // eigenValuesDesc: if not NULL, outputs eigenvalues in descending order
  // returns values in AVS node order
  /* scheme for achieving constistent orientation of the eigenvectors:
     1. do a PCA of all eigenvectors that belong to the cell
     2. choose an orientation for the principal eigenvector
     3. orient all eigenvectors according to the oriented principal eigenvector

     implementation: according to:
     http://en.wikipedia.org/wiki/Principal_components_analysis
  */

  for (int i=0; i<nodeCnt; i++) {
    nodeValues[i] = 0.0;
  }

  // assemble data matrix X (each observation is a column vector)
  // each eigenvector produces 2 data points, one in each direction from origin
  float *X[3];
  X[0]=new float[2*nodeCnt];
  X[1]=new float[2*nodeCnt];
  X[2]=new float[2*nodeCnt];
  {
    // go over nodes 
    for (int n=0; n<nodeCnt; n++) {

      int globalNodeId = nodes[n];
      vec3 ev;
      unst->getVector3(globalNodeId, compEV, ev);
      vec3nrm(ev, ev);

      X[0][n*2 + 0] = ev[0];
      X[1][n*2 + 0] = ev[1];
      X[2][n*2 + 0] = ev[2];

      X[0][n*2 + 1] = -ev[0];
      X[1][n*2 + 1] = -ev[1];
      X[2][n*2 + 1] = -ev[2];
    }
  }

  // compute covariance matrix C
  // mean is already at origin because adding each eigenvector as two points
  // in both directions of the eigenvector, centered around origin and of
  // same length
  mat3 C;
  {
    for (int j=0; j<3; j++) {
      for (int i=0; i<3; i++) {
        
        double sum = 0.0;
        for (int k=0; k<2*nodeCnt; k++) {
          sum += X[i][k] * X[j][k];
        }

        C[i][j] = sum / (2*nodeCnt - 1);
      }
    }
  }

  // compute eigenvalues and eigenvectors
  vec3 eigenvalues;
  vec3 eigenvectors[3];
  {
    // force C to be symmetric (added 2007-08-15, untested)
    mat3symm(C, C);

    // eigenvalues
    bool allReal = (mat3eigenvalues(C, eigenvalues) == 3);

    if (!allReal) {
      //printf("got complex eigenvalues: %g, %g, %g, returning zero\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
      //mat3dump(C, stdout);

      return;
    }

    // eigenvectors
    mat3realEigenvector(C, eigenvalues[0], eigenvectors[0]);
    mat3realEigenvector(C, eigenvalues[1], eigenvectors[1]);
    mat3realEigenvector(C, eigenvalues[2], eigenvectors[2]);
  }

#if 0
  // get largest eigenvalue
  int maxEVIdx;
  {
    if (eigenvalues[0] > eigenvalues[1]) {
      if (eigenvalues[0] > eigenvalues[2]) {
        maxEVIdx = 0;
      }
      else { // ev2 >= ev0 and ev0 > ev1
        maxEVIdx = 2;
      }
    }
    else { // ev1 >= ev0
      if (eigenvalues[2] > eigenvalues[1]) {
        maxEVIdx = 2;
      }
      else { // ev1 >= ev2 and ev1 >= ev0
        maxEVIdx = 1;
      }
    }
  }
#else
  // sort eigenvalues in descending order
  int evalDescIndices[3];
  {
    if (eigenvalues[0] > eigenvalues[1]) {
      if (eigenvalues[0] > eigenvalues[2]) {
        evalDescIndices[0] = 0;
      }
      else { // ev2 >= ev0 and ev0 > ev1
        evalDescIndices[0] = 2;
      }
    }
    else { // ev1 >= ev0
      if (eigenvalues[2] > eigenvalues[1]) {
        evalDescIndices[0] = 2;
      }
      else { // ev1 >= ev2 and ev1 >= ev0
        evalDescIndices[0] = 1;
      }
    }

    int remainingIndices[2];
    switch(evalDescIndices[0]) {
    case 0: remainingIndices[0] = 1; remainingIndices[1] = 2; break;
    case 1: remainingIndices[0] = 0; remainingIndices[1] = 2; break;
    case 2: remainingIndices[0] = 0; remainingIndices[1] = 1; break; 
    }

    if (eigenvalues[remainingIndices[0]] > eigenvalues[remainingIndices[1]]) {
      evalDescIndices[1] = remainingIndices[0];
      evalDescIndices[2] = remainingIndices[1];
    }
    else {
      evalDescIndices[1] = remainingIndices[1];
      evalDescIndices[2] = remainingIndices[0];
    }

    if (eigenValuesDesc) {
      eigenValuesDesc[0] = eigenvalues[evalDescIndices[0]];
      eigenValuesDesc[1] = eigenvalues[evalDescIndices[1]];
      eigenValuesDesc[2] = eigenvalues[evalDescIndices[2]];
    }
  }
#endif

  // get eigenvector belonging to largest eigenvalue
  // keep the sign (this is now the signed direction for the node set)
  vec3 evMax;
  {
    vec3copy(eigenvectors[evalDescIndices[0]], evMax);
  }

  // orient eigenvectors at nodes
  vec3 *nodeEigenVects= new vec3[nodeCnt];
  {
    for (int n=0; n<nodeCnt; n++) {
      vec3 ev = { X[0][n*2+0], X[1][n*2+0], X[2][n*2+0] };
      
      if (vec3dot(ev, evMax) < 0) {
        // invert sign
        vec3scal(ev, -1.0, nodeEigenVects[n]);
      }
      else {
        // no inversion
        vec3copy(ev, nodeEigenVects[n]);
      }
    }
  }

  // compute ridge criterion at nodes
  {
    for (int n=0; n<nodeCnt; n++) {
      
      int globalNodeId = nodes[n];
      vec3 grad;
      unst->getVector3(globalNodeId, compGrad, grad);
      
      nodeValues[n] = vec3dot(nodeEigenVects[n], grad);
    }
  }
  delete[] nodeEigenVects;
  delete[] X[0];
  delete[] X[1];
  delete[] X[2];
}
#else
void computeConsistentNodeValuesByPCA(Unstructured *unstEV, int compEV,
                                      Unstructured *unstGrad, int compGrad,
                                      int *nodes, int nodeCnt,
                                      float *nodeValues,
                                      double *eigenValuesDesc)
{ // compEV: Unstructured component for eigenvectors
    // length of eigenvectors is normalized for uniform weights
    // eigenValuesDesc: if not NULL, outputs eigenvalues in descending order
    // returns values in AVS node order
    /* scheme for achieving constistent orientation of the eigenvectors:
     1. do a PCA of all eigenvectors that belong to the cell
     2. choose an orientation for the principal eigenvector
     3. orient all eigenvectors according to the oriented principal eigenvector

     implementation: according to:
     http://en.wikipedia.org/wiki/Principal_components_analysis
  */

    for (int i = 0; i < nodeCnt; i++)
    {
        nodeValues[i] = 0.0;
    }

    // assemble data matrix X (each observation is a column vector)
    // each eigenvector produces 2 data points, one in each direction from origin
    float *X[3];
    X[0] = new float[2 * nodeCnt];
    X[1] = new float[2 * nodeCnt];
    X[2] = new float[2 * nodeCnt];
    {
        // go over nodes
        for (int n = 0; n < nodeCnt; n++)
        {

            int globalNodeId = nodes[n];
            vec3 ev;
            unstEV->getVector3(globalNodeId, compEV, ev);
            vec3nrm(ev, ev);

            X[0][n * 2 + 0] = ev[0];
            X[1][n * 2 + 0] = ev[1];
            X[2][n * 2 + 0] = ev[2];

            X[0][n * 2 + 1] = -ev[0];
            X[1][n * 2 + 1] = -ev[1];
            X[2][n * 2 + 1] = -ev[2];
        }
    }

    // compute covariance matrix C
    // mean is already at origin because adding each eigenvector as two points
    // in both directions of the eigenvector, centered around origin and of
    // same length
    mat3 C;
    {
        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < 3; i++)
            {

                double sum = 0.0;
                for (int k = 0; k < 2 * nodeCnt; k++)
                {
                    sum += X[i][k] * X[j][k];
                }

                C[i][j] = sum / (2 * nodeCnt - 1);
            }
        }
    }

    // compute eigenvalues and eigenvectors
    vec3 eigenvalues;
    vec3 eigenvectors[3];
    {
        // force C to be symmetric (added 2007-08-15, untested)
        mat3symm(C, C);

        // eigenvalues
        bool allReal = (mat3eigenvalues(C, eigenvalues) == 3);

        if (!allReal)
        {
            //printf("got complex eigenvalues: %g, %g, %g, returning zero\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
            //mat3dump(C, stdout);

            return;
        }

        // eigenvectors
        mat3realEigenvector(C, eigenvalues[0], eigenvectors[0]);
        mat3realEigenvector(C, eigenvalues[1], eigenvectors[1]);
        mat3realEigenvector(C, eigenvalues[2], eigenvectors[2]);
    }

#if 0
  // get largest eigenvalue
  int maxEVIdx;
  {
    if (eigenvalues[0] > eigenvalues[1]) {
      if (eigenvalues[0] > eigenvalues[2]) {
        maxEVIdx = 0;
      }
      else { // ev2 >= ev0 and ev0 > ev1
        maxEVIdx = 2;
      }
    }
    else { // ev1 >= ev0
      if (eigenvalues[2] > eigenvalues[1]) {
        maxEVIdx = 2;
      }
      else { // ev1 >= ev2 and ev1 >= ev0
        maxEVIdx = 1;
      }
    }
  }
#else
    // sort eigenvalues in descending order
    int evalDescIndices[3];
    {
        if (eigenvalues[0] > eigenvalues[1])
        {
            if (eigenvalues[0] > eigenvalues[2])
            {
                evalDescIndices[0] = 0;
            }
            else
            { // ev2 >= ev0 and ev0 > ev1
                evalDescIndices[0] = 2;
            }
        }
        else
        { // ev1 >= ev0
            if (eigenvalues[2] > eigenvalues[1])
            {
                evalDescIndices[0] = 2;
            }
            else
            { // ev1 >= ev2 and ev1 >= ev0
                evalDescIndices[0] = 1;
            }
        }

        int remainingIndices[2];
        switch (evalDescIndices[0])
        {
        case 0:
            remainingIndices[0] = 1;
            remainingIndices[1] = 2;
            break;
        case 1:
            remainingIndices[0] = 0;
            remainingIndices[1] = 2;
            break;
        case 2:
            remainingIndices[0] = 0;
            remainingIndices[1] = 1;
            break;
        }

        if (eigenvalues[remainingIndices[0]] > eigenvalues[remainingIndices[1]])
        {
            evalDescIndices[1] = remainingIndices[0];
            evalDescIndices[2] = remainingIndices[1];
        }
        else
        {
            evalDescIndices[1] = remainingIndices[1];
            evalDescIndices[2] = remainingIndices[0];
        }

        if (eigenValuesDesc)
        {
            eigenValuesDesc[0] = eigenvalues[evalDescIndices[0]];
            eigenValuesDesc[1] = eigenvalues[evalDescIndices[1]];
            eigenValuesDesc[2] = eigenvalues[evalDescIndices[2]];
        }
    }
#endif

    // get eigenvector belonging to largest eigenvalue
    // keep the sign (this is now the signed direction for the node set)
    vec3 evMax;
    {
        vec3copy(eigenvectors[evalDescIndices[0]], evMax);
    }

    // orient eigenvectors at nodes
    vec3 *nodeEigenVects = new vec3[nodeCnt];
    {
        for (int n = 0; n < nodeCnt; n++)
        {
            vec3 ev = { X[0][n * 2 + 0], X[1][n * 2 + 0], X[2][n * 2 + 0] };

            if (vec3dot(ev, evMax) < 0)
            {
                // invert sign
                vec3scal(ev, -1.0, nodeEigenVects[n]);
            }
            else
            {
                // no inversion
                vec3copy(ev, nodeEigenVects[n]);
            }
        }
    }

    // compute ridge criterion at nodes
    {
        for (int n = 0; n < nodeCnt; n++)
        {

            int globalNodeId = nodes[n];
            vec3 grad;
            unstGrad->getVector3(globalNodeId, compGrad, grad);

            nodeValues[n] = vec3dot(nodeEigenVects[n], grad);
        }
    }
    delete[] nodeEigenVects;
    delete[] X[0];
    delete[] X[1];
    delete[] X[2];
}
#endif

void computeConsistentNodeValuesByEdgePCA(Unstructured *unst,
                                          int compEV, int compGrad,
                                          int cell, float *nodeValues,
                                          int *falsePositiveIntersections,
                                          int *falseNegativeIntersections)
{
    int nodeCnt, edgeCnt;
    if (unst->getCellType(cell) == Unstructured::CELL_HEX)
    {
        nodeCnt = 8;
        edgeCnt = 12;
    }
    else if (unst->getCellType(cell) == Unstructured::CELL_TET)
    {
        nodeCnt = 4;
        edgeCnt = 6;
    }
    else
    {
        printf("unsupported cell type at cell %d\n", cell);
        return;
    }

    // compute (unsigned) node values
    for (int n = 0; n < nodeCnt; n++)
    {

        int *cellNodes = unst->getCellNodesAVS(cell);
        int globalNodeId = cellNodes[n];

        vec3 grad;
        unst->getVector3(globalNodeId, compGrad, grad);
        vec3 ev;
        unst->getVector3(globalNodeId, compEV, ev);

        nodeValues[n] = vec3dot(ev, grad);
    }
    bool *nodeValueComputed = new bool[nodeCnt];
    for (int i = 0; i < nodeCnt; i++)
        nodeValueComputed[i] = false;
    // first node determines signs
    nodeValueComputed[0] = true;

    // correct signs: go over edges of cell, in "connected order"
    for (int e = 0; e < edgeCnt; e++)
    {

        // get global nodes of edge
        int n1, n2;
        unst->getCellEdgeNodesConnOrderAVS(cell, e, n1, n2);

        // do PCA on edge
        int nodes[2] = { n1, n2 };
        float nodeValues[2];
        computeConsistentNodeValuesByPCA(unst, compEV, unst, compGrad, nodes, 2, nodeValues, NULL);

        // #######################################################################
        // ### there seems to be a bug below, nodeValues is defined twice and
        //     the output-nodeValues is not manipulated !!!!!!!!!!!!

        if (nodeValues[0] * nodeValues[1] < 0.0)
        {
            // intersected edge

            // get local node IDs
            int l1, l2;
            unst->getCellEdgeLocNodesConnOrderAVS(cell, e, l1, l2);

            if ((nodeValueComputed[l1]) && (nodeValueComputed[l2]))
            {
                if (nodeValues[l1] * nodeValues[l2] >= 0.0)
                {
                    //printf("warning: intersected edge but already determined same signs\n");
                    if (falseNegativeIntersections)
                        (*falseNegativeIntersections)++;
                }
            }
            else if (nodeValueComputed[l1])
            {
                // l2 gets sign opposite to l1
                nodeValues[l2] = -signBin(nodeValues[l1]) * fabs(nodeValues[l2]);
                nodeValueComputed[l2] = true;
            }
            else if (nodeValueComputed[l2])
            {
                // l1 gets sign opposite to l2
                nodeValues[l1] = -signBin(nodeValues[l2]) * fabs(nodeValues[l1]);
                nodeValueComputed[l1] = true;
            }
            else
            {
                printf("neither l1 nor l2 computed, this is a bug\n");
            }
        }
        else
        {
            // no intersection

            // get local node IDs
            int l1, l2;
            unst->getCellEdgeLocNodesConnOrderAVS(cell, e, l1, l2);

            if ((nodeValueComputed[l1]) && (nodeValueComputed[l2]))
            {
                if (nodeValues[l1] * nodeValues[l2] < 0.0)
                {
                    //printf("warning: edge not intersected but already determined different signs\n");
                    if (falsePositiveIntersections)
                        (*falsePositiveIntersections)++;
                }
            }
            else if (nodeValueComputed[l1])
            {
                // l2 gets same sign as l1
                nodeValues[l2] = signBin(nodeValues[l1]) * fabs(nodeValues[l2]);
                nodeValueComputed[l2] = true;
            }
            else if (nodeValueComputed[l2])
            {
                // l1 gets same sign as l2
                nodeValues[l1] = signBin(nodeValues[l2]) * fabs(nodeValues[l1]);
                nodeValueComputed[l1] = true;
            }
            else
            {
                printf("neither l1 nor l2 computed, this is a bug\n");
            }
        }
        delete[] nodeValueComputed;
    }
    // printf("%d false positive and %d false negative intersections\n",
    //     falsePositiveIntersections, falseNegativeIntersections);
}

inline double findTByBisection(Unstructured *unstVec, int compVec,
                               Unstructured *unstMat, int compMat,
                               bool modePV, bool ridge, int n0, int n1)
{ // if modePV false, ridge for ridges, !ridge for valleys

    mat3 T;
    {
        fmat3 Tf;
        unstMat->getMatrix3(n0, compMat, Tf);
        fmat3tomat3(Tf, T);
    }

    if (T[0][2] == 0 && T[1][2] == 0 && T[2][2] == 0 && T[2][0] == 0 && T[2][1] == 0)
    {

        // 2d matrix

        // true 3d matrix

        // get vectors at nodes
        vec2 v0, v1;
        {
            vec3 vv0, vv1;
            unstVec->getVector3(n0, compVec, vv0);
            unstVec->getVector3(n1, compVec, vv1);
            vec3tovec2(vv0, v0);
            vec3tovec2(vv1, v1);
        }

        // get matrices at nodes
        mat2 M0, M1;
        {
            fmat3 M0f, M1f;
            unstMat->getMatrix3(n0, compMat, M0f);
            unstMat->getMatrix3(n1, compMat, M1f);
            mat3 MM0, MM1;
            fmat3tomat3(M0f, MM0);
            fmat3tomat3(M1f, MM1);
            mat3tomat2(MM0, M0);
            mat3tomat2(MM1, M1);
        }

        double valL, valR;
        if (modePV)
        {
            // parallel vectors
            vec2 Mv0, Mv1;
            mat2vec(M0, v0, Mv0);
            mat2vec(M1, v1, Mv1);

            mat2 comb0, comb1;
            mat2setcols(comb0, Mv0, v0);
            mat2setcols(comb1, Mv1, v1);

            valL = mat2det(comb0);
            valR = mat2det(comb1);
        }
        else
        {
            // pre-selected eigenvalue (Eberly)
            vec2 eigenvals;
            vec2 evec;
            double ev;

            mat2eigenvalues(M0, eigenvals);
            if (ridge)
            {
                if (eigenvals[1] < eigenvals[0])
                    ev = eigenvals[1];
                else
                    ev = eigenvals[0];
            }
            else
            {
                if (eigenvals[1] < eigenvals[0])
                    ev = eigenvals[0];
                else
                    ev = eigenvals[1];
            }
            mat2realEigenvector(M0, ev, evec);
            valL = vec2dot(evec, v0);

            mat2eigenvalues(M1, eigenvals);
            if (ridge)
            {
                if (eigenvals[1] < eigenvals[0])
                    ev = eigenvals[1];
                else
                    ev = eigenvals[0];
            }
            else
            {
                if (eigenvals[1] < eigenvals[0])
                    ev = eigenvals[0];
                else
                    ev = eigenvals[1];
            }
            mat2realEigenvector(M1, ev, evec);
            valR = vec2dot(evec, v1);
        }
        double tL = 0;
        double tR = 1;
        double tMid;

        int iterationCnt = 10; // ###### TODO
        for (int iter = 0; iter < iterationCnt; iter++)
        {

            tMid = (tR - tL) / 2 + tL;

            // get determinant at mid
            double valMid;
            {

                // interpolate vector at mid
                vec2 vecMid;
                vec2lerp(v0, v1, tMid, vecMid);

                // interpolate matrix at mid
                mat2 matMid;
                mat2lerp(M0, M1, tMid, matMid);

                if (modePV)
                {
                    // compute determinant

                    vec2 Midv;
                    mat2vec(matMid, vecMid, Midv);

                    mat2 comb;
                    mat2setcols(comb, Midv, vecMid);

                    valMid = mat2det(comb);
                }
                else
                {
                    vec2 eigenvals;
                    vec2 evec;
                    double ev;

                    mat2eigenvalues(matMid, eigenvals);
                    if (ridge)
                    {
                        if (eigenvals[1] < eigenvals[0])
                            ev = eigenvals[1];
                        else
                            ev = eigenvals[0];
                    }
                    else
                    {
                        if (eigenvals[1] < eigenvals[0])
                            ev = eigenvals[0];
                        else
                            ev = eigenvals[1];
                    }
                    mat2realEigenvector(matMid, ev, evec);
                    valMid = vec2dot(evec, vecMid);
                }
            }

            // decide interval
            if (valL < valR)
            {
                if (valMid > 0)
                {
                    tR = tMid;
                    valR = valMid;
                }
                else
                {
                    tL = tMid;
                    valL = valMid;
                }
            }
            else if (valL > valR)
            {
                if (valMid < 0)
                {
                    tR = tMid;
                    valR = valMid;
                }
                else
                {
                    tL = tMid;
                    valL = valMid;
                }
            }
            else
            { // equal
                break;
            }
        }

        return tMid;
    }
    else
    {

        // true 3d matrix

        // get vectors at nodes
        vec3 v0, v1;
        unstVec->getVector3(n0, compVec, v0);
        unstVec->getVector3(n1, compVec, v1);

        // get matrices at nodes
        mat3 M0, M1;
        {
            fmat3 M0f, M1f;
            unstMat->getMatrix3(n0, compMat, M0f);
            unstMat->getMatrix3(n1, compMat, M1f);
            fmat3tomat3(M0f, M0);
            fmat3tomat3(M1f, M1);
        }

        double valL, valR;
        if (modePV)
        {
            // parallel vectors
            mat3 MM0, MM1;
            mat3mul(M0, M0, MM0);
            mat3mul(M1, M1, MM1);

            vec3 MMv0, MMv1;
            mat3vec(MM0, v0, MMv0);
            mat3vec(MM1, v1, MMv1);

            vec3 Mv0, Mv1;
            mat3vec(M0, v0, Mv0);
            mat3vec(M1, v1, Mv1);

            mat3 comb0, comb1;
            mat3setcols(comb0, MMv0, Mv0, v0);
            mat3setcols(comb1, MMv1, Mv1, v1);

            valL = mat3det(comb0);
            valR = mat3det(comb1);
        }
        else
        {
            // pre-selected eigenvalue (Eberly)
            vec3 eigenvals, eigenvalsSorted;
            vec3 evec;
            double ev;

            mat3eigenvalues(M0, eigenvals);
            vec3sortd(eigenvals, eigenvalsSorted, NULL);
            if (ridge)
            {
                ev = eigenvalsSorted[2];
            }
            else
            {
                ev = eigenvalsSorted[0];
            }
            mat3realEigenvector(M0, ev, evec);
            valL = vec3dot(evec, v0);

            mat3eigenvalues(M1, eigenvals);
            vec3sortd(eigenvals, eigenvalsSorted, NULL);
            if (ridge)
            {
                ev = eigenvalsSorted[2];
            }
            else
            {
                ev = eigenvalsSorted[0];
            }
            mat3realEigenvector(M1, ev, evec);
            valR = vec3dot(evec, v1);
        }

        double tL = 0;
        double tR = 1;
        double tMid;

        int iterationCnt = 10; // ###### TODO
        for (int iter = 0; iter < iterationCnt; iter++)
        {

            tMid = (tR - tL) / 2 + tL;

            // get determinant at mid
            double valMid;
            {

                // interpolate vector at mid
                vec3 vecMid;
                vec3lerp(v0, v1, tMid, vecMid);

                // interpolate matrix at mid
                mat3 matMid;
                mat3lerp(M0, M1, tMid, matMid);

                if (modePV)
                {
                    // compute determinant

                    mat3 MMid;
                    mat3mul(matMid, matMid, MMid);

                    vec3 MMidv;
                    mat3vec(MMid, vecMid, MMidv);

                    vec3 Midv;
                    mat3vec(matMid, vecMid, Midv);

                    mat3 comb;
                    mat3setcols(comb, MMidv, Midv, vecMid);

                    valMid = mat3det(comb);
                }
                else
                {
                    vec3 eigenvals, eigenvalsSorted;
                    vec3 evec;
                    double ev;

                    mat3eigenvalues(matMid, eigenvals);
                    vec3sortd(eigenvals, eigenvalsSorted, NULL);
                    if (ridge)
                    {
                        ev = eigenvalsSorted[2];
                    }
                    else
                    {
                        ev = eigenvalsSorted[0];
                    }
                    mat3realEigenvector(matMid, ev, evec);
                    valMid = vec3dot(evec, vecMid);
                }
            }

            // decide interval
            if (valL < valR)
            {
                if (valMid > 0)
                {
                    tR = tMid;
                    valR = valMid;
                }
                else
                {
                    tL = tMid;
                    valL = valMid;
                }
            }
            else if (valL > valR)
            {
                if (valMid < 0)
                {
                    tR = tMid;
                    valR = valMid;
                }
                else
                {
                    tL = tMid;
                    valL = valMid;
                }
            }
            else
            { // equal
                break;
            }
        }

        return tMid;
    }
}

int makeTrianglesConsistent_rek(std::vector<int> *triangleConn, int triaCnt,
                                int triangle,
                                std::map<pair<int, int>, vector<int> > *edges,
                                bool *triaVisited, int *flippedCnt,
                                int compLabel,
                                std::vector<int> *triangleComponents)
{
    int visitedCnt = 0;

    // go over edges of triangle
    for (int eIdx = 0; eIdx < 3; eIdx++)
    {

        int locIdx1 = eIdx;
        int locIdx2 = (eIdx + 1 < 3 ? eIdx + 1 : 0);

        // get vertex IDs
        int id1 = (*triangleConn)[triangle * 3 + locIdx1];
        int id2 = (*triangleConn)[triangle * 3 + locIdx2];

        // get lowId, highId
        int lowId = id1;
        int highId = id2;
        if (id1 > id2)
        {
            lowId = id2;
            highId = id1;
        }

        // get key
        std::pair<int, int> key;
        key.first = lowId;
        key.second = highId;

        if ((*edges)[key].size() != 2)
        {
            if ((*edges)[key].size() != 1)
            {
                printf("skipping edge shared by more than 2 triangles\n");
            }
            continue;
        }

        // go over triangles of edge
        for (int tIdx = 0; tIdx < (int)(*edges)[key].size(); tIdx++)
        {
            int nt = (*edges)[key][tIdx];
            if (!triaVisited[nt])
            {

                // get corresponding edge and flip triangle if necessary
                {
                    if (((id1 == (*triangleConn)[nt * 3 + 0]) && (id2 == (*triangleConn)[nt * 3 + 1])) || ((id1 == (*triangleConn)[nt * 3 + 1]) && (id2 == (*triangleConn)[nt * 3 + 2])) || ((id1 == (*triangleConn)[nt * 3 + 2]) && (id2 == (*triangleConn)[nt * 3 + 0])))
                    {
                        // inconsistency -> flip triangle
                        int w = (*triangleConn)[nt * 3 + 0];
                        (*triangleConn)[nt * 3 + 0] = (*triangleConn)[nt * 3 + 1];
                        (*triangleConn)[nt * 3 + 1] = w;
                        (*flippedCnt)++;
                    }
                }

                triaVisited[nt] = true;
                visitedCnt++;
                (*triangleComponents)[nt] = compLabel;

                // recurr
                visitedCnt += makeTrianglesConsistent_rek(triangleConn, triaCnt,
                                                          nt,
                                                          edges,
                                                          triaVisited, flippedCnt,
                                                          compLabel, triangleComponents);
            }
        }
    }
    return visitedCnt;
}

void makeTrianglesConsistent(std::vector<int> *triangleConn, int triaCnt,
                             std::vector<int> *triangleComponents,
                             std::vector<int> *triangleComponentSizes)
{ // triangleComponents, triangleComponentSizes: output

    triangleComponents->clear();
    triangleComponents->resize(triaCnt);
    triangleComponentSizes->clear();

    // get triangles that share each edge
    std::map<pair<int, int>, vector<int> > edges; // edge, triangles
    for (int t = 0; t < triaCnt; t++)
    {

        // go over edges of triangle
        for (int e = 0; e < 3; e++)
        {
            int locIdx1 = e;
            int locIdx2 = (e + 1 < 3 ? e + 1 : 0);

            // get vertex IDs
            int id1 = (*triangleConn)[t * 3 + locIdx1];
            int id2 = (*triangleConn)[t * 3 + locIdx2];

            // get lowId, highId
            int lowId = id1;
            int highId = id2;
            if (id1 > id2)
            {
                lowId = id2;
                highId = id1;
            }

            // collect info
            std::pair<int, int> key;
            key.first = lowId;
            key.second = highId;
            if (edges.find(key) == edges.end())
            {
                // new edge
                vector<int> trias;
                trias.push_back(t);
                edges[key] = trias;
            }
            else
            {
                // existing edge
                edges[key].push_back(t);
            }
        }
    }

    // flip inconsistent triangles (if possible due to non-orientable surfaces)
    {
        bool *triaVisited = new bool[triaCnt];
        for (int i = 0; i < triaCnt; i++)
        {
            triaVisited[i] = false;
        };

        // loop over components
        int connCompCnt = 0;
        int flippedCnt = 0;
        //for (int t=0; t<triaCnt; t++) {
        for (int t = triaCnt - 1; t >= 0; t--)
        {

            if (!triaVisited[t])
            {
                // triangle defines orientation for current connected component
                triaVisited[t] = true;
                int compSize = 1;
                (*triangleComponents)[t] = connCompCnt;

                compSize += makeTrianglesConsistent_rek(triangleConn, triaCnt, t, &edges, triaVisited, &flippedCnt, connCompCnt, triangleComponents);

                triangleComponentSizes->push_back(compSize);
                connCompCnt++;
            }
        }

        printf("%d edge-connected components\n", connCompCnt);

        // ####
        int visitedCnt = 0;
        for (int i = 0; i < triaCnt; i++)
        {
            if (triaVisited[i])
            {
                visitedCnt++;
            }
        }
        delete[] triaVisited;
        printf("%d visited and %d flipped triangles\n", visitedCnt, flippedCnt);
    }
}

void filteringConditions(Unstructured *unst,
                         bool *excludeNodes,
                         Unstructured *temp,
                         int compHess,
                         int compEigenvals,
                         int compScalar,
                         int compClipScalar,
                         vec3 pos,
                         double Hess_extr_eigenval_min,
                         double scalar_min,
                         float *scalar_min_cell,
                         double scalar_max,
                         double clip_scalar_min, double clip_scalar_max,
                         //char *extremum,
                         int extremumNr,
                         bool combine_exceptions,
                         bool &skip, bool &exception, int &exceptionCnt)
{ // excludeNodes: may be NULL
    // scalar_min_cell: if not NULL, scalar_min is ignored and this array defines
    //                  a scalar_min for each cell

    skip = false;
    exception = false;
    // ridge criterion
    {
        temp->selectVectorNodeData(compEigenvals);

        // unbeautiful because this findCell is also used for getting evals below
        // ###
        if (!temp->findCell(pos))
        {
            printf("filteringConditions: ERROR: findCell() failed for pos=(%g, %g, %g)\n", pos[0], pos[1], pos[2]);
        }

        // test fails if one of cell nodes is excluded
        if (excludeNodes)
        {
            int *nodes = temp->getCellNodesAVS();
            // ### fixed to hexahedra
            for (int n = 0; n < 8; n++)
            {
                if (excludeNodes[nodes[n]])
                {
                    skip = true;
                    //printf("rejected cell %d due to excluded node\n", temp->getCellIndex());
                    return;
                }
            }
        }

        vec3 evals;
#if 0
    // ###### interpolation is dangerous!
    temp->interpolateVector3(evals);
#else
        {
            int oldSel = temp->getVectorNodeDataComponent();
            mat3 H;
            temp->selectVectorNodeData(compHess);
            temp->interpolateMatrix3(H);

            // compute eigenvalues
            {
                mat3 dmat;
                //fmat3tomat3(H, dmat);
                mat3copy(H, dmat);

                // force dmat to be symmetric (added 2007-08-15, untested)
                mat3symm(dmat, dmat);

                vec3 eigenvalues;
                bool allReal = (mat3eigenvalues(dmat, eigenvalues) == 3);

                if (!allReal)
                {
                    //printf("filteringConditions: got complex eigenvalue at pos=(%g,%g,%g)!\n  H=(%g,%g,%g\n     %g,%g,%g\n     %g,%g,%g)\n",
                    //       pos[0], pos[1], pos[2],
                    //       H[0][0], H[0][1], H[0][2],
                    //       H[1][0], H[1][1], H[1][2],
                    //       H[2][0], H[2][1], H[2][2]);

                    //complexEVCnt++;
                    //vec3 vec;
                    //vec3zero(vec);
                    //out->setVector3(n, vec);
                    //continue;
                    // ####
                    eigenvalues[0] = 1;
                    eigenvalues[1] = 1;
                    eigenvalues[2] = 1; // ############ assuming maximum ridges -> 1 will be minimum and hence fail
                }

                if (false /*absoluteSorted*/)
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

                vec3copy(eigenvalues, evals);
            }

            temp->selectVectorNodeData(oldSel);
        }
#endif

        //if (strcmp(extremum, EX_MIN) == 0) {
        if (extremumNr == EXNR_MIN)
        {
            //if (evals[0] < 0.0) {
            if (evals[0] - Hess_extr_eigenval_min <= 0.0)
            {
// largest eigenvalue <= 0  -> skip
#if !FILTER_BY_EXCEPTION_CNT
                skip = true;
                //break;
                return;
#else
                exceptionCnt++;
                if (!combine_exceptions)
                {
                    //continue;
                    exception = true;
                    return;
                }
#endif
            }
        }
        else
        {
            //if (evals[2] > 0.0) {
            if (evals[2] + Hess_extr_eigenval_min >= 0.0)
            {
// smallest eigenvalue >= 0  -> skip
#if !FILTER_BY_EXCEPTION_CNT
                skip = true;
                //break;
                return;
#else
                exceptionCnt++;
                if (!combine_exceptions)
                {
                    //continue;
                    exception = true;
                    return;
                }
#endif
            }
        }
    }

    // filtering by primary scalar
    {
        unst->selectScalarNodeData(compScalar);

        unst->findCell(pos);
        unst->loadCellData(); // ###

        double scal_min = (scalar_min_cell ? scalar_min_cell[unst->getCellIndex()] : scalar_min);
        if ((unst->interpolateScalar() < scal_min) || (unst->interpolateScalar() > scalar_max))
        {
#if !FILTER_BY_EXCEPTION_CNT
            skip = true;
            //break;
            return;
#else
            exceptionCnt++;
            if (!combine_exceptions)
            {
                //continue;
                exception = true;
                return;
            }
#endif
        }
    }

    // filtering (clipping) by secondary scalar
    if (compClipScalar >= 0)
    {
        unst->selectScalarNodeData(compClipScalar);

        unst->findCell(pos); // ### overhead (for assuring data)
        unst->loadCellData(); // ###
        if ((unst->interpolateScalar() > clip_scalar_max) ||

            (unst->interpolateScalar() < clip_scalar_min))
        {
#if !FILTER_BY_EXCEPTION_CNT
            skip = true;
            //break;
            return;
#else
            exceptionCnt++;
            if (!combine_exceptions)
            {
                //continue;
                exception = true;
                return;
            }
#endif
        }
    }
}

void markNodesAtIntersectedEdges(UniSys *us,
                                 //UCD_structure* ucd,
                                 std::vector<int> *cellsToProcess,
                                 bool *excludeNodes,
                                 Unstructured *unst,
                                 //char *mode,
                                 int modeNr,
                                 //char *extremum,
                                 int extremumNr,
                                 int filter_by_cell,
                                 int compScalar,
                                 Unstructured *temp,
                                 int compGradient,
                                 int compHess,
                                 int compEigenvals,
                                 int compEigenvectExtr,
                                 bool useBisection,
                                 double level,
                                 double Hess_extr_eigenval_min,
                                 double PCA_subdom_maxperc,
                                 double scalar_min,
                                 float *scalar_min_cell,
                                 double scalar_max,
                                 int compClipScalar,
                                 double clip_scalar_min,
                                 double clip_scalar_max,
                                 int combine_exceptions,
                                 //int *node_list,
                                 Nodeinfo *nodeinfo,
                                 int &count_tet,
                                 int &count_hex,
                                 int &count_other)
{ // Mark nodes adjacent to intersected edges.
    // cellsToProcess: if not NULL, only these cells are processed
    // scalar_min_cell: if not NULL, scalar_min is ignored and this array defines
    //                  a scalar_min for each cell

    /* Store partner node for each such edge. */
    //AVSmodify_parameter("status", AVS_VALUE, "Marking nodes ...", 0, 0);
    //AVSmodule_status("Marking nodes ...", 0);
    //us->moduleStatus("marking nodes ...", 0);
    us->info("marking nodes ...");
    //int* node_list = ucd->node_list;
    int markedEdgeCnt = 0;
    //for (int i = 0; i < ucd->ncells; i++) {
    for (int cIdx = 0; cIdx < (cellsToProcess ? (int)cellsToProcess->size() : unst->nCells); cIdx++)
    {

        int i;
        if (cellsToProcess)
            i = (*cellsToProcess)[cIdx];
        else
            i = cIdx;

        int type;
        int nodenr[8];

        //type = ucd->cell_type[i];
        type = unst->getCellType(i);
        //if (type == UCD_HEXAHEDRON) {
        if (type == Unstructured::CELL_HEX)
        {
            count_hex++;
            int *node_list = unst->getCellNodesAVS(i);
            nodenr[4] = *node_list++;
            nodenr[5] = *node_list++;
            nodenr[7] = *node_list++;
            nodenr[6] = *node_list++;
            nodenr[0] = *node_list++;
            nodenr[1] = *node_list++;
            nodenr[3] = *node_list++;
            nodenr[2] = *node_list++;

#if COMPUTE_RIDGE
            // make eigenvectors at nodes consistent over cell
            // and then compute values at nodes
            float nodeValuesAVS[8], nodeValues[8];
            //if (strcmp(mode, MO_CELL_PCA) == 0) {
            if (modeNr == MONR_CELL_PCA)
            {
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByPCA(unst, compHessEigenvect, compGradient, unst->getCellNodesAVS(i), 8, nodeValuesAVS);
#else
#if 0
        computeConsistentNodeValuesByPCA(temp, compEigenvectExtr, compGradient, temp->getCellNodesAVS(i), 8, nodeValuesAVS, NULL);
#else
                // this version avoids generating triangles at cells where eigenvector
                // directions vary too much
                double eigenValuesDesc[3];
                computeConsistentNodeValuesByPCA(temp, compEigenvectExtr, temp, compGradient, temp->getCellNodesAVS(i), 8, nodeValuesAVS, eigenValuesDesc);

                if (PCA_subdom_maxperc < 1.0)
                {
                    double limitEV = eigenValuesDesc[0] * PCA_subdom_maxperc;
                    if (eigenValuesDesc[1] > limitEV)
                    {
                        // second eigenvalue is larger than limit -> skip cell
                        // ### TODO: this can produce holes, should somehow base the test
                        //           on triangle instead of cell
                        continue;
                    }
                }
#endif
#endif
            }
            else
            { // edge PCA
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByEdgePCA(unst, compHessEigenvect, compGradient, i, nodeValuesAVS);
#else
                computeConsistentNodeValuesByEdgePCA(temp, compEigenvectExtr, compGradient, i, nodeValuesAVS, NULL, NULL);
#endif
            }
            // adapt node order
            nodeValues[4] = nodeValuesAVS[0];
            nodeValues[5] = nodeValuesAVS[1];
            nodeValues[7] = nodeValuesAVS[2];
            nodeValues[6] = nodeValuesAVS[3];
            nodeValues[0] = nodeValuesAVS[4];
            nodeValues[1] = nodeValuesAVS[5];
            nodeValues[3] = nodeValuesAVS[6];
            nodeValues[2] = nodeValuesAVS[7];
#endif

            // skip cells that contain a node which does not
            // obey the inequalities for a ridge
            if (filter_by_cell)
            { // ################
                bool skip = false;
                for (int j = 0; j < 12; j++)
                {
                    int n0, n1; //float s0, s1;
                    //int master, slave;

                    n0 = nodenr[edge[j][0]];
                    n1 = nodenr[edge[j][1]];

                    vec3 evals0, evals1;
                    temp->getVector3(n0, compEigenvals, evals0);
                    temp->getVector3(n1, compEigenvals, evals1);

                    //if (strcmp(extremum, EX_MIN) == 0) {
                    if (extremumNr == EXNR_MIN)
                    {
                        if ((evals0[0] < 0.0) || (evals1[0] < 0.0))
                        {
                            // largest eigenvalue < 0  -> skip
                            skip = true;
                            break;
                        }
                    }
                    else
                    {
                        if ((evals0[2] > 0.0) || (evals1[2] > 0.0))
                        {
                            // smallest eigenvalue > 0  -> skip
                            skip = true;
                            break;
                        }
                    }

                    if ((unst->getScalar(n0, compScalar) < scalar_min) || (unst->getScalar(n1, compScalar) < scalar_min) || (unst->getScalar(n0, compScalar) > scalar_max) || (unst->getScalar(n1, compScalar) > scalar_max))
                    {
                        skip = true;
                        break;
                    }

// clip by scalar
#if 0
          if (compClipScalar >= 0) {
            vec3 pos0, pos1;
            // get node positions
            unst->getCoords(n0, pos0);
            unst->getCoords(n1, pos1);

            // find cell and clip
            if (unst2->findCell(pos0)) {
              double scal = unst2->interpolateScalar();
                                
              if ((scal < *clip_scalar_min) || (scal > clip_scalar_max)) {
                skip = true;
                break;
              }
            }
            if (unst2->findCell(pos1)) {
              double scal = unst2->interpolateScalar();
                                
              if ((scal < *clip_scalar_min) || (scal > clip_scalar_max)) {
                skip = true;
                break;
              }
            }
          }
#else
                    if ((compClipScalar >= 0) && ((unst->getScalar(n0, compClipScalar) > clip_scalar_max) || (unst->getScalar(n0, compClipScalar) < clip_scalar_min) || (unst->getScalar(n1, compClipScalar) > clip_scalar_max) || (unst->getScalar(n1, compClipScalar) < clip_scalar_min)))
                    {
                        skip = true;
                        break;
                    }
#endif
                }

                if (skip)
                    continue;
            }
            // ### mode !filter_by_cell still generates all vertices!
            // no, not anymore, 2007-xx-xx

            for (int j = 0; j < 12; j++)
            {
                int n0, n1;
                float s0, s1;
                int master, slave;

                n0 = nodenr[edge[j][0]];
                n1 = nodenr[edge[j][1]];
#if !COMPUTE_RIDGE
                s0 = surface_data[n0];
                s1 = surface_data[n1];
#else // ridge
                s0 = nodeValues[edge[j][0]];
                s1 = nodeValues[edge[j][1]];
#endif
                // check sign of second derivative
                if (0)
                { // ################3
                    vec3 evals0, evals1;
                    temp->getVector3(n0, compEigenvals, evals0);
                    temp->getVector3(n1, compEigenvals, evals1);

                    //if (strcmp(extremum, EX_MIN) == 0) {
                    if (extremumNr == EXNR_MIN)
                    {
                        if ((evals0[0] < 0.0) || (evals1[0] < 0.0))
                        {
                            // largest eigenvalue < 0  -> skip
                            continue;
                        }
                    }
                    else
                    {
                        if ((evals0[2] > 0.0) || (evals1[2] > 0.0))
                        {
                            // smallest eigenvalue > 0  -> skip
                            continue;
                        }
                    }
                }

                if ((s0 < level) != (s1 < level))
                {

#if 1
                    if (!filter_by_cell)
                    {
                        // filter on intersection-level

                        fvec3 x0, x1, x;
                        float t;
                        if (!useBisection)
                        {
                            if (s0 != s1)
                                t = (s0 - level) / (s0 - s1);
                            else
                                t = 0.5; // TODO ##### ok?
                        }
                        else
                        {
#if 0
              t = findTByBisection(temp, compGradient, temp, compHess, false /*PV*/, extremumNr == EXNR_MAX, n0, n1);
#else
                            t = findTByBisection(temp, compGradient, temp, compHess, true /*PV*/, extremumNr == EXNR_MAX, n0, n1);
#endif
                        }

                        unst->getCoords(n0, x0);
                        unst->getCoords(n1, x1);

                        // Avoid zero-length edges in ridge TODO:ok?
                        float eps = 1e-4;
                        if (t < eps)
                            t = eps;
                        if (t > 1 - eps)
                            t = 1 - eps;

                        fvec3lerp(x0, x1, t, x);
                        vec3 xd;
                        fvec3tovec3(x, xd);

                        bool skip2, exception;
                        int exceptionCnt = 0; // makes no sense on vertex level TODO
#if 0 // DELETEME
            filteringConditions(unst,
                                excludeNodes,
                                temp, compHess, compEigenvals,
                                compScalar,
                                compClipScalar, xd,
                                Hess_extr_eigenval_min, scalar_min, scalar_max,
                                clip_scalar_min, clip_scalar_max,
                                extremum, combine_exceptions,
                                skip2, exception, exceptionCnt);
#else
                        if (excludeNodes && (excludeNodes[n0] || excludeNodes[n1]))
                        {
                            skip2 = true;
                        }
                        else
                        {
                            filteringConditions(unst,
                                                NULL,
                                                temp, compHess, compEigenvals,
                                                compScalar,
                                                compClipScalar, xd,
                                                Hess_extr_eigenval_min,
                                                scalar_min,
                                                scalar_min_cell,
                                                scalar_max,
                                                clip_scalar_min, clip_scalar_max,
                                                extremumNr, combine_exceptions,
                                                skip2, exception, exceptionCnt);
                        }
#endif
                        if (skip2 || exception)
                        {
                            continue;
                        }
                    }
#endif

                    Nodeinfo *n;
                    int m;

                    nodeinfo[n0].marked = nodeinfo[n1].marked = TRUE;

                    if (n0 < n1)
                    {
                        master = n0;
                        slave = n1;
                    }
                    else
                    {
                        master = n1;
                        slave = n0;
                    }
                    n = &(nodeinfo[master]);

                    // see if edge is already stored
                    bool exists = false;
                    for (int l = 0; l < n->niedges; l++)
                    {
                        if (n->iedge[l].partner == slave)
                            exists = true;
                    }
                    if (exists)
                        continue;

                    m = ++n->niedges;
                    n->iedge = (IEdge *)realloc(n->iedge, m * sizeof(IEdge));
                    n->iedge[m - 1].partner = slave;

                    markedEdgeCnt++;
                }
            }
        }
        //else if (type == UCD_TETRAHEDRON) {
        else if (type == Unstructured::CELL_TET)
        {
            count_tet++;
            int *node_list = unst->getCellNodesAVS(i);
            nodenr[0] = *node_list++;
            nodenr[1] = *node_list++;
            nodenr[2] = *node_list++;
            nodenr[3] = *node_list++;

#if COMPUTE_RIDGE
            // make eigenvectors at nodes consistent over cell
            // and then compute values at nodes
            float nodeValuesAVS[4], nodeValues[4];
            //if (strcmp(mode, MO_CELL_PCA) == 0) {
            if (modeNr == MONR_CELL_PCA)
            {
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByPCA(unst, compHessEigenvect, compGradient, unst->getCellNodesAVS(i), 4, nodeValuesAVS);
#else
                computeConsistentNodeValuesByPCA(temp, compEigenvectExtr, temp, compGradient, temp->getCellNodesAVS(i), 4, nodeValuesAVS, NULL);
#endif
            }
            else
            {
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByEdgePCA(unst, compHessEigenvect, unst, compGradient, i, nodeValuesAVS);
#else
                computeConsistentNodeValuesByEdgePCA(temp, compEigenvectExtr, compGradient, i, nodeValuesAVS, NULL, NULL);
#endif
            }

            // adapt node order
            nodeValues[0] = nodeValuesAVS[0];
            nodeValues[1] = nodeValuesAVS[1];
            nodeValues[2] = nodeValuesAVS[2];
            nodeValues[3] = nodeValuesAVS[3];
#endif

            for (int j = 0; j < 6; j++)
            {
                int n0, n1;
                float s0, s1;
                int master, slave;

                n0 = nodenr[tedge[j][0]];
                n1 = nodenr[tedge[j][1]];
#if !COMPUTE_RIDGE
                s0 = surface_data[n0];
                s1 = surface_data[n1];
#else // ridge
                s0 = nodeValues[edge[j][0]];
                s1 = nodeValues[edge[j][1]];
#endif

                // check sign of second derivative
                if (0)
                { // ##############
                    vec3 evals0, evals1;
                    temp->getVector3(n0, compEigenvals, evals0);
                    temp->getVector3(n1, compEigenvals, evals1);

                    //if (strcmp(extremum, EX_MIN) == 0) {
                    if (extremumNr == EXNR_MIN)
                    {
                        if ((evals0[0] < 0.0) || (evals1[0] < 0.0))
                        {
                            // largest eigenvalue < 0  -> skip
                            continue;
                        }
                    }
                    else
                    {
                        if ((evals0[2] > 0.0) || (evals1[0] > 0.0))
                        {
                            // smallest eigenvalue > 0  -> skip
                            continue;
                        }
                    }
                }

                if ((s0 < level) != (s1 < level))
                {

#if 0 // ############### TODO (repaste and adapt from HEX above)
          if (!filter_by_cell) {
            // filter on intersection-level

            fvec3 x0, x1, x;
            float t;
            if (s0 != s1) t = (s0 - level) / (s0 - s1);
            else          t = 0.5; // TODO ##### ok?

            unst->getCoords(n0, x0);
            unst->getCoords(n1, x1);
            
            // Avoid zero-length edges in ridge TODO:ok?
            float eps = 1e-4;
            if (t < eps) t = eps;
            if (t > 1-eps) t = 1-eps;
            
            fvec3lerp(x0, x1, t, x);
            vec3 xd;
            fvec3tovec3(x, xd);
            
            bool skip2, exception;
            int exceptionCnt = 0; // makes no sense on vertex level TODO
            filteringConditions(unst,
                                excludeNodes,
                                temp, compHess, compEigenvals,
                                compScalar,
                                compClipScalar, xd,
                                Hess_extr_eigenval_min, scalar_min, scalar_max,
                                clip_scalar_min, clip_scalar_max,
                                extremumNr, combine_exceptions,
                                skip2, exception, exceptionCnt);
            if (skip2 || exception) {
              continue;
            }
          }
#endif

                    Nodeinfo *n;
                    int m;

                    nodeinfo[n0].marked = nodeinfo[n1].marked = TRUE;

                    if (n0 < n1)
                    {
                        master = n0;
                        slave = n1;
                    }
                    else
                    {
                        master = n1;
                        slave = n0;
                    }
                    n = &(nodeinfo[master]);

                    // see if edge is already stored
                    bool exists = false;
                    for (int l = 0; l < n->niedges; l++)
                    {
                        if (n->iedge[l].partner == slave)
                            exists = true;
                    }
                    if (exists)
                        continue;

                    m = ++n->niedges;
                    n->iedge = (IEdge *)realloc(n->iedge, m * sizeof(IEdge));
                    n->iedge[m - 1].partner = slave;
                    markedEdgeCnt++;
                }
            }
        }
        else
        {
            count_other++;
            //DELETEME node_list += nvertices[type];
        }
    }
    printf("marked %d edges\n", markedEdgeCnt);
}

void createVerticesAndNormals(UniSys *us,
                              //UCD_structure* ucd,
                              Unstructured *unst,
                              Unstructured *temp, int compGradient, int compHess, int compEigenvectExtr, bool useBisection, int extremumNr, double level, Nodeinfo *nodeinfo, int clip_lower_data, int clip_higher_data,
#if SUPPORT_COLORMAP
                              AVScolormap *cmap, float RGB[MAXCOLORS][3], float lower, float upper, float *color_data,
#endif
                              //GEOMobj *obj,
                              UniGeom *ugeom,
                              int &last_vertex_nr, bool **clip)
{
    //AVSmodify_parameter("status", AVS_VALUE, "Creating vertices ...", 0, 0);
    //AVSmodule_status("Creating vertices ...", 0);
    //us->moduleStatus("creating vertices ...", 0);
    us->info("creating vertices ...", 0);
    //for (int i = 0; i < ucd->nnodes; i++) {
    for (int i = 0; i < unst->nNodes; i++)
    {
        Nodeinfo *n;

        n = &(nodeinfo[i]);
        for (int j = 0; j < n->niedges; j++)
        {
            IEdge *e;
            float s0, s1;
            fvec3 x0, x1, x;
            float t;
            //fvec3 gradient, normal;

            e = &(n->iedge[j]);
#if !COMPUTE_RIDGE
            s0 = surface_data[i];
            s1 = surface_data[e->partner];
#else // ridge

            // make eigenvectors at nodes consistent over edge
            // and then compute values at nodes
            // ############################## this seems to be a bug, EV should be
            // made consistent over cell, not only over edge (CONSISTENCY ..)
            int nodes[2] = { i, e->partner };
            float nodeValues[2];
#if EXTERNAL_DATA // DELETEME:
            computeConsistentNodeValuesByPCA(unst, compHessEigenvect, compGradient, nodes, 2, nodeValues);
#else
            computeConsistentNodeValuesByPCA(temp, compEigenvectExtr, temp, compGradient, nodes, 2, nodeValues, NULL);
#endif

            s0 = nodeValues[0];
            s1 = nodeValues[1];
#endif
#if 0 // orig:
      t = (s0 - level) / (s0 - s1);
#else
            if (!useBisection)
            {
                if (s0 != s1)
                    t = (s0 - level) / (s0 - s1);
                else
                    t = 0.5; // TODO ##### ok?
            }
            else
            {
#if 0
        t = findTByBisection(temp, compGradient, temp, compHess, false /*PV*/, extremumNr == EXNR_MAX, i, e->partner);
#else
                t = findTByBisection(temp, compGradient, temp, compHess, true /*PV*/, extremumNr == EXNR_MAX, i, e->partner);
#endif
            }
#endif
            unst->getCoords(i, x0);
            unst->getCoords(e->partner, x1);

            // Avoid zero-length edges in ridge TODO:ok?
            float eps = 1e-4;
            if (t < eps)
                t = eps;
            if (t > 1 - eps)
                t = 1 - eps;

            fvec3lerp(x0, x1, t, x);
            //DELETEME GEOMadd_vertices(obj, x, 1, GEOM_COPY_DATA);
            ugeom->addVertices(x, 1);

            bool clipIt = false;
#if SUPPORT_COLORMAP
            if (cmap != NULL)
            {
                float c0, c1, c;

                c0 = color_data[i];
                c1 = color_data[e->partner];
                c = c0 + t * (c1 - c0);
                if (clip_lower_data && c < lower)
                    clipIt = true;
                if (clip_higher_data && c > upper)
                    clipIt = true;
                int ci = (int)((MAXCOLORS - 1.) / (upper - lower) * (c - lower));
                if (ci >= MAXCOLORS)
                    ci = MAXCOLORS - 1;
                else if (ci < 0)
                    ci = 0;
                GEOMadd_float_colors(obj, RGB[ci], 1, GEOM_COPY_DATA);
            }
#endif

            e->vertexnr = ++last_vertex_nr;

            *clip = (bool *)realloc(*clip, (e->vertexnr + 1) * sizeof(bool));
            (*clip)[e->vertexnr] = clipIt;

#if NORMALS_FROM_GRAD

            fvec3lerp(nodeinfo[i].grad, nodeinfo[e->partner].grad, t, gradient);
            fvec3norm(gradient, normal);
            if (flip_normals)
            {
                normal[0] *= -1;
                normal[1] *= -1;
                normal[2] *= -1;
            }
            GEOMadd_normals(obj, normal, 1, GEOM_COPY_DATA);
#endif
        }
    }
}

int generateTriangles(UniSys *us,
                      //UCD_structure* ucd,
                      bool *excludeNodes,
                      Unstructured *unst,
                      int cell,
                      //char *mode,
                      int modeNr,
                      //char *extremum,
                      int extremumNr,
                      int filter_by_cell,
                      int compScalar,
                      Unstructured *temp,
                      int compGradient,
                      int compHess,
                      int compEigenvals,
                      int compEigenvectExtr,
                      double level,
                      double scalar_min,
                      float *scalar_min_cell,
                      double scalar_max,
                      int compClipScalar,
                      double clip_scalar_min,
                      double clip_scalar_max,
                      double Hess_extr_eigenval_min,
                      int combine_exceptions,
                      int max_exceptions,
                      float clip_min_x,
                      float clip_max_x,
                      float clip_min_y,
                      float clip_max_y,
                      float clip_min_z,
                      float clip_max_z,
                      //int *node_list,
                      Nodeinfo *nodeinfo,
                      bool *clip,
                      int &count_hit,
                      int &count_tria,
                      int &falsePositiveIntersections,
                      int &falseNegativeIntersections,
                      //GEOMobj *obj,
                      UniGeom *ugeom,
                      std::vector<int> *triangleConn)
{ // cell: unused if <0, otherwise only triangles for this cell are generated
    // scalar_min_cell: if not NULL, scalar_min is ignored and this array defines
    //                  a scalar_min for each cell
    // returns <0 if error

    //AVSmodify_parameter("status", AVS_VALUE, "Generating triangles ...", 0, 0);
    //AVSmodule_status("Generating triangles ...", 0);
    //us->moduleStatus("generating triangles ...", 0);
    us->info("generating triangles ...");
    //if (cell<0) {
    // node_list = ucd->node_list;
    //}
    //else {
    // node_list = ucd->node_list + 8 * cell; // ### assuming hex cells
    //}
    //std::vector<int> triangleConn;
    //int falsePositiveIntersections = 0;
    //int falseNegativeIntersections = 0;
    //for (int i = (cell<0 ? 0 : cell); i < (cell<0 ? ucd->ncells : cell+1); i++) {
    for (int i = (cell < 0 ? 0 : cell); i < (cell < 0 ? unst->nCells : cell + 1); i++)
    {
        int type;
        int nodenr[8];

        //type = ucd->cell_type[i];
        type = unst->getCellType(i);
        //if (type == UCD_HEXAHEDRON) {
        if (type == Unstructured::CELL_HEX)
        {
            float s[8];
            int mctype;
            int ntria;

            /* Change from UCD HEX order to usual order */
            int *node_list = unst->getCellNodesAVS(i);
            nodenr[4] = *node_list++;
            nodenr[5] = *node_list++;
            nodenr[7] = *node_list++;
            nodenr[6] = *node_list++;
            nodenr[0] = *node_list++;
            nodenr[1] = *node_list++;
            nodenr[3] = *node_list++;
            nodenr[2] = *node_list++;

#if COMPUTE_RIDGE
            // make eigenvectors at nodes consistent over cell
            // and then compute values at nodes
            float nodeValuesAVS[8], nodeValues[8];
            //if (strcmp(mode, MO_CELL_PCA) == 0) {
            if (modeNr == MONR_CELL_PCA)
            {
#if EXTERNAL_DATA // DELELTEME:
                computeConsistentNodeValuesByPCA(unst, compHessEigenvect, compGradient, unst->getCellNodesAVS(i), 8, nodeValuesAVS);
#else
                computeConsistentNodeValuesByPCA(temp, compEigenvectExtr, temp, compGradient, temp->getCellNodesAVS(i), 8, nodeValuesAVS, NULL);
#endif
            }
            else
            {
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByEdgePCA(unst, compHessEigenvect, compGradient, i, nodeValuesAVS, &falsePositiveIntersections, &falseNegativeIntersections);
#else
                computeConsistentNodeValuesByEdgePCA(temp, compEigenvectExtr, compGradient, i, nodeValuesAVS, &falsePositiveIntersections, &falseNegativeIntersections);
#endif
            }

            // adapt node order
            nodeValues[4] = nodeValuesAVS[0];
            nodeValues[5] = nodeValuesAVS[1];
            nodeValues[7] = nodeValuesAVS[2];
            nodeValues[6] = nodeValuesAVS[3];
            nodeValues[0] = nodeValuesAVS[4];
            nodeValues[1] = nodeValuesAVS[5];
            nodeValues[3] = nodeValuesAVS[6];
            nodeValues[2] = nodeValuesAVS[7];
#endif

            // skip cells that contain a node which does not
            // obey the inequalities for a ridge
            if (filter_by_cell)
            { // ################3
                bool skip = false;
                for (int j = 0; j < 12; j++)
                {
                    int n0, n1;
                    //float s0, s1;
                    //int master, slave;

                    n0 = nodenr[edge[j][0]];
                    n1 = nodenr[edge[j][1]];

                    vec3 evals0, evals1;
                    temp->getVector3(n0, compEigenvals, evals0);
                    temp->getVector3(n1, compEigenvals, evals1);

                    //if (strcmp(extremum, EX_MIN) == 0) {
                    if (extremumNr == EXNR_MIN)
                    {
                        if ((evals0[0] < 0.0) || (evals1[0] < 0.0))
                        {
                            // largest eigenvalue < 0  -> skip
                            skip = true;
                            break;
                        }
                    }
                    else
                    {
                        if ((evals0[2] > 0.0) || (evals1[2] > 0.0))
                        {
                            // smallest eigenvalue > 0  -> skip
                            skip = true;
                            break;
                        }
                    }

                    if ((unst->getScalar(n0, compScalar) < scalar_min) || (unst->getScalar(n1, compScalar) < scalar_min) || (unst->getScalar(n0, compScalar) > scalar_max) || (unst->getScalar(n1, compScalar) > scalar_max))
                    {
                        skip = true;
                        break;
                    }

// clip by scalar
#if 0
          if (compClipScalar >= 0) {
            vec3 pos0, pos1;
            // get node positions
            unst->getCoords(n0, pos0);
            unst->getCoords(n1, pos1);

            // find cell and clip
            if (unst2->findCell(pos0)) {
              double scal = unst2->interpolateScalar();
                                
              if ((scal < *clip_scalar_min) || (scal > clip_scalar_max)) {
                skip = true;
                break;
              }
            }
            if (unst2->findCell(pos1)) {
              double scal = unst2->interpolateScalar();
                                
              if ((scal < *clip_scalar_min) || (scal > clip_scalar_max)) {
                skip = true;
                break;
              }
            }
          }
#else
                    if ((compClipScalar >= 0) && ((unst->getScalar(n0, compClipScalar) > clip_scalar_max) || (unst->getScalar(n0, compClipScalar) < clip_scalar_min) || (unst->getScalar(n1, compClipScalar) > clip_scalar_max) || (unst->getScalar(n1, compClipScalar) < clip_scalar_min)))
                    {
                        skip = true;
                        break;
                    }
#endif
                }

                if (skip)
                    continue;
            }

// test for excluded (disabled) nodes
#if 1 // reject cell if any of its nodes is excluded (disabled)
            // triangle-based rejection below (but does not work correctly)
            {
                bool skipCell = false;
                int *nodes = temp->getCellNodesAVS(i);
                for (int n = 0; n < 8; n++)
                {
                    if (excludeNodes && excludeNodes[nodes[n]])
                    {
                        skipCell = true;
                        break;
                    }
                }
                if (skipCell)
                    continue;
            }
#endif

            mctype = 0;
            for (int j = 7; j >= 0; j--)
            {
                mctype *= 2;
#if !COMPUTE_RIDGE
                s[j] = surface_data[nodenr[j]];
#else // ridge
                s[j] = nodeValues[j];
#endif
                //if (s[j] >= level) mctype++;
                if (!(s[j] < level))
                    mctype++;
            }
            ntria = mc_table[mctype].len;
            if (ntria > 0)
                count_hit++;
            for (int j = 0; j < ntria; j++)
            {
                int *ptr;
                int k;
                int indices[3];

                ptr = &(mc_table[mctype].tria[j][0][0]);
                bool clipIt = false;
                for (k = 0; k < 3; k++)
                {
                    int n0, n1;
                    int master, slave;
                    int l;

                    n0 = nodenr[*ptr++];
                    n1 = nodenr[*ptr++];
                    if (n0 < n1)
                    {
                        master = n0;
                        slave = n1;
                    }
                    else
                    {
                        master = n1;
                        slave = n0;
                    }
                    for (l = 0; l < nodeinfo[master].niedges; l++)
                    {
                        if (nodeinfo[master].iedge[l].partner == slave)
                            break;
                    }

// reject triangle if any of involved nodes is disabled (excluded)
// this does not work because MC case was determined without
// filtering the edge intersections
#if 0
          if (excludeNodes && (excludeNodes[n0] || excludeNodes[n1])) {
            clipIt = true;
            break;
          }
#endif

                    if (!filter_by_cell && (l == nodeinfo[master].niedges))
                    {
                        // vertex got suppressed by filter criteria
                        // (or bad connectivity) HACK?
                        //printf("HACK? excl[n0]=%d excl[n1]=%d cell=%d\n", excludeNodes[n0], excludeNodes[n1], i);
                        clipIt = true;
                        break;
                    }

#if 0 //debug ####
          if (i < 3) {
            printf("cell=%d master=%d slave=%d\n", i, master, slave);
          }
#endif
                    if (l == nodeinfo[master].niedges || nodeinfo[master].iedge[l].partner != slave)
                    {
                        fprintf(stderr, "Bad connectivity in hex cell %d: master=%d slave=%d l[%d]=%d master-partner=%d\n",
                                i, master, slave, nodeinfo[master].niedges, l,
                                (l != nodeinfo[master].niedges ? nodeinfo[master].iedge[l].partner : -1));
#if 1 // orig:
#if !EXTERNAL_DATA
                        delete temp;
#endif
                        return -1;
#else // HACK ###
                        clipIt = true;
                        break;
#endif
                    }
                    indices[k] = nodeinfo[master].iedge[l].vertexnr;
                    if (clip[indices[k]])
                    {
                        clipIt = true;
                        break;
                    }
                }

                if (clipIt)
                    continue;

                if (!filter_by_cell)
                {
                    // filter out triangles that contain a vertex
                    // that does not fulfill filter conditions

                    bool skip = false;
                    int exceptionCnt = 0;
                    for (int v = 0; v < 3; v++)
                    {

                        // DELETEME
                        //vec3 pos = { PH(obj).verts.l[(indices[v]-1)*3+0],
                        //             PH(obj).verts.l[(indices[v]-1)*3+1],
                        //             PH(obj).verts.l[(indices[v]-1)*3+2] };
                        vec3 pos;
                        ugeom->getVertex(indices[v] - 1, pos);

                        bool exception;
                        filteringConditions(unst,
                                            //excludeNodes,
                                            NULL,
                                            temp, compHess, compEigenvals,
                                            compScalar,
                                            compClipScalar, pos,
                                            Hess_extr_eigenval_min,
                                            scalar_min,
                                            scalar_min_cell,
                                            scalar_max,
                                            clip_scalar_min, clip_scalar_max,
                                            extremumNr, combine_exceptions,
                                            skip, exception, exceptionCnt);

#if 1 // ### added for geom clipping 2008-03-28
                        if (pos[0] < clip_min_x || pos[0] > clip_max_x || pos[1] < clip_min_y || pos[1] > clip_max_y || pos[2] < clip_min_z || pos[2] > clip_max_z)
                        {
                            skip = true;
                            clipIt = true;
                            break;
                        }
#endif

                        if (skip)
                        {
                            printf("generateTriangles: skipped due to filteringConditions() at cell %d\n", i);
                            break;
                        }
                        if (exception)
                            continue;
                    }
#if !FILTER_BY_EXCEPTION_CNT
                    if (skip)
                        clipIt = true;
#else
                    if (exceptionCnt >= max_exceptions)
                        clipIt = true;
#endif
                }

                // #########################
                if (!clipIt)
                {
                    //if (false) {
                    count_tria++;
                    //GEOMadd_polygon(obj, 3, indices, 0, GEOM_COPY_DATA);
                    triangleConn->push_back(indices[0]);
                    triangleConn->push_back(indices[1]);
                    triangleConn->push_back(indices[2]);
                }
            }
        }
        //else if (type == UCD_TETRAHEDRON) {
        else if (type == Unstructured::CELL_TET)
        {
            float s[4];
            int ntria;
            int mttype;

            int *node_list = unst->getCellNodesAVS(i);
            nodenr[0] = *node_list++;
            nodenr[1] = *node_list++;
            nodenr[2] = *node_list++;
            nodenr[3] = *node_list++;

#if COMPUTE_RIDGE
            // make eigenvectors at nodes consistent over cell
            // and then compute values at nodes
            float nodeValuesAVS[4], nodeValues[4];
            //if (strcmp(mode, MO_CELL_PCA) == 0) {
            if (modeNr == MONR_CELL_PCA)
            {
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByPCA(unst, compHessEigenvect, compGradient, unst->getCellNodesAVS(i), 4, nodeValuesAVS);
#else
                computeConsistentNodeValuesByPCA(temp, compEigenvectExtr, temp, compGradient, temp->getCellNodesAVS(i), 4, nodeValuesAVS, NULL);
#endif
            }
            else
            {
#if EXTERNAL_DATA // DELETEME:
                computeConsistentNodeValuesByEdgePCA(unst, compHessEigenvect, compGradient, i, nodeValuesAVS);
#else
                computeConsistentNodeValuesByEdgePCA(temp, compEigenvectExtr, compGradient, i, nodeValuesAVS, NULL, NULL);
#endif
            }

            // adapt node order
            nodeValues[0] = nodeValuesAVS[0];
            nodeValues[1] = nodeValuesAVS[1];
            nodeValues[2] = nodeValuesAVS[2];
            nodeValues[3] = nodeValuesAVS[3];
#endif

            // TODO: implement skip_by_cell

            mttype = 0;
            for (int j = 3; j >= 0; j--)
            {
                mttype *= 2;
#if !COMPUTE_RIDGE
                s[j] = surface_data[nodenr[j]];
#else // ridge
                s[j] = nodeValues[j];
#endif
                //if (s[j] >= level) mttype++;
                if (!(s[j] < level))
                    mttype++;
            }
            ntria = mt_table[mttype].len;
            if (ntria > 0)
                count_hit++;
            for (int j = 0; j < ntria; j++)
            {
                int *ptr;
                int k;
                int indices[3];

                ptr = &(mt_table[mttype].tria[j][0][0]);
                bool clipIt = false;
                for (k = 0; k < 3; k++)
                {
                    int n0, n1;
                    int master, slave;
                    int l;

                    n0 = nodenr[*ptr++];
                    n1 = nodenr[*ptr++];
                    if (n0 < n1)
                    {
                        master = n0;
                        slave = n1;
                    }
                    else
                    {
                        master = n1;
                        slave = n0;
                    }
                    for (l = 0; l < nodeinfo[master].niedges; l++)
                    {
                        if (nodeinfo[master].iedge[l].partner == slave)
                            break;
                    }

                    if (!filter_by_cell && (l == nodeinfo[master].niedges))
                    {
                        // vertex got suppressed by filter criteria
                        // (or bad connectivity) HACK?
                        clipIt = true;
                        break;
                    }

                    if (l == nodeinfo[master].niedges || nodeinfo[master].iedge[l].partner != slave)
                    {
                        fprintf(stderr, "Bad connectivity in tet cell %d\n", i);
#if !EXTERNAL_DATA
                        delete temp;
#endif
                        return -1;
                    }
                    indices[k] = nodeinfo[master].iedge[l].vertexnr;
                    if (clip[indices[k]])
                    {
                        clipIt = true;
                        break;
                    }
                }

                if (clipIt)
                    continue;

                if (!filter_by_cell)
                {
                    // filter out triangles that contain a vertex
                    // that does not fulfill filter conditions

                    bool skip = false;
                    int exceptionCnt = 0;
                    for (int v = 0; v < 3; v++)
                    {

                        // DELETEME
                        //vec3 pos = { PH(obj).verts.l[(indices[v]-1)*3+0],
                        //            PH(obj).verts.l[(indices[v]-1)*3+1],
                        //            PH(obj).verts.l[(indices[v]-1)*3+2] };
                        vec3 pos;
                        ugeom->getVertex(indices[v] - 1, pos);

                        bool exception;
                        filteringConditions(unst,
                                            excludeNodes,
                                            temp, compHess, compEigenvals,
                                            compScalar,
                                            compClipScalar, pos,
                                            Hess_extr_eigenval_min,
                                            scalar_min,
                                            scalar_min_cell,
                                            scalar_max,
                                            clip_scalar_min, clip_scalar_max,
                                            extremumNr, combine_exceptions,
                                            skip, exception, exceptionCnt);

#if 1 // ### added for geom clipping 2008-03-28
                        if (pos[0] < clip_min_x || pos[0] > clip_max_x || pos[1] < clip_min_y || pos[1] > clip_max_y || pos[2] < clip_min_z || pos[2] > clip_max_z)
                        {
                            skip = true;
                            clipIt = true;
                            break;
                        }
#endif

                        if (skip)
                        {
                            break;
                        }
                        if (exception)
                            continue;
                    }
#if !FILTER_BY_EXCEPTION_CNT
                    if (skip)
                        clipIt = true;
#else
                    if (exceptionCnt >= max_exceptions)
                        clipIt = true;
#endif
                }

                if (!clipIt)
                {
                    count_tria++;
                    //GEOMadd_polygon(obj, 3, indices, 0, GEOM_COPY_DATA);
                    triangleConn->push_back(indices[0]);
                    triangleConn->push_back(indices[1]);
                    triangleConn->push_back(indices[2]);
                }
            }
        }
        else
        {
            //DELETEME node_list += nvertices[type];
        }
    }

    return 1;
}

#endif
