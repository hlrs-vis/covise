/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_UNSTRUCTURED_GRID_H
#define CO_DO_UNSTRUCTURED_GRID_H

enum ELEM_TYPE
{
    TYPE_NONE = 0,
    TYPE_BAR = 1,
    TYPE_TRIANGLE = 2,
    TYPE_QUAD = 3,
    TYPE_TETRAHEDER = 4,
    TYPE_PYRAMID = 5,
    TYPE_PRISM = 6,
    TYPE_HEXAGON = 7,
    TYPE_HEXAEDER = 7,
    TYPE_POINT = 10,
    TYPE_POLYHEDRON = 11
};

#ifndef CELL_TYPES_ONLY
#include "coDoGrid.h"

/***********************************************************************\
 **                                                                     **
 **   Unstructured classe Routines                  Version: 1.0        **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of unstructured grids     **
 **                  in a distributed manner.                           **
 **                                                                     **
 **   Classe      : coDoUnstructuredGrid                                 **
 **                                                                     **
 **                                                                     **
 **   Copyright (C) 1995     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : Uwe Woessner                                       **
 **                                                                     **
 **   History      :                                                    **
 **                  10.02.95  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
\***********************************************************************/
namespace covise
{

DOEXPORT extern int UnstructuredGrid_Num_Nodes[20];

class coDoOctTree;

class DOEXPORT coDoUnstructuredGrid : public coDoGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    enum
    {
        SHM_OBJ = 12
    };
    coIntShm numelem; // number of elements
    coIntShm numconn; // number of connections
    coIntShm numcoord; // number of coords
    coIntShm numneighbor; // number of neighbors
    coFloatShmArray x_coord; // coordinates in x-direction (length numcoord)
    coFloatShmArray y_coord; // coordinates in y-direction (length numcoord)
    coFloatShmArray z_coord; // coordinates in z-direction (length numcoord)
    coIntShmArray connections; // connection list (length numconn)
    coIntShmArray elements; // element list (length numelem)
    coIntShmArray elementtypes; // elementtypes list (length numelem)
    coIntShmArray neighborlist; // neighborlist list (length numneighbor)
    coIntShmArray neighborindex; // neighborindex list (length numcoord)
    mutable const coDistributedObject *oct_tree;

    int testACell(float *v_interp, const float *point,
                  int cell, int no_arrays, int array_dim,
                  float tolerance, const float *const *velo) const;
    /*
            bool IsInBBox(int cell,const float *point);
            void BBoxForElement(float *cell_bbox,int elem);
            float *cellBBoxes_; // do not delete this in the destructor!!!
                                // this should be destroyed by the module.
      */

    void MakeOctTree(const char *octSurname) const;

    int hastypes;
    int hasneighbors;
    mutable int *lnl; // lists all elements that contain a certain vertex
    // consecutively
    mutable int *lnli; // points into the lnl array to the starting index for
    // each vertex
    mutable int *el, *cl, *tl;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoUnstructuredGrid *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoUnstructuredGrid(const coObjInfo &info)
        : coDoGrid(info)
        , oct_tree(NULL)
        , lnl(0)
        , lnli(0)
    {
        setType("UNSGRD", "UNSTRUCTURED GRID");
        hastypes = 0;
        hasneighbors = 0;
        if (name != (char *)0L)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };

    coDoUnstructuredGrid(const coObjInfo &info, coShmArray *arr);

    coDoUnstructuredGrid(const coObjInfo &info, int nelem, int nconn, int ncoord, int ht);
    coDoUnstructuredGrid(const coObjInfo &info, int nelem, int nconn, int ncoord, int ht, int nneighbor);
    coDoUnstructuredGrid(const coObjInfo &info, int nelem, int nconn, int ncoord,
                         int *el, int *cl, float *xc, float *yc, float *zc);
    coDoUnstructuredGrid(const coObjInfo &info, int nelem, int nconn, int ncoord,
                         int *el, int *cl, float *xc, float *yc, float *zc,
                         int *tl);

    virtual ~coDoUnstructuredGrid();

    void computeNeighborList() const;

    void freeNeighborList() const
    {
        delete[] lnl;
        lnl = NULL;
        delete[] lnli;
        lnli = NULL;
    };

    void getAddresses(int **elem, int **conn,
                      float **x_c, float **y_c, float **z_c) const
    {
        *x_c = (float *)x_coord.getDataPtr();
        *y_c = (float *)y_coord.getDataPtr();
        *z_c = (float *)z_coord.getDataPtr();
        *elem = (int *)elements.getDataPtr();
        *conn = (int *)connections.getDataPtr();
    };

    void getGridSize(int *e, int *c, int *p) const
    {
        *e = numelem;
        *c = numconn;
        *p = numcoord;
    }

    int getNumPoints() const
    {
        return numcoord;
    }

    int getNeighbor(int element, std::vector<int> face_nodes_list);
    int getNeighbor(int element, int n1, int n2, int n3, int n4);
    int getNeighbor(int element, int n1, int n2, int n3);
    int getNeighbor(int element, int n1, int n2);
    int getNeighbors(int element, int n1, int n2, int *neighbors);
    int getNeighbors(int element, int n1, int *neighbors);

    void getNeighborList(int *n, int **l, int **li) const
    {
        if (!lnl)
            computeNeighborList();
        *l = lnl;
        *li = lnli;
        *n = numconn;
    }

    void getTypeList(int **l) const
    {
        *l = (int *)elementtypes.getDataPtr();
    }

    int hasNeighborList() const
    {
        return hasneighbors;
    }

    int hasTypeList() const
    {
        return (elementtypes.get_length() > 0 ? 1 : 0);
    }

    /** set new values for sizes: only DECREASING is allowed
       *  @return   0 if ok, -1 on error
       *  @param    numElem    New size of element list
       *  @param    numConn    New size of connectivity list
       *  @param    numCoord   New size of coordinale list
       */
    int setSizes(int numElem, int numConn, int numCoord);

    // use oct-trees for cell location and field interpolation
    // Interpolates fields of any nature given a point and an input field
    // on the gitter nodes.
    // return -1: point is not in domain
    // return  0: is in domain, velo is interpolated in v_interp if this
    //    pointer and velo are not NULL.
    // Inputs: point, no_arrays, array_dim, velo.
    // Outputs: v_interp, cell.

    // For interpolation the function assumes that the input
    // onsists of vertex-based data.
    // velo is an array of $no_arrays pointers to float arrays,
    // whose contents are grouped in groups of $array_dim floats.
    // These groups are values for a point. This organisation
    // is so complicated because of the unfelicitous fact that
    // scalars and tensors are defined in a unique array and
    // vectors in three. So if you get a scalar, you typically
    // have no_arrays==1 and array_dim==1. If you get a vector,
    // no_arrays==3 and array_dim==1. And if you get a tensor,
    // then no_arrays==1 and array_dim==dimensionality of tensor type.
    // The organisation of the output is as follows:
    // cell is an array with 3 integer values determining the cell.
    // v_interp contains no_arrays groups of array_dim floats:
    // the caller is responsible for memory allocation.
    int interpolateField(float *v_interp, const float *point,
                         int *cell, int no_arrays, int array_dim,
                         float tolerance, const float *const *velo) const;

    // Map scalar fields (used in PStreamline; analogous to interpolateField)
    int mapScalarField(float *v_interp, const float *point,
                       int *cell, int no_arrays, int array_dim,
                       /*float tolerance,*/ const float *const *velo);

    // Interpolate data within the cell (analogous to testACell)
    void mapInCell(float *v_interp, const float *point,
                   int cell, int no_arrays, int array_dim,
                   const float *const *velo);

    // searchCell in OctTree (create OctTree if not available)
    int getCell(const float *point, float tolerance);

    const coDoOctTree *GetOctTree(const coDistributedObject *reuseOctTree,
                                  const char *OctTreeSurname) const;

    // checks all hexahedron elements if their connectivity
    // leads to PRISMs, QUADs or TETRAHEDRONs and fix them
    void compressConnectivity();
    int getNumConnOfElement(int elem) const;
};
}
#endif /* CELL_TYPES_ONLY */

#endif
