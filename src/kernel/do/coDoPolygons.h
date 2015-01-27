/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_POLYGONS_H
#define CO_DO_POLYGONS_H

#include "coDoGrid.h"
#include "coDoLines.h"

/*
 $Log: covise_unstr.h,v $
 * Revision 1.1  1993/09/25  20:52:42  zrhk0125
 * Initial revision
 *
*/

/***********************************************************************\ 
 **                                                                     **
 **   Untructured class                              Version: 1.0       **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of an unstructured grid   **
 **                  and the data on it in a distributed manner.        **
 **                                                                     **
 **   Classes      :                                                    **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  23.06.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise
{

class coDoOctTreeP;

class DOEXPORT coDoPolygons : public coDoGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    mutable int *lnl;
    mutable int *lnli;
    mutable int numneighbor;
    mutable int numelem;
    mutable int numconn;
    int numpoints;
    mutable int *el, *cl;
    float *x_c_, *y_c_, *z_c_;
    mutable const coDistributedObject *oct_tree;

    int testACell(float *v_interp, const float *point,
                  int cell, int no_arrays, int array_dim,
                  float tolerance, const float *const *velo) const;

    void MakeOctTree(const char *octSurname) const;

    float Distance(int cell, const float *point) const;
    void Project(float *point, int cell) const;

protected:
    coDoLines *lines;
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoPolygons *cloneObject(const coObjInfo &newinfo) const;

public:
    ~coDoPolygons();
    coDoPolygons(const coObjInfo &info)
        : coDoGrid(info, "POLYGN")
        , lnl(NULL)
        , lnli(NULL)
        , oct_tree(NULL)
    {
        lines = new coDoLines(coObjInfo());
        if (name)
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
    coDoPolygons(const coObjInfo &info, coShmArray *arr);
    coDoPolygons(const coObjInfo &info, int no_p, int no_v, int no_l);
    coDoPolygons(const coObjInfo &info, int no_p, float *x_c,
                 float *y_c, float *z_c, int no_v, int *v_l, int no_pol, int *pol_l);
    int getNumPolygons() const
    {
        return lines->getNumLines();
    }
    int setNumPolygons(int num)
    {
        return lines->setNumLines(num);
    }
    int getNumVertices() const
    {
        return lines->getNumVertices();
    }
    int setNumVertices(int num)
    {
        return lines->setNumVertices(num);
    }
    int getNumPoints() const
    {
        return lines->getNumPoints();
    }
    int setNumPoints(int num)
    {
        return lines->setNumPoints(num);
    }
    int getNeighbor(int element, int n1, int n2);
    int getNeighbors(int element, int n1, int *neighbors);
    void getNeighborList(int *n, int **l, int **li) const
    {
        if (1)
        {
            computeNeighborList();
            *l = lnl;
            *li = lnli;
            numneighbor = lines->getNumVertices();
        }
        else
        {
            // *l = (int *)neighborlist.getDataPtr();
            // *li = (int *)neighborindex.getDataPtr();
        }
        *n = numneighbor;
    };
    void getAddresses(float **x_c, float **y_c, float **z_c, int **v_l, int **l_l) const
    {
        lines->getAddresses(x_c, y_c, z_c, v_l, l_l);
    };
    void computeNeighborList() const;
    // For interpolation the function assumes that the input
    // consists of vertex-based data.
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
    int interpolateField(float *v_interp, float *point,
                         int *cell, int no_arrays, int array_dim,
                         float tolerance, const float *const *velo,
                         int search_level) const;
    // make sure there is an octree, before using interpolateField
    const coDoOctTreeP *GetOctTree(const coDistributedObject *reuseOctTree,
                                   const char *OctTreeSurname) const;
};

class DOEXPORT coDoTriangles : public coDoGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    coIntShm no_of_vertices;
    coIntShmArray vertex_list;

protected:
    coDoPoints *points;
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoTriangles *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoTriangles(const coObjInfo &info)
        : coDoGrid(info, "TRITRI")
    {
        points = new coDoPoints(coObjInfo());
        if (name)
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
    coDoTriangles(const coObjInfo &info, coShmArray *arr);
    coDoTriangles(const coObjInfo &info, int no_p, int no_v);
    coDoTriangles(const coObjInfo &info, int no_p,
                  float *x_c, float *y_c, float *z_c, int no_v, int *v_l);
    int getNumTriangles() const
    {
        return no_of_vertices / 3;
    }
    int getNumVertices() const
    {
        return no_of_vertices;
    }
    int setNumVertices(int num_elem);
    int getNumPoints() const
    {
        return points->getNumPoints();
    }
    int setNumPoints(int num)
    {
        return points->setSize(num);
    };
    void getAddresses(float **x_c, float **y_c, float **z_c, int **v_l) const
    {
        points->getAddresses(x_c, y_c, z_c);
        *v_l = (int *)vertex_list.getDataPtr();
    };
};

class DOEXPORT coDoQuads : public coDoGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    coIntShm no_of_vertices;
    coIntShmArray vertex_list;

protected:
    coDoPoints *points;
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoQuads *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoQuads(const coObjInfo &info)
        : coDoGrid(info, "QUADS")
    {
        points = new coDoPoints(coObjInfo());
        if (name)
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
    coDoQuads(const coObjInfo &info, coShmArray *arr);
    coDoQuads(const coObjInfo &info, int no_p, int no_v);
    coDoQuads(const coObjInfo &info, int no_p,
              float *x_c, float *y_c, float *z_c, int no_v, int *v_l);
    int getNumTriangles() const
    {
        return no_of_vertices / 3;
    }
    int getNumVertices() const
    {
        return no_of_vertices;
    }
    int setNumVertices(int num_elem);
    int getNumPoints() const
    {
        return points->getNumPoints();
    }
    int setNumPoints(int num)
    {
        return points->setSize(num);
    };
    void getAddresses(float **x_c, float **y_c, float **z_c, int **v_l) const
    {
        points->getAddresses(x_c, y_c, z_c);
        *v_l = (int *)vertex_list.getDataPtr();
    };
};
}
#endif
