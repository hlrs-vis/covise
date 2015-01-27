/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRIANGLES_H
#define _TRIANGLES_H

#include "nrutil.h"

class Triangles
{

private:
    int id; //0...(tnum-1)
    bool packed; //triangle packed in a triangle patch ?

    ivec nodes; //nodes of every triangle
    i2ten edges; //3 edges, every edge has 2 nodes
    f2ten coord; //triangle coordinates
    i2ten neighbours; //indices of neighbour triangles
    f2ten vdata; //projection of V3D data on the triangle
    f2ten c2d; //2D euclidean coordinates of the triangle edges

public:
    inline void setId(int tri)
    {
        id = tri;
        return;
    };

    inline void setPacked(bool pa)
    {
        packed = pa;
    };

    void setNodes(ivec n);
    void setEdges(i2ten e);
    void setCoord(f2ten c);
    void setVdata(fvec v, int index);
    void setC2d(const f2ten &c);
    void shiftC2d(const fvec &shift);
    void normaliseC2d();

    inline void setC2d(float val, int ii, int jj)
    {
        c2d[ii][jj] = val;
    };

    void setNeighbour(int triangle, int neighbour_type);
    inline int getId()
    {
        return this->id;
    };

    inline bool getPacked()
    {
        return packed;
    };

    inline ivec getNodes()
    {
        return this->nodes;
    };

    inline i2ten getEdges()
    {
        return this->edges;
    };

    inline f2ten getCoord()
    {
        return this->coord;
    };

    inline f2ten getVdata()
    {
        return this->vdata;
    };

    inline f2ten getC2d()
    {
        return this->c2d;
    };

    inline float getC2d(int ii, int jj)
    {
        return this->c2d[ii][jj];
    };

    int getC2dIndex(int which);
    float getLength();
    float getHeight();

    //returns 1, if (0,0) = A
    //returns 2, if (0,0) = B
    //returns 3, if (0,0) = C
    int get2dStart();

    //returns 1 if counter-clockwise
    //returns -1 if clockwise
    int get2dOrientation();

    inline i2ten getNeighbours()
    {
        return this->neighbours;
    };

    //test printout of coordinates
    void prCoord();

    //test printout of 2D euclidean coordinates
    void prC2d();

    Triangles();
    //~Triangles();
};
typedef vector<Triangles, allocator<Triangles> > trivec;

bool commonEdge(Triangles *first, Triangles *second);
bool commonNode(Triangles *first, Triangles *second);

//l pointer to vector of dimension 3
void lsort(float *c, float *a, float *b, int *l);
ivec edgeOrder(ivec edge_order, float ab, float ac, float bc);

float doubleArea(const fvec &a, const fvec &b, const fvec &c);
float area(const fvec &a, const fvec &b, const fvec &c);

fvec rotate(fvec a, float alpha);

f2ten triXflect(f2ten coord, int start, int orientation);
//f2ten triYflect(f2ten coord, int start, int orientation);

f2ten triXshift(f2ten coord, float val);
f2ten triShift(f2ten coord, fvec val);

//recomputes triangle coordinates so that the longest side
//has nodes (0,0) & (length,0); the third node has coordinates (cx, height)
//with cx < length/2 & height > 0;
//f2ten normalise(const f2ten& tri_coord);

//provides the index of the highest triangle of a set
//criterion are the normalised 2D-euclidean coordinates
//the index list can be chosen to restrict the set of triangles
int getHighest(trivec &triangles, ivec &index, IntList &list, bool sorted);
#endif
