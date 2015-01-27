/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _IP_TRIANGLES_H
#define _IP_TRIANGLES_H

#include "nrutil.h"

class Triangles
{

private:
    ivec nodes; //nodes of every triangle
    f2ten coord; //triangle coordinates
    f2ten vdata; //projection of V3D data on the triangle
    //f2ten c2d; //2D euclidean coordinates of the triangle edges
    f2ten c2dTex; //2D euclidean coordinates of the triangle edges
    f2ten vTex;
    //f2ten trafo;

public:
    void setNodes(ivec n);
    void setCoord(f2ten c);
    inline void setVdata(const f2ten &v)
    {
        vdata = v;
    };
    void setVdata(const f2ten &v, ivec index);
    void setVdata(fvec v, int ii);
    void setVTex(const f2ten &v);
    void setVTex();
    void initC2dTex();
    inline void setC2dTex(float val, int ii, int jj)
    {
        c2dTex[ii][jj] = val;
    };
    inline ivec getNodes()
    {
        return this->nodes;
    };
    inline f2ten getCoord()
    {
        return this->coord;
    };
    fvec getCoord(int j);
    inline float getCoord(int i, int j)
    {
        return this->coord[i][j];
    };
    inline f2ten getVdata()
    {
        return this->vdata;
    };
    fvec getVdata(int j);
    inline f2ten getVTex()
    {
        return this->vTex;
    };
    fvec getVTex(int j);
    inline float getVTex(int ii, int jj)
    {
        return this->vTex[ii][jj];
    };
    inline f2ten getC2dTex()
    {
        return this->c2dTex;
    };
    fvec getC2dTex(int j);
    inline float getC2dTex(int ii, int jj)
    {
        return this->c2dTex[ii][jj];
    };
    int getCoordIndex(int which);
    float getLength();
    float getHeight();

    //test printout of coordinates
    void prCoord();

    Triangles();
    //~Triangles();
};
typedef vector<Triangles, allocator<Triangles> > trivec;

f2ten createVt(const f2ten &v, const ivec &index, const fvec &e1, int le1, const fvec &e2, int le2);

fvec solveNormalEq(const f2ten &matrix, int le1, int le2, const fvec &v);

//l pointer to vector of dimension 3
//void lsort(float* c, float* a, float* b, int* l);
ivec edgeOrder(ivec edge_order, float ab, float ac, float bc);

float doubleArea(const fvec &a, const fvec &b, const fvec &c);
float area(const fvec &a, const fvec &b, const fvec &c);

//fvec rotate(fvec a, float alpha);
#endif
