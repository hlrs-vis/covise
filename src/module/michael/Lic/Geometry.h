/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GEOMERTY_H
#define _GEOMETRY_H

#include "nrutil.h"
#include "Triangles.h"
#include "LicUtil.h"

//const float Q = 1.1;  //makes square area (1.1*1.1)-times larger
//then the summed area of all triangles

//contains the triangle indices of the triangles packed in a row
//it also contains the right bounds of the patch the left one's are at x=0
class Patch
{
private:
    float Q;
    int num_rows;
    i2ten rows; //first index: which row; second index: which triangle
    f2ten xbounds; //0-component: lower node, 1-component: upper node
    f2ten ybounds; //0-component: lower node, 1-component: upper node
    float square_length; //length of the square the triangles are packed into

public:
    inline void setQ(float qq)
    {
        Q = qq;
    };
    inline float getQ()
    {
        return Q;
    };
    void appendRow();

    int nextTriangle(trivec &triangles, float quad_length, int row_index,
                     const ivec &index, IntList &list,
                     fvec &ybdry, fvec &xbdry);

    void insertTriangle(int row_index, int triangle_index,
                        Triangles *triangle, f2ten bdry);

    //... a try ...
    void pack(trivec &triangles);

    //number of rows of the patch
    inline void setNumRows(int num)
    {
        num_rows = num;
    };
    inline int getNumRows()
    {
        return this->num_rows;
    };

    //the whole patch
    inline i2ten getRows()
    {
        return this->rows;
    };
    inline int getRowsSize()
    {
        return rows.size();
    }

    //a single row
    inline ivec getRows(int row_index)
    {
        return this->rows[row_index];
    };
    inline int getRowSize(int row_index)
    {
        return (rows[row_index]).size();
    }

    //a specific element of a specific row
    inline int getRows(int row_index, int col_index)
    {
        return this->rows[row_index][col_index];
    };

    //the right bounds of each row of the patch
    inline f2ten getXbounds()
    {
        return this->xbounds;
    };
    inline f2ten getYbounds()
    {
        return this->ybounds;
    };

    //a single row
    fvec getXbounds(int row_index);

    fvec getYbounds(int row_index);

    //a specific element of a specific row
    inline float getXbounds(int row_index, int component)
    {
        return xbounds[component][row_index];
    };
    inline void setXbounds(int row_index, int component, float value)
    {
        xbounds[component][row_index] = value;
    };
    inline float getYbounds(int row_index, int component)
    {
        return ybounds[component][row_index];
    };
    inline void setYbounds(int row_index, int component, float value)
    {
        ybounds[component][row_index] = value;
    };

    inline void setSquareLength(float sl)
    {
        square_length = sl;
    };
    inline float getSquareLength()
    {
        return this->square_length;
    };

    Patch();
};

//**********************************************************************

void points(fvec &p1, fvec &p2, fvec &p3, const f2ten &nodes, int first, int second, int third);

void tryPoints(fvec &p1, fvec &p2, fvec &p3, fvec xbdry, fvec ybdry, fvec &xbNew, fvec &ybNew);

float waste(fvec p1, fvec p2, fvec p3, fvec xbdry, fvec ybdry, fvec xbNew, fvec ybNew);

void rotate180(fvec &p1, fvec &p2, fvec &p3);

//**********************************************************************

//input: velocity at the three nodes, v[0] corresponds to coord [0], ...
void project2triangle(Triangles *tri, const f2ten &v);

//returns vector lambda
//bary(point, coord) = sum( i=1...3, lambda[i]*coord[i] )
//for vectors v1, v2, v3 defined on the three vertices create tensor
//v=(v1,v2,v3); then bary(point, v) = sum( i=1...3, lambda[i]*v[i] )
fvec bary(const fvec &point, const f2ten &coord);

float quadLength(trivec &triangles);

void modify(int *start, int *orientation);
#endif
