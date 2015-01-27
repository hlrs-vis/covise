/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Geometry
//
//  Container of pointers to coordinates of the input and output geometries
//
//  Initial version: 2002-05-?? Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _TRANSFORM_GEOMETRY_H_
#define _TRANSFORM_GEOMETRY_H_

class Geometry
{
public:
    /// constructor
    Geometry();
    /// destructor
    ~Geometry();
    /// keep input and ouput geometry coordinates
    void setInfo(float *, float *, float *, int no,
                 float *, float *, float *); // unsgrid
    /// keep input and ouput geometry coordinates
    void setInfo(float *, float *, float *, int nx, int ny, int nz,
                 float *, float *, float *); // strgrid
    /// keep input and ouput geometry coordinates
    void setInfo(float *, int, float *, int, float *, int,
                 float *, float *, float *); // rct
    /// keep input and ouput geometry coordinates
    void setInfo(float, float, float, float, float, float, int, int, int,
                 float, float, float, float, float, float); // uniform
    enum GeomType
    {
        NPI,
        UNS,
        STR,
        RCT,
        UNI
    };
    /// return geometry type
    GeomType getType() const;
    /// return total number of points
    int getSize() const;
    /// get number of points in the 3 direction for non-unstructured types
    void getSize(int *, int *, int *) const;

    /// dump in xc, yc, zc input (if out==false) or input (if out==true) coordinates
    void dumpGeometry(float *xc, float *yc, float *zc, bool out) const;

protected:
private:
    GeomType type_;
    float *x_c_; // input geometry
    float *y_c_;
    float *z_c_;
    float *x_o_c_; //output geometry
    float *y_o_c_;
    float *z_o_c_;
    int no_;
    int nox_;
    int noy_;
    int noz_;
    // for uniform case
    float minx_;
    float miny_;
    float minz_;
    float maxx_;
    float maxy_;
    float maxz_;
    float minx_o_;
    float miny_o_;
    float minz_o_;
    float maxx_o_;
    float maxy_o_;
    float maxz_o_;
};
#endif
