/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NURBS_H
#define NURBS_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  Class-Declaration of NURBS                               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Reiner Beller                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  23.06.97  V1.0                                                  **
\**************************************************************************/

// include files
#include <iostream.h>
#include <fstream.h>
#include <assert.h>
#include <stdlib.h>
#include "Vec.h"

// typedefs
#ifndef FLAG
typedef short FLAG;
#endif

#ifndef REAL
typedef double REAL;
#endif

#ifndef TOL // Tolerance for degree reduction
#define TOL 1.5
#endif

#ifndef remTOL // Tolerance for knot removal
#define remTOL 0.5
#endif

#ifndef U_DIR // flag for direction
#define U_DIR 0
#endif

#ifndef V_DIR // flag for direction
#define V_DIR 1
#endif

extern ofstream ErrFile;

// help functions
REAL Distance3D(const Vec3d &, const Vec3d &);
REAL Distance4D(const Vec4d &, const Vec4d &);
int Max(int, int);
int Min(int, int);

//===========================================================================
// Knot Vector
//===========================================================================

class Knot
{

private:
    int n; // number of knots
    REAL *knt; // knot vector

public:
    // constructors and destructors
    Knot()
    {
        n = 0;
        knt = NULL;
    };
    Knot(int num);
    Knot(int num, REAL *knots);
    Knot(const Knot &_k);
    ~Knot();

    //sets and gets
    void set(int i, REAL val)
    {
        knt[i] = val;
    };
    int get_n()
    {
        return n;
    };

    // operators
    REAL &operator[](int i)
    {
        return knt[i];
    };
    const REAL &operator[](int i) const
    {
        return knt[i];
    };
    Knot &operator=(const Knot &_k);
};

//===========================================================================
// Control Polygon in parameter space
//===========================================================================

class Pcpol
{

private:
    int n; // number of control points
    Vec3d *cpts; // homogeneous coordinates (u, v, w) of control points

public:
    // constructors and destructors
    Pcpol()
    {
        n = 0;
        cpts = NULL;
    };
    Pcpol(int num);
    Pcpol(int num, Vec3d *cpoints);
    Pcpol(const Pcpol &_p);
    ~Pcpol();

    // set/get
    void set(int i, Vec2d Cpt);
    void set(int i, Vec3d Cpt);

    // operators
    Vec3d &operator[](int index)
    {
        return (cpts[index]);
    };
    const Vec3d &operator[](int index) const
    {
        return (cpts[index]);
    };
    Pcpol &operator=(const Pcpol &_p);
};

//===========================================================================
// Control Polygon in object space
//===========================================================================

class Cpol
{

private:
    int n; // number of control points
    Vec4d *cpts; // homogeneous coordinates (x, y, z, w) of control points

public:
    // constructors and destructors
    Cpol()
    {
        n = 0;
        cpts = NULL;
    };
    Cpol(int num);
    Cpol(int num, Vec4d *cpoints);
    Cpol(const Cpol &_c);
    ~Cpol();

    // set/get
    void set(int i, Vec3d Cpt);
    void set(int i, Vec4d Cpt);
    int get_n()
    {
        return n;
    };

    // operators
    Vec4d &operator[](int index)
    {
        return (cpts[index]);
    };
    const Vec4d &operator[](int index) const
    {
        return (cpts[index]);
    };
    Cpol &operator=(const Cpol &_c);
};

//===========================================================================
// Control Net in object space
//===========================================================================

class Cnet
{

private:
    int n; // number of control points in u direction
    int m; // number of control points in v direction
    Vec4d **cpts; // homogeneous coordinates (x, y, z, w) of control points

public:
    // constructors and destructors
    Cnet()
    {
        n = m = 0;
        cpts = NULL;
    };
    Cnet(int numU, int numV);
    Cnet(int numU, int numV, Vec4d **cpoints);
    Cnet(const Cnet &_c);
    ~Cnet();

    // set/get
    void set(int row, int col, Vec3d Cpt);
    void set(int row, int col, Vec4d Cpt);
    int get_n()
    {
        return n;
    };
    int get_m()
    {
        return m;
    };
    Vec4d get(int row, int col)
    {
        return cpts[row][col];
    };

    // operators
    Vec4d *&operator[](int index)
    {
        return (cpts[index]);
    };
    //const Vec4d*& operator [](int index) const { return( cpts[index] );};
    Cnet &operator=(const Cnet &_c);
};

//===========================================================================
// Trim NURBS Curve
//===========================================================================

class TrimCurve
{

private:
    int n_knts; // number of knots
    Knot *knt; // knot vector
    int n_cpts; // number of control points
    Pcpol *pol; // control polygon

public:
    // constructors and destructors
    TrimCurve()
    {
        n_knts = 0;
        knt = NULL;
        n_cpts = 0;
        pol = NULL;
    };
    TrimCurve(int n, int m);
    TrimCurve(const TrimCurve &trim);
    ~TrimCurve();

    // set/get
    void set_knt(REAL *U);
    void set_pol(Vec2d *CPs);
    void set_pol(Vec3d *CPs);
    int get_n_knts()
    {
        return n_knts;
    };
    int get_n_cpts()
    {
        return n_cpts;
    };
    REAL get_knot(int i)
    {
        return (*knt)[i];
    };
    Vec3d get_controlPoint(int i)
    {
        return (*pol)[i];
    };

    // implement functions
    void BezDegreeReduce(Vec3d *bPts, Vec3d *rbPts, REAL &MaxError);
    FLAG DegreeReduceTrimCurve();
    // FLAG MultipleDegreeReduce(const int t);
    int FindSpan(const REAL u);
    void FindSpanMult(const REAL u, int &k, int &s);
    void RemoveCurveKnot(const int r, const int s, int num, int &t);
    void MaximumKnotRemoval();

    // operators
    TrimCurve &operator=(const TrimCurve &trim);
    friend ostream &operator<<(ostream &, const TrimCurve &);
    friend istream &operator>>(istream &, TrimCurve &);
};

//===========================================================================
// NURBS Curve
//===========================================================================

class NurbsCurve
{

private:
    int n_knts; // number of knots
    Knot *knt; // knot vector
    int n_cpts; // number of control points
    Cpol *pol; // control polygon

public:
    // constructors and destructors
    NurbsCurve()
    {
        n_knts = 0;
        knt = NULL;
        n_cpts = 0;
        pol = NULL;
    };
    NurbsCurve(int n, int m);
    NurbsCurve(const NurbsCurve &nurbs);
    ~NurbsCurve();

    // set/get
    void set_knt(REAL *U);
    void set_knt(Knot *U);
    void set_pol(Vec3d *CPs);
    void set_pol(Vec4d *CPs);
    void set_pol(Cpol &CP);
    int get_n_knts()
    {
        return n_knts;
    };
    int get_n_cpts()
    {
        return n_cpts;
    };
    REAL get_knot(int i)
    {
        return (*knt)[i];
    };
    Vec4d get_controlPoint(int i)
    {
        return (*pol)[i];
    };

    // implement functions
    void BezDegreeReduce(Vec4d *bPts, Vec4d *rbPts, REAL &MaxError);
    FLAG DegreeReduceCurve();
    int FindSpan(const REAL u);
    void FindSpanMult(const REAL u, int &k, int &s);
    void RemoveCurveKnot(const int r, const int s, int num, int &t);
    void MaximumKnotRemoval();
    void output();

    // operators
    NurbsCurve &operator=(const NurbsCurve &nurbs);
    friend ostream &operator<<(ostream &, const NurbsCurve &);
    friend istream &operator>>(istream &, NurbsCurve &);
};

//===========================================================================
// NURBS Surface
//===========================================================================

class NurbsSurface
{

private:
    int n_Uknts; // number of knots in u direction
    Knot *Uknt; // knot vector in u direction
    int n_Vknts; // number of knots in v direction
    Knot *Vknt; // knot vector in v direction
    int Udim; // dimension of control net in u direction
    int Vdim; // dimension of control net in v direction
    Cnet *net; // control net

public:
    // constructors and destructors
    NurbsSurface()
    {
        n_Uknts = n_Vknts = 0;
        Vknt = NULL;
        Uknt = NULL;
        Udim = Vdim = 0;
        net = NULL;
    };
    NurbsSurface(int r, int s, int n, int m);
    NurbsSurface(const NurbsSurface &nurbs);
    ~NurbsSurface();

    // set/get
    void set_Uknt(REAL *U);
    void set_Uknt(Knot *U);
    void set_Vknt(REAL *V);
    void set_Vknt(Knot *V);
    void set_net(Vec3d **CPs);
    void set_net(Vec4d **CPs);
    void set_net(Cnet &CN);
    int get_n_Uknts()
    {
        return n_Uknts;
    };
    int get_n_Vknts()
    {
        return n_Vknts;
    };
    int get_Udim()
    {
        return Udim;
    };
    int get_Vdim()
    {
        return Vdim;
    };
    REAL get_Uknot(int i)
    {
        return (*Uknt)[i];
    };
    REAL get_Vknot(int i)
    {
        return (*Vknt)[i];
    };
    Vec4d get_controlPoint(int i, int j)
    {
        return net->get(i, j);
    };

    // implement functions
    int FindSpan(const int DIR, const REAL u);
    void FindSpanMult(const int DIR, const REAL u, int &k, int &s);
    void RemoveKnot(const int DIR, const int l, const int mult, int num, int &t);
    void MaximumKnotRemoval();

    // operators
    NurbsSurface &operator=(const NurbsSurface &nurbs);
    friend ostream &operator<<(ostream &, const NurbsSurface &);
    friend istream &operator>>(istream &, NurbsSurface &);
};
#endif // NURBS_H
