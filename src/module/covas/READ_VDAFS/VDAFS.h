/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VDAFS_H
#define VDAFS_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  Class-Declaration of VDAFS Data Objects                  **
 **               (+ Declaration of Non-member Help Functions)             **
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
 ** Date:  09.05.97  V1.0                                                  **
\**************************************************************************/

// include files
#include <iostream.h>
#include <assert.h>
#include <string.h>
#include "Vec.h"
#include "nurbs.h"

// LEDA
#include <LEDA/basic.h>
#include <LEDA/string.h>
#include <LEDA/list.h>

// symbols
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // M_PI
#define EPS 1.0e-15
#define U_DIR 0
#define V_DIR 1
#define SURF 2
#define FACE 3

enum MODE
{
    ADD,
    DELETE
};

// typedefs
#ifndef FLAG
typedef short FLAG;
#endif

#ifndef REAL
typedef double REAL;
#endif

// macro for degree to radians
#define RAD(alpha) (alpha * M_PI / 180)

extern list<NurbsCurve> nurbscurveList;
extern list<NurbsSurface> nurbssurfaceList;
extern list<TrimCurve> curveDefList;
extern list<NurbsSurface> surfaceDefList;
extern list<int> connectionList;
extern list<int> trimLoopList;

REAL BiCo(int, int);
int Max(int, int);
int Min(int, int);
void Exchange(REAL *v1, REAL *v2);
Matrix BezierToPowerMatrix(int);
Matrix PowerToBezierMatrix(int);
FLAG Intersect3DLines(const Vec3d &p0, const Vec3d &t0,
                      const Vec3d &p2, const Vec3d &t2, Vec3d &p1);

//===========================================================================
// Circle
//===========================================================================

class Circle
{

private:
    char *name; // name
    Vec3d center; // center of circle
    REAL radius; // radius of circle
    Vec3d V; // unit length  vector lying in the plane of
    // definition of the circle
    Vec3d W; // unit length vector in the plane of definition
    // of the circle, orthogonal to V
    REAL alpha; // start angle
    REAL beta; // end angle (measured with respect to V)

public:
    Circle()
    {
        name = NULL;
    };
    Circle(const Circle &);
    Circle(
        char *name, Vec3d center, REAL radius,
        Vec3d V, Vec3d W, REAL alpha, REAL beta);
    ~Circle();

    // operators
    Circle &operator=(const Circle &_cir);
    friend ostream &operator<<(ostream &, const Circle &);
    friend istream &operator>>(istream &, Circle &);

    // other functions
    void MakeNurbsCircle();
};

//===========================================================================
// Curve
//===========================================================================

class Curve_Segment;
class Curve
{

private:
    char *name; // name
    int n_segments; // number of segments
    int *order; // order of the polynoms (order= degree + 1)
    Curve_Segment *Tbl; // Table of curve segments

public:
    Curve()
    {
        name = NULL;
        n_segments = 0;
        order = NULL;
        Tbl = NULL;
    };
    Curve(const Curve &);
    Curve(char *name, int n_segments, int *order, Curve_Segment *Tbl);
    ~Curve();

    // sets and gets
    Vec3d get_coeff(int index, int power);

    // operators
    Curve &operator=(const Curve &_c);
    friend ostream &operator<<(ostream &, const Curve &);
    friend istream &operator>>(istream &, Curve &);

    // other functions
    void MakeNurbsCurve();
    Vec3d *DegreeElevateBezierSegment(
        int p,
        const Vec3d *CP,
        int t);
    Vec3d *CoeffTrafo(int no, const REAL alpha, const REAL beta);
    void PartOfCurve(char *nme, REAL s1, REAL s2);
};

class Curve_Segment
{
    friend class Curve;

private:
    int index; // index of curve segment
    int seg_ord; // order of the curve segment polynom
    REAL par_val[2]; // values of the global parameter at the
    // beginning and end of the curve segment

    Vec3d *a; // coefficients of the polynom

public:
    Curve_Segment()
    {
        index = 0;
        seg_ord = 0;
        par_val[0] = par_val[1] = 0.0;
        a = NULL;
    };
    Curve_Segment(const Curve_Segment &);
    Curve_Segment(int ind, int seg_ord, REAL p_i, REAL p_e, Vec3d *a);
    ~Curve_Segment();

    //sets and gets
    void setIndex(int index);
    void setOrder(int seg_ord);
    void setParam(REAL p_i, REAL p_e);
    void setCoeff(int seg_ord, Vec3d *a);

    // operators
    Curve_Segment &operator=(const Curve_Segment &_seg);

    // other functions
    Matrix BezierToPowerMatrix();
    Matrix PowerToBezierMatrix();
};

//===========================================================================
// Surface
//===========================================================================

class Surf_Segment;
class Surf
{

private:
    char *name; // name
    int nps; // number of segments in u direction
    int npt; // number of segments in v direction
    int *order_u; // order of polynom of the segments
    // in u-direction
    int *order_v; // order of polynom of the segments
    // in v-direction
    // (order = degree +1)
    Surf_Segment *Tbl; // Table of surface segments

public:
    Surf()
    {
        name = NULL;
        nps = 0;
        npt = 0;
        order_u = NULL;
        order_v = NULL;
        Tbl = NULL;
    };
    Surf(const Surf &);
    Surf(char *name, int nps, int npt, int *order_u, int *order_v,
         Surf_Segment *Tbl);
    ~Surf();

    // sets and gets
    Vec3d get_coeff(int num, int pow_u, int pow_v);

    // operators
    Surf &operator=(const Surf &_s);
    friend ostream &operator<<(ostream &, const Surf &);
    friend istream &operator>>(istream &, Surf &);

    // other functions
    void MakeNurbsSurface(int tag);
    Vec3d *DegreeElevateBezierRow(int p, Vec3d *CRow, int t);
    void DegreeElevateBezierPatch(int p, int q, int DIR, Vec3d **CP, int t, Vec3d **CQ);
};

class Surf_Segment
{
    friend class Surf;

private:
    int number; // segment number
    int seg_ord_u; // order of polynom of the surface segment
    // in u direction
    int seg_ord_v; // order of polynom of the surface segment
    // in v direction
    REAL par_s[2]; // values of the global parameter s at the
    // beginning and end of the surface segment
    REAL par_t[2]; // values of the global parameter t at the
    // beginning and end of the surface segment
    //      ==> range of definition

    Vec3d **a; // coefficients of segment

public:
    Surf_Segment()
    {
        number = 0;
        seg_ord_u = 0;
        seg_ord_v = 0;
        par_s[0] = par_s[1] = 0.0;
        par_t[0] = par_t[1] = 0.0;
        a = NULL;
    };
    Surf_Segment(const Surf_Segment &);
    Surf_Segment(int number, int seg_ord_u, int seg_ord_v,
                 REAL ps_i, REAL ps_e, REAL pt_i, REAL pt_e, Vec3d **a);
    ~Surf_Segment();

    // operators
    Surf_Segment &operator=(const Surf_Segment &);

    // other functions
};

//===========================================================================
// Curve on surface
//===========================================================================

class Cons_Segment;
class Cons
{

private:
    char *name; // name
    char *surfnme; // name of surface (on which cons is lying)
    char *curvenme; // name of curve (secondary representation)
    REAL s1; // initial global curve parameter
    REAL s2; // final global curve parameter
    // (in [s1, s2] a curve with name
    //  "curvenme" approximates cons in object space)
    int n_segments; // number of segments
    int *order; // order of the polynoms (order= degree + 1)
    Cons_Segment *Tbl; // Table of cons segments

public:
    Cons()
    {
        name = NULL;
        surfnme = NULL;
        curvenme = NULL;
        s1 = s2 = 0.0;
        n_segments = 0;
        order = NULL;
        Tbl = NULL;
    };
    Cons(const Cons &);
    Cons(char *name, char *surfnme, char *curvenme, REAL s1, REAL s2,
         int n_segments, int *order, Cons_Segment *Tbl);
    ~Cons();

    // sets and gets
    Vec2d get_coeff(int index, int power);
    char *get_curvenme();
    REAL get_s1();
    REAL get_s2();

    // operators
    Cons &operator=(const Cons &_c);
    friend ostream &operator<<(ostream &, const Cons &);
    friend istream &operator>>(istream &, Cons &);

    // other functions
    void MakeNurbsCons();
    Vec2d *DegreeElevateBezierSegment(
        int p,
        const Vec2d *CP,
        int t);
    Vec2d *CoeffTrafo(int no, REAL alpha, REAL beta);
    void PartOfCons(REAL w1, REAL w2);
};

class Cons_Segment
{
    friend class Cons;

private:
    int index; // index of cons segment
    int seg_ord; // order of the cons segment polynom
    REAL par_val[2]; // values of the global parameter at the
    // beginning and end of the cons segment

    Vec2d *a; // coefficients of the polynom

public:
    Cons_Segment()
    {
        index = 0;
        seg_ord = 0;
        par_val[0] = par_val[1] = 0.0;
        a = NULL;
    };
    Cons_Segment(const Cons_Segment &);
    Cons_Segment(int ind, int seg_ord, REAL p_i, REAL p_e, Vec2d *a);
    ~Cons_Segment();

    //sets and gets
    void setIndex(int index);
    void setOrder(int seg_ord);
    void setParam(REAL p_i, REAL p_e);
    void setCoeff(int seg_ord, Vec2d *a);

    // operators
    Cons_Segment &operator=(const Cons_Segment &_seg);

    // other functions
    Matrix BezierToPowerMatrix();
    Matrix PowerToBezierMatrix();
};

//===========================================================================
// Face
//===========================================================================

class Cons_Ensemble;
class Face
{

private:
    char *name; // name
    char *surfnme; // name of surface defining the face
    int m; // number of closed cons ensemble
    Cons_Ensemble *Tbl; // table of cons ensembles

public:
    Face()
    {
        name = NULL;
        surfnme = NULL;
        m = 0;
        Tbl = NULL;
    };
    Face(const Face &);
    Face(char *name, char *surfname, int m, Cons_Ensemble *Tbl);
    ~Face();

    // sets and gets
    void set_connectionList(int mode);
    void set_trimLoopList(int mode);
    int get_n_trimLoops()
    {
        return m;
    };
    char *get_surfnme()
    {
        return surfnme;
    };
    Cons_Ensemble *get_ConsEnsembles()
    {
        return Tbl;
    };

    // operators
    Face &operator=(const Face &_f);
    friend ostream &operator<<(ostream &, const Face &);
    friend istream &operator>>(istream &, Face &);
};

class Cons_Ensemble
{
    friend class Face;
    friend class Top;

private:
    int n_cons; // number of cons elements
    char **name; // table of names of cons elements
    REAL *w1; // table of initial global parameters (defining
    // the beginning of a part of the cons element)
    REAL *w2; // table of final global parameters (defining the
    // end of a part of the cons element)
public:
    Cons_Ensemble()
    {
        n_cons = 0;
        name = NULL;
        w1 = NULL;
        w2 = NULL;
    };
    Cons_Ensemble(const Cons_Ensemble &);
    Cons_Ensemble(int n_cons, char **name, REAL *w1, REAL *w2);
    ~Cons_Ensemble();

    // sets and gets
    int get_n_cons()
    {
        return n_cons;
    };
    REAL *get_w1()
    {
        return w1;
    };
    REAL *get_w2()
    {
        return w2;
    };
    char **get_ConsNames()
    {
        return name;
    };

    // operators
    Cons_Ensemble &operator=(const Cons_Ensemble &_ens);
};

//===========================================================================
// Top
//===========================================================================

class Top
{

private:
    char *name; // name
    int m; // number of pairs of adjacent surfaces or faces
    char **fsname; // names of surfaces or faces
    Cons_Ensemble *Tbl; // table of cons ensemble
    int *icont; // continuity parameters

public:
    Top();
    Top(const Top &);
    Top(char *name, int m, char **fsname, Cons_Ensemble *Tbl, int *icont);
    Top(char *name, int m);
    ~Top();

    // sets and gets
    int get_m_pairs();
    char **get_fsNames();

    // operators
    Top &operator=(const Top &_f);
    friend ostream &operator<<(ostream &, const Top &);
    friend istream &operator>>(istream &, Top &);
};

//============================================================================
// Group
//============================================================================

class Group
{

private:
    char *name; // name
    int n; // number of elements
    char **element; // table of element names

public:
    Group()
    {
        name = NULL;
        n = 0;
        element = NULL;
    };
    Group(const Group &);
    Group(char *name, int n, char **element);
    ~Group();

    // sets and gets
    int get_n_elements()
    {
        return n;
    };
    char *getElement_name(int i)
    {
        return element[i];
    };

    // operators
    Group &operator=(const Group &_grp);
    friend ostream &operator<<(ostream &, const Group &);
    friend istream &operator>>(istream &, Group &);
};

//============================================================================
// Set
//============================================================================

class Set
{

private:
    char *name; // name
    list<string> set_elements; // list of set elements

public:
    Set()
    {
        name = NULL;
    };
    Set(const Set &);
    Set(char *name, list<string> element_names);
    ~Set();

    // sets and gets
    int get_n_elements()
    {
        return set_elements.length();
    };
    string getElement_name(int i)
    {
        return set_elements.contents(set_elements.item(i));
    };

    // operators
    Set &operator=(const Set &_set);
    friend ostream &operator<<(ostream &, const Set &);
    friend istream &operator>>(istream &, Set &);
};
#endif // VDAFS_H
