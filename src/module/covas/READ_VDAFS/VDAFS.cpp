/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Constructors and Member-Functions for VDAFS Data Objects  **
 **              ( + Non-member Help Functions)                            **
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

#include "VDAFS.h"

// maximum of two values
inline int Max(int v1, int v2)
{
    return (v1 >= v2 ? v1 : v2);
}

// minimum of two values
inline int Min(int v1, int v2)
{
    return (v1 <= v2 ? v1 : v2);
}

inline void Exchange(REAL *v1, REAL *v2)
{
    REAL tmp = *v2;
    *v2 = *v1;
    *v1 = tmp;
}

//==========================================================================
// Circle
//==========================================================================

// Constructors and Member-Functions of the class Circle

////////////////////////////////// Circle //////////////////////////////////
Circle::Circle(const Circle &_cir)
    : center(_cir.center)
    , radius(_cir.radius)
    , V(_cir.V)
    , W(_cir.W)
    , alpha(_cir.alpha)
    , beta(_cir.beta)
{
    name = new char[strlen(_cir.name) + 1];
    assert(name != 0);
    strcpy(name, _cir.name);
}

Circle::Circle(char *nme, Vec3d origin, REAL r,
               Vec3d X, Vec3d Y, REAL phi_s, REAL phi_e)
{
    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    center = origin;
    radius = r;
    V = X;
    W = Y;
    alpha = phi_s;
    beta = phi_e;
}

Circle::~Circle()
{
    delete[] name;
}

Circle &Circle::operator=(const Circle &_cir)
{
    if (this == &_cir)
        return *this;

    delete[] name;
    name = new char[strlen(_cir.name) + 1];
    assert(name != 0);
    strcpy(name, _cir.name);

    center = _cir.center;
    radius = _cir.radius;
    V = _cir.V;
    W = _cir.W;
    alpha = _cir.alpha;
    beta = _cir.beta;

    return *this;
}

void Circle::MakeNurbsCircle()
{
    // Create arbitrary NURBS circular arc
    // Output:  (to global list)
    //          knot vector: U
    //          weighted control vectors: CPw

    int n;
    int i, j;
    int index;
    int n_arcs; // number of arcs
    REAL phi;
    REAL dphi;
    REAL w1;
    REAL angle;
    REAL *U;

    Vec3d P0, P1, P2;
    Vec3d T0, T2;
    Vec3d *Pw;
    Vec4d *CPw;

    NurbsCurve *nurbs;

    phi = beta - alpha;

    // get number of arcs
    if (phi <= 90.0)
        n_arcs = 1;
    else if (phi <= 180.0)
        n_arcs = 2;
    else if (phi <= 270.0)
        n_arcs = 3;
    else
        n_arcs = 4;

    dphi = phi / n_arcs;
    n = 2 * n_arcs; // n+1 control points
    Pw = new Vec3d[n + 1];
    assert(Pw != 0);
    // Allocate control hull with homogeneous coordinates
    CPw = new Vec4d[n + 1];
    assert(CPw != 0);

    w1 = cos(RAD(dphi / 2.0)); // dphi/2 is base angle

    // Initialize start values
    P0 = center + radius * cos(RAD(alpha)) * V + radius * sin(RAD(alpha)) * W;
    T0 = -sin(RAD(alpha)) * V + cos(RAD(alpha)) * W;

    Pw[0] = P0;
    CPw[0].set(P0, 1.0);

    index = 0;
    angle = alpha;

    // create n_arcs segments
    for (i = 1; i <= n_arcs; i++)
    {
        angle += dphi;
        P2 = center + radius * cos(RAD(angle)) * V + radius * sin(RAD(angle)) * W;
        Pw[index + 2] = P2;
        CPw[index + 2].set(P2, 1.0);

        T2 = -sin(RAD(angle)) * V + cos(RAD(angle)) * W;

        Intersect3DLines(P0, T0, P2, T2, P1);
        Pw[index + 1] = w1 * P1;
        CPw[index + 1].set(Pw[index + 1], w1);
        index += 2;

        if (i < n_arcs)
        {
            P0 = P2;
            T0 = T2;
        }
    }

    // load the knot vector
    j = 2 * n_arcs + 1;
    U = new REAL[j + 3];
    assert(U != 0);

    for (i = 0; i < 3; i++)
    {
        U[i] = 0.0;
        U[i + j] = 1.0;
    }

    switch (n_arcs)
    {
    case 1:
        break;

    case 2:
        U[3] = U[4] = 0.5;
        break;

    case 3:
        U[3] = U[4] = 1.0 / 3.0;
        U[5] = U[6] = 2.0 / 3.0;
        break;

    case 4:
        U[3] = U[4] = 0.25;
        U[5] = U[6] = 0.5;
        U[7] = U[8] = 0.75;
        break;
    }

    // set NURBS object and add to list
    nurbs = new NurbsCurve(j + 3, n + 1);
    nurbs->set_knt(U);
    nurbs->set_pol(CPw);
    nurbscurveList.append(*nurbs);

    delete[] Pw;
    delete[] CPw;
    delete[] U;
    delete nurbs;
}

//===========================================================================
// Curve
//===========================================================================

// Constructors and Member-Functions of the classes Curve and Curve_Segment

////////////////////////////////// Curve //////////////////////////////////
Curve::Curve(const Curve &_c)
    : n_segments(_c.n_segments)
{
    name = new char[strlen(_c.name) + 1];
    assert(name != 0);
    strcpy(name, _c.name);

    order = new int[n_segments];
    assert(order != 0);
    memcpy(order, _c.order, (size_t)(sizeof(int) * n_segments));

    Tbl = new Curve_Segment[n_segments];
    assert(Tbl != 0);
    int i;
    for (i = 0; i < n_segments; i++)
        Tbl[i] = _c.Tbl[i];
}

Curve::Curve(char *nme, int n, int *ord, Curve_Segment *segm)
{

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);
    n_segments = n;

    order = new int[n_segments];
    assert(order != 0);
    memcpy(order, ord, (size_t)(sizeof(int) * n_segments));

    Tbl = new Curve_Segment[n_segments];
    assert(Tbl != 0);
    int i;
    for (i = 0; i < n_segments; i++)
        Tbl[i] = segm[i];
}

Curve::~Curve()
{
    delete[] name;
    delete[] order;
    delete[] Tbl;
}

Vec3d Curve::get_coeff(int ind, int power)
{
    // range of power: 0 <= power <= degree
    return (Tbl[ind - 1].a[power]);
}

Curve &Curve::operator=(const Curve &_c)
{
    if (this == &_c)
        return *this;

    delete[] name;
    name = new char[strlen(_c.name) + 1];
    assert(name != 0);
    strcpy(name, _c.name);

    n_segments = _c.n_segments;
    delete[] order;
    delete[] Tbl;

    order = new int[n_segments];
    assert(order != 0);
    memcpy(order, _c.order, (size_t)(sizeof(int) * n_segments));

    Tbl = new Curve_Segment[n_segments];
    assert(Tbl != 0);
    int i;
    for (i = 0; i < n_segments; i++)
        Tbl[i] = _c.Tbl[i];

    return *this;
}

Vec3d *Curve::DegreeElevateBezierSegment(
    int p,
    const Vec3d *CP,
    int t)
{

    // Degree elevate a Bezier curve segment t times
    // Input:
    //         p     : degree
    //         CP    : control points
    //         t     : number of degree elevations
    // Output:
    //         CQ    : new control points

    int i, j;
    int pe; // elevated degree
    int mpi;

    REAL pe2;
    REAL inv;
    REAL **bdec; // Bezier degree elevation coefficients

    Vec3d *CQ = new Vec3d[p + t + 1];

    pe = p + t;
    pe2 = pe / 2;

    // allocate matrix of Bezier degree elevation coefficients
    bdec = new REAL *[pe + 1];
    for (i = 0; i <= pe; i++)
    {
        bdec[i] = new REAL[p + 1];
        for (j = 0; j < p + 1; j++)
            bdec[i][j] = 0.0;
    }

    bdec[0][0] = bdec[pe][p] = 1.0;
    // Compute Bezier degree elevation coefficients
    for (i = 1; i <= pe2; i++)
    {
        inv = 1.0 / BiCo(pe, i);
        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
        {
            bdec[i][j] = inv * BiCo(p, j) * BiCo(t, i - j);
        }
    }
    for (i = pe2 + 1; i <= pe - 1; i++)
    {
        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
        {
            bdec[i][j] = bdec[pe - i][p - j];
        }
    }

    // Compute the control points of degree elevated Bezier segment
    for (i = 0; i <= pe; i++)
    {
        CQ[i][0] = 0.0;
        CQ[i][1] = 0.0;
        CQ[i][2] = 0.0;

        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
            CQ[i] = CQ[i] + bdec[i][j] * CP[j];
    }
    /*
       // verify
       cout << "degree elevation:" << endl;
       for (i=0; i<=pe; i++)
       {
             CQ[i].output();
       }
       cout << endl;
   */
    // free
    for (i = 0; i <= pe; i++)
        delete[] bdec[i];
    delete[] bdec;

    return CQ;
}

void Curve::MakeNurbsCurve()
{
    // Create a NURBS curve from all power basis segments
    // C(0) continuity is assumed!

    int i, j, k;
    int num = 0;
    int order_max; // maximum order of all segments
    int diff;
    int num_U; // number of knots of composed Bezier curve
    int index_U;
    int num_CP;
    int index_CP;
    int num_CQ; // number of control points of composed Bezier curve
    Vec3d *CR;
    Vec3d *CS;

    NurbsCurve *nurbs;

    // Determine number of 3-dimensional control points for all segments
    num_CP = 0;
    for (i = 0; i < n_segments; i++)
        num_CP = num_CP + order[i];
    num_CP = num_CP - (n_segments - 1);

    Vec3d *CP = new Vec3d[num_CP]; // control points of non degree
    // elevated segments

    // Determine the order of the NURBS curve
    order_max = 0;
    for (i = 0; i < n_segments; i++)
        order_max = Max(order[i], order_max);
    if (order_max == 0)
        cerr << "ERROR: order of NURBS curve wrong" << endl;

    // Determine number of control points for the composed Bezier curve
    num_CQ = n_segments * order_max - (n_segments - 1);

    Vec3d *CQ = new Vec3d[num_CQ]; // control points of composed
    // Bezier curve

    // Determine number of knots for the composed Bezier curve
    num_U = 2 * order_max + (n_segments - 1) * (order_max - 1);
    REAL *U = new REAL[num_U];

    // Compute the control points of segments
    for (i = 0; i < n_segments; i++)
    {
        Matrix p_m = Tbl[i].PowerToBezierMatrix();

        for (j = 0; j < order[i]; j++)
        {
            CP[j + num][0] = CP[j + num][1] = CP[j + num][2] = 0;
            for (k = 0; k < order[i]; k++)
            {
                CP[j + num] = CP[j + num] + (p_m[j][k] * Tbl[i].a[k]);
            }

            // cout << j+num << ": ";
            // CP[j+num].output();
        }
        num += order[i] - 1;
    }

    num = 0;
    // Degree elevation of segments and composition
    for (i = 0; i < n_segments; i++)
    {
        if (order[i] < order_max)
        {
            diff = order_max - order[i];
            CR = new Vec3d[order[i]];
            // select control points
            for (j = 0; j < order[i]; j++)
            {
                CR[j] = CP[j + num];
                // cout << j << ": ";
                // CR[j].output();
            }

            CS = DegreeElevateBezierSegment(order[i] - 1, CR, diff);

            // assign control points of degree elevated segment
            for (j = 0; j < order_max; j++)
            {
                CQ[i * (order_max - 1) + j] = CS[j];
            }

            delete[] CR;
            delete[] CS;
        }
        num += order[i] - 1;

        if (order[i] == order_max)
        {
            // assign original control points
            index_CP = 0;
            for (k = 0; k < i; k++)
                index_CP = index_CP + order[k] - 1;
            for (j = 0; j < order_max; j++)
                CQ[i * (order_max - 1) + j] = CP[index_CP + j];
        }
    }
    // Compute knot vector
    index_U = 0;
    for (j = 0; j < order_max; j++)
    {
        U[j] = Tbl[0].par_val[0];
    }
    index_U += order_max;
    if (n_segments > 1)
    {
        for (i = 1; i < n_segments; i++)
        {
            for (j = 0; j < order_max - 1; j++)
            {
                U[index_U + j] = Tbl[i].par_val[0];
            }
            index_U += order_max - 1;
        }
    }
    for (j = 0; j < order_max; j++)
    {
        U[index_U + j] = Tbl[n_segments - 1].par_val[1];
    }

    /*
      // verify
      cout << "control points:" << endl;
      for (i=0; i< num_CQ; i++)
      {
           cout << i << ": ";
           CQ[i].output();
      }

      cout << "knot vector:" << endl;
      cout << "( ";
   for (i=0; i< num_U; i++)
   cout << U[i] << "\t";
   cout << " )" << endl;
   */

    // set NURBS object and add to list
    nurbs = new NurbsCurve(num_U, num_CQ);
    nurbs->set_knt(U);
    nurbs->set_pol(CQ);
    // maximum knot removal
    nurbs->MaximumKnotRemoval();
    nurbscurveList.append(*nurbs);

    delete[] CQ;
    delete[] CP;
    delete[] U;
    delete nurbs;
}

Vec3d *Curve::CoeffTrafo(int no, const REAL alpha, const REAL beta)
{
    // Compute transformed coefficients of polynom for
    // a segment of partial curve
    // INPUT:
    //           no: number of original segment (no = index -1)
    //        alpha: transformation coefficient
    //         beta: transformation coefficient
    // OUTPUT:
    //           at: transformed coefficients

    int i, j;
    int ord; // order of segment
    ord = Tbl[no].seg_ord;

    // Allocate and initialize the transformed coefficients
    Vec3d *at = new Vec3d[ord];
    for (i = 0; i < ord; i++)
    {
        at[i][0] = at[i][1] = at[i][2] = 0.0;
    }

    // Transformation
    for (i = 0; i < ord; i++)
    {
        if (beta != 0)
            for (j = i; j < ord; j++)
                at[i] = at[i] + BiCo(j, i) * pow(beta, j - i) * pow(alpha, i) * Tbl[no].a[j];
        else
            at[i] = at[i] + pow(alpha, i) * Tbl[no].a[i];
    }

    return at;
}

void Curve::PartOfCurve(char *nme, REAL s1, REAL s2)
{
    // Create a partial curve of the original curve
    // defined by the global parameters s1 and s2
    // Input:
    //        nme: name of partial curve
    //         s1: start value of defintion range of partial curve
    //         s2: end of definition range of partial curve

    int i;
    int it;
    int it_i, it_e;
    int num;
    int index_i; // index of first curve segment of partial curve
    int index_e; // index of last curve segment of partial curve
    int n_part; // number of segments of partial curve

    int *ord; // order of polynoms for the partial curve
    REAL denom;
    REAL alpha, beta;
    Vec3d *coeff;
    Curve_Segment *seg; // table of curve segments of partial curve
    Curve *c; // partial curve

    // We don't care about the orientation of the partial curve.
    // This is irrelevant to rendering.
    // Only ascending intervals are allowed.
    if (s1 > s2)
    {
        Exchange(&s1, &s2);
    }
    // Now: s1 < s2 !!!

    // Determine the curve segment indices where the interval borders
    // are lying
    for (i = 0; i < n_segments; i++)
    {
        if (s1 >= Tbl[i].par_val[0] && s1 < Tbl[i].par_val[1])
            index_i = Tbl[i].index;

        if (s2 > Tbl[i].par_val[0] && s2 <= Tbl[i].par_val[1])
            index_e = Tbl[i].index;
    }

    // fix error in file bsp.vdafs (CCR19 und C019)
    if (s2 > Tbl[n_segments - 1].par_val[1])
    {
        s2 = Tbl[n_segments - 1].par_val[1];
        index_e = Tbl[n_segments - 1].index;
    }

    // verify
    // cout << "Partial curve" << endl;
    // cout << "Index-Tupel: (" << index_i << ", " << index_e << ")" << endl;

    if (index_i == index_e) // only one segment
    {
        n_part = 1;
        seg = new Curve_Segment;
        it = index_i - 1;
        denom = Tbl[it].par_val[1] - Tbl[it].par_val[0];
        alpha = (s2 - s1) / denom;
        beta = (s1 - Tbl[it].par_val[0]) / denom;

        if (s1 == Tbl[it].par_val[0] && s2 == Tbl[it].par_val[1])
        {
            // select original segment (with new curve segment index)
            seg->setIndex(n_part);
            seg->setParam(s1, s2);
            seg->setCoeff(Tbl[it].seg_ord, Tbl[it].a);
            c = new Curve(
                nme,
                n_part,
                &(order[it]),
                seg);
            c->MakeNurbsCurve();
        }
        else
        {
            // Computing of new coefficients
            coeff = CoeffTrafo(it, alpha, beta);
            seg->setIndex(n_part);
            seg->setParam(s1, s2);
            seg->setCoeff(Tbl[it].seg_ord, coeff);
            c = new Curve(
                nme,
                n_part,
                &(order[it]),
                seg);
            c->MakeNurbsCurve();
            delete[] coeff;
        }
        delete seg;
        delete c;
    }

    if (index_i < index_e) // several segments
    {
        n_part = index_e - index_i + 1;
        ord = new int[n_part];
        seg = new Curve_Segment[n_part];
        it_i = index_i - 1;
        it_e = index_e - 1;
        num = 0;

        if (s1 == Tbl[it_i].par_val[0])
        {
            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it_i].par_val[0], Tbl[it_i].par_val[1]);
            seg[num].setCoeff(Tbl[it_i].seg_ord, Tbl[it_i].a);
            ord[num] = Tbl[it_i].seg_ord;
            num++;
        }
        else
        {
            // Computing new coefficients
            denom = Tbl[it_i].par_val[1] - Tbl[it_i].par_val[0];
            alpha = (Tbl[it_i].par_val[1] - s1) / denom;
            beta = (s1 - Tbl[it_i].par_val[0]) / denom;
            coeff = CoeffTrafo(it_i, alpha, beta);

            seg[num].setIndex(num + 1);
            seg[num].setParam(s1, Tbl[it_i].par_val[1]);
            seg[num].setCoeff(Tbl[it_i].seg_ord, coeff);
            ord[num] = Tbl[it_i].seg_ord;
            num++;
        }

        for (it = it_i + 1; it < it_e; it++)
        {
            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it].par_val[0], Tbl[it].par_val[1]);
            seg[num].setCoeff(Tbl[it].seg_ord, Tbl[it].a);
            ord[num] = Tbl[it].seg_ord;
            num++;
        }

        if (s2 == Tbl[it_e].par_val[1])
        {
            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it_e].par_val[0], Tbl[it_e].par_val[1]);
            seg[num].setCoeff(Tbl[it_e].seg_ord, Tbl[it_e].a);
            ord[num] = Tbl[it_e].seg_ord;
        }
        else
        {
            // Computing new coefficients
            denom = Tbl[it_e].par_val[1] - Tbl[it_e].par_val[0];
            alpha = (s2 - Tbl[it_e].par_val[0]) / denom;
            beta = 0;
            coeff = CoeffTrafo(it_e, alpha, beta);

            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it_e].par_val[0], s2);
            seg[num].setCoeff(Tbl[it_e].seg_ord, coeff);
            ord[num] = Tbl[it_e].seg_ord;
        }

        c = new Curve(
            nme,
            n_part,
            ord,
            seg);
        c->MakeNurbsCurve();
        delete[] seg;
        delete[] ord;
        delete c;
    }

    if (index_i > index_e)
        cerr << "ERROR: Index sequence of partial curve incorrect!"
             << endl;
}

////////////////////////////// Curve_Segment //////////////////////////////
Curve_Segment::Curve_Segment(const Curve_Segment &_seg)
    : index(_seg.index)
    , seg_ord(_seg.seg_ord)
{
    memcpy(par_val, _seg.par_val, (size_t)(sizeof(REAL) * 2));

    a = new Vec3d[seg_ord];
    assert(a != 0);
    memcpy(a, _seg.a, (size_t)(sizeof(Vec3d) * seg_ord));
}

Curve_Segment::Curve_Segment(int ind, int ord, REAL p_i, REAL p_e, Vec3d *coeff)
{
    index = ind;
    seg_ord = ord;
    par_val[0] = p_i;
    par_val[1] = p_e;

    a = new Vec3d[seg_ord];
    assert(a != 0);
    memcpy(a, coeff, (size_t)(sizeof(Vec3d) * seg_ord));
}

Curve_Segment::~Curve_Segment()
{
    delete[] a;
}

Curve_Segment &Curve_Segment::operator=(const Curve_Segment &_seg)
{
    if (this == &_seg)
        return *this;

    index = _seg.index;
    seg_ord = _seg.seg_ord;
    memcpy(par_val, _seg.par_val, (size_t)(sizeof(REAL) * 2));
    delete[] a;

    a = new Vec3d[seg_ord];
    assert(a != 0);
    memcpy(a, _seg.a, (size_t)(sizeof(Vec3d) * seg_ord));

    return *this;
}

void Curve_Segment::setIndex(int ind)
{
    index = ind;
}

void Curve_Segment::setOrder(int ord)
{
    seg_ord = ord;
}

void Curve_Segment::setParam(REAL p_i, REAL p_e)
{
    par_val[0] = p_i;
    par_val[1] = p_e;
}

void Curve_Segment::setCoeff(int ord, Vec3d *coeff)
{
    seg_ord = ord;
    a = new Vec3d[seg_ord];
    assert(a != 0);
    memcpy(a, coeff, (size_t)(sizeof(Vec3d) * seg_ord));
}

Matrix Curve_Segment::BezierToPowerMatrix()
{
    // Compute the pth degree Bezier matrix (p = seg_ord-1)
    // This matrix transforms Bezier form to power basis form
    int i, k, pk, j;
    REAL sign;
    REAL kf;

    int p = seg_ord - 1; // degree of curve segment
    Matrix m(p + 1); // (p+1)x(p+1) Matrix set to zero

    m[0][0] = m[p][p] = 1.0; // Set corner elements

    if (p % 2 != 0)
        m[p][0] = -1.0;
    else
        m[p][0] = 1.0;

    sign = -1.0;
    // Compute first column, last row, and the diagonal
    for (i = 1; i < p; i++)
    {
        m[i][i] = BiCo(p, i);
        m[i][0] = m[p][p - i] = sign * m[i][i];
        sign *= -1.0;
    }
    // Compute remaining elements
    kf = (p + 1) / 2;
    pk = p - 1;
    for (k = 1; k < kf; k++)
    {
        sign = -1;
        for (j = k + 1; j <= pk; j++)
        {
            m[j][k] = m[pk][p - j] = sign * BiCo(p, k) * BiCo(p - k, j - k);
            sign *= -1.0;
        }
        pk--;
    }

    return m;
}

Matrix Curve_Segment::PowerToBezierMatrix()
{
    // Compute inverse of pth-degree Bezier matrix (p = seg_ord-1)
    // This matrix transforms power basis form to Bezier form

    int i, j, k;
    int pk;
    REAL kf;
    REAL sum;

    int p = seg_ord - 1; // degree of curve segment
    Matrix m_i(seg_ord); // (p+1)x(p+1) Matrix set to zero

    // Compute the pth-degree Bezier matrix
    Matrix m = BezierToPowerMatrix();

    // Set first column, last row, and diagonal
    for (i = 0; i <= p; i++)
    {
        m_i[i][0] = m_i[p][i] = 1.0;
        m_i[i][i] = 1.0 / m[i][i];
    }
    // Compute remaining elements
    kf = (p + 1) / 2;
    pk = p - 1;

    for (k = 1; k < kf; k++)
    {
        for (j = k + 1; j <= pk; j++)
        {
            sum = 0.0;
            for (i = k; i < j; i++)
                sum = sum - m[j][i] * m_i[i][k];
            m_i[j][k] = sum / m[j][j];
            m_i[pk][p - j] = m_i[j][k];
        }
        pk--;
    }

    return m_i;
}

//============================================================================
// Surface
//============================================================================

// Constructors and Member-Functions of the classes Surf and Surf_Segment

////////////////////////////////// Surf //////////////////////////////////
Surf::Surf(const Surf &_s)
    : nps(_s.nps)
    , npt(_s.npt)
{

    name = new char[strlen(_s.name) + 1];
    assert(name != 0);
    strcpy(name, _s.name);

    order_u = new int[nps * npt];
    assert(order_u != 0);
    memcpy(order_u, _s.order_u, (size_t)(sizeof(int) * nps * npt));

    order_v = new int[nps * npt];
    assert(order_v != 0);
    memcpy(order_v, _s.order_v, (size_t)(sizeof(int) * nps * npt));

    Tbl = new Surf_Segment[nps * npt];
    assert(Tbl != 0);
    for (int i = 0; i < nps * npt; i++)
        Tbl[i] = _s.Tbl[i];
}

Surf::Surf(char *nme, int n_u, int n_v, int *ord_u, int *ord_v, Surf_Segment *segm)
{

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);
    nps = n_u;
    npt = n_v;

    order_u = new int[nps * npt];
    assert(order_u != 0);
    memcpy(order_u, ord_u, (size_t)(sizeof(int) * nps * npt));

    order_v = new int[nps * npt];
    assert(order_v != 0);
    memcpy(order_v, ord_v, (size_t)(sizeof(int) * nps * npt));

    Tbl = new Surf_Segment[nps * npt];
    assert(Tbl != 0);
    for (int i = 0; i < nps * npt; i++)
        Tbl[i] = segm[i];
}

Surf::~Surf()
{
    delete[] name;
    delete[] order_u;
    delete[] order_v;
    delete[] Tbl;
}

Surf &Surf::operator=(const Surf &_s)
{
    if (this == &_s)
        return *this;

    delete[] name;
    name = new char[strlen(_s.name) + 1];
    assert(name != 0);
    strcpy(name, _s.name);

    nps = _s.nps;
    npt = _s.npt;
    delete[] order_u;
    delete[] order_v;
    delete[] Tbl;

    order_u = new int[nps * npt];
    assert(order_u != 0);
    memcpy(order_u, _s.order_u, (size_t)(sizeof(int) * nps * npt));

    order_v = new int[nps * npt];
    assert(order_v != 0);
    memcpy(order_v, _s.order_v, (size_t)(sizeof(int) * nps * npt));

    Tbl = new Surf_Segment[nps * npt];
    assert(Tbl != 0);
    for (int i = 0; i < nps * npt; i++)
        Tbl[i] = _s.Tbl[i];

    return *this;
}

Vec3d Surf::get_coeff(int num, int pow_u, int pow_v)
{
    return (Tbl[num].a[pow_u][pow_v]);
}

Vec3d *Surf::DegreeElevateBezierRow(int p, Vec3d *CRow, int t)
{

    // Degree elevate row of a Bezier surface patch t times
    // Input:
    //         p     : degree
    //         CRow  : row of control net
    //         t     : number of degree elevations
    // Output:
    //         CRowE : degree elevated row

    int i, j;
    int pe; // elevated degree
    int mpi;

    REAL pe2;
    REAL inv;
    REAL **bdec; // Bezier degree elevation coefficients

    Vec3d *CRowE = new Vec3d[p + t + 1];

    pe = p + t;
    pe2 = pe / 2;

    // allocate matrix of Bezier degree elevation coefficients
    bdec = new REAL *[pe + 1];
    for (i = 0; i <= pe; i++)
    {
        bdec[i] = new REAL[p + 1];
        for (j = 0; j < p + 1; j++)
            bdec[i][j] = 0.0;
    }

    bdec[0][0] = bdec[pe][p] = 1.0;
    // Compute Bezier degree elevation coefficients
    for (i = 1; i <= pe2; i++)
    {
        inv = 1.0 / BiCo(pe, i);
        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
        {
            bdec[i][j] = inv * BiCo(p, j) * BiCo(t, i - j);
        }
    }
    for (i = pe2 + 1; i <= pe - 1; i++)
    {
        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
        {
            bdec[i][j] = bdec[pe - i][p - j];
        }
    }

    // Compute the control points of degree elevated row of Bezier surface patch
    for (i = 0; i <= pe; i++)
    {
        CRowE[i][0] = 0.0;
        CRowE[i][1] = 0.0;
        CRowE[i][2] = 0.0;

        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
            CRowE[i] = CRowE[i] + bdec[i][j] * CRow[j];
    }

    // free
    for (i = 0; i <= pe; i++)
        delete[] bdec[i];
    delete[] bdec;

    return CRowE;
}

void Surf::DegreeElevateBezierPatch(int p, int q, int DIR, Vec3d **CP, int t, Vec3d **CQ)
{

    // Degree elevate a Bezier surface patch t times (in the appropriate
    // direction)
    // Input:
    //         p     : degree in u direction
    //         q     : degree in v direction
    //         DIR   : flag for direction
    //         CP    : control net
    //         t     : number of degree elevations
    //
    // Output:
    //         CQ    : degree elevated control net of Bezier surface patch

    int i, j;

    Vec3d *CRow; // row of control net of Bezier patch
    Vec3d *CRowE; // degree elevated row

    if (DIR == 0) // u direction
    {
        CRow = new Vec3d[p + 1];

        for (i = 0; i <= q; i++)
        {
            for (j = 0; j <= p; j++)
                CRow[j] = CP[j][i];
            CRowE = DegreeElevateBezierRow(p, CRow, t);
            for (j = 0; j <= p + t; j++)
                CQ[j][i] = CRowE[j];
            delete[] CRowE;
        }
        delete[] CRow;
    }

    if (DIR == 1) // v direction

    {
        CRow = new Vec3d[q + 1];

        for (i = 0; i <= p; i++)
        {
            for (j = 0; j <= q; j++)
                CRow[j] = CP[i][j];
            CRowE = DegreeElevateBezierRow(q, CRow, t);
            for (j = 0; j <= q + t; j++)
                CQ[i][j] = CRowE[j];
            delete[] CRowE;
        }
        delete[] CRow;
    }
}

void Surf::MakeNurbsSurface(int tag)
{
    // Create a NURBS surface from all power basis patches
    // C(0) continuity is assumed!

    int i, j, k, l, r;
    int ordu_max; // maximum order in u direction of all patches
    int ordv_max; // maximum order in v direction of all patches
    int diff;
    int dim_u;
    int dim_v;

    int u_index;
    int v_index;

    int num_U; // number of knots of composed Bezier surface
    // in u direction
    int num_V; // number of knots of composed Bezier surface
    // in v direction
    int index_U;
    int index_V;

    Vec3d **temp;

    Matrix *p_MqT;

    NurbsSurface *nurbs;

    // Allocate control nets CP (consisting of 3-dimensional control points)
    // for all Bezier patches
    Vec3d ***CP = new Vec3d **[nps * npt];
    for (i = 0; i < nps * npt; i++)
    {
        CP[i] = new Vec3d *[order_u[i]];
        for (k = 0; k < order_u[i]; k++)
            CP[i][k] = new Vec3d[order_v[i]];
    }

    // Determine the order of the NURBS surface
    // 1st in u direction:
    ordu_max = 0;
    for (i = 0; i < nps * npt; i++)
        ordu_max = Max(order_u[i], ordu_max);
    if (ordu_max == 0)
        cerr << "ERROR: order of NURBS surface "
             << name << " in u direction wrong" << endl;
    // 2nd in v direction:
    ordv_max = 0;
    for (i = 0; i < nps * npt; i++)
        ordv_max = Max(order_v[i], ordv_max);
    if (ordv_max == 0)
        cerr << "ERROR: order of NURBS surface "
             << name << " in v direction wrong" << endl;

    // Allocate table of control nets CN (i.d. control nets after possible degree elevation)
    // for all Bezier patches
    Vec3d ***CN = new Vec3d **[nps * npt];
    for (i = 0; i < nps * npt; i++)
    {
        CN[i] = new Vec3d *[ordu_max];
        for (j = 0; j < ordu_max; j++)
            CN[i][j] = new Vec3d[ordv_max];
    }

    // Determine dimension of control net for the composed Bezier surface
    dim_u = nps * ordu_max - (nps - 1);
    dim_v = npt * ordv_max - (npt - 1);

    // Allocate control net CQ of composed Bezier surface
    Vec3d **CQ = new Vec3d *[dim_u];
    for (i = 0; i < dim_u; i++)
        CQ[i] = new Vec3d[dim_v];

    // Determine number of knots in u direction
    // for the composed Bezier surface
    num_U = 2 * ordu_max + (nps - 1) * (ordu_max - 1);
    REAL *U = new REAL[num_U];

    // Determine number of knots in v direction
    // for the composed Bezier surface
    num_V = 2 * ordv_max + (npt - 1) * (ordv_max - 1);
    REAL *V = new REAL[num_V];

    // Compute the control net of every patch
    for (i = 0; i < nps * npt; i++)
    {
        // transformation matrix Mp
        Matrix p_Mp = PowerToBezierMatrix(order_u[i] - 1);
        // transformation matrix Mq
        Matrix p_Mq = PowerToBezierMatrix(order_v[i] - 1);
        // transpose Mq
        p_MqT = new Matrix(order_v[i]);
        p_MqT->transpose(p_Mq);

        // matrix multiplication (a * MqT)
        temp = new Vec3d *[order_u[i]];
        for (k = 0; k < order_u[i]; k++)
        {
            temp[k] = new Vec3d[order_v[i]];
            for (l = 0; l < order_v[i]; l++)
            {
                temp[k][l][0] = 0.0;
                temp[k][l][1] = 0.0;
                temp[k][l][2] = 0.0;

                for (r = 0; r < order_v[i]; r++)
                    temp[k][l] = temp[k][l] + Tbl[i].a[k][r] * (*p_MqT)[r][l];
            }
        }

        // matrix multiplication (Mp * (a * MqT))
        for (k = 0; k < order_u[i]; k++)
            for (l = 0; l < order_v[i]; l++)
            {
                Vec3d init(0.0, 0.0, 0.0);
                CP[i][k][l] = init;

                for (r = 0; r < order_u[i]; r++)
                    CP[i][k][l] = CP[i][k][l] + p_Mp[k][r] * temp[r][l];
            }

        /*
           // verify the control nets
           cout << "Control net of patch "
                << Tbl[i].number << ":" << endl;
           for (k=0; k<order_u[i]; k++)
      {
               for (l=0; l<order_v[i]; l++)
          {
                    cout << k << "," << l << ": ";
                    CP[i][k][l].output();
               }
      }
      cout << endl;
      */

        for (k = 0; k < order_u[i]; k++)
            delete[] temp[k];
        delete[] temp;
        delete p_MqT;
    }

    // Degree elevation of patches and composition
    for (i = 0; i < nps * npt; i++)
    {
        if (order_u[i] < ordu_max)
        {
            diff = ordu_max - order_u[i];
            /*
         cout << "Degree elevation in u direction of patch "
                        << Tbl[i].number << endl;
         */
            DegreeElevateBezierPatch(
                order_u[i] - 1,
                order_v[i] - 1,
                U_DIR,
                CP[i],
                diff,
                CN[i]);

            if (order_v[i] < ordv_max) // degree elevation in both directions
            {

                // temporary control net after first degree elevation in u direction
                temp = new Vec3d *[ordu_max];
                for (k = 0; k < ordu_max; k++)
                {
                    temp[k] = new Vec3d[order_v[i]];
                    for (l = 0; l < order_v[i]; l++)
                        temp[k][l] = CN[i][k][l];
                }
                diff = ordv_max - order_v[i];
                /*
              cout << "Degree elevation in v direction of patch "
                   << Tbl[i].number << " after degree elevation in u direction" << endl;
            */
                DegreeElevateBezierPatch(
                    ordu_max - 1,
                    order_v[i] - 1,
                    V_DIR,
                    temp,
                    diff,
                    CN[i]);
                for (k = 0; k < ordu_max; k++)
                    delete[] temp[k];
                delete[] temp;
            }
        }

        else if (order_v[i] < ordv_max)
        {
            diff = ordv_max - order_v[i];
            /*
                cout << "Degree elevation in v direction of patch "
                     << Tbl[i].number << endl;
         */
            DegreeElevateBezierPatch(
                order_u[i] - 1,
                order_v[i] - 1,
                V_DIR,
                CP[i],
                diff,
                CN[i]);
        }

        else if (order_u[i] == ordu_max && order_v[i] == ordv_max)
        {
            for (k = 0; k < ordu_max; k++)
                for (l = 0; l < ordv_max; l++)
                    CN[i][k][l] = CP[i][k][l]; // simple assignment (no degree
            // elevation necessary)
        }
    }

    // Compose the control nets of different patches to one control net of
    // the NURBS surface
    u_index = 0;
    v_index = 0;
    for (i = 0; i < nps; i++)
    {
        for (j = 0; j < npt; j++)
        {

            for (k = 0; k < ordu_max; k++)
                for (l = 0; l < ordv_max; l++)
                    CQ[u_index + k][v_index + l] = CN[i * npt + j][k][l];
            v_index += ordv_max - 1;
        }
        v_index = 0;
        u_index += ordu_max - 1;
    }

    /*
   // verify the control net CQ
   cout << "Control net CQ:" << endl;
   for (i=0; i<dim_u; i++)
        for (j=0; j<dim_v; j++)
   {
             cout << i << "," << j << ": ";
             CQ[i][j].output();

   }
   */

    // Compute knot vectors
    // U vector:
    index_U = 0;
    for (j = 0; j < ordu_max; j++)
    {
        U[j] = Tbl[0].par_s[0];
    }
    index_U += ordu_max;
    if (nps > 1)
    {
        for (i = 1; i < nps; i++)
        {
            for (j = 0; j < ordu_max - 1; j++)
            {
                U[index_U + j] = Tbl[i * npt].par_s[0];
            }
            index_U += ordu_max - 1;
        }
    }
    for (j = 0; j < ordu_max; j++)
    {
        U[index_U + j] = Tbl[(nps - 1) * npt].par_s[1];
    }
    // V vector:
    index_V = 0;
    for (j = 0; j < ordv_max; j++)
    {
        V[j] = Tbl[0].par_t[0];
    }
    index_V += ordv_max;
    if (npt > 1)
    {
        for (i = 1; i < npt; i++)
        {
            for (j = 0; j < ordv_max - 1; j++)
            {
                V[index_V + j] = Tbl[i].par_t[0];
            }
            index_V += ordv_max - 1;
        }
    }
    for (j = 0; j < ordv_max; j++)
    {
        V[index_V + j] = Tbl[npt - 1].par_t[1];
    }

    /*
   // verify knot vectors
   cout << "knot vector U:" << endl;
   cout << "( ";
   for (i=0; i< num_U; i++)
        cout << U[i] << "\t";
   cout << " )" << endl;

   cout << "knot vector V:" << endl;
   cout << "( ";
   for (i=0; i< num_V; i++)
   cout << V[i] << "\t";
   cout << " )" << endl;
   */

    // set NURBS object and add to list
    nurbs = new NurbsSurface(num_U, num_V, dim_u, dim_v);
    nurbs->set_Uknt(U);
    nurbs->set_Vknt(V);
    nurbs->set_net(CQ);
    // maximum knot removal
    nurbs->MaximumKnotRemoval();

    switch (tag)
    {
    case SURF: // surface without trimming regions
        nurbssurfaceList.append(*nurbs);
        break;
    case FACE: // surface with trimming regions
        surfaceDefList.append(*nurbs);
        break;
    default:
        break;
    }

    // free
    for (i = 0; i < nps * npt; i++)
    {
        for (k = 0; k < order_u[i]; k++)
            delete[] CP[i][k];
        delete[] CP[i];
    }
    delete[] CP;

    for (i = 0; i < nps * npt; i++)
    {
        for (k = 0; k < ordu_max; k++)
            delete[] CN[i][k];
        delete[] CN[i];
    }
    delete[] CN;

    for (i = 0; i < dim_u; i++)
        delete[] CQ[i];
    delete[] CQ;

    delete[] U;
    delete[] V;
    delete nurbs;
}

////////////////////////////// Surf_Segment //////////////////////////////
Surf_Segment::Surf_Segment(const Surf_Segment &_seg)
    : number(_seg.number)
    , seg_ord_u(_seg.seg_ord_u)
    , seg_ord_v(_seg.seg_ord_v)
{
    memcpy(par_s, _seg.par_s, (size_t)(sizeof(REAL) * 2));
    memcpy(par_t, _seg.par_t, (size_t)(sizeof(REAL) * 2));

    a = new Vec3d *[seg_ord_u];
    assert(a != 0);
    for (int i = 0; i < seg_ord_u; i++)
    {
        a[i] = new Vec3d[seg_ord_v];
        assert(a[i] != 0);
        memcpy(a[i], _seg.a[i], (size_t)(sizeof(Vec3d) * seg_ord_v));
    }
}

Surf_Segment::Surf_Segment(int num, int ord_u, int ord_v,
                           REAL ps_i, REAL ps_e,
                           REAL pt_i, REAL pt_e,
                           Vec3d **coeff)
{
    number = num;
    seg_ord_u = ord_u;
    seg_ord_v = ord_v;
    par_s[0] = ps_i;
    par_s[1] = ps_e;
    par_t[0] = pt_i;
    par_t[1] = pt_e;

    a = new Vec3d *[seg_ord_u];
    assert(a != 0);
    for (int i = 0; i < seg_ord_u; i++)
    {
        a[i] = new Vec3d[seg_ord_v];
        assert(a[i] != 0);
        memcpy(a[i], coeff[i], (size_t)(sizeof(Vec3d) * seg_ord_v));
    }
}

Surf_Segment::~Surf_Segment()
{
    for (int i = 0; i < seg_ord_u; i++)
        delete[] a[i];
    delete[] a;
}

Surf_Segment &Surf_Segment::operator=(const Surf_Segment &_seg)
{
    if (this == &_seg)
        return *this;

    int i;
    for (i = 0; i < seg_ord_u; i++)
        delete[] a[i];
    delete[] a;

    number = _seg.number;
    seg_ord_u = _seg.seg_ord_u;
    seg_ord_v = _seg.seg_ord_v;

    memcpy(par_s, _seg.par_s, (size_t)(sizeof(REAL) * 2));
    memcpy(par_t, _seg.par_t, (size_t)(sizeof(REAL) * 2));

    a = new Vec3d *[seg_ord_u];
    assert(a != 0);
    for (i = 0; i < seg_ord_u; i++)
    {
        a[i] = new Vec3d[seg_ord_v];
        assert(a[i] != 0);
        memcpy(a[i], _seg.a[i], (size_t)(sizeof(Vec3d) * seg_ord_v));
    }

    return *this;
}

//============================================================================
// Cons
//============================================================================

// Constructors and Member-Functions of the classes Cons and Cons_Segment

////////////////////////////////// Cons //////////////////////////////////
Cons::Cons(const Cons &_c)
    : s1(_c.s1)
    , s2(_c.s2)
    , n_segments(_c.n_segments)
{
    name = new char[strlen(_c.name) + 1];
    assert(name != 0);
    strcpy(name, _c.name);

    surfnme = new char[strlen(_c.surfnme) + 1];
    assert(surfnme != 0);
    strcpy(surfnme, _c.surfnme);

    curvenme = new char[strlen(_c.curvenme) + 1];
    assert(curvenme != 0);
    strcpy(curvenme, _c.curvenme);

    order = new int[n_segments];
    assert(order != 0);
    memcpy(order, _c.order, (size_t)(sizeof(int) * n_segments));

    Tbl = new Cons_Segment[n_segments];
    assert(Tbl != 0);
    for (int i = 0; i < n_segments; i++)
        Tbl[i] = _c.Tbl[i];
}

Cons::Cons(char *nme, char *sfnme, char *cvnme, REAL p_i, REAL p_f,
           int n, int *ord, Cons_Segment *segm)
{

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    surfnme = new char[strlen(sfnme) + 1];
    assert(surfnme != 0);
    strcpy(surfnme, sfnme);

    curvenme = new char[strlen(cvnme) + 1];
    assert(curvenme != 0);
    strcpy(curvenme, cvnme);

    s1 = p_i;
    s2 = p_f;
    n_segments = n;

    order = new int[n_segments];
    assert(order != 0);
    memcpy(order, ord, (size_t)(sizeof(int) * n_segments));

    Tbl = new Cons_Segment[n_segments];
    assert(Tbl != 0);
    for (int i = 0; i < n_segments; i++)
        Tbl[i] = segm[i];
}

Cons::~Cons()
{
    delete[] name;
    delete[] surfnme;
    delete[] curvenme;
    delete[] order;
    delete[] Tbl;
}

Vec2d Cons::get_coeff(int ind, int power)
{
    // range of power: 0 <= power <= degree
    return (Tbl[ind - 1].a[power]);
}

char *Cons::get_curvenme()
{
    return curvenme;
}

REAL Cons::get_s1()
{
    return s1;
}

REAL Cons::get_s2()
{
    return s2;
}

Cons &Cons::operator=(const Cons &_c)
{
    if (this == &_c)
        return *this;

    delete[] name;
    delete[] surfnme;
    delete[] curvenme;
    delete[] order;
    delete[] Tbl;

    name = new char[strlen(_c.name) + 1];
    assert(name != 0);
    strcpy(name, _c.name);

    surfnme = new char[strlen(_c.surfnme) + 1];
    assert(surfnme != 0);
    strcpy(surfnme, _c.surfnme);

    curvenme = new char[strlen(_c.curvenme) + 1];
    assert(curvenme != 0);
    strcpy(curvenme, _c.curvenme);

    s1 = _c.s1;
    s2 = _c.s2;
    n_segments = _c.n_segments;

    order = new int[n_segments];
    assert(order != 0);
    memcpy(order, _c.order, (size_t)(sizeof(int) * n_segments));

    Tbl = new Cons_Segment[n_segments];
    assert(Tbl != 0);
    for (int i = 0; i < n_segments; i++)
        Tbl[i] = _c.Tbl[i];

    return *this;
}

Vec2d *Cons::DegreeElevateBezierSegment(
    int p,
    const Vec2d *CP,
    int t)
{

    // Degree elevate a Bezier Cons segment t times
    // Input:
    //         p     : degree
    //         CP    : control points
    //         t     : number of degree elevations
    // Output:
    //         CQ    : new control points

    int i, j;
    int pe; // elevated degree
    int mpi;

    REAL pe2;
    REAL inv;
    REAL **bdec; // Bezier degree elevation coefficients

    Vec2d *CQ = new Vec2d[p + t + 1];

    pe = p + t;
    pe2 = pe / 2;

    // allocate matrix of Bezier degree elevation coefficients
    bdec = new REAL *[pe + 1];
    for (i = 0; i <= pe; i++)
    {
        bdec[i] = new REAL[p + 1];
        for (j = 0; j < p + 1; j++)
            bdec[i][j] = 0.0;
    }

    bdec[0][0] = bdec[pe][p] = 1.0;
    // Compute Bezier degree elevation coefficients
    for (i = 1; i <= pe2; i++)
    {
        inv = 1.0 / BiCo(pe, i);
        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
        {
            bdec[i][j] = inv * BiCo(p, j) * BiCo(t, i - j);
        }
    }
    for (i = pe2 + 1; i <= pe - 1; i++)
    {
        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
        {
            bdec[i][j] = bdec[pe - i][p - j];
        }
    }

    // Compute the control points of degree elevated Bezier segment
    for (i = 0; i <= pe; i++)
    {
        CQ[i][0] = 0.0;
        CQ[i][1] = 0.0;

        mpi = Min(p, i);
        for (j = Max(0, i - t); j <= mpi; j++)
            CQ[i] = CQ[i] + bdec[i][j] * CP[j];
    }

    /*
       cout << "degree elevation:" << endl;
       for (i=0; i<=pe; i++)
       {
             CQ[i].output();
       }
       cout << endl;
   */

    // free
    for (i = 0; i <= pe; i++)
        delete[] bdec[i];
    delete[] bdec;

    return CQ;
}

void Cons::MakeNurbsCons()
{
    // Create a NURBS Cons from all power basis segments
    // C(0) continuity is assumed!

    int i, j, k;
    int num = 0;
    int order_max; // maximum order of all segments
    int diff;
    int num_U; // number of knots of composed Bezier Cons
    int index_U;
    int num_CP;
    int index_CP;
    int num_CQ; // number of control points of composed
    // Bezier Cons
    Vec2d *CR;
    Vec2d *CS;

    TrimCurve *nurbs;
    enum STATUS
    {
        failed,
        succeeded
    };

    // Determine number of 2-dimensional control points for all segments
    num_CP = 0;
    for (i = 0; i < n_segments; i++)
        num_CP = num_CP + order[i];
    num_CP = num_CP - (n_segments - 1);

    Vec2d *CP = new Vec2d[num_CP]; // control points of non degree
    // elevated segments

    // Determine the order of the NURBS Cons
    order_max = 0;
    for (i = 0; i < n_segments; i++)
        order_max = Max(order[i], order_max);
    if (order_max == 0)
        cerr << "ERROR: order of NURBS Cons wrong" << endl;

    // Determine number of control points for the composed Bezier Cons
    num_CQ = n_segments * order_max - (n_segments - 1);

    Vec2d *CQ = new Vec2d[num_CQ]; // control points of composed
    // Bezier Cons

    // Determine number of knots for the composed Bezier Cons
    num_U = 2 * order_max + (n_segments - 1) * (order_max - 1);
    REAL *U = new REAL[num_U];

    // Compute the control points of segments
    for (i = 0; i < n_segments; i++)
    {
        Matrix p_m = Tbl[i].PowerToBezierMatrix();

        for (j = 0; j < order[i]; j++)
        {
            CP[j + num][0] = CP[j + num][1] = 0;
            for (k = 0; k < order[i]; k++)
            {
                CP[j + num] = CP[j + num] + (p_m[j][k] * Tbl[i].a[k]);
            }

            // cout << j+num << ": ";
            // CP[j+num].output();
        }
        num += order[i] - 1;
    }

    num = 0;
    // Degree elevation of segments and composition
    for (i = 0; i < n_segments; i++)
    {
        if (order[i] < order_max)
        {
            diff = order_max - order[i];
            CR = new Vec2d[order[i]];
            // select control points
            for (j = 0; j < order[i]; j++)
            {
                CR[j] = CP[j + num];
                // cout << j << ": ";
                // CR[j].output();
            }

            CS = DegreeElevateBezierSegment(order[i] - 1, CR, diff);

            // assign control points of degree elevated segment
            for (j = 0; j < order_max; j++)
            {
                CQ[i * (order_max - 1) + j] = CS[j];
            }

            delete[] CR;
            delete[] CS;
        }
        num += order[i] - 1;

        if (order[i] == order_max)
        {
            // assign original control points
            index_CP = 0;
            for (k = 0; k < i; k++)
                index_CP = index_CP + order[k] - 1;
            for (j = 0; j < order_max; j++)
                CQ[i * (order_max - 1) + j] = CP[index_CP + j];
        }
    }
    // Compute knot vector
    index_U = 0;
    for (j = 0; j < order_max; j++)
    {
        U[j] = Tbl[0].par_val[0];
    }
    index_U += order_max;
    if (n_segments > 1)
    {
        for (i = 1; i < n_segments; i++)
        {
            for (j = 0; j < order_max - 1; j++)
            {
                U[index_U + j] = Tbl[i].par_val[0];
            }
            index_U += order_max - 1;
        }
    }
    for (j = 0; j < order_max; j++)
    {
        U[index_U + j] = Tbl[n_segments - 1].par_val[1];
    }

    /*
      // verify
      cout << "control points:" << endl;
      for (i=0; i< num_CQ; i++)
      {
           cout << i << ": ";
           CQ[i].output();
      }

      cout << "knot vector:" << endl;
      cout << "( ";
   for (i=0; i< num_U; i++)
   cout << U[i] << "\t";
   cout << " )" << endl;
   */

    // set NURBS object and add to list
    nurbs = new TrimCurve(num_U, num_CQ);
    nurbs->set_knt(U);
    nurbs->set_pol(CQ);
    // maximum knot removal
    nurbs->MaximumKnotRemoval();
    curveDefList.append(*nurbs);

    /*
      // GLU (our rendering library) can only handle trimming NURBS curve with order up to 9
      // (! for OpenGL Optimizer not valid !)
      if (order_max > 9)
        {
          int t = order_max - 9;
          FLAG *error = new FLAG[t];     // error flags
          STATUS tag = succeeded;
          TrimCurve rnurbs = *nurbs;     // degree reduced trimming NURBS curve
                                         // initialized by copying the original NURBS object

   for (i=0; i<t; i++)
   {
   // Note: The object rnurbs is overwritten by its member function
   error[i] = rnurbs.DegreeReduceTrimCurve();
   if (error[i] != 0) tag = failed;
   }

   if (tag == succeeded)
   {
   // trimming curve t times degree reducible
   rnurbs.MaximumKnotRemoval();
   curveDefList.append(rnurbs);
   }
   else // tag = failed
   {
   // We cannot render the original trimming curve.
   // But for consistency we add it to the list.
   cerr << "Error: Trimming curve not degree reducible" << endl;
   rnurbs.MaximumKnotRemoval();
   curveDefList.append(*nurbs);
   }
   delete [] error;
   }
   else
   {
   nurbs->MaximumKnotRemoval();
   curveDefList.append(*nurbs);
   }
   */

    delete[] CQ;
    delete[] CP;
    delete[] U;
    delete nurbs;
};

Vec2d *Cons::CoeffTrafo(int no, REAL alpha, REAL beta)
{
    // Compute transformed coefficients of polynom for
    // a segment of partial cons
    // INPUT:
    //           no: number of original segment (no = index -1)
    //        alpha: transformation coefficient
    //         beta: transformation coefficient

    int i, j;
    int ord; // order of segment
    ord = Tbl[no].seg_ord;

    // Allocate and initialize the transformed coefficients
    Vec2d *at = new Vec2d[ord];
    for (i = 0; i < ord; i++)
    {
        at[i][0] = at[i][1] = 0.0;
    }

    // Transformation
    for (i = 0; i < ord; i++)
    {
        if (beta != 0)
            for (j = i; j < ord; j++)
                at[i] = at[i] + BiCo(j, i) * pow(beta, j - i) * pow(alpha, i) * Tbl[no].a[j];
        else
            at[i] = at[i] + pow(alpha, i) * Tbl[no].a[i];
    }

    return at;
}

void Cons::PartOfCons(REAL w1, REAL w2)
{
    // Create a partial cons of the original cons (in NURBS representation)
    // defined by the global parameters w1 and w2
    // Input:
    //         w1: start value of defintion range of partial cons
    //         w2: end of definition range of partial cons

    int i;
    int it;
    int it_i, it_e;
    int num;
    int index_i; // index of first cons segment of partial cons
    int index_e; // index of last cons segment of partial cons
    int n_part; // number of segments of partial cons

    int *ord; // order of polynoms for the partial cons
    REAL denom;
    REAL alpha, beta;
    Vec2d *coeff;
    Cons_Segment *seg; // table of cons segments of partial cons
    Cons *c; // partial cons

    // We don't care here about the orientation of the partial cons.
    // Only ascending intervals are allowed.
    // If the partial cons is used as a trimming curve
    // the orientation is only considered in the renderer.
    // Alterations are carried out there if necessary.
    if (w1 > w2)
    {
        Exchange(&w1, &w2);
    }
    // Now: w1 < w2 !!!

    // Determine the cons segment indices where the interval borders
    // are lying
    for (i = 0; i < n_segments; i++)
    {
        if (w1 >= Tbl[i].par_val[0] && w1 < Tbl[i].par_val[1])
            index_i = Tbl[i].index;

        if (w2 > Tbl[i].par_val[0] && w2 <= Tbl[i].par_val[1])
            index_e = Tbl[i].index;
    }

    // verify
    // cout << "Partial cons" << endl;
    // cout << "Index-Tupel: (" << index_i << ", " << index_e << ")" << endl;

    if (index_i == index_e) // only one segment
    {
        n_part = 1;
        seg = new Cons_Segment;
        it = index_i - 1;
        denom = Tbl[it].par_val[1] - Tbl[it].par_val[0];
        alpha = (w2 - w1) / denom;
        beta = (w1 - Tbl[it].par_val[0]) / denom;

        if (w1 == Tbl[it].par_val[0] && w2 == Tbl[it].par_val[1])
        {
            // select original segment (with new cons segment index)
            seg->setIndex(n_part);
            seg->setParam(w1, w2);
            seg->setCoeff(Tbl[it].seg_ord, Tbl[it].a);
            c = new Cons(
                name,
                surfnme,
                curvenme,
                s1,
                s2,
                n_part,
                &(order[it]),
                seg);
            c->MakeNurbsCons();
        }
        else
        {
            // Computing of new coefficients
            coeff = CoeffTrafo(it, alpha, beta);
            seg->setIndex(n_part);
            seg->setParam(w1, w2);
            seg->setCoeff(Tbl[it].seg_ord, coeff);
            c = new Cons(
                name,
                surfnme,
                curvenme,
                s1,
                s2,
                n_part,
                &(order[it]),
                seg);
            c->MakeNurbsCons();
            delete[] coeff;
        }
        delete seg;
        delete c;
    }

    if (index_i < index_e) // several segments
    {
        n_part = index_e - index_i + 1;
        ord = new int[n_part];
        seg = new Cons_Segment[n_part];
        it_i = index_i - 1;
        it_e = index_e - 1;
        num = 0;

        if (w1 == Tbl[it_i].par_val[0])
        {
            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it_i].par_val[0], Tbl[it_i].par_val[1]);
            seg[num].setCoeff(Tbl[it_i].seg_ord, Tbl[it_i].a);
            ord[num] = Tbl[it_i].seg_ord;
            num++;
        }
        else
        {
            // Computing new coefficients
            denom = Tbl[it_i].par_val[1] - Tbl[it_i].par_val[0];
            alpha = (Tbl[it_i].par_val[1] - w1) / denom;
            beta = (w1 - Tbl[it_i].par_val[0]) / denom;
            coeff = CoeffTrafo(it_i, alpha, beta);

            seg[num].setIndex(num + 1);
            seg[num].setParam(w1, Tbl[it_i].par_val[1]);
            seg[num].setCoeff(Tbl[it_i].seg_ord, coeff);
            ord[num] = Tbl[it_i].seg_ord;
            num++;
        }

        for (it = it_i + 1; it < it_e; it++)
        {
            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it].par_val[0], Tbl[it].par_val[1]);
            seg[num].setCoeff(Tbl[it].seg_ord, Tbl[it].a);
            ord[num] = Tbl[it].seg_ord;
            num++;
        }

        if (w2 == Tbl[it_e].par_val[1])
        {
            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it_e].par_val[0], Tbl[it_e].par_val[1]);
            seg[num].setCoeff(Tbl[it_e].seg_ord, Tbl[it_e].a);
            ord[num] = Tbl[it_e].seg_ord;
        }
        else
        {
            // Computing new coefficients
            denom = Tbl[it_e].par_val[1] - Tbl[it_e].par_val[0];
            alpha = (w2 - Tbl[it_e].par_val[0]) / denom;
            beta = 0;
            coeff = CoeffTrafo(it_e, alpha, beta);

            seg[num].setIndex(num + 1);
            seg[num].setParam(Tbl[it_e].par_val[0], w2);
            seg[num].setCoeff(Tbl[it_e].seg_ord, coeff);
            ord[num] = Tbl[it_e].seg_ord;
        }

        c = new Cons(
            name,
            surfnme,
            curvenme,
            s1,
            s2,
            n_part,
            ord,
            seg);
        c->MakeNurbsCons();
        delete[] seg;
        delete[] ord;
        delete c;
    }

    if (index_i > index_e)
        cerr << "ERROR: Index sequence of partial cons incorrect!"
             << endl;
}

////////////////////////////// Cons_Segment //////////////////////////////
Cons_Segment::Cons_Segment(const Cons_Segment &_seg)
    : index(_seg.index)
    , seg_ord(_seg.seg_ord)
{
    memcpy(par_val, _seg.par_val, (size_t)(sizeof(REAL) * 2));

    a = new Vec2d[seg_ord];
    assert(a != 0);
    memcpy(a, _seg.a, (size_t)(sizeof(Vec2d) * seg_ord));
};

Cons_Segment::Cons_Segment(int ind, int ord, REAL p_i, REAL p_e, Vec2d *coeff)
{
    index = ind;
    seg_ord = ord;
    par_val[0] = p_i;
    par_val[1] = p_e;

    a = new Vec2d[seg_ord];
    assert(a != 0);
    memcpy(a, coeff, (size_t)(sizeof(Vec2d) * seg_ord));
};

Cons_Segment::~Cons_Segment()
{
    delete[] a;
};

Cons_Segment &Cons_Segment::operator=(const Cons_Segment &_seg)
{
    if (this == &_seg)
        return *this;

    index = _seg.index;
    seg_ord = _seg.seg_ord;
    memcpy(par_val, _seg.par_val, (size_t)(sizeof(REAL) * 2));
    delete[] a;

    a = new Vec2d[seg_ord];
    assert(a != 0);
    memcpy(a, _seg.a, (size_t)(sizeof(Vec2d) * seg_ord));

    return *this;
};

void Cons_Segment::setIndex(int ind)
{
    index = ind;
}

void Cons_Segment::setOrder(int ord)
{
    seg_ord = ord;
}

void Cons_Segment::setParam(REAL p_i, REAL p_e)
{
    par_val[0] = p_i;
    par_val[1] = p_e;
}

void Cons_Segment::setCoeff(int ord, Vec2d *coeff)
{
    seg_ord = ord;
    a = new Vec2d[seg_ord];
    assert(a != 0);
    memcpy(a, coeff, (size_t)(sizeof(Vec2d) * seg_ord));
}

Matrix Cons_Segment::BezierToPowerMatrix()
{
    // Compute the pth degree Bezier matrix (p = seg_ord-1)
    // This matrix transforms Bezier form to power basis form
    int i, k, pk, j;
    REAL sign;
    REAL kf;

    int p = seg_ord - 1; // degree of Cons segment
    Matrix m(p + 1); // (p+1)x(p+1) Matrix set to zero

    m[0][0] = m[p][p] = 1.0; // Set corner elements

    if (p % 2 != 0)
        m[p][0] = -1.0;
    else
        m[p][0] = 1.0;

    sign = -1.0;
    // Compute first column, last row, and the diagonal
    for (i = 1; i < p; i++)
    {
        m[i][i] = BiCo(p, i);
        m[i][0] = m[p][p - i] = sign * m[i][i];
        sign *= -1.0;
    }
    // Compute remaining elements
    kf = (p + 1) / 2;
    pk = p - 1;
    for (k = 1; k < kf; k++)
    {
        sign = -1;
        for (j = k + 1; j <= pk; j++)
        {
            m[j][k] = m[pk][p - j] = sign * BiCo(p, k) * BiCo(p - k, j - k);
            sign *= -1.0;
        }
        pk--;
    }

    return m;
}

Matrix Cons_Segment::PowerToBezierMatrix()
{
    // Compute inverse of pth-degree Bezier matrix (p = seg_ord-1)
    // This matrix transforms power basis form to Bezier form

    int i, j, k;
    int pk;
    REAL kf;
    REAL sum;

    int p = seg_ord - 1; // degree of Cons segment
    Matrix m_i(seg_ord); // (p+1)x(p+1) Matrix set to zero

    // Compute the pth-degree Bezier matrix
    Matrix m = BezierToPowerMatrix();

    // Set first column, last row, and diagonal
    for (i = 0; i <= p; i++)
    {
        m_i[i][0] = m_i[p][i] = 1.0;
        m_i[i][i] = 1.0 / m[i][i];
    }
    // Compute remaining elements
    kf = (p + 1) / 2;
    pk = p - 1;

    for (k = 1; k < kf; k++)
    {
        for (j = k + 1; j <= pk; j++)
        {
            sum = 0.0;
            for (i = k; i < j; i++)
                sum = sum - m[j][i] * m_i[i][k];
            m_i[j][k] = sum / m[j][j];
            m_i[pk][p - j] = m_i[j][k];
        }
        pk--;
    }

    return m_i;
}

//============================================================================
// Face
//============================================================================

// Constructors and Member-Functions of the classes Face and Cons_Ensemble

/////////////////////////////////// Face ///////////////////////////////////
Face::Face(const Face &_f)
    : m(_f.m)
{
    name = new char[strlen(_f.name) + 1];
    assert(name != 0);
    strcpy(name, _f.name);

    surfnme = new char[strlen(_f.surfnme) + 1];
    assert(surfnme != 0);
    strcpy(surfnme, _f.surfnme);

    Tbl = new Cons_Ensemble[m];
    assert(Tbl != 0);
    for (int i = 0; i < m; i++)
        Tbl[i] = _f.Tbl[i];
}

Face::Face(char *nme, char *sfnme, int n, Cons_Ensemble *ens)
    : m(n)
{
    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    surfnme = new char[strlen(sfnme) + 1];
    assert(surfnme != 0);
    strcpy(surfnme, sfnme);

    Tbl = new Cons_Ensemble[m];
    assert(Tbl != 0);
    for (int i = 0; i < m; i++)
        Tbl[i] = ens[i];
}

Face::~Face()
{
    delete[] name;
    delete[] surfnme;
    delete[] Tbl;
}

void Face::set_connectionList(int mode)
{
    static int tLoop_pos = 0;

    if (mode == DELETE)
        tLoop_pos = 0;

    connectionList.append(tLoop_pos);
    tLoop_pos += m;
}

void Face::set_trimLoopList(int mode)
{
    static int tCurve_pos = 0;

    if (mode == DELETE)
        tCurve_pos = 0;

    for (int i = 0; i < m; i++)
    {
        trimLoopList.append(tCurve_pos);
        tCurve_pos += Tbl[i].n_cons;
    }
}

Face &Face::operator=(const Face &_f)
{
    if (this == &_f)
        return *this;

    delete[] name;
    delete[] surfnme;
    delete[] Tbl;

    name = new char[strlen(_f.name) + 1];
    assert(name != 0);
    strcpy(name, _f.name);

    surfnme = new char[strlen(_f.surfnme) + 1];
    assert(surfnme != 0);
    strcpy(surfnme, _f.surfnme);

    m = _f.m;
    Tbl = new Cons_Ensemble[m];
    assert(Tbl != 0);
    for (int i = 0; i < m; i++)
        Tbl[i] = _f.Tbl[i];

    return *this;
}

////////////////////////////// Cons_Ensemble //////////////////////////////
Cons_Ensemble::Cons_Ensemble(const Cons_Ensemble &_ens)
    : n_cons(_ens.n_cons)
{
    name = new char *[n_cons];
    for (int i = 0; i < n_cons; i++)
    {
        name[i] = new char[strlen(_ens.name[i]) + 1];
        assert(name[i] != 0);
        strcpy(name[i], _ens.name[i]);
    }

    w1 = new REAL[n_cons];
    assert(w1 != 0);
    memcpy(w1, _ens.w1, (size_t)(sizeof(REAL) * n_cons));

    w2 = new REAL[n_cons];
    assert(w2 != 0);
    memcpy(w2, _ens.w2, (size_t)(sizeof(REAL) * n_cons));
}

Cons_Ensemble::Cons_Ensemble(int nc, char **nme, REAL *ip, REAL *fp)
    : n_cons(nc)
{

    name = new char *[n_cons];
    for (int i = 0; i < n_cons; i++)
    {
        name[i] = new char[strlen(nme[i]) + 1];
        assert(name[i] != 0);
        strcpy(name[i], nme[i]);
    }

    w1 = new REAL[n_cons];
    assert(w1 != 0);
    memcpy(w1, ip, (size_t)(sizeof(REAL) * n_cons));

    w2 = new REAL[n_cons];
    assert(w2 != 0);
    memcpy(w2, fp, (size_t)(sizeof(REAL) * n_cons));
}

Cons_Ensemble::~Cons_Ensemble()
{
    for (int i = 0; i < n_cons; i++)
        delete[] name[i];
    delete[] name;
    delete[] w1;
    delete[] w2;
}

Cons_Ensemble &Cons_Ensemble::operator=(const Cons_Ensemble &_ens)
{

    if (this == &_ens)
        return *this;
    //variables must be declared outside 'for'
    //since otheriwse under Linux we get
    //name lookup of `i' changed for new ANSI `for' scoping
    int i;
    for (i = 0; i < n_cons; i++)
        delete[] name[i];
    delete[] name;
    delete[] w1;
    delete[] w2;

    n_cons = _ens.n_cons;
    name = new char *[n_cons];
    for (i = 0; i < n_cons; i++)
    {
        name[i] = new char[strlen(_ens.name[i]) + 1];
        assert(name[i] != 0);
        strcpy(name[i], _ens.name[i]);
    }

    w1 = new REAL[n_cons];
    assert(w1 != 0);
    memcpy(w1, _ens.w1, (size_t)(sizeof(REAL) * n_cons));

    w2 = new REAL[n_cons];
    assert(w2 != 0);
    memcpy(w2, _ens.w2, (size_t)(sizeof(REAL) * n_cons));

    return *this;
}

//============================================================================
// Top
//============================================================================

// Constructors and Member-Functions of the class Top
Top::Top()
{
    name = NULL;
    m = 0;
    fsname = NULL;
    Tbl = NULL;
    icont = NULL;
}

Top::Top(const Top &_t)
    : m(_t.m)
{
    int i;
    int num = 2 * m;

    name = new char[strlen(_t.name) + 1];
    assert(name != 0);
    strcpy(name, _t.name);

    fsname = new char *[num];
    assert(fsname != 0);
    Tbl = new Cons_Ensemble[num];
    assert(Tbl != 0);

    for (i = 0; i < num; i++)
    {
        fsname[i] = new char[strlen(_t.fsname[i]) + 1];
        assert(fsname[i] != 0);
        strcpy(fsname[i], _t.fsname[i]);
        Tbl[i] = _t.Tbl[i];
    }

    icont = new int[m];
    assert(icont != 0);
    memcpy(icont, _t.icont, (size_t)(sizeof(int) * m));
}

Top::Top(char *nme, int n_pairs, char **fs, Cons_Ensemble *ens, int *cont)
    : m(n_pairs)
{
    int i;
    int num = 2 * m;

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    fsname = new char *[num];
    assert(fsname != 0);
    Tbl = new Cons_Ensemble[num];
    assert(Tbl != 0);

    for (i = 0; i < num; i++)
    {
        fsname[i] = new char[strlen(fs[i]) + 1];
        assert(fsname[i] != 0);
        strcpy(fsname[i], fs[i]);
        Tbl[i] = ens[i];
    }

    icont = new int[m];
    assert(icont != 0);
    memcpy(icont, cont, (size_t)(sizeof(int) * m));
}

Top::Top(char *nme, int n_pairs)
    : m(n_pairs)
{
    int num = 2 * m;

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    fsname = new char *[num];
    assert(fsname != 0);

    Tbl = new Cons_Ensemble[num];
    assert(Tbl != 0);

    icont = new int[m];
    assert(icont != 0);
}

Top::~Top()
{
    delete[] name;
    for (int i = 0; i < 2 * m; i++)
    {
        delete[] fsname[i];
    }
    delete[] fsname;
    delete[] Tbl;
    delete[] icont;
}

int Top::get_m_pairs()
{
    return m;
}

char **Top::get_fsNames()
{
    return fsname;
}

Top &Top::operator=(const Top &_t)
{
    if (this == &_t)
        return *this;

    int i;

    delete[] name;
    for (i = 0; i < 2 * m; i++)
        delete[] fsname[i];
    delete[] fsname;
    delete[] Tbl;
    delete[] icont;

    name = new char[strlen(_t.name) + 1];
    assert(name != 0);
    strcpy(name, _t.name);

    m = _t.m;
    fsname = new char *[2 * m];
    assert(fsname != 0);
    Tbl = new Cons_Ensemble[2 * m];
    assert(Tbl != 0);
    for (i = 0; i < 2 * m; i++)
    {
        fsname[i] = new char[strlen(_t.fsname[i]) + 1];
        assert(fsname[i] != 0);
        strcpy(fsname[i], _t.fsname[i]);
        Tbl[i] = _t.Tbl[i];
    }

    icont = new int[m];
    assert(icont != 0);
    memcpy(icont, _t.icont, (size_t)(sizeof(int) * m));

    return *this;
}

//============================================================================
// Group
//============================================================================

// Constructors and Member-Functions of the class Group

////////////////////////////////// Group //////////////////////////////////
Group::Group(const Group &_grp)
    : n(_grp.n)
{
    name = new char[strlen(_grp.name) + 1];
    assert(name != 0);
    strcpy(name, _grp.name);

    element = new char *[n];
    for (int i = 0; i < n; i++)
    {
        element[i] = new char[strlen(_grp.element[i]) + 1];
        assert(element[i] != 0);
        strcpy(element[i], _grp.element[i]);
    }
}

Group::Group(char *nme, int num, char **el)
    : n(num)
{

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    element = new char *[n];
    for (int i = 0; i < n; i++)
    {
        element[i] = new char[strlen(el[i]) + 1];
        assert(element[i] != 0);
        strcpy(element[i], el[i]);
    }
}

Group::~Group()
{
    delete[] name;
    for (int i = 0; i < n; i++)
        delete[] element[i];
    delete[] element;
}

Group &Group::operator=(const Group &_grp)
{

    if (this == &_grp)
        return *this;

    delete[] name;
    //variables must be declared outside 'for'
    //since otheriwse under Linux we get
    //name lookup of `i' changed for new ANSI `for' scoping
    int i;
    for (i = 0; i < n; i++)
        delete[] element[i];
    delete[] element;

    name = new char[strlen(_grp.name) + 1];
    assert(name != 0);
    strcpy(name, _grp.name);

    n = _grp.n;
    element = new char *[n];
    for (i = 0; i < n; i++)
    {
        element[i] = new char[strlen(_grp.element[i]) + 1];
        assert(element[i] != 0);
        strcpy(element[i], _grp.element[i]);
    }

    return *this;
}

//============================================================================
// Set
//============================================================================

// Constructors and Member-Functions of the class Set

////////////////////////////////// Group //////////////////////////////////
Set::Set(const Set &_set)
{
    name = new char[strlen(_set.name) + 1];
    assert(name != 0);
    strcpy(name, _set.name);

    set_elements = _set.set_elements;
}

Set::Set(char *nme, list<string> elem_names)
{

    name = new char[strlen(nme) + 1];
    assert(name != 0);
    strcpy(name, nme);

    set_elements = elem_names;
}

Set::~Set()
{
    delete[] name;
}

Set &Set::operator=(const Set &_set)
{

    if (this == &_set)
        return *this;

    delete[] name;

    name = new char[strlen(_set.name) + 1];
    assert(name != 0);
    strcpy(name, _set.name);

    set_elements = _set.set_elements;

    return *this;
}

//---------------------------------------------------------------------------
// non-member help functions

// binomial coefficient a over b
REAL BiCo(int a, int b)
{
    REAL result = 1;
    int f;

    if (a < b)
        return 0;

    for (f = a; f > b; f--)
        result *= f;
    for (f = a - b; f > 1; f--)
        result /= f;

    return result;
}

Matrix BezierToPowerMatrix(int p)
{
    // Compute the pth degree Bezier matrix
    // This matrix transforms Bezier form to power basis form
    // Input:
    //         p: degree of surface patch in either u or v direction
    //
    // Output:
    //         m: Bezier matrix

    int i, k, pk, j;
    REAL sign;
    REAL kf;

    Matrix m(p + 1); // (p+1)x(p+1) Matrix set to zero

    m[0][0] = m[p][p] = 1.0; // Set corner elements

    if (p % 2 != 0)
        m[p][0] = -1.0;
    else
        m[p][0] = 1.0;

    sign = -1.0;
    // Compute first column, last row, and the diagonal
    for (i = 1; i < p; i++)
    {
        m[i][i] = BiCo(p, i);
        m[i][0] = m[p][p - i] = sign * m[i][i];
        sign *= -1.0;
    }
    // Compute remaining elements
    kf = (p + 1) / 2;
    pk = p - 1;
    for (k = 1; k < kf; k++)
    {
        sign = -1;
        for (j = k + 1; j <= pk; j++)
        {
            m[j][k] = m[pk][p - j] = sign * BiCo(p, k) * BiCo(p - k, j - k);
            sign *= -1.0;
        }
        pk--;
    }

    return m;
}

Matrix PowerToBezierMatrix(int p)
{
    // Compute inverse of pth-degree Bezier matrix
    // This matrix transforms power basis form to Bezier form
    // Input:
    //           p: degree of surface patch in either u or v direction
    //
    // Output:
    //         m_i: inverse of Bezier matrix

    int i, j, k;
    int pk;
    REAL kf;
    REAL sum;

    Matrix m_i(p + 1); // (p+1)x(p+1) Matrix set to zero

    // Compute the pth-degree Bezier matrix
    Matrix m = BezierToPowerMatrix(p);

    // Set first column, last row, and diagonal
    for (i = 0; i <= p; i++)
    {
        m_i[i][0] = m_i[p][i] = 1.0;
        m_i[i][i] = 1.0 / m[i][i];
    }
    // Compute remaining elements
    kf = (p + 1) / 2;
    pk = p - 1;

    for (k = 1; k < kf; k++)
    {
        for (j = k + 1; j <= pk; j++)
        {
            sum = 0.0;
            for (i = k; i < j; i++)
                sum = sum - m[j][i] * m_i[i][k];
            m_i[j][k] = sum / m[j][j];
            m_i[pk][p - j] = m_i[j][k];
        }
        pk--;
    }

    return m_i;
}

FLAG Intersect3DLines(const Vec3d &p0, const Vec3d &t0,
                      const Vec3d &p2, const Vec3d &t2, Vec3d &p1)
{
    // evaluate intersection of two lines in three-dimensional space lying
    // in a plane given by the point/tangent pairs
    // Output:
    //             p1: point of intersection

    Vec3d tmp;

    // test if the lines are parallel
    tmp.VecProd(t0, t2);
    if (tmp.length() < EPS)
    {
        return 1; // "1" means parallel
    }
    else
    {
        int i;
        int pos;
        REAL det_subm[3]; // submatrices

        Vec2d par; // parameter tupel
        Vec2d *vec = new Vec2d[3]; // vectors according to submatrices
        assert(vec != 0);

        Matrix_2x2 invers;
        Matrix_2x2 *subm = new Matrix_2x2[3];
        assert(subm != 0);

        // initialize the linear equation systems
        subm[0].set(Matrix_2x2(t0[0], -t2[0], t0[1], -t2[1]));
        subm[1].set(Matrix_2x2(t0[1], -t2[1], t0[2], -t2[2]));
        subm[2].set(Matrix_2x2(t0[0], -t2[0], t0[2], -t2[2]));

        vec[0].set(p2[0] - p0[0], p2[1] - p0[1]);
        vec[1].set(p2[1] - p0[1], p2[2] - p0[2]);
        vec[2].set(p2[0] - p0[0], p2[2] - p0[2]);

        // select the system with best conditioned matrix
        for (i = 0; i < 3; i++)
        {
            det_subm[i] = fabs(subm[i].det());
        }

        if (det_subm[0] >= det_subm[1])
        {
            if (det_subm[0] >= det_subm[2])
                pos = 0;
            else
                pos = 2;
        }
        else
        {
            if (det_subm[1] >= det_subm[2])
                pos = 1;
            else
                pos = 2;
        }

        // evaluate the intersection point
        invers.invert(subm[pos]);
        par = invers * vec[pos]; // 2-tupel of intersection
        p1 = p0 + par[0] * t0;

        delete[] subm;
        delete[] vec;

        return 0; //  lines not parallel
    }
}

//---------------------------------------------------------------------------
// Friend functions(operators) of classes
// Overloading of "<<" and ">>" (only for correct instantiating of LEDA lists)

ostream &operator<<(ostream &OS, const Circle &CIRCLE)
{
    OS << "<" << CIRCLE.name << "," << CIRCLE.radius << ","
       << CIRCLE.alpha << "," << CIRCLE.beta << ">";
    return OS;
}

istream &operator>>(istream &IS, Circle &CIRCLE)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> CIRCLE.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> CIRCLE.radius;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> CIRCLE.alpha;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> CIRCLE.beta;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Curve &CURVE)
{
    OS << "<" << CURVE.name << "," << CURVE.n_segments << ">";
    return OS;
}

istream &operator>>(istream &IS, Curve &CURVE)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> CURVE.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> CURVE.n_segments;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Surf &SURFACE)
{
    OS << "<" << SURFACE.name << "," << SURFACE.nps << "," << SURFACE.npt << ">";
    return OS;
}

istream &operator>>(istream &IS, Surf &SURFACE)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> SURFACE.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> SURFACE.nps;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> SURFACE.npt;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Cons &CONS)
{
    OS << "<" << CONS.name << "," << CONS.n_segments << ">";
    return OS;
}

istream &operator>>(istream &IS, Cons &CONS)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> CONS.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> CONS.n_segments;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Face &FAC)
{
    OS << "<" << FAC.name << "," << FAC.m << ">";
    return OS;
}

istream &operator>>(istream &IS, Face &FAC)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> FAC.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> FAC.m;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Top &TOP)
{
    OS << "<" << TOP.name << "," << TOP.m << ">";
    return OS;
}

istream &operator>>(istream &IS, Top &TOP)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> TOP.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> TOP.m;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Group &GROUP)
{
    OS << "<" << GROUP.name << "," << GROUP.n << ">";
    return OS;
}

istream &operator>>(istream &IS, Group &GROUP)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> GROUP.name;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> GROUP.n;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

ostream &operator<<(ostream &OS, const Set &SET)
{
    OS << "<" << SET.name << ">";
    return OS;
}

istream &operator>>(istream &IS, Set &SET)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> SET.name;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}
