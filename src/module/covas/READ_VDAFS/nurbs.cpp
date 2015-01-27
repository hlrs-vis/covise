/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Constructors and Member-Functions for NURBS data object   **
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

#include "nurbs.h"

inline int Max(int v1, int v2)
{
    // maximum of two values
    return (v1 >= v2 ? v1 : v2);
}

inline int Min(int v1, int v2)
{
    // minimum of two values
    return (v1 <= v2 ? v1 : v2);
}

//===========================================================================
// Knot Vector
//===========================================================================
Knot::Knot(int num, REAL *knots)
{
    n = num;

    knt = new REAL[n];
    assert(knt != 0);

    for (int i = 0; i < n; i++)
        set(i, knots[i]);
}

Knot::Knot(int num)
{
    n = num;

    knt = new REAL[n];
    assert(knt != 0);

    // initialize
    for (int i = 0; i < n; i++)
        set(i, 0.0);
}

Knot::Knot(const Knot &_k)
    : n(_k.n)
{
    knt = new REAL[n];
    assert(knt != 0);
    memcpy(knt, _k.knt, (size_t)(n * sizeof(REAL)));
}

Knot::~Knot()
{
    delete[] knt;
}

Knot &Knot::operator=(const Knot &_k)
{
    if (this == &_k)
        return *this;

    n = _k.n;
    delete[] knt;

    knt = new REAL[n];
    assert(knt != 0);
    memcpy(knt, _k.knt, (size_t)(n * sizeof(REAL)));

    return *this;
}

//===========================================================================
// Parameter Control Polygon
//===========================================================================
Pcpol::Pcpol(int num, Vec3d *cpoints)
{
    n = num;

    cpts = new Vec3d[n];
    assert(cpts != 0);

    for (int i = 0; i < n; i++)
        cpts[i] = cpoints[i];
}

Pcpol::Pcpol(int num)
{
    n = num;

    cpts = new Vec3d[n];
    assert(cpts != 0);

    // initialize
    for (int i = 0; i < n; i++)
    {
        cpts[i][0] = 0.0; // u
        cpts[i][1] = 0.0; // v
        cpts[i][2] = 1.0; // w
    }
}

Pcpol::Pcpol(const Pcpol &_p)
    : n(_p.n)
{
    cpts = new Vec3d[n];
    assert(cpts != 0);
    memcpy(cpts, _p.cpts, (size_t)(n * sizeof(Vec3d)));
}

Pcpol::~Pcpol()
{
    delete[] cpts;
}

void Pcpol::set(int i, Vec2d Cpt)
{
    cpts[i][0] = Cpt[0];
    cpts[i][1] = Cpt[1];
}

void Pcpol::set(int i, Vec3d Cpt)
{
    cpts[i] = Cpt;
}

Pcpol &Pcpol::operator=(const Pcpol &_p)
{
    if (this == &_p)
        return *this;

    n = _p.n;
    delete[] cpts;

    cpts = new Vec3d[n];
    assert(cpts != 0);
    memcpy(cpts, _p.cpts, (size_t)(n * sizeof(Vec3d)));

    return *this;
}

//===========================================================================
// Control Polygon
//===========================================================================
Cpol::Cpol(int num, Vec4d *cpoints)
{
    n = num;

    cpts = new Vec4d[n];
    assert(cpts != 0);

    for (int i = 0; i < n; i++)
        cpts[i] = cpoints[i];
}

Cpol::Cpol(int num)
{
    n = num;

    cpts = new Vec4d[n];
    assert(cpts != 0);

    // initialize
    for (int i = 0; i < n; i++)
    {
        cpts[i][0] = 0.0; // x
        cpts[i][1] = 0.0; // y
        cpts[i][2] = 0.0; // z
        cpts[i][3] = 1.0; // w
    }
}

Cpol::Cpol(const Cpol &_c)
    : n(_c.n)
{
    cpts = new Vec4d[n];
    assert(cpts != 0);
    memcpy(cpts, _c.cpts, (size_t)(n * sizeof(Vec4d)));
}

Cpol::~Cpol()
{
    delete[] cpts;
}

void Cpol::set(int i, Vec3d Cpt)
{
    cpts[i][0] = Cpt[0];
    cpts[i][1] = Cpt[1];
    cpts[i][2] = Cpt[2];
}

void Cpol::set(int i, Vec4d Cpt)
{
    cpts[i] = Cpt;
}

Cpol &Cpol::operator=(const Cpol &_c)
{
    if (this == &_c)
        return *this;

    n = _c.n;
    delete[] cpts;

    cpts = new Vec4d[n];
    assert(cpts != 0);
    memcpy(cpts, _c.cpts, (size_t)(n * sizeof(Vec4d)));

    return *this;
}

//===========================================================================
// Control Net
//===========================================================================
Cnet::Cnet(int numU, int numV, Vec4d **cpoints)
{
    int i, j;

    n = numU;
    m = numV;

    cpts = new Vec4d *[n];
    assert(cpts != 0);
    for (i = 0; i < n; i++)
    {
        cpts[i] = new Vec4d[m];
        assert(cpts[i] != 0);
    }

    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            cpts[i][j] = cpoints[i][j];
}

Cnet::Cnet(int numU, int numV)
    : n(numU)
    , m(numV)
{
    int i, j;
    Vec4d init(0.0, 0.0, 0.0, 1.0);

    cpts = new Vec4d *[n];
    assert(cpts != 0);
    for (i = 0; i < n; i++)
    {
        cpts[i] = new Vec4d[m];
        assert(cpts[i] != 0);
    }

    // initialize
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            cpts[i][j] = init;
}

Cnet::Cnet(const Cnet &_c)
    : n(_c.n)
    , m(_c.m)
{
    int i;

    cpts = new Vec4d *[n];
    assert(cpts != 0);
    for (i = 0; i < n; i++)
    {
        cpts[i] = new Vec4d[m];
        assert(cpts[i] != 0);
        memcpy(cpts[i], _c.cpts[i], (size_t)(m * sizeof(Vec4d)));
    }
}

Cnet::~Cnet()
{
    for (int i = 0; i < n; i++)
        delete[] cpts[i];
    delete[] cpts;
}

void Cnet::set(int row, int col, Vec3d Cpt)
{
    cpts[row][col][0] = Cpt[0];
    cpts[row][col][1] = Cpt[1];
    cpts[row][col][2] = Cpt[2];
}

void Cnet::set(int row, int col, Vec4d Cpt)
{
    cpts[row][col] = Cpt;
}

Cnet &Cnet::operator=(const Cnet &_c)
{
    if (this == &_c)
        return *this;

    int i;
    for (i = 0; i < n; i++)
        delete[] cpts[i];
    delete[] cpts;

    n = _c.n;
    m = _c.m;
    cpts = new Vec4d *[n];
    assert(cpts != 0);
    for (i = 0; i < n; i++)
    {
        cpts[i] = new Vec4d[m];
        assert(cpts[i] != 0);
        memcpy(cpts[i], _c.cpts[i], (size_t)(m * sizeof(Vec4d)));
    }

    return *this;
}

//===========================================================================
// Trim NURBS Curve
//===========================================================================
TrimCurve::TrimCurve(int n, int m)
    : n_knts(n)
    , n_cpts(m)
{
    knt = new Knot(n_knts);
    assert(knt != 0);

    pol = new Pcpol(n_cpts);
    assert(pol != 0);
}

TrimCurve::TrimCurve(const TrimCurve &trim)
    : n_knts(trim.n_knts)
    , n_cpts(trim.n_cpts)
{
    knt = new Knot;
    assert(knt != 0);
    *knt = *(trim.knt);

    pol = new Pcpol;
    assert(pol != 0);
    *pol = *(trim.pol);
}

TrimCurve::~TrimCurve()
{
    delete knt;
    delete pol;
}

void TrimCurve::set_knt(REAL *U)
{
    for (int i = 0; i < n_knts; i++)
        knt->set(i, U[i]);
}

void TrimCurve::set_pol(Vec2d *CPs)
{
    for (int i = 0; i < n_cpts; i++)
        pol->set(i, CPs[i]);
}

void TrimCurve::set_pol(Vec3d *CPs)
{
    for (int i = 0; i < n_cpts; i++)
        pol->set(i, CPs[i]);
}

void TrimCurve::BezDegreeReduce(Vec3d *bPts, Vec3d *rbPts, REAL &MaxError)
{
    // Degree reduce a Bezier curve in 2 dimensional parameter space
    // and compute the maximum error
    // Input:
    //         bPts[] : control points of the Bezier curve
    // Output:
    //         rbPts[]: degree reduced Bezier points
    //        MaxError: maximum error bound

    int i;
    int p; // degree
    int r;
    Vec3d PtL, PtR; // left and right control point

    // Initialize variables
    p = n_knts - n_cpts - 1; //  degree = order -1
    if (p == 0)
    {
        cerr << "Error: Incorrect degree of Bezier curve!" << endl;
        exit(-1);
    }
    r = (p - 1) / 2; // !! integer divison

    // degree reduction coefficients
    REAL *alpha = new REAL[p];
    for (i = 0; i <= p - 1; i++)
        alpha[i] = (float)(1.0 * i) / (1.0 * p);

    if (p % 2 == 0)
    { // p even
        rbPts[0] = bPts[0];
        rbPts[p - 1] = bPts[p];

        if (r > 0) // p>2
        {
            for (i = 1; i <= r; i++)
                rbPts[i] = (bPts[i] - alpha[i] * rbPts[i - 1]) / (1.0 - alpha[i]);
            for (i = p - 2; i >= r + 1; i--)
                rbPts[i] = (bPts[i + 1] - (1.0 - alpha[i + 1]) * rbPts[i + 1]) / alpha[i + 1];
        }

        MaxError = Distance3D(bPts[r + 1], 0.5 * (rbPts[r] + rbPts[r + 1]));
    }
    else
    { // p odd
        rbPts[0] = bPts[0];
        if (p > 1)
            rbPts[p - 1] = bPts[p];

        if (r > 0) // p>2
        {
            for (i = 1; i <= r - 1; i++)
                rbPts[i] = (bPts[i] - alpha[i] * rbPts[i - 1]) / (1.0 - alpha[i]);
            for (i = p - 2; i >= r + 1; i--)
                rbPts[i] = (bPts[i + 1] - (1.0 - alpha[i + 1]) * rbPts[i + 1]) / alpha[i + 1];

            // symmetrize rbPts[r]
            PtL = (bPts[r] - alpha[r] * rbPts[r - 1]) / (1.0 - alpha[r]);
            PtR = (bPts[r + 1] - (1.0 - alpha[r + 1]) * rbPts[r + 1]) / alpha[r + 1];
            rbPts[r] = 0.5 * (PtL + PtR);

            MaxError = Distance3D(PtL, PtR);
        }

        if (p == 1) // => r=0
        {
            // the line segment is represented by its start point
            // => maximum error = Length of line
            MaxError = Distance3D(bPts[0], bPts[1]);
        }
    }

    // free
    delete[] alpha;
}

FLAG TrimCurve::DegreeReduceTrimCurve()
{

    // Degree reduce a trimming NURBS curve from degree p to p-1
    // Output:
    //          rnurbs: degree reduced trimming NURBS curve

    int i, j, k;
    int count = 0;
    int ii;
    int p; // degree
    int ph;
    int nh; // nh+1: number of control points of degree reduced curve
    int m; //  m+1: number of knots
    int mh;
    int kind; // knot vector index
    int r = -1;
    int oldr;
    int lbz;
    int a;
    int b;
    int cind = 1; // control points index
    int mult; // multiplicity of knots
    int save;
    int s;
    int first;
    int last;
    int kj;
    int K, L;
    int q;

    REAL numer; // numerator of alpha
    REAL MaxErr;
    REAL alfa, beta;
    REAL delta;
    REAL Br;

    Vec3d A;

    Pcpol Qw = *pol;
    Knot U = *knt;

    // Control polygon of the degree-reduced trimming curve
    Pcpol Pw(n_cpts); // Notice: memory possibly oversized
    // Knot vector of the degree-reduced trimming curve
    Knot Uh(n_knts); // Notice: memory possibly oversized

    // Degree reduced trimming NURBS curve
    TrimCurve *rnurbs = NULL;

    // Initialize variables
    p = n_knts - n_cpts - 1; //  degree = order -1
    ph = p - 1;
    m = n_knts - 1;
    mh = ph;
    kind = ph + 1;
    a = p;
    b = p + 1;
    mult = p;
    Pw[0] = Qw[0];

    // Allocate

    // Bezier control points of the current segment
    Vec3d *bpts = new Vec3d[p + 1];
    // leftmost control points
    Vec3d *Nextbpts = new Vec3d[p - 1];
    // degree reduced Bezier control points
    Vec3d *rbpts = new Vec3d[p];
    // knot insertion alphas
    REAL *alphas = new REAL[p - 1];
    // error vector
    REAL *e = new REAL[m];

    // Compute left hand of knot vector
    for (i = 0; i <= ph; i++)
        Uh[i] = U[0];

    // Initialize first Bezier segment
    for (i = 0; i <= p; i++)
        bpts[i] = Qw[i];

    // Initialize error vector
    for (i = 0; i < m; i++)
        e[i] = 0.0;

    /* Loop through the Knot vector */
    while (b < m)
    {
        // First compute knot multiplicity
        i = b;
        while (b < m && U[b] == U[b + 1])
            b++;
        mult = b - i + 1;
        mh += mult - 1;
        oldr = r;
        r = p - mult;
        if (oldr > 0)
            lbz = (oldr + 2) / 2; // !! integer division
        else
            lbz = 1;

        // Insert knot knt[b] r times
        if (r > 0)
        {
            numer = U[b] - U[a];

            for (k = p; k >= mult; k--)
                alphas[k - mult - 1] = numer / (U[a + k] - U[a]);

            for (j = 1; j <= r; j++)
            {
                save = r - j;
                s = mult + j;
                for (k = p; k >= s; k--)
                    bpts[k] = alphas[k - s] * bpts[k] + (1.0 - alphas[k - s]) * bpts[k - 1];
                Nextbpts[save] = bpts[p];
            }
        }

        // Degree reduce Bezier segment
        BezDegreeReduce(bpts, rbpts, MaxErr);
        count++;
        // cout << endl << count << ". Maximum Error Bound: " << MaxErr << endl;
        /*
      if ( ErrFile )
        ErrFile << endl << count << ". Maximum Error Bound: " << MaxErr << endl;
      else cerr << "File 'error.out' could not be opened!" << endl;
      */

        e[a] += MaxErr;
        if (e[a] > TOL)
            return (1); // Curve not degree reducible

        // Remove knot knt[a] oldr times
        if (oldr > 0)
        {
            first = kind;
            last = kind;
            for (k = 0; k < oldr; k++)
            {
                i = first;
                j = last;
                kj = j - kind;

                while (j - i > k)
                {
                    alfa = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1]);
                    beta = (U[a] - Uh[j - k - 1]) / (U[b] - Uh[j - k - 1]);
                    Pw[i - 1] = (Pw[i - 1] - (1.0 - alfa) * Pw[i - 2]) / alfa;
                    rbpts[kj] = (rbpts[kj] - beta * rbpts[kj + 1]) / (1.0 - beta);
                    i++;
                    j--;
                    kj--;
                }

                // Compute knot removal error bounds (Br)
                if (j - i < k)
                    Br = Distance3D(Pw[i - 2], rbpts[kj + 1]);
                else
                {
                    delta = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1]);
                    A = delta * rbpts[kj + 1] + (1.0 - delta) * Pw[i - 2];
                    Br = Distance3D(Pw[i - 1], A);
                }

                // Update the error vector
                K = a + oldr - k;
                q = (2 * p - k + 1) / 2; // !! integer division
                L = K - q;
                for (ii = L; ii <= a; ii++)
                { // These knot spans were affected

                    e[ii] += Br;
                    if (e[ii] > TOL)
                        return (1); // Curve not degree reducible
                }

                first--;
                last++;
            } // End for (k=0; k<oldr; k++) loop

            cind = i - 1;
        } // End of (oldr > 0) loop

        if (a != p)
            for (i = 0; i < ph - oldr; i++)
            {
                Uh[kind] = U[a];
                kind++;
            }

        for (i = lbz; i <= ph; i++)
        {
            Pw[cind] = rbpts[i];
            cind++;
        }

        // Set up for the next pass through
        if (b < m)
        {
            for (i = 0; i < r; i++)
                bpts[i] = Nextbpts[i];
            for (i = r; i <= p; i++)
                bpts[i] = Qw[b - p + i];
            a = b;
            b++;
        }
        else
            for (i = 0; i <= ph; i++)
                Uh[kind + i] = U[b];
    } // End of while (b < m) loop

    nh = mh - ph - 1;

    // Set up the degree reduced NURBS curve
    rnurbs = new TrimCurve(mh + 1, nh + 1);
    rnurbs->set_knt(&(Uh[0]));
    rnurbs->set_pol(&(Pw[0]));
    // Overwrite the class object with the degree reduced one
    *this = *rnurbs;

    /*
      // output for debugging
      REAL u;
      Vec3d cpt;
      cout << endl << "Knot Vector: " << endl;
      cout << "(";
      for (i=0; i<mh; i++)
        {
          u = rnurbs->get_knot(i);
          cout << u << "," << "\t";
        }
   u = rnurbs->get_knot(mh);
   cout << u << ")" << endl;
   cout << endl << "Control Polygon: " << endl;
   for (i=0; i<=nh; i++)
   {
   cpt = rnurbs->get_controlPoint(i);
   cout << i << ": ";
   cpt.output();
   }
   */

    // free
    delete[] bpts;
    delete[] Nextbpts;
    delete[] rbpts;
    delete[] alphas;
    delete[] e;
    delete rnurbs;

    return (0);
}

/*
FLAG TrimCurve::MultipleDegreeReduce(const int t)
{
   // Degree reduce a trimming NURBS curve from degree p to p-t (if possible)
   // Input :
   //                 t:  number of  degree reduction
   // Output:
   //            rnurbs:  degree reduced trimming nurbs curve

   int i;
   FLAG error;

for (i=0; i<t; i++)
{
error = DegreeReduceTrimCurve();
if (error != 0) return (1);
}

return (0);
}
*/

int TrimCurve::FindSpan(const REAL u)
{
    // Determine the knot span index
    // Input:    knot u
    // Output:   the knot span index

    int m = n_knts - 1; // m+1: number of knots
    int p = n_knts - n_cpts - 1; // degree
    int n = m - p - 1; // n+1: number of control points
    int low, high;
    int mid;
    Knot U = *knt; // knot vector

    if (u == U[m])
        return (n + 1); // special case

    // Do binary search
    low = p;
    high = n + 1; // high = m - p
    mid = (low + high) / 2; // !! integer division
    while (u < U[mid] || u >= U[mid + 1])
    {
        if (u < U[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return (mid);
}

void TrimCurve::FindSpanMult(const REAL u, int &k, int &s)
{
    // Find knot span k in which u lies and multiplicity s of u
    // Input:  knot u
    // Output: knot span index  k (0 <= k <= m-p-1)
    //         multiplicity     s (0 <= s <= p+1)

    int i;

    k = FindSpan(u);

    // Compute knot multiplicity
    i = k;
    if (u == (*knt)[i])
    {
        // u is knot of U
        if (i == n_cpts)
        {
            // special case
            s = n_knts - n_cpts; // s=p+1
        }
        else
        {
            while (i > 0 && (*knt)[i] == (*knt)[i - 1])
                i--;
            s = k - i + 1;
        }
    }
    else
    {
        s = 0;
    }
}

void TrimCurve::RemoveCurveKnot(const int r, const int s, int num, int &t)
{
    // Remove knot u (index r) num times
    // Input:
    //         r: index of knot
    //         s: multiplicity
    // Output:
    //         t: actual number of times the knot is removed

    int i, j, k;
    int ii, jj;
    int n = n_cpts - 1; // n+1: number of control points
    int nh;
    int m = n_knts - 1; // m+1: number of knots
    int mh;
    int ord = n_knts - n_cpts; // order
    int p = ord - 1; // degree
    int fout = (2 * r - s - p) / 2; // First control point out (Note: integer division!)
    int last = r - s;
    int first = r - p; // Note: u must be an internal knot
    int off;

    FLAG remflag; // removal flag
    REAL u; // knot to be removed
    REAL alfi;
    REAL alfj;

    Pcpol Pw = *pol; // control points in parameter space
    Knot U = *knt; // knot vector

    // knot removed trimming NURBS curve
    TrimCurve *rnurbs;

    // Allocate
    // Local array of temporary control points (in homogeneous coordinates)
    Vec3d *temp = new Vec3d[2 * p + 1];

    u = U[r];

    for (t = 0; t < num; t++)
    {
        off = first - 1; // Difference in index between temp and P
        temp[0] = Pw[off];
        temp[last + 1 - off] = Pw[last + 1];

        i = first;
        j = last;
        ii = 1;
        jj = last - off;
        remflag = 0;

        while (j - i > t)
        {
            // Compute new control points for one removal step
            alfi = (u - U[i]) / (U[i + ord + t] - U[i]);
            alfj = (u - U[j - t]) / (U[j + ord] - U[j - t]);
            temp[ii] = (Pw[i] - (1.0 - alfi) * temp[ii - 1]) / alfi;
            temp[jj] = (Pw[j] - alfj * temp[jj + 1]) / (1.0 - alfj);
            i++;
            ii++;
            j--;
            jj--;
        } // End of while-loop

        if (j - i < t) // Check if knot rmovable
        {
            REAL dis = Distance3D(temp[ii - 1], temp[jj + 1]);
            //cout << "Deviation: " << dis << endl;
            if (dis <= remTOL)
                remflag = 1;
        }
        else
        {
            alfi = (u - U[i]) / (U[i + ord + t] - U[i]);
            REAL dis = Distance3D(Pw[i], alfi * temp[ii + t + 1] + (1.0 - alfi) * temp[ii - 1]);
            //cout << "Deviation: " << dis << endl;
            if (dis <= remTOL)
                remflag = 1;
        }
        if (remflag == 0) // Cannot remove any more knots
            break; // Get out of for-loop
        else
        {
            // Successful removal. Save new control points.
            i = first;
            j = last;

            while (j - i > t)
            {
                Pw[i] = temp[i - off];
                Pw[j] = temp[j - off];
                i++;
                j--;
            }
        }
        first--;
        last++;
    } // End of for-loop

    if (t == 0)
        return;
    for (k = r + 1; k <= m; k++)
        U[k - t] = U[k]; // Shift knots
    j = fout;
    i = j; // Pj thru Pi will be overwritten.
    for (k = 1; k < t; k++)
        if (k % 2 == 1) // k modulo 2
        {
            // odd
            i++;
        }
        else
        {
            // even
            j--;
        }

    for (k = i + 1; k <= n; k++) // Shift
    {
        Pw[j] = Pw[k];
        j++;
    }

    mh = m - t;
    nh = j - 1;

    // Set up the knot removed trimming NURBS curve
    rnurbs = new TrimCurve(mh + 1, nh + 1);
    rnurbs->set_knt(&(U[0]));
    rnurbs->set_pol(&(Pw[0]));
    // Overwrite the class object with the degree reduced
    *this = *rnurbs;

    /*
      // output for debugging
      Vec3d cpt;
      cout << endl << "Knot Vector: " << endl;
      cout << "(";
      for (i=0; i<mh; i++)
        {
          u = rnurbs->get_knot(i);
          cout << u << "," << "\t";
        }
      u = rnurbs->get_knot(mh);
   cout << u << ")" << endl;
   cout << endl << "Control Polygon: " << endl;
   for (i=0; i<=nh; i++)
   {
   cpt = rnurbs->get_controlPoint(i);
   cout << i << ": ";
   cpt.output();
   }
   */

    // free
    delete[] temp;
    delete rnurbs;

    return;
}

void TrimCurve::MaximumKnotRemoval()
{
    // tries to remove all internal knots (i.e. maximum knot removal)
    int i;
    int p = n_knts - n_cpts - 1; // degree
    int index; // knot span index
    int mult; // multiplicity
    int t;

    for (i = p + 1; i < n_cpts;)
    {
        FindSpanMult((*knt)[i], index, mult);
        RemoveCurveKnot(index, mult, mult, t);
        // Old class object is overwritten
        // cerr << endl << mult << "\t" << t << endl;
        i += mult - t;
    }
}

TrimCurve &TrimCurve::operator=(const TrimCurve &trim)
{
    if (this == &trim)
        return *this;

    delete knt;
    delete pol;

    n_knts = trim.n_knts;
    knt = new Knot;
    assert(knt != 0);
    *knt = *(trim.knt);

    n_cpts = trim.n_cpts;
    pol = new Pcpol;
    assert(pol != 0);
    *pol = *(trim.pol);

    return *this;
}

ostream &operator<<(ostream &OS, const TrimCurve &TRIM)
{
    OS << "<" << TRIM.n_knts << "," << TRIM.n_cpts << ">";
    return OS;
}

istream &operator>>(istream &IS, TrimCurve &TRIM)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> TRIM.n_knts;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> TRIM.n_cpts;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

//===========================================================================
// NURBS Curve
//===========================================================================
NurbsCurve::NurbsCurve(int n, int m)
    : n_knts(n)
    , n_cpts(m)
{
    knt = new Knot(n_knts);
    assert(knt != 0);

    pol = new Cpol(n_cpts);
    assert(pol != 0);
}

NurbsCurve::NurbsCurve(const NurbsCurve &nurbs)
    : n_knts(nurbs.n_knts)
    , n_cpts(nurbs.n_cpts)
{
    knt = new Knot;
    assert(knt != 0);
    *knt = *(nurbs.knt);

    pol = new Cpol;
    assert(pol != 0);
    *pol = *(nurbs.pol);
}

NurbsCurve::~NurbsCurve()
{
    delete knt;
    delete pol;
}

void NurbsCurve::set_knt(REAL *U)
{
    for (int i = 0; i < n_knts; i++)
        knt->set(i, U[i]);
}

void NurbsCurve::set_knt(Knot *U)
{
    if (n_knts == U->get_n())
    {
        *knt = *U;
    }
    else
    {
        cerr << "Error: Knot vector not compatible." << endl;
    }
}

void NurbsCurve::set_pol(Vec3d *CPs)
{
    for (int i = 0; i < n_cpts; i++)
        pol->set(i, CPs[i]);
}

void NurbsCurve::set_pol(Vec4d *CPs)
{
    for (int i = 0; i < n_cpts; i++)
        pol->set(i, CPs[i]);
}

void NurbsCurve::set_pol(Cpol &CP)
{
    int num = CP.get_n();
    if (n_cpts == num)
    {
        for (int i = 0; i < n_cpts; i++)
            pol->set(i, CP[i]);
    }
    else
    {
        cerr << "Error: Control polygon not compatible." << endl;
    }
}

void NurbsCurve::BezDegreeReduce(Vec4d *bPts, Vec4d *rbPts, REAL &MaxError)
{
    // Degree reduce a Bezier curve and compute the maximum error
    // Input:
    //         bPts[] : control points of the Bezier curve
    // Output:
    //         rbPts[]: degree reduced Bezier points
    //        MaxError: maximum error bound

    int i;
    int p; // degree
    int r;
    Vec4d PtL, PtR; // left and right control point

    // Initialize variables
    p = n_knts - n_cpts - 1; //  degree = order -1
    if (p == 0)
    {
        cerr << "Error: Incorrect degree of Bezier curve!" << endl;
        exit(-1);
    }
    r = (p - 1) / 2; // !! integer divison

    // degree reduction coefficients
    REAL *alpha = new REAL[p];
    for (i = 0; i <= p - 1; i++)
        alpha[i] = (float)(1.0 * i) / (1.0 * p);

    if (p % 2 == 0)
    { // p even
        rbPts[0] = bPts[0];
        rbPts[p - 1] = bPts[p];

        if (r > 0) // p>2
        {
            for (i = 1; i <= r; i++)
                rbPts[i] = (bPts[i] - alpha[i] * rbPts[i - 1]) / (1.0 - alpha[i]);
            for (i = p - 2; i >= r + 1; i--)
                rbPts[i] = (bPts[i + 1] - (1.0 - alpha[i + 1]) * rbPts[i + 1]) / alpha[i + 1];
        }

        MaxError = Distance4D(bPts[r + 1], 0.5 * (rbPts[r] + rbPts[r + 1]));
    }
    else
    { // p odd
        rbPts[0] = bPts[0];
        if (p > 1)
            rbPts[p - 1] = bPts[p];

        if (r > 0) // p>2
        {
            for (i = 1; i <= r - 1; i++)
                rbPts[i] = (bPts[i] - alpha[i] * rbPts[i - 1]) / (1.0 - alpha[i]);
            for (i = p - 2; i >= r + 1; i--)
                rbPts[i] = (bPts[i + 1] - (1.0 - alpha[i + 1]) * rbPts[i + 1]) / alpha[i + 1];

            // symmetrize rbPts[r]
            PtL = (bPts[r] - alpha[r] * rbPts[r - 1]) / (1.0 - alpha[r]);
            PtR = (bPts[r + 1] - (1.0 - alpha[r + 1]) * rbPts[r + 1]) / alpha[r + 1];
            rbPts[r] = 0.5 * (PtL + PtR);

            MaxError = Distance4D(PtL, PtR);
        }

        if (p == 1) // => r=0
        {
            // the line segment is represented by its start point
            // => maximum error = Length of line
            MaxError = Distance4D(bPts[0], bPts[1]);
        }
    }

    // free
    delete[] alpha;
}

FLAG NurbsCurve::DegreeReduceCurve()
{

    // Degree reduce a NURBS curve from degree p to p-1
    // Output:
    //          rnurbs: degree reduced NURBS curve

    int i, j, k;
    int ii;
    int p; // degree
    int ph;
    int nh; // nh+1: number of control points of degree reduced curve
    int m; //  m+1: number of knots
    int mh;
    int kind; // knot vector index
    int r = -1;
    int oldr;
    int lbz;
    int a;
    int b;
    int cind = 1; // control points index
    int mult; // multiplicity of knots
    int save;
    int s;
    int first;
    int last;
    int kj;
    int K, L;
    int q;

    REAL numer; // numerator of alpha
    REAL MaxErr;
    REAL alfa, beta;
    REAL delta;
    REAL Br;

    Vec4d A;

    Cpol Qw = *pol;
    Knot U = *knt;

    // Control polygon of the degree-reduced curve
    Cpol Pw(n_cpts); // Notice: memory possibly oversized
    // Knot vector of the degree-reduced curve
    Knot Uh(n_knts); // Notice: memory possibly oversized

    // degree reduced NURBS curve
    NurbsCurve *rnurbs = NULL;

    // Initialize variables
    p = n_knts - n_cpts - 1; //  degree = order -1
    ph = p - 1;
    m = n_knts - 1;
    mh = ph;
    kind = ph + 1;
    a = p;
    b = p + 1;
    mult = p;
    Pw[0] = Qw[0];

    // Allocate

    // Bezier control points of the current segment
    Vec4d *bpts = new Vec4d[p + 1];
    // leftmost control points
    Vec4d *Nextbpts = new Vec4d[p - 1];
    // degree reduced Bezier control points
    Vec4d *rbpts = new Vec4d[p];
    // knot insertion alphas
    REAL *alphas = new REAL[p - 1];
    // error vector
    REAL *e = new REAL[m];

    // Compute left hand of knot vector
    for (i = 0; i <= ph; i++)
        Uh[i] = U[0];

    // Initialize first Bezier segment
    for (i = 0; i <= p; i++)
        bpts[i] = Qw[i];

    // Initialize error vector
    for (i = 0; i < m; i++)
        e[i] = 0.0;

    /* Loop through the Knot vector */
    while (b < m)
    {
        // First compute knot multiplicity
        i = b;
        while (b < m && U[b] == U[b + 1])
            b++;
        mult = b - i + 1;
        mh += mult - 1;
        oldr = r;
        r = p - mult;
        if (oldr > 0)
            lbz = (oldr + 2) / 2; // !! integer division
        else
            lbz = 1;

        // Insert knot knt[b] r times
        if (r > 0)
        {
            numer = U[b] - U[a];

            for (k = p; k >= mult; k--)
                alphas[k - mult - 1] = numer / (U[a + k] - U[a]);

            for (j = 1; j <= r; j++)
            {
                save = r - j;
                s = mult + j;
                for (k = p; k >= s; k--)
                    bpts[k] = alphas[k - s] * bpts[k] + (1.0 - alphas[k - s]) * bpts[k - 1];
                Nextbpts[save] = bpts[p];
            }
        }

        // Degree reduce Bezier segment
        BezDegreeReduce(bpts, rbpts, MaxErr);
        cout << endl << "Maximum Error Bound: " << MaxErr << endl;
        e[a] += MaxErr;
        if (e[a] > TOL)
            return (1); // Curve not degree reducible

        // Remove knot knt[a] oldr times
        if (oldr > 0)
        {
            first = kind;
            last = kind;
            for (k = 0; k < oldr; k++)
            {
                i = first;
                j = last;
                kj = j - kind;

                while (j - i > k)
                {
                    alfa = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1]);
                    beta = (U[a] - Uh[j - k - 1]) / (U[b] - Uh[j - k - 1]);
                    Pw[i - 1] = (Pw[i - 1] - (1.0 - alfa) * Pw[i - 2]) / alfa;
                    rbpts[kj] = (rbpts[kj] - beta * rbpts[kj + 1]) / (1.0 - beta);
                    i++;
                    j--;
                    kj--;
                }

                // Compute knot removal error bounds (Br)
                if (j - i < k)
                    Br = Distance4D(Pw[i - 2], rbpts[kj + 1]);
                else
                {
                    delta = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1]);
                    A = delta * rbpts[kj + 1] + (1.0 - delta) * Pw[i - 2];
                    Br = Distance4D(Pw[i - 1], A);
                }

                // Update the error vector
                K = a + oldr - k;
                q = (2 * p - k + 1) / 2; // !! integer division
                L = K - q;
                for (ii = L; ii <= a; ii++)
                { // These knot spans were affected

                    e[ii] += Br;
                    if (e[ii] > TOL)
                        return (1); // Curve not degree reducible
                }

                first--;
                last++;
            } // End for (k=0; k<oldr; k++) loop

            cind = i - 1;
        } // End of (oldr > 0) loop

        if (a != p)
            for (i = 0; i < ph - oldr; i++)
            {
                Uh[kind] = U[a];
                kind++;
            }

        for (i = lbz; i <= ph; i++)
        {
            Pw[cind] = rbpts[i];
            cind++;
        }

        // Set up for the next pass through
        if (b < m)
        {
            for (i = 0; i < r; i++)
                bpts[i] = Nextbpts[i];
            for (i = r; i <= p; i++)
                bpts[i] = Qw[b - p + i];
            a = b;
            b++;
        }
        else
            for (i = 0; i <= ph; i++)
                Uh[kind + i] = U[b];
    } // End of while (b < m) loop

    nh = mh - ph - 1;

    // Set up the degree reduced NURBS curve
    rnurbs = new NurbsCurve(mh + 1, nh + 1);
    rnurbs->set_knt(&(Uh[0]));
    rnurbs->set_pol(&(Pw[0]));
    // Overwrite the class object with the degree reduced
    *this = *rnurbs;

    /*
      // output for debugging
      REAL u;
      Vec4d cpt;
      cout << endl << "Knot Vector: " << endl;
      cout << "(";
      for (i=0; i<mh; i++)
        {
          u = rnurbs->get_knot(i);
          cout << u << "," << "\t";
        }
   u = rnurbs->get_knot(mh);
   cout << u << ")" << endl;
   cout << endl << "Control Polygon: " << endl;
   for (i=0; i<=nh; i++)
   {
   cpt = rnurbs->get_controlPoint(i);
   cout << i << ": ";
   cpt.output();
   }
   */

    // free
    delete[] bpts;
    delete[] Nextbpts;
    delete[] rbpts;
    delete[] alphas;
    delete[] e;
    delete rnurbs;

    return (0);
}

int NurbsCurve::FindSpan(const REAL u)
{
    // Determine the knot span index
    // Input:    knot u
    // Output:   the knot span index

    int m = n_knts - 1; // m+1: number of knots
    int p = n_knts - n_cpts - 1; // degree
    int n = m - p - 1; // n+1: number of control points
    int low, high;
    int mid;
    Knot U = *knt; // knot vector

    if (u == U[m])
        return (n + 1); // special case

    // Do binary search
    low = p;
    high = n + 1; // high = m - p
    mid = (low + high) / 2; // !! integer division
    while (u < U[mid] || u >= U[mid + 1])
    {
        if (u < U[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return (mid);
}

void NurbsCurve::FindSpanMult(const REAL u, int &k, int &s)
{
    // Find knot span k in which u lies and multiplicity s of u
    // Input:  knot u
    // Output: knot span index  k (0 <= k <= m-p-1)
    //         multiplicity     s (0 <= s <= p+1)

    int i;

    k = FindSpan(u);

    // Compute knot multiplicity
    i = k;
    if (u == (*knt)[i])
    {
        // u is knot of U
        if (i == n_cpts)
        {
            // special case
            s = n_knts - n_cpts; // s=p+1
        }
        else
        {
            while (i > 0 && (*knt)[i] == (*knt)[i - 1])
                i--;
            s = k - i + 1;
        }
    }
    else
    {
        s = 0;
    }
}

void NurbsCurve::RemoveCurveKnot(const int r, const int s, int num, int &t)
{
    // Remove knot u (index r) num times
    // Input:
    //         r: index of knot
    //         s: multiplicity
    // Output:
    //         t: actual number of times the knot is removed

    int i, j, k;
    int ii, jj;
    int n = n_cpts - 1; // n+1: number of control points
    int nh;
    int m = n_knts - 1; // m+1: number of knots
    int mh;
    int ord = n_knts - n_cpts; // order
    int p = ord - 1; // degree
    int fout = (2 * r - s - p) / 2; // First control point out (Note: integer division!)
    int last = r - s;
    int first = r - p; // Note: u must be an internal knot
    int off;

    FLAG remflag; // removal flag
    REAL u; // knot to be removed
    REAL alfi;
    REAL alfj;

    Cpol Pw = *pol; // control points
    Knot U = *knt; // knot vector

    // knot removed NURBS curve
    NurbsCurve *rnurbs;

    // Allocate
    // Local array of temporary control points (in homogeneous coordinates)
    Vec4d *temp = new Vec4d[2 * p + 1];

    u = U[r];

    for (t = 0; t < num; t++)
    {
        off = first - 1; // Difference in index between temp and P
        temp[0] = Pw[off];
        temp[last + 1 - off] = Pw[last + 1];

        i = first;
        j = last;
        ii = 1;
        jj = last - off;
        remflag = 0;

        while (j - i > t)
        {
            // Compute new control points for one removal step
            alfi = (u - U[i]) / (U[i + ord + t] - U[i]);
            alfj = (u - U[j - t]) / (U[j + ord] - U[j - t]);
            temp[ii] = (Pw[i] - (1.0 - alfi) * temp[ii - 1]) / alfi;
            temp[jj] = (Pw[j] - alfj * temp[jj + 1]) / (1.0 - alfj);
            i++;
            ii++;
            j--;
            jj--;
        } // End of while-loop

        if (j - i < t) // Check if knot rmovable
        {
            REAL dis = Distance4D(temp[ii - 1], temp[jj + 1]);
            //cout << "Deviation: " << dis << endl;
            if (dis <= remTOL)
                remflag = 1;
        }
        else
        {
            alfi = (u - U[i]) / (U[i + ord + t] - U[i]);
            REAL dis = Distance4D(Pw[i], alfi * temp[ii + t + 1] + (1.0 - alfi) * temp[ii - 1]);
            //cout << "Deviation: " << dis << endl;
            if (dis <= remTOL)
                remflag = 1;
        }
        if (remflag == 0) // Cannot remove any more knots
            break; // Get out of for-loop
        else
        {
            // Successful removal. Save new control points.
            i = first;
            j = last;

            while (j - i > t)
            {
                Pw[i] = temp[i - off];
                Pw[j] = temp[j - off];
                i++;
                j--;
            }
        }
        first--;
        last++;
    } // End of for-loop

    if (t == 0)
        return;
    for (k = r + 1; k <= m; k++)
        U[k - t] = U[k]; // Shift knots
    j = fout;
    i = j; // Pj thru Pi will be overwritten.
    for (k = 1; k < t; k++)
        if (k % 2 == 1) // k modulo 2
        {
            // odd
            i++;
        }
        else
        {
            // even
            j--;
        }

    for (k = i + 1; k <= n; k++) // Shift
    {
        Pw[j] = Pw[k];
        j++;
    }

    mh = m - t;
    nh = j - 1;

    // Set up the knot removed NURBS curve
    rnurbs = new NurbsCurve(mh + 1, nh + 1);
    rnurbs->set_knt(&(U[0]));
    rnurbs->set_pol(&(Pw[0]));
    // Overwrite the class object with the degree reduced
    *this = *rnurbs;

    /*
      // output for debugging
      Vec4d cpt;
      cout << endl << "Knot Vector: " << endl;
      cout << "(";
      for (i=0; i<mh; i++)
        {
          u = rnurbs->get_knot(i);
          cout << u << "," << "\t";
        }
      u = rnurbs->get_knot(mh);
   cout << u << ")" << endl;
   cout << endl << "Control Polygon: " << endl;
   for (i=0; i<=nh; i++)
   {
   cpt = rnurbs->get_controlPoint(i);
   cout << i << ": ";
   cpt.output();
   }
   */

    // free
    delete[] temp;
    delete rnurbs;

    return;
}

void NurbsCurve::MaximumKnotRemoval()
{
    // tries to remove all internal knots (i.e. maximum knot removal)
    int i;
    int p = n_knts - n_cpts - 1; // degree
    int index; // knot span index
    int mult; // multiplicity
    int t;

    for (i = p + 1; i < n_cpts;)
    {
        FindSpanMult((*knt)[i], index, mult);
        RemoveCurveKnot(index, mult, mult, t);
        // Old class object is overwritten
        // cerr << endl << mult << "\t" << t << endl;
        i += mult - t;
    }
}

void NurbsCurve::output()
{
    cout << endl << "Knot: "
         << "(";
    int i;
    for (i = 0; i < n_knts; i++)
        cout << "\t" << (*knt)[i];
    cout << " )" << endl;

    Vec4d c;
    cout << "Control polygon: " << endl;
    for (i = 0; i < n_cpts; i++)
    {
        c = get_controlPoint(i);
        c.output();
    }
    cout << endl;
}

NurbsCurve &NurbsCurve::operator=(const NurbsCurve &nurbs)
{
    if (this == &nurbs)
        return *this;

    delete knt;
    delete pol;

    n_knts = nurbs.n_knts;
    knt = new Knot;
    assert(knt != 0);
    *knt = *(nurbs.knt);

    n_cpts = nurbs.n_cpts;
    pol = new Cpol;
    assert(pol != 0);
    *pol = *(nurbs.pol);

    return *this;
}

ostream &operator<<(ostream &OS, const NurbsCurve &NURBS)
{
    OS << "<" << NURBS.n_knts << "," << NURBS.n_cpts << ">";
    return OS;
}

istream &operator>>(istream &IS, NurbsCurve &NURBS)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> NURBS.n_knts;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> NURBS.n_cpts;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

//===========================================================================
// NURBS Surface
//===========================================================================
NurbsSurface::NurbsSurface(int r, int s, int n, int m)
    : n_Uknts(r)
    , n_Vknts(s)
    , Udim(n)
    , Vdim(m)
{
    Uknt = new Knot(n_Uknts);
    assert(Uknt != 0);

    Vknt = new Knot(n_Vknts);
    assert(Vknt != 0);

    net = new Cnet(Udim, Vdim);
    assert(net != 0);
}

NurbsSurface::NurbsSurface(const NurbsSurface &nurbs)
    : n_Uknts(nurbs.n_Uknts)
    , n_Vknts(nurbs.n_Vknts)
    , Udim(nurbs.Udim)
    , Vdim(nurbs.Vdim)
{
    Uknt = new Knot;
    assert(Uknt != 0);
    *Uknt = *(nurbs.Uknt);

    Vknt = new Knot;
    assert(Vknt != 0);
    *Vknt = *(nurbs.Vknt);

    net = new Cnet;
    assert(net != 0);
    *net = *(nurbs.net);
}

NurbsSurface::~NurbsSurface()
{
    delete Uknt;
    delete Vknt;
    delete net;
}

void NurbsSurface::set_Uknt(REAL *U)
{
    for (int i = 0; i < n_Uknts; i++)
        Uknt->set(i, U[i]);
}

void NurbsSurface::set_Uknt(Knot *U)
{
    if (n_Uknts == U->get_n())
    {
        *Uknt = *U;
    }
    else
    {
        cerr << "Error: u knot vector not compatible." << endl;
    }
}

void NurbsSurface::set_Vknt(REAL *V)
{
    for (int i = 0; i < n_Vknts; i++)
        Vknt->set(i, V[i]);
}

void NurbsSurface::set_Vknt(Knot *V)
{
    if (n_Vknts == V->get_n())
    {
        *Vknt = *V;
    }
    else
    {
        cerr << "Error: v knot vector not compatible." << endl;
    }
}

void NurbsSurface::set_net(Vec3d **CPs)
{
    for (int i = 0; i < Udim; i++)
        for (int j = 0; j < Vdim; j++)
            net->set(i, j, CPs[i][j]);
}

void NurbsSurface::set_net(Cnet &CN)
{
    if (Udim == CN.get_n() && Vdim == CN.get_m())
    {
        *net = CN;
    }
    else
    {
        cerr << "Error: Control net not compatible." << endl;
    }
}

int NurbsSurface::FindSpan(const int DIR, const REAL u)
{
    // Determine the knot span index
    // Input:    flag for direction DIR
    //           knot u
    // Output:   the knot span index

    int m; // m+1: number of knots in u (or v) direction
    int p; // degree in u (or v) direction
    int n; // n+1: number of control points in u (or v) direction
    int low, high;
    int mid;
    Knot U; // knot vector in u (or v) direction

    if (DIR == 0) // u direction
    {
        m = n_Uknts - 1;
        p = n_Uknts - Udim - 1;
        U = *Uknt;
    }
    else // v direction
    {
        m = n_Vknts - 1;
        p = n_Vknts - Vdim - 1;
        U = *Vknt;
    }
    n = m - p - 1;

    if (u == U[m])
        return (n + 1); // special case

    // Do binary search
    low = p;
    high = n + 1; // high = m - p
    mid = (low + high) / 2; // !! integer division
    while (u < U[mid] || u >= U[mid + 1])
    {
        if (u < U[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return (mid);
}

void NurbsSurface::FindSpanMult(const int DIR, const REAL u, int &k, int &s)
{
    // Find knot span k in which u lies and multiplicity s of u
    // Input:  flag for direction DIR
    //         knot u
    // Output: knot span index  k
    //         multiplicity     s

    int i;

    k = FindSpan(DIR, u);

    // Compute knot multiplicity
    i = k;
    if (DIR == 0) // u direction
    {
        if (u == (*Uknt)[i])
        {
            // u is knot of U
            if (i == Udim)
            {
                // special case
                s = n_Uknts - Udim; // s=p+1
            }
            else
            {
                while (i > 0 && (*Uknt)[i] == (*Uknt)[i - 1])
                    i--;
                s = k - i + 1;
            }
        }
        else
        {
            s = 0;
        }
    }
    else // v direction
    {
        if (u == (*Vknt)[i])
        {
            // u is knot of V
            if (i == Vdim)
            {
                // special case
                s = n_Vknts - Vdim; // s=q+1
            }
            else
            {
                while (i > 0 && (*Vknt)[i] == (*Vknt)[i - 1])
                    i--;
                s = k - i + 1;
            }
        }
        else
        {
            s = 0;
        }
    }
}

void NurbsSurface::RemoveKnot(const int DIR, const int l, const int mult, int num, int &t)
{
    // Remove the u- (or v-) knot w (index l) num times
    // Input:
    //          DIR: flag for direction
    //            l: index of knot
    //         mult: multiplicity
    // Output:
    //         t: actual number of times the knot is removed

    int i, j, k;
    int actual; // actual number of times the knot is removed
    // from a column (or row) curve

    REAL *W; // knot vector
    Cpol *Pw; // control points
    NurbsCurve **curves; // array of column (or row) curves
    Cnet *CP; // control net
    NurbsSurface *rnurbs; // knot removed NURBS surface

    t = num;
    if (DIR == 0) // u direction
    {
        curves = new NurbsCurve *[Vdim];
        Pw = new Cpol(Udim);
        for (i = 0; i < Vdim; i++)
        {
            for (j = 0; j < Udim; j++)
            {
                Pw->set(j, get_controlPoint(j, i));
            }
            // column curves
            curves[i] = new NurbsCurve(n_Uknts, Udim);
            curves[i]->set_pol(*Pw);
            curves[i]->set_knt(Uknt);
            // !! column curves may be overwritten
            // column curves may be overwritten
            curves[i]->RemoveCurveKnot(l, mult, num, actual);
            t = Min(actual, t);
            delete curves[i];
        }
    }
    else if (DIR == 1) // v direction
    {
        curves = new NurbsCurve *[Udim];
        Pw = new Cpol(Vdim);
        for (i = 0; i < Udim; i++)
        {
            for (j = 0; j < Vdim; j++)
            {
                Pw->set(j, get_controlPoint(i, j));
            }
            // row curves
            curves[i] = new NurbsCurve(n_Vknts, Vdim);
            curves[i]->set_pol(*Pw);
            curves[i]->set_knt(Vknt);
            //curves[i]->output();
            // row curves may be overwritten
            curves[i]->RemoveCurveKnot(l, mult, num, actual);
            //curves[i]->output();
            t = Min(actual, t);
            delete curves[i];
        }
    }

    if (t == 0)
        return;

    if (DIR == 0) // u direction
    {
        for (i = 0; i < Vdim; i++)
        {
            for (j = 0; j < Udim; j++)
            {
                Pw->set(j, get_controlPoint(j, i));
            }
            // column curves
            curves[i] = new NurbsCurve(n_Uknts, Udim);
            curves[i]->set_pol(*Pw);
            curves[i]->set_knt(Uknt);
            // column curves are overwritten
            curves[i]->RemoveCurveKnot(l, mult, t, actual);
            if (actual != t)
                cerr << "Error in knot removal of column curves." << endl;
        }
        CP = new Cnet(Udim - t, Vdim);
        for (i = 0; i < Udim - t; i++)
        {
            for (j = 0; j < Vdim; j++)
            {
                CP->set(i, j, curves[j]->get_controlPoint(i));
            }
        }
        // Set up the knot removed NURBS surface
        // Knot vector and number of control points identical for all column curves
        // Choose always the 1st curve
        int n_Ukntsh = curves[0]->get_n_knts();
        rnurbs = new NurbsSurface(n_Ukntsh, n_Vknts, curves[0]->get_n_cpts(), Vdim);
        W = new REAL[n_Ukntsh];
        for (k = 0; k < n_Ukntsh; k++)
            W[k] = curves[0]->get_knot(k);
        rnurbs->set_Uknt(W);
        rnurbs->set_Vknt(Vknt);
        rnurbs->set_net(*CP);

        for (i = 0; i < Vdim; i++)
            delete curves[i];
    }
    else if (DIR == 1) // v direction
    {
        for (i = 0; i < Udim; i++)
        {
            for (j = 0; j < Vdim; j++)
            {
                Pw->set(j, get_controlPoint(i, j));
            }
            // row curves
            curves[i] = new NurbsCurve(n_Vknts, Vdim);
            curves[i]->set_pol(*Pw);
            curves[i]->set_knt(Vknt);
            //curves[i]->output();
            // row curves are overwritten
            curves[i]->RemoveCurveKnot(l, mult, t, actual);
            if (actual != t)
                cerr << "Error in knot removal of row curves." << endl;
        }
        CP = new Cnet(Udim, Vdim - t);
        for (i = 0; i < Udim; i++)
        {
            for (j = 0; j < Vdim - t; j++)
            {
                CP->set(i, j, curves[i]->get_controlPoint(j));
            }
        }
        // Set up the knot removed NURBS surface
        // Knot vector and number of control points identical for all row curves
        // Choose always the 1st curve
        int n_Vkntsh = curves[0]->get_n_knts();
        W = new REAL[n_Vkntsh];
        for (k = 0; k < n_Vkntsh; k++)
            W[k] = curves[0]->get_knot(k);
        rnurbs = new NurbsSurface(n_Uknts, n_Vkntsh, Udim, curves[0]->get_n_cpts());
        rnurbs->set_Uknt(Uknt);
        rnurbs->set_Vknt(W);
        rnurbs->set_net(*CP);

        for (i = 0; i < Udim; i++)
            delete curves[i];
    }

    // Overwrite the class object with the degree reduced
    *this = *rnurbs;

    /*
   // output for debugging
   Vec4d cpt;
   REAL w;
   int r, s;
   int n, m;

   r = rnurbs -> get_n_Uknts();
   s = rnurbs -> get_n_Vknts();
   n = rnurbs -> get_Udim();
   m = rnurbs -> get_Vdim();

   cout << endl << "u knot: " << endl;
   cout << "(";
   for (i=0; i<r-1; i++)
   {
   w = rnurbs->get_Uknot(i);
   cout << w << "," << "\t";
   }
   w = rnurbs->get_Uknot(r-1);
   cout << w << ")" << endl;

   cout << endl << "v knot: " << endl;
   cout << "(";
   for (i=0; i<s-1; i++)
   {
   w = rnurbs->get_Vknot(i);
   cout << w << "," << "\t";
   }
   w = rnurbs->get_Vknot(s-1);
   cout << w << ")" << endl;

   cout << endl << "Control Net CP[n][m]: " << endl;
   for (i=0; i<n; i++)
   for (j=0; j<m; j++)
   {
   cout << i << "," << j << ": ";
   cpt = rnurbs->get_controlPoint(i,j);
   cpt.output();
   }
   */

    // free
    delete Pw;
    delete[] curves;
    delete[] W;
    delete CP;
    delete rnurbs;

    return;
}

void NurbsSurface::MaximumKnotRemoval()
{
    // tries to remove all internal knots in u- and v-direction (i.e. maximum knot removal)
    int i;
    int p = n_Uknts - Udim - 1; // degree in u direction
    int q = n_Vknts - Vdim - 1; // degree in v direction
    int index; // knot span index
    int mult; // multiplicity
    int t;

    // u direction
    for (i = p + 1; i < Udim;)
    {
        FindSpanMult(U_DIR, (*Uknt)[i], index, mult);
        RemoveKnot(U_DIR, index, mult, mult, t);
        // Old class object is overwritten
        //cerr << endl << "u direction: " << mult << "\t" << t << endl;
        i += mult - t;
    }

    // v direction
    for (i = q + 1; i < Vdim;)
    {
        FindSpanMult(V_DIR, (*Vknt)[i], index, mult);
        RemoveKnot(V_DIR, index, mult, mult, t);
        // Old class object is overwritten
        //cerr << endl << "v direction: " << mult << "\t" << t << endl;
        i += mult - t;
    }
}

NurbsSurface &NurbsSurface::operator=(const NurbsSurface &nurbs)
{
    if (this == &nurbs)
        return *this;

    delete Uknt;
    delete Vknt;
    delete net;

    n_Uknts = nurbs.n_Uknts;
    Uknt = new Knot;
    assert(Uknt != 0);
    *Uknt = *(nurbs.Uknt);

    n_Vknts = nurbs.n_Vknts;
    Vknt = new Knot;
    assert(Vknt != 0);
    *Vknt = *(nurbs.Vknt);

    Udim = nurbs.Udim;
    Vdim = nurbs.Vdim;
    net = new Cnet;
    assert(net != 0);
    *net = *(nurbs.net);

    return *this;
}

ostream &operator<<(ostream &OS, const NurbsSurface &NURBS)
{
    OS << "<" << NURBS.n_Uknts << "," << NURBS.n_Vknts << ","
       << NURBS.Udim << "," << NURBS.Vdim << ">";
    return OS;
}

istream &operator>>(istream &IS, NurbsSurface &NURBS)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> NURBS.n_Uknts;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> NURBS.n_Vknts;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> NURBS.Udim;

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> NURBS.Vdim;

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

// help functions
REAL Distance3D(const Vec3d &v1, const Vec3d &v2)
{
    // Distance between control points with homogeneous coordinates in parameter space

    Vec3d v = v1 - v2;
    return (v.length());
}

REAL Distance4D(const Vec4d &v1, const Vec4d &v2)
{
    // Distance between control points with homogeneous coordinates in object space

    Vec4d v = v1 - v2;
    return (v.length());
}
