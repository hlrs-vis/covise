/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "linalg.h"
#include <cassert>
#include <iomanip>
//---------------------------------------------------------------------------

#include "ext/gmm/gmm_solver_cg.h"
#include "ext/gmm/gmm_precond_ildlt.h"
#include "ext/gmm/gmm_inoutput.h"
//#include "ext/gmm/gmm_util.hpp"

typedef gmm::rsvector<double> sparse_vector;
typedef gmm::row_matrix<sparse_vector> sparse_matrix;
typedef std::vector<double> dense_vector;

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

class MyVector
{
public:
    MyVector()
    {
        comp = 0;
        sizeOfArray = 0;
    }
    MyVector(vec3 vector)
    {
        sizeOfArray = 3;
        comp = new double[sizeOfArray];
        memcpy(comp, vector, sizeOfArray * sizeof(double));
    }
    MyVector(unsigned int s)
    {
        comp = new double[s];
        sizeOfArray = s;
        memset(comp, 0, s * sizeof(double));
    }
    MyVector(double x, double y)
    {
        sizeOfArray = 2;
        comp = new double[2];
        comp[0] = x;
        comp[1] = y;
    }

    MyVector(double x, double y, double z)
    {
        sizeOfArray = 3;
        comp = new double[3];
        comp[0] = x;
        comp[1] = y;
        comp[2] = z;
    }
    MyVector(const MyVector &a)
    {
        sizeOfArray = a.sizeOfArray;
        comp = new double[sizeOfArray];
        memcpy(comp, a.comp, sizeOfArray * sizeof(double));
    }

    virtual ~MyVector()
    {
        if (comp)
            delete[] comp;
    }

    inline void operator*=(double d)
    {
        for (unsigned int i = 0; i < sizeOfArray; i++)
            comp[i] *= d;
    }

    inline MyVector operator*(double d) const
    {
        MyVector result(*this);
        result *= d;
        return result;
    }

    inline double operator*(const MyVector &v) const
    {
        unsigned int i;
        double result = 0.0;

        for (i = 0; i < sizeOfArray; i++)
            result += comp[i] * v.comp[i];

        return result;
    }

    friend inline MyVector operator*(double d, const MyVector &a)
    {
        MyVector result(a.sizeOfArray);
        for (unsigned int i = 0; i < a.sizeOfArray; i++)
            result.comp[i] = a.comp[i] * d;
        return result;
    }
    inline MyVector operator-(MyVector a) const
    {
#ifndef NODEBUG
        assert(a.sizeOfArray == sizeOfArray);
#endif

        MyVector result(a.sizeOfArray);
        for (unsigned int i = 0; i < a.sizeOfArray; i++)
            result.comp[i] = comp[i] - a.comp[i];
        return result;
    }

    inline void operator-=(const MyVector &a)
    {
#ifndef NODEBUG
        assert(a.sizeOfArray == sizeOfArray);
#endif
        for (unsigned int i = 0; i < sizeOfArray; i++)
            comp[i] -= a.comp[i];
    }

    inline void operator+=(const MyVector &a)
    {
#ifndef NODEBUG
        assert(a.sizeOfArray == sizeOfArray);
#endif
        // Add...
        for (unsigned int i = 0; i < sizeOfArray; i++)
            comp[i] += a.comp[i];
    }
    inline MyVector &operator=(const MyVector &a)
    {
        if (&a == this)
            return *this;
        if (a.sizeOfArray != sizeOfArray)
        {
            sizeOfArray = a.sizeOfArray;
            delete[] comp;
            comp = new double[sizeOfArray];
        }
        memcpy(comp, a.comp, sizeOfArray * sizeof(double));
        return *this;
    }

    inline double norm() const
    {
        unsigned int i;
        double normSquared = 0.0;

        for (i = 0; i < sizeOfArray; i++)
            normSquared += comp[i] * comp[i];

        return sqrt(normSquared);
    }

    friend inline double crossProduct2D(const MyVector &a, const MyVector &b)
    {
        return (a.comp[0] * b.comp[1] - a.comp[1] * b.comp[0]);
    }

    friend inline MyVector crossProduct(const MyVector &a, const MyVector &b)
    {
#ifndef NODEBUG
        assert(!((a.sizeOfArray != 3) || (b.sizeOfArray != 3)));
#endif

        double x, y, z;
        x = a.comp[1] * b.comp[2] - a.comp[2] * b.comp[1];
        y = a.comp[2] * b.comp[0] - a.comp[0] * b.comp[2];
        z = a.comp[0] * b.comp[1] - a.comp[1] * b.comp[0];
        return MyVector(x, y, z);
    }

    inline double &operator[](unsigned int i)
    {
#ifndef NODEBUG
        assert(i < sizeOfArray);
#endif
        return comp[i];
    }
    inline const double &operator[](unsigned int i) const
    {
#ifndef NODEBUG
        assert(i < sizeOfArray);
#endif
        return comp[i];
    }

    inline unsigned int size(void) const
    {
        return sizeOfArray;
    }

    friend class MyMatrix;

private:
    /// pointer to memory holding the entries
    double *comp;
    /// the amount of currently allocated doubles
    unsigned int sizeOfArray;

    friend std::ostream &operator<<(std::ostream &os, const MyVector &a);
};
typedef MyVector MyPosition;

ostream &operator<<(ostream &os, const MyVector &a)
{
    unsigned int i;
    os << "[ ";
    for (i = 0; i < a.size(); i++)
    {
        os << setw(16) << setprecision(16) << a.comp[i];
        if (i < a.size() - 1)
            os << ",";
    }
    os << " ]";
    return os;
}

class MyMatrix
{
public:
    inline MyMatrix()
    {
        dimy = 0;
        dimx = 0;
        sizeOfArray = 0;
        comp = 0;
    }

    inline MyMatrix(const MyMatrix &m)
    {
        dimx = m.dimx;
        dimy = m.dimy;
        unsigned int s = dimx * dimy;

        comp = new double[s];
        sizeOfArray = s;
        for (unsigned int i = 0; i < s; i++)
            comp[i] = m.comp[i];
    }

    inline MyMatrix(unsigned int m, unsigned int n)
    {
        dimy = m;
        dimx = n;
        comp = new double[m * n];
        sizeOfArray = m * n;
        memset(comp, 0, sizeOfArray * sizeof(double));
    }

    inline MyMatrix(unsigned int m)
    {
        dimy = dimx = m;
        comp = new double[m * m];
        sizeOfArray = m * m;
        memset(comp, 0, sizeOfArray * sizeof(double));
    }
    virtual ~MyMatrix()
    {
        if (comp)
            delete[] comp;
    }

    inline unsigned int getDimensionX(void) const
    {
        return (dimx);
    }

    inline unsigned int getDimensionY(void) const // Number of row-vectors in matrix
    {
        return (dimy);
    }

    inline MyMatrix &operator=(const MyMatrix &m)
    {
        dimx = m.dimx;
        dimy = m.dimy;

        // both matrices match ? -> just copy elements
        if (m.sizeOfArray != sizeOfArray)
        {
            sizeOfArray = m.sizeOfArray;
            delete[] comp;
            comp = new double[sizeOfArray];
        }

        memcpy(comp, m.comp, sizeOfArray * sizeof(double));
        return *this;
    }

    inline MyMatrix operator*(const double &lamda) const
    {
        unsigned int i;
        MyMatrix result(*this); //copy of self

        for (i = 0; i < sizeOfArray; i++)
            result.comp[i] *= lamda; // multiply all entries by value...

        return result;
    }

    double invert(void)
    {
#ifndef NODEBUG
        assert(dimy == dimx);
#endif
        switch (dimx)
        {
        case 1:
        {
            assert(comp[0] != 0.);
            double denom = comp[0];
            comp[0] = 1. / denom;
            return denom;
        }
        case 2:
        {
            double denom, a11, a21, a12, a22;
            a11 = comp[0];
            a21 = comp[1];
            a12 = comp[2];
            a22 = comp[3];
            denom = (a11 * a22 - a12 * a21);
            assert(denom != 0.);

            double invdenom = 1.0 / denom;

            comp[0] = a22 * invdenom;
            comp[1] = -a12 * invdenom;
            comp[2] = -a21 * invdenom;
            comp[3] = a11 * invdenom;
            return denom;
        }
        case 3:
        {
            double denom, a11, a12, a13, a21, a22, a23, a31, a32, a33;
            a11 = comp[0];
            a21 = comp[1];
            a31 = comp[2];
            a12 = comp[3];
            a22 = comp[4];
            a32 = comp[5];
            a13 = comp[6];
            a23 = comp[7];
            a33 = comp[8];

            comp[0] = (a22 * a33 - a23 * a32);
            comp[1] = -(a21 * a33 - a23 * a31);
            comp[2] = (a21 * a32 - a22 * a31);

            denom = comp[0] * a11 + comp[1] * a12 + comp[2] * a13;

            assert(denom != 0.);

            double invdenom = 1.0 / denom;

            comp[0] *= invdenom;
            comp[1] *= invdenom;
            comp[2] *= invdenom;
            comp[3] = -(a12 * a33 - a13 * a32) * invdenom;
            comp[4] = (a11 * a33 - a13 * a31) * invdenom;
            comp[5] = -(a11 * a32 - a12 * a31) * invdenom;
            comp[6] = (a12 * a23 - a13 * a22) * invdenom;
            comp[7] = -(a11 * a23 - a13 * a21) * invdenom;
            comp[8] = (a11 * a22 - a12 * a21) * invdenom;
            return denom;
        }
        default:
        {
            MyMatrix B(0, 0);
            B.dimx = 0;
            B.dimy = dimy;
            // this call should invert this Matrix without needing too many overhead
            gaussJ(B);

            //no return of denominator:
            return 0;
        }
        }
    }

    MyMatrix &gaussJ(MyMatrix &b)
    // PAR: a[1..n][1..n] contains the matrix of the system to solve
    //	n   is the number of rows and columns in the system
    //	b[1..n][1..m] contains m right sides, which are solved simultan
    // PRE: no singular matrix. TESTED
    // POST: **a contains the inverse of the original matrix
    //       **b conatins the solutions for the m right sides
    // REMARK: the program was taken from "Numerical Recipes in C", p.39/40
    {
        int *indxc, *indxr, *ipiv;
        int i, icol(0), irow(0), j, k, l, ll, n;
        double big;
        double dum, pivinv, c;

#ifndef NODEBUG
        assert(dimx == dimy);
        assert(dimx == b.dimy);
#endif

        n = b.getDimensionX();

        indxc = new int[dimx];
        indxr = new int[dimx];
        ipiv = new int[dimx];

        // IPIV preloaden
        for (j = 0; j < (int)dimx; j++)
            ipiv[j] = 0;

        //
        for (i = 0; i < (int)dimx; i++)
        {
            big = 0.0;
            for (j = 0; j < (int)dimx; j++)
                if (ipiv[j] != 1)
                    for (k = 0; k < (int)dimx; k++)
                    {
                        if (ipiv[k] == 0)
                        {
                            if (fabs((*this)(j, k)) >= big)
                            {
                                big = fabs((*this)(j, k));
                                irow = j;
                                icol = k;
                            }
                        }
                    }
            ++(ipiv[icol]);
// pivot found, interchange rows if necessary
#undef SWAP
#define SWAP(a, b, c) \
    {                 \
        c = a;        \
        a = b;        \
        b = c;        \
    }

            if (irow != icol)
            {
                for (l = 0; l < (int)dimx; l++)
                    SWAP((*this)(irow, l), (*this)(icol, l), c);
                for (l = 0; l < n; l++)
                    SWAP(b(irow, l), b(icol, l), c);
            }

            indxr[i] = irow;
            indxc[i] = icol;

            pivinv = 1.0 / (*this)(icol, icol);
            (*this)(icol, icol) = 1.0;
            for (l = 0; l < (int)dimx; l++)
                (*this)(icol, l) *= pivinv;
            for (l = 0; l < n; l++)
                b(icol, l) *= pivinv;
            for (ll = 0; ll < (int)dimx; ll++)
                if (ll != icol)
                {
                    dum = (*this)(ll, icol);
                    (*this)(ll, icol) = 0.0;
                    for (l = 0; l < (int)dimx; l++)
                        (*this)(ll, l) -= (*this)(icol, l) * dum;
                    for (l = 0; l < n; l++)
                        b(ll, l) -= b(icol, l) * dum;
                }
        }

        // system solved, interchange pairs of columns back in original order :
        for (l = (int)dimx - 1; l >= 0; l--)
        {
            if (indxr[l] != indxc[l])
                for (k = 0; k < (int)dimx; k++)
                    SWAP((*this)(k, indxr[l]), (*this)(k, indxc[l]), c);
        }
        delete[] ipiv;
        delete[] indxr;
        delete[] indxc;
        return b;
    }
#undef SWAP

    inline double &operator()(unsigned int m, unsigned int n)
    {
#ifndef NODEBUG
        if ((m >= dimy) || (n >= dimx))
        {

            std::cout << "m = y =" << m << ", n = x =" << n << ", dimx = " << dimx
                      << ", dimy = " << dimy << std::endl;
            assert(0);
        }
#endif
        return comp[m * dimx + n];
    }

    inline const double &operator()(unsigned int m, unsigned int n) const
    {
#ifndef NODEBUG
        if ((m >= dimy) || (n >= dimx))
        {

            std::cout << "m = y =" << m << ", n = x =" << n << ", dimx = " << dimx
                      << ", dimy = " << dimy << std::endl;
            assert(0);
        }
#endif
        return comp[m * dimx + n];
    }

    MyMatrix &mult(const MyMatrix &m, MyMatrix &d) const
    {
        unsigned int i, j, k;
        double *src1, *src2, *dst;

#ifndef NODEBUG
        assert(m.dimy == dimx);
        assert(!((dimx == 0) || (dimy == 0) || (m.dimx == 0)));
#endif

        d.dimy = dimy;
        d.dimx = m.dimx;

        if (d.sizeOfArray != m.dimx * dimy)
        {
            // make the new array...
            d.sizeOfArray = dimy * m.dimx;
            d.comp = new double[d.sizeOfArray];
        }
        memset(d.comp, 0, d.sizeOfArray * sizeof(double));

        for (i = 0; i < dimy; i++)
            for (k = 0; k < m.dimx; k++)
            {
                dst = &d.comp[i * m.dimx + k];
                src1 = &comp[i * dimx];
                src2 = &m.comp[k];

                for (j = 0; j < dimx; j++)
                {
                    (*dst) += (*src1) * (*src2);
                    src1++;
                    src2 += m.dimx;
                }
            }
        return d;
    }

    inline MyMatrix &transpose(void)
    {
        double *tmp = new double[sizeOfArray];
        double *src, *dst;

        for (unsigned int i = 0; i < dimy; i++)
        {
            dst = &tmp[i];
            src = &comp[i * dimx];
            for (unsigned int j = 0; j < dimx; j++)
            {
                (*dst) = (*src);
                dst += dimy;
                src++;
            }
        }

        unsigned int t = dimx;
        dimx = dimy;
        dimy = t;

        delete[] comp;
        comp = tmp;

        return *this;
    }

private:
    /// pointer to memory holding the entries
    double *comp;
    /// the amount of currently allocated doubles
    unsigned int sizeOfArray;
    /// the dimensions of our matrix
    unsigned int dimx, dimy;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
namespace
{
inline MyVector normal(const std::vector<MyVector> &pos)
{
    MyVector normal;
    assert(pos.size() == 3);
    MyVector d1 = pos[1] - pos[0];
    MyVector d2 = pos[2] - pos[0];
    normal = crossProduct(d2, d1);
    return normal;
}

inline MyVector mean(const std::vector<MyVector> &t)
{
    if (t.empty())
        return MyVector();

    MyVector sum(t.front());

    for (unsigned int i = 1; i < t.size(); ++i)
        sum += t[i];

    return 1. / t.size() * sum;
}

inline double volume(std::vector<MyVector> pos)
{
    double bla;
    MyVector v0, v1, v2, vc;
    switch (pos.size())
    {
    case 2:
        return (pos[1] - pos[0]).norm();
        break;
    case 3:
        //       bla= 0.5*crossProduct2D( pos[1]-pos[0], pos[2]-pos[0] );
        assert(0);
        break;
    case 4:
    {
        v0 = pos[1] - pos[0];
        v1 = pos[2] - pos[0];
        v2 = pos[3] - pos[0];
        vc = crossProduct(v1, v2);
        bla = (v0 * vc) / 6;
    }
    break;
    default:
        assert(false);
        break;
    }
    return bla;
}
//---------------------------------------------------------------------------
void matout(const MyMatrix &G)
{
    for (unsigned int mi = 0; mi < G.getDimensionX(); ++mi)
        for (unsigned int mj = 0; mj < G.getDimensionY(); ++mj)
        {
            cout << " " << G(mj, mi);
        }
    cout << endl;
}
inline MyMatrix stiffness(const std::vector<MyVector> &pos)
{
    const unsigned int N = pos.size();

    assert(N == 3 || N == 4);

    MyMatrix G_(N), G(N, N - 1), H(N, N - 1);

    double vol = volume(pos);

    if (fabs(vol) < 1e-20)
    {
        std::cout << "zero volume cell\n";
        return MyMatrix(N);
    }

    // construct G and H
    for (unsigned int i = 0; i < N; ++i)
    {
        G_(0, i) = 1;

        for (unsigned int j = 1; j < N; ++j)
        {
            G_(j, i) = pos[i][j - 1];
        }
    }

    for (unsigned int i = 1; i < N; ++i)
        H(i, i - 1) = 1;

    G_.invert();
    G_.mult(H, G);
    for (unsigned int mi = 0; mi < G.getDimensionX(); ++mi)
        for (unsigned int mj = 0; mj < G.getDimensionY(); ++mj)
        {
            if (fabs(G(mj, mi)) < 1e-14)
                G(mj, mi) = 0;
        }

    MyMatrix Gtmp = MyMatrix(), G_T;
    G_T = G;
    G_T.transpose();
    G.mult(G_T, Gtmp);

    return Gtmp * vol;
}

inline void toLocalBasis(std::vector<MyVector> &pos,
                         std::vector<MyVector> &vec)
{
    MyVector e1 = pos[1] - pos[0];
    MyVector e2 = pos[2] - pos[0];

    e2 -= (e1 * e2) * e1;

    for (std::vector<MyVector>::iterator pi = pos.begin(); pi != pos.end(); ++pi)
        (*pi) = MyVector((*pi - pos[0]) * e1, (*pi - pos[0]) * e2);

    for (std::vector<MyVector>::iterator vi = vec.begin(); vi != vec.end(); ++vi)
    {
        (*vi) = MyVector(*vi * e1, *vi * e2);
    }
}

template <typename PC>
void singular_cg(const double &mu,
                 const sparse_matrix &A,
                 dense_vector &x,
                 const dense_vector &b,
                 const PC &P,
                 gmm::iteration &iter)
{
    const unsigned int N = gmm::vect_size(x);

    double rho, rho_1(0), a;
    dense_vector p(N), q(N), r(N), z(N), c(N, 1.0);

    iter.set_rhsnorm(gmm::vect_norm2(b));

    gmm::clear(x);
    gmm::copy(b, r);
    gmm::mult(P, r, z);

    rho = gmm::vect_sp(z, r);
    gmm::copy(z, p);

    while (!iter.finished_vect(r))
    {
        if (!iter.first())
        {
            gmm::mult(P, r, z);
            rho = gmm::vect_sp(z, r);
            gmm::add(z, gmm::scaled(p, rho / rho_1), p);
        }

        gmm::mult(A, p, q);
        gmm::add(q, gmm::scaled(c, mu * gmm::vect_sp(c, p)), q);

        a = rho / gmm::vect_sp(q, p);
        gmm::add(gmm::scaled(p, a), x);
        gmm::add(gmm::scaled(q, -a), r);

        rho_1 = rho;
        ++iter;
    }
}

unsigned int translateFtoA[] = { 3, 0, 1, 2 }; //translate Filip's to Alex's enumeration
unsigned int translateAtoF[] = { 1, 2, 3, 0 }; //translate Alex's to Filip's enumeration
int faceIds[4][3] = { { 0, 1, 2 }, { 0, 3, 1 }, { 1, 3, 2 }, { 2, 3, 0 } };
void detectBoundaryFaces(vector<bool> &isBoundaryFace, std::vector<int> *cellNeighs,
                         Unstructured *unst_in, const int &cellId)
{
    //bool debug=0;
    int *cellVertIndices = unst_in->getCellNodesAVS(cellId);
    isBoundaryFace.resize(4);
    for (unsigned int f = 0; f < 4; f++)
    {
        isBoundaryFace[f] = true;
        unsigned int same;
        for (unsigned int n = 0; n < cellNeighs->size(); n++)
        {
            int *neighVertIndices = unst_in->getCellNodesAVS((*cellNeighs)[n]);
            same = 0;
            for (unsigned int fv = 0; fv < 3; fv++)
            {
                for (unsigned int cv = 0; cv < 4; cv++)
                {
                    //This face cannot be on the boundary
                    //as neighbor cell has all vertices in common
                    //with it
                    if (neighVertIndices[cv] == cellVertIndices[translateAtoF[faceIds[f][fv]]])
                    {
                        same++;
                        break;
                    } //endif
                } //endfor cv
            } //endfor fv
            if (same == 3)
            {
                isBoundaryFace[f] = false;
                break;
            }
        } //endfor n
    } //endfor i
}
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void localized_flow_impl(UniSys *us,
                         Unstructured *unst_in, int compVelo,
                         Unstructured *unst_scalar,
                         double residual,
                         int maxIter,
                         char *quantity_name)
{
    us->info("Computing Localized Flow");
    us->moduleStatus("Computing Localized Flow", 25);

    // build the matrix A and the right hand side
    sparse_matrix A;
    dense_vector b, x;

    int nbPos = unst_in->nNodes;
    std::cout << "nbpos: " << nbPos << endl;

    A.resize(nbPos, nbPos);
    x.resize(nbPos, 0.0);
    b.resize(nbPos, 0.0);

    double hmin = std::numeric_limits<double>::max();

    std::vector<std::vector<std::vector<unsigned int> > > bdryList;

    us->info("Localized Flow -- computing matrix and right hand side ...");
    for (int c = 0; c < unst_in->nCells; ++c)
    {
        float progressPercentage = c * (1. / unst_in->nCells);
        us->moduleStatus("Localized Flow -- computing matrix and right hand side ...", (int)(progressPercentage * 100));

        int type = unst_in->getCellType(c);
        assert(type == Unstructured::CELL_TET);

        int *ind = unst_in->getCellNodesAVS(c);

        std::vector<MyPosition> positions(4);
        std::vector<MyVector> ten(4);
        for (unsigned int i = 0; i < 4; i++)
        {
            vec3 tmpVec;
            unst_in->getVector3(ind[i], tmpVec);
            ten[translateFtoA[i]] = MyVector(tmpVec);
            unst_in->getCoords(ind[i], tmpVec);
            positions[translateFtoA[i]] = MyVector(tmpVec);
        }

        double integral_multiplier = 1. / 6.;

        MyMatrix M = stiffness(positions);

        // fill in A
        for (unsigned int mi = 0; mi < M.getDimensionX(); ++mi)
            for (unsigned int mj = 0; mj < M.getDimensionY(); ++mj)
            {
                A(ind[translateAtoF[mj]], ind[translateAtoF[mi]]) += M(mi, mj);
            }

        // check if element is on the boundary
        std::vector<std::vector<unsigned int> > bdry;
        //unsigned int nb;

        std::vector<int> *cellNeighs = unst_in->getCellNeighbors(c);

        vector<bool> isBoundaryFace;
        detectBoundaryFaces(isBoundaryFace, cellNeighs, unst_in, c);

        for (unsigned int f = 0; f < 4; ++f)
        {
            if (isBoundaryFace[f])
            {
                bdry.push_back(std::vector<unsigned int>());
                bdry.back().push_back(faceIds[f][0]);
                bdry.back().push_back(faceIds[f][1]);
                bdry.back().push_back(faceIds[f][2]);
            }
        }
        bdryList.push_back(bdry);

        // finally, find shortest edge
        for (unsigned int p1 = 0; p1 < 3; ++p1)
            for (unsigned int p2 = p1 + 1; p2 < 4; ++p2)
                hmin = std::min(hmin, (positions[p1] - positions[p2]).norm());

        for (unsigned int i = 0; i < bdryList[c].size(); ++i)
        {
            std::vector<MyVector> p(bdryList[c][i].size());
            std::vector<MyVector> t(bdryList[c][i].size());
            for (unsigned int j = 0; j < bdryList[c][i].size(); ++j)
            {
                p[j] = positions[bdryList[c][i][j]];
                t[j] = ten[bdryList[c][i][j]];
            }

            double bi = (normal(p) * mean(t)) * integral_multiplier;

            for (unsigned int j = 0; j < bdryList[c][i].size(); ++j)
                b[ind[translateAtoF[bdryList[c][i][j]]]] += bi;
        }
    }

    us->info("Preparing iterative solving");
    std::cout << "norm of b:" << gmm::vect_norm2(b) << '\n';

    gmm::mult(A, b, x);
    std::cout << "compatibility of b:" << gmm::vect_norm2(x) << '\n';

    us->info("Computing preconditioner");
    gmm::ildlt_precond<sparse_matrix> pc(A);
    cout << "done" << endl;

    bool abortFlag = false;
    gmm::iteration iter(abortFlag, residual, 1, maxIter);

    us->info("Solving");
    std::fill(x.begin(), x.end(), 0);

    singular_cg(hmin * hmin, A, x, b, pc, iter);

    //   Unstructured *unst_potential = new Unstructured(unst_in, 1);
    //  for(unsigned int i=0; i<unst_in->getNodeCompNb();i++)
    //   {
    //     cout<<"Name "<<(unst_in->getNodeCompLabel(i))<<endl;
    //   }
    //   for(unsigned int i=0; i<unst_scalar->getNodeCompNb();i++)
    //   {
    //     cout<<"Name."<<(unst_scalar->getNodeCompLabel(i))<<endl;
    //   }
    //   for(unsigned int i=0; i<unst_potential->getNodeCompNb();i++)
    //   {
    //     cout<<"Name:"<<(unst_potential->getNodeCompLabel(i))<<endl;
    //   }

    //   Unstructured *unst_potential;
    //   (*unst_potential) = (*unst_scalar);

    for (unsigned int i = 0; i < x.size(); i++)
        unst_scalar->setScalar(i, x[i]);

    //   for( unsigned int nodeId = 0; nodeId < unst_in->nNodes; ++nodeId )
    //   {
    //     cout<<nodeId<<" HALLLLO5"<<endl;
    //     vec3 grad;
    //     unst_potential->getVector3( nodeId, 0, Unstructured::OP_GRADIENT, grad);
    //     cout<<nodeId<<" HALLLLO6"<<endl;
    //     vec3 tmpVec;
    //     unst_in->getVector3( nodeId, tmpVec );
    //     cout<<nodeId<<" HALLLLO7"<<endl;
    //     double dotProd = vec3dot( grad,  tmpVec);
    //     cout<<nodeId<<" HALLLLO8"<<endl;

    //     unst_scalar->setScalar( nodeId, dotProd );

    //   }
}
