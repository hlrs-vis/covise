/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          arrays.h  -  vector, matrix and tensor classes
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef _ARRAYS_H_

#define _ARRAYS_H_

#include <vector>
#include <new>
#ifndef __sgi
#include <cstdlib>
#include <cmath>
#endif
#include <complex>

#include "error.h"

using namespace std;

typedef double prec; // definition variable precision
typedef complex<prec> cmplx;

#if defined(WIN32)
template <class T>
inline const T &min(const T &a, const T &b)
{
    return b < a ? b : a;
}

template <class T>
inline const T &max(const T &a, const T &b)
{
    return a < b ? b : a;
}
#endif

template <class T>
T sqr(T t)
{
    return t * t;
}

// ***************************************************************************
// basis point class
// ***************************************************************************

template <class T>
class point_base
{
public:
    explicit point_base(T fx = 0, T fy = 0)
    {
        x = fx;
        y = fy;
    }
    point_base(const point_base<T> &ptsrc)
    {
        x = ptsrc.x;
        y = ptsrc.y;
    }

    void Set(T fx, T fy)
    {
        x = fx;
        y = fy;
    }
    void SetEmpty()
    {
        x = y = 0;
    }
    point_base<T> &operator=(const T f)
    {
        x = f;
        y = 0;
        return *this;
    }
    T &operator()(int i)
    {
        if (i < 0 || i > 1)
            throw TException("ERROR: index out of range");
        if (i == 0)
            return x;
        return y;
    }
    T operator()(int i) const
    {
        if (i < 0 || i > 1)
            throw TException("ERROR: index out of range");
        if (i == 0)
            return x;
        return y;
    }
    T &operator[](int i)
    {
        return (*this)(i);
    }
    T operator[](int i) const
    {
        return (*this)(i);
    }
    point_base<T> &operator+=(const point_base<T> &ptsrc)
    {
        x += ptsrc.x;
        y += ptsrc.y;
        return *this;
    }
    point_base<T> &operator-=(const point_base<T> &ptsrc)
    {
        x -= ptsrc.x;
        y -= ptsrc.y;
        return *this;
    }
    point_base<T> operator-()
    {
        point_base<T> pt;
        return pt -= *this;
    }
    point_base<T> &operator*=(const T t)
    {
        x *= t;
        y *= t;
        return *this;
    }
    T operator*(const point_base<T> &ptsrc)
    {
        return x * ptsrc.x + y * ptsrc.y;
    }
    point_base<T> &operator/=(T t)
    {
        x /= t;
        y /= t;
        return *this;
    }
    bool operator==(const point_base<T> &ptsrc)
    {
        return (x == ptsrc.x && y == ptsrc.y);
    }
    bool operator!=(const point_base<T> &ptsrc)
    {
        return !(*this == ptsrc);
    }
    T Norm()
    {
        return x * x + y * y;
    }
    T Abs()
    {
        return sqrt(Norm());
    }
    void Normalize()
    {
        *this /= Abs();
    }

    T x, y;
};

template <class T>
inline point_base<T> operator+(const point_base<T> &pt1,
                               const point_base<T> &pt2)
{
    point_base<T> p = pt1;
    return p += pt2;
}

template <class T>
inline point_base<T> operator-(const point_base<T> &pt1,
                               const point_base<T> &pt2)
{
    point_base<T> p = pt1;
    return p -= pt2;
}

template <class T>
inline point_base<T> operator*(const int src,
                               const point_base<T> &pt)
{
    point_base<T> p = pt;
    return p *= src;
}

template <class T>
inline point_base<T> operator*(const double src,
                               const point_base<T> &pt)
{
    point_base<T> p = pt;
    return p *= src;
}

template <class T>
inline point_base<T> operator*(const point_base<T> &pt, const T src)
{
    point_base<T> p = pt;
    return p *= src;
}

template <class T>
inline point_base<T> operator/(const point_base<T> &pt,
                               const T src)
{
    point_base<T> p = pt;
    return p /= src;
}

template <class T>
inline istream &operator>>(istream &ps,
                           point_base<T> &pt)
{
    prec cx, cy;
    char ch;

    if (ps >> ch && ch != '(') // leading character
        ps.putback(ch);
    ps >> cx;
    if ((ch = ps.get()) == '\t' || ch == ',')
        ps >> cy;
    else if ((int)ch != -1)
        ps.putback(ch);
    if (ps >> ch && ch != ')') // final character
        ps.putback(ch);
    if (!ps.fail())
    {
        pt.x = cx;
        pt.y = cy;
    }
    return ps;
}

template <class T>
inline ostream &operator<<(ostream &ps,
                           const point_base<T> &pt)
{
    return ps << pt.x << '\t' << pt.y;
}

typedef point_base<prec> TPoint;
typedef point_base<int> point;

/*****************************************************************************
 streamable point output base class
 *****************************************************************************/

template <class T>
class point_output
{
public:
    explicit point_output(const point_base<T> &src)
    {
        pt = src;
    }

    point_base<T> pt;
};

template <class T>
inline ostream &operator<<(ostream &ps,
                           const point_output<T> &base)
{
    return ps << '(' << base.pt.x << ',' << base.pt.y << ')';
}

typedef point_output<prec> pPT;
typedef point_output<int> iPT;

/*****************************************************************************
 3D point base class
 *****************************************************************************/

template <class T>
class point3D_base
{
public:
    explicit point3D_base(const T fx = 0, const T fy = 0, const T fz = 0)
    {
        x = fx;
        y = fy;
        z = fz;
    }
    point3D_base(const point3D_base<T> &ptsrc)
    {
        x = ptsrc.x;
        y = ptsrc.y;
        z = ptsrc.z;
    }

    point3D_base<T> &operator=(const T f)
    {
        x = f;
        y = z = 0;
        return *this;
    }
    void Set(const T fx, const T fy, const T fz)
    {
        x = fx;
        y = fy;
        z = fz;
    }
    void SetEmpty()
    {
        x = y = z = 0;
    }
    T &operator()(const int i)
    {
        if (i < 0 || i > 2)
            throw TException("ERROR: index out of range");
        if (i == 0)
            return x;
        else if (i == 1)
            return y;
        return z;
    }
    T operator()(const int i) const
    {
        if (i < 0 || i > 2)
            throw TException("ERROR: index out of range");
        if (i == 0)
            return x;
        else if (i == 1)
            return y;
        return z;
    }
    T &operator[](int i)
    {
        return (*this)(i);
    }
    T operator[](int i) const
    {
        return (*this)(i);
    }
    point3D_base<T> operator-()
    {
        point3D_base<T> pt;
        pt.x = -x;
        pt.y = -y;
        pt.z = -z;
        return pt;
    }
    point3D_base<T> &operator+=(const point3D_base<T> &ptsrc)
    {
        x += ptsrc.x;
        y += ptsrc.y;
        z += ptsrc.z;
        return *this;
    }
    point3D_base<T> &operator-=(const point3D_base<T> &ptsrc)
    {
        x -= ptsrc.x;
        y -= ptsrc.y;
        z -= ptsrc.z;
        return *this;
    }
    point3D_base<T> &operator*=(const T t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }
    T operator*(const point3D_base &ptsrc)
    {
        return x * ptsrc.x + y * ptsrc.y + z * ptsrc.z;
    }
    point3D_base<T> &operator/=(const T t)
    {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }
    bool operator==(const point3D_base<T> &ptsrc)
    {
        return (x == ptsrc.x && y == ptsrc.y && z == ptsrc.z);
    }
    bool operator!=(const point3D_base<T> &ptsrc)
    {
        return !(*this == ptsrc);
    }
    prec Norm()
    {
        return norm(cmplx(x)) + norm(cmplx(y)) + norm(cmplx(z));
    }
    prec Abs()
    {
        return sqrt(Norm());
    }
    void Normalize()
    {
        *this /= Abs();
    }

    T x, y, z;
};

template <class T>
inline point3D_base<T> operator+(const point3D_base<T> &pt1, const point3D_base<T> &pt2)
{
    point3D_base<T> p = pt1;
    return p += pt2;
}

template <class T>
inline point3D_base<T> operator-(const point3D_base<T> &pt1, const point3D_base<T> &pt2)
{
    point3D_base<T> p = pt1;
    return p -= pt2;
}

template <class T>
inline point3D_base<T> operator*(const point3D_base<T> &pt, T src)
{
    point3D_base<T> p = pt;
    return p *= src;
}

template <class T>
inline point3D_base<T> operator*(const int src, const point3D_base<T> &pt)
{
    point3D_base<T> p = pt;
    return p *= src;
}

template <class T>
inline point3D_base<T> operator*(const double src, const point3D_base<T> &pt)
{
    point3D_base<T> p = pt;
    return p *= src;
}

template <class T, class X>
inline point3D_base<T> operator/(const point3D_base<T> &pt, const X src)
{
    point3D_base<T> p = pt;
    return p /= src;
}

template <class T>
inline istream &operator>>(istream &ps,
                           point3D_base<T> &pt)
{
    prec cx, cy, cz;
    char ch;

    if (ps >> ch && ch != '(') // leading character
        ps.putback(ch);
    ps >> cx;
    if ((ch = ps.get()) == '\t' || ch == ',')
    {
        ps >> cy;
        if ((ch = ps.get()) == '\t' || ch == ',')
            ps >> cz;
        else if ((int)ch != -1)
            ps.putback(ch);
    }
    else if ((int)ch != -1)
        ps.putback(ch);
    if (ps >> ch && ch != ')') // final character
        ps.putback(ch);
    if (!ps.fail())
    {
        pt.x = cx;
        pt.y = cy;
        pt.z = cz;
    }
    return ps;
}

template <class T>
inline ostream &operator<<(ostream &ps,
                           const point3D_base<T> &pt)
{
    return ps << pt.x << '\t' << pt.y << '\t' << pt.z;
}

typedef point3D_base<int> point3D;
typedef point3D_base<prec> TPoint3D;

inline TPoint3D Min(TPoint3D pt1, TPoint3D pt2)
{
    TPoint3D dest;

    dest.x = min(pt1.x, pt2.x);
    dest.y = min(pt1.y, pt2.y);
    dest.z = min(pt1.z, pt2.z);
    return dest;
}

inline TPoint3D Max(TPoint3D pt1, TPoint3D pt2)
{
    TPoint3D dest;

    dest.x = max(pt1.x, pt2.x);
    dest.y = max(pt1.y, pt2.y);
    dest.z = max(pt1.z, pt2.z);
    return dest;
}

class TCPoint3D : public point3D_base<cmplx>
{
public:
    explicit TCPoint3D(const cmplx fx = 0, const cmplx fy = 0, const cmplx fz = 0)
        : point3D_base<cmplx>(fx, fy, fz){};
    TCPoint3D(const point3D_base<cmplx> &ptsrc)
    {
        x = ptsrc.x;
        y = ptsrc.y;
        z = ptsrc.z;
    }
    TCPoint3D(const TPoint3D &ptsrc)
    {
        x = cmplx(ptsrc.x, 0);
        y = cmplx(ptsrc.y, 0);
        z = cmplx(ptsrc.z, 0);
    }
};

inline cmplx operator*(const TCPoint3D &src1, const TPoint3D &src2)
{
    return src1.x * src2.x + src1.y * src2.y + src1.z * src2.z;
}

inline cmplx operator*(const TPoint3D &src1, const TCPoint3D &src2)
{
    return src1.x * src2.x + src1.y * src2.y + src1.z * src2.z;
}

inline TCPoint3D operator*(const TCPoint3D &pt, const cmplx &src)
{
    TCPoint3D p = pt;
    return p *= src;
}

inline TCPoint3D operator*(const cmplx &src, const TCPoint3D &pt)
{
    TCPoint3D p = pt;
    return p *= src;
}

inline TCPoint3D operator*(const cmplx &src, const TPoint3D &pt)
{
    TCPoint3D p = pt;
    return p *= src;
}

inline TCPoint3D operator*(const TPoint3D &pt, const cmplx &src)
{
    TCPoint3D p = pt;
    return p *= src;
}

// ***************************************************************************
// 3D complex point class
// ***************************************************************************

/*class TCPoint3D
  {
  public:
    explicit TCPoint3D(const cmplx fx=0, const cmplx fy=0, const cmplx fz=0)
      {x = fx; y = fy; z = fz;}
    TCPoint3D(const TCPoint3D& ptsrc)
      {x = ptsrc.x;y = ptsrc.y;z = ptsrc.z;}
    TCPoint3D(const TPoint3D& ptsrc)
      {x = cmplx(ptsrc.x,0);y = cmplx(ptsrc.y,0);z = cmplx(ptsrc.z,0);
       cbegin='\0';csep='\t';cend='\0';}
    TCPoint3D& operator= (const cmplx f)
{
x = f;
y = z = 0;
return *this;
}

void Set(const cmplx fx, const cmplx fy, const cmplx fz)
{x = fx; y = fy; z = fz;}
void SetEmpty() {x = y = z = 0;}
cmplx& operator ()(const int i)
{
if(i<0 || i>2) throw TException("ERROR: index out of range");
if(i==0)
return x;
else if(i==1)
return y;
return z;
}
cmplx operator ()(const int i) const
{
if(i<0 || i>2) throw TException("ERROR: index out of range");
if(i==0)
return x;
else if(i==1)
return y;
return z;
}
cmplx& operator [](int i) {return (*this)(i);}
cmplx operator [](int i) const {return (*this)(i);}
TCPoint3D operator -()
{
TCPoint3D pt;
pt.x = -x;
pt.y = -y;
pt.z = -z;
return pt;
}
TCPoint3D& operator += (const TCPoint3D& ptsrc)
{
x += ptsrc.x;
y += ptsrc.y;
z += ptsrc.z;
return *this;
}
TCPoint3D& operator -= (const TCPoint3D& ptsrc)
{
x -= ptsrc.x;
y -= ptsrc.y;
z -= ptsrc.z;
return *this;
}
TCPoint3D& operator *= (const cmplx t)
{
x *= t;
y *= t;
z *= t;
return *this;
}
cmplx operator * (const TCPoint3D& ptsrc)
{
return x*ptsrc.x+y*ptsrc.y+z*ptsrc.z;
}
TCPoint3D& operator /= (const cmplx t)
{
x /= t;
y /= t;
z /= t;
return *this;
}
bool operator == (const TCPoint3D& ptsrc)
{return (x==ptsrc.x && y==ptsrc.y && z==ptsrc.z);}
bool operator != (const TCPoint3D& ptsrc)
{return !(*this==ptsrc);}
prec Norm() {return norm(cmplx(x))+norm(cmplx(y))+norm(cmplx(z));}
prec Abs()  {return sqrt(Norm());}
void Normalize() {*this /= Abs();}

cmplx   x, y, z;
};

inline TCPoint3D operator+ (const TCPoint3D& pt1, const TCPoint3D& pt2)
{
TCPoint3D p = pt1;
return p += pt2;
}

inline TCPoint3D operator- (const TCPoint3D& pt1, const TCPoint3D& pt2)
{
TCPoint3D p = pt1;
return p -= pt2;
}

inline TCPoint3D operator* (const TCPoint3D& pt, const cmplx src)
{
TCPoint3D p = pt;
return p *= src;
}

inline TCPoint3D operator* (const int src, const TCPoint3D& pt)
{
TCPoint3D p = pt;
return p *= src;
}

inline TCPoint3D operator* (const double src, const TCPoint3D& pt)
{
TCPoint3D p = pt;
return p *= src;
}

inline TCPoint3D operator* (const cmplx& src, const TCPoint3D& pt)
{
TCPoint3D p = pt;
return p *= src;
}

inline TCPoint3D operator/ (const TCPoint3D& pt, const cmplx src)
{
TCPoint3D p = pt;
return p /= src;
}

inline cmplx operator* (const TCPoint3D& src1, const TPoint3D& src2)
{
return src1.x*src2.x+src1.y*src2.y+src1.z*src2.z;
}

inline cmplx operator* (const TPoint3D& src1, const TCPoint3D& src2)
{
return src1.x*src2.x+src1.y*src2.y+src1.z*src2.z;
}

template <class T> inline istream& operator>> (istream& ps,
TCPoint3D& pt)
{
prec    cx, cy, cz;
char    ch;

if(ps >> ch && ch!='(')  // leading character
ps.putback(ch);
ps >> cx;
if((ch = ps.get())=='\t' || ch==',')
{
ps >> cy;
if((ch = ps.get())=='\t' || ch==',')
ps >> cz;
else if(ch!=-1)
ps.putback(ch);
}
else if(ch!=-1)
ps.putback(ch);
if(ps >> ch && ch!=')') // final character
ps.putback(ch);
if (!ps.fail())
{
pt.x = cx;
pt.y = cy;
pt.z = cz;
}
return ps;
}

template <class T> inline ostream& operator<< (ostream& ps,
const TCPoint3D& pt)
{
return ps << pt.x << '\t' << pt.y << '\t' << pt.z;
}*/

/*****************************************************************************
 streamable point output base class
 *****************************************************************************/

template <class T>
class point3D_output
{
public:
    explicit point3D_output(const point3D_base<T> &src)
    {
        pt = src;
    }

    point3D_base<T> pt;
};

template <class T>
inline ostream &operator<<(ostream &ps,
                           const point3D_output<T> &base)
{
    return ps << '(' << base.pt.x << ',' << base.pt.y << ')';
}

typedef point3D_output<cmplx> cPT3D;
typedef point3D_output<prec> pPT3D;
typedef point3D_output<int> iPT3D;

inline TPoint3D CrossProduct(const TPoint3D &src1,
                             const TPoint3D &src2)
{
    TPoint3D dest;

    dest.x = src1.y * src2.z - src1.z * src2.y;
    dest.y = src1.z * src2.x - src1.x * src2.z;
    dest.z = src1.x * src2.y - src1.y * src2.x;
    return dest;
}

inline TCPoint3D CrossProduct(const TCPoint3D &src1,
                              const TPoint3D &src2)
{
    TCPoint3D dest;

    dest.x = src1.y * src2.z - src1.z * src2.y;
    dest.y = src1.z * src2.x - src1.x * src2.z;
    dest.z = src1.x * src2.y - src1.y * src2.x;
    return dest;
}

inline TCPoint3D CrossProduct(const TPoint3D &src1,
                              const TCPoint3D &src2)
{
    TCPoint3D dest;

    dest.x = src1.y * src2.z - src1.z * src2.y;
    dest.y = src1.z * src2.x - src1.x * src2.z;
    dest.z = src1.x * src2.y - src1.y * src2.x;
    return dest;
}

// vector class

template <class T>
class Vector
{
public:
    Vector(void)
    {
        imax = 0;
        p = 0;
    }
    Vector(unsigned i)
    {
        if (i == 0)
        {
            imax = 0;
            p = 0;
            return;
        }
        try
        {
            p = new T[i];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = i;
        ZeroInit();
    };
    Vector(const Vector<T> &vsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new T[vsrc.imax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = vsrc.imax;
        for (unsigned i = 0; i < imax; i++)
            p[i] = vsrc.p[i];
    }
    ~Vector()
    {
        if (p)
            delete[] p;
    };

    void ZeroInit()
    {
        T DefT = T();
        for (unsigned i = 0; i < imax; i++)
            p[i] = DefT;
    }
    void Reallocate(unsigned i)
    {
        T *pnew;

        if (i == 0)
        {
            if (p)
            {
                delete[] p;
                p = 0;
            }
            imax = 0;
            return;
        }
        try
        {
            pnew = new T[i];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        if (p)
            delete[] p;
        p = pnew;
        imax = i;
    }

    // resize vector to NewSize and shift contents by nOffset
    void ReSize(int isize, int nOffset = 0)
    {
        unsigned i, NewSize;
        long iold;
        T *ptemp, DefT = T();

        NewSize = max(0, isize);
        try
        {
            ptemp = new T[NewSize];
            for (i = 0, iold = -nOffset; i < NewSize; i++, iold++)
                if (iold >= 0 && iold < imax)
                    ptemp[i] = p[iold];
                else
                    ptemp[i] = DefT;
            delete[] p;
            p = ptemp;
            imax = NewSize;
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
    }
    T &At(const int i)
    {
        if (CheckIndex(i) == false)
            throw TException("ERROR: index out of range");
        return p[i];
    }
    T At(const int i) const
    {
        if (CheckIndex(i) == false)
            throw TException("ERROR: index out of range");
        return p[i];
    }
    T &operator()(const int i)
    {
        return At(i);
    }
    T operator()(const int i) const
    {
        return At(i);
    }
    T &operator[](const int i)
    {
        return At(i);
    }
    T operator[](const int i) const
    {
        return At(i);
    }
    const Vector<T> &operator=(const Vector<T> &vsrc)
    {
        if (this != &vsrc)
        {
            if (p)
                delete[] p;
            try
            {
                p = new T[vsrc.imax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = vsrc.imax;
            for (unsigned i = 0; i < imax; i++)
                p[i] = vsrc.p[i];
        }
        return *this;
    }
    unsigned GetSize() const
    {
        return imax;
    }
    bool CheckIndex(const unsigned i) const
    {
        return i < imax;
    }
    void InsertBegin(int i)
    {
        ReSize(imax + i, i);
    }
    void InsertEnd(int i)
    {
        ReSize(imax + i);
    }

protected:
    T *p;
    unsigned imax;
};

template <class T>
ostream &operator<<(ostream &ps, Vector<T> &src)
{
    int i, imax = src.GetSize();

    ps << "vector size:\t" << imax << endl;
    for (i = 0; i < imax - 1; i++)
        ps << src(i) << '\t';
    if (imax > 0)
        ps << src(i) << endl;
    return ps;
}

template <class T>
istream &operator>>(istream &ps, Vector<T> &src)
{
    int i, imax;

    ps.ignore(1000, '\t');
    ps >> imax;
    src.Reallocate(imax);
    ps.get();
    for (i = 0; i < imax; i++)
        ps >> src(i);
    ps.get();
    return ps;
}

// matrix class

template <class T>
class Matrix
{
public:
    Matrix(void)
    {
        p = 0;
        imax = 0;
        jmax = 0;
    }
    Matrix(unsigned i, unsigned j = 1)
    {
        int ij = i * j;
        if (ij == 0)
        {
            imax = jmax = 0;
            p = 0;
            return;
        }
        try
        {
            p = new T[ij];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = i;
        jmax = j;
        ZeroInit();
    }
    Matrix(const Matrix<T> &msrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new T[msrc.imax * msrc.jmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = msrc.imax;
        jmax = msrc.jmax;
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] = msrc(i, j);
    }
    ~Matrix()
    {
        if (p)
            delete[] p;
        p = 0;
    }

    void ZeroInit()
    {
        T DefT = T();
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] = DefT;
    }
    void Reallocate(unsigned i, unsigned j)
    {
        unsigned NewSize = i * j;
        T *pnew;

        if (NewSize == 0)
        {
            if (p)
            {
                delete[] p;
                p = 0;
            }
            imax = 0;
            jmax = 0;
            return;
        }
        try
        {
            pnew = new T[NewSize];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        if (p)
            delete[] p;
        p = pnew;
        imax = i;
        jmax = j;
    }
    void ReSize(unsigned iNewSize, unsigned jNewSize, int niOffset = 0,
                int njOffset = 0)
    {
        //, OldSize=imax*jmax
        unsigned i, j, NewSize = iNewSize * jNewSize;
        long iold, jold, oldoff, newoff;
        T *ptemp;
        T DefT = T();

        try
        {
            ptemp = new T[NewSize];
            newoff = 0;
            oldoff = -njOffset * imax;
            for (j = 0, jold = -njOffset; j < jNewSize; j++, jold++, newoff += iNewSize,
                oldoff += imax)
                for (i = 0, iold = -niOffset; i < iNewSize; i++, iold++)
                    if (iold >= 0 && iold < imax && jold >= 0 && jold < jmax)
                        ptemp[i + newoff] = p[iold + oldoff];
                    else
                        ptemp[i + newoff] = DefT;
            delete[] p;
            p = ptemp;
            imax = iNewSize;
            jmax = jNewSize;
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
    }
    T &At(unsigned i, unsigned j)
    {
        if (CheckIndex(i, j) == false)
            throw TException("ERROR: index out of range");
        return p[i + imax * j];
    }
    T At(unsigned i, unsigned j) const
    {
        if (CheckIndex(i, j) == false)
            throw TException("ERROR: index out of range");
        return p[i + imax * j];
    }
    T &operator()(unsigned i, unsigned j)
    {
        return At(i, j);
    }
    T operator()(unsigned i, unsigned j) const
    {
        return At(i, j);
    }
    const Matrix<T> &operator=(const Matrix<T> &msrc)
    {
        if (this != &msrc)
        {
            if (p)
                delete[] p;
            try
            {
                p = new T[msrc.imax * msrc.jmax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = msrc.imax;
            jmax = msrc.jmax;
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * j] = msrc(i, j);
        }
        return *this;
    }
    bool CheckIndex(const unsigned i, const unsigned j) const
    {
        return (i < imax && j < jmax);
    }
    unsigned GetiMax() const
    {
        return imax;
    }
    unsigned GetjMax() const
    {
        return jmax;
    }
    void InsertBegin(int i, int j)
    {
        ReSize(imax + i, jmax + j, i, j);
    }
    void InsertEnd(int i, int j)
    {
        ReSize(imax + i, jmax + j);
    }

protected:
    unsigned imax, jmax;
    T *p;
};

template <class T>
ostream &operator<<(ostream &ps, Matrix<T> &src)
{
    int i, j;
    point pt = point(src.GetiMax(), src.GetjMax());

    ps << "matrix size:\t" << pt << endl;
    for (i = 0; i < pt.x; i++)
    {
        for (j = 0; j < pt.y - 1; j++)
            ps << src(i, j) << '\t';
        if (pt.y > 0)
            ps << src(i, j) << endl;
    }
    return ps;
}

template <class T>
istream &operator>>(istream &ps, Matrix<T> &src)
{
    int i, j;
    point pt;

    ps.ignore(1000, '\t');
    ps >> pt;
    src.Reallocate(pt.x, pt.y);
    ps.get();
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            ps >> src(i, j);
    ps.get();
    return ps;
}

// tensor class

template <class T>
class Tensor
{
public:
    Tensor(void)
    {
        imax = 0;
        jmax = 0;
        kmax = 0;
        p = 0;
    }
    Tensor(unsigned i, unsigned j = 1, unsigned k = 1)
    {
        int ijk = i * j * k;
        if (ijk == 0)
        {
            imax = jmax = kmax = 0;
            p = 0;
            return;
        }
        try
        {
            p = new T[i * j * k];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = i;
        jmax = j;
        kmax = k;
        ZeroInit();
    }
    Tensor(const Tensor<T> &tsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new T[tsrc.imax * tsrc.jmax * tsrc.kmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = tsrc.imax;
        jmax = tsrc.jmax;
        kmax = tsrc.kmax;
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] = tsrc(i, j, k);
    }
    ~Tensor()
    {
        if (p)
            delete[] p;
    }

    void ZeroInit()
    {
        T DefT = T();
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] = DefT;
    }
    void Reallocate(unsigned i, unsigned j, unsigned k)
    {
        unsigned NewSize = i * j * k;
        T *pnew;

        if (NewSize == 0)
        {
            if (p)
            {
                delete[] p;
                p = 0;
            }
            imax = 0;
            jmax = 0;
            kmax = 0;
            return;
        }
        try
        {
            pnew = new T[NewSize];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        if (p)
            delete[] p;
        p = pnew;
        imax = i;
        jmax = j;
        kmax = k;
    }
    void ReSize(unsigned iNewSize, unsigned jNewSize, unsigned kNewSize,
                int niOffset = 0, int njOffset = 0, int nkOffset = 0)
    {
        unsigned i, j, k, ijold = imax * jmax, ijnew = iNewSize * jNewSize;
        unsigned NewSize = ijnew * kNewSize; //OldSize=ijold*kmax,
        long iold, jold, kold, joldoff, jnewoff, koldoff, knewoff;
        T *ptemp;
        T DefT = T();

        try
        {
            ptemp = new T[NewSize];
            knewoff = 0;
            koldoff = -nkOffset * ijold;
            for (k = 0, kold = -nkOffset; k < kNewSize; k++, kold++, knewoff += ijnew,
                koldoff += ijold)
            {
                jnewoff = knewoff;
                joldoff = koldoff - njOffset * imax;
                for (j = 0, jold = -njOffset; j < jNewSize; j++, jold++,
                    jnewoff += iNewSize, joldoff += imax)
                    for (i = 0, iold = -niOffset; i < iNewSize; i++, iold++)
                        if (iold >= 0 && iold < imax && jold >= 0 && jold < jmax && kold >= 0 && kold < kmax)
                            ptemp[i + jnewoff] = p[iold + joldoff];
                        else
                            ptemp[i + jnewoff] = DefT;
            }
            delete[] p;
            p = ptemp;
            imax = iNewSize;
            jmax = jNewSize;
            kmax = kNewSize;
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
    }
    T &At(const unsigned i, const unsigned j, const unsigned k)
    {
        if (CheckIndex(i, j, k) == false)
            throw TException("ERROR: index out of range");
        return p[i + imax * (j + jmax * k)];
    }
    T At(const unsigned i, const unsigned j, const unsigned k) const
    {
        if (CheckIndex(i, j, k) == false)
            throw TException("ERROR: index out of range");
        return p[i + imax * (j + jmax * k)];
    }
    T &operator()(const unsigned i, const unsigned j, const unsigned k)
    {
        return At(i, j, k);
    }
    T operator()(const unsigned i, const unsigned j, const unsigned k) const
    {
        return At(i, j, k);
    }
    const Tensor<T> &operator=(const Tensor<T> &tsrc)
    {
        if (this != &tsrc)
        {
            if (p)
                delete[] p;
            try
            {
                p = new T[tsrc.imax * tsrc.jmax * tsrc.kmax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = tsrc.imax;
            jmax = tsrc.jmax;
            kmax = tsrc.kmax;
            for (unsigned k = 0; k < kmax; k++)
                for (unsigned j = 0; j < jmax; j++)
                    for (unsigned i = 0; i < imax; i++)
                        p[i + imax * (j + jmax * k)] = tsrc(i, j, k);
        }
        return *this;
    }
    bool CheckIndex(const unsigned i, const unsigned j, const unsigned k) const
    {
        return (i < imax && j < jmax && k < kmax);
    }
    unsigned GetiMax() const
    {
        return imax;
    }
    unsigned GetjMax() const
    {
        return jmax;
    }
    unsigned GetkMax() const
    {
        return kmax;
    }
    void InsertBegin(int i, int j, int k)
    {
        ReSize(imax + i, jmax + j, kmax + k, i, j, k);
    }
    void InsertEnd(int i, int j, int k)
    {
        ReSize(imax + i, jmax + j, kmax + k);
    }

protected:
    unsigned imax, jmax, kmax;
    T *p;
};

template <class T>
ostream &operator<<(ostream &ps, Tensor<T> &src)
{
    int i, j, k;
    point3D pt = point(src.GetiMax(), src.GetjMax());

    ps << "tensor size:\t" << pt << endl;
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
        {
            for (k = 0; k < pt.z - 1; k++)
                ps << src(i, j, k) << '\t';
            if (pt.z > 0)
                ps << src(i, j, k) << endl;
        }
    return ps;
}

template <class T>
istream &operator>>(istream &ps, Tensor<T> &src)
{
    int i, j, k;
    point3D pt;

    ps.ignore(1000, '\t');
    ps >> pt;
    src.Reallocate(pt.x, pt.y, pt.z);
    ps.get();
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            for (k = 0; k < pt.z; k++)
                ps >> src(i, j, k);
    ps.get();
    return ps;
}

typedef class Vector<int> ivector;
typedef class Matrix<int> imatrix;
typedef class Tensor<int> itensor;

typedef class Vector<prec> rvector;
typedef class Matrix<prec> rmatrix;
typedef class Tensor<prec> rtensor;

typedef class Vector<cmplx> cvector;
typedef class Matrix<cmplx> cmatrix;
typedef class Tensor<cmplx> ctensor;

template <class T>
class MathMatrix;

// mathematical vector class

template <class T>
class MathVector : public Vector<T>
{
public:
    MathVector(void)
        : Vector<T>(){};
    MathVector(unsigned i)
        : Vector<T>(i){};
    MathVector(const MathVector<T> &vsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new T[vsrc.imax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = vsrc.imax;
        for (unsigned i = 0; i < imax; i++)
            p[i] = vsrc.p[i];
    }

    /*    const MathVector<T>& operator = (const MathVector<T>& vsrc)
            {
            if(this!=&vsrc)
              {
              if(p) delete[] p;
              try{p = new T[vsrc.imax];}
              catch(bad_alloc){throw TException("ERROR: not enough memory");}
              imax = vsrc.imax;
              for(unsigned i=0;i<imax;i++)
                p[i] = vsrc.p[i];
              }
      return *this;
      }*/
    MathVector<T> &operator+=(MathVector<T> vsrc)
    {
        if (imax != vsrc.imax)
            throw TException("ERROR: index out of range");
        for (unsigned i = 0; i < imax; i++)
            p[i] += vsrc.p[i];
        return *this;
    }
    MathVector<T> operator+(MathVector<T> vsrc)
    {
        MathVector<T> v = *this;
        return v += vsrc;
    }
    MathVector<T> &operator-=(MathVector<T> vsrc)
    {
        if (imax != vsrc.imax)
            throw TException("ERROR: index out of range");
        for (unsigned i = 0; i < imax; i++)
            p[i] -= vsrc.p[i];
        return *this;
    }
    MathVector<T> operator-(MathVector<T> vsrc)
    {
        MathVector<T> v = *this;
        return v -= vsrc;
    }
    MathVector<T> operator-()
    {
        MathVector<T> v;
        for (unsigned i = 0; i < imax; i++)
            v[i] = -p[i];
        return v;
    }
    MathVector<T> &operator*=(prec t)
    {
        for (unsigned i = 0; i < imax; i++)
            p[i] *= t;
        return *this;
    }
    MathVector<T> &operator*=(int t)
    {
        for (unsigned i = 0; i < imax; i++)
            p[i] *= t;
        return *this;
    }
    MathVector<T> operator*(prec t)
    {
        MathVector<T> v = *this;
        return v *= t;
    }
    MathVector<T> operator*(int t)
    {
        MathVector<T> v = *this;
        return v *= t;
    }
    T operator*(MathVector<T> vsrc)
    {
        T t = 0;

        if (imax != vsrc.imax)
            throw TException("ERROR: index out of range");
        for (unsigned i = 0; i < imax; i++)
            t += p[i] * vsrc.p[i];
        return t;
    }
    MathVector<T> &operator/=(prec t)
    {
        for (unsigned i = 0; i < imax; i++)
            p[i] /= t;
        return *this;
    }
    MathVector<T> &operator/=(int t)
    {
        for (unsigned i = 0; i < imax; i++)
            p[i] /= t;
        return *this;
    }
    MathVector<T> operator/(prec t)
    {
        MathVector<T> v = *this;
        return v /= t;
    }
    MathVector<T> operator/(int t)
    {
        MathVector<T> v = *this;
        return v /= t;
    }
    bool operator==(MathVector<T> vsrc)
    {
        if (imax != vsrc.imax)
            return false;
        for (unsigned i = 0; i < imax; i++)
            if (p[i] != vsrc.p[i])
                return false;
        return true;
    }
    bool operator!=(MathVector<T> vsrc)
    {
        return !(*this == vsrc);
    }
    template <class TT>
    friend MathVector<TT> operator*(const MathMatrix<TT> &, const MathVector<TT> &);
};

//  multiply vector with scalar

template <class T, class X>
MathVector<T> operator*(X t, MathVector<T> vsrc)
{
    MathVector<T> v = vsrc;
    return v *= t;
}

// prec vector

class rmvector : public MathVector<prec>
{
public:
    rmvector(void)
        : MathVector<prec>()
    {
    }
    rmvector(unsigned i)
        : MathVector<prec>(i)
    {
    }
    rmvector(const rmvector &vsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new prec[vsrc.imax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = vsrc.imax;
        for (unsigned i = 0; i < imax; i++)
            p[i] = vsrc.p[i];
    }

    const rmvector &operator=(const rmvector &vsrc)
    {
        if (this != &vsrc)
        {
            if (p)
                delete[] p;
            try
            {
                p = new prec[vsrc.imax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = vsrc.imax;
            for (unsigned i = 0; i < imax; i++)
                p[i] = vsrc.p[i];
        }
        return *this;
    }
    prec Norm()
    {
        prec t = 0;

        for (unsigned i = 0; i < imax; i++)
            t += p[i] * p[i];
        return t;
    }
    prec Abs()
    {
        return sqrt(Norm());
    }
};

// complex vector

class cmvector : public MathVector<cmplx>
{
public:
    cmvector(void)
        : MathVector<cmplx>()
    {
    }
    cmvector(unsigned i)
        : MathVector<cmplx>(i)
    {
    }
    cmvector(const cmvector &vsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new cmplx[vsrc.imax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = vsrc.imax;
        for (unsigned i = 0; i < imax; i++)
            p[i] = vsrc.p[i];
    }

    const cmvector &operator=(const cmvector &vsrc)
    {
        if (this != &vsrc)
        {
            if (p)
                delete[] p;
            try
            {
                p = new cmplx[vsrc.imax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = vsrc.imax;
            for (unsigned i = 0; i < imax; i++)
                p[i] = vsrc.p[i];
        }
        return *this;
    }
    prec Abs()
    {
        prec t = 0;

        for (unsigned i = 0; i < imax; i++)
            t += norm(p[i]);
        return sqrt(t);
    }
};

// mathematical matrix class

template <class T>
class MathMatrix : public Matrix<T>
{
public:
    MathMatrix(void)
        : Matrix<T>(){};
    MathMatrix(unsigned i, unsigned j)
        : Matrix<T>(i, j){};
    MathMatrix(const MathMatrix<T> &msrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new T[msrc.imax * msrc.jmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = msrc.imax;
        jmax = msrc.jmax;
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] = msrc(i, j);
    }

    /*    const MathMatrix<T>& operator = (const MathMatrix<T>& msrc)
            {
            if(this!=&msrc)
              {
              if(p) delete[] p;
              try{p = new T[msrc.imax*msrc.jmax];}
              catch(bad_alloc){throw TException("ERROR: not enough memory");}
              imax = msrc.imax;
              jmax = msrc.jmax;
              for(unsigned j=0;j<jmax;j++)
                for(unsigned i=0;i<imax;i++)
      p[i+imax*j] = msrc(i,j);
      }
      return *this;
      }*/
    MathMatrix<T> &operator+=(MathMatrix<T> msrc)
    {
        if (imax != msrc.imax || jmax != msrc.jmax)
            throw TException("ERROR: index out of range");
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] += msrc(i, j);
        return *this;
    }
    MathMatrix<T> operator+(MathMatrix<T> msrc)
    {
        MathMatrix<T> m = *this;
        return m += msrc;
    }
    MathMatrix<T> &operator-=(MathMatrix<T> msrc)
    {
        if (imax != msrc.imax || jmax != msrc.jmax)
            throw TException("ERROR: index out of range");
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] -= msrc(i, j);
        return *this;
    }
    MathMatrix<T> operator-(MathMatrix<T> msrc)
    {
        MathMatrix<T> m = *this;
        return m -= msrc;
    }
    MathMatrix<T> operator-()
    {
        MathMatrix<T> m;
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                v(i, j) = -p[i + imax * j];
        return m;
    }
    MathMatrix<T> &operator*=(prec t)
    {
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] *= t;
        return *this;
    }
    MathMatrix<T> &operator*=(int t)
    {
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] *= t;
        return *this;
    }
    MathMatrix<T> operator*(prec t)
    {
        MathMatrix<T> m = *this;
        return m *= t;
    }
    MathMatrix<T> operator*(int t)
    {
        MathMatrix<T> m = *this;
        return m *= t;
    }
    MathMatrix<T> operator*(MathMatrix<T> msrc)
    {
        if (jmax != msrc.imax)
            throw TException("ERROR: index out of range");

        MathMatrix<T> m(imax, msrc.jmax);

        for (unsigned j = 0; j < msrc.jmax; j++)
            for (unsigned i = 0; i < imax; i++)
            {
                m(i, j) = 0;
                for (unsigned k = 0; k < jmax; k++)
                    m(i, j) += p[i + imax * k] * msrc(k, j);
            }
        return m;
    }
    MathMatrix<T> &operator*=(MathMatrix<T> msrc)
    {
        T t, *pnew;

        if (jmax != msrc.imax)
            throw TException("ERROR: index out of range");
        try
        {
            pnew = new T[imax * msrc.jmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        for (unsigned j = 0; j < msrc.jmax; j++)
            for (unsigned i = 0; i < imax; i++)
            {
                t = 0;
                for (unsigned k = 0; k < jmax; k++)
                    t += p[i + imax * k] * msrc(k, j);
                pnew[i + j * imax] = t;
            }
        if (p)
            delete[] p;
        p = pnew;
        jmax = msrc.jmax;
        return *this;
    }
    MathMatrix<T> &operator/=(prec t)
    {
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] /= t;
        return *this;
    }
    MathMatrix<T> &operator/=(int t)
    {
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] /= t;
        return *this;
    }
    MathMatrix<T> operator/(prec t)
    {
        MathMatrix<T> m = *this;
        return m /= t;
    }
    MathMatrix<T> operator/(int t)
    {
        MathMatrix<T> m = *this;
        return m /= t;
    }
    bool operator==(MathMatrix<T> msrc)
    {
        if (imax != msrc.imax || jmax != msrc.jmax)
            return false;
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                if (p[i + imax * j] != msrc(i, j))
                    return false;
        return true;
    }
    bool operator!=(MathMatrix<T> msrc)
    {
        return !(*this == msrc);
    }
    template <class TT>
    friend MathVector<TT> operator*(const MathMatrix<TT> &, const MathVector<TT> &);
};

// multiply matrix with scalar

template <class T, class X>
MathMatrix<T> operator*(X t, MathMatrix<T> msrc)
{
    MathMatrix<T> m = msrc;
    return m *= t;
}

// multiply matrix with vector

template <class T>
MathVector<T> operator*(const MathMatrix<T> &m,
                        const MathVector<T> &v)
{
    MathVector<T> t;

    if (m.jmax != v.imax)
        throw TException("ERROR: index out of range");
    for (unsigned i = 0; i < m.imax; i++)
    {
        t[i] = 0;
        for (unsigned j = 0; j < m.jmax; j++)
            t[i] += m(i, j) * v[j];
    }
    return t;
}

// prec matrix

class rmmatrix : public MathMatrix<prec>
{
public:
    rmmatrix(void)
        : MathMatrix<prec>(){};
    rmmatrix(unsigned i, unsigned j)
        : MathMatrix<prec>(i, j){};
    rmmatrix(const rmmatrix &msrc)
        : MathMatrix<prec>(msrc.GetiMax(), msrc.GetjMax())
    {
        if (imax != msrc.GetiMax() || jmax != msrc.GetjMax())
            throw TException("ERROR: index out of range");
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] = msrc.p[i + imax * j];
    }

    /*    rmmatrix& operator = (const rmmatrix& msrc)
            {
            if(this!=&msrc)
              {
              if(p) delete[] p;
              try{p = new prec[msrc.imax*msrc.jmax];}
              catch(bad_alloc){throw TException("ERROR: not enough memory");}
              imax = msrc.imax;
              jmax = msrc.jmax;
              for(unsigned j=0;j<jmax;j++)
                for(unsigned i=0;i<imax;i++)
      p[i+imax*j] = msrc(i,j);
      }
      return *this;
      }*/
    enum MathResult
    {
        MathResult_Ok,
        MathResult_MultSolutions,
        MathResult_NoSolution,
        MathResult_Singular
    };

    MathResult GaussElim(rmvector *pv = 0);
    MathResult BackElim(rmvector &, rmvector *pv = 0);
    MathResult LUDecomp(rmatrix &A, ivector &, int &);
    MathResult Determinant(rmmatrix &, prec &);
    prec LUDeterminant(rmmatrix &, int);
};

// mathematical tensor class

template <class T>
class MathTensor : public Tensor<T>
{
public:
    MathTensor(void)
        : Tensor<T>(){};
    MathTensor(unsigned i, unsigned j, unsigned k)
        : Tensor<T>(i, j, k){};
    MathTensor(const MathTensor<T> &tsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new T[tsrc.imax * tsrc.jmax * tsrc.kmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = tsrc.imax;
        jmax = tsrc.jmax;
        kmax = tsrc.kmax;
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] = tsrc(i, j, k);
    }

    /*    const MathTensor<T>& operator = (const MathTensor<T>& tsrc)
            {
            if(this!=&tsrc)
              {
              if(p) delete[] p;
              try{p = new T[tsrc.imax*tsrc.jmax*tsrc.kmax];}
              catch(bad_alloc){throw TException("ERROR: not enough memory");}
              imax = tsrc.imax;
              jmax = tsrc.jmax;
              kmax = tsrc.kmax;
              for(unsigned k=0;k<kmax;k++)
      for(unsigned j=0;j<jmax;j++)
      for(unsigned i=0;i<imax;i++)
      p[i+imax*(j+jmax*k)] = tsrc(i,j,k);
      }
      return *this;
      }*/
    MathTensor<T> &operator+=(MathTensor<T> msrc)
    {
        if (imax != tsrc.imax || jmax != tsrc.jmax || kmax != tsrc.kmax)
            throw TException("ERROR: index out of range");
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] += msrc(i, j, k);
        return *this;
    }
    MathTensor<T> operator+(MathTensor<T> msrc)
    {
        MathTensor<T> t = *this;
        return t += msrc;
    }
    MathTensor<T> &operator-=(MathTensor<T> msrc)
    {
        if (imax != tsrc.imax || jmax != tsrc.jmax || kmax != tsrc.kmax)
            throw TException("ERROR: index out of range");
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] -= msrc(i, j, k);
        return *this;
    }
    MathTensor<T> operator-(MathTensor<T> msrc)
    {
        MathTensor<T> t = *this;
        return t -= msrc;
    }
    MathTensor<T> operator-()
    {
        MathTensor<T> m;
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    m(i, j, k) = -p[i + imax * (j + jmax * k)];
        return m;
    }
    MathTensor<T> &operator*=(prec t)
    {
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] *= t;
        return *this;
    }
    MathTensor<T> &operator*=(int t)
    {
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] *= t;
        return *this;
    }
    MathTensor<T> operator*(prec t)
    {
        MathTensor<T> m = *this;
        return m *= t;
    }
    MathTensor<T> operator*(int t)
    {
        MathTensor<T> m = *this;
        return m *= t;
    }
    MathTensor<T> &operator/=(prec t)
    {
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] /= t;
        return *this;
    }
    MathTensor<T> &operator/=(int t)
    {
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] /= t;
        return *this;
    }
    MathTensor<T> operator/(prec t)
    {
        MathTensor<T> m = *this;
        return m /= t;
    }
    MathTensor<T> operator/(int t)
    {
        MathTensor<T> m = *this;
        return m /= t;
    }
    bool operator==(MathTensor<T> msrc)
    {
        if (imax != msrc.imax || jmax != msrc.jmax || kmax != msrc.kmax)
            return false;
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    if (p[i + imax * (j + jmax * k)] != msrc(i, j, k))
                        return false;
        return true;
    }
    bool operator!=(MathTensor<T> msrc)
    {
        return !(*this == msrc);
    }
};

// multiply tensor with scalar

template <class T, class X>
MathTensor<T> operator*(X t, MathTensor<T> msrc)
{
    MathTensor<T> m = msrc;
    return m *= t;
}

typedef class MathTensor<prec> rmtensor;
#endif
