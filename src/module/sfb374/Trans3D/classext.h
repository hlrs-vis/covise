/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          classext.h  -  class extensions
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __CLASSEXT_H_

#define __CLASSEXT_H_

#include <fstream>

#include "arrays.h"

// Transferstruktur für Fensterdateiroutinen

typedef struct
{
    int unit;
    int type;
    float vers;
} CmFileStruct;

// ***************************************************************************
// float rectangle class
// ***************************************************************************

class TRectF
{
public:
    TRectF()
    {
        SetEmpty();
    }

    float left, right, top, bottom;

    void Set(float, float, float, float);
    void SetEmpty()
    {
        left = 0;
        right = 0;
        top = 0;
        bottom = 0;
    }
};

// ***************************************************************************
// interpolation class
// ***************************************************************************

template <class T>
class Interpolation
{
public:
    Interpolation(int i = 0)
    {
        iPos = 0;
        if (i == 0)
        {
            pDataX = 0;
            pDataY = 0;
            iMaxSize = 0;
            return;
        }
        pDataX = new prec[iMaxSize = i];
        if (pDataX == 0)
            ErrorFunction("out of memory", "ERROR: not enough memory");
        pDataY = new T[i];
        if (pDataY == 0)
            ErrorFunction("out of memory", "ERROR: not enough memory");
    }
    ~Interpolation()
    {
        if (pDataX != 0)
            delete[] pDataX;
        if (pDataY != 0)
            delete[] pDataY;
    }

    T Get(prec);
    prec GetX(int i) const
    {
        return pDataX[i];
    }
    T GetY(int i) const
    {
        return pDataY[i];
    }
    void Set(prec, T);
    void SetX(int i, prec v)
    {
        pDataX[i] = v;
    }
    void SetY(int i, T v)
    {
        pDataY[i] = v;
    }
    void Delete();
    void ReSize(int);
    void Save(int, bool bText = true);
    void Read(int, bool bText = true);
    Interpolation<T> &operator=(const Interpolation<T> &);
    T Integrate(prec, prec);
    int GetEntries() const
    {
        return iPos;
    }
    void SetEntries(int i)
    {
        iPos = i;
    }
    Interpolation<T> &operator*=(const TPoint &pt)
    {
        for (unsigned i = 0; i < iPos; i++)
        {
            pDataX[i] *= pt.x;
            pDataY[i] *= pt.y;
        }
        return *this;
    }
    Interpolation<T> &operator/=(const TPoint &pt)
    {
        for (unsigned i = 0; i < iPos; i++)
        {
            pDataX[i] /= pt.x;
            pDataY[i] /= pt.y;
        }
        return *this;
    }

private:
    prec *pDataX;
    T *pDataY;
    int iMaxSize;
    int iPos;
};

template <class T>
inline void Interpolation<T>::ReSize(int iNewSize)
{
    int i;

    if (iNewSize == 0)
    {
        if (pDataX != 0)
            delete[] pDataX;
        if (pDataY != 0)
            delete[] pDataY;
        pDataX = 0;
        pDataY = 0;
        iMaxSize = 0;
        iPos = 0;
        return;
    }
    prec *pNewX = new prec[iNewSize];
    if (pNewX == 0)
    {
        ErrorFunction("Nicht genug Speicher", "ERROR: not enough memory");
        return;
    }
    T *pNewY = new T[iNewSize];
    if (pNewY == 0)
    {
        ErrorFunction("Nicht genug Speicher", "ERROR: not enough memory");
        delete[] pNewX;
        return;
    }
    i = 0;
    while (i < min(iMaxSize, iNewSize))
    {
        pNewX[i] = pDataX[i];
        pNewY[i] = pDataY[i];
        i++;
    }
    iMaxSize = iNewSize;
    delete[] pDataX;
    delete[] pDataY;
    pDataX = pNewX;
    pDataY = pNewY;
    iPos = min(iPos, iMaxSize);
}

// delete entries

template <class T>
inline void Interpolation<T>::Delete()
{
    ReSize(0);
}

// set entry

template <class T>
inline void Interpolation<T>::Set(prec x, T y)
{
    int i, iIns;

    if (iPos <= iMaxSize)
        ReSize(iMaxSize + 1);
    i = 0;
    iIns = iPos;
    while (i < iPos)
    {
        if (pDataX[i] > x)
        {
            iIns = i;
            break;
        }
        i++;
    }
    i = iPos;
    while (i > iIns)
    {
        pDataX[i] = pDataX[i - 1];
        pDataY[i] = pDataY[i - 1];
        i--;
    }
    pDataX[iIns] = x;
    pDataY[iIns] = y;
    iPos++;
}

// get entry

template <class T>
inline T Interpolation<T>::Get(prec x)
{
    int i;

    if (iPos == 0)
        return 0;
    if (x <= pDataX[0] || iPos < 2)
        return pDataY[0];
    i = 1;
    do
    {
        if (x <= pDataX[i])
            return pDataY[i] + (pDataY[i - 1] - pDataY[i]) / (pDataX[i] - pDataX[i - 1]) * (pDataX[i] - x);
        i++;
    } while (i < iPos);

    return pDataY[iPos - 1];
}

template <class T>
inline Interpolation<T> &Interpolation<T>::operator=(const Interpolation<T> &ipol)
{
    int i;

    if (this == &ipol)
        return *this;
    ReSize(ipol.iMaxSize);
    iPos = 0;
    i = 0;
    while (i < ipol.iMaxSize)
    {
        pDataX[i] = ipol.pDataX[i];
        pDataY[i] = ipol.pDataY[i];
        i++;
    }
    iPos = ipol.iPos;
    return *this;
}

template <class T>
inline T Interpolation<T>::Integrate(prec x0, prec x1)
{
    prec xold, xp, dx;
    T y = 0, y0, yold, yp, dy;
    int i;

    if (iPos == 0)
        return 0;
    i = 0;
    xold = x0;
    yold = pDataY[0];
    while (i < iPos && x0 < x1)
    {
        if (x0 < pDataX[i])
        {
            dy = pDataY[i] - yold;
            dx = pDataX[i] - xold;
            if (dx != 0)
            {
                xp = min(pDataX[i], x1);
                yp = yold + dy / dx * (xp - xold);
                y0 = yold + dy / dx * (x0 - xold);
                y += (xp - x0) * (yp + y0) / 2;
                x0 = xp;
            }
        }
        xold = pDataX[i];
        yold = pDataY[i];
        i++;
    }
    if (x0 < x1)
        y += (x1 - x0) * pDataY[iPos - 1];
    return y;
}

// save class

template <class T>
inline void Interpolation<T>::Save(int unit, bool)
{
    int i;

    unitwrite(unit, iPos, "Eintraege =\t%i\n");
    if (iPos < 1)
        return;
    for (i = 0; i < iPos; i++)
    {
        unitwrite(unit, (double)pDataX[i], "%le\t");
        unitwrite(unit, (double)pDataY[i], "%le\n");
    }
}

// read class

template <class T>
inline void Interpolation<T>::Read(int unit, bool)
{
    int i;
    prec x;
    T y;

    iPos = 0;
    i = readint(unit);
    ReSize(i);
    while (i > 0)
    {
        x = readreal(unit);
        y = T(readreal(unit));
        Set(x, y);
        i--;
    }
}

// write data table

template <class T>
inline ostream &operator<<(ostream &ps,
                           const Interpolation<T> &src)
{
    int imax = src.GetEntries();

    ps << imax;
    if (imax == 1)
        ps << " entry" << endl;
    else
        ps << " entries" << endl;
    for (int i = 0; i < imax; i++)
        ps << src.GetX(i) << '\t' << src.GetY(i) << endl;

    return ps;
}

// read data table

template <class T>
inline istream &operator>>(istream &ps,
                           Interpolation<T> &src)
{
    int imax;
    char ch;
    prec p;
    T t;

    ps >> imax;
    ps.ignore(1000, '\n');
    src.ReSize(imax);
    for (int i = 0; i < imax; i++)
    {
        ps >> p;
        if ((ch = ps.get()) != '\t')
        {
            ps.putback(ch);
            ps.setstate(ios::failbit);
            break;
        }
        ps >> t;
        src.SetX(i, p);
        src.SetY(i, t);
    }
    return ps;
}

typedef Interpolation<prec> rinterpol;
typedef Interpolation<cmplx> cinterpol;

// ***************************************************************************
// 2D interpolation class
// ***************************************************************************

class Interpolation2D
{
public:
    Interpolation2D(int i = 0, int j = 0)
    {
        dx = 0;
        dy = 0;
        Data.ReSize(i, j);
    }

    void ReSize(int i, int j)
    {
        Data.ReSize(i, j);
    }
    prec GetPrec(prec, prec);
    prec Get(int i, int j) const
    {
        return Data(i, j);
    }
    void Set(int i, int j, prec p)
    {
        Data(i, j) = p;
    }
    void Delete();
    Interpolation2D &operator=(const Interpolation2D &);
    int GetXEntries() const
    {
        return Data.GetiMax();
    }
    int GetYEntries() const
    {
        return Data.GetjMax();
    }
    void SetDx(prec p)
    {
        dx = p;
    }
    void SetDy(prec p)
    {
        dy = p;
    }
    prec GetDx() const
    {
        return dx;
    }
    prec GetDy() const
    {
        return dy;
    }
    void Save(int, bool bText = true);
    void Read(int, bool bText = true);
    Interpolation2D &operator*=(const TPoint3D &pt)
    {
        dx *= pt.x;
        dy *= pt.y;
        Data *= pt.z;
        return *this;
    }
    Interpolation2D &operator/=(const TPoint3D &pt)
    {
        dx /= pt.x;
        dy /= pt.y;
        Data /= pt.z;
        return *this;
    }

private:
    rmmatrix Data;
    prec dx, dy;
};

ostream &operator<<(ostream &ps, const Interpolation2D &src);
istream &operator>>(istream &ps, Interpolation2D &src);

// ***************************************************************************
// memory array class
// ***************************************************************************

// constructor

template <class T>
class TStorage : public vector<T>
{
public:
    TStorage()
        : vector<T>(){};
    TStorage(int upper)
        : vector<T>(upper){};
    TStorage(const TStorage &);

    typedef bool (*CondFunc)(const T &, void *);

    //    TStorage<T>& operator = (const TStorage<T>&);
    T LastCondition(CondFunc, const T);
    //    T At(int i) const {return *(begin()+i);}
    //    T& At(int i) {return *(begin()+i);}
    void Save(int);
    void Read(int);
};

// copy class

template <class T>
inline TStorage<T>::TStorage(const TStorage<T> &tv)
{
    int imax, i;

    //  if(this==&tv)
    //    return *this;
    imax = tv.size();
    resize(imax);
    for (i = 0; i < imax; i++)
        (*this)[i] = tv[i];

    return *this;
}

// search for last element fulfilling condition cf

template <class T>
T TStorage<T>::LastCondition(CondFunc cf, const T entry)
{
    T tv;
    int i, imax;

    imax = size();
    if (imax == 0)
        return tv;
    for (i = 0; i < imax; i++)
        if (!cf((*this)[i], (void *)&entry))
        {
            if (i == 0)
                return tv;
            return (*this)[i - 1];
        }

    return (*this)[imax - 1];
}

// save elements

template <class T>
void TStorage<T>::Save(int unit)
{
    int i, imax;

    imax = size();
    unitwrite(unit, imax, "Eintraege =\t%i\n");
    for (i = 0; i < imax; i++)
        (*this)[i].Save(unit);
}

// read elements

template <class T>
void TStorage<T>::Read(int unit)
{
    int i, imax;

    imax = readint(unit);
    resize(imax);
    for (i = 0; i < imax; i++)
        (*this)[i].Read(unit);
}

template <class T>
inline ostream &operator<<(ostream &ps,
                           const TStorage<T> &src)
{
    int i, imax;

    imax = src.size();
    ps << imax;
    if (imax == 1)
        ps << " entry" << endl;
    else
        ps << " entries" << endl;
    for (i = 0; i < imax; i++)
        ps << src[i];
    return ps;
}

template <class T>
inline istream &operator>>(istream &ps, TStorage<T> &src)
{
    int i, imax;

    ps >> imax;
    ps.ignore(1000, '\n');
    src.resize(imax);
    for (i = 0; i < imax; i++)
        ps >> src[i];
    return ps;
}

// conversion int <-> bool

bool ConvBool(int i);
int ConvInt(bool b);
#endif
