/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------------------------------------------
//  $Id: FC_Array.h,v 1.9 2000/03/07 09:12:19 goesslet Exp $
//----------------------------------------------------------------------
//
//  $Log: FC_Array.h,v $
//  Revision 1.9  2000/03/07 09:12:19  goesslet
//  removed bug in copy constructor
//
//  Revision 1.8  2000/02/24 14:37:47  goesslet
//  speedup with set size
//
//  Revision 1.7  2000/01/19 11:41:27  goesslet
//  correct operator=
//
//  Revision 1.6  2000/01/04 11:13:46  goesslet
//  retunr of GetMemUsage is unsigned long for 64bit
//
//  Revision 1.5  1999/12/22 15:04:17  goesslet
//  Free and ~ virtual
//
//  Revision 1.4  1999/12/16 11:52:36  goesslet
//  implementation of methode GetMemoryUsage
//
//  Revision 1.3  1999/12/15 07:28:15  goesslet
//  inlining of init
//
//  Revision 1.2  1999/11/09 10:07:11  kickingf
//  Added Method Init
//
//  Revision 1.1  1999/11/08 13:24:36  goesslet
//  initial version
//
//
//----------------------------------------------------------------------
#ifndef _FC_ARRAY_H_
#define _FC_ARRAY_H_

#include "FC_Base.h"

template <class T>
class FC_Array : public FC_Base
{
    IMPLEMENT_FIRE_BASE(FC_Array, FC_Base)
protected:
    int N;
    T *DATA;

public:
    FC_Array(int n)
    {
        Alloc(n);
    }
    FC_Array(const FC_Array<T> &a);
    FC_Array()
    {
        N = 0;
        DATA = NULL;
    }
    virtual ~FC_Array()
    {
        Free();
    }
    virtual void Free()
    {
        if (N)
            delete[] DATA;
        DATA = NULL;
        N = 0;
    }
    void Alloc(int n)
    {
        N = 0;
        DATA = 0;
        if (n)
        {
            N = n;
            DATA = new T[N];
        }
    }
    void SetSize(int n)
    {
        if (n != N)
        {
            Free();
            Alloc(n);
        }
    }
    int GetSize() const
    {
        return N;
    }
    virtual void SetNumElems(int n)
    {
        SetSize(n);
    }
    virtual int GetNumElems() const
    {
        return N;
    }
    void ReSetSize(int n);
    T *GetPtr()
    {
        return DATA;
    }
    const T *GetPtr() const
    {
        return DATA;
    }
    inline operator T *()
    {
        return DATA;
    }
    inline operator const T *() const
    {
        return DATA;
    }
    inline T &operator[](int i)
    {
        return DATA[i];
    }
    inline const T &operator[](int i) const
    {
        return DATA[i];
    }
    inline T &operator()(int i)
    {
        if (GetSize() <= i)
            ReSetSize(i + 1);
        return DATA[i];
    }
    inline const T &operator()(int i) const
    {
        return DATA[i];
    }
    FC_Array<T> &operator=(const FC_Array<T> &a);
    void Init(const T &t);
    unsigned long GetMemoryUsage()
    {
        return sizeof(FC_Array<T>) + GetNumElems() * sizeof(T);
    }
};

template <class T>
inline void FC_Array<T>::Init(const T &t)
{
    int i;

    for (i = 0; i < GetNumElems(); i++)
    {
        DATA[i] = t;
    }
}

template <class T>
inline ostream &operator<<(ostream &o, const FC_Array<T> &a)
{
    int i;

    o << a.GetSize() << "\n";

    for (i = 0; i < a.GetSize(); i++)
    {
        o << a[i] << " ";
    }
    o << "\n";
    return o;
}

template <class T>
inline istream &operator>>(istream &o, FC_Array<T> &a)
{
    int i, n;

    a.Free();
    o >> n;
    a.Alloc(n);

    for (i = 0; i < n; i++)
    {
        o >> a[i];
    }

    return o;
}

template <class T>
inline FC_Array<T>::FC_Array(const FC_Array<T> &a)
{
    int i;
    N = 0;
    SetSize(a.GetNumElems());

    for (i = 0; i < N; i++)
    {
        DATA[i] = a[i];
    }
}

template <class T>
inline FC_Array<T> &FC_Array<T>::operator=(const FC_Array<T> &a)
{
    int i;

    SetSize(a.GetNumElems());

    for (i = 0; i < N; i++)
    {
        DATA[i] = a[i];
    }
    return (*this);
}

template <class T>
inline void FC_Array<T>::ReSetSize(int n)
{
    if (n == N)
        return;

    T *DATA1;
    int i;
    int n1;
    n1 = N;
    if (n < N)
        n1 = n;
    if (n != 0)
    {
        DATA1 = new T[n];

        for (i = 0; i < n1; i++)
        {
            DATA1[i] = DATA[i];
        }
        Free();
        DATA = DATA1;
    }
    else
    {
        Free();
    }
    N = n;
}

typedef FC_Array<int> FC_IntArray;
typedef FC_Array<double> FC_DoubleArray;
#endif
