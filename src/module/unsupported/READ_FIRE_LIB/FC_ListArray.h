/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------------------------------------------
//   $Id: FC_ListArray.h,v 1.10 2000/03/20 07:37:56 goesslet Exp $
//----------------------------------------------------------------------
//
//  $Log: FC_ListArray.h,v $
//  Revision 1.10  2000/03/20 07:37:56  goesslet
//  use  GetSize for GetMemoryUsage
//
//  Revision 1.9  2000/02/24 14:37:47  goesslet
//  speedup with set size
//
//  Revision 1.8  2000/01/04 11:13:43  goesslet
//  retunr of GetMemUsage is unsigned long for 64bit
//
//  Revision 1.7  1999/12/22 15:17:34  goesslet
//  virtual destructor
//
//  Revision 1.6  1999/12/16 11:52:35  goesslet
//  implementation of methode GetMemoryUsage
//
//  Revision 1.5  1999/12/07 21:58:14  kickingf
//  Synchronisation Check in
//  Cleared Bug in ListArray and tried to remove leaked memory
//
//  Revision 1.4  1999/12/07 14:25:54  kickingf
//  Modificatoins for Groups in Command Gui
//
//  Revision 1.3  1999/11/11 12:20:26  kickingf
//  Added Method AppendList and InsertPosition
//
//  Revision 1.2  1999/11/09 13:19:45  bauert
//  Added this-> to all locals.
//
//  Revision 1.1  1999/11/08 13:24:39  goesslet
//  initial version
//
//----------------------------------------------------------------------

#ifndef LISTARRAY
#define LISTARRAY

#include "FC_Array.h"

template <class T>
class FC_ListArray : public FC_Array<T>
{
    IMPLEMENT_FIRE_BASE(FC_ListArray, FC_Array<T>)
protected:
    int M;

public:
    FC_ListArray()
    {
        this->M = 0;
    }
    FC_ListArray(int n)
    {
        this->Alloc(n);
        this->M = 0;
    }
    FC_ListArray(const FC_ListArray<T> &s);
    virtual ~FC_ListArray()
    {
        this->Free();
        this->M = 0;
    }
    FC_ListArray<T> &operator=(const FC_ListArray<T> &s);
    int Append(const T &t);
    void AppendList(const FC_ListArray<T> &a);
    int IsIn(const T &t) const;
    void Replace(T t1, T t2);
    void DeleteAll()
    {
        this->Free();
        this->Alloc(0);
        this->M = 0;
    }
    void DeletePosition(int n);
    void DeleteValue(const T &n);
    void SetNumElems(int i)
    {
        M = i;
    }
    int GetNumElems() const
    {
        return this->M;
    }
    void OptimizeMemory()
    {
        this->ReSetSize(this->M);
    }
    int InsertPosition(int p, const T &t);
    unsigned long GetMemoryUsage()
    {
        return sizeof(FC_ListArray<T>) + GetSize() * sizeof(T);
    }
};

template <class T>
inline int FC_ListArray<T>::InsertPosition(int p, const T &t)
{
    int i;
    if (this->M == this->N)
    {
        this->ReSetSize(2 * this->N + 1);
    }
    for (i = this->M - 1; i >= p; i--)
    {
        this->DATA[i + 1] = this->DATA[i];
    }
    this->DATA[p] = t;
    this->M++;
    return this->M;
}

template <class T>
inline void FC_ListArray<T>::AppendList(const FC_ListArray<T> &a)
{
    int i;
    for (i = 0; i < a.GetNumElems(); i++)
    {
        Append(a[i]);
    }
}

template <class T>
inline FC_ListArray<T>::FC_ListArray(const FC_ListArray<T> &s)
{
    this->N = s.GetNumElems();
    this->Alloc(this->N);
    this->M = s.GetNumElems();
    for (int i = 0; i < this->M; i++)
    {
        this->DATA[i] = s[i];
    }
}

template <class T>
inline FC_ListArray<T> &FC_ListArray<T>::operator=(const FC_ListArray<T> &s)
{
    if (this->N < s.GetNumElems())
    {
        this->Free();
        this->Alloc(s.GetNumElems());
    }
    this->M = s.GetNumElems();
    for (int i = 0; i < this->M; i++)
    {
        this->DATA[i] = s[i];
    }
    return (*this);
}

template <class T>
inline int FC_ListArray<T>::Append(const T &t)
{
    if (this->M == this->N)
    {
        this->ReSetSize(2 * this->N + 1);
    }
    this->DATA[this->M] = t;
    this->M++;
    return this->M;
}

template <class T>
inline void FC_ListArray<T>::Replace(T t1, T t2)
{
    int i;
    for (i = 0; i < this->M; i++)
    {
        if (this->DATA[i] == t1)
            this->DATA[i] = t2;
    }
}

template <class T>
inline int FC_ListArray<T>::IsIn(const T &t) const
{
    int i;
    for (i = 0; i < this->M; i++)
    {
        if (this->DATA[i] == t)
        {
            return i;
        }
    }
    return -1;
}

template <class T>
inline void FC_ListArray<T>::DeletePosition(int i)
{
    if (i < this->M)
    {
        if (i < this->M - 1)
        {
            this->DATA[i] = this->DATA[this->M - 1];
        }
        this->M--;
    }
}

template <class T>
inline void FC_ListArray<T>::DeleteValue(const T &t)
{
    int i;
    for (i = this->M - 1; i >= 0; i--)
    {
        if (this->DATA[i] == t)
            this->DeletePosition(i);
    }
}

template <class T>
inline ostream &operator<<(ostream &o, const FC_ListArray<T> &a)
{
    int i;

    o << a.GetNumElems() << "\n";
    for (i = 0; i < a.GetNumElems(); i++)
    {
        o << a[i] << " ";
    }
    o << "\n";
    return o;
}

template <class T>
inline istream &operator>>(istream &o, FC_ListArray<T> &a)
{
    int val;
    int i;
    T t;

    a.DeleteAll();
    o >> val;

    for (i = 0; i < val; i++)
    {

        o >> t;
        a.Append(t);
    }

    return o;
}

typedef FC_ListArray<int> FC_IntListArray;
typedef FC_ListArray<double> FC_DoubleListArray;
#endif
