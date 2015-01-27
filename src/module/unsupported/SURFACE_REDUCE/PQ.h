/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PQ_H
#define _PQ_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Priority queue class template                     **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  December 1997  V1.0                                             **
\**************************************************************************/
#include <math.h>

class Vertex_with_coords
{
private:
    int key;
    int type;
    double co_x;
    double co_y;

public:
    int is_less(Vertex_with_coords b)
    {
        return (co_y - b.co_y >= 1E-06 || ((fabs(co_y - b.co_y) < 1E-06) && (co_x < b.co_x)));
    }
    void set_key(float a)
    {
        key = a;
    }
    int get_key()
    {
        return (key);
    }
};

class Vertex
{
private:
    int key;
    float weight;

public:
    int is_less(Vertex b)
    {
        return (weight < b.weight);
    }
    void set_key(float a)
    {
        weight = a;
    }
    float get_key()
    {
        return (weight);
    }
};

class Edge
{
private:
    int v1;
    int v2;
    float length;

public:
    int is_less(Edge b)
    {
        return (length < b.length);
    }
    void set_key(float a)
    {
        length = a;
    }
    float get_key()
    {
        return (length);
    }
    int get_v1()
    {
        return (v1);
    }
    int get_v2()
    {
        return (v2);
    }
    float get_length()
    {
        return (length);
    }
    void set_v1(int v)
    {
        v1 = v;
    }
    void set_v2(int v)
    {
        v2 = v;
    }
    void set_endpoints(int x1, int x2)
    {
        v1 = x1;
        v2 = x2;
    }
};

class Triangle
{
private:
    int key;
    float weight;

public:
    int is_less(Triangle b)
    {
        return (weight < b.weight);
    }
    void set_key(float a)
    {
        weight = a;
    }
    float get_key()
    {
        return (weight);
    }
};

#ifdef CO_linux
#pragma interface
#endif

template <class itemtype>
class PQ
{
private:
    itemtype *q;
    int *ind;
    int *info;
    int N;

    int is_less(itemtype a, itemtype b)
    {
        return (a.is_less(b));
    }
    int is_less(int k1, int k2)
    {
        return (is_less(q[k1], q[k2]));
    }

public:
    PQ(int size)
    {
        N = 0;
        q = new itemtype[size + 1];
        ind = new int[size + 1];
        info = new int[size + 1];
        info[0] = ind[0] = 0;
    }
    ~PQ()
    {
        delete[] q;
        delete[] ind;
        delete[] info;
    }
    void insert(itemtype e);
    void append(itemtype e);
    itemtype get_next();
    int get_index(int k);
    int getSize();
    itemtype get_item(int k);
    void set_item(int k, itemtype e);
    void downheap(int k);
    void upheap(int k);
    void construct();
    void remove(int k);
    void change(int k, itemtype e);
};

#ifndef CO_linux_2

template <class itemtype>
void PQ<itemtype>::upheap(int k)
{
    int i;

    i = info[k];
    while (k / 2 != 0 && !is_less(info[k / 2], i))
    {
        ind[info[k / 2]] = k;
        info[k] = info[k / 2];
        k = k / 2;
    }
    ind[i] = k;
    info[k] = i;
}

template <class itemtype>
void PQ<itemtype>::downheap(int k)
{
    int j, l;

    j = info[k];
    while (k <= N / 2)
    {
        l = k + k;
        if (l < N && is_less(info[l + 1], info[l]))
            l++;
        if (!is_less(info[l], j))
            break;
        ind[info[l]] = k;
        info[k] = info[l];
        k = l;
    }
    ind[j] = k;
    info[k] = j;
}

template <class itemtype>
itemtype PQ<itemtype>::get_next()
{
    itemtype e;
    int i;
    if (N >= 1)
    {
        e = q[info[1]];
        ind[info[N]] = ind[info[1]];
        ind[info[1]] = N;
        i = info[1];
        info[1] = info[N];
        info[N--] = i;
        downheap(1);
    }
    else
        e.set_key(-1);
    return (e);
}

template <class itemtype>
int PQ<itemtype>::get_index(int k)
{
    return ind[k];
}

template <class itemtype>
itemtype PQ<itemtype>::get_item(int k)
{
    return q[k];
}

template <class itemtype>
void PQ<itemtype>::set_item(int k, itemtype e)
{
    q[k] = e;
}

template <class itemtype>
int PQ<itemtype>::getSize()
{
    return (N);
}

template <class itemtype>
void PQ<itemtype>::append(itemtype e)
{
    q[++N] = e;
    ind[N] = N;
    info[N] = N;
}

template <class itemtype>
void PQ<itemtype>::insert(itemtype e)
{
    q[++N] = e;
    ind[N] = N;
    info[N] = N;
    upheap(N);
}

template <class itemtype>
void PQ<itemtype>::remove(int k)
{
    int i;
    ind[info[N]] = k;
    ind[info[k]] = N;
    i = info[k];
    info[k] = info[N];
    info[N--] = i;
    downheap(k);
}

template <class itemtype>
void PQ<itemtype>::construct()
{
    int k;

    for (k = N / 2; k >= 1; k--)
        downheap(k);
}

template <class itemtype>
void PQ<itemtype>::change(int k, itemtype e)
{
    itemtype previous = q[info[k]];

    q[info[k]] = e;
    if (is_less(previous, e))
        downheap(k);
    else
        upheap(k);
}
#endif
#endif // _PQ_H
