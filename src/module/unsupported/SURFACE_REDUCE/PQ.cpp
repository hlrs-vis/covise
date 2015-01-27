/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
#include "PQ.h"

#ifdef CO_linux_2

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
