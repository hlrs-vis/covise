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
#include <util/coviseCompat.h>

class Vertex_with_coords
{
private:
    int key;
    int type;
    double co_x;
    double co_y;

public:
    inline int is_less(Vertex_with_coords b)
    {
        return (co_y - b.co_y >= 1E-06 || ((fabs(co_y - b.co_y) < 1E-06) && (co_x < b.co_x)));
    }
    inline void set_key(int a)
    {
        key = a;
    }
    inline int get_key()
    {
        return (key);
    }
    inline double get_x()
    {
        return (co_x);
    }
    inline void set_x(double a)
    {
        co_x = a;
    }
    inline double get_y()
    {
        return (co_y);
    }
    inline void set_y(double a)
    {
        co_y = a;
    }
    inline void set_type(int a)
    {
        type = a;
    }
    inline int getType()
    {
        return (type);
    }
};

class Vertex
{
private:
    int key;
    float weight;

public:
    inline int is_less(Vertex b)
    {
        return (weight < b.weight);
    }
    inline void set_key(int a)
    {
        key = a;
    }
    inline float get_key()
    {
        return (key);
    }
    inline void set_weight(float a)
    {
        weight = a;
    }
    inline float get_weight()
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
    inline int is_less(Edge b)
    {
        return (length < b.length);
    }
    inline void set_length(float a)
    {
        length = a;
    }
    inline float get_length()
    {
        return (length);
    }
    inline int get_v1()
    {
        return (v1);
    }
    inline int get_v2()
    {
        return (v2);
    }
    inline void set_v1(int v)
    {
        v1 = v;
    }
    inline void set_v2(int v)
    {
        v2 = v;
    }
    inline void set_endpoints(int x1, int x2)
    {
        v1 = x1;
        v2 = x2;
    }
    inline void set_key(int a)
    {
        v1 = a;
    }
};

class Triangle
{
private:
    int key;
    float weight;

public:
    inline int is_less(Triangle b)
    {
        return (weight < b.weight);
    }
    inline void set_weight(float a)
    {
        weight = a;
    }
    inline float get_weight()
    {
        return (weight);
    }
};

/////////////////////////////////////////////////////////////////////////////
// This class is an implementation of the heap structured priority queue   //
// described in R. Sedgewick: Algorithms in C++, Chapter 11.               //
// It demands that the class used as <itemtype> has the public member      //
// functions:                                                              //
// inline int is_less(itemtype a)    returns true iff the weight of the    //
//                                   is less than the weight of the item a //
// inline void set_key(float x)      needed to return an item with a key   //
//                                   -1 (error value), if get_next() fails //
//                                   because the queue is empty            //
/////////////////////////////////////////////////////////////////////////////
// The heap is double-indexed. That means that the array ind[1...size]     //
// contains the information about the position of items in the PQ, while   //
// the array info[1...size] contains information about the keys of items   //
// in the PQ. Example:                                                     //
// ind[10] = 1  : The item with key 10 is the first element of the PQ      //
//                (has the lowest weight, is the first to be taken out)    //
// info[2] = 25 : On the second position of the PQ stands the item with    //
//                the key 25.                                              //
/////////////////////////////////////////////////////////////////////////////

template <class itemtype>
class PQ
{
private:
    itemtype *q;
    int *ind;
    int *info;
    int N;

    inline int is_less(itemtype a, itemtype b)
    {
        return (a.is_less(b));
    }
    inline int is_less(int k1, int k2)
    {
        return (is_less(q[k1], q[k2]));
    }

    // Shuffles the item on position k as far down the heap as necessary.
    void downheap(int k)
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

    // Shuffles the item on position k as far up the heap as necessary.
    void upheap(int k)
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

    // Returns the position of the element with the key k in the PQ.
    int get_index(int k)
    {
        return ind[k];
    }

    // Returns the size of the heap (number of items in the PQ).
    int getSize()
    {
        return (N);
    }

    // Returns the item which stands on the k-th position of the PQ.
    itemtype get_item(int k)
    {
        return q[k];
    }

    // Assigns the item e to the k-th position of the PQ.
    void set_item(int k, itemtype e)
    {
        q[k] = e;
    }

    // Appends a new element to the PQ, but does not put it on the right
    // position according to its weight. Effort: O(1)
    // CAUTION: Please use this function for the initial construction
    // of the heap ONLY, in connection with the construct() function,
    // otherwise it overwrites items in the array q!!!
    void append(itemtype e)
    {
        q[++N] = e;
        ind[N] = N;
        info[N] = N;
    }

    // Appends a new element to the PQ and puts it on the right position
    // according to its weight. Effort: O(log N)
    // CAUTION: Please use this function in the construction phase of the
    // heap ONLY!!! Otherwise it overwrites the array q in inappropriate
    // places, resulting in doubled entries etc.
    //void insert(itemtype e)
    //{ q[++N] = e;
    //  info[N} = ind[N] = N;
    //  upheap(N);
    //}

    // The heapsort procedure. A Bottom-Up construction of the heap.
    // Effort: O(N)
    void construct()
    {
        int k;
        for (k = N / 2; k >= 1; k--)
            downheap(k);
    }

    // Returns the item on the first position of the heap (the item with
    // the smallest weight, the highest priority) and removes it
    // from the PQ.
    itemtype get_next()
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

    // Removes the item with the position number k from the priority queue.
    void remove(int k)
    {
        int i;
        ind[info[N]] = k;
        ind[info[k]] = N;
        i = info[k];
        info[k] = info[N];
        info[N--] = i;
        downheap(k);
    }

    // Changes the item on the k-th position of the PQ (usually employed to
    // change the weight of an item) and corrects the position of the item
    // in the priority queue.
    void change(int k, itemtype e)
    {
        itemtype previous = q[info[k]];

        q[info[k]] = e;
        if (is_less(previous, e))
            downheap(k);
        else
            upheap(k);
    }

    // Changes an item which was already taken out of the heap, but has
    // to be freshly inserted, and corrects the position of the item
    // in the priority queue. k is the position of the item (which is then
    // greater than the current heapsize), and x its place in the physical
    // structure q.
    void insert_again(int k, int x, itemtype e)
    {
        set_item(x, e);
        // increase heapsize
        N++;
        //swap elements on positions k and N
        ind[info[N]] = k;
        info[k] = info[N];

        upheap(N);
    }
};
#endif // _PQ_H
