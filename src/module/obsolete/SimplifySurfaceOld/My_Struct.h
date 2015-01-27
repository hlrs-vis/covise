/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MYSTRUCT_H
#define _MYSTRUCT_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  Data structures for the triangulation of simple polygons **
 **               without holes (Header file)                              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  September 1997  V1.0                                            **
\**************************************************************************/
#include <util/coviseCompat.h>

class Stack
{
private:
    int *stack;
    int p;

public:
    Stack(int max)
    {
        stack = new int[max];
        p = 0;
    }
    ~Stack()
    {
        delete[] stack;
    }
    inline void push(int v)
    {
        stack[p++] = v;
    }
    inline int pop()
    {
        return (stack[--p]);
    }
    inline int empty()
    {
        return (!p);
    }
};

class Tree
{
private:
    struct knot
    {
        int edge;
        int helper;
        knot *left;
        knot *right;
    };

    struct knot *root;

    double *co_x;
    double *co_y;
    int N;

    double get_x(struct knot *k, double sweep);

public:
    Tree(int num, float (*coords)[2])
    {
        int i;
        root = NULL;
        N = num;
        co_x = new double[num];
        co_y = new double[num];
        for (i = 0; i < num; i++)
        {
            co_x[i] = (double)coords[i][0];
            co_y[i] = (double)coords[i][1];
        }
    }

    Tree(int num, double (*coords)[2])
    {
        int i;
        root = NULL;
        N = num;
        co_x = new double[num];
        co_y = new double[num];
        for (i = 0; i < num; i++)
        {
            co_x[i] = coords[i][0];
            co_y[i] = coords[i][1];
        }
    }

    ~Tree()
    {
        if (root != NULL)
            delete root;
        delete[] co_x;
        delete[] co_y;
    }

    void insert(int ed, int help, double sweep);
    int get_helper(int ed, double sweep);
    void set_helper(int ed, int help, double sweep);
    int get_left_neighbour(int v, double sweep);
    int get_left_neighbour(double x_v, double sweep);
    void remove(int ed, double sweep);
};
#endif // _MYSTRUCT_H
