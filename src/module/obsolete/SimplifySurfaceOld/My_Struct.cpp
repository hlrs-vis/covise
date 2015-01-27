/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  Data structures for the triangulation of simple polygons **
 **               without holes                                            **
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
#include "My_Struct.h"

////////////////////////////////////////////////////////////////////////////////
// BINARY SEARCH TREE FOR EDGES                                               //
////////////////////////////////////////////////////////////////////////////////

double Tree::get_x(struct knot *k, double sweep)
{
    double line[3];

    line[0] = co_y[(k->edge + 1) % N] - co_y[k->edge];
    line[1] = -(co_x[(k->edge + 1) % N] - co_x[k->edge]);
    line[2] = line[0] * co_x[k->edge] + line[1] * co_y[k->edge];

    if (line[0] > 1E-06 || line[0] < -1E-06)
        return ((line[2] - line[1] * sweep) / line[0]);
    else
        return (co_x[k->edge]);
}

void Tree::insert(int ed, int help, double sweep)
{
    struct knot *k, *k_cur, *father;
    double x_s, x;

    k = new knot;
    k->edge = ed;
    k->helper = help;
    k->left = NULL;
    k->right = NULL;

    x_s = get_x(k, sweep);

    if (root == NULL)
    {
        root = k;
        root->left = NULL;
        root->right = NULL;
    }
    else
    {
        k_cur = root;
        while (k_cur != NULL)
        {
            x = get_x(k_cur, sweep);
            father = k_cur;
            if (x > x_s)
                k_cur = k_cur->left;
            else
                k_cur = k_cur->right;
        }
        if (x > x_s)
            father->left = k;
        else
            father->right = k;
    }
}

int Tree::get_helper(int ed, double sweep)
{
    struct knot *k1, *k;
    double x, x_v;

    k1 = new knot;
    k1->edge = ed;

    x_v = get_x(k1, sweep);

    delete k1;

    k = root;
    while (k != NULL && k->edge != ed)
    {
        x = get_x(k, sweep);
        if (x > x_v)
            k = k->left;
        else
            k = k->right;
    }

    if (k == NULL)
    {
        cout << "Edge not found!!!" << endl;
        return (-1);
    }

    return (k->helper);
}

void Tree::set_helper(int ed, int help, double sweep)
{
    struct knot *k1, *k;
    double x, x_v;

    k1 = new knot;
    k1->edge = ed;

    x_v = get_x(k1, sweep);
    delete k1;

    k = root;
    while (k != NULL && k->edge != ed)
    {
        x = get_x(k, sweep);
        if (x > x_v)
            k = k->left;
        else
            k = k->right;
    }
    if (k == NULL)
    {
        cout << "Edge not found!!!" << endl;
        return;
    }

    k->helper = help;
}

int Tree::get_left_neighbour(double x_v, double sweep)
{
    struct knot *k;
    double x;
    int found;

    k = root;
    found = 0;
    while (k != NULL && !found)
    {
        x = get_x(k, sweep);
        if (x < x_v)
        {
            if (k->right != NULL)
            {
                x = get_x(k->right, sweep);
                if (x < x_v)
                    k = k->right;
                else
                    found = 1;
            }
            else
                found = 1;
        }
        else
            k = k->left;
    }
    if (k == NULL)
        return (-1);
    else
        return (k->edge);
}

int Tree::get_left_neighbour(int v, double sweep)
{
    struct knot *k;
    double x;
    double x_v;
    int found;

    x_v = co_x[v];

    k = root;
    found = 0;
    while (k != NULL && !found)
    {
        x = get_x(k, sweep);
        if (x < x_v)
        {
            if (k->right != NULL)
            {
                x = get_x(k->right, sweep);
                if (x < x_v)
                    k = k->right;
                else
                    found = 1;
            }
            else
                found = 1;
        }
        else
            k = k->left;
    }
    if (k == NULL)
        return (-1);
    else
        return (k->edge);
}

void Tree::remove(int ed, double sweep)
{
    double x, x_cur;
    struct knot *k, *father = NULL, *new_start, *old_start;

    k = new knot;
    k->edge = ed;

    x = get_x(k, sweep);
    delete k;

    k = root;
    while (k != NULL && k->edge != ed)
    {
        x_cur = get_x(k, sweep);
        father = k;
        if (x > x_cur)
            k = k->right;
        else
            k = k->left;
    }

    if (k == NULL)
    {
        cout << "Edge not found!!!" << endl;
        return;
    }

    if (root == k && k->right == NULL && k->left == NULL)
    {
        root = NULL;
        delete k;
        return;
    }

    if (k->right == NULL)
    {
        if (root == k)
            root = k->left;
        else
        {
            if (father->left == k)
                father->left = k->left;
            else
                father->right = k->left;
        }
    }
    else
    {
        if (k->left == NULL)
        {
            if (root == k)
                root = k->right;
            else
            {
                if (father->left == k)
                    father->left = k->right;
                else
                    father->right = k->right;
            }
        }
        else
        {
            new_start = k->right;
            old_start = k;
            while (new_start->left != NULL)
            {
                old_start = new_start;
                new_start = new_start->left;
            }
            if (old_start == k)
            {
                new_start->left = k->left;
                if (root == k)
                    root = new_start;
                else if (father->left == k)
                    father->left = new_start;
                else
                    father->right = new_start;
            }
            else
            {
                old_start->left = new_start->right;
                new_start->right = k->right;
                new_start->left = k->left;
                if (root == k)
                    root = new_start;
                else if (father->left == k)
                    father->left = new_start;
                else
                    father->right = new_start;
            }
        }
    }
    delete k;
}
