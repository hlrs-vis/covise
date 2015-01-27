/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TREE_H
#define TREE_H

#include <vector>

#include <cutil.h>
#include <cutil_math.h>

#include "bb.h"

#define SAME_CLOCKNESS 1
#define DIFF_CLOCKNESS 0

#define MAXBUCKET 32

class SortXYZ
{
public:
    SortXYZ(struct usg *u, int a)
        : usg(u)
        , axis(a)
    {
    }

    bool operator()(int i, int j)
    {
        switch (axis)
        {
        case 0:
            return (usg->boundingSpheres[i * 4] < usg->boundingSpheres[j * 4]);
        case 1:
            return (usg->boundingSpheres[i * 4 + 1] < usg->boundingSpheres[j * 4 + 1]);
        case 2:
            return (usg->boundingSpheres[i * 4 + 2] < usg->boundingSpheres[j * 4 + 2]);
        }
        return false;
    }

    struct usg *usg;
    int axis;
};

class Tree
{

    enum
    {
        MINX = 0,
        MINY = 1,
        MINZ = 2,
        MAXX = 3,
        MAXY = 4,
        MAXZ = 5,
        MIDX = 6,
        MIDY = 7,
        MIDZ = 8
    };

public:
    Tree *left, *right;

    Tree(struct usg *u);

    bool inside(int elem, float *bb);
    bool inside(const float x, const float y, const float z, const float *bb);

    void insert(int elem);
    int findAxis(float *bbox);

    bool build(int level = 0);

    bool getIntersection(float fDst0, float fDst1, float3 p0, float3 p1,
                         float3 &hit);
    bool inBox(float3 hit, float3 b1, float3 b2, const int axis);
    bool checkLineBox(float3 b0, float3 b1, float3 p0, float3 p1, float3 &hit);
    int flatten(std::vector<BB> &box, std::vector<int> &cells);

    Tree(Tree *parent);

    struct usg *usg;
    float bbox[9];

    std::vector<int> *elements;
};

#endif
