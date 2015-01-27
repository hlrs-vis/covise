/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <values.h>
#include <algorithm>
#include "utils.h"
#include "tree.h"

int check_same_clock_dir(float3 pt1, float3 pt2, float3 pt3, float3 norm)
{
    float testi, testj, testk;
    float dot;
    // normal of triangle

    testi = (((pt2.y - pt1.y) * (pt3.z - pt1.z)) - ((pt3.y - pt1.y) * (pt2.z - pt1.z)));
    testj = (((pt2.z - pt1.z) * (pt3.x - pt1.x)) - ((pt3.z - pt1.z) * (pt2.x - pt1.x)));
    testk = (((pt2.x - pt1.x) * (pt3.y - pt1.y)) - ((pt3.x - pt1.x) * (pt2.y - pt1.y)));

    // Dot product with triangle normal
    dot = testi * norm.x + testj * norm.y + testk * norm.z;

    //answer
    if (dot < 0)
        return DIFF_CLOCKNESS;
    else
        return SAME_CLOCKNESS;
}

bool check_intersect_tri(float3 pt1, float3 pt2, float3 pt3,
                         float3 linept, float3 vect,
                         float3 *pt_int)
{
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;
    float3 norm;
    float dot;
    float t;

    // vector form triangle pt1 to pt2
    v1x = pt2.x - pt1.x;
    v1y = pt2.y - pt1.y;
    v1z = pt2.z - pt1.z;

    // vector form triangle pt2 to pt3
    v2x = pt3.x - pt2.x;
    v2y = pt3.y - pt2.y;
    v2z = pt3.z - pt2.z;

    // vector normal of triangle
    norm.x = v1y * v2z - v1z * v2y;
    norm.y = v1z * v2x - v1x * v2z;
    norm.z = v1x * v2y - v1y * v2x;

    // dot product of normal and line's vector if zero line is parallel to triangle
    dot = norm.x * vect.x + norm.y * vect.y + norm.z * vect.z;

    if (dot < 0)
    {
        //Find point of intersect to triangle plane.
        //find t to intersect point
        t = -(norm.x * (linept.x - pt1.x) + norm.y * (linept.y - pt1.y) + norm.z * (linept.z - pt1.z)) / (norm.x * vect.x + norm.y * vect.y + norm.z * vect.z);

        // if ds is neg line started past triangle so can't hit triangle.
        if (t < 0)
            return 0;

        pt_int->x = linept.x + vect.x * t;
        pt_int->y = linept.y + vect.y * t;
        pt_int->z = linept.z + vect.z * t;

        if ((check_same_clock_dir(pt1, pt2, *pt_int, norm) == SAME_CLOCKNESS) && (check_same_clock_dir(pt2, pt3, *pt_int, norm) == SAME_CLOCKNESS) && (check_same_clock_dir(pt3, pt1, *pt_int, norm) == SAME_CLOCKNESS))
            return true;
    }
    return false;
}

Tree::Tree(struct usg *u)
    : left(NULL)
    , right(NULL)
    , usg(u)
{

    elements = new std::vector<int>;

    for (int index = 0; index < 3; index++)
    {
        bbox[index] = FLT_MAX;
        bbox[index + 3] = -FLT_MAX;
    }

    for (int elem = 0; elem < usg->numElements; elem++)
        insert(elem);
}

bool Tree::inside(const float x, const float y, const float z, const float *bb)
{
    return (x >= bb[0] && y >= bb[1] && z >= bb[2] && x <= bb[3] && y <= bb[4] && z <= bb[5]);
}

bool Tree::getIntersection(float fDst0, float fDst1, float3 p0, float3 p1,
                           float3 &hit)
{

    if ((fDst0 * fDst1) > 0.0f)
        return false;

    if (fDst0 == fDst1)
        return false;
    hit = p0 + (p1 - p0) * (-fDst0 / (fDst1 - fDst0));

    return true;
}

bool Tree::inBox(float3 hit, float3 b0, float3 b1, const int axis)
{

    if (axis == 1 && hit.z > b0.z && hit.z < b1.z && hit.y > b0.y && hit.y < b1.y)
        return true;

    if (axis == 2 && hit.z > b0.z && hit.z < b1.z && hit.x > b0.x && hit.x < b1.x)
        return true;

    if (axis == 3 && hit.x > b0.x && hit.x < b1.x && hit.y > b0.y && hit.y < b1.y)
        return true;

    return false;
}

// returns true if line (p0, p1) intersects with the box (B0, B1)
// returns intersection point in hit
bool Tree::checkLineBox(float3 b0, float3 b1, float3 p0, float3 p1, float3 &hit)
{
    if (p1.x < b0.x && p0.x < b0.x)
        return false;
    if (p1.x > b1.x && p0.x > b1.x)
        return false;
    if (p1.y < b0.y && p0.y < b0.y)
        return false;
    if (p1.y > b1.y && p0.y > b1.y)
        return false;
    if (p1.z < b0.z && p0.z < b0.z)
        return false;
    if (p1.z > b1.z && p0.z > b1.z)
        return false;
    if (p0.x > b0.x && p0.x < b1.x && p0.y > b0.y && p0.y < b1.y && p0.z > b0.z && p0.z < b1.z)
    {
        hit = p0;
        return true;
    }
    if ((getIntersection(p0.x - b0.x, p1.x - b0.x, p0, p1, hit) && inBox(hit, b0, b1, 1))
        || (getIntersection(p0.y - b0.y, p1.y - b0.y, p0, p1, hit) && inBox(hit, b0, b1, 2))
        || (getIntersection(p0.z - b0.z, p1.z - b0.z, p0, p1, hit) && inBox(hit, b0, b1, 3))
        || (getIntersection(p0.x - b1.x, p1.x - b1.x, p0, p1, hit) && inBox(hit, b0, b1, 1))
        || (getIntersection(p0.y - b1.y, p1.y - b1.y, p0, p1, hit) && inBox(hit, b0, b1, 2))
        || (getIntersection(p0.z - b1.z, p1.z - b1.z, p0, p1, hit) && inBox(hit, b0, b1, 3)))
        return true;

    return false;
}

bool Tree::inside(int elem, float *bb)
{

    float3 b0 = make_float3(bb[0], bb[1], bb[2]);
    float3 b1 = make_float3(bb[3], bb[4], bb[5]);

    int start = usg->elementList[elem];
    int end;
    if (elem < usg->numElements - 1)
        end = usg->elementList[elem + 1] - 1;
    else
        end = usg->numCorners - 1;

    int lines[12][2] = { { 0, 1 },
                         { 0, 3 },
                         { 0, 4 },
                         { 4, 5 },
                         { 4, 7 },
                         { 1, 2 },
                         { 1, 5 },
                         { 5, 6 },
                         { 2, 3 },
                         { 2, 6 },
                         { 3, 7 },
                         { 6, 7 } };

    for (int line = 0; line < 12; line++)
    {

        int l = start + lines[line][0];
        int v = start + lines[line][1];

        float3 lp = make_float3(usg->x[usg->cornerList[l]],
                                usg->y[usg->cornerList[l]],
                                usg->z[usg->cornerList[l]]);
        float3 vp = make_float3(usg->x[usg->cornerList[v]],
                                usg->y[usg->cornerList[v]],
                                usg->z[usg->cornerList[v]]);

        float3 hit;
        if (checkLineBox(b0, b1, lp, vp, hit))
            return true;
    }
    return false;
    /*
      inside(xp[cornerList[start]], yp[cornerList[start]],
                 zp[cornerList[start]], bb);
   */
}

void Tree::insert(int elem)
{

    elements->push_back(elem);

    int start = usg->elementList[elem];
    int end;
    if (elem < usg->numElements - 1)
        end = usg->elementList[elem + 1] - 1;
    else
        end = usg->numCorners - 1;

    for (int corner = start; corner <= end; corner++)
    {
        if (usg->x[usg->cornerList[corner]] < bbox[0])
            bbox[0] = usg->x[usg->cornerList[corner]];
        if (usg->y[usg->cornerList[corner]] < bbox[1])
            bbox[1] = usg->y[usg->cornerList[corner]];
        if (usg->z[usg->cornerList[corner]] < bbox[2])
            bbox[2] = usg->z[usg->cornerList[corner]];

        if (usg->x[usg->cornerList[corner]] > bbox[3])
            bbox[3] = usg->x[usg->cornerList[corner]];
        if (usg->y[usg->cornerList[corner]] > bbox[4])
            bbox[4] = usg->y[usg->cornerList[corner]];
        if (usg->z[usg->cornerList[corner]] > bbox[5])
            bbox[5] = usg->z[usg->cornerList[corner]];
    }
}

int Tree::findAxis(float *bb)
{

    float sizex = bb[3] - bb[0];
    float sizey = bb[4] - bb[1];
    float sizez = bb[5] - bb[2];

    if (sizex > sizey)
    {
        if (sizex > sizez)
            return 0;
        else
            return 2;
    }
    else if (sizez > sizey)
    {
        return 2;
    }
    else
        return 1;
}

static int maxlevel = 0;

bool Tree::build(int level)
{

    if (level > maxlevel)
    {
        maxlevel = level;
        printf("  subdivide(%d)\n", level);
    }
    /*
     printf("#VRML V2.0 utf8\n\n");


     printf("   bbox: ( ");
     for (int index = 0; index < 6; index ++)
     printf("%f ", bbox[index]);
     printf(")\n");
   */
    left = new Tree(this);
    right = new Tree(this);

    bbox[MIDX] = bbox[MINX] + (bbox[MAXX] - bbox[MINX]) / 2;
    bbox[MIDY] = bbox[MINY] + (bbox[MAXY] - bbox[MINY]) / 2;
    bbox[MIDZ] = bbox[MINZ] + (bbox[MAXZ] - bbox[MINZ]) / 2;

    int axis = findAxis(bbox);

    int bx[12] = { MINX, MINY, MINZ, MIDX, MAXY, MAXZ,
                   MIDX, MINY, MINZ, MAXX, MAXY, MAXZ };

    int by[12] = { MINX, MINY, MINZ, MAXX, MIDY, MAXZ,
                   MINX, MIDY, MINZ, MAXX, MAXY, MAXZ };

    int bz[12] = { MINX, MINY, MINZ, MAXX, MAXY, MIDZ,
                   MINX, MINY, MIDZ, MAXX, MAXY, MAXZ };

    float bb[12];
    /*
   SortXYZ s(usg, axis);
   std::sort(elements->begin(), elements->end(), s);
   */
    if (axis == 0)
    {
        for (int index = 0; index < 12; index++)
            bb[index] = bbox[bx[index]];
        /*
      float d = usg->boundingSpheres[(*elements)[elements->size() / 2] * 4];
      bb[3] = d;
      bb[6] = d;
      */
    }
    else if (axis == 1)
    {
        for (int index = 0; index < 12; index++)
            bb[index] = bbox[by[index]];
        /*
      float d = usg->boundingSpheres[(*elements)[elements->size() / 2] * 4 + 1];
      bb[4] = d;
      bb[7] = d;
      */
    }
    else
    {
        for (int index = 0; index < 12; index++)
            bb[index] = bbox[bz[index]];
        /*
      float d = usg->boundingSpheres[(*elements)[elements->size() / 2] * 4 + 2];
      bb[5] = d;
      bb[8] = d;
      */
    }

    /*
     printf("    axis %d\n", axis);
     printf("     left bbox  : ( ");
     for (int index = 0; index < 6; index ++)
     printf("%f ", bb[index]);
     printf(")\n");

     printf("     right bbox : ( ");
     for (int index = 0; index < 6; index ++)
     printf("%f ", bb[index + 6]);
     printf(")\n");
   */
    // populate
    std::vector<int>::iterator i;
    for (i = elements->begin(); i != elements->end(); i++)
    {

        if (inside(*i, bb))
        {
            left->insert(*i);
        }
        else if (inside(*i, bb + 6))
            right->insert(*i);
        else
        {
            if (!elements->size() >= 4 * MAXBUCKET)
                printf("!!!    element is not in left or right subtree\n");
            return false;
        }
    }

    if (left->elements->size() < MAXBUCKET / 3.0 || right->elements->size() < MAXBUCKET / 3.0)
        return false;

    /*
   if (left->elements->size() > right->elements->size() * 10.0 ||
       right->elements->size() > left->elements->size() * 10.0)
      printf("------- imbalance: %d/%d\n",
             left->elements->size(), right->elements->size());
   */

    //printf("      children (%d/%d)\n", left->elements->size(), right->elements->size());
    /*
     printf("left %d right %d\n", left->elements->size(),
     right->elements->size());

     printf("     left bbox  : ( ");
     for (int index = 0; index < 6; index ++)
     printf("%f ", left->bbox[index]);
     printf(")\n");
   */

    if (elements->size() > MAXBUCKET)
    {
        if (!left->build(level + 1))
        {
            delete left;
            left = NULL;
        }
        if (!right->build(level + 1))
        {
            delete right;
            right = NULL;
        }
    }
    //printBB(bbox);
    return true;
}

int Tree::flatten(std::vector<BB> &box, std::vector<int> &cells)
{

    BB b;
    b.minx = bbox[0];
    b.miny = bbox[1];
    b.minz = bbox[2];
    b.maxx = bbox[3];
    b.maxy = bbox[4];
    b.maxz = bbox[5];
    b.right = -1;
    b.left = -1;

    int index = box.size();
    box.push_back(b);

    if (left && right)
    {
        b.left = left->flatten(box, cells);
        b.right = right->flatten(box, cells);
        box[index] = b;
    }
    else
    {
        int i = cells.size();
        box[index].cells = i;
        cells.push_back(elements->size());
        for (unsigned int idx = 0; idx < elements->size(); idx++)
            cells.push_back((*elements)[idx]);
    }
    return index;
}

Tree::Tree(Tree *parent)
    : left(NULL)
    , right(NULL)
    , usg(parent->usg)
{

    for (int index = 0; index < 3; index++)
    {
        bbox[index] = FLT_MAX;
        bbox[index + 3] = -FLT_MAX;
    }
    elements = new std::vector<int>;
}
