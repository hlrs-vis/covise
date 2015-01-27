/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef KD_TREE_H
#define KD_TREE_H

#include <vector>
#include "bb.h"

#define MAX_BUCKET 32

bool inside(const float x, const float y, const float z, const BB &b)
{

    if (x >= b.minx && x <= b.maxx && y >= b.miny && y <= b.maxy && z >= b.minz && z <= b.maxz)
    {
        /*
      printf(" inside %f %f %f [ %f %f, %f %f, %f %f]\n", x, y, z,
             b.minx, b.maxx, b.miny, b.maxy, b.minz, b.maxz);
*/
        return true;
    }
    else
    {
        /*
      printf(" outside %f %f %f [ %f %f, %f %f, %f %f]\n", x, y, z,
             b.minx, b.maxx, b.miny, b.maxy, b.minz, b.maxz);
*/
        return false;
    }
}

void search(const std::vector<BB> *flat, const std::vector<int> *cells,
            std::vector<int> &result,
            const float x, const float y, const float z)
{

    std::list<int> to_search;

    to_search.push_back(0);

    while (!to_search.empty())
    {

        int idx = to_search.front();
        to_search.pop_front();
        const BB current = (*flat)[idx];
        if (inside(x, y, z, current))
        {
            int left = current.left;
            int right = current.right;
            if (left != -1)
                to_search.push_back(left);
            if (right != -1)
                to_search.push_back(right);
            if (left == -1 && right == -1)
            {
                int num = (*cells)[current.cells];
                for (int i = 0; i < num; i++)
                    result.push_back((*cells)[current.cells + i + 1]);
            }
        }
    }
}

bool inside(const float x, const float y, const float z, float b[6])
{
    if (x >= b[0] && x <= b[3] && y >= b[1] && y <= b[4] && z >= b[2] && z <= b[5])
    {
        /*
  printf(" inside %f %f %f [ %f %f, %f %f, %f %f]\n", x, y, z,
  b[0], b[3], b[1], b[4], b[2], b[5]);
*/
        return true;
    }
    else
    {
        /*
  printf(" outside %f %f %f [ %f %f, %f %f, %f %f]\n", x, y, z,
  b[0], b[3], b[1], b[4], b[2], b[5]);
*/
        return false;
    }
}

class Element
{
public:
    Element(const int _index, const float bbox[6])
        : index(_index)
    {
        for (int i = 0; i < 6; i++)
            this->bbox[i] = bbox[i];
    }

    // one of the edges is inside the box
    bool isPartOf(float b[6])
    {

        if (inside(bbox[0], bbox[1], bbox[2], b))
            return true;
        if (inside(bbox[3], bbox[1], bbox[2], b))
            return true;
        if (inside(bbox[0], bbox[4], bbox[2], b))
            return true;
        if (inside(bbox[3], bbox[4], bbox[2], b))
            return true;
        if (inside(bbox[0], bbox[1], bbox[5], b))
            return true;
        if (inside(bbox[3], bbox[1], bbox[5], b))
            return true;
        if (inside(bbox[0], bbox[4], bbox[5], b))
            return true;
        if (inside(bbox[3], bbox[4], bbox[5], b))
            return true;
        return false;
    }

    int index;
    float bbox[6];
};

int findAxis(const float *bbox)
{

    float sizex = bbox[3] - bbox[0];
    float sizey = bbox[4] - bbox[1];
    float sizez = bbox[5] - bbox[2];

    if (sizex > sizey)
    {
        if (sizex > sizez)
            return 0;
        else
            return 2;
    }
    else if (sizez > sizey)
    {
        if (sizez > sizex)
            return 2;
        else
            return 0;
    }
    else
        return 1;
}

class KD
{

public:
    KD()
    {
        for (int index = 0; index < 3; index++)
        {
            bbox[index] = FLT_MAX;
            bbox[index + 3] = -FLT_MAX;
        }
        subtrees[0] = NULL;
        subtrees[1] = NULL;
    }

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

    void insert(Element *elem)
    {

        for (int i = 0; i < 3; i++)
        {
            if (elem->bbox[i] < bbox[i])
                bbox[i] = elem->bbox[i];
            if (elem->bbox[i + 3] > bbox[i + 3])
                bbox[i + 3] = elem->bbox[i + 3];
        }
        elements.push_back(elem);
    }

    bool contains(const float x, const float y, const float z)
    {
        /*
      if (value >= val[0] && value <= val[1])
         return true;
*/
        if (bbox[MINX] <= x && bbox[MAXX] >= x && bbox[MINY] <= y && bbox[MAXY] >= y && bbox[MINZ] <= z && bbox[MAXZ] >= z)
            return true;

        return false;
    }

    void search(std::vector<int> &e, const float x, const float y, const float z) const
    {

        bool sub = false;
        for (int index = 0; index < 2; index++)
            if (subtrees[index] && subtrees[index]->contains(x, y, z))
            {
                subtrees[index]->search(e, x, y, z);
                sub = true;
            }

        if (!sub)
            for (int index = 0; index < elements.size(); index++)
                e.push_back(elements[index]->index);
    }

    int build(int level = 0)
    {

        printf("kd_tree::build(%d) %ld elements\n", level, elements.size());
        if (elements.size() > MAX_BUCKET)
        {

            bbox[MIDX] = bbox[MINX] + (bbox[MAXX] - bbox[MINX]) / 2;
            bbox[MIDY] = bbox[MINY] + (bbox[MAXY] - bbox[MINY]) / 2;
            bbox[MIDZ] = bbox[MINZ] + (bbox[MAXZ] - bbox[MINZ]) / 2;

            int bx[12] = { MINX, MINY, MINZ, MIDX, MAXY, MAXZ,
                           MIDX, MINY, MINZ, MAXX, MAXY, MAXZ };

            int by[12] = { MINX, MINY, MINZ, MAXX, MIDY, MAXZ,
                           MINX, MIDY, MINZ, MAXX, MAXY, MAXZ };

            int bz[12] = { MINX, MINY, MINZ, MAXX, MAXY, MIDZ,
                           MINX, MINY, MIDZ, MAXX, MAXY, MAXZ };

            float bb[12];

            // axis of subdivision
            int axis = findAxis(bbox);
            if (axis == 0)
                for (int index = 0; index < 12; index++)
                    bb[index] = bbox[bx[index]];
            else if (axis == 1)
                for (int index = 0; index < 12; index++)
                    bb[index] = bbox[by[index]];
            else
                for (int index = 0; index < 12; index++)
                    bb[index] = bbox[bz[index]];

            //printf("    subdivide axis %d\n", axis);
            this->axis = axis;
            subtrees[0] = new KD();
            subtrees[1] = new KD();

            for (int elem = 0; elem < elements.size(); elem++)
            {
                bool inserted = false;
                for (int part = 0; part < 2; part++)
                {
                    if (elements[elem]->isPartOf(&bb[part * 6]))
                    {
                        subtrees[part]->insert(elements[elem]);
                        inserted = true;
                    }
                }
                if (!inserted)
                    printf("didn't insert %d\n", elem);
            }
            elements.clear();
            /*
         printf("    distribution: %d %d\n", subtrees[0]->elements.size(),
                subtrees[1]->elements.size());
         
         printf("     part 0 bbox: %f %f %f %f %f %f\n", subtrees[0]->bbox[MINX], subtrees[0]->bbox[MAXX],
                subtrees[0]->bbox[MINY], subtrees[0]->bbox[MAXY], subtrees[0]->bbox[MINZ], subtrees[0]->bbox[MAXZ]);

         printf("     part 1 bbox: %f %f %f %f %f %f\n", subtrees[1]->bbox[MINX], subtrees[1]->bbox[MAXX],
                subtrees[1]->bbox[MINY], subtrees[1]->bbox[MAXY], subtrees[1]->bbox[MINZ], subtrees[1]->bbox[MAXZ]);
*/

            long long s0 = subtrees[0]->elements.size();
            long long s1 = subtrees[1]->elements.size();

            if (s0 > (s1 * 1000) || s1 > (s0 * 1000))
            {
                printf("# distribution is imbalanced [%lld/%lld], stopping\n", s0, s1);
            }
            else
            {
                int ll = subtrees[0]->build(level + 1);
                int lr = subtrees[1]->build(level + 1);
                level = ll > lr ? ll : lr;
            }
        }
        return level;
    }

    void vis(const int depth = 0)
    {

        for (int index = 0; index < depth; index++)
            printf("    ");
        printf(" (%f-%f %f-%f %f-%f)\n", bbox[0], bbox[3], bbox[1], bbox[4], bbox[2], bbox[5]);
        if (depth < 4 && subtrees[0])
            subtrees[0]->vis(depth + 1);
        if (depth < 4 && subtrees[1])
            subtrees[1]->vis(depth + 1);
    }

    void visflat(const std::vector<BB> &box, const int element = 0, const int depth = 0)
    {

        BB current = box[element];

        for (int index = 0; index < depth; index++)
            printf("    ");
        printf(" (%f-%f %f-%f %f-%f)\n", current.minx, current.maxx, current.miny, current.maxy, current.minz, current.maxz);
        if (depth < 4 && current.left != -1)
            visflat(box, current.left, depth + 1);
        if (depth < 4 && current.right != -1)
            visflat(box, current.right, depth + 1);
    }

    int flatten(std::vector<BB> *box, std::vector<int> *cells)
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

        int index = box->size();
        box->push_back(b);

        if (subtrees[0] && subtrees[1])
        {
            b.left = subtrees[0]->flatten(box, cells);
            b.right = subtrees[1]->flatten(box, cells);
            (*box)[index] = b;
        }
        else
        {
            int i = cells->size();
            (*box)[index].cells = i;
            cells->push_back(elements.size());
            for (int idx = 0; idx < elements.size(); idx++)
                cells->push_back(elements[idx]->index);
        }
        return index;
    }

    std::vector<Element *> elements;
    KD *subtrees[2];

    int axis;
    float bbox[9]; // minx, miny, minz, maxx, maxy, maxz, midx, midy, midz
};

#endif
