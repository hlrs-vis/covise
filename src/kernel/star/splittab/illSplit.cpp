/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <fstream.h>
#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "Hexa.h"
#include "Prism.h"
#include "Pyra.h"
#include "Tetra.h"
#include "Shape.h"

inline int eq(int v0, int v1, int v2, int v3)
{
    return v0 == v1 && v1 == v2 && v2 == v3;
}

inline int eq(int v0, int v1, int v2)
{
    return v0 == v1 && v1 == v2;
}

//// condensed for 1-part shapes
void print(int mask, const Shape &shape, ostream &str)
{
    str << "\n   // ============ mask= " << mask << ": [ "
        << ((mask & 1) ? "0-1 " : "")
        << ((mask & 2) ? "1-2 " : "")
        << ((mask & 4) ? "2-3 " : "")
        << ((mask & 8) ? "0-3 " : "")
        << ((mask & 16) ? "4-5 " : "")
        << ((mask & 32) ? "5-6 " : "")
        << ((mask & 64) ? "6-7 " : "")
        << ((mask & 128) ? "4-7 " : "")
        << ((mask & 256) ? "0-4 " : "")
        << ((mask & 512) ? "1-5 " : "")
        << ((mask & 1024) ? "2-6 " : "")
        << ((mask & 2048) ? "3-7 " : "")
        << "]" << endl;
    str << "   { 1,\n      { ";
    shape.printForm(str);
    str << " }"
        << "  // " << shape << endl;
    str << "   },\n" << endl;
}

void print(int mask, const Shape *const *shape, int num, ostream &str)
{
    str << "\n   // ============ mask= " << mask << ": [ "
        << ((mask & 1) ? "0-1 " : "")
        << ((mask & 2) ? "1-2 " : "")
        << ((mask & 4) ? "2-3 " : "")
        << ((mask & 8) ? "0-3 " : "")
        << ((mask & 16) ? "4-5 " : "")
        << ((mask & 32) ? "5-6 " : "")
        << ((mask & 64) ? "6-7 " : "")
        << ((mask & 128) ? "4-7 " : "")
        << ((mask & 256) ? "0-4 " : "")
        << ((mask & 512) ? "1-5 " : "")
        << ((mask & 1024) ? "2-6 " : "")
        << ((mask & 2048) ? "3-7 " : "")
        << "]" << endl;
    str << "   { " << num << ",\n      {\n";
    int i;
    for (i = 0; i < num; i++)
    {
        str << "         ";
        shape[i]->printForm(str);
        if (i < num - 1)
            str << ",";
        else
            str << " ";

        str << "  // " << *shape[i] << endl;
    }
    str << "      }\n   },\n" << endl;
    ;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void setSame(int vert[8], int oldVal, int newVal)
{
    int i;
    for (i = 0; i < 8; i++)
        if (vert[i] == oldVal)
            vert[i] = newVal;
}

int main(int argc, char *argv[])
{
    //sleep (10);
    int mask;

    ofstream splitFile("IllSplit.inc");
    splitFile << "// Split.h auto-generated\n" << endl;

    // Create all possible cases; 12 edges may miss
    for (mask = 0; mask < 4096; mask++)
    //for (mask=3 ; mask<4 ; mask++)
    {
        int vert[8] = // up to now, all are different
            {
              0, 1, 2, 3, 4, 5, 6, 7
            };

        // eliminate edges according to mask
        if (mask & 1)
            setSame(vert, vert[1], vert[0]);
        if (mask & 2)
            setSame(vert, vert[2], vert[1]);
        if (mask & 4)
            setSame(vert, vert[3], vert[2]);
        if (mask & 8)
            setSame(vert, vert[3], vert[0]);

        if (mask & 16)
            setSame(vert, vert[5], vert[4]);
        if (mask & 32)
            setSame(vert, vert[6], vert[5]);
        if (mask & 64)
            setSame(vert, vert[7], vert[6]);
        if (mask & 128)
            setSame(vert, vert[4], vert[7]);

        if (mask & 256)
            setSame(vert, vert[4], vert[0]);
        if (mask & 512)
            setSame(vert, vert[5], vert[1]);
        if (mask & 1024)
            setSame(vert, vert[6], vert[2]);
        if (mask & 2048)
            setSame(vert, vert[7], vert[3]);

        // check whether this is really case # mask
        // When eliminating edges, other edges might be eliminated as well
        int realMask = 0;

        if (vert[1] == vert[0])
            realMask |= 1;
        if (vert[2] == vert[1])
            realMask |= 2;
        if (vert[3] == vert[2])
            realMask |= 4;
        if (vert[3] == vert[0])
            realMask |= 8;

        if (vert[5] == vert[4])
            realMask |= 16;
        if (vert[6] == vert[5])
            realMask |= 32;
        if (vert[7] == vert[6])
            realMask |= 64;
        if (vert[7] == vert[4])
            realMask |= 128;

        if (vert[4] == vert[0])
            realMask |= 256;
        if (vert[5] == vert[1])
            realMask |= 512;
        if (vert[6] == vert[2])
            realMask |= 1024;
        if (vert[7] == vert[3])
            realMask |= 2048;

        // do not create non-existing case masks
        if (realMask != mask)
        {
            splitFile << "\n   {   0, { } },  // " << mask << " => " << realMask << endl;
            continue;
        }

        ///////////////////////////////////
        int v0 = vert[0];
        int v1 = vert[1];
        int v2 = vert[2];
        int v3 = vert[3];
        int v4 = vert[4];
        int v5 = vert[5];
        int v6 = vert[6];
        int v7 = vert[7];

        ///// First cases: one face lost = Pyra or Tetra
        Pyra *pyr = NULL;

        if (eq(v0, v1, v2, v3))
            pyr = new Pyra(v7, v6, v5, v4, v3);
        else if (eq(v1, v2, v6, v5))
            pyr = new Pyra(v0, v3, v7, v4, v1);
        else if (eq(v2, v3, v6, v7))
            pyr = new Pyra(v0, v4, v5, v1, v2);
        else if (eq(v0, v3, v7, v4))
            pyr = new Pyra(v1, v5, v6, v2, v0);
        else if (eq(v0, v1, v5, v4))
            pyr = new Pyra(v2, v6, v7, v3, v1);
        else if (eq(v4, v5, v6, v7))
            pyr = new Pyra(v0, v1, v2, v3, v4);

        if (pyr)
        {
            if (pyr->isConservativeFlat())
                print(mask, NULL, 0, splitFile);
            else if (pyr->isRegular())
                print(mask, *pyr, splitFile);
            else
            {
                int numParts, numCollapse;
                Shape **split = pyr->split(numParts, numCollapse);
                // always 1 part: either itself or split tetra

                if (split)
                {
                    if (numParts > 1)
                        print(mask, split, numParts, splitFile);
                    else
                        print(mask, *split[0], splitFile);
                    Shape::deleteShapeList(split);
                }
                else
                    abort();
            }

            delete pyr;
            continue;
        }

        ///// Next cases: Two parallel edges lost = Prism
        Prism *pris = NULL;
        if (v0 == v1 && v3 == v2)
            pris = new Prism(v0, v4, v5, v3, v7, v6);
        if (v1 == v2 && v0 == v3)
            pris = new Prism(v1, v5, v6, v0, v4, v7);

        if (v1 == v2 && v5 == v6)
            pris = new Prism(v5, v4, v7, v1, v0, v3);
        if (v1 == v5 && v2 == v6)
            pris = new Prism(v1, v0, v4, v2, v3, v7);

        if (v2 == v3 && v6 == v7)
            pris = new Prism(v6, v5, v4, v2, v1, v0);
        if (v2 == v6 && v3 == v7)
            pris = new Prism(v2, v1, v5, v3, v0, v4);

        if (v0 == v3 && v4 == v7)
            pris = new Prism(v7, v6, v5, v3, v2, v1);
        if (v0 == v4 && v3 == v7)
            pris = new Prism(v4, v5, v1, v7, v6, v2);

        if (v0 == v1 && v4 == v5)
            pris = new Prism(v1, v2, v3, v5, v6, v7);
        if (v0 == v4 && v1 == v5)
            pris = new Prism(v5, v6, v2, v4, v7, v3);

        if (v4 == v7 && v5 == v6)
            pris = new Prism(v6, v2, v1, v7, v3, v0);
        if (v4 == v5 && v6 == v7)
            pris = new Prism(v5, v1, v0, v6, v2, v3);

        if (pris)
        {
            if (pris->isFlat())
                print(mask, NULL, 0, splitFile);
            else if (pris->isRegular())
                print(mask, *pris, splitFile);
            else
            {
                int numParts, numCollapse;
                Shape **split = pris->split(numParts, numCollapse);
                if (split)
                {
                    if (numParts > 1)
                        print(mask, split, numParts, splitFile);
                    else
                        print(mask, *split[0], splitFile);
                    Shape::deleteShapeList(split);
                }
                else
                    abort();
            }

            delete pris;
            continue;
        }

        Hexa hexa(vert);
        int numParts, numCollapse;
        Shape **split = hexa.split(numParts, numCollapse);

        if (split)
        {
            if (numParts > 1)
                print(mask, split, numParts, splitFile);
            else
                print(mask, *split[0], splitFile);
            Shape::deleteShapeList(split);
        }
        else
            print(mask, hexa, splitFile);
    }

    return 0;
}
