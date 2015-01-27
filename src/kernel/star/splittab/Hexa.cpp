/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Hexa.h"
#include "Prism.h"
#include "Pyra.h"
#include "Tetra.h"
#include <assert.h>

#undef VERBOSE

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
Hexa::Hexa(const Hexa &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
Hexa &Hexa::operator=(const Hexa &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
Hexa::Hexa()
{
    assert(0);
}

/// ----- Never forget the Destructor !! -------

Hexa::~Hexa()
{
}

/// All rotations
const int Hexa::s_rot[24][8] = {
    { // 0-1 front
      0, 1, 2, 3, 4, 5, 6, 7
    },
    { // 1-2 front
      1, 2, 3, 0, 5, 6, 7, 4
    },
    { // 2-3 front
      2, 3, 0, 1, 6, 7, 4, 5
    },
    { // 3-0 front
      3, 0, 1, 2, 7, 4, 5, 6
    },

    { // 1-0 front
      1, 0, 4, 5, 2, 3, 7, 6
    },
    { // 2-1 front
      2, 1, 5, 6, 3, 0, 4, 7
    },
    { // 3-2 front
      3, 2, 6, 7, 0, 1, 5, 4
    },
    { // 0-3 front
      0, 3, 7, 4, 1, 2, 6, 5
    },

    { // 4-5 front
      4, 5, 1, 0, 7, 6, 2, 3
    },
    { // 5-6 front
      5, 6, 2, 1, 4, 7, 3, 0
    },
    { // 6-7 front
      6, 7, 3, 2, 5, 4, 0, 1
    },
    { // 7-4 front
      7, 4, 0, 3, 6, 5, 1, 2
    },

    { // 4-7 front
      4, 7, 6, 5, 0, 3, 2, 1
    },
    { // 7-6 front
      7, 6, 5, 4, 3, 2, 1, 0
    },
    { // 6-5 front
      6, 5, 4, 7, 2, 1, 0, 3
    },
    { // 5-4 front
      5, 4, 7, 6, 1, 0, 3, 2
    },

    { // 0-4 front
      0, 4, 5, 1, 3, 7, 6, 2
    },
    { // 1-5 front
      1, 5, 6, 2, 0, 4, 7, 3
    },
    { // 2-6 front
      2, 6, 7, 3, 1, 5, 4, 0
    },
    { // 3-7 front
      3, 7, 4, 0, 2, 6, 5, 1
    },

    { // 4-0 front
      4, 0, 3, 7, 5, 1, 2, 6
    },
    { // 5-1 front
      5, 1, 0, 4, 6, 2, 3, 7
    },
    { // 6-2 front
      6, 2, 1, 5, 7, 3, 0, 4
    },
    { // 7-3 front
      7, 3, 2, 6, 4, 0, 1, 5
    },
};

/// Construct from 6 integers
Hexa::Hexa(int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7)
{
    d_vert[0] = v0;
    d_vert[1] = v1;
    d_vert[2] = v2;
    d_vert[3] = v3;
    d_vert[4] = v4;
    d_vert[5] = v5;
    d_vert[6] = v6;
    d_vert[7] = v7;
}

/// Construct from array
Hexa::Hexa(int v[8])
{
    d_vert[0] = v[0];
    d_vert[1] = v[1];
    d_vert[2] = v[2];
    d_vert[3] = v[3];
    d_vert[4] = v[4];
    d_vert[5] = v[5];
    d_vert[6] = v[6];
    d_vert[7] = v[7];
}

// is this a regular shape ?
int Hexa::isRegular()
{
    // all vertices must be different: compare i=0..4 against j=(i+1)...5
    int i, j;
    for (i = 0; i < 7; i++)
        for (j = i + 1; j < 8; j++)
            if (d_vert[i] == d_vert[j])
                return 0;
    return 1;
}

// is this shape flat?
int Hexa::isFlat()
{
    int numCollapses;
    // regular is non-empty
    if (isRegular())
        return 0;

    // all others are non-empty if one split part is non-flat
    int numParts;
    Shape **splitList = split(numParts, numCollapses);

    if (splitList)
    {
        do
            delete splitList[--numParts];
        while (numParts);
        return 0;
    }
    else
        return 1; // NO splitlist, no parts
}

// rotate this shape to a certain rotation number: return Shape
Shape *Hexa::rotate(int rotNo)
{
    return rotateHexa(rotNo);
}

// rotate this shape to a certain rotation number: return Hexa
Hexa *Hexa::rotateHexa(int rotNo) const
{
    if (rotNo >= 0 && rotNo < 24)
        return new Hexa(d_vert[s_rot[rotNo][0]],
                        d_vert[s_rot[rotNo][1]],
                        d_vert[s_rot[rotNo][2]],
                        d_vert[s_rot[rotNo][3]],
                        d_vert[s_rot[rotNo][4]],
                        d_vert[s_rot[rotNo][5]],
                        d_vert[s_rot[rotNo][6]],
                        d_vert[s_rot[rotNo][7]]);
    else
        return NULL;
}

// return number of possible rotations for this shape
int Hexa::numRotations()
{
    return 24;
}

// check all possible splits and use the one with least elements
Shape **Hexa::split(int &numParts, int &numCollapse)
{
    if (isRegular())
        return NULL;

    Shape **res = NULL;
    int i, minParts = 99999, minCollapse = 99999;

    for (i = 0; i < 24; i++)
    {
        // we try all 24 rotations here
        int tryParts = 0, tryCollapse = 0;
        Hexa *tryRot = this->rotateHexa(i);
        Shape **trySplit = tryRot->splitPart(tryParts, tryCollapse);
        //cout << " rot: " << *tryRot << " : " << tryParts << " Parts" << endl;

        // this try was better
        if (trySplit
            && tryParts > 1 // we have 1-part solutions
            && (tryCollapse < minCollapse
                || (tryCollapse == minCollapse && tryParts < minParts)))
        {
            res = trySplit;
            minParts = tryParts;
            minCollapse = tryCollapse;
        }
        else
            Shape::deleteShapeList(trySplit);
        delete tryRot;
    }

    if (minParts > 0 && minParts < 99999)
    {
        numParts = minParts;
        return res;
    }
    else
    {
        numParts = 0;
        return NULL;
    }
}

// split this shape into multiple more regular sub-parts:
Shape **Hexa::splitPart(int &numParts, int &numCollapse)
{

    numCollapse = 0;

    // regular or 0-1 not missing
    if (isRegular() || d_vert[0] != d_vert[1])
        return NULL;

#ifdef VERBOSE
    cerr << "\n===========================================" << endl;
    cerr << " Splitting: " << *this << endl;
    cerr << "===========================================" << endl;
#endif

    // We always use two base shapes
    Shape *part[2];

    if (d_vert[6] == d_vert[7]) // two opposing edge losts : split orthogonal to
    { // lost points direction
        part[0] = new Pyra(d_vert[5], d_vert[4], d_vert[3], d_vert[2], d_vert[1]);
        part[1] = new Pyra(d_vert[2], d_vert[3], d_vert[4], d_vert[5], d_vert[6]);
    }
    else // otherwise split along diagonal from lost edge
    {
        part[0] = new Pyra(d_vert[2], d_vert[6], d_vert[7], d_vert[3], d_vert[0]);
        part[1] = new Pyra(d_vert[5], d_vert[4], d_vert[7], d_vert[6], d_vert[0]);
    }

    // build the shape list
    Shape **shapelist = new Shape *[5]; // max. 2x2 + NULL element
    numParts = 0;
    int i, p;

    for (p = 0; p < 2; p++)
    {
        if (part[p]->isRegular())
        {
#ifdef VERBOSE
            cerr << "Part[" << p << "] : " << *part[p] << " regular" << endl;
#endif
            shapelist[numParts++] = part[p];
        }
        else
        {
            int partNumParts;
            int partCollapses;
            Shape **partParts = part[p]->split(partNumParts, partCollapses);
#ifdef VERBOSE
            cerr << "\nPart1: " << *part[p] << " " << partNumParts << " Parts, "
                 << partCollapses << " Collapses" << endl;
#endif
            if (partParts)
                for (i = 0; i < partNumParts; i++)
                    shapelist[numParts++] = partParts[i];
            delete partParts;
            numCollapse += partCollapses;
        }
    }

#ifdef VERBOSE
    cerr << "SPLIT :::: ";
    for (i = 0; i < numParts; i++)
        cerr << *shapelist[i] << " ";
    cerr << " :::: " << numCollapse << " collapses" << endl;
#endif

    if (numParts)
    {
        shapelist[numParts] = NULL;
        return shapelist;
    }
    else
    {
        delete shapelist;
        return NULL;
    }
}

// print to a stream
void Hexa::print(ostream &str) const
{
    str << "Hexa(" << d_vert[0] << "," << d_vert[1]
        << "," << d_vert[2] << "," << d_vert[3]
        << "," << d_vert[4] << "," << d_vert[5]
        << "," << d_vert[6] << "," << d_vert[7]
        << ")";
}

// check rotation table
void Hexa::checkRot()
{
    static int conter[8] = { 6, 7, 4, 5, 2, 3, 0, 1 };

    int sum[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    // each line must contain 0-7 once
    int i, rot;
    for (rot = 0; rot < 24; rot++)
    {
        int ind[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (i = 0; i < 8; i++)
        {
            // no duplicate vertices
            if (ind[s_rot[rot][i]])
                cerr << " rotation " << rot << " re-uses " << ind[s_rot[rot][i]] << endl;
            else
                ind[s_rot[rot][i]]++;

            // volume diagonal check
            if (conter[s_rot[rot][i]] != s_rot[rot][conter[i]])
                cerr << " rotation " << rot
                     << " failed diagonal check on vertex " << i << endl;

            // sum up columns (symmetry check)
            sum[i] += s_rot[rot][i];
        }
    }
    cerr << "Sums:";
    for (i = 0; i < 8; i++)
        cerr << " " << sum[i];
    cerr << endl;
}

// print in IllConv form:
void Hexa::printForm(ostream &str) const
{
    str << "{ " << d_vert[0] << ", " << d_vert[1]
        << ", " << d_vert[2] << ", " << d_vert[3]
        << ", " << d_vert[4] << ", " << d_vert[5]
        << ", " << d_vert[6] << ", " << d_vert[7]
        << " }";
}

// return my Edge-Mask
int Hexa::getMask() const
{
    int realMask = 0;

    if (d_vert[1] == d_vert[0])
        realMask |= 1;
    if (d_vert[2] == d_vert[1])
        realMask |= 2;
    if (d_vert[3] == d_vert[2])
        realMask |= 4;
    if (d_vert[3] == d_vert[0])
        realMask |= 8;
    if (d_vert[5] == d_vert[4])
        realMask |= 16;
    if (d_vert[6] == d_vert[5])
        realMask |= 32;
    if (d_vert[7] == d_vert[6])
        realMask |= 64;
    if (d_vert[7] == d_vert[4])
        realMask |= 128;
    if (d_vert[4] == d_vert[0])
        realMask |= 256;
    if (d_vert[5] == d_vert[1])
        realMask |= 512;
    if (d_vert[6] == d_vert[2])
        realMask |= 1024;
    if (d_vert[7] == d_vert[3])
        realMask |= 2048;

    return realMask;
}

// return minimal mask and rotation for it
void Hexa::minMask(int &minMask, int &minRot) const
{
    minMask = 4096; // all masks are below this, one will replace it

    int rot;
    for (rot = 0; rot < 24; rot++)
    {
        Hexa *hex = rotateHexa(rot);
        int hexMask = hex->getMask();
        if (hexMask < minMask)
        {
            minMask = hexMask;
            minRot = rot;
        }
    }
}

// return inverse rotation
int Hexa::invRot(int rot)
{
    int inv;
    for (inv = 0; inv < 24; inv++)
        if (s_rot[inv][s_rot[rot][0]] == 0
            && s_rot[inv][s_rot[rot][1]] == 1
            && s_rot[inv][s_rot[rot][2]] == 2
            && s_rot[inv][s_rot[rot][3]] == 3
            && s_rot[inv][s_rot[rot][4]] == 4
            && s_rot[inv][s_rot[rot][5]] == 5
            && s_rot[inv][s_rot[rot][6]] == 6
            && s_rot[inv][s_rot[rot][7]] == 7)
            return inv;
    abort();
    return -1;
}

inline int hasArea(int v0, int v1, int v2, int v3)
{
    return (v0 != v1 && v1 != v2 && v2 != v0) // triangle 0-1-2
           || (v1 != v2 && v2 != v3 && v3 != v1) // triangle 1-2-3
           || (v2 != v3 && v3 != v0 && v0 != v2) // triangle 2-3-0
           || (v3 != v0 && v0 != v1 && v1 != v3); // triangle 3-0-1
}

// is this a legal STAR Hexa Cell?
int Hexa::isLegal()
{
    int &v0 = d_vert[0], &v1 = d_vert[1], &v2 = d_vert[2], &v3 = d_vert[3],
        &v4 = d_vert[4], &v5 = d_vert[5], &v6 = d_vert[6], &v7 = d_vert[7];

    // legal cells have either 6 sides with area or 'special' shape

    return (hasArea(v0, v1, v2, v3)
            && hasArea(v4, v5, v6, v7)
            && hasArea(v1, v2, v6, v5)
            && hasArea(v2, v3, v7, v6)
            && hasArea(v3, v0, v4, v7)
            && hasArea(v0, v1, v5, v4))
           || (v2 == v3 && v4 == v5 && v6 == v7
               && v0 != v1 && v0 != v2 && v0 != v4 && v0 != v6
               && v1 != v2 && v1 != v4 && v1 != v6
               && v2 != v4 && v2 != v6
               && v4 != v6);
}
