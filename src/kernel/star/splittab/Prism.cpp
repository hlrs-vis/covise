/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Pyra.h"
#include "Prism.h"
#include "Tetra.h"
#include <assert.h>

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
Prism::Prism(const Prism &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
Prism &Prism::operator=(const Prism &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
Prism::Prism()
{
    assert(0);
}

/// ----- Never forget the Destructor !! -------

Prism::~Prism()
{
}

/// All rotations
const int Prism::s_rot[6][6] = {
    { // 0-1 front
      0, 1, 2, 3, 4, 5
    },
    { // 1-2 front
      1, 2, 0, 4, 5, 3
    },
    { // 2-0 front
      2, 0, 1, 5, 3, 4
    },
    { // 3-4 front
      4, 3, 5, 1, 0, 2
    },
    { // 4-5 front
      5, 4, 3, 2, 1, 0
    },
    { // 5-3 front
      3, 5, 4, 0, 2, 1
    },
};

/// Construct from 6 integers
Prism::Prism(int v0, int v1, int v2, int v3, int v4, int v5)
{
    d_vert[0] = v0;
    d_vert[1] = v1;
    d_vert[2] = v2;
    d_vert[3] = v3;
    d_vert[4] = v4;
    d_vert[5] = v5;
}

/// Construct from array
Prism::Prism(int v[6])
{
    d_vert[0] = v[0];
    d_vert[1] = v[1];
    d_vert[2] = v[2];
    d_vert[3] = v[3];
    d_vert[4] = v[4];
    d_vert[5] = v[5];
}

// is this a regular shape ?
int Prism::isRegular()
{
    // all vertices must be different: compare i=0..4 against j=(i+1)...5
    int i, j;
    for (i = 0; i < 5; i++)
        for (j = i + 1; j < 6; j++)
            if (d_vert[i] == d_vert[j])
                return 0;
    return 1;
}

// check whether 4 vertices are different
inline int alldiff(int v0, int v1, int v2, int v3)
{
    return v0 != v1 && v0 != v2 && v0 != v3
           && v1 != v2 && v1 != v3
           && v2 != v3;
}

// is this shape flat?
int Prism::isFlat()
{
    int &v0 = d_vert[0], &v1 = d_vert[1], &v2 = d_vert[2],
        &v3 = d_vert[3], &v4 = d_vert[4], &v5 = d_vert[5];

    // check all possible tetras within prism
    if (alldiff(v0, v1, v2, v3)) // bottom and one top vertex
        return 0;
    if (alldiff(v0, v1, v2, v4))
        return 0;
    if (alldiff(v0, v1, v2, v5))
        return 0;

    if (alldiff(v3, v4, v5, v0)) // top and one bottom vertex
        return 0;
    if (alldiff(v3, v4, v5, v1))
        return 0;
    if (alldiff(v3, v4, v5, v2))
        return 0;

    /*****/
    //  no, these are flat!
    if (alldiff(v1, v5, v0, v3)) // quad diagonal and oppos. line 0-3
        return 0;
    if (alldiff(v2, v4, v0, v3))
        return 0;

    if (alldiff(v0, v5, v1, v4)) // quad diagonal and oppos. line 1-4
        return 0;
    if (alldiff(v2, v3, v1, v4))
        return 0;

    if (alldiff(v0, v4, v2, v5)) // quad diagonal and oppos. line 2-5
        return 0;
    if (alldiff(v1, v3, v3, v5))
        return 0;
    /******/
    return 1;
}

// rotate this shape to a certain rotation number
Shape *Prism::rotate(int rotNo)
{
    if (rotNo >= 0 && rotNo < 6)
        return new Prism(d_vert[s_rot[rotNo][0]],
                         d_vert[s_rot[rotNo][1]],
                         d_vert[s_rot[rotNo][2]],
                         d_vert[s_rot[rotNo][3]],
                         d_vert[s_rot[rotNo][4]],
                         d_vert[s_rot[rotNo][5]]);
    else
        return NULL;
}

// return number of possible rotations for this shape
int Prism::numRotations()
{
    return 6;
}

// split this shape into multiple more regular sub-parts:
Shape **Prism::split(int &numParts, int &numCollapse)
{
    int &v0 = d_vert[0], &v1 = d_vert[1], &v2 = d_vert[2],
        &v3 = d_vert[3], &v4 = d_vert[4], &v5 = d_vert[5];

    // we haven't yes collapsed any part
    numCollapse = 0;

    // never split regular parts
    if (isRegular())
        return NULL;

    ///////////// 'Master edge' missing
    if (v0 == v1)
    {
        if (v3 == v4)
        {
            if (alldiff(v0, v4, v2, v5))
            {
                Shape **res = new Shape *[2];
                res[0] = new Tetra(v4, v5, v2, v0);
                res[1] = NULL;
                numCollapse = 2;
                numParts = 1;
                return res;
            }
            else
            {
                numCollapse = 3;
                numParts = 0;
                return NULL;
            }
        }
        else
        {
            numParts = 0;
            Shape **res = new Shape *[3];

            // 1st split tetra
            res[numParts] = new Tetra(d_vert[2], d_vert[3], d_vert[4], d_vert[5]);
            if (!res[numParts]->isFlat())
                numParts++;
            else
                numCollapse++;

            // 2nd split tetra
            res[numParts] = new Tetra(d_vert[2], d_vert[4], d_vert[3], d_vert[1]);
            if (!res[numParts]->isFlat())
                numParts++;
            else
                numCollapse++;

            res[numParts] = NULL;

            if (numParts > 0)
                return res;
            else
                return NULL;
        }
    }

    ///////////// 'Master Topdown' missing
    if (d_vert[0] == d_vert[3])
    {
        Pyra *pyr = new Pyra(d_vert[1], d_vert[4], d_vert[5], d_vert[2], d_vert[0]);
        if (pyr->isRegular())
        {
            Shape **res = new Shape *[2];
            res[0] = pyr;
            res[1] = NULL;
            numParts = 1;
            return res;
        }
        else
        {
            Shape **res = pyr->split(numParts, numCollapse);
            if (!res)
                numCollapse += 2; // two collapsed tetras
            if (isRegular() && !res)
                abort();
            delete pyr;
            return res;
        }
    }

    /////// other cases: rotate until 'Master' case found
    int rotNo;
    for (rotNo = 1; rotNo < 6; rotNo++)
    {
        if (d_vert[s_rot[rotNo][0]] == d_vert[s_rot[rotNo][1]]
            || d_vert[s_rot[rotNo][0]] == d_vert[s_rot[rotNo][3]])
        {
            Shape *rot = rotate(rotNo);
            //cout << " Using rotation " << rotNo << ": " << *rot << endl;
            Shape **res = rot->split(numParts, numCollapse);
            delete rot;
            if (res)
                return res;
        }
    }
    assert(0);
    return NULL;
}

// print to a stream
void Prism::print(ostream &str) const
{
    str << "Prism(" << d_vert[0] << "," << d_vert[1]
        << "," << d_vert[2] << "," << d_vert[3]
        << "," << d_vert[4] << "," << d_vert[5]
        << ")";
}

// print in IllConv form:
void Prism::printForm(ostream &str) const
{
    str << "{ " << d_vert[0] << ", " << d_vert[1]
        << ", " << d_vert[2] << ", " << d_vert[2]
        << ", " << d_vert[3] << ", " << d_vert[4]
        << ", " << d_vert[5] << ", " << d_vert[5]
        << " }";
}
