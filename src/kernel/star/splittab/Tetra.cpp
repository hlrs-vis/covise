/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Tetra.h"
#include <assert.h>

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
Tetra::Tetra(const Tetra &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
Tetra &Tetra::operator=(const Tetra &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
Tetra::Tetra()
{
    assert(0);
}

/// ----- Never forget the Destructor !! -------

Tetra::~Tetra()
{
}

/// constructor: construct from 4 vertex indices
Tetra::Tetra(int v0, int v1, int v2, int v3)
{
    d_vert[0] = v0;
    d_vert[1] = v1;
    d_vert[2] = v2;
    d_vert[3] = v3;
}

/// constructor: construct from 4 vertex indices
Tetra::Tetra(int v[4])
{
    d_vert[0] = v[0];
    d_vert[1] = v[1];
    d_vert[2] = v[2];
    d_vert[3] = v[3];
}

/// All rotations
const int Tetra::s_rot[12][4] = {
    { // 0-1 front
      0, 1, 2, 3
    },
    { 1, 0, 3, 2 },
    { // 1-2 front
      1, 2, 0, 3
    },
    { 2, 1, 3, 0 },
    { // 2-0 front
      2, 0, 1, 3
    },
    { 0, 2, 3, 1 },
    { // 0-3 front
      3, 0, 2, 1
    },
    { 0, 3, 1, 2 },
    { // 1-3 front
      1, 3, 2, 0
    },
    { 3, 1, 0, 2 },
    { // 2-3 front
      2, 3, 0, 1
    },
    { 3, 2, 1, 0 },
};

int Tetra::isRegular()
{
    return (d_vert[0] != d_vert[1])
           && (d_vert[0] != d_vert[2])
           && (d_vert[0] != d_vert[3])
           && (d_vert[1] != d_vert[2])
           && (d_vert[1] != d_vert[3])
           && (d_vert[2] != d_vert[3]);
}

// is this shape flat?
int Tetra::isFlat()
{
    return !isRegular(); // all non-regular tetras are flat
}

// rotate this shape to a certain rotation number
Shape *Tetra::rotate(int rotNo)
{
    if (rotNo >= 0 && rotNo < 12)
        return new Tetra(d_vert[s_rot[rotNo][0]],
                         d_vert[s_rot[rotNo][1]],
                         d_vert[s_rot[rotNo][2]],
                         d_vert[s_rot[rotNo][3]]);
    else
        return NULL;
}

// return number of possible rotations
int Tetra::numRotations()
{
    return 12;
}

// A tetra is either flat or regular: cannot split
Shape **Tetra::split(int &numParts, int &numCollapse)
{
    numCollapse = 0;
    return NULL;
}

void Tetra::print(ostream &str) const
{
    str << "Tetra(" << d_vert[0] << "," << d_vert[1]
        << "," << d_vert[2] << "," << d_vert[3] << ")";
}

// print in IllConv form:
void Tetra::printForm(ostream &str) const
{
    str << "{ " << d_vert[0] << ", " << d_vert[1]
        << ", " << d_vert[2] << ", " << d_vert[2]
        << ", " << d_vert[3] << ", " << d_vert[3]
        << ", " << d_vert[3] << ", " << d_vert[3]
        << " }";
}
