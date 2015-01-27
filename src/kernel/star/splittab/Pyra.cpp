/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Pyra.h"
#include "Tetra.h"
#include <assert.h>
#include <stdlib.h>

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
Pyra::Pyra(const Pyra &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
Pyra &Pyra::operator=(const Pyra &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
Pyra::Pyra()
{
    assert(0);
}

/// ----- Never forget the Destructor !! -------

Pyra::~Pyra()
{
}

/// construct from points or from array
Pyra::Pyra(int v0, int v1, int v2, int v3, int v4)
{
    d_vert[0] = v0;
    d_vert[1] = v1;
    d_vert[2] = v2;
    d_vert[3] = v3;
    d_vert[4] = v4;
}

Pyra::Pyra(int v[5])
{
    d_vert[0] = v[0];
    d_vert[1] = v[1];
    d_vert[2] = v[2];
    d_vert[3] = v[3];
    d_vert[4] = v[4];
}

/// 4 rotations
const int Pyra::s_rot[4][4] = {
    { 0, 1, 2, 3 },
    { 1, 2, 3, 0 },
    { 2, 3, 0, 1 },
    { 3, 0, 1, 2 }
};

// is this a regular shape ?
int Pyra::isRegular()
{
    return (d_vert[0] != d_vert[1])
           && (d_vert[0] != d_vert[2])
           && (d_vert[0] != d_vert[3])
           && (d_vert[0] != d_vert[4])
           && (d_vert[1] != d_vert[2])
           && (d_vert[1] != d_vert[3])
           && (d_vert[1] != d_vert[4])
           && (d_vert[2] != d_vert[3])
           && (d_vert[2] != d_vert[4])
           && (d_vert[3] != d_vert[4]);
}

// is this shape conservatively flat?
int Pyra::isConservativeFlat()
{
    // need 3 points in floor:

    int numSame = 0;

    if (d_vert[0] == d_vert[1])
        numSame++;
    if (d_vert[0] == d_vert[2])
        numSame++;
    if (d_vert[0] == d_vert[3])
        numSame++;

    if (d_vert[1] == d_vert[2])
        numSame++;
    if (d_vert[1] == d_vert[3])
        numSame++;

    if (d_vert[2] == d_vert[3])
        numSame++;

    return
        // Top in lower plane
        d_vert[4] == d_vert[0]
        || d_vert[4] == d_vert[1]
        || d_vert[4] == d_vert[2]
        || d_vert[4] == d_vert[3]

        || numSame > 1;
}

// is this shape flat?
int Pyra::isFlat()
{
    int numSame = 0;

    if (d_vert[0] == d_vert[1])
        numSame++;
    if (d_vert[0] == d_vert[2])
        numSame++;
    if (d_vert[0] == d_vert[3])
        numSame++;
    if (d_vert[0] == d_vert[4])
        numSame++;

    if (d_vert[1] == d_vert[2])
        numSame++;
    if (d_vert[1] == d_vert[3])
        numSame++;
    if (d_vert[1] == d_vert[4])
        numSame++;

    if (d_vert[2] == d_vert[3])
        numSame++;
    if (d_vert[2] == d_vert[4])
        numSame++;

    if (d_vert[3] == d_vert[4])
        numSame++;

    return numSame > 1 // 3 points same
           || (d_vert[0] == d_vert[2]) // ground diagonal missing
           || (d_vert[1] == d_vert[3]);
}

// rotate this shape to a certain rotation number
Shape *Pyra::rotate(int rotNo)
{
    if (rotNo >= 0 && rotNo < 4)
        return new Pyra(d_vert[s_rot[rotNo][0]],
                        d_vert[s_rot[rotNo][1]],
                        d_vert[s_rot[rotNo][2]],
                        d_vert[s_rot[rotNo][3]],
                        d_vert[4]);
    else
        return NULL;
}

// return number of possible rotations for this shape
int Pyra::numRotations()
{
    return 4;
}

// split this shape into multiple more regular sub-parts:
Shape **Pyra::split(int &numParts, int &numCollapse)
{
    numCollapse = 0;

    if (isRegular() || isFlat())
        return NULL;
    else
    {
        // always one part if splitting at all
        Shape **res = new Shape *[2];
        res[1] = NULL;
        numParts = 1;

        // one ground edge missing
        if (d_vert[0] == d_vert[1])
            res[0] = new Tetra(d_vert[0], d_vert[2], d_vert[3], d_vert[4]);
        else if (d_vert[1] == d_vert[2])
            res[0] = new Tetra(d_vert[0], d_vert[1], d_vert[3], d_vert[4]);
        else if (d_vert[2] == d_vert[3])
            res[0] = new Tetra(d_vert[0], d_vert[1], d_vert[2], d_vert[4]);
        else if (d_vert[3] == d_vert[0])
            res[0] = new Tetra(d_vert[0], d_vert[1], d_vert[2], d_vert[4]);

        // one side missing
        else if (d_vert[0] == d_vert[4])
            res[0] = new Tetra(d_vert[1], d_vert[2], d_vert[3], d_vert[4]);
        else if (d_vert[1] == d_vert[4])
            res[0] = new Tetra(d_vert[0], d_vert[2], d_vert[3], d_vert[4]);
        else if (d_vert[2] == d_vert[4])
            res[0] = new Tetra(d_vert[0], d_vert[1], d_vert[3], d_vert[4]);
        else if (d_vert[3] == d_vert[4])
            res[0] = new Tetra(d_vert[0], d_vert[1], d_vert[3], d_vert[4]);

        // oops, what is that?
        else
        {
            cerr << "we should never get here: splitting "
                 << *this
                 << endl;
            abort();
        }
        return res;
    }
}

// print to a stream
void Pyra::print(ostream &str) const
{
    str << "Pyra(" << d_vert[0] << "," << d_vert[1]
        << "," << d_vert[2] << "," << d_vert[3]
        << "," << d_vert[4]
        << ")";
}

// print in IllConv form:
void Pyra::printForm(ostream &str) const
{
    str << "{ " << d_vert[0] << ", " << d_vert[1]
        << ", " << d_vert[2] << ", " << d_vert[3]
        << ", " << d_vert[4] << ", " << d_vert[4]
        << ", " << d_vert[4] << ", " << d_vert[4]
        << " }";
}
