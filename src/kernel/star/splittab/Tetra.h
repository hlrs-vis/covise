/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TETRA_H_
#define __TETRA_H_

// 09.01.01

#include "Shape.h"

/**
 * Class
 *
 */
class Tetra : public Shape
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    Tetra(const Tetra &);

    /// Assignment operator: NOT  IMPLEMENTED
    Tetra &operator=(const Tetra &);

    /// Default constructor: NOT  IMPLEMENTED
    Tetra();

    // vertex indices
    int d_vert[4];

    /// 18 rotations
    static const int s_rot[12][4];

public:
    /// constructor: construct from 4 vertex indices
    Tetra(int v0, int v1, int v2, int v3);

    /// constructor: construct from array of 4 vertex indices
    Tetra(int v[4]);

    /// Destructor : virtual in case we derive objects
    virtual ~Tetra();

    // is this a regular shape ?
    virtual int isRegular();

    // is this shape flat?
    virtual int isFlat();

    // rotate this shape to a certain rotation number
    virtual Shape *rotate(int rotNo);

    // return number of possible rotations for this shape
    virtual int numRotations();

    // split this shape into multiple more regular sub-parts:
    virtual Shape **split(int &numParts, int &numCollapse);

    // split this shape into multiple more regular sub-parts:
    virtual void print(ostream &str) const;

    // print in IllConv form:
    virtual void printForm(ostream &str) const;
};
#endif
