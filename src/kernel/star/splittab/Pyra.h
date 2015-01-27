/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __PYRA_H_
#define __PYRA_H_

// 09.01.01

#include "Shape.h"

/**
 * Class
 *
 */
class Pyra : public Shape
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    Pyra(const Pyra &);

    /// Assignment operator: NOT  IMPLEMENTED
    Pyra &operator=(const Pyra &);

    /// Default constructor: NOT  IMPLEMENTED
    Pyra();

    // vertex indices
    int d_vert[5];

    /// 4 rotations of floor
    static const int s_rot[4][4];

public:
    /// Destructor : virtual in case we derive objects
    virtual ~Pyra();

    /// construct from points or from array
    Pyra(int v0, int v1, int v2, int v3, int v4);
    Pyra(int v[5]);

    // is this a regular shape ?
    virtual int isRegular();

    // is this shape flat? (allows high deformations)
    virtual int isFlat();

    // flat without deformations: bottom lost or top==bottom
    int isConservativeFlat();

    // rotate this shape to a certain rotation number
    virtual Shape *rotate(int rotNo);

    // return number of possible rotations for this shape
    virtual int numRotations();

    // split this shape into multiple more regular sub-parts:
    virtual Shape **split(int &numParts, int &numCollapse);

    // print to a stream
    virtual void print(ostream &str) const;

    // print in IllConv form:
    virtual void printForm(ostream &str) const;
};
#endif
