/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __HEXA_H_
#define __HEXA_H_

// 09.01.01

#include "Shape.h"

/**
 * Class
 *
 */
class Hexa : public Shape
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    Hexa(const Hexa &);

    /// Assignment operator: NOT  IMPLEMENTED
    Hexa &operator=(const Hexa &);

    /// Default constructor: NOT  IMPLEMENTED
    Hexa();

    // vertex indices
    int d_vert[8];

    /// 18 rotations
    static const int s_rot[24][8];

    /// do a split if 0-1 is missing
    Shape **splitPart(int &numParts, int &numCollapse);

    // rotate this shape to a certain rotation number
    Hexa *rotateHexa(int rotNo) const;

public:
    /// Construct from 4 integers
    Hexa(int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7);
    Hexa(int v[8]);

    /// Destructor : virtual in case we derive objects
    virtual ~Hexa();

    // is this a regular shape ?
    virtual int isRegular();

    // is this shape flat?
    virtual int isFlat();

    // rotate this shape to a certain rotation number
    virtual Shape *rotate(int rotNo);

    // return number of possible rotations for this shape
    virtual int numRotations();

    // split this shape into multiple more regular sub-parts:
    virtual Shape **split(int &numPart, int &numCollapse);

    // print to a stream
    virtual void print(ostream &str) const;

    // print in IllConv form:
    virtual void printForm(ostream &str) const;

    // check rotation table
    static void checkRot();

    // return my Edge-Mask
    int getMask() const;

    // return minimal mask and rotation for it
    void minMask(int &mask, int &rot) const;

    // return inverse rotation
    static int invRot(int rot);

    // is this a legal STAR Cell?
    int isLegal();
};
#endif
