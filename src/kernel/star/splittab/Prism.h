/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __PRISM_H_
#define __PRISM_H_

// 09.01.01

#include "Shape.h"

/**
 * Class
 *
 */
class Prism : public Shape
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    Prism(const Prism &);

    /// Assignment operator: NOT  IMPLEMENTED
    Prism &operator=(const Prism &);

    /// Default constructor: NOT  IMPLEMENTED
    Prism();

    // vertex indices
    int d_vert[6];

    /// 18 rotations
    static const int s_rot[6][6];

public:
    /// Construct from 4 integers
    Prism(int v0, int v1, int v2, int v3, int v4, int v5);
    Prism(int v[6]);

    /// Destructor : virtual in case we derive objects
    virtual ~Prism();

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

    // print to a stream
    virtual void print(ostream &str) const;

    // print in IllConv form:
    virtual void printForm(ostream &str) const;
};
#endif
