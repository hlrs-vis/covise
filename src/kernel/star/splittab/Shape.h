/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SHAPE_H_
#define __SHAPE_H_

// 09.01.01

/**
 * Base class for all Shapes
 *
 */

#include <iostream.h>
#include <stdlib.h>

class Shape
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    Shape(const Shape &);

    /// Assignment operator: NOT  IMPLEMENTED
    Shape &operator=(const Shape &);

public:
    /// Default constructor: empty in base class
    Shape(){};

    /// Destructor : virtual in case we derive objects
    virtual ~Shape(){};

    // is this a regular shape ?
    virtual int isRegular() = 0;

    // ist this a 0-volume shape ?
    virtual int isFlat() = 0;

    // rotate this shape to a certain rotation number
    virtual Shape *rotate(int rotNo) = 0;

    // return number of possible rotations for this shape
    virtual int numRotations() = 0;

    // split this shape into multiple more regular sub-parts:
    virtual Shape **split(int &numParts, int &didCollapse) = 0;

    // print to a stream
    virtual void print(ostream &str) const = 0;

    // delete list of shapes
    static void deleteShapeList(Shape **list);

    // print in IllConv form:
    virtual void printForm(ostream &str) const = 0;
};

inline ostream &operator<<(ostream &str, const Shape &shape)
{
    shape.print(str);
    return str;
}
#endif
