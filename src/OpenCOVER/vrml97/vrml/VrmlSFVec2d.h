/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFVEC2D_
#define _VRMLSFVEC2D_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFVec2d : public VrmlSField
{
public:
    VrmlSFVec2d(double x = 0.0, double y = 0.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFVec2d *toSFVec2d() const;
    virtual VrmlSFVec2d *toSFVec2d();

    double x(void)
    {
        return d_x[0];
    }
    double y(void)
    {
        return d_x[1];
    }
    double *get()
    {
        return &d_x[0];
    }

    void set(double x, double y)
    {
        d_x[0] = x;
        d_x[1] = y;
    }

    // return result
    double dot(VrmlSFVec2d *);
    double length();

    // modifiers
    void normalize();

    void add(VrmlSFVec2d *);
    void divide(double);
    void multiply(double);
    void subtract(VrmlSFVec2d *);

private:
    double d_x[2];
};
}
#endif //_VRMLSFVEC2D_
