/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFVEC2F_
#define _VRMLSFVEC2F_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFVec2f : public VrmlSField
{
public:
    VrmlSFVec2f(float x = 0.0, float y = 0.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFVec2f *toSFVec2f() const;
    virtual VrmlSFVec2f *toSFVec2f();

    float x(void)
    {
        return d_x[0];
    }
    float y(void)
    {
        return d_x[1];
    }
    float *get()
    {
        return &d_x[0];
    }

    void set(float x, float y)
    {
        d_x[0] = x;
        d_x[1] = y;
    }

    // return result
    double dot(VrmlSFVec2f *);
    double length();

    // modifiers
    void normalize();

    void add(VrmlSFVec2f *);
    void divide(float);
    void multiply(float);
    void subtract(VrmlSFVec2f *);

private:
    float d_x[2];
};
}
#endif //_VRMLSFVEC2F_
