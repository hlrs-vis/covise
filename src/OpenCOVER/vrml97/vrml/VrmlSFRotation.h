/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFROTATION_
#define _VRMLSFROTATION_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFRotation : public VrmlSField
{
public:
    VrmlSFRotation(float x = 0.0, float y = 0.0, float z = 1.0, float r = 0.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFRotation *toSFRotation() const;
    virtual VrmlSFRotation *toSFRotation();

    float x(void)
    {
        return d_x[0];
    }
    float y(void)
    {
        return d_x[1];
    }
    float z(void)
    {
        return d_x[2];
    }
    float r(void)
    {
        return d_x[3];
    }
    float *get()
    {
        return &d_x[0];
    }

    void set(float x, float y, float z, float r)
    {
        d_x[0] = x;
        d_x[1] = y;
        d_x[2] = z;
        d_x[3] = r;
    }

    void invert(void);
    void multiply(VrmlSFRotation *);
    void slerp(VrmlSFRotation *, float);

private:
    float d_x[4];
};
}
#endif //_VRMLSFROTATION_
