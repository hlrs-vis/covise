/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFVEC3D_
#define _VRMLSFVEC3D_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFVec3d : public VrmlSField
{
public:
    VrmlSFVec3d(double x = 0.0, double y = 0.0, double z = 0.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFVec3d *toSFVec3d() const;
    virtual VrmlSFVec3d *toSFVec3d();

    double x(void)
    {
        return d_x[0];
    }
    double y(void)
    {
        return d_x[1];
    }
    double z(void)
    {
        return d_x[2];
    }

    double *get()
    {
        return &d_x[0];
    }

    void set(double x, double y, double z)
    {
        d_x[0] = x;
        d_x[1] = y;
        d_x[2] = z;
    }

    // return result
    double dot(VrmlSFVec3d *);
    double length();

    // modifiers
    void normalize();

    void add(VrmlSFVec3d *);
    void cross(VrmlSFVec3d *);
    void divide(double);
    void multiply(double);
    void subtract(VrmlSFVec3d *);

private:
    double d_x[3];
};
}
#endif //_VRMLSFVEC3D_
