/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFMATRIX_
#define _VRMLSFMATRIX_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFMatrix : public VrmlSField
{
public:
    VrmlSFMatrix(float x0 = 1.0, float y0 = 0.0, float z0 = 0.0,float x1 = 0.0, float y1 = 1.0, float z1 = 0.0, float x2 = 0.0, float y2 = 0.0, float z2 = 1.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFMatrix *toSFMatrix() const;
    virtual VrmlSFMatrix *toSFMatrix();


    float *get()
    {
        return &d_x[0];
    }

    void set(float x0 = 1.0, float y0 = 0.0, float z0 = 0.0, float x1 = 0.0, float y1 = 1.0, float z1 = 0.0, float x2 = 0.0, float y2 = 0.0, float z2 = 1.0)
    {
        d_x[0] = x0;
        d_x[1] = y0;
        d_x[2] = z0;
        d_x[3] = x1;
        d_x[4] = y1;
        d_x[5] = z1;
        d_x[6] = x2;
        d_x[7] = y2;
        d_x[8] = z2;
    }

    // return result

    // modifiers
    void normalize();

    void multiply(VrmlSFMatrix*);

private:
    float d_x[9];
};
}
#endif //_VRMLSFMatrix_
