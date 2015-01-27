/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFCOLOR_
#define _VRMLSFCOLOR_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFColor : public VrmlSField
{
public:
    VrmlSFColor(float r = 1.0, float g = 1.0, float b = 1.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFColor *toSFColor() const;
    virtual VrmlSFColor *toSFColor();

    float r(void)
    {
        return d_rgb[0];
    }
    float g(void)
    {
        return d_rgb[1];
    }
    float b(void)
    {
        return d_rgb[2];
    }
    float *get()
    {
        return &d_rgb[0];
    }
    void set(float r, float g, float b)
    {
        d_rgb[0] = r;
        d_rgb[1] = g;
        d_rgb[2] = b;
    }

    static void HSVtoRGB(float h, float s, float v,
                         float &r, float &g, float &b);
    static void RGBtoHSV(float r, float g, float b,
                         float &h, float &s, float &v);

    void setHSV(float h, float s, float v);
    void getHSV(float &h, float &s, float &v);

private:
    float d_rgb[3];
};
}
#endif //_VRMLSFCOLOR_
