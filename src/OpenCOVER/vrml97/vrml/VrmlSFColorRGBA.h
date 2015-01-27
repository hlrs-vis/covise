/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLSFCOLORRGBA_
#define _VRMLSFCOLORRGBA_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlSFColorRGBA : public VrmlSField
{
public:
    VrmlSFColorRGBA(float r = 1.0, float g = 1.0, float b = 1.0, float a = 1.0);

    virtual std::ostream &print(std::ostream &os) const;

    virtual VrmlField *clone() const;

    virtual VrmlFieldType fieldType() const;
    virtual const VrmlSFColorRGBA *toSFColorRGBA() const;
    virtual VrmlSFColorRGBA *toSFColorRGBA();

    float r(void)
    {
        return d_rgba[0];
    }
    float g(void)
    {
        return d_rgba[1];
    }
    float b(void)
    {
        return d_rgba[2];
    }
    float a(void)
    {
        return d_rgba[3];
    }
    float *get()
    {
        return &d_rgba[0];
    }
    void set(float r, float g, float b, float a)
    {
        d_rgba[0] = r;
        d_rgba[1] = g;
        d_rgba[2] = b;
        d_rgba[3] = a;
    }

    static void HSVtoRGB(float h, float s, float v,
                         float &r, float &g, float &b);
    static void RGBtoHSV(float r, float g, float b,
                         float &h, float &s, float &v);

    void setHSV(float h, float s, float v);
    void getHSV(float &h, float &s, float &v);

private:
    float d_rgba[4];
};
}
#endif //_VRMLSFCOLORRGBA_
