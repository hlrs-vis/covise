/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlField.cpp
//  The VrmlField class is the base field class.
//

#include "config.h"
#include "VrmlField.h"
// SFNode

#include "VrmlSFNode.h"
#include "VrmlNode.h"
#include "MathUtils.h"
#include <math.h>
#include <string.h>

#include <iostream>
using std::ostream;
using std::endl;
using namespace vrml;

VrmlField::VrmlField() {}

// Even though it is a pure virtual function, the destructor needs a definition
// since each destructor in the inheritance hierarchy is called when a derived
// object is destroyed.

VrmlField::~VrmlField() {}

VrmlField::VrmlFieldType
VrmlField::fieldType() const { return NO_FIELD; }

// A static method to convert a type name to an ID.

VrmlField::VrmlFieldType
VrmlField::fieldType(const char *type)
{
    if (strcmp(type, "SFBool") == 0)
        return SFBOOL;
    if (strcmp(type, "SFColor") == 0)
        return SFCOLOR;
    if (strcmp(type, "SFColorRGBA") == 0)
        return SFCOLORRGBA;
    if (strcmp(type, "SFDouble") == 0)
        return SFDOUBLE;
    if (strcmp(type, "SFFloat") == 0)
        return SFFLOAT;
    if (strcmp(type, "SFImage") == 0)
        return SFIMAGE;
    if (strcmp(type, "SFInt32") == 0)
        return SFINT32;
    if (strcmp(type, "SFNode") == 0)
        return SFNODE;
    if (strcmp(type, "SFRotation") == 0)
        return SFROTATION;
    if (strcmp(type, "SFString") == 0)
        return SFSTRING;
    if (strcmp(type, "SFTime") == 0)
        return SFTIME;
    if (strcmp(type, "SFVec2d") == 0)
        return SFVEC2D;
    if (strcmp(type, "SFVec3d") == 0)
        return SFVEC3D;
    if (strcmp(type, "SFVec2f") == 0)
        return SFVEC2F;
    if (strcmp(type, "SFVec3f") == 0)
        return SFVEC3F;
    if (strcmp(type, "MFBool") == 0)
        return MFBOOL;
    if (strcmp(type, "MFColor") == 0)
        return MFCOLOR;
    if (strcmp(type, "MFColorRGBA") == 0)
        return MFCOLORRGBA;
    if (strcmp(type, "MFDouble") == 0)
        return MFDOUBLE;
    if (strcmp(type, "MFFloat") == 0)
        return MFFLOAT;
    if (strcmp(type, "MFInt32") == 0)
        return MFINT32;
    if (strcmp(type, "MFNode") == 0)
        return MFNODE;
    if (strcmp(type, "MFRotation") == 0)
        return MFROTATION;
    if (strcmp(type, "MFString") == 0)
        return MFSTRING;
    if (strcmp(type, "MFTime") == 0)
        return MFTIME;
    if (strcmp(type, "MFVec2d") == 0)
        return MFVEC2D;
    if (strcmp(type, "MFVec3d") == 0)
        return MFVEC3D;
    if (strcmp(type, "MFVec2f") == 0)
        return MFVEC2F;
    if (strcmp(type, "MFVec3f") == 0)
        return MFVEC3F;

    return NO_FIELD;
}

static const char *ftn[] = {
    "SFBool",
    "SFColor",
    "SFColorRGBA",
    "SFDouble",
    "SFFloat",
    "SFInt32",
    "SFRotation",
    "SFTime",
    "SFVec2d",
    "SFVec3d",
    "SFVec2f",
    "SFVec3f",
    "SFImage",
    "SFString",
    "MFBool",
    "MFColor",
    "MFColorRGBA",
    "MFDouble",
    "MFFloat",
    "MFInt32",
    "MFRotation",
    "MFString",
    "MFTime",
    "MFVec2d",
    "MFVec3d",
    "MFVec2f",
    "MFVec3f",
    "SFNode",
    "MFNode"
};

// Return the type name of a field

const char *
VrmlField::fieldTypeName() const
{
    int ft = (int)this->fieldType();
    if (ft > 0 && ft <= (int)VrmlField::MFNODE)
        return ftn[ft - 1];
    return "<invalid field type>";
}

// Printing methods
namespace vrml
{
ostream &operator<<(ostream &os, const VrmlField &f)
{
    return f.print(os);
}
}
// SFBool

#include "VrmlSFBool.h"

VrmlSFBool::VrmlSFBool(bool value)
    : d_value(value)
{
}

ostream &VrmlSFBool::print(ostream &os) const
{
    return (os << (d_value ? "TRUE" : "FALSE"));
}

VrmlField *VrmlSFBool::clone() const { return new VrmlSFBool(d_value); }

VrmlField::VrmlFieldType VrmlSFBool::fieldType() const { return SFBOOL; }

// SFColor

#include "VrmlSFColor.h"

VrmlSFColor::VrmlSFColor(float r, float g, float b)
{
    d_rgb[0] = r;
    d_rgb[1] = g;
    d_rgb[2] = b;
}

ostream &VrmlSFColor::print(ostream &os) const
{
    return (os << d_rgb[0] << ' ' << d_rgb[1] << ' ' << d_rgb[2]);
}

VrmlField *VrmlSFColor::clone() const
{
    return new VrmlSFColor(d_rgb[0], d_rgb[1], d_rgb[2]);
}

VrmlField::VrmlFieldType VrmlSFColor::fieldType() const { return SFCOLOR; }

// Conversion functions between RGB each in [0,1] and HSV with
// h in [0,360), s,v in [0,1]. From Foley, van Dam p615-616.

void VrmlSFColor::HSVtoRGB(float h, float s, float v,
                           float &r, float &g, float &b)
{
    if (s == 0.0)
    {
        r = g = b = v;
    }
    else
    {
        if (h >= 360.0)
            h -= 360.0;
        h /= 60.0;
        float i = (float)floor(h);
        float f = h - i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));
        switch ((int)i)
        {
        default:
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        }
    }
}

void VrmlSFColor::RGBtoHSV(float r, float g, float b,
                           float &h, float &s, float &v)
{
    float maxrgb = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
    float minrgb = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);

    h = 0.0;
    v = maxrgb;
    if (maxrgb > 0.0)
        s = (maxrgb - minrgb) / maxrgb;
    else
        s = 0.0;

    if (s != 0.0)
    {
        float rc = (maxrgb - r) / (maxrgb - minrgb);
        float gc = (maxrgb - g) / (maxrgb - minrgb);
        float bc = (maxrgb - b) / (maxrgb - minrgb);
        if (r == maxrgb)
            h = bc - gc;
        else if (g == maxrgb)
            h = 2 + rc - bc;
        else if (b == maxrgb)
            h = 4 + gc - rc;

        h *= 60.0;
        if (h < 0.0)
            h += 360.0;
    }
}

void VrmlSFColor::setHSV(float h, float s, float v)
{
    HSVtoRGB(h, s, v, d_rgb[0], d_rgb[1], d_rgb[2]);
}

void VrmlSFColor::getHSV(float &h, float &s, float &v)
{
    RGBtoHSV(d_rgb[0], d_rgb[1], d_rgb[2], h, s, v);
}

// SFColorRGBA

#include "VrmlSFColorRGBA.h"

VrmlSFColorRGBA::VrmlSFColorRGBA(float r, float g, float b, float a)
{
    d_rgba[0] = r;
    d_rgba[1] = g;
    d_rgba[2] = b;
    d_rgba[3] = a;
}

ostream &VrmlSFColorRGBA::print(ostream &os) const
{
    return (os << d_rgba[0] << ' ' << d_rgba[1] << ' ' << d_rgba[2] << ' ' << d_rgba[3]);
}

VrmlField *VrmlSFColorRGBA::clone() const
{
    return new VrmlSFColorRGBA(d_rgba[0], d_rgba[1], d_rgba[2], d_rgba[3]);
}

VrmlField::VrmlFieldType VrmlSFColorRGBA::fieldType() const { return SFCOLORRGBA; }

// Conversion functions between RGB each in [0,1] and HSV with
// h in [0,360), s,v in [0,1]. From Foley, van Dam p615-616.

void VrmlSFColorRGBA::HSVtoRGB(float h, float s, float v,
                               float &r, float &g, float &b)
{
    if (s == 0.0)
    {
        r = g = b = v;
    }
    else
    {
        if (h >= 360.0)
            h -= 360.0;
        h /= 60.0;
        float i = (float)floor(h);
        float f = h - i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));
        switch ((int)i)
        {
        default:
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        }
    }
}

void VrmlSFColorRGBA::RGBtoHSV(float r, float g, float b,
                               float &h, float &s, float &v)
{
    float maxrgb = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
    float minrgb = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);

    h = 0.0;
    v = maxrgb;
    if (maxrgb > 0.0)
        s = (maxrgb - minrgb) / maxrgb;
    else
        s = 0.0;

    if (s != 0.0)
    {
        float rc = (maxrgb - r) / (maxrgb - minrgb);
        float gc = (maxrgb - g) / (maxrgb - minrgb);
        float bc = (maxrgb - b) / (maxrgb - minrgb);
        if (r == maxrgb)
            h = bc - gc;
        else if (g == maxrgb)
            h = 2 + rc - bc;
        else if (b == maxrgb)
            h = 4 + gc - rc;

        h *= 60.0;
        if (h < 0.0)
            h += 360.0;
    }
}

void VrmlSFColorRGBA::setHSV(float h, float s, float v)
{
    HSVtoRGB(h, s, v, d_rgba[0], d_rgba[1], d_rgba[2]);
}

void VrmlSFColorRGBA::getHSV(float &h, float &s, float &v)
{
    RGBtoHSV(d_rgba[0], d_rgba[1], d_rgba[2], h, s, v);
}

// SFDouble

#include "VrmlSFDouble.h"

VrmlSFDouble::VrmlSFDouble(double value)
    : d_value(value)
{
}

ostream &VrmlSFDouble::print(ostream &os) const
{
    return (os << d_value);
}

VrmlField *VrmlSFDouble::clone() const { return new VrmlSFDouble(d_value); }

VrmlField::VrmlFieldType VrmlSFDouble::fieldType() const { return SFDOUBLE; }

// SFFloat

#include "VrmlSFFloat.h"

VrmlSFFloat::VrmlSFFloat(float value)
    : d_value(value)
{
}

ostream &VrmlSFFloat::print(ostream &os) const
{
    return (os << d_value);
}

VrmlField *VrmlSFFloat::clone() const { return new VrmlSFFloat(d_value); }

VrmlField::VrmlFieldType VrmlSFFloat::fieldType() const { return SFFLOAT; }

// SFImage

#include "VrmlSFImage.h"

VrmlSFImage::VrmlSFImage(int w, int h, int nc, unsigned char *pixels)
    : d_w(w)
    , d_h(h)
    , d_nc(nc)
    , d_pixels(pixels)
{
}

VrmlSFImage::VrmlSFImage(const VrmlSFImage &sfi)
    : VrmlField()
    , d_w(sfi.d_w)
    , d_h(sfi.d_h)
    , d_nc(sfi.d_nc)
    , d_pixels(0)
{
    int nbytes = d_w * d_h * d_nc;
    if ((d_pixels = new unsigned char[nbytes]) != 0)
    {
        memcpy(d_pixels, sfi.d_pixels, nbytes);
    }
}

VrmlSFImage::~VrmlSFImage()
{
    delete[] d_pixels;
}

VrmlSFImage &VrmlSFImage::operator=(const VrmlSFImage &rhs)
{
    if (this == &rhs)
        return *this;
    if (d_pixels)
        delete[] d_pixels;
    d_w = d_h = d_nc = 0;
    int nbytes = rhs.d_w * rhs.d_h * rhs.d_nc;
    if ((d_pixels = new unsigned char[nbytes]) != 0)
    {
        d_w = rhs.d_w;
        d_h = rhs.d_h;
        d_nc = rhs.d_nc;
        memcpy(d_pixels, rhs.d_pixels, nbytes);
    }
    return *this;
}

ostream &VrmlSFImage::print(ostream &os) const
{
    os << d_w << " " << d_h << " " << d_nc;

    int np = d_w * d_h;
    unsigned char *p = d_pixels;

    for (int i = 0; i < np; ++i)
    {
        unsigned int pixval = 0;
        for (int j = 0; j < d_nc; ++j)
            pixval = (pixval << 8) | *p++;
        os << pixval << " ";
    }
    return os;
}

VrmlField *VrmlSFImage::clone() const
{
    VrmlSFImage *i = new VrmlSFImage();
    *i = *this;
    return i;
}

VrmlField::VrmlFieldType VrmlSFImage::fieldType() const { return SFIMAGE; }

// SFInt

#include "VrmlSFInt.h"

VrmlSFInt::VrmlSFInt(int value)
    : d_value(value)
{
}

ostream &VrmlSFInt::print(ostream &os) const
{
    return (os << d_value);
}

VrmlField *VrmlSFInt::clone() const { return new VrmlSFInt(d_value); }

VrmlField::VrmlFieldType VrmlSFInt::fieldType() const { return SFINT32; }

VrmlSFNode::VrmlSFNode(VrmlNode *value)
    : d_value(value)
{
    if (d_value)
        d_value->reference();
}

VrmlSFNode::VrmlSFNode(const VrmlSFNode &n)
    : VrmlField()
    , d_value(n.d_value)
{
    if (d_value)
        d_value->reference();
}

VrmlSFNode::~VrmlSFNode()
{
    if (d_value)
        d_value->dereference();
}

VrmlSFNode &VrmlSFNode::operator=(const VrmlSFNode &rhs)
{
    if (this == &rhs)
        return *this;
    if (d_value)
        d_value->dereference();
    d_value = rhs.d_value;
    if (d_value)
        d_value->reference();
    return *this;
}

ostream &VrmlSFNode::print(ostream &os) const
{
    return os << *(d_value) << endl;
}

VrmlField *VrmlSFNode::clone() const { return new VrmlSFNode(d_value); }

VrmlField::VrmlFieldType VrmlSFNode::fieldType() const { return SFNODE; }

void VrmlSFNode::set(VrmlNode *value)
{
    if (d_value)
        d_value->dereference();
    if ((d_value = value) != 0)
        d_value->reference();
}

// SFRotation

#include "VrmlSFRotation.h"

VrmlSFRotation::VrmlSFRotation(float x, float y, float z, float r)
{
    d_x[0] = x;
    d_x[1] = y;
    d_x[2] = z;
    d_x[3] = r;
}

ostream &VrmlSFRotation::print(ostream &os) const
{
    return (os << d_x[0] << " " << d_x[1] << " " << d_x[2] << " " << d_x[3]);
}

VrmlField *VrmlSFRotation::clone() const
{
    return new VrmlSFRotation(d_x[0], d_x[1], d_x[2], d_x[3]);
}

VrmlField::VrmlFieldType VrmlSFRotation::fieldType() const
{
    return SFROTATION;
}

void VrmlSFRotation::invert(void)
{
    d_x[3] = -d_x[3];
}

void VrmlSFRotation::multiply(VrmlSFRotation *)
{
    // not implemented...
}

void VrmlSFRotation::slerp(VrmlSFRotation *dest, float t)
{
    float fromQuat[4], toQuat[4];
    float a_2 = d_x[3] / 2.0f;
    float s_a_2 = sin(a_2);
    fromQuat[0] = d_x[0] * s_a_2;
    fromQuat[1] = d_x[1] * s_a_2;
    fromQuat[2] = d_x[2] * s_a_2;
    fromQuat[3] = cos(a_2);

    a_2 = dest->d_x[3] / 2.0f;
    s_a_2 = sin(a_2);
    toQuat[0] = dest->d_x[0] * s_a_2;
    toQuat[1] = dest->d_x[1] * s_a_2;
    toQuat[2] = dest->d_x[2] * s_a_2;
    toQuat[3] = cos(a_2);

    if (fromQuat[3] == 0)
    {
        fromQuat[0] = toQuat[0];
        fromQuat[1] = toQuat[1];
        fromQuat[2] = toQuat[2];
    }
    if (toQuat[3] == 0)
    {
        toQuat[0] = fromQuat[0];
        toQuat[1] = fromQuat[1];
        toQuat[2] = fromQuat[2];
    }
    //
    // Calculate cosine.
    //
    double cosom = (fromQuat[0] * toQuat[0]) + (fromQuat[1] * toQuat[1]) + (fromQuat[2] * toQuat[2]) + (fromQuat[3] * toQuat[3]);

    double acosvec = (fromQuat[0] * toQuat[0]) + (fromQuat[1] * toQuat[1]) + (fromQuat[2] * toQuat[2]);

    if (acosvec < 0.0)
    {
        acosvec = -acosvec;
    }
    //
    // Adjust signs (if necessary).
    //
    float to1[4];
    if (cosom < 0.0)
    {
        cosom = -cosom;
        to1[0] = -toQuat[0];
        to1[1] = -toQuat[1];
        to1[2] = -toQuat[2];
        to1[3] = -toQuat[3];
    }
    else
    {
        to1[0] = toQuat[0];
        to1[1] = toQuat[1];
        to1[2] = toQuat[2];
        to1[3] = toQuat[3];
    }
    //cerr << "fromQuat " << fromQuat[0] << ";"<< fromQuat[1] << ";"<< fromQuat[2] << ";"<< fromQuat[3] << ";" << endl;
    //cerr << "toQuat " << toQuat[0] << ";"<< toQuat[1] << ";"<< toQuat[2] << ";"<< toQuat[3] << ";" << endl;
    //cerr << "acosvec" << acosvec << endl;

    //
    // Calculate coefficients.
    //
    double scale0, scale1;
    /*if ((1.0 - acosvec) > 1.0e-2)
   {*/
    const double omega = acos(cosom);
    const double sinom = sin(omega);
    if (sinom < 0.00001 && sinom > -0.00001)
    {
        scale0 = 1;
        scale1 = 0;
    }
    else
    {
        scale0 = sin((1.0 - t) * omega) / sinom;
        scale1 = sin(t * omega) / sinom;
    }
    /* }
   else
   {
      //
      // "From" and "to" quaternions are very close, so do linear
      // interpolation.
      //
      scale0 = 1.0 - t;
      scale1 = t;
   }*/

    //
    // Calculate the final values.
    //
    float resultQuat[4];
    resultQuat[0] = (float)((scale0 * fromQuat[0]) + (scale1 * to1[0]));
    resultQuat[1] = (float)((scale0 * fromQuat[1]) + (scale1 * to1[1]));
    resultQuat[2] = (float)((scale0 * fromQuat[2]) + (scale1 * to1[2]));
    resultQuat[3] = (float)((scale0 * fromQuat[3]) + (scale1 * to1[3]));
    a_2 = acos(resultQuat[3]);
    s_a_2 = sin(a_2);
    if (s_a_2 < 0.000001)
        dest->set(1, 0, 0, a_2 * 2);
    else
        dest->set(resultQuat[0] / s_a_2, resultQuat[1] / s_a_2, resultQuat[2] / s_a_2, a_2 * 2);
}

// SFString

#include "VrmlSFString.h"

VrmlSFString::VrmlSFString(const char *s)
{
    if (s)
    {
        d_s = new char[strlen(s) + 1];
        strcpy(d_s, s);
    }
    else
        d_s = 0;
}

VrmlSFString::VrmlSFString(const VrmlSFString &sfs)
    : VrmlField()
{
    const char *s = sfs.get();
    if (s)
    {
        d_s = new char[strlen(s) + 1];
        strcpy(d_s, s);
    }
    else
        d_s = 0;
}

VrmlSFString::~VrmlSFString()
{
    if (d_s)
        delete[] d_s;
}

void VrmlSFString::set(const char *s)
{
    if (d_s)
        delete[] d_s;
    if (s)
    {
        d_s = new char[strlen(s) + 1];
        strcpy(d_s, s);
    }
    else
        d_s = 0;
}

// Assignment. Just reallocate for now...
VrmlSFString &VrmlSFString::operator=(const VrmlSFString &rhs)
{
    if (this != &rhs)
        set(rhs.d_s);
    return *this;
}

ostream &VrmlSFString::print(ostream &os) const
{
    return (os << '\"' << d_s << '\"');
}

VrmlField *VrmlSFString::clone() const { return new VrmlSFString(d_s); }

VrmlField::VrmlFieldType VrmlSFString::fieldType() const { return SFSTRING; }

// SFTime

#include "VrmlSFTime.h"

VrmlSFTime::VrmlSFTime(double value)
    : d_value(value)
{
}
VrmlSFTime::VrmlSFTime(const VrmlSFTime &c)
    : VrmlField()
    , d_value(c.d_value)
{
}

ostream &VrmlSFTime::print(ostream &os) const
{
    return (os << d_value);
}

VrmlField *VrmlSFTime::clone() const { return new VrmlSFTime(d_value); }

VrmlField::VrmlFieldType VrmlSFTime::fieldType() const { return SFTIME; }

VrmlSFTime &VrmlSFTime::operator=(const VrmlSFTime &rhs)
{
    if (this != &rhs)
        set(rhs.d_value);
    return *this;
}

VrmlSFTime &VrmlSFTime::operator=(double rhs)
{
    set(rhs);
    return *this;
}

// SFVec2d

#include "VrmlSFVec2d.h"

VrmlSFVec2d::VrmlSFVec2d(double x, double y)
{
    d_x[0] = x;
    d_x[1] = y;
}

ostream &VrmlSFVec2d::print(ostream &os) const
{
    return (os << d_x[0] << " " << d_x[1]);
}

VrmlField *VrmlSFVec2d::clone() const
{
    return new VrmlSFVec2d(d_x[0], d_x[1]);
}

VrmlField::VrmlFieldType VrmlSFVec2d::fieldType() const { return SFVEC2D; }

double VrmlSFVec2d::dot(VrmlSFVec2d *v)
{
    return d_x[0] * v->x() + d_x[1] * v->y();
}

double VrmlSFVec2d::length()
{
    return sqrt(d_x[0] * d_x[0] + d_x[1] * d_x[1]);
}

void VrmlSFVec2d::normalize()
{
    double len = length();
    if (FPZERO(len))
        return;
    d_x[0] /= len;
    d_x[1] /= len;
}

void VrmlSFVec2d::add(VrmlSFVec2d *v)
{
    d_x[0] += v->x();
    d_x[1] += v->y();
}

void VrmlSFVec2d::divide(double f)
{
    d_x[0] /= f;
    d_x[1] /= f;
}

void VrmlSFVec2d::multiply(double f)
{
    d_x[0] *= f;
    d_x[1] *= f;
}

void VrmlSFVec2d::subtract(VrmlSFVec2d *v)
{
    d_x[0] -= v->x();
    d_x[1] -= v->y();
}

// SFVec2f

#include "VrmlSFVec2f.h"

VrmlSFVec2f::VrmlSFVec2f(float x, float y)
{
    d_x[0] = x;
    d_x[1] = y;
}

ostream &VrmlSFVec2f::print(ostream &os) const
{
    return (os << d_x[0] << " " << d_x[1]);
}

VrmlField *VrmlSFVec2f::clone() const
{
    return new VrmlSFVec2f(d_x[0], d_x[1]);
}

VrmlField::VrmlFieldType VrmlSFVec2f::fieldType() const { return SFVEC2F; }

double VrmlSFVec2f::dot(VrmlSFVec2f *v)
{
    return d_x[0] * v->x() + d_x[1] * v->y();
}

double VrmlSFVec2f::length()
{
    return sqrt(d_x[0] * d_x[0] + d_x[1] * d_x[1]);
}

void VrmlSFVec2f::normalize()
{
    float len = (float)length();
    if (FPZERO(len))
        return;
    d_x[0] /= len;
    d_x[1] /= len;
}

void VrmlSFVec2f::add(VrmlSFVec2f *v)
{
    d_x[0] += v->x();
    d_x[1] += v->y();
}

void VrmlSFVec2f::divide(float f)
{
    d_x[0] /= f;
    d_x[1] /= f;
}

void VrmlSFVec2f::multiply(float f)
{
    d_x[0] *= f;
    d_x[1] *= f;
}

void VrmlSFVec2f::subtract(VrmlSFVec2f *v)
{
    d_x[0] -= v->x();
    d_x[1] -= v->y();
}

// SFVec3d

#include "VrmlSFVec3d.h"

VrmlSFVec3d::VrmlSFVec3d(double x, double y, double z)
{
    d_x[0] = x;
    d_x[1] = y;
    d_x[2] = z;
}

ostream &VrmlSFVec3d::print(ostream &os) const
{
    return (os << d_x[0] << " " << d_x[1] << " " << d_x[2]);
}

VrmlField *VrmlSFVec3d::clone() const
{
    return new VrmlSFVec3d(d_x[0], d_x[1], d_x[2]);
}

VrmlField::VrmlFieldType VrmlSFVec3d::fieldType() const { return SFVEC3D; }

double VrmlSFVec3d::dot(VrmlSFVec3d *v)
{
    return d_x[0] * v->x() + d_x[1] * v->y() + d_x[2] * v->z();
}

double VrmlSFVec3d::length()
{
    return sqrt(d_x[0] * d_x[0] + d_x[1] * d_x[1] + d_x[2] * d_x[2]);
}

void VrmlSFVec3d::normalize()
{
    double len = length();
    if (FPZERO(len))
        return;
    d_x[0] /= len;
    d_x[1] /= len;
    d_x[2] /= len;
}

void VrmlSFVec3d::cross(VrmlSFVec3d *v)
{
    double x, y, z; // Use temps so V can be A or B
    x = d_x[1] * v->z() - d_x[2] * v->y();
    y = d_x[2] * v->x() - d_x[0] * v->z();
    z = d_x[0] * v->y() - d_x[1] * v->x();
    d_x[0] = x;
    d_x[1] = y;
    d_x[2] = z;
}

void VrmlSFVec3d::add(VrmlSFVec3d *v)
{
    d_x[0] += v->x();
    d_x[1] += v->y();
    d_x[2] += v->z();
}

void VrmlSFVec3d::divide(double f)
{
    d_x[0] /= f;
    d_x[1] /= f;
    d_x[2] /= f;
}

void VrmlSFVec3d::multiply(double f)
{
    d_x[0] *= f;
    d_x[1] *= f;
    d_x[2] *= f;
}

void VrmlSFVec3d::subtract(VrmlSFVec3d *v)
{
    d_x[0] -= v->x();
    d_x[1] -= v->y();
    d_x[2] -= v->z();
}

// SFVec3f

#include "VrmlSFVec3f.h"

VrmlSFVec3f::VrmlSFVec3f(float x, float y, float z)
{
    d_x[0] = x;
    d_x[1] = y;
    d_x[2] = z;
}

ostream &VrmlSFVec3f::print(ostream &os) const
{
    return (os << d_x[0] << " " << d_x[1] << " " << d_x[2]);
}

VrmlField *VrmlSFVec3f::clone() const
{
    return new VrmlSFVec3f(d_x[0], d_x[1], d_x[2]);
}

VrmlField::VrmlFieldType VrmlSFVec3f::fieldType() const { return SFVEC3F; }

double VrmlSFVec3f::dot(VrmlSFVec3f *v)
{
    return d_x[0] * v->x() + d_x[1] * v->y() + d_x[2] * v->z();
}

double VrmlSFVec3f::length()
{
    return sqrt(d_x[0] * d_x[0] + d_x[1] * d_x[1] + d_x[2] * d_x[2]);
}

void VrmlSFVec3f::normalize()
{
    float len = (float)length();
    if (FPZERO(len))
        return;
    d_x[0] /= len;
    d_x[1] /= len;
    d_x[2] /= len;
}

void VrmlSFVec3f::cross(VrmlSFVec3f *v)
{
    float x, y, z; // Use temps so V can be A or B
    x = d_x[1] * v->z() - d_x[2] * v->y();
    y = d_x[2] * v->x() - d_x[0] * v->z();
    z = d_x[0] * v->y() - d_x[1] * v->x();
    d_x[0] = x;
    d_x[1] = y;
    d_x[2] = z;
}

void VrmlSFVec3f::add(VrmlSFVec3f *v)
{
    d_x[0] += v->x();
    d_x[1] += v->y();
    d_x[2] += v->z();
}

void VrmlSFVec3f::divide(float f)
{
    d_x[0] /= f;
    d_x[1] /= f;
    d_x[2] /= f;
}

void VrmlSFVec3f::multiply(float f)
{
    d_x[0] *= f;
    d_x[1] *= f;
    d_x[2] *= f;
}

void VrmlSFVec3f::subtract(VrmlSFVec3f *v)
{
    d_x[0] -= v->x();
    d_x[1] -= v->y();
    d_x[2] -= v->z();
}

// MFBool

#include "VrmlMFBool.h"

VrmlMFBool::VrmlMFBool()
    : d_data(new BData(false))
{
}

VrmlMFBool::VrmlMFBool(bool value)
    : d_data(new BData(true))
{
    d_data->d_v[0] = value;
}

VrmlMFBool::VrmlMFBool(int n, bool *v)
    : d_data(new BData(n))
{
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(bool));
}

VrmlMFBool::VrmlMFBool(const VrmlMFBool &src)
    : VrmlField()
    , d_data(src.d_data->ref())
{
}

VrmlMFBool::~VrmlMFBool() { d_data->deref(); }

void VrmlMFBool::set(int n, bool *v)
{
    d_data->deref();
    d_data = new BData(n);
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(bool));
}

VrmlMFBool &VrmlMFBool::operator=(const VrmlMFBool &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFBool::clone() const { return new VrmlMFBool(*this); }

VrmlField::VrmlFieldType VrmlMFBool::fieldType() const { return MFBOOL; }

// MFColor

#include "VrmlMFColor.h"

VrmlMFColor::VrmlMFColor()
    : d_data(new FData(0))
{
}

VrmlMFColor::VrmlMFColor(float r, float g, float b)
    : d_data(new FData(3))
{
    d_data->d_v[0] = r;
    d_data->d_v[1] = g;
    d_data->d_v[2] = b;
}

VrmlMFColor::VrmlMFColor(int n, float *v)
    : d_data(new FData(3 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 3 * n * sizeof(float));
}

VrmlMFColor::VrmlMFColor(const VrmlMFColor &source)
    : VrmlField()
    , d_data(source.d_data->ref())
{
}

VrmlMFColor::~VrmlMFColor() { d_data->deref(); }

void VrmlMFColor::set(int n, float *v)
{
    d_data->deref();
    d_data = new FData(3 * n);
    if (v)
        memcpy(d_data->d_v, v, 3 * n * sizeof(float));
}

VrmlMFColor &VrmlMFColor::operator=(const VrmlMFColor &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFColor::clone() const { return new VrmlMFColor(*this); }

VrmlField::VrmlFieldType VrmlMFColor::fieldType() const { return MFCOLOR; }

// MFColorRGBA

#include "VrmlMFColorRGBA.h"

VrmlMFColorRGBA::VrmlMFColorRGBA()
    : d_data(new FData(0))
{
}

VrmlMFColorRGBA::VrmlMFColorRGBA(float r, float g, float b, float a)
    : d_data(new FData(3))
{
    d_data->d_v[0] = r;
    d_data->d_v[1] = g;
    d_data->d_v[2] = b;
    d_data->d_v[2] = a;
}

VrmlMFColorRGBA::VrmlMFColorRGBA(int n, float *v)
    : d_data(new FData(4 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 4 * n * sizeof(float));
}

VrmlMFColorRGBA::VrmlMFColorRGBA(const VrmlMFColorRGBA &source)
    : VrmlField()
    , d_data(source.d_data->ref())
{
}

VrmlMFColorRGBA::~VrmlMFColorRGBA() { d_data->deref(); }

void VrmlMFColorRGBA::set(int n, float *v)
{
    d_data->deref();
    d_data = new FData(3 * n);
    if (v)
        memcpy(d_data->d_v, v, 4 * n * sizeof(float));
}

VrmlMFColorRGBA &VrmlMFColorRGBA::operator=(const VrmlMFColorRGBA &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFColorRGBA::clone() const { return new VrmlMFColorRGBA(*this); }

VrmlField::VrmlFieldType VrmlMFColorRGBA::fieldType() const { return MFCOLORRGBA; }

// MFDouble

#include "VrmlMFDouble.h"

VrmlMFDouble::VrmlMFDouble()
    : d_data(new FData(0))
{
}

VrmlMFDouble::VrmlMFDouble(double value)
    : d_data(new FData(1))
{
    d_data->d_v[0] = value;
}

VrmlMFDouble::VrmlMFDouble(int n, double *v)
    : d_data(new FData(n))
{
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(double));
}

VrmlMFDouble::VrmlMFDouble(const VrmlMFDouble &src)
    : VrmlField()
    , d_data(src.d_data->ref())
{
}

VrmlMFDouble::~VrmlMFDouble() { d_data->deref(); }

void VrmlMFDouble::set(int n, double *v)
{
    d_data->deref();
    d_data = new FData(n);
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(double));
}

VrmlMFDouble &VrmlMFDouble::operator=(const VrmlMFDouble &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFDouble::clone() const { return new VrmlMFDouble(*this); }

VrmlField::VrmlFieldType VrmlMFDouble::fieldType() const { return MFDOUBLE; }

// MFFloat

#include "VrmlMFFloat.h"

VrmlMFFloat::VrmlMFFloat()
    : d_data(new FData(0))
{
}

VrmlMFFloat::VrmlMFFloat(float value)
    : d_data(new FData(1))
{
    d_data->d_v[0] = value;
}

VrmlMFFloat::VrmlMFFloat(int n, float *v)
    : d_data(new FData(n))
{
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(float));
}

VrmlMFFloat::VrmlMFFloat(const VrmlMFFloat &src)
    : VrmlField()
    , d_data(src.d_data->ref())
{
}

VrmlMFFloat::~VrmlMFFloat() { d_data->deref(); }

void VrmlMFFloat::set(int n, float *v)
{
    d_data->deref();
    d_data = new FData(n);
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(float));
}

VrmlMFFloat &VrmlMFFloat::operator=(const VrmlMFFloat &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFFloat::clone() const { return new VrmlMFFloat(*this); }

VrmlField::VrmlFieldType VrmlMFFloat::fieldType() const { return MFFLOAT; }

// MFInt

#include "VrmlMFInt.h"

VrmlMFInt::VrmlMFInt()
    : d_data(new IData(0))
{
}

VrmlMFInt::VrmlMFInt(int value)
    : d_data(new IData(1))
{
    d_data->d_v[0] = value;
}

VrmlMFInt::VrmlMFInt(int n, int *v)
    : d_data(new IData(n))
{
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(int));
}

VrmlMFInt::VrmlMFInt(const VrmlMFInt &src)
    : VrmlField()
    , d_data(src.d_data->ref())
{
}

VrmlMFInt::~VrmlMFInt() { d_data->deref(); }

void VrmlMFInt::set(int n, int *v)
{
    d_data->deref();
    d_data = new IData(n);
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(int));
}

VrmlMFInt &VrmlMFInt::operator=(const VrmlMFInt &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFInt::clone() const { return new VrmlMFInt(*this); }

VrmlField::VrmlFieldType VrmlMFInt::fieldType() const { return MFINT32; }

//  MFNode

#include "VrmlMFNode.h"

VrmlMFNode::VrmlMFNode()
    : d_v(0)
    , d_allocated(0)
    , d_size(0)
{
}

VrmlMFNode::VrmlMFNode(VrmlNode *value)
    : d_v(new VrmlNode *[1])
    , d_allocated(1)
    , d_size(1)
{
    d_v[0] = value ? value->reference() : 0;
}

VrmlMFNode::VrmlMFNode(int n, VrmlNode **v)
    : d_v(new VrmlNode *[n])
    , d_allocated(n)
    , d_size(n)
{
    if (v)
        for (int i = 0; i < n; ++i)
            d_v[i] = v[i] ? (v[i]->reference()) : 0;
    else
        memset(d_v, 0, n * sizeof(VrmlNode *));
}

VrmlMFNode::VrmlMFNode(const VrmlMFNode &rhs)
    : VrmlField()
    , d_v(new VrmlNode *[rhs.d_size])
    , d_allocated(rhs.d_size)
    , d_size(rhs.d_size)
{
    int n = rhs.d_size;
    for (int i = 0; i < n; ++i)
        d_v[i] = rhs.d_v[i] ? (rhs.d_v[i]->reference()) : 0;
}

VrmlMFNode::~VrmlMFNode()
{
    for (int i = 0; i < d_size; ++i)
        if (d_v[i])
            d_v[i]->dereference();
    delete[] d_v;
}

// Assignment. Since the nodes themselves are ref counted, we
// don't bother trying to share the NodeLists.
VrmlMFNode &VrmlMFNode::operator=(const VrmlMFNode &rhs)
{
    if (this == &rhs)
        return *this;

    int i;
    for (i = 0; i < d_size; ++i)
        if (d_v[i])
            d_v[i]->dereference();

    if (d_allocated < rhs.d_size)
    {
        delete[] d_v;
        d_size = d_allocated = 0;
        d_v = 0;
        d_v = new VrmlNode *[rhs.d_size];
        d_allocated = rhs.d_size;
    }

    d_size = rhs.d_size;

    for (i = 0; i < d_size; ++i)
        d_v[i] = rhs.d_v[i] ? rhs.d_v[i]->reference() : 0;

    return *this;
}

VrmlField *VrmlMFNode::clone() const
{
    return new VrmlMFNode(*this);
}

VrmlField::VrmlFieldType VrmlMFNode::fieldType() const { return MFNODE; }

bool VrmlMFNode::exists(VrmlNode *n)
{
    for (int i = 0; i < d_size; ++i)
        if (d_v[i] == n)
            return true;
    return false;
}

void VrmlMFNode::addNode(VrmlNode *n)
{
    if (!exists(n))
    {
        if (d_allocated < d_size + 1)
        {
            int newSize = d_allocated + 10; // ...
            VrmlNode **newNodes = new VrmlNode *[newSize];
            memcpy(newNodes, d_v, d_size * sizeof(VrmlNode *));
            d_allocated = newSize;
            delete[] d_v;
            d_v = newNodes;
        }
        d_v[d_size++] = n ? n->reference() : 0;
    }
}

void VrmlMFNode::removeNode(VrmlNode *n)
{
    for (int i = 0; i < d_size; ++i)
        if (d_v[i] == n)
        {
            if (i < d_size - 1)
                memmove(&d_v[i], &d_v[i + 1], (d_size - i - 1) * sizeof(VrmlNode *));
            if (n)
            {
                n->dereference();
            }
            --d_size;
            break;
        }
}

// MFRotation

#include "VrmlMFRotation.h"

VrmlMFRotation::VrmlMFRotation()
    : d_data(new FData(0))
{
}

VrmlMFRotation::VrmlMFRotation(float x, float y, float z, float r)
    : d_data(new FData(4))
{
    d_data->d_v[0] = x;
    d_data->d_v[1] = y;
    d_data->d_v[2] = z;
    d_data->d_v[3] = r;
}

VrmlMFRotation::VrmlMFRotation(int n, float *v)
    : d_data(new FData(4 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 4 * n * sizeof(float));
}

VrmlMFRotation::VrmlMFRotation(const VrmlMFRotation &src)
    : VrmlField()
    , d_data(src.d_data->ref())
{
}

VrmlMFRotation::~VrmlMFRotation() { d_data->deref(); }

void VrmlMFRotation::set(int n, float *v)
{
    d_data->deref();
    d_data = new FData(4 * n);
    if (v)
        memcpy(d_data->d_v, v, 4 * n * sizeof(float));
}

VrmlMFRotation &VrmlMFRotation::operator=(const VrmlMFRotation &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFRotation::clone() const { return new VrmlMFRotation(*this); }

VrmlField::VrmlFieldType VrmlMFRotation::fieldType() const { return MFROTATION; }

// MFString

#include "VrmlMFString.h"

VrmlMFString::VrmlMFString()
    : d_v(0)
    , d_allocated(0)
    , d_size(0)
{
}

VrmlMFString::VrmlMFString(const char *s)
    : d_v(new char *[1])
    , d_allocated(1)
    , d_size(1)
{
    if (s)
    {
        d_v[0] = new char[strlen(s) + 1];
        strcpy(d_v[0], s);
    }
    else
        d_v[0] = 0;
}

VrmlMFString::VrmlMFString(int n, const char **v)
    : d_v(new char *[n])
    , d_allocated(n)
    , d_size(n)
{
    if (v)
        for (int i = 0; i < n; ++i, ++v)
        {
            if (*v)
            {
                d_v[i] = new char[strlen(*v) + 1];
                strcpy(d_v[i], *v);
            }
            else
                d_v[i] = 0;
        }
    else
        memset(d_v, 0, n * sizeof(char *));
}

VrmlMFString::VrmlMFString(const VrmlMFString &rhs)
    : VrmlField()
    , d_v(new char *[rhs.d_size])
    , d_allocated(rhs.d_size)
    , d_size(rhs.d_size)
{
    int n = rhs.d_size;
    for (int i = 0; i < n; ++i)
    {
        if (rhs.d_v[i])
        {
            d_v[i] = new char[strlen(rhs.d_v[i]) + 1];
            strcpy(d_v[i], rhs.d_v[i]);
        }
        else
            d_v[i] = 0;
    }
}

VrmlMFString::~VrmlMFString()
{
    for (int i = 0; i < d_size; ++i)
        delete[] d_v[i];
    delete[] d_v;
}

void VrmlMFString::set(int n, const char *v[])
{
    for (int i = 0; i < d_size; ++i)
        delete[] d_v[i];

    if (d_allocated < n)
    {
        delete[] d_v;
        d_v = 0;
        d_allocated = d_size = 0;
        d_v = new char *[n];
        d_allocated = n;
    }
    d_size = n;

    for (int j = 0; j < n; ++j)
    {
        if (v[j])
        {
            d_v[j] = new char[strlen(v[j]) + 1];
            strcpy(d_v[j], v[j]);
        }
        else
            d_v[j] = 0;
    }
}

VrmlMFString &VrmlMFString::operator=(const VrmlMFString &rhs)
{
    if (this != &rhs)
        set(rhs.d_size, const_cast<const char **>(rhs.d_v));
    return *this;
}

VrmlField *VrmlMFString::clone() const
{
    return new VrmlMFString(*this);
}

VrmlField::VrmlFieldType VrmlMFString::fieldType() const { return MFSTRING; }

// MFTime

#include "VrmlMFTime.h"

VrmlMFTime::VrmlMFTime()
    : d_data(new FData(0))
{
}

VrmlMFTime::VrmlMFTime(double value)
    : d_data(new FData(1))
{
    d_data->d_v[0] = value;
}

VrmlMFTime::VrmlMFTime(int n, double *v)
    : d_data(new FData(n))
{
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(double));
}

VrmlMFTime::VrmlMFTime(const VrmlMFTime &src)
    : VrmlField()
    , d_data(src.d_data->ref())
{
}

VrmlMFTime::~VrmlMFTime() { d_data->deref(); }

void VrmlMFTime::set(int n, double *v)
{
    d_data->deref();
    d_data = new FData(n);
    if (v)
        memcpy(d_data->d_v, v, n * sizeof(double));
}

VrmlMFTime &VrmlMFTime::operator=(const VrmlMFTime &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFTime::clone() const { return new VrmlMFTime(*this); }

VrmlField::VrmlFieldType VrmlMFTime::fieldType() const { return MFTIME; }

// MFVec2d

#include "VrmlMFVec2d.h"

VrmlMFVec2d::VrmlMFVec2d()
    : d_data(new FData(0))
{
}

VrmlMFVec2d::VrmlMFVec2d(double x, double y)
    : d_data(new FData(2))
{
    d_data->d_v[0] = x;
    d_data->d_v[1] = y;
}

VrmlMFVec2d::VrmlMFVec2d(int n, double *v)
    : d_data(new FData(2 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 2 * n * sizeof(double));
}

VrmlMFVec2d::VrmlMFVec2d(const VrmlMFVec2d &source)
    : VrmlField()
    , d_data(source.d_data->ref())
{
}

VrmlMFVec2d::~VrmlMFVec2d() { d_data->deref(); }

void VrmlMFVec2d::set(int n, double *v)
{
    d_data->deref();
    d_data = new FData(2 * n);
    if (v)
        memcpy(d_data->d_v, v, 2 * n * sizeof(double));
}

VrmlMFVec2d &VrmlMFVec2d::operator=(const VrmlMFVec2d &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFVec2d::clone() const { return new VrmlMFVec2d(*this); }

VrmlField::VrmlFieldType VrmlMFVec2d::fieldType() const { return MFVEC2D; }

// MFVec2f

#include "VrmlMFVec2f.h"

VrmlMFVec2f::VrmlMFVec2f()
    : d_data(new FData(0))
{
}

VrmlMFVec2f::VrmlMFVec2f(float x, float y)
    : d_data(new FData(2))
{
    d_data->d_v[0] = x;
    d_data->d_v[1] = y;
}

VrmlMFVec2f::VrmlMFVec2f(int n, float *v)
    : d_data(new FData(2 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 2 * n * sizeof(float));
}

VrmlMFVec2f::VrmlMFVec2f(const VrmlMFVec2f &source)
    : VrmlField()
    , d_data(source.d_data->ref())
{
}

VrmlMFVec2f::~VrmlMFVec2f() { d_data->deref(); }

void VrmlMFVec2f::set(int n, float *v)
{
    d_data->deref();
    d_data = new FData(2 * n);
    if (v)
        memcpy(d_data->d_v, v, 2 * n * sizeof(float));
}

VrmlMFVec2f &VrmlMFVec2f::operator=(const VrmlMFVec2f &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFVec2f::clone() const { return new VrmlMFVec2f(*this); }

VrmlField::VrmlFieldType VrmlMFVec2f::fieldType() const { return MFVEC2F; }

// MFVec3d

#include "VrmlMFVec3d.h"

VrmlMFVec3d::VrmlMFVec3d(int n)
    : d_data(new FData(n))
{
}

VrmlMFVec3d::VrmlMFVec3d(double x, double y, double z)
    : d_data(new FData(3))
{
    d_data->d_v[0] = x;
    d_data->d_v[1] = y;
    d_data->d_v[2] = z;
}

VrmlMFVec3d::VrmlMFVec3d(int n, double *v)
    : d_data(new FData(3 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 3 * n * sizeof(double));
}

VrmlMFVec3d::VrmlMFVec3d(const VrmlMFVec3d &source)
    : VrmlField()
    , d_data(source.d_data->ref())
{
}

VrmlMFVec3d::~VrmlMFVec3d() { d_data->deref(); }

void VrmlMFVec3d::set(int n, double *v)
{
    d_data->deref();
    d_data = new FData(3 * n);
    if (v)
        memcpy(d_data->d_v, v, 3 * n * sizeof(double));
}

VrmlMFVec3d &VrmlMFVec3d::operator=(const VrmlMFVec3d &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFVec3d::clone() const { return new VrmlMFVec3d(*this); }

VrmlField::VrmlFieldType VrmlMFVec3d::fieldType() const { return MFVEC3D; }

// MFVec3f

#include "VrmlMFVec3f.h"

VrmlMFVec3f::VrmlMFVec3f(int n)
    : d_data(new FData(n))
{
}

VrmlMFVec3f::VrmlMFVec3f(float x, float y, float z)
    : d_data(new FData(3))
{
    d_data->d_v[0] = x;
    d_data->d_v[1] = y;
    d_data->d_v[2] = z;
}

VrmlMFVec3f::VrmlMFVec3f(int n, float *v)
    : d_data(new FData(3 * n))
{
    if (v)
        memcpy(d_data->d_v, v, 3 * n * sizeof(float));
}

VrmlMFVec3f::VrmlMFVec3f(const VrmlMFVec3f &source)
    : VrmlField()
    , d_data(source.d_data->ref())
{
}

VrmlMFVec3f::~VrmlMFVec3f() { d_data->deref(); }

void VrmlMFVec3f::set(int n, float *v)
{
    d_data->deref();
    d_data = new FData(3 * n);
    if (v)
        memcpy(d_data->d_v, v, 3 * n * sizeof(float));
}

VrmlMFVec3f &VrmlMFVec3f::operator=(const VrmlMFVec3f &rhs)
{
    if (this != &rhs)
    {
        d_data->deref();
        d_data = rhs.d_data->ref();
    }
    return *this;
}

VrmlField *VrmlMFVec3f::clone() const { return new VrmlMFVec3f(*this); }

VrmlField::VrmlFieldType VrmlMFVec3f::fieldType() const { return MFVEC3F; }

// Generic MF float print function

static ostream &mffprint(ostream &os, float *c, int n, int eltsize)
{
    int e;

    if (n == 1)
        for (e = 0; e < eltsize; ++e)
            os << c[e] << ((e < eltsize - 1) ? " " : "");

    else
    {
        os << '[';
        for (int i = 0; i < n; ++i, c += eltsize)
        {
            for (e = 0; e < eltsize; ++e)
                os << c[e] << ((e < eltsize - 1) ? " " : "");

            os << ((i < n - 1) ? ", " : " ");
        }
        os << ']';
    }

    return os;
}

ostream &VrmlMFColor::print(ostream &os) const
{
    return mffprint(os, get(), size(), 3);
}

ostream &VrmlMFColorRGBA::print(ostream &os) const
{
    return mffprint(os, get(), size(), 4);
}

ostream &VrmlMFFloat::print(ostream &os) const
{
    return mffprint(os, get(), size(), 1);
}

ostream &VrmlMFRotation::print(ostream &os) const
{
    return mffprint(os, get(), size(), 4);
}

ostream &VrmlMFVec2f::print(ostream &os) const
{
    return mffprint(os, get(), size(), 2);
}

ostream &VrmlMFVec3f::print(ostream &os) const
{
    return mffprint(os, get(), size(), 3);
}

// Generic MF double print function

static ostream &mfdprint(ostream &os, double *c, int n, int eltsize)
{
    int e;

    if (n == 1)
        for (e = 0; e < eltsize; ++e)
            os << c[e] << ((e < eltsize - 1) ? " " : "");

    else
    {
        os << '[';
        for (int i = 0; i < n; ++i, c += eltsize)
        {
            for (e = 0; e < eltsize; ++e)
                os << c[e] << ((e < eltsize - 1) ? " " : "");

            os << ((i < n - 1) ? ", " : " ");
        }
        os << ']';
    }

    return os;
}

ostream &VrmlMFDouble::print(ostream &os) const
{
    return mfdprint(os, get(), size(), 1);
}

ostream &VrmlMFTime::print(ostream &os) const
{
    return mfdprint(os, get(), size(), 1);
}

ostream &VrmlMFVec2d::print(ostream &os) const
{
    return mfdprint(os, get(), size(), 2);
}

ostream &VrmlMFVec3d::print(ostream &os) const
{
    return mfdprint(os, get(), size(), 3);
}

ostream &VrmlMFBool::print(ostream &os) const
{
    int n = size();
    bool *c = get();

    if (n == 1)
        os << (*c ? "TRUE" : "FALSE");
    else
    {
        os << '[';
        for (int i = 0; i < n; ++i, ++c)
        {
            os << (*c ? "TRUE" : "FALSE");
            os << ((i < n - 1) ? ", " : " ");
        }
        os << ']';
    }

    return os;
}

ostream &VrmlMFInt::print(ostream &os) const
{
    int n = size();
    int *c = get();

    if (n == 1)
        os << *c;
    else
    {
        os << '[';
        for (int i = 0; i < n; ++i, ++c)
        {
            os << *c;
            os << ((i < n - 1) ? ", " : " ");
        }
        os << ']';
    }

    return os;
}

ostream &VrmlMFNode::print(ostream &os) const
{
    int n = size();

    if (n != 1)
        os << '[';
    for (int i = 0; i < n; ++i)
        os << *(d_v[i]) << endl;
    if (n != 1)
        os << ']';

    return os;
}

ostream &VrmlMFString::print(ostream &os) const
{
    int n = size();

    if (n != 1)
        os << '[';
    for (int i = 0; i < n; ++i)
        os << '\"' << (d_v[i]) << "\" ";
    if (n != 1)
        os << ']';

    return os;
}

// Define the const and non-const generic and specific safe downcast methods

#define DOWNCAST(_t)                                          \
    const Vrml##_t *VrmlField::to##_t() const { return 0; }   \
    Vrml##_t *VrmlField::to##_t() { return 0; }               \
    const Vrml##_t *Vrml##_t::to##_t() const { return this; } \
    Vrml##_t *Vrml##_t::to##_t() { return this; }

DOWNCAST(SFBool)
DOWNCAST(SFColor)
DOWNCAST(SFColorRGBA)
DOWNCAST(SFDouble)
DOWNCAST(SFFloat)
DOWNCAST(SFImage)
DOWNCAST(SFInt)
DOWNCAST(SFNode)
DOWNCAST(SFRotation)
DOWNCAST(SFString)
DOWNCAST(SFTime)
DOWNCAST(SFVec2d)
DOWNCAST(SFVec3d)
DOWNCAST(SFVec2f)
DOWNCAST(SFVec3f)

DOWNCAST(MFBool)
DOWNCAST(MFColor)
DOWNCAST(MFColorRGBA)
DOWNCAST(MFDouble)
DOWNCAST(MFFloat)
DOWNCAST(MFInt)
DOWNCAST(MFNode)
DOWNCAST(MFRotation)
DOWNCAST(MFString)
DOWNCAST(MFTime)
DOWNCAST(MFVec2d)
DOWNCAST(MFVec3d)
DOWNCAST(MFVec2f)
DOWNCAST(MFVec3f)
