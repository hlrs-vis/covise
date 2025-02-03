/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  VrmlField.h
//  Each field type should support clone, get, and set methods. (Most of these
//  field types could be handled by SF and MF templates ...)
//

#ifndef _VRMLFIELD_
#define _VRMLFIELD_

#include "config.h"

#include <iostream>

namespace vrml
{

class VRMLEXPORT VrmlSFBool;
class VRMLEXPORT VrmlSFColor;
class VRMLEXPORT VrmlSFColorRGBA;
class VRMLEXPORT VrmlSFDouble;
class VRMLEXPORT VrmlSFFloat;
class VRMLEXPORT VrmlSFImage;
class VRMLEXPORT VrmlSFInt;
class VRMLEXPORT VrmlSFNode;
class VRMLEXPORT VrmlSFRotation;
class VRMLEXPORT VrmlSFString;
class VRMLEXPORT VrmlSFTime;
class VRMLEXPORT VrmlSFVec2d;
class VRMLEXPORT VrmlSFVec3d;
class VRMLEXPORT VrmlSFVec2f;
class VRMLEXPORT VrmlSFVec3f;
class VRMLEXPORT VrmlSFMatrix;

class VRMLEXPORT VrmlMFBool;
class VRMLEXPORT VrmlMFColor;
class VRMLEXPORT VrmlMFColorRGBA;
class VRMLEXPORT VrmlMFDouble;
class VRMLEXPORT VrmlMFFloat;
class VRMLEXPORT VrmlMFInt;
class VRMLEXPORT VrmlMFNode;
class VRMLEXPORT VrmlMFRotation;
class VRMLEXPORT VrmlMFString;
class VRMLEXPORT VrmlMFTime;
class VRMLEXPORT VrmlMFVec2d;
class VRMLEXPORT VrmlMFVec3d;
class VRMLEXPORT VrmlMFVec2f;
class VRMLEXPORT VrmlMFVec3f;

// Abstract base class for field values

class VRMLEXPORT VrmlField
{
    friend std::ostream &operator<<(std::ostream &os, const VrmlField &f);

public:
    // Field type identifiers
    typedef enum
    {
        NO_FIELD,
        SFBOOL,
        SFCOLOR,
        SFCOLORRGBA,
        SFDOUBLE,
        SFFLOAT,
        SFINT32,
        SFROTATION,
        SFTIME,
        SFVEC2D,
        SFVEC3D,
        SFVEC2F,
        SFVEC3F,
        SFIMAGE,
        SFSTRING,
        MFBOOL,
        MFCOLOR,
        MFCOLORRGBA,
        MFDOUBLE,
        MFFLOAT,
        MFINT32,
        MFROTATION,
        MFSTRING,
        MFTIME,
        MFVEC2D,
        MFVEC3D,
        MFVEC2F,
        MFVEC3F,
        SFNODE,
        MFNODE,
        SFMATRIX,
        SFFIELD
    } VrmlFieldType;

    // Constructors/destructor
    VrmlField();
    virtual ~VrmlField() = 0;

    // Copy self
    virtual VrmlField *clone() const = 0;

    // Write self
    virtual std::ostream &print(std::ostream &os) const = 0;

    // Field type
    virtual VrmlFieldType fieldType() const;

    // Field type name
    const char *fieldTypeName() const;

    // Name to field type
    static VrmlFieldType fieldType(const char *fieldTypeName);

    // safe downcasts, const and non-const versions.
    // These avoid casts of VrmlField* but are ugly in that this class
    // must know of the existence of all of its subclasses...

    virtual const VrmlSFBool *toSFBool() const;
    virtual const VrmlSFColor *toSFColor() const;
    virtual const VrmlSFColorRGBA *toSFColorRGBA() const;
    virtual const VrmlSFDouble *toSFDouble() const;
    virtual const VrmlSFFloat *toSFFloat() const;
    virtual const VrmlSFImage *toSFImage() const;
    virtual const VrmlSFInt *toSFInt() const;
    virtual const VrmlSFNode *toSFNode() const;
    virtual const VrmlSFRotation *toSFRotation() const;
    virtual const VrmlSFString *toSFString() const;
    virtual const VrmlSFTime *toSFTime() const;
    virtual const VrmlSFVec2d *toSFVec2d() const;
    virtual const VrmlSFVec3d *toSFVec3d() const;
    virtual const VrmlSFVec2f *toSFVec2f() const;
    virtual const VrmlSFVec3f* toSFVec3f() const;
    virtual const VrmlSFMatrix* toSFMatrix() const;

    virtual const VrmlMFBool *toMFBool() const;
    virtual const VrmlMFColor *toMFColor() const;
    virtual const VrmlMFColorRGBA *toMFColorRGBA() const;
    virtual const VrmlMFDouble *toMFDouble() const;
    virtual const VrmlMFFloat *toMFFloat() const;
    virtual const VrmlMFInt *toMFInt() const;
    virtual const VrmlMFNode *toMFNode() const;
    virtual const VrmlMFRotation *toMFRotation() const;
    virtual const VrmlMFString *toMFString() const;
    virtual const VrmlMFTime *toMFTime() const;
    virtual const VrmlMFVec2d *toMFVec2d() const;
    virtual const VrmlMFVec3d *toMFVec3d() const;
    virtual const VrmlMFVec2f *toMFVec2f() const;
    virtual const VrmlMFVec3f* toMFVec3f() const;

    virtual VrmlSFBool *toSFBool();
    virtual VrmlSFColor *toSFColor();
    virtual VrmlSFColorRGBA *toSFColorRGBA();
    virtual VrmlSFDouble *toSFDouble();
    virtual VrmlSFFloat *toSFFloat();
    virtual VrmlSFImage *toSFImage();
    virtual VrmlSFInt *toSFInt();
    virtual VrmlSFNode *toSFNode();
    virtual VrmlSFRotation *toSFRotation();
    virtual VrmlSFString *toSFString();
    virtual VrmlSFTime *toSFTime();
    virtual VrmlSFVec2d *toSFVec2d();
    virtual VrmlSFVec3d *toSFVec3d();
    virtual VrmlSFVec2f *toSFVec2f();
    virtual VrmlSFVec3f* toSFVec3f();
    virtual VrmlSFMatrix* toSFMatrix();

    virtual VrmlMFBool *toMFBool();
    virtual VrmlMFColor *toMFColor();
    virtual VrmlMFColorRGBA *toMFColorRGBA();
    virtual VrmlMFDouble *toMFDouble();
    virtual VrmlMFFloat *toMFFloat();
    virtual VrmlMFInt *toMFInt();
    virtual VrmlMFNode *toMFNode();
    virtual VrmlMFRotation *toMFRotation();
    virtual VrmlMFString *toMFString();
    virtual VrmlMFTime *toMFTime();
    virtual VrmlMFVec2d *toMFVec2d();
    virtual VrmlMFVec3d *toMFVec3d();
    virtual VrmlMFVec2f *toMFVec2f();
    virtual VrmlMFVec3f *toMFVec3f();
};
}

// Abstract base classes for single-valued & multi-valued fields
// So far they don't do anything, so they don't really exist yet,
// but I would like to make VrmlMFields be ref counted (I think)...

#define VrmlSField VrmlField
#define VrmlMField VrmlField
#endif //_VRMLFIELD_
