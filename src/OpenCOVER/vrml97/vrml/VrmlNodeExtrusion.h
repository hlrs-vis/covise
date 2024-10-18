/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//  %W% %G%
//  VrmlNodeExtrusion.h

#ifndef _VRMLNODEEXTRUSION_
#define _VRMLNODEEXTRUSION_

#include "VrmlNodeGeometry.h"

#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"

#include "VrmlMFRotation.h"
#include "VrmlMFVec2f.h"
#include "VrmlMFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeExtrusion : public VrmlNodeGeometry
{

public:
    // Define the fields of extrusion nodes
    static void initFields(VrmlNodeExtrusion *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeExtrusion(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *);

    // Larry
    virtual VrmlNodeExtrusion *toExtrusion() const;
    virtual bool getBeginCap()
    {
        return d_beginCap.get();
    }
    virtual bool getCcw()
    {
        return d_ccw.get();
    }
    virtual bool getConvex()
    {
        return d_convex.get();
    }
    virtual float getCreaseAngle()
    {
        return d_creaseAngle.get();
    }
    virtual VrmlMFVec2f &getCrossSection()
    {
        return d_crossSection;
    }
    virtual bool getEndCap()
    {
        return d_endCap.get();
    }
    virtual VrmlMFRotation &getOrientation()
    {
        return d_orientation;
    }
    virtual VrmlMFVec2f &getScale()
    {
        return d_scale;
    }
    virtual bool getSolid()
    {
        return d_solid.get();
    }
    virtual VrmlMFVec3f &getSpine()
    {
        return d_spine;
    }

protected:
    VrmlSFBool d_beginCap;
    VrmlSFBool d_ccw;
    VrmlSFBool d_convex;
    VrmlSFFloat d_creaseAngle;
    VrmlMFVec2f d_crossSection;
    VrmlSFBool d_endCap;
    VrmlMFRotation d_orientation;
    VrmlMFVec2f d_scale;
    VrmlSFBool d_solid;
    VrmlMFVec3f d_spine;
};
}
#endif //_VRMLNODEEXTRUSION_
