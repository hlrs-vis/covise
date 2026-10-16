/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeGeoreferenced.h

#ifndef _VRML_NODE_GEOREFERENCED_
#define _VRML_NODE_GEOREFERENCED_

#include "vrml97/vrml/VrmlNodeGroup.h"
#include "vrml97/vrml/VrmlSFVec3f.h"
#include <sigslot/signal.hpp>

#include <osg/Vec3>
#include <osg/Quat>

namespace vrml
{

class VrmlNodeGeoreferenced : public VrmlNodeGroup
{

public:
    // Define the fields of Transform nodes
    static void initFields(VrmlNodeGeoreferenced *node, VrmlNodeType *t);
    static const char *typeName();

    VrmlNodeGeoreferenced(const VrmlNodeGeoreferenced &);
    VrmlNodeGeoreferenced(VrmlScene *);

    virtual void render(Viewer *);

    osg::Quat getRotation();
    osg::Vec3 getTranslation();

    virtual void accumulateTransform(VrmlNode *) override;
    virtual void inverseTransform(Viewer *) override;
    virtual void inverseTransform(double *mat) override;

protected:
    /// The ENU location at which to place the children's origin.
    VrmlSFVec3f d_geolocation { 0, 0, 0 };

    /// Set to true to flip X and Y axes, useful when inlining non-VRML data
    /// into VRML, which has a different convention.
    VrmlSFBool d_fixAxes { false };

    /// To use a different EPSG for the "geolocation" field, specify it here,
    /// preferably as an `ESPG:<code>` string (or anything else that gdal understands).
    /// Only transforms the geolocation, possible rotations are lost.
    VrmlSFString d_geolocationReference;

    // TODO
    // VrmlSFFloat d_northAngle;

    // TODO
    // VrmlSFFloat d_scale;

    Viewer::Object d_xformObject { 0 };
    sigslot::scoped_connection m_connection;
};
}
#endif //_VRMLNODETRANSFORM_
