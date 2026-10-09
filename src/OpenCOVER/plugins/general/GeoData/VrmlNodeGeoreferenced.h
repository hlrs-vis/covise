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

class VRMLEXPORT VrmlNodeGeoreferenced : public VrmlNodeGroup
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

    virtual void accumulateTransform(VrmlNode *);
    virtual void inverseTransform(Viewer *);
    virtual void inverseTransform(double *mat);

protected:
    VrmlSFVec3f d_geolocation;
    VrmlSFBool d_fixAxes;

    // VrmlSFString d_geolocationReference;
    // VrmlSFFloat d_north;
    // VrmlSFFloat d_scale;

    Viewer::Object d_xformObject;

    sigslot::scoped_connection m_connection;
};
}
#endif //_VRMLNODETRANSFORM_
