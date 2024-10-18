/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePrecipitation.h

#ifndef _VRMLNODEPrecipitation_
#define _VRMLNODEPrecipitation_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <coPrecipitationEffect.h>

using namespace opencover;
using namespace vrml;

class VRML97COVEREXPORT VrmlNodePrecipitation : public VrmlNodeChild
{

public:
    static void initFields(VrmlNodePrecipitation *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodePrecipitation(VrmlScene *scene = 0);
    VrmlNodePrecipitation(const VrmlNodePrecipitation &n);
    virtual ~VrmlNodePrecipitation();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNodePrecipitation *toPrecipitation() const;

    const VrmlField *getField(const char *fieldName) const override;

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    // Fields
    VrmlSFInt d_numPrecipitation;
    VrmlSFFloat d_fraction_changed;

    VrmlSFBool d_enabled;
    VrmlSFBool d_loop;
    osg::ref_ptr<coPrecipitationEffect> precipitationEffect;
};
#endif //_VRMLNODEPrecipitation_
