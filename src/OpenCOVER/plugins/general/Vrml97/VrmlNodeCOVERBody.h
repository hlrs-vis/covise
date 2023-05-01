/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCOVERBody.h

#ifndef _VRMLNODECOVERBody_
#define _VRMLNODECOVERBody_

#include <util/coTypes.h>
#include <cover/input/input.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace opencover;
using namespace vrml;

class VRML97COVEREXPORT VrmlNodeCOVERBody : public VrmlNodeChild
{

public:
    // Define the fields of COVERBody nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeCOVERBody(VrmlScene *scene = 0);
    VrmlNodeCOVERBody(const VrmlNodeCOVERBody &n);
    virtual ~VrmlNodeCOVERBody();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeCOVERBody *toCOVERBody() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual void render(Viewer *);

    static void update();

private:
    // Fields
    VrmlSFVec3f d_position;
    VrmlSFRotation d_orientation;
    VrmlSFString d_name;
    VrmlSFBool d_vrmlCoordinates;
    TrackingBody *body;
};
#endif //_VRMLNODECOVERBody_
