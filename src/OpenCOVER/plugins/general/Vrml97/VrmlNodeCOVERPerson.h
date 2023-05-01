/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCOVERPerson.h

#ifndef _VRMLNODECOVERPerson_
#define _VRMLNODECOVERPerson_

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

using namespace opencover;
using namespace vrml;

class VRML97COVEREXPORT VrmlNodeCOVERPerson : public VrmlNodeChild
{

public:
    // Define the fields of COVERPerson nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeCOVERPerson(VrmlScene *scene = 0);
    VrmlNodeCOVERPerson(const VrmlNodeCOVERPerson &n);
    virtual ~VrmlNodeCOVERPerson();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeCOVERPerson *toCOVERPerson() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual void render(Viewer *);

    static void update();

private:
    // Fields
    VrmlSFInt d_activePerson;
    VrmlSFFloat d_eyeDistance;
};
#endif //_VRMLNODECOVERPerson_
