/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeOffice.h

#ifndef _VRMLNODEOffice_
#define _VRMLNODEOffice_

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

#include <iostream>

class OfficeConnection;

namespace opencover
{
class ARMarker;
}
using namespace opencover;
using namespace vrml;

class VrmlNodeOffice : public VrmlNodeChild
{

public:
    static std::list<VrmlNodeOffice *> allOffice;

    static void initFields(VrmlNodeOffice *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeOffice(VrmlScene *scene = 0);
    VrmlNodeOffice(const VrmlNodeOffice &n);
    virtual ~VrmlNodeOffice();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNodeOffice *toOffice() const;

    virtual std::ostream &printFields(std::ostream &os, int indent) override;

    virtual void render(Viewer *);

    static void update();

    std::string getApplicationType(){return d_applicationType.get();};
    void setMessage(const char *s);

    OfficeConnection *officeConnection;

private:
    // Fields
    
    VrmlSFString d_applicationType;
    VrmlSFString d_command;
    VrmlSFString d_events;
};
#endif //_VRMLNODEOffice_
