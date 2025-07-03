/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeJAKA.h

#ifndef _VRMLNODEJAKA_
#define _VRMLNODEJAKA_

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

class JAKAConnection;

using namespace opencover;
using namespace vrml;
namespace opencover
{
    
//------------------------------------------------------------------------------------------------------------------------------

class Flap
{
public:
};

}

class VrmlNodeJAKA : public VrmlNodeChild
{

public:
    // Define the fields of JAKA nodes
    static void initFields(VrmlNodeJAKA* node, vrml::VrmlNodeType* t);
    static const char *typeName();

    VrmlNodeJAKA(VrmlScene *scene = 0);
    VrmlNodeJAKA(const VrmlNodeJAKA &n);
    virtual ~VrmlNodeJAKA();
    virtual void addToScene(VrmlScene *s, const char *);


    void setSpeed(bool v);


    virtual void render(Viewer *);




private:
    // Fields
    


    Flap MainFlap[3];
    Flap SecFlap[3];
    VrmlSFFloat d_speed;
    VrmlSFBool d_lademodus;
    VrmlSFVec3f d_position;
    VrmlSFRotation d_rotation;
    VrmlSFFloat d_main0Angle;
    VrmlSFFloat d_main1Angle;
    VrmlSFFloat d_main2Angle;
    VrmlSFFloat d_sec0Angle;
    VrmlSFFloat d_sec1Angle;
    VrmlSFFloat d_sec2Angle;
};
#endif //_VRMLNODEJAKA_
