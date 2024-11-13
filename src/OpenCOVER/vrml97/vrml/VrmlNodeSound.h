/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSound.h
//    contributed by Kumaran Santhanam

#ifndef _VRMLNODESOUND_
#define _VRMLNODESOUND_

#include "VrmlNode.h"

#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFNode.h"
#include "VrmlSFVec3f.h"

#include "VrmlNodeChild.h"

#include "Player.h"

namespace vrml
{

class VrmlScene;

class VRMLEXPORT VrmlNodeSound : public VrmlNodeChild
{

public:
    // Define the fields of Sound nodes
    static void initFields(VrmlNodeSound *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeSound(VrmlScene *scene = 0);
    VrmlNodeSound(VrmlNodeSound *sound);
    virtual ~VrmlNodeSound();

    virtual void cloneChildren(VrmlNamespace *);

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual void render(Viewer *);

private:
    // Fields
    VrmlSFVec3f d_direction;
    VrmlSFFloat d_intensity;
    VrmlSFVec3f d_location;
    VrmlSFFloat d_maxBack;
    VrmlSFFloat d_maxFront;
    VrmlSFFloat d_minBack;
    VrmlSFFloat d_minFront;
    VrmlSFFloat d_priority;
    VrmlSFNode d_source;
    VrmlSFBool d_spatialize;
    VrmlSFBool d_doppler;

    // data for rendering
    VrmlSFVec3f lastLocation;
    double lastTime;
    Player::Source *source;
};
}
#endif //_VRMLNODESOUND_
