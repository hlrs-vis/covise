/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeARSensor.h

#ifndef _VRMLNODEARSensor_
#define _VRMLNODEARSensor_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

namespace opencover
{
class MarkerTrackingMarker;
}
using namespace vrml;
using namespace opencover;

namespace vrml
{

class VRML97COVEREXPORT VrmlNodeARSensor : public VrmlNodeChild
{

public:
    // Define the fields of ARSensor nodes
    static void initFields(VrmlNodeARSensor *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeARSensor(VrmlScene *scene = 0);
    VrmlNodeARSensor(const VrmlNodeARSensor &n);
    virtual ~VrmlNodeARSensor();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNodeARSensor *toARSensor() const;

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }
    void remoteUpdate(bool visible, float *pos, float *orientation);
    const char *getMarkerName();
    bool wasVisible;
    bool remoteWasVisible;

    static coEventType AREventType;

private:
    // Fields
    VrmlSFBool d_freeze;
    VrmlSFBool d_trackObjects;
    VrmlSFBool d_enabled;
    VrmlSFBool d_currentCamera;
    VrmlSFBool d_headingOnly;
    VrmlSFBool d_rotationEnabled;
    VrmlSFFloat d_positionThreshold;
    VrmlSFFloat d_orientationThreshold;
    VrmlSFVec3f d_maxPosition;
    VrmlSFVec3f d_minPosition;
    VrmlSFVec3f d_invisiblePosition;
    VrmlSFVec3f d_cameraPosition;
    VrmlSFVec3f d_cameraOrientation;
    VrmlSFString d_markerName;

    VrmlSFBool d_visible;

    VrmlSFVec3f d_translation;
    VrmlSFRotation d_rotation;
    VrmlSFVec3f d_scale;
    MarkerTrackingMarker *marker;
    float oldPos[3];
    float oldAngles[3];
    double oldTime;

    void sendAREvent(const char *markerName, bool visible, float *pos, float *ori);

    static void addARNode(VrmlNodeARSensor *node);
    static void removeARNode(VrmlNodeARSensor *node);

    static void handleAREvent(int type, int len, const void *buf);
};
}

#endif //_VRMLNODEARSensor_
