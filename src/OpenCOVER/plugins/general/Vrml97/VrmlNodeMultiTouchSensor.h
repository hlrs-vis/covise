/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeMultiTouchSensor.h

#ifndef _VRMLNODEMultiTouchSensor_
#define _VRMLNODEMultiTouchSensor_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec2f.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace opencover;
using namespace vrml;

class VRML97COVEREXPORT VrmlNodeMultiTouchSensor : public VrmlNodeChild
{

public:
    static void initFields(VrmlNodeMultiTouchSensor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeMultiTouchSensor(VrmlScene *scene = 0);
    VrmlNodeMultiTouchSensor(const VrmlNodeMultiTouchSensor &n);
    virtual ~VrmlNodeMultiTouchSensor();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNodeMultiTouchSensor *toMultiTouchSensor() const;

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }
    void remoteUpdate(bool visible, float *pos, float angle);
    const char *getMarkerName();
    bool wasVisible;
    bool remoteWasVisible;

    static coEventType MultiTouchEventType;

private:
    // Fields
    VrmlSFBool d_freeze;
    VrmlSFBool d_adjustHeight;
    VrmlSFBool d_adjustOrientation;
    VrmlSFBool d_trackObjects;
    VrmlSFBool d_enabled;
    VrmlSFBool d_currentCamera;
    VrmlSFBool d_rotationEnabled;
    VrmlSFFloat d_positionThreshold;
    VrmlSFFloat d_orientationThreshold;
    VrmlSFRotation d_orientation;
    VrmlSFVec2f d_size;
    VrmlSFVec3f d_minPosition;
    VrmlSFVec3f d_markerPosition;
    VrmlSFRotation d_markerRotation;
    VrmlSFVec3f d_invisiblePosition;
    VrmlSFString d_markerName;
    VrmlSFBool d_headingOnly;
    VrmlSFBool d_visible;

    VrmlSFVec3f d_translation;
    VrmlSFRotation d_rotation;
    VrmlSFVec3f d_scale;
    float oldPos[2];
    float oldAngle;
    double oldTime;
    int markerID;
    osg::Matrix markerTrans;
    osg::Matrix surfaceTrans;

    void sendMultiTouchEvent(const char *markerName, bool visible, float *pos, float angle);

    static void addMultiTouchNode(VrmlNodeMultiTouchSensor *node);
    static void removeMultiTouchNode(VrmlNodeMultiTouchSensor *node);

    static void handleMultiTouchEvent(int type, int len, const void *buf);

    void recalcMatrix();
};
#endif //_VRMLNODEMultiTouchSensor_
