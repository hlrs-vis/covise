/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeMultiTouchSensor.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRTouchTable.h>
#include <cover/coIntersection.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <util/byteswap.h>

#include "VrmlNodeMultiTouchSensor.h"
#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>

static list<VrmlNodeMultiTouchSensor *> multiTouchSensors;

coEventType VrmlNodeMultiTouchSensor::MultiTouchEventType = { MULTI_TOUCH_EVENTS, &VrmlNodeMultiTouchSensor::handleMultiTouchEvent };

void VrmlNodeMultiTouchSensor::initFields(VrmlNodeMultiTouchSensor *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("trackObjects", node->d_trackObjects),
                     exposedField("freeze", node->d_freeze),
                     exposedField("adjustHeight", node->d_adjustHeight),
                     exposedField("adjustOrientation", node->d_adjustOrientation),
                     exposedField("enabled", node->d_enabled),
                     exposedField("currentCamera", node->d_currentCamera),
                     exposedField("headingOnly", node->d_headingOnly),
                     exposedField("size", node->d_size),
                     exposedField("minPosition", node->d_minPosition, [node](auto f){
                        node->recalcMatrix();
                     }),
                     exposedField("markerPosition", node->d_markerPosition, [node](auto f){
                        node->recalcMatrix();
                     }),
                     exposedField("markerRotation", node->d_markerRotation, [node](auto f){
                        node->recalcMatrix();
                     }),
                     exposedField("orientation", node->d_orientation, [node](auto f){
                        node->recalcMatrix();
                     }),
                     exposedField("orientationThreshold", node->d_orientationThreshold),
                     exposedField("positionThreshold", node->d_positionThreshold),
                     exposedField("invisiblePosition", node->d_invisiblePosition),
                     exposedField("markerName", node->d_markerName, [node](auto f){
                        node->markerID = coVRTouchTable::instance()->ttInterface->getMarker(node->d_markerName.get());
                     }));

    if(t)
    {
        t->addEventOut("isVisible", VrmlField::SFBOOL);
        t->addEventOut("translation_changed", VrmlField::SFVEC3F);
        t->addEventOut("rotation_changed", VrmlField::SFROTATION);
        t->addEventOut("scale_changed", VrmlField::SFVEC3F);
    }                     
}

const char *VrmlNodeMultiTouchSensor::name()
{
    return "MultiTouchSensor";
}

VrmlNodeMultiTouchSensor::VrmlNodeMultiTouchSensor(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_freeze(false)
    , d_adjustHeight(false)
    , d_adjustOrientation(false)
    , d_trackObjects(false)
    , d_enabled(true)
    , d_currentCamera(true)
    , d_positionThreshold(0.3)
    , d_orientationThreshold(15.0)
    , d_orientation(0, 1, 0, 0)
    , d_size(-1.0, -1.0)
    , d_minPosition(-1.0, -1.0, -1.0)
    , d_markerPosition(0, 0, 0)
    , d_markerRotation(0, 1, 0, 0)
    , d_invisiblePosition(100000, 100000, 100000)
    , d_markerName(NULL)
    , d_visible(false)
    , d_translation(0, 0, 0)
    , d_rotation(0, 1, 0, 0)
    , d_scale(0.0000001, 0.0000001, 0.0000001)
{
    markerID = -1;
    setModified();
    oldPos[0] = 0;
    oldPos[1] = 0;
    oldAngle = 0;
    oldTime = 0.0;
    recalcMatrix();
}

void VrmlNodeMultiTouchSensor::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        addMultiTouchNode(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeMultiTouchSensor::VrmlNodeMultiTouchSensor(const VrmlNodeMultiTouchSensor &n)
    : VrmlNodeChild(n)
    , d_freeze(n.d_freeze)
    , d_adjustHeight(n.d_adjustHeight)
    , d_adjustOrientation(n.d_adjustOrientation)
    , d_trackObjects(n.d_trackObjects)
    , d_enabled(n.d_enabled)
    , d_currentCamera(n.d_currentCamera)
    , d_positionThreshold(n.d_positionThreshold)
    , d_orientationThreshold(n.d_orientationThreshold)
    , d_orientation(n.d_orientation)
    , d_size(n.d_size)
    , d_minPosition(n.d_minPosition)
    , d_markerPosition(n.d_markerPosition)
    , d_markerRotation(n.d_markerRotation)
    , d_invisiblePosition(n.d_invisiblePosition)
    , d_markerName(NULL)
    , d_visible(n.d_visible)
    , d_translation(n.d_translation)
    , d_rotation(n.d_rotation)
    , d_scale(n.d_scale)
{
    if (d_markerName.get())
    {
        if (coVRTouchTable::instance()->ttInterface)
        {
            markerID = coVRTouchTable::instance()->ttInterface->getMarker(d_markerName.get());
        }
    }
    else
        markerID = -1;
    setModified();
    oldPos[0] = 0;
    oldPos[1] = 0;
    oldAngle = 0;
    oldTime = 0.0;
    recalcMatrix();
}

VrmlNodeMultiTouchSensor::~VrmlNodeMultiTouchSensor()
{
    removeMultiTouchNode(this);
}

VrmlNodeMultiTouchSensor *VrmlNodeMultiTouchSensor::toMultiTouchSensor() const
{
    return (VrmlNodeMultiTouchSensor *)this;
}

const char *VrmlNodeMultiTouchSensor::getMarkerName()
{
    return d_markerName.get();
}

void VrmlNodeMultiTouchSensor::remoteUpdate(bool visible, float *pos, float angle)
{
    double timeNow = System::the->time();
    bool wasVisible = d_visible.get();
    if (visible)
    {
        if (!wasVisible)
        {
            d_visible.set(true);
            eventOut(timeNow, "isVisible", d_visible);
            d_scale.set(1, 1, 1);
            eventOut(timeNow, "scale_changed", d_scale);
        }

        d_translation.set(pos[0], pos[1], pos[2]);
        eventOut(timeNow, "translation_changed", d_translation);

        osg::Vec3 normal(0, 0, 1);
        normal = osg::Matrix::transform3x3(surfaceTrans, normal);
        d_rotation.set(normal[0], normal[1], normal[2], angle);
        eventOut(timeNow, "rotation_changed", d_rotation);
    }
    else
    {
        if (wasVisible)
        {
            d_visible.set(false);
            eventOut(timeNow, "isVisible", d_visible);
            if (!d_freeze.get())
            {
                d_translation = d_invisiblePosition;
                eventOut(timeNow, "translation_changed", d_translation);

                d_scale.set(0.000001, 0.0000001, 0.0000001);
                eventOut(timeNow, "scale_changed", d_scale);
            }
        }
    }
    setModified();
}

void VrmlNodeMultiTouchSensor::render(Viewer *viewer)
{
    (void)viewer;

    if (markerID < 0 && coVRTouchTable::instance()->ttInterface != NULL && d_markerName.get())
    {
        markerID = coVRTouchTable::instance()->ttInterface->getMarker(d_markerName.get());
    }
    if (markerID >= 0)
    {
        double timeNow = System::the->time();
        bool wasVisible = d_visible.get();
        coVRTouchTableInterface *tti = coVRTouchTable::instance()->ttInterface;

        if (tti->isVisible(markerID))
        {

            if (!wasVisible)
            {
                d_visible.set(true);
                eventOut(timeNow, "isVisible", d_visible);
                d_scale.set(1, 1, 1);
                eventOut(timeNow, "scale_changed", d_scale);
            }
            osg::Matrix MarkerPos; // marker position in camera coordinate system
            osg::Vec2 pos2 = tti->getPosition(markerID);
            float angle = tti->getOrientation(markerID);
            osg::Vec3 pos;
            pos.set(pos2[0] * d_size.x(), pos2[1] * d_size.y(), 0);
            pos = surfaceTrans.preMult(pos);
            osg::Vec3 normal(0, 0, 1);
            normal = osg::Matrix::transform3x3(surfaceTrans, normal);

            if (fabs(pos[0] - oldPos[0]) > d_positionThreshold.get() || fabs(pos[1] - oldPos[1]) > d_positionThreshold.get()
                || fabs(angle - oldAngle) > d_orientationThreshold.get()
                || ((timeNow - oldTime) < 1.0))
            {
			 if(d_adjustHeight.get())
			 {

				 // move object to ground

				 // down segment
				 osg::Vec3 p0,q0;
				 p0.set(pos[0],  pos[1]+100.0,  pos[2]);
				 q0.set(pos[0],  pos[1]-100.0, pos[2]);


                 coIntersector* isect = coIntersection::instance()->newIntersector(p0, q0);
                 osgUtil::IntersectionVisitor visitor(isect);
				 visitor.setTraversalMask(Isect::Collision);

				 ViewerOsg::viewer->VRMLRoot->accept(visitor);

				 //std::cerr << "Hits ray num: " << num1 << ", down (" << ray->start()[0] << ", " << ray->start()[1] <<  ", " << ray->start()[2] << "), up (" << ray->end()[0] << ", " << ray->end()[1] <<  ", " << ray->end()[2] << ")" <<  std::endl;
				 if (isect->containsIntersections())
				 {
                     auto hitInformation1 = isect->getFirstIntersection();

					 pos[1] = hitInformation1.getLocalIntersectPoint()[1];

				 }
			 }
                d_translation.set(pos[0], pos[1], pos[2]);
                d_translation.subtract(&d_markerPosition);
                eventOut(timeNow, "translation_changed", d_translation);
                d_rotation.set(normal[0], normal[1], normal[2], angle);

                d_rotation.multiply(&d_markerRotation);
                eventOut(timeNow, "rotation_changed", d_rotation);
                if (fabs(pos[0] - oldPos[0]) > d_positionThreshold.get() || fabs(pos[1] - oldPos[1]) > d_positionThreshold.get()
                    || fabs(angle - oldAngle) > d_orientationThreshold.get())
                {
                    oldTime = timeNow;
                }
                oldPos[0] = pos[0];
                oldPos[1] = pos[1];
                oldAngle = angle;
                float t[3];
                t[0] = pos[0];
                t[1] = pos[1];
                t[2] = pos[2];
                sendMultiTouchEvent(d_markerName.get(), true, t, angle);
            }
            else
            {
                //oldPos[0] = t[0]; oldPos[1] = t[1]; oldPos[2] = t[2];
                //oldAngles[0] = coord.hpr[0]; oldAngles[1] = coord.hpr[1]; oldAngles[2] = coord.hpr[2];
            }
        }
        else
        {
            if (wasVisible)
            {
                d_visible.set(false);
                eventOut(timeNow, "isVisible", d_visible);
                if (!d_freeze.get())
                {
                    d_translation = d_invisiblePosition;
                    eventOut(timeNow, "translation_changed", d_translation);

                    d_scale.set(0.000001, 0.0000001, 0.0000001);
                    eventOut(timeNow, "scale_changed", d_scale);
                }
                oldPos[0] = 3212;
                oldPos[1] = 100000;
                oldAngle = 1000;
                oldTime = 0.0;
                float pos[2];
                pos[0] = pos[1] = /*pos[2]=*/0;
                float angle = 0.0;
                sendMultiTouchEvent(d_markerName.get(), false, pos, angle);
            }
        }
    }
    setModified();
}

void VrmlNodeMultiTouchSensor::eventIn(double timeStamp,
                                       const char *eventName,
                                       const VrmlField *fieldValue)
{
    if (strcmp(eventName, "markerName"))
    {
        if (fieldValue->toSFString()->get())
            markerID = coVRTouchTable::instance()->ttInterface->getMarker(fieldValue->toSFString()->get());
        else
            markerID = -1;
    }
    else if (strcmp(eventName, "markerRotation"))
    {
        recalcMatrix();
    }
    else if (strcmp(eventName, "markerPosition"))
    {
        recalcMatrix();
    }
    else if (strcmp(eventName, "orientation"))
    {
        recalcMatrix();
    }
    else if (strcmp(eventName, "minPosition"))
    {
        recalcMatrix();
    }
    VrmlNodeChild::eventIn(timeStamp, eventName, fieldValue);
}

void VrmlNodeMultiTouchSensor::recalcMatrix()
{
    float *ct = d_markerPosition.get();
    osg::Vec3 tr(ct[0], ct[1], ct[2]);
    markerTrans.makeTranslate(tr);
    osg::Matrix rot;
    ct = d_markerRotation.get();
    tr.set(ct[0], ct[1], ct[2]);
    rot.makeRotate(ct[3], tr);
    markerTrans.preMult(rot);

    ct = d_minPosition.get();
    tr.set(ct[0], ct[1], ct[2]);
    surfaceTrans.makeTranslate(tr);
    ct = d_orientation.get();
    tr.set(ct[0], ct[1], ct[2]);
    rot.makeRotate(ct[3], tr);
    surfaceTrans.preMult(rot);
}

void VrmlNodeMultiTouchSensor::sendMultiTouchEvent(const char *markerName, bool visible, float *pos, float angle)
{
    int buflen = sizeof(float); //ori
    buflen += 2 * sizeof(float); //pos
    buflen += 1 * sizeof(int); //visible
    buflen += 2 * sizeof(int); //header
    buflen += strlen(markerName) + 1;

    VrmlMessage *msg = System::the->newMessage(buflen);
    int idummy;
    msg->append(idummy = MULTI_TOUCH_EVENTS);
    msg->append(idummy = (int)visible);
    for (int i = 0; i < 3; i++)
        msg->append(pos[i]);
    msg->append(angle);
    msg->append(markerName, strlen(markerName) + 1);
    System::the->sendAndDeleteMessage(msg);
}

void VrmlNodeMultiTouchSensor::handleMultiTouchEvent(int type, int len, const void *buf)
{
    (void)len;
    if (type == MULTI_TOUCH_EVENTS)
    {
        bool visible;
        float pos[3];
        float angle;
        float *fbuf;
        int *ibuf = (int *)buf;
#ifdef BYTESWAP
        byteSwap(*ibuf);
#endif
        visible = (*ibuf) != 0;
        ibuf++;
        int i;
        fbuf = (float *)ibuf;
        for (i = 0; i < 3; i++)
        {
            pos[i] = *fbuf;
// cerr << "bs " << pos[i]<<endl;
#ifdef BYTESWAP
            byteSwap(pos[i]);
#endif
            //cerr << "real " << pos[i]<<endl;
            fbuf++;
        }
        angle = *fbuf;
#ifdef BYTESWAP
        byteSwap(angle);
#endif
        fbuf++;
        char *cbuf = (char *)fbuf;
        cerr << "name:" << cbuf << endl;
        //strcpy(buf,markerName);
        list<VrmlNodeMultiTouchSensor *>::iterator n;
        for (n = multiTouchSensors.begin(); n != multiTouchSensors.end(); ++n)
        {
            if ((*n)->getMarkerName() == std::string(cbuf))
            {
                (*n)->remoteUpdate(visible, pos, angle);
                return;
            }
        }
    }
}

void VrmlNodeMultiTouchSensor::addMultiTouchNode(VrmlNodeMultiTouchSensor *node)
{
    multiTouchSensors.push_front(node);
}

void VrmlNodeMultiTouchSensor::removeMultiTouchNode(VrmlNodeMultiTouchSensor *node)
{
    multiTouchSensors.remove(node);
}
