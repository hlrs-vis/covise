/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeARSensor.cpp
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
#include <cover/MarkerTracking.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>

#include "VrmlNodeARSensor.h"
#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>

static list<VrmlNodeARSensor *> arSensors;

coEventType VrmlNodeARSensor::AREventType = { AR_EVENTS, &VrmlNodeARSensor::handleAREvent };

// ARSensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeARSensor(scene);
}

// Define the built in VrmlNodeType:: "ARSensor" fields

VrmlNodeType *VrmlNodeARSensor::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ARSensor", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("trackObjects", VrmlField::SFBOOL);
    t->addExposedField("freeze", VrmlField::SFBOOL);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("currentCamera", VrmlField::SFBOOL);
    t->addExposedField("headingOnly", VrmlField::SFBOOL);
    t->addExposedField("maxPosition", VrmlField::SFVEC3F);
    t->addExposedField("minPosition", VrmlField::SFVEC3F);
    t->addExposedField("orientationThreshold", VrmlField::SFFLOAT);
    t->addExposedField("positionThreshold", VrmlField::SFFLOAT);
    t->addExposedField("invisiblePosition", VrmlField::SFVEC3F);
    t->addExposedField("cameraPosition", VrmlField::SFVEC3F);
    t->addExposedField("cameraOrientation", VrmlField::SFVEC3F);
    t->addExposedField("markerName", VrmlField::SFSTRING);
    t->addEventOut("isVisible", VrmlField::SFBOOL);
    t->addEventOut("translation_changed", VrmlField::SFVEC3F);
    t->addEventOut("rotation_changed", VrmlField::SFROTATION);
    t->addEventOut("scale_changed", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeARSensor::nodeType() const
{
    return defineType(0);
}

VrmlNodeARSensor::VrmlNodeARSensor(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_freeze(false)
    , d_trackObjects(false)
    , d_enabled(true)
    , d_currentCamera(true)
    , d_headingOnly(false)
    , d_positionThreshold(0.3)
    , d_orientationThreshold(15.0)
    , d_maxPosition(-1.0, -1.0, -1.0)
    , d_minPosition(-1.0, -1.0, -1.0)
    , d_invisiblePosition(100000, 100000, 100000)
    , d_cameraPosition(0, 0, 24)
    , d_cameraOrientation(90, 90, 0)
    , d_markerName(NULL)
    , d_visible(false)
    , d_translation(0, 0, 0)
    , d_rotation(0, 1, 0, 0)
    , d_scale(0.0000001, 0.0000001, 0.0000001)
{
    marker = NULL;
    setModified();
    oldPos[0] = 0;
    oldPos[1] = 0;
    oldPos[2] = 0;
    oldAngles[0] = 0;
    oldAngles[1] = 0;
    oldAngles[2] = 0;
    oldTime = 0.0;
}

void VrmlNodeARSensor::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        addARNode(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeARSensor::VrmlNodeARSensor(const VrmlNodeARSensor &n)
    : VrmlNodeChild(n.d_scene)
    , d_freeze(n.d_freeze)
    , d_trackObjects(n.d_trackObjects)
    , d_enabled(n.d_enabled)
    , d_currentCamera(n.d_currentCamera)
    , d_headingOnly(n.d_headingOnly)
    , d_positionThreshold(n.d_positionThreshold)
    , d_orientationThreshold(n.d_orientationThreshold)
    , d_maxPosition(n.d_maxPosition)
    , d_minPosition(n.d_minPosition)
    , d_invisiblePosition(n.d_invisiblePosition)
    , d_cameraPosition(n.d_cameraPosition)
    , d_cameraOrientation(n.d_cameraOrientation)
    , d_markerName(NULL)
    , d_visible(n.d_visible)
    , d_translation(n.d_translation)
    , d_rotation(n.d_rotation)
    , d_scale(n.d_scale)
{
    marker = NULL;
    setModified();
    oldPos[0] = 0;
    oldPos[1] = 0;
    oldPos[2] = 0;
    oldAngles[0] = 0;
    oldAngles[1] = 0;
    oldAngles[2] = 0;
    oldTime = 0.0;
}

VrmlNodeARSensor::~VrmlNodeARSensor()
{
    delete marker;
    removeARNode(this);
}

VrmlNode *VrmlNodeARSensor::cloneMe() const
{
    return new VrmlNodeARSensor(*this);
}

VrmlNodeARSensor *VrmlNodeARSensor::toARSensor() const
{
    return (VrmlNodeARSensor *)this;
}

const char *VrmlNodeARSensor::getMarkerName()
{
    return d_markerName.get();
}

void VrmlNodeARSensor::remoteUpdate(bool visible, float *pos, float *orientation)
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
        //cerr << "visible: " << visible << " pos " << pos[0] << " "<< pos[1] << " " << pos[2] << endl;
        //cerr << "orientation: " << orientation[0] << " " << orientation[1] << " "<< orientation[2] << " " << orientation[3] << endl;

        d_translation.set(pos[0], pos[1], pos[2]);
        eventOut(timeNow, "translation_changed", d_translation);

        d_rotation.set(orientation[0], orientation[1], orientation[2], orientation[3]);
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

void VrmlNodeARSensor::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();
    if (!marker)
    {
        if (d_markerName.get())
        {
            marker = MarkerTracking::instance()->getMarker(d_markerName.get());
        }
    }
    if ((marker) && (MarkerTracking::instance()->isRunning()))
    {
        bool wasVisible = d_visible.get();
        if (marker->isVisible())
        {
            float orientation[4];

            osg::Quat q;
            if (!wasVisible)
            {
                d_visible.set(true);
                eventOut(timeNow, "isVisible", d_visible);
                d_scale.set(1, 1, 1);
                eventOut(timeNow, "scale_changed", d_scale);
            }
            osg::Matrix MarkerPos; // marker position in camera coordinate system
            MarkerPos = marker->getMarkerTrans();
            osg::Matrix MarkerInLocalCoords;
            osg::Matrix MarkerInWorld, leftCameraTrans;
            osg::Matrix VRMLRootMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
            //VRMLRootMat.print(1,1,"VRMLRootMat: ",stderr);
            if (d_currentCamera.get())
            {
                leftCameraTrans = VRViewer::instance()->getViewerMat();
                if (coVRConfig::instance()->stereoState())
                {
                    leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                {
                    leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                {
                    leftCameraTrans.preMult(osg::Matrix::translate((VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                }

                //leftCameraTrans.print(1,1,"leftCameraTrans: ",stderr);
                MarkerInWorld = MarkerPos * leftCameraTrans;

                osg::Matrix vrmlBaseMat;
                vrmlBaseMat = cover->getBaseMat();
                //vrmlBaseMat.print(1,1,"cover->getBaseMat: ",stderr);
                vrmlBaseMat.preMult(VRMLRootMat);
                //vrmlBaseMat.print(1,1,"vrmlBaseMat: ",stderr);

                osg::Matrix inv, tr;
                double M[16];
                viewer->getTransform(M);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        tr(i, j) = M[i * 4 + j];
                //viewer->currentTransform.print(1,1,"currentTransform: ",stderr);
                tr.postMult(vrmlBaseMat);
                inv.invert(tr);
                MarkerInLocalCoords = MarkerInWorld * inv;
            }
            else
            {
                /* osg::Matrix  vrmlBaseMat;
             vrmlBaseMat = cover->getBaseMat();
             vrmlBaseMat.print(1,1,"getBaseMat: ",stderr);
             //vrmlBaseMat.print(1,1,"cover->getBaseMat: ",stderr);
             vrmlBaseMat.preMult(VRMLRootMat);
             vrmlBaseMat.print(1,1,"vrmlBaseMat: ",stderr);
             //VRMLRootMat.print(1,1,"VRMLRootMat: ",stderr);
              osg::Matrix rot;
              rot.makeEuler(d_cameraOrientation.x(),d_cameraOrientation.y(),d_cameraOrientation.z());
            leftCameraTrans.postTrans(rot,d_cameraPosition.x(),d_cameraPosition.y(),d_cameraPosition.z());

            leftCameraTrans.print(1,1,"leftCameraTrans: ",stderr);

            osg::Matrix camInWorld = leftCameraTrans * vrmlBaseMat;
            pfCoord c;
            camInWorld.getOrthoCoord(&c);
            camInWorld.makeCoord(&c);

            camInWorld.print(1,1,"camInWorld: ",stderr);

            MarkerInWorld = MarkerPos * camInWorld ;

            osg::Matrix inv,tr=viewer->currentTransform;
            //viewer->currentTransform.print(1,1,"viewer->currentTransform: ",stderr);
            tr.postMult(vrmlBaseMat);
            inv.invertAff(tr);
            MarkerInLocalCoords = MarkerInWorld * inv;
            */
                //MarkerPos.print(1,1,"MarkerPos: ",stderr);
                osg::Matrix rot;
                MAKE_EULER_MAT(rot, d_cameraOrientation.x(), d_cameraOrientation.y(), d_cameraOrientation.z());
                leftCameraTrans = rot;
                leftCameraTrans.postMult(osg::Matrixd::translate(d_cameraPosition.x(), d_cameraPosition.y(), d_cameraPosition.z()));
                MarkerInWorld = MarkerPos * leftCameraTrans;
                MarkerInLocalCoords = MarkerInWorld * VRMLRootMat;
                osg::Vec3 pos;
                pos = MarkerInLocalCoords.getTrans();
                pos *= 1 / 1000.0;
                MarkerInLocalCoords(3, 0) = pos[0];
                MarkerInLocalCoords(3, 1) = pos[1];
                MarkerInLocalCoords(3, 2) = pos[2];
                //MarkerInLocalCoords.print(1,1,"MarkerInLocalCoords: ",stderr);
            }

            float t[3];
            t[0] = MarkerInLocalCoords(3, 0);
            t[1] = MarkerInLocalCoords(3, 1);
            t[2] = MarkerInLocalCoords(3, 2);
            if (d_minPosition.x() == d_maxPosition.x())
                t[0] = d_minPosition.x();
            else if (d_minPosition.x() < d_maxPosition.x())
            {
                if (t[0] < d_minPosition.x())
                    t[0] = d_minPosition.x();
                else if (t[0] > d_maxPosition.x())
                    t[0] = d_maxPosition.x();
            }

            if (d_minPosition.y() == d_maxPosition.y())
                t[1] = d_minPosition.y();
            else if (d_minPosition.y() < d_maxPosition.y())
            {
                if (t[1] < d_minPosition.y())
                    t[1] = d_minPosition.y();
                else if (t[1] > d_maxPosition.y())
                    t[1] = d_maxPosition.y();
            }
            if (d_minPosition.z() == d_maxPosition.z())
                t[2] = d_minPosition.z();
            else if (d_minPosition.z() < d_maxPosition.z())
            {
                if (t[2] < d_minPosition.z())
                    t[2] = d_minPosition.z();
                else if (t[2] > d_maxPosition.z())
                    t[2] = d_maxPosition.z();
            }

            osg::Matrix tmp;
            tmp.makeScale(cover->getScale(), cover->getScale(), cover->getScale());
            MarkerInLocalCoords = tmp * MarkerInLocalCoords;
            //MarkerInLocalCoords.print(1,1,"MarkerInLocalCoords: ",stderr);
            coCoord coord = MarkerInLocalCoords;
            if (d_headingOnly.get())
            {
                coord.hpr[0] = 0;
                coord.hpr[1] = 0;
                //coord.hpr[2]=0;
            }
            coord.makeMat(MarkerInLocalCoords);
            //MarkerInLocalCoords.makeCoord(&coord);
            if (fabs(t[0] - oldPos[0]) > d_positionThreshold.get() || fabs(t[1] - oldPos[1]) > d_positionThreshold.get() || fabs(t[2] - oldPos[2]) > d_positionThreshold.get()
                || fabs(coord.hpr[0] - oldAngles[0]) > d_orientationThreshold.get() || fabs(coord.hpr[1] - oldAngles[1]) > d_orientationThreshold.get() || fabs(coord.hpr[2] - oldAngles[2]) > d_orientationThreshold.get()
                || ((timeNow - oldTime) < 1.0))
            {
                d_translation.set(t[0], t[1], t[2]);
                eventOut(timeNow, "translation_changed", d_translation);
                //MarkerInLocalCoords.getOrthoQuat(q);
                q.set(MarkerInLocalCoords);
                osg::Quat::value_type orient[4];
                q.getRotate(orient[3], orient[0], orient[1], orient[2]);
                for (int i = 0; i < 4; i++)
                    orientation[i] = orient[i];
                //cerr << orientation[0] << "  " << orientation[1] << "  "<< orientation[2] << "  "<< orientation[3] << endl;
                d_rotation.set(orientation[0], orientation[1], orientation[2], orientation[3]);

                eventOut(timeNow, "rotation_changed", d_rotation);
                sendAREvent(d_markerName.get(), true, t, orientation);
                if (fabs(t[0] - oldPos[0]) > d_positionThreshold.get() || fabs(t[1] - oldPos[1]) > d_positionThreshold.get() || fabs(t[2] - oldPos[2]) > d_positionThreshold.get()
                    || fabs(coord.hpr[0] - oldAngles[0]) > d_orientationThreshold.get() || fabs(coord.hpr[1] - oldAngles[1]) > d_orientationThreshold.get() || fabs(coord.hpr[2] - oldAngles[2]) > d_orientationThreshold.get())
                {
                    /*float tmax,omax;
               tmax = fabs(t[0]-oldPos[0]);
               if(tmax < fabs(t[1]-oldPos[1]))
                   tmax = fabs(t[1]-oldPos[1]);
               if(tmax < fabs(t[2]-oldPos[2]))
                   tmax = fabs(t[2]-oldPos[2]);
               omax = fabs(coord.hpr[0]-oldAngles[0]);
               if(omax < fabs(coord.hpr[1]-oldAngles[1]))
                   omax = fabs(coord.hpr[1]-oldAngles[1]);
               if(omax < fabs(coord.hpr[2]-oldAngles[2]))
                   omax = fabs(coord.hpr[2]-oldAngles[2]);
               cerr << "pos" << tmax << "ori" << omax << endl;*/
                    oldTime = timeNow;
                }
                oldPos[0] = t[0];
                oldPos[1] = t[1];
                oldPos[2] = t[2];
                oldAngles[0] = coord.hpr[0];
                oldAngles[1] = coord.hpr[1];
                oldAngles[2] = coord.hpr[2];
                if (d_trackObjects.get())
                {
                    cover->setXformMat(MarkerInWorld);
                }
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
                oldPos[2] = 123450;
                oldAngles[0] = 1;
                oldAngles[1] = 0;
                oldAngles[2] = 0;
                oldTime = 0.0;
                float pos[3];
                float ori[4];
                sendAREvent(d_markerName.get(), false, pos, ori);
            }
        }
    }
    setModified();
}

ostream &VrmlNodeARSensor::printFields(ostream &os, int indent)
{
    if (!d_trackObjects.get())
        PRINT_FIELD(trackObjects);
    if (!d_freeze.get())
        PRINT_FIELD(freeze);
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_currentCamera.get())
        PRINT_FIELD(currentCamera);
    if (!d_headingOnly.get())
        PRINT_FIELD(headingOnly);
    if (!FPEQUAL(d_maxPosition.x(), -1.0) || !FPEQUAL(d_maxPosition.y(), -1.0) || !FPEQUAL(d_maxPosition.z(), -1.0))
        PRINT_FIELD(maxPosition);
    if (!FPEQUAL(d_minPosition.x(), -1.0) || !FPEQUAL(d_minPosition.y(), -1.0) || !FPEQUAL(d_minPosition.z(), -1.0))
        PRINT_FIELD(minPosition);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeARSensor::setField(const char *fieldName,
                                const VrmlField &fieldValue)
{
    if
        TRY_FIELD(trackObjects, SFBool)
    else if
        TRY_FIELD(freeze, SFBool)
    else if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(currentCamera, SFBool)
    else if
        TRY_FIELD(headingOnly, SFBool)
    else if
        TRY_FIELD(maxPosition, SFVec3f)
    else if
        TRY_FIELD(minPosition, SFVec3f)
    else if
        TRY_FIELD(invisiblePosition, SFVec3f)
    else if
        TRY_FIELD(cameraPosition, SFVec3f)
    else if
        TRY_FIELD(cameraOrientation, SFVec3f)
    else if
        TRY_FIELD(orientationThreshold, SFFloat)
    else if
        TRY_FIELD(positionThreshold, SFFloat)
    else if
        TRY_FIELD(markerName, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeARSensor::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "trackObjects") == 0)
        return &d_trackObjects;
    if (strcmp(fieldName, "freeze") == 0)
        return &d_freeze;
    else if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "currentCamera") == 0)
        return &d_currentCamera;
    else if (strcmp(fieldName, "headingOnly") == 0)
        return &d_headingOnly;
    else if (strcmp(fieldName, "maxPosition") == 0)
        return &d_maxPosition;
    else if (strcmp(fieldName, "minPosition") == 0)
        return &d_minPosition;
    else if (strcmp(fieldName, "positionThreshold") == 0)
        return &d_positionThreshold;
    else if (strcmp(fieldName, "orientationThreshold") == 0)
        return &d_orientationThreshold;
    else if (strcmp(fieldName, "invisiblePosition") == 0)
        return &d_invisiblePosition;
    else if (strcmp(fieldName, "cameraPosition") == 0)
        return &d_cameraPosition;
    else if (strcmp(fieldName, "cameraOrientation") == 0)
        return &d_cameraOrientation;
    else if (strcmp(fieldName, "markerName") == 0)
        return &d_markerName;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeARSensor::sendAREvent(const char *markerName, bool visible, float *pos, float *ori)
{
    int buflen = 4 * sizeof(float); //ori
    buflen += 3 * sizeof(float); //pos
    buflen += 1 * sizeof(int); //visible
    buflen += 2 * sizeof(int); //header
    buflen += strlen(markerName) + 1;

    VrmlMessage *msg = System::the->newMessage(buflen);
    int idummy;
    msg->append(idummy = AR_EVENTS);
    msg->append(idummy = (int)visible);
    for (int i = 0; i < 3; i++)
        msg->append(pos[i]);
    for (int i = 0; i < 4; i++)
        msg->append(ori[i]);
    msg->append(markerName, strlen(markerName) + 1);
    System::the->sendAndDeleteMessage(msg);
}

void VrmlNodeARSensor::handleAREvent(int type, int len, const void *buf)
{
    (void)len;
    if (type == AR_EVENTS)
    {
        bool visible;
        float pos[3];
        float ori[4];
        float *fbuf;
        int *ibuf = (int *)buf;
#ifdef BYTESWAP
        byteSwap(*ibuf);
#endif
        visible = *ibuf;
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
        float or2[4];
        memcpy(or2, fbuf, 4 * sizeof(float));
        for (i = 0; i < 4; i++)
        {
#ifdef BYTESWAP
            byteSwap(or2[i]);
#endif
            //cerr << "real2" << or2[i]<<endl;
        }

        for (i = 0; i < 4; i++)
        {
            ori[i] = *fbuf;
//cerr << "bs " << ori[i]<<endl;
#ifdef BYTESWAP
            byteSwap(ori[i]);
#endif
            //cerr << "real " << ori[i]<<endl;
            fbuf++;
        }
        char *cbuf = (char *)fbuf;
        cerr << "name:" << cbuf << endl;
        //strcpy(buf,markerName);
        list<VrmlNodeARSensor *>::iterator n;
        for (n = arSensors.begin(); n != arSensors.end(); ++n)
        {
            if (strcmp((*n)->getMarkerName(), cbuf) == 0)
            {
                (*n)->remoteUpdate(visible, pos, ori);
                return;
            }
        }
    }
}

void VrmlNodeARSensor::addARNode(VrmlNodeARSensor *node)
{
    arSensors.push_front(node);
}

void VrmlNodeARSensor::removeARNode(VrmlNodeARSensor *node)
{
    arSensors.remove(node);
}
