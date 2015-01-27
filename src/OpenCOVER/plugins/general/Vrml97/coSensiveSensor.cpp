/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) Uwe Woessner
//
//  %W% %G%
//  ViewerOsg.cpp
//  Display of VRML models using Performer/COVER.
//

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include "coSensiveSensor.h"

static const int NUM_TEXUNITS = 4;

#include <vrml97/vrml/config.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNodeNavigationInfo.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/Player.h>

#include <osg/Group>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>
#include <osgDB/WriteFile>

#include "ViewerOsg.h"
#include <cover/coVRPluginSupport.h>

using namespace osg;

bool coSensiveSensor::modified = false;

coSensiveSensor::~coSensiveSensor()
{
    ViewerOsg::viewer->removeSensor(this);
    //delete tt;
    delete VrmlInteraction;
    delete button;
}

coSensiveSensor::coSensiveSensor(Node *n, osgViewerObject *vObj, void *object, VrmlScene *s, MatrixTransform *VRoot)
    : coPickSensor(n)
{

    (void)VRoot;
    vrmlObject = object;
    viewerObj = vObj;
    pointerGrabbed = 0;
    d_scene = s;
    wasReleased = 1;
    parentSensor = NULL;
    childActive = false;
    osgViewerObject *parentVObj = vObj;
    while ((parentVObj = parentVObj->parent))
    {
        if (parentVObj->sensor)
        {
            parentSensor = parentVObj->sensor;
            break;
        }
    }
    VrmlInteraction = new coTrackerButtonInteraction(coInteraction::ButtonA, "Vrml", coInteraction::Medium);

    //tt = new PointerTooltip(n,"*",0.5);
    //tt = NULL;
    distance = -1;
    button = new coButtonMenuItem(n->getName());
    modified = true;
}

void coSensiveSensor::disactivate()
{
    double timeNow = System::the->time();
    Vec3 position;
    double hitCoord[3];
    Matrix objToVRML;
    objToVRML.makeRotate(-M_PI_2, 1.0, 0.0, 0.0);
    Matrix relMat;
    relMat.mult(firstInvPointerMat, cover->getPointerMat());
    //position.fullXformPt(firstHitPoint,relMat);
    position = relMat.postMult(firstHitPoint);

    if (viewerObj->rootNode == NULL)
        //position.fullXformPt(position,cover->getInvBaseMat());
        position = cover->getInvBaseMat().postMult(position);
    //position.fullXformPt(position,objToVRML);
    position = objToVRML.postMult(position);
    hitCoord[0] = position[0];
    hitCoord[1] = position[1];
    hitCoord[2] = position[2];
    Matrix pmat;
    if (viewerObj->rootNode == NULL)
        pmat = (cover->getPointerMat()) * objToVRML;
    else
        pmat = (cover->getPointerMat() * cover->getInvBaseMat()) * objToVRML;
    double M[16];
    ViewerOsg::matToVrml(M, pmat);
    d_scene->sensitiveEvent(vrmlObject,
                            timeNow,
                            false, false, // isOver, isActive
                            hitCoord, M);
    d_scene->update(timeNow);
    wasReleased = 1;
    resetChildActive();
}

void coSensiveSensor::setChildActive()
{
    if (parentSensor)
        parentSensor->setChildActive();
    childActive = true;
}

void coSensiveSensor::resetChildActive()
{
    if (parentSensor)
        parentSensor->resetChildActive();
    childActive = false;
}

static Vec3
LocalToVRML(const Vec3 &hitPoint, const MatrixTransform *VRMLRoot, const Node *node)
{
    Matrix tr;
    tr.makeIdentity();
    //cerr << "LocalToVRML: hitPoint: "<<hitPoint[0]<<' '<<hitPoint[1]<<' '<<hitPoint[2]<<endl;
    const Node *parent = node->getParent(0);
    while (parent != NULL && parent != VRMLRoot)
    {
        Matrix dcsMat;
        const MatrixTransform *mtParent = dynamic_cast<const MatrixTransform *>(parent);
        //if (pfIsOfType(parent,MatrixTransform::getClassType()))
        if (mtParent)
        {
            dcsMat = mtParent->getMatrix();
            tr.postMult(dcsMat);
        }
#if 0
      else if (pfIsOfType(parent,pfSCS::getClassType()))
      {
         ((pfSCS *)parent)->getMat(dcsMat);
         tr.postMult(dcsMat);
      }
#endif
        if (parent->getNumParents())
            parent = parent->getParent(0);
        else
            parent = NULL;
    }
    Vec3 hitPointVRML = hitPoint;
    //hitPointVRML.fullXformPt(hitPoint, tr);
    hitPointVRML = tr.postMult(hitPoint);
    //cerr << "LocalToVRML: hitPointVRML: "<<hitPointVRML[0]<<' '<<hitPointVRML[1]<<' '<<hitPointVRML[2]<<endl;
    return hitPointVRML;
}

void coSensiveSensor::update()
{
    double timeNow = System::the->time();
    coPickSensor::update();
    double hitCoord[3];
    if (active)
    {
        if (!childActive)
        {
            if (!VrmlInteraction->isRegistered())
            {
                coInteractionManager::the()->registerInteraction(VrmlInteraction);
            }

            if (parentSensor)
                parentSensor->setChildActive();
        }
        Vec3 position;
        Vec3 pposition;
        pposition = cover->getPointerMat().getTrans();
        Matrix objToVRML;
        objToVRML.makeRotate(-M_PI_2, 1.0, 0.0, 0.0);
        if (VrmlInteraction->isRunning())
        {
            Matrix relMat;
            relMat.mult(firstInvPointerMat, cover->getPointerMat());
            position = relMat.preMult(firstHitPoint);

            distance = (pposition - position).length();
            //cerr << "Dist: " << distance << endl;
            if (viewerObj->rootNode == NULL)
                position = cover->getInvBaseMat().preMult(position);
            position = objToVRML.preMult(position);
            hitCoord[0] = position[0];
            hitCoord[1] = position[1];
            hitCoord[2] = position[2];
            if (!childActive)
            {
                Matrix pmat;
                if (viewerObj->rootNode == NULL)
                    pmat = (cover->getPointerMat()) * objToVRML;
                else
                    pmat = (cover->getPointerMat() * cover->getInvBaseMat()) * objToVRML;
                double M[16];
                ViewerOsg::matToVrml(M, pmat);
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        true, true, // isOver, isActive
                                        hitCoord, M);
            }
        }
        if (VrmlInteraction->wasStarted())
        {
            Matrix tr; //@@@ = viewerObj->parentTransform;
            tr.makeIdentity();
            Matrix vrmlBaseMat;
            if (viewerObj->rootNode == NULL)
                vrmlBaseMat = cover->getBaseMat();
            else
                vrmlBaseMat.makeIdentity();
            Matrix dcsMat;
            dcsMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
            vrmlBaseMat.preMult(dcsMat);
            tr.postMult(vrmlBaseMat);
            // firstHitPoint should be in world coordinates
            // and tr seems to transform vrml into world coordinates.
            // But hitPoint is given in local coordinates =>
            // we have to transform it to vrml coordinates
            Vec3 hitPointVRML = LocalToVRML(hitPoint, ViewerOsg::viewer->VRMLRoot, node);
            firstHitPoint = tr.preMult(hitPointVRML);
            //                cerr << "firstHitPoint wasStarted: " << firstHitPoint[0] <<":"<< firstHitPoint[1] <<":" << firstHitPoint[2] << endl;
            //cerr << "hitPoint " << hitPoint[0] <<":"<< hitPoint[1] <<":" << hitPoint[2] << endl;
            firstInvPointerMat.invert(cover->getPointerMat());
            position = firstHitPoint;
            distance = (pposition - position).length();
            //cerr << "HitDist: " << distance << endl;
            if (viewerObj->rootNode == NULL)
                position = cover->getInvBaseMat().preMult(position);
            position = objToVRML.preMult(position);
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            if (!childActive)
            {
                Matrix pmat;
                if (viewerObj->rootNode == NULL)
                    pmat = (cover->getPointerMat()) * objToVRML;
                else
                    pmat = (cover->getPointerMat() * cover->getInvBaseMat()) * objToVRML;
                double M[16];
                ViewerOsg::matToVrml(M, pmat);
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        true, true, // isOver, isActive
                                        hitCoord, M);
            }
            wasReleased = 0;
        }

        if (VrmlInteraction->wasStopped())
        {
            Matrix tr; //@@@ = viewerObj->parentTransform;
            tr.makeIdentity();
            Matrix vrmlBaseMat;
            if (viewerObj->rootNode == NULL)
                vrmlBaseMat = cover->getBaseMat();
            else
                vrmlBaseMat.makeIdentity();
            Matrix dcsMat;
            dcsMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
            vrmlBaseMat.preMult(dcsMat);
            tr.postMult(vrmlBaseMat);
            Vec3 hitPointVRML = LocalToVRML(hitPoint, ViewerOsg::viewer->VRMLRoot, node);
            firstHitPoint = tr.preMult(hitPointVRML);
            firstInvPointerMat.invert(cover->getPointerMat());
            position = firstHitPoint;
            distance = (pposition - position).length();
            if (viewerObj->rootNode == NULL)
                position = cover->getInvBaseMat().preMult(position);
            position = objToVRML.preMult(position);
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            if (!childActive)
            {
                Matrix pmat;
                if (viewerObj->rootNode == NULL)
                    pmat = (cover->getPointerMat()) * objToVRML;
                else
                    pmat = (cover->getPointerMat() * cover->getInvBaseMat()) * objToVRML;
                double M[16];
                ViewerOsg::matToVrml(M, pmat);
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        true, false, // isOver, isActive
                                        hitCoord, M);
            }
        }

        if (VrmlInteraction->getState() == coInteraction::Idle)
        {
            Matrix relMat;
            relMat.mult(firstInvPointerMat, cover->getPointerMat());
            //cerr << "Idle; hitPoint "<<hitPoint[0] << ' '<<
            //             hitPoint[1] << ' '<<hitPoint[2] << endl;
            //cerr << "Idle; firstHitPoint "<<firstHitPoint[0] << ' '<<
            //             firstHitPoint[1] << ' '<<firstHitPoint[2] << endl;
            position = relMat.preMult(firstHitPoint);
            //cerr << "Idle; position "<<position[0] << ' '<<
            //             position[1] << ' '<<position[2] << endl;

            // projektion der Transformation auf die XZ Ebene (evtl umstellen au Viewer richtung)
            /* Vec3 plane(0.0,firstHitPoint[1],0.0);
          Vec3 v=position-pposition;
          float a,b;
          a = pposition.dot(plane);
          b = v.dot(plane);
          if(b!=0)
          {
              position = pposition + v * (a/b);
          }*/
            Vec3 hitPointVRML = LocalToVRML(hitPoint, ViewerOsg::viewer->VRMLRoot, node);

            distance = (pposition - position).length();
            //cerr << "Dist: " << distance << endl;
            if (viewerObj->rootNode == NULL)
                position = cover->getInvBaseMat().preMult(position);
            position = objToVRML.preMult(position);
            /*
                         hitCoord[0]=position[0];
                         hitCoord[1]=position[1];
                         hitCoord[2]=position[2];
         */
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            if (!childActive)
            {
                Matrix pmat;
                if (viewerObj->rootNode == NULL)
                    pmat = (cover->getPointerMat()) * objToVRML;
                else
                    pmat = (cover->getPointerMat() * cover->getInvBaseMat()) * objToVRML;
                //cerr << "coSensiveSensor::idle "<<hitCoord[0]<<' '<<
                //                                         hitCoord[1]<<' '<<hitCoord[2]<<' '<<endl;
                double M[16];
                ViewerOsg::matToVrml(M, pmat);
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        true, false, // isOver, isActive
                                        hitCoord, M);
            }
        }

        if (d_scene)
            d_scene->update(timeNow);
    }
    else
    {
        if (VrmlInteraction->isRegistered() && (VrmlInteraction->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(VrmlInteraction);
        }
    }
}
