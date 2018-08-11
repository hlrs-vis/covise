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
#include <osgDB/WriteFile>

#include "ViewerOsg.h"
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Action.h>

using namespace osg;

bool coSensiveSensor::modified = false;

coSensiveSensor::~coSensiveSensor()
{
    ViewerOsg::viewer->removeSensor(this);
    //delete tt;
}

coSensiveSensor::coSensiveSensor(Node *n, osgViewerObject *vObj, void *object, VrmlScene *s, MatrixTransform *VRoot)
    : coPickSensor(n)
    , ui::Owner(std::string("coSensiveSensor")+n->getName(), cover->ui)
{

    (void)VRoot;
    vrmlObject = object;
    viewerObj = vObj;
    pointerGrabbed = 0;
    d_scene = s;
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

    //tt = new PointerTooltip(n,"*",0.5);
    //tt = NULL;
#if 0
    button = new ui::Action(n->getName(), this);
#endif
    modified = true;
    hitWasActive = true;
}

void coSensiveSensor::disactivate()
{
    double timeNow = System::the->time();
    Vec3 position;
    double hitCoord[3];
    Matrix objToVRML;
    objToVRML.makeRotate(-M_PI_2, 1.0, 0.0, 0.0);
    Matrix relMat;
    relMat.mult(firstInvPointerMat, getPointerMat());
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
        pmat = (getPointerMat()) * objToVRML;
    else
        pmat = (getPointerMat() * cover->getInvBaseMat()) * objToVRML;
    double M[16];
    ViewerOsg::matToVrml(M, pmat);
    d_scene->sensitiveEvent(vrmlObject,
                            timeNow,
                            hitActive, false, // isOver, isActive
                            hitCoord, M);
    d_scene->update(timeNow);
    resetChildActive();
}

Matrix coSensiveSensor::getPointerMat() const
{
    if (interaction->isMouse())
        return cover->getMouseMat();

    return cover->getPointerMat();
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
    const Node *parent = nullptr;
    if (node->getNumParents())
        parent = node->getParent(0);
    while (parent != NULL && parent != VRMLRoot)
    {
        const MatrixTransform *mtParent = dynamic_cast<const MatrixTransform *>(parent);
        //if (pfIsOfType(parent,MatrixTransform::getClassType()))
        if (mtParent)
        {
            Matrix dcsMat = mtParent->getMatrix();
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
    if (active)
    {
        if (!childActive)
        {
            if (!interaction->isRegistered())
            {
                vrui::coInteractionManager::the()->registerInteraction(interaction);
            }

            if (parentSensor)
                parentSensor->setChildActive();
        }
        //Vec3 pposition = getPointerMat().getTrans();
        Matrix objToVRML;
        objToVRML.makeRotate(-M_PI_2, 1.0, 0.0, 0.0);
        double M[16];
        {
            Matrix pmat;
            if (viewerObj->rootNode == NULL)
                pmat = (getPointerMat()) * objToVRML;
            else
                pmat = (getPointerMat() * cover->getInvBaseMat()) * objToVRML;
            ViewerOsg::matToVrml(M, pmat);
        }

        if (interaction->wasStarted())
        {
            Matrix tr; //@@@ = viewerObj->parentTransform;
            tr.makeIdentity();
            Matrix vrmlBaseMat;
            if (viewerObj->rootNode == NULL)
                vrmlBaseMat = cover->getBaseMat();
            else
                vrmlBaseMat.makeIdentity();
            Matrix dcsMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
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
            firstInvPointerMat.invert(getPointerMat());
            double hitCoord[3];
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            if (!childActive)
            {
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        hitActive, true, // isOver, isActive
                                        hitCoord, M);
            }
        }

        if (interaction->isRunning() && !childActive)
        {
            Matrix relMat;
            relMat.mult(firstInvPointerMat, getPointerMat());
            Vec3 position = relMat.preMult(firstHitPoint);

            if (viewerObj->rootNode == NULL)
                position = cover->getInvBaseMat().preMult(position);
            position = objToVRML.preMult(position);
            double hitCoord[3];
            hitCoord[0] = position[0];
            hitCoord[1] = position[1];
            hitCoord[2] = position[2];

            d_scene->sensitiveEvent(vrmlObject,
                                    timeNow,
                                    hitActive, true, // isOver, isActive
                                    hitCoord, M);
        }

        if (interaction->wasStopped())
        {
            Matrix tr; //@@@ = viewerObj->parentTransform;
            tr.makeIdentity();
            Matrix vrmlBaseMat;
            if (viewerObj->rootNode == NULL)
                vrmlBaseMat = cover->getBaseMat();
            else
                vrmlBaseMat.makeIdentity();
            Matrix dcsMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
            vrmlBaseMat.preMult(dcsMat);
            tr.postMult(vrmlBaseMat);
            Vec3 hitPointVRML = LocalToVRML(hitPoint, ViewerOsg::viewer->VRMLRoot, node);
            firstHitPoint = tr.preMult(hitPointVRML);
            firstInvPointerMat.invert(getPointerMat());
            double hitCoord[3];
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            if (!childActive)
            {
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        hitActive, false, // isOver, isActive
                                        hitCoord, M);
            }
        }

        if (interaction->getState() == vrui::coInteraction::Idle)
        {
            //Matrix relMat;
            //relMat.mult(firstInvPointerMat, getPointerMat());
            //cerr << "Idle; hitPoint "<<hitPoint[0] << ' '<<
            //             hitPoint[1] << ' '<<hitPoint[2] << endl;
            //cerr << "Idle; firstHitPoint "<<firstHitPoint[0] << ' '<<
            //             firstHitPoint[1] << ' '<<firstHitPoint[2] << endl;
            //Vec3 position = relMat.preMult(firstHitPoint);
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
            /*
            if (viewerObj->rootNode == NULL)
                position = cover->getInvBaseMat().preMult(position);
            position = objToVRML.preMult(position);
            */
            /*
                         hitCoord[0]=position[0];
                         hitCoord[1]=position[1];
                         hitCoord[2]=position[2];
         */
            Vec3 hitPointVRML = LocalToVRML(hitPoint, ViewerOsg::viewer->VRMLRoot, node);
            double hitCoord[3];
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            if (!childActive)
            {
                d_scene->sensitiveEvent(vrmlObject,
                                        timeNow,
                                        hitActive, false, // isOver, isActive
                                        hitCoord, M);
            }
        }

        if (d_scene)
            d_scene->update(timeNow);
    }
    else if(hitActive)
    {
        if (!childActive)
        {
            Matrix objToVRML;
            objToVRML.makeRotate(-M_PI_2, 1.0, 0.0, 0.0);
            double M[16];
            {
                Matrix pmat;
                if (viewerObj->rootNode == NULL)
                    pmat = (getPointerMat()) * objToVRML;
                else
                    pmat = (getPointerMat() * cover->getInvBaseMat()) * objToVRML;
                ViewerOsg::matToVrml(M, pmat);
            }
            hitWasActive = true;
            Vec3 hitPointVRML = LocalToVRML(hitPoint, ViewerOsg::viewer->VRMLRoot, node);
            double hitCoord[3];
            hitCoord[0] = hitPointVRML[0];
            hitCoord[1] = hitPointVRML[1];
            hitCoord[2] = hitPointVRML[2];
            d_scene->sensitiveEvent(vrmlObject,
                timeNow,
                hitActive, false, // isOver, isActive
                hitCoord, M);
        }
    }
    else
    {
        if (hitWasActive)
        {
            double hitCoord[3];
            hitCoord[0] = 0.0;
            hitCoord[1] = 0.0;
            hitCoord[2] = 0.0;
            double M[16];
            d_scene->sensitiveEvent(vrmlObject,
                timeNow,
                false, false, // isOver, isActive
                hitCoord, M);
        }
        if (interaction->isRegistered() && (interaction->getState() != vrui::coInteraction::Active))
        {
            vrui::coInteractionManager::the()->unregisterInteraction(interaction);
            
        }
    }
}
