/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TransformBehavior.h"
#include "../Asset.h"
#include "../Events/StartMouseEvent.h"
#include "../Events/DoMouseEvent.h"
#include "../Events/SetTransformAxisEvent.h"
#include "../Events/PostInteractionEvent.h"
#include "../Events/MoveObjectEvent.h"
#include "../SceneUtils.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRMSController.h>
#include <cover/coIntersection.h>
#include <cover/VRViewer.h>

#include <appl/RenderInterface.h>
#include <cover/coVRMSController.h>
#include <covise/covise_appproc.h>
#include <grmsg/coGRObjMovedMsg.h>

#include <iostream>
#include <math.h>

#include "../Settings.h"

/*#ifdef _WINDOWS
double round(double r)
{
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}
#endif*/

TransformBehavior::TransformBehavior()
{
    _type = BehaviorTypes::TRANSFORM_BEHAVIOR;
    _isTranslating = false;
    _isRotating = false;
    _translateSnapping = -1.0f;
    _rotateSnapping = 45.0f;

    _translatePlane = new opencover::coPlane(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(0.0, 0.0, 0.0));
    _rotatePlane = new opencover::coPlane(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(0.0, 0.0, 0.0));

    osg::Vec4 fgcolor(1.0f, 1.0f, 1.0f, 1.0f);
    float fontSize = 0.02f * opencover::cover->getSceneSize();
    _label = new opencover::coVRLabel("", fontSize, 0.0f, fgcolor, opencover::VRViewer::instance()->getBackgroundColor());
    _label->keepDistanceFromCamera(true, 50.0f);
    _label->hide();
}

TransformBehavior::~TransformBehavior()
{
    delete _translatePlane;
    delete _rotatePlane;
    delete _label;
}

int TransformBehavior::attach(SceneObject *so)
{
    Behavior::attach(so);
    return 1;
}

int TransformBehavior::detach()
{
    Behavior::detach();
    return 1;
}

EventErrors::Type TransformBehavior::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::START_MOUSE_EVENT && _isEnabled)
    {
        if (e->wasHandled())
        {
            return EventErrors::UNHANDLED;
        }

        _isTranslating = (dynamic_cast<StartMouseEvent *>(e)->getMouseButton() == MouseEvent::TYPE_BUTTON_A);
        _isRotating = (dynamic_cast<StartMouseEvent *>(e)->getMouseButton() == MouseEvent::TYPE_BUTTON_C);

        _startTranslateMat = _sceneObject->getTranslate();
        _startRotateMat = _sceneObject->getRotate();
        _startPickPos = opencover::cover->getIntersectionHitPointWorld();

        // move planes to startPickPos (don't adjust the normal! It might have been set right before StartMouseEvent!)
        _translatePlane->update(_translatePlane->getNormal(), _startPickPos);
        _rotatePlane->update(_rotatePlane->getNormal(), _startPickPos);

        if (_isTranslating && Settings::instance()->isGridVisible())
        {
            _label->show();
        }

        return EventErrors::SUCCESS;
    }
    else if (e->getType() == EventTypes::STOP_MOUSE_EVENT)
    {
        if (_isTranslating || _isRotating)
        {
            _isTranslating = false;
            _isRotating = false;

            PostInteractionEvent pie;
            pie.setSender(this);
            _sceneObject->receiveEvent(&pie);

            _label->hide();

            return EventErrors::SUCCESS;
        }
    }
    // if a node is picked, check the mouse movement and translate asset-node
    else if (e->getType() == EventTypes::DO_MOUSE_EVENT && _isEnabled)
    {
        if (_isTranslating)
        {
            osg::Vec3 planeIntersectionPoint;
            if (SceneUtils::getPlaneIntersection(_translatePlane, opencover::cover->getPointerMat(), planeIntersectionPoint))
            {

                osg::Vec3 start = _startPickPos;
                start = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(start);
                start /= opencover::VRSceneGraph::instance()->scaleFactor();

                osg::Vec3 current = planeIntersectionPoint;
                current = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(current);
                current /= opencover::VRSceneGraph::instance()->scaleFactor();

                osg::Vec3 transVec = current - start;

                // add the translation thats already present
                transVec += _startTranslateMat.getTrans();

                if (_translateSnapping > 0.0f)
                {
                    transVec = osg::Vec3(round(transVec[0] / _translateSnapping) * _translateSnapping,
                                         round(transVec[1] / _translateSnapping) * _translateSnapping,
                                         round(transVec[2] / _translateSnapping) * _translateSnapping);
                }

                osg::Matrix trans;
                trans = osg::Matrix::translate(transVec);
                _sceneObject->setTranslate(trans);

                std::stringstream ss;
                ss << int(current[0]) << " cm / " << int(current[1]) << " cm / " << int(current[2]) << " cm";
                _label->setString(ss.str().c_str());
                _label->setPositionInScene(current);
            }
        }

        if (_isRotating)
        {
            osg::Vec3 planeIntersectionPoint;
            if (SceneUtils::getPlaneIntersection(_rotatePlane, opencover::cover->getPointerMat(), planeIntersectionPoint))
            {

                osg::Vec3 start = _startPickPos;
                start = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(start);
                start /= opencover::VRSceneGraph::instance()->scaleFactor();

                osg::Vec3 current = planeIntersectionPoint;
                current = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(current);
                current /= opencover::VRSceneGraph::instance()->scaleFactor();

                osg::Vec3 projectedCenter = _getRotationCenterInWorld();
                projectedCenter = _rotatePlane->getProjectedPoint(projectedCenter);
                projectedCenter = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(projectedCenter);
                projectedCenter /= opencover::VRSceneGraph::instance()->scaleFactor();

                osg::Vec3 vec1 = start - projectedCenter;
                osg::Vec3 vec2 = current - projectedCenter;

                osg::Matrix rot;
                rot.makeRotate(vec1, vec2);

                if (_rotateSnapping > 0.0f)
                {
                    osg::Vec3 axis;
                    double angle;
                    rot.getRotate().getRotate(angle, axis);
                    angle = osg::RadiansToDegrees(angle);
                    angle = round(angle / _rotateSnapping) * _rotateSnapping;
                    angle = osg::DegreesToRadians(angle);
                    rot.makeRotate(angle, axis);
                }

                // add the rotation thats already present
                osg::Matrix doRotMat;
                doRotMat.mult(_startRotateMat, rot);

                _sceneObject->setRotate(doRotMat);
            }

            return EventErrors::SUCCESS;
        }
    }
    else if (e->getType() == EventTypes::SET_TRANSFORM_AXIS_EVENT)
    {
        SetTransformAxisEvent *stae = dynamic_cast<SetTransformAxisEvent *>(e);

        if (stae->hasTranslateAxis() || (stae->hasResetTranslate()))
        {
            osg::Vec3 axis;
            if (stae->hasResetTranslate())
            {
                axis = osg::Vec3(0.0, 1.0, 0.0);
            }
            else
            {
                if (fabs(SceneUtils::getPlaneVisibility(_sceneObject->getTranslate().getTrans(), stae->getTranslateAxis())) < 0.1f)
                {
                    axis = osg::Vec3(0.0, 1.0, 0.0);
                }
                else
                {
                    axis = osg::Matrix::inverse(opencover::cover->getXformMat()) * stae->getTranslateAxis();
                }
            }

            // startPickPos has to be updated if the axis changes
            if (_isTranslating)
            {
                _startTranslateMat = _sceneObject->getTranslate();
                _startRotateMat = _sceneObject->getRotate();
                osg::Vec3 planeIntersectionPoint;
                if (SceneUtils::getPlaneIntersection(_translatePlane, opencover::cover->getPointerMat(), planeIntersectionPoint))
                {
                    _startPickPos = planeIntersectionPoint;
                }
            }

            _translatePlane->update(axis, _startPickPos);
        }

        if (stae->hasRotateAxis() || (stae->hasResetRotate()))
        {
            osg::Vec3 axis;
            if (stae->hasResetRotate())
            {
                axis = osg::Vec3(0.0, 1.0, 0.0);
            }
            else
            {
                axis = osg::Matrix::inverse(opencover::cover->getXformMat()) * stae->getRotateAxis();
            }
            _rotatePlane->update(axis, _startPickPos);
        }

        return EventErrors::SUCCESS;
    }
    else if (e->getType() == EventTypes::POST_INTERACTION_EVENT)
    {
        sendMessageToGUI();
    }
    else if (e->getType() == EventTypes::MOVE_OBJECT_EVENT)
    {
        MoveObjectEvent *moe = dynamic_cast<MoveObjectEvent *>(e);
        osg::Vec3 offset = moe->getDirection();
        if (_translateSnapping > 0.0f)
        {
            offset *= _translateSnapping;
        }
        else
        {
            offset *= 100.0f;
        }
        osg::Matrix m = _sceneObject->getTranslate();
        m *= osg::Matrix::translate(offset);
        _sceneObject->setTranslate(m, this);

        PostInteractionEvent pie;
        pie.setSender(this);
        _sceneObject->receiveEvent(&pie);
    }

    return EventErrors::UNHANDLED;
}

osg::Vec3 TransformBehavior::_getRotationCenterInWorld()
{
    osg::Vec3 transVec = _sceneObject->getTranslate().getTrans();
    transVec *= opencover::VRSceneGraph::instance()->scaleFactor();
    return transVec * opencover::cover->getXformMat();
}

bool TransformBehavior::buildFromXML(QDomElement *behaviorElement)
{
    QDomElement elem;
    elem = behaviorElement->firstChildElement("translation");
    if (!elem.isNull())
    {
        QDomElement subelem;
        subelem = elem.firstChildElement("snapping");
        if (!subelem.isNull())
        {
            _translateSnapping = subelem.attribute("value", "-1.0f").toFloat();
        }
    }
    elem = behaviorElement->firstChildElement("rotation");
    if (!elem.isNull())
    {
        QDomElement subelem;
        subelem = elem.firstChildElement("snapping");
        if (!subelem.isNull())
        {
            _rotateSnapping = subelem.attribute("value", "45.0f").toFloat();
        }
    }
    return true;
}

void TransformBehavior::sendMessageToGUI()
{
    //std::cerr << "TransformBehavior::sendMessageToGUI" << std::endl;
    // get translation
    osg::Vec3 trans = _sceneObject->getTranslate().getTrans();
    // get rotation
    osg::Quat rot = _sceneObject->getRotate().getRotate();
    // create Message
    if (opencover::coVRMSController::instance()->isMaster())
    {
        grmsg::coGRObjMovedMsg movedMsg(_sceneObject->getCoviseKey().c_str(), trans.x(), trans.y(), trans.z(), rot.x(), rot.y(), rot.z(), rot.w());
        opencover::cover->sendGrMessage(movedMsg);
    }
}
