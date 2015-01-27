/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TRANSFORM_BEHAVIOR_H
#define TRANSFORM_BEHAVIOR_H

#include "Behavior.h"

#include <PluginUtil/coPlane.h>
#include <osg/MatrixTransform>
#include <osg/Vec3>
#include <osg/Shape>
#include <osg/Drawable>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>

#include "cover/coVRLabel.h"

class TransformBehavior : public Behavior
{
public:
    TransformBehavior();
    virtual ~TransformBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

protected:
    virtual void sendMessageToGUI();

private:
    osg::Vec3 _getRotationCenterInWorld();
    bool _getPlaneIntersection(opencover::coPlane *plane, osg::Vec3 &point);

    bool _isTranslating;
    bool _isRotating;
    osg::Matrix _startTranslateMat;
    osg::Matrix _startRotateMat;

    osg::Vec3 _startPickPos;
    opencover::coPlane *_translatePlane;
    opencover::coPlane *_rotatePlane;

    float _translateSnapping;
    float _rotateSnapping;

    opencover::coVRLabel *_label;
};

#endif
