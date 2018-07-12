/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SENSOR_H
#define CO_SENSOR_H

/*! \file
 \brief helper classes for vrml sensors

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/DLinkList.h>
#include <util/coExport.h>
#include <osg/Vec3>
#include <osg/BoundingSphere>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/osg/OSGVruiNode.h>

namespace osg
{
class Node;
class MatrixTransform;
}

class PLUGIN_UTILEXPORT coSensor
{
protected:
    osg::Node *node;
    osg::MatrixTransform **path;
    int pathLength;
    int active; // status of the sensor, active = 1 as long as pointer intersects node
    float threshold;
    float sqrDistance;
    int buttonSensitive;
    int enabled;

public:
    enum
    {
        NONE = 0,
        PROXIMITY,
        TOUCH,
        ISECT,
        PICK,
        HAND
    };
    coSensor(osg::Node *n);
    virtual ~coSensor();

    // this method is called if intersection just started
    // and should be overloaded
    virtual void activate();

    // should be overloaded, is called if intersection finishes
    virtual void disactivate();

    // enable intersection testing, default is enabled
    virtual void enable();

    // disable intersection testing
    virtual void disable();

    virtual int getType();
    virtual void calcDistance(){};
    virtual float getDistance()
    {
        return (sqrt(sqrDistance));
    };
    virtual void setThreshold(float d)
    {
        threshold = d * d;
    };
    void addToPath(osg::Node *n);
    virtual void update();
    virtual void setButtonSensitive(int s);
    osg::Node *getNode()
    {
        return node;
    };
};

class PLUGIN_UTILEXPORT coPickSensor : public coSensor, public vrui::coAction
{
public:
    osg::Vec3 hitPoint; // last hitPoint in world coordinates

    bool hitActive;
    vrui::OSGVruiNode *vNode;

    virtual int hit(vrui::vruiHit *hit);
    virtual void miss();
    coPickSensor(osg::Node *n);
    virtual ~coPickSensor();
    virtual void update();
    virtual int getType();
};

class PLUGIN_UTILEXPORT coSensorList : public covise::DLinkList<coSensor *>
{
public:
    coSensorList();
    void update();
};
#endif
