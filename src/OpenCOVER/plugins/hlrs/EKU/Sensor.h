#pragma once

#include<osg/ShapeDrawable>
#include<OpenVRUI/coTrackerButtonInteraction.h>
#include<cover/coVRPluginSupport.h>
#include<cover/coVRSelectionManager.h>
#include<PluginUtil/coSensor.h>

using namespace covise;
using namespace opencover;

class mySensor : public coPickSensor
{
public:
    mySensor(osg::Node *node, std::string name,vrui::coTrackerButtonInteraction *_interactionA, osg::ShapeDrawable *cSphDr);
    ~mySensor();

    void activate();
    void disactivate();
    std::string getSensorName();
    bool isSensorActive();

private:
    std::string sensorName;
    bool isActive;
    vrui::coTrackerButtonInteraction *_interA;
    osg::ShapeDrawable *shapDr;
};

