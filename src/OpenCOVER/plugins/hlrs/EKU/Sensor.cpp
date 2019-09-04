#include <Sensor.h>
#include <cover/RenderObject.h>


mySensor::mySensor(osg::Node *node, std::string name, vrui::coTrackerButtonInteraction *_interactionA, osg::ShapeDrawable *cSphDr)
    : coPickSensor(node)
{
    sensorName = name;
    isActive = false;
    _interA = _interactionA;
    shapDr = cSphDr;
}

mySensor::~mySensor()
{
}

//-----------------------------------------------------------
void mySensor::activate()
{
    isActive = true;
    cout << "---Activate--" << sensorName.c_str() << endl;
    vrui::coInteractionManager::the()->registerInteraction(_interA);
    shapDr->setColor(osg::Vec4(1., 1., 0., 1.0f));
}

//-----------------------------------------------------------
void mySensor::disactivate()
{
    cout << "---Disactivate--" << sensorName.c_str() << endl;
    isActive = false;
    vrui::coInteractionManager::the()->unregisterInteraction(_interA);
    shapDr->setColor(osg::Vec4(1., 0., 0., 1.0f));
}

//-----------------------------------------------------------

std::string mySensor::getSensorName()
{
    return sensorName;
}

//-----------------------------------------------------------
bool mySensor::isSensorActive()
{
    if (isActive)
        return true;
    else
        return false;
}

//-----------------------------------------------------------
