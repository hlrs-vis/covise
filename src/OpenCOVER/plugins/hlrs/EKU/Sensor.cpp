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

mySensor::mySensor(osg::Node *node, std::string name, vrui::coTrackerButtonInteraction *_interactionA, CamDrawable *camDraw,std::vector<Truck*> *observationPoints)
    : observationPoints(observationPoints), coPickSensor(node)
{
    sensorName = name;
    isActive = false;
    _interA = _interactionA;
    camDr = camDraw;
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
    if(shapDr != nullptr)
        shapDr->setColor(osg::Vec4(1., 1., 0., 1.0f));
    else
    {
        size_t cnt=0;
        camDr->updateColor();
        for(const auto& x: camDr->cam->visMat)
        {
            if(x==1)
            {
                observationPoints->at(cnt)->updateColor();
            }
        cnt++;
        }
    }
}

//-----------------------------------------------------------
void mySensor::disactivate()
{
    cout << "---Disactivate--" << sensorName.c_str() << endl;
    isActive = false;
    vrui::coInteractionManager::the()->unregisterInteraction(_interA);
    if(shapDr != nullptr)
        shapDr->setColor(osg::Vec4(1., 0., 0., 1.0f));
    else
    {
        size_t cnt=0;
        camDr->resetColor();
        for(const auto& x: camDr->cam->visMat)
        {
            if(x==1)
            {
                observationPoints->at(cnt)->resetColor();
            }
        cnt++;
        }
    }

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
