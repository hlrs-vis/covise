#include <iostream>
#include <memory> 
#include <future>
#include "SensorPlacement.h"
#include "Profiling.h"
#include "UI.h"
using namespace opencover;




bool SensorPlacementPlugin::init()
{
    std::cout<<"SensorPlacementPlugin loaded"<<std::endl;

    return true;
}

void SensorPlacementPlugin::preFrame()
{
    DataManager::preFrame();
}

SensorPlacementPlugin::SensorPlacementPlugin()
{
    DataManager::GetInstance(); //Create Instance of Singleton
    m_UI = myHelpers::make_unique<UI>();


    SP_PROFILE_BEGIN_SESSION("Init","SensorPlacement-Startup.json");

    SP_PROFILE_FUNCTION();

}

bool SensorPlacementPlugin::destroy()
{
    std::cout<<"Destroy Sensor Plugin"<<std::endl;
    DataManager::Destroy();
    return true;
}

SensorPlacementPlugin::~SensorPlacementPlugin()
{
    std::cout<<"Destructor Sensor Placement"<<std::endl;
    SP_PROFILE_END_SESSION();

}

COVERPLUGIN(SensorPlacementPlugin)