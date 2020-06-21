
#include <iostream>
#include <memory> 

#include "SensorPlacement.h"
#include "Helper.h"
#include "Profiling.h"
#include "UI.h"
#include "DataManager.h"

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
}

COVERPLUGIN(SensorPlacementPlugin)
