
#include <iostream>
#include <memory> 

#include <cover/coVRMSController.h>

#include "SensorPlacement.h"
#include "Helper.h"
#include "Profiling.h"
#include "UI.h"
#include "DataManager.h"
#include "GA.h"

using namespace opencover;

int calcNumberOfSensors()
{
    int numberOfSensorsInZones{0};
    for(const auto& zone : DataManager::GetSensorZones())
        numberOfSensorsInZones += zone->getNumberOfSensors();

    return DataManager::GetSensors().size() + numberOfSensorsInZones;
}

//
int convert(int sensorPos) // finish me !!!!!
{
    if(sensorPos < DataManager::GetSensors().size())
    {
        std::cout<< "is a single sensor"<< std::endl;
        return -1;
    }
    else if(sensorPos >= calcNumberOfSensors())
    {
        std::cout<<"not a sensor anymore!"<< std::endl;
        return -1;
}
    int posOfSafetyZone{0};
    int pos = sensorPos - DataManager::GetSensors().size();;
    int nbrOfSensors{0};

    for(const auto& zone : DataManager::GetSensorZones())
    {
        nbrOfSensors += zone->getNumberOfSensors();

        if(nbrOfSensors < pos )
            posOfSafetyZone++;
        else
            break;

        /*if(pos >= nbrOfSensors)
            posOfSafetyZone++;
        if(pos <= nbrOfSensors)
            break;

        */
    }
    std::cout<<"Sensor Pos: " <<posOfSafetyZone<<".."<<std::endl;
    return posOfSafetyZone;
}

void calcVisibility()
{
   for(const auto& sensor : DataManager::GetInstance().GetSensors())
      sensor->calcVisibility();

   for(const auto& sensorZone : DataManager::GetInstance().GetSensorZones() )
      sensorZone->createAllSensors();
}

void optimize(FitnessFunctionType fitnessFunction)
{
  calcVisibility();
  
  std::vector<Orientation> finalSensorOrientations;
  
  // Optimization is only done on master. Problem with random generator and multithreading on Slaves -> results are different on each slave!
  if(coVRMSController::instance()->isMaster())
  {
    auto ga(myHelpers::make_unique<GA>(fitnessFunction));
    finalSensorOrientations = ga->getFinalOrientations();
  }

  DataManager::UpdateAllSensors(finalSensorOrientations);
  // TODO: resize Vector of SensorPosition. Problem: size of Orientation Object not knwon
  
  //else if(!coVRMSController::instance()->isMaster())
  //  finalSensorOrientations.resize(calcNumberOfSensors());
  
  //coVRMSController::instance()->syncData(finalSensorOrientations.data(),sizeof(Orientation) * calcNumberOfSensors());
  //updateAllSensors(finalSensorOrientations);   
}

void updateAllSensors(std::vector<Orientation> orientations) // finish here ------------------------------------------------
{
  size_t count{0};
  if(orientations.size() != calcNumberOfSensors())
  {
    std::cout<< " "<<std::endl;
    return;
  }
  else
  {
    for(const auto& sensor : DataManager::GetSensors())
    {
      sensor->setCurrentOrientation(orientations.at(count));
      count++;
    }
  }
}

SensorPlacementPlugin::SensorPlacementPlugin()
{
  DataManager::GetInstance(); //Create Instance of Singleton
  m_UI = myHelpers::make_unique<UI>();
  
  #if SHOW_UDP_LIVE_OBJECTS
    m_udp = myHelpers::make_unique<UDP>();
  #else  
    std::cout << "Sensorplacement: UDP is turned Off" <<std::endl;
  #endif
  
}

bool SensorPlacementPlugin::init()
{
    std::cout<<"SensorPlacementPlugin loaded"<<std::endl;

    return true;
}

bool SensorPlacementPlugin::update()
{
  #if SHOW_UDP_LIVE_OBJECTS
    return true;              
  #else 
    return false;
  #endif
}

void SensorPlacementPlugin::preFrame()
{
  DataManager::preFrame();
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
