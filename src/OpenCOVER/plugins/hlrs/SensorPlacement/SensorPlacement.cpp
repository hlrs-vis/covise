
#include <iostream>
#include <memory> 
#include <future>

#include <cover/coVRMSController.h>

#include "SensorPlacement.h"
#include "Helper.h"
#include "Profiling.h"
#include "UI.h"
#include "DataManager.h"
#include "GA.h"

using namespace opencover;
std::unique_ptr<UI>SensorPlacementPlugin::s_UI = myHelpers::make_unique<UI>();

int calcNumberOfSensors()
{
    int numberOfSensorsInZones{0};
    for(const auto& zone : DataManager::GetSensorZones())
        numberOfSensorsInZones += zone->getTargetNumberOfSensors();

    return DataManager::GetSensors().size() + numberOfSensorsInZones;
}


int getSensorInSensorZone(int sensorPos) 
{

 if(sensorPos < DataManager::GetSensors().size())
  {
     // std::cout<< "is a single sensor"<< std::endl;
      return -1;
  }
  else if(sensorPos >= calcNumberOfSensors())
  {
     // std::cout<<"not a sensor anymore!"<< std::endl;
      return -1;
  }

  sensorPos = sensorPos - DataManager::GetSensors().size();
  std::vector<int> sensorsPerZone;
  for(const auto& zone : DataManager::GetSensorZones())
    sensorsPerZone.push_back(zone->getTargetNumberOfSensors());

  int nbrOfCameras{0};
  int first{0};
  int pos{0};
  for(const auto& c : sensorsPerZone)
  {
    nbrOfCameras += c;
    //std::cout<<"sensorPos"<<sensorPos << " first " <<first <<" nbrOfCameras "<< nbrOfCameras <<std::endl;
    if(sensorPos >= first && sensorPos < nbrOfCameras)
    {
     // std::cout<<"return Pos: " <<pos<<".."<<std::endl;
      return pos;
    }
    first+= c;
    pos++;
  }

  return -1;
}


void calcVisibility()
{
  {
  std::vector<std::future<void>> futures;
  for(const auto& sensor :  DataManager::GetInstance().GetSensors())  
    futures.push_back(std::async(std::launch::async, &SensorPosition::calcVisibility, sensor.get()));
  }
  // useful to use async here and also in SensorZone::createAllSensors ? 
  for(const auto& sensorZone : DataManager::GetInstance().GetSensorZones() )
  {
    //futures.push_back(std::async(std::launch::async, &SensorZone::createAllSensors, sensorZone.get()));
    sensorZone->createAllSensors();
  }
}

void optimize(FitnessFunctionType fitnessFunction)
{
  calcVisibility();
  
  std::vector<Orientation> finalSensorOrientations;
  
  // Optimization is only done on master. Problem with random generator and multithreading on Slaves -> results are different on each slave!
  // if(coVRMSController::instance()->isMaster())
  // {
    auto ga(myHelpers::make_unique<GA>(fitnessFunction));
    finalSensorOrientations = ga->getFinalOrientations();
    SensorPlacementPlugin::s_UI->updateOptimizationResults(ga->getTotalCoverage(), ga->getPrio1Coverage(),  ga->getPrio2Coverage(), ga->getFinalFitness(), ga->getOptimizationTime()  );
  // }
  // else if(!coVRMSController::instance()->isMaster())
    // finalSensorOrientations.resize(calcNumberOfSensors());
  
  //coVRMSController::instance()->syncData(finalSensorOrientations.data(), sizeof(Orientation) * calcNumberOfSensors()); // not sure if this is working with type Orientation

  DataManager::UpdateAllSensors(finalSensorOrientations);
  DataManager::visualizeCoverage();

}

//creates a vector that contains for each observation point the required number of sensors, so that the sensor is observed
std::vector<int> calcRequiredSensorsPerPoint()
{
  std::vector<int> requiredSensorsPerPoint;
  for(const auto& zone : DataManager::GetSafetyZones())
    requiredSensorsPerPoint.insert(requiredSensorsPerPoint.end(),zone.get()->getNumberOfPoints(), (int)zone->getPriority());

  if(!DataManager::GetUDPSafetyZones().empty())
  {
      for(const auto& zone : DataManager::GetUDPSafetyZones())
          requiredSensorsPerPoint.insert(requiredSensorsPerPoint.end(),zone.get()->getNumberOfPoints(), (int)zone->getPriority());
  }

  return requiredSensorsPerPoint;
}

SensorPlacementPlugin::SensorPlacementPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
  DataManager::GetInstance(); //Create Instance of Singleton
  //m_UI = myHelpers::make_unique<UI>();
  
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
