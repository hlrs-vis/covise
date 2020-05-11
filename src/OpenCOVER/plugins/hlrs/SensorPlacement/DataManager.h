#pragma once

#include<memory>
#include<iostream>
#include<vector>

#include<osg/Group>

#include "Zone.h"
void setStateSet(osg::StateSet *stateSet);


typedef std::unique_ptr<SensorPosition> upSensor;
typedef std::unique_ptr<Zone> upZone;

//Singleton Class
class DataManager
{
public:
    DataManager(const DataManager& other) = delete;
    DataManager operator=(const DataManager& other) = delete;
    ~DataManager(){    
        std::cout<<"Singleton is Destroyed!"<<std::endl;}
    static DataManager& GetInstance()
    {
        static DataManager instance;
        return instance;
    }
    static void Destroy();
    static const std::vector<upZone>& GetSafetyZones(){return GetInstance().m_SafetyZones;}
    static const std::vector<upSensor>& GetSensors(){return GetInstance().m_Sensors;}
    static void AddSafetyZone(upZone zone);
    static void AddSensor(upSensor sensor);
    
    template<typename T>
    static void Remove(T* object);

    static void preFrame();

private:
    DataManager();
    std::vector<upSensor> m_Sensors;
    std::vector<upZone> m_SafetyZones;
    osg::ref_ptr<osg::Group> m_Root;


};
