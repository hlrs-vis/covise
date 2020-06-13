#pragma once

#include<memory>
#include<iostream>
#include<vector>

#include<osg/Group>

#include "Zone.h"
#include "Sensor.h"

void setStateSet(osg::StateSet *stateSet);
std::vector<VisibilityMatrix<float>> convertVisMatTo2D(const VisibilityMatrix<float>& visMat);


typedef std::unique_ptr<SensorPosition> upSensor;
typedef std::unique_ptr<Zone> upZone;
typedef std::unique_ptr<SafetyZone> upSafetyZone;
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
    static const std::vector<osg::Vec3> GetWorldPosOfObervationPoints();
    static const osg::ref_ptr<osg::Group>& GetRootNode() {return GetInstance().m_Root;}
    static void highlitePoints(const VisibilityMatrix<float>& visMat);
    static void setOriginalPointColor();
    static void AddZone(upZone zone);
    static void AddSensor(upSensor sensor);

    template<typename T>
    static void Remove(T* object);

    static void preFrame();

private:
    DataManager();
    std::vector<upSensor> m_Sensors;
    std::vector<upZone> m_SafetyZones;
    std::vector<upZone> m_SensorZones;
    osg::ref_ptr<osg::Group> m_Root;

};
