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
typedef std::unique_ptr<SensorZone> upSensorZone;

// Data Members need mutex ? :https://stackoverflow.com/questions/27035446/do-we-need-mutex-for-accessing-the-data-field-in-singleton-object-in-c11-mul
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
        static DataManager instance;    //Static variable with block scope -> should be thread safe! 
        return instance;
    }
    static void Destroy();
    static const std::vector<upSafetyZone>& GetSafetyZones(){return GetInstance().m_SafetyZones;}
    static const std::vector<upSensorZone>& GetSensorZones(){return GetInstance().m_SensorZones;}
    static const std::vector<upSensor>& GetSensors(){return GetInstance().m_Sensors;}
    static const std::vector<osg::Vec3> GetWorldPosOfObervationPoints();
    static const osg::ref_ptr<osg::Group>& GetRootNode() {return GetInstance().m_Root;}
    
    static void highlitePoints(const VisibilityMatrix<float>& visMat);
    static void setOriginalPointColor();
    static void AddSafetyZone(upSafetyZone zone);
    static void AddSensorZone(upSensorZone zone);
    static void AddSensor(upSensor sensor);

    static void RemoveSensor(SensorPosition* sensor);
    static void RemoveZone(Zone* zone);

    static void UpdateAllSensors(std::vector<Orientation>& orientations);


    // Functions to handle incoming UDP messages
    static void RemoveUDPSensor(int pos);
    static void RemoveUDPZone(int pos);
    static void RemoveUDPObstacle(int pos);

    static void AddUDPSensor(upSensor sensor);
    static void AddUDPZone(upSafetyZone zone);
    static void AddUDPObstacle(osg::ref_ptr<osg::Node> node,const osg::Matrix& mat);

    static void UpdateUDPSensorPosition(int pos, const osg::Matrix& mat);
    static void UpdateUDPZone(int pos, const osg::Matrix& mat, int nbrOfSensors);
    static void UpdateUDPObstacle(int pos, const osg::Matrix& mat);


    
    static void preFrame();

private:
    DataManager();
    std::vector<upSensor> m_Sensors;            // virtual sensor positions
    std::vector<upSafetyZone> m_SafetyZones;//TODO: should use safety zone here as type ?           
    std::vector<upSensorZone> m_SensorZones;
    
    // live UDP positions
    std::vector<upSensor> m_UDPSensors;        
    std::vector<upSafetyZone> m_UDPSafetyZones;
    std::vector<osg::ref_ptr<osg::MatrixTransform>> m_UDPObstacles;


    osg::ref_ptr<osg::Group> m_Root;

};
