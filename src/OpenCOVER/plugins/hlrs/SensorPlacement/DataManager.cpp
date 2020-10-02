#include<osg/Material>
#include<osg/LightModel>
#include<osg/StateSet>
#include <cover/coVRPluginSupport.h>

#include <algorithm>

#include "DataManager.h"
#include "Zone.h"
#include "Sensor.h"

void setStateSet(osg::StateSet *stateSet)
{
    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE); 
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
}

std::vector<VisibilityMatrix<float>> convertVisMatTo2D(const VisibilityMatrix<float>& visMat)
{
    std::vector<int> sizes;
    for(const auto& zone : DataManager::GetSafetyZones())
    {
        sizes.push_back(zone->getNumberOfPoints());
    } 
    
    std::vector<VisibilityMatrix<float>> result;
    result.reserve(sizes.size());
    
    size_t startPos{0};
    size_t endPos;
    for(const auto& size : sizes)
    {
        endPos = startPos + size;
        VisibilityMatrix<float> temp = {visMat.begin() + startPos,visMat.begin() + endPos};
        result.push_back(std::move(temp));
        startPos += size;
    }
    
    return result;
}

DataManager::DataManager()
{
    m_Root = new osg::Group();
    m_Root->setName("SensorPlacement");
    m_Root->setNodeMask(m_Root->getNodeMask() & ~ 4096);
    osg::StateSet *mystateset = m_Root->getOrCreateStateSet();
    setStateSet(mystateset);
    cover->getObjectsRoot()->addChild(m_Root.get());
    std::cout<<"Singleton DataManager created!"<<std::endl;
};

void DataManager::preFrame()
{
    for(const auto& zone : GetInstance().m_SafetyZones)
    {
        bool status = zone->preFrame();
        if(!status)
        {
            GetInstance().RemoveZone(zone.get());
            return;
        }
        
    }

    for(const auto& zone : GetInstance().m_SensorZones)
    {
        bool status = zone->preFrame();
        if(!status)
        {
            GetInstance().RemoveZone(zone.get());
            return;
        } 
      
    }

    for(const auto& sensor : GetInstance().m_Sensors)
    {
        bool status = sensor->preFrame();
        if(!status)
        {
            GetInstance().RemoveSensor(sensor.get());
            return;
        }
    }

    for(const auto& udpSensor : GetInstance().m_UDPSensors)
    {
        bool status = udpSensor->preFrame();
    }
};
void DataManager::Destroy()
{
    GetInstance().m_Root->getParent(0)->removeChild(GetInstance().m_Root.get());

};

const std::vector<osg::Vec3> DataManager::GetWorldPosOfObervationPoints()
{
    std::vector<osg::Vec3> allPoints;
    size_t reserve_size {0};
    
    for(const auto& i : GetInstance().m_SafetyZones)
        reserve_size += i->getNumberOfPoints();

    allPoints.reserve(reserve_size);

    for(const auto& points :  GetInstance().m_SafetyZones)
    {
        auto vecWorldPositions = points->getWorldPositionOfPoints();
        allPoints.insert(allPoints.end(),vecWorldPositions.begin(),vecWorldPositions.end());
    }

    return allPoints;
}

void DataManager::AddSafetyZone(upSafetyZone zone)
{   
    GetInstance().m_Root->addChild(zone.get()->getZone().get());
    GetInstance().m_SafetyZones.push_back(std::move(zone));  
}

void DataManager::AddSensorZone(upSensorZone zone)
{
    GetInstance().m_Root->addChild(zone.get()->getZone().get());
    GetInstance().m_SensorZones.push_back(std::move(zone));  
}

void DataManager::AddSensor(upSensor sensor)
{
    GetInstance().m_Root->addChild(sensor.get()->getSensor().get());
    GetInstance().m_Sensors.push_back(std::move(sensor));     
}


void DataManager::AddUDPSensor(upSensor sensor)
{
    GetInstance().m_Root->addChild(sensor.get()->getSensor().get());
    GetInstance().m_UDPSensors.push_back(std::move(sensor));     
}

void DataManager::AddUDPZone(upSafetyZone zone)
{
    GetInstance().m_Root->addChild(zone.get()->getZone().get());
    GetInstance().m_UDPSafetyZones.push_back(std::move(zone));  
}

void DataManager::AddUDPObstacle(osg::ref_ptr<osg::Node> node, const osg::Matrix& mat)
{
    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform(mat);
    mt->addChild(node);

    GetInstance().m_UDPObstacles.push_back(mt);
    GetInstance().m_Root->addChild(mt);
}

void DataManager::RemoveSensor(SensorPosition* sensor)
{  
    GetInstance().m_Root->removeChild(sensor->getSensor());

    GetInstance().m_Sensors.erase(std::remove_if(GetInstance().m_Sensors.begin(),GetInstance().m_Sensors.end(),[sensor](std::unique_ptr<SensorPosition>const& it){return sensor == it.get();}));  
}

void DataManager::RemoveUDPSensor(int pos)
{
    GetInstance().m_Root->removeChild(GetInstance().m_UDPSensors.at(pos)->getSensor());
    GetInstance().m_UDPSensors.erase(GetInstance().m_UDPSensors.begin() + pos);
}

void DataManager::RemoveZone(Zone* zone)
{
    GetInstance().m_Root->removeChild(zone->getZone());

    if(dynamic_cast<SensorZone*>(zone))
         GetInstance().m_SensorZones.erase(std::remove_if(GetInstance().m_SensorZones.begin(),GetInstance().m_SensorZones.end(),[zone](std::unique_ptr<SensorZone>const& it){return zone == it.get();}));
    else if(dynamic_cast<SafetyZone*>(zone))
        GetInstance().m_SafetyZones.erase(std::remove_if(GetInstance().m_SafetyZones.begin(),GetInstance().m_SafetyZones.end(),[zone](std::unique_ptr<SafetyZone>const& it){return zone == it.get();}));
}

void DataManager::RemoveUDPObstacle(int pos)
{
    GetInstance().m_Root->removeChild(GetInstance().m_UDPObstacles.at(pos));
    GetInstance().m_UDPObstacles.erase(GetInstance().m_UDPObstacles.begin() + pos);

}

void DataManager::RemoveUDPZone(int pos)
{
    GetInstance().m_Root->removeChild(GetInstance().m_UDPSafetyZones.at(pos)->getZone());
    GetInstance().m_UDPSafetyZones.erase(GetInstance().m_UDPSafetyZones.begin() + pos);
}

void DataManager::highlitePoints(const VisibilityMatrix<float>& visMat)
{
    auto visMat2D = convertVisMatTo2D(visMat);

    size_t count{0};
    for(const auto& zone : GetSafetyZones())
    {
        zone->highlitePoints(visMat2D.at(count));
        count ++;
    }

}

void DataManager::UpdateAllSensors(std::vector<Orientation>& orientations)
{
    auto size =  GetInstance().m_Sensors.size();
    size_t incrementor{0}; // incrementor is also used in the loop later!
    for(incrementor; incrementor< size; incrementor++)
        GetInstance().m_Sensors.at(incrementor)->setCurrentOrientation(orientations.at(incrementor));

    //maybe increment incrementor here ?

    std::cout<<"incrementor befor second loop: " << incrementor << std::endl;
    for(const auto& zone : GetInstance().m_SensorZones)
    {
        int nbrOfSensors = zone->getNumberOfSensors();
        std::vector<osg::Matrix> matrixesForOneSZ;
        for(int cnt{0}; cnt<nbrOfSensors; cnt++)
        {
            matrixesForOneSZ.push_back(orientations.at(incrementor).getMatrix());
            incrementor++;
        }
        zone->createSpecificNbrOfSensors(matrixesForOneSZ);
    }


    // SensorZones -> wie viele Sensoren pro Zone ?

    //  was wenn Sensoren gelöscht werden müssen

    //  Exception wenn an Anzahl der Sensoren nicht passt
}

void DataManager::UpdateUDPSensorPosition(int pos, const osg::Matrix& mat)
{
    GetInstance().m_UDPSensors.at(pos)->setMatrix(mat);
};

void DataManager::UpdateUDPZone(int pos, const osg::Matrix& mat, int nbrOfSensors)
{
    GetInstance().m_UDPSafetyZones.at(pos)->setPosition(mat);
    GetInstance().m_UDPSafetyZones.at(pos)->setCurrentNbrOfSensors(nbrOfSensors);

}; 

void DataManager::UpdateUDPObstacle(int pos, const osg::Matrix& mat)
{
    GetInstance().m_UDPObstacles.at(pos)->setMatrix(mat);
};



void DataManager::setOriginalZoneColor()
{
    for(const auto& zone : GetSafetyZones())
        zone->setOriginalColor();
}


