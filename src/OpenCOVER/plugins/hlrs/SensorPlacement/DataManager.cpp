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

DataManager::DataManager()
{
    m_Root = new osg::Group();
    m_Root->setName("SensorPlacement");
   // m_Root->setNodeMask(m_Root->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
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
       /* if(status)
        {
            GetInstance().Remove(zone.get());
            return;
        }
        */
    }

    for(const auto& sensor : GetInstance().m_Sensors)
    {
        bool status = sensor->preFrame();
        if(!status)
        {
            GetInstance().Remove(sensor.get());
            return;
        }
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


void DataManager::AddSafetyZone(upZone zone)
{
    GetInstance().m_Root->addChild(zone.get()->getZone().get());
    GetInstance().m_SafetyZones.push_back(std::move(zone));        
}

void DataManager::AddSensor(upSensor sensor)
{
    GetInstance().m_Root->addChild(sensor.get()->getSensor().get());
    GetInstance().m_Sensors.push_back(std::move(sensor));     
}

template<typename T>
void DataManager::Remove(T* object)
{
    std::cout <<"befor: "<<GetInstance().m_Sensors.size()<<std::endl;
    if(dynamic_cast<SensorPosition*>(object))
    {
        std::cout<<"Cast successful"<<std::endl;
        GetInstance().m_Root->removeChild(object->getSensor());
        GetInstance().m_Sensors.erase(std::remove_if(GetInstance().m_Sensors.begin(),GetInstance().m_Sensors.end(),[object](std::unique_ptr<SensorPosition>const& it){return object == it.get();}));
    }
    // else if(dynamic_cast<Zone*>(object))
    // {
        // GetInstance().m_Root->removeChild(object->getZone());
        // GetInstance().m_SafetyZones.erase(std::remove_if(GetInstance().m_SafetyZones.begin(),GetInstance().m_SafetyZones.end(),[object](std::unique_ptr<Zone>const& it){return object == it.get();}));
    // }
    else
    {
        std::cout<<"Object of unknown Type can't be removed"<<std::endl;
    }
    
    std::cout <<"after: "<<GetInstance().m_Sensors.size()<<std::endl;
}
