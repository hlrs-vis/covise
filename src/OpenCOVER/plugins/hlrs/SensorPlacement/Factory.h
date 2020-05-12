#pragma once

#include "Helper.h"
#include "Sensor.h"
#include "Zone.h"

enum class SensorType
{
    Camera = 0,
};

std::unique_ptr<SensorPosition> createSensor(SensorType sensor)
{
    if(sensor == SensorType::Camera){
        osg::Matrix position;
        position.setTrans(10,10,10);
        return myHelpers::make_unique<Camera>(position);
    }
    else{
        assert(false);
    }
}

enum class ZoneType
{
    ROIzone = 0,
    SensorZone = 1
};

std::unique_ptr<Zone> createZone(ZoneType zone)
{
    osg::Matrix position;
    position.setTrans(8,8,8);

    if(zone == ZoneType::ROIzone){
        
        return myHelpers::make_unique<SafetyZone>(position);
    }
    else if(zone == ZoneType::SensorZone){

        return myHelpers::make_unique<SensorZone>(position);
    }
    else{
        assert(false);
    }
}