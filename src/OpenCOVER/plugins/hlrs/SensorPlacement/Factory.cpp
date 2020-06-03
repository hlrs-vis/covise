#include "Factory.h"

std::unique_ptr<SensorPosition> createSensor(SensorType sensor, osg::Matrix matrix)
{
    if(sensor == SensorType::Camera){
        return myHelpers::make_unique<Camera>(matrix);
    }
    else{
        assert(false);
    }
    return std::unique_ptr<SensorPosition>();
}

std::unique_ptr<Zone> createZone(ZoneType zone)
{
    osg::Matrix position;
    position.setTrans(8,8,8);

    if(zone == ZoneType::ROIzone){
        
        return myHelpers::make_unique<SafetyZone>(position);
    }
    else if(zone == ZoneType::CameraZone){

        return myHelpers::make_unique<SensorZone>(SensorType::Camera,position);
    }
    else{
        assert(false);
    }
    return std::unique_ptr<Zone>();
}