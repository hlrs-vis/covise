#include "Factory.h"

std::unique_ptr<SensorPosition> Factory::createSensor(SensorType sensor, osg::Matrix matrix, bool visible, osg::Vec4 color)
{
	
    if(sensor == SensorType::Camera){
        return std::make_unique<Camera>(matrix, visible, color);
    }
    else{
        assert(false);
    }
    return std::unique_ptr<SensorPosition>();
}


/*std::unique_ptr<Zone> createZone(ZoneType zone, osg::Matrix matrix,float length, float width , float height)
{
   
    if(zone == ZoneType::ROIzone){
        
        return std::make_unique<SafetyZone>(matrix,length,width,height);
    }
    else if(zone == ZoneType::CameraZone){

        return std::make_unique<SensorZone>(SensorType::Camera, matrix,length,width,height);
    }
    else{
        assert(false);
    }
    return std::unique_ptr<Zone>();
}
*/

std::unique_ptr<SafetyZone> Factory::createSafetyZone(SafetyZone::Priority prio, osg::Matrix matrix, float length, float width , float height)
{
    return std::make_unique<SafetyZone>(matrix, prio, length, width, height);
}

std::unique_ptr<SensorZone> Factory::createSensorZone(SensorType type, osg::Matrix matrix, float length, float width, float height)
{
    
    return std::make_unique<SensorZone>(type,matrix, length, width, height);
}
std::unique_ptr<SensorZone> Factory::createSensorZone(SensorType type, osg::Matrix matrix, float radius)
{
    
    return std::make_unique<SensorZone>(type,matrix, radius);
}
