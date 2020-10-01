#pragma once

#include "Helper.h"
#include "Camera.h"
#include "Zone.h"

enum class SensorType
{
    Camera = 0,
};

enum class ZoneType
{
    ROIzone = 0,
    CameraZone = 1
};

std::unique_ptr<SensorPosition> createSensor(SensorType sensor,osg::Matrix matrix = osg::Matrix::translate(osg::Vec3(8,8,8)), bool visible = true,osg::Vec4 color = osg::Vec4(0,1,0,1));

//std::unique_ptr<Zone> createZone(ZoneType zone, osg::Matrix matrix = osg::Matrix::translate(osg::Vec3(10,8,8)),float length = 10.0f, float width = 5.0f, float height = 3.0f);

std::unique_ptr<SafetyZone> createSafetyZone(SafetyZone::Priority prio, osg::Matrix matrix = osg::Matrix::translate(osg::Vec3(10,8,8)), float length = 10.0f, float width = 5.0f, float height = 3.0f);

std::unique_ptr<SensorZone> createSensorZone();