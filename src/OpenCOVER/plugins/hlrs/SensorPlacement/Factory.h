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

std::unique_ptr<SensorPosition> createSensor(SensorType sensor,osg::Matrix matrix = osg::Matrix::translate(osg::Vec3(8,8,8)));

std::unique_ptr<Zone> createZone(ZoneType zone);