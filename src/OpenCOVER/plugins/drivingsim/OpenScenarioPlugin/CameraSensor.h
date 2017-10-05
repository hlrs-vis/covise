#ifndef CAMERA_SENSOR_H
#define CAMERA_SENSOR_H

#include "Entity.h"
#include <OpenScenario/schema/oscVehicle.h>


class CameraSensor
{
	osg::Matrix cameraPosition;
	double FoV;
	Entity *myEntity;
	OpenScenario::oscVehicle *myVehicle;
 public:
	CameraSensor(Entity *e,OpenScenario::oscVehicle *v, osg::Matrix pos, double FOV);
	~CameraSensor();
	void updateView();
	

};

#endif // CAMERA_SENSOR_H
