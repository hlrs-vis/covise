/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/*
* inputhdw.cpp
*
*  Created on: Dec 9, 2014
*      Author: woessner
*/
#include "SpatialLabsDriver.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>

#include <config/CoviseConfig.h>

#include <iostream>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;

#include <util/unixcompat.h>
#include <iostream>
#include <osg/Vec2>

//#include <quat.h>

using namespace std;
using namespace opencover;
SpatialLabsDriver::SpatialLabsDriver(const std::string &config)
    : InputDevice(config)
{

	spatialLabsAPI = new SpatialLabsCoreLib::SpatialLabsCoreLibAPI;



}

bool SpatialLabsDriver::poll()
{
    if (spatialLabsAPI ==NULL)
        return false;


	float ScaleFactor = 1.0;
	float ScaleFactorForScreen = 1.0;
	float CameraOffset = 0.3;
	bool bUseDynamicCameraFOV = true;
	bool bUseFixedHeadPosition = false;
	osg::Vec3 FixedHeadPosition(0.0, -1, 0.0);

	float width = coVRConfig::instance()->screens[0].hsize;
	float height = coVRConfig::instance()->screens[0].vsize;

	osg::Vec2f MonitorSize(width, height);
	osg::Vec2f MonitorSizeHalf = MonitorSize / 2.0;

	SpatialLabsCoreLib::ViewData inViewData
	{
		{
			0,
			0,
			0
		},
		{
			0.0,
			-1.0,
			0.0
		},
		ScaleFactor,
		ScaleFactorForScreen,
		CameraOffset,
		bUseDynamicCameraFOV,
		bUseFixedHeadPosition,
		{
			0.0,
			-1.0,
			0.0
		},
		FocalLengthPlayerCamera,
		{
			width / 2.0f,
			height / 2.0f
		}
	};

    //float x,y,z;
    //zsGetDisplayAngle(displayHandle,&x,&y,&z);
    //fprintf(stderr,"x: %f  y:%f  z:%f\n",x,y,z);


	float outViewPose[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	float outRawEyePos[] = { 0.0, 0.0, 0.0 };
#ifndef _DEBUG
	spatialLabsAPI->GetViewPose(inViewData, outViewPose, outRawEyePos);
#endif // !_DEBUG

	float outViewPose0 = outViewPose[0];
	float outViewPose1 = outViewPose[1];
	float outViewPose2 = outViewPose[2];

    osg::Matrix matrix;
	matrix.makeIdentity();
	matrix(3, 0) * outViewPose0,1000;
	matrix(3, 1) * outViewPose1,1000;
	matrix(3, 2) * outViewPose2,1000;

	/*view.pose.orientation.x = outViewPose[3];
	view.pose.orientation.y = outViewPose[4];
	view.pose.orientation.z = outViewPose[5];
	view.pose.orientation.w = outViewPose[6];

	view.fov.angleLeft = outViewPose[7];
	view.fov.angleRight = outViewPose[8];
	view.fov.angleUp = outViewPose[9];
	view.fov.angleDown = outViewPose[10];*/

		CachedEyeLeft[0] = -outViewPose2;
		CachedEyeLeft[1] = outViewPose0;
		CachedEyeLeft[2] = outViewPose1;
		CachedEyeLeftScreenSpace[0] = outViewPose0;
		CachedEyeLeftScreenSpace[1] = outViewPose1;
		CachedEyeLeftScreenSpace[2] = outViewPose2;
		CachedRawEyeLeft[0] = outRawEyePos[0];
		CachedRawEyeLeft[1] = outRawEyePos[1];
		CachedRawEyeLeft[2] = outRawEyePos[2];

    return true;
}

//====================END of init section============================


SpatialLabsDriver::~SpatialLabsDriver()
{
    stopLoop();
	delete spatialLabsAPI;
}

//==========================main loop =================

bool SpatialLabsDriver::init()
{
    return true;
}

void SpatialLabsDriver::update() //< called by Input::update()
{
    poll();
    InputDevice::update();
}

INPUT_PLUGIN(SpatialLabsDriver)
