/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#ifndef _Revit_PLUGIN_Doors_H
#define _Revit_PLUGIN_Doors_H

#include <cover/coVRPluginSupport.h>
#include <osg/MatrixTransform>
#include <net/tokenbuffer.h>

class DoorInfo
{
public:

    enum SlidingDirection { dirLeft=-1, dirNone=0,dirRight=1 };

	DoorInfo(int id, const char *Name, osg::MatrixTransform *tn, covise::TokenBuffer &tb);
	std::string name;
	osg::MatrixTransform *transformNode;
	int ID;
	bool HandFlipped;
	bool FaceFlipped;
    SlidingDirection isSliding;
	osg::Vec3 HandOrientation;
	osg::Vec3 FaceOrientation;
	osg::Vec3 Direction;
	osg::Vec3 Origin;
	double maxDistance;
    double openingPercentage=0;
	osg::Vec3 Center;
	float activationDistance2;
	bool entered;
	bool left;
	bool isActive;
	double startTime;
	double animationTime;
	void checkStart(osg::Vec3 &viewerPosition); 
	void translateDoor(float fraction);
	osg::BoundingBox boundingBox;
	bool update(osg::Vec3 &viewerPosition); // returns false if updates are done and it can be removed from the list
};



#endif
