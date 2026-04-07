/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Doors for Revit Plugin                                      **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-09  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
 \****************************************************************************/
#define QT_NO_EMIT

#include "Doors.h"
#include "RevitPlugin.h"


DoorInfo::DoorInfo(int id, const char *Name, osg::MatrixTransform *tn, covise::TokenBuffer &tb)
{
	ID = id;
	name = Name;
	transformNode = tn;
	tb >> HandFlipped;
	tb >> HandOrientation;
	tb >> FaceFlipped;
	tb >> FaceOrientation;
	tb >> Origin;
	int dir;
	tb >> dir;
	isSliding = SlidingDirection(dir);
	osg::Vec3 BBMin;
	osg::Vec3 BBMax;
	tb >> BBMin;
	tb >> BBMax;
	tb >> maxDistance;
	tb >> openingPercentage;
	boundingBox.set(BBMin, BBMax);
	if (isSliding != SlidingDirection::dirNone)
	{
		if(maxDistance == 0)
		    maxDistance = boundingBox.xMax() - boundingBox.xMin();
		if (maxDistance == 0)
			maxDistance = 1;
		HandOrientation.normalize(); // HandOrientation is in Revit World coordinates so either transform this back to local coordinates orjust use X for now.
		Direction = osg::Vec3(1, 0, 0)*maxDistance;
		if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
		{
			//Direction *= -1;
		}
		else if(strncmp(Name, "DoorMovingParts_Left", 20) == 0)
		{
			Direction *= -1;
		}
		else
		{
			Direction *= isSliding;
		}
	//	if (!HandFlipped)
	//	{
	//		Direction *= -1;
	//	}
	}
	else
	{
		Direction = HandOrientation ^ FaceOrientation;
		Direction.normalize();
		maxDistance = M_PI_2;
		//fprintf(stderr, "HandFlipped %d\n", (int)HandFlipped);
		//fprintf(stderr, "FaceFlipped %d\n", (int)FaceFlipped);

		Direction *= -1;
		if(Origin.x() == 10000)
		{
			Origin = boundingBox._max;
			if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
			{
				Origin = boundingBox._min;
			}
		}
		if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
		{
			Direction *= -1;
		}
	}

	Center = boundingBox.center();
	// transform Center to Object Coordinates
	osg::Matrix tr;
	tr.makeIdentity();
	//cerr << "LocalToVRML: hitPoint: "<<hitPoint[0]<<' '<<hitPoint[1]<<' '<<hitPoint[2]<<endl;
	const osg::Node *parent = transformNode->getParent(0);
	while (parent != NULL && parent != cover->getObjectsRoot())
	{
		osg::Matrix dcsMat;
		const osg::MatrixTransform *mtParent = dynamic_cast<const osg::MatrixTransform *>(parent);
		if (mtParent)
		{
			dcsMat = mtParent->getMatrix();
			tr.postMult(dcsMat);
		}
		if (parent->getNumParents())
			parent = parent->getParent(0);
		else
			parent = NULL;
	}
	Center = Center * tr;
	activationDistance2 = 3; // three m
	activationDistance2 *= activationDistance2; // squared
	animationTime = 1.0; // one second
	isActive = false;
	left = false;
	entered = false;
	translateDoor(0);
}

void DoorInfo::checkStart(osg::Vec3 &viewerPosition)
{
	if (!isActive)
	{
		if ((!(Center[0]==0 && Center[1] == 0 && Center[2] == 0)) && (Center - viewerPosition).length2() < activationDistance2)
		{
			isActive = true;
			RevitPlugin::instance()->activeDoors.push_back(this);
			entered = true;
			left = false;
			startTime = cover->frameTime();
		}
	}
}

void DoorInfo::translateDoor(float fraction)
{
	if (isSliding)
	{
		transformNode->setMatrix(osg::Matrix::translate((Direction*fraction)-(Direction * (openingPercentage/100.0))));
	}
	else
	{
		transformNode->setMatrix(osg::Matrix::translate(-Origin) * osg::Matrix::rotate(maxDistance*fraction, Direction)*osg::Matrix::translate(Origin));
	}
}

bool DoorInfo::update(osg::Vec3 &viewerPosition)
{
	if (!left && ((Center - viewerPosition).length2() > activationDistance2))
	{
		left = true;
		startTime = cover->frameTime();
	}
	if (left)
	{
		if (cover->frameTime()- startTime <  animationTime)
		{
			float fraction = 1.0 - ((cover->frameTime() - startTime) / animationTime);
			translateDoor(fraction);
		}
		else
		{
			translateDoor(0.0);
			left = false;
			isActive = false;
			return false;
		}
	}
	if (entered)
	{
		if (cover->frameTime()- startTime <  animationTime)
		{
			float fraction = (cover->frameTime() - startTime) / animationTime;
			translateDoor(fraction);
		}
		else
		{
			translateDoor(1.0);
			entered = false;
		}
	}
	return true;
}

