/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef coEntity_h
#define coEntity_h
#include <util/coExport.h>
#include <osg/Vec3>


class TRAFFICSIMULATIONEXPORT coEntity // base class for vehicles and pedestrians(pedestrianGeometries) 
{
public:
    coEntity();
    virtual ~coEntity();
	bool isActive() { return activeState; };
	void setActive(bool state) { activeState = state; };
	double newAngle;
	double currentAngle;
	osg::Vec3 speed;
	osg::Vec3 newPosition;
	osg::Vec3 currentPosition;

protected:
	bool activeState = true;
};

#endif
