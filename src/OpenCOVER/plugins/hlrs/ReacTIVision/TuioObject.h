/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
	TUIO C++ Library - part of the reacTIVision project
	http://reactivision.sourceforge.net/

	Copyright (c) 2005-2008 Martin Kaltenbrunner <mkalten@iua.upf.edu>
	
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef INCLUDED_TUIOOBJECT_H
#define INCLUDED_TUIOOBJECT_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <list>
#include <math.h>
#include "TuioContainer.h"

class TuioObject : public TuioContainer
{

protected:
    long session_id;
    int fiducial_id;
    float xpos, ypos, angle;
    float x_speed, y_speed, motion_speed, motion_accel;
    float rotation_speed, rotation_accel;
    std::list<TuioPoint> path;

public:
    TuioObject(long s_id, int f_id, float xpos, float ypos, float angle)
        : TuioContainer(s_id, xpos, ypos)
    {
        this->fiducial_id = f_id;
        this->angle = angle;
        this->rotation_speed = 0.0f;
        this->rotation_accel = 0.0f;
    };

    TuioObject(TuioObject *tuioObject)
        : TuioContainer(tuioObject)
    {
        this->fiducial_id = tuioObject->getFiducialID();
        this->angle = angle;
        this->rotation_speed = 0.0f;
        this->rotation_accel = 0.0f;
    };

    ~TuioObject(){};

    void update(float xpos, float ypos, float angle, float xspeed, float yspeed, float rspeed, float maccel, float raccel)
    {
        TuioContainer::update(xpos, ypos, xspeed, yspeed, maccel);
        this->angle = angle;
        this->rotation_speed = rspeed;
        this->rotation_accel = raccel;
    };

    void update(TuioObject *tuioObject)
    {
        TuioContainer::update(tuioObject);
        this->angle = tuioObject->getAngle();
        this->rotation_speed = tuioObject->getRotationSpeed();
        this->rotation_accel = tuioObject->getRotationAccel();
    };

    int getFiducialID()
    {
        return fiducial_id;
    };

    float getAngle()
    {
        return angle;
    }
    float getAngleDegrees()
    {
        return (float)(angle / M_PI * 180);
    }

    float getRotationSpeed()
    {
        return rotation_speed;
    };
    float getRotationAccel()
    {
        return rotation_accel;
    };
};

#endif
