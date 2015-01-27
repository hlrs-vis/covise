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

#ifndef INCLUDED_TUIOCONTAINER_H
#define INCLUDED_TUIOCONTAINER_H

#include <list>
#include <math.h>
#include "TuioPoint.h"
#include <iostream>

#define TUIO_ADDED 0
#define TUIO_UPDATED 1
#define TUIO_REMOVED 2

class TuioContainer : public TuioPoint
{

protected:
    long session_id;
    float xpos, ypos;
    float x_speed, y_speed;
    float motion_speed, motion_accel;
    std::list<TuioPoint> path;

    int state;

public:
    TuioContainer(long s_id, float xpos, float ypos)
        : TuioPoint(xpos, ypos)
    {
        this->session_id = s_id;
        this->x_speed = 0.0f;
        this->y_speed = 0.0f;
        this->motion_speed = 0.0f;
        this->motion_accel = 0.0f;
        TuioPoint p(xpos, ypos);
        path.push_back(p);

        state = TUIO_ADDED;
    };

    TuioContainer(TuioContainer *tuioContainer)
        : TuioPoint(tuioContainer)
    {
        this->session_id = tuioContainer->getSessionID();
        this->x_speed = 0.0f;
        this->y_speed = 0.0f;
        this->motion_speed = 0.0f;
        this->motion_accel = 0.0f;
        TuioPoint p(xpos, ypos);
        path.push_back(p);

        state = TUIO_ADDED;
    };

    virtual ~TuioContainer(){};

    virtual void update(float xpos, float ypos, float xspeed, float yspeed, float maccel)
    {
        TuioPoint::update(xpos, ypos);
        this->x_speed = xspeed;
        this->y_speed = yspeed;
        this->motion_speed = (float)sqrt(xspeed * xspeed + yspeed * yspeed);
        this->motion_accel = maccel;
        TuioPoint p(xpos, ypos);
        path.push_back(p);

        state = TUIO_UPDATED;
    };

    virtual void update(TuioContainer *tuioContainer)
    {
        TuioPoint::update(tuioContainer);
        this->x_speed = tuioContainer->getXSpeed();
        this->y_speed = tuioContainer->getYSpeed();
        this->motion_speed = tuioContainer->getMotionSpeed();
        this->motion_accel = tuioContainer->getMotionAccel();
        TuioPoint p(xpos, ypos);
        path.push_back(p);

        state = TUIO_UPDATED;
    };

    virtual long getSessionID()
    {
        return session_id;
    };

    virtual TuioPoint getPosition()
    {
        TuioPoint p(xpos, ypos);
        return p;
    };

    virtual std::list<TuioPoint> getPath()
    {
        return path;
    };

    virtual void remove()
    {
        state = TUIO_REMOVED;
        timestamp = TUIO_UNDEFINED;
    }

    virtual float getXSpeed()
    {
        return x_speed;
    };
    virtual float getYSpeed()
    {
        return y_speed;
    };
    virtual float getMotionSpeed()
    {
        return motion_speed;
    };
    virtual float getMotionAccel()
    {
        return motion_accel;
    };

    virtual int getState()
    {
        return state;
    };
    virtual void setUpdateTime(long timestamp)
    {
        this->timestamp = timestamp;
        TuioPoint *lastPoint = &path.back();
        if (lastPoint != NULL)
            lastPoint->setUpdateTime(timestamp);
    };
};

#endif
