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

#ifndef INCLUDED_TUIOPOINT_H
#define INCLUDED_TUIOPOINT_H

#define TUIO_UNDEFINED -1

class TuioPoint
{

protected:
    float xpos, ypos;
    long timestamp;

public:
    TuioPoint(float xpos, float ypos)
    {
        this->xpos = xpos;
        this->ypos = ypos;
        timestamp = TUIO_UNDEFINED;
    };

    TuioPoint(TuioPoint *tuioPoint)
    {
        this->xpos = tuioPoint->getX();
        this->ypos = tuioPoint->getY();
        timestamp = TUIO_UNDEFINED;
    };

    ~TuioPoint(){};

    void update(TuioPoint *tuioPoint)
    {
        this->xpos = tuioPoint->getX();
        this->ypos = tuioPoint->getY();
        timestamp = TUIO_UNDEFINED;
    };

    void update(float xpos, float ypos)
    {
        this->xpos = xpos;
        this->ypos = ypos;
        timestamp = TUIO_UNDEFINED;
    };

    float getX()
    {
        return xpos;
    };
    float getY()
    {
        return ypos;
    };

    float getDistance(float x, float y)
    {
        float dx = xpos - x;
        float dy = ypos - y;
        return sqrtf(dx * dx + dy * dy);
    }

    float getDistance(TuioPoint *tuioPoint)
    {
        float dx = xpos - tuioPoint->getX();
        float dy = ypos - tuioPoint->getY();
        return sqrtf(dx * dx + dy * dy);
    }

    float getAngle(TuioPoint *tuioPoint)
    {

        float side = tuioPoint->getX() - xpos;
        float height = tuioPoint->getY() - ypos;
        float distance = tuioPoint->getDistance(xpos, ypos);

        float angle = (float)(asin(side / distance) + M_PI / 2);
        if (height < 0)
            angle = 2.0f * (float)M_PI - angle;

        return angle;
    }

    float getAngleDegrees(TuioPoint *tuioPoint)
    {
        return ((getAngle(tuioPoint) / (float)M_PI) * 180.0f);
    }

    float getScreenX(int w)
    {
        return xpos * w;
    };
    float getScreenY(int h)
    {
        return ypos * h;
    };

    long getUpdateTime()
    {
        return timestamp;
    };
    void setUpdateTime(long timestamp)
    {
        this->timestamp = timestamp;
    };
};

#endif
