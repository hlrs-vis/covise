/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
	
*/

#ifndef INCLUDED_BBM_Event_H
#define INCLUDED_BBM_Event_H

#define TUIO_UNDEFINED -1

// TuioPoint --> BBM_Event
//	BBM_Event newEvent ( 1 ,  tobj->getFiducialID(),  tobj->getSessionID(),  tobj->getX(),  tobj->getY() ,  tobj->getAngle() );

class BBM_Event
{

protected:
    int EventID, FiducialID, SessionID;
    float xpos, ypos, angle;
    long timestamp;

public:
    BBM_Event(int EventID, int FiducialID, int SessionID, float xpos, float ypos, float angle)
    {
        this->EventID = EventID;
        this->FiducialID = FiducialID;
        this->SessionID = SessionID;
        this->xpos = xpos;
        this->ypos = ypos;
        this->angle = angle;
        timestamp = TUIO_UNDEFINED;
    };

    ~BBM_Event(){};

    void update(int I_event, int marker, float xpos, float ypos)
    {
        this->EventID = EventID;
        this->FiducialID = FiducialID;
        this->SessionID = SessionID;
        this->xpos = xpos;
        this->ypos = ypos;
        this->angle = angle;
        timestamp = TUIO_UNDEFINED;
    };

    int getEventID()
    {
        return EventID;
    };
    int getFiducialID()
    {
        return FiducialID;
    };
    int getSessionID()
    {
        return SessionID;
    };
    float getX()
    {
        return xpos;
    };
    float getY()
    {
        return ypos;
    };
    float getangle()
    {
        return angle;
    };
};

#endif
