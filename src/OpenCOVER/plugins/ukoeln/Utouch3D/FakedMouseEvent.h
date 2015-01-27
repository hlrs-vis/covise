/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FAKEDMOUSEEVENT_H
#define FAKEDMOUSEEVENT_H

class FakedMouseEvent
{
public:
    FakedMouseEvent(int eventType, int x, int y);
    ~FakedMouseEvent();

    int getEventType() const;
    int getXPos() const;
    int getYPos() const;

private:
    // osgGA::GUIEventAdapter::EventType
    int eventType;

    // window coordinates in pixel
    int xPos;
    int yPos;
};

#endif // FAKEDMOUSEEVENT_H
