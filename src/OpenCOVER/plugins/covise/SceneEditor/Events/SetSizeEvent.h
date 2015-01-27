/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SET_SIZE_EVENT_H
#define SET_SIZE_EVENT_H

#include "Event.h"

class SetSizeEvent : public Event
{
public:
    SetSizeEvent();
    virtual ~SetSizeEvent();

    void setWidth(float w);
    float getWidth();

    float getHeight();
    void setHeight(float h);

    float getLength();
    void setLength(float l);

private:
    float _width;
    float _height;
    float _length;
};

#endif
