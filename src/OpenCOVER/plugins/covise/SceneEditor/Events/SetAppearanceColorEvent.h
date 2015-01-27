/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SET_APPEARANCE_COLOR_EVENT_H
#define SET_APPEARANCE_COLOR_EVENT_H

#include "Event.h"

#include <osg/Vec4>

#include <string>

class SetAppearanceColorEvent : public Event
{
public:
    SetAppearanceColorEvent();
    virtual ~SetAppearanceColorEvent();

    void setScope(std::string group);
    std::string getScope();

    void setColor(osg::Vec4 color);
    osg::Vec4 getColor();

private:
    std::string _scope;
    osg::Vec4 _color;
};

#endif
