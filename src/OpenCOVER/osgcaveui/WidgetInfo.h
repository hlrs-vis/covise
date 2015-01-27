/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_WIDGETINFO_H_
#define _CUI_WIDGETINFO_H_

// OSG:
#include <osg/Geode>
#include <list>

#include "Widget.h"

namespace cui
{

class PickBox;
class Events;
class Widget;

/** This class stores two types of widgets: geometry and bounding box.
  They are distinguished by which of the attributes _geode and _bbox is NULL.
  The difference is that geometry based widgets can be found by using OSG's 
  scenegraph traversal based intersection test, while bounding box widgets
  need to be treated separately because the cursor does not necessarily 
  intersect with the boundaries of the box but with an implicit surface of
  the box.
*/
class CUIEXPORT WidgetInfo
{
public:
    Widget *_widget;
    Events *_events; ///< required field for widgets of type Geode and PickBox
    std::list<osg::Geode *> _geodeList; ///< required field for widgets of type Geode
    osg::Geode *_isectGeode; ///< contains the intersected geode
    PickBox *_box; ///< required field for widgets of type PickBox

    WidgetInfo();
    WidgetInfo(Events *, Widget *);
    WidgetInfo(PickBox *);
    void reset();
};

class CUIEXPORT GeodeVisitor : public osg::NodeVisitor
{
public:
    GeodeVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN){};

    std::list<osg::Geode *> getGeodes()
    {
        return _geodes;
    }

    virtual void apply(osg::Geode &geode)
    {
        _geodes.push_back(&geode);
    }

private:
    std::list<osg::Geode *> _geodes;
};
}

#endif
