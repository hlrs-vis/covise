/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++
#include <iostream>
#include <fstream>
#include <assert.h>

// Local:
#include "WidgetInfo.h"
#include "Events.h"
#include "PickBox.h"

using namespace osg;
using namespace cui;
using namespace std;

WidgetInfo::WidgetInfo()
{
    reset();
}

WidgetInfo::WidgetInfo(Events *events, Widget *widget)
{
    _widget = widget;
    _events = events;

    // find all geode nodes which belong to _widget
    GeodeVisitor gv;
    _widget->getNode()->accept(gv);
    _geodeList = gv.getGeodes();

    assert(!_geodeList.empty());

    //std::cerr << "number of geodes in widget: " << _geodeList.size() << endl;

    list<Geode *>::iterator iter;
    for (iter = _geodeList.begin(); iter != _geodeList.end(); iter++)
        (*iter)->setNodeMask((*iter)->getNodeMask() | 2); // Philip changed this needed intersect bit set
    //(*iter)->setNodeMask(1);

    _isectGeode = 0;
    _box = NULL;
}

WidgetInfo::WidgetInfo(PickBox *box)
{
    _widget = box;
    _events = box;
    _isectGeode = 0;
    _box = box;
}

void WidgetInfo::reset()
{
    _widget = 0;
    _events = NULL;
    _geodeList.clear();
    _isectGeode = 0;
    _box = NULL;
}
