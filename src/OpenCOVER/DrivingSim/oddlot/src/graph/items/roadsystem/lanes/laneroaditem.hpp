/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/15/2010
**
**************************************************************************/

#ifndef LANEROADITEM_HPP
#define LANEROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;
class LaneEditor;

class LaneRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~LaneRoadItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    LaneRoadItem(); /* not allowed */
    LaneRoadItem(const LaneRoadItem &); /* not allowed */
    LaneRoadItem &operator=(const LaneRoadItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    LaneEditor *laneEditor_;
};

#endif // LANEROADITEM_HPP
