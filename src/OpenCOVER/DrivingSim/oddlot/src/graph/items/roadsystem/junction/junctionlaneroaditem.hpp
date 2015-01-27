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

#ifndef JUNCTIONLANEROADITEM_HPP
#define JUNCTIONLANEROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;
class JunctionEditor;

class JunctionLaneRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionLaneRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~JunctionLaneRoadItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    JunctionLaneRoadItem(); /* not allowed */
    JunctionLaneRoadItem(const JunctionLaneRoadItem &); /* not allowed */
    JunctionLaneRoadItem &operator=(const JunctionLaneRoadItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    JunctionEditor *junctionEditor_;
};

#endif // JUNCTIONLANEROADITEM_HPP
