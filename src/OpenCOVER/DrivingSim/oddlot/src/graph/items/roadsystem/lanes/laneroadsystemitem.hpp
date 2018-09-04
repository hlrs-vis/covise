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

#ifndef LANEROADSYSTEMITEM_HPP
#define LANEROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class LaneRoadItem;

class LaneRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneRoadSystemItem(TopviewGraph *projectGraph, RoadSystem *roadSystem);
    virtual ~LaneRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

	// RoadItems //
	//
	void addRoadItem(LaneRoadItem *item);
	int removeRoadItem(LaneRoadItem *item);
	LaneRoadItem *getRoadItem(RSystemElementRoad *road);

	// Handles //
	//
	void rebuildMoveRotateHandles();
	void deleteHandles();

private:
    LaneRoadSystemItem(); /* not allowed */
    LaneRoadSystemItem(const LaneRoadSystemItem &); /* not allowed */
    LaneRoadSystemItem &operator=(const LaneRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
	// RoadItems //
	//
	QMap<RSystemElementRoad *, LaneRoadItem *> laneRoadItems_;
};

#endif // LANEROADSYSTEMITEM_HPP
