/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#ifndef ROADLINKROADSYSTEMITEM_HPP
#define ROADLINKROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class RoadLinkEditor;
class RoadLinkRoadItem;

class RoadLinkRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem, RoadLinkEditor *editor);
    virtual ~RoadLinkRoadSystemItem();

    void setHandlesSelectable(bool selectable);
    void setRoadsSelectable(bool selectable);

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    RoadLinkRoadSystemItem(); /* not allowed */
    RoadLinkRoadSystemItem(const RoadLinkRoadSystemItem &); /* not allowed */
    RoadLinkRoadSystemItem &operator=(const RoadLinkRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadItems //
    //
    QMap<RSystemElementRoad *, RoadLinkRoadItem *> roadLinkRoadItems_;

    RoadLinkEditor *editor_;
};

#endif // ROADLINKROADSYSTEMITEM_HPP
