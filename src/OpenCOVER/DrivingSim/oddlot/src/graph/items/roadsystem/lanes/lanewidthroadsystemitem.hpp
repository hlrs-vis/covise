/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.07.2010
**
**************************************************************************/

#ifndef LANEWIDTHROADSYSTEMITEM_HPP
#define LANEWIDTHROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/settings/widgets/lanesettings.hpp"

class LaneWidthRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneWidthRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    explicit LaneWidthRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem);
    virtual ~LaneWidthRoadSystemItem();
    void setSettings(LaneSettings *s);

    // Obsever Pattern //
    //
    virtual void updateObserver();

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

private:
    LaneWidthRoadSystemItem(); /* not allowed */
    LaneWidthRoadSystemItem(const LaneWidthRoadSystemItem &); /* not allowed */
    LaneWidthRoadSystemItem &operator=(const LaneWidthRoadSystemItem &); /* not allowed */

    LaneSettings *settings;

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // LANEWIDTHROADSYSTEMITEM_HPP
