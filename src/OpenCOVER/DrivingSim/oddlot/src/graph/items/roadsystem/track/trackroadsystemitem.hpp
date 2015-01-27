/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   05.07.2010
**
**************************************************************************/

#ifndef TRACKROADSYSTEMITEM_HPP
#define TRACKROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class RSystemElementRoad;
class TrackRoadItem;

class TrackRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    virtual ~TrackRoadSystemItem();

    // Garbage //
    //
    virtual void notifyDeletion(); // to be implemented by subclasses

    // RoadItems //
    //
    void addRoadItem(TrackRoadItem *item);
    int removeRoadItem(TrackRoadItem *item);

    // Handles //
    //
    void rebuildMoveRotateHandles();
    void rebuildAddHandles();
    void rebuildRoadMoveRotateHandles();
    void deleteHandles();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    TrackRoadSystemItem(); /* not allowed */
    TrackRoadSystemItem(const TrackRoadSystemItem &); /* not allowed */
    TrackRoadSystemItem &operator=(const TrackRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadItems //
    //
    QMap<RSystemElementRoad *, TrackRoadItem *> trackRoadItems_;
};

#endif // TRACKROADSYSTEMITEM_HPP
