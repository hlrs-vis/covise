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

#ifndef JUNCTIONROADSYSTEMITEM_HPP
#define JUNCTIONROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class RSystemElementRoad;
class JunctionRoadItem;

class JunctionRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    virtual ~JunctionRoadSystemItem();

    // Garbage //
    //
    virtual void notifyDeletion(); // to be implemented by subclasses

    // RoadItems //
    //
    void addRoadItem(JunctionRoadItem *item);
    int removeRoadItem(JunctionRoadItem *item);

    // Handles //
    //
    void rebuildMoveHandles();
    void rebuildAddHandles();
    void deleteHandles();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    JunctionRoadSystemItem(); /* not allowed */
    JunctionRoadSystemItem(const JunctionRoadSystemItem &); /* not allowed */
    JunctionRoadSystemItem &operator=(const JunctionRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadItems //
    //
    QMap<RSystemElementRoad *, JunctionRoadItem *> junctionRoadItems_;
};

#endif // TRACKROADSYSTEMITEM_HPP
