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

#ifndef JUNCTIONROADITEM_HPP
#define JUNCTIONROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class JunctionRoadSystemItem;
class RSystemElementRoad;

class JunctionEditor;

class JunctionRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionRoadItem(JunctionRoadSystemItem *parentJunctionRoadSystemItem, RSystemElementRoad *road);
    virtual ~JunctionRoadItem();

    // Garbage //
    //
    virtual void notifyDeletion(); // to be implemented by subclasses

    // Handles //
    //
    void rebuildMoveHandles();
    void rebuildAddHandles();
    void deleteHandles();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    JunctionRoadItem(); /* not allowed */
    JunctionRoadItem(const JunctionRoadItem &); /* not allowed */
    JunctionRoadItem &operator=(const JunctionRoadItem &); /* not allowed */

    void init();
    void rebuildSections(bool fullRebuild = false);

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);

    //################//
    // PROPERTIES     //
    //################//

private:
    // JunctionEditor //
    //
    JunctionEditor *junctionEditor_;

    // JunctionRoadSystemItem //
    //
    JunctionRoadSystemItem *parentJunctionRoadSystemItem_;

    // Handles //
    //
    QGraphicsPathItem *handlesItem_;
};

#endif // TRACKROADITEM_HPP
