/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef CROSSFALLROADITEM_HPP
#define CROSSFALLROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class CrossfallEditor;

class CrossfallRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~CrossfallRoadItem();

    // Garbage //
    //
    virtual void notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    CrossfallRoadItem(); /* not allowed */
    CrossfallRoadItem(const CrossfallRoadItem &); /* not allowed */
    CrossfallRoadItem &operator=(const CrossfallRoadItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);

    //################//
    // PROPERTIES     //
    //################//

private:
    CrossfallEditor *crossfallEditor_;
};

#endif // CROSSFALLROADITEM_HPP
