/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#ifndef SUPERELEVATIONROADITEM_HPP
#define SUPERELEVATIONROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class SuperelevationEditor;

class SuperelevationRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~SuperelevationRoadItem();

    // Garbage //
    //
    virtual void notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    SuperelevationRoadItem(); /* not allowed */
    SuperelevationRoadItem(const SuperelevationRoadItem &); /* not allowed */
    SuperelevationRoadItem &operator=(const SuperelevationRoadItem &); /* not allowed */

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
    SuperelevationEditor *superelevationEditor_;
};

#endif // SUPERELEVATIONROADITEM_HPP
