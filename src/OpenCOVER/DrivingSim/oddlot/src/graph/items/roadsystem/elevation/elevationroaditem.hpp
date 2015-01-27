/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.06.2010
**
**************************************************************************/

#ifndef ELEVATIONROADITEM_HPP
#define ELEVATIONROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class ElevationEditor;

class ElevationRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~ElevationRoadItem();

    // Garbage //
    //
    virtual void notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    ElevationRoadItem(); /* not allowed */
    ElevationRoadItem(const ElevationRoadItem &); /* not allowed */
    ElevationRoadItem &operator=(const ElevationRoadItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    //################//
    // PROPERTIES     //
    //################//

private:
    ElevationEditor *elevationEditor_;
};

#endif // ELEVATIONROADITEM_HPP
