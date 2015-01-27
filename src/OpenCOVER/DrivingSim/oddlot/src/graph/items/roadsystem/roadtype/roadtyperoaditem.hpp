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

#ifndef ROADTYPEROADITEM_HPP
#define ROADTYPEROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class RoadTypeEditor;

class RoadTypeRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadTypeRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~RoadTypeRoadItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    RoadTypeRoadItem(); /* not allowed */
    RoadTypeRoadItem(const RoadTypeRoadItem &); /* not allowed */
    RoadTypeRoadItem &operator=(const RoadTypeRoadItem &); /* not allowed */

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
    RoadTypeEditor *roadTypeEditor_;
};

#endif // ROADTYPEROADITEM_HPP
