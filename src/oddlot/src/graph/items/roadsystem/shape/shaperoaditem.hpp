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

#ifndef SHAPEROADITEM_HPP
#define SHAPEROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class ShapeEditor;

class ShapeRoadItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~ShapeRoadItem();

    // Garbage //
    //
    virtual void notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    ShapeRoadItem(); /* not allowed */
    ShapeRoadItem(const ShapeRoadItem &); /* not allowed */
    ShapeRoadItem &operator=(const ShapeRoadItem &); /* not allowed */

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
    ShapeEditor *shapeEditor_;
};

#endif // SHAPEROADITEM_HPP
