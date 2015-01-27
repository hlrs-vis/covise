/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   24.06.2010
**
**************************************************************************/

#ifndef ELEVATIONROADPOLYNOMIALITEM_HPP
#define ELEVATIONROADPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class ElevationEditor;

class ElevationRoadPolynomialItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationRoadPolynomialItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~ElevationRoadPolynomialItem();

    virtual QRectF boundingRect() const;

    QRectF translate(qreal x, qreal y);

    // ElevationEditor //
    //
    ElevationEditor *getElevationEditor() const
    {
        return elevationEditor_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    ElevationRoadPolynomialItem(); /* not allowed */
    ElevationRoadPolynomialItem(const ElevationRoadPolynomialItem &); /* not allowed */
    ElevationRoadPolynomialItem &operator=(const ElevationRoadPolynomialItem &); /* not allowed */

    void init();

    // Handles //
    //
    void rebuildMoveHandles();
    void deleteMoveHandles();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // ElevationEditor //
    //
    ElevationEditor *elevationEditor_;

    // Handles //
    //
    QGraphicsPathItem *moveHandles_;
};

#endif // ELEVATIONROADPOLYNOMIALITEM_HPP
