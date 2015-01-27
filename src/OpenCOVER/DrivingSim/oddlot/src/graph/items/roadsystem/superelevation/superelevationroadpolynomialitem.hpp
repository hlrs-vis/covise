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

#ifndef SUPERELEVATIONROADPOLYNOMIALITEM_HPP
#define SUPERELEVATIONROADPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class SuperelevationEditor;

class SuperelevationRoadPolynomialItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationRoadPolynomialItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~SuperelevationRoadPolynomialItem();

    virtual QRectF boundingRect() const;

    // SuperelevationEditor //
    //
    SuperelevationEditor *getSuperelevationEditor() const
    {
        return superelevationEditor_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    SuperelevationRoadPolynomialItem(); /* not allowed */
    SuperelevationRoadPolynomialItem(const SuperelevationRoadPolynomialItem &); /* not allowed */
    SuperelevationRoadPolynomialItem &operator=(const SuperelevationRoadPolynomialItem &); /* not allowed */

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
    // SuperelevationEditor //
    //
    SuperelevationEditor *superelevationEditor_;

    // Handles //
    //
    QGraphicsPathItem *moveHandles_;
};

#endif // SUPERELEVATIONROADPOLYNOMIALITEM_HPP
