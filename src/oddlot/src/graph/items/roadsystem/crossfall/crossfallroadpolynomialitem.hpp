/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.07.2010
**
**************************************************************************/

#ifndef CROSSFALLROADPOLYNOMIALITEM_HPP
#define CROSSFALLROADPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;

class CrossfallEditor;

class CrossfallRoadPolynomialItem : public RoadItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallRoadPolynomialItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~CrossfallRoadPolynomialItem();

    virtual QRectF boundingRect() const;

    // CrossfallEditor //
    //
    CrossfallEditor *getCrossfallEditor() const
    {
        return crossfallEditor_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    CrossfallRoadPolynomialItem(); /* not allowed */
    CrossfallRoadPolynomialItem(const CrossfallRoadPolynomialItem &); /* not allowed */
    CrossfallRoadPolynomialItem &operator=(const CrossfallRoadPolynomialItem &); /* not allowed */

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
    // CrossfallEditor //
    //
    CrossfallEditor *crossfallEditor_;

    // Handles //
    //
    QGraphicsPathItem *moveHandles_;
};

#endif // CROSSFALLROADPOLYNOMIALITEM_HPP
