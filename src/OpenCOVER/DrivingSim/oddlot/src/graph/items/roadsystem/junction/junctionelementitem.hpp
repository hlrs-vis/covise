/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   06.04.2010
**
**************************************************************************/

#ifndef JUNCTIONELEMENTITEM_HPP
#define JUNCTIONELEMENTITEM_HPP

#include "junctioncomponentitem.hpp"

class TrackElement;

class JunctionElementItem : public JunctionComponentItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionElementItem(JunctionRoadItem *parentJunctionRoadItem, TrackElement *trackElement);
    explicit JunctionElementItem(JunctionComponentItem *parentJunctionComponentItem, TrackElement *trackElement);
    virtual ~JunctionElementItem();

    // Graphics //
    //
    virtual void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    JunctionElementItem(); /* not allowed */
    JunctionElementItem(const JunctionElementItem &); /* not allowed */
    JunctionElementItem &operator=(const JunctionElementItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Track //
    //
    TrackElement *trackElement_;

    // Mouse //
    //
    QPointF pressPos_;
};

#endif // JUNCTIONELEMENTITEM_HPP
