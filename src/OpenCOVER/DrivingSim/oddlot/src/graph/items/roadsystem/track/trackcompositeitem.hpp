/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.04.2010
**
**************************************************************************/

#ifndef TRACKCOMPOSITEITEM_HPP
#define TRACKCOMPOSITEITEM_HPP

#include "trackcomponentitem.hpp"

class TrackComposite;

class TrackCompositeItem : public TrackComponentItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackCompositeItem(TrackRoadItem *parentTrackRoadItem, TrackComposite *trackComposite);
    explicit TrackCompositeItem(TrackComponentItem *parentTrackComponentItem, TrackComposite *trackComposite);
    virtual ~TrackCompositeItem();

    // Graphics //
    //
    virtual void updateColor() = 0;
    virtual void createPath() = 0;

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    TrackCompositeItem(); /* not allowed */
    TrackCompositeItem(const TrackCompositeItem &); /* not allowed */
    TrackCompositeItem &operator=(const TrackCompositeItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual void ungroupComposite();

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual void			mousePressEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);
    //
    //	virtual void			hoverEnterEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverLeaveEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverMoveEvent(QGraphicsSceneHoverEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Track //
    //
    TrackComposite *trackComposite_;
};

#endif // TRACKCOMPOSITEITEM_HPP
