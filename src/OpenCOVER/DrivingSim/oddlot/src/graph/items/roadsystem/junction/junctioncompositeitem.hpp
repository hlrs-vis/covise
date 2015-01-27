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

#ifndef JUNCTIONCOMPOSITEITEM_HPP
#define JUNCTIONCOMPOSITEITEM_HPP

#include "junctioncomponentitem.hpp"

class TrackComposite;

class JunctionCompositeItem : public JunctionComponentItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionCompositeItem(JunctionRoadItem *parentJunctionRoadItem, TrackComposite *trackComposite);
    explicit JunctionCompositeItem(JunctionComponentItem *parentJunctionComponentItem, TrackComposite *trackComposite);
    virtual ~JunctionCompositeItem();

    // Graphics //
    //
    virtual void updateColor() = 0;
    virtual void createPath() = 0;

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    JunctionCompositeItem(); /* not allowed */
    JunctionCompositeItem(const JunctionCompositeItem &); /* not allowed */
    JunctionCompositeItem &operator=(const JunctionCompositeItem &); /* not allowed */

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

#endif // JUNCTIONCOMPOSITEITEM_HPP
