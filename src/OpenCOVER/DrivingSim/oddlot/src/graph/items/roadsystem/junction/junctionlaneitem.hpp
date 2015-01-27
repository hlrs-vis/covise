/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/18/2010
**
**************************************************************************/

#ifndef JUNCTIONLANEITEM_HPP
#define JUNCTIONLANEITEM_HPP

#include "src/graph/items/graphelement.hpp"

class JunctionLaneSectionItem;

class JunctionLaneItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionLaneItem(JunctionLaneSectionItem *parentJunctionLaneSectionItem, Lane *lane);
    virtual ~JunctionLaneItem();

    // Lane //
    //
    JunctionLaneSectionItem *getParentJunctionLaneSectionItem() const
    {
        return parentJunctionLaneSectionItem_;
    }
    Lane *getLane() const
    {
        return lane_;
    }

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    JunctionLaneItem(); /* not allowed */
    JunctionLaneItem(const JunctionLaneItem &); /* not allowed */
    JunctionLaneItem &operator=(const JunctionLaneItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual void hideParentRoad();
    //	virtual void			removeSection();
    virtual void removeParentRoad();

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual void			mousePressEvent(QGraphicsSceneMouseEvent * event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    JunctionLaneSectionItem *parentJunctionLaneSectionItem_;

    LaneSection *parentLaneSection_;
    Lane *lane_;

    RSystemElementRoad *grandparentRoad_;
};

#endif // JUNCTIONLANEITEM_HPP
