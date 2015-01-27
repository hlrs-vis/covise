/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   18.03.2010
**
**************************************************************************/

#ifndef SECTIONHANDLE_HPP
#define SECTIONHANDLE_HPP

#include <QGraphicsPathItem>

class RoadSystemItem;
class RoadItem;
class SectionItem;

class RSystemElementRoad;

class SectionHandle : public QGraphicsPathItem
{
public:
    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SectionHandle(SectionItem *parentSectionItem);
    explicit SectionHandle(RoadSystemItem *parentRoadSystemItem);
    virtual ~SectionHandle()
    { /* does nothing */
    }

    void updateTransform();
    void updatePos(RoadItem *roadItem, const QPointF &position, double sStartHint, double sEndHint);

protected:
private:
    SectionHandle(); /* not allowed */
    SectionHandle(const SectionHandle &); /* not allowed */
    SectionHandle &operator=(const SectionHandle &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    RoadSystemItem *parentRoadSystemItem_;
    SectionItem *parentSectionItem_;

    bool moveItem_;

    //################//
    // STATIC         //
    //################//

private:
    // Path Template //
    //
    static void createPath();
    static QPainterPath *pathTemplate_; // this path will be shared by all handles
};

#endif // SECTIONHANDLE_HPP
