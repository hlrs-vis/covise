/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.05.2010
**
**************************************************************************/

#ifndef LINKHANDLE_HPP
#define LINKHANDLE_HPP

#include "handle.hpp"

class LinkHandle : public Handle
{

    //################//
    // STATIC         //
    //################//

public:
    enum LinkHandleType
    {
        DHLT_START = 0x1,
        DHLT_CENTER = 0x2,
        DHLT_END = 0x4
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LinkHandle(QGraphicsItem *parent);
    virtual ~LinkHandle();

protected:
    void setHandleType(LinkHandle::LinkHandleType linkHandleType);

private:
    LinkHandle(); /* not allowed */
    LinkHandle(const LinkHandle &); /* not allowed */
    LinkHandle &operator=(const LinkHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
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
    LinkHandle::LinkHandleType linkHandleType_;

    //################//
    // STATIC         //
    //################//

private:
    // Path Template //
    //
    // this path will be shared by all handles
    static void createPath();
    static QPainterPath *pathTemplate_;

    static double halfheight_;
};

#endif // LINKHANDLE_HPP
