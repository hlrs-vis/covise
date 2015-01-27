/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_NODELINK_H
#define ME_NODELINK_H

#include <QGraphicsPathItem>

class QString;
class QPainterPath;

class MEPort;

//================================================
class MENodeLink : public QGraphicsPathItem
//================================================
{

public:
    MENodeLink(MEPort *p1, MEPort *p2, QGraphicsItem *parent = 0);
    ~MENodeLink();

    MEPort *port1, *port2;

    void moveLines();
    void highlightLines(bool);
    void removeLines();
    void removeLink(QGraphicsSceneContextMenuEvent *);

private:
    void checkDepPort(bool);
    void makeLine(QPointF, QPointF);
    void makeTopBottomCurve(QPointF, QPointF);
    void makeBottomTopLeftCurve(QPointF, QPointF, QPointF, QPointF);
    void makeBottomTopRightCurve(QPointF, QPointF, QPointF, QPointF);
    void definePortLines();
    void defineLines();

    qreal xend, yend;

protected:
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *e);
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *e);
    void mousePressEvent(QGraphicsSceneMouseEvent *e);
    void hoverEnterEvent(QGraphicsSceneHoverEvent *e);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *e);
};

#endif
