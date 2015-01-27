/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QDebug>

#include "MEPort.h"
#include "nodes/MENode.h"
#include "handler/MEMainHandler.h"
#include "handler/MELinkListHandler.h"
#include "widgets/MEGraphicsView.h"

/*!
   \class MEPort
   \brief Base class for parameter and data port of a module node
*/

MEPort::MEPort(MENode *n, QGraphicsScene *scene)
    : QGraphicsRectItem(n)
    , shown(false)
    , hasObjData(false)
    , selected(0)
    , xpos(-1)
    , ypos(-1)
    , hostid(-1)
    , links(0)
    , node(n)
{
    setCursor(Qt::CrossCursor);
    setAcceptHoverEvents(true);
    setZValue(4);

    ex = MEMainHandler::instance()->getPortSize();
    ey = ex;
    distance = 2;
    setRect(0, 0, ex, ex);
}

MEPort::MEPort(MENode *n, QGraphicsScene *scene, const QString &pname, const QString &description, int ptype)
    : QGraphicsRectItem(n)
    , shown(false)
    , hasObjData(false)
    , selected(0)
    , portType(ptype)
    , xpos(-1)
    , ypos(-1)
    , hostid(-1)
    , links(0)
    , description(description)
    , portname(pname)
    , node(n)
{
    setCursor(Qt::CrossCursor);
    setAcceptHoverEvents(true);

    ex = MEMainHandler::instance()->getPortSize();
    ey = ex;
    distance = 2;
    setRect(0, 0, ex, ex);
}

MEPort::~MEPort()
{
}

//!
//! paint the data or parameter port
//!
void MEPort::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{

    // highlight port
    if (selected == 0)
    // draw a round, shadowed ball
    {
        // shadow
        painter->setPen(Qt::NoPen);
        painter->setBrush(Qt::darkGray);
        painter->drawRect(distance, distance, ex, ex);

        // ball
        QRadialGradient gradient(4, 4, ex / 2);
        gradient.setCenter(6, 6);
        gradient.setFocalPoint(6, 6);

        painter->setPen(QPen(Qt::black, 0));
        gradient.setColorAt(0, Qt::white);
        gradient.setColorAt(1, portcolor);
        painter->setBrush(gradient);
        painter->drawRect(0, 0, ex, ex);
    }
    else
    {
        painter->setPen(QPen(Qt::black, 0));
        painter->setBrush(MEMainHandler::s_highlightColor);
        painter->drawRect(0, 0, ex, ex);

        // crossed
        if (selected & Clicked)
        {
            painter->setPen(QPen(Qt::black, 3));
            painter->setBrush(Qt::NoBrush);
            painter->drawLine(0, 0, ex, ex);
            painter->drawLine(0, ex, ex, 0);
        }

        // circled
        if (selected & Target)
        {
            painter->setPen(QPen(Qt::black, 3));
            painter->setBrush(Qt::NoBrush);
            painter->drawEllipse(0, 0, ex, ex);
        }
    }

    if (hasObjData)
    {
        painter->setPen(QPen(Qt::white, 3));
        painter->setBrush(Qt::NoBrush);
        painter->drawRect(0, 0, ex, ex);
    }
}

//!
//! select/deselect a port
//!
void MEPort::setSelected(SelectionType type, bool flag)
{
    if (type == Deselect)
    {
        setZValue(4);
        selected = 0;
    }
    else
    {
        selected &= ~type;
        if (flag)
            selected |= type;
    }

    prepareGeometryChange();
    update();
    //highlightPorts(flag);
}

//!
//! get connection state
//!
bool MEPort::isConnected()
{
    if (links > 0)
        return true;
    else
        return false;
}

//!
//! show/hide a parameter inside a module icon (YAC only for parameter ports)
//!
void MEPort::setShown(bool flag)
{
    setVisible(flag);
    shown = flag;

#ifdef YAC
    node->layoutItem();
#endif
}

//!
//!  highlight connection lines & ports
//!
void MEPort::hoverEnterEvent(QGraphicsSceneHoverEvent *)
{
    MELinkListHandler::instance()->highlightPortAndLinks(this, true);
    if (isConnectable())
        MEGraphicsView::instance()->hoverEnterPort(this);
}

//!
//! reset connection lines & ports
//!
void MEPort::hoverLeaveEvent(QGraphicsSceneHoverEvent *)
{
    MELinkListHandler::instance()->highlightPortAndLinks(this, false);
    MEGraphicsView::instance()->hoverLeavePort(this);
}

//!
//! mouse pressed
//!
void MEPort::mousePressEvent(QGraphicsSceneMouseEvent *e)
{
    if (MEMainHandler::instance()->isMaster() && isConnectable())
        MEGraphicsView::instance()->portPressed(this, e);
}

//!
//! mouse released
//!
void MEPort::mouseReleaseEvent(QGraphicsSceneMouseEvent *e)
{
    MEGraphicsView::instance()->portReleased(this, e);
}

//!
//! mouse move
//!
void MEPort::mouseMoveEvent(QGraphicsSceneMouseEvent *)
{
    MEGraphicsView::instance()->portMoved(this);
}

//!
//! add a link
//!
void MEPort::addLink(MEPort *)
{
    links++;
}

//!
//! remove a link
//!
void MEPort::delLink(MEPort *)
{
    links--;
}

//!
//! descide if a port can be connected
//!
bool MEPort::isConnectable()
{
    return portType == DOUT
           || links == 0
           || "Renderer" == getNode()->getCategory()
           || "SRenderer" == getNode()->getCategory();
}
