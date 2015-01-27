/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include <QPainterPathStroker>
#include <QPen>

#include <covise/covise_msg.h>

#include "MENodeLink.h"
#include "MENode.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "widgets/MEGraphicsView.h"
#include "ports/MEPort.h"
#include "ports/MEDataPort.h"

static int normalLevel = 2;
static int highLevel = 10;
static QPen normalPen(Qt::black, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
static QPen highPen(Qt::red, 3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);

/*!
   \class MENodeLink
   \brief This class provides the connection lines between ports
*/

MENodeLink::MENodeLink(MEPort *first, MEPort *second, QGraphicsItem *)
    : QGraphicsPathItem()
{
    setCursor(Qt::ClosedHandCursor);
    setFlag(ItemIsSelectable, false);
    setAcceptHoverEvents(true);
    setZValue(normalLevel);
    setPen(normalPen);
    MEGraphicsView::instance()->scene()->addItem(this);

    if (first->getPortType() == MEPort::DOUT)
    {
        port1 = first;
        port2 = second;
    }
    else
    {
        port2 = first;
        port1 = second;
    }

    if (port1)
        port1->addLink(port2);

    if (port2)
        port2->addLink(port1);

    // reset port color to optional
    // look for depending input port of module & inform controller

    checkDepPort(false);
    defineLines();
}

MENodeLink::~MENodeLink()
{
    // decrease port links
    if (port1)
    {
        port1->delLink(port2);
        MEPortSelectionHandler::instance()->removePort(MEPortSelectionHandler::HoverConnected, port1);
    }

    if (port2)
    {
        port2->delLink(port1);
        MEPortSelectionHandler::instance()->removePort(MEPortSelectionHandler::HoverConnected, port2);
    }

    // reset port color to optional
    // look for depending
    checkDepPort(true);
}

//!
//! check if depended port is given, change port color, inform controller
//!
void MENodeLink::checkDepPort(bool flag)
{

    QColor color;
    QString text;

    if (flag)
    {
        color = MEMainHandler::s_optionalColor;
        text = "opt";
    }

    else
    {
        color = MEMainHandler::s_requestedColor;
        text = "req";
    }

    MEDataPort *dp = qobject_cast<MEDataPort *>(port1);
    if (dp)
    {
        if (dp->getDemand() == MEDataPort::DEP)
        {
            MENode *node = dp->getNode();
            foreach (MEPort *ptr, node->dinlist)
            {
                if (ptr->getName() == dp->getDependency())
                {
                    ptr->setBrush(color);
                    ptr->setPortColor(color);

                    QStringList buffer;
                    buffer << "DEPEND";
                    buffer << node->getName() << node->getNumber() << node->getHostname() << ptr->getName() << text;
                    QString data = buffer.join("\n");

                    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
                    buffer.clear();
                }
            }
        }
    }
}

//!
//! create the parameter connection line
//!
void MENodeLink::definePortLines()
{
    /*  	MENodeLine *line;

      MENode *node1 = port1->getNode();
      MENode *node2 = port2->getNode();

      // make ports visible
      if(!port1->isVisible() )
         port1->setShown(true);

      if(!port2->isVisible() )
         port2->setShown(true);

      int dx1 = port1->x();
      int dx2 = port2->x();

      // calculate start position 1 and 2
      int x1 = node1->x() + dx1 + port1->width()/2;
      int x2 = node2->x() + dx2 + port2->width()/2;
      int y1 = node1->y();
      int y2 = node2->y();

      // connection of an input and output port
      if(x2 < x1)
         makeLine0(QPointF start, QPointF end);
      else
         makeLine1(QPointF start, QPointF end);

       */
}

//!
//! prepare for the right connection line
//!
void MENodeLink::defineLines()
{
    MENode *node1 = NULL;
    MENode *node2 = NULL;

    // calculate start position 1 and 2
    qreal dx = qreal(MEMainHandler::instance()->getPortSize() / 2);

    if (!port1 || !port2)
        return;

    node1 = port1->getNode();
    QPointF nstart = node1->mapToScene(node1->boundingRect().bottomLeft().x(), node1->boundingRect().bottomLeft().y());
    QPointF start = node1->mapToScene(port1->x(), port1->y());
    start.setX(start.x() + dx);
    start.setY(nstart.y());

    node2 = port2->getNode();
    QPointF nend = node2->mapToScene(node2->boundingRect().topLeft().x(), node2->boundingRect().topLeft().y());
    QPointF end = node2->mapToScene(port2->x(), port2->y());
    end.setX(end.x() + dx);
    end.setY(nend.y());

    // connection of an input and output port
    if (start.x() == end.x())
        makeLine(start, end);

    else if (end.y() > start.y())
        makeTopBottomCurve(start, end);

    else
    {
        if (nend.x() < nstart.x())
            makeBottomTopLeftCurve(start, end, nstart, nend);
        else
            makeBottomTopRightCurve(start, end, nstart, nend);
    }
}

//!
//! remove the connection lines, delete old connection lines & define new one
//!
void MENodeLink::moveLines()
{
    defineLines();
}

//!
//!  user has used the context menu to delete the link
//!
void MENodeLink::removeLink(QGraphicsSceneContextMenuEvent *e)
{
    MEGraphicsView::instance()->deletePossibleLink(this, e);
}

//!
//! remove the connection lines
//!
void MENodeLink::removeLines()
{
    MEMainHandler::instance()->mapWasChanged("DELETE_LINK");

    QStringList buffer;
    QString data;

    MENode *node1 = port1->getNode();
    MENode *node2 = port2->getNode();

    MEDataPort *dp = static_cast<MEDataPort *>(port1);
    qobject_cast<MEDataPort *>(port2)->setPortObjectName(dp->getPortObjectName());

    buffer << "DELETE_LINK";
    buffer << node1->getName() << node1->getNumber() << node1->getHostname() << port1->getName();
    buffer << node2->getName() << node2->getNumber() << node2->getHostname() << port2->getName();
    data = buffer.join("\n");

    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();

    // look if there are synced nodes
    /*int imax = qMin( node1->m_syncList.count(), node2->m_syncList.count());
   for( int i = 0; i< imax; i++)
   {
      MENode *n1 = node1->m_syncList.at(i);
      MENode *n2 = node2->m_syncList.at(i);
      if(n1 && n2)
      {
         MEPort *p1 = n1->getPort(port1->getName());
         MEPort *p2 = n2->getPort(port2->getName());
         if(p1 && p2)
         {
            p1->sendObjMsg("DODEL");
            p2->setPortObjectName(port1->getObjectName() );
            p2->sendObjMsg("DIDEL");

            buffer << "DELETE_LINK";
            buffer << n1->getName() << n1->getNumber() << n1->getHostname() << p1->getName();
            buffer << n2->getName() << n2->getNumber() << n2->getHostname() << p2->getName() ;
            data = buffer.join("\n");

            MEMessageHandler::instance()->sendMessage(COVISE_MESSAGE_UI, data);
            buffer.clear();

            QPalette palette;
            palette.setBrush(p1->backgroundRole(), p1->getPortColor());
            p1->setPalette(palette);
            palette.setBrush(p2->backgroundRole(), p2->getPortColor());
            p2->setPalette(palette);
         }
      }
   }*/

    // reset ports
    port1->setBrush(port1->getPortColor());
    port2->setBrush(port2->getPortColor());
    port1->update();
    port2->update();
}

//!
//! highlight lines
//!
void MENodeLink::highlightLines(bool highlight)
{
    QPen pen;
    int level;

    if (highlight)
    {
        highPen.setColor(MEMainHandler::s_highlightColor);
        pen = highPen;
        level = highLevel;
        MEPortSelectionHandler::instance()->addPort(MEPortSelectionHandler::HoverConnected, port1);
        MEPortSelectionHandler::instance()->addPort(MEPortSelectionHandler::HoverConnected, port2);
    }

    else
    {
        pen = normalPen;
        level = normalLevel;
        MEPortSelectionHandler::instance()->removePort(MEPortSelectionHandler::HoverConnected, port1);
        MEPortSelectionHandler::instance()->removePort(MEPortSelectionHandler::HoverConnected, port2);
    }

    setPen(pen);
    setZValue(level);
}

//!
//! create a single line
//!
void MENodeLink::makeLine(QPointF start, QPointF end)
{
    QPolygonF poly;
    poly << start << end;
    QPainterPath path;
    path.addPolygon(poly);
    QPainterPathStroker stroker;
    setPath(stroker.createStroke(path));
}

//!
//! create a single line from bottom to top
//!
void MENodeLink::makeTopBottomCurve(QPointF start, QPointF end)
{
    qreal dd = (end.y() - start.y()) * 0.5;
    QPainterPath path(start);
    path.cubicTo(QPointF(start.x(), start.y() + dd), QPointF(end.x(), end.y() - dd), end);
    QPainterPathStroker stroker;
    setPath(stroker.createStroke(path));
}

//!
//! create a single line from bottom to top on the left side
//!
void MENodeLink::makeBottomTopLeftCurve(QPointF start, QPointF end, QPointF nstart, QPointF nend)
{
    qreal dd = 20;
    qreal yy = (nstart.y() - nend.y()) * 0.5;
    qreal xx = (start.x() - end.x() * 0.2);
    QPointF mid(nend.x() - xx - dd, nend.y() + yy);

    QPainterPath path(start);
    path.cubicTo(QPointF(start.x(), start.y() + dd), QPointF(mid.x(), start.y() + dd), mid);
    path.cubicTo(QPointF(mid.x(), nend.y() - dd), QPointF(end.x(), end.y() - dd), end);
    QPainterPathStroker stroker;
    setPath(stroker.createStroke(path));
}

//!
//! create a single line from bottom to top on the right side
//!
void MENodeLink::makeBottomTopRightCurve(QPointF start, QPointF end, QPointF nstart, QPointF nend)
{
    qreal dd = 20;
    qreal yy = (nstart.y() - nend.y()) * 0.5;
    qreal xx = (end.x() - nend.x());
    QPointF mid(nend.x() - xx - dd, nend.y() + yy);

    QPainterPath path(start);
    path.cubicTo(QPointF(start.x(), start.y() + xx), QPointF(nend.x() - dd, start.y() + xx), mid);
    path.cubicTo(QPointF(nend.x() - dd, nend.y() - xx), QPointF(end.x(), end.y() - xx), end);
    QPainterPathStroker stroker;
    setPath(stroker.createStroke(path));
}

//!
//! handle context menu requested
//!
void MENodeLink::contextMenuEvent(QGraphicsSceneContextMenuEvent *e)
{
    MEGraphicsView::instance()->deletePossibleLink(this, e);
}

//!
//! handle mouse double click events
//!
void MENodeLink::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *)
{
    removeLines();
}

//!
//! handle mouse press events
//!
void MENodeLink::mousePressEvent(QGraphicsSceneMouseEvent *e)
{
    if (e->modifiers() == Qt::ControlModifier)
        removeLines();
}

//!
//!  handle hover enter events
//!
void MENodeLink::hoverEnterEvent(QGraphicsSceneHoverEvent *)
{
    highlightLines(true);
}

//!
//!  handle hover leave events
//!
void MENodeLink::hoverLeaveEvent(QGraphicsSceneHoverEvent *)
{
    highlightLines(false);
}
