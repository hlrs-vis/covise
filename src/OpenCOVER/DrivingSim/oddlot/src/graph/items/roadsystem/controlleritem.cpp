/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/25/2010
**
**************************************************************************/

#include "controlleritem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/controllercommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"

//#include "src/data/commands/

// Graph //
//
#include "roadsystemitem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Editor //
//
#include "src/graph/editors/signaleditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QAction>

// Utils //
//
#include "src/util/colorpalette.hpp"

ControllerItem::ControllerItem(RoadSystemItem *roadSystemItem, RSystemElementController *controller)
    : GraphElement(roadSystemItem, controller)
    , controller_(controller)
    , roadSystemItem_(roadSystemItem)
{
    init();
}

ControllerItem::~ControllerItem()
{
}

void
ControllerItem::init()
{
    // Selection/Hovering //
    //
    //	setAcceptHoverEvents(true);
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();

    // Signal Editor
    //
    signalEditor_ = dynamic_cast<SignalEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

    // Text //
    //
    textHandle_ = new TextHandle(controller_->getID(), this);
    textHandle_->setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
    textHandle_->setPen(QPen(ODD::instance()->colors()->darkOrange()));
    textHandle_->setFlag(QGraphicsItem::ItemIgnoresParentOpacity, false); // use highlighting of the road
    textHandle_->setFlag(QGraphicsItem::ItemIsMovable, false);
    textHandle_->setZValue(1.0);

    setBrush(ODD::instance()->colors()->brightOrange());
    setPen(ODD::instance()->colors()->darkOrange());

    updatePath();

    // ContextMenu //
    //
    QAction *removeAction = getRemoveMenu()->addAction(tr("Controller"));
    connect(removeAction, SIGNAL(triggered()), this, SLOT(removeController()));

    QAction *hideAction = getHideMenu()->addAction(tr("Controller"));
    connect(hideAction, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

    // Context Menu
    //
    QAction *addToTileAction = getContextMenu()->addAction("Add to current tile");
    connect(addToTileAction, SIGNAL(triggered()), this, SLOT(addToCurrentTile()));

}

void
ControllerItem::updatePath()
{
    QPainterPath thePath;
    QVector<QPointF> points;
    foreach (Signal *signal, controller_->getSignals())
    {
        RSystemElementRoad * road = signal->getParentRoad();
        points.append(road->getGlobalPoint(signal->getSStart()));
    }

    thePath.addPolygon(QPolygonF(points));
    setPath(thePath);

    // Text //
    //
    if (points.count() > 0)
    {
        textHandle_->setPos(points[0]);
    }
}

//################//
// SLOTS          //
//################//

bool
ControllerItem::removeController()
{
    RemoveControllerCommand *command = new RemoveControllerCommand(controller_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
ControllerItem::addToCurrentTile()
{
    QStringList parts = controller_->getID().split("_");
    if (parts.at(0) != getProjectData()->getTileSystem()->getCurrentTile()->getID())
    {
        QString name = controller_->getName();
        QString newId = controller_->getRoadSystem()->getUniqueId("", name);
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(controller_->getRoadSystem(), controller_, newId, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

//################//
// EVENTS         //
//################//

void
ControllerItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

    // Text //
    //
   // textHandle_->setVisible(true);
   // textHandle_->setPos(event->scenePos());

    // Parent //
    //
    //GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
ControllerItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{

    // Text //
    //
  //  textHandle_->setVisible(false);

    // Parent //
    //
    //GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
ControllerItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    //GraphElement::hoverMoveEvent(event);
}

void
ControllerItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = signalEditor_->getCurrentTool(); // Editor Delete Bridge
    if (tool == ODD::TSG_DEL)
    {
        removeController();
    }
    else
    {
        GraphElement::mousePressEvent(event); // pass to baseclass
    }
}

//##################//
// Observer Pattern //
//##################//

void
ControllerItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }


    // Get change flags //
    //
    int changes = controller_->getControllerChanges();
    if (changes & RSystemElementController::CRC_EntryChange)
    {
        updatePath();
    }

}

//*************//
// Delete Item
//*************//

bool
ControllerItem::deleteRequest()
{
    return removeController();
}
