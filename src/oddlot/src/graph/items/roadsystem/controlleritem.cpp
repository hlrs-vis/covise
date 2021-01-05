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
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

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
    setAcceptHoverEvents(true);
    setSelectable();

    // Signal Editor
    //
    signalEditor_ = dynamic_cast<SignalEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

    // Text //
    //
    textItem_ = new QGraphicsTextItem(controller_->getID().speakingName(), this);
    textItem_->setFlag(QGraphicsItem::ItemIsSelectable, false);
    QTransform trafo;
    trafo.rotate(180, Qt::XAxis);
    textItem_->setTransform(trafo);
    textItem_->setZValue(1.0);

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
    foreach(Signal * signal, controller_->getSignals())
    {
        RSystemElementRoad *road = signal->getParentRoad();
        points.append(road->getGlobalPoint(signal->getSStart()));
    }


    QTransform trafo;
    trafo.rotate(180, Qt::XAxis);
    // Text //
    //
    if (points.count() > 0)
    {
        textItem_->setPos(points[0]);
        QRectF rect = textItem_->boundingRect();
        trafo.mapRect(rect);
        rect.translate(points[0] - rect.bottomLeft());
        thePath.addRect(rect);
        thePath.addPolygon(QPolygonF(points));
    }
    else
    {

        textItem_->setPos(getTopviewGraph()->getScene()->sceneRect().center());
        thePath.addRect(trafo.mapRect(textItem_->boundingRect()));
        setPath(thePath);

    }
    setPath(thePath);
}

//################//
// SLOTS          //
//################//

bool
ControllerItem::removeController()
{
    RemoveControllerCommand *command = new RemoveControllerCommand(controller_, roadSystemItem_->getRoadSystem());
    return getProjectGraph()->executeCommand(command);
}

void
ControllerItem::addToCurrentTile()
{
    odrID newId = controller_->getID();
    newId.setTileID(getProjectData()->getTileSystem()->getCurrentTile()->getID());
    SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(controller_->getRoadSystem(), controller_, newId, NULL);
    getProjectGraph()->executeCommand(command);
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
