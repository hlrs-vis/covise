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

#include "junctionitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/junctioncommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
//#include "src/data/commands/

// Graph //
//
#include "roadsystemitem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QAction>

// Utils //
//
#include "src/util/colorpalette.hpp"

JunctionItem::JunctionItem(RoadSystemItem *roadSystemItem, RSystemElementJunction *junction)
    : GraphElement(roadSystemItem, junction)
    , junction_(junction)
    , roadSystemItem_(roadSystemItem)
{
    init();
}

JunctionItem::~JunctionItem()
{
    foreach (RSystemElementRoad *path, paths_)
    {
        path->detachObserver(this);
    }
}

void
JunctionItem::init()
{
    // Selection/Hovering //
    //
    //	setAcceptHoverEvents(true);
    setSelectable();

    // Text //
    //
    textHandle_ = new TextHandle(junction_->getID(), this);
    textHandle_->setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
    textHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    textHandle_->setFlag(QGraphicsItem::ItemIgnoresParentOpacity, false); // use highlighting of the road
    textHandle_->setFlag(QGraphicsItem::ItemIsMovable, false);
    textHandle_->setZValue(1.0);

    setBrush(ODD::instance()->colors()->brightGrey());
    setPen(ODD::instance()->colors()->darkGrey());

    // Path //
    //
    updatePathList();

    // ContextMenu //
    //
    QAction *removeAction = getRemoveMenu()->addAction(tr("Junction"));
    connect(removeAction, SIGNAL(triggered()), this, SLOT(removeJunction()));

    QAction *hideAction = getHideMenu()->addAction(tr("Junction"));
    connect(hideAction, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

    // Context Menu
    //
    QAction *addToTileAction = getContextMenu()->addAction("Add to current tile");
    connect(addToTileAction, SIGNAL(triggered()), this, SLOT(addToCurrentTile()));
}

void
JunctionItem::updatePathList()
{
    QList<RSystemElementRoad *> newPaths;

    foreach (JunctionConnection *connection, junction_->getConnections())
    {
        RSystemElementRoad *path = junction_->getRoadSystem()->getRoad(connection->getConnectingRoad());

        if (path)
        {
            newPaths.append(path);
            if (!paths_.contains(path))
            {
                path->attachObserver(this);
            }
        }
        else
        {
            junction_->delConnection(connection);
        }
    }

    foreach (RSystemElementRoad *path, paths_)
    {
        if (!newPaths.contains(path))
        {
            path->detachObserver(this);
        }
    }

    paths_ = newPaths;

    // Redraw //
    //
    updatePath();
}

void
JunctionItem::updatePath()
{
    QPainterPath thePath;

    if (paths_.size() < 2)
    {
        return;
    }

    QPointF firstPoint = paths_.at(0)->getGlobalPoint(0.0);
    double minX = firstPoint.x();
    double maxX = minX;
    double minY = firstPoint.y();
    double maxY = minY;

    foreach (RSystemElementRoad *path, paths_)
    {
        QPointF startPoint = path->getGlobalPoint(0.0);
        if (startPoint.x() < minX)
            minX = startPoint.x();
        if (startPoint.x() > maxX)
            maxX = startPoint.x();
        if (startPoint.y() < minY)
            minY = startPoint.y();
        if (startPoint.y() > maxY)
            maxY = startPoint.y();

        QPointF endPoint = path->getGlobalPoint(path->getLength());
        if (endPoint.x() < minX)
            minX = endPoint.x();
        if (endPoint.x() > maxX)
            maxX = endPoint.x();
        if (endPoint.y() < minY)
            minY = endPoint.y();
        if (endPoint.y() > maxY)
            maxY = endPoint.y();
    }

    thePath.addRect(minX - 5.0, minY - 5.0, maxX - minX + 10.0, maxY - minY + 10.0);
    setPath(thePath);

    // Text //
    //
    textHandle_->setPos(minX - 5.0, maxY + 5.0);
}

//################//
// SLOTS          //
//################//

bool
JunctionItem::removeJunction()
{
    RemoveJunctionCommand *command = new RemoveJunctionCommand(junction_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
JunctionItem::addToCurrentTile()
{
    QStringList parts = junction_->getID().split("_");
    if (parts.at(0) != getProjectData()->getTileSystem()->getCurrentTile()->getID())
    {
        QString name = junction_->getName();
        QString newId = junction_->getRoadSystem()->getUniqueId("", name);
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(junction_->getRoadSystem(), junction_, newId, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

//##################//
// Observer Pattern //
//##################//

void
JunctionItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    bool doUpdate = false;

    // Get change flags //
    //
    int changes = junction_->getJunctionChanges();
    if ((changes & RSystemElementJunction::CJN_ConnectionChanged)
        || (changes & RSystemElementRoad::CRD_JunctionChange))
    {
        doUpdate = true;
    }

    // TODO: ID CHANGE

    // Paths //
    //
    foreach (RSystemElementRoad *path, paths_)
    {
        int changes = path->getRoadChanges();
        if ((changes & RSystemElementRoad::CRD_LengthChange)
            || (changes & RSystemElementRoad::CRD_ShapeChange)
            || (changes & RSystemElementRoad::CRD_TrackSectionChange))
        {
            doUpdate = true;
            break;
        }
    }

    if (doUpdate)
    {
        updatePathList();
    }
}

//*************//
// Delete Item
//*************//

bool
JunctionItem::deleteRequest()
{
    return removeJunction();
}
