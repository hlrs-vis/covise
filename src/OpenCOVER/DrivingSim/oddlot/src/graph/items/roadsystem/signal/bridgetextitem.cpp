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

#include "bridgetextitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "bridgeitem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

#include "src/graph/graphview.hpp"

//################//
// CONSTRUCTOR    //
//################//

BridgeTextItem::BridgeTextItem(BridgeItem *bridgeItem)
    : GraphElement(bridgeItem, bridgeItem->getBridge())
    , bridgeItem_(bridgeItem)
    , bridge_(bridgeItem->getBridge())
{
    // Text //
    //
    textHandle_ = new TextHandle(bridge_->getName(), this);
    textHandle_->setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
    textHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    textHandle_->setFlag(QGraphicsItem::ItemIgnoresParentOpacity, false); // use highlighting of the road
    //	textHandle_->setFlag(QGraphicsItem::ItemIsSelectable, true);
    //	connect(textHandle_, SIGNAL(requestPositionChange(QPointF)), this, SLOT(handlePositionChange(QPointF)));
    //	connect(textHandle_, SIGNAL(selectionChanged(bool)), this, SLOT(handleSelectionChange(bool)));

    // Flags //
    //
    //	setSelectable();
    //	setFlag(QGraphicsItem::ItemSendsScenePositionChanges, true);
    setFlag(QGraphicsItem::ItemIgnoresParentOpacity, false); // use highlighting of the road

    // Path //
    //
    updatePosition();

    // Hide the text item on creation and show it only on mouse hover of the parent //
    //
    setVisible(false);
}

BridgeTextItem::~BridgeTextItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
BridgeTextItem::createPath()
{
    QPainterPath thePath;

    // Line to road //
    //
    thePath.moveTo(0.0, 0.0);
    thePath.lineTo(mapFromScene(bridge_->getParentRoad()->getGlobalPoint(bridge_->getSStart(), 0.0)));

    //	thePath.addPath(getProjectGraph()->getView()->viewportTransform().inverted().map(textHandle_->deviceTransform(getProjectGraph()->getView()->viewportTransform()).map(textHandle_->path())));

    setPath(thePath);
}

void
BridgeTextItem::updatePosition()
{
    setPos(bridge_->getParentRoad()->getGlobalPoint(bridge_->getSStart(), 0.0));
    createPath();
}

void
BridgeTextItem::updateName()
{
    textHandle_->setText(bridge_->getName());
}

//################//
// SLOTS          //
//################//

void
BridgeTextItem::handlePositionChange(const QPointF &dpos)
{
    setPos(scenePos() + dpos);
    createPath();
}

//void
//	BridgeTextItem
//	::handleSelectionChange(bool selected)
//{
////	setSelected(selected);
//}

QPainterPath
BridgeTextItem::shape() const
{
    return mapFromItem(textHandle_, textHandle_->path());
}

//##################//
// Observer Pattern //
//##################//

void
BridgeTextItem::updateObserver()
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
    int changes = bridge_->getBridgeChanges();

    if (changes & Bridge::CEL_ParameterChange)
    {
        updatePosition();
        updateName();
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
BridgeTextItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    if (change == QGraphicsItem::ItemScenePositionHasChanged)
    {
        createPath();
    }

    return GraphElement::itemChange(change, value);
}

void
BridgeTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Not the Label is selected, but the road beneath
    //
    getBridgeItem()->mousePressEvent(event);
}
