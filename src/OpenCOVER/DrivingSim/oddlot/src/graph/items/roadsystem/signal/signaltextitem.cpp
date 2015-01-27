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

#include "signaltextitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "signalitem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

#include "src/graph/graphview.hpp"

//################//
// CONSTRUCTOR    //
//################//

SignalTextItem::SignalTextItem(SignalItem *signalItem)
    : GraphElement(signalItem, signalItem->getSignal())
    , signalItem_(signalItem)
    , signal_(signalItem->getSignal())
{
    // Text //
    //
    textHandle_ = new TextHandle(signal_->getName(), this);
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

SignalTextItem::~SignalTextItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
SignalTextItem::createPath()
{
    QPainterPath thePath;

    // Line to road //
    //
    thePath.moveTo(0.0, 0.0);
    thePath.lineTo(mapFromScene(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), signal_->getT())));

    //	thePath.addPath(getProjectGraph()->getView()->viewportTransform().inverted().map(textHandle_->deviceTransform(getProjectGraph()->getView()->viewportTransform()).map(textHandle_->path())));

    setPath(thePath);
}

void
SignalTextItem::updatePosition()
{
    setPos(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), signal_->getT()));
    createPath();
}

void
SignalTextItem::updateName()
{
    textHandle_->setText(signal_->getName());
}

//################//
// SLOTS          //
//################//

void
SignalTextItem::handlePositionChange(const QPointF &dpos)
{
    setPos(scenePos() + dpos);
    createPath();
}

//void
//	SignalTextItem
//	::handleSelectionChange(bool selected)
//{
////	setSelected(selected);
//}

QPainterPath
SignalTextItem::shape() const
{
    return mapFromItem(textHandle_, textHandle_->path());
}

//##################//
// Observer Pattern //
//##################//

void
SignalTextItem::updateObserver()
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
    int changes = signal_->getSignalChanges();

    if (changes & Signal::CEL_ParameterChange)
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
SignalTextItem::itemChange(GraphicsItemChange change, const QVariant &value)
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
SignalTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Not the Label is selected, but the road beneath
    //
    getSignalItem()->mousePressEvent(event);
}
