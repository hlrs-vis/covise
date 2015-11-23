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

#include "objecttextitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "objectitem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

#include "src/graph/graphview.hpp"

//################//
// CONSTRUCTOR    //
//################//

ObjectTextItem::ObjectTextItem(ObjectItem *objectItem)
    : GraphElement(objectItem, objectItem->getObject())
    , objectItem_(objectItem)
    , object_(objectItem->getObject())
{
    // Text //
    //
    textHandle_ = new TextHandle(object_->getName(), this);
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
	updateName();

    // Hide the text item on creation and show it only on mouse hover of the parent //
    //
    setVisible(false);
}

ObjectTextItem::~ObjectTextItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
ObjectTextItem::createPath()
{
    QPainterPath thePath;

    // Line to road //
    //
    thePath.moveTo(0.0, 0.0);
    thePath.lineTo(mapFromScene(object_->getParentRoad()->getGlobalPoint(object_->getSStart(), object_->getT())));

    //	thePath.addPath(getProjectGraph()->getView()->viewportTransform().inverted().map(textHandle_->deviceTransform(getProjectGraph()->getView()->viewportTransform()).map(textHandle_->path())));

    setPath(thePath);
}

void
ObjectTextItem::updatePosition()
{
    setPos(object_->getParentRoad()->getGlobalPoint(object_->getSStart(), object_->getT()));
    createPath();
}

void
ObjectTextItem::updateName()
{
	if ((object_->getName() == "") || (object_->getName() == "unnamed"))
	{
		textHandle_->setText(object_->getType());
	}
	else
	{
		textHandle_->setText(object_->getName());
	}
}

//################//
// SLOTS          //
//################//

void
ObjectTextItem::handlePositionChange(const QPointF &dpos)
{
    setPos(scenePos() + dpos);
    createPath();
}

//void
//	ObjectTextItem
//	::handleSelectionChange(bool selected)
//{
////	setSelected(selected);
//}

QPainterPath
ObjectTextItem::shape() const
{
    return mapFromItem(textHandle_, textHandle_->path());
}

//##################//
// Observer Pattern //
//##################//

void
ObjectTextItem::updateObserver()
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
    int changes = object_->getObjectChanges();

    if (changes & Object::CEL_ParameterChange)
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
ObjectTextItem::itemChange(GraphicsItemChange change, const QVariant &value)
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
ObjectTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Not the Label is selected, but the road beneath
    //
    getObjectItem()->mousePressEvent(event);
}
