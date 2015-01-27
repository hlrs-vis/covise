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

#include "roadtextitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "roaditem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

#include "src/graph/graphview.hpp"

//################//
// CONSTRUCTOR    //
//################//

RoadTextItem::RoadTextItem(RoadItem *roadItem)
    : GraphElement(roadItem, roadItem->getRoad())
    , roadItem_(roadItem)
    , road_(roadItem->getRoad())
{
    // Text //
    //
    textHandle_ = new TextHandle(road_->getName(), this);
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

RoadTextItem::~RoadTextItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
RoadTextItem::createPath()
{
    QPainterPath thePath;

    // Line to road //
    //
    thePath.moveTo(0.0, 0.0);
    thePath.lineTo(mapFromScene(road_->getGlobalPoint(1.0)));

    //	thePath.addPath(getProjectGraph()->getView()->viewportTransform().inverted().map(textHandle_->deviceTransform(getProjectGraph()->getView()->viewportTransform()).map(textHandle_->path())));

    setPath(thePath);
}

void
RoadTextItem::updatePosition()
{
    /*	setPos(road_->getGlobalPoint(1.0, 10.0));
	createPath();*/
}

void
RoadTextItem::updateName()
{
    textHandle_->setText(road_->getName());
}

//################//
// SLOTS          //
//################//

void
RoadTextItem::handlePositionChange(const QPointF &dpos)
{
    setPos(scenePos() + dpos);
    createPath();
}

//void
//	RoadTextItem
//	::handleSelectionChange(bool selected)
//{
////	setSelected(selected);
//}

QPainterPath
RoadTextItem::shape() const
{
    return mapFromItem(textHandle_, textHandle_->path());
}

//##################//
// Observer Pattern //
//##################//

void
RoadTextItem::updateObserver()
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
    int changes = road_->getRoadChanges();

    if ((changes & RSystemElementRoad::CRD_LengthChange)
        || (changes & RSystemElementRoad::CRD_ShapeChange)
        || (changes & RSystemElementRoad::CRD_TrackSectionChange))
    {
        updatePosition();
    }

    // RSystemElement //
    //
    changes = road_->getRSystemElementChanges();

    if (changes & RSystemElement::CRE_NameChange)
    {
        updateName();
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
RoadTextItem::itemChange(GraphicsItemChange change, const QVariant &value)
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
RoadTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Not the Label is selected, but the road beneath
    //
    getRoadItem()->mousePressEvent(event);
}
