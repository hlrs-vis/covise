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

#include "osctextitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "oscitem.hpp"
#include "src/graph/items/handles/texthandle.hpp"

#include "src/graph/graphview.hpp"

// OpenScenario //
//
#include "oscObject.h"
#include "oscMemberValue.h"

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

OSCTextItem::OSCTextItem(GraphElement *item, OpenScenario::oscObject *oscObject, const QPointF &pos)
    : GraphElement(item, NULL)
    , oscObject_(oscObject)
	, pos_(pos)
{
    // Text //
    //
	updateName();
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
	updateName();
    updatePosition();

    // Hide the text item on creation and show it only on mouse hover of the parent //
    //
    setVisible(false);
}

OSCTextItem::~OSCTextItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
OSCTextItem::createPath()
{
    QPainterPath thePath;

    // Line to road //
    //
   // thePath.moveTo(0.0, 0.0);
   // thePath.lineTo(pos_);

    //	thePath.addPath(getProjectGraph()->getView()->viewportTransform().inverted().map(textHandle_->deviceTransform(getProjectGraph()->getView()->viewportTransform()).map(textHandle_->path())));

    setPath(thePath);
}

void
OSCTextItem::updatePosition()
{
 //   setPos(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), signal_->getT()));
    createPath();
}

void
OSCTextItem::updateName()
{
	// Text //
    //
	OpenScenario::oscMemberValue *value =  oscObject_->getMember("name")->getValue();
	oscStringValue *sv = dynamic_cast<oscStringValue *>(value);
	if (sv)
	{
//		textHandle_ = new TextHandle(QString::fromStdString(sv->getValue()), this);
		textHandle_ = new TextHandle(QString::fromStdString("my text"), this);
	}
	else
	{
		textHandle_ = new TextHandle("", this);
	}
}

//################//
// SLOTS          //
//################//

void
OSCTextItem::handlePositionChange(const QPointF &dpos)
{
    setPos(scenePos() + dpos);
    createPath();
}

//void
//	OSCTextItem
//	::handleSelectionChange(bool selected)
//{
////	setSelected(selected);
//}

QPainterPath
OSCTextItem::shape() const
{
    return mapFromItem(textHandle_, textHandle_->path());
}

//##################//
// Observer Pattern //
//##################//

void
OSCTextItem::updateObserver()
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
 /*   int changes = signal_->getSignalChanges();

    if (changes & Signal::CEL_ParameterChange)
    {
        updatePosition();
        updateName();
    }*/
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
OSCTextItem::itemChange(GraphicsItemChange change, const QVariant &value)
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
OSCTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // Not the Label is selected, but the road beneath
    //
 //   getSignalItem()->mousePressEvent(event);
}
