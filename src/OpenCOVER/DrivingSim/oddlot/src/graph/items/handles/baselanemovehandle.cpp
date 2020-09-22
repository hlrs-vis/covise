/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.06.2010
**
**************************************************************************/

#include "baselanemovehandle.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/editors/laneeditor.hpp"
#include "src/graph/items/handles/texthandle.hpp"
#include "src/graph/items/handles/editablehandle.hpp"

//################//
// CONSTRUCTOR    //
//################//
BaseLaneMoveHandle::BaseLaneMoveHandle(LaneEditor *laneEditor, QGraphicsItem *parent)
		: MoveHandle(parent)
	{
		// Editor //
		//
		laneEditor_ = laneEditor;


		// ContextMenu //
		//
		removeAction_ = getContextMenu()->addAction("Remove Corner");
		smoothAction_ = getContextMenu()->addAction("Smooth");
		cornerAction_ = getContextMenu()->addAction("Corner");

		connect(removeAction_, SIGNAL(triggered()), this, SLOT(removeCorner()));
		connect(smoothAction_, SIGNAL(triggered()), this, SLOT(smoothCorner()));
		connect(cornerAction_, SIGNAL(triggered()), this, SLOT(corner()));

		// Text //
		//

		widthItem_ = new EditableHandle(0.0, this, true);
		widthItem_->setVisible(false); 

	}

BaseLaneMoveHandle::~BaseLaneMoveHandle()
	{}

void 
BaseLaneMoveHandle::updateWidthItemValue()
{
	widthItem_->setValue(getWidth());
}

//################//
// EVENTS         //
//################//

void
BaseLaneMoveHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::LeftButton)
	{
/*		ODD::ToolId tool = laneEditor_->getCurrentTool();
		if (tool == ODD::TLE_MODIFY_POLY) */
		{
			pressPos_ = scenePos();
		}
	}

	MoveHandle::mousePressEvent(event);
}

void
BaseLaneMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	// Let the LaneBorderEditor handle the movement //
	//
	if (event->modifiers() & Qt::ControlModifier)
	{
		QPointF mousePos = QPointF(floor(event->scenePos().x() + 0.5), floor(event->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
		laneEditor_->translateLaneBorder(scenePos(), mousePos);
	}
	else
	{
		QPointF newPos = event->scenePos();
		laneEditor_->translateLaneBorder(pressPos_, newPos);
		pressPos_ = newPos;
	}

	widthItem_->setValue(getWidth());

	MoveHandle::mouseMoveEvent(event); // pass to baseclass
}

void
BaseLaneMoveHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

	setFocus();

	widthItem_->setValue(getWidth());
	widthItem_->setVisible(true);

	widthItem_->setPos(mapFromScene(pos()) + QPointF(boundingRect().width()/2, widthItem_->boundingRect().height()/2)); 


	// Parent //
	//
//	MoveHandle::hoverEnterEvent(event); // pass to baseclass
}

void
BaseLaneMoveHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{


	// Text //
	//
	if (!isSelected())
	{
		widthItem_->setVisible(false);
	}

	// Parent //
	//
	MoveHandle::hoverLeaveEvent(event); // pass to baseclass
}


void
BaseLaneMoveHandle::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
	laneEditor_->getProjectGraph()->postponeGarbageDisposal();
	getContextMenu()->exec(event->screenPos());
	laneEditor_->getProjectGraph()->finishGarbageDisposal();
}

QVariant
BaseLaneMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
	// NOTE: position is relative to parent!!! //
	//

	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		if (isSelected())
		{
			widthItem_->setVisible(true);
		}
		else
		{
			widthItem_->setVisible(false);
		} 
		return value;
	}

	return QGraphicsItem::itemChange(change, value);
}



