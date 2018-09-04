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
#include "src/graph/editors/laneeditor.hpp"
#include "src/graph/items/handles/texthandle.hpp"


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

		widthTextItem_ = new TextHandle("", this);
		widthTextItem_->setZValue(1.0); // stack before siblings
		widthTextItem_->setVisible(false);

	}

BaseLaneMoveHandle::~BaseLaneMoveHandle()
	{}



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

	MoveHandle::mouseMoveEvent(event); // pass to baseclass
}

void
BaseLaneMoveHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

	setFocus();

	// Text //
	//
	QString text;
	LaneWidth *highSlot = dynamic_cast<LaneMoveHandle<LaneWidth, LaneWidth> *>(this)->getHighSlot();
	if (highSlot)
	{
		text = QString("%1").arg(highSlot->f(0.0), 0, 'f', 2);
	}
	else
	{
		LaneWidth *lowSlot = dynamic_cast<LaneMoveHandle<LaneWidth, LaneWidth> *>(this)->getLowSlot();
		if (lowSlot)
		{
			text = QString("%1").arg(lowSlot->f(lowSlot->getSSectionEnd() - lowSlot->getSSectionStartAbs()), 0, 'f', 2);
		}
	}

	widthTextItem_->setText(text);
	widthTextItem_->setVisible(true);
	widthTextItem_->setPos(mapFromScene(scenePos()));

	// Parent //
	//
//	MoveHandle::hoverEnterEvent(event); // pass to baseclass
}

void
BaseLaneMoveHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{


	// Text //
	//
	widthTextItem_->setVisible(false);

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


//################//
// SLOTS          //
//################//

void
BaseLaneMoveHandle::removeCorner()
{
	dynamic_cast<LaneMoveHandle<LaneWidth, LaneWidth> *>(this)->removeCorner();
}

void
BaseLaneMoveHandle::smoothCorner()
{
	dynamic_cast<LaneMoveHandle<LaneWidth, LaneWidth> *>(this)->smoothCorner();
}

void
BaseLaneMoveHandle::corner()
{
	dynamic_cast<LaneMoveHandle<LaneWidth, LaneWidth> *>(this)->corner();
}


