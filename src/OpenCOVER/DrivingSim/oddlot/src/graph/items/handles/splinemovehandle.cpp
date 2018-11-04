/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   20.04.2010
**
**************************************************************************/

#include "splinemovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphview.hpp"
#include "src/graph/editors/shapeeditor.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitem.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitems.hpp"
#include "src/graph/items/handles/texthandle.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneMouseEvent>
#include<QGraphicsTextItem>

// Utils //
//
#include "math.h"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

SplineMoveHandle::SplineMoveHandle(ShapeEditor *shapeEditor, ShapeSectionPolynomialItem *parent, SplineControlPoint *corner)
    : MoveHandle(parent)
	, corner_(corner)
	, parentPolynomialItem_(parent)
	, textHandle_(NULL)
{
    // Editor //
    //
    shapeEditor_ = shapeEditor;
	parentPolynomialItems_ = parentPolynomialItem_->getParentPolynomialItems();

	init();
}

SplineMoveHandle::~SplineMoveHandle()
{
	lateralSection_->detachObserver(this);
}

//################//
// FUNCTIONS      //
//################//

void
SplineMoveHandle::init()
{
	lateralSection_ = corner_->getParent();
	lateralSection_->attachObserver(this);
	shapeSection_ = lateralSection_->getParentSection();

	if ((shapeSection_->getFirstPolynomialLateralSection()->getRealPointLow() != corner_) 
	&& (shapeSection_->getLastPolynomialLateralSection()->getRealPointHigh() != corner_))
	{
		// ContextMenu //
		//

		deleteAction_ = getContextMenu()->addAction("Delete point");
		connect(deleteAction_, SIGNAL(triggered()), this, SLOT(deleteCorner()));

	}

	QString text = QString("%1,%2").arg(corner_->getPoint().x(), 0, 'f', 2).arg(corner_->getPoint().y(), 0, 'f', 2);
	textHandle_ = new QGraphicsTextItem(text, this);
	QTransform trafo;
	trafo.rotate(180, Qt::XAxis);
	textHandle_->setTransform(trafo);
	textHandle_->setVisible(false);

	// Color //
	//
	updateColor();
	setPos(corner_->getPoint());
}

void
SplineMoveHandle::updateColor()
{
	setBrush(QBrush(ODD::instance()->colors()->brightRed()));
	setPen(QPen(ODD::instance()->colors()->darkRed()));
}


//################//
// SLOTS          //
//################//

void
SplineMoveHandle::deleteCorner()
{
	// Command: Merge ShapeSection Polynomials
	shapeEditor_->deleteLateralSection(corner_);
}


//################//
// EVENTS         //
//################//

void 
SplineMoveHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	// handled by ShapeEditor
}

void 
SplineMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	const double snapDistance = 0.1;

	// handled by ShapeEditor
	//
	double y = event->scenePos().y();
	if ((y < 0) && (y > -snapDistance))
	{
		y = 0;
	}

	if (shapeSection_->getFirstPolynomialLateralSection()->getRealPointLow() == corner_)
	{
		double width = shapeSection_->getParentRoad()->getMinWidth(shapeSection_->getSStart());
		if (abs(event->scenePos().x() - width) < snapDistance)
		{
			shapeEditor_->translateMoveHandles(QPointF(width, y), corner_);
			return;
		}

	}
	else if (shapeSection_->getLastPolynomialLateralSection()->getRealPointHigh() == corner_)
	{
		double width = shapeSection_->getParentRoad()->getMaxWidth(shapeSection_->getSStart());
		if (abs(event->scenePos().x() - width) < snapDistance)
		{
			shapeEditor_->translateMoveHandles(QPointF(width, y), corner_);
			return;
		}

	}
	shapeEditor_->translateMoveHandles(QPointF(event->scenePos().x(), y), corner_);
}

void
SplineMoveHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	shapeEditor_->fitBoundingBoxInView();
}

void
SplineMoveHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	if (textHandle_)
	{
		textHandle_->setVisible(true);
	}
	// handled by ShapeEditor
}

void
SplineMoveHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	if (textHandle_)
	{
		textHandle_->setVisible(false);
	}
	// handled by ShapeEditor
}

void
SplineMoveHandle::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{

	shapeEditor_->getProfileGraph()->postponeGarbageDisposal();
	getContextMenu()->exec(event->screenPos());
	shapeEditor_->getProfileGraph()->finishGarbageDisposal();

}


//################//
// OBSERVER       //
//################//

/*!
*
* Does not know if highSlot or lowSlot has changed.
*/
void
SplineMoveHandle::updateObserver()
{

	MoveHandle::updateObserver();

    int changes = 0x0;

	// PolynomialLateralSectionChanges //
	//
	changes = lateralSection_->getPolynomialLateralSectionChanges();

	// Parameter Changed //
	//
	if (changes & PolynomialLateralSection::CPL_ParameterChange)
	{
		setPos(corner_->getPoint());
		updateColor();
		if (textHandle_)
		{
			QString text = QString("%1,%2").arg(corner_->getPoint().x(), 0, 'f', 2).arg(corner_->getPoint().y(), 0, 'f', 2);
			textHandle_->setPlainText(text);
		}
	}

	// LateralSectionChange //
	//
	// Parameter Changed //
	//
	changes = lateralSection_->getLateralSectionChanges();
	if ((changes & LateralSection::CLS_LengthChange) || (changes & LateralSection::CLS_TChange))
	{
		setPos(corner_->getPoint());
		updateColor();
	}

}



