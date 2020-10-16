/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/24/2010
**
**************************************************************************/

#include "editablehandle.hpp"


// Graph //
//
#include "src/graph/editors/projecteditor.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

#include<QGraphicsScene>
#include<QDebug>
#include<QRectF>

//################//
// CONSTRUCTOR    //
//################//

EditableHandle::EditableHandle(double value, BaseLaneMoveHandle *parent, bool flip)
	: QGraphicsProxyWidget(parent)
	, Observer()
{

	// Flags //
	//
	setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
	setFlag(QGraphicsItem::ItemIgnoresParentOpacity, true);
	setAcceptHoverEvents(true);

	// Transformation //
	//
	// Note: The y-Axis flip is done here, because the item ignores
	// the view's transformation
	if (flip)
	{
		QTransform trafo;
		trafo.rotate(180, Qt::XAxis);
		setTransform(trafo);
	} 

	// ZValue //
	//
	setZValue(1.0); // stack before siblings

	editableItem_ = new QDoubleSpinBox();
	editableItem_->setButtonSymbols(QAbstractSpinBox::NoButtons);
	editableItem_->setMinimum(0.0);
	editableItem_->setMaximum(99.99);
	editableItem_->setStyleSheet("QDoubleSpinBox { background-color : white; color : black; }");
	connect(editableItem_, SIGNAL(valueChanged(double)), parent, SLOT(setLaneWidth(double)));
	setWidget(editableItem_);

}


EditableHandle::~EditableHandle()
{
}

//################//
// FUNCTIONS      //
//################//

double
EditableHandle::getValue() const
{
    return editableItem_->value();
}

void
EditableHandle::setValue(double value)
{
	editableItem_->blockSignals(true);
	editableItem_->setValue(value);
	editableItem_->blockSignals(false);
}


