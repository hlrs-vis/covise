/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#include "shapesectionpolynomialitems.hpp"

// Data //
//

#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitem.hpp"
#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/items/handles/splinemovehandle.hpp"

// Editor //
//
#include "src/graph/editors/shapeeditor.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"


// Qt //
//



//################//
// CONSTRUCTOR    //
//################//

ShapeSectionPolynomialItems::ShapeSectionPolynomialItems(ProfileGraph *profileGraph, ShapeSection *shapeSection)
    : GraphElement(NULL, shapeSection)
	, profileGraph_(profileGraph)
    , shapeSection_(shapeSection)
{
    // Init //
    //
    init();
}

ShapeSectionPolynomialItems::~ShapeSectionPolynomialItems()
{
}

void
ShapeSectionPolynomialItems::init()
{
	transform_ = new QTransform();
	transformInverted_ = new QTransform();
	shapeEditor_ = dynamic_cast<ShapeEditor *>(profileGraph_->getProjectWidget()->getProjectEditor());
	createPolynomialItems();

}

void 
ShapeSectionPolynomialItems::createPolynomialItems()
{
	splineCanvas_ = new QRectF(0, 0, 0, 0);
/*	foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
	{
		shape->getRealPointsFromParameters();
	} */

	if (shapeEditor_)
	{
		// Create all lateral polynoms and cotrol handles //
		//
		foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
		{
			ShapeSectionPolynomialItem *shapeItem = new ShapeSectionPolynomialItem(this, shape, shapeEditor_);
		}
	}

}

QRectF 
ShapeSectionPolynomialItems::boundingRect()
{
	double ymin = 1000;
	double ymax = -1000;
	
	foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
	{
		QPointF p0 = shape->getRealPointLow()->getPoint();
		QPointF p1 = shape->getSplineControlPointLow()->getPoint();
		QPointF p2 = shape->getSplineControlPointHigh()->getPoint();
		QPointF p3 = shape->getRealPointHigh()->getPoint();
		double y = fmin(fmin(p0.y(), p1.y()), fmin(p2.y(), p3.y()));
		if (y < ymin)
		{
			ymin = y;
		}
		y = fmax(fmax(p0.y(), p1.y()), fmax(p2.y(), p3.y()));
		if (y > ymax)
		{
			ymax = y;
		}
	}
	return QRectF { 0, ymax, shapeSection_->getParentRoad()->getMaxWidth(shapeSection_->getSStart()) - shapeSection_->getParentRoad()->getMinWidth(shapeSection_->getSStart()), ymax - ymin };
}

void
ShapeSectionPolynomialItems::initNormalization()
{
	QPointF end = splineCanvas_->topRight();
	QPointF start = splineCanvas_->bottomLeft();

	transform_->reset();
	double alpha = atan((end.y() - start.y()) / (end.x() - start.x())) * 180 / M_PI;
	transform_->rotate(45.0 - alpha);
	transform_->translate(-start.x(), -start.y());
	QPointF end1 = transform_->map(end);
	transform_->reset();
	transform_->scale(1 / end1.x(), 1 / end1.y());
	transform_->rotate(45.0 - alpha);
	transform_->translate(-start.x(), -start.y());
	
	*transformInverted_ = transform_->inverted();
}

QPointF 
ShapeSectionPolynomialItems::normalize(const QPointF &p)
{

	return transform_->map(p);
}

QPointF 
ShapeSectionPolynomialItems::sceneCoords(const QPointF &start, const QPointF &p)
{

	return transformInverted_->map(p);
}


//################//
// OBSERVER       //
//################//

void
ShapeSectionPolynomialItems::updateObserver()
{
    // Parent //
    //
	GraphElement::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ShapeSection //
    //
    int changes = shapeSection_->getShapeSectionChanges();

    if (changes & ShapeSection::CSS_ShapeSectionChange)
    {
		// A section has been added.
		//
		foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
		{
			if ((shape->getDataElementChanges() & DataElement::CDE_DataElementCreated)
				|| (shape->getDataElementChanges() & DataElement::CDE_DataElementAdded))
			{
				// Create all lateral polynoms and cotrol handles //
				//

				ShapeSectionPolynomialItem *shapeItem = new ShapeSectionPolynomialItem(this, shape, shapeEditor_);
			}


		}

    }
}



