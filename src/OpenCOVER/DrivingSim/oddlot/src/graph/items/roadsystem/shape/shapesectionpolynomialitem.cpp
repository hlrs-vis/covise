/* This file is part of COVISE->

   You can use it under the terms of the GNU Lesser General Public License
   version 2->1 or later, see lgpl-2->1->txt->

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele->de>
**   15->07->2010
**
**************************************************************************/

#include "shapesectionpolynomialitem.hpp"

#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/roadsystem/lateralsections/polynomiallateralsection.hpp"

#include "src/data/commands/shapesectioncommands.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphview.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitems.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionitem.hpp"
#include "src/graph/items/handles/splinemovehandle.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Qt //
//
#include <QCursor>
#include <QBrush>
#include <QPen>
#include <QContextMenuEvent>

// Utils //
//
#include "math.h"
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

ShapeSectionPolynomialItem::ShapeSectionPolynomialItem(ShapeSectionPolynomialItems *parentShapeSectionPolynomialItems, PolynomialLateralSection *polynomialLateralSection, ShapeEditor *shapeEditor)
    : LateralSectionItem(parentShapeSectionPolynomialItems, polynomialLateralSection)
    , parentShapeSectionPolynomialItems_(parentShapeSectionPolynomialItems)
    , polynomialLateralSection_(polynomialLateralSection)
	, shapeEditor_(shapeEditor)
	, realPointHighHandle_(NULL)
{

    init();
}

ShapeSectionPolynomialItem::~ShapeSectionPolynomialItem()
{

}

void
ShapeSectionPolynomialItem::init()
{
	if (abs(polynomialLateralSection_->getRealPointLow()->getPoint().x() - polynomialLateralSection_->getRealPointHigh()->getPoint().x()) < NUMERICAL_ZERO6)
	{
		polynomialLateralSection_->getControlPointsFromParameters(false);
	}

	realPointLowHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getRealPointLow());

	shapeSection_ = polynomialLateralSection_->getParentSection();
	if (shapeSection_->getLastPolynomialLateralSection() == polynomialLateralSection_)
	{
		realPointHighHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getRealPointHigh());
	}


	// Hover Events //
	//
	setAcceptHoverEvents(true);

	// Color & Path //
	//
	updateColor();
	createPath();

}

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to the road shape->
*/
void
ShapeSectionPolynomialItem::updateColor()
{
    QPen pen;
    pen.setWidth(2);
    pen.setCosmetic(true); // constant size independent of scaling

	pen.setColor(ODD::instance()->colors()->darkRed());

    setPen(pen);
}

void
ShapeSectionPolynomialItem::createPath()
{
	QPainterPath path;

	//	double pointsPerMeter = 1.0; // BAD: hard coded!
	double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
	double tStart = polynomialLateralSection_->getRealPointLow()->getPoint().x();
	double tEnd = polynomialLateralSection_->getRealPointHigh()->getPoint().x();
	int pointCount = int(ceil((tEnd - tStart) * pointsPerMeter)); // TODO curvature...
	if (pointCount <= 1)
	{
		pointCount = 2; // should be at least 2 to get a quad
	}
	pointCount = pointCount * 10;
	QVector<QPointF> points(pointCount + 1);
	double segmentLength = (tEnd - tStart) / pointCount;

	for (int i = 0; i <= pointCount; i++)
	{
		double x = i * segmentLength; 
		points[i] = QPointF(tStart + x, polynomialLateralSection_->f(x));
	}

	path.addPolygon(QPolygonF(points)); 
	setPath(path);
}


//################//
// OBSERVER       //
//################//

void
ShapeSectionPolynomialItem::updateObserver()
{
    // Parent //
    //
	LateralSectionItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ShapeSection //
    //
	int shapeChanges = polynomialLateralSection_->getPolynomialLateralSectionChanges();
	if (shapeChanges & PolynomialLateralSection::CPL_ParameterChange)
    {
        createPath();
        updateColor();
    }

	// Parameter Changed //
	//
	shapeChanges = polynomialLateralSection_->getLateralSectionChanges();
	if ((shapeChanges & LateralSection::CLS_LengthChange) || (shapeChanges & LateralSection::CLS_TChange))
	{
		if (realPointHighHandle_)
		{
			if (shapeSection_->getLastPolynomialLateralSection() != polynomialLateralSection_)
			{
				realPointHighHandle_->setParent(NULL);
				getProjectGraph()->addToGarbage(realPointHighHandle_);
				realPointHighHandle_ = NULL;
			}
		}
		else if (shapeSection_->getLastPolynomialLateralSection() == polynomialLateralSection_)
		{
			realPointHighHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getRealPointHigh());
		}

		createPath();
		updateColor();

	}
}

//################//
// SLOTS          //
//################//

bool
ShapeSectionPolynomialItem::removeSection()
{
    RemoveShapeSectionCommand *command = new RemoveShapeSectionCommand(shapeSection_, NULL);
    return getProjectGraph()->executeCommand(command); 
} 
