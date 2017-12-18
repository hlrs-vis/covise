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
#include "src/data/roadsystem/lateralSections/polynomiallateralsection.hpp"

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
	deleteControlPointHandles();
	if (realPointHighHandle_)
	{
		deleteRealPointHighHandle();
	}
}

void
ShapeSectionPolynomialItem::init()
{
	if (abs(polynomialLateralSection_->getRealPointLow()->getPoint().x() - polynomialLateralSection_->getRealPointHigh()->getPoint().x()) < NUMERICAL_ZERO6)
	{
		polynomialLateralSection_->getControlPointsFromParameters();
	}

	realPointLowHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getRealPointLow());
	controlPointLowHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getSplineControlPointLow());
	controlPointHighHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getSplineControlPointHigh());

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

/*    if (shapeSection_->getDegree() == 3)
    {
        pen->setColor(ODD::instance()->colors()->darkRed());
    }
    else if (shapeSection_->getDegree() == 2)
    {
        pen->setColor(ODD::instance()->colors()->darkOrange());
    }
    else if (shapeSection_->getDegree() == 1)
    {
        pen->setColor(ODD::instance()->colors()->darkGreen());
    }
    else if (shapeSection_->getDegree() == 0)
    {
        pen->setColor(ODD::instance()->colors()->darkCyan());
    }
    else
    {
        pen->setColor(ODD::instance()->colors()->darkBlue());
    } */

    setPen(pen);
}

void
ShapeSectionPolynomialItem::createPath()
{
	QPainterPath path;

	path.moveTo(polynomialLateralSection_->getRealPointLow()->getPoint());
	path.cubicTo(polynomialLateralSection_->getSplineControlPointLow()->getPoint(), polynomialLateralSection_->getSplineControlPointHigh()->getPoint(), polynomialLateralSection_->getRealPointHigh()->getPoint());

	path.moveTo(polynomialLateralSection_->getRealPointLow()->getPoint());
	path.lineTo(polynomialLateralSection_->getSplineControlPointLow()->getPoint());
	path.moveTo(polynomialLateralSection_->getRealPointHigh()->getPoint());
	path.lineTo(polynomialLateralSection_->getSplineControlPointHigh()->getPoint()); 

	setPath(path);
}

void 
ShapeSectionPolynomialItem::deleteControlPointHandles()
{
	ProfileGraph *profileGraph = getProfileGraph();
	if (profileGraph)
	{
		profileGraph->getScene()->removeItem(realPointLowHandle_);
		profileGraph->getScene()->removeItem(controlPointLowHandle_);
		profileGraph->getScene()->removeItem(controlPointHighHandle_);

		realPointLowHandle_->setParent(NULL);
		getProjectGraph()->addToGarbage(realPointLowHandle_);
		realPointLowHandle_ = NULL;

		controlPointLowHandle_->setParent(NULL);
		getProjectGraph()->addToGarbage(controlPointLowHandle_);
		controlPointLowHandle_ = NULL;

		controlPointHighHandle_->setParent(NULL);
		getProjectGraph()->addToGarbage(controlPointHighHandle_);
		controlPointHighHandle_ = NULL;
	}
}

void
ShapeSectionPolynomialItem::deleteRealPointHighHandle()
{
	ProfileGraph *profileGraph = getProfileGraph();
	if (profileGraph)
	{
		profileGraph->getScene()->removeItem(realPointHighHandle_);

		realPointHighHandle_->setParent(NULL);
		getProjectGraph()->addToGarbage(realPointHighHandle_);
		realPointHighHandle_ = NULL;
	}
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
				deleteRealPointHighHandle();
			}
		}
		else if (shapeSection_->getLastPolynomialLateralSection() == polynomialLateralSection_)
		{
			realPointHighHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getRealPointHigh());
		}
		
	/*	deleteControlPointHandles();

		controlPointLowHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getSplineControlPointLow());
		controlPointHighHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getSplineControlPointHigh());
		realPointLowHandle_ = new SplineMoveHandle(shapeEditor_, this, polynomialLateralSection_->getRealPointLow()); */

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
