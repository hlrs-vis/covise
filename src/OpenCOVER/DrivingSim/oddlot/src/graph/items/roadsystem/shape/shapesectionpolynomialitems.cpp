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
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"

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
	shapeEditor_ = dynamic_cast<ShapeEditor *>(profileGraph_->getProjectWidget()->getProjectEditor());
	createPolynomialItems();
	createPath();
	
}

void 
ShapeSectionPolynomialItems::createPolynomialItems()
{

	if (shapeEditor_)
	{
		// Remove shapes with zero width //
		//
		bool shapeDeleted;
		do
		{
			shapeDeleted = false;
			foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
			{
				if (!shape->getRealPointHigh())
				{
					shapeSection_->delShape(shape->getTStart());
					shapeDeleted = true;
					break;
				}
			}
		}
		while (shapeDeleted);

		// Create all lateral polynoms and cotrol handles //
		//
		foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
		{
			ShapeSectionPolynomialItem *shapeItem = new ShapeSectionPolynomialItem(this, shape, shapeEditor_);
		}
	}

}

void
ShapeSectionPolynomialItems::createPath()
{
	double s = shapeSection_->getSStart();
	RSystemElementRoad *road = shapeSection_->getParentRoad();
	LaneSection *laneSection = road->getLaneSection(shapeSection_->getSStart());

	QRectF bb = boundingRect();

	QGraphicsItemGroup *graphicItemGroup = new QGraphicsItemGroup(this);
	QGraphicsPathItem *pathItemRL = new QGraphicsPathItem(this);
	graphicItemGroup->addToGroup(pathItemRL);
	QGraphicsPathItem *pathItemM = new QGraphicsPathItem(this);
	graphicItemGroup->addToGroup(pathItemM);
	QPainterPath pathRL;
	QPainterPath pathM;

	pathItemRL->setPen(QPen(ODD::instance()->colors()->brightGrey(), 0.05));

	pathItemM->setPen(QPen(ODD::instance()->colors()->brightGrey()));
	foreach(Lane *lane, laneSection->getLanes())
	{
		if (lane->getId() > 0)
		{
			pathRL.moveTo(laneSection->getLaneSpanWidth(0, lane->getId(), s) + road->getLaneOffset(s), -5.0);
			pathRL.lineTo(laneSection->getLaneSpanWidth(0, lane->getId(), s) + road->getLaneOffset(s), 5.0);
		}
		else if (lane->getId() < 0)
		{
			pathRL.moveTo(-laneSection->getLaneSpanWidth(0, lane->getId(), s) + road->getLaneOffset(s), -5.0);
			pathRL.lineTo(-laneSection->getLaneSpanWidth(0, lane->getId(), s) + road->getLaneOffset(s), 5.0);
		}
		else
		{

			// Width //
			//
			LaneRoadMark *roadMark = lane->getRoadMarkEntry(laneSection->getSStart());
			double width = roadMark->getRoadMarkWidth();
			if (width >= 0.0)
			{
				pathItemM->setPen(QPen(ODD::instance()->colors()->brightGrey(), width));

				// Type //
				//
				LaneRoadMark::RoadMarkType type = roadMark->getRoadMarkType();
				if (type == LaneRoadMark::RMT_BROKEN)
				{
					// TODO: no hard coding
					const QVector<qreal> pattern(2, 3.0 / width); // hard coded 3.0m for town
																  // init with 2 values
																  // 0: length of the line, 1: length of the free space
																  // actual length will be multiplied with the width
					QPen pen = pathItemM->pen();
					pen.setDashPattern(pattern);
					pathItemM->setPen(pen);
				}

				pathM.moveTo(0.0, -5.0);
				pathM.lineTo(0.0, 5.0);
			}
		}
	}
	pathItemM->setPath(pathM);
	pathItemRL->setPath(pathRL);
}

QRectF 
ShapeSectionPolynomialItems::boundingRect()
{
	double ymin = 1000;
	double ymax = -1000;
	
	foreach(PolynomialLateralSection *shape, shapeSection_->getShapes())
	{
		double y0 = fmin(shape->getRealPointLow()->getPoint().y(), shape->getRealPointHigh()->getPoint().y());
		if (y0 < ymin)
		{
			ymin = y0;
		}
		double y1;
		if (ymin == shape->getRealPointLow()->getPoint().y())
		{
			y1 = shape->getRealPointHigh()->getPoint().y();
		}
		else
		{
			y1 = shape->getRealPointLow()->getPoint().y();
		}
		if (y1 > ymax)
		{
			ymax = y1;
		}
	}
	return QRectF { 0, ymin, shapeSection_->getParentRoad()->getMaxWidth(shapeSection_->getSStart()) - shapeSection_->getParentRoad()->getMinWidth(shapeSection_->getSStart()), ymax - ymin };
}


double 
ShapeSectionPolynomialItems::getSectionWidth()
{
	return shapeSection_->getWidth();
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



