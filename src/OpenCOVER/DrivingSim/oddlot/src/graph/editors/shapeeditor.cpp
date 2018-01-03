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

#include "shapeeditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/commands/shapesectioncommands.hpp"
#include "src/data/commands/shapelateralsectioncommands.hpp"
#include "src/data/commands/lateralsectioncommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"

#include "src/graph/items/roadsystem/shape/shaperoadsystemitem.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitems.hpp"
#include "src/graph/items/roadsystem/shape/shaperoaditem.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/handles/splinemovehandle.hpp"


// Tools //
//
#include "src/gui/tools/shapeeditortool.hpp"
#include "src/gui/mouseaction.hpp"

// Qt //
//
#include <QGraphicsItem>
#include <QEasingCurve>

//################//
// CONSTRUCTORS   //
//################//

ShapeEditor::ShapeEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , profileGraph_(profileGraph)
    , insertSectionHandle_(NULL)
	, boundingBox_(QRectF( -7.5, 0, 15.0, 15.0))
{

}

ShapeEditor::~ShapeEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

SectionHandle *
ShapeEditor::getInsertSectionHandle()
{
if (!insertSectionHandle_)
{
	qDebug("ERROR 1006211555! ShapeEditor not yet initialized.");
}
return insertSectionHandle_;
}

/*! \brief Adds a road to the list of selected roads.
*
* Calls fitInView for the selected road and displays it in the ProfileGraph.
*/
void
ShapeEditor::addSelectedShapeSection(ShapeSection *shapeSection)
{
	if (!selectedShapeSectionItems_.contains(shapeSection))
	{
		// Activate Section in ProfileGraph //
		//
		ShapeSectionPolynomialItems *shapeSectionPolynomialItems = new ShapeSectionPolynomialItems(profileGraph_, shapeSection);
		selectedShapeSectionItems_.insert(shapeSection, shapeSectionPolynomialItems);

		profileGraph_->getScene()->addItem(shapeSectionPolynomialItems);

		// Fit View //
		//
		QRectF BB = shapeSectionPolynomialItems->boundingRect();
		if (BB.width() > boundingBox_.width())
		{
			boundingBox_.setWidth(BB.width());
		}
		if (BB.height() > boundingBox_.height())
		{
			boundingBox_.setHeight(BB.height());
		}

		profileGraph_->getView()->fitInView(boundingBox_);
		profileGraph_->getView()->zoomOut(Qt::Horizontal | Qt::Vertical);

	}
	else
	{
		//qDebug("already there");
	}
}

int
ShapeEditor::delSelectedShapeSection(ShapeSection *shapeSection)
{
	ShapeSectionPolynomialItems *shapeSectionPolynomialItems = selectedShapeSectionItems_.take(shapeSection);
	if (!shapeSectionPolynomialItems)
	{
		return 0;
	}
	else
	{
		// Deactivate ShapeSectionPolynomials in ProfileGraph //
		//
		profileGraph_->getScene()->removeItem(shapeSectionPolynomialItems);
		return 1;
	}
}

void
ShapeEditor::fitBoundingBoxInView()
{
	// Fit View //
	//
/*	QRectF newBB(0, 0, 15.0, 15.0);
	foreach (ShapeSectionPolynomialItems* polynomialItems, selectedShapeSectionItems_)
	{
		QRectF BB = polynomialItems->boundingRect();
		if (BB.y() < newBB.y())
		{
			newBB.setY(BB.y());
		}
		if (BB.width() > newBB.width())
		{
			newBB.setWidth(BB.width());
		}
		if (BB.height() > newBB.height())
		{
			newBB.setHeight(BB.height());
		}
	}

	boundingBox_ = newBB;
	profileGraph_->getView()->fitInView(boundingBox_); */
}

QMap<double, PolynomialLateralSection *>::ConstIterator ShapeEditor::addLateralSectionsBeforeToEasingCurve(QEasingCurve &easingCurve, QMap<double, PolynomialLateralSection *>::ConstIterator it,
	PolynomialLateralSection *lateralSectionBefore, PolynomialLateralSection *nextLateralSection, ShapeSectionPolynomialItems *polyItems)
{
	if (lateralSectionBefore)
	{
		PolynomialLateralSection *polySection = it.value();
		while (polySection != lateralSectionBefore)
		{
			easingCurve.addCubicBezierSegment(polyItems->normalize(polySection->getSplineControlPointLow()->getPoint()),
				polyItems->normalize(polySection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(polySection->getRealPointHigh()->getPoint()));
			it++;
			polySection = it.value();
		}
	}

	return it;
}

void 
ShapeEditor::addLateralSectionsNextToEasingCurve(QEasingCurve &easingCurve, QMap<double, PolynomialLateralSection *>::ConstIterator it, ShapeSection *shapeSection, ShapeSectionPolynomialItems *polyItems)
{
	while (it != shapeSection->getPolynomialLateralSections().constEnd())
	{
		PolynomialLateralSection *polySection = it.value();
		easingCurve.addCubicBezierSegment(polyItems->normalize(polySection->getSplineControlPointLow()->getPoint()),
			polyItems->normalize(polySection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(polySection->getRealPointHigh()->getPoint()));
		it++;
	}
}

void
ShapeEditor::translateMoveHandles(const QPointF &mousePos, SplineControlPoint *corner)
{
	PolynomialLateralSection *lateralSection = corner->getParent();
	ShapeSection *shapeSection = lateralSection->getParentSection();
	double t = lateralSection->getTStart();
	PolynomialLateralSection *lateralSectionBefore = shapeSection->getPolynomialLateralSectionBefore(t);
	PolynomialLateralSection *nextLateralSection = shapeSection->getPolynomialLateralSectionNext(t);

	if (corner->isReal())
	{
		if ((lateralSectionBefore && (mousePos.x() <= lateralSectionBefore->getTStart())) || (lateralSection && (mousePos.x() >= lateralSection->getTEnd())))
		{
			qDebug() << "return";
			return;
		}
	}

	ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);

	polyItems->initNormalization();
	SplineControlPoint *startPoint = shapeSection->getFirstPolynomialLateralSection()->getRealPointLow()->getClone();

	QEasingCurve easingCurve(QEasingCurve::BezierSpline);

	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
	it = addLateralSectionsBeforeToEasingCurve(easingCurve, it, lateralSectionBefore, nextLateralSection, polyItems);

	if (corner == shapeSection->getFirstPolynomialLateralSection()->getRealPointLow()) // Lateralprofile start
	{
		qreal d = QLineF(mousePos, corner->getPoint()).length();
		if (d < 10)
		{
			QPointF distance = mousePos - corner->getPoint();  
			startPoint->getPoint().setY(mousePos.y());
			QPointF p1 = lateralSection->getSplineControlPointLow()->getPoint();
			p1 += distance;
			easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(lateralSection->getSplineControlPointHigh()->getPoint()), 
				polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
			it++;

			if (nextLateralSection)
			{
				easingCurve.addCubicBezierSegment(polyItems->normalize(nextLateralSection->getSplineControlPointLow()->getPoint()),
					polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
				it++;
			}

		}
	}

	else
	{
		if (corner->isReal()) 
		{
			//move also the tangents
			QPointF distance = mousePos - corner->getPoint();

			if (corner->isLow())
			{
				QPointF p0 = lateralSection->getRealPointLow()->getPoint();
				p0 += distance;
				QPointF p1 = lateralSection->getSplineControlPointLow()->getPoint();
				p1 += distance;

				if (lateralSectionBefore)
				{
					QPointF p2 = lateralSectionBefore->getSplineControlPointHigh()->getPoint();
					p2 += distance;
					easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()),
						polyItems->normalize(p2), polyItems->normalize(p0));
					it++;
				}

				easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(lateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
				it++;


				if (nextLateralSection)
				{
					easingCurve.addCubicBezierSegment(polyItems->normalize(nextLateralSection->getSplineControlPointLow()->getPoint()),
						polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
					it++;
				} 
			}
			else
			{
				if (lateralSectionBefore)
				{
					easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()),
						polyItems->normalize(lateralSectionBefore->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSectionBefore->getRealPointHigh()->getPoint()));
					it++;
				}

				QPointF p3 = lateralSection->getRealPointHigh()->getPoint();
				p3 += distance;
				QPointF p2 = lateralSection->getSplineControlPointHigh()->getPoint();
				p2 += distance;
				easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSection->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(p3));
				it++;

				if (nextLateralSection)
				{
					QPointF p1 = nextLateralSection->getSplineControlPointLow()->getPoint();
					p1 += distance;
					easingCurve.addCubicBezierSegment(polyItems->normalize(p1),
						polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
					it++;
				}
			}
		}
		else
		{
			if (!corner->isSmooth())
			{
				if (lateralSectionBefore)
				{
					easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()),
						polyItems->normalize(lateralSectionBefore->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSectionBefore->getRealPointHigh()->getPoint()));
					it++;
				}


				if (corner->isLow())
				{
					easingCurve.addCubicBezierSegment(polyItems->normalize(mousePos), polyItems->normalize(lateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
				}
				else
				{
					easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSection->getSplineControlPointLow()->getPoint()), polyItems->normalize(mousePos), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
				}
				it++;
			}
			else
			{
				QPointF distance = mousePos - corner->getPoint();

				if (corner->isLow())
				{
					if (lateralSectionBefore)
					{
						QPointF p2 = lateralSectionBefore->getSplineControlPointHigh()->getPoint();
						p2 -= distance;
						easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(lateralSectionBefore->getRealPointHigh()->getPoint()));
						it++;
					}

					QPointF p1 = lateralSection->getSplineControlPointLow()->getPoint();
					p1 += distance;
					easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(lateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
					it++;

					if (nextLateralSection)
					{
						easingCurve.addCubicBezierSegment(polyItems->normalize(nextLateralSection->getSplineControlPointLow()->getPoint()),
							polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
						it++;
					}
				}
				else
				{
					if (lateralSectionBefore)
					{
						easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()),
							polyItems->normalize(lateralSectionBefore->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSectionBefore->getRealPointHigh()->getPoint()));
						it++;
					}

					QPointF p2 = lateralSection->getSplineControlPointHigh()->getPoint();
					p2 += distance;
					easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSection->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
					it++;

					PolynomialLateralSection *nextLateralSection = shapeSection->getPolynomialLateralSectionNext(t);
					if (nextLateralSection)
					{
						QPointF p1 = nextLateralSection->getSplineControlPointLow()->getPoint();
						p1 += distance;
						easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
						it++;
					}
				}
			}

		}
	}

	addLateralSectionsNextToEasingCurve(easingCurve, it, shapeSection, polyItems);

	QList<QPointF> scenePoints = setEasingCurve(polyItems, startPoint, easingCurve);


	MovePointLateralShapeSectionCommand *command = new MovePointLateralShapeSectionCommand(shapeSection, corner, scenePoints);
	getProjectGraph()->executeCommand(command);
}


QList<QPointF>
ShapeEditor::setEasingCurve(ShapeSectionPolynomialItems *polyItems, SplineControlPoint *start, const QEasingCurve &easingCurve)
{

	QVector<QPointF> points = easingCurve.toCubicSpline();
	QList<QPointF> scenePoints;

	QPointF startPoint = start->getPoint();
	QPointF p0 = startPoint;
	int i = 0;
	while (i < points.size())
	{
		scenePoints.append(p0);
		QPointF p = polyItems->sceneCoords(startPoint, points.at(i));
		scenePoints.append(polyItems->sceneCoords(startPoint, points.at(i++)));
		p = polyItems->sceneCoords(startPoint, points.at(i));
		scenePoints.append(polyItems->sceneCoords(startPoint, points.at(i++)));
		p = polyItems->sceneCoords(startPoint, points.at(i));
		scenePoints.append(polyItems->sceneCoords(startPoint, points.at(i++)));
		p0 = scenePoints.last();
	}

	return scenePoints;
}

void 
ShapeEditor::addLateralSection(ShapeSection *shapeSection, const QPointF &mousePos)
{
	double t = mousePos.x();
	PolynomialLateralSection *polySection = shapeSection->getShape(t);
	PolynomialLateralSection *nextLateralSection = shapeSection->getPolynomialLateralSectionNext(t);
	ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);

	polyItems->initNormalization();
	SplineControlPoint *startPoint = shapeSection->getFirstPolynomialLateralSection()->getRealPointLow()->getClone();

	QEasingCurve easingCurve(QEasingCurve::BezierSpline);

	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
	it = addLateralSectionsBeforeToEasingCurve(easingCurve, it, polySection, nextLateralSection, polyItems);

	PolynomialLateralSection *newPolySection = new PolynomialLateralSection(t);


	QPointF realPointBefore = polySection->getRealPointLow()->getPoint();
	QPointF p2 = (mousePos + realPointBefore) / 2;
	easingCurve.addCubicBezierSegment(polyItems->normalize(polySection->getSplineControlPointLow()->getPoint()),
		polyItems->normalize(p2), polyItems->normalize(mousePos));
	it++;

	if (polySection->getRealPointHigh()->isSmooth())
	{
		QPointF after;
		if (nextLateralSection)
		{
			after = nextLateralSection->getRealPointHigh()->getPoint();
		}
		else
		{
			after = polySection->getRealPointHigh()->getPoint();
		}

		QPointF tangent = (after - mousePos) / 6;

		QPointF realPointNext = polySection->getRealPointHigh()->getPoint();
		QPointF p1 = (mousePos + realPointNext) / 2;
		QPointF p2 = realPointNext - tangent;
		easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(p2), polyItems->normalize(realPointNext));

		if (nextLateralSection)
		{
			QPointF p1 = realPointNext + tangent;
			easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
			it++;
		}
	}
	else
	{

		QPointF realPointNext = polySection->getRealPointHigh()->getPoint();
		QPointF p1 = (mousePos + realPointNext) / 2;
		easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(polySection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(polySection->getRealPointHigh()->getPoint()));

		if (nextLateralSection)
		{
			easingCurve.addCubicBezierSegment(polyItems->normalize(nextLateralSection->getSplineControlPointLow()->getPoint()),
				polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
			it++;
		}
	}

	addLateralSectionsNextToEasingCurve(easingCurve, it, shapeSection, polyItems);


	QList<QPointF> scenePoints = setEasingCurve(polyItems, startPoint, easingCurve);

	if (polySection->getRealPointHigh()->isSmooth())
	{


		getProjectData()->getUndoStack()->beginMacro(QObject::tr("Add Point"));

		newPolySection->getRealPointHigh()->setSmooth(true);
		newPolySection->getSplineControlPointHigh()->setSmooth(true);

		AddLateralShapeSectionCommand *command = new AddLateralShapeSectionCommand(shapeSection, newPolySection, scenePoints);
		getProjectGraph()->executeCommand(command);

		QList<SplineControlPoint *> corners;
		corners.append(polySection->getSplineControlPointHigh());
		corners.append(polySection->getRealPointHigh());
		SmoothPointLateralShapeSectionCommand *smoothCommand = new SmoothPointLateralShapeSectionCommand(corners, false);
		getProjectGraph()->executeCommand(smoothCommand);

		getProjectData()->getUndoStack()->endMacro();
	}
	else
	{
		AddLateralShapeSectionCommand *command = new AddLateralShapeSectionCommand(shapeSection, newPolySection, scenePoints);
		getProjectGraph()->executeCommand(command);
	}
}

void
ShapeEditor::smoothPoint(bool smooth, SplineControlPoint *corner)
{
	PolynomialLateralSection *lateralSection = corner->getParent();
	ShapeSection *shapeSection = lateralSection->getParentSection();
	double t = lateralSection->getTStart();
	PolynomialLateralSection *lateralSectionBefore = shapeSection->getPolynomialLateralSectionBefore(t);
	PolynomialLateralSection *nextLateralSection = shapeSection->getPolynomialLateralSectionNext(t);
	if (smooth)
	{
		ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);

		SplineControlPoint *startPoint = shapeSection->getFirstPolynomialLateralSection()->getRealPointLow()->getClone();
//		QPointF before = polyItems->normalize(lateralSectionBefore->getRealPointLow()->getPoint());
		QPointF before = lateralSectionBefore->getRealPointLow()->getPoint();
		QPointF after;
		if (nextLateralSection)
		{
			after = nextLateralSection->getRealPointHigh()->getPoint();
		}
		else
		{
			after = lateralSection->getRealPointHigh()->getPoint();
		}

		QPointF tangent = (after - before) / 6;

		QEasingCurve easingCurve(QEasingCurve::BezierSpline);
		QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
		it = addLateralSectionsBeforeToEasingCurve(easingCurve, it, lateralSectionBefore, lateralSection, polyItems);

		QList<SplineControlPoint *> corners;

		if (lateralSectionBefore)
		{
			QPointF p2 = corner->getPoint() - tangent;
			easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(lateralSectionBefore->getRealPointHigh()->getPoint()));
			it++;
			corners.append(lateralSectionBefore->getRealPointHigh());
			corners.append(lateralSectionBefore->getSplineControlPointHigh());
		}

		corners.append(corner);

		QPointF p1 = corner->getPoint() + tangent;
		easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(lateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
		it++;
		corners.append(lateralSection->getSplineControlPointLow());

		addLateralSectionsNextToEasingCurve(easingCurve, it, shapeSection, polyItems);


		QList<QPointF> scenePoints = setEasingCurve(polyItems, startPoint, easingCurve);

		getProjectData()->getUndoStack()->beginMacro(QObject::tr("Smooth Point"));
		MovePointLateralShapeSectionCommand *command = new MovePointLateralShapeSectionCommand(shapeSection, corner, scenePoints);
		getProjectGraph()->executeCommand(command);

		SmoothPointLateralShapeSectionCommand *smoothCommand = new SmoothPointLateralShapeSectionCommand(corners, smooth);
		getProjectGraph()->executeCommand(smoothCommand);
		getProjectData()->getUndoStack()->endMacro();

	}
	else
	{
		QList<SplineControlPoint *> corners;
		corners.append(corner);
		corners.append(lateralSection->getSplineControlPointLow());

		if (lateralSectionBefore)
		{
			corners.append(lateralSectionBefore->getSplineControlPointHigh());
		}
		SmoothPointLateralShapeSectionCommand *smoothCommand = new SmoothPointLateralShapeSectionCommand(corners, smooth);
		getProjectGraph()->executeCommand(smoothCommand);
	}

}

void
ShapeEditor::cornerPoint(SplineControlPoint *corner)
{
	PolynomialLateralSection *lateralSection = corner->getParent();
	ShapeSection *shapeSection = lateralSection->getParentSection();
	double t = lateralSection->getTStart();
	PolynomialLateralSection *lateralSectionBefore = shapeSection->getPolynomialLateralSectionBefore(t);
	PolynomialLateralSection *nextLateralSection = shapeSection->getPolynomialLateralSectionNext(t);
	ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);

	SplineControlPoint *startPoint = shapeSection->getFirstPolynomialLateralSection()->getRealPointLow()->getClone();

	QEasingCurve easingCurve(QEasingCurve::BezierSpline);
	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
	it = addLateralSectionsBeforeToEasingCurve(easingCurve, it, lateralSectionBefore, lateralSection, polyItems);


	QList<SplineControlPoint *> corners;
	if (lateralSectionBefore)
	{
		QPointF p2 = (lateralSectionBefore->getRealPointLow()->getPoint() - corner->getPoint()) / 3 + corner->getPoint();
		easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(lateralSectionBefore->getRealPointHigh()->getPoint()));
		it++;

		corners.append(corner);
		corners.append(lateralSectionBefore->getSplineControlPointHigh());
	}


	QPointF p1 = (lateralSection->getRealPointHigh()->getPoint() - corner->getPoint()) / 3 + corner->getPoint();
	easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(lateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(lateralSection->getRealPointHigh()->getPoint()));
	it++;
	corners.append(lateralSection->getSplineControlPointLow());

	addLateralSectionsNextToEasingCurve(easingCurve, it, shapeSection, polyItems);


	QList<QPointF> scenePoints = setEasingCurve(polyItems, startPoint, easingCurve);

	getProjectData()->getUndoStack()->beginMacro(QObject::tr("Smooth Point"));
	MovePointLateralShapeSectionCommand *command = new MovePointLateralShapeSectionCommand(shapeSection, corner, scenePoints);
	getProjectGraph()->executeCommand(command);

	SmoothPointLateralShapeSectionCommand *smoothCommand = new SmoothPointLateralShapeSectionCommand(corners, false);
	getProjectGraph()->executeCommand(smoothCommand);
	getProjectData()->getUndoStack()->endMacro();

}

void 
ShapeEditor::deleteLateralSection(SplineControlPoint *corner)
{
	PolynomialLateralSection *polySection = corner->getParent();
	ShapeSection *shapeSection = polySection->getParentSection();

	PolynomialLateralSection *lateralSectionBefore = shapeSection->getPolynomialLateralSectionBefore(polySection->getTStart());  // Section to be deleted
	if (lateralSectionBefore)
	{
		ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);

		polyItems->initNormalization();
		SplineControlPoint *startPoint = shapeSection->getFirstPolynomialLateralSection()->getRealPointLow()->getClone();

		QEasingCurve easingCurve(QEasingCurve::BezierSpline);

		QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
		it = addLateralSectionsBeforeToEasingCurve(easingCurve, it, lateralSectionBefore, polySection, polyItems);

		QList<SplineControlPoint *> smoothCorners;
		QList<SplineControlPoint *> corners;
		PolynomialLateralSection *nextLateralSection = shapeSection->getPolynomialLateralSectionNext(polySection->getTStart());

		if (nextLateralSection && nextLateralSection->getRealPointLow()->isSmooth())
		{
			QPointF before = lateralSectionBefore->getRealPointLow()->getPoint();
			QPointF after = nextLateralSection->getRealPointHigh()->getPoint();

			QPointF tangent = (after - before) / 6;

			QPointF p2 = nextLateralSection->getRealPointLow()->getPoint() - tangent;
			easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(polySection->getRealPointHigh()->getPoint()));
			it++;

			QPointF p1 = nextLateralSection->getRealPointLow()->getPoint() + tangent;
			easingCurve.addCubicBezierSegment(polyItems->normalize(p1), polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
			it++;

			if (!corner->isSmooth())
			{
				smoothCorners.append(lateralSectionBefore->getSplineControlPointHigh()); //smooth true
				smoothCorners.append(lateralSectionBefore->getRealPointHigh());
			}
		}
		else
		{
			QPointF p2 = (lateralSectionBefore->getRealPointLow()->getPoint() - polySection->getRealPointHigh()->getPoint()) / 3 + polySection->getRealPointHigh()->getPoint();
			easingCurve.addCubicBezierSegment(polyItems->normalize(lateralSectionBefore->getSplineControlPointLow()->getPoint()), polyItems->normalize(p2), polyItems->normalize(polySection->getRealPointHigh()->getPoint()));
			it++;

			if (nextLateralSection)
			{
				easingCurve.addCubicBezierSegment(polyItems->normalize(nextLateralSection->getSplineControlPointLow()->getPoint()),
					polyItems->normalize(nextLateralSection->getSplineControlPointHigh()->getPoint()), polyItems->normalize(nextLateralSection->getRealPointHigh()->getPoint()));
				it++;
			}

			if (corner->isSmooth())
			{
				corners.append(lateralSectionBefore->getSplineControlPointHigh()); //smooth false
				corners.append(lateralSectionBefore->getRealPointHigh());
			}
		}
		it++;

		addLateralSectionsNextToEasingCurve(easingCurve, it, shapeSection, polyItems);

		QList<QPointF> scenePoints = setEasingCurve(polyItems, startPoint, easingCurve);


		if (corners.isEmpty() && smoothCorners.isEmpty())
		{
			DeleteLateralShapeSectionCommand *command = new DeleteLateralShapeSectionCommand(shapeSection, polySection, scenePoints);
			getProjectGraph()->executeCommand(command);
		}
		else
		{
			getProjectData()->getUndoStack()->beginMacro(QObject::tr("Delete Point"));

			DeleteLateralShapeSectionCommand *command = new DeleteLateralShapeSectionCommand(shapeSection, polySection, scenePoints);
			getProjectGraph()->executeCommand(command);

			if (!smoothCorners.isEmpty())
			{
				SmoothPointLateralShapeSectionCommand *smoothCommand = new SmoothPointLateralShapeSectionCommand(smoothCorners, true);
				getProjectGraph()->executeCommand(smoothCommand);
			}

			if (!corners.isEmpty())
			{
				SmoothPointLateralShapeSectionCommand *smoothCommand = new SmoothPointLateralShapeSectionCommand(corners, false);
				getProjectGraph()->executeCommand(smoothCommand);
			}

			getProjectData()->getUndoStack()->endMacro();
		}
	}
}

//################//
// TOOL           //
//################//

/*! \brief ToolAction 'Chain of Responsibility'.
*
*/
void
ShapeEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    // Tools //
    //
 /*   ShapeEditorToolAction *shapeEditorToolAction = dynamic_cast<ShapeEditorToolAction *>(toolAction);
    if (shapeEditorToolAction)
    {
    } */
}

//###################//
// MOUSE ACTION     //
//##################//
void
ShapeEditor::mouseAction(MouseAction *mouseAction)
{
	static bool mousePressed = false;

	QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();
	ProjectEditor::mouseAction(mouseAction);

	// Add point to curve //
	//
	QPointF mousePoint = mouseEvent->scenePos();

	if (mouseAction->getMouseActionType() == MouseAction::PATM_MOVE)
	{
		if (!mousePressed && !selectedShapeSectionItems_.empty())
		{
			ShapeSection *shapeSection = selectedShapeSectionItems_.firstKey();
			PolynomialLateralSection *lateralSection = shapeSection->getPolynomialLateralSection(mousePoint.x());
			double d;
			if (lateralSection)
			{
				d = QLineF(lateralSection->getRealPointLow()->getPoint(), mousePoint).length();
				if (d > 0.5)
				{
					d = QLineF(lateralSection->getSplineControlPointLow()->getPoint(), mousePoint).length();
					if (d > 0.5)
					{
						d = QLineF(lateralSection->getSplineControlPointHigh()->getPoint(), mousePoint).length();
						if (d > 0.5)
						{
							d = QLineF(lateralSection->getRealPointHigh()->getPoint(), mousePoint).length();
						}
					}
				}
				if (d < 0.5)
				{
					profileGraph_->getView()->setCursor(Qt::OpenHandCursor);
				}
				else
				{
					QPointF curvePoint = QPointF(mousePoint.x(), lateralSection->f(mousePoint.x() - lateralSection->getTStart()));

					d = QLineF(mousePoint, curvePoint).length();
					if (d < 0.5)
					{

						profileGraph_->getView()->setCursor(Qt::CrossCursor);
					}
				}
			}

			if (!lateralSection || (d >= 0.5))
			{
				profileGraph_->getView()->setCursor(Qt::ArrowCursor);
			}
		}
	}
	else if (mouseAction->getMouseActionType() == MouseAction::PATM_PRESS)
	{
		if (mouseEvent->button() == Qt::LeftButton)
		{
			if (!selectedShapeSectionItems_.empty())
			{
				ShapeSection *shapeSection = selectedShapeSectionItems_.firstKey();
				PolynomialLateralSection *lateralSection = shapeSection->getPolynomialLateralSection(mousePoint.x());
				if (lateralSection)
				{
					double d = QLineF(lateralSection->getRealPointLow()->getPoint(), mousePoint).length();
					if (d > 0.5)
					{
						d = QLineF(lateralSection->getSplineControlPointLow()->getPoint(), mousePoint).length();
						if (d > 0.5)
						{
							d = QLineF(lateralSection->getSplineControlPointHigh()->getPoint(), mousePoint).length();
							if (d > 0.5)
							{
								d = QLineF(lateralSection->getRealPointHigh()->getPoint(), mousePoint).length();
							}
						}
					}
					if (d < 0.5)
					{
						profileGraph_->getView()->setCursor(Qt::ClosedHandCursor);
						mousePressed = true;
					}
					else
					{

						QPointF curvePoint = QPointF(mousePoint.x(), lateralSection->f(mousePoint.x() - lateralSection->getTStart()));
	//					qDebug() << "CurvePoint: " << curvePoint.x() << "," << curvePoint.y() << " MousePoint: " << mousePoint.x() << "," << mousePoint.y();

						d = QLineF(mousePoint, curvePoint).length();
						if (d < 0.5)
						{
							addLateralSection(selectedShapeSectionItems_.firstKey(), mousePoint);
						}

					}
				}
			}
		}
	} 
	else if (mouseAction->getMouseActionType() == MouseAction::PATM_RELEASE)
	{
		if (mousePressed)
		{
			mousePressed = false;
		}
	}
}


//################//
// SLOTS          //
//################//

/*!
*
*/
void
ShapeEditor::init()
{
    // Graph //
    //
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new ShapeRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);
    }

    // ProfileGraph //
    //

    profileGraph_->getScene()->setSceneRect(-1000.0, -45.0, 20000.0, 90.0);


    // Section Handle //
    //
    // TODO: Is this really the best object for holding this?
    insertSectionHandle_ = new SectionHandle(roadSystemItem_);
    insertSectionHandle_->hide();
}

/*!
*/
void
ShapeEditor::kill()
{
    delete roadSystemItem_;
    roadSystemItem_ = NULL;

	foreach (ShapeSectionPolynomialItems *polynomialItems, selectedShapeSectionItems_)
	{
		delete polynomialItems;
	}
}
