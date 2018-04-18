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
/*		double width = shapeSectionPolynomialItems->getSectionWidth();
		QRectF BB = shapeSectionPolynomialItems->boundingRect();
		boundingBox_ = QRectF(-width, BB.y(), 2 * width, (BB.height() > 8.0) ? BB.height() : 8.0); */

		if (boundingBox_.isEmpty())
		{
			boundingBox_ = QRectF(-20.0, -5.0, 40.0, 10.0);
			profileGraph_->getView()->fitInView(boundingBox_);
			profileGraph_->getView()->zoomOut(Qt::Horizontal | Qt::Vertical);
		}

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

QMap<double, PolynomialLateralSection *>::ConstIterator 
ShapeEditor::addLateralSectionsBefore(QList<QPointF> &scenePoints, QMap<double, PolynomialLateralSection *>::ConstIterator it, PolynomialLateralSection *lateralSectionBefore, ShapeSectionPolynomialItems *polyItems)
{
	if (lateralSectionBefore)
	{
		PolynomialLateralSection *polySection = it.value();
		while (polySection != lateralSectionBefore)
		{
			scenePoints.append(polySection->getRealPointLow()->getPoint());
			scenePoints.append(polySection->getRealPointHigh()->getPoint());
			it++;
			polySection = it.value();
		}
	}

	return it;
}

void 
ShapeEditor::addLateralSectionsNext(QList<QPointF> &scenePoints, QMap<double, PolynomialLateralSection *>::ConstIterator it, ShapeSection *shapeSection, ShapeSectionPolynomialItems *polyItems)
{
	while (it != shapeSection->getPolynomialLateralSections().constEnd())
	{
		PolynomialLateralSection *polySection = it.value();
		scenePoints.append(polySection->getRealPointLow()->getPoint());
		scenePoints.append(polySection->getRealPointHigh()->getPoint());
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

	if ((lateralSectionBefore && (mousePos.x() <= lateralSectionBefore->getTStart())) || (nextLateralSection && (mousePos.x() >= nextLateralSection->getTStart())))
	{
		return;
	}
	if (!nextLateralSection && (corner->isLow() && (mousePos.x() >= lateralSection->getRealPointHigh()->getPoint().x())) || (!corner->isLow() && (mousePos.x()) <= lateralSection->getTStart()))
	{
		return;
	}

	ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);
	QList<QPointF> scenePoints;

	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();

	if (shapeSection->getLastPolynomialLateralSection()->getRealPointHigh() == corner)   // last point
	{
		it = addLateralSectionsBefore(scenePoints, it, lateralSection, polyItems);
		scenePoints.append(lateralSection->getRealPointLow()->getPoint());
		scenePoints.append(mousePos);
	}
	else
	{
		it = addLateralSectionsBefore(scenePoints, it, lateralSectionBefore, polyItems);
		if (lateralSectionBefore)
		{
			scenePoints.append(lateralSectionBefore->getRealPointLow()->getPoint());
			scenePoints.append(mousePos);
			it++;
		}

		scenePoints.append(mousePos);
		scenePoints.append(lateralSection->getRealPointHigh()->getPoint());
	}
	it++;

	addLateralSectionsNext(scenePoints, it, shapeSection, polyItems);

	MovePointLateralShapeSectionCommand *command = new MovePointLateralShapeSectionCommand(shapeSection, corner, scenePoints);
	getProjectGraph()->executeCommand(command);

}

void
ShapeEditor::addLateralSection(ShapeSection *shapeSection, const QPointF &mousePos)
{
	double t = mousePos.x();
	PolynomialLateralSection *polySection = shapeSection->getShape(t);
	ShapeSectionPolynomialItems *polyItems = selectedShapeSectionItems_.value(shapeSection);

	QList<QPointF> scenePoints;
	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
	it = addLateralSectionsBefore(scenePoints, it, polySection, polyItems);

	PolynomialLateralSection *newPolySection = new PolynomialLateralSection(t);
	scenePoints.append(polySection->getRealPointLow()->getPoint());
	scenePoints.append(mousePos);

	scenePoints.append(mousePos);
	scenePoints.append(polySection->getRealPointHigh()->getPoint());
	it++;

	addLateralSectionsNext(scenePoints, it, shapeSection, polyItems);


	AddLateralShapeSectionCommand *command = new AddLateralShapeSectionCommand(shapeSection, newPolySection, scenePoints);
	getProjectGraph()->executeCommand(command);
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

		QList<QPointF> scenePoints;
		QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection->getPolynomialLateralSections().constBegin();
		it = addLateralSectionsBefore(scenePoints, it, lateralSectionBefore, polyItems);

		scenePoints.append(lateralSectionBefore->getRealPointLow()->getPoint());
		scenePoints.append(polySection->getRealPointHigh()->getPoint());
		it++;
		it++;

		addLateralSectionsNext(scenePoints, it, shapeSection, polyItems);

		DeleteLateralShapeSectionCommand *command = new DeleteLateralShapeSectionCommand(shapeSection, polySection, scenePoints);
		getProjectGraph()->executeCommand(command);
	}
}

void 
ShapeEditor::pastePolynomialLateralSections(ShapeSection *section)
{
	QMap<double, PolynomialLateralSection *> oldSections = section->getPolynomialLateralSections();
	QMap<double, PolynomialLateralSection *> newSections = clipboardShapeSection_->getPolynomialLateralSections();

	PasteLateralShapeSectionsCommand *command = new PasteLateralShapeSectionsCommand(section, oldSections, newSections);
	getProjectGraph()->executeCommand(command);
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
			if (!lateralSection)
			{
				lateralSection = shapeSection->getFirstPolynomialLateralSection();
			}
			double d;
			if (lateralSection)
			{
				d = QLineF(lateralSection->getRealPointLow()->getPoint(), mousePoint).length();
				if (d > 0.5)
				{
					d = QLineF(lateralSection->getRealPointHigh()->getPoint(), mousePoint).length();
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
				if (!lateralSection)
				{
					lateralSection = shapeSection->getFirstPolynomialLateralSection();
				}
				if (lateralSection)
				{
					double d = QLineF(lateralSection->getRealPointLow()->getPoint(), mousePoint).length();
					if (d > 0.5)
					{
						d = QLineF(lateralSection->getRealPointHigh()->getPoint(), mousePoint).length();
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

//    profileGraph_->getScene()->setSceneRect(-1000.0, -45.0, 20000.0, 90.0);
	profileGraph_->getScene()->setSceneRect(-200.0, -50.0, 400.0, 100.0);


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
