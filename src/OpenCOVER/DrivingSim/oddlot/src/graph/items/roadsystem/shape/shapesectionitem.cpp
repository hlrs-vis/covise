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

#include "shapesectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/shapesectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadtextitem.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitem.hpp"
#include "src/graph/items/roadsystem/shape/shapesectionpolynomialitems.hpp"
#include "src/graph/items/handles/splinemovehandle.hpp"

// Editor //
//
#include "src/graph/editors/shapeeditor.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/lodsettings.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>
#include <QMessageBox>

#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

ShapeSectionItem::ShapeSectionItem(ShapeEditor *shapeEditor, RoadItem *parentRoadItem, ShapeSection *shapeSection)
    : SectionItem(parentRoadItem, shapeSection)
    , shapeEditor_(shapeEditor)
    , shapeSection_(shapeSection)
	, shapeSectionPolynomialItems_(NULL)
{
    // Init //
    //
    init();
}

ShapeSectionItem::~ShapeSectionItem()
{
}

void
ShapeSectionItem::init()
{
	if (shapeSection_->isElementSelected())
	{
		shapeSectionPolynomialItems_ = new ShapeSectionPolynomialItems(getProfileGraph(), shapeSection_);
		shapeEditor_->addSelectedShapeSection(shapeSection_);
	}



    // Selection/Hovering //
    //
    setAcceptHoverEvents(true);
    setSelectable();

    // Color & Path //
    //
    updateColor();
    createPath();
}

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to the road shape.
*/
void
ShapeSectionItem::updateColor()
{

	int degree = shapeSection_->getShapesMaxDegree();
    if (degree == 3)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkRed()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (degree == 2)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkOrange()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (degree == 1)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkGreen()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (degree == 0)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightCyan()));
        setPen(QPen(ODD::instance()->colors()->darkCyan()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkCyan()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else
    {
        setBrush(QBrush(ODD::instance()->colors()->brightBlue()));
        setPen(QPen(ODD::instance()->colors()->darkBlue()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkBlue()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
}

void
ShapeSectionItem::createPath()
{
    RSystemElementRoad *road = shapeSection_->getParentRoad();

    // Initialization //
    //
    double sStart = shapeSection_->getSStart();
    double sEnd = shapeSection_->getSEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->HeightEditorPointsPerMeter;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    if (pointCount < 2)
    {
        pointCount = 2;
    }
    QVector<QPointF> points(2 * pointCount + 1);
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    // Right side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sStart + i * segmentLength; // [sStart, sEnd]
        points[i] = road->getGlobalPoint(s, road->getMinWidth(s));
    }

    // Left side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sEnd - i * segmentLength; // [sEnd, sStart]
        if (s < 0.0)
            s = 0.0; // can happen due to numerical inaccuracy (around -1.0e-15)
        points[i + pointCount] = road->getGlobalPoint(s, road->getMaxWidth(s));
    }

    // End point //
    //
    points[2 * pointCount] = road->getGlobalPoint(sStart, road->getMinWidth(sStart));

    // Psycho-Path //
    //
    QPainterPath path;
    path.addPolygon(QPolygonF(points));

    setPath(path);
}

//################//
// OBSERVER       //
//################//

void
ShapeSectionItem::updateObserver()
{
    // Parent //
    //
    SectionItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ShapeSection //
    //
    int changes = shapeSection_->getShapeSectionChanges();

    if (changes & ShapeSection::CSS_ParameterChange)
    {
        // Change of the road coordinate s or degree //
        //
		updateColor();
        createPath();
    }


	// DataElement //
	//
	int dataElementChanges = shapeSection_->getDataElementChanges();
	if ((dataElementChanges & DataElement::CDE_SelectionChange)
		|| (dataElementChanges & DataElement::CDE_ChildSelectionChange))
	{
		// Selection //
		//
		if (shapeSection_->isElementSelected())
		{
			shapeEditor_->addSelectedShapeSection(shapeSection_);
		}
		else
		{
			shapeEditor_->delSelectedShapeSection(shapeSection_);
		}
	}
}

//################//
// SLOTS          //
//################//

bool
ShapeSectionItem::removeSection()
{
    RemoveShapeSectionCommand *command = new RemoveShapeSectionCommand(shapeSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
ShapeSectionItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = shapeEditor_->getCurrentTool();
    if (tool == ODD::TRS_SELECT)
    {
        // parent: selection //
        SectionItem::mousePressEvent(event); // pass to baseclass
    }
    else
    {
        return; // prevent selection by doing nothing
    }
}

void
ShapeSectionItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = shapeEditor_->getCurrentTool();
    if (tool == ODD::TRS_SELECT)
    {
        // parent: selection //
        SectionItem::mouseReleaseEvent(event); // pass to baseclass
    }
    else if (tool == ODD::TRS_ADD && (event->button() == Qt::LeftButton))
    {
  //      if (shapeSection_->getDegree() <= 1) // only lines can be split
        {
            // New Line //
            //
            RSystemElementRoad *road = shapeSection_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->pos(), shapeSection_->getSStart(), shapeSection_->getSEnd());

            // Split Section //
            //
            SplitShapeSectionCommand *command = new SplitShapeSectionCommand(shapeSection_, s, NULL);
            if (command->isValid())
            {
                getProjectData()->getUndoStack()->push(command);
            }
            else
            {
                delete command;
            }
        } 
    }
    else if (tool == ODD::TRS_DEL && (event->button() == Qt::LeftButton))
    {
        removeSection();
    }
}

void
ShapeSectionItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = shapeEditor_->getCurrentTool();
    if (tool == ODD::TRS_SELECT)
    {
        // parent: selection //
        SectionItem::mouseMoveEvent(event); // pass to baseclass
    }
    else if (tool == ODD::TRS_ADD)
    {
        // does nothing //
    }
    else if (tool == ODD::TRS_DEL)
    {
        // does nothing //
    }
}

void
ShapeSectionItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = shapeEditor_->getCurrentTool();
    if (tool == ODD::TRS_SELECT)
    {
        setCursor(Qt::PointingHandCursor);
    }
    else if (tool == ODD::TRS_ADD)
    {
        setCursor(Qt::CrossCursor);
        shapeEditor_->getInsertSectionHandle()->show();
    }
    else if (tool == ODD::TRS_DEL)
    {
        setCursor(Qt::CrossCursor);
    }

    // Text //
    //
    getParentRoadItem()->getTextItem()->setVisible(true);
    getParentRoadItem()->getTextItem()->setPos(event->scenePos());

    // Parent //
    //
    SectionItem::hoverEnterEvent(event); // pass to baseclass
}

void
ShapeSectionItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = shapeEditor_->getCurrentTool();
    if (tool == ODD::TRS_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TRS_ADD)
    {
        shapeEditor_->getInsertSectionHandle()->hide();
    }
    else if (tool == ODD::TRS_DEL)
    {
        // does nothing //
    }

    setCursor(Qt::ArrowCursor);

    // Text //
    //
    getParentRoadItem()->getTextItem()->setVisible(false);

    // Parent //
    //
    SectionItem::hoverLeaveEvent(event); // pass to baseclass
}

void
ShapeSectionItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = shapeEditor_->getCurrentTool();
    if (tool == ODD::TRS_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TRS_ADD)
    {
        shapeEditor_->getInsertSectionHandle()->updatePos(parentRoadItem_, event->scenePos(), shapeSection_->getSStart(), shapeSection_->getSEnd());
    }
    else if (tool == ODD::TRS_DEL)
    {
        // does nothing //
    }

    // Parent //
    //
    SectionItem::hoverMoveEvent(event);
}

//*************//
// Delete Item
//*************//

bool
ShapeSectionItem::deleteRequest()
{
    if (removeSection())
        return true;
    else
        return false;
}
