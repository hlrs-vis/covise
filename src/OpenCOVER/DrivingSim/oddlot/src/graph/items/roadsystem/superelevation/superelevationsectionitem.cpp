/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#include "superelevationsectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/superelevationsectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadtextitem.hpp"

// Editor //
//
#include "src/graph/editors/superelevationeditor.hpp"

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

SuperelevationSectionItem::SuperelevationSectionItem(SuperelevationEditor *superelevationEditor, RoadItem *parentRoadItem, SuperelevationSection *superelevationSection)
    : SectionItem(parentRoadItem, superelevationSection)
    , superelevationEditor_(superelevationEditor)
    , superelevationSection_(superelevationSection)
{
    // Init //
    //
    init();
}

SuperelevationSectionItem::~SuperelevationSectionItem()
{
}

void
SuperelevationSectionItem::init()
{
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

/*! \brief Sets the color according to the road superelevation.
*/
void
SuperelevationSectionItem::updateColor()
{
    if (superelevationSection_->getDegree() == 3)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkRed()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (superelevationSection_->getDegree() == 2)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkOrange()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (superelevationSection_->getDegree() == 1)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkGreen()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (superelevationSection_->getDegree() == 0)
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
SuperelevationSectionItem::createPath()
{
    RSystemElementRoad *road = superelevationSection_->getParentRoad();

    // Initialization //
    //
    double sStart = superelevationSection_->getSStart();
    double sEnd = superelevationSection_->getSEnd();
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
SuperelevationSectionItem::updateObserver()
{
    // Parent //
    //
    SectionItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // SuperelevationSection //
    //
    int changes = superelevationSection_->getSuperelevationSectionChanges();

    if (changes & SuperelevationSection::CSE_ParameterChange)
    {
        // Change of the road coordinate s //
        //
        createPath();
        updateColor();
    }
}

//################//
// SLOTS          //
//################//

bool
SuperelevationSectionItem::removeSection()
{
    RemoveSuperelevationSectionCommand *command = new RemoveSuperelevationSectionCommand(superelevationSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
SuperelevationSectionItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = superelevationEditor_->getCurrentTool();
    if (tool == ODD::TSE_SELECT)
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
SuperelevationSectionItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = superelevationEditor_->getCurrentTool();
    if (tool == ODD::TSE_SELECT)
    {
        // parent: selection //
        SectionItem::mouseReleaseEvent(event); // pass to baseclass
    }
    else if (tool == ODD::TSE_ADD && (event->button() == Qt::LeftButton))
    {
        if (superelevationSection_->getDegree() <= 1) // only lines can be split
        {
            // New Line //
            //
            RSystemElementRoad *road = superelevationSection_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->pos(), superelevationSection_->getSStart(), superelevationSection_->getSEnd());

            // Split Section //
            //
            SplitSuperelevationSectionCommand *command = new SplitSuperelevationSectionCommand(superelevationSection_, s, NULL);
            getProjectGraph()->executeCommand(command);
        }
    }
    else if (tool == ODD::TSE_DEL && (event->button() == Qt::LeftButton))
    {
        removeSection();
    }
}

void
SuperelevationSectionItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = superelevationEditor_->getCurrentTool();
    if (tool == ODD::TSE_SELECT)
    {
        // parent: selection //
        SectionItem::mouseMoveEvent(event); // pass to baseclass
    }
    else if (tool == ODD::TSE_ADD)
    {
        // does nothing //
    }
    else if (tool == ODD::TSE_DEL)
    {
        // does nothing //
    }
}

void
SuperelevationSectionItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = superelevationEditor_->getCurrentTool();
    if (tool == ODD::TSE_SELECT)
    {
        setCursor(Qt::PointingHandCursor);
    }
    else if (tool == ODD::TSE_ADD)
    {
        setCursor(Qt::CrossCursor);
        superelevationEditor_->getInsertSectionHandle()->show();
    }
    else if (tool == ODD::TSE_DEL)
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
SuperelevationSectionItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = superelevationEditor_->getCurrentTool();
    if (tool == ODD::TSE_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TSE_ADD)
    {
        superelevationEditor_->getInsertSectionHandle()->hide();
    }
    else if (tool == ODD::TSE_DEL)
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
SuperelevationSectionItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = superelevationEditor_->getCurrentTool();
    if (tool == ODD::TSE_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TSE_ADD)
    {
        superelevationEditor_->getInsertSectionHandle()->updatePos(parentRoadItem_, event->scenePos(), superelevationSection_->getSStart(), superelevationSection_->getSEnd());
    }
    else if (tool == ODD::TSE_DEL)
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
SuperelevationSectionItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
