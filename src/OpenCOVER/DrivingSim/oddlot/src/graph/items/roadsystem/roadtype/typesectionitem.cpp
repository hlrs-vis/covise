/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.03.2010
**
**************************************************************************/

#include "typesectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/commands/typesectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadtextitem.hpp"

// Editor //
//
#include "src/graph/editors/roadtypeeditor.hpp"

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

TypeSectionItem::TypeSectionItem(RoadTypeEditor *typeEditor, RoadItem *parentRoadItem, TypeSection *typeSection)
    : SectionItem(parentRoadItem, typeSection)
    , typeEditor_(typeEditor)
    , typeSection_(typeSection)
{
    init();
}

TypeSectionItem::~TypeSectionItem()
{
}

void
TypeSectionItem::init()
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
// TOOLS          //
//################//

void
TypeSectionItem::changeRoadType(TypeSection::RoadType roadType)
{
    SetTypeTypeSectionCommand *command = new SetTypeTypeSectionCommand(typeSection_, roadType, NULL);
    getProjectGraph()->executeCommand(command);
}

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to the road type
* \todo colors (global table)
*/
void
TypeSectionItem::updateColor()
{
    TypeSection::RoadType type = typeSection_->getRoadType();
    if (type == TypeSection::RTP_MOTORWAY)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkRed()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (type == TypeSection::RTP_RURAL)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkOrange()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (type == TypeSection::RTP_TOWN)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkGreen()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (type == TypeSection::RTP_LOWSPEED)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightCyan()));
        setPen(QPen(ODD::instance()->colors()->darkCyan()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkCyan()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (type == TypeSection::RTP_PEDESTRIAN)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightBlue()));
        setPen(QPen(ODD::instance()->colors()->darkBlue()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkBlue()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else if (type == TypeSection::RTP_UNKNOWN)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
        setPen(QPen(ODD::instance()->colors()->darkGrey()));
        sectionHandle_->setBrush(QBrush(ODD::instance()->colors()->darkGrey()));
        sectionHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else
    {
        setBrush(QBrush(QColor(255, 0, 0)));
        qDebug("WARNING 1003311120! Unknown TypeSection RoadType!");
    }
}

void
TypeSectionItem::createPath()
{
    RSystemElementRoad *road = typeSection_->getParentRoad();

    // Initialization //
    //
    double sStart = typeSection_->getSStart();
    double sEnd = typeSection_->getSEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = this->getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
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
// SLOTS          //
//################//

bool
TypeSectionItem::removeSection()
{
    DeleteTypeSectionCommand *command = new DeleteTypeSectionCommand(typeSection_->getParentRoad(), typeSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

//################//
// OBSERVER       //
//################//

void
TypeSectionItem::updateObserver()
{
    // Parent //
    //
    SectionItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // TypeSection //
    //
    int changes = typeSection_->getTypeSectionChanges();

    if (changes & TypeSection::CTS_TypeChange)
    {
        // Change of the road coordinate s //
        //
        updateColor();
    }
}

//################//
// EVENTS         //
//################//

void
TypeSectionItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = typeEditor_->getCurrentTool();
    if (tool == ODD::TRT_SELECT || tool == ODD::TRT_MOVE)
    {
        // Parent: selection //
        //
        return SectionItem::mousePressEvent(event);
    }
    else
    {
        return; // prevent selection by doing nothing
    }
}

void
TypeSectionItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = typeEditor_->getCurrentTool();
    if (tool == ODD::TRT_SELECT || tool == ODD::TRT_MOVE)
    {
        // Parent: selection //
        //
        SectionItem::mouseReleaseEvent(event);
    }

    else if (tool == ODD::TRT_ADD)
    {
        // Add new TypeSection //
        //
        RSystemElementRoad *road = typeSection_->getParentRoad();
        double s = road->getSFromGlobalPoint(event->scenePos(), typeSection_->getSStart(), typeSection_->getSEnd());
        TypeSection *newTypeSection = new TypeSection(s, typeEditor_->getCurrentRoadType());

        AddTypeSectionCommand *command = new AddTypeSectionCommand(road, newTypeSection, NULL);
        getProjectGraph()->executeCommand(command);
    }

    else if (tool == ODD::TRT_DEL)
    {
        removeSection();
    }
}

void
TypeSectionItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = typeEditor_->getCurrentTool();
    if (tool == ODD::TRT_SELECT || tool == ODD::TRT_MOVE)
    {
        // Parent: selection //
        //
        SectionItem::mouseMoveEvent(event); // pass to baseclass
    }
    else if (tool == ODD::TRT_ADD)
    {
        // does nothing //
    }
    else if (tool == ODD::TRT_DEL)
    {
        // does nothing //
    }
}

void
TypeSectionItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = typeEditor_->getCurrentTool();
    if (tool == ODD::TRT_SELECT || tool == ODD::TRT_MOVE)
    {
        setCursor(Qt::PointingHandCursor);
    }
    else if (tool == ODD::TRT_ADD)
    {
        setCursor(Qt::CrossCursor);
        typeEditor_->getInsertSectionHandle()->show();
    }
    else if (tool == ODD::TRT_DEL)
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
TypeSectionItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = typeEditor_->getCurrentTool();
    if (tool == ODD::TRT_SELECT || tool == ODD::TRT_MOVE)
    {
        // does nothing //
    }
    else if (tool == ODD::TRT_ADD)
    {
        typeEditor_->getInsertSectionHandle()->hide();
    }
    else if (tool == ODD::TRT_DEL)
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
TypeSectionItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = typeEditor_->getCurrentTool();
    if (tool == ODD::TRT_SELECT || tool == ODD::TRT_MOVE)
    {
        // does nothing //
    }
    else if (tool == ODD::TRT_ADD)
    {
        typeEditor_->getInsertSectionHandle()->updatePos(parentRoadItem_, event->scenePos(), typeSection_->getSStart(), typeSection_->getSEnd());
    }
    else if (tool == ODD::TRT_DEL)
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
TypeSectionItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}