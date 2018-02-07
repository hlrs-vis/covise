/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/22/2010
**
**************************************************************************/

#include "roadmarkitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roadmark/roadmarklaneitem.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "math.h"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/lodsettings.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

//################//
// CONSTRUCTOR    //
//################//

RoadMarkItem::RoadMarkItem(RoadMarkLaneItem *parentLaneItem, LaneRoadMark *roadMark)
    : GraphElement(parentLaneItem, roadMark)
    , parentLaneItem_(parentLaneItem)
    , roadMark_(roadMark)
    , lastRoadMarkType_(roadMark_->getRoadMarkType())
{
    parentLane_ = roadMark->getParentLane();
    parentLaneSection_ = parentLane_->getParentLaneSection();
    parentRoad_ = parentLaneSection_->getParentRoad();

    init();
}

RoadMarkItem::~RoadMarkItem()
{
    // Observer Pattern //
    //
    parentRoad_->detachObserver(this);
}

void
RoadMarkItem::init()
{
    // Observer Pattern //
    //
    parentRoad_->attachObserver(this);

    // Hover Events //
    //
    setAcceptHoverEvents(false);
    setFlag(QGraphicsItem::ItemIgnoresParentOpacity, true);
    setOpacitySettings(1.0, 1.0); // ...always highlighted

    // Color & Path //
    //
    updateColor();
    createPath();
    updatePen();
}

//################//
// GRAPHICS       //
//################//

void
RoadMarkItem::updateColor()
{
    QPen thePen = pen();

    LaneRoadMark::RoadMarkColor color = roadMark_->getRoadMarkColor();
    if (color == LaneRoadMark::RMC_STANDARD)
    {
        thePen.setColor(ODD::instance()->colors()->darkGrey());
    }
    else if (color == LaneRoadMark::RMC_YELLOW)
    {
        thePen.setColor(ODD::instance()->colors()->darkOrange());
    }

    setPen(thePen);
}

void
RoadMarkItem::createPath()
{
    QPainterPath path;

    // Check some stuff //
    //
    LaneRoadMark::RoadMarkType type = roadMark_->getRoadMarkType();
    double width = roadMark_->getRoadMarkWidth();
    if ((type == LaneRoadMark::RMT_NONE)
        || (width <= 0.0))
    {
        setPath(path);
        return;
    }

    // RoadMarks //
    //
    double sStart = parentLaneSection_->getSStart() + roadMark_->getSSectionStart();
    double sEnd = parentLaneSection_->getSStart() + roadMark_->getSSectionEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    if (pointCount < 2)
    {
        setPath(path);
        return;
    }
    QVector<QPointF> points(pointCount);
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    int id = parentLane_->getId();
    if (id == 0)
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            points[i] = parentRoad_->getGlobalPoint(s);
        }
    }
    else if (id < 0)
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            points[i] = parentRoad_->getGlobalPoint(s, -parentLaneSection_->getLaneSpanWidth(0, id, s) + parentRoad_->getLaneOffset(s));
        }
    }
    else
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            points[i] = parentRoad_->getGlobalPoint(s, parentLaneSection_->getLaneSpanWidth(0, id, s) + parentRoad_->getLaneOffset(s));
        }
    }

    // Path //
    //
    path.addPolygon(QPolygonF(points));

    setPath(path);
}

void
RoadMarkItem::updatePen()
{
    QPen thePen = pen();

    // Weight //
    //

    // TODO

    // Width //
    //
    double width = roadMark_->getRoadMarkWidth();
    if ((width >= 0.0))
    {
        thePen.setWidthF(width);
    }
    else
    {
        createPath();
    }

    // Type //
    //
    LaneRoadMark::RoadMarkType type = roadMark_->getRoadMarkType();
    if ((type == LaneRoadMark::RMT_NONE)
        || (lastRoadMarkType_ == LaneRoadMark::RMT_NONE))
    {
        createPath();
    }
    else if (type == LaneRoadMark::RMT_BROKEN)
    {
        // TODO: no hard coding
        const QVector<qreal> pattern(2, 3.0 / width); // hard coded 3.0m for town
        // init with 2 values
        // 0: length of the line, 1: length of the free space
        // actual length will be multiplied with the width
        thePen.setDashPattern(pattern);
    }
    else if (type == LaneRoadMark::RMT_SOLID)
    {
        thePen.setStyle(Qt::SolidLine);
    }

    lastRoadMarkType_ = type;
    setPen(thePen);
}

//################//
// EVENTS         //
//################//

void
RoadMarkItem::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    event->ignore(); // ignores the event so it goes to the next item in line
}

//################//
// OBSERVER       //
//################//

void
RoadMarkItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Road //
    //
    int roadChanges = parentLaneSection_->getParentRoad()->getRoadChanges();
    if ((roadChanges & RSystemElementRoad::CRD_TrackSectionChange)
        || (roadChanges & RSystemElementRoad::CRD_LaneSectionChange)
        || (roadChanges & RSystemElementRoad::CRD_ShapeChange))
    {
        createPath();
    }

    // RoadMark //
    //
    int changes = roadMark_->getRoadMarkChanges();
    if ((changes & LaneRoadMark::CLR_TypeChanged)
        || (changes & LaneRoadMark::CLR_WidthChanged)
        || (changes & LaneRoadMark::CLR_WeightChanged))
    {
        updatePen();
    }

    if ((changes & LaneRoadMark::CLR_ColorChanged))
    {
        updateColor();
    }
}
