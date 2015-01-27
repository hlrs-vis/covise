/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   17.03.2010
**
**************************************************************************/

#include "roadmarkvisitor.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"

// Graph //
//

// Qt //
//
#include <QtGui>
#include <QGraphicsPathItem>

/*!
*
*/
RoadMarkVisitor::RoadMarkVisitor(ProjectEditor *editor, QGraphicsItem *rootItem)
    : editor_(editor)
    , rootItem_(rootItem)
    , currentRoad_(NULL)
    , currentLaneSection_(NULL)
    , currentLane_(NULL)
{
}

/*!
*
*/
void
RoadMarkVisitor::visit(RoadSystem *system)
{

    // TODO JUNCTIONS???

    system->acceptForChildNodes(this);
}

/*!
*
*/
void
RoadMarkVisitor::visit(RSystemElementRoad *road)
{
    currentRoad_ = road;
    //	currentRoad_->attach(currentRoadItem_); // TODO: observe road

    road->acceptForLaneSections(this);
}

/*!
*
*/
void
RoadMarkVisitor::visit(LaneSection *laneSection)
{
    currentSectionItem_ = new QGraphicsPathItem(rootItem_);

    currentLaneSection_ = laneSection;
    //	currentLaneSection_->attach(currentSectionItem_); // TODO: inherit from observer

    // Visit Lanes //
    //
    laneSection->acceptForLanes(this);
}

/*!
*
*/
void
RoadMarkVisitor::visit(Lane *lane)
{
    currentLane_ = lane;
    lane->acceptForRoadMarks(this);
}

/*!
*
*/
void
RoadMarkVisitor::visit(LaneRoadMark *mark)
{
    // Check some stuff //
    //
    LaneRoadMark::RoadMarkType type = mark->getRoadMarkType();
    if (type == LaneRoadMark::RMT_NONE)
        return;

    double width = mark->getRoadMarkWidth();
    if (width <= 0.0)
        return;

    // RoadMarks //
    //
    double sStart = currentLaneSection_->getSStart() + mark->getSSectionStart();
    double sEnd = currentLaneSection_->getSStart() + mark->getSSectionEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    double pointsPerMeter = 1.0; // BAD: hard coded!
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    QVector<QPointF> points(pointCount);
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    int id = currentLane_->getId();
    if (id == 0)
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            points[i] = currentRoad_->getGlobalPoint(s);
        }
    }
    else if (id < 0)
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            points[i] = currentRoad_->getGlobalPoint(s, -currentLaneSection_->getLaneSpanWidth(0, id, s));
        }
    }
    else
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            points[i] = currentRoad_->getGlobalPoint(s, currentLaneSection_->getLaneSpanWidth(0, id, s));
        }
    }

    QPainterPath path;
    path.addPolygon(QPolygonF(points));

    // TODO: style options enum flags (road marks, etc): a | b | c... if(aIsInIt)
    // user can choose what to display

    QPen pen(QColor(82, 111, 96)); // greyish
    //	QPen pen(QColor(238, 243, 238)); // whitish
    pen.setWidthF(width);

    if (type == LaneRoadMark::RMT_BROKEN)
    {
        // TODO: no hard coding
        const QVector<qreal> pattern(2, 3.0 / width); // hard coded 3.0m for town
        // init with 2 values
        // 0: length of the line, 1: length of the free space
        // actual length will be multiplied with the width
        pen.setDashPattern(pattern);
    }
    // TODO: more types

    // Item //
    //
    QGraphicsPathItem *item = new QGraphicsPathItem(path);
    item->setPen(pen);
    item->setBrush(Qt::NoBrush);
    item->setParentItem(rootItem_);
}
