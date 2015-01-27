/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   06.04.2010
**
**************************************************************************/

#include "trackelementitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/track/trackelement.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/trackcommands.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadtextitem.hpp"
#include "trackroaditem.hpp"

// Editor //
//
#include "src/graph/editors/trackeditor.hpp"

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
//#include <QMessageBox>

// Utils //
//
#include "src/util/colorpalette.hpp"

#include <QDebug>

#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

TrackElementItem::TrackElementItem(TrackRoadItem *parentTrackRoadItem, TrackElement *trackElement)
    : TrackComponentItem(parentTrackRoadItem, trackElement)
    , trackElement_(trackElement)
{
    // Init //
    //
    // NOTE: this init is called after the base class init!
    // Be carefull with calling virtual functions in constructors. Ask Scott Meyers why!
    init();
}

TrackElementItem::TrackElementItem(TrackComponentItem *parentTrackComponentItem, TrackElement *trackElement)
    : TrackComponentItem(parentTrackComponentItem, trackElement)
    , trackElement_(trackElement)
{
    // Init //
    //
    init();
}

TrackElementItem::~TrackElementItem()
{
}

void
TrackElementItem::init()
{
    // Selection/Highlighting //
    //
    setAcceptHoverEvents(true);

    if (getParentTrackComponentItem())
    {
        setFlag(QGraphicsItem::ItemIsMovable, false); // move the whole group
    }
    else
    {
        setFlag(QGraphicsItem::ItemIsMovable, true);
    }

    // Color & Path //
    //
    updateColor();
    createPath();

    // ContextMenu //
    //
    QAction *hideAction = getHideMenu()->addAction(tr("Track"));
    connect(hideAction, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

    if (getParentTrackComponentItem())
    {
        QAction *hideParentTrackComponentAction = getHideMenu()->addAction(tr("Group"));
        connect(hideParentTrackComponentAction, SIGNAL(triggered()), this, SLOT(hideParentTrackComponent()));

        QAction *ungroupAction = getContextMenu()->addAction("Ungroup");
        connect(ungroupAction, SIGNAL(triggered()), getParentTrackComponentItem(), SLOT(ungroupComposite()));
    }

    QAction *hideRoadAction = getHideMenu()->addAction(tr("Road"));
    connect(hideRoadAction, SIGNAL(triggered()), this, SLOT(hideParentRoad()));

    QAction *removeSectionAction = getRemoveMenu()->addAction(tr("Track(s)"));
    connect(removeSectionAction, SIGNAL(triggered()), this, SLOT(removeSection()));

    QAction *removeRoadAction = getRemoveMenu()->addAction(tr("Road"));
    connect(removeRoadAction, SIGNAL(triggered()), this, SLOT(removeParentRoad()));
}

//################//
// GRAPHICS       //
//################//

void
TrackElementItem::updateColor()
{
    if (trackElement_->getTrackType() == TrackComponent::DTT_LINE)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
    }
    else if (trackElement_->getTrackType() == TrackComponent::DTT_ARC)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
    }
    else if (trackElement_->getTrackType() == TrackComponent::DTT_SPIRAL)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
    }
    else if (trackElement_->getTrackType() == TrackComponent::DTT_POLY3)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightCyan()));
        setPen(QPen(ODD::instance()->colors()->darkCyan()));
    }
    else
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
        setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
}

void
TrackElementItem::createPath()
{
    RSystemElementRoad *road = trackElement_->getParentRoad();

    // Initialization //
    //
    double sStart = trackElement_->getSStart() + 0.001; // 1mm gap
    double sEnd = trackElement_->getSEnd() - 0.001; // 1mm gap
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    if (pointCount <= 1)
    {
        pointCount = 2; // should be at least 2 to get a quad
    }
    QVector<QPointF> points(2 * pointCount + 1);
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    // Right side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sStart + i * segmentLength; // [sStart, sEnd]
        if (road)
        {
            points[i] = trackElement_->getPoint(s, road->getMinWidth(s));
        }
        else
        {

            points[i] = trackElement_->getPoint(s, 0.0);
        }
        //		qDebug() << s << " " << points[i] << " " << road->getMinWidth(s);
        if (i > 0 && (points[i] - points[i - 1]).manhattanLength() > 30)
        {
            //			QPointF tmp = trackElement_->getPoint(s, road->getMinWidth(s));
            //			qDebug() << "*************";
        }
    }

    // Left side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sEnd - i * segmentLength; // [sEnd, sStart]
        if (s < 0.0)
            s = 0.0; // can happen due to numerical inaccuracy (around -1.0e-15)
        points[i + pointCount] = trackElement_->getPoint(s, road->getMaxWidth(s));
    }

    // End point //
    //
    points[2 * pointCount] = trackElement_->getPoint(sStart, road->getMinWidth(sStart));

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
TrackElementItem::updateObserver()
{
    // Parent //
    //
    TrackComponentItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
TrackElementItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //
    if (change == QGraphicsItem::ItemPositionChange)
    {
        return pos();
    }

    return TrackComponentItem::itemChange(change, value);
}

void
TrackElementItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        ODD::ToolId tool = getTrackEditor()->getCurrentTool();
        if (tool == ODD::TTE_ROAD_DELETE)
        {
            return; // does nothing
        }
        else if (tool == ODD::TTE_MOVE_ROTATE)
        {
            pressPos_ = event->scenePos();
        }
        else if (tool == ODD::TTE_DELETE)
        {
            return; // does nothing
        }
        else if (tool == ODD::TTE_TRACK_SPLIT)
        {
            return; // does nothing
        }
        else if (tool == ODD::TTE_ROAD_SPLIT)
        {
            return; // does nothing
        }
        else if (tool == ODD::TTE_TRACK_ROAD_SPLIT)
        {
            return; // does nothing
        }
        else if ((tool == ODD::TTE_ROAD_MERGE) || (tool == ODD::TTE_ROAD_SNAP))
        {
            getTrackEditor()->registerRoad(getParentTrackRoadItem()->getRoad());
        }
    }

    // parent: selection //
    TrackComponentItem::mousePressEvent(event); // pass to baseclass
}

void
TrackElementItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = getTrackEditor()->getCurrentTool();
    if (tool == ODD::TTE_ROAD_DELETE)
    {
        return; // does nothing
    }
    else if (tool == ODD::TTE_DELETE)
    {
        return; // does nothing
    }
    else if (tool == ODD::TTE_MOVE_ROTATE)
    {
        getTrackEditor()->translateTrack(trackElement_, pressPos_, event->scenePos());
        pressPos_ = event->scenePos();
        return; // does nothing
    }

    // parent: selection //
    TrackComponentItem::mouseMoveEvent(event); // pass to baseclass
}

void
TrackElementItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        ODD::ToolId tool = getTrackEditor()->getCurrentTool();
        if (tool == ODD::TTE_ROAD_DELETE)
        {
            removeParentRoad();
            return;
        }
        else if (tool == ODD::TTE_DELETE)
        {
            removeSection();
            return;
        }
        else if (tool == ODD::TTE_TRACK_SPLIT)
        {
            RSystemElementRoad *road = trackElement_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->scenePos(), trackElement_->getSStart(), trackElement_->getSEnd());

            // Split Track //
            //
            SplitTrackComponentCommand *command = new SplitTrackComponentCommand(trackElement_, s, NULL);
            getProjectGraph()->executeCommand(command);
            return;
        }
        else if (tool == ODD::TTE_ROAD_SPLIT)
        {
            RSystemElementRoad *road = trackElement_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->scenePos(), trackElement_->getSStart(), trackElement_->getSEnd());

            // Split Road //
            //
            SplitRoadCommand *command = new SplitRoadCommand(road, s, NULL);
            getProjectGraph()->executeCommand(command);
            return;
        }
        else if (tool == ODD::TTE_TRACK_ROAD_SPLIT)
        {
            RSystemElementRoad *road = trackElement_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->scenePos(), trackElement_->getSStart(), trackElement_->getSEnd());

            // Split Road //
            //
            SplitTrackRoadCommand *command = new SplitTrackRoadCommand(road, s, NULL);
            getProjectGraph()->executeCommand(command);
            return;
        }
        //		else if(tool == ODD::TTE_MOVE)
        //		{
        //			return; // does nothing
        //		}
    }

    // parent: selection //
    TrackComponentItem::mouseReleaseEvent(event); // pass to baseclass
}

void
TrackElementItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = getTrackEditor()->getCurrentTool();
    if ((tool == ODD::TTE_TRACK_SPLIT)
        || (tool == ODD::TTE_ROAD_SPLIT)
        || (tool == ODD::TTE_TRACK_ROAD_SPLIT))
    {
        getTrackEditor()->getSectionHandle()->show();
        // Text //
        //
        getParentTrackRoadItem()->getTextItem()->setVisible(false);
    }
    else
    {

        // Text //
        //
        getParentTrackRoadItem()->getTextItem()->setVisible(true);
        getParentTrackRoadItem()->getTextItem()->setPos(event->scenePos());
    }

    TrackComponentItem::hoverEnterEvent(event); // pass to baseclass
}

void
TrackElementItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = getTrackEditor()->getCurrentTool();
    if ((tool == ODD::TTE_TRACK_SPLIT)
        || (tool == ODD::TTE_ROAD_SPLIT)
        || (tool == ODD::TTE_TRACK_ROAD_SPLIT))
    {
        getTrackEditor()->getSectionHandle()->updatePos(getParentTrackRoadItem(), event->scenePos(), trackElement_->getSStart(), trackElement_->getSEnd());
    }

    // Parent //
    //
    TrackComponentItem::hoverMoveEvent(event);
}

void
TrackElementItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = getTrackEditor()->getCurrentTool();
    if ((tool == ODD::TTE_TRACK_SPLIT)
        || (tool == ODD::TTE_ROAD_SPLIT)
        || (tool == ODD::TTE_TRACK_ROAD_SPLIT))
    {
        getTrackEditor()->getSectionHandle()->hide();
    }

    setCursor(Qt::ArrowCursor);

    // Text //
    //
    getParentTrackRoadItem()->getTextItem()->setVisible(false);

    // Parent //
    //
    TrackComponentItem::hoverLeaveEvent(event); // pass to baseclass
}

//*************//
// Delete Item
//*************//

bool
TrackElementItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
