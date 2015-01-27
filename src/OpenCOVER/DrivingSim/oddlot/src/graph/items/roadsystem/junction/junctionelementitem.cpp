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

#include "junctionelementitem.hpp"

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
#include "junctionroaditem.hpp"

// Editor //
//
#include "src/graph/editors/junctioneditor.hpp"

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

JunctionElementItem::JunctionElementItem(JunctionRoadItem *parentJunctionRoadItem, TrackElement *trackElement)
    : JunctionComponentItem(parentJunctionRoadItem, trackElement)
    , trackElement_(trackElement)
{
    // Init //
    //
    // NOTE: this init is called after the base class init!
    // Be carefull with calling virtual functions in constructors. Ask Scott Meyers why!
    init();
}

JunctionElementItem::JunctionElementItem(JunctionComponentItem *parentJunctionComponentItem, TrackElement *trackElement)
    : JunctionComponentItem(parentJunctionComponentItem, trackElement)
    , trackElement_(trackElement)
{
    // Init //
    //
    init();
}

JunctionElementItem::~JunctionElementItem()
{
}

void
JunctionElementItem::init()
{
    // Selection/Highlighting //
    //
    setAcceptHoverEvents(true);

    if (getParentJunctionComponentItem())
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

    if (getParentJunctionComponentItem())
    {
        QAction *hideParentTrackComponentAction = getHideMenu()->addAction(tr("Group"));
        connect(hideParentTrackComponentAction, SIGNAL(triggered()), this, SLOT(hideParentTrackComponent()));

        QAction *ungroupAction = getContextMenu()->addAction("Ungroup");
        connect(ungroupAction, SIGNAL(triggered()), getParentJunctionComponentItem(), SLOT(ungroupComposite()));
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
JunctionElementItem::updateColor()
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
JunctionElementItem::createPath()
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
JunctionElementItem::updateObserver()
{
    // Parent //
    //
    JunctionComponentItem::updateObserver();
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
JunctionElementItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //
    if (change == QGraphicsItem::ItemPositionChange)
    {
        return pos();
    }

    return JunctionComponentItem::itemChange(change, value);
}

void
JunctionElementItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        ODD::ToolId tool = getJunctionEditor()->getCurrentTool();
        if (tool == ODD::TTE_ROAD_DELETE)
        {
            return; // does nothing
        }
        else if (tool == ODD::TTE_MOVE)
        {
            pressPos_ = event->scenePos();
        }
        else if (tool == ODD::TTE_DELETE)
        {
            return; // does nothing
        }
        else if (tool == ODD::TJE_SPLIT)
        {
            return; // does nothing
        }
    }

    // parent: selection //
    JunctionComponentItem::mousePressEvent(event); // pass to baseclass
}

void
JunctionElementItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = getJunctionEditor()->getCurrentTool();
    if (tool == ODD::TTE_ROAD_DELETE)
    {
        return; // does nothing
    }
    else if (tool == ODD::TTE_DELETE)
    {
        return; // does nothing
    }
    else if (tool == ODD::TTE_MOVE)
    {
        getJunctionEditor()->translateTrack(trackElement_, pressPos_, event->scenePos());
        pressPos_ = event->scenePos();
        return; // does nothing
    }

    // parent: selection //
    JunctionComponentItem::mouseMoveEvent(event); // pass to baseclass
}

void
JunctionElementItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        ODD::ToolId tool = getJunctionEditor()->getCurrentTool();
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
        else if (tool == ODD::TJE_SPLIT)
        {
            RSystemElementRoad *road = trackElement_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->scenePos(), trackElement_->getSStart(), trackElement_->getSEnd());

            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Split Track and Road"));

            // Split Track //
            //
            SplitTrackComponentCommand *splitTrackCommand = new SplitTrackComponentCommand(trackElement_, s, NULL);
            getProjectGraph()->executeCommand(splitTrackCommand);

            // Split Road //
            //
            SplitRoadCommand *splitRoadCommand = new SplitRoadCommand(road, s, NULL);
            getProjectGraph()->executeCommand(splitRoadCommand);

            getProjectData()->getUndoStack()->endMacro();
            return;
        }
        //		else if(tool == ODD::TTE_MOVE)
        //		{
        //			return; // does nothing
        //		}
    }

    // parent: selection //
    JunctionComponentItem::mouseReleaseEvent(event); // pass to baseclass
}

void
JunctionElementItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = getJunctionEditor()->getCurrentTool();
    if (tool == ODD::TJE_SPLIT)
    {
        getJunctionEditor()->getSectionHandle()->show();
    }

    // Text //
    //
    getParentJunctionRoadItem()->getTextItem()->setVisible(true);
    getParentJunctionRoadItem()->getTextItem()->setPos(event->scenePos());

    JunctionComponentItem::hoverEnterEvent(event); // pass to baseclass
}

void
JunctionElementItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = getJunctionEditor()->getCurrentTool();
    if (tool == ODD::TJE_SPLIT)
    {
        getJunctionEditor()->getSectionHandle()->updatePos(getParentJunctionRoadItem(), event->scenePos(), trackElement_->getSStart(), trackElement_->getSEnd());
    }

    // Parent //
    //
    JunctionComponentItem::hoverMoveEvent(event);
}

void
JunctionElementItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = getJunctionEditor()->getCurrentTool();
    if (tool == ODD::TJE_SPLIT)
    {
        getJunctionEditor()->getSectionHandle()->hide();
    }

    setCursor(Qt::ArrowCursor);

    // Text //
    //
    getParentJunctionRoadItem()->getTextItem()->setVisible(false);

    // Parent //
    //
    JunctionComponentItem::hoverLeaveEvent(event); // pass to baseclass
}

//*************//
// Delete Item
//*************//

bool
JunctionElementItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
