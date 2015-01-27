/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#include "roadlinkroaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/roadlink.hpp"

// Items //
//
#include "roadlinkroadsystemitem.hpp"
#include "roadlinkitem.hpp"
#include "roadlinksinkitem.hpp"

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

RoadLinkRoadItem::RoadLinkRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , predecessor_(NULL)
    , successor_(NULL)
{
    init();
}

RoadLinkRoadItem::~RoadLinkRoadItem()
{
}

void
RoadLinkRoadItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();

    // Context Menu //
    //
    QAction *hideParentRoadAction = getHideMenu()->addAction(tr("Road"));
    connect(hideParentRoadAction, SIGNAL(triggered()), this, SLOT(hideRoad()));

    QAction *removeRoadAction = getRemoveMenu()->addAction(tr("Road"));
    connect(removeRoadAction, SIGNAL(triggered()), this, SLOT(removeRoad()));

    QAction *removeRoadLinkAction = getRemoveMenu()->addAction(tr("Links"));
    connect(removeRoadLinkAction, SIGNAL(triggered()), this, SLOT(removeRoadLink()));

    // Sink items //
    //
    startSinkItem_ = new RoadLinkSinkItem(this, true);
    endSinkItem_ = new RoadLinkSinkItem(this, false);

    // RoadLinks //
    //
    updatePredecessor();
    updateSuccessor();

    // Color and Path //
    //
    updateColor();
    createPath();
}

void
RoadLinkRoadItem::updatePredecessor()
{
    if (getRoad()->getPredecessor())
    {
        predecessor_ = new RoadLinkItem(this, getRoad()->getPredecessor()); // old one deletes itself
    }
    else
    {
        predecessor_ = new RoadLinkItem(this, RoadLink::DRL_PREDECESSOR);
    }
    predecessor_->updateTransform();
}

void
RoadLinkRoadItem::updateSuccessor()
{
    if (getRoad()->getSuccessor())
    {
        successor_ = new RoadLinkItem(this, getRoad()->getSuccessor());
    }
    else
    {
        successor_ = new RoadLinkItem(this, RoadLink::DRL_SUCCESSOR);
    }
    successor_->updateTransform();
}

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to the number of links.
*/
void
RoadLinkRoadItem::updateColor()
{
    if ((getRoad()->getPredecessor() && !getRoad()->getPredecessor()->isLinkValid())
        || (getRoad()->getSuccessor() && !getRoad()->getSuccessor()->isLinkValid()))
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
    }
    else if (getRoad()->getPredecessor() && getRoad()->getSuccessor())
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
    }
    else
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
    }
}

void
RoadLinkRoadItem::createPath()
{
    RSystemElementRoad *road = getRoad();

    // Initialization //
    //
    double sStart = 0.0;
    double sEnd = road->getLength();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
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

    // RoadLink //
    //
    if (predecessor_)
    {
        predecessor_->updateTransform();
    }

    if (successor_)
    {
        successor_->updateTransform();
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
RoadLinkRoadItem::updateObserver()
{
    // Parent //
    //
    RoadItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Road //
    //
    bool recreatePath = false;

    int changes = getRoad()->getRoadChanges();
    if ((changes & RSystemElementRoad::CRD_TrackSectionChange)
        || (changes & RSystemElementRoad::CRD_LaneSectionChange)
        || (changes & RSystemElementRoad::CRD_ShapeChange))
    {
        recreatePath = true;
    }

    if ((changes & RSystemElementRoad::CRD_PredecessorChange))
    {
        updatePredecessor();
        updateColor();
        //recreatePath = true;
    }

    if ((changes & RSystemElementRoad::CRD_SuccessorChange))
    {
        updateSuccessor();
        updateColor();
        //recreatePath = true;
    }

    if (recreatePath)
    {
        createPath();
    }
}
