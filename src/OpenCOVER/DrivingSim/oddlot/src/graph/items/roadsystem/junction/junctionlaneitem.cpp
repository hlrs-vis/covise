/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/18/2010
**
**************************************************************************/

#include "junctionlaneitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"

// Graph //
//
#include "junctionlanesectionitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanelinkitem.hpp"

// Editor //
//
//#include "src/graph/editors/

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

#include <QMessageBox>

#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

JunctionLaneItem::JunctionLaneItem(JunctionLaneSectionItem *parentJunctionLaneSectionItem, Lane *lane)
    : GraphElement(parentJunctionLaneSectionItem, lane)
    , parentJunctionLaneSectionItem_(parentJunctionLaneSectionItem)
    , parentLaneSection_(parentJunctionLaneSectionItem->getLaneSection())
    , lane_(lane)
{
    grandparentRoad_ = parentLaneSection_->getParentRoad();

    init();
}

JunctionLaneItem::~JunctionLaneItem()
{
    // Observer Pattern //
    //
    grandparentRoad_->detachObserver(this);
}

void
JunctionLaneItem::init()
{
    // Observer Pattern //
    //
    grandparentRoad_->attachObserver(this);

    // Selection/Hovering //
    //
    //	setAcceptHoverEvents(true);
    setSelectable();

    // Color & Path //
    //
    updateColor();
    createPath();

    // LaneLinkItem //
    //
    //	new LaneLinkItem(this, Lane::DLLT_Predecessor);
    //	new LaneLinkItem(this, Lane::DLLT_Successor);

    // ContextMenu //
    //
    QAction *hideSectionAction = getHideMenu()->addAction(tr("Lane"));
    connect(hideSectionAction, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

    QAction *hideParentRoadAction = getHideMenu()->addAction(tr("Road"));
    connect(hideParentRoadAction, SIGNAL(triggered()), this, SLOT(hideParentRoad()));

    //	removeSectionAction = getRemoveMenu()->addAction(tr("Section"));
    //	connect(removeSectionAction, SIGNAL(triggered()), this, SLOT(removeSection()));

    QAction *removeParentRoadAction = getRemoveMenu()->addAction(tr("Road"));
    connect(removeParentRoadAction, SIGNAL(triggered()), this, SLOT(removeParentRoad()));
}

//################//
// SLOTS          //
//################//

void
JunctionLaneItem::hideParentRoad()
{
    HideDataElementCommand *command = new HideDataElementCommand(grandparentRoad_, NULL);
    if (getProjectGraph())
    {
        getProjectGraph()->postponeGarbageDisposal();
        getProjectGraph()->executeCommand(command);
    }
    else
    {
        qDebug("JunctionLaneItem::removeRoad() not yet supported for profile graph!");
    }
}

//void
//	SectionItem
//	::removeSection()
//{
//}

void
JunctionLaneItem::removeParentRoad()
{
    RemoveRoadCommand *command = new RemoveRoadCommand(grandparentRoad_, NULL);
    if (getProjectGraph())
    {
        getProjectGraph()->executeCommand(command);
    }
    else
    {
        qDebug("JunctionLaneItem::removeRoad() not yet supported for profile graph!");
    }
}

//################//
// TOOLS          //
//################//

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to ...
*/
void
JunctionLaneItem::updateColor()
{
    Lane::LaneType type = lane_->getLaneType();

    if (type == Lane::LT_DRIVING)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
    }
    else if ((type == Lane::LT_MWYENTRY) || (type == Lane::LT_MWYEXIT))
    {
        setBrush(QBrush(ODD::instance()->colors()->brightCyan()));
        setPen(QPen(ODD::instance()->colors()->darkCyan()));
    }
    else if ((type == Lane::LT_SHOULDER) || (type == Lane::LT_PARKING))
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
    }
    else if ((type == Lane::LT_BORDER) || (type == Lane::LT_RESTRICTED) || (type == Lane::LT_SIDEWALK) || (type == Lane::LT_STOP))
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
    }
    else if ((type == Lane::LT_BIKING) || (type == Lane::LT_SPECIAL1) || (type == Lane::LT_SPECIAL2) || (type == Lane::LT_SPECIAL3))
    {
        setBrush(QBrush(ODD::instance()->colors()->brightBlue()));
        setPen(QPen(ODD::instance()->colors()->darkBlue()));
    }
    else if (type == Lane::LT_NONE)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
        setPen(QPen(ODD::instance()->colors()->darkGrey()));
    }
    else
    {
        setBrush(QBrush(QColor(255, 0, 0)));
        qDebug("WARNING 1010181018! Unknown Lane Type!");
    }

    if (grandparentRoad_->getJunction() != "-1")
    {
        setPen(QPen(ODD::instance()->colors()->darkPurple()));
    }
}

void
JunctionLaneItem::createPath()
{
    RSystemElementRoad *road = parentLaneSection_->getParentRoad();

    // Initialization //
    //
    double sStart = parentLaneSection_->getSStart();
    double sEnd = parentLaneSection_->getSEnd();
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

    int laneSide = 1;
    if (lane_->getId() < 0)
    {
        laneSide = -1;
    }

    // Right side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sStart + i * segmentLength; // [sStart, sEnd]
        points[i] = road->getGlobalPoint(s, laneSide * parentLaneSection_->getLaneSpanWidth(0, lane_->getId() - laneSide, s));
    }

    // Left side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sEnd - i * segmentLength; // [sEnd, sStart]
        if (s < 0.0)
            s = 0.0; // can happen due to numerical inaccuracy (around -1.0e-15)
        points[i + pointCount] = road->getGlobalPoint(s, laneSide * parentLaneSection_->getLaneSpanWidth(0, lane_->getId(), s));
    }

    // End point //
    //
    points[2 * pointCount] = road->getGlobalPoint(sStart, laneSide * parentLaneSection_->getLaneSpanWidth(0, lane_->getId() - laneSide, sStart));

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
JunctionLaneItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Lane //
    //
    int changes = lane_->getLaneChanges();
    if ((changes & Lane::CLN_TypeChanged))
    {
        updateColor();
    }
    else if ((changes & Lane::CLN_WidthsChanged))
    {
        createPath();
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
	else if (roadChanges & RSystemElementRoad::CRD_JunctionChange)
	{
		updateColor();
	}

}

//################//
// EVENTS         //
//################//

//void
//	LaneItem
//	::mousePressEvent(QGraphicsSceneMouseEvent * event)
//{
//}

void
JunctionLaneItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    parentJunctionLaneSectionItem_->mouseReleaseEvent(event);
}

//void
//	LaneItem
//	::mouseMoveEvent(QGraphicsSceneMouseEvent * event)
//{

//}

//*************//
// Delete Item
//*************//

bool
JunctionLaneItem::deleteRequest()
{
    return parentJunctionLaneSectionItem_->deleteRequest();
}
