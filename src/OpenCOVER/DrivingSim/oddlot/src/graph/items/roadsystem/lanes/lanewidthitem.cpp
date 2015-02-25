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

#include "lanewidthitem.hpp"
#include "lanesectionwidthitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"

// Graph //
//
#include "lanesectionitem.hpp"
#include "lanelinkitem.hpp"

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

LaneWidthItem::LaneWidthItem(LaneSectionWidthItem *parentLaneSectionWidthItem, LaneWidth *laneWidth)
    : GraphElement(parentLaneSectionWidthItem, laneWidth)
    , parentLaneSectionWidthItem_(parentLaneSectionWidthItem)
    , parentLane_(parentLaneSectionWidthItem_->getLane())
    , laneWidth_(laneWidth)
{

    init();
}

LaneWidthItem::~LaneWidthItem()
{
    // Observer Pattern //
    //
}

void
LaneWidthItem::init()
{
    // Observer Pattern //
    //

    // Selection/Hovering //
    //
    //setAcceptHoverEvents(true);
    //setSelectable();

    // Color & Path //
    //
    updateColor();
    createPath();

    // LaneLinkItem //
    //

    // ContextMenu //
    //
    /*	QAction * hideSectionAction = getHideMenu()->addAction(tr("Lane"));
	connect(hideSectionAction, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

	QAction * hideParentRoadAction = getHideMenu()->addAction(tr("Road"));
	connect(hideParentRoadAction, SIGNAL(triggered()), this, SLOT(hideParentRoad()));

//	removeSectionAction = getRemoveMenu()->addAction(tr("Section"));
//	connect(removeSectionAction, SIGNAL(triggered()), this, SLOT(removeSection()));

	QAction * removeParentRoadAction = getRemoveMenu()->addAction(tr("Road"));
	connect(removeParentRoadAction, SIGNAL(triggered()), this, SLOT(removeParentRoad()));*/
}

//################//
// SLOTS          //
//################//

//################//
// TOOLS          //
//################//

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to ...
*/
void
LaneWidthItem::updateColor()
{
    Lane::LaneType type = parentLane_->getLaneType();
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
}

void
LaneWidthItem::createPath()
{

    // Initialization //
    //
    double sStart = laneWidth_->getSSectionStart() + laneWidth_->getParentLane()->getParentLaneSection()->getSStart();
    double sEnd = laneWidth_->getSSectionEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    QVector<QPointF> points(pointCount);
    double segmentLength = (sEnd - sStart) / (pointCount);

    // Right side //
    //

    for (int i = 0; i < pointCount; ++i)
    {
        double s = i * segmentLength; // [sStart, sEnd]
        points[i] = QPointF(sStart + s, parentLane_->getWidth(s));
    }

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
LaneWidthItem::updateObserver()
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
    int changes = parentLane_->getLaneChanges();
    if ((changes & Lane::CLN_TypeChanged))
    {
        updateColor();
    }

    {
        createPath();
    }
}

//################//
// EVENTS         //
//################//

//void
//	LaneWidthItem
//	::mousePressEvent(QGraphicsSceneMouseEvent * event)
//{
//}

/*void
	LaneWidthItem
	::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
parentLaneSectionItem_->mouseReleaseEvent(event);
}
*/
//void
//	LaneWidthItem
//	::mouseMoveEvent(QGraphicsSceneMouseEvent * event)
//{

//}
