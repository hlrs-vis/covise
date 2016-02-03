/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#include "oscroaditem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/tunnelobject.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//

#include "src/graph/items/roadsystem/roadtextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscsignalitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscobjectitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscbridgeitem.hpp"
#include "src/graph/items/roadsystem/signal/signalhandle.hpp"


// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

OSCRoadItem::OSCRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , roadSystemItem_(roadSystemItem)
    , road_(road)
{
    init();
}

OSCRoadItem::~OSCRoadItem()
{
}

void
OSCRoadItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
	setFlag(QGraphicsItem::ItemIsSelectable, false);

    foreach (Signal *signal, road_->getSignals())
    {
        new OSCSignalItem(roadSystemItem_, signal, road_->getGlobalPoint(signal->getSStart(), signal->getT()));
    }

    foreach (Object *object, road_->getObjects())
    {
        new OSCObjectItem(roadSystemItem_, object, road_->getGlobalPoint(object->getSStart(), object->getT()));
    }

    foreach (Bridge *bridge, road_->getBridges())
    {
        new OSCBridgeItem(roadSystemItem_, bridge, road_->getGlobalPoint(bridge->getSStart(), 0.0));
    }

    updateColor();
    createPath();
}
/*! \brief Sets the color according to the number of links.
*/
void
OSCRoadItem::updateColor()
{

	setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
    setPen(QPen(ODD::instance()->colors()->brightGrey()));
}

void
OSCRoadItem::createPath()
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
}

//################//
// EVENTS         //
//################//

void
OSCRoadItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	// Parent //
	//
	RoadItem::hoverEnterEvent(event); // pass to baseclass

	
}

void
OSCRoadItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{

	// Parent //
	//
	RoadItem::hoverLeaveEvent(event); // pass to baseclass

	
}

void
OSCRoadItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
	// Parent //
	//
	RoadItem::hoverMoveEvent(event);

	
}

