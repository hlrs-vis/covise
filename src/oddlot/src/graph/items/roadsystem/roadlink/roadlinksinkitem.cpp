/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   11/19/2010
 **
 **************************************************************************/

#include "roadlinksinkitem.hpp"

 // Data //
 //
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/roadlink.hpp"

// Graph //
//
#include "src/graph/editors/roadlinkeditor.hpp"

// Items //
//
#include "roadlinkroaditem.hpp"
#include "roadlinkhandle.hpp"
#include "roadlinksinkhandle.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

#define DISTANCE 2.0

RoadLinkSinkItem::RoadLinkSinkItem(RoadLinkRoadItem *parent, RoadLinkEditor *editor, bool isStart)
    : GraphElement(parent, parent->getRoad())
    , // observes parent road
    isStart_(isStart)
    , parentRoadItem_(parent)
    , editor_(editor)
    , parentRoad_(parent->getRoad())
    , sinkHandle_(NULL)
{
    init();
}

RoadLinkSinkItem::~RoadLinkSinkItem()
{
}

void
RoadLinkSinkItem::init()
{
    // Color and Path //
    //
    QPen thePen;
    thePen.setWidth(1);
    thePen.setCosmetic(true);
    setPen(thePen);

    QBrush theBrush = QBrush(ODD::instance()->colors()->brightGreen());
    setBrush(theBrush);

    setFlag(QGraphicsItem::ItemIgnoresParentOpacity, true);
 //   setFlag(QGraphicsItem::ItemIsSelectable, true);
    setOpacitySettings(1.0, 1.0); // ...always highlighted

    sinkHandle_ = new RoadLinkSinkHandle(this, editor_);
 //   sinkHandle_->setFlag(QGraphicsItem::ItemIsSelectable, true);
    // sinkHandle_->setPassSelectionToParent(true);

    updateColor();
    createPath();
}

void 
RoadLinkSinkItem::setHandlesSelectable(bool selectable)
{
    sinkHandle_->setFlag(QGraphicsItem::ItemIsSelectable, selectable);
}

//################//
// GRAPHICS       //
//################//

void
RoadLinkSinkItem::updateColor()
{
    QBrush theBrush = brush();
    QPen thePen = pen();

    theBrush.setColor(ODD::instance()->colors()->brightGreen());
    thePen.setColor(ODD::instance()->colors()->darkGreen());

    setBrush(theBrush);
    setPen(thePen);
}

void
RoadLinkSinkItem::createPath()
{
    // Coordinate //
    //
    double s = 0.0;
    if (isStart_)
    {
        s = DISTANCE;
    }
    else
    {
        s = parentRoad_->getLength() - DISTANCE;
    }

    // Transform //
    //
    setPos(parentRoad_->getGlobalPoint(s));
    setRotation(parentRoad_->getGlobalHeading(s));
    if (isStart_)
    {
        sinkHandle_->setPos(0.0, parentRoad_->getMinWidth(s) - DISTANCE);
    }
    else
    {
        sinkHandle_->setPos(0.0, parentRoad_->getMaxWidth(s) + DISTANCE);
    }

    // Path //
    //
    QPainterPath thePath;

    if (isStart_)
    {
        thePath.moveTo(0.0, parentRoad_->getMaxWidth(s) + DISTANCE);
        thePath.lineTo(0.0, parentRoad_->getMinWidth(s) - DISTANCE);
    }
    else
    {
        thePath.moveTo(0.0, parentRoad_->getMaxWidth(s) + DISTANCE);
        thePath.lineTo(0.0, parentRoad_->getMinWidth(s) - DISTANCE);
    }

    setPath(thePath);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
RoadLinkSinkItem::updateObserver()
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
    int changes = parentRoad_->getRoadChanges();
    if ((changes & RSystemElementRoad::CRD_ShapeChange)
        || (changes & RSystemElementRoad::CRD_LengthChange)
        || (changes & RSystemElementRoad::CRD_TrackSectionChange))
    {
        createPath();
    }

}
