/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/29/2010
**
**************************************************************************/

#include "lanelinkitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"

// Graph //
//
#include "laneitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanesectionitem.hpp"
#include "src/graph/items/handles/linkhandle.hpp"

// Editor //
//
//#include "src/graph/editors/

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

#include <QMessageBox>

#include "math.h"

#define LINKHANDLESIZE 2.0

//################//
// CONSTRUCTOR    //
//################//

LaneLinkItem::LaneLinkItem(LaneItem *laneItem, Lane::D_LaneLinkType linkType)
    : GraphElement(laneItem, laneItem->getLane())
    , laneItem_(laneItem)
    , linkType_(linkType)
    , lane_(laneItem->getLane())
    , laneSection_(NULL)
    , road_(NULL)
    , linkedLaneSection_(NULL)
    , linkedRoad_(NULL)
{
    laneSection_ = laneItem_->getParentLaneSectionItem()->getLaneSection();
    road_ = laneSection_->getParentRoad();

    if (linkType_ == Lane::DLLT_Predecessor)
    {
        linkedLaneSection_ = road_->getLaneSectionBefore(laneSection_->getSStart());
        qDebug() << linkedLaneSection_;
    }
    else
    {
    }

    init();
}

LaneLinkItem::~LaneLinkItem()
{
    // Observer Pattern //
    //
    road_->detachObserver(this);
}

void
LaneLinkItem::init()
{
    // Observer Pattern //
    //
    road_->attachObserver(this);

    // Selection/Hovering //
    //
    //	setAcceptHoverEvents(true);
    //	setSelectable();

    // LinkHandle //
    //
    linkHandle_ = new LinkHandle(this);

    // Color & Path //
    //
    updateColor();
    createPath();
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
LaneLinkItem::updateColor()
{
    //	Lane::LaneType type = lane_->getLaneType();
    //	if(type == Lane::LT_DRIVING)
    //	{
    //		setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
    //		setPen(QPen(ODD::instance()->colors()->darkGreen()));
    //	}
    //	else if((type == Lane::LT_MWYENTRY) || (type == Lane::LT_MWYEXIT))
    //	{
    //		setBrush(QBrush(ODD::instance()->colors()->brightCyan()));
    //		setPen(QPen(ODD::instance()->colors()->darkCyan()));
    //	}
    //	else if((type == Lane::LT_SHOULDER) || (type == Lane::LT_PARKING))
    //	{
    //		setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
    //		setPen(QPen(ODD::instance()->colors()->darkOrange()));
    //	}
    //	else if((type == Lane::LT_BORDER) || (type == Lane::LT_RESTRICTED) || (type == Lane::LT_SIDEWALK) || (type == Lane::LT_STOP))
    //	{
    //		setBrush(QBrush(ODD::instance()->colors()->brightRed()));
    //		setPen(QPen(ODD::instance()->colors()->darkRed()));
    //	}
    //	else if((type == Lane::LT_BIKING) || (type == Lane::LT_SPECIAL1) || (type == Lane::LT_SPECIAL2) || (type == Lane::LT_SPECIAL3))
    //	{
    //		setBrush(QBrush(ODD::instance()->colors()->brightBlue()));
    //		setPen(QPen(ODD::instance()->colors()->darkBlue()));
    //	}
    //	else if(type == Lane::LT_NONE)
    //	{
    setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
    setPen(QPen(ODD::instance()->colors()->darkGrey()));
    linkHandle_->setBrush(QBrush(ODD::instance()->colors()->brightGrey()));
    linkHandle_->setPen(QPen(ODD::instance()->colors()->darkGrey()));
    //	}
    //	else
    //	{
    //		setBrush(QBrush(QColor(255, 0, 0)));
    //		qDebug("WARNING 1010181018! Unknown Lane Type!");
    //	}
}

void
LaneLinkItem::createPath()
{

    double s = 0.0;

    if (linkType_ == Lane::DLLT_Predecessor)
    {
        s = laneSection_->getSStart() + 1.0;
    }
    else
    {
        s = road_->getLength() - 1.0;
    }

    int laneId = lane_->getId();

    double d = laneSection_->getLaneSpanWidth(0, laneId, s) + road_->getLaneOffset(s) - 0.5 * laneSection_->getLaneWidth(laneId, s);

    QPointF pos = road_->getGlobalPoint(s, d);

    linkHandle_->setPos(pos.x(), pos.y());
    linkHandle_->setRotation(road_->getGlobalHeading(s));

    // Path //
    //
    QPainterPath path;

    path.moveTo(pos.x(), pos.y());

    setPath(path);
}

//################//
// OBSERVER       //
//################//

void
LaneLinkItem::updateObserver()
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
    if ((changes & Lane::CLN_WidthsChanged))
    {
        createPath();
    }

    if ((linkType_ == Lane::DLLT_Predecessor) && (changes & Lane::CLN_PredecessorChanged))
    {
        createPath();
    }

    if ((linkType_ == Lane::DLLT_Successor) && (changes & Lane::CLN_SuccessorChanged))
    {
        createPath();
    }

    // Road //
    //
    int roadChanges = road_->getRoadChanges();
    if ((roadChanges & RSystemElementRoad::CRD_TrackSectionChange)
        || (roadChanges & RSystemElementRoad::CRD_LaneSectionChange)
        || (roadChanges & RSystemElementRoad::CRD_ShapeChange))
    {
        createPath();
    }

    if (linkedRoad_)
    {
        int roadChanges = linkedRoad_->getRoadChanges();
        if ((roadChanges & RSystemElementRoad::CRD_TrackSectionChange)
            || (roadChanges & RSystemElementRoad::CRD_LaneSectionChange)
            || (roadChanges & RSystemElementRoad::CRD_ShapeChange))
        {
            createPath();
        }
    }
}
