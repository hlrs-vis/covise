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

#include "sectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/roadsection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "sectionhandle.hpp"

// Qt //
//
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

//################//
// CONSTRUCTOR    //
//################//

SectionItem::SectionItem(RoadItem *parentRoadItem, RoadSection *roadSection)
    : GraphElement(parentRoadItem, roadSection)
    , parentRoadItem_(parentRoadItem)
    , sectionHandle_(NULL)
    , roadSection_(roadSection)
    , road_(NULL)
{
    init();
}

SectionItem::~SectionItem()
{
    // Observer Pattern //
    //
    road_->detachObserver(this);
}

void
SectionItem::init()
{
    road_ = roadSection_->getParentRoad();

    // Observer Pattern //
    //
    road_->attachObserver(this);

    // SectionHandle (Start) //
    //
    if (getTopviewGraph()) // not for profilegraph
    {
        sectionHandle_ = new SectionHandle(this);
        sectionHandle_->setPos(road_->getGlobalPoint(roadSection_->getSStart()));
        sectionHandle_->setRotation(road_->getGlobalHeading(roadSection_->getSStart()));
    }

    // ContextMenu //
    //
    hideSectionAction_ = getHideMenu()->addAction(tr("Section"));
    connect(hideSectionAction_, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

    hideParentRoadAction_ = getHideMenu()->addAction(tr("Road"));
    connect(hideParentRoadAction_, SIGNAL(triggered()), this, SLOT(hideParentRoad()));

    removeSectionAction_ = getRemoveMenu()->addAction(tr("Section"));
    connect(removeSectionAction_, SIGNAL(triggered()), this, SLOT(removeSection()));

    removeParentRoadAction_ = getRemoveMenu()->addAction(tr("Road"));
    connect(removeParentRoadAction_, SIGNAL(triggered()), this, SLOT(removeParentRoad()));
}

//################//
// SLOTS          //
//################//

void
SectionItem::hideParentRoad()
{
    if (getParentRoadItem())
    {
        getParentRoadItem()->hideRoad();
    }
}

void
SectionItem::hideRoads()
{
    hideParentRoad();
}

/*
bool
	SectionItem
	::removeSection()
{
	// does nothing by default - to be implemented by subclasses
}*/

void
SectionItem::removeParentRoad()
{
    if (getParentRoadItem())
    {
        getParentRoadItem()->removeRoad();
    }
}

//################//
// OBSERVER       //
//################//

void
SectionItem::updateObserver()
{
    // Parent first //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Redraw //
    //
    bool recreatePath = false; // create path only once

    // RoadSection //
    //
    int changes = roadSection_->getRoadSectionChanges();
    if (changes & RoadSection::CRS_SChange)
    {
        // Change of the road coordinate s //
        //
        recreatePath = true;
    }
    if (changes & RoadSection::CRS_LengthChange)
    {
        // Change of the length of the section //
        //
        recreatePath = true;
    }

    // Road //
    //
    int roadChanges = roadSection_->getParentRoad()->getRoadChanges();
    if ((roadChanges & RSystemElementRoad::CRD_TrackSectionChange)
        || (roadChanges & RSystemElementRoad::CRD_LaneSectionChange)
        || (roadChanges & RSystemElementRoad::CRD_ShapeChange))
    {
        // Change of the shape //
        //
        recreatePath = true;
    }

    // Redraw //
    //
    if (recreatePath)
    {
        createPath();
        if (sectionHandle_)
        {
            sectionHandle_->updateTransform();
        }
    }
}

//*************//
// Delete Item
//*************//

bool
SectionItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
