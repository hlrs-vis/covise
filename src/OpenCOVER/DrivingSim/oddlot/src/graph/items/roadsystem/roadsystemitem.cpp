/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   30.03.2010
**
**************************************************************************/

#include "roadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Editor //
//
#include "src/graph/editors/projecteditor.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/profilegraph.hpp"

#include "roaditem.hpp"
#include "junctionitem.hpp"

//################//
// CONSTRUCTOR    //
//################//

RoadSystemItem::RoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : GraphElement(NULL, roadSystem)
    , topviewGraph_(topviewGraph)
    , profileGraph_(NULL)
    , roadSystem_(roadSystem)
{
    init();
}

RoadSystemItem::RoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem)
    : GraphElement(NULL, roadSystem)
    , topviewGraph_(NULL)
    , profileGraph_(profileGraph)
    , roadSystem_(roadSystem)
{
    init();
}

RoadSystemItem::~RoadSystemItem()
{
}

void
RoadSystemItem::init()
{
    // Highlighting //
    //
    setOpacitySettings(1.0, 1.0); // ...always highlighted

    // Junctions //
    //
    if (getTopviewGraph()) // not for profile graph
    {
        foreach (RSystemElementJunction *junction, roadSystem_->getJunctions())
        {
            (new JunctionItem(this, junction))->setZValue(-1.0);
        }
    }
}

void
RoadSystemItem::appendRoadItem(RoadItem *roadItem)
{
    QString id = roadItem->getRoad()->getID();
    if (!roadItems_.contains(id))
    {
        roadItems_.insert(id, roadItem);
    }
}

bool
RoadSystemItem::removeRoadItem(RoadItem *roadItem)
{
    return roadItems_.remove(roadItem->getRoad()->getID());
}

//##################//
// Observer Pattern //
//##################//

void
RoadSystemItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = roadSystem_->getRoadSystemChanges();

    if (changes & RoadSystem::CRS_JunctionChange)
    {
        // Junctions //
        //
        if (getTopviewGraph()) // not for profile graph
        {
            foreach (RSystemElementJunction *junction, roadSystem_->getJunctions())
            {
                (new JunctionItem(this, junction))->setZValue(-1.0);
            }
        }
    }

    //	if((changes & RoadSystem::CRS_RoadChange)
    //	|| (changes & RoadSystem::CRS_FiddleyardChange)
    //	|| (changes & RoadSystem::CRS_JunctionChange)
    //	|| (changes & RoadSystem::CRS_ControllerChange)
    //	)
}
