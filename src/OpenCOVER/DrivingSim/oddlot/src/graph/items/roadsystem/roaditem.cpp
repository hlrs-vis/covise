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

#include "roaditem.hpp"

#include "src/util/odd.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "roadsystemitem.hpp"
#include "roadmark/roadmarklanesectionitem.hpp"

#include "src/graph/items/roadsystem/roadtextitem.hpp"

// TODO:
#include "src/graph/profilegraph.hpp"

RoadItem::RoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : GraphElement(roadSystemItem, road)
    , road_(road)
    , roadSystemItem_(roadSystemItem)
{
    init();
}

RoadItem::~RoadItem()
{
}

void
RoadItem::init()
{
    roadSystemItem_->appendRoadItem(this);

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        textItem_ = new RoadTextItem(this);
        textItem_->setZValue(1.0); // stack before siblings

        // RoadMarks //
        //
        foreach (LaneSection *section, getRoad()->getLaneSections())
        {
            if (section->getSStart() < getRoad()->getLength())
            {
                (new RoadMarkLaneSectionItem(this, section))->setZValue(1.0);
            }
        }
    }

    // Context Menu
    //
    QAction *addToTileAction = getContextMenu()->addAction("Add to current tile");
    connect(addToTileAction, SIGNAL(triggered()), this, SLOT(addToCurrentTile()));
}

/*! \brief Called when the item has been moved to the garbage.
*
*/
void
RoadItem::notifyDeletion()
{
    roadSystemItem_->removeRoadItem(this);
    GraphElement::notifyDeletion();
}

//################//
// SLOTS          //
//################//

bool
RoadItem::removeRoad()
{
    RemoveRoadCommand *command = new RemoveRoadCommand(road_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
RoadItem::hideRoad()
{
    HideDataElementCommand *command = new HideDataElementCommand(road_, NULL);
    getProjectGraph()->executeCommand(command);
}

bool
RoadItem::removeRoadLink()
{
    RemoveRoadLinkCommand *command = new RemoveRoadLinkCommand(road_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
RoadItem::addToCurrentTile()
{
    QStringList parts = road_->getID().split("_");
    if (parts.at(0) != getProjectData()->getTileSystem()->getCurrentTile()->getID())
    {
        QString name = road_->getName();
        QString newId = road_->getRoadSystem()->getUniqueId("", name);
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(road_->getRoadSystem(), road_, newId, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

//##################//
// Observer Pattern //
//##################//

void
RoadItem::updateObserver()
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
    int changes = road_->getRoadChanges();

    //	if((changes & RSystemElementRoad::CRD_LengthChange)
    //		|| (changes & RSystemElementRoad::CRD_ShapeChange)
    //		|| (changes & RSystemElementRoad::CRD_TrackSectionChange)
    //	)
    //	{
    //	}

    //	if((changes & RSystemElementRoad::CRD_TypeSectionChange )
    //		|| (changes & RSystemElementRoad::CRD_TrackSectionChange)
    //		|| (changes & RSystemElementRoad::CRD_ElevationSectionChange)
    //		|| (changes & RSystemElementRoad::CRD_SuperelevationSectionChange)
    //		|| (changes & RSystemElementRoad::CRD_CrossfallSectionChange)
    //		|| (changes & RSystemElementRoad::CRD_LaneSectionChange)
    //		)
    //	{
    //		// A RoadSection has been added or deleted. Let the child classes decide if it should be refreshed or not!
    //		//
    //		refreshRoadSectionItems();
    //	}

    if (changes & RSystemElementRoad::CRD_LaneSectionChange)
    {
        // A section has been added.
        //
        foreach (LaneSection *section, getRoad()->getLaneSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                if (getTopviewGraph()) // not for profile graph
                {
                    if (section->getSStart() < getRoad()->getLength())
                    {
                        (new RoadMarkLaneSectionItem(this, section))->setZValue(1.0);
                    }
                }
            }
        }
    }
}

//*************//
// Delete Item
//*************//

bool
RoadItem::deleteRequest()
{
    if (removeRoad())
    {
        return true;
    }

    return false;
}

//################//
// EVENTS         //
//################//

void
RoadItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    textItem_->setPos(event->scenePos());
    textItem_->setVisible(true);

    GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
RoadItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    textItem_->setVisible(false);

    GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
RoadItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    GraphElement::mousePressEvent(event); // pass to baseclass
}
