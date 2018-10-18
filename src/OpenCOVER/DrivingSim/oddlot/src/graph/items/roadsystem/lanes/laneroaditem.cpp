/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/15/2010
**
**************************************************************************/

#include "laneroaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/laneborder.hpp"

// Items //
//
#include "laneroadsystemitem.hpp"
#include "lanesectionitem.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/items/handles/lanemovehandle.hpp"

// Editor //
//
#include "src/graph/editors/laneeditor.hpp"

LaneRoadItem::LaneRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
	: RoadItem(roadSystemItem, road)
	, road_(road)
{

	init();
}

LaneRoadItem::~LaneRoadItem()
{
}

void
LaneRoadItem::init()
{
	// ElevationEditor //
	//
	laneEditor_ = dynamic_cast<LaneEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
	if (!laneEditor_)
	{
		qDebug("Warning 1006241105! ElevationRoadItem not created by an ElevationEditor");
	}
	// SectionItems //
	//
	foreach(LaneSection *section, getRoad()->getLaneSections())
	{
		laneSectionItems_.insert(section, new LaneSectionItem(laneEditor_, this, section));
	}

	rebuildMoveRotateHandles(false);
}

// SectionItems //
//
void
LaneRoadItem::addSectionItem(LaneSectionItem *item)
{
	laneSectionItems_.insert(item->getLaneSection(), item);
}

int
LaneRoadItem::removeSectionItem(LaneSectionItem *item)
{
	return laneSectionItems_.remove(item->getLaneSection());
}

LaneSectionItem *
LaneRoadItem::getSectionItem(LaneSection *section)
{
	return laneSectionItems_.value(section);
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
void
LaneRoadItem::rebuildMoveRotateHandles(bool delHandles)
{

	// Move Handles are build per road. Adjacent lanes share a handle, adjacent roads not. //
	//
	
	if (delHandles)
	{
		deleteHandles();
	}


	handlesItem_ = new QGraphicsPathItem(this);
	handlesItem_->setZValue(1.0); // Stack handles before items

	int leftmostLaneId = 0;
	int rightmostLaneId = 0;
	foreach(LaneSection *laneSection, road_->getLaneSections())
	{
		if (laneSection->getLeftmostLaneId() > leftmostLaneId)
		{
			leftmostLaneId = laneSection->getLeftmostLaneId();
		}

		if (laneSection->getRightmostLaneId() < rightmostLaneId)
		{
			rightmostLaneId = laneSection->getRightmostLaneId();
		}
	}


	bool newHandle = true;
	for (int id = leftmostLaneId; id >= rightmostLaneId; id--)
	{

		LaneSection *laneSection;
		Lane *lane;
		foreach(laneSection, road_->getLaneSections())
		{
			lane = laneSection->getLane(id);
			if (lane)
			{
				break;
			}
		}

		LaneMoveHandle<LaneWidth, LaneWidth> *currentWWMoveHandle = NULL;
		LaneMoveHandle<LaneWidth, LaneBorder> *currentWBMoveHandle = NULL;
		LaneMoveHandle<LaneBorder, LaneWidth> *currentBWMoveHandle = NULL;
		LaneMoveHandle<LaneBorder, LaneBorder> *currentBBMoveHandle = NULL;
		bool lastWidth;

		while (laneSection)
		{
			if (lane)
			{
				if (lane->isWidthActive())
				{
					QMap<double, LaneWidth *> widthEntries = lane->getWidthEntries();
					if (widthEntries.isEmpty())
					{
						laneSection = road_->getLaneSectionNext(laneSection->getSStart());
						if (laneSection)
						{
							lane = laneSection->getLane(id);
						}
						newHandle = true;
						continue;
					}

					QMap<double, LaneWidth *>::const_iterator it = widthEntries.constBegin();

					while (it != widthEntries.constEnd())
					{
						LaneWidth *laneWidth = it.value();

						if (newHandle)
						{
							currentWWMoveHandle = new LaneMoveHandle<LaneWidth, LaneWidth>(laneEditor_, handlesItem_);
							currentWWMoveHandle->registerHighSlot(laneWidth);

							newHandle = false;
						}
						else if (!lastWidth)
						{
							currentBWMoveHandle->registerHighSlot(laneWidth);
						}
						else
						{
							currentWWMoveHandle->registerHighSlot(laneWidth); // last handle
						}

						if (++it == widthEntries.constEnd())
						{
							laneSection = road_->getLaneSectionNext(laneSection->getSStart());
							if (laneSection)
							{
								lane = laneSection->getLane(id);
								if (lane)
								{
									if (lane->getPredecessor() != -99)
									{
										if (lane->isWidthActive())
										{
											currentWWMoveHandle = new LaneMoveHandle<LaneWidth, LaneWidth>(laneEditor_, handlesItem_); // new handle
											currentWWMoveHandle->registerLowSlot(laneWidth);
										}
										else
										{
											currentWBMoveHandle = new LaneMoveHandle<LaneWidth, LaneBorder>(laneEditor_, handlesItem_); // new handle
											currentWBMoveHandle->registerLowSlot(laneWidth);
										}
										lastWidth = true;
										continue;
									}
								}
							}

							newHandle = true;
						}

						currentWWMoveHandle = new LaneMoveHandle<LaneWidth, LaneWidth>(laneEditor_, handlesItem_); // new handle
						currentWWMoveHandle->registerLowSlot(laneWidth); // new handle
						lastWidth = true;
					}
				}
				else
				{
					QMap<double, LaneBorder *> borderEntries = lane->getBorderEntries();
					if (borderEntries.isEmpty())
					{
						laneSection = road_->getLaneSectionNext(laneSection->getSStart());
						if (laneSection)
						{
							lane = laneSection->getLane(id);
						}
						newHandle = true;
						continue;
					}

					QMap<double, LaneBorder *>::const_iterator it = borderEntries.constBegin();
					while (it != borderEntries.constEnd())
					{
						LaneBorder *laneBorder = it.value();

						if (newHandle)
						{
							currentBBMoveHandle = new LaneMoveHandle<LaneBorder, LaneBorder>(laneEditor_, handlesItem_);
							currentBBMoveHandle->registerHighSlot(laneBorder);
							newHandle = false;
						}
						else if (lastWidth)
						{
							currentWBMoveHandle->registerHighSlot(laneBorder);
						}
						else
						{
							currentBBMoveHandle->registerHighSlot(laneBorder); // last handle
						}

						if (++it == borderEntries.constEnd())
						{
							laneSection = road_->getLaneSectionNext(laneSection->getSStart());
							if (laneSection)
							{
								lane = laneSection->getLane(id);
								if (lane)
								{
									if (lane->getPredecessor() != -99)
									{
										if (lane->isWidthActive())
										{
											currentBWMoveHandle = new LaneMoveHandle<LaneBorder, LaneWidth>(laneEditor_, handlesItem_); // new handle
											currentBWMoveHandle->registerLowSlot(laneBorder);
										}
										else
										{
											currentBBMoveHandle = new LaneMoveHandle<LaneBorder, LaneBorder>(laneEditor_, handlesItem_); // new handle
											currentBBMoveHandle->registerLowSlot(laneBorder);
										}
										lastWidth = false;

										continue;
									}
								}
							}

							newHandle = true;
						}

						currentBBMoveHandle = new LaneMoveHandle<LaneBorder, LaneBorder>(laneEditor_, handlesItem_); // new handle
						currentBBMoveHandle->registerLowSlot(laneBorder); // new handle
						lastWidth = false;
					}
				}
			}
		} 
	} 
	
}




/*! \brief .
*
*/
void
LaneRoadItem::deleteHandles()
{
	//	delete handlesItem_;
	if (handlesItem_ != NULL)
	{
		if (laneEditor_)
		{
			laneEditor_->getTopviewGraph()->getScene()->removeItem(handlesItem_);
		}

		handlesItem_->setParentItem(NULL);
		getProjectGraph()->addToGarbage(handlesItem_);
		handlesItem_ = NULL;
	}
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
LaneRoadItem::updateObserver()
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
    int changes = getRoad()->getRoadChanges();

    if (changes & RSystemElementRoad::CRD_LaneSectionChange)
    {
        // A section has been added.
        //
        foreach (LaneSection *section, getRoad()->getLaneSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
				laneSectionItems_.insert(section, new LaneSectionItem(laneEditor_, this, section));
            }
        }
		rebuildMoveRotateHandles(true);
    }


	
}
