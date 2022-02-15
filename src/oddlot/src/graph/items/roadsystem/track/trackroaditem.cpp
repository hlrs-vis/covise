/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   05.07.2010
 **
 **************************************************************************/

#include "trackroaditem.hpp"

 // Data //
 //
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/projectgraph.hpp"
#include "src/graph/graphscene.hpp"

// Items //
//
#include "trackroadsystemitem.hpp"

#include "trackelementitem.hpp"
#include "tracksparcsitem.hpp"

#include "trackmovehandle.hpp"
#include "trackrotatehandle.hpp"
#include "trackaddhandle.hpp"

#include "roadmovehandle.hpp"
#include "roadrotatehandle.hpp"

// Editor //
//
#include "src/graph/editors/trackeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

TrackRoadItem::TrackRoadItem(TrackRoadSystemItem *parentTrackRoadSystemItem, RSystemElementRoad *road, TrackEditor *trackEditor)
    : RoadItem(parentTrackRoadSystemItem, road)
    , trackEditor_(trackEditor)
    , parentTrackRoadSystemItem_(parentTrackRoadSystemItem)
    , handlesItem_(NULL)
    , handlesAddItem_(NULL)
{
    init();
}

TrackRoadItem::~TrackRoadItem()
{
    deleteHandles(); // already deleted when moved to garbage
}

void
TrackRoadItem::init()
{
    // Parent //
    //
    parentTrackRoadSystemItem_->addRoadItem(this);

    // TrackSection Items //
    //
    rebuildSections(true);
}

TopviewGraph *TrackRoadItem::getTopviewGraph() const
{
    if (trackEditor_ != NULL)
        return trackEditor_->getTopviewGraph();
    else
        return GraphElement::getTopviewGraph();
}

void
TrackRoadItem::rebuildSections(bool fullRebuild)
{
    if (fullRebuild)
    {
        trackComponentItems_.clear();
    }
    else
    {
        QMap<TrackComponent *, TrackComponentItem *>::iterator it = trackComponentItems_.begin();
        while (it != trackComponentItems_.end())
        {
            TrackComponent *track = it.key();
            if (track->getDataElementChanges() & DataElement::CDE_DataElementRemoved)
            {
                it = trackComponentItems_.erase(it);
            }
            else
            {
                it++;
            }
        }
    }

    foreach(TrackComponent * track, getRoad()->getTrackSections())
    {

        // TrackItems //
        //
        if ((fullRebuild)
            || (track->getDataElementChanges() & DataElement::CDE_DataElementCreated)
            || (track->getDataElementChanges() & DataElement::CDE_DataElementAdded))
        {
            if (track->getTrackType() == TrackComponent::DTT_SPARCS)
            {
                TrackSpiralArcSpiral *sparcs = dynamic_cast<TrackSpiralArcSpiral *>(track);
                trackComponentItems_.insert(track, new TrackSpArcSItem(this, sparcs, trackEditor_));
            }
            else if ((track->getTrackType() == TrackComponent::DTT_LINE)
                || (track->getTrackType() == TrackComponent::DTT_ARC)
                || (track->getTrackType() == TrackComponent::DTT_SPIRAL)
                || (track->getTrackType() == TrackComponent::DTT_POLY3)
                || (track->getTrackType() == TrackComponent::DTT_CUBICCURVE))
            {
                TrackElement *trackElement = dynamic_cast<TrackElement *>(track);
                trackComponentItems_.insert(track, new TrackElementItem(this, trackElement, trackEditor_));
            }
            else
            {
                qDebug("WARNING 1007051616! TrackRoadItem::rebuildSections() Unknown TrackType.");
            }
        }

        // Handles //
        //
        if ((trackEditor_->isCurrentTool(ODD::TTE_ADD))
            || (trackEditor_->isCurrentTool(ODD::TTE_ADD_PROTO)))
        {
            if (fullRebuild) // the handles need not be rebuilt every time since they adjust their position automatically
            {
                rebuildAddHandles(false);
            }
        }
        else if (trackEditor_->isCurrentTool(ODD::TTE_MOVE_ROTATE))
        {
            rebuildMoveRotateHandles(false);
        }
        else if (trackEditor_->isCurrentTool(ODD::TTE_ROAD_MOVE_ROTATE))
        {
            rebuildRoadMoveRotateHandles(false);
        }
        else
        {
            deleteHandles();
        }
    }
}

/*! \brief Called when the item has been moved to the garbage.
*
*/
void
TrackRoadItem::notifyDeletion()
{
    // Parent //
    //
    parentTrackRoadSystemItem_->removeRoadItem(this);

    deleteHandles();
}

void 
TrackRoadItem::setComponentItemsSelectable(bool selectable)
{
    foreach(TrackComponentItem * item, trackComponentItems_)
    {
        item->setFlag(QGraphicsItem::ItemIsSelectable, selectable);
    }
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
void
TrackRoadItem::rebuildMoveRotateHandles(bool delHandles)
{
    if (delHandles)
    {
        deleteHandles();
    }

    handlesItem_ = new QGraphicsPathItem(this);
    handlesItem_->setZValue(1.0); // Stack handles before items

    TrackMoveHandle *currentTrackMoveHandle = new TrackMoveHandle(trackEditor_, handlesItem_); // first handle
    foreach(TrackComponent * track, getRoad()->getTrackSections())
    {
        currentTrackMoveHandle->registerHighSlot(track); // last handle
        currentTrackMoveHandle = new TrackMoveHandle(trackEditor_, handlesItem_); // new handle
        currentTrackMoveHandle->registerLowSlot(track); // new handle
    }

    TrackRotateHandle *currentTrackRotateHandle = new TrackRotateHandle(trackEditor_, handlesItem_); // first handle
    foreach(TrackComponent * track, getRoad()->getTrackSections())
    {
        currentTrackRotateHandle->registerHighSlot(track); // last handle
        currentTrackRotateHandle = new TrackRotateHandle(trackEditor_, handlesItem_); // new handle
        currentTrackRotateHandle->registerLowSlot(track); // new handle
    }
}

/*! \brief .
*
*/
void
TrackRoadItem::rebuildAddHandles(bool delHandles)
{
    if (delHandles)
    {
        deleteHandles();
    }
    handlesItem_ = new QGraphicsPathItem(this);
    handlesItem_->setZValue(1.0); // Stack handles before items

    new TrackAddHandle(trackEditor_, handlesItem_, getRoad(), true);
    new TrackAddHandle(trackEditor_, handlesItem_, getRoad(), false);
}

/*! \brief .
*
*/
void
TrackRoadItem::rebuildRoadMoveRotateHandles(bool delHandles)
{
    if (delHandles)
    {
        deleteHandles();
    }

    handlesItem_ = new QGraphicsPathItem(this);
    handlesItem_->setZValue(1.0); // Stack handles before items

    new RoadMoveHandle(trackEditor_, getRoad(), handlesItem_);

    new RoadRotateHandle(trackEditor_, getRoad(), handlesItem_);
}

/*! \brief .
*
*/
void
TrackRoadItem::deleteHandles()
{
    // delete handlesItem_;
    if (handlesItem_ != NULL)
    {
        if (trackEditor_)
        {
            trackEditor_->getTopviewGraph()->getScene()->removeItem(handlesItem_);
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
TrackRoadItem::updateObserver()
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

    if (changes & RSystemElementRoad::CRD_TrackSectionChange)
    {
        // A section has been added.
        rebuildSections(false); // no full rebuild, only update
        return;
    }

    changes = getRoad()->getDataElementChanges();

    if (changes & DataElement::CDE_SelectionChange)
    {
        ODD::ToolId tool = trackEditor_->getCurrentTool();
        if (getRoad()->isElementSelected())
        {
            if ((tool == ODD::TTE_ROAD_APPEND) || (tool == ODD::TTE_ROAD_MERGE) || (tool == ODD::TTE_ROAD_SNAP))
            {
                foreach(TrackComponentItem * item, trackComponentItems_)
                {
                    item->setSelected(true);
                }
            }
        }
        else
        {
            if ((tool == ODD::TTE_ROAD_APPEND) || (tool == ODD::TTE_ROAD_MERGE) || (tool == ODD::TTE_ROAD_SNAP))
            {
                foreach(TrackComponentItem * item, trackComponentItems_)
                {
                    item->setSelected(false);
                }
            }
        }
    }


}

//################//
// EVENTS         //
//################//

///*!
//* Handles Item Changes.
//*/
//QVariant
// TrackRoadItem
// ::itemChange(GraphicsItemChange change, const QVariant & value)
//{
// return RoadItem::itemChange(change, value);
//}
