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

#include "junctionroaditem.hpp"

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
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "junctionroadsystemitem.hpp"

#include "junctionelementitem.hpp"
#include "junctionsparcsitem.hpp"

#include "junctionmovehandle.hpp"
#include "junctionaddhandle.hpp"

// Editor //
//
#include "src/graph/editors/junctioneditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

JunctionRoadItem::JunctionRoadItem(JunctionRoadSystemItem *parentJunctionRoadSystemItem, RSystemElementRoad *road)
    : RoadItem(parentJunctionRoadSystemItem, road)
    , junctionEditor_(NULL)
    , parentJunctionRoadSystemItem_(parentJunctionRoadSystemItem)
    , handlesItem_(NULL)
{
    init();
}

JunctionRoadItem::~JunctionRoadItem()
{
    //deleteHandles(); // already deleted when moved to garbage
}

void
JunctionRoadItem::init()
{
    // JunctionEditor //
    //
    junctionEditor_ = dynamic_cast<JunctionEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!junctionEditor_)
    {
        qDebug("Warning 1007041414! JunctionRoadItem not created by an JunctionEditor");
    }

    // Parent //
    //
    parentJunctionRoadSystemItem_->addRoadItem(this);

    // JunctionSection Items //
    //
    rebuildSections(true);
}

void
JunctionRoadItem::rebuildSections(bool fullRebuild)
{

    foreach (TrackComponent *track, getRoad()->getTrackSections())
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
                new JunctionSpArcSItem(this, sparcs);
            }
            else if ((track->getTrackType() == TrackComponent::DTT_LINE)
                     || (track->getTrackType() == TrackComponent::DTT_ARC)
                     || (track->getTrackType() == TrackComponent::DTT_SPIRAL)
                     || (track->getTrackType() == TrackComponent::DTT_POLY3))
            {
                TrackElement *trackElement = dynamic_cast<TrackElement *>(track);
                new JunctionElementItem(this, trackElement);
            }
            else
            {
                qDebug("WARNING 1007051616! JunctionRoadItem::rebuildSections() Unknown JunctionType.");
            }
        }

        // Handles //
        //
        if ((junctionEditor_->isCurrentTool(ODD::TTE_ADD))
            || (junctionEditor_->isCurrentTool(ODD::TTE_ADD_CURVE))
            || (junctionEditor_->isCurrentTool(ODD::TTE_ADD_LINE)))
        {
            if (fullRebuild) // the handles need not be rebuilt every time since they adjust their position automatically
            {
                rebuildAddHandles();
            }
        }
        else if (junctionEditor_->isCurrentTool(ODD::TTE_MOVE))
        {
            rebuildMoveHandles();
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
JunctionRoadItem::notifyDeletion()
{
    // Parent //
    //
    parentJunctionRoadSystemItem_->removeRoadItem(this);

    deleteHandles();
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
void
JunctionRoadItem::rebuildMoveHandles()
{
    deleteHandles();
    handlesItem_ = new QGraphicsPathItem(this);
    handlesItem_->setZValue(1.0); // Stack handles before items

    JunctionMoveHandle *currentJunctionMoveHandle = new JunctionMoveHandle(junctionEditor_, handlesItem_); // first handle
    foreach (TrackComponent *track, getRoad()->getTrackSections())
    {
        currentJunctionMoveHandle->registerHighSlot(track); // last handle
        currentJunctionMoveHandle = new JunctionMoveHandle(junctionEditor_, handlesItem_); // new handle
        currentJunctionMoveHandle->registerLowSlot(track); // new handle
    }
}

/*! \brief .
*
*/
void
JunctionRoadItem::rebuildAddHandles()
{
    deleteHandles();
    handlesItem_ = new QGraphicsPathItem(this);
    handlesItem_->setZValue(1.0); // Stack handles before items

    new JunctionAddHandle(junctionEditor_, handlesItem_, getRoad(), true);
    new JunctionAddHandle(junctionEditor_, handlesItem_, getRoad(), false);
}

/*! \brief .
*
*/
void
JunctionRoadItem::deleteHandles()
{
    //delete handlesItem_;
    getProjectGraph()->addToGarbage(handlesItem_);
    handlesItem_ = NULL;
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
JunctionRoadItem::updateObserver()
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
    }
}

//################//
// EVENTS         //
//################//

///*!
//* Handles Item Changes.
//*/
//QVariant
//	JunctionRoadItem
//	::itemChange(GraphicsItemChange change, const QVariant & value)
//{
//	return RoadItem::itemChange(change, value);
//}
