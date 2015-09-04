/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.04.2010
**
**************************************************************************/

#include "trackcomponentitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackelement.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/trackcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"

// Graph //
//
#include "src/gui/projectwidget.hpp"

#include "src/graph/projectgraph.hpp"

#include "src/graph/items/roadsystem/track/trackroaditem.hpp"
#include "src/graph/items/roadsystem/track/trackelementitem.hpp"
#include "src/graph/items/roadsystem/track/tracksparcsitem.hpp"

// Editor //
//
#include "src/graph/editors/trackeditor.hpp"

//################//
// CONSTRUCTOR    //
//################//

TrackComponentItem::TrackComponentItem(TrackRoadItem *parentTrackRoadItem, TrackComponent *trackComponent)
    : GraphElement(parentTrackRoadItem, trackComponent)
    , trackEditor_(NULL)
    , parentTrackRoadItem_(parentTrackRoadItem)
    , parentTrackComponentItem_(NULL)
    , trackComponent_(trackComponent)
{
    // Init //
    //
    init();
}

TrackComponentItem::TrackComponentItem(TrackComponentItem *parentTrackComponentItem, TrackComponent *trackComponent)
    : GraphElement(parentTrackComponentItem, trackComponent)
    , trackEditor_(NULL)
    , parentTrackRoadItem_(NULL)
    , parentTrackComponentItem_(parentTrackComponentItem)
    , trackComponent_(trackComponent)
{
    parentTrackRoadItem_ = parentTrackComponentItem_->getParentTrackRoadItem();

    // Init //
    //
    init();
}

TrackComponentItem::~TrackComponentItem()
{
}

void
TrackComponentItem::init()
{
    // TrackEditor //
    //
    trackEditor_ = dynamic_cast<TrackEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!trackEditor_)
    {
        qDebug("Warning 1007131313! TrackComponentItem not created by an TrackEditor");
    }

    // TrackItems //
    //
    foreach (TrackComponent *track, trackComponent_->getChildTrackComponents())
    {
        if (track->getTrackType() == TrackComponent::DTT_SPARCS)
        {
            TrackSpiralArcSpiral *sparcs = dynamic_cast<TrackSpiralArcSpiral *>(track);
            new TrackSpArcSItem(this, sparcs);
        }
        else if ((track->getTrackType() == TrackComponent::DTT_LINE)
                 || (track->getTrackType() == TrackComponent::DTT_ARC)
                 || (track->getTrackType() == TrackComponent::DTT_SPIRAL))
        {
            TrackElement *trackElement = dynamic_cast<TrackElement *>(track);
            new TrackElementItem(this, trackElement);
        }
        else
        {
            qDebug("WARNING 1007051646! TrackComponentItem::init() Unknown TrackType.");
        }
    }

    // Selection/Highlighting //
    //
    //setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Transformation //
    //
    setTransform(trackComponent_->getLocalTransform());

    // ContextMenu //
    //
    if (!((trackComponent_->getTrackType() == TrackComponent::DTT_POLY3) && !parentTrackRoadItem_)) // if poly3 and no parent: no morphing
    {
        QAction *action = getContextMenu()->addAction("Morph into poly3");
        connect(action, SIGNAL(triggered()), this, SLOT(morphIntoPoly3()));
    }

    QAction *addToTileAction = getContextMenu()->addAction("Add to current tile");
    connect(addToTileAction, SIGNAL(triggered()), this, SLOT(addToCurrentTile()));
}

//################//
// OBSERVER       //
//################//

void
TrackComponentItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Changes //
    //
    int changes = trackComponent_->getTrackComponentChanges();
    if (changes & TrackComponent::CTC_TransformChange)
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
        setTransform(trackComponent_->getLocalTransform());
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    if ((changes & TrackComponent::CTC_LengthChange)
        || (changes & TrackComponent::CTC_ShapeChange)
        || (changes & (TrackComponent::CTC_AddedChild | TrackComponent::CTC_DeletedChild)))
    {
        createPath();
    }
}

//################//
// SLOTS          //
//################//

void
TrackComponentItem::ungroupComposite()
{
    // does nothing at all - reimplemented by TrackCompositeItem
}

void
TrackComponentItem::hideParentTrackComponent()
{
    if (getParentTrackComponentItem())
    {
        HideDataElementCommand *command = new HideDataElementCommand(getParentTrackComponentItem()->getDataElement(), NULL);
        getProjectGraph()->executeCommand(command);
    }
}

void
TrackComponentItem::hideParentRoad()
{
    if (getParentTrackRoadItem())
    {
        getParentTrackRoadItem()->hideRoad();
    }
}

bool
TrackComponentItem::removeSection()
{
    // Get the highest TrackComponent in the Hierarchy //
    //
    TrackComponent *trackComponent = trackComponent_;
    while (trackComponent->getParentComponent())
    {
        trackComponent = trackComponent->getParentComponent();
    }

    RSystemElementRoad *road = trackComponent->getParentRoad();
    if (trackComponent == road->getTrackComponent(0.0) // Track is the first...
                          //	&& trackComponent == road->getTrackComponent(road->getLength()) // ...and the last one of the road - i.e. the only remaining one.
        )
    {
        //TODO: copy road from trackComponent->getLenght to end and then replace the road
        // Remove road //
        //
        return removeParentRoad();
    }
    else
    {
        // Remove track //
        //
        getProjectGraph()->beginMacro(QObject::tr("Remove Track"));
        RemoveTrackCommand *command = new RemoveTrackCommand(trackComponent->getParentRoad(), trackComponent, false, NULL);
        //RemoveTrackCommand * command = new RemoveTrackCommand(trackComponent->getParentRoad(), trackComponent, true, NULL);
        bool commandExecuted = getProjectGraph()->executeCommand(command);
        if (commandExecuted)
        {
            LinkLanesCommand *linkLanesCommand = new LinkLanesCommand(road);
            getProjectGraph()->executeCommand(linkLanesCommand);
        }

        getProjectGraph()->endMacro();
        return commandExecuted;
    }
}

bool
TrackComponentItem::removeParentRoad()
{
    return getParentTrackRoadItem()->removeRoad();
}

void
TrackComponentItem::morphIntoPoly3()
{
    if (parentTrackComponentItem_)
    {
        parentTrackComponentItem_->morphIntoPoly3();
    }
    else
    {
        MorphIntoPoly3Command *command = new MorphIntoPoly3Command(trackComponent_, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

void
TrackComponentItem::addToCurrentTile()
{
    RSystemElementRoad *road = getParentTrackRoadItem()->getRoad();
    QStringList parts = road->getID().split("_");
    if (parts.at(0) != getProjectData()->getTileSystem()->getCurrentTile()->getID())
    {
        QString name = road->getName();
        QString newId = road->getRoadSystem()->getUniqueId("", name);
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(road->getRoadSystem(), road, newId, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

//*************//
// Delete Item
//*************//

bool
TrackComponentItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
