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

#include "junctioncomponentitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackelement.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/trackcommands.hpp"

// Graph //
//
#include "src/gui/projectwidget.hpp"

#include "src/graph/projectgraph.hpp"

#include "src/graph/items/roadsystem/junction/junctionroaditem.hpp"
#include "src/graph/items/roadsystem/junction/junctionelementitem.hpp"
#include "src/graph/items/roadsystem/junction/junctionsparcsitem.hpp"

// Editor //
//
#include "src/graph/editors/junctioneditor.hpp"

//################//
// CONSTRUCTOR    //
//################//

JunctionComponentItem::JunctionComponentItem(JunctionRoadItem *parentJunctionRoadItem, TrackComponent *trackComponent)
    : GraphElement(parentJunctionRoadItem, trackComponent)
    , junctionEditor_(NULL)
    , parentJunctionRoadItem_(parentJunctionRoadItem)
    , parentJunctionComponentItem_(NULL)
    , trackComponent_(trackComponent)
{
    // Init //
    //
    init();
}

JunctionComponentItem::JunctionComponentItem(JunctionComponentItem *parentJunctionComponentItem, TrackComponent *trackComponent)
    : GraphElement(parentJunctionComponentItem, trackComponent)
    , junctionEditor_(NULL)
    , parentJunctionRoadItem_(NULL)
    , parentJunctionComponentItem_(parentJunctionComponentItem)
    , trackComponent_(trackComponent)
{
    parentJunctionRoadItem_ = parentJunctionComponentItem_->getParentJunctionRoadItem();

    // Init //
    //
    init();
}

JunctionComponentItem::~JunctionComponentItem()
{
}

void
JunctionComponentItem::init()
{
    // JunctionEditor //
    //
    junctionEditor_ = dynamic_cast<JunctionEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!junctionEditor_)
    {
        qDebug("Warning 1007131313! JunctionComponentItem not created by an JunctionEditor");
    }

    // JunctionItems //
    //
    foreach (TrackComponent *track, trackComponent_->getChildTrackComponents())
    {
        if (track->getTrackType() == TrackComponent::DTT_SPARCS)
        {
            TrackSpiralArcSpiral *sparcs = dynamic_cast<TrackSpiralArcSpiral *>(track);
            new JunctionSpArcSItem(this, sparcs);
        }
        else if ((track->getTrackType() == TrackComponent::DTT_LINE)
                 || (track->getTrackType() == TrackComponent::DTT_ARC)
                 || (track->getTrackType() == TrackComponent::DTT_SPIRAL))
        {
            TrackElement *trackElement = dynamic_cast<TrackElement *>(track);
            new JunctionElementItem(this, trackElement);
        }
        else
        {
            qDebug("WARNING 1007051646! JunctionComponentItem::init() Unknown TrackType.");
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
    if (!((trackComponent_->getTrackType() == TrackComponent::DTT_POLY3) && !parentJunctionRoadItem_)) // if poly3 and no parent: no morphing
    {
        QAction *action = getContextMenu()->addAction("Morph into poly3");
        connect(action, SIGNAL(triggered()), this, SLOT(morphIntoPoly3()));
    }
}

//################//
// OBSERVER       //
//################//

void
JunctionComponentItem::updateObserver()
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
JunctionComponentItem::ungroupComposite()
{
    // does nothing at all - reimplemented by JunctionCompositeItem
}

void
JunctionComponentItem::hideParentTrackComponent()
{
    if (getParentJunctionComponentItem())
    {
        HideDataElementCommand *command = new HideDataElementCommand(getParentJunctionComponentItem()->getDataElement(), NULL);
        getProjectGraph()->executeCommand(command);
    }
}

void
JunctionComponentItem::hideParentRoad()
{
    if (getParentJunctionRoadItem())
    {
        getParentJunctionRoadItem()->hideRoad();
    }
}

bool
JunctionComponentItem::removeSection()
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
                          //		&& trackComponent == road->getTrackComponent(road->getLength()) // ...and the last one of the road - i.e. the only remaining one.
        )
    {
        // Remove road //
        //
        return removeParentRoad();
    }
    else
    {
        // Remove track //
        //
        RemoveTrackCommand *command = new RemoveTrackCommand(trackComponent->getParentRoad(), trackComponent, false, NULL);
        //RemoveTrackCommand * command = new RemoveTrackCommand(trackComponent->getParentRoad(), trackComponent, true, NULL);
        return getProjectGraph()->executeCommand(command);
    }
}

bool
JunctionComponentItem::removeParentRoad()
{
    return getParentJunctionRoadItem()->removeRoad();
}

void
JunctionComponentItem::morphIntoPoly3()
{
    if (parentJunctionComponentItem_)
    {
        parentJunctionComponentItem_->morphIntoPoly3();
    }
    else
    {
        MorphIntoPoly3Command *command = new MorphIntoPoly3Command(trackComponent_, NULL);
        getProjectGraph()->executeCommand(command);
    }
}

//*************//
// Delete Item
//*************//

bool
JunctionComponentItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
