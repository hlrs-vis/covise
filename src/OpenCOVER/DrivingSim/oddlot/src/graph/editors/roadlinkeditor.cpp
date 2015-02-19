/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11.03.2010
**
**************************************************************************/

#include "roadlinkeditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/commands/roadcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/roadlink/roadlinkroadsystemitem.hpp"
#include "src/graph/items/roadsystem/roadlink/roadlinkroaditem.hpp"

#include "src/graph/items/roadsystem/roadlink/roadlinkitem.hpp"
#include "src/graph/items/roadsystem/roadlink/roadlinkhandle.hpp"
#include "src/graph/items/roadsystem/roadlink/roadlinksinkitem.hpp"

#include "src/graph/items/handles/circularhandle.hpp"

// Tools //
//
#include "src/gui/tools/roadlinkeditortool.hpp"

// Qt //
//

//################//
// CONSTRUCTORS   //
//################//

RoadLinkEditor::RoadLinkEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , threshold_(10.0)
{
}

RoadLinkEditor::~RoadLinkEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

/**
*
*/
void
RoadLinkEditor::init()
{
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new RoadLinkRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);
    }
}

/*!
*/
void
RoadLinkEditor::kill()
{
    delete roadSystemItem_;
    roadSystemItem_ = NULL;
}

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
RoadLinkEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    // RoadLink //
    //
    RoadLinkEditorToolAction *action = dynamic_cast<RoadLinkEditorToolAction *>(toolAction);
    if (action)
    {
        if (action->getToolId() == ODD::TRL_LINK)
        {
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

            // Change types of selected items //
            //
            RoadLinkHandle *handle = NULL;
            RoadLinkSinkItem *sink = NULL;

            foreach (QGraphicsItem *item, selectedItems)
            {
                RoadLinkHandle *maybeHandle = dynamic_cast<RoadLinkHandle *>(item);
                if (maybeHandle)
                {
                    if (handle)
                    {
                        return; // TODO
                    }
                    else
                    {
                        handle = maybeHandle;
                    }
                }

                CircularHandle *maybeSinkHandle = dynamic_cast<CircularHandle *>(item);
                if (maybeSinkHandle)
                {
                    RoadLinkSinkItem *maybeSink = dynamic_cast<RoadLinkSinkItem *>(maybeSinkHandle->parentItem()); // yeah, not so clean
                    if (maybeSink)
                    {
                        if (sink)
                        {
                            return; // TODO
                        }
                        else
                        {
                            sink = maybeSink;
                        }
                    }
                }
            }

            if (handle && sink)
            {
                RoadLinkItem *item = handle->getParentRoadLinkItem();

                RSystemElementRoad *road = sink->getParentRoad();

                QString contactPoint;
                if (sink->getIsStart())
                {
                    contactPoint = "start";
                }
                else
                {
                    contactPoint = "end";
                }
                RoadLink *newRoadLink = NULL;
                RSystemElementJunction *junction = NULL;
                JunctionConnection *newConnection = NULL;
                if (road->getJunction() != "-1" && road->getJunction() != "")
                {
                    junction = getProjectData()->getRoadSystem()->getJunction(road->getJunction());
                    int numConn = 0;
                    if (junction)
                    {
                        numConn = junction->getConnections().size();
                    }
                    newConnection = new JunctionConnection(QString("jc%1").arg(numConn), item->getParentRoad()->getID(), road->getID(), contactPoint, 1);

                    QMap<double, LaneSection *> lanes = road->getLaneSections();
                    LaneSection *lsC = *lanes.begin();
                    bool even = false;
                    if (sink->getIsStart())
                    {
                        even = true;
                        lsC = *lanes.begin();
                    }
                    else
                    {
                        even = false;
                        QMap<double, LaneSection *>::iterator it = lanes.end();
                        it--;
                        lsC = *(it);
                    }
                    if (handle->getParentRoadLinkItem()->getRoadLinkType() == RoadLink::DRL_PREDECESSOR)
                    {
                        if (even)
                        {
                            foreach (Lane *lane, lsC->getLanes())
                            {
                                if (lane->getId() > 0)
                                    newConnection->addLaneLink(lane->getId(), -lane->getId());
                            }
                        }
                        else
                        {
                            foreach (Lane *lane, lsC->getLanes())
                            {
                                if (lane->getId() > 0)
                                    newConnection->addLaneLink(lane->getId(), lane->getId());
                            }
                        }
                    }
                    else
                    {
                        if (even)
                        {
                            foreach (Lane *lane, lsC->getLanes())
                            {
                                if (lane->getId() < 0)
                                    newConnection->addLaneLink(lane->getId(), lane->getId());
                            }
                        }
                        else
                        {
                            foreach (Lane *lane, lsC->getLanes())
                            {
                                if (lane->getId() < 0)
                                    newConnection->addLaneLink(lane->getId(), -lane->getId());
                            }
                        }
                    }
                    newRoadLink = new RoadLink("junction", road->getJunction(), contactPoint);
                }
                else
                {

                    newRoadLink = new RoadLink("road", road->getID(), contactPoint);
                }

                SetRoadLinkCommand *command = new SetRoadLinkCommand(item->getParentRoad(), item->getRoadLinkType(), newRoadLink, newConnection, junction);
                getProjectGraph()->executeCommand(command);
            }
        }
        else if (action->getToolId() == ODD::TRL_ROADLINK)
        {
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

            QList<RSystemElementRoad *> roadLinkRoadItems;

            foreach (QGraphicsItem *item, selectedItems)
            {
                RoadLinkRoadItem *maybeRoad = dynamic_cast<RoadLinkRoadItem *>(item);
                if (maybeRoad)
                {
                    roadLinkRoadItems.append(maybeRoad->getRoad());
                }
            }

            SetRoadLinkRoadsCommand *command = new SetRoadLinkRoadsCommand(roadLinkRoadItems, threshold_);
            getProjectGraph()->executeCommand(command);
        }
  /*       else if (action->getToolId() == ODD::TRL_LANELINK)
        {
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

            QList<RSystemElementRoad *> laneLinkRoadItems;

            foreach (QGraphicsItem *item, selectedItems)
            {
                RoadLinkRoadItem *maybeRoad = dynamic_cast<RoadLinkRoadItem *>(item);
                if (maybeRoad)
                {
                    laneLinkRoadItems.append(maybeRoad->getRoad());
                }
            }

            SetRoadLinkRoadsCommand *command = new SetRoadLinkRoadsCommand(roadLinkRoadItems, threshold_);
            getProjectGraph()->executeCommand(command);
        }*/
        else if (action->getToolId() == ODD::TRL_SELECT)
        {
            threshold_ = action->getThreshold();
        }
        else if (action->getToolId() == ODD::TRL_UNLINK)
        {
            // Macro Command //
            //
            int numberSelectedItems = getTopviewGraph()->getScene()->selectedItems().size();
            if (numberSelectedItems > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Unlink Roads"));
            }

            bool deletedSomething = false;
            do
            {
                deletedSomething = false;
                QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

                foreach (QGraphicsItem *item, selectedItems)
                {
                    RoadLinkRoadItem *maybeRoad = dynamic_cast<RoadLinkRoadItem *>(item);
                    if (maybeRoad)
                    {
                        maybeRoad->removeRoadLink();
                        maybeRoad->setSelected(false);

                        deletedSomething = true;
                        break;
                    }
                }

            } while (deletedSomething);

            // Macro Command //
            //
            if (numberSelectedItems > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }
        }
    }
}
