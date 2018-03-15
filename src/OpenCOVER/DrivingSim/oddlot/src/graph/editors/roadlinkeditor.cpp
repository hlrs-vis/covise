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
#include "src/data/commands/lanesectioncommands.hpp"
#include "src/data/commands/junctioncommands.hpp"

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

                JunctionConnection::ContactPointValue contactPoint;
                if (sink->getIsStart())
                {
                    contactPoint = JunctionConnection::JCP_START;
                }
                else
                {
                    contactPoint = JunctionConnection::JCP_END;
                }
                RoadLink *newRoadLink = NULL;
                RSystemElementJunction *junction = NULL;
                JunctionConnection *newConnection = NULL;
                if (road->getJunction().isValid())
                {
                    junction = getProjectData()->getRoadSystem()->getJunction(road->getJunction());
                    int numConn = 0;
                    if (junction)
                    {
                        numConn = junction->getConnections().size();
                    }
                    newConnection = new JunctionConnection(QString("%1").arg(numConn), item->getParentRoad()->getID(), road->getID(), contactPoint, 1);

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
                    RSystemElementRoad * road = maybeRoad->getRoad();
                    if (!road->getPredecessor() || !road->getSuccessor())
                    {
                        removeZeroWidthLanes(road);             // error correction must not be undone TODO: move to laneeditor
                        roadLinkRoadItems.append(road);
                    }
                }
            }

            LinkRoadsAndLanesCommand *command = new LinkRoadsAndLanesCommand(roadLinkRoadItems, threshold_);
            getProjectGraph()->executeCommand(command);
        }
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

//##################################//
// removeZeroWidthLanes //
//##################################//
void 
RoadLinkEditor::removeZeroWidthLanes(RSystemElementRoad * road)
{

    RoadSystem * roadSystem = road->getRoadSystem();

    // Lane links between the lane sections of the same road
    //
    QMap<double, LaneSection *>::ConstIterator laneSectionsIt = road->getLaneSections().constBegin();

    while(laneSectionsIt != road->getLaneSections().constEnd())
    {
        LaneSection * laneSection = laneSectionsIt.value();
        QMap<int, Lane *>::const_iterator laneIt = laneSection->getLanes().constBegin();
        bool deleteLane;
        while (laneIt != laneSection->getLanes().constEnd())
        {
            Lane * lane = laneIt.value();

            if (lane->getId() == 0)
            {
                deleteLane = false;
            }
            else
            {
                deleteLane = true;
                QMap<double, LaneWidth *>::ConstIterator iter = lane->getWidthEntries().constBegin();

                while (iter != lane->getWidthEntries().constEnd())
                {
                    if ((fabs(iter.value()->getA()) > NUMERICAL_ZERO3) || (fabs(iter.value()->getB()) > NUMERICAL_ZERO3)
                        || (fabs(iter.value()->getC()) > NUMERICAL_ZERO3) || (fabs(iter.value()->getD()) > NUMERICAL_ZERO3))
                    {
                        deleteLane = false;
                        break;
                    }
                    iter++;
                }

                if (deleteLane)
                {
                    RemoveLaneCommand * command = new RemoveLaneCommand(laneSection, lane);
                    getProjectGraph()->executeCommand(command);

                    break;
                }
            }
            laneIt++;
        }
        if (!deleteLane)
        {
            laneSectionsIt++;
        }
    }
}
                
    