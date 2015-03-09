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
        else if (action->getToolId() == ODD::TRL_LANELINK)
        {
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

            QList<RSystemElementRoad *> laneLinkRoadItems;

            // Macro Command //
            //
            int numberOfSelectedItems = selectedItems.size();
            if(numberOfSelectedItems > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set Lane Links"));
            }
            foreach (QGraphicsItem *item, selectedItems)
            {
                RoadLinkRoadItem *maybeRoad = dynamic_cast<RoadLinkRoadItem *>(item);
                if (maybeRoad)
                {
                    createLaneLinks(maybeRoad->getRoad());
                }

            }
            // Macro Command //
            //
            if(numberOfSelectedItems > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }

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
// Create Lane Links //
//#################################//
void
RoadLinkEditor::createLaneLinks(RSystemElementRoad * road)
{
    RoadSystem * roadSystem = road->getRoadSystem();

    // Lane links between the laneSections of the same road
    //
    QMap<double, LaneSection *>::const_iterator laneSectionsIt = road->getLaneSections().constBegin();
    while(laneSectionsIt != road->getLaneSections().constEnd() - 1)
    {
        LaneSection * laneSection = laneSectionsIt.value();
        QMap<int, Lane *>::const_iterator laneIt = laneSection->getLanes().constBegin();
        while (laneIt != laneSection->getLanes().constEnd())
        {
            Lane * lane = laneIt.value();

            if (lane->getId() != 0)
            {
                double laneWidth = lane->getWidth(laneSection->getSEnd());
                if (laneWidth < NUMERICAL_ZERO3)
                {
                    SetLaneSuccessorIdCommand * command = new SetLaneSuccessorIdCommand(lane, Lane::NOLANE);
                    getProjectGraph()->executeCommand(command);
                    laneIt++;
                    continue;
                }
                double t = getTValue(laneSection, lane, laneSection->getSEnd(), laneWidth);

                LaneSection * nextLaneSection = (laneSectionsIt + 1).value();
                int laneId = nextLaneSection->getLaneId(0, t);

                SetLaneSuccessorIdCommand * successorCommand = new SetLaneSuccessorIdCommand(lane, laneId);
                getProjectGraph()->executeCommand(successorCommand);

                SetLanePredecessorIdCommand * predecessorCommand = new SetLanePredecessorIdCommand(nextLaneSection->getLane(laneId), lane->getId());
                getProjectGraph()->executeCommand(predecessorCommand);
            }

            laneIt++;
        }
        laneSectionsIt++;
    }

    

    // Lane Links to another road
    //

    LaneSection * startLaneSection = road->getLaneSection(0.0);
    LaneSection * endLaneSection = road->getLaneSection(road->getLength());

    RoadLink * roadPredecessor = road->getPredecessor();
    RSystemElementJunction * junction = NULL;
    JunctionConnection * junctionConnection = NULL;

    if (road->getJunction() != "-1")
    {
        junction = roadSystem->getJunction(road->getJunction());
       
        if (roadPredecessor)
        {
            junctionConnection = junction->getConnection(roadPredecessor->getElementId(), road->getID());
            if (junctionConnection)                         // cleanup
            {
                RemoveConnectionLaneLinksCommand * command = new RemoveConnectionLaneLinksCommand(junctionConnection);
                getProjectGraph()->executeCommand(command);
            }
        }
    }

    QMap<int, Lane *>::const_iterator laneIt = startLaneSection->getLanes().constBegin(); // Predecessor
    while (laneIt != startLaneSection->getLanes().constEnd())
    {
        Lane * lane = laneIt.value();
        if (lane->getId() != 0)
        {

            if (!roadPredecessor || (roadPredecessor->getElementType() == "junction") || (lane->getWidth(0.0) < NUMERICAL_ZERO3))
            {
                SetLanePredecessorIdCommand * predecessorCommand = new SetLanePredecessorIdCommand(lane, Lane::NOLANE);
                getProjectGraph()->executeCommand(predecessorCommand);
            }
            else
            {
                RSystemElementRoad * predRoad = roadSystem->getRoad(roadPredecessor->getElementId());
                if (predRoad)
                {
                    double t = getTValue(startLaneSection, lane, 0.0, lane->getWidth(0.0));
                    QPointF laneMid = road->getGlobalPoint(0.0, t);
                    QVector2D vecStart = QVector2D(laneMid - predRoad->getGlobalPoint(0.0));
                    QVector2D vecEnd = QVector2D(laneMid - predRoad->getGlobalPoint(predRoad->getLength()));
                    double distStart = vecStart.length();
                    double distEnd = vecEnd.length();
                    if (distStart < distEnd)
                    {
                        if (distStart < NUMERICAL_ZERO3)
                        {
                            if (lane->getId() > 0)
                            {
                                distStart = -distStart;
                            }
                        }
                        else if (QVector2D::dotProduct(predRoad->getGlobalNormal(0.0), vecStart.normalized()) > 0)
                        {
                            distStart = -distStart;
                        }

                        LaneSection * nextLaneSection = predRoad->getLaneSection(0.0);
                        int laneId = nextLaneSection->getLaneId(0, distStart);
                           
                        SetLanePredecessorIdCommand * lanePredecessorCommand = new SetLanePredecessorIdCommand(lane, laneId);
                        getProjectGraph()->executeCommand(lanePredecessorCommand);

                        SetLanePredecessorIdCommand * nextLanePredecessorCommand = new SetLanePredecessorIdCommand(nextLaneSection->getLane(laneId), lane->getId());
                        getProjectGraph()->executeCommand(nextLanePredecessorCommand);

                    }
                    else
                    {
                        if (distEnd < NUMERICAL_ZERO3)
                        {
                            if (lane->getId() < 0)
                            {
                                distEnd = -distEnd;
                            }
                        }
                        else if (QVector2D::dotProduct(predRoad->getGlobalNormal(predRoad->getLength()), vecEnd.normalized()) > 0)
                        {
                            distEnd = -distEnd;
                        }

                        LaneSection * nextLaneSection = predRoad->getLaneSection(predRoad->getLength());
                        int laneId = nextLaneSection->getLaneId(predRoad->getLength(), distEnd);

                        SetLanePredecessorIdCommand * predecessorCommand = new SetLanePredecessorIdCommand(lane, laneId);
                        getProjectGraph()->executeCommand(predecessorCommand);


                        SetLaneSuccessorIdCommand * successorCommand = new SetLaneSuccessorIdCommand(nextLaneSection->getLane(laneId), lane->getId());
                        getProjectGraph()->executeCommand(successorCommand);

                    }
                }
            }
 
            if (junctionConnection && (junctionConnection->getFromLane(lane->getId()) != lane->getPredecessor()))
            {
                SetConnectionLaneLinkCommand * command = new SetConnectionLaneLinkCommand(junctionConnection, lane->getPredecessor(), lane->getId());
                getProjectGraph()->executeCommand(command);
            }
        }

        laneIt++;
    }

    RoadLink * roadSuccessor = road->getSuccessor();        // Successor

    if (junction && roadSuccessor)
    {
        junctionConnection =  junction->getConnection(roadSuccessor->getElementId(), road->getID());
        if (junctionConnection)                         // cleanup
        {
            RemoveConnectionLaneLinksCommand * command = new RemoveConnectionLaneLinksCommand(junctionConnection);
            getProjectGraph()->executeCommand(command);
        }
    }

    laneIt = endLaneSection->getLanes().constBegin();
    while (laneIt != endLaneSection->getLanes().constEnd())
    {
        Lane * lane = laneIt.value();
        if (lane->getId() != 0)
        {

            if (!roadSuccessor || (roadSuccessor->getElementType() == "junction") || (lane->getWidth(road->getLength()) < NUMERICAL_ZERO3))
            {
                SetLaneSuccessorIdCommand * successorCommand = new SetLaneSuccessorIdCommand(lane, Lane::NOLANE);
                getProjectGraph()->executeCommand(successorCommand);
            }
            else
            {
                RSystemElementRoad * succRoad = roadSystem->getRoad(roadSuccessor->getElementId());
                if (succRoad)
                {
                    double t = getTValue(endLaneSection, lane, road->getLength(), lane->getWidth(road->getLength()));
                    QPointF laneMid = road->getGlobalPoint(road->getLength(), t);
                    QVector2D vecStart = QVector2D(laneMid - succRoad->getGlobalPoint(0.0));
                    QVector2D vecEnd = QVector2D(laneMid - succRoad->getGlobalPoint(succRoad->getLength()));
                    double distStart = vecStart.length();
                    double distEnd = vecEnd.length();

                    if (distStart < distEnd)
                    {
                        if (distStart < NUMERICAL_ZERO3)
                        {
                            if (lane->getId() < 0)
                            {
                                distStart = -distStart;
                            }
                        }
                        else if (QVector2D::dotProduct(succRoad->getGlobalNormal(0.0), vecStart.normalized()) > 0)
                        {
                            distStart = -distStart;
                        }

                        LaneSection * nextLaneSection = succRoad->getLaneSection(0.0);
                        int laneId = nextLaneSection->getLaneId(0, distStart);

                        SetLaneSuccessorIdCommand * successorCommand = new SetLaneSuccessorIdCommand(lane, laneId);
                        getProjectGraph()->executeCommand(successorCommand);

                        SetLanePredecessorIdCommand * predecessorCommand = new SetLanePredecessorIdCommand(nextLaneSection->getLane(laneId), lane->getId());
                        getProjectGraph()->executeCommand(predecessorCommand);
                    }
                    else
                    {

                        if (distEnd < NUMERICAL_ZERO3) 
                        {
                            if (lane->getId() > 0)
                            {
                                distEnd = -distEnd;
                            }
                        }
                        else if (QVector2D::dotProduct(succRoad->getGlobalNormal(succRoad->getLength()), vecEnd.normalized()) > 0)
                        {
                            distEnd = -distEnd;
                        }
                        LaneSection * nextLaneSection = succRoad->getLaneSection(succRoad->getLength());
                        int laneId = nextLaneSection->getLaneId(succRoad->getLength(), distEnd);

                        SetLaneSuccessorIdCommand * laneSuccessorCommand = new SetLaneSuccessorIdCommand(lane, laneId);
                        getProjectGraph()->executeCommand(laneSuccessorCommand);

                        SetLaneSuccessorIdCommand * nextLaneSuccessorCommand = new SetLaneSuccessorIdCommand(nextLaneSection->getLane(laneId), lane->getId());
                        getProjectGraph()->executeCommand(nextLaneSuccessorCommand);
                    }
                }
            }


            if (junctionConnection && (junctionConnection->getFromLane(lane->getId()) != lane->getSuccessor()))
            {
                SetConnectionLaneLinkCommand * command = new SetConnectionLaneLinkCommand(junctionConnection, lane->getSuccessor(), lane->getId());
                getProjectGraph()->executeCommand(command);
            }
        }

        laneIt++;
    }

}

double
RoadLinkEditor::getTValue(LaneSection * laneSection, Lane * lane, double s, double laneWidth)
{
    double t = 0.0;

    if (lane->getId() < 0)
    {
        if (laneWidth < NUMERICAL_ZERO3)
        {
            if (lane->getId() == laneSection->getRightmostLaneId())
            {
                t = NUMERICAL_ZERO3;
            }
            else
            {
                t = -NUMERICAL_ZERO3;
            }
        }
        else
        {
            t = -laneWidth / 2;
        }

        t = -laneSection->getLaneSpanWidth(lane->getId() + 1, 0, s) + t;
    }
    else if (lane->getId() > 0)
    {
        if (laneWidth < NUMERICAL_ZERO3)
        {
            if (lane->getId() == laneSection->getLeftmostLaneId())
            {
                t = -NUMERICAL_ZERO3;
            }
            else
            {
                t = NUMERICAL_ZERO3;
            }
        }
        else
        {
            t = laneWidth / 2;
        }

        t = laneSection->getLaneSpanWidth(0, lane->getId() - 1, s) + t;
    }

    return t;
}