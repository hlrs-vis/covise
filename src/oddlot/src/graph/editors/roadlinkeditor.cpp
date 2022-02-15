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

#include "src/mainwindow.hpp"

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
#include "src/data/commands/dataelementcommands.hpp"

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
#include "src/graph/items/roadsystem/roadlink/roadlinksinkhandle.hpp"

#include "src/graph/items/roadsystem/junctionitem.hpp"

#include "src/graph/items/handles/circularhandle.hpp"

// Tools //
//
#include "src/gui/tools/roadlinkeditortool.hpp"
#include "src/gui/mouseaction.hpp"

// GUI //
//
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

// Qt //
//

//################//
// CONSTRUCTORS   //
//################//

RoadLinkEditor::RoadLinkEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , threshold_(10.0)
    , linkHandle_(NULL)
    , sinkHandle_(NULL)
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
        roadSystemItem_ = new RoadLinkRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem(), this);
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);
    }
}

/*!
*/
void
RoadLinkEditor::kill()
{
    if (tool_)
    {
        reset();
        ODD::mainWindow()->showParameterDialog(false);
    }

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
    if (tool_ && !tool_->containsToolId(toolAction->getToolId()))
    {
        resetTool();
    }

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
            ODD::ToolId paramTool = getCurrentParameterTool();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                getTopviewGraph()->getScene()->deselectAll();
                roadSystemItem_->setRoadsSelectable(false);

                ToolValue<RoadLinkHandle> *param = new ToolValue<RoadLinkHandle>(ODD::TRL_LINK, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Arrow Handle", true);
                tool_ = new Tool(ODD::TRL_LINK, 4);
                tool_->readParams(param);
                ToolValue<RoadLinkSinkHandle> *roadParam = new ToolValue<RoadLinkSinkHandle>(ODD::TRL_SINK, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Circular Sink");
                tool_->readParams(roadParam);

                createToolParameterSettingsApplyBox(tool_, ODD::ERL);
                ODD::mainWindow()->showParameterDialog(true, "Link Handle and Sink", "SELECT arrow handle and circular sink and press APPLY");

                applyCount_ = 2;
            }
        }
        else if (action->getToolId() == ODD::TRL_ROADLINK)
        {
            ODD::ToolId paramTool = getCurrentParameterTool();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                roadSystemItem_->setHandlesSelectable(false);

                ToolValue<double> *param = new ToolValue<double>(ODD::TRL_THRESHOLD, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::DOUBLE, "Threshold");
                param->setValue(threshold_);
                tool_ = new Tool(ODD::TRL_ROADLINK, 4);
                tool_->readParams(param);
                ToolValue<RSystemElementRoad> *roadParam = new ToolValue<RSystemElementRoad>(ODD::TRL_ROADLINK, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove", true);
                tool_->readParams(roadParam);

                createToolParameterSettingsApplyBox(tool_, ODD::ERL);
                ODD::mainWindow()->showParameterDialog(true, "Link roads", "Specify threshold, SELECT/DESELECT roads and press APPLY");

                applyCount_ = 2;

                assignParameterSelection(ODD::TRL_ROADLINK);
            }

        }
        else if (action->getToolId() == ODD::TRL_UNLINK)
        {
            ODD::ToolId paramTool = getCurrentParameterTool();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                roadSystemItem_->setHandlesSelectable(false);

                ToolValue<RSystemElementRoad> *param = new ToolValue<RSystemElementRoad>(ODD::TRL_UNLINK, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove", true);
                tool_ = new Tool(ODD::TRL_UNLINK, 4);
                tool_->readParams(param);

                createToolParameterSettingsApplyBox(tool_, ODD::ERL);
                ODD::mainWindow()->showParameterDialog(true, "Unlink roads", "SELECT/DESELECT roads and press APPLY");

                applyCount_ = 2;
                assignParameterSelection(ODD::TRL_UNLINK);
            }
        }

    }
    else if (toolAction->getToolId() == ODD::TRL_THRESHOLD)
    {

        ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
        if (action)
        {
            threshold_ = action->getValue();
        }
    }
    else
    {
        ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
        if (action)
        {
            if (action->getToolId() == ODD::TRL_ROADLINK)
            {
                ODD::ToolId paramTool = action->getParamToolId();
                if (paramTool == ODD::TPARAM_SELECT)
                {
                    currentParamId_ = action->getParamId();
                    if (!action->getState())
                    {

                        QList<RSystemElementRoad *>roads = tool_->removeToolParameters<RSystemElementRoad>(currentParamId_);
                        foreach(RSystemElementRoad * road, roads)
                        {
                            DeselectDataElementCommand *command = new DeselectDataElementCommand(road, NULL);
                            getProjectGraph()->executeCommand(command); 
                            deregisterRoad(road);
                        }
                    }
                }
            }
            else if (action->getToolId() == ODD::TRL_UNLINK)
            {
                ODD::ToolId paramTool = action->getParamToolId();
                if (paramTool == ODD::TPARAM_SELECT)
                {
                    currentParamId_ = action->getParamId();
                    if (!action->getState())
                    {

                        QList<RSystemElementRoad *>roads = tool_->removeToolParameters<RSystemElementRoad>(currentParamId_);
                        foreach(RSystemElementRoad * road, roads)
                        {
                            DeselectDataElementCommand *command = new DeselectDataElementCommand(road, NULL);
                            getProjectGraph()->executeCommand(command);
                            selectedRoads_.removeOne(road);
                        }

                        // verify if apply has to be hidden //
                        if (tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool()) < applyCount_)
                        {
                            settingsApplyBox_->setApplyButtonVisible(false);
                        }
                    }
                }
            }
        }
    }
}

void
RoadLinkEditor::assignParameterSelection(ODD::ToolId toolId)
{
    if ((toolId == ODD::TRL_ROADLINK) || (toolId == ODD::TRL_UNLINK))
    {
        QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

        for (int i = 0; i < selectedItems.size();)
        {
            QGraphicsItem *item = selectedItems.at(i);
            RoadLinkRoadItem *roadItem = dynamic_cast<RoadLinkRoadItem *>(item);
            if (roadItem)
            {
                RSystemElementRoad *road = roadItem->getRoad();

                createToolParameters<RSystemElementRoad>(road);
                selectedRoads_.append(road);

                item->setSelected(true);

                i++;
            }
            else
            {
                RoadLinkSinkItem *linkSinkItem = dynamic_cast<RoadLinkSinkItem *>(item);
                if (!linkSinkItem)
                {
                    item->setSelected(false);
                }
                selectedItems.removeAt(i);
            }
        }

        // verify if apply can be displayed //

        int objectCount = tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool());
        if (objectCount >= applyCount_)
        {
            settingsApplyBox_->setApplyButtonVisible(true);
        }
    }
}

//##################################//
// removeZeroWidthLanes //
//##################################//
void
RoadLinkEditor::removeZeroWidthLanes(RSystemElementRoad *road)
{

    RoadSystem *roadSystem = road->getRoadSystem();

    // Lane links between the lane sections of the same road
    //
    QMap<double, LaneSection *>::ConstIterator laneSectionsIt = road->getLaneSections().constBegin();

    while (laneSectionsIt != road->getLaneSections().constEnd())
    {
        LaneSection *laneSection = laneSectionsIt.value();
        QMap<int, Lane *>::const_iterator laneIt = laneSection->getLanes().constBegin();
        bool deleteLane;
        while (laneIt != laneSection->getLanes().constEnd())
        {
            Lane *lane = laneIt.value();

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
                    RemoveLaneCommand *command = new RemoveLaneCommand(laneSection, lane);
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

void 
RoadLinkEditor::registerRoad(RSystemElementRoad *road)
{
    createToolParameters<RSystemElementRoad>(road);
    selectedRoads_.append(road);

    int objectCount = tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool());
    if (objectCount >= applyCount_)
    {
        settingsApplyBox_->setApplyButtonVisible(true);
    }
}

void 
RoadLinkEditor::deregisterRoad(RSystemElementRoad *road)
{
    removeToolParameters<RSystemElementRoad>(road);
    selectedRoads_.removeOne(road);

    int objectCount = tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool());
    if (objectCount < applyCount_)
    {
        settingsApplyBox_->setApplyButtonVisible(false);
    }
}

bool 
RoadLinkEditor::registerLinkHandle(RoadLinkHandle *roadLinkHandle, RSystemElementRoad *road)
{
    if (roadLinkHandle != linkHandle_)
    {
        if (sinkHandle_ && (road == sinkHandle_->getParentRoadLinkSinkItem()->getParentRoad()))
        {
            return false;
        }

        if (linkHandle_)
        {
            linkHandle_->setSelected(false);
        }

        setToolValue<RoadLinkHandle>(roadLinkHandle, road->getIdName());

        linkHandle_ = roadLinkHandle;

        // verify if apply can be displayed //
        if (tool_->verify())
        {
            settingsApplyBox_->setApplyButtonVisible(true);
        }

        return true;
    }

    return false;
}

void
RoadLinkEditor::deregisterHandle(Handle *handle, ODD::ToolId toolId)
{
    delToolValue(toolId, ODD::TPARAM_SELECT);
    if (handle == linkHandle_)
    {
        linkHandle_ = NULL;
    }
    else if (handle == sinkHandle_)
    {
        sinkHandle_ = NULL;
    }
    settingsApplyBox_->setApplyButtonVisible(false);
}

bool
RoadLinkEditor::registerLinkSinkHandle(RoadLinkSinkHandle *linkSinkHandle, RSystemElementRoad *road)
{
    if (linkSinkHandle != sinkHandle_)
    {
        if (linkHandle_ && (road == linkHandle_->getParentRoadLinkItem()->getParentRoad()))
        {
            return false;
        }

        if (sinkHandle_)
        {
            sinkHandle_->setSelected(false);
        }

        setToolValue<RoadLinkSinkHandle>(linkSinkHandle, road->getIdName());
        sinkHandle_ = linkSinkHandle;

        // verify if apply can be displayed //
        if (tool_->verify())
        {
            settingsApplyBox_->setApplyButtonVisible(true);
        }

        return true;
    }

    return false;
}

void
RoadLinkEditor::apply()
{
    ODD::ToolId toolId = tool_->getToolId();
    if (toolId == ODD::TRL_ROADLINK)
    {

        for (int i = 0; i < selectedRoads_.size();)
        {
            RSystemElementRoad *road = selectedRoads_.at(i);
            /* if (road->getPredecessor() || road->getSuccessor())
                {
                    selectedRoads_.takeAt(i);
                }
                else
                { */
            removeZeroWidthLanes(road);             // error correction must not be undone TODO: move to laneeditor
            i++;
            // }
        }

        LinkRoadsAndLanesCommand *command = new LinkRoadsAndLanesCommand(selectedRoads_, threshold_);
        getProjectGraph()->executeCommand(command);

    }
    else if (toolId == ODD::TRL_UNLINK)
    {
        // Macro Command //
        //
        int numberSelectedItems = selectedRoads_.size();
        if (numberSelectedItems > 1)
        {
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Unlink Roads"));
        }
        foreach(RSystemElementRoad * road, selectedRoads_)
        {
            RemoveRoadLinkCommand *command = new RemoveRoadLinkCommand(road, NULL);
            getProjectGraph()->executeCommand(command);
        }

        // Macro Command //
        //
        if (numberSelectedItems > 1)
        {
            getProjectData()->getUndoStack()->endMacro();
        }
    }
    else if (toolId == ODD::TRL_LINK)
    {
        RoadLinkHandle *handle = dynamic_cast<ToolValue<RoadLinkHandle> *>(tool_->getParam(ODD::TRL_LINK, ODD::TPARAM_SELECT))->getValue();
        RoadLinkSinkHandle *sinkHandle = dynamic_cast<ToolValue<RoadLinkSinkHandle> *>(tool_->getParam(ODD::TRL_SINK, ODD::TPARAM_SELECT))->getValue();

        RoadLinkItem *item = handle->getParentRoadLinkItem();
        RoadLinkSinkItem *sink = sinkHandle->getParentRoadLinkSinkItem();

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
                    foreach(Lane * lane, lsC->getLanes())
                    {
                        if (lane->getId() > 0)
                            newConnection->addLaneLink(lane->getId(), -lane->getId());
                    }
                }
                else
                {
                    foreach(Lane * lane, lsC->getLanes())
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
                    foreach(Lane * lane, lsC->getLanes())
                    {
                        if (lane->getId() < 0)
                            newConnection->addLaneLink(lane->getId(), lane->getId());
                    }
                }
                else
                {
                    foreach(Lane * lane, lsC->getLanes())
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

        linkHandle_ = NULL;

    }
}

void
RoadLinkEditor::clearToolObjectSelection()
{
    if (sinkHandle_)
    {
        sinkHandle_->setSelected(false);
        sinkHandle_ = NULL;
    }
    if (linkHandle_)
    {
        linkHandle_->setSelected(false);
        linkHandle_ = NULL;
    }

    QList<DataElement *> dataElementList;
    foreach(RSystemElementRoad * road, selectedRoads_)
    {
        dataElementList.append(road);
    }
    DeselectDataElementCommand *command = new DeselectDataElementCommand(dataElementList, NULL);
    getProjectGraph()->executeCommand(command);

    selectedRoads_.clear();
}


void
RoadLinkEditor::reset()
{
    clearToolObjectSelection();
    delToolParameters();
}

void
RoadLinkEditor::resetTool()
{

    if (tool_)
    {
        ODD::ToolId toolId = tool_->getToolId();
        if (toolId == ODD::TRL_ROADLINK)
        {
            roadSystemItem_->setHandlesSelectable(true);
        }
        else if (toolId == ODD::TRL_LINK)
        {
            roadSystemItem_->setRoadsSelectable(true);
        }
        clearToolObjectSelection();
        delToolParameters();
        ODD::mainWindow()->showParameterDialog(false);
    }
}

void RoadLinkEditor::reject()
{
    ProjectEditor::reject();

    resetTool();
}

// ################//
// MOUSE & KEY    //
//################//




