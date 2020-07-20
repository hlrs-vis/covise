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
	, linkItem_(NULL)
	, sinkItem_(NULL)
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
			ODD::ToolId paramTool = getCurrentParameterTool();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<RoadLinkHandle> *param = new ToolValue<RoadLinkHandle>(ODD::TRL_LINK, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Arrow Handle");
				tool_ = new Tool(ODD::TRL_LINK, 4);
				tool_->readParams(param);
				ToolValue<RoadLinkSinkItem> *roadParam = new ToolValue<RoadLinkSinkItem>(ODD::TRL_SINK, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Circular Sink");
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
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<double> *param = new ToolValue<double>(ODD::TRL_THRESHOLD, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::DOUBLE, "Threshold");
				param->setValue(threshold_);
				tool_ = new Tool(ODD::TRL_ROADLINK, 4);
				tool_->readParams(param);
				ToolValue<RSystemElementRoad> *roadParam = new ToolValue<RSystemElementRoad>(ODD::TRL_ROADLINK, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
				tool_->readParams(roadParam);

				createToolParameterSettingsApplyBox(tool_, ODD::ERL);
				ODD::mainWindow()->showParameterDialog(true, "Link roads", "Specify threshold, SELECT/DESELECT roads and press APPLY");

				applyCount_ = 2;
			}
			else if (paramTool == ODD::TPARAM_SELECT)
			{
				ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
				if (action)
				{
					currentParamId_ = action->getParamId();
					if (!action->getState())
					{

						QList<RSystemElementRoad *>roads = tool_->removeToolParameters<RSystemElementRoad>(currentParamId_);
						foreach(RSystemElementRoad *road, roads)
						{
							DeselectDataElementCommand *command = new DeselectDataElementCommand(road, NULL);
							getProjectGraph()->executeCommand(command);
							selectedRoads_.removeOne(road);
						}

						// verify if apply has to be hidden //
						if (tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool()) <= applyCount_)
						{
							settingsApplyBox_->setApplyButtonVisible(false);
						}
					}
				}
			}
		}
		else if (action->getToolId() == ODD::TRL_UNLINK)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<RSystemElementRoad> *param = new ToolValue<RSystemElementRoad>(ODD::TRL_UNLINK, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
				tool_ = new Tool(ODD::TRL_UNLINK, 4);
				tool_->readParams(param);

				createToolParameterSettingsApplyBox(tool_, ODD::ERL);
				ODD::mainWindow()->showParameterDialog(true, "Unlink roads", "SELECT/DESELECT roads and press APPLY");

				applyCount_ = 2;
			}
			else if (paramTool == ODD::TPARAM_SELECT)
			{
				ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
				if (action)
				{
					currentParamId_ = action->getParamId();
					if (!action->getState())
					{

						QList<RSystemElementRoad *>roads = tool_->removeToolParameters<RSystemElementRoad>(currentParamId_);
						foreach(RSystemElementRoad *road, roads)
						{
							DeselectDataElementCommand *command = new DeselectDataElementCommand(road, NULL);
							getProjectGraph()->executeCommand(command);
							selectedRoads_.removeOne(road);
						}

						// verify if apply has to be hidden //
						if (tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool()) <= applyCount_)
						{
							settingsApplyBox_->setApplyButtonVisible(false);
						}
					}
				}
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
			if (action->getToolId() == ODD::TRL_LINK)
			{
				if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					ToolValue<RoadLinkHandle> *param = new ToolValue<RoadLinkHandle>(ODD::TRL_LINK, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Arrow Handle");
					tool_ = new Tool(ODD::TRL_LINK, 4);
					tool_->readParams(param);
					ToolValue<RoadLinkSinkItem> *roadParam = new ToolValue<RoadLinkSinkItem>(ODD::TRL_SINK, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Circular Sink");
					tool_->readParams(roadParam);

					generateToolParameterUI(tool_);
				}
			}
			else if (action->getToolId() == ODD::TRL_ROADLINK)
			{
				if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					ToolValue<double> *param = new ToolValue<double>(ODD::TRL_THRESHOLD, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::DOUBLE, "Threshold");
					param->setValue(threshold_);
					tool_ = new Tool(ODD::TRL_ROADLINK, 4);
					tool_->readParams(param);
					ToolValue<RSystemElementRoad> *roadParam = new ToolValue<RSystemElementRoad>(ODD::TRL_ROADLINK, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
					tool_->readParams(roadParam);

					generateToolParameterUI(tool_);
				}
			}
			else if (action->getToolId() == ODD::TRL_UNLINK)
			{
				if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					ToolValue<RSystemElementRoad> *param = new ToolValue<RSystemElementRoad>(ODD::TRL_UNLINK, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
					tool_ = new Tool(ODD::TRL_UNLINK, 4);
					tool_->readParams(param);
					generateToolParameterUI(tool_);
				}
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

void
RoadLinkEditor::apply()
{
	ODD::ToolId toolId = tool_->getToolId();
	if (toolId == ODD::TRL_ROADLINK)
	{

		for (int i = 0; i < selectedRoads_.size();)
		{
			RSystemElementRoad *road = selectedRoads_.at(i);
			if (road->getPredecessor() || road->getSuccessor())
			{
				selectedRoads_.takeAt(i);
			}
			else
			{ 
				removeZeroWidthLanes(road);             // error correction must not be undone TODO: move to laneeditor
				i++;
			}
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
		foreach(RSystemElementRoad *road, selectedRoads_)
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
		RoadLinkSinkItem *sink = dynamic_cast<ToolValue<RoadLinkSinkItem> *>(tool_->getParam(ODD::TRL_SINK, ODD::TPARAM_SELECT))->getValue();

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
					foreach(Lane *lane, lsC->getLanes())
					{
						if (lane->getId() > 0)
							newConnection->addLaneLink(lane->getId(), -lane->getId());
					}
				}
				else
				{
					foreach(Lane *lane, lsC->getLanes())
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
					foreach(Lane *lane, lsC->getLanes())
					{
						if (lane->getId() < 0)
							newConnection->addLaneLink(lane->getId(), lane->getId());
					}
				}
				else
				{
					foreach(Lane *lane, lsC->getLanes())
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

		linkItem_ = NULL;

	}
}

void
RoadLinkEditor::clearToolObjectSelection()
{
	if (sinkItem_)
	{
		sinkItem_->setSelected(false);
		sinkItem_ = NULL;
	}
	if (linkItem_)
	{
		linkItem_->setSelected(false);
		linkItem_ = NULL;
	}

	QList<DataElement *> dataElementList;
	foreach(RSystemElementRoad *road, selectedRoads_)
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
	ODD::ToolId toolId = tool_->getToolId();
	clearToolObjectSelection();
	delToolParameters();
}

void RoadLinkEditor::reject()
{
	ProjectEditor::reject();

	clearToolObjectSelection();
	deleteToolParameterSettings();
	ODD::mainWindow()->showParameterDialog(false);
}

// ################//
// MOUSE & KEY    //
//################//

/*! \brief .
*
*/
void
RoadLinkEditor::mouseAction(MouseAction *mouseAction)
{
	static QList<QGraphicsItem *> oldSelectedItems;

	QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();
	ProjectEditor::mouseAction(mouseAction);

	if ((getCurrentTool() == ODD::TRL_ROADLINK) || (getCurrentTool() == ODD::TRL_UNLINK))
	{
		if (getCurrentParameterTool() == ODD::TPARAM_SELECT)
		{
			if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
			{
				if (mouseAction->getEvent()->button() == Qt::LeftButton)
				{
					if (selectedRoads_.empty())
					{
						oldSelectedItems.clear();
					}

					QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
					QList<RSystemElementRoad *>selectionChangedRoads;
					QMap<RSystemElementRoad *, QGraphicsItem *>graphicRoadItems;

					for (int i = 0; i < selectedItems.size();)
					{
						QGraphicsItem *item = selectedItems.at(i);
						RoadLinkRoadItem *roadItem = dynamic_cast<RoadLinkRoadItem *>(item);
						if (roadItem)
						{
							RSystemElementRoad *road = roadItem->getRoad();
							if (!oldSelectedItems.contains(item))
							{
								if (!selectionChangedRoads.contains(road))
								{
									if (!selectedRoads_.contains(road))
									{
										createToolParameters<RSystemElementRoad>(road);
										selectedRoads_.append(road);

										item->setSelected(true);
									}
									else
									{
										item->setSelected(false);
										graphicRoadItems.insert(road, item);

										removeToolParameters<RSystemElementRoad>(road);
										selectedRoads_.removeOne(road);
									}
									selectionChangedRoads.append(road);
								}
								else if (!selectedRoads_.contains(road))
								{
									graphicRoadItems.insert(road, item);
								}
							}
							else
							{
								int j = oldSelectedItems.indexOf(item);
								oldSelectedItems.takeAt(j);
								graphicRoadItems.insert(road, item);
							}
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

					for (int i = 0; i < selectionChangedRoads.size(); i++)
					{
						RSystemElementRoad *road = selectionChangedRoads.at(i);
						if (!selectedRoads_.contains(road))
						{
							QGraphicsItem *roadItem = graphicRoadItems.value(road);
							selectedItems.removeOne(roadItem);
							oldSelectedItems.removeOne(roadItem);
							graphicRoadItems.remove(road);
						}
					}

					for (int i = 0; i < oldSelectedItems.size(); i++)
					{
						QGraphicsItem *item = oldSelectedItems.at(i);
						RoadLinkRoadItem *roadItem = dynamic_cast<RoadLinkRoadItem *>(item);
						if (roadItem)
						{
							RSystemElementRoad *road = roadItem->getRoad();
							if (!selectionChangedRoads.contains(road))
							{
								item->setSelected(false);

								removeToolParameters<RSystemElementRoad>(road);
								selectedRoads_.removeOne(road);

								selectionChangedRoads.append(road);
							}
						}
					}

					for (int i = 0; i < selectionChangedRoads.size(); i++)
					{
						RSystemElementRoad *road = selectionChangedRoads.at(i);
						if (!selectedRoads_.contains(road))
						{
							QGraphicsItem *roadItem = graphicRoadItems.value(road);
							selectedItems.removeOne(roadItem);
							graphicRoadItems.remove(road);
						}
					}
					oldSelectedItems = selectedItems;
					mouseAction->intercept();

					// verify if apply can be displayed //

					int objectCount = tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool());
					if (objectCount >= applyCount_)
					{
						settingsApplyBox_->setApplyButtonVisible(true);
					} 
				}
			}
		}
		else
		{
			mouseAction->intercept();
		}
	}
	else if (getCurrentTool() == ODD::TRL_LINK)
	{
		if (getCurrentParameterTool() == ODD::TPARAM_SELECT)
		{
			if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
			{
				if (mouseAction->getEvent()->button() == Qt::LeftButton)
				{

					QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
					for(int i = 0; i < selectedItems.size(); i++)
					{
						QGraphicsItem *item = selectedItems.at(i);
						RoadLinkHandle *roadLinkHandle = dynamic_cast<RoadLinkHandle *>(item);
						if (roadLinkHandle && (item != linkItem_))
						{
							if (linkItem_)
							{
								linkItem_->setSelected(false);
								int index = selectedItems.indexOf(linkItem_);
								if (index > i)
								{
									selectedItems.removeAt(index);
								}
							}

							RSystemElementRoad *road = roadLinkHandle->getParentRoadLinkItem()->getParentRoad();
							setToolValue<RoadLinkHandle>(roadLinkHandle, road->getIdName());
							
							linkItem_ = item;
							linkItem_->setSelected(true);
						}
						else if ((item != sinkItem_) && (item != linkItem_))
						{
							item->setSelected(false);
						}
					}
					mouseAction->intercept();

					// verify if apply can be displayed //
					if (tool_->verify())
					{
						settingsApplyBox_->setApplyButtonVisible(true);
					}
				}
			}
		}
		else
		{
			mouseAction->intercept();
		}
	}
	else if (getCurrentTool() == ODD::TRL_SINK)
	{
		if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
		{
			if (getCurrentParameterTool() == ODD::TPARAM_SELECT)
			{
				if (mouseAction->getEvent()->button() == Qt::LeftButton)
				{

					QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
					for (int i = 0; i < selectedItems.size(); i++)
					{
						QGraphicsItem *item = selectedItems.at(i);
						CircularHandle *linkSinkHandle = dynamic_cast<CircularHandle *>(item);
						if (linkSinkHandle && (item != sinkItem_))
						{
							RoadLinkSinkItem *linkSinkItem = dynamic_cast<RoadLinkSinkItem *>(linkSinkHandle->parentItem());
							if (linkSinkItem)
							{
								if (sinkItem_)
								{
									sinkItem_->setSelected(false);
									int index = selectedItems.indexOf(sinkItem_);
									if (index > i)
									{
										selectedItems.removeAt(index);
									}
								}

								setToolValue<RoadLinkSinkItem>(linkSinkItem, linkSinkItem->getParentRoad()->getIdName());
								sinkItem_ = item;
								sinkItem_->setSelected(true);
							}
						}
						else if ((item != linkItem_) && (item != sinkItem_))
						{
							item->setSelected(false);
						}
					}
					mouseAction->intercept();

					// verify if apply can be displayed //
					if (tool_->verify())
					{
						settingsApplyBox_->setApplyButtonVisible(true);
					}
				}
			}
		}

	}
}

                
    