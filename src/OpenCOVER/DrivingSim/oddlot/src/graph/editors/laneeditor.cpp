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

#include "laneeditor.hpp"

#include "src/mainwindow.hpp"

#include <qundostack.h>

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/commands/lanesectioncommands.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/roadlink.hpp"

#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/items/handles/lanemovehandle.hpp"

#include "src/graph/items/roadsystem/lanes/laneroadsystemitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanesectionitem.hpp"
#include "src/graph/items/roadsystem/lanes/laneitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

// Tools //
//
#include "src/gui/tools/laneeditortool.hpp"
#include "src/gui/mouseaction.hpp"

// GUI //
//
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

// Qt //
//
#include <QGraphicsItem>

//################//
// CONSTRUCTORS   //
//################//

LaneEditor::LaneEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , insertSectionHandle_(NULL)
	, pointHandle_(NULL)
	, laneItem_(NULL)
	, selectControls_(false)
{
}

LaneEditor::~LaneEditor()
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
LaneEditor::init()
{
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new LaneRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);

        // Section Handle //
        //
        insertSectionHandle_ = new SectionHandle(roadSystemItem_);
        insertSectionHandle_->hide();

		// Width Add Handle //
		//
		pointHandle_ = new PointHandle(roadSystemItem_);
		pointHandle_->hide();
    }

	if (selectControls_)
	{
		setItemsSelectable(false);
	}

}

/*!
*/
void
LaneEditor::kill()
{
    delete roadSystemItem_;
    roadSystemItem_ = NULL;
}

SectionHandle *
LaneEditor::getInsertSectionHandle() const
{
    if (!insertSectionHandle_)
    {
        qDebug("ERROR 1010151634! LaneEditor not yet initialized.");
    }
    return insertSectionHandle_;
}

PointHandle *
LaneEditor::getAddWidthHandle() const
{
	if (!pointHandle_)
	{
		qDebug("ERROR 1010151634! LaneEditor not yet initialized.");
	}
	return pointHandle_;
}

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
LaneEditor::toolAction(ToolAction *toolAction)
{
	lastTool_ = getCurrentTool();

    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

	LaneEditorToolAction *laneEditorToolAction = dynamic_cast<LaneEditorToolAction *>(toolAction);

	if (laneEditorToolAction)
	{
		if (laneEditorToolAction->getToolId() == ODD::TLE_SET_WIDTH)
		{

			if (selectedLaneMoveHandles_.size() > 0)
			{
				translateLaneBorder(QPointF(0, 0), QPointF(0, 0), laneEditorToolAction->getWidth(), true);
			}
		}
		else if (laneEditorToolAction->getToolId() == ODD::TLE_INSERT)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			getTopviewGraph()->getScene()->deselectAll();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<Lane> *laneParam = new ToolValue<Lane>(ODD::TLE_INSERT, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Lane");
				tool_ = new Tool(ODD::TLE_INSERT, 1);
				tool_->readParams(laneParam);
				ToolValue<int> *param = new ToolValue<int>(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_ID, 0, ToolParameter::ParameterTypes::INT, "New Lane ID");
				param->setValue(0);
				tool_->readParams(param);
				ToolValue<double> *widthParam = new ToolValue<double>(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_WIDTH, 0, ToolParameter::ParameterTypes::DOUBLE, "Lane Width");
				widthParam->setValue(3.0);
				tool_->readParams(widthParam);

				createToolParameterSettingsApplyBox(tool_, ODD::ELN);
				ODD::mainWindow()->showParameterDialog(true, "Insert new Lane", "Specify lane id and width, select a lane and press APPLY");

				applyCount_ = 1;
			}
		}
		else if (selectControls_ && (laneEditorToolAction->getToolId() == ODD::TLE_SELECT_ALL))
		{
			selectControls_ = false;
			if (lastTool_ == ODD::TLE_SELECT)
			{
				setItemsSelectable(true);
			}
		}
		else if (!selectControls_ && (laneEditorToolAction->getToolId() == ODD::TLE_SELECT_CONTROLS))
		{
			selectControls_ = true;
			setItemsSelectable(false);
		}
	}
	else
	{
		ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
		if (action)
		{
			if (action->getToolId() == ODD::TLE_INSERT)
			{
				if (action->getParamToolId() == ODD::TLE_INSERT_LANE_ID)
				{
					ToolParameter *p = tool_->getParam(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_ID);
					ToolValue<int> *v = dynamic_cast<ToolValue<int> *>(p);
					int laneId = *v->getValue();
					Lane *lane = dynamic_cast<ToolValue<Lane> *>(tool_->getParam(ODD::TLE_INSERT, ODD::TPARAM_SELECT))->getValue();

					if (lane)
					{
						int rightmost = lane->getParentLaneSection()->getRightmostLaneId() - 1;
						int leftmost = lane->getParentLaneSection()->getLeftmostLaneId() + 1;

						if (laneId < rightmost)
						{
							v->setValue(rightmost);
							updateToolParameterUI(p);
						}
						else if (laneId > leftmost)
						{
							v->setValue(leftmost);
							updateToolParameterUI(p);
						}
					}
				}
				else if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					ToolValue<Lane> *laneParam = new ToolValue<Lane>(ODD::TLE_INSERT, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Lane");
					tool_ = new Tool(ODD::TLE_INSERT, 1);
					tool_->readParams(laneParam);
					ToolValue<int> *param = new ToolValue<int>(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_ID, 0, ToolParameter::ParameterTypes::INT, "New Lane ID");
					param->setValue(0);
					tool_->readParams(param);
					ToolValue<double> *widthParam = new ToolValue<double>(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_WIDTH, 0, ToolParameter::ParameterTypes::DOUBLE, "Lane Width");
					widthParam->setValue(3.0);
					tool_->readParams(widthParam);

					generateToolParameterUI(tool_);
				}
			}
		}
	}
}

void
LaneEditor::mouseAction(MouseAction *mouseAction)
{
	QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();
	ODD::ToolId tool = getCurrentTool();

	if ((tool != lastTool_) && (tool != ODD::TLE_SELECT_CONTROLS) && (tool != ODD::TLE_SELECT_ALL))
	{
		if (selectControls_)
		{
			if ((tool == ODD::TLE_SELECT) && (lastTool_ != ODD::TLE_SELECT_CONTROLS))
			{
				setItemsSelectable(false);
			}
			else if (tool != ODD::TLE_SELECT) 
			{
				setItemsSelectable(true);
			}
		}

		lastTool_ = tool;
	}


	if (tool == ODD::TTE_ROAD_NEW)
	{
		QPointF mousePoint = mouseAction->getEvent()->scenePos();
	}
	else if (tool == ODD::TLE_INSERT)
	{
		if (getCurrentParameterTool() == ODD::TPARAM_SELECT)
		{
			if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
			{
				if (mouseAction->getEvent()->button() == Qt::LeftButton)
				{
					QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
					for (int i = 0; i < selectedItems.size(); i++)
					{
						QGraphicsItem *item = selectedItems.at(i);
						LaneItem *laneItem = dynamic_cast<LaneItem *>(item);
						if (laneItem && (item != laneItem_))
						{
							if (laneItem_)
							{
								laneItem_->setSelected(false);
								int index = selectedItems.indexOf(laneItem_);
								if (index > i)
								{
									selectedItems.removeAt(index);
								}
							}

							Lane *lane = laneItem->getLane();
							QString textDisplayed = QString("%1 Lane %2").arg(lane->getParentLaneSection()->getParentRoad()->getIdName()).arg(lane->getId());
							setToolValue<Lane>(lane, textDisplayed);

							laneItem_ = item;

							ToolValue<int> *v = dynamic_cast<ToolValue<int> *>(tool_->getParam(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_ID));
							v->setValue(lane->getId());
							updateToolParameterUI(v);
						}
						else if (item != laneItem_)
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

void 
LaneEditor::setItemsSelectable(bool selectable)
{
	QList<QGraphicsItem *> itemList = getTopviewGraph()->getScene()->items();
	foreach(QGraphicsItem * item, itemList)
	{
		Handle *handle = dynamic_cast<Handle *>(item);
		if (!handle)
		{
			GraphElement *graphElement = dynamic_cast<GraphElement *>(item);
			if (graphElement)
			{
				item->setFlag(QGraphicsItem::ItemIsSelectable, selectable);
				item->setSelected(graphElement->getDataElement()->isElementSelected());
			}
		}
	}
}


//################//
// LaneMoveHandles    //
//################//

void
LaneEditor::registerMoveHandle(BaseLaneMoveHandle *handle)
{
	selectedLaneMoveHandles_.append(handle);
}

int
LaneEditor::unregisterMoveHandle(BaseLaneMoveHandle *handle)
{
	return selectedLaneMoveHandles_.removeOne(handle);
} 

bool
LaneEditor::translateLaneBorder(const QPointF &pressPos, const QPointF &mousePosConst, double width, bool setWidth)
{

	QMap<RSystemElementRoad *, QMultiMap<double, LaneMoveProperties *>> selectedLaneMoveProps;
	foreach (BaseLaneMoveHandle *baseMoveHandle, selectedLaneMoveHandles_)
	{
		LaneMoveProperties *props = new LaneMoveProperties();
		LaneWidth *lowSlot = baseMoveHandle->getLowSlot();
		RSystemElementRoad *road;
		double s;
		if (lowSlot)
		{
			props->lowSlot = lowSlot;
			road = lowSlot->getParentLane()->getParentLaneSection()->getParentRoad();
			s = lowSlot->getSSectionEnd();
		}
		LaneWidth *highSlot = baseMoveHandle->getHighSlot();
		if (highSlot)
		{
			props->highSlot = highSlot;
			if (!lowSlot)
			{
				road = highSlot->getParentLane()->getParentLaneSection()->getParentRoad();
				s = highSlot->getSSectionStartAbs();
			}
		}

		QMultiMap<double, LaneMoveProperties *> propsMap;
		if (selectedLaneMoveProps.find(road) != selectedLaneMoveProps.end())
		{
			propsMap = selectedLaneMoveProps.value(road);
		}

		propsMap.insert(s, props);
		selectedLaneMoveProps.insert(road, propsMap);
	}
	
	if (setWidth)
	{
		TranslateLaneBorderCommand *command = new TranslateLaneBorderCommand(selectedLaneMoveProps, width, QPointF(0,0), NULL);
		return getProjectGraph()->executeCommand(command);
	}
	else
	{
		QPointF mousePos = mousePosConst;

		// Snap to MoveHandle //
		//
		foreach(QGraphicsItem *item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
		{
			BaseLaneMoveHandle *handle = dynamic_cast<BaseLaneMoveHandle *>(item);

			if (handle)
			{
				mousePos = handle->pos();
				break;
			}
		}

		if ((mousePos - pressPos).manhattanLength() < NUMERICAL_ZERO6)
		{
			return false;
		}

		// DeltaPos  //
		//
		QPointF dPos = mousePos - pressPos;

		TranslateLaneBorderCommand *command = new TranslateLaneBorderCommand(selectedLaneMoveProps, width, dPos, NULL);
		return getProjectGraph()->executeCommand(command);
	}

}

//################//
// SLOTS          //
//################//
void
LaneEditor::apply()
{
	clearToolObjectSelection();

	ODD::ToolId toolId = tool_->getToolId();
	if (toolId == ODD::TLE_INSERT)
	{
		int laneId = *dynamic_cast<ToolValue<int> *>(tool_->getParam(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_ID))->getValue();
		double width = *dynamic_cast<ToolValue<double> *>(tool_->getParam(ODD::TLE_INSERT, ODD::TLE_INSERT_LANE_WIDTH))->getValue();
		Lane *lane = dynamic_cast<ToolValue<Lane> *>(tool_->getParam(ODD::TLE_INSERT, ODD::TPARAM_SELECT))->getValue();

		Lane *newLane = new Lane(laneId, Lane::LT_DRIVING);
		LaneSection *parentLaneSection = lane->getParentLaneSection();

		if (laneId != 0)
		{

			LaneWidth *laneWidth = new LaneWidth(0.0, width, 0.0, 0.0, 0.0);
			newLane->addWidthEntry(laneWidth);

			LaneRoadMark *roadMark = new LaneRoadMark(0.0, LaneRoadMark::RMT_SOLID, LaneRoadMark::RMW_STANDARD, LaneRoadMark::RMC_STANDARD, 0.12);
			newLane->addRoadMarkEntry(roadMark);

		}

		InsertLaneCommand *command = new InsertLaneCommand(parentLaneSection, newLane, lane);
		if (command)
		{
			getProjectGraph()->executeCommand(command);
		}


		DeselectDataElementCommand *deselectCommand = new DeselectDataElementCommand(lane, NULL);
		getProjectGraph()->executeCommand(deselectCommand);
	}
}

void
LaneEditor::clearToolObjectSelection()
{
	if (laneItem_)
	{
		laneItem_->setSelected(false);
		laneItem_ = NULL;
	}
}

void
LaneEditor::reset()
{
	ODD::ToolId toolId = tool_->getToolId();
	clearToolObjectSelection();
	delToolParameters();
}

void
LaneEditor::reject()
{
	ProjectEditor::reject();

	clearToolObjectSelection();
	deleteToolParameterSettings();
	ODD::mainWindow()->showParameterDialog(false);
}
