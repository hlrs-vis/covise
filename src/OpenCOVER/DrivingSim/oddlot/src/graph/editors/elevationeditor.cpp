/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.06.2010
**
**************************************************************************/

#include "elevationeditor.hpp"

#include "src/mainwindow.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/commands/elevationsectioncommands.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"

#include "src/graph/items/roadsystem/elevation/elevationroadsystemitem.hpp"
#include "src/graph/items/roadsystem/elevation/elevationroadpolynomialitem.hpp"
#include "src/graph/items/roadsystem/elevation/elevationsectionpolynomialitem.hpp"
#include "src/graph/items/roadsystem/elevation/elevationsectionitem.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/elevation/elevationmovehandle.hpp"

// Tools //
//
#include "src/gui/mouseaction.hpp"
#include "src/gui/tools/elevationeditortool.hpp"
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

// Qt //
//
#include <QGraphicsItem>

//################//
// CONSTRUCTORS   //
//################//

ElevationEditor::ElevationEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , profileGraph_(profileGraph)
    , roadSystemItemPolyGraph_(NULL)
    , insertSectionHandle_(NULL)
    , smoothRadius_(900.0)
	, slope_(10.0)
    , xtrans_(0.0)
	, elevationSectionItem_(NULL)
	, elevationSectionAdjacentItem_(NULL)
	, selectedElevationItem_(NULL)
{
}

ElevationEditor::~ElevationEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

SectionHandle *
ElevationEditor::getInsertSectionHandle()
{
    if (!insertSectionHandle_)
    {
        qDebug("ERROR 1006211555! ElevationEditor not yet initialized.");
    }
    return insertSectionHandle_;
}

/*! \brief Adds a road to the list of selected roads.
*/

void
ElevationEditor::insertSelectedRoad(RSystemElementRoad *road)
{
	if (getCurrentTool() == ODD::TEL_SELECT)
	{
		if (!selectedRoads_.contains(road))
		{
			selectedRoads_.append(road);
			QList<DataElement*> sectionList;
			foreach(ElevationSection * section, road->getElevationSections())
			{
				if (!section->isElementSelected())
				{
					sectionList.append(section);
				}
			}

			int listSize = selectedRoads_.size();
			if (listSize == 1)
			{
				if (!selectedElevationItem_)
				{
					selectedElevationItem_ = new ElevationRoadPolynomialItem(roadSystemItemPolyGraph_, road);
				}
			}
			else if (listSize == 2)
			{
				selectedElevationItem_->registerForDeletion();
				selectedElevationItem_ = NULL; 
			}

			SelectDataElementCommand* command = new SelectDataElementCommand(sectionList);
			getProjectGraph()->executeCommand(command);
		}
	}
}

/*! \brief Initialize BoundingBox and offset.
*/

void
ElevationEditor::initBox()
{
    xtrans_ = 0.0;
    boundingBox_.setRect(0.0, 0.0, 0.0, 0.0);
}

/*
* Calls fitInView for the selected road and displays it in the ProfileGraph.
*/
void
ElevationEditor::addSelectedRoad(ElevationRoadPolynomialItem *roadItem)
{

    // Compute the BoundingBox of all selected roads //
    //
    QRectF roadItemBoundingBox = roadItem->translate(xtrans_, 0.0);

    boundingBox_ = boundingBox_.united(roadItemBoundingBox);
    xtrans_ = boundingBox_.width();
}

void
ElevationEditor::fitView()
{
    if (boundingBox_.width() < 15.0)
    {
        boundingBox_.setWidth(15.0);
    }
    if (boundingBox_.height() < 15.0)
    {
        boundingBox_.setHeight(15.0);
    }

    profileGraph_->getView()->fitInView(boundingBox_);
    profileGraph_->getView()->zoomOut(Qt::Horizontal | Qt::Vertical);
}

void
ElevationEditor::delSelectedRoad(RSystemElementRoad *road)
{
	if (selectedRoads_.contains(road))
	{
		selectedRoads_.removeAll(road);

		QList<DataElement*> sectionList;
		foreach(ElevationSection * section, road->getElevationSections())
		{
			if (section->isElementSelected())
			{
				sectionList.append(section);
			}
		}
	
		int listSize = selectedRoads_.size();
		if (listSize == 1)
		{
			if (!selectedElevationItem_)
			{
				selectedElevationItem_ = new ElevationRoadPolynomialItem(roadSystemItemPolyGraph_, selectedRoads_.first());

				DeselectDataElementCommand* command = new DeselectDataElementCommand(sectionList);
				getProjectGraph()->executeCommand(command);
			}
		}
		else
		{
			DeselectDataElementCommand* command = new DeselectDataElementCommand(sectionList);
			getProjectGraph()->executeCommand(command);

			if (listSize == 0)
			{
				selectedElevationItem_->registerForDeletion();
				selectedElevationItem_ = NULL;
			}
		}
	}
}

//################//
// TOOL           //
//################//

/*! \brief ToolAction 'Chain of Responsibility'.
*
*/
void
ElevationEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    // Tools //
    //

    //	if(getCurrentTool() == ODD::TEL_SELECT)
    //	{
    //		// does nothing //
    //	}
    //	else if(getCurrentTool() == ODD::TEL_ADD)
    //	{
    //		// does nothing //
    //	}
    //	else if(getCurrentTool() == ODD::TEL_DEL)
    //	{
    //		// does nothing //
    //		// Note the problem: The ToolAction is re-sent, after a warning message has been clicked away. (Due to re-send on getting the focus back?)
    //	}

    // Tools //
    //
	ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(toolAction);
	if (elevationEditorToolAction)
	{
		if (elevationEditorToolAction->getToolId() == ODD::TEL_SELECT)
		{
			if (elevationEditorToolAction->getParamToolId() == ODD::TEL_RADIUS)
			{
				smoothRadius_ = elevationEditorToolAction->getRadius();
			}
			else if ((elevationEditorToolAction->getParamToolId() == ODD::TEL_HEIGHT) || (elevationEditorToolAction->getParamToolId() == ODD::TEL_IHEIGHT))
			{
				if (selectedMoveHandles_.size() > 0)
				{
					QList<ElevationSection *> endPointSections;
					QList<ElevationSection *> startPointSections;
					foreach(ElevationMoveHandle *moveHandle, selectedMoveHandles_)
					{
						ElevationSection *lowSlot = moveHandle->getLowSlot();
						if (lowSlot)
						{
							endPointSections.append(lowSlot);
						}

						ElevationSection *highSlot = moveHandle->getHighSlot();
						if (highSlot)
						{
							startPointSections.append(highSlot);
						}
					}

					// Command //
					//
					ElevationSetHeightCommand *command;
					if (elevationEditorToolAction->getParamToolId() == ODD::TEL_HEIGHT)
					{
						command = new ElevationSetHeightCommand(endPointSections, startPointSections, elevationEditorToolAction->getHeight(), true, NULL);
					}
					else
					{
						command = new ElevationSetHeightCommand(endPointSections, startPointSections, elevationEditorToolAction->getIHeight(), false, NULL);
					}
					if (command->isValid())
					{
						getProjectData()->getUndoStack()->push(command);

						// Message //
						//
						printStatusBarMsg(QString("setHeight to: %1").arg(elevationEditorToolAction->getHeight()), 0);
					}
					else
					{
						if (command->text() != "")
						{
							printStatusBarMsg(command->text(), 0);
						}
						delete command;
					}
				}
			}
			else if (elevationEditorToolAction->getParamToolId() == ODD::TEL_MOVE)
			{

				QList<ElevationSection *> endPointSections;
				QList<ElevationSection *> startPointSections;

				foreach(ElevationMoveHandle *moveHandle, selectedMoveHandles_)
				{
					ElevationSection *lowSlot = moveHandle->getLowSlot();
					if (lowSlot)
					{
						if (lowSlot->getDegree() > 1)
						{
							return;
						}
						endPointSections.append(lowSlot);
					}

					ElevationSection *highSlot = moveHandle->getHighSlot();
					if (!highSlot || (highSlot->getDegree() > 1))
					{
						return;
					}
					else
					{
						startPointSections.append(highSlot);
					}


					// Command //
					//
					QPointF dPos = QPointF(elevationEditorToolAction->getSectionStart() - highSlot->getSStart(), 0.0);
					ElevationMovePointsCommand *command = new ElevationMovePointsCommand(endPointSections, startPointSections, dPos, NULL);

					if (command->isValid())
					{
						getProjectData()->getUndoStack()->push(command);
					}
					else
					{
						delete command;
					}
				}
			}
		}
		else if ((elevationEditorToolAction->getToolId() == ODD::TEL_ADD) || (elevationEditorToolAction->getToolId() == ODD::TEL_DEL))
		{
			getTopviewGraph()->getScene()->deselectAll();
		}
		else if (elevationEditorToolAction->getToolId() == ODD::TEL_SMOOTH)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			getTopviewGraph()->getScene()->deselectAll();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<ElevationSection> *elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection");
				tool_ = new Tool(ODD::TEL_SMOOTH, 1);
				tool_->readParams(elevationSectionParam);
				ToolValue<ElevationSection> *adjacentSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Adjacent Section");
				tool_->readParams(adjacentSectionParam);

				createToolParameterSettingsApplyBox(tool_, ODD::EEL);
				ODD::mainWindow()->showParameterDialog(true, "Smooth Heights across Roads", "Specify smoothing radius, select adjacent elevation sections on two roads and press APPLY");
			}
		}
		else if (elevationEditorToolAction->getToolId() == ODD::TEL_SLOPE)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			getTopviewGraph()->getScene()->deselectAll();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<double> *slopeParam = new ToolValue<double>(ODD::TEL_PERCENTAGE, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::DOUBLE, "Slope Percentage");
				slopeParam->setValue(slope_);
				tool_ = new Tool(ODD::TEL_SLOPE, 1);
				tool_->readParams(slopeParam);
				ToolValue<ElevationSection> *elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SLOPE, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection");
				tool_->readParams(elevationSectionParam);

				createToolParameterSettingsApplyBox(tool_, ODD::EEL);
				ODD::mainWindow()->showParameterDialog(true, "Change Slope of Elevation Section", "Specify slope percentage, select elevation section and press APPLY");
			}
		} 
	}
	else
	{
		ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
		if (action)
		{
			if (action->getToolId() == ODD::TEL_SMOOTH)
			{
				if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					ToolValue<ElevationSection> *elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection");
					tool_ = new Tool(ODD::TEL_SMOOTH, 1);
					tool_->readParams(elevationSectionParam);
					ToolValue<ElevationSection> *adjacentSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Adjacent Section");
					tool_->readParams(adjacentSectionParam);

					generateToolParameterUI(tool_);
				}
			}
			else if (action->getToolId() == ODD::TEL_SLOPE)
			{
				if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					ToolValue<double> *slopeParam = new ToolValue<double>(ODD::TEL_PERCENTAGE, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::DOUBLE, "Slope Percentage");
					slopeParam->setValue(slope_);
					tool_ = new Tool(ODD::TEL_SLOPE, 1);
					tool_->readParams(slopeParam);
					ToolValue<ElevationSection> *elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SLOPE, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection");
					tool_->readParams(elevationSectionParam);

					generateToolParameterUI(tool_);
				}
			}
			else if (action->getToolId() == ODD::TEL_PERCENTAGE)
			{
				slope_ = action->getValue();
			}
		}
	}
}

//################//
// MOUSE & KEY    //
//################//

/*! \brief .
*
*/
void
ElevationEditor::mouseAction(MouseAction *mouseAction)
{
	QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();

	// SMOOTH //
	//
	if ((getCurrentTool() == ODD::TEL_SMOOTH) || (getCurrentTool() == ODD::TEL_SLOPE))
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
						ElevationSectionItem *sectionItem = dynamic_cast<ElevationSectionItem *>(item);
						if (sectionItem && (item != elevationSectionItem_) && (item != elevationSectionAdjacentItem_))
						{
							if (elevationSectionItem_)
							{
								elevationSectionItem_->setSelected(false);
								int index = selectedItems.indexOf(elevationSectionItem_);
								if (index > i)
								{
									selectedItems.removeAt(index);
								}
							}

							ElevationSection *elevationSection = sectionItem->getElevationSection();
							QString textDisplayed = QString("%1 Section at %2").arg(elevationSection->getParentRoad()->getIdName()).arg(elevationSection->getSStart());
							setToolValue<ElevationSection>(elevationSection, textDisplayed);

							elevationSectionItem_ = item;
						}
						else if ((item != elevationSectionItem_) && (item != elevationSectionAdjacentItem_))
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
	else if (getCurrentTool() == ODD::TEL_SMOOTH_SECTION)
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
						ElevationSectionItem *sectionItem = dynamic_cast<ElevationSectionItem *>(item);
						if (sectionItem && (item != elevationSectionItem_) && (item != elevationSectionAdjacentItem_))
						{
							if (elevationSectionAdjacentItem_)
							{
								elevationSectionAdjacentItem_->setSelected(false);
								int index = selectedItems.indexOf(elevationSectionAdjacentItem_);
								if (index > i)
								{
									selectedItems.removeAt(index);
								}
							}

							ElevationSection *elevationSection = sectionItem->getElevationSection();
							QString textDisplayed = QString("%1 Section at %2").arg(elevationSection->getParentRoad()->getIdName()).arg(elevationSection->getSStart());
							setToolValue<ElevationSection>(elevationSection, textDisplayed);

							elevationSectionAdjacentItem_ = item;
						}
						else if ((item != elevationSectionItem_) && (item != elevationSectionAdjacentItem_))
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

//################//
// MoveHandles    //
//################//

void
ElevationEditor::registerMoveHandle(ElevationMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! ElevationEditor ElevationMoveHandle DOF not in [0,1,2].");
    }
    selectedMoveHandles_.insert(handle->getPosDOF(), handle);


    // The observers have to be notified, that a different handle has been selected
    // 2: Two degrees of freedom //
    //
    QList<ElevationSection *> endPointSections;
    QList<ElevationSection *> startPointSections;
    foreach (ElevationMoveHandle *moveHandle, selectedMoveHandles_)
    {
        ElevationSection *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointSections.append(lowSlot);
        }

        ElevationSection *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointSections.append(highSlot);
        }
    }

    // Command //
    //
    SelectElevationSectionCommand *command = new SelectElevationSectionCommand(endPointSections, startPointSections, NULL);

    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);

        // Message //
        //
        printStatusBarMsg(QString("Select Elevation Section "), 0);
    }
    else
    {
        if (command->text() != "")
        {
            printStatusBarMsg(command->text(), 0);
        }
        delete command;
    }
}

int
ElevationEditor::unregisterMoveHandle(ElevationMoveHandle *handle)
{
    return selectedMoveHandles_.remove(handle->getPosDOF(), handle);
}

bool
ElevationEditor::translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos)
{
    QPointF dPos = mousePos - pressPos;

    // No entries //
    //
    if (selectedMoveHandles_.size() == 0)
    {
        return false;
    }

    // 0: Check for zero degrees of freedom //
    //
    if (selectedMoveHandles_.count(0) > 0)
    {
        return false;
    }

    // 1: Check for one degree of freedom //
    //
    if (selectedMoveHandles_.count(1) > 0)
    {
        printStatusBarMsg(tr("Sorry, you can't move yellow items."), 0);
        qDebug("One DOF not supported yet");
        return false;
    }

    // 2: Two degrees of freedom //
    //
    QList<ElevationSection *> endPointSections;
    QList<ElevationSection *> startPointSections;
    foreach (ElevationMoveHandle *moveHandle, selectedMoveHandles_)
    {
        ElevationSection *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointSections.append(lowSlot);
        }

        ElevationSection *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointSections.append(highSlot);
        }
    }

    // Command //
    //
    ElevationMovePointsCommand *command = new ElevationMovePointsCommand(endPointSections, startPointSections, dPos, NULL);

    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);

        // Message //
        //
        printStatusBarMsg(QString("Move to: %1, %2").arg(pressPos.x()).arg(pressPos.y() + dPos.y()), 0);
    }
    else
    {
        if (command->text() != "")
        {
            printStatusBarMsg(command->text(), 0);
        }
        delete command;
    }

    return true;
}

bool
ElevationEditor::selectionChangedRoadSection()
{
	QList<ElevationSection *> endPointSections;
	QList<ElevationSection *> startPointSections;
	QList<QGraphicsItem *> selectedItems = getProfileGraph()->getScene()->selectedItems();
	foreach(QGraphicsItem *item, selectedItems)
	{
		ElevationMoveHandle *moveHandle = dynamic_cast<ElevationMoveHandle *>(item);
		if (moveHandle)
		{
			ElevationSection *lowSlot = moveHandle->getLowSlot();
			if (lowSlot)
			{
				endPointSections.append(lowSlot);
			}

			ElevationSection *highSlot = moveHandle->getHighSlot();
			if (highSlot)
			{
				startPointSections.append(highSlot);
			}
		}
	}


	bool deselect = false;
	getProjectData()->getUndoStack()->beginMacro("Select RoadSection");
	selectedItems = getTopviewGraph()->getScene()->selectedItems();
	foreach(QGraphicsItem *item, selectedItems)
	{
		ElevationSectionItem *sectionItem = dynamic_cast<ElevationSectionItem *>(item);
		if (sectionItem)
		{
			ElevationSection *elevationSection = sectionItem->getElevationSection();
			if (!endPointSections.contains(elevationSection) && !startPointSections.contains(elevationSection))
			{
				DeselectDataElementCommand *deselectCommand = new DeselectDataElementCommand(elevationSection);
				if (deselectCommand->isValid())
				{
					getProjectData()->getUndoStack()->push(deselectCommand);
					deselect = true;
				}
			}
		}
	}

/*	foreach(ElevationSection *section, endPointSections)
	{
		if (!section->isElementSelected())
		{
			SelectDataElementCommand *selectCommand = new SelectDataElementCommand(section);
			if (selectCommand->isValid())
			{
				getProjectData()->getUndoStack()->push(selectCommand);
				insertSelectedRoad(section->getParentRoad());
			}
		}
	} */
	foreach(ElevationSection *section, startPointSections)
	{
		if (!section->isElementSelected())
		{
			SelectDataElementCommand *selectCommand = new SelectDataElementCommand(section);
			if (selectCommand->isValid())
			{
				getProjectData()->getUndoStack()->push(selectCommand);
			}
		}
	} 
	getProjectData()->getUndoStack()->endMacro();

	return deselect;
}

//################//
// SLOTS          //
//################//

/*!
*
*/
void
ElevationEditor::init()
{

	// ProfileGraph //
	//
	if (!roadSystemItemPolyGraph_)
	{
		// Root item //
		//
		roadSystemItemPolyGraph_ = new RoadSystemItem(profileGraph_, getProjectData()->getRoadSystem());
		profileGraph_->getScene()->addItem(roadSystemItemPolyGraph_);
		profileGraph_->getScene()->setSceneRect(-1000.0, -1000.0, 20000.0, 2000.0);
	}

    // Graph //
    //
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new ElevationRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);
    }

    // Section Handle //
    //
    // TODO: Is this really the best object for holding this?
    insertSectionHandle_ = new SectionHandle(roadSystemItem_);
    insertSectionHandle_->hide();
}

/*!
*/
void
ElevationEditor::kill()
{
	selectedRoads_.clear();
	if (selectedElevationItem_ && !selectedElevationItem_->isInGarbage())
	{
		selectedElevationItem_->registerForDeletion();
		selectedElevationItem_ = NULL;
	} 

    delete roadSystemItem_;
    roadSystemItem_ = NULL;

    delete roadSystemItemPolyGraph_;
    roadSystemItemPolyGraph_ = NULL;
}

void
ElevationEditor::apply()
{
	clearToolObjectSelection();

	ODD::ToolId toolId = tool_->getToolId();
	if (toolId == ODD::TEL_SMOOTH)
	{
		ElevationSection *firstSection = dynamic_cast<ToolValue<ElevationSection> *>(tool_->getParam(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT))->getValue();
		ElevationSection *secondSection = dynamic_cast<ToolValue<ElevationSection> *>(tool_->getParam(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT))->getValue();
	
		SmoothElevationRoadsCommand *command = new SmoothElevationRoadsCommand(firstSection, secondSection, smoothRadius_);
		getProjectGraph()->executeCommand(command);
	}
	else if (toolId == ODD::TEL_SLOPE)
	{
		QList<ElevationSection *> endPointSections;
		QList<ElevationSection *> startPointSections;

		ElevationSection *section = dynamic_cast<ToolValue<ElevationSection> *>(tool_->getParam(ODD::TEL_SLOPE, ODD::TPARAM_SELECT))->getValue();
		ElevationSection *sectionNext = section->getParentRoad()->getElevationSectionNext(section->getSStart());


		if (sectionNext)
		{
			endPointSections.append(section);
			startPointSections.append(sectionNext);

			double s = 100 * fabs(section->getElevation(section->getSStart()) - section->getElevation(section->getSEnd())) / slope_ + section->getSStart();
			if (s < section->getParentRoad()->getLength())
			{
				QPointF dPos = QPointF(s - sectionNext->getSStart(), 0.0);
				ElevationMovePointsCommand *command = new ElevationMovePointsCommand(endPointSections, startPointSections, dPos, NULL);

				if (command->isValid())
				{
					getProjectData()->getUndoStack()->push(command);
				}
				else
				{
					delete command;
				}
			}
		}
	} 
}

void
ElevationEditor::clearToolObjectSelection()
{
	if (elevationSectionItem_)
	{
		elevationSectionItem_->setSelected(false);
		elevationSectionItem_ = NULL;
	}

	if (elevationSectionAdjacentItem_)
	{
		elevationSectionAdjacentItem_->setSelected(false);
		elevationSectionAdjacentItem_ = NULL;
	}
}

void
ElevationEditor::reset()
{
	ODD::ToolId toolId = tool_->getToolId();
	clearToolObjectSelection();
	delToolParameters();
}

void
ElevationEditor::reject()
{
	ProjectEditor::reject();

	clearToolObjectSelection();
	deleteToolParameterSettings();
	ODD::mainWindow()->showParameterDialog(false);
}
