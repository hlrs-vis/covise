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

#include <qundostack.h>

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/commands/lanesectioncommands.hpp"
#include "src/data/visitors/trackmovevalidator.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/items/roadsystem/lanes/lanewidthmovehandle.hpp"
#include "src/graph/items/handles/lanemovehandle.hpp"

#include "src/graph/items/roadsystem/lanes/laneroadsystemitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanesectionitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

// Tools //
//
#include "src/gui/tools/laneeditortool.hpp"
#include "src/gui/mouseaction.hpp"

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

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
LaneEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

	if (getCurrentTool() == ODD::TLE_SELECT)
	{
		LaneEditorToolAction *laneEditorToolAction = dynamic_cast<LaneEditorToolAction *>(toolAction);
		if (laneEditorToolAction)
		{
			LaneEditorToolAction::ActionType type = laneEditorToolAction->getType();
			if (selectedLaneMoveHandles_.size() > 0 && (type == LaneEditorToolAction::Width))
			{
				translateLaneBorder(QPointF(0, 0), QPointF(0, 0), laneEditorToolAction->getWidth(), true);
			}
		}
	}

    // Tools //
    //
	// Move & Rotate Tool //
	//
/*	if (getCurrentTool() == ODD::TLE_MODIFY_POLY)
	{
		if (roadSystemItem_)
		{
			roadSystemItem_->rebuildMoveRotateHandles();
		}
	}
	else */
/*	{
		LaneEditorToolAction *laneEditorToolAction = dynamic_cast<LaneEditorToolAction *>(toolAction);
		if (laneEditorToolAction)
		{
			LaneEditorToolAction::ActionType type = laneEditorToolAction->getType();
			if (selectedMoveHandles_.size() > 0 && (type == LaneEditorToolAction::Width))
			{

				QList<LaneWidth *> endPointWidths;
				QList<LaneWidth *> startPointWidths;
				foreach(LaneWidthMoveHandle *moveHandle, selectedMoveHandles_)
				{
					LaneWidth *lowSlot = moveHandle->getLowSlot();
					if (lowSlot)
					{
						endPointWidths.append(lowSlot);
					}

					LaneWidth *highSlot = moveHandle->getHighSlot();
					if (highSlot)
					{
						startPointWidths.append(highSlot);
					}
				}

				// Command //
				//
				LaneSetWidthCommand *command = new LaneSetWidthCommand(endPointWidths, startPointWidths, laneEditorToolAction->getWidth(), laneEditorToolAction->getType() == LaneEditorToolAction::Width, NULL);
				if (command->isValid())
				{
					getProjectData()->getUndoStack()->push(command);

					// Message //
					//
					printStatusBarMsg(QString("setWidth to: %1").arg(laneEditorToolAction->getWidth()), 1000);
				}
				else
				{
					if (command->text() != "")
					{
						printStatusBarMsg(command->text(), 4000);
					}
					delete command;
				}
			} */
			// Editing Mode //
			//
/*			else if (type == LaneEditorToolAction::EditMode)
			{
				borderEditMode_ = !borderEditMode_;
			} 
		}
	} */
}

void
LaneEditor::mouseAction(MouseAction *mouseAction)
{
	QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();

	if (getCurrentTool() == ODD::TTE_ROAD_NEW)
	{
		QPointF mousePoint = mouseAction->getEvent()->scenePos();
	}
}

//################//
// MoveHandles    //
//################//

void
LaneEditor::registerMoveHandle(LaneWidthMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! ElevationEditor ElevationMoveHandle DOF not in [0,1,2].");
    }
    selectedMoveHandles_.insert(handle->getPosDOF(), handle);



    // The observers have to be notified, that a different handle has been selected
    // 2: Two degrees of freedom //
    //
    QList<LaneWidth *> endPointWidths;
    QList<LaneWidth *> startPointWidths;
    foreach (LaneWidthMoveHandle *moveHandle, selectedMoveHandles_)
    {
        LaneWidth *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointWidths.append(lowSlot);
        }

        LaneWidth *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointWidths.append(highSlot);
        }
    }

    // Command //
    //
    SelectLaneWidthCommand *command = new SelectLaneWidthCommand(endPointWidths, startPointWidths, NULL);

    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);

        // Message //
        //
        printStatusBarMsg(QString("Select Lane Width "), 100);
    }
    else
    {
        if (command->text() != "")
        {
            printStatusBarMsg(command->text(), 4000);
        }
        delete command;
    }

}

int
LaneEditor::unregisterMoveHandle(LaneWidthMoveHandle *handle)
{
    return selectedMoveHandles_.remove(handle->getPosDOF(), handle);
}

void LaneEditor::setWidth(double w)
{
    QList<LaneWidth *> endPointSections;
    QList<LaneWidth *> startPointSections;
    foreach (LaneWidthMoveHandle *moveHandle, selectedMoveHandles_)
    {
        LaneWidth *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointSections.append(lowSlot);
        }

        LaneWidth *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointSections.append(highSlot);
        }
    }
    LaneSetWidthCommand *command = new LaneSetWidthCommand(endPointSections, startPointSections, w, false);

    getProjectData()->getUndoStack()->push(command);
}

bool
LaneEditor::translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos)
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
        printStatusBarMsg(tr("Sorry, you can't move yellow items."), 4000);
        qDebug("One DOF not supported yet");
        return false;
    }

    // 2: Two degrees of freedom //
    //
    QList<LaneWidth *> endPointSections;
    QList<LaneWidth *> startPointSections;
    foreach (LaneWidthMoveHandle *moveHandle, selectedMoveHandles_)
    {
        LaneWidth *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointSections.append(lowSlot);
        }

        LaneWidth *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointSections.append(highSlot);
        }
    }

    // Command //
    LaneWidthMovePointsCommand *command = new LaneWidthMovePointsCommand(endPointSections, startPointSections, dPos, NULL);
    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);

        // Message //
        //
        //printStatusBarMsg(QString("Move to: %1, %2").arg(pressPos.x()).arg(pressPos.y()+dPos.y()), 1000);
    }
    else
    {
        if (command->text() != "")
        {
            printStatusBarMsg(command->text(), 4000);
        }
        delete command;
    }

    return true;
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

	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> selectedLaneMoveProps;
	foreach (BaseLaneMoveHandle *baseMoveHandle, selectedLaneMoveHandles_)
	{
		LaneMoveHandle<LaneWidth, LaneWidth> *moveHandle = dynamic_cast<LaneMoveHandle<LaneWidth,LaneWidth>*>(baseMoveHandle);
		LaneMoveProperties *props = new LaneMoveProperties();
		LaneWidth *lowSlot = moveHandle->getLowSlot();
		RSystemElementRoad *road;
		double s;
		if (lowSlot)
		{
			props->lowSlot = lowSlot;
			road = lowSlot->getParentLane()->getParentLaneSection()->getParentRoad();
			s = lowSlot->getSSectionEnd();
		}
		LaneWidth *highSlot = moveHandle->getHighSlot();
		if (highSlot)
		{
			props->highSlot = highSlot;
			if (!lowSlot)
			{
				road = highSlot->getParentLane()->getParentLaneSection()->getParentRoad();
				s = highSlot->getSSectionStartAbs();
			}
		}

		QMap<double, LaneMoveProperties *> propsMap;
		if (selectedLaneMoveProps.find(road) != selectedLaneMoveProps.end())
		{
			propsMap = selectedLaneMoveProps.value(road);
		}

		propsMap.insert(s, props);
		selectedLaneMoveProps.insert(road, propsMap);
	}
	
	if (setWidth)
	{
		TranslateLaneBorderCommand<LaneWidth> *command = new TranslateLaneBorderCommand<LaneWidth>(selectedLaneMoveProps, width, QPointF(0,0), NULL);
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

		TranslateLaneBorderCommand<LaneWidth> *command = new TranslateLaneBorderCommand<LaneWidth>(selectedLaneMoveProps, width, dPos, NULL);
		return getProjectGraph()->executeCommand(command);
	}

}


void 
LaneEditor::removeMoveHandle()
{
	while (!selectedLaneMoveHandles_.empty()) 
	{
		BaseLaneMoveHandle *baseMoveHandle = selectedLaneMoveHandles_.first();
		LaneMoveHandle<LaneWidth, LaneWidth> *moveHandle = dynamic_cast<LaneMoveHandle<LaneWidth, LaneWidth>*>(baseMoveHandle);

		MergeLaneWidthSectionCommand<LaneWidth> *command = new MergeLaneWidthSectionCommand<LaneWidth>(moveHandle->getLowSlot(), moveHandle->getHighSlot(), NULL);
		getTopviewGraph()->executeCommand(command);

		selectedLaneMoveHandles_.removeOne(baseMoveHandle);

	}

}