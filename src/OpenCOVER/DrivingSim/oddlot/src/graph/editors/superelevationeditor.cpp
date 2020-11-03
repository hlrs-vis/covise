/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#include "superelevationeditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/commands/superelevationsectioncommands.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"

#include "src/graph/items/roadsystem/superelevation/superelevationroadsystemitem.hpp"
#include "src/graph/items/roadsystem/superelevation/superelevationroadpolynomialitem.hpp"
//#include "src/graph/items/roadsystem/superelevation/superelevationsectionitem.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/superelevation/superelevationmovehandle.hpp"

// Tools //
//
#include "src/gui/tools/superelevationeditortool.hpp"

// Qt //
//
#include <QGraphicsItem>

//################//
// CONSTRUCTORS   //
//################//

SuperelevationEditor::SuperelevationEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , profileGraph_(profileGraph)
    , roadSystemItemPolyGraph_(NULL)
    , insertSectionHandle_(NULL)
    , smoothRadius_(1000.0)
    , selectedSuperelevationItem_(NULL)
{
}

SuperelevationEditor::~SuperelevationEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

SectionHandle *
SuperelevationEditor::getInsertSectionHandle()
{
    if (!insertSectionHandle_)
    {
        qDebug("ERROR 1006211555! SuperelevationEditor not yet initialized.");
    }
    return insertSectionHandle_;
}

/*! \brief Adds a road to the list of selected roads.
*/

void
SuperelevationEditor::insertSelectedRoad(RSystemElementRoad* road)
{
    if (getCurrentTool() == ODD::TSE_SELECT)
    {
        if (!selectedRoads_.contains(road))
        {
            selectedRoads_.append(road);
            QList<DataElement*> sectionList;
            foreach(SuperelevationSection * section, road->getSuperelevationSections())
            {
                if (!section->isElementSelected())
                {
                    sectionList.append(section);
                }
            }

            int listSize = selectedRoads_.size();
            if (listSize == 1)
            {
                if (!selectedSuperelevationItem_)
                {
                    selectedSuperelevationItem_ = new SuperelevationRoadPolynomialItem(roadSystemItemPolyGraph_, road);
                }
            }
            else if (listSize == 2)
            {
                selectedSuperelevationItem_->registerForDeletion();
                selectedSuperelevationItem_ = NULL;
            }

            SelectDataElementCommand* command = new SelectDataElementCommand(sectionList);
            getProjectGraph()->executeCommand(command);
        }
    }
}

/*! \brief Initialize BoundingBox and offset.
*/

void
SuperelevationEditor::initBox()
{
    xtrans_ = 0.0;
    boundingBox_.setRect(0.0, 0.0, 0.0, 0.0);
}

/*
* Calls fitInView for the selected road and displays it in the ProfileGraph.
*/
void
SuperelevationEditor::addSelectedRoad(SuperelevationRoadPolynomialItem* roadItem)
{

    // Compute the BoundingBox of all selected roads //
    //
    QRectF roadItemBoundingBox = roadItem->translate(xtrans_, 0.0);

    boundingBox_ = boundingBox_.united(roadItemBoundingBox);
    xtrans_ = boundingBox_.width();
}

void
SuperelevationEditor::fitView()
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
SuperelevationEditor::delSelectedRoad(RSystemElementRoad *road)
{
    if (selectedRoads_.contains(road))
    {
        selectedRoads_.removeAll(road);

        QList<DataElement*> sectionList;
        foreach(SuperelevationSection * section, road->getSuperelevationSections())
        {
            if (section->isElementSelected())
            {
                sectionList.append(section);
            }
        }

        int listSize = selectedRoads_.size();
        if (listSize == 1)
        {
            if (!selectedSuperelevationItem_)
            {
                selectedSuperelevationItem_ = new SuperelevationRoadPolynomialItem(roadSystemItemPolyGraph_, selectedRoads_.first());

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
                selectedSuperelevationItem_->registerForDeletion();
                selectedSuperelevationItem_ = NULL;
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
SuperelevationEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    //	// Tools //
    //	//
    //	if(getCurrentTool() == ODD::TSE_SELECT)
    //	{
    //		// does nothing //
    //	}
    //	else if(getCurrentTool() == ODD::TSE_ADD)
    //	{
    //		// does nothing //
    //	}
    //	else if(getCurrentTool() == ODD::TSE_DEL)
    //	{
    //		// does nothing //
    //		// Note the problem: The ToolAction is re-sent, after a warning message has been clicked away. (Due to re-send on getting the focus back?)
    //	}

    // Tools //
    //
    SuperelevationEditorToolAction *superelevationEditorToolAction = dynamic_cast<SuperelevationEditorToolAction *>(toolAction);

	if (superelevationEditorToolAction)
	{
		if (superelevationEditorToolAction->getToolId() == ODD::TSE_SELECT)
		{
			if (superelevationEditorToolAction->getParamToolId() == ODD::TSE_RADIUS)
			{
				if (superelevationEditorToolAction->getRadius() > 0.0)
				{
					smoothRadius_ = superelevationEditorToolAction->getRadius();
				}
			}
		}
		else if ((superelevationEditorToolAction->getToolId() == ODD::TSE_ADD) || (superelevationEditorToolAction->getToolId() == ODD::TSE_DEL))
		{
			getTopviewGraph()->getScene()->deselectAll();
		}
	}
}

//################//
// MoveHandles    //
//################//

void
SuperelevationEditor::registerMoveHandle(SuperelevationMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! SuperelevationEditor SuperelevationMoveHandle DOF not in [0,1,2].");
    }
    selectedMoveHandles_.insert(handle->getPosDOF(), handle);
}

int
SuperelevationEditor::unregisterMoveHandle(SuperelevationMoveHandle *handle)
{
    return selectedMoveHandles_.remove(handle->getPosDOF(), handle);
}

bool
SuperelevationEditor::translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos)
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
    QList<SuperelevationSection *> endPointSections;
    QList<SuperelevationSection *> startPointSections;
    foreach (SuperelevationMoveHandle *moveHandle, selectedMoveHandles_)
    {
        SuperelevationSection *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointSections.append(lowSlot);
        }

        SuperelevationSection *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointSections.append(highSlot);
        }
    }

    // Command //
    //
    SuperelevationMovePointsCommand *command = new SuperelevationMovePointsCommand(endPointSections, startPointSections, dPos, NULL);
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

//################//
// SLOTS          //
//################//

/*!
*
*/
void
SuperelevationEditor::init()
{
	// ProfileGraph //
//
	if (!roadSystemItemPolyGraph_)
	{
		// Root item //
		//
		roadSystemItemPolyGraph_ = new RoadSystemItem(profileGraph_, getProjectData()->getRoadSystem());
		profileGraph_->getScene()->addItem(roadSystemItemPolyGraph_);
		profileGraph_->getScene()->setSceneRect(-1000.0, -45.0, 20000.0, 90.0);
	}

    // Graph //
    //
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new SuperelevationRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
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
SuperelevationEditor::kill()
{
    selectedRoads_.clear();
    if (selectedSuperelevationItem_ && !selectedSuperelevationItem_->isInGarbage())
    {
        selectedSuperelevationItem_->registerForDeletion();
        selectedSuperelevationItem_ = NULL;
    }

    delete roadSystemItem_;
    roadSystemItem_ = NULL;

    delete roadSystemItemPolyGraph_;
    roadSystemItemPolyGraph_ = NULL;
}
