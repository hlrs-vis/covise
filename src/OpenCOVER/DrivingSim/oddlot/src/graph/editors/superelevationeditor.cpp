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
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/commands/superelevationsectioncommands.hpp"

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
*
* Calls fitInView for the selected road and displays it in the ProfileGraph.
*/
void
SuperelevationEditor::addSelectedRoad(RSystemElementRoad *road)
{
    if (!selectedSuperelevationRoadItems_.contains(road))
    {
        // Activate Road in ProfileGraph //
        //
        SuperelevationRoadPolynomialItem *roadItem = new SuperelevationRoadPolynomialItem(roadSystemItemPolyGraph_, road);
        selectedSuperelevationRoadItems_.insert(road, roadItem);

        // Fit View //
        //
        QRectF boundingBox = roadItem->boundingRect();
        if (boundingBox.width() < 15.0)
        {
            boundingBox.setWidth(15.0);
        }
        if (boundingBox.height() < 15.0)
        {
            boundingBox.setHeight(15.0);
        }

        profileGraph_->getView()->fitInView(boundingBox);
        profileGraph_->getView()->zoomOut(Qt::Horizontal | Qt::Vertical);
    }
    else
    {
        //qDebug("already there");
    }
}

int
SuperelevationEditor::delSelectedRoad(RSystemElementRoad *road)
{
    SuperelevationRoadPolynomialItem *roadItem = selectedSuperelevationRoadItems_.take(road);
    if (!roadItem)
    {
        return 0;
    }
    else
    {
        // Deactivate Road in ProfileGraph //
        //
        roadItem->registerForDeletion();
        return 1;
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
        // Smooth radius //
        //
        if (superelevationEditorToolAction->getRadius() > 0.0)
        {
            smoothRadius_ = superelevationEditorToolAction->getRadius();
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
        printStatusBarMsg(tr("Sorry, you can't move yellow items."), 4000);
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
        printStatusBarMsg(QString("Move to: %1, %2").arg(pressPos.x()).arg(pressPos.y() + dPos.y()), 1000);
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
// SLOTS          //
//################//

/*!
*
*/
void
SuperelevationEditor::init()
{
    // Graph //
    //
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new SuperelevationRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);
    }

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
    delete roadSystemItem_;
    roadSystemItem_ = NULL;

    delete roadSystemItemPolyGraph_;
    roadSystemItemPolyGraph_ = NULL;
}
