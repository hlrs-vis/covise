/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#include "crossfalleditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/commands/crossfallsectioncommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"

#include "src/graph/items/roadsystem/crossfall/crossfallroadsystemitem.hpp"
#include "src/graph/items/roadsystem/crossfall/crossfallroadpolynomialitem.hpp"
//#include "src/graph/items/roadsystem/crossfall/crossfallsectionitem.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/crossfall/crossfallmovehandle.hpp"

// Tools //
//
#include "src/gui/tools/crossfalleditortool.hpp"

// Qt //
//
#include <QGraphicsItem>

//################//
// CONSTRUCTORS   //
//################//

CrossfallEditor::CrossfallEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , profileGraph_(profileGraph)
    , roadSystemItemPolyGraph_(NULL)
    , insertSectionHandle_(NULL)
    , smoothRadius_(1000.0)
{
}

CrossfallEditor::~CrossfallEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

SectionHandle *
CrossfallEditor::getInsertSectionHandle()
{
    if (!insertSectionHandle_)
    {
        qDebug("ERROR 1006211555! CrossfallEditor not yet initialized.");
    }
    return insertSectionHandle_;
}

/*! \brief Adds a road to the list of selected roads.
*
* Calls fitInView for the selected road and displays it in the ProfileGraph.
*/
void
CrossfallEditor::addSelectedRoad(RSystemElementRoad *road)
{
    if (!selectedCrossfallRoadItems_.contains(road))
    {
        // Activate Road in ProfileGraph //
        //
        CrossfallRoadPolynomialItem *roadItem = new CrossfallRoadPolynomialItem(roadSystemItemPolyGraph_, road);
        selectedCrossfallRoadItems_.insert(road, roadItem);

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
CrossfallEditor::delSelectedRoad(RSystemElementRoad *road)
{
    CrossfallRoadPolynomialItem *roadItem = selectedCrossfallRoadItems_.take(road);
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
CrossfallEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    //	// Tools //
    //	//
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
    CrossfallEditorToolAction *crossfallEditorToolAction = dynamic_cast<CrossfallEditorToolAction *>(toolAction);
    if (crossfallEditorToolAction)
    {
        // Smooth radius //
        //
        if (crossfallEditorToolAction->getRadius() > 0.0)
        {
            smoothRadius_ = crossfallEditorToolAction->getRadius();
        }
    }
}

//################//
// MoveHandles    //
//################//

void
CrossfallEditor::registerMoveHandle(CrossfallMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! CrossfallEditor CrossfallMoveHandle DOF not in [0,1,2].");
    }
    selectedMoveHandles_.insert(handle->getPosDOF(), handle);
}

int
CrossfallEditor::unregisterMoveHandle(CrossfallMoveHandle *handle)
{
    return selectedMoveHandles_.remove(handle->getPosDOF(), handle);
}

bool
CrossfallEditor::translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos)
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
    QList<CrossfallSection *> endPointSections;
    QList<CrossfallSection *> startPointSections;
    foreach (CrossfallMoveHandle *moveHandle, selectedMoveHandles_)
    {
        CrossfallSection *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointSections.append(lowSlot);
        }

        CrossfallSection *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointSections.append(highSlot);
        }
    }

    // Command //
    //
    CrossfallMovePointsCommand *command = new CrossfallMovePointsCommand(endPointSections, startPointSections, dPos, NULL);
    if (command->isValid())
    {
        if (!endPointSections.isEmpty())
        {
            getProjectData()->getUndoStack()->push(command);
        }
        else
        {
            getProjectData()->getUndoStack()->push(command);
        }

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
CrossfallEditor::init()
{
    // Graph //
    //
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new CrossfallRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
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
CrossfallEditor::kill()
{
    delete roadSystemItem_;
    roadSystemItem_ = NULL;

    delete roadSystemItemPolyGraph_;
    roadSystemItemPolyGraph_ = NULL;
}
