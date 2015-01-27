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

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/commands/elevationsectioncommands.hpp"

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
#include "src/graph/items/roadsystem/elevation/elevationsectionitem.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/elevation/elevationmovehandle.hpp"

// Tools //
//
#include "src/gui/tools/elevationeditortool.hpp"

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
    , smoothRadius_(1000.0)
    , xtrans_(0.0)
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
    if (!selectedElevationRoadItems_.contains(road))
    {
        ElevationRoadPolynomialItem *roadItem = new ElevationRoadPolynomialItem(roadSystemItemPolyGraph_, road);
        selectedElevationRoadItems_.insert(road, roadItem);
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

int
ElevationEditor::delSelectedRoad(RSystemElementRoad *road)
{
    ElevationRoadPolynomialItem *roadItem = selectedElevationRoadItems_.take(road);

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
        if (elevationEditorToolAction->getType() == ElevationEditorToolAction::Radius)
        {
            // Smooth radius //
            //
            if (elevationEditorToolAction->getRadius() > 0.0)
            {
                smoothRadius_ = elevationEditorToolAction->getRadius();
            }
        }
        else if (selectedMoveHandles_.size() > 0 && (elevationEditorToolAction->getType() == ElevationEditorToolAction::Height || elevationEditorToolAction->getType() == ElevationEditorToolAction::IncrementalHeight))
        {

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
            ElevationSetHeightCommand *command = new ElevationSetHeightCommand(endPointSections, startPointSections, elevationEditorToolAction->getHeight(), elevationEditorToolAction->getType() == ElevationEditorToolAction::Height, NULL);
            if (command->isValid())
            {
                getProjectData()->getUndoStack()->push(command);

                // Message //
                //
                printStatusBarMsg(QString("setHeight to: %1").arg(elevationEditorToolAction->getHeight()), 1000);
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
        else if (elevationEditorToolAction->getToolId() == ODD::TEL_SMOOTH)
        {
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

            QList<ElevationSection *> elevationSections;

            foreach (QGraphicsItem *item, selectedItems)
            {
                ElevationSectionItem *maybeSection = dynamic_cast<ElevationSectionItem *>(item);
                if (maybeSection)
                {
                    elevationSections.append(maybeSection->getElevationSection());
                }
            }

            if (elevationSections.size() != 2)
            {
                printStatusBarMsg(tr("Sorry, you have to select 2 roads."), 4000);
                return;
            }

            SmoothElevationRoadsCommand *command = new SmoothElevationRoadsCommand(elevationSections.first(), elevationSections.last(), smoothRadius_);
            getProjectGraph()->executeCommand(command);
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
        printStatusBarMsg(QString("Select Elevation Section "), 100);
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
        printStatusBarMsg(tr("Sorry, you can't move yellow items."), 4000);
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
ElevationEditor::init()
{
    // Graph //
    //
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new ElevationRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
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
        profileGraph_->getScene()->setSceneRect(-1000.0, -1000.0, 20000.0, 2000.0);
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
    delete roadSystemItem_;
    roadSystemItem_ = NULL;

    delete roadSystemItemPolyGraph_;
    roadSystemItemPolyGraph_ = NULL;
}
