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
    , selectedAdjacentElevationItem_(NULL)
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
            QList<DataElement *> sectionList;
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

            SelectDataElementCommand *command = new SelectDataElementCommand(sectionList);
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
    ODD::ToolId currentTool = getCurrentTool();

    if (currentTool == ODD::TEL_SELECT)
    {
        if (selectedRoads_.contains(road))
        {
            selectedRoads_.removeAll(road);

            QList<DataElement *> sectionList;
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

                    DeselectDataElementCommand *command = new DeselectDataElementCommand(sectionList);
                    getProjectGraph()->executeCommand(command);
                }
            }
            else
            {
                DeselectDataElementCommand *command = new DeselectDataElementCommand(sectionList);
                getProjectGraph()->executeCommand(command);

                if (listSize == 0)
                {
                    selectedElevationItem_->registerForDeletion();
                    selectedElevationItem_ = NULL;
                }
            }
        }
    }
    else if ((currentTool == ODD::TEL_SLOPE) || (currentTool == ODD::TEL_SMOOTH) || (currentTool == ODD::TEL_SMOOTH_SECTION))
    {
        if (selectedElevationItem_ && (selectedElevationItem_->getRoad() == road))
        {
            if (!selectedElevationItem_->getRoad()->isChildElementSelected())
            {
                selectedElevationItem_->registerForDeletion();
                selectedElevationItem_ = NULL;
            }
        }
        else if (selectedAdjacentElevationItem_ && (selectedAdjacentElevationItem_->getRoad() == road))
        {
            if (!selectedAdjacentElevationItem_->getRoad()->isChildElementSelected())
            {
                selectedAdjacentElevationItem_->registerForDeletion();
                selectedAdjacentElevationItem_ = NULL;
            }
        }
    }
}

void
ElevationEditor::delSelectedRoads()
{
    QList<DataElement *> sectionList;
    foreach(RSystemElementRoad * road, selectedRoads_)
    {
        foreach(ElevationSection * section, road->getElevationSections())
        {
            if (section->isElementSelected())
            {
                sectionList.append(section);
            }
        }
    }

    selectedRoads_.clear();
    if (selectedElevationItem_ && !selectedElevationItem_->isInGarbage())
    {
        selectedElevationItem_->registerForDeletion();
        selectedElevationItem_ = NULL;
    }

    if (selectedAdjacentElevationItem_ && !selectedAdjacentElevationItem_->isInGarbage())
    {
        selectedAdjacentElevationItem_->registerForDeletion();
        selectedAdjacentElevationItem_ = NULL;
    }

    DeselectDataElementCommand *command = new DeselectDataElementCommand(sectionList);
    getProjectGraph()->executeCommand(command);
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
    ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(toolAction);

    if (elevationEditorToolAction)
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
                foreach(ElevationMoveHandle * moveHandle, selectedMoveHandles_)
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

            foreach(ElevationMoveHandle * moveHandle, selectedMoveHandles_)
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
        else if (tool_ && !tool_->containsToolId(toolAction->getToolId()))
        {
            clearToolObjectSelection();
            delToolParameters();
            ODD::mainWindow()->showParameterDialog(false);
        }

        if ((elevationEditorToolAction->getToolId() == ODD::TEL_ADD) || (elevationEditorToolAction->getToolId() == ODD::TEL_DEL))
        {
            delSelectedRoads();
        }
        else if (elevationEditorToolAction->getToolId() == ODD::TEL_SMOOTH)
        {
            ODD::ToolId paramTool = getCurrentParameterTool();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                QMap<QGraphicsItem *, ElevationSection *>items = getSelectedElevationSections(2);
                ToolValue<ElevationSection> *elevationSectionParam;

                if (!items.isEmpty())
                {
                    elevationSectionItem_ = items.firstKey();
                    ElevationSection *elevationSection = items.take(elevationSectionItem_);
                    QString textDisplayed = QString("%1 Section at %2").arg(elevationSection->getParentRoad()->getIdName()).arg(elevationSection->getSStart());
                    elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection", true, "", textDisplayed, elevationSection);
                }
                else
                {
                    elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection", true);
                }
                tool_ = new Tool(ODD::TEL_SMOOTH, 1);
                tool_->readParams(elevationSectionParam);
                ToolValue<ElevationSection> *adjacentSectionParam;
                if (!items.isEmpty())
                {
                    ElevationSection *elevationSection = items.first();
                    QString textDisplayed = QString("%1 Section at %2").arg(elevationSection->getParentRoad()->getIdName()).arg(elevationSection->getSStart());
                    adjacentSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Adjacent Section", true, "", textDisplayed, elevationSection);
                    elevationSectionAdjacentItem_ = items.firstKey();
                }
                else
                {
                    adjacentSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Adjacent Section");
                }
                tool_->readParams(adjacentSectionParam);

                createToolParameterSettingsApplyBox(tool_, ODD::EEL);
                ODD::mainWindow()->showParameterDialog(true, "Smooth Heights across Roads", "Specify smoothing radius, select adjacent elevation sections on two roads and press APPLY");

                // verify if apply can be displayed //
                if (tool_->verify())
                {
                    settingsApplyBox_->setApplyButtonVisible(true);
                }
            }
        }
        else if (elevationEditorToolAction->getToolId() == ODD::TEL_SLOPE)
        {
            ODD::ToolId paramTool = getCurrentParameterTool();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                ToolValue<double> *slopeParam = new ToolValue<double>(ODD::TEL_PERCENTAGE, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::DOUBLE, "Slope Percentage");
                slopeParam->setValue(slope_);
                tool_ = new Tool(ODD::TEL_SLOPE, 1);
                tool_->readParams(slopeParam);

                QMap<QGraphicsItem *, ElevationSection *>items = getSelectedElevationSections(1);
                ToolValue<ElevationSection> *elevationSectionParam;

                if (!items.isEmpty())
                {
                    elevationSectionItem_ = items.firstKey();
                    ElevationSection *elevationSection = items.take(elevationSectionItem_);
                    QString textDisplayed = QString("%1 Section at %2").arg(elevationSection->getParentRoad()->getIdName()).arg(elevationSection->getSStart());
                    elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SLOPE, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection", true, "", textDisplayed, elevationSection);
                }
                else
                {
                    elevationSectionParam = new ToolValue<ElevationSection>(ODD::TEL_SLOPE, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select ElevationSection", true);
                }
                tool_->readParams(elevationSectionParam);

                createToolParameterSettingsApplyBox(tool_, ODD::EEL);
                ODD::mainWindow()->showParameterDialog(true, "Change Slope of Elevation Section", "Specify slope percentage, select elevation section and press APPLY");

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
        ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
        if (action)
        {
            if (action->getToolId() == ODD::TEL_PERCENTAGE)
            {
                slope_ = action->getValue();
            }
        }
    }
}

QMap<QGraphicsItem *, ElevationSection *>
ElevationEditor::getSelectedElevationSections(int count)
{
    QMap<QGraphicsItem *, ElevationSection *> selected;
    QList<RSystemElementRoad *> selectedRoads;
    QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
    for (int i = 0; i < selectedItems.size(); i++)
    {
        QGraphicsItem *item = selectedItems.at(i);

        if (selected.size() < count)
        {
            ElevationSectionItem *elevationSectionItem = dynamic_cast<ElevationSectionItem *>(item);
            if (elevationSectionItem)
            {
                ElevationSection *elevationSection = elevationSectionItem->getElevationSection();
                if (!selected.key(elevationSection, NULL))
                {
                    RSystemElementRoad *road = elevationSection->getParentRoad();
                    if (!selectedRoads.contains(road))
                    {
                        selected.insert(elevationSectionItem, elevationSection);
                        selectedRoads.append(road);
                        continue;
                    }
                }
            }
        }
        item->setSelected(false);
    }

    return selected;
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
    foreach(ElevationMoveHandle * moveHandle, selectedMoveHandles_)
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
    foreach(ElevationMoveHandle * moveHandle, selectedMoveHandles_)
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
    foreach(QGraphicsItem * item, selectedItems)
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
    foreach(QGraphicsItem * item, selectedItems)
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

    /* foreach(ElevationSection *section, endPointSections)
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
    foreach(ElevationSection * section, startPointSections)
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

//###################//
// ELEVATION SECTIONS //
//###################//
void
ElevationEditor::arrangeAdjacentRoads(ElevationSection *section, RSystemElementRoad *road, ElevationSection *adjacentSection, RSystemElementRoad *adjacentRoad)
{
    {
        int i = -1;
        if (section == road->getElevationSection(0.0))
        {
            if (adjacentSection == adjacentRoad->getElevationSection(0.0))
            {
                i = 0;
            }
            else if (adjacentSection == adjacentRoad->getElevationSection(adjacentRoad->getLength()))
            {
                i = 3;
            }
        }
        else if (section == road->getElevationSection(road->getLength()))
        {
            if (adjacentSection == adjacentRoad->getElevationSection(0.0))
            {
                i = 1;
            }
            else if (adjacentSection == adjacentRoad->getElevationSection(adjacentRoad->getLength()))
            {
                i = 2;
            }
        }

        if (i < 0)
        {
            qreal dist[4];
            dist[0] = (road->getGlobalPoint(0.0) - adjacentRoad->getGlobalPoint(0.0)).manhattanLength();
            dist[1] = (road->getGlobalPoint(road->getLength()) - adjacentRoad->getGlobalPoint(0.0)).manhattanLength();
            dist[2] = (road->getGlobalPoint(road->getLength()) - adjacentRoad->getGlobalPoint(adjacentRoad->getLength())).manhattanLength();
            dist[3] = (road->getGlobalPoint(0.0) - adjacentRoad->getGlobalPoint(adjacentRoad->getLength())).manhattanLength();


            int j = 1;
            while (j < 4)
            {
                if (dist[i] > dist[j])
                {
                    i = j;
                }
                j++;
            }
        }

        switch (i)
        {
        case 0:
        {
            qreal width = selectedAdjacentElevationItem_->boundingRect().width();
            QTransform mirror(-1.0, 0, 0, 1.0, width + selectedElevationItem_->boundingRect().width(), 0.0);
            selectedElevationItem_->setPos(width, 0.0);
            selectedAdjacentElevationItem_->setTransform(mirror);
        }
        break;
        case 1:
            break;
        case 2:
        {
            qreal width = selectedElevationItem_->boundingRect().width();
            QTransform mirror(-1.0, 0, 0, 1.0, width + selectedAdjacentElevationItem_->boundingRect().width(), 0.0);
            selectedElevationItem_->setTransform(mirror);
            selectedAdjacentElevationItem_->setPos(-width, 0.0);
        }
        break;
        case 3:
            selectedElevationItem_->setPos(selectedAdjacentElevationItem_->boundingRect().width(), 0.0);
            selectedAdjacentElevationItem_->setPos(-selectedElevationItem_->boundingRect().width(), 0.0);
            break;
        }
    }
}

bool
ElevationEditor::registerElevationSection(QGraphicsItem *item, ElevationSection *section)
{
    ODD::ToolId currentTool = getCurrentTool();

    if ((currentTool == ODD::TEL_SMOOTH) || (currentTool == ODD::TEL_SLOPE))
    {
        RSystemElementRoad *adjacentRoad = NULL;
        ElevationSection *adjacentSection;
        if (currentTool == ODD::TEL_SMOOTH)
        {
            adjacentSection = dynamic_cast<ToolValue<ElevationSection> *>(tool_->getParam(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT))->getValue();
            if (adjacentSection)
            {
                adjacentRoad = adjacentSection->getParentRoad();
            }
        }

        if ((item != elevationSectionItem_) && (item != elevationSectionAdjacentItem_))
        {
            RSystemElementRoad *road = section->getParentRoad();

            if (road != adjacentRoad)
            {
                if (elevationSectionItem_)
                {
                    elevationSectionItem_->setSelected(false);
                }

                QString textDisplayed = QString("%1 Section at %2").arg(road->getIdName()).arg(section->getSStart());
                setToolValue<ElevationSection>(section, textDisplayed);

                selectedElevationItem_ = new ElevationRoadPolynomialItem(roadSystemItemPolyGraph_, road);
                if (adjacentRoad)
                {
                    arrangeAdjacentRoads(section, road, adjacentSection, adjacentRoad);
                }

                elevationSectionItem_ = item;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        RSystemElementRoad *adjacentRoad = NULL;
        ElevationSection *adjacentSection = dynamic_cast<ToolValue<ElevationSection> *>(tool_->getParam(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT))->getValue();
        if (adjacentSection)
        {
            adjacentRoad = adjacentSection->getParentRoad();
        }

        if ((item != elevationSectionItem_) && (item != elevationSectionAdjacentItem_))
        {
            RSystemElementRoad *road = section->getParentRoad();
            if (road != adjacentRoad)
            {
                if (elevationSectionAdjacentItem_)
                {
                    elevationSectionAdjacentItem_->setSelected(false);
                }

                QString textDisplayed = QString("%1 Section at %2").arg(road->getIdName()).arg(section->getSStart());
                setToolValue<ElevationSection>(section, textDisplayed);

                selectedAdjacentElevationItem_ = new ElevationRoadPolynomialItem(roadSystemItemPolyGraph_, road);
                if (adjacentRoad)
                {
                    arrangeAdjacentRoads(adjacentSection, adjacentRoad, section, road);
                }
                elevationSectionAdjacentItem_ = item;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    
    // verify if apply can be displayed //
    if (tool_->verify())
    {
        settingsApplyBox_->setApplyButtonVisible(true);
    }

    return true;
}

bool
ElevationEditor::deregisterElevationSection(QGraphicsItem *item)
{
    ToolParameter *param = NULL;
    if (item == elevationSectionItem_)
    {
        if (tool_->containsToolId(ODD::TEL_SMOOTH))
        {
            param = tool_->getParam(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT);
        }
        else
        {
            param = tool_->getParam(ODD::TEL_SLOPE, ODD::TPARAM_SELECT);
        }
        elevationSectionItem_ = NULL;
    }
    else if (item == elevationSectionAdjacentItem_)
    {
        param = tool_->getParam(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT);
        elevationSectionAdjacentItem_ = NULL;
    }

    if (param)
    {
        delToolValue(param);
    }
    settingsApplyBox_->setApplyButtonVisible(false);

  /*  DeselectDataElementCommand *command = new DeselectDataElementCommand(road);
    if (command->isValid())
    {
        getTopviewGraph()->executeCommand(command);
    } */

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
    if (tool_)
    {
        reset();
        ODD::mainWindow()->showParameterDialog(false);
    }
    else
    {
        delSelectedRoads();
    }

    delete roadSystemItem_;
    roadSystemItem_ = NULL;

    delete roadSystemItemPolyGraph_;
    roadSystemItemPolyGraph_ = NULL;
}

void
ElevationEditor::apply()
{

    ODD::ToolId toolId = tool_->getToolId();
    if (toolId == ODD::TEL_SMOOTH)
    {
        ToolValue<ElevationSection> *firstValue = dynamic_cast<ToolValue<ElevationSection>*>(tool_->getParam(ODD::TEL_SMOOTH, ODD::TPARAM_SELECT));
        ElevationSection *firstSection = firstValue->getValue();
        ToolValue<ElevationSection> *secondValue = dynamic_cast<ToolValue<ElevationSection>*>(tool_->getParam(ODD::TEL_SMOOTH_SECTION, ODD::TPARAM_SELECT));
        ElevationSection *secondSection = secondValue->getValue();

        SmoothElevationRoadsCommand *command = new SmoothElevationRoadsCommand(firstSection, secondSection, smoothRadius_);
        if (command->isValid())
        {
            clearToolObjectSelection();
            getProjectGraph()->executeCommand(command);

            // reset the values and parameter settings because the selected elements do not exist anymore
            //
            firstValue->setActive(true);
            QList<ToolParameter *> list({ firstValue, secondValue });
            resetToolValues(list);
        }
    }
    else if (toolId == ODD::TEL_SLOPE)
    {
        QList<ElevationSection *> endPointSections;
        QList<ElevationSection *> startPointSections;

        ToolValue<ElevationSection> *toolValue = dynamic_cast<ToolValue<ElevationSection>*>(tool_->getParam(ODD::TEL_SLOPE, ODD::TPARAM_SELECT));
        ElevationSection *section = toolValue->getValue();
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

                    // set the values and parameter settings
                    //
                    toolValue->setActive(true);
                    QList<ToolParameter *>list({ toolValue });
                    setToolValues(list);
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
    delSelectedRoads();

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
    clearToolObjectSelection();
    delToolParameters();
}

void
ElevationEditor::reject()
{
    ProjectEditor::reject();

    if (tool_)
    {
        clearToolObjectSelection();
        delToolParameters();
        ODD::mainWindow()->showParameterDialog(false);
    }
}
