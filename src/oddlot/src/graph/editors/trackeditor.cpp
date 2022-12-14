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

#include "trackeditor.hpp"

#include "src/mainwindow.hpp"

 // Project //
 //
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/trackcommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
#include "src/data/commands/tilecommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/track/trackroadsystemitem.hpp"
#include "src/graph/items/roadsystem/track/trackmovehandle.hpp"
#include "src/graph/items/roadsystem/track/trackaddhandle.hpp"
#include "src/graph/items/roadsystem/track/trackrotatehandle.hpp"
#include "src/graph/items/roadsystem/track/trackelementitem.hpp"
#include "src/graph/items/roadsystem/track/trackroaditem.hpp"

#include "src/graph/items/roadsystem/track/roadmovehandle.hpp"
#include "src/graph/items/roadsystem/track/roadrotatehandle.hpp"

#include "src/graph/items/handles/circularrotatehandle.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadmark/roadmarkitem.hpp"

// GUI //
//
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

// Tools //
//
#include "src/gui/tools/trackeditortool.hpp"
#include "src/gui/mouseaction.hpp"

// Visitor //
//
#include "src/graph/visitors/roadmarkvisitor.hpp"
#include "src/data/visitors/trackmovevalidator.hpp"

// Qt //
//
#include <QGraphicsLineItem>
#include <QVector2D>
#include <QGraphicsSceneMouseEvent>

// Utils //
//
#include "math.h"
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

/*! \todo Delete visitor
*/
TrackEditor::TrackEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , trackRoadSystemItem_(NULL)
    , pressPoint_(0.0, 0.0)
    , newRoadLineItem_(NULL)
    , newRoadPolyItem_(NULL)
    , newPolyRoad_(NULL)
    , addRoadSystemHandle_(NULL)
    , currentRoadPrototype_(NULL)
    , currentRoadSystemPrototype_(NULL)
    , state_(TrackEditor::STE_NONE)
    , sectionHandle_(NULL)
    , mergeItem_(NULL)
    , appendItem_(NULL)
{
    prototypeManager_ = ODD::mainWindow()->getPrototypeManager();
    geometryPrimitiveType_ = GeometryPrimitive::LINE;
}

TrackEditor::~TrackEditor()
{
    kill();
}

SectionHandle *
TrackEditor::getSectionHandle() const
{
    if (!sectionHandle_)
    {
        qDebug("ERROR 1010141039! TrackEditor not yet initialized.");
    }
    return sectionHandle_;
}

//################//
// TOOL           //
//################//

ToolParameter *
TrackEditor::addToolParameter(const PrototypeManager::PrototypeType &type, const ODD::ToolId &toolId, const QString &labelText)
{
    QStringList text;
    QList<PrototypeContainer<RSystemElementRoad *> *> prototypes = prototypeManager_->getRoadPrototypes(type);
    int i = 0;
    for (; i < prototypes.size(); i++)
    {
        text.append(prototypes.at(i)->getPrototypeName());
    }

    ToolValue<int> *param = new ToolValue<int>(getCurrentTool(), toolId, 0, ToolParameter::ParameterTypes::ENUM, text.join(","), false, labelText);
    if (!currentPrototypes_.value(type))
    {
        currentPrototypes_.insert(type, prototypeManager_->getRoadPrototypes(type).at(0));
        param->setValue(0);
    }
    else
    {
        param->setValue(text.indexOf(currentPrototypes_.value(type)->getPrototypeName()));
    }

    return param;
}

void
TrackEditor::appendToolParameter(const PrototypeManager::PrototypeType &type, const ODD::ToolId &toolId, const ODD::ToolId &paramToolId, RSystemElementRoad *road)
{
    QString text = road->getIdName();
    QList<PrototypeContainer<RSystemElementRoad *> *> prototypes = prototypeManager_->getRoadPrototypes(type);
    int i = 0;
    for (; i < prototypes.size(); i++)
    {
        if (prototypes.at(i)->getPrototypeName() == text)
        {
            settings_->setComboBoxIndex(tool_->getParam(toolId, paramToolId), text);
            break;
        }
    }
    if (i == prototypes.size())
    {
        QString empty = "";
        prototypeManager_->addRoadPrototype(text, QIcon(empty), road->getClone(), type, empty, empty, empty);
        settings_->addComboBoxEntry(tool_->getParam(toolId, paramToolId), prototypeManager_->getIndexOfLastInsertedPrototype(type), text);
    }
}

/*! \brief Handles the ToolActions.
*
*/
void
TrackEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ODD::ToolId lastTool = getCurrentTool();

    if (tool_ && !tool_->containsToolId(toolAction->getToolId()))
    {
        clearToolObjectSelection();
        delToolParameters();
        ODD::mainWindow()->showParameterDialog(false);
    }
    ProjectEditor::toolAction(toolAction);

    TrackEditorToolAction *trackEditorToolAction = dynamic_cast<TrackEditorToolAction *>(toolAction);
    if (trackEditorToolAction)
    {
        ODD::ToolId toolId = trackEditorToolAction->getToolId();
        if ((toolId == ODD::TTE_ROAD_NEW) || (toolId == ODD::TTE_ADD)
            || (toolId == ODD::TTE_ADD_PROTO) || (toolId == ODD::TTE_ROAD_CIRCLE)
            || (toolId == ODD::TTE_ROADSYSTEM_ADD))
        {
            ODD::ToolId paramTool = trackEditorToolAction->getParamToolId();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                getTopviewGraph()->getScene()->deselectAll();

                tool_ = new Tool(getCurrentTool(), 1);
                if ((toolId == ODD::TTE_ROAD_NEW) || (toolId == ODD::TTE_ADD))
                {
                    ToolValue<int> *paramValue = new ToolValue<int>(getCurrentTool(), getCurrentTool(), 0, ToolParameter::ParameterTypes::ENUM, "LINE, ARC/SPIRAL CURVE, POLYNOMIAL", false, "Geometry Primitive Type");
                    switch (geometryPrimitiveType_)
                    {
                    case GeometryPrimitive::LINE:
                        paramValue->setValue(0);
                        break;
                    case GeometryPrimitive::ARC_SPIRAL:
                        paramValue->setValue(1);
                        break;
                    case GeometryPrimitive::POLYNOMIAL:
                        paramValue->setValue(2);
                    }

                    tool_->readParams(paramValue);
                }

                if (toolId == ODD::TTE_ROADSYSTEM_ADD)
                {
                    QStringList text;
                    int i = 0;
                    QList<PrototypeContainer<RoadSystem *> *> roadSystemContainer = prototypeManager_->getRoadSystemPrototypes();
                    for (; i < roadSystemContainer.size(); i++)
                    {
                        text.append(roadSystemContainer.at(i)->getPrototypeName());
                    }

                    ToolValue<int> *paramValue = new ToolValue<int>(ODD::TTE_ROADSYSTEM_ADD, ODD::TTE_PROTO_ROADSYSTEM, 0, ToolParameter::ParameterTypes::ENUM, text.join(","), false, "RoadSystem Prototype");
                    if (!currentRoadSystemPrototype_)
                    {
                        currentRoadSystemPrototype_ = prototypeManager_->getRoadSystemPrototypes().at(0);
                        paramValue->setValue(0);
                    }
                    else
                    {
                        paramValue->setValue(text.indexOf(currentRoadSystemPrototype_->getPrototypeName()));
                    }
                    tool_->readParams(paramValue);
                }
                else if (toolId == ODD::TTE_ADD_PROTO)
                {
                    ToolParameter *param = addToolParameter(PrototypeManager::PTP_TrackPrototype, ODD::TTE_PROTO_TRACK, "Track Prototype");
                    tool_->readParams(param);
                }

                ToolParameter *param = addToolParameter(PrototypeManager::PTP_LaneSectionPrototype, ODD::TTE_PROTO_LANE, "LaneSection Prototype");
                tool_->readParams(param);

                param = addToolParameter(PrototypeManager::PTP_RoadTypePrototype, ODD::TTE_PROTO_TYPE, "RoadType Prototype");
                tool_->readParams(param);

                param = addToolParameter(PrototypeManager::PTP_ElevationPrototype, ODD::TTE_PROTO_ELEVATION, "Elevation Prototype");
                tool_->readParams(param);

                param = addToolParameter(PrototypeManager::PTP_SuperelevationPrototype, ODD::TTE_PROTO_SUPERELEVATION, "Superelevation Prototype");
                tool_->readParams(param);

                param = addToolParameter(PrototypeManager::PTP_CrossfallPrototype, ODD::TTE_PROTO_CROSSFALL, "Crossfall Prototype");
                tool_->readParams(param);

                param = addToolParameter(PrototypeManager::PTP_RoadShapePrototype, ODD::TTE_PROTO_ROADSHAPE, "RoadShape Prototype");
                tool_->readParams(param);

                ToolValue<RSystemElementRoad> *roadParam = new ToolValue<RSystemElementRoad>(ODD::TTE_PROTO_FETCH, toolId, 0, ToolParameter::ParameterTypes::OBJECT, "Fetch Prototypes from Road");
                tool_->readParams(roadParam);

                createToolParameterSettingsApplyBox(tool_, ODD::ETE);
                //   createToolParameterSettings(tool_, ODD::ETE);

                if (isCurrentTool(ODD::TTE_ROAD_NEW))
                {
                    ODD::mainWindow()->showParameterDialog(true, "New Road", "Choose a geometry primitive and the prototypes and draw a line.");
                }
                else if (isCurrentTool(ODD::TTE_ROAD_CIRCLE))
                {
                    ODD::mainWindow()->showParameterDialog(true, "Circular Road", "Choose the prototypes and draw.");
                }
                else if (isCurrentTool(ODD::TTE_ROADSYSTEM_ADD))
                {
                    ODD::mainWindow()->showParameterDialog(true, "Draw Roadsystem", "Choose a prototype and draw.");
                }
                else
                {
                    ODD::mainWindow()->showParameterDialog(true, "Add Track (ESC to deselect)", "Choose a geometry primitive and the prototypes and append the new track.");
                }

                // Initialize prototypes //
                //
                delete currentRoadPrototype_;

                currentRoadPrototype_ = new RSystemElementRoad("prototype");

                // Superpose user prototypes //
                //
                foreach(PrototypeContainer<RSystemElementRoad *> *container, currentPrototypes_)
                {
                    currentRoadPrototype_->superposePrototype(container->getPrototype());
                }

            }
        }
        else if ((toolId == ODD::TTE_ROAD_MERGE) || (toolId == ODD::TTE_ROAD_SNAP))
        {
            ODD::ToolId paramTool = trackEditorToolAction->getParamToolId();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                QMap<QGraphicsItem *, RSystemElementRoad *> selected = getSelectedRoads(2);
                ToolValue<RSystemElementRoad> *param;
                if (!selected.isEmpty())
                {
                    mergeItem_ = selected.firstKey();
                    RSystemElementRoad *road = selected.take(mergeItem_);
                    param = new ToolValue<RSystemElementRoad>(toolId, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Road", false, "", road->getIdName(), road);
                    
                    SelectDataElementCommand *command = new SelectDataElementCommand(road);
                    if (command->isValid())
                    {
                        getTopviewGraph()->executeCommand(command);
                    }
                }
                else
                {
                    param = new ToolValue<RSystemElementRoad>(toolId, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Road", true);
                }
                tool_ = new Tool(toolId, 4);
                tool_->readParams(param);

                ToolValue<RSystemElementRoad> *roadParam;
                if (!selected.isEmpty())
                {
                    appendItem_ = selected.firstKey();
                    RSystemElementRoad *road = selected.take(appendItem_);
                    roadParam = new ToolValue<RSystemElementRoad>(ODD::TTE_ROAD_APPEND, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Road to append", false, "", road->getIdName(), road);

                    SelectDataElementCommand *command = new SelectDataElementCommand(road);
                    if (command->isValid())
                    {
                        getTopviewGraph()->executeCommand(command);
                    }
                }
                else
                {
                    roadParam = new ToolValue<RSystemElementRoad>(ODD::TTE_ROAD_APPEND, ODD::TPARAM_SELECT, 0, ToolParameter::ParameterTypes::OBJECT, "Select Road to append");
                    if (param->isValid())
                    {
                        roadParam->setActive(true);
                    }
                }
                tool_->readParams(roadParam);

                createToolParameterSettingsApplyBox(tool_, ODD::ETE);

                if (toolId == ODD::TTE_ROAD_MERGE)
                {
                    ODD::mainWindow()->showParameterDialog(true, "Append Second Road at End of First", "SELECT the two roads and press APPLY");
                }
                else
                {
                    ODD::mainWindow()->showParameterDialog(true, "Extend one road to fill the gap between two roads", "SELECT the two roads and press APPLY");
                }

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
            ODD::ToolId toolId = action->getToolId();
            ODD::ToolId paramToolId = action->getParamToolId();

            if ((toolId == ODD::TTE_ROAD_NEW) || (toolId == ODD::TTE_ADD)
                || (toolId == ODD::TTE_ADD_PROTO) || (toolId == ODD::TTE_ROAD_CIRCLE)
                || (toolId == ODD::TTE_ROADSYSTEM_ADD))
            {
                int index = action->getParamId();
                if (toolId == paramToolId)
                {
                    if (index == 0)
                    {
                        geometryPrimitiveType_ = GeometryPrimitive::LINE;
                    }
                    else if (index == 1)
                    {
                        geometryPrimitiveType_ = GeometryPrimitive::ARC_SPIRAL;
                    }
                    else
                    {
                        geometryPrimitiveType_ = GeometryPrimitive::POLYNOMIAL;
                    }

                    ToolParameter* p = tool_->getLastParam(settings_->getCurrentParameterID());
                    ToolValue<int>* v = dynamic_cast<ToolValue<int> *>(p);
                    v->setValue(index);
                }
                else if (paramToolId == ODD::TTE_PROTO_ROADSYSTEM)
                {
                    currentRoadSystemPrototype_ = prototypeManager_->getRoadSystemPrototypes().at(index);
                }
                else
                {
                    PrototypeContainer<RSystemElementRoad*>* container;
                    if (paramToolId == ODD::TTE_PROTO_TRACK)
                    {
                        currentRoadPrototype_->delTrackSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_TrackPrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_TrackPrototype, container);
                    }
                    else if (paramToolId == ODD::TTE_PROTO_LANE)
                    {
                        currentRoadPrototype_->delLaneSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_LaneSectionPrototype, container);
                    }
                    else if (paramToolId == ODD::TTE_PROTO_ELEVATION)
                    {
                        currentRoadPrototype_->delElevationSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_ElevationPrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_ElevationPrototype, container);
                    }
                    else if (paramToolId == ODD::TTE_PROTO_SUPERELEVATION)
                    {
                        currentRoadPrototype_->delSuperelevationSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_SuperelevationPrototype, container);
                    }
                    else if (paramToolId == ODD::TTE_PROTO_CROSSFALL)
                    {
                        currentRoadPrototype_->delCrossfallSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_CrossfallPrototype, container);
                    }
                    else if (paramToolId == ODD::TTE_PROTO_TYPE)
                    {
                        currentRoadPrototype_->delTypeSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadTypePrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_RoadTypePrototype, container);
                    }
                    else if (paramToolId == ODD::TTE_PROTO_ROADSHAPE)
                    {
                        currentRoadPrototype_->delShapeSections();
                        container = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadShapePrototype).at(index);
                        currentPrototypes_.insert(PrototypeManager::PTP_RoadShapePrototype, container);
                    }

                    currentRoadPrototype_->superposePrototype(container->getPrototype());

                    ToolParameter* p = tool_->getLastParam(settings_->getCurrentParameterID());
                    ToolValue<int>* v = dynamic_cast<ToolValue<int> *>(p);
                    v->setValue(index);
                }
            }
            if ((lastTool == ODD::TTE_PROTO_FETCH) && ((toolAction->getToolId() == ODD::TTE_ADD) || (toolAction->getToolId() == ODD::TTE_ADD_PROTO)))
            {
                settings_->uncheckButton();
                trackRoadSystemItem_->rebuildAddHandles();
            }
        }
    }

    // Change Tool //
    //
    if (lastTool != getCurrentTool())
    {

        // State //
        //
        state_ = TrackEditor::STE_NONE;

        // Move Tile Tool //
        //
        if (lastTool == ODD::TTE_TILE_MOVE)
        {
            selectedRoads_.clear();
        }

        // Move and Rotate Tool //
        //
        else if (getCurrentTool() == ODD::TTE_MOVE_ROTATE)
        {
            if (trackRoadSystemItem_)
            {
                trackRoadSystemItem_->rebuildMoveRotateHandles();
            }
        }

        // Add Tool //
        //
        else if ((toolAction->getToolId() == ODD::TTE_ADD) || (toolAction->getToolId() == ODD::TTE_ADD_PROTO))
        {
            ODD::ToolId paramTool = getCurrentParameterTool();
            if ((paramTool == ODD::TNO_TOOL) && trackRoadSystemItem_)
            {
                trackRoadSystemItem_->rebuildAddHandles();
            }
        }


        // Move & Rotate Tool //
        //
        else if (getCurrentTool() == ODD::TTE_ROAD_MOVE_ROTATE)
        {
            if (trackRoadSystemItem_)
            {
                trackRoadSystemItem_->rebuildRoadMoveRotateHandles();
            }
        }

        // Tool Without Handles //
        //
        else
        {
            if (trackRoadSystemItem_)
            {
                trackRoadSystemItem_->deleteHandles();
            }
        }

        // Garbage disposal //
        //
        getTopviewGraph()->garbageDisposal();

    }


    if (trackEditorToolAction)
    {

        if (isCurrentTool(ODD::TTE_TILE_NEW))
        {
            Tile *tile = new Tile(getProjectData()->getRoadSystem()->getID(odrID::ID_Tile));
            NewTileCommand *command = new NewTileCommand(tile, getProjectData()->getTileSystem(), NULL);
            getProjectGraph()->executeCommand(command);
        }
        else if (isCurrentTool(ODD::TTE_TILE_DELETE))
        {
            RemoveTileCommand *command = new RemoveTileCommand(getProjectData()->getTileSystem()->getCurrentTile(), getProjectData()->getRoadSystem(), NULL);
            getProjectGraph()->executeCommand(command);
        }
    }

}

QMap<QGraphicsItem *, RSystemElementRoad *>
TrackEditor::getSelectedRoads(int count)
{
    QMap<QGraphicsItem *, RSystemElementRoad *> selected;
    QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
    for (int i = 0; i < selectedItems.size(); i++)
    {
        QGraphicsItem *item = selectedItems.at(i);

        if (selected.size() < count)
        {
            TrackElementItem *trackItem = dynamic_cast<TrackElementItem *>(item);
            if (trackItem)
            {
                RSystemElementRoad *road = trackItem->getParentTrackRoadItem()->getRoad();
                if (!selected.key(road, NULL))
                {
                    selected.insert(item, road);
                    continue;
                }
            }
        }

        item->setSelected(false);
    }

    return selected;
}

//################//
// MOUSE & KEY    //
//################//

/*! \brief .
*
*/
void
TrackEditor::mouseAction(MouseAction *mouseAction)
{
    QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();

    // ADD //
    //
    if ((getCurrentTool() == ODD::TTE_ADD) || (getCurrentTool() == ODD::TTE_ADD_PROTO))
    {
        if (selectedTrackAddHandles_.size() == 0)
        {
            // do nothing //
        }

        else if (selectedTrackAddHandles_.size() == 1)
        {
            if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
            {
                // Intercept mouse action //
                //
//                mouseAction->intercept();

                // Abort //
                //
                if (mouseAction->getEvent()->button() == Qt::RightButton)
                {
                    selectedTrackAddHandles_.begin().value()->setSelected(false); // deselect
                }

                // Add //
                //
                else if (mouseAction->getEvent()->button() == Qt::LeftButton)
                {
                    bool isStart = selectedTrackAddHandles_.begin().value()->isStart();
                    RSystemElementRoad *road = selectedTrackAddHandles_.begin().value()->getRoad();
                    pressPoint_ = mouseAction->getEvent()->scenePos();

                    if (getCurrentTool() == ODD::TTE_ADD)
                    {
                        // Line //
                        //
                        if (geometryPrimitiveType_ == GeometryPrimitive::LINE)
                        {
                            TrackComponent *track = NULL;
                            RSystemElementRoad *linePrototype = NULL;
                            double length = 0.0;

                            if (isStart)
                            {
                                // Last track (at the front) where the new one should be appended //
                                //
                                track = road->getTrackComponent(0.0);

                                // Calculate Length //
                                //
                                QVector2D t = track->getGlobalTangent(track->getSStart()); // length = 1
                                QVector2D d = QVector2D(track->getGlobalPoint(track->getSStart()) - pressPoint_); // length != 1
                                length = QVector2D::dotProduct(t, d);
                            }
                            else
                            {
                                // Last track (at the end) where the new one should be appended //
                                //
                                track = road->getTrackComponent(road->getLength());

                                // Calculate Length //
                                //
                                QVector2D t = track->getGlobalTangent(track->getSEnd()); // length = 1
                                QVector2D d = QVector2D(pressPoint_ - track->getGlobalPoint(track->getSEnd())); // length != 1
                                length = QVector2D::dotProduct(t, d);
                            }

                            if (length <= 1.0)
                            {
                                printStatusBarMsg(tr("A line can not be inserted here."), 0);
                                mouseAction->intercept();
                                return; // line with this length not possible
                            }

                            // Protoype //
                            //
                            TrackElementLine *line = new TrackElementLine(0.0, 0.0, 0.0, 0.0, length);

                            linePrototype = new RSystemElementRoad("prototype");
                            linePrototype->addTrackComponent(line);

                            // Append Prototype //
                            //
                            linePrototype->superposePrototype(currentRoadPrototype_);

                            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Append Roads"));
                            AppendRoadPrototypeCommand *command = new AppendRoadPrototypeCommand(road, linePrototype, isStart, NULL);
                            getProjectGraph()->executeCommand(command);
                            LinkLanesCommand *linkLanesCommand = new LinkLanesCommand(road);
                            getProjectGraph()->executeCommand(linkLanesCommand);
                            getProjectData()->getUndoStack()->endMacro();
                        }

                        // Polynomial //
                        //

                        // Curve //
                        //
                        else if ((geometryPrimitiveType_ == GeometryPrimitive::ARC_SPIRAL) || (geometryPrimitiveType_ == GeometryPrimitive::POLYNOMIAL))
                        {
                            // Declarations //
                            //
                            TrackComponent *track = NULL;
                            RSystemElementRoad *spiralPrototype = NULL;
                            double newHeadingDeg = 0.0;
                            bool foundHandle = false;
                            bool foundHandleIsStart = false;

                            // Look for existing handles //
                            //
                            foreach(QGraphicsItem * item, getTopviewGraph()->getScene()->items(pressPoint_, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
                            {
                                if (!foundHandle)
                                {
                                    TrackAddHandle *handle = dynamic_cast<TrackAddHandle *>(item);
                                    if (handle)
                                    {
                                        pressPoint_ = handle->pos();
                                        newHeadingDeg = handle->rotation();
                                        foundHandle = true;
                                        foundHandleIsStart = handle->isStart();
                                    }
                                }
                            }


                            // Adjust translation and rotation //
                            //
                            if (isStart)
                            {
                                // Old code: see backups until 18.06.2010

                                // Last track (at the front) where the new one should be appended //
                                //
                                track = road->getTrackComponent(0.0);

                                // Calculate Transformation //
                                //
                                QVector2D t = track->getGlobalTangent(track->getSStart()); // length = 1
                                QVector2D d = QVector2D(track->getGlobalPoint(track->getSStart()) - pressPoint_).normalized(); // length = 1
                                if (!foundHandle)
                                {
                                    newHeadingDeg = track->getGlobalHeading(track->getSStart()) + 2.0 * (atan2(d.y(), d.x()) - atan2(t.y(), t.x())) * 360.0 / (2.0 * M_PI);

                                }

                                QPointF startPoint = track->getGlobalPoint(track->getSStart());
                                double startHeadingDeg = track->getGlobalHeading(track->getSStart());

                                QTransform trafo;
                                trafo.translate(pressPoint_.x(), pressPoint_.y());
                                trafo.rotate(newHeadingDeg);


                                if (geometryPrimitiveType_ == GeometryPrimitive::ARC_SPIRAL)
                                {
                                    if (QVector2D::dotProduct(t, d) <= 0.1 /*NUMERICAL_ZERO8*/) // zero or negative
                                    {
                                        printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 0);
                                        mouseAction->intercept();
                                        return; // symmetric curve not possible
                                    }

                                    // Protoype //
                                    //
                                    TrackSpiralArcSpiral *spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(startPoint), 0.0, startHeadingDeg - newHeadingDeg, 0.5);

                                    if (!spiral->validParameters())
                                    {
                                        delete spiral;

                                        // Try opposite heading // (neccessary when the found handle points to the other direction)
                                        //
                                        trafo.rotate(180);
                                        spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(startPoint), 0.0, startHeadingDeg - newHeadingDeg + 180, 0.5);
                                        ;
                                        if (!spiral->validParameters())
                                        {
                                            delete spiral;
                                            printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 0);
                                            mouseAction->intercept();
                                            return;
                                        }
                                    }

                                    spiralPrototype = new RSystemElementRoad("prototype");
                                    spiralPrototype->addTrackComponent(spiral);
                                }
                                else
                                {
                                    // Protoype //
                                    //

                                    if (foundHandle && foundHandleIsStart)
                                    {
                                        newHeadingDeg += 180;
                                        trafo.rotate(180);
                                    }
                                    QPointF mappedPoint = trafo.inverted().map(startPoint);

                                    while (newHeadingDeg > 360) newHeadingDeg -= 360.0;
                                    while (newHeadingDeg < -360) newHeadingDeg += 360.0;
                                    double angle = startHeadingDeg - newHeadingDeg;
                                    double absAngle = fabs(angle);

                                    if (absAngle >= 90.0 && absAngle <= 270.0)
                                    {
                                        if (foundHandle)
                                        {
                                            if (angle < 180)
                                            {
                                                newHeadingDeg += 180;
                                            }
                                            else
                                            {
                                                newHeadingDeg -= 180;
                                            }
                                        }
                                        else
                                        {
                                            printStatusBarMsg(tr("A polynomial curve can not be inserted here."), 0);
                                            mouseAction->intercept();
                                            return;
                                        }

                                    }


                                    double df1 = tan(angle * 2.0 * M_PI / 360.0);
                                    Polynomial *poly3 = new Polynomial(0.0, 0.0, mappedPoint.y(), df1, mappedPoint.x());

                                    if (poly3->getCurveLength(0.0, mappedPoint.x()) < 0.0)
                                    {
                                        printStatusBarMsg(tr("A polynomial curve can not be inserted here."), 0);
                                        mouseAction->intercept();
                                        return;
                                    }

                                    // Track //
                                    //
                                    TrackElementPoly3 *poly = new TrackElementPoly3(pressPoint_.x(), pressPoint_.y(), newHeadingDeg, 0.0, poly3->getCurveLength(0.0, mappedPoint.x()), *poly3);


                                    spiralPrototype = new RSystemElementRoad("prototype");
                                    spiralPrototype->addTrackComponent(poly);

                                }


                            }
                            else
                            {
                                // Old code: see backups until 18.06.2010

                                // Last track (at the end) where the new one should be appended //
                                //
                                track = road->getTrackComponent(road->getLength());

                                // Calculate Transformation //
                                //
                                QVector2D t = track->getGlobalTangent(track->getSEnd()); // length = 1
                                QVector2D d = QVector2D(pressPoint_ - track->getGlobalPoint(track->getSEnd())).normalized(); // length = 1
                                if (!foundHandle)
                                {
                                    newHeadingDeg = track->getGlobalHeading(track->getSEnd()) + 2.0 * (atan2(d.y(), d.x()) - atan2(t.y(), t.x())) * 360.0 / (2.0 * M_PI);
                                }

                                QPointF startPoint = track->getGlobalPoint(track->getSEnd());
                                double startHeadingDeg = track->getGlobalHeading(track->getSEnd());

                                QTransform trafo;
                                trafo.translate(startPoint.x(), startPoint.y());
                                trafo.rotate(startHeadingDeg);

                                if (geometryPrimitiveType_ == GeometryPrimitive::ARC_SPIRAL)
                                {

                                    if (QVector2D::dotProduct(t, d) <= 0.1 /*NUMERICAL_ZERO8*/) // zero or negative
                                    {
                                        printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 0);
                                        mouseAction->intercept();
                                        return; // symmetric curve not possible
                                    }

                                    // Protoype //
                                    //
                                    TrackSpiralArcSpiral *spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(pressPoint_), 0.0, newHeadingDeg - startHeadingDeg, 0.5);
                                    if (!spiral->validParameters())
                                    {
                                        delete spiral;

                                        // Try opposite heading // (neccessary when the found handle points to the other direction)
                                        //
                                        spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(pressPoint_), 0.0, newHeadingDeg - startHeadingDeg + 180.0, 0.5);
                                        if (!spiral->validParameters())
                                        {
                                            delete spiral;
                                            printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 0);
                                            mouseAction->intercept();
                                            return;
                                        }
                                    }

                                    spiralPrototype = new RSystemElementRoad("prototype");
                                    spiralPrototype->addTrackComponent(spiral);
                                }
                                else
                                {

                                    // Prototype //
                                    //
                                    double angle = newHeadingDeg - startHeadingDeg;
                                    while (angle > 360) angle -= 360.0;
                                    while (angle < -360) angle += 360.0;
                                    double absAngle = fabs(angle);

                                    if (absAngle >= 90.0 && absAngle <= 270.0)
                                    {
                                        if (foundHandle)
                                        {
                                            if (absAngle < 180)
                                            {
                                                angle += 180;
                                            }
                                            else
                                            {
                                                angle -= 180;
                                            }
                                        }
                                        else
                                        {
                                            printStatusBarMsg(tr("A polynomial curve can not be inserted here."), 0);
                                            mouseAction->intercept();
                                            return;
                                        }
                                    }


                                    /*      QTransform rot;
                                                            double df1;
                                                            TrackElementPoly3 *poly;
                                                            if (angle < 180)
                                                            {
                                                                rot.rotate(-90.0);
                                                                df1 = tan((newHeadingDeg - startHeadingDeg - 90.0) * 2.0 * M_PI / 360.0);

                                                                QPointF mappedPoint = trafo.inverted().map(pressPoint_);
                                                                Polynomial *poly3 = new Polynomial(0.0, 0.0, mappedPoint.x(), df1, mappedPoint.y());
                                                                double length = poly3->getCurveLength(0.0, mappedPoint.x());
                                                                double f = poly3->f(mappedPoint.y());

                                                                // Track //
                                                                //
                                                                double s = track->getSEnd();
                                                                poly = new TrackElementPoly3(startPoint.x(), startPoint.y(), track->getLocalHeading(s) + 90.0, 0.0, poly3->getCurveLength(0.0, mappedPoint.y()), *poly3);

                                                            }
                                                            else
                                                            {
                                                                rot.rotate(90.0);
                                                                df1 = tan((newHeadingDeg - startHeadingDeg + 90.0) * 2.0 * M_PI / 360.0);

                                                                QPointF mappedPoint = trafo.inverted().map(pressPoint_);
                                                                Polynomial *poly3 = new Polynomial(0.0, 0.0, rot.map(mappedPoint).y(), df1, rot.map(mappedPoint).x());

                                                                // Track //
                                                                //
                                                                double s = track->getSEnd();
                                                                poly = new TrackElementPoly3(startPoint.x(), startPoint.y(), track->getLocalHeading(s) + 90.0, 0.0, poly3->getCurveLength(0.0, rot.map(mappedPoint).x()), *poly3);
                                                            }


                                                            spiralPrototype = new RSystemElementRoad("prototype");
                                                            spiralPrototype->addTrackComponent(poly);
                                                            road->getRoadSystem()->addRoad(spiralPrototype); */



                                    QPointF mappedPoint = trafo.inverted().map(pressPoint_);
                                    double df1 = tan(angle * 2.0 * M_PI / 360.0);
                                    Polynomial *poly3 = new Polynomial(0.0, 0.0, mappedPoint.y(), df1, mappedPoint.x());
                                    if (poly3->getCurveLength(0.0, mappedPoint.x()) < 0.0)
                                    {
                                        printStatusBarMsg(tr("A polynomial curve can not be inserted here."), 0);
                                        mouseAction->intercept();
                                        return;
                                    }

                                    // Track //
                                    //
                                    double s = track->getSEnd();
                                    TrackElementPoly3 *poly = new TrackElementPoly3(startPoint.x(), startPoint.y(), track->getLocalHeading(s), 0.0, poly3->getCurveLength(0.0, mappedPoint.x()), *poly3);

                                    spiralPrototype = new RSystemElementRoad("prototype");
                                    spiralPrototype->addTrackComponent(poly);


                                }
                            }

                            // Append Prototype //
                            //
                            spiralPrototype->superposePrototype(currentRoadPrototype_);

                            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Append Roads"));

                            AppendRoadPrototypeCommand *command = new AppendRoadPrototypeCommand(road, spiralPrototype, isStart, NULL);
                            getProjectGraph()->executeCommand(command);
                            LinkLanesCommand *linkLanesCommand = new LinkLanesCommand(road);
                            getProjectGraph()->executeCommand(linkLanesCommand);
                            getProjectData()->getUndoStack()->endMacro();
                        }
                        mouseAction->intercept();
                    }

                    // Prototypes //
                    //
                    else if (getCurrentTool() == ODD::TTE_ADD_PROTO)
                    {
                        getProjectData()->getUndoStack()->beginMacro(QObject::tr("Append Roads"));
                        AppendRoadPrototypeCommand *command = new AppendRoadPrototypeCommand(road, currentRoadPrototype_, isStart, NULL);
                        getProjectGraph()->executeCommand(command);
                        LinkLanesCommand *linkLanesCommand = new LinkLanesCommand(road);
                        getProjectGraph()->executeCommand(linkLanesCommand);
                        getProjectData()->getUndoStack()->endMacro();
                    }
                }
            }

            else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
            {
            }

            else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
            {
            }
        }
    }

    // NEW //
    //
    else if (tool_ && (tool_->getToolId() == ODD::TTE_ROAD_NEW))
    {

        QPointF mousePoint = mouseAction->getEvent()->scenePos();

        // Length //
        //
        if (mouseEvent->modifiers() & Qt::ControlModifier)
        {
            mousePoint = QPointF(floor(mousePoint.x() + 0.5), floor(mousePoint.y() + 0.5)); // rounded = floor(x+0.5)
        }
        else
        {
            mousePoint = mouseAction->getEvent()->scenePos();
        }
        QVector2D mouseLine(mousePoint - pressPoint_);
        double length = mouseLine.length();

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
        {
            if (mouseEvent->button() == Qt::LeftButton)
            {
                if (mouseEvent->modifiers() & Qt::ControlModifier)
                {
                    pressPoint_ = QPointF(floor(mouseAction->getEvent()->scenePos().x() + 0.5), floor(mouseAction->getEvent()->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
                }
                else
                {
                    pressPoint_ = mouseAction->getEvent()->scenePos();
                }

                newRoadLineItem_->setLine(QLineF(pressPoint_, mousePoint));
                newRoadLineItem_->show();

                state_ = TrackEditor::STE_NEW_PRESSED;
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (state_ == TrackEditor::STE_NEW_PRESSED)
            {
                newRoadLineItem_->setLine(QLineF(pressPoint_, mousePoint));
                printStatusBarMsg(QString("New road: (%1, %2) to (%3, %4). Length: %5.").arg(pressPoint_.x()).arg(pressPoint_.y()).arg(mousePoint.x()).arg(mousePoint.y()).arg(length), 0);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
            if (mouseAction->getEvent()->button() == Qt::LeftButton)
            {
                state_ = TrackEditor::STE_NONE;
                newRoadLineItem_->hide();

                if (length < 10.0)
                {
                    printStatusBarMsg("New road: to short. Please click and drag.", 0);
                }
                else
                {
                    // Road //
                    //
                    RSystemElementRoad *newRoad = new RSystemElementRoad("unnamed");

                    // Track //
                    //
                    if ((geometryPrimitiveType_ == GeometryPrimitive::ARC_SPIRAL) || (geometryPrimitiveType_ == GeometryPrimitive::LINE))
                    {
                        TrackElementLine *line = new TrackElementLine(pressPoint_.x(), pressPoint_.y(), atan2(mouseLine.y(), mouseLine.x()) * 360.0 / (2.0 * M_PI), 0.0, length);
                        newRoad->addTrackComponent(line);
                    }
                    else
                    {

                        Polynomial *poly3 = new Polynomial(0.0, 0.0, 0.0, 0.0);  // always start with a straight line

                        // Track //
                        //
                        TrackElementPoly3 *poly = new TrackElementPoly3(pressPoint_.x(), pressPoint_.y(), atan2(mouseLine.y(), mouseLine.x()) * 180 / M_PI, 0.0, mouseLine.length(), *poly3);
                        newRoad->addTrackComponent(poly);
                    }


                    // Superpose user prototypes //
                    //
                    newRoad->superposePrototype(currentRoadPrototype_);

                    NewRoadCommand *command = new NewRoadCommand(newRoad, getProjectData()->getRoadSystem(), NULL);
                    if (command->isValid())
                    {
                        getProjectData()->getUndoStack()->push(command);
                    }
                    else
                    {
                        printStatusBarMsg(command->text(), 0);
                        delete command;
                        return; // usually not the case, only if road or prototype are NULL
                    }


                }
            }
        }
    }
    // NEW CIRCLE //
    //
    else if (getCurrentTool() == ODD::TTE_ROAD_CIRCLE)
    {
        QPointF mousePoint = mouseAction->getEvent()->scenePos();

        // Length //
        //
        if (mouseEvent->modifiers() & Qt::ControlModifier)
        {
            mousePoint = QPointF(floor(mousePoint.x() + 0.5), floor(mousePoint.y() + 0.5)); // rounded = floor(x+0.5)
        }
        else
        {
            mousePoint = mouseAction->getEvent()->scenePos();
        }
        QVector2D mouseLine(mousePoint - pressPoint_);
        double length = mouseLine.length();

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
        {
            if (mouseEvent->button() == Qt::LeftButton)
            {
                if (mouseEvent->modifiers() & Qt::ControlModifier)
                {
                    pressPoint_ = QPointF(floor(mouseAction->getEvent()->scenePos().x() + 0.5), floor(mouseAction->getEvent()->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
                }
                else
                {
                    pressPoint_ = mouseAction->getEvent()->scenePos();
                }
                QPointF diff = pressPoint_ - mousePoint;
                double radius = sqrt(pow(diff.x(), 2) + pow(diff.y(), 2));
                state_ = TrackEditor::STE_NEW_PRESSED;
                newRoadCircleItem_->setRect(pressPoint_.x() - radius, pressPoint_.y() - radius, radius * 2, radius * 2);
                newRoadCircleItem_->show();
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (state_ == TrackEditor::STE_NEW_PRESSED)
            {
                QPointF diff = pressPoint_ - mousePoint;
                double radius = sqrt(pow(diff.x(), 2) + pow(diff.y(), 2));
                state_ = TrackEditor::STE_NEW_PRESSED;
                newRoadCircleItem_->setRect(pressPoint_.x() - radius, pressPoint_.y() - radius, radius * 2, radius * 2);
                printStatusBarMsg(QString("New circle: (%1, %2) to (%3, %4). Length: %5.").arg(pressPoint_.x()).arg(pressPoint_.y()).arg(mousePoint.x()).arg(mousePoint.y()).arg(length), 0);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
            if (mouseAction->getEvent()->button() == Qt::LeftButton)
            {
                state_ = TrackEditor::STE_NONE;
                newRoadCircleItem_->hide();

                if (length < 10.0)
                {
                    printStatusBarMsg("New circle: to short. Please click and drag.", 0);
                }
                else
                {
                    // Road //
                    //
                    RSystemElementRoad *newRoad = new RSystemElementRoad("unnamed");

                    // Track //
                    //
                    QPointF diff = pressPoint_ - mousePoint;
                    double radius = sqrt(pow(diff.x(), 2) + pow(diff.y(), 2));
                    TrackElementArc *arc1 = new TrackElementArc(pressPoint_.x(), pressPoint_.y() - radius, 0.0, 0.0, 2 * M_PI * radius, 1 / radius);
                    newRoad->addTrackComponent(arc1);

                    // Append Prototype //
                    //
                    newRoad->superposePrototype(currentRoadPrototype_);
                    NewRoadCommand *command = new NewRoadCommand(newRoad, getProjectData()->getRoadSystem(), NULL);
                    if (command->isValid())
                    {
                        getProjectData()->getUndoStack()->push(command);
                    }
                    else
                    {
                        printStatusBarMsg(command->text(), 0);
                        delete command;
                        return; // usually not the case, only if road or prototype are NULL
                    }
                }
            }
        }
    }

    // DELETE //
    //
    else if (getCurrentTool() == ODD::TTE_DELETE)
    {
        //   qDebug("TODO: TrackEditor: DELETE");
    }

    // NEW //
    //
    else if (getCurrentTool() == ODD::TTE_ROADSYSTEM_ADD)
    {
        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
        {
            if (mouseEvent->button() == Qt::LeftButton)
            {
                addRoadSystemHandle_->setVisible(true);
                addRoadSystemHandle_->setPos(mouseEvent->scenePos());
                addRoadSystemHandle_->setMousePos(mouseEvent->scenePos());
                state_ = TrackEditor::STE_ROADSYSTEM_ADD;
                printStatusBarMsg(QString("Add Prototype at (%1,%2), angle %3").arg(addRoadSystemHandle_->getPos().x()).arg(addRoadSystemHandle_->getPos().y()).arg(addRoadSystemHandle_->getAngle()), 0);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (state_ == TrackEditor::STE_ROADSYSTEM_ADD)
            {
                addRoadSystemHandle_->setMousePos(mouseEvent->scenePos());
                printStatusBarMsg(QString("Add Prototype at (%1,%2), angle %3").arg(addRoadSystemHandle_->getPos().x()).arg(addRoadSystemHandle_->getPos().y()).arg(addRoadSystemHandle_->getAngle()), 0);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
            if (state_ == TrackEditor::STE_ROADSYSTEM_ADD)
            {
                state_ = TrackEditor::STE_NONE;
                addRoadSystemHandle_->setVisible(false);

                foreach(RSystemElementRoad * road, currentRoadSystemPrototype_->getPrototype()->getRoads())
                {
                    road->delLaneSections();
                    road->delTypeSections();
                    road->delElevationSections();
                    road->delSuperelevationSections();
                    road->delCrossfallSections();
                    road->delShapeSections();
                    road->superposePrototype(currentRoadPrototype_);
                }

                AddRoadSystemPrototypeCommand *command = new AddRoadSystemPrototypeCommand(getProjectData()->getRoadSystem(), currentRoadSystemPrototype_->getPrototype(), addRoadSystemHandle_->getPos(), addRoadSystemHandle_->getAngle());
                //    AddRoadSystemPrototypeCommand * command = new AddRoadSystemPrototypeCommand(getProjectData()->getRoadSystem(), currentRoadSystemPrototype_, QPointF(0.0, 0.0), 0.0);
                if (command->isValid())
                {
                    getProjectData()->getUndoStack()->push(command);
                }
                else
                {
                    printStatusBarMsg(command->text(), 0);
                    delete command;
                    return; // usually not the case, only if road or prototype are NULL
                }
            }
        }
    }

    // Move Tile //
    //
    else if (getCurrentTool() == ODD::TTE_TILE_MOVE)
    {
        QPointF mousePoint = mouseAction->getEvent()->scenePos();

        // Length //
        //
        if (mouseEvent->modifiers() & Qt::ControlModifier)
        {
            mousePoint = QPointF(floor(mousePoint.x() + 0.5), floor(mousePoint.y() + 0.5)); // rounded = floor(x+0.5)
        }
        else
        {
            mousePoint = mouseAction->getEvent()->scenePos();
        }

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS) //Bounding Box
        {
            if (mouseEvent->button() == Qt::LeftButton)
            {
                if (mouseEvent->modifiers() & Qt::ControlModifier)
                {
                    pressPoint_ = QPointF(floor(mouseAction->getEvent()->scenePos().x() + 0.5), floor(mouseAction->getEvent()->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
                }
                else
                {
                    pressPoint_ = mouseAction->getEvent()->scenePos();
                }
                state_ = TrackEditor::STE_NEW_PRESSED;

                currentTile_ = getProjectData()->getTileSystem()->getCurrentTile()->getID();
                selectedRoads_ = getProjectData()->getRoadSystem()->getTileRoads(currentTile_);

                for (int i = 0; i < selectedRoads_.size(); i++)
                {
                    selectedRoads_.at(i)->setElementSelected(true);
                }
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE) //Bounding Box
        {
            if (state_ == TrackEditor::STE_NEW_PRESSED)
            {
                MoveRoadCommand *command = new MoveRoadCommand(selectedRoads_, mousePoint - pressPoint_, NULL);
                getProjectGraph()->executeCommand(command);

                pressPoint_ = mousePoint;
            }
        }
        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
            if (mouseAction->getEvent()->button() == Qt::LeftButton)
            {
                state_ = TrackEditor::STE_NONE;
            }
        }
    }

    ProjectEditor::mouseAction(mouseAction);
}

/*! \brief .
*
*/
void
TrackEditor::keyAction(KeyAction *keyAction)
{
    ProjectEditor::keyAction(keyAction);
}

bool
TrackEditor::translateTrack(TrackComponent *track, const QPointF &pressPos, const QPointF &mousePos)
{
    if (!track)
    {
        return false;
    }

    QPointF dPos = mousePos - pressPos;

    MoveTrackCommand *command = new MoveTrackCommand(track, track->getGlobalPoint(track->getSStart()) + dPos, NULL);
    return getProjectGraph()->executeCommand(command);
}

//################//
// MoveHandles    //
//################//

void
TrackEditor::registerTrackMoveHandle(TrackMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! TrackEditor TrackMoveHandle DOF not in [0,1,2].");
    }
    selectedTrackMoveHandles_.insert(handle->getPosDOF(), handle);
}

int
TrackEditor::unregisterTrackMoveHandle(TrackMoveHandle *handle)
{
    return selectedTrackMoveHandles_.remove(handle->getPosDOF(), handle);
}

bool
TrackEditor::translateTrackMoveHandles(const QPointF &pressPos, const QPointF &mousePosConst)
{
    QPointF mousePos = mousePosConst;

    // Snap to MoveHandle //
    //
    foreach(QGraphicsItem * item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
    {
        TrackMoveHandle *handle = dynamic_cast<TrackMoveHandle *>(item);
        if (handle)
        {
            mousePos = handle->pos();
            //   newHeadingDeg = handle->rotation();
            break;
        }
    }

    // DeltaPos //
    //
    QPointF dPos = mousePos - pressPos;

    // TODO:
    //
    // DOF bestimmung in die (un)register functions auslagern
    // => effizienter

    //qDebug() << "selectedTrackMoveHandles_: " << selectedTrackMoveHandles_.size();

    // No entries //
    //
    if (selectedTrackMoveHandles_.size() == 0)
    {
        return false;
    }

    // 0: Check for zero degrees of freedom //
    //
    if (selectedTrackMoveHandles_.count(0) > 0)
    {
        return false;
    }

    // 1: Check for one degree of freedom //
    //
    if (selectedTrackMoveHandles_.count(1) > 0)
    {
        // Check if all are the same //
        //
        double posDofHeading = selectedTrackMoveHandles_.constBegin().value()->getPosDOFHeading(); // first
        foreach(TrackMoveHandle * moveHandle, selectedTrackMoveHandles_.values(1))
        {
            if (fabs(posDofHeading - moveHandle->getPosDOFHeading()) > NUMERICAL_ZERO3)
            {
                return false;
            }
        }

        // Map dPos to DOF //
        //
        QVector2D direction = QVector2D(cos(posDofHeading * 2.0 * M_PI / 360.0), sin(posDofHeading * 2.0 * M_PI / 360.0));
        dPos = (QVector2D::dotProduct(QVector2D(dPos), direction) * direction).toPointF();
    }
    // QPointF targetPos = pressPos + dPos;
    //qDebug() << "dPos2: " << dPos;

    // 2: Two degrees of freedom //
    //
    TrackMoveValidator *validationVisitor = new TrackMoveValidator();
    validationVisitor->setGlobalDeltaPos(dPos);

    foreach(TrackMoveHandle * moveHandle, selectedTrackMoveHandles_)
    {
        if (moveHandle->getHighSlot())
        {
            validationVisitor->setState(TrackMoveValidator::STATE_STARTPOINT);
            moveHandle->getHighSlot()->accept(validationVisitor);
        }

        if (moveHandle->getLowSlot())
        {
            validationVisitor->setState(TrackMoveValidator::STATE_ENDPOINT);
            moveHandle->getLowSlot()->accept(validationVisitor);
        }
    }

    // Check if translation is valid //
    //
    if (!validationVisitor->isValid())
    {
        delete validationVisitor;
        return false; // not valid => no translation
    }
    else
    {
        delete validationVisitor;
    }

    // Translate all handles //
    //
    QList<TrackComponent *> endPointTracks;
    QList<TrackComponent *> startPointTracks;
    foreach(TrackMoveHandle * moveHandle, selectedTrackMoveHandles_)
    {
        TrackComponent *lowSlot = moveHandle->getLowSlot();
        if (lowSlot)
        {
            endPointTracks.append(lowSlot);
        }

        TrackComponent *highSlot = moveHandle->getHighSlot();
        if (highSlot)
        {
            startPointTracks.append(highSlot);
        }
    }

    TrackComponentGlobalPointsCommand *command = new TrackComponentGlobalPointsCommand(endPointTracks, startPointTracks, dPos, NULL);
    getProjectGraph()->executeCommand(command);

#if 0
    foreach(TrackMoveHandle * moveHandle, selectedTrackMoveHandles_)
    {
        TrackComponent *lowSlot = moveHandle->getLowSlot();
        TrackComponent *highSlot = moveHandle->getHighSlot();
        if (lowSlot || highSlot)
        {
            //   TrackComponentSinglePointCommand * command = new TrackComponentSinglePointCommand(lowSlot, highSlot, targetPos, NULL);
            TrackComponentSinglePointCommand *command = new TrackComponentSinglePointCommand(lowSlot, highSlot, dPos, NULL);
            getProjectGraph()->executeCommand(command);
        }
    }
#endif

    return true;
}

bool
TrackEditor::translateTrackComponents(const QPointF &pressPos, const QPointF &mousePosConst)
{
    QPointF mousePos = mousePosConst;

    // Snap to MoveHandle //
    //
    foreach(QGraphicsItem * item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
    {
        TrackMoveHandle *handle = dynamic_cast<TrackMoveHandle *>(item);
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

    QMultiMap<TrackComponent *, bool> selectedTrackComponents;
    QMultiMap<int, TrackMoveHandle *>::const_iterator it = selectedTrackMoveHandles_.constBegin();
    while (it != selectedTrackMoveHandles_.constEnd())
    {
        if ((*it)->getLowSlot())
        {
            selectedTrackComponents.insert((*it)->getLowSlot(), false);
        }
        if ((*it)->getHighSlot())
        {
            selectedTrackComponents.insert((*it)->getHighSlot(), true);
        }

        it++;
    }

    TranslateTrackComponentsCommand *command = new TranslateTrackComponentsCommand(selectedTrackComponents, mousePos, pressPos, NULL);
    return getProjectGraph()->executeCommand(command);
}

#if 0

//################//
// RotateHandles  //
//################//

void
TrackEditor
::registerTrackRotateHandle(TrackRotateHandle *handle)
{
    if (handle->getRotDOF() < 0 || handle->getRotDOF() > 1)
    {
        qDebug("WARNING 1004261417! TrackEditor TrackRotateHandle DOF not in [0,1].");
    }
    selectedTrackRotateHandles_.insert(handle->getRotDOF(), handle);
}

int
TrackEditor
::unregisterTrackRotateHandle(TrackRotateHandle *handle)
{
    return selectedTrackRotateHandles_.remove(handle->getRotDOF(), handle);
}



double
TrackEditor
::rotateTrackRotateHandles(double dHeading, double globalHeading)
{
    // No entries //
    //
    if (selectedTrackRotateHandles_.size() == 0)
    {
        return dHeading;
    }

    // 0: Check for zero degrees of freedom //
    //
    if (selectedTrackRotateHandles_.count(0) > 0)
    {
        return 0.0; // no rotation
    }


    // 1: Check for one degree of freedom //
    //
    TrackMoveValidator *validationVisitor = new TrackMoveValidator();
    validationVisitor->setGlobalHeading(dHeading + globalHeading);
    foreach(TrackRotateHandle * handle, selectedTrackRotateHandles_)
    {
        if (handle->getHighSlot())
        {
            validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
            handle->getHighSlot()->accept(validationVisitor, false);
        }

        if (handle->getLowSlot())
        {
            validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
            handle->getLowSlot()->accept(validationVisitor, false);
        }
    }

    // Check if rotation is valid //
    //
    if (validationVisitor->isValid())
    {
        //  qDebug("valid");
        return dHeading;
    }
    else
    {
        //  qDebug("not valid");
        return 0.0; // not valid => no rotation
//  return dHeading;
    }
}
#endif

//################//
// AddHandles     //
//################//

void
TrackEditor::registerTrackAddHandle(TrackAddHandle *handle)
{
    selectedTrackAddHandles_.insert(0, handle);
    return;
}

int
TrackEditor::unregisterTrackAddHandle(TrackAddHandle *handle)
{
    return selectedTrackAddHandles_.remove(0, handle);
}

//################//
// RoadMoveHandles //
//################//

void
TrackEditor::registerRoadMoveHandle(RoadMoveHandle *handle)
{
    selectedRoadMoveHandles_.insert(2, handle);
}

int
TrackEditor::unregisterRoadMoveHandle(RoadMoveHandle *handle)
{
    return selectedRoadMoveHandles_.remove(2, handle);
}

bool
TrackEditor::translateRoadMoveHandles(const QPointF &pressPos, const QPointF &mousePos)
{
    QPointF dPos = mousePos - pressPos;

    // No entries //
    //
    if (selectedRoadMoveHandles_.size() == 0)
    {
        return false;
    }

    // Translate all handles //
    //
    QList<RSystemElementRoad *> roads;
    foreach(RoadMoveHandle * moveHandle, selectedRoadMoveHandles_)
    {
        roads.append(moveHandle->getRoad());
    }

    MoveRoadCommand *command = new MoveRoadCommand(roads, dPos, NULL);
    getProjectGraph()->executeCommand(command);

    return true;
}

//################//
// RoadRotateHandles //
//################//

void
TrackEditor::registerRoadRotateHandle(RoadRotateHandle *handle)
{
    selectedRoadRotateHandles_.insert(1, handle);
}

int
TrackEditor::unregisterRoadRotateHandle(RoadRotateHandle *handle)
{
    return selectedRoadRotateHandles_.remove(1, handle);
}

bool
TrackEditor::rotateRoadRotateHandles(const QPointF &pivotPoint, double angleDegrees)
{
    // No entries //
    //
    if (selectedRoadRotateHandles_.size() == 0)
    {
        return false;
    }

    // Translate all handles //
    //
    QList<RSystemElementRoad *> roads;
    foreach(RoadRotateHandle * handle, selectedRoadRotateHandles_)
    {
        roads.append(handle->getRoad());
    }

    RotateRoadAroundPointCommand *command = new RotateRoadAroundPointCommand(roads, pivotPoint, angleDegrees, NULL);
    getProjectGraph()->executeCommand(command);

    return true;
}

void
TrackEditor::setChildCacheMode(QGraphicsItem *child, QGraphicsItem::CacheMode mode)
{
    foreach(QGraphicsItem * item, child->childItems())
    {
        RoadMarkItem *roadMarkItem = dynamic_cast<RoadMarkItem *>(item);
        if (roadMarkItem)
        {
            LaneRoadMark *roadMark = dynamic_cast<LaneRoadMark *>(roadMarkItem->getDataElement());
            if (roadMark && (roadMark->getRoadMarkType() != LaneRoadMark::RMT_NONE))
            {
                item->setCacheMode(mode);
            }
        }
        setChildCacheMode(item, mode);
    }
}

void
TrackEditor::setCacheMode(RSystemElementRoad *road, CacheMode cache)
{
    TrackRoadItem *roadItem = trackRoadSystemItem_->getRoadItem(road);
    if (roadItem)
    {
        if (cache == CacheMode::NoCache)
        {
            foreach(QGraphicsItem * item, roadItem->childItems())
            {
                setChildCacheMode(item, QGraphicsItem::CacheMode::NoCache);
            }
        }
        else
        {
            foreach(QGraphicsItem * item, roadItem->childItems())
            {
                setChildCacheMode(item, QGraphicsItem::CacheMode::DeviceCoordinateCache);
            }
        }
    }
}

//###################//
// ROADS   //
//##################//

bool
TrackEditor::registerRoad(QGraphicsItem *trackItem, RSystemElementRoad *road)
{
    if ((trackItem != mergeItem_) && (trackItem != appendItem_))
    {
        if (!tool_->getValue<RSystemElementRoad>(road))
        {
            ODD::ToolId currentTool = getCurrentTool();

            if (currentTool == ODD::TTE_ROAD_APPEND)
            {
                appendItem_ = trackItem;
            }
            else
            {
                mergeItem_ = trackItem;
            }

            RSystemElementRoad *oldRoad = dynamic_cast<ToolValue<RSystemElementRoad> *>(tool_->getParam(currentTool, ODD::TPARAM_SELECT))->getValue();
            if (oldRoad)
            {
                DeselectDataElementCommand *command = new DeselectDataElementCommand(oldRoad);
                if (command->isValid())
                {
                    getTopviewGraph()->executeCommand(command);
                }
            }

            setToolValue<RSystemElementRoad>(road, road->getIdName());
            SelectDataElementCommand *command = new SelectDataElementCommand(road);
            if (command->isValid())
            {
                getTopviewGraph()->executeCommand(command);
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
TrackEditor::deregisterRoad(QGraphicsItem *trackItem, RSystemElementRoad *road)
{
    ToolParameter *param;
    if (trackItem == mergeItem_)
    {
        if (tool_->containsToolId(ODD::TTE_ROAD_SNAP))
        {
            param = tool_->getParam(ODD::TTE_ROAD_SNAP, ODD::TPARAM_SELECT);
        }
        else
        {
            param = tool_->getParam(ODD::TTE_ROAD_MERGE, ODD::TPARAM_SELECT);
        }
        mergeItem_ = NULL;
    }
    else if (trackItem == appendItem_)
    {
        param = tool_->getParam(ODD::TTE_ROAD_APPEND, ODD::TPARAM_SELECT);
        appendItem_ = NULL;
    }

    delToolValue(param);
    settingsApplyBox_->setApplyButtonVisible(false);

    DeselectDataElementCommand *command = new DeselectDataElementCommand(road);
    if (command->isValid())
    {
        getTopviewGraph()->executeCommand(command);
    }

    return true;
}

//###################//
// PROTOTYPES   //
//##################//

void TrackEditor::fetchRoadPrototypes(RSystemElementRoad* road)
{
    setToolValue<RSystemElementRoad>(road, road->getIdName());
    ODD::ToolId toolId = getCurrentParameterTool();

    appendToolParameter(PrototypeManager::PTP_LaneSectionPrototype, toolId, ODD::TTE_PROTO_LANE, road);
    appendToolParameter(PrototypeManager::PTP_RoadTypePrototype, toolId, ODD::TTE_PROTO_TYPE, road);
    appendToolParameter(PrototypeManager::PTP_ElevationPrototype, toolId, ODD::TTE_PROTO_ELEVATION, road);
    appendToolParameter(PrototypeManager::PTP_SuperelevationPrototype, toolId, ODD::TTE_PROTO_SUPERELEVATION, road);
    appendToolParameter(PrototypeManager::PTP_CrossfallPrototype, toolId, ODD::TTE_PROTO_CROSSFALL, road);
    appendToolParameter(PrototypeManager::PTP_RoadShapePrototype, toolId, ODD::TTE_PROTO_ROADSHAPE, road);

}


//################//
// SLOTS          //
//################//

/*!
*/
void
TrackEditor::init()
{
    if (!trackRoadSystemItem_)
    {
        // Root item //
        //
        trackRoadSystemItem_ = new TrackRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem(), this);
        getTopviewGraph()->getScene()->addItem(trackRoadSystemItem_);

        // New Road Item //
        //
        QPen pen;
        pen.setWidth(2);
        pen.setCosmetic(true); // constant size independent of scaling
        pen.setColor(ODD::instance()->colors()->brightGreen());
        newRoadLineItem_ = new QGraphicsLineItem(trackRoadSystemItem_);
        newRoadLineItem_->setPen(pen);
        newRoadLineItem_->hide();

        newRoadCircleItem_ = new QGraphicsEllipseItem(trackRoadSystemItem_);
        newRoadCircleItem_->setPen(pen);
        newRoadCircleItem_->hide();

        newRoadPolyItem_ = new TrackMoveHandle(this, trackRoadSystemItem_);
        newRoadPolyItem_->setVisible(false);


        // Add RoadSystem Item //
        //
        addRoadSystemHandle_ = new CircularRotateHandle(trackRoadSystemItem_);
        addRoadSystemHandle_->setVisible(false);
    }

    // Section Handle //
    //
    // TODO: Is this really the best object for holding this?
    sectionHandle_ = new SectionHandle(trackRoadSystemItem_);
    sectionHandle_->hide();
}

/*!
*/
void
TrackEditor::kill()
{
    // RoadSystemItem //
    //
    // trackRoadSystemItem_->deleteHandles();
    // getTopviewGraph()->getScene()->removeItem(trackRoadSystemItem_);
    // topviewGraph_->graphScene()->removeItem(trackRoadSystemItem_);

    if (tool_)
    {
        reset();
        ODD::mainWindow()->showParameterDialog(false);
    }

    delete trackRoadSystemItem_;
    trackRoadSystemItem_ = NULL;

    // ToolHandles //
    //
    selectedTrackMoveHandles_.clear();
    selectedTrackAddHandles_.clear();
}

void
TrackEditor::apply()
{
    ODD::ToolId toolId = tool_->getToolId();
    if (toolId == ODD::TTE_ROAD_MERGE)
    {
        RSystemElementRoad *firstRoad = NULL;
        RSystemElementRoad *secondRoad = NULL;
        QList<ToolParameter *> paramList;

        getToolObjectSelection(&paramList, &firstRoad, &secondRoad);


        // Find closest positions of the two roads
    /* double distances[4];

        distances[0] = QVector2D(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(0.0)).length(); // Start Start
        distances[1] = QVector2D(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(0.0)).length(); // End Start
        distances[2] = QVector2D(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(secondRoad->getLength())).length(); // End End
        distances[3] = QVector2D(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(secondRoad->getLength())).length(); // Start End
        MergeRoadsCommand *command = NULL;
        if (distances[0] < distances[1] && distances[0] < distances[2] && distances[0] < distances[3])
            command = new MergeRoadsCommand(firstRoad, secondRoad, true, true);
        if (distances[1] < distances[0] && distances[1] < distances[2] && distances[1] < distances[3])
            command = new MergeRoadsCommand(firstRoad, secondRoad, false, true);
        if (distances[2] < distances[0] && distances[2] < distances[1] && distances[2] < distances[3])
            command = new MergeRoadsCommand(firstRoad, secondRoad, false, false);
        if (distances[3] < distances[0] && distances[3] < distances[1] && distances[3] < distances[2])
            command = new MergeRoadsCommand(firstRoad, secondRoad, true, false); */

        MergeRoadsCommand *command = new MergeRoadsCommand(firstRoad, secondRoad, false, true);
        if (command->isValid())
        {
            paramList.removeFirst();
            clearToolObjectSelection(&paramList, &firstRoad, &secondRoad);
            getProjectGraph()->executeCommand(command);

            SelectDataElementCommand *selectCommand = new SelectDataElementCommand(firstRoad);
            getProjectGraph()->executeCommand(selectCommand);

            paramList.first()->setActive(true);
            settingsApplyBox_->setApplyButtonVisible(false);
        }
    }
    else if (toolId == ODD::TTE_ROAD_SNAP)
    {
        RSystemElementRoad *firstRoad = NULL;
        RSystemElementRoad *secondRoad = NULL;
        QList<ToolParameter *> paramList;

        getToolObjectSelection(&paramList, &firstRoad, &secondRoad);

        if (firstRoad && secondRoad)
        {
            // Find closest positions of the two roads
            double lineLength[4];

            lineLength[0] = QVector2D(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(0.0)).length(); // Start Start
            lineLength[1] = QVector2D(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(0.0)).length(); // End End
            lineLength[2] = QVector2D(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(secondRoad->getLength())).length(); // End Start
            lineLength[3] = QVector2D(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(secondRoad->getLength())).length(); // Start End

            short int min = 0;

            for (short int k = 1; k < 4; k++)
            {
                if (lineLength[k] < lineLength[min])
                {
                    min = k;
                }
            }
            short int roadPosition[4] = { SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadStart, SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart, SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadEnd, SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd };

            // Validate //
            //
            bool valid = false;
            for (int i = 0; i < 2; i++)
            {
                TrackComponent *track;

                TrackMoveValidator *validationVisitor = new TrackMoveValidator();

                switch (roadPosition[min])
                {
                case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadStart:
                {
                    track = secondRoad->getTrackComponent(0.0);
                    validationVisitor->setState(TrackMoveValidator::STATE_STARTPOINT);
                    validationVisitor->setGlobalDeltaPos(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(0.0));
                }
                break;
                case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart:
                {
 
                    track = secondRoad->getTrackComponent(0.0);
                    validationVisitor->setState(TrackMoveValidator::STATE_STARTPOINT);
                    validationVisitor->setGlobalDeltaPos(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(0.0));
                }
                break;
                case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadEnd:
                {
                    track = secondRoad->getTrackComponent(secondRoad->getLength());
                    validationVisitor->setState(TrackMoveValidator::STATE_ENDPOINT);
                    validationVisitor->setGlobalDeltaPos(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(secondRoad->getLength()));
                }
                break;
                case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd:
                {
                    track = secondRoad->getTrackComponent(secondRoad->getLength());
                    validationVisitor->setState(TrackMoveValidator::STATE_ENDPOINT);
                    validationVisitor->setGlobalDeltaPos(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(secondRoad->getLength()));
                }
                }

                track->accept(validationVisitor);

                TrackSpiralArcSpiral *maybeSparc = dynamic_cast<TrackSpiralArcSpiral *>(track);

                if (!maybeSparc || !validationVisitor->isValid())
                {
                    delete validationVisitor;
                    if (i > 0)
                    {
                        printStatusBarMsg(tr("Validate Move: not valid or tracks are not SpiralArcSpiral"), 0);
                        return; // not valid => no translation
                    }
                    else
                    {
                        RSystemElementRoad *tmpRoad = firstRoad;
                        firstRoad = secondRoad;
                        secondRoad = tmpRoad;
                        if (min == 1)
                        {
                            roadPosition[min] = SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd;
                        }
                        else if (min == 3)
                        {
                            roadPosition[min] = SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart;
                        }
                        continue;
                    }
                }

                switch (roadPosition[min])
                {
                case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadStart:
                {
                    validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
                    validationVisitor->setGlobalHeading(firstRoad->getGlobalHeading(0.0) + 180);
                }
                break;
                case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart:
                {
                    validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
                    validationVisitor->setGlobalHeading(firstRoad->getGlobalHeading(firstRoad->getLength()));
                }
                break;
                case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadEnd:
                {
                    validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
                    validationVisitor->setGlobalHeading(firstRoad->getGlobalHeading(firstRoad->getLength()) + 180);
                }
                break;
                case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd:
                {
                    validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
                    validationVisitor->setGlobalHeading(firstRoad->getGlobalHeading(0.0));
                }
                break;
                }

                track->accept(validationVisitor);

                if (!validationVisitor->isValid())
                {
                    delete validationVisitor;

                    if (i > 0)
                    {
                        printStatusBarMsg(tr("Validate Rotate: not valid"), 0);
                        return; // not valid => no rotation
                    }
                    else
                    {
                        RSystemElementRoad *tmpRoad = firstRoad;
                        firstRoad = secondRoad;
                        secondRoad = tmpRoad;
                        if (min == 1)
                        {
                            roadPosition[min] = SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd;
                        }
                        else if (min == 3)
                        {
                            roadPosition[min] = SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart;
                        }
                        continue;
                    }
                }

                break;
            }


            SnapRoadsCommand *command = new SnapRoadsCommand(firstRoad, secondRoad, roadPosition[min]);
            if (command->isValid())
            {
                paramList.removeFirst();
                clearToolObjectSelection(&paramList, &firstRoad, &secondRoad);
                getProjectGraph()->executeCommand(command);

                SelectDataElementCommand *selectCommand = new SelectDataElementCommand(firstRoad);
                getProjectGraph()->executeCommand(selectCommand);

                paramList.first()->setActive(true);
                settingsApplyBox_->setApplyButtonVisible(false);
            }
        }
    }
}

void
TrackEditor::getToolObjectSelection(QList<ToolParameter *> *parameterList, RSystemElementRoad **firstRoad, RSystemElementRoad **secondRoad)
{

    if (mergeItem_)
    {
        ODD::ToolId toolId = tool_->getToolId();
        ToolValue<RSystemElementRoad> *firstValue = dynamic_cast<ToolValue<RSystemElementRoad> *>(tool_->getParam(toolId, ODD::TPARAM_SELECT));

        *firstRoad = firstValue->getValue();

        parameterList->append(firstValue);
    }

    if (appendItem_)
    {
        ToolValue<RSystemElementRoad> *secondValue = dynamic_cast<ToolValue<RSystemElementRoad> *>(tool_->getParam(ODD::TTE_ROAD_APPEND, ODD::TPARAM_SELECT));
        *secondRoad = secondValue->getValue();
        
        parameterList->append(secondValue);
    }

}

void
TrackEditor::clearToolObjectSelection(QList<ToolParameter *> *parameterList, RSystemElementRoad **firstRoad, RSystemElementRoad **secondRoad)
{
    if (firstRoad == nullptr)
    {
        RSystemElementRoad *road1 = NULL;
        RSystemElementRoad *road2 = NULL;

        QList<ToolParameter *> list;
        getToolObjectSelection(&list, &road1, &road2);

        clearToolObjectSelection(&list, &road1, &road2);
        return;
    }

    if (mergeItem_)
    {
        DeselectDataElementCommand *command = new DeselectDataElementCommand(*firstRoad);
        if (command->isValid())
        {
            getTopviewGraph()->executeCommand(command);
            
            if ((parameterList->first()->getToolId() == ODD::TTE_ROAD_MERGE) || (parameterList->first()->getToolId() == ODD::TTE_ROAD_SNAP))
            {
                mergeItem_ = NULL;
            }
        }
    }

    if (appendItem_)
    {
        DeselectDataElementCommand *command = new DeselectDataElementCommand(*secondRoad);
        if (command->isValid())
        {
            getTopviewGraph()->executeCommand(command);
            appendItem_ = NULL;
        }
    }

    resetToolValues(*parameterList);
}


void
TrackEditor::reset()
{
    ODD::ToolId toolId = tool_->getToolId();
    clearToolObjectSelection();
    delToolParameters();
}

void
TrackEditor::reject()
{
    ProjectEditor::reject();

    if (tool_)
    {
        clearToolObjectSelection();
        delToolParameters();

        ODD::mainWindow()->showParameterDialog(false);
    }
}
