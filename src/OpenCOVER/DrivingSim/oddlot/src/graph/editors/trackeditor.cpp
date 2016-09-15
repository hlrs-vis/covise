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
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

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

// Tools //
//
#include "src/gui/tools/trackeditortool.hpp"
#include "src/gui/mouseaction.hpp"
#include "src/gui/keyaction.hpp"

// Visitor //
//
#include "src/graph/visitors/roadmarkvisitor.hpp"
#include "src/data/visitors/trackmovevalidator.hpp"

// Qt //
//
#include <QGraphicsItem>
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
    , addRoadSystemHandle_(NULL)
    , currentRoadPrototype_(NULL)
    , currentRoadSystemPrototype_(NULL)
    , state_(TrackEditor::STE_NONE)
    , sectionHandle_(NULL)
{
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

/*! \brief Handles the ToolActions.
*
*/
void
TrackEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ODD::ToolId lastTool = getCurrentTool();
    ProjectEditor::toolAction(toolAction);

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

        // Select Tool //
        //
        if (getCurrentTool() == ODD::TTE_SELECT)
        {
        }
        
        if (getCurrentTool() == ODD::TTE_ROAD_MERGE)
        {
            selectedRoads_.clear();
        }
        if (getCurrentTool() == ODD::TTE_ROAD_SNAP)
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
        else if (getCurrentTool() == ODD::TTE_ADD
                 || getCurrentTool() == ODD::TTE_ADD_LINE
                 || getCurrentTool() == ODD::TTE_ADD_CURVE)
        {
            if (trackRoadSystemItem_)
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

    // Prototypes //
    //
    TrackEditorToolAction *trackEditorToolAction = dynamic_cast<TrackEditorToolAction *>(toolAction);
    if (trackEditorToolAction)
    {
        if (isCurrentTool(ODD::TTE_ADD)
            || isCurrentTool(ODD::TTE_ADD_LINE)
            || isCurrentTool(ODD::TTE_ADD_CURVE)
            || isCurrentTool(ODD::TTE_ROAD_NEW))
        {
            delete currentRoadPrototype_;

            currentRoadPrototype_ = new RSystemElementRoad("prototype", "prototype", "-1");

            // Superpose user prototypes //
            //
            foreach (RSystemElementRoad *road, trackEditorToolAction->getPrototypes())
            {
                currentRoadPrototype_->superposePrototype(road);
            }
        }

        if (isCurrentTool(ODD::TTE_ROADSYSTEM_ADD))
        {
            //			if(currentRoadSystemPrototype_ != trackEditorToolAction->getRoadSystemPrototype())
            //			{
            //				delete currentRoadSystemPrototype_; // do not delete this, it's owned by the prototype manager
            currentRoadSystemPrototype_ = trackEditorToolAction->getRoadSystemPrototype();
            //			}
        }

        else if (isCurrentTool(ODD::TTE_TILE_NEW))
        {
            Tile *tile = new Tile("Tile0", "0");
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
    if (getCurrentTool() == ODD::TTE_ADD
        || getCurrentTool() == ODD::TTE_ADD_LINE
        || getCurrentTool() == ODD::TTE_ADD_CURVE)
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
                mouseAction->intercept();

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

                    // Line //
                    //
                    if (getCurrentTool() == ODD::TTE_ADD_LINE)
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
                            printStatusBarMsg(tr("A line can not be inserted here."), 4000);
                            return; // line with this length not possible
                        }

                        // Protoype //
                        //
                        TrackElementLine *line = new TrackElementLine(0.0, 0.0, 0.0, 0.0, length);

                        linePrototype = new RSystemElementRoad("prototype", "prototype", "-1");
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

                    // Curve //
                    //
                    else if (getCurrentTool() == ODD::TTE_ADD_CURVE)
                    {
                        // Declarations //
                        //
                        TrackComponent *track = NULL;
                        RSystemElementRoad *spiralPrototype = NULL;
                        double newHeadingDeg = 0.0;
                        bool foundHandle = false;

                        // Look for existing handles //
                        //
                        foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items(pressPoint_, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
                        {
                            if (!foundHandle)
                            {
                                TrackAddHandle *handle = dynamic_cast<TrackAddHandle *>(item);
                                if (handle)
                                {
                                    pressPoint_ = handle->pos();
                                    newHeadingDeg = handle->rotation();
                                    foundHandle = true;
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

                            if (QVector2D::dotProduct(t, d) <= 0.1 /*NUMERICAL_ZERO8*/) // zero or negative
                            {
                                printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 4000);
                                return; // symmetric curve not possible
                            }

                            QPointF startPoint = track->getGlobalPoint(track->getSStart());
                            double startHeadingDeg = track->getGlobalHeading(track->getSStart());

                            QTransform trafo;
                            trafo.translate(pressPoint_.x(), pressPoint_.y());
                            trafo.rotate(newHeadingDeg);

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
                                    printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 4000);
                                    return;
                                }
                            }

                            spiralPrototype = new RSystemElementRoad("prototype", "prototype", "-1");
                            spiralPrototype->addTrackComponent(spiral);
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

                            if (QVector2D::dotProduct(t, d) <= 0.1 /*NUMERICAL_ZERO8*/) // zero or negative
                            {
                                printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 4000);
                                return; // symmetric curve not possible
                            }

                            QPointF startPoint = track->getGlobalPoint(track->getSEnd());
                            double startHeadingDeg = track->getGlobalHeading(track->getSEnd());

                            QTransform trafo;
                            trafo.translate(startPoint.x(), startPoint.y());
                            trafo.rotate(startHeadingDeg);

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
                                    printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 4000);
                                    return;
                                }
                            }

                            spiralPrototype = new RSystemElementRoad("prototype", "prototype", "-1");
                            spiralPrototype->addTrackComponent(spiral);
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

                    // Prototypes //
                    //
                    else if (getCurrentTool() == ODD::TTE_ADD)
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

    else if ((getCurrentTool() == ODD::TTE_ROAD_MERGE) || (getCurrentTool() == ODD::TTE_ROAD_SNAP))
    {
        if (selectedRoads_.size() == 0)
        {
            // do nothing
        }
        else if (selectedRoads_.size() == 1)
        {
            if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
            {

                // Intercept mouse action //
                //
                mouseAction->intercept();

                // Abort //
                //
                if (mouseAction->getEvent()->button() == Qt::RightButton)
                {
                    selectedRoads_.clear(); // deselect
                }

                // Add //
                //
                else if (mouseAction->getEvent()->button() == Qt::LeftButton)
                {
                    pressPoint_ = mouseAction->getEvent()->scenePos();

                    RSystemElementRoad *firstRoad = selectedRoads_.first();
                    RSystemElementRoad *secondRoad = NULL;
                    selectedRoads_.clear();

                    foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items(pressPoint_, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
                    {
                        RoadMarkItem *maybeRoad = dynamic_cast<RoadMarkItem *>(item);
                        if (maybeRoad)
                        {
                            secondRoad = maybeRoad->getParentRoad();
                            if(secondRoad != firstRoad)
                                break;
                            else
                                secondRoad = NULL;
                        }
                        TrackElementItem *maybeTrack = dynamic_cast<TrackElementItem *>(item);
                        if (maybeTrack)
                        {
                            secondRoad = maybeTrack->getParentTrackRoadItem()->getRoad();
                            if(secondRoad != firstRoad)
                                break;
                            else
                                secondRoad = NULL;
                        }
                    }

                    if (secondRoad)
                    {
                        if (getCurrentTool() == ODD::TTE_ROAD_SNAP)
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
                                        qDebug() << "Validate Move: not valid or tracks are not SpiralArcSpiral";
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
                                        qDebug() << "Validate Rotate: not valid";
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
                            getProjectGraph()->executeCommand(command);
                        }
                        else
                        {
                            // Find closest positions of the two roads
                            double distances[4];

                            distances[0] = QVector2D(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(0.0)).length(); // Start Start
                            distances[1] = QVector2D(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(0.0)).length(); // End End
                            distances[2] = QVector2D(firstRoad->getGlobalPoint(firstRoad->getLength()) - secondRoad->getGlobalPoint(secondRoad->getLength())).length(); // End Start
                            distances[3] = QVector2D(firstRoad->getGlobalPoint(0.0) - secondRoad->getGlobalPoint(secondRoad->getLength())).length(); // Start End
                            MergeRoadsCommand *command=NULL;
                            if(distances[0] < distances[1] && distances[0] < distances[2] && distances[0] < distances[3])
                                command = new MergeRoadsCommand(firstRoad, secondRoad, true,true);
                            if(distances[1] < distances[0] && distances[1] < distances[2] && distances[1] < distances[3])
                                command = new MergeRoadsCommand(firstRoad, secondRoad, false,true);
                            if(distances[2] < distances[0] && distances[2] < distances[1] && distances[2] < distances[3])
                                command = new MergeRoadsCommand(firstRoad, secondRoad, false,false);
                            if(distances[3] < distances[0] && distances[3] < distances[1] && distances[3] < distances[2])
                                command = new MergeRoadsCommand(firstRoad, secondRoad, true,false);
                            if(command)
                            {
                                getProjectGraph()->executeCommand(command);
                            }
                        }

                        selectedRoads_.clear();
                    }
                }
            }
        }
    }

    // NEW //
    //
    else if (getCurrentTool() == ODD::TTE_ROAD_NEW)
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
                state_ = TrackEditor::STE_NEW_PRESSED;
                newRoadLineItem_->setLine(QLineF(pressPoint_, mousePoint));
                newRoadLineItem_->show();
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (state_ == TrackEditor::STE_NEW_PRESSED)
            {
                newRoadLineItem_->setLine(QLineF(pressPoint_, mousePoint));
                printStatusBarMsg(QString("New road: (%1, %2) to (%3, %4). Length: %5.").arg(pressPoint_.x()).arg(pressPoint_.y()).arg(mousePoint.x()).arg(mousePoint.y()).arg(length), 4000);
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
                    printStatusBarMsg("New road: to short. Please click and drag.", 4000);
                }
                else
                {
                    if (currentRoadPrototype_)
                    {
                        // Road //
                        //
                        RSystemElementRoad *newRoad = new RSystemElementRoad("unnamed", "", "-1");

                        // Track //
                        //
                        TrackElementLine *line = new TrackElementLine(pressPoint_.x(), pressPoint_.y(), atan2(mouseLine.y(), mouseLine.x()) * 360.0 / (2.0 * M_PI), 0.0, length);
                        newRoad->addTrackComponent(line);

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
                            printStatusBarMsg(command->text(), 4000);
                            delete command;
                            return; // usually not the case, only if road or prototype are NULL
                        }
                    }
                    else
                    {
                        printStatusBarMsg("New road: Please reselect a Prototype.", 8000);
                    }
                }
            }
        }
    }

    // DELETE //
    //
    else if (getCurrentTool() == ODD::TTE_DELETE)
    {
        //			qDebug("TODO: TrackEditor: DELETE");
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
                printStatusBarMsg(QString("Add Prototype at (%1,%2), angle %3").arg(addRoadSystemHandle_->getPos().x()).arg(addRoadSystemHandle_->getPos().y()).arg(addRoadSystemHandle_->getAngle()), 4000);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (state_ == TrackEditor::STE_ROADSYSTEM_ADD)
            {
                addRoadSystemHandle_->setMousePos(mouseEvent->scenePos());
                printStatusBarMsg(QString("Add Prototype at (%1,%2), angle %3").arg(addRoadSystemHandle_->getPos().x()).arg(addRoadSystemHandle_->getPos().y()).arg(addRoadSystemHandle_->getAngle()), 4000);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
            if (state_ == TrackEditor::STE_ROADSYSTEM_ADD)
            {
                state_ = TrackEditor::STE_NONE;
                addRoadSystemHandle_->setVisible(false);

                AddRoadSystemPrototypeCommand *command = new AddRoadSystemPrototypeCommand(getProjectData()->getRoadSystem(), currentRoadSystemPrototype_, addRoadSystemHandle_->getPos(), addRoadSystemHandle_->getAngle());
                //				AddRoadSystemPrototypeCommand * command = new AddRoadSystemPrototypeCommand(getProjectData()->getRoadSystem(), currentRoadSystemPrototype_, QPointF(0.0, 0.0), 0.0);
                if (command->isValid())
                {
                    getProjectData()->getUndoStack()->push(command);
                }
                else
                {
                    printStatusBarMsg(command->text(), 4000);
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
    foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
    {
        TrackMoveHandle *handle = dynamic_cast<TrackMoveHandle *>(item);
        if (handle)
        {
            mousePos = handle->pos();
            //			newHeadingDeg = handle->rotation();
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
        foreach (TrackMoveHandle *moveHandle, selectedTrackMoveHandles_.values(1))
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
    //	QPointF targetPos = pressPos + dPos;
    //qDebug() << "dPos2: " << dPos;

    // 2: Two degrees of freedom //
    //
    TrackMoveValidator *validationVisitor = new TrackMoveValidator();
    validationVisitor->setGlobalDeltaPos(dPos);

    foreach (TrackMoveHandle *moveHandle, selectedTrackMoveHandles_)
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
    foreach (TrackMoveHandle *moveHandle, selectedTrackMoveHandles_)
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
		TrackComponent * lowSlot = moveHandle->getLowSlot();
		TrackComponent * highSlot = moveHandle->getHighSlot();
		if(lowSlot || highSlot)
		{
//			TrackComponentSinglePointCommand * command = new TrackComponentSinglePointCommand(lowSlot, highSlot, targetPos, NULL);
			TrackComponentSinglePointCommand * command = new TrackComponentSinglePointCommand(lowSlot, highSlot, dPos, NULL);
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
    foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
    {
        TrackMoveHandle *handle = dynamic_cast<TrackMoveHandle *>(item);
        if (handle)
        {
            mousePos = handle->pos();
            break;
        }
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
	::registerTrackRotateHandle(TrackRotateHandle * handle)
{
	if(handle->getRotDOF() < 0 || handle->getRotDOF() > 1)
	{
		qDebug("WARNING 1004261417! TrackEditor TrackRotateHandle DOF not in [0,1].");
	}
	selectedTrackRotateHandles_.insert(handle->getRotDOF(), handle);
}

int
	TrackEditor
	::unregisterTrackRotateHandle(TrackRotateHandle * handle)
{
	return selectedTrackRotateHandles_.remove(handle->getRotDOF(), handle);
}



double
	TrackEditor
	::rotateTrackRotateHandles(double dHeading, double globalHeading)
{
	// No entries //
	//
	if(selectedTrackRotateHandles_.size() == 0)
	{
		return dHeading;
	}

	// 0: Check for zero degrees of freedom //
	//
	if(selectedTrackRotateHandles_.count(0) > 0)
	{
		return 0.0; // no rotation
	}


	// 1: Check for one degree of freedom //
	//
	TrackMoveValidator * validationVisitor = new TrackMoveValidator();
	validationVisitor->setGlobalHeading(dHeading + globalHeading);
	foreach(TrackRotateHandle * handle, selectedTrackRotateHandles_)
	{
		if(handle->getHighSlot())
		{
			validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
			handle->getHighSlot()->accept(validationVisitor, false);
		}

		if(handle->getLowSlot())
		{
			validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
			handle->getLowSlot()->accept(validationVisitor, false);
		}
	}

	// Check if rotation is valid //
	//
	if(validationVisitor->isValid())
	{
//		qDebug("valid");
		return dHeading;
	}
	else
	{
//		qDebug("not valid");
		return 0.0; // not valid => no rotation
//		return dHeading;
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
    foreach (RoadMoveHandle *moveHandle, selectedRoadMoveHandles_)
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
    foreach (RoadRotateHandle *handle, selectedRoadRotateHandles_)
    {
        roads.append(handle->getRoad());
    }

    RotateRoadAroundPointCommand *command = new RotateRoadAroundPointCommand(roads, pivotPoint, angleDegrees, NULL);
    getProjectGraph()->executeCommand(command);

    return true;
}

//###################//
// ROADS			//
//##################//

void
TrackEditor::registerRoad(RSystemElementRoad *road)
{
    if (!selectedRoads_.contains(road))
    {
        selectedRoads_.append(road);
    }
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
        trackRoadSystemItem_ = new TrackRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
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
    //	trackRoadSystemItem_->deleteHandles();
    //	getTopviewGraph()->getScene()->removeItem(trackRoadSystemItem_);
    //	topviewGraph_->graphScene()->removeItem(trackRoadSystemItem_);
    delete trackRoadSystemItem_;
    trackRoadSystemItem_ = NULL;

    // ToolHandles //
    //
    selectedTrackMoveHandles_.clear();
    selectedTrackAddHandles_.clear();
}
