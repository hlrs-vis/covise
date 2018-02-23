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

#include "junctioneditor.hpp"

#include "src/mainwindow.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/roadlink.hpp"

#include "src/data/commands/trackcommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
#include "src/data/commands/junctioncommands.hpp"
#include "src/data/commands/elevationsectioncommands.hpp"
#include "src/data/commands/lanesectioncommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/junction/junctionroadsystemitem.hpp"
#include "src/graph/items/roadsystem/junction/junctionmovehandle.hpp"
#include "src/graph/items/roadsystem/junction/junctionaddhandle.hpp"
#include "src/graph/items/roadsystem/junction/junctionlanewidthmovehandle.hpp"
#include "src/graph/items/roadsystem/junction/junctionlaneroadsystemitem.hpp"
#include "src/graph/items/roadsystem/junction/junctionroaditem.hpp"
#include "src/graph/items/roadsystem/junction/junctionlaneitem.hpp"
#include "src/graph/items/roadsystem/junction/junctionlanesectionitem.hpp"
#include "src/graph/items/roadsystem/junctionitem.hpp"

#include "src/graph/items/handles/circularrotatehandle.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

// Tools //
//
#include "src/gui/tools/junctioneditortool.hpp"
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
JunctionEditor::JunctionEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , junctionRoadSystemItem_(NULL)
    , laneRoadSystemItem_(NULL)
    , pressPoint_(0.0, 0.0)
    , newRoadLineItem_(NULL)
    , addRoadSystemHandle_(NULL)
    , state_(JunctionEditor::STE_NONE)
    , sectionHandle_(NULL)
    , junction_(NULL)
{
}

JunctionEditor::~JunctionEditor()
{
    kill();
}

SectionHandle *
JunctionEditor::getSectionHandle() const
{
    if (!sectionHandle_)
    {
        qDebug("ERROR 1010141039! JunctionEditor not yet initialized.");
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
JunctionEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ODD::ToolId lastTool = getCurrentTool();
    ProjectEditor::toolAction(toolAction);

    if (getCurrentTool() == ODD::TJE_ADD_TO_JUNCTION)
    {
        if (!junction_)
        {
            printStatusBarMsg(tr("First a junction has to be selected."), 4000);
            return;
        }

        QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
        if (selectedItems.size() > 1)
        {
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Add to junction"));
        }

        foreach (QGraphicsItem *item, selectedItems)
        {
            JunctionLaneItem *laneItem = dynamic_cast<JunctionLaneItem *>(item);
            if (laneItem)
            {
                RSystemElementRoad *road = laneItem->getLane()->getParentLaneSection()->getParentRoad();
                AddToJunctionCommand *command = new AddToJunctionCommand(getProjectData()->getRoadSystem(), road, junction_);
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
        if (selectedItems.size() > 1)
        {
            getProjectData()->getUndoStack()->endMacro();
        }
    }

    // Select Tool //
    //
    else if (getCurrentTool() == ODD::TJE_SELECT)
    {
    }

    else if (getCurrentTool() == ODD::TJE_THRESHOLD)
    {
        JunctionEditorToolAction *action = dynamic_cast<JunctionEditorToolAction *>(toolAction);
        if (action)
        {
            threshold_ = action->getThreshold();
        }
    }

    // Change Tool //
    //
    if (lastTool != getCurrentTool())
    {
        // State //
        //
        state_ = JunctionEditor::STE_NONE;

        if (getCurrentTool() == ODD::TJE_LINK_ROADS)
        {

            // Add all selected roads to the junction //
            //
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
            QList<RSystemElementRoad *> selectedRoads;
            foreach (QGraphicsItem *item, selectedItems)
            {
                JunctionLaneItem *laneItem = dynamic_cast<JunctionLaneItem *>(item);
                if (laneItem)
                {
                    RSystemElementRoad *road = laneItem->getParentJunctionLaneSectionItem()->getLaneSection()->getParentRoad();
                    if (!selectedRoads.contains(road))
                    {
                        selectedRoads.append(road);
                    }
                }
            }

            SetRoadLinkRoadsCommand *command = new SetRoadLinkRoadsCommand(selectedRoads, 10);
            getProjectGraph()->executeCommand(command);
        }

        else if (getCurrentTool() == ODD::TJE_UNLINK_ROADS)
        {

            // Remove all selected road links //
            //
            // Macro Command //
            //
            int numberSelectedItems = getTopviewGraph()->getScene()->selectedItems().size();
            if (numberSelectedItems > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Unlink Roads"));
            }

            bool deletedSomething = false;
            do
            {
                deletedSomething = false;
                QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

                foreach (QGraphicsItem *item, selectedItems)
                {
                    JunctionLaneItem *laneItem = dynamic_cast<JunctionLaneItem *>(item);
                    if (laneItem)
                    {
                        RSystemElementRoad *road = laneItem->getLane()->getParentLaneSection()->getParentRoad();
                        RemoveRoadLinkCommand *command = new RemoveRoadLinkCommand(road, NULL);
                        if (getProjectGraph()->executeCommand(command))
                        {
                            deletedSomething = true;
                        }
                        laneItem->setSelected(false);

                        break;
                    }
                }

            } while (deletedSomething);

            // Macro Command //
            //
            if (numberSelectedItems > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }
        }

        // Move Tool //
        //
        else if (getCurrentTool() == ODD::TJE_MOVE)
        {
            if (junctionRoadSystemItem_)
            {
                junctionRoadSystemItem_->rebuildMoveHandles();
            }
        }

        // Create Junction //
        //
        else if (getCurrentTool() == ODD::TJE_CREATE_JUNCTION)
        {
            // This is one undo //
            //
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Create Junction"));

            // Create new Junction Element //
            //
			odrID ID;
			ID.setType(odrID::ID_Junction);
            RSystemElementJunction *newJunction = new RSystemElementJunction("junction");

            NewJunctionCommand *command = new NewJunctionCommand(newJunction, getProjectData()->getRoadSystem(), NULL);
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
            junction_ = newJunction;
            junction_->setElementSelected(true);

            // Add all selected roads to the junction //
            //
            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
            foreach (QGraphicsItem *item, selectedItems)
            {
                JunctionLaneItem *laneItem = dynamic_cast<JunctionLaneItem *>(item);
                if (laneItem)
                {
                    RSystemElementRoad *road = laneItem->getParentJunctionLaneSectionItem()->getParentRoadItem()->getRoad();
                    AddToJunctionCommand *command = new AddToJunctionCommand(getProjectData()->getRoadSystem(), road, junction_);
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

            // End of undo macro //
            //
            getProjectData()->getUndoStack()->endMacro();
        }

        else if (getCurrentTool() == ODD::TJE_REMOVE_FROM_JUNCTION)
        {
            if (!junction_)
            {
                printStatusBarMsg("No junction selected", 4000);
                return;
            }

            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Remove from Junction"));

            QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
            foreach (QGraphicsItem *item, selectedItems)
            {
                JunctionLaneItem *laneItem = dynamic_cast<JunctionLaneItem *>(item);
                if (laneItem)
                {
                    RSystemElementRoad *road = laneItem->getLane()->getParentLaneSection()->getParentRoad();

                    // Remove all links
                    //
                    RemoveRoadLinkCommand *command = new RemoveRoadLinkCommand(road);
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

                    // Set Junction id -1
                    //
                    RemoveFromJunctionCommand *junctionRemoveCommand = new RemoveFromJunctionCommand(junction_, road);
                    if (junctionRemoveCommand->isValid())
                    {
                        getProjectData()->getUndoStack()->push(junctionRemoveCommand);
                    }
                    else
                    {
                        printStatusBarMsg(command->text(), 4000);
                        delete junctionRemoveCommand;
                        return; // usually not the case, only if road or prototype are NULL
                    }
                }
            }
            getProjectData()->getUndoStack()->endMacro();
        }

        // Tool Without Handles //
        //
        else
        {
            if (junctionRoadSystemItem_)
            {
                junctionRoadSystemItem_->deleteHandles();
            }
        }

        // Garbage disposal //
        //
        getTopviewGraph()->garbageDisposal();
    }

    // Prototypes //
    //
    /*	JunctionEditorToolAction * junctionEditorToolAction = dynamic_cast<JunctionEditorToolAction *>(toolAction);
	if(junctionEditorToolAction)
	{
		if(isCurrentTool(ODD::TJE_CREATE_JUNCTION))
		{

			RSystemElementRoad * currentRoadPrototype_ = new RSystemElementRoad("prototype", odrID::invalidID(), odrID::invalidID());

			// Superpose user prototypes //
			//

			//	currentRoadPrototype_->superposePrototype(road);
		}

	}*/
}

// Calculates the offset of a lane from the center of the road.
//
double
JunctionEditor::widthOffset(RSystemElementRoad *road, Lane *lane, LaneSection *laneSection, double s, bool addOwnLaneWidth)
{
    double offset = road->getLaneOffset(s);
    Lane *nextLane;

    double sSection = s - laneSection->getSStart();
    if (addOwnLaneWidth)
    {
        offset += lane->getWidth(sSection);
    }

    int i = 0;
    if (lane->getId() < 0)
    {
        while ((nextLane = laneSection->getNextLower(i)) != lane)
        {
            offset += nextLane->getWidth(sSection);
            i = nextLane->getId();
        }
    }
    else
    {
        while ( (nextLane = laneSection->getNextUpper(i)) != lane)
        {
			if (nextLane)
			{
				offset += nextLane->getWidth(sSection);
				i = nextLane->getId();
			}
        }
        return -offset;
    }

    return offset;
}

void
JunctionEditor::createRoad(QList<Lane *> lanes)
{
    Lane *lane1 = lanes.first();
    Lane *lane2 = lanes.last();

    LaneSection *laneSection1 = lane1->getParentLaneSection();
    LaneSection *laneSection2 = lane2->getParentLaneSection();

    RSystemElementRoad *road1 = laneSection1->getParentRoad();
    RSystemElementRoad *road2 = laneSection2->getParentRoad();

    TrackComponent *track = NULL;
    RSystemElementRoad *spiralPrototype = NULL;
    double newHeadingDeg = 0.0;
    bool foundHandle = false;

    QPointF road1Start = road1->getGlobalPoint(laneSection1->getSStart());
    QPointF road1End = road1->getGlobalPoint(laneSection1->getSEnd());
    QPointF road2Start = road2->getGlobalPoint(laneSection2->getSStart());
    QPointF road2End = road2->getGlobalPoint(laneSection2->getSEnd());

    // Find closest positions of the two roads
    double lineLength[4];

    lineLength[0] = QVector2D(road1Start - road2Start).length(); // Start Start
    lineLength[1] = QVector2D(road1End - road2End).length(); // End End
    lineLength[2] = QVector2D(road1End - road2Start).length(); // End Start
    lineLength[3] = QVector2D(road1Start - road2End).length(); // Start End

    short int min = 0;

    for (short int k = 1; k < 4; k++)
    {
        if (lineLength[k] < lineLength[min])
        {
            min = k;
        }
    }

    double a, b;
    bool addOwnLaneWidth = false;

    // Group undo commands
    //
    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Connect Lanes"));

    switch (min)
    {
    case 0:
    {
        if (((lane1->getId() < 0) && (lane2->getId() < 0)) || ((lane1->getId() > 0) && (lane2->getId() > 0)))
        {
            addOwnLaneWidth = true;
        }
        laneSection1 = road1->getLaneSection(0.0);
        laneSection2 = road2->getLaneSection(0.0);
        // Calculate the width of the adjacent lanes for the offset of the spline
        double offset1 = widthOffset(road1, lane1, laneSection1, laneSection1->getSStart(), false);
        double offset2 = widthOffset(road2, lane2, laneSection2, laneSection2->getSStart(), addOwnLaneWidth);
        spiralPrototype = createSpiral(road1, road2, true, true, offset1, offset2);

        if (!spiralPrototype)
        {
            return;
        }
        a = lane2->getWidth(0.0);
        b = (lane1->getWidth(0.0) - a) / spiralPrototype->getLength();
    }
    break;
    case 1:
    {
        if (((lane1->getId() < 0) && (lane2->getId() < 0)) || ((lane1->getId() > 0) && (lane2->getId() > 0)))
        {
            addOwnLaneWidth = true;
        }
        laneSection1 = road1->getLaneSection(road1->getLength());
        laneSection2 = road2->getLaneSection(road2->getLength());
        // Calculate the width of the adjacent lanes for the offset of the spline
        double offset1 = widthOffset(road1, lane1, laneSection1, laneSection1->getSEnd(), false);
        double offset2 = widthOffset(road2, lane2, laneSection2, laneSection2->getSEnd(), addOwnLaneWidth);
        spiralPrototype = createSpiral(road1, road2, false, false, offset1, offset2);

        if (!spiralPrototype)
        {
            getProjectData()->getUndoStack()->endMacro();
            return;
        }
        a = lane1->getWidth(laneSection1->getLength());
        b = (lane2->getWidth(laneSection2->getLength()) - a) / spiralPrototype->getLength();
    }
    break;
    case 2:
    {
        if (((lane1->getId() < 0) && (lane2->getId() > 0)) || ((lane1->getId() > 0) && (lane2->getId() < 0)))
        {
            addOwnLaneWidth = true;
        }
        laneSection1 = road1->getLaneSection(road1->getLength());
        laneSection2 = road2->getLaneSection(0.0);
        // Calculate the width of the adjacent lanes for the offset of the spline
        double offset1 = widthOffset(road1, lane1, laneSection1, laneSection1->getSEnd(), false);
        double offset2 = widthOffset(road2, lane2, laneSection2, laneSection2->getSStart(), addOwnLaneWidth);
        spiralPrototype = createSpiral(road1, road2, false, true, offset1, offset2);

        if (!spiralPrototype)
        {
            getProjectData()->getUndoStack()->endMacro();
            return;
        }

        a = lane1->getWidth(laneSection1->getLength());
        b = (lane2->getWidth(0.0) - a) / spiralPrototype->getLength();
    }
    break;

    case 3:
    {
        if (((lane1->getId() < 0) && (lane2->getId() > 0)) || ((lane1->getId() > 0) && (lane2->getId() < 0)))
        {
            addOwnLaneWidth = true;
        }
        laneSection1 = road1->getLaneSection(0.0);
        laneSection2 = road2->getLaneSection(road2->getLength());
        // Calculate the width of the adjacent lanes for the offset of the spline
        double offset1 = widthOffset(road1, lane1, laneSection1, laneSection1->getSStart(), false);
        double offset2 = widthOffset(road2, lane2, laneSection2, laneSection2->getSEnd(), addOwnLaneWidth);
        spiralPrototype = createSpiral(road1, road2, true, false, offset1, offset2);

        if (!spiralPrototype)
        {
            getProjectData()->getUndoStack()->endMacro();
            return;
        }
        a = lane2->getWidth(laneSection2->getLength());
        b = (lane1->getWidth(0.0) - a) / spiralPrototype->getLength();
    }
    break;

    default:
    {
        getProjectData()->getUndoStack()->endMacro();
        return;
    }
    }

    // Append Prototype //
    //
    // Lookup a simple prototype - just the central border
    //
    RSystemElementRoad *simplePrototype;
    QList<PrototypeContainer<RSystemElementRoad *> *> laneSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype);

    for (int i = 0; i < laneSectionPrototypes.size(); i++)
    {
        if (laneSectionPrototypes.at(i)->getPrototypeName() == "SimplePrototype")
        {
            simplePrototype = laneSectionPrototypes.at(i)->getPrototype();
        }
    }

    if (simplePrototype)
    {
        spiralPrototype->superposePrototype(simplePrototype);
        LaneSection *newLaneSection = spiralPrototype->getLaneSection(0.0);
        Lane *newLane;
        if (lane1->getId() < 0)
        {
            newLane = new Lane(-1, lane1->getLaneType());
        }
        else
        {
            newLane = new Lane(1, lane1->getLaneType());
        }

        LaneWidth *width = new LaneWidth(0.0, a, b, 0.0, 0.0);
        newLane->addWidthEntry(width);
        newLaneSection->addLane(newLane);
    }

    //AppendRoadPrototypeCommand * command = new AppendRoadPrototypeCommand(road1, spiralPrototype, isStart, NULL);
    NewRoadCommand *command = new NewRoadCommand(spiralPrototype, getProjectData()->getRoadSystem(), NULL);
    getProjectGraph()->executeCommand(command);

    getProjectData()->getUndoStack()->endMacro();
}

void
JunctionEditor::createRoad(QList<RSystemElementRoad *> roads)
{

    TrackComponent *track = NULL;
    RSystemElementRoad *spiralPrototype = NULL;
    double newHeadingDeg = 0.0;
    bool foundHandle = false;

    QPointF road1Start = roads.first()->getGlobalPoint(0.0);
    QPointF road1End = roads.first()->getGlobalPoint(roads.first()->getLength());
    QPointF road2Start = roads.last()->getGlobalPoint(0.0);
    QPointF road2End = roads.last()->getGlobalPoint(roads.last()->getLength());

    // Find closest positions of the two roads
    double lineLength[4];

    lineLength[0] = QVector2D(road1Start - road2Start).length(); // Start Start
    lineLength[1] = QVector2D(road1End - road2End).length(); // End End
    lineLength[2] = QVector2D(road1End - road2Start).length(); // End Start
    lineLength[3] = QVector2D(road1Start - road2End).length(); // Start End

    short int min = 0;

    for (short int k = 1; k < 4; k++)
    {
        if (lineLength[k] < lineLength[min])
        {
            min = k;
        }
    }

    switch (min)
    {
    case 0:
    {
        spiralPrototype = createSpiral(roads.first(), roads.last(), true, true);
    }
    break;
    case 1:
    {
        spiralPrototype = createSpiral(roads.first(), roads.last(), false, false);
    }
    break;
    case 2:
    {
        spiralPrototype = createSpiral(roads.first(), roads.last(), false, true);
    }
    break;

    case 3:
    {
        spiralPrototype = createSpiral(roads.first(), roads.last(), true, false);
    }
    break;
    }

    if (!spiralPrototype)
    {
        return;
    }

    // Create the lanes, interpolating the lane width
    //
    // Lookup a simple prototype - just the central border
    //
    RSystemElementRoad *simplePrototype;
    QList<PrototypeContainer<RSystemElementRoad *> *> laneSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype);

    for (int i = 0; i < laneSectionPrototypes.size(); i++)
    {
        if (laneSectionPrototypes.at(i)->getPrototypeName() == "SimplePrototype")
        {
            simplePrototype = laneSectionPrototypes.at(i)->getPrototype();
        }
    }

    // Test if the new road starts near road1 or near road2
    //
    double dist1, dist2, paramStart, paramEnd;
    LaneSection *prototypeLaneSectionStart, *prototypeLaneSectionEnd;
    int laneIDStart;

    if ((min = 0) || (min == 3))
    {
        dist1 = QVector2D(roads.first()->getGlobalPoint(0.0) - spiralPrototype->getGlobalPoint(0.0)).length();
    }
    else
    {
        dist1 = QVector2D(roads.first()->getGlobalPoint(roads.first()->getLength()) - spiralPrototype->getGlobalPoint(0.0)).length();
    }

    if ((min = 0) || (min == 2))
    {
        dist2 = QVector2D(roads.last()->getGlobalPoint(0.0) - spiralPrototype->getGlobalPoint(0.0)).length();
    }
    else
    {
        dist2 = QVector2D(roads.last()->getGlobalPoint(roads.last()->getLength()) - spiralPrototype->getGlobalPoint(0.0)).length();
    }

    if (dist1 < dist2)
    {
        if ((min = 0) || (min == 3))
        {
            laneIDStart = -1;
            paramStart = 0.0;
            prototypeLaneSectionStart = roads.first()->getLaneSection(0.0);
        }
        else
        {
            laneIDStart = 1;
            paramStart = roads.first()->getLength();
            prototypeLaneSectionStart = roads.first()->getLaneSection(roads.first()->getLength());
        }
        if ((min = 0) || (min == 2))
        {
            paramEnd = 0.0;
            prototypeLaneSectionEnd = roads.last()->getLaneSection(0.0);
        }
        else
        {
            prototypeLaneSectionEnd = roads.last()->getLaneSection(roads.last()->getLength());
            paramEnd = prototypeLaneSectionEnd->getLength();
        }
    }
    else
    {
        if ((min = 0) || (min == 3))
        {
            paramEnd = 0.0;
            prototypeLaneSectionEnd = roads.first()->getLaneSection(0.0);
        }
        else
        {
            prototypeLaneSectionEnd = roads.first()->getLaneSection(roads.first()->getLength());
            paramEnd = prototypeLaneSectionEnd->getLength();
        }
        if ((min = 0) || (min == 2))
        {
            laneIDStart = -1;
            paramStart = 0.0;
            prototypeLaneSectionStart = roads.last()->getLaneSection(0.0);
        }
        else
        {
            laneIDStart = 1;
            prototypeLaneSectionStart = roads.last()->getLaneSection(roads.last()->getLength());
            paramStart = prototypeLaneSectionStart->getLength();
        }
    }

    // Copy the lanes and interpolate the lane width
    //
    QMap<int, Lane *> endLanes = prototypeLaneSectionEnd->getLanes();
    QMap<int, Lane *> startLanes = prototypeLaneSectionStart->getLanes();
    QMap<int, Lane *>::const_iterator startLanesIterator = startLanes.constBegin();
    LaneSection *newLaneSection;
    if (startLanes.value(0) && simplePrototype) // The central border is not a lane
    {
        spiralPrototype->superposePrototype(simplePrototype);
        newLaneSection = spiralPrototype->getLaneSection(0.0);
    }
    else
    {
        newLaneSection = new LaneSection(0.0, false);
        spiralPrototype->addLaneSection(newLaneSection);
    }

    while (startLanesIterator != prototypeLaneSectionStart->getLanes().constEnd())
    {
        Lane *startLane = startLanesIterator.value();
        Lane *newLane = new Lane(startLane->getId() * laneIDStart, startLane->getLaneType());

        LaneRoadMark *roadMark = startLane->getRoadMarkEntry(0.0);
        if (roadMark)
        {
            LaneRoadMark *newRoadMark = new LaneRoadMark(0.0, roadMark->getRoadMarkType(), roadMark->getRoadMarkWeight(), roadMark->getRoadMarkColor(), 0.12);
            newLane->addRoadMarkEntry(newRoadMark);
        }

        if (startLane->getLaneType() != Lane::LT_BORDER)
        {
            Lane *endLane;
            if ((min == 0) || (min == 1))
            {
                endLane = endLanes.value(-startLane->getId());
            }
            else
            {
                endLane = endLanes.value(startLane->getId());
            }

            double a, b;

            a = startLane->getWidth(paramStart);
            b = (endLane->getWidth(paramEnd) - a) / spiralPrototype->getLength();

            LaneWidth *width = new LaneWidth(0.0, a, b, 0.0, 0.0);
            newLane->addWidthEntry(width);
        }

        InsertLaneCommand *command = new InsertLaneCommand(newLaneSection, newLane);
        getProjectGraph()->executeCommand(command);

        startLanesIterator++;
    }

    //AppendRoadPrototypeCommand * command = new AppendRoadPrototypeCommand(road1, spiralPrototype, isStart, NULL);
    NewRoadCommand *command = new NewRoadCommand(spiralPrototype, getProjectData()->getRoadSystem(), NULL);
    getProjectGraph()->executeCommand(command);
}

RSystemElementRoad *
JunctionEditor::createSpiral(RSystemElementRoad *road1, RSystemElementRoad *road2, bool startContact1, bool startContact2, double offset1, double offset2)
{

    double startHeadingDeg, endHeadingDeg;
    QVector2D t, d;
    QPointF spiralStartPoint, startPoint, endPoint;
    double height1, height2;
    RSystemElementRoad *spiralPrototype;
    TrackSpiralArcSpiral *spiral;

    if (startContact2)
    {
        TrackComponent *track = road2->getTrackComponent(0.0);
        QVector2D endVec = QVector2D(track->getGlobalPoint(track->getSStart())) + offset2 * track->getGlobalNormal(track->getSStart());
        endPoint = endVec.toPointF();
        endHeadingDeg = track->getGlobalHeading(track->getSStart()) + 180;

        height2 = road2->getElevationSection(0.0)->getElevation(0.0);
    }
    else
    {
        TrackComponent *track = road2->getTrackComponent(road2->getLength());
        QVector2D endVec = QVector2D(track->getGlobalPoint(track->getSEnd())) + offset2 * track->getGlobalNormal(track->getSEnd());
        endPoint = endVec.toPointF();
        endHeadingDeg = track->getGlobalHeading(track->getSEnd());

        height2 = road2->getElevationSection(road2->getLength())->getElevation(road2->getLength());
    }

    if (startContact1)
    {
        TrackComponent *track = road1->getTrackComponent(0.0);

        QVector2D startVec = QVector2D(track->getGlobalPoint(track->getSStart())) + offset1 * track->getGlobalNormal(track->getSStart());
        startPoint = startVec.toPointF();
        startHeadingDeg = track->getGlobalHeading(track->getSStart());

        t = track->getGlobalTangent(track->getSStart()); // length = 1

        height1 = road1->getElevationSection(0.0)->getElevation(0.0);

        //Calculate Transformation /
        //
        d = QVector2D(startPoint - endPoint).normalized();
        if (QVector2D::dotProduct(t, d) <= 0.1)
        {
            printStatusBarMsg(tr("A curve cannot be inserted here."), 4000);
            return NULL;
        }

        QTransform trafo;
        trafo.translate(endPoint.x(), endPoint.y());
        trafo.rotate(endHeadingDeg);

        //Prototype//
        //
        spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(startPoint), 0.0, startHeadingDeg - endHeadingDeg, 0.5);

        if (!spiral->validParameters())
        {
            delete spiral;

            // Try opposite heading // (neccessary when the found handle points to the other direction)
            //
            trafo.rotate(180);
            spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(startPoint), 0.0, startHeadingDeg - endHeadingDeg + 180, 0.5);
            ;
            if (!spiral->validParameters())
            {
                delete spiral;
                printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 4000);
                return NULL;
            }
        }

        spiralPrototype = new RSystemElementRoad("prototype");
        spiralPrototype->addTrackComponent(spiral);

        // Transform the spiralprototype to endPoint //
        //
        QTransform roadTrafo = road1->getGlobalTransform(0.0, -offset1);
        QTransform protoTrafo = spiralPrototype->getGlobalTransform(spiralPrototype->getLength());
        double protoHeading = spiralPrototype->getGlobalHeading(spiralPrototype->getLength());

        foreach (TrackComponent *track, spiralPrototype->getTrackSections())
        {
            track->setGlobalTranslation(roadTrafo.map(protoTrafo.inverted().map(track->getGlobalPoint(track->getSStart()))));
            track->setGlobalRotation(track->getGlobalHeading(track->getSStart()) - protoHeading + startHeadingDeg);
        }
    }
    else
    {
        TrackComponent *track = road1->getTrackComponent(road1->getLength());

        QVector2D startVec = QVector2D(track->getGlobalPoint(track->getSEnd())) + offset1 * track->getGlobalNormal(track->getSEnd());
        startPoint = startVec.toPointF();
        startHeadingDeg = track->getGlobalHeading(track->getSEnd());

        t = track->getGlobalTangent(track->getSEnd()); // length = 1

        height1 = road1->getElevationSection(road1->getLength())->getElevation(road1->getLength());

        //Calculate Transformation //
        //
        d = QVector2D(endPoint - startPoint).normalized();
        if (QVector2D::dotProduct(t, d) <= 0.1)
        {
            printStatusBarMsg(tr("A curve cannot be inserted here."), 4000);
            return NULL;
        }

        QTransform trafo;
        trafo.translate(startPoint.x(), startPoint.y());
        trafo.rotate(startHeadingDeg);

        //Prototype//
        //
        spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(endPoint), 0.0, endHeadingDeg - startHeadingDeg, 0.5);

        if (!spiral->validParameters())
        {
            delete spiral;

            // Try opposite heading // (neccessary when the found handle points to the other direction)
            //
            spiral = new TrackSpiralArcSpiral(QPointF(0.0, 0.0), trafo.inverted().map(endPoint), 0.0, endHeadingDeg - startHeadingDeg + 180.0, 0.5);
            if (!spiral->validParameters())
            {
                delete spiral;
                printStatusBarMsg(tr("A symmetric curve can not be inserted here."), 4000);
                return NULL;
            }
        }

        spiralPrototype = new RSystemElementRoad("prototype");
        spiralPrototype->addTrackComponent(spiral);

        // Transform the spiralprototype to endPoint //
        //
        QTransform roadTrafo = road1->getGlobalTransform(road1->getLength(), -offset1);
        QTransform protoTrafo = spiralPrototype->getGlobalTransform(0.0);
        double protoHeading = spiralPrototype->getGlobalHeading(0.0);

        foreach (TrackComponent *track, spiralPrototype->getTrackSections())
        {
            track->setGlobalTranslation(roadTrafo.map(protoTrafo.inverted().map(track->getGlobalPoint(track->getSStart()))));
            track->setGlobalRotation(track->getGlobalHeading(track->getSStart()) - protoHeading + startHeadingDeg);
        }
    }

    // Superpose planar ElevationSection //
    //
    QList<PrototypeContainer<RSystemElementRoad *> *> elevationSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_ElevationPrototype);

    for (int i = 0; i < elevationSectionPrototypes.size(); i++)
    {
        if (elevationSectionPrototypes.at(i)->getPrototypeName() == "Planar 0.0")
        {
            spiralPrototype->superposePrototype(elevationSectionPrototypes.at(i)->getPrototype());
            break;
        }
    }

    // Set Elevation //
    //
    ElevationSection *elevationSection = spiralPrototype->getElevationSection(0.0);

    QList<ElevationSection *> startElevationSections;
    QList<ElevationSection *> endElevationSections;
    startElevationSections.append(elevationSection);

    // Command //
    //
    ElevationSetHeightCommand *command = new ElevationSetHeightCommand(endElevationSections, startElevationSections, height2, true, NULL);
    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);

        // Message //
        //
        printStatusBarMsg(QString("setHeight to: %1").arg(height2), 1000);
    }
    else
    {
        if (command->text() != "")
        {
            printStatusBarMsg(command->text(), 4000);
        }
        delete command;
    }

    startElevationSections.clear();
    endElevationSections.append(elevationSection);

    // Command //
    //
    command = new ElevationSetHeightCommand(endElevationSections, startElevationSections, height1, true, NULL);
    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);

        // Message //
        //
        printStatusBarMsg(QString("setHeight to: %1").arg(height1), 1000);
    }
    else
    {
        if (command->text() != "")
        {
            printStatusBarMsg(command->text(), 4000);
        }
        delete command;
    }

	// Superpose planar SuperelevationSection //
	//
	QList<PrototypeContainer<RSystemElementRoad *> *> superelevationSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype);

	for (int i = 0; i < superelevationSectionPrototypes.size(); i++)
	{
		if (superelevationSectionPrototypes.at(i)->getPrototypeName() == "Planar 0.0")
		{
			spiralPrototype->superposePrototype(superelevationSectionPrototypes.at(i)->getPrototype());
			break;
		}
	}

	// Superpose planar CrossfallSection //
	//
	QList<PrototypeContainer<RSystemElementRoad *> *> crossfallSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype);

	for (int i = 0; i < crossfallSectionPrototypes.size(); i++)
	{
		if (crossfallSectionPrototypes.at(i)->getPrototypeName() == "Planar 0.0")
		{
			spiralPrototype->superposePrototype(crossfallSectionPrototypes.at(i)->getPrototype());
			break;
		}
	}


    return spiralPrototype;
}

//################//
// MOUSE & KEY    //
//################//

/*! \brief .
*
*/
void
JunctionEditor::mouseAction(MouseAction *mouseAction)
{

    QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();
    ProjectEditor::mouseAction(mouseAction);

    // NEW //
    //
    if (getCurrentTool() == ODD::TTE_ROAD_NEW)
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
                state_ = JunctionEditor::STE_NEW_PRESSED;
                newRoadLineItem_->setLine(QLineF(pressPoint_, mousePoint));
                newRoadLineItem_->show();
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (state_ == JunctionEditor::STE_NEW_PRESSED)
            {
                newRoadLineItem_->setLine(QLineF(pressPoint_, mousePoint));
                printStatusBarMsg(QString("New road: (%1, %2) to (%3, %4). Length: %5.").arg(pressPoint_.x()).arg(pressPoint_.y()).arg(mousePoint.x()).arg(mousePoint.y()).arg(length), 4000);
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
            if (mouseAction->getEvent()->button() == Qt::LeftButton)
            {
                state_ = JunctionEditor::STE_NONE;
                newRoadLineItem_->hide();

                if (length < 10.0)
                {
                    printStatusBarMsg("New road: to short. Please click and drag.", 4000);
                }
                else
                {
                    //				if(currentRoadPrototype_)
                    {
                        // Road //
                        //
                        /*		RSystemElementRoad * newRoad = new RSystemElementRoad("unnamed");

						// Track //
						//
						TrackElementLine * line = new TrackElementLine(pressPoint_.x(), pressPoint_.y(), atan2(mouseLine.y(), mouseLine.x()) * 360.0/(2.0*M_PI), 0.0, length);
						newRoad->addTrackComponent(line);

						// Append Prototype //
						//
						newRoad->superposePrototype(currentRoadPrototype_);
						NewRoadCommand * command = new NewRoadCommand(newRoad, getProjectData()->getRoadSystem(), NULL);
						if(command->isValid())
						{
							getProjectData()->getUndoStack()->push(command);
						}
						else
						{
							printStatusBarMsg(command->text(), 4000);
							delete command;
							return; // usually not the case, only if road or prototype are NULL
						}*/
                    }
                    //					else
                    {
                        printStatusBarMsg("New road: Please reselect a Prototype.", 8000);
                    }
                }
            }
        }
    }
    else if (getCurrentTool() == ODD::TJE_SELECT_JUNCTION)
    {
        if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {

            if (mouseAction->getEvent()->button() == Qt::LeftButton)
            {

                QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
                foreach (QGraphicsItem *item, selectedItems)
                {
                    JunctionItem *junction = dynamic_cast<JunctionItem *>(item);
                    if (junction)
                    {
                        junction_ = junction->getJunction();
                    }
                }
            }
        }
    }

    // DELETE //
    //
    else if (getCurrentTool() == ODD::TJE_REMOVE_FROM_JUNCTION)
    {
        //			qDebug("TODO: JunctionEditor: DELETE");
    }

    else if (getCurrentTool() == ODD::TJE_CIRCLE)
	{

        if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {

            if (mouseAction->getEvent()->button() == Qt::LeftButton)
            {
                // Find selected roads
                //
                QMap<RSystemElementRoad *, QList<double> > selectedRoads; // keep minimal and maximal lanesection start and end values
                QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

                foreach (QGraphicsItem *item, selectedItems)
                {
                    JunctionLaneItem *laneItem = dynamic_cast<JunctionLaneItem *>(item);
                    if (laneItem)
                    {
                        RSystemElementRoad *road = laneItem->getLane()->getParentLaneSection()->getParentRoad();
                        double start = laneItem->getLane()->getParentLaneSection()->getSStart();
                        double end = laneItem->getLane()->getParentLaneSection()->getSEnd();
                        QList<double> StartEnd;
                        if (!selectedRoads.contains(road))
                        {
                            StartEnd.append(start);
                            StartEnd.append(end);
                            selectedRoads.insert(road, StartEnd);
                        }
                        else
                        {
                            StartEnd = selectedRoads.find(road).value();
                            if (StartEnd.at(0) > start)
                            {
                                StartEnd[0] = start;
                            }

                            if (StartEnd.at(1) < end)
                            {
                                StartEnd[1] = end;
                            }
                        }
                    }
                }

                if (selectedRoads.size() > 0)
                {
                    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Split Track and Road"));

                    // Create new Junction Element //
                    //
                    RSystemElementJunction *newJunction = new RSystemElementJunction("junction",getProjectData()->getRoadSystem()->getID(odrID::ID_Junction));

                    NewJunctionCommand *junctionCommand = new NewJunctionCommand(newJunction, getProjectData()->getRoadSystem(), NULL);
                    if (junctionCommand->isValid())
                    {
                        getProjectData()->getUndoStack()->push(junctionCommand);
                    }
                    else
                    {
                        printStatusBarMsg(junctionCommand->text(), 4000);
                        delete junctionCommand;
                        return; // usually not the case, only if road or prototype are NULL
                    }

                    QList<SetRoadLinkRoadsCommand::RoadPair *> roadPairs; // RoadPairs to link together
                    QMap<RSystemElementRoad *, RSystemElementRoad *> connectedRoads; // Roads are intersected twice, but the connecting road already exists
                    QMap<RSystemElementRoad *, bool> incomingRoads; // List of Roads that have to be connected at start or end point
                    QMap<RSystemElementRoad *, QList<double> >::const_iterator roadIterator = selectedRoads.constBegin();
                    while (roadIterator != selectedRoads.constEnd())
                    {
                        RSystemElementRoad *road = roadIterator.key();
                        RoadSystem *roadSystem = road->getRoadSystem();

                        QPointF cc = getTopviewGraph()->getView()->getCircleCenter(); // CircleCenter

                        // Find intersection points //
                        //
                        // Find the closest point of track //
                        //
                        double minPoint = road->getSFromGlobalPoint(cc, roadIterator.value().at(0), roadIterator.value().at(1));

                        // minPoint is the start point or end point of the road.
                        // In this case the cutting points are approximated
                        //
                        if (minPoint < NUMERICAL_ZERO3)
                        {
                            double interceptPoint = threshold_ - QVector2D(road->getGlobalPoint(0.0) - cc).length();

                            SplitTrackRoadCommand *splitTrackRoadCommand = new SplitTrackRoadCommand(road, interceptPoint, NULL);
                            if (!getProjectGraph()->executeCommand(splitTrackRoadCommand))
                            {
                                roadIterator++;
                                continue;
                            }

                            // Remove road at start point
                            //
                            if (!road->getPredecessor())
                            {
                                RemoveRoadCommand *removeRoadCommand = new RemoveRoadCommand(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad());
                                getProjectGraph()->executeCommand(removeRoadCommand);
                            }
                            else
                            {
                                // The connecting road already exists. Store the road pairs.
                                //
                                SetRoadLinkRoadsCommand::RoadPair roadPair = { splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart };
                                roadPairs.append(&roadPair);
                                connectedRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad());

                                QMap<RSystemElementRoad *, RSystemElementRoad *>::iterator it = connectedRoads.find(road); // The connection with road can be removed from the map, because it is replaced
                                while ((it != connectedRoads.end()) && (it.value() == road))
                                {
                                    if (it.key() == road->getPredecessor()->getParentRoad())
                                    {
                                        connectedRoads.erase(it);
                                        break;
                                    }
                                }
                                connectedRoads.insert(road->getPredecessor()->getParentRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());

                                // add the connecting road to the junction
                                AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), newJunction);
                                if (addToJunctionCommand->isValid())
                                {
                                    getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                }
                                else
                                {
                                    printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                    delete addToJunctionCommand;
                                }
                            }

                            incomingRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), true); // Connect at start

                            roadIterator++;
                            continue;
                        }
                        else if (road->getLength() - minPoint < NUMERICAL_ZERO3)
                        {
                            double interceptPoint = road->getLength() - (threshold_ - QVector2D(road->getGlobalPoint(road->getLength()) - cc).length());
                            SplitTrackRoadCommand *splitTrackRoadCommand = new SplitTrackRoadCommand(road, interceptPoint, NULL);
                            if (!getProjectGraph()->executeCommand(splitTrackRoadCommand))
                            {
                                roadIterator++;
                                continue;
                            }

                            if (!road->getSuccessor())
                            {
                                // Remove road at start point
                                //
                                RemoveRoadCommand *removeRoadCommand = new RemoveRoadCommand(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());
                                getProjectGraph()->executeCommand(removeRoadCommand);
                            }
                            else
                            {
                                // The connecting road already exists. Store the road pairs.
                                //
                                SetRoadLinkRoadsCommand::RoadPair roadPair = { splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd };
                                roadPairs.append(&roadPair);
                                connectedRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());

                                QMap<RSystemElementRoad *, RSystemElementRoad *>::iterator it = connectedRoads.find(road); // The connection with road can be removed from the map, because it is replaced
                                while ((it != connectedRoads.end()) && (it.value() == road))
                                {
                                    if (it.key() == road->getSuccessor()->getParentRoad())
                                    {
                                        connectedRoads.erase(it);
                                        break;
                                    }
                                }
                                connectedRoads.insert(road->getSuccessor()->getParentRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());

                                // add the connecting road to the junction
                                AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), newJunction);
                                if (addToJunctionCommand->isValid())
                                {
                                    getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                }
                                else
                                {
                                    printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                    delete addToJunctionCommand;
                                }
                            }

                            incomingRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), false); //connect at end

                            roadIterator++;
                            continue;
                        }

                        // Normal length
                        //
                        double mindist = QVector2D(road->getGlobalPoint(minPoint) - cc).length();

                        if (mindist < threshold_)
                        {
                            double dist = sqrt(threshold_ * threshold_ - mindist * mindist);

                            double interceptPoint1 = minPoint - dist;
                            double interceptPoint2 = minPoint + dist;

                            if ((interceptPoint1 < 0.0) && (interceptPoint2 > road->getLength())) // road lays within circle
                            {
                                if (!road->getPredecessor() || !road->getSuccessor())
                                {
                                    RemoveRoadCommand *removeRoadCommand = new RemoveRoadCommand(road);
                                    getProjectGraph()->executeCommand(removeRoadCommand);
                                }
                                else
                                {
                                    // The connecting road already exists, predecessor and successor are already there. Add it to the junction.
                                    //

                                    AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), road, newJunction);
                                    if (addToJunctionCommand->isValid())
                                    {
                                        getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                    }
                                    else
                                    {
                                        printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                        delete addToJunctionCommand;
                                    }
                                }
                            }
                            else if (interceptPoint1 < 0.0) // start of road lays within circle
                            {
                                SplitTrackRoadCommand *splitTrackRoadCommand = new SplitTrackRoadCommand(road, interceptPoint2, NULL);
                                if (!getProjectGraph()->executeCommand(splitTrackRoadCommand))
                                {
                                    roadIterator++;
                                    continue;
                                }

                                if (!road->getPredecessor())
                                {
                                    // Remove road at start point
                                    //
                                    RemoveRoadCommand *removeRoadCommand = new RemoveRoadCommand(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad());
                                    getProjectGraph()->executeCommand(removeRoadCommand);
                                }
                                else
                                {
                                    // The connecting road already exists. Store the road pairs.
                                    //
                                    SetRoadLinkRoadsCommand::RoadPair roadPair = { splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart };
                                    roadPairs.append(&roadPair);
                                    connectedRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad());

                                    QMap<RSystemElementRoad *, RSystemElementRoad *>::iterator it = connectedRoads.find(road); // The connection with road can be removed from the map, because it is replaced
                                    while ((it != connectedRoads.end()) && (it.value() == road))
                                    {
                                        if (it.key() == road->getPredecessor()->getParentRoad())
                                        {
                                            connectedRoads.erase(it);
                                            break;
                                        }
                                    }
                                    connectedRoads.insert(road->getPredecessor()->getParentRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());

                                    // add the connecting road to the junction
                                    AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), newJunction);
                                    if (addToJunctionCommand->isValid())
                                    {
                                        getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                    }
                                    else
                                    {
                                        printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                        delete addToJunctionCommand;
                                    }
                                }

                                incomingRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), true); // Connect at start
                            }
                            else if (interceptPoint2 > road->getLength()) // end of road lays within circle
                            {
                                SplitTrackRoadCommand *splitTrackRoadCommand = new SplitTrackRoadCommand(road, interceptPoint1, NULL);
                                if (!getProjectGraph()->executeCommand(splitTrackRoadCommand))
                                {
                                    roadIterator++;
                                    continue;
                                }

                                if (!road->getSuccessor())
                                {
                                    // Remove road at start point
                                    //
                                    RemoveRoadCommand *removeRoadCommand = new RemoveRoadCommand(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());
                                    getProjectGraph()->executeCommand(removeRoadCommand);
                                }
                                else
                                {
                                    // The connecting road already exists. Store the road pairs.
                                    //
                                    SetRoadLinkRoadsCommand::RoadPair roadPair = { splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd };
                                    roadPairs.append(&roadPair);
                                    connectedRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());

                                    QMap<RSystemElementRoad *, RSystemElementRoad *>::iterator it = connectedRoads.find(road); // The connection with road can be removed from the map, because it is replaced
                                    while ((it != connectedRoads.end()) && (it.value() == road))
                                    {
                                        if (it.key() == road->getSuccessor()->getParentRoad())
                                        {
                                            connectedRoads.erase(it);
                                            break;
                                        }
                                    }
                                    connectedRoads.insert(road->getSuccessor()->getParentRoad(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad());

                                    // add the connecting road to the junction
                                    AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), newJunction);
                                    if (addToJunctionCommand->isValid())
                                    {
                                        getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                    }
                                    else
                                    {
                                        printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                        delete addToJunctionCommand;
                                    }
                                }

                                incomingRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), false); //connect at end
                            }
                            else //circle cuts road two times
                            {
                                SplitTrackRoadCommand *splitTrackRoadCommand = new SplitTrackRoadCommand(road, interceptPoint1, NULL);
                                if (!getProjectGraph()->executeCommand(splitTrackRoadCommand))
                                {
                                    roadIterator++;
                                    continue;
                                }

                                SplitTrackRoadCommand *secondSplitCommand = new SplitTrackRoadCommand(splitTrackRoadCommand->getSplitRoadCommand()->getSecondNewRoad(), interceptPoint2 - interceptPoint1, NULL);
                                if (!getProjectGraph()->executeCommand(secondSplitCommand))
                                {
                                    roadIterator++;
                                    continue;
                                }

                                // Remove road between interceptPoint1 and interceptPoint2
                                //
                                //							RemoveRoadCommand * removeRoadCommand = new RemoveRoadCommand(roadSystem->getRoad(road->getID()+"_2_1"));
                                //							getProjectGraph()->executeCommand(removeRoadCommand);

                                // The connecting road already exists. Store the road pairs.
                                //

                                SetRoadLinkRoadsCommand::RoadPair *pair = new SetRoadLinkRoadsCommand::RoadPair;
                                pair->road1 = secondSplitCommand->getSplitRoadCommand()->getFirstNewRoad();
                                pair->road2 = splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad();
                                pair->positionIndex = SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd;
                                roadPairs.append(pair);

                                SetRoadLinkRoadsCommand::RoadPair *pair1 = new SetRoadLinkRoadsCommand::RoadPair;
                                pair1->road1 = secondSplitCommand->getSplitRoadCommand()->getFirstNewRoad();
                                pair1->road2 = secondSplitCommand->getSplitRoadCommand()->getSecondNewRoad();
                                pair1->positionIndex = SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart;
                                roadPairs.append(pair1);

                                connectedRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), secondSplitCommand->getSplitRoadCommand()->getFirstNewRoad());
                                connectedRoads.insert(secondSplitCommand->getSplitRoadCommand()->getSecondNewRoad(), secondSplitCommand->getSplitRoadCommand()->getFirstNewRoad());
                                connectedRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), secondSplitCommand->getSplitRoadCommand()->getSecondNewRoad());

                                // add the connecting road to the junction
                                AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), secondSplitCommand->getSplitRoadCommand()->getFirstNewRoad(), newJunction);
                                if (addToJunctionCommand->isValid())
                                {
                                    getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                }
                                else
                                {
                                    printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                    delete addToJunctionCommand;
                                }

                                incomingRoads.insert(splitTrackRoadCommand->getSplitRoadCommand()->getFirstNewRoad(), false);
                                incomingRoads.insert(secondSplitCommand->getSplitRoadCommand()->getSecondNewRoad(), true);
                            }
                        }

                        roadIterator++;
                    }

                    // Create new roads between incoming roads
                    //
                    if (incomingRoads.size() > 1)
                    {

                        // Lookup a simple prototype - just the central border
                        //
                        RSystemElementRoad *simplePrototype;
                        QList<PrototypeContainer<RSystemElementRoad *> *> laneSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype);

                        for (int i = 0; i < laneSectionPrototypes.size(); i++)
                        {
                            if (laneSectionPrototypes.at(i)->getPrototypeName() == "SimplePrototype")
                            {
                                simplePrototype = laneSectionPrototypes.at(i)->getPrototype();
                            }
                        }

                        QMap<RSystemElementRoad *, bool>::const_iterator incomingRoadsIterator = incomingRoads.constBegin();

                        while (incomingRoadsIterator != incomingRoads.constEnd() - 1)
                        {
                            RSystemElementRoad *road1 = incomingRoadsIterator.key();
                            bool start1 = incomingRoadsIterator.value();

                            LaneSection *prototypeLaneSection1;
                            QMap<int, Lane *>::const_iterator laneMapIterator;
                            QMap<int, Lane *> laneMap;
                            if (start1)
                            {
                                prototypeLaneSection1 = road1->getLaneSection(0.0);
                                laneMap = road1->getLaneSection(0.0)->getLanes();
                            }
                            else
                            {
                                prototypeLaneSection1 = road1->getLaneSection(road1->getLength());
                                laneMap = road1->getLaneSection(road1->getLength())->getLanes();
                            }

                            QMap<RSystemElementRoad *, bool>::const_iterator iter = incomingRoadsIterator + 1;
                            while (iter != incomingRoads.constEnd())
                            {
                                RSystemElementRoad *road2 = iter.key();


                                // If this pair is already connected, skip the rest
                                //
                                QMap<RSystemElementRoad *, RSystemElementRoad *>::const_iterator roadsConnected = connectedRoads.constBegin();
                                while (roadsConnected != connectedRoads.constEnd())
                                {
                                    if (((roadsConnected.key() == road1) && (roadsConnected.value() == road2)) || ((roadsConnected.key() == road2) && (roadsConnected.value() == road1)))
                                    {
                                        break;
                                    }
                                    roadsConnected++;
                                }
                                if (roadsConnected != connectedRoads.constEnd())
                                {
                                    iter++;
                                    continue;
                                }

                                bool start2 = iter.value();
                                LaneSection *prototypeLaneSection2;

                                if (start2)
                                {
                                    prototypeLaneSection2 = road2->getLaneSection(0.0);
                                    laneMap = road2->getLaneSection(0.0)->getLanes();
                                }
                                else
                                {
                                    prototypeLaneSection2 = road2->getLaneSection(road2->getLength());
                                    laneMap = road1->getLaneSection(road1->getLength())->getLanes();
                                }


                                RSystemElementRoad *spiralPrototype = createSpiral(incomingRoadsIterator.key(), iter.key(), incomingRoadsIterator.value(), iter.value());
                                if (!spiralPrototype)
                                {
                                    iter++;
                                    continue;
                                }

                                // Create the lanes, interpolating the lane width
                                //

                                // Test if the new road starts near road1 or near road2
                                //
                                double dist1, dist2, paramStart, paramEnd;
                                LaneSection *prototypeLaneSectionStart, *prototypeLaneSectionEnd;
                                int laneIDStart;

                                SetRoadLinkRoadsCommand::RoadPair *roadPairStart = new SetRoadLinkRoadsCommand::RoadPair();
                                SetRoadLinkRoadsCommand::RoadPair *roadPairEnd = new SetRoadLinkRoadsCommand::RoadPair();

                                if (incomingRoadsIterator.value())
                                {
                                    dist1 = QVector2D(road1->getGlobalPoint(0.0) - spiralPrototype->getGlobalPoint(0.0)).length();
                                }
                                else
                                {
                                    dist1 = QVector2D(road1->getGlobalPoint(road1->getLength()) - spiralPrototype->getGlobalPoint(0.0)).length();
                                }

                                if (iter.value())
                                {
                                    dist2 = QVector2D(road2->getGlobalPoint(0.0) - spiralPrototype->getGlobalPoint(0.0)).length();
                                }
                                else
                                {
                                    dist2 = QVector2D(road2->getGlobalPoint(road2->getLength()) - spiralPrototype->getGlobalPoint(0.0)).length();
                                }

                                if (dist1 < dist2)
                                {
                                    roadPairStart->road2 = road1;
                                    roadPairEnd->road2 = road2;
                                    if (incomingRoadsIterator.value())
                                    {
                                        laneIDStart = -1;
                                        paramStart = 0.0;
                                        roadPairStart->positionIndex = SetRoadLinkRoadsCommand::SecondRoadStart | SetRoadLinkRoadsCommand::FirstRoadStart;
                                    }
                                    else
                                    {
                                        laneIDStart = 1;
                                        paramStart = prototypeLaneSection1->getLength();
                                        roadPairStart->positionIndex = SetRoadLinkRoadsCommand::SecondRoadEnd | SetRoadLinkRoadsCommand::FirstRoadStart;
                                    }
                                    if (iter.value())
                                    {
                                        paramEnd = 0.0;
                                        roadPairEnd->positionIndex = SetRoadLinkRoadsCommand::SecondRoadStart | SetRoadLinkRoadsCommand::FirstRoadEnd;
                                    }
                                    else
                                    {
                                        paramEnd = prototypeLaneSection2->getLength();
                                        roadPairEnd->positionIndex = SetRoadLinkRoadsCommand::SecondRoadEnd | SetRoadLinkRoadsCommand::FirstRoadEnd;
                                    }
                                    prototypeLaneSectionStart = prototypeLaneSection1;
                                    prototypeLaneSectionEnd = prototypeLaneSection2;
                                }
                                else
                                {
                                    roadPairStart->road2 = road2;
                                    roadPairEnd->road2 = road1;
                                    if (incomingRoadsIterator.value())
                                    {
                                        paramEnd = 0.0;
                                        roadPairEnd->positionIndex = SetRoadLinkRoadsCommand::SecondRoadStart | SetRoadLinkRoadsCommand::FirstRoadEnd;
                                    }
                                    else
                                    {
                                        paramEnd = prototypeLaneSection1->getLength();
                                        roadPairEnd->positionIndex = SetRoadLinkRoadsCommand::SecondRoadEnd | SetRoadLinkRoadsCommand::FirstRoadEnd;
                                        ;
                                    }
                                    if (iter.value())
                                    {
                                        laneIDStart = -1;
                                        paramStart = 0.0;
                                        roadPairStart->positionIndex = SetRoadLinkRoadsCommand::SecondRoadStart | SetRoadLinkRoadsCommand::FirstRoadStart;
                                    }
                                    else
                                    {
                                        laneIDStart = 1;
                                        paramStart = prototypeLaneSection2->getLength();
                                        roadPairStart->positionIndex = SetRoadLinkRoadsCommand::SecondRoadEnd | SetRoadLinkRoadsCommand::FirstRoadStart;
                                    }
                                    prototypeLaneSectionStart = prototypeLaneSection2;
                                    prototypeLaneSectionEnd = prototypeLaneSection1;
                                }

                                // Copy the lanes and interpolate the lane width
                                //
                                QMap<int, Lane *> endLanes = prototypeLaneSectionEnd->getLanes();
                                QMap<int, Lane *> startLanes = prototypeLaneSectionStart->getLanes();
                                QMap<int, Lane *>::const_iterator startLanesIterator = startLanes.constBegin();
                                LaneSection *newLaneSection;
                                if (startLanes.value(0) && simplePrototype) // The central border is not a lane
                                {
                                    spiralPrototype->superposePrototype(simplePrototype);
                                    newLaneSection = spiralPrototype->getLaneSection(0.0);
                                }
                                else
                                {
                                    newLaneSection = new LaneSection(0.0, false);
                                    spiralPrototype->addLaneSection(newLaneSection);
                                }
                                Lane::LaneType lastType = startLanes.value(0)->getLaneType();
                                
                                
                                for(int lr=0;lr<2;lr++)
                                {
                                    int lrMult=1;
                                    if(lr==1)
                                        lrMult=-1;

                                    // try to connect all possible lanes (inner, driving and outer lanes)
                                    int startLaneNums[3]; // number of inner lanes, number of driving lanes, nummer outer lanes
                                    int endLaneNums[3]; // number of inner lanes, number of driving lanes, nummer outer lanes
                                    int n=0;
                                    int num=0;
                                    for(int i=1;i<=startLanes.size();i++) // fill startLaneNums
                                    {
                                        if(!startLanes.contains(i*lrMult))
                                        {
                                            while(n<3)
                                            {
                                                startLaneNums[n]=num;
                                                num=0;
                                                n++;
                                            }
                                            break;
                                        }
                                        if(n==0 && startLanes[i*lrMult]->getLaneType()==Lane::LT_DRIVING)
                                        {
                                            startLaneNums[n]=num;
                                            num=0;
                                            n++;
                                        }
                                        if(n==1 && startLanes[i*lrMult]->getLaneType()!=Lane::LT_DRIVING)
                                        {
                                            startLaneNums[n]=num;
                                            num=0;
                                            n++;
                                        }
                                        num++;
                                    }
                                    n=0;
                                    num=0;
                                    for(int i=1;i<=endLanes.size();i++) // fill endLaneNums
                                    {
                                        if(!endLanes.contains(i*lrMult))
                                        {
                                            while(n<3)
                                            {
                                                endLaneNums[n]=num;
                                                num=0;
                                                n++;
                                            }
                                            break;
                                        }
                                        if(n==0 && endLanes[i*lrMult]->getLaneType()==Lane::LT_DRIVING)
                                        {
                                            endLaneNums[n]=num;
                                            num=0;
                                            n++;
                                        }
                                        if(n==1 && endLanes[i*lrMult]->getLaneType()!=Lane::LT_DRIVING)
                                        {
                                            endLaneNums[n]=num;
                                            num=0;
                                            n++;
                                        }
                                        num++;
                                    }

                                    int endLaneNum = 1;
                                    int startLaneNum=1;
                                    int newLaneNum=1;
                                    Lane *startLane = NULL;
                                    Lane *endLane = NULL;
                                    // do not connect anything if ther is no driving lane on either start or end
                                    if(startLaneNums[1]!=0 && endLaneNums[1]!=0)
                                    {
                                        for(int i=0;i<3;i++) // connect inner, driving and outer lanes
                                        {
                                            int sn=0,en=0;
                                            while(sn < startLaneNums[i] || en < endLaneNums[i])
                                            {
                                                startLane=NULL;
                                                endLane=NULL;
                                                Lane *newLane=NULL;
                                                LaneRoadMark *roadMark=NULL;
                                                double a=0, b=0;



                                                if(sn < startLaneNums[i])
                                                {
                                                    startLane = startLanes[startLaneNum*lrMult];
                                                    startLaneNum++;
                                                    newLane = new Lane(newLaneNum*lrMult, startLane->getLaneType());
                                                    roadMark = startLane->getRoadMarkEntry(paramStart);
                                                    a = startLane->getWidth(paramStart);
                                                }
                                                b = (0 - a) / spiralPrototype->getLength();
                                                if(en < endLaneNums[i])
                                                {
                                                    endLane = endLanes[endLaneNum*lrMult];
                                                    endLaneNum++;
                                                    newLane = new Lane(newLaneNum*lrMult, endLane->getLaneType());
                                                    roadMark = endLane->getRoadMarkEntry(paramEnd);
                                                    b = (endLane->getWidth(paramEnd) - a) / spiralPrototype->getLength();
                                                }
                                                sn++;
                                                en++;
                                                if (roadMark)
                                                {
                                                    LaneRoadMark *newRoadMark = new LaneRoadMark(0.0, roadMark->getRoadMarkType(), roadMark->getRoadMarkWeight(), roadMark->getRoadMarkColor(), 0.12);
                                                    newLane->addRoadMarkEntry(newRoadMark);
                                                }
                                                newLaneNum++;

                                                LaneWidth *width = new LaneWidth(0.0, a, b, 0.0, 0.0);
                                                newLane->addWidthEntry(width);
                                                InsertLaneCommand *command = new InsertLaneCommand(newLaneSection, newLane);
                                                getProjectGraph()->executeCommand(command);
                                            }
                                        }
                                    }
                                }

                   /*             while (startLanesIterator != prototypeLaneSectionStart->getLanes().constEnd())
                                {
                                    Lane *startLane = startLanesIterator.value();
                                    Lane *newLane = new Lane(startLane->getId() * laneIDStart, startLane->getLaneType());

                                    LaneRoadMark *roadMark = startLane->getRoadMarkEntry(0.0);
                                    if (roadMark)
                                    {
                                        LaneRoadMark *newRoadMark = new LaneRoadMark(0.0, roadMark->getRoadMarkType(), roadMark->getRoadMarkWeight(), roadMark->getRoadMarkColor(), 0.12);
                                        newLane->addRoadMarkEntry(newRoadMark);
                                    }

                                    if (startLane->getLaneType() != Lane::LT_BORDER)
                                    {
                                        Lane *endLane;
                                        if ((incomingRoadsIterator.value() & iter.value()) || !(incomingRoadsIterator.value() | iter.value()))
                                        {
                                            // If the number of lanes on both sides is not equal, skip the rest
                                            //
                                            if ((prototypeLaneSectionEnd->getLeftmostLaneId() != -prototypeLaneSectionStart->getRightmostLaneId()) || (-prototypeLaneSectionEnd->getRightmostLaneId() != prototypeLaneSectionStart->getLeftmostLaneId()))
                                            {
                                                startLanesIterator++;    // the number of lanes of the incoming roads are not equal
                                                continue;
                                            }

                                            endLane = endLanes.value(-startLane->getId());
                                        }
                                        else
                                        {
                                            // If the number of lanes on both sides is not equal, skip the rest
                                            //
                                            if ((prototypeLaneSectionEnd->getLeftmostLaneId() != prototypeLaneSectionStart->getLeftmostLaneId()) || (prototypeLaneSectionEnd->getRightmostLaneId() != prototypeLaneSectionStart->getRightmostLaneId()))
                                            {
                                                startLanesIterator++;    // the number of lanes of the incoming roads are not equal
                                                continue;
                                            }

                                            endLane = endLanes.value(startLane->getId());
                                        }

                                        double a, b;

                                        a = startLane->getWidth(paramStart);
                                        b = (endLane->getWidth(paramEnd) - a) / spiralPrototype->getLength();

                                        LaneWidth *width = new LaneWidth(0.0, a, b, 0.0, 0.0);
                                        newLane->addWidthEntry(width);
                                    }

                                    InsertLaneCommand *command = new InsertLaneCommand(newLaneSection, newLane);
                                    getProjectGraph()->executeCommand(command);

                                    startLanesIterator++;
                                }*/

								// Superpose planar ShapeSection //
								//
								QList<PrototypeContainer<RSystemElementRoad *> *> shapeSectionPrototypes = ODD::mainWindow()->getPrototypeManager()->getRoadPrototypes(PrototypeManager::PTP_RoadShapePrototype);

								for (int i = 0; i < shapeSectionPrototypes.size(); i++)
								{
									if (shapeSectionPrototypes.at(i)->getPrototypeName() == "Planar 0.0")
									{
										spiralPrototype->superposePrototype(shapeSectionPrototypes.at(i)->getPrototype());
										break;
									}
								}

                                NewRoadCommand *command = new NewRoadCommand(spiralPrototype, getProjectData()->getRoadSystem(), NULL);
                                getProjectGraph()->executeCommand(command);

                                // add the new connecting road to the junction
                                AddToJunctionCommand *addToJunctionCommand = new AddToJunctionCommand(getProjectData()->getRoadSystem(), spiralPrototype, newJunction);
                                if (addToJunctionCommand->isValid())
                                {
                                    getProjectData()->getUndoStack()->push(addToJunctionCommand);
                                }
                                else
                                {
                                    printStatusBarMsg(addToJunctionCommand->text(), 4000);
                                    delete addToJunctionCommand;
                                }

                                roadPairStart->road1 = spiralPrototype;
                                roadPairEnd->road1 = spiralPrototype;

                                roadPairs.append(roadPairStart);
                                roadPairs.append(roadPairEnd);

                                iter++;
                            }

                            incomingRoadsIterator++;
                        }

                        // LInk new roads
                        //
                        SetRoadLinkRoadsCommand *command = new SetRoadLinkRoadsCommand(roadPairs, NULL);
                        getProjectGraph()->executeCommand(command);
                    }

                    getProjectData()->getUndoStack()->endMacro();
                }
            }
	   }
    }

    //	ProjectEditor::mouseAction(mouseAction);
}

/*! \brief .
*
*/
void
JunctionEditor::keyAction(KeyAction *keyAction)
{
    ProjectEditor::keyAction(keyAction);
}

bool
JunctionEditor::translateTrack(TrackComponent *track, const QPointF &pressPos, const QPointF &mousePos)
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
JunctionEditor::registerJunctionMoveHandle(JunctionMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! JunctionEditor JunctionMoveHandle DOF not in [0,1,2].");
    }
    selectedJunctionMoveHandles_.insert(handle->getPosDOF(), handle);
}

int
JunctionEditor::unregisterJunctionMoveHandle(JunctionMoveHandle *handle)
{
    return selectedJunctionMoveHandles_.remove(handle->getPosDOF(), handle);
}

bool
JunctionEditor::translateJunctionMoveHandles(const QPointF &pressPos, const QPointF &mousePosConst)
{
    QPointF mousePos = mousePosConst;

    // Snap to MoveHandle //
    //
    foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
    {
        JunctionMoveHandle *handle = dynamic_cast<JunctionMoveHandle *>(item);
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

    //qDebug() << "selectedJunctionMoveHandles_: " << selectedJunctionMoveHandles_.size();

    // No entries //
    //
    if (selectedJunctionMoveHandles_.size() == 0)
    {
        return false;
    }

    // 0: Check for zero degrees of freedom //
    //
    if (selectedJunctionMoveHandles_.count(0) > 0)
    {
        return false;
    }

    // 1: Check for one degree of freedom //
    //
    if (selectedJunctionMoveHandles_.count(1) > 0)
    {
        // Check if all are the same //
        //
        double posDofHeading = selectedJunctionMoveHandles_.constBegin().value()->getPosDOFHeading(); // first
        foreach (JunctionMoveHandle *moveHandle, selectedJunctionMoveHandles_.values(1))
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

    foreach (JunctionMoveHandle *moveHandle, selectedJunctionMoveHandles_)
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
    foreach (JunctionMoveHandle *moveHandle, selectedJunctionMoveHandles_)
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

#if 0

//################//
// RotateHandles  //
//################//

void
	JunctionEditor
	::registerTrackRotateHandle(TrackRotateHandle * handle)
{
	if(handle->getRotDOF() < 0 || handle->getRotDOF() > 1)
	{
		qDebug("WARNING 1004261417! JunctionEditor TrackRotateHandle DOF not in [0,1].");
	}
	selectedTrackRotateHandles_.insert(handle->getRotDOF(), handle);
}

int
	JunctionEditor
	::unregisterTrackRotateHandle(TrackRotateHandle * handle)
{
	return selectedTrackRotateHandles_.remove(handle->getRotDOF(), handle);
}



double
	JunctionEditor
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
JunctionEditor::registerJunctionAddHandle(JunctionAddHandle *handle)
{
    selectedJunctionAddHandles_.insert(0.0, handle);
    return;
}

int
JunctionEditor::unregisterJunctionAddHandle(JunctionAddHandle *handle)
{
    return selectedJunctionAddHandles_.remove(0.0, handle);
}

//################//
// AddLaneSections    //
//################//

void
JunctionEditor::registerLane(Lane *lane)
{
    if (selectedLanes_.size() == 0)
    {
        selectedLanes_.append(lane);
    }
    else if (lane != selectedLanes_.first())
    {
        selectedLanes_.append(lane);
        if (selectedLanes_.size() == 2)
        {
            createRoad(selectedLanes_);
            selectedLanes_.clear();
        }
    }
    return;
}

//################//
// Add roads to be connected    //
//################//

void
JunctionEditor::registerRoad(RSystemElementRoad *road)
{
    if (selectedRoads_.size() == 0)
    {
        selectedRoads_.append(road);
    }
    else if (road != selectedRoads_.first())
    {
        selectedRoads_.append(road);
        if (selectedRoads_.size() == 2)
        {
            createRoad(selectedRoads_);
            selectedRoads_.clear();
        }
    }
    return;
}

//################//
// SLOTS          //
//################//

/*!
*/
void
JunctionEditor::init()
{
    if (!junctionRoadSystemItem_)
    {
        // Root item //
        //
        junctionRoadSystemItem_ = new JunctionRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(junctionRoadSystemItem_);

        // New Road Item //
        //
        QPen pen;
        pen.setWidth(2);
        pen.setCosmetic(true); // constant size independent of scaling
        pen.setColor(ODD::instance()->colors()->brightGreen());
        newRoadLineItem_ = new QGraphicsLineItem(junctionRoadSystemItem_);
        newRoadLineItem_->setPen(pen);
        newRoadLineItem_->hide();

        // Add RoadSystem Item //
        //
        addRoadSystemHandle_ = new CircularRotateHandle(junctionRoadSystemItem_);
        addRoadSystemHandle_->setVisible(false);
    }

    // Section Handle //
    //
    // TODO: Is this really the best object for holding this?
    sectionHandle_ = new SectionHandle(junctionRoadSystemItem_);
    sectionHandle_->hide();

    // Lanes //
    //
    if (!laneRoadSystemItem_)
    {
        // Root item //
        //
        laneRoadSystemItem_ = new JunctionLaneRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(laneRoadSystemItem_);
    }
}

/*!
*/
void
JunctionEditor::kill()
{
    // RoadSystemItem //
    //
    //topviewGraph_->graphScene()->removeItem(junctionRoadSystemItem_);
    delete junctionRoadSystemItem_;
    junctionRoadSystemItem_ = NULL;

    // ToolHandles //
    //
    selectedJunctionMoveHandles_.clear();
    selectedJunctionAddHandles_.clear();

    delete laneRoadSystemItem_;
    laneRoadSystemItem_ = NULL;

	getTopviewGraph()->getView()->deleteCircle();
}

//################//
// LaneMoveHandles    //
//################//

void
JunctionEditor::registerLaneMoveHandle(JunctionLaneWidthMoveHandle *handle)
{
    if (handle->getPosDOF() < 0 || handle->getPosDOF() > 2)
    {
        qDebug("WARNING 1004261416! ElevationEditor ElevationMoveHandle DOF not in [0,1,2].");
    }
    selectedLaneMoveHandles_.insert(handle->getPosDOF(), handle);
}

int
JunctionEditor::unregisterLaneMoveHandle(JunctionLaneWidthMoveHandle *handle)
{
    return selectedLaneMoveHandles_.remove(handle->getPosDOF(), handle);
}

/*void Editor::setWidth(double w)
{
	QList<LaneWidth *> endPointSections;
	QList<LaneWidth *> startPointSections;
	foreach(LaneWidthMoveHandle * moveHandle, selectedMoveHandles_)
	{
		LaneWidth * lowSlot = moveHandle->getLowSlot();
		if(lowSlot)
		{
			endPointSections.append(lowSlot);
		}

		LaneWidth * highSlot = moveHandle->getHighSlot();
		if(highSlot)
		{
			startPointSections.append(highSlot);
		}
	}
	LaneSetWidthCommand * command = new LaneSetWidthCommand(endPointSections, startPointSections, w, NULL);

	getProjectData()->getUndoStack()->push(command);
}*/

bool
JunctionEditor::translateLaneMoveHandles(const QPointF &pressPos, const QPointF &mousePos)
{
    QPointF dPos = mousePos - pressPos;

    // No entries //
    //
    if (selectedLaneMoveHandles_.size() == 0)
    {
        return false;
    }

    // 0: Check for zero degrees of freedom //
    //
    if (selectedLaneMoveHandles_.count(0) > 0)
    {
        return false;
    }

    // 1: Check for one degree of freedom //
    //
    if (selectedLaneMoveHandles_.count(1) > 0)
    {
        printStatusBarMsg(tr("Sorry, you can't move yellow items."), 4000);
        qDebug("One DOF not supported yet");
        return false;
    }

    // 2: Two degrees of freedom //
    //
    QList<LaneWidth *> endPointSections;
    QList<LaneWidth *> startPointSections;
    foreach (JunctionLaneWidthMoveHandle *moveHandle, selectedLaneMoveHandles_)
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
