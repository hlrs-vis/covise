/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/8/2010
**
**************************************************************************/

#include "circlewizard.hpp"
#include "ui_circlewizard.h"

// Data //
//
#include "src/data/projectdata.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"

#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/heightmap.hpp"

#include "src/data/commands/trackcommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
#include "src/data/commands/junctioncommands.hpp"

#include "src/cover/coverconnection.hpp"
#include "src/data/prototypemanager.hpp"
#include <qdialogbuttonbox.h>
#include <qpushbutton.h>
#include <qundostack.h>


//################//
// CONSTRUCTOR    //
//################//

CircleWizard::CircleWizard(ProjectData *projectData, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::CircleWizard)
    , projectData_(projectData)
{
    ui->setupUi(this);

    init();
}

CircleWizard::~CircleWizard()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
CircleWizard::init()
{
    // Signals //
    //
    //connect(ui->selectAllRoads, SIGNAL(released()), this, SLOT(selectAllRoads()));
    //connect(ui->deselectAllRoads, SIGNAL(released()), this, SLOT(deselectAllRoads()));

    connect(ui->buttonBox->button(QDialogButtonBox::Apply), SIGNAL(released()), this, SLOT(runCalculation()));

}

//################//
// EVENTS         //
//################//

//################//
// SLOTS          //
//################//

/*void
CircleWizard::validateRunButton()
{
    ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(true);
}*/

void
CircleWizard::runCalculation()
{

	float enterExitLength = ui->ALengthBox->value();
	float linearLength = ui->LengthBox->value();
	float laneWidth = 3.5;
  
    // Macro Command //
    //
    projectData_->getUndoStack()->beginMacro(tr("CreateCircle"));

	RSystemElementRoad *Prototype3;
	RSystemElementRoad *Prototype4;


	Prototype3 = new RSystemElementRoad("prototype");
	Prototype3->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_ElevationPrototype, "Planar 0.0"));
	Prototype3->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_CrossfallPrototype, "Planar 0.0"));
	Prototype3->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_LaneSectionPrototype, "Circle:Right:3"));

	Prototype3 = new RSystemElementRoad("prototype");
	Prototype4->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_ElevationPrototype, "Planar 0.0"));
	Prototype4->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_CrossfallPrototype, "Planar 0.0"));
	Prototype4->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_LaneSectionPrototype, "Circle:Right:4"));

	// Road //
	//
	RSystemElementRoad *newRoad = new RSystemElementRoad("RightCircle");
	// Track //
	double radius = ui->DiameterBox->value() / 2.0;
	TrackElementArc *arc1 = new TrackElementArc(0.0, 0.0 - radius, 0.0, 0.0, M_PI*radius, 1 / radius);
	newRoad->addTrackComponent(arc1);
	newRoad->superposePrototype(Prototype4);
	RSystemElementRoad *newRoadLeft = new RSystemElementRoad("LeftCircle");
	TrackElementArc *arc2 = new TrackElementArc(0.0, 0.0 + radius, 180.0, 0.0, M_PI_2*radius, 1 / radius);
	newRoadLeft->addTrackComponent(arc2);
	newRoadLeft->superposePrototype(Prototype3);
	RSystemElementRoad *newRoadLowerLeft = new RSystemElementRoad("LowerLeftCircle");
	TrackElementArc *arc3 = new TrackElementArc(0.0-radius, 0.0, 270.0, 0.0, M_PI_2*radius, 1 / radius);
	newRoadLowerLeft->addTrackComponent(arc3);
	newRoadLowerLeft->superposePrototype(Prototype3);





	LaneSection *lsThreeLanes = newRoadLeft->getLaneSection(0)->getClone();
	lsThreeLanes->setSStart(enterExitLength);
	LaneSection *lsFourLanes = newRoad->getLaneSection(0)->getClone();
	LaneWidth *lw1 = lsFourLanes->getLane(-3)->getWidthEntry(0.0);
	lw1->setParameters(0.0, .25, 0.0, 0.0);
	LaneWidth *lw2 = new LaneWidth(14, 3.5, 0.0, 0.0, 0.0);
	lsFourLanes->getLane(-3)->addWidthEntry(lw1);
	lsFourLanes->getLane(-3)->addWidthEntry(lw2);
	lsFourLanes->setSStart(M_PI*radius - enterExitLength);
	newRoad->addLaneSection(lsThreeLanes);
	newRoad->addLaneSection(lsFourLanes);

	LaneSection *lsFourLanesFirst = newRoad->getLaneSection(0);
	LaneWidth *lw3 = new LaneWidth(lsFourLanesFirst->getLength()-14.0, 3.5, -0.25, 0.0, 0.0);
	lsFourLanesFirst->getLane(-3)->addWidthEntry(lw3);


	NewRoadCommand *command = new NewRoadCommand(newRoad, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), NULL);
	if (command->isValid())
	{
		projectData_->getUndoStack()->push(command);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete command;
		return; // usually not the case, only if road or prototype are NULL
	}

	
	command = new NewRoadCommand(newRoadLeft, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), NULL);
	if (command->isValid())
	{
		projectData_->getUndoStack()->push(command);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete command;
		return; // usually not the case, only if road or prototype are NULL
	}

	command = new NewRoadCommand(newRoadLowerLeft, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), NULL);
	if (command->isValid())
	{
		projectData_->getUndoStack()->push(command);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete command;
		return; // usually not the case, only if road or prototype are NULL
	}



	RSystemElementRoad *PrototypeSingle = new RSystemElementRoad("prototype", odrID::invalidID(), odrID::invalidID());
	PrototypeSingle->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_ElevationPrototype, "Planar 0.0"));
	PrototypeSingle->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_CrossfallPrototype, "Planar 0.0"));
	PrototypeSingle->superposePrototype(ODD::mainWindow()->getPrototypeManager()->getRoadPrototype(PrototypeManager::PTP_LaneSectionPrototype, "Circle:Right:1"));

	RSystemElementRoad *newRoadEntry = new RSystemElementRoad("CircleEntry", odrID::invalidID(), odrID::invalidID());
	TrackElementLine *line1 = new TrackElementLine(0.0, 0.0 - (radius + (laneWidth*4)), 180.0, 0.0, linearLength);
	newRoadEntry->addTrackComponent(line1);
	newRoadEntry->superposePrototype(PrototypeSingle);

	command = new NewRoadCommand(newRoadEntry, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), NULL);
	if (command->isValid())
	{
		projectData_->getUndoStack()->push(command);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete command;
		return; // usually not the case, only if road or prototype are NULL
	}



	RSystemElementRoad *newRoadExit = new RSystemElementRoad("CircleExit");
	TrackElementLine *line2 = new TrackElementLine(0.0, 0.0 + radius+(laneWidth*3), 180.0, 0.0, linearLength);
	newRoadExit->addTrackComponent(line2);
	newRoadExit->superposePrototype(PrototypeSingle);

	command = new NewRoadCommand(newRoadExit, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), NULL);
	if (command->isValid())
	{
		projectData_->getUndoStack()->push(command);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete command;
		return; // usually not the case, only if road or prototype are NULL
	}
	RSystemElementJunction *newJunction = new RSystemElementJunction("EntryJunction");
	NewJunctionCommand *jcommand = new NewJunctionCommand(newJunction, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem());
	if (jcommand->isValid())
	{
		projectData_->getUndoStack()->push(jcommand);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete jcommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	AddToJunctionCommand *ajcommand = new AddToJunctionCommand(ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), newRoadEntry, newJunction);
	if (ajcommand->isValid())
	{
		projectData_->getUndoStack()->push(ajcommand);
	}
	else
	{
		delete ajcommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	ajcommand = new AddToJunctionCommand(ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), newRoadLowerLeft, newJunction);
	if (ajcommand->isValid())
	{
		projectData_->getUndoStack()->push(ajcommand);
	}
	else
	{
		delete ajcommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	RSystemElementJunction *newExitJunction = new RSystemElementJunction("ExitJunction");
	jcommand = new NewJunctionCommand(newExitJunction, ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem());
	if (jcommand->isValid())
	{
		projectData_->getUndoStack()->push(jcommand);
	}
	else
	{
		//printStatusBarMsg(command->text(), 4000);
		delete jcommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	ajcommand = new AddToJunctionCommand(ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), newRoadExit, newExitJunction);
	if (ajcommand->isValid())
	{
		projectData_->getUndoStack()->push(ajcommand);
	}
	else
	{
		delete ajcommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	ajcommand = new AddToJunctionCommand(ODD::mainWindow()->getActiveProject()->getProjectData()->getRoadSystem(), newRoadLeft, newExitJunction);
	if (ajcommand->isValid())
	{
		projectData_->getUndoStack()->push(ajcommand);
	}
	else
	{
		delete ajcommand;
		return; // usually not the case, only if road or prototype are NULL
	}


	JunctionConnection* newConnection = new JunctionConnection("jc0", newRoad->getID(), newRoadEntry->getID(), JunctionConnection::JCP_START, 0);

	RoadLink *roadLink = new RoadLink("junction", newJunction->getID(), JunctionConnection::JCP_NONE);
	SetRoadLinkCommand *slCommand = new SetRoadLinkCommand(newRoad, RoadLink::DRL_PREDECESSOR, roadLink, newConnection, newJunction);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	newConnection = new JunctionConnection("jc1",  newRoad->getID(), newRoadLowerLeft->getID(), JunctionConnection::JCP_END, 0);

	roadLink = new RoadLink("junction", newJunction->getID(), JunctionConnection::JCP_NONE);
	slCommand = new SetRoadLinkCommand(newRoad, RoadLink::DRL_PREDECESSOR, roadLink, newConnection, newJunction);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	newConnection = new JunctionConnection("jc0", newRoad->getID(), newRoadExit->getID(), JunctionConnection::JCP_START, 0);
	roadLink = new RoadLink("junction", newExitJunction->getID(), JunctionConnection::JCP_NONE);
	slCommand = new SetRoadLinkCommand(newRoad, RoadLink::DRL_SUCCESSOR, roadLink, newConnection, newExitJunction);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	newConnection = new JunctionConnection("jc1", newRoad->getID(), newRoadLeft->getID(), JunctionConnection::JCP_START, 0);

	roadLink = new RoadLink("junction", newExitJunction->getID(), JunctionConnection::JCP_NONE);
	slCommand = new SetRoadLinkCommand(newRoad, RoadLink::DRL_SUCCESSOR, roadLink, newConnection, newExitJunction);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	roadLink = new RoadLink("road", newRoadLowerLeft->getID(), JunctionConnection::JCP_END);
	slCommand = new SetRoadLinkCommand(newRoad, RoadLink::DRL_PREDECESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	roadLink = new RoadLink("road", newRoadLowerLeft->getID(), JunctionConnection::JCP_START);
	slCommand = new SetRoadLinkCommand(newRoadLeft, RoadLink::DRL_SUCCESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	roadLink = new RoadLink("road", newRoadLeft->getID(), JunctionConnection::JCP_END);
	slCommand = new SetRoadLinkCommand(newRoadLowerLeft, RoadLink::DRL_PREDECESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	roadLink = new RoadLink("road", newRoad->getID(), JunctionConnection::JCP_END);
	slCommand = new SetRoadLinkCommand(newRoadLeft, RoadLink::DRL_PREDECESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}
	roadLink = new RoadLink("road", newRoad->getID(), JunctionConnection::JCP_END);
	slCommand = new SetRoadLinkCommand(newRoadExit, RoadLink::DRL_PREDECESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	roadLink = new RoadLink("road", newRoad->getID(), JunctionConnection::JCP_START);
	slCommand = new SetRoadLinkCommand(newRoadEntry, RoadLink::DRL_PREDECESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	roadLink = new RoadLink("road", newRoad->getID(), JunctionConnection::JCP_START);
	slCommand = new SetRoadLinkCommand(newRoadLowerLeft, RoadLink::DRL_SUCCESSOR, roadLink, NULL, NULL);
	if (slCommand->isValid())
	{
		projectData_->getUndoStack()->push(slCommand);
	}
	else
	{
		delete slCommand;
		return; // usually not the case, only if road or prototype are NULL
	}

	Object::ObjectProperties objectProps{ -8.60, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	Object::ObjectRepeatRecord repeatProps{ 0.0, M_PI*radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
	Object *gr1 = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectProps, repeatProps, "none");
	newRoad->addObject(gr1);

	Object::ObjectProperties objectPropsInnen{ 8.60, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	Object *gr1i = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectPropsInnen, repeatProps, "none");
	newRoad->addObject(gr1i);

	Object::ObjectProperties objectProps2{ -5.0, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	
	float sOffset = sin(acos((radius + (1 * laneWidth))/(radius + (2 * laneWidth))))* (radius+(2*laneWidth));


	Object::ObjectRepeatRecord repeatProps2{ sOffset, (M_PI*radius) - (2* sOffset), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
	Object *gr2 = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectProps2, repeatProps2, "none");
	newRoadLeft->addObject(gr2);
	Object *gr2i = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectPropsInnen, repeatProps, "none");
	newRoadLeft->addObject(gr2i);

	Object::ObjectProperties objectProps3{ 1.6, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	Object::ObjectRepeatRecord repeatProps3{ 0, linearLength-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
	Object *gr3 = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectProps3, repeatProps3, "none");
	newRoadEntry->addObject(gr3);

	Object::ObjectProperties objectProps4{-5.1, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	Object::ObjectRepeatRecord repeatProps4{ sOffset, linearLength-(sOffset+1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
	Object *gr4 = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectProps4, repeatProps4, "none");
	newRoadEntry->addObject(gr4);


	Object::ObjectProperties objectProps5{  - 5.1, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	Object::ObjectRepeatRecord repeatProps5{ 0, linearLength - 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
	Object *gr5 = new Object(projectData_->getRoadSystem()->getID("guardRail", odrID::ID_Object), "guardRail", 0.0, objectProps5, repeatProps5, "none");
	newRoadExit->addObject(gr5);

	Object::ObjectProperties objectProps6{ 1.6, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "guardRail", 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, false };
	Object::ObjectRepeatRecord repeatProps6{ sOffset, linearLength - (sOffset + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
	Object *gr6 = new Object(projectData_->getRoadSystem()->getID("guardRail",odrID::ID_Object), "guardRail", 0.0, objectProps6, repeatProps6, "none");
	newRoadExit->addObject(gr6);

    // Macro Command //
    //
    projectData_->getUndoStack()->endMacro();

    // Quit //
    //
    done(0);
}
