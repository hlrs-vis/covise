/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#include "projectdata.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "roadsystem/roadsystem.hpp"
#include "tilesystem/tilesystem.hpp"
#include "vehiclesystem/vehiclesystem.hpp"
#include "pedestriansystem/pedestriansystem.hpp"
#include "scenerysystem/scenerysystem.hpp"
#include "tilesystem/tilesystem.hpp"
#include "oscsystem/oscbase.hpp"

#include "changemanager.hpp"

// OpenScenario //
//
#include "OpenScenarioBase.h"

using namespace OpenScenario;

// Qt //
//
#include <QUndoStack>

/*!
* CONSTRUCTOR.
*
*/
ProjectData::ProjectData(ProjectWidget *projectWidget, QUndoStack *undoStack, ChangeManager *changeManager, int revMajor, int revMinor, QString name, float version, QString date, double north, double south, double east, double west)
    : QObject(projectWidget)
    , DataElement()
    , projectDataChanges_(0x0)
    , projectWidget_(projectWidget)
    , revMajor_(revMajor)
    , revMinor_(revMinor)
    , name_(name)
    , version_(version)
    , date_(date)
    , north_(north)
    , south_(south)
    , east_(east)
    , west_(west)
    , roadSystem_(NULL)
    , tileSystem_(NULL)
    , vehicleSystem_(NULL)
    , pedestrianSystem_(NULL)
    , scenerySystem_(NULL)
	, oscBase_(NULL)
    , undoStack_(undoStack)
    , changeManager_(changeManager)
{
    linkToProject(this); // link to itself
}

/*!
* DESTRUCTOR.
*
*/
ProjectData::~ProjectData()
{
    delete undoStack_;

    delete roadSystem_;
    delete tileSystem_;
    delete vehicleSystem_;
    delete pedestrianSystem_;
    delete scenerySystem_;
	delete oscBase_;
}

//##################//
// OpenDRIVE        //
//##################//

void
ProjectData::setRevMajor(int revMajor)
{
    if (revMajor_ != revMajor)
    {
        revMajor_ = revMajor;
        addProjectDataChanges(ProjectData::CPD_RevChange);
    }
}

void
ProjectData::setRevMinor(int revMinor)
{
    if (revMinor_ != revMinor)
    {
        revMinor_ = revMinor;
        addProjectDataChanges(ProjectData::CPD_RevChange);
    }
}

void
ProjectData::setName(const QString &name)
{
    if (name_ != name)
    {
        name_ = name;
        addProjectDataChanges(ProjectData::CPD_NameChange);
    }
}

void
ProjectData::setVersion(float version)
{
    if (version_ != version)
    {
        version_ = version;
        addProjectDataChanges(ProjectData::CPD_VersionChange);
    }
}

void
ProjectData::setDate(const QString &date)
{
    if (date_ != date)
    {
        date_ = date;
        addProjectDataChanges(ProjectData::CPD_DateChange);
    }
}

void
ProjectData::setNorth(double north)
{
    if (north_ != north)
    {
        north_ = north;
        addProjectDataChanges(ProjectData::CPD_SizeChange);
    }
}

void
ProjectData::setSouth(double south)
{
    if (south_ != south)
    {
        south_ = south;
        addProjectDataChanges(ProjectData::CPD_SizeChange);
    }
}

void
ProjectData::setEast(double east)
{
    if (east_ != east)
    {
        east_ = east;
        addProjectDataChanges(ProjectData::CPD_SizeChange);
    }
}

void
ProjectData::setWest(double west)
{
    if (west_ != west)
    {
        west_ = west;
        addProjectDataChanges(ProjectData::CPD_SizeChange);
    }
}

//##################//
// SLOTS            //
//##################//

/*! \brief This slot is called when the ProjectData's project has been activated or deactivated.
*
* \li Sets its UndoStack active, which is then displayed in the MainWindow's undo dock.
*/
void
ProjectData::projectActivated(bool active)
{
    if (active)
    {
        // UndoStack //
        //
        undoStack_->setActive();
    }
}

//##################//
// Systems          //
//##################//

void
ProjectData::setRoadSystem(RoadSystem *roadSystem)
{
    // Can only be assigned once //
    if (!roadSystem_)
    {
        roadSystem_ = roadSystem;
        roadSystem_->setParentProjectData(this);
        addProjectDataChanges(ProjectData::CPD_RoadSystemChanged);
    }
    else
    {
        qDebug("WARNING 1003230927! Cannot assign RoadSystem to ProjectData twice!");
    }
}

void
ProjectData::setTileSystem(TileSystem *tileSystem)
{
    // Can only be assigned once //
    if (!tileSystem_)
    {
        tileSystem_ = tileSystem;
        tileSystem_->setParentProjectData(this);
        addProjectDataChanges(ProjectData::CPD_TileSystemChanged);
    }
    else
    {
        qDebug("WARNING 1003230927! Cannot assign RoadSystem to ProjectData twice!");
    }
}

void
ProjectData::setVehicleSystem(VehicleSystem *vehicleSystem)
{
    // Can only be assigned once //
    if (!vehicleSystem_)
    {
        vehicleSystem_ = vehicleSystem;
        vehicleSystem_->setParentProjectData(this);
        addProjectDataChanges(ProjectData::CPD_VehicleSystemChanged);
    }
    else
    {
        qDebug("WARNING 1003230928! Cannot assign VehicleSystem to ProjectData twice!");
    }
}

void
ProjectData::setPedestrianSystem(PedestrianSystem *pedestrianSystem)
{
    // Can only be assigned once //
    if (!pedestrianSystem_)
    {
        pedestrianSystem_ = pedestrianSystem;
        pedestrianSystem_->setParentProjectData(this);
        addProjectDataChanges(ProjectData::CPD_PedestrianSystemChanged);
    }
    else
    {
        qDebug("WARNING 1003230930! Cannot assign PedestrianSystem to ProjectData twice!");
    }
}

void
ProjectData::setScenerySystem(ScenerySystem *scenerySystem)
{
    // Can only be assigned once //
    if (!scenerySystem_)
    {
        scenerySystem_ = scenerySystem;
        scenerySystem_->setParentProjectData(this);
        addProjectDataChanges(ProjectData::CPD_ScenerySystemChanged);
    }
    else
    {
        qDebug("WARNING 1003230929! Cannot assign ScenerySystem to ProjectData twice!");
    }
}

void
ProjectData::setOSCBase(OSCBase *base)
{
    // Can only be assigned once //
    if (!oscBase_)
    {
        oscBase_ = base;
        oscBase_->setParentProjectData(this);

		OpenScenario::OpenScenarioBase *openScenarioBase = new OpenScenario::OpenScenarioBase();     // make OpenScenarioBase //
		oscBase_->setOpenScenarioBase(openScenarioBase);
  //      addProjectDataChanges(ProjectData::CPD_RoadSystemChanged);
    }
    else
    {
        qDebug("WARNING 1003230927! Cannot assign RoadSystem to ProjectData twice!");
    }
}

DataElement *
ProjectData::getActiveElement() const
{
    if (selectedElements_.isEmpty())
    {
        return NULL;
    }
    else
    {
        return selectedElements_.last();
    }
}

void
ProjectData::addSelectedElement(DataElement *dataElement)
{
    selectedElements_.removeOne(dataElement); // remove it, if in list
    selectedElements_.append(dataElement); // put it to the back (so it is the first)
    addProjectDataChanges(CPD_SelectedElementsChanged);
}

void
ProjectData::removeSelectedElement(DataElement *dataElement)
{
    selectedElements_.removeOne(dataElement);
    addProjectDataChanges(CPD_SelectedElementsChanged);
}

void
ProjectData::addHiddenElement(DataElement *dataElement)
{
    hiddenElements_.removeOne(dataElement); // remove it, if in list
    hiddenElements_.append(dataElement); // put it to the back (so it is the first)
    addProjectDataChanges(CPD_HiddenElementsChanged);
}

void
ProjectData::removeHiddenElement(DataElement *dataElement)
{
    hiddenElements_.removeOne(dataElement);
    addProjectDataChanges(CPD_HiddenElementsChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Add one or more change flags.
*
*/
void
ProjectData::addProjectDataChanges(int changes)
{
    if (changes)
    {
        projectDataChanges_ |= changes;
        notifyObservers();
    }
}

void
ProjectData::notificationDone()
{
    projectDataChanges_ = 0x0;
}

//##################//
// Visitor Pattern  //
//##################//

/*!
* Accepts a visitor.
* With autotraverse: visitor will be send to roadsystem, backgrounds etc.
* Without: accepts visitor as 'this'.
*/
void
ProjectData::accept(Visitor *visitor)
{
    visitor->visit(this);
}
