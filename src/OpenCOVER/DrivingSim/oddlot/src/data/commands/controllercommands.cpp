/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.05.2010
**
**************************************************************************/
#include "src/data/commands/controllercommands.hpp"

#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"


//#########################//
// AddControllerCommand //
//#########################//

AddControllerCommand::AddControllerCommand(RSystemElementController *controller, RoadSystem *roadSystem, DataCommand *parent)
    : DataCommand(parent)
    , controller_(controller)
    , roadSystem_(roadSystem)
{
    // Check for validity //
    //
    if (!controller || !roadSystem_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddControllerCommand: Internal error! No RoadSystem specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddController"));
    }
}

/*! \brief .
*
*/
AddControllerCommand::~AddControllerCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete controller_;
    }
    else
    {
        // nothing to be done (controller is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddControllerCommand::redo()
{
    roadSystem_->addController(controller_);

    setRedone();
}

/*! \brief
*
*/
void
AddControllerCommand::undo()
{
    roadSystem_->delController(controller_);

    setUndone();
}

//#########################//
// RemoveControllerCommand //
//#########################//

RemoveControllerCommand::RemoveControllerCommand(RSystemElementController *controller, RoadSystem *roadSystem, DataCommand *parent)
    : DataCommand(parent)
    , controller_(controller)
    , roadSystem_(roadSystem)
{
    // Check for validity //
    //
    if (!controller || !roadSystem_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveControllerCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveController"));
    }
}

/*! \brief .
*
*/
RemoveControllerCommand::~RemoveControllerCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (controller is now owned by the road)
    }
    else
    {
        delete controller_;
    }
}

/*! \brief .
*
*/
void
RemoveControllerCommand::redo()
{
    roadSystem_->delController(controller_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveControllerCommand::undo()
{
    roadSystem_->addController(controller_);

    setUndone();
}

//#########################//
// SetControllerPropertiesCommand //
//#########################//
SetControllerPropertiesCommand::SetControllerPropertiesCommand(RSystemElementController *controller, const QString &id, const QString &name, int sequence, const QString &script, double cycleTime, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , newSequence_(sequence)
    , controller_(controller)
{
    // Check for validity //
    //
    if (!controller)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetControllerPropertiesCommand: Internal error! No controller specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetProperties"));
    }

    newControllerUserData_.cycleTime = cycleTime;
    newControllerUserData_.script = script;

    oldId_ = controller_->getID();
    oldName_ = controller_->getName();
    oldSequence_ = controller_->getSequence();
    oldControllerUserData_ = controller_->getControllerUserData();
}

SetControllerPropertiesCommand::SetControllerPropertiesCommand(RSystemElementController *controller, const QString &id, const QString &name, int sequence, const RSystemElementController::ControllerUserData &controllerUserData, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , newSequence_(sequence)
    , newControllerUserData_(controllerUserData)
    , controller_(controller)
{
    // Check for validity //
    //
    if (!controller)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetControllerPropertiesCommand: Internal error! No controller specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetProperties"));
    }


    oldId_ = controller_->getID();
    oldName_ = controller_->getName();
    oldSequence_ = controller_->getSequence();
    oldControllerUserData_ = controller_->getControllerUserData();
}

/*! \brief .
*
*/
SetControllerPropertiesCommand::~SetControllerPropertiesCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
    }
    else
    {
        // nothing to be done (controller is now owned by the road)
    }
}

/*! \brief .
*
*/
void
SetControllerPropertiesCommand::redo()
{
    controller_->setID(newId_);
    controller_->setName(newName_);
    controller_->setSequence(newSequence_);
    controller_->setControllerUserData(newControllerUserData_);

    controller_->addControllerChanges(RSystemElementController::CRC_ParameterChange);

    setRedone();
}

/*! \brief
*
*/
void
SetControllerPropertiesCommand::undo()
{
    controller_->setID(oldId_);
    controller_->setName(oldName_);
    controller_->setSequence(oldSequence_);
    controller_->setControllerUserData(oldControllerUserData_);

    controller_->addControllerChanges(RSystemElementController::CRC_ParameterChange);

    setUndone();
}

//#########################//
// AddControlEntryCommand //
//#########################//

AddControlEntryCommand::AddControlEntryCommand(RSystemElementController *controller, ControlEntry *controlEntry, Signal * signal, DataCommand *parent)
    : DataCommand(parent)
    , controller_(controller)
    , controlEntry_(controlEntry)
    , signal_(signal)
{
    // Check for validity //
    //
    if (!controller || !controlEntry_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddControlEntryCommand: Internal error! No Control Entry specified."));
        return;
    }

    // Check if the entry is in the list //
    //
    foreach (ControlEntry * entry, controller_->getControlEntries())
    {
        if (entry->getSignalId() == controlEntry_->getSignalId())
        {  
            setInvalid(); // Invalid
            setText(QObject::tr("AddControlEntryCommand: ControlEntry has already been added."));
            return;
        }
    }


    setValid();
    setText(QObject::tr("AddControlEntry"));
}

/*! \brief .
*
*/
AddControlEntryCommand::~AddControlEntryCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete controlEntry_;
    }
    else
    {
        // nothing to be done (controller is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddControlEntryCommand::redo()
{
    controller_->addControlEntry(controlEntry_, signal_);

    setRedone();
}

/*! \brief
*
*/
void
AddControlEntryCommand::undo()
{
    controller_->delControlEntry(controlEntry_, signal_);

    setUndone();
}

//#########################//
// DelControlEntryCommand //
//#########################//

DelControlEntryCommand::DelControlEntryCommand(RSystemElementController *controller, ControlEntry *controlEntry, Signal * signal, DataCommand *parent)
    : DataCommand(parent)
    , controller_(controller)
    , controlEntry_(controlEntry)
    , signal_(signal)
{
    // Check for validity //
    //
    if (!controller || !controlEntry_ || !signal_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("DelControlEntryCommand: Internal error! No Control Entry specified."));
        return;
    }
    else if (!controller_->getControlEntries().contains(controlEntry))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("DelControlEntryCommand: Internal error! Entry not in list."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("DelControlEntry"));
    }
}

/*! \brief .
*
*/
DelControlEntryCommand::~DelControlEntryCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete controlEntry_;
    }
    else
    {
        // nothing to be done (controller is now owned by the road)
    }
}

/*! \brief .
*
*/
void
DelControlEntryCommand::redo()
{
    controller_->delControlEntry(controlEntry_, signal_);

    setRedone();
}

/*! \brief
*
*/
void
DelControlEntryCommand::undo()
{
    controller_->addControlEntry(controlEntry_, signal_);

    setUndone();
}
