/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "projectdatacommands.hpp"

#include "src/data/projectdata.hpp"

//#######################//
// SetProjectDimensionsCommand //
//#######################//

SetProjectDimensionsCommand::SetProjectDimensionsCommand(ProjectData *projectData, double north, double south, double east, double west, DataCommand *parent)
    : DataCommand(parent)
    , projectData_(projectData)
    , newNorth_(north)
    , newSouth_(south)
    , newEast_(east)
    , newWest_(west)
{
    // Old size //
    //
    oldNorth_ = projectData_->getNorth();
    oldSouth_ = projectData_->getSouth();
    oldEast_ = projectData_->getEast();
    oldWest_ = projectData_->getWest();

    // Check for validity //
    //
    if (((newNorth_ - newSouth_) < 10.0)
        || ((newEast_ - newWest_) < 10.0))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Canvas size: invalid parameters! Width and height must be at least 10.0m each."));
        return;
    }

    if ((newNorth_ == oldNorth_ && newSouth_ == oldSouth_ && newEast_ == oldEast_ && newWest_ == oldWest_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Canvas size: ignored! No change."));
        return;
    }

    if ((!projectData))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Canvas size: invalid parameters! No ProjectData given or no changes."));
        return;
    }

    setValid();
    setText(QObject::tr("Set Canvas size"));
}

/*! \brief .
*
*/
SetProjectDimensionsCommand::~SetProjectDimensionsCommand()
{
}

/*! \brief .
*
*/
void
SetProjectDimensionsCommand::redo()
{
    // Set Variables //
    //
    projectData_->setNorth(newNorth_);
    projectData_->setSouth(newSouth_);
    projectData_->setEast(newEast_);
    projectData_->setWest(newWest_);

    setRedone();
}

/*! \brief .
*
*/
void
SetProjectDimensionsCommand::undo()
{
    // Reset Variables //
    //
    projectData_->setNorth(oldNorth_);
    projectData_->setSouth(oldSouth_);
    projectData_->setEast(oldEast_);
    projectData_->setWest(oldWest_);

    setUndone();
}

//#######################//
// SetProjectNameCommand //
//#######################//

SetProjectNameCommand::SetProjectNameCommand(ProjectData *projectData, const QString &name, DataCommand *parent)
    : DataCommand(parent)
    , projectData_(projectData)
    , newName_(name)
{
    oldName_ = projectData_->getName();

    // Check for validity //
    //
    if ((!projectData)
        || (newName_ == oldName_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set project name: invalid parameters! No ProjectData given or no changes"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set project name"));
    }
}

/*! \brief .
*
*/
SetProjectNameCommand::~SetProjectNameCommand()
{
}

/*! \brief .
*
*/
void
SetProjectNameCommand::redo()
{
    projectData_->setName(newName_);

    setRedone();
}

/*! \brief .
*
*/
void
SetProjectNameCommand::undo()
{
    projectData_->setName(oldName_);

    setUndone();
}

//#######################//
// SetProjectVersionCommand //
//#######################//

SetProjectVersionCommand::SetProjectVersionCommand(ProjectData *projectData, double version, DataCommand *parent)
    : DataCommand(parent)
    , projectData_(projectData)
    , newVersion_(version)
{
    oldVersion_ = projectData_->getVersion();

    // Check for validity //
    //
    if ((!projectData)
        || (newVersion_ == oldVersion_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set project version: invalid parameters! No ProjectData given or no changes"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set project version"));
    }
}

/*! \brief .
*
*/
SetProjectVersionCommand::~SetProjectVersionCommand()
{
}

/*! \brief .
*
*/
void
SetProjectVersionCommand::redo()
{
    projectData_->setVersion(newVersion_);

    setRedone();
}

/*! \brief .
*
*/
void
SetProjectVersionCommand::undo()
{
    projectData_->setVersion(oldVersion_);

    setUndone();
}

//#######################//
// SetProjectDateCommand //
//#######################//

SetProjectDateCommand::SetProjectDateCommand(ProjectData *projectData, const QString &date, DataCommand *parent)
    : DataCommand(parent)
    , projectData_(projectData)
    , newDate_(date)
{
    oldDate_ = projectData_->getDate();

    // Check for validity //
    //
    if ((!projectData)
        || (newDate_ == oldDate_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set project date: invalid parameters! No ProjectData given or no changes"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set project date"));
    }
}

/*! \brief .
*
*/
SetProjectDateCommand::~SetProjectDateCommand()
{
}

/*! \brief .
*
*/
void
SetProjectDateCommand::redo()
{
    projectData_->setDate(newDate_);

    setRedone();
}

/*! \brief .
*
*/
void
SetProjectDateCommand::undo()
{
    projectData_->setDate(oldDate_);

    setUndone();
}
