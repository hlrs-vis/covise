/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#include "lateralsectioncommands.hpp"

#include "src/data/roadsystem/lateralsections/lateralsection.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"


//###############//
// MoveLateralSectionCommand //
//###############//

MoveLateralSectionCommand::MoveLateralSectionCommand(LateralSection *lateralSection, double t,  DataCommand *parent)
    : DataCommand(parent)
    , lateralSection_(lateralSection)
    , newT_(t)
{
    // Old s //
    //
    oldT_ = lateralSection->getTStart();

    // Check for validity //
    //

    if (!lateralSection
        || (newT_ == oldT_)) // no change
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Move Lateral Section (invalid!)"));
		qDebug() << "false";
        return;
    }


    // Done //
    //
    setValid();
    setText(QObject::tr("Move Lateral Section"));
}

/*! \brief .
*
*/
MoveLateralSectionCommand::~MoveLateralSectionCommand()
{
}

/*! \brief .
*
*/
void
MoveLateralSectionCommand::redo()
{
	lateralSection_->getParentSection()->moveLateralSection(lateralSection_, newT_);

    setRedone();
}

/*! \brief
*
*/
void
MoveLateralSectionCommand::undo()
{
	lateralSection_->getParentSection()->moveLateralSection(lateralSection_, oldT_);

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
MoveLateralSectionCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
		qDebug() << "false";
        return false;
    }

    const MoveLateralSectionCommand *command = static_cast<const MoveLateralSectionCommand *>(other);

    // Check parameters //
    //
    if (lateralSection_ != command->lateralSection_)
    {
		qDebug() << "false";
        return false;
    }

    // Success //
    //
    newT_ = command->newT_; // adjust to new point, then let the undostack kill the new command

    return true;
}
