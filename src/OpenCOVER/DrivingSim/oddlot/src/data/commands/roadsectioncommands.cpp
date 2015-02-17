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

#include "roadsectioncommands.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/roadsection.hpp"

#define MINROADSECTIONLENGTH 1.0

//###############//
// MoveRoadSectionCommand //
//###############//

MoveRoadSectionCommand::MoveRoadSectionCommand(RoadSection *roadSection, double s, RSystemElementRoad::DRoadSectionType sectionType, DataCommand *parent)
    : DataCommand(parent)
    , roadSection_(roadSection)
    , newS_(s)
    , sectionType_(sectionType)
{
    // Old s //
    //
    oldS_ = roadSection->getSStart();

    // Check for validity //
    //
    if ((sectionType_ != RSystemElementRoad::DRS_TypeSection)
        && (sectionType_ != RSystemElementRoad::DRS_SuperelevationSection)
        && (sectionType_ != RSystemElementRoad::DRS_CrossfallSection)
        && (sectionType_ != RSystemElementRoad::DRS_SignalSection)
        && (sectionType_ != RSystemElementRoad::DRS_ObjectSection)
        && (sectionType_ != RSystemElementRoad::DRS_BridgeSection)
        && (sectionType_ != RSystemElementRoad::DRS_ElevationSection))
    {
        setInvalid();
        setText(QObject::tr("You cannot move a section of this section type."));
        return;
    }

    if (!roadSection
        || (newS_ == oldS_)) // no change
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Move Road Section (invalid!)"));
        return;
    }

    if ((sectionType != RSystemElementRoad::DRS_SignalSection) && (sectionType != RSystemElementRoad::DRS_ObjectSection) && (sectionType != RSystemElementRoad::DRS_BridgeSection))
    {
        if ((roadSection->getSStart() < NUMERICAL_ZERO7) // first section of the road
            || ((roadSection->getSEnd() - newS_ < MINROADSECTIONLENGTH) // min length at end
            || (newS_ - roadSection->getParentRoad()->getRoadSectionBefore(roadSection->getSStart(), sectionType_)->getSStart() < MINROADSECTIONLENGTH)))
        {
            setInvalid(); // Invalid
            setText(QObject::tr("Move Road Section (invalid!)"));
            return;
        }
    }

    if (((sectionType == RSystemElementRoad::DRS_SignalSection) || (sectionType == RSystemElementRoad::DRS_ObjectSection) || (sectionType == RSystemElementRoad::DRS_BridgeSection)) && (newS_ > roadSection->getParentRoad()->getLength()))
    {
        newS_ = roadSection->getParentRoad()->getLength();
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Move Road Section"));
}

/*! \brief .
*
*/
MoveRoadSectionCommand::~MoveRoadSectionCommand()
{
}

/*! \brief .
*
*/
void
MoveRoadSectionCommand::redo()
{
    roadSection_->getParentRoad()->moveRoadSection(roadSection_, newS_, sectionType_);

    setRedone();
}

/*! \brief
*
*/
void
MoveRoadSectionCommand::undo()
{
    roadSection_->getParentRoad()->moveRoadSection(roadSection_, oldS_, sectionType_);

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
MoveRoadSectionCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const MoveRoadSectionCommand *command = static_cast<const MoveRoadSectionCommand *>(other);

    // Check parameters //
    //
    if (roadSection_ != command->roadSection_)
    {
        return false;
    }

    // Success //
    //
    newS_ = command->newS_; // adjust to new point, then let the undostack kill the new command

    return true;
}
