/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   17.03.2010
**
**************************************************************************/

#include "typesectioncommands.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/typesection.hpp"

#define MINTYPESECTIONDISTANCE 1.0

//#######################//
// AddTypeSectionCommand //
//#######################//

AddTypeSectionCommand::AddTypeSectionCommand(RSystemElementRoad *road, TypeSection *typeSection, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , newTypeSection_(typeSection)
{
    // Check for validity //
    //
    double s = typeSection->getSStart();
    TypeSection *sqeezedTypeSection = road->getTypeSection(s);
    double sStart = sqeezedTypeSection->getSStart();
    double sEnd = sqeezedTypeSection->getSEnd();
    if ((s - sStart < MINTYPESECTIONDISTANCE) || (sEnd - s < MINTYPESECTIONDISTANCE))
    {
        setInvalid(); // Invalid because to close to an other type section.
        setText(QObject::tr("Add Type Section (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Add Type Section"));
    }
}

/*! \brief Deletes the TypeSection if the command has been undone.
*
* Otherwise the RSystemElementRoad now ownes the TypeSection
* and takes care of it.
*/
AddTypeSectionCommand::~AddTypeSectionCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete newTypeSection_;
    }
    else
    {
        // nothing to be done
        // the typeSection is now owned by the road
    }
}

/*! \brief Adds the TypeSection to the RSystemElementRoad.
*
* This also notifies the RSystemElementRoad.
*/
void
AddTypeSectionCommand::redo()
{
    // Set //
    //
    road_->addTypeSection(newTypeSection_);

    setRedone();
}

/*! \brief Removes the TypeSection from the RSystemElementRoad.
*
* This also notifies the RSystemElementRoad.
*/
void
AddTypeSectionCommand::undo()
{
    // Reset //
    //
    road_->delTypeSection(newTypeSection_);

    setUndone();
}

//##########################//
// DeleteTypeSectionCommand //
//##########################//

DeleteTypeSectionCommand::DeleteTypeSectionCommand(RSystemElementRoad *road, TypeSection *typeSection, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , deletedTypeSection_(typeSection)
{
    if (!road)
    {
        setInvalid(); // Invalid because first section can not be deleted!
        setText(QObject::tr("No road (invalid!)"));

        return;
    }

    // Check for validity //
    //
    if (road->getTypeSection(0.0) == typeSection)
    {
        setInvalid(); // Invalid because first section can not be deleted!
        setText(QObject::tr("Delete Type Section (invalid!)"));

        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Delete Type Section"));
    }
}

/*! \brief Deletes the TypeSection if the command has not been undone.
*
* Otherwise the RSystemElementRoad now ownes the TypeSection
* and takes care of it.
*/
DeleteTypeSectionCommand::~DeleteTypeSectionCommand()
{
    if (isValid()) // do not delete anything if this command hasn't been run.
    {
        // Clean up //
        //
        if (isUndone())
        {
            // nothing to be done
            // the typeSection is now again owned by the road
        }
        else
        {
            delete deletedTypeSection_;
        }
    }
}

/*! \brief Removes the TypeSection from the RSystemElementRoad.
*
* This also notifies the RSystemElementRoad.
*/
void
DeleteTypeSectionCommand::redo()
{
    if (!isValid())
        return;

    // Set //
    //
    road_->delTypeSection(deletedTypeSection_);

    setRedone();
}

/*! \brief Adds the TypeSection to the RSystemElementRoad.
*
* This also notifies the RSystemElementRoad.
*/
void
DeleteTypeSectionCommand::undo()
{
    if (!isValid())
        return;

    // Reset //
    //
    road_->addTypeSection(deletedTypeSection_);

    setUndone();
}

//###########################//
// SetTypeTypeSectionCommand //
//###########################//

SetTypeTypeSectionCommand::SetTypeTypeSectionCommand(TypeSection *typeSection, TypeSection::RoadType newRoadType, DataCommand *parent)
    : DataCommand(parent)
    , typeSection_(typeSection)
{
    // Check for validity //
    //
    if (typeSection->getRoadType() == newRoadType)
    {
        setInvalid(); // Invalid because new RoadType is the same as the old
        setText(QObject::tr("Set Road Type (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Type"));
    }

    // Road Type //
    //
    newRoadType_ = newRoadType;
    oldRoadType_ = typeSection->getRoadType();
}

/*! \brief Does nothing.
*/
SetTypeTypeSectionCommand::~SetTypeTypeSectionCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done
    }
    else
    {
        // nothing to be done
    }
}

/*! \brief Sets the Road Type of a TypeSection.
*/
void
SetTypeTypeSectionCommand::redo()
{
    // Set //
    //
    typeSection_->setRoadType(newRoadType_);

    setRedone();
}

/*! \brief Resets the Road Type of a TypeSection.
*/
void
SetTypeTypeSectionCommand::undo()
{
    // Reset //
    //
    typeSection_->setRoadType(oldRoadType_);

    setUndone();
}
