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

#ifndef TYPESECTIONCOMMANDS_HPP
#define TYPESECTIONCOMMANDS_HPP

// 2200

#include "datacommand.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"

class RSystemElementRoad;

//################//
// Add            //
//################//

class AddTypeSectionCommand : public DataCommand
{
public:
    explicit AddTypeSectionCommand(RSystemElementRoad *road, TypeSection *typeSection, DataCommand *parent = NULL);
    virtual ~AddTypeSectionCommand();

    virtual int id() const
    {
        return 0x2201;
    }

    virtual void undo();
    virtual void redo();

private:
    AddTypeSectionCommand(); /* not allowed */
    AddTypeSectionCommand(const AddTypeSectionCommand &); /* not allowed */
    AddTypeSectionCommand &operator=(const AddTypeSectionCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;

    TypeSection *newTypeSection_;
};

//################//
// Delete         //
//################//

class DeleteTypeSectionCommand : public DataCommand
{
public:
    explicit DeleteTypeSectionCommand(RSystemElementRoad *road, TypeSection *typeSection, DataCommand *parent = NULL);
    virtual ~DeleteTypeSectionCommand();

    virtual int id() const
    {
        return 0x2202;
    }

    virtual void undo();
    virtual void redo();

private:
    DeleteTypeSectionCommand(); /* not allowed */
    DeleteTypeSectionCommand(const DeleteTypeSectionCommand &); /* not allowed */
    DeleteTypeSectionCommand &operator=(const DeleteTypeSectionCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;

    TypeSection *deletedTypeSection_;
};

//################//
// SetType        //
//################//

class SetTypeTypeSectionCommand : public DataCommand
{
public:
    explicit SetTypeTypeSectionCommand(TypeSection *typeSection, TypeSection::RoadType newRoadType, DataCommand *parent = NULL);
    virtual ~SetTypeTypeSectionCommand();

    virtual int id() const
    {
        return 0x2204;
    }

    virtual void undo();
    virtual void redo();

private:
    SetTypeTypeSectionCommand(); /* not allowed */
    SetTypeTypeSectionCommand(const SetTypeTypeSectionCommand &); /* not allowed */
    SetTypeTypeSectionCommand &operator=(const SetTypeTypeSectionCommand &); /* not allowed */

private:
    TypeSection *typeSection_;

    TypeSection::RoadType newRoadType_;
    TypeSection::RoadType oldRoadType_;
};

//################//
// SetSpeedRecord        //
//################//

class SetSpeedTypeSectionCommand : public DataCommand
{
public:
    explicit SetSpeedTypeSectionCommand(TypeSection *typeSection, double maxSpeed, DataCommand *parent = NULL);
    virtual ~SetSpeedTypeSectionCommand();

    virtual int id() const
    {
        return 0x2205;
    }

    virtual void undo();
    virtual void redo();

private:
    SetSpeedTypeSectionCommand(); /* not allowed */
    SetSpeedTypeSectionCommand(const SetSpeedTypeSectionCommand &); /* not allowed */
    SetSpeedTypeSectionCommand &operator=(const SetSpeedTypeSectionCommand &); /* not allowed */

private:
    TypeSection *typeSection_;

    SpeedRecord *newSpeedRecord;
    SpeedRecord *oldSpeedRecord;
};

#endif // TYPESECTIONCOMMANDS_HPP
