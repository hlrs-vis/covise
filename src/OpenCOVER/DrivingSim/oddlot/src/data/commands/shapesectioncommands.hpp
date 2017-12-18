/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef SHAPESECTIONCOMMANDS_HPP
#define SHAPESECTIONCOMMANDS_HPP

// 1100

#include "datacommand.hpp"

class RSystemElementRoad;
class ShapeSection;


//################//
// Split          //
//################//

class SplitShapeSectionCommand : public DataCommand
{
public:
    explicit SplitShapeSectionCommand(ShapeSection *shapeSection, double splitPos, DataCommand *parent = NULL);
    virtual ~SplitShapeSectionCommand();

    virtual int id() const
    {
        return 0x1101;
    }

    virtual void undo();
    virtual void redo();

private:
    SplitShapeSectionCommand(); /* not allowed */
    SplitShapeSectionCommand(const SplitShapeSectionCommand &); /* not allowed */
    SplitShapeSectionCommand &operator=(const SplitShapeSectionCommand &); /* not allowed */

private:
    ShapeSection *oldSection_;
    ShapeSection *newSection_;

    double splitPos_;
};

//################//
// Merge          //
//################//

class MergeShapeSectionCommand : public DataCommand
{
public:
    explicit MergeShapeSectionCommand(ShapeSection *shapeSectionLow, ShapeSection *shapeSectionHigh, DataCommand *parent = NULL);
    virtual ~MergeShapeSectionCommand();

    virtual int id() const
    {
        return 0x1102;
    }

    virtual void undo();
    virtual void redo();

private:
    MergeShapeSectionCommand(); /* not allowed */
    MergeShapeSectionCommand(const MergeShapeSectionCommand &); /* not allowed */
    MergeShapeSectionCommand &operator=(const MergeShapeSectionCommand &); /* not allowed */

private:
    ShapeSection *oldSectionLow_;
    ShapeSection *oldSectionHigh_;
    ShapeSection *newSection_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Remove          //
//################//

class RemoveShapeSectionCommand : public DataCommand
{
public:
    explicit RemoveShapeSectionCommand(ShapeSection *shapeSection, DataCommand *parent = NULL);
    virtual ~RemoveShapeSectionCommand();

    virtual int id() const
    {
        return 0x1104;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveShapeSectionCommand(); /* not allowed */
    RemoveShapeSectionCommand(const RemoveShapeSectionCommand &); /* not allowed */
    RemoveShapeSectionCommand &operator=(const RemoveShapeSectionCommand &); /* not allowed */

private:
    ShapeSection *oldSectionLow_;
    ShapeSection *oldSectionMiddle_;
    ShapeSection *oldSectionHigh_;

    ShapeSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;
};


#endif // SHAPESECTIONCOMMANDS_HPP
