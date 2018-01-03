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

#include "shapesectioncommands.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"

// Utils //
//
#include "math.h"

#define MIN_SHAPESECTION_LENGTH 1.0

//#######################//
// SplitShapeSectionCommand //
//#######################//

SplitShapeSectionCommand::SplitShapeSectionCommand(ShapeSection *shapeSection, double splitPos, DataCommand *parent)
    : DataCommand(parent)
    , oldSection_(shapeSection)
    , newSection_(NULL)
    , splitPos_(splitPos)
{
    // Check for validity //
    //
 //   if ((oldSection_->getDegree() > 1) // only lines allowed
 //       || (fabs(splitPos_ - oldSection_->getSStart()) < MIN_SHAPESECTION_LENGTH)
	if ((fabs(splitPos_ - oldSection_->getSStart()) < MIN_SHAPESECTION_LENGTH)
        || (fabs(oldSection_->getSEnd() - splitPos_) < MIN_SHAPESECTION_LENGTH) // minimum length 1.0 m
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Split ShapeSection (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Split ShapeSection"));
    }

    // New section //
    //

	newSection_ = new ShapeSection(splitPos, oldSection_->getParentRoad()->getMinWidth(splitPos));
}

/*! \brief .
*
*/
SplitShapeSectionCommand::~SplitShapeSectionCommand()
{
    if (isUndone())
    {
        delete newSection_;
    }
    else
    {
        // nothing to be done
        // the section is now owned by the road
    }
}

/*! \brief .
*
*/
void
SplitShapeSectionCommand::redo()
{
    // Add section to road //
    //
    oldSection_->getParentRoad()->addShapeSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
SplitShapeSectionCommand::undo()
{
    // Remove section from road //
    //
    newSection_->getParentRoad()->delShapeSection(newSection_);

    setUndone();
}

//#######################//
// MergeShapeSectionCommand //
//#######################//

MergeShapeSectionCommand::MergeShapeSectionCommand(ShapeSection *shapeSectionLow, ShapeSection *shapeSectionHigh, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(shapeSectionLow)
    , oldSectionHigh_(shapeSectionHigh)
    , newSection_(NULL)
{
    parentRoad_ = shapeSectionLow->getParentRoad();

    // Check for validity //
    //
/*    if ((oldSectionLow_->getDegree() > 1)
        || (oldSectionHigh_->getDegree() > 1) // only lines allowed
	  if  (oldSectionHigh_->getDegree() > 1)
        || (parentRoad_ != shapeSectionHigh->getParentRoad()) // not the same parents
        || shapeSectionHigh != parentRoad_->getShapeSection(shapeSectionLow->getSEnd()) // not consecutive
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Merge ShapeSection (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Merge ShapeSection"));
    }

    // New section //
    //
    double deltaLength = shapeSectionHigh->getSEnd() - shapeSectionLow->getSStart();
    double deltaHeight = shapeSectionHigh->getShapeDegrees(shapeSectionHigh->getSEnd()) - shapeSectionLow->getShapeDegrees(shapeSectionLow->getSStart());
    newSection_ = new ShapeSection(oldSectionLow_->getSide(), oldSectionLow_->getSStart(), oldSectionLow_->getA(), deltaHeight / deltaLength, 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected() || oldSectionLow_->isElementSelected())
    {
        newSection_->setElementSelected(true); // keep selection
    } */
}

/*! \brief .
*
*/
MergeShapeSectionCommand::~MergeShapeSectionCommand()
{
    if (isUndone())
    {
        delete newSection_;
    }
    else
    {
        delete oldSectionLow_;
        delete oldSectionHigh_;
    }
}

/*! \brief .
*
*/
void
MergeShapeSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delShapeSection(oldSectionLow_);
    parentRoad_->delShapeSection(oldSectionHigh_);

    parentRoad_->addShapeSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
MergeShapeSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delShapeSection(newSection_);

    parentRoad_->addShapeSection(oldSectionLow_);
    parentRoad_->addShapeSection(oldSectionHigh_);

    setUndone();
}

//#######################//
// RemoveShapeSectionCommand //
//#######################//

RemoveShapeSectionCommand::RemoveShapeSectionCommand(ShapeSection *shapeSection, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(NULL)
    , oldSectionMiddle_(shapeSection)
    , oldSectionHigh_(NULL)
    , newSectionHigh_(NULL)
{
 /*   parentRoad_ = oldSectionMiddle_->getParentRoad();
    if (!parentRoad_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ShapeSection. No ParentRoad."));
        return;
    }

    oldSectionLow_ = parentRoad_->getShapeSectionBefore(oldSectionMiddle_->getSStart());
    if (!oldSectionLow_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ShapeSection. No section to the left."));
        return;
    }

    oldSectionHigh_ = parentRoad_->getShapeSection(oldSectionMiddle_->getSEnd());
    if (!oldSectionHigh_ || (oldSectionHigh_ == oldSectionMiddle_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ShapeSection. No section to the right."));
        return;
    }

    if (oldSectionLow_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ShapeSection. Section to the left is not linear."));
        return;
    }

    if (oldSectionHigh_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ShapeSection. Section to the right is not linear."));
        return;
    }

    // Intersection point //
    //
    double sLow = oldSectionLow_->getSStart();
    double s = 0.0;
    if (fabs(oldSectionLow_->getB() - oldSectionHigh_->getB()) < NUMERICAL_ZERO8)
    {
        // Parallel //
        //
        if (fabs(oldSectionLow_->getA() - oldSectionHigh_->getA()) < NUMERICAL_ZERO8)
        {
            s = 0.5 * (oldSectionLow_->getSEnd() + oldSectionHigh_->getSStart()); // meet half way
        }
        else
        {
            setInvalid(); // Invalid
            setText(QObject::tr("Cannot remove ShapeSection. Sections to the left and right do not intersect."));
            return;
        }
    }
    else
    {
        // Not parallel //
        //
        s = sLow + (oldSectionHigh_->getShapeDegrees(sLow) - oldSectionLow_->getShapeDegrees(sLow)) / (oldSectionLow_->getB() - oldSectionHigh_->getB());
        if ((s - sLow < MIN_SHAPESECTION_LENGTH)
            || (oldSectionHigh_->getSEnd() - s < MIN_SHAPESECTION_LENGTH))
        {
            setInvalid(); // Invalid
            setText(QObject::tr("Cannot remove ShapeSection. Sections to the left and right do not intersect."));
            return;
        }
    }
    //	qDebug() << "s: " << s << ", sLow: " << sLow;

    // New section //
    //
    newSectionHigh_ = new ShapeSection(oldSectionLow_->getSide(), s, oldSectionLow_->getShapeDegrees(s), oldSectionHigh_->getB(), 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected() || oldSectionMiddle_->isElementSelected())
    {
        newSectionHigh_->setElementSelected(true); // keep selection
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Remove ShapeSection")); */
}

/*! \brief .
*
*/
RemoveShapeSectionCommand::~RemoveShapeSectionCommand()
{
    if (isUndone())
    {
        delete newSectionHigh_;
    }
    else
    {
        delete oldSectionMiddle_;
        delete oldSectionHigh_;
    }
}

/*! \brief .
*
*/
void
RemoveShapeSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delShapeSection(oldSectionMiddle_);
    parentRoad_->delShapeSection(oldSectionHigh_);

    parentRoad_->addShapeSection(newSectionHigh_);

    setRedone();
}

/*! \brief .
*
*/
void
RemoveShapeSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delShapeSection(newSectionHigh_);

    parentRoad_->addShapeSection(oldSectionMiddle_);
    parentRoad_->addShapeSection(oldSectionHigh_);

    setUndone();
}
