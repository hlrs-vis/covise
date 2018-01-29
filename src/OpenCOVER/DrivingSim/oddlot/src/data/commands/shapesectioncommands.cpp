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
// RemoveShapeSectionCommand //
//#######################//

RemoveShapeSectionCommand::RemoveShapeSectionCommand(ShapeSection *shapeSection, DataCommand *parent)
    : DataCommand(parent)
    , oldSection_(shapeSection)
{
    parentRoad_ = oldSection_->getParentRoad();
    if (!parentRoad_ || (parentRoad_->getShapeSections().size() <= 1))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ShapeSection. "));
        return;
    }


    // Done //
    //
    setValid();
    setText(QObject::tr("Remove ShapeSection")); 
}

/*! \brief .
*
*/
RemoveShapeSectionCommand::~RemoveShapeSectionCommand()
{
    if (isUndone())
    {
    }
    else
    {
        delete oldSection_;
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
    parentRoad_->delShapeSection(oldSection_);

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
    parentRoad_->addShapeSection(oldSection_);

    setUndone();
}

//##################################//
// PasteLateralShapeSectionsCommand //
//##################################//

PasteLateralShapeSectionsCommand::PasteLateralShapeSectionsCommand(ShapeSection *shapeSection, QMap<double, PolynomialLateralSection *> oldSections, QMap<double, PolynomialLateralSection *> newSections, DataCommand *parent)
	: DataCommand(parent)
	, shapeSection_(shapeSection)
	, oldSections_(oldSections)
{
	// Check for validity //
	//
	if (!shapeSection)
	{
		setInvalid(); // Invalid
		setText(QObject::tr("Paste LateralShapeSections (invalid!)"));
		return;
	}
	else
	{
		setValid();
		setText(QObject::tr("Paste LateralShapeSections"));
	}

	foreach(PolynomialLateralSection *poly, newSections)
	{
		newSections_.insert(poly->getTStart(), poly->getClone());
	}

}

/*! \brief .
*
*/
PasteLateralShapeSectionsCommand::~PasteLateralShapeSectionsCommand()
{
	if (isUndone())
	{
		newSections_.clear();
	}
	else
	{
		oldSections_.clear();
	}
}

/*! \brief .
*
*/
void
PasteLateralShapeSectionsCommand::redo()
{
	// set new polynomials //
	//
	QMap<double, PolynomialLateralSection *>::ConstIterator it = oldSections_.constBegin();
	while (it != oldSections_.constEnd())
	{
		shapeSection_->delShape(it.key());
		it++;
	}

	it = newSections_.constBegin();
	while (it != newSections_.constEnd())
	{
		shapeSection_->addShape(it.key(), it.value());
		it++;
	}

	setRedone();
}

/*! \brief .
*
*/
void
PasteLateralShapeSectionsCommand::undo()
{
	// set old polynomials //
	//
	QMap<double, PolynomialLateralSection *>::ConstIterator it = newSections_.constBegin();
	while (it != newSections_.constEnd())
	{
		shapeSection_->delShape(it.key());
		it++;
	}

	it = oldSections_.constBegin();
	while (it != oldSections_.constEnd())
	{
		shapeSection_->addShape(it.key(), it.value());
		it++;
	}

	setUndone();
}

