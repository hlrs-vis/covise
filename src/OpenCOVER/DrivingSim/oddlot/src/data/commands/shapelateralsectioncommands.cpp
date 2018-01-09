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

#include "shapelateralsectioncommands.hpp"

 // Data //
 //
#include "src/data/roadsystem/sections/shapesection.hpp"



//#######################//
// MovePointLateralShapeSectionCommand //
//#######################//

MovePointLateralShapeSectionCommand::MovePointLateralShapeSectionCommand(ShapeSection *shapeSection, SplineControlPoint *corner, const QList<QPointF> &points, DataCommand *parent)
	: DataCommand(parent)
	, shapeSection_(shapeSection)
	, corner_(corner)
	, newPoints_(points)
{
	// Check for validity //
	//
	if (points.size() == 0)
	{
		setInvalid(); // Invalid
		setText(QObject::tr("MovePoint LateralShapeSection (invalid!)"));
		return;
	}
	else
	{
		setValid();
		setText(QObject::tr("MovePoint LateralShapeSection"));
	}


	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		oldPoints_.append(lateralSection->getRealPointLow()->getPoint());
		oldPoints_.append(lateralSection->getRealPointHigh()->getPoint());
	}

}

/*! \brief .
*
*/
MovePointLateralShapeSectionCommand::~MovePointLateralShapeSectionCommand()
{
	if (isUndone())
	{

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
MovePointLateralShapeSectionCommand::redo()
{
	// set new points //
	//
	int i = 0;
	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		QPointF p0 = newPoints_.at(i++);
		QPointF p1 = newPoints_.at(i++);
		if (abs(p0.x() - lateralSection->getTStart()) > NUMERICAL_ZERO6)
		{
			shapeSection_->moveLateralSection(lateralSection, p0.x());
		}
		lateralSection->setControlPoints(p0, p1);    // , p2, p3);
	}
	shapeSection_->calculateShapeParameters();

    setRedone();
}

/*! \brief .
*
*/
void
MovePointLateralShapeSectionCommand::undo()
{
    // revert to old points //
    //
	int i = 0;
	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		QPointF p0 = oldPoints_.at(i++);
		QPointF p1 = oldPoints_.at(i++);
		if (abs(p0.x() - lateralSection->getRealPointLow()->getPoint().x()) > NUMERICAL_ZERO6)
		{
			lateralSection->getParentSection()->moveLateralSection(lateralSection, p0.x());
		}
		lateralSection->setControlPoints(p0, p1);   // , p2, p3);
	}
	shapeSection_->calculateShapeParameters();


    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
MovePointLateralShapeSectionCommand::mergeWith(const QUndoCommand *other)
{
	// Check Ids //
	//
	if (other->id() != id())
	{
		return false;
	}

	const MovePointLateralShapeSectionCommand *command = static_cast<const MovePointLateralShapeSectionCommand *>(other);

	// Check parameters //
	//
	if ((corner_ != command->corner_) || (shapeSection_ != command->shapeSection_))
	{
		return false;
	}

	// Success //
	//
	newPoints_ = command->newPoints_; // adjust to new point, then let the undostack kill the new command

	return true;
}

//#######################//
// AddLateralShapeSectionCommand //
//#######################//

AddLateralShapeSectionCommand::AddLateralShapeSectionCommand(ShapeSection *shapeSection, PolynomialLateralSection *newSection, const QList<QPointF> &points, DataCommand *parent)
	: DataCommand(parent)
	, shapeSection_(shapeSection)
	, newSection_(newSection)
	, newPoints_(points)
{
	// Check for validity //
	//
	if (points.size() == 0)
	{
		setInvalid(); // Invalid
		setText(QObject::tr("Add LateralShapeSection (invalid!)"));
		return;
	}
	else
	{
		setValid();
		setText(QObject::tr("Add LateralShapeSection"));
	}


	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		oldPoints_.append(lateralSection->getRealPointLow()->getPoint());
		oldPoints_.append(lateralSection->getRealPointHigh()->getPoint());
	}

}

/*! \brief .
*
*/
AddLateralShapeSectionCommand::~AddLateralShapeSectionCommand()
{
	if (isUndone())
	{

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
AddLateralShapeSectionCommand::redo()
{
	// set new points //
	//
	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection_->getPolynomialLateralSections().constBegin();
	int i = 0;
	while ((it != shapeSection_->getPolynomialLateralSections().constEnd()) && (it.value() != shapeSection_->getPolynomialLateralSectionNext(newSection_->getTStart())))
	{
		it.value()->setControlPoints(newPoints_.at(i), newPoints_.at(i + 1)); 
		it++;
		i += 2;
	}

	newSection_->setControlPoints(newPoints_.at(i), newPoints_.at(i + 1)); 
	i += 2;

	while (it != shapeSection_->getPolynomialLateralSections().constEnd())
	{
		it.value()->setControlPoints(newPoints_.at(i), newPoints_.at(i + 1)); 
		it++;
		i += 2;
	} 

	shapeSection_->addShape(newSection_->getTStart(), newSection_);

	shapeSection_->calculateShapeParameters();

	setRedone();
}

/*! \brief .
*
*/
void
AddLateralShapeSectionCommand::undo()
{
	shapeSection_->delShape(newSection_->getTStart());

	// revert to old points //
	//
	int i = 0;
	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		lateralSection->setControlPoints(oldPoints_.at(i), oldPoints_.at(i + 1)); 
		i += 2;//		
	}

	shapeSection_->calculateShapeParameters();

	setUndone();
}

//##################################//
// DeleteLateralShapeSectionCommand //
//#################################//

DeleteLateralShapeSectionCommand::DeleteLateralShapeSectionCommand(ShapeSection *shapeSection, PolynomialLateralSection *polySection, const QList<QPointF> &points, DataCommand *parent)
	: DataCommand(parent)
	, shapeSection_(shapeSection)
	, polySection_(polySection)
	, newPoints_(points)
{
	// Check for validity //
	//
	if (points.size() == 0)
	{
		setInvalid(); // Invalid
		setText(QObject::tr("Delete LateralShapeSection (invalid!)"));
		return;
	}
	else
	{
		setValid();
		setText(QObject::tr("Delete LateralShapeSection"));
	}


	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		oldPoints_.append(lateralSection->getRealPointLow()->getPoint());
		oldPoints_.append(lateralSection->getRealPointHigh()->getPoint());
	}

}

/*! \brief .
*
*/
DeleteLateralShapeSectionCommand::~DeleteLateralShapeSectionCommand()
{
	if (isUndone())
	{

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
DeleteLateralShapeSectionCommand::undo()
{
	// set new points //
	//
	QMap<double, PolynomialLateralSection *>::ConstIterator it = shapeSection_->getPolynomialLateralSections().constBegin();
	int i = 0;
	while ((it != shapeSection_->getPolynomialLateralSections().constEnd()) && (it.value() != shapeSection_->getPolynomialLateralSectionNext(polySection_->getTStart())))
	{
		it.value()->setControlPoints(oldPoints_.at(i), oldPoints_.at(i + 1)); 
		it++;
		i += 2;//		i += 4;
	}

	polySection_->setControlPoints(oldPoints_.at(i), oldPoints_.at(i + 1)); 
	i += 2;//		i += 4;

	while (it != shapeSection_->getPolynomialLateralSections().constEnd())
	{
		it.value()->setControlPoints(oldPoints_.at(i), oldPoints_.at(i + 1)); 
		it++;
		i += 2;//		i += 4;
	}

	shapeSection_->addShape(polySection_->getTStart(), polySection_);

	shapeSection_->calculateShapeParameters();

	setUndone();
}

/*! \brief .
*
*/
void
DeleteLateralShapeSectionCommand::redo()
{
	shapeSection_->delShape(polySection_->getTStart());

	// revert to old points //
	//
	int i = 0;
	foreach(PolynomialLateralSection *lateralSection, shapeSection_->getPolynomialLateralSections())
	{
		lateralSection->setControlPoints(newPoints_.at(i), newPoints_.at(i + 1)); 
		i += 2;//		i += 4;
	}

	shapeSection_->calculateShapeParameters();

	setRedone();
}




