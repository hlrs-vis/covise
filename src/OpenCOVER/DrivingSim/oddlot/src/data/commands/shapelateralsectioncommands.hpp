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

#ifndef SHAPELATERALSECTIONCOMMANDS_HPP
#define SHAPELATERALSECTIONCOMMANDS_HPP

// 1100

#include "datacommand.hpp"

// Data //
//
#include "src/data/roadsystem/lateralSections/polynomiallateralsection.hpp"

class ShapeSection;
class SplineControlPoint;


//################//
// Move points    //
//################//

class MovePointLateralShapeSectionCommand : public DataCommand
{
public:
    explicit MovePointLateralShapeSectionCommand(ShapeSection *shapeSection, SplineControlPoint *corner, const QList<QPointF> &points, DataCommand *parent = NULL);
    virtual ~MovePointLateralShapeSectionCommand();

    virtual int id() const
    {
        return 0x1301;
    }

    virtual void undo();
    virtual void redo();

	virtual bool mergeWith(const QUndoCommand *other);

private:
    MovePointLateralShapeSectionCommand(); /* not allowed */
    MovePointLateralShapeSectionCommand(const MovePointLateralShapeSectionCommand &); /* not allowed */
    MovePointLateralShapeSectionCommand &operator=(const MovePointLateralShapeSectionCommand &); /* not allowed */

private:
	ShapeSection *shapeSection_;
	SplineControlPoint *corner_;

	QList<QPointF> oldPoints_;
	QList<QPointF> newPoints_;
};

//######################//
// Add lateral shape    //
//#####################//

class AddLateralShapeSectionCommand : public DataCommand
{
public:
	explicit AddLateralShapeSectionCommand(ShapeSection *shapeSection, PolynomialLateralSection *newSection, const QList<QPointF> &points, DataCommand *parent = NULL);
	virtual ~AddLateralShapeSectionCommand();

	virtual int id() const
	{
		return 0x1302;
	}

	virtual void undo();
	virtual void redo();

private:
	AddLateralShapeSectionCommand(); /* not allowed */
	AddLateralShapeSectionCommand(const AddLateralShapeSectionCommand &); /* not allowed */
	AddLateralShapeSectionCommand &operator=(const AddLateralShapeSectionCommand &); /* not allowed */

private:
	ShapeSection *shapeSection_;
	PolynomialLateralSection *newSection_;

	QList<QPointF> oldPoints_;
	QList<QPointF> newPoints_;
};

//######################//
// Delete lateral shape    //
//#####################//

class DeleteLateralShapeSectionCommand : public DataCommand
{
public:
	explicit DeleteLateralShapeSectionCommand(ShapeSection *shapeSection, PolynomialLateralSection *polySection, const QList<QPointF> &points, DataCommand *parent = NULL);
	virtual ~DeleteLateralShapeSectionCommand();

	virtual int id() const
	{
		return 0x1303;
	}

	virtual void undo();
	virtual void redo();

private:
	DeleteLateralShapeSectionCommand(); /* not allowed */
	DeleteLateralShapeSectionCommand(const DeleteLateralShapeSectionCommand &); /* not allowed */
	DeleteLateralShapeSectionCommand &operator=(const DeleteLateralShapeSectionCommand &); /* not allowed */

private:
	ShapeSection *shapeSection_;
	PolynomialLateralSection *polySection_;

	QList<QPointF> oldPoints_;
	QList<QPointF> newPoints_;
};



#endif // SHAPELATERALSECTIONCOMMANDS_HPP
