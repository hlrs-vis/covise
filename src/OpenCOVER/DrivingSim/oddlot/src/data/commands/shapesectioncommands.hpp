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
class PolynomialLateralSection;

#include <QMap>


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
    ShapeSection *oldSection_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Paste Lateral Sections          //
//################//

class PasteLateralShapeSectionsCommand : public DataCommand
{
public:
	explicit PasteLateralShapeSectionsCommand(ShapeSection *shapeSection, QMap<double, PolynomialLateralSection *> oldSections, QMap<double, PolynomialLateralSection *> newSections, DataCommand *parent = NULL);
	virtual ~PasteLateralShapeSectionsCommand();

	virtual int id() const
	{
		return 0x1104;
	}

	virtual void undo();
	virtual void redo();

private:
	PasteLateralShapeSectionsCommand(); /* not allowed */
	PasteLateralShapeSectionsCommand(const PasteLateralShapeSectionsCommand &); /* not allowed */
	PasteLateralShapeSectionsCommand &operator=(const PasteLateralShapeSectionsCommand &); /* not allowed */

private:
	ShapeSection *shapeSection_;
	QMap<double, PolynomialLateralSection *> oldSections_;
	QMap<double, PolynomialLateralSection *> newSections_;
};


#endif // SHAPESECTIONCOMMANDS_HPP
