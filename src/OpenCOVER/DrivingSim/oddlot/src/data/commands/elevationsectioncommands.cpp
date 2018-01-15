/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.06.2010
**
**************************************************************************/

#include "elevationsectioncommands.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

#include "src/data/scenerysystem/heightmap.hpp"
#include "src/cover/coverconnection.hpp"

// Utils //
//
#include "math.h"

#include <QStringList>

#define MIN_ELEVATIONSECTION_LENGTH 1.0

//#######################//
// SplitElevationSectionCommand //
//#######################//

SplitElevationSectionCommand::SplitElevationSectionCommand(ElevationSection *elevationSection, double splitPos, DataCommand *parent)
    : DataCommand(parent)
    , oldSection_(elevationSection)
    , newSection_(NULL)
    , splitPos_(splitPos)
{
    // Check for validity //
    //
    if ((oldSection_->getDegree() > 1) // only lines allowed
        || (fabs(splitPos_ - elevationSection->getSStart()) < MIN_ELEVATIONSECTION_LENGTH)
        || (fabs(elevationSection->getSEnd() - splitPos_) < MIN_ELEVATIONSECTION_LENGTH) // minimum length 1.0 m
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Split ElevationSection (invalid!)"));
        return;
    }

    // New section //
    //
    newSection_ = new ElevationSection(splitPos, elevationSection->getElevation(splitPos), elevationSection->getB(), 0.0, 0.0);

    // Done //
    //
    setValid();
    setText(QObject::tr("Split ElevationSection"));
}

/*! \brief .
*
*/
SplitElevationSectionCommand::~SplitElevationSectionCommand()
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
SplitElevationSectionCommand::redo()
{
    //  //
    //
    oldSection_->getParentRoad()->addElevationSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
SplitElevationSectionCommand::undo()
{
    //  //
    //
    newSection_->getParentRoad()->delElevationSection(newSection_);

    setUndone();
}

//#######################//
// MergeElevationSectionCommand //
//#######################//

MergeElevationSectionCommand::MergeElevationSectionCommand(ElevationSection *elevationSectionLow, ElevationSection *elevationSectionHigh, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(elevationSectionLow)
    , oldSectionHigh_(elevationSectionHigh)
    , newSection_(NULL)
{
    parentRoad_ = elevationSectionLow->getParentRoad();

    // Check for validity //
    //
    if (/*(oldSectionLow_->getDegree() > 1)
		|| (oldSectionHigh_->getDegree() > 1) // only lines allowed
		||*/ (parentRoad_ != elevationSectionHigh->getParentRoad()) // not the same parents
        || elevationSectionHigh != parentRoad_->getElevationSection(elevationSectionLow->getSEnd()) // not consecutive
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Merge ElevationSection (invalid!)"));
        return;
    }

    // New section //
    //
    //	double deltaLength = elevationSectionHigh->getSEnd() - elevationSectionLow->getSStart();
    //	double deltaHeight = elevationSectionHigh->getElevation(elevationSectionHigh->getSEnd()) - elevationSectionLow->getElevation(elevationSectionLow->getSStart());
    //	newSection_ = new ElevationSection(oldSectionLow_->getSStart(), oldSectionLow_->getA(), deltaHeight/deltaLength, 0.0, 0.0);
    //	if(oldSectionHigh_->isElementSelected() || oldSectionLow_->isElementSelected())
    //	{
    //		newSection_->setElementSelected(true); // keep selection
    //	}

    double l = elevationSectionHigh->getSEnd() - elevationSectionLow->getSStart();

    double h0 = elevationSectionLow->getElevation(elevationSectionLow->getSStart());
    double dh0 = elevationSectionLow->getSlope(elevationSectionLow->getSStart());

    double h1 = elevationSectionHigh->getElevation(elevationSectionHigh->getSEnd());
    double dh1 = elevationSectionHigh->getSlope(elevationSectionHigh->getSEnd());

    double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
    double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
    newSection_ = new ElevationSection(oldSectionLow_->getSStart(), h0, dh0, c, d);

    // Done //
    //
    setValid();
    setText(QObject::tr("Merge ElevationSection"));
}

/*! \brief .
*
*/
MergeElevationSectionCommand::~MergeElevationSectionCommand()
{
    // Clean up //
    //
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
MergeElevationSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delElevationSection(oldSectionLow_);
    parentRoad_->delElevationSection(oldSectionHigh_);

    parentRoad_->addElevationSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
MergeElevationSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delElevationSection(newSection_);

    parentRoad_->addElevationSection(oldSectionLow_);
    parentRoad_->addElevationSection(oldSectionHigh_);

    setUndone();
}

//#######################//
// RemoveElevationSectionCommand //
//#######################//

RemoveElevationSectionCommand::RemoveElevationSectionCommand(ElevationSection *elevationSection, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(NULL)
    , oldSectionMiddle_(elevationSection)
    , oldSectionHigh_(NULL)
    , newSectionHigh_(NULL)
{
    parentRoad_ = oldSectionMiddle_->getParentRoad();

    if (!parentRoad_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ElevationSection. No ParentRoad."));
        return;
    }

    oldSectionLow_ = parentRoad_->getElevationSectionBefore(oldSectionMiddle_->getSStart());
    if (!oldSectionLow_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ElevationSection. No section to the left."));
        return;
    }

    oldSectionHigh_ = parentRoad_->getElevationSection(oldSectionMiddle_->getSEnd());
    if (!oldSectionHigh_ || (oldSectionHigh_ == oldSectionMiddle_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ElevationSection. No section to the right."));
        return;
    }

    if (oldSectionLow_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ElevationSection. Section to the left is not linear."));
        return;
    }

    if (oldSectionHigh_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove ElevationSection. Section to the right is not linear."));
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
            setText(QObject::tr("Cannot remove ElevationSection. Sections to the left and right do not intersect."));
            return;
        }
    }
    else
    {
        // Not parallel //
        //
        s = sLow + (oldSectionHigh_->getElevation(sLow) - oldSectionLow_->getElevation(sLow)) / (oldSectionLow_->getB() - oldSectionHigh_->getB());
        if ((s - sLow < MIN_ELEVATIONSECTION_LENGTH)
            || (oldSectionHigh_->getSEnd() - s < MIN_ELEVATIONSECTION_LENGTH))
        {
            setInvalid(); // Invalid
            setText(QObject::tr("Cannot remove ElevationSection. Sections to the left and right do not intersect."));
            return;
        }
    }
    //	qDebug() << "s: " << s << ", sLow: " << sLow;

    // New section //
    //
    newSectionHigh_ = new ElevationSection(s, oldSectionLow_->getElevation(s), oldSectionHigh_->getB(), 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected() || oldSectionMiddle_->isElementSelected())
    {
        newSectionHigh_->setElementSelected(true); // keep selection
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Remove ElevationSection"));
}

/*! \brief .
*
*/
RemoveElevationSectionCommand::~RemoveElevationSectionCommand()
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
RemoveElevationSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delElevationSection(oldSectionMiddle_);
    parentRoad_->delElevationSection(oldSectionHigh_);

    parentRoad_->addElevationSection(newSectionHigh_);

    setRedone();
}

/*! \brief .
*
*/
void
RemoveElevationSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delElevationSection(newSectionHigh_);

    parentRoad_->addElevationSection(oldSectionMiddle_);
    parentRoad_->addElevationSection(oldSectionHigh_);

    setUndone();
}

//#######################//
// SmoothElevationSectionCommand //
//#######################//

SmoothElevationSectionCommand::SmoothElevationSectionCommand(ElevationSection *elevationSectionLow, ElevationSection *elevationSectionHigh, double radius, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(elevationSectionLow)
    , oldSectionHigh_(elevationSectionHigh)
    , newSection_(NULL)
    , newSectionHigh_(NULL)
    , radius_(radius)
{
    parentRoad_ = elevationSectionLow->getParentRoad();

    if (radius_ < 0.01 || radius_ > 1000000.0)
    {
        setInvalid();
        setText(QObject::tr("Cannot smooth ElevationSection. Radius not in interval [0.01, 100000.0]."));
        return;
    }

    if (oldSectionLow_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Section to the left is not linear."));
        return;
    }

    if (oldSectionHigh_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Section to the right is not linear."));
        return;
    }

    if (parentRoad_ != elevationSectionHigh->getParentRoad()) // not the same parents
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Section do not belong to the same road."));
        return;
    }

    if (elevationSectionHigh != parentRoad_->getElevationSection(elevationSectionLow->getSEnd())) // not consecutive
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Sections are not consecutive."));
        return;
    }

    // Coordinates //
    //
    double bLow = oldSectionLow_->getB();
    double bHigh = oldSectionHigh_->getB();
    if (fabs(bHigh - bLow) < 0.005)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Sections are (approximately) parallel."));
        return;
    }

    double b = bLow;
    //	double c = 1.0/radius_; // curvature
    double c = 1.0 / (2.0 * radius_) * pow(1 + b * b, 1.5);
    double l = (bHigh - b) / (2.0 * c);
    if (l < 0)
    {
        c = -c;
        l = -l;
    }
    if (l < MIN_ELEVATIONSECTION_LENGTH)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. New Section would be too short."));
        return;
    }

    double h = b * l + c * l * l;
    sLow_ = elevationSectionHigh->getSStart() + (h - bHigh * l) / (bHigh - bLow);
    sHigh_ = l + sLow_;
    //	qDebug() << "sLow_: " << sLow_ << ", sHigh_: " << sHigh_;

    if ((sLow_ < elevationSectionLow->getSStart() + MIN_ELEVATIONSECTION_LENGTH) // plus one meter
        || (sHigh_ > elevationSectionHigh->getSEnd() - MIN_ELEVATIONSECTION_LENGTH) // minus one meter
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. New Section would be too long."));
        return;
    }

    // New sections //
    //
    newSection_ = new ElevationSection(sLow_, oldSectionLow_->getElevation(sLow_), b, c, 0.0);
    newSectionHigh_ = new ElevationSection(sHigh_, oldSectionHigh_->getElevation(sHigh_), oldSectionHigh_->getB(), 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected())
    {
        newSectionHigh_->setElementSelected(true);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Smooth ElevationSection"));
}

/*! \brief .
*
*/
SmoothElevationSectionCommand::~SmoothElevationSectionCommand()
{
    if (isUndone())
    {
        delete newSection_;
        delete newSectionHigh_;
    }
    else
    {
        delete oldSectionHigh_;
    }
}

/*! \brief .
*
*/
void
SmoothElevationSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delElevationSection(oldSectionHigh_);

    parentRoad_->addElevationSection(newSection_);
    parentRoad_->addElevationSection(newSectionHigh_);

    setRedone();
}

/*! \brief .
*
*/
void
SmoothElevationSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delElevationSection(newSection_);
    parentRoad_->delElevationSection(newSectionHigh_);

    parentRoad_->addElevationSection(oldSectionHigh_);

    setUndone();
}

//#############################//
// SmoothElevationRoadsCommand //
//#############################//

SmoothElevationRoadsCommand::SmoothElevationRoadsCommand(ElevationSection *elevationSection1, ElevationSection *elevationSection2, double radius, DataCommand *parent)
    : DataCommand(parent)
    , radius_(radius)
{
    parentRoad1_ = elevationSection1->getParentRoad();
    parentRoad2_ = elevationSection2->getParentRoad();

    if (radius_ < 0.01 || radius_ > 1000000.0)
    {
        setInvalid();
        setText(QObject::tr("Cannot smooth ElevationSection. Radius not in interval [0.01, 100000.0]."));
        return;
    }

    if (elevationSection1->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Section to the left is not linear."));
        return;
    }

    if (elevationSection2->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Section to the right is not linear."));
        return;
    }

    if (parentRoad1_ == parentRoad2_) // not the same parents
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Sections belong to the same road."));
        return;
    }

    if ((!parentRoad1_->getSuccessor() || (parentRoad1_->getSuccessor()->getElementId() != parentRoad2_->getID())) && (!parentRoad1_->getPredecessor() || (parentRoad1_->getPredecessor()->getElementId() != parentRoad2_->getID()))) // not the same parents
    {
        setInvalid(); // Invalid
        setText(QObject::tr("The roads have to be linked."));
        return;
    }

    if ((parentRoad1_->getElevationSectionBefore(elevationSection1->getSStart()) && parentRoad1_->getElevationSectionNext(elevationSection1->getSStart())) || (parentRoad2_->getElevationSectionBefore(elevationSection2->getSStart()) && parentRoad2_->getElevationSectionNext(elevationSection2->getSStart())))
    // at least one section is in-between
    {
        setInvalid(); // Invalid
        setText(QObject::tr("One section of each road has to be the start or end section."));
        return;
    }

    bool smoothingSuccess = false;
    if (parentRoad1_->getSuccessor() && (parentRoad1_->getSuccessor()->getElementId() == parentRoad2_->getID()) && !parentRoad1_->getElevationSectionNext(elevationSection1->getSStart()))
    {
        if (parentRoad2_->getPredecessor() && (parentRoad2_->getPredecessor()->getElementId() == parentRoad1_->getID()) && !parentRoad2_->getElevationSectionBefore(elevationSection2->getSStart())
            && (fabs(elevationSection1->getElevation(elevationSection1->getSEnd()) - elevationSection2->getElevation(elevationSection2->getSStart())) < NUMERICAL_ZERO8))
        {
            smoothingSuccess = createSmoothSections(elevationSection1, parentRoad1_, elevationSection2, parentRoad2_, EndStart);
        }
        else if (parentRoad2_->getSuccessor() && (parentRoad2_->getSuccessor()->getElementId() == parentRoad1_->getID()) && !parentRoad2_->getElevationSectionNext(elevationSection2->getSStart())
                 && (fabs(elevationSection1->getElevation(elevationSection1->getSEnd()) - elevationSection2->getElevation(elevationSection2->getSEnd())) < NUMERICAL_ZERO8))
        {
            smoothingSuccess = createSmoothSections(elevationSection1, parentRoad1_, elevationSection2, parentRoad2_, EndEnd);
        }
    }

    if (parentRoad1_->getPredecessor() && (parentRoad1_->getPredecessor()->getElementId() == parentRoad2_->getID()) && !parentRoad1_->getElevationSectionBefore(elevationSection1->getSStart()))
    {
        if (parentRoad2_->getPredecessor() && (parentRoad2_->getPredecessor()->getElementId() == parentRoad1_->getID()) && !parentRoad2_->getElevationSectionBefore(elevationSection2->getSStart())
            && (fabs(elevationSection1->getElevation(elevationSection1->getSStart()) - elevationSection2->getElevation(elevationSection2->getSStart())) < NUMERICAL_ZERO8))
        {
            smoothingSuccess = createSmoothSections(elevationSection2, parentRoad2_, elevationSection1, parentRoad1_, StartStart);
        }
        else if (parentRoad2_->getSuccessor() && (parentRoad2_->getSuccessor()->getElementId() == parentRoad1_->getID()) && !parentRoad2_->getElevationSectionNext(elevationSection2->getSStart())
                 && (fabs(elevationSection1->getElevation(elevationSection1->getSStart()) - elevationSection2->getElevation(elevationSection2->getSEnd())) < NUMERICAL_ZERO8))
        {
            smoothingSuccess = createSmoothSections(elevationSection2, parentRoad2_, elevationSection1, parentRoad1_, EndStart);
        }
    }

    if (!smoothingSuccess)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth the roads. Check if the elevations at the adjacent corners are the same."));
        return;
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Smooth ElevationSection"));
}

/*! \brief .
*
*/
SmoothElevationRoadsCommand::~SmoothElevationRoadsCommand()
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
bool
SmoothElevationRoadsCommand::createSmoothSections(ElevationSection *elevationSectionLow, RSystemElementRoad *parentLow, ElevationSection *elevationSectionHigh, RSystemElementRoad *parentHigh, contactPoints contact)
{
    // Coordinates //
    //
    double bLow = elevationSectionLow->getB();
    double bHigh = elevationSectionHigh->getB();
    if (contact == EndEnd) // the gradient of the two sections must have different sign
    {
        bHigh = -bHigh;
    }
    else if (contact == StartStart)
    {
        bLow = -bLow;
    }

    if (fabs(bHigh - bLow) < 0.005)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. Sections are (approximately) parallel."));
        return false;
    }

    double b = bLow;
    //	double c = 1.0/radius_; // curvature
    double c = 1.0 / (2.0 * radius_) * pow(1 + b * b, 1.5);
    double l = (bHigh - b) / (2.0 * c);
    if (l < 0)
    {
        c = -c;
        l = -l;
    }
    if (l < MIN_ELEVATIONSECTION_LENGTH)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. New Section would be too short."));
        return false;
    }

    double h = b * l + c * l * l;
    double sp = (h - bHigh * l) / (bHigh - bLow);
    double sLow;
    double sHigh;
    if (contact == EndStart)
    {
        sLow = parentLow->getLength() + sp;
        sHigh = l + sp;
    }
    else if (contact == EndEnd)
    {
        sLow = parentLow->getLength() + sp;
        sHigh = parentHigh->getLength() - (l + sp);
    }
    else
    {
        sLow = -sp;
        sHigh = l + sp;
    }

    //	qDebug() << "sLow: " << sLow << ", sHigh: " << sHigh;

    if ((sLow < elevationSectionLow->getSStart() + MIN_ELEVATIONSECTION_LENGTH) // plus one meter
        || (l + sp > elevationSectionHigh->getSEnd() - MIN_ELEVATIONSECTION_LENGTH) // minus one meter
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth ElevationSection. New Section would be too long."));
        return false;
    }

    // New sections //
    //
    ElevationSection *newSection;

    if (contact == EndStart)
    {
        newSection = new ElevationSection(sLow, elevationSectionLow->getElevation(sLow), b, c, 0.0);
        newSections_.insert(parentLow, newSection);

        double a1 = elevationSectionLow->getElevation(sLow) - b * sp + c * sp * sp;
        double b1 = b - 2 * c * sp;

        newSection = new ElevationSection(0.0, a1, b1, c, 0.0);
        newSections_.insert(parentHigh, newSection);
        oldSections_.insert(parentHigh, elevationSectionHigh);

        newSection = new ElevationSection(sHigh, elevationSectionHigh->getElevation(sHigh), elevationSectionHigh->getB(), 0.0, 0.0);
        newSections_.insert(parentHigh, newSection);
    }
    else if (contact == EndEnd)
    {
        newSection = new ElevationSection(sLow, elevationSectionLow->getElevation(sLow), b, c, 0.0);
        newSections_.insert(parentLow, newSection);

        double a1 = elevationSectionLow->getElevation(sLow) - b * sp + c * sp * sp;
        double b1 = b - 2 * c * sp;

        a1 = a1 + b1 * (l + sp) + c * (l + sp) * (l + sp);
        b1 = -b1 - 2 * c * (l + sp);
        newSection = new ElevationSection(sHigh, a1, b1, c, 0.0);
        newSections_.insert(parentHigh, newSection);
    }
    else if (contact == StartStart)
    {
        oldSections_.insert(parentLow, elevationSectionLow);

        double a1 = elevationSectionLow->getElevation(sLow) - b * sp + c * sp * sp;
        double b1 = -b + 2 * c * sp;

        newSection = new ElevationSection(0.0, a1, b1, c, 0.0);
        newSections_.insert(parentLow, newSection);

        newSection = new ElevationSection(sLow, elevationSectionLow->getElevation(sLow), elevationSectionLow->getB(), 0.0, 0.0);
        newSections_.insert(parentLow, newSection);

        a1 = elevationSectionLow->getElevation(sLow) - b * sp + c * sp * sp;
        b1 = b - 2 * c * sp;

        newSection = new ElevationSection(0.0, a1, b1, c, 0.0);
        newSections_.insert(parentHigh, newSection);
        oldSections_.insert(parentHigh, elevationSectionHigh);

        newSection = new ElevationSection(sHigh, elevationSectionHigh->getElevation(sHigh), elevationSectionHigh->getB(), 0.0, 0.0);
        newSections_.insert(parentHigh, newSection);
    }

    if (elevationSectionHigh->isElementSelected())
    {
        newSection->setElementSelected(true);
    }

    return true;
}

/*! \brief .
*
*/
void
SmoothElevationRoadsCommand::redo()
{
    //  //
    //
    QList<ElevationSection *> roadSections = oldSections_.values(parentRoad1_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad1_->delElevationSection(roadSections.at(i));
    }

    roadSections = newSections_.values(parentRoad1_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad1_->addElevationSection(roadSections.at(i));
    }

    roadSections = oldSections_.values(parentRoad2_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad2_->delElevationSection(roadSections.at(i));
    }

    roadSections = newSections_.values(parentRoad2_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad2_->addElevationSection(roadSections.at(i));
    }

    setRedone();
}

/*! \brief .
*
*/
void
SmoothElevationRoadsCommand::undo()
{
    //  //
    //
    QList<ElevationSection *> roadSections = newSections_.values(parentRoad1_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad1_->delElevationSection(roadSections.at(i));
    }

    roadSections = oldSections_.values(parentRoad1_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad1_->addElevationSection(roadSections.at(i));
    }

    roadSections = newSections_.values(parentRoad2_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad2_->delElevationSection(roadSections.at(i));
    }

    roadSections = oldSections_.values(parentRoad2_);

    for (int i = 0; i < roadSections.size(); i++)
    {
        parentRoad2_->addElevationSection(roadSections.at(i));
    }

    setUndone();
}

//##########################//
// SetElevationCommand //
//##########################//

SetElevationCommand::SetElevationCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, double deltaHeight, DataCommand *parent)
    : DataCommand(parent)
    , deltaHeight_(deltaHeight)
{
    // Lists //
    //
    endPointSections_ = endPointSections;
    startPointSections_ = startPointSections;

    foreach (ElevationSection *section, endPointSections_)
    {
        oldEndHeights_.append(section->getElevation(section->getSEnd()));
    }
    foreach (ElevationSection *section, startPointSections_)
    {
        oldStartHeights_.append(section->getElevation(section->getSStart()));
    }

    // Check for validity //
    //
    if (fabs(deltaHeight_) < NUMERICAL_ZERO8 || (endPointSections_.isEmpty() && startPointSections_.isEmpty()))
    {
        setInvalid(); // Invalid because no change.
        setText(QObject::tr("Set elevation (invalid!)"));
        return;
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Set elevation"));
}

/*! \brief .
*
*/
SetElevationCommand::~SetElevationCommand()
{
}

/*! \brief .
*
*/
void
SetElevationCommand::redo()
{
    // Set points //
    //
    int i = 0;
    foreach (ElevationSection *section, endPointSections_)
    {
        double startElevation = section->getElevation(section->getSStart());
        double endElevation = section->getElevation(section->getSEnd()) + deltaHeight_;
        double slope = (endElevation - startElevation) / section->getLength();
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (ElevationSection *section, startPointSections_)
    {
        double startElevation = section->getElevation(section->getSStart()) + deltaHeight_;
        double endElevation = section->getElevation(section->getSEnd());
        double slope = (endElevation - startElevation) / section->getLength();
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }

    setRedone();
}

/*! \brief
*
*/
void
SetElevationCommand::undo()
{
    // Set points //
    //
    int i = 0;
    foreach (ElevationSection *section, endPointSections_)
    {
        double startElevation = section->getElevation(section->getSStart());
        double endElevation = oldEndHeights_[i];
        double slope = (endElevation - startElevation) / section->getLength();
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (ElevationSection *section, startPointSections_)
    {
        double startElevation = oldStartHeights_[i];
        double endElevation = section->getElevation(section->getSEnd());
        double slope = (endElevation - startElevation) / section->getLength();
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
SetElevationCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SetElevationCommand *command = static_cast<const SetElevationCommand *>(other);

    // Check sections //
    //
    if (endPointSections_.size() != command->endPointSections_.size()
        || startPointSections_.size() != command->startPointSections_.size())
    {
        return false; // not the same amount of sections
    }

    for (int i = 0; i < endPointSections_.size(); ++i)
    {
        if (endPointSections_[i] != command->endPointSections_[i])
        {
            return false; // different sections
        }
    }
    for (int i = 0; i < startPointSections_.size(); ++i)
    {
        if (startPointSections_[i] != command->startPointSections_[i])
        {
            return false; // different sections
        }
    }

    // Success //
    //
    deltaHeight_ += command->deltaHeight_; // adjust to new height, then let the undostack kill the new command

    return true;
}

//##########################//
// ElevationMovePointsCommand //
//##########################//

ElevationMovePointsCommand::ElevationMovePointsCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, const QPointF &deltaPos, DataCommand *parent)
    : DataCommand(parent)
    , endPointSections_(endPointSections)
    , startPointSections_(startPointSections)
    , elevationOnly_(false)
    , deltaPos_(deltaPos)
{
    // Check for validity //
    //
    if (fabs(deltaPos_.manhattanLength()) < NUMERICAL_ZERO8 || (endPointSections_.isEmpty() && startPointSections_.isEmpty()))
    {
        setInvalid(); // Invalid because no change.
        //		setText(QObject::tr("Cannot move elevation point. Nothing to be done."));
        setText("");
        return;
    }

    foreach (ElevationSection *section, endPointSections_)
    {
        //		oldEndHeights_.append(section->getElevation(section->getSEnd())); // save these, so no drifting when doing continuous undo/redo/undo/redo/...
        oldEndPointsBs_.append(section->getB());

        if (fabs(section->getSEnd() - section->getParentRoad()->getLength()) < NUMERICAL_ZERO8) //
        {
            elevationOnly_ = true;
        }
    }

    bool tooShort = false;
    foreach (ElevationSection *section, startPointSections_)
    {
        //		oldStartHeights_.append(section->getElevation(section->getSStart()));
        oldStartPointsAs_.append(section->getA());
        oldStartPointsBs_.append(section->getB());
        oldStartPointsSs_.append(section->getSStart());

        if (fabs(section->getSStart()) < NUMERICAL_ZERO8) // first section of the road
        {
            elevationOnly_ = true;
        }
        else if ((section->getLength() - deltaPos_.x() < MIN_ELEVATIONSECTION_LENGTH) // min length at end
                 || (section->getParentRoad()->getElevationSectionBefore(section->getSStart())->getLength() + deltaPos_.x() < MIN_ELEVATIONSECTION_LENGTH))
        {
            tooShort = true;
        }
        //		oldStartSStarts_.append(section->getSStart());
        //		oldStartLengths_.append(section->getLength());
    }

    if (!elevationOnly_ && tooShort)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot move elevation point. A section would be too short."));
        return;
    }

    if (elevationOnly_)
    {
        deltaPos_.setX(0.0);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Move Elevation Point"));
}

/*! \brief .
*
*/
ElevationMovePointsCommand::~ElevationMovePointsCommand()
{
}

/*! \brief .
*
*/
void
ElevationMovePointsCommand::redo()
{
    // Set points //
    //
    int i = 0;
    foreach (ElevationSection *section, endPointSections_)
    {
        double startElevation = section->getElevation(section->getSStart());
        double endElevation = section->getElevation(section->getSEnd()) + deltaPos_.y();
        double slope = (endElevation - startElevation) / (section->getLength() + deltaPos_.x());
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (ElevationSection *section, startPointSections_)
    {
        double startElevation = section->getElevation(section->getSStart()) + deltaPos_.y();
        double endElevation = section->getElevation(section->getSEnd());
        double slope = (endElevation - startElevation) / (section->getLength() - deltaPos_.x());
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }

    // Move //
    //
    if (!elevationOnly_)
    {
        foreach (ElevationSection *section, startPointSections_)
        {
            section->getParentRoad()->moveElevationSection(section->getSStart(), section->getSStart() + deltaPos_.x());
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
ElevationMovePointsCommand::undo()
{
    // Set points //
    //
    int i = 0;
    foreach (ElevationSection *section, endPointSections_)
    {
        //		double startElevation = section->getElevation(section->getSStart());
        //		double endElevation = oldEndHeights_[i];
        //		double slope = (endElevation-startElevation)/oldStartLengths_[i];
        //		section->setParameters(startElevation, slope, 0.0, 0.0);
        section->setParameters(section->getA(), oldEndPointsBs_[i], 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (ElevationSection *section, startPointSections_)
    {
        //		double startElevation = oldStartHeights_[i];
        //		double endElevation = section->getElevation(section->getSEnd());
        //		double slope = (endElevation-startElevation)/(section->getSEnd()-oldStartSStarts_[i]);
        //		section->setParameters(startElevation, slope, 0.0, 0.0);
        section->setParameters(oldStartPointsAs_[i], oldStartPointsBs_[i], 0.0, 0.0);
        ++i;
    }

    // Move //
    //
    if (!elevationOnly_)
    {
        i = 0;
        foreach (ElevationSection *section, startPointSections_)
        {
            section->getParentRoad()->moveElevationSection(section->getSStart(), oldStartPointsSs_[i]);
            ++i;
        }
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
ElevationMovePointsCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const ElevationMovePointsCommand *command = static_cast<const ElevationMovePointsCommand *>(other);

    if (!command->elevationOnly_)
        elevationOnly_ = false;

    // Check sections //
    //
    if (endPointSections_.size() != command->endPointSections_.size()
        || startPointSections_.size() != command->startPointSections_.size())
    {
        return false; // not the same amount of sections
    }

    for (int i = 0; i < endPointSections_.size(); ++i)
    {
        if (endPointSections_[i] != command->endPointSections_[i])
        {
            return false; // different sections
        }
    }
    for (int i = 0; i < startPointSections_.size(); ++i)
    {
        if (startPointSections_[i] != command->startPointSections_[i])
        {
            return false; // different sections
        }
    }

    // Success //
    //
    deltaPos_ += command->deltaPos_; // adjust to new pos, then let the undostack kill the new command

    return true;
}

//##########################//
// ElevationSetHeightCommand //
//##########################//

ElevationSetHeightCommand::ElevationSetHeightCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, float height, bool absolute, DataCommand *parent)
    : DataCommand(parent)
    , endPointSections_(endPointSections)
    , startPointSections_(startPointSections)
    , newHeight(height)
    , absoluteHeight(absolute)
{
    // Check for validity //
    //
    if ((absoluteHeight == false && (fabs(newHeight) < NUMERICAL_ZERO8)) || (endPointSections_.isEmpty() && startPointSections_.isEmpty()))
    {
        setInvalid(); // Invalid because no change.
        //		setText(QObject::tr("Cannot move elevation point. Nothing to be done."));
        setText("");
        return;
    }

    foreach (ElevationSection *section, endPointSections_)
    {
        oldEndPointsAs_.append(section->getA());
        oldEndPointsBs_.append(section->getB());
        oldEndPointsCs_.append(section->getC());
        oldEndPointsDs_.append(section->getD());
    }

    foreach (ElevationSection *section, startPointSections_)
    {
        oldStartPointsAs_.append(section->getA());
        oldStartPointsBs_.append(section->getB());
        oldStartPointsCs_.append(section->getC());
        oldStartPointsDs_.append(section->getD());
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Set Elevation Height"));
}

/*! \brief .
*
*/
ElevationSetHeightCommand::~ElevationSetHeightCommand()
{
}

/*! \brief .
*
*/
void
ElevationSetHeightCommand::redo()
{
    // Set points //
    //
    int i = 0;
    foreach (ElevationSection *section, endPointSections_)
    {
        double startElevation = section->getElevation(section->getSStart());
        double endElevation = newHeight;
        if (!absoluteHeight)
            endElevation = section->getElevation(section->getSEnd()) + newHeight;
        double slope = (endElevation - startElevation) / (section->getLength());
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (ElevationSection *section, startPointSections_)
    {
        double startElevation = newHeight;
        if (!absoluteHeight)
            startElevation = section->getElevation(section->getSStart()) + newHeight;
        double endElevation = section->getElevation(section->getSEnd());
        double slope = (endElevation - startElevation) / (section->getLength());
        section->setParameters(startElevation, slope, 0.0, 0.0);
        ++i;
    }

    setRedone();
}

/*! \brief
*
*/
void
ElevationSetHeightCommand::undo()
{
    // Set points //
    //
    int i = 0;
    foreach (ElevationSection *section, endPointSections_)
    {
        section->setParameters(oldEndPointsAs_[i], oldEndPointsBs_[i], oldEndPointsCs_[i], oldEndPointsDs_[i]);
        ++i;
    }
    i = 0;
    foreach (ElevationSection *section, startPointSections_)
    {
        section->setParameters(oldStartPointsAs_[i], oldStartPointsBs_[i], oldStartPointsCs_[i], oldStartPointsDs_[i]);
        ++i;
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
ElevationSetHeightCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const ElevationSetHeightCommand *command = static_cast<const ElevationSetHeightCommand *>(other);

    // Check sections //
    //
    if (endPointSections_.size() != command->endPointSections_.size()
        || startPointSections_.size() != command->startPointSections_.size())
    {
        return false; // not the same amount of sections
    }

    for (int i = 0; i < endPointSections_.size(); ++i)
    {
        if (endPointSections_[i] != command->endPointSections_[i])
        {
            return false; // different sections
        }
    }
    for (int i = 0; i < startPointSections_.size(); ++i)
    {
        if (startPointSections_[i] != command->startPointSections_[i])
        {
            return false; // different sections
        }
    }

    // Success //
    //
    if (command->absoluteHeight)
        absoluteHeight = true;
    if (absoluteHeight)
        newHeight = command->newHeight;
    else
        newHeight += command->newHeight;

    return true;
}

//##########################//
// ApplyHeightMapElevationCommand //
//##########################//

ApplyHeightMapElevationCommand::ApplyHeightMapElevationCommand(RSystemElementRoad *road, const QList<Heightmap *> &maps, double heightOffset, double sampleDistance, double maxDeviation, double lowPassFilter, bool useCubic, bool smoothLinear, double radius, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , maps_(maps)
    , heightOffset_(heightOffset)
    , sampleDistance_(sampleDistance)
    , maxDeviation_(maxDeviation)
    , lowPassFilter_(lowPassFilter)
    , useCubic_(useCubic)
    , smoothLinear_(smoothLinear)
    , radius_(radius)
{
    // Check for validity //
    //
    if (!road || (maps.isEmpty() && !COVERConnection::instance()->isConnected()) || sampleDistance < NUMERICAL_ZERO3 || maxDeviation < NUMERICAL_ZERO3)
    {
        setInvalid(); // Invalid because no change.
        setText("Apply Heightmap: invalid parameters!");
        return;
    }

    // Sections //
    //
    oldSections_ = road->getElevationSections();

    // Initialization //
    //
    double sStart = 0.0;
    double sEnd = road_->getLength();
    if (sEnd < sStart)
        sEnd = sStart;

#if 1
    double pointsPerMeter = 1.0 / sampleDistance;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter));
    if (pointCount < 2)
    {
        // TODO
        pointCount = 2;
        qDebug() << road->getID().speakingName() << " Segment too short: duplicate points per meter";
    }
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    // Read Heights //
    //
    double *sampleHeights = new double[pointCount];
    if(COVERConnection::instance()->isConnected())
    {
        covise::TokenBuffer tb;
        tb << MSG_GetHeight;
        tb << pointCount;
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            QPointF pos = road_->getGlobalPoint(s);
            tb << (float)pos.x();
            tb << (float)pos.y();
        }
        COVERConnection::instance()->send(tb);
        covise::Message *msg=NULL;
        if(COVERConnection::instance()->waitForMessage(&msg))
        {
            covise::TokenBuffer rtb(msg);
            int type;
            rtb >>  type;
            if(type == MSG_GetHeight)
            {
                int pc;
                rtb >> pc;
                if(pc == pointCount)
                {
                    float h;
                    for (int i = 0; i < pointCount; ++i)
                    {
                        rtb >> h;
                        sampleHeights[i] = h + heightOffset_;
                    }
                }
                else
                {
                    return;
                }
            }
            else
            {
                return;
            }
        }
        else
        {
            pointCount =0;
        }
        
    }
    else
    {
        for (int i = 0; i < pointCount; ++i)
        {
            double s = sStart + i * segmentLength; // [sStart, sEnd]
            sampleHeights[i] = getHeight(s) + heightOffset_;
        }
    }

#if 0
	// Low pass filter //
	//
	double heights[pointCount];
	heights[0] = sampleHeights[0];
	double alpha = lowPassFilter_;
	for(int i = 1; i < pointCount; ++i)
	{
		heights[i] = alpha*sampleHeights[i] + (1-alpha)*heights[i-1];
	}
#endif

#if 1
    // Low pass filter //
    //
    double *heights = new double[pointCount];
    heights[0] = sampleHeights[0];
    double alpha = lowPassFilter_;
    for (int i = 1; i < pointCount - 1; ++i)
    {
        heights[i] = 0.5 * (1 - alpha) * sampleHeights[i - 1] + alpha * sampleHeights[i] + 0.5 * (1 - alpha) * sampleHeights[i + 1];
    }
    heights[pointCount - 1] = sampleHeights[pointCount - 1];
    delete[] sampleHeights;
#endif

    // Cubic approximation //
    //
    if (useCubic_ && pointCount>0)
    {
        // Calculate Slopes //
        //
        double *dheights = new double[pointCount];
        dheights[0] = (heights[1] - heights[0]) / segmentLength;

        // Create Sections //
        //
        int lastPoint=0;
        ElevationSection *oldSection=NULL;
        
        for (int i = 1; i < pointCount; ++i)
        {
            double currentLength = segmentLength*(i-lastPoint);
            if(i < pointCount - 1)
            {
                dheights[i] = 0.5 * (heights[i] - heights[lastPoint]) / (segmentLength*(i-lastPoint)) + 0.5 * (heights[i + 1] - heights[i]) / segmentLength;
                // evtl. gewichten
            }
            else
            {
                dheights[pointCount - 1] = (heights[pointCount - 1] - heights[lastPoint]) / (segmentLength*((pointCount - 1)-lastPoint));
            }
            double s = sStart + lastPoint * segmentLength; // [sStart, sEnd]
/*
            double predictedHeight = lastSection->getElevation(s);
            double predictedSlope = lastSection->getSlope(s);
            if ((fabs(predictedHeight - heights[i]) < maxDeviation_)
                && (fabs(predictedSlope - dheights[i]) < maxDeviation_ * 0.1))
            {
                heights[i] = predictedHeight;
                dheights[i] = predictedSlope;
                continue;
            }*/

            double d = heights[lastPoint];
            double c = dheights[lastPoint];
            double a = (dheights[i] + c - 2.0 * heights[i] / currentLength + 2.0 * d / currentLength) / (currentLength * currentLength);
            double b = (heights[i] - a * currentLength * currentLength * currentLength - c * currentLength - d) / (currentLength * currentLength);

            double maxDistance=0;
            
            ElevationSection *section = new ElevationSection(s, d, c, b, a);
            for(int n=(lastPoint+1);n<i;n++)
            {
                double dist = fabs(heights[n] - section->getElevation(n*segmentLength));
                if(dist > maxDistance)
                    maxDistance = dist;
            }
            if(maxDistance < maxDeviation_)
            {
                delete oldSection;
                oldSection = section;
            }
            else
            {
                newSections_.insert(s, oldSection);
                lastPoint = i-1;
                
                double d = heights[i-1];
                double c = dheights[i-1];
                double a = (dheights[i] + c - 2.0 * heights[i] / segmentLength + 2.0 * d / segmentLength) / (segmentLength * segmentLength);
                double b = (heights[i] - a * segmentLength * segmentLength * segmentLength - c * segmentLength - d) / (segmentLength * segmentLength);
                double s = sStart + (i-1) * segmentLength; // [sStart, sEnd]
                oldSection = new ElevationSection(s, d, c, b, a);
                lastPoint = i-1;
            }

        }
        delete[] dheights;
        if(oldSection)
        {
            double s = sStart + lastPoint * segmentLength;
            newSections_.insert(s, oldSection);
        }
    }

    // Linear approximation //
    //
    else if(pointCount > 0)
    {
        // Create Sections //
        //

        int lastIndex = 0;
        double sectionStart = sStart;
        for (int i = 2; i < pointCount; ++i)
        {
            double slope = (heights[i] - heights[lastIndex]) / ((i - lastIndex) * segmentLength);

            int j;
            for (j = i - 1; j > lastIndex; j--)
            {
                double predictedHeight = heights[lastIndex] + slope * (sStart + j * segmentLength);
                if (fabs(predictedHeight - heights[j]) > maxDeviation) // take the last one
                {
                    break;
                }
            }

            if ((i != pointCount - 1) && (j == lastIndex))
            {
                continue;
            }

            slope = (heights[i - 1] - heights[lastIndex]) / (((i - 1) - lastIndex) * segmentLength); // slope of the new section refers to the start of the section

            ElevationSection *section = new ElevationSection(sectionStart, heights[lastIndex], slope, 0.0, 0.0);
            newSections_.insert(sectionStart, section);

            if ((i == pointCount - 1) && (j != lastIndex))
            {
                lastIndex = i - 1;
                sectionStart = sStart + lastIndex * segmentLength;
                slope = (heights[i] - heights[lastIndex]) / segmentLength;

                ElevationSection *section = new ElevationSection(sectionStart, heights[lastIndex], slope, 0.0, 0.0);
                newSections_.insert(sectionStart, section);
            }
            else
            {
                lastIndex = i - 1;
                sectionStart = sStart + lastIndex * segmentLength;
                slope = (heights[i] - heights[lastIndex]) / segmentLength;
            }
        }
    }
#endif

#if 0
	double pointsPerMeter = 0.1; // BAD: hard coded!
	int pointCount = int(ceil((sEnd-sStart)*pointsPerMeter)); // TODO curvature...
	double segmentLength = (sEnd-sStart)/(pointCount-1);

	// Sections //
	//
	double heights[pointCount];
	ElevationSection * lastSection = NULL;
	for(int i = 0; i < pointCount; ++i)
	{
		double s = sStart + i * segmentLength; // [sStart, sEnd]

		// Height //
		//
		QPointF pos = road->getGlobalPoint(s);
		double height = 0.0;
		int count = 0;
		foreach(Heightmap * map, maps_)
		{
			if(map->isIntersectedBy(pos))
			{
				height = height + map->getHeightmapValue(pos.x(), pos.y());
				++count;
			}
		}
		if(count != 0)
		{
			height = height/count;
		}

		heights[i] = height + heightOffset_;
	}


	for(int i = 0; i < pointCount-1; ++i)
	{
		double s = sStart + i * segmentLength; // [sStart, sEnd]
		double slope = (heights[i+1]-heights[i])/segmentLength;
		if(lastSection
			&& (fabs(lastSection->getB() - slope) < NUMERICAL_ZERO6)
			&& (fabs(lastSection->getElevation(s) - heights[i]) < NUMERICAL_ZERO6)
		)
		{
			continue;
		}

		ElevationSection * section = new ElevationSection(s, heights[i], slope, 0.0, 0.0);
		newSections_.insert(s, section);
		lastSection = section;
	}
#endif

    delete[] heights;
    // No sections created //
    //
    if (newSections_.isEmpty())
    {
        ElevationSection *section = new ElevationSection(0.0, 0.0, 0.0, 0.0, 0.0);
        newSections_.insert(0.0, section);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Apply Heightmap"));
}

/*! \brief .
*
*/
ApplyHeightMapElevationCommand::~ApplyHeightMapElevationCommand()
{

    if (isUndone())
    {
        foreach (ElevationSection *section, newSections_)
        {
            delete section;
        }
    }
    else
    {
        foreach (ElevationSection *section, oldSections_)
        {
            delete section;
        }
    }
}

/*! \brief .
*
*/
void
ApplyHeightMapElevationCommand::redo()
{
    road_->setElevationSections(newSections_);

    if (smoothLinear_)
    {
        ElevationSection *elevationSectionLow = road_->getElevationSection(0.0);
        double highSEnd = elevationSectionLow->getSEnd();
        ElevationSection *elevationSectionHigh;
        double myDist = radius_;

        while (fabs(road_->getLength() - highSEnd) > NUMERICAL_ZERO3)
        {
            elevationSectionHigh = road_->getElevationSection(highSEnd);
            elevationSectionLow = road_->getElevationSectionBefore(highSEnd);

            if ((elevationSectionHigh->getLength() < elevationSectionLow->getLength()) && (elevationSectionHigh->getLength() < myDist))
            {
                myDist = elevationSectionHigh->getLength() / 2;
            }
            else if (elevationSectionLow->getLength() < myDist)
            {
                myDist = elevationSectionLow->getLength() / 2;
            }

            double slopeLow = (elevationSectionLow->getElevation(elevationSectionLow->getSEnd()) - elevationSectionLow->getElevation(elevationSectionLow->getSStart())) / elevationSectionLow->getLength();

            double slopeHigh = (elevationSectionHigh->getElevation(elevationSectionHigh->getSEnd()) - elevationSectionHigh->getElevation(elevationSectionHigh->getSStart())) / elevationSectionHigh->getLength();
            double angle = 180.0 - atan((slopeHigh - slopeLow) / (1 + slopeHigh * slopeLow));
            double tanAngle = tan(angle * M_PI / 360);
            double myRadius = fabs(myDist * tanAngle);

            highSEnd = elevationSectionHigh->getSEnd();
            SmoothElevationSectionCommand *command = new SmoothElevationSectionCommand(elevationSectionLow, elevationSectionHigh, myRadius);
            if (command->isValid())
            {
                command->redo();
            }
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
ApplyHeightMapElevationCommand::undo()
{
    road_->setElevationSections(oldSections_);

    setUndone();
}

double
ApplyHeightMapElevationCommand::getHeight(double s)
{
    QPointF pos = road_->getGlobalPoint(s);
    double height = 0.0;
    int count = 0;

    // Take the average over all maps //
    //
    foreach (Heightmap *map, maps_)
    {
        if (map->isIntersectedBy(pos))
        {
            height = height + map->getHeightmapValue(pos.x(), pos.y());
            ++count;
        }
    }
    if (count != 0)
    {
        height = height / count;
    }

    return height + heightOffset_;
}

//##########################//
// FlatJunctionsElevationCommand //
//##########################//

FlatJunctionsElevationCommand::FlatJunctionsElevationCommand(RSystemElementJunction *junction, double transitionLength, DataCommand *parent)
    : DataCommand(parent)
    , junction_(junction)
{
    // Check for validity //
    //
    if (!junction_ || !junction->getRoadSystem())
    {
        setInvalid();
        setText("Flatten Junctions: invalid parameters!");
        return;
    }

    // Roads //
    //
    RoadSystem *roadSystem = junction_->getRoadSystem();
    QList<odrID> pathIds;
	QList<odrID> endRoadIds;
	QList<odrID> startRoadIds;

    foreach (JunctionConnection *connection, junction_->getConnections())
    {
        odrID pathId = connection->getConnectingRoad();
        RSystemElementRoad *road = roadSystem->getRoad(pathId);
        if (!road)
        {
            continue;
        }
        if (!pathIds.contains(pathId))
        {
            pathIds.append(pathId);
            paths_.append(road);
        }

        RoadLink *link = NULL;
        if (connection->getContactPoint() == JunctionConnection::JCP_START)
        {
            link = road->getSuccessor(); // junction -> path start ... path end -> path successor
        }
        else if (connection->getContactPoint() == JunctionConnection::JCP_END)
        {
            link = road->getPredecessor();
        }

        if (link)
        {
            if (link->getContactPoint() == JunctionConnection::JCP_START && !startRoadIds.contains(link->getElementId()))
            {
                startRoadIds.append(link->getElementId());
                RSystemElementRoad *road = roadSystem->getRoad(link->getElementId());
                if (road)
                {
                    startRoads_.append(road);
                }
            }
            if (link->getContactPoint() == JunctionConnection::JCP_END && !endRoadIds.contains(link->getElementId()))
            {
                endRoadIds.append(link->getElementId());
                RSystemElementRoad *road = roadSystem->getRoad(link->getElementId());
                if (road)
                {
                    endRoads_.append(road);
                }
            }
        }
    }

    if (paths_.isEmpty())
    {
        setInvalid();
        setText("Flatten Junctions: invalid parameters! No paths.");
        return;
    }

    // Average Height //
    //
    double averageHeight = 0.0;
    foreach (RSystemElementRoad *road, paths_)
    {
        foreach (ElevationSection *section, road->getElevationSections())
        {
            oldSections_.insert(road->getID(), section);
        }

        averageHeight += road->getElevationSection(0.0)->getElevation(0.0);
        averageHeight += road->getElevationSection(road->getLength())->getElevation(road->getLength());
    }
    averageHeight = averageHeight / (2.0 * paths_.count());

    // Sections //
    //
    foreach (RSystemElementRoad *road, paths_)
    {
        newSections_.insert(road->getID(), new ElevationSection(0.0, averageHeight, 0.0, 0.0, 0.0));
    }

    foreach (RSystemElementRoad *road, startRoads_)
    {
        roadCommands_.append(new SetEndElevationCommand(road, true, averageHeight, 0.0, transitionLength));
    }

    foreach (RSystemElementRoad *road, endRoads_)
    {
        roadCommands_.append(new SetEndElevationCommand(road, false, averageHeight, 0.0, transitionLength));
    }

    // No sections created //
    //
    //	if(newSections_.isEmpty())
    //	{
    //		ElevationSection * section = new ElevationSection(0.0, 0.0, 0.0, 0.0, 0.0);
    //		newSections_.insert(0.0, section);
    //	}

    // Done //
    //
    setValid();
    setText(QObject::tr("Flatten Junctions"));
}

/*! \brief .
*
*/
FlatJunctionsElevationCommand::~FlatJunctionsElevationCommand()
{

    if (isUndone())
    {
        foreach (ElevationSection *section, newSections_)
        {
            delete section;
        }
    }
    else
    {
        foreach (ElevationSection *section, oldSections_)
        {
            delete section;
        }
    }

    foreach (SetEndElevationCommand *command, roadCommands_)
    {
        delete command;
    }
}

/*! \brief .
*
*/
void
FlatJunctionsElevationCommand::redo()
{
    foreach (RSystemElementRoad *road, paths_)
    {
        QMap<double, ElevationSection *> sections;
        sections.insert(0.0, newSections_.value(road->getID()));
        road->setElevationSections(sections);
    }

    foreach (SetEndElevationCommand *command, roadCommands_)
    {
        command->redo();
    }

    setRedone();
}

/*! \brief
*
*/
void
FlatJunctionsElevationCommand::undo()
{
    foreach (RSystemElementRoad *road, paths_)
    {
        QMap<double, ElevationSection *> sections;
        foreach (ElevationSection *section, oldSections_.values(road->getID()))
        {
            sections.insert(section->getSStart(), section);
        }
        road->setElevationSections(sections);
    }

    foreach (SetEndElevationCommand *command, roadCommands_)
    {
        command->undo();
    }

    setUndone();
}

//##########################//
// SetEndElevationCommand //
//##########################//

SetEndElevationCommand::SetEndElevationCommand(RSystemElementRoad *road, bool isStart, double newHeight, double newSlope, double transitionLength, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , newSection_(NULL)
    , newSectionB_(NULL)
{
    // Check for validity //
    //
    if (!road_)
    {
        setInvalid();
        setText("Set end elevation: invalid parameters!");
        return;
    }

    if (transitionLength > 0.5 * road->getLength())
    {
        transitionLength = road->getLength() * 0.4;
    }

    if (isStart)
    {
        double s = transitionLength;

        // Put old sections in list //
        //
        foreach (ElevationSection *section, road->getElevationSections())
        {
            if (section->getSStart() <= s)
            {
                oldSections_.append(section);
            }
        }

        double d0 = newHeight;
        double c0 = newSlope;

        ElevationSection *secondSection = road->getElevationSection(s);
        double d1 = secondSection->getElevation(s);
        double c1 = secondSection->getSlope(s);

        double length = secondSection->getSEnd() - s;
        double d2 = secondSection->getElevation(secondSection->getSEnd());
        double c2 = secondSection->getSlope(secondSection->getSEnd());

        // New Section //
        //
        double a0 = (c1 + c0 - 2.0 * d1 / transitionLength + 2.0 * d0 / transitionLength) / (transitionLength * transitionLength);
        double b0 = (d1 - a0 * transitionLength * transitionLength * transitionLength - c0 * transitionLength - d0) / (transitionLength * transitionLength);
        newSection_ = new ElevationSection(0.0, d0, c0, b0, a0);

        double a1 = (c2 + c1 - 2.0 * d2 / length + 2.0 * d1 / length) / (length * length);
        double b1 = (d2 - a1 * length * length * length - c1 * length - d1) / (length * length);
        newSectionB_ = new ElevationSection(s, d1, c1, b1, a1);
    }
    else
    {
        double s = road->getLength() - transitionLength;

        // Put old sections in list //
        //
        ElevationSection *prevSection = NULL;
        foreach (ElevationSection *section, road->getElevationSections())
        {
            if (section->getSStart() >= s)
            {
                oldSections_.append(section);
            }
            else
            {
                prevSection = section;
            }
        }

        // New Section //
        //
        double d = prevSection->getElevation(s);
        double c = prevSection->getSlope(s);
        double a = (newSlope + c - 2.0 * newHeight / transitionLength + 2.0 * d / transitionLength) / (transitionLength * transitionLength);
        double b = (newHeight - a * transitionLength * transitionLength * transitionLength - c * transitionLength - d) / (transitionLength * transitionLength);
        newSection_ = new ElevationSection(s, d, c, b, a);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Flatten Junctions"));
}

/*! \brief .
*
*/
SetEndElevationCommand::~SetEndElevationCommand()
{

    if (isUndone())
    {
        delete newSection_;
        delete newSectionB_;
    }
    else
    {
        foreach (ElevationSection *section, oldSections_)
        {
            delete section;
        }
    }
}

/*! \brief .
*
*/
void
SetEndElevationCommand::redo()
{
    foreach (ElevationSection *section, oldSections_)
    {
        road_->delElevationSection(section);
    }

    road_->addElevationSection(newSection_);
    if (newSectionB_)
    {
        road_->addElevationSection(newSectionB_);
    }

    setRedone();
}

/*! \brief
*
*/
void
SetEndElevationCommand::undo()
{
    road_->delElevationSection(newSection_);
    if (newSectionB_)
    {
        road_->delElevationSection(newSectionB_);
    }

    foreach (ElevationSection *section, oldSections_)
    {
        road_->addElevationSection(section);
    }

    setUndone();
}

//################################//
// SelectElevationSectionCommand //
//##############################//

SelectElevationSectionCommand::SelectElevationSectionCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, DataCommand *parent)
    : DataCommand(parent)
{
    // Lists //
    //
    endPointSections_ = endPointSections;
    startPointSections_ = startPointSections;

    // Check for validity //
    //
    if (endPointSections_.isEmpty() && startPointSections_.isEmpty())
    {
        setInvalid(); // Invalid because no change.
        setText(QObject::tr("Set elevation section (invalid!)"));
        return;
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Set elevation section"));
}

/*! \brief .
*
*/
SelectElevationSectionCommand::~SelectElevationSectionCommand()
{
}

void
SelectElevationSectionCommand::
    redo()
{
    // Send a notification to the observers
    //
    foreach (ElevationSection *section, endPointSections_)
    {
        section->addElevationSectionChanges(true);
    }

    foreach (ElevationSection *section, startPointSections_)
    {
        section->addElevationSectionChanges(true);
    }

    setRedone();
}
