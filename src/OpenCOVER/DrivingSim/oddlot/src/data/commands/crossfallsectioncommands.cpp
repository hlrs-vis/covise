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

#include "crossfallsectioncommands.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"

// Utils //
//
#include "math.h"

#define MIN_CROSSFALLSECTION_LENGTH 1.0

//#######################//
// SplitCrossfallSectionCommand //
//#######################//

SplitCrossfallSectionCommand::SplitCrossfallSectionCommand(CrossfallSection *crossfallSection, double splitPos, DataCommand *parent)
    : DataCommand(parent)
    , oldSection_(crossfallSection)
    , newSection_(NULL)
    , splitPos_(splitPos)
{
    // Check for validity //
    //
    if ((oldSection_->getDegree() > 1) // only lines allowed
        || (fabs(splitPos_ - oldSection_->getSStart()) < MIN_CROSSFALLSECTION_LENGTH)
        || (fabs(oldSection_->getSEnd() - splitPos_) < MIN_CROSSFALLSECTION_LENGTH) // minimum length 1.0 m
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Split CrossfallSection (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Split CrossfallSection"));
    }

    // New section //
    //
    newSection_ = new CrossfallSection(oldSection_->getSide(), splitPos, oldSection_->getCrossfallDegrees(splitPos), oldSection_->getB(), 0.0, 0.0);
}

/*! \brief .
*
*/
SplitCrossfallSectionCommand::~SplitCrossfallSectionCommand()
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
SplitCrossfallSectionCommand::redo()
{
    // Add section to road //
    //
    oldSection_->getParentRoad()->addCrossfallSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
SplitCrossfallSectionCommand::undo()
{
    // Remove section from road //
    //
    newSection_->getParentRoad()->delCrossfallSection(newSection_);

    setUndone();
}

//#######################//
// MergeCrossfallSectionCommand //
//#######################//

MergeCrossfallSectionCommand::MergeCrossfallSectionCommand(CrossfallSection *crossfallSectionLow, CrossfallSection *crossfallSectionHigh, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(crossfallSectionLow)
    , oldSectionHigh_(crossfallSectionHigh)
    , newSection_(NULL)
{
    parentRoad_ = crossfallSectionLow->getParentRoad();

    // Check for validity //
    //
    if ((oldSectionLow_->getDegree() > 1)
        || (oldSectionHigh_->getDegree() > 1) // only lines allowed
        || (parentRoad_ != crossfallSectionHigh->getParentRoad()) // not the same parents
        || crossfallSectionHigh != parentRoad_->getCrossfallSection(crossfallSectionLow->getSEnd()) // not consecutive
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Merge CrossfallSection (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Merge CrossfallSection"));
    }

    // New section //
    //
    double deltaLength = crossfallSectionHigh->getSEnd() - crossfallSectionLow->getSStart();
    double deltaHeight = crossfallSectionHigh->getCrossfallDegrees(crossfallSectionHigh->getSEnd()) - crossfallSectionLow->getCrossfallDegrees(crossfallSectionLow->getSStart());
    newSection_ = new CrossfallSection(oldSectionLow_->getSide(), oldSectionLow_->getSStart(), oldSectionLow_->getA(), deltaHeight / deltaLength, 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected() || oldSectionLow_->isElementSelected())
    {
        newSection_->setElementSelected(true); // keep selection
    }
}

/*! \brief .
*
*/
MergeCrossfallSectionCommand::~MergeCrossfallSectionCommand()
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
MergeCrossfallSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delCrossfallSection(oldSectionLow_);
    parentRoad_->delCrossfallSection(oldSectionHigh_);

    parentRoad_->addCrossfallSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
MergeCrossfallSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delCrossfallSection(newSection_);

    parentRoad_->addCrossfallSection(oldSectionLow_);
    parentRoad_->addCrossfallSection(oldSectionHigh_);

    setUndone();
}

//#######################//
// RemoveCrossfallSectionCommand //
//#######################//

RemoveCrossfallSectionCommand::RemoveCrossfallSectionCommand(CrossfallSection *crossfallSection, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(NULL)
    , oldSectionMiddle_(crossfallSection)
    , oldSectionHigh_(NULL)
    , newSectionHigh_(NULL)
{
    parentRoad_ = oldSectionMiddle_->getParentRoad();
    if (!parentRoad_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove CrossfallSection. No ParentRoad."));
        return;
    }

    oldSectionLow_ = parentRoad_->getCrossfallSectionBefore(oldSectionMiddle_->getSStart());
    if (!oldSectionLow_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove CrossfallSection. No section to the left."));
        return;
    }

    oldSectionHigh_ = parentRoad_->getCrossfallSection(oldSectionMiddle_->getSEnd());
    if (!oldSectionHigh_ || (oldSectionHigh_ == oldSectionMiddle_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove CrossfallSection. No section to the right."));
        return;
    }

    if (oldSectionLow_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove CrossfallSection. Section to the left is not linear."));
        return;
    }

    if (oldSectionHigh_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove CrossfallSection. Section to the right is not linear."));
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
            setText(QObject::tr("Cannot remove CrossfallSection. Sections to the left and right do not intersect."));
            return;
        }
    }
    else
    {
        // Not parallel //
        //
        s = sLow + (oldSectionHigh_->getCrossfallDegrees(sLow) - oldSectionLow_->getCrossfallDegrees(sLow)) / (oldSectionLow_->getB() - oldSectionHigh_->getB());
        if ((s - sLow < MIN_CROSSFALLSECTION_LENGTH)
            || (oldSectionHigh_->getSEnd() - s < MIN_CROSSFALLSECTION_LENGTH))
        {
            setInvalid(); // Invalid
            setText(QObject::tr("Cannot remove CrossfallSection. Sections to the left and right do not intersect."));
            return;
        }
    }
    //	qDebug() << "s: " << s << ", sLow: " << sLow;

    // New section //
    //
    newSectionHigh_ = new CrossfallSection(oldSectionLow_->getSide(), s, oldSectionLow_->getCrossfallDegrees(s), oldSectionHigh_->getB(), 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected() || oldSectionMiddle_->isElementSelected())
    {
        newSectionHigh_->setElementSelected(true); // keep selection
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Remove CrossfallSection"));
}

/*! \brief .
*
*/
RemoveCrossfallSectionCommand::~RemoveCrossfallSectionCommand()
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
RemoveCrossfallSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delCrossfallSection(oldSectionMiddle_);
    parentRoad_->delCrossfallSection(oldSectionHigh_);

    parentRoad_->addCrossfallSection(newSectionHigh_);

    setRedone();
}

/*! \brief .
*
*/
void
RemoveCrossfallSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delCrossfallSection(newSectionHigh_);

    parentRoad_->addCrossfallSection(oldSectionMiddle_);
    parentRoad_->addCrossfallSection(oldSectionHigh_);

    setUndone();
}

//#######################//
// SmoothCrossfallSectionCommand //
//#######################//

SmoothCrossfallSectionCommand::SmoothCrossfallSectionCommand(CrossfallSection *crossfallSectionLow, CrossfallSection *crossfallSectionHigh, double radius, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(crossfallSectionLow)
    , oldSectionHigh_(crossfallSectionHigh)
    , newSection_(NULL)
    , newSectionHigh_(NULL)
    , radius_(radius)
{
    parentRoad_ = crossfallSectionLow->getParentRoad();

    if (radius_ < 0.01 || radius_ > 1000000.0)
    {
        setInvalid();
        setText(QObject::tr("Cannot smooth CrossfallSection. Radius not in interval [0.01, 100000.0]."));
        return;
    }

    if (oldSectionLow_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. Section to the left is not linear."));
        return;
    }

    if (oldSectionHigh_->getDegree() > 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. Section to the right is not linear."));
        return;
    }

    if (parentRoad_ != crossfallSectionHigh->getParentRoad()) // not the same parents
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. Section do not belong to the same road."));
        return;
    }

    if (crossfallSectionHigh != parentRoad_->getCrossfallSection(crossfallSectionLow->getSEnd())) // not consecutive
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. Sections are not consecutive."));
        return;
    }

    // Coordinates //
    //
    double bLow = oldSectionLow_->getB();
    double bHigh = oldSectionHigh_->getB();
    if (fabs(bHigh - bLow) < 0.005)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. Sections are (approximately) parallel."));
        return;
    }

    double b = bLow;
    double c = 1.0 / radius_; // curvature
    double l = (bHigh - b) / (2.0 * c);
    if (l < 0)
    {
        c = -c;
        l = -l;
    }
    if (l < MIN_CROSSFALLSECTION_LENGTH)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. New Section would be too short."));
        return;
    }

    double h = b * l + c * l * l;
    sLow_ = crossfallSectionHigh->getSStart() + (h - bHigh * l) / (bHigh - bLow);
    sHigh_ = l + sLow_;
    //	qDebug() << "sLow_: " << sLow_ << ", sHigh_: " << sHigh_;

    if ((sLow_ < crossfallSectionLow->getSStart() + MIN_CROSSFALLSECTION_LENGTH) // plus one meter
        || (sHigh_ > crossfallSectionHigh->getSEnd() - MIN_CROSSFALLSECTION_LENGTH) // minus one meter
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot smooth CrossfallSection. New Section would be too long."));
        return;
    }

    // New sections //
    //
    newSection_ = new CrossfallSection(oldSectionLow_->getSide(), sLow_, oldSectionLow_->getCrossfallDegrees(sLow_), b, c, 0.0);
    newSectionHigh_ = new CrossfallSection(oldSectionHigh_->getSide(), sHigh_, oldSectionHigh_->getCrossfallDegrees(sHigh_), oldSectionHigh_->getB(), 0.0, 0.0);
    if (oldSectionHigh_->isElementSelected())
    {
        newSectionHigh_->setElementSelected(true);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Smooth CrossfallSection"));
}

/*! \brief .
*
*/
SmoothCrossfallSectionCommand::~SmoothCrossfallSectionCommand()
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
SmoothCrossfallSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delCrossfallSection(oldSectionHigh_);

    parentRoad_->addCrossfallSection(newSection_);
    parentRoad_->addCrossfallSection(newSectionHigh_);

    setRedone();
}

/*! \brief .
*
*/
void
SmoothCrossfallSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delCrossfallSection(newSection_);
    parentRoad_->delCrossfallSection(newSectionHigh_);

    parentRoad_->addCrossfallSection(oldSectionHigh_);

    setUndone();
}

//##########################//
// CrossfallMovePointsCommand //
//##########################//

CrossfallMovePointsCommand::CrossfallMovePointsCommand(const QList<CrossfallSection *> &endPointSections, const QList<CrossfallSection *> &startPointSections, const QPointF &deltaPos, DataCommand *parent)
    : DataCommand(parent)
    , endPointSections_(endPointSections)
    , startPointSections_(startPointSections)
    , crossfallOnly_(false)
    , deltaPos_(deltaPos)
{
    // Check for validity //
    //
    if (fabs(deltaPos_.manhattanLength()) < NUMERICAL_ZERO8 || (endPointSections_.isEmpty() && startPointSections_.isEmpty()))
    {
        setInvalid(); // Invalid because no change.
        //		setText(QObject::tr("Cannot move crossfall point. Nothing to be done."));
        setText("");
        return;
    }

    foreach (CrossfallSection *section, endPointSections_)
    {
        oldEndPointsBs_.append(section->getB());

        if (fabs(section->getSEnd() - section->getParentRoad()->getLength()) < NUMERICAL_ZERO8) //
        {
            crossfallOnly_ = true;
        }
    }

    bool tooShort = false;
    foreach (CrossfallSection *section, startPointSections_)
    {
        oldStartPointsAs_.append(section->getA());
        oldStartPointsBs_.append(section->getB());
        oldStartPointsSs_.append(section->getSStart());

        if (fabs(section->getSStart()) < NUMERICAL_ZERO8) // first section of the road
        {
            crossfallOnly_ = true;
        }
        else if ((section->getLength() - deltaPos_.x() < MIN_CROSSFALLSECTION_LENGTH) // min length at end
                 || (section->getParentRoad()->getCrossfallSectionBefore(section->getSStart())->getLength() + deltaPos_.x() < MIN_CROSSFALLSECTION_LENGTH))
        {
            tooShort = true;
        }
    }

    if (!crossfallOnly_ && tooShort)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot move crossfall point. A section would be too short."));
        return;
    }

    if (crossfallOnly_)
    {
        deltaPos_.setX(0.0);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Move Crossfall Point"));
}

/*! \brief .
*
*/
CrossfallMovePointsCommand::~CrossfallMovePointsCommand()
{
}

/*! \brief .
*
*/
void
CrossfallMovePointsCommand::redo()
{
    // Set points //
    //
    int i = 0;
    foreach (CrossfallSection *section, endPointSections_)
    {
        double startCrossfall = section->getCrossfallDegrees(section->getSStart());
        double endCrossfall = section->getCrossfallDegrees(section->getSEnd()) + deltaPos_.y();
        double slope = (endCrossfall - startCrossfall) / (section->getLength() + deltaPos_.x());
        section->setParametersDegrees(startCrossfall, slope, 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (CrossfallSection *section, startPointSections_)
    {
        double startCrossfall = section->getCrossfallDegrees(section->getSStart()) + deltaPos_.y();
        double endCrossfall = section->getCrossfallDegrees(section->getSEnd());
        double slope = (endCrossfall - startCrossfall) / (section->getLength() - deltaPos_.x());
        section->setParametersDegrees(startCrossfall, slope, 0.0, 0.0);
        ++i;
    }

    // Move //
    //
    if (!crossfallOnly_)
    {
        foreach (CrossfallSection *section, startPointSections_)
        {
            section->getParentRoad()->moveCrossfallSection(section->getSStart(), section->getSStart() + deltaPos_.x());
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
CrossfallMovePointsCommand::undo()
{
    // Set points //
    //
    int i = 0;
    foreach (CrossfallSection *section, endPointSections_)
    {
        section->setParametersDegrees(section->getA(), oldEndPointsBs_[i], 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (CrossfallSection *section, startPointSections_)
    {
        section->setParametersDegrees(oldStartPointsAs_[i], oldStartPointsBs_[i], 0.0, 0.0);
        ++i;
    }

    // Move //
    //
    if (!crossfallOnly_)
    {
        i = 0;
        foreach (CrossfallSection *section, startPointSections_)
        {
            section->getParentRoad()->moveCrossfallSection(section->getSStart(), oldStartPointsSs_[i]);
            ++i;
        }
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
CrossfallMovePointsCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const CrossfallMovePointsCommand *command = static_cast<const CrossfallMovePointsCommand *>(other);

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
