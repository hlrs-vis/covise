/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.04.2010
**
**************************************************************************/

#include "trackcommands.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"

#include "src/data/visitors/trackmovevalidator.hpp"

#include "math.h"

#include <QGraphicsScene>

#define MIN_SECTION_LENGTH 1.0

//##########################//
// TrackComponentGlobalPointsCommand //
//##########################//

TrackComponentGlobalPointsCommand::TrackComponentGlobalPointsCommand(const QList<TrackComponent *> &endPointTracks, const QList<TrackComponent *> &startPointTracks, const QPointF &dPos, DataCommand *parent)
    : DataCommand(parent)
    , dPos_(dPos)
{
    // Lists //
    //
    endPointTracks_ = endPointTracks;
    startPointTracks_ = startPointTracks;
    foreach (TrackComponent *track, endPointTracks_)
    {
        oldEndPoints_.append(track->getGlobalPoint(track->getSEnd()));
    }
    foreach (TrackComponent *track, startPointTracks_)
    {
        oldStartPoints_.append(track->getGlobalPoint(track->getSStart()));
    }
    foreach (TrackComponent *track, endPointTracks_)
    {
        if (!roads_.contains(track->getParentRoad()))
        {
            roads_.append(track->getParentRoad());
        }
    }
    foreach (TrackComponent *track, startPointTracks_)
    {
        if (!roads_.contains(track->getParentRoad()))
        {
            roads_.append(track->getParentRoad());
        }
    }

    // Check for validity //
    //
    if (dPos_.isNull() || (endPointTracks_.isEmpty() && startPointTracks_.isEmpty()))
    {
        setInvalid(); // Invalid because no change.
        setText(QObject::tr("Set point (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set point"));
    }
}

/*! \brief .
*
*/
TrackComponentGlobalPointsCommand::~TrackComponentGlobalPointsCommand()
{
}

/*! \brief .
*
*/
void
TrackComponentGlobalPointsCommand::redo()
{
    // Set points //
    //
    int i = 0;
    foreach (TrackComponent *track, endPointTracks_)
    {
        track->setGlobalEndPoint(oldEndPoints_[i] + dPos_);
        ++i;
    }
    i = 0;
    foreach (TrackComponent *track, startPointTracks_)
    {
        track->setGlobalStartPoint(oldStartPoints_[i] + dPos_);
        ++i;
    }

    // Update road length and s coordinates //
    //
    foreach (RSystemElementRoad *road, roads_)
    {
        road->rebuildTrackComponentList();
    }

    setRedone();
}

/*! \brief
*
*/
void
TrackComponentGlobalPointsCommand::undo()
{
    // Set points //
    //
    int i = 0;
    foreach (TrackComponent *track, endPointTracks_)
    {
        track->setGlobalEndPoint(oldEndPoints_[i]);
        ++i;
    }
    i = 0;
    foreach (TrackComponent *track, startPointTracks_)
    {
        track->setGlobalStartPoint(oldStartPoints_[i]);
        ++i;
    }

    // Update road length and s coordinates //
    //
    foreach (RSystemElementRoad *road, roads_)
    {
        road->rebuildTrackComponentList();
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
TrackComponentGlobalPointsCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const TrackComponentGlobalPointsCommand *command = static_cast<const TrackComponentGlobalPointsCommand *>(other);

    // Check tracks //
    //
    if (endPointTracks_.size() != command->endPointTracks_.size()
        || startPointTracks_.size() != command->startPointTracks_.size())
    {
        return false;
    }
    for (int i = 0; i < endPointTracks_.size(); ++i)
    {
        if (endPointTracks_[i] != command->endPointTracks_[i])
        {
            return false;
        }
    }
    for (int i = 0; i < startPointTracks_.size(); ++i)
    {
        if (startPointTracks_[i] != command->startPointTracks_[i])
        {
            return false;
        }
    }

    // Success //
    //
    dPos_ += command->dPos_; // adjust to new point, then let the undostack kill the new command

    return true;
}

//##########################//
// TrackComponentPointAndHeadingCommand //
//##########################//

TrackComponentPointAndHeadingCommand::TrackComponentPointAndHeadingCommand(TrackComponent *track, const QPointF &globalPoint, double globalHeadingDeg, bool isStart, DataCommand *parent)
    : DataCommand(parent)
    , track_(track)
    , newPoint_(globalPoint)
    , newHeading_(globalHeadingDeg)
    , isStart_(isStart)
{
    if (track_)
    {
        if (isStart_)
        {
            oldPoint_ = track->getGlobalPoint(track->getSStart());
            oldHeading_ = track_->getGlobalHeading(track_->getSStart());
        }
        else
        {
            oldPoint_ = track->getGlobalPoint(track->getSEnd());
            oldHeading_ = track_->getGlobalHeading(track_->getSEnd());
        }
    }
    else
    {
        setInvalid();
        setText(QObject::tr("Set point/heading (invalid!)"));
        return;
    }

    // Check for validity //
    //
    if ((newHeading_ == oldHeading_) && (newPoint_ == oldPoint_))
    {
        setInvalid(); // Invalid because no change.
        setText(QObject::tr("Set point/heading (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set point/heading"));
    }
}

/*! \brief .
*
*/
TrackComponentPointAndHeadingCommand::~TrackComponentPointAndHeadingCommand()
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

/*! \brief .
*
*/
void
TrackComponentPointAndHeadingCommand::redo()
{
    // RoadSection length //
    //
    //	double oldSStart = track_->getSStart();
    //	double oldLength = track_->getLength();
    //	qDebug() << "oldS: " << oldSStart << ", oldLength: " << oldLength;

    // Set Point/Heading //
    //
    track_->setGlobalPointAndHeading(newPoint_, newHeading_, isStart_);

    //	double newSStart = track_->getSStart();
    //	double newLength = track_->getLength();
    //	qDebug() << "newS: " << newSStart << ", newLength: " << newLength;

    // Update road length and s coordinates //
    //
    track_->getParentRoad()->rebuildTrackComponentList();

    setRedone();
}

/*! \brief
*
*/
void
TrackComponentPointAndHeadingCommand::undo()
{
    // Set Point/Heading //
    //
    track_->setGlobalPointAndHeading(oldPoint_, oldHeading_, isStart_);

    // Update road length and s coordinates //
    //
    track_->getParentRoad()->rebuildTrackComponentList();

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
TrackComponentPointAndHeadingCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const TrackComponentPointAndHeadingCommand *command = static_cast<const TrackComponentPointAndHeadingCommand *>(other);

    // Check tracks //
    //
    if (track_ == command->track_ && isStart_ == command->isStart_)
    {
        newPoint_ = command->newPoint_;
        newHeading_ = command->newHeading_; // adjust to new heading, then let the undostack kill the new command
        return true;
    }
    else
    {
        return false;
    }
}

//##########################//
// TrackComponentSingleHeadingCommand //
//##########################//

TrackComponentSingleHeadingCommand::TrackComponentSingleHeadingCommand(TrackComponent *lowSlotTrack, TrackComponent *highSlotTrack, double newHeading, DataCommand *parent)
    : DataCommand(parent)
    , lowSlotTrack_(lowSlotTrack)
    , highSlotTrack_(highSlotTrack)
    , newHeading_(newHeading)
{
    if (lowSlotTrack_)
    {
        oldHeading_ = lowSlotTrack_->getGlobalHeading(lowSlotTrack_->getSEnd());
    }
    else if (highSlotTrack_)
    {
        oldHeading_ = highSlotTrack_->getGlobalHeading(highSlotTrack_->getSStart());
    }
    else
    {
        setInvalid();
        setText(QObject::tr("Set heading (invalid!)"));
        return;
    }

    // Check for validity //
    //
    if (newHeading_ == oldHeading_)
    {
        setInvalid(); // Invalid because no change.
        setText(QObject::tr("Set heading (invalid!)"));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set heading"));
    }
}

/*! \brief .
*
*/
TrackComponentSingleHeadingCommand::~TrackComponentSingleHeadingCommand()
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

/*! \brief .
*
*/
void
TrackComponentSingleHeadingCommand::redo()
{
    // Set Heading //
    //
    if (lowSlotTrack_)
    {
        lowSlotTrack_->setGlobalEndHeading(newHeading_);
    }

    if (highSlotTrack_)
    {
        highSlotTrack_->setGlobalStartHeading(newHeading_);
    }

    // Update road length and s coordinates //
    //
    if (lowSlotTrack_)
    {
        lowSlotTrack_->getParentRoad()->rebuildTrackComponentList();
    }
    else
    {
        highSlotTrack_->getParentRoad()->rebuildTrackComponentList();
    }

    setRedone();
}

/*! \brief
*
*/
void
TrackComponentSingleHeadingCommand::undo()
{
    // Set Heading //
    //
    if (lowSlotTrack_)
    {
        lowSlotTrack_->setGlobalEndHeading(oldHeading_);
    }

    if (highSlotTrack_)
    {
        highSlotTrack_->setGlobalStartHeading(oldHeading_);
    }

    // Update road length and s coordinates //
    //
    if (lowSlotTrack_)
    {
        lowSlotTrack_->getParentRoad()->rebuildTrackComponentList();
    }
    else
    {
        highSlotTrack_->getParentRoad()->rebuildTrackComponentList();
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
TrackComponentSingleHeadingCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const TrackComponentSingleHeadingCommand *command = static_cast<const TrackComponentSingleHeadingCommand *>(other);

    // Check tracks //
    //
    if (lowSlotTrack_ == command->lowSlotTrack_ && highSlotTrack_ == command->highSlotTrack_)
    {
        newHeading_ = command->newHeading_; // adjust to new heading, then let the undostack kill the new command
        return true;
    }
    else
    {
        return false;
    }
}

//#######################//
// SplitTrackComponentCommand //
//#######################//

SplitTrackComponentCommand::SplitTrackComponentCommand(TrackComponent *component, double sSplit, DataCommand *parent)
    : DataCommand(parent)
    , oldComponent_(component)
    , newComponentA_(NULL)
    , newComponentB_(NULL)
    , oldComposite_(NULL)
    , road_(NULL)
{
    // Check for validity //
    //
    if ((!component)
        || (!component->getParentRoad())
        || ((sSplit - component->getSStart()) < MIN_SECTION_LENGTH)
        || ((component->getSEnd() - sSplit) < MIN_SECTION_LENGTH))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Split TrackComponent: invalid parameters or section would be too short!"));
        return;
    }

    road_ = component->getParentRoad();

    if (dynamic_cast<TrackComposite *>(component->getParentComponent()))
    {
        oldComposite_ = dynamic_cast<TrackComposite *>(component->getParentComponent());

        foreach (TrackComponent *track, oldComposite_->getChildTrackComponents())
        {
            TrackComponent *clone = track->getClone();
            clone->setGlobalTranslation(oldComposite_->getGlobalPoint(clone->getSStart()));
            clone->setGlobalRotation(oldComposite_->getGlobalHeading(clone->getSStart()));

            if (track == component)
            {
                component = clone; // The clone has to be cut
            }
            else
            {
                newComponents_.append(clone);
            }
        }
    }

    // New components //
    //
    double sStart = component->getSStart();
    double sEnd = component->getSEnd();

    if (component->getTrackType() == TrackComponent::DTT_LINE)
    {
        newComponentA_ = new TrackElementLine(component->getGlobalPoint(sStart).x(), component->getGlobalPoint(sStart).y(), component->getGlobalHeading(sStart), sStart, sSplit - sStart);
        newComponentB_ = new TrackElementLine(component->getGlobalPoint(sSplit).x(), component->getGlobalPoint(sSplit).y(), component->getGlobalHeading(sSplit), sSplit, sEnd - sSplit);
    }

    else if (component->getTrackType() == TrackComponent::DTT_POLY3)
    {
        TrackElementPoly3 *poly3 = dynamic_cast<TrackElementPoly3 *>(component);
        newComponentA_ = new TrackElementPoly3(component->getGlobalPoint(sStart).x(), component->getGlobalPoint(sStart).y(), component->getGlobalHeading(sStart), sStart, sSplit - sStart, poly3->getA(), poly3->getB(), poly3->getC(), poly3->getD());
        ;

        // Local to internal (Parameters are given in internal coordinates) //
        //
        QTransform trafo;
        trafo.translate(component->getGlobalPoint(sSplit).x(), component->getGlobalPoint(sSplit).y());
        trafo.rotate(component->getGlobalHeading(sSplit));
        QPointF endPoint = trafo.inverted().map(component->getGlobalPoint(sEnd));

        double l = endPoint.x();
        double h0 = 0.0; // pos...
        double dh0 = 0.0; // ... and heading are zero because this is represented by the transformation
        double h1 = endPoint.y();
        double angle = component->getGlobalHeadingRad(sEnd) - component->getGlobalHeadingRad(sSplit);
        while (angle > M_PI)
        {
            angle = angle - 2.0 * M_PI;
        }
        while (angle <= -M_PI)
        {
            angle = angle + 2.0 * M_PI;
        }
        double dh1 = tan(angle);

        double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
        double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
        Polynomial poly(h0, dh0, c, d);

        newComponentB_ = new TrackElementPoly3(component->getGlobalPoint(sSplit).x(), component->getGlobalPoint(sSplit).y(), component->getGlobalHeading(sSplit), sSplit, poly.getCurveLength(0.0, l), poly);

        TrackMoveValidator *validator = new TrackMoveValidator();
        validator->setState(TrackMoveValidator::STATE_ENDPOINT);
        validator->setGlobalDeltaPos(QPointF());
        newComponentB_->accept(validator);

        if (!validator->isValid())
        {
            setInvalid();
            setText(QObject::tr("Split track: sorry, splitting is not possible here."));
            return;
        }
    }

    else if (component->getTrackType() == TrackComponent::DTT_ARC)
    {
        TrackElementArc *arc = dynamic_cast<TrackElementArc *>(component);
        newComponentA_ = new TrackElementArc(arc->getGlobalPoint(sStart).x(), arc->getGlobalPoint(sStart).y(), arc->getGlobalHeading(sStart), sStart, sSplit - sStart, arc->getCurvature(sStart));
        newComponentB_ = new TrackElementArc(arc->getGlobalPoint(sSplit).x(), arc->getGlobalPoint(sSplit).y(), arc->getGlobalHeading(sSplit), sSplit, sEnd - sSplit, arc->getCurvature(sSplit));
    }

    else if (component->getTrackType() == TrackComponent::DTT_SPIRAL)
    {
        TrackElementSpiral *spiral = dynamic_cast<TrackElementSpiral *>(component);
        newComponentA_ = new TrackElementSpiral(spiral->getGlobalPoint(sStart).x(), spiral->getGlobalPoint(sStart).y(), spiral->getGlobalHeading(sStart), sStart, sSplit - sStart, spiral->getCurvature(sStart), spiral->getCurvature(sSplit));
        newComponentB_ = new TrackElementSpiral(spiral->getGlobalPoint(sSplit).x(), spiral->getGlobalPoint(sSplit).y(), spiral->getGlobalHeading(sSplit), sSplit, sEnd - sSplit, spiral->getCurvature(sSplit), spiral->getCurvature(sEnd));
    }

    else
    {
        setInvalid();
        setText(QObject::tr("Split TrackComponent: sorry the type of this component is not yet supported!"));
        return;
    }

    if (oldComposite_)
    {
        newComponents_.append(newComponentA_);
        newComponents_.append(newComponentB_);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Split TrackComponent"));
}

SplitTrackComponentCommand::~SplitTrackComponentCommand()
{
    if (isUndone())
    {
        delete newComponentA_;
        delete newComponentB_;
        if (oldComposite_)
        {
            newComponents_.clear();
        }
    }
    else
    {
        delete oldComponent_;
        if (oldComposite_)
        {
            delete oldComposite_;
        }
    }

    //delete ungroupCommand;
}

void
SplitTrackComponentCommand::redo()
{
    if (oldComposite_)
    {
        road_->delTrackComponent(oldComposite_);
        foreach (TrackComponent *track, newComponents_)
        {
            road_->addTrackComponent(track);
        }

        road_->rebuildTrackComponentList();
    }
    else if (isValid())
    {
        road_->delTrackComponent(oldComponent_);

        road_->addTrackComponent(newComponentA_);
        road_->addTrackComponent(newComponentB_);
    }

    setRedone();
}

void
SplitTrackComponentCommand::undo()
{
    if (oldComposite_)
    {
        foreach (TrackComponent *track, newComponents_)
        {
            road_->delTrackComponent(track);
        }

        road_->addTrackComponent(oldComposite_);
        road_->rebuildTrackComponentList();

        setUndone();
    }
    else if (isValid())
    {
        road_->delTrackComponent(newComponentB_);
        road_->delTrackComponent(newComponentA_);

        road_->addTrackComponent(oldComponent_);
    }

    setUndone();
}

//##########################//
// SetSpArcSFactorCommand //
//##########################//

SetSpArcSFactorCommand::SetSpArcSFactorCommand(TrackSpiralArcSpiral *sparcs, double newFactor, DataCommand *parent)
    : DataCommand(parent)
    , sparcs_(sparcs)
    , newFactor_(newFactor)
{
    // Factor //
    //
    oldFactor_ = sparcs_->getFactor();

    if (newFactor_ > 1.0)
    {
        newFactor_ = 1.0;
    }
    if (newFactor_ < 0.0)
    {
        newFactor_ = 0.0;
    }

    // Check for validity //
    //
    if (!sparcs_ || (newFactor_ == oldFactor_))
    {
        setInvalid();
        setText(QObject::tr("Set Parameter invalid. No Element given or no change."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Parameter"));
    }
}

/*! \brief .
*
*/
SetSpArcSFactorCommand::~SetSpArcSFactorCommand()
{
}

/*! \brief .
*
*/
void
SetSpArcSFactorCommand::redo()
{
    sparcs_->setFactor(newFactor_);

    setRedone();
}

/*! \brief
*
*/
void
SetSpArcSFactorCommand::undo()
{
    sparcs_->setFactor(oldFactor_);

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
SetSpArcSFactorCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SetSpArcSFactorCommand *command = static_cast<const SetSpArcSFactorCommand *>(other);

    // Check tracks //
    //
    if (sparcs_ == command->sparcs_)
    {
        newFactor_ = command->newFactor_;
        return true;
    }
    else
    {
        return false;
    }
}

//##########################//
// TrackComponentSinglePointCommand //
//##########################//
#if 0
	TrackComponentSinglePointCommand
	::TrackComponentSinglePointCommand(TrackComponent * lowSlotTrack, TrackComponent * highSlotTrack, const QPointF & dPos, DataCommand * parent)
	:	DataCommand(parent),
		lowSlotTrack_(lowSlotTrack),
		highSlotTrack_(highSlotTrack),
		dPos_(dPos)
{
	if(lowSlotTrack_)
	{
		oldPoint_ = lowSlotTrack_->getGlobalPoint(lowSlotTrack_->getSEnd());
	}
	else if(highSlotTrack_)
	{
		oldPoint_ = highSlotTrack_->getGlobalPoint(highSlotTrack_->getSStart());
	}
	else
	{
		setInvalid();
		setText(QObject::tr("Set point (invalid!)"));
		return;
	}


	// Check for validity //
	//
	if(dPos_.isNull())
	{
		setInvalid(); // Invalid because no change.
		setText(QObject::tr("Set point (invalid!)"));
	}
	else
	{
		setValid();
		setText(QObject::tr("Set point"));
	}
}


/*! \brief .
*
*/
	TrackComponentSinglePointCommand
	::~TrackComponentSinglePointCommand()
{
	// Clean up //
	//
	if(isUndone())
	{
		// nothing to be done
	}
	else
	{
		// nothing to be done
	}
}


/*! \brief .
*
*/
void
	TrackComponentSinglePointCommand
	::redo()
{
	// Set points //
	//
	if(lowSlotTrack_)
	{
		lowSlotTrack_->setGlobalEndPoint(oldPoint_ + dPos_);
	}

	if(highSlotTrack_)
	{
		highSlotTrack_->setGlobalStartPoint(oldPoint_ + dPos_);
	}


	// Update road length and s coordinates //
	//
	if(lowSlotTrack_)
	{
		lowSlotTrack_->getParentRoad()->rebuildTrackComponentList();
	}
	else
	{
		highSlotTrack_->getParentRoad()->rebuildTrackComponentList();
	}


	setRedone();
}


/*! \brief
*
*/
void
	TrackComponentSinglePointCommand
	::undo()
{
	// Set points //
	//
	if(lowSlotTrack_)
	{
		lowSlotTrack_->setGlobalEndPoint(oldPoint_);
	}

	if(highSlotTrack_)
	{
		highSlotTrack_->setGlobalStartPoint(oldPoint_);
	}


	// Update road length and s coordinates //
	//
	if(lowSlotTrack_)
	{
		lowSlotTrack_->getParentRoad()->rebuildTrackComponentList();
	}
	else
	{
		highSlotTrack_->getParentRoad()->rebuildTrackComponentList();
	}


	setUndone();
}


/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
	TrackComponentSinglePointCommand
	::mergeWith(const QUndoCommand * other)
{
	// Check Ids //
	//
	if (other->id() != id())
	{
		return false;
	}

	const TrackComponentSinglePointCommand * command = static_cast<const TrackComponentSinglePointCommand *>(other);

	// Check tracks //
	//
	if(lowSlotTrack_ == command->lowSlotTrack_ && highSlotTrack_ == command->highSlotTrack_)
	{
		dPos_ += command->dPos_; // adjust to new point, then let the undostack kill the new command
		return true;
	}
	else
	{
		return false;
	}
}
#endif

//##########################//
// SetGlobalTrackPosCommand //
//##########################//

SetGlobalTrackPosCommand::SetGlobalTrackPosCommand(TrackComponent *track, const QPointF &newPos, DataCommand *parent)
    : DataCommand(parent)
    , track_(track)
    , newPos_(newPos)
{
    if (!track)
    {
        setInvalid();
        setText(QObject::tr("Set Translation: invalid parameters! No track given."));
    }

    oldPos_ = track_->getGlobalPoint(track_->getSStart());

    if (oldPos_ == newPos_)
    {
        setInvalid();
        setText(QObject::tr("Set Translation: no change. Command ignored."));
    }

    setValid();
    setText(QObject::tr("Set Translation"));
}

SetGlobalTrackPosCommand::~SetGlobalTrackPosCommand()
{
}

void
SetGlobalTrackPosCommand::redo()
{
    track_->setGlobalTranslation(newPos_);

    setRedone();
}

void
SetGlobalTrackPosCommand::undo()
{
    track_->setGlobalTranslation(oldPos_);

    setUndone();
}

bool
SetGlobalTrackPosCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SetGlobalTrackPosCommand *command = static_cast<const SetGlobalTrackPosCommand *>(other);

    // Check tracks //
    //
    if (track_ == command->track_)
    {
        newPos_ = command->newPos_;
        return true;
    }
    else
    {
        return false;
    }
}

//##########################//
// SetTrackLengthCommand //
//##########################//

  
SetTrackLengthCommand::SetTrackLengthCommand(TrackComponent *track, double newLength, DataCommand *parent)
    : DataCommand(parent)
    , track_(track)
    , newLength_(newLength)
{
    if (!track)
    {
        setInvalid();
        setText(QObject::tr("Set Translation: invalid parameters! No track given."));
    }

    oldLength_ = track_->getLength();

    if (oldLength_ == newLength_)
    {
        setInvalid();
        setText(QObject::tr("Set Length: no change. Command ignored."));
    }

    setValid();
    setText(QObject::tr("Set Length"));
}

SetTrackLengthCommand::~SetTrackLengthCommand()
{
}

void
SetTrackLengthCommand::redo()
{
    track_->setLength(newLength_);
    
    track_->getParentRoad()->rebuildTrackComponentList();
    setRedone();
}

void
SetTrackLengthCommand::undo()
{
    track_->setLength(oldLength_);
    
    track_->getParentRoad()->rebuildTrackComponentList();
    setUndone();
}

bool
SetTrackLengthCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SetTrackLengthCommand *command = static_cast<const SetTrackLengthCommand *>(other);

    // Check tracks //
    //
    if (track_ == command->track_)
    {
        newLength_ = command->newLength_;
        return true;
    }
    else
    {
        return false;
    }
}


//##########################//
// TrackComponentHeadingCommand //
//##########################//
#if 0
	TrackComponentHeadingCommand
	::TrackComponentHeadingCommand(TrackComponent * track, double newHeading, DataCommand * parent)
	:	DataCommand(parent),
		track_(track)
{
	// Check for validity //
	//
	if(false)
	{
		setInvalid(); // Invalid because...
		setText(QObject::tr("Set Heading (invalid!)"));
	}
	else
	{
		setValid();
		setText(QObject::tr("Set Heading"));
	}

	// Heading //
	//
	oldHeading_ = track_->getLocalHeading(track_->getSStart());
	newHeading_ = newHeading;

}


/*! \brief .
*
*/
	TrackComponentHeadingCommand
	::~TrackComponentHeadingCommand()
{
	// Clean up //
	//
	if(isUndone())
	{
		// nothing to be done
	}
	else
	{
		// nothing to be done
	}
}


/*! \brief .
*
*/
void
	TrackComponentHeadingCommand
	::redo()
{
	track_->setLocalRotation(newHeading_);
	setRedone();
}


/*! \brief
*
*/
void
	TrackComponentHeadingCommand
	::undo()
{
	track_->setLocalRotation(oldHeading_);
	setUndone();
}
#endif

//##################//
// ArcCurvatureCommand //
//##################//
#if 0
	ArcCurvatureCommand
	::ArcCurvatureCommand(TrackElementArc * arc, double newCurvature, DataCommand * parent)
	:	DataCommand(parent),
		arc_(arc)
{
	// Check for validity //
	//
	if(false)
	{
		setInvalid(); // Invalid because...
		setText(QObject::tr("Set Arc Curvature (invalid!)"));
	}
	else
	{
		setValid();
		setText(QObject::tr("Set Arc Curvature"));
	}

	// Curvature //
	//
	oldCurvature_ = arc_->getCurvature(arc_->getSStart());
	newCurvature_ = newCurvature;

}


/*! \brief .
*
*/
	ArcCurvatureCommand
	::~ArcCurvatureCommand()
{
	// Clean up //
	//
	if(isUndone())
	{
		// nothing to be done
	}
	else
	{
		// nothing to be done
	}
}


/*! \brief .
*
*/
void
	ArcCurvatureCommand
	::redo()
{
	arc_->setCurvature(newCurvature_);
	setRedone();
}


/*! \brief
*
*/
void
	ArcCurvatureCommand
	::undo()
{
	arc_->setCurvature(oldCurvature_);
	setUndone();
}
#endif

//##########################//
// MoveTrackCommand //
//##########################//

MoveTrackCommand::MoveTrackCommand(TrackComponent *track, const QPointF &newPos, DataCommand *parent)
    : DataCommand(parent)
    , track_(track)
    , preTrack_(NULL)
    , postTrack_(NULL)
    , newPos_(newPos)
    , subCommand_(NULL)
{
    if (!track)
    {
        setInvalid();
        setText(QObject::tr("Move track: invalid parameters! No track given."));
        return;
    }

    oldPos_ = track_->getGlobalPoint(track_->getSStart());

    if (oldPos_ == newPos_)
    {
        setInvalid();
        setText(QObject::tr("Move track: no change. Command ignored."));
        return;
    }

    // Get tracks before and after moving track //
    //
    preTrack_ = track->getParentRoad()->getTrackComponentBefore(track->getSStart());
    if (preTrack_ == track)
    {
        preTrack_ = NULL;
    }

    postTrack_ = track->getParentRoad()->getTrackComponent(track->getSEnd());
    if (postTrack_ == track)
    {
        postTrack_ = NULL;
    }

    if (!buildSubCommand())
    {
        setInvalid();
        setText(QObject::tr("Move track: not possible."));
        return;
    }

    setValid();
    setText(QObject::tr("Move track"));
}

MoveTrackCommand::~MoveTrackCommand()
{
    delete subCommand_;
}

bool
MoveTrackCommand::buildSubCommand()
{
    QPointF dPos = newPos_ - oldPos_;

    // Check if translation is valid //
    //
    TrackMoveValidator *validationVisitor = new TrackMoveValidator();
    validationVisitor->setGlobalDeltaPos(dPos);

    QList<TrackComponent *> endPointTracks;
    QList<TrackComponent *> startPointTracks;

    if (postTrack_)
    {
        validationVisitor->setState(TrackMoveValidator::STATE_STARTPOINT);
        postTrack_->accept(validationVisitor);
        startPointTracks.append(postTrack_);
    }

    if (preTrack_)
    {
        validationVisitor->setState(TrackMoveValidator::STATE_ENDPOINT);
        preTrack_->accept(validationVisitor);
        endPointTracks.append(preTrack_);
    }

    if (!validationVisitor->isValid())
    {
        delete validationVisitor;
        return false;
    }
    else
    {
        delete validationVisitor;
        if (subCommand_)
        {
            // Create a new command and merge it with the old one //
            //
            TrackComponentGlobalPointsCommand *subCommand = new TrackComponentGlobalPointsCommand(endPointTracks, startPointTracks, dPos - subCommand_->getDPos(), NULL);
            return subCommand_->mergeWith(subCommand);
        }
        else
        {
            subCommand_ = new TrackComponentGlobalPointsCommand(endPointTracks, startPointTracks, dPos, NULL);
            return true;
        }
    }
}

void
MoveTrackCommand::redo()
{
    track_->setGlobalTranslation(newPos_);

    subCommand_->redo();

    setRedone();
}

void
MoveTrackCommand::undo()
{
    track_->setGlobalTranslation(oldPos_);

    subCommand_->undo();

    setUndone();
}

bool
MoveTrackCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const MoveTrackCommand *command = static_cast<const MoveTrackCommand *>(other);

    // Check tracks //
    //
    if ((track_ == command->track_)
        && (preTrack_ == command->preTrack_)
        && (postTrack_ == command->postTrack_))
    {
        newPos_ = command->newPos_;
        return buildSubCommand();
    }
    else
    {
        return false;
    }
}

//##########################//
// MorphIntoPoly3Command //
//##########################//

MorphIntoPoly3Command::MorphIntoPoly3Command(TrackComponent *track, DataCommand *parent)
    : DataCommand(parent)
    , oldTrack_(track)
    , newTrack_(NULL)
    , parentRoad_(NULL)
{
    if (!oldTrack_)
    {
        setInvalid();
        setText(QObject::tr("Morph track: invalid parameters! No track given."));
        return;
    }

    parentRoad_ = track->getParentRoad();
    if (!parentRoad_)
    {
        setInvalid();
        setText(QObject::tr("Morph track: invalid parameters! Track has no parent road."));
        return;
    }

    // New Track //
    //
    //	if((track->getTrackType() == TrackComponent::DTT_ARC)
    //		|| (track->getTrackType() == TrackComponent::DTT_LINE)
    //		|| (track->getTrackType() == TrackComponent::DTT_SPIRAL)
    //		|| (track->getTrackType() == TrackComponent::DTT_)
    //	)
    //	{

    double endHeading = track->getHeading(track->getSEnd());
    if (endHeading >= 90.0 && endHeading <= 270.0)
    {
        setInvalid();
        setText(QObject::tr("Morph track: sorry, morphing is not possible with these parameters."));
        return;
    }

    double df1 = tan(track->getHeadingRad(track->getSEnd()));
    Polynomial *poly3 = new Polynomial(0.0, 0.0, track->getPoint(track->getSEnd()).y(), df1, track->getPoint(track->getSEnd()).x());

    double s = track->getSStart();
    newTrack_ = new TrackElementPoly3(track->getLocalPoint(s).x(), track->getLocalPoint(s).y(), track->getLocalHeading(s), s, poly3->getCurveLength(0.0, track->getPoint(track->getSEnd()).x()), poly3->getA(), poly3->getB(), poly3->getC(), poly3->getD());

    //	}
    //	else
    //	{
    //		setInvalid();
    //		setText(QObject::tr("Morph track: invalid parameters! Track type not yet supported."));
    //		return;
    //	}

    TrackMoveValidator *validator = new TrackMoveValidator();
    validator->setState(TrackMoveValidator::STATE_ENDPOINT);
    validator->setGlobalDeltaPos(QPointF());
    newTrack_->accept(validator);

    if (!validator->isValid())
    {
        setInvalid();
        setText(QObject::tr("Morph track: sorry, morphing is not possible with these parameters."));
        return;
    }

    setValid();
    setText(QObject::tr("Morph track"));
}

MorphIntoPoly3Command::~MorphIntoPoly3Command()
{
    if (isUndone())
    {
        delete newTrack_;
    }
    else
    {
        delete oldTrack_;
    }
}

void
MorphIntoPoly3Command::redo()
{
    parentRoad_->delTrackComponent(oldTrack_);
    parentRoad_->addTrackComponent(newTrack_);
    parentRoad_->rebuildTrackComponentList();

    setRedone();
}

void
MorphIntoPoly3Command::undo()
{
    parentRoad_->delTrackComponent(newTrack_);
    parentRoad_->addTrackComponent(oldTrack_);
    parentRoad_->rebuildTrackComponentList();

    setUndone();
}

//##########################//
// UngroupTrackCompositeCommand //
//##########################//

// NOTE: this won't work if the TrackComposite has another parent TrackComposite

UngroupTrackCompositeCommand::UngroupTrackCompositeCommand(TrackComposite *composite, DataCommand *parent)
    : DataCommand(parent)
    , oldComposite_(composite)
{
    if (!oldComposite_)
    {
        setInvalid();
        setText(QObject::tr("Ungroup: invalid parameters! No group given."));
        return;
    }

    parentRoad_ = oldComposite_->getParentRoad();
    if (!parentRoad_)
    {
        setInvalid();
        setText(QObject::tr("Ungroup: invalid parameters! Group has no parent road."));
        return;
    }

    foreach (TrackComponent *track, oldComposite_->getChildTrackComponents())
    {
        TrackComponent *clone = track->getClone();
        clone->setGlobalTranslation(oldComposite_->getGlobalPoint(clone->getSStart()));
        clone->setGlobalRotation(oldComposite_->getGlobalHeading(clone->getSStart()));
        newComponents_.append(clone);
    }

    if (newComponents_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Ungroup: invalid parameters! Group has no child elements."));
        return;
    }

    setValid();
    setText(QObject::tr("Ungroup"));
}

UngroupTrackCompositeCommand::~UngroupTrackCompositeCommand()
{
    if (isUndone())
    {
        foreach (TrackComponent *track, newComponents_)
        {
            delete track;
        }
    }
    else
    {
        delete oldComposite_;
    }
}

void
UngroupTrackCompositeCommand::redo()
{

    parentRoad_->delTrackComponent(oldComposite_);

    foreach (TrackComponent *track, newComponents_)
    {
        parentRoad_->addTrackComponent(track);
    }

    parentRoad_->rebuildTrackComponentList();

    setRedone();
}

void
UngroupTrackCompositeCommand::undo()
{
    foreach (TrackComponent *track, newComponents_)
    {
        parentRoad_->delTrackComponent(track);
    }

    parentRoad_->addTrackComponent(oldComposite_);

    parentRoad_->rebuildTrackComponentList();

    setUndone();
}

//##########################//
// TranslateTrackComponentsCommand //
//##########################//

TranslateTrackComponentsCommand::TranslateTrackComponentsCommand(const QMultiMap<TrackComponent *, bool> &selectedTrackComponents, const QPointF &mousePos, const QPointF &pressPos, DataCommand *parent)
    : DataCommand(parent)
    , selectedTrackComponents_(selectedTrackComponents)
{

    // No entries //
    //
    if (selectedTrackComponents_.size() == 0)
    {
        setInvalid();
        setText(QObject::tr("No elements to move."));
        return;
    }

    // DeltaPos //
    //
    QPointF dPos = mousePos - pressPos;

    QList<TrackComponent *> translatedTracks;
    TrackComponent *sparcBefore = NULL;
    TrackComponent *sparcNext = NULL;
    bool midOfLine = false;
    TrackComponent *nextSplineToHandle = NULL;
    double totalLength = 0.0;

    QMultiMap<TrackComponent *, bool>::iterator it = selectedTrackComponents_.begin();

    while (it != selectedTrackComponents_.end())
    {
        QList<TrackMoveProperties *> itTranslateTracks;

        TrackComponent *track = it.key();
        RSystemElementRoad *parentRoad = track->getParentRoad();

        if (translatedTracks.contains(track)) // already done
        {
            it++;
            continue;
        }

        if (track->getStartPosDOF() != 2) // track isn't a sparc
        {
            TrackMoveProperties *props = new TrackMoveProperties(); // look for the next tracks before and after this track
            props->highSlot = track;
            props->lowSlot = NULL;
            itTranslateTracks.append(props);
            track = parentRoad->getTrackComponentBefore(track->getSStart());
            TrackComponent *nextTrack = parentRoad->getTrackComponentNext(it.key()->getSStart());

            if ((it.value() && track && (track->getEndPosDOF() == 1)) || (!it.value() && nextTrack && (nextTrack->getStartPosDOF() == 1))) // handle is in the mid of lines
            {
                midOfLine = true;
            }
            else
            {
                if (track && (it.value() && (track->getEndPosDOF() == 2))) // handle is next to spline
                {
                    nextSplineToHandle = track;
                }
                else if (nextTrack && (!it.value() && (nextTrack->getStartPosDOF() == 2)))
                {
                    nextSplineToHandle = nextTrack;
                }
                totalLength = it.key()->getLength();
            }

            props->lowSlot = track;
            while (track)
            {
                if (track->getStartPosDOF() == 2)
                {
                    sparcBefore = track;
                    break;
                }
                else if (!midOfLine)
                {
                    totalLength += track->getLength();
                }
                props = new TrackMoveProperties();
                props->highSlot = track;
                itTranslateTracks.append(props);

                track = parentRoad->getTrackComponentBefore(track->getSStart());
                props->lowSlot = track;
            };

            track = parentRoad->getTrackComponentNext(it.key()->getSStart());
            TrackComponent *lastTrack = it.key();
            props = new TrackMoveProperties();
            props->highSlot = track;
            props->lowSlot = lastTrack;
            itTranslateTracks.append(props);

            while (track)
            {

                if (track->getEndPosDOF() == 2)
                {
                    sparcNext = track;
                    break;
                }
                else if (!midOfLine)
                {
                    totalLength += track->getLength();
                }
                lastTrack = track;
                track = parentRoad->getTrackComponentNext(track->getSStart());
                props = new TrackMoveProperties();
                props->highSlot = track;
                props->lowSlot = lastTrack;
                itTranslateTracks.append(props);
            }

            if (midOfLine || (!sparcBefore && !sparcNext))
            {
                for (int i = 0; i < itTranslateTracks.size(); i++)
                {
                    itTranslateTracks.at(i)->dPos = dPos;
                    itTranslateTracks.at(i)->transform = TT_MOVE;
                }
            }
            else
            {
                TrackComponent *lastLineFromHandle = parentRoad->getTrackComponentBefore(it.key()->getSStart() + totalLength);

                for (int i = 0; i < itTranslateTracks.size(); i++)
                {
                    if ((itTranslateTracks.at(i)->highSlot && itTranslateTracks.at(i)->lowSlot) && (itTranslateTracks.at(i)->highSlot->getStartPosDOF() == 1) && (itTranslateTracks.at(i)->lowSlot->getStartPosDOF() == 1))
                    {

                        double lengthStart;
                        QVector2D dVec;
                        if (it.value())
                        {
                            lengthStart = totalLength - (itTranslateTracks.at(i)->highSlot->getSStart() - it.key()->getSStart());
                        }
                        else
                        {
                            lengthStart = totalLength - (it.key()->getSEnd() - itTranslateTracks.at(i)->lowSlot->getSEnd());
                        }

                        dVec = (lengthStart * QVector2D(dPos).length() / totalLength) * QVector2D(dPos).normalized();
                        itTranslateTracks.at(i)->dPos = dVec.toPointF();
                        itTranslateTracks.at(i)->transform = TT_MOVE;
                    }
                    if (!nextSplineToHandle && ((!itTranslateTracks.at(i)->lowSlot && (itTranslateTracks.at(i)->highSlot == (it).key())) || (!itTranslateTracks.at(i)->highSlot && (itTranslateTracks.at(i)->lowSlot == (it).key()))))
                    {

                        itTranslateTracks.at(i)->transform = TT_MOVE;
                        itTranslateTracks.at(i)->dPos = dPos;
                    }
                    else if ((itTranslateTracks.at(i)->highSlot == nextSplineToHandle) || (itTranslateTracks.at(i)->lowSlot == nextSplineToHandle))
                    {

                        itTranslateTracks.at(i)->transform = TT_MOVE | TT_ROTATE;
                        itTranslateTracks.at(i)->dPos = dPos;
                    }
                    else if ((itTranslateTracks.at(i)->highSlot && (itTranslateTracks.at(i)->highSlot->getStartPosDOF() == 2)) || (itTranslateTracks.at(i)->lowSlot && (itTranslateTracks.at(i)->lowSlot->getStartPosDOF() == 2)) || ((!itTranslateTracks.at(i)->lowSlot && !nextSplineToHandle && (itTranslateTracks.at(i)->highSlot == lastLineFromHandle)) || (!itTranslateTracks.at(i)->highSlot && (itTranslateTracks.at(i)->lowSlot == lastLineFromHandle))))

                    {
                        itTranslateTracks.at(i)->transform = TT_ROTATE;
                        itTranslateTracks.at(i)->dPos = dPos;

                        QPointF newPos;

                        if (it.value())
                        {
                            newPos = mousePos - lastLineFromHandle->getGlobalPoint(lastLineFromHandle->getSEnd());
                        }
                        else
                        {
                            newPos = lastLineFromHandle->getGlobalPoint(lastLineFromHandle->getSStart()) - mousePos;
                        }

                        double heading = (atan2(newPos.y(), newPos.x()) * 360.0 / (2.0 * M_PI));
                        itTranslateTracks.at(i)->heading = heading + 180;
                    }
                }
            }

            for (int i = 0; i < itTranslateTracks.size(); i++)
            {
                if (selectedTrackComponents_.contains(itTranslateTracks.at(i)->highSlot) && !translatedTracks.contains(itTranslateTracks.at(i)->highSlot))
                {
                    translatedTracks.append(itTranslateTracks.at(i)->highSlot);
                }
                if (selectedTrackComponents_.contains(itTranslateTracks.at(i)->lowSlot) && !translatedTracks.contains(itTranslateTracks.at(i)->lowSlot))
                {
                    translatedTracks.append(itTranslateTracks.at(i)->lowSlot);
                }
            }

            if (it.value()) // Tracks have to be sorted to perform the translations in the right order
            {
                for (int i = itTranslateTracks.size() - 1; i >= 0; i--)
                {
                    translateTracks_.append(itTranslateTracks.at(i));
                }
            }
            else
            {
                for (int i = 0; i < itTranslateTracks.size(); i++)
                {
                    translateTracks_.append(itTranslateTracks.at(i));
                }
            }
            itTranslateTracks.clear();
        }

        it++;
    }

    it = selectedTrackComponents_.begin();
    while (it != selectedTrackComponents_.end())
    {
        TrackComponent *track = it.key();
        RSystemElementRoad *parentRoad = track->getParentRoad();

        if (translatedTracks.contains(track) || (track->getStartPosDOF() != 2)) // only sparcs should be left)
        {
            it++;
            continue;
        }

        TrackMoveProperties *props = new TrackMoveProperties();
        props->dPos = dPos;
        props->transform = TT_MOVE;
        if (it.value())
        {
            props->highSlot = track;
            props->lowSlot = parentRoad->getTrackComponentBefore(track->getSStart());
        }
        else
        {
            props->highSlot = parentRoad->getTrackComponentNext(it.key()->getSStart());
            props->lowSlot = track;
        }

        translateTracks_.append(props);

        if (selectedTrackComponents_.contains(props->highSlot) && !translatedTracks.contains(props->highSlot))
        {
            translatedTracks.append(props->highSlot);
        }
        if (selectedTrackComponents_.contains(props->lowSlot) && !translatedTracks.contains(props->lowSlot))
        {
            translatedTracks.append(props->lowSlot);
        }

        it++;
    }

    translatedTracks.clear();

    for (int i = 0; i < translateTracks_.size(); i++)
    {
        if (!validate(translateTracks_.at(i)))
        {
            setInvalid();
            setText(QObject::tr("Translation not valid"));
            return;
        }

        if (translateTracks_.at(i)->highSlot)
        {

            RSystemElementRoad *parentRoad = translateTracks_.at(i)->highSlot->getParentRoad();
            if (!roads_.contains(parentRoad))
            {
                roads_.append(parentRoad);
            }
        }
        if (translateTracks_.at(i)->lowSlot)
        {

            RSystemElementRoad *parentRoad = translateTracks_.at(i)->lowSlot->getParentRoad();
            if (!roads_.contains(parentRoad))
            {
                roads_.append(parentRoad);
            }
        }
    }

    // sections to remove //
    //
    for (int i = 0; i < roads_.size(); i++)
    {
        // typesections //
        if (!roads_.at(i)->getTypeSections().isEmpty())
        {
            foreach (TypeSection *section, roads_.at(i)->getTypeSections())
            {
                if (roads_.at(i)->getLength() - section->getSStart() < NUMERICAL_ZERO6)
                {
                    typeSections_.insert(i, section);
                }
            }
        }

        // elevation //
        if (!roads_.at(i)->getElevationSections().isEmpty())
        {
            foreach (ElevationSection *section, roads_.at(i)->getElevationSections())
            {
                if (roads_.at(i)->getLength() - section->getSStart() < NUMERICAL_ZERO6)
                {
                    elevationSections_.insert(i, section);
                }
            }
        }

        // superelevation //
        if (!roads_.at(i)->getSuperelevationSections().isEmpty())
        {
            foreach (SuperelevationSection *section, roads_.at(i)->getSuperelevationSections())
            {
                if (roads_.at(i)->getLength() - section->getSStart() < NUMERICAL_ZERO6)
                {
                    superelevationSections_.insert(i, section);
                }
            }
        }

        // crossfall //
        if (!roads_.at(i)->getCrossfallSections().isEmpty())
        {
            foreach (CrossfallSection *section, roads_.at(i)->getCrossfallSections())
            {
                if (roads_.at(i)->getLength() - section->getSStart() < NUMERICAL_ZERO6)
                {
                    crossfallSections_.insert(i, section);
                }
            }
        }

        // lanes //
        if (!roads_.at(i)->getLaneSections().isEmpty())
        {
            foreach (LaneSection *section, roads_.at(i)->getLaneSections())
            {
                if (roads_.at(i)->getLength() - section->getSStart() < NUMERICAL_ZERO6) // completely outside
                {
                    laneSections_.insert(i, section);
                }
                else if (roads_.at(i)->getLength() - section->getSEnd() < NUMERICAL_ZERO6)
                {
                    laneSections_.insert(section->getSStart(), section);
                    LaneSection *newSection;

                    newSection = section->getClone(section->getSStart(), roads_.at(i)->getLength());
                    laneSectionsAdd_.insert(i, newSection);
                }
            }
        }

		// shape //
		if (!roads_.at(i)->getShapeSections().isEmpty())
		{
			foreach(ShapeSection *section, roads_.at(i)->getShapeSections())
			{
				if (roads_.at(i)->getLength() - section->getSStart() < NUMERICAL_ZERO6)
				{
					shapeSections_.insert(i, section);
				}
			}
		}
    }

    setValid();
    setText(QObject::tr("Move"));
}

/*! \brief .
*
*/
TranslateTrackComponentsCommand::~TranslateTrackComponentsCommand()
{
}

bool
TranslateTrackComponentsCommand::validate(TrackMoveProperties *props)
{

    if (!props)
    {
        return true;
    }

    TrackMoveValidator *validationVisitor = new TrackMoveValidator();

    if (props->transform & TT_MOVE)
    {
        validationVisitor->setGlobalDeltaPos(props->dPos);
        if (props->highSlot)
        {
            validationVisitor->setState(TrackMoveValidator::STATE_STARTPOINT);
            props->highSlot->accept(validationVisitor);
        }

        if (props->lowSlot)
        {
            validationVisitor->setState(TrackMoveValidator::STATE_ENDPOINT);
            props->lowSlot->accept(validationVisitor);
        }

        // Check if translation is valid //
        //
        if (!validationVisitor->isValid())
        {
            delete validationVisitor;
            return false; // not valid => no translation
        }

        delete validationVisitor;
        validationVisitor = new TrackMoveValidator();

        if (props->transform & TT_ROTATE)
        {
            if (props->highSlot && (props->highSlot->getStartPosDOF() == 2))
            {
                TrackComponent *highSlotCopy = props->highSlot->getClone();
                highSlotCopy->setGlobalStartPoint(highSlotCopy->getGlobalPoint(highSlotCopy->getSStart()) + props->dPos);

                QPointF endPoint = props->lowSlot->getGlobalPoint(props->lowSlot->getSEnd()) + props->dPos;
                QPointF startPoint = props->lowSlot->getLocalPoint(props->lowSlot->getSStart());
                QVector2D line = QVector2D(endPoint - startPoint);

                props->oldHeading = props->highSlot->getGlobalHeading(props->highSlot->getSStart());
                props->heading = atan2(line.y(), line.x()) * 360.0 / (2.0 * M_PI);
                validationVisitor->setGlobalHeading(props->heading);

                validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
                highSlotCopy->accept(validationVisitor);

                // Check if rotation is valid //
                //
                if (!validationVisitor->isValid())
                {
                    delete validationVisitor;
                    qDebug() << "Validate Rotate: not valid";
                    return false; // not valid => no translation
                }
                delete highSlotCopy;
            }

            if ((props->lowSlot) && (props->lowSlot->getEndPosDOF() == 2))
            {
                TrackComponent *lowSlotCopy = props->lowSlot->getClone();
                lowSlotCopy->setGlobalEndPoint(lowSlotCopy->getGlobalPoint(lowSlotCopy->getSEnd()) + props->dPos);

                QPointF startPoint = props->highSlot->getGlobalPoint(props->highSlot->getSStart()) + props->dPos;
                QPointF endPoint = props->highSlot->getLocalPoint(props->highSlot->getSEnd());
                QVector2D line = QVector2D(endPoint - startPoint);

                props->oldHeading = props->lowSlot->getGlobalHeading(props->lowSlot->getSEnd());
                props->heading = atan2(line.y(), line.x()) * 360.0 / (2.0 * M_PI);
                validationVisitor->setGlobalHeading(props->heading);

                validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
                lowSlotCopy->accept(validationVisitor);

                // Check if rotation is valid //
                //
                if (!validationVisitor->isValid())
                {
                    delete validationVisitor;
                    qDebug() << "Validate Rotate: not valid";
                    return false; // not valid => no translation
                }
                delete lowSlotCopy;
            }
        }
    }

    else if (props->transform & TT_ROTATE)
    {
        if (props->highSlot && (props->highSlot->getStartPosDOF() == 2))
        {
            QPointF endPoint = props->lowSlot->getLocalPoint(props->lowSlot->getSEnd());
            QPointF startPoint = props->lowSlot->getGlobalPoint(props->lowSlot->getSStart()) + props->dPos;
            QVector2D line = QVector2D(endPoint - startPoint);

            props->oldHeading = props->highSlot->getGlobalHeading(props->highSlot->getSStart());
            props->heading = atan2(line.y(), line.x()) * 360.0 / (2.0 * M_PI);
            validationVisitor->setGlobalHeading(props->heading);
            validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
            props->highSlot->accept(validationVisitor);

            // Check if rotation is valid //
            //
            if (!validationVisitor->isValid())
            {
                delete validationVisitor;
                qDebug() << "Validate Rotate: not valid";
                return false; // not valid => no translation
            }
        }

        if (props->lowSlot && (props->lowSlot->getEndPosDOF() == 2))
        {
            QPointF startPoint = props->highSlot->getLocalPoint(props->highSlot->getSStart());
            QPointF endPoint = props->highSlot->getGlobalPoint(props->highSlot->getSEnd()) + props->dPos;
            QVector2D line = QVector2D(endPoint - startPoint);

            props->oldHeading = props->lowSlot->getGlobalHeading(props->lowSlot->getSEnd());
            props->heading = atan2(line.y(), line.x()) * 360.0 / (2.0 * M_PI);
            validationVisitor->setGlobalHeading(props->heading);
            validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
            props->lowSlot->accept(validationVisitor);

            // Check if rotation is valid //
            //
            if (!validationVisitor->isValid())
            {
                delete validationVisitor;
                qDebug() << "Validate Rotate: not valid";
                return false; // not valid => no translation
            }
        }
    }

    delete validationVisitor;

    return true;
}

void
TranslateTrackComponentsCommand::translate(TrackMoveProperties *props)
{

    // Translate all handles //
    //

    if (props->transform & TT_MOVE)
    {
        if (props->lowSlot)
        {
            props->lowSlot->setGlobalEndPoint(props->lowSlot->getGlobalPoint(props->lowSlot->getSEnd()) + props->dPos);
        }
        if (props->highSlot)
        {
            props->highSlot->setGlobalStartPoint(props->highSlot->getGlobalPoint(props->highSlot->getSStart()) + props->dPos);
        }
    }

    if (props->transform & TT_ROTATE)
    {

        if (props->lowSlot && (props->lowSlot->getEndPosDOF() == 2))
        {
            props->lowSlot->setGlobalEndHeading(props->heading);
        }
        else if (props->highSlot && (props->highSlot->getStartPosDOF() == 2))
        {
            props->highSlot->setGlobalStartHeading(props->heading);
        }
        else if (props->lowSlot && props->highSlot)
        {
            props->lowSlot->setGlobalEndHeading(props->heading);
            props->highSlot->setGlobalStartHeading(props->heading);
        }
    }
}

void
TranslateTrackComponentsCommand::undoTranslate(TrackMoveProperties *props)
{

    // Undo Translate all handles //
    //

    if (props->transform & TT_ROTATE)
    {

        if (props->lowSlot && (props->lowSlot->getEndPosDOF() == 2))
        {
            props->lowSlot->setGlobalEndHeading(props->oldHeading);
        }
        else if (props->highSlot && (props->highSlot->getStartPosDOF() == 2))
        {
            props->highSlot->setGlobalStartHeading(props->oldHeading);
        }
        else if (props->lowSlot || props->highSlot)
        {
            props->highSlot->setGlobalStartHeading(props->oldHeading);
            props->lowSlot->setGlobalEndHeading(props->oldHeading);
        }
    }

    if (props->transform & TT_MOVE)
    {
        if (props->lowSlot)
        {
            props->lowSlot->setGlobalEndPoint(props->lowSlot->getGlobalPoint(props->lowSlot->getSEnd()) - props->dPos);
        }
        if (props->highSlot)
        {
            props->highSlot->setGlobalStartPoint(props->highSlot->getGlobalPoint(props->highSlot->getSStart()) - props->dPos);
        }
    }
}

/*! \brief .
*
*/
void
TranslateTrackComponentsCommand::redo()
{
    for (int i = 0; i < translateTracks_.size(); i++)
    {

        translate(translateTracks_.at(i));
    }

    for (int i = 0; i < roads_.size(); i++)
    {
        roads_.at(i)->rebuildTrackComponentList();

        QMap<int, TypeSection *>::const_iterator iterType = typeSections_.find(i);
        while ((iterType != typeSections_.end()) && (iterType.key() == i))
        {
            roads_.at(i)->delTypeSection(iterType.value());
            iterType++;
        }

        QMap<int, ElevationSection *>::const_iterator iterElevation = elevationSections_.find(i);
        while ((iterElevation != elevationSections_.end()) && (iterElevation.key() == i))
        {
            roads_.at(i)->delElevationSection(iterElevation.value());
            iterElevation++;
        }

        QMap<int, SuperelevationSection *>::const_iterator iterSuperelevation = superelevationSections_.find(i);
        while ((iterSuperelevation != superelevationSections_.end()) && (iterSuperelevation.key() == i))
        {
            roads_.at(i)->delSuperelevationSection(iterSuperelevation.value());
            iterSuperelevation++;
        }

        QMap<int, CrossfallSection *>::const_iterator iterCrossfall = crossfallSections_.find(i);
        while ((iterCrossfall != crossfallSections_.end()) && (iterCrossfall.key() == i))
        {
            roads_.at(i)->delCrossfallSection(iterCrossfall.value());
            iterCrossfall++;
        }

        QMap<int, LaneSection *>::const_iterator iterLane = laneSections_.find(i);
        while ((iterLane != laneSections_.end()) && (iterLane.key() == i))
        {
            roads_.at(i)->delLaneSection(iterLane.value());
            iterLane++;
        }

        QMap<int, LaneSection *>::const_iterator iterLaneAdd = laneSectionsAdd_.find(i);
        while ((iterLane != laneSectionsAdd_.end()) && (iterLaneAdd.key() == i))
        {
            roads_.at(i)->addLaneSection(iterLaneAdd.value());
            iterLaneAdd++;
        }

		QMap<int, ShapeSection *>::const_iterator iterShape = shapeSections_.find(i);
		while ((iterShape != shapeSections_.end()) && (iterShape.key() == i))
		{
			roads_.at(i)->delShapeSection(iterShape.value());
			iterShape++;
		}
    }

    setRedone();
}

/*! \brief
*
*/
void
TranslateTrackComponentsCommand::undo()
{
    for (int i = translateTracks_.size() - 1; i >= 0; i--)
    {

        undoTranslate(translateTracks_.at(i));
    }

    for (int i = 0; i < roads_.size(); i++)
    {
        roads_.at(i)->rebuildTrackComponentList();

        QMap<int, TypeSection *>::const_iterator iterType = typeSections_.find(i);
        while ((iterType != typeSections_.end()) && (iterType.key() == i))
        {
            roads_.at(i)->addTypeSection(iterType.value());
            iterType++;
        }

        QMap<int, ElevationSection *>::const_iterator iterElevation = elevationSections_.find(i);
        while ((iterElevation != elevationSections_.end()) && (iterElevation.key() == i))
        {
            roads_.at(i)->addElevationSection(iterElevation.value());
            iterElevation++;
        }

        QMap<int, SuperelevationSection *>::const_iterator iterSuperelevation = superelevationSections_.find(i);
        while ((iterSuperelevation != superelevationSections_.end()) && (iterSuperelevation.key() == i))
        {
            roads_.at(i)->addSuperelevationSection(iterSuperelevation.value());
            iterSuperelevation++;
        }

        QMap<int, CrossfallSection *>::const_iterator iterCrossfall = crossfallSections_.find(i);
        while ((iterCrossfall != crossfallSections_.end()) && (iterCrossfall.key() == i))
        {
            roads_.at(i)->addCrossfallSection(iterCrossfall.value());
            iterCrossfall++;
        }

        QMap<int, LaneSection *>::const_iterator iterLane = laneSections_.find(i);
        while ((iterLane != laneSections_.end()) && (iterLane.key() == i))
        {
            roads_.at(i)->addLaneSection(iterLane.value());
            iterLane++;
        }

		QMap<int, ShapeSection *>::const_iterator iterShape = shapeSections_.find(i);
		while ((iterShape != shapeSections_.end()) && (iterShape.key() == i))
		{
			roads_.at(i)->addShapeSection(iterShape.value());
			iterShape++;
		}
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
TranslateTrackComponentsCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const TranslateTrackComponentsCommand *command = static_cast<const TranslateTrackComponentsCommand *>(other);

    // Check selected tracks
    //
    if (selectedTrackComponents_.size() != command->selectedTrackComponents_.size())
    {
        qDebug() << "Selection not equal";
        return false;
    }

    QMultiMap<TrackComponent *, bool>::const_iterator it = selectedTrackComponents_.begin();
    QMultiMap<TrackComponent *, bool>::const_iterator itOther = command->selectedTrackComponents_.begin();

    while (it != selectedTrackComponents_.end())
    {
        if ((*it) != (*itOther))
        {
            qDebug() << "Selected Tracks are not equal";
            return false;
        }
        it++;
        itOther++;
    }

    // Check roads and tracks
    //
    if (translateTracks_.size() != command->translateTracks_.size())
    {
        qDebug() << "Tracks to translate not equal";
        return false;
    }

    // Success //
    //

    for (int i = 0; i < translateTracks_.size(); i++)
    {
        translateTracks_.at(i)->dPos += command->translateTracks_.at(i)->dPos;
    }

    for (int i = 0; i < roads_.size(); i++)
    {
        QMap<int, TypeSection *>::const_iterator iterType = command->typeSections_.find(i);
        while ((iterType != command->typeSections_.end()) && (iterType.key() == i))
        {
            typeSections_.insert(i, iterType.value());
            iterType++;
        }

        QMap<int, ElevationSection *>::const_iterator iterElevation = command->elevationSections_.find(i);
        while ((iterElevation != command->elevationSections_.end()) && (iterElevation.key() == i))
        {
            elevationSections_.insert(i, iterElevation.value());
            iterElevation++;
        }

        QMap<int, SuperelevationSection *>::const_iterator iterSuperelevation = command->superelevationSections_.find(i);
        while ((iterSuperelevation != command->superelevationSections_.end()) && (iterSuperelevation.key() == i))
        {
            superelevationSections_.insert(i, iterSuperelevation.value());
            iterSuperelevation++;
        }

        QMap<int, CrossfallSection *>::const_iterator iterCrossfall = command->crossfallSections_.find(i);
        while ((iterCrossfall != command->crossfallSections_.end()) && (iterCrossfall.key() == i))
        {
            crossfallSections_.insert(i, iterCrossfall.value());
            iterCrossfall++;
        }

        QMap<int, LaneSection *>::const_iterator iterLane = command->laneSections_.find(i);
        while ((iterLane != command->laneSections_.end()) && (iterLane.key() == i))
        {
            laneSections_.insert(i, iterLane.value());
            iterLane++;
        }

		QMap<int, ShapeSection *>::const_iterator iterShape = command->shapeSections_.find(i);
		while ((iterShape != command->shapeSections_.end()) && (iterShape.key() == i))
		{
			shapeSections_.insert(i, iterShape.value());
			iterShape++;
		}
    }

    return true;
}
