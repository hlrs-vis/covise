/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/29/2010
**
**************************************************************************/

#include "lanesectioncommands.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/roadlink.hpp"

#include <math.h>

#include <QList>
#include <QPointF>
#include <QMap>
#include <QMultiMap>

//###########################//
// InsertLaneCommand //
//###########################//

InsertLaneCommand::InsertLaneCommand(LaneSection *laneSection, Lane *lane, DataCommand *parent)
    : DataCommand(parent)
    , laneSection_(laneSection)
    , newLane_(lane)
{
    // Check for validity //
    //
    if (!newLane_ || !laneSection_)
    {
        setInvalid();
        setText(QObject::tr("Insert lane: Parameters invalid! No element given or no parent."));
        return;
    }

    id_ = newLane_->getId();
    if (id_ == 0)
    {
        setInvalid();
        setText(QObject::tr("Insert lane: can't insert a lane with id 0."));
        return;
    }

    setValid();
    setText(QObject::tr("Insert lane"));
}

InsertLaneCommand::~InsertLaneCommand()
{
    if (isUndone())
    {
        delete newLane_;
    }
}

void
InsertLaneCommand::redo()
{

    if (laneSection_->getLanes()[id_] != NULL)
    {
        if (id_ < 0)
        {
            for (int i = laneSection_->getRightmostLaneId(); i <= id_;  i++)
            {
                Lane *lane = laneSection_->getLane(i);
                if (lane)
                {
                    lane->setId(i-1);
                }
            }
        }
        else
        {
            for (int i = laneSection_->getLeftmostLaneId(); i >= id_;  i--)
            {
                Lane *lane = laneSection_->getLane(i);
                if (lane)
                {
                    lane->setId(i+1);
                }
            }
        }
    }

    laneSection_->addLane(newLane_);


    setRedone();
}

void
InsertLaneCommand::undo()
{

    laneSection_->removeLane(newLane_);

    if (id_ < 0)
    {
        for (int i = id_ - 1; i >= laneSection_->getRightmostLaneId(); i--)
        {
            Lane *lane = laneSection_->getLane(i);
            if (lane)
            {
                lane->setId(i+1);
            }
        }
    }
    else
    {
        for (int i = id_ + 1; i <= laneSection_->getLeftmostLaneId(); i++)
        {
            Lane *lane = laneSection_->getLane(i);
            if (lane)
            {
                lane->setId(i-1);
            }
        }

    }

    setUndone();
}

//###########################//
// RemoveLaneCommand //
//###########################//

RemoveLaneCommand::RemoveLaneCommand(LaneSection *laneSection, Lane *lane, DataCommand *parent)
    : DataCommand(parent)
    , laneSection_(laneSection)
    , oldLane_(lane)
{
    // Check for validity //
    //
    if (!oldLane_ || !laneSection_)
    {
        setInvalid();
        setText(QObject::tr("Remove lane: Parameters invalid! No element given or no parent."));
        return;
    }

    id_ = oldLane_->getId();
    if (id_ == 0)
    {
        setInvalid();
        setText(QObject::tr("Remove lane: can't remove a lane with id 0."));
        return;
    }

    setValid();
    setText(QObject::tr("Remove lane"));
}

RemoveLaneCommand::~RemoveLaneCommand()
{
    if (!isUndone())
    {
        delete oldLane_;
    }
}

void
RemoveLaneCommand::redo()
{

    laneSection_->removeLane(oldLane_);

    Lane *lane;
    if (id_ < 0)
    {
        int i = id_;
        while ((lane = laneSection_->getNextLower(i)) != NULL)
        {
            int index = lane->getId();
            lane->setId(i);
            i = index;
        }
    }
    else
    {
        int i = id_;
        while ((lane = laneSection_->getNextUpper(i)) != NULL)
        {
            int index = lane->getId();
            lane->setId(i);
            i = index;
        }
    }

    setRedone();
}

void
RemoveLaneCommand::undo()
{
    Lane *lane;
    if (id_ < 0)
    {
        lane = laneSection_->getFirst();
        int i;
        while (lane && ((i = lane->getId()) <= id_))
        {
            lane->setId(i - 1);
            lane = laneSection_->getNextUpper(i);
        }
    }
    else
    {
        lane = laneSection_->getLast();
        int i;
        while (lane && ((i = lane->getId()) >= id_))
        {
            lane->setId(i + 1);
            lane = laneSection_->getNextLower(i);
        }
    }
    laneSection_->addLane(oldLane_);

    setUndone();
}

//################//
// InsertLaneWidthCommand //
//################//

InsertLaneWidthCommand::InsertLaneWidthCommand(Lane *lane, LaneWidth *laneWidth, LaneBorder *laneBorder, DataCommand *parent)
    : DataCommand(parent)
    , lane_(lane)
    , newLaneWidth_(laneWidth)
	, newLaneBorder_(laneBorder)
{
    // Check for validity //
    //
    if ((!newLaneWidth_  && !newLaneBorder_) || !lane_)
    {
        setInvalid();
        setText(QObject::tr("Insert laneWidth: Parameters invalid! No element given or no parent."));
        return;
    }

    setValid();
    setText(QObject::tr("Insert lanewidth"));
}

InsertLaneWidthCommand::~InsertLaneWidthCommand()
{
    if (isUndone())
    {
        delete newLaneWidth_;
    }
}

void
InsertLaneWidthCommand::redo()
{
	if (newLaneWidth_)
	{
		lane_->addWidthEntry(newLaneWidth_);
	}
	else
	{
		lane_->addWidthEntry(newLaneBorder_);
	}
	lane_->getParentLaneSection()->addLaneSectionChanges(LaneSection::CLS_LanesWidthsChanged);
    setRedone();
}

void
InsertLaneWidthCommand::undo()
{
	if (newLaneWidth_)
	{
		lane_->delWidthEntry(newLaneWidth_);
	}
	else
	{
		lane_->delWidthEntry(newLaneBorder_);
	}
	lane_->getParentLaneSection()->addLaneSectionChanges(LaneSection::CLS_LanesWidthsChanged);

    setUndone();
}

//###########################//
// SetLaneRoadMarkSOffsetCommand //
//###########################//

SetLaneRoadMarkSOffsetCommand::SetLaneRoadMarkSOffsetCommand(LaneRoadMark *mark, double sOffset, DataCommand *parent)
    : DataCommand(parent)
    , mark_(mark)
    , newSOffset_(sOffset)
{
    // Check for validity //
    //
    if (!mark || (sOffset == mark_->getSOffset()))
    {
        setInvalid();
        setText(QObject::tr("Set Road Mark Offset: Parameters invalid! No element given or no change."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Mark Offset"));
    }

    // Road Type //
    //
    oldSOffset_ = mark_->getSOffset();
}

SetLaneRoadMarkSOffsetCommand::~SetLaneRoadMarkSOffsetCommand()
{
}

void
SetLaneRoadMarkSOffsetCommand::redo()
{
    mark_->setSOffset(newSOffset_);

    setRedone();
}

void
SetLaneRoadMarkSOffsetCommand::undo()
{
    mark_->setSOffset(oldSOffset_);

    setUndone();
}

bool
SetLaneRoadMarkSOffsetCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids and parse //
    //
    if (other->id() != id())
    {
        return false;
    }
    const SetLaneRoadMarkSOffsetCommand *command = static_cast<const SetLaneRoadMarkSOffsetCommand *>(other);

    // Check parameters //
    //
    if (mark_ != command->mark_)
    {
        return false;
    }

    // Success //
    //
    newSOffset_ = command->newSOffset_;
    return true;
}

//###########################//
// SetLaneRoadMarkTypeCommand //
//###########################//

SetLaneRoadMarkTypeCommand::SetLaneRoadMarkTypeCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkType type, DataCommand *parent)
    : DataCommand(parent)
    , marks_(marks)
    , newType_(type)
{
    construct();
}

SetLaneRoadMarkTypeCommand::SetLaneRoadMarkTypeCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkType type, DataCommand *parent)
    : DataCommand(parent)
    , newType_(type)
{
    marks_.append(mark);
    construct();
}

void
SetLaneRoadMarkTypeCommand::construct()
{
    // Check for validity //
    //
    if (marks_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Set Road Mark Type: Parameters invalid! No element given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Mark Type"));
    }

    // Old Types //
    //
    foreach (LaneRoadMark *element, marks_)
    {
        oldTypes_.append(element->getRoadMarkType());
    }
}

SetLaneRoadMarkTypeCommand::~SetLaneRoadMarkTypeCommand()
{
}

void
SetLaneRoadMarkTypeCommand::redo()
{
    foreach (LaneRoadMark *element, marks_)
    {
        element->setRoadMarkType(newType_);
    }

    setRedone();
}

void
SetLaneRoadMarkTypeCommand::undo()
{
    for (int i = 0; i < marks_.size(); ++i)
    {
        marks_.at(i)->setRoadMarkType(oldTypes_.at(i));
    }

    setUndone();
}

//###########################//
// SetLaneRoadMarkWeightCommand //
//###########################//

SetLaneRoadMarkWeightCommand::SetLaneRoadMarkWeightCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkWeight weight, DataCommand *parent)
    : DataCommand(parent)
    , marks_(marks)
    , newWeight_(weight)
{
    construct();
}

SetLaneRoadMarkWeightCommand::SetLaneRoadMarkWeightCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkWeight weight, DataCommand *parent)
    : DataCommand(parent)
    , newWeight_(weight)
{
    marks_.append(mark);
    construct();
}

void
SetLaneRoadMarkWeightCommand::construct()
{
    // Check for validity //
    //
    if (marks_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Set Road Mark Weight: Parameters invalid! No element given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Mark Weight"));
    }

    // Old Weights //
    //
    foreach (LaneRoadMark *element, marks_)
    {
        oldWeights_.append(element->getRoadMarkWeight());
    }
}

SetLaneRoadMarkWeightCommand::~SetLaneRoadMarkWeightCommand()
{
}

void
SetLaneRoadMarkWeightCommand::redo()
{
    foreach (LaneRoadMark *element, marks_)
    {
        element->setRoadMarkWeight(newWeight_);
    }

    setRedone();
}

void
SetLaneRoadMarkWeightCommand::undo()
{
    for (int i = 0; i < marks_.size(); ++i)
    {
        marks_.at(i)->setRoadMarkWeight(oldWeights_.at(i));
    }

    setUndone();
}

//###########################//
// SetLaneRoadMarkColorCommand //
//###########################//

SetLaneRoadMarkColorCommand::SetLaneRoadMarkColorCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkColor color, DataCommand *parent)
    : DataCommand(parent)
    , marks_(marks)
    , newColor_(color)
{
    construct();
}

SetLaneRoadMarkColorCommand::SetLaneRoadMarkColorCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkColor color, DataCommand *parent)
    : DataCommand(parent)
    , newColor_(color)
{
    marks_.append(mark);
    construct();
}

void
SetLaneRoadMarkColorCommand::construct()
{
    // Check for validity //
    //
    if (marks_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Set Road Mark Color: Parameters invalid! No element given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Mark Color"));
    }

    // Old Colors //
    //
    foreach (LaneRoadMark *element, marks_)
    {
        oldColors_.append(element->getRoadMarkColor());
    }
}

SetLaneRoadMarkColorCommand::~SetLaneRoadMarkColorCommand()
{
}

void
SetLaneRoadMarkColorCommand::redo()
{
    foreach (LaneRoadMark *element, marks_)
    {
        element->setRoadMarkColor(newColor_);
    }

    setRedone();
}

void
SetLaneRoadMarkColorCommand::undo()
{
    for (int i = 0; i < marks_.size(); ++i)
    {
        marks_.at(i)->setRoadMarkColor(oldColors_.at(i));
    }

    setUndone();
}

//###########################//
// SetLaneRoadMarkWidthCommand //
//###########################//

SetLaneRoadMarkWidthCommand::SetLaneRoadMarkWidthCommand(QList<LaneRoadMark *> marks, double width, DataCommand *parent)
    : DataCommand(parent)
    , marks_(marks)
    , newWidth_(width)
{
    construct();
}

SetLaneRoadMarkWidthCommand::SetLaneRoadMarkWidthCommand(LaneRoadMark *mark, double width, DataCommand *parent)
    : DataCommand(parent)
    , newWidth_(width)
{
    marks_.append(mark);
    construct();
}

void
SetLaneRoadMarkWidthCommand::construct()
{
    // Check for validity //
    //
    if (marks_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Set Road Mark Width: Parameters invalid! No element given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Mark Width"));
    }

    // Old Widths //
    //
    foreach (LaneRoadMark *element, marks_)
    {
        oldWidths_.append(element->getRoadMarkWidth());
    }
}

SetLaneRoadMarkWidthCommand::~SetLaneRoadMarkWidthCommand()
{
}

void
SetLaneRoadMarkWidthCommand::redo()
{
    foreach (LaneRoadMark *element, marks_)
    {
        element->setRoadMarkWidth(newWidth_);
    }

    setRedone();
}

void
SetLaneRoadMarkWidthCommand::undo()
{
    for (int i = 0; i < marks_.size(); ++i)
    {
        marks_.at(i)->setRoadMarkWidth(oldWidths_.at(i));
    }

    setUndone();
}

//###########################//
// SetLaneRoadMarkLaneChangeCommand //
//###########################//

SetLaneRoadMarkLaneChangeCommand::SetLaneRoadMarkLaneChangeCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkLaneChange laneChange, DataCommand *parent)
    : DataCommand(parent)
    , marks_(marks)
    , newLaneChange_(laneChange)
{
    construct();
}

SetLaneRoadMarkLaneChangeCommand::SetLaneRoadMarkLaneChangeCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkLaneChange laneChange, DataCommand *parent)
    : DataCommand(parent)
    , newLaneChange_(laneChange)
{
    marks_.append(mark);
    construct();
}

void
SetLaneRoadMarkLaneChangeCommand::construct()
{
    // Check for validity //
    //
    if (marks_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Set Road Mark LaneChange: Parameters invalid! No element given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Road Mark LaneChange"));
    }

    // Old LaneChanges //
    //
    foreach (LaneRoadMark *element, marks_)
    {
        oldLaneChanges_.append(element->getRoadMarkLaneChange());
    }
}

SetLaneRoadMarkLaneChangeCommand::~SetLaneRoadMarkLaneChangeCommand()
{
}

void
SetLaneRoadMarkLaneChangeCommand::redo()
{
    foreach (LaneRoadMark *element, marks_)
    {
        element->setRoadMarkLaneChange(newLaneChange_);
    }

    setRedone();
}

void
SetLaneRoadMarkLaneChangeCommand::undo()
{
    for (int i = 0; i < marks_.size(); ++i)
    {
        marks_.at(i)->setRoadMarkLaneChange(oldLaneChanges_.at(i));
    }

    setUndone();
}

//###########################//
// SetLaneIdCommand //
//###########################//

SetLaneIdCommand::SetLaneIdCommand(Lane *lane, int id, DataCommand *parent)
    : DataCommand(parent)
    , lane_(lane)
    , newId_(id)
{
    construct();
}

void
SetLaneIdCommand::construct()
{
    // Check for validity //
    //
    if (!lane_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane ID: Parameters invalid! No element given."));
        return;
    }

    oldId_ = lane_->getId();

    if (oldId_ == newId_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane ID: Parameters invalid! No change."));
        return;
    }

    setValid();
    setText(QObject::tr("Set Lane ID"));
}

SetLaneIdCommand::~SetLaneIdCommand()
{
}

void
SetLaneIdCommand::redo()
{
    lane_->setId(newId_);

    setRedone();
}

void
SetLaneIdCommand::undo()
{
    lane_->setId(oldId_);

    setUndone();
}

//###########################//
// SetLaneTypeCommand //
//###########################//

SetLaneTypeCommand::SetLaneTypeCommand(Lane *lane, Lane::LaneType laneType, DataCommand *parent)
    : DataCommand(parent)
    , lane_(lane)
    , newType_(laneType)
{
    construct();
}

void
SetLaneTypeCommand::construct()
{
    // Check for validity //
    //
    if (!lane_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Type: Parameters invalid! No element given."));
        return;
    }

    oldType_ = lane_->getLaneType();

    if (oldType_ == newType_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Type: Parameters invalid! No change."));
        return;
    }

    setValid();
    setText(QObject::tr("Set Lane Type"));
}

SetLaneTypeCommand::~SetLaneTypeCommand()
{
}

void
SetLaneTypeCommand::redo()
{
    lane_->setLaneType(newType_);

    setRedone();
}

void
SetLaneTypeCommand::undo()
{
    lane_->setLaneType(oldType_);

    setUndone();
}

//###########################//
// SetLaneLevelCommand //
//###########################//

SetLaneLevelCommand::SetLaneLevelCommand(Lane *lane, bool level, DataCommand *parent)
    : DataCommand(parent)
    , lane_(lane)
    , newLevel_(level)
{
    construct();
}

void
SetLaneLevelCommand::construct()
{
    // Check for validity //
    //
    if (!lane_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Level: Parameters invalid! No element given."));
        return;
    }

    oldLevel_ = lane_->getLevel();

    if (oldLevel_ == newLevel_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Level: Parameters invalid! No change."));
        return;
    }

    setValid();
    setText(QObject::tr("Set Lane Level"));
}

SetLaneLevelCommand::~SetLaneLevelCommand()
{
}

void
SetLaneLevelCommand::redo()
{
    lane_->setLevel(newLevel_);

    setRedone();
}

void
SetLaneLevelCommand::undo()
{
    lane_->setLevel(oldLevel_);

    setUndone();
}

//###########################//
// SetLanePredecessorIdCommand //
//###########################//

SetLanePredecessorIdCommand::SetLanePredecessorIdCommand(Lane *lane, int id, DataCommand *parent)
    : DataCommand(parent)
    , lane_(lane)
    , newPredecessorId_(id)
{
    if (newPredecessorId_ == Lane::NOLANE)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetLanePredecessorIdCommand: Internal error! No lane link."));
        return;
    }
    construct();

}

void
SetLanePredecessorIdCommand::construct()
{
    // Check for validity //
    //
    if (!lane_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Predecessor: Parameters invalid! No element given."));
        return;
    }

    RSystemElementRoad * road = lane_->getParentLaneSection()->getParentRoad();
    if (lane_->getParentLaneSection() == road->getLaneSection(0))
    {
        if (!road->getPredecessor() || (road->getPredecessor()->getElementType() == "junction"))
        {
            if (lane_->getPredecessor() != Lane::NOLANE)
            {
                lane_->setPredecessor(Lane::NOLANE);
            }
            setInvalid();
            setText(QObject::tr("Set Lane Predecessor: Road has no predecessor."));
            return;
        }
    }

    oldPredecessorId_ = lane_->getPredecessor();

    if (oldPredecessorId_ == newPredecessorId_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Predecessor: Parameters invalid! No change."));
        return;
    }

    setValid();
    setText(QObject::tr("Set Lane Predecessor"));
}

SetLanePredecessorIdCommand::~SetLanePredecessorIdCommand()
{
}

void
SetLanePredecessorIdCommand::redo()
{
    lane_->setPredecessor(newPredecessorId_);

    setRedone();
}

void
SetLanePredecessorIdCommand::undo()
{
    lane_->setPredecessor(oldPredecessorId_);

    setUndone();
}

//###########################//
// SetLaneSuccessorIdCommand //
//###########################//

SetLaneSuccessorIdCommand::SetLaneSuccessorIdCommand(Lane *lane, int id, DataCommand *parent)
    : DataCommand(parent)
    , lane_(lane)
    , newSuccessorId_(id)
{
    if (newSuccessorId_ == Lane::NOLANE)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetLaneSuccessorIdCommand: Internal error! No lane link."));
        return;
    }
    construct();
}

void
SetLaneSuccessorIdCommand::construct()
{
    // Check for validity //
    //
    if (!lane_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Successor: Parameters invalid! No element given."));
        return;
    }

    RSystemElementRoad * road = lane_->getParentLaneSection()->getParentRoad();
    if (lane_->getParentLaneSection() == road->getLaneSection(road->getLength()))
    {
        if (!road->getSuccessor() || (road->getSuccessor()->getElementType() == "junction"))
        {
            if (lane_->getSuccessor() != Lane::NOLANE)
            {
                lane_->setSuccessor(Lane::NOLANE);
            }
            setInvalid();
            setText(QObject::tr("Set Lane Successor: Road has no successor."));
            return;
        }
    }

    oldSuccessorId_ = lane_->getSuccessor();

    if (oldSuccessorId_ == newSuccessorId_)
    {
        setInvalid();
        setText(QObject::tr("Set Lane Successor: Parameters invalid! No change."));
        return;
    }

    setValid();
    setText(QObject::tr("Set Lane Successor"));
}

SetLaneSuccessorIdCommand::~SetLaneSuccessorIdCommand()
{
}

void
SetLaneSuccessorIdCommand::redo()
{
    lane_->setSuccessor(newSuccessorId_);

    setRedone();
}

void
SetLaneSuccessorIdCommand::undo()
{
    lane_->setSuccessor(oldSuccessorId_);

    setUndone();
}

//#######################//
// SplitLaneSectionCommand //
//#######################//

SplitLaneSectionCommand::SplitLaneSectionCommand(LaneSection *laneSection, double splitPos, DataCommand *parent)
    : DataCommand(parent)
    , oldSection_(laneSection)
    , newSection_(NULL)
    , splitPos_(splitPos)
{
    // Check for validity //
    //
    if ((fabs(splitPos_ - laneSection->getSStart()) < MIN_LANESECTION_LENGTH)
        || (fabs(laneSection->getSEnd() - splitPos_) < MIN_LANESECTION_LENGTH) // minimum length 1.0 m
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Split LaneSection (invalid!)"));
        return;
    }

    // New section //
    //
    newSection_ = new LaneSection(splitPos, oldSection_->getSide(), laneSection);

    // Done //
    //
    setValid();
    setText(QObject::tr("Split LaneSection"));
}

/*! \brief .
*
*/
SplitLaneSectionCommand::~SplitLaneSectionCommand()
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
SplitLaneSectionCommand::redo()
{
    //  //
    //
    oldSection_->getParentRoad()->addLaneSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
SplitLaneSectionCommand::undo()
{
    //  //
    //
    newSection_->getParentRoad()->delLaneSection(newSection_);

    setUndone();
}

//#######################//
// MergeLaneSectionCommand //
//#######################//

MergeLaneSectionCommand::MergeLaneSectionCommand(LaneSection *laneSectionLow, LaneSection *laneSectionHigh, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(laneSectionLow)
    , oldSectionHigh_(laneSectionHigh)
    , newSection_(NULL)
{
    parentRoad_ = laneSectionLow->getParentRoad();

    // Check for validity //
    //
    if (/*(oldSectionLow_->getDegree() > 1)
		|| (oldSectionHigh_->getDegree() > 1) // only lines allowed
		||*/ (parentRoad_ != laneSectionHigh->getParentRoad()) // not the same parents
        || laneSectionHigh != parentRoad_->getLaneSection(laneSectionLow->getSEnd()) // not consecutive
        )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Merge LaneSection (invalid!)"));
        return;
    }

    // New section //
    //
    //	double deltaLength = laneSectionHigh->getSEnd() - laneSectionLow->getSStart();
    //	double deltaHeight = laneSectionHigh->getLane(laneSectionHigh->getSEnd()) - laneSectionLow->getLane(laneSectionLow->getSStart());
    //	newSection_ = new LaneSection(oldSectionLow_->getSStart(), oldSectionLow_->getA(), deltaHeight/deltaLength, 0.0, 0.0);
    //	if(oldSectionHigh_->isElementSelected() || oldSectionLow_->isElementSelected())
    //	{
    //		newSection_->setElementSelected(true); // keep selection
    //	}

    newSection_ = new LaneSection(laneSectionLow->getSStart(), oldSectionLow_->getSide(), laneSectionLow, laneSectionHigh);

    // Done //
    //
    setValid();
    setText(QObject::tr("Merge LaneSection"));
}

/*! \brief .
*
*/
MergeLaneSectionCommand::~MergeLaneSectionCommand()
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
MergeLaneSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delLaneSection(oldSectionLow_);
    parentRoad_->delLaneSection(oldSectionHigh_);

    parentRoad_->addLaneSection(newSection_);

    setRedone();
}

/*! \brief .
*
*/
void
MergeLaneSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delLaneSection(newSection_);

    parentRoad_->addLaneSection(oldSectionLow_);
    parentRoad_->addLaneSection(oldSectionHigh_);

    setUndone();
}

//#######################//
// RemoveLaneSectionCommand //
//#######################//

RemoveLaneSectionCommand::RemoveLaneSectionCommand(LaneSection *laneSection, DataCommand *parent)
    : DataCommand(parent)
    , oldSectionLow_(NULL)
    , oldSectionMiddle_(laneSection)
    , oldSectionHigh_(NULL)
    , newSectionHigh_(NULL)
{
    parentRoad_ = oldSectionMiddle_->getParentRoad();
    if (!parentRoad_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove LaneSection. No ParentRoad."));
        return;
    }

    oldSectionLow_ = parentRoad_->getLaneSectionBefore(oldSectionMiddle_->getSStart());
    if (!oldSectionLow_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove LaneSection. No section to the left."));
        return;
    }

    oldSectionHigh_ = parentRoad_->getLaneSection(oldSectionMiddle_->getSEnd());
    if (!oldSectionHigh_ || (oldSectionHigh_ == oldSectionMiddle_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot remove LaneSection. No section to the right."));
        return;
    }

    // New section //
    //
    newSectionHigh_ = new LaneSection(oldSectionLow_->getSStart(), oldSectionLow_->getSide(), oldSectionLow_, oldSectionHigh_);
    if (oldSectionHigh_->isElementSelected() || oldSectionMiddle_->isElementSelected())
    {
        newSectionHigh_->setElementSelected(true); // keep selection
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Remove LaneSection"));
}

/*! \brief .
*
*/
RemoveLaneSectionCommand::~RemoveLaneSectionCommand()
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
RemoveLaneSectionCommand::redo()
{
    //  //
    //
    parentRoad_->delLaneSection(oldSectionMiddle_);
    parentRoad_->delLaneSection(oldSectionHigh_);
	parentRoad_->delLaneSection(oldSectionLow_);

    parentRoad_->addLaneSection(newSectionHigh_);

	double sOffset = oldSectionHigh_->getSStart() - oldSectionLow_->getSStart();
	foreach(Lane *lane, newSectionHigh_->getLanes())
	{
		if (!lane->getWidthEntries().isEmpty())
		{
			QMap<double, LaneMoveProperties *> props;
			LaneMoveProperties *laneProps = new LaneMoveProperties();

			laneProps->lowSlot = lane->getWidthEntryBefore(sOffset);
			laneProps->highSlot = lane->getWidthEntry(sOffset);

			props.insert(oldSectionHigh_->getSStart(), laneProps);
			QList<Lane *> lanes;
			QList<QMap<double, WidthPoints*> *> pointList;
			parentRoad_->getLaneWidthsLists(props, false, lanes, pointList);
			parentRoad_->translateLaneWidths(lanes, pointList);
		}
	}

    setRedone();
}

/*! \brief .
*
*/
void
RemoveLaneSectionCommand::undo()
{
    //  //
    //
    parentRoad_->delLaneSection(newSectionHigh_);

	parentRoad_->addLaneSection(oldSectionLow_);
    parentRoad_->addLaneSection(oldSectionMiddle_);
    parentRoad_->addLaneSection(oldSectionHigh_);

    setUndone();
}

//##########################//
// LaneWidthMovePointsCommand //
//##########################//

LaneWidthMovePointsCommand::LaneWidthMovePointsCommand(const QList<LaneWidth *> &endPointWidth, const QList<LaneWidth *> &startPointWidth, const QPointF &deltaPos, DataCommand *parent)
    : DataCommand(parent)
    , endPointWidth_(endPointWidth)
    , startPointWidth_(startPointWidth)
    , widthOnly_(false)
    , deltaPos_(deltaPos)
{
    // Check for validity //
    //
    if (fabs(deltaPos_.manhattanLength()) < NUMERICAL_ZERO8 || (endPointWidth_.isEmpty() && startPointWidth_.isEmpty()))
    {
        setInvalid(); // Invalid because no change.
        //		setText(QObject::tr("Cannot move elevation point. Nothing to be done."));
        setText("");
        return;
    }

    foreach (LaneWidth *section, endPointWidth_)
    {
        //		oldEndHeights_.append(section->getElevation(section->getSEnd())); // save these, so no drifting when doing continuous undo/redo/undo/redo/...
        oldEndPointsBs_.append(section->getB());

        if (fabs(section->getSSectionEnd() - section->getParentLane()->getParentLaneSection()->getLength()) < NUMERICAL_ZERO8) //
        {
            widthOnly_ = true;
        }
    }

    bool tooShort = false;
    foreach (LaneWidth *section, startPointWidth_)
    {
        //		oldStartHeights_.append(section->getElevation(section->getSStart()));
        oldStartPointsAs_.append(section->getA());
        oldStartPointsBs_.append(section->getB());
        oldStartPointsSs_.append(section->getSSectionStart());

        if (fabs(section->getSSectionStart()) < NUMERICAL_ZERO8) // first section of the road
        {
            widthOnly_ = true;
        }
        else if ((section->getLength() - deltaPos_.x() < MIN_LANESECTION_LENGTH) // min length at end

                 )
        {
            tooShort = true;
            widthOnly_ = true;
        }
        //		oldStartSStarts_.append(section->getSStart());
        //		oldStartLengths_.append(section->getLength());
    }

    if (!widthOnly_ && tooShort)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Cannot move elevation point. A section would be too short."));
        return;
    }

    if (widthOnly_)
    {
        deltaPos_.setX(0.0);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Move Width Point"));
}

/*! \brief .
*
*/
LaneWidthMovePointsCommand::~LaneWidthMovePointsCommand()
{
}

/*! \brief .
*
*/
void
LaneWidthMovePointsCommand::redo()
{
    // Set points //
    //
    int i = 0;
    foreach (LaneWidth *section, endPointWidth_)
    {
        double startWidth = section->getWidth(section->getSSectionStart());
        double endWidth = section->getWidth(section->getSSectionEnd() - section->getParentLane()->getParentLaneSection()->getSStart()) + deltaPos_.y();
        if (endWidth < 0)
            endWidth = 0;
		double slope = (endWidth - startWidth) / section->getLength();
        section->setParameters(startWidth, slope, 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (LaneWidth *section, startPointWidth_)
    {
        double startWidth = section->getWidth(section->getSSectionStart());
        startWidth += deltaPos_.y();
        if (startWidth < 0)
            startWidth = 0;

        double endWidth = section->getWidth(section->getSSectionEnd() - section->getParentLane()->getParentLaneSection()->getSStart());
		double slope = (endWidth - startWidth) / section->getLength();
        section->setParameters(startWidth, slope, 0.0, 0.0);
        ++i;
    }

    // Move //
    //
    if (!widthOnly_)
    {
        foreach (LaneWidth *section, startPointWidth_)
        {
            section->getParentLane()->moveWidthEntry(section->getSSectionStart(), section->getSSectionStart() + deltaPos_.x());
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
LaneWidthMovePointsCommand::undo()
{
    // Set points //
    //
    int i = 0;
    foreach (LaneWidth *section, endPointWidth_)
    {
        section->setParameters(section->getA(), oldEndPointsBs_[i], 0.0, 0.0);
        ++i;
    }
    i = 0;
    foreach (LaneWidth *section, startPointWidth_)
    {
        section->setParameters(oldStartPointsAs_[i], oldStartPointsBs_[i], 0.0, 0.0);
        ++i;
    }

    // Move //
    //
    if (!widthOnly_)
    {
        i = 0;
        foreach (LaneWidth *section, startPointWidth_)
        {
            section->getParentLane()->moveWidthEntry(section->getSSectionStart(), oldStartPointsSs_[i]);
            ++i;
        }
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
LaneWidthMovePointsCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const LaneWidthMovePointsCommand *command = static_cast<const LaneWidthMovePointsCommand *>(other);

    if (!command->widthOnly_)
        widthOnly_ = false;

    // Check sections //
    //
    if (endPointWidth_.size() != command->endPointWidth_.size()
        || startPointWidth_.size() != command->startPointWidth_.size())
    {
        return false; // not the same amount of sections
    }

    for (int i = 0; i < endPointWidth_.size(); ++i)
    {
        if (endPointWidth_[i] != command->endPointWidth_[i])
        {
            return false; // different sections
        }
    }
    for (int i = 0; i < startPointWidth_.size(); ++i)
    {
        if (startPointWidth_[i] != command->startPointWidth_[i])
        {
            return false; // different sections
        }
    }

    // Success //
    //
    deltaPos_ += command->deltaPos_; // adjust to new pos, then let the undostack kill the new command

    return true;
}


//################################//
// SelectLaneWidthCommand //
//##############################//

SelectLaneWidthCommand::SelectLaneWidthCommand(const QList<LaneWidth *> &endPointWidths, const QList<LaneWidth *> &startPointWidths, DataCommand *parent)
    : DataCommand(parent)
{
    // Lists //
    //
    endPointWidths_ = endPointWidths;
    startPointWidths_ = startPointWidths;

    // Check for validity //
    //
    if (endPointWidths_.isEmpty() && startPointWidths_.isEmpty())
    {
        setInvalid(); // Invalid because no change.
        setText(QObject::tr("Set Lane width (invalid!)"));
        return;
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Set Lane width"));
}

/*! \brief .
*
*/
SelectLaneWidthCommand::~SelectLaneWidthCommand()
{
}

void
SelectLaneWidthCommand::
    redo()
{
    // Send a notification to the observers
    //
    foreach (LaneWidth *width, endPointWidths_)
    {
        width->addLaneWidthChanges(true);
    }

    foreach (LaneWidth *width, startPointWidths_)
    {
        width->addLaneWidthChanges(true);
    }

    setRedone();
}

//##########################//
// TranslateLaneBorderCommand //
//##########################//

TranslateLaneBorderCommand::TranslateLaneBorderCommand(const QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> &selectedLaneBorders, double width, const QPointF &dPos, DataCommand *parent)
	: DataCommand(parent)
	, selectedLaneBorders_(selectedLaneBorders)
	, dPos_(dPos)
	, width_(width)
{

	// No entries //
	//
	if (selectedLaneBorders_.size() == 0)
	{
		setInvalid();
		setText(QObject::tr("No elements to move."));
		return;
	}

	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>>::ConstIterator it = selectedLaneBorders_.constBegin();
	while (it != selectedLaneBorders_.constEnd())
	{
		RSystemElementRoad *road = it.key();
		QMap<double, LaneMoveProperties *> propsMap = it.value();

		QList<Lane *> lanes;
		QList<QMap<double, WidthPoints*> *> pointList;
		QMap<double, LaneMoveProperties *> newProps;

		newProps = road->getLaneWidthsLists(propsMap, false, lanes, pointList);

		newPropsMap_.insert(road, newProps);
		oldPointList_.insert(road, pointList);
		lanes_.insert(road, lanes);

		it++;
	}

	setValid();
	setText(QObject::tr("Move Lane Width"));
}

/*! \brief .
*
*/
TranslateLaneBorderCommand::~TranslateLaneBorderCommand()
{
}

void
TranslateLaneBorderCommand::undo()
{
	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> selectedLaneBordersChanged;
	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>>::ConstIterator it = selectedLaneBorders_.constBegin();
	while (it != selectedLaneBorders_.constEnd())
	{
		RSystemElementRoad *road = it.key();

		QList<Lane *> lanes = lanes_.value(road);
		QList<QMap<double, WidthPoints*> *> pointList = oldPointList_.value(road);
		road->translateLaneWidths(lanes, pointList);

		selectedLaneBordersChanged.insert(road, newPropsMap_.value(road));
		it++;
	}
	selectedLaneBorders_ = selectedLaneBordersChanged;

	setUndone();
}

void
TranslateLaneBorderCommand::redo()
{
	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> selectedLaneBordersChanged;
	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>>::ConstIterator it = selectedLaneBorders_.constBegin();
	while (it != selectedLaneBorders_.constEnd())
	{
		RSystemElementRoad *road = it.key();
		QMap<double, LaneMoveProperties *> propsMap = it.value();
		QMap<double, LaneMoveProperties *> newPropsMap;

		QList<Lane *> lanes;
		QList<QMap<double, WidthPoints*> *> pointList;

		if (dPos_.isNull())
		{
			newPropsMap = road->getLaneWidthsLists(propsMap, width_, lanes, pointList);
		}
		else
		{
			newPropsMap = road->getLaneWidthsLists(propsMap, dPos_, false, lanes, pointList);
		}

		road->translateLaneWidths(lanes, pointList);

		selectedLaneBordersChanged.insert(road, newPropsMap);
		it++;
	}
	selectedLaneBorders_ = selectedLaneBordersChanged;

	setRedone();
}

bool
TranslateLaneBorderCommand::mergeWith(const QUndoCommand *other)
{
	// Check Ids //
	//
	if (other->id() != id())
	{
		return false;
	}

	const TranslateLaneBorderCommand *command = static_cast<const TranslateLaneBorderCommand *>(other);

	// Check selected tracks
	//
	if (selectedLaneBorders_.size() != command->selectedLaneBorders_.size())
	{
		qDebug() << "Selection not equal";
		return false;
	}

	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>>::const_iterator it = selectedLaneBorders_.begin();

	while (it != selectedLaneBorders_.end())
	{
		RSystemElementRoad *road = it.key();
		QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>>::const_iterator itOther = command->selectedLaneBorders_.constFind(road);
		if (itOther == command->selectedLaneBorders_.constEnd())
		{
			qDebug() << "Selected Borders are not equal";
			return false;
		}

		QMap<double, LaneMoveProperties *> map = it.value();
		QMap<double, LaneMoveProperties *> mapOther = itOther.value();
		QMap<double, LaneMoveProperties *>::const_iterator mapIt = map.constBegin();

		while (mapIt != map.constEnd())
		{
			LaneMoveProperties *props = mapIt.value();
			LaneWidth *low = props->lowSlot;
			LaneWidth *high = props->highSlot;

			QMap<double, LaneMoveProperties *>::const_iterator mapOtherIt = mapOther.constBegin();
			while (mapOtherIt != mapOther.constEnd())
			{
				LaneMoveProperties *otherProps = mapOtherIt.value();
				LaneWidth *lowOther = otherProps->lowSlot;
				LaneWidth *highOther = otherProps->highSlot;

				if ((low == lowOther) && (high == highOther))
				{
					break;
				}

				mapOtherIt++;
			}

			if (mapOtherIt == mapOther.constEnd())
			{
				qDebug() << "Selected Borders are not equal";
				return false;
			}

			mapIt++;
		}

		it++;
	}

	// Success //
	//
	dPos_ += command->dPos_;
	width_ += command->width_;
	selectedLaneBorders_ = command->selectedLaneBorders_;

	return true; 
}

//##########################//
// LaneBorderCornerCommand //
//##########################//

LaneBorderCornerCommand::LaneBorderCornerCommand(LaneWidth *low, LaneWidth *high, bool changeGradient, DataCommand *parent)
	: DataCommand(parent)
	, low_(low)
	, high_(high)
{
	// No entries //
	//
	if (!low || !high)
	{

		setInvalid();
		setText(QObject::tr("No corner defined."));
		return;
	}

	road_ = low_->getParentLane()->getParentLaneSection()->getParentRoad();
	s_ = high_->getSSectionStartAbs();
	setValid();
	setText(QObject::tr("Change Gradient"));
}

/*! \brief .
*
*/
LaneBorderCornerCommand::~LaneBorderCornerCommand()
{
}

void
LaneBorderCornerCommand::undo()
{
	// Make smooth again
	//
	LaneMoveProperties *props = new LaneMoveProperties();
	props->lowSlot = low_;
	props->highSlot = high_;
	QMap<double, LaneMoveProperties *> propsMap;
	propsMap.insert(s_, props);

	QList<Lane *> lanes;
	QList<QMap<double, WidthPoints*> *> pointList;
	road_->getLaneWidthsLists(propsMap, true, lanes, pointList);
	road_->translateLaneWidths(lanes, pointList);

	low_->addLaneWidthChanges(LaneWidth::CLW_GradientChanged);

	setUndone();
}

void
LaneBorderCornerCommand::redo()
{
	// Make a corner
	//
	LaneMoveProperties *props = new LaneMoveProperties();
	props->lowSlot = low_;
	props->highSlot = high_;
	QMap<double, LaneMoveProperties *> propsMap;
	propsMap.insert(s_, props);

	QList<Lane *> lanes;
	QList<QMap<double, WidthPoints*> *> pointList;
	road_->getLaneWidthsLists(propsMap, true, lanes, pointList);
	road_->translateLaneWidths(lanes, pointList);

	low_->addLaneWidthChanges(LaneWidth::CLW_GradientChanged);

	setRedone();
}



