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

#ifndef LANESECTIONCOMMANDS_HPP
#define LANESECTIONCOMMANDS_HPP

// 2400

#include "datacommand.hpp"

#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"

// Qt //
//
#include <QList>

#define MIN_LANESECTION_LENGTH 1.0

//################//
// InsertLaneCommand //
//################//

class InsertLaneCommand : public DataCommand
{

public:
    explicit InsertLaneCommand(LaneSection *laneSection, Lane *lane, DataCommand *parent = NULL);
    virtual ~InsertLaneCommand();

    virtual int id() const
    {
        return 0x2410;
    }

    virtual void undo();
    virtual void redo();

private:
    InsertLaneCommand(); /* not allowed */
    InsertLaneCommand(const InsertLaneCommand &); /* not allowed */
    InsertLaneCommand &operator=(const InsertLaneCommand &); /* not allowed */

private:
    LaneSection *laneSection_;
    Lane *newLane_;

    int id_;
};

//##########################//
// Remove Lane Command //
//#########################//

class RemoveLaneCommand : public DataCommand
{
public:
    explicit RemoveLaneCommand(LaneSection *laneSection, Lane *lane, DataCommand *parent = NULL);
    virtual ~RemoveLaneCommand();

    virtual int id() const
    {
        return 0x2412;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveLaneCommand(); /* not allowed */
    RemoveLaneCommand(const RemoveLaneCommand &); /* not allowed */
    RemoveLaneCommand &operator=(const RemoveLaneCommand &); /* not allowed */

private:
    LaneSection *laneSection_;
    Lane *oldLane_;

    int id_;
};

//################//
// SetLaneRoadMarkSOffsetCommand //
//################//

class SetLaneRoadMarkSOffsetCommand : public DataCommand
{
public:
    explicit SetLaneRoadMarkSOffsetCommand(LaneRoadMark *mark, double sOffset, DataCommand *parent = NULL);
    virtual ~SetLaneRoadMarkSOffsetCommand();

    virtual int id() const
    {
        return 0x2431;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetLaneRoadMarkSOffsetCommand(); /* not allowed */
    SetLaneRoadMarkSOffsetCommand(const SetLaneRoadMarkSOffsetCommand &); /* not allowed */
    SetLaneRoadMarkSOffsetCommand &operator=(const SetLaneRoadMarkSOffsetCommand &); /* not allowed */

private:
    LaneRoadMark *mark_;

    double newSOffset_;
    double oldSOffset_;
};

//################//
// SetLaneRoadMarkTypeCommand //
//################//

class SetLaneRoadMarkTypeCommand : public DataCommand
{
public:
    explicit SetLaneRoadMarkTypeCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkType type, DataCommand *parent = NULL);
    explicit SetLaneRoadMarkTypeCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkType type, DataCommand *parent = NULL);
    virtual ~SetLaneRoadMarkTypeCommand();

    virtual int id() const
    {
        return 0x2432;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneRoadMarkTypeCommand(); /* not allowed */
    SetLaneRoadMarkTypeCommand(const SetLaneRoadMarkTypeCommand &); /* not allowed */
    SetLaneRoadMarkTypeCommand &operator=(const SetLaneRoadMarkTypeCommand &); /* not allowed */

    void construct();

private:
    QList<LaneRoadMark *> marks_;

    LaneRoadMark::RoadMarkType newType_;
    QList<LaneRoadMark::RoadMarkType> oldTypes_;
};

//################//
// SetLaneRoadMarkWeightCommand //
//################//

class SetLaneRoadMarkWeightCommand : public DataCommand
{
public:
    explicit SetLaneRoadMarkWeightCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkWeight weight, DataCommand *parent = NULL);
    explicit SetLaneRoadMarkWeightCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkWeight weight, DataCommand *parent = NULL);
    virtual ~SetLaneRoadMarkWeightCommand();

    virtual int id() const
    {
        return 0x2433;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneRoadMarkWeightCommand(); /* not allowed */
    SetLaneRoadMarkWeightCommand(const SetLaneRoadMarkWeightCommand &); /* not allowed */
    SetLaneRoadMarkWeightCommand &operator=(const SetLaneRoadMarkWeightCommand &); /* not allowed */

    void construct();

private:
    QList<LaneRoadMark *> marks_;

    LaneRoadMark::RoadMarkWeight newWeight_;
    QList<LaneRoadMark::RoadMarkWeight> oldWeights_;
};

//################//
// SetLaneRoadMarkColorCommand //
//################//

class SetLaneRoadMarkColorCommand : public DataCommand
{
public:
    explicit SetLaneRoadMarkColorCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkColor color, DataCommand *parent = NULL);
    explicit SetLaneRoadMarkColorCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkColor color, DataCommand *parent = NULL);
    virtual ~SetLaneRoadMarkColorCommand();

    virtual int id() const
    {
        return 0x2434;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneRoadMarkColorCommand(); /* not allowed */
    SetLaneRoadMarkColorCommand(const SetLaneRoadMarkColorCommand &); /* not allowed */
    SetLaneRoadMarkColorCommand &operator=(const SetLaneRoadMarkColorCommand &); /* not allowed */

    void construct();

private:
    QList<LaneRoadMark *> marks_;

    LaneRoadMark::RoadMarkColor newColor_;
    QList<LaneRoadMark::RoadMarkColor> oldColors_;
};

//################//
// SetLaneRoadMarkWidthCommand //
//################//

class SetLaneRoadMarkWidthCommand : public DataCommand
{
public:
    explicit SetLaneRoadMarkWidthCommand(QList<LaneRoadMark *> marks, double width, DataCommand *parent = NULL);
    explicit SetLaneRoadMarkWidthCommand(LaneRoadMark *mark, double width, DataCommand *parent = NULL);
    virtual ~SetLaneRoadMarkWidthCommand();

    virtual int id() const
    {
        return 0x2435;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneRoadMarkWidthCommand(); /* not allowed */
    SetLaneRoadMarkWidthCommand(const SetLaneRoadMarkWidthCommand &); /* not allowed */
    SetLaneRoadMarkWidthCommand &operator=(const SetLaneRoadMarkWidthCommand &); /* not allowed */

    void construct();

private:
    QList<LaneRoadMark *> marks_;

    double newWidth_;
    QList<double> oldWidths_;
};

//################//
// SetLaneRoadMarkLaneChangeCommand //
//################//

class SetLaneRoadMarkLaneChangeCommand : public DataCommand
{
public:
    explicit SetLaneRoadMarkLaneChangeCommand(QList<LaneRoadMark *> marks, LaneRoadMark::RoadMarkLaneChange permission, DataCommand *parent = NULL);
    explicit SetLaneRoadMarkLaneChangeCommand(LaneRoadMark *mark, LaneRoadMark::RoadMarkLaneChange permission, DataCommand *parent = NULL);
    virtual ~SetLaneRoadMarkLaneChangeCommand();

    virtual int id() const
    {
        return 0x2436;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneRoadMarkLaneChangeCommand(); /* not allowed */
    SetLaneRoadMarkLaneChangeCommand(const SetLaneRoadMarkLaneChangeCommand &); /* not allowed */
    SetLaneRoadMarkLaneChangeCommand &operator=(const SetLaneRoadMarkLaneChangeCommand &); /* not allowed */

    void construct();

private:
    QList<LaneRoadMark *> marks_;

    LaneRoadMark::RoadMarkLaneChange newLaneChange_;
    QList<LaneRoadMark::RoadMarkLaneChange> oldLaneChanges_;
};

//################//
// SetLaneIdCommand //
//################//

class SetLaneIdCommand : public DataCommand
{
public:
    explicit SetLaneIdCommand(Lane *lane, int id, DataCommand *parent = NULL);
    virtual ~SetLaneIdCommand();

    virtual int id() const
    {
        return 0x2450;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneIdCommand(); /* not allowed */
    SetLaneIdCommand(const SetLaneIdCommand &); /* not allowed */
    SetLaneIdCommand &operator=(const SetLaneIdCommand &); /* not allowed */

    void construct();

private:
    Lane *lane_;

    int oldId_;
    int newId_;
};

//################//
// SetLaneTypeCommand //
//################//

class SetLaneTypeCommand : public DataCommand
{
public:
    explicit SetLaneTypeCommand(Lane *lane, Lane::LaneType laneType, DataCommand *parent = NULL);
    virtual ~SetLaneTypeCommand();

    virtual int id() const
    {
        return 0x2451;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneTypeCommand(); /* not allowed */
    SetLaneTypeCommand(const SetLaneTypeCommand &); /* not allowed */
    SetLaneTypeCommand &operator=(const SetLaneTypeCommand &); /* not allowed */

    void construct();

private:
    Lane *lane_;

    Lane::LaneType oldType_;
    Lane::LaneType newType_;
};

//################//
// SetLaneLevelCommand //
//################//

class SetLaneLevelCommand : public DataCommand
{
public:
    explicit SetLaneLevelCommand(Lane *lane, bool level, DataCommand *parent = NULL);
    virtual ~SetLaneLevelCommand();

    virtual int id() const
    {
        return 0x2452;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneLevelCommand(); /* not allowed */
    SetLaneLevelCommand(const SetLaneLevelCommand &); /* not allowed */
    SetLaneLevelCommand &operator=(const SetLaneLevelCommand &); /* not allowed */

    void construct();

private:
    Lane *lane_;

    bool oldLevel_;
    bool newLevel_;
};

//################//
// SetLanePredecessorIdCommand //
//################//

class SetLanePredecessorIdCommand : public DataCommand
{
public:
    explicit SetLanePredecessorIdCommand(Lane *lane, int id, DataCommand *parent = NULL);
    virtual ~SetLanePredecessorIdCommand();

    virtual int id() const
    {
        return 0x2453;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLanePredecessorIdCommand(); /* not allowed */
    SetLanePredecessorIdCommand(const SetLanePredecessorIdCommand &); /* not allowed */
    SetLanePredecessorIdCommand &operator=(const SetLanePredecessorIdCommand &); /* not allowed */

    void construct();

private:
    Lane *lane_;

    int oldPredecessorId_;
    int newPredecessorId_;
};

//################//
// SetLaneSuccessorIdCommand //
//################//

class SetLaneSuccessorIdCommand : public DataCommand
{
public:
    explicit SetLaneSuccessorIdCommand(Lane *lane, int id, DataCommand *parent = NULL);
    virtual ~SetLaneSuccessorIdCommand();

    virtual int id() const
    {
        return 0x2454;
    }

    virtual void undo();
    virtual void redo();

private:
    SetLaneSuccessorIdCommand(); /* not allowed */
    SetLaneSuccessorIdCommand(const SetLaneSuccessorIdCommand &); /* not allowed */
    SetLaneSuccessorIdCommand &operator=(const SetLaneSuccessorIdCommand &); /* not allowed */

    void construct();

private:
    Lane *lane_;

    int oldSuccessorId_;
    int newSuccessorId_;
};

//################//
// Split          //
//################//

class SplitLaneSectionCommand : public DataCommand
{
public:
    explicit SplitLaneSectionCommand(LaneSection *laneSection, double splitPos, DataCommand *parent = NULL);
    virtual ~SplitLaneSectionCommand();

    virtual int id() const
    {
        return 0x801;
    }

    virtual void undo();
    virtual void redo();

private:
    SplitLaneSectionCommand(); /* not allowed */
    SplitLaneSectionCommand(const SplitLaneSectionCommand &); /* not allowed */
    SplitLaneSectionCommand &operator=(const SplitLaneSectionCommand &); /* not allowed */

private:
    LaneSection *oldSection_;
    LaneSection *newSection_;

    double splitPos_;
};

//################//
// Merge          //
//################//

class MergeLaneSectionCommand : public DataCommand
{
public:
    explicit MergeLaneSectionCommand(LaneSection *laneSectionLow, LaneSection *laneSectionHigh, DataCommand *parent = NULL);
    virtual ~MergeLaneSectionCommand();

    virtual int id() const
    {
        return 0x802;
    }

    virtual void undo();
    virtual void redo();

private:
    MergeLaneSectionCommand(); /* not allowed */
    MergeLaneSectionCommand(const MergeLaneSectionCommand &); /* not allowed */
    MergeLaneSectionCommand &operator=(const MergeLaneSectionCommand &); /* not allowed */

private:
    LaneSection *oldSectionLow_;
    LaneSection *oldSectionHigh_;
    LaneSection *newSection_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Remove          //
//################//

class RemoveLaneSectionCommand : public DataCommand
{
public:
    explicit RemoveLaneSectionCommand(LaneSection *laneSection, DataCommand *parent = NULL);
    virtual ~RemoveLaneSectionCommand();

    virtual int id() const
    {
        return 0x804;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveLaneSectionCommand(); /* not allowed */
    RemoveLaneSectionCommand(const RemoveLaneSectionCommand &); /* not allowed */
    RemoveLaneSectionCommand &operator=(const RemoveLaneSectionCommand &); /* not allowed */

private:
    LaneSection *oldSectionLow_;
    LaneSection *oldSectionMiddle_;
    LaneSection *oldSectionHigh_;

    LaneSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Move           //
//################//

class LaneWidthMovePointsCommand : public DataCommand
{
public:
    explicit LaneWidthMovePointsCommand(const QList<LaneWidth *> &endPointWidth, const QList<LaneWidth *> &startPointWidth, const QPointF &deltaPos, DataCommand *parent = NULL);
    virtual ~LaneWidthMovePointsCommand();

    virtual int id() const
    {
        return 0x820;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    LaneWidthMovePointsCommand(); /* not allowed */
    LaneWidthMovePointsCommand(const LaneWidthMovePointsCommand &); /* not allowed */
    LaneWidthMovePointsCommand &operator=(const LaneWidthMovePointsCommand &); /* not allowed */

private:
    QList<LaneWidth *> endPointWidth_;
    QList<LaneWidth *> startPointWidth_;

    bool widthOnly_;

    QList<double> oldStartPointsSs_;
    QList<double> oldStartPointsAs_;
    QList<double> oldStartPointsBs_;

    QList<double> oldEndPointsBs_;

    QPointF deltaPos_;
};

//################//
// LaneSetWidth           //
//################//

class LaneSetWidthCommand : public DataCommand
{
public:
    explicit LaneSetWidthCommand(const QList<LaneWidth *> &endPointWidth, const QList<LaneWidth *> &startPointWidth, float width, bool absolute = true, DataCommand *parent = NULL);
    virtual ~LaneSetWidthCommand();

    virtual int id() const
    {
        return 0x830;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    LaneSetWidthCommand(); /* not allowed */
    LaneSetWidthCommand(const LaneSetWidthCommand &); /* not allowed */
    LaneSetWidthCommand &operator=(const LaneSetWidthCommand &); /* not allowed */

private:
    QList<LaneWidth *> endPointWidth_;
    QList<LaneWidth *> startPointWidth_;

    QList<double> oldStartPointsAs_;
    QList<double> oldStartPointsBs_;
    QList<double> oldStartPointsCs_;
    QList<double> oldStartPointsDs_;

    QList<double> oldEndPointsAs_;
    QList<double> oldEndPointsBs_;
    QList<double> oldEndPointsCs_;
    QList<double> oldEndPointsDs_;

    float newWidth;
    bool absoluteWidth;
};

//################//
// InsertLaneWidthCommand //
//################//

class InsertLaneWidthCommand : public DataCommand
{
public:
    explicit InsertLaneWidthCommand(Lane *lane, LaneWidth *laneWidth, DataCommand *parent = NULL);
    virtual ~InsertLaneWidthCommand();

    virtual int id() const
    {
        return 0x2410;
    }

    virtual void undo();
    virtual void redo();

private:
    InsertLaneWidthCommand(); /* not allowed */
    InsertLaneWidthCommand(const InsertLaneWidthCommand &); /* not allowed */
    InsertLaneWidthCommand &operator=(const InsertLaneWidthCommand &); /* not allowed */

private:
    Lane *lane_;
    LaneWidth *newLaneWidth_;

    int id_;
};

//##################################//
// Merge LaneWidth         //
//################################//

template<typename T>
class MergeLaneWidthSectionCommand : public DataCommand
{
public:
	explicit MergeLaneWidthSectionCommand(T *low, T *high, DataCommand *parent)
		: DataCommand(parent)
		, oldLow_(low)
		, oldHigh_(high)
		, newEntry_(NULL)
	{
		parentLane_ = low->getParentLane();

		// Check for validity //
		//
		if ((!low || !high) && (parentLane_ != high->getParentLane()) // not the same parents
			|| high != parentLane_->getWidthEntryNext(low->getSSectionStart()) // not consecutive
			)
		{
			setInvalid(); // Invalid
			setText(QObject::tr("Merge LaneWidthSection (invalid!)"));
			return;
		}

		newEntry_ = new T(low->getSSectionStart(), low->getA(), low->getB(), low->getC(), low->getD());

		LaneMoveProperties *props = new LaneMoveProperties();
		props->lowSlot = low;
		props->highSlot = high;
		QMap<double, LaneMoveProperties *> propsMap;
		propsMap.insert(low->getSSectionStartAbs(), props);

		LaneSection *parentLaneSection = parentLane_->getParentLaneSection();
		road_ = parentLaneSection->getParentRoad();
		road_->getLaneWidthsLists(propsMap, false, lanes_, pointList_);

		// Remember index and points of old highSlot start
		//
		indexHigh_ = indexLow_ = -1;
		int i = 0;
		while ((i < pointList_.size()) && ((indexLow_ < 0) || (indexHigh_ < 0)))
		{
			QMap<double, WidthPoints*> *points = pointList_.at(i);
			QMap<double, WidthPoints*>::const_iterator it = points->find(high->getSSectionStartAbs());
			if (it != points->end())
			{
				oldPoints_ = it.value();
				indexHigh_ = i;
			}

			it = points->find(low->getSSectionStartAbs());
			if (it != points->end())
			{
				indexLow_ = i;
			}

			i++;
		}

		// Done //
		//
		setValid();
		setText(QObject::tr("Merge LaneWidthSection"));
	}

	/*! \brief .
	*
	*/
	virtual ~MergeLaneWidthSectionCommand()
	{
		// Clean up //
		//
		if (isUndone())
		{
			delete newEntry_;
		}
		else
		{
			delete oldLow_;
			delete oldHigh_;
		}
	}

	virtual int id() const
	{
		return 0x112;
	}

	/*! \brief .
	*
	*/
	virtual void redo()
	{

		WidthPoints *wp = pointList_.at(indexLow_)->value(oldLow_->getSSectionStartAbs(), NULL);
		wp->slot = newEntry_;
		wp->pEnd = oldPoints_->pEnd;
		pointList_.at(indexHigh_)->remove(oldHigh_->getSSectionStartAbs());

		parentLane_->delWidthEntry(oldLow_);
		parentLane_->delWidthEntry(oldHigh_);

		parentLane_->addWidthEntry(newEntry_);

		road_->translateLaneWidths(lanes_, pointList_);

		parentLane_->getParentLaneSection()->addLaneSectionChanges(LaneSection::CLS_LanesWidthsChanged);

		setRedone();
	}

	/*! \brief .
	*
	*/
	virtual void undo()
	{
		//  //
		//
		parentLane_->delWidthEntry(newEntry_);

		parentLane_->addWidthEntry(oldLow_);
		parentLane_->addWidthEntry(oldHigh_);

		WidthPoints *wp = pointList_.at(indexLow_)->value(oldLow_->getSSectionStartAbs(), NULL);
		wp->slot = oldLow_;
		wp->pEnd = oldPoints_->pStart;
		pointList_.at(indexHigh_)->insert(oldHigh_->getSSectionStartAbs(), oldPoints_);

		road_->translateLaneWidths(lanes_, pointList_);

		parentLane_->getParentLaneSection()->addLaneSectionChanges(LaneSection::CLS_LanesWidthsChanged);

		setUndone();
	}

private:
	Lane *parentLane_;

	T *oldLow_, *oldHigh_, *newEntry_;
	RSystemElementRoad *road_;
	QList<Lane *> lanes_;
	QList<QMap<double, WidthPoints*> *> pointList_;
	
	int indexLow_, indexHigh_;
	WidthPoints* oldPoints_;

};

//##################################//
// Select LaneWidth         //
//################################//

class SelectLaneWidthCommand : public DataCommand
{
public:
    explicit SelectLaneWidthCommand(const QList<LaneWidth *> &endPointWidths, const QList<LaneWidth *> &startPointWidths, DataCommand *parent = NULL);
    virtual ~SelectLaneWidthCommand();

    virtual int id() const
    {
        return 0x810;
    }

    virtual void undo(){};
    virtual void redo();

private:
    SelectLaneWidthCommand(); /* not allowed */
    SelectLaneWidthCommand(const SelectLaneWidthCommand &); /* not allowed */
    SelectLaneWidthCommand &operator=(const SelectLaneWidthCommand &); /* not allowed */

private:
    QList<LaneWidth *> endPointWidths_;
    QList<LaneWidth *> startPointWidths_;
};

//##########################//
// LaneBorderCornerCommand //
//##########################//

template<typename T>
class LaneBorderCornerCommand : public DataCommand
{

public:
	explicit LaneBorderCornerCommand(T *low, T *high, bool changeGradient, DataCommand *parent = NULL)
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
	virtual ~LaneBorderCornerCommand()
	{
	}


	virtual int id() const
	{
		return 0x110;
	}

	virtual void undo()
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

	virtual void redo()
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

private:
	T *low_, *high_;
	RSystemElementRoad *road_;
	double s_;
};

//##########################//
// TranslateLaneBorderCommand //
//##########################//

template<typename T>
class TranslateLaneBorderCommand : public DataCommand
{

public:
	explicit TranslateLaneBorderCommand(const QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> &selectedLaneBorders, double width, const QPointF &dPos,DataCommand *parent = NULL)
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
	virtual ~TranslateLaneBorderCommand()
	{
	}


	virtual int id() const
	{
		return 0x109;
	}

	virtual void undo()
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

	virtual void redo()
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

	virtual bool mergeWith(const QUndoCommand *other)
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

private:
	TranslateLaneBorderCommand(); /* not allowed */
	TranslateLaneBorderCommand(const TranslateLaneBorderCommand &); /* not allowed */


private:
	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> selectedLaneBorders_;
	QPointF dPos_;
	double width_;

	QMap<RSystemElementRoad *, QList<QMap<double, WidthPoints*> *>> oldPointList_;
	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> newPropsMap_;
	QMap<RSystemElementRoad *, QList<Lane *>> lanes_;
};

#endif // LANESECTIONCOMMANDS_HPP
