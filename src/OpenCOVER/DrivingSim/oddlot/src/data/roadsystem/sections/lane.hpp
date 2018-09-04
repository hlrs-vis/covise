/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.02.2010
**
**************************************************************************/

#ifndef LANE_HPP
#define LANE_HPP

#include "src/data/dataelement.hpp"

#include <QString>
#include <QMap>

class LaneWidth;
class LaneRoadMark;
class LaneSpeed;
class LaneHeight;
class LaneRule;
class LaneAccess;

template<typename T, typename U>
struct props
{
	T *highSlot;
	U *lowSlot;
	QPointF dPos;
};

typedef props<LaneWidth, LaneWidth> LaneMoveProperties;

struct WidthPoints
{
	LaneWidth *slot;
	double sStart;
	QPointF pStart;
	QPointF pEnd;
};

class Lane : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneChange
    {
        CLN_IdChanged = 0x1,
        CLN_TypeChanged = 0x2,
        CLN_LevelChanged = 0x4,
        CLN_PredecessorChanged = 0x8,
        CLN_SuccessorChanged = 0x10,
        CLN_ParentLaneSectionChanged = 0x20,
        CLN_WidthsChanged = 0x40,
        CLN_RoadMarksChanged = 0x80,
        CLN_SpeedsChanged = 0x100,
        CLN_HeightsChanged = 0x200,
		CLN_LaneRulesChanged = 0x400,
		CLN_LaneAccessChanged = 0x800,
		CLN_BorderChanged = 0x1000
    };

    // Lane Type //
    //
    enum LaneType
    {
        LT_NONE,
        LT_DRIVING,
        LT_STOP,
        LT_SHOULDER,
        LT_BIKING,
        LT_SIDEWALK,
        LT_BORDER,
        LT_RESTRICTED,
        LT_PARKING,
		LT_BIDIRECTIONAL,
		LT_MEDIAN,
        LT_MWYENTRY,
        LT_MWYEXIT,
        LT_SPECIAL1,
        LT_SPECIAL2,
        LT_SPECIAL3,
		LT_ROADWORKS,
		LT_TRAM,
		LT_RAIL,
		LT_ENTRY,
		LT_EXIT,
		LT_OFFRAMP,
		LT_ONRAMP
    };
    static Lane::LaneType parseLaneType(const QString &type);
    static QString parseLaneTypeBack(Lane::LaneType type);

    // LaneLink //
    //
    static const int NOLANE;

    enum D_LaneLinkType
    {
        DLLT_Predecessor,
        DLLT_Successor
    };


    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Lane(int id, LaneType type, bool level = false, int predecessorId = Lane::NOLANE, int successorId = Lane::NOLANE);
    virtual ~Lane();

    // Lane Functions //
    //
    int getId() const
    {
        return id_;
    }
    void setId(int id);

    Lane::LaneType getLaneType() const
    {
        return type_;
    }
    void setLaneType(Lane::LaneType laneType);

    bool getLevel() const
    {
        return level_;
    }
    void setLevel(bool level);

    int getPredecessor() const
    {
        return predecessorId_;
    }
    void setPredecessor(int id);

    int getSuccessor() const
    {
        return successorId_;
    }
    void setSuccessor(int id);

    // LaneSection Functions //
    //
    LaneSection *getParentLaneSection() const
    {
        return parentLaneSection_;
    }
    void setParentLaneSection(LaneSection *parentLaneSection);

    // Width entries //
    //
    void addWidthEntry(LaneWidth *widthEntry);
    bool delWidthEntry(LaneWidth *widthEntry);
    bool moveWidthEntry(double oldS, double newS);
    LaneWidth *getWidthEntry(double sSection) const;
	LaneWidth *getWidthEntryContains(double sSection) const;
	LaneWidth *getWidthEntryBefore(double sSection) const;
	LaneWidth *getWidthEntryNext(double sSection) const;
	LaneWidth *getLastWidthEntry() const;
    double getWidth(double sSection) const;
    double getSlope(double sSection) const;
    double getWidthEnd(double sSection) const;
    const QMap<double, LaneWidth *> &getWidthEntries() const
    {
        return widths_;
    }

	// Border entries //
	//
	void addBorderEntry(LaneBorder *widthEntry);
	bool delBorderEntry(LaneBorder *widthEntry);
	LaneBorder *getBorderEntry(double sSection) const;
	void delBorderEntries();
	const QMap<double, LaneBorder *> &getBorderEntries() const
	{
		return borders_;
	}
	LaneBorder *getLaneBorderBefore(double s) const;
	LaneBorder *getLaneBorderNext(double s) const;

    // RoadMark entries //
    //
    void addRoadMarkEntry(LaneRoadMark *roadMarkEntry);
    LaneRoadMark *getRoadMarkEntry(double sSection) const;
    double getRoadMarkEnd(double sSection) const;
    const QMap<double, LaneRoadMark *> &getRoadMarkEntries() const
    {
        return marks_;
    }

    // Speed entries //
    //
    void addSpeedEntry(LaneSpeed *speedEntry);
    double getSpeed(double sSection) const;
    double getSpeedEnd(double sSection) const;
    const QMap<double, LaneSpeed *> &getSpeedEntries() const
    {
        return speeds_;
    }

    // Height entries //
    //
    void addHeightEntry(LaneHeight *heightEntry);
    double getHeightEnd(double sSection) const;

	// LaneRule entries //
	//
	void addLaneRuleEntry(LaneRule *roadMarkEntry);
	LaneRule *getLaneRuleEntry(double sSection) const;
	double getLaneRuleEnd(double sSection) const;
	const QMap<double, LaneRule *> &getLaneRuleEntries() const
	{
		return rules_;
	}

	// LaneAccess entries //
	//
	void addLaneAccessEntry(LaneAccess *accessEntry);
	LaneAccess *getLaneAccessEntry(double sSection) const;
	double getLaneAccessEnd(double sSection) const;
	const QMap<double, LaneAccess *> &getLaneAccessEntries() const
	{
		return accesses_;
	}

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLaneChanges() const
    {
        return laneChanges_;
    }

    // Prototype Pattern //
    //
    Lane *getClone() const;
    Lane *getClone(double sOffsetStart, double sOffsetEnd) const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);
    virtual void acceptForWidths(Visitor *visitor);
    virtual void acceptForRoadMarks(Visitor *visitor);
    virtual void acceptForSpeeds(Visitor *visitor);
    virtual void acceptForHeights(Visitor *visitor);
	virtual void acceptForRules(Visitor *visitor);
	virtual void acceptForAccess(Visitor *visitor);

private:
    Lane(); /* not allowed */
    Lane(const Lane &); /* not allowed */
    Lane &operator=(const Lane &); /* not allowed */

public:
    // Observer Pattern //
    //
    void addLaneChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneChanges_;

    // Parent LaneSection //
    //
    LaneSection *parentLaneSection_; // linked

    // Lane Properties //
    //
    int id_;
    Lane::LaneType type_;
    bool level_;

    int predecessorId_;
    int successorId_;

    // Entries //
    //
    QMap<double, LaneWidth *> widths_; // owned
	QMap<double, LaneBorder *> borders_; // owned
    QMap<double, LaneRoadMark *> marks_; // owned
    QMap<double, LaneSpeed *> speeds_; // owned
    QMap<double, LaneHeight *> heights_; // owned
	QMap<double, LaneRule *> rules_; //owned
	QMap<double, LaneAccess *> accesses_; //owned
};

#endif // LANE_HPP
