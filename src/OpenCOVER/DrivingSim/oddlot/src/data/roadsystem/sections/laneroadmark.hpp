/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.02.2010
**
**************************************************************************/

#ifndef LANEROADMARK_HPP
#define LANEROADMARK_HPP

#include "src/data/dataelement.hpp"

#include <QString>

class LaneRoadMark : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneRoadMarkChange
    {
        CLR_ParentLaneChanged = 0x1,
        CLR_OffsetChanged = 0x2,
        CLR_TypeChanged = 0x4,
        CLR_WeightChanged = 0x8,
        CLR_ColorChanged = 0x10,
        CLR_WidthChanged = 0x20,
        CLR_LaneChangeChanged = 0x40,
    };

    // RoadMark Parameters //
    //
    enum RoadMarkType
    {
        RMT_NONE,
        RMT_SOLID,
        RMT_BROKEN,
        RMT_SOLID_SOLID,
        RMT_SOLID_BROKEN,
        RMT_BROKEN_SOLID,
		RMT_BROKEN_BROKEN,
		RMT_BOTTS_DOTS,
		RMT_GRASS,
		RMT_CURB
    };
    static LaneRoadMark::RoadMarkType parseRoadMarkType(const QString &type);
    static QString parseRoadMarkTypeBack(LaneRoadMark::RoadMarkType type);

    enum RoadMarkWeight
    {
        RMW_STANDARD,
        RMW_BOLD
    };
    static LaneRoadMark::RoadMarkWeight parseRoadMarkWeight(const QString &type);
    static QString parseRoadMarkWeightBack(LaneRoadMark::RoadMarkWeight type);

	enum RoadMarkColor
	{
		RMC_STANDARD,
		RMC_YELLOW,
		RMC_BLUE,
		RMC_GREEN,
		RMC_RED,
		RMC_WHITE
    };
    static LaneRoadMark::RoadMarkColor parseRoadMarkColor(const QString &type);
    static QString parseRoadMarkColorBack(LaneRoadMark::RoadMarkColor type);

    enum RoadMarkLaneChange
    {
        RMLC_INCREASE,
        RMLC_DECREASE,
        RMLC_BOTH,
        RMLC_NONE
    };
    static LaneRoadMark::RoadMarkLaneChange parseRoadMarkLaneChange(const QString &type);
    static QString parseRoadMarkLaneChangeBack(LaneRoadMark::RoadMarkLaneChange type);

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneRoadMark(double sOffset, RoadMarkType type, RoadMarkWeight weight = LaneRoadMark::RMW_STANDARD, RoadMarkColor color = LaneRoadMark::RMC_STANDARD, double width = -1.0, RoadMarkLaneChange langeChange = RMLC_BOTH);
    virtual ~LaneRoadMark();

    // Lane Functions //
    //
    Lane *getParentLane() const
    {
        return parentLane_;
    }
    void setParentLane(Lane *parentLane);

    double getSSectionStart() const
    {
        return sOffset_;
    }
    double getSSectionEnd() const;
    double getLength() const;

    // RoadMark Parameters //
    //
    double getSOffset() const
    {
        return sOffset_;
    }
    void setSOffset(double sOffset);

    RoadMarkType getRoadMarkType() const
    {
        return type_;
    }
    void setRoadMarkType(RoadMarkType type);

    RoadMarkWeight getRoadMarkWeight() const
    {
        return weight_;
    }
    void setRoadMarkWeight(RoadMarkWeight weight);

    RoadMarkColor getRoadMarkColor() const
    {
        return color_;
    }
    void setRoadMarkColor(RoadMarkColor color);

    double getRoadMarkWidth() const
    {
        return width_;
    } // -1.0: none
    void setRoadMarkWidth(double width);

    RoadMarkLaneChange getRoadMarkLaneChange() const
    {
        return laneChange_;
    }
    void setRoadMarkLaneChange(RoadMarkLaneChange permission);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getRoadMarkChanges() const
    {
        return roadMarkChanges_;
    }

    // Prototype Pattern //
    //
    LaneRoadMark *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneRoadMark(); /* not allowed */
    LaneRoadMark(const LaneRoadMark &); /* not allowed */
    LaneRoadMark &operator=(const LaneRoadMark &); /* not allowed */

    // Observer Pattern //
    //
    void addRoadMarkChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int roadMarkChanges_;

    // Lane Properties //
    //
    Lane *parentLane_; // linked

    // RoadMark Properties //
    //
    double sOffset_;
    RoadMarkType type_;
    RoadMarkWeight weight_;
    RoadMarkColor color_;
    double width_;
    RoadMarkLaneChange laneChange_;
};

#endif // LANEROADMARK_HPP
