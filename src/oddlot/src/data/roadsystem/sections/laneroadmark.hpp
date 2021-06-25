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

class LaneRoadMark;
class LaneRoadMarkType;


class LaneRoadMarkType
{
	// nested class for line defintition
	//
	class RoadMarkTypeLine
	{
	public:
		enum RoadMarkTypeLineRule
		{
			RMTL_NO_PASSING,
			RMTL_CAUTION,
			RMTL_NONE
		};

		static RoadMarkTypeLineRule parseRoadMarkTypeLineRule(const QString &rule);
		static QString parseRoadMarkTypeLineRuleBack(RoadMarkTypeLineRule rule);

	public:
		explicit RoadMarkTypeLine(LaneRoadMark *parentRoadMark, double length, double space, double tOffset, double sOffset, RoadMarkTypeLineRule rule, double width);
		~RoadMarkTypeLine()
		{
		}

		double getLineLength()
		{
			return length_;
		}
		void setLineLength(double length);

		double getLineSpace()
		{
			return space_;
		}
		void setLineSpace(double space);

		double getLineTOffset()
		{
			return tOffset_;
		}
		void setLineTOffset(double sOffset);

		double getLineSOffset()
		{
			return sOffset_;
		}
		void setLineSOffset(double sOffset);

		RoadMarkTypeLineRule getLineRule()
		{
			return rule_;
		}
		void setLineRule(RoadMarkTypeLineRule rule);

		double getLineWidth()
		{
			return width_;
		}
		void setLineWidth(double width);

	private:
		// Parent road mark //
		//
		LaneRoadMark *parentRoadMark_;

		double length_;
		double space_;
		double tOffset_;
		double sOffset_;
		RoadMarkTypeLineRule rule_;
		double width_;

	};


public:

	explicit LaneRoadMarkType(const QString &name, double width);
	~LaneRoadMarkType();

	void setRoadMarkParent(LaneRoadMark *roadMark)
	{
		parentRoadMark_ = roadMark;
	} 

	QString getLaneRoadMarkTypeName()
	{
		return name_;
	}
	void setLaneRoadMarkTypeName(const QString &name);

	double getLaneRoadMarkTypeWidth()
	{
		return width_;
	}
	void setLaneRoadMarkTypeWidth(double width);

	void addRoadMarkTypeLine(LaneRoadMark *parentRoadMark, double length, double space, double tOffset, double sOffset, const QString &rule, double width);
	void addRoadMarkTypeLine(RoadMarkTypeLine *typeLine);
	bool delRoadMarkTypeLine(RoadMarkTypeLine *typeLine);
	int sizeOfRoadMarkTypeLines()
	{
		return lines_.size();
	}
	bool getRoadMarkTypeLine(int i, double &length, double &space, double &tOffset, double &sOffset, QString &rule, double &width);

private:
	LaneRoadMarkType(); /* not allowed */
	LaneRoadMarkType(const LaneRoadMarkType &); /* not allowed */
	LaneRoadMarkType &operator=(const LaneRoadMarkType &); /* not allowed */

//################//
// PROPERTIES     //
//################//

private:
	LaneRoadMark *parentRoadMark_;

	QString name_;
	double width_;

	QMap<double, RoadMarkTypeLine *> lines_;
};

class LaneRoadMark : public DataElement
{
	friend class LaneRoadMarkType;

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
		CLR_MaterialChanged = 0x100,
		CLR_HeightChanged = 0x200
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
		RMT_CURB,
		RMT_USER
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
    explicit LaneRoadMark(double sOffset, RoadMarkType type, RoadMarkWeight weight = LaneRoadMark::RMW_STANDARD, RoadMarkColor color = LaneRoadMark::RMC_STANDARD, double width = -1.0, RoadMarkLaneChange langeChange = RMLC_BOTH, const QString &material = "standard", double height = 0.0);
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

	QString getRoadMarkMaterial()
	{
		return material_;
	}
	void setRoadMarkMaterial(const QString &material);

	double getRoadMarkHeight()
	{
		return height_;
	}
	void setRoadMarkHeight(double height);

	LaneRoadMarkType *getUserType()
	{
		return userType_;
	}
	void setUserType(LaneRoadMarkType *);
	bool delUserType();

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
	QString material_;
	double height_;
	LaneRoadMarkType *userType_;
};

#endif // LANEROADMARK_HPP
