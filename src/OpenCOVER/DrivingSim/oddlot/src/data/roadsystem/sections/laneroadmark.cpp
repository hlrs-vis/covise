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

#include "laneroadmark.hpp"

#include "lane.hpp"

LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLineRule
LaneRoadMarkType::RoadMarkTypeLine::parseRoadMarkTypeLineRule(const QString &rule)
{
	if (rule == "none")
	{
		return LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLineRule::RMTL_NONE;
	}
	else if (rule == "no passing")
	{
		return LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLineRule::RMTL_NO_PASSING;
	}
	else if (rule == "caution")
	{
		return LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLineRule::RMTL_CAUTION;
	}
	else
	{
		qDebug("WARNING: unknown road mark type line rule: %s", rule.toUtf8().constData());
		return LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLineRule::RMTL_NONE;
	}
}

QString
LaneRoadMarkType::RoadMarkTypeLine::parseRoadMarkTypeLineRuleBack(LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLineRule rule)
{
	if (rule == RMTL_NONE)
	{
		return QString("none");
	}
	else if (rule == RMTL_NO_PASSING)
	{
		return QString("no passing");
	}
	else if (rule == RMTL_CAUTION)
	{
		return QString("caution");
	}
	else
	{
		qDebug("WARNING: unknown road mark type.");
		return QString("none");
	}
}

//#################//
// Road Mark Type Line //
//################//
LaneRoadMarkType::RoadMarkTypeLine::RoadMarkTypeLine(LaneRoadMark *laneRoadMark, double length, double space, double tOffset, double sOffset, RoadMarkTypeLineRule rule, double width)
	: length_(length)
	, space_(space)
	, tOffset_(tOffset)
	, sOffset_(sOffset)
	, rule_(rule)
	, width_(width)
	, parentRoadMark_(laneRoadMark)
{

}

void 
LaneRoadMarkType::RoadMarkTypeLine::setLineLength(double length)
{
	length_ = length;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void 
LaneRoadMarkType::RoadMarkTypeLine::setLineSpace(double space)
{
	space_ = space;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void
LaneRoadMarkType::RoadMarkTypeLine::setLineTOffset(double tOffset)
{
	tOffset_ = tOffset;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void 
LaneRoadMarkType::RoadMarkTypeLine::setLineSOffset(double sOffset)
{
	LaneRoadMarkType *parentRoadMarkType = parentRoadMark_->getUserType();
	parentRoadMarkType->delRoadMarkTypeLine(this);
	sOffset_ = sOffset;
	parentRoadMarkType->addRoadMarkTypeLine(this);
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void 
LaneRoadMarkType::RoadMarkTypeLine::setLineRule(RoadMarkTypeLineRule rule)
{
	rule_ = rule;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void 
LaneRoadMarkType::RoadMarkTypeLine::setLineWidth(double width)
{
	width_ = width;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

//#################//
// Road Mark Type //
//################//

LaneRoadMarkType::LaneRoadMarkType(const QString &name, double width)
	: name_(name)
	, width_(width)
{
}

LaneRoadMarkType::~LaneRoadMarkType()
{
	foreach(RoadMarkTypeLine *typeLine, lines_)
		delete typeLine;

}

void
LaneRoadMarkType::setLaneRoadMarkTypeName(const QString &name)
{
	name_ = name;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void
LaneRoadMarkType::setLaneRoadMarkTypeWidth(double width)
{
	width_ = width;
	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}


void
LaneRoadMarkType::addRoadMarkTypeLine(LaneRoadMark *parentRoadMark, double length, double space, double tOffset, double sOffset, const QString &rule, double width)
{	
	RoadMarkTypeLine *typeLine = new RoadMarkTypeLine(parentRoadMark, length, space, tOffset, sOffset, RoadMarkTypeLine::parseRoadMarkTypeLineRule(rule), width);
	lines_.insert(sOffset, typeLine);

	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

void
LaneRoadMarkType::addRoadMarkTypeLine(RoadMarkTypeLine *typeLine)
{
	double key = typeLine->getLineSOffset();
	lines_.insert(key, typeLine);

	parentRoadMark_->addRoadMarkChanges(LaneRoadMark::RMT_USER);
}

bool 
LaneRoadMarkType::delRoadMarkTypeLine(RoadMarkTypeLine *typeLine)
{
	return lines_.remove(typeLine->getLineSOffset());
}

bool 
LaneRoadMarkType::getRoadMarkTypeLine(int i, double &length, double &space, double &tOffset, double &sOffset, QString &rule, double &width)
{
	if (i >= lines_.size())
	{
		return false;
	}
	
	QMap<double, LaneRoadMarkType::RoadMarkTypeLine *>::const_iterator it = lines_.constBegin() + i;
	RoadMarkTypeLine * line = it.value();
	length = line->getLineLength();
	space = line->getLineSpace();
	tOffset = line->getLineTOffset();
	sOffset = line->getLineSOffset();
	rule = RoadMarkTypeLine::parseRoadMarkTypeLineRuleBack(line->getLineRule());
	width = line->getLineWidth();

	return true;
}

//###################//
// Static Functions  //
//###################//

LaneRoadMark::RoadMarkType
LaneRoadMark::parseRoadMarkType(const QString &type)
{
    if (type == "none")
        return LaneRoadMark::RMT_NONE;
    else if (type == "solid")
        return LaneRoadMark::RMT_SOLID;
    else if (type == "broken")
        return LaneRoadMark::RMT_BROKEN;
    else if (type == "solid solid")
        return LaneRoadMark::RMT_SOLID_SOLID;
    else if (type == "solid broken")
        return LaneRoadMark::RMT_SOLID_BROKEN;
    else if (type == "broken solid")
        return LaneRoadMark::RMT_BROKEN_SOLID;
	else if (type == "broken broken")
		return LaneRoadMark::RMT_BROKEN_BROKEN;
	else if (type == "botts dots")
		return LaneRoadMark::RMT_BOTTS_DOTS;
	else if (type == "grass")
		return LaneRoadMark::RMT_GRASS;
	else if (type == "curb")
		return LaneRoadMark::RMT_CURB;
    else
    {
        qDebug("WARNING: unknown road mark type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMT_NONE;
    }
};

QString
LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RoadMarkType type)
{
    if (type == LaneRoadMark::RMT_NONE)
        return QString("none");
    else if (type == LaneRoadMark::RMT_SOLID)
        return QString("solid");
    else if (type == LaneRoadMark::RMT_BROKEN)
        return QString("broken");
    else if (type == LaneRoadMark::RMT_SOLID_SOLID)
        return QString("solid solid");
    else if (type == LaneRoadMark::RMT_SOLID_BROKEN)
        return QString("solid broken");
    else if (type == LaneRoadMark::RMT_BROKEN_SOLID)
        return QString("broken broken");
	else if (type == LaneRoadMark::RMT_BROKEN_BROKEN)
		return QString("botts dots");
	else if (type == LaneRoadMark::RMT_BOTTS_DOTS)
		return QString("broken solid");
	else if (type == LaneRoadMark::RMT_GRASS)
		return QString("grass");
	else if (type == LaneRoadMark::RMT_CURB)
		return QString("curb");
    else
    {
        qDebug("WARNING: unknown road mark type.");
        return QString("none");
    }
};

LaneRoadMark::RoadMarkWeight
LaneRoadMark::parseRoadMarkWeight(const QString &type)
{
    if (type == "standard")
        return LaneRoadMark::RMW_STANDARD;
    else if (type == "bold")
        return LaneRoadMark::RMW_BOLD;
    else
    {
        qDebug("WARNING: unknown road mark weight type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMW_STANDARD;
    }
};

QString
LaneRoadMark::parseRoadMarkWeightBack(LaneRoadMark::RoadMarkWeight type)
{
    if (type == LaneRoadMark::RMW_STANDARD)
        return QString("standard");
    else if (type == LaneRoadMark::RMW_BOLD)
        return QString("bold");
    else
    {
        qDebug("WARNING: unknown road mark weight type.");
        return "standard";
    }
};

LaneRoadMark::RoadMarkColor
LaneRoadMark::parseRoadMarkColor(const QString &type)
{
    if (type == "standard")
        return LaneRoadMark::RMC_STANDARD;
    else if (type == "yellow")
        return LaneRoadMark::RMC_YELLOW;
	else if (type == "blue")
		return LaneRoadMark::RMC_BLUE;
	else if (type == "green")
		return LaneRoadMark::RMC_GREEN;
	else if (type == "red")
		return LaneRoadMark::RMC_RED;
	else if (type == "white")
		return LaneRoadMark::RMC_WHITE;
    else
    {
        qDebug("WARNING: unknown road mark color type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMC_STANDARD;
    }
};

QString
LaneRoadMark::parseRoadMarkColorBack(LaneRoadMark::RoadMarkColor type)
{
    if (type == LaneRoadMark::RMC_STANDARD)
        return "standard";
    else if (type == LaneRoadMark::RMC_YELLOW)
        return "yellow";
	else if (type == LaneRoadMark::RMC_BLUE)
		return "blue";
	else if (type == LaneRoadMark::RMC_GREEN)
		return "green";
	else if (type == LaneRoadMark::RMC_RED)
		return "red";
	else if (type == LaneRoadMark::RMC_WHITE)
		return "white";
    else
    {
        qDebug("WARNING: unknown road mark color type");
        return "standard";
    }
};

LaneRoadMark::RoadMarkLaneChange
LaneRoadMark::parseRoadMarkLaneChange(const QString &type)
{
    if (type == "increase")
        return LaneRoadMark::RMLC_INCREASE;
    else if (type == "decrease")
        return LaneRoadMark::RMLC_DECREASE;
    else if (type == "both")
        return LaneRoadMark::RMLC_BOTH;
    else if (type == "none")
        return LaneRoadMark::RMLC_NONE;
    else
    {
        qDebug("WARNING: unknown road mark lane change type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMLC_BOTH;
    }
}

QString
LaneRoadMark::parseRoadMarkLaneChangeBack(LaneRoadMark::RoadMarkLaneChange type)
{
    if (type == LaneRoadMark::RMLC_INCREASE)
        return "increase";
    else if (type == LaneRoadMark::RMLC_DECREASE)
        return "decrease";
    else if (type == LaneRoadMark::RMLC_BOTH)
        return "both";
    else if (type == LaneRoadMark::RMLC_NONE)
        return "none";
    else
    {
        qDebug("WARNING: unknown road mark lane change type");
        return "none";
    }
}

//###################//
// Constructors      //
//###################//

LaneRoadMark::LaneRoadMark(double sOffset, RoadMarkType type, RoadMarkWeight weight, RoadMarkColor color, double width, RoadMarkLaneChange laneChange, const QString &material, double height)
    : DataElement()
    , roadMarkChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
    , type_(type)
    , weight_(weight)
    , color_(color)
    , width_(width)
    , laneChange_(laneChange)
	, material_(material)
	, height_(height)
	, userType_(NULL)
{
}

LaneRoadMark::~LaneRoadMark()
{
	if (userType_)
	{
		delUserType();
	}
}

void
LaneRoadMark::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addRoadMarkChanges(LaneRoadMark::CLR_ParentLaneChanged);
}

//####################//
// RoadMark Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneRoadMark::getSSectionEnd() const
{
    return parentLane_->getRoadMarkEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneRoadMark::getLength() const
{
    return parentLane_->getRoadMarkEnd(sOffset_) - sOffset_;
}

void
LaneRoadMark::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addRoadMarkChanges(LaneRoadMark::CLR_OffsetChanged);
}

void
LaneRoadMark::setRoadMarkType(RoadMarkType type)
{
    type_ = type;
    addRoadMarkChanges(LaneRoadMark::CLR_TypeChanged);
}

void
LaneRoadMark::setRoadMarkWeight(RoadMarkWeight weight)
{
    weight_ = weight;
    addRoadMarkChanges(LaneRoadMark::CLR_WeightChanged);
}

void
LaneRoadMark::setRoadMarkColor(RoadMarkColor color)
{
    color_ = color;
    addRoadMarkChanges(LaneRoadMark::CLR_ColorChanged);
}

void
LaneRoadMark::setRoadMarkWidth(double width)
{
    width_ = width;
    addRoadMarkChanges(LaneRoadMark::CLR_WidthChanged);
}

void
LaneRoadMark::setRoadMarkLaneChange(RoadMarkLaneChange permission)
{
    laneChange_ = permission;
    addRoadMarkChanges(LaneRoadMark::CLR_LaneChangeChanged);
}

void
LaneRoadMark::setRoadMarkMaterial(const QString &material)
{
	material_ = material;
	addRoadMarkChanges(LaneRoadMark::CLR_MaterialChanged);
}

void
LaneRoadMark::setRoadMarkHeight(double height)
{
	height_ = height;
	addRoadMarkChanges(LaneRoadMark::CLR_HeightChanged);
}

void
LaneRoadMark::setUserType(LaneRoadMarkType *roadMarkType)
{
	if (type_ != RoadMarkType::RMT_USER)
	{
		type_ = RoadMarkType::RMT_USER;
	}

	userType_ = roadMarkType;
	userType_->setRoadMarkParent(this);

	addRoadMarkChanges(LaneRoadMark::CLR_TypeChanged);
}

bool
LaneRoadMark::delUserType()
{
	if (!userType_)
	{
		qDebug() << "no user road mark type set";
		return false;
	}

	delete userType_;
	userType_ = NULL;

    return true;
}


//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneRoadMark::notificationDone()
{
    roadMarkChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneRoadMark::addRoadMarkChanges(int changes)
{
    if (changes)
    {
        roadMarkChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneRoadMark *
LaneRoadMark::getClone()
{
    LaneRoadMark *clone = new LaneRoadMark(sOffset_, type_, weight_, color_, width_, laneChange_, material_, height_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneRoadMark::accept(Visitor *visitor)
{
    visitor->visit(this);
}
