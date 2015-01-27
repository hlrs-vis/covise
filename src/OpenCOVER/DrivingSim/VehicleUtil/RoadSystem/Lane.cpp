/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Lane.h"
#include <iostream>

const int Lane::NOLANE = -214748364;

Lane::Lane(int laneId, std::string typeString, std::string levelString)
{
    id = laneId;
    widthMap[0.0] = (new Polynom(0.0));
    roadMarkMap[0.0] = (new RoadMark(0.0));

    if (typeString == "driving")
    {
        this->typeString = typeString;
        type = DRIVING;
    }
    else if (typeString == "stop")
    {
        this->typeString = typeString;
        type = STOP;
    }
    else if (typeString == "shoulder")
    {
        this->typeString = typeString;
        type = SHOULDER;
    }
    else if (typeString == "biking")
    {
        this->typeString = typeString;
        type = BIKING;
    }
    else if (typeString == "sidewalk")
    {
        this->typeString = typeString;
        type = SIDEWALK;
    }
    else if (typeString == "border")
    {
        this->typeString = typeString;
        type = BORDER;
    }
    else if (typeString == "restricted")
    {
        this->typeString = typeString;
        type = RESTRICTED;
    }
    else if (typeString == "parking")
    {
        this->typeString = typeString;
        type = PARKING;
    }
    else if (typeString == "mwyEntry")
    {
        this->typeString = typeString;
        type = MWYENTRY;
    }
    else if (typeString == "mwyExit")
    {
        this->typeString = typeString;
        type = MWYEXIT;
    }
    else if (typeString == "special1")
    {
        this->typeString = typeString;
        type = SPECIAL1;
    }
    else if (typeString == "special2")
    {
        this->typeString = typeString;
        type = SPECIAL2;
    }
    else if (typeString == "special3")
    {
        this->typeString = typeString;
        type = SPECIAL3;
    }
    else
    {
        this->typeString = "none";
        type = NONE;
    }

    if (levelString == "true")
    {
        level = true;
    }
    else
    {
        level = false;
    }

    predecessor = NOLANE;
    successor = NOLANE;
}

Lane::Lane(int laneId, LaneType laneType, bool laneLevel)
{
    id = laneId;
    widthMap[0.0] = (new Polynom(0.0));
    roadMarkMap[0.0] = (new RoadMark(0.0));

    type = laneType;
    switch (type)
    {
    case NONE:
        typeString = "none";
        break;
    case DRIVING:
        typeString = "driving";
        break;
    case STOP:
        typeString = "stop";
        break;
    case SHOULDER:
        typeString = "shoulder";
        break;
    case BIKING:
        typeString = "biking";
        break;
    case SIDEWALK:
        typeString = "sidewalk";
        break;
    case BORDER:
        typeString = "border";
        break;
    case RESTRICTED:
        typeString = "restricted";
        break;
    case PARKING:
        typeString = "parking";
        break;
    case MWYENTRY:
        typeString = "mwyentry";
        break;
    case MWYEXIT:
        typeString = "mwyexit";
        break;
    case SPECIAL1:
        typeString = "special1";
        break;
    case SPECIAL2:
        typeString = "special2";
        break;
    case SPECIAL3:
        typeString = "special3";
        break;
    default:
        typeString = "none";
    }

    level = laneLevel;

    predecessor = NOLANE;
    successor = NOLANE;
}

Batter::Batter(int batterId, bool tess)
{
    //batterid = (batterId>0) ?  batterId-1 : batterId+1 ;
    batterid = batterId;
    tessellate = tess;
    batterWidthMap[0.0] = 6.0;
    batterFallMap[0.0] = 0.3;
}

void Batter::addBatterWidth(double s, double a)
{
    batterWidthMap[s] = (a);
}

double Batter::getBatterwidth(double s)
{
    return ((--batterWidthMap.upper_bound(s))->second);
}

std::map<double, double> Batter::getBatterWidthMap()
{
    return batterWidthMap;
}

void Batter::addBatterFall(double s, double a)
{
    batterFallMap[s] = (a);
}

double Batter::getBatterfall(double s)
{
    return ((--batterFallMap.upper_bound(s))->second);
}

int Batter::getBatterId()
{
    return batterid;
}

int Lane::getId()
{
    return id;
}

void Lane::setId(int i)
{
    id = i;
}

Lane::LaneType Lane::getLaneType()
{
    return type;
}

void Lane::setLaneType(Lane::LaneType t)
{
    type = t;
}

std::string Lane::getLaneTypeString()
{
    return typeString;
}

void Lane::setPredecessor(int pred)
{
    predecessor = pred;
    //std::cout << "Lane: " << id << ", predecessor set: " << pred << std::endl;
}

void Lane::setSuccessor(int succ)
{
    successor = succ;
    //std::cout << "Lane: " << id << ", successor set: " << succ << std::endl;
}

int Lane::getPredecessor()
{
    return predecessor;
}

int Lane::getSuccessor()
{
    return successor;
}

bool Lane::isOnLevel()
{
    return level;
}

void Lane::addWidth(double s, double a, double b, double c, double d)
{
    widthMap[s] = (new Polynom(s, a, b, c, d));
}
double Lane::getWidth(double s)
{
    return ((--widthMap.upper_bound(s))->second)->getValue(s);
}

double Lane::getWidthSlope(double s)
{
    return ((--widthMap.upper_bound(s))->second)->getSlopeAngle(s);
}

// Neu Andreas 27-11-2012
void Lane::addSpeedLimit(double s, double sl)
{
    speedLimitMap[s] = sl;
}
double Lane::getSpeedLimit(double s)
{
    while (!speedLimitMap.empty())
    {
        return (--speedLimitMap.upper_bound(s))->second;
    }
    return 0;
}
void Lane::addRoadMark(RoadMark *rm)
{
    roadMarkMap[rm->getStart()] = rm;
}

std::map<double, Polynom *> Lane::getWidthMap()
{
    return widthMap;
}

std::map<double, RoadMark *> Lane::getRoadMarkMap()
{
    return roadMarkMap;
}

void Lane::accept(XodrWriteRoadSystemVisitor *visitor)
{
    visitor->visit(this);
}

RoadMark *Lane::getRoadMark(double s)
{
    return (--roadMarkMap.upper_bound(s))->second;
}

void Lane::addHeight(double s, double inner, double outer)
{
    if (heightMap.begin() == heightMap.end())
    {
        heightMap[0.0] = std::make_pair((double)0.0, (double)0.0);
    }

    heightMap[s] = std::make_pair(inner, outer);
}

double Lane::getHeight(double s, double t)
{
    if (heightMap.begin() == heightMap.end())
    {
        return 0.0;
    }

    std::map<double, std::pair<double, double> >::iterator upper = ((heightMap.upper_bound(s)));
    std::map<double, std::pair<double, double> >::iterator lower;
    double inner, outer;
    if (upper == heightMap.end())
    {
        --upper;
        lower = upper;

        inner = lower->second.first;
        outer = lower->second.second;
    }
    else
    {
        lower = upper;
        --lower;

        inner = lower->second.first + (upper->second.first - lower->second.first) / (upper->first - lower->first) * s;
        outer = lower->second.second + (upper->second.second - lower->second.second) / (upper->first - lower->first) * s;
    }

    double natT = fabs(t / getWidth(s));
    if (natT > 1.0)
    {
        natT = 1.0;
    }

    return inner + (outer - inner) * natT;
}

void Lane::getInnerOuterHeights(double s, double &inner, double &outer)
{
    if (heightMap.begin() == heightMap.end())
    {
        inner = 0.0;
        outer = 0.0;
        return;
    }

    std::map<double, std::pair<double, double> >::iterator upper = ((heightMap.upper_bound(s)));
    std::map<double, std::pair<double, double> >::iterator lower;
    if (upper == heightMap.end())
    {
        --upper;
        lower = upper;

        inner = lower->second.first;
        outer = lower->second.second;
    }
    else
    {
        lower = upper;
        --lower;

        inner = lower->second.first + (upper->second.first - lower->second.first) / (upper->first - lower->first) * s;
        outer = lower->second.second + (upper->second.second - lower->second.second) / (upper->first - lower->first) * s;
    }

    return;
}

RoadMark::RoadMark(double s, double w, RoadMarkType rmt, RoadMarkWeight rmw, RoadMarkColor rmc, RoadMarkLaneChange rmlc)
{
    start = s;
    width = w;
    type = rmt;
    switch (type)
    {
    case TYPE_NONE:
        typeString = "none";
        break;
    case TYPE_SOLID:
        typeString = "solid";
        break;
    case TYPE_BROKEN:
        typeString = "broken";
        break;
    case TYPE_SOLIDSOLID:
        typeString = "solid solid";
        break;
    case TYPE_SOLIDBROKEN:
        typeString = "solid broken";
        break;
    case TYPE_BROKENSOLID:
        typeString = "broken solid";
        break;
    default:
        typeString = "none";
    };
    weight = rmw;
    switch (weight)
    {
    case WEIGHT_STANDARD:
        weightString = "standard";
        break;
    case WEIGHT_BOLD:
        weightString = "bold";
        break;
    default:
        weightString = "standard";
    };
    color = rmc;
    switch (color)
    {
    case COLOR_STANDARD:
        colorString = "standard";
        break;
    case COLOR_YELLOW:
        colorString = "yellow";
        break;
    default:
        colorString = "standard";
    };
    laneChange = rmlc;
    switch (laneChange)
    {
    case LANECHANGE_INCREASE:
        laneChangeString = "increase";
        break;
    case LANECHANGE_DECREASE:
        laneChangeString = "decrease";
        break;
    case LANECHANGE_BOTH:
        laneChangeString = "both";
        break;
    case LANECHANGE_NONE:
        laneChangeString = "none";
        break;
    default:
        laneChangeString = "both";
    };
}

RoadMark::RoadMark(double s, double w, std::string tString, std::string wString, std::string cString, std::string lcString)
{
    start = s;
    width = w;

    if (tString == "solid")
    {
        type = TYPE_SOLID;
        typeString = tString;
    }
    else if (tString == "broken")
    {
        type = TYPE_BROKEN;
        typeString = tString;
    }
    else if (tString == "solid solid")
    {
        type = TYPE_SOLIDSOLID;
        typeString = tString;
    }
    else if (tString == "solid broken")
    {
        type = TYPE_SOLIDBROKEN;
        typeString = tString;
    }
    else if (tString == "broken solid")
    {
        type = TYPE_BROKENSOLID;
        typeString = tString;
    }
    else /*if(tString == "none")*/
    {
        type = TYPE_NONE;
        typeString = "none";
    }

    if (wString == "bold")
    {
        weight = WEIGHT_BOLD;
        weightString = wString;
    }
    else /*if(wString == "standard")*/
    {
        weight = WEIGHT_STANDARD;
        weightString = "standard";
    }

    if (cString == "yellow")
    {
        color = COLOR_YELLOW;
        colorString = cString;
    }
    else /*if(cString == "standard")*/
    {
        color = COLOR_STANDARD;
        colorString = "standard";
    }

    if (lcString == "increase")
    {
        laneChange = LANECHANGE_INCREASE;
        laneChangeString = lcString;
    }
    else if (lcString == "decrease")
    {
        laneChange = LANECHANGE_DECREASE;
        laneChangeString = lcString;
    }
    else if (lcString == "none")
    {
        laneChange = LANECHANGE_NONE;
        laneChangeString = lcString;
    }
    else /*if(lcString == "both")*/
    {
        laneChange = LANECHANGE_BOTH;
        laneChangeString = "both";
    }
}

RoadMark::RoadMarkType RoadMark::getType()
{
    return type;
}

RoadMark::RoadMarkWeight RoadMark::getWeight()
{
    return weight;
}

RoadMark::RoadMarkColor RoadMark::getColor()
{
    return color;
}

double RoadMark::getWidth()
{
    return width;
}

RoadMark::RoadMarkLaneChange RoadMark::getLangeChange()
{
    return laneChange;
}

std::string RoadMark::getTypeString()
{
    return typeString;
}

std::string RoadMark::getWeightString()
{
    return weightString;
}

std::string RoadMark::getColorString()
{
    return colorString;
}

std::string RoadMark::getLaneChangeString()
{
    return laneChangeString;
}

void RoadMark::accept(XodrWriteRoadSystemVisitor *visitor)
{
    visitor->visit(this);
}
