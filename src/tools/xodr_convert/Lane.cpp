/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Lane.h"

#include <iostream>

const int Lane::NOLANE = -2147483648;

Lane::Lane(int laneId, std::string typeString, std::string levelString)
{
    id = laneId;
    widthMap[0.0] = (new Polynom(0.0));
    roadMarkMap[0.0] = (new RoadMark(0.0));

    if (typeString == "driving")
    {
        type = DRIVING;
    }
    else if (typeString == "stop")
    {
        type = STOP;
    }
    else if (typeString == "shoulder")
    {
        type = SHOULDER;
    }
    else if (typeString == "biking")
    {
        type = BIKING;
    }
    else if (typeString == "sidewalk")
    {
        type = SIDEWALK;
    }
    else if (typeString == "border")
    {
        type = BORDER;
    }
    else if (typeString == "restricted")
    {
        type = RESTRICTED;
    }
    else if (typeString == "parking")
    {
        type = PARKING;
    }
    else if (typeString == "mwyEntry")
    {
        type = MWYENTRY;
    }
    else if (typeString == "mwyExit")
    {
        type = MWYEXIT;
    }
    else if (typeString == "special1")
    {
        type = SPECIAL1;
    }
    else if (typeString == "special2")
    {
        type = SPECIAL2;
    }
    else if (typeString == "special3")
    {
        type = SPECIAL3;
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
    level = laneLevel;

    predecessor = NOLANE;
    successor = NOLANE;
}

int Lane::getId()
{
    return id;
}

Lane::LaneType Lane::getLaneType()
{
    return type;
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
    return ((--widthMap.upper_bound(s))->second)->getSlope(s);
}

void Lane::addRoadMark(RoadMark *rm)
{
    roadMarkMap[rm->getStart()] = rm;
}

RoadMark *Lane::getRoadMark(double s)
{
    return (--roadMarkMap.upper_bound(s))->second;
}

RoadMark::RoadMark(double s, double w, RoadMarkType rmt, RoadMarkWeight rmw, RoadMarkColor rmc, RoadMarkLaneChange rmlc)
{
    start = s;
    width = w;
    type = rmt;
    weight = rmw;
    color = rmc;
    laneChange = rmlc;
}

RoadMark::RoadMark(double s, double w, std::string tString, std::string wString, std::string cString, std::string lcString)
{
    start = s;
    width = w;

    if (tString == "none")
    {
        type = TYPE_NONE;
    }
    else if (tString == "solid")
    {
        type = TYPE_SOLID;
    }
    else if (tString == "broken")
    {
        type = TYPE_BROKEN;
    }
    else if (tString == "solid solid")
    {
        type = TYPE_SOLIDSOLID;
    }
    else if (tString == "solid broken")
    {
        type = TYPE_SOLIDBROKEN;
    }
    else if (tString == "broken solid")
    {
        type = TYPE_BROKENSOLID;
    }

    if (wString == "standard")
    {
        weight = WEIGHT_STANDARD;
    }
    else if (wString == "bold")
    {
        weight = WEIGHT_BOLD;
    }

    if (cString == "standard")
    {
        color = COLOR_STANDARD;
    }
    else if (cString == "yellow")
    {
        color = COLOR_YELLOW;
    }

    if (lcString == "increase")
    {
        laneChange = LANECHANGE_INCREASE;
    }
    else if (lcString == "decrease")
    {
        laneChange = LANECHANGE_DECREASE;
    }
    else if (lcString == "both")
    {
        laneChange = LANECHANGE_BOTH;
    }
    else if (lcString == "none")
    {
        laneChange = LANECHANGE_NONE;
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
