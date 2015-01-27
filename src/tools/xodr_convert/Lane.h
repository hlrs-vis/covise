/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Lane_h
#define Lane_h

#include <set>
#include <string>

#include "Types.h"

class RoadMark;

class Lane
{
public:
    static const int NOLANE;

    enum LaneType
    {
        NONE,
        DRIVING,
        STOP,
        SHOULDER,
        BIKING,
        SIDEWALK,
        BORDER,
        RESTRICTED,
        PARKING,
        MWYENTRY,
        MWYEXIT,
        SPECIAL1,
        SPECIAL2,
        SPECIAL3
    };

    Lane(int, std::string, std::string);
    Lane(int, LaneType = NONE, bool = false);

    int getId();

    Lane::LaneType getLaneType();

    void setPredecessor(int);
    void setSuccessor(int);

    int getPredecessor();
    int getSuccessor();

    void addWidth(double, double, double, double, double);
    double getWidth(double);
    double getWidthSlope(double);

    void addRoadMark(RoadMark *);
    RoadMark *getRoadMark(double);

private:
    int id;
    LaneType type;
    bool level;

    int predecessor;
    int successor;

    std::map<double, RoadMark *> roadMarkMap;

    std::map<double, Polynom *> widthMap;
};

class RoadMark : public Curve
{
public:
    enum RoadMarkType
    {
        TYPE_NONE = 0,
        TYPE_SOLID = 1,
        TYPE_BROKEN = 2,
        TYPE_SOLIDSOLID = 11,
        TYPE_SOLIDBROKEN = 12,
        TYPE_BROKENSOLID = 21
    };
    enum RoadMarkWeight
    {
        WEIGHT_STANDARD,
        WEIGHT_BOLD
    };
    enum RoadMarkColor
    {
        COLOR_STANDARD,
        COLOR_YELLOW
    };
    enum RoadMarkLaneChange
    {
        LANECHANGE_INCREASE = 1,
        LANECHANGE_DECREASE = -1,
        LANECHANGE_BOTH = 2,
        LANECHANGE_NONE = 0
    };

    RoadMark(double, double = 0.12, RoadMarkType = TYPE_NONE, RoadMarkWeight = WEIGHT_STANDARD, RoadMarkColor = COLOR_STANDARD, RoadMarkLaneChange = LANECHANGE_NONE);
    RoadMark(double, double, std::string = "none", std::string = "standard", std::string = "standard", std::string = "none");

    RoadMarkType getType();
    RoadMarkWeight getWeight();
    RoadMarkColor getColor();
    double getWidth();
    RoadMarkLaneChange getLangeChange();

private:
    RoadMarkType type;
    RoadMarkWeight weight;
    RoadMarkColor color;
    double width;
    RoadMarkLaneChange laneChange;
};

#endif
