/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Lane_h
#define Lane_h

#include <set>
#include <string>
#include <util/coExport.h>

#include "Types.h"

class RoadMark;

class VEHICLEUTILEXPORT Batter
{
public:
    Batter(int, bool);
    int getBatterId();
    bool shouldBeTessellated()
    {
        return tessellate;
    }
    void addBatterWidth(double, double);
    void addBatterFall(double, double);
    double getBatterwidth(double);
    double getBatterfall(double);
    std::map<double, double> getBatterWidthMap();

    void accept(XodrWriteRoadSystemVisitor *);

private:
    int batterid;
    bool tessellate;
    std::map<double, double> batterWidthMap;
    std::map<double, double> batterFallMap;
};

class VEHICLEUTILEXPORT Lane
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
    void setId(int);

    Lane::LaneType getLaneType();
    void setLaneType(Lane::LaneType);
    std::string getLaneTypeString();

    void setPredecessor(int);
    void setSuccessor(int);

    int getPredecessor();
    int getSuccessor();

    bool isOnLevel();

    void addWidth(double, double, double, double, double);
    double getWidth(double);
    double getWidthSlope(double);
    // Neu Andreas 27-11-2012
    std::map<double, double> speedLimitMap;
    void addSpeedLimit(double, double);
    double getSpeedLimit(double);

    void addRoadMark(RoadMark *);
    RoadMark *getRoadMark(double);

    void addHeight(double, double, double);
    double getHeight(double, double);
    void getInnerOuterHeights(double, double &, double &);

    std::map<double, Polynom *> getWidthMap();
    std::map<double, RoadMark *> getRoadMarkMap();

    void accept(XodrWriteRoadSystemVisitor *);

private:
    int id;
    LaneType type;
    bool level;

    int predecessor;
    int successor;

    std::string typeString;

    std::map<double, RoadMark *> roadMarkMap;

    std::map<double, Polynom *> widthMap;

    std::map<double, std::pair<double, double> > heightMap;
};

class VEHICLEUTILEXPORT RoadMark : public Curve
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
        COLOR_STANDARD = 0,
        COLOR_YELLOW = 1
    };
    enum RoadMarkLaneChange
    {
        LANECHANGE_INCREASE = 1,
        LANECHANGE_DECREASE = -1,
        LANECHANGE_BOTH = 2,
        LANECHANGE_NONE = 0
    };

    RoadMark(double, double = 0.12, RoadMarkType = TYPE_NONE, RoadMarkWeight = WEIGHT_STANDARD, RoadMarkColor = COLOR_STANDARD, RoadMarkLaneChange = LANECHANGE_BOTH);
    RoadMark(double, double, std::string = "none", std::string = "standard", std::string = "standard", std::string = "both");

    double getWidth();
    RoadMarkType getType();
    RoadMarkWeight getWeight();
    RoadMarkColor getColor();
    RoadMarkLaneChange getLangeChange();
    std::string getTypeString();
    std::string getWeightString();
    std::string getColorString();
    std::string getLaneChangeString();

    void accept(XodrWriteRoadSystemVisitor *);

private:
    double width;
    RoadMarkType type;
    std::string typeString;
    RoadMarkWeight weight;
    std::string weightString;
    RoadMarkColor color;
    std::string colorString;
    RoadMarkLaneChange laneChange;
    std::string laneChangeString;
};

#endif
