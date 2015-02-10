/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LaneSection_h
#define LaneSection_h

#include <map>
#include <util/coExport.h>

#include "Lane.h"

class VEHICLEUTILEXPORT LaneSection
{
public:
    LaneSection(double);

    double getStart();

    void addLane(Lane *);
    Lane *getLane(int);
    void addBatter(Batter *);

    int searchLane(double, double);
    int searchLane(double, double, double &, double &);

    int getLanePredecessor(int);
    int getLaneSuccessor(int);

    double getLaneWidth(double, int = 0);
    double getBatterWidth(double, int = 0);
    double getBatterFall(double, int = 0);
    bool getBatterShouldBeTessellated(int);
    std::map<int, Batter *> getBatterMap();
    double getLaneWidthSlope(double, int = 0);
    double getLaneSpanWidth(int, int, double);

    double getDistanceToLane(double, int);
    Vector2D getLaneCenter(int, double);

    void getRoadWidth(double, double &, double &);

    int getNumLanesLeft();
    int getNumLanesRight();

    bool isSidewalk(int ln);

    RoadMark *getLaneRoadMark(double, int = 0);

    int getTopRightLane();
    int getTopLeftLane();

    std::map<int, Lane *> getLaneMap();

    double getHeight(double, double);
    void getLaneBorderHeights(double, int, double &, double &);

    void accept(XodrWriteRoadSystemVisitor *);

    void checkAndFixLanes();

protected:
    double start;

    std::map<int, Lane *> laneMap;
    std::map<int, Batter *> batterMap;
    int numLanesLeft;
    int numLanesRight;
};

#endif
