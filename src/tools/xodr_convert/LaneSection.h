/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LaneSection_h
#define LaneSection_h

#include <map>

#include "Lane.h"

class LaneSection
{
public:
    LaneSection(double);

    double getStart();

    void addLane(Lane *);
    Lane *getLane(int);

    int getLanePredecessor(int);
    int getLaneSuccessor(int);

    double getLaneWidth(double, int = 0);
    double getLaneWidthSlope(double, int = 0);

    double getDistanceToLane(double, int);

    void getRoadWidth(double, double &, double &);

    int getNumLanesLeft();
    int getNumLanesRight();

    RoadMark *getLaneRoadMark(double, int = 0);

    int getTopRightLane();
    int getTopLeftLane();

protected:
    double start;

    std::map<int, Lane *> laneMap;
    int numLanesLeft;
    int numLanesRight;
};

#endif
