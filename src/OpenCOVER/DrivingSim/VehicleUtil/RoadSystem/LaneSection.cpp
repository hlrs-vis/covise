/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LaneSection.h"

#include <iostream>

LaneSection::LaneSection(double s)
{
    start = s;
    laneMap[0] = (new Lane(0));

    numLanesLeft = 0;
    numLanesRight = 0;
}

void LaneSection::addBatter(Batter *batter)
{
    std::cout << "adding batter id: " << batter->getBatterId() << std::endl;
    batterMap[batter->getBatterId()] = batter;
}

double LaneSection::getBatterWidth(double s, int l)
{
    std::map<int, Batter *>::iterator batterIt = batterMap.find(l);
    if (batterIt != batterMap.end())
    {
        s -= start;
        int It = (l > 0) ? 1 : -1;
        return It * batterIt->second->getBatterwidth(s);
    }
    else
    {
        return 0.0;
    }
}

double LaneSection::getBatterFall(double s, int l)
{
    std::map<int, Batter *>::iterator batterIt = batterMap.find(l);
    if (batterIt != batterMap.end())
    {
        s -= start;
        int It = (l > 0) ? 1 : -1;
        return It * batterIt->second->getBatterfall(s);
    }
    else
    {
        return 0.0;
    }
}

bool LaneSection::getBatterShouldBeTessellated(int l)
{
    std::map<int, Batter *>::iterator batterIt = batterMap.find(l);
    if (batterIt != batterMap.end())
    {
        return batterIt->second->shouldBeTessellated();
    }
    else
    {
        return true;
    }
}

std::map<int, Batter *> LaneSection::getBatterMap()
{
    return batterMap;
}

double LaneSection::getStart()
{
    return start;
}

void LaneSection::addLane(Lane *lane)
{
    laneMap[lane->getId()] = lane;

    if (lane->getId() > 0)
    {
        ++numLanesLeft;
    }
    else if (lane->getId() < 0)
    {
        ++numLanesRight;
    }
}

Lane *LaneSection::getLane(int lane)
{
    std::map<int, Lane *>::iterator laneIt = laneMap.find(lane);
    if (laneIt != laneMap.end())
    {
        return laneIt->second;
    }
    else
    {
        return NULL;
    }
}

int LaneSection::searchLane(double s, double v)
{
    double inDist = 0;
    double outDist = 0;

    return searchLane(s, v, inDist, outDist);
}

int LaneSection::searchLane(double s, double v, double &inDist, double &outDist)
{
    s -= start;

    if (v < 0)
    {
        for (int laneIt = -1; laneIt >= -numLanesRight; --laneIt)
        {
            /*if(!getLane(laneIt)) {
            std::cout << "s: " << s << ", v: " << v << ", no lane " << laneIt << ", num lanes right: " << numLanesRight << std::endl;
            std::cout << "Lanes in lane map:";
            for(std::map<int, Lane*>::iterator laneIt=laneMap.begin(); laneIt!=laneMap.end(); ++laneIt) {
               std::cout << " " << laneIt->first;
            }
            std::cout << std::endl;
         }*/
            inDist = outDist;
            outDist -= getLane(laneIt)->getWidth(s);
            if (v < inDist && v >= outDist)
            {
                return laneIt;
            }
        }
    }
    else if (v > 0)
    {
        for (int laneIt = 1; laneIt <= numLanesLeft; ++laneIt)
        {
            inDist = outDist;
            outDist += getLane(laneIt)->getWidth(s);
            if (v > inDist && v <= outDist)
            {
                return laneIt;
            }
        }
    }
    else
    {
        return 0;
    }

    //std::cout << " --inDist: " << inDist << ", outDist: " << outDist << ", s: " << s << ", v: " << v << "-- ";
    return Lane::NOLANE;
}

int LaneSection::getLanePredecessor(int l)
{
    std::map<int, Lane *>::iterator it = laneMap.find(l);
    if (it == laneMap.end())
    {
        return Lane::NOLANE;
        //return l;
        //std::cout << "Lane: " << l << " not found!" << std::endl;
    }
    else
    {
        int pred = it->second->getPredecessor();
        //std::cout << "Lane: " << l << ", predecessor: " << pred << std::endl;
        //return (pred==Lane::NOLANE) ? l : pred;
        return pred;
    }
}

int LaneSection::getLaneSuccessor(int l)
{
    std::map<int, Lane *>::iterator it = laneMap.find(l);
    if (it == laneMap.end())
    {
        return Lane::NOLANE;
        //return l;
        //std::cout << "Lane: " << l << " not found!" << std::endl;
    }
    else
    {
        int succ = it->second->getSuccessor();
        //std::cout << "Lane: " << l << ", successor: " << succ << std::endl;
        //return (succ==Lane::NOLANE) ? l : succ;
        return succ;
    }
}

void LaneSection::checkAndFixLanes()
{
    for (int i = 0; i < numLanesLeft; i++)
    {
        std::map<int, Lane *>::iterator laneIt = laneMap.find(i);
        if (laneIt == laneMap.end()) // not found
        {
            for (int n = 0; n < 100; n++)
            {
                if (laneMap.find(i + n) != laneMap.end())
                {
                    Lane *lane = laneMap.find(i + n)->second;
                    lane->setId(i);
                    laneMap[i] = lane;
                }
            }
        }
    }
}

double LaneSection::getLaneWidth(double s, int l)
{
    std::map<int, Lane *>::iterator laneIt = laneMap.find(l);
    if (laneIt != laneMap.end())
    {
        s -= start;
        if (s < 0)
            s = 0;
        int it = (l > 0) ? 1 : -1;
        return it * laneIt->second->getWidth(s);
    }
    else
    {
        return 0.0;
    }
}

double LaneSection::getLaneWidthSlope(double s, int l)
{
    std::map<int, Lane *>::iterator laneIt = laneMap.find(l);
    if (laneIt != laneMap.end())
    {
        s -= start;
        int it = (l > 0) ? 1 : -1;
        return it * laneIt->second->getWidthSlope(s);
    }
    else
    {
        return 0.0;
    }
}

double LaneSection::getDistanceToLane(double s, int l)
{
    std::map<int, Lane *>::iterator laneIt;
    s -= start;

    int it = (l > 0) ? 1 : -1;

    double d = 0;
    for (int i = 0; i != l; i += it)
    {
        laneIt = laneMap.find(i);
        if (laneIt != laneMap.end())
        {
            d += laneIt->second->getWidth(s);
        }
    }
    return it * d;
}

Vector2D LaneSection::getLaneCenter(int l, double s)
{
    //s-=start;
    if ((l != Lane::NOLANE) && !((s - start) < 0.0))
    {
        //LaneSection* section = (--laneSectionMap.upper_bound(s))->second;
        //std::cout << "getLaneCenter: Road: " << id << ", section: " << section << std::endl;
        return Vector2D(this->getDistanceToLane(s, l) + 0.5 * this->getLaneWidth(s, l), atan(0.5 * this->getLaneWidthSlope(s, l)));
    }
    else
    {
        return Vector2D(0.0, 0.0);
    }
}

void LaneSection::getRoadWidth(double s, double &left, double &right)
{
    s -= start;
    left = 0;
    right = 0;

    for (int i = 1; i <= numLanesLeft; ++i)
    {
        left += (laneMap.find(i)->second)->getWidth(s);
    }
    for (int i = -1; i >= -numLanesRight; --i)
    {
        right += (laneMap.find(i)->second)->getWidth(s);
    }
}

int LaneSection::getNumLanesLeft()
{
    return numLanesLeft;
}
int LaneSection::getNumLanesRight()
{
    return numLanesRight;
}

/**
 * Returns true if the given lane is a pedestrian sidewalk
 */
bool LaneSection::isSidewalk(int ln)
{
    if (ln != Lane::NOLANE)
    {
        Lane *l = getLane(ln);
        Lane::LaneType lt = Lane::NONE;
        if (l != NULL)
        {
            lt = l->getLaneType();
            if (lt == Lane::SIDEWALK)
            {
                return true;
            }
        }
    }
    return false;
}

RoadMark *LaneSection::getLaneRoadMark(double s, int l)
{
    s -= start;
    Lane *lane = laneMap.find(l)->second;
    if (lane == NULL)
        return NULL;
    return lane->getRoadMark(s);
}

int LaneSection::getTopRightLane()
{
    std::map<int, Lane *>::iterator laneIt = laneMap.begin();
    if (laneIt != laneMap.end())
    {
        return laneIt->first;
    }
    else
    {
        return Lane::NOLANE;
    }
}

int LaneSection::getTopLeftLane()
{
    std::map<int, Lane *>::iterator laneIt = laneMap.end();
    if (laneIt != laneMap.begin())
    {
        return (--laneIt)->first;
    }
    else
    {
        return Lane::NOLANE;
    }
}

std::map<int, Lane *> LaneSection::getLaneMap()
{
    return laneMap;
}

double LaneSection::getHeight(double s, double t)
{
    double inDist = 0;
    double outDist = 0;

    int laneId = searchLane(s, t, inDist, outDist);
    Lane *lane = getLane(laneId);
    if (!lane)
    {
        if (t >= 0.0)
            lane = getLane(getTopLeftLane());
        else
            lane = getLane(getTopRightLane());
        if (!lane)
        {
            return 0.0;
        }
        inDist = getDistanceToLane(s, lane->getId());
    }

    s -= start;
    t -= (t >= 0.0) ? inDist : -inDist;

    return lane->getHeight(s, t);
}

void LaneSection::getLaneBorderHeights(double s, int laneId, double &inner, double &outer)
{
    Lane *lane = getLane(laneId);
    if (!lane)
    {
        inner = 0.0;
        outer = 0.0;
        return;
    }

    s -= start;

    lane->getInnerOuterHeights(s, inner, outer);

    return;
}

void LaneSection::accept(XodrWriteRoadSystemVisitor *visitor)
{
    visitor->visit(this);
}
