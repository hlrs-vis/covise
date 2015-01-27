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
        return (pred == Lane::NOLANE) ? l : pred;
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
        return (succ == Lane::NOLANE) ? l : succ;
    }
}

double LaneSection::getLaneWidth(double s, int l)
{
    std::map<int, Lane *>::iterator laneIt = laneMap.find(l);
    if (laneIt != laneMap.end())
    {
        s -= start;
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

RoadMark *LaneSection::getLaneRoadMark(double s, int l)
{
    s -= start;
    return (laneMap.find(l)->second)->getRoadMark(s);
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
