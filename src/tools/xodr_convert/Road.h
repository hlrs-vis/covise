/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Road_h
#define Road_h

#include <iostream>
#include <map>
#include <set>
#include <cmath>

#include "Tarmac.h"
#include "Types.h"
#include "LaneSection.h"

#include <osg/Geode>

class Junction;

class Road : public Tarmac
{
public:
    enum RoadType
    {
        UNKNOWN,
        RURAL,
        MOTORWAY,
        TOWN,
        LOWSPEED,
        PEDESTRIAN
    };

    Road(std::string, std::string = "no name", double = 0, Junction * = NULL);

    void setPredecessorConnection(TarmacConnection *);
    void setSuccessorConnection(TarmacConnection *);
    void setLeftNeighbour(Road *, int);
    void setRightNeighbour(Road *, int);
    void setJunction(Junction *);

    void addPlanViewGeometryLine(double, double, double, double, double);
    void addPlanViewGeometrySpiral(double, double, double, double, double, double, double);
    void addPlanViewGeometryArc(double, double, double, double, double, double);
    void addPlanViewGeometryPolynom(double, double, double, double, double, double, double, double, double);

    void addElevationPolynom(double, double, double, double, double);

    void addSuperelevationPolynom(double, double, double, double, double);
    void addCrossfallPolynom(double, double, double, double, double, std::string);

    void addRoadType(double, RoadType);
    void addRoadType(double, std::string);

    void addLaneSection(LaneSection *);

    LaneSection *getLaneSection(double);

    double getLength();

    double getHeading(double);

    RoadPoint getChordLinePoint(double);

    RoadPoint getRoadPoint(double, double);

    void getLaneRoadPoints(double, int, RoadPoint &, RoadPoint &, double &, double &);

    void getLaneRoadPointVectors(double, RoadPointVector &, RoadPointVector &, DistanceVector &, DistanceVector &);

    Transform getRoadTransform(const double &, const double &);
    Vector2D getLaneCenter(const int &, const double &);

    Junction *getJunction();
    bool isJunctionPath();

    TarmacConnection *getPredecessorConnection();
    TarmacConnection *getSuccessorConnection();

    virtual osg::Geode *getRoadGeode();

protected:
    osg::Geode *createRoadGeode();

    TarmacConnection *predecessor;
    TarmacConnection *successor;
    Road *leftNeighbour;
    int leftNeighbourDirection;
    Road *rightNeighbour;
    int rightNeighbourDirection;

    Junction *junction;

    double length;

    std::map<double, PlaneCurve *> xyMap;
    std::map<double, Polynom *> zMap;
    std::map<double, LateralProfile *> lateralProfileMap;

    std::map<double, RoadType> roadTypeMap;

    std::map<double, LaneSection *> laneSectionMap;

    osg::Geode *roadGeode;
};

#endif
