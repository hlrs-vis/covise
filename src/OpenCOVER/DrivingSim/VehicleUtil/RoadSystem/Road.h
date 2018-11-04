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
#include "RoadSurface.h"
#include "RoadObject.h"
#include "Crosswalk.h"
#include "RoadSignal.h"
#include "RoadSensor.h"

#include <osg/Geode>
#include <osg/Texture2D>
#include <util/coExport.h>

class Junction;
class RoadTransition;

class VEHICLEUTILEXPORT Road : public Tarmac
{
public:
    enum RoadType
    {
        UNKNOWN = 0,
        RURAL,
        MOTORWAY,
        TOWN,
        LOWSPEED,
        PEDESTRIAN,
        CUSTOM,
        BRIDGE,
        NUM_ROAD_TYPES
    };

    static float *speedLimits;

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
	void addPlanViewGeometryPolynom(double, double, double, double, double, double, double, double, double, double, double, double, double, bool);

    void addElevationPolynom(double, double, double, double, double);

    void addSuperelevationPolynom(double, double, double, double, double);
	void addCrossfallPolynom(double, double, double, double, double, std::string);
	void addShapePolynom(double, double, double, double, double, double);

    void addRoadType(double, RoadType);
    void addRoadType(double, std::string);

    // Neu Andreas 27-11-2012
    void addSpeedLimit(double, double);
    void addSpeedLimit(double, std::string);
    double getSpeedLimit(double, int);

    RoadType getRoadType(double);

    void addLaneSection(LaneSection *);

    LaneSection *getLaneSection(double);
    LaneSection *getLaneSectionNext(double);
    double getLaneSectionEnd(double);
    int traceLane(int, double, double);
    double getLaneEnd(int &, double, int, bool &);

    void addRoadSurface(RoadSurface *, double, double);

    void addRoadObject(RoadObject *);

    void addCrosswalk(Crosswalk *);
    Crosswalk *getCrosswalk(double, int);
    Crosswalk *getNextCrosswalk(double, int, int);
    bool isAnyCrosswalks(double, double, int);

    std::vector<int> getSidewalks(double);

    void addRoadSignal(RoadSignal *);

    std::string getModelFile(double);
    RoadSignal *getRoadSignal(const unsigned int &);
    unsigned int getNumRoadSignals();

    void addRoadSensor(RoadSensor *);
    std::map<double, RoadSensor *>::iterator getRoadSensorMapEntry(const double &);
    std::map<double, RoadSensor *>::iterator getRoadSensorMapEnd();

    void setLength(double);
    double getLength();

    double getHeading(double);
    double getCurvature(double);
    double getSuperelevation(double, double);

    Vector3D getCenterLinePoint(double);
    Vector3D getTangentVector(double);
    Vector3D getNormalVector(double);

    RoadPoint getChordLinePoint(double);
    Vector3D getChordLinePlanViewPoint(double);
    double getChordLineElevation(double);
    double getChordLineElevationSlope(double);
    double getChordLineElevationCurvature(double);

    RoadPoint getRoadPoint(double, double);

    bool isOnRoad(Vector2D);

    double random();

    void getBatterPoint(double, int, RoadPoint &, RoadPoint &, RoadPoint &, double &, int);
    void getGuardRailPoint(RoadPoint &, RoadPoint &, bool, bool, std::map<double, LaneSection *>::iterator);
    osg::Geometry *getGuardRailPost(RoadPoint, RoadPoint, RoadPoint, RoadPoint);

    void getLaneRoadPoints(double, int, RoadPoint &, RoadPoint &, double &, double &);
    void getLaneWidthAndOffset(double, int, double &width, double &widthOffset);

    void getLaneRoadPointVectors(double, RoadPointVector &, RoadPointVector &, DistanceVector &, DistanceVector &);

    Transform getRoadTransform(const double &, const double &);
    Transform getSignalTransform(const double &, const double &);

    void getRoadSideWidths(double, double &, double &);

    int getLaneNumber(double s, double t); //get lane number for a given t. if t is off road, lane is equal to the last lane

    double getRoadLaneOuterPos(double s, int lane);

    Junction *getJunction();
    bool isJunctionPath();

    void addCrossingRoadPosition(Road *, double);
    const std::map<Road *, double> &getCrossingRoadPositionMap();

    std::string getTypeSpecifier();
    std::map<double, PlaneCurve *> getPlaneCurveMap();
    std::map<double, Polynom *> getElevationMap();
    std::map<double, LateralProfile *> getLateralMap();
	std::map<double, LaneSection *> getLaneSectionMap();

    TarmacConnection *getPredecessorConnection();
    TarmacConnection *getSuccessorConnection();
    TarmacConnection *getConnection(int);

    Vector2D searchPositionNoBorder(const Vector3D &, double);
    Vector2D searchPosition(const Vector3D &, double);
    int searchLane(double, double);

    std::set<RoadTransition> getConnectingRoadTransitionSet(const RoadTransition &);

    virtual osg::Geode *getRoadGeode();
    virtual osg::Group *getRoadBatterGroup(bool, bool);
    virtual osg::Geode *getGuardRailGeode();

    osg::Group *createObjectsGroup();

    void accept(RoadSystemVisitor *);

    static bool compare(Road *, Road *);
    int getPriority()
    {
        return priority;
    };
    void setPriority(int p)
    {
        priority = p;
    };

	void addLaneOffset(double, double, double, double, double);
	double getLaneOffset(double);
	double getLaneOffsetSlope(double s);
	std::map<double, Polynom *> getLaneOffsetMap();

protected:
    osg::Group *createRoadGroup(bool, bool);
    osg::Geode *createGuardRailGeode(std::map<double, LaneSection *>::iterator lsIt);

	std::map<double, Polynom *> laneOffsetMap;

    TarmacConnection *predecessor;
    TarmacConnection *successor;
    Road *leftNeighbour;
    int leftNeighbourDirection;
    Road *rightNeighbour;
    int rightNeighbourDirection;

    Junction *junction;

    double length;
    int priority;

    std::map<double, PlaneCurve *> xyMap;
    std::map<double, Polynom *> zMap;
    std::map<double, LateralProfile *> lateralProfileMap;

    std::map<double, RoadType> roadTypeMap;

    // Neu Andreas 27-11-2012
    std::map<double, double> speedLimitMap;

    std::map<double, LaneSection *> laneSectionMap;

    std::map<double, RoadSurface *> roadSurfaceMap;

    std::vector<RoadObject *> roadObjectVector;

    std::vector<Crosswalk *> crosswalkVector;

    std::vector<RoadSignal *> roadSignalVector;

    std::map<double, RoadSensor *> roadSensorMap;

    std::map<Road *, double> crossingRoadPositionMap;

	roadShapeSections *shapeSections;

    osg::Geode *roadGeode;
    osg::Geode *batterGeode;
    osg::Geode *guardRailGeode;
    osg::Group *roadGroup;

    static osg::StateSet *roadStateSet;
    static osg::StateSet *batterStateSet;
    static osg::StateSet *concreteBatterStateSet;

    static osg::Texture2D *roadTex;
    static osg::Texture2D *batterTex;
    static osg::Texture2D *concreteTex;
};

class VEHICLEUTILEXPORT RoadTransition
{
public:
    RoadTransition(Road *r, int d, Junction *j = NULL)
        : road(r)
        , direction(d)
        , junction(j)
    {
    }
	RoadTransition()
		: road(NULL)
		, direction(0)
		, junction(NULL)
	{
	}

    Road *road;
    int direction;
    Junction *junction;

    bool operator==(const RoadTransition &trans) const
    {
        return ((road == trans.road) && (direction == trans.direction));
    }

    bool operator<(const RoadTransition &trans) const
    {
        return (road->getHeading(direction == -1 ? road->getLength() : 0.0) < trans.road->getHeading(trans.direction == -1 ? trans.road->getLength() : 0.0));
    }
};

#endif
