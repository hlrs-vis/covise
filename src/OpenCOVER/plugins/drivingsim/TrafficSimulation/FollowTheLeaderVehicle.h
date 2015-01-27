/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FollowTheLeaderVehicle_h
#define FollowTheLeaderVehicle_h

#include "Vehicle.h"
#include "CarGeometry.h"
#include "LaneSection.h"
#include "Road.h"
#include "VehicleUtils.h"

#include <osg/MatrixTransform>
#include <map>
#include <iostream>

class VehicleAction;
typedef std::multimap<double, VehicleAction *> VehicleActionMap;

class FollowTheLeaderVehicle : public Vehicle
{
public:
    enum DrivingState
    {
        RURAL,
        MOTORWAY,
        OVERTAKE,
        EXIT_MOTORWAY,
        ENTER_MOTORWAY
    };

    FollowTheLeaderVehicle(std::string, CarGeometry * = NULL, Road * = NULL, double = 0.0, int = -1, double = 100, int = 1);

    ~FollowTheLeaderVehicle();

    void move(double);

    Road *getRoad() const;

    double getU() const;
    double getDu() const;

    double getBoundingCircleRadius();

    int getLane() const;
    //std::set<int> getLaneSet() const;
    bool isOnLane(int) const;

    void brake();

    void addRouteTransition(const RoadTransition &);

    VehicleGeometry *getVehicleGeometry();
    CarGeometry *getCarGeometry();

protected:
    LaneSection *currentSection;
    int currentLane;

    double u, v, s;
    double du, dv;

    RoadTransitionList roadTransitionList;
    RoadTransitionList routeTransitionList;
    std::list<RoadTransitionList::iterator> junctionPathLookoutList;
    RoadTransitionList::iterator currentTransition;
    DrivingState currentDrivingState;
    Road::RoadType currentRoadType;
    VehicleActionMap vehiclePositionActionMap;
    VehicleActionMap vehicleTimerActionMap;

    CarGeometry *geometry;

    Transform vehicleTransform;

    double vel;
    double kv, dampingV, Tv;
    double hdg, dHdg, Khdg, Dhdg, Kv, Dv;
    double dist, velt, velw, aqmax;
    double T;
    double a, b;
    double delta, cDelta;
    double p;
    double brakeDec;
    double minTransitionListExtent;
    double boundRadius;
    double junctionWaitTime;
    double junctionWaitTriggerDist;
    bool sowedDetermineNextRoadVehicleAction;

    std::map<DrivingState, Matrix3D3D> wMatMap;

    double timer;

    double etd(double, double); //effective target distance
    Vehicle *getNextVehicle(double &, double &, int, int);
    VehicleRelation locateVehicle(int, int);
    double locateLaneEnd(int);
    double locateNextJunction();

    Matrix3D3D buildAccelerationMatrix(int);
    double getAccelerationQuality(const Matrix3D3D &, const double &) const;
    double getAccelerationQuality(const Matrix3D3D &, const double &, DrivingState) const;
    void setDrivingState(DrivingState);
    void setRoadType(Road::RoadType);

    bool findRoute(RoadTransition);
    bool planRoute();

    bool executeActionMap();

    friend class VehicleAction;
    friend class DetermineNextRoadVehicleAction;
    friend class WaitAtJunctionVehicleAction;
    friend class SetDrivingStateVehicleAction;
};

class VehicleAction
{
public:
    virtual ~VehicleAction()
    {
    }

    virtual void operator()(FollowTheLeaderVehicle *veh)
    {
        std::cout << "Vehicle " << veh->name << ": Performing dummy action at: " << veh->s << std::endl;
    }

protected:
};

class DetermineNextRoadVehicleAction : public VehicleAction
{
public:
    DetermineNextRoadVehicleAction()
        : VehicleAction()
    {
    }

    void operator()(FollowTheLeaderVehicle *);

    static int removeAllActions(FollowTheLeaderVehicle *);

protected:
};

class WaitAtJunctionVehicleAction : public VehicleAction
{
public:
    WaitAtJunctionVehicleAction()
        : VehicleAction()
    {
    }

    void operator()(FollowTheLeaderVehicle *);

protected:
};

class SetDrivingStateVehicleAction : public VehicleAction
{
public:
    SetDrivingStateVehicleAction(FollowTheLeaderVehicle::DrivingState s)
        : VehicleAction()
        , state(s)
    {
    }

    void operator()(FollowTheLeaderVehicle *);

protected:
    FollowTheLeaderVehicle::DrivingState state;
};

#endif
