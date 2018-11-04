/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef AgentVehicle_h
#define AgentVehicle_h
#include "Vehicle.h"
#include "CarGeometry.h"
#include <VehicleUtil/RoadSystem/LaneSection.h>
#include <VehicleUtil/RoadSystem/Road.h>
#include <VehicleUtil/RoadSystem/Crosswalk.h>
#include "VehicleUtils.h"
#include "PorscheFFZ.h"

#include <cover/coVRPluginSupport.h>
#include <vector>

struct TRAFFICSIMULATIONEXPORT RoadTransitionPoint
{
    RoadTransitionPoint(const RoadTransitionList::iterator &transIt, double uset)
        : transition(transIt)
        , u(uset)
    {
    }

    RoadTransitionList::iterator transition;
    double u;
};

struct TRAFFICSIMULATIONEXPORT VehicleParameters
{

    VehicleParameters(double dist = 2, double velw = 30, double aqmax = 6.0, double T = 1.4, double a = 3.0, double b = 1.5,
                      double bsave = 15.0, double delta = 4.0, double cDelta = 0.8, double p = 0.3, double pD = 20.0, double lod = 3.4e38, double pt = 5.0, std::string ot = "car")
        : deltaSmin(dist)
        , //minimaler Abstand
        dUtarget(velw)
        , //max. Geschwindigkeit, die der Fahrzeugagent zu fahren bereit ist
        accCrossmax(aqmax)
        , //maximal zulässige Kreisbeschleunigung
        respTime(T)
        , //zeitlicher Sicherheitsabstand
        accMax(a)
        , //maximal mögliche Beschleunigung des Fahrzeugangenten
        decComf(b)
        , //komfortable Bremsverzögerung
        decSave(bsave)
        , //maximal zulässige Verzögerung beim Spurwechsel
        approachFactor(delta)
        , //bestimmt die Aggressivität der Annäherung an die Sollgeschwind.
        lcTreshold(cDelta)
        , //Rechtsfahrgebotsgewichtung
        politeFactor(p)
        , panicDistance(pD)
        , rangeLOD(lod)
        , //(LOD = Leven of Detail) regelt das Ein- und Ausschalten des Fahzeugs in Abhängigkeit vom Abstand vom Betrachter
        passThreshold(pt)
        , //Mindestabstand zum Überholen
        obstacleType(ot)

    {
    }

    double deltaSmin;
    double dUtarget;
    double accCrossmax;
    double respTime;
    double accMax;
    double decComf;
    double decSave;
    double approachFactor;
    double lcTreshold;
    double politeFactor;
    double panicDistance;
    double rangeLOD; // level of detail range
    double passThreshold; // minimum amount of space needed to pass the car in front (e.g., if there's less space between the cars, no passing)
    std::string obstacleType;
};

class VehicleAction;
typedef std::multimap<double, VehicleAction *> VehicleActionMap;

//class AgentVehicle : public FollowTheLeaderVehicle
class TRAFFICSIMULATIONEXPORT AgentVehicle : public Vehicle
{
public:
    AgentVehicle(AgentVehicle *, std::string, const VehicleParameters & = VehicleParameters(), Road * = NULL, double = 0.0, int = -1, double = 100, int = 1);
    AgentVehicle(std::string, CarGeometry * = NULL, const VehicleParameters & = VehicleParameters(), Road * = NULL, double = 0.0, int = -1, double = 100, int = 1);
    ~AgentVehicle();

    void move(double dt);
	void setPosition(osg::Vec3 &pos, osg::Vec3 &vec);
	void setTransform(osg::Matrix m);
	void setTransform(Transform vehicleTransform, double hdg);
    void makeDecision();

    void checkForCrosswalk(double dt);
    bool canPass()
    {
        return canBePassed;
    }

    Road *getRoad() const;

    double getU() const;
    double getDu() const;
    double getYaw() const; // hinzugefügt Stefan: 24.07.12
    RoadTransition getRoadTransition() const;

    double getBoundingCircleRadius();

    int getLane() const;
    //std::set<int> getLaneSet() const;
    bool isOnLane(int) const;

    void brake();

    void addRouteTransition(const RoadTransition &);
    void setRepeatRoute(bool rr);

    VehicleGeometry *getVehicleGeometry();
    Transform getVehicleTransform();
    CarGeometry *getCarGeometry();

    const VehicleParameters &getVehicleParameters();
    void setVehicleParameters(const VehicleParameters &);
    VehicleState &getVehicleState()
    {
        return vehState;
    };
    ObstacleData::ObstacleTypeEnum getVehType(); //NEU 02-02-11
    //void setVehicleState(const VehicleState&);

protected:
    void init();

    bool laneChangeIsSafe(std::vector<ObstacleRelation> vehRelVec);

    std::vector<double> computeVehicleAccelerations(std::vector<ObstacleRelation> vehRelVec);
    double determineAcceleration(const std::map<int, std::vector<ObstacleRelation> > &, int);

    std::vector<ObstacleRelation> getNeighboursOnLane(int);

    std::set<int> getStaticLaneSet(double);

    RoadTransitionPoint getRoadTransitionPoint(double);

    void panicCantReachNextRoad();
    void vanish();

    bool findRoute(RoadTransition);
    bool planRoute();

    ObstacleRelation locateVehicle(int, int);
    double locateLaneEnd(int,bool resetIfLaneEnds=true);

    void extendSignalBarrierList(const RoadTransition &);

    double getAcceleration(const ObstacleRelation &, const double &, const double &, const double & = 0.0, const double & = 0.0) const;
    double etd(double, double) const; //effective target distance

    bool executeActionMap();

    VehicleParameters vehPars;
    VehicleState vehState;

    LaneSection *currentSection;
    int currentLane;

    double u, v, s; //u, v, s im Straßensystem
    double du, dv; //du und dv im Lokalsystem des Fahrzeugs
    double lcTime; //lange change time... Zeitpunkt, zu dem der Spurwechsel eingeleitet wurde bzw. die Entscheidung zum Spurwechsel gefällt wurde

    double hdgAlt; //Hilfsvariable, um das Heading vom vorherigen Zeitpunkt zu speichern
    std::deque<std::pair<double, double> > coordDeque; //Koordinaten-Deque mit den Koordinanten der letzten 2.5m --> für die Berechnung des Heading

    //-->für den Sinusspurwechsel
    double lcV; //v-Koordinate am Anfang des Spurwechsels
    double velo; //Hilftsvariable für die Berechnung der Geschwindigkeit
    // Neu Andreas 27-11-2012
    double currentSpeedLimit;
    double velDev; //Abweichung von Soll Geschwindigkeit in Prozent

    RoadTransitionList roadTransitionList;
    RoadTransitionList routeTransitionList;
    RoadTransitionList routeTransitionListBackup;
    std::list<RoadTransitionList::iterator> junctionPathLookoutList;
    RoadTransitionList::iterator currentTransition;
    Road::RoadType currentRoadType;
    VehicleActionMap vehiclePositionActionMap;
    VehicleActionMap vehicleTimerActionMap;

    std::list<std::pair<RoadTransition, RoadSignal *> > signalBarrierList;
    double signalBarrierTimer;

    CarGeometry *geometry;

    Transform vehicleTransform;

    double vel;
    double kv, dampingV, Tv;
    double hdg, dHdg, Khdg, Dhdg, Kv, Dv;
    double velt;
    double minTransitionListExtent;
    double boundRadius;
    double junctionWaitTime;
    double junctionWaitTriggerDist;
    double ms2kmh;
    bool sowedDetermineNextRoadVehicleAction;
    bool repeatRoute;

    double timer;

    ObstacleData::ObstacleTypeEnum vehType;

    // For cooperating with pedestrians near crosswalks
    Crosswalk *currentCrosswalk;
    int crossId;
    bool crossing;
    double crossPollTimer;
    bool canBePassed;

    std::set<Lane::LaneType> drivableLaneTypeSet; //Fahrbahntypen, auf denen der Fahrzeugagent fahren darf

    friend class VehicleAction;
    friend class DetermineNextRoadVehicleAction;
    friend class JunctionIndicatorVehicleAction;
};

class TRAFFICSIMULATIONEXPORT VehicleAction
{
public:
    virtual ~VehicleAction()
    {
    }

    virtual void operator()(AgentVehicle *veh)
    {
        std::cout << "Vehicle " << veh->name << ": Performing dummy action at: " << veh->s << std::endl;
    }

protected:
};

class TRAFFICSIMULATIONEXPORT DetermineNextRoadVehicleAction : public VehicleAction
{
public:
    DetermineNextRoadVehicleAction()
        : VehicleAction()
    {
    }

    void operator()(AgentVehicle *);

    static int removeAllActions(AgentVehicle *);

protected:
};

/** Vehicle Action for indicators at junctions.
* Action that turns on or off indicators at some junction ahead.
*/
class TRAFFICSIMULATIONEXPORT JunctionIndicatorVehicleAction : public VehicleAction
{
public:
    JunctionIndicatorVehicleAction(int indicator)
        : VehicleAction()
        , indicator_(indicator)
    {
    }

    void operator()(AgentVehicle *);
    static int removeAllActions(AgentVehicle *);

    int indicator_;

protected:
};
/** JunctionIndicatorVehicleAction */

#endif
