/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Pedestrian_h
#define Pedestrian_h

#include "RoadSystem/Crosswalk.h"
#include "RoadSystem/LaneSection.h"
#include "RoadSystem/Junction.h"
#include "RoadSystem/Road.h"
#include <string>
#include <stdarg.h>

struct TRAFFICSIMULATIONEXPORT PedestrianParameters
{
    PedestrianParameters(double _headingAdj = 0.0, double _duMax = 1.5, double _dduMax = 3.0, double _duWait = 1.5, double _dduWait = 3.0, double _dvPass = 1.0, double _ddvPass = 1.0, double _dvMax = 0.2, double _ddvMax = 0.2)
        : headingAdj(_headingAdj)
        , duMax(_duMax)
        , dduMax(_dduMax)
        , duWait(_duWait)
        , dduWait(_dduWait)
        , dvPass(_dvPass)
        , ddvPass(_ddvPass)
        , dvMax(_dvMax)
        , ddvMax(_ddvMax)
    {
    }

    double headingAdj; // Heading adjustment
    double duMax; // Speed when walking/passing
    double dduMax; // Acceleration when walking/passing
    double duWait; // Maximal speed when waiting for slower neighbor to finish a pass
    double dduWait; // Acceleration when waiting for slower neighbor
    double dvPass; // Transversal speed when passing/avoiding
    double ddvPass; // Transversal acceleration when passing/avoiding
    double dvMax; // Transversal speed
    double ddvMax; // Transversal acceleration
};

struct TRAFFICSIMULATIONEXPORT PedestrianLocation
{
    PedestrianLocation(Road *_r = NULL, int _ln = 0, Lane *_l = NULL, LaneSection *_ls = NULL, Crosswalk *_cw = NULL, int _d = 0)
        : road(_r)
        , laneNum(_ln)
        , lane(_l)
        , laneSec(_ls)
        , crosswalk(_cw)
        , dir(_d)
    {
    }

    Road *road; // Current road
    int laneNum; // Lane number on current road
    Lane *lane; // Pointer to current lane
    LaneSection *laneSec; // Pointer to current lane section
    Crosswalk *crosswalk; // Pointer to current crosswalk
    int dir; // Current direction on road (positive towards higher s-values)
};

struct TRAFFICSIMULATIONEXPORT PedestrianState
{
    PedestrianState(bool _act = false, double _u = 0.0, double _v = 0.0, double _s = 0.0, double _dus = 0.0, double _dut = 0.0, double _du = 0.0, double _dv = 0.0, double _vs = 0.0, double _vt = 0.0, double _vo = 0.0, double _va = 0.0, double _ddu = 0.0, double _ddv = 0.0, double _hdg = 0.0, bool _or = false, bool _c = false, int _cid = Crosswalk::DONOTENTER, int _cwid = Crosswalk::DONOTENTER, double _cv = 0.0, double _cw = 0.0, int _cp = 0, bool _cd = false, bool _srch = false, bool _fnd = false, bool _ls = false, bool _ld = false, std::string _sink = "")
        : active(_act)
        , u(_u)
        , v(_v)
        , s(_s)
        , duStart(_dus)
        , duTgt(_dut)
        , du(_du)
        , dv(_dv)
        , vStart(_vs)
        , vTgt(_vt)
        , vOff(_vo)
        , vAccelDist(_va)
        , ddu(_ddu)
        , ddv(_ddv)
        , hdg(_hdg)
        , onRoad(_or)
        , cross(_c)
        , crossId(_cid)
        , crossWaitId(_cwid)
        , crossVel(_cv)
        , crossWaitTimer(_cw)
        , crossProg(_cp)
        , crossDone(_cd)
        , searchSidewalk(_srch)
        , foundSidewalk(_fnd)
        , leavingSidewalk(_ls)
        , leavingDone(_ld)
        , passingSink(_sink)
    {
    }

    bool active; // true if pedestrian is active
    double u; // longitudinal position
    double v; // transversal position
    double s; // total distance traveled
    double duStart; // starting longitudinal speed
    double duTgt; // longitudinal speed target
    double du; // longitudinal speed
    double dv; // transversal speed
    double vStart; // starting point of latest transversal motion
    double vTgt; // target transversal offset
    double vOff; // transversal offset
    double vAccelDist; // distance to accelerate during last transversal motion
    double ddu; // longitudinal acceleration
    double ddv; // lateral acceleration
    double hdg; // heading
    bool onRoad; // true if pedestrian is currently on a road; otherwise, false
    bool cross; // true if the pedestrian is crossing driving lanes via a crosswalk
    int crossId; // pedestrian's id in the crosswalk (needed when exiting)
    int crossWaitId; // pedestrian's id when waiting to cross the crosswalk (needed when done waiting)
    double crossVel; // safe velocity at which to cross the road (depends on max forward vel and current sidewalk widths)
    double crossWaitTimer; // timer to control the pedestrian's wait before entering the crosswalk
    int crossProg; // lane number that updates as pedestrian crosses a crosswalk lane section
    bool crossDone; // true if pedestrian has walked across the crosswalk and is returning to the sidewalk
    bool searchSidewalk; // true if pedestrian is searching horizontally for a sidewalk
    bool foundSidewalk; // true if pedestrian is searching and has reached the sidewalk
    bool leavingSidewalk; // true if pedestrian is leaving the sidewalk and walking to the vOffset
    bool leavingDone; // true if pedestrian has left the sidewalk
    std::string passingSink;
};

class PedestrianFactory;
class PedestrianGeometry;
class PedestrianManager;
class TRAFFICSIMULATIONEXPORT Pedestrian
{
public:
    Pedestrian(std::string &name, Road *startRoad = NULL, int startLaneNum = 0, int startDir = 0, double startPos = 0.0, double startVOff = 0.0, double startVel = 0.0, double startAcc = 0.0, double headingAdj = 0.0, int dbg = 0, PedestrianGeometry *g = NULL);
    ~Pedestrian();

    void updateActive();
    bool setActive(bool act);
    bool isActive() const
    {
        return pedState.active;
    }

    bool isWithinRange(const double r) const;

    void setName(const std::string &n)
    {
        name = n;
    }
    const char *getName() const
    {
        return name.c_str();
    }
    int getDebugLvl() const
    {
        return debugLvl;
    }
    void setPedestrianGeometry(PedestrianGeometry *g)
    {
        geometry = g;
    }
    PedestrianGeometry *getPedestrianGeometry() const
    {
        return geometry;
    }
    Road *getRoad() const
    {
        return pedLoc.road;
    }
    bool isOnRoad() const
    {
        return pedState.onRoad;
    }
    Lane *getLane() const
    {
        return pedLoc.lane;
    }
    int getLaneNum() const
    {
        return pedLoc.laneNum;
    }
    int getDir() const
    {
        return pedLoc.dir;
    }
    double getU() const
    {
        return pedState.u;
    }
    double getDu() const
    {
        return pedState.du;
    }
    double getDuMax() const
    {
        return pedParams.duMax;
    }
    double getVOffset() const
    {
        return pedState.vOff;
    }
    Pedestrian *getAvoiding() const
    {
        return avoiding;
    }
    Pedestrian *getPassing() const
    {
        return passing;
    }
    Pedestrian *getWaitingFor() const
    {
        return waitingFor;
    }
    Pedestrian *getPassedBy() const
    {
        return passedBy;
    }
    bool isDoneLeavingSidewalk() const
    {
        return pedState.leavingSidewalk && pedState.leavingDone;
    }
    bool isLeavingSidewalk() const
    {
        return pedState.leavingSidewalk;
    }

    void passSink(const std::string id)
    {
        pedState.passingSink = id;
    }
    std::string passSink() const
    {
        return pedState.passingSink;
    }

    void onRoad(const bool r);

    double getWalkwayWidth() const;
    double getLaneWidth(LaneSection *ls, const int ln, const double s = 0.0) const;

    void leaveSidewalk(const double v);

    void setDuMax(const double du, const double ddu);
    void setDuWait(const double du);

    void setVTarget(const double vTarget);
    void setDvPass(const double dv);
    void setDvMax(const double dv);

    void setAvoiding(Pedestrian *p, const double time);
    void setPassing(Pedestrian *p, const double time);
    void setWaitingFor(Pedestrian *p, const double time);
    void setPassedBy(Pedestrian *p, const double time);

    void move(const double dt);

    void executeWave();
    void executeAction(const int idx);

    static bool comparePedestrians(const Pedestrian *a, const Pedestrian *b);

protected:
    int debugLvl;
    std::string name;
    PedestrianGeometry *geometry;
    double timer;

    PedestrianParameters pedParams; // Parameters of movement (e.g., velocity limits)
    PedestrianLocation pedLoc; // Location in road system (e.g., road, lane)
    PedestrianState pedState; // State at current location (e.g., position, velocity)
    Transform pedTrans; // Current geometry transformation

    Pedestrian *avoiding; // Fellow pedestrian currently being avoided (none if NULL)
    Pedestrian *passing; // Fellow pedestrian currently being passed (none if NULL)
    Pedestrian *waitingFor; // Fellow pedestrian currently being waited for (none if NULL)
    Pedestrian *passedBy; // Fellow pedestrian currently being passed by (none if NULL)

private:
    void setNewU(const double dt);
    void setNewV(const double dt);
    void setNewHdg();
    void checkForNewRoad();
    void checkForNewLaneSection(const double dt);
    void checkForNewCrosswalk();
    void crossStreet(const double dt);
    void findSidewalk(const double dt);
    void leavingSidewalk(const double dt);
    void updateGeometry();

    void printDebug(int lvl, const char *msg, ...);
};

/**
 * Compare the two pedestrians; return true if pedestrian a's position is less than pedestrian b's position
 */
inline bool Pedestrian::comparePedestrians(const Pedestrian *a, const Pedestrian *b)
{
    return (a->getU() < b->getU());
}

#endif
