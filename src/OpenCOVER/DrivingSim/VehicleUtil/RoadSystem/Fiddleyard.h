/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Fiddleyard_h
#define Fiddleyard_h

#include "Tarmac.h"
#include "Road.h"
#include "Lane.h"
#include <util/coExport.h>

#include <map>

class Fiddleyard;
class PedFiddleyard;

class VEHICLEUTILEXPORT FiddleRoad : public Road
{
public:
    FiddleRoad(Fiddleyard *, std::string, std::string = "no name", double = 1000.0, Junction * = NULL, bool = false);

    std::string getTypeSpecifier();

    Fiddleyard *getFiddleyard();

    osg::Geode *getRoadGeode();

protected:
    Fiddleyard *fiddleyard;
    bool geometry;
};

class VEHICLEUTILEXPORT VehicleSource : public Element
{
public:
    VehicleSource(std::string id, int l, double startt, double repeatt, double vel = 30, double velDev = 5);

    bool spawnNextVehicle(double);

    void setFiddleroad(FiddleRoad *, TarmacConnection *);

    int getLane();
    int getFiddleLane() const;

    double getStartTime();
    double getRepeatTime();
    double getStartVelocity();
    double getStartVelocityDeviation();
    double getOverallRatio();

    void addVehicleRatio(std::string, double);
    bool isRatioVehicleMapEmpty();
    std::string getVehicleByRatio(double);

private:
    int startLane;
    int fiddleLane;
    double startTime;
    double nextTime;
    double repeatTime;
    double startVel;
    double startVelDev;
    double overallRatio;

    FiddleRoad *fiddleroad;

    std::map<double, std::string> ratioVehicleMap;
};

class VEHICLEUTILEXPORT VehicleSink : public Element
{
public:
    VehicleSink(std::string, int);

    void setFiddleroad(FiddleRoad *, TarmacConnection *);

    int getLane();
    int getFiddleLane() const;

protected:
    int endLane;
    int fiddleLane;

    FiddleRoad *fiddleroad;
};

class VEHICLEUTILEXPORT Fiddleyard : public Tarmac
{
public:
    Fiddleyard(std::string, std::string, FiddleRoad * = NULL);

    void setTarmacConnection(TarmacConnection *);
    TarmacConnection *getTarmacConnection();

    FiddleRoad *getFiddleroad() const;

    double incrementTimer(double);

    void addVehicleSource(VehicleSource *);
    void addVehicleSink(VehicleSink *);

    std::string getTypeSpecifier();

    const std::map<int, VehicleSource *> &getVehicleSourceMap() const;
    const std::map<int, VehicleSink *> &getVehicleSinkMap() const;

    void accept(RoadSystemVisitor *);

protected:
    TarmacConnection *connection;
    FiddleRoad *fiddleroad;
    double timer;
    bool active;

    std::map<int, VehicleSource *> sourceMap;
    std::map<int, VehicleSink *> sinkMap;
};

class VEHICLEUTILEXPORT PedSource : public VehicleSource
{
public:
    PedSource(PedFiddleyard *fy, std::string id, double start, double repeat, double dev, int l, int d, double so, double vo, double v, double vd, double a, double ad);

    double getStartTime()
    {
        return startTime;
    }
    double getRepeatTime()
    {
        return repeatTime;
    }
    double getTimeDev()
    {
        return timeDev;
    }
    int getDir()
    {
        return dir;
    }
    double getSOffset()
    {
        return sOffset;
    }
    double getVOffset()
    {
        return vOffset;
    }
    double getAccel()
    {
        return acc;
    }
    double getAccelDev()
    {
        return accDev;
    }

    bool spawnNextPed(double);
    void updateTimer(double, double);
    void addPedRatio(std::string, double);
    std::string getPedByRatio(double);
    bool isRatioMapEmpty();

private:
    PedFiddleyard *fiddleyard;

    double startTime;
    double repeatTime;
    double nextTime;
    double timeDev;
    int dir;
    double sOffset;
    double vOffset;
    double acc;
    double accDev;
};

class VEHICLEUTILEXPORT PedSink : public VehicleSink
{
public:
    PedSink(PedFiddleyard *fy, std::string id, double p, int l, int d, double so, double vo);

    double getProbability()
    {
        return prob;
    }
    int getDir()
    {
        return dir;
    }
    double getSOffset()
    {
        return sOffset;
    }
    double getVOffset()
    {
        return vOffset;
    }

private:
    PedFiddleyard *fiddleyard;

    double prob;
    int dir;
    double sOffset;
    double vOffset;
};

class VEHICLEUTILEXPORT PedFiddleyard : public Fiddleyard
{
public:
    PedFiddleyard(std::string id, std::string name, std::string r);

    std::string getRoadId()
    {
        return roadId;
    }

    void addSource(PedSource *);
    void addSink(PedSink *);

    const std::map<std::pair<int, double>, PedSource *> &getSourceMap() const;
    const std::map<std::pair<int, double>, PedSink *> &getSinkMap() const;

protected:
    std::map<std::pair<int, double>, PedSource *> pedSourceMap;
    std::map<std::pair<int, double>, PedSink *> pedSinkMap;

private:
    std::string roadId;
};

//Ergänzung für die beweglichen Fiddleyards

class VEHICLEUTILEXPORT Pool
{
public:
    Pool(std::string id, std::string name, double vel = 30, double velDev = 5, int direction = 1, double poolNumerator = 20);

    std::string getName();
    std::string getId();
    //double getRepeatTime();
    double getStartVelocity();
    double getStartVelocityDeviation();
    double getOverallRatio();
    int getMapSize();

    int getDir();
    void changeDir();
    void addVehicle(std::string, double);
    bool isRatioVehicleMapEmpty();
    std::string getVehicleByRatio(double);

private:
    //double startTime;
    //double nextTime;
    std::string name;
    std::string id;
    // double repeatTime;
    double startVel;
    double startVelDev;
    double overallRatio;
    int dir;
    double numerator;

    std::map<double, std::string> ratioVehicleMap;
};

class VEHICLEUTILEXPORT Carpool //: public Tarmac
{
public:
    static Carpool *Instance();
    static void Destroy();

    void addPool(Pool *pool);

    // double incrementTimer(double);

    //std::string getTypeSpecifier();

    std::vector<Pool *> getPoolVector();

    double getOverallRatio();
    void addPoolRatio(std::string, double);
    bool isRatioPoolMapEmpty();
    std::string getPoolIdByRatio(double);

    Pool *getPoolById(std::string id);

    //   void accept(RoadSystemVisitor*);

protected:
    Carpool();
    //Carpool(std::string);
    //TarmacConnection* connection;
    //FiddleRoad* fiddleroad;
    //double timer;
    bool active;

    std::vector<Pool *> poolVector;

private:
    static Carpool *__instance;
    double overallRatio;
    std::map<double, std::string> ratioPoolMap;
    //std::string name;
};
#endif