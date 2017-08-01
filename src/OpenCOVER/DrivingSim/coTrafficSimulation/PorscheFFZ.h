/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PORSCHEFFZ_H
#define PORSCHEFFZ_H

// Base Class //
//
#include <OpenThreads/Thread>

// Includes //
//
#include <OpenThreads/Barrier>
#include <string>
#include "Vehicle.h"

//#include "../SteeringWheel/PorscheRealtimeDynamics.h"

// Forward Declarations //
class UDPBroadcast;
class RadarCones;

// Receive Data Block //
//
struct TRAFFICSIMULATIONEXPORT RadarConeData
{
    RadarConeData(
        float instPosX = 0.0,
        float instPosY = 0.0,
        float instPosZ = 0.0,
        float instRoll = 0.0,
        float instPitch = 0.0,
        float instYaw = 0.0,
        float minR = 0.0,
        float maxR = 0.0,
        float azim = 0.0,
        float elev = 0.0,
        int col = 0)
        : installPosX(instPosX)
        , installPosY(instPosY)
        , installPosZ(instPosZ)
        , installRoll(instRoll)
        , installPitch(instPitch)
        , installYaw(instYaw)
        , minRange(minR)
        , maxRange(maxR)
        , azimuth(azim)
        , elevation(elev)
        , color(col)
    {
    }

    float installPosX;
    float installPosY;
    float installPosZ;
    float installRoll;
    float installPitch;
    float installYaw;
    float minRange;
    float maxRange;
    float azimuth;
    float elevation;
    int color;
};

struct TRAFFICSIMULATIONEXPORT RadarConesData
{
    RadarConesData(unsigned short n = 0)
        : messageID(513)
        , nSensors(n)
    {
    }

    int messageID;
    int nSensors;
    RadarConeData cones[8];
};

// Send Data Block //
//
struct TRAFFICSIMULATIONEXPORT ObstacleData
{
    enum ObstacleTypeEnum
    {
        OBSTACLE_UNKNOWN = 0,
        OBSTACLE_HUMAN = 1,
        OBSTACLE_CAR = 2,
        OBSTACLE_VAN = 3,
        OBSTACLE_TRUCK = 4,
        OBSTACLE_TRACTOR = 5,
        OBSTACLE_POLICE = 6,
        OBSTACLE_BICYCLE = 7,
        OBSTACLE_PEDESTRIAN = 8,
        OBSTACLE_MOTORCYCLE = 9,
        OBSTACLE_SPORTSCAR = 10,
        OBSTACLE_SUV = 11,
        OBSTACLE_SPECIAL = 12,
        OBSTACLE_SMALLCAR = 13
    };
    typedef enum ObstacleTypeEnum ObstacleType;

    ObstacleData(
        int id = 0,
        ObstacleType obType = OBSTACLE_UNKNOWN,
        float dX = 0.0f, float dY = 0.0f, float dZ = 0.0f,
        float sX = 0.0f, float sY = 0.0f, float sZ = 0.0f,
        float dRoll = 0.0f, float dPitch = 0.0f, float dYaw = 0.0f,
        float spd = 0.0f /*, float dAcceleration = 0.0f*/)
        : deltaX(dX)
        , deltaY(dY)
        , deltaZ(dZ)
        , sizeX(sX)
        , sizeY(sY)
        , sizeZ(sZ)
        , deltaRoll(dRoll)
        , deltaPitch(dPitch)
        , deltaYaw(dYaw)
        , speed(spd) /*,
		deltaAcceleration(dAcceleration)*/
    {
        obstacleID = (unsigned short)(id);
        obstacleType = (unsigned short)(obType);
    }
    unsigned short obstacleID; // unique ID of the obstacle
    unsigned short obstacleType; // type of the obstacle
    float deltaX; // rel. to HumanDriver center
    float deltaY;
    float deltaZ;
    float sizeX;
    float sizeY;
    float sizeZ;
    float deltaRoll;
    float deltaPitch;
    float deltaYaw;
    float speed;
    // 	float deltaAcceleration;
};

struct TRAFFICSIMULATIONEXPORT VehicleBroadcastOutData
{
    // message to be broadcasted //
    enum WeatherTypeEnum
    {
        WEATHER_UNKNOWN = 0,
        WEATHER_SUNNY = 1,
        WEATHER_CLOUDY = 2,
        WEATHER_RAINY = 3,
        WEATHER_SNOWY = 4
    };
    typedef enum WeatherTypeEnum WeatherType;

    VehicleBroadcastOutData(unsigned short nOb = 0, WeatherType wType = WEATHER_UNKNOWN, float visRange = 0.0f)
        : messageID(512)
        , nObstacles(nOb)
        , visibilityRange(visRange)
    {
        weatherType = (unsigned short)(wType);
    }
    int messageID;
    unsigned short nObstacles; // number of obstacles (vehicles, people, motorcyles,...) 0-65535
    unsigned short weatherType; // ID of the weather type
    float visibilityRange; // [m]
    unsigned int time; // simulation time [ms] since creation of this object
    //ObstacleData obstacle[32];
    ObstacleData obstacle[10];
};

// Main Class //
//
class TRAFFICSIMULATIONEXPORT PorscheFFZ : public OpenThreads::Thread
{
public:
    PorscheFFZ(double sendFrequency);
    ~PorscheFFZ();

    void setupDSPACE(std::string destinationIP, int port, int localPort);
    void setupKMS(std::string destinationIP, int port, int localPort);

    void sendData(const VehicleList &vehicleList);
    void receiveData();

    double getSendFrequency()
    {
        return sendFrequency_;
    }

private:
    PorscheFFZ(){ /* not allowed */ };

    // Settings //
    //
    double sendFrequency_; // how often should the list be sent
    double startTime_;

    // dSPACE //
    //
    UDPBroadcast *dSpaceComm_;
    std::string dSpaceIp_;
    int dSpacePort_;
    int dSpaceLocalPort_;

    // KMS //
    //
    UDPBroadcast *kmsComm_;
    std::string kmsIp_;
    int kmsPort_;
    int kmsLocalPort_;

    // Receive Thread //
    //
    OpenThreads::Barrier endBarrier;

    void run();
    bool readData();
    bool doRun_;

    RadarConesData *radarConesData_;
    RadarConesData *incomingRadarConesData_;

    // Projects //
    //
    // Porsche Radar Cones
    RadarCones *cones_;
};

#endif
