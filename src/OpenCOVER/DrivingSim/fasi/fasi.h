/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FASI_INCLUDE
#define FASI_INCLUDE
#include <FourWheelDynamicsRealtime.h>
#include <list>
#include "XenomaiSteeringWheel.h"
#include "GasPedal.h"
#include "KI.h"
#include "KLSM.h"
#include "Klima.h"
#include "Beckhoff.h"
#include "IgnitionLock.h"
#include <sys/time.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <net/covise_connect.h>

class fasi
{
public:
    fasi(const char *filename);
    ~fasi();
    struct RemoteData
    {
        float V;
        float A;
        float rpm;
        float torque;
        osg::Matrix chassisTransform;
        int buttonStates;
        int gear;
    };

    KI *p_kombi;
    KLSM *p_klsm;
    Klima *p_klima;
    VehicleUtil *vehicleUtil;
    Beckhoff *p_beckhoff;
    GasPedal *p_gaspedal;
    IgnitionLock *p_ignitionLock;
    RemoteData remoteData;

    struct
    {
        float steeringWheelAngle;
        float pedalA;
        float pedalB;
        float pedalC;
        int gear;
        bool SportMode;
        bool PSMState;
        bool SpoilerState;
        bool DamperState;
        double frameTime;
    } sharedState;

    bool oldFanButtonState;
    bool oldParkState;
    bool automatic;
    fasiUpdateManager *fum;
    FourWheelDynamicsRealtime *vehicleDynamics;

    std::string xodrDirectory;
    RoadSystem *system;
    xercesc::DOMElement *rootElement;

    void run();
    int getAutoGearDiff(float downRPM = 50, float upRPM = 100);
    static fasi *myFasi;
    static fasi *instance()
    {
        return myFasi;
    };
    bool loadRoadSystem(const char *filename_chars);
    void parseOpenDrive(xercesc::DOMElement *rootElement);
    xercesc::DOMElement *getOpenDriveRootElement(std::string filename);
    covise::ServerConnection *serverConn;
    covise::SimpleServerConnection *toClientConn;
    bool readClientVal(void *buf, unsigned int numBytes);
};
#endif
