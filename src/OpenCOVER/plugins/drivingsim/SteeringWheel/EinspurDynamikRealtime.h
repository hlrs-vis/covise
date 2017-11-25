/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __EINSPUR_DYNAMIK_REALTIME_H
#define __EINSPUR_DYNAMIK_REALTIME_H

#include <TrafficSimulation/Vehicle.h>
#include "VehicleDynamics.h"
#include <VehicleUtil/ValidateMotionPlatform.h>
#include <VehicleUtil/CanOpenController.h>
#include <VehicleUtil/XenomaiTask.h>
#include <VehicleUtil/XenomaiSteeringWheel.h>

class EinspurDynamikRealtimeState
{
public:
    double x;
    double y;
    double kappa;
    double v;
    double alpha;

    double u;

    double epsilon;
    double dEpsilon;

    double zeta;
    double dZeta;

    EinspurDynamikRealtimeState operator+(const EinspurDynamikRealtimeState &state) const;
    EinspurDynamikRealtimeState operator-(const EinspurDynamikRealtimeState &state) const;
    EinspurDynamikRealtimeState &operator=(const EinspurDynamikRealtimeState &state);
    EinspurDynamikRealtimeState operator*(const double scalar) const;
};

class EinspurDynamikRealtimeDState
{
public:
    double dX;
    double dY;
    double dKappa;
    double dV;
    double dAlpha;

    double dU;

    double dEpsilon;
    double ddEpsilon;

    double dZeta;
    double ddZeta;

    EinspurDynamikRealtimeDState operator+(const EinspurDynamikRealtimeDState &state) const;
    EinspurDynamikRealtimeDState &operator=(const EinspurDynamikRealtimeDState &state);

    EinspurDynamikRealtimeState operator*(const double scalar) const;

    //EinspurDynamikRealtimeDState operator*(const double scalar) const;
};

class EinspurDynamikRealtime : public VehicleDynamics, public XenomaiTask
{
public:
    EinspurDynamikRealtime();
    ~EinspurDynamikRealtime();

    double getSteerWheelAngle();
    double getAlpha();
    double getX();
    double getY();
    double getVelocity();
    double getAcceleration();
    double getEpsilon();
    double getDEpsilon();
    double getDDEpsilon();
    double getZeta();
    double getDZeta();
    double getDDZeta();
    double getEngineSpeed();
    const osg::Matrix &getVehicleTransformation();
    virtual void setVehicleTransformation(const osg::Matrix &);
    virtual void setRoadType(int);

    void platformToGround();
    void platformReturnToAction();

    void move(VrmlNodeVehicle *vehicle);

    void resetState();

private:
    void setFactors();
    void setInitialState(double x, double y, double v, double alpha);

    void run();
    void integrate(double h, double x, double y, double alpha);

    EinspurDynamikRealtimeDState dFunction(EinspurDynamikRealtimeState y);

    double norm(const EinspurDynamikRealtimeState &state);

    unsigned long overruns;
    static const RTIME period = 1000000;
    bool runTask;
    bool taskFinished;

    double beta; // Wheel angle
    double gamma; // Angle acceleration vector and roll axis (Mass point)
    double gammav; // Angle acceleration vector and roll axis (Middle of Vehicle)

    double a; // Acceleration

    double s; // Distance velocity pole (Momentanpol) to rear axes
    double r; // Distance velocity pole (Momentanpol) to vehicle mass point
    double rv; // Distance velocity pole (Momentanpol) to vehicle middle point
    double q; // Distance velocity pole (Momentanpol) to front axes

    double m; // Vehicle mass
    double I; // Vehicle inertia
    double lv; // Vehicle distance front to mass point
    double lh; // Vehicle distance rear to mass point

    double roadFrictionForce;

    double b;
    double h;
    double Aw;
    double cw;

    double Iw;
    double hs;
    double kw;
    double dw;

    double In;
    double kn;
    double dn;

    double wr; // Wheel radius

    double Im; // Motor moment of inertia
    double ach; // Motor characteristic parameters
    double bch;
    double cch;
    double ul; // Engine speed limit
    double ui; // Engine idle speed
    double accPedalIdle;

    double iag; // Axle gearbox
    double i[8]; // Vehicle gearbox

    double cc;
    double kc;

    double cm;
    double km;

    EinspurDynamikRealtimeState yNull; // Initial value
    EinspurDynamikRealtimeState yOne; // Integration result
    EinspurDynamikRealtimeDState dYOne; // Integration dState

    double errorcontrol;
    int num_intsteps;

    int stepcount;

    osg::Matrix chassisTrans;
    osg::Matrix bodyTrans;

    double accPedal;
    double brakePedal;
    double clutchPedal;
    int gear;
    double steerWheelAngle;

    ValidateMotionPlatform *motPlat;
    bool returningToAction;
    bool movingToGround;
    bool pause;

    CanOpenController *steerCon;
    XenomaiSteeringWheel *steerWheel;
    int32_t steerPosition;
    int32_t steerSpeed;
};

inline double EinspurDynamikRealtime::getSteerWheelAngle()
{
    return steerWheelAngle;
}

inline double EinspurDynamikRealtime::getAlpha()
{
    return yOne.alpha;
}

inline double EinspurDynamikRealtime::getX()
{
    return yOne.x;
}

inline double EinspurDynamikRealtime::getY()
{
    return yOne.y;
}

inline double EinspurDynamikRealtime::getVelocity()
{
    return yOne.v;
}
inline double EinspurDynamikRealtime::getAcceleration()
{
    return a;
}

inline double EinspurDynamikRealtime::getEpsilon()
{
    return yOne.epsilon;
}
inline double EinspurDynamikRealtime::getDEpsilon()
{
    return yOne.dEpsilon;
}
inline double EinspurDynamikRealtime::getDDEpsilon()
{
    return dYOne.ddEpsilon;
}

inline double EinspurDynamikRealtime::getZeta()
{
    return yOne.zeta;
}
inline double EinspurDynamikRealtime::getDZeta()
{
    return yOne.dZeta;
}
inline double EinspurDynamikRealtime::getDDZeta()
{
    return dYOne.ddZeta;
}
inline const osg::Matrix &EinspurDynamikRealtime::getVehicleTransformation()
{
    return chassisTrans;
}

#endif
