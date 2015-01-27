/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CAR_DYNAMICSREALTIME_H
#define __CAR_DYNAMICSREALTIME_H

#include "gaalet.h"
#include "MagicFormula2004.h"
#include "RungeKuttaClassic.h"
#include <tuple>

#include "Vehicle.h"
#include "VehicleDynamics.h"

#include "XenomaiTask.h"
#include "ValidateMotionPlatform.h"
#include "CanOpenController.h"
#include "XenomaiSteeringWheel.h"

#include "RoadSystem/RoadSystem.h"

#include "CarDynamicsCGA.h"

//class PLUGINEXPORT CarDynamicsCGARealtime
class CarDynamicsCGARealtime : public VehicleDynamics, public XenomaiTask
{
public:
    typedef gaalet::algebra<gaalet::signature<3, 0> > ega;

    CarDynamicsCGARealtime();
    virtual ~CarDynamicsCGARealtime();

    virtual void move(VrmlNodeVehicle *vehicle);

    virtual void resetState();

    virtual double getVelocity()
    {
        return 0.0;
    }
    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return 0.0;
    }
    virtual double getEngineTorque()
    {
        return 0.0;
    }
    virtual double getTyreSlipFL()
    {
        return 0.0;
    }
    virtual double getTyreSlipFR()
    {
        return 0.0;
    }
    virtual double getTyreSlipRL()
    {
        return 0.0;
    }
    virtual double getTyreSlipRR()
    {
        return 0.0;
    }

    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }

    virtual const osg::Matrix &getVehicleTransformation()
    {
        chassisTrans;
    };

    virtual void setVehicleTransformation(const osg::Matrix &);

    cardyncga::Plane getRoadTangentPlane(Road *&road, Vector2D v_c);
    void getRoadSystemContactPoint(const cardyncga::Point &p_w, Road *&road, double &u, cardyncga::Plane &s_c);
    void getFirstRoadSystemContactPoint(const cardyncga::Point &p_w, Road *&road, double &u, cardyncga::Plane &s_c);

    std::pair<Road *, Vector2D> getStartPositionOnRoad();

    //xenomai
    void platformToGround();
    void platformReturnToAction();
    void centerSteeringWheel()
    {
        centerSteeringWheelOnNextRun = true;
    }

protected:
    void run();

    cardyncga::InputVector z;
    cardyncga::OutputVector o;
    cardyncga::StateVector y;

    magicformula2004::TyrePropertyPack tyrePropLeft;
    magicformula2004::TyrePropertyPack tyrePropRight;
    magicformula2004::ContactWrench tyreFL;
    magicformula2004::ContactWrench tyreFR;
    magicformula2004::ContactWrench tyreRL;
    magicformula2004::ContactWrench tyreRR;
    cardyncga::StateEquation f;

    Road *road_wheel_fl, *road_wheel_fr, *road_wheel_rl, *road_wheel_rr;
    double u_wheel_fl, u_wheel_fr, u_wheel_rl, u_wheel_rr;

    //EulerIntegrator<cardyncga::StateEquation, cardyncga::StateVector> integrator;
    RungeKuttaClassicIntegrator<cardyncga::StateEquation, cardyncga::StateVector> integrator;

    osg::Matrix chassisTrans;

    bool firstMoveCall;

    std::pair<Road *, Vector2D> startPos;

    //xenomai
    static const RTIME period = 1000000;
    bool runTask;
    bool taskFinished;
    unsigned long overruns;
    enum HapticSimulationState
    {
        PAUSING = 0,
        DRIVING = 1,
        PLATFORM_RAISING = 2,
        PLATFORM_LOWERING = 3
    } hapSimState;
    bool centerSteeringWheelOnNextRun;

    ValidateMotionPlatform *motPlat;
    CanOpenController *steerCon;
    XenomaiSteeringWheel *steerWheel;

    void checkHapticSimulationState();
};

#endif
