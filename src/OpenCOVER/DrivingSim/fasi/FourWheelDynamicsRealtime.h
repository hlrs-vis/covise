/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FourWheelDynamicsRealtime_h
#define __FourWheelDynamicsRealtime_h

#include "../VehicleUtil/gealg/CarDynamics.h"
#include "XenomaiTask.h"
#include "ValidateMotionPlatform.h"
#include "CanOpenController.h"
#include "XenomaiSteeringWheel.h"

#include "RoadSystem/RoadSystem.h"
#include "KLSM.h"

class FourWheelDynamicsRealtime : public XenomaiTask
{
public:
    FourWheelDynamicsRealtime();
    ~FourWheelDynamicsRealtime();

    void move();
    void initState();
    void resetState();

    double getVelocity()
    {
        return magnitude(cardyn::dp_b)(y_frame)[0];
    }

    virtual double getEngineTorque()
    {
        return cardyn::Me_e(y)[0];
    }
    virtual double getTyreSlipFL()
    {
        return magnitude(part<2, 0x0201>(cardyn::dr_wfl) * 1000.0)(y)[0];
    }
    virtual double getTyreSlipFR()
    {
        return magnitude(part<2, 0x0201>(cardyn::dr_wfr) - cardyn::x * cardyn::w_wfr * cardyn::r_w)(y)[0];
    }
    virtual double getTyreSlipRL()
    {
        return magnitude(part<2, 0x0201>(cardyn::dr_wrl) - cardyn::x * cardyn::w_wrl * cardyn::r_w)(y)[0];
    }
    virtual double getTyreSlipRR()
    {
        return magnitude(part<2, 0x0201>(cardyn::dr_wrr) - cardyn::x * cardyn::w_wrr * cardyn::r_w)(y)[0];
    }

    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return rpms;
    }

    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }
    void setVehicleTransformation(const osg::Matrix &m);

    void platformToGround();
    void platformMiddleLift();
    void platformReturnToAction();
    void centerWheel();
    void setSportDamper(bool);

    double getSteerWheelAngle()
    {
        return steerWheelAngle;
    }

    std::pair<Road *, Vector2D> getStartPositionOnRoad();
    virtual const osg::Matrix &getVehicleTransformation()
    {
        return chassisTrans;
    };

protected:
    void run();

    void determineGroundPlane();
    bool newIntersections;
    double hermite_dt;
    void determineHermite();

    unsigned long overruns;
    static const RTIME period = 1000000;
    bool runTask;
    bool taskFinished;
    bool pause;

    ValidateMotionPlatform *motPlat;
    bool returningToAction;
    bool movingToGround;
    bool doCenter;

    CanOpenController *steerCon;
    XenomaiSteeringWheel *steerWheel;
    int32_t steerPosition;
    int32_t steerSpeed;
    double steerWheelAngle;
    double rpms;

    osg::Matrix chassisTrans;

    osg::Quat wheelQuatFL;
    osg::Quat wheelQuatFR;
    osg::Quat wheelQuatRL;
    osg::Quat wheelQuatRR;

    cardyn::StateVectorType y;
    cardyn::ExpressionVectorType dy;
    gealg::RungeKutta<cardyn::ExpressionVectorType, cardyn::StateVectorType, 17> integrator;

    //gealg::mv<1, 0x0f>::type i_proj;

    std::vector<gealg::mv<3, 0x040201>::type> r_i;
    std::vector<gealg::mv<3, 0x040201>::type> n_i;
    std::vector<gealg::mv<3, 0x040201>::type> r_n;
    std::vector<gealg::mv<3, 0x040201>::type> t_n;
    std::vector<gealg::mv<3, 0x040201>::type> r_o;
    std::vector<gealg::mv<3, 0x040201>::type> t_o;
    std::vector<gealg::mv<3, 0x040201>::type> i_w;
    cardyn::StateVectorType y_frame;
    gealg::mv<6, 0x060504030201LL>::type getRoadSystemContactPoint(const gealg::mv<3, 0x040201>::type &, Road *&, double &);

    Road *currentRoad[4];
    double currentLongPos[4];

    std::pair<Road *, Vector2D> startPos;
    bool leftRoad;
    double k_wf_Slider;
    double d_wf_Slider;
    double k_wr_Slider;
    double d_wr_Slider;
    double clutchPedal;
};
#endif
