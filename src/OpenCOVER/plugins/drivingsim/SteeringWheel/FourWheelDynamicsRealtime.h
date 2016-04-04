/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FourWheelDynamicsRealtime_h
#define __FourWheelDynamicsRealtime_h

#include "gealg/CarDynamics.h"
//#include "../../../VehicleUtil/gealg/CarDynamicsPA2004.h"

#include "Vehicle.h"
#include "VehicleDynamics.h"
#ifdef __XENO__
#include "XenomaiTask.h"
#include "ValidateMotionPlatform.h"
#include "CanOpenController.h"
#include "XenomaiSteeringWheel.h"
#ifdef debug
#undef debug
#endif
#endif


#include <cover/coVRTui.h>

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>

#include "RoadSystem/RoadSystem.h"

//#include <deque>

#ifdef __XENO__
class FourWheelDynamicsRealtime : public VehicleDynamics, public XenomaiTask
#else
class FourWheelDynamicsRealtime : public VehicleDynamics
#endif
{
public:
    FourWheelDynamicsRealtime();
    ~FourWheelDynamicsRealtime();

    void move(VrmlNodeVehicle *vehicle);

    void initState();
    void resetState();

    double getVelocity()
    {
        return magnitude(cardyn::dp_b)(y_frame)[0];
    }
    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return rpms;
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

    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }
    virtual void setVehicleTransformation(const osg::Matrix &);

    virtual const osg::Matrix &getVehicleTransformation()
    {
        return chassisTrans;
    };

#ifdef __XENO__
    void platformToGround();
    void platformMiddleLift();
    void platformReturnToAction();
    void centerWheel();
#endif
    void setSportDamper(bool);

    double getSteerWheelAngle()
    {
        return steerWheelAngle;
    }

    std::pair<Road *, Vector2D> getStartPositionOnRoad();

protected:
    void run();

    void determineGroundPlane();

    osg::Matrix chassisTrans;

    osg::Quat wheelQuatFL;
    osg::Quat wheelQuatFR;
    osg::Quat wheelQuatRL;
    osg::Quat wheelQuatRR;

    cardyn::StateVectorType y;
    cardyn::ExpressionVectorType dy;
#ifdef __XENO__
    gealg::RungeKutta<cardyn::ExpressionVectorType, cardyn::StateVectorType, 17> integrator;
#endif

    gealg::mv<4, 0x0e0d0b07>::type groundPlane;
    //std::deque<gealg::mv<4, 0x0e0d0b07>::type> groundPlaneDeque;
    gealg::mv<1, 0x0f>::type i_proj;

    std::vector<gealg::mv<3, 0x040201>::type> r_i;
    std::vector<gealg::mv<3, 0x040201>::type> n_i;
    std::vector<gealg::mv<3, 0x040201>::type> r_n;
    std::vector<gealg::mv<3, 0x040201>::type> t_n;
    std::vector<gealg::mv<3, 0x040201>::type> r_o;
    std::vector<gealg::mv<3, 0x040201>::type> t_o;
    bool newIntersections;
    double hermite_dt;
    void determineHermite();

    std::vector<gealg::mv<3, 0x040201>::type> i_w;
    Road *currentRoad[4];
    double currentLongPos[4];
    gealg::mv<6, 0x060504030201LL>::type getRoadSystemContactPoint(const gealg::mv<3, 0x040201>::type &, Road *&, double &);

    unsigned long overruns;
#ifdef __XENO__
    static const RTIME period = 1000000;
#endif
    bool runTask;
    bool taskFinished;
    bool pause;

#ifdef __XENO__
    ValidateMotionPlatform *motPlat;
#endif
    bool returningToAction;
    bool movingToGround;
    bool doCenter;

#ifdef __XENO__
    CanOpenController *steerCon;
    XenomaiSteeringWheel *steerWheel;
#endif
    int32_t steerPosition;
    int32_t steerSpeed;
    double steerWheelAngle;
    double rpms;

    cardyn::StateVectorType y_frame;

    std::pair<Road *, Vector2D> startPos;
    bool leftRoad;

    coTUITab *vdTab;
    coTUILabel *k_Pp_Label;
    coTUILabel *d_Pp_Label;
    coTUILabel *k_Pq_Label;
    coTUILabel *d_Pq_Label;
    coTUILabel *k_wf_Label;
    coTUILabel *d_wf_Label;
    coTUILabel *k_wr_Label;
    coTUILabel *d_wr_Label;
    coTUISlider *k_Pp_Slider;
    coTUISlider *d_Pp_Slider;
    coTUISlider *k_Pq_Slider;
    coTUISlider *d_Pq_Slider;
    coTUIFloatSlider *k_wf_Slider;
    coTUIFloatSlider *d_wf_Slider;
    coTUIFloatSlider *k_wr_Slider;
    coTUIFloatSlider *d_wr_Slider;
};

#endif
