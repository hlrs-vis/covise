/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FourWheelDynamics_h
#define __FourWheelDynamics_h

#include "../../../VehicleUtil/gealg/CarDynamics.h"

#include "Vehicle.h"
#include "VehicleDynamics.h"

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>

class FourWheelDynamics : public VehicleDynamics, public OpenThreads::Thread
{
public:
    FourWheelDynamics();
    ~FourWheelDynamics();

    void move(VrmlNodeVehicle *vehicle);

    void resetState(){};

    double getVelocity()
    {
        return magnitude(cardyn::dp_b)(y)[0];
    }
    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return ((cardyn::w_wrl + cardyn::w_wrr) * (0.5))(y)[0];
    }

    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }
    virtual void setVehicleTransformation(const osg::Matrix &){};

    virtual const osg::Matrix &getVehicleTransformation()
    {
        return chassisTrans;
    };

    void run();

protected:
    osg::Matrix chassisTrans;

    osg::Quat wheelQuatFL;
    osg::Quat wheelQuatFR;
    osg::Quat wheelQuatRL;
    osg::Quat wheelQuatRR;

    cardyn::StateVectorType y;
    cardyn::ExpressionVectorType dy;
    gealg::RungeKutta<cardyn::ExpressionVectorType, cardyn::StateVectorType, 17> integrator;

    gealg::mv<4, 0x0e0d0b07>::type groundPlane;
    gealg::mv<1, 0x0f>::type i_proj;
    void determineGroundPlane();

    std::vector<gealg::mv<3, 0x040201>::type> r_i;
    std::vector<gealg::mv<3, 0x040201>::type> n_i;
    bool newIntersections;
    double hermite_dt;
    void determineHermite();

    osg::PositionAttitudeTransform *wheelTransformFL;
    osg::PositionAttitudeTransform *wheelTransformFR;
    osg::PositionAttitudeTransform *wheelTransformRL;
    osg::PositionAttitudeTransform *wheelTransformRR;

    OpenThreads::Barrier endBarrier;
    bool doRun;
};

#endif
