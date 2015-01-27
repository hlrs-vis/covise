/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CarDynamicsRtus.h"
#include <iostream>
#include <cmath>

#include "SteeringWheel.h"

namespace cardyn_rtus
{
cm::mv<0x01>::type e1 = { 1.0 };
cm::mv<0x02>::type e2 = { 1.0 };
cm::mv<0x04>::type e3 = { 1.0 };
cm::mv<0x08>::type ep = { 1.0 };
cm::mv<0x10>::type em = { 1.0 };

cm::mv<0x00>::type one = { 1.0 };

cm::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
cm::mv<0x08, 0x10>::type einf = em + ep;

cm::mv<0x18>::type E = ep * em;

cm::mv<0x1f>::type Ic = e1 * e2 * e3 * ep * em;
cm::mv<0x07>::type Ie = e1 * e2 * e3;
}

CarDynamicsRtus::CarDynamicsRtus()
    : tyrePropLeft(magicformula2004::TyrePropertyPack::TYRE_LEFT)
    , tyrePropRight(magicformula2004::TyrePropertyPack::TYRE_RIGHT)
    , tyreFL(tyrePropLeft)
    , tyreFR(tyrePropRight)
    , tyreRL(tyrePropLeft)
    , tyreRR(tyrePropRight)
    , f(z, o, tyreFL, tyreFR, tyreRL, tyreRR)
    , integrator(f)
    , runTask(true)
    , firstMoveCall(true)
    , startPos(getStartPositionOnRoad())
{
    resetState();
}

CarDynamicsRtus::~CarDynamicsRtus()
{
    runTask = false;
}

void CarDynamicsRtus::resetState()
{
    auto &D_b = std::get<0>(y);
    auto &V_b = std::get<1>(y);
    //auto& q_b = std::get<2>(y);
    //auto& w_b = std::get<3>(y);
    auto &u_wfl = std::get<4>(y);
    auto &du_wfl = std::get<5>(y);
    auto &u_wfr = std::get<6>(y);
    auto &du_wfr = std::get<7>(y);
    auto &u_wrl = std::get<8>(y);
    auto &du_wrl = std::get<9>(y);
    auto &u_wrr = std::get<10>(y);
    auto &du_wrr = std::get<11>(y);
    auto &w_wfl = std::get<12>(y);
    auto &w_wfr = std::get<13>(y);
    auto &w_wrl = std::get<14>(y);
    auto &w_wrr = std::get<15>(y);
    auto &w_e = std::get<16>(y);

    auto &P_wfl = std::get<0>(z);
    auto &P_wfr = std::get<1>(z);
    auto &P_wrl = std::get<2>(z);
    auto &P_wrr = std::get<3>(z);
    double &a_steer = std::get<4>(z);
    double &i_pt = std::get<5>(z);
    double &s_gp = std::get<6>(z);

    const auto &D_wfl = std::get<0>(o);
    const auto &D_wfr = std::get<1>(o);
    const auto &D_wrl = std::get<2>(o);
    const auto &D_wrr = std::get<3>(o);

    //Ground planes
    P_wfl = (1.0) * cardyn_rtus::e3 + 0.0 * cardyn_rtus::einf;
    P_wfr = (1.0) * cardyn_rtus::e3 + 0.0 * cardyn_rtus::einf;
    P_wrl = (1.0) * cardyn_rtus::e3 + 0.0 * cardyn_rtus::einf;
    P_wrr = (1.0) * cardyn_rtus::e3 + 0.0 * cardyn_rtus::einf;

    //Set initial values
    cardyn_rtus::cm::mv<1, 2, 4>::type p_b = { 0.0, 0.0, 0.75 };
    cardyn_rtus::cm::mv<0, 3, 5, 6>::type R_b = { 1.0, 0.0, 0.0, 0.0 };

    //Road info
    road_wheel_fl = NULL;
    road_wheel_fr = NULL;
    road_wheel_rl = NULL;
    road_wheel_rr = NULL;
    u_wheel_fl = -1.0;
    u_wheel_fr = -1.0;
    u_wheel_rl = -1.0;
    u_wheel_rr = -1.0;

    if (startPos.first)
    {
        Transform transform = startPos.first->getRoadTransform(startPos.second.u(), startPos.second.v());
        //cardyn_rtus::Vector r_c = {point.y(),-point.x(),point.z()};
        R_b[0] = transform.q().w();
        R_b[1] = transform.q().z();
        R_b[2] = -transform.q().y();
        R_b[3] = transform.q().x();
        cardyn_rtus::cm::mv<0, 3, 5, 6>::type R_xodr = exp(0.5 * (0.5 * M_PI * cardyn_rtus::e1 * cardyn_rtus::e2));
        R_b = R_b * R_xodr;

        p_b = cardyn_rtus::cm::mv<1, 2, 4>::type({ transform.v().y(), -transform.v().x(), transform.v().z() }) + R_b * p_b * ~R_b;

        std::cout << "Found road: " << startPos.first->getId() << ", u: " << startPos.second.u() << ", v: " << startPos.second.v() << std::endl;
        std::cout << "\t p_b: " << p_b << ", R_b: " << R_b << std::endl;
    }
    auto T_b = cardyn_rtus::one + cardyn_rtus::einf * p_b * 0.5;
    D_b = T_b * R_b;

    V_b = cardyn_rtus::S_type();
    u_wfl, u_wfr, u_wrl, u_wrr = 0.0;
    du_wfl, du_wfr, du_wrl, du_wrr = 0.0;
    w_wfl, w_wfr, w_wrl, w_wrr = 0.0;
    w_e = 0.0;

    //dp_b[0] = 0.0;
    //q_b[0] = 1.0;
    a_steer = 0.0;
    i_pt = 0.0;
    s_gp = 0.0;

    chassisTrans.makeIdentity();

    f(0.0, y);

    if (startPos.first)
    {
        road_wheel_fl = road_wheel_fr = road_wheel_rl = road_wheel_rr = startPos.first;
        u_wheel_fl = u_wheel_fr = u_wheel_rl = u_wheel_rr = startPos.second.u();

        const double wheel_radius = 0.255;
        auto p_g0 = eval(wheel_radius * cardyn_rtus::e3 + 0.5 * wheel_radius * wheel_radius * cardyn_rtus::einf + cardyn_rtus::e0);
        auto p_g_wfl = eval(grade<1>(D_b * D_wfl * p_g0 * ~D_wfl * ~D_b));
        getFirstRoadSystemContactPoint(p_g_wfl, road_wheel_fl, u_wheel_fl, P_wfl);
        auto p_g_wfr = eval(grade<1>(D_b * D_wfr * p_g0 * ~D_wfr * ~D_b));
        getFirstRoadSystemContactPoint(p_g_wfr, road_wheel_fr, u_wheel_fr, P_wfr);
        auto p_g_wrl = eval(grade<1>(D_b * D_wrl * p_g0 * ~D_wrl * ~D_b));
        getFirstRoadSystemContactPoint(p_g_wrl, road_wheel_rl, u_wheel_rl, P_wrl);
        auto p_g_wrr = eval(grade<1>(D_b * D_wrr * p_g0 * ~D_wrr * ~D_b));
        getFirstRoadSystemContactPoint(p_g_wrr, road_wheel_rr, u_wheel_rr, P_wrr);

        //std::cout << "\t P_wfl: " << P_wfl << ", road: " << road_wheel_fl->getId() << std::endl;
        //std::cout << "\t P_wfr: " << P_wfl << ", road: " << road_wheel_fr->getId() << std::endl;
        //std::cout << "\t P_wrl: " << P_wrl << ", road: " << road_wheel_rl->getId() << std::endl;
        //std::cout << "\t P_wrr: " << P_wrr << ", road: " << road_wheel_rr->getId() << std::endl;
    }
}

void CarDynamicsRtus::run()
{
    auto &D_b = std::get<0>(y);
    auto &V_b = std::get<1>(y);
    //auto& q_b = std::get<2>(y);
    //auto& w_b = std::get<3>(y);
    auto &u_wfl = std::get<4>(y);
    auto &du_wfl = std::get<5>(y);
    auto &u_wfr = std::get<6>(y);
    auto &du_wfr = std::get<7>(y);
    auto &u_wrl = std::get<8>(y);
    auto &du_wrl = std::get<9>(y);
    auto &u_wrr = std::get<10>(y);
    auto &du_wrr = std::get<11>(y);
    auto &w_wfl = std::get<12>(y);
    auto &w_wfr = std::get<13>(y);
    auto &w_wrl = std::get<14>(y);
    auto &w_wrr = std::get<15>(y);
    auto &w_e = std::get<16>(y);

    auto &P_wfl = std::get<0>(z);
    auto &P_wfr = std::get<1>(z);
    auto &P_wrl = std::get<2>(z);
    auto &P_wrr = std::get<3>(z);
    double &a_steer = std::get<4>(z);
    double &i_pt = std::get<5>(z);
    double &s_gp = std::get<6>(z);
    double &brake = std::get<7>(z);

    const auto &D_wfl = std::get<0>(o);
    const auto &D_wfr = std::get<1>(o);
    const auto &D_wrl = std::get<2>(o);
    const auto &D_wrr = std::get<3>(o);

    while (runTask)
    {
        //Steering
        a_steer = InputDevice::instance()->getSteeringWheelAngle() * 0.05;
        if (a_steer < (-M_PI * 0.4))
            a_steer = -M_PI * 0.4;
        if (a_steer > (M_PI * 0.4))
            a_steer = M_PI * 0.4;

        //Gear, gas, brake
        i_pt = f.i_g[InputDevice::instance()->getGear() + 1] * f.i_a;
        s_gp = InputDevice::instance()->getAccelerationPedal();
        brake = InputDevice::instance()->getBrakePedal() * 90000.0;
        //double clutchPedal = InputDevice::instance()->getClutchPedal();

        //Wheel ground plane
        const double wheel_radius = 0.255;
        auto p_g0 = eval(wheel_radius * cardyn_rtus::e3 + 0.5 * wheel_radius * wheel_radius * cardyn_rtus::einf + cardyn_rtus::e0);

        auto p_g_wfl = eval(grade<1>(D_b * D_wfl * p_g0 * ~D_wfl * ~D_b));
        getRoadSystemContactPoint(p_g_wfl, road_wheel_fl, u_wheel_fl, P_wfl);
        auto p_g_wfr = eval(grade<1>(D_b * D_wfr * p_g0 * ~D_wfr * ~D_b));
        getRoadSystemContactPoint(p_g_wfr, road_wheel_fr, u_wheel_fr, P_wfr);
        auto p_g_wrl = eval(grade<1>(D_b * D_wrl * p_g0 * ~D_wrl * ~D_b));
        getRoadSystemContactPoint(p_g_wrl, road_wheel_rl, u_wheel_rl, P_wrl);
        auto p_g_wrr = eval(grade<1>(D_b * D_wrr * p_g0 * ~D_wrr * ~D_b));
        getRoadSystemContactPoint(p_g_wrr, road_wheel_rr, u_wheel_rr, P_wrr);

        P_wfl = P_wfl + 0.5 * cardyn_rtus::einf;
        P_wfr = P_wfr + 0.5 * cardyn_rtus::einf;

        integrator(0.001, y);

        OpenThreads::Thread::microSleep(10000);
        //OpenThreads::Thread::YieldCurrentThread();
    }
}

void CarDynamicsRtus::move(VrmlNodeVehicle *vehicle)
{
    if (firstMoveCall)
    {
        start();
        firstMoveCall = false;
    }

    auto &D_b = std::get<0>(y);

    //Set new body displacements
    auto p_b = eval(grade<1>(D_b * cardyn_rtus::e0 * ~D_b));
    auto R_b = eval(~part<0, 3, 5, 6>(D_b));

    chassisTrans.setTrans(osg::Vec3(-p_b[1], p_b[2], -p_b[0]));
    chassisTrans.setRotate(osg::Quat(R_b[2],
                                     R_b[1],
                                     -R_b[3],
                                     R_b[0]));

    vehicle->setVRMLVehicle(chassisTrans);
}

void CarDynamicsRtus::setVehicleTransformation(const osg::Matrix &m)
{
    resetState();

    osg::Vec3d t_c = chassisTrans.getTrans();
    //chassisTrans.setTrans(osg::Vec3(-p_b[1], p_b[2], -p_b[0]));
    //cardyn_rtus::cm::mv<1,2,4>::type p_vrml = {t_c[0], t_c[1], t_c[2]};
    //auto p_b = eval(grade<1>(R_vrml*p_vrml*~R_vrml));
    cardyn_rtus::cm::mv<1, 2, 4>::type p_b = { -m(3, 2), -m(3, 0), m(3, 1) };
    auto T_b = cardyn_rtus::one + cardyn_rtus::einf * p_b * 0.5;

    //osg::Quat q_c = chassisTrans.getRotate();
    //cardyn_rtus::cm::mv<0,3,5,6>::type R_b_vrml = {q_c[3], -q_c[2], q_c[1], q_c[0]};
    //auto R_b = R_vrml*R_b_vrml*~R_vrml;

    auto &D_b = std::get<0>(y);
    //D_b = T_b*R_b;
    D_b = T_b;
}

cardyn_rtus::Plane CarDynamicsRtus::getRoadTangentPlane(Road *&road, Vector2D v_c)
{
    if (road)
    {
        RoadPoint point = road->getRoadPoint(v_c.u(), v_c.v());
        cardyn_rtus::Vector r_c = { point.y(), -point.x(), point.z() };
        cardyn_rtus::Vector n_c = { point.ny(), -point.nx(), point.nz() };
        double d_p = eval(r_c & n_c);
        return (n_c + d_p * cardyn_rtus::einf);
    }
    else
    {
        return cardyn_rtus::Plane({ 0.0, 0.0, 1.0, 0.0 });
    }
}

void CarDynamicsRtus::getRoadSystemContactPoint(const cardyn_rtus::Point &p_w, Road *&road, double &u, cardyn_rtus::Plane &s_c)
{
    cardyn_rtus::Vector r_w = part_type<cardyn_rtus::Vector>(p_w);

    Vector3D v_w(-r_w[1], r_w[0], r_w[2]);

    Vector2D v_c(0.0, 0.0);
    v_c = RoadSystem::Instance()->searchPositionFollowingRoad(v_w, road, u);

    if (!v_c.isNaV())
    {
        //s_c = getRoadTangentPlane(road, v_c);

        cardyn_rtus::Plane new_s_c = getRoadTangentPlane(road, v_c);
        double err = eval(magnitude((new_s_c * s_c) & cardyn_rtus::e0));
        std::cout << "Error: " << err << std::endl;
        if (err < 2.0)
        {
            s_c = new_s_c;
        }
    }
}

void CarDynamicsRtus::getFirstRoadSystemContactPoint(const cardyn_rtus::Point &p_w, Road *&road, double &u, cardyn_rtus::Plane &s_c)
{
    cardyn_rtus::Vector r_w = part_type<cardyn_rtus::Vector>(p_w);
    Vector3D v_w(-r_w[1], r_w[0], r_w[2]);

    Vector2D v_c(0.0, 0.0);
    if (RoadSystem::Instance())
    {
        v_c = RoadSystem::Instance()->searchPosition(v_w, road, u);
    }
    else
    {
        std::cerr << "VehicleDynamicsPlugin::getFirstContactPoint(): no road system!" << std::endl;
    }

    s_c = getRoadTangentPlane(road, v_c);
}

std::pair<Road *, Vector2D> CarDynamicsRtus::getStartPositionOnRoad()
{
    double targetS = 5.0;

    //Road* targetRoad = NULL;
    RoadSystem *system = RoadSystem::Instance();

    for (int roadIt = 0; roadIt < system->getNumRoads(); ++roadIt)
    {
        Road *road = system->getRoad(roadIt);
        if (road->getLength() >= 2.0 * targetS)
        {
            LaneSection *section = road->getLaneSection(targetS);
            if (section)
            {
                for (int laneIt = -1; laneIt >= -section->getNumLanesRight(); --laneIt)
                {
                    Lane *lane = section->getLane(laneIt);
                    if (lane->getLaneType() == Lane::DRIVING)
                    {
                        double t = 0.0;
                        for (int laneWidthIt = -1; laneWidthIt > laneIt; --laneWidthIt)
                        {
                            t -= section->getLane(laneWidthIt)->getWidth(targetS);
                            //t += section->getLaneWidth(laneWidthIt, targetS);
                        }
                        t -= 0.5 * lane->getWidth(targetS);
                        return std::make_pair(road, Vector2D(targetS, t));
                    }
                }
            }
        }
    }

    return std::make_pair((Road *)NULL, Vector2D::NaV());
}
