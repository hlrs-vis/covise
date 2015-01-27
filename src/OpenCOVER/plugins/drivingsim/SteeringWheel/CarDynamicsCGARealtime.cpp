/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CarDynamicsCGARealtime.h"
#include <iostream>
#include <cmath>

#include "SteeringWheel.h"

CarDynamicsCGARealtime::CarDynamicsCGARealtime()
    : XenomaiTask::XenomaiTask("CarDynamicsCGARealtimeTask", 0, 99, T_FPU | T_CPU(5))
    , tyrePropLeft(magicformula2004::TyrePropertyPack::TYRE_LEFT)
    , tyrePropRight(magicformula2004::TyrePropertyPack::TYRE_RIGHT)
    , tyreFL(tyrePropLeft)
    , tyreFR(tyrePropRight)
    , tyreRL(tyrePropLeft)
    , tyreRR(tyrePropRight)
    , f(z, o, tyreFL, tyreFR, tyreRL, tyreRR)
    , integrator(f)
    , firstMoveCall(true)
    , startPos(getStartPositionOnRoad())
    ,

    runTask(true)
    , taskFinished(false)
    , overruns(0)
    , hapSimState(PAUSING)
    , centerSteeringWheelOnNextRun(false)
    , motPlat(NULL)
    , steerCon(NULL)
    , steerWheel(NULL)
{
    resetState();

    if (coVRMSController::instance()->isMaster())
    {
        motPlat = ValidateMotionPlatform::instance();

        steerCon = new CanOpenController("rtcan1");
        steerWheel = new XenomaiSteeringWheel(*steerCon, 1);

        //start();
    }
}

CarDynamicsCGARealtime::~CarDynamicsCGARealtime()
{
    if (coVRMSController::instance()->isMaster())
    {
        RT_TASK_INFO info;
        inquire(info);

        if (info.status & T_STARTED)
        {
            runTask = false;
            while (!taskFinished)
            {
                usleep(100000);
            }
        }

        delete steerWheel;
        delete steerCon;

        delete motPlat;
    }
}

void CarDynamicsCGARealtime::resetState()
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
    P_wfl = (1.0) * cardyncga::e3 + 0.0 * cardyncga::einf;
    P_wfr = (1.0) * cardyncga::e3 + 0.0 * cardyncga::einf;
    P_wrl = (1.0) * cardyncga::e3 + 0.0 * cardyncga::einf;
    P_wrr = (1.0) * cardyncga::e3 + 0.0 * cardyncga::einf;

    //Set initial values
    cardyncga::cm::mv<1, 2, 4>::type p_b = { 0.0, 0.0, 0.75 };
    cardyncga::cm::mv<0, 3, 5, 6>::type R_b = { 1.0, 0.0, 0.0, 0.0 };

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
        //cardyncga::Vector r_c = {point.y(),-point.x(),point.z()};
        R_b[0] = transform.q().w();
        R_b[1] = transform.q().z();
        R_b[2] = -transform.q().y();
        R_b[3] = transform.q().x();
        cardyncga::cm::mv<0, 3, 5, 6>::type R_xodr = exp(0.5 * (0.5 * M_PI * cardyncga::e1 * cardyncga::e2));
        R_b = R_b * R_xodr;

        p_b = cardyncga::cm::mv<1, 2, 4>::type({ transform.v().y(), -transform.v().x(), transform.v().z() }) + R_b * p_b * ~R_b;

        std::cout << "Found road: " << startPos.first->getId() << ", u: " << startPos.second.u() << ", v: " << startPos.second.v() << std::endl;
        std::cout << "\t p_b: " << p_b << ", R_b: " << R_b << std::endl;
    }
    auto T_b = cardyncga::one + cardyncga::einf * p_b * 0.5;
    D_b = T_b * R_b;

    V_b = cardyncga::S_type();
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

        const double wheel_radius = 0.325;
        auto p_g0 = eval(wheel_radius * cardyncga::e3 + 0.5 * wheel_radius * wheel_radius * cardyncga::einf + cardyncga::e0);
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

void CarDynamicsCGARealtime::run()
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
    const auto &F_wfl = std::get<4>(o);
    const auto &F_wfr = std::get<5>(o);
    const auto &F_wrl = std::get<6>(o);
    const auto &F_wrr = std::get<7>(o);

    //Motion platform
    steerWheel->init();
    motPlat->start();

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();
    rt_task_sleep(period);

    set_periodic(period);

    while (runTask)
    {
        checkHapticSimulationState();

        double steeringWheelCurrent = 0.0;

        if (hapSimState == DRIVING)
        {

            //Gear, gas, brake
            i_pt = f.i_g[InputDevice::instance()->getGear() + 1] * f.i_a;
            s_gp = InputDevice::instance()->getAccelerationPedal();
            brake = InputDevice::instance()->getBrakePedal() * 90000.0;
            //double clutchPedal = InputDevice::instance()->getClutchPedal();

            //Wheel ground plane
            const double wheel_radius = 0.325;
            auto p_g0 = eval(wheel_radius * cardyncga::e3 + 0.5 * wheel_radius * wheel_radius * cardyncga::einf + cardyncga::e0);

            auto p_g_wfl = eval(grade<1>(D_b * D_wfl * p_g0 * ~D_wfl * ~D_b));
            getRoadSystemContactPoint(p_g_wfl, road_wheel_fl, u_wheel_fl, P_wfl);
            auto p_g_wfr = eval(grade<1>(D_b * D_wfr * p_g0 * ~D_wfr * ~D_b));
            getRoadSystemContactPoint(p_g_wfr, road_wheel_fr, u_wheel_fr, P_wfr);
            auto p_g_wrl = eval(grade<1>(D_b * D_wrl * p_g0 * ~D_wrl * ~D_b));
            getRoadSystemContactPoint(p_g_wrl, road_wheel_rl, u_wheel_rl, P_wrl);
            auto p_g_wrr = eval(grade<1>(D_b * D_wrr * p_g0 * ~D_wrr * ~D_b));
            getRoadSystemContactPoint(p_g_wrr, road_wheel_rr, u_wheel_rr, P_wrr);

            integrator(0.001, y);

            //Steering
            steeringWheelCurrent = -5000.0 * (F_wfl.element<0x03>() + F_wfr.element<0x03>()) - (double)steerWheel->getSmoothedSpeed() * 0.0002;
        }

        //Steering
        steerWheel->setCurrent(steeringWheelCurrent);

        steerCon->sendSync();
        steerCon->recvPDO(1);
        steerCon->sendPDO();

        double steerPosition = steerWheel->getPosition();
        a_steer = -2 * M_PI * ((double)steerPosition / (double)steerWheel->countsPerTurn) * 0.1;
        if (a_steer < (-M_PI * 0.4))
            a_steer = -M_PI * 0.4;
        else if (a_steer > (M_PI * 0.4))
            a_steer = M_PI * 0.4;

        if (overruns > 0)
        {
            std::cout << "CarDynamicsCGARealtime::run(): task periodic time overrun: " << overruns << std::endl;
        }

        rt_task_wait_period(&overruns);
    }

    steerWheel->setCurrent(0);
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
    while (!motPlat->isGrounded())
    {
        rt_task_wait_period(&overruns);
    }
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();

    set_periodic(TM_INFINITE);
    steerWheel->shutdown();

    taskFinished = true;
}

void CarDynamicsCGARealtime::move(VrmlNodeVehicle *vehicle)
{
    cardyncga::StateVector sync_y;

    if (coVRMSController::instance()->isMaster())
    {
        if (firstMoveCall)
        {
            start();
            firstMoveCall = false;
        }

        sync_y = y;
        coVRMSController::instance()->sendSlaves((char *)&sync_y, sizeof(cardyncga::StateVector));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&sync_y, sizeof(cardyncga::StateVector));
    }

    auto &D_b = std::get<0>(sync_y);

    //Set new body displacements
    auto p_b = eval(grade<1>(D_b * cardyncga::e0 * ~D_b));
    auto R_b = eval(~part<0, 3, 5, 6>(D_b));

    chassisTrans.setTrans(osg::Vec3(-p_b[1], p_b[2], -p_b[0]));
    chassisTrans.setRotate(osg::Quat(R_b[2],
                                     R_b[1],
                                     -R_b[3],
                                     R_b[0]));

    vehicle->setVRMLVehicle(chassisTrans);
}

void CarDynamicsCGARealtime::setVehicleTransformation(const osg::Matrix &)
{
    resetState();
}

cardyncga::Plane CarDynamicsCGARealtime::getRoadTangentPlane(Road *&road, Vector2D v_c)
{
    if (road)
    {
        RoadPoint point = road->getRoadPoint(v_c.u(), v_c.v());
        cardyncga::Vector r_c = { point.y(), -point.x(), point.z() };
        cardyncga::Vector n_c = { point.ny(), -point.nx(), point.nz() };
        double d_p = eval(r_c & n_c);
        return (n_c + d_p * cardyncga::einf);
    }
    else
    {
        return cardyncga::Plane({ 0.0, 0.0, 1.0, 0.0 });
    }
}

void CarDynamicsCGARealtime::getRoadSystemContactPoint(const cardyncga::Point &p_w, Road *&road, double &u, cardyncga::Plane &s_c)
{
    cardyncga::Vector r_w = part_type<cardyncga::Vector>(p_w);

    Vector3D v_w(-r_w[1], r_w[0], r_w[2]);

    Vector2D v_c(0.0, 0.0);
    v_c = RoadSystem::Instance()->searchPositionFollowingRoad(v_w, road, u);

    if (!v_c.isNaV())
    {
        //s_c = getRoadTangentPlane(road, v_c);

        cardyncga::Plane new_s_c = getRoadTangentPlane(road, v_c);
        double err = eval(magnitude((new_s_c * s_c) & cardyncga::e0));
        //std::cout << "Error: " << err << std::endl;
        if (err < 2.0)
        {
            s_c = new_s_c;
        }
    }
}

void CarDynamicsCGARealtime::getFirstRoadSystemContactPoint(const cardyncga::Point &p_w, Road *&road, double &u, cardyncga::Plane &s_c)
{
    cardyncga::Vector r_w = part_type<cardyncga::Vector>(p_w);
    Vector3D v_w(-r_w[1], r_w[0], r_w[2]);

    Vector2D v_c(0.0, 0.0);
    if (RoadSystem::Instance())
    {
        v_c = RoadSystem::Instance()->searchPosition(v_w, road, u);
    }
    else
    {
        std::cerr << "VehicleDynamicsPlugin::getFirstContactPoint(): no road system!" << std::endl;
        return;
    }

    s_c = getRoadTangentPlane(road, v_c);
}

std::pair<Road *, Vector2D> CarDynamicsCGARealtime::getStartPositionOnRoad()
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

void CarDynamicsCGARealtime::checkHapticSimulationState()
{
    if (hapSimState == PLATFORM_LOWERING)
    {
        if (motPlat->isGrounded())
        {
            hapSimState = PAUSING;
            motPlat->getSendMutex().acquire(period);
            motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
            motPlat->getSendMutex().release();
        }
    }
    else if (hapSimState == PLATFORM_RAISING)
    {
        if (motPlat->isLifted())
        {
            hapSimState = DRIVING;

            /*motPlat->getSendMutex().acquire(period);
         motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>();
         for(unsigned int motIt = 0; motIt < motPlat->numLinMots; ++motIt) {
            motPlat->setVelocitySetpoint(motIt, ValidateMotionPlatform::velMax);
            motPlat->setAccelerationSetpoint(motIt, ValidateMotionPlatform::accMax);
         }
         motPlat->getSendMutex().release();*/
        }
    }
    else if (hapSimState == PAUSING)
    {
        if (centerSteeringWheelOnNextRun)
        {
            steerWheel->center();
            centerSteeringWheelOnNextRun = false;
        }
    }
}

void CarDynamicsCGARealtime::platformToGround()
{
    hapSimState = PLATFORM_LOWERING;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
}

void CarDynamicsCGARealtime::platformReturnToAction()
{
    hapSimState = PLATFORM_RAISING;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();

    motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle); //Right
    motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle); //Left
    motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle); //Rear

    motPlat->setVelocitySetpoint(0, ValidateMotionPlatform::velMax * 0.1);
    motPlat->setVelocitySetpoint(1, ValidateMotionPlatform::velMax * 0.1);
    motPlat->setVelocitySetpoint(2, ValidateMotionPlatform::velMax * 0.1);

    motPlat->setAccelerationSetpoint(0, ValidateMotionPlatform::accMax * 0.1);
    motPlat->setAccelerationSetpoint(1, ValidateMotionPlatform::accMax * 0.1);
    motPlat->setAccelerationSetpoint(2, ValidateMotionPlatform::accMax * 0.1);

    motPlat->getSendMutex().release();
}
