/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FourWheelDynamicsRealtime.h"
#include <VehicleUtil/VehicleUtil.h>


#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

FourWheelDynamicsRealtime::FourWheelDynamicsRealtime()
    :
#ifdef __XENO__
#ifdef MERCURY
    XenomaiTask::XenomaiTask("FourWheelDynamicsRealtimeTask", 0, 99, 0)
#else
    XenomaiTask::XenomaiTask("FourWheelDynamicsRealtimeTask", 0, 99, T_FPU | T_CPU(5))
#endif
    ,
#endif
    dy(cardyn::getExpressionVector())
    ,
#ifdef __XENO__
    integrator(dy, y)
    ,
#endif
    r_i(4)
    , n_i(4)
    , r_n(4)
    , //Start position of hermite
    t_n(4)
    , //Start tangent of hermite
    r_o(4)
    , //End position of hermite
    t_o(4)
    , //End tangent of hermite
    newIntersections(false)
    , hermite_dt(0.02)
    , i_w(4)
    , startPos(getStartPositionOnRoad())
    , leftRoad(true)
{
    initState();

    i_proj[0] = 1.0;

    runTask = true;
    doCenter = false;
    taskFinished = false;
    returningToAction = false;
    movingToGround = false;
    pause = true;
    overruns = 0;
    if (coVRMSController::instance()->isMaster())
    {
#ifdef __XENO__
        //motPlat = new ValidateMotionPlatform("rtcan0", 0, 99, T_FPU|T_CPU(4));
        //std::cerr << "--- FourWheelDynamicsRealtime::FourWheelDynamicsRealtime(): Entering master startup ---" << std::endl;
        motPlat = ValidateMotionPlatform::instance();

        steerCon = new CanOpenController("rtcan1");
        steerWheel = new XenomaiSteeringWheel(*steerCon, 1);

        //std::cerr << "--- FourWheelDynamicsRealtime::FourWheelDynamicsRealtime(): Starting simulation task ---" << std::endl;
        start();
#endif
    }

    vdTab = new coTUITab("Four wheel dynamics", coVRTui::instance()->mainFolder->getID());
    vdTab->setPos(0, 0);
    k_Pp_Label = new coTUILabel("k_Pp:", vdTab->getID());
    k_Pp_Label->setPos(0, 0);
    d_Pp_Label = new coTUILabel("d_Pp:", vdTab->getID());
    d_Pp_Label->setPos(2, 0);
    k_Pq_Label = new coTUILabel("k_Pq:", vdTab->getID());
    k_Pq_Label->setPos(4, 0);
    d_Pq_Label = new coTUILabel("d_Pq:", vdTab->getID());
    d_Pq_Label->setPos(6, 0);
    k_wf_Label = new coTUILabel("Federkonstante vorne:", vdTab->getID());
    k_wf_Label->setPos(0, 2);
    d_wf_Label = new coTUILabel("Daempfung vorne:", vdTab->getID());
    d_wf_Label->setPos(2, 2);
    k_wr_Label = new coTUILabel("Federkonstante hinten:", vdTab->getID());
    k_wr_Label->setPos(4, 2);
    d_wr_Label = new coTUILabel("Daempfung hinten:", vdTab->getID());
    d_wr_Label->setPos(6, 2);

    k_Pp_Slider = new coTUISlider("k_Pp slider", vdTab->getID());
    k_Pp_Slider->setRange(0, 10000);
    k_Pp_Slider->setValue(100);
    k_Pp_Slider->setPos(0, 1);
    d_Pp_Slider = new coTUISlider("d_Pp slider", vdTab->getID());
    d_Pp_Slider->setRange(0, 1000);
    d_Pp_Slider->setValue(10);
    d_Pp_Slider->setPos(2, 1);
    k_Pq_Slider = new coTUISlider("k_Pq slider", vdTab->getID());
    k_Pq_Slider->setRange(0, 10000);
    k_Pq_Slider->setValue(100);
    k_Pq_Slider->setPos(4, 1);
    d_Pq_Slider = new coTUISlider("d_Pq slider", vdTab->getID());
    d_Pq_Slider->setRange(0, 1000);
    d_Pq_Slider->setValue(10);
    d_Pq_Slider->setPos(6, 1);
    k_wf_Slider = new coTUIFloatSlider("k_wf slider", vdTab->getID());
    k_wf_Slider->setRange(000, 50000);
    k_wf_Slider->setValue(17400);
    k_wf_Slider->setPos(0, 3);
    d_wf_Slider = new coTUIFloatSlider("d_wf slider", vdTab->getID());
    d_wf_Slider->setRange(00, 10000);
    d_wf_Slider->setValue(2600);
    d_wf_Slider->setPos(2, 3);
    k_wr_Slider = new coTUIFloatSlider("k_wr slider", vdTab->getID());
    k_wr_Slider->setRange(000, 50000);
    k_wr_Slider->setValue(26100);
    k_wr_Slider->setPos(4, 3);
    d_wr_Slider = new coTUIFloatSlider("d_wr slider", vdTab->getID());
    d_wr_Slider->setRange(00, 10000);
    d_wr_Slider->setValue(2600);
    d_wr_Slider->setPos(6, 3);
}

FourWheelDynamicsRealtime::~FourWheelDynamicsRealtime()
{
    if (coVRMSController::instance()->isMaster())
    {
#ifdef __XENO__
        RT_TASK_INFO info;
        inquire(info);

#ifdef MERCURY
    if (info.stat.status & __THREAD_S_STARTED)
#else
    if (info.status & T_STARTED)
#endif
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
#endif
    }
}

void FourWheelDynamicsRealtime::initState()
{
    y = cardyn::StateVectorType();

    if (startPos.first)
    {
        Transform transform = startPos.first->getRoadTransform(startPos.second.u(), startPos.second.v());
        //cardyncga::Vector r_c = {point.y(),-point.x(),point.z()};
        gealg::mv<4, 0x06050300>::type R_b;
        R_b[0] = transform.q().w();
        R_b[1] = transform.q().z();
        R_b[2] = -transform.q().y();
        R_b[3] = transform.q().x();
        gealg::mv<4, 0x06050300>::type R_xodr = exp(0.5 * (-0.5 * M_PI * cardyn::x * cardyn::y));
        std::get<2>(y) = !(R_b * R_xodr);

        gealg::mv<3, 0x040201>::type p_b_init;
        p_b_init[2] = 0.75;
        gealg::mv<3, 0x040201>::type p_road;
        p_road[0] = transform.v().y();
        p_road[1] = -transform.v().x();
        p_road[2] = transform.v().z();
        std::get<0>(y) = p_road + grade<1>((!std::get<2>(y)) * p_b_init * (std::get<2>(y)));

        currentRoad[0] = startPos.first;
        currentRoad[1] = startPos.first;
        currentRoad[2] = startPos.first;
        currentRoad[3] = startPos.first;
        currentLongPos[0] = startPos.second.u();
        currentLongPos[1] = startPos.second.u();
        currentLongPos[2] = startPos.second.u();
        currentLongPos[3] = startPos.second.u();

        leftRoad = false;

        std::cout << "Found road: " << startPos.first->getId() << ", u: " << startPos.second.u() << ", v: " << startPos.second.v() << std::endl;
        std::cout << "\t p_b: " << std::get<0>(y) << ", R_b: " << std::get<2>(y) << std::endl;
    }
    else
    {
        std::get<0>(y)[2] = -0.2; //Initial position
        std::get<2>(y)[0] = 1.0; //Initial orientation (Important: magnitude be one!)
        //std::get<2>(y)[0] = 0.982131;  std::get<2>(y)[2] = 0.188203;   //Initial orientation (Important: magnitude be one!)
        //std::get<2>(y)[0] = cos(0.5*M_PI); std::get<2>(y)[1] = sin(0.5*M_PI);   //Initial orientation (Important: magnitude be one!)
        currentRoad[0] = NULL;
        currentRoad[1] = NULL;
        currentRoad[2] = NULL;
        currentRoad[3] = NULL;
        currentLongPos[0] = -1.0;
        currentLongPos[1] = -1.0;
        currentLongPos[2] = -1.0;
        currentLongPos[3] = -1.0;
        leftRoad = true;
    }
    //std::get<1>(y)[0] = 1.0;    //Initial velocity
    //std::get<1>(y)[1] = 5.0;    //Initial velocity
    //std::get<3>(y)[1] = -0.3;    //Initial angular velocity
    //std::get<3>(y)[2] = 0.3;    //Initial angular velocity
    //std::get<32>(y)[0] = M_PI*0.1;    //Initial steering wheel position
    //std::get<33>(y)[0] = 10.0;    //Permanent torque on rear wheels
    std::get<39>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::get<40>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::get<41>(y)[0] = cardyn::i_a; //Initial steering wheel position: magnitude be one!

    /*cardyn::P_b.e_(y) = part<4, 0x06050403>(cardyn::p_b + cardyn::P_xy)(y);
   cardyn::k_Pp.e_(y)[0] = 100.0;
   cardyn::d_Pp.e_(y)[0] = 10.0;
   cardyn::k_Pq.e_(y)[0] = 100.0;
   cardyn::d_Pq.e_(y)[0] = 10.0;*/

    r_i[0] = gealg::mv<3, 0x040201>::type();
    r_i[1] = gealg::mv<3, 0x040201>::type();
    r_i[2] = gealg::mv<3, 0x040201>::type();
    r_i[3] = gealg::mv<3, 0x040201>::type();
    n_i[0] = gealg::mv<3, 0x040201>::type();
    n_i[1] = gealg::mv<3, 0x040201>::type();
    n_i[2] = gealg::mv<3, 0x040201>::type();
    n_i[3] = gealg::mv<3, 0x040201>::type();

    r_n[0] = (cardyn::p_b + cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl)(y);
    r_o[0] = r_n[0];
    r_n[1] = (cardyn::p_b + cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr)(y);
    r_o[1] = r_n[1];
    r_n[2] = (cardyn::p_b + cardyn::r_wrl - cardyn::z * cardyn::r_w - cardyn::u_wrl)(y);
    r_o[2] = r_n[2];
    r_n[3] = (cardyn::p_b + cardyn::r_wrr - cardyn::z * cardyn::r_w - cardyn::u_wrr)(y);
    r_o[3] = r_n[3];
    t_n[0] = gealg::mv<3, 0x040201>::type();
    t_n[1] = gealg::mv<3, 0x040201>::type();
    t_n[2] = gealg::mv<3, 0x040201>::type();
    t_n[3] = gealg::mv<3, 0x040201>::type();
    t_o[0] = gealg::mv<3, 0x040201>::type();
    t_o[1] = gealg::mv<3, 0x040201>::type();
    t_o[2] = gealg::mv<3, 0x040201>::type();
    t_o[3] = gealg::mv<3, 0x040201>::type();

    cardyn::k_wf.e_(y)[0] = 17400.0;
    cardyn::k_wr.e_(y)[0] = 26100.0;
    //cardyn::k_wf.e_(y)[0] = 1740.0;
    //cardyn::k_wr.e_(y)[0] = 2610.0;
    cardyn::d_wf.e_(y)[0] = 2600.0;
    cardyn::d_wr.e_(y)[0] = 2600.0;

    newIntersections = false;

    rpms = 0.0;
}

void FourWheelDynamicsRealtime::setVehicleTransformation(const osg::Matrix &m)
{
    resetState();

    std::get<0>(y)[0] = -m(3, 2);
    std::get<0>(y)[1] = -m(3, 0);
    std::get<0>(y)[2] = m(3, 1);

    std::cout << "Reset: position: " << std::get<0>(y) << std::endl;
}

void FourWheelDynamicsRealtime::resetState()
{
    if (!pause)
    {
        pause = true;
        initState();
#ifdef __XENO__
        platformReturnToAction();
#endif
    }
    else
    {
        initState();
    }
}

void FourWheelDynamicsRealtime::move(VrmlNodeVehicle *vehicle)
{
    if (coVRMSController::instance()->isMaster())
    {
        y_frame = this->y;
        coVRMSController::instance()->sendSlaves((char *)&y_frame, sizeof(cardyn::StateVectorType));

        if (VehicleUtil::instance()->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            rpms = ((cardyn::w_e) * (1.0 / (2.0 * M_PI)))(y_frame)[0];
            if (rpms < 13.33)
            {
                rpms = 13.33;
            }
        }
        else
        {
            rpms = 0.0;
        }
        coVRMSController::instance()->sendSlaves((char *)&rpms, sizeof(rpms));

        //determineGroundPlane();
        //determineHermite();

        /*if(RoadSystem::Instance()) {
         for(int i=0; i<4; ++i) {
            if(currentRoad[i]==NULL) {
               gealg::mv<3, 0x040201>::type r_w = (cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfl-cardyn::u_wfl)*cardyn::q_b))(y_frame);
               Vector3D v_w(-r_w[1], r_w[0], r_w[2]);
               Vector2D v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad[i], currentLongPos[i]);
               //if(i==0 && currentRoad[i]==NULL) std::cerr << "Searching wheel position... nothing found for wheel front left!" << std::endl;
               if(currentRoad[i]) {
                  RoadPoint point = currentRoad[i]->getRoadPoint(v_c.u(), v_c.v());
   		  //std::cerr << "Dynamics v_w: " << v_w.x() << ", " << v_w.y() << ", " << v_w.z() << std::endl;
                  //std::cerr << "i: " << point.x() << ", " << point.y() << ", " << point.z() << std::endl;
               }
            }
         }
      }*/
        //std::cout << "SteeringWheel speed: " << steerWheel->getSpeed() << std::endl;
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&y_frame, sizeof(cardyn::StateVectorType));
        coVRMSController::instance()->readMaster((char *)&rpms, sizeof(rpms));
    }

    gealg::mv<3, 0x040201>::type r_bg = (cardyn::r_wfl + cardyn::r_wfr + cardyn::r_wrl + cardyn::r_wrr) * 0.25 - cardyn::z * cardyn::r_w;
    gealg::mv<3, 0x040201>::type p_bg = (cardyn::p_b + grade<1>((!cardyn::q_b) * r_bg * cardyn::q_b))(y_frame);

    chassisTrans.setTrans(osg::Vec3(-p_bg[1], p_bg[2], -p_bg[0]));
    chassisTrans.setRotate(osg::Quat(std::get<2>(y_frame)[2],
                                     std::get<2>(y_frame)[1],
                                     -std::get<2>(y_frame)[3],
                                     std::get<2>(y_frame)[0]));

    vehicle->setVRMLVehicle(chassisTrans);

    //std::cerr << "w_e: " << std::get<16>(y) << std::endl;

    /*osg::Quat steerRot(std::get<32>(y)[0], osg::Vec3(0,0,1));

   osg::Matrix wheelMatrixFL;
   wheelMatrixFL.setTrans(osg::Vec3(cardyn::v_wn, cardyn::w_wn, -cardyn::u_wn - std::get<4>(y)[0]));
   osg::Quat wheelRotarySpeedFL(0.0, -std::get<12>(y)[0], 0.0, 0.0);
   wheelQuatFL = wheelQuatFL + wheelQuatFL*wheelRotarySpeedFL*(0.5*dt);
   wheelQuatFL = wheelQuatFL*(1/wheelQuatFL.length());
   wheelMatrixFL.setRotate(wheelQuatFL*steerRot);

   osg::Matrix wheelMatrixFR;
   wheelMatrixFR.setTrans(osg::Vec3(cardyn::v_wn, -cardyn::w_wn, -cardyn::u_wn - std::get<6>(y)[0]));
   osg::Quat wheelRotarySpeedFR(0.0, -std::get<13>(y)[0], 0.0, 0.0);
   wheelQuatFR = wheelQuatFR + wheelQuatFR*wheelRotarySpeedFR*(0.5*dt);
   wheelQuatFR = wheelQuatFR*(1/wheelQuatFR.length());
   wheelMatrixFR.setRotate(wheelQuatFR*steerRot);

   osg::Matrix wheelMatrixRL;
   wheelMatrixRL.setTrans(osg::Vec3(-cardyn::v_wn, cardyn::w_wn, -cardyn::u_wn - std::get<8>(y)[0]));
   osg::Quat wheelRotarySpeedRL(0.0, -std::get<14>(y)[0], 0.0, 0.0);
   wheelQuatRL = wheelQuatRL + wheelQuatRL*wheelRotarySpeedRL*(0.5*dt);
   wheelQuatRL = wheelQuatRL*(1/wheelQuatRL.length());
   wheelMatrixRL.setRotate(wheelQuatRL);

   osg::Matrix wheelMatrixRR;
   wheelMatrixRR.setTrans(osg::Vec3(-cardyn::v_wn, -cardyn::w_wn, -cardyn::u_wn - std::get<10>(y)[0]));
   osg::Quat wheelRotarySpeedRR(0.0, -std::get<15>(y)[0], 0.0, 0.0);
   wheelQuatRL = wheelQuatRR + wheelQuatRR*wheelRotarySpeedRR*(0.5*dt);
   wheelQuatRL = wheelQuatRR*(1/wheelQuatRR.length());
   wheelMatrixRR.setRotate(wheelQuatRR);

   vehicle->setVRMLVehicleFrontWheels(wheelMatrixFL, wheelMatrixFR);

   vehicle->setVRMLVehicleRearWheels(wheelMatrixRL, wheelMatrixRR);*/

    /*cardyn::k_Pp.e_(y)[0] = (double)k_Pp_Slider->getValue();
   cardyn::d_Pp.e_(y)[0] = (double)d_Pp_Slider->getValue();
   cardyn::k_Pq.e_(y)[0] = (double)k_Pq_Slider->getValue();
   cardyn::d_Pq.e_(y)[0] = (double)d_Pq_Slider->getValue();*/

    cardyn::k_wf.e_(y)[0] = (double)k_wf_Slider->getValue();
    cardyn::d_wf.e_(y)[0] = (double)d_wf_Slider->getValue();
    cardyn::k_wr.e_(y)[0] = (double)k_wr_Slider->getValue();
    cardyn::d_wr.e_(y)[0] = (double)d_wr_Slider->getValue();

    //std::cerr << "cardyn::k_wf: " << cardyn::k_wf(y) << ", u_wfl: " << cardyn::u_wfl(y) << ", Fsd_wfl: " << cardyn::Fsd_wfl(y) << std::endl;
}

void FourWheelDynamicsRealtime::setSportDamper(bool sport)
{
    if (sport)
    {
        k_wf_Slider->setValue(94000.0);
        k_wr_Slider->setValue(152000.0);
        d_wf_Slider->setValue(4600.0);
        d_wr_Slider->setValue(4600.0);
    }
    else
    {
        k_wf_Slider->setValue(17400.0);
        k_wr_Slider->setValue(26100.0);
        d_wf_Slider->setValue(2600.0);
        d_wr_Slider->setValue(2600.0);
    }
}

#ifdef __XENO__
void FourWheelDynamicsRealtime::run()
{
    double current = 0.0;
    std::deque<double> currentDeque(10, 0.0);
    std::deque<double>::iterator currentDequeIt;

    //std::cerr << "--- FourWheelDynamicsRealtime::FourWheelDynamicsRealtime(): Initing steering wheel ---" << std::endl;
    //Steering wheel
    //steerWheel->center();

    steerWheel->init();

    //std::cerr << "--- FourWheelDynamicsRealtime::FourWheelDynamicsRealtime(): Starting ValidateMotionPlatform task ---" << std::endl;
    //Motion platform
    motPlat->start();

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();
    rt_task_sleep(period);

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

    double dt = hermite_dt;
    int step = 0;

    set_periodic(period);
    while (runTask)
    {
        if (overruns != 0)
        {
            std::cerr << "FourWheelDynamicsRealtimeRealtime::run(): overruns: " << overruns << std::endl;
        }
        if (leftRoad)
        {
            std::cout << "Left Road!" << std::endl;
        }

        gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b) * cardyn::z * cardyn::q_b)(y);
        gealg::mv<1, 0x0>::type proj_n_z = (n_b % cardyn::z);
        if ((proj_n_z[0] < 0.0) || leftRoad)
            resetState();

        if (!pause)
        {
            double h = (double)(period * (overruns + 1)) * 1e-9;

            std::get<42>(y)[0] = InputDevice::instance()->getAccelerationPedal();
            std::get<43>(y)[0] = motPlat->getBrakeForce() * 200.0;
            std::get<44>(y)[0] = (1.0 - InputDevice::instance()->getClutchPedal()) * cardyn::k_cn;
            int gear = InputDevice::instance()->getGear();
            std::get<41>(y)[0] = cardyn::i_g[gear + 1] * cardyn::i_a;
            if (gear == 0)
            {
                std::get<44>(y)[0] = 0.0;
            }

            //Cubic hermite parameter determination
            if (newIntersections)
            {
                std::vector<gealg::mv<3, 0x040201>::type> v_w(4);

                v_w[0] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wfl * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wfl * cardyn::q_b))))(y);
                v_w[1] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wfr * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wfr * cardyn::q_b))))(y);
                v_w[2] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wrl * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wrl * cardyn::q_b))))(y);
                v_w[3] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wrr * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wrr * cardyn::q_b))))(y);

                for (int i = 0; i < 4; ++i)
                {
                    double tau = (double)step * h / dt;
                    if (tau > 1.0)
                    {
                        //std::cerr << "FourWheelDynamicsRealtime::run(): Overlapped tau: " << (double)step*h/dt << ", setting to 1.0!" << std::endl;
                        tau = 1.0;
                    }
                    double ttau = tau * tau;
                    double tttau = ttau * tau;

                    r_n[i] = r_n[i] * (2 * tttau - 3 * ttau + 1) + t_n[i] * (tttau - 2 * ttau + tau) + r_o[i] * (-2 * tttau + 3 * ttau) + t_o[i] * (tttau - ttau);
                    t_n[i] = r_n[i] * (6 * ttau - 6 * tau) + t_n[i] * (3 * ttau - 4 * tau + 1) + r_o[i] * (-6 * ttau + 6 * tau) + t_o[i] * (3 * ttau - 2 * tau);

                    r_o[i] = r_i[i];
                    t_o[i] = part<3, 0x040201>(((v_w[i]) ^ n_i[i]) * (~n_i[i]) * hermite_dt);
                }

                step = 0;
                dt = hermite_dt;
                newIntersections = false;
            }

            //Cubic hermite approximation
            std::vector<gealg::mv<3, 0x040201>::type> r_w(4);

            r_w[0] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::u_wfl) * cardyn::q_b))(y);
            r_w[1] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::u_wfr) * cardyn::q_b))(y);
            r_w[2] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrl - cardyn::u_wrl) * cardyn::q_b))(y);
            r_w[3] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrr - cardyn::u_wrr) * cardyn::q_b))(y);

            for (int i = 0; i < 4; ++i)
            {
                Vector2D v_c(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
                if (currentRoad[i])
                {
                    Vector3D v_w(-r_w[i][1], r_w[i][0], r_w[i][2]);

                    if (RoadSystem::Instance())
                    {
                        v_c = RoadSystem::Instance()->searchPositionFollowingRoad(v_w, currentRoad[i], currentLongPos[i]);
                    }
                    if (!v_c.isNaV())
                    {
                        RoadPoint point = currentRoad[i]->getRoadPoint(v_c.u(), v_c.v());
                        i_w[i][0] = point.y();
                        i_w[i][1] = -point.x();
                        i_w[i][2] = point.z();
                    }
                    else if (currentLongPos[i] < -0.1 || currentLongPos[i] > currentRoad[i]->getLength() + 0.1)
                    {
                        leftRoad = true;
                    }
                }
                else
                {
                    leftRoad = true;
                }
            }
            ++step;

            gealg::mv<1, 0x04>::type dh[4];
            for (int i = 0; i < 4; ++i)
            {
                dh[i] = part<1, 0x04>((cardyn::q_b * (r_w[i] - i_w[i]) * (!cardyn::q_b)) - cardyn::z * cardyn::r_w)(y);
                if (fabs(dh[i][0]) > 0.15)
                {
                    leftRoad = true;
                }
            }

            if (!leftRoad)
            {
                std::get<35>(y) = dh[0];
                std::get<36>(y) = dh[1];
                std::get<37>(y) = dh[2];
                std::get<38>(y) = dh[3];
            }

            /*std::get<35>(y) = part<1, 0x04>((cardyn::q_b*(r_w[0]-i_w[0])*(!cardyn::q_b))-cardyn::z*cardyn::r_w)(y);
         std::get<36>(y) = part<1, 0x04>((cardyn::q_b*(r_w[1]-i_w[1])*(!cardyn::q_b))-cardyn::z*cardyn::r_w)(y);
         std::get<37>(y) = part<1, 0x04>((cardyn::q_b*(r_w[2]-i_w[2])*(!cardyn::q_b))-cardyn::z*cardyn::r_w)(y);
         std::get<38>(y) = part<1, 0x04>((cardyn::q_b*(r_w[3]-i_w[3])*(!cardyn::q_b))-cardyn::z*cardyn::r_w)(y);*/

            integrator.integrate(h);
        }

        current = 0.0;
        //Motion platform handling
        if (movingToGround)
        {
            steerWheel->setCurrent(0);
            if (motPlat->isGrounded())
            {
                movingToGround = false;
                motPlat->getSendMutex().acquire(period);
                motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
                motPlat->getSendMutex().release();
            }
        }
        else if (returningToAction)
        {
            //if( motPlat->isMiddleLifted() ) {
            if (motPlat->isLifted())
            {
                returningToAction = false;
                pause = false;

                motPlat->getSendMutex().acquire(period);
                //motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();
                motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>();
                for (unsigned int motIt = 0; motIt < motPlat->numLinMots; ++motIt)
                {
                    motPlat->setVelocitySetpoint(motIt, ValidateMotionPlatform::velMax);
                    motPlat->setAccelerationSetpoint(motIt, ValidateMotionPlatform::accMax);
                }
                motPlat->getSendMutex().release();
            }
        }
        else if (!pause)
        {
            double longAngle = atan(((std::get<8>(y)[0] + std::get<10>(y)[0]) * 0.5 - (std::get<4>(y)[0] + std::get<6>(y)[0]) * 0.5) / (cardyn::r_wfl[0] - cardyn::r_wrl[0])) * 0.5;
            double latAngle = atan(((std::get<4>(y)[0] + std::get<8>(y)[0]) * 0.5 - (std::get<6>(y)[0] + std::get<10>(y)[0]) * 0.5) / (cardyn::r_wfl[1] - cardyn::r_wfr[1])) * 0.5;

            /*if(longAngle==longAngle && latAngle==latAngle) {
            motPlat->getSendMutex().acquire(period);
            motPlat->setLongitudinalAngleSetpoint(longAngle);
            motPlat->setLateralAngleSetpoint(latAngle);
            motPlat->getSendMutex().release();
         }*/

            gealg::mv<2, 0x0201>::type r_fl;
            r_fl[0] = 0.41;
            r_fl[1] = 1.0;
            gealg::mv<2, 0x0201>::type r_fr;
            r_fr[0] = 0.41;
            r_fr[1] = -0.3;
            gealg::mv<2, 0x0201>::type r_r;
            r_r[0] = -0.73;
            r_r[1] = 0.35;
            //gealg::mv<1, 0x07>::type d_Pb = ((part<2, 0x0201>(cardyn::p_b) + part<1, 0x04>(cardyn::P_b)) ^ grade<2>(cardyn::P_b))(y);
            //gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(r_fl)*cardyn::q_b))^(grade<2>(cardyn::P_b)))-d_Pb)(y);
            //gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(r_fr)*cardyn::q_b))^(grade<2>(cardyn::P_b)))-d_Pb)(y);
            //gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(r_r)*cardyn::q_b))^(grade<2>(cardyn::P_b)))-d_Pb)(y);
            gealg::mv<1, 0x07>::type d_Pb = (grade<1>(cardyn::p_b) ^ grade<2>(cardyn::P_xy))(y);
            gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fl)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
            gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fr)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
            gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_r)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);

            if (d_P_fl[0] == d_P_fl[0] && d_P_fr[0] == d_P_fr[0] && d_P_r[0] == d_P_r[0])
            {
                motPlat->getSendMutex().acquire(period);
                motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + d_P_fr[0]); //Right
                motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + d_P_fl[0]); //Left
                motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + d_P_r[0]); //Rear
                motPlat->getSendMutex().release();
            }
            //std::cerr << "d_P_fl: " << d_P_fl << std::endl;

            //Steering wheel handling
            current = -4000.0 * (std::get<31>(y)[2] + std::get<32>(y)[2]) - (double)steerWheel->getSmoothedSpeed() * 0.0002;
            //current = -10000.0*(std::get<31>(y)[2]+std::get<32>(y)[2]) - (double)steerWheel->getSmoothedSpeed()*0.0001;
            //double current = 0.0;

            //parking moment model
            /*double drillElasticity = tanh(fabs(yOne.v));
        double drillRigidness = 1.0 - drillElasticity;
        if((driftPosition-steerPosition)>100000*drillRigidness) driftPosition = steerPosition + (int32_t)(100000*drillRigidness);
        else if((driftPosition-steerPosition)<-100000*drillRigidness) driftPosition = steerPosition - (int32_t)(100000*drillRigidness);
        current += (int32_t)((double)(driftPosition-steerPosition)*steerWheel->getDrillConstant());*/
        }
        else
        {
            if (doCenter)
            {
                doCenter = false;
                fprintf(stderr, "center\n");
                steerWheel->center();
                fprintf(stderr, "center done\n");
            }
            current = 0.0;
        }

        /*currentDeque.pop_front();
      currentDeque.push_back(current);
      current = 0;
      for(currentDequeIt=currentDeque.begin(); currentDequeIt!=currentDeque.end(); ++currentDequeIt)
      {
         current += (*currentDequeIt);
      }
      current = current / (double)currentDeque.size();*/
        steerWheel->setCurrent(current);

        steerCon->sendSync();
        steerCon->recvPDO(1);
        steerPosition = steerWheel->getPosition();
        steerWheelAngle = -2 * M_PI * ((double)steerPosition / (double)steerWheel->countsPerTurn);

        double cotSteerAngle = (cardyn::r_wfl[0] - cardyn::r_wrl[0]) * (1.0 / tan(steerWheelAngle * 0.07777778));

        double angleFL = atan(1.0 / (cotSteerAngle - cardyn::r_wfl[1] / (cardyn::r_wfl[0] - cardyn::r_wrl[0])));
        double angleFR = atan(1.0 / (cotSteerAngle - cardyn::r_wfr[1] / (cardyn::r_wfl[0] - cardyn::r_wrl[0])));

        std::get<39>(y)[0] = cos(angleFL * 0.5);
        std::get<39>(y)[1] = sin(angleFL * 0.5);
        std::get<40>(y)[0] = cos(angleFR * 0.5);
        std::get<40>(y)[1] = sin(angleFR * 0.5);

        steerCon->sendPDO();

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
#endif //#ifdef __XENO__

void FourWheelDynamicsRealtime::determineGroundPlane()
{
    gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b) * cardyn::z * cardyn::q_b)(y);

    gealg::mv<1, 0x04>::type Dv_wfl = std::get<35>(y);
    if (Dv_wfl[0] > 0.0)
        Dv_wfl[0] = 0.0;
    gealg::mv<1, 0x04>::type Dv_wfr = std::get<36>(y);
    if (Dv_wfr[0] > 0.0)
        Dv_wfr[0] = 0.0;
    gealg::mv<1, 0x04>::type Dv_wrl = std::get<37>(y);
    if (Dv_wrl[0] > 0.0)
        Dv_wrl[0] = 0.0;
    gealg::mv<1, 0x04>::type Dv_wrr = std::get<38>(y);
    if (Dv_wrr[0] > 0.0)
        Dv_wrr[0] = 0.0;

    gealg::mv<3, 0x040201>::type p_bfl = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl - Dv_wfl) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_bfr = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr - Dv_wfr) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_br = (cardyn::p_b + grade<1>((!cardyn::q_b) * ((cardyn::r_wrl - cardyn::u_wrl - Dv_wrl + cardyn::r_wrr - cardyn::u_wrr - Dv_wrr) * 0.5 - cardyn::z * cardyn::r_w) * cardyn::q_b))(y);

    osg::LineSegment *normalFL = new osg::LineSegment(osg::Vec3(-(p_bfl - n_b * 10.0)()[1], (p_bfl - n_b * 10.0)()[0], (p_bfl - n_b * 10.0)()[2]),
                                                      osg::Vec3(-(p_bfl + n_b * 0.2)()[1], (p_bfl + n_b * 0.2)()[0], (p_bfl + n_b * 0.2)()[2]));
    osg::LineSegment *normalFR = new osg::LineSegment(osg::Vec3(-(p_bfr - n_b * 10.0)()[1], (p_bfr - n_b * 10.0)()[0], (p_bfr - n_b * 10.0)()[2]),
                                                      osg::Vec3(-(p_bfr + n_b * 0.2)()[1], (p_bfr + n_b * 0.2)()[0], (p_bfr + n_b * 0.2)()[2]));
    osg::LineSegment *normalR = new osg::LineSegment(osg::Vec3(-(p_br - n_b * 10.0)()[1], (p_br - n_b * 10.0)()[0], (p_br - n_b * 10.0)()[2]),
                                                     osg::Vec3(-(p_br + n_b * 0.2)()[1], (p_br + n_b * 0.2)()[0], (p_br + n_b * 0.2)()[2]));

    //wheelTransformRR->setPosition(osg::Vec3(-(p_br-n_b)()[1], (p_br-n_b)()[0], (p_br-n_b)()[2]));

    osgUtil::IntersectVisitor visitor;
    visitor.setTraversalMask(Isect::Collision);
    visitor.addLineSegment(normalFL);
    visitor.addLineSegment(normalFR);
    visitor.addLineSegment(normalR);
    cover->getObjectsRoot()->accept(visitor);

    gealg::mv<4, 0x08040201>::type p_ifl;
    gealg::mv<4, 0x08040201>::type p_ifr;
    gealg::mv<4, 0x08040201>::type p_ir;

    if (visitor.getNumHits(normalFL))
    {
        osg::Vec3d intersectFL = visitor.getHitList(normalFL).front().getWorldIntersectPoint();
        p_ifl[0] = intersectFL.y();
        p_ifl[1] = -intersectFL.x();
        p_ifl[2] = intersectFL.z();
        p_ifl[3] = 1.0;
        osg::Node *n = visitor.getHitList(normalFL).front().getNodePath().back();
        if (n)
            std::cerr << "Node: " << n->getName();
        //fprintf(stderr,"Node: %s",n->getName());

        else
            std::cerr << "Node: NoName";
        //fprintf(stderr,"Node: NoName");
    }
    else
    {
        //p_ifl = part<4, 0x08040201>(p_bfl - n_b*10.0); p_ifl[3] = 1.0;
        p_ifl = part<4, 0x08040201>(p_bfl);
        p_ifl[3] = 1.0;
    }
    if (visitor.getNumHits(normalFR))
    {
        osg::Vec3d intersectFR = visitor.getHitList(normalFR).front().getWorldIntersectPoint();
        p_ifr[0] = intersectFR.y();
        p_ifr[1] = -intersectFR.x();
        p_ifr[2] = intersectFR.z();
        p_ifr[3] = 1.0;
    }
    else
    {
        //p_ifr = part<4, 0x08040201>(p_bfr - n_b*10.0); p_ifr[3] = 1.0;
        p_ifr = part<4, 0x08040201>(p_bfr);
        p_ifr[3] = 1.0;
    }
    if (visitor.getNumHits(normalR))
    {
        osg::Vec3d intersectR = visitor.getHitList(normalR).front().getWorldIntersectPoint();
        p_ir[0] = intersectR.y();
        p_ir[1] = -intersectR.x();
        p_ir[2] = intersectR.z();
        p_ir[3] = 1.0;
    }
    else
    {
        //p_ir = part<4, 0x08040201>(p_br - n_b*10.0); p_ir[3] = 1.0;
        p_ir = part<4, 0x08040201>(p_br);
        p_ir[3] = 1.0;
    }

    //wheelTransformFL->setPosition(osg::Vec3(-p_ifl[1], p_ifl[0], p_ifl[2]));
    //wheelTransformFR->setPosition(osg::Vec3(-p_ifr[1], p_ifr[0], p_ifr[2]));
    //wheelTransformRL->setPosition(osg::Vec3(-p_ir[1], p_ir[0], p_ir[2]));

    //std::cerr << "p_ifl: " << p_ifl << ", p_ifr: " << p_ifr << ", p_ir: " << p_ir << std::endl;

    /*if(groundPlaneDeque.size()>20) {
      groundPlaneDeque.pop_front();
   }
   groundPlaneDeque.push_back((p_ir^p_ifr^p_ifl)());
   groundPlane[0] = 0.0; groundPlane[1] = 0.0; groundPlane[2] = 0.0; groundPlane[3] = 0.0;
   for(int i=0; i<groundPlaneDeque.size(); ++i) {
      groundPlane = groundPlane + groundPlaneDeque[i];
   }
   groundPlane = groundPlane*(1.0/(double)groundPlaneDeque.size());*/

    groundPlane = p_ir ^ p_ifr ^ p_ifl;
}

void FourWheelDynamicsRealtime::determineHermite()
{
    double dt = cover->frameDuration() * 1.1;
    //double dt = cover->frameDuration();

    std::vector<gealg::mv<3, 0x040201>::type> r(4);
    std::vector<gealg::mv<3, 0x040201>::type> v(4);

    gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b) * cardyn::z * cardyn::q_b)(y);

    r[0] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::u_wfl) * cardyn::q_b))(y);
    r[1] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::u_wfr) * cardyn::q_b))(y);
    r[2] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrl - cardyn::u_wrl) * cardyn::q_b))(y);
    r[3] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrr - cardyn::u_wrr) * cardyn::q_b))(y);

    v[0] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wfl * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wfl * cardyn::q_b))))(y); //v[0]=(v[0] ^ n_b)*(~n_b);
    v[1] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wfr * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wfr * cardyn::q_b))))(y); //v[1]=(v[1] ^ n_b)*(~n_b);
    v[2] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wrl * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wrl * cardyn::q_b))))(y); //v[2]=(v[2] ^ n_b)*(~n_b);
    v[3] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wrr * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wrr * cardyn::q_b))))(y); //v[3]=(v[3] ^ n_b)*(~n_b);

    osgUtil::IntersectVisitor visitor;
    visitor.setTraversalMask(Isect::Collision);

    std::vector<osg::LineSegment *> normal(4);
    for (int i = 0; i < 4; ++i)
    {
        gealg::mv<3, 0x040201>::type p = r[i] + v[i] * dt;
        normal[i] = new osg::LineSegment(osg::Vec3(-(p - n_b * 5.0)()[1], (p - n_b * 5.0)()[0], (p - n_b * 5.0)()[2]),
                                         osg::Vec3(-(p + n_b * 0.2)()[1], (p + n_b * 0.2)()[0], (p + n_b * 0.2)()[2]));
        visitor.addLineSegment(normal[i]);
    }
    cover->getObjectsRoot()->accept(visitor);

    for (int i = 0; i < 4; ++i)
    {
        if (visitor.getNumHits(normal[i]))
        {
            //osgUtil::Hit& hit = visitor.getHitList(normal[i]).front();
            osgUtil::Hit &hit = visitor.getHitList(normal[i]).back();

            osg::Vec3d intersect = hit.getWorldIntersectPoint();
            r_i[i][0] = intersect.y();
            r_i[i][1] = -intersect.x();
            r_i[i][2] = intersect.z();

            osg::Vec3d normal = hit.getWorldIntersectNormal();
            n_i[i][0] = normal.y();
            n_i[i][1] = -normal.x();
            n_i[i][2] = normal.z();
        }
        else
        {
            n_i[i][0] = 0.0;
            n_i[i][1] = 0.0;
            n_i[i][2] = 1.0;
            r_i[i] = part<3, 0x040201>(r_i[i] + (((r[i] + v[i] * dt) - r_i[i]) ^ n_i[i]) * (~n_i[i]));
        }
    }

    hermite_dt = dt;
    newIntersections = true;
}

gealg::mv<6, 0x060504030201LL>::type FourWheelDynamicsRealtime::getRoadSystemContactPoint(const gealg::mv<3, 0x040201>::type &p_w, Road *&road, double &u)
{
    Vector3D v_w(p_w[0], p_w[1], p_w[2]);

    gealg::mv<6, 0x060504030201LL>::type s_c;
    Vector2D v_c(0.0, 0.0);
    if (RoadSystem::Instance())
    {
        v_c = RoadSystem::Instance()->searchPosition(v_w, road, u);
    }
    else
    {
        std::cerr << "VehicleDynamicsPlugin::getContactPoint(): no road system!" << std::endl;
    }

    if (road)
    {
        RoadPoint point = road->getRoadPoint(v_c.u(), v_c.v());
        s_c[0] = point.x();
        s_c[1] = point.y();
        s_c[3] = point.z();
        s_c[2] = point.nx();
        s_c[4] = -point.ny();
        s_c[5] = point.nz();
        //std::cerr << "Road: " << road->getId() << ", point: " << s_c << std::endl;
    }
    else
    {
        s_c[0] = p_w[0];
        s_c[1] = p_w[1];
        s_c[3] = 0.0;
        s_c[2] = 1.0;
        s_c[4] = 0.0;
        s_c[5] = 0.0;
        //std::cerr << "No road! Point: " << s_c << std::endl;
    }

    return s_c;
}

#ifdef __XENO__
void FourWheelDynamicsRealtime::platformToGround()
{
    pause = true;
    movingToGround = true;
    returningToAction = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
}
void FourWheelDynamicsRealtime::centerWheel()
{
    if (pause)
    {
        doCenter = true;
    }
}

void FourWheelDynamicsRealtime::platformMiddleLift()
{
    pause = true;
    returningToAction = true;
    movingToGround = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlMiddleLift>();
    motPlat->getSendMutex().release();
}

void FourWheelDynamicsRealtime::platformReturnToAction()
{
    pause = true;
    returningToAction = true;
    movingToGround = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();
    motPlat->getSendMutex().release();

    gealg::mv<2, 0x0201>::type r_fl;
    r_fl[0] = 0.41;
    r_fl[1] = 1.0;
    gealg::mv<2, 0x0201>::type r_fr;
    r_fr[0] = 0.41;
    r_fr[1] = -0.3;
    gealg::mv<2, 0x0201>::type r_r;
    r_r[0] = -0.73;
    r_r[1] = 0.35;
    //gealg::mv<1, 0x07>::type d_Pb = ((part<2, 0x0201>(cardyn::p_b) + part<1, 0x04>(cardyn::P_b)) ^ grade<2>(cardyn::P_b))(y);
    //gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(r_fl)*cardyn::q_b))^(grade<2>(cardyn::P_b)))-d_Pb)(y);
    //gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(r_fr)*cardyn::q_b))^(grade<2>(cardyn::P_b)))-d_Pb)(y);
    //gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(r_r)*cardyn::q_b))^(grade<2>(cardyn::P_b)))-d_Pb)(y);
    gealg::mv<1, 0x07>::type d_Pb = (grade<1>(cardyn::p_b) ^ grade<2>(cardyn::P_xy))(y);
    gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fl)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
    gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fr)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
    gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_r)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);

    if (d_P_fl[0] == d_P_fl[0] && d_P_fr[0] == d_P_fr[0] && d_P_r[0] == d_P_r[0])
    {
        motPlat->getSendMutex().acquire(period);
        motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + d_P_fr[0]); //Right
        motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + d_P_fl[0]); //Left
        motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + d_P_r[0]); //Rear

        motPlat->setVelocitySetpoint(0, ValidateMotionPlatform::velMax * 0.1);
        motPlat->setVelocitySetpoint(1, ValidateMotionPlatform::velMax * 0.1);
        motPlat->setVelocitySetpoint(2, ValidateMotionPlatform::velMax * 0.1);

        motPlat->setAccelerationSetpoint(0, ValidateMotionPlatform::accMax * 0.1);
        motPlat->setAccelerationSetpoint(1, ValidateMotionPlatform::accMax * 0.1);
        motPlat->setAccelerationSetpoint(2, ValidateMotionPlatform::accMax * 0.1);

        motPlat->getSendMutex().release();
    }
}

std::pair<Road *, Vector2D> FourWheelDynamicsRealtime::getStartPositionOnRoad()
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

#endif
