/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FourWheelDynamicsRealtime.h"

#include "GasPedal.h"
#include <fasi.h>

FourWheelDynamicsRealtime::FourWheelDynamicsRealtime()
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("FourWheelDynamicsRealtimeTask", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("FourWheelDynamicsRealtimeTask", 0, 99, T_FPU | T_CPU(5))
#endif
    , dy(cardyn::getExpressionVector())
    , integrator(dy, y)
    , r_i(4)
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

    //i_proj[0] = 1.0;

    runTask = true;
    doCenter = false;
    taskFinished = false;
    returningToAction = false;
    movingToGround = false;
    pause = true;
    overruns = 0;
    motPlat = ValidateMotionPlatform::instance();

    steerCon = new CanOpenController("can1");
    steerWheel = new XenomaiSteeringWheel(*steerCon, 1);

    start();
    k_wf_Slider = 17400.0;
    k_wr_Slider = 26100.0;
    d_wf_Slider = 2600.0;
    d_wr_Slider = 2600.0;
    clutchPedal = 0.0;
}

FourWheelDynamicsRealtime::~FourWheelDynamicsRealtime()
{
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
}

void FourWheelDynamicsRealtime::initState()
{
    y = cardyn::StateVectorType();

    if (startPos.first)
    {
        Transform transform = startPos.first->getRoadTransform(startPos.second.u(), startPos.second.v());
        gealg::mv<4, 0x06050300>::type R_b;
        R_b[0] = transform.q().w();
        R_b[1] = transform.q().z();
        R_b[2] = -transform.q().y();
        R_b[3] = transform.q().x();
        gealg::mv<4, 0x06050300>::type R_xodr = exp(0.5 * (-0.5 * M_PI * cardyn::x * cardyn::y));
        std::tr1::get<2>(y) = !(R_b * R_xodr);

        gealg::mv<3, 0x040201>::type p_b_init;
        p_b_init[2] = 0.75;
        gealg::mv<3, 0x040201>::type p_road;
        p_road[0] = transform.v().y();
        p_road[1] = -transform.v().x();
        p_road[2] = transform.v().z();
        std::tr1::get<0>(y) = p_road + grade<1>((!std::tr1::get<2>(y)) * p_b_init * (std::tr1::get<2>(y)));

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
        std::cout << "\t p_b: " << std::tr1::get<0>(y) << ", R_b: " << std::tr1::get<2>(y) << std::endl;
    }
    else
    {
        std::tr1::get<0>(y)[2] = -0.2; //Initial position
        std::tr1::get<2>(y)[0] = 1.0; //Initial orientation (Important: magnitude be one!)
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
    std::tr1::get<39>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::tr1::get<40>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::tr1::get<41>(y)[0] = cardyn::i_a; //Initial steering wheel position: magnitude be one!

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
    cardyn::d_wf.e_(y)[0] = 2600.0;
    cardyn::d_wr.e_(y)[0] = 2600.0;

    newIntersections = false;
    rpms = 0.0;
}

void FourWheelDynamicsRealtime::setVehicleTransformation(const osg::Matrix &m)
{
    resetState();

    std::tr1::get<0>(y)[0] = -m(3, 2);
    std::tr1::get<0>(y)[1] = -m(3, 0);
    std::tr1::get<0>(y)[2] = m(3, 1);

    std::cout << "Reset: position: " << std::tr1::get<0>(y) << std::endl;
}

void FourWheelDynamicsRealtime::resetState()
{
    if (!pause)
    {
        pause = true;
        initState();
        platformReturnToAction();
    }
    else
    {
        initState();
    }
}

void FourWheelDynamicsRealtime::move()
{
    y_frame = this->y;

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

    gealg::mv<3, 0x040201>::type r_bg = (cardyn::r_wfl + cardyn::r_wfr + cardyn::r_wrl + cardyn::r_wrr) * 0.25 - cardyn::z * cardyn::r_w;
    gealg::mv<3, 0x040201>::type p_bg = (cardyn::p_b + grade<1>((!cardyn::q_b) * r_bg * cardyn::q_b))(y_frame);

    chassisTrans.setTrans(osg::Vec3(-p_bg[1], p_bg[2], -p_bg[0]));
    chassisTrans.setRotate(osg::Quat(std::tr1::get<2>(y_frame)[2],
                                     std::tr1::get<2>(y_frame)[1],
                                     -std::tr1::get<2>(y_frame)[3],
                                     std::tr1::get<2>(y_frame)[0]));

    //vehicle->setVRMLVehicle(chassisTrans);

    cardyn::k_wf.e_(y)[0] = k_wf_Slider;
    cardyn::d_wf.e_(y)[0] = d_wf_Slider;
    cardyn::k_wr.e_(y)[0] = k_wr_Slider;
    cardyn::d_wr.e_(y)[0] = d_wr_Slider;
}

void FourWheelDynamicsRealtime::setSportDamper(bool sport)
{
    if (sport)
    {
        /*k_wf_Slider = 94000.0;
d_wf_Slider = 152000.0;
k_wr_Slider = 4600.0;
d_wr_Slider = 4600.0;*/
        k_wf_Slider = 94000.0;
        k_wr_Slider = 152000.0;
        d_wf_Slider = 4600.0;
        d_wr_Slider = 4600.0;
    }
    else
    {
        /*k_wf_Slider = 17400.0;
d_wf_Slider = 26100.0;
k_wr_Slider = 2600.0;
d_wr_Slider = 2600.0;*/
        k_wf_Slider = 17400.0;
        k_wr_Slider = 26100.0;
        d_wf_Slider = 2600.0;
        d_wr_Slider = 2600.0;
    }
}

void FourWheelDynamicsRealtime::run()
{
    double current = 0.0;
    std::deque<double> currentDeque(10, 0.0);
    std::deque<double>::iterator currentDequeIt;

    std::cerr << "--- steerWheel->init(); ---" << std::endl;
    steerWheel->init();

    std::cerr << "--- FourWheelDynamicsRealtime::FourWheelDynamicsRealtime(): Starting ValidateMotionPlatform task ---" << std::endl;
    //Motion platform
    motPlat->start();
    std::cerr << "--- motPlat->start();  ---" << std::endl;
    while (!motPlat->isInitialized())
    {
        rt_task_sleep(1000000);
        std::cerr << "--- motPlat->waiting for initialization();  ---" << std::endl;
    }
    std::cerr << "--- motPlat->start(); done ---" << std::endl;
    set_periodic(period);
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
    while (!motPlat->isGrounded())
    {
        rt_task_wait_period(&overruns);
    }
    std::cerr << "--- isGrounded(); done ---" << std::endl;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();

    double dt = hermite_dt;
    int step = 0;
    static bool oldleftRoad = false;

    while (runTask)
    {
        if (overruns != 0)
        {
            std::cerr << "FourWheelDynamicsRealtimeRealtime::run(): overruns: " << overruns << std::endl;
            overruns = 0;
        }
        bool leftRoadOnce = false;
        if (oldleftRoad != leftRoad)
        {
            oldleftRoad = leftRoad;
            if (leftRoad)
                leftRoadOnce = true;
        }

        gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b) * cardyn::z * cardyn::q_b)(y);
        gealg::mv<1, 0x0>::type proj_n_z = (n_b % cardyn::z);
        if ((proj_n_z[0] < 0.0) || leftRoad)
        {
            if (leftRoadOnce)
            {
                std::cout << "Left Road!" << std::endl;
            }
            else
            {
                std::cout << "reset" << std::endl;
            }
            resetState();
        }
	if (!pause)
        {
            double h = (double)(period * (overruns + 1)) * 1e-9;

            std::tr1::get<42>(y)[0] = GasPedal::instance()->getActualAngle() / 100.0;
            std::tr1::get<43>(y)[0] = motPlat->getBrakeForce() * 200.0;
            std::tr1::get<44>(y)[0] = (1.0 - clutchPedal) * cardyn::k_cn;
            int gear = fasi::instance()->sharedState.gear;
            std::tr1::get<41>(y)[0] = cardyn::i_g[gear + 1] * cardyn::i_a;
            if (gear == 0)
            {
                std::tr1::get<44>(y)[0] = 0.0;
            }
            //std::cout << "clutch!" << std::tr1::get<44>(y)[0] << "Gas " << std::tr1::get<42>(y)[0]  << "Brake " << std::tr1::get<43>(y)[0] << std::endl;
            //std::cout << "gear" << gear  << std::endl;

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
                std::cout << "hermite_dt" << hermite_dt << std::endl;
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
                    { // left road searching for the next road over all roads in the system
                        currentRoad[i] = NULL;
                        v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad[i], currentLongPos[i]);
                        std::cout << "search road from scratch" << std::endl;
                        if (!v_c.isNaV())
                        {
                            RoadPoint point = currentRoad[i]->getRoadPoint(v_c.u(), v_c.v());
                            i_w[i][0] = point.y();
                            i_w[i][1] = -point.x();
                            i_w[i][2] = point.z();
                        }
                        else
                        {
                            leftRoad = true;
                        }
                    }
                }
                else
                {
                    // left road searching for the next road over all roads in the system
                    currentRoad[i] = NULL;
                    Vector3D v_w(-r_w[i][1], r_w[i][0], r_w[i][2]);
                    v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad[i], currentLongPos[i]);
                    std::cout << "search road from scratch2" << std::endl;
                    if (!v_c.isNaV())
                    {
                        RoadPoint point = currentRoad[i]->getRoadPoint(v_c.u(), v_c.v());
                        i_w[i][0] = point.y();
                        i_w[i][1] = -point.x();
                        i_w[i][2] = point.z();
                    }
                    else
                    {
                        leftRoad = true;
                    }
                }
            }
            ++step;

            gealg::mv<1, 0x04>::type dh[4];
            for (int i = 0; i < 4; ++i)
            {
                dh[i] = part<1, 0x04>((cardyn::q_b * (r_w[i] - i_w[i]) * (!cardyn::q_b)) - cardyn::z * cardyn::r_w)(y);
                if (fabs(dh[i][0]) > 0.15)
                {
                    std::cout << "left road for unknown reason" << std::endl;
                    leftRoad = true;
                }
            }

            if (!leftRoad)
            {
                std::tr1::get<35>(y) = dh[0];
                std::tr1::get<36>(y) = dh[1];
                std::tr1::get<37>(y) = dh[2];
                std::tr1::get<38>(y) = dh[3];
            }
            else
            {
                //std::tr1::get<35>(y) = 0;
                // std::tr1::get<36>(y) = 0;
                //std::tr1::get<37>(y) = 0;
                // std::tr1::get<38>(y) = 0;
            }

            //  std::cout << "h" << h  << std::endl;
            //std::cout << "dh[0]: " << dh[0] << ", dh[1]: " << dh[1] << ", dh[2]: " << dh[2] << ", dh[3]: " << dh[3] << std::endl;
            integrator.integrate(h);
        }
        current = 0.0;
        if(pause || movingToGround || returningToAction)
        {
            float hr = -ValidateMotionPlatform::posMiddle+ motPlat->getPosition(0);
            float hl = -ValidateMotionPlatform::posMiddle+ motPlat->getPosition(1);
            float hh = -ValidateMotionPlatform::posMiddle+ motPlat->getPosition(2);
            osg::Vec3 left(-ValidateMotionPlatform::sideMotDist,0,hl);
            osg::Vec3 right(ValidateMotionPlatform::sideMotDist,0,hr);
            osg::Vec3 rear(0,-ValidateMotionPlatform::rearMotDist,hh);
            osg::Vec3 toRight = ((left - right)/2.0);
            osg::Vec3 middle =  left + toRight;
            osg::Vec3 toFront = rear - middle;
            toRight.normalize();
            toFront.normalize();
            osg::Vec3 up = toRight ^ toFront;
            osg::Vec3 y = toRight ^ up;
            osg::Matrix m;
            m.makeTranslate(middle);
            m(0,0) = toRight[0];
            m(1,0) = toRight[1];
            m(2,0) = toRight[2];
            m(0,1) = y[0];
            m(1,1) = y[1];
            m(2,1) = y[2];
            m(0,2) = up[0];
            m(1,2) = up[1];
            m(2,2) = up[2];
        }
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

            double longAngle = atan(((std::tr1::get<8>(y)[0] + std::tr1::get<10>(y)[0]) * 0.5 - (std::tr1::get<4>(y)[0] + std::tr1::get<6>(y)[0]) * 0.5) / (cardyn::r_wfl[0] - cardyn::r_wrl[0])) * 0.5;
            double latAngle = atan(((std::tr1::get<4>(y)[0] + std::tr1::get<8>(y)[0]) * 0.5 - (std::tr1::get<6>(y)[0] + std::tr1::get<10>(y)[0]) * 0.5) / (cardyn::r_wfl[1] - cardyn::r_wfr[1])) * 0.5;

            gealg::mv<2, 0x0201>::type r_fl;
            r_fl[0] = 0.41;
            r_fl[1] = 1.0;
            gealg::mv<2, 0x0201>::type r_fr;
            r_fr[0] = 0.41;
            r_fr[1] = -0.3;
            gealg::mv<2, 0x0201>::type r_r;
            r_r[0] = -0.73;
            r_r[1] = 0.35;
            gealg::mv<1, 0x07>::type d_Pb = (grade<1>(cardyn::p_b) ^ grade<2>(cardyn::P_xy))(y);
            gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fl)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
            gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fr)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
            gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_r)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);

            if (d_P_fl[0] == d_P_fl[0] && d_P_fr[0] == d_P_fr[0] && d_P_r[0] == d_P_r[0])
            {
                motPlat->getSendMutex().acquire(period);
                //Right
                motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + d_P_fr[0]);
                //Left
                motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + d_P_fl[0]);
                //Rear
                motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + d_P_r[0]);
                motPlat->getSendMutex().release();
            }
            //std::cerr << "d_P_fl: " << d_P_fl << std::endl;
            //std::cerr << "p_b: " << cardyn::p_b(y) << std::endl;
            //std::cerr << "q_b: " << cardyn::q_b(y) << std::endl;
            //std::cerr << "k_wf: " << cardyn::k_wf(y) << ", k_wr: " << cardyn::k_wr(y) << std::endl;

            //Steering wheel handling
            current = -4000.0 * (std::tr1::get<31>(y)[2] + std::tr1::get<32>(y)[2]) - (double)steerWheel->getSmoothedSpeed() * 0.0002;

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
        //if(fabs(current)>300)
        //fprintf(stderr,"current %f\n",current);
        steerWheel->setCurrent(current);

        steerCon->sendSync();
        steerCon->recvPDO(1);
        steerPosition = steerWheel->getPosition();
        steerWheelAngle = -2 * M_PI * ((double)steerPosition / (double)steerWheel->countsPerTurn);

        double cotSteerAngle = (cardyn::r_wfl[0] - cardyn::r_wrl[0]) * (1.0 / tan(steerWheelAngle * 0.07777778));

        double angleFL = atan(1.0 / (cotSteerAngle - cardyn::r_wfl[1] / (cardyn::r_wfl[0] - cardyn::r_wrl[0])));
        double angleFR = atan(1.0 / (cotSteerAngle - cardyn::r_wfr[1] / (cardyn::r_wfl[0] - cardyn::r_wrl[0])));

        std::tr1::get<39>(y)[0] = cos(angleFL * 0.5);
        std::tr1::get<39>(y)[1] = sin(angleFL * 0.5);
        std::tr1::get<40>(y)[0] = cos(angleFR * 0.5);
        std::tr1::get<40>(y)[1] = sin(angleFR * 0.5);

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
    gealg::mv<1, 0x07>::type d_Pb = (grade<1>(cardyn::p_b) ^ grade<2>(cardyn::P_xy))(y);
    gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fl)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
    gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fr)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
    gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_r)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);

    if (d_P_fl[0] == d_P_fl[0] && d_P_fr[0] == d_P_fr[0] && d_P_r[0] == d_P_r[0])
    {
        motPlat->getSendMutex().acquire(period);
        //Right
        motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + d_P_fr[0]);
        //Left
        motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + d_P_fl[0]);
        //Rear
        motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + d_P_r[0]);

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
