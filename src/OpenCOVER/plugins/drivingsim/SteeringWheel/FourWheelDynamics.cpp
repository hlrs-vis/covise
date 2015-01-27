/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FourWheelDynamics.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

FourWheelDynamics::FourWheelDynamics()
    : dy(cardyn::getExpressionVector())
    , integrator(dy, y)
    , r_i(4)
    , n_i(4)
    , newIntersections(false)
    , hermite_dt(0.02)
{
    std::tr1::get<0>(y)[2] = 1.0; //Initial position
    //std::tr1::get<1>(y)[0] = 1.0;    //Initial velocity
    //std::tr1::get<1>(y)[1] = 5.0;    //Initial velocity
    std::tr1::get<2>(y)[0] = 1.0; //Initial orientation (Important: magnitude be one!)
    //std::tr1::get<2>(y)[0] = 0.982131;  std::tr1::get<2>(y)[2] = 0.188203;   //Initial orientation (Important: magnitude be one!)
    //std::tr1::get<2>(y)[0] = cos(0.5*M_PI); std::tr1::get<2>(y)[1] = sin(0.5*M_PI);   //Initial orientation (Important: magnitude be one!)
    //std::tr1::get<3>(y)[1] = -0.3;    //Initial angular velocity
    //std::tr1::get<3>(y)[2] = 0.3;    //Initial angular velocity
    //std::tr1::get<32>(y)[0] = M_PI*0.1;    //Initial steering wheel position
    //std::tr1::get<33>(y)[0] = 10.0;    //Permanent torque on rear wheels
    std::tr1::get<37>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::tr1::get<38>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::tr1::get<39>(y)[0] = cardyn::i_a; //Initial steering wheel position: magnitude be one!

    i_proj[0] = 1.0;

    osg::Sphere *wheelShape = new osg::Sphere(osg::Vec3(0, 0, 0), 0.5);
    osg::Drawable *wheelDrawable = new osg::ShapeDrawable(wheelShape);
    osg::Geode *wheelGeode = new osg::Geode();
    wheelGeode->addDrawable(wheelDrawable);
    wheelTransformFL = new osg::PositionAttitudeTransform();
    wheelTransformFL->addChild(wheelGeode);
    wheelTransformFL->setNodeMask(wheelTransformFL->getNodeMask() & (~Isect::Collision));
    cover->getObjectsRoot()->addChild(wheelTransformFL);
    wheelTransformFR = new osg::PositionAttitudeTransform();
    wheelTransformFR->addChild(wheelGeode);
    wheelTransformFR->setNodeMask(wheelTransformFR->getNodeMask() & (~Isect::Collision));
    cover->getObjectsRoot()->addChild(wheelTransformFR);
    wheelTransformRL = new osg::PositionAttitudeTransform();
    wheelTransformRL->addChild(wheelGeode);
    wheelTransformRL->setNodeMask(wheelTransformRL->getNodeMask() & (~Isect::Collision));
    cover->getObjectsRoot()->addChild(wheelTransformRL);
    wheelTransformRR = new osg::PositionAttitudeTransform();
    wheelTransformRR->addChild(wheelGeode);
    wheelTransformRR->setNodeMask(wheelTransformRR->getNodeMask() & (~Isect::Collision));
    cover->getObjectsRoot()->addChild(wheelTransformRR);

    doRun = true;

    int affin_return = setProcessorAffinity(1);
    if (affin_return < 0)
    {
        std::cerr << "FourWheelDynamics::FourWheelDynamics(): couldn't set processor affinity, error code: " << affin_return << std::endl;
    }
    start();
}

FourWheelDynamics::~FourWheelDynamics()
{
    doRun = false;
    endBarrier.block(2);
}

void FourWheelDynamics::move(VrmlNodeVehicle *vehicle)
{
    //determineGroundPlane();
    determineHermite();

    /*gealg::mv<3, 0x040201>::type p_wfl = (cardyn::p_b+grade<1>((!cardyn::q_b)*(cardyn::r_wfl-cardyn::u_wfl-cardyn::z*(cardyn::r_w))*cardyn::q_b))(y);
     osg::Vec2 wheelPosFL(-p_wfl[1], p_wfl[0]);
     gealg::mv<3, 0x040201>::type p_wfr = (cardyn::p_b+grade<1>((!cardyn::q_b)*(cardyn::r_wfr-cardyn::u_wfr-cardyn::z*(cardyn::r_w))*cardyn::q_b))(y);
     osg::Vec2 wheelPosFR(-p_wfr[1], p_wfr[0]);
     gealg::mv<3, 0x040201>::type p_wrl = (cardyn::p_b+grade<1>((!cardyn::q_b)*(cardyn::r_wrl-cardyn::u_wrl-cardyn::z*(cardyn::r_w))*cardyn::q_b))(y);
     osg::Vec2 wheelPosRL(-p_wrl[1], p_wrl[0]);
     gealg::mv<3, 0x040201>::type p_wrr = (cardyn::p_b+grade<1>((!cardyn::q_b)*(cardyn::r_wrr-cardyn::u_wrr-cardyn::z*(cardyn::r_w))*cardyn::q_b))(y);
     osg::Vec2 wheelPosRR(-p_wrr[1], p_wrr[0]);

     wheelTransformFL->setPosition(osg::Vec3(-p_wfl[1], p_wfl[0], p_wfl[2]));
     wheelTransformFR->setPosition(osg::Vec3(-p_wfr[1], p_wfr[0], p_wfr[2]));
     wheelTransformRL->setPosition(osg::Vec3(-p_wrl[1], p_wrl[0], p_wrl[2]));
     wheelTransformRR->setPosition(osg::Vec3(-p_wrr[1], p_wrr[0], p_wrr[2]));

     double e_wfl = p_wfl[2];
     double e_wfr = p_wfr[2];
     double e_wrl = p_wrl[2];
     double e_wrr = p_wrr[2];

     getWheelElevation( wheelPosFL, wheelPosFR, wheelPosRL, wheelPosRR,
     e_wfl, e_wfr, e_wrl, e_wrr);

     std::tr1::get<32>(y)[0] = p_wfl[2] - e_wfl;
     std::tr1::get<33>(y)[0] = p_wfr[2] - e_wfr;
     std::tr1::get<34>(y)[0] = p_wrl[2] - e_wrl;
     std::tr1::get<35>(y)[0] = p_wrr[2] - e_wrr;*/

    gealg::mv<3, 0x040201>::type p_bg = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::z * (-cardyn::u_wn - cardyn::r_w)) * cardyn::q_b))(y);

    chassisTrans.setTrans(osg::Vec3(-p_bg[1], p_bg[2], -p_bg[0]));
    //chassisTrans.setTrans(osg::Vec3(-std::tr1::get<0>(y)[1], std::tr1::get<0>(y)[2], -std::tr1::get<0>(y)[0]));
    chassisTrans.setRotate(osg::Quat(std::tr1::get<2>(y)[2],
                                     std::tr1::get<2>(y)[1],
                                     -std::tr1::get<2>(y)[3],
                                     std::tr1::get<2>(y)[0]));

    //vehicle->moveToStreet(chassisTrans);
    vehicle->setVRMLVehicle(chassisTrans);

    /*osg::Quat steerRot(std::tr1::get<32>(y)[0], osg::Vec3(0,0,1));

   osg::Matrix wheelMatrixFL;
   wheelMatrixFL.setTrans(osg::Vec3(cardyn::v_wn, cardyn::w_wn, -cardyn::u_wn - std::tr1::get<4>(y)[0]));
   osg::Quat wheelRotarySpeedFL(0.0, -std::tr1::get<12>(y)[0], 0.0, 0.0);
   wheelQuatFL = wheelQuatFL + wheelQuatFL*wheelRotarySpeedFL*(0.5*dt);
   wheelQuatFL = wheelQuatFL*(1/wheelQuatFL.length());
   wheelMatrixFL.setRotate(wheelQuatFL*steerRot);

   osg::Matrix wheelMatrixFR;
   wheelMatrixFR.setTrans(osg::Vec3(cardyn::v_wn, -cardyn::w_wn, -cardyn::u_wn - std::tr1::get<6>(y)[0]));
   osg::Quat wheelRotarySpeedFR(0.0, -std::tr1::get<13>(y)[0], 0.0, 0.0);
   wheelQuatFR = wheelQuatFR + wheelQuatFR*wheelRotarySpeedFR*(0.5*dt);
   wheelQuatFR = wheelQuatFR*(1/wheelQuatFR.length());
   wheelMatrixFR.setRotate(wheelQuatFR*steerRot);

   osg::Matrix wheelMatrixRL;
   wheelMatrixRL.setTrans(osg::Vec3(-cardyn::v_wn, cardyn::w_wn, -cardyn::u_wn - std::tr1::get<8>(y)[0]));
   osg::Quat wheelRotarySpeedRL(0.0, -std::tr1::get<14>(y)[0], 0.0, 0.0);
   wheelQuatRL = wheelQuatRL + wheelQuatRL*wheelRotarySpeedRL*(0.5*dt);
   wheelQuatRL = wheelQuatRL*(1/wheelQuatRL.length());
   wheelMatrixRL.setRotate(wheelQuatRL);

   osg::Matrix wheelMatrixRR;
   wheelMatrixRR.setTrans(osg::Vec3(-cardyn::v_wn, -cardyn::w_wn, -cardyn::u_wn - std::tr1::get<10>(y)[0]));
   osg::Quat wheelRotarySpeedRR(0.0, -std::tr1::get<15>(y)[0], 0.0, 0.0);
   wheelQuatRL = wheelQuatRR + wheelQuatRR*wheelRotarySpeedRR*(0.5*dt);
   wheelQuatRL = wheelQuatRR*(1/wheelQuatRR.length());
   wheelMatrixRR.setRotate(wheelQuatRR);

   vehicle->setVRMLVehicleFrontWheels(wheelMatrixFL, wheelMatrixFR);

   vehicle->setVRMLVehicleRearWheels(wheelMatrixRL, wheelMatrixRR);*/
}

void FourWheelDynamics::run()
{
    microSleep(2000000);

    std::vector<gealg::mv<3, 0x040201>::type> r_n(4); //Start position of hermite
    std::vector<gealg::mv<3, 0x040201>::type> t_n(4); //Start tangent of hermite
    std::vector<gealg::mv<3, 0x040201>::type> r_o(4); //End position of hermite
    std::vector<gealg::mv<3, 0x040201>::type> t_o(4); //End tangent of hermite
    double dt = hermite_dt;
    int step = 0;

    r_n[0] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl) * cardyn::q_b))(y);
    r_o[0] = r_n[0];
    r_n[1] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr) * cardyn::q_b))(y);
    r_o[1] = r_n[1];
    r_n[2] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrl - cardyn::z * cardyn::r_w - cardyn::u_wrl) * cardyn::q_b))(y);
    r_o[2] = r_n[2];
    r_n[3] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrr - cardyn::z * cardyn::r_w - cardyn::u_wrr) * cardyn::q_b))(y);
    r_o[3] = r_n[3];

    while (doRun)
    {
        std::tr1::get<40>(y)[0] = InputDevice::instance()->getAccelerationPedal();
        std::tr1::get<41>(y)[0] = InputDevice::instance()->getBrakePedal() * 90000.0;
        std::tr1::get<42>(y)[0] = (1.0 - InputDevice::instance()->getClutchPedal()) * cardyn::k_cn;
        int gear = InputDevice::instance()->getGear();
        std::tr1::get<39>(y)[0] = cardyn::i_g[gear + 1] * cardyn::i_a;
        if (gear == 0)
        {
            std::tr1::get<42>(y)[0] = 0.0;
        }
        double steerWheelAngle = InputDevice::instance()->getSteeringWheelAngle();

        double cotSteerAngle = 2.0 * cardyn::v_wn * (1.0 / tan(steerWheelAngle * 0.07777778));

        double angleFL = atan(1.0 / (cotSteerAngle - cardyn::w_wn / (cardyn::v_wn * 2.0)));
        double angleFR = atan(1.0 / (cotSteerAngle + cardyn::w_wn / (cardyn::v_wn * 2.0)));
        std::tr1::get<37>(y)[0] = cos(angleFL * 0.5);
        std::tr1::get<37>(y)[1] = sin(angleFL * 0.5);
        std::tr1::get<38>(y)[0] = cos(angleFR * 0.5);
        std::tr1::get<38>(y)[1] = sin(angleFR * 0.5);

        for (int i = 0; i < 10; ++i)
        {
            /*gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b)*cardyn::z*cardyn::q_b)(y);

         gealg::mv<4, 0x08040201>::type p_wfl = part<4, 0x08040201>((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfl-cardyn::z*cardyn::r_w-cardyn::u_wfl)*cardyn::q_b)))(y); p_wfl[3] = 1.0;
         gealg::mv<4, 0x08040201>::type m_wfl = ((groundPlane%(!i_proj)) ^ ((p_wfl^(p_wfl+n_b))%(!i_proj)))%i_proj;
         gealg::mv<3, 0x040201>::type i_wfl; i_wfl[0] = m_wfl[0]/m_wfl[3]; i_wfl[1] = m_wfl[1]/m_wfl[3]; i_wfl[2] = m_wfl[2]/m_wfl[3];

         gealg::mv<4, 0x08040201>::type p_wfr = part<4, 0x08040201>((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfr-cardyn::z*cardyn::r_w-cardyn::u_wfr)*cardyn::q_b)))(y); p_wfr[3] = 1.0;
         gealg::mv<4, 0x08040201>::type m_wfr = ((groundPlane%(!i_proj)) ^ ((p_wfr^(p_wfr+n_b))%(!i_proj)))%i_proj;
         gealg::mv<3, 0x040201>::type i_wfr; i_wfr[0] = m_wfr[0]/m_wfr[3]; i_wfr[1] = m_wfr[1]/m_wfr[3]; i_wfr[2] = m_wfr[2]/m_wfr[3];

         gealg::mv<4, 0x08040201>::type p_wrl = part<4, 0x08040201>((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wrl-cardyn::z*cardyn::r_w-cardyn::u_wrl)*cardyn::q_b)))(y); p_wrl[3] = 1.0;
         gealg::mv<4, 0x08040201>::type m_wrl = ((groundPlane%(!i_proj)) ^ ((p_wrl^(p_wrl+n_b))%(!i_proj)))%i_proj;
         gealg::mv<3, 0x040201>::type i_wrl; i_wrl[0] = m_wrl[0]/m_wrl[3]; i_wrl[1] = m_wrl[1]/m_wrl[3]; i_wrl[2] = m_wrl[2]/m_wrl[3];

         gealg::mv<4, 0x08040201>::type p_wrr = part<4, 0x08040201>((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wrr-cardyn::z*cardyn::r_w-cardyn::u_wrr)*cardyn::q_b)))(y); p_wrr[3] = 1.0;
         gealg::mv<4, 0x08040201>::type m_wrr = ((groundPlane%(!i_proj)) ^ ((p_wrr^(p_wrr+n_b))%(!i_proj)))%i_proj;
         gealg::mv<3, 0x040201>::type i_wrr; i_wrr[0] = m_wrr[0]/m_wrr[3]; i_wrr[1] = m_wrr[1]/m_wrr[3]; i_wrr[2] = m_wrr[2]/m_wrr[3];
         //std::cout << "i_wfl: " << i_wfl << ", i_wfr: " << i_wfr << ", i_wrl: " << i_wrl << ", i_wrr: " << i_wrr << std::endl;*/

            //Cubic hermite parameter determination
            if (newIntersections)
            {
                std::vector<gealg::mv<3, 0x040201>::type> v_w(4);

                v_w[0] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wfl * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wfl * cardyn::q_b))))(y);
                v_w[1] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wfr * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wfr * cardyn::q_b))))(y);
                v_w[2] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wrl * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wrl * cardyn::q_b))))(y);
                v_w[3] = (((cardyn::dp_b + grade<1>((!cardyn::q_b) * cardyn::r_wrr * cardyn::q_b * cardyn::w_b - (!cardyn::q_b) * cardyn::du_wrr * cardyn::q_b))))(y);

                double tau = (double)step * 0.01 / dt;
                if (tau > 1.0)
                    tau = 1.0;
                double ttau = tau * tau;
                double tttau = ttau * tau;

                for (int i = 0; i < 4; ++i)
                {

                    r_n[i] = r_n[i] * (2 * tttau - 3 * ttau + 1) + t_n[i] * (tttau - 2 * ttau + tau) + r_o[i] * (-2 * tttau + 3 * ttau) + t_o[i] * (tttau - ttau);
                    t_n[i] = r_n[i] * (6 * ttau - 6 * tau) + t_n[i] * (3 * ttau - 4 * tau + 1) + r_o[i] * (-6 * ttau + 6 * tau) + t_o[i] * (3 * ttau - 2 * tau);

                    r_o[i] = r_i[i];
                    t_o[i] = ((v_w[i]) ^ n_i[i]) * (~n_i[i]) * hermite_dt;
                    //std::cerr << "Wheel " << i << ": tau: " << tau << ", r_n: " << r_n[i] << ", t_n: " << t_n[i] << ", r_o: " << r_o[i] << ", t_o: " << t_o[i] << std::endl;
                    //std::cerr << "p_b: " << std::tr1::get<0>(y) << ", dp_b: " << std::tr1::get<1>(y) << std::endl;
                    //std::cerr << "wfl: r_n: " << r_n[0] << ", t_n: " << t_n[0] << ", r_o: " << r_o[0] << ", t_o: " << t_o[0] << ", tau: " << tau << std::endl;
                }

                step = 0;
                dt = hermite_dt;
                newIntersections = false;
            }

            //Cubic hermite approximation
            std::vector<gealg::mv<3, 0x040201>::type> r_w(4);

            r_w[0] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl) * cardyn::q_b))(y);
            r_w[1] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr) * cardyn::q_b))(y);
            r_w[2] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrl - cardyn::z * cardyn::r_w - cardyn::u_wrl) * cardyn::q_b))(y);
            r_w[3] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrr - cardyn::z * cardyn::r_w - cardyn::u_wrr) * cardyn::q_b))(y);

            double tau = (double)step * 0.01 / dt;
            if (tau > 1.0)
                tau = 1.0;
            double ttau = tau * tau;
            double tttau = ttau * tau;

            std::vector<gealg::mv<3, 0x040201>::type> i_w(4);
            for (int i = 0; i < 4; ++i)
            {
                i_w[i] = r_n[i] * (2 * tttau - 3 * ttau + 1) + t_n[i] * (tttau - 2 * ttau + tau) + r_o[i] * (-2 * tttau + 3 * ttau) + t_o[i] * (tttau - ttau);
                //std::cerr << "Wheel " << i << ": tau: " << tau << ", i_w: " << i_w[i] << std::endl;
            }
            ++step;

            std::tr1::get<33>(y)[0] = (cardyn::q_b * (r_w[0] - i_w[0]) * (!cardyn::q_b))(y)[2];
            std::tr1::get<34>(y)[0] = (cardyn::q_b * (r_w[1] - i_w[1]) * (!cardyn::q_b))(y)[2];
            std::tr1::get<35>(y)[0] = (cardyn::q_b * (r_w[2] - i_w[2]) * (!cardyn::q_b))(y)[2];
            std::tr1::get<36>(y)[0] = (cardyn::q_b * (r_w[3] - i_w[3]) * (!cardyn::q_b))(y)[2];

            //std::cerr << "p_b: " << std::tr1::get<0>(y) << ", dp_b: " << std::tr1::get<1>(y) << ", dist wfl: " << std::tr1::get<33>(y)[0] << ", wfr: " << std::tr1::get<34>(y)[0] << ", wrl: " << std::tr1::get<35>(y)[0] << ", wrr: " << std::tr1::get<36>(y)[0] << std::endl;

            integrator.integrate(0.001);
        }

        microSleep(10000);
    }

    endBarrier.block(2);
}

void FourWheelDynamics::determineGroundPlane()
{
    gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b) * cardyn::z * cardyn::q_b)(y);

    gealg::mv<3, 0x040201>::type p_bfl = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_bfr = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_br = (cardyn::p_b + grade<1>((!cardyn::q_b) * ((cardyn::r_wrl - cardyn::u_wrl + cardyn::r_wrr - cardyn::u_wrr) * 0.5 - cardyn::z * cardyn::r_w) * cardyn::q_b))(y);

    osg::LineSegment *normalFL = new osg::LineSegment(osg::Vec3(-(p_bfl - n_b)()[1], (p_bfl - n_b)()[0], (p_bfl - n_b)()[2]), osg::Vec3(-(p_bfl + n_b)()[1], (p_bfl + n_b)()[0], (p_bfl + n_b)()[2]));
    osg::LineSegment *normalFR = new osg::LineSegment(osg::Vec3(-(p_bfr - n_b)()[1], (p_bfr - n_b)()[0], (p_bfr - n_b)()[2]), osg::Vec3(-(p_bfr + n_b)()[1], (p_bfr + n_b)()[0], (p_bfr + n_b)()[2]));
    osg::LineSegment *normalR = new osg::LineSegment(osg::Vec3(-(p_br - n_b)()[1], (p_br - n_b)()[0], (p_br - n_b)()[2]), osg::Vec3(-(p_br + n_b)()[1], (p_br + n_b)()[0], (p_br + n_b)()[2]));

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

    groundPlane = p_ir ^ p_ifr ^ p_ifl;
}

void FourWheelDynamics::determineHermite()
{
    double dt = cover->frameDuration() * 1.5;

    std::vector<gealg::mv<3, 0x040201>::type> r(4);
    std::vector<gealg::mv<3, 0x040201>::type> v(4);

    gealg::mv<3, 0x040201>::type n_b = grade<1>((!cardyn::q_b) * cardyn::z * cardyn::q_b)(y);

    r[0] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl) * cardyn::q_b))(y);
    r[1] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr) * cardyn::q_b))(y);
    r[2] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrl - cardyn::z * cardyn::r_w - cardyn::u_wrl) * cardyn::q_b))(y);
    r[3] = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrr - cardyn::z * cardyn::r_w - cardyn::u_wrr) * cardyn::q_b))(y);

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
        normal[i] = new osg::LineSegment(osg::Vec3(-(p - n_b * 10.0)()[1], (p - n_b * 10.0)()[0], (p - n_b * 10.0)()[2]),
                                         osg::Vec3(-(p + n_b * 0.2)()[1], (p + n_b * 0.2)()[0], (p + n_b * 0.2)()[2]));
        visitor.addLineSegment(normal[i]);
    }
    cover->getObjectsRoot()->accept(visitor);

    for (int i = 0; i < 4; ++i)
    {
        if (visitor.getNumHits(normal[i]))
        {
            osgUtil::Hit &hit = visitor.getHitList(normal[i]).front();

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
            r_i[i] = r_i[i] + v[i] * dt;
        }
        //std::cerr << "Wheel " << i << ": r:" << r[i] << ", v: " << v[i] << ",r_i: " << r_i[i] << ", n_i: " << n_i[i] << std::endl;
    }

    hermite_dt = dt;
    newIntersections = true;
}
