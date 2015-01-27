/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <ctime>

#include "CarDynamicsCGA.h"
#include "RungeKuttaClassic.h"

#include <osgViewer/Viewer>
#include <osgGA/StateSetManipulator>
#include <osgGA/TrackballManipulator>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osg/Texture2D>
#include <osgGA/GUIEventHandler>

class CarSteeringHandler : public osgGA::GUIEventHandler
{
public:
    CarSteeringHandler()
        : GUIEventHandler()
        , steerAngle(0.0)
        , gear(0)
        , gasPedal(0.0)
        , brakePedal(0.0)
    {
    }

    double getSteeringAngle()
    {
        return steerAngle;
    }
    int getGear()
    {
        return gear;
    }
    double getGasPedal()
    {
        return gasPedal;
    }
    double getBrakePedal()
    {
        return brakePedal;
    }

    virtual bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &)
    {
        if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
        {
            switch (ea.getKey())
            {
            case 65361:
                steerAngle += 0.1;
                break;
            case 65363:
                steerAngle -= 0.1;
                break;
            case 65362:
                gasPedal = 1.0;
                break;
            case 65364:
                brakePedal = 1.0;
                break;
            }
            return true;
        }
        else if (ea.getEventType() == osgGA::GUIEventAdapter::KEYUP)
        {
            switch (ea.getKey())
            {
            case 65362:
                gasPedal = 0.0;
                break;
            case 65364:
                brakePedal = 0.0;
                break;
            case 103:
                gear += 1;
                if (gear > 5)
                    gear = 5;
                break;
            case 102:
                gear -= 1;
                if (gear < -1)
                    gear = -1;
                break;
            }
            return true;
        }

        return false;
    }

    /*virtual void accept(osgGA::GUIEventHandlerVisitor& v) {
      v.visit(*this);
   }*/

protected:
    double steerAngle;
    int gear;
    double gasPedal;
    double brakePedal;
};

int main()
{
    cardyn::InputVector z;
    cardyn::OutputVector o;
    cardyn::StateVector y;

    magicformula2004::TyrePropertyPack tyrePropLeft(magicformula2004::TyrePropertyPack::TYRE_LEFT);
    magicformula2004::TyrePropertyPack tyrePropRight(magicformula2004::TyrePropertyPack::TYRE_RIGHT);
    magicformula2004::ContactWrench tyreFL(tyrePropLeft);
    magicformula2004::ContactWrench tyreFR(tyrePropRight);
    magicformula2004::ContactWrench tyreRL(tyrePropLeft);
    magicformula2004::ContactWrench tyreRR(tyrePropRight);
    cardyn::StateEquation f(z, o, tyreFL, tyreFR, tyreRL, tyreRR);

    //EulerIntegrator<cardyn::StateEquation, cardyn::StateVector> integrator(f);
    RungeKuttaClassicIntegrator<cardyn::StateEquation, cardyn::StateVector> integrator(f);

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

    //Set initial values
    cardyn::cm::mv<1, 2, 4>::type p_b = { 0.0, 0.0, 0.75 };
    auto T_b = cardyn::one + cardyn::einf * p_b * 0.5;
    D_b = T_b;

    V_b = cardyn::einf * 10.0 * cardyn::e2;

    //dp_b[0] = 0.0;
    //q_b[0] = 1.0;
    a_steer = 0.0;
    i_pt = 0.0;
    s_gp = 0.0;

    //Ground planes
    P_wfl = (-1.0) * cardyn::e3;
    P_wfr = (-1.0) * cardyn::e3;
    P_wrl = (-1.0) * cardyn::e3;
    P_wrr = (-1.0) * cardyn::e3;

    //Visualisation with OpenSceneGraph
    osg::Group *sceneRoot = new osg::Group;

    osg::Cylinder *wheelShape = new osg::Cylinder(osg::Vec3(0, 0, 0), f.r_w, 0.3);
    osg::ShapeDrawable *wheelDrawable = new osg::ShapeDrawable(wheelShape);
    osg::Geode *wheelGeode = new osg::Geode();
    wheelGeode->addDrawable(wheelDrawable);
    osg::PositionAttitudeTransform *wheelTransform = new osg::PositionAttitudeTransform();
    wheelTransform->addChild(wheelGeode);
    wheelTransform->setAttitude(osg::Quat(-M_PI / 2, osg::Vec3(1, 0, 0)));
    osg::Image *wheelTexImage = osgDB::readImageFile("wheel.png");
    if (wheelTexImage)
    {
        osg::Texture2D *wheelTex = new osg::Texture2D;
        wheelTex->setImage(wheelTexImage);
        wheelTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP);
        wheelTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
        osg::StateSet *wheelGeodeState = wheelGeode->getOrCreateStateSet();
        wheelGeodeState->setTextureAttributeAndModes(0, wheelTex);
    }

    osg::PositionAttitudeTransform *wheelTransformFL = new osg::PositionAttitudeTransform();
    wheelTransformFL->addChild(wheelTransform);
    osg::PositionAttitudeTransform *wheelTransformFR = new osg::PositionAttitudeTransform();
    wheelTransformFR->addChild(wheelTransform);
    osg::PositionAttitudeTransform *wheelTransformRL = new osg::PositionAttitudeTransform();
    wheelTransformRL->addChild(wheelTransform);
    osg::PositionAttitudeTransform *wheelTransformRR = new osg::PositionAttitudeTransform();
    wheelTransformRR->addChild(wheelTransform);

    osg::Box *bodyShape = new osg::Box(osg::Vec3(0, 0, 0), f.r_wfl[0] - f.r_wrl[0] + 1.0, f.r_wfl[1] - f.r_wfr[1], -2.0 * f.r_wfl[2]);
    osg::ShapeDrawable *bodyDrawable = new osg::ShapeDrawable(bodyShape);
    osg::Geode *bodyGeode = new osg::Geode();
    bodyGeode->addDrawable(bodyDrawable);
    osg::Image *bodyTexImage = osgDB::readImageFile("body.png");
    if (bodyTexImage)
    {
        osg::Texture2D *bodyTex = new osg::Texture2D;
        bodyTex->setImage(bodyTexImage);
        bodyTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP);
        bodyTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
        osg::StateSet *bodyGeodeState = bodyGeode->getOrCreateStateSet();
        bodyGeodeState->setTextureAttributeAndModes(0, bodyTex);
    }

    osg::Box *cabineShape = new osg::Box(osg::Vec3((f.r_wfl[0] - f.r_wrl[0] + 1.0) * 0.2, 0, -2.0 * f.r_wfl[2]),
                                         (f.r_wfl[0] - f.r_wrl[0]) * 0.4, (f.r_wfl[1] - f.r_wfr[1]) * 0.8, -2.0 * f.r_wfl[2]);
    osg::ShapeDrawable *cabineDrawable = new osg::ShapeDrawable(cabineShape);
    osg::Geode *cabineGeode = new osg::Geode();
    cabineGeode->addDrawable(cabineDrawable);
    osg::Image *cabineTexImage = osgDB::readImageFile("window.png");
    if (cabineTexImage)
    {
        osg::Texture2D *cabineTex = new osg::Texture2D;
        cabineTex->setImage(cabineTexImage);
        cabineTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP);
        cabineTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
        osg::StateSet *cabineGeodeState = cabineGeode->getOrCreateStateSet();
        cabineGeodeState->setTextureAttributeAndModes(0, cabineTex);
    }

    osg::PositionAttitudeTransform *bodyTransform = new osg::PositionAttitudeTransform();
    bodyTransform->addChild(bodyGeode);
    bodyTransform->addChild(cabineGeode);
    sceneRoot->addChild(bodyTransform);

    bodyTransform->addChild(wheelTransformFL);
    bodyTransform->addChild(wheelTransformFR);
    bodyTransform->addChild(wheelTransformRL);
    bodyTransform->addChild(wheelTransformRR);

    //Wheel rotation animation helpers
    osg::Quat wheelQuatFL;
    osg::Quat wheelQuatFR;
    osg::Quat wheelQuatRL;
    osg::Quat wheelQuatRR;

    //Setting up viewer
    osgViewer::Viewer viewer;
    viewer.setSceneData(sceneRoot);

    CarSteeringHandler *carHandler = new CarSteeringHandler();
    viewer.addEventHandler(carHandler);

    viewer.addEventHandler(new osgGA::StateSetManipulator(viewer.getCamera()->getOrCreateStateSet()));
    if (!viewer.getCameraManipulator() && viewer.getCamera()->getAllowEventFocus())
    {
        viewer.setCameraManipulator(new osgGA::TrackballManipulator());
    }
    viewer.setReleaseContextAtEndOfFrameHint(false);

    //Camera control
    osg::Vec3 bodyCameraPosition(-15.0, 0.0, 1.0);
    osg::Vec3 bodyLookAtCenterPosition(0.0, 0.0, 1.0);

    if (!viewer.isRealized())
    {
        viewer.realize();
    }

    //Animation loop: Integration of equations of motion.
    double frameTime = 0.001;
    while (!viewer.done())
    {
        double minFrameTime = 0.0;
        osg::Timer_t startFrameTick = osg::Timer::instance()->tick();

        /*std::cout << "e0: "<< cardyn::e0 << ", D_b*e0*~D_b: " << grade<1>(D_b*cardyn::e0*~D_b) << std::endl;
      std::cout << "\tr_wfl: "<< f.r_wfl << ", D_b*f.r_wfl*~D_b: " << grade<1>(D_b*f.r_wfl*~D_b) << std::endl;
      std::cout << "\tr_wfr: "<< f.r_wfr << ", D_b*f.r_wfr*~D_b: " << grade<1>(D_b*f.r_wfr*~D_b) << std::endl;
      std::cout << "\tr_wrl: "<< f.r_wrl << ", D_b*f.r_wrl*~D_b: " << grade<1>(D_b*f.r_wrl*~D_b) << std::endl;
      std::cout << "\tr_wrr: "<< f.r_wrr << ", D_b*f.r_wrr*~D_b: " << grade<1>(D_b*f.r_wrr*~D_b) << std::endl;*/

        //Setting distance between ground and wheel reference contact point
        double r_w = f.r_w;
        /*std::get<0>(z) = eval((std::get<0>(y) + grade<1>((!std::get<2>(y))*(f.r_wfl-f.z*r_w-f.z*std::get<4>(y))*std::get<2>(y)))&f.z);
      std::get<1>(z) = eval((std::get<0>(y) + grade<1>((!std::get<2>(y))*(f.r_wfr-f.z*r_w-f.z*std::get<6>(y))*std::get<2>(y)))&f.z);
      std::get<2>(z) = eval((std::get<0>(y) + grade<1>((!std::get<2>(y))*(f.r_wrl-f.z*r_w-f.z*std::get<8>(y))*std::get<2>(y)))&f.z);
      std::get<3>(z) = eval((std::get<0>(y) + grade<1>((!std::get<2>(y))*(f.r_wrr-f.z*r_w-f.z*std::get<10>(y))*std::get<2>(y)))&f.z);*/
        auto T_wfl = cardyn::one + cardyn::einf * ((-u_wfl) * cardyn::e3) * 0.5;
        auto T_wfr = cardyn::one + cardyn::einf * ((-u_wfr) * cardyn::e3) * 0.5;
        auto T_wrl = cardyn::one + cardyn::einf * ((-u_wrl) * cardyn::e3) * 0.5;
        auto T_wrr = cardyn::one + cardyn::einf * ((-u_wrr) * cardyn::e3) * 0.5;

        /*d_wfl = eval(grade<1>((D_b*T_wfl)*f.r_wfl* ~(D_b*T_wfl))&cardyn::e3);
      d_wfr = eval(grade<1>((D_b*T_wfr)*f.r_wfr* ~(D_b*T_wfr))&cardyn::e3);
      d_wrl = eval(grade<1>((D_b*T_wrl)*f.r_wrl* ~(D_b*T_wrl))&cardyn::e3);
      d_wrr = eval(grade<1>((D_b*T_wrr)*f.r_wrr* ~(D_b*T_wrr))&cardyn::e3);*/

        //Set steering angle
        a_steer = carHandler->getSteeringAngle();
        if (a_steer < (-M_PI * 0.4))
            a_steer = -M_PI * 0.4;
        if (a_steer > (M_PI * 0.4))
            a_steer = M_PI * 0.4;

        //Gear, gas, brake
        std::get<5>(z) = f.i_g[carHandler->getGear() + 1] * f.i_a;
        std::get<6>(z) = carHandler->getGasPedal();
        std::get<7>(z) = carHandler->getBrakePedal() * 90000.0;

        //Propagate vehicle state
        integrator(frameTime, y);

        //Set new body displacements
        p_b = grade<1>(D_b * cardyn::e0 * ~D_b);
        bodyTransform->setPosition(osg::Vec3(p_b[0], p_b[1], p_b[2]));
        bodyTransform->setAttitude(osg::Quat(-D_b[3],
                                             D_b[2],
                                             -D_b[1],
                                             D_b[0]));

        const double &steerAngle = std::get<4>(z);

        /*double cotSteerAngle = (f.r_wfl[0]-f.r_wrl[0])*(1.0/tan(steerAngle));
      double angleFL = atan(1.0/(cotSteerAngle - f.w_wn/(f.v_wn*2.0)));
      double angleFR = atan(1.0/(cotSteerAngle + f.w_wn/(f.v_wn*2.0)));
      gaalet::mv<0,3>::type q_wfl = {cos(angleFL*0.5), sin(angleFL*0.5)};
      gaalet::mv<0,3>::type q_wfr = {cos(angleFR*0.5), sin(angleFR*0.5)};*/

        //osg::Quat steerRotFL(0,0,q_wfl[1], q_wfl[0]);
        //osg::Quat steerRotFR(0,0,q_wfr[1], q_wfr[0]);

        //auto R_wfl = eval(f.R_n_wfl*exp(cardyn::e2*cardyn::e3*u_wfl*(-0.5)));
        //auto R_wfr = eval(f.R_n_wfr*exp(cardyn::e2*cardyn::e3*u_wfr*(0.5)));
        //auto R_wrl = eval(f.R_n_wrl*exp(cardyn::e2*cardyn::e3*u_wrl*(-0.5)));
        //auto R_wrr = eval(f.R_n_wrr*exp(cardyn::e2*cardyn::e3*u_wrr*(0.5)));
        auto R_wfl = eval(~part<0, 3, 5, 6>(D_wfl));
        auto R_wfr = eval(~part<0, 3, 5, 6>(D_wfr));
        auto R_wrl = eval(~part<0, 3, 5, 6>(D_wrl));
        auto R_wrr = eval(~part<0, 3, 5, 6>(D_wrr));
        osg::Quat camberRotFL(R_wfl[3], -R_wfl[2], R_wfl[1], R_wfl[0]);
        osg::Quat camberRotFR(R_wfr[3], -R_wfr[2], R_wfr[1], R_wfr[0]);
        osg::Quat camberRotRL(R_wrl[3], -R_wrl[2], R_wrl[1], R_wrl[0]);
        osg::Quat camberRotRR(R_wrr[3], -R_wrr[2], R_wrr[1], R_wrr[0]);

        //std::cout << "R_wfl: " << R_wfl << ", " << grade<1>((!R_wfl)*cardyn::e1*R_wfl) << std::endl;
        //wheelTransformFL->setPosition(osg::Vec3(f.v_wn, f.w_wn, -f.u_wn - u_wfl));
        auto p_wfl = eval(grade<1>(D_wfl * cardyn::e0 * ~D_wfl));
        wheelTransformFL->setPosition(osg::Vec3(p_wfl[0], p_wfl[1], p_wfl[2]));
        osg::Quat wheelRotarySpeedFL(0.0, w_wfl, 0.0, 0.0);
        wheelQuatFL = wheelQuatFL + wheelQuatFL * wheelRotarySpeedFL * (0.5 * frameTime);
        wheelQuatFL = wheelQuatFL * (1 / wheelQuatFL.length());
        wheelTransformFL->setAttitude(wheelQuatFL * camberRotFL);

        //wheelTransformFR->setPosition(osg::Vec3(f.v_wn, -f.w_wn, -f.u_wn - u_wfr));
        auto p_wfr = eval(grade<1>(D_wfr * cardyn::e0 * ~D_wfr));
        wheelTransformFR->setPosition(osg::Vec3(p_wfr[0], p_wfr[1], p_wfr[2]));
        osg::Quat wheelRotarySpeedFR(0.0, w_wfr, 0.0, 0.0);
        wheelQuatFR = wheelQuatFR + wheelQuatFR * wheelRotarySpeedFR * (0.5 * frameTime);
        wheelQuatFR = wheelQuatFR * (1 / wheelQuatFR.length());
        wheelTransformFR->setAttitude(wheelQuatFR * camberRotFR);

        //wheelTransformRL->setPosition(osg::Vec3(-f.v_wn, f.w_wn, -f.u_wn - u_wrl));
        auto p_wrl = eval(grade<1>(D_wrl * cardyn::e0 * ~D_wrl));
        wheelTransformRL->setPosition(osg::Vec3(p_wrl[0], p_wrl[1], p_wrl[2]));
        osg::Quat wheelRotarySpeedRL(0.0, w_wrl, 0.0, 0.0);
        wheelQuatRL = wheelQuatRL + wheelQuatRL * wheelRotarySpeedRL * (0.5 * frameTime);
        wheelQuatRL = wheelQuatRL * (1 / wheelQuatRL.length());
        wheelTransformRL->setAttitude(wheelQuatRL * camberRotRL);

        //wheelTransformRR->setPosition(osg::Vec3(-f.v_wn, -f.w_wn, -f.u_wn - u_wrr));
        auto p_wrr = eval(grade<1>(D_wrr * cardyn::e0 * ~D_wrr));
        wheelTransformRR->setPosition(osg::Vec3(p_wrr[0], p_wrr[1], p_wrr[2]));
        osg::Quat wheelRotarySpeedRR(0.0, w_wrr, 0.0, 0.0);
        wheelQuatRR = wheelQuatRR + wheelQuatRR * wheelRotarySpeedRR * (0.5 * frameTime);
        wheelQuatRR = wheelQuatRR * (1 / wheelQuatRR.length());
        wheelTransformRR->setAttitude(wheelQuatRR * camberRotRR);

        //Set camera viewer matrix
        /*viewer.getCamera()->setViewMatrixAsLookAt(bodyTransform->getPosition() + bodyCameraPosition,
                                    bodyTransform->getPosition() + bodyLookAtCenterPosition,
                                    osg::Vec3(0.0, 0.0, 1.0));*/

        //Render
        viewer.frame();

        //Work out if we need to force a sleep to hold back the frame rate
        osg::Timer_t endFrameTick = osg::Timer::instance()->tick();
        frameTime = osg::Timer::instance()->delta_s(startFrameTick, endFrameTick);
        if (frameTime < minFrameTime)
            OpenThreads::Thread::microSleep(static_cast<unsigned int>(1000000.0 * (minFrameTime - frameTime)));
        if (frameTime > 0.001)
            frameTime = 0.001;

        //OpenThreads::Thread::microSleep(static_cast<unsigned int>(10000.0));
        //frameTime = 0.001;
    }

    return 0;
}
