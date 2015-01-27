/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///----------------------------------
///Author: Florian Seybold, 2009
///www.hlrs.de
///----------------------------------
///Cube body connected to two fixed points via pull and torsion springs, gravity acting on cube body.
///
/// Operators: Geometric Product: *, Inner Product: &, Outer Product: ^.
/// Attention with inner and outer product: Best enclose operation with operands in brackets, because of C++ order of operations.
#define _USE_MATH_DEFINES
#include "gaalet.h"

#include <osgViewer/Viewer>
#include <osgGA/StateSetManipulator>
#include <osgGA/TrackballManipulator>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>

typedef gaalet::algebra<gaalet::signature<4, 1> > cm;

int main()
{
    //defintion of basisvectors, null basis, pseudoscalars, helper unit scalar
    cm::mv<0x01>::type e1(1.0);
    cm::mv<0x02>::type e2(1.0);
    cm::mv<0x04>::type e3(1.0);
    cm::mv<0x08>::type ep(1.0);
    cm::mv<0x10>::type em(1.0);

    cm::mv<0x00>::type one(1.0);

    cm::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
    cm::mv<0x08, 0x10>::type einf = em + ep;

    cm::mv<0x18>::type E = ep * em;

    cm::mv<0x1f>::type I = e1 * e2 * e3 * ep * em;
    cm::mv<0x07>::type i = e1 * e2 * e3;

    typedef cm::mv<0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type S_type;
    typedef cm::mv<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x0f, 0x11, 0x12, 0x14, 0x17>::type D_type;

    //Initial position of cube
    cm::mv<1, 2, 4>::type m(0.0, 0.0, 0.2 * M_PI);
    cm::mv<1, 2, 4>::type n(1.0, 1.0, 0.0);

    //Expressions: Screw of pure translation, corresponding versor
    cm::mv<0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type S_n = einf * n;
    cm::mv<0x00, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type T = one + S_n * 0.5;
    std::cout << "T: " << T << std::endl;

    //Expressions: Screw of pure rotation, corresponding versor
    //auto S_m = i*m*(-1.0);
    //auto R = exp(S_m*0.5);
    cm::mv<0, 3, 5, 6>::type R = one * cos(-0.5 * 0.2 * M_PI) + e1 * e2 * sin(-0.5 * 0.2 * M_PI) + e2 * e3 * 0.0 + e3 * e1 * 0.0;
    std::cout << "R: " << R << std::endl;

    //Displacement versor, pay attention to the evaluation of the expression: D is a multivector, no expression.
    D_type D = T * R;
    std::cout << "D: " << D << std::endl;

    //Initial velocity of cube
    cm::mv<1, 2, 4>::type omega(0.0, 0.0, 0.0);
    cm::mv<1, 2, 4>::type v(0.0, 0.0, 0.0);

    //Velocity twist: evaluation to a multivector
    S_type V_b = i * omega * (-1.0) + einf * v;

    //Mass and principal inertias
    double M = 1.0;
    double In_1 = 0.4;
    double In_2 = 0.4;
    double In_3 = 0.7;

    //Springs' stiffness' and dampings
    double k_s1 = 50.0;
    double k_s2 = 20.0;
    double k_d = 0.5;

    //Displacements of fixed points connected to cube via springs
    cm::mv<0, 9, 0xa, 0xc, 0x11, 0x12, 0x14>::type T_s1 = (one + einf * (e1 * 1.0 + e2 * 0.0 + e3 * 1.0) * 0.5);
    cm::mv<0, 9, 0xa, 0xc, 0x11, 0x12, 0x14>::type T_s2 = (one + einf * (e1 * 0.0 + e2 * 0.0 + e3 * 0.0) * 0.5);
    cm::mv<0, 5>::type R_s1 = one * cos(-M_PI * 0.25 * 0.5) + e3 * e1 * sin(-M_PI * 0.25 * 0.5);
    cm::mv<0, 5>::type R_s2 = one * cos(-M_PI * 0.25 * 0.5) + e3 * e1 * sin(-M_PI * 0.25 * 0.5);
    D_type D_s1 = T_s1 * R_s1;
    D_type D_s2 = T_s2 * R_s2;

    //Visualisation with OpenSceneGraph
    osg::Group *sceneRoot = new osg::Group;

    osg::Box *cube = new osg::Box(osg::Vec3(0, 0, 0), 0.2f);
    osg::ShapeDrawable *cubeDrawable = new osg::ShapeDrawable(cube);
    osg::Geode *cubeGeode = new osg::Geode();
    cubeGeode->addDrawable(cubeDrawable);
    osg::PositionAttitudeTransform *cubeTransform = new osg::PositionAttitudeTransform();
    cubeTransform->addChild(cubeGeode);
    sceneRoot->addChild(cubeTransform);

    cm::mv<1, 2, 4, 8, 16>::type p_s1 = grade<1>(D_s1 * e0 * (~D_s1));
    osg::Sphere *s1Sphere = new osg::Sphere(osg::Vec3(p_s1[0], p_s1[1], p_s1[2]), 0.1f);
    osg::ShapeDrawable *s1SphereDrawable = new osg::ShapeDrawable(s1Sphere);
    osg::Geode *s1SphereGeode = new osg::Geode();
    s1SphereGeode->addDrawable(s1SphereDrawable);
    sceneRoot->addChild(s1SphereGeode);

    cm::mv<1, 2, 4, 8, 16>::type p_s2 = eval(grade<1>(D_s2 * e0 * (~D_s2)));
    osg::Sphere *s2Sphere = new osg::Sphere(osg::Vec3(p_s2[0], p_s2[1], p_s2[2]), 0.1f);
    osg::ShapeDrawable *s2SphereDrawable = new osg::ShapeDrawable(s2Sphere);
    osg::Geode *s2SphereGeode = new osg::Geode();
    s2SphereGeode->addDrawable(s2SphereDrawable);
    sceneRoot->addChild(s2SphereGeode);

    osgViewer::Viewer viewer;
    viewer.setSceneData(sceneRoot);
    viewer.addEventHandler(new osgGA::StateSetManipulator(viewer.getCamera()->getOrCreateStateSet()));
    if (!viewer.getCameraManipulator() && viewer.getCamera()->getAllowEventFocus())
    {
        viewer.setCameraManipulator(new osgGA::TrackballManipulator());
    }
    viewer.setReleaseContextAtEndOfFrameHint(false);

    if (!viewer.isRealized())
    {
        viewer.realize();
    }

    //Animation loop: Integration of equations of motion. Generelly Euler-backwards, exception: implicit Euler-forward for solving Euler's equations
    double frameTime = 0.0;
    double sumFrameTime = 0.0;
    unsigned int counter = 0;
    while (!viewer.done())
    {
        double minFrameTime = 0.0;
        osg::Timer_t startFrameTick = osg::Timer::instance()->tick();

        //Displacement propagation
        //auto dD = part<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x0f, 0x11, 0x12, 0x14, 0x17>(D*V_b*0.5);
        D_type dD = part_type<D_type>(D * V_b * 0.5);
        D = D + dD * frameTime;

        //Generalized trigonometric formula for the tangent of a half angle (after Hestenes)
        S_type B_s1 = part_type<S_type>(grade<2>(grade<2>((~D) * D_s1) * (!(one + grade<0>((~D) * D_s1) + grade<4>((~D) * D_s1)))));
        S_type B_s2 = part_type<S_type>(grade<2>(grade<2>((~D) * D_s2) * (!(one + grade<0>((~D) * D_s2) + grade<4>((~D) * D_s2)))));

        //Force law of spring (not necessarily linear)
        S_type F_s1_b = B_s1 * k_s1;
        S_type F_s2_b = B_s2 * k_s2;
        //Force law of damping
        S_type F_d = V_b * k_d * (-1.0);
        //Gravity acting on body
        S_type F_g = part_type<S_type>((~D) * einf * (e3 * (-9.81)) * D);
        //Resultant force wrench
        S_type F_b = F_s1_b + F_s2_b + F_d + F_g;

        //--- Start: velocity propagation due to force laws ---
        //Torque part of force wrench
        cm::mv<1, 2, 4>::type t_b = i * part_type<cm::mv<0x03, 0x05, 0x06>::type>(F_b);
        //Linear force part of force wrench
        cm::mv<1, 2, 4>::type f_b = grade<1>(part_type<cm::mv<0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type>(F_b) * e0);

        //Linear velocity part of velocity twist propagation
        cm::mv<1, 2, 4>::type v = grade<1>(part_type<cm::mv<0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type>(V_b) * e0) + f_b * (frameTime / M);

        //Angular velocity part of velocity twist propagation (implicit Euler-forward, solving Euler's equations)
        cm::mv<1, 2, 4>::type oldOm = eval((~i) * (-1.0) * part_type<cm::mv<0x03, 0x05, 0x06>::type>(V_b));
        cm::mv<1, 2, 4>::type om = oldOm;
        cm::mv<1, 2, 4>::type prevOm;
        double maxError = 1e-5;
        //do {
        for (int j = 0; j < 5; ++j)
        {
            prevOm = om;
            om[0] = oldOm[0] + (t_b[0] - (In_3 - In_2) * om[1] * om[2]) / In_1 * frameTime;
            om[1] = oldOm[1] + (t_b[1] - (In_1 - In_3) * om[2] * om[0]) / In_2 * frameTime;
            om[2] = oldOm[2] + (t_b[2] - (In_2 - In_1) * om[0] * om[1]) / In_3 * frameTime;
        }
        //} while((magnitude(om-prevOm).element<0x00>()) > maxError);

        //Combining velocity propagations to velocity twist
        V_b = i * om * (-1.0) + einf * v;
        //--- End: velocity propagation due to force laws ---

        //Updating new position of cube
        cm::mv<1, 2, 4, 8, 16>::type p_m = eval(grade<1>(D * e0 * (~D)));
        cubeTransform->setPosition(osg::Vec3(p_m[0], p_m[1], p_m[2]));
        cubeTransform->setAttitude(osg::Quat(-D[3], D[2], -D[1], D[0]));

        viewer.frame();

        //work out if we need to force a sleep to hold back the frame rate
        osg::Timer_t endFrameTick = osg::Timer::instance()->tick();
        frameTime = osg::Timer::instance()->delta_s(startFrameTick, endFrameTick);

        sumFrameTime += frameTime;
        if (counter == 1000)
        {
            std::cout << "Average frame time: " << sumFrameTime / 1000.0 << std::endl;
            sumFrameTime = 0.0;
            counter = 0;
        }
        else
        {
            counter++;
        }

        if (frameTime < minFrameTime)
            OpenThreads::Thread::microSleep(static_cast<unsigned int>(1000000.0 * (minFrameTime - frameTime)));
    }

    return 0;
}
