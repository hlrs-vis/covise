/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///----------------------------------
///Author: Florian Seybold, 2010
///www.hlrs.de
///----------------------------------

#include "cga_osg.h"

#include <osgViewer/Viewer>
#include <osgGA/StateSetManipulator>
#include <osgGA/TrackballManipulator>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/Depth>
#include <osg/Material>

using namespace cga;

int main()
{
    ::osg::Group *sceneRoot = new ::osg::Group;

    ::osg::Group *transparentSpheres = new ::osg::Group;
    sceneRoot->addChild(transparentSpheres);
    ::osg::StateSet *tsStateSet = transparentSpheres->getOrCreateStateSet();
    // Enable blending, select transparent bin.
    tsStateSet->setMode(GL_BLEND, ::osg::StateAttribute::ON);
    tsStateSet->setRenderingHint(::osg::StateSet::TRANSPARENT_BIN);

    // Enable depth test so that an opaque polygon will occlude a transparent one behind it.
    tsStateSet->setMode(GL_DEPTH_TEST, ::osg::StateAttribute::ON);

    // Conversely, disable writing to depth buffer so that
    // a transparent polygon will allow polygons behind it to shine thru.
    // OSG renders transparent polygons after opaque ones.
    ::osg::Depth *depth = new ::osg::Depth;
    depth->setWriteMask(false);
    tsStateSet->setAttributeAndModes(depth, ::osg::StateAttribute::ON);

    // Disable conflicting modes.
    //tsStateSet->setMode( GL_LIGHTING, ::osg::StateAttribute::OFF );

    ::osg::Material *tsMat = (::osg::Material *)tsStateSet->getAttribute(::osg::StateAttribute::MATERIAL);
    if (!tsMat)
    {
        double opacity = 0.2;
        tsMat = new ::osg::Material;
        tsMat->setAlpha(::osg::Material::FRONT_AND_BACK, opacity);
        tsStateSet->setAttributeAndModes(tsMat, ::osg::StateAttribute::ON);
    }

    cga::osg::Sphere *s_wbf_p = new cga::osg::Sphere();
    cga::osg::Sphere &s_wbf = *s_wbf_p;
    ::osg::ShapeDrawable *s_wbf_Drawable = new ::osg::ShapeDrawable(s_wbf_p);
    s_wbf_Drawable->setDataVariance(::osg::Object::DYNAMIC);
    ::osg::Geode *s_wbf_Geode = new ::osg::Geode();
    s_wbf_Geode->addDrawable(s_wbf_Drawable);
    transparentSpheres->addChild(s_wbf_Geode);

    cga::osg::Sphere *s_wbr_p = new cga::osg::Sphere();
    cga::osg::Sphere &s_wbr = *s_wbr_p;
    ::osg::ShapeDrawable *s_wbr_Drawable = new ::osg::ShapeDrawable(s_wbr_p);
    s_wbr_Drawable->setDataVariance(::osg::Object::DYNAMIC);
    ::osg::Geode *s_wbr_Geode = new ::osg::Geode();
    s_wbr_Geode->addDrawable(s_wbr_Drawable);
    transparentSpheres->addChild(s_wbr_Geode);

    cga::osg::Sphere *s_mps_p = new cga::osg::Sphere();
    cga::osg::Sphere &s_mps = *s_mps_p;
    ::osg::ShapeDrawable *s_mps_Drawable = new ::osg::ShapeDrawable(s_mps_p);
    s_mps_Drawable->setDataVariance(::osg::Object::DYNAMIC);
    ::osg::Geode *s_mps_Geode = new ::osg::Geode();
    s_mps_Geode->addDrawable(s_mps_Drawable);
    transparentSpheres->addChild(s_mps_Geode);

    cga::osg::Point *p_wc_p = new cga::osg::Point();
    cga::osg::Point &p_wc = *p_wc_p;
    ::osg::ShapeDrawable *p_wc_Drawable = new ::osg::ShapeDrawable(p_wc_p);
    p_wc_Drawable->setDataVariance(::osg::Object::DYNAMIC);
    ::osg::Geode *p_wc_Geode = new ::osg::Geode();
    p_wc_Geode->addDrawable(p_wc_Drawable);
    sceneRoot->addChild(p_wc_Geode);

    cga::osg::Point *p_sa_p = new cga::osg::Point();
    cga::osg::Point &p_sa = *p_sa_p;
    ::osg::ShapeDrawable *p_sa_Drawable = new ::osg::ShapeDrawable(p_sa_p);
    p_sa_Drawable->setDataVariance(::osg::Object::DYNAMIC);
    ::osg::Geode *p_sa_Geode = new ::osg::Geode();
    p_sa_Geode->addDrawable(p_sa_Drawable);
    sceneRoot->addChild(p_sa_Geode);

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
    std::cout << "Cull mode: " << std::hex << viewer.getCamera()->getCullingMode() << std::endl;
    viewer.getCamera()->setCullingMode(0);

    double frameTime = 0.0;
    double sumFrameTime = 0.0;
    double minFrameTime = 0.0;
    double timer = 0.0;
    unsigned int counter = 0;
    while (!viewer.done())
    {
        ::osg::Timer_t startFrameTick = ::osg::Timer::instance()->tick();

        //s1.update(x1 + 0.5*(eval(x1&x1)-r*r*timer*timer)*einf + e0);
        //s1Drawable->dirtyDisplayList();

        double stroke = -0.24 + 0.01 * timer;
        double steer = 0.0;
        auto x_wbf = 1.638 * e1 + 0.333 * e2 + (-0.488) * e3;
        auto p_wbf = x_wbf + 0.5 * (x_wbf & x_wbf) * einf + e0;
        auto x_wbr = 1.4 * e1 + 0.333 * e2 + (-0.488) * e3;
        auto p_wbr = x_wbr + 0.5 * (x_wbr & x_wbr) * einf + e0;
        auto x_mps = (1.335) * e1 + 0.473 * e2 + (-0.043) * e3;
        auto p_mps = x_mps + 0.5 * (x_mps & x_mps) * einf + e0;
        auto x_steer0 = 1.3 * e1 + 0.141 * e2 + (-0.388) * e3;
        auto p_steer0 = x_steer0 + 0.5 * (x_steer0 & x_steer0) * einf + e0;
        double side = 1.0;

        stroke = std::max(-0.24, std::min(0.16, stroke));
        steer = std::max(-0.05, std::min(0.05, steer));
        //rod lengthes:
        //mcpherson strut: spring and wheel carrier
        double r_mps = 0.486 + stroke;
        //wishbone
        double r_wbf = 0.365;
        double r_wbr = 0.23;

        //wishbone circle
        s_wbf.update(p_wbf - 0.5 * r_wbf * r_wbf * einf);
        s_wbf_Drawable->dirtyDisplayList();
        s_wbr.update(p_wbr - 0.5 * r_wbr * r_wbr * einf);
        s_wbr_Drawable->dirtyDisplayList();
        auto c_wb = s_wbf ^ s_wbr;

        //sphere of mcpherson strut: spring and wheel carrier
        s_mps.update(p_mps - 0.5 * r_mps * r_mps * einf);
        s_mps_Drawable->dirtyDisplayList();

        //wheel carrier lower joint
        auto Pp_wc = Ic * (c_wb ^ s_mps);
        p_wc.update((Pp_wc + one * side * sqrt(eval(Pp_wc & Pp_wc))) * !(Pp_wc & einf)); //Point
        p_wc_Drawable->dirtyDisplayList();

        //steering link
        double r_sl = 0.4;
        //steering arm:
        //from mcpherson struct dome
        double r_samps = sqrt(pow(r_mps - 0.15, 2) + pow(0.12, 2));
        //from wheel carrier lower joint
        double r_sawc = sqrt(pow(0.15, 2) + pow(0.12, 2));

        //Translation induced to steering link inner joint by steering wheel (e.g. via a cograil)
        auto T_steer = one - einf * (steer * e2) * 0.5;
        auto p_steer = grade<1>(T_steer * p_steer0 * (~T_steer));

        auto s_samps = p_mps - 0.5 * r_samps * r_samps * einf;

        auto s_sawc = p_wc - 0.5 * r_sawc * r_sawc * einf;

        auto s_steer = p_steer - 0.5 * r_sl * r_sl * einf;

        //steering arm
        auto Pp_sa = (s_sawc ^ s_samps ^ s_steer) * Ic;
        p_sa.update((Pp_sa - one * side * sqrt(eval(Pp_sa & Pp_sa))) * !(Pp_sa & einf)); //Point
        p_sa_Drawable->dirtyDisplayList();

        viewer.frame();

        //work out if we need to force a sleep to hold back the frame rate
        ::osg::Timer_t endFrameTick = ::osg::Timer::instance()->tick();
        frameTime = ::osg::Timer::instance()->delta_s(startFrameTick, endFrameTick);

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

        timer += frameTime;

        if (frameTime < minFrameTime)
            OpenThreads::Thread::microSleep(static_cast<unsigned int>(1000000.0 * (minFrameTime - frameTime)));
    }

    return 0;
}
