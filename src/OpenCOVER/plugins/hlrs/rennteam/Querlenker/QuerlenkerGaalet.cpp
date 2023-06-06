/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "QuerlenkerGaalet.h"
#include <cover/coVRPluginSupport.h>

Querlenker::Querlenker()
: coVRPlugin(COVER_PLUGIN_NAME)
, ep({ 1.0 })
, em({ 1.0 })
, en((em - ep) * 0.5)
, ei(em + ep)
, E(ei ^ en)
, e1({ 1.0 })
, e2({ 1.0 })
, e3({ 1.0 })
, I(e1 * e2 * e3 * ep * em)
, i(e1 * e2 * e3)
, one({ 1.0 })
{
    alpha = 0.0;
    beta = 0.0;
    gamma = 0.0;
}

Querlenker::~Querlenker()
{
    delete alphaSlider;
    delete betaSlider;
    delete gammaSlider;
    delete querlenkerTab;
}

bool Querlenker::init()
{
    osg::Sphere *sphereSE = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereSEDrawable = new osg::ShapeDrawable(sphereSE);
    osg::Geode *sphereSEGeode = new osg::Geode();
    sphereSEGeode->addDrawable(sphereSEDrawable);
    sphereSETransform = new osg::PositionAttitudeTransform();
    sphereSETransform->addChild(sphereSEGeode);
    cover->getObjectsRoot()->addChild(sphereSETransform);

    osg::Sphere *sphereNE = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereNEDrawable = new osg::ShapeDrawable(sphereNE);
    osg::Geode *sphereNEGeode = new osg::Geode();
    sphereNEGeode->addDrawable(sphereNEDrawable);
    sphereNETransform = new osg::PositionAttitudeTransform();
    sphereNETransform->addChild(sphereNEGeode);
    cover->getObjectsRoot()->addChild(sphereNETransform);

    osg::Sphere *sphereNW = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereNWDrawable = new osg::ShapeDrawable(sphereNW);
    osg::Geode *sphereNWGeode = new osg::Geode();
    sphereNWGeode->addDrawable(sphereNWDrawable);
    sphereNWTransform = new osg::PositionAttitudeTransform();
    sphereNWTransform->addChild(sphereNWGeode);
    cover->getObjectsRoot()->addChild(sphereNWTransform);

    osg::Sphere *sphereSW = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereSWDrawable = new osg::ShapeDrawable(sphereSW);
    osg::Geode *sphereSWGeode = new osg::Geode();
    sphereSWGeode->addDrawable(sphereSWDrawable);
    sphereSWTransform = new osg::PositionAttitudeTransform();
    sphereSWTransform->addChild(sphereSWGeode);
    cover->getObjectsRoot()->addChild(sphereSWTransform);

    osg::Sphere *sphereFSL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFSLDrawable = new osg::ShapeDrawable(sphereFSL);
    osg::Geode *sphereFSLGeode = new osg::Geode();
    sphereFSLGeode->addDrawable(sphereFSLDrawable);
    sphereFSLTransform = new osg::PositionAttitudeTransform();
    sphereFSLTransform->addChild(sphereFSLGeode);
    cover->getObjectsRoot()->addChild(sphereFSLTransform);

    osg::Sphere *sphereFSBL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFSBLDrawable = new osg::ShapeDrawable(sphereFSBL);
    osg::Geode *sphereFSBLGeode = new osg::Geode();
    sphereFSBLGeode->addDrawable(sphereFSBLDrawable);
    sphereFSBLTransform = new osg::PositionAttitudeTransform();
    sphereFSBLTransform->addChild(sphereFSBLGeode);
    cover->getObjectsRoot()->addChild(sphereFSBLTransform);

    osg::Sphere *sphereFBL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFBLDrawable = new osg::ShapeDrawable(sphereFBL);
    osg::Geode *sphereFBLGeode = new osg::Geode();
    sphereFBLGeode->addDrawable(sphereFBLDrawable);
    sphereFBLTransform = new osg::PositionAttitudeTransform();
    sphereFBLTransform->addChild(sphereFBLGeode);
    cover->getObjectsRoot()->addChild(sphereFBLTransform);

    osg::Sphere *sphereFPBL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFPBLDrawable = new osg::ShapeDrawable(sphereFPBL);
    osg::Geode *sphereFPBLGeode = new osg::Geode();
    sphereFPBLGeode->addDrawable(sphereFPBLDrawable);
    sphereFPBLTransform = new osg::PositionAttitudeTransform();
    sphereFPBLTransform->addChild(sphereFPBLGeode);
    cover->getObjectsRoot()->addChild(sphereFPBLTransform);

    osg::Sphere *sphereFTOL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFTOLDrawable = new osg::ShapeDrawable(sphereFTOL);
    osg::Geode *sphereFTOLGeode = new osg::Geode();
    sphereFTOLGeode->addDrawable(sphereFTOLDrawable);
    sphereFTOLTransform = new osg::PositionAttitudeTransform();
    sphereFTOLTransform->addChild(sphereFTOLGeode);
    cover->getObjectsRoot()->addChild(sphereFTOLTransform);

    osg::Sphere *sphereFTL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFTLDrawable = new osg::ShapeDrawable(sphereFTL);
    osg::Geode *sphereFTLGeode = new osg::Geode();
    sphereFTLGeode->addDrawable(sphereFTLDrawable);
    sphereFTLTransform = new osg::PositionAttitudeTransform();
    sphereFTLTransform->addChild(sphereFTLGeode);
    cover->getObjectsRoot()->addChild(sphereFTLTransform);

    osg::Sphere *sphereWFL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereWFLDrawable = new osg::ShapeDrawable(sphereWFL);
    osg::Geode *sphereWFLGeode = new osg::Geode();
    sphereWFLGeode->addDrawable(sphereWFLDrawable);
    sphereWFLTransform = new osg::PositionAttitudeTransform();
    sphereWFLTransform->addChild(sphereWFLGeode);
    cover->getObjectsRoot()->addChild(sphereWFLTransform);

    osg::Sphere *sphereWFL1 = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereWFL1Drawable = new osg::ShapeDrawable(sphereWFL1);
    osg::Geode *sphereWFL1Geode = new osg::Geode();
    sphereWFL1Geode->addDrawable(sphereWFL1Drawable);
    sphereWFL1Transform = new osg::PositionAttitudeTransform();
    sphereWFL1Transform->addChild(sphereWFL1Geode);
    cover->getObjectsRoot()->addChild(sphereWFL1Transform);

    osg::Sphere *sphereWFL2 = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereWFL2Drawable = new osg::ShapeDrawable(sphereWFL2);
    osg::Geode *sphereWFL2Geode = new osg::Geode();
    sphereWFL2Geode->addDrawable(sphereWFL2Drawable);
    sphereWFL2Transform = new osg::PositionAttitudeTransform();
    sphereWFL2Transform->addChild(sphereWFL2Geode);
    cover->getObjectsRoot()->addChild(sphereWFL2Transform);

    osg::Sphere *sphereWFL3 = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereWFL3Drawable = new osg::ShapeDrawable(sphereWFL3);
    osg::Geode *sphereWFL3Geode = new osg::Geode();
    sphereWFL3Geode->addDrawable(sphereWFL3Drawable);
    sphereWFL3Transform = new osg::PositionAttitudeTransform();
    sphereWFL3Transform->addChild(sphereWFL3Geode);
    cover->getObjectsRoot()->addChild(sphereWFL3Transform);

    cylinderfl = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderflDrawable = new osg::ShapeDrawable(cylinderfl);
    osg::Geode *cylinderflGeode = new osg::Geode();
    cylinderflGeode->addDrawable(cylinderflDrawable);
    cover->getObjectsRoot()->addChild(cylinderflGeode);

    cylinderfu = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfuDrawable = new osg::ShapeDrawable(cylinderfu);
    osg::Geode *cylinderfuGeode = new osg::Geode();
    cylinderfuGeode->addDrawable(cylinderfuDrawable);
    cover->getObjectsRoot()->addChild(cylinderfuGeode);

    cylinderfp = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfpDrawable = new osg::ShapeDrawable(cylinderfp);
    osg::Geode *cylinderfpGeode = new osg::Geode();
    cylinderfpGeode->addDrawable(cylinderfpDrawable);
    cover->getObjectsRoot()->addChild(cylinderfpGeode);

    cylinderfpb = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfpbDrawable = new osg::ShapeDrawable(cylinderfpb);
    osg::Geode *cylinderfpbGeode = new osg::Geode();
    cylinderfpbGeode->addDrawable(cylinderfpbDrawable);
    cover->getObjectsRoot()->addChild(cylinderfpbGeode);

    cylinderfsb = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfsbDrawable = new osg::ShapeDrawable(cylinderfsb);
    osg::Geode *cylinderfsbGeode = new osg::Geode();
    cylinderfsbGeode->addDrawable(cylinderfsbDrawable);
    cover->getObjectsRoot()->addChild(cylinderfsbGeode);

    cylinderfpsb = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfpsbDrawable = new osg::ShapeDrawable(cylinderfpsb);
    osg::Geode *cylinderfpsbGeode = new osg::Geode();
    cylinderfpsbGeode->addDrawable(cylinderfpsbDrawable);
    cover->getObjectsRoot()->addChild(cylinderfpsbGeode);

    cylinderdpwfl = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.005f, 0);
    cylinderdpwflDrawable = new osg::ShapeDrawable(cylinderdpwfl);
    osg::Geode *cylinderdpwflGeode = new osg::Geode();
    cylinderdpwflGeode->addDrawable(cylinderdpwflDrawable);
    cover->getObjectsRoot()->addChild(cylinderdpwflGeode);

    cylinderdpfsbl = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.005f, 0);
    cylinderdpfsblDrawable = new osg::ShapeDrawable(cylinderdpfsbl);
    osg::Geode *cylinderdpfsblGeode = new osg::Geode();
    cylinderdpfsblGeode->addDrawable(cylinderdpfsblDrawable);
    cover->getObjectsRoot()->addChild(cylinderdpfsblGeode);

    cylinderdpflol = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.005f, 0);
    cylinderdpflolDrawable = new osg::ShapeDrawable(cylinderdpflol);
    osg::Geode *cylinderdpflolGeode = new osg::Geode();
    cylinderdpflolGeode->addDrawable(cylinderdpflolDrawable);
    cover->getObjectsRoot()->addChild(cylinderdpflolGeode);

    querlenkerTab = new coTUITab("Querlenker", coVRTui::instance()->mainFolder->getID());
    querlenkerTab->setPos(0, 0);

    alphaSlider = new coTUIFloatSlider("Alpha", querlenkerTab->getID());
    alphaSlider->setEventListener(this);
    alphaSlider->setRange(-0.02, 0.02);
    alphaSlider->setPos(0, 0);

    betaSlider = new coTUIFloatSlider("Beta", querlenkerTab->getID());
    betaSlider->setEventListener(this);
    betaSlider->setRange(-0.5, 0.5);
    betaSlider->setPos(0, 1);

    gammaSlider = new coTUIFloatSlider("Gamma", querlenkerTab->getID());
    gammaSlider->setEventListener(this);
    gammaSlider->setRange(-0.035, 0.035);
    gammaSlider->setPos(0, 2);

    return true;
}

void Querlenker::preFrame()
{
    double frameTime = cover->frameDuration();

    alpha = alphaSlider->getValue();
    gamma = gammaSlider->getValue();

    //Feder-Daempfer System
    auto r_fsl = e1 * 0.004 + e2 * 0.05159 + e3 * 0.58555;
    auto p_fsl = eval(grade<1>(r_fsl + (r_fsl & r_fsl) * ei * 0.5 + en));
    auto r_fbl = e1 * 0.004 + e2 * 0.266 + e3 * 0.456;
    auto p_fbl = eval(grade<1>(r_fbl + (r_fbl & r_fbl) * ei * 0.5 + en));
    //Querlenkerpunkt front lower frame left
    auto r_fll = e1 * 0.004 + e2 * 0.195 + e3 * 0.097;
    auto p_fll = eval(grade<1>(r_fll + (r_fll & r_fll) * ei * 0.5 + en));

    //double alpha = 0.0;

    double r_fsb = 0.04633;
    double r_fsd = 0.25772 - alpha;
    double dr_fsd = betaSlider->getValue();
    //auto ddr_fsd(0.0);

    auto s_fsl = eval(grade<1>(p_fsl - ei * r_fsd * r_fsd * 0.5));
    auto ds_fsl = grade<1>(ei * dr_fsd * r_fsd * (-1.0));
    //auto dds_fsl = ei*(ddr_fsd*r_fsd + dr_fsd*dr_fsd)*(-1.0);
    auto s_fbsl = eval(grade<1>(p_fbl - ei * r_fsb * r_fsb * 0.5));
    auto c_fsbl = (s_fsl ^ s_fbsl);
    auto dc_fsbl = (ds_fsl ^ s_fbsl);
    //auto ddc_fsbl = (dds_fsl^s_fbsl);
    auto phi_fsd = (p_fll ^ p_fsl ^ p_fbl ^ ei) * I;
    auto Pp_fsbl = (phi_fsd ^ c_fsbl) * I;
    auto dPp_fsbl = (phi_fsd ^ dc_fsbl) * I;
    //auto ddPp_fsbl = (phi_fsd^ddc_fsbl)*I;
    auto p_fsbl = eval(grade<1>((Pp_fsbl + one * sqrt(eval(Pp_fsbl & Pp_fsbl))) * (!(Pp_fsbl & ei))));
    //std::cout << "p_fsbl:  " << p_fsbl << std::endl;
    auto dp_fsbl = eval(grade<1>((dPp_fsbl + (dPp_fsbl & Pp_fsbl) * (!(one * (sqrt(eval(Pp_fsbl & Pp_fsbl)))))) * (!(Pp_fsbl & ei)) + (Pp_fsbl + one * sqrt(eval(Pp_fsbl & Pp_fsbl))) * (dPp_fsbl & ei) * (-1.0) * (!((Pp_fsbl & ei) * (Pp_fsbl & ei)))));
    std::cout << "dp_fsbl:  " << dp_fsbl << std::endl;
    //auto ddp_fsbl = (grade<1>((ddPp_fsbl + ddPp_fsbl*Pp_fsbl*(!(sqrt(element<0x00>(Pp_fsbl&Pp_fsbl)))) + dPp_fsbl*Pp_fsbl*(dPp_fsbl*(-1.0)*Pp_fsbl)*(!(sqrt(element<0x00>(Pp_fsbl&Pp_fsbl)))))()*(!(Pp_fsbl&ei)) + (dPp_fsbl + dPp_fsbl*Pp_fsbl*sqrt(element<0x00>(Pp_fsbl&Pp_fsbl)))()*(dPp_fsbl*(-1.0))*ei*(!(Pp_fsbl&ei*Pp_fsbl&ei))*2.0 + (Pp_fsbl + sqrt(element<0x00>(Pp_fsbl&Pp_fsbl)))()*((ddPp_fsbl*(-1.0))*ei*(!(Pp_fsbl&ei*Pp_fsbl&ei)) + dPp_fsbl&ei*dPp_fsbl&ei*(!(Pp_fsbl&ei*Pp_fsbl&ei*Pp_fsbl&ei))*2.0)()))();

    double r_fpb = 0.0764;
    double r_fpsb = 0.05116;

    auto s_fsbl = grade<1>(p_fsbl - ei * r_fpsb * r_fpsb * 0.5);
    auto ds_fsbl = grade<1>(dp_fsbl);
    //auto dds_fsbl = ddp_fsbl;
    auto s_fbl = grade<1>(p_fbl - ei * r_fpb * r_fpb * 0.5);
    auto c_fpbl = (s_fsbl ^ s_fbl);
    auto dc_fpbl = (ds_fsbl ^ s_fbl);
    //auto ddc_fpbl = (dds_fsbl^s_fbl);
    auto Pp_fpbl = (phi_fsd ^ c_fpbl) * I;
    auto dPp_fpbl = (phi_fsd ^ dc_fpbl) * I;
    //auto ddPp_fpbl = (phi_fsd^ddc_fpbl)*I;
    auto p_fpbl = eval(grade<1>((Pp_fpbl + one * sqrt(eval(Pp_fpbl & Pp_fpbl))) * (!(Pp_fpbl & ei))));
    //std::cout << "p_fpbl:  " << p_fpbl << std::endl;
    auto dp_fpbl = eval(grade<1>((dPp_fpbl + (dPp_fpbl & Pp_fpbl) * (!(one * sqrt(eval(Pp_fpbl & Pp_fpbl))))) * (!(Pp_fpbl & ei)) + (Pp_fpbl + one * sqrt(eval(Pp_fpbl & Pp_fpbl))) * (dPp_fpbl & ei) * (-1.0) * (!((Pp_fpbl & ei) * (Pp_fpbl & ei)))));
    std::cout << "dp_fpbl:  " << dp_fpbl << std::endl;
    //auto ddp_fpbl = (grade<1>((ddPp_fpbl + ddPp_fpbl*Pp_fpbl*(!(sqrt(element<0x00>(Pp_fpbl&Pp_fpbl))))() + dPp_fpbl*Pp_fpbl*(dPp_fpbl*(-1.0)*Pp_fpbl)*(!(sqrt(element<0x00>(Pp_fpbl&Pp_fpbl)))))()*(!(Pp_fpbl&ei)) + (dPp_fpbl + dPp_fpbl*Pp_fpbl*sqrt(element<0x00>(Pp_fpbl&Pp_fpbl)))()*(dPp_fpbl*(-1.0))*ei*(!(Pp_fpbl&ei*Pp_fpbl&ei))*2.0 + (Pp_fpbl + sqrt(element<0x00>(Pp_fpbl&Pp_fpbl)))()*((ddPp_fpbl*(-1.0))*ei*(!(Pp_fpbl&ei*Pp_fpbl&ei)) + dPp_fpbl&ei*dPp_fpbl&ei*(!(Pp_fpbl&ei*Pp_fpbl&ei*Pp_fpbl&ei))*2.0)))();

    //Querlenker

    double r_fp = 0.38418;
    double r_fl = 0.35726;

    auto s_fll = grade<1>(p_fll - ei * r_fl * r_fl * 0.5);
    auto s_fpbl = grade<1>(p_fpbl - ei * r_fp * r_fp * 0.5);
    auto ds_fpbl = grade<1>(dp_fpbl);
    //auto dds_fpbl = ddp_fpbl;
    //auto c_flol = (s_fll^s_fpbl);
    //auto dc_flol = (s_fll^ds_fpbl);
    auto c_flol = (s_fpbl ^ s_fll);
    auto dc_flol = (ds_fpbl ^ s_fll);
    //auto ddc_flol = (s_fll^dds_fpbl);
    auto Pp_flol = (phi_fsd ^ c_flol) * I;
    auto dPp_flol = (phi_fsd ^ dc_flol) * I;
    //auto ddPp_flol = (phi_fsd^ddc_flol)*I;
    auto p_flol = eval(grade<1>((Pp_flol + one * sqrt(eval(Pp_flol & Pp_flol))) * (!(Pp_flol & ei))));
    //std::cout << "p_flol:  " << p_flol << std::endl;
    auto dp_flol = eval(grade<1>((dPp_flol + (dPp_flol & Pp_flol) * (!(one * sqrt(eval(Pp_flol & Pp_flol))))) * (!(Pp_flol & ei)) + (Pp_flol + one * sqrt(eval(Pp_flol & Pp_flol))) * (dPp_flol & ei) * (-1.0) * (!((Pp_flol & ei) * (Pp_flol & ei)))));
    std::cout << "dp_flol:  " << dp_flol << std::endl;
    //auto ddp_flol = (grade<1>((ddPp_flol + ddPp_flol*Pp_flol*(!(sqrt(element<0x00>(Pp_flol&Pp_flol)))) + dPp_flol*Pp_flol*(dPp_flol*(-1.0)*Pp_flol)*(!(sqrt(element<0x00>(Pp_flol&Pp_flol)))))*(!(Pp_flol&ei)) + (dPp_flol + dPp_flol*Pp_flol*sqrt(element<0x00>(Pp_flol&Pp_flol)))*(dPp_flol*(-1.0))*ei*(!(Pp_flol&ei*Pp_flol&ei))*2.0 + (Pp_flol + sqrt(element<0x00>(Pp_flol&Pp_flol)))*((ddPp_flol*(-1.0))*ei*(!(Pp_flol&ei*Pp_flol&ei)) + dPp_flol&ei*dPp_flol&ei*(!(Pp_flol&ei*Pp_flol&ei*Pp_flol&ei))*2.0)))();

    auto r_ful = e1 * 0.037 + e2 * 0.288 + e3 * 0.261;
    auto p_ful = eval(grade<1>(r_ful + (r_ful & r_ful) * ei * 0.5 + en));

    double r_fo = 0.21921;
    double r_fu = 0.26086;

    //Punkte fuer Ebene oberer Querlenker
    auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
    auto p_phi1 = eval(grade<1>(r_phi1 + (r_phi1 & r_phi1) * ei * 0.5 + en));
    auto r_phi2 = e1 * 0.037 + e2 * 1.0 + e3 * 0.0;
    auto p_phi2 = eval(grade<1>(r_phi2 + (r_phi2 & r_phi2) * ei * 0.5 + en));

    auto s_ful = grade<1>(p_ful - ei * r_fu * r_fu * 0.5);
    auto s_flol = grade<1>(p_flol - ei * r_fo * r_fo * 0.5);
    auto ds_flol = grade<1>(dp_flol);
    //auto dds_flol = ddp_flol;
    //auto c_flul = (s_ful^s_flol);
    //auto dc_flul = (s_ful^ds_flol);
    auto c_flul = (s_flol ^ s_ful);
    auto dc_flul = (ds_flol ^ s_ful);
    //auto ddc_flul = (s_ful^dds_flol);
    auto phi_fuo = (p_ful ^ p_phi1 ^ p_phi2 ^ ei) * I;
    auto Pp_fuol = (phi_fuo ^ c_flul) * I;
    //std::cout << "Pp_fuol:  " << Pp_fuol << std::endl;
    auto dPp_fuol = (phi_fuo ^ dc_flul) * I;
    //std::cout << "dPp_fuol:  " << dPp_fuol << std::endl;
    //auto ddPp_fuol = (phi_fuo^ddc_flul)*I;
    auto p_fuol = eval(grade<1>((Pp_fuol + one * sqrt(eval(Pp_fuol & Pp_fuol))) * (!(Pp_fuol & ei))));
    //std::cout << "p_fuol:  " << p_fuol << std::endl;
    auto dp_fuol = eval(grade<1>((dPp_fuol + (dPp_fuol & Pp_fuol) * (!(one * sqrt(eval(Pp_fuol & Pp_fuol))))) * (!(Pp_fuol & ei)) + (Pp_fuol + one * sqrt(eval(Pp_fuol & Pp_fuol))) * (dPp_fuol & ei) * (-1.0) * (!((Pp_fuol & ei) * (Pp_fuol & ei)))));
    std::cout << "dp_fuol:  " << dp_fuol << std::endl;
    //auto ddp_fuol = (grade<1>((ddPp_fuol + ddPp_fuol*Pp_fuol*(!(sqrt(element<0x00>(Pp_fuol&Pp_fuol)))) + dPp_fuol*Pp_fuol*(dPp_fuol*(-1.0)*Pp_fuol)*(!(sqrt(element<0x00>(Pp_fuol&Pp_fuol)))))*(!(Pp_fuol&ei)) + (dPp_fuol + dPp_fuol*Pp_fuol*sqrt(element<0x00>(Pp_fuol&Pp_fuol)))*(dPp_fuol*(-1.0))*ei*(!(Pp_fuol&ei*Pp_fuol&ei))*2.0 + (Pp_fuol + sqrt(element<0x00>(Pp_fuol&Pp_fuol)))*((ddPp_fuol*(-1.0))*ei*(!(Pp_fuol&ei*Pp_fuol&ei)) + dPp_fuol&ei*dPp_fuol&ei*(!(Pp_fuol&ei*Pp_fuol&ei*Pp_fuol&ei))*2.0)))();

    //Spurstange
    auto r_ftl = e1 * (-0.055) + e2 * (0.204 + gamma) + e3 * 0.101; //Anbindungspunkt tie rod an Rahmen
    auto p_ftl = eval(grade<1>(r_ftl + (r_ftl & r_ftl) * ei * 0.5 + en));

    auto dp_ftl = one * 0.0;
    //auto ddp_ftl = one*0.0;

    double r_ftr = 0.39760; //Länge tie rod
    double r_fto = 0.08377; //Abstand p_flol zu p_ftol
    double r_fuo = 0.23717; //Abstand p_fuol zu p_ftol

    auto s_ftol = grade<1>(p_flol - ei * r_fto * r_fto * 0.5);
    //std::cout << "s_ftol:  " << s_ftol << std::endl;
    auto ds_ftol = grade<1>(dp_flol);
    //std::cout << "ds_ftol:  " << ds_ftol << std::endl;
    //auto dds_ftol = ddp_flol;
    auto s_ftl = grade<1>(p_ftl - ei * r_ftr * r_ftr * 0.5);
    //std::cout << "s_ftl:  " << s_ftl << std::endl;
    auto ds_ftl = cm::mv<0x02>::type({ 1.0 }) * dp_ftl;
    //std::cout << "ds_ftl:  " << ds_ftl << std::endl;
    //auto dds_ftl = ddp_ftl;
    auto s_fuol = grade<1>(p_fuol - ei * r_fuo * r_fuo * 0.5);
    //std::cout << "s_fuol:  " << s_fuol << std::endl;
    auto ds_fuol = grade<1>(dp_fuol);
    //std::cout << "ds_fuol:  " << ds_fuol << std::endl;
    //auto dds_fuol = ddp_fuol;
    auto Pp_ftol = (s_ftol ^ s_ftl ^ s_fuol) * I;
    //std::cout << "Pp_ftol:  " << Pp_ftol << std::endl;
    //auto dPp_ftol = ((ds_ftol^s_ftl^s_fuol) + (s_ftol^s_ftl^ds_fuol) + (s_ftol^ds_ftl^s_fuol))*I;
    auto dPp_ftol = ((ds_ftol ^ s_ftl ^ s_fuol) + (s_ftol ^ s_ftl ^ ds_fuol)) * I;
    //std::cout << "dPp_ftol:  " << dPp_ftol << std::endl;
    //auto ddPp_ftol = (((dds_ftol^s_ftl^s_fuol) + (ds_ftol^ds_ftl^s_fuol) + (ds_ftol^s_ftl^ds_fuol) + (ds_ftol^s_ftl^ds_fuol) + (s_ftol^ds_ftl^ds_fuol) + (s_ftol^s_ftl^dds_fuol) + (ds_ftol^ds_ftl^s_fuol) + (s_ftol^dds_ftl^s_fuol) + (s_ftol^ds_ftl^ds_fuol))*I)();
    auto p_ftol = eval(grade<1>((Pp_ftol + one * sqrt(eval(Pp_ftol & Pp_ftol))) * (!(Pp_ftol & ei))));
    //std::cout << "p_ftol:  " << p_ftol << std::endl;
    auto dp_ftol = eval(grade<1>((dPp_ftol + (dPp_ftol & Pp_ftol) * (!(one * sqrt(eval(Pp_ftol & Pp_ftol))))) * (!(Pp_ftol & ei)) + (Pp_ftol + one * sqrt(eval(Pp_ftol & Pp_ftol))) * (dPp_ftol & ei) * (-1.0) * (!((Pp_ftol & ei) * (Pp_ftol & ei)))));
    std::cout << "dp_ftol:  " << dp_ftol << std::endl;
    //auto ddp_ftol = (grade<1>((ddPp_ftol + ddPp_ftol*Pp_ftol*(!(sqrt(element<0x00>(Pp_ftol&Pp_ftol)))) + dPp_ftol*Pp_ftol*(dPp_ftol*(-1.0)*Pp_ftol)*(!(sqrt(element<0x00>(Pp_ftol&Pp_ftol)))))*(!(Pp_ftol&ei)) + (dPp_ftol + dPp_ftol*Pp_ftol*sqrt(element<0x00>(Pp_ftol&Pp_ftol)))*(dPp_ftol*(-1.0))*ei*(!(Pp_ftol&ei*Pp_ftol&ei))*2.0 + (Pp_ftol + sqrt(element<0x00>(Pp_ftol&Pp_ftol)))*((ddPp_ftol*(-1.0))*ei*(!(Pp_ftol&ei*Pp_ftol&ei)) + dPp_ftol&ei*dPp_ftol&ei*(!(Pp_ftol&ei*Pp_ftol&ei*Pp_ftol&ei))*2.0)))();

    //Bestimmung Radaufstandspunkt
    double r_wheel = 0.255; //Reifenradius

    auto phi_fpol = (p_flol ^ p_fuol ^ p_ftol ^ ei) * I; //Ebene front points outer left
    //auto dphi_fpol = eval(((dp_flol^p_fuol^p_ftol^ei) + (p_flol^dp_fuol^p_ftol^ei) + (p_flol^p_fuol^dp_ftol^ei))*I);
    //auto ddphi_fpol = (((ddp_flol^p_fuol^p_ftol^ei) + (dp_flol^dp_fuol^p_ftol^ei) + (dp_flol^p_fuol^dp_ftol^ei) + (dp_flol^dp_fuol^p_ftol^ei) + (p_flol^ddp_fuol^p_ftol^ei) + (p_flol^dp_fuol^dp_ftol^ei) + (dp_flol^p_fuol^dp_ftol^ei) + (p_flol^dp_fuol^dp_ftol^ei) + (p_flol^p_fuol^ddp_ftol^ei))*I)();
    auto phi_fpoln = eval(phi_fpol * (!(magnitude(phi_fpol))) * (-1.0));
    //auto dphi_fpoln = eval(dphi_fpol*(!(magnitude(phi_fpol)))*(-1.0) - phi_fpol*(dphi_fpol*phi_fpol)*(!(one*sqrt(eval(grade<0>(phi_fpol*phi_fpol)))*(phi_fpol*phi_fpol))));
    auto T_fwrl = eval(one + ei * (p_flol - en) * 0.5); //Definition Translator
    //auto dT_fwrl = ei*dp_flol*0.5;
    //auto ddT_fwrl = ei*ddp_flol*0.5;
    auto phi_fwrl = eval(en & (ei ^ (e2 * one * sqrt(eval(grade<0>(phi_fpoln * phi_fpoln))) + phi_fpoln))); //Ebene front wheel reference left
    //auto dphi_fwrl = eval(en&(ei^(e2*(dphi_fpoln*phi_fpoln*(!(one*sqrt(eval(grade<0>(phi_fpoln*phi_fpoln)))))) + dphi_fpoln)));
    //auto ddphi_fwrl = (en&(ei^(e2*((ddphi_fpol*phi_fpol + (dphi_fpol*dphi_fpol))*(!(sqrt(phi_fpol*phi_fpol))) + (dphi_fpol*phi_fpol)*(dphi_fpol*phi_fpol)*(!(phi_fpol*phi_fpol*sqrt(phi_fpol*phi_fpol)))*(-1.0)) + ddphi_fpol)))();
    auto R_fwrl = part<0, 3, 5, 6>(phi_fwrl * e2); //Definition Rotor
    //auto dR_fwrl = dphi_fwrl*e2;
    //auto ddR_fwrl = ddphi_fwrl*e2;
    auto p_frwl = e1 * 0.004603 - e2 * 0.003048 + e3 * 0.086825; //Vektor front rotation wheel left
    auto R_frwl = eval(exp((!(one * sqrt(eval(grade<0>(p_frwl * p_frwl))))) * 0.5 * p_frwl * e1 * e2 * e3 * (144.548 * 3.141 / 180.0)));
    auto T_fwp = eval(one + ei * (e1 * (-0.004) + e2 * 0.059 + e3 * 0.103) * 0.5);
    auto D_fwp = eval(part_type<D_type>(T_fwrl * R_fwrl * T_fwp * R_frwl));
    //auto dD_fwp = eval((dT_fwrl*R_fwrl + T_fwrl*dR_fwrl)*T_fwp*R_frwl);
    //auto ddD_fwp = (ddT_fwrl*R_fwrl + dT_fwrl*dR_fwrl*2.0 + T_fwrl*ddR_fwrl)*T_fwp*R_frwl;
    auto p_fwl = eval(grade<1>(e3 * (-r_wheel) + ei * r_wheel * r_wheel * 0.5 + en));
    auto p_fwl1 = eval(grade<1>(e3 * r_wheel + ei * r_wheel * r_wheel * 0.5 + en));
    auto p_fwl2 = eval(grade<1>(e1 * (-r_wheel) + ei * r_wheel * r_wheel * 0.5 + en));
    auto p_fwl3 = eval(grade<1>(e1 * r_wheel + ei * r_wheel * r_wheel * 0.5 + en));

    //Bewegung Radaufstandspunkt

    auto p_wfl = eval(grade<1>(D_fwp * p_fwl * (!(D_fwp))));
    //auto dp_wfl = eval(grade<1>(dD_fwp*p_fwl*(!(D_fwp)) + D_fwp*p_fwl*(!(D_fwp*D_fwp))*dD_fwp*(-1.0)));
    //std::cout << "dp_wfl:  " << dp_wfl << std::endl;
    auto p_wfl1 = eval(grade<1>(D_fwp * p_fwl1 * (!(D_fwp))));
    auto p_wfl2 = eval(grade<1>(D_fwp * p_fwl2 * (!(D_fwp))));
    auto p_wfl3 = eval(grade<1>(D_fwp * p_fwl3 * (!(D_fwp))));
    //auto ddp_wfl = (ddD_fwp*p_fwl*(!(D_fwp)) + dD_fwp*p_fwl*(!(D_fwp*D_fwp))*dD_fwp*(-1.0) + dD_fwp*p_fwl*(!(D_fwp*D_fwp))*dD_fwp*(-1.0) + D_fwp*p_fwl*((!(D_fwp*D_fwp))*ddD_fwp*(-1.0) + (!(D_fwp*D_fwp*D_fwp))*dD_fwp*dD_fwp*2.0))();

    sphereSETransform->setPosition(osg::Vec3(p_fll[0], p_fll[1], p_fll[2]));
    sphereNETransform->setPosition(osg::Vec3(p_ful[0], p_ful[1], p_ful[2]));
    sphereNWTransform->setPosition(osg::Vec3(p_flol[0], p_flol[1], p_flol[2]));
    sphereSWTransform->setPosition(osg::Vec3(p_fuol[0], p_fuol[1], p_fuol[2]));
    sphereFSLTransform->setPosition(osg::Vec3(p_fsl[0], p_fsl[1], p_fsl[2]));
    sphereFSBLTransform->setPosition(osg::Vec3(p_fsbl[0], p_fsbl[1], p_fsbl[2]));
    sphereFBLTransform->setPosition(osg::Vec3(p_fbl[0], p_fbl[1], p_fbl[2]));
    sphereFPBLTransform->setPosition(osg::Vec3(p_fpbl[0], p_fpbl[1], p_fpbl[2]));
    sphereFTOLTransform->setPosition(osg::Vec3(p_ftol[0], p_ftol[1], p_ftol[2]));
    sphereFTLTransform->setPosition(osg::Vec3(p_ftl[0], p_ftl[1], p_ftl[2]));
    sphereWFLTransform->setPosition(osg::Vec3(p_wfl[0], p_wfl[1], p_wfl[2]));
    sphereWFL1Transform->setPosition(osg::Vec3(p_wfl1[0], p_wfl1[1], p_wfl1[2]));
    sphereWFL2Transform->setPosition(osg::Vec3(p_wfl2[0], p_wfl2[1], p_wfl2[2]));
    sphereWFL3Transform->setPosition(osg::Vec3(p_wfl3[0], p_wfl3[1], p_wfl3[2]));

    cylinderfl->setCenter(osg::Vec3(0.5 * (p_fll[0] + p_flol[0]), 0.5 * (p_fll[1] + p_flol[1]), 0.5 * (p_fll[2] + p_flol[2])));
    cylinderfl->setHeight(sqrt(-2.0 * eval(p_fll & p_flol)[0]));
    osg::Quat cylinderflrot;
    cylinderflrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fll[0] - p_flol[0], p_fll[1] - p_flol[1], p_fll[2] - p_flol[2]));
    cylinderfl->setRotation(cylinderflrot);
    cylinderflDrawable->dirtyDisplayList();

    cylinderfu->setCenter(osg::Vec3(0.5 * (p_ful[0] + p_fuol[0]), 0.5 * (p_ful[1] + p_fuol[1]), 0.5 * (p_ful[2] + p_fuol[2])));
    cylinderfu->setHeight(sqrt(-2.0 * eval(p_ful & p_fuol)[0]));
    osg::Quat cylinderfurot;
    cylinderfurot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_ful[0] - p_fuol[0], p_ful[1] - p_fuol[1], p_ful[2] - p_fuol[2]));
    cylinderfu->setRotation(cylinderfurot);
    cylinderfuDrawable->dirtyDisplayList();

    cylinderfp->setCenter(osg::Vec3(0.5 * (p_fpbl[0] + p_flol[0]), 0.5 * (p_fpbl[1] + p_flol[1]), 0.5 * (p_fpbl[2] + p_flol[2])));
    cylinderfp->setHeight(sqrt(-2.0 * eval(p_fpbl & p_flol)[0]));
    osg::Quat cylinderfprot;
    cylinderfprot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fpbl[0] - p_flol[0], p_fpbl[1] - p_flol[1], p_fpbl[2] - p_flol[2]));
    cylinderfp->setRotation(cylinderfprot);
    cylinderfpDrawable->dirtyDisplayList();

    cylinderfpb->setCenter(osg::Vec3(0.5 * (p_fbl[0] + p_fpbl[0]), 0.5 * (p_fbl[1] + p_fpbl[1]), 0.5 * (p_fbl[2] + p_fpbl[2])));
    cylinderfpb->setHeight(sqrt(-2.0 * eval(p_fbl & p_fpbl)[0]));
    osg::Quat cylinderfpbrot;
    cylinderfpbrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fbl[0] - p_fpbl[0], p_fbl[1] - p_fpbl[1], p_fbl[2] - p_fpbl[2]));
    cylinderfpb->setRotation(cylinderfpbrot);
    cylinderfpbDrawable->dirtyDisplayList();

    cylinderfsb->setCenter(osg::Vec3(0.5 * (p_fbl[0] + p_fsbl[0]), 0.5 * (p_fbl[1] + p_fsbl[1]), 0.5 * (p_fbl[2] + p_fsbl[2])));
    cylinderfsb->setHeight(sqrt(-2.0 * eval(p_fbl & p_fsbl)[0]));
    osg::Quat cylinderfsbrot;
    cylinderfsbrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fbl[0] - p_fsbl[0], p_fbl[1] - p_fsbl[1], p_fbl[2] - p_fsbl[2]));
    cylinderfsb->setRotation(cylinderfsbrot);
    cylinderfsbDrawable->dirtyDisplayList();

    cylinderfpsb->setCenter(osg::Vec3(0.5 * (p_fpbl[0] + p_fsbl[0]), 0.5 * (p_fpbl[1] + p_fsbl[1]), 0.5 * (p_fpbl[2] + p_fsbl[2])));
    cylinderfpsb->setHeight(sqrt(-2.0 * eval(p_fpbl & p_fsbl)[0]));
    osg::Quat cylinderfpsbrot;
    cylinderfpsbrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fpbl[0] - p_fsbl[0], p_fpbl[1] - p_fsbl[1], p_fpbl[2] - p_fsbl[2]));
    cylinderfpsb->setRotation(cylinderfpsbrot);
    cylinderfpsbDrawable->dirtyDisplayList();

    /*cylinderdpwfl->setCenter(osg::Vec3(p_wfl[0] + (dp_wfl[0])*0.5, p_wfl[1] + (dp_wfl[1])*0.5, p_wfl[2] + (dp_wfl[2])*0.5));
	cylinderdpwfl->setHeight(eval(magnitude(dp_wfl))[0]);
	osg::Quat cylinderdpwflrot;
	cylinderdpwflrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(dp_wfl[0], dp_wfl[1], dp_wfl[2]));
	cylinderdpwfl->setRotation(cylinderdpwflrot);
	cylinderdpwflDrawable->dirtyDisplayList();

	cylinderdpfsbl->setCenter(osg::Vec3(p_fsbl[0] + (dp_fsbl[0])*0.5, p_fsbl[1] + (dp_fsbl[1])*0.5, p_fsbl[2] + (dp_fsbl[2])*0.5));
	cylinderdpfsbl->setHeight(eval(magnitude(dp_fsbl))[0]);
	osg::Quat cylinderdpfsblrot;
	cylinderdpfsblrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(dp_fsbl[0], dp_fsbl[1], dp_fsbl[2]));
	cylinderdpfsbl->setRotation(cylinderdpfsblrot);
	cylinderdpfsblDrawable->dirtyDisplayList();
	
	cylinderdpflol->setCenter(osg::Vec3(p_flol[0] + (dp_flol[0])*0.5, p_flol[1] + (dp_flol[1])*0.5, p_flol[2] + (dp_flol[2])*0.5));
	cylinderdpflol->setHeight(eval(magnitude(dp_flol))[0]);
	osg::Quat cylinderdpflolrot;
	cylinderdpflolrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(dp_flol[0], dp_flol[1], dp_flol[2]));
	cylinderdpflol->setRotation(cylinderdpflolrot);
	cylinderdpflolDrawable->dirtyDisplayList()*/;
}
COVERPLUGIN(Querlenker)
