/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Querlenker.h"
#include <cover/coVRPluginSupport.h>

Querlenker::Querlenker()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    alpha = 0.0;
    gamma = 0.0;
    e1 = { 1.0 };
    e2 = { 1.0 };
    e3 = { 1.0 };
    ep = { 1.0 };
    em = { 1.0 };

    one = { 1.0 };

    e0 = 0.5 * (em - ep);
    einf = em + ep;

    E = ep * em;

    Ic = e1 * e2 * e3 * ep * em;
    Ie = e1 * e2 * e3;
}

Querlenker::~Querlenker()
{
    delete alphaSlider;
    delete gammaSlider;
    delete querlenkerTab;
}

bool Querlenker::init()
{
    osg::Sphere *sphereFLL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFLLDrawable = new osg::ShapeDrawable(sphereFLL);
    osg::Geode *sphereFLLGeode = new osg::Geode();
    sphereFLLGeode->addDrawable(sphereFLLDrawable);
    sphereFLLTransform = new osg::PositionAttitudeTransform();
    sphereFLLTransform->addChild(sphereFLLGeode);
    cover->getObjectsRoot()->addChild(sphereFLLTransform);

    osg::Sphere *sphereFLL2 = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFLL2Drawable = new osg::ShapeDrawable(sphereFLL2);
    osg::Geode *sphereFLL2Geode = new osg::Geode();
    sphereFLL2Geode->addDrawable(sphereFLL2Drawable);
    sphereFLL2Transform = new osg::PositionAttitudeTransform();
    sphereFLL2Transform->addChild(sphereFLL2Geode);
    cover->getObjectsRoot()->addChild(sphereFLL2Transform);

    osg::Sphere *sphereFUL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFULDrawable = new osg::ShapeDrawable(sphereFUL);
    osg::Geode *sphereFULGeode = new osg::Geode();
    sphereFULGeode->addDrawable(sphereFULDrawable);
    sphereFULTransform = new osg::PositionAttitudeTransform();
    sphereFULTransform->addChild(sphereFULGeode);
    cover->getObjectsRoot()->addChild(sphereFULTransform);

    osg::Sphere *sphereFUL2 = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFUL2Drawable = new osg::ShapeDrawable(sphereFUL2);
    osg::Geode *sphereFUL2Geode = new osg::Geode();
    sphereFUL2Geode->addDrawable(sphereFUL2Drawable);
    sphereFUL2Transform = new osg::PositionAttitudeTransform();
    sphereFUL2Transform->addChild(sphereFUL2Geode);
    cover->getObjectsRoot()->addChild(sphereFUL2Transform);

    osg::Sphere *sphereFLOL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFLOLDrawable = new osg::ShapeDrawable(sphereFLOL);
    osg::Geode *sphereFLOLGeode = new osg::Geode();
    sphereFLOLGeode->addDrawable(sphereFLOLDrawable);
    sphereFLOLTransform = new osg::PositionAttitudeTransform();
    sphereFLOLTransform->addChild(sphereFLOLGeode);
    cover->getObjectsRoot()->addChild(sphereFLOLTransform);

    osg::Sphere *sphereFUOL = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereFUOLDrawable = new osg::ShapeDrawable(sphereFUOL);
    osg::Geode *sphereFUOLGeode = new osg::Geode();
    sphereFUOLGeode->addDrawable(sphereFUOLDrawable);
    sphereFUOLTransform = new osg::PositionAttitudeTransform();
    sphereFUOLTransform->addChild(sphereFUOLGeode);
    cover->getObjectsRoot()->addChild(sphereFUOLTransform);

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

    osg::Sphere *sphereWFL4 = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereWFL4Drawable = new osg::ShapeDrawable(sphereWFL4);
    osg::Geode *sphereWFL4Geode = new osg::Geode();
    sphereWFL4Geode->addDrawable(sphereWFL4Drawable);
    sphereWFL4Transform = new osg::PositionAttitudeTransform();
    sphereWFL4Transform->addChild(sphereWFL4Geode);
    cover->getObjectsRoot()->addChild(sphereWFL4Transform);

    osg::Sphere *sphereMA = new osg::Sphere(osg::Vec3(0, 0, 0), 0.01f);
    osg::ShapeDrawable *sphereMADrawable = new osg::ShapeDrawable(sphereMA);
    osg::Geode *sphereMAGeode = new osg::Geode();
    sphereMAGeode->addDrawable(sphereMADrawable);
    sphereMATransform = new osg::PositionAttitudeTransform();
    sphereMATransform->addChild(sphereMAGeode);
    cover->getObjectsRoot()->addChild(sphereMATransform);

    cylinderfl = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderflDrawable = new osg::ShapeDrawable(cylinderfl);
    osg::Geode *cylinderflGeode = new osg::Geode();
    cylinderflGeode->addDrawable(cylinderflDrawable);
    cover->getObjectsRoot()->addChild(cylinderflGeode);

    cylinderfl2 = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfl2Drawable = new osg::ShapeDrawable(cylinderfl2);
    osg::Geode *cylinderfl2Geode = new osg::Geode();
    cylinderfl2Geode->addDrawable(cylinderfl2Drawable);
    cover->getObjectsRoot()->addChild(cylinderfl2Geode);

    cylinderfu = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfuDrawable = new osg::ShapeDrawable(cylinderfu);
    osg::Geode *cylinderfuGeode = new osg::Geode();
    cylinderfuGeode->addDrawable(cylinderfuDrawable);
    cover->getObjectsRoot()->addChild(cylinderfuGeode);

    cylinderfu2 = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfu2Drawable = new osg::ShapeDrawable(cylinderfu2);
    osg::Geode *cylinderfu2Geode = new osg::Geode();
    cylinderfu2Geode->addDrawable(cylinderfu2Drawable);
    cover->getObjectsRoot()->addChild(cylinderfu2Geode);

    cylinderfp = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.01f, 0);
    cylinderfpDrawable = new osg::ShapeDrawable(cylinderfp);
    osg::Geode *cylinderfpGeode = new osg::Geode();
    cylinderfpGeode->addDrawable(cylinderfpDrawable);
    cover->getObjectsRoot()->addChild(cylinderfpGeode);

    cylinderfs = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.005f, 0);
    cylinderfsDrawable = new osg::ShapeDrawable(cylinderfs);
    osg::Geode *cylinderfsGeode = new osg::Geode();
    cylinderfsGeode->addDrawable(cylinderfsDrawable);
    cover->getObjectsRoot()->addChild(cylinderfsGeode);

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

    cylinderftl = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.005f, 0);
    cylinderftlDrawable = new osg::ShapeDrawable(cylinderftl);
    osg::Geode *cylinderftlGeode = new osg::Geode();
    cylinderftlGeode->addDrawable(cylinderftlDrawable);
    cover->getObjectsRoot()->addChild(cylinderftlGeode);

    /*cylinderffv = new osg::Cylinder(osg::Vec3(0,0,0), 0.005f, 0);
   cylinderffvDrawable = new osg::ShapeDrawable(cylinderffv);
   osg::Geode* cylinderffvGeode = new osg::Geode();
   cylinderffvGeode->addDrawable(cylinderffvDrawable);
   cover->getObjectsRoot()->addChild(cylinderffvGeode);   

   cylinderffvp = new osg::Cylinder(osg::Vec3(0,0,0), 0.005f, 0);
   cylinderffvpDrawable = new osg::ShapeDrawable(cylinderffvp);
   osg::Geode* cylinderffvpGeode = new osg::Geode();
   cylinderffvpGeode->addDrawable(cylinderffvpDrawable);
   cover->getObjectsRoot()->addChild(cylinderffvpGeode);

   cylinderffvo = new osg::Cylinder(osg::Vec3(0,0,0), 0.005f, 0);
   cylinderffvoDrawable = new osg::ShapeDrawable(cylinderffvo);
   osg::Geode* cylinderffvoGeode = new osg::Geode();
   cylinderffvoGeode->addDrawable(cylinderffvoDrawable);
   cover->getObjectsRoot()->addChild(cylinderffvoGeode); */

    querlenkerTab = new coTUITab("Querlenker", coVRTui::instance()->mainFolder->getID());
    querlenkerTab->setPos(0, 0);

    alphaSlider = new coTUIFloatSlider("Alpha", querlenkerTab->getID());
    alphaSlider->setEventListener(this);
    alphaSlider->setRange(-0.02, 0.02);
    alphaSlider->setPos(0, 0);

    gammaSlider = new coTUIFloatSlider("Gamma", querlenkerTab->getID());
    gammaSlider->setEventListener(this);
    gammaSlider->setRange(-0.035, 0.035);
    gammaSlider->setPos(0, 2);

    return true;
}

void Querlenker::Radaufhaengung_wfr(double u_wfr, double steerAngle, D_type &D_wfr, Vector &nrc_wfr) const
{
    u_wfr = std::max(-0.02, std::min(0.02, u_wfr));
    //Feder-Daempfer System
    auto r_fsr = e1 * 0.004 + e2 * 0.05159 + e3 * 0.58555;
    auto p_fsr = eval(grade<1>(r_fsr + (r_fsr & r_fsr) * einf * 0.5 + e0));
    auto r_fbr = e1 * 0.004 + e2 * 0.266 + e3 * 0.456;
    auto p_fbr = eval(grade<1>(r_fbr + (r_fbr & r_fbr) * einf * 0.5 + e0));
    //Querlenkerpunkt front lower frame left
    auto r_flr = e1 * 0.004 + e2 * 0.195 + e3 * 0.097;
    auto p_flr1 = eval(grade<1>(r_flr + (r_flr & r_flr) * einf * 0.5 + e0));
    auto r_flr2 = e1 * 0.280 + e2 * 0.195 + e3 * 0.097;
    auto p_flr2 = eval(grade<1>(r_flr2 + (r_flr2 & r_flr2) * einf * 0.5 + e0));

    double r_fsb = 0.04633;
    double r_fsd = 0.25772 - u_wfr;

    auto s_fsr = eval(grade<1>(p_fsr - einf * r_fsd * r_fsd * 0.5));
    auto s_fbsr = eval(grade<1>(p_fbr - einf * r_fsb * r_fsb * 0.5));
    auto c_fsbr = (s_fsr ^ s_fbsr);
    auto phi_fsd = (p_flr1 ^ p_fsr ^ p_fbr ^ einf) * Ic;
    auto Pp_fsbr = (phi_fsd ^ c_fsbr) * Ic;
    auto p_fsbr = eval(grade<1>((Pp_fsbr + one * sqrt((Pp_fsbr & Pp_fsbr).element<0x00>())) * (!(Pp_fsbr & einf))));

    double r_fpb = 0.0764;
    double r_fpsb = 0.05116;

    auto s_fsbr = grade<1>(p_fsbr - einf * r_fpsb * r_fpsb * 0.5);
    auto s_fbr = grade<1>(p_fbr - einf * r_fpb * r_fpb * 0.5);
    auto c_fpbr = (s_fsbr ^ s_fbr);
    auto Pp_fpbr = (phi_fsd ^ c_fpbr) * Ic;
    auto p_fpbr = eval(grade<1>((Pp_fpbr + one * sqrt((Pp_fpbr & Pp_fpbr).element<0x00>())) * (!(Pp_fpbr & einf))));

    //Querlenker

    double r_fp = 0.38418;
    double r_fl = 0.35726;

    auto s_flr = grade<1>(p_flr1 - einf * r_fl * r_fl * 0.5);
    auto s_fpbr = grade<1>(p_fpbr - einf * r_fp * r_fp * 0.5);
    auto c_flor = (s_fpbr ^ s_flr);
    auto Pp_flor = (phi_fsd ^ c_flor) * Ic;
    auto p_flor = eval(grade<1>((Pp_flor + one * sqrt((Pp_flor & Pp_flor).element<0x00>())) * (!(Pp_flor & einf))));

    auto r_fur1 = e1 * 0.037 + e2 * 0.288 + e3 * 0.261;
    auto p_fur1 = eval(grade<1>(r_fur1 + (r_fur1 & r_fur1) * einf * 0.5 + e0));
    auto r_fur2 = e1 * 0.210 + e2 * 0.288 + e3 * 0.261;
    auto p_fur2 = eval(grade<1>(r_fur2 + (r_fur2 & r_fur2) * einf * 0.5 + e0));

    double r_fo = 0.21921;
    double r_fu = 0.26086;

    //Punkte fuer Ebene oberer Querlenker
    auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
    auto p_phi1 = eval(grade<1>(r_phi1 + (r_phi1 & r_phi1) * einf * 0.5 + e0));
    auto r_phi2 = e1 * 0.037 + e2 * 1.0 + e3 * 0.0;
    auto p_phi2 = eval(grade<1>(r_phi2 + (r_phi2 & r_phi2) * einf * 0.5 + e0));

    auto s_fur1 = grade<1>(p_fur1 - einf * r_fu * r_fu * 0.5);
    auto s_flor = grade<1>(p_flor - einf * r_fo * r_fo * 0.5);
    auto c_flur = (s_flor ^ s_fur1);
    auto phi_fuo = (p_fur1 ^ p_phi1 ^ p_phi2 ^ einf) * Ic;
    auto Pp_fuor = (phi_fuo ^ c_flur) * Ic;
    auto p_fuor = eval(grade<1>((Pp_fuor + one * sqrt((Pp_fuor & Pp_fuor).element<0x00>())) * (!(Pp_fuor & einf))));

    //Spurstange
    double steering = steerAngle * 11.4592; //Umrechnung vom Lenkwinkel auf Lenkgetriebeweg
    auto r_ftr = e1 * (-0.055) + e2 * (0.204 + steering) + e3 * 0.101; //Anbindungspunkt tie rod an Rahmen
    auto p_ftr = eval(grade<1>(r_ftr + (r_ftr & r_ftr) * einf * 0.5 + e0));

    double r_ft = 0.39760; //Länge tie rod
    double r_fto = 0.08377; //Abstand p_flol zu p_ftol
    double r_fuo = 0.23717; //Abstand p_fuol zu p_ftol

    auto s_ftor = grade<1>(p_flor - einf * r_fto * r_fto * 0.5);
    auto s_ftr = grade<1>(p_ftr - einf * r_ft * r_ft * 0.5);
    auto s_fuor = grade<1>(p_fuor - einf * r_fuo * r_fuo * 0.5);
    auto Pp_ftor = (s_ftor ^ s_ftr ^ s_fuor) * Ic;
    auto p_ftor = eval(grade<1>((Pp_ftor + one * sqrt((Pp_ftor & Pp_ftor).element<0>())) * (!(Pp_ftor & einf))));

    //Bestimmung Radaufstandspunkt

    double r_wheel = 0.255; //Reifenradius
    auto phi_fpor = (p_flor ^ p_fuor ^ p_ftor ^ einf) * Ic; //Ebene front points outer left
    auto phi_fporn = eval(phi_fpor * (!(magnitude(phi_fpor))) * (-1.0));
    auto T_fwrr = eval(one + einf * (p_flor - e0) * 0.5); //Definition Translator
    auto phi_fwrr = eval(e0 & (einf ^ (e2 * sqrt(eval(phi_fporn & phi_fporn)) + phi_fporn))); //Ebene front wheel reference left
    auto R_fwrr_raw = part<0, 3, 5, 6>(phi_fwrr * e2); //Definition Rotor
    auto R_fwrr = R_fwrr_raw * !magnitude(R_fwrr_raw);
    auto R_frwr1 = one * (0.5 * (-1.0)) + (e2 ^ e3) * 0.00187 + (e1 ^ e2) * 0.161;
    //auto R_frwr1 = one*(0.986952634679) + ((e2^e3)*0.00187) + ((e1^e2)*0.161);
    //auto R_frwr2 = eval(exp((~(sqrt(e1*e1)))*0.5*e1*e1*e2*e3*(2.0*3.141/180.0)*(-1.0)));
    auto R_frwr2 = eval(exp(0.5 * e1 * e1 * e2 * e3 * (2.0 * 3.141 / 180.0) * (-1.0)));
    //auto R_frwr3 = eval(exp((~(sqrt(e3*e3)))*0.5*e3*e1*e2*e3*(0.5*3.141/180.0)));
    auto R_frwr3 = eval(exp(0.5 * e3 * e1 * e2 * e3 * (0.5 * 3.141 / 180.0)));
    auto T_fwp = eval(one + einf * (e1 * (-0.004) + e2 * 0.050 + e3 * 0.1028) * 0.5);
    auto D_wfr_ks = T_fwrr * R_fwrr * R_frwr1 * T_fwp * R_frwr3 * R_frwr2;

    //cm::mv<0,3,5,6>::type R_ks;
    //R_ks[0] = cos(-0.5*M_PI); R_ks[1] = sin(-0.5*M_PI); R_ks[2] = 0.0; R_ks[3] = 0.0;
    //D_wfr = R_ks*D_wfr_ks;
    auto T_sp_f = eval(one + einf * (e1 * (-0.8711) + e2 * 0.0 + e3 * (-0.30532)) * 0.5); //Translator Schwerpunkt

    //D_wfr = T_sp_f*D_wfr_ks;
    D_wfr = D_wfr_ks;

    auto p_fwr1 = eval(grade<1>(e3 * (-r_wheel) + einf * r_wheel * r_wheel * 0.5 + e0));
    auto p_wfr1 = eval(grade<1>(D_wfr * p_fwr1 * (~(D_wfr)))); //Radaufstandspunkt

    //Bestimmung Kraftaufteilung

    auto phi_flr = (p_flr1 ^ p_flr2 ^ p_flor ^ einf) * Ic; //Ebene Querlenker unten
    auto phi_fur = (p_fur1 ^ p_fur2 ^ p_fuor ^ einf) * Ic; //Ebene Querlenker oben
    auto ma_fr = eval((phi_flr ^ phi_fur) * Ic); //Momentanpolachse
    auto prc_wfr = (ma_fr ^ p_wfr1) * Ic; //Kraftebene
    auto nprc_wfr = ((prc_wfr ^ E) * E) * (!magnitude((prc_wfr ^ E) * E));
    auto nrc_wfr_ks = nprc_wfr * (1.0 / sqrt(eval(nprc_wfr & nprc_wfr)));

    //Drehung des Koordinatensytems um 180° um die z-Achse
    //nrc_wfr = R_ks*nrc_wfr_ks;
    nrc_wfr = T_sp_f * nrc_wfr_ks * (~T_sp_f);

    //Bestimmung Kraftaufteilung

    //auto phi_flr = (p_flr^p_flr2^p_flor^ei)*Ic; //Ebene Querlenker unten
    //auto phi_fur = (p_fur^p_fur2^p_fuor^ei)*Ic; //Ebene Querlenker oben
    //auto ma_fr = eval((phi_flr^phi_fur)*Ic); //Momentanpolachse

    sphereFLLTransform->setPosition(osg::Vec3(p_flr1[0], p_flr1[1], p_flr1[2]));
    sphereFLL2Transform->setPosition(osg::Vec3(p_flr2[0], p_flr2[1], p_flr2[2]));
    sphereFULTransform->setPosition(osg::Vec3(p_fur1[0], p_fur1[1], p_fur1[2]));
    sphereFUL2Transform->setPosition(osg::Vec3(p_fur2[0], p_fur2[1], p_fur2[2]));
    sphereFLOLTransform->setPosition(osg::Vec3(p_flor[0], p_flor[1], p_flor[2]));
    sphereFUOLTransform->setPosition(osg::Vec3(p_fuor[0], p_fuor[1], p_fuor[2]));
    sphereFSLTransform->setPosition(osg::Vec3(p_fsr[0], p_fsr[1], p_fsr[2]));
    sphereFSBLTransform->setPosition(osg::Vec3(p_fsbr[0], p_fsbr[1], p_fsbr[2]));
    sphereFBLTransform->setPosition(osg::Vec3(p_fbr[0], p_fbr[1], p_fbr[2]));
    sphereFPBLTransform->setPosition(osg::Vec3(p_fpbr[0], p_fpbr[1], p_fpbr[2]));
    sphereFTOLTransform->setPosition(osg::Vec3(p_ftor[0], p_ftor[1], p_ftor[2]));
    sphereFTLTransform->setPosition(osg::Vec3(p_ftr[0], p_ftr[1], p_ftr[2]));

    cylinderfl->setCenter(osg::Vec3(0.5 * (p_flr1[0] + p_flor[0]), 0.5 * (p_flr1[1] + p_flor[1]), 0.5 * (p_flr1[2] + p_flor[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_flr1 & p_flor).element<0>()));
    osg::Quat cylinderflrot;
    cylinderflrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_flr1[0] - p_flor[0], p_flr1[1] - p_flor[1], p_flr1[2] - p_flor[2]));
    cylinderfl->setRotation(cylinderflrot);
    cylinderflDrawable->dirtyDisplayList();

    cylinderfl2->setCenter(osg::Vec3(0.5 * (p_flr2[0] + p_flor[0]), 0.5 * (p_flr2[1] + p_flor[1]), 0.5 * (p_flr2[2] + p_flor[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_flr2 & p_flor).element<0>()));
    osg::Quat cylinderfl2rot;
    cylinderfl2rot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_flr2[0] - p_flor[0], p_flr2[1] - p_flor[1], p_flr2[2] - p_flor[2]));
    cylinderfl2->setRotation(cylinderfl2rot);
    cylinderfl2Drawable->dirtyDisplayList();

    cylinderfu->setCenter(osg::Vec3(0.5 * (p_fur1[0] + p_fuor[0]), 0.5 * (p_fur1[1] + p_fuor[1]), 0.5 * (p_fur1[2] + p_fuor[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fur1 & p_fuor).element<0>()));
    osg::Quat cylinderfurot;
    cylinderfurot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fur1[0] - p_fuor[0], p_fur1[1] - p_fuor[1], p_fur1[2] - p_fuor[2]));
    cylinderfu->setRotation(cylinderfurot);
    cylinderfuDrawable->dirtyDisplayList();

    cylinderfu2->setCenter(osg::Vec3(0.5 * (p_fur2[0] + p_fuor[0]), 0.5 * (p_fur2[1] + p_fuor[1]), 0.5 * (p_fur2[2] + p_fuor[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fur2 & p_fuor).element<0>()));
    osg::Quat cylinderfu2rot;
    cylinderfu2rot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fur2[0] - p_fuor[0], p_fur2[1] - p_fuor[1], p_fur2[2] - p_fuor[2]));
    cylinderfu2->setRotation(cylinderfu2rot);
    cylinderfu2Drawable->dirtyDisplayList();

    cylinderfp->setCenter(osg::Vec3(0.5 * (p_fpbr[0] + p_flor[0]), 0.5 * (p_fpbr[1] + p_flor[1]), 0.5 * (p_fpbr[2] + p_flor[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fpbr & p_flor).element<0>()));
    osg::Quat cylinderfprot;
    cylinderfprot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fpbr[0] - p_flor[0], p_fpbr[1] - p_flor[1], p_fpbr[2] - p_flor[2]));
    cylinderfp->setRotation(cylinderfprot);
    cylinderfpDrawable->dirtyDisplayList();

    cylinderfs->setCenter(osg::Vec3(0.5 * (p_fsbr[0] + p_fsr[0]), 0.5 * (p_fsbr[1] + p_fsr[1]), 0.5 * (p_fsbr[2] + p_fsr[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fsbr & p_fsr).element<0>()));
    osg::Quat cylinderfsrot;
    cylinderfsrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fsbr[0] - p_fsr[0], p_fsbr[1] - p_fsr[1], p_fsbr[2] - p_fsr[2]));
    cylinderfs->setRotation(cylinderfsrot);
    cylinderfsDrawable->dirtyDisplayList();

    cylinderfpb->setCenter(osg::Vec3(0.5 * (p_fbr[0] + p_fpbr[0]), 0.5 * (p_fbr[1] + p_fpbr[1]), 0.5 * (p_fbr[2] + p_fpbr[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fbr & p_fpbr).element<0>()));
    osg::Quat cylinderfpbrot;
    cylinderfpbrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fbr[0] - p_fpbr[0], p_fbr[1] - p_fpbr[1], p_fbr[2] - p_fpbr[2]));
    cylinderfpb->setRotation(cylinderfpbrot);
    cylinderfpbDrawable->dirtyDisplayList();

    cylinderfsb->setCenter(osg::Vec3(0.5 * (p_fbr[0] + p_fsbr[0]), 0.5 * (p_fbr[1] + p_fsbr[1]), 0.5 * (p_fbr[2] + p_fsbr[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fbr & p_fsbr).element<0>()));
    osg::Quat cylinderfsbrot;
    cylinderfsbrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fbr[0] - p_fsbr[0], p_fbr[1] - p_fsbr[1], p_fbr[2] - p_fsbr[2]));
    cylinderfsb->setRotation(cylinderfsbrot);
    cylinderfsbDrawable->dirtyDisplayList();

    cylinderfpsb->setCenter(osg::Vec3(0.5 * (p_fpbr[0] + p_fsbr[0]), 0.5 * (p_fpbr[1] + p_fsbr[1]), 0.5 * (p_fpbr[2] + p_fsbr[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fpbr & p_fsbr).element<0>()));
    osg::Quat cylinderfpsbrot;
    cylinderfpsbrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_fpbr[0] - p_fsbr[0], p_fpbr[1] - p_fsbr[1], p_fpbr[2] - p_fsbr[2]));
    cylinderfpsb->setRotation(cylinderfpsbrot);
    cylinderfpsbDrawable->dirtyDisplayList();

    cylinderftl->setCenter(osg::Vec3(0.5 * (p_ftr[0] + p_ftor[0]), 0.5 * (p_ftr[1] + p_ftor[1]), 0.5 * (p_ftr[2] + p_ftor[2])));
    cylinderfl->setHeight(sqrt(-2.0 * (p_fpbr & p_ftor).element<0>()));
    osg::Quat cylinderftlrot;
    cylinderftlrot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(p_ftr[0] - p_ftor[0], p_ftr[1] - p_ftor[1], p_ftr[2] - p_ftor[2]));
    cylinderftl->setRotation(cylinderftlrot);
    cylinderftlDrawable->dirtyDisplayList();
}

/*Querlenker::DL_type Querlenker::Radaufhaengung(double alpha, double gamma) 
{
	//Feder-Daempfer System
	auto r_fsl = e1*0.004 + e2*0.05159 + e3*0.58555;
        auto p_fsl = (grade<1>(r_fsl + (r_fsl%r_fsl)*ei*0.5 + en))();
	auto r_fbl = e1*0.004 + e2*0.266 + e3*0.456;
        auto p_fbl = (grade<1>(r_fbl + (r_fbl%r_fbl)*ei*0.5 + en))();
	//Querlenkerpunkt front lower frame left
	auto r_fll = e1*0.004 + e2*0.195 + e3*0.097;
        auto p_fll = (grade<1>(r_fll + (r_fll%r_fll)*ei*0.5 + en))();
	auto r_fll2 = e1*0.280 + e2*0.195 + e3*0.097;
        auto p_fll2 = (grade<1>(r_fll2 + (r_fll2%r_fll2)*ei*0.5 + en))();
	auto r_fll1 = e1*(-0.005) + e2*0.195 + e3*0.097;
        auto p_fll1 = (grade<1>(r_fll1 + (r_fll1%r_fll1)*ei*0.5 + en))();

	//double alpha = 0.0;

	double r_fsb  = 0.04633;
	double r_fsd  = 0.25772 - alpha;
	
        auto s_fsl = (grade<1>(p_fsl - ei*r_fsd*r_fsd*0.5))();
        auto s_fbsl = (grade<1>(p_fbl - ei*r_fsb*r_fsb*0.5))();
	auto c_fsbl = (s_fsl^s_fbsl);
	auto phi_fsd = (p_fll^p_fsl^p_fbl^ei)*I;
	auto Pp_fsbl = (phi_fsd^c_fsbl)*I;
	auto p_fsbl = (grade<1>((Pp_fsbl + sqrt(element<0x00>(Pp_fsbl%Pp_fsbl)))*(~(Pp_fsbl%ei))))();

	
	double r_fpb  = 0.0764;
	double r_fpsb = 0.05116;
	
	auto s_fsbl = grade<1>(p_fsbl - ei*r_fpsb*r_fpsb*0.5);
	auto s_fbl = grade<1>(p_fbl - ei*r_fpb*r_fpb*0.5);
	auto c_fpbl = (s_fsbl^s_fbl);
	auto Pp_fpbl = (phi_fsd^c_fpbl)*I;
	auto p_fpbl = (grade<1>((Pp_fpbl + sqrt(element<0x00>(Pp_fpbl%Pp_fpbl)))*(~(Pp_fpbl%ei))))();

	
	//Querlenker
		
	double r_fp  = 0.38418;
	double r_fl  = 0.35726;

	auto s_fll = grade<1>(p_fll - ei*r_fl*r_fl*0.5);
	auto s_fpbl = grade<1>(p_fpbl - ei*r_fp*r_fp*0.5);
	auto c_flol = (s_fpbl^s_fll);
	auto Pp_flol = (phi_fsd^c_flol)*I;
	auto p_flol = (grade<1>((Pp_flol + sqrt(element<0x00>(Pp_flol%Pp_flol)))*(~(Pp_flol%ei))))();


	auto r_ful = e1*0.037 + e2*0.288 + e3*0.261;
        auto p_ful = (grade<1>(r_ful + (r_ful%r_ful)*ei*0.5 + en))();
	auto r_ful2 = e1*0.210 + e2*0.288 + e3*0.261;
        auto p_ful2 = (grade<1>(r_ful2 + (r_ful2%r_ful2)*ei*0.5 + en))();
	auto r_ful1 = e1*(-0.1) + e2*0.288 + e3*0.261;
        auto p_ful1 = (grade<1>(r_ful1 + (r_ful1%r_ful1)*ei*0.5 + en))();

	double r_fo  = 0.21921;
	double r_fu  = 0.26086;

	//Punkte fuer Ebene oberer Querlenker
	auto r_phi1 = e1*0.037 + e2*0.0 + e3*0.0;
        auto p_phi1 = (grade<1>(r_phi1 + (r_phi1%r_phi1)*ei*0.5 + en))();
	auto r_phi2 = e1*0.037 + e2*1.0 + e3*0.0;
        auto p_phi2 = (grade<1>(r_phi2 + (r_phi2%r_phi2)*ei*0.5 + en))();
	
	auto s_ful = grade<1>(p_ful - ei*r_fu*r_fu*0.5);
	auto s_flol = grade<1>(p_flol - ei*r_fo*r_fo*0.5);
	auto c_flul = (s_flol^s_ful);
	auto phi_fuo = (p_ful^p_phi1^p_phi2^ei)*I;
	auto Pp_fuol = (phi_fuo^c_flul)*I;
	auto p_fuol = (grade<1>((Pp_fuol + sqrt(element<0x00>(Pp_fuol%Pp_fuol)))*(~(Pp_fuol%ei))))();


	//Spurstange
	auto r_ftl = e1*(-0.055) + e2*(0.204 + gamma) + e3*0.101; //Anbindungspunkt tie rod an Rahmen
        auto p_ftl = (grade<1>(r_ftl + (r_ftl%r_ftl)*ei*0.5 + en))();

	double r_ftr  = 0.39760; //Länge tie rod
	double r_fto  = 0.08377; //Abstand p_flol zu p_ftol
	double r_fuo  = 0.23717; //Abstand p_fuol zu p_ftol

	auto s_ftol = grade<1>(p_flol - ei*r_fto*r_fto*0.5);
	auto s_ftl = grade<1>(p_ftl - ei*r_ftr*r_ftr*0.5);
	auto s_fuol = grade<1>(p_fuol - ei*r_fuo*r_fuo*0.5);
	auto Pp_ftol = (s_ftol^s_ftl^s_fuol)*I;
	auto p_ftol = (grade<1>((Pp_ftol + sqrt(element<0x00>(Pp_ftol%Pp_ftol)))*(~(Pp_ftol%ei))))();


	//Bestimmung Radaufstandspunkt
	
	auto phi_fpol = (p_flol^p_fuol^p_ftol^ei)*I; //Ebene front points outer left
	auto phi_fpoln = (phi_fpol*(~(magnitude(phi_fpol)))*(-1.0))();
	auto T_fwrl = (one + ei*(p_flol - en)*0.5)(); //Definition Translator
	auto phi_fwrl = (en%(ei^(e2*sqrt(phi_fpoln*phi_fpoln) + phi_fpoln)))(); //Ebene front wheel reference left
	auto R_fwrl = part<4, 0x06050300>(phi_fwrl*e2); //Definition Rotor
	//auto p_frwl = e1*0.004603 - e2*0.003048 + e3*0.086825; //Vektor front rotation wheel left
	//auto R_frwl = (exp((~(sqrt(p_frwl*p_frwl)))*0.5*p_frwl*e1*e2*e3*(144.548*3.141/180.0)))();
	auto R_frwl1 = one*(0.5*(-1.0)) + (e2^e3)*0.00187 + (e1^e2)*0.161;
	auto R_frwl2 = (exp((~(sqrt(e1*e1)))*0.5*e1*e1*e2*e3*(2.0*3.141/180.0)*(-1.0)))();
	auto R_frwl3 = (exp((~(sqrt(e3*e3)))*0.5*e3*e1*e2*e3*(0.5*3.141/180.0)))();
	auto T_fwp = (one + ei*(e1*(-0.004) + e2*0.050 + e3*0.1028)*0.5)();
	auto D_fwp = (part<6, 0x171412110f0c>(T_fwrl*R_fwrl*R_frwl1*T_fwp*R_frwl3*R_frwl2)
                   + part<6, 0x0a0906050300>(T_fwrl*R_fwrl*R_frwl1*T_fwp*R_frwl3*R_frwl2))();
	

	//Bestimmung Kraftaufteilung

	auto phi_fll = (p_fll^p_fll2^p_flol^ei)*I; //Ebene Querlenker unten
	auto phi_ful = (p_ful^p_ful2^p_fuol^ei)*I; //Ebene Querlenker oben
	auto ma_fl = ((phi_fll^phi_ful)*I)(); //Momentanpolachse
			

	sphereFLLTransform->setPosition(osg::Vec3(p_fll1[0], p_fll1[1], p_fll1[2]));
	sphereFLL2Transform->setPosition(osg::Vec3(p_fll2[0], p_fll2[1], p_fll2[2]));
	sphereFULTransform->setPosition(osg::Vec3(p_ful1[0], p_ful1[1], p_ful1[2]));
	sphereFUL2Transform->setPosition(osg::Vec3(p_ful2[0], p_ful2[1], p_ful2[2]));
	sphereFLOLTransform->setPosition(osg::Vec3(p_flol[0], p_flol[1], p_flol[2]));
	sphereFUOLTransform->setPosition(osg::Vec3(p_fuol[0], p_fuol[1], p_fuol[2]));
	sphereFSLTransform->setPosition(osg::Vec3(p_fsl[0], p_fsl[1], p_fsl[2]));
	sphereFSBLTransform->setPosition(osg::Vec3(p_fsbl[0], p_fsbl[1], p_fsbl[2]));
	sphereFBLTransform->setPosition(osg::Vec3(p_fbl[0], p_fbl[1], p_fbl[2]));
	sphereFPBLTransform->setPosition(osg::Vec3(p_fpbl[0], p_fpbl[1], p_fpbl[2]));
	sphereFTOLTransform->setPosition(osg::Vec3(p_ftol[0], p_ftol[1], p_ftol[2]));
	sphereFTLTransform->setPosition(osg::Vec3(p_ftl[0], p_ftl[1], p_ftl[2]));

	cylinderfl->setCenter(osg::Vec3(0.5*(p_fll1[0] + p_flol[0]), 0.5*(p_fll1[1] + p_flol[1]), 0.5*(p_fll1[2] + p_flol[2])));
	cylinderfl->setHeight(sqrt(-2.0*(p_fll1%p_flol)()[0]));
	osg::Quat cylinderflrot;
	cylinderflrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fll1[0] - p_flol[0], p_fll1[1] - p_flol[1], p_fll1[2] - p_flol[2]));
	cylinderfl->setRotation(cylinderflrot);
	cylinderflDrawable->dirtyDisplayList();

	cylinderfl2->setCenter(osg::Vec3(0.5*(p_fll2[0] + p_flol[0]), 0.5*(p_fll2[1] + p_flol[1]), 0.5*(p_fll2[2] + p_flol[2])));
	cylinderfl2->setHeight(sqrt(-2.0*(p_fll2%p_flol)()[0]));
	osg::Quat cylinderfl2rot;
	cylinderfl2rot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fll2[0] - p_flol[0], p_fll2[1] - p_flol[1], p_fll2[2] - p_flol[2]));
	cylinderfl2->setRotation(cylinderfl2rot);
	cylinderfl2Drawable->dirtyDisplayList();

	cylinderfu->setCenter(osg::Vec3(0.5*(p_ful1[0] + p_fuol[0]), 0.5*(p_ful1[1] + p_fuol[1]), 0.5*(p_ful1[2] + p_fuol[2])));
	cylinderfu->setHeight(sqrt(-2.0*(p_ful1%p_fuol)()[0]));
	osg::Quat cylinderfurot;
	cylinderfurot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_ful1[0] - p_fuol[0], p_ful1[1] - p_fuol[1], p_ful1[2] - p_fuol[2]));
	cylinderfu->setRotation(cylinderfurot);
	cylinderfuDrawable->dirtyDisplayList();

	cylinderfu2->setCenter(osg::Vec3(0.5*(p_ful2[0] + p_fuol[0]), 0.5*(p_ful2[1] + p_fuol[1]), 0.5*(p_ful2[2] + p_fuol[2])));
	cylinderfu2->setHeight(sqrt(-2.0*(p_ful2%p_fuol)()[0]));
	osg::Quat cylinderfu2rot;
	cylinderfu2rot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_ful2[0] - p_fuol[0], p_ful2[1] - p_fuol[1], p_ful2[2] - p_fuol[2]));
	cylinderfu2->setRotation(cylinderfu2rot);
	cylinderfu2Drawable->dirtyDisplayList();

	cylinderfp->setCenter(osg::Vec3(0.5*(p_fpbl[0] + p_flol[0]), 0.5*(p_fpbl[1] + p_flol[1]), 0.5*(p_fpbl[2] + p_flol[2])));
	cylinderfp->setHeight(sqrt(-2.0*(p_fpbl%p_flol)()[0]));
	osg::Quat cylinderfprot;
	cylinderfprot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fpbl[0] - p_flol[0], p_fpbl[1] - p_flol[1], p_fpbl[2] - p_flol[2]));
	cylinderfp->setRotation(cylinderfprot);
	cylinderfpDrawable->dirtyDisplayList();

	cylinderfs->setCenter(osg::Vec3(0.5*(p_fsbl[0] + p_fsl[0]), 0.5*(p_fsbl[1] + p_fsl[1]), 0.5*(p_fsbl[2] + p_fsl[2])));
	cylinderfs->setHeight(sqrt(-2.0*(p_fsbl%p_fsl)()[0]));
	osg::Quat cylinderfsrot;
	cylinderfsrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fsbl[0] - p_fsl[0], p_fsbl[1] - p_fsl[1], p_fsbl[2] - p_fsl[2]));
	cylinderfs->setRotation(cylinderfsrot);
	cylinderfsDrawable->dirtyDisplayList();

	cylinderfpb->setCenter(osg::Vec3(0.5*(p_fbl[0] + p_fpbl[0]), 0.5*(p_fbl[1] + p_fpbl[1]), 0.5*(p_fbl[2] + p_fpbl[2])));
	cylinderfpb->setHeight(sqrt(-2.0*(p_fbl%p_fpbl)()[0]));
	osg::Quat cylinderfpbrot;
	cylinderfpbrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fbl[0] - p_fpbl[0], p_fbl[1] - p_fpbl[1], p_fbl[2] - p_fpbl[2]));
	cylinderfpb->setRotation(cylinderfpbrot);
	cylinderfpbDrawable->dirtyDisplayList();

	cylinderfsb->setCenter(osg::Vec3(0.5*(p_fbl[0] + p_fsbl[0]), 0.5*(p_fbl[1] + p_fsbl[1]), 0.5*(p_fbl[2] + p_fsbl[2])));
	cylinderfsb->setHeight(sqrt(-2.0*(p_fbl%p_fsbl)()[0]));
	osg::Quat cylinderfsbrot;
	cylinderfsbrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fbl[0] - p_fsbl[0], p_fbl[1] - p_fsbl[1], p_fbl[2] - p_fsbl[2]));
	cylinderfsb->setRotation(cylinderfsbrot);
	cylinderfsbDrawable->dirtyDisplayList();

	cylinderfpsb->setCenter(osg::Vec3(0.5*(p_fpbl[0] + p_fsbl[0]), 0.5*(p_fpbl[1] + p_fsbl[1]), 0.5*(p_fpbl[2] + p_fsbl[2])));
	cylinderfpsb->setHeight(sqrt(-2.0*(p_fpbl%p_fsbl)()[0]));
	osg::Quat cylinderfpsbrot;
	cylinderfpsbrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_fpbl[0] - p_fsbl[0], p_fpbl[1] - p_fsbl[1], p_fpbl[2] - p_fsbl[2]));
	cylinderfpsb->setRotation(cylinderfpsbrot);
	cylinderfpsbDrawable->dirtyDisplayList();

	cylinderftl->setCenter(osg::Vec3(0.5*(p_ftl[0] + p_ftol[0]), 0.5*(p_ftl[1] + p_ftol[1]), 0.5*(p_ftl[2] + p_ftol[2])));
	cylinderftl->setHeight(sqrt(-2.0*(p_fpbl%p_ftol)()[0]));
	osg::Quat cylinderftlrot;
	cylinderftlrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(p_ftl[0] - p_ftol[0], p_ftl[1] - p_ftol[1], p_ftl[2] - p_ftol[2]));
	cylinderftl->setRotation(cylinderftlrot);
	cylinderftlDrawable->dirtyDisplayList();

	return DL_type(D_fwp, ma_fl);
	
}*/

void Querlenker::preFrame()
{
    double frameTime = cover->frameDuration();

    alpha = alphaSlider->getValue();
    gamma = gammaSlider->getValue();

    /*DL_type Rad=Radaufhaengung(alpha, gamma);
	D_type& D_fwp = Rad.first; 
	L_type& ma_fl = Rad.second;	*/
    D_type D_fwp;
    Vector nrc_wfr;
    Radaufhaengung_wfr(alpha, gamma, D_fwp, nrc_wfr);

    double r_wheel = 0.255; //Reifenradius

    auto p_fwl1 = eval(grade<1>(e3 * (-r_wheel) + einf * r_wheel * r_wheel * 0.5 + e0));
    auto p_fwl2 = eval(grade<1>(e3 * r_wheel + einf * r_wheel * r_wheel * 0.5 + e0));
    auto p_fwl3 = eval(grade<1>(e1 * (-r_wheel) + einf * r_wheel * r_wheel * 0.5 + e0));
    auto p_fwl4 = eval(grade<1>(e1 * r_wheel + einf * r_wheel * r_wheel * 0.5 + e0));

    auto p_wfl1 = eval(grade<1>(D_fwp * p_fwl1 * (~(D_fwp)))); //Radaufstandspunkt
    auto p_wfl2 = eval(grade<1>(D_fwp * p_fwl2 * (~(D_fwp))));
    auto p_wfl3 = eval(grade<1>(D_fwp * p_fwl3 * (~(D_fwp))));
    auto p_wfl4 = eval(grade<1>(D_fwp * p_fwl4 * (~(D_fwp))));
    auto v_ref = eval(e1 * 0.0 + e2 * 0.0 + e3 * 1.0); //Referenzvektor Sturz
    auto v_wfl = eval(grade<1>(D_fwp * v_ref * (~(D_fwp)))); //momentaner Radsturzvektor
    auto ca_wfl = acos((v_ref & v_wfl).element<0>()); //Sturzwert
    std::cout << ca_wfl << std::endl;

    auto r_phi1 = e1 * 0.0 + e2 * 1.0 + e3 * 0.0;
    auto p_phi1 = eval(grade<1>(r_phi1 + (r_phi1 & r_phi1) * einf * 0.5 + e0));
    auto r_phi2 = e1 * 0.0 + e2 * 1.0 + e3 * 1.0;
    auto p_phi2 = eval(grade<1>(r_phi2 + (r_phi2 & r_phi2) * einf * 0.5 + e0));
    auto r_phi3 = e1 * 0.0 + e2 * 0.0 + e3 * 1.0;
    auto p_phi3 = eval(grade<1>(r_phi3 + (r_phi3 & r_phi3) * einf * 0.5 + e0));
    /*auto phi_yz = (p_phi1^p_phi2^p_phi3^ei)*I; //Schnittebene Momentanpolachse
	auto p_mafl = ((phi_yz^((ma_fl)*I))*I)();
	auto p_ma = ((p_mafl%en)*(~(p_mafl%E)))(); //Punkt auf der Momentanpolachse
	auto phi_ffl = (ma_fl^p_wfl1)*I; //Kraftebene
	auto ffv = (e1*0.3 + e2*0.7 + e3*(-1.0))(); //Beispielkraftvektor
	auto ffvp = ((ffv^phi_ffl)*(~(phi_ffl)))(); //Kraftkomponente auf den Aufbau
	auto ffvo = ((phi_ffl%ffv)*(~(phi_ffl)))(); //Kraftkomponente in Feder-Dämpfer Element
	
        std::cout << "Plane: " << phi_ffl << std::endl;
	
*/
    sphereWFL1Transform->setPosition(osg::Vec3(p_wfl1[0], p_wfl1[1], p_wfl1[2]));
    sphereWFL2Transform->setPosition(osg::Vec3(p_wfl2[0], p_wfl2[1], p_wfl2[2]));
    sphereWFL3Transform->setPosition(osg::Vec3(p_wfl3[0], p_wfl3[1], p_wfl3[2]));
    sphereWFL4Transform->setPosition(osg::Vec3(p_wfl4[0], p_wfl4[1], p_wfl4[2]));
    //sphereMATransform->setPosition(osg::Vec3(p_ma[0], p_ma[1], p_ma[2]));

    /*cylinderffv->setCenter(osg::Vec3(p_wfl1[0] + (ffv[0])*0.5, p_wfl1[1] + (ffv[1])*0.5, p_wfl1[2] + (ffv[2])*0.5));
	cylinderffv->setHeight(magnitude(ffv)()[0]);
	osg::Quat cylinderffvrot;
	cylinderffvrot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(ffv[0], ffv[1], ffv[2]));
	cylinderffv->setRotation(cylinderffvrot);
	cylinderffvDrawable->dirtyDisplayList();
	
	cylinderffvp->setCenter(osg::Vec3(p_wfl1[0] + (ffvp[0])*0.5, p_wfl1[1] + (ffvp[1])*0.5, p_wfl1[2] + (ffvp[2])*0.5));
	cylinderffvp->setHeight(magnitude(ffvp)()[0]);
	osg::Quat cylinderffvprot;
	cylinderffvprot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(ffvp[0], ffvp[1], ffvp[2]));
	cylinderffvp->setRotation(cylinderffvprot);
	cylinderffvpDrawable->dirtyDisplayList();
	
	cylinderffvo->setCenter(osg::Vec3(p_wfl1[0] + (ffvo[0])*0.5, p_wfl1[1] + (ffvo[1])*0.5, p_wfl1[2] + (ffvo[2])*0.5));
	cylinderffvo->setHeight(magnitude(ffvo)()[0]);
	osg::Quat cylinderffvorot;
	cylinderffvorot.makeRotate(osg::Vec3(0,0,1), osg::Vec3(ffvo[0], ffvo[1], ffvo[2]));
	cylinderffvo->setRotation(cylinderffvorot);
	cylinderffvoDrawable->dirtyDisplayList(); */
}
COVERPLUGIN(Querlenker)
