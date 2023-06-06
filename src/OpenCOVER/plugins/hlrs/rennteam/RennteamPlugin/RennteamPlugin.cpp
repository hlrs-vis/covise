/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Rennteam Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold		                                     **
 **                                                                          **
 ** History:  								     **
 ** Nov-01  v1	    				       		             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "RennteamPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRShader.h>
#include <cover/coVRFileManager.h>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <cover/RenderObject.h>
#include <osgDB/ReadFile>
#include <osg/Texture2D>

RennteamPlugin::RennteamPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, dy(cardyn::getExpressionVector())
, integrator(dy, y)
, ep(1.0)
, em(1.0)
, en((em - ep) * 0.5)
, ei(em + ep)
, E(ei ^ en)
, e1(1.0)
, e2(1.0)
, e3(1.0)
, I(e1 * e2 * e3 * ep * em)
, i(e1 * e2 * e3)
, one(1.0)
{
    fprintf(stderr, "RennteamPlugin::RennteamPlugin\n");

    /*get<0>(y)[0] = 1085.0;    //Initial position
   get<0>(y)[1] = 1501.0;    //Initial position
   get<0>(y)[2] = 11.0;    //Initial position
   //get<1>(y)[0] = -5.0;    //Initial velocity
   //get<1>(y)[1] = -1.0;    //Initial velocity
   get<2>(y)[0] = 1.0;    //Initial orientation (Important: magnitude be one!)
   //get<2>(y)[0] = 0.982131;  get<2>(y)[2] = 0.188203;   //Initial orientation (Important: magnitude be one!)
   //get<2>(y)[0] = cos(0.5*M_PI); get<2>(y)[1] = sin(0.5*M_PI);   //Initial orientation (Important: magnitude be one!)
   //get<3>(y)[1] = 0.3;    //Initial angular velocity
   //get<3>(y)[2] = 0.3;    //Initial angular velocity
   //get<3>(y)[0] = 1.0;    //Initial angular velocity
   //get<36>(y)[0] = cos(M_PI*0.1); get<36>(y)[1] = sin(M_PI*0.1);   //Initial steering wheel position
   get<37>(y)[0] = 1.0;   //Initial steering wheel position: magnitude be one!
   get<38>(y)[0] = 1.0;   //Initial steering wheel position: magnitude be one!
   get<39>(y)[0] = cardyn::i_a;   //Transmission
   std::tr1::get<39>(y)[0] = cardyn::i_g[gear+1]*cardyn::i_a;   //Transmission
   std::tr1::get<42>(y)[0] = 0.0;*/

    gear = 0;
    steerAngle = 0.0;

    /*cardyn::p_b.e_(y)[0] = 1085.0;
   cardyn::p_b.e_(y)[1] = 1501.0;
   cardyn::p_b.e_(y)[2] = 11.0;*/
    cardyn::p_b.e_(y)[0] = 5.0;
    cardyn::p_b.e_(y)[1] = 0.0;
    cardyn::p_b.e_(y)[2] = 1.0;
    cardyn::q_b.e_(y)[0] = 1.0;
    cardyn::q_wfl.e_(y)[0] = 1.0;
    cardyn::q_wfr.e_(y)[0] = 1.0;
    cardyn::i_pt.e_(y)[0] = cardyn::i_g[gear + 1] * cardyn::i_a;
    cardyn::k_c.e_(y)[0] = 0.0;

    /*cardyn::P_b.e_(y) = part<4, 0x06050403>(cardyn::p_b + cardyn::P_xy)(y);
   cardyn::k_Pp.e_(y)[0] = 10.0;
   cardyn::d_Pp.e_(y)[0] = 10.0;
   cardyn::k_Pq.e_(y)[0] = 10.0;
   cardyn::d_Pq.e_(y)[0] = 10.0;*/

    steerAngle = 0.0;

    gear = 0;

    currentRoad[0] = NULL;
    currentRoad[1] = NULL;
    currentRoad[2] = NULL;
    currentRoad[3] = NULL;
    currentLongPos[0] = 0.0;
    currentLongPos[1] = 0.0;
    currentLongPos[2] = 0.0;
    currentLongPos[3] = 0.0;

    v_wf[0] = 0.8711;
    v_wf[1] = 0.0;
    v_wf[2] = 0.30532;
    R_ks[0] = cos(-0.5 * M_PI);
    R_ks[1] = sin(-0.5 * M_PI);
    R_ks[2] = 0.0;
    R_ks[3] = 0.0;

    roadSystem = RoadSystem::Instance();
    //roadSystem->parseOpenDrive("sample1.1.xodr");

    /*roadGroup = new osg::Group;
   roadGroup->setName("RoadSystem");
   osg::StateSet* roadGroupState = roadGroup->getOrCreateStateSet();

   int numRoads = roadSystem->getNumRoads();
   coVRShader *shader=coVRShaderList::instance()->get("roadMark");
   if(shader==NULL)
   {
      cerr << "ERROR: no shader found with name: roadMark"<< endl;
   }
   for(int i=0; i<numRoads; ++i)
   {
      Road* road=roadSystem->getRoad(i);
      //if(!road->isJunctionPath())
      if(true)
      {
         osg::Geode *roadGeode=road->getRoadGeode();
         if(roadGeode) {
            //if(shader && (!road->isJunctionPath()))
            if(shader)
            {
               shader->apply(roadGeode);
            }
            roadGroup->addChild(roadGeode);
         }
      }
   }

   osg::Image* roadTexImage=NULL;
   const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/roadTex.jpg");
   if(fileName) {
      roadTexImage = osgDB::readImageFile(fileName);
      osg::Texture2D* roadTex = new osg::Texture2D;
      roadTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
      roadTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
      if(roadTexImage)
         roadTex->setImage(roadTexImage);
      roadGroupState->setTextureAttributeAndModes(0, roadTex);
   }
   else {
      std::cerr << "ERROR: no texture found named: share/covise/materials/roadTex.jpg";
   }
   if(roadGroup->getNumChildren()>0) {
      cover->getObjectsRoot()->addChild(roadGroup);
   }*/
}

bool RennteamPlugin::init()
{
    wheelShape = new osg::Cylinder(osg::Vec3(0, 0, 0), cardyn::r_w, 0.3);
    wheelDrawable = new osg::ShapeDrawable(wheelShape);
    wheelGeode = new osg::Geode();
    wheelGeode->addDrawable(wheelDrawable);
    wheelTransform = new osg::PositionAttitudeTransform();
    wheelTransform->addChild(wheelGeode);
    wheelTransform->setAttitude(osg::Quat(-M_PI / 2, osg::Vec3(1, 0, 0)));

    osg::Image *surfaceTexImage = osgDB::readImageFile("earth.png");
    if (surfaceTexImage)
    {
        osg::Texture2D *surfaceTex = new osg::Texture2D;
        surfaceTex->setImage(surfaceTexImage);
        surfaceTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
        surfaceTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
        osg::StateSet *wheelGeodeState = wheelGeode->getOrCreateStateSet();
        wheelGeodeState->setTextureAttributeAndModes(0, surfaceTex);
    }

    wheelTransformFL = new osg::PositionAttitudeTransform();
    wheelTransformFL->addChild(wheelTransform);
    wheelTransformFR = new osg::PositionAttitudeTransform();
    wheelTransformFR->addChild(wheelTransform);
    wheelTransformRL = new osg::PositionAttitudeTransform();
    wheelTransformRL->addChild(wheelTransform);
    wheelTransformRR = new osg::PositionAttitudeTransform();
    wheelTransformRR->addChild(wheelTransform);

    bodyShape = new osg::Box(osg::Vec3(0, 0, 0), cardyn::r_wfl2[0] - cardyn::r_wrl[0], cardyn::r_wfl2[1] - cardyn::r_wfr2[1], 2.0 * cardyn::r_wfl2[2]);
    bodyDrawable = new osg::ShapeDrawable(bodyShape);
    bodyGeode = new osg::Geode();
    bodyGeode->addDrawable(bodyDrawable);

    bodyTransform = new osg::PositionAttitudeTransform();
    bodyTransform->addChild(bodyGeode);
    cover->getObjectsRoot()->addChild(bodyTransform);

    bodyTransform->addChild(wheelTransformFL);
    bodyTransform->addChild(wheelTransformFR);
    bodyTransform->addChild(wheelTransformRL);
    bodyTransform->addChild(wheelTransformRR);

    planeGeode = new osg::Geode();
    planeTransform = new osg::PositionAttitudeTransform();
    planeTransform->addChild(planeGeode);
    //cover->getObjectsRoot()->addChild(planeTransform);

    osg::Geometry *planeGeometry;
    planeGeometry = new osg::Geometry();
    planeGeode->addDrawable(planeGeometry);

    osg::Vec3Array *planeVertices;
    planeVertices = new osg::Vec3Array;
    planeGeometry->setVertexArray(planeVertices);

    osg::Vec3Array *planeNormals;
    planeNormals = new osg::Vec3Array;
    planeGeometry->setNormalArray(planeNormals);
    planeGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec3Array *planeColors = new osg::Vec3Array;
    planeGeometry->setColorArray(planeColors);

    planeVertices->push_back(osg::Vec3(-2000.0, -2000.0, 0.0));
    planeVertices->push_back(osg::Vec3(-2000.0, 2000.0, 0.0));
    planeVertices->push_back(osg::Vec3(2000.0, -2000.0, 0.0));
    planeVertices->push_back(osg::Vec3(2000.0, 2000.0, 0.0));
    planeNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeColors->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeColors->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeColors->push_back(osg::Vec3(0.0, 0.0, 1.0));
    planeColors->push_back(osg::Vec3(0.0, 0.0, 1.0));

    osg::DrawArrays *planeBase;
    planeBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, 4);
    planeGeometry->addPrimitiveSet(planeBase);

    vdTab = new coTUITab("Xenomai", coVRTui::instance()->mainFolder->getID());
    vdTab->setPos(0, 0);
    k_Pp_Label = new coTUILabel("k_Pp:", vdTab->getID());
    k_Pp_Label->setPos(0, 0);
    d_Pp_Label = new coTUILabel("d_Pp:", vdTab->getID());
    d_Pp_Label->setPos(2, 0);
    k_Pq_Label = new coTUILabel("k_Pq:", vdTab->getID());
    k_Pq_Label->setPos(4, 0);
    d_Pq_Label = new coTUILabel("d_Pq:", vdTab->getID());
    d_Pq_Label->setPos(6, 0);

    k_Pp_Slider = new coTUISlider("k_Pp slider", vdTab->getID());
    k_Pp_Slider->setRange(0, 1000);
    k_Pp_Slider->setValue(10);
    k_Pp_Slider->setPos(0, 1);
    d_Pp_Slider = new coTUISlider("d_Pp slider", vdTab->getID());
    d_Pp_Slider->setRange(0, 1000);
    d_Pp_Slider->setValue(10);
    d_Pp_Slider->setPos(2, 1);
    k_Pq_Slider = new coTUISlider("k_Pq slider", vdTab->getID());
    k_Pq_Slider->setRange(0, 1000);
    k_Pq_Slider->setValue(10);
    k_Pq_Slider->setPos(4, 1);
    d_Pq_Slider = new coTUISlider("d_Pq slider", vdTab->getID());
    d_Pq_Slider->setRange(0, 1000);
    d_Pq_Slider->setValue(10);
    d_Pq_Slider->setPos(6, 1);

    return true;
}

// this is called if the plugin is removed at runtime
RennteamPlugin::~RennteamPlugin()
{
    fprintf(stderr, "RennteamPlugin::~RennteamPlugin\n");
    cover->getObjectsRoot()->removeChild(wheelTransformFL);
    cover->getObjectsRoot()->removeChild(wheelTransformFR);
    cover->getObjectsRoot()->removeChild(wheelTransformRL);
    cover->getObjectsRoot()->removeChild(wheelTransformRR);
    cover->getObjectsRoot()->removeChild(bodyTransform);
    cover->getObjectsRoot()->removeChild(planeTransform);
}

RennteamPlugin::DP_type RennteamPlugin::Radaufhaengung_wfr(double u_wfr, double steerAngle)
{

    //Feder-Daempfer System
    auto r_fsr = e1 * 0.004 + e2 * 0.05159 + e3 * 0.58555;
    auto p_fsr = (grade<1>(r_fsr + (r_fsr % r_fsr) * ei * 0.5 + en))();
    auto r_fbr = e1 * 0.004 + e2 * 0.266 + e3 * 0.456;
    auto p_fbr = (grade<1>(r_fbr + (r_fbr % r_fbr) * ei * 0.5 + en))();
    //Querlenkerpunkt front lower frame left
    auto r_flr = e1 * 0.004 + e2 * 0.195 + e3 * 0.097;
    auto p_flr = (grade<1>(r_flr + (r_flr % r_flr) * ei * 0.5 + en))();
    auto r_flr2 = e1 * 0.280 + e2 * 0.195 + e3 * 0.097;
    auto p_flr2 = (grade<1>(r_flr2 + (r_flr2 % r_flr2) * ei * 0.5 + en))();

    double r_fsb = 0.04633;
    double r_fsd = 0.25772 - u_wfr;

    auto s_fsr = (grade<1>(p_fsr - ei * r_fsd * r_fsd * 0.5))();
    auto s_fbsr = (grade<1>(p_fbr - ei * r_fsb * r_fsb * 0.5))();
    auto c_fsbr = (s_fsr ^ s_fbsr);
    auto phi_fsd = (p_flr ^ p_fsr ^ p_fbr ^ ei) * I;
    auto Pp_fsbr = (phi_fsd ^ c_fsbr) * I;
    auto p_fsbr = (grade<1>((Pp_fsbr + sqrt(element<0x00>(Pp_fsbr % Pp_fsbr))) * (~(Pp_fsbr % ei))))();

    double r_fpb = 0.0764;
    double r_fpsb = 0.05116;

    auto s_fsbr = grade<1>(p_fsbr - ei * r_fpsb * r_fpsb * 0.5);
    auto s_fbr = grade<1>(p_fbr - ei * r_fpb * r_fpb * 0.5);
    auto c_fpbr = (s_fsbr ^ s_fbr);
    auto Pp_fpbr = (phi_fsd ^ c_fpbr) * I;
    auto p_fpbr = (grade<1>((Pp_fpbr + sqrt(element<0x00>(Pp_fpbr % Pp_fpbr))) * (~(Pp_fpbr % ei))))();

    //Querlenker

    double r_fp = 0.38418;
    double r_fl = 0.35726;

    auto s_flr = grade<1>(p_flr - ei * r_fl * r_fl * 0.5);
    auto s_fpbr = grade<1>(p_fpbr - ei * r_fp * r_fp * 0.5);
    auto c_flor = (s_fpbr ^ s_flr);
    auto Pp_flor = (phi_fsd ^ c_flor) * I;
    auto p_flor = (grade<1>((Pp_flor + sqrt(element<0x00>(Pp_flor % Pp_flor))) * (~(Pp_flor % ei))))();

    auto r_fur = e1 * 0.037 + e2 * 0.288 + e3 * 0.261;
    auto p_fur = (grade<1>(r_fur + (r_fur % r_fur) * ei * 0.5 + en))();
    auto r_fur2 = e1 * 0.210 + e2 * 0.288 + e3 * 0.261;
    auto p_fur2 = (grade<1>(r_fur2 + (r_fur2 % r_fur2) * ei * 0.5 + en))();

    double r_fo = 0.21921;
    double r_fu = 0.26086;

    //Punkte fuer Ebene oberer Querlenker
    auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
    auto p_phi1 = (grade<1>(r_phi1 + (r_phi1 % r_phi1) * ei * 0.5 + en))();
    auto r_phi2 = e1 * 0.037 + e2 * 1.0 + e3 * 0.0;
    auto p_phi2 = (grade<1>(r_phi2 + (r_phi2 % r_phi2) * ei * 0.5 + en))();

    auto s_fur = grade<1>(p_fur - ei * r_fu * r_fu * 0.5);
    auto s_flor = grade<1>(p_flor - ei * r_fo * r_fo * 0.5);
    auto c_flur = (s_flor ^ s_fur);
    auto phi_fuo = (p_fur ^ p_phi1 ^ p_phi2 ^ ei) * I;
    auto Pp_fuor = (phi_fuo ^ c_flur) * I;
    auto p_fuor = (grade<1>((Pp_fuor + sqrt(element<0x00>(Pp_fuor % Pp_fuor))) * (~(Pp_fuor % ei))))();

    //Spurstange
    auto r_ftr = e1 * (-0.055) + e2 * (0.204 + steerAngle) + e3 * 0.101; //Anbindungspunkt tie rod an Rahmen
    auto p_ftr = (grade<1>(r_ftr + (r_ftr % r_ftr) * ei * 0.5 + en))();

    double r_ft = 0.39760; //Länge tie rod
    double r_fto = 0.08377; //Abstand p_flol zu p_ftol
    double r_fuo = 0.23717; //Abstand p_fuol zu p_ftol

    auto s_ftor = grade<1>(p_flor - ei * r_fto * r_fto * 0.5);
    auto s_ftr = grade<1>(p_ftr - ei * r_ft * r_ft * 0.5);
    auto s_fuor = grade<1>(p_fuor - ei * r_fuo * r_fuo * 0.5);
    auto Pp_ftor = (s_ftor ^ s_ftr ^ s_fuor) * I;
    auto p_ftor = (grade<1>((Pp_ftor + sqrt(element<0x00>(Pp_ftor % Pp_ftor))) * (~(Pp_ftor % ei))))();

    //Bestimmung Radaufstandspunkt

    double r_wheel = 0.255; //Reifenradius
    auto phi_fpor = (p_flor ^ p_fuor ^ p_ftor ^ ei) * I; //Ebene front points outer left
    auto phi_fporn = (phi_fpor * (~(magnitude(phi_fpor))) * (-1.0))();
    auto T_fwrr = (one + ei * (p_flor - en) * 0.5)(); //Definition Translator
    auto phi_fwrr = (en % (ei ^ (e2 * sqrt(phi_fporn * phi_fporn) + phi_fporn)))(); //Ebene front wheel reference left
    auto R_fwrr = part<4, 0x06050300>(phi_fwrr * e2); //Definition Rotor
    auto R_frwr1 = one * (0.5 * (-1.0)) + (e2 ^ e3) * 0.00187 + (e1 ^ e2) * 0.161;
    auto R_frwr2 = (exp((~(sqrt(e1 * e1))) * 0.5 * e1 * e1 * e2 * e3 * (2.0 * 3.141 / 180.0) * (-1.0)))();
    auto R_frwr3 = (exp((~(sqrt(e3 * e3))) * 0.5 * e3 * e1 * e2 * e3 * (0.5 * 3.141 / 180.0)))();
    auto T_fwp = (one + ei * (e1 * (-0.004) + e2 * 0.050 + e3 * 0.1028) * 0.5)();
    auto D_wfr = (part<6, 0x171412110f0c>(T_fwrr * R_fwrr * R_frwr1 * T_fwp * R_frwr3 * R_frwr2)
                  + part<6, 0x0a0906050300>(T_fwrr * R_fwrr * R_frwr1 * T_fwp * R_frwr3 * R_frwr2))();
    auto p_fwr1 = (grade<1>(e3 * (-r_wheel) + ei * r_wheel * r_wheel * 0.5 + en))();
    auto p_wfr1 = (grade<1>(D_wfr * p_fwr1 * (~(D_wfr))))(); //Radaufstandspunkt

    //Bestimmung Kraftaufteilung

    auto phi_flr = (p_flr ^ p_flr2 ^ p_flor ^ ei) * I; //Ebene Querlenker unten
    auto phi_fur = (p_fur ^ p_fur2 ^ p_fuor ^ ei) * I; //Ebene Querlenker oben
    auto ma_fr = ((phi_flr ^ phi_fur) * I)(); //Momentanpolachse
    auto phi_ffr = (ma_fr ^ p_wfr1) * I; //Kraftebene

    return DP_type(D_wfr, phi_ffr);
}

RennteamPlugin::DP_type RennteamPlugin::Radaufhaengung_wfl(double u_wfl, double steerAngle)
{

    //Feder-Daempfer System
    auto r_fsl = e1 * 0.004 + (e2 * 0.05159 * (-1.0)) + e3 * 0.58555;
    auto p_fsl = (grade<1>(r_fsl + (r_fsl % r_fsl) * ei * 0.5 + en))();
    auto r_fbl = e1 * 0.004 + (e2 * 0.266 * (-1.0)) + e3 * 0.456;
    auto p_fbl = (grade<1>(r_fbl + (r_fbl % r_fbl) * ei * 0.5 + en))();
    //Querlenkerpunkt front lower frame left
    auto r_fll = e1 * 0.004 + (e2 * 0.195 * (-1.0)) + e3 * 0.097;
    auto p_fll = (grade<1>(r_fll + (r_fll % r_fll) * ei * 0.5 + en))();
    auto r_fll2 = e1 * 0.280 + (e2 * 0.195 * (-1.0)) + e3 * 0.097;
    auto p_fll2 = (grade<1>(r_fll2 + (r_fll2 % r_fll2) * ei * 0.5 + en))();

    double r_fsb = 0.04633;
    double r_fsd = 0.25772 - u_wfl;

    auto s_fsl = (grade<1>(p_fsl - ei * r_fsd * r_fsd * 0.5))();
    auto s_fbsl = (grade<1>(p_fbl - ei * r_fsb * r_fsb * 0.5))();
    auto c_fsbl = (s_fsl ^ s_fbsl);
    auto phi_fsd = (p_fll ^ p_fsl ^ p_fbl ^ ei) * I;
    auto Pp_fsbl = (phi_fsd ^ c_fsbl) * I;
    auto p_fsbl = (grade<1>((Pp_fsbl + (sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)) * (-1.0))) * (~(Pp_fsbl % ei))))();

    double r_fpb = 0.0764;
    double r_fpsb = 0.05116;

    auto s_fsbl = grade<1>(p_fsbl - ei * r_fpsb * r_fpsb * 0.5);
    auto s_fbl = grade<1>(p_fbl - ei * r_fpb * r_fpb * 0.5);
    auto c_fpbl = (s_fsbl ^ s_fbl);
    auto Pp_fpbl = (phi_fsd ^ c_fpbl) * I;
    auto p_fpbl = (grade<1>((Pp_fpbl + (sqrt(element<0x00>(Pp_fpbl % Pp_fpbl)) * (-1.0))) * (~(Pp_fpbl % ei))))();

    //Querlenker

    double r_fp = 0.38418;
    double r_fl = 0.35726;

    auto s_fll = grade<1>(p_fll - ei * r_fl * r_fl * 0.5);
    auto s_fpbl = grade<1>(p_fpbl - ei * r_fp * r_fp * 0.5);
    auto c_flol = (s_fpbl ^ s_fll);
    auto Pp_flol = (phi_fsd ^ c_flol) * I;
    auto p_flol = (grade<1>((Pp_flol + (sqrt(element<0x00>(Pp_flol % Pp_flol)) * (-1.0))) * (~(Pp_flol % ei))))();

    auto r_ful = e1 * 0.037 + (e2 * 0.288 * (-1.0)) + e3 * 0.261;
    auto p_ful = (grade<1>(r_ful + (r_ful % r_ful) * ei * 0.5 + en))();
    auto r_ful2 = e1 * 0.210 + (e2 * 0.288 * (-1.0)) + e3 * 0.261;
    auto p_ful2 = (grade<1>(r_ful2 + (r_ful2 % r_ful2) * ei * 0.5 + en))();

    double r_fo = 0.21921;
    double r_fu = 0.26086;

    //Punkte fuer Ebene oberer Querlenker
    auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
    auto p_phi1 = (grade<1>(r_phi1 + (r_phi1 % r_phi1) * ei * 0.5 + en))();
    auto r_phi2 = e1 * 0.037 + (e2 * (-1.0)) + e3 * 0.0;
    auto p_phi2 = (grade<1>(r_phi2 + (r_phi2 % r_phi2) * ei * 0.5 + en))();

    auto s_ful = grade<1>(p_ful - ei * r_fu * r_fu * 0.5);
    auto s_flol = grade<1>(p_flol - ei * r_fo * r_fo * 0.5);
    auto c_flul = (s_flol ^ s_ful);
    auto phi_fuo = (p_ful ^ p_phi1 ^ p_phi2 ^ ei) * I;
    auto Pp_fuol = (phi_fuo ^ c_flul) * I;
    auto p_fuol = (grade<1>((Pp_fuol + (sqrt(element<0x00>(Pp_fuol % Pp_fuol)) * (-1.0))) * (~(Pp_fuol % ei))))();

    //Spurstange
    auto r_ftl = e1 * (-0.055) + e2 * ((0.204 * (-1.0)) + steerAngle) + e3 * 0.101; //Anbindungspunkt tie rod an Rahmen
    auto p_ftl = (grade<1>(r_ftl + (r_ftl % r_ftl) * ei * 0.5 + en))();

    double r_ft = 0.39760; //Länge tie rod
    double r_fto = 0.08377; //Abstand p_flol zu p_ftol
    double r_fuo = 0.23717; //Abstand p_fuol zu p_ftol

    auto s_ftol = grade<1>(p_flol - ei * r_fto * r_fto * 0.5);
    auto s_ftl = grade<1>(p_ftl - ei * r_ft * r_ft * 0.5);
    auto s_fuol = grade<1>(p_fuol - ei * r_fuo * r_fuo * 0.5);
    auto Pp_ftol = (s_ftol ^ s_ftl ^ s_fuol) * I;
    auto p_ftol = (grade<1>((Pp_ftol + sqrt(element<0x00>(Pp_ftol % Pp_ftol))) * (~(Pp_ftol % ei))))();

    //Bestimmung Radaufstandspunkt

    double r_wheel = 0.255; //Reifenradius
    auto phi_fpol = (p_flol ^ p_fuol ^ p_ftol ^ ei) * I; //Ebene front points outer left
    auto phi_fpoln = (phi_fpol * (~(magnitude(phi_fpol))) * (-1.0))();
    auto T_fwrl = (one + ei * (p_flol - en) * 0.5)(); //Definition Translator
    auto phi_fwrl = (en % (ei ^ (e2 * sqrt(phi_fpoln * phi_fpoln) + phi_fpoln)))(); //Ebene front wheel reference left
    auto R_fwrl = part<4, 0x06050300>(phi_fwrl * e2); //Definition Rotor
    auto R_frwl1 = one * (0.5 * (-1.0)) + ((e2 ^ e3) * 0.00187 * (-1.0)) + ((e1 ^ e2) * 0.161 * (-1.0));
    auto R_frwl2 = (exp((~(sqrt(e1 * e1))) * 0.5 * e1 * e1 * e2 * e3 * (2.0 * 3.141 / 180.0) * (-1.0)))();
    auto R_frwl3 = (exp((~(sqrt(e3 * e3))) * 0.5 * e3 * e1 * e2 * e3 * (0.5 * 3.141 / 180.0)))();
    auto T_fwp = (one + ei * (e1 * (-0.004) + (e2 * 0.050 * (-1.0)) + e3 * 0.1028) * 0.5)();
    auto D_wfl = (part<6, 0x171412110f0c>(T_fwrl * R_fwrl * R_frwl1 * T_fwp * R_frwl3 * R_frwl2)
                  + part<6, 0x0a0906050300>(T_fwrl * R_fwrl * R_frwl1 * T_fwp * R_frwl3 * R_frwl2))();
    auto p_fwl1 = (grade<1>(e3 * (-r_wheel) + ei * r_wheel * r_wheel * 0.5 + en))();
    auto p_wfl1 = (grade<1>(D_wfl * p_fwl1 * (~(D_wfl))))(); //Radaufstandspunkt

    //Bestimmung Kraftaufteilung

    auto phi_fll = (p_fll ^ p_fll2 ^ p_flol ^ ei) * I; //Ebene Querlenker unten
    auto phi_ful = (p_ful ^ p_ful2 ^ p_fuol ^ ei) * I; //Ebene Querlenker oben
    auto ma_fl = ((phi_fll ^ phi_ful) * I)(); //Momentanpolachse
    auto phi_ffl = (ma_fl ^ p_wfl1) * I; //Kraftebene

    return DP_type(D_wfl, phi_ffl);
}

void
RennteamPlugin::preFrame()
{
    double dt = cover->frameDuration();

    //get<33>(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfl-cardyn::z*cardyn::r_w-cardyn::u_wfl)*cardyn::q_b))(y));
    //get<34>(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfr-cardyn::z*cardyn::r_w-cardyn::u_wfr)*cardyn::q_b))(y));
    //get<35>(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wrl-cardyn::z*cardyn::r_w-cardyn::u_wrl)*cardyn::q_b))(y));
    //get<36>(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wrr-cardyn::z*cardyn::r_w-cardyn::u_wrr)*cardyn::q_b))(y));
    /*cardyn::Dv_wfl.e_(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfl-cardyn::z*cardyn::r_w-cardyn::u_wfl)*cardyn::q_b))(y));
   cardyn::Dv_wfr.e_(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wfr-cardyn::z*cardyn::r_w-cardyn::u_wfr)*cardyn::q_b))(y));
   cardyn::Dv_wrl.e_(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wrl-cardyn::z*cardyn::r_w-cardyn::u_wrl)*cardyn::q_b))(y));
   cardyn::Dv_wrr.e_(y) = getGroundDistance((cardyn::p_b + grade<1>((!cardyn::q_b)*(cardyn::r_wrr-cardyn::z*cardyn::r_w-cardyn::u_wrr)*cardyn::q_b))(y));*/

    DP_type Rad_wfl = Radaufhaengung_wfl(cardyn::u_wfl(y)[0], steerAngle);
    cardyn::D_wfl.e_(y) = R_ks * Rad_wfl.first;
    //cardyn::P_wfl.e_(y) = Rad.second;
    gealg::mv<3, 0x040201, 0x10>::type r_wfl_tmp = part<3, 0x040201>(cardyn::D_wfl * (e3 * (-0.255) + ei * 0.255 * 0.255 * 0.5 + en) * (~cardyn::D_wfl))(y) + v_wf;
    cardyn::r_wfl.e_(y) = *reinterpret_cast<gealg::mv<3, 0x040201>::type *>(&r_wfl_tmp);
    gealg::mv<3, 0x040201, 0x10>::type n_nfl_tmp = part<3, 0x040201>((Rad_wfl.second ^ E) * E);
    n_nfl_tmp = n_nfl_tmp * (~magnitude(n_nfl_tmp));
    cardyn::n_nfl.e_(y) = *reinterpret_cast<gealg::mv<3, 0x040201>::type *>(&n_nfl_tmp);

    DP_type Rad_wfr = Radaufhaengung_wfr(cardyn::u_wfr(y)[0], steerAngle);
    cardyn::D_wfr.e_(y) = R_ks * Rad_wfr.first;
    //cardyn::P_wfl.e_(y) = Rad.second;
    gealg::mv<3, 0x040201, 0x10>::type r_wfr_tmp = part<3, 0x040201>(cardyn::D_wfr * (e3 * (-0.255) + ei * 0.255 * 0.255 * 0.5 + en) * (~cardyn::D_wfr))(y) + v_wf;
    cardyn::r_wfr.e_(y) = *reinterpret_cast<gealg::mv<3, 0x040201>::type *>(&r_wfr_tmp);
    gealg::mv<3, 0x040201, 0x10>::type n_nfr_tmp = part<3, 0x040201>((Rad_wfr.second ^ E) * E);
    n_nfr_tmp = n_nfr_tmp * (~magnitude(n_nfr_tmp));
    cardyn::n_nfr.e_(y) = *reinterpret_cast<gealg::mv<3, 0x040201>::type *>(&n_nfr_tmp);

    gealg::mv<3, 0x040201>::type p_cwfl = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfl) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_cwfr = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wfr) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_cwrl = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrl - cardyn::u_wrl) * cardyn::q_b))(y);
    gealg::mv<3, 0x040201>::type p_cwrr = (cardyn::p_b + grade<1>((!cardyn::q_b) * (cardyn::r_wrr - cardyn::u_wrr) * cardyn::q_b))(y);

    cardyn::Dv_wfl.e_(y) = part<1, 0x04>(p_cwfl - grade<1>(getContactPoint(p_cwfl, currentRoad[0], currentLongPos[0]))) - cardyn::z * cardyn::r_w;
    cardyn::Dv_wfr.e_(y) = part<1, 0x04>(p_cwfr - grade<1>(getContactPoint(p_cwfr, currentRoad[1], currentLongPos[1]))) - cardyn::z * cardyn::r_w;
    cardyn::Dv_wrl.e_(y) = part<1, 0x04>(p_cwrl - grade<1>(getContactPoint(p_cwrl, currentRoad[2], currentLongPos[2]))) - cardyn::z * cardyn::r_w;
    cardyn::Dv_wrr.e_(y) = part<1, 0x04>(p_cwrr - grade<1>(getContactPoint(p_cwrr, currentRoad[3], currentLongPos[3]))) - cardyn::z * cardyn::r_w;

    //std::cout << "Distance: fl: " << cardyn::Dv_wfl.e_(y) << ", fr: " << cardyn::Dv_wfr.e_(y) << std::endl;

    if (steerAngle > (-M_PI * 0.4) && steerAngle < (M_PI * 0.4))
    {
        double cotSteerAngle = (cardyn::r_wfl2[0] - cardyn::r_wrl[0]) * (1.0 / tan(steerAngle));

        double angleFL = atan(1.0 / (cotSteerAngle - cardyn::w_wn / (cardyn::v_wn * 2.0)));
        double angleFR = atan(1.0 / (cotSteerAngle + cardyn::w_wn / (cardyn::v_wn * 2.0)));
        cardyn::q_wfl.e_(y)[0] = cos(angleFL * 0.5);
        cardyn::q_wfl.e_(y)[1] = sin(angleFL * 0.5);
        cardyn::q_wfr.e_(y)[0] = cos(angleFR * 0.5);
        cardyn::q_wfr.e_(y)[1] = sin(angleFR * 0.5);
    }

    integrator.integrate(dt);

    bodyTransform->setPosition(osg::Vec3(cardyn::p_b(y)[0], cardyn::p_b(y)[1], cardyn::p_b(y)[2]));
    bodyTransform->setAttitude(osg::Quat(cardyn::q_b(y)[3],
                                         -cardyn::q_b(y)[2],
                                         cardyn::q_b(y)[1],
                                         cardyn::q_b(y)[0]));

    //osg::Quat steerRot(std::tr1::get<36>(y)[0], osg::Vec3(0,0,1));
    osg::Quat steerRotFL(0, 0, cardyn::q_wfl(y)[1], cardyn::q_wfl(y)[0]);
    osg::Quat steerRotFR(0, 0, cardyn::q_wfr(y)[1], cardyn::q_wfr(y)[0]);

    osg::Quat camberRotFL(cardyn::R_wfl(y)[3], -cardyn::R_wfl(y)[2], cardyn::R_wfl(y)[1], cardyn::R_wfl(y)[0]);
    osg::Quat camberRotFR(cardyn::R_wfr(y)[3], -cardyn::R_wfr(y)[2], cardyn::R_wfr(y)[1], cardyn::R_wfr(y)[0]);
    osg::Quat camberRotRL(cardyn::R_wrl(y)[3], -cardyn::R_wrl(y)[2], cardyn::R_wrl(y)[1], cardyn::R_wrl(y)[0]);
    osg::Quat camberRotRR(cardyn::R_wrr(y)[3], -cardyn::R_wrr(y)[2], cardyn::R_wrr(y)[1], cardyn::R_wrr(y)[0]);

    std::cout << cardyn::p_b(y) << " \t" << std::endl;
    std::cout << cardyn::u_wfl(y) << " \t" << p_cwfl << std::endl;
    std::cout << cardyn::u_wfr(y) << " \t" << p_cwfr << std::endl;
    wheelTransformFL->setPosition(osg::Vec3(p_cwfl[0], p_cwfl[1], p_cwfl[2]));
    osg::Quat wheelRotarySpeedFL(0.0, cardyn::w_wfl(y)[0], 0.0, 0.0);
    wheelQuatFL = wheelQuatFL + wheelQuatFL * wheelRotarySpeedFL * (0.5 * dt);
    wheelQuatFL = wheelQuatFL * (1 / wheelQuatFL.length());
    wheelTransformFL->setAttitude(wheelQuatFL * camberRotFL * steerRotFL);

    wheelTransformFR->setPosition(osg::Vec3(p_cwfr[0], p_cwfr[1], p_cwfr[2]));
    osg::Quat wheelRotarySpeedFR(0.0, cardyn::w_wfr(y)[0], 0.0, 0.0);
    wheelQuatFR = wheelQuatFR + wheelQuatFR * wheelRotarySpeedFR * (0.5 * dt);
    wheelQuatFR = wheelQuatFR * (1 / wheelQuatFR.length());
    wheelTransformFR->setAttitude(wheelQuatFR * camberRotFR * steerRotFR);

    wheelTransformRL->setPosition(osg::Vec3(-cardyn::v_wn, cardyn::w_wn, -cardyn::u_wn - cardyn::u_wrl(y)[0]));
    osg::Quat wheelRotarySpeedRL(0.0, cardyn::w_wrl(y)[0], 0.0, 0.0);
    wheelQuatRL = wheelQuatRL + wheelQuatRL * wheelRotarySpeedRL * (0.5 * dt);
    wheelQuatRL = wheelQuatRL * (1 / wheelQuatRL.length());
    wheelTransformRL->setAttitude(wheelQuatRL * camberRotRL);

    wheelTransformRR->setPosition(osg::Vec3(-cardyn::v_wn, -cardyn::w_wn, -cardyn::u_wn - cardyn::u_wrr(y)[0]));
    osg::Quat wheelRotarySpeedRR(0.0, cardyn::w_wrr(y)[0], 0.0, 0.0);
    wheelQuatRR = wheelQuatRR + wheelQuatRR * wheelRotarySpeedRR * (0.5 * dt);
    wheelQuatRR = wheelQuatRR * (1 / wheelQuatRR.length());
    wheelTransformRR->setAttitude(wheelQuatRR * camberRotRR);

    gealg::mv<3, 0x040201>::type p_P = part<3, 0x040201>(cardyn::P_b)(y);
    planeTransform->setPosition(osg::Vec3(p_P[0], p_P[1], p_P[2]));
    gealg::mv<3, 0x040201>::type n_P_l;
    n_P_l[2] = 1.0;
    gealg::mv<4, 0x06050300>::type q_P = ((grade<2>(cardyn::P_b) % (~cardyn::i) + n_P_l) * (grade<2>(cardyn::P_b) % (~cardyn::i)))(y);
    planeTransform->setAttitude(osg::Quat(q_P[3],
                                          -q_P[2],
                                          q_P[1],
                                          q_P[0]));

    /*gealg::mv<6, 0x060504030201>::type M_p = ((cardyn::p_b-grade<1>(cardyn::P_b))*cardyn::k_Pp-grade<1>(cardyn::dP_b)*cardyn::d_Pp + (grade<2>((!cardyn::q_b)*cardyn::P_xy*cardyn::q_b)-grade<2>(cardyn::P_b))*cardyn::k_Pq-grade<2>(cardyn::dP_b)*cardyn::d_Pq)(y);                                    
   std::cerr << "P_p: " << cardyn::P_b.e_(y) << ", dP_b: " << cardyn::dP_b.e_(y) << ", M_p: " << M_p << std::endl;*/
    //steerAngle += dt*(get<28>(y)[2]+get<29>(y)[2]);
    //std::cerr << "Wrench wheel fl: " << get<28>(y) << ", \tfr: " << get<29>(y) << std::endl;
    //std::cerr << "Moment: " << get<28>(y)[2]+get<29>(y)[2] << std::endl;
    osg::Matrix bodyMatrix;
    bodyMatrix.setTrans(bodyTransform->getPosition());
    //bodyMatrix.setRotate(bodyTransform->getAttitude());
    osg::Matrix cameraTransform;
    //cameraTransform.makeRotate(M_PI_2,osg::Vec3(0,0,-1));
    //cameraTransform *= osg::Matrix::translate(1.5,0.5,1.3);
    cameraTransform.makeTranslate(1.5, 0.5, 1.3);
    osg::Matrix invViewSpace = osg::Matrix::inverse(cameraTransform * bodyMatrix * cover->getObjectsScale()->getMatrix());
    osg::Matrix objSpace = invViewSpace * cover->getObjectsScale()->getMatrix();
    //cover->setXformMat(objSpace);

    /*gealg::mv<1, 0x07>::type d_P_wfl = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(part<2, 0x0201>(cardyn::r_wfl))*cardyn::q_b))^(grade<2>(cardyn::P_b)))-grade<3>(cardyn::P_b))(y);
   gealg::mv<1, 0x07>::type d_P_wfr = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(part<2, 0x0201>(cardyn::r_wfr))*cardyn::q_b))^(grade<2>(cardyn::P_b)))-grade<3>(cardyn::P_b))(y);
   gealg::mv<1, 0x07>::type d_P_wrl = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(part<2, 0x0201>(cardyn::r_wrl))*cardyn::q_b))^(grade<2>(cardyn::P_b)))-grade<3>(cardyn::P_b))(y);
   gealg::mv<1, 0x07>::type d_P_wrr = (((cardyn::p_b + grade<1>((!cardyn::q_b)*(part<2, 0x0201>(cardyn::r_wrr))*cardyn::q_b))^(grade<2>(cardyn::P_b)))-grade<3>(cardyn::P_b))(y);
   std::cout << "P_b: " << cardyn::P_b.e_(y) << ", d_P_wfl: " << d_P_wfl << ", d_P_wfr: " << d_P_wfr << ", d_P_wrl: " << d_P_wrl << ", d_P_wrr: " << d_P_wrr << std::endl;*/
    //gealg::mv<4, 0x07060503>::type P_c = ((cardyn::one+cardyn::p_b)^(grade<2>((!cardyn::q_b)*cardyn::P_xy*cardyn::q_b)))(y);
    //std::cout << "P_b: " << cardyn::P_b.e_(y) << ", dP_b: " << cardyn::dP_b.e_(y) << ", P_c: " << P_c << std::endl;

    /*cardyn::k_Pp.e_(y)[0] = (double)k_Pp_Slider->getValue();
   cardyn::d_Pp.e_(y)[0] = (double)d_Pp_Slider->getValue();
   cardyn::k_Pq.e_(y)[0] = (double)k_Pq_Slider->getValue();
   cardyn::d_Pq.e_(y)[0] = (double)d_Pq_Slider->getValue();*/
}

void RennteamPlugin::key(int type, int keySym, int /*mod*/)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        switch (keySym)
        {
        case 65361:
            steerAngle += 0.1;
            //get<36>(y)[0] = cos(steerAngle*0.5); get<36>(y)[1] = sin(steerAngle*0.5);
            //keyb->leftKeyDown();
            break;
        case 65363:
            //keyb->rightKeyDown();
            steerAngle -= 0.1;
            //get<36>(y)[0] = cos(steerAngle*0.5); get<36>(y)[1] = sin(steerAngle*0.5);
            break;
        case 65362:
            //keyb->foreKeyDown();
            cardyn::s_gp.e_(y)[0] = 1.0;
            break;
        case 65364:
            //keyb->backKeyDown();
            cardyn::F_b.e_(y)[0] = 90000.0;
            break;
            /*case 103:
            keyb->gearShiftUpKeyDown();
            break;
         case 102:
            keyb->gearShiftDownKeyDown();
            break;
         case 104:
            keyb->hornKeyDown();
            break;
         case 114:
            keyb->resetKeyDown();
            break;*/
        }
    }
    else if (type == osgGA::GUIEventAdapter::KEYUP)
    {
        switch (keySym)
        {
        case 65361:
            //keyb->leftKeyUp();
            break;
        case 65363:
            //keyb->rightKeyUp();
            break;
        case 65362:
            //keyb->foreKeyUp();
            cardyn::s_gp.e_(y)[0] = 0.0;
            break;
        case 65364:
            //keyb->backKeyUp();
            cardyn::F_b.e_(y)[0] = 0.0;
            break;
        case 103:
            gear += 1;
            if (gear > 5)
                gear = 5;
            cardyn::i_pt.e_(y)[0] = cardyn::i_g[gear + 1] * cardyn::i_a;
            if (gear == 0)
            {
                cardyn::k_c.e_(y)[0] = 0.0;
            }
            else
            {
                cardyn::k_c.e_(y)[0] = cardyn::k_cn;
            }
            break;
        case 102:
            gear -= 1;
            if (gear < -1)
                gear = -1;
            cardyn::i_pt.e_(y)[0] = cardyn::i_g[gear + 1] * cardyn::i_a;
            if (gear == 0)
            {
                cardyn::k_c.e_(y)[0] = 0.0;
            }
            else
            {
                cardyn::k_c.e_(y)[0] = cardyn::k_cn;
            }
            break;
            /*case 104:
            keyb->hornKeyUp();
            break;
         case 114:
            keyb->resetKeyUp();
            break;*/
        }
    }
}

gealg::mv<1, 0x04>::type RennteamPlugin::getGroundDistance(const gealg::mv<3, 0x040201>::type &p)
{
    gealg::mv<1, 0x04>::type e;

    if (p[0] >= 10.0 && p[0] < 15.0)
    {
        e[0] = p[2] - (p[0] - 10.0) * 0.2;
    }
    else if (p[0] >= 15.0 && p[0] < 20.0)
    {
        e[0] = p[2] - 1.0;
    }
    else if (p[0] >= 20.0 && p[0] < 25.0)
    {
        e[0] = p[2] - (1.0 - (p[0] - 20.0) * 0.2);
    }

    else if (p[0] < -20.0)
    {
        e[0] = p[2] - 0.2;
    }

    else
    {
        e[0] = p[2];
    }

    return e;
}

gealg::mv<6, 0x060504030201>::type RennteamPlugin::getContactPoint(const gealg::mv<3, 0x040201>::type &p_w, Road *&road, double &u)
{
    Vector3D v_w(p_w[0], p_w[1], p_w[2]);

    gealg::mv<6, 0x060504030201>::type s_c;
    Vector2D v_c(0.0, 0.0);
    if (roadSystem)
    {
        v_c = roadSystem->searchPosition(v_w, road, u);
    }
    else
    {
        std::cerr << "RennteamPlugin::getContactPoint(): no road system!" << std::endl;
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

COVERPLUGIN(RennteamPlugin)
