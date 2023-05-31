/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include "cover/VRSceneGraph.h"
#include <net/message.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>

#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <osg/Shape>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <vrb/client/VRBClient.h>
#include <grmsg/coGRSendCurrentDocMsg.h>
#include <math.h>
#include <cmath>
#include <iostream>

#include <osg/Vec3>
#include <osg/Geode>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgViewer/Viewer>
#include <osg/PositionAttitudeTransform>
#include <osgGA/TrackballManipulator>
#include <grmsg/coGRKeyWordMsg.h>
#include <OpenVRUI/coNavInteraction.h>
#include <vector>
#include <cover/coTranslator.h>
//

#include "RuledSurfaces.h"
//#include "Windows.h"
//#include "HfT_osg_Viewer.h"
//#include "HfT_osg_Config.h"
#include <osg/LineWidth>
#include <osgDB/ReadFile>
#include "HfT_osg_StateSet.h"
#include <osg/AnimationPath>
#include <osg/ShapeDrawable>

using namespace osg;
using namespace covise;
using namespace grmsg;

//---------------------------------------------------
//Implements ParametricSurfaces *plugin variable
//---------------------------------------------------
RuledSurfaces *RuledSurfaces::plugin = NULL;
//
RuledSurfaces::RuledSurfaces()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    root = new osg::Group();
    m_presentationStepCounter = 0; //current
    m_numPresentationSteps = 0; //gesamt
    m_presentationStep = 0; //current
    m_Mode = 1;
    m_ModeAnz = 5;
    m_permitDeleteGeode = true;
    m_permitSelectGeode = true;
    m_permitMoveGeode = true;
    m_sliderValueDetailstufe = 0;
    m_sliderValuePhi = 0;
    m_sliderValueErzeugung = 0;
    m_sliderValueRadAussen = 0;
    m_sliderValueHoeheAussen = 0;
    m_sliderValueSchraubhoehe = 0;
    m_pObjectMenu1 = NULL; //Pointer to the menu
    m_pObjectMenu2 = NULL; //Pointer to the menu
    m_pObjectMenu3 = NULL; //Pointer to the menu
    m_pObjectMenu4 = NULL; //Pointer to the menu
    m_pObjectMenu5 = NULL; //Pointer to the menu
    m_pObjectMenu6 = NULL; //Pointer to the menu
    m_pObjectMenu7 = NULL; //Pointer to the menu
    m_pObjectMenu8 = NULL; //Pointer to the menu
    m_pObjectMenu9 = NULL; //Pointer to the menu
    m_pObjectMenu10 = NULL; //Pointer to the menu
    m_pButtonMenuDreibein = NULL;
    m_pCheckboxMenuTangEbene = NULL;
    m_pSliderMenuDetailstufe = NULL;
    m_pSliderMenuPhi = NULL;
    m_pSliderMenuErzeugung = NULL;
    m_pSliderMenuRadAussen = NULL;
    m_pSliderMenuHoeheAussen = NULL;
    m_pSliderMenuSchraubhoehe = NULL;
    m_pCheckboxMenuGitternetz = NULL;
    m_pCheckboxMenuStriktion = NULL;
    m_pCheckboxMenuTrennung = NULL;
    m_pCheckboxMenuDrall = NULL;
    m_pCheckboxMenuGrad = NULL;
    m_pCheckboxMenuTangente = NULL;
}
RuledSurfaces::~RuledSurfaces()
{
    m_presentationStepCounter = 0; //current
    m_numPresentationSteps = 0; //gesamt
    m_presentationStep = 0; //current
    m_sliderValueDetailstufe = 0;
    m_sliderValuePhi = 0;
    m_sliderValueErzeugung = 0;
    m_sliderValueRadAussen = 0;
    m_sliderValueHoeheAussen = 0;
    m_sliderValueSchraubhoehe = 0;
    if (m_pObjectMenu1 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu1;
    }
    if (m_pObjectMenu2 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu2;
    }
    if (m_pObjectMenu3 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu3;
    }
    if (m_pObjectMenu4 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu4;
    }
    if (m_pObjectMenu5 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu5;
    }
    if (m_pObjectMenu6 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu6;
    }
    if (m_pObjectMenu7 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu7;
    }
    if (m_pObjectMenu8 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu8;
    }
    if (m_pObjectMenu9 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu9;
    }
    if (m_pObjectMenu10 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu10;
    }
    if (m_pButtonMenuDreibein != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuDreibein;
    }
    if (m_pCheckboxMenuTangEbene != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuTangEbene;
    }
    if (m_pSliderMenuDetailstufe != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuDetailstufe;
    }
    if (m_pSliderMenuPhi != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuPhi;
    }
    if (m_pSliderMenuErzeugung != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuErzeugung;
    }
    if (m_pSliderMenuRadAussen != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuRadAussen;
    }
    if (m_pSliderMenuHoeheAussen != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuHoeheAussen;
    }
    if (m_pSliderMenuSchraubhoehe != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuSchraubhoehe;
    }
    if (m_pCheckboxMenuGitternetz != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuGitternetz;
    }
    if (m_pCheckboxMenuStriktion != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuStriktion;
    }
    if (m_pCheckboxMenuTrennung != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuTrennung;
    }
    if (m_pCheckboxMenuDrall != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuDrall;
    }
    if (m_pCheckboxMenuGrad != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuGrad;
    }
    if (m_pCheckboxMenuTangente != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuTangente;
    }
    //Removes the root node from the scene graph
    cover->getObjectsRoot()->removeChild(root.get());
}
bool RuledSurfaces::init() //wird von OpenCover automatisch aufgerufen
{
    if (plugin)
        return false;
    //Set plugin
    RuledSurfaces::plugin = this;

    //Sets the possible number of presentation steps
    m_numPresentationSteps = 17;
    //Sets the surface´s formula

    TextureMode = false;
    Striktionslinie = false;
    Trennung = false;
    Gradlinie = false;
    Tangente = false;
    drall = false;
    stop = false; //stop Animation
    dreibeinAnim = false;
    tangEbene = false;

    black = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    white = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    grey = Vec4(0.8f, 0.8f, 0.8f, 1.0f);

    red = Vec4(1.0f, 0.0f, 0.0f, 1.0f);
    red2 = Vec4(1.0f, 0.0f, 0.0f, 0.0f);
    lime = Vec4(0.0f, 1.0f, 0.0f, 1.0f);
    blue = Vec4(0.0f, 0.0f, 1.0f, 1.0f);

    yellow = Vec4(1.0f, 1.0f, 0.0f, 1.0f);
    aqua = Vec4(0.0f, 1.0f, 1.0f, 1.0f);
    fuchsia = Vec4(1.0f, 0.0f, 1.0f, 1.0f);

    gold = Vec4(0.93f, 0.71f, 0.13f, 1.0f);
    //gold = Vec4(0.8f, 0.8f, 0.3f, 1.0f);
    silver = Vec4(0.25f, 0.25f, 0.25f, 1.0f); //0.33f, 0.33f, 0.33f

    purple = Vec4(0.8f, 0.0f, 0.8f, 1.0f);
    olive = Vec4(0.3f, 0.0f, 1.0f, 1.0f); //violett
    teal = Vec4(0.0f, 0.8f, 0.8f, 1.0f);

    navy = Vec4(0.2f, 0.6f, 0.8f, 1.0f); //hellblau
    maroon = Vec4(1.0f, 0.2f, 0.0f, 1.0f); //orange..
    green = Vec4(0.0f, 0.8f, 0.0f, 1.0f);

    pink = Vec4(1.0f, 0.8f, 0.8f, 1.0f);
    brown = Vec4(143.0f / 255.0f, 71.0f / 255.0f, 71.0f / 255.0f, 1.0f);

    colorArray = new Vec4Array;

    colorArray->push_back(aqua);
    colorArray->push_back(white);
    colorArray->push_back(lime);
    colorArray->push_back(red);
    colorArray->push_back(red2);
    colorArray->push_back(blue);
    colorArray->push_back(navy);
    colorArray->push_back(fuchsia);
    colorArray->push_back(purple);
    colorArray->push_back(brown);
    colorArray->push_back(pink);
    colorArray->push_back(yellow);
    colorArray->push_back(teal);
    colorArray->push_back(maroon);
    colorArray->push_back(green);
    colorArray->push_back(olive);
    colorArray->push_back(gold);
    colorArray->push_back(black);
    colorArray->push_back(grey);

    //dreibein
    ref_ptr<Geode> dreibein1 = new Geode;
    ref_ptr<Geode> dreibein2 = new Geode;
    ref_ptr<Geode> dreibein3 = new Geode;
    dreibeinAchse(dreibein1, green);
    dreibeinAchse(dreibein2, red);
    dreibeinAchse(dreibein3, blue);

    Matrixd matrixdreibein1;
    matrixdreibein1.makeRotate(-PI / 2, osg::Vec3(1, 0, 0));

    Matrixd matrixdreibein2;
    matrixdreibein2.makeRotate(PI / 2, osg::Vec3(0, 1, 0));

    osg::ref_ptr<osg::MatrixTransform> dreibeinNode1 = new osg::MatrixTransform(matrixdreibein1);
    dreibeinNode1->addChild(dreibein1);
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode2 = new osg::MatrixTransform(matrixdreibein2);
    dreibeinNode2->addChild(dreibein2);

    dreibein = new Group;
    dreibein->addChild(dreibeinNode1);
    dreibein->addChild(dreibeinNode2);
    dreibein->addChild(dreibein3);

    /* const std::string dreibeinPath=
	(std::string)coCoviseConfig::getEntry
	("value", "COVER.Plugin.RuledSurfaces.DataPath") + "Data/";

	dreibein = osgDB::readNodeFile(dreibeinPath + "Dreibein.osg");*/

    mp_GeoGroup = new Group;
    Matrix start, start1, start2;

    start1.makeRotate(PI / 3, osg::Vec3f(1, 0, -1));
    start2.makeRotate(PI / 8, osg::Vec3f(0, 1, 0));
    start = start2 * start1;
    osg::ref_ptr<osg::MatrixTransform> StartMatrix = new MatrixTransform(start);
    StartMatrix->addChild(mp_GeoGroup);
    root->addChild(StartMatrix);

    //Torsen
    ebenegeo = new Geode;
    ebenegeo2 = new Geode;
    kegelgeo = new Geode;
    okgeo = new Geode;
    zylindergeo = new Geode;
    oloidgeo = new Geode;
    ozgeo = new Geode;
    tangentialflaechegeo = new Geode;

    //wsRF
    hypergeo = new Geode;
    sattelgeo = new Geode;
    konoidgeo = new Geode;
    helixgeo = new Geode;
    helix2geo = new Geode;

    //Striktionslinien
    shypergeo = new Geode;
    ssattelgeo = new Geode;
    skonoidgeo = new Geode;
    shelixgeo = new Geode;
    gradliniegeo = new Geode;
    tangentegeo = new Geode;

    Ebene();

    createMenu();

    changePresentationStep();

    interactionA = new coNavInteraction(coInteraction::ButtonA, "Selection", coInteraction::NavigationHigh);

    cover->getObjectsRoot()->addChild(root.get());

    //zoom scene
    //attention: never use if the scene is empty!!
    VRSceneGraph::instance()->viewAll();

    std::cerr << "Plugin laedt am Ende" << std::endl;
    return true;
}

ref_ptr<Material> m_MaterialLine = new Material();
ref_ptr<LineWidth> m_Line = new LineWidth(1.5f);
ref_ptr<LineWidth> n_Line = new LineWidth(5);

//Detailstuffe
int stepm = 50;
//mKegel="50";
//Drehung
double stepd = 1.5f;
//Erzeugung
float stepe = 2;
//Helixradie
float stepr1 = 1;
float stepr2 = 2;
float stepr3 = -2;
//Helixhöhen
float steph1 = 0;
float steph2 = 0;
float steph3 = 1;
//
float steph = 1;

void RuledSurfaces::dreibeinAchse(Geode *geod, Vec4 color)
{
    float Radius = 10.0f; /*2.0f;*/

    osg::ref_ptr<osg::Material> material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0] * 0.3f, color[1] * 0.3f, color[2] * 0.3f, color[3]));
    osg::ref_ptr<osg::Material> material_sphere = new osg::Material();
    color = silver;
    material_sphere->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material_sphere->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0] * 0.3f, color[1] * 0.3f, color[2] * 0.3f, color[3]));
    osg::StateSet *stateSet;

    osg::ref_ptr<osg::Sphere> sphereG = new osg::Sphere(Vec3(0.0f, 0.0f, 0.0f), 0.005 * Radius);
    osg::ref_ptr<osg::ShapeDrawable> sphereD = new osg::ShapeDrawable(sphereG.get());
    stateSet = sphereD->getOrCreateStateSet();
    stateSet->setAttribute /*AndModes*/ (material_sphere.get(), osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    sphereD->setStateSet(stateSet);
    geod->addDrawable(sphereD.get());

    osg::ref_ptr<osg::Cylinder> cylinderG = new osg::Cylinder(Vec3(0.0f, 0.0f, 0.05 * Radius), 0.002 * Radius, 0.1 * Radius);
    osg::ref_ptr<osg::ShapeDrawable> cylinderD = new osg::ShapeDrawable(cylinderG.get());
    stateSet = cylinderD->getOrCreateStateSet();
    stateSet->setAttribute /*AndModes*/ (material.get(), osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    cylinderD->setStateSet(stateSet);
    geod->addDrawable(cylinderD.get());

    osg::ref_ptr<osg::Cone> coneG = new osg::Cone(Vec3(0.0f, 0.0f, 0.1 * Radius), 0.008 * Radius, 0.02 * Radius);
    osg::ref_ptr<osg::ShapeDrawable> coneD = new osg::ShapeDrawable(coneG.get());
    stateSet = coneD->getOrCreateStateSet();
    stateSet->setAttribute /*AndModes*/ (material.get(), osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    coneD->setStateSet(stateSet);
    geod->addDrawable(coneD.get());
}
void RuledSurfaces::setMaterial_surface(Geometry *geom, Vec4 color)
{
    /*ref_ptr <Vec4Array> patchColorList = new Vec4Array(1);
	patchColorList.get()->at(0) = Vec4(0.0f, 0.6f, 0.6f, 0.8f);
	geom->setColorArray(patchColorList);
	geom->setColorBinding(osg::Geometry::BIND_OVERALL);*/
    if (drall == true && surface != "Konoid")
    {
        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SSHADE,color);
		geom->setStateSet(mp_StateSet);*/

        ref_ptr<StateSet> geo_state = geom->getOrCreateStateSet();
        ref_ptr<Material> mtl = new Material;
        mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        mtl->setAmbient(Material::FRONT_AND_BACK, color); //aqua
        mtl->setDiffuse(Material::FRONT_AND_BACK, color * 0.7);
        mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setEmission(Material::FRONT_AND_BACK, color);
        //mtl->setShininess(Material::FRONT_AND_BACK,0.1f);
        geo_state->setAttribute(mtl.get(), osg::StateAttribute::PROTECTED);
        geo_state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    }
    else
    {
        ref_ptr<StateSet> geo_state = geom->getOrCreateStateSet();
        ref_ptr<Material> mtl = new Material;
        mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.1, 0.18725, 0.1745, 1.0)); //tuerikis veraendert
        mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.296, 0.74151, 0.69102, 1.0));
        mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.297254, 0.30829, 0.306678, 1.0));
        //mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        mtl->setShininess(Material::FRONT_AND_BACK, 12.8f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.25,     0.22, 0.06,  1.0));//gold poliert-dunkel :(
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.35,   0.31, 0.09, 1.0));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.80,0.72, 0.21,1.0));
        ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setShininess(Material::FRONT_AND_BACK,83.2f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.1,     0.18725, 0.1745,  1.0));//tuerikis :)
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.396,   0.74151, 0.69102, 1.0));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.297254,0.30829, 0.306678,1.0));
        ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setShininess(Material::FRONT_AND_BACK,12.8f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.0,   0.1,    0.06,    1.0));//plastic cyan-zu dunkel
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.0,   0.50980392,0.50980392,1.0));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.50196078,0.50196078,0.50196078,1.0));
        ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setShininess(Material::FRONT_AND_BACK,32.f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.24725,  0.1995,   0.0745,    1.0));//gold
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.75164,  0.60648,  0.22648,   1.0));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.628281, 0.555802, 0.366065,  1.0));
        ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setShininess(Material::FRONT_AND_BACK,51.2f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.329412,  0.223529, 0.027451, 1.0));//Messing/brass-orange
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.780392,  0.568627, 0.113725, 1.0));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.992157,  0.941176, 0.807843, 1.0));
        ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setShininess(Material::FRONT_AND_BACK,27.89743616f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.135, 0.2225, 0.1575, 0.95));//Jade
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.54, 0.89, 0.63, 0.95));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.316228, 0.316228, 0.316228, 0.95));
        //mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0,0.0,0.0,0.0));
        //mtl->setShininess(Material::FRONT_AND_BACK,12.8f);

        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.0215, 0.1745, 0.0215, 0.55));//Emerald
        //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.07568, 0.61424, 0.07568, 0.55));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.633, 0.727811, 0.633, 0.55));
        //mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0,0.0,0.0,0.0));
        //mtl->setShininess(Material::FRONT_AND_BACK,76.8f);

        ////aqua = Vec4(0.0f, 1.0f, 1.0f, 1.0f);
        //ref_ptr <StateSet> geo_state = geom->getOrCreateStateSet();
        //ref_ptr <Material> mtl = new Material;
        //mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.0f, 1.0f, 1.0f, 1.0f));//aqua
        //mtl->setDiffuse( Material::FRONT_AND_BACK, 0.7*Vec4(0.0f, 1.0f, 1.0f, 1.0f));
        //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f,1.f));
        ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
        //mtl->setShininess(Material::FRONT_AND_BACK,0.1f);

        geo_state->setAttribute(mtl.get(), osg::StateAttribute::PROTECTED);
        geo_state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    }
}
void RuledSurfaces::setMaterial_line(Geometry *geom2, Vec4 color)
{
    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(color);
    geom2->setColorArray(colors.get());
    geom2->setColorBinding(Geometry::BIND_OVERALL);
    osg::StateSet *stateset = geom2->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(/*1.5*/ 1.5f);
    stateset->setAttribute(lw);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED);
    ref_ptr<PolygonMode> m_Quad = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    stateset->setAttributeAndModes(m_Quad, StateAttribute::PROTECTED);
}
void RuledSurfaces::setMaterial_striktline(Geometry *geom2, Vec4 color)
{
    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(color);
    geom2->setColorArray(colors.get());
    geom2->setColorBinding(Geometry::BIND_OVERALL);
    osg::StateSet *stateset = geom2->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(3.5f);
    stateset->setAttribute(lw);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED);
    ref_ptr<PolygonMode> m_Line = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    stateset->setAttributeAndModes(m_Line, StateAttribute::PROTECTED);
}
//Animation
void RuledSurfaces::DreibeinKegel()
{
    osg::ref_ptr<osg::Vec3Array> pathpoints = new Vec3Array();
    if (!pathpoints3BKegel)
        return;
    pathpoints = pathpoints3BKegel;
    Matrixd matrix3B, matrix3B2, matrix3B3;

    matrix3B.makeRotate(-PI, osg::Vec3(0, 0, 1));
    matrix3B2.makeRotate(PI / 4, osg::Vec3(0, 1, 0));
    matrix3B3 = matrix3B2 * matrix3B;

    osg::ref_ptr<osg::MatrixTransform> dreibeinMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode = new osg::MatrixTransform(matrix3B3);
    dreibeinNode->addChild(dreibein);

    dreibeinMatrix->addChild(dreibeinNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::LOOP);

    //kreisen
    int k = pathpoints->size();
    for (int i = 0; i < k; i++)
    {
        osg::Quat rotation((stepe)*PI * (float)i / (pathpoints->size() - 1), osg::Vec3(0, 0, 1));
        path->insert(0.25 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    dreibeinMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(dreibeinMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(dreibeinMatrix);
}
void RuledSurfaces::DreibeinZylinder()
{
    osg::ref_ptr<osg::Vec3Array> pathpoints = new Vec3Array();
    if (!pathpoints3BZylinder)
        return;
    pathpoints = pathpoints3BZylinder;
    Matrixd matrix3B;

    matrix3B.makeRotate(-PI, osg::Vec3(0, 0, 1));

    osg::ref_ptr<osg::MatrixTransform> dreibeinMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode = new osg::MatrixTransform(matrix3B);
    dreibeinNode->addChild(dreibein);

    dreibeinMatrix->addChild(dreibeinNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::LOOP);

    //kreisen
    int k = pathpoints->size();
    for (int i = 0; i < k; i++)
    {
        osg::Quat rotation((stepe)*PI * (float)i / (pathpoints->size() - 1), osg::Vec3(0, 0, 1));
        path->insert(0.25 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    dreibeinMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(dreibeinMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(dreibeinMatrix);
}
void RuledSurfaces::DreibeinHyper()
{
    osg::ref_ptr<osg::Vec3Array> pathpoints = new Vec3Array();
    if (!pathpoints3BHyper)
        return;
    pathpoints = pathpoints3BHyper;
    Matrixd matrix3B, matrix3B2, matrix3B3;

    //Drehachse
    float drehx = -sin(stepd * PI);
    float drehy = (cos(stepd * PI) - 1);

    //Drehwinkel
    float r = sqrt((cos(stepd * PI) - 1) * (cos(stepd * PI) - 1) + sin(stepd * PI) * sin(stepd * PI) + 1);
    float alpha = acos(1 / r);

    matrix3B.makeRotate(alpha, osg::Vec3(drehx, drehy, 0));
    Vec3f xn = Vec3f(1, 0, 0) * matrix3B;
    Vec3f yn = Vec3f(0, 1, 0) * matrix3B;
    Vec3f zn = Vec3f(0, 0, 1) * matrix3B;
    Vec3f zna = Vec3f(0, 0, 1) * matrix3B;

    float xz = -sin(stepd * PI);
    float yz = (cos(stepd * PI) - 1);

    float norm1 = sqrt(xn.x() * xn.x() + xn.y() * xn.y() + xn.z() * xn.z());
    float norm2 = sqrt(xz * xz + yz * yz);
    float scalar3 = xn.x() * xz + xn.y() * yz;

    float drehz = scalar3 / (norm1 * norm2);
    float beta = acos(drehz);
    if (stepd > 1)
        matrix3B2.makeRotate(beta, osg::Vec3(-zna));
    else
        matrix3B2.makeRotate(beta, osg::Vec3(-zna));
    matrix3B3 = matrix3B * matrix3B2;
    Vec3f ynn = yn * matrix3B2;
    Vec3f ynn2 = Vec3f(0, 1, 0) * matrix3B3;
    osg::ref_ptr<osg::MatrixTransform> dreibeinMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode = new osg::MatrixTransform(matrix3B3);
    dreibeinNode->addChild(dreibein);

    dreibeinMatrix->addChild(dreibeinNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::LOOP);

    //kreisen
    int k = pathpoints->size();
    for (int i = 0; i < k; i++)
    {
        osg::Quat rotation((stepe)*PI * (float)i / (pathpoints->size() - 1), osg::Vec3(0, 0, 1));
        path->insert(0.25 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    dreibeinMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(dreibeinMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(dreibeinMatrix);
}
void RuledSurfaces::DreibeinSattel()
{
    osg::ref_ptr<osg::Vec3Array> pathpoints = new Vec3Array();
    if (!pathpoints3BSattel)
        return;
    pathpoints = pathpoints3BSattel;
    pathpoints2 = pathpoints3BSattel2;
    Matrixd matrix3B, matrix3B2, matrix3B3, rotation;

    //Drehwinkel
    //float r = sqrt((cos(stepd*PI)-1)*(cos(stepd*PI)-1)+sin(stepd*PI)*sin(stepd*PI)+1);
    //float alpha = acos(2/(sqrt(6.)));
    //float beta = acos(3/(sqrt(30.)));
    //float beta = alpha * 2;

    /*matrix3B3.makeRotate(alpha,osg::Vec3(1,1,0));
	matrix3B2.makeRotate(PI/2,osg::Vec3(0,0,-1));
	matrix3B=matrix3B2*matrix3B3;*/
    matrix3B.makeRotate(0.f, osg::Vec3(1, 1, 0));
    osg::ref_ptr<osg::MatrixTransform> dreibeinMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode = new osg::MatrixTransform(matrix3B);
    dreibeinNode->addChild(dreibein);
    dreibeinMatrix->addChild(dreibeinNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //kreisen
    //pathpoints->push_back(pathpoints->at(pathpoints->size()-1));
    for (unsigned int i = 0; i < pathpoints->size(); i++)
    {
        /*float xi = pathpoints2->at(i).x()-pathpoints->at(i).x();
		float yi = pathpoints2->at(i).y()-pathpoints->at(i).y();
		float zi = pathpoints2->at(i).z()-pathpoints->at(i).z();

		float xi2 = pathpoints2->at(i+1).x()-pathpoints->at(i+1).x();
		float yi2 = pathpoints2->at(i+1).y()-pathpoints->at(i+1).y();
		float zi2 = pathpoints2->at(i+1).z()-pathpoints->at(i+1).z();

		float scalari = sqrt(xi*xi+yi*yi+zi*zi);
		float scalari2 = sqrt(xi2*xi2+yi2*yi2+zi2*zi2);
		float spaat = xi*xi2+yi*yi2+zi*zi2;

		float beta = acos(spaat/(scalari*scalari2));

		float drehx = yi*zi2-zi*yi2;
		float drehy = xi*zi2-zi*xi2;
		float drehz = xi*yi2-yi*xi2;*/
        Vec3f punkt_ankEnd = pathpoints2->at(i);
        punkt_ankEnd.y() = 0.;

        Vec3f vec_ank = punkt_ankEnd - pathpoints->at(i);
        Vec3f vec_wink = pathpoints2->at(i) - pathpoints->at(i);

        float skalar = vec_ank.x() * vec_wink.x() + vec_ank.y() * vec_wink.y() + vec_ank.z() * vec_wink.z();
        float norm_vec_ank = vec_ank.x() * vec_ank.x() + vec_ank.y() * vec_ank.y() + vec_ank.z() * vec_ank.z();
        float norm_vec_wink = vec_wink.x() * vec_wink.x() + vec_wink.y() * vec_wink.y() + vec_wink.z() * vec_wink.z();
        float winkel = acosf(skalar / (norm_vec_ank * norm_vec_wink));

        osg::Quat rotation(winkel /*beta*/, osg::Vec3(1, 0, 0 /*drehx,drehy,drehz*/));
        path->insert(0.1 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    dreibeinMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(dreibeinMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(dreibeinMatrix);
}
void RuledSurfaces::DreibeinKonoid()
{
    osg::ref_ptr<osg::Vec3Array> pathpoints = new Vec3Array();
    osg::ref_ptr<osg::Vec2Array> angle = new Vec2Array();
    if (!pathpoints3BKonoid)
        return;
    pathpoints = pathpoints3BKonoid;
    angle = angle3BKonoid;
    Matrixd matrix3B, matrix3B2;
    osg::Quat rotation(0, osg::Vec3(0, 0, 0), 0, osg::Vec3(0, -1, 0), PI / 2, osg::Vec3(0, 0, -1));
    matrix3B.setRotate(rotation);
    //matrix3B.makeRotate(PI/4,osg::Vec3(0,-1,0));
    osg::ref_ptr<osg::MatrixTransform> dreibeinMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode = new osg::MatrixTransform(matrix3B);
    dreibeinNode->addChild(dreibein);

    dreibeinMatrix->addChild(dreibeinNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    int k = pathpoints->size();
    for (int i = 0; i < k; i++)
    {
        osg::Quat rotation(angle->at(i).x() /*sin(PI*((float)i/(pathpoints->size()-1)))*atan(0.5)*/, osg::Vec3(-1, 0, 0));
        path->insert(0.1 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    dreibeinMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(dreibeinMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(dreibeinMatrix);
}
void RuledSurfaces::DreibeinHelix()
{
    osg::ref_ptr<osg::Vec3Array> pathpoints = new Vec3Array();
    if (!pathpoints3BHelix)
        return;
    pathpoints = pathpoints3BHelix;
    pathpoints2 = pathpoints_helix2;
    Matrixd matrix3B, matrix3B2, matrix3B3, matrix3B4, matrix3B5;

    matrix3B3.makeRotate(PI / 2, osg::Vec3(0, 1, 0));
    matrix3B2.makeRotate(PI / 2, osg::Vec3(0, 0, 1));
    matrix3B4 = matrix3B2 * matrix3B3;

    float normx = sqrt(pathpoints2->at(0).x() * pathpoints2->at(0).x() + pathpoints2->at(0).y() * pathpoints2->at(0).y() + pathpoints2->at(0).z() * pathpoints2->at(0).z());
    float gamma = acos(pathpoints2->at(0).x() / (normx));

    matrix3B5.makeRotate(-gamma, osg::Vec3(0, pathpoints2->at(0).z(), pathpoints2->at(0).y()));
    matrix3B = matrix3B4 * matrix3B5;

    osg::ref_ptr<osg::MatrixTransform> dreibeinMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> dreibeinNode = new osg::MatrixTransform(matrix3B);
    dreibeinNode->addChild(dreibein);

    dreibeinMatrix->addChild(dreibeinNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //kreisen
    int k = pathpoints->size();
    for (int i = 0; i < k; i++)
    {
        osg::Quat rotation(stepe * PI * (float)i / (pathpoints->size() - 1), osg::Vec3(0, 0, 1));
        path->insert(0.25 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    dreibeinMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(dreibeinMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(dreibeinMatrix);
}
void RuledSurfaces::Tangentialebene_Kegel()
{
    if (!pathpoints_kegel)
        return;
    Matrixd matrixk, matrixk2, matrixk3;
    pathpoints = pathpoints_kegel;
    osg::Quat rotation(0, osg::Vec3(-1, 0, 0), 0, osg::Vec3(0, 1, 0), 0, osg::Vec3(0, 0, 1));
    matrixk2.makeRotate(PI / 2, osg::Vec3(0, 0, -1));
    matrixk3.makeRotate(PI / 4, osg::Vec3(0, -1, 0));
    matrixk = matrixk2 * matrixk3;

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixk);
    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(0 * (float)i, osg::Vec3(0, 0, -1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}
void RuledSurfaces::Tangentialebene_OK()
{
    if (!pathpoints_ok)
        return;
    Matrixd matrixk, matrixk2, matrixk3, matrixk4;
    pathpoints = pathpoints_ok;

    matrixk4.makeRotate(-PI / 4, osg::Vec3(1, 0, 0));
    matrixk2.makeRotate(-2.53, osg::Vec3(0, 0, 1));
    matrixk = matrixk2 * matrixk4;

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixk);
    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(0 * (float)i, osg::Vec3(0, 0, -1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i) + osg::Vec3(0, -0.01, 0.01), rotation));
    }
    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}
void RuledSurfaces::Tangentialebene_Zylinder()
{
    if (!pathpoints_zylinder)
        return;
    Matrixd matrixk;
    pathpoints = pathpoints_zylinder;
    osg::Quat rotation((PI / 2), osg::Vec3(0, 0, 1));
    matrixk.setRotate(rotation);

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixk);
    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(0 * (float)i, osg::Vec3(0, 0, -1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}
void RuledSurfaces::Tangentialebene_OZ()
{
    if (!pathpoints_oz)
        return;
    Matrixd matrixk;
    pathpoints = pathpoints_oz;
    /*osg::Quat rotation((PI/2),osg::Vec3(0,0,1));
	matrixk.setRotate(rotation);*/

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixk);
    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(0 * (float)i, osg::Vec3(0, 0, -1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}

void RuledSurfaces::Tangentialebene_Hyper()
{
    if (!pathpoints_hyper)
        return;
    Matrixd matrixk, matrixk2, matrixk3, matrix3B, matrix3B2, matrix3B3;
    pathpoints = pathpoints_hyper;

    //Drehwinkel
    float l = 1 - cos(stepd * PI);
    float alpha_x = atan(l);
    float l2 = sin(stepd * PI);
    float l3 = sqrtf(1 + l * l);
    float alpha_y = atan(l2 / l3);

    matrixk.makeRotate(PI / 2, osg::Vec3(0, 0, 1));
    matrix3B.makeRotate(-alpha_x, osg::Vec3(1, 0, 0));
    matrix3B2.makeRotate(alpha_y, osg::Vec3(0, 1, 0));
    matrix3B3 = matrix3B2 * matrix3B * matrixk;
    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrix3B3);

    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    ref_ptr<osg::DrawElementsUInt> mp_TriangleEdges_Geom = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 6);
    int k = 0;
    float delta = 0.01;
    vec3->push_back(osg::Vec3(pathpoints->at(0)));
    vec3->push_back(osg::Vec3(cos(stepd * PI - delta), sin(stepd * PI - delta), 1));
    vec3->push_back(osg::Vec3(cos(stepd * PI + delta), sin(stepd * PI + delta), 1));
    (*mp_TriangleEdges_Geom)[k] = k;
    k++;
    (*mp_TriangleEdges_Geom)[k] = k;
    k++;
    (*mp_TriangleEdges_Geom)[k] = k;
    k++;
    vec3->push_back(osg::Vec3(pathpoints->at(1)));
    vec3->push_back(osg::Vec3(cos(delta), sin(delta), 0));
    vec3->push_back(osg::Vec3(cos(-delta), sin(-delta), 0));

    (*mp_TriangleEdges_Geom)[k] = k;
    k++;
    (*mp_TriangleEdges_Geom)[k] = k;
    k++;
    (*mp_TriangleEdges_Geom)[k] = k;
    k++;

    fillNormalArrayTriangles(vec3, NormalArray, mp_TriangleEdges_Geom, 2);

    Vec3f normale1 = NormalArray->at(0);
    Vec3f normale2 = NormalArray->at(3);

    Vec3f vec_unten = normale1;
    Vec3f vec_oben = normale2;
    float skalar = vec_unten.x() * vec_oben.x() + vec_unten.y() * vec_oben.y() + vec_unten.z() * vec_oben.z();
    float norm_vec_unten = sqrtf(vec_unten.x() * vec_unten.x() + vec_unten.y() * vec_unten.y() + vec_unten.z() * vec_unten.z());
    float norm_vec_oben = sqrtf(vec_oben.x() * vec_oben.x() + vec_oben.y() * vec_oben.y() + vec_oben.z() * vec_oben.z());
    float winkel = acos(skalar / (norm_vec_oben * norm_vec_unten));
    if (stepd >= 2.991 && stepd <= 3.001)
    {
        winkel = 0;
    }
    else if (stepd < 3)
    {
        winkel = 2 * PI - winkel;
    }
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation((-winkel) * (float)i, osg::Vec3(cos(stepd * PI) - 1, sin(stepd * PI), 1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }
    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}
void RuledSurfaces::Tangentialebene_Sattel()
{
    if (!pathpoints_sattel)
        return;
    Matrixd matrixk;
    pathpoints = pathpoints_sattel;
    /*osg::Quat rotation((PI/2),osg::Vec3(1,0,0));
	matrixk.setRotate(rotation);*/

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixk);
    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(PI / 2 * (float)i, osg::Vec3(0, 0, 1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}
void RuledSurfaces::Tangentialebene1()
{
    if (!pathpointst1)
        return;
    Matrixd matrixt1;

    /*osg::Quat rotation((PI/2),osg::Vec3(0,0,1));
	matrixt1.setRotate(rotation);*/

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixt1);
    tangentialNode->addChild(ebenegeo);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //Animation
    pathpointst1->push_back(pathpointst1->at(pathpointst1->size() - 1));
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation((PI / 2) * (float)i, osg::Vec3(0, 0, -1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpointst1->at(i) /*+osg::Vec3(0.03,0.05,0)*/, rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}
void RuledSurfaces::Tangentialebene2()
{
    if (!pathpointst2)
        return;
    Matrixd matrixt2;

    /*osg::Quat rotation((PI/180)*22,osg::Vec3(-1,0,0));
	matrixt2.setRotate(rotation);*/

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix2 = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode2 = new osg::MatrixTransform(matrixt2);
    tangentialNode2->addChild(ebenegeo);

    tangentialMatrix2->addChild(tangentialNode2);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //Animation
    pathpointst2->push_back(pathpointst2->at(pathpointst2->size() - 1));
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(-(PI / 6) * (float)i, osg::Vec3(0, 0, 1), atan(sqrt(1 - 0.25) / 2), osg::Vec3(-1, 0, 0), 0, osg::Vec3(0, 0, 1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpointst2->at(i) + osg::Vec3(0, 0.01, 0), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix2->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix2) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix2);
}

void RuledSurfaces::Tangentialebene3()
{
    if (!pathpointst3)
        return;
    Matrixd matrixt3;

    osg::Quat rotation((PI / 180) * 26.5, osg::Vec3(-1, 0, 0));
    matrixt3.setRotate(rotation);

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix3 = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode3 = new osg::MatrixTransform(matrixt3);
    tangentialNode3->addChild(ebenegeo);

    tangentialMatrix3->addChild(tangentialNode3);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //Animation
    pathpointst3->push_back(pathpointst3->at(pathpointst3->size() - 1));
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(0 * (float)i, osg::Vec3(0, 0, 1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpointst3->at(i) + osg::Vec3(0, 0.01, 0), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix3->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix3) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix3);
}

void RuledSurfaces::Tangentialebene4()
{
    if (!pathpointst4)
        return;
    Matrixd matrixt4;

    /*osg::Quat rotation((PI/180)*22,osg::Vec3(-1,0,0));
	matrixt4.setRotate(rotation);*/

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix4 = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode4 = new osg::MatrixTransform(matrixt4);
    tangentialNode4->addChild(ebenegeo);

    tangentialMatrix4->addChild(tangentialNode4);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //Animation
    pathpointst4->push_back(pathpointst4->at(pathpointst4->size() - 1));
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation((PI / 6) * (float)i, osg::Vec3(0, 0, 1), atan(sqrt(1 - 0.25) / 2), osg::Vec3(-1, 0, 0), 0, osg::Vec3(0, 0, 1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpointst4->at(i) + osg::Vec3(0, 0.01, 0), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix4->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix4) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix4);
}

void RuledSurfaces::Tangentialebene5()
{
    if (!pathpointst5)
        return;
    Matrixd matrixt5;

    /*osg::Quat rotation((PI/2),osg::Vec3(0,0,-1));
	matrixt5.setRotate(rotation);*/

    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix5 = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode5 = new osg::MatrixTransform(matrixt5);
    tangentialNode5->addChild(ebenegeo);

    tangentialMatrix5->addChild(tangentialNode5);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    //Animation
    pathpointst5->push_back(pathpointst5->at(pathpointst5->size() - 1));
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation((PI / 2) * (float)i, osg::Vec3(0, 0, 1));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpointst5->at(i), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix5->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix5) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix5);
}
void RuledSurfaces::Tangentialebene_Helix()
{
    if (!pathpoints_helix)
        return;
    Matrixd matrixk, matrixk2, matrixk3;
    pathpoints = pathpoints_helix;
    pathpoints2 = pathpoints_helix2;

    float normx = sqrt(pathpoints2->at(0).x() * pathpoints2->at(0).x() + pathpoints2->at(0).y() * pathpoints2->at(0).y() + pathpoints2->at(0).z() * pathpoints2->at(0).z());
    float gamma = acos(pathpoints2->at(0).x() / (normx));

    matrixk2.makeRotate(-gamma, osg::Vec3(0, 1, 0));
    osg::ref_ptr<osg::MatrixTransform> tangentialMatrix = new MatrixTransform();
    osg::ref_ptr<osg::MatrixTransform> tangentialNode = new osg::MatrixTransform(matrixk2);
    tangentialNode->addChild(ebenegeo2);

    float dYz = pathpoints2->at(1).z() - pathpoints2->at(0).z();
    float dYy = pathpoints2->at(1).y() - pathpoints2->at(0).y();
    float beta = PI / 2 - atan(dYz / dYy);

    tangentialMatrix->addChild(tangentialNode);

    osg::ref_ptr<osg::AnimationPathCallback> apcb1 = new osg::AnimationPathCallback;
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::SWING);

    if (steph3 <= 0)
    {
        beta = -(PI - beta);
    }
    for (unsigned int i = 0; i < 2; i++)
    {
        osg::Quat rotation(-beta * (float)i, osg::Vec3(pathpoints->at(1).x(), pathpoints->at(1).y(), pathpoints->at(1).z()));
        path->insert(5 * (float)i, osg::AnimationPath::ControlPoint(pathpoints->at(i) + osg::Vec3(0, -0.01, 0), rotation));
    }

    apcb1->setAnimationPath(path);
    tangentialMatrix->setUpdateCallback(apcb1.get());

    int index = root->getChildIndex(tangentialMatrix) - 1;
    mp_GeoGroup->removeChildren(index, index);
    mp_GeoGroup->addChild(tangentialMatrix);
}

//Flaechen
//Normalen

void RuledSurfaces::createTextureCoordinates(int Numfaces, osg::ref_ptr<osg::Geometry> geo)
{
    osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array;
    unsigned int n = Numfaces;
    double delta = 1.0f / n;
    for (unsigned int i = 0; i < n; i++)
    {
        // Gleich TexturKoordinaten je Dreieck mitgeben
        texcoords->push_back(osg::Vec2(0.f, i * delta));
        texcoords->push_back(osg::Vec2(i * delta, i * delta));
        texcoords->push_back(osg::Vec2(i * delta, 1 - i * delta));
        /*
		texcoords->push_back( osg::Vec2(0.0f,i*delta) );
		texcoords->push_back( osg::Vec2(0.5f,i*delta) );
		texcoords->push_back( osg::Vec2(1.0f,i*delta) );
		*/
    }
    geo->setTexCoordArray(0, texcoords.get());
}

void RuledSurfaces::fillNormalArrayQuads(Vec3Array *Points, Vec3Array *Normals, DrawElementsUInt *IndexList, int numFaces)
{
    int k = 0;
    int index1 = 0, index2 = 0, index3 = 0, index4 = 0;
    Vec3f P1, P2, P3, P4, D1, D2, N1, N2;
    double x, y, z, bt;
    std::list<osg::Vec3f *>::iterator it;
    unsigned int n = numFaces;
    k = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        index1 = IndexList->at(k);
        P1 = Points->at(index1);
        k++;
        index2 = IndexList->at(k);
        P2 = Points->at(index2);
        k++;
        index3 = IndexList->at(k);
        P3 = Points->at(index3);
        k++;
        index4 = IndexList->at(k);
        P4 = Points->at(index4);
        k++;

        D1 = P3 - P1;
        D2 = P2 - P4;

        x = D1.y() * D2.z() - D1.z() * D2.y();
        y = -D1.x() * D2.z() + D1.z() * D2.x();
        z = D1.x() * D2.y() - D1.y() * D2.x();
        bt = sqrt(x * x + y * y + z * z);
        if (bt > 0.000001)
        {
            x = -x / bt;
            y = -y / bt;
            z = -z / bt;
        }
        N1.set(x, y, z);
        (*Normals).push_back(N1);
        (*Normals).push_back(N1);
        (*Normals).push_back(N1);
        (*Normals).push_back(N1);
    }
}
void RuledSurfaces::fillNormalArrayTangentialflaeche(Vec3Array *Points, Vec3Array *Normals, DrawElementsUInt *IndexList, int numFaces)
{
    int k = 0;
    int index1 = 0, index2 = 0, index3 = 0, index4 = 0;
    Vec3f P1, P2, P3, P4, D1, D2, N1, N2;
    double x, y, z, bt;
    std::list<osg::Vec3f *>::iterator it;
    unsigned int n = numFaces;
    k = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        index1 = IndexList->at(k);
        P1 = Points->at(index1);
        k++;
        index2 = IndexList->at(k);
        P2 = Points->at(index2);
        k++;
        index3 = IndexList->at(k);
        P3 = Points->at(index3);
        k++;
        index4 = IndexList->at(k);
        P4 = Points->at(index4);
        k++;

        D1 = P4 - P1;
        D2 = P2 - P3;

        x = D1.y() * D2.z() - D1.z() * D2.y();
        y = -D1.x() * D2.z() + D1.z() * D2.x();
        z = D1.x() * D2.y() - D1.y() * D2.x();
        bt = sqrt(x * x + y * y + z * z);
        if (bt > 0.000001)
        {
            x = -x / bt;
            y = -y / bt;
            z = -z / bt;
        }
        N1.set(x, y, z);
        (*Normals).push_back(N1);
        (*Normals).push_back(N1);
        (*Normals).push_back(N1);
        (*Normals).push_back(N1);
    }
}

void RuledSurfaces::fillNormalArrayTriangles(Vec3Array *Points, Vec3Array *Normals, DrawElementsUInt *IndexList, int numFaces)
{
    int k = 0;
    int index1 = 0, index2 = 0, index3 = 0;
    Vec3f P1, P2, P3, D1, D2, N;
    double x, y, z, bt;
    std::list<osg::Vec3f *>::iterator it;
    unsigned int n = numFaces;
    k = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        index1 = IndexList->at(k);
        P1 = Points->at(index1);
        k++;
        index2 = IndexList->at(k);
        P2 = Points->at(index2);
        k++;
        index3 = IndexList->at(k);
        P3 = Points->at(index3);
        k++;

        D1 = P3 - P1;
        D2 = P2 - P1;

        x = D1.y() * D2.z() - D1.z() * D2.y();
        y = -D1.x() * D2.z() + D1.z() * D2.x();
        z = D1.x() * D2.y() - D1.y() * D2.x();
        bt = sqrt(x * x + y * y + z * z);
        if (bt > 0.000001)
        {
            x = -x / bt;
            y = -y / bt;
            z = -z / bt;
        }
        N.set(x, y, z);
        (*Normals).push_back(N);
        (*Normals).push_back(N);
        (*Normals).push_back(N);
    }
}

//Torsen

void RuledSurfaces::Ebene()
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3_2 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray2 = new Vec3Array();

    int k = 0;
    int k_2 = 0;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom2 = new DrawElementsUInt(PrimitiveSet::QUADS, 4);

    //1.Ebene
    vec3->push_back(osg::Vec3f(-0.1, 0, -0.1));
    vec3->push_back(osg::Vec3f(0.1, 0, -0.1));
    vec3->push_back(osg::Vec3f(0.1, 0, 0.1));
    vec3->push_back(osg::Vec3f(-0.1, 0, 0.1));

    (*mp_QuadEdges_Geom)[k] = k;
    k++;
    (*mp_QuadEdges_Geom)[k] = k;
    k++;
    (*mp_QuadEdges_Geom)[k] = k;
    k++;
    (*mp_QuadEdges_Geom)[k] = k;
    k++;

    vec3_2->push_back(osg::Vec3f(-0.1, 0, -0.1) * 2);
    vec3_2->push_back(osg::Vec3f(0.1, 0, -0.1) * 2);
    vec3_2->push_back(osg::Vec3f(0.1, 0, 0.1) * 2);
    vec3_2->push_back(osg::Vec3f(-0.1, 0, 0.1) * 2);

    (*mp_QuadEdges_Geom2)[k_2] = k_2;
    k_2++;
    (*mp_QuadEdges_Geom2)[k_2] = k_2;
    k_2++;
    (*mp_QuadEdges_Geom2)[k_2] = k_2;
    k_2++;
    (*mp_QuadEdges_Geom2)[k_2] = k_2;
    k_2++;
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, 1);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);

    geo2->addPrimitiveSet(mp_QuadEdges_Geom2);
    geo2->setVertexArray(vec3_2.get());
    fillNormalArrayQuads(vec3_2, NormalArray2, mp_QuadEdges_Geom2, 1);
    geo->setNormalArray(NormalArray2);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);

    ref_ptr<StateSet> geo_state = geo->getOrCreateStateSet();
    ref_ptr<StateSet> geo_state2 = geo2->getOrCreateStateSet();
    ref_ptr<Material> mtl = new Material;

    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.1, 0.1, 0.1745, 1.0));
    mtl->setDiffuse(Material::FRONT_AND_BACK, blue);
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.297254, 0.30829, 0.306678, 1.0));
    //mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setShininess(Material::FRONT_AND_BACK, 12.8f);
    geo_state->setAttribute(mtl.get(), osg::StateAttribute::PROTECTED);
    geo_state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    geo_state2->setAttribute(mtl.get(), osg::StateAttribute::PROTECTED);
    geo_state2->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    ebenegeo->removeDrawables(0, 1);
    ebenegeo->addDrawable(geo);

    ebenegeo2->removeDrawables(0, 1);
    ebenegeo2->addDrawable(geo2);
}

//Kegel

void RuledSurfaces::Kegel(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path2 = new Vec3Array();
    //m=m*e/2;//ausklammern: kein Absturz mehr bei kleinem u + veraendern der detailstufe
    //->detailstufe bleibt bei jedem u gleich
    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (e >= 2)
    {
        e = 2;
        stepe = 2;
    }

    float l = 2;
    float u_l = 0;
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int k = 0;
    Vec3f p1, p2, p3;
    ref_ptr<osg::DrawElementsUInt> mp_TriangleEdges_Geom = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 3 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v;
        v_j_l = v_j + delta_v;

        p1 = (osg::Vec3f(x(u_i), y(u_i), z(v_l)));
        p2 = (osg::Vec3f(x(u_i_l), y(u_i_l), z(v_l)));
        p3 = (osg::Vec3f(0, 0, 1));

        if (i == 0)
        {
            path->push_back(p1);
            path->push_back(p3);
            path->push_back(p2);
        }
        path2->push_back(p1);
        if (i == m - 1)
        {
            path2->push_back(p2);
        }
        vec3->push_back(p1);
        vec3->push_back(p2);
        vec3->push_back(p3);

        //Fläche Zeichnen
        (*mp_TriangleEdges_Geom)[k] = k;
        k++;
        (*mp_TriangleEdges_Geom)[k] = k;
        k++;
        (*mp_TriangleEdges_Geom)[k] = k;
        k++;
    }
    geo->addPrimitiveSet(mp_TriangleEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayTriangles(vec3, NormalArray, mp_TriangleEdges_Geom, m);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_TriangleEdges_Geom);
    geo2->setVertexArray(vec3.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(STRIANGLES, green);
        geo->setStateSet(mp_StateSet);

        kegelgeo->removeDrawables(0, 2);
        kegelgeo->addDrawable(geo);
    }
    else
    {
        setMaterial_surface(geo, aqua);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayTriangles(vec3, NormalArray_lines, mp_TriangleEdges_Geom, m);
        normalenMitteln_Triangles(NormalArray_lines);
        createVec3_lines_Triangles(vec3, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        kegelgeo->removeDrawables(0, 2);
        kegelgeo->addDrawable(geo);
        kegelgeo->addDrawable(geo3);
    }
    pathpoints_kegel = path;
    pathpoints3BKegel = path2;
}
//offener Kegel
void RuledSurfaces::OK(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();

    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (e >= 2)
    {
        e = 2;
        stepe = 2;
    }

    float u_l = 0;
    float u_r = (float)(e * PI);
    float u_i, u_i_l;
    float delta_u = (u_r - u_l) / m;

    int k = 0;
    Vec3f p1, p2, p3, p4;
    ref_ptr<osg::DrawElementsUInt> mp_TriangleEdges_Geom = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 3 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;

        p1 = (osg::Vec3f(sin(u_i), u_i, 0));
        p2 = (osg::Vec3f(sin(u_i_l), u_i_l, 0));
        p3 = (osg::Vec3f(0, PI, PI));
        p4 = (osg::Vec3f(1, 0, 0));

        vec3->push_back(p1);
        vec3->push_back(p2);
        vec3->push_back(p3);
        if (i == 0)
        {
            path->push_back(p1);
            path->push_back(p3);
            path->push_back(p2);
        }
        //Fläche Zeichnen
        (*mp_TriangleEdges_Geom)[k] = k;
        k++;
        (*mp_TriangleEdges_Geom)[k] = k;
        k++;
        (*mp_TriangleEdges_Geom)[k] = k;
        k++;
    }

    geo->addPrimitiveSet(mp_TriangleEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayTriangles(vec3, NormalArray, mp_TriangleEdges_Geom, m);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_TriangleEdges_Geom);
    geo2->setVertexArray(vec3.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(STRIANGLES, green);
        geo2->setStateSet(mp_StateSet);

        okgeo->removeDrawables(0, 2);
        okgeo->addDrawable(geo2);
    }
    else
    {
        setMaterial_surface(geo, aqua);
        //setMaterial_line(geo2, black);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayTriangles(vec3, NormalArray_lines, mp_TriangleEdges_Geom, m);
        normalenMitteln_Triangles(NormalArray_lines);
        createVec3_lines_Triangles(vec3, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        okgeo->removeDrawables(0, 2);
        okgeo->addDrawable(geo);
        okgeo->addDrawable(geo3);
    }
    pathpoints_ok = path;
}
void RuledSurfaces::Zylinder(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path2 = new Vec3Array();

    //m=m*e/2;
    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (e >= 2)
    {
        e = 2;
        stepe = 2;
    }

    float l = 2;
    float u_l = (float)(0);
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int facettierung = 20;
    int k = 0;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6, p7, p8;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l;
        v_j_l = v_j + delta_v;

        p1 = (osg::Vec3f(cos(u_i), sin(u_i), v_j));
        p2 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j));
        p3 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j_l));
        p4 = (osg::Vec3f(cos(u_i), sin(u_i), v_j_l));
        p7 = (osg::Vec3f(1, 0, 0));
        p8 = (osg::Vec3f(1, 0, 2));

        vec3_lines->push_back(p1);
        vec3_lines->push_back(p2);
        vec3_lines->push_back(p3);
        vec3_lines->push_back(p4);
        path->push_back(p7);
        path->push_back(p8);
        path2->push_back(p1);
        if (i == m - 1)
        {
            path2->push_back(p2);
        }

        //Fläche Zeichnen
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        for (int j = 0; j < facettierung; j++)
        {
            p1 = (osg::Vec3f(cos(u_i), sin(u_i), v_j));
            p2 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j));
            p3 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j_l));
            p4 = (osg::Vec3f(cos(u_i), sin(u_i), v_j_l));
            p5 = (osg::Vec3f(0, -1, 0));
            p6 = (osg::Vec3f(0, -1, 2));

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);
            //Fläche Zeichnen
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
        }
    }
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, m * facettierung);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        zylindergeo->removeDrawables(0, 2);
        zylindergeo->addDrawable(geo2);
    }
    else
    {
        setMaterial_surface(geo, aqua);

        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array(); //fuer Gitternetz verschieben
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, m);
        normalenMitteln_Quads(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        zylindergeo->removeDrawables(0, 2);
        zylindergeo->addDrawable(geo);
        zylindergeo->addDrawable(geo3);
    }
    pathpoints_zylinder = path;
    pathpoints3BZylinder = path2;
}
void RuledSurfaces::OZ(int m, float e) //offener Zylinder
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();

    float l = 2;
    float u_l = 0;
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int facettierung = 20;
    int k = 0;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6, p7, p8;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v;
        v_j_l = v_j + delta_v;

        vec3_lines->push_back(osg::Vec3f(u_i, x(u_i), 0));
        vec3_lines->push_back(osg::Vec3f(u_i_l, x(u_i_l), 0));
        vec3_lines->push_back(osg::Vec3f(u_i_l, x(u_i_l), 1));
        vec3_lines->push_back(osg::Vec3f(u_i, x(u_i), 1));

        p7 = (osg::Vec3f(PI, cos(PI), 0));
        p8 = (osg::Vec3f(PI, cos(PI), 1));
        path->push_back(p7);
        path->push_back(p8);
        //Fläche Zeichnen
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        for (int j = 0; j < facettierung; j++)
        {
            p1 = osg::Vec3f(u_i, x(u_i), 0);
            p2 = osg::Vec3f(u_i_l, x(u_i_l), 0);
            p3 = osg::Vec3f(u_i_l, x(u_i_l), 1);
            p4 = osg::Vec3f(u_i, x(u_i), 1);

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            //Fläche Zeichnen
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
        }
    }
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, m * facettierung);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        ozgeo->removeDrawables(0, 2);
        ozgeo->addDrawable(geo2);
    }
    else
    {
        setMaterial_surface(geo, aqua);
        //setMaterial_line(geo2, black);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, m);
        normalenMitteln_Quads(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        ozgeo->removeDrawables(0, 2);
        ozgeo->addDrawable(geo);
        ozgeo->addDrawable(geo3);
    }
    pathpoints_oz = path;
}
void RuledSurfaces::Tangentialflaeche(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    //osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();

    //m=m*e/2;

    float l = 2;
    float u_l = 0;
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int facettierung = 50;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6, p7, p8, p11, p12, p13, p14;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung /**2*/);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m /**2*/);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v;
        v_j_l = v_j + delta_v;

        p1 = osg::Vec3f(cos(u_i) + sin(u_i), sin(u_i) - cos(u_i), u_i - 1);
        p2 = osg::Vec3f(cos(u_i_l) + sin(u_i_l), sin(u_i_l) - cos(u_i_l), u_i_l - 1);
        p3 = osg::Vec3f(-sin(u_i_l) + cos(u_i_l), cos(u_i_l) + sin(u_i_l), 1 + u_i_l);
        p4 = osg::Vec3f(-sin(u_i) + cos(u_i), cos(u_i) + sin(u_i), 1 + u_i);

        vec3->push_back(p1);
        vec3->push_back(p2);
        vec3->push_back(p3);
        vec3->push_back(p4);
        //path->push_back(p1);
        //path->push_back(p4);

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;

        //for(int j=0;j<facettierung;j++){

        //	p1=osg::Vec3f(cos(u_i)+sin(u_i),sin(u_i)-cos(u_i), u_i-1);
        //p2=osg::Vec3f(cos(u_i_l)+sin(u_i_l),sin(u_i_l)-cos(u_i_l),u_i_l-1);
        //p3=osg::Vec3f(-sin(u_i_l)+cos(u_i_l),cos(u_i_l)+sin(u_i_l),1+u_i_l);
        //p4=osg::Vec3f(-sin(u_i)+cos(u_i),cos(u_i)+sin(u_i),1+u_i);
        //
        //	p5=p4-p1;
        //	p6=p3-p2;
        //
        //	p3=p2+(p6*(j+1)/facettierung);
        //	p4=p1+(p5*(j+1)/facettierung);
        //	p1=p1+(p5*j/facettierung);
        //	p2=p2+(p6*j/facettierung);

        //	vec3->push_back(p1);
        //	vec3->push_back(p2);
        //	vec3->push_back(p3);
        //	vec3->push_back(p4);

        //	(*mp_QuadEdges_Geom)[k]=k;
        //	k++;
        //	(*mp_QuadEdges_Geom)[k]=k;
        //	k++;
        //	(*mp_QuadEdges_Geom)[k]=k;
        //	k++;
        //	(*mp_QuadEdges_Geom)[k]=k;
        //	k++;
        //}
    }
    geo->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo->setVertexArray(vec3_lines.get());
    fillNormalArrayTangentialflaeche(vec3_lines, NormalArray, mp_QuadEdges_Geom_lines, m /**facettierung*/ /**2*/);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        tangentialflaechegeo->removeDrawables(0, 2);
        tangentialflaechegeo->addDrawable(geo2);
    }
    else
    {
        setMaterial_surface(geo, aqua);
        //setMaterial_line(geo2, black);
        //osg::ref_ptr<osg::Geometry> geo3 = new Geometry();//gitternetz in pos u. neg normalenrichtung verschoben
        //osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array();//Gitternetz in pos u. neg normalenrichtung verschoben
        //osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        //fillNormalArrayTangentialflaeche(vec3_lines,NormalArray_lines,mp_QuadEdges_Geom_lines,m);
        //normalenMitteln_Quads(NormalArray_lines);
        //createVec3_lines_Quads(vec3_lines,NormalArray_lines,geo3,vec3_lines_,m);
        //geo3->setVertexArray(vec3_lines_.get());
        //setMaterial_line(geo3, black);

        tangentialflaechegeo->removeDrawables(0, 2);
        tangentialflaechegeo->addDrawable(geo);
        //tangentialflaechegeo->addDrawable(geo2);
    }
    //pathpoints_tangentialflaeche=path;
}
//windschiefe Regelflächen
void RuledSurfaces::Oloid(int m)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    float sh = 0;
    if (Trennung)
    {
        sh = 0.5f;
    }
    else
    {
        sh = 0;
    }
    float u_l = 0;
    float u_r = (float)(PI / 2);
    float v_l = (float)(3 * PI / 2);
    float v_r = (float)(2 * PI);
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l) / m;

    float a_l = (float)(PI / 2);
    float a_r = 0;
    float b_l = (float)(3 * PI / 2);
    float b_r = (float)(2 * PI);
    float a_i, b_j, a_i_l, b_j_l;
    float delta_a = (a_r - a_l) / m;
    float delta_b = (b_r - b_l) / m;

    int facettierung = 60;
    int k = 0;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 16 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 16 * m);
    for (int i = 0; i < m; i++)
    {

        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v * i;
        v_j_l = v_j + delta_v;

        a_i = a_l + delta_a * i;
        a_i_l = a_i + delta_a;
        b_j = b_l + delta_b * i;
        b_j_l = b_j + delta_b;

        //Erste Fläche
        vec3_lines->push_back(osg::Vec3f(cos(u_i_l) + sh, sin(u_i_l), sh));
        vec3_lines->push_back(osg::Vec3f(cos(u_i) + sh, sin(u_i), sh));
        vec3_lines->push_back(osg::Vec3f(sh, sin(v_j) - 1, cos(v_j) + sh));
        vec3_lines->push_back(osg::Vec3f(sh, sin(v_j_l) - 1, cos(v_j_l) + sh));

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;

        //zweite Fläche
        vec3_lines->push_back(osg::Vec3f(cos(a_i_l) + sh, sin(a_i_l), -sh));
        vec3_lines->push_back(osg::Vec3f(cos(a_i) + sh, sin(a_i), -sh));
        vec3_lines->push_back(osg::Vec3f(sh, sin(v_j - PI / 2) - 1, cos(v_j - PI / 2) - sh));
        vec3_lines->push_back(osg::Vec3f(sh, sin(v_j_l - PI / 2) - 1, cos(v_j_l - PI / 2) - sh));

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;

        //dritte Fläche
        vec3_lines->push_back(osg::Vec3f(cos(u_i_l + PI / 2) - sh, sin(u_i_l + PI / 2), sh));
        vec3_lines->push_back(osg::Vec3f(cos(u_i + PI / 2) - sh, sin(u_i + PI / 2), sh));
        vec3_lines->push_back(osg::Vec3f(-sh, sin(v_j - PI / 2) - 1, -cos(v_j - PI / 2) + sh));
        vec3_lines->push_back(osg::Vec3f(-sh, sin(v_j_l - PI / 2) - 1, -cos(v_j_l - PI / 2) + sh));

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        //vierte Fläche
        vec3_lines->push_back(osg::Vec3f(cos(u_i_l + PI / 2) - sh, sin(u_i_l + PI / 2), -sh));
        vec3_lines->push_back(osg::Vec3f(cos(u_i + PI / 2) - sh, sin(u_i + PI / 2), -sh));
        vec3_lines->push_back(osg::Vec3f(-sh, sin(v_j - PI / 2) - 1, cos(v_j - PI / 2) - sh));
        vec3_lines->push_back(osg::Vec3f(-sh, sin(v_j_l - PI / 2) - 1, cos(v_j_l - PI / 2) - sh));

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        for (int j = 0; j < facettierung; j++)
        {
            p1 = osg::Vec3f(cos(u_i_l) + sh, sin(u_i_l), sh);
            p2 = osg::Vec3f(cos(u_i) + sh, sin(u_i), sh);
            p3 = osg::Vec3f(sh, sin(v_j) - 1, cos(v_j) + sh);
            p4 = osg::Vec3f(sh, sin(v_j_l) - 1, cos(v_j_l) + sh);

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;

            //zweite Fläche
            p1 = osg::Vec3f(cos(a_i_l) + sh, sin(a_i_l), -sh);
            p2 = osg::Vec3f(cos(a_i) + sh, sin(a_i), -sh);
            p3 = osg::Vec3f(sh, sin(v_j - PI / 2) - 1, cos(v_j - PI / 2) - sh);
            p4 = osg::Vec3f(sh, sin(v_j_l - PI / 2) - 1, cos(v_j_l - PI / 2) - sh);

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;

            //dritte Fläche
            p1 = osg::Vec3f(cos(u_i_l + PI / 2) - sh, sin(u_i_l + PI / 2), sh);
            p2 = osg::Vec3f(cos(u_i + PI / 2) - sh, sin(u_i + PI / 2), sh);
            p3 = osg::Vec3f(-sh, sin(v_j - PI / 2) - 1, -cos(v_j - PI / 2) + sh);
            p4 = osg::Vec3f(-sh, sin(v_j_l - PI / 2) - 1, -cos(v_j_l - PI / 2) + sh);

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            //vierte Fläche
            p1 = osg::Vec3f(cos(u_i_l + PI / 2) - sh, sin(u_i_l + PI / 2), -sh);
            p2 = osg::Vec3f(cos(u_i + PI / 2) - sh, sin(u_i + PI / 2), -sh);
            p3 = osg::Vec3f(-sh, sin(v_j - PI / 2) - 1, cos(v_j - PI / 2) - sh);
            p4 = osg::Vec3f(-sh, sin(v_j_l - PI / 2) - 1, cos(v_j_l - PI / 2) - sh);

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
        }
    }
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, 4 * m * facettierung);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        oloidgeo->removeDrawables(0, 2);
        oloidgeo->addDrawable(geo2);
    }
    else
    {
        setMaterial_surface(geo, aqua);
        //setMaterial_line(geo2, black);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, 4 * m);
        normalenMitteln_Oloid(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, 4 * m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        oloidgeo->removeDrawables(0, 2);
        oloidgeo->addDrawable(geo);
        oloidgeo->addDrawable(geo3);
    }
}
void RuledSurfaces::Hyper(int m, float d, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Geometry> geo3 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec5 = new Vec3Array();

    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path2 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (e >= 2)
    {
        e = 2;
        stepe = 2;
    }

    float u_l = 0;
    float u_r = (float)(e * PI);
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;

    int facettierung = 100;
    int k = 0;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = 0;
        v_j_l = 1;

        p1 = (osg::Vec3f(cos(u_i), sin(u_i), v_j));
        p2 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j));
        p3 = (osg::Vec3f(cos(u_i_l + (d * PI)), sin(u_i_l + (d * PI)), v_j_l));
        p4 = (osg::Vec3f(cos(u_i + (d * PI)), sin(u_i + (d * PI)), v_j_l));

        path2->push_back(p1);
        path2->push_back(p4);
        path2->push_back(p2);
        path2->push_back(p3);
        path->push_back(p1);
        if (i == m - 1)
        {
            path->push_back(p2);
        }
        vec3_lines->push_back(p1);
        vec3_lines->push_back(p2);
        vec3_lines->push_back(p3);
        vec3_lines->push_back(p4);

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        for (int j = 0; j < facettierung; j++)
        {

            p1 = (osg::Vec3f(cos(u_i), sin(u_i), v_j));
            p2 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j));
            p3 = (osg::Vec3f(cos(u_i_l + (d * PI)), sin(u_i_l + (d * PI)), v_j_l));
            p4 = (osg::Vec3f(cos(u_i + (d * PI)), sin(u_i + (d * PI)), v_j_l));

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            //Fläche Zeichnen
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
        }
    }
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, m * facettierung);
    geo->setNormalArray(NormalArray);
    if (stepd <= 1)
    {
        geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    }
    else
    {
        geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    }
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());
    pathpoints3BHyper = path;
    pathpoints_hyper = path2;

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        hypergeo->removeDrawables(0, 2);
        hypergeo->addDrawable(geo2);
    }
    else
    {
        Vec4 colory; //drall
        float cd = sin(stepd * PI);
        if (cd <= 0)
            colory = colorArray->at(1) * (1 - abs(cd)) + colorArray->at(4) * abs(cd);
        else
            colory = colorArray->at(1) * (1 - abs(cd)) + colorArray->at(6) * abs(cd);

        setMaterial_surface(geo, colory);
        //setMaterial_line(geo2, black);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, m);
        normalenMitteln_Quads(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        hypergeo->removeDrawables(0, 2);
        hypergeo->addDrawable(geo);
        //hypergeo->addDrawable(geo2);
        hypergeo->addDrawable(geo3);
    }
}

void RuledSurfaces::Sattel(int m)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry(); //flaeche
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry(); //gitternetz
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array(); //Gitternetz

    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array(); //fuer Gitternetz verschieben
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path2 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path3 = new Vec3Array();

    float u_l = -0.5;
    float u_r = 1.5;
    float v_l = 1.f;
    float v_r = -1.f;
    /*float u_l=0;
	float u_r=1;
	float v_l=0.5f;
	float v_r=-0.5f;*/
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / (m);
    float delta_v = (v_l - v_r) / (m);

    int facettierung = 50;
    int k = 0;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m);

    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_r + delta_v * i;
        v_j_l = v_j + delta_v;

        p1 = (osg::Vec3f(u_i, 0, 0));
        p2 = (osg::Vec3f(u_i_l, 0, 0));
        p3 = (osg::Vec3f(0.5, v_j_l, 2));
        p4 = (osg::Vec3f(0.5, v_j, 2));

        path->push_back(p1);
        path2->push_back(p4);
        if (i == m - 1)
        {
            path->push_back(p2);
            path2->push_back(p3);
        }
        vec3_lines->push_back(p1);
        vec3_lines->push_back(p2);
        vec3_lines->push_back(p3);
        vec3_lines->push_back(p4);

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;

        for (int j = 0; j < facettierung; j++)
        {
            p1 = (osg::Vec3f(u_i, 0, 0));
            p2 = (osg::Vec3f(u_i_l, 0, 0));
            p3 = (osg::Vec3f(0.5, v_j_l, 2));
            p4 = (osg::Vec3f(0.5, v_j, 2));

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p1);
            vec3->push_back(p2);
            vec3->push_back(p3);
            vec3->push_back(p4);

            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
        }
    }
    path3->push_back(osg::Vec3(0.5, 0.01, 0));
    path3->push_back(osg::Vec3(0.5 - 0.01, 0, 2));

    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, m * facettierung);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        sattelgeo->removeDrawables(0, 2);
        sattelgeo->addDrawable(geo2);
    }
    else
    {
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, m);
        normalenMitteln_Quads(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        setMaterial_surface(geo, aqua);

        sattelgeo->removeDrawables(0, 2);
        sattelgeo->addDrawable(geo);
        //sattelgeo->addDrawable(geo2);
        sattelgeo->addDrawable(geo3);
    }
    pathpoints3BSattel = path;
    pathpoints3BSattel2 = path2;
    pathpoints_sattel = path3;
}
void RuledSurfaces::normalenMitteln_Oloid(Vec3Array *Normals)
{
    int anzahl = Normals->size() / 4; //anzahl pro Flaeche
    for (int i = 0; i < anzahl - 4; i++)
    {
        if (mod(i, 4) == 1)
        {
            Vec3f normale = Normals->at(i) + (Normals->at(i) - Normals->at(i + 3)) / 2;
            float normalize = 1 / sqrtf(normale.x() * normale.x() + normale.y() * normale.y() + normale.z() * normale.z());
            normale = normale * normalize;
            Normals->at(i) = normale;
            Normals->at(i + 3) = normale;
            Normals->at(i + 1) = normale;
            Normals->at(i + 6) = normale;
        }
    }
    for (int i = (anzahl + 1); i < 2 * anzahl - 4; i++)
    {
        if (mod(i, 4) == 1)
        {
            Vec3f normale = Normals->at(i) + (Normals->at(i) - Normals->at(i + 3)) / 2;
            float normalize = 1 / sqrtf(normale.x() * normale.x() + normale.y() * normale.y() + normale.z() * normale.z());
            normale = normale * normalize;
            Normals->at(i) = normale;
            Normals->at(i + 3) = normale;
            Normals->at(i + 1) = normale;
            Normals->at(i + 6) = normale;
        }
    }
    for (int i = (2 * anzahl + 1); i < 3 * anzahl - 4; i++)
    {
        if (mod(i, 4) == 1)
        {
            Vec3f normale = Normals->at(i) + (Normals->at(i) - Normals->at(i + 3)) / 2;
            float normalize = 1 / sqrtf(normale.x() * normale.x() + normale.y() * normale.y() + normale.z() * normale.z());
            normale = normale * normalize;
            Normals->at(i) = normale;
            Normals->at(i + 3) = normale;
            Normals->at(i + 1) = normale;
            Normals->at(i + 6) = normale;
        }
    }
    for (int i = (3 * anzahl + 1); i < 4 * anzahl - 4; i++)
    {
        if (mod(i, 4) == 1)
        {
            Vec3f normale = Normals->at(i) + (Normals->at(i) - Normals->at(i + 3)) / 2;
            float normalize = 1 / sqrtf(normale.x() * normale.x() + normale.y() * normale.y() + normale.z() * normale.z());
            normale = normale * normalize;
            Normals->at(i) = normale;
            Normals->at(i + 3) = normale;
            Normals->at(i + 1) = normale;
            Normals->at(i + 6) = normale;
        }
    }
}
void RuledSurfaces::normalenMitteln_Quads(Vec3Array *Normals)
{
    for (size_t i = 0; i < Normals->size() - 4; i++)
    {
        if (mod(i, 4) == 1)
        {
            Vec3f normale = Normals->at(i) + (Normals->at(i) - Normals->at(i + 3)) / 2;
            float normalize = 1 / sqrtf(normale.x() * normale.x() + normale.y() * normale.y() + normale.z() * normale.z());
            normale = normale * normalize;
            Normals->at(i) = normale;
            Normals->at(i + 3) = normale;
            Normals->at(i + 1) = normale;
            Normals->at(i + 6) = normale;
        }
    }
}
void RuledSurfaces::normalenMitteln_Triangles(Vec3Array *Normals)
{
    for (size_t i = 0; i < Normals->size() - 3; i++)
    {
        if (mod(i, 3) == 1)
        {
            Vec3f normale = Normals->at(i) + (Normals->at(i) - Normals->at(i + 2)) / 2;
            float normalize = 1 / sqrtf(normale.x() * normale.x() + normale.y() * normale.y() + normale.z() * normale.z());
            normale = normale * normalize;
            Normals->at(i) = normale;
            Normals->at(i + 2) = normale;
        }
        if (mod(i, 3) == 2)
        {
            Vec3f normale = osg::Vec3f(0, 0, 1);
            Normals->at(i) = normale;
        }
    }
    Vec3f normale = osg::Vec3f(0, 0, 1);
    Normals->at(Normals->size() - 1) = normale;
}
void RuledSurfaces::createVec3_lines_Quads(Vec3Array *Points, Vec3Array *Normals, Geometry *geom, Vec3Array *Points2, int numFaces)
{
    float abstand = 0.0003f;
    int k = 0;
    int m = numFaces;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines_ = new DrawElementsUInt(PrimitiveSet::QUADS, 2 * 4 * m);
    for (unsigned int i = 0; i < (*Points).size(); i++)
    {
        Vec3f punkt = (Normals->at(i) * abstand);
        punkt = Points->at(i) + punkt;
        Points2->push_back(punkt);
        (*mp_QuadEdges_Geom_lines_)[k] = k;
        k++;
    }
    for (unsigned int i = 0; i < (*Points).size(); i++)
    {
        Vec3f punkt = (Normals->at(i) * abstand);
        punkt = Points->at(i) - punkt;
        Points2->push_back(punkt);
        (*mp_QuadEdges_Geom_lines_)[k] = k;
        k++;
    }
    geom->addPrimitiveSet(mp_QuadEdges_Geom_lines_);
}
void RuledSurfaces::createVec3_lines_Tangentialflaeche(Vec3Array *Points, Vec3Array *Normals, Geometry *geom, Vec3Array *Points2, int numFaces)
{
    float abstand = 0.001f; /*0.004*/
    int k = 0;
    int m = numFaces;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines_ = new DrawElementsUInt(PrimitiveSet::LINES, 2 * (2 * (4) * m + 4));
    for (int ii = 0; ii < 2; ii++)
    {
        if (ii == 1)
        {
            abstand = -abstand;
        }
        for (unsigned int i = 0; i < (*Points).size(); i++)
        {
            if (mod(i, 4) == 0 || mod(i, 4) == 1)
            {
                Vec3f punkt = (Normals->at(i) * abstand);
                punkt = Points->at(i) + punkt;
                Points2->push_back(punkt);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;
            }
        }
        for (unsigned int i = 0; i < (*Points).size(); i++)
        {
            if (mod(i, 4) == 3 || mod(i, 4) == 2)
            {
                Vec3f punkt = (Normals->at(i) * abstand);
                punkt = Points->at(i) + punkt;
                Points2->push_back(punkt);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;
            }
        }
        for (unsigned int i = 0; i < (*Points).size(); i++)
        {
            if (i == 0)
            {
                Vec3f punkt = (Normals->at(i) * abstand);
                punkt = Points->at(i) + punkt;
                Points2->push_back(punkt);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;

                Vec3f punkt2 = (Normals->at(i) * abstand);
                Vec3f kurvenpunkt = Points->at(i) + (Points->at(i + 3) - Points->at(i)) / 2;
                punkt2 = kurvenpunkt + punkt2;
                Points2->push_back(punkt2);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;
                Points2->push_back(punkt2);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;

                Vec3f punkt3 = (Normals->at(i) * abstand);
                punkt3 = Points->at(i + 3) + punkt3;
                Points2->push_back(punkt3);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;
            }
            if (mod(i, 4) == 1)
            { //erste noch extra machen
                Vec3f punkt = (Normals->at(i) * abstand);
                punkt = Points->at(i) + punkt;
                Points2->push_back(punkt);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;

                Vec3f punkt2 = (Normals->at(i) * abstand);
                Vec3f kurvenpunkt = Points->at(i) + (Points->at(i + 1) - Points->at(i)) / 2;
                punkt2 = kurvenpunkt + punkt2;
                Points2->push_back(punkt2);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;
                Points2->push_back(punkt2);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;

                Vec3f punkt3 = (Normals->at(i /*+1*/) * abstand);
                punkt3 = Points->at(i + 1) + punkt3;
                Points2->push_back(punkt3);
                (*mp_QuadEdges_Geom_lines_)[k] = k;
                k++;
            }
        }
    }
    geom->addPrimitiveSet(mp_QuadEdges_Geom_lines_);
}
void RuledSurfaces::createVec3_lines_Triangles(Vec3Array *Points, Vec3Array *Normals, Geometry *geom, Vec3Array *Points2, int numFaces)
{
    float abstand = 0.0003f;
    int k = 0;
    int m = numFaces;
    ref_ptr<osg::DrawElementsUInt> mp_TriangleEdges_Geom_lines_ = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 2 * 3 * m);

    for (unsigned int i = 0; i < (*Points).size(); i++)
    {
        Vec3f punkt = (Normals->at(i) * abstand);
        punkt = Points->at(i) + punkt;
        Points2->push_back(punkt);
        (*mp_TriangleEdges_Geom_lines_)[k] = k;
        k++;
    }
    for (unsigned int i = 0; i < (*Points).size(); i++)
    {
        Vec3f punkt = (Normals->at(i) * abstand);
        punkt = Points->at(i) - punkt;
        Points2->push_back(punkt);
        (*mp_TriangleEdges_Geom_lines_)[k] = k;
        k++;
    }
    geom->addPrimitiveSet(mp_TriangleEdges_Geom_lines_);
}

void RuledSurfaces::Konoid(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path1 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path2 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path4 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path5 = new Vec3Array();
    osg::ref_ptr<osg::Vec2Array> angle = new Vec2Array();
    osg::ref_ptr<osg::Vec4Array> Dr_color = new Vec4Array();

    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0.1;
    }
    if (e >= 1)
    {
        e = 1;
        stepe = 1;
    }

    float u_l = 0;
    float u_r = (float)((e)*PI);
    float v_l = 1;
    float v_r = -1;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l) / m;

    int facettierung = 40;
    int k = 0, k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v * i;
        v_j_l = v_j + delta_v;

        if (u_i <= PI / 2)
        {
            p11 = (osg::Vec3f(cos(u_i_l), 0, -1));
            p12 = (osg::Vec3f(cos(u_i), 0, -1));
            //Viertelkreis1
            p13 = (osg::Vec3f(cos(u_i), sin(u_i), 1));
            p14 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), 1));

            /*if(drall==true){
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-cos(u_i))+colorArray->at(6)*cos(u_i)));
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-cos(u_i_l))+colorArray->at(6)*cos(u_i_l)));
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-cos(u_i_l))+colorArray->at(6)*cos(u_i_l)));
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-cos(u_i))+colorArray->at(6)*cos(u_i)));
			}*/

            //path->push_back(p11);
            vec3_lines->push_back(p11);
            vec3_lines->push_back(p12);
            vec3_lines->push_back(p13);
            vec3_lines->push_back(p14);
            path->push_back(p12);

            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;
            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;
            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;
            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;

            //Winkel fuer Dreibein
            osg::Vec3f vec_oben = p13 - p12;
            osg::Vec3f vec_unten = osg::Vec3f(cos(u_i), 0, 1) - p12;

            double norm_oben = sqrt(vec_oben.x() * vec_oben.x() + vec_oben.y() * vec_oben.y() + vec_oben.z() * vec_oben.z());
            double norm_unten = sqrt(vec_unten.x() * vec_unten.x() + vec_unten.y() * vec_unten.y() + vec_unten.z() * vec_unten.z());
            double skalar_obenunten = vec_oben.x() * vec_unten.x() + vec_oben.y() * vec_unten.y() + vec_oben.z() * vec_unten.z();
            float angf = (float)acos(skalar_obenunten / (norm_oben * norm_unten)); //Winkel fuer Animation
            angle->push_back(Vec2f(angf, 0.0f));
        }
        else
        {
            //Viertelkreis2
            p15 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), 1));
            p16 = (osg::Vec3f(cos(u_i), sin(u_i), 1));
            //Gerade2
            p17 = (osg::Vec3f(cos(u_i), 0, -1));
            p18 = (osg::Vec3f(cos(u_i_l), 0, -1));

            /*if(drall==true){
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-abs(cos(u_i)))+colorArray->at(4)*abs(cos(u_i))));
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-abs(cos(u_i_l)))+colorArray->at(4)*abs(cos(u_i_l))));
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-abs(cos(u_i_l)))+colorArray->at(4)*abs(cos(u_i_l))));
			Dr_color->push_back(osg::Vec4f(colorArray->at(1)*(1-abs(cos(u_i)))+colorArray->at(4)*abs(cos(u_i))));
			}*/
            vec3_lines->push_back(p15);
            vec3_lines->push_back(p18);
            vec3_lines->push_back(p17);
            vec3_lines->push_back(p16);
            path->push_back(p17);
            if (i == m - 1)
                path->push_back(p18);

            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;
            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;
            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;
            (*mp_QuadEdges_Geom_lines)[k2] = k2;
            k2++;

            //Winkel fuer Dreibein
            osg::Vec3f vec_oben = p16 - p17;
            osg::Vec3f vec_unten = osg::Vec3f(cos(u_i), 0, 1) - p17;

            double norm_oben = sqrt(vec_oben.x() * vec_oben.x() + vec_oben.y() * vec_oben.y() + vec_oben.z() * vec_oben.z());
            double norm_unten = sqrt(vec_unten.x() * vec_unten.x() + vec_unten.y() * vec_unten.y() + vec_unten.z() * vec_unten.z());
            double skalar_obenunten = vec_oben.x() * vec_unten.x() + vec_oben.y() * vec_unten.y() + vec_oben.z() * vec_unten.z();
            float angf = (float)acos(skalar_obenunten / (norm_oben * norm_unten)); //Winkel fuer Animation
            angle->push_back(Vec2f(angf, 0.0f));
            if (i == m - 1)
            {
                vec_oben = p15 - p18;
                vec_unten = osg::Vec3f(cos(u_i_l), 0, 1) - p18;

                norm_oben = sqrt(vec_oben.x() * vec_oben.x() + vec_oben.y() * vec_oben.y() + vec_oben.z() * vec_oben.z());
                norm_unten = sqrt(vec_unten.x() * vec_unten.x() + vec_unten.y() * vec_unten.y() + vec_unten.z() * vec_unten.z());
                skalar_obenunten = vec_oben.x() * vec_unten.x() + vec_oben.y() * vec_unten.y() + vec_oben.z() * vec_unten.z();
                angf = (float)acos(skalar_obenunten / (norm_oben * norm_unten)); //Winkel fuer Animation
                angle->push_back(Vec2f(angf, 0.0f));
            }
        }
        for (int j = 0; j < facettierung; j++)
        {
            if (u_i <= PI / 2)
            {
                p11 = (osg::Vec3f(cos(u_i_l), 0, -1));
                p12 = (osg::Vec3f(cos(u_i), 0, -1));
                //Viertelkreis1
                p13 = (osg::Vec3f(cos(u_i), sin(u_i), 1));
                p14 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), 1));

                if (drall == true)
                {
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i_l))) + colorArray->at(4) * abs(cos(u_i_l))));
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i))) + colorArray->at(4) * abs(cos(u_i))));
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i))) + colorArray->at(4) * abs(cos(u_i))));
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i_l))) + colorArray->at(4) * abs(cos(u_i_l))));
                }

                p15 = p14 - p11;
                p16 = p13 - p12;
                p13 = p12 + (p16 * (j + 1) / (facettierung));
                p14 = p11 + (p15 * (j + 1) / (facettierung));
                p11 = p11 + (p15 * j / (facettierung));
                p12 = p12 + (p16 * j / (facettierung));

                vec3->push_back(p11);
                vec3->push_back(p12);
                vec3->push_back(p13);
                vec3->push_back(p14);

                (*mp_QuadEdges_Geom)[k] = k;
                k++;
                (*mp_QuadEdges_Geom)[k] = k;
                k++;
                (*mp_QuadEdges_Geom)[k] = k;
                k++;
                (*mp_QuadEdges_Geom)[k] = k;
                k++;
            }
            else
            {
                //Viertelkreis2
                p21 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), 1));
                p22 = (osg::Vec3f(cos(u_i), sin(u_i), 1));
                //Gerade2
                p23 = (osg::Vec3f(cos(u_i), 0, -1));
                p24 = (osg::Vec3f(cos(u_i_l), 0, -1));

                if (drall == true)
                {
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i_l))) + colorArray->at(6) * abs(cos(u_i_l))));
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i))) + colorArray->at(6) * abs(cos(u_i))));
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i))) + colorArray->at(6) * abs(cos(u_i))));
                    Dr_color->push_back(osg::Vec4f(colorArray->at(1) * (1 - abs(cos(u_i_l))) + colorArray->at(6) * abs(cos(u_i_l))));
                }
                p25 = p24 - p21;
                p26 = p23 - p22;
                p23 = p22 + (p26 * (j + 1) / (facettierung));
                p24 = p21 + (p25 * (j + 1) / (facettierung));
                p21 = p21 + (p25 * j / (facettierung));
                p22 = p22 + (p26 * j / (facettierung));

                vec3->push_back(p21);
                vec3->push_back(p22);
                vec3->push_back(p23);
                vec3->push_back(p24);

                (*mp_QuadEdges_Geom)[k] = k;
                k++;
                (*mp_QuadEdges_Geom)[k] = k;
                k++;
                (*mp_QuadEdges_Geom)[k] = k;
                k++;
                (*mp_QuadEdges_Geom)[k] = k;
                k++;
            }
        }

        //Pathpoints für Animation
        /*p1=(osg::Vec3f(1,0,-1));
	p2=(osg::Vec3f(1,0,1));*/
        p3 = (osg::Vec3f(0.5, 0.01, -1));
        p4 = (osg::Vec3f(0.5, sqrtf(1 - 0.25) + 0.01, 1));
        p5 = (osg::Vec3f(cos(PI / 2), 0 + 0.02, -1));
        p6 = (osg::Vec3f(cos(PI / 2), sin(PI / 2) + 0.01, 1));
        p7 = (osg::Vec3f(-0.5, 0 + 0.01, -1));
        p8 = (osg::Vec3f(-0.5, sqrtf(1 - 0.25) + 0.01, 1));
        /*p9=(osg::Vec3f(-1,0,-1));
	p10=(osg::Vec3f(-1,0,1));*/

        p1 = (osg::Vec3f(1.02, 0.02, -1));
        p2 = (osg::Vec3f(1.02, 0.02, 1));
        /*p3=(osg::Vec3f(0.52,0.02,-1));
	p4=(osg::Vec3f(cos(PI/4)+0.02,sin(PI/4)+0.02,1));
	p5=(osg::Vec3f(cos(PI/2),0.02,-1));
	p6=(osg::Vec3f(0,1.02,1));
	p7=(osg::Vec3f(-0.52,0.02,-1));
	p8=(osg::Vec3f(cos(3*PI/4)-0.02,sin(3*PI/4)+0.02,1));*/
        p9 = (osg::Vec3f(-1.02, 0.02, -1));
        p10 = (osg::Vec3f(-1.02, 0.02, 1));
        //orig
        //p1=(osg::Vec3f(1,0.02,-1));
        //p2=(osg::Vec3f(1.02,0,1));
        //p3=(osg::Vec3f(0.52,0.02,-1));
        //p4=(osg::Vec3f(cos(PI/4)+0.02,sin(PI/4)+0.02,1));
        //p5=(osg::Vec3f(cos(PI/2),0.02,-1));
        //p6=(osg::Vec3f(0,1.02,1));
        //p7=(osg::Vec3f(-0.52,0.02,-1));
        //p8=(osg::Vec3f(cos(3*PI/4)-0.02,sin(3*PI/4)+0.02,1));
        //p9=(osg::Vec3f(-1,0.02,-1));
        //p10=(osg::Vec3f(-1.02,0,1));
        path1->push_back(p1);
        path1->push_back(p2);
        path2->push_back(p3);
        path2->push_back(p4);
        path3->push_back(p5);
        path3->push_back(p6);
        path4->push_back(p7);
        path4->push_back(p8);
        path5->push_back(p9);
        path5->push_back(p10);
    }
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, m * facettierung);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    if (drall == true)
    {
        geo->setColorArray(Dr_color.get());
        geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        osg::StateSet *sset = geo->getOrCreateStateSet();
        osg::Material *mat = new osg::Material();

        sset->setAttributeAndModes(mat, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);

        /*mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
		sset->setAttributeAndModes(mat,StateAttribute::PROTECTED );*/
    }
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        konoidgeo->removeDrawables(0, 2);
        konoidgeo->addDrawable(geo2);
    }
    else
    {
        if (drall != true)
        {
            setMaterial_surface(geo, aqua);
        }
        //setMaterial_line(geo2, black);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, m);
        normalenMitteln_Quads(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        konoidgeo->removeDrawables(0, 2);
        konoidgeo->addDrawable(geo);
        konoidgeo->addDrawable(geo3);
    }
    angle3BKonoid = angle;
    pathpoints3BKonoid = path;
    pathpointst1 = path1;
    pathpointst2 = path2;
    pathpointst3 = path3;
    pathpointst4 = path4;
    pathpointst5 = path5;
}
//Schraubfläche1
void RuledSurfaces::Helix(int m, float e, float r1, float r2, float h1, float h2, float h3)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec3_lines = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path2 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> path3 = new Vec3Array();

    r1 = 0;
    h1 = 0;

    //m=m*e/2;
    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (r1 <= 0)
    {
        r1 = 0;
        stepr1 = 0;
    }
    if (r2 <= 0)
    {
        r2 = r1 + 0.1;
        stepr2 = stepr1 + 0.1;
    }
    if (r1 >= r2)
    {
        r1 = r2 - 0.1;
        stepr1 = stepr2 - 0.1;
    }

    float l = 2;
    float u_l = 0;
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int facettierung = 40;
    int k = 0;
    int k2 = 0;
    Vec3f p1, p2, p3, p4, p5, p6;
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m * facettierung);
    ref_ptr<osg::DrawElementsUInt> mp_QuadEdges_Geom_lines = new DrawElementsUInt(PrimitiveSet::QUADS, 4 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v;
        v_j_l = v_j + delta_v;

        p1 = (osg::Vec3f(r1 * cos(u_i), r1 * sin(u_i), h3 * u_i + h1));
        p2 = (osg::Vec3f(r1 * cos(u_i_l), r1 * sin(u_i_l), h3 * u_i_l + h1));
        p3 = (osg::Vec3f(r2 * cos(u_i_l), r2 * sin(u_i_l), h3 * u_i_l + h2));
        p4 = (osg::Vec3f(r2 * cos(u_i), r2 * sin(u_i), h3 * u_i + h2));

        vec3_lines->push_back(p4);
        vec3_lines->push_back(p3);
        vec3_lines->push_back(p2);
        vec3_lines->push_back(p1);

        path->push_back(p1);
        if (i == m - 1)
        {
            path->push_back(p2);
        }
        path2->push_back(p1);
        path2->push_back(p4);
        path3->push_back(p4);
        path3->push_back(p3);

        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;
        (*mp_QuadEdges_Geom_lines)[k2] = k2;
        k2++;

        for (int j = 0; j < facettierung; j++)
        {
            p1 = (osg::Vec3f(r1 * cos(u_i), r1 * sin(u_i), h3 * u_i + h1));
            p2 = (osg::Vec3f(r1 * cos(u_i_l), r1 * sin(u_i_l), h3 * u_i_l + h1));
            p3 = (osg::Vec3f(r2 * cos(u_i_l), r2 * sin(u_i_l), h3 * u_i_l + h2));
            p4 = (osg::Vec3f(r2 * cos(u_i), r2 * sin(u_i), h3 * u_i + h2));

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            vec3->push_back(p4);
            vec3->push_back(p3);
            vec3->push_back(p2);
            vec3->push_back(p1);

            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
            (*mp_QuadEdges_Geom)[k] = k;
            k++;
        }
    }
    geo->addPrimitiveSet(mp_QuadEdges_Geom);
    geo->setVertexArray(vec3.get());
    fillNormalArrayQuads(vec3, NormalArray, mp_QuadEdges_Geom, m * facettierung);
    geo->setNormalArray(NormalArray);
    geo->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geo2->addPrimitiveSet(mp_QuadEdges_Geom_lines);
    geo2->setVertexArray(vec3_lines.get());

    if (TextureMode)
    {
        HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SQUADS, green);
        geo2->setStateSet(mp_StateSet);

        helixgeo->removeDrawables(0, 2);
        helixgeo->addDrawable(geo2);
    }
    else
    {
        setMaterial_surface(geo, aqua);
        //setMaterial_line(geo2, black);
        osg::ref_ptr<osg::Geometry> geo3 = new Geometry(); //gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> vec3_lines_ = new Vec3Array(); //Gitternetz in pos u. neg normalenrichtung verschoben
        osg::ref_ptr<osg::Vec3Array> NormalArray_lines = new Vec3Array();
        fillNormalArrayQuads(vec3_lines, NormalArray_lines, mp_QuadEdges_Geom_lines, m);
        normalenMitteln_Quads(NormalArray_lines);
        createVec3_lines_Quads(vec3_lines, NormalArray_lines, geo3, vec3_lines_, m);
        geo3->setVertexArray(vec3_lines_.get());
        setMaterial_line(geo3, black);

        helixgeo->removeDrawables(0, 2);
        helixgeo->addDrawable(geo);
        helixgeo->addDrawable(geo3);
    }
    pathpoints3BHelix = path;
    pathpoints_helix = path2;
    pathpoints_helix2 = path3;
}

//Striktionslinien

void RuledSurfaces::StriktionslinieHyper(int m, float d, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> vec5 = new Vec3Array();

    osg::ref_ptr<osg::Vec3Array> path = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (e >= 2)
    {
        e = 2;
        stepe = 2;
    }

    float u_l = 0;
    float u_r = (float)(e * PI);
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;

    int facettierung = 100;
    int k = 0;
    Vec3f p1, p2, p3, p4, p5, p6;
    ref_ptr<osg::DrawElementsUInt> mp_CircEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINE_LOOP, 2 * 2 * m);
    for (int ii = 0; ii < 2; ii++)
    {
        for (int i = 0; i < m; i++)
        {
            u_i = u_l + delta_u * i;
            u_i_l = u_i + delta_u;
            v_j = 0;
            v_j_l = 1;

            int j = facettierung / 2;

            p1 = (osg::Vec3f(cos(u_i), sin(u_i), v_j));
            p2 = (osg::Vec3f(cos(u_i_l), sin(u_i_l), v_j));
            p3 = (osg::Vec3f(cos(u_i_l + (d * PI)), sin(u_i_l + (d * PI)), v_j_l));
            p4 = (osg::Vec3f(cos(u_i + (d * PI)), sin(u_i + (d * PI)), v_j_l));

            p5 = p4 - p1;
            p6 = p3 - p2;
            p3 = p2 + (p6 * (j + 1) / facettierung);
            p4 = p1 + (p5 * (j + 1) / facettierung);
            p1 = p1 + (p5 * j / facettierung);
            p2 = p2 + (p6 * j / facettierung);

            float abstand = 0.0003f;
            Vec3f normale1 = osg::Vec3f(p1.x(), p1.y(), 0) * abstand;
            Vec3f normale2 = osg::Vec3f(p3.x(), p3.y(), 0) * abstand;

            if (ii == 0)
            {
                p1 = p1 + normale1;
                p2 = p2 + normale2;
            }
            else
            {
                p1 = p1 - normale1;
                p2 = p2 - normale2;
            }
            vec3->push_back(p1);
            vec3->push_back(p2);

            (*mp_CircEdges_Geom)[k] = k;
            k++;
            (*mp_CircEdges_Geom)[k] = k;
            k++;
        }
    }
    geo->addPrimitiveSet(mp_CircEdges_Geom);
    geo->setVertexArray(vec3.get());

    if (Striktionslinie && m_pSliderMenuPhi->getValue() != 0 && m_pSliderMenuPhi->getValue() != 1 && m_pSliderMenuPhi->getValue() != 2)
    {
        setMaterial_striktline(geo, red);

        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SLINES,red);
		geo->setStateSet(mp_StateSet);
		mp_StateSet->setColorAndMaterial(red,m_MaterialLine);
		mp_StateSet->setAttributeAndModes(n_Line,StateAttribute::ON);*/

        shypergeo->removeDrawables(0, 1);
        shypergeo->addDrawable(geo);
    }
    else
    {
        shypergeo->removeDrawables(0, 1);
    }
    ////pathpoints3B=path;
    //pathpoints3BHyper2=path;
}

void RuledSurfaces::StriktionslinieSattel(int m)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    int k = 0;
    Vec3f p1, p2;
    ref_ptr<osg::DrawElementsUInt> mp_LineEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINES, 2 * 2 * m);

    for (int ii = 0; ii < 2; ii++)
    {
        p1 = (osg::Vec3f(1, 0.5, 1));
        p2 = (osg::Vec3f(0, -0.5, 1));

        float abstand = 0.0003f;
        Vec3f normale1 = osg::Vec3f(0, 0, 1) * abstand;
        Vec3f normale2 = osg::Vec3f(0, 0, 1) * abstand;

        if (ii == 0)
        {
            p1 = p1 + normale1;
            p2 = p2 + normale2;
        }
        else
        {
            p1 = p1 - normale1;
            p2 = p2 - normale2;
        }

        vec3->push_back(p1);
        vec3->push_back(p2);

        (*mp_LineEdges_Geom)[k] = k;
        k++;
        (*mp_LineEdges_Geom)[k] = k;
        k++;
    }

    geo->addPrimitiveSet(mp_LineEdges_Geom);
    geo->setVertexArray(vec3.get());

    if (Striktionslinie)
    {
        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SLINES,red);
		geo->setStateSet(mp_StateSet);
		mp_StateSet->setColorAndMaterial(red,m_MaterialLine);
		mp_StateSet->setAttributeAndModes(n_Line,StateAttribute::ON);*/
        setMaterial_striktline(geo, red);

        ssattelgeo->removeDrawables(0, 1);
        ssattelgeo->addDrawable(geo);
    }
    else
    {
        ssattelgeo->removeDrawables(0, 1);
    }
}
void RuledSurfaces::StriktionslinieKonoid(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    int k = 0;
    Vec3f p1, p2;
    ref_ptr<osg::DrawElementsUInt> mp_LineEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINES, /*2**/ 2 * m);
    //for(int ii=0;ii<2;ii++){
    p1 = osg::Vec3f(1, 0, -1);
    p2 = osg::Vec3f(cos((e)*PI), 0, -1);

    /*	float abstand=0.0003f;
		Vec3f normale1 = osg::Vec3f(0,0,1)*abstand;
		Vec3f normale2 = osg::Vec3f(0,0,1)*abstand;

		if(ii==0){
			p1=p1+normale1;
			p2=p2+normale2;
		}
		else{
			p1=p1-normale1;
			p2=p2-normale2;
		}*/
    vec3->push_back(p1);
    vec3->push_back(p2);

    (*mp_LineEdges_Geom)[k] = k;
    k++;
    (*mp_LineEdges_Geom)[k] = k;
    k++;
    //}
    geo->addPrimitiveSet(mp_LineEdges_Geom);
    geo->setVertexArray(vec3.get());

    if (Striktionslinie)
    {
        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SLINES,red);
		geo->setStateSet(mp_StateSet);
		mp_StateSet->setColorAndMaterial(red,m_MaterialLine);
		mp_StateSet->setAttributeAndModes(n_Line,StateAttribute::ON);*/

        setMaterial_striktline(geo, red);
        skonoidgeo->removeDrawables(0, 1);
        skonoidgeo->addDrawable(geo);
    }
    else
    {
        skonoidgeo->removeDrawables(0, 1);
    }
}
void RuledSurfaces::StriktionslinieHelix(int m, float e, float r1, float r2, float r3, float h1, float h3)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    //m=m*e/2;
    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }
    if (r1 <= 0)
    {
        r1 = 0;
        stepr1 = 0;
    }
    if (r2 <= 0)
    {
        r2 = 0;
        stepr2 = 0;
    }
    if (r1 >= r2)
    {
        r1 = r2;
        stepr1 = stepr2;
    }
    if (r3 >= r2)
    {
        r3 = r2;
        stepr3 = stepr2;
    }

    float l = 2;
    float u_l = 0;
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int k = 0;
    Vec3f p1, p2, p3, p4, p5, p6;
    ref_ptr<osg::DrawElementsUInt> mp_LineEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINES, 2 * m);

    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v;
        v_j_l = v_j + delta_v;

        vec3->push_back(osg::Vec3f(r1 * x(u_i), r1 * y(u_i), h3 * u_i + h1));
        vec3->push_back(osg::Vec3f(r1 * x(u_i_l), r1 * y(u_i_l), h3 * u_i_l + h1));

        (*mp_LineEdges_Geom)[k] = k;
        k++;
        (*mp_LineEdges_Geom)[k] = k;
        k++;
    }

    geo->addPrimitiveSet(mp_LineEdges_Geom);
    geo->setVertexArray(vec3.get());

    if (Striktionslinie)
    {
        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SLINES,red);
		geo->setStateSet(mp_StateSet);
		mp_StateSet->setColorAndMaterial(red,m_MaterialLine);
		mp_StateSet->setAttributeAndModes(n_Line,StateAttribute::ON);*/

        setMaterial_striktline(geo, red);
        shelixgeo->removeDrawables(0, 1);
        shelixgeo->addDrawable(geo);
    }
    else
    {
        shelixgeo->removeDrawables(0, 1);
    }
}
void RuledSurfaces::GradlinieTf(int m, float e)
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    //m=m*e/2;
    if (e <= 0)
    {
        e = 1 / PI;
        stepe = 0;
    }

    float l = 2;
    float u_l = 0;
    float u_r = (float)(e * PI);
    float v_l = 0.0f;
    float v_r = l;
    float u_i, v_j, u_i_l, v_j_l;
    float delta_u = (u_r - u_l) / m;
    float delta_v = (v_r - v_l);

    int k = 0;
    Vec3f p1, p2;
    ref_ptr<osg::DrawElementsUInt> mp_LineEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINES, 2 * m);
    for (int i = 0; i < m; i++)
    {
        u_i = u_l + delta_u * i;
        u_i_l = u_i + delta_u;
        v_j = v_l + delta_v;
        v_j_l = v_j + delta_v;

        vec3->push_back(osg::Vec3f(cos(u_i), sin(u_i), u_i));
        vec3->push_back(osg::Vec3f(cos(u_i_l), sin(u_i_l), u_i_l));

        (*mp_LineEdges_Geom)[k] = k;
        k++;
        (*mp_LineEdges_Geom)[k] = k;
        k++;
    }
    geo->addPrimitiveSet(mp_LineEdges_Geom);
    geo->setVertexArray(vec3.get());

    if (Gradlinie)
    {
        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SLINES,red);
		geo->setStateSet(mp_StateSet);
		mp_StateSet->setColorAndMaterial(red,m_MaterialLine);
		mp_StateSet->setAttributeAndModes(n_Line,StateAttribute::ON);*/

        setMaterial_striktline(geo, red); //ist keine Striktionslinie

        gradliniegeo->removeDrawables(0, 1);
        gradliniegeo->addDrawable(geo);
    }
    else
    {
        gradliniegeo->removeDrawables(0, 1);
    }
}

void RuledSurfaces::TangenteTf()
{
    osg::ref_ptr<osg::Geometry> geo = new Geometry();
    osg::ref_ptr<osg::Geometry> geo2 = new Geometry();
    osg::ref_ptr<osg::Vec3Array> vec3 = new Vec3Array();
    osg::ref_ptr<osg::Vec3Array> NormalArray = new Vec3Array();

    int k = 0;
    ref_ptr<osg::DrawElementsUInt> mp_LineEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINES, 2);

    vec3->push_back(osg::Vec3f(1, -1, -1));
    vec3->push_back(osg::Vec3f(1, 1, 1));

    (*mp_LineEdges_Geom)[k] = k;
    k++;
    (*mp_LineEdges_Geom)[k] = k;
    k++;

    geo->addPrimitiveSet(mp_LineEdges_Geom);
    geo->setVertexArray(vec3.get());

    if (Tangente)
    {
        /*HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SLINES,yellow);
		geo->setStateSet(mp_StateSet);
		mp_StateSet->setColorAndMaterial(red,m_MaterialLine);
		mp_StateSet->setAttributeAndModes(n_Line,StateAttribute::ON);*/

        setMaterial_striktline(geo, yellow); //ist keine Striktionslinie

        tangentegeo->removeDrawables(0, 1);
        tangentegeo->addDrawable(geo);
    }
    else
    {
        tangentegeo->removeDrawables(0, 1);
    }
}
void RuledSurfaces::setColorGeode(osg::ref_ptr<osg::Geode> member, Vec4 color)
{
    HfT_osg_StateSet *mp_StateSet = new HfT_osg_StateSet(SSHADE, color);
    member->getDrawable(0)->setStateSet(mp_StateSet);
}
void RuledSurfaces::step1()
{
    TextureMode = false;
    m_pCheckboxMenuGitternetz->setState(false);

    stepm = 20;
    stepe = 2;

    Kegel(stepm, stepe);
    OK(stepm, stepe);
    Zylinder(stepm, stepe);
    OZ(stepm, stepe);

    if (m_presentationStep == 4)
    {
        if (greenKegel == true && greenOK == true && greenZylinder == true && greenOZ == true)
        {
            setColorGeode(kegelgeo, lime);
            setColorGeode(okgeo, gold);
            setColorGeode(zylindergeo, blue);
            setColorGeode(ozgeo, red);
        }
        else
        {
            setColorGeode(kegelgeo, black);
            setColorGeode(okgeo, black);
            setColorGeode(zylindergeo, black);
            setColorGeode(ozgeo, black);
        }
    }
    else
    {
        setColorGeode(kegelgeo, lime);
        setColorGeode(okgeo, gold);
        setColorGeode(zylindergeo, blue);
        setColorGeode(ozgeo, red);
    }

    torsengroup = new Group;
    Torsen();
    mp_GeoGroup->addChild(torsengroup);
}
void RuledSurfaces::step2()
{
    TextureMode = false;
    m_pCheckboxMenuGitternetz->setState(false);

    stepm = 20;
    stepe = 2;
    stepd = 1.5;
    stepr1 = 1;
    stepr2 = 2;
    stepr3 = -2;
    steph1 = 0;
    steph2 = 0;
    steph3 = 1;

    Oloid(stepm = 10);

    Hyper(stepm = 40, stepd, stepe);

    Konoid(stepm = 30, 1);

    Sattel(stepm = 10);

    Helix(stepm = 40, stepe = 2, stepr1, stepr2, steph1, steph2, steph3);

    if (m_presentationStep == 4)
    {
        setColorGeode(oloidgeo, black);
        setColorGeode(hypergeo, black);
        setColorGeode(konoidgeo, black);
        setColorGeode(sattelgeo, black);
        setColorGeode(helixgeo, black);
    }
    else
    {
        setColorGeode(oloidgeo, maroon /* teal*/);
        setColorGeode(hypergeo, brown);
        setColorGeode(konoidgeo, olive);
        setColorGeode(sattelgeo, navy);
        setColorGeode(helixgeo, purple);
    }

    windschiefegroup = new Group;
    Windschiefe();
    mp_GeoGroup->addChild(windschiefegroup);
}
void RuledSurfaces::Torsen()
{

    g_Kegel = kegelgeo;
    MT_Kegel = new MatrixTransform();
    Matrix mK;
    mK.makeTranslate(PI, 1.0, 0.0f);
    MT_Kegel->setMatrix(mK);
    torsengroup->addChild(MT_Kegel);
    MT_Kegel->addChild(g_Kegel);

    g_OK = okgeo;
    MT_OK = new MatrixTransform();
    Matrix mOK;
    mOK.makeTranslate(2 * PI + 0.5, -1.0, 0.0f);
    MT_OK->setMatrix(mOK);
    torsengroup->addChild(MT_OK);
    MT_OK->addChild(g_OK);

    g_Zylinder = zylindergeo;
    MT_Zylinder = new MatrixTransform();
    Matrix mZ;
    mZ.makeTranslate(0, -1.f, 0.0f);
    MT_Zylinder->setMatrix(mZ);
    torsengroup->addChild(MT_Zylinder);
    MT_Zylinder->addChild(g_Zylinder);

    g_OZ = ozgeo;
    MT_OZ = new MatrixTransform();
    Matrix mOZ;
    mOZ.makeTranslate(0.0f, 0.0f, 0.0f);
    MT_OZ->setMatrix(mOZ);
    torsengroup->addChild(MT_OZ);
    MT_OZ->addChild(g_OZ);
}
void RuledSurfaces::Windschiefe()
{

    g_Oloid = oloidgeo;
    MT_Oloid = new MatrixTransform();
    Matrix mOL;
    mOL.makeTranslate(0.f, 5.f, 1.0f);
    MT_Oloid->setMatrix(mOL);
    windschiefegroup->addChild(MT_Oloid);
    MT_Oloid->addChild(g_Oloid);

    g_Hyper = hypergeo;
    MT_Hyper = new MatrixTransform();
    Matrix mHy;
    mHy.makeTranslate(PI, 5.5f, 0.0f);
    MT_Hyper->setMatrix(mHy);
    windschiefegroup->addChild(MT_Hyper);
    MT_Hyper->addChild(g_Hyper);

    g_Sattel = sattelgeo;
    MT_Sattel = new MatrixTransform();
    Matrix mSt;
    mSt.makeTranslate(2 * PI - 1, 7.5f, 0.0f);
    MT_Sattel->setMatrix(mSt);
    windschiefegroup->addChild(MT_Sattel);
    MT_Sattel->addChild(g_Sattel);

    g_Konoid = konoidgeo;
    MT_Konoid = new MatrixTransform();
    Matrix mKo;
    mKo.makeTranslate(-0.5f, 7.5f, 0.0f);
    MT_Konoid->setMatrix(mKo);
    windschiefegroup->addChild(MT_Konoid);
    MT_Konoid->addChild(g_Konoid);

    g_Schraube = helixgeo;
    MT_Schraube = new MatrixTransform();
    Matrix mHe;
    mHe.makeTranslate(PI, 10.f, -PI);
    MT_Schraube->setMatrix(mHe);
    windschiefegroup->addChild(MT_Schraube);
    MT_Schraube->addChild(g_Schraube);
}
float RuledSurfaces::x(float u)
{
    return (float)cos(u);
}
float RuledSurfaces::y(float u)
{
    return (float)sin(u);
}
float RuledSurfaces::z(float v)
{
    return v;
}

void RuledSurfaces::changePresentationStep()
{
    int tempStepCounter = m_presentationStepCounter;

    //If the counter has a negative value add the number of presentation steps as
    //long as the counter is positive.
    if (m_presentationStepCounter < 0)
    {
        do
        {
            tempStepCounter = tempStepCounter + m_numPresentationSteps;
        } while (tempStepCounter < 0);
    }
    //Computation of the presentation step by modulo calculation
    m_presentationStep = (int)mod((double)tempStepCounter, m_numPresentationSteps);

    switch (m_presentationStep)
    {
    case 0: //Einfuehrung
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(0);
        mp_GeoGroup->removeChildren(0, 20);
        stepm = 50;
        stepd = 2.5f;
        stepe = 2;
        stepr1 = 1;
        stepr2 = 2;
        stepr3 = -2;
        steph1 = 0;
        steph2 = 0;
        steph3 = 1;
        step1();
        step2();
        VRSceneGraph::instance()->viewAll();
        break;

    case 1: //allg Regelflaechen
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(1);
        mp_GeoGroup->removeChildren(0, 20);
        stepm = 50;
        stepd = 2.5f;
        stepe = 2;
        stepr1 = 1;
        stepr2 = 2;
        stepr3 = -2;
        steph1 = 0;
        steph2 = 0;
        steph3 = 1;
        step1();
        step2();
        VRSceneGraph::instance()->viewAll();
        break;

    case 2: //Kreiskegel
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Kegel";
        setMenuVisible(2);
        mp_GeoGroup->removeChildren(0, 20);
        Kegel(stepm = 50, stepe = 2);
        mp_GeoGroup->addChild(kegelgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 3: //Kreiskegel-Tangentialebene
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Kegel";
        setMenuVisible(3);
        mp_GeoGroup->removeChildren(0, 20);
        Kegel(stepm = 50, stepe = 2);
        mp_GeoGroup->addChild(kegelgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 4: //Torsale Regelflaechen + Aufgabe
        root->setNodeMask(root->getNodeMask() | Isect::Intersection | Isect::Pick); //mit Fläche was machbar
        setMenuVisible(4);
        mp_GeoGroup->removeChildren(0, 20);
        greenKegel = false;
        greenOK = false;
        greenZylinder = false;
        greenOZ = false;
        greenOloid = false;
        greenHyper = false;
        greenSattel = false;
        greenKonoid = false;
        greenHelix = false;
        stepm = 50;
        stepd = 2.5f;
        stepe = 2;
        stepr1 = 1;
        stepr2 = 2;
        stepr3 = -2;
        steph1 = 0;
        steph2 = 0;
        steph3 = 1;
        step1();
        step2();
        interact = true;
        VRSceneGraph::instance()->viewAll();
        break;

    case 5: //OK offener Kegel
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "OK";
        setMenuVisible(5);
        mp_GeoGroup->removeChildren(0, 20);
        OK(stepm = 50, stepe = 2);
        mp_GeoGroup->addChild(okgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 6: //Zylinder
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Zylinder";
        setMenuVisible(6);
        mp_GeoGroup->removeChildren(0, 20);
        Zylinder(stepm = 50, stepe = 2);
        mp_GeoGroup->addChild(zylindergeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 7: //OZ(offener Zylinder)
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "OZ";
        setMenuVisible(7);
        mp_GeoGroup->removeChildren(0, 20);
        OZ(stepm = 50, stepe = 2);
        mp_GeoGroup->addChild(ozgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    //case 7://Tangentialflaeche
    //	root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick);//mit Fläche nichts machbar,aber Menue
    //	surface="Tangentialflaeche";
    //	setMenuVisible(7);
    //	mp_GeoGroup->removeChildren(0,20);
    //	Tangentialflaeche(stepm=50,stepe=2);
    //	mp_GeoGroup->addChild(tangentialflaechegeo);
    //	VRSceneGraph::instance()->viewAll();
    //	break;

    case 8: //windschiefe Regelflaechen
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(8);
        mp_GeoGroup->removeChildren(0, 20);
        stepm = 50;
        stepd = 3.5f;
        stepe = 2;
        stepr1 = 1;
        stepr2 = 2;
        stepr3 = -2;
        steph1 = 0;
        steph2 = 0;
        steph3 = 1;
        step2();
        VRSceneGraph::instance()->viewAll();
        break;

    case 9: //Sattelflaeche
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Sattel";
        Striktionslinie = false;
        setMenuVisible(9);
        mp_GeoGroup->removeChildren(0, 20);
        Sattel(stepm = 50);
        mp_GeoGroup->addChild(sattelgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 10: //Sattelflaeche-Strktionslinie
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Sattel";
        Striktionslinie = false;
        setMenuVisible(10);
        mp_GeoGroup->removeChildren(0, 20);
        Sattel(stepm = 50);
        mp_GeoGroup->addChild(sattelgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 11: //Oloid
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Oloid";
        setMenuVisible(11);
        mp_GeoGroup->removeChildren(0, 20);
        Oloid(stepm = 20);
        mp_GeoGroup->addChild(oloidgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 12: //Konoid
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Konoid";
        Striktionslinie = false;
        setMenuVisible(12);
        mp_GeoGroup->removeChildren(0, 20);
        Konoid(stepm = 50, 2);
        mp_GeoGroup->addChild(konoidgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 13: //Konoid-Drall
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Konoid";
        Striktionslinie = false;
        setMenuVisible(13);
        drall = true;
        mp_GeoGroup->removeChildren(0, 20);
        Konoid(stepm = 50, 2);
        mp_GeoGroup->addChild(konoidgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 14: //Hyperboloid
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Hyperboloid";
        Striktionslinie = false;
        setMenuVisible(14);
        mp_GeoGroup->removeChildren(0, 20);
        Hyper(stepm = 50, stepd = 3.5, stepe = 2.0);
        mp_GeoGroup->addChild(hypergeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 15: //Schraubflaeche
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        surface = "Helix1";
        Striktionslinie = false;
        setMenuVisible(15);
        mp_GeoGroup->removeChildren(0, 20);
        Helix(stepm = 50, stepe = 2, stepr1, stepr2 = 2.0, steph1 = 0.0, steph2 = 0.0, steph3 = 1.0);
        mp_GeoGroup->addChild(helixgeo);
        VRSceneGraph::instance()->viewAll();
        break;

    case 16: //summary
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(16);
        mp_GeoGroup->removeChildren(0, 20);
        stepm = 40;
        stepd = 3.5f;
        stepe = 2;
        stepr1 = 1;
        stepr2 = 2;
        stepr3 = -2;
        steph1 = 0;
        steph2 = 0;
        steph3 = 1;
        step1();
        step2();
        VRSceneGraph::instance()->viewAll();
        break;
    }
}
//---------------------------------------------------
//Implements RuledSurfaces::guiToRenderMsg(const grmsg::coGRMsg &msg) 
//---------------------------------------------------
void RuledSurfaces::createMenu()
{
    m_pObjectMenu1 = new coRowMenu("Menu-Flaechen");
    m_pObjectMenu1->setVisible(false);
    m_pObjectMenu1->setAttachment(coUIElement::RIGHT);

    m_pObjectMenu2 = new coRowMenu("Menu-Animation");
    m_pObjectMenu2->setVisible(false);
    m_pObjectMenu2->setAttachment(coUIElement::RIGHT);

    /* m_pObjectMenu3 = new coRowMenu("Menu-Boundary");
   m_pObjectMenu3->setVisible(false);
   m_pObjectMenu3->setAttachment(coUIElement::RIGHT);

   m_pObjectMenu4 = new coRowMenu("Menu-Boundary");
   m_pObjectMenu4->setVisible(false);
   m_pObjectMenu4->setAttachment(coUIElement::RIGHT);

   m_pObjectMenu5 = new coRowMenu("Menu-SurfaceCurves");
   m_pObjectMenu5->setVisible(false);
   m_pObjectMenu5->setAttachment(coUIElement::RIGHT);

   m_pObjectMenu6 = new coRowMenu("Menu-Animation");
   m_pObjectMenu6->setVisible(false);
   m_pObjectMenu6->setAttachment(coUIElement::RIGHT);

   m_pObjectMenu8 = new coRowMenu("Menu-Normalschnitt-Animation");
   m_pObjectMenu8->setVisible(false);
   m_pObjectMenu8->setAttachment(coUIElement::RIGHT);

   m_pObjectMenu9 = new coRowMenu("Menu-Hauptkruemmungen");
   m_pObjectMenu9->setVisible(false);
   m_pObjectMenu9->setAttachment(coUIElement::RIGHT);

   m_pObjectMenu10 = new coRowMenu("Menu-Schiefschnitt-Animation");
   m_pObjectMenu10->setVisible(false);
   m_pObjectMenu10->setAttachment(coUIElement::RIGHT);*/

    //matrices to position the menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;

    // position menu with values from config file
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", 0);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", -5);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 0);
    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.RuledSurfaces.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.RuledSurfaces.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.RuledSurfaces.MenuPosition", pz);
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("s", "COVER.Plugin.RuledSurfaces.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    m_pObjectMenu1->setTransformMatrix(&matrix);
    m_pObjectMenu1->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu2->setTransformMatrix(&matrix);
    m_pObjectMenu2->setScale(cover->getSceneSize() / 2500);

    /* m_pObjectMenu3->setTransformMatrix(&matrix);
   m_pObjectMenu3->setScale(cover->getSceneSize()/2500);

   m_pObjectMenu4->setTransformMatrix(&matrix);
   m_pObjectMenu4->setScale(cover->getSceneSize()/2500);

   m_pObjectMenu5->setTransformMatrix(&matrix);
   m_pObjectMenu5->setScale(cover->getSceneSize()/2500);

   m_pObjectMenu6->setTransformMatrix(&matrix);
   m_pObjectMenu6->setScale(cover->getSceneSize()/2500);

   m_pObjectMenu8->setTransformMatrix(&matrix);
   m_pObjectMenu8->setScale(cover->getSceneSize()/2500);

   m_pObjectMenu9->setTransformMatrix(&matrix);
   m_pObjectMenu9->setScale(cover->getSceneSize()/2500);

   m_pObjectMenu10->setTransformMatrix(&matrix);
   m_pObjectMenu10->setScale(cover->getSceneSize()/2500);*/

    m_pButtonMenuDreibein = new coButtonMenuItem(coTranslator::coTranslate("begleitendes Dreibein"));
    m_pButtonMenuDreibein->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuDreibein);

    m_pCheckboxMenuTangEbene = new coCheckboxMenuItem(coTranslator::coTranslate("Tangentialebene"), false);
    m_pCheckboxMenuTangEbene->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);

    m_pSliderMenuDetailstufe = new coSliderMenuItem(coTranslator::coTranslate("Detailstufe"), -1.0, 1.0, 0.0);
    m_pSliderMenuDetailstufe->setMenuListener(this);
    m_pSliderMenuDetailstufe->setInteger(true);
    m_pObjectMenu1->add(m_pSliderMenuDetailstufe);

    m_pSliderMenuPhi = new coSliderMenuItem(coTranslator::coTranslate("*pi = Phi veraendern"), -1.0, 1.0, 0.0);
    m_pSliderMenuPhi->setMenuListener(this);
    //m_pSliderMenuPhi->setAttachment(0);
    m_pObjectMenu1->add(m_pSliderMenuPhi);

    m_pSliderMenuErzeugung = new coSliderMenuItem(coTranslator::coTranslate("*pi = u-Parameter"), -1.0, 1.0, 0.0);
    m_pSliderMenuErzeugung->setMenuListener(this);
    m_pObjectMenu1->add(m_pSliderMenuErzeugung);

    m_pSliderMenuRadAussen = new coSliderMenuItem(coTranslator::coTranslate("Aussenradius"), -1.0, 1.0, 0.0);
    m_pSliderMenuRadAussen->setMenuListener(this);
    m_pObjectMenu1->add(m_pSliderMenuRadAussen);

    m_pSliderMenuHoeheAussen = new coSliderMenuItem(coTranslator::coTranslate("Aeussere Hoehe"), -1.0, 1.0, 0.0);
    m_pSliderMenuHoeheAussen->setMenuListener(this);
    m_pObjectMenu1->add(m_pSliderMenuHoeheAussen);

    m_pSliderMenuSchraubhoehe = new coSliderMenuItem(coTranslator::coTranslate("Schraubhoehe"), -1.0, 1.0, 0.0);
    m_pSliderMenuSchraubhoehe->setMenuListener(this);
    m_pObjectMenu1->add(m_pSliderMenuSchraubhoehe);

    m_pCheckboxMenuGitternetz = new coCheckboxMenuItem(coTranslator::coTranslate("Gitternetz"), false);
    m_pCheckboxMenuGitternetz->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);

    m_pCheckboxMenuStriktion = new coCheckboxMenuItem(coTranslator::coTranslate("Striktionslinie"), false);
    m_pCheckboxMenuStriktion->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuStriktion);

    m_pCheckboxMenuTrennung = new coCheckboxMenuItem(coTranslator::coTranslate("Flaechentrennung"), false);
    m_pCheckboxMenuTrennung->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuTrennung);

    m_pCheckboxMenuDrall = new coCheckboxMenuItem(coTranslator::coTranslate("Drall anzeigen"), false);
    m_pCheckboxMenuDrall->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuDrall);

    m_pCheckboxMenuGrad = new coCheckboxMenuItem(coTranslator::coTranslate("Grad"), false);
    m_pCheckboxMenuGrad->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuGrad);

    m_pCheckboxMenuTangente = new coCheckboxMenuItem(coTranslator::coTranslate("Tangente"), false);
    m_pCheckboxMenuTangente->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuTangente);
}
void RuledSurfaces::runden(double phi)
{
    stepd = (int)(phi * 10 + 0.5) / 10.0;
}
void RuledSurfaces::preFrame()
{
    if (m_presentationStep == 4)
    {
        if (interact == true)
        {
            if (cover->getIntersectedNode() == kegelgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenKegel == false)
                    {
                        setColorGeode(kegelgeo, green);
                        greenKegel = true;
                    }
                    else
                    {
                        setColorGeode(kegelgeo, black);
                        greenKegel = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == okgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenOK == false)
                    {
                        setColorGeode(okgeo, green);
                        greenOK = true;
                    }
                    else
                    {
                        setColorGeode(okgeo, black);
                        greenOK = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == zylindergeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenZylinder == false)
                    {
                        setColorGeode(zylindergeo, green);
                        greenZylinder = true;
                    }
                    else
                    {
                        setColorGeode(zylindergeo, black);
                        greenZylinder = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == ozgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenOZ == false)
                    {
                        setColorGeode(ozgeo, green);
                        greenOZ = true;
                    }
                    else
                    {
                        setColorGeode(ozgeo, black);
                        greenOZ = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == oloidgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenOloid == false)
                    {
                        setColorGeode(oloidgeo, green);
                        greenOloid = true;
                    }
                    else
                    {
                        setColorGeode(oloidgeo, black);
                        greenOloid = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == hypergeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenHyper == false)
                    {
                        setColorGeode(hypergeo, green);
                        greenHyper = true;
                    }
                    else
                    {
                        setColorGeode(hypergeo, black);
                        greenHyper = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == konoidgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenKonoid == false)
                    {
                        setColorGeode(konoidgeo, green);
                        greenKonoid = true;
                    }
                    else
                    {
                        setColorGeode(konoidgeo, black);
                        greenKonoid = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == sattelgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenSattel == false)
                    {
                        setColorGeode(sattelgeo, green);
                        greenSattel = true;
                    }
                    else
                    {
                        setColorGeode(sattelgeo, black);
                        greenSattel = false;
                    }
                }
            }
            else if (cover->getIntersectedNode() == helixgeo)
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                if (interactionA->wasStarted())
                {
                    if (greenHelix == false)
                    {
                        setColorGeode(helixgeo, green);
                        greenHelix = true;
                    }
                    else
                    {
                        setColorGeode(helixgeo, black);
                        greenHelix = false;
                    }
                }
            }
            else
            {
                coInteractionManager::the()->unregisterInteraction(interactionA); //kann Szene wieder drehen
            }

            if (greenKegel == true && greenOK == true && greenZylinder == true && greenOZ == true && greenOloid == false
                && greenSattel == false && greenKonoid == false && greenHyper == false && greenHelix == false)
            {
                mp_GeoGroup->removeChildren(0, 20);
                stepm = 50;
                stepd = 2.5f;
                stepe = 2;
                stepr1 = 1;
                stepr2 = 2;
                stepr3 = -2;
                steph1 = 0;
                steph2 = 0;
                steph3 = 1;
                step1();
                interact = false;
            }
        }
        else
        {
            coInteractionManager::the()->unregisterInteraction(interactionA); //kann Szene wieder drehen
        }
    }
    else
    {
        coInteractionManager::the()->unregisterInteraction(interactionA); //kann Szene wieder drehen
    }
}
void RuledSurfaces::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- RuledSurfacesPlugin coVRGuiToRenderMsg %s\n", msg.getString().c_str());

    if (msg.isValid() && msg.getType() == coGRMsg::KEYWORD)
    {
        auto &keyWordMsg = msg.as<coGRKeyWordMsg>();
        const char *keyword = keyWordMsg.getKeyWord();

        //fprintf(stderr,"\tcoGRMsg::KEYWORD keyword=%s\n", keyword);

        //Forward button pressed
        if (strcmp(keyword, "presForward") == 0)
        {
            //fprintf(stderr, "presForward\n");
            m_presentationStepCounter++;
            this->changePresentationStep();
        }
        //Backward button pressed
        else if (strcmp(keyword, "presBackward") == 0)
        {
            //fprintf(stderr, "presBackward\n");
            m_presentationStepCounter--;
            this->changePresentationStep();
        }
        else if (strncmp(keyword, "goToStep", 8) == 0)
        {
            //fprintf(stderr,"\n--- RuledSurfacesPlugin coVRGuiToRenderMsg keyword %s\n", msg);
            stringstream sstream(keyword);
            string sub;
            int step;
            sstream >> sub;
            sstream >> step;
            //fprintf(stderr, "goToStep %d\n", step);
            //fprintf(stderr,"VRMoleculeViewer::message for %s\n", chbuf );
            m_presentationStepCounter = step;
            //fprintf(stderr, "PresStepCounter %d\n", m_presentationStepCounter);
            this->changePresentationStep();
        }
        else if (strcmp(keyword, "reload") == 0)
        {
            //fprintf(stderr, "reload\n");
            VRSceneGraph::instance()->viewAll();
        }
    }
}
void RuledSurfaces::removeAddSurface(string surface_)
{
    if (surface_ == "Kegel")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(kegelgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Kegel(stepm, stepe);
        mp_GeoGroup->addChild(kegelgeo);
        if (dreibeinAnim == true)
        {
            DreibeinKegel();
        }
        if (tangEbene == true)
        {
            Tangentialebene_Kegel();
        }
    }
    if (surface_ == "OK")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(okgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        OK(stepm, stepe);
        mp_GeoGroup->addChild(okgeo);
        /*if(dreibeinAnim==true){
			DreibeinOK();
		}
		else */ if (tangEbene == true)
        {
            Tangentialebene_OK();
        }
    }
    if (surface_ == "Zylinder")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(zylindergeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Zylinder(stepm, stepe);
        mp_GeoGroup->addChild(zylindergeo);
        if (dreibeinAnim == true)
        {
            DreibeinZylinder();
        }
        if (tangEbene == true)
        {
            Tangentialebene_Zylinder();
        }
    }
    if (surface_ == "OZ")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(ozgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        OZ(stepm, stepe);
        mp_GeoGroup->addChild(ozgeo);
        /*if(dreibeinAnim==true){
			DreibeinOZ();
		}
		else */ if (tangEbene == true)
        {
            Tangentialebene_OZ();
        }
    }
    //if(surface_ == "Tangentialflaeche"){
    //	if(dreibeinAnim==true || tangEbene==true || stop==true){
    //		mp_GeoGroup->removeChildren(0,20);
    //	}
    //	int index=root->getChildIndex(tangentialflaechegeo) - 1;
    //	mp_GeoGroup->removeChildren(index,index);
    //	Tangentialflaeche(stepm,stepe);
    //	mp_GeoGroup->addChild(tangentialflaechegeo);
    //	int index2=root->getChildIndex(gradliniegeo) - 1;
    //	mp_GeoGroup->removeChildren(index2,index2);
    //	GradlinieTf(stepm,stepe);
    //	mp_GeoGroup->addChild(gradliniegeo);
    //	int index3=root->getChildIndex(tangentegeo) - 1;
    //	mp_GeoGroup->removeChildren(index3,index3);
    //	TangenteTf();
    //	mp_GeoGroup->addChild(tangentegeo);
    //	/*if(dreibeinAnim==true){
    //		DreibeinOZ();
    //	}
    //	else if(tangEbene==true){
    //		Tangentialebene_OZ();
    //	}*/
    //}
    if (surface_ == "Oloid")
    {
        int index = root->getChildIndex(oloidgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Oloid(stepm);
        mp_GeoGroup->addChild(oloidgeo);
    }
    if (surface_ == "Hyperboloid")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(hypergeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Hyper(stepm, stepd, stepe);
        mp_GeoGroup->addChild(hypergeo);
        int index2 = root->getChildIndex(shypergeo) - 1;
        mp_GeoGroup->removeChildren(index2, index2);
        StriktionslinieHyper(stepm, stepd, stepe);
        mp_GeoGroup->addChild(shypergeo);
        if (dreibeinAnim == true)
        {
            DreibeinHyper();
        }
        if (tangEbene == true)
        {
            Tangentialebene_Hyper();
        }
    }
    if (surface_ == "Sattel")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(sattelgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Sattel(stepm);
        mp_GeoGroup->addChild(sattelgeo);
        int index2 = root->getChildIndex(ssattelgeo) - 1;
        mp_GeoGroup->removeChildren(index2, index2);
        StriktionslinieSattel(stepm);
        mp_GeoGroup->addChild(ssattelgeo);
        if (dreibeinAnim == true)
        {
            DreibeinSattel();
        }
        if (tangEbene == true)
        {
            Tangentialebene_Sattel();
        }
    }
    if (surface_ == "Konoid")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(konoidgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Konoid(stepm, stepe);
        mp_GeoGroup->addChild(konoidgeo);
        int index2 = root->getChildIndex(skonoidgeo) - 1;
        mp_GeoGroup->removeChildren(index2, index2);
        StriktionslinieKonoid(stepm, stepe);
        mp_GeoGroup->addChild(skonoidgeo);
        if (dreibeinAnim == true)
        {
            DreibeinKonoid();
        }
        if (tangEbene == true)
        {
            Tangentialebene1();
            Tangentialebene2();
            Tangentialebene3();
            Tangentialebene4();
            Tangentialebene5();
        }
    }
    if (surface_ == "Helix1")
    {
        if (dreibeinAnim == true || tangEbene == true || stop == true)
        {
            mp_GeoGroup->removeChildren(0, 40);
        }
        int index = root->getChildIndex(helixgeo) - 1;
        mp_GeoGroup->removeChildren(index, index);
        Helix(stepm, stepe, stepr1, stepr2, steph1, steph2, steph3);
        mp_GeoGroup->addChild(helixgeo);
        int index2 = root->getChildIndex(shelixgeo) - 1;
        mp_GeoGroup->removeChildren(index2, index2);
        StriktionslinieHelix(stepm, stepe, stepr1, stepr2, stepr3, steph1, steph3);
        mp_GeoGroup->addChild(shelixgeo);
        if (dreibeinAnim == true)
        {
            DreibeinHelix();
        }
        if (tangEbene == true)
        {
            Tangentialebene_Helix();
        }
    }
}
void RuledSurfaces::menuEvent(coMenuItem *iMenuItem)
{
    if (iMenuItem == m_pButtonMenuDreibein)
    {
        dreibeinAnim = true;
        removeAddSurface(surface);
        dreibeinAnim = false;
    }
    if (iMenuItem == m_pCheckboxMenuTangEbene)
    {
        if (m_pCheckboxMenuTangEbene->getState() == true)
        {
            tangEbene = true;
            removeAddSurface(surface);
        }
        else
        {
            stop = true;
            tangEbene = false;
            removeAddSurface(surface);
        }
    }
    if (iMenuItem == m_pSliderMenuDetailstufe)
    {
        stop = true;
        stepm = m_pSliderMenuDetailstufe->getValue();
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pSliderMenuPhi) //Drehung
    {
        stop = true;
        stepd = m_pSliderMenuPhi->getValue() + 2.0;
        if (stepd < 3.05 && stepd >= 2.95)
        {
            stepd = 3.0001;
        }
        if (stepd < 2.05)
        {
            stepd = 2.0;
        }
        if (stepd >= 3.95)
        {
            stepd = 4.0;
        }
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pSliderMenuErzeugung) //=u
    {
        stop = true;
        stepe = m_pSliderMenuErzeugung->getValue();
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pSliderMenuRadAussen)
    {
        stop = true;
        stepr2 = m_pSliderMenuRadAussen->getValue();
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pSliderMenuHoeheAussen)
    {
        stop = true;
        steph2 = m_pSliderMenuHoeheAussen->getValue();
        if (steph2 < 0.05 && steph2 >= -0.05)
        {
            steph2 = 0.0;
        }
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pSliderMenuSchraubhoehe)
    {
        stop = true;
        steph3 = m_pSliderMenuSchraubhoehe->getValue();
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pCheckboxMenuGitternetz)
    {
        if (drall == true && m_pCheckboxMenuGitternetz->getState() == true)
        {
            m_pCheckboxMenuDrall->setState(false);
        }
        else if (drall == true && m_pCheckboxMenuGitternetz->getState() == false)
        {
            m_pCheckboxMenuDrall->setState(true);
        }
        stop = false;
        TextureMode = !TextureMode;
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pCheckboxMenuStriktion)
    {
        //erstellt osg, kann dann mit C:\EXTERNLIBS\OpenSceneGraph-3.0.1\bin\osgconv
        //zu (.obj) oder .stl(besser) konvertiert werden fuer Blender
        /*const string imagPath = "C:\\Temp/ruledsurface2.osg";

	   Node* knoten = mp_GeoGroup->getChild(0);
	   osgDB::writeNodeFile(*knoten,imagPath);*/
        stop = true;
        Striktionslinie = !Striktionslinie;
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pCheckboxMenuTrennung)
    { //Oloid trennen
        Trennung = !Trennung;
        removeAddSurface("Oloid");
    }
    if (iMenuItem == m_pCheckboxMenuDrall)
    {
        drall = !drall;

        if (m_pCheckboxMenuGitternetz->getState() == true && m_pCheckboxMenuDrall->getState() == false)
        {
        }
        else
        {
            stop = false;
            TextureMode = false;
            m_pCheckboxMenuGitternetz->setState(false);
            removeAddSurface(surface);
        }
    }
    if (iMenuItem == m_pCheckboxMenuGrad)
    {
        stop = false;
        Gradlinie = !Gradlinie;
        removeAddSurface(surface);
    }
    if (iMenuItem == m_pCheckboxMenuTangente)
    {
        stop = false;
        Tangente = !Tangente;
        removeAddSurface(surface);
    }
}

void RuledSurfaces::setMenuVisible(int step)
{
    m_pObjectMenu1->removeAll();
    m_pCheckboxMenuStriktion->setState(false);
    m_pCheckboxMenuTangEbene->setState(false);
    tangEbene = false;
    m_pCheckboxMenuGrad->setState(false);
    m_pCheckboxMenuTangente->setState(false);
    m_pCheckboxMenuDrall->setState(false);
    drall = false;

    if (surface == "Tangentialflaeche")
        m_pSliderMenuDetailstufe->setMin(20);
    else
        m_pSliderMenuDetailstufe->setMin(3);
    m_pSliderMenuDetailstufe->setMax(100);
    m_pSliderMenuDetailstufe->setPrecision(0);
    m_pSliderMenuDetailstufe->setValue(50);
    m_sliderValueDetailstufe = m_pSliderMenuDetailstufe->getValue();

    m_pSliderMenuPhi->setMin(0.0);
    m_pSliderMenuPhi->setMax(2.0);
    m_pSliderMenuPhi->setPrecision(1);
    m_pSliderMenuPhi->setValue(1.5);
    m_sliderValuePhi = m_pSliderMenuPhi->getValue();

    m_pSliderMenuErzeugung->setMin(0.1);
    if (surface == "Konoid")
    {
        m_pSliderMenuErzeugung->setMax(1.0);
    }
    else if (surface == "Helix1")
    {
        m_pSliderMenuErzeugung->setMax(5.0);
    }
    else
    {
        m_pSliderMenuErzeugung->setMax(2.0);
    }
    m_pSliderMenuErzeugung->setPrecision(1);
    if (surface == "Konoid")
    {
        m_pSliderMenuErzeugung->setValue(1.0);
    }
    else
    {
        m_pSliderMenuErzeugung->setValue(2.0);
    }
    m_sliderValueErzeugung = m_pSliderMenuErzeugung->getValue();

    m_pSliderMenuRadAussen->setMin(0.5);
    m_pSliderMenuRadAussen->setMax(5.0);
    m_pSliderMenuRadAussen->setPrecision(1);
    m_pSliderMenuRadAussen->setValue(2.0);
    m_sliderValueRadAussen = m_pSliderMenuRadAussen->getValue();

    m_pSliderMenuHoeheAussen->setMin(-3.0);
    m_pSliderMenuHoeheAussen->setMax(3.0);
    m_pSliderMenuHoeheAussen->setPrecision(1);
    m_pSliderMenuHoeheAussen->setValue(0.0);
    m_sliderValueHoeheAussen = m_pSliderMenuHoeheAussen->getValue();

    m_pSliderMenuSchraubhoehe->setMin(-5.0);
    m_pSliderMenuSchraubhoehe->setMax(5.0);
    m_pSliderMenuSchraubhoehe->setPrecision(1);
    m_pSliderMenuSchraubhoehe->setValue(1.0);
    m_sliderValueSchraubhoehe = m_pSliderMenuSchraubhoehe->getValue();

    if (step == 2)
    { //Kreiskegel
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pSliderMenuErzeugung);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        //m_pObjectMenu1->add(m_pButtonMenuDreibein);
    }
    if (step == 3)
    { //Kreiskegel-Tangentialebene
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    }
    if (step == 5 || step == 6 || step == 7)
    {
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pSliderMenuErzeugung);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        //m_pObjectMenu1->add(m_pButtonMenuDreibein);
        m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    }
    //if(step==7){//Tangentialflaeche=Schraubtorse
    //	m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
    //	m_pObjectMenu1->add(m_pSliderMenuErzeugung);
    //	m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
    //	m_pObjectMenu1->add(m_pCheckboxMenuGrad);
    //	m_pObjectMenu1->add(m_pCheckboxMenuTangente);
    //	m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    //}
    if (step == 9)
    { //Sattel
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        //m_pObjectMenu1->add(m_pButtonMenuDreibein);
        m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    }
    if (step == 10)
    { //Sattel-Striktionslinie
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        m_pObjectMenu1->add(m_pCheckboxMenuStriktion);
        //m_pObjectMenu1->add(m_pButtonMenuDreibein);
    }
    if (step == 11)
    { //oloidaehnliche flaeche
        m_pObjectMenu1->add(m_pCheckboxMenuTrennung);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
    }
    if (step == 12)
    { //Konoid
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pSliderMenuErzeugung);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        m_pObjectMenu1->add(m_pCheckboxMenuStriktion);
        m_pObjectMenu1->add(m_pCheckboxMenuDrall);
        m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    }
    if (step == 13)
    { //Konoid-begl dreibein
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        m_pObjectMenu1->add(m_pButtonMenuDreibein);
    }
    if (step == 14)
    { //Hyper
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pSliderMenuErzeugung);
        m_pObjectMenu1->add(m_pSliderMenuPhi);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        m_pObjectMenu1->add(m_pCheckboxMenuStriktion);
        m_pObjectMenu1->add(m_pCheckboxMenuDrall);
        m_pObjectMenu1->add(m_pButtonMenuDreibein);
        m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    }

    if (step == 15)
    {
        m_pObjectMenu1->add(m_pSliderMenuDetailstufe);
        m_pObjectMenu1->add(m_pSliderMenuErzeugung);
        /*if(surface=="Helix2"){
			m_pObjectMenu1->add(m_pSliderMenuRadInnen);
		}*/
        m_pObjectMenu1->add(m_pSliderMenuRadAussen);
        //m_pObjectMenu1->add(m_pSliderMenuHoeheInnen);
        m_pObjectMenu1->add(m_pSliderMenuHoeheAussen);
        m_pObjectMenu1->add(m_pSliderMenuSchraubhoehe);
        m_pObjectMenu1->add(m_pCheckboxMenuGitternetz);
        m_pObjectMenu1->add(m_pCheckboxMenuStriktion);
        m_pObjectMenu1->add(m_pButtonMenuDreibein);
        m_pObjectMenu1->add(m_pCheckboxMenuTangEbene);
    }
    m_pObjectMenu1->setVisible(true);

    VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
}

COVERPLUGIN(RuledSurfaces)
