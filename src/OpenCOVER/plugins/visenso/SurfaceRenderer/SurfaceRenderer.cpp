/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: Plugin class for the renderer                             **
 **                                                                        **
 ** Original cpp file                                                      **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** Extended cpp file                                                      **
 ** Author: J.Wolz	                                                   **
 ** History:                                                               **
 **     01.2011 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "SurfaceRenderer.h"
#include "cover/VRSceneGraph.h"
#include <Python.h>
#include <iostream>

#ifndef WIN32
#include <dlfcn.h>
#endif

using namespace osg;
using namespace std;
using namespace vrui;
using namespace opencover;

//---------------------------------------------------
//Implements SurfaceRenderer *plugin variable
//---------------------------------------------------
SurfaceRenderer *SurfaceRenderer::plugin = NULL;
//---------------------------------------------------
//Implements SurfaceRenderer::SurfaceRenderer()
//---------------------------------------------------
SurfaceRenderer::SurfaceRenderer()
: coVRPlugin(COVER_PLUGIN_NAME)
, GenericGuiObject("SurfaceRenderer")
, m_rpRootSwitchNode(NULL)
, m_rpTransMatrixNode(NULL)
,
//m_pPlaneSurface(NULL),
//m_Plane(NULL),
//m_Surface(NULL),
m_Sphere(NULL)
,
// SphereSurface not available m_pSphereSurface(NULL),
m_Strip(NULL)
, m_pMobiusStrip(NULL)
,
//m_Wendel(NULL),
//m_pWendelSurface(NULL),
//m_Zylinder(NULL),
//m_pZylinderSurface(NULL),
//m_Kegel(NULL),
//m_pKegelSurface(NULL),
//m_Dini(NULL),
//m_Torus(NULL),
//m_Sattel(NULL),
//m_Boloid(NULL),
//m_Flasche(NULL),
m_CreatedCurve(NULL)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n SurfaceRenderer::SurfaceRenderer \n");

/* Linking of Python Library
     * for using of Math Dictionary
    */
#ifndef WIN32
    dlopen("libpython2.5.so", RTLD_LAZY | RTLD_GLOBAL);
#endif
    /* Python Script initialization
     * Import from Math Module
     * Get attribute from Math Module
     */
    Py_Initialize();

    r = "2*(1 - cos(paramV)/2)";

    //   m_rpRootSwitchNode = new Switch;
}

//---------------------------------------------------
//Implements SurfaceRenderer::~SurfaceRenderer()
//---------------------------------------------------
SurfaceRenderer::~SurfaceRenderer()
{
    /*if(m_pSurface != NULL){
      delete m_pSurface;
      delete gui_showMenu;
      delete gui_stringxRow;
      delete gui_stringyRow;
      delete gui_stringzRow;
      delete gui_stringNormalXRow;
      delete gui_stringNormalYRow;
      delete gui_stringNormalZRow;
      delete gui_surfaceMode;
      delete gui_lowBoundU;
      delete gui_upBoundU;
      delete gui_lowBoundV;
      delete gui_upBoundV;
      delete gui_patU;
      delete gui_patV;
   }
   if(m_Sphere!= NULL){
       delete m_Sphere;
   }
   if(m_Strip!= NULL){
         delete m_Strip;
     }
     if(m_Sattel!= NULL){
         delete m_Sattel;
     }
   if(m_Wendel!= NULL){
         delete m_Wendel;
     }
   if(m_Zylinder!= NULL){
         delete m_Zylinder;
     }
   if(m_Boloid!= NULL){
         delete m_Boloid;
     }
   if(m_Dini!= NULL){
         delete m_Dini;
     }
   if(m_CreatedCurve!= NULL){
         delete m_CreatedCurve;
     }
   if(m_Kegel!= NULL){
         delete m_Kegel;
     }
   if(m_Torus!= NULL){
         delete m_Torus;
     }
   if(m_Plane!= NULL){
	 delete m_Plane;
   }*/

    //Removes the root node from the scene graph
    cover->getObjectsRoot()->removeChild(m_rpRootSwitchNode.get());
    cover->getObjectsRoot()->removeChild(m_rpTransMatrixNode.get());
}

//---------------------------------------------------
//Implements SurfaceRenderer::init()
//---------------------------------------------------
bool SurfaceRenderer::init()
{

    if (plugin)
        return false;
    //Set plugin
    SurfaceRenderer::plugin = this;

    // SphereSurface not availablem_pSphereSurface = new SphereSurface();

    //Initialization of the switch node as root
    m_rpRootSwitchNode = new Switch();
    m_rpRootSwitchNode->ref();
    m_rpRootSwitchNode->setName("SwitchNodeForSurfaces");
    std::cout << "1111111 m_rpRootSwitchNode = " << m_rpRootSwitchNode << std::endl;
    std::cout << "m_rpRootSwitchNode->referenceCount():  " << m_rpRootSwitchNode->referenceCount() << std::endl;

    m_rpTransMatrixNode = new MatrixTransform();
    m_rpTransMatrixNode->setName("MatrixTransformNodeForSurface");
    std::cout << "1111111 m_rpTransMatrixNode = " << m_rpTransMatrixNode << std::endl;

    //Add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(m_rpRootSwitchNode.get());
    cover->getObjectsRoot()->addChild(m_rpTransMatrixNode.get());

    this->initializeSurfaces();

    //Create Tree
    m_rpRootSwitchNode->insertChild(0, m_Sphere->m_rpTransformNode.get(), false);
    m_rpRootSwitchNode->insertChild(1, m_Strip->m_rpTransformNode.get(), false);
    /*  m_rpRootSwitchNode -> insertChild(2, m_Sattel->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(3, m_Torus->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(4, m_Flasche->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(5, m_Boloid->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(6, m_Dini->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(7, m_Plane->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(8, m_Kegel->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(9, m_Zylinder->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(10, m_Wendel->m_rpTransformNode.get(),false);
   m_rpRootSwitchNode -> insertChild(11, m_Surface->m_rpTransformNode.get(),false);
  */ //default visible surface
    m_rpRootSwitchNode->setValue(0, true);

    createMenu();

    //------------------------VR-Prepare---------------------------------//
    //vr-prepare (add param after menu was created)
    gui_showMenu = addGuiParamBool("Mathematics Menu", false);
    gui_creationMode = addGuiParamBool("Creation Mode", false);

    //zoom scene
    //attention: never use if the scene is empty!!
    VRSceneGraph::instance()->viewAll();

    return true;
}

//---------------------------------------------------
//Implements SurfaceRenderer::initializeSurfaces()
//---------------------------------------------------
void SurfaceRenderer::initializeSurfaces()
{

    /* //Classic Sphere
    m_Surface = new ParamSurface( 100, 100, -PI, PI, -PI/2, PI/2, 2,  "cos(paramU) * cos(paramV)",
   								       "sin(paramU) * cos(paramV)",
   								       "sin(paramV)",
   								       "cos(paramU) * (cos(paramV) * cos(paramV))",
   								       "sin(paramU) * (cos(paramV) * cos(paramV))",
   								       "sin(paramV) * cos(paramV)");

    m_Surface ->createSurface();
    m_Surface ->setImage(0);*/
    //Sphere with Texture
    m_Sphere = new ParamSurface(100, 100, -PI, PI, -PI / 2, PI / 2, 3, "cos(paramU) * cos(paramV)",
                                "sin(paramU) * cos(paramV)",
                                "sin(paramV)",
                                "cos(paramU) * (cos(paramV) * cos(paramV))",
                                "sin(paramU) * (cos(paramV) * cos(paramV))",
                                "sin(paramV) * cos(paramV)");
    m_Sphere->setImage(1);
    m_Sphere->createSurface();
    //Mobius Strip
    m_Strip = new ParamSurface(100, 100, 0.0, 2 * PI, -1.0, 1.0, 3, "cos(paramU)*( 1 + (paramV/2) * cos(paramU/2))",
                               "sin(paramU)*( 1 + (paramV/2)* cos(paramU/2))",
                               "(paramV/2) * sin(paramU/2)",
                               "sin(paramU/2) * cos(paramU) * ( 1 + paramV * cos(paramU/2)) * 1/2 * paramV * sin(paramU)",
                               "sin(paramU/2) * sin(paramU) * ( 1 + paramV * cos(paramU/2)) * 1/2 * paramV * cos(paramU)",
                               "-(cos(paramU/2)) * ( 1 + paramV * cos(paramU/2))");
    m_Strip->setImage(2);
    m_Strip->createSurface();
    /* //Helikoid or WendelSurface
   m_Wendel = new ParamSurface( 100, 100, -PI, PI, -PI, PI, 3,
   								       "paramU * cos(paramV)",
   	               						       "paramU * sin(paramV)",
   	               						       "1 * paramV",
   	               						       "paramU * 1 * paramV",
   	               						       "paramU * 1 * paramV",
   	               						       "paramU * sin(paramV) - paramU * cos(paramV)");
   m_Wendel ->setImage(0);
   m_Wendel ->createSurface();

   //Zylinder
   m_Zylinder = new ParamSurface(100, 100, 0, 2*PI, -1, 1, 1, 	       "cos(paramU)",
   								       "paramV",
   								       "sin(paramU)",
   							               "-sin(paramU) * paramV",
   								       "0",
   								       "cos(paramU)* paramV");
   m_Zylinder ->setImage(0);
   m_Zylinder ->createSurface();

   //Dini Surface
   m_Dini = new ParamSurface(100, 100, 0, 4*PI, 0.1, 2, 3,            "2*(cos(paramU)*sin(paramV))",
								       "2*(sin(paramU)*sin(paramV))",
								       "2*((cos(paramV)+log(tan(paramV/2))) + 0.2  *paramU)",
								       "cos(paramV) + log(tan(paramV/2)) * sin(paramV) - sin(paramU) * paramU",
								       "cos(paramU) * paramU - cos(paramV) + log(tan(paramV/2)) * sin(paramV)",
								       "sin(paramU) * sin(paramV) - cos(paramU) * sin(paramV)");

   m_Dini ->setImage(0);
   m_Dini ->createSurface();
   //Torus
   m_Torus = new ParamSurface(100, 100, 0, 2*PI, 0, 2*PI, 3,          "cos(paramU) * ( 3 + cos(paramV))",
   								       "sin(paramU) * (3 + cos(paramV))",
   								       "sin(paramV)",
   								       "-sin(paramV) * (3 + cos(paramV))",
   								       "sin(paramV) * (3 + cos(paramV))",
   								       "cos(paramU) * (3 + cos(paramV)) - sin(paramU) * (3 + cos(paramV))");

   m_Torus ->setImage(0);
   m_Torus ->createSurface();
   //Sattelfläche
   m_Sattel = new ParamSurface(100, 100, -0.5, 0.5, -0.5, 0.5, 3,     "paramU",
   								       "paramV",
   								       "paramV*paramU",
   								       "-(paramU*paramV)",
   								       "-(paramU*paramV)",
   								       "paramU * paramV");
   m_Sattel ->setImage(0);
   m_Sattel ->createSurface();
   //Paraboloid
   m_Boloid = new ParamSurface(100, 100, -PI, PI, 0, 2, 3,	      "cos(paramU) * paramV",
 								      "sin(paramU) * paramV",
 								      "paramV * paramV",
 								      "sin(paramU) * (paramV * paramV)",
 								      "- (paramV * paramV)",
 								      "cos(paramU) * paramV - sin(paramU) * paramV");
   m_Boloid ->setImage(0);
   m_Boloid ->createSurface();
   //Kegel
   m_Kegel = new ParamSurface(100, 100, -PI, PI, 0.0, 2, 3, 	      "(0.5/2 * paramV) * cos(paramU)",
 								      "(0.5/2 * paramV) * sin(paramU)",
 								      "paramV",
 								      "sin(paramU) * paramV",
 								      "cos(paramU) * paramV",
 								      "(cos(paramU) * (0.5/2* paramV)) - sin(paramU) * 0.5/2 * paramV");
   m_Kegel ->setImage(0);
   m_Kegel ->createSurface();
   //Klein'sche Flasche
   m_Flasche = new ParamSurface(100, 100, 0, 2*PI, 0, 2*PI, 1,       "(3*cos(paramV) * (1 + sin(paramV))" + r + "*cos(paramV) * cos(paramU)",
								      "(8*sin(paramV)" + r +"* sin(paramV) * cos(paramU))",
								      "(sin(paramU))",
								      "0",
								      "0",
								      "0");
   m_Flasche = new ParamSurface(100, 100, 0, 2*PI, 0, 2*PI, 1,       "(3*cos(paramV) * (1 + sin(paramV)) +" +r+" * cos(paramU+3.14))",
								      "(8*sin(paramV))",
								      "(sin(paramU) *" + r +")",
								      "0",
								      "0",
								      "0");
   m_Flasche ->setImage(0);
   m_Flasche ->createSurface();
   //Ebene
   m_Plane = new ParamSurface(100, 100, -1, 1, -1, 1, 2,              "paramU",
								      "0",
      							              "paramV",
      							              "paramU",
      							              "(paramU * paramV)",
      							              "paramV");
   m_Plane ->setImage(0);
   m_Plane ->createSurface();*/
}
//---------------------------------------------------
//Implements SurfaceRenderer::createMenu()
//---------------------------------------------------
void SurfaceRenderer::createMenu()
{

    menuItemMenu = new coRowMenu("Auswahl der Flaeche");
    menuItemMenu->setVisible(false);
    // position Menu
    /*  OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;

    //position the menu
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ReadCollada.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ReadCollada.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ReadCollada.MenuPosition", pz);

    // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
    float s = coCoviseConfig::getFloat("value","COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("value","COVER.Plugin.ElectricField.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0,90,0);
    scaleMatrix.makeScale(s,s,s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    menuItemMenu->setTransformMatrix(&matrix);
    menuItemMenu->setScale(cover->getSceneSize()/2500);

    menuItemAddSphere = new coButtonMenuItem("Sphaere");
    menuItemAddSphere->setMenuListener(this);
    menuItemAddPlane = new coButtonMenuItem("Ebene");
    menuItemAddPlane->setMenuListener(this);
    menuItemAddKegel = new coButtonMenuItem("Kegel");
    menuItemAddKegel->setMenuListener(this);
    menuItemAddZylinder = new coButtonMenuItem("Zylinder");
    menuItemAddZylinder->setMenuListener(this);
    menuItemAddTorus = new coButtonMenuItem("Torus");
    menuItemAddTorus->setMenuListener(this);
    menuItemAddLine = new coButtonMenuItem("____________________");
    menuItemAddStrip = new coButtonMenuItem("Moebiusband");
    menuItemAddStrip->setMenuListener(this);
    menuItemAddWendel = new coButtonMenuItem("Minimalflaeche");
    menuItemAddWendel->setMenuListener(this);
    menuItemAddSattel = new coButtonMenuItem("Sattelflaeche");
    menuItemAddSattel->setMenuListener(this);
    menuItemAddBoloid = new coButtonMenuItem("Paraboloid");
    menuItemAddBoloid->setMenuListener(this);
    menuItemAddFlasche = new coButtonMenuItem("Klein'sche Flasche");
    menuItemAddFlasche->setMenuListener(this);
    menuItemAddDini = new coButtonMenuItem("Dini Surface");
    menuItemAddDini->setMenuListener(this);
	menuItemMenu->add(menuItemAddSphere);
	menuItemMenu->add(menuItemAddPlane);
	menuItemMenu->add(menuItemAddKegel);
	menuItemMenu->add(menuItemAddZylinder);
	menuItemMenu->add(menuItemAddTorus);
	menuItemMenu->add(menuItemAddLine);
	menuItemMenu->add(menuItemAddStrip);
	menuItemMenu->add(menuItemAddWendel);
	menuItemMenu->add(menuItemAddSattel);
	menuItemMenu->add(menuItemAddBoloid);
	menuItemMenu->add(menuItemAddFlasche);
	menuItemMenu->add(menuItemAddDini);
*/
}

//---------------------------------------------------/
//Implements SurfaceRenderer::preFrame()
//---------------------------------------------------/
void SurfaceRenderer::preFrame()
{
}

//--------------------------------------------
//Implements SurfaceRenderer::setVisible(char _visible)
//--------------------------------------------
void SurfaceRenderer::setVisible(char visible)
{
    std::cout << "Visible:  " << visible << std::endl;
    std::cout << "2222222 m_rpTransMatrixNode = " << m_rpTransMatrixNode << std::endl;
    std::cout << "dazwischen" << std::endl;
    std::cout << "2222222 m_rpRootSwitchNode = " << m_rpRootSwitchNode << std::endl;
    std::cout << "m_rpRootSwitchNode->referenceCount():  " << m_rpRootSwitchNode->referenceCount() << std::endl;
    switch (visible)
    {
    case 'A':
        m_rpRootSwitchNode->setValue(0, true);
        m_rpRootSwitchNode->setValue(1, false);
        /* m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);*/
        break;

    case 'S':
        std::cout << "CaseS::m_rpRootSwitchNode->referenceCount()" << m_rpRootSwitchNode->referenceCount() << std::endl;
        m_rpRootSwitchNode->setValue(0, true);
        /*m_rpRootSwitchNode -> setValue(2, false);
	        	    m_rpRootSwitchNode -> setValue(3, false);
	        	    m_rpRootSwitchNode -> setValue(4, false);
	        	    m_rpRootSwitchNode -> setValue(5, false);
	                m_rpRootSwitchNode -> setValue(6, false);
	                m_rpRootSwitchNode -> setValue(7, false);
	                m_rpRootSwitchNode -> setValue(8, false);
	                m_rpRootSwitchNode -> setValue(9, false);
	                m_rpRootSwitchNode -> setValue(10, false);
	                m_rpRootSwitchNode -> setValue(11, false);*/
        break;

        /*case 'C':
	    std::cout << "setVisible::case 2 " << std::endl;
	    m_rpRootSwitchNode -> setValue(0, false);
	    m_rpRootSwitchNode -> setValue(1, false);
	    m_rpRootSwitchNode -> setValue(2, true);
	    m_rpRootSwitchNode -> setValue(3, false);
	    m_rpRootSwitchNode -> setValue(4, false);
	    m_rpRootSwitchNode -> setValue(5, false);
	    m_rpRootSwitchNode -> setValue(6, false);
	    m_rpRootSwitchNode -> setValue(7, false);
	    m_rpRootSwitchNode -> setValue(8, false);
	    m_rpRootSwitchNode -> setValue(9, false);
	    m_rpRootSwitchNode -> setValue(10, false);
	    m_rpRootSwitchNode -> setValue(11, false);
	    break;
    case 3:
    	    std::cout << "setVisible::case 3 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, true);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;
    case 4:
    	    std::cout << "setVisible::case 4 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, true);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;
    case 5:
    	    std::cout << "setVisible::case 5 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, true);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;
    case 6:
    	    std::cout << "setVisible::case 6 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, true);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;
    case 7:
    	    std::cout << "setVisible::case 7 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, true);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;

    case 8:
	    std::cout << "setVisible::case 8  " << std::endl;
	    m_rpRootSwitchNode -> setValue(0, false);
	    m_rpRootSwitchNode -> setValue(1, false);
	    m_rpRootSwitchNode -> setValue(2, false);
	    m_rpRootSwitchNode -> setValue(3, false);
	    m_rpRootSwitchNode -> setValue(4, false);
	    m_rpRootSwitchNode -> setValue(5, false);
	    m_rpRootSwitchNode -> setValue(6, false);
	    m_rpRootSwitchNode -> setValue(7, false);
	    m_rpRootSwitchNode -> setValue(8, true);
	    m_rpRootSwitchNode -> setValue(9, false);
	    m_rpRootSwitchNode -> setValue(10, false);
	    m_rpRootSwitchNode -> setValue(11, false);
	    break;
    case 9:
    	    std::cout << "setVisible::case 9 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, true);
    	    m_rpRootSwitchNode -> setValue(10, false);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;
    case 10:
    	    std::cout << "setVisible::case 10 " << std::endl;
    	    m_rpRootSwitchNode -> setValue(0, false);
    	    m_rpRootSwitchNode -> setValue(1, false);
    	    m_rpRootSwitchNode -> setValue(2, false);
    	    m_rpRootSwitchNode -> setValue(3, false);
    	    m_rpRootSwitchNode -> setValue(4, false);
    	    m_rpRootSwitchNode -> setValue(5, false);
    	    m_rpRootSwitchNode -> setValue(6, false);
    	    m_rpRootSwitchNode -> setValue(7, false);
    	    m_rpRootSwitchNode -> setValue(8, false);
    	    m_rpRootSwitchNode -> setValue(9, false);
    	    m_rpRootSwitchNode -> setValue(10, true);
    	    m_rpRootSwitchNode -> setValue(11, false);
    	    break;*/
    }
}
//---------------------------------------------------
//Implements SurfaceRenderer::guiToRenderMsg(const grmsg::coGRMsg &msg) 
//---------------------------------------------------
void SurfaceRenderer::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{

    GenericGuiObject::guiToRenderMsg(msg);
    fprintf(stderr, "SurfaceRenderer::guiToRenderMsg:  %s\n", msg.getString().c_str());
    // SphereSurface not available m_pSphereSurface->guiToRenderMsg(msg);
}
//--------------------------------------------
//Implements SurfaceRenderer::setCharge(float charge)
//----------------------------------------------
void SurfaceRenderer::setCharge(float charge)
{
    this->charge = charge;
    switch (gui_charge_)
    {
    case 1:
        gui_lowBoundU->setValue(charge);
        break;
    case 2:
        gui_upBoundU->setValue(charge);
        break;
    case 3:
        gui_lowBoundV->setValue(charge);
        break;
    case 4:
        gui_upBoundV->setValue(charge);
        break;
    }
}

//--------------------------------------------
//Implements SurfaceRenderer::setCharge(float charge)
//----------------------------------------------
void SurfaceRenderer::setCharge(int charge)
{
    this->charge_ = charge;
    std::cout << "charge: " << charge << std::endl;
    std::cout << "charge_:  " << charge_ << std::endl;
    switch (gui_charge_)
    {
    case 1:
        gui_surfaceMode->setValue(charge);
        break;
    }
}

//---------------------------------------------
//Implements SurfaceRenderer::setStringX(std::string string_x)
//---------------------------------------------
void SurfaceRenderer::setString(std::string string_)
{

    this->_string = string_;
    switch (gui_string)
    {
    case 1:
        gui_stringxRow->setValue(string_);
        break;
    case 2:
        gui_stringyRow->setValue(string_);
        break;
    case 3:
        gui_stringzRow->setValue(string_);
        break;
    case 4:
        gui_stringNormalXRow->setValue(string_);
        break;
    case 5:
        gui_stringNormalYRow->setValue(string_);
        break;
    case 6:
        gui_stringNormalZRow->setValue(string_);
        break;
    }
}
//---------------------------------------------------
//Implements SurfaceRenderer::menuEvent(coMenuItem* menuItem)
//---------------------------------------------------
void SurfaceRenderer::menuEvent(coMenuItem *menuItem)
{

    if (menuItem == menuItemAddSphere)
    {
        if (m_rpRootSwitchNode->getValue(0))
        {
            std::cout << "addSphere()" << std::endl;
            //addSphere();
        }
    }
    if (menuItem == menuItemAddStrip)
    {
        if (m_rpRootSwitchNode->getValue(1) == true && gui_creationMode->getValue() == 1)
        {
            std::cout << "addMobius_Strip()" << std::endl;
            //addMobius_Strip();
        }
    }
    /*  if (menuItem == menuItemAddWendel){
	if (m_rpRootSwitchNode->getValue(10) == true && gui_creationMode->getValue() == 1){
	    addWendel_Surface();
	}
    }
    if (menuItem == menuItemAddZylinder){
	if (m_rpRootSwitchNode->getValue(9) == true && gui_creationMode->getValue() == 1){
	    addZylinder();
	}
    }
    if (menuItem == menuItemAddDini){
	if (m_rpRootSwitchNode->getValue(6) == true && gui_creationMode->getValue() == 1){
	    addDini_Surface();
	}
    }
    if (menuItem == menuItemAddTorus){
	if (m_rpRootSwitchNode->getValue(3) == true && gui_creationMode->getValue() == 1){
	    addTorus();
	}
    }
    if (menuItem == menuItemAddSattel){
	if (m_rpRootSwitchNode->getValue(2) == true && gui_creationMode->getValue() == 1){
	    addSattelflaeche();
	}
    }
    if (menuItem == menuItemAddBoloid){
	if (m_rpRootSwitchNode->getValue(5) == true && gui_creationMode->getValue() == 1){
	    addBoloid();
	}
    }
    if (menuItem == menuItemAddKegel){
	if (m_rpRootSwitchNode->getValue(8) == true && gui_creationMode->getValue() == 1){
	    addKegel();
	}
    }
    if (menuItem == menuItemAddFlasche){
	if (m_rpRootSwitchNode->getValue(4) == true && gui_creationMode->getValue() == 1){
	    addFlasche();
	}
    }*/
}
//---------------------------------------------------
//Implements SurfaceRenderer::guiParamChanged(GuiParam* guiParam);
//---------------------------------------------------
void SurfaceRenderer::guiParamChanged(GuiParam *guiParam)
{

    if (guiParam == gui_showMenu)
    {
        menuItemMenu->setVisible(gui_showMenu->getValue());
        m_rpRootSwitchNode->setValue(0, false);
    }
    if (guiParam == gui_creationMode)
    {
        if (gui_creationMode->getValue() == 1)
        {
            std::cout << "gui_creationMode->getValue():  " << gui_creationMode->getValue() << std::endl;
            gui_stringxRow = addGuiParamString("X-Row", "x_row");
            gui_stringyRow = addGuiParamString("Y-Row", "y_row");
            gui_stringzRow = addGuiParamString("Z-Row", "z_row");
            gui_stringNormalXRow = addGuiParamString("Normal X-Row", "mx_row");
            gui_stringNormalYRow = addGuiParamString("Normal Y-Row", "my_row");
            gui_stringNormalZRow = addGuiParamString("Normal Z-Row", "mz_row");
            gui_surfaceMode = addGuiParamInt("Mode Surface", 0);
            gui_lowBoundU = addGuiParamFloat("Lower Bound ParamU", 0);
            gui_upBoundU = addGuiParamFloat("Upper Bound ParamU", 0);
            gui_lowBoundV = addGuiParamFloat("Lower Bound ParamV", 0);
            gui_upBoundV = addGuiParamFloat("Upper Bound ParamV", 0);
            gui_patU = addGuiParamInt("Patch U", 100);
            gui_patV = addGuiParamInt("Patch V", 100);
        }
    }

    if (guiParam == gui_lowBoundU)
    {
        gui_charge_ = 1;
        setCharge(gui_lowBoundU->getValue());
        getModifications();
    }
    if (guiParam == gui_upBoundU)
    {
        gui_charge_ = 2;
        setCharge(gui_upBoundU->getValue());
        getModifications();
    }
    if (guiParam == gui_lowBoundV)
    {
        gui_charge_ = 3;
        setCharge(gui_lowBoundV->getValue());
        getModifications();
    }
    if (guiParam == gui_upBoundV)
    {
        gui_charge_ = 4;
        setCharge(gui_upBoundV->getValue());
        getModifications();
    }
    if (guiParam == gui_surfaceMode)
    {
        gui_charge_ = 1;
        setCharge(gui_surfaceMode->getValue());
        getModifications();
    }
    if (guiParam == gui_stringxRow)
    {
        gui_string = 1;
        setString(gui_stringxRow->getValue());
        getModifications();
    }
    if (guiParam == gui_stringyRow)
    {
        gui_string = 2;
        setString(gui_stringyRow->getValue());
        getModifications();
    }
    if (guiParam == gui_stringzRow)
    {
        gui_string = 3;
        setString(gui_stringzRow->getValue());
        getModifications();
    }
    if (guiParam == gui_stringNormalXRow)
    {
        gui_string = 4;
        setString(gui_stringNormalXRow->getValue());
        getModifications();
    }
    if (guiParam == gui_stringNormalYRow)
    {
        gui_string = 5;
        setString(gui_stringNormalYRow->getValue());
        getModifications();
    }
    if (guiParam == gui_stringNormalZRow)
    {
        gui_string = 6;
        setString(gui_stringNormalZRow->getValue());
        getModifications();
    }
}

/*//---------------------------------------------------/
//Implements SurfaceRenderer::addSphere()
//---------------------------------------------------/
void SurfaceRenderer::addSphere(){

        gui_stringxRow->setValue("cos(paramU) * cos(paramV)");
	gui_stringyRow->setValue("sin(paramU) * cos(paramV)");
	gui_stringzRow->setValue("sin(paramV)");
	gui_stringNormalXRow->setValue("cos(paramU) * (cos(paramV) * cos(paramV))");
	gui_stringNormalYRow->setValue("sin(paramU) * (cos(paramV) * cos(paramV))");
	gui_stringNormalZRow->setValue("sin(paramV) * cos(paramV)");
	gui_lowBoundU->setValue(-0.5);
	gui_upBoundU->setValue(0.5);
	gui_lowBoundV->setValue(-0.5);
	gui_upBoundV->setValue(0.5);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(2);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addMobius_Strip()
//---------------------------------------------------
void SurfaceRenderer::addMobius_Strip(){
	gui_stringxRow->setValue("cos(paramU)*( 1 + (paramV/2) * cos(paramU/2))");
	gui_stringyRow->setValue("sin(paramU)*( 1 + (paramV/2)* cos(paramU/2))");
	gui_stringzRow->setValue("(paramV/2) * sin(paramU/2)");
	gui_stringNormalXRow->setValue("sin(paramU/2) * cos(paramU) * ( 1 + paramV * cos(paramU/2)) * 1/2 * paramV * sin(paramU)");
	gui_stringNormalYRow->setValue("sin(paramU/2) * sin(paramU) * ( 1 + paramV * cos(paramU/2)) * 1/2 * paramV * cos(paramU)");
	gui_stringNormalZRow->setValue("-(cos(paramU/2)) * ( 1 + paramV * cos(paramU/2))");
	gui_lowBoundU->setValue(0.0);
	gui_upBoundU->setValue(2*3.14);
	gui_lowBoundV->setValue(-1.0);
	gui_upBoundV->setValue(1.0);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(3);
}

//---------------------------------------------------
//Implements SurfaceRenderer::addEnepeer_Surface()
//---------------------------------------------------
void SurfaceRenderer::addWendel_Surface(){
	gui_stringxRow->setValue("paramU * cos(paramV)");
	gui_stringyRow->setValue("paramU * sin(paramV)");
	gui_stringzRow->setValue("1 * paramV");
	gui_stringNormalXRow->setValue("paramU * 1 * paramV");
	gui_stringNormalYRow->setValue("paramU * 1 * paramV");
	gui_stringNormalZRow->setValue("paramU * sin(paramV) - paramU * cos(paramV)");
	gui_lowBoundU->setValue(-3.14);
	gui_upBoundU->setValue(3.14);
	gui_lowBoundV->setValue(-3.14);
	gui_upBoundV->setValue(3.14);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(3);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addZylinder()
//---------------------------------------------------
void SurfaceRenderer::addZylinder(){
	gui_stringxRow->setValue("cos(paramU)");
	gui_stringyRow->setValue("paramV");
	gui_stringzRow->setValue("sin(paramU)");
	gui_stringNormalXRow->setValue("-sin(paramU) * paramV");
	gui_stringNormalYRow->setValue("0");
	gui_stringNormalZRow->setValue("cos(paramU)* paramV");
	gui_lowBoundU->setValue(0);
	gui_upBoundU->setValue(2*3.14);
	gui_lowBoundV->setValue(-1.0);
	gui_upBoundV->setValue(1.0);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(1);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addDini_Surface()
//---------------------------------------------------
void SurfaceRenderer::addDini_Surface(){
	gui_stringxRow->setValue("cos(paramU)*sin(paramV)");
	gui_stringyRow->setValue("cos(paramV)+log(tan(paramV/2)) + paramU");
	gui_stringzRow->setValue("sin(paramU)*sin(paramV)");
	gui_stringNormalXRow->setValue("cos(paramV) + log(tan(paramV/2)) * sin(paramV) - sin(paramU) * paramU");
	gui_stringNormalYRow->setValue("sin(paramU) * sin(paramV) - cos(paramU) * sin(paramV)");
	gui_stringNormalZRow->setValue("cos(paramU) * paramU - cos(paramV) + log(tan(paramV/2)) * sin(paramV)");
	gui_lowBoundU->setValue(0);
	gui_upBoundU->setValue(2*3.14);
	gui_lowBoundV->setValue(-1.0);
	gui_upBoundV->setValue(1.0);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(3);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addTorus()
//---------------------------------------------------
void SurfaceRenderer::addTorus(){
	gui_stringxRow->setValue("cos(paramU) * ( 3 + cos(paramV))");
	gui_stringyRow->setValue("sin(paramU) * (3 + cos(paramV))");
	gui_stringzRow->setValue("sin(paramV)");
	gui_stringNormalXRow->setValue("-sin(paramV) * (3 + cos(paramV))");
	gui_stringNormalYRow->setValue("sin(paramV) * (3 + cos(paramV))");
	gui_stringNormalZRow->setValue("cos(paramU) * (3 + cos(paramV)) - sin(paramU) * (3 + cos(paramV))");
	gui_lowBoundU->setValue(0);
	gui_upBoundU->setValue(2*3.14);
	gui_lowBoundV->setValue(0);
	gui_upBoundV->setValue(2*3.14);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(3);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addSattelflaeche()
//---------------------------------------------------
void SurfaceRenderer::addSattelflaeche(){
        gui_stringxRow->setValue("paramU");
        gui_stringyRow->setValue("paramV");
        gui_stringzRow->setValue("paramV*paramU");
        gui_stringNormalXRow->setValue("-(paramU*paramV)");
        gui_stringNormalYRow->setValue("-(paramU*paramV)");
        gui_stringNormalZRow->setValue("paramU * paramV");
        gui_lowBoundU->setValue(-0.5);
        gui_upBoundU->setValue(0.5);
        gui_lowBoundV->setValue(-0.5);
        gui_upBoundV->setValue(0.5);
        gui_patU->setValue(100);
        gui_patV->setValue(100);
        gui_surfaceMode->setValue(3);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addBoloid();
//---------------------------------------------------
void SurfaceRenderer::addBoloid(){
        gui_stringxRow->setValue("cos(paramU) * paramV");
        gui_stringyRow->setValue("sin(paramU) * paramV");
        gui_stringzRow->setValue("paramV * paramV");
        gui_stringNormalXRow->setValue("sin(paramU) * (paramV * paramV)");
        gui_stringNormalYRow->setValue("- (paramV * paramV)");
        gui_stringNormalZRow->setValue("cos(paramU) * paramV - sin(paramU) * paramV");
        gui_lowBoundU->setValue(-3.14);
        gui_upBoundU->setValue(3.14);
        gui_lowBoundV->setValue(0);
        gui_upBoundV->setValue(2);
        gui_patU->setValue(100);
        gui_patV->setValue(100);
        gui_surfaceMode->setValue(3);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addBoloid();
//---------------------------------------------------
void SurfaceRenderer::addKegel(){
        gui_stringxRow->setValue("(0.5/2 * paramV) * cos(paramU)");
    	gui_stringyRow->setValue("(0.5/2 * paramV) * sin(paramU)");
    	gui_stringzRow->setValue("paramV");
    	gui_stringNormalXRow->setValue("sin(paramU) * paramV");
    	gui_stringNormalYRow->setValue("cos(paramU) * paramV");
    	gui_stringNormalZRow->setValue("(cos(paramU) * (0.5/2* paramV)) - sin(paramU) * 0.5/2 * paramV");
    	gui_lowBoundU->setValue(-3.14);
    	gui_upBoundU->setValue(3.14);
    	gui_lowBoundV->setValue(0);
    	gui_upBoundV->setValue(2);
    	gui_patU->setValue(100);
    	gui_patV->setValue(100);
    	gui_surfaceMode->setValue(3);
}
//---------------------------------------------------
//Implements SurfaceRenderer::addBoloid();
//---------------------------------------------------
void SurfaceRenderer::addFlasche(){
        gui_stringxRow->setValue("3 * cos(paramV) * (1 + sin(paramV))" + r + "*cos(paramV) * cos(paramU)");
        gui_stringyRow->setValue("8 * sin(paramV)" +r +"* sin(paramV) * cos(paramU)");
        gui_stringzRow->setValue("sin(paramU)");
        //In this Case, these are not the normals but the second three equations//
        gui_stringNormalXRow->setValue("3 * cos(paramV) * (1 + sin(paramV)) +" +r+" * cos(paramU+3.14)");
        gui_stringNormalYRow->setValue("8 * sin(paramV)");
	gui_stringNormalZRow->setValue("sin(paramU) *" +r+"");
	gui_lowBoundU->setValue(0);
	gui_upBoundU->setValue(2 * 3.14);
	gui_lowBoundV->setValue(0);
	gui_upBoundV->setValue(2 * 3.14);
	gui_patU->setValue(100);
	gui_patV->setValue(100);
	gui_surfaceMode->setValue(1);
}
*/

//---------------------------------------------------
//Implements SurfaceRenderer::getModifications()
//---------------------------------------------------
void SurfaceRenderer::getModifications()
{
    //---------------------Values-in-Variablen-speichern-----------------------//
    p_x_row = gui_stringxRow->getValue();
    p_y_row = gui_stringyRow->getValue();
    p_z_row = gui_stringzRow->getValue();
    p_mx_row = gui_stringNormalXRow->getValue();
    p_my_row = gui_stringNormalYRow->getValue();
    p_mz_row = gui_stringNormalZRow->getValue();
    p_surface_mode = gui_surfaceMode->getValue();
    p_lowbound_u = gui_lowBoundU->getValue();
    p_upbound_u = gui_upBoundU->getValue();
    p_lowbound_v = gui_lowBoundV->getValue();
    p_upbound_v = gui_upBoundV->getValue();
    recreationSurface();
}

//--------------------------------------------------
//Implements SurfaceRenderer::recreationSurface()
//--------------------------------------------------
void SurfaceRenderer::recreationSurface()
{

    m_CreatedCurve = new ParamSurface(100, 100,
                                      p_lowbound_u, p_upbound_u, p_lowbound_v,
                                      p_upbound_v, p_surface_mode, p_x_row,
                                      p_y_row, p_z_row, p_mx_row,
                                      p_my_row, p_mz_row);

    m_CreatedCurve->createSurface();
    m_CreatedCurve->setImage(0);

    m_rpTransMatrixNode->addChild(m_CreatedCurve->m_rpGeode.get());

    m_rpRootSwitchNode->setValue(0, false);
    m_rpRootSwitchNode->setValue(1, false);
    /*m_rpRootSwitchNode -> setValue(2, false);
    m_rpRootSwitchNode -> setValue(3, false);
    m_rpRootSwitchNode -> setValue(4, false);
    m_rpRootSwitchNode -> setValue(5, false);
    m_rpRootSwitchNode -> setValue(6, false);
    m_rpRootSwitchNode -> setValue(7, false);
    m_rpRootSwitchNode -> setValue(8, false);
    m_rpRootSwitchNode -> setValue(9, false);
    m_rpRootSwitchNode -> setValue(10, false);
    m_rpRootSwitchNode -> setValue(11, false);*/
}

COVERPLUGIN(SurfaceRenderer)
