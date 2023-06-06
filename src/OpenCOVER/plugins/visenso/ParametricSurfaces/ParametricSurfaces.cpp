/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <cover/RenderObject.h>
#include "cover/VRSceneGraph.h"
#include "net/message.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>
#include <osg/Switch>

#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <img/coImage.h>
#include <osg/Shape>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <vrb/client/VRBClient.h>
#include <grmsg/coGRSendCurrentDocMsg.h>
#include <grmsg/coGRSetDocPageSizeMsg.h>
#include <grmsg/coGRSetDocPositionMsg.h>
#include <grmsg/coGRSetDocScaleMsg.h>
#include <math.h>
#include <iostream>

#include <osg/Vec3>
#include <osg/Geode>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osg/PositionAttitudeTransform>
#include <osgGA/TrackballManipulator>
#include <OpenVRUI/coMovableBackgroundMenuItem.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <OpenVRUI/coNavInteraction.h>
#include <vector>
//
//Plugin für parametrisierte Flächen
#include <osgDB/ReaderWriter>
#include <osgDB/WriteFile>
#include <osgDB/ReaderWriter>
#include <osgDB/ReadFile>
#include <osgDB/fstream>

#include "ParametricSurfaces.h"
#include "HfT_osg_Plugin01_Animation.h"
#include "HfT_osg_Plugin01_ReadSurfDescription.h"
#include "HfT_string.h"
#include "HfT_osg_FindNode.h"

#include "cover/coTranslator.h"

using namespace covise;
using namespace grmsg;

//---------------------------------------------------
//Implements ParametricSurfaces *plugin variable
//---------------------------------------------------
ParametricSurfaces *ParametricSurfaces::plugin = NULL;
//
ParametricSurfaces::ParametricSurfaces()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //Wurzelknoten von allen
    root = new osg::Group();
    m_presentationStepCounter = 0; //current
    m_numPresentationSteps = 0; //gesamt
    m_presentationStep = 0; //current
    m_sliderValueU = 0.0; //current u-slider value
    m_sliderValueV = 0.0;
    actualizeSlider = false;
    sliderStart = true;
    actualizeSliderStep8 = false;
    sliderStartValueA = 0.0;
    sliderStartValueB = 0.0;
    sliderStartValueC = 0.0;
    m_sliderValueA = 0.0;
    m_sliderValueB = 0.0;
    m_sliderValueC = 0.0;
    m_sliderValueA_ = 0.0;
    m_sliderValueB_ = 0.0;
    m_sliderValueC_ = 0.0;
    m_sliderValueUa = 0.0;
    m_sliderValueUe = 0.0;
    m_sliderValueVa = 0.0;
    m_sliderValueVe = 0.0;
    m_sliderValueLOD = 0; //current Level of Detail value
    checkboxPlaneState = true;
    lastStepPlane = 0;
    constyp = CNOTHING;

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
    //Slider
    m_pSliderMenuU = NULL; //Pointer to the slider for the u parameter
    m_pSliderMenuV = NULL;
    m_pSliderMenuA = NULL;
    m_pSliderMenuB = NULL;
    m_pSliderMenuC = NULL;
    m_pSliderMenuA_ = NULL;
    m_pSliderMenuB_ = NULL;
    m_pSliderMenuC_ = NULL;
    m_pSliderMenuUa = NULL;
    m_pSliderMenuUe = NULL;
    m_pSliderMenuVa = NULL;
    m_pSliderMenuVe = NULL;
    m_pSliderMenuLOD = NULL;
    //Surfmode
    m_pButtonMenuDarst = NULL;
    m_pButtonMenuPoints = NULL;
    m_pButtonMenuLines = NULL;
    m_pButtonMenuTriangles = NULL;
    m_pButtonMenuQuads = NULL;
    m_pButtonMenuShade = NULL;
    m_pButtonMenuTransparent = NULL;
    m_pButtonMenuPoints_ = NULL;
    m_pButtonMenuLines_ = NULL;
    m_pButtonMenuTriangles_ = NULL;
    m_pButtonMenuQuads_ = NULL;
    m_pButtonMenuShade_ = NULL;
    m_pButtonMenuTransparent_ = NULL;
    m_pCheckboxMenuGauss = NULL;
    m_pCheckboxMenuMean = NULL;
    //Surface
    m_pButtonMenuSurface = NULL;
    m_pButtonMenuBonan = NULL;
    m_pButtonMenuBoy = NULL;
    m_pButtonMenuCrossCap = NULL;
    m_pButtonMenuDini = NULL;
    m_pButtonMenuEnneper = NULL;
    m_pButtonMenuHelicalTorus = NULL;
    m_pButtonMenuKatenoid = NULL;
    m_pButtonMenuKlein = NULL;
    m_pButtonMenuKuen = NULL;
    m_pButtonMenuKugel = NULL;
    m_pButtonMenuMoebius = NULL;
    m_pButtonMenuPluecker = NULL;
    // m_pButtonMenuPseudoSphere = NULL;
    m_pButtonMenuKegel = NULL;
    m_pButtonMenuParaboloid = NULL;
    m_pButtonMenuZylinder = NULL;
    m_pButtonMenuRevolution = NULL;
    m_pButtonMenuRoman = NULL;
    m_pButtonMenuShell = NULL;
    m_pButtonMenuSnake = NULL;
    m_pButtonMenuTrumpet = NULL;
    m_pButtonMenuTwistedSphere = NULL;
    //Cons
    m_pButtonMenuCons = NULL;
    m_pButtonMenuNothing = NULL;
    m_pButtonMenuUCenter = NULL;
    m_pButtonMenuVCenter = NULL;
    m_pButtonMenuDiagonal = NULL;
    m_pButtonMenuTriangle = NULL;
    m_pButtonMenuEllipse = NULL;
    m_pButtonMenuSquare = NULL;
    m_pButtonMenuNatbound = NULL;
    //Textur
    m_pButtonMenuTextur1 = NULL;
    m_pButtonMenuTextur2 = NULL;
    m_pButtonMenuTextur1_ = NULL;
    m_pButtonMenuTextur2_ = NULL;
    //Animation
    m_pButtonMenuAnimationSphere = NULL;
    m_pButtonMenuAnimationOff = NULL;
    //Normalschnittanimation...
    m_pButtonMenuNormalschnitt = NULL;
    m_pCheckboxMenuHauptKrRichtungen = NULL;
    m_pCheckboxMenuHauptKr = NULL;
    m_pCheckboxMenuSchmiegTang = NULL;
    m_pButtonMenuSchiefschnitt = NULL;
    m_pCheckboxMenuDarstMeusnierKugel = NULL;
    m_pButtonMenuTransparent = NULL;
    m_pButtonMenuClearNormalschnitt = NULL;
    //Parameterebene ein/aus
    m_pCheckboxMenuPlane = NULL;
    m_pCheckboxMenuPlane2 = NULL;
    m_pCheckboxMenuPlane3 = NULL;
    //Flaechenrand ein/aus
    m_pCheckboxMenuNatbound = NULL;
    //Flaechennormalen ein/aus
    m_pCheckboxMenuNormals = NULL;

    // Fr. Uebele
    HfT_osg_Plugin01_NormalschnittAnimation *na = new HfT_osg_Plugin01_NormalschnittAnimation();
    mp_NormalschnittAnimation = na;
    // Fr Uebele
    m_maxcreateAnimation = 6; // Anzahl der Animationen
    m_iscreateAnimation = 0; //mom. Anzahl der Animationen(Kugeln)
    m_Imagelistcounter = 1; //fuer Textur
    m_natbound = false; // für ursprüngliche Parametergrenzen
    m_Pua = 0.; //Parameterebene u,v
    m_Pue = 0.;
    m_Pva = 0.;
    m_Pve = 0.;
    m_Surfname = "Kugel";
    m_ColorAnimation = new Vec4f[m_maxcreateAnimation]; // Farbe der Kugeln
    m_ColorAnimation[0] = Vec4(1.f, 1.f, 0.f, 1.f); //gelb
    m_ColorAnimation[1] = Vec4(0.f, 1.f, 0.f, 1.f); //grün
    m_ColorAnimation[2] = Vec4(1.f, 0.f, 0.f, 1.f); //rot
    m_ColorAnimation[3] = Vec4(0.f, 0.f, 1.f, 1.f); //blau
    m_ColorAnimation[4] = Vec4(0.f, 1.f, 1.f, 1.f); //cyan
    m_ColorAnimation[5] = Vec4(1.f, 1.f, 1.f, 1.f); //weiß

    m_sliderValueU_old = m_sliderValueU;
    m_sliderValueV_old = m_sliderValueV;
}
ParametricSurfaces::~ParametricSurfaces()
{
    m_presentationStepCounter = 0;
    m_numPresentationSteps = 0;
    m_presentationStep = 0;
    m_sliderValueU = 0.0;
    m_sliderValueV = 0.0;
    actualizeSlider = false;
    sliderStart = true;
    actualizeSliderStep8 = false;
    sliderStartValueA = 0.0;
    sliderStartValueB = 0.0;
    sliderStartValueC = 0.0;
    m_sliderValueA = 0.0;
    m_sliderValueB = 0.0;
    m_sliderValueC = 0.0;
    m_sliderValueA_ = 0.0;
    m_sliderValueB_ = 0.0;
    m_sliderValueC_ = 0.0;
    m_sliderValueUa = 0.0;
    m_sliderValueUe = 0.0;
    m_sliderValueVa = 0.0;
    m_sliderValueVe = 0.0;
    m_sliderValueLOD = 0;
    m_sliderValueU_old = 0.0;
    m_sliderValueV_old = 0.0;
    m_sliderValueLOD_old = mp_Plane->getPatchesU();
    checkboxPlaneState = true;
    lastStepPlane = 0;
    constyp = CNOTHING;

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
    if (m_pSliderMenuU != NULL)
    {
        //Cleanup the slider menu using the pointer,
        //which points at the slider menu object.
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuU;
    }

    if (m_pSliderMenuV != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuV;
    }
    if (m_pSliderMenuA != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuA;
    }
    if (m_pSliderMenuB != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuB;
    }
    if (m_pSliderMenuC != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuC;
    }
    if (m_pSliderMenuA_ != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuA_;
    }
    if (m_pSliderMenuB_ != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuB_;
    }
    if (m_pSliderMenuC_ != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuC_;
    }
    if (m_pSliderMenuUa != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuUa;
    }
    if (m_pSliderMenuUe != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuUe;
    }
    if (m_pSliderMenuVa != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuVa;
    }
    if (m_pSliderMenuVe != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuVe;
    }
    if (m_pSliderMenuLOD != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuLOD;
    }
    if (m_pButtonMenuDarst != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuDarst;
    }
    if (m_pButtonMenuPoints != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuPoints;
    }
    if (m_pButtonMenuLines != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuLines;
    }
    if (m_pButtonMenuTriangles != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTriangles;
    }
    if (m_pButtonMenuQuads != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuQuads;
    }
    if (m_pButtonMenuShade != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuShade;
    }
    if (m_pButtonMenuPoints_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuPoints_;
    }
    if (m_pButtonMenuLines_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuLines_;
    }
    if (m_pButtonMenuTriangles_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTriangles_;
    }
    if (m_pButtonMenuQuads_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuQuads_;
    }
    if (m_pButtonMenuShade_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuShade_;
    }
    if (m_pButtonMenuTransparent != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTransparent;
    }
    if (m_pButtonMenuTransparent_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTransparent_;
    }
    if (m_pCheckboxMenuGauss != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuGauss;
    }
    if (m_pCheckboxMenuMean != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuMean;
    }
    if (m_pButtonMenuSurface != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuSurface;
    }
    if (m_pButtonMenuBonan != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuBonan;
    }
    if (m_pButtonMenuBoy != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuBoy;
    }
    if (m_pButtonMenuCrossCap != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuCrossCap;
    }
    if (m_pButtonMenuDini != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuDini;
    }
    if (m_pButtonMenuEnneper != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuEnneper;
    }
    if (m_pButtonMenuHelicalTorus != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuHelicalTorus;
    }
    if (m_pButtonMenuKatenoid != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuKatenoid;
    }
    if (m_pButtonMenuKlein != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuKlein;
    }
    if (m_pButtonMenuKuen != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuKuen;
    }
    if (m_pButtonMenuKugel != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuKugel;
    }
    if (m_pButtonMenuMoebius != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuMoebius;
    }
    if (m_pButtonMenuPluecker != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuPluecker;
    }
    if (m_pButtonMenuKegel != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuKegel;
    }
    if (m_pButtonMenuParaboloid != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuParaboloid;
    }
    if (m_pButtonMenuZylinder != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuZylinder;
    }
    if (m_pButtonMenuRevolution != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuRevolution;
    }
    if (m_pButtonMenuRoman != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuRoman;
    }
    if (m_pButtonMenuShell != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuShell;
    }
    if (m_pButtonMenuSnake != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuSnake;
    }
    if (m_pButtonMenuTrumpet != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTrumpet;
    }
    if (m_pButtonMenuTwistedSphere != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTwistedSphere;
    }
    //Cons
    if (m_pButtonMenuCons != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuCons;
    }
    if (m_pButtonMenuNothing != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuNothing;
    }
    if (m_pButtonMenuUCenter != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuUCenter;
    }
    if (m_pButtonMenuVCenter != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuVCenter;
    }
    if (m_pButtonMenuDiagonal != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuDiagonal;
    }
    if (m_pButtonMenuTriangle != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTriangle;
    }
    if (m_pButtonMenuEllipse != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuEllipse;
    }
    if (m_pButtonMenuSquare != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuSquare;
    }
    if (m_pButtonMenuNatbound != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuNatbound;
    }
    if (m_pButtonMenuTextur1 != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTextur1;
    }
    if (m_pButtonMenuTextur2 != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTextur2;
    }
    if (m_pButtonMenuTextur1_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTextur1_;
    }
    if (m_pButtonMenuTextur2_ != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTextur2_;
    }
    if (m_pButtonMenuAnimationSphere != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuAnimationSphere;
    }
    if (m_pButtonMenuAnimationOff != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuAnimationOff;
    }
    //Normalschnittanimation...
    if (m_pButtonMenuNormalschnitt != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuNormalschnitt;
    }
    if (m_pCheckboxMenuHauptKrRichtungen != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuHauptKrRichtungen;
    }
    if (m_pCheckboxMenuHauptKr != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuHauptKr;
    }
    if (m_pCheckboxMenuSchmiegTang != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuSchmiegTang;
    }
    if (m_pButtonMenuSchiefschnitt != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuSchiefschnitt;
    }
    if (m_pCheckboxMenuDarstMeusnierKugel != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pCheckboxMenuDarstMeusnierKugel;
    }
    if (m_pButtonMenuTransparent != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuTransparent;
    }
    if (m_pButtonMenuClearNormalschnitt != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuClearNormalschnitt;
    }
    //
    if (m_pCheckboxMenuPlane != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuPlane;
    }
    if (m_pCheckboxMenuPlane2 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuPlane2;
    }
    if (m_pCheckboxMenuPlane3 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuPlane3;
    }
    if (m_pCheckboxMenuNatbound != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuNatbound;
    }
    if (m_pCheckboxMenuNormals != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuNormals;
    }
    if (labelKr1 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete labelKr1;
    }
    if (labelKr2 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete labelKr2;
    }
    //Removes the root node from the scene graph
    cover->getObjectsRoot()->removeChild(root.get());
}
Node *ParametricSurfaces::Get_Node_byName(const std::string searchName)
{
    HfT_osg_FindNode *findGeode = new HfT_osg_FindNode(searchName, false);
    root->accept(*findGeode);
    Geode *geode = dynamic_cast<Geode *>(findGeode->getFirstNode());
    return geode;
}
bool ParametricSurfaces::init() //wird von OpenCover automatisch aufgerufen //initializeMembers()
{
    if (plugin)
        return false;
    //Set plugin
    ParametricSurfaces::plugin = this;

    changesize = false;

    //Sets the possible number of presentation steps
    m_numPresentationSteps = 14;
    //Sets the surface´s formula
    m_formula = m_numPresentationSteps + 2; //-->16.png

    string menuname = "Formel";
    imageMenu_ = new coRowMenu(menuname.c_str(), NULL, 0, false);

    createMenu();

    mp_TextureImage = 0L;

    const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("value", "COVER.Plugin.ParametricSurfaces.DataPath") + "/Data/";
    // Pfade zur Flächenbeschreibung und den Texturen

    string surffiledir = m_imagePath + "Surfaces/";
    string surffilepath = surffiledir + "Kugel.txt";
    string imagefile = "Images/"; //TexturImage
    string imagefilepath = m_imagePath + imagefile;
    // Flächengruppe erzeugen

    mp_GeoGroup = InitSurfgroup(surffilepath, imagefilepath);

    // Unter Plugin = Gruppe (Teilgraph) hängen
    root->addChild(mp_GeoGroup);
    // Gleich Fläche und Parametebene ermitteln
    mp_Surf = dynamic_cast<HfT_osg_Plugin01_ParametricSurface *>(Get_Node_byName("Flaeche"));
    mp_Plane = dynamic_cast<HfT_osg_Plugin01_ParametricPlane *>(Get_Node_byName("Parameterebene"));

    interactionA = new coNavInteraction(coInteraction::ButtonA, "Selection", coInteraction::NavigationHigh);

    labelKr1 = new coVRLabel("", 20, 0.04 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.2f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    labelKr2 = new coVRLabel("", 20, 0.04 * cover->getSceneSize(), osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    labelKr1->hide();
    labelKr2->hide();

    cover->getObjectsRoot()->addChild(root.get());

    //zoom scene
    //attention: never use if the scene is empty!!
    VRSceneGraph::instance()->viewAll();

    std::cerr << "Plugin laedt am Ende" << std::endl;
    return true;
}

//Wechsel der Darstellungsweise(Textur, Transparent,...)
void ParametricSurfaces::Change_Mode(HfT_osg_Plugin01_ParametricSurface *surf, int increase)
{
    surf->recomputeMode((SurfMode)increase);
}
bool ParametricSurfaces::Gauss_Curvature_Mode(HfT_osg_Plugin01_ParametricSurface *surf)
{
    surf->recomputeMode(SGAUSS);

    return true;
}
bool ParametricSurfaces::Mean_Curvature_Mode(HfT_osg_Plugin01_ParametricSurface *surf)
{
    surf->recomputeMode(SMEAN);

    return true;
}
//nur zum Level_of_Deatail(LOD) erhöhen
bool ParametricSurfaces::Change_Geometry(HfT_osg_Plugin01_ParametricSurface *surf, int lod)
{
    int n = surf->getPatchesU();
    int m = surf->getPatchesV();
    int su = surf->getSegmentsU();
    int sv = surf->getSegmentsV();

    if (lod <= 15)
    {
        n = lod;
    }
    else
    {
        n = 15;
    }
    if (lod <= 15)
    {
        m = lod;
    }
    else
    {
        m = 15;
    }

    if ((lod - 5) >= 3)
    {
        su = lod - 5;
    }
    else
    {
        su = 3;
    }
    if ((lod - 5) >= 3)
    {
        sv = lod - 5;
    }
    else
    {
        sv = 3;
    }

    if (surf->getSurfType() == 0) //Parameterebene
    {

        HfT_osg_Plugin01_Cons *natbound = dynamic_cast<HfT_osg_Plugin01_ParametricPlane *>(surf)->getBoundaryorg();
        surf->recomputeGeometry(n, m, su, sv);
        HfT_osg_Plugin01_Cons *newnatbound = new HfT_osg_Plugin01_Cons(*natbound);
        dynamic_cast<HfT_osg_Plugin01_ParametricPlane *>(surf)->replaceBoundaryorg(newnatbound);
    }
    else
    { // Normale Fläche
        surf->recomputeGeometry(n, m, su, sv);
    }

    Disable_Normals(surf, 1);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
//Wechsel festgelegter Parameterlinien, z.B. Diagonale, Dreieck.,...
bool ParametricSurfaces::Change_Cons(HfT_osg_Plugin01_ParametricSurface *surf, ConsType ctype)
{
    int Pointanz_C = surf->getConsPointanz();
    Vec4d color = surf->getCons()->getColor();
    double ua = surf->getLowerBoundU();
    double ue = surf->getUpperBoundU();
    double va = surf->getLowerBoundV();
    double ve = surf->getUpperBoundV();

    HfT_osg_Plugin01_Cons *newcons = new HfT_osg_Plugin01_Cons(Pointanz_C, ctype, ua, ue, va, ve, color, 0);
    surf->replaceCons(newcons);
    // Eventuell Animationskugeln aus Scenegraph entfernen
    Remove_Animation();

    return true;
}
MatrixTransform *ParametricSurfaces::Get_MatrixTransformNode_Parent(Node *node)
{
    NodePathList npathlist = node->getParentalNodePaths();
    NodePathList::iterator it;
    for (it = npathlist.begin(); it != npathlist.end(); it++)
    {
        NodePath npath = *it;
        NodePath::reverse_iterator itp;
        for (itp = npath.rbegin(); itp != npath.rend(); ++itp)
        {
            Node *node = *itp;
            if (node->asTransform())
            {
                MatrixTransform *mt = dynamic_cast<MatrixTransform *>(node);
                return mt;
            }
        }
    }
    return 0L;
}
//Animations-Kugeln
bool ParametricSurfaces::Sphere_AnimationCreate(HfT_osg_Plugin01_ParametricSurface *surf)
{
    MatrixTransform *mts = Get_MatrixTransformNode_Parent(surf);
    if (!mts)
        return false;

    HfT_osg_Plugin01_Cons *Cons = surf->getCons();
    if (Cons->getType() == CNOTHING)
        return false;

    // AnimationsKugel erstellen und in Ursprung platzieren
    BoundingBox bs = surf->getBoundingBox();

    double Radius = bs.radius();
    double a = 0.05 * Radius;
    double Pi = 4.0 * atan(1.0);
    HfT_osg_Plugin01_ParametricSurface *Kugel = new HfT_osg_Plugin01_ParametricSurface(a, 0., 0., 5, 5, 5, 5, 0., 2 * Pi, -Pi, 0., SLINES, CNOTHING, 10, 0L);
    Kugel->setParametrization(a, 0, 0, 0., 2 * Pi, -Pi, 0., "A*cos(u)*sin(v)", "A*sin(u)*sin(v)", "A*cos(v)");
    Kugel->createGeometryandMode();
    Kugel->setName("Animations-Kugel");
    Kugel->recomputeMode(SLINES, m_ColorAnimation[m_iscreateAnimation], 0L);

    HfT_osg_Plugin01_Animation *Animation = new HfT_osg_Plugin01_Animation(Cons, 0.05 * Radius);
    // Falls noch keine Animationskugeln erstellt
    ref_ptr<MatrixTransform> mt = new MatrixTransform();
    mt->addChild(Kugel);
    mt->setUpdateCallback(Animation);
    if (mts->getNumChildren() == 1)
    {
        Switch *sw = new Switch();
        mts->addChild(sw);
        sw->addChild(mt);
    }
    else
    {
        Switch *sw = dynamic_cast<Switch *>(mts->Group::getChild(1));
        sw->addChild(mt);
    }
    // Farbe der Animationskugeln ändern
    m_iscreateAnimation++;
    return true;
}
//Texturbild aendern
bool ParametricSurfaces::Change_Image(HfT_osg_Plugin01_ParametricSurface *surf)
{
    if ((surf->getSurfMode() == STEXTURE || surf->getSurfMode() == STRANSPARENT))
    {
        Image *newimage = mp_TextureImage->readImage(m_Imagelistcounter);
        surf->recomputeMode(surf->getSurfMode(), newimage);

        return true;
    }
    else
        return false;
}
//ConsType: Parameterlinienarten auf Fläche verschieben
bool ParametricSurfaces::Change_ConsPosition(HfT_osg_Plugin01_ParametricSurface *surf)
{
    int Pointanz_C = surf->getConsPointanz();
    ConsType Constype = surf->getConsType();
    HfT_osg_Plugin01_Cons *cons = surf->getCons();
    double ua = surf->getLowerBoundU();
    double ue = surf->getUpperBoundU();
    double va = surf->getLowerBoundV();
    double ve = surf->getUpperBoundV();

    Vec2d surfpos;

    if (m_sliderValueU_old != m_sliderValueU || m_sliderValueV_old != m_sliderValueV)
    { //damit ConsLinie nicht in Ausgangsposition springt
        surfpos[0] = m_sliderValueU * (ue - ua);
        surfpos[1] = m_sliderValueV * (ve - va);

        Vec2d newpos = surfpos;

        HfT_osg_Plugin01_Cons *newcons = new HfT_osg_Plugin01_Cons(Pointanz_C, Constype, newpos, ua, ue, va, ve, cons->getColor());
        surf->replaceCons(newcons);
        // Animation aus Scenegraph entfernen
        Remove_Animation();
    }
    return true;
}
Switch *ParametricSurfaces::Get_SwitchNode_Parent(Node *node)
{
    NodePathList npathlist = node->getParentalNodePaths();
    NodePathList::iterator it;
    for (it = npathlist.begin(); it != npathlist.end(); it++)
    {
        NodePath npath = *it;
        NodePath::reverse_iterator itp;
        for (itp = npath.rbegin(); itp != npath.rend(); ++itp)
        {
            Node *node = *itp;
            if (node->asSwitch())
                return dynamic_cast<Switch *>(node);
        }
    }
    return 0L;
}
// parameterebene aus oder einblenden
bool ParametricSurfaces::Disable_Plane(bool disable)
{
    Switch *sw = Get_SwitchNode_Parent(mp_Plane);
    if (sw)
    {
        if (disable)
        {
            sw->Switch::setAllChildrenOff();
        }
        else
        {
            sw->Switch::setAllChildrenOn();
        }
    }
    return true;
}
void ParametricSurfaces::ChangePlanePosition(HfT_osg_Plugin01_ParametricSurface *surf)
{
    BoundingBox bs = surf->getBoundingBox();
    double xmin = bs.xMin();
    double xmax = bs.xMax();
    double zmin = bs.zMin();
    double zmax = bs.zMax();

    Vec3d Center = bs.center();

    BoundingBox bplane = mp_Plane->getBoundingBox();
    Vec3d centerp = bplane.center();

    //Abstand zwischen Parameterebene und Fläche
    double vergleich = (xmax - xmin) / 2.5;
    double z = zmax - zmin;
    double abstand;
    if (z <= vergleich)
    {
        abstand = (zmax - zmin);
    }
    else
        abstand = (zmax - zmin) / 3.;

    Vec3d planeposition = Vec3d(Center.x() - centerp.x(), Center.y() - centerp.y(), Center.z() - (zmax - zmin) / 2. - abstand);

    maplane.makeTranslate(planeposition);
    mtplane->setMatrix(maplane);
}

bool ParametricSurfaces::Change_Radius(HfT_osg_Plugin01_ParametricSurface *surf, double rad)
{ // Radius der Flächen ändern

    surf->setRadian(rad);
    surf->recomputeGeometry();

    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);
    ChangePlanePosition(surf);

    if (m_pCheckboxMenuNormals->getState() == true)
        Disable_Normals(mp_Surf, false);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
bool ParametricSurfaces::Change_Length(HfT_osg_Plugin01_ParametricSurface *surf, double len)
{ // Länge der Fläche ändern

    surf->setLength(len);
    surf->recomputeGeometry();
    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);
    ChangePlanePosition(surf);

    if (m_pCheckboxMenuNormals->getState() == true)
        Disable_Normals(mp_Surf, false);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}

bool ParametricSurfaces::Change_Height(HfT_osg_Plugin01_ParametricSurface *surf, double height)
{ // Höhe der Fläche ändern

    surf->setHeight(height /*newheight*/);
    surf->recomputeGeometry();

    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);
    ChangePlanePosition(surf);

    if (m_pCheckboxMenuNormals->getState() == true)
        Disable_Normals(mp_Surf, false);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
//Veränderter u_anfang_param aendert Fläche und Paramebene
bool ParametricSurfaces::Change_Ua(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double du)
{ // Höhe der Fläche ändern

    surf->setLowerBoundU(du);
    surf->recomputeGeometry();

    plane->setLowerBoundU(du);
    HfT_osg_Plugin01_Cons *orgbound = plane->getBoundaryorg();
    orgbound->setColor(Vec4d(1., 1., 0., 1.));

    plane->recomputeGeometry();

    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
bool ParametricSurfaces::Change_Ue(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double du)
{ // Höhe der Fläche ändern

    surf->setUpperBoundU(du);
    surf->recomputeGeometry();

    plane->setUpperBoundU(du);
    HfT_osg_Plugin01_Cons *orgbound = plane->getBoundaryorg();
    orgbound->setColor(Vec4d(1., 1., 0., 1.));

    plane->recomputeGeometry();

    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
bool ParametricSurfaces::Change_Va(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double dv)
{ // Höhe der Fläche ändern

    surf->setLowerBoundV(dv);
    surf->recomputeGeometry();

    plane->setLowerBoundV(dv);
    HfT_osg_Plugin01_Cons *orgbound = plane->getBoundaryorg();
    orgbound->setColor(Vec4d(1., 1., 0., 1.));

    plane->recomputeGeometry();

    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
//Veränderter v_end_param aendert Fläche und Paramebene
bool ParametricSurfaces::Change_Ve(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricPlane *plane, double dv)
{ // Höhe der Fläche ändern

    surf->setUpperBoundV(dv);
    surf->recomputeGeometry();

    plane->setUpperBoundV(dv);
    HfT_osg_Plugin01_Cons *orgbound = plane->getBoundaryorg();
    orgbound->setColor(Vec4d(1., 1., 0., 1.));

    plane->recomputeGeometry();

    // Eventuell Animationskugeln aus Scenegraph entfernen
    int animation = m_iscreateAnimation;
    Remove_Animation();
    m_iscreateAnimation = animation;
    Remove_Animation();
    Disable_Normals(surf, 1);

    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    return true;
}
bool ParametricSurfaces::Disable_Boundary(bool disable)
{
    // Rand ausblenden
    HfT_osg_Plugin01_Cons *boundary = mp_Surf->getBoundary();
    if (boundary && disable)
    {
        mp_Surf->removeDrawable(boundary);
        mp_Surf->setBoundary(0L);
    }
    else if (!disable)
    {
        if (!mp_Surf->getBoundary())
        {
            double m_ua = mp_Surf->getLowerBoundU();
            double m_ue = mp_Surf->getUpperBoundU();
            double m_va = mp_Surf->getLowerBoundV();
            double m_ve = mp_Surf->getUpperBoundV();
            int m_n = mp_Surf->getPatchesU();
            int m_m = mp_Surf->getPatchesV();
            int m_su = mp_Surf->getSegmentsU();
            int m_sv = mp_Surf->getSegmentsV();

            int m_Pointanz_B;
            m_Pointanz_B = 2 * ((m_n * (m_su + 1) - (m_n - 1)) + (m_m * (m_sv + 1) - (m_m - 1)));
            HfT_osg_Plugin01_Cons *boundary = new HfT_osg_Plugin01_Cons(m_Pointanz_B, CNATBOUND, m_ua, m_ue, m_va, m_ve, Vec4(0.2f, 1.f, 0.f, 1.f), m_formula);

            mp_Surf->setBoundary(boundary);
        }
    }
    return true;
}
bool ParametricSurfaces::Disable_Normals(HfT_osg_Plugin01_ParametricSurface *surf, bool disable)
{
    if (disable)
    { // Falls Normalen schon erzeugt wurden --> wieder löschen
        if (mp_GeomNormals)
        {
            surf->removeDrawable(mp_GeomNormals);
            mp_GeomNormals = 0L;
            return true;
        }
        else // Falls Normalen nicht da --> nix tun
        {
            return false;
        }
    }
    else // Normalen erzeugen, falls nicht da
    {
        if (mp_GeomNormals) // wurden sind Normalen schon erzeugt
        {
            return false;
        }
        else // Normalen jetzt erzeugen
        {
            // Geometrieobjekt für Normalen erzeugen
            mp_GeomNormals = new Geometry();
            ref_ptr<DrawElementsUInt> lines = new DrawElementsUInt(PrimitiveSet::LINES, 0);

            Vec3Array *normals = surf->getNormalArray();
            Vec3Array *points = surf->getPointArray();
            Vec3Array *normalpoints = new Vec3Array();
            BoundingBox bs = surf->getBoundingBox();
            double Radius = bs.radius();

            unsigned int k = 0;
            for (unsigned int i = 0; i < points->size(); i++)
            {
                Vec3d startpoint = (*points)[i];
                normalpoints->push_back(startpoint);
                Vec3d nscale = (*normals)[i].operator*(Radius / 10);
                Vec3d endpoint = (*points)[i].operator+(nscale);
                normalpoints->push_back(endpoint);
                lines->push_back(k);
                lines->push_back(k + 1);
                k = k + 2;
            }
            ref_ptr<Vec4Array> colors = new Vec4Array;
            colors->push_back(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
            mp_GeomNormals->setColorArray(colors.get());
            mp_GeomNormals->setColorBinding(Geometry::BIND_OVERALL);
            mp_GeomNormals->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
            mp_GeomNormals->addPrimitiveSet(lines);
            mp_GeomNormals->setVertexArray(normalpoints);

            osg::StateSet *stateset = mp_GeomNormals->getOrCreateStateSet();
            osg::LineWidth *lw = new osg::LineWidth(2.0f);
            stateset->setAttribute(lw);

            surf->addDrawable(mp_GeomNormals);
        }
    }
    return true;
}
// Animationskugeln aus SceneGraph entfernen
bool ParametricSurfaces::Remove_Animation()
{
    for (int i = 0; i < m_iscreateAnimation; i++)
    {
        HfT_osg_Plugin01_ParametricSurface *Kugel = dynamic_cast<HfT_osg_Plugin01_ParametricSurface *>(Get_Node_byName("Animations-Kugel"));
        if (Kugel)
        {
            ref_ptr<osg::Group> mt = Kugel->getParent(0);
            mt->removeChild(Kugel);
            Kugel = 0L;
        }
    }
    m_iscreateAnimation = 0;
    return true;
}

//neue Flaeche einlesen
bool ParametricSurfaces::Read_Surface(Group *surfgroup, std::string surfname)
{
    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

    sliderStart = true;
    actualizeSlider = false;
    actualizeSliderStep8 = false;

    m_natbound = false;

    ref_ptr<Group> newgeogroup = InitSurfgroup(surfname, mp_TextureImage->m_imagedir); //liest neue Flaeche ein
    Group *group = surfgroup->getParent(0);
    group->replaceChild(surfgroup, newgeogroup);

    HfT_osg_Plugin01_ParametricSurface *newsurf = dynamic_cast<HfT_osg_Plugin01_ParametricSurface *>(Get_Node_byName("Flaeche"));
    HfT_osg_Plugin01_ParametricPlane *newplane = dynamic_cast<HfT_osg_Plugin01_ParametricPlane *>(Get_Node_byName("Parameterebene"));

    if (!newsurf)
    {
        std::cout << "Keine neue Flaeche erzeugt" << std::endl;
        return false;
    }
    // Neue Flächengruppe merken
    mp_Plane = newplane;
    mp_Surf = newsurf;
    mp_GeoGroup = newgeogroup;

    // Falls Animation lief
    m_iscreateAnimation = 0;

    return true;
}
bool ParametricSurfaces::CheckParametrization(HfT_osg_Plugin01_ParametricSurface *newsurf, std::string xpstr, std::string ypstr, std::string zpstr)
{
    mp_ParserSurface = newsurf->getParserSurface();
    // Strings checken
    bool xdef = mp_ParserSurface->SetFunktionX(xpstr);
    bool ydef = mp_ParserSurface->SetFunktionY(ypstr);
    bool zdef = mp_ParserSurface->SetFunktionZ(zpstr);
    bool defined = xdef && ydef && zdef;

    return defined;
}

//neue Flaeche einlesen mit TexturImage
ref_ptr<Group> ParametricSurfaces::InitSurfgroup(std::string surffilepath, std::string imagefilepath)
{
    mp_GeomNormals = 0L;
    ref_ptr<Group> surfgroup = new Group();
    string xstr, ystr, zstr;
    double a, b, c, ua, ue, va, ve;
    HfT_osg_Plugin01_ReadSurfDescription *Readobj = new HfT_osg_Plugin01_ReadSurfDescription(surffilepath);
    std::cerr << "Plugin laedt in InitSurfgroup nach ReadSurfDescription" << std::endl;
    // Image gleich einlesen
    if (!mp_TextureImage)
        mp_TextureImage = new HfT_osg_ReadTextureImage(imagefilepath, 1);
    else
        mp_TextureImage->readImage(1);
    if (Readobj)
    {
        Readobj->get_Param(xstr, ystr, zstr, a, b, c, ua, ue, va, ve);

        //Werte der Schieberegler festlegen
        //Verschieben der Cons(Parameterlinientypen)
        m_pSliderMenuU->setMin(-1.0);
        m_pSliderMenuU->setMax(1.0);
        m_pSliderMenuV->setMin(-1.0);
        m_pSliderMenuV->setMax(1.0);
        m_pSliderMenuU->setPrecision(2.0);
        m_pSliderMenuV->setPrecision(2.0);
        m_pSliderMenuU->setValue(0.0);
        m_pSliderMenuV->setValue(0.0);
        //Beginning value of the slider menu
        //m_sliderValueU = m_pSliderMenuU -> getValue();
        //m_sliderValueV = m_pSliderMenuV -> getValue();
        ////Darstellung ändern: Radius,Länge,Höhe
        sliderStartValueA = a;
        sliderStartValueB = b;
        sliderStartValueC = c;
        createMenu3();

        //Parametergrenzen ändern:ua,ue,va,ve
        m_pSliderMenuUa->setMin(ua);
        m_pSliderMenuUa->setMax(ue);
        m_pSliderMenuUe->setMin(ua);
        m_pSliderMenuUe->setMax(ue);
        m_pSliderMenuVa->setMin(va);
        m_pSliderMenuVa->setMax(ve);
        m_pSliderMenuVe->setMin(va);
        m_pSliderMenuVe->setMax(ve);
        m_pSliderMenuUa->setPrecision(2);
        m_pSliderMenuUe->setPrecision(2);
        m_pSliderMenuVa->setPrecision(2);
        m_pSliderMenuVe->setPrecision(2);
        m_pSliderMenuUa->setValue(ua);
        m_pSliderMenuUe->setValue(ue);
        m_pSliderMenuVa->setValue(va);
        m_pSliderMenuVe->setValue(ve);
        //Beginning value of the slider menu
        m_sliderValueUa = m_pSliderMenuUa->getValue();
        m_sliderValueUe = m_pSliderMenuUe->getValue();
        m_sliderValueVa = m_pSliderMenuVa->getValue();
        m_sliderValueVe = m_pSliderMenuVe->getValue();
        //LOD
        //aktueller Schiebereglerwert
        m_pSliderMenuLOD->setMin(2);
        m_pSliderMenuLOD->setMax(18);
        m_pSliderMenuLOD->setPrecision(0);
        m_pSliderMenuLOD->setValue(10); //da surf->getPatchesU()=10
        //Beginning value of the slider menu
        m_sliderValueLOD = m_pSliderMenuLOD->getValue();

        Fill_SurfGroup(surfgroup, a, b, c, 10, 10, 5, 5, ua, ue, va, ve, xstr, ystr, zstr,
                       STEXTURE, CNOTHING, 500, mp_TextureImage->getImage());
    }

    return (surfgroup);
}
//Darstellungsweise: Punkte, Transparent,...
bool ParametricSurfaces::Fill_SurfGroup(Group *surfgroup, double a, double b, double c,
                                        int n, int m, int su, int sv,
                                        double ua, double ue, double va, double ve,
                                        string xstr, string ystr, string zstr,
                                        SurfMode smode, ConsType ctype, int canz, Image *image)
{
    HfT_osg_Plugin01_ParametricSurface *surf = new HfT_osg_Plugin01_ParametricSurface(a, b, c, n, m, su, sv,
                                                                                      ua, ue, va, ve, smode, ctype, canz, image);
    surf->setParametrization(a, b, c, ua, ue, va, ve, xstr, ystr, zstr);
    surf->createGeometryandMode();

    ref_ptr<MatrixTransform> mt = new MatrixTransform();
    ref_ptr<Switch> sw = new Switch;
    //root->addChild(surfgroup);//ausschalten
    surfgroup->addChild(sw);
    sw.get()->addChild(mt);
    mt.get()->addChild(surf);

    surf->setSurfType(99);
    surf->setName("Flaeche");
    BoundingBox bs = surf->getBoundingBox();
    double xmin = bs.xMin();
    double xmax = bs.xMax();
    double ymin = bs.yMin();
    double ymax = bs.yMax();
    double zmin = bs.zMin();
    double zmax = bs.zMax();

    Vec3d Center = bs.center();

    double ap = (xmax - xmin) / (ue - ua);
    double bp = (ymax - ymin) / (ve - va);

    HfT_osg_Plugin01_ParametricPlane *plane = new HfT_osg_Plugin01_ParametricPlane(ap, bp, n, m, su, sv, ua, ue, va, ve,
                                                                                   smode, ctype, canz, image);

    if (!m_natbound)
    {
        m_Pua = ua;
        m_Pue = ue;
        m_Pva = va;
        m_Pve = ve;
        m_natbound = true;
    }
    plane->setSurfType(0);
    plane->setName("Parameterebene");
    // Original Rand merken und auch mit einzeichnen
    HfT_osg_Plugin01_Cons *orgbound = new HfT_osg_Plugin01_Cons(canz, CNATBOUND, ua, ue, va, ve, Vec4d(1., 0.5, 0., 1.), 0);
    plane->setBoundaryorg(orgbound);
    plane->computeCons(orgbound);

    BoundingBox bplane = plane->getBoundingBox();

    Vec3d centerp = bplane.center();

    //Abstand zwischen Parameterebene und Fläche
    double vergleich = (xmax - xmin) / 2.5;
    double z = zmax - zmin;
    double abstand;
    if (z <= vergleich)
    {
        abstand = (zmax - zmin);
    }
    else
        abstand = (zmax - zmin) / 3.;

    Vec3d planeposition = Vec3d(Center.x() - centerp.x(), Center.y() - centerp.y(), Center.z() - (zmax - zmin) / 2. - abstand);
    maplane.makeTranslate(planeposition);
    mtplane = new MatrixTransform(maplane);
    ref_ptr<Switch> swplane = new Switch;
    surfgroup->addChild(swplane);
    swplane.get()->addChild(mtplane);
    mtplane.get()->addChild(plane);
    return true;
}

//neue Parameterebene erzeugen für Normalschnittanimation (für Change_Surface: wird nicht benutzt)
HfT_osg_Plugin01_ParametricPlane *ParametricSurfaces::Create_Plane(HfT_osg_Plugin01_ParametricSurface *plane,
                                                                   int n, int m, int su, int sv,
                                                                   double m_Pua, double m_Pue, double m_Pva, double m_Pve,
                                                                   double ua, double ue, double va, double ve,
                                                                   SurfMode smode, ConsType ctype, int canz, Image *image)
{
    double a = plane->getRadian();
    double b = plane->getLength();
    HfT_osg_Plugin01_ParametricPlane *newplane = new HfT_osg_Plugin01_ParametricPlane(a, b, n, m, su, sv, ua, ue, va, ve,
                                                                                      smode, ctype, canz, image);
    // Original Rand merken und auch mit einzeichnen
    HfT_osg_Plugin01_Cons *orgbound = new HfT_osg_Plugin01_Cons(canz, CNATBOUND, m_Pua, m_Pue, m_Pva, m_Pve, Vec4d(1., 0.5, 0., 1.), 0);
    newplane->setBoundaryorg(orgbound);
    newplane->computeCons(orgbound);
    newplane->setSurfType(0);
    newplane->setName("Parameterebene");

    return newplane;
}

void ParametricSurfaces::ResetSliderUV()
{
    m_pSliderMenuU->setValue(0.0);
    m_pSliderMenuV->setValue(0.0);
    m_sliderValueU = 0.0;
    m_sliderValueV = 0.0;
    m_sliderValueU_old = 0.0;
    m_sliderValueV_old = 0.0;
}

void ParametricSurfaces::createMenu3()
{
    m_pObjectMenu3 = new coRowMenu("Menu-ABC");
    m_pObjectMenu3->setVisible(false);
    m_pObjectMenu3->setAttachment(coUIElement::RIGHT);

    m_pObjectMenu7 = new coRowMenu("Menu-Curvatures");
    m_pObjectMenu7->setVisible(false);
    m_pObjectMenu7->setAttachment(coUIElement::RIGHT);

    std::string imageName = HfT_int_to_string(m_formula) + ".png";
    std::string imagePath = (std::string)coCoviseConfig::getEntry("value", "COVER.Plugin.ParametricSurfaces.DataPath") + "/" + imageName;

    std::string localizedPath = coTranslator::translatePath(imagePath);

    std::cerr << "Image to load_Loc: " << localizedPath << std::endl;
    std::cerr << "Image to load: " << imagePath << std::endl;

    float vsize_ = coCoviseConfig::getFloat("COVER.Plugin.DocumentViewer.Vsize", -1);
    float aspect_ratio_ = coCoviseConfig::getFloat("COVER.Plugin.DocumentViewer.AspectRatio", 0);
    float aspect_;

    osg::Image *bild = osgDB::readImageFile(localizedPath);
    if (bild)
    {
        int width = bild->s();
        int hight = bild->t();
        if (changesize == true)
        {
            aspect_ = ((float)hsize_DocView / (float)vsize_DocView);
            vsize_ = vsize_DocView;
        }
        else
        {
            if (aspect_ratio_ > 0.)
            {
                aspect_ = aspect_ratio_;
                vsize_ = hight;
            }
            else
            {
                aspect_ = ((float)width / (float)hight);
                vsize_ = hight;
            }
        }
    }

    imageItemList_ = new coMovableBackgroundMenuItem(localizedPath.c_str(), aspect_, vsize_);

    //matrices to position the menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;

    // position menu with values from config file
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", 0);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", -5);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 0);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ParametricSurfaces.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ParametricSurfaces.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ParametricSurfaces.MenuPosition", pz);
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("s", "COVER.Plugin.ParametricSurfaces.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    m_pObjectMenu3->setTransformMatrix(&matrix);
    m_pObjectMenu3->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu7->setTransformMatrix(&matrix);
    m_pObjectMenu7->setScale(cover->getSceneSize() / 2500);

    imageMenu_->setVisible(true);
    imageMenu_->removeAll();
    imageMenu_->add(imageItemList_);

    //Menu-ABC
    m_pSliderMenuA = new coSliderMenuItem(coTranslator::coTranslate("Radius-A"), -1.0, 1.0, 0.0);
    m_pSliderMenuA->setMenuListener(this);
    if (sliderStartValueA != 0.0)
    {
        m_pObjectMenu3->add(m_pSliderMenuA);
    }
    m_pSliderMenuB = new coSliderMenuItem(coTranslator::coTranslate("Laenge-B"), -1.0, 1.0, 0.0);
    m_pSliderMenuB->setMenuListener(this);
    if (sliderStartValueB != 0.0)
    {
        m_pObjectMenu3->add(m_pSliderMenuB);
    }
    m_pSliderMenuC = new coSliderMenuItem(coTranslator::coTranslate("Hoehe-C"), -1.0, 1.0, 0.0);
    m_pSliderMenuC->setMenuListener(this);
    if (sliderStartValueC != 0)
    {
        m_pObjectMenu3->add(m_pSliderMenuC);
    }
    m_pCheckboxMenuPlane2 = new coCheckboxMenuItem(coTranslator::coTranslate("zeige Parameterebene"), checkboxPlaneState);
    m_pCheckboxMenuPlane2->setMenuListener(this);
    m_pObjectMenu3->add(m_pCheckboxMenuPlane2);

    //Menu-Kruemmungen
    m_pCheckboxMenuGauss = new coCheckboxMenuItem(coTranslator::coTranslate("Gauss-Kruemmung"), false);
    m_pCheckboxMenuGauss->setMenuListener(this);
    m_pObjectMenu7->add(m_pCheckboxMenuGauss);

    m_pCheckboxMenuMean = new coCheckboxMenuItem(coTranslator::coTranslate("Mittlere Kruemmung"), false);
    m_pCheckboxMenuMean->setMenuListener(this);
    m_pObjectMenu7->add(m_pCheckboxMenuMean);

    m_pSliderMenuA_ = new coSliderMenuItem(coTranslator::coTranslate("Radius-A"), -1.0, 1.0, 0.0);
    m_pSliderMenuA_->setMenuListener(this);
    if (sliderStartValueA != 0.0)
    {
        m_pObjectMenu7->add(m_pSliderMenuA_);
    }
    m_pSliderMenuB_ = new coSliderMenuItem(coTranslator::coTranslate("Laenge-B"), -1.0, 1.0, 0.0);
    m_pSliderMenuB_->setMenuListener(this);
    if (sliderStartValueB != 0.0)
    {
        m_pObjectMenu7->add(m_pSliderMenuB_);
    }
    m_pSliderMenuC_ = new coSliderMenuItem(coTranslator::coTranslate("Hoehe-C"), -1.0, 1.0, 0.0);
    m_pSliderMenuC_->setMenuListener(this);
    if (sliderStartValueC != 0.0)
    {
        m_pObjectMenu7->add(m_pSliderMenuC_);
    }

    //Darstellung ändern: Radius,Länge,Höhe
    m_pSliderMenuA->setMin(0.1);
    m_pSliderMenuA->setMax(2.0 * sliderStartValueA);
    m_pSliderMenuB->setMin(0.1);
    m_pSliderMenuB->setMax(2.0 * sliderStartValueB);
    m_pSliderMenuC->setMin(0.1);
    m_pSliderMenuC->setMax(2.0 * sliderStartValueC);
    m_pSliderMenuA->setPrecision(1); //da in Formel: Kommazahlen
    m_pSliderMenuB->setPrecision(1);
    m_pSliderMenuC->setPrecision(1);

    m_pSliderMenuA_->setMin(0.1);
    m_pSliderMenuA_->setMax(2.0 * sliderStartValueA);
    m_pSliderMenuB_->setMin(0.1);
    m_pSliderMenuB_->setMax(2.0 * sliderStartValueB);
    m_pSliderMenuC_->setMin(0.1);
    m_pSliderMenuC_->setMax(2.0 * sliderStartValueC);
    m_pSliderMenuA_->setPrecision(1); //da in Formel: Kommazahlen
    m_pSliderMenuB_->setPrecision(1);
    m_pSliderMenuC_->setPrecision(1);

    m_sliderValueA = sliderStartValueA;
    m_sliderValueB = sliderStartValueB;
    m_sliderValueC = sliderStartValueC;
}

//---------------------------------------------------
//Implements ParametricCurves::createMenu() //wird in init() aufgerufen
//---------------------------------------------------
void ParametricSurfaces::createMenu()
{
    m_pObjectMenu1 = new coRowMenu("Menu-Surface");
    m_pObjectMenu1->setVisible(false);
    m_pObjectMenu1->setAttachment(coUIElement::RIGHT);

    m_pObjectMenu2 = new coRowMenu("Menu-Darstellung");
    m_pObjectMenu2->setVisible(false);
    m_pObjectMenu2->setAttachment(coUIElement::RIGHT);

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
    m_pObjectMenu10->setAttachment(coUIElement::RIGHT);

    //matrices to position the menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;

    // position menu with values from config file
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", 0);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", -5);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 0);
    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ParametricSurfaces.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ParametricSurfaces.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ParametricSurfaces.MenuPosition", pz);
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("s", "COVER.Plugin.ParametricSurfaces.MenuSize", s);

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

    m_pObjectMenu4->setTransformMatrix(&matrix);
    m_pObjectMenu4->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu5->setTransformMatrix(&matrix);
    m_pObjectMenu5->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu6->setTransformMatrix(&matrix);
    m_pObjectMenu6->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu8->setTransformMatrix(&matrix);
    m_pObjectMenu8->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu9->setTransformMatrix(&matrix);
    m_pObjectMenu9->setScale(cover->getSceneSize() / 2500);

    m_pObjectMenu10->setTransformMatrix(&matrix);
    m_pObjectMenu10->setScale(cover->getSceneSize() / 2500);

    //Surface
    m_pButtonMenuSurface = new coLabelMenuItem(coTranslator::coTranslate("Einfache Geometrien"));
    m_pButtonMenuSurface->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuSurface);

    m_pButtonMenuKegel = new coButtonMenuItem(coTranslator::coTranslate("Kegel"));
    m_pButtonMenuKegel->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuKegel);

    m_pButtonMenuKugel = new coButtonMenuItem(coTranslator::coTranslate("Kugel"));
    m_pButtonMenuKugel->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuKugel);

    m_pButtonMenuMoebius = new coButtonMenuItem(coTranslator::coTranslate("Moebiusband"));
    m_pButtonMenuMoebius->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuMoebius);

    m_pButtonMenuParaboloid = new coButtonMenuItem(coTranslator::coTranslate("Paraboloid"));
    m_pButtonMenuParaboloid->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuParaboloid);

    m_pButtonMenuZylinder = new coButtonMenuItem(coTranslator::coTranslate("Zylinder"));
    m_pButtonMenuZylinder->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuZylinder);

    m_pLabelMenuUnterstreichen1 = new coLabelMenuItem(coTranslator::coTranslate("Komplexere"));
    m_pLabelMenuUnterstreichen1->setMenuListener(this);
    m_pObjectMenu1->add(m_pLabelMenuUnterstreichen1);

    m_pButtonMenuBonan = new coButtonMenuItem(coTranslator::coTranslate("Bonan"));
    m_pButtonMenuBonan->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuBonan);

    m_pButtonMenuBoy = new coButtonMenuItem(coTranslator::coTranslate("Boy"));
    m_pButtonMenuBoy->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuBoy);

    m_pButtonMenuCrossCap = new coButtonMenuItem(coTranslator::coTranslate("Cross Cap"));
    m_pButtonMenuCrossCap->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuCrossCap);

    m_pButtonMenuDini = new coButtonMenuItem(coTranslator::coTranslate("Dini"));
    m_pButtonMenuDini->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuDini);

    m_pButtonMenuEnneper = new coButtonMenuItem(coTranslator::coTranslate("Enneper"));
    m_pButtonMenuEnneper->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuEnneper);

    m_pButtonMenuHelicalTorus = new coButtonMenuItem(coTranslator::coTranslate("Helical Torus"));
    m_pButtonMenuHelicalTorus->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuHelicalTorus);

    m_pButtonMenuKatenoid = new coButtonMenuItem(coTranslator::coTranslate("Katenoid"));
    m_pButtonMenuKatenoid->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuKatenoid);

    m_pButtonMenuKlein = new coButtonMenuItem(coTranslator::coTranslate("Kleinsche Flasche"));
    m_pButtonMenuKlein->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuKlein);

    m_pButtonMenuKuen = new coButtonMenuItem(coTranslator::coTranslate("Kuen"));
    m_pButtonMenuKuen->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuKuen);

    m_pButtonMenuPluecker = new coButtonMenuItem(coTranslator::coTranslate("Pluecker"));
    m_pButtonMenuPluecker->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuPluecker);

    /*  m_pButtonMenuPseudoSphere = new coButtonMenuItem(coTranslator::coTranslate("PseudoSphere"));
   m_pButtonMenuPseudoSphere -> setMenuListener(this);
   m_pObjectMenu1 -> add(m_pButtonMenuPseudoSphere);*/

    m_pButtonMenuRevolution = new coButtonMenuItem(coTranslator::coTranslate("Rotationsflaeche"));
    m_pButtonMenuRevolution->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuRevolution);

    m_pButtonMenuRoman = new coButtonMenuItem(coTranslator::coTranslate("Roman"));
    m_pButtonMenuRoman->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuRoman);

    m_pButtonMenuShell = new coButtonMenuItem(coTranslator::coTranslate("Shell"));
    m_pButtonMenuShell->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuShell);

    m_pButtonMenuSnake = new coButtonMenuItem(coTranslator::coTranslate("Snake"));
    m_pButtonMenuSnake->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuSnake);

    m_pButtonMenuTrumpet = new coButtonMenuItem(coTranslator::coTranslate("Trumpet"));
    m_pButtonMenuTrumpet->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuTrumpet);

    m_pButtonMenuTwistedSphere = new coButtonMenuItem(coTranslator::coTranslate("Twisted Sphere"));
    m_pButtonMenuTwistedSphere->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuTwistedSphere);
    //Menu-Darstellung
    m_pButtonMenuDarst = new coLabelMenuItem(coTranslator::coTranslate("Darstellung auswaehlen"));
    m_pButtonMenuDarst->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuDarst);

    m_pLabelMenuUnterstreichen2 = new coLabelMenuItem(coTranslator::coTranslate("-------------------------------"));
    m_pLabelMenuUnterstreichen2->setMenuListener(this);
    m_pObjectMenu2->add(m_pLabelMenuUnterstreichen2);

    m_pButtonMenuPoints = new coButtonMenuItem(coTranslator::coTranslate("Punkte"));
    m_pButtonMenuPoints->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuPoints);

    m_pButtonMenuLines = new coButtonMenuItem(coTranslator::coTranslate("Parameterlinien"));
    m_pButtonMenuLines->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuLines);

    m_pButtonMenuTriangles = new coButtonMenuItem(coTranslator::coTranslate("Dreiecke"));
    m_pButtonMenuTriangles->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuTriangles);

    m_pButtonMenuQuads = new coButtonMenuItem(coTranslator::coTranslate("Quadrate"));
    m_pButtonMenuQuads->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuQuads);

    m_pButtonMenuShade = new coButtonMenuItem(coTranslator::coTranslate("Shade"));
    m_pButtonMenuShade->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuShade);

    m_pButtonMenuTextur1 = new coButtonMenuItem(coTranslator::coTranslate("Textur 1"));
    m_pButtonMenuTextur1->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuTextur1);

    m_pButtonMenuTextur2 = new coButtonMenuItem(coTranslator::coTranslate("Textur 2"));
    m_pButtonMenuTextur2->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuTextur2);

    m_pButtonMenuTransparent = new coButtonMenuItem(coTranslator::coTranslate("Transparent"));
    m_pButtonMenuTransparent->setMenuListener(this);
    m_pObjectMenu2->add(m_pButtonMenuTransparent);

    m_pSliderMenuLOD = new coSliderMenuItem(coTranslator::coTranslate("Level of Detail"), -1.0, 1.0, 0.0);
    m_pSliderMenuLOD->setMenuListener(this);
    m_pObjectMenu2->add(m_pSliderMenuLOD);
    //Parameterebene ein/aus
    m_pCheckboxMenuPlane = new coCheckboxMenuItem(coTranslator::coTranslate("zeige Parameterebene"), checkboxPlaneState);
    m_pCheckboxMenuPlane->setMenuListener(this);
    m_pObjectMenu2->add(m_pCheckboxMenuPlane);
    //Flaechenrand ein/aus
    m_pCheckboxMenuNatbound = new coCheckboxMenuItem(coTranslator::coTranslate("zeige Flaechenrand"), true);
    m_pCheckboxMenuNatbound->setMenuListener(this);
    m_pObjectMenu2->add(m_pCheckboxMenuNatbound);
    //Flaechennormalen ein/aus
    m_pCheckboxMenuNormals = new coCheckboxMenuItem(coTranslator::coTranslate("zeige Flaechennormalen"), false);
    m_pCheckboxMenuNormals->setMenuListener(this);
    m_pObjectMenu2->add(m_pCheckboxMenuNormals);

    //Menu-Grenzen
    m_pSliderMenuUa = new coSliderMenuItem(coTranslator::coTranslate("Anfang des u-Parameterbereichs"), -1.0, 1.0, 0.0);
    m_pSliderMenuUa->setMenuListener(this);
    m_pObjectMenu4->add(m_pSliderMenuUa);

    m_pSliderMenuUe = new coSliderMenuItem(coTranslator::coTranslate("Ende des u-Parameterbereichs"), -1.0, 1.0, 0.0);
    m_pSliderMenuUe->setMenuListener(this);
    m_pObjectMenu4->add(m_pSliderMenuUe);

    m_pSliderMenuVa = new coSliderMenuItem(coTranslator::coTranslate("Anfang des v-Parameterbereichs"), -1.0, 1.0, 0.0);
    m_pSliderMenuVa->setMenuListener(this);
    m_pObjectMenu4->add(m_pSliderMenuVa);

    m_pSliderMenuVe = new coSliderMenuItem(coTranslator::coTranslate("Ende des v-Parameterbereichs"), -1.0, 1.0, 0.0);
    m_pSliderMenuVe->setMenuListener(this);
    m_pObjectMenu4->add(m_pSliderMenuVe);
    //Cons
    m_pButtonMenuCons = new coLabelMenuItem(coTranslator::coTranslate("Flaechenkurve auswaehlen"));
    m_pButtonMenuCons->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuCons);

    m_pLabelMenuUnterstreichen3 = new coLabelMenuItem(coTranslator::coTranslate("------------------------------------"));
    m_pLabelMenuUnterstreichen3->setMenuListener(this);
    m_pObjectMenu5->add(m_pLabelMenuUnterstreichen3);

    m_pButtonMenuUCenter = new coButtonMenuItem(coTranslator::coTranslate("Mitte des u-Parameters"));
    m_pButtonMenuUCenter->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuUCenter);

    m_pButtonMenuVCenter = new coButtonMenuItem(coTranslator::coTranslate("Mitte des v-Parameters"));
    m_pButtonMenuVCenter->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuVCenter);

    m_pButtonMenuDiagonal = new coButtonMenuItem(coTranslator::coTranslate("Diagonale"));
    m_pButtonMenuDiagonal->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuDiagonal);

    m_pButtonMenuTriangle = new coButtonMenuItem(coTranslator::coTranslate("Dreieck"));
    m_pButtonMenuTriangle->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuTriangle);

    m_pButtonMenuEllipse = new coButtonMenuItem(coTranslator::coTranslate("Ellipse"));
    m_pButtonMenuEllipse->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuEllipse);

    m_pButtonMenuSquare = new coButtonMenuItem(coTranslator::coTranslate("Rechteck"));
    m_pButtonMenuSquare->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuSquare);

    m_pButtonMenuNatbound = new coButtonMenuItem(coTranslator::coTranslate("Flaechenrand"));
    m_pButtonMenuNatbound->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuNatbound);

    m_pButtonMenuNothing = new coButtonMenuItem(coTranslator::coTranslate("keine"));
    m_pButtonMenuNothing->setMenuListener(this);
    m_pObjectMenu5->add(m_pButtonMenuNothing);

    //Menu-Animation
    m_pSliderMenuU = new coSliderMenuItem(coTranslator::coTranslate("Verschieben in u-Richtung"), -1.0, 1.0, 0.0);
    m_pSliderMenuU->setMenuListener(this);
    m_pObjectMenu6->add(m_pSliderMenuU);

    m_pSliderMenuV = new coSliderMenuItem(coTranslator::coTranslate("Verschieben in v-Richtung"), -1.0, 1.0, 0.0);
    m_pSliderMenuV->setMenuListener(this);
    m_pObjectMenu6->add(m_pSliderMenuV);

    m_pButtonMenuAnimationSphere = new coButtonMenuItem(coTranslator::coTranslate("Animation: Kugel losschicken"));
    m_pButtonMenuAnimationSphere->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuAnimationSphere);

    m_pButtonMenuAnimationOff = new coButtonMenuItem(coTranslator::coTranslate("Animation beenden"));
    m_pButtonMenuAnimationOff->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuAnimationOff);

    m_pButtonMenuLeerzeile = new coLabelMenuItem(coTranslator::coTranslate("  "));
    m_pButtonMenuLeerzeile->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuLeerzeile);

    m_pButtonMenuDarstellungAend = new coLabelMenuItem(coTranslator::coTranslate("Darstellung aendern"));
    m_pButtonMenuDarstellungAend->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuDarstellungAend);

    m_pLabelMenuUnterstreichen4 = new coLabelMenuItem(coTranslator::coTranslate("---------------------------"));
    m_pLabelMenuUnterstreichen4->setMenuListener(this);
    m_pObjectMenu6->add(m_pLabelMenuUnterstreichen4);

    m_pButtonMenuPoints_ = new coButtonMenuItem(coTranslator::coTranslate("Punkte"));
    m_pButtonMenuPoints_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuPoints_);

    m_pButtonMenuLines_ = new coButtonMenuItem(coTranslator::coTranslate("Parameterlinien"));
    m_pButtonMenuLines_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuLines_);

    m_pButtonMenuTriangles_ = new coButtonMenuItem(coTranslator::coTranslate("Dreiecke"));
    m_pButtonMenuTriangles_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuTriangles_);

    m_pButtonMenuQuads_ = new coButtonMenuItem(coTranslator::coTranslate("Quadrate"));
    m_pButtonMenuQuads_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuQuads_);

    m_pButtonMenuShade_ = new coButtonMenuItem(coTranslator::coTranslate("Shade"));
    m_pButtonMenuShade_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuShade_);

    m_pButtonMenuTransparent_ = new coButtonMenuItem(coTranslator::coTranslate("Transparent"));
    m_pButtonMenuTransparent_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuTransparent_);

    m_pButtonMenuTextur1_ = new coButtonMenuItem(coTranslator::coTranslate("Textur 1"));
    m_pButtonMenuTextur1_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuTextur1_);

    m_pButtonMenuTextur2_ = new coButtonMenuItem(coTranslator::coTranslate("Textur 2"));
    m_pButtonMenuTextur2_->setMenuListener(this);
    m_pObjectMenu6->add(m_pButtonMenuTextur2_);

    m_pCheckboxMenuPlane3 = new coCheckboxMenuItem(coTranslator::coTranslate("zeige Parameterebene"), checkboxPlaneState);
    m_pCheckboxMenuPlane3->setMenuListener(this);
    m_pObjectMenu6->add(m_pCheckboxMenuPlane3);
    //Menu-Schiefschnitt-Animation
    m_pButtonMenuNormalschnitt = new coButtonMenuItem(coTranslator::coTranslate("Normalschnitt-Animation"));
    m_pButtonMenuNormalschnitt->setMenuListener(this);
    m_pObjectMenu8->add(m_pButtonMenuNormalschnitt);

    m_pCheckboxMenuHauptKrRichtungen = new coCheckboxMenuItem(coTranslator::coTranslate("Hauptkruemmungsrichtungen"), false);
    m_pCheckboxMenuHauptKrRichtungen->setMenuListener(this);
    m_pObjectMenu9->add(m_pCheckboxMenuHauptKrRichtungen);

    m_pCheckboxMenuHauptKr = new coCheckboxMenuItem(coTranslator::coTranslate("Hauptkruemmungen"), false);
    m_pCheckboxMenuHauptKr->setMenuListener(this);
    m_pObjectMenu9->add(m_pCheckboxMenuHauptKr);

    m_pCheckboxMenuSchmiegTang = new coCheckboxMenuItem(coTranslator::coTranslate("Schmiegtangenten"), false);
    m_pCheckboxMenuSchmiegTang->setMenuListener(this);
    m_pObjectMenu9->add(m_pCheckboxMenuSchmiegTang);

    m_pButtonMenuSchiefschnitt = new coButtonMenuItem(coTranslator::coTranslate("Schiefschnitt-Animation"));
    m_pButtonMenuSchiefschnitt->setMenuListener(this);
    m_pObjectMenu10->add(m_pButtonMenuSchiefschnitt);

    m_pCheckboxMenuDarstMeusnierKugel = new coCheckboxMenuItem(coTranslator::coTranslate("Meusnier-Kugel transparent"), false);
    m_pCheckboxMenuDarstMeusnierKugel->setMenuListener(this);
    m_pObjectMenu10->add(m_pCheckboxMenuDarstMeusnierKugel);

    m_pCheckboxMenuTransparent = new coCheckboxMenuItem(coTranslator::coTranslate("Flaeche transparent"), true);
    m_pCheckboxMenuTransparent->setMenuListener(this);
    m_pObjectMenu10->add(m_pCheckboxMenuTransparent);

    m_pButtonMenuClearNormalschnitt = new coButtonMenuItem(coTranslator::coTranslate("Clear"));
    m_pButtonMenuClearNormalschnitt->setMenuListener(this);
    m_pObjectMenu10->add(m_pButtonMenuClearNormalschnitt);
}

//---------------------------------------------------
//Implements ParametricSurfaces::preFrame()  //für Anzeige der Animation
//---------------------------------------------------
void ParametricSurfaces::preFrame()
{

    if (m_presentationStep == 9)
    {
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);
        ResetSliderUV();
        if (cover->getIntersectedNode() == mp_Surf)
        {
            coInteractionManager::the()->registerInteraction(interactionA);
            if (interactionA->wasStarted())
            {
                std::cerr << "Punkt auswaehlen" << std::endl;
                _startPickPos = cover->getIntersectionHitPointWorld(); //Punktauswahl
                //Untransformierten Punkt berechnen
                osg::Vec3 start = _startPickPos;
                start = osg::Matrixd::inverse(cover->getXformMat()).preMult(start);
                start /= VRSceneGraph::instance()->scaleFactor();
                _startPickPos = start;
                if (mp_NormalschnittAnimation->m_pfeil_exist)
                {
                    if (!(mp_NormalschnittAnimation->KruemmungsMethode_aktiv()))
                    {
                        mp_NormalschnittAnimation->Remove_Pfeil(Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                        mp_NormalschnittAnimation->Create_Pfeil(_startPickPos, mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                    }
                    else
                    {
                        clearNormalAnim();
                        mp_NormalschnittAnimation->Remove_Pfeil(Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                        mp_NormalschnittAnimation->Create_Pfeil(_startPickPos, mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                    }
                }
                else
                {
                    mp_NormalschnittAnimation->Create_Pfeil(_startPickPos, mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                }
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
    if (m_pCheckboxMenuHauptKr->getState() == true)
    {
        labelKr1->setPosition(mp_NormalschnittAnimation->position1 * cover->getBaseMat());
        labelKr2->setPosition(mp_NormalschnittAnimation->position2 * cover->getBaseMat());
    }
}
void ParametricSurfaces::changePresentationStep()
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

    //Definition of the presentation steps
    switch (m_presentationStep)
    {
    //Paramtetric Surfaces
    case 0:
        //imageMenu_->show();
        std::cerr << "Step 0__________________" << std::endl;
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        setMenuVisible(0);
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        //coInteractionManager::the()->unregisterInteraction(interactionA);
        break;

    //How does a surface develop?
    case 1:
        std::cerr << "Step 1___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        setMenuVisible(1);
        //Surfmode: Points
        changeMode = 0;
        Change_Mode(mp_Plane, 0);
        Change_Mode(mp_Surf, 0);
        //coInteractionManager::the()->unregisterInteraction(interactionA);
        //VRSceneGraph::instance()->viewAll();
        break;

    //Choose a surface
    case 2:
        std::cerr << "Step 2___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        setMenuVisible(2);
        // coInteractionManager::the()->unregisterInteraction(interactionA);
        //VRSceneGraph::instance()->viewAll();
        break;

    //different views of the surface
    case 3:
        std::cerr << "Step 3___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        m_pCheckboxMenuPlane->setState(checkboxPlaneState);
        setMenuVisible(3);
        //coInteractionManager::the()->unregisterInteraction(interactionA);
        //VRSceneGraph::instance()->viewAll();
        break;

    //Change the shape
    case 4:
        std::cerr << "Step 4___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        m_pCheckboxMenuPlane2->setState(checkboxPlaneState);
        if (actualizeSlider)
        { //sliderveraenderung von case 8
            m_sliderValueA = m_pSliderMenuA_->getValue();
            m_sliderValueB = m_pSliderMenuB_->getValue();
            m_sliderValueC = m_pSliderMenuC_->getValue();
            m_pSliderMenuA->setValue(m_sliderValueA);
            m_pSliderMenuB->setValue(m_sliderValueB);
            m_pSliderMenuC->setValue(m_sliderValueC);
        }
        else if (sliderStart)
        { //slider in Ausgangsstellung
            m_sliderValueA = sliderStartValueA;
            m_sliderValueB = sliderStartValueB;
            m_sliderValueC = sliderStartValueC;

            m_pSliderMenuA->setValue(m_sliderValueA);
            m_pSliderMenuB->setValue(m_sliderValueB);
            m_pSliderMenuC->setValue(m_sliderValueC);
        }
        else
        {
            m_sliderValueA = m_pSliderMenuA->getValue();
            m_sliderValueB = m_pSliderMenuB->getValue();
            m_sliderValueC = m_pSliderMenuC->getValue();

            m_pSliderMenuA->setValue(m_sliderValueA);
            m_pSliderMenuB->setValue(m_sliderValueB);
            m_pSliderMenuC->setValue(m_sliderValueC);
        }
        setMenuVisible(4);
        sliderStart = false;
        actualizeSlider = false;
        actualizeSliderStep8 = false;
        //VRSceneGraph::instance()->viewAll();
        break;
    //Change the boundary
    case 5:
        std::cerr << "Step 5___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        setMenuVisible(5);
        //VRSceneGraph::instance()->viewAll();
        break;

    //Choose a surface curve
    case 6:
        std::cerr << "Step 6___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        //coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        setMenuVisible(6);
        // VRSceneGraph::instance()->viewAll();
        break;

    //Animation of spheres on the surface curve
    case 7:
        std::cerr << "Step 7___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        //VRSceneGraph::instance()->viewAll();
        m_pCheckboxMenuPlane3->setState(checkboxPlaneState);
        setMenuVisible(7);
        break;

    //Curvatures
    case 8:
        std::cerr << "Step 8___________________________________" << std::endl;
        // imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        //coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        // VRSceneGraph::instance()->viewAll();
        if (actualizeSliderStep8)
        {
            m_sliderValueA_ = m_pSliderMenuA_->getValue();
            m_sliderValueB_ = m_pSliderMenuB_->getValue();
            m_sliderValueC_ = m_pSliderMenuC_->getValue();
            m_pSliderMenuA_->setValue(m_sliderValueA_);
            m_pSliderMenuB_->setValue(m_sliderValueB_);
            m_pSliderMenuC_->setValue(m_sliderValueC_);
        }
        else
        {
            m_sliderValueA_ = m_pSliderMenuA->getValue();
            m_sliderValueB_ = m_pSliderMenuB->getValue();
            m_sliderValueC_ = m_pSliderMenuC->getValue();
            m_pSliderMenuA_->setValue(m_sliderValueA_);
            m_pSliderMenuB_->setValue(m_sliderValueB_);
            m_pSliderMenuC_->setValue(m_sliderValueC_);
        }
        setMenuVisible(8);
        actualizeSlider = true;
        actualizeSliderStep8 = true;
        break;

    //Choose a point
    case 9:
        std::cerr << "Step 9___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() | Isect::Intersection | Isect::Pick); //mit Fläche was machbar
        //coInteractionManager::the()->registerInteraction(interactionA);
        setMenuVisible(9);
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        break;

    //Normalschnittanimation
    case 10:
        std::cerr << "Step 10___________________________________" << std::endl;
        //imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        setMenuVisible(10);
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        if ((mp_NormalschnittAnimation->KruemmungsMethode_aktiv() && m_pCheckboxMenuTransparent->getState() == true))
        {
            changeMode = 6;
            Change_Mode(mp_Plane, 6);
            Change_Mode(mp_Surf, 6);
        }
        break;

    //Hauptkruemmungen
    case 11:
        std::cerr << "Step 11___________________________________" << std::endl;
        //imageMenu_->show();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        setMenuVisible(11);
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        if ((mp_NormalschnittAnimation->KruemmungsMethode_aktiv() && m_pCheckboxMenuTransparent->getState() == true))
        {
            Change_Mode(mp_Plane, 6);
            Change_Mode(mp_Surf, 6);
        }
        break;

    //Schiefschnittanimation
    case 12:
        std::cerr << "Step 12___________________________________" << std::endl;
        // imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        setMenuVisible(12);
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        if (mp_NormalschnittAnimation->m_pfeil_exist == false && changeMode != 6)
            m_pCheckboxMenuTransparent->setState(false);
        if ((mp_NormalschnittAnimation->KruemmungsMethode_aktiv() && m_pCheckboxMenuTransparent->getState() == true))
        {
            Change_Mode(mp_Plane, 6);
            Change_Mode(mp_Surf, 6);
        }
        break;

    //summary
    case 13:
        std::cerr << "Step 13___________________________________" << std::endl;
        //  imageMenu_->show();
        labelKr1->hide();
        labelKr2->hide();
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar, aber Menu
        setMenuVisible(13);
        // coInteractionManager::the()->unregisterInteraction(interactionA);//kann Szene wieder drehen
        break;
    }
}
//---------------------------------------------------
//Implements ParametricSurfaces::guiToRenderMsg(const grmsg::coGRMsg &msg) 
//---------------------------------------------------
void ParametricSurfaces::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- ParametricSurfacesPlugin coVRGuiToRenderMsg %s\n", msg.getString().c_str());

    //string fullMsg_ = fullMsg;
    if (msg.isValid())
    {
        switch (msg.getType())
        {
        case coGRMsg::KEYWORD:
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
                //fprintf(stderr,"\n--- ParametricCurvesPlugin coVRGuiToRenderMsg keyword %s\n", msg);
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
        break;
        case coGRMsg::SET_DOCUMENT_POSITION:
        {
            auto &setDocPositionMsg = msg.as<coGRSetDocPositionMsg>();
            setDocPositionMsg.getPosition(x_posDocView, y_posDocView, z_posDocView);
            createMenu3();
            OSGVruiMatrix *t, *r, *mat;
            r = new OSGVruiMatrix();
            t = new OSGVruiMatrix();
            mat = new OSGVruiMatrix();

            t->makeTranslate(x_posDocView, y_posDocView + 0.02, z_posDocView);
            r->makeEuler(0, 90, 0);
            //t.print(0, 1, "t: ",stderr);
            mat = dynamic_cast<OSGVruiMatrix *>(r->mult(t));
            imageMenu_->setTransformMatrix(mat);
            //imageMenu_->setScale(cover->getSceneSize()/2500);
        }
        break;
        case coGRMsg::SET_DOCUMENT_SCALE:
        {
            auto &setDocScaleMsg = msg.as<coGRSetDocScaleMsg>();
            scale_DocView = setDocScaleMsg.getScale();
            createMenu3();
            imageMenu_->setScale(scale_DocView);
        }
        break;
        case coGRMsg::SET_DOCUMENT_PAGESIZE:
        {
            changesize = true;
            auto &setDocPageSizeMsg = msg.as<coGRSetDocPageSizeMsg>();
            hsize_DocView = setDocPageSizeMsg.getHSize();
            vsize_DocView = setDocPageSizeMsg.getVSize();
            createMenu3();
            //imageItemList_->setSize(hsize_DocView, vsize_DocView);
        }
        break;
        default:
            break;
        }
    }
}
void ParametricSurfaces::clearNormalAnim()
{
    mp_NormalschnittAnimation->Clear(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
    m_pCheckboxMenuDarstMeusnierKugel->setState(false);
    m_pCheckboxMenuHauptKr->setState(false);
    m_pCheckboxMenuHauptKrRichtungen->setState(false);
    m_pCheckboxMenuSchmiegTang->setState(false);
    if (changeMode == 6)
        m_pCheckboxMenuTransparent->setState(true);
    else
        m_pCheckboxMenuTransparent->setState(false);
    labelKr1->hide();
    labelKr2->hide();
}
//---------------------------------------------------
//Implements ParametricSurfaces::menuEvent(coMenuItem *iMenuItem)
//---------------------------------------------------
void ParametricSurfaces::menuReleaseEvent(coMenuItem *iMenuItem) //Slider zeigt erst bei loslassen //wird automatisch aufgerufen
{
    if (iMenuItem == m_pSliderMenuA)
    {
        std::cerr << "Change A" << std::endl;
        clearNormalAnim();
        m_sliderValueA = m_pSliderMenuA->getValue();
        //Interval with the A parameter(radius)
        Change_Radius(mp_Surf, m_sliderValueA);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuB)
    {
        std::cerr << "Change B" << std::endl;
        clearNormalAnim();
        m_sliderValueB = m_pSliderMenuB->getValue();
        //Interval with the B parameter(length)
        Change_Length(mp_Surf, m_sliderValueB);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuC)
    {
        std::cerr << "Change C" << std::endl;
        clearNormalAnim();
        m_sliderValueC = m_pSliderMenuC->getValue();
        //Interval with the C parameter(height)
        Change_Height(mp_Surf, m_sliderValueC);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuA_)
    {
        std::cerr << "Change A_" << std::endl;
        clearNormalAnim();
        m_sliderValueA_ = m_pSliderMenuA_->getValue();
        //Interval with the A parameter(radius)
        Change_Radius(mp_Surf, m_sliderValueA_);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuB_)
    {
        std::cerr << "Change B_" << std::endl;
        clearNormalAnim();
        m_sliderValueB_ = m_pSliderMenuB_->getValue();
        //Interval with the B parameter(length)
        Change_Length(mp_Surf, m_sliderValueB_);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuC_)
    {
        std::cerr << "Change C_" << std::endl;
        clearNormalAnim();
        m_sliderValueC_ = m_pSliderMenuC_->getValue();
        //Interval with the C parameter(height)
        Change_Height(mp_Surf, m_sliderValueC_);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuUa)
    {
        std::cerr << "Change Ua" << std::endl;
        clearNormalAnim();
        m_sliderValueUa = m_pSliderMenuUa->getValue();
        //Interval with the u-beginning parameter
        Change_Ua(mp_Surf, mp_Plane, m_sliderValueUa);

        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);
        ResetSliderUV();

        if (m_pCheckboxMenuNormals->getState() == true)
            Disable_Normals(mp_Surf, false);

        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuUe)
    {
        std::cerr << "Change Ue" << std::endl;
        clearNormalAnim();
        m_sliderValueUe = m_pSliderMenuUe->getValue();
        //Interval with the u-ending parameter
        Change_Ue(mp_Surf, mp_Plane, m_sliderValueUe);

        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);
        ResetSliderUV();

        if (m_pCheckboxMenuNormals->getState() == true)
            Disable_Normals(mp_Surf, false);

        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuVa)
    {
        std::cerr << "Change Va" << std::endl;
        clearNormalAnim();
        m_sliderValueVa = m_pSliderMenuVa->getValue();
        //Interval with the v-beginning parameter
        Change_Va(mp_Surf, mp_Plane, m_sliderValueVa);

        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);
        ResetSliderUV();

        if (m_pCheckboxMenuNormals->getState() == true)
            Disable_Normals(mp_Surf, false);

        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuVe)
    {
        std::cerr << "Change Ve" << std::endl;
        clearNormalAnim();
        m_sliderValueVe = m_pSliderMenuVe->getValue();
        //Interval with the v-ending parameter
        Change_Ve(mp_Surf, mp_Plane, m_sliderValueVe);

        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);

        ResetSliderUV();

        if (m_pCheckboxMenuNormals->getState() == true)
            Disable_Normals(mp_Surf, false);

        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
    if (iMenuItem == m_pSliderMenuLOD)
    {
        std::cerr << "Change LOD" << std::endl;
        clearNormalAnim();

        // m_sliderValueLOD_old = m_sliderValueLOD;
        m_sliderValueLOD = m_pSliderMenuLOD->getValue();
        //Interval with the LOD parameter
        Change_Geometry(mp_Surf, m_sliderValueLOD);
        Change_Geometry(mp_Plane, m_sliderValueLOD);

        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);

        ResetSliderUV();
        if (m_pCheckboxMenuNormals->getState() == true)
            Disable_Normals(mp_Surf, false);

        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
    }
}
//---------------------------------------------------
//Implements ParametricSurfaces::menuEvent(coMenuItem *iMenuItem)
//---------------------------------------------------
void ParametricSurfaces::menuEvent(coMenuItem *iMenuItem)
{
    //Slider
    if (iMenuItem == m_pSliderMenuU)
    {
        std::cerr << "Change u" << std::endl;
        clearNormalAnim();
        if (constyp == CNOTHING)
        {
            m_pSliderMenuU->setValue(0.0);
        }
        else
        {
            m_sliderValueU = m_pSliderMenuU->getValue();
            int oldm = m_iscreateAnimation;
            Change_ConsPosition(mp_Surf);

            m_iscreateAnimation = oldm;
            Change_ConsPosition(mp_Plane);

            m_sliderValueU_old = m_sliderValueU;
        }
    }
    if (iMenuItem == m_pSliderMenuV)
    {
        std::cerr << "Change v" << std::endl;
        clearNormalAnim();
        if (constyp == CNOTHING)
        {
            m_pSliderMenuV->setValue(0.0);
        }
        else
        {
            m_sliderValueV = m_pSliderMenuV->getValue();
            int oldm = m_iscreateAnimation;
            Change_ConsPosition(mp_Surf);
            m_iscreateAnimation = oldm;
            Change_ConsPosition(mp_Plane);
            m_sliderValueV_old = m_sliderValueV;
        }
    }
    //
    if (iMenuItem == m_pButtonMenuPoints)
    {
        std::cerr << "Points" << std::endl;
        //Surfmode: Points
        changeMode = 0;
        Change_Mode(mp_Plane, 0);
        Change_Mode(mp_Surf, 0);
    }
    if (iMenuItem == m_pButtonMenuLines)
    {
        std::cerr << "Lines" << std::endl;
        //Surfmode: Lines
        changeMode = 1;
        Change_Mode(mp_Plane, 1);
        Change_Mode(mp_Surf, 1);
    }
    if (iMenuItem == m_pButtonMenuTriangles)
    {
        std::cerr << "Triangles" << std::endl;
        //Surfmode: Triangles
        changeMode = 2;
        Change_Mode(mp_Plane, 2);
        Change_Mode(mp_Surf, 2);
    }
    if (iMenuItem == m_pButtonMenuQuads)
    {
        std::cerr << "Quads" << std::endl;
        //Surfmode: Quads
        changeMode = 3;
        Change_Mode(mp_Plane, 3);
        Change_Mode(mp_Surf, 3);
    }
    if (iMenuItem == m_pButtonMenuShade)
    {
        std::cerr << "Shade" << std::endl;
        //Surfmode: Shade
        changeMode = 4;
        Change_Mode(mp_Plane, 4);
        Change_Mode(mp_Surf, 4);
    }
    if (iMenuItem == m_pButtonMenuTransparent)
    {
        std::cerr << "Transparent" << std::endl;
        //Surfmode: Transparent
        changeMode = 6;
        Change_Mode(mp_Plane, 6);
        Change_Mode(mp_Surf, 6);
    }
    if (iMenuItem == m_pButtonMenuPoints_)
    {
        std::cerr << "Points_" << std::endl;
        //Surfmode: Points
        changeMode = 0;
        Change_Mode(mp_Plane, 0);
        Change_Mode(mp_Surf, 0);
    }
    if (iMenuItem == m_pButtonMenuLines_)
    {
        std::cerr << "Lines_" << std::endl;
        //Surfmode: Lines
        changeMode = 1;
        Change_Mode(mp_Plane, 1);
        Change_Mode(mp_Surf, 1);
    }
    if (iMenuItem == m_pButtonMenuTriangles_)
    {
        std::cerr << "Triangles_" << std::endl;
        //Surfmode: Triangles
        changeMode = 2;
        Change_Mode(mp_Plane, 2);
        Change_Mode(mp_Surf, 2);
    }
    if (iMenuItem == m_pButtonMenuQuads_)
    {
        std::cerr << "Quads_" << std::endl;
        //Surfmode: Quads
        changeMode = 3;
        Change_Mode(mp_Plane, 3);
        Change_Mode(mp_Surf, 3);
    }
    if (iMenuItem == m_pButtonMenuShade_)
    {
        std::cerr << "Shade_" << std::endl;
        //Surfmode: Shade
        changeMode = 4;
        Change_Mode(mp_Plane, 4);
        Change_Mode(mp_Surf, 4);
    }
    if (iMenuItem == m_pButtonMenuTransparent_)
    {
        std::cerr << "Transparent_" << std::endl;
        //Surfmode: Transparent
        changeMode = 6;
        Change_Mode(mp_Plane, 6);
        Change_Mode(mp_Surf, 6);
    }
    if (iMenuItem == m_pCheckboxMenuGauss)
    {
        std::cerr << "Gauss" << std::endl;
        //Surfmode: Gauss-Kruemmung
        //change_mode(mp_plane,7);
        //change_mode(mp_surf,7);
        if (m_pCheckboxMenuMean->getState() == true)
            m_pCheckboxMenuMean->setState(false);

        if (m_pCheckboxMenuGauss->getState() == false)
        {
            Change_Mode(mp_Plane, changeMode);
            Change_Mode(mp_Surf, changeMode);
        }
        else
        {
            Gauss_Curvature_Mode(mp_Plane);
            Gauss_Curvature_Mode(mp_Surf);
        }
    }
    if (iMenuItem == m_pCheckboxMenuMean)
    {
        std::cerr << "Mittlere" << std::endl;
        //Surfmode: Mean
        //Change_Mode(mp_Plane,8);
        //Change_Mode(mp_Surf,8);
        if (m_pCheckboxMenuGauss->getState() == true)
            m_pCheckboxMenuGauss->setState(false);
        if (m_pCheckboxMenuMean->getState() == false)
        {
            Change_Mode(mp_Plane, changeMode);
            Change_Mode(mp_Surf, changeMode);
        }
        else
        {
            Mean_Curvature_Mode(mp_Plane);
            Mean_Curvature_Mode(mp_Surf);
        }
    }
    const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("value", "COVER.Plugin.ParametricSurfaces.DataPath") + "/Data/";

    string surffiledir = m_imagePath + "Surfaces/";
    if (iMenuItem == m_pButtonMenuKegel)
    {
        std::cerr << "Kegel" << std::endl;
        //Surface: Kegel
        m_formula = m_numPresentationSteps + 1;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Kegel.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuKugel)
    {
        std::cerr << "Kugel" << std::endl;
        //Surface: Kugel
        m_formula = m_numPresentationSteps + 2;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Kugel.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuMoebius)
    {
        std::cerr << "Moebius" << std::endl;
        //Surface: Moebius
        m_formula = m_numPresentationSteps + 3;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Moebius.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuParaboloid)
    {
        std::cerr << "Paraboloid" << std::endl;
        //Surface: Paraboloid
        m_formula = m_numPresentationSteps + 4;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Paraboloid.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuZylinder)
    {
        std::cerr << "Zylinder" << std::endl;
        //Surface: Zylinder
        m_formula = m_numPresentationSteps + 5;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Zylinder.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuBonan)
    {
        //Surface: Bonan
        m_formula = m_numPresentationSteps + 6;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Bonan.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuBoy)
    {
        //Surface: Boy
        m_formula = m_numPresentationSteps + 7;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Boy.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuCrossCap)
    {
        //Surface: CrossCap
        m_formula = m_numPresentationSteps + 8;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "CrossCap.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuDini)
    {
        //Surface: Dini
        m_formula = m_numPresentationSteps + 9;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Dini.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuEnneper)
    {
        //Surface: Enneper
        m_formula = m_numPresentationSteps + 10;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Enneper.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuHelicalTorus)
    {
        //Surface: HelicalTorus
        m_formula = m_numPresentationSteps + 11;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "HelicalTorus.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuKatenoid)
    {
        //Surface: Katenoid
        m_formula = m_numPresentationSteps + 12;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Katenoid.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuKlein)
    {
        //Surface: Klein
        m_formula = m_numPresentationSteps + 13;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Klein.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuKuen)
    {
        //Surface: Kuen
        m_formula = m_numPresentationSteps + 14;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Kuen.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuPluecker)
    {
        //Surface: Pluecker
        m_formula = m_numPresentationSteps + 15;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Pluecker.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    // if(iMenuItem == m_pButtonMenuPseudoSphere){
    //  //Surface: PseudoSphere
    //clearNormalAnim();
    //int image = m_Imagelistcounter;
    //string surfname = "PseudoSphere.txt";
    //  string surffilepath = surffiledir + surfname;
    //  std::string str = HfT_replace_string_in_string(surfname,".txt","");
    //  m_Surfname = str;
    //  Read_Surface(mp_GeoGroup,surffilepath);
    //  m_Imagelistcounter = image;
    //  Change_Image(mp_Surf);
    //  Change_Image(mp_Plane);
    //  VRSceneGraph::instance()->viewAll();
    // }
    if (iMenuItem == m_pButtonMenuRevolution)
    {
        //Surface: Revolution
        m_formula = m_numPresentationSteps + 16;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Revolution.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuRoman)
    {
        //Surface: Roman
        m_formula = m_numPresentationSteps + 17;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Roman.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuShell)
    {
        //Surface: Shell
        m_formula = m_numPresentationSteps + 18;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Shell.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuSnake)
    {
        //Surface: Snake
        m_formula = m_numPresentationSteps + 19;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Snake.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuTrumpet)
    {
        //Surface: Trumpet
        m_formula = m_numPresentationSteps + 20;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "Trumpet.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    if (iMenuItem == m_pButtonMenuTwistedSphere)
    {
        //Surface: TwistedSphere
        m_formula = m_numPresentationSteps + 21;
        clearNormalAnim();
        m_pCheckboxMenuNormals->setState(false);
        checkboxPlaneState = true;
        m_pCheckboxMenuNatbound->setState(true);
        int image = m_Imagelistcounter;
        string surfname = "TwistedSphere.txt";
        string surffilepath = surffiledir + surfname;
        std::string str = HfT_replace_string_in_string(surfname, ".txt", "");
        m_Surfname = str;
        Read_Surface(mp_GeoGroup, surffilepath);
        m_Imagelistcounter = image;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
        if (m_pCheckboxMenuNatbound->getState() == true)
        {
            Disable_Boundary(true);
            Disable_Boundary(false);
        }
        VRSceneGraph::instance()->viewAll();
    }
    //Cons
    if (iMenuItem == m_pButtonMenuUCenter)
    {
        std::cerr << "uCenter" << std::endl;
        //Cons: CUCENTER
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CUCENTER);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CUCENTER);
        constyp = CUCENTER;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuVCenter)
    {
        std::cerr << "vCenter" << std::endl;
        //Cons: CVCENTER
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CVCENTER);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CVCENTER);
        constyp = CVCENTER;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuDiagonal)
    {
        std::cerr << "Diagonal" << std::endl;
        //Cons: CDIAGONAL
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CDIAGONAL);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CDIAGONAL);
        constyp = CDIAGONAL;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuTriangle)
    {
        std::cerr << "Dreieck" << std::endl;
        //Cons: CTRIANGLE
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CTRIANGLE);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CTRIANGLE);
        constyp = CTRIANGLE;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuEllipse)
    {
        std::cerr << "Ellipse" << std::endl;
        //Cons: CELLIPSE
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CELLIPSE);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CELLIPSE);
        constyp = CELLIPSE;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuSquare)
    {
        std::cerr << "Rechteck" << std::endl;
        //Cons: CSQUARE
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CSQUARE);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CSQUARE);
        constyp = CSQUARE;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuNatbound)
    {
        std::cerr << "Rand" << std::endl;
        //Cons: CNATBOUND
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNATBOUND);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNATBOUND);
        constyp = CNATBOUND;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuNothing)
    {
        std::cerr << "keine" << std::endl;
        //Cons: CNOTHING
        clearNormalAnim();
        int anim = m_iscreateAnimation;
        Change_Cons(mp_Surf, CNOTHING);
        m_iscreateAnimation = anim;
        Change_Cons(mp_Plane, CNOTHING);
        constyp = CNOTHING;
        ResetSliderUV();
    }
    if (iMenuItem == m_pButtonMenuTextur1)
    {
        std::cerr << "Textur1" << std::endl;
        //Textur 1
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        m_Imagelistcounter = 1;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
    }
    if (iMenuItem == m_pButtonMenuTextur2)
    {
        std::cerr << "Textur2" << std::endl;
        //Textur 2
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        m_Imagelistcounter = 2;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
    }
    if (iMenuItem == m_pButtonMenuTextur1_)
    {
        std::cerr << "Textur1_" << std::endl;
        //Textur 1
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        m_Imagelistcounter = 1;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
    }
    if (iMenuItem == m_pButtonMenuTextur2_)
    {
        std::cerr << "Textur2_" << std::endl;
        //Textur 2
        changeMode = 5;
        Change_Mode(mp_Plane, 5);
        Change_Mode(mp_Surf, 5);
        m_Imagelistcounter = 2;
        Change_Image(mp_Surf);
        Change_Image(mp_Plane);
    }
    //Animation
    if (iMenuItem == m_pButtonMenuAnimationSphere)
    {
        std::cerr << "Animation Kugeln" << std::endl;
        //Animation starten
        if (m_iscreateAnimation < m_maxcreateAnimation)
        {
            int oldm = m_iscreateAnimation;
            Sphere_AnimationCreate(mp_Surf);
            m_iscreateAnimation = oldm;
            Sphere_AnimationCreate(mp_Plane);
        }
    }
    if (iMenuItem == m_pButtonMenuAnimationOff)
    {
        std::cerr << "Animation beenden" << std::endl;
        //Animation beenden
        if (m_iscreateAnimation)
        {
            int animation = m_iscreateAnimation;
            Remove_Animation();
            m_iscreateAnimation = animation;
            Remove_Animation();
        }
    }
    //Normalschnitt-Animation
    if (iMenuItem == m_pButtonMenuNormalschnitt)
    {
        std::cerr << "Normalschnittanim" << std::endl;
        //Normalschnitte erzeugen
        if (mp_NormalschnittAnimation->m_normalwinkel < 180)
        {
            m_pCheckboxMenuTransparent->setState(true);
            mp_NormalschnittAnimation->Normalschnitt_Animation(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
        }
        if (mp_NormalschnittAnimation->m_normalwinkel > 180)
        {
            if (mp_NormalschnittAnimation->m_normalebene_exist)
            {
                m_pCheckboxMenuTransparent->setState(true);
                mp_NormalschnittAnimation->Remove_NormalEbenen(Get_MatrixTransformNode_Parent(mp_Surf));
            }
            else
            {
                if (changeMode == 6)
                    m_pCheckboxMenuTransparent->setState(true);
                else
                    m_pCheckboxMenuTransparent->setState(false);
                mp_NormalschnittAnimation->Remove_Schnittkurve(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
            }
        }
        if (mp_NormalschnittAnimation->m_normalwinkel == 180)
        {
            m_pCheckboxMenuTransparent->setState(true);
            mp_NormalschnittAnimation->m_normalwinkel = mp_NormalschnittAnimation->m_normalwinkel + mp_NormalschnittAnimation->m_normalwinkel_schritt;
        }
    }
    if (iMenuItem == m_pCheckboxMenuHauptKrRichtungen)
    {
        std::cerr << "hauptkrrichtungen" << std::endl;
        //Hauptkruemmungsrichtungen anzeigen
        if (mp_NormalschnittAnimation->m_pfeil_exist)
        {
            m_pCheckboxMenuTransparent->setState(true);
            if (m_pCheckboxMenuHauptKrRichtungen->getState() == true)
            {
                if (mp_NormalschnittAnimation->m_pfeil_exist && (mp_NormalschnittAnimation->m_hauptkruemmungsrichtung_exist == false))
                {
                    m_pCheckboxMenuTransparent->setState(true);
                    mp_NormalschnittAnimation->Hauptkruemmungsrichtungen(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                }
            }
            else
            {
                if (mp_NormalschnittAnimation->m_hauptkruemmungsrichtung_exist)
                {
                    if (changeMode == 6)
                        m_pCheckboxMenuTransparent->setState(true);
                    else
                        m_pCheckboxMenuTransparent->setState(false);
                    mp_NormalschnittAnimation->Remove_Hauptkruemmungsrichtungen(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                }
            }
        }
        else
        {
            m_pCheckboxMenuHauptKrRichtungen->setState(false);
        }
    }
    if (iMenuItem == m_pCheckboxMenuHauptKr)
    {
        std::cerr << "Hauptkruemmmungen" << std::endl;
        //Hauptkruemmungen anzeigen
        if (mp_NormalschnittAnimation->m_pfeil_exist)
        {
            if (m_pCheckboxMenuHauptKr->getState() == true /*&& mp_NormalschnittAnimation->m_pfeil_exist)*/)
            {
                if (mp_NormalschnittAnimation->m_pfeil_exist && (mp_NormalschnittAnimation->m_hauptkruemmung_exist == false))
                {
                    m_pCheckboxMenuTransparent->setState(true);
                    mp_NormalschnittAnimation->Hauptkruemmungen(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));

                    labelKr1->setString(mp_NormalschnittAnimation->text_1.c_str());
                    labelKr2->setString(mp_NormalschnittAnimation->text_2.c_str());
                    labelKr1->setFGColor(mp_NormalschnittAnimation->color1);
                    labelKr2->setFGColor(mp_NormalschnittAnimation->color2);
                    labelKr1->show();
                    labelKr2->show();
                    labelKr1->showLine();
                    labelKr2->showLine();
                }
            }
            else
            { //ausschalten
                if (m_pCheckboxMenuHauptKr->getState() == false && mp_NormalschnittAnimation->m_hauptkruemmung_exist)
                {
                    if (changeMode == 6)
                        m_pCheckboxMenuTransparent->setState(true);
                    else
                        m_pCheckboxMenuTransparent->setState(false);
                    mp_NormalschnittAnimation->Remove_Hauptkruemmungen(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                    labelKr1->hide();
                    labelKr2->hide();
                }
                if (mp_NormalschnittAnimation->KruemmungsMethode_aktiv())
                    m_pCheckboxMenuTransparent->setState(true);
            }
        }
        else
        { //bleibt ausgeschalten,wenn kein Punkt ausgewaehlt wurde
            m_pCheckboxMenuHauptKr->setState(false);
            labelKr1->hide();
            labelKr2->hide();
        }
    }
    if (iMenuItem == m_pCheckboxMenuSchmiegTang)
    {
        std::cerr << "Schmiegtangenten" << std::endl;
        //Schmiegtangenten anzeigen
        if (mp_NormalschnittAnimation->m_pfeil_exist)
        {

            if (m_pCheckboxMenuSchmiegTang->getState() == true)
            {
                if (mp_NormalschnittAnimation->m_pfeil_exist && (mp_NormalschnittAnimation->m_schmiegtangente_exist == false))
                {
                    mp_NormalschnittAnimation->Schmiegtangenten(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                    if (mp_NormalschnittAnimation->m_schmiegtangente_exist)
                    {
                        m_pCheckboxMenuTransparent->setState(true);
                    }
                }
            }
            else
            {
                if (mp_NormalschnittAnimation->m_schmiegtangente_exist)
                {
                    if (changeMode == 6)
                        m_pCheckboxMenuTransparent->setState(true);
                    else
                        m_pCheckboxMenuTransparent->setState(false);
                    mp_NormalschnittAnimation->Remove_Schmiegtangenten(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                }
            }
        }
        else
        {
            m_pCheckboxMenuSchmiegTang->setState(false);
        }
    }
    if (iMenuItem == m_pButtonMenuSchiefschnitt)
    {
        std::cerr << "Schiefschnittanim" << std::endl;
        //Schmiefschnittanimation
        if (mp_NormalschnittAnimation->m_pfeil_exist == true)
        {
            if (mp_NormalschnittAnimation->m_schiefwinkel < 90)
            {
                m_pCheckboxMenuTransparent->setState(true);
                mp_NormalschnittAnimation->Schiefschnitt_Animation(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
            }
            if (mp_NormalschnittAnimation->m_schiefwinkel > 90)
            {
                if (mp_NormalschnittAnimation->m_kruemmungskreise_exist)
                {
                    if (changeMode == 6)
                        m_pCheckboxMenuTransparent->setState(true);
                    else
                        m_pCheckboxMenuTransparent->setState(false);
                    mp_NormalschnittAnimation->Remove_MeusnierKugel(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                    mp_NormalschnittAnimation->Remove_Kruemmungskreise(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                    mp_NormalschnittAnimation->m_schiefwinkel = -mp_NormalschnittAnimation->m_schiefwinkel_schritt;
                }
            }
            if (mp_NormalschnittAnimation->m_schiefwinkel == 90)
            {
                m_pCheckboxMenuTransparent->setState(true);
                mp_NormalschnittAnimation->Meusnier_Kugel(mp_Surf, 1, Get_MatrixTransformNode_Parent(mp_Surf));
                mp_NormalschnittAnimation->Remove_SchiefEbenen(Get_MatrixTransformNode_Parent(mp_Surf));
                mp_NormalschnittAnimation->Remove_SchiefSchnittkurve(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf), Get_MatrixTransformNode_Parent(mp_Plane));
                mp_NormalschnittAnimation->m_schiefwinkel = 90;
            }
            mp_NormalschnittAnimation->m_schiefwinkel = mp_NormalschnittAnimation->m_schiefwinkel + mp_NormalschnittAnimation->m_schiefwinkel_schritt;
        }
    }
    if (iMenuItem == m_pCheckboxMenuTransparent)
    {
        std::cerr << "Flaeche transparent" << std::endl;
        //Darstellung der Flaeche auf transparent schalten
        if (m_pCheckboxMenuTransparent->getState() == true)
        {
            changeMode = 6;
            Change_Mode(mp_Plane, 6);
            Change_Mode(mp_Surf, 6);
        }
        else
        {
            changeMode = 5;
            Change_Mode(mp_Plane, 5);
            Change_Mode(mp_Surf, 5);
        }
    }
    if (iMenuItem == m_pCheckboxMenuDarstMeusnierKugel)
    {
        std::cerr << "MeusnierKugel transparent" << std::endl;
        //Darstellung der MeusnierKugel auf transparent schalten
        if (mp_NormalschnittAnimation->m_meusnierkugel_exist)
        {
            if (mp_NormalschnittAnimation->m_meusnierkugel_exist)
            {
                mp_NormalschnittAnimation->Remove_MeusnierKugel(mp_Surf, mp_Plane, Get_MatrixTransformNode_Parent(mp_Surf));
                if (m_pCheckboxMenuDarstMeusnierKugel->getState() == false)
                {
                    mp_NormalschnittAnimation->Meusnier_Kugel(mp_Surf, 1, Get_MatrixTransformNode_Parent(mp_Surf));
                }
                if (m_pCheckboxMenuDarstMeusnierKugel->getState() == true)
                {
                    mp_NormalschnittAnimation->Meusnier_Kugel(mp_Surf, 2, Get_MatrixTransformNode_Parent(mp_Surf));
                }
            }
        }
        else
        {
            m_pCheckboxMenuDarstMeusnierKugel->setState(false);
        }
    }
    if (iMenuItem == m_pButtonMenuClearNormalschnitt)
    {
        std::cerr << "ClearNormalAnim" << std::endl;
        //alles ab Normalschnittanimation loeschen
        clearNormalAnim();
    }
    //Parameterebene ein/aus
    if (iMenuItem == m_pCheckboxMenuPlane)
    {
        std::cerr << "Plane onOff" << std::endl;
        //erstellt osg, kann dann mit C:\EXTERNLIBS\OpenSceneGraph-3.0.1\bin\osgconv
        //zu (.obj) oder .stl konvertiert werden fuer Blender
        //const string imagPath = (std::string)coCoviseConfig::getEntry
        // ("value", "COVER.Plugin.ParametricSurfaces.DataPath" )+ "/flaeche.osg";

        //Node* knoten = mp_GeoGroup->getChild(0);
        //osgDB::writeNodeFile(*knoten,imagPath);

        bool disable = m_pCheckboxMenuPlane->getState();
        if (disable)
        {
            Disable_Plane(false);
            checkboxPlaneState = true;
        }
        else
        {
            Disable_Plane(true);
            checkboxPlaneState = false;
        }
    }
    if (iMenuItem == m_pCheckboxMenuPlane2)
    {
        std::cerr << "Plane2 onOff" << std::endl;
        bool disable = m_pCheckboxMenuPlane2->getState();
        if (disable)
        {
            Disable_Plane(false);
            checkboxPlaneState = true;
        }
        else
        {
            Disable_Plane(true);
            checkboxPlaneState = false;
        }
    }
    if (iMenuItem == m_pCheckboxMenuPlane3)
    {
        std::cerr << "Plane3 onOff" << std::endl;
        bool disable = m_pCheckboxMenuPlane3->getState();
        if (disable)
        {
            Disable_Plane(false);
            checkboxPlaneState = true;
        }
        else
        {
            Disable_Plane(true);
            checkboxPlaneState = false;
        }
    }
    //Flaechenrand ein/aus
    if (iMenuItem == m_pCheckboxMenuNatbound)
    {
        std::cerr << "Rand einAus" << std::endl;
        bool disable = m_pCheckboxMenuNatbound->getState();
        if (disable)
            Disable_Boundary(false);
        else
            Disable_Boundary(true);
    }
    //Flaechennormalen ein/aus
    if (iMenuItem == m_pCheckboxMenuNormals)
    {
        std::cerr << "Normalen" << std::endl;
        bool disable = m_pCheckboxMenuNormals->getState();
        if (disable)
            Disable_Normals(mp_Surf, false);
        else
            Disable_Normals(mp_Surf, true);
    }
}
void ParametricSurfaces::setMenuVisible(int step)
{
    m_pObjectMenu1->setVisible(false);
    m_pObjectMenu2->setVisible(false);
    m_pObjectMenu3->setVisible(false);
    m_pObjectMenu4->setVisible(false);
    m_pObjectMenu5->setVisible(false);
    m_pObjectMenu6->setVisible(false);
    m_pObjectMenu7->setVisible(false);
    m_pObjectMenu8->setVisible(false);
    m_pObjectMenu9->setVisible(false);
    m_pObjectMenu10->setVisible(false);

    if (step == 2)
    {
        m_pObjectMenu1->setVisible(true);
        std::cerr << "Menu step2 visible" << std::endl;
    }
    else if (step == 3)
    {
        m_pObjectMenu2->setVisible(true);
        std::cerr << "Menu step3 visible" << std::endl;
    }
    else if (step == 4)
    {
        std::cerr << "Menu step4 visible" << std::endl;
        m_pObjectMenu3->setVisible(true);
    }
    else if (step == 5)
    {
        std::cerr << "Menu step5 visible" << std::endl;
        m_pObjectMenu4->setVisible(true);
    }
    else if (step == 6)
    {
        std::cerr << "Menu step6 visible" << std::endl;
        m_pObjectMenu5->setVisible(true);
    }
    else if (step == 7)
    {
        std::cerr << "Menu step7 visible" << std::endl;
        m_pObjectMenu6->setVisible(true);
    }
    else if (step == 8)
    {
        std::cerr << "Menu step8 visible" << std::endl;
        m_pObjectMenu7->setVisible(true);
    }
    else if (step == 10)
    {
        std::cerr << "Menu step10 visible" << std::endl;
        m_pObjectMenu8->setVisible(true);
    }
    else if (step == 11)
    {
        std::cerr << "Menu step11 visible" << std::endl;
        m_pObjectMenu9->setVisible(true);
    }
    else if (step == 12)
    {
        std::cerr << "Menu step12 visible" << std::endl;
        m_pObjectMenu10->setVisible(true);
    }
    /*m_pObjectMenu1->setVisible(step==2);
	m_pObjectMenu2->setVisible(step==3);
	m_pObjectMenu3->setVisible(step==4);
	m_pObjectMenu4->setVisible(step==5);
	m_pObjectMenu5->setVisible(step==6);
	m_pObjectMenu6->setVisible(step==7);
	m_pObjectMenu7->setVisible(step==8);
	m_pObjectMenu8->setVisible(step==10);
	m_pObjectMenu9->setVisible(step==11);
	m_pObjectMenu10->setVisible(step==12);*/

    VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
}

COVERPLUGIN(ParametricSurfaces)
