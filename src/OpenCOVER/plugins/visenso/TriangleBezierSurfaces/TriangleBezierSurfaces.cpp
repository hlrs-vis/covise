/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//Plugin zur Darstellung von Dreiecks-Bezierflaechen
#include <cover/RenderObject.h>
#include "cover/VRSceneGraph.h"
#include <net/message.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>

#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <vrb/client/VRBClient.h>
#include <grmsg/coGRSendCurrentDocMsg.h>
#include <cmath>
#include <iostream>

#include <osg/Vec3>
#include <osgDB/WriteFile>

#include <osgText/Text>
#include <osg/Switch>
#include <osg/Geode>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/PositionAttitudeTransform>
#include <osgGA/TrackballManipulator>
#include <grmsg/coGRKeyWordMsg.h>
#include <OpenVRUI/coNavInteraction.h>
#include <vector>
#include "cover/coTranslator.h"

#include <osgDB/ReaderWriter>
#include <osg/ShapeDrawable>

#include "TriangleBezierSurfaces.h"

#include "HfT_osg_StateSet.h"

#ifndef WIN32
#include <stdexcept>
#endif

using namespace osg;
using namespace grmsg;

//---------------------------------------------------
//Implements TriangleBezierSurfaces *plugin variable
//---------------------------------------------------
TriangleBezierSurfaces *TriangleBezierSurfaces::plugin = NULL;

TriangleBezierSurfaces::TriangleBezierSurfaces()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    root = new osg::Group();
    m_presentationStepCounter = 0; //current
    m_numPresentationSteps = 0; //gesamt
    m_presentationStep = 0; //current
    sliderChanged = false;

    m_pObjectMenu1 = NULL; //Pointer to the menu
    m_pButtonMenuSchritt = NULL;
    m_pSliderMenuGenauigkeit = NULL;
    m_pSliderMenu_n = NULL;
    m_pSliderMenu_i = NULL;
    m_pSliderMenu_j = NULL;
    m_pSliderMenu_k = NULL;
    m_pSliderMenuGrad = NULL;
    m_pSliderMenu_u = NULL;
    m_pSliderMenu_v = NULL;
    m_pSliderMenu_w = NULL;
    m_pCheckboxMenuGrad1 = NULL;
    m_pCheckboxMenuGrad2 = NULL;
    m_pCheckboxMenuNetz = NULL;
    m_pCheckboxMenuUrsprungsnetz = NULL;
    m_pCheckboxMenuFlaeche = NULL;
    m_pCheckboxMenuLabels = NULL;
    m_pCheckboxMenuSegment1 = NULL;
    m_pCheckboxMenuSegment2 = NULL;
    m_pCheckboxMenuSegment3 = NULL;
    m_pCheckboxMenuCasteljauNetz = NULL;
}

TriangleBezierSurfaces::~TriangleBezierSurfaces()
{
    m_presentationStepCounter = 0; //current
    m_numPresentationSteps = 0; //gesamt
    m_presentationStep = 0; //current
    sliderChanged = false;
    if (m_pObjectMenu1 != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu1;
    }
    if (m_pButtonMenuSchritt != NULL)
    {
        //Calls destructor of the coButtonMenuItem object
        delete m_pButtonMenuSchritt;
    }
    if (m_pSliderMenuGenauigkeit != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuGenauigkeit;
    }
    if (m_pSliderMenu_n != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_n;
    }
    if (m_pSliderMenu_i != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_i;
    }
    if (m_pSliderMenu_j != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_j;
    }
    if (m_pSliderMenu_k != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_k;
    }
    if (m_pSliderMenuGrad != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuGrad;
    }
    if (m_pSliderMenu_u != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_u;
    }
    if (m_pSliderMenu_v != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_v;
    }
    if (m_pSliderMenu_w != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenu_w;
    }
    if (m_pCheckboxMenuGrad1 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuGrad1;
    }
    if (m_pCheckboxMenuGrad2 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuGrad2;
    }
    if (m_pCheckboxMenuNetz != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuNetz;
    }
    if (m_pCheckboxMenuUrsprungsnetz != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuUrsprungsnetz;
    }
    if (m_pCheckboxMenuFlaeche != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuFlaeche;
    }
    if (m_pCheckboxMenuLabels != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuLabels;
    }
    if (m_pCheckboxMenuSegment1 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuSegment1;
    }
    if (m_pCheckboxMenuSegment2 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuSegment2;
    }
    if (m_pCheckboxMenuSegment3 != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuSegment3;
    }
    if (m_pCheckboxMenuCasteljauNetz != NULL)
    {
        //Calls destructor of the coCheckboxMenuItem object
        delete m_pCheckboxMenuCasteljauNetz;
    }
    if (label_b003 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b003;
    }
    if (label_b030 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b030;
    }
    if (label_b300 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b300;
    }
    if (label_b012 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b012;
    }
    if (label_b021 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b021;
    }
    if (label_b120 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b120;
    }
    if (label_b210 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b210;
    }
    if (label_b102 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b102;
    }
    if (label_b201 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b201;
    }
    if (label_b111 != NULL)
    {
        //Calls destructor of the coVRLabel object
        delete label_b111;
    }
    //Removes the root node from the scene graph
    cover->getObjectsRoot()->removeChild(root.get());
}

bool TriangleBezierSurfaces::init() //wird von OpenCover automatisch aufgerufen
{
    if (plugin)
        return false;
    //Set plugin
    TriangleBezierSurfaces::plugin = this;

    //Sets the possible number of presentation steps
    m_numPresentationSteps = 8;

    //hier setzen: fuer Abfrage zum Zeichnen der Segment-Punkte
    segmentCounter = 0;
    //fuer Slider_Genauigkeit + Checkbox_Flaeche, damit nicht new Interaktorpkt eingefuegt wird
    changeGenauigkeit = false;
    changeShowFlaeche = false;
    changeGrad = false;
    changeShowSegm = false;
    step4showCaption = false;

    sw = new osg::Group();
    root->addChild(sw);

    flaechenPkt = new Geode;
    // globale Linienstärke für Béziernetze
    linienstaerke = 2.7f;

    // Definition von Farben
    black = Vec4(0.2f, 0.2f, 0.2f, 1.0f);
    white = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    orange = Vec4(1.0f, 0.5f, 0.0f, 1.0f);

    red = Vec4(1.0f, 0.0f, 0.0f, 1.0f);
    lime = Vec4(0.0f, 1.0f, 0.0f, 1.0f);
    blue = Vec4(0.0f, 0.0f, 1.0f, 1.0f);

    yellow = Vec4(1.0f, 1.0f, 0.0f, 1.0f);
    aqua = Vec4(0.0f, 1.0f, 1.0f, 1.0f);
    fuchsia = Vec4(1.0f, 0.0f, 1.0f, 1.0f);

    gold = Vec4(0.8f, 0.8f, 0.3f, 1.0f);
    gold2 = Vec4(1.f, 1.f, 0.55f, 1.0f);
    silver = Vec4(0.33f, 0.33f, 0.33f, 1.0f);

    purple = Vec4(0.5f, 0.0f, 0.5f, 1.0f);
    olive = Vec4(0.5f, 0.5f, 0.0f, 1.0f);
    teal = Vec4(0.0f, 0.8f, 0.8f, 1.0f);

    navy = Vec4(0.0f, 0.0f, 0.8f, 1.0f);
    maroon = Vec4(0.8f, 0.0f, 0.0f, 1.0f);
    green = Vec4(0.0f, 0.8f, 0.0f, 1.0f);

    pink = Vec4(1.0f, 0.8f, 0.8f, 1.0f);
    brown = Vec4(143.0f / 255.0f, 71.0f / 255.0f, 71.0f / 255.0f, 1.0f);

    gruen_grad = Vec4(0.13f, 0.45f, 0.35f, 1.0f); //0.4f, 0.64f, 0.51f, 1.0f//0.4f, 0.6f, 0.0f, 1.0f
    purple_segm = Vec4(0.6f, 0.0f, 0.65f, 1.0f);

    // Anzahl der Unterteilungen
    precision = 20;
    // Default-Farbe für das Netz
    colorSelectorMeshIndex = 1;
    OrgColorMesh = lime;

    // Default-Farbe für die Fläcche
    colorSelectorSurfaceIndex = 19;
    OrgColorSurface = gold2;

    // Feld, in dem alle Farben gespeichert werden
    colorArray = new Vec4Array;

    //Standardflächenfarbe weiß
    colorArray->push_back(white);
    // Standardnetzfarbe grellgrün
    colorArray->push_back(lime);
    colorArray->push_back(red);
    colorArray->push_back(blue);
    colorArray->push_back(orange); //4
    colorArray->push_back(navy);
    colorArray->push_back(aqua);
    colorArray->push_back(fuchsia);
    colorArray->push_back(purple);
    colorArray->push_back(brown);
    colorArray->push_back(pink);
    colorArray->push_back(yellow);
    colorArray->push_back(teal);
    colorArray->push_back(maroon);
    colorArray->push_back(green); //14
    colorArray->push_back(olive);
    colorArray->push_back(gold);
    colorArray->push_back(silver);
    colorArray->push_back(black); //18
    colorArray->push_back(gold2);

    createMenu();

    m_presentationStepCounter = 0;
    changePresentationStep();

    //Beschriftung Kontrollpunkte step4
    label_b003 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b030 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b300 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b012 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b021 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b120 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b210 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b102 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b201 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));
    label_b111 = new coVRLabel("", 40, 0.06 * cover->getSceneSize(), osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f), osg::Vec4(0.2, 0.2, 0.2, 1.0));

    label_b003->setString(coTranslator::coTranslate("b003").c_str());
    label_b030->setString(coTranslator::coTranslate("b030").c_str());
    label_b300->setString(coTranslator::coTranslate("b300").c_str());
    label_b012->setString(coTranslator::coTranslate("b012").c_str());
    label_b021->setString(coTranslator::coTranslate("b021").c_str());
    label_b120->setString(coTranslator::coTranslate("b120").c_str());
    label_b210->setString(coTranslator::coTranslate("b210").c_str());
    label_b102->setString(coTranslator::coTranslate("b102").c_str());
    label_b201->setString(coTranslator::coTranslate("b201").c_str());
    label_b111->setString(coTranslator::coTranslate("b111").c_str());

    label_b003->hide();
    label_b030->hide();
    label_b300->hide();
    label_b012->hide();
    label_b021->hide();
    label_b120->hide();
    label_b210->hide();
    label_b102->hide();
    label_b201->hide();
    label_b111->hide();

    cover->getObjectsRoot()->addChild(root.get());

    //zoom scene
    //attention: never use if the scene is empty!!
    VRSceneGraph::instance()->viewAll();

    return true;
}
std::string TriangleBezierSurfaces::HfT_int_to_string(int d)
{
    std::stringstream ss;
    ss << d;
    std::string dstr = ss.str();
    return (dstr);
}

// ------------------------------------------------------------------------
// eigene Methoden
// ------------------------------------------------------------------------

// Step 1: Bernsteinpolynome vom Grad 1 und 2
void TriangleBezierSurfaces::step1()
{

    step4showCaption = false;
    grad1group = new Group;
    grad2group = new Group;

    // Grad 1 initialisieren
    grad1();

    // Grad 2 initialisieren
    grad2();

    // Variablen setzen
    showGrad1 = true;
    showGrad2 = false;

    // Grad 1 einblenden
    sw->addChild(grad1group);

    // gegebenenfalls Trennlinie zeichnen
    initializeTrennlinie();
    checkTrennlinie();
}

// Step 2: Bernsteinpolynome vom Grad 3
void TriangleBezierSurfaces::step2()
{

    step4showCaption = false;
    grad3group = new Group;

    // Anzahl der Unterteilungen
    precision = 20;

    // Farben der einzelnen Polynome setzen
    color003 = blue;
    color300 = red;
    color030 = lime;

    color111 = Vec4(1.0f / 2.0f, 1.0f / 2.0f, 1.0f / 2.0f, 1.0f);
    ;

    color102 = Vec4(1.0f / 2.0f, 0.0f, 2.0f / 2.0f, 1.0f);
    color201 = Vec4(2.0f / 2.0f, 0.0f, 1.0f / 2.0f, 1.0f);

    color012 = Vec4(0.0f, 1.0f / 2.0f, 2.0f / 2.0f, 1.0f);
    color021 = Vec4(0.0f, 0.8f, 0.6f, 1.0f);

    color210 = Vec4(2.0f / 2.0f, 1.0f / 2.0f, 0.0f, 1.0f);
    color120 = Vec4(0.6f, 0.8f, 0.0f, 1.0f);

    // Grad 3 initialisieren
    grad3(precision);

    // Grad 3 einblenden
    sw->addChild(grad3group);
}

// Step 3: beliebiges Bernseinpolynom
void TriangleBezierSurfaces::step3()
{

    step4showCaption = false; //wegen preframe

    // Variablen setzen
    iInt = 1;
    jInt = 2;
    kInt = 3;

    nInt = iInt + jInt + kInt;

    // beliebiges Bernsteinpolynom zeichnen
    bernstein = bernsteinBaryPlot(nInt, iInt, jInt, kInt, 100, white);

    // beliebiges Bernsteinpolynom einblenden
    sw->addChild(bernstein);
}

// Step 4: Bezierflaeche
void TriangleBezierSurfaces::step4()
{

    step4showSurface = true;
    step4showMesh = true;
    step4showCaption = false;

    // Bezierpunkte vom Grad 3
    step4BPgrad3 = new Vec3Array;
    // hier kann man die Bézierfläche ändern!!!
    step4b003 = Vec3(0.0f, 0.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b102 = Vec3(1.0f, 0.0f, 1.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b201 = Vec3(2.0f, 0.0f, 1.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b300 = Vec3(3.0f, 0.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b012 = Vec3(0.5f, 1.0f, 2.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b111 = Vec3(1.5f, 1.0f, 4.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b210 = Vec3(2.5f, 1.0f, 2.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b021 = Vec3(1.0f, 2.0f, 3.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b120 = Vec3(2.0f, 2.0f, 3.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step4b030 = Vec3(1.5f, 3.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    step4BPgrad3->push_back(step4b003);
    step4BPgrad3->push_back(step4b102);
    step4BPgrad3->push_back(step4b201);
    step4BPgrad3->push_back(step4b300);
    step4BPgrad3->push_back(step4b012);
    step4BPgrad3->push_back(step4b111);
    step4BPgrad3->push_back(step4b210);
    step4BPgrad3->push_back(step4b021);
    step4BPgrad3->push_back(step4b120);
    step4BPgrad3->push_back(step4b030);

    // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
    step4BSgrad3 = bezierSurfacePlot(3, step4BPgrad3, precision, OrgColorSurface, OrgColorMesh, true, step4showMesh, false, false);

    sw->addChild(step4BSgrad3);
}

// Step 5: Graderhoehung
void TriangleBezierSurfaces::step5()
{

    // Variablen setzen
    step5showMesh = true; // in diesem Step immer true!
    step5showCaption = false; // in diesem Step immer false!

    step5showSurface = true;
    step5showOrigin = true;

    // Anzahl der Unterteilungen
    precision = 20;

    // anzuzeigender Grad (hier Ursprungsgrad)
    step5degree = 2;

    // Bezierpunkte vom Grad 2
    step5BPgrad2 = new Vec3Array;
    // hier kann man die Bézierfläche ändern!!!
    step5b002 = Vec3(0.0f, 0.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step5b101 = Vec3(1.0f, 0.0f, 1.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step5b200 = Vec3(2.0f, 0.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step5b011 = Vec3(0.5f, 1.0f, 2.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step5b110 = Vec3(1.5f, 1.0f, 2.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step5b020 = Vec3(1.0f, 1.0f, 4.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    step5BPgrad2->push_back(step5b002);
    step5BPgrad2->push_back(step5b101);
    step5BPgrad2->push_back(step5b200);
    step5BPgrad2->push_back(step5b011);
    step5BPgrad2->push_back(step5b110);
    step5BPgrad2->push_back(step5b020);

    // Bezierpunkte vom Grad 3
    step5BPgrad3 = degreeElevation(2, step5BPgrad2);

    // Bezierpunkte vom Grad 4
    step5BPgrad4 = degreeElevation(3, step5BPgrad3);

    // Bezierpunkte vom Grad 5
    step5BPgrad5 = degreeElevation(4, step5BPgrad4);

    // Bezierpunkte vom Grad 6
    step5BPgrad6 = degreeElevation(5, step5BPgrad5);

    // Bezierpunkte vom Grad 7
    step5BPgrad7 = degreeElevation(6, step5BPgrad6);

    // Bezierpunkte vom Grad 8
    step5BPgrad8 = degreeElevation(7, step5BPgrad7);

    // Bezierpunkte vom Grad 9
    step5BPgrad9 = degreeElevation(8, step5BPgrad8);

    // Bezierpunkte vom Grad 10
    step5BPgrad10 = degreeElevation(9, step5BPgrad9);

    // alle Netze der Graderhöhung initialisieren
    // die Fläche wird dabei immer mit Hilfe des Ursprungsnetzes erstellt
    // da die Fläche natürlich ständig die selbe bleibt!!!

    // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
    step5BSgrad2 = bezierSurfacePlot(2, step5BPgrad2, precision, colorArray->at(colorSelectorSurfaceIndex), colorArray->at(colorSelectorMeshIndex), true, true, false, false);
    step5BSgrad3 = bezierSurfacePlot(3, step5BPgrad3, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad4 = bezierSurfacePlot(4, step5BPgrad4, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad5 = bezierSurfacePlot(5, step5BPgrad5, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad6 = bezierSurfacePlot(6, step5BPgrad6, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad7 = bezierSurfacePlot(7, step5BPgrad7, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad8 = bezierSurfacePlot(8, step5BPgrad8, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad9 = bezierSurfacePlot(9, step5BPgrad9, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad10 = bezierSurfacePlot(10, step5BPgrad10, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);

    currentDegree = step5BSgrad2;
    currentPoints = step5BPgrad2;

    // Ursprungsnetz mit Fläche einblenden
    sw->addChild(step5BSgrad2);
}

// Step 6: Casteljau-Plot
void TriangleBezierSurfaces::step6()
{

    step4showCaption = false;

    // Segmentierung zu Beginn ausblenden und auch nicht erlauben
    // soll erst nach voller Casteljau Vorführung erlaubt werden
    step6showSegmentation = false;
    step6allowSegmentation = false;
    casteljauIsOn = false;

    segmentCounter = 0;

    // Anzahl der Unterteilungen
    precision = 20;

    // anzuzeigender Grad
    step6degree = 4;

    // Variablen setzten
    step6casteljauSchritt = 0;

    step6u = 0.33f;
    step6v = 0.33f;
    step6w = 0.33f;

    // Bezierpunkte vom Grad 3
    step6BPgrad3 = new Vec3Array;

    // hier kann man die Bézierfläche ändern!!!
    step6b003 = Vec3(0.0f, 0.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b102 = Vec3(1.0f, 0.0f, 1.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b201 = Vec3(2.0f, 0.0f, 1.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b300 = Vec3(3.0f, 0.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b012 = Vec3(0.5f, 1.0f, 2.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b111 = Vec3(1.5f, 1.0f, 4.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b210 = Vec3(2.5f, 1.0f, 2.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b021 = Vec3(1.0f, 2.0f, 3.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b120 = Vec3(2.0f, 2.0f, 3.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step6b030 = Vec3(1.5f, 3.0f, 0.0f); // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    step6BPgrad3->push_back(step6b003);
    step6BPgrad3->push_back(step6b102);
    step6BPgrad3->push_back(step6b201);
    step6BPgrad3->push_back(step6b300);
    step6BPgrad3->push_back(step6b012);
    step6BPgrad3->push_back(step6b111);
    step6BPgrad3->push_back(step6b210);
    step6BPgrad3->push_back(step6b021);
    step6BPgrad3->push_back(step6b120);
    step6BPgrad3->push_back(step6b030);

    // Bezierpunkte vom Grad 4												// künstliche Graderhöhung, das man beim Casteljau-Algorithmus ein Schritt mehr sieht...
    step6BPgrad4 = degreeElevation(3, step6BPgrad3);

    // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
    step6BSgrad4 = bezierSurfacePlot(step6degree, step6BPgrad4, precision, colorArray->at(colorSelectorSurfaceIndex), colorArray->at(colorSelectorMeshIndex), true, true, false, true);

    // (leere) Casteljau-Zeichnung initialisieren
    casteljauGroup = triangleCasteljauPlot(step6degree, step6casteljauSchritt, step6BPgrad4, step6u, step6v, step6w);

    // Fläche einblenden
    sw->addChild(step6BSgrad4);

    // Casteljau-Zeichnung einblenden
    sw->addChild(casteljauGroup);

    //schwarzen Flaechenpkt zeichnen
    Casteljau_FlaechenPkt();
}

// Fakultaet-Methoden

int TriangleBezierSurfaces::fak(int k)
{

    int i = 1;
    for (int n = 2; n <= k; n++)
    {
        i = i * n;
    }
    return i;
}

int TriangleBezierSurfaces::fakRek(int k)
{
    if (k <= 1)
    {
        return 1;
    }
    else
    {
        return k * fakRek(k - 1);
    }
}
void TriangleBezierSurfaces::setMaterial_line(Geometry *geom2, Vec4 color)
{
    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(color);
    geom2->setColorArray(colors.get());
    geom2->setColorBinding(Geometry::BIND_OVERALL);
    osg::StateSet *stateset = geom2->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(2.0f);
    stateset->setAttribute(lw);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED);
    ref_ptr<PolygonMode> m_Triangle = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    stateset->setAttributeAndModes(m_Triangle, StateAttribute::PROTECTED);
}
void TriangleBezierSurfaces::setMaterial(Geometry *geom, Vec4 color)
{
    ref_ptr<StateSet> geo_state = geom->getOrCreateStateSet();
    ref_ptr<Material> mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, color); //aqua
    mtl->setDiffuse(Material::FRONT_AND_BACK, color * 0.7);
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    //mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setShininess(Material::FRONT_AND_BACK, 0.1f);
    //mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(0.1,     0.18725, 0.1745,  1.0));//tuerikis veraendert
    //mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(0.296,   0.74151, 0.69102, 1.0));
    //mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.297254,0.30829, 0.306678,1.0));
    ////mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    //mtl->setShininess(Material::FRONT_AND_BACK,12.8f);

    geo_state->setAttribute(mtl.get(), osg::StateAttribute::ON);
    geo_state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
}
// Bernstein-Methoden

float TriangleBezierSurfaces::bernsteinBary(unsigned int grad, unsigned int i, unsigned int j, unsigned int k, float u, float v, float w)
{

    // gibt direkt die Formel der Bernsteinpolynome zurück
    return (fak(grad) / (fak(i) * fak(j) * fak(k))) * (pow(u, float(i)) * pow(v, float(j)) * pow(w, float(k)));
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::bernsteinBaryPlot(unsigned int grad, unsigned int i, unsigned int j, unsigned int k, unsigned int unterteilungen, Vec4 color)
{

    ref_ptr<Geode> geode = new Geode;
    ref_ptr<Geometry> geom = new Geometry;

    ref_ptr<Vec3Array> punkte = new Vec3Array();
    ref_ptr<Vec3Array> normals = new Vec3Array();
    ref_ptr<osg::DrawElementsUInt> TriangleEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);

    Vec3 p1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 p2 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 p3 = Vec3(0.0f, 0.0f, 0.0f);

    float vp1 = 0.0f;
    float vp2 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;

    float jj = 0.0f;
    float ii = 0.0f;

    Vec3f D1, D2, N;
    double x, y, z, bt;
    int kind = 0;
    // kleine Dreiecke zeichnen
    for (unsigned int jl = 0; jl < unterteilungen; jl++)
    {
        for (unsigned int il = 0; il < unterteilungen - jl; il++)
        {

            vp1 = jj / unterteilungen;
            vp2 = (jj + 1) / unterteilungen;
            up1 = ii / unterteilungen;
            up2 = (ii + 1) / unterteilungen;

            p1 = Vec3(up1, vp1, bernsteinBary(grad, i, j, k, up1, vp1, 1 - up1 - vp1));
            p2 = Vec3(up2, vp1, bernsteinBary(grad, i, j, k, up2, vp1, 1 - up2 - vp1));
            p3 = Vec3(up1, vp2, bernsteinBary(grad, i, j, k, up1, vp2, 1 - up1 - vp2));
            punkte->push_back(p1);
            punkte->push_back(p2);
            punkte->push_back(p3);

            //fjs gleich Normalen berechnen
            D1 = p3 - p1;
            D2 = p2 - p1;
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
            normals->push_back(N);
            normals->push_back(N);
            normals->push_back(N);

            TriangleEdges->push_back(kind);
            kind++;
            TriangleEdges->push_back(kind);
            kind++;
            TriangleEdges->push_back(kind);
            kind++;

            ii = ii + 1;
        }
        jj = jj + 1;
        ii = 0.0f;
    }
    geom->setVertexArray(punkte.get());
    geom->addPrimitiveSet(TriangleEdges);
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_PER_VERTEX);

    HfT_osg_StateSet *stateset = new HfT_osg_StateSet(STRIANGLES, color);
    ref_ptr<LineWidth> Line = new LineWidth(2.f);
    stateset->setAttributeAndModes(Line, StateAttribute::ON);
    geom->setStateSet(stateset);

    geode->addDrawable(geom);

    // Bernsteinpolynom beschriften

    osg::ref_ptr<osgText::Font> font = coVRFileManager::instance()->loadFont(NULL);

    osgText::Text *text = new osgText::Text;
    text->setFont(font);
    text->setColor(color);
    text->setFontResolution(100, 100);
    //text->setAxisAlignment(osgText::Text::SCREEN);
    text->setCharacterSize(0.2);
    text->setPosition(Vec3(0.35f, -0.2f, 0.0f));
    std::string txt = "B";
    text->setText(txt);

    osgText::Text *index = new osgText::Text;
    index->setFont(font);
    index->setColor(color);

    ref_ptr<StateSet> text_state = index->getOrCreateStateSet();
    ref_ptr<Material> mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.f));
    text_state->setAttribute(mtl.get(), osg::StateAttribute::ON);
    text_state->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    index->setStateSet(text_state);
    index->setFontResolution(50, 50);
    //index->setAxisAlignment(osgText::Text::SCREEN);
    index->setCharacterSize(0.1);
    index->setPosition(Vec3(0.5f, -0.21f, 0.0f));
    std::string txt1 = HfT_int_to_string(i);
    std::string txt2 = HfT_int_to_string(j);
    std::string txt3 = HfT_int_to_string(k);
    index->setText(txt1 + txt2 + txt3);

    osgText::Text *hoch = new osgText::Text;
    hoch->setFont(font);
    hoch->setColor(color);
    hoch->setStateSet(text_state);
    hoch->setFontResolution(50, 50);
    //index->setAxisAlignment(osgText::Text::SCREEN);
    hoch->setCharacterSize(0.1);
    hoch->setPosition(Vec3(0.5f, -0.09f, 0.0f));
    std::string txt4 = HfT_int_to_string(grad);
    hoch->setText(txt4);

    geode->addDrawable(text);
    geode->addDrawable(index);
    geode->addDrawable(hoch);

    // das Parametergebiet (den Boden) zeichnen
    ref_ptr<Geometry> dreieck = new Geometry;
    ref_ptr<Vec3Array> eckPunkte = new Vec3Array();
    ref_ptr<Vec3Array> eckNormal = new Vec3Array();
    ref_ptr<osg::DrawElementsUInt> eckTriangle = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);

    Vec4f farbe = brown;
    p1 = Vec3(0.0f, 0.0f, 0.0f);
    p2 = Vec3(0.0f, 1.0f, 0.0f);
    p3 = Vec3(1.0f, 0.0f, 0.0f);

    eckPunkte->push_back(p1);
    eckPunkte->push_back(p2);
    eckPunkte->push_back(p3);
    eckNormal->push_back(Vec3(0.0f, 0.0f, -1.0f));
    eckNormal->push_back(Vec3(0.0f, 0.0f, -1.0f));
    eckNormal->push_back(Vec3(0.0f, 0.0f, -1.0f));

    dreieck->setVertexArray(eckPunkte.get());
    dreieck->setNormalArray(eckNormal.get());
    dreieck->setNormalBinding(Geometry::BIND_PER_VERTEX);
    eckTriangle->push_back(0);
    eckTriangle->push_back(1);
    eckTriangle->push_back(2);
    dreieck->addPrimitiveSet(eckTriangle);
    //fjs
    setMaterial(dreieck, farbe);

    /*HfT_osg_StateSet *stateset2 = new HfT_osg_StateSet(SSHADE,farbe);
	dreieck->setStateSet(stateset2);*/
    geode->addDrawable(dreieck);

    //fertig!
    return geode;
}

// Casteljau-Methoden

Vec3 TriangleBezierSurfaces::triangleCasteljau(unsigned int grad, Vec3Array *bezPoints, float u, float v, float w)
{

    // Parameter in lokale Konstanten/Variablen kopieren
    const unsigned int G = grad;
    const unsigned int ANZAHL = grad + 1;
    Vec3Array *bezPointsCopy = bezPoints;

    // 3-dimensionalen Vektor anlegen und länge defieren
    std::vector<std::vector<std::vector<Vec3> > > my3dimArray(ANZAHL);

    for (unsigned int a = 0; a < ANZAHL; a++)
    {
        my3dimArray[a].resize(ANZAHL);
        for (unsigned int b = 0; b < ANZAHL; b++)
        {
            my3dimArray[a][b].resize(ANZAHL);
        }
    }

    // folgende for-Schleifen machen aus einem Vec3Array ein dreifach-Feld
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;

    for (unsigned int stufe = 0; stufe < ANZAHL; stufe++)
    { // stufe läuft in v-Richtung

        Vec3Array *kuerzung = new Vec3Array;

        x = 0;
        y = stufe;
        z = G - stufe;

        for (unsigned int zaehler = 0; zaehler < ANZAHL - stufe; zaehler++)
        { // zaehler läuft in u-Richtung
            my3dimArray[x][y][z] = bezPointsCopy->at(zaehler);
            x++;
            y = stufe; // nicht unbedingt notwendig
            z--;
        }

        // bezPoints vorne um entsprechende Länge kürzen
        for (unsigned int kopieZaehler = ANZAHL - stufe; kopieZaehler < bezPointsCopy->size(); kopieZaehler++)
        {
            kuerzung->push_back(bezPointsCopy->at(kopieZaehler));
        }
        bezPointsCopy = kuerzung;
    }
    // ab hier kann man ueber baryzentrische Koordinaten auf die Bezierpunkte zugreifen! (my3dimArray[i][j][k])

    // Kopie der Bezierpunkte von einem 3-fach in ein 4-fach Feld
    std::vector<std::vector<std::vector<std::vector<Vec3> > > > bezFeld(ANZAHL);

    for (unsigned int a = 0; a < ANZAHL; a++)
    {
        bezFeld[a].resize(ANZAHL);
        for (unsigned int b = 0; b < ANZAHL; b++)
        {
            bezFeld[a][b].resize(ANZAHL);
            for (unsigned int c = 0; c < ANZAHL; c++)
            {
                bezFeld[a][b][c].resize(ANZAHL);
            }
        }
    }

    int k = 0;

    for (unsigned int j = 0; j < ANZAHL; j++)
    {

        for (unsigned int i = 0; i < ANZAHL - j; i++)
        {

            k = G - i - j;

            if (k >= 0)
            {
                bezFeld[0][i][j][k] = my3dimArray[i][j][k];
            }
        }
    }

    // ab hier Casteljau-Algorithmus
    for (unsigned int r = 1; r < ANZAHL; r++)
    {
        for (unsigned int j = 0; j < ANZAHL; j++)
        {
            for (unsigned int i = 0; i < ANZAHL - j; i++)
            {
                k = G - r - i - j;
                if (k >= 0)
                {
                    bezFeld[r][i][j][k] = bezFeld[r - 1][i + 1][j][k] * u + bezFeld[r - 1][i][j + 1][k] * v + bezFeld[r - 1][i][j][k + 1] * w;
                }
            }
        }
    }

    return bezFeld[G][0][0][0];
}

std::vector<std::vector<std::vector<std::vector<Vec3> > > > TriangleBezierSurfaces::triangleCasteljauPoints(unsigned int grad, Vec3Array *bezPoints, float u, float v, float w)
{

    // Methode genau gleich wie triangleCasteljau, nur anderer Rückgabewert!

    // Parameter in lokale Konstanten/Variablen kopieren
    const unsigned int G = grad;
    const unsigned int ANZAHL = grad + 1;
    Vec3Array *bezPointsCopy = bezPoints;

    // 3-dimensionalen Vektor anlegen und länge defieren
    std::vector<std::vector<std::vector<Vec3> > > my3dimArray(ANZAHL);

    for (unsigned int a = 0; a < ANZAHL; a++)
    {
        my3dimArray[a].resize(ANZAHL);
        for (unsigned int b = 0; b < ANZAHL; b++)
        {
            my3dimArray[a][b].resize(ANZAHL);
        }
    }

    // folgende for-Schleifen machen aus einem Vec3Array ein dreifach-Feld
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;

    for (unsigned int stufe = 0; stufe < ANZAHL; stufe++)
    { // stufe läuft in v-Richtung

        Vec3Array *kuerzung = new Vec3Array;

        x = 0;
        y = stufe;
        z = G - stufe;

        for (unsigned int zaehler = 0; zaehler < ANZAHL - stufe; zaehler++)
        { // zaehler läuft in u-Richtung
            my3dimArray[x][y][z] = bezPointsCopy->at(zaehler);
            x++;
            y = stufe; // nicht unbedingt notwendig
            z--;
        }

        // bezPoints vorne um entsprechende Länge kürzen
        for (unsigned int kopieZaehler = ANZAHL - stufe; kopieZaehler < bezPointsCopy->size(); kopieZaehler++)
        {
            kuerzung->push_back(bezPointsCopy->at(kopieZaehler));
        }

        bezPointsCopy = kuerzung;
    }
    // ab hier kann man ueber baryzentrische Koordinaten auf die Bezierpunkte zugreifen! (my3dimArray[i][j][k])

    // Kopie der Bezierpunkte von einem 3-fach in ein 4-fach Feld
    std::vector<std::vector<std::vector<std::vector<Vec3> > > > bezFeld(ANZAHL);

    for (unsigned int a = 0; a < ANZAHL; a++)
    {
        bezFeld[a].resize(ANZAHL);
        for (unsigned int b = 0; b < ANZAHL; b++)
        {
            bezFeld[a][b].resize(ANZAHL);
            for (unsigned int c = 0; c < ANZAHL; c++)
            {
                bezFeld[a][b][c].resize(ANZAHL);
            }
        }
    }

    int k = 0;

    for (unsigned int j = 0; j < ANZAHL; j++)
    {

        for (unsigned int i = 0; i < ANZAHL - j; i++)
        {

            k = G - i - j;

            if (k >= 0)
            {
                bezFeld[0][i][j][k] = my3dimArray[i][j][k];
            }
        }
    }

    // ab hier Casteljau-Algorithmus
    for (unsigned int r = 1; r < ANZAHL; r++)
    {

        for (unsigned int j = 0; j < ANZAHL; j++)
        {

            for (unsigned int i = 0; i < ANZAHL - j; i++)
            {
                k = G - r - i - j;
                if (k >= 0)
                {
                    bezFeld[r][i][j][k] = bezFeld[r - 1][i + 1][j][k] * u + bezFeld[r - 1][i][j + 1][k] * v + bezFeld[r - 1][i][j][k + 1] * w;
                }
            }
        }
    }
    return bezFeld;
}

osg::ref_ptr<osg::Group> TriangleBezierSurfaces::triangleCasteljauPlot(unsigned int grad, unsigned int schritt, Vec3Array *bezPoints, float u, float v, float w)
{

    // gesamte Zeichnung ist eine Gruppe
    // das Netz ist eine Geode und
    // die einzelnen Punkte sind einzelne Geoden

    ref_ptr<osg::Group> gruppe = new Group;
    ref_ptr<osg::Geode> net = new Geode;

    // Parameter in lokale Konstanten kopieren
    const int G = grad;
    const int S = schritt;

    // dient dazu, dass die neuen Punkte, die alten Punkte übermalen, falls diese genau übereinander liegen
    // wird hier mit 0 initialisiert
    float vergroesserung = 0.0f;

    if (0.99f < u + v + w && u + v + w < 1.01f && 0 <= S && S <= G)
    { // Sicherheitsabfrage

        // Bézierpunkte lokal kopieren
        Vec3Array *bezPointsCopy = bezPoints;

        // c dient als Farbselektor
        unsigned int c = 11;

        int k = 0;

        // alle Casteljau-Punkte im Vorfeld berechnen lassen
        std::vector<std::vector<std::vector<std::vector<Vec3> > > > bezHelp = triangleCasteljauPoints(G, bezPointsCopy, u, v, w);

        for (int r = 1; r <= S; r++)
        {

            Vec4 netColor = colorArray->at(c);

            ref_ptr<Vec4Array> netColors = new Vec4Array();
            netColors->push_back(netColor);

            for (int i = 0; i <= G; i++)
            {

                for (int j = 0; j < G - i - r; j++)
                {

                    k = G - i - j - r;

                    if ((i + 1) < G && (j + 1) < G && (k - 1) >= 0)
                    {

                        // Dreiecksnetz zeichnen
                        ref_ptr<Vec3Array> pointList = new Vec3Array();

                        ref_ptr<Geometry> geom = new Geometry;

                        Vec3 p1 = bezHelp[r][i][j][k];
                        Vec3 p2 = bezHelp[r][i][j + 1][k - 1];
                        Vec3 p3 = bezHelp[r][i + 1][j][k - 1];

                        pointList->push_back(p1);
                        pointList->push_back(p2);
                        pointList->push_back(p3);
                        pointList->push_back(p1);

                        geom->setVertexArray(pointList.get());
                        geom->setColorArray(netColors);
                        geom->setColorBinding(Geometry::BIND_OVERALL);
                        ref_ptr<StateSet> stateset = geom->getOrCreateStateSet();
                        ref_ptr<LineWidth> line = new LineWidth(linienstaerke);
                        stateset->setAttribute(line);
                        stateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED);
                        geom->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, 4));

                        net->addDrawable(geom);

                        // Eckpunkte zeichnen

                        // Punkt wird pro Schritt um 1/1000 erhöht
                        vergroesserung = float(r) / 1000.0f;

                        ref_ptr<osg::Geode> bezPoint01 = new Geode;
                        ref_ptr<osg::Geode> bezPoint02 = new Geode;
                        ref_ptr<osg::Geode> bezPoint03 = new Geode;

                        bezPoint01 = createSphere(0.06f + vergroesserung, netColor, p1);
                        bezPoint02 = createSphere(0.06f + vergroesserung, netColor, p2);
                        bezPoint03 = createSphere(0.06f + vergroesserung, netColor, p3);

                        gruppe->addChild(bezPoint01);
                        gruppe->addChild(bezPoint02);
                        gruppe->addChild(bezPoint03);
                    }
                }
            }

            if (r == 1)
                c = 4;
            else if (r == 2)
                c = 2;
            else if (r == 3)
                c = 18;
            else
            {
                // Farbselektor wird um eins erhöht
                c = (c + 1) % (colorArray->size());
            }
        }

        // Endpunkt fehlt noch...
        vergroesserung = float(grad) / 1000.0f;

        if (schritt == grad)
        {
            ref_ptr<osg::Geode> surfPoint = new Geode;
            surfPoint = createSphere(0.06f + vergroesserung, black, bezHelp[G][0][0][0]);
            gruppe->addChild(surfPoint);
        }
        gruppe->addChild(net);
    }

    // ...fertig!
    return gruppe;
}

// Bezier-Methoden

Vec3 TriangleBezierSurfaces::getSurfacePoint(unsigned int grad, Vec3Array *bezPoints, float u, float v, float w)
{

    // Parameter in lokale Konstanten/Variablen kopieren
    const unsigned int G = grad;
    const unsigned int ANZAHL = grad + 1;
    Vec3Array *bezPointsCopy = bezPoints;

    // 3-dimensionalen Vektor anlegen und länge defieren
    std::vector<std::vector<std::vector<Vec3> > > my3dimArray(ANZAHL);

    for (unsigned int a = 0; a < ANZAHL; a++)
    {
        my3dimArray[a].resize(ANZAHL);
        for (unsigned int b = 0; b < ANZAHL; b++)
        {
            my3dimArray[a][b].resize(ANZAHL);
        }
    }

    // folgende for-Schleifen machen aus einem Vec3Array ein dreifach-Feld
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;

    for (unsigned int stufe = 0; stufe < ANZAHL; stufe++)
    { // stufe läuft in v-Richtung

        Vec3Array *kuerzung = new Vec3Array;

        x = 0;
        y = stufe;
        z = G - stufe;

        for (unsigned int zaehler = 0; zaehler < ANZAHL - stufe; zaehler++)
        { // zaehler läuft in u-Richtung

            my3dimArray[x][y][z] = bezPointsCopy->at(zaehler);

            x++;
            y = stufe; // nicht unbedingt notwendig
            z--;
        }

        // bezPoints vorne um entsprechende Länge kürzen
        for (unsigned int kopieZaehler = ANZAHL - stufe; kopieZaehler < bezPointsCopy->size(); kopieZaehler++)
        {
            kuerzung->push_back(bezPointsCopy->at(kopieZaehler));
        }
        bezPointsCopy = kuerzung;
    }

    // ab hier kann man ueber baryzentrische Koordinaten auf die Bezierpunkte zugreifen! (my3dimArray[i][j][k])

    // Ermittlung des Kurvenpunktes mittels Berechnungsformel
    Vec3 surfacePoint = Vec3(0.0f, 0.0f, 0.0f);

    unsigned int k = 0;

    for (unsigned int j = 0; j < ANZAHL; j++)
    {

        for (unsigned int i = 0; i < ANZAHL - j; i++)
        {

            k = G - i - j;

            surfacePoint = surfacePoint + my3dimArray[i][j][k] * bernsteinBary(grad, i, j, k, u, v, w);
        }
    }

    return surfacePoint;
}

osg::ref_ptr<osg::Group> TriangleBezierSurfaces::bezierSurfacePlot(unsigned int grad, Vec3Array *bezPoints, unsigned int unterteilungen, Vec4 surfaceColor, Vec4 meshColor, bool showSurface, bool showMesh, bool showCaption, bool useCasteljau)
{
    // Bézierfläche ist eine Geode
    // Netz ist eine Gruppe, bestehend aus dem Netz und den einzelnen Punkten,
    // die einzelnen Punkte sind jeweils einzelne Geoden
    // falls die Beschriftung angewählt wurde, dann sind die einzelnen Beschriftungen jeweils auch einzelne Geoden
    ref_ptr<osg::Group> gruppe = new Group;

    if (showSurface) // Bezierfläche digitalisieren und als Dreiecksnetz darstellen
    {
        osg::ref_ptr<osg::Geode> surface = bezierSurfPointsPlot(grad, bezPoints, unterteilungen, surfaceColor, useCasteljau);
        gruppe->insertChild(1, surface);
    }
    if (showMesh)
    {
        // ----------------------------------------------------------------
        // Bezierpunkte als Kugeln zeichnen und mit Linien verbinden
        // ----------------------------------------------------------------
        ref_ptr<osg::Geode> linien = bezierNetPlot(grad, bezPoints, meshColor);

        //-1, keine doppelten Punkte bei Nichtinteraktion
        //grad 2 in step5, sonst sind Interaktoren fuer alle Grade
        if (interactPlot == -1 && ((grad == 2 && changeGenauigkeit == false && changeShowFlaeche == false && changeGrad == false && m_presentationStep == 5)
                                   || (grad == 3 && changeGenauigkeit == false && changeShowFlaeche == false && (m_presentationStep == 0 || m_presentationStep == 4 || m_presentationStep == 7))
                                   || (m_presentationStep == 6 && changeShowSegm == false)))
        {
            for (unsigned int i = 0; i < bezPoints->size(); i++)
            {
                controlPoints.push_back(new InteractionPoint(bezPoints->at(i)));
            }
        }
        gruppe->insertChild(2, linien);
        if ((grad != 2 && m_presentationStep == 5) || segmentCounter != 0)
        {
            ref_ptr<osg::Group> bezierpunktegruppe = bezierPointsPlot(grad, bezPoints, meshColor);
            gruppe->insertChild(3, bezierpunktegruppe);
        }
    }
    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        controlPoints[i]->showInteractor(showMesh);
    }
    return gruppe.release();
}

osg::ref_ptr<osg::Group> TriangleBezierSurfaces::bezierPointsPlot(unsigned int grad, Vec3Array *bezPoints, Vec4 meshColor)
{
    ref_ptr<osg::Group> pointgroup = new Group();
    pointgroup->setName("PUNKTE");
    for (unsigned int maler = 0; maler < bezPoints->size(); maler++)
    {
        float rad = 0.05f;
        if (segmentCounter == 2)
            rad = 0.052f;
        if (segmentCounter == 3)
            rad = 0.045f;
        ref_ptr<osg::Geode> bezPoint = new Geode;
        bezPoint = createSphere(rad, meshColor, bezPoints->at(maler));
        bezPoint->setName("KUGEL");
        pointgroup->addChild(bezPoint);
    }
    return pointgroup.release();
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::bezierSurfPointsPlot(unsigned int grad, Vec3Array *bezPoints, unsigned int unterteilungen, Vec4 surfaceColor, bool useCasteljau)
{
    ref_ptr<osg::Geode> surface = new Geode();
    surface->setName("SURFACE");
    ref_ptr<Geometry> geom1 = new Geometry;
    ref_ptr<Geometry> geom3 = new Geometry;
    ref_ptr<Vec3Array> surfplist1 = new Vec3Array();
    ref_ptr<Vec3Array> surfplist3 = new Vec3Array();
    ref_ptr<Vec3Array> normals1 = new Vec3Array();
    ref_ptr<Vec3Array> normals3 = new Vec3Array();
    ref_ptr<osg::DrawElementsUInt> TriangleEdges1 = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
    ref_ptr<osg::DrawElementsUInt> TriangleEdges3 = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);

    Vec3 p1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 p2 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 p3 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 p4 = Vec3(0.0f, 0.0f, 0.0f);

    float up1 = 0.0f, up2 = 0.0f, vp1 = 0.0f, vp2 = 0.0f, jj = 0.0f, ii = 0.0f;
    int kind1 = 0, kind3 = 0;
    Vec3f D1, D2, N;
    double x, y, z, bt;
    for (unsigned int j = 0; j < unterteilungen; j++)
    {
        for (unsigned int i = 0; i < unterteilungen - j; i++)
        {
            up1 = ii / unterteilungen;
            up2 = (ii + 1) / unterteilungen;
            vp1 = jj / unterteilungen;
            vp2 = (jj + 1) / unterteilungen;

            if (useCasteljau)
            { // Anwendung von Casteljau
                p1 = triangleCasteljau(grad, bezPoints, up1, vp1, 1 - up1 - vp1);
                p2 = triangleCasteljau(grad, bezPoints, up2, vp1, 1 - up2 - vp1);
                p3 = triangleCasteljau(grad, bezPoints, up1, vp2, 1 - up1 - vp2);
                p4 = triangleCasteljau(grad, bezPoints, up2, vp2, 1 - up2 - vp2);
            }
            else
            { //Anwendung des Berechnungsformel
                p1 = getSurfacePoint(grad, bezPoints, up1, vp1, 1 - up1 - vp1);
                p2 = getSurfacePoint(grad, bezPoints, up2, vp1, 1 - up2 - vp1);
                p3 = getSurfacePoint(grad, bezPoints, up1, vp2, 1 - up1 - vp2);
                p4 = getSurfacePoint(grad, bezPoints, up2, vp2, 1 - up2 - vp2);
            }
            //fjs gleich Normalen berechnen
            D1 = p3 - p1;
            D2 = p2 - p1;
            x = D1.y() * D2.z() - D1.z() * D2.y();
            y = -D1.x() * D2.z() + D1.z() * D2.x();
            z = D1.x() * D2.y() - D1.y() * D2.x();
            bt = sqrt(x * x + y * y + z * z);
            if (bt > 0.000001)
            {
                x = x / bt;
                y = y / bt;
                z = z / bt;
            }
            N.set(x, y, z);
            normals1->push_back(N);
            normals1->push_back(N);
            normals1->push_back(N);
            normals3->push_back(N);
            normals3->push_back(N);
            normals3->push_back(N);

            surfplist1->push_back(p1);
            surfplist1->push_back(p2);
            surfplist1->push_back(p3);
            surfplist3->push_back(p1);
            surfplist3->push_back(p2);
            surfplist3->push_back(p3);
            if (ii < unterteilungen - j - 1)
            {
                D1 = p3 - p2;
                D2 = p4 - p2;
                x = D1.y() * D2.z() - D1.z() * D2.y();
                y = -D1.x() * D2.z() + D1.z() * D2.x();
                z = D1.x() * D2.y() - D1.y() * D2.x();
                bt = sqrt(x * x + y * y + z * z);
                if (bt > 0.000001)
                {
                    x = x / bt;
                    y = y / bt;
                    z = z / bt;
                }
                N.set(x, y, z);
                normals1->push_back(N);
                normals1->push_back(N);
                normals1->push_back(N);
                surfplist1->push_back(p2);
                surfplist1->push_back(p4);
                surfplist1->push_back(p3);
            }

            // Dreiecke
            TriangleEdges1->push_back(kind1);
            kind1++;
            TriangleEdges1->push_back(kind1);
            kind1++;
            TriangleEdges1->push_back(kind1);
            kind1++;

            TriangleEdges3->push_back(kind3);
            kind3++;
            TriangleEdges3->push_back(kind3);
            kind3++;
            TriangleEdges3->push_back(kind3);
            kind3++;

            if (ii < unterteilungen - j - 1)
            {
                TriangleEdges1->push_back(kind1);
                kind1++;
                TriangleEdges1->push_back(kind1);
                kind1++;
                TriangleEdges1->push_back(kind1);
                kind1++;
            }
            ii = ii + 1;
        }
        jj = jj + 1;
        ii = 0;
    }
    geom1->setVertexArray(surfplist1.get());
    geom1->addPrimitiveSet(TriangleEdges1);
    geom1->setNormalArray(normals1.get());
    geom1->setNormalBinding(Geometry::BIND_PER_VERTEX);
    HfT_osg_StateSet *stateset1 = new HfT_osg_StateSet(SSHADE, surfaceColor);
    geom1->setStateSet(stateset1);

    geom3->setVertexArray(surfplist3.get());
    geom3->addPrimitiveSet(TriangleEdges3);
    geom3->setNormalArray(normals3.get());
    geom3->setNormalBinding(Geometry::BIND_PER_VERTEX);
    osg::Vec4 color_lines = Vec4(0.5f, 0.5f, 0.5f, 1.0f);
    setMaterial_line(geom3, color_lines);

    surface->addDrawable(geom1);
    surface->addDrawable(geom3);
    return surface.release();
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::bezierNetPlot(unsigned int grad, Vec3Array *bezPoints, Vec4 meshColor)
{
    // ----------------------------------------------------------------
    // Netz = Bezierpunkte einfach als Linien  zeichnen
    // ----------------------------------------------------------------
    ref_ptr<osg::Geode> net = new Geode();
    net->setName("NETZ");
    ref_ptr<Geometry> netGeo = new Geometry();
    netGeo->setVertexArray(bezPoints);
    net->addDrawable(netGeo);

    ref_ptr<osg::DrawElementsUInt> netLines = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    // Linien in u-Richtung
    int anz = grad, danz = anz, index = 0;
    for (unsigned int k = 0; k < grad; k++)
    {
        for (unsigned int j = 0; j < grad - k; j++)
        {
            netLines->push_back(index);
            netLines->push_back(index + 1);
            index++;
        }
        index = anz + 1;
        anz = anz + danz;
        danz--;
    }

    // Linien in v Richtung
    anz = grad, danz = anz, index = 0;
    for (unsigned int k = 0; k < grad; k++)
    {
        for (unsigned int j = 0; j < grad - k; j++)
        {
            netLines->push_back(index);
            netLines->push_back(index + 1 + grad - k);
            index++;
        }
        index = anz + 1;
        anz = anz + danz;
        danz--;
    }
    // Linien in w Richtung
    anz = grad, danz = anz, index = 0;
    for (unsigned int k = 0; k < grad; k++)
    {
        for (unsigned int j = 0; j < grad - k; j++)
        {
            netLines->push_back(index + 1);
            netLines->push_back(index + 1 + grad - k);
            index++;
        }
        index = anz + 1;
        anz = anz + danz;
        danz--;
    }
    // Mesh hat Bezierpunkte mit Verbindungslinien
    float width = 1.7f;
    if (meshColor == gruen_grad)
    {
        width = 2.5f;
    }
    netGeo->addPrimitiveSet(netLines);
    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(meshColor);
    netGeo->setColorArray(colors.get());
    netGeo->setColorBinding(Geometry::BIND_OVERALL);
    osg::StateSet *netstateset = netGeo->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(width);
    netstateset->setAttribute(lw);
    netstateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED);

    return net.release();
}

Vec3Array *TriangleBezierSurfaces::degreeElevation(unsigned int gradVorErhoehung, Vec3Array *bezpoints)
{

    // Parameter in lokale Konstanten/Variablen kopieren
    int G = gradVorErhoehung;
    Vec3Array *bezPointsCopy = bezpoints;
    const unsigned int ANZAHL = gradVorErhoehung + 1;

    // 3-dimensionalen Vektor anlegen und mit Nullen füllen, damit hier (*) nichts schief gehen kann
    std::vector<std::vector<std::vector<Vec3> > > my3dimArray(ANZAHL + 2);

    std::vector<Vec3> dim1(ANZAHL + 2);
    std::vector<std::vector<Vec3> > dim2(ANZAHL + 2);

    for (unsigned int a = 0; a < ANZAHL + 2; a++)
    {
        dim1[a] = Vec3(0.0f, 0.0f, 0.0f);
    }

    for (unsigned int b = 0; b < ANZAHL + 2; b++)
    {
        dim2[b] = dim1;
    }

    for (unsigned int c = 0; c < ANZAHL + 2; c++)
    {
        my3dimArray[c] = dim2;
    }

    // 3-dimensionalen Vektor anlegen und länge defieren
    std::vector<std::vector<std::vector<Vec3> > > my3dimArrayNew(ANZAHL + 1);

    for (unsigned int a = 0; a < ANZAHL + 1; a++)
    {
        my3dimArrayNew[a].resize(ANZAHL + 1);
        for (unsigned int b = 0; b < ANZAHL + 1; b++)
        {
            my3dimArrayNew[a][b].resize(ANZAHL + 1);
        }
    }

    // folgende for-Schleifen machen aus einem Vec3Array ein dreifach-Feld
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;

    for (unsigned int stufe = 0; stufe < ANZAHL; stufe++)
    { // stufe läuft in v-Richtung

        Vec3Array *kuerzung = new Vec3Array;

        x = 0;
        y = stufe;
        z = G - stufe;

        for (unsigned int zaehler = 0; zaehler < ANZAHL - stufe; zaehler++)
        { // zaehler läuft in u-Richtung
            my3dimArray[x + 1][y + 1][z + 1] = bezPointsCopy->at(zaehler);
            x++;
            y = stufe; // nicht unbedingt notwendig
            z--;
        }

        // bezPoints vorne um entsprechende Länge kürzen
        for (unsigned int kopieZaehler = ANZAHL - stufe; kopieZaehler < bezPointsCopy->size(); kopieZaehler++)
        {
            kuerzung->push_back(bezPointsCopy->at(kopieZaehler));
        }

        bezPointsCopy = kuerzung;
    }
    // ab hier kann man ueber baryzentrische Koordinaten auf die Bezierpunkte zugreifen! (my3dimArray[i][j][k])

    int k = 0;

    for (unsigned int j = 1; j <= ANZAHL + 2; j++)
    {

        for (unsigned int i = 1; i <= ANZAHL + 2 - j; i++)
        {

            k = ANZAHL + 3 - i - j;

            if (int(i - 1 + j + k) == G + 3)
            {

                float n = G * 1.0f;
                float iFloat = i * 1.0f;
                float jFloat = j * 1.0f;
                float kFloat = k * 1.0f;

                my3dimArrayNew[i - 1][j - 1][k - 1] = my3dimArray[i - 1][j][k] * ((iFloat - 1) / (n + 1)) + my3dimArray[i][j - 1][k] * ((jFloat - 1) / (n + 1)) + my3dimArray[i][j][k - 1] * ((kFloat - 1) / (n + 1)); // (*)
            }
        }
    }

    Vec3Array *ergebnis = new Vec3Array;

    // folgende for-Schleifen machen aus einem dreifach-Feld ein Vec3Array
    x = 0;
    y = 0;
    z = 0;

    for (unsigned int stufe = 0; stufe <= ANZAHL; stufe++)
    { // stufe läuft in v-Richtung

        x = 0;
        y = stufe;
        z = ANZAHL - stufe;

        for (unsigned int zaehler = 0; zaehler <= ANZAHL - stufe; zaehler++)
        { // zaehler läuft in u-Richtung
            ergebnis->push_back(my3dimArrayNew[x][y][z]);
            x++;
            y = stufe; // nicht unbedingt notwendig
            z--;
        }
    }
    // ab hier sind die Bezierpunkte in geordneter Reihenfolge in Vec3Array "ergebnis"

    return ergebnis;
}

// Hilfsmethoden zur Erstellung von Punkten

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::createSphere()
{

    // rote Kugel im Ursprung
    ref_ptr<ShapeDrawable> sphere = new ShapeDrawable;
    sphere->setShape(new Sphere(Vec3(0.0f, 0.0f, 0.0f), 0.025));
    sphere->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    ref_ptr<Geode> kugel = new Geode;
    kugel->addDrawable(sphere.get());

    return kugel;
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::createSphere(float radius)
{

    // rote Kugel im Ursprung mit vorgegebenem Radius
    ref_ptr<ShapeDrawable> sphere = new ShapeDrawable;
    sphere->setShape(new Sphere(Vec3(0.0f, 0.0f, 0.0f), radius));
    sphere->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    ref_ptr<Geode> kugel = new Geode;
    kugel->addDrawable(sphere.get());

    return kugel;
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::createSphere(float radius, Vec4 color)
{

    // Kugel im Ursprung mit vorgegebenem Radius und vorgegebener Farbe
    ref_ptr<ShapeDrawable> sphere = new ShapeDrawable;
    sphere->setShape(new Sphere(Vec3(0.0f, 0.0f, 0.0f), radius));
    sphere->setColor(color);

    ref_ptr<Geode> kugel = new Geode;
    kugel->addDrawable(sphere.get());

    return kugel;
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::createSphere(float radius, Vec4 color, Vec3 center)
{
    // allgemeine Kugel
    ref_ptr<ShapeDrawable> sphere = new ShapeDrawable;
    sphere->setShape(new Sphere(center, radius));
    ref_ptr<Geode> kugel = new Geode;

    osg::ref_ptr<osg::Material> material_sphere = new osg::Material();
    material_sphere->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material_sphere->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0] * 0.3f, color[1] * 0.3f, color[2] * 0.3f, color[3]));

    osg::StateSet *stateSet = sphere->getOrCreateStateSet();
    stateSet->setAttribute(material_sphere.get(), osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    sphere->setStateSet(stateSet);
    kugel->addDrawable(sphere.get());

    return kugel;
}

osg::ref_ptr<osg::Geode> TriangleBezierSurfaces::createSpheres(float radius, Vec4 color, Vec3Array *wo)
{

    // allgemeine Kugeln
    ref_ptr<Geode> geode = new Geode;

    for (unsigned int i = 0; i < wo->size(); i++)
    {

        ref_ptr<ShapeDrawable> punkt = new ShapeDrawable;
        punkt->setShape(new Sphere(wo->at(i), radius));
        punkt->setColor(color);
        geode->addDrawable(punkt);
    }

    return geode;
}

// Hilfsmethoden fuer Step 1

void TriangleBezierSurfaces::grad1()
{

    bernstein100 = bernsteinBaryPlot(1, 1, 0, 0, 20, white);

    v100 = new MatrixTransform();
    Matrix m100;
    m100.makeTranslate(-2.0f, -1.25f, 0.0f);
    v100->setMatrix(m100);

    grad1group->addChild(v100);
    v100->addChild(bernstein100);

    bernstein010 = bernsteinBaryPlot(1, 0, 1, 0, 20, white);

    v010 = new MatrixTransform();
    Matrix m010;
    m010.makeTranslate(-0.5f, -1.25f, 0.0f);
    v010->setMatrix(m010);

    grad1group->addChild(v010);
    v010->addChild(bernstein010);

    bernstein001 = bernsteinBaryPlot(1, 0, 0, 1, 20, white);

    v001 = new MatrixTransform();
    Matrix m001;
    m001.makeTranslate(1.0f, -1.25f, 0.0f);
    v001->setMatrix(m001);

    grad1group->addChild(v001);
    v001->addChild(bernstein001);
}

void TriangleBezierSurfaces::grad2()
{

    bernstein200 = bernsteinBaryPlot(2, 2, 0, 0, 20, red);

    v200 = new MatrixTransform();
    Matrix m200;
    m200.makeTranslate(-2.0f, 0.5f, 0.0f);
    v200->setMatrix(m200);

    grad2group->addChild(v200);
    v200->addChild(bernstein200);

    bernstein020 = bernsteinBaryPlot(2, 0, 2, 0, 20, lime);

    v020 = new MatrixTransform();
    Matrix m020;
    m020.makeTranslate(-0.5f, 0.5f, 0.0f);
    v020->setMatrix(m020);

    grad2group->addChild(v020);
    v020->addChild(bernstein020);

    bernstein002 = bernsteinBaryPlot(2, 0, 0, 2, 20, blue);

    v002 = new MatrixTransform();
    Matrix m002;
    m002.makeTranslate(1.0f, 0.5f, 0.0f);
    v002->setMatrix(m002);

    grad2group->addChild(v002);
    v002->addChild(bernstein002);

    bernstein110 = bernsteinBaryPlot(2, 1, 1, 0, 20, purple);

    v110 = new MatrixTransform();
    Matrix m110;
    m110.makeTranslate(-2.0f, 2.0f, 0.0f);
    v110->setMatrix(m110);

    grad2group->addChild(v110);
    v110->addChild(bernstein110);

    bernstein101 = bernsteinBaryPlot(2, 1, 0, 1, 20, yellow);

    v101 = new MatrixTransform();
    Matrix m101;
    m101.makeTranslate(-0.5f, 2.0f, 0.0f);
    v101->setMatrix(m101);

    grad2group->addChild(v101);
    v101->addChild(bernstein101);

    bernstein011 = bernsteinBaryPlot(2, 0, 1, 1, 20, aqua);

    v011 = new MatrixTransform();
    Matrix m011;
    m011.makeTranslate(1.0f, 2.0f, 0.0f);
    v011->setMatrix(m011);

    grad2group->addChild(v011);
    v011->addChild(bernstein011);
}

void TriangleBezierSurfaces::initializeTrennlinie()
{

    trennlinie = new Geode;
    ref_ptr<Geometry> lineStrip = new Geometry;

    ref_ptr<Vec3Array> punkte = new Vec3Array();
    punkte->push_back(Vec3(-3.0f, 0.0f, 0.0f));
    punkte->push_back(Vec3(3.0f, 0.0f, 0.0f));

    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(white);

    lineStrip->setVertexArray(punkte.get());
    lineStrip->setColorArray(colors);
    lineStrip->setColorBinding(Geometry::BIND_OVERALL);
    lineStrip->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, punkte->size()));

    trennlinie->addDrawable(lineStrip);
}

void TriangleBezierSurfaces::checkTrennlinie()
{

    if (showGrad1 && showGrad2)
    {
        sw->addChild(trennlinie);
    }
    else
    {
        sw->removeChild(trennlinie);
    }
}

// Hilfsmethoden fuer Step 2

void TriangleBezierSurfaces::grad3(int genauigkeit)
{

    bernstein003 = bernsteinBaryPlot(3, 0, 0, 3, genauigkeit, color003);

    v003 = new MatrixTransform();
    Matrix m003;
    m003.makeTranslate(-2.75f, -2.0f, 0.0f);
    v003->setMatrix(m003);

    grad3group->addChild(v003);
    v003->addChild(bernstein003);

    bernstein102 = bernsteinBaryPlot(3, 1, 0, 2, genauigkeit, color102);

    v102 = new MatrixTransform();
    Matrix m102;
    m102.makeTranslate(-1.25f, -2.0f, 0.0f);
    v102->setMatrix(m102);

    grad3group->addChild(v102);
    v102->addChild(bernstein102);

    bernstein201 = bernsteinBaryPlot(3, 2, 0, 1, genauigkeit, color201);

    v201 = new MatrixTransform();
    Matrix m201;
    m201.makeTranslate(0.25f, -2.0f, 0.0f);
    v201->setMatrix(m201);

    grad3group->addChild(v201);
    v201->addChild(bernstein201);

    bernstein300 = bernsteinBaryPlot(3, 3, 0, 0, genauigkeit, color300);

    v300 = new MatrixTransform();
    Matrix m300;
    m300.makeTranslate(1.75f, -2.0f, 0.0f);
    v300->setMatrix(m300);

    grad3group->addChild(v300);
    v300->addChild(bernstein300);

    bernstein012 = bernsteinBaryPlot(3, 0, 1, 2, genauigkeit, color012);

    v012 = new MatrixTransform();
    Matrix m012;
    m012.makeTranslate(-2.0f, -0.5f, 0.0f);
    v012->setMatrix(m012);

    grad3group->addChild(v012);
    v012->addChild(bernstein012);

    bernstein111 = bernsteinBaryPlot(3, 1, 1, 1, genauigkeit, color111);

    v111 = new MatrixTransform();
    Matrix m111;
    m111.makeTranslate(-0.5f, -0.5f, 0.0f);
    v111->setMatrix(m111);

    grad3group->addChild(v111);
    v111->addChild(bernstein111);

    bernstein210 = bernsteinBaryPlot(3, 2, 1, 0, genauigkeit, color210);

    v210 = new MatrixTransform();
    Matrix m210;
    m210.makeTranslate(1.0f, -0.5f, 0.0f);
    v210->setMatrix(m210);

    grad3group->addChild(v210);
    v210->addChild(bernstein210);

    bernstein021 = bernsteinBaryPlot(3, 0, 2, 1, genauigkeit, color021);

    v021 = new MatrixTransform();
    Matrix m021;
    m021.makeTranslate(-1.25f, 1.0f, 0.0f);
    v021->setMatrix(m021);

    grad3group->addChild(v021);
    v021->addChild(bernstein021);

    bernstein120 = bernsteinBaryPlot(3, 1, 2, 0, genauigkeit, color120);

    v120 = new MatrixTransform();
    Matrix m120;
    m120.makeTranslate(0.25f, 1.0f, 0.0f);
    v120->setMatrix(m120);

    grad3group->addChild(v120);
    v120->addChild(bernstein120);

    bernstein030 = bernsteinBaryPlot(3, 0, 3, 0, genauigkeit, color030);

    v030 = new MatrixTransform();
    Matrix m030;
    m030.makeTranslate(-0.5f, 2.5f, 0.0f);
    v030->setMatrix(m030);

    grad3group->addChild(v030);
    v030->addChild(bernstein030);
}

// Hilfsmethoden fuer Step 5

void TriangleBezierSurfaces::selectDegree(unsigned int grad)
{

    // ohne diesen Befehl würden die neuen Punkte die alten Eckpunkte nicht überzeichnen...
    sw->removeChildren(0, 100);

    switch (step5degree)
    {
    case 2:
    {
    }
    case 3:
    {
        currentDegree = step5BSgrad3;
        currentPoints = step5BPgrad3;
        sw->addChild(currentDegree);
        break;
    }
    case 4:
    {
        currentDegree = step5BSgrad4;
        currentPoints = step5BPgrad4;
        sw->addChild(currentDegree);
        break;
    }
    case 5:
    {
        currentDegree = step5BSgrad5;
        currentPoints = step5BPgrad5;
        sw->addChild(currentDegree);
        break;
    }
    case 6:
    {
        sliderChanged = false;

        currentDegree = step5BSgrad6;
        currentPoints = step5BPgrad6;
        sw->addChild(currentDegree);
        break;
    }
    case 7:
    {
        currentDegree = step5BSgrad7;
        currentPoints = step5BPgrad7;
        sw->addChild(currentDegree);
        break;
    }
    case 8:
    {
        currentDegree = step5BSgrad8;
        currentPoints = step5BPgrad8;
        sw->addChild(currentDegree);
        break;
    }
    case 9:
    {
        currentDegree = step5BSgrad9;
        currentPoints = step5BPgrad9;
        sw->addChild(currentDegree);
        break;
    }
    case 10:
    {
        currentDegree = step5BSgrad10;
        currentPoints = step5BPgrad10;
        sw->addChild(currentDegree);
        break;
    }
    default:
    {

        break;
    }
    }

    // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?

    step5BSgrad2 = bezierSurfacePlot(2, step5BPgrad2, precision, OrgColorSurface,
                                     colorArray->at(colorSelectorMeshIndex), m_pCheckboxMenuFlaeche->getState(),
                                     m_pCheckboxMenuNetz->getState(), false, false);

    sw->addChild(step5BSgrad2);
}

void TriangleBezierSurfaces::updateDegrees()
{

    step5BPgrad3 = degreeElevation(2, step5BPgrad2);
    step5BPgrad4 = degreeElevation(3, step5BPgrad3);
    step5BPgrad5 = degreeElevation(4, step5BPgrad4);
    step5BPgrad6 = degreeElevation(5, step5BPgrad5);
    step5BPgrad7 = degreeElevation(6, step5BPgrad6);
    step5BPgrad8 = degreeElevation(7, step5BPgrad7);
    step5BPgrad9 = degreeElevation(8, step5BPgrad8);
    step5BPgrad10 = degreeElevation(9, step5BPgrad9);

    step5BSgrad3 = bezierSurfacePlot(3, step5BPgrad3, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad4 = bezierSurfacePlot(4, step5BPgrad4, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad5 = bezierSurfacePlot(5, step5BPgrad5, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad6 = bezierSurfacePlot(6, step5BPgrad6, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad7 = bezierSurfacePlot(7, step5BPgrad7, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad8 = bezierSurfacePlot(8, step5BPgrad8, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad9 = bezierSurfacePlot(9, step5BPgrad9, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
    step5BSgrad10 = bezierSurfacePlot(10, step5BPgrad10, precision, colorArray->at(colorSelectorSurfaceIndex), gruen_grad, false, true, false, false);
}

float TriangleBezierSurfaces::runden(float wert)
{
    return (int)(wert * 100 + 0.5) / 100.0;
}
void TriangleBezierSurfaces::clear_ControlPoints()
{
    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        delete controlPoints[i]; //loescht Interktoren
    }
    controlPoints.clear(); //sonst Absturz bei stepWechsel
}
void TriangleBezierSurfaces::changePresentationStep()
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
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step4();
        VRSceneGraph::instance()->viewAll();
        break;

    case 1: //Bernsteinpolynome-Grad1+2
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(1);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step1();
        VRSceneGraph::instance()->viewAll();
        break;

    case 2: //Bernsteinpolynome-Grad3
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(2);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step2();
        VRSceneGraph::instance()->viewAll();
        break;

    case 3: //beliebiges Bernsteinpolynom
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(3);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step3();
        VRSceneGraph::instance()->viewAll();
        break;

    case 4: //Dreiecks-Bezierflaeche
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(4);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step4();
        VRSceneGraph::instance()->viewAll();
        break;

    case 5: //Graderhoehung
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(5);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step5();
        VRSceneGraph::instance()->viewAll();
        break;

    case 6: //Casteljau-Algorithmus
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(6);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step6();

        VRSceneGraph::instance()->viewAll();
        break;

    case 7: //summary
        root->setNodeMask(root->getNodeMask() & ~Isect::Intersection & ~Isect::Pick); //mit Fläche nichts machbar,aber Menue
        setMenuVisible(7);
        sw->removeChildren(0, 100);
        clear_ControlPoints();
        step4();
        VRSceneGraph::instance()->viewAll();
        break;
    }
}
void TriangleBezierSurfaces::createMenu()
{
    m_pObjectMenu1 = new coRowMenu("Menu-Flaechen");
    m_pObjectMenu1->setVisible(false);
    m_pObjectMenu1->setAttachment(coUIElement::RIGHT);

    //matrices to position the menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;

    // position menu with values from config file
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", 0);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", -5);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 0);
    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.TriangleBezierSurfaces.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.TriangleBezierSurfaces.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.TriangleBezierSurfaces.MenuPosition", pz);
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("s", "COVER.Plugin.TriangleBezierSurfaces.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    m_pObjectMenu1->setTransformMatrix(&matrix);
    m_pObjectMenu1->setScale(cover->getSceneSize() / 2500);

    m_pButtonMenuSchritt = new coButtonMenuItem(coTranslator::coTranslate("Casteljau-Schritte"));
    m_pButtonMenuSchritt->setMenuListener(this);
    m_pObjectMenu1->add(m_pButtonMenuSchritt);

    m_pSliderMenuGenauigkeit = new coSliderMenuItem(coTranslator::coTranslate("Genauigkeit"), -1.0, 1.0, 0.0);
    m_pSliderMenuGenauigkeit->setMenuListener(this);
    m_pSliderMenuGenauigkeit->setInteger(true);
    m_pObjectMenu1->add(m_pSliderMenuGenauigkeit);

    m_pSliderMenu_n = new coSliderMenuItem(coTranslator::coTranslate("n: Grad einstellen"), -1.0, 1.0, 0.0);
    m_pSliderMenu_n->setMenuListener(this);
    m_pSliderMenu_n->setInteger(true);
    m_pObjectMenu1->add(m_pSliderMenu_n);

    m_pSliderMenu_i = new coSliderMenuItem(coTranslator::coTranslate("i"), -1.0, 1.0, 0.0);
    m_pSliderMenu_i->setMenuListener(this);
    m_pSliderMenu_i->setInteger(true);
    m_pObjectMenu1->add(m_pSliderMenu_i);

    m_pSliderMenu_j = new coSliderMenuItem(coTranslator::coTranslate("j"), -1.0, 1.0, 0.0);
    m_pSliderMenu_j->setMenuListener(this);
    m_pSliderMenu_j->setInteger(true);
    m_pObjectMenu1->add(m_pSliderMenu_j);

    m_pSliderMenu_k = new coSliderMenuItem(coTranslator::coTranslate("k = n - i - j"), -1.0, 1.0, 0.0);
    m_pSliderMenu_k->setMenuListener(this);
    m_pSliderMenu_k->setInteger(true);
    m_pSliderMenu_k->setActive(false);
    m_pObjectMenu1->add(m_pSliderMenu_k);

    m_pSliderMenuGrad = new coSliderMenuItem(coTranslator::coTranslate("Grad der Graderhoehung"), -1.0, 1.0, 0.0);
    m_pSliderMenuGrad->setMenuListener(this);
    m_pSliderMenuGrad->setInteger(true);
    m_pObjectMenu1->add(m_pSliderMenuGrad);

    m_pSliderMenu_u = new coSliderMenuItem(coTranslator::coTranslate("u-Koordinate"), -1.0, 1.0, 0.0);
    m_pSliderMenu_u->setMenuListener(this);
    m_pObjectMenu1->add(m_pSliderMenu_u);

    m_pSliderMenu_v = new coSliderMenuItem(coTranslator::coTranslate("v-Koordinate"), -1.0, 1.0, 0.0);
    m_pSliderMenu_v->setMenuListener(this);
    m_pObjectMenu1->add(m_pSliderMenu_v);

    m_pSliderMenu_w = new coSliderMenuItem(coTranslator::coTranslate("w-Koordinate = 1 - u - v"), -1.0, 1.0, 0.0);
    m_pSliderMenu_w->setMenuListener(this);
    m_pSliderMenu_w->setActive(false);
    m_pObjectMenu1->add(m_pSliderMenu_w);

    m_pCheckboxMenuGrad1 = new coCheckboxMenuItem(coTranslator::coTranslate("Grad 1 anzeigen"), true);
    m_pCheckboxMenuGrad1->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuGrad1);

    m_pCheckboxMenuGrad2 = new coCheckboxMenuItem(coTranslator::coTranslate("Grad 2 anzeigen"), false);
    m_pCheckboxMenuGrad2->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuGrad2);

    m_pCheckboxMenuNetz = new coCheckboxMenuItem(coTranslator::coTranslate("Netz anzeigen"), true);
    m_pCheckboxMenuNetz->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuNetz);

    m_pCheckboxMenuUrsprungsnetz = new coCheckboxMenuItem(coTranslator::coTranslate("Ursprungsnetz"), true);
    m_pCheckboxMenuUrsprungsnetz->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuUrsprungsnetz);

    m_pCheckboxMenuFlaeche = new coCheckboxMenuItem(coTranslator::coTranslate("Flaeche anzeigen"), true);
    m_pCheckboxMenuFlaeche->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuFlaeche);

    m_pCheckboxMenuLabels = new coCheckboxMenuItem(coTranslator::coTranslate("Beschriftungen anzeigen"), false);
    m_pCheckboxMenuLabels->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuLabels);

    m_pCheckboxMenuSegment1 = new coCheckboxMenuItem(coTranslator::coTranslate("1. Segmentierungsnetz"), false);
    m_pCheckboxMenuSegment1->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuSegment1);

    m_pCheckboxMenuSegment2 = new coCheckboxMenuItem(coTranslator::coTranslate("2. Segmentierungsnetz"), false);
    m_pCheckboxMenuSegment2->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuSegment2);

    m_pCheckboxMenuSegment3 = new coCheckboxMenuItem(coTranslator::coTranslate("3. Segmentierungsnetz"), false);
    m_pCheckboxMenuSegment3->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuSegment3);

    m_pCheckboxMenuCasteljauNetz = new coCheckboxMenuItem(coTranslator::coTranslate("Casteljau-Netz anzeigen"), false);
    m_pCheckboxMenuCasteljauNetz->setMenuListener(this);
    m_pObjectMenu1->add(m_pCheckboxMenuCasteljauNetz);
}
void TriangleBezierSurfaces::preFrame()
{
    interactPlot = -1;
    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        controlPoints[i]->preFrame();
        controlPoints[i]->showInteractor(true);
        if (controlPoints[i]->interact == true)
        {
            interactPlot = i;
        }
    }
    if (interactPlot != -1)
    { //Interaktor wurde bewegt
        if (m_presentationStep == 4 || m_presentationStep == 7 || m_presentationStep == 0)
        {
            step4BPgrad3->at(interactPlot) = controlPoints[interactPlot]->getPosition();
            sw->removeChild(step4BSgrad3);
            step4BSgrad3 = bezierSurfacePlot(3, step4BPgrad3, precision, OrgColorSurface, OrgColorMesh,
                                             m_pCheckboxMenuFlaeche->getState(), m_pCheckboxMenuNetz->getState(), false, false);
            sw->addChild(step4BSgrad3);
        }
        if (m_presentationStep == 5)
        {
            step5BPgrad2->at(interactPlot) = controlPoints[interactPlot]->getPosition();
            sw->removeChild(step5BSgrad2);
            step5degree = m_pSliderMenuGrad->getValue();
            if (step5degree != 2)
                selectDegree(step5degree);
            else
            {
                sw->removeChild(step5BSgrad3);
                step5BSgrad2 = bezierSurfacePlot(2, step5BPgrad2, precision, OrgColorSurface,
                                                 colorArray->at(colorSelectorMeshIndex), m_pCheckboxMenuFlaeche->getState(),
                                                 m_pCheckboxMenuNetz->getState(), false, false);
            }
            sw->addChild(step5BSgrad2);

            updateDegrees();
        }
        if (m_presentationStep == 6)
        {
            step6BPgrad4->at(interactPlot) = controlPoints[interactPlot]->getPosition();
            sw->removeChild(step6BSgrad4);
            step6BSgrad4 = bezierSurfacePlot(step6degree, step6BPgrad4, precision, OrgColorSurface, OrgColorMesh,
                                             m_pCheckboxMenuFlaeche->getState(), m_pCheckboxMenuNetz->getState(), false, m_pCheckboxMenuCasteljauNetz->getState());
            sw->addChild(step6BSgrad4);
            //geaenderten schwarzen Flaechenpkt neu zeichnen
            Casteljau_FlaechenPkt();

            step6_interact();
        }
    }
    if (step4showCaption == true && m_presentationStep == 4)
    {
        label_b003->setPosition(step4BPgrad3->at(0) * cover->getBaseMat());
        label_b030->setPosition(step4BPgrad3->at(9) * cover->getBaseMat());
        label_b300->setPosition(step4BPgrad3->at(3) * cover->getBaseMat());
        label_b012->setPosition(step4BPgrad3->at(4) * cover->getBaseMat());
        label_b021->setPosition(step4BPgrad3->at(7) * cover->getBaseMat());
        label_b120->setPosition(step4BPgrad3->at(8) * cover->getBaseMat());
        label_b210->setPosition(step4BPgrad3->at(6) * cover->getBaseMat());
        label_b102->setPosition(step4BPgrad3->at(1) * cover->getBaseMat());
        label_b201->setPosition(step4BPgrad3->at(2) * cover->getBaseMat());
        label_b111->setPosition(step4BPgrad3->at(5) * cover->getBaseMat());

        label_b003->show();
        label_b030->show();
        label_b300->show();
        label_b012->show();
        label_b021->show();
        label_b120->show();
        label_b210->show();
        label_b102->show();
        label_b201->show();
        label_b111->show();

        label_b003->showLine();
        label_b030->showLine();
        label_b300->showLine();
        label_b012->showLine();
        label_b021->showLine();
        label_b120->showLine();
        label_b210->showLine();
        label_b102->showLine();
        label_b201->showLine();
        label_b111->showLine();
    }
    else
    {
        label_b003->hide();
        label_b030->hide();
        label_b300->hide();
        label_b012->hide();
        label_b021->hide();
        label_b120->hide();
        label_b210->hide();
        label_b102->hide();
        label_b201->hide();
        label_b111->hide();
    }
}
void TriangleBezierSurfaces::Casteljau_FlaechenPkt()
{
    std::vector<std::vector<std::vector<std::vector<Vec3> > > > bezHelp = triangleCasteljauPoints(
        4, step6BPgrad4, m_pSliderMenu_u->getValue(), m_pSliderMenu_v->getValue(), m_pSliderMenu_w->getValue());

    if (sw->containsNode(flaechenPkt) == true)
    {
        sw->removeChild(flaechenPkt);
    }
    flaechenPkt = createSphere(0.054f + (4.0f / 1000.0f), black, bezHelp[4][0][0][0]);
    if (sw->containsNode(flaechenPkt) == false)
        sw->addChild(flaechenPkt);
}
void TriangleBezierSurfaces::step6_interact()
{
    // (leere) Casteljau-Zeichnung initialisieren
    ref_ptr<Group> newcasteljauGroup = triangleCasteljauPlot(step6degree, step6casteljauSchritt, step6BPgrad4, step6u, step6v, step6w);

    // Neue Casteljau-Gruppe gegen alte tauschen
    sw->replaceChild(casteljauGroup, newcasteljauGroup);
    casteljauGroup = newcasteljauGroup;

    //	---------------------------------------------------------------
    std::vector<std::vector<std::vector<std::vector<Vec3> > > > segmentation = triangleCasteljauPoints(4, step6BPgrad4, step6u, step6v, step6w);

    if (m_pCheckboxMenuSegment1->getState() == true)
    { // 1. Dreiecksteilnetz berechnen und zeichnen
        segmentCounter = 1;
        Vec3Array *segmentationCalc13 = new Vec3Array;

        segmentationCalc13->push_back(segmentation[0][4][0][0]);
        segmentationCalc13->push_back(segmentation[0][3][0][1]);
        segmentationCalc13->push_back(segmentation[0][2][0][2]);
        segmentationCalc13->push_back(segmentation[0][1][0][3]);
        segmentationCalc13->push_back(segmentation[0][0][0][4]);

        segmentationCalc13->push_back(segmentation[1][3][0][0]);
        segmentationCalc13->push_back(segmentation[1][2][0][1]);
        segmentationCalc13->push_back(segmentation[1][1][0][2]);
        segmentationCalc13->push_back(segmentation[1][0][0][3]);

        segmentationCalc13->push_back(segmentation[2][2][0][0]);
        segmentationCalc13->push_back(segmentation[2][1][0][1]);
        segmentationCalc13->push_back(segmentation[2][0][0][2]);

        segmentationCalc13->push_back(segmentation[3][1][0][0]);
        segmentationCalc13->push_back(segmentation[3][0][0][1]);

        segmentationCalc13->push_back(segmentation[4][0][0][0]);

        ref_ptr<Group> seg13 = bezierSurfacePlot(4, segmentationCalc13, precision, OrgColorSurface, fuchsia, false, true, false, false);
        if (mp_seg13)
            sw->replaceChild(mp_seg13, seg13);
        mp_seg13 = seg13;
    }
    if (m_pCheckboxMenuSegment2->getState() == true)
    { // 2. Dreiecksteilnetz berechnen und zeichnen
        segmentCounter = 2;
        Vec3Array *segmentationCalc23 = new Vec3Array;

        segmentationCalc23->push_back(segmentation[0][4][0][0]);
        segmentationCalc23->push_back(segmentation[0][3][1][0]);
        segmentationCalc23->push_back(segmentation[0][2][2][0]);
        segmentationCalc23->push_back(segmentation[0][1][3][0]);
        segmentationCalc23->push_back(segmentation[0][0][4][0]);

        segmentationCalc23->push_back(segmentation[1][3][0][0]);
        segmentationCalc23->push_back(segmentation[1][2][1][0]);
        segmentationCalc23->push_back(segmentation[1][1][2][0]);
        segmentationCalc23->push_back(segmentation[1][0][3][0]);

        segmentationCalc23->push_back(segmentation[2][2][0][0]);
        segmentationCalc23->push_back(segmentation[2][1][1][0]);
        segmentationCalc23->push_back(segmentation[2][0][2][0]);

        segmentationCalc23->push_back(segmentation[3][1][0][0]);
        segmentationCalc23->push_back(segmentation[3][0][1][0]);

        segmentationCalc23->push_back(segmentation[4][0][0][0]);

        ref_ptr<Group> seg23 = bezierSurfacePlot(4, segmentationCalc23, precision, OrgColorSurface, purple, false, true, false, false);
        sw->replaceChild(mp_seg23, seg23);
        mp_seg23 = seg23;
    }
    if (m_pCheckboxMenuSegment3->getState() == true)
    { // 3. Dreiecksteilnetz berechnen und zeichnen
        segmentCounter = 3;
        Vec3Array *segmentationCalc33 = new Vec3Array;

        segmentationCalc33->push_back(segmentation[0][0][0][4]);
        segmentationCalc33->push_back(segmentation[0][0][1][3]);
        segmentationCalc33->push_back(segmentation[0][0][2][2]);
        segmentationCalc33->push_back(segmentation[0][0][3][1]);
        segmentationCalc33->push_back(segmentation[0][0][4][0]);

        segmentationCalc33->push_back(segmentation[1][0][0][3]);
        segmentationCalc33->push_back(segmentation[1][0][1][2]);
        segmentationCalc33->push_back(segmentation[1][0][2][1]);
        segmentationCalc33->push_back(segmentation[1][0][3][0]);

        segmentationCalc33->push_back(segmentation[2][0][0][2]);
        segmentationCalc33->push_back(segmentation[2][0][1][1]);
        segmentationCalc33->push_back(segmentation[2][0][2][0]);

        segmentationCalc33->push_back(segmentation[3][0][0][1]);
        segmentationCalc33->push_back(segmentation[3][0][1][0]);

        segmentationCalc33->push_back(segmentation[4][0][0][0]);

        ref_ptr<Group> seg33 = bezierSurfacePlot(4, segmentationCalc33, precision, OrgColorSurface, blue, false, true, false, false);
        sw->replaceChild(mp_seg33, seg33);
        mp_seg33 = seg33;
    }
    segmentCounter = 0;
}
void TriangleBezierSurfaces::guiToRenderMsg(const grmsg::coGRMsg &msg) 
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
void TriangleBezierSurfaces::showCasteljauNetz()
{
    // Sonderfall 1/3
    if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
    {
        step6u = 0.333f;
        step6v = 0.333f;
        step6w = 0.333f;
    }

    step6casteljauSchritt = 4;
    if (m_pCheckboxMenuCasteljauNetz->getState() == true)
    {
        sw->removeChild(casteljauGroup);
        casteljauGroup = triangleCasteljauPlot(step6degree, 4, step6BPgrad4, step6u, step6v, step6w);
        sw->addChild(casteljauGroup);
        casteljauIsOn = true;
    }
    else
    {
        sw->removeChild(casteljauGroup);
        casteljauIsOn = false;
    }
}
void TriangleBezierSurfaces::Menu_Schritt()
{
    if (m_pCheckboxMenuSegment1->getState() == false)
    {
        sw->removeChild(mp_seg13);
    }
    if (m_pCheckboxMenuSegment2->getState() == false)
    {
        sw->removeChild(mp_seg23);
    }
    if (m_pCheckboxMenuSegment3->getState() == false)
    {
        sw->removeChild(mp_seg33);
    }

    //sonst muss zweimal klicken fuer Anzeige ersten Casteljauschritt
    if (m_pCheckboxMenuCasteljauNetz->getState() == false && m_pCheckboxMenuSegment1->getState() == false
        && m_pCheckboxMenuSegment2->getState() == false && m_pCheckboxMenuSegment3->getState() == false)
    {
        step6casteljauSchritt = 0;
    }
    if (step6casteljauSchritt < step6degree)
    {
        step6u = m_pSliderMenu_u->getValue();
        step6v = m_pSliderMenu_v->getValue();
        step6w = m_pSliderMenu_w->getValue();

        sw->removeChild(casteljauGroup);

        step6casteljauSchritt++;

        casteljauGroup = triangleCasteljauPlot(step6degree, step6casteljauSchritt, step6BPgrad4, step6u, step6v, step6w);

        sw->addChild(casteljauGroup);
    }
    else
    {
        sw->removeChild(casteljauGroup);
        step6casteljauSchritt = 0;

        casteljauGroup = triangleCasteljauPlot(step6degree, step6casteljauSchritt, step6BPgrad4, step6u, step6v, step6w);
        sw->addChild(casteljauGroup);
    }
    if (step6casteljauSchritt != 0)
        m_pCheckboxMenuCasteljauNetz->setState(true);
    else
        m_pCheckboxMenuCasteljauNetz->setState(false);
}
void TriangleBezierSurfaces::Menu_Segment1()
{
    // Sonderfall 1/3
    if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
    {
        step6u = 0.333f;
        step6v = 0.333f;
        step6w = 0.333f;
    }
    if (m_pCheckboxMenuSegment1->getState() == true)
        segmentCounter = 1;
    else
        segmentCounter = 0;

    std::vector<std::vector<std::vector<std::vector<Vec3> > > > segmentation = triangleCasteljauPoints(4, step6BPgrad4, step6u, step6v, step6w);

    if (segmentCounter == 1)
    {
        changeShowSegm = true;
        Vec3Array *segmentationCalc13 = new Vec3Array;

        segmentationCalc13->push_back(segmentation[0][4][0][0]);
        segmentationCalc13->push_back(segmentation[0][3][0][1]);
        segmentationCalc13->push_back(segmentation[0][2][0][2]);
        segmentationCalc13->push_back(segmentation[0][1][0][3]);
        segmentationCalc13->push_back(segmentation[0][0][0][4]);

        segmentationCalc13->push_back(segmentation[1][3][0][0]);
        segmentationCalc13->push_back(segmentation[1][2][0][1]);
        segmentationCalc13->push_back(segmentation[1][1][0][2]);
        segmentationCalc13->push_back(segmentation[1][0][0][3]);

        segmentationCalc13->push_back(segmentation[2][2][0][0]);
        segmentationCalc13->push_back(segmentation[2][1][0][1]);
        segmentationCalc13->push_back(segmentation[2][0][0][2]);

        segmentationCalc13->push_back(segmentation[3][1][0][0]);
        segmentationCalc13->push_back(segmentation[3][0][0][1]);

        segmentationCalc13->push_back(segmentation[4][0][0][0]);
        if (mp_seg13)
            sw->removeChild(mp_seg13);
        mp_seg13 = bezierSurfacePlot(4, segmentationCalc13, precision, OrgColorSurface, fuchsia, false, true, false, false);
        sw->addChild(mp_seg13);
        changeShowSegm = false;
    }
    if (segmentCounter == 0)
    {
        sw->removeChild(mp_seg13);
    }
    if (step6casteljauSchritt != 0) //CasteljauSchritt ist an
        if (sw->containsNode(casteljauGroup) == false)
            sw->addChild(casteljauGroup);
    if (m_pCheckboxMenuCasteljauNetz->getState() == false)
    {
        if (sw->containsNode(casteljauGroup))
            sw->removeChild(casteljauGroup);
    }
}
void TriangleBezierSurfaces::Menu_Segment2()
{
    // Sonderfall 1/3
    if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
    {
        step6u = 0.333f;
        step6v = 0.333f;
        step6w = 0.333f;
    }
    if (m_pCheckboxMenuSegment2->getState() == true)
        segmentCounter = 2;
    else
        segmentCounter = 0;

    std::vector<std::vector<std::vector<std::vector<Vec3> > > > segmentation = triangleCasteljauPoints(4, step6BPgrad4, step6u, step6v, step6w);

    if (segmentCounter == 2)
    {
        changeShowSegm = true;
        Vec3Array *segmentationCalc23 = new Vec3Array;
        segmentationCalc23->push_back(segmentation[0][4][0][0]);
        segmentationCalc23->push_back(segmentation[0][3][1][0]);
        segmentationCalc23->push_back(segmentation[0][2][2][0]);
        segmentationCalc23->push_back(segmentation[0][1][3][0]);
        segmentationCalc23->push_back(segmentation[0][0][4][0]);

        segmentationCalc23->push_back(segmentation[1][3][0][0]);
        segmentationCalc23->push_back(segmentation[1][2][1][0]);
        segmentationCalc23->push_back(segmentation[1][1][2][0]);
        segmentationCalc23->push_back(segmentation[1][0][3][0]);

        segmentationCalc23->push_back(segmentation[2][2][0][0]);
        segmentationCalc23->push_back(segmentation[2][1][1][0]);
        segmentationCalc23->push_back(segmentation[2][0][2][0]);

        segmentationCalc23->push_back(segmentation[3][1][0][0]);
        segmentationCalc23->push_back(segmentation[3][0][1][0]);

        segmentationCalc23->push_back(segmentation[4][0][0][0]);
        if (mp_seg23)
            sw->removeChild(mp_seg23);
        mp_seg23 = bezierSurfacePlot(4, segmentationCalc23, precision, OrgColorSurface, purple, false, true, false, false);
        sw->addChild(mp_seg23);
        changeShowSegm = false;
    }
    if (segmentCounter == 0)
    {
        sw->removeChild(mp_seg23);
    }
    if (step6casteljauSchritt != 0)
        if (sw->containsNode(casteljauGroup) == false)
            sw->addChild(casteljauGroup);
    if (m_pCheckboxMenuCasteljauNetz->getState() == false)
    {
        if (sw->containsNode(casteljauGroup))
            sw->removeChild(casteljauGroup);
    }
}
void TriangleBezierSurfaces::Menu_Segment3()
{
    // Sonderfall 1/3
    if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
    {
        step6u = 0.333f;
        step6v = 0.333f;
        step6w = 0.333f;
    }
    if (m_pCheckboxMenuSegment3->getState() == true)
        segmentCounter = 3;
    else
        segmentCounter = 0;

    std::vector<std::vector<std::vector<std::vector<Vec3> > > > segmentation = triangleCasteljauPoints(4, step6BPgrad4, step6u, step6v, step6w);

    if (segmentCounter == 3)
    {
        changeShowSegm = true;
        Vec3Array *segmentationCalc33 = new Vec3Array;

        segmentationCalc33->push_back(segmentation[0][0][0][4]);
        segmentationCalc33->push_back(segmentation[0][0][1][3]);
        segmentationCalc33->push_back(segmentation[0][0][2][2]);
        segmentationCalc33->push_back(segmentation[0][0][3][1]);
        segmentationCalc33->push_back(segmentation[0][0][4][0]);

        segmentationCalc33->push_back(segmentation[1][0][0][3]);
        segmentationCalc33->push_back(segmentation[1][0][1][2]);
        segmentationCalc33->push_back(segmentation[1][0][2][1]);
        segmentationCalc33->push_back(segmentation[1][0][3][0]);

        segmentationCalc33->push_back(segmentation[2][0][0][2]);
        segmentationCalc33->push_back(segmentation[2][0][1][1]);
        segmentationCalc33->push_back(segmentation[2][0][2][0]);

        segmentationCalc33->push_back(segmentation[3][0][0][1]);
        segmentationCalc33->push_back(segmentation[3][0][1][0]);

        segmentationCalc33->push_back(segmentation[4][0][0][0]);
        if (mp_seg33)
            sw->removeChild(mp_seg33);
        mp_seg33 = bezierSurfacePlot(4, segmentationCalc33, precision, OrgColorSurface, blue, false, true, false, false);
        sw->addChild(mp_seg33);
        changeShowSegm = false;
    }
    if (segmentCounter == 0)
    {
        sw->removeChild(mp_seg33);
    }
    if (step6casteljauSchritt != 0)
        if (sw->containsNode(casteljauGroup) == false)
            sw->addChild(casteljauGroup);
    if (m_pCheckboxMenuCasteljauNetz->getState() == false)
    {
        if (sw->containsNode(casteljauGroup))
            sw->removeChild(casteljauGroup);
    }
}
void TriangleBezierSurfaces::menuEvent(coMenuItem *iMenuItem)
{
    if (iMenuItem == m_pButtonMenuSchritt)
    {
        Menu_Schritt();
    }
    /////////////////
    if (iMenuItem == m_pSliderMenuGenauigkeit)
    {
        changeGenauigkeit = true;
        precision = m_pSliderMenuGenauigkeit->getValue();
        if (m_presentationStep == 2)
        {
            sw->removeChildren(0, 100);
            grad3group->removeChildren(0, grad3group->getNumChildren());
            precision = m_pSliderMenuGenauigkeit->getValue();
            grad3(precision);
            sw->addChild(grad3group);
        }
        else if (m_presentationStep == 4)
        {
            sw->removeChild(step4BSgrad3);
            // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
            step4BSgrad3 = bezierSurfacePlot(3, step4BPgrad3, precision, colorArray->at(colorSelectorSurfaceIndex), colorArray->at(colorSelectorMeshIndex), step4showSurface, step4showMesh, step4showCaption, false);

            sw->addChild(step4BSgrad3);
        }
        else if (m_presentationStep == 5)
        {
            sw->removeChild(step5BSgrad2);

            // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
            step5BSgrad2 = bezierSurfacePlot(2, step5BPgrad2, precision, colorArray->at(colorSelectorSurfaceIndex), colorArray->at(colorSelectorMeshIndex), step5showSurface, step5showOrigin, step5showCaption, false);

            sw->addChild(step5BSgrad2);
        }
        changeGenauigkeit = false;
    }
    /////////////////////
    if (iMenuItem == m_pSliderMenu_n)
    {
        nInt = m_pSliderMenu_n->getValue();

        m_pSliderMenu_i->setMax(nInt);
        iInt = m_pSliderMenu_i->getValue();

        m_pSliderMenu_j->setMax(nInt);
        jInt = m_pSliderMenu_j->getValue();

        kInt = nInt - iInt - jInt;

        if (kInt > 100)
        { //wegen unsigned int, ist nicht < 0
            kInt = 0;
            iInt = nInt - kInt - jInt;
        }
        if (iInt > 100)
        {
            iInt = 0;
            jInt = nInt - iInt - kInt;
        }
        m_pSliderMenu_i->setValue(iInt);
        m_pSliderMenu_j->setValue(jInt);
        m_pSliderMenu_k->setValue(kInt);
        m_pSliderMenu_k->setMax(nInt);
        kInt = m_pSliderMenu_k->getValue();

        sw->removeChild(bernstein);
        bernstein = bernsteinBaryPlot(nInt, iInt, jInt, kInt, 100, white);
        sw->addChild(bernstein);
    }
    ////////////////////
    if (iMenuItem == m_pSliderMenu_i)
    {
        iInt = m_pSliderMenu_i->getValue();
        float w_wert = m_pSliderMenu_n->getValue() - m_pSliderMenu_i->getValue() - m_pSliderMenu_j->getValue();
        m_pSliderMenu_k->setValue(w_wert);
        kInt = w_wert;
        if (w_wert < 0)
        {
            kInt = 0;
            jInt = m_pSliderMenu_n->getValue() - m_pSliderMenu_i->getValue();
            m_pSliderMenu_w->setValue(0);
            m_pSliderMenu_j->setValue(m_pSliderMenu_n->getValue() - m_pSliderMenu_i->getValue());
        }
        sw->removeChild(bernstein);
        bernstein = bernsteinBaryPlot(nInt, iInt, jInt, kInt, 100, white);
        sw->addChild(bernstein);
    }
    ///////////////////
    if (iMenuItem == m_pSliderMenu_j)
    {
        jInt = m_pSliderMenu_j->getValue();
        float w_wert = m_pSliderMenu_n->getValue() - m_pSliderMenu_i->getValue() - m_pSliderMenu_j->getValue();
        m_pSliderMenu_k->setValue(w_wert);
        kInt = w_wert;
        if (w_wert < 0)
        {
            kInt = 0;
            iInt = m_pSliderMenu_n->getValue() - m_pSliderMenu_j->getValue();
            m_pSliderMenu_w->setValue(0);
            m_pSliderMenu_i->setValue(m_pSliderMenu_n->getValue() - m_pSliderMenu_j->getValue());
        }

        sw->removeChild(bernstein);
        bernstein = bernsteinBaryPlot(nInt, iInt, jInt, kInt, 100, white);
        sw->addChild(bernstein);
    }
    //////////////////////////
    if (iMenuItem == m_pSliderMenu_k)
    {
        kInt = m_pSliderMenu_n->getValue() - m_pSliderMenu_i->getValue() - m_pSliderMenu_j->getValue();
        m_pSliderMenu_k->setValue(kInt);

        sw->removeChild(bernstein);
        bernstein = bernsteinBaryPlot(nInt, iInt, jInt, kInt, 100, white);
        sw->addChild(bernstein);
    }
    ///////////////////////
    if (iMenuItem == m_pSliderMenuGrad)
    {
        changeGrad = true;
        step5degree = m_pSliderMenuGrad->getValue();
        selectDegree(step5degree);
        if (step5degree == 2)
        {
            sw->removeChild(step5BSgrad3);
        }
        changeGrad = false;
    }
    //////////////////////
    if (iMenuItem == m_pSliderMenu_u)
    {
        step6u = m_pSliderMenu_u->getValue();

        if (step6u >= 0.995)
        {
            step6u = 1.00;
        }

        if (0.005 > step6u)
        {
            step6u = 0.00;
        }
        float w_wert = 1.0 - m_pSliderMenu_u->getValue() - m_pSliderMenu_v->getValue();

        // Sonderfall 1/3
        if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
        {
            step6u = 0.333f;
            step6v = 0.333f;
            step6w = 0.333f;
            m_pSliderMenu_w->setValue(0.33);
        }
        else
        {
            step6w = w_wert;
            m_pSliderMenu_w->setValue(w_wert);
        }
        if (w_wert < 0.0)
        {
            step6w = 0.00;
            step6v = 1.00 - m_pSliderMenu_u->getValue();
            m_pSliderMenu_w->setValue(0.00);
            m_pSliderMenu_v->setValue(1.00 - m_pSliderMenu_u->getValue());
        }
        Menu_Segment1();
        Menu_Segment2();
        Menu_Segment3();

        if (m_pCheckboxMenuCasteljauNetz->getState() == true)
        {
            sw->removeChild(casteljauGroup);

            casteljauGroup = triangleCasteljauPlot(step6degree, step6casteljauSchritt, step6BPgrad4, step6u, step6v, step6w);
            sw->addChild(casteljauGroup);
        }
        //geaenderten schwarzen Flaechenpkt neu zeichnen
        Casteljau_FlaechenPkt();

        sliderChanged = true;
    }
    ////////////////////////////
    if (iMenuItem == m_pSliderMenu_v)
    {
        step6v = m_pSliderMenu_v->getValue();

        if (step6v >= 0.995)
        {
            step6v = 1.00;
        }
        if (0.005 > step6v)
        {
            step6v = 0.00;
        }
        float w_wert = 1.0 - m_pSliderMenu_u->getValue() - m_pSliderMenu_v->getValue();

        // Sonderfall 1/3
        if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
        {
            step6u = 0.333f;
            step6v = 0.333f;
            step6w = 0.333f;
            m_pSliderMenu_w->setValue(0.33);
        }
        else
        {
            step6w = w_wert;
            m_pSliderMenu_w->setValue(w_wert);
        }
        if (w_wert < 0.0)
        {
            step6w = 0.00;
            step6u = 1.00 - m_pSliderMenu_v->getValue();
            m_pSliderMenu_w->setValue(0.00);
            m_pSliderMenu_u->setValue(1.00 - m_pSliderMenu_v->getValue());
        }
        Menu_Segment1();
        Menu_Segment2();
        Menu_Segment3();
        if (m_pCheckboxMenuCasteljauNetz->getState() == true)
        {
            sw->removeChild(casteljauGroup);

            casteljauGroup = triangleCasteljauPlot(step6degree, step6casteljauSchritt, step6BPgrad4, step6u, step6v, step6w);
            sw->addChild(casteljauGroup);
        }
        //geaenderten schwarzen Flaechenpkt neu zeichnen
        Casteljau_FlaechenPkt();

        sliderChanged = true;
    }
    ///////////////////////
    if (iMenuItem == m_pSliderMenu_w)
    {
        float w_wert = 1.0 - m_pSliderMenu_u->getValue() - m_pSliderMenu_v->getValue();
        // Sonderfall 1/3
        if (m_pSliderMenu_u->getValue() < 0.335 && m_pSliderMenu_u->getValue() >= 0.325 && m_pSliderMenu_v->getValue() < 0.335 && m_pSliderMenu_v->getValue() >= 0.325)
        {
            step6u = 0.333f;
            step6v = 0.333f;
            step6w = 0.333f;
            m_pSliderMenu_w->setValue(0.33);
        }
        else
        {
            m_pSliderMenu_w->setValue(w_wert);
        }
    }
    /////////////////
    if (iMenuItem == m_pCheckboxMenuGrad1)
    {
        if (showGrad1)
        {
            showGrad1 = false;
            sw->removeChild(grad1group);

            checkTrennlinie();
        }
        else
        {
            showGrad1 = true;
            sw->addChild(grad1group);
            checkTrennlinie();
        }
    }
    //////////////////////
    if (iMenuItem == m_pCheckboxMenuGrad2)
    {
        if (showGrad2)
        {
            showGrad2 = false;
            sw->removeChild(grad2group);

            checkTrennlinie();
        }
        else
        {
            showGrad2 = true;
            sw->addChild(grad2group);
            checkTrennlinie();
        }
    }
    /////////////
    if (iMenuItem == m_pCheckboxMenuNetz)
    {
        if (sw->containsNode(step4BSgrad3))
            sw->removeChild(step4BSgrad3);

        if (m_pCheckboxMenuNetz->getState() == false)
        {
            step4showMesh = false;
            step4showCaption = false;
            m_pCheckboxMenuLabels->setState(false);
            clear_ControlPoints(); //Punkte nicht mehr anzeigen
        }
        else
        {
            step4showMesh = true;
        }

        // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
        step4BSgrad3 = bezierSurfacePlot(3, step4BPgrad3, precision,
                                         colorArray->at(colorSelectorSurfaceIndex), colorArray->at(colorSelectorMeshIndex),
                                         step4showSurface, step4showMesh, step4showCaption, false);

        if (sw->containsNode(step4BSgrad3) == false)
            sw->addChild(step4BSgrad3);
    }
    ///////////
    if (iMenuItem == m_pCheckboxMenuUrsprungsnetz)
    {
        if (sw->containsNode(step5BSgrad2))
            sw->removeChild(step5BSgrad2);

        if (step5showOrigin)
        {
            step5showOrigin = false;
            clear_ControlPoints();
        }
        else
        {
            step5showOrigin = true;
        }
        // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
        step5BSgrad2 = bezierSurfacePlot(2, step5BPgrad2, precision, colorArray->at(colorSelectorSurfaceIndex),
                                         colorArray->at(colorSelectorMeshIndex), step5showSurface, step5showOrigin, step5showCaption, false);

        if (sw->containsNode(step5BSgrad2) == false)
            sw->addChild(step5BSgrad2);
    }
    /////////
    if (iMenuItem == m_pCheckboxMenuFlaeche)
    {
        //erstellt osg, kann dann mit C:\EXTERNLIBS\OpenSceneGraph-3.0.1\bin\osgconv
        //zu (.obj) oder .stl(besser) konvertiert werden fuer Blender
        /*const string imagPath = "C:\\Temp/trianglebeziersurface.osg";

	   Node* knoten = root->getChild(0);
	   osgDB::writeNodeFile(*knoten,imagPath);*/

        changeShowFlaeche = true;
        if (m_presentationStep == 4)
        {
            sw->removeChild(step4BSgrad3);

            if (step4showSurface)
            {
                step4showSurface = false;
            }
            else
            {
                step4showSurface = true;
            }
            // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
            step4BSgrad3 = bezierSurfacePlot(3, step4BPgrad3, precision, colorArray->at(colorSelectorSurfaceIndex),
                                             colorArray->at(colorSelectorMeshIndex), step4showSurface, step4showMesh, step4showCaption, false);

            sw->addChild(step4BSgrad3);
        }
        else if (m_presentationStep == 5)
        {

            sw->removeChild(step5BSgrad2);

            if (step5showSurface)
            {
                step5showSurface = false;
            }
            else
            {
                step5showSurface = true;
            }
            // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
            step5BSgrad2 = bezierSurfacePlot(2, step5BPgrad2, precision, colorArray->at(colorSelectorSurfaceIndex), colorArray->at(colorSelectorMeshIndex), step5showSurface, step5showOrigin, step5showCaption, false);

            sw->addChild(step5BSgrad2);
        }
        changeShowFlaeche = false;
    }
    //////////////////
    if (iMenuItem == m_pCheckboxMenuLabels)
    {
        if (step4showCaption)
        {
            step4showCaption = false; //wird in preframe genutzt
        }
        else
        {
            step4showCaption = true;
        }
        if (step4showMesh == false)
        {
            step4showCaption = true;
            step4showMesh = true;
            m_pCheckboxMenuNetz->setState(true);

            sw->removeChild(step4BSgrad3);
            // Grad, Bezierpunkte, Unterteilungen, Flächenfarbe, Netzfarbe, Fläche?, Netz?, Beschriftung?, Casteljau?
            step4BSgrad3 = bezierSurfacePlot(3, step4BPgrad3, precision, colorArray->at(colorSelectorSurfaceIndex),
                                             colorArray->at(colorSelectorMeshIndex), step4showSurface, step4showMesh, step4showCaption, false);

            sw->addChild(step4BSgrad3);
        }
    }
    /////////////
    if (iMenuItem == m_pCheckboxMenuSegment1)
    {
        Menu_Segment1();
    }
    //
    if (iMenuItem == m_pCheckboxMenuSegment2)
    {
        Menu_Segment2();
    }
    //
    if (iMenuItem == m_pCheckboxMenuSegment3)
    {
        Menu_Segment3();
    }
    //
    if (iMenuItem == m_pCheckboxMenuCasteljauNetz)
    {
        showCasteljauNetz();
    }
}

void TriangleBezierSurfaces::setMenuVisible(int step)
{
    m_pObjectMenu1->removeAll();

    m_pCheckboxMenuGrad1->setState(true);
    m_pCheckboxMenuGrad2->setState(false);
    m_pCheckboxMenuNetz->setState(true);
    m_pCheckboxMenuFlaeche->setState(true);
    m_pCheckboxMenuUrsprungsnetz->setState(true);
    m_pCheckboxMenuLabels->setState(false);
    m_pCheckboxMenuCasteljauNetz->setState(false);
    m_pCheckboxMenuSegment1->setState(false);
    m_pCheckboxMenuSegment2->setState(false);
    m_pCheckboxMenuSegment3->setState(false);

    m_pSliderMenuGenauigkeit->setMin(5);
    m_pSliderMenuGenauigkeit->setMax(50);
    m_pSliderMenuGenauigkeit->setPrecision(0);
    m_pSliderMenuGenauigkeit->setValue(20);
    precision = m_pSliderMenuGenauigkeit->getValue();

    m_pSliderMenu_n->setMin(1);
    m_pSliderMenu_n->setMax(9);
    m_pSliderMenu_n->setPrecision(0);
    m_pSliderMenu_n->setValue(6);

    m_pSliderMenu_i->setMin(0);
    m_pSliderMenu_i->setMax(m_pSliderMenu_n->getValue());
    m_pSliderMenu_i->setPrecision(0);
    m_pSliderMenu_i->setValue(1);

    m_pSliderMenu_j->setMin(0);
    m_pSliderMenu_j->setMax(m_pSliderMenu_n->getValue());
    m_pSliderMenu_j->setPrecision(0);
    m_pSliderMenu_j->setValue(2);

    m_pSliderMenu_k->setMin(0);
    m_pSliderMenu_k->setMax(m_pSliderMenu_n->getValue());
    m_pSliderMenu_k->setPrecision(0);
    m_pSliderMenu_k->setValue(3);

    m_pSliderMenuGrad->setMin(2);
    m_pSliderMenuGrad->setMax(10);
    m_pSliderMenuGrad->setPrecision(0);
    m_pSliderMenuGrad->setValue(2);

    m_pSliderMenu_u->setMin(0.00);
    m_pSliderMenu_u->setMax(1.00);
    m_pSliderMenu_u->setPrecision(2);
    m_pSliderMenu_u->setValue(0.333);

    m_pSliderMenu_v->setMin(0.00);
    m_pSliderMenu_v->setMax(1.00);
    m_pSliderMenu_v->setPrecision(2);
    m_pSliderMenu_v->setValue(0.333);

    m_pSliderMenu_w->setMin(0.00);
    m_pSliderMenu_w->setMax(1.00);
    m_pSliderMenu_w->setPrecision(2);
    m_pSliderMenu_w->setValue(0.333);

    if (step == 1)
    {
        m_pObjectMenu1->add(m_pCheckboxMenuGrad1);
        m_pObjectMenu1->add(m_pCheckboxMenuGrad2);
    }
    if (step == 2)
    {
        m_pObjectMenu1->add(m_pSliderMenuGenauigkeit);
    }
    if (step == 3)
    {
        m_pObjectMenu1->add(m_pSliderMenu_n);
        m_pObjectMenu1->add(m_pSliderMenu_i);
        m_pObjectMenu1->add(m_pSliderMenu_j);
        m_pObjectMenu1->add(m_pSliderMenu_k);
    }
    if (step == 4)
    {
        m_pObjectMenu1->add(m_pCheckboxMenuNetz);
        m_pObjectMenu1->add(m_pCheckboxMenuFlaeche);
        m_pObjectMenu1->add(m_pCheckboxMenuLabels);
        m_pObjectMenu1->add(m_pSliderMenuGenauigkeit);
    }
    if (step == 5)
    {
        m_pObjectMenu1->add(m_pCheckboxMenuFlaeche);
        m_pObjectMenu1->add(m_pCheckboxMenuUrsprungsnetz);
        m_pObjectMenu1->add(m_pSliderMenuGrad);
        m_pObjectMenu1->add(m_pSliderMenuGenauigkeit);
    }
    if (step == 6)
    {
        m_pObjectMenu1->add(m_pSliderMenu_u);
        m_pObjectMenu1->add(m_pSliderMenu_v);
        m_pObjectMenu1->add(m_pSliderMenu_w);
        m_pObjectMenu1->add(m_pButtonMenuSchritt);
        m_pObjectMenu1->add(m_pCheckboxMenuSegment1);
        m_pObjectMenu1->add(m_pCheckboxMenuSegment2);
        m_pObjectMenu1->add(m_pCheckboxMenuSegment3);
        m_pObjectMenu1->add(m_pCheckboxMenuCasteljauNetz);
    }

    m_pObjectMenu1->setVisible(true);

    VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
}

COVERPLUGIN(TriangleBezierSurfaces)
