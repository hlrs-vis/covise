/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                                                                      *
 *                            (C) 1996-2000                             *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 *   File            VRSceneGraph.C (Performer 2.0)                     *
 *                                                                      *
 *   Description     scene graph class                                  *
 *                                                                      *
 *   Author          D. Rainer                                          *
 *                 F. Foehl                                             *
 *                 U. Woessner                                          *
 *                                                                      *
 *   Date            20.08.97                                           *
 *                 10.07.98 Performer C++ Interface                     *
 *                 20.11.00 Pinboard config through covise.config       *
 ************************************************************************/

#include <util/common.h>
#include <config/CoviseConfig.h>
#include <net/tokenbuffer.h>

#include "OpenCOVER.h"
#include "coVRFileManager.h"
#include "coVRNavigationManager.h"
#include "coVRCollaboration.h"
#include "coVRLighting.h"
#include "VRSceneGraph.h"
#include "coVRPluginList.h"
#include "VRRegisterSceneGraph.h"
#include "coVRPluginSupport.h"
#include "coVRMSController.h"
#include "VRViewer.h"
#include "coVRSelectionManager.h"
#include <osgGA/GUIActionAdapter>
#include "coVRLabel.h"
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <input/VRKeys.h>
#include <input/input.h>
#include "coVRConfig.h"
#include <OpenVRUI/coJoystickManager.h>
#include "coVRIntersectionInteractorManager.h"
#include "coVRShadowManager.h"

#include <osg/LightModel>
#include <osg/ClipNode>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Sequence>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Group>
#include <osg/PolygonMode>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/StateAttribute>
#include <osg/Depth>
#include <osg/ComputeBoundsVisitor>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/Version>
#include <osg/io_utils>
#include <osg/PointSprite>
#include <osgDB/WriteFile>
#include <osgFX/Scribe>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <OpenVRUI/osg/mathUtils.h>

#include "ui/Menu.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/SelectionList.h"
#include "ui/View.h"

using namespace osg;
using namespace opencover;
using covise::coCoviseConfig;

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

VRSceneGraph *VRSceneGraph::s_instance = NULL;

VRSceneGraph *VRSceneGraph::instance()
{
    if (!s_instance)
        s_instance = new VRSceneGraph;
    return s_instance;
}

VRSceneGraph::VRSceneGraph()
    : ui::Owner("VRSceneGraph", cover->ui)
    , m_vectorInteractor(0)
    , m_pointerDepth(0.f)
    , m_floorHeight(-1250.0)
    , m_wireframe(Disabled)
    , m_textured(true)
    , m_showMenu(true)
    , m_showObjects(true)
    , m_pointerType(0)
    , m_scaleMode(-1.0)
    , m_scaleFactor(1.0f)
    , m_scaleFactorButton(0.1f)
    , m_scaleRestrictFactorMin(0.0f)
    , m_scaleRestrictFactorMax(0.0f)
    , m_transRestrictMinX(0.0f)
    , m_transRestrictMinY(0.0f)
    , m_transRestrictMinZ(0.0f)
    , m_transRestrictMaxX(0.0f)
    , m_transRestrictMaxY(0.0f)
    , m_transRestrictMaxZ(0.0f)
    , m_scalingAllObjects(false)
    , m_scaleTransform(NULL)
    , isScenegraphProtected_(false)
    , m_enableHighQualityOption(true)
    , m_switchToHighQuality(false)
    , m_highQuality(false)
    , m_interactionHQ(NULL)
{
    assert(!s_instance);
}
/*bis hier neu*/

void VRSceneGraph::init()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew VRSceneGraph\n");

    readConfigFile();

    initMatrices();

    initSceneGraph();

    // create coordinate axis and put them in
    // scenegraph acoording to config
    initAxis();

    // create hand device geometries
    initHandDeviceGeometry();

    coVRLighting::instance();

    emptyProgram_ = new osg::Program();

    m_interactionHQ = new vrui::coCombinedButtonInteraction(vrui::coInteraction::AllButtons, "Anchor", vrui::coInteraction::Highest);
    m_interactionHQ->setNotifyOnly(true);

    m_trackHead = new ui::Button(cover->viewOptionsMenu, "TrackHead");
    m_trackHead->setText("Track head");
    m_trackHead->setShortcut("Shift+F");
    m_trackHead->setState(!coVRConfig::instance()->frozen());
    m_trackHead->setCallback([this](bool state){
        toggleHeadTracking(state);
    });

    m_allowHighQuality= new ui::Button(cover->viewOptionsMenu, "AllowHighQuality");
    cover->viewOptionsMenu->add(m_allowHighQuality);
    m_allowHighQuality->setState(m_enableHighQualityOption);
    m_allowHighQuality->setText("Allow high quality");
    m_allowHighQuality->setCallback([this](bool state){
        toggleHighQuality(state);
    });

    auto switchToHighQuality = new ui::Action(cover->viewOptionsMenu, "SwitchToHighQuality");
    switchToHighQuality->setVisible(false);
    cover->viewOptionsMenu->add(switchToHighQuality);
    switchToHighQuality->setText("Enable high quality rendering");
    switchToHighQuality->setShortcut("Shift+H");
    switchToHighQuality->setCallback([this](){
        if (m_enableHighQualityOption && !m_highQuality)
            m_switchToHighQuality = true;
    });

    m_drawStyle = new ui::SelectionList(cover->viewOptionsMenu, "DrawStyle");
    m_drawStyle->setText("Draw style");
    m_drawStyle->append("As is");
    m_drawStyle->append("Wireframe");
    m_drawStyle->append("Hidden lines (dark)");
    m_drawStyle->append("Hidden lines (bright)");
    m_drawStyle->append("Points");
    cover->viewOptionsMenu->add(m_drawStyle);
    m_drawStyle->setShortcut("Alt+w");
    m_drawStyle->setCallback([this](int style){
        setWireframe(WireframeMode(style));
    });
    m_drawStyle->select(0);

    m_showAxis = new ui::Button(cover->viewOptionsMenu, "ShowAxis");
    cover->viewOptionsMenu->add(m_showAxis);
    m_showAxis->setState(m_coordAxis);
    m_showAxis->setText("Show axis");
    m_showAxis->setShortcut("Shift+A");
    m_showAxis->setCallback([this](bool state){
        toggleAxis(state);
    });

    m_useTextures = new ui::Button(cover->viewOptionsMenu, "Texturing");
    m_useTextures->setVisible(false, ui::View::VR);
    m_useTextures->setShortcut("Shift+T");
    m_useTextures->setState(m_textured);
    cover->viewOptionsMenu->add(m_useTextures);
    m_useTextures->setCallback([this](bool state){
        m_textured = state;
        osg::Texture *texture = new osg::Texture2D;
        for (int unit=0; unit<4; ++unit)
        {
            if (m_textured)
                m_objectsStateSet->setTextureAttributeAndModes(unit, texture, osg::StateAttribute::OFF);
            else
                m_objectsStateSet->setTextureAttributeAndModes(unit, texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
        }
    });

    m_useShaders = new ui::Button(cover->viewOptionsMenu, "UseShaders");
    m_useShaders->setText("Use shaders");
    m_useShaders->setVisible(false, ui::View::VR);
    m_useShaders->setShortcut("Alt+s");
    m_useShaders->setState(m_shaders);
    cover->viewOptionsMenu->add(m_useShaders);
    m_useShaders->setCallback([this](bool state){
        m_shaders = state;
        osg::Program *program = new osg::Program;
        if (m_shaders)
        {
            m_objectsStateSet->setAttributeAndModes(program, osg::StateAttribute::OFF);
        }
        else
        {
            m_objectsStateSet->setAttributeAndModes(program, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
        }
    });


    auto tm = new ui::Button(cover->viewOptionsMenu, "ShowMenu");
    cover->viewOptionsMenu->add(tm);
    tm->setText("Show VR menu");
    tm->setState(m_showMenu);
    tm->addShortcut("m");
    tm->setCallback([this](bool state){
            setMenu(state);
            });
    tm->setVisible(false);
    m_showMenuButton = tm;

    auto fc = new ui::Action(cover->viewOptionsMenu, "ForceCompile");
    fc->setText("Force display list compilation");
    fc->setVisible(false);
    fc->addShortcut("Alt+Shift+C");
    fc->setCallback([this](){
            VRViewer::instance()->forceCompile();
    });

    auto fs = new ui::Action(cover->viewOptionsMenu, "FlipStereo");
    fs->setText("Flip stereo 3D eyes");
    fs->setVisible(false);
    fs->addShortcut("Alt+e");
    fs->setCallback([this](){
            VRViewer::instance()->flipStereo();
    });

    auto cw = new ui::Action(cover->viewOptionsMenu, "ClearWindow");
    cw->setText("Clear window");
    cw->setVisible(false);
    cw->addShortcut("Alt+c");
    cw->setCallback([this](){
            windowStruct *ws = &(coVRConfig::instance()->windows[0]);
            ws->window->setWindowRectangle(ws->ox, ws->oy, ws->sx, ws->sy);
            ws->window->setWindowDecoration(false);
            VRViewer::instance()->clearWindow = true;
    });
}

VRSceneGraph::~VRSceneGraph()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete VRSceneGraph\n");

    m_specialBoundsNodeList.clear();

    delete coVRShadowManager::instance();

    while (m_objectsRoot->getNumChildren() > 0)
    {
        int n = m_objectsRoot->getNumChildren();
        osg::ref_ptr<osg::Node> thisNode = m_objectsRoot->getChild(n-1);
        while (thisNode && thisNode->getNumParents() > 0)
            thisNode->getParent(0)->removeChild(thisNode.get());
    }

    s_instance = NULL;
}

void VRSceneGraph::config()
{
    // if menu mode is on -- hide menus
    if (coVRConfig::instance()->isMenuModeOn())
        setMenuMode(false);

    setMenu(m_showMenu);
}

int VRSceneGraph::readConfigFile()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::readConfigFile\n");

    m_scaleFactor = coCoviseConfig::getFloat("COVER.DefaultScaleFactor", 1.0f);

    m_floorHeight = coCoviseConfig::getFloat("COVER.FloorHeight", -1250.0f);

    m_scaleAllOn = coCoviseConfig::isOn("COVER.ScaleAll", false);

    std::string line = coCoviseConfig::getEntry("COVER.ModelFile");
    if (!line.empty())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "loading model file '%s'\n", line.c_str());
        coVRFileManager::instance()->loadFile(line.c_str());
    }

    m_coordAxis = coCoviseConfig::isOn("COVER.CoordAxis", false);

    wiiPos = coCoviseConfig::getFloat("COVER.WiiPointerPos", -250.0);

    bool configured = false;
    m_showMenu = coCoviseConfig::isOn("visible", "COVER.UI.VRUI", m_showMenu, &configured);

    return 0;
}

void VRSceneGraph::initMatrices()
{
    m_invBaseMatrix.makeIdentity();
    m_oldInvBaseMatrix.makeIdentity();
}

void VRSceneGraph::initSceneGraph()
{
    // create the scene node
    m_scene = coVRShadowManager::instance()->newScene();
    m_scene->setName("VR_RENDERER_SCENE_NODE");
    std::string shadowTechnique = covise::coCoviseConfig::getEntry("value","COVER.ShadowTechnique","none");
    coVRShadowManager::instance()->setTechnique(shadowTechnique);

    m_objectsScene = new osg::Group();
    m_objectsScene->setName("VR_RENDER_OBJECT_SCENE_NODE");

    // Create a lit scene osg::StateSet for the scene
    m_rootStateSet = loadGlobalGeostate();
    m_objectsStateSet = loadGlobalGeostate();

    int numClipPlanes = cover->getNumClipPlanes();
    for (int i = 0; i < numClipPlanes; i++)
    {
        m_rootStateSet->setAttributeAndModes(cover->getClipPlane(i), osg::StateAttribute::OFF);
        m_objectsStateSet->setAttributeAndModes(cover->getClipPlane(i), osg::StateAttribute::OFF);
    }

    // attach the osg::StateSet to the scene
    m_scene->setStateSet(m_rootStateSet);
    m_objectsScene->setStateSet(m_rootStateSet);

    // create the pointer (hand) DCS node
    // add it to the scene graph as the first child
    //                                  -----
    m_handTransform = new osg::MatrixTransform();
    m_handTransform->setName("HandTransform");
    m_handIconScaleTransform = new osg::MatrixTransform();
    m_handIconScaleTransform->setName("HandIconScaleTransform");
    m_handAxisScaleTransform = new osg::MatrixTransform();
    m_handAxisScaleTransform->setName("HandAxisScaleTransform");
    m_pointerDepthTransform = new osg::MatrixTransform();
    m_pointerDepthTransform->setName("PointerDepthTransform");
    m_handTransform->addChild(m_pointerDepthTransform.get());
    m_handTransform->addChild(m_handAxisScaleTransform);
    m_pointerDepthTransform->addChild(m_handIconScaleTransform);

    // scale the axis to cover->getSceneSize()/2500.0;
    const float scale = cover->getSceneSize() / 2500.0;
    m_handAxisScaleTransform->setMatrix(osg::Matrix::scale(scale, scale, scale));

    // read icons size from covise.config, default is sceneSize/10
    m_handIconSize = coCoviseConfig::getFloat("COVER.IconSize", cover->getSceneSize() / 25.0);
    m_handIconOffset = coCoviseConfig::getFloat("COVER.IconOffset", 0.0);

    // read icon displacement
    m_pointerDepth = coCoviseConfig::getFloat("COVER.PointerDepth", m_pointerDepth);
    m_pointerDepthTransform->setMatrix(osg::Matrix::translate(0, m_pointerDepth, 0));

    // do not intersect with objects under handDCS
    m_handTransform->setNodeMask(m_handTransform->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    // we: render Menu first
    m_menuGroupNode = new osg::Group();
    m_menuGroupNode->setNodeMask(m_menuGroupNode->getNodeMask()&~Isect::ReceiveShadow);
    m_menuGroupNode->setName("MenuGroupNode");
    if (m_showMenu)
        m_scene->addChild(m_menuGroupNode.get());

    // dcs for translating/rotating all objects
    m_objectsTransform = new osg::MatrixTransform();
    m_objectsTransform->setName("ObjectsTransform");
    m_objectsTransform->setStateSet(m_objectsStateSet);

    // dcs for scaling all objects
    m_scaleTransform = new osg::MatrixTransform();
    m_scaleTransform->setName("ScaleTransform");
    m_objectsTransform->addChild(m_scaleTransform);
    m_scene->addChild(m_objectsTransform);
    m_objectsScene->addChild(m_objectsTransform);

    // root node for all objects
    osg::ClipNode *clipNode = new osg::ClipNode();
    m_objectsRoot = clipNode;
    m_objectsRoot->setName("OBJECTS_ROOT");

    m_scaleTransform->addChild(m_objectsRoot);

    osg::StateSet *ss = m_menuGroupNode->getOrCreateStateSet();
    for (int i = 0; i < cover->getNumClipPlanes(); i++)
    {
        ss->setAttributeAndModes(cover->getClipPlane(i), osg::StateAttribute::OFF);
    }

    m_alwaysVisibleGroupNode = new osg::Group();
    m_alwaysVisibleGroupNode->setNodeMask(m_alwaysVisibleGroupNode->getNodeMask()&~Isect::ReceiveShadow);
    m_alwaysVisibleGroupNode->setName("AlwaysVisibleGroupNode");
    m_scene->addChild(m_alwaysVisibleGroupNode.get());
    osg::StateSet *ssav = m_alwaysVisibleGroupNode->getOrCreateStateSet();
    ssav->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF|osg::StateAttribute::PROTECTED|osg::StateAttribute::OVERRIDE);
    ssav->setRenderBinDetails(INT_MAX, "RenderBin");
    //ssav->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    ssav->setNestRenderBins(false);
    for (int i = 0; i < cover->getNumClipPlanes(); i++)
    {
        ssav->setAttributeAndModes(cover->getClipPlane(i), osg::StateAttribute::OFF);
    }

    m_lineHider = new osgFX::Scribe();
}

void VRSceneGraph::initAxis()
{
    // geode for coordinate system axis
    m_worldAxis = loadAxisGeode(4);
    m_handAxis = loadAxisGeode(2);
    m_viewerAxis = loadAxisGeode(4);
    m_objectAxis = loadAxisGeode(0.01f);
    m_viewerAxisTransform = new osg::MatrixTransform();
    m_viewerAxisTransform->setName("ViewerAxisTransform");
    m_scene->addChild(m_viewerAxisTransform.get());

    showSmallSceneAxis_ = coCoviseConfig::isOn("COVER.SmallSceneAxis", false);
    if (showSmallSceneAxis_)
    {
        float sx = 0.1 * cover->getSceneSize();
        sx = coCoviseConfig::getFloat("COVER.SmallSceneAxis.Size", sx);
        if (cover->debugLevel(3))
            fprintf(stderr, "COVER.SmallSceneAxis.Size=%f\n", sx);
        float xp = coCoviseConfig::getFloat("x", "COVER.SmallSceneAxis.Position", 500.0);
        float yp = coCoviseConfig::getFloat("y", "COVER.SmallSceneAxis.Position", 0.0);
        float zp = coCoviseConfig::getFloat("z", "COVER.SmallSceneAxis.Position", -500);
        m_smallSceneAxis = loadAxisGeode(0.01 * sx);
        m_smallSceneAxis->setNodeMask(m_objectAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
        m_smallSceneAxisTransform = new osg::MatrixTransform();
        m_smallSceneAxisTransform->setName("SmallSceneAxisTransform");
        m_smallSceneAxisTransform->setMatrix(osg::Matrix::translate(xp, yp, zp));
        m_scene->addChild(m_smallSceneAxisTransform.get());

        coVRLabel *XLabel, *YLabel, *ZLabel;
        float fontSize = coCoviseConfig::getFloat("COVER.SmallSceneAxis.FontSize", 100);
        float lineLen = 0;
        Vec4 red(1, 0, 0, 1);
        Vec4 green(0, 1, 0, 1);
        Vec4 blue(0, 0, 1, 1);
        Vec4 bg(1, 1, 1, 0);
        XLabel = new coVRLabel("X", fontSize, lineLen, red, bg);
        XLabel->reAttachTo(m_smallSceneAxisTransform.get());
        XLabel->setPosition(osg::Vec3(1.1 * sx, 0, 0));
        XLabel->hideLine();

        YLabel = new coVRLabel("Y", fontSize, lineLen, green, bg);
        YLabel->reAttachTo(m_smallSceneAxisTransform.get());
        YLabel->setPosition(osg::Vec3(0, 1.1 * sx, 0));
        YLabel->hideLine();

        ZLabel = new coVRLabel("Z", fontSize, lineLen, blue, bg);
        ZLabel->reAttachTo(m_smallSceneAxisTransform.get());
        ZLabel->setPosition(osg::Vec3(0, 0, 1.1 * sx));
        ZLabel->hideLine();

        m_smallSceneAxisTransform->addChild(m_smallSceneAxis.get());
    }

    // no intersection with the axis
    m_worldAxis->setNodeMask(m_worldAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_handAxis->setNodeMask(m_handAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_viewerAxis->setNodeMask(m_viewerAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_objectAxis->setNodeMask(m_objectAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    if (coCoviseConfig::isOn("COVER.CoordAxis", false))
    {
        m_scene->addChild(m_worldAxis.get());
        m_viewerAxisTransform->addChild(m_viewerAxis.get());
        m_handAxisScaleTransform->addChild(m_handAxis.get());
        m_scaleTransform->addChild(m_objectAxis.get());
    }
}

void VRSceneGraph::initHandDeviceGeometry()
{
    m_handLine = loadHandLine();
    if (m_handLine)
        m_handTransform->addChild(m_handLine.get());

    // add hand device geometry
    m_handSphere = loadHandIcon("HandSphere");
    m_handPlane = loadHandIcon("Plane");
    m_handCube = loadHandIcon("XForm");
    m_handFly = loadHandIcon("Fly");
    m_handDrive = loadHandIcon("Drive");
    m_handWalk = loadHandIcon("Walk");
    m_handPyramid = loadHandIcon("Scale");
    m_handProbe = loadHandIcon("Probe");
    m_handNormal = loadHandIcon("HandNormal");
}

// process key events
bool VRSceneGraph::keyEvent(int type, int keySym, int mod)
{
    bool handled = false;

    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::keyEvent\n");

    // Beschleunigung
    if (type == osgGA::GUIEventAdapter::KEYUP)
    {
    }
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == '%')
        {
            //matrixCorrect->separateColor = !matrixCorrect->separateColor;
            //fprintf(stderr,"separateColor = %d\n",matrixCorrect->separateColor);
            //handled=true;
        }

        if (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "alt: sym=%d\n", keySym);
            if (keySym == 'i' || keySym == 710) // i
            {
                saveScenegraph(false);
                handled = true;
            }
            else if (keySym == 'W' || keySym == 8722) // W
            {
                saveScenegraph(true);
                handled = true;
            }
            else if (keySym == 'm' || keySym == 181) //m
            {
            }
#ifdef _OPENMP
            else if (keySym >= '0' && keySym <= '9')
            {
                static int numInitialThreads = omp_get_max_threads();
                int targetNumThreads = (keySym == '0' ? numInitialThreads : keySym - '0');
                if (std::cerr.bad())
                    std::cerr.clear();
                std::cerr << "Setting maximum OpenMP threads to " << targetNumThreads << std::endl;
                omp_set_num_threads(targetNumThreads);
                handled = true;
            }
#endif
        } // unmodified keys
    }
    return handled;
}

void
VRSceneGraph::setObjects(bool state)
{
    m_showObjects = state;

    if (m_showObjects)
    {
        if (m_objectsTransform->getNumParents() == 1)
        {
            if (m_wireframe == HiddenLineWhite || m_wireframe == HiddenLineBlack)
                m_scene->addChild(m_lineHider.get());
            else
                m_scene->addChild(m_objectsTransform.get());
        }
    }
    else
    {
        m_scene->removeChild(m_objectsTransform.get());
        m_lineHider->removeChild(m_objectsTransform.get());
    }
}

void
VRSceneGraph::setMenu(bool state)
{
    m_showMenu = state;
    m_showMenuButton->setState(state);

    if (m_showMenu)
    {
        if (m_menuGroupNode->getNumParents() == 0)
        {
            m_scene->addChild(m_menuGroupNode.get());
        }
    }
    else
    {
        m_scene->removeChild(m_menuGroupNode.get());
    }
}

bool
VRSceneGraph::menuVisible() const
{
    return m_showMenu;
}

void VRSceneGraph::setMenuMode(bool state)
{
    if (state)
    {
        menusAreHidden = false;

        applyMenuModeToMenus();

        // set joystickmanager active
        vrui::coJoystickManager::instance()->setActive(true);

        // set scene and documents not intersectable
        m_objectsTransform->setNodeMask(m_objectsTransform->getNodeMask() & (~Isect::Intersection));
        // set color of scene to grey
        if (coVRConfig::instance()->colorSceneInMenuMode())
        {
            osg::Material *mat = new osg::Material();
            mat->setColorMode(osg::Material::OFF);
            mat->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.8, 0.8, 0.8, 0.5));
            mat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0, 0.0, 0.0, 0.0));
            mat->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0, 0.0, 0.0, 0.0));
            mat->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0, 0.0, 0.0, 0.0));
            m_rootStateSet->setAttributeAndModes(mat, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            // disable shader
            m_rootStateSet->setAttributeAndModes(emptyProgram_, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_NORMALIZE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            m_rootStateSet->setMode(osg::StateAttribute::PROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::VERTEXPROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::FRAGMENTPROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::TEXTURE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
        }
    }
    else
    {
        menusAreHidden = true;

        // set color of scene
        if (coVRConfig::instance()->colorSceneInMenuMode())
        {
            osg::Material *mat = new osg::Material();
            m_rootStateSet->setAttributeAndModes(mat, osg::StateAttribute::ON);
            // disable shader override
            m_rootStateSet->removeAttribute(emptyProgram_);
            m_rootStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_BLEND, osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::PROGRAM, osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::VERTEXPROGRAM, osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::FRAGMENTPROGRAM, osg::StateAttribute::OFF);
            m_rootStateSet->setMode(osg::StateAttribute::TEXTURE, osg::StateAttribute::OFF);
        }
        // set scene intersectable
        m_objectsTransform->setNodeMask(0xffffffff);

#if 0
        // hide quit menu
        for (unsigned int i = 0; i < m_menuGroupNode->getNumChildren(); i++)
        {
            if (m_menuGroupNode->getChild(i)->getName().find("Quit") != std::string::npos || m_menuGroupNode->getChild(i)->getName().find("beenden") != std::string::npos)
            {
                VRPinboard::instance()->hideQuitMenu();
                break;
            }
        }
#endif

        applyMenuModeToMenus();

        // set joystickmanager inactive
        vrui::coJoystickManager::instance()->setActive(false);
    }
}

void VRSceneGraph::applyMenuModeToMenus()
{
    if (!coVRConfig::instance()->isMenuModeOn())
    {
        return;
    }

    for (unsigned int i = 0; i < m_menuGroupNode->getNumChildren(); i++)
    {
        if ((m_menuGroupNode->getChild(i)->getName().find("Quit") == std::string::npos)
            && (m_menuGroupNode->getChild(i)->getName().find("beenden") == std::string::npos)
            && (m_menuGroupNode->getChild(i)->getName().find("Document") == std::string::npos))
        {
            if (menusAreHidden)
                m_menuGroupNode->getChild(i)->setNodeMask(0x0);
            else
                m_menuGroupNode->getChild(i)->setNodeMask(0xffffffff & ~Isect::ReceiveShadow);
        }
    }
}

void VRSceneGraph::toggleHeadTracking(bool state)
{
    coVRConfig::instance()->setFrozen(!state);
    m_trackHead->setState(state);
}

void
VRSceneGraph::toggleMenu()
{
    m_showMenu = !m_showMenu;

    setMenu(m_showMenu);
}

void VRSceneGraph::setScaleFactor(float s, bool sync)
{
    m_scaleFactor = s;
    osg::Matrix scale_mat;
    // restrict scale
    if (m_scaleRestrictFactorMin != m_scaleRestrictFactorMax)
    {
        if (m_scaleFactor < m_scaleRestrictFactorMin)
            m_scaleFactor = m_scaleRestrictFactorMin;
        else if (m_scaleFactor > m_scaleRestrictFactorMax)
            m_scaleFactor = m_scaleRestrictFactorMax;
    }
    scale_mat.makeScale(m_scaleFactor, m_scaleFactor, m_scaleFactor);
    if (coVRNavigationManager::instance()->getRotationPointActive())
    {
        osg::Vec3 tv = coVRNavigationManager::instance()->getRotationPoint();
        tv = tv * -1;
        osg::Matrix trans;
        trans.makeTranslate(tv);
        osg::Matrix temp = trans * scale_mat;
        trans.makeTranslate(coVRNavigationManager::instance()->getRotationPoint());
        scale_mat.mult(temp, trans);
    }
    m_scaleTransform->setMatrix(scale_mat);
}

void
VRSceneGraph::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "VRSceneGraph::update\n");
    if (m_firstTime)
    {
        if (coVRConfig::instance()->numWindows() > 0)
        {
            const auto &ws = coVRConfig::instance()->windows[0];
            if (ws.window)
            {
#ifdef WIN32
                if (OpenCOVER::instance()->parentWindow == NULL)
#endif
                {
                    ws.window->setWindowRectangle(ws.ox, ws.oy, ws.sx, ws.sy);
                    ws.window->setWindowDecoration(ws.decoration);
                }
            }
        }
        m_firstTime = false;
    }

    if (Input::instance()->hasHand() && Input::instance()->isTrackingOn())
    {
        if (!m_pointerVisible)
        {
            m_scene->addChild(m_handTransform.get());
            m_pointerVisible = true;
        }
    }
    else
    {
        if (m_pointerVisible)
        {
            m_scene->removeChild(m_handTransform.get());
            m_pointerVisible = false;
        }
    }

    coPointerButton *button = cover->getPointerButton();
    osg::Matrix handMat = cover->getPointerMat();

    if (!coVRConfig::instance()->isMenuModeOn())
    {
        if (button->wasPressed(vrui::vruiButtons::MENU_BUTTON))
            toggleMenu();
    }

    // transparency of handLine
    if (transparentPointer_ && m_handLine)
    {
        osg::Material *mat = dynamic_cast<osg::Material *>(m_handLine->getOrCreateStateSet()->getAttribute(osg::StateAttribute::MATERIAL));
        if (mat)
        {
            osg::Vec3 forward(0.0f, 1.0f, 0.0f);
            osg::Vec3 vector = forward * handMat;
            float alpha = min(max(1.0f + (vector * forward) * 1.5f, 0.0f), 1.0f);
            mat->setAlpha(osg::Material::FRONT_AND_BACK, alpha);
        }
    }

    if (coVRConfig::instance()->useWiiNavigationVisenso())
    {
        // for wii interaction
        // restrict pointer for wii

        if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor())
        {
            osg::Matrix o_to_w = cover->getBaseMat();
            osg::Vec3 currentInterPos_w = coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->getMatrix().getTrans() * o_to_w;
            handMat.setTrans(currentInterPos_w);
            m_handTransform->setMatrix(handMat);
        }
        else
        {
            m_handTransform->setMatrix(handMat);

            double bottom, top, left, right;
            bottom = -coVRConfig::instance()->screens[0].vsize / 2;
            top = coVRConfig::instance()->screens[0].vsize / 2;
            left = -coVRConfig::instance()->screens[0].hsize / 2;
            right = coVRConfig::instance()->screens[0].hsize / 2;
            Vec3 trans = handMat.getTrans();
            if (trans.x() < left)
                trans[0] = left;
            else if (trans.x() > right)
                trans[0] = right;
            if (trans.z() < bottom)
                trans[2] = bottom;
            else if (trans.z() > top)
                trans[2] = top;

            trans[1] = wiiPos;

            handMat.setTrans(trans);
#ifdef OLDINPUT
            VRTracker::instance()->setHandMat(handMat);
#endif
            m_handTransform->setMatrix(handMat);
        }
    }
    else // use tracking
    {
        //fprintf(stderr,"VRScenegraph::update !wiiMode\n");

        if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor())
        {
            //fprintf(stderr,"coVRNavigationManager::traverseInteractors && getCurrent\n");
            osg::Matrix o_to_w = cover->getBaseMat();
            osg::Vec3 currentInterPos_w = coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->getMatrix().getTrans() * o_to_w;

            //osg::Vec3 diff = currentInterPos_w - handMat.getTrans();
            //handMat.setTrans(handMat.getTrans()+diff);
            handMat.setTrans(currentInterPos_w);
            //if (coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->isRunning())
            //   fprintf(stderr,"diff = [%f %f %f]\n", diff[0], diff[1], diff[2]);
        }
        m_handTransform->setMatrix(handMat);
    }

    // static osg::Vec3Array *coord = new osg::Vec3Array(4*6);
    osg::Matrix dcs_mat, rot_mat, tmpMat;
    //int collided[6] = {0, 0, 0, 0, 0, 0}; /* front,back,right,left,up,down */

#ifdef OLDINPUT
    if (VRTracker::instance()->getCameraSensorStation() != -1)
    {
        osg::Matrix viewerTrans;
        osg::Vec3 viewerPos = -VRViewer::instance()->getInitialViewerPos();
        viewerTrans.makeTranslate(viewerPos);
        osg::Matrix newWorld = viewerTrans * VRTracker::instance()->getCameraMat();
        newWorld.invert(newWorld);
        m_objectsTransform->setMatrix(newWorld);
    }
#endif

    // update the viewer axis dcs with tracker not viewer mat
    // otherwise it is not updated when freeze is on
    if (Input::instance()->isHeadValid())
        m_viewerAxisTransform->setMatrix(Input::instance()->getHeadMat());

    if (showSmallSceneAxis_)
    {
        Matrix sm;
        sm = m_objectsTransform->getMatrix();
        Vec3 t;
        t = m_smallSceneAxisTransform->getMatrix().getTrans();
        sm(3, 0) = t[0];
        sm(3, 1) = t[1];
        sm(3, 2) = t[2];
        m_smallSceneAxisTransform->setMatrix(sm);
    }
#ifdef PHANTOM_TRACKER
    if (feedbackList)
    {
        float pos[3];
        static int first = 1;
        int i, n, curr, num;
        if (first)
        {
            buttonSpecCell spec;
            strcpy(spec.name, "ForceFeedback");
            spec.actionType = BUTTON_SWITCH;
            spec.callback = &manipulateCallback;
            if (forceFeedbackON)
                spec.state = 1.0;
            else
                spec.state = 0.0;
            spec.calledClass = (void *)this;
            spec.dashed = false;
            spec.group = -1;
            VRPinboard::mainPinboard->addButtonToMainMenu(&spec);
            strcpy(spec.name, "FeedbackScale");
            spec.actionType = BUTTON_SLIDER;
            spec.callback = &manipulateCallback;
            spec.state = forceScale;
            spec.calledClass = (void *)this;
            spec.dashed = false;
            spec.group = -1;
            spec.sliderMin = 0.0;
            spec.sliderMax = 100.0;
            VRPinboard::mainPinboard->addButtonToMainMenu(&spec);
            first = 0;
        }
        if (forceFeedbackON)
        {
            osg::Matrix xf, sc;
            osg::Node *thisNode;
            xf = objectsTransform->getMatrix();
            sc = m_scaleTransform->getMatrix();
            getHandWorldPosition(pos, NULL, NULL);
            // work through all children
            n = objectsRoot->getNumChildren();

            curr = 0;
            for (i = 0; i < n; i++)
            {

                thisNode = objectsRoot->getChild(i);
                if (pfIsOfType(thisNode, osg::MatrixTransform::getClassType()))
                {
                    thisNode = ((osg::MatrixTransform *)thisNode)->getChild(0);
                    if (pfIsOfType(thisNode, pfSequence::getClassType()))
                    {
                        curr = ((pfSequence *)thisNode)->getFrame(&num);
                        break;
                    }
                }
            }
            feedbackList->updateForce(xf, sc, pos[0], pos[1], pos[2], curr, forceFeedbackMode, forceFeedbackON, forceScale);
        }
    }
#endif

    // check if translation is restricted
    if ((m_transRestrictMinX != m_transRestrictMaxX) || (m_transRestrictMinY != m_transRestrictMaxY) || (m_transRestrictMinZ != m_transRestrictMaxZ))
    {
        Vec3 trans = m_objectsTransform->getMatrix().getTrans();
        if (m_transRestrictMinX != m_transRestrictMaxX)
        {
            if (trans.x() < m_transRestrictMinX)
                trans[0] = m_transRestrictMinX;
            else if (trans.x() > m_transRestrictMaxX)
                trans[0] = m_transRestrictMaxX;
        }
        if (m_transRestrictMinY != m_transRestrictMaxY)
        {
            if (trans.y() < m_transRestrictMinY)
                trans[1] = m_transRestrictMinY;
            else if (trans.y() > m_transRestrictMaxY)
                trans[1] = m_transRestrictMaxY;
        }
        if (m_transRestrictMinZ != m_transRestrictMaxZ)
        {
            if (trans.z() < m_transRestrictMinZ)
                trans[2] = m_transRestrictMinZ;
            else if (trans.z() > m_transRestrictMaxZ)
                trans[2] = m_transRestrictMaxZ;
        }
        Matrix m = m_objectsTransform->getMatrix();
        m.setTrans(trans);
        m_objectsTransform->setMatrix(m);
    }

    if (m_enableHighQualityOption && !m_highQuality)
    {
        // HQ mode ON, if button is pressed while the mouse is higher then the head
        Vec3 pointerPosWld = cover->getPointerMat().getTrans();
        Vec3 viewerPosWld = cover->getViewerMat().getTrans();
        if (pointerPosWld[2] > viewerPosWld[2] && cover->getPointerButton()->wasPressed())
            m_switchToHighQuality = true;
    }

    if (m_switchToHighQuality)
    {
        assert(m_enableHighQualityOption == true);
        m_switchToHighQuality = false;
        if (!m_interactionHQ->isRegistered())
        {
            vrui::coInteractionManager::the()->registerInteraction(m_interactionHQ);
        }
        m_highQuality = true;
    }
    else if (m_highQuality && (cover->getPointerButton()->wasPressed() || (cover->getMouseButton() && cover->getMouseButton()->wasPressed())))
    {
        m_highQuality = false;
        vrui::coInteractionManager::the()->unregisterInteraction(m_interactionHQ);
    }
}

void
VRSceneGraph::setWireframe(WireframeMode wf)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::setWireframe\n");

    m_wireframe = wf;
    m_drawStyle->select(m_wireframe);

    m_lineHider->removeChild(m_objectsTransform);
    m_scene->removeChild(m_objectsTransform);
    m_scene->removeChild(m_lineHider);

    switch(m_wireframe)
    {
        case Disabled:
        case Enabled:
        case Points:
        {
            osg::PolygonMode *polymode = new osg::PolygonMode;
            if (m_wireframe == Disabled)
            {
                m_objectsStateSet->setAttributeAndModes(polymode, osg::StateAttribute::ON);
            }
            else if (m_wireframe == Points)
            {
                polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::POINT);
                m_objectsStateSet->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }
            else
            {
                polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
                m_objectsStateSet->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }
            m_scene->addChild(m_objectsTransform);
            break;
        }
        case HiddenLineBlack:
        case HiddenLineWhite:
        {
            if (m_wireframe == HiddenLineBlack) {
                m_lineHider->setWireframeColor(osg::Vec4(0, 0, 0, 1));
            } else {
                m_lineHider->setWireframeColor(osg::Vec4(1, 1, 1, 1));
            }

            osg::PolygonMode *polymode = new osg::PolygonMode;
            m_objectsStateSet->setAttributeAndModes(polymode, osg::StateAttribute::ON);

            m_scene->addChild(m_lineHider);
            m_lineHider->addChild(m_objectsTransform);
            m_scene->removeChild(m_objectsTransform);
            break;
        }
    }
}

void
VRSceneGraph::toggleHighQuality(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::toggleHighQuality %d\n", state);

    m_enableHighQualityOption = state;
    m_allowHighQuality->setState(state);
}

bool
VRSceneGraph::highQuality() const
{
    return m_highQuality;
}

void
VRSceneGraph::toggleAxis(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::toggleAxis %d\n", state);

    m_coordAxis = state;
    m_showAxis->setState(m_coordAxis);
    if (m_coordAxis)
    {
        if (m_worldAxis->getNumParents() == 0)
            m_scene->addChild(m_worldAxis.get());
        if (m_handAxis->getNumParents() == 0)
            m_handAxisScaleTransform->addChild(m_handAxis.get());
        if (m_objectAxis->getNumParents() == 0)
            m_scaleTransform->addChild(m_objectAxis.get());
        if (coVRConfig::instance()->frozen()
            && Input::instance()->hasHead())
        {
            //show viewer axis only if headtracking is disabled
            if (m_viewerAxis->getNumParents() == 0)
                m_viewerAxisTransform->addChild(m_viewerAxis.get());
        }
    }
    else
    {
        while (m_worldAxis->getNumParents() > 0)
            m_worldAxis->getParent(0)->removeChild(m_worldAxis.get());
        while (m_viewerAxis->getNumParents() > 0)
            m_viewerAxis->getParent(0)->removeChild(m_viewerAxis.get());
        while (m_handAxis->getNumParents() > 0)
            m_handAxis->getParent(0)->removeChild(m_handAxis.get());
        while (m_objectAxis->getNumParents())
            m_objectAxis->getParent(0)->removeChild(m_objectAxis.get());
    }
}

void
VRSceneGraph::setPointerType(int pointerType)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::setHandType\n");

    m_pointerType = pointerType;
    while (m_handIconScaleTransform->getNumChildren())
        m_handIconScaleTransform->removeChild(m_handIconScaleTransform->getChild(0));

    m_pointerDepthTransform->addChild(m_handIconScaleTransform);

    switch (m_pointerType)
    {
    case HAND_LINE:
        break;
    case HAND_PLANE:
        m_handIconScaleTransform->addChild(m_handPlane.get());
        //m_handIconScaleTransform->addChild(m_handPlane.get());
        break;
    case HAND_PROBE:
        m_handIconScaleTransform->addChild(m_handPlane.get());
        m_handIconScaleTransform->addChild(m_handNormal.get());
        m_handIconScaleTransform->addChild(m_handProbe.get());
        break;
    case HAND_SPHERE:
        m_handIconScaleTransform->addChild(m_handSphere.get());
        break;
    case HAND_CUBE:
        m_handIconScaleTransform->addChild(m_handCube.get());
        break;
    case HAND_PYRAMID:
        m_handIconScaleTransform->addChild(m_handPyramid.get());
        break;
    case HAND_DRIVE:
        m_handIconScaleTransform->addChild(m_handDrive.get());
        break;
    case HAND_WALK:
        m_handIconScaleTransform->addChild(m_handWalk.get());
        break;
    case HAND_FLY_LINE:
        m_handIconScaleTransform->addChild(m_handFly.get());
        break;
    }
}

class SetBoundingSphereCallback : public osg::Node::ComputeBoundingSphereCallback
{
    virtual osg::BoundingSphere computeBound(const osg::Node &node) const
    {
        osg::BoundingSphere bs;
        if (VRSceneGraph::instance()->isScalingAllObjects())
            bs = node.getInitialBound();
        else
            bs = node.computeBound();
        return bs;
    }
};

void VRSceneGraph::setNodeBounds(osg::Node *node, const osg::BoundingSphere *bs)
{
    if (bs)
    {
        node->setInitialBound(*bs);
        m_specialBoundsNodeList.insert(node);
        node->setComputeBoundingSphereCallback(new SetBoundingSphereCallback);
    }
    else
    {
        m_specialBoundsNodeList.erase(node);
    }
}

void
VRSceneGraph::adjustScale()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::adjustScale mode=%f\n", m_scaleMode);

    // SCALE attribute not set
    // or SCALE attribute set to a number < 0.0

    if (m_scaleMode < 0.0)
    {
        // the scale to view all, but only if explicitly enables in covise.config
        if (m_scaleAllOn && coVRNavigationManager::instance()->getMode() != coVRNavigationManager::Scale)
            scaleAllObjects();
    }

    // SCALE keep
    else if (m_scaleMode == 0.0)
    { // keep scale so do nothing
    }

    // SCALE <number>
    else
    {
        osg::Matrix mat;
        m_scaleFactor = m_scaleMode;
        mat.makeScale(m_scaleFactor, m_scaleFactor, m_scaleFactor);
        m_scaleTransform->setMatrix(mat);
    }
}

osg::MatrixTransform *
VRSceneGraph::loadAxisGeode(float s)
{

    if (cover->debugLevel(4))
        fprintf(stderr, "VRSceneGraph::loadAxisGeode\n");

    osg::MatrixTransform *mt = new osg::MatrixTransform;
	mt->setName("AxisGeodeMatrixTransform");
    mt->addChild(coVRFileManager::instance()->loadIcon("Axis"));
    mt->setMatrix(osg::Matrix::scale(s, s, s));

    return (mt);
}

osg::Node *
VRSceneGraph::loadHandIcon(const char *name)
{
    osg::Node *n;
    n = coVRFileManager::instance()->loadIcon(name);
    if (n == NULL)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "failed to load hand icon: %s\n", name);
        n = coVRFileManager::instance()->loadIcon("hlrsIcon");
    }
    return (n);
}

osg::Node *
VRSceneGraph::loadHandLine()
{
    if (!coCoviseConfig::isOn("visible", "COVER.PointerAppearance.IconName", true))
        return nullptr;

    osg::Node *result = nullptr;

    osg::Node *n = nullptr;
    string iconName = coCoviseConfig::getEntry("COVER.PointerAppearance.IconName");
    if (!iconName.empty())
    {
        n = coVRFileManager::instance()->loadIcon(iconName.c_str());
        if (n)
        {
            n->dirtyBound();
            osg::ComputeBoundsVisitor cbv;
            osg::BoundingBox &bb(cbv.getBoundingBox());
            n->accept(cbv);
            float sx = bb._max.x() - bb._min.x();
            float sy = bb._max.y() - bb._min.y();
            // move icon in front of pointer (laser sword)
            float width = coCoviseConfig::getFloat("COVER.PointerAppearance.Width", sx);
            float length = coCoviseConfig::getFloat("COVER.PointerAppearance.Length", sy);

            osg::MatrixTransform *m = new osg::MatrixTransform;
			m->setName("HandLineMatrixTransform");
            m->setMatrix(osg::Matrix::scale(width / sx, length / sy, width / sx));
            m->addChild(n);
            result = m;
        }
        else
        {
            result = loadHandIcon("HandLine");
        }
    }
    else
    {
        result = loadHandIcon("HandLine");
    }

    transparentPointer_ = (coCoviseConfig::isOn("COVER.PointerAppearance.Transparent", false));
    if (transparentPointer_)
    {
        osg::StateSet *sset = result->getOrCreateStateSet();
        sset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        sset->setMode(GL_BLEND, osg::StateAttribute::ON);
        osg::Material *mat = new osg::Material();
        sset->setAttributeAndModes(mat, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        //mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    }

    return result;
}

osg::Vec3
VRSceneGraph::getWorldPointOfInterest() const
{
    osg::Vec3 pointOfInterest(0, 0, 0);
    pointOfInterest = m_pointerDepthTransform.get()->getMatrix().preMult(pointOfInterest);
    pointOfInterest = m_handIconScaleTransform->getMatrix().preMult(pointOfInterest);
    pointOfInterest = m_handTransform.get()->getMatrix().preMult(pointOfInterest);

    return pointOfInterest;
}

osg::BoundingSphere
VRSceneGraph::getBoundingSphere()
{
    m_scalingAllObjects = true;
    dirtySpecialBounds();

    osg::Group *scaleNode;
    if (coCoviseConfig::isOn("COVER.ScaleWithInteractors", false))
        scaleNode = m_scaleTransform;
    else
        scaleNode = m_objectsRoot;

    scaleNode->dirtyBound();
    //osg::BoundingSphere bsphere = coVRSelectionManager::instance()->getBoundingSphere(m_objectsRoot);
    osg::BoundingSphere bsphere;

    // determine center of bounding sphere
    BoundingBox bb;
    bb.init();
    osg::Node *currentNode = NULL;
    for (unsigned int i = 0; i < scaleNode->getNumChildren(); i++)
    {
        currentNode = scaleNode->getChild(i);
        const osg::Transform *transform = currentNode->asTransform();
        if ((!transform || transform->getReferenceFrame() == osg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
        {
            // Using the Visitor from scaleNode doesn't work if it's m_scaleTransform (ScaleWithInteractors==true) -> therefore the loop)
            NodeSet::iterator it = m_specialBoundsNodeList.find(currentNode);
            if (it == m_specialBoundsNodeList.end())
            {
                osg::ComputeBoundsVisitor cbv;
                cbv.setTraversalMask(Isect::Visible);
                currentNode->accept(cbv);
                bb.expandBy(cbv.getBoundingBox());
            }
            else
            {
                bb.expandBy(currentNode->getBound());
            }
        }
    }

    // determine radius of bounding sphere
    if (bb.valid())
    {
        bsphere._center = bb.center();
        bsphere._radius = 0.0f;
        for (unsigned int i = 0; i < scaleNode->getNumChildren(); i++)
        {
            currentNode = scaleNode->getChild(i);
            const osg::Transform *transform = currentNode->asTransform();
            if ((!transform || transform->getReferenceFrame() == osg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
            {
                NodeSet::iterator it = m_specialBoundsNodeList.find(currentNode);
                if (it == m_specialBoundsNodeList.end())
                {
                    osg::ComputeBoundsVisitor cbv;
                    cbv.setTraversalMask(Isect::Visible);
                    currentNode->accept(cbv);
                    bsphere.expandRadiusBy(cbv.getBoundingBox());
                }
                else
                {
                    bsphere.expandRadiusBy(currentNode->getBound());
                }
            }
        }
    }

    m_scalingAllObjects = false;

    dirtySpecialBounds();
    scaleNode->dirtyBound();

    coVRPluginList::instance()->expandBoundingSphere(bsphere);

    if (coVRMSController::instance()->isCluster())
    {
        struct BoundsData
        {
            BoundsData(const osg::BoundingSphere &bs)
            {
                x = bs.center()[0];
                y = bs.center()[1];
                z = bs.center()[2];
                r = bs.radius();
            }

            float x, y, z;
            float r;
        };
        if (coVRMSController::instance()->isSlave())
        {
            BoundsData bd(bsphere);
            coVRMSController::instance()->sendMaster(&bd, sizeof(bd));

            coVRMSController::instance()->readMaster(&bd, sizeof(bd));
            bsphere.center() = osg::Vec3(bd.x, bd.y, bd.z);
            bsphere.radius() = bd.r;
        }
        else
        {
            coVRMSController::SlaveData boundsData(sizeof(BoundsData));
            coVRMSController::instance()->readSlaves(&boundsData);
            for (int i = 0; i < coVRMSController::instance()->getNumSlaves(); ++i)
            {
                const BoundsData *bd = static_cast<const BoundsData *>(boundsData.data[i]);
                bsphere.expandBy(osg::BoundingSphere(osg::Vec3(bd->x, bd->y, bd->z), bd->r));
            }
            BoundsData bd(bsphere);
            coVRMSController::instance()->sendSlaves(&bd, sizeof(bd));
            bsphere.center() = osg::Vec3(bd.x, bd.y, bd.z);
            bsphere.radius() = bd.r;
        }
    }

    return bsphere;
}

void VRSceneGraph::scaleAllObjects(bool resetView)
{
    osg::BoundingSphere bsphere = getBoundingSphere();
    if (bsphere.radius() <= 0.f)
        bsphere.radius() = 1.f;

    // scale and translate but keep current orientation
    float scaleFactor = 1.f, masterScale = 1.f;
    osg::Matrix matrix, masterMatrix;
    boundingSphereToMatrices(bsphere, resetView, &matrix, &scaleFactor);

    int len=0;
    if (coVRMSController::instance()->isMaster())
    {
	covise::TokenBuffer tb;
        tb << scaleFactor;
        tb << matrix;
        len = tb.getData().length();
        coVRMSController::instance()->sendSlaves(&len, sizeof(len));
        coVRMSController::instance()->sendSlaves(tb.getData().data(), tb.getData().length());
    }
    else
    {
        coVRMSController::instance()->readMaster(&len, sizeof(len));
        covise::DataHandle buf(len);
        coVRMSController::instance()->readMaster(&buf.accessData()[0], len);
        covise::TokenBuffer tb(buf);
        tb >> masterScale;
        if (masterScale != scaleFactor)
        {
            fprintf(stderr, "VRSceneGraph::scaleAllObjects: scaleFactor mismatch on %d, is %f, should be %f\n",
                    coVRMSController::instance()->getID(), scaleFactor, masterScale);
        }
        scaleFactor = masterScale;
        tb >> masterMatrix;
        if (masterMatrix != matrix)
        {
            std::cerr << "VRSceneGraph::scaleAllObjects: matrix mismatch on " << coVRMSController::instance()->getID()
                << ", is " << matrix << ", should be " << masterMatrix << endl;
        }
        matrix = masterMatrix;
    }

    cover->setScale(scaleFactor);
    //fprintf(stderr, "bbox [x:%f %f\n      y:%f %f\n      z:%f %f]\n", bb.xMin(), bb.xMax(), bb.yMin(), bb.yMax(), bb.zMin(), bb.zMax());
    //fprintf(stderr, "VRSceneGraph::scaleAllObjects scalefactor: %f\n", scaleFactor);
    m_objectsTransform->setMatrix(matrix);
}

void VRSceneGraph::dirtySpecialBounds()
{
    for (NodeSet::iterator it = m_specialBoundsNodeList.begin();
         it != m_specialBoundsNodeList.end();
         ++it)
    {
        (*it)->dirtyBound();
    }
}

void VRSceneGraph::boundingSphereToMatrices(const osg::BoundingSphere &boundingSphere,
                                            bool resetView, osg::Matrix *currentMatrix, float *scaleFactor) const
{
    *scaleFactor = std::abs(cover->getSceneSize() / 2.f / boundingSphere.radius());
    currentMatrix->makeTranslate(-(boundingSphere.center() * *scaleFactor));
    if (!resetView)
    {
        osg::Quat rotation = m_objectsTransform->getMatrix().getRotate();
        osg::Matrix rotMat;
        rotMat.makeRotate(rotation);
        currentMatrix->postMult(rotMat);
    }
}

#if 0
void
VRSceneGraph::manipulate(buttonSpecCell *spec)
{
#ifdef PHANTOM_TRACKER
    if (strcmp(spec->name, "ForceFeedback") == 0)
    {
        if (spec->state)
        {
            forceFeedbackON = 1;
        }
        else
        {
            forceFeedbackON = 0;
        }
    }
    else if (strcmp(spec->name, "ForceScale") == 0)
    {
        forceScale = spec->state;
    }
    else
#endif
        if (strcmp(spec->name, "FloatSlider") == 0 || strcmp(spec->name, "IntSlider") == 0)
    {
        if (spec->state)
        {
            setPointerType(HAND_SPHERE);
        }
        else
        {
            setPointerType(HAND_LINE);
        }
    }
    else if (strcmp(spec->name, "FloatVector") == 0 || strcmp(spec->name, "IntVector") == 0)
    {
        if (spec->state)
        {
            m_vectorInteractor = 1;
            setPointerType(HAND_SPHERE);
        }
        else
        {
            m_vectorInteractor = 0;
            setPointerType(HAND_LINE);
        }
    }
}
#endif

#ifdef PHANTOM_TRACKER
void
VRSceneGraph::manipulateCallback(void *sceneGraph, buttonSpecCell *spec)
{
    ((VRSceneGraph *)sceneGraph)->manipulate(spec);
}
#endif

void
VRSceneGraph::viewAll(bool resetView)
{
    coVRNavigationManager::instance()->enableViewerPosRotation(false);

    scaleAllObjects(resetView);

    coVRCollaboration::instance()->SyncXform();
}

bool
VRSceneGraph::isHighQuality() const
{
    return m_highQuality;
}

bool
VRSceneGraph::saveScenegraph(bool storeWithMenu)
{
    std::string filename = coCoviseConfig::getEntry("value", "COVER.SaveFile", "/var/tmp/OpenCOVER.osgb");
    return saveScenegraph(filename, storeWithMenu);
}

bool
VRSceneGraph::saveScenegraph(const std::string &filename, bool storeWithMenu)
{
    if (isScenegraphProtected_)
    {
        fprintf(stderr, "Cannot store scenegraph. Not allowed!");
        return false;
    }

    if (cover->debugLevel(3))
        fprintf(stderr, "save file as: %s\n", filename.c_str());
    size_t len = filename.length();
    if ((len > 4 && (!strcmp(filename.c_str() + len - 4, ".ive") || !strcmp(filename.c_str() + len - 4, ".osg")))
        || (len > 5 && (!strcmp(filename.c_str() + len - 5, ".osgt")
                        || !strcmp(filename.c_str() + len - 5, ".osgb")
                        || !strcmp(filename.c_str() + len - 5, ".osgx"))))
    {
    }
    else
    {
        if (cover->debugLevel(1))
            std::cerr << "Writing to \"" << filename << "\": unknown extension, use .ive, .osg, .osgt, .osgb, or .osgx." << std::endl;
    }

    if (osgDB::writeNodeFile(storeWithMenu ? *static_cast<osg::Group *>(m_scene) : *m_objectsRoot, filename.c_str()))
    {
        if (cover->debugLevel(3))
            std::cerr << "Data written to \"" << filename << "\"." << std::endl;
        return true;
    }

    if (cover->debugLevel(1))
        std::cerr << "Writing to \"" << filename << "\" failed." << std::endl;
    return false;
}

void
VRSceneGraph::setScaleFromButton(float direction)
{
    cover->setScale(cover->getScale() * (1.0 + direction * m_scaleFactorButton));
    setScaleFactor(cover->getScale());
}

void VRSceneGraph::setScaleFactorButton(float f)
{
    m_scaleFactorButton = f;
}

void VRSceneGraph::setScaleRestrictFactor(float min, float max)
{
    m_scaleRestrictFactorMin = min;
    m_scaleRestrictFactorMax = max;
}

void VRSceneGraph::setRestrictBox(float minX, float maxX, float minY, float maxY, float minZ, float maxZ)
{
    m_transRestrictMinX = minX;
    m_transRestrictMinY = minY;
    m_transRestrictMinZ = minZ;
    m_transRestrictMaxX = maxX;
    m_transRestrictMaxY = maxY;
    m_transRestrictMaxZ = maxZ;
}

void
VRSceneGraph::getHandWorldPosition(float *pos, float *direction, float *direction2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::getHandWorldPosition\n");

    osg::Matrix mat = cover->getPointerMat();
    osg::Vec3 position = mat.getTrans();
    osg::Vec3 normal;
    normal.set(mat(1, 0), mat(1, 1), mat(1, 2));
    osg::Vec3 normal2;
    normal2.set(mat(0, 0), mat(0, 1), mat(0, 2));
    mat = cover->getInvBaseMat();
    position = mat.preMult(position);
    normal = mat.transform3x3(normal, mat);
    normal2 = mat.transform3x3(normal2, mat);
    // was normal2.xformVec(normal2,mat);
    if (pos)
    {
        pos[0] = position[0];
        pos[1] = position[1];
        pos[2] = position[2];
    }
    if (direction)
    {
        normal.normalize();
        direction[0] = normal[0];
        direction[1] = normal[1];
        direction[2] = normal[2];
    }
    if (direction2)
    {
        normal2.normalize();
        direction2[0] = normal2[0];
        direction2[1] = normal2[1];
        direction2[2] = normal2[2];
    }
    return;
}

osg::StateSet *
VRSceneGraph::loadGlobalGeostate()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRSceneGraph::loadGlobalGeostate\n");

    osg::LightModel *defaultLm;

    defaultLm = new osg::LightModel();

    defaultLm->setTwoSided(coCoviseConfig::isOn("COVER.TwoSide", true));

    defaultLm->setLocalViewer(true);
    //defaultLm->setTwoSided(true);
    if (coCoviseConfig::isOn("COVER.SeparateSpecular", true))
        defaultLm->setColorControl(osg::LightModel::SEPARATE_SPECULAR_COLOR);
    else
        defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);

    osg::Vec4 globalAmbient(0,0,0,1);
    defaultLm->setAmbientIntensity(globalAmbient);

    osg::Material *material = new osg::Material();

    material->setColorMode(osg::Material::OFF);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    // make sure that the global color mask exists.
    osg::ColorMask *rootColorMask = new osg::ColorMask;
    rootColorMask->setMask(true, true, true, true);

    // set up depth to be inherited by the rest of the scene unless
    // overrideen. this is overridden in bin 3.
    osg::Depth *rootDepth = new osg::Depth;
    rootDepth->setFunction(osg::Depth::LESS);
    rootDepth->setRange(0.0, 1.0);

    osg::StateSet *stateSet = new osg::StateSet();
    stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    if (coVRConfig::instance()->numChannels()==0
            || coVRConfig::instance()->channels[0].stereoMode != osg::DisplaySettings::ANAGLYPHIC)
    {
        stateSet->setAttribute(rootColorMask);
    }
    stateSet->setAttribute(rootDepth);

	osg::PointSprite *sprite = new osg::PointSprite();
	stateSet->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::OFF);

    osg::TexEnv* texEnv = new osg::TexEnv;
    texEnv->setMode(osg::TexEnv::MODULATE);
    stateSet->setTextureAttributeAndModes(0, texEnv, osg::StateAttribute::ON);

    osg::ref_ptr<osg::TexGen> texGen = new osg::TexGen();
    stateSet->setTextureAttributeAndModes(0, texGen.get(), osg::StateAttribute::OFF);

    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    stateSet->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.0);
    stateSet->setAttributeAndModes(alphaFunc, StateAttribute::OFF);
    m_Multisample = new Multisample;
    if (coVRConfig::instance()->doMultisample())
    {
        stateSet->setAttributeAndModes(m_Multisample.get(), StateAttribute::ON);
        m_Multisample->setHint(coVRConfig::instance()->getMultisampleMode());
        m_Multisample->setSampleCoverage(coVRConfig::instance()->getMultisampleCoverage(), coVRConfig::instance()->getMultisampleInvert());
    }

    return (stateSet);
}

void VRSceneGraph::setMultisampling(osg::Multisample::Mode mode)
{
    m_Multisample->setHint(mode);
}

static const float Bright = 0.8f;

osg::StateSet *
VRSceneGraph::loadDefaultGeostate(osg::Material::ColorMode mode)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRSceneGraph::loadDefaultGeostate\n");

    osg::Material *material = new osg::Material();

    material->setColorMode(mode);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(Bright, Bright, Bright, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::GREATER, 0.0);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);

    osg::LightModel *defaultLm = new osg::LightModel();
    defaultLm->setLocalViewer(true);
    defaultLm->setTwoSided(true);
    //defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);

    osg::StateSet *stateSet = new osg::StateSet();
    stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
    return stateSet;
}

osg::StateSet *
VRSceneGraph::loadTransparentGeostate(osg::Material::ColorMode mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::loadDefaultGeostate\n");

    osg::Material *material = new osg::Material();
    material->setColorMode(mode);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 0.5f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(Bright, Bright, Bright, 0.5f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 0.5f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 0.5f);

    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::GREATER, 0.0);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);

    osg::LightModel *defaultLm = new osg::LightModel();
    defaultLm->setLocalViewer(true);
    defaultLm->setTwoSided(true);
    defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);

    osg::StateSet *stateTransp = new osg::StateSet();
    stateTransp->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateTransp->setNestRenderBins(false);
    stateTransp->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateTransp->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    stateTransp->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);
    stateTransp->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    stateTransp->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);

    return stateTransp;
}

osg::StateSet *
VRSceneGraph::loadUnlightedGeostate(osg::Material::ColorMode mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::loadUnlightedGeostate\n");

    osg::Material *material = new osg::Material();

    material->setColorMode(mode);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    osg::StateSet *stateSet = new osg::StateSet();
    stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    return stateSet;
}

void VRSceneGraph::addPointerIcon(osg::Node *node)
{
    // recompute scale factor
    node->dirtyBound();
    osg::ComputeBoundsVisitor cbv;
    cbv.setTraversalMask(Isect::Visible);
    osg::BoundingBox &bb(cbv.getBoundingBox());
    node->accept(cbv);
    float sx, sy, sz, s;
    sx = bb._max.x() - bb._min.x();
    sy = bb._max.y() - bb._min.y();
    sz = bb._max.z() - bb._min.z();
    s = max(sx, sy);
    s = max(s, sz);
    float scale = m_handIconSize / s;
    if (node->getName() == "sphere")
    {
        scale = 1.;
    }
    osg::Matrix m = osg::Matrix::scale(scale, scale, scale);
    m.setTrans(0, m_handIconOffset, 0);
    m_handIconScaleTransform->setMatrix(m);

    // add icon
    m_handIconScaleTransform->addChild(node);
}

void VRSceneGraph::removePointerIcon(osg::Node *node)
{
    // remove icon
    m_handIconScaleTransform->removeChild(node);
}

//******************************************
//           Coloring
//******************************************
/*
 * sets the color of the node nodeName
 */
void VRSceneGraph::setColor(const char *nodeName, int *color, float transparency)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::setColor\n");
    Geode *node;
    //get the node
    node = findFirstNode<Geode>(nodeName);
    if (node != NULL)
    {
        //set the color
        setColor(node, color, transparency);
    }
}

void VRSceneGraph::setColor(osg::Geode *geode, int *color, float transparency)
{
    ref_ptr<Drawable> drawable;
    if (geode != NULL)
    {
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            drawable = geode->getDrawable(i);
            drawable->ref();
            //bool mtlOn = false;
            /*if (geode->getStateSet())
            mtlOn = (geode->getStateSet()->getMode(StateAttribute::MATERIAL) == StateAttribute::ON) || (geode->getStateSet()->getMode(StateAttribute::MATERIAL) == StateAttribute::INHERIT);
         if (mtlOn)
         {*/

            storeMaterial(drawable);

            if (color[0] == -1)
            {
                StoredMaterialsMap::iterator matIt = storedMaterials.find(drawable);
                if (matIt != storedMaterials.end())
                {
                    drawable->getStateSet()->setAttributeAndModes(matIt->second, osg::StateAttribute::ON);
                }
            }
            else
            {
                // Create a new material since we dont want to change our default material.
                drawable->setStateSet(loadDefaultGeostate(osg::Material::OFF));
                Material *mtl = (Material *)drawable->getStateSet()->getAttribute(StateAttribute::MATERIAL);
                mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, transparency));
                mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, transparency));
            }

            drawable->asGeometry()->setColorBinding(Geometry::BIND_OFF);

            drawable->getStateSet()->setNestRenderBins(false);
            if (transparency < 1.0)
            {
                drawable->getStateSet()->setRenderingHint(StateSet::TRANSPARENT_BIN);

                drawable->getStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);

                // Disable writing to depth buffer.
                osg::Depth *depth = new osg::Depth;
                depth->setWriteMask(false);
                drawable->getStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
            }
            else
            {
                drawable->getStateSet()->setRenderingHint(StateSet::OPAQUE_BIN);

                drawable->getStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);

                // Enable writing to depth buffer.
                osg::Depth *depth = new osg::Depth;
                depth->setWriteMask(true);
                drawable->getStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
            }
            /*}
         else
         {
            fprintf(stderr, "material off\n");
            if (drawable->asGeometry())
            {
               drawable->asGeometry()->setColorBinding(Geometry::BIND_OVERALL);
               drawable->asGeometry()->setColorArray(colorArray.get());
            }
         }*/
            drawable->dirtyBound();
            drawable->unref();
        }
    }
}

// set the transparency of the node with nodename
void VRSceneGraph::setTransparency(const char *nodeName, float transparency)
{
    Geode *node;
    //get the node
    node = findFirstNode<Geode>(nodeName);
    if (node != NULL)
    {
        setTransparency(node, transparency);
    }
}

// set transparency of node geode
void VRSceneGraph::setTransparency(osg::Geode *geode, float transparency)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::setTransparency\n");
    Drawable *drawable;
    Material *mtl = NULL;
    if (geode != NULL)
    {
        //fprintf(stderr, "VRSceneGraph::setTransparency %s %f\n", geode->getName().c_str(), transparency);
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            drawable = geode->getDrawable(i);
            bool mtlOn = false;
            bool mtlInherit = false;
            if (drawable->getStateSet())
            {
                mtlOn = (drawable->getStateSet()->getMode(StateAttribute::MATERIAL) == StateAttribute::ON);
                mtlInherit = (drawable->getStateSet()->getMode(StateAttribute::MATERIAL) == StateAttribute::INHERIT);
            }
            if (!mtlInherit && mtlOn)
            {
                mtl = (Material *)drawable->getOrCreateStateSet()->getAttribute(StateAttribute::MATERIAL);
            }
            else
            {
                // if material is inherited get the material with the color
                StateSet *state = drawable->getStateSet();
                if (state)
                    mtl = (Material *)state->getAttribute(StateAttribute::MATERIAL);
                else
                    mtl = NULL;
                if (mtl == NULL && drawable->getNumParents() > 0)
                {
                    Node *node = drawable->getParent(0);
                    state = node->getStateSet();
                    if (state)
                        mtl = (Material *)state->getAttribute(StateAttribute::MATERIAL);
                    else
                        mtl = NULL;
                    while (mtl == NULL)
                    {
                        if (node->getNumParents() == 0)
                            break;
                        node = node->getParent(0);
                        state = node->getStateSet();
                        if (state)
                            mtl = (Material *)state->getAttribute(StateAttribute::MATERIAL);
                        else
                            mtl = NULL;
                    }
                }
            }

            if ((drawable->asGeometry() != NULL) && (drawable->asGeometry()->getColorBinding() == (osg::Geometry::BIND_PER_VERTEX)
#if 0
               || drawable->asGeometry()->getColorBinding() == (osg::Geometry::BIND_PER_PRIMITIVE)
               || drawable->asGeometry()->getColorBinding() == (osg::Geometry::BIND_PER_PRIMITIVE_SET)
#endif
                                                     || drawable->asGeometry()->getColorBinding() == (osg::Geometry::BIND_OVERALL)))
            {
                //fprintf(stderr, "VRSceneGraph::setTransparency %s color \n", geode->getName().c_str());
                //mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
                // copy color array and use new transparency
                Vec4Array *oldColors = dynamic_cast<osg::Vec4Array *>(drawable->asGeometry()->getColorArray());
                if (oldColors)
                {
                    ref_ptr<Vec4Array> newColors = new Vec4Array();
                    for (unsigned int i = 0; i < oldColors->size(); ++i)
                    {
                        osg::Vec4 *color = &oldColors->operator[](i);
                        newColors->push_back(Vec4(color->_v[0], color->_v[1], color->_v[2], transparency));
                    }
                    drawable->asGeometry()->setColorArray(newColors.get());
                }
            }
            else if (mtl == NULL)
            {
                mtl = (Material *)drawable->getOrCreateStateSet()->getAttribute(StateAttribute::MATERIAL);
            }

            if (mtl != NULL)
            {
                //fprintf(stderr, "VRSceneGraph::setTransparency %s material \n", geode->getName().c_str());
                AlphaFunc *alphaFunc = new AlphaFunc();
                alphaFunc->setFunction(AlphaFunc::ALWAYS, 1.0);
                drawable->getOrCreateStateSet()->setAttributeAndModes(alphaFunc, StateAttribute::OFF);
                BlendFunc *blendFunc = new BlendFunc();
                blendFunc->setFunction(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
                drawable->getStateSet()->setAttributeAndModes(blendFunc, StateAttribute::ON);

                Vec4 color;
                color = mtl->getAmbient(Material::FRONT_AND_BACK);
                color[3] = transparency;
                mtl->setAmbient(Material::FRONT_AND_BACK, color);

                color = mtl->getDiffuse(Material::FRONT_AND_BACK);
                color[3] = transparency;
                mtl->setDiffuse(Material::FRONT_AND_BACK, color);

                color = mtl->getSpecular(Material::FRONT_AND_BACK);
                color[3] = transparency;
                mtl->setSpecular(Material::FRONT_AND_BACK, color);

                color = mtl->getEmission(Material::FRONT_AND_BACK);
                color[3] = transparency;
                mtl->setEmission(Material::FRONT_AND_BACK, color);

                if (drawable->asGeometry()->getColorBinding() == osg::Geometry::BIND_PER_VERTEX)
                {
                    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
                    // copy color array and use new transparency
                    Vec4Array *oldColors = dynamic_cast<osg::Vec4Array *>(drawable->asGeometry()->getColorArray());
                    if (oldColors)
                    {
                        ref_ptr<Vec4Array> newColors = new Vec4Array();
                        for (unsigned int i = 0; i < oldColors->size(); ++i)
                        {
                            osg::Vec4 *color = &oldColors->operator[](i);
                            newColors->push_back(Vec4(color->_v[0], color->_v[1], color->_v[2], transparency));
                        }
                        drawable->asGeometry()->setColorArray(newColors.get());
                    }
                }

                drawable->getOrCreateStateSet()->setAttributeAndModes(mtl, StateAttribute::ON);
                drawable->dirtyBound(); // trigger display list generation
            }
            drawable->getOrCreateStateSet()->setNestRenderBins(false);
            if (transparency < 1.0)
            {
                drawable->getOrCreateStateSet()->setRenderingHint(StateSet::TRANSPARENT_BIN);

                drawable->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);

                // Disable writing to depth buffer.
                osg::Depth *depth = new osg::Depth;
                depth->setWriteMask(false);
                drawable->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
            }
            else
            {
                drawable->getOrCreateStateSet()->setRenderingHint(StateSet::OPAQUE_BIN);

                drawable->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);

                // Enable writing to depth buffer.
                osg::Depth *depth = new osg::Depth;
                depth->setWriteMask(true);
                drawable->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
            }
        }
    }
}

/*
 * sets a shader
 * not implemented yet
 */
void VRSceneGraph::setShader(const char *nodeName, const char *shaderName, const char *paraFloat, const char *paraVec2, const char *paraVec3, const char *paraVec4, const char *paraInt, const char *paraBool, const char *paraMat2, const char *paraMat3, const char *paraMat4)
{
    (void)nodeName;
    (void)shaderName;
    (void)paraFloat;
    (void)paraVec2;
    (void)paraVec3;
    (void)paraVec4;
    (void)paraInt;
    (void)paraBool;
    (void)paraMat2;
    (void)paraMat3;
    (void)paraMat4;
}

void VRSceneGraph::setShader(osg::Geode *geode, const char *shaderName, const char *paraFloat, const char *paraVec2, const char *paraVec3, const char *paraVec4, const char *paraInt, const char *paraBool, const char *paraMat2, const char *paraMat3, const char *paraMat4)
{
    (void)geode;
    (void)shaderName;
    (void)paraFloat;
    (void)paraVec2;
    (void)paraVec3;
    (void)paraVec4;
    (void)paraInt;
    (void)paraBool;
    (void)paraMat2;
    (void)paraMat3;
    (void)paraMat4;
}

// set opengl-material of node with nodename
void VRSceneGraph::setMaterial(const char *nodeName, int *ambient, int *diffuse, int *specular, float shininess, float transparency)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::setMaterial %f\n", transparency);
    Geode *node;
    //get the node
    node = findFirstNode<Geode>(nodeName);
    if (node != NULL)
    {
        setMaterial(node, ambient, diffuse, specular, shininess, transparency);
    }
}

// set opengl-material of node geode
void VRSceneGraph::setMaterial(osg::Geode *geode, const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRSceneGraph::setMaterial 2 %f\n", transparency);

    ref_ptr<Drawable> geoset;
    ref_ptr<StateSet> gstate;
    if (geode != NULL)
    {
        //set the material
        ref_ptr<Material> mtl = new Material();
        mtl->ref();
        mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(ambient[0] / 255.0, ambient[1] / 255.0, ambient[2] / 255.0, transparency));
        mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuse[0] / 255.0, diffuse[1] / 255.0, diffuse[2] / 255.0, transparency));
        mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(specular[0] / 255.0, specular[1] / 255.0, specular[2] / 255.0, transparency));
        mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, transparency));
        //mtl->setTransparency(Material::FRONT_AND_BACK ,transparency);
        mtl->setShininess(Material::FRONT_AND_BACK, shininess);
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            geoset = geode->getDrawable(i);
            geoset->ref();
            storeMaterial(geoset);
            geoset->asGeometry()->setColorBinding(Geometry::BIND_OFF);
            gstate = geoset->getOrCreateStateSet();
            gstate->ref();
            gstate->setAttributeAndModes(mtl.get(), StateAttribute::ON);
            gstate->setNestRenderBins(false);
            if (transparency < 1.0)
            {
                gstate->setRenderingHint(StateSet::TRANSPARENT_BIN);

                gstate->setMode(GL_BLEND, osg::StateAttribute::ON);

                // Disable writing to depth buffer.
                osg::Depth *depth = new osg::Depth;
                depth->setWriteMask(false);
                gstate->setAttributeAndModes(depth, osg::StateAttribute::ON);
            }
            else
            {
                gstate->setRenderingHint(StateSet::OPAQUE_BIN);

                gstate->setMode(GL_BLEND, osg::StateAttribute::OFF);

                // Enable writing to depth buffer.
                osg::Depth *depth = new osg::Depth;
                depth->setWriteMask(true);
                gstate->setAttributeAndModes(depth, osg::StateAttribute::ON);
            }

            geoset->setStateSet(gstate.get());
            geoset->unref();
            gstate->unref();
        }
        mtl->unref();
    }
}

void VRSceneGraph::storeMaterial(osg::Drawable *drawable)
{
    StoredMaterialsMap::iterator it = storedMaterials.find(drawable);
    if (it != storedMaterials.end())
    {
        return;
    }
    osg::StateSet *sset = drawable->getStateSet();
    if (!sset)
    {
        return;
    }
    osg::Material *mat = dynamic_cast<osg::Material *>(sset->getAttribute(osg::StateAttribute::MATERIAL, 0));
    if (!mat)
    {
        return;
    }
    storedMaterials.insert(std::pair<osg::Drawable *, osg::Material *>(drawable, mat));
}

void VRSceneGraph::protectScenegraph()
{
    isScenegraphProtected_ = true;
}
