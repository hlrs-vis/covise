/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/common.h>
#include <config/CoviseConfig.h>
#include <net/tokenbuffer.h>

#include "vvSceneGraph.h"
#include "vvVIVE.h"
#include "vvFileManager.h"
#include "vvNavigationManager.h"
#include "vvCollaboration.h"
#include "vvLighting.h"
#include "vvPluginList.h"
#include "vvRegisterSceneGraph.h"
#include "vvPluginSupport.h"
#include "vvMSController.h"
#include "vvViewer.h"
#include "vvSelectionManager.h"
#include "vvLabel.h"
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <input/VRKeys.h>
#include <input/input.h>
#include "vvConfig.h"
#include <OpenVRUI/coJoystickManager.h>
#include "vvIntersectionInteractorManager.h"
/*#include "vvShadowManager.h"*/
#include <vsg/all.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <OpenVRUI/vsg/mathUtils.h>

#include "ui/Menu.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/SelectionList.h"
#include "ui/View.h"

using namespace vsg;
using namespace vive;
using covise::coCoviseConfig;

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

vvSceneGraph *vvSceneGraph::s_instance = NULL;

vvSceneGraph *vvSceneGraph::instance()
{
    if (!s_instance)
        s_instance = new vvSceneGraph;
    return s_instance;
}

vvSceneGraph::vvSceneGraph()
: ui::Owner("vvSceneGraph", vv->ui)
, m_vectorInteractor(0)
, m_pointerDepth(0.f)
, m_floorHeight(-1250.0)
, m_wireframe(Disabled)
, m_textured(true)
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
, isScenegraphProtected_(false)
, m_enableHighQualityOption(true)
, m_switchToHighQuality(false)
, m_highQuality(false)
{
    assert(!s_instance);
}
/*bis hier neu*/

void vvSceneGraph::init()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\nnew vvSceneGraph\n");

    readConfigFile();

    initMatrices();

    initSceneGraph();

    // create coordinate axis and put them in
    // scenegraph acoording to config
    initAxis();

    // create hand device geometries
    initHandDeviceGeometry();

    vvLighting::instance();

    //emptyProgram_ = new vsg::Program();

    m_interactionHQ = new vrui::coCombinedButtonInteraction(vrui::coInteraction::AllButtons, "Anchor", vrui::coInteraction::Highest);
    m_interactionHQ->setNotifyOnly(true);

    m_trackHead = new ui::Button(vv->viewOptionsMenu, "TrackHead");
    m_trackHead->setText("Track head");
    m_trackHead->setShortcut("Shift+F");
    m_trackHead->setState(!vvConfig::instance()->frozen());
    m_trackHead->setCallback([this](bool state){
        toggleHeadTracking(state);
    });
    m_hidePointer = new ui::Button(vv->viewOptionsMenu, "HidePointer");
    m_hidePointer->setText("Hide pointer");
    m_hidePointer->setShortcut("Shift+P");
    m_hidePointer->setState(false);
    m_hidePointer->setCallback([this](bool state){
    if (!state)
    {
        if (!m_pointerVisible)
        {
            m_scene->addChild(m_handTransform);
            m_pointerVisible = true;
        }
    }
    else
    {
        if (m_pointerVisible)
        {
            //m_scene->removeChild(m_handTransform);
            m_pointerVisible = false;
        }
    }

    });

    m_allowHighQuality= new ui::Button(vv->viewOptionsMenu, "AllowHighQuality");
    vv->viewOptionsMenu->add(m_allowHighQuality);
    m_allowHighQuality->setState(m_enableHighQualityOption);
    m_allowHighQuality->setText("Allow high quality");
    m_allowHighQuality->setCallback([this](bool state){
        toggleHighQuality(state);
    });

    auto switchToHighQuality = new ui::Action(vv->viewOptionsMenu, "SwitchToHighQuality");
    switchToHighQuality->setVisible(false);
    vv->viewOptionsMenu->add(switchToHighQuality);
    switchToHighQuality->setText("Enable high quality rendering");
    switchToHighQuality->setShortcut("Shift+H");
    switchToHighQuality->setCallback([this](){
        if (m_enableHighQualityOption && !m_highQuality)
            m_switchToHighQuality = true;
    });

    m_drawStyle = new ui::SelectionList(vv->viewOptionsMenu, "DrawStyle");
    m_drawStyle->setText("Draw style");
    m_drawStyle->append("As is");
    m_drawStyle->append("Wireframe");
    m_drawStyle->append("Hidden lines (dark)");
    m_drawStyle->append("Hidden lines (bright)");
    m_drawStyle->append("Points");
    vv->viewOptionsMenu->add(m_drawStyle);
    m_drawStyle->setShortcut("Alt+w");
    m_drawStyle->setCallback([this](int style){
        setWireframe(WireframeMode(style));
    });
    m_drawStyle->select(0);

    m_showAxis = new ui::Button(vv->viewOptionsMenu, "ShowAxis");
    vv->viewOptionsMenu->add(m_showAxis);
    m_showAxis->setState(m_coordAxis);
    m_showAxis->setText("Show axis");
    m_showAxis->setShortcut("Shift+A");
    m_showAxis->setCallback([this](bool state){
        toggleAxis(state);
    });

    m_useTextures = new ui::Button(vv->viewOptionsMenu, "Texturing");
    m_useTextures->setVisible(false, ui::View::VR);
    m_useTextures->setShortcut("Shift+T");
    m_useTextures->setState(m_textured);
    vv->viewOptionsMenu->add(m_useTextures);
    m_useTextures->setCallback([this](bool state){
        m_textured = state;
        /*vsg::Texture* texture = new vsg::Texture2D;
        for (int unit=0; unit<4; ++unit)
        {
            if (m_textured)
                m_objectsStateSet->setTextureAttributeAndModes(unit, texture, vsg::StateAttribute::OFF);
            else
                m_objectsStateSet->setTextureAttributeAndModes(unit, texture, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::OFF);
        }*/
    });

    m_useShaders = new ui::Button(vv->viewOptionsMenu, "UseShaders");
    m_useShaders->setText("Use shaders");
    m_useShaders->setVisible(false, ui::View::VR);
    m_useShaders->setShortcut("Alt+s");
    m_useShaders->setState(m_shaders);
    vv->viewOptionsMenu->add(m_useShaders);
    m_useShaders->setCallback([this](bool state){
        m_shaders = state;
        /*vsg::Program* program = new vsg::Program;
        if (m_shaders)
        {
            m_objectsStateSet->setAttributeAndModes(program, vsg::StateAttribute::OFF);
        }
        else
        {
            m_objectsStateSet->setAttributeAndModes(program, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::OFF);
        }*/
    });


    auto tm = new ui::Action(vv->viewOptionsMenu, "ToggleMenu");
    vv->viewOptionsMenu->add(tm);
    tm->setText("Toggle VR menu");
    tm->addShortcut("m");
    tm->setCallback([this]() { toggleMenu(); });
    tm->setVisible(false);

    auto sm = new ui::Button(vv->viewOptionsMenu, "ShowMenu");
    vv->viewOptionsMenu->add(sm);
    sm->setText("Show VR menu");
    sm->setState(m_showMenu);
    sm->setCallback([this](bool state) { setMenu(state ? MenuAndObjects : MenuHidden); });
    sm->setVisible(false);
    m_showMenuButton = sm;

    auto fc = new ui::Action(vv->viewOptionsMenu, "ForceCompile");
    fc->setText("Force display list compilation");
    fc->setVisible(false);
    fc->addShortcut("Alt+Shift+C");
    fc->setCallback([this](){
            //vvViewer::instance()->forceCompile();
    });

    auto fs = new ui::Action(vv->viewOptionsMenu, "FlipStereo");
    fs->setText("Flip stereo 3D eyes");
    fs->setVisible(false);
    fs->addShortcut("Alt+e");
    fs->setCallback([this](){
            //vvViewer::instance()->flipStereo();
    });

    auto cw = new ui::Action(vv->viewOptionsMenu, "ClearWindow");
    cw->setText("Clear window");
    cw->setVisible(false);
    cw->addShortcut("Alt+c");
    cw->setCallback([this](){
            windowStruct *ws = &(vvConfig::instance()->windows[0]);
           // ws->window->setWindowRectangle(ws->ox, ws->oy, ws->sx, ws->sy);
           // ws->window->setWindowDecoration(false);
            vvViewer::instance()->clearWindow = true;
    });
}

vvSceneGraph::~vvSceneGraph()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\ndelete vvSceneGraph\n");

    //m_specialBoundsNodeList.clear();

    //delete vvShadowManager::instance();

    while (m_objectsRoot->children.size() > 0)
    {
        size_t n = m_objectsRoot->children.size();
        /*vsg::ref_ptr<vsg::Node> thisNode = m_objectsRoot->getChild(n - 1);
        while (thisNode && thisNode->getNumParents() > 0)
            thisNode->getParent(0)->removeChild(thisNode);*/
    }

    s_instance = NULL;
}

void vvSceneGraph::config()
{
    // if menu mode is on -- hide menus
    if (vvConfig::instance()->isMenuModeOn())
        setMenuMode(false);

    setMenu(m_showMenu);
}

int vvSceneGraph::readConfigFile()
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::readConfigFile\n");

    m_scaleFactor = coCoviseConfig::getFloat("VIVE.DefaultScaleFactor", 1.0f);

    m_floorHeight = coCoviseConfig::getFloat("VIVE.FloorHeight", -1250.0f);

    m_scaleAllOn = coCoviseConfig::isOn("VIVE.ScaleAll", false);

    std::string line = coCoviseConfig::getEntry("VIVE.ModelFile");
    if (!line.empty())
    {
        if (vv->debugLevel(3))
            fprintf(stderr, "loading model file '%s'\n", line.c_str());
        vvFileManager::instance()->loadFile(line.c_str());
    }

    m_coordAxis = coCoviseConfig::isOn("VIVE.CoordAxis", false);

    wiiPos = coCoviseConfig::getFloat("VIVE.WiiPointerPos", -250.0);

    bool configured = false;
    m_showMenu = coCoviseConfig::isOn("visible", "VIVE.UI.VRUI", m_showMenu == MenuAndObjects, &configured)
                     ? MenuAndObjects
                     : MenuHidden;

    return 0;
}

void vvSceneGraph::initMatrices()
{
    m_invBaseMatrix = vsg::dmat4();
    m_oldInvBaseMatrix = vsg::dmat4();
}

void vvSceneGraph::initSceneGraph()
{
    // create the scene node
    //m_scene = vvShadowManager::instance()->newScene();
    //m_scene->setName("VR_RENDERER_SCENE_NODE");
    m_scene = vsg::MatrixTransform::create();


    std::string shadowTechnique = covise::coCoviseConfig::getEntry("value","VIVE.ShadowTechnique","none");
    //vvShadowManager::instance()->setTechnique(shadowTechnique);

    m_objectsScene = new vsg::Group();
    //m_objectsScene->setName("VR_RENDER_OBJECT_SCENE_NODE");

   /* int numClipPlanes = vv->getNumClipPlanes();
    for (int i = 0; i < numClipPlanes; i++)
    {
        m_rootStateSet->setAttributeAndModes(vv->getClipPlane(i), vsg::StateAttribute::OFF);
        m_objectsStateSet->setAttributeAndModes(vv->getClipPlane(i), vsg::StateAttribute::OFF);
    }*/


    // create the pointer (hand) DCS node
    // add it to the scene graph as the first child
    //                                  -----
    m_handSwitch = vsg::Switch::create();
    m_scene->addChild(m_handSwitch);
    m_handTransform = vsg::MatrixTransform::create();
    m_handSwitch->addChild(false,m_handTransform);

    m_handIconScaleTransform = vsg::MatrixTransform::create();
    m_handIconSwitch = vsg::Switch::create();
    m_handIconScaleTransform->addChild(m_handIconSwitch);
    m_handAxisScaleTransform = vsg::MatrixTransform::create();
    m_pointerDepthTransform = vsg::MatrixTransform::create();
    m_handTransform->addChild(m_pointerDepthTransform);
    m_handTransform->addChild(m_handAxisScaleTransform);
    m_pointerDepthTransform->addChild(m_handIconScaleTransform);

    // scale the axis to vv->getSceneSize()/2500.0;
    const double scale = vv->getSceneSize() / 2500.0;
    m_handAxisScaleTransform->matrix = (vsg::scale(scale, scale, scale));

    // read icons size from covise.config, default is sceneSize/10
    m_handIconSize = coCoviseConfig::getFloat("VIVE.IconSize", vv->getSceneSize() / 25.0f);
    m_handIconOffset = coCoviseConfig::getFloat("VIVE.IconOffset", 0.0);

    // read icon displacement
    m_pointerDepth = coCoviseConfig::getFloat("VIVE.PointerDepth", m_pointerDepth);
    m_pointerDepthTransform->matrix = vsg::translate(0.0, (double)m_pointerDepth, 0.0);

    // do not intersect with objects under handDCS
    //m_handTransform->setNodeMask(m_handTransform->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    // we: render Menu first
    m_menuGroupNode = new vsg::Group();
    //m_menuGroupNode->setNodeMask(m_menuGroupNode->getNodeMask()&~Isect::ReceiveShadow);
    m_menuGroupNode->setValue("name","MenuGroupNode");
    if (m_showMenu)
        m_scene->addChild(m_menuGroupNode);

    // dcs for translating/rotating all objects
    m_objectsTransform = vsg::MatrixTransform::create();

    // dcs for scaling all objects
    m_scaleTransform = vsg::MatrixTransform::create();
    m_objectsTransform->addChild(m_scaleTransform);
    m_scene->addChild(m_objectsTransform);
    m_objectsScene->addChild(m_objectsTransform);

    // root node for all objects
    m_objectsRoot = vsg::MatrixTransform::create();
    m_objectsRoot->setValue("name", "OBJECTS_ROOT");

    m_scaleTransform->addChild(m_objectsRoot);

   /* vsg::StateSet* ss = m_menuGroupNode->getOrCreateStateSet();
    for (int i = 0; i < vv->getNumClipPlanes(); i++)
    {
        ss->setAttributeAndModes(vv->getClipPlane(i), vsg::StateAttribute::OFF);
    }*/

    m_alwaysVisibleGroupNode = new vsg::Group();
    /*m_alwaysVisibleGroupNode->setNodeMask(m_alwaysVisibleGroupNode->getNodeMask() & ~Isect::ReceiveShadow);
    m_alwaysVisibleGroupNode->setName("AlwaysVisibleGroupNode");*/
    m_scene->addChild(m_alwaysVisibleGroupNode);
    //vsg::StateSet *ssav = m_alwaysVisibleGroupNode->getOrCreateStateSet();
    //ssav->setMode(GL_DEPTH_TEST, vsg::StateAttribute::OFF|vsg::StateAttribute::PROTECTED|vsg::StateAttribute::OVERRIDE);
    //ssav->setRenderBinDetails(INT_MAX, "RenderBin");
    //ssav->setRenderingHint(vsg::StateSet::TRANSPARENT_BIN);
    /*ssav->setNestRenderBins(false);
    for (int i = 0; i < vv->getNumClipPlanes(); i++)
    {
        ssav->setAttributeAndModes(vv->getClipPlane(i), vsg::StateAttribute::OFF);
    }

    m_lineHider = new osgFX::Scribe();*/
}

void vvSceneGraph::initAxis()
{
    // geode for coordinate system axis
    m_worldAxis = loadAxisGeode(4);
    m_handAxis = loadAxisGeode(2);
    m_viewerAxis = loadAxisGeode(4);
    m_objectAxis = loadAxisGeode(0.01f);
    m_viewerAxisTransform = vsg::MatrixTransform::create();
    //m_viewerAxisTransform->setName("ViewerAxisTransform");
    m_scene->addChild(m_viewerAxisTransform);

    showSmallSceneAxis_ = coCoviseConfig::isOn("VIVE.SmallSceneAxis", false);
    if (showSmallSceneAxis_)
    {
        float sx = 0.1f * vv->getSceneSize();
        sx = coCoviseConfig::getFloat("VIVE.SmallSceneAxis.Size", sx);
        if (vv->debugLevel(3))
            fprintf(stderr, "VIVE.SmallSceneAxis.Size=%f\n", sx);
        float xp = coCoviseConfig::getFloat("x", "VIVE.SmallSceneAxis.Position", 500.0);
        float yp = coCoviseConfig::getFloat("y", "VIVE.SmallSceneAxis.Position", 0.0);
        float zp = coCoviseConfig::getFloat("z", "VIVE.SmallSceneAxis.Position", -500);
        m_smallSceneAxis = loadAxisGeode(0.01f * sx);
        //m_smallSceneAxis->setNodeMask(m_objectAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
        m_smallSceneAxisTransform = vsg::MatrixTransform::create();
        //m_smallSceneAxisTransform->setName("SmallSceneAxisTransform");
        m_smallSceneAxisTransform->matrix = (vsg::translate(xp, yp, zp));
        m_scene->addChild(m_smallSceneAxisTransform);

        vvLabel *XLabel, *YLabel, *ZLabel;
        float fontSize = coCoviseConfig::getFloat("VIVE.SmallSceneAxis.FontSize", 100);
        float lineLen = 0;
        vsg::vec4 red(1, 0, 0, 1);
        vsg::vec4 green(0, 1, 0, 1);
        vsg::vec4 blue(0, 0, 1, 1);
        vsg::vec4 bg(1, 1, 1, 0);
        XLabel = new vvLabel("X", fontSize, lineLen, red, bg);
        XLabel->reAttachTo(m_smallSceneAxisTransform);
        XLabel->setPosition(vsg::dvec3(1.1 * sx, 0.0, 0.0));
        XLabel->hideLine();

        YLabel = new vvLabel("Y", fontSize, lineLen, green, bg);
        YLabel->reAttachTo(m_smallSceneAxisTransform);
        YLabel->setPosition(vsg::dvec3(0.0, 1.1 * sx, 0.0));
        YLabel->hideLine();

        ZLabel = new vvLabel("Z", fontSize, lineLen, blue, bg);
        ZLabel->reAttachTo(m_smallSceneAxisTransform);
        ZLabel->setPosition(vsg::dvec3(0.0, 0.0, 1.1 * sx));
        ZLabel->hideLine();

        m_smallSceneAxisTransform->addChild(m_smallSceneAxis);
    }

    // no intersection with the axis
    /*m_worldAxis->setNodeMask(m_worldAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_handAxis->setNodeMask(m_handAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_viewerAxis->setNodeMask(m_viewerAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_objectAxis->setNodeMask(m_objectAxis->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));*/

    if (coCoviseConfig::isOn("VIVE.CoordAxis", false))
    {
        m_scene->addChild(m_worldAxis);
        m_viewerAxisTransform->addChild(m_viewerAxis);
        m_handAxisScaleTransform->addChild(m_handAxis);
        m_scaleTransform->addChild(m_objectAxis);
    }
}

void vvSceneGraph::initHandDeviceGeometry()
{
    m_handLine = loadHandLine();

    if (m_handLine)
        m_handIconSwitch->addChild(true, m_handLine);
    else
        fprintf(stderr, "vvSceneGraph::initHandDeviceGeometry: could not load hand line (at least a hand line object is required)\n");

    vsg::ref_ptr<vsg::Node> m_handIcon;
    for (int i = 1; i < NUM_HAND_TYPES; i++)
    {
        m_handIcon = loadHandIcon(handIconNames[i]);
        if (m_handIcon)
            m_handIconSwitch->addChild(false, m_handIcon);
        else
            m_handIconSwitch->addChild(false, m_handLine);
    }
}

// process key events
bool vvSceneGraph::keyEvent(vsg::KeyPressEvent& keyPress)
{
    bool handled = false;

    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::keyEvent\n");

    // Beschleunigung
   /* if (type == osgGA::GUIEventAdapter::KEYUP)
    {
    }
    if (type == osgGA::GUIEventAdapter::KEYDOWN)*/
    {
        if (keyPress.keyBase == '%')
        {
            //matrixCorrect->separateColor = !matrixCorrect->separateColor;
            //fprintf(stderr,"separateColor = %d\n",matrixCorrect->separateColor);
            //handled=true;
        }

        if (keyPress.keyModifier == vsg::MODKEY_Alt)
        {
            if (vv->debugLevel(3))
                fprintf(stderr, "alt: sym=%d\n", keyPress.keyBase);
            if (keyPress.keyBase == 'i' || keyPress.keyBase == 710) // i
            {
                saveScenegraph(false);
                handled = true;
            }
            else if (keyPress.keyBase == 'W' || keyPress.keyBase == 8722) // W
            {
                saveScenegraph(true);
                handled = true;
            }
            else if (keyPress.keyBase == 'm' || keyPress.keyBase == 181) //m
            {
            }
#ifdef _OPENMP
            else if (keyPress.keyBase >= '0' && keyPress.keyBase <= '9')
            {
                static int numInitialThreads = omp_get_max_threads();
                int targetNumThreads = (keyPress.keyBase == '0' ? numInitialThreads : keyPress.keyBase - '0');
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
vvSceneGraph::setObjects(bool state)
{
    m_showObjects = state;
    applyObjectVisibility();
}

void vvSceneGraph::applyObjectVisibility()
{
    if (m_showObjects && m_showMenu != MenuOnly)
    {
        /*if (m_objectsTransform->getNumParents() == 1)
        {
            if (m_wireframe == HiddenLineWhite || m_wireframe == HiddenLineBlack)
                m_scene->addChild(m_lineHider);
            else
                m_scene->addChild(m_objectsTransform);
        }*/
    }
    else
    {
        vvPluginSupport::removeChild(m_scene, m_objectsTransform);
        //vvPluginSupport::removeChild(m_lineHider,m_objectsTransform);
    }
}

void vvSceneGraph::setMenu(MenuMode state)
{
    m_showMenu = state;
    m_showMenuButton->setState(state != MenuHidden);

    if (m_showMenu == MenuHidden)
    {
        vvPluginSupport::removeChild(m_scene,m_menuGroupNode);
    }
    else
    {
            m_scene->addChild(m_menuGroupNode);
    }

    applyObjectVisibility();
}

bool
vvSceneGraph::menuVisible() const
{
    return m_showMenu;
}

void vvSceneGraph::setMenuMode(bool state)
{
    if (state)
    {
        menusAreHidden = false;

        applyMenuModeToMenus();

        // set joystickmanager active
        vrui::coJoystickManager::instance()->setActive(true);

        // set scene and documents not intersectable
        /*m_objectsTransform->setNodeMask(m_objectsTransform->getNodeMask() & (~Isect::Intersection));
        // set color of scene to grey
        if (vvConfig::instance()->colorSceneInMenuMode())
        {
            vsg::Material *mat = new vsg::Material();
            mat->setColorMode(vsg::Material::OFF);
            mat->setDiffuse(vsg::Material::FRONT_AND_BACK, vsg::Vec4(0.8, 0.8, 0.8, 0.5));
            mat->setAmbient(vsg::Material::FRONT_AND_BACK, vsg::Vec4(0.0, 0.0, 0.0, 0.0));
            mat->setSpecular(vsg::Material::FRONT_AND_BACK, vsg::Vec4(0.0, 0.0, 0.0, 0.0));
            mat->setEmission(vsg::Material::FRONT_AND_BACK, vsg::Vec4(0.0, 0.0, 0.0, 0.0));
            m_rootStateSet->setAttributeAndModes(mat, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            // disable shader
            m_rootStateSet->setAttributeAndModes(emptyProgram_, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_LIGHTING, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_NORMALIZE, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_BLEND, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            m_rootStateSet->setMode(vsg::StateAttribute::PROGRAM, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::VERTEXPROGRAM, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::FRAGMENTPROGRAM, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::TEXTURE, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::OFF);
        }*/
    }
    else
    {
        menusAreHidden = true;

        // set color of scene
        if (vvConfig::instance()->colorSceneInMenuMode())
        {
           /* vsg::Material* mat = new vsg::Material();
            m_rootStateSet->setAttributeAndModes(mat, vsg::StateAttribute::ON);
            // disable shader override
            m_rootStateSet->removeAttribute(emptyProgram_);
            m_rootStateSet->setMode(GL_LIGHTING, vsg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_NORMALIZE, vsg::StateAttribute::ON);
            m_rootStateSet->setMode(GL_BLEND, vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::PROGRAM, vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::VERTEXPROGRAM, vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::FRAGMENTPROGRAM, vsg::StateAttribute::OFF);
            m_rootStateSet->setMode(vsg::StateAttribute::TEXTURE, vsg::StateAttribute::OFF);*/
        }
        // set scene intersectable
        //m_objectsTransform->setNodeMask(0xffffffff);

#if 0
        // hide quit menu
        for (unsigned int i = 0; i < m_menuGroupNode->children.size(); i++)
        {
            if (m_menuGroupNode->children[i]->getName().find("Quit") != std::string::npos || m_menuGroupNode->children[i]->getName().find("beenden") != std::string::npos)
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

void vvSceneGraph::applyMenuModeToMenus()
{
    if (!vvConfig::instance()->isMenuModeOn())
    {
        return;
    }

    for (unsigned int i = 0; i < m_menuGroupNode->children.size(); i++)
    {
        /*if ((m_menuGroupNode->children[i]->getName().find("Quit") == std::string::npos)
            && (m_menuGroupNode->children[i]->getName().find("beenden") == std::string::npos)
            && (m_menuGroupNode->children[i]->getName().find("Document") == std::string::npos))
        {
            if (menusAreHidden)
                m_menuGroupNode->children[i]->setNodeMask(0x0);
            else
                m_menuGroupNode->children[i]->setNodeMask(0xffffffff & ~Isect::ReceiveShadow);
        }*/
    }
}

void vvSceneGraph::toggleHeadTracking(bool state)
{
    vvConfig::instance()->setFrozen(!state);
    m_trackHead->setState(state);
}

void
vvSceneGraph::toggleMenu()
{
    switch (m_showMenu)
    {
    case MenuOnly:
        m_showMenu = MenuAndObjects;
        break;
    case MenuAndObjects:
        m_showMenu = MenuHidden;
        break;
    case MenuHidden:
        if (vv->frameTime() - m_menuToggleTime < 3.)
            m_showMenu = MenuOnly;
        else
            m_showMenu = MenuAndObjects;
        break;
    }

    setMenu(m_showMenu);
    m_menuToggleTime = vv->frameTime();
}

void vvSceneGraph::setScaleFactor(double s, bool sync)
{
    m_scaleFactor = s;
    vsg::dmat4 scale_mat;
    // restrict scale
    if (m_scaleRestrictFactorMin != m_scaleRestrictFactorMax)
    {
        if (m_scaleFactor < m_scaleRestrictFactorMin)
            m_scaleFactor = m_scaleRestrictFactorMin;
        else if (m_scaleFactor > m_scaleRestrictFactorMax)
            m_scaleFactor = m_scaleRestrictFactorMax;
    }
    scale_mat= vsg::scale(m_scaleFactor, m_scaleFactor, m_scaleFactor);
    if (vvNavigationManager::instance()->getRotationPointActive())
    {
        vsg::dvec3 tv = vvNavigationManager::instance()->getRotationPoint();
        tv = tv * -1.0;
        vsg::dmat4 trans;
        trans= vsg::translate(tv);
        vsg::dmat4 temp = trans * scale_mat;
        trans= vsg::translate(vvNavigationManager::instance()->getRotationPoint());
        scale_mat = trans * temp;
    }
    m_scaleTransform->matrix = (scale_mat);
    if (sync)
        vvCollaboration::instance()->SyncXform();
}

void
vvSceneGraph::update()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "vvSceneGraph::update\n");
    if (m_firstTime)
    {
        if (vvConfig::instance()->numWindows() > 0)
        {
            const auto &ws = vvConfig::instance()->windows[0];
            if (ws.window)
            {
#ifdef WIN32
                if (vvVIVE::instance()->parentWindow == NULL)
#endif
                {
                   // ws.window->setWindowRectangle(ws.ox, ws.oy, ws.sx, ws.sy);
                   // ws.window->setWindowDecoration(ws.decoration);
                }
            }
        }
        m_firstTime = false;
    }
    
    if(!m_hidePointer->state())
    {
    if (Input::instance()->hasHand() && Input::instance()->isTrackingOn())
    {
        if (!m_pointerVisible)
        {
            m_handSwitch->children[0].mask = boolToMask(true);
            m_pointerVisible = true;
        }
    }
    else
    {
        if (m_pointerVisible)
        {
            m_handSwitch->children[0].mask = boolToMask(false);
            m_pointerVisible = false;
        }
    }
    }

    coPointerButton *button = vv->getPointerButton();
    vsg::dmat4 handMat = vv->getPointerMat();
    //cerr << "HandMat:" << handMat << endl;

    if (!vvConfig::instance()->isMenuModeOn())
    {
        if (button->wasPressed(vrui::vruiButtons::MENU_BUTTON))
            toggleMenu();
    }

    // transparency of handLine
    if (transparentPointer_ && m_handLine)
    {
        /*vsg::Material* mat = dynamic_cast<vsg::Material*>(m_handLine->getOrCreateStateSet()->getAttribute(vsg::StateAttribute::MATERIAL));
        if (mat)
        {
            vsg::Vec3 forward(0.0f, 1.0f, 0.0f);
            vsg::Vec3 vector = forward * handMat;
            float alpha = min(max(1.0f + (vector * forward) * 1.5f, 0.0f), 1.0f);
            mat->setAlpha(vsg::Material::FRONT_AND_BACK, alpha);
        }*/
    }

    if (vvConfig::instance()->useWiiNavigationVisenso())
    {
        // for wii interaction
        // restrict pointer for wii

        if (vvNavigationManager::instance()->getMode() == vvNavigationManager::TraverseInteractors && vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor())
        {
            vsg::dmat4 o_to_w = vv->getBaseMat();
            vsg::dvec3 currentInterPos_w = getTrans(vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->getMatrix()) * o_to_w;
            m_handTransform->matrix = translate(currentInterPos_w);
        }
        else
        {
            m_handTransform->matrix = (handMat);

            double bottom, top, left, right;
            bottom = -vvConfig::instance()->screens[0].vsize / 2;
            top = vvConfig::instance()->screens[0].vsize / 2;
            left = -vvConfig::instance()->screens[0].hsize / 2;
            right = vvConfig::instance()->screens[0].hsize / 2;
            vsg::dvec3 trans = getTrans(handMat);
            if (trans.x < left)
                trans[0] = left;
            else if (trans.x > right)
                trans[0] = right;
            if (trans.z < bottom)
                trans[2] = bottom;
            else if (trans.z > top)
                trans[2] = top;

            trans[1] = wiiPos;

            setTrans(handMat,trans);
            m_handTransform->matrix = (handMat);
        }
    }
    else // use tracking
    {
        //fprintf(stderr,"vvSceneGraph::update !wiiMode\n");

        if (vvNavigationManager::instance()->getMode() == vvNavigationManager::TraverseInteractors && vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor())
        {
            //fprintf(stderr,"vvNavigationManager::traverseInteractors && getCurrent\n");
            vsg::dmat4 o_to_w = vv->getBaseMat();
            vsg::dvec3 currentInterPos_w = getTrans(vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->getMatrix() * o_to_w);

            //vsg::Vec3 diff = currentInterPos_w - handMat.getTrans();
            //handMat.setTrans(handMat.getTrans()+diff);
            setTrans(handMat,currentInterPos_w);
            //if (vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->isRunning())
            //   fprintf(stderr,"diff = [%f %f %f]\n", diff[0], diff[1], diff[2]);
        }
        m_handTransform->matrix = (handMat);
    }

    // static vsg::Vec3Array *coord = new vsg::Vec3Array(4*6);
    vsg::dmat4 dcs_mat, rot_mat, tmpMat;
    //int collided[6] = {0, 0, 0, 0, 0, 0}; /* front,back,right,left,up,down */

    // update the viewer axis dcs with tracker not viewer mat
    // otherwise it is not updated when freeze is on
    if (Input::instance()->isHeadValid())
        m_viewerAxisTransform->matrix = (Input::instance()->getHeadMat());

    if (showSmallSceneAxis_)
    {
        dmat4 sm;
        sm = m_objectsTransform->matrix;
        dvec3 t;
        t = getTrans(m_smallSceneAxisTransform->matrix);
        sm(3, 0) = t[0];
        sm(3, 1) = t[1];
        sm(3, 2) = t[2];
        m_smallSceneAxisTransform->matrix = (sm);
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
            vsg::dmat4 xf, sc;
            vsg::Node *thisNode;
            xf = objectsTransform->matrix;
            sc = m_scaleTransform->matrix;
            getHandWorldPosition(pos, NULL, NULL);
            // work through all children
            n = objectsRoot->children.size();

            curr = 0;
            for (i = 0; i < n; i++)
            {

                thisNode = objectsRoot->children[i];
                if (pfIsOfType(thisNode, vsg::MatrixTransform::getClassType()))
                {
                    thisNode = ((vsg::MatrixTransform *)thisNode)->getChild(0);
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
        dvec3 trans = getTrans(m_objectsTransform->matrix);
        if (m_transRestrictMinX != m_transRestrictMaxX)
        {
            if (trans.x < m_transRestrictMinX)
                trans[0] = m_transRestrictMinX;
            else if (trans.x > m_transRestrictMaxX)
                trans[0] = m_transRestrictMaxX;
        }
        if (m_transRestrictMinY != m_transRestrictMaxY)
        {
            if (trans.y < m_transRestrictMinY)
                trans[1] = m_transRestrictMinY;
            else if (trans.y > m_transRestrictMaxY)
                trans[1] = m_transRestrictMaxY;
        }
        if (m_transRestrictMinZ != m_transRestrictMaxZ)
        {
            if (trans.z < m_transRestrictMinZ)
                trans[2] = m_transRestrictMinZ;
            else if (trans.z > m_transRestrictMaxZ)
                trans[2] = m_transRestrictMaxZ;
        }
        dmat4 m = m_objectsTransform->matrix;
        setTrans(m,trans);
        m_objectsTransform->matrix = (m);
    }

    if (m_enableHighQualityOption && !m_highQuality)
    {
        // HQ mode ON, if button is pressed while the mouse is higher then the head
        dvec3 pointerPosWld = getTrans(vv->getPointerMat());
        dvec3 viewerPosWld = getTrans(vv->getViewerMat());
        if (pointerPosWld[2] > viewerPosWld[2] && vv->getPointerButton()->wasPressed())
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
    else if (m_highQuality && (vv->getPointerButton()->wasPressed() || (vv->getMouseButton() && vv->getMouseButton()->wasPressed())))
    {
        m_highQuality = false;
        vrui::coInteractionManager::the()->unregisterInteraction(m_interactionHQ);
    }
}

void
vvSceneGraph::setWireframe(WireframeMode wf)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::setWireframe\n");

    m_wireframe = wf;
    m_drawStyle->select(m_wireframe);

    //vvPluginSupport::removeChild(m_lineHider,m_objectsTransform);
    vvPluginSupport::removeChild(m_scene, m_objectsTransform);
    //vvPluginSupport::removeChild(m_scene, m_lineHider);

    switch(m_wireframe)
    {
        case Disabled:
        case Enabled:
        case Points:
        {
            /*vsg::PolygonMode* polymode = new vsg::PolygonMode;
            if (m_wireframe == Disabled)
            {
                m_objectsStateSet->setAttributeAndModes(polymode, vsg::StateAttribute::ON);
            }
            else if (m_wireframe == Points)
            {
                polymode->setMode(vsg::PolygonMode::FRONT_AND_BACK, vsg::PolygonMode::POINT);
                m_objectsStateSet->setAttributeAndModes(polymode, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            }
            else
            {
                polymode->setMode(vsg::PolygonMode::FRONT_AND_BACK, vsg::PolygonMode::LINE);
                m_objectsStateSet->setAttributeAndModes(polymode, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::ON);
            }*/
            m_scene->addChild(m_objectsTransform);
            break;
        }
        case HiddenLineBlack:
        case HiddenLineWhite:
        {
            /*if (m_wireframe == HiddenLineBlack) {
                m_lineHider->setWireframeColor(vsg::Vec4(0, 0, 0, 1));
            } else {
                m_lineHider->setWireframeColor(vsg::Vec4(1, 1, 1, 1));
            }*/

            /*vsg::PolygonMode* polymode = new vsg::PolygonMode;
            m_objectsStateSet->setAttributeAndModes(polymode, vsg::StateAttribute::ON);
            */
            //m_scene->addChild(m_lineHider);
            //m_lineHider->addChild(m_objectsTransform);
            vvPluginSupport::removeChild(m_scene, m_objectsTransform);
            break;
        }
    }
}

void
vvSceneGraph::toggleHighQuality(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::toggleHighQuality %d\n", state);

    m_enableHighQualityOption = state;
    m_allowHighQuality->setState(state);
}

bool
vvSceneGraph::highQuality() const
{
    return m_highQuality;
}

void
vvSceneGraph::toggleAxis(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::toggleAxis %d\n", state);

    m_coordAxis = state;
    m_showAxis->setState(m_coordAxis);
    if (m_coordAxis)
    {
        m_scene->addChild(m_worldAxis);
        m_handAxisScaleTransform->addChild(m_handAxis);
        m_scaleTransform->addChild(m_objectAxis);
        if (vvConfig::instance()->frozen()
            && Input::instance()->hasHead())
        {
            //show viewer axis only if headtracking is disabled
            m_viewerAxisTransform->addChild(m_viewerAxis);
        }
        vvViewer::instance()->compile();
    }
    else
    {
        vvPluginSupport::removeChild(m_scene, m_worldAxis);
        vvPluginSupport::removeChild(m_viewerAxisTransform, m_viewerAxis);
        vvPluginSupport::removeChild(m_handAxisScaleTransform, m_handAxis);
        vvPluginSupport::removeChild(m_scaleTransform, m_objectAxis);
    }
}

void
vvSceneGraph::setPointerType(int pointerType)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::setHandType\n");

    m_pointerType = pointerType;
   fprintf(stderr, "vvSceneGraph::setHandType %d\n",pointerType);
    m_handIconSwitch->setAllChildren(false);
    m_handIconSwitch->setSingleChildOn(m_pointerType);

}

void
vvSceneGraph::adjustScale()
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::adjustScale mode=%f\n", m_scaleMode);

    // SCALE attribute not set
    // or SCALE attribute set to a number < 0.0

    if (m_scaleMode < 0.0)
    {
        // the scale to view all, but only if explicitly enables in covise.config
        if (m_scaleAllOn && vvNavigationManager::instance()->getMode() != vvNavigationManager::Scale)
            scaleAllObjects();
    }

    // SCALE keep
    else if (m_scaleMode == 0.0)
    { // keep scale so do nothing
    }

    // SCALE <number>
    else
    {
        vsg::dmat4 mat;
        m_scaleFactor = m_scaleMode;
        mat= vsg::scale(m_scaleFactor, m_scaleFactor, m_scaleFactor);
        m_scaleTransform->matrix = (mat);
    }
}

vsg::ref_ptr<vsg::MatrixTransform>
vvSceneGraph::loadAxisGeode(float s)
{

    if (vv->debugLevel(4))
        fprintf(stderr, "vvSceneGraph::loadAxisGeode\n");

     auto mt = vsg::MatrixTransform::create();
	//mt->setName("AxisGeodeMatrixTransform");
    if (!m_AxisGeometry.get())
    {
        m_AxisGeometry = vvFileManager::instance()->loadIcon("Axis");
    }
    if (m_AxisGeometry.get())
    {
        mt->addChild(m_AxisGeometry);
    }
    mt->matrix = vsg::scale(s, s, s);

    return (mt);
}

vsg::ref_ptr<vsg::Node>
vvSceneGraph::loadHandIcon(const std::string &name)
{
    vsg::ref_ptr<vsg::Node> n;
    n = vvFileManager::instance()->loadIcon(name);
    if (!n)
    {
        if (vv->debugLevel(3))
            fprintf(stderr, "failed to load hand icon: %s\n", name.c_str());
        n = vvFileManager::instance()->loadIcon("hlrsIcon");
    }
    return (n);
}

vsg::ref_ptr<vsg::Node>
vvSceneGraph::loadHandLine()
{
    if (!coCoviseConfig::isOn("visible", "VIVE.PointerAppearance.IconName", true))
        return nullptr;

    vsg::ref_ptr<vsg::Node> result;

    
    string iconName = coCoviseConfig::getEntry("VIVE.PointerAppearance.IconName");
    if (!iconName.empty())
    {
        auto n = vvFileManager::instance()->loadIcon(iconName.c_str());
        if (n.get())
        {
            float sx = 1.0;
            float sy = 1.0;
            float width = coCoviseConfig::getFloat("VIVE.PointerAppearance.Width", sx);
            float length = coCoviseConfig::getFloat("VIVE.PointerAppearance.Length", sy);

            vsg::MatrixTransform *m = vsg::MatrixTransform::create();
			//m->setName("HandLineMatrixTransform");
            m->matrix = (vsg::scale(width / sx, length / sy, width / sx));
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

    transparentPointer_ = (coCoviseConfig::isOn("VIVE.PointerAppearance.Transparent", false));
    if (transparentPointer_)
    {
        /*vsg::StateSet* sset = result->getOrCreateStateSet();
        sset->setRenderingHint(vsg::StateSet::TRANSPARENT_BIN);
        sset->setMode(GL_BLEND, vsg::StateAttribute::ON);
        vsg::Material *mat = new vsg::Material();
        sset->setAttributeAndModes(mat, vsg::StateAttribute::OVERRIDE | vsg::StateAttribute::PROTECTED);*/
        //mat->setColorMode(vsg::Material::AMBIENT_AND_DIFFUSE);
    }

    return result;
}

vsg::dvec3
vvSceneGraph::getWorldPointOfInterest() const
{
    vsg::dvec3 pointOfInterest(0.0, 0.0, 0.0);
    pointOfInterest = m_pointerDepthTransform->matrix * pointOfInterest;
    pointOfInterest = m_handIconScaleTransform->matrix *pointOfInterest;
    pointOfInterest = m_handTransform->matrix *pointOfInterest;

    return pointOfInterest;
}
/*
vsg::BoundingSphere
vvSceneGraph::getBoundingSphere()
{
    m_scalingAllObjects = true;
    dirtySpecialBounds();

    vsg::Group *scaleNode;
    if (coCoviseConfig::isOn("VIVE.ScaleWithInteractors", false))
        scaleNode = m_scaleTransform;
    else
        scaleNode = m_objectsRoot;

    scaleNode->dirtyBound();
    //vsg::BoundingSphere bsphere = vvSelectionManager::instance()->getBoundingSphere(m_objectsRoot);
    vsg::BoundingSphere bsphere;

    // determine center of bounding sphere
    BoundingBox bb;
    bb.init();
    vsg::Node *currentNode = NULL;
    for (unsigned int i = 0; i < scaleNode->children.size(); i++)
    {
        currentNode = scaleNode->children[i];
        const vsg::Transform *transform = currentNode->asTransform();
        if ((!transform || transform->getReferenceFrame() == vsg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
        {
            // Using the Visitor from scaleNode doesn't work if it's m_scaleTransform (ScaleWithInteractors==true) -> therefore the loop)
            NodeSet::iterator it = m_specialBoundsNodeList.find(currentNode);
            if (it == m_specialBoundsNodeList.end())
            {
                vsg::ComputeBoundsVisitor cbv;
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
        for (unsigned int i = 0; i < scaleNode->children.size(); i++)
        {
            currentNode = scaleNode->children[i];
            const vsg::Transform *transform = currentNode->asTransform();
            if ((!transform || transform->getReferenceFrame() == vsg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
            {
                NodeSet::iterator it = m_specialBoundsNodeList.find(currentNode);
                if (it == m_specialBoundsNodeList.end())
                {
                    vsg::ComputeBoundsVisitor cbv;
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

    vvPluginList::instance()->expandBoundingSphere(bsphere);

    if (vvMSController::instance()->isCluster())
    {
        struct BoundsData
        {
            BoundsData(const vsg::BoundingSphere &bs)
            {
                x = bs.center()[0];
                y = bs.center()[1];
                z = bs.center()[2];
                r = bs.radius();
            }

            float x, y, z;
            float r;
        };
        if (vvMSController::instance()->isSlave())
        {
            BoundsData bd(bsphere);
            vvMSController::instance()->sendMaster(&bd, sizeof(bd));

            vvMSController::instance()->readMaster(&bd, sizeof(bd));
            bsphere.center() = vsg::Vec3(bd.x, bd.y, bd.z);
            bsphere.radius() = bd.r;
        }
        else
        {
            vvMSController::SlaveData boundsData(sizeof(BoundsData));
            vvMSController::instance()->readSlaves(&boundsData);
            for (int i = 0; i < vvMSController::instance()->getNumSlaves(); ++i)
            {
                const BoundsData *bd = static_cast<const BoundsData *>(boundsData.data[i]);
                bsphere.expandBy(vsg::BoundingSphere(vsg::Vec3(bd->x, bd->y, bd->z), bd->r));
            }
            BoundsData bd(bsphere);
            vvMSController::instance()->sendSlaves(&bd, sizeof(bd));
            bsphere.center() = vsg::Vec3(bd.x, bd.y, bd.z);
            bsphere.radius() = bd.r;
        }
    }

    return bsphere;
}*/

void vvSceneGraph::scaleAllObjects(bool resetView, bool simple)
{


    // create view
    // compute the bounds of the scene graph to help position camera
    vsg::ComputeBounds computeBounds;
    m_objectsRoot->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.5;

    if (radius <= 0.f)
        radius = 1.f;

    // scale and translate but keep current orientation
    double scaleFactor = 1.f, masterScale = 1.f;
    vsg::dmat4 matrix, masterMatrix;
    boundingBoxToMatrices(computeBounds.bounds, resetView, matrix, scaleFactor);

    int len=0;
    if (vvMSController::instance()->isMaster())
    {
	covise::TokenBuffer tb;
        tb << scaleFactor;
        tb << matrix;
        len = tb.getData().length();
        vvMSController::instance()->sendSlaves(&len, sizeof(len));
        vvMSController::instance()->sendSlaves(tb.getData().data(), tb.getData().length());
    }
    else
    {
        vvMSController::instance()->readMaster(&len, sizeof(len));
        covise::DataHandle buf(len);
        vvMSController::instance()->readMaster(&buf.accessData()[0], len);
        covise::TokenBuffer tb(buf);
        tb >> masterScale;
        if (masterScale != scaleFactor)
        {
            fprintf(stderr, "vvSceneGraph::scaleAllObjects: scaleFactor mismatch on %d, is %f, should be %f\n",
                    vvMSController::instance()->getID(), scaleFactor, masterScale);
        }
        scaleFactor = masterScale;
        tb >> masterMatrix;
        if (masterMatrix != matrix)
        {
            std::cerr << "vvSceneGraph::scaleAllObjects: matrix mismatch on " << vvMSController::instance()->getID()
                << ", is " << matrix << ", should be " << masterMatrix << endl;
        }
        matrix = masterMatrix;
    }

    vv->setScale(scaleFactor);
    //fprintf(stderr, "bbox [x:%f %f\n      y:%f %f\n      z:%f %f]\n", bb.xMin(), bb.xMax(), bb.yMin(), bb.yMax(), bb.zMin(), bb.zMax());
    //fprintf(stderr, "vvSceneGraph::scaleAllObjects scalefactor: %f\n", scaleFactor);
    m_objectsTransform->matrix = (matrix);
}
/*
void vvSceneGraph::dirtySpecialBounds()
{
    for (NodeSet::iterator it = m_specialBoundsNodeList.begin();
         it != m_specialBoundsNodeList.end();
         ++it)
    {
        (*it)->dirtyBound();
    }
}*/

void vvSceneGraph::boundingBoxToMatrices(const vsg::dbox &boundingBox,
                                            bool resetView, vsg::dmat4 &currentMatrix, double &scaleFactor) const
{

    vsg::dvec3 center = (boundingBox.min + boundingBox.max) * 0.5;
    double radius = vsg::length(boundingBox.max - boundingBox.min) * 0.5;

    scaleFactor = std::abs(vv->getSceneSize() * radius);
    currentMatrix=translate(center * -scaleFactor);
    if (!resetView)
    {
        vsg::dvec3 t,s;
        vsg::dquat rotation;
        vsg::decompose(m_objectsTransform->matrix, t, rotation, s);
        vsg::dmat4 rotMat;
        rotMat=rotate(rotation);
        currentMatrix=currentMatrix*rotMat;
    }
}

#if 0
void
vvSceneGraph::manipulate(buttonSpecCell *spec)
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
vvSceneGraph::manipulateCallback(void *sceneGraph, buttonSpecCell *spec)
{
    ((vvSceneGraph *)sceneGraph)->manipulate(spec);
}
#endif

void
vvSceneGraph::viewAll(bool resetView, bool simple)
{
    vvNavigationManager::instance()->enableViewerPosRotation(false);

    scaleAllObjects(resetView, simple);

    vvCollaboration::instance()->SyncXform();
}

bool
vvSceneGraph::isHighQuality() const
{
    return m_highQuality;
}

bool
vvSceneGraph::saveScenegraph(bool storeWithMenu)
{
    std::string filename = coCoviseConfig::getEntry("value", "VIVE.SaveFile", "/var/tmp/VIVE.osgb");
    return saveScenegraph(filename, storeWithMenu);
}

bool
vvSceneGraph::saveScenegraph(const std::string &filename, bool storeWithMenu)
{
    if (isScenegraphProtected_)
    {
        fprintf(stderr, "Cannot store scenegraph. Not allowed!");
        return false;
    }

    if (vv->debugLevel(3))
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
        if (vv->debugLevel(1))
            std::cerr << "Writing to \"" << filename << "\": unknown extension, use .ive, .osg, .osgt, .osgb, or .osgx." << std::endl;
    }
    if(storeWithMenu)
    vsg::write( m_scene , filename, vvPluginSupport::instance()->options);
    else
    vsg::write( m_objectsRoot, filename, vvPluginSupport::instance()->options);
    /*if (osgDB::writeNodeFile(, filename.c_str()))
    {
        if (vv->debugLevel(3))
            std::cerr << "Data written to \"" << filename << "\"." << std::endl;
        return true;
    }*/

    if (vv->debugLevel(1))
        std::cerr << "Writing to \"" << filename << "\" failed." << std::endl;
    return false;
}

void
vvSceneGraph::setScaleFromButton(double direction)
{
    vv->setScale(vv->getScale() * (1.0f + (float)direction * m_scaleFactorButton));
    setScaleFactor(vv->getScale());
}

void vvSceneGraph::setScaleFactorButton(float f)
{
    m_scaleFactorButton = f;
}

void vvSceneGraph::setScaleRestrictFactor(float min, float max)
{
    m_scaleRestrictFactorMin = min;
    m_scaleRestrictFactorMax = max;
}

void vvSceneGraph::setRestrictBox(float minX, float maxX, float minY, float maxY, float minZ, float maxZ)
{
    m_transRestrictMinX = minX;
    m_transRestrictMinY = minY;
    m_transRestrictMinZ = minZ;
    m_transRestrictMaxX = maxX;
    m_transRestrictMaxY = maxY;
    m_transRestrictMaxZ = maxZ;
}

void
vvSceneGraph::getHandWorldPosition(double *pos, double *direction, double *direction2)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::getHandWorldPosition\n");

    vsg::dmat4 mat = vv->getPointerMat();
    vsg::dvec3 position = getTrans(mat);
    vsg::dvec3 normal;
    normal.set(mat(0, 1), mat(1, 1), mat(2, 1));
    vsg::dvec3 normal2;
    normal2.set(mat(0, 0), mat(1, 0), mat(2, 0));
    mat = vv->getInvBaseMat();
    position = position * mat;
    vsg::dmat3 m3;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            m3[r][c] = mat[r][c];
    normal = m3 * normal;
    normal2 = m3 * normal2;
    if (pos)
    {
        pos[0] = position[0];
        pos[1] = position[1];
        pos[2] = position[2];
    }
    if (direction)
    {
        normalize(normal);
        direction[0] = normal[0];
        direction[1] = normal[1];
        direction[2] = normal[2];
    }
    if (direction2)
    {
        normalize(normal2);
        direction2[0] = normal2[0];
        direction2[1] = normal2[1];
        direction2[2] = normal2[2];
    }
    return;
}

void vvSceneGraph::addPointerIcon(vsg::ref_ptr<vsg::Node> node)
{
    // recompute scale factor
    double scale = m_handIconSize;// / s;
    vsg::dmat4 m = vsg::scale(scale, scale, scale);
    setTrans(m,dvec3(0.0, (double)m_handIconOffset, 0.0));
    m_handIconScaleTransform->matrix = (m);

    // add icon
    //m_handIconScaleTransform->addChild(node);
    for (auto& child : m_handSwitch->children)
    {
        if (child.node.get() == node.get())
            child.mask = boolToMask(true);
    }
}

void vvSceneGraph::removePointerIcon(const vsg::Node *node)
{
    // remove icon
    for (auto &child : m_handSwitch->children)
    {
        if(child.node.get() == node)
            child.mask = boolToMask(false);
    }
    //vvPluginSupport::removeChild(m_handIconScaleTransform,node);
}

//******************************************
//           Coloring
//******************************************
/*
 * sets the color of the node nodeName
 */
/*
void vvSceneGraph::setColor(const char *nodeName, int *color, float transparency)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::setColor\n");
    Geode *node;
    //get the node
    node = findFirstNode<Geode>(nodeName);
    if (node != NULL)
    {
        //set the color
        setColor(node, color, transparency);
    }
}

void vvSceneGraph::setColor(vsg::Geode *geode, int *color, float transparency)
{
    ref_ptr<Drawable> drawable;
    if (geode != NULL)
    {
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            drawable = geode->getDrawable(i);
            //bool mtlOn = false;
            

            storeMaterial(drawable);

            if (color[0] == -1)
            {
                StoredMaterialsMap::iterator matIt = storedMaterials.find(drawable);
                if (matIt != storedMaterials.end())
                {
                    drawable->getStateSet()->setAttributeAndModes(matIt->second, vsg::StateAttribute::ON);
                }
            }
            else
            {
                // Create a new material since we dont want to change our default material.
                drawable->setStateSet(loadDefaultGeostate(vsg::Material::OFF));
                Material *mtl = (Material *)drawable->getStateSet()->getAttribute(StateAttribute::MATERIAL);
                mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, transparency));
                mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, transparency));
            }

            drawable->asGeometry()->setColorBinding(Geometry::BIND_OFF);

            drawable->getStateSet()->setNestRenderBins(false);
            if (transparency < 1.0)
            {
                drawable->getStateSet()->setRenderingHint(StateSet::TRANSPARENT_BIN);

                drawable->getStateSet()->setMode(GL_BLEND, vsg::StateAttribute::ON);

                // Disable writing to depth buffer.
                vsg::Depth *depth = new vsg::Depth;
                depth->setWriteMask(false);
                drawable->getStateSet()->setAttributeAndModes(depth, vsg::StateAttribute::ON);
            }
            else
            {
                drawable->getStateSet()->setRenderingHint(StateSet::OPAQUE_BIN);

                drawable->getStateSet()->setMode(GL_BLEND, vsg::StateAttribute::OFF);

                // Enable writing to depth buffer.
                vsg::Depth *depth = new vsg::Depth;
                depth->setWriteMask(true);
                drawable->getStateSet()->setAttributeAndModes(depth, vsg::StateAttribute::ON);
            }
            
            drawable->dirtyBound();
        }
    }
}

// set the transparency of the node with nodename
void vvSceneGraph::setTransparency(const char *nodeName, float transparency)
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
void vvSceneGraph::setTransparency(vsg::Geode *geode, float transparency)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::setTransparency\n");
    Drawable *drawable;
    Material *mtl = NULL;
    if (geode != NULL)
    {
        //fprintf(stderr, "vvSceneGraph::setTransparency %s %f\n", geode->getName().c_str(), transparency);
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

            if ((drawable->asGeometry() != NULL) && (drawable->asGeometry()->getColorBinding() == (vsg::Geometry::BIND_PER_VERTEX)
#if 0
               || drawable->asGeometry()->getColorBinding() == (vsg::Geometry::BIND_PER_PRIMITIVE)
               || drawable->asGeometry()->getColorBinding() == (vsg::Geometry::BIND_PER_PRIMITIVE_SET)
#endif
                                                     || drawable->asGeometry()->getColorBinding() == (vsg::Geometry::BIND_OVERALL)))
            {
                //fprintf(stderr, "vvSceneGraph::setTransparency %s color \n", geode->getName().c_str());
                //mtl->setColorMode(vsg::Material::AMBIENT_AND_DIFFUSE);
                // copy color array and use new transparency
                Vec4Array *oldColors = dynamic_cast<vsg::Vec4Array *>(drawable->asGeometry()->getColorArray());
                if (oldColors)
                {
                    ref_ptr<Vec4Array> newColors = new Vec4Array();
                    for (unsigned int i = 0; i < oldColors->size(); ++i)
                    {
                        vsg::Vec4 *color = &oldColors->operator[](i);
                        newColors->push_back(Vec4(color->_v[0], color->_v[1], color->_v[2], transparency));
                    }
                    drawable->asGeometry()->setColorArray(newColors);
                }
            }
            else if (mtl == NULL)
            {
                mtl = (Material *)drawable->getOrCreateStateSet()->getAttribute(StateAttribute::MATERIAL);
            }

            if (mtl != NULL)
            {
                //fprintf(stderr, "vvSceneGraph::setTransparency %s material \n", geode->getName().c_str());
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

                if (drawable->asGeometry()->getColorBinding() == vsg::Geometry::BIND_PER_VERTEX)
                {
                    mtl->setColorMode(vsg::Material::AMBIENT_AND_DIFFUSE);
                    // copy color array and use new transparency
                    Vec4Array *oldColors = dynamic_cast<vsg::Vec4Array *>(drawable->asGeometry()->getColorArray());
                    if (oldColors)
                    {
                        ref_ptr<Vec4Array> newColors = new Vec4Array();
                        for (unsigned int i = 0; i < oldColors->size(); ++i)
                        {
                            vsg::Vec4 *color = &oldColors->operator[](i);
                            newColors->push_back(Vec4(color->_v[0], color->_v[1], color->_v[2], transparency));
                        }
                        drawable->asGeometry()->setColorArray(newColors);
                    }
                }

                drawable->getOrCreateStateSet()->setAttributeAndModes(mtl, StateAttribute::ON);
                drawable->dirtyBound(); // trigger display list generation
            }
            drawable->getOrCreateStateSet()->setNestRenderBins(false);
            if (transparency < 1.0)
            {
                drawable->getOrCreateStateSet()->setRenderingHint(StateSet::TRANSPARENT_BIN);

                drawable->getOrCreateStateSet()->setMode(GL_BLEND, vsg::StateAttribute::ON);

                // Disable writing to depth buffer.
                vsg::Depth *depth = new vsg::Depth;
                depth->setWriteMask(false);
                drawable->getOrCreateStateSet()->setAttributeAndModes(depth, vsg::StateAttribute::ON);
            }
            else
            {
                drawable->getOrCreateStateSet()->setRenderingHint(StateSet::OPAQUE_BIN);

                drawable->getOrCreateStateSet()->setMode(GL_BLEND, vsg::StateAttribute::OFF);

                // Enable writing to depth buffer.
                vsg::Depth *depth = new vsg::Depth;
                depth->setWriteMask(true);
                drawable->getOrCreateStateSet()->setAttributeAndModes(depth, vsg::StateAttribute::ON);
            }
        }
    }
}

void vvSceneGraph::setShader(const char *nodeName, const char *shaderName, const char *paraFloat, const char *paraVec2, const char *paraVec3, const char *paraVec4, const char *paraInt, const char *paraBool, const char *paraMat2, const char *paraMat3, const char *paraMat4)
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

void vvSceneGraph::setShader(vsg::Geode *geode, const char *shaderName, const char *paraFloat, const char *paraVec2, const char *paraVec3, const char *paraVec4, const char *paraInt, const char *paraBool, const char *paraMat2, const char *paraMat3, const char *paraMat4)
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
void vvSceneGraph::setMaterial(const char *nodeName, int *ambient, int *diffuse, int *specular, float shininess, float transparency)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::setMaterial %f\n", transparency);
    Geode *node;
    //get the node
    node = findFirstNode<Geode>(nodeName);
    if (node != NULL)
    {
        setMaterial(node, ambient, diffuse, specular, shininess, transparency);
    }
}

// set opengl-material of node geode
void vvSceneGraph::setMaterial(vsg::Geode *geode, const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvSceneGraph::setMaterial 2 %f\n", transparency);

    if (geode != NULL)
    {
        //set the material
        ref_ptr<Material> mtl = new Material();
        mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(ambient[0] / 255.0, ambient[1] / 255.0, ambient[2] / 255.0, transparency));
        mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuse[0] / 255.0, diffuse[1] / 255.0, diffuse[2] / 255.0, transparency));
        mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(specular[0] / 255.0, specular[1] / 255.0, specular[2] / 255.0, transparency));
        mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, transparency));
        //mtl->setTransparency(Material::FRONT_AND_BACK ,transparency);
        mtl->setShininess(Material::FRONT_AND_BACK, shininess);
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            ref_ptr<Drawable> geoset = geode->getDrawable(i);
            storeMaterial(geoset);
            geoset->asGeometry()->setColorBinding(Geometry::BIND_OFF);
            ref_ptr<StateSet> gstate = geoset->getOrCreateStateSet();
            gstate->setAttributeAndModes(mtl.get(), StateAttribute::ON);
            gstate->setNestRenderBins(false);
            if (transparency < 1.0)
            {
                gstate->setRenderingHint(StateSet::TRANSPARENT_BIN);

                gstate->setMode(GL_BLEND, vsg::StateAttribute::ON);

                // Disable writing to depth buffer.
                vsg::Depth *depth = new vsg::Depth;
                depth->setWriteMask(false);
                gstate->setAttributeAndModes(depth, vsg::StateAttribute::ON);
            }
            else
            {
                gstate->setRenderingHint(StateSet::OPAQUE_BIN);

                gstate->setMode(GL_BLEND, vsg::StateAttribute::OFF);

                // Enable writing to depth buffer.
                vsg::Depth *depth = new vsg::Depth;
                depth->setWriteMask(true);
                gstate->setAttributeAndModes(depth, vsg::StateAttribute::ON);
            }

            geoset->setStateSet(gstate);
        }
    }
}

void vvSceneGraph::storeMaterial(vsg::Drawable *drawable)
{
    StoredMaterialsMap::iterator it = storedMaterials.find(drawable);
    if (it != storedMaterials.end())
    {
        return;
    }
    vsg::StateSet *sset = drawable->getStateSet();
    if (!sset)
    {
        return;
    }
    vsg::Material *mat = dynamic_cast<vsg::Material *>(sset->getAttribute(vsg::StateAttribute::MATERIAL, 0));
    if (!mat)
    {
        return;
    }
    storedMaterials.insert(std::pair<vsg::Drawable *, vsg::Material *>(drawable, mat));
}
*/
void vvSceneGraph::protectScenegraph()
{
    isScenegraphProtected_ = true;
}
