/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <math.h>
#include <cstdlib>
#include <array>

#include <util/common.h>
#include <config/CoviseConfig.h>

#include "vvVIVE.h"
#include "vvInteractor.h"
#include "vvTranslator.h"
#include "vvSceneGraph.h"
#include "vvCollaboration.h"
#include "vvNavigationManager.h"
#include "vvPluginSupport.h"
#include "vvConfig.h"
#include "vvMSController.h"
#include "vvCommunication.h"
#include "vvPluginList.h"
#include "vvViewer.h"
#include "vvLabel.h"
#include "vvSelectionManager.h"
#include "vvIntersection.h"
#include <input/input.h>
#include <input/vvMousePointer.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMouseButtonInteraction.h>
#include <OpenVRUI/coRelativeInputInteraction.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>
#include <OpenVRUI/vsg/mathUtils.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include "vvAnimationManager.h"
#include <ui/Menu.h>
#include <ui/Action.h>
#include <ui/Button.h>
#include <ui/ButtonGroup.h>
#include <ui/Slider.h>
#include <ui/View.h>
#include "vvIntersectionInteractorManager.h"
#include "vvMeasurement.h"
#include <vsg/utils/LineSegmentIntersector.h>
#include <vsg/utils/ComputeBounds.h>
#include <vsg/state/material.h>
#include <vsg/ui/KeyEvent.h>

#include <boost/algorithm/string/predicate.hpp>

using namespace vive;
using namespace vrui;
using covise::coCoviseConfig;

vvNavigationManager *vvNavigationManager::s_instance = NULL;

static float mouseX()
{
    return Input::instance()->mouse()->x();
}

static float mouseY()
{
    return Input::instance()->mouse()->y();
}

static float mouseWinWidth()
{
    return Input::instance()->mouse()->winWidth();
}

static float mouseWinHeight()
{
    return Input::instance()->mouse()->winHeight();
}

static float mouseScreenWidth()
{
    return Input::instance()->mouse()->screenWidth();
}

static float mouseScreenHeight()
{
    return Input::instance()->mouse()->screenHeight();
}

coVRNavigationProvider::coVRNavigationProvider(const std::string n, vvPlugin* p)
{
    name = n;
    plugin = p;
    ID = -1;
}
coVRNavigationProvider::~coVRNavigationProvider()
{
}

void coVRNavigationProvider::setEnabled(bool state)
{
    enabled = state;
}

vvNavigationManager *vvNavigationManager::instance()
{
    if (!s_instance)
        s_instance = new vvNavigationManager;
    return s_instance;
}

vvNavigationManager::vvNavigationManager()
    : ui::Owner("NavigationManager", vv->ui)
    , AnalogX(-10.0)
    , AnalogY(-10.0)
    , collision(false)
    , ignoreCollision(true)
    , navMode(NavNone)
    , oldNavMode(NavNone)
    , shiftEnabled(false)
    , shiftMouseNav(false)
    , isViewerPosRotation(false)
    , currentVelocity(0)
    , rotationPoint(false)
    , rotationPointVisible(false)
    , rotationAxis(false)
    , guiTranslateFactor(-1.0)
    , actScaleFactor(0)
    , jsEnabled(true)
    , joystickActive(false)
    , navExp(2.0)
    , driveSpeed(1.0)
    , jump(true)
    , snapping(false)
    , snappingD(false)
    , snapDegrees(-1.0)
    , rotationSpeed(50.0f)
    , turntable(false)
{
    assert(!s_instance);

    oldSelectedNode_ = NULL;
    oldShowNamesNode_ = NULL;

    init();
}

void vvNavigationManager::init()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\nnew vvNavigationManager\n");

    initInteractionDevice();

    readConfigFile();

    initMatrices();

    initMenu();

    initShowName();

    updatePerson();

    oldHandPos = vsg::dvec3(0, 0, 0);

    float radius = coCoviseConfig::getFloat("VIVE.rotationPointSize", 0.5); // obsolete config-rotationPointSize



    vsg::GeometryInfo geomInfo;
    geomInfo.position = vsg::vec3(0, 0, 0);
    geomInfo.dx.set(radius, 0.0f, 0.0f);
    geomInfo.dy.set(0.0f, radius, 0.0f);
    geomInfo.dz.set(0.0f, 0.0f, radius);

    vsg::StateInfo stateInfo;
    stateInfo.lighting = true;

    vvPluginSupport::instance()->builder->shaderSet = vsg::createPhongShaderSet(vvPluginSupport::instance()->options);
    if (auto& materialBinding = vvPluginSupport::instance()->builder->shaderSet->getDescriptorBinding("material"))
    {
        auto mat = vsg::PhongMaterialValue::create();
        mat->value().specular = vsg::vec4(1, 1, 1, 1);
        mat->value().diffuse = vsg::vec4(1, 0, 0, 1);
        materialBinding.data = mat;
    }

    rotPoint = vsg::MatrixTransform::create();
    rotPoint->addChild(vvPluginSupport::instance()->builder->createSphere(geomInfo,stateInfo));
    setRotationPointVisible(rotationPointVisible);

    setNavMode(XForm);
}

vvNavigationManager::~vvNavigationManager()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\ndelete vvNavigationManager\n");

    if (interactionA->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(interactionA);
    }
    if (interactionMenu->isRegistered())
        coInteractionManager::the()->unregisterInteraction(interactionMenu);
    coInteractionManager::the()->unregisterInteraction(interactionB);
    coInteractionManager::the()->unregisterInteraction(interactionC);
    delete interactionA;
    delete interactionB;
    delete interactionC;
    delete interactionMenu;
    delete interactionMA;
    delete interactionMB;
    delete interactionMC;
    delete interactionRel;
    delete interactionShortcut;

    s_instance = NULL;
}

void vvNavigationManager::updatePerson()
{
    if (!vvConfig::instance()->mouseTracking())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        coInteractionManager::the()->registerInteraction(interactionC);
    }
    else
    {
        coInteractionManager::the()->unregisterInteraction(interactionB);
        coInteractionManager::the()->unregisterInteraction(interactionC);
    }

    doMouseNav = vvConfig::instance()->mouseNav();
    if (doMouseNav)
    {
        //cerr << "using mouse nav " << endl;
        coInteractionManager::the()->registerInteraction(interactionMB);
        coInteractionManager::the()->registerInteraction(interactionMC);
    }
    else
    {
        coInteractionManager::the()->unregisterInteraction(interactionMB);
        coInteractionManager::the()->unregisterInteraction(interactionMC);
    }

    NavMode savedMode = oldNavMode;
    setNavMode(navMode);
    oldNavMode = savedMode;
}

void vvNavigationManager::initInteractionDevice()
{
    interactionA = new coNavInteraction(coInteraction::ButtonA, "ProbeMode", coInteraction::Navigation);
    interactionB = new coNavInteraction(coInteraction::ButtonB, "ProbeMode", coInteraction::Navigation);
    interactionC = new coNavInteraction(coInteraction::ButtonC, "ProbeMode", coInteraction::Navigation);
    interactionMenu = new coNavInteraction(coInteraction::ButtonA, "MenuMode", coInteraction::Menu);
    interactionShortcut = new coNavInteraction(coInteraction::NoButton, "Keyboard", coInteraction::Low);

    mouseNavButtonRotate = coCoviseConfig::getInt("RotateButton", "VIVE.Input.MouseNav", 0);
    mouseNavButtonScale = coCoviseConfig::getInt("ScaleButton", "VIVE.Input.MouseNav", 1);
    mouseNavButtonTranslate = coCoviseConfig::getInt("TranslateButton", "VIVE.Input.MouseNav", 2);
    interactionMA = new coMouseButtonInteraction(coInteraction::ButtonA, "MouseNav");
    interactionMA->setGroup(coInteraction::GroupNavigation);
    interactionMB = new coMouseButtonInteraction(coInteraction::ButtonB, "MouseNav");
    interactionMB->setGroup(coInteraction::GroupNavigation);
    interactionMC = new coMouseButtonInteraction(coInteraction::ButtonC, "MouseNav");
    interactionMC->setGroup(coInteraction::GroupNavigation);
    interactionRel = new coRelativeInputInteraction("SpaceMouse");
    interactionRel->setGroup(coInteraction::GroupNavigation);

    wiiNav = vvConfig::instance()->useWiiNavigationVisenso();
    /*if (wiiNav)
      fprintf(stderr, "Wii Navigation Visenso on\n");
   else
      fprintf(stderr, "Wii Navigation Visenso OFF\n");*/
    wiiFlag = 0;
}

void vvNavigationManager::registerNavigationProvider(coVRNavigationProvider *np)
{
    for (const auto& n : navigationProviders)
    {
        if (n->getName() == np->getName())
        {
            cerr << "Navigation \"" << n->getName() << "\" already registered" << endl;
            return;
        }
    }
    np->ID = NumNavModes + (int)navigationProviders.size();
    np->navMenuButton = new ui::Button(navModes_, np->getName(), navGroup_, np->ID);

    navigationProviders.push_back(np);

}
void vvNavigationManager::unregisterNavigationProvider(coVRNavigationProvider *np)
{
    if(np->isEnabled())
    {
        setNavMode(vvNavigationManager::XForm);
    }
    for (const auto& n : navigationProviders)
    {
        if (n == np)
        {
            navigationProviders.remove(np);
            delete np->navMenuButton;
            np->navMenuButton = nullptr;
            return;
        }
    }
    cerr << "Navigation \"" << np->getName() << "\" not found" << endl;
}

int vvNavigationManager::readConfigFile()
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::readConfigFile\n");

    navExp = coCoviseConfig::getFloat("VIVE.NavExpBase", (float)navExp);

    collisionDist = coCoviseConfig::getFloat("VIVE.CollisionDist", 80.0f);

    stepSize = coCoviseConfig::getFloat("VIVE.StepSize", 400.0f);

    jsXmax = coCoviseConfig::getInt("xmax", "VIVE.Input.Joystick", 1000);
    jsYmax = coCoviseConfig::getInt("ymax", "VIVE.Input.Joystick", 1000);
    jsXmin = coCoviseConfig::getInt("xmin", "VIVE.Input.Joystick", 0);
    jsYmin = coCoviseConfig::getInt("ymin", "VIVE.Input.Joystick", 0);
    jsZeroPosX = coCoviseConfig::getInt("zerox", "VIVE.Input.Joystick", 500);
    jsZeroPosY = coCoviseConfig::getInt("zeroy", "VIVE.Input.Joystick", 500);

    jsOffsetX = coCoviseConfig::getInt("x", "VIVE.Input.Offset", 20);
    jsOffsetY = coCoviseConfig::getInt("y", "VIVE.Input.Offset", 20);

    jsEnabled = coCoviseConfig::isOn("VIVE.Input.Joystick", false);
    visensoJoystick = coCoviseConfig::isOn("VIVE.Input.VisensoJoystick", false);

    menuButtonQuitInterval = coCoviseConfig::getFloat("value", "VIVE.MenuButtonQuitInterval", -1.0f);

    // parse snapping and tell NavigationManager
    snapping = coCoviseConfig::isOn("VIVE.Snap", snapping);
    snapDegrees = coCoviseConfig::getFloat("VIVE.SnapDegrees", (float)snapDegrees);
    rotationSpeed = coCoviseConfig::getFloat("VIVE.RotationSpeed", (float)rotationSpeed);
    if (snapDegrees > 0)
    {
        snappingD = true;
        snapping = true;
    }
    else
    {
        snappingD = false;
    }
    turntable = coCoviseConfig::isOn("VIVE.Turntable", turntable);

    m_restrict = coCoviseConfig::isOn("VIVE.Restrict", m_restrict);

    return 0;
}

void vvNavigationManager::initMatrices()
{
    invBaseMatrix = vsg::dmat4();
    oldInvBaseMatrix = vsg::dmat4();
}

void vvNavigationManager::initMenu()
{
    auto protectNav = [this](std::function<void()> f){
        bool registered = false;
        if (!interactionShortcut->isRegistered())
        {
            registered = true;
            coInteractionManager::the()->registerInteraction(interactionShortcut);
        }
        if (interactionShortcut->activate())
        {
            f();
        }
        if (registered)
        {
            coInteractionManager::the()->unregisterInteraction(interactionShortcut);
        }
    };

    navMenu_ = new ui::Menu("Navigation", this);

    m_viewAll = new ui::Action(navMenu_, "ViewAll");
    m_viewAll->setText("View all");
    m_viewAll->setShortcut("v");
    m_viewAll->setCallback([this, protectNav](){
        protectNav([](){
            vvSceneGraph::instance()->viewAll(false);
        });
    });
    m_viewAll->setPriority(ui::Element::Toolbar);
    m_viewAll->setIcon("zoom-fit-best");

    m_viewVisible = new ui::Action(navMenu_, "ViewVisible");
    m_viewVisible->setText("View visible");
    m_viewVisible->setShortcut("Shift+Alt+V");
    m_viewVisible->setCallback([this, protectNav](){
        protectNav([](){
            vvSceneGraph::instance()->viewAll(false, true);
        });
    });
    m_viewVisible->setPriority(ui::Element::Toolbar);
    m_viewVisible->setIcon("zoom-to-visible");


    centerViewButton = new ui::Action(navMenu_, "centerView");
    centerViewButton->setText("Straighten view");
    centerViewButton->setShortcut("r");
    centerViewButton->setCallback([this, protectNav]() {
        protectNav([this]() {
            centerView();
        });
    });
    centerViewButton->setIcon("zoom-in");

    m_resetView = new ui::Action(navMenu_, "ResetView");
    m_resetView->setText("Reset view");
    m_resetView->setShortcut("Shift+V");
    m_resetView->setCallback([this, protectNav]() {
        protectNav([]() {
            vvSceneGraph::instance()->viewAll(true);
        });
    });
    m_resetView->setIcon("zoom-original");

    scaleSlider_ = new ui::Slider(navMenu_, "ScaleFactor");
    scaleSlider_->setVisible(false, ui::View::VR);
    scaleSlider_->setText("Scale factor");
    scaleSlider_->setBounds(1e-5, 1e5);
    scaleSlider_->setScale(ui::Slider::Logarithmic);
    scaleSlider_->setValue(vv->getScale());
    scaleSlider_->setCallback([this, protectNav](double val, bool released) {
        protectNav([this, val]() {
            startMouseNav();
            doMouseScale((float)val);
            stopMouseNav();
        });
    });

    scaleUpAction_ = new ui::Action(navMenu_, "ScaleUp");
    scaleUpAction_->setText("Scale up");
    scaleUpAction_->setVisible(false);
    scaleUpAction_->setCallback([this, protectNav]() {
        protectNav([this]() {
            startMouseNav();
            doMouseScale(vv->getScale() * 1.1f);
            stopMouseNav();
        });
    });
    scaleUpAction_->setShortcut("=");
    scaleUpAction_->addShortcut("+");
    scaleUpAction_->addShortcut("Shift++");
    scaleUpAction_->addShortcut("Button:WheelDown");
    scaleDownAction_ = new ui::Action(navMenu_, "ScaleDown");
    scaleDownAction_->setText("Scale down");
    scaleDownAction_->setVisible(false);
    scaleDownAction_->setCallback([this, protectNav]() {
        protectNav([this]() {
            startMouseNav();
            doMouseScale(vv->getScale() / 1.1f);
            stopMouseNav();
        });
    });
    scaleDownAction_->setShortcut("-");
    scaleDownAction_->addShortcut("Button:WheelUp");

    navGroup_ = new ui::ButtonGroup(navMenu_, "NavigationGroup");
    //navGroup_->enableDeselect(true);
    navGroup_->setDefaultValue(NavNone);
    navModes_ = new ui::Group(navMenu_, "Modes");
    navModes_->setText("");
    noNavButton_ = new ui::Button(navModes_, "NavNone", navGroup_, NavNone);
    noNavButton_->setText("Navigation disabled");
    noNavButton_->setVisible(false);
    xformButton_ = new ui::Button(navModes_, "MoveWorld", navGroup_, XForm);
    xformButton_->setText("Move world");
    xformButton_->setShortcut("t");
    xformButton_->setPriority(ui::Element::Toolbar);
    scaleButton_ = new ui::Button(navModes_, "Scale", navGroup_, Scale);
    scaleButton_->setShortcut("s");
    flyButton_ = new ui::Button(navModes_, "Fly", navGroup_, Fly);
    flyButton_->setShortcut("f");
    walkButton_ = new ui::Button(navModes_, "Walk", navGroup_, Walk);
    walkButton_->setShortcut("w");
    driveButton_ = new ui::Button(navModes_, "Drive", navGroup_, Glide);
    driveButton_->setShortcut("d");
    driveButton_->setPriority(ui::Element::Toolbar);
    selectButton_ = new ui::Button(navModes_, "Selection", navGroup_, Select);
    showNameButton_ = new ui::Button(navModes_, "ShowName", navGroup_, ShowName);
    showNameButton_->setText("Show name");
    selectInteractButton_ = new ui::Button(navModes_, "SelectInteract", navGroup_, SelectInteract);
    selectInteractButton_->setText("Pick & interact");
    selectInteractButton_->setPriority(ui::Element::Toolbar);
    selectInteractButton_->setShortcut("p");
    selectInteractButton_->setEnabled(false);
    selectInteractButton_->setVisible(false);
    measureButton_ = new ui::Button(navModes_, "Measure", navGroup_, Measure);
    measureButton_->setVisible(false);
    traverseInteractorButton_ = new ui::Button(navModes_, "TraverseInteractors", navGroup_, TraverseInteractors);
    traverseInteractorButton_->setText("Traverse interactors");
    traverseInteractorButton_->setEnabled(false);
    traverseInteractorButton_->setVisible(false);
    navGroup_->setCallback([this](int value) { setNavMode(NavMode(value), false); });

    driveSpeedSlider_ = new ui::Slider(navMenu_, "DriveSpeed");
    driveSpeedSlider_->setText("Drive speed");
    driveSpeedSlider_->setBounds(.01, 100.);
    driveSpeedSlider_->setScale(ui::Slider::Logarithmic);
    driveSpeedSlider_->setValue(driveSpeed);
    driveSpeedSlider_->setCallback([this](double val, bool released) { driveSpeed = val; });

    collisionButton_ = new ui::Button(navMenu_, "Collision");
    collisionButton_->setText("Collision detection");
    collisionButton_->setState(collision);
    collisionButton_->setCallback([this](bool state) { collision = state; });

    snapButton_ = new ui::Button(navMenu_, "Snap");
    snapButton_->setShortcut("n");
    snapButton_->setState(snapping);
    snapButton_->setCallback([this](bool state) { snapping = state; });

    auto restrictButton = new ui::Button(navMenu_, "Restrict");
    restrictButton->setText("Restrict interactors");
    restrictButton->setState(m_restrict);
    restrictButton->setCallback([this](bool state) {
        m_restrict = state;
    });
    restrictButton->setVisible(true);
    restrictButton->setVisible(false, ui::View::VR);

#if 0
    ui::RadioButton *xformRotButton_=nullptr, *xformTransButton_=nullptr;
#endif
}

void vvNavigationManager::initShowName()
{
    vsg::vec4 fgcolor(0.5451f, 0.7020f, 0.2431f, 1.0f);
    vsg::vec4 bgcolor(0.0f, 0.0f, 0.0f, 0.8f);
    float lineLen = 0.06f * vv->getSceneSize();
    float fontSize = 0.02f * vv->getSceneSize();
    nameLabel_ = new vvLabel("test", fontSize, lineLen, fgcolor, bgcolor);
    nameLabel_->hide();

    showGeodeName_ = coCoviseConfig::isOn("VIVE.ShowGeodeName", true);

    nameMenu_ = new vrui::coRowMenu("Object Name");
    nameMenu_->setVisible(false);
    vrui::VSGVruiMatrix  t, r, m;
    double px = coCoviseConfig::getFloat("x", "VIVE.NameMenuPosition", -0.5f * vv->getSceneSize());
    double py = coCoviseConfig::getFloat("y", "VIVE.NameMenuPosition", 0.0f);
    double pz = coCoviseConfig::getFloat("z", "VIVE.NameMenuPosition", 0.3f * vv->getSceneSize());
    float scale;
    scale = coCoviseConfig::getFloat("VIVE.Menu.Size", 1.0);
    scale = coCoviseConfig::getFloat("VIVE.NameMenuScale", scale);

    t.setMatrix(vsg::translate(px, py, pz));
    r.makeEuler(0, 90, 0);
    r.mult(&t);
    nameMenu_->setTransformMatrix(&r);
    nameMenu_->setVisible(false);
    nameMenu_->setScale(scale * vv->getSceneSize() / 2500);
    nameButton_ = new coButtonMenuItem("-");
    nameMenu_->add(nameButton_);
}

vsg::dvec3 vvNavigationManager::getCenter() const
{
    if (rotationPoint)
    {
        return rotPointVec;
    }

    if (isViewerPosRotation)
    {
        return getTrans(vv->getViewerMat());
    }

    vsg::ComputeBounds computeBounds;
    vv->getObjectsRoot()->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    return(centre);
    //double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    /*
    vsg::BoundingSphere bsphere = vvSceneGraph::instance()->getBoundingSphere();
    if (bsphere.radius() == 0.f)
        bsphere.radius() = 1.f;
    vsg::vec3 originInWorld = bsphere.center() * vv->getBaseMat();

    if (originInWorld.length() < 0.5f*vv->getSceneSize())
    {
        return originInWorld;
    }*/

    return vsg::dvec3(0,0,0);
}

void vvNavigationManager::centerView(){
    vsg::LookAt la;
    la.set(vv->getXformMat());

        std::array<float, 6> angles;
        for (size_t i = 0; i < 3; i++)
        {
            vsg::dvec3 v;
            v[i] = 1;
            angles[i] = (float)std::acos(vsg::dot(la.up ,v));
            angles[i+3] = (float)std::acos(vsg::dot(la.up,-v));
        }
        auto maxAngle = std::min_element(angles.begin(), angles.end());
        size_t maxAngleIndex = maxAngle - angles.begin();
        la.up.set(0, 0, 0);
        if (maxAngleIndex < 3)
        {
            la.up[maxAngleIndex] = 1;
        }
        else
        {
            la.up[maxAngleIndex - 3] = -1;
        }

        float minDif = -1;
        size_t minDifIndex = 0, staySameIndex = 0;
        for (size_t i = 0; i < 3; i++)
        {
            if (la.up[i])
            {
                //remove inclination
                la.center[i] = la.eye[i];
            }
            else if(std::abs(la.center[i] - la.eye[i]) < minDif || minDif == -1)
            {
                minDif = (float)std::abs(la.center[i] - la.eye[i]);
                minDifIndex = i;
            }
        }
        for (size_t i = 0; i < 3; i++)
        {
            if (i != minDifIndex && la.up[i] == 0)
            {
                staySameIndex = i;
            }
        }
        if (la.center[staySameIndex] != la.eye[staySameIndex])
        {
            la.center[minDifIndex] = la.eye[minDifIndex];
        }

        if (la.eye == la.center)
        {
            std::cerr << "failed to center camera: eye = center" << std::endl;
            return;
        }

        vv->setXformMat(vsg::lookAt(la.eye, la.center, la.up));
}

// process key events
bool vvNavigationManager::keyEvent(vsg::KeyPressEvent& keyPress)
{
    bool handled = false;

    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::keyEvent\n");

    shiftEnabled = (keyPress.keyModifier & vsg::MODKEY_Shift);
    if (keyPress.keyBase == vsg::KEY_Up)
    {
        currentVelocity += (float)driveSpeed * 1.0f;
        handled = true;
    }
    else if (keyPress.keyBase == vsg::KEY_Right)
    {
        currentVelocity += (float)(currentVelocity * driveSpeed / 10.0);
        handled = true;
    }
    else if (keyPress.keyBase == vsg::KEY_Down)
    {
        // Verzoegerung
        currentVelocity -= (float)driveSpeed * 1.0f;
        handled = true;
    }
    else if (keyPress.keyBase == vsg::KEY_Left)
    {
        // halt
        currentVelocity = 0.0f;
        handled = true;
    }
    return handled;
}

void vvNavigationManager::saveCurrentBaseMatAsOldBaseMat()
{
    oldInvBaseMatrix = vv->getInvBaseMat();
    oldLeftPos = currentLeftPos;
    oldRightPos = currentRightPos;
}

// returns true if collision occurred
bool vvNavigationManager::avoidCollision(vsg::vec3 &glideVec)
{
    vsg::dvec3 oPos[2], Tip[2], nPos[2];
    glideVec.set(0, 0, 0);
    vsg::dvec3 rightPos(vvViewer::instance()->getSeparation() / 2.0f, 0.0f, 0.0f);
    vsg::dvec3 leftPos(-(vvViewer::instance()->getSeparation() / 2.0f), 0.0f, 0.0f);
    vsg::dmat4 vmat = vv->getViewerMat();
    currentRightPos = vmat * rightPos;
    currentLeftPos = vmat * leftPos;

    vsg::dmat4 currentBase = vv->getBaseMat();
    vsg::dmat4 diffMat = oldInvBaseMatrix * currentBase;

    // left segment
    nPos[0].set(currentLeftPos[0], currentLeftPos[1], currentLeftPos[2]);
    // right segment
    nPos[1].set(currentRightPos[0], currentRightPos[1], currentRightPos[2]);
    oPos[0] = diffMat * oldLeftPos;
    oPos[1] = diffMat * oldRightPos;

    vsg::dvec3 diff = nPos[0] - oPos[0];
    double dist = length(diff);
    if (dist < 1)
    {
        return false;
    }

    if (jump)
    {
        saveCurrentBaseMatAsOldBaseMat();
        jump = false;
        return false;
    }

    diff *= collisionDist / dist;
    Tip[0] = nPos[0] + diff;

    diff = nPos[1] - oPos[1];
    diff *= collisionDist / length(diff);
    Tip[1] = nPos[1] + diff;



    vsg::ref_ptr<vsg::LineSegmentIntersector> intersector[2];
    for (int i=0; i<1; ++i)
    {
        intersector[i] = vsg::LineSegmentIntersector::create(oPos[i], Tip[i]);
    }
    vv->getScene()->accept(*intersector[0]);
    
    vsg::dvec3 hitPoint[2];
    vsg::dvec3 hitNormal[2];

    vsg::dvec3 distVec[2];
    vsg::dvec3 glideV[2];
    double dists[2]{0.0, 0.0};
    for (int i=0; i<1; ++i)
    {
        if (!intersector[i]->intersections.size())
            continue;

        // sort the intersections front to back
        std::sort(intersector[i]->intersections.begin(), intersector[i]->intersections.end(), [](auto& lhs, auto& rhs) { return lhs->ratio < rhs->ratio; });
        auto isect = intersector[i]->intersections.front();
        hitPoint[i] = isect->worldIntersection;
        //hitNormal[i] = isect.getWorldIntersectNormal();

        distVec[i] = hitPoint[i] - oPos[i];
        distVec[i] *= 0.9;
        hitPoint[i] = oPos[i] + distVec[i];
        distVec[i] = hitPoint[i] - Tip[i];
        dists[i] = length(distVec[i]);
        //fprintf(stderr,"hitPoint: %f %f %f\n",hitPoint[i][0],hitPoint[i][1],hitPoint[i][2]);
        //fprintf(stderr,"Tip: %f %f %f\n",Tip[i][0],Tip[i][1],Tip[i][2]);
        //fprintf(stderr,"oPos: %f %f %f\n",oPos[i][0],oPos[i][1],oPos[i][2]);
        //fprintf(stderr,"nPos: %f %f %f\n",nPos[i][0],nPos[i][1],nPos[i][2]);
        //fprintf(stderr,"distVec: %f %f %f %d\n",distVec[i][0],distVec[i][1],distVec[i][2],i);
        //fprintf(stderr,"hitNormal: %f %f %f %d\n",hitNormal[i][0],hitNormal[i][1],hitNormal[i][2],i);

        /*auto tmp = distVec[i];
        tmp.normalize();
        hitNormal[i].normalize();
        float cangle = hitNormal[i] * (tmp);
        if (cangle > M_PI_2 || cangle < -M_PI_2)
        {
            hitNormal[i] = -hitNormal[i];
        }
        //fprintf(stderr,"hitNormal: %f %f %f %d\n",hitNormal[i][0],hitNormal[i][1],hitNormal[i][2],i);
        //fprintf(stderr,"tmp: %f %f %f %f\n",tmp[0],tmp[1],tmp[2],cangle);
        hitNormal[i] *= (dists[i]) * cangle;
        glideV[i] = hitNormal[i] - distVec[i];*/
        // fprintf(stderr,"glideV: %f %f %f %d\n",glideV[i][0],glideV[i][1],glideV[i][2],i);
    }

    if (intersector[0]->intersections.size())
    {
        int i = dists[0]>dists[1] ? 0 : 1;

        // get xform matrix
        vsg::dmat4 dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
        vsg::dmat4 tmp;
        tmp= vsg::translate(-distVec[i][0], -distVec[i][1], -distVec[i][2]);
        dcs_mat = tmp * dcs_mat;
        glideVec = -glideV[i];
        // set new xform matrix
        vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
        saveCurrentBaseMatAsOldBaseMat();
        vvCollaboration::instance()->SyncXform();
        return true;
    }

    saveCurrentBaseMatAsOldBaseMat();
    return false;
}

void
vvNavigationManager::adjustFloorHeight()
{
    if (navMode == Walk)
    {
        doWalkMoveToFloor();
    }
}

void
vvNavigationManager::update()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "vvNavigationManager::update\n");

    scaleSlider_->setValue(vv->getScale());

    coPointerButton *button = vv->getPointerButton();
    oldHandDir = handDir;
    if (!wiiNav)
    {
        handMat = vv->getPointerMat();
    }
    else
    {
        if (Input::instance()->isHandValid())
            handMat = Input::instance()->getHandMat();
    }
    oldHandPos = handPos;
    handPos = getTrans(handMat);
    handDir[0] = handMat(1, 0);
    handDir[1] = handMat(1, 1);
    handDir[2] = handMat(1, 2);

    // static vsg::vec3Array *coord = new vsg::vec3Array(4*6);
    vsg::dmat4 dcs_mat, rot_mat, tmpMat;
    vsg::dvec3 handRight, oldHandRight;
    vsg::dvec3 dirAxis, rollAxis;
    //int collided[6] = {0, 0, 0, 0, 0, 0}; /* front,back,right,left,up,down */
    //static int numframes=0;

    if (doMouseNav)
    {
        mx = mouseX();
        my = mouseY();
        float widthX = mouseWinWidth(), widthY = mouseWinHeight();
        originX = widthX / 2;
        originY = widthY / 2; // verallgemeinern!!!

        dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    }

    if (interactionMA->wasStarted() || interactionMB->wasStarted() || interactionMC->wasStarted())
    {
        switch (navMode)
        {
        case ShowName:
            startShowName();
            break;
        case SelectInteract:
            startSelectInteract();
            break;
        case Measure:
            startMeasure();
            break;
        default:
            startMouseNav();
            break;
        }
    }

    if (interactionMA->wasStopped() || interactionMB->wasStopped() || interactionMC->wasStopped())
    {
        switch (navMode)
        {
        case ShowName:
            stopShowName();
            break;
        case SelectInteract:
            stopSelectInteract(true);
            break;
        case Measure:
            stopMeasure();
            break;
        default:
            stopMouseNav();
            break;
        }
    }
	if (interactionMA->isRunning() || interactionMB->isRunning() || interactionMC->isRunning())
	{
		switch (navMode)
		{
		case Walk:
		case Glide:
			doMouseWalk();
			break;
		case Scale:
			doMouseScale();
			break;
		case XForm:
		case XFormTranslate:
		case XFormRotate:
			doMouseXform();
			break;
		case Fly:
			doMouseFly();
			break;
		case ShowName:
			doShowName();
			break;
		case SelectInteract:
			doSelectInteract();
			break;
		case Measure:
			doMeasure();
			break;
		case NavNone:
		case TraverseInteractors:
		case Menu:
			break;
		default:
			fprintf(stderr, "vvNavigationManager::update: unhandled navigation mode interaction %d\n", navMode);
			break;
		}
	}

	//fprintf(stderr, "vvNavigationManager: doMouseNav=%d, mouseFlag=%d, navMode=%d, handLocked=%d\n", (int)doMouseNav, (int)mouseFlag, (int)navMode, (int)handLocked);

	if (interactionA->wasStarted())
	{
		switch (navMode)
		{
		case XForm:
		case XFormTranslate:
		case XFormRotate:
			startXform();
			break;
		case Fly:
			startFly();
			break;
		case Walk:
			startWalk();
			break;
		case Glide:
			startDrive();
			break;
		case Scale:
			startScale();
			break;
		case ShowName:
			startShowName();
			break;
		case SelectInteract:
			startSelectInteract();
			break;
		case Measure:
			startMeasure();
			break;
		case TraverseInteractors:
		case Menu:
		case NavNone:
			break;
		default:
			fprintf(stderr, "vvNavigationManager::update: unhandled navigation mode %d\n", (int)navMode);
			break;
		}
	}
	if (interactionB->wasStarted())
	{

		if (navMode != TraverseInteractors)
			startDrive();
	}
	if (interactionC->wasStarted())
	{
		startXform();
	}

	if (interactionA->isRunning())
	{
		switch (navMode)
		{
		case XForm:
			doXform();
			break;
		case Fly:
			doFly();
			break;
		case Walk:
			doWalk();
			break;
		case Glide:
			doDrive();
			break;
		case Scale:
			doScale();
			break;
		case ShowName:
			doShowName();
			break;
		case SelectInteract:
			doSelectInteract();
			break;
		case Measure:
			doMeasure();
			break;
		case XFormTranslate:
			doXformTranslate();
			break;
		case XFormRotate:
			doXformRotate();
			break;
		case TraverseInteractors:
		case Menu:
		case NavNone:
			break;
		default:
			fprintf(stderr, "vvNavigationManager::update: unhandled navigation mode %d\n", (int)navMode);
			break;
		}
	}
	if (interactionB->isRunning())
	{
		if (navMode != TraverseInteractors)
			doDrive();
	}
	if (interactionC->isRunning())
	{
		doXform();
	}

	if (interactionA->wasStopped())
	{
		switch (navMode)
		{
		case XForm:
		case XFormTranslate:
		case XFormRotate:
			stopXform();
			break;
		case Fly:
			stopFly();
			break;
		case Walk:
			stopWalk();
			break;
		case Glide:
			stopDrive();
			break;
		case Scale:
			stopScale();
			break;
		case ShowName:
			stopShowName();
			break;
		case SelectInteract:
			stopSelectInteract(false);
			break;
		case Measure:
			stopMeasure();
			break;
		case TraverseInteractors:
		case Menu:
		case NavNone:
			break;
		default:
			fprintf(stderr, "vvNavigationManager::update: unhandled navigation mode %d\n", (int)navMode);
			break;
		}
	}
	if (interactionB->wasStopped())
	{
		stopDrive();
	}
	if (interactionC->wasStopped())
	{
		stopXform();
	}

	if (interactionRel->wasStarted())
	{
		switch (getMode())
		{
		case Scale:
			startMouseNav();
			break;
		default:
			break;
		}
	}

	if (interactionRel->isRunning())
	{
        auto viewer = vv->getViewerMat();
        setTrans(viewer,vsg::dvec3(0, 0, 0));
        auto viewerInv = vsg::inverse(viewer);

        vsg::dmat4 relMat = Input::instance()->getRelativeMat();
        relMat = viewerInv * relMat * viewer;
        coCoord co(relMat);
		vsg::dmat4 tf = vvSceneGraph::instance()->getTransform()->matrix;
		auto tr = applySpeedFactor(getTrans(relMat));

		switch (getMode())
		{
		case Scale:
		{
			float s = pow(1.03f, (float)co.hpr[0]);
			doMouseScale(vv->getScale() * s);
			break;
		}
		case XForm:
		{
			vsg::dvec3 center = getCenter();

			relMat= vsg::translate(-tr);

            tf = relMat * tf;

            relMat = makeEulerMat(-co.hpr[0], -co.hpr[1], -co.hpr[2]);
            vsg::dmat4 originTrans, invOriginTrans;
			originTrans= vsg::translate(center); // rotate arround the center of the objects in objectsRoot
			invOriginTrans= vsg::translate(-center);
			relMat =   originTrans * relMat * invOriginTrans;
			tf = relMat* tf;
			break;
		}
		case Fly:
		{
			setTrans(relMat,tr);
            tf = relMat * tf;
			break;
		}
		case Glide:
		case Walk:
		{
            relMat = makeEulerMat(co.hpr[0], 0.0,0.0);
            setTrans(relMat, tr);
            tf = relMat * tf;
            break;
        }
        default:
        {
            break;
        }
        }

        if (tf != vvSceneGraph::instance()->getTransform()->matrix)
        {
            vvSceneGraph::instance()->getTransform()->matrix = (tf);
			vvCollaboration::instance()->SyncXform();
        }
    }

	if (interactionRel->wasStopped())
	{
		switch (getMode())
		{
		case Scale:
			stopMouseNav();
			break;
		default:
			break;
		}
	}


    // was ist wenn mehrere Objekte geladen sind???

    if (jsEnabled && AnalogX >= 0.0 && AnalogY >= 0.0
        && (!vv->isPointerLocked())
        && (!(vv->getMouseButton() && (doMouseNav))))
    {
        dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
        int xMove = (fabs(AnalogX - (float)jsZeroPosX) > (float)jsOffsetX);
        int yMove = (fabs(AnalogY - (float)jsZeroPosY) > (float)jsOffsetY);
        //wenn joystick in der nullstellung ist: zuruecksetzen vo firstTime
        static int firstTime = 1;
        if (!xMove && !yMove)
            firstTime = 1;

        AnalogX = (AnalogX - jsZeroPosX) / 10;
        AnalogY = (AnalogY - jsZeroPosY) / 10;
        float maxAnalogX = (jsXmax - jsZeroPosX) / 10.0f;
        float maxAnalogY = (jsYmax - jsZeroPosY) / 10.0f;

        if ((xMove || yMove) && navMode != Walk && navMode != Fly && navMode != Glide && navMode != ShowName && navMode != Measure && navMode != TraverseInteractors && navMode != SelectInteract)
        {

            //einfache Translation in xy-Ebene
            if ((button->getState() & vruiButtons::ACTION_BUTTON) == 0)
            {
                double maxVelo = 10.0 * driveSpeed;
                vsg::dvec3 relVelo, absVelo;
                relVelo[0] = -(maxVelo / maxAnalogX) * AnalogX;
                relVelo[1] = -(maxVelo / maxAnalogY) * AnalogY;
                relVelo[2] = 0.0;
                vsg::dmat3 m3;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        m3[r][c] = handMat[r][c];
                absVelo = m3 * relVelo;
                //absVelo = vsg::transform3x3(handMat, relVelo);
                vsg::dmat4 tmp;
                tmp= vsg::translate((double)absVelo[0], (double)absVelo[1], 0.0);
                dcs_mat = tmp * dcs_mat;
            }

            //Rotation um Objekt
            else if (button->getState() & vruiButtons::ACTION_BUTTON)
            {

                static double trX, trY, trZ;
                trX = dcs_mat(3, 0);
                trY = dcs_mat(3, 1); //diese drei sind fuer Rotation um Objektursprung
                trZ = dcs_mat(3, 2);
                double maxRotDegrees = driveSpeed;
                double degreesX = (maxRotDegrees / maxAnalogY) * AnalogY;
                double degreesY = (maxRotDegrees / maxAnalogX) * AnalogX;
                vsg::dmat4 tmp;
                tmp= vsg::translate(-trX, -trY, -trZ);
                dcs_mat = tmp * dcs_mat;

                /*               cout<<"\ndcs_mat vor Rotation\n";
                           for(int row=0; row<=3; row++)
                           {
                               for(int col=0; col<=3; col++)
                                   cout << dcs_mat[row][col] << " ";
                               cout << "\n";
                           }
            */
                //rotate
                if (xMove) //nur Rotation um Achse die y entspricht
                {
                    vsg::dmat4 rot;
                    rot = makeEulerMat(degreesY, 0.0, 0.0);
                    dcs_mat = rot * dcs_mat;
                }
                if (yMove) // nurRotation um Achse die x entspricht
                {
                    vsg::dmat4 rot;
                    rot = makeEulerMat(0.0, degreesX, 0.0);
                    dcs_mat = rot * dcs_mat;
                }

                tmp= vsg::translate(trX, trY, trZ);
                dcs_mat = tmp * dcs_mat;
            }
        }

        //drive- mode, mit Translation in Joystick y
        else if ((xMove || yMove) && navMode == Glide)
        //und Rotation um Hand in Joystick x
        {
            double maxVelo = 10.0 * driveSpeed;
            vsg::dvec3 relVelo, absVelo;
            relVelo[0] = 0.0;
            relVelo[1] = -(maxVelo / maxAnalogY) * AnalogY;
            relVelo[2] = 0.0;
            vsg::dmat3 m3;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    m3[r][c] = handMat[r][c];
            absVelo = m3 * relVelo;
            //absVelo = vsg::dmat4::transform3x3(handMat, relVelo);

            //Ende Translation Joystick bottom-top

            //jetzt Rotation Joystick left-right, zunaechst Werte fuer TRansformatio berechnen
            //unten dann zwischen +-handPos[] ausfuehren
            dirAxis[0] = 0.0;
            dirAxis[1] = 0.0;
            dirAxis[2] = 1.0;

            //normieren auf Abstand Objekt- BetrachterHand
            vsg::dvec3 objPos;
            objPos = getTrans(dcs_mat);
            vsg::dvec3 distance = handPos - objPos;
            //Schaetzwert in der richtigen Groessenordnung
            double distObjHand = vsg::length(distance) / 500.0;
            double maxRotDegrees = driveSpeed; //n bisschen allgemeiner waer gut
            double degrees = (maxRotDegrees / (maxAnalogX * distObjHand)) * -AnalogX;

            vsg::dmat4 tmp;
            tmp= vsg::translate(-handPos[0], -handPos[1], -handPos[2]);
            dcs_mat = tmp * dcs_mat;
            if (xMove)
            {
                tmp = vsg::rotate(degrees, dirAxis[0], dirAxis[1], dirAxis[2]);
                dcs_mat = dcs_mat * tmp;
            }
            tmp= vsg::translate(handPos[0], handPos[1], handPos[2]);
            dcs_mat = tmp * dcs_mat;
            if (yMove)
            {
                tmp= vsg::translate(absVelo[0], absVelo[1], 0.0);
                dcs_mat = tmp * dcs_mat;
            }
        }

        else if ((xMove || yMove) && navMode == Fly)
        //translation in Richtung von pointer in Joystick y und roll der Hand
        //in Joystick x: roll und Kreisbewegung mit Radius abhaengig vom Joystick-ausschlag
        //Kreisbewegung wie bei drive aber in der ebene in die der pointer zeigt und Hin- und
        //hertranslation inPunkt mit Abstand des radius , idee hierfuer, drehzentrum in der xy-ebene bestimmen
        //z-Koord. des objekts aber auch beruecksichtigen?? und dann mit handMat multiplizieren,
        //mit neu erhaltenen Koordinaten Hintranslation -> drehung -> Ruecktranslation
        //Handbewegung mitbeachten
        {

            double maxVelo = 10.0 * driveSpeed;
            vsg::dvec3 relVelo, absVelo;
            relVelo[0] = 0.0; //-(maxVelo / maxAnalogX) * AnalogX;
            relVelo[1] = -(maxVelo / maxAnalogY) * AnalogY;
            relVelo[2] = 0.0;
            vsg::dmat3 m3;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    m3[r][c] = handMat[r][c];
            absVelo = m3 * relVelo;
            //absVelo = vsg::transform3x3(handMat, relVelo);
            dcs_mat * vsg::translate(absVelo[0], absVelo[1], absVelo[2]);

            //translation in z-richtung so ok oder anderes vorzeichen ???

            // if xform just initiated   //noch andere if-Bedingungen ueberlegen fuer joystickstellung

            //if(!xMove && !yMove)

            //an dieser stelle im zusammenspiel mit joystick noch nicht fehlerfrei, objekt springt gelegentlich
            /*           if(   oldHandLocked
              || (   button->wasPressed()
              && (   button->getState() & DRIVE_BUTTON)) )
         */

            if (firstTime)
            {
                firstTime = 0;

                old_mat = handMat;
                old_mat = inverse(old_mat);
                old_dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
            }
            // if xform in progress

            //else

            //diese if macht im moment nichts gescheites
            if (button->getState() & vruiButtons::DRIVE_BUTTON)
            {
                setTrans(handMat, vsg::dvec3(0.0, 0.0, 0.0));
                setTrans(old_mat, vsg::dvec3(0.0, 0.0, 0.0));
                setTrans(old_dcs_mat, vsg::dvec3(0.0, 0.0, 0.0));

                vsg::dmat4 rel_mat; //, dcs_mat;
                rel_mat = handMat * old_mat; //erste handMat * aktualisierte handMat
                dcs_mat = rel_mat * old_dcs_mat;
                //vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
                vvCollaboration::instance()->SyncXform();
            }

            // calc axis and angle for roll change (from hand roll change)
            handRight.set(handMat(0, 0), handMat(0, 1), handMat(0, 2));
            oldHandRight.set(old_mat(0, 0), old_mat(0, 1), old_mat(0, 2));
            double rollScale;
            double rollAngle;
            rollAxis = cross(handRight, oldHandRight);
            rollScale = fabs(vsg::dot(rollAxis,handDir));
            rollAxis = rollAxis * rollScale;
            rollAngle = length(rollAxis);

            // apply roll change
            //cout << "rollAxis " << rollAxis[0] << "  " << rollAxis[1] << "  " << rollAxis[2] << endl;
            if ((rollAxis[0] != 0.0) || (rollAxis[1] != 0.0) || (rollAxis[2] != 0.0))
            {

                rot_mat = vsg::rotate(rollAngle * M_PI / 180, (double)rollAxis[0], (double)rollAxis[1], (double)rollAxis[2]);
                dcs_mat = rot_mat * dcs_mat;
            }

            //Ziel: aus Handbewegung auf heading schliessen

            static double trFlyX, trFlyY, trFlyZ;
            trFlyX = dcs_mat(3, 0);
            trFlyY = dcs_mat(3, 1); //diese drei sind fuer Rotation um Objektursprung
            trFlyZ = dcs_mat(3, 2);

            //cos(winkel) = a*b / abs(a) * abs(b)

            double lastDegrees = acos(oldHandDir[2] / length(oldHandDir));
            double degrees = acos(handDir[2] / length(handDir));
            //cout << "winkel " << 90.0 / acos(0.0) * degrees << endl;

            //float degreesX = 0.0;
            double degreesY = (90.0 / acos(0.0)) * (degrees - lastDegrees);
            dcs_mat = vsg::translate(-trFlyX, -trFlyY, -trFlyZ)* dcs_mat;

            //rotate
            vsg::dmat4 rot;
            rot = makeEulerMat(degreesY, 0.0, 0.0);
            dcs_mat = rot * dcs_mat;

            /*  vsg::dmat4 rot;
             MAKE_EULER_MAT(rot,0.0, degreesX, 0.0);
            dcs_mat = rot * dcs_mat;
         */

            vsg::dmat4 tmp;
            tmp= vsg::translate(trFlyX, trFlyY, trFlyZ);
            dcs_mat = tmp * dcs_mat;
        }

        // nur mal zum ausprobieren, rotation um pointerachse
        else if ((xMove || yMove) && navMode == Walk)
        {
            //normieren auf Abstand Objekt- BetrachterHand
            double maxRotDegrees = driveSpeed;
            double degrees = (maxRotDegrees / maxAnalogX) * -AnalogX;
            dirAxis = handDir;

            if (xMove)
            {
                dcs_mat = vsg::rotate(degrees, dirAxis[0], dirAxis[1], dirAxis[2]) * dcs_mat;
            }
            if (yMove)
            {
            }
        }

        vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
        vvCollaboration::instance()->SyncXform();
    }

    if ((visensoJoystick) && (AnalogX != -10.0f || AnalogY != -10.0f))
    {
        bool previousJoystickActive = joystickActive;
        joystickActive = (fabs(AnalogX) > 0.01f) || (fabs(AnalogY) > 0.01f);

        // scale and translate with joystick
        if (getMode() == XForm)
        {
            if (joystickActive)
            {
                vvSceneGraph::instance()->setScaleFromButton(AnalogY * vv->frameDuration());
                doGuiTranslate((float)( -AnalogX * vv->frameDuration()), 0.0f, 0.0f);
            }
        }
        // walk with joystick
        if (getMode() == Walk || getMode() == Glide || getMode() == Fly)
        {
            if (joystickActive && !previousJoystickActive)
            {
                switch (getMode())
                {
                case Walk:
                    startWalk();
                    break;
                case Glide:
                    startDrive();
                    break;
                case Fly:
                    startFly();
                    break;
                default:
                    break;
                }
            }
            if (!joystickActive && previousJoystickActive)
            {
                switch (getMode())
                {
                case Walk:
                    stopWalk();
                    break;
                case Glide:
                    stopDrive();
                    break;
                case Fly:
                    stopFly();
                    break;
                default:
                    break;
                }
            }
            if (joystickActive)
            {
                switch (getMode())
                {
                case Walk:
                    doWalk();
                    break;
                case Glide:
                    doDrive();
                    break;
                case Fly:
                    doFly();
                    break;
                default:
                    break;
                }
                dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
                dcs_mat = vsg::translate(-AnalogX * driveSpeed, -AnalogY * driveSpeed, 0.0) * dcs_mat;
                vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
				vvCollaboration::instance()->SyncXform();
            }
        }
    }

    if (navMode == Walk)
    {
        doWalkMoveToFloor();
    }

    // collision detection & correction
    // we ignore collision until the first movement because active collision during loading of geometry can cause problems
    if (collision && !ignoreCollision && ((getMode() == Walk) || (getMode() == Glide) || (getMode() == Fly)))
    {

        int i = 0;
        vsg::vec3 glideVec;
        while (i < 4 && avoidCollision(glideVec))
        {

            dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;

            vsg::dmat4 tmp;
            tmp= vsg::translate(glideVec[0], glideVec[1], glideVec[2]);
            dcs_mat = tmp * dcs_mat;;

            vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);

            i++;
        }
    }
    else
    {
        jump = true;
    }

    if (navMode == ShowName || navMode == SelectInteract)
        highlightSelectedNode(vv->getIntersectedNode());

    if (vvConfig::instance()->isMenuModeOn())
    {
        if (button->wasReleased(vruiButtons::MENU_BUTTON))
        {
            setMenuMode(true);
            //menuCallback(vvNavigationManager::instance(), NULL);
        }
    }

    if ((button->getState() == vruiButtons::MENU_BUTTON) && (menuButtonQuitInterval >= 0.0))
    {
        if (button->wasPressed(vruiButtons::MENU_BUTTON))
        {
            menuButtonStartTime = vv->currentTime();
        }
        else if (menuButtonStartTime + menuButtonQuitInterval < vv->currentTime())
        {
            vvVIVE::instance()->requestQuit();
        }
    }

    if (getMode() == TraverseInteractors)
    {
        if (button->wasReleased(vruiButtons::INTER_NEXT))
            vvIntersectionInteractorManager::the()->cycleThroughInteractors();
        else if (button->wasReleased(vruiButtons::INTER_PREV))
            vvIntersectionInteractorManager::the()->cycleThroughInteractors(false);
    }

    for (std::vector<vvMeasurement *>::iterator it = measurements.begin(); it != measurements.end(); it++)
    {
        (*it)->preFrame();
    }
}

void vvNavigationManager::setNavMode(std::string modeName)
{

    if (modeName == "")
    {
        // do nothing
    }
    else if (boost::iequals(modeName,"NavNone") || boost::iequals(modeName, "None"))
    {
        setNavMode(vvNavigationManager::NavNone);
    }
    else if (boost::iequals(modeName, "XForm") || boost::iequals(modeName, "EXAMINE"))
    {
        setNavMode(vvNavigationManager::XForm);
    }
    else if (boost::iequals(modeName, "Scale"))
    {
        setNavMode(vvNavigationManager::Scale);
    }
    else if (boost::iequals(modeName, "Walk") || boost::iequals(modeName, "ANY"))
    {
        setNavMode(vvNavigationManager::Walk);
    }
    else if (boost::iequals(modeName, "Drive"))
    {
        setNavMode(vvNavigationManager::Glide);
    }
    else if (boost::iequals(modeName, "Fly"))
    {
        setNavMode(vvNavigationManager::Fly);
    }
    else
    {
        for (const auto& n : navigationProviders)
        {
            if (n->getName() == modeName)
            {
                setNavMode((NavMode)n->ID);
                return;
            }
        }
        std::cerr << "VRCoviseConnection: navigation mode " << modeName << " not implemented" << std::endl;
    }
}

void vvNavigationManager::setNavMode(NavMode mode, bool updateGroup)
{
    //std::cerr << "navMode = " << mode << ", was " <<  navMode << ", old = " << oldNavMode << std::endl;

    if (navMode != NavNone)
        oldNavMode = navMode;

    if (navMode == ShowName || mode == ShowName)
    {
        toggleShowName(mode == ShowName);
    }
    if (navMode == TraverseInteractors || mode == TraverseInteractors)
    {
        toggleInteractors(mode == TraverseInteractors);
    }
    if (navMode == Menu || mode == Menu)
    {
        setMenuMode(mode == Menu);
    }
    if (navMode == Select || mode == Select)
    {
        vvSelectionManager::instance()->setSelectionOnOff(mode == Select);
    }
    if (navMode == SelectInteract || mode == SelectInteract)
    {
        toggleSelectInteract(mode == SelectInteract);
    }

    navMode = mode;
        for (const auto& n : navigationProviders)
        {
                if(n->isEnabled())
                   n->setEnabled(false);
        }

    switch (mode)
    {
    case XForm:
        interactionA->setName("Xform");
        if (xformButton_)
            xformButton_->setState(true, updateGroup);
        break;
    case XFormRotate:
        interactionA->setName("XformRotate");
        if (xformRotButton_)
            xformRotButton_->setState(true, updateGroup);
        break;
    case XFormTranslate:
        interactionA->setName("XformTranslate");
        if (xformTransButton_)
            xformTransButton_->setState(true, updateGroup);
        break;
    case Scale:
        interactionA->setName("Scale");
        if (scaleButton_)
            scaleButton_->setState(true, updateGroup);
        break;
    case Fly:
        interactionA->setName("Fly");
        if (flyButton_)
            flyButton_->setState(true, updateGroup);
        break;
    case Walk:
        interactionA->setName("Walk");
        if (walkButton_)
            walkButton_->setState(true, updateGroup);
        break;
    case Glide:
        interactionA->setName("Drive");
        if (driveButton_)
            driveButton_->setState(true, updateGroup);
        break;
    case ShowName:
        interactionA->setName("ShowName");
        if (showNameButton_)
            showNameButton_->setState(true, updateGroup);
        break;
    case TraverseInteractors:
        interactionA->setName("TraverseInteractors");
        if (traverseInteractorButton_)
            traverseInteractorButton_->setState(true, updateGroup);
        break;
    case Menu:
        interactionA->setName("Menu");
        interactionMenu->setName("Menu");
        break;
    case NavNone:
        interactionA->setName("NoNavigation");
        if (noNavButton_)
            noNavButton_->setState(true, updateGroup);
        break;
    case Measure:
        interactionA->setName("Measure");
        if (measureButton_)
            measureButton_->setState(true, updateGroup);
        break;
    case Select:
        interactionA->setName("Select");
        if (selectButton_)
            selectButton_->setState(true, updateGroup);
        break;
    case SelectInteract:
        interactionA->setName("SelectInteract");
        if (selectInteractButton_)
            selectInteractButton_->setState(true, updateGroup);
        break;
    default:
        bool found = false;
        for (const auto& n : navigationProviders)
        {
            if (n->ID == mode)
            {
                interactionA->setName(n->getName());
                if (n->navMenuButton)
                    n->navMenuButton->setState(true, updateGroup);
                n->setEnabled(true);
                found = true;
            }
        }
        if (!found && navMode != NavOther)
            fprintf(stderr, "vvNavigationManager::setNavMode: unknown mode %d\n", (int)navMode);
        break;
    }

    if (mode == NavNone)
    {
        if (interactionA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionMA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(interactionMA);
        }
        if (interactionMenu->isRegistered())
            coInteractionManager::the()->unregisterInteraction(interactionMenu);
        if (interactionRel->isRegistered())
            coInteractionManager::the()->unregisterInteraction(interactionRel);
    }
    else if (mode == Menu)
    {
        coInteractionManager::the()->registerInteraction(interactionMenu);
    }
    else
    {
        if (!interactionA->isRegistered())
        {
            if (!vvConfig::instance()->mouseTracking())
            {
                coInteractionManager::the()->registerInteraction(interactionA);
            }
        }
        if (interactionA->isRegistered())
        {
            if (vvConfig::instance()->mouseTracking())
            {
                coInteractionManager::the()->unregisterInteraction(interactionA);
            }
        }
        if (!interactionMA->isRegistered())
        {
            if (doMouseNav)
            {
                coInteractionManager::the()->registerInteraction(interactionMA);
            }
        }
        if (interactionMenu->isRegistered())
            coInteractionManager::the()->unregisterInteraction(interactionMenu);
        if (!interactionRel->isRegistered())
            coInteractionManager::the()->registerInteraction(interactionRel);
    }

    if (vvConfig::instance()->isMenuModeOn() && oldNavMode == Menu && mode != Menu)
        vvSceneGraph::instance()->setMenuMode(false);

    if (navMode != Measure && oldNavMode == Measure)
    {
        for (std::vector<vvMeasurement *>::iterator it = measurements.begin(); it != measurements.end(); it++)
        {
            delete (*it);
        }
        measurements.clear();
    }
}

ui::ButtonGroup *vvNavigationManager::navGroup() const
{
    return navGroup_;
}

void vvNavigationManager::toggleCollide(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::toggleCollide %d\n", state);

    collision = state;
    collisionButton_->setState(collision);
}

void vvNavigationManager::toggleShowName(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::toggleShowName %d\n", state);

    if (state)
    {
        // enable intersection with scene
        vvIntersection::instance()->isectAllNodes(true);
    }
    else
    {
        if (vv->debugLevel(4))
            fprintf(stderr, "realtoggle\n");
        nameLabel_->hide();
        nameMenu_->setVisible(false);
        highlightSelectedNode(NULL);
        // disable intersection with scene
        vvIntersection::instance()->isectAllNodes(false);
    }
}

void vvNavigationManager::toggleInteractors(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::toggleInteractors %d\n", state);

    if (state)
    {
        vvIntersectionInteractorManager::the()->enableCycleThroughInteractors();
    }
    else if (navMode == TraverseInteractors || (oldNavMode == TraverseInteractors && navMode == Menu))
    {
        vvIntersectionInteractorManager::the()->disableCycleThroughInteractors();
    }
}

void vvNavigationManager::setMenuMode(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::seMenuMode(%d) old=%d\n", state, oldNavMode);

    if (!vvConfig::instance()->isMenuModeOn())
        return;

    if (state)
    {
        vvSceneGraph::instance()->setMenuMode(true);
    }
    else
    {
        vvSceneGraph::instance()->setMenuMode(false);

    } /* else if (!state)
   {
      vvSceneGraph::instance()->setMenuMode(true);
      vvSceneGraph::instance()->setMenuMode(false);
   }*/
}

float vvNavigationManager::getPhiZVerti(float y2, float y1, float x2, float widthX, float widthY)
{
    float phiZMax = 180.0;
    float y = (y2 - y1);
    float phiZ =(float) (4 * pow(x2, 2) * phiZMax * y / (pow(widthX, 2) * widthY));
    if (x2 < 0.0)
        return (phiZ);
    else
        return (-phiZ);
}

float vvNavigationManager::getPhiZHori(float x2, float x1, float y2, float widthY, float widthX)
{
    float phiZMax = 180.0;
    float x = x2 - x1;
    float phiZ = (float)(4 * pow(y2, 2) * phiZMax * x / (pow(widthY, 2) * widthX));

    if (y2 > 0.0)
        return (phiZ);
    else
        return (-phiZ);
}

float vvNavigationManager::getPhi(float relCoord1, float width1)
{
    float phi, phiMax = 180.0;
    phi = phiMax * relCoord1 / width1;
    return (phi);
}

void vvNavigationManager::makeRotate(float heading, float pitch, float roll, int headingBool, int pitchBool, int rollBool)
{
    vsg::dmat4 actRot = vvSceneGraph::instance()->getTransform()->matrix;
    vsg::dmat4 finalRot;

    if (turntable)
    {
        vsg::dmat4 newHeadingRot;
        vsg::dmat4 newPitchRot;

        newHeadingRot = makeEulerMat((double)headingBool * heading, 0.0, 0.0);
        newPitchRot = makeEulerMat(0.0, (double)pitchBool * pitch, 0.0);

        vsg::dvec3 rotAxis(0.0, 0.0, 0.0);
        rotAxis = rotAxis * vsg::inverse(actRot); // transform axis into local coordinates
        newHeadingRot = vsg::translate(-rotAxis) * newHeadingRot * vsg::translate(rotAxis); // rotate around the axis

        finalRot = newHeadingRot * actRot * newPitchRot; // apply headingRot in local coordinates (we want to rotate according to the objects rotated up vector)
    }
    else
    {
        vsg::dmat4 newRot;
        newRot = makeEulerMat((double)headingBool * heading, (double)pitchBool * pitch, (double)rollBool * roll / 7.0);
        finalRot = newRot * actRot;
    }

    vvSceneGraph::instance()->getTransform()->matrix = (finalRot);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::doMouseFly()
{
    vsg::vec3 velDir;

    vsg::dmat4 dcs_mat;
    dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    double heading = 0.0;
    double pitch = (my - y0) / -300;
    double roll = (mx - x0) / -300;
    vsg::dmat4 rot;
    rot = makeEulerMat(heading, pitch, roll);
    dcs_mat = rot * dcs_mat;
    velDir[0] = 0.0;
    velDir[1] = -1.0;
    velDir[2] = 0.0;
    vsg::dmat4 tmp;
    tmp= vsg::translate(velDir[0] * currentVelocity, velDir[1] * currentVelocity, velDir[2] * currentVelocity);
    dcs_mat = tmp * dcs_mat;;
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::doMouseXform()
{
    vsg::dmat4 dcs_mat;
    float widthX = mouseWinWidth(), widthY = mouseWinHeight();
    //Rotation funktioniert
    if ((navMode==XFormRotate && (interactionMA->isRunning() || interactionMB->isRunning() || interactionMC->isRunning()))
            || (navMode==XForm &&
                ((interactionMA->isRunning() && (mouseNavButtonRotate == 0))
                 || (interactionMC->isRunning() && (mouseNavButtonRotate == 1))
                 || (interactionMB->isRunning() && (mouseNavButtonRotate == 2)))))
    {

        if (!shiftMouseNav && !isViewerPosRotation) //Rotation um Weltursprung funktioniert
        {
           /* vv->setCurrentCursor(osgViewer::GraphicsWindow::CycleCursor);*/

            float newx, newy; //newz;//relz0
            float heading, pitch, roll, rollVerti, rollHori;

            newx = mouseX() - originX;
            newy = mouseY() - originY;

            //getPhi kann man noch aendern, wenn xy-Rotation nicht gewichtet werden soll
            heading = getPhi(newx - relx0, widthX);
            pitch = getPhi(newy - rely0, widthY);

            rollVerti = getPhiZVerti(newy, rely0, newx, widthX, widthY);
            rollHori = getPhiZHori(newx, relx0, newy, widthY, widthX);
            roll = (rollVerti + rollHori);

            makeRotate(heading, pitch, roll, 1, 1, 1);

            relx0 = mouseX() - originX;
            rely0 = mouseY() - originY;
        }
        else if (shiftMouseNav || isViewerPosRotation) //Rotation um beliebigen Punkt funktioniert
        {
            /*vv->setCurrentCursor(osgViewer::GraphicsWindow::CycleCursor);*/

            vsg::dmat4 doTrans, rot, doRot, doRotObj;
            dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
            //Rotation um beliebigen Punkt -> Mauszeiger, Viewer-Position, ...
            doTrans= vsg::translate(-transRel);
            doRotObj = doTrans * dcs_mat;
            vvSceneGraph::instance()->getTransform()->matrix = (doRotObj);
            dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;

            float newx, newy; //newz;//relz0
            float heading, pitch, roll, rollVerti, rollHori;

            newx = mouseX() - originX;
            newy = mouseY() - originY;

            //getPhi kann man noch aendern, wenn xy-Rotation nicht gewichtet werden soll
            heading = getPhi(newx - relx0, widthX);
            pitch = getPhi(newy - rely0, widthY);

            rollVerti = getPhiZVerti(newy, rely0, newx, widthX, widthY);
            rollHori = getPhiZHori(newx, relx0, newy, widthY, widthX);
            roll = rollVerti + rollHori;

            if (turntable)
            {
                vsg::dmat4 newHeadingRot;
                vsg::dmat4 newPitchRot;

                newHeadingRot = makeEulerMat((double)heading, 0.0, 0.0);
                newPitchRot = makeEulerMat( 0.0, (double)pitch, 0.0);

                vsg::dvec3 rotAxis(0.0, 0.0, 0.0);
                rotAxis = rotAxis * vsg::inverse(dcs_mat); // transform axis into local coordinates
                newHeadingRot = vsg::translate(rotAxis) * newHeadingRot * vsg::translate(-rotAxis); // rotate around the axis

                doRot = newHeadingRot * dcs_mat * newPitchRot; // apply headingRot in local coordinates (we want to rotate according to the objects rotated up vector)
            }
            else
            {

                rot = makeEulerMat( (double)heading, (double)pitch, (double)roll);
                doRot = rot * dcs_mat;
            }

            relx0 = mouseX() - originX;
            rely0 = mouseY() - originY;
            doTrans= vsg::translate(transRel);
            doRotObj = doTrans * doRot;
            vvSceneGraph::instance()->getTransform()->matrix = (doRotObj);
        }
        vvCollaboration::instance()->SyncXform();
    }

    //Translation funktioniert
    if ((navMode==XFormTranslate && (interactionMA->isRunning() || interactionMB->isRunning() || interactionMC->isRunning()))
            || (navMode==XForm && ((interactionMA->isRunning() && (mouseNavButtonTranslate == 0))
                    || (interactionMC->isRunning() && (mouseNavButtonTranslate == 1))
                    || (interactionMB->isRunning() && (mouseNavButtonTranslate == 2)))))
    {
        //irgendwas einbauen, damit die folgenden Anweisungen vor der
        //naechsten if-Schleife nicht unnoetigerweise dauernd ausgefuehrt werden

        //Translation in Bildschirmebene//letzte Ueberpruefung auch mal weglassen
        if (!shiftMouseNav /* && (yValObject >= yValViewer) */)
        {
            /*vv->setCurrentCursor(osgViewer::GraphicsWindow::HandCursor);*/

            float newxTrans = mouseX() - originX;
            float newyTrans = mouseY() - originY;
            float xTrans, yTrans;
            xTrans = (newxTrans - relx0) * modifiedHSize / widthX;
            yTrans = (newyTrans - rely0) * modifiedVSize / widthY;
            vsg::dmat4 actTransState = mat0;
            vsg::dmat4 trans;
            vsg::dmat4 doTrans;
            trans= vsg::translate(1.1 * xTrans, 0.0, 1.1 * yTrans);
            //die 1.1 sind geschaetzt, um die Ungenauigkeit des errechneten Faktors auszugleichen
            //es sieht aber nicht so aus, als wuerde man es auf diese
            //Weise perfekt hinbekommen
            doTrans = trans * actTransState;
            vvSceneGraph::instance()->getTransform()->matrix = (doTrans);
        }

        else if (shiftMouseNav) //Translation in der Tiefe
        {
            /*vv->setCurrentCursor(osgViewer::GraphicsWindow::UpDownCursor);*/

            float newzTrans = mouseY() - originY;
            float zTrans;
            zTrans = (newzTrans - rely0) * mouseScreenHeight() * 2 / widthY;
            vsg::dmat4 actTransState = mat0;
            vsg::dmat4 trans;
            vsg::dmat4 doTrans;
            trans= vsg::translate(0.0, 2.0 * zTrans, 0.0);
            doTrans = trans * actTransState;
            vvSceneGraph::instance()->getTransform()->matrix = (doTrans);
            vvCollaboration::instance()->SyncXform();
        }
        vvCollaboration::instance()->SyncXform();
    }
    if (navMode==XForm &&
            ((interactionMA->isRunning() && (mouseNavButtonScale == 0))
             || (interactionMC->isRunning() && (mouseNavButtonScale == 1))
             || (interactionMB->isRunning() && (mouseNavButtonScale == 2))))
    {
        if (navMode==XForm)
        {
            doMouseScale();
        }
    }
}

void vvNavigationManager::doMouseScale(float newScaleFactor)
{
    //getting the old translation and adjust it according to the new scale factor
    vsg::dvec3 center = mouseNavCenter;
    //calculate the new translation matrix
    //and move the objects by the new translation
    vsg::dvec3 delta2 = getTrans(mat0);
    vsg::dvec3 delta = delta2 - center;

    delta *= (newScaleFactor / actScaleFactor);
    delta += center;
    delta -= delta2;
    vsg::dmat4 tmp;
    tmp= vsg::translate(delta[0], delta[1], delta[2]);
    vsg::dmat4 xform_mat = mat0 * tmp;

    vvSceneGraph::instance()->getTransform()->matrix = (xform_mat);
    vvSceneGraph::instance()->setScaleFactor(newScaleFactor);
    
}

void vvNavigationManager::doMouseScale()
{
    /*vv->setCurrentCursor(osgViewer::GraphicsWindow::UpDownCursor);*/
    float ampl = mouseWinHeight()/2;
    float newScaleY = mouseY() - originY;

    //calculate the new scale factor
    float newScaleFactor = actScaleFactor * exp(((rely0 - newScaleY) / -ampl));
    doMouseScale(newScaleFactor);
}

void vvNavigationManager::doMouseWalk()
{
    vsg::dvec3 velDir;
    vsg::dmat4 dcs_mat;
    dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    vsg::dmat4 tmp;
    vsg::dmat4 tmp2;

    if (interactionMA->wasStarted())
    {
        tmp= vsg::translate((double)(mx - x0) * driveSpeed * -1, 0.0, (double)(my - y0) * driveSpeed);
        dcs_mat = tmp * dcs_mat;;
    }
    else if (interactionMA->isRunning())
    {
        float angle = (mx - x0) / 300;
        tmp = vsg::rotate(angle * M_PI / 180, 0.0, 0.0, 1.0);
        vsg::dvec3 viewerPos = getTrans(vv->getViewerMat());
        tmp2= vsg::translate(viewerPos);
        tmp = tmp2 * tmp;
        tmp2= vsg::translate(-viewerPos);
        tmp = tmp * tmp2;

        dcs_mat = tmp * dcs_mat;;
        velDir[0] = 0.0;
        velDir[1] = 1.0;
        velDir[2] = 0.0;
        currentVelocity = (float)((my - y0) * driveSpeed * 0.5f);
        tmp= vsg::translate(velDir[0] * currentVelocity, velDir[1] * currentVelocity, velDir[2] * currentVelocity);
        dcs_mat = tmp * dcs_mat;;
    }
    else if (interactionMB->wasStarted())
    { // pan
        tmp= vsg::translate((double)((mx - x0) * driveSpeed * -1), 0.0, (double)(my - y0) * driveSpeed * -1);
        dcs_mat = tmp * dcs_mat;;
    }
    else if (interactionMB->isRunning())
    { // pan
        tmp= vsg::translate((double)((mx - x0) * driveSpeed * -1), 0.0, (double)(my - y0) * driveSpeed * -1);
        dcs_mat = tmp * dcs_mat;;
    }
#if 0
   fprintf(stderr, "buttons=0x%x, mouseNav=%d, handLocked=%d\n",
         vv->getMouseButton(), (int)doMouseNav, handLocked);
#endif
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::stopMouseNav()
{
    /*vv->setCurrentCursor(osgViewer::GraphicsWindow::LeftArrowCursor);*/

    actScaleFactor = vv->getScale();
    x0 = mx;
    y0 = my;
    currentVelocity = 10;
    relx0 = x0 - originX; //relativ zum Ursprung des Koordinatensystems
    rely0 = y0 - originY; //dito
    vvCollaboration::instance()->UnSyncXform();
}

void vvNavigationManager::startMouseNav()
{
    shiftMouseNav = shiftEnabled;

    vsg::dmat4 dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    mat0 = dcs_mat;
    mouseNavCenter = getCenter();

    actScaleFactor = vv->getScale();
    x0 = mx;
    y0 = my;
    //cerr << "mouseNav" << endl;
    currentVelocity = 10;

    relx0 = x0 - originX; //relativ zum Ursprung des Koordinatensystems
    rely0 = y0 - originY; //dito

    vsg::dmat4 whereIsViewer;
    whereIsViewer = vv->getViewerMat();
    double yValViewer = whereIsViewer(3, 1);
    double yValObject = dcs_mat(3, 1); //was gibt dcs_mat genau an ?? gibt es eine besser geeignete
    double alphaY = fabs(atan(mouseScreenHeight() / (2.0 * yValViewer)));
    modifiedVSize = (float)(2.0f * tan(alphaY) * fabs(yValObject - yValViewer));
    double alphaX = fabs(atan(mouseScreenWidth() / (2.0 * yValViewer)));
    modifiedHSize = (float)(2.0f * tan(alphaX) * fabs(yValObject - yValViewer));
    //float newxTrans = relx0;  //declared but never referenced
    //float newyTrans = rely0;  //dito


    if (isViewerPosRotation)
        transRel = getTrans(vv->getViewerMat());
    else
        transRel = vv->getIntersectionHitPointWorld();
}

void vvNavigationManager::startXform()
{
    old_mat = inverse(handMat);
    old_dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    if (!rotationPoint)
    {
      /*  transformVec = vv->getBBox(vvSceneGraph::instance()->getTransform()).center();
        if (coCoviseConfig::isOn("VIVE.ScaleWithInteractors", false))
            transformVec = vv->getBBox(vvSceneGraph::instance()->getScaleTransform()).center();
        //fprintf(stderr, "****set TransformVec to BBOx-Center %f %f %f\n",transformVec[0], transformVec[1], transformVec[2]);*/
    }
    else
    {
        transformVec = vv->getBaseMat() * rotPointVec;
        //fprintf(stderr, "****set TransformVec to %f %f %f\n",transformVec[0], transformVec[1], transformVec[2]);
    }
}

void vvNavigationManager::doXform()
{
    if (wiiNav)
    {
        // do xformRotate oder translate
        if (wiiFlag != 2 && handPos - oldHandPos == vsg::dvec3(0, 0, 0))
        {
            wiiFlag = 1;
            return doXformRotate();
        }
        else
        {
            wiiFlag = 2;
            return doXformTranslate();
        }
    }
    vsg::dmat4 rel_mat, dcs_mat;
    rel_mat = handMat * old_mat; //erste handMat * aktualisierte handMat
    dcs_mat = rel_mat * old_dcs_mat;
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);

    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::doXformRotate()
{
    vsg::dmat4 rel_mat, dcs_mat;
    vsg::dmat4 rev_mat, trans_mat;
    // zurueck in ursprung verschieben
    trans_mat= vsg::translate(-transformVec);
    rev_mat = trans_mat * old_dcs_mat;
    // erste handMat * aktualisierte handMat
    vsg::dmat4 oh = old_mat;
    setTrans(oh, vsg::dvec3(0.0, 0.0, 0.0));
    vsg::dmat4 hm = handMat;
    setTrans(hm, vsg::dvec3(0.0, 0.0, 0.0));
    rel_mat = hm * oh;
    //vsg::Quat qrot = rel_mat.getRotate();
    // nur rotieren nicht transformieren
    setTrans(rel_mat, vsg::dvec3(0.0, 0.0, 0.0));
    dcs_mat = rel_mat * rev_mat;
    // zurueck verschieben
    trans_mat= vsg::translate(transformVec);
    dcs_mat = trans_mat * dcs_mat;
    // setzen der transformationsmatrix
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::doXformTranslate()
{
    vsg::dmat4 rel_mat, dcs_mat;
    rel_mat = handMat * old_mat;//erste handMat * aktualisierte handMat
    vsg::dvec3 t = getTrans(rel_mat);
    rel_mat= vsg::translate(t[0], t[1] / 5.0, t[2]);
    dcs_mat = rel_mat * old_dcs_mat;
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::stopXform()
{
    vvCollaboration::instance()->UnSyncXform();
    if (wiiNav)
        wiiFlag = 0;
}

void vvNavigationManager::startScale()
{
    // save start position
    startHandPos = handPos;
    startHandDir = handDir;

    oldDcsScaleFactor = vvSceneGraph::instance()->scaleFactor();

    old_xform_mat = vvSceneGraph::instance()->getTransform()->matrix;
}

void vvNavigationManager::doScale()
{

    double d = handMat(3, 0) - startHandPos[0];

    double dcsScaleFactor = 1.;
    if (d > 0)
        dcsScaleFactor = oldDcsScaleFactor * (1 + (10 * d / vv->getSceneSize()));
    else
        dcsScaleFactor = oldDcsScaleFactor / (1 + (10 * -d / vv->getSceneSize()));
    if (dcsScaleFactor < 0.0f)
        dcsScaleFactor = 0.1;

    /* move XformDCS to keep click position fixed */
    vsg::dvec3 delta2 = getTrans(old_xform_mat);
    vsg::dvec3 delta = delta2 - startHandPos;
    vsg::dvec3 rot;
    if (rotationPoint)
    {
        rot = rotPointVec + getTrans(vv->getBaseMat());
        delta = delta2 - rot;
    }

    delta *= (dcsScaleFactor / oldDcsScaleFactor);
    if (rotationPoint)
        delta += rot;
    else
        delta += startHandPos;
    delta -= delta2;
    vsg::dmat4 tmp;
    tmp= vsg::translate(delta[0], delta[1], delta[2]);
    vsg::dmat4 xform_mat = old_xform_mat * tmp;

    vvSceneGraph::instance()->getTransform()->matrix = (xform_mat);
    vvSceneGraph::instance()->setScaleFactor((float)dcsScaleFactor);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::stopScale()
{
    vvCollaboration::instance()->UnSyncXform();
}

void vvNavigationManager::startFly()
{
    old_mat = handMat;
    startHandPos = handPos;
    startHandDir = handDir;
    ignoreCollision = false;
}

void vvNavigationManager::doFly()
{
    vsg::dvec3 dirAxis, handRight, oldHandRight, rollAxis;
    double dirAngle, rollAngle, rollScale;
    vsg::dmat4 dcs_mat, rot_mat;
    vsg::dvec3 delta = startHandPos - handPos;

    /* calc axis and angle for direction change (from hand direction change) */
    dirAxis = cross(handDir,startHandDir);
    dirAngle = length(dirAxis) * vv->frameDuration() * rotationSpeed;

    /* calc axis and angle for roll change (from hand roll change) */
    handRight.set(handMat(0, 0), handMat(0, 1), handMat(0, 2));
    oldHandRight.set(old_mat(0, 0), old_mat(0, 1), old_mat(0, 2));
    rollAxis = vsg::cross(handRight , oldHandRight);
    rollScale = fabs(dot(rollAxis ,handDir));
    rollAxis = rollAxis * rollScale;
    rollAngle = length(rollAxis) * vv->frameDuration() * 50;

    /* get xform matrix, translate to make viewPos the origin */
    dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    vsg::dmat4 tmp;
    vsg::dvec3 viewerPos = getTrans(vv->getViewerMat());
    tmp= vsg::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2]);
    dcs_mat = tmp * dcs_mat;

    /* apply translation */
    dcs_mat = vsg::translate(applySpeedFactor(delta))* dcs_mat;

    /* apply direction change */
    if ((dirAxis[0] != 0.0) || (dirAxis[1] != 0.0) || (dirAxis[2] != 0.0))
    {
        rot_mat = vsg::rotate(dirAngle * 3 * M_PI / 180, dirAxis[0], dirAxis[1], dirAxis[2]);
        dcs_mat = rot_mat * dcs_mat;
    }

    /* apply roll change */
    if ((rollAxis[0] != 0.0) || (rollAxis[1] != 0.0) || (rollAxis[2] != 0.0))
    {
        rot_mat = vsg::rotate(rollAngle * M_PI / 180, rollAxis[0], rollAxis[1], rollAxis[2]);
        dcs_mat = rot_mat * dcs_mat;
    }

    /* undo viewPos translation, set new xform matrix */
    dcs_mat = vsg::translate(viewerPos[0], viewerPos[1], viewerPos[2]) * dcs_mat;
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::stopFly()
{
    vvCollaboration::instance()->UnSyncXform();
}

void vvNavigationManager::startDrive()
{
    old_mat = handMat;
    startHandPos = handPos;
    startHandDir = handDir;
    ignoreCollision = false;
}
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
double vvNavigationManager::speedFactor(double delta) const
{
    /*
   return pow(navExp, fmax(fabs(delta),0.0)/8.0)
       * delta * driveSpeed * vv->frameDuration();
       */
    return sign(delta) * pow((double)(fabs(delta) / 40.0), (double)navExp) * 40.0
           * driveSpeed * (vv->frameDuration() > 0.05 ? 0.05 : vv->frameDuration());
}

vsg::dvec3 vvNavigationManager::applySpeedFactor(vsg::dvec3 vec) const
{
    double l = length(vec);
    double s = speedFactor(l);
    normalize(vec);
    return vec * s;
}

void vvNavigationManager::doDrive()
{
    /* calc delta vector for translation (hand position relative to click point) */
    vsg::dvec3 delta = startHandPos - handPos;

    /* calc axis and angle for direction change (from hand direction change) */

    vsg::dvec3 tmpv1 = handDir;
    vsg::dvec3 tmpv2 = startHandDir;
    tmpv1[2] = 0.0;
    tmpv2[2] = 0.0;
    vsg::dvec3 dirAxis = cross(tmpv1 , tmpv2);
    double dirAngle = length(dirAxis) * vv->frameDuration() * rotationSpeed;

    /* get xform matrix, translate to make viewPos the origin */
    vsg::dmat4 dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    vsg::dvec3 viewerPos = getTrans(vv->getViewerMat());
    dcs_mat = vsg::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2])* dcs_mat;

    /* apply translation */
    dcs_mat = vsg::translate(applySpeedFactor(delta))*dcs_mat;

    /* apply direction change */
    vsg::dmat4 rot_mat;
    if ((dirAxis[0] != 0.0) || (dirAxis[1] != 0.0) || (dirAxis[2] != 0.0))
    {
        rot_mat = vsg::rotate(dirAngle * 3 * M_PI / 180, dirAxis[0], dirAxis[1], dirAxis[2]);
        dcs_mat = rot_mat *dcs_mat;
    }

    /* undo handPos translation, set new xform matrix */
    vsg::dmat4 tmp;
    tmp= vsg::translate(viewerPos[0], viewerPos[1], viewerPos[2]);
    dcs_mat = tmp * dcs_mat;;
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);

    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::stopDrive()
{
    vvCollaboration::instance()->UnSyncXform();
}

void vvNavigationManager::startWalk()
{
    startDrive();
    ignoreCollision = false;
    old_mat = handMat;

    /* get xform matrix */
    vsg::dmat4 TransformMat = vvSceneGraph::instance()->getTransform()->matrix;

    vsg::dmat4 viewer = vv->getViewerMat();
    // snap 45 degrees in orientation
    coCoord coordOld = TransformMat;
    coCoord coord;
    coord.hpr = coordOld.hpr;
    coordOld.xyz[0] = 0;
    coordOld.xyz[1] = 0;
    coordOld.xyz[2] = 0;

    if (coord.hpr[1] > 0.0)
        coord.hpr[1] = 45.0 * ((int)(coord.hpr[1] + 22.5) / 45);
    else
        coord.hpr[1] = 45.0 * ((int)(coord.hpr[1] - 22.5) / 45);
    if (coord.hpr[2] > 0.0)
        coord.hpr[2] = 45.0 * ((int)(coord.hpr[2] + 22.5) / 45);
    else
        coord.hpr[2] = 45.0 * ((int)(coord.hpr[2] - 22.5) / 45);
    /* apply translation , so that world snaps to 45 degrees in pitch an roll */
    vsg::dmat4 oldRot, newRot;
    coordOld.makeMat(oldRot);
    coord.makeMat(newRot);
    vsg::dmat4 diffRot = newRot* inverse(oldRot);
    vsg::dvec3 viewerPos = getTrans(viewer);
    vsg::dmat4 vToOrigin;
    vToOrigin= vsg::translate(-viewerPos);
    vsg::dmat4 originToV;
    originToV= vsg::translate(viewerPos);

    /* set new xform matrix */
    vvSceneGraph::instance()->getTransform()->matrix = (TransformMat * vToOrigin * diffRot * originToV);
}

void vvNavigationManager::doWalk()
{
    doDrive();
}

void vvNavigationManager::doWalkMoveToFloor()
{
    float floorHeight = vvSceneGraph::instance()->floorHeight();

    //  just adjust height here

    vsg::dmat4 viewer = vv->getViewerMat();
    vsg::dvec3 pos = getTrans(viewer);

    // down segment
    vsg::dvec3 p0, q0;
    p0.set(pos[0], pos[1], floorHeight + stepSize);
    q0.set(pos[0], pos[1], floorHeight - stepSize);

    LineSegment ray[2];
    ray[0].start = p0;
    ray[0].end = q0;

    // down segment 2
    p0.set(pos[0], pos[1] + 10, floorHeight + stepSize);
    q0.set(pos[0], pos[1] + 10, floorHeight - stepSize);
    ray[1].start = p0;
    ray[1].end = q0;

    vsg::ref_ptr<vsg::LineSegmentIntersector> intersectors[2];
    for (int i=0; i<2; ++i)
    {
        intersectors[i] = vvIntersection::instance()->newIntersector(ray[i].start, ray[i].end);
    }

    vsg::Node* floorNode = NULL;

    double dist = FLT_MAX;
    //  get xform matrix
    vsg::dmat4 dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
 /*   osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Walk);
    vvSceneGraph::instance()->getTransform()->accept(visitor);

    bool haveIsect[2];
    for (int i=0; i<2; ++i)
        haveIsect[i] = intersectors[i]->containsIntersections();
    if (!haveIsect[0] && !haveIsect[1])
    {
        oldFloorNode = NULL;
        return;

    }

    osgUtil::LineSegmentIntersector::Intersection isect;
    if (haveIsect[0])
    {
        isect = intersectors[0]->getFirstIntersection();
        dist = isect.getWorldIntersectPoint()[2] - floorHeight;
        floorNode = isect.nodePath.back();
    }
    if (haveIsect[1] && fabs(intersectors[1]->getFirstIntersection().getWorldIntersectPoint()[2] - floorHeight) < fabs(dist))
    {
        isect = intersectors[1]->getFirstIntersection();
        dist = isect.getWorldIntersectPoint()[2] - floorHeight;
        floorNode = isect.nodePath.back();
    }


    if (floorNode && floorNode == oldFloorNode)
    {
        // we are walking on the same object as last time so move with the object if it is moving
        vsg::dmat4 modelTransform;
        int on = oldNodePath.size() - 1;
        bool notSamePath = false;
        for (int i = isect.nodePath.size() - 1; i >= 0; i--)
        {
            vsg::Node*n = isect.nodePath[i];
            if (n == vv->getObjectsRoot())
                break;
            vsg::MatrixTransform *t = dynamic_cast<vsg::MatrixTransform *>(n);
            if (t != NULL)
            {
                modelTransform = modelTransform * t->matrix;
            }
            // check if this is really the same object as it could be a reused object thus compare the whole NodePath
            // instead of just the last node
            if (on < 0 || n != oldNodePath[on])
            {
                //oops, not same path
                notSamePath = true;
            }
            on--;
        }
        if (notSamePath)
        {
            oldFloorMatrix = modelTransform;
            oldFloorNode = floorNode;
            oldNodePath = isect.nodePath;
        }
        else if (modelTransform != oldFloorMatrix)
        {

            //vsg::dmat4 iT;
            vsg::dmat4 iS;
            vsg::dmat4 S;
            vsg::dmat4 imT;
            //iT.invert_4x4(dcs_mat);
            float sf = vv->getScale();
            S= vsg::scale(sf, sf, sf);
            sf = 1.0 / sf;
            iS= vsg::scale(sf, sf, sf);
            imT = inverse(modelTransform);
            dcs_mat = iS *imT*oldFloorMatrix * S * dcs_mat;
            oldFloorMatrix = modelTransform;
            // set new xform matrix
            vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
            // now we have a new base matrix and we have to compute the floor height again, otherwise we will jump up and down
            //
            vvSceneGraph::instance()->getTransform()->accept(visitor);

            osgUtil::IntersectionVisitor visitor(igroup);
            igroup->reset();
            visitor.setTraversalMask(Isect::Walk);
            vvSceneGraph::instance()->getTransform()->accept(visitor);

            for (int i=0; i<2; ++i)
                haveIsect[i] = intersectors[i]->containsIntersections();
            dist = FLT_MAX;
            if (haveIsect[0])
            {
                isect = intersectors[0]->getFirstIntersection();
                dist = isect.getWorldIntersectPoint()[2] - floorHeight;
                floorNode = isect.nodePath.back();
            }
            if (haveIsect[1] && fabs(intersectors[1]->getFirstIntersection().getWorldIntersectPoint()[2] - floorHeight) < fabs(dist))
            {
                isect = intersectors[1]->getFirstIntersection();
                dist = isect.getWorldIntersectPoint()[2] - floorHeight;
                floorNode = isect.nodePath.back();
            }
        }
    }*/


    //  apply translation , so that isectPt is at floorLevel
    vsg::dmat4 tmp;
    tmp= vsg::translate(0.0, 0.0, -dist);
    dcs_mat = tmp * dcs_mat;;

    // set new xform matrix
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);

   /* if ((floorNode != oldFloorNode) && !isect.nodePath.empty())
    {
        vsg::dmat4 modelTransform;
        for (int i = isect.nodePath.size() - 1; i >= 0; i--)
        {
            vsg::Node*n = isect.nodePath[i];
            if (n == vv->getObjectsRoot())
                break;
            vsg::MatrixTransform *t = dynamic_cast<vsg::MatrixTransform *>(n);
            if (t != NULL)
            {
                modelTransform = modelTransform * t->matrix;
            }
        }
        oldFloorMatrix = modelTransform;
        oldNodePath = isect.nodePath;
    }*/

    oldFloorNode = floorNode;

    // do not sync with remote, they will do the same
    // on their side SyncXform();

    // vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::stopWalk()
{
    stopDrive();
}

void vvNavigationManager::startShowName()
{
    //fprintf(stderr, "vvNavigationManager::startShowName\n");
}

void vvNavigationManager::highlightSelectedNode(vsg::Node *selectedNode)
{

    /*if (selectedNode != oldSelectedNode_)
    {
        vvSelectionManager::instance()->clearSelection();
        if (selectedNode && (selectedNode->getNumParents() > 0))
        {
            vvSelectionManager::instance()->setSelectionWire(0);
            vvSelectionManager::instance()->setSelectionColor(1, 0, 0);
            vvSelectionManager::instance()->addSelection(selectedNode->getParent(0), selectedNode);
            vvSelectionManager::instance()->pickedObjChanged();
        }
        oldSelectedNode_ = selectedNode;
    }*/
}

void vvNavigationManager::doShowName()
{
    // get the intersected node
    if (vv->getIntersectedNode())
    {
      /*  // position label with name
        nameLabel_->setPosition(vv->getIntersectionHitPointWorld());
        if (vv->getIntersectedNode() != oldShowNamesNode_)
        {
            // get the node name
            string nodeName;
            //char *labelName = NULL;

            // show only node names under objects root
            // so check if node is under objects root
            Node *currentNode = vv->getIntersectedNode();
            while (currentNode != NULL)
            {
                if (currentNode == vv->getObjectsRoot())
                {
                    if (showGeodeName_)
                    {
                        // first look for a node description beginning with _SCGR_
                        std::vector<std::string> dl = vv->getIntersectedNode()->getDescriptions();
                        for (size_t i = 0; i < dl.size(); i++)
                        {
                            std::string descr = dl[i];
                            if (descr.find("_SCGR_") != string::npos)
                            {
                                nodeName = dl[i];
                                //fprintf(stderr,"found description %s\n", nodeName.c_str());
                                break;
                            }
                        }
                        if (nodeName.empty())
                        { // if there is no description we take the node name
                            nodeName = vv->getIntersectedNode()->getName();
                            //fprintf(stderr,"taking the node name %s\n", nodeName.c_str());
                        }
                    }
                    else // show name of the dcs above
                    {
                        vsg::Node *parentDcs = vv->getIntersectedNode()->getParent(0);
                        nodeName = parentDcs->getName();
                        if (nodeName.empty())
                        {
                            // if the dcs has no name it could be a helper node
                            if ((parentDcs->getNumDescriptions() > 0) && (parentDcs->getDescription(1) == "SELECTIONHELPER"))
                            {
                                nodeName = parentDcs->getParent(0)->getName();
                            }
                        }
                    }
                    break;
                }
                if (currentNode->getNumParents() > 0)
                    currentNode = currentNode->getParent(0);
                else
                    currentNode = NULL;
            }

            // show label
            if (!nodeName.empty())
            {
                std::string onam = nodeName;
                // eliminate _SCGR_
                if (nodeName.find("_SCGR_") != string::npos)
                {
                    nodeName = nodeName.substr(0, nodeName.rfind("_SCGR_"));
                }
                // eliminate 3D studio max name -faces
                if (nodeName.find("-FACES") != string::npos)
                {
                    nodeName = nodeName.substr(0, nodeName.rfind("-FACES"));
                }
                // eliminate 3D studio max name _faces
                if (nodeName.find("_FACES") != string::npos)
                {
                    nodeName = nodeName.substr(0, nodeName.rfind("_FACES"));
                }
                // eliminate numbers at the end
                while ((nodeName.length() > 0) && (nodeName[nodeName.length() - 1] >= '0') && (nodeName[nodeName.length() - 1] <= '9'))
                {
                    nodeName.erase(nodeName.length() - 1, 1);
                }

                // replace underlines with blanks
                if (nodeName.length() > 0 && nodeName[nodeName.length() - 1] == '_')
                {
                    nodeName.erase(nodeName.length() - 1);
                }

                // check if we now have an empty string (containing spaces only)
                if (nodeName.length() > 0)
                {
                    nodeName = vvTranslator::coTranslate(nodeName);

                    //now remove special names
                    

                    nameLabel_->setString(nodeName.c_str());
                    nameLabel_->show();

                    nameMenu_->setVisible(true);
                    nameButton_->setLabel(nodeName);

                    // delete []labelName;
                    oldShowNamesNode_ = vv->getIntersectedNode();
                }
                else
                {
                    nameLabel_->hide();
                    nameMenu_->setVisible(false);
                    nameButton_->setLabel("-");
                    oldShowNamesNode_ = NULL;
                }
            }
            else
            {
                nameLabel_->hide();
                nameMenu_->setVisible(false);
                nameButton_->setLabel("-");
                oldShowNamesNode_ = NULL;
            }
        }
        else
        {
            nameLabel_->show();
            nameMenu_->setVisible(true);
        }*/
    }
    else
    {
        oldShowNamesNode_ = NULL;
        nameLabel_->hide();
        nameMenu_->setVisible(false);
        nameButton_->setLabel("-");
    }
}

void vvNavigationManager::stopShowName()
{
    // remove label and highlight
    nameLabel_->hide();
    nameMenu_->setVisible(false);
}

void vvNavigationManager::startMeasure()
{
    if (interactionMC->wasStarted())
    {
        if (measurements.size() > 0)
        {
            delete measurements.back();
            measurements.pop_back();
        }
    }
    else
    {
        vvMeasurement *measurement = new vvMeasurement();
        measurement->start();
        measurements.push_back(measurement);
    }
}

void vvNavigationManager::doMeasure()
{
    if (interactionMC->isRunning())
    {
        return;
    }
    if (measurements.size() > 0)
    {
        measurements.back()->update();
    }
}

void vvNavigationManager::stopMeasure()
{
}

void vvNavigationManager::toggleSelectInteract(bool state)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvNavigationManager::toggleSelectInteract %d\n", state);

    if (state)
    {
        // enable intersection with scene
        vvIntersection::instance()->isectAllNodes(true);
    }
    else
    {
        if (vv->debugLevel(4))
            fprintf(stderr, "realtoggle\n");
        highlightSelectedNode(NULL);
        // disable intersection with scene
        vvIntersection::instance()->isectAllNodes(false);
    }
}

void vvNavigationManager::startSelectInteract()
{
    //fprintf(stderr, "vvNavigationManager::startSelectInteract\n");
    animationWasRunning = vvAnimationManager::instance()->animationRunning();
    vvAnimationManager::instance()->enableAnimation(false);
}

void vvNavigationManager::doSelectInteract()
{
}

void vvNavigationManager::stopSelectInteract(bool mouse)
{
    cerr << "TODO: implement it as visitor" << endl;
    /*vvAnimationManager::instance()->enableAnimation(animationWasRunning);

    if (!vv->getIntersectedNode())
        return;

    bool changeToInteraction = false;
    vsg::Node *currentNode = vv->getIntersectedNode();
    while (currentNode != NULL)
    {
        InteractorReference* ir = nullptr;
        if (currentNode->getValue("interactor", ir))
        {
            auto inter = ir->interactor();
            if (inter)
            {
                //std::cerr << "doSelectInteract: requesting interaction for node " << currentNode->getName() << std::endl;
                changeToInteraction = vvPluginList::instance()->requestInteraction(inter, vv->getIntersectedNode(), mouse);
                if (changeToInteraction)
                    break;
            }
        }
        if (currentNode == vv->getObjectsRoot())
        {
            break;
        }

        if (currentNode->getNumParents() > 0)
        {
            currentNode = currentNode->getParent(0);
        }
        else
        {
            currentNode = nullptr;
        }
    }

    if (changeToInteraction && mouse)
    {
        std::cerr << "doSelectInteract: Accepted interaction" << std::endl;
        setNavMode(oldNavMode);
    }*/
}

void vvNavigationManager::wasJumping()
{
    jump = true;
}

void vvNavigationManager::setDriveSpeed(float speed)
{
    driveSpeed = speed;
    driveSpeedSlider_->setValue(speed);
}

float vvNavigationManager::getDriveSpeed()
{
    return (float)driveSpeed;
}

bool vvNavigationManager::isSnapping() const
{
    return snapping;
}

void vvNavigationManager::enableSnapping(bool val)
{
    snapping = val;
}

bool vvNavigationManager::isDegreeSnapping() const
{
    return snappingD;
}

void vvNavigationManager::enableDegreeSnapping(bool val, float degrees)
{
    snapDegrees = degrees;
    snappingD = val;
}

bool vvNavigationManager::restrictOn() const
{
    return m_restrict;
}

float vvNavigationManager::snappingDegrees() const
{
    return (float)snapDegrees;
}

void vvNavigationManager::setStepSize(float stepsize)
{
    stepSize = stepsize;
}

float vvNavigationManager::getStepSize() const
{
    return (float)stepSize;
}

void vvNavigationManager::processHotKeys(int keymask)
{
    //printf("keymask = %d \n", keymask);
    /*  for(int i=0;i<MaxHotkeys;i++)
   {
      if(keymask & (1 << i))
      {
         if(keyMapping[i]=="ViewAll")
      }
      else
      {
      }
   }*/

    if ((keymask & 1024) > 0)
    {
        fprintf(stderr, "ViewAll");
        vvSceneGraph::instance()->viewAll(false);
    }
}

void vvNavigationManager::setRotationPoint(float x, float y, float z, float size)
{
    //fprintf(stderr, "vvNavigationManager::setRotationPoint x=%f y=%f z=%f rad=%f\n",x,y,z,size);

    rotationPoint = true;
    rotPointVec.set(x, y, z);
    vsg::dmat4 m;
    m= vsg::scale(size, size, size);
    setTrans(m,rotPointVec);
    rotPoint->matrix = (m);
}

void vvNavigationManager::setRotationPointVisible(bool visible)
{
    //fprintf(stderr, "vvNavigationManager::setRotationPointVisibl %d\n", visible);
    rotationPointVisible = visible;
    if (rotationPointVisible)
        vvSceneGraph::instance()->objectsRoot()->addChild(rotPoint);
    else
        vvPluginSupport::removeChild(vvSceneGraph::instance()->objectsRoot(),rotPoint);
}

void vvNavigationManager::setRotationAxis(float x, float y, float z)
{
    rotationAxis = true;
    rotationVec.set(x, y, z);
}

void vvNavigationManager::doGuiRotate(float x, float y, float z)
{
    vsg::dmat4 rel_mat, dcs_mat;
    vsg::dmat4 rev_mat, rotx, roty, rotz, trans_mat;
    double factor = 5.0;
    old_dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;
    // zurueck in rotationspunkt verschieben
    if (!rotationPoint && !rotationAxis)
    {
        /*transformVec = vv->getBBox(vvSceneGraph::instance()->getTransform()).center();
        if (coCoviseConfig::isOn("VIVE.ScaleWithInteractors", false))
            transformVec = vv->getBBox(vvSceneGraph::instance()->getScaleTransform()).center();*/
    }
    else
    {
        transformVec = rotPointVec * vv->getBaseMat();
    }
    if (!rotationAxis)
    {
        trans_mat= vsg::translate(-transformVec);
        rev_mat = trans_mat * old_dcs_mat;
        // rotationsmatrix
        rotx = vsg::rotate(x * factor, 1.0, 0.0, 0.0);
        roty = vsg::rotate(y * factor, 0.0, 1.0, 0.0);
        rotz = vsg::rotate(z * factor, 0.0, 0.0, 1.0);
        rel_mat = rotz * roty * rotx;
        dcs_mat = rel_mat * rev_mat;
        // zurueck verschieben
        trans_mat= vsg::translate(transformVec);
        dcs_mat = trans_mat * dcs_mat;
    }
    else
    {
        rel_mat = rotate(x * factor, rotationVec);
        dcs_mat = rel_mat * old_dcs_mat;
    }
    // setzen der transformationsmatrix
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
    vvCollaboration::instance()->SyncXform();
}

void vvNavigationManager::doGuiTranslate(float x, float y, float z)
{
    vsg::dmat4 dcs_mat;
    vsg::dmat4 doTrans;

    float sceneSize = vv->getSceneSize();
    float scale = 0.05f; // * sqrt(vvSceneGraph::instance()->getScaleTransform()->matrix.getScale()[0]);

    float translatefac = sceneSize * scale;
    if (guiTranslateFactor > 0)
        translatefac = translatefac * guiTranslateFactor;

    doTrans= vsg::translate(x * translatefac, y * translatefac, z * translatefac);
    dcs_mat = vvSceneGraph::instance()->getTransform()->matrix;

    dcs_mat = dcs_mat * doTrans;
    vvSceneGraph::instance()->getTransform()->matrix = (dcs_mat);
}
