/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996-200                              *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 *   File            coVRNavigationManager.cpp                          *
 *                                                                      *
 *   Description     scene graph class                                  *
 *                /                                                      *
 *   Author          D. Rainer                                          *
 *                   F. Foehl                                           *
 *                   U. Woessner                                        *
 *                                                                      *
 *   Date            20.08.97                                           *
 *                   10.07.98 Performer C++ Interface                   *
 *                   20.11.00 Pinboard config through covise.config     *
 ************************************************************************/

#include <math.h>
#include <cstdlib>

#include <util/common.h>
#include <config/CoviseConfig.h>

#include "OpenCOVER.h"
#include "coInteractor.h"
#include "coTranslator.h"
#include "VRSceneGraph.h"
#include "coVRCollaboration.h"
#include "coVRNavigationManager.h"
#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include "coVRMSController.h"
#include "coVRCommunication.h"
#include "coVRPluginList.h"
#include "VRViewer.h"
#include <osgGA/GUIActionAdapter>
#include "coVRLabel.h"
#include "coVRSelectionManager.h"
#include "coIntersection.h"
#include <input/input.h>
#include <input/coMousePointer.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMouseButtonInteraction.h>
#include <OpenVRUI/coRelativeInputInteraction.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include "coVRAnimationManager.h"
#include <ui/Menu.h>
#include <ui/Action.h>
#include <ui/Button.h>
#include <ui/ButtonGroup.h>
#include <ui/Slider.h>
#include "coVRIntersectionInteractorManager.h"
#include "coMeasurement.h"

#include <osg/Vec4>
#include <osg/LineSegment>
#include <osgUtil/IntersectVisitor>
#include <osgDB/WriteFile>
#include <osg/ShapeDrawable>
#include <osg/io_utils>


using namespace osg;
using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

coVRNavigationManager *coVRNavigationManager::s_instance = NULL;

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

coVRNavigationManager *coVRNavigationManager::instance()
{
    if (!s_instance)
        s_instance = new coVRNavigationManager;
    return s_instance;
}

coVRNavigationManager::coVRNavigationManager()
    : ui::Owner("NavigationManager", cover->ui)
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

void coVRNavigationManager::init()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew coVRNavigationManager\n");

    initInteractionDevice();

    readConfigFile();

    initMatrices();

    initMenu();

    initShowName();

    updatePerson();

    oldHandPos = Vec3(0, 0, 0);

    osg::Sphere *rotPointSphere = new osg::Sphere(Vec3(0, 0, 0), coCoviseConfig::getFloat("COVER.rotationPointSize", 0.5)); // obsolete config-rotationPointSize

    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *sphereDrawable = new osg::ShapeDrawable(rotPointSphere, hint);
    osg::Geode *rotPointGeode = new osg::Geode();
    rotPointGeode->addDrawable(sphereDrawable);
    rotPoint = new osg::MatrixTransform();
    rotPoint->addChild(rotPointGeode);
    setRotationPointVisible(rotationPointVisible);

    setNavMode(XForm);
}

coVRNavigationManager::~coVRNavigationManager()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete coVRNavigationManager\n");

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

void coVRNavigationManager::updatePerson()
{
    if (!coVRConfig::instance()->mouseTracking())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        coInteractionManager::the()->registerInteraction(interactionC);
    }
    else
    {
        coInteractionManager::the()->unregisterInteraction(interactionB);
        coInteractionManager::the()->unregisterInteraction(interactionC);
    }

    doMouseNav = coVRConfig::instance()->mouseNav();
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

void coVRNavigationManager::initInteractionDevice()
{
    interactionA = new coNavInteraction(coInteraction::ButtonA, "ProbeMode", coInteraction::Navigation);
    interactionB = new coNavInteraction(coInteraction::ButtonB, "ProbeMode", coInteraction::Navigation);
    interactionC = new coNavInteraction(coInteraction::ButtonC, "ProbeMode", coInteraction::Navigation);
    interactionMenu = new coNavInteraction(coInteraction::ButtonA, "MenuMode", coInteraction::Menu);
    interactionShortcut = new coNavInteraction(coInteraction::NoButton, "Keyboard", coInteraction::Low);

    mouseNavButtonRotate = coCoviseConfig::getInt("RotateButton", "COVER.Input.MouseNav", 0);
    mouseNavButtonScale = coCoviseConfig::getInt("ScaleButton", "COVER.Input.MouseNav", 1);
    mouseNavButtonTranslate = coCoviseConfig::getInt("TranslateButton", "COVER.Input.MouseNav", 2);
    interactionMA = new coMouseButtonInteraction(coInteraction::ButtonA, "MouseNav");
    interactionMA->setGroup(coInteraction::GroupNavigation);
    interactionMB = new coMouseButtonInteraction(coInteraction::ButtonB, "MouseNav");
    interactionMB->setGroup(coInteraction::GroupNavigation);
    interactionMC = new coMouseButtonInteraction(coInteraction::ButtonC, "MouseNav");
    interactionMC->setGroup(coInteraction::GroupNavigation);
    interactionRel = new coRelativeInputInteraction("SpaceMouse");
    interactionRel->setGroup(coInteraction::GroupNavigation);

    wiiNav = coVRConfig::instance()->useWiiNavigationVisenso();
    /*if (wiiNav)
      fprintf(stderr, "Wii Navigation Visenso on\n");
   else
      fprintf(stderr, "Wii Navigation Visenso OFF\n");*/
    wiiFlag = 0;
}

int coVRNavigationManager::readConfigFile()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::readConfigFile\n");

    navExp = coCoviseConfig::getFloat("COVER.NavExpBase", navExp);

    collisionDist = coCoviseConfig::getFloat("COVER.CollisionDist", 80.0f);

    stepSize = coCoviseConfig::getFloat("COVER.StepSize", 400.0f);

    jsXmax = coCoviseConfig::getInt("xmax", "COVER.Input.Joystick", 1000);
    jsYmax = coCoviseConfig::getInt("ymax", "COVER.Input.Joystick", 1000);
    jsXmin = coCoviseConfig::getInt("xmin", "COVER.Input.Joystick", 0);
    jsYmin = coCoviseConfig::getInt("ymin", "COVER.Input.Joystick", 0);
    jsZeroPosX = coCoviseConfig::getInt("zerox", "COVER.Input.Joystick", 500);
    jsZeroPosY = coCoviseConfig::getInt("zeroy", "COVER.Input.Joystick", 500);

    jsOffsetX = coCoviseConfig::getInt("x", "COVER.Input.Offset", 20);
    jsOffsetY = coCoviseConfig::getInt("y", "COVER.Input.Offset", 20);

    jsEnabled = coCoviseConfig::isOn("COVER.Input.Joystick", false);
    visensoJoystick = coCoviseConfig::isOn("COVER.Input.VisensoJoystick", false);

    menuButtonQuitInterval = coCoviseConfig::getFloat("value", "COVER.MenuButtonQuitInterval", -1.0f);

    // parse snapping and tell NavigationManager
    snapping = coCoviseConfig::isOn("COVER.Snap", snapping);
    snapDegrees = coCoviseConfig::getFloat("COVER.SnapDegrees", snapDegrees);
    rotationSpeed = coCoviseConfig::getFloat("COVER.RotationSpeed", rotationSpeed);
    if (snapDegrees > 0)
    {
        snappingD = true;
        snapping = true;
    }
    else
    {
        snappingD = false;
    }
    turntable = coCoviseConfig::isOn("COVER.Turntable", turntable);

    m_restrict = coCoviseConfig::isOn("COVER.Restrict", m_restrict);

    return 0;
}

void coVRNavigationManager::initMatrices()
{
    invBaseMatrix.makeIdentity();
    oldInvBaseMatrix.makeIdentity();
}

void coVRNavigationManager::initMenu()
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
            VRSceneGraph::instance()->viewAll(false);
        });
    });
    m_viewAll->setPriority(ui::Element::Toolbar);
    m_viewAll->setIcon("zoom-fit-best");

    m_resetView = new ui::Action(navMenu_, "ResetView");
    m_resetView->setText("Reset view");
    m_resetView->setShortcut("Shift+V");
    m_resetView->setCallback([this, protectNav](){
        protectNav([](){
            VRSceneGraph::instance()->viewAll(true);
        });
    });
    m_resetView->setIcon("zoom-original");

    scaleSlider_ = new ui::Slider(navMenu_, "ScaleFactor");
    scaleSlider_->setVisible(false, ui::View::VR);
    scaleSlider_->setText("Scale factor");
    scaleSlider_->setBounds(1e-5, 1e5);
    scaleSlider_->setScale(ui::Slider::Logarithmic);
    scaleSlider_->setValue(cover->getScale());
    scaleSlider_->setCallback([this, protectNav](double val, bool released){
        protectNav([this, val](){
            startMouseNav();
            doMouseScale(val);
            stopMouseNav();
        });
    });

    scaleUpAction_ = new ui::Action(navMenu_, "ScaleUp");
    scaleUpAction_->setText("Scale up");
    scaleUpAction_->setVisible(false);
    scaleUpAction_->setCallback([this, protectNav](){
        protectNav([this](){
            startMouseNav();
            doMouseScale(cover->getScale() * 1.1f);
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
    scaleDownAction_->setCallback([this, protectNav](){
        protectNav([this](){
            startMouseNav();
            doMouseScale(cover->getScale() / 1.1f);
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
    navGroup_->setCallback([this](int value){ setNavMode(NavMode(value), false); });

    driveSpeedSlider_ = new ui::Slider(navMenu_, "DriveSpeed");
    driveSpeedSlider_->setText("Drive speed");
    driveSpeedSlider_->setBounds(0., 30.);
    driveSpeedSlider_->setValue(driveSpeed);;
    driveSpeedSlider_->setCallback([this](double val, bool released){driveSpeed=val;});
    collisionButton_ = new ui::Button(navMenu_, "Collision");
    collisionButton_->setText("Collision detection");
    collisionButton_->setState(collision);
    collisionButton_->setCallback([this](bool state){collision = state;});

    snapButton_ = new ui::Button(navMenu_, "Snap");
    snapButton_->setState(snapping);
    snapButton_->setCallback([this](bool state){snapping=state;});

    auto restrictButton = new ui::Button(navMenu_, "Restrict");
    restrictButton->setText("Restrict interactors");
    restrictButton->setState(m_restrict);
    restrictButton->setCallback([this](bool state){
        m_restrict = state;
    });
    restrictButton->setVisible(true);
    restrictButton->setVisible(false, ui::View::VR);

#if 0
    ui::RadioButton *xformRotButton_=nullptr, *xformTransButton_=nullptr;
#endif
}

void coVRNavigationManager::initShowName()
{
    Vec4 fgcolor(0.5451, 0.7020, 0.2431, 1.0);
    Vec4 bgcolor(0.0, 0.0, 0.0, 0.8);
    float lineLen = 0.06 * cover->getSceneSize();
    float fontSize = 0.02 * cover->getSceneSize();
    nameLabel_ = new coVRLabel("test", fontSize, lineLen, fgcolor, bgcolor);
    nameLabel_->hide();

    showGeodeName_ = coCoviseConfig::isOn("COVER.ShowGeodeName", true);

    nameMenu_ = new vrui::coRowMenu("Object Name");
    nameMenu_->setVisible(false);
    vrui::OSGVruiMatrix t, r, m;
    float px = coCoviseConfig::getFloat("x", "COVER.NameMenuPosition", -0.5 * cover->getSceneSize());
    float py = coCoviseConfig::getFloat("y", "COVER.NameMenuPosition", 0.0);
    float pz = coCoviseConfig::getFloat("z", "COVER.NameMenuPosition", 0.3 * cover->getSceneSize());
    float scale;
    scale = coCoviseConfig::getFloat("COVER.Menu.Size", 1.0);
    scale = coCoviseConfig::getFloat("COVER.NameMenuScale", scale);

    t.makeTranslate(px, py, pz);
    r.makeEuler(0, 90, 0);
    r.mult(&t);
    nameMenu_->setTransformMatrix(&r);
    nameMenu_->setVisible(false);
    nameMenu_->setScale(scale * cover->getSceneSize() / 2500);
    nameButton_ = new coButtonMenuItem("-");
    nameMenu_->add(nameButton_);
}

Vec3 coVRNavigationManager::getCenter() const
{
    if (rotationPoint)
    {
        return rotPointVec;
    }

    if (isViewerPosRotation)
    {
        return cover->getViewerMat().getTrans();
    }

    osg::BoundingSphere bsphere = VRSceneGraph::instance()->getBoundingSphere();
    if (bsphere.radius() == 0.f)
        bsphere.radius() = 1.f;
    osg::Vec3 originInWorld = bsphere.center() * cover->getBaseMat();
    return originInWorld;
}

// process key events
bool coVRNavigationManager::keyEvent(int type, int keySym, int mod)
{
    bool handled = false;

    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::keyEvent\n");

    shiftEnabled = (mod & osgGA::GUIEventAdapter::MODKEY_SHIFT)!=0;
    // Beschleunigung
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == osgGA::GUIEventAdapter::KEY_Up)
        {
            currentVelocity += driveSpeed * 1.0;
            handled = true;
        }
        else if (keySym == osgGA::GUIEventAdapter::KEY_Right)
        {
            currentVelocity += currentVelocity * driveSpeed / 10.0;
            handled = true;
        }
        else if (keySym == osgGA::GUIEventAdapter::KEY_Down)
        {
            // Verzoegerung
            currentVelocity -= driveSpeed * 1.0;
            handled = true;
        }
        else if (keySym == osgGA::GUIEventAdapter::KEY_Left)
        {
            // halt
            currentVelocity = 0;
            handled = true;
        }
    }
    return handled;
}

void coVRNavigationManager::saveCurrentBaseMatAsOldBaseMat()
{
    oldInvBaseMatrix = cover->getInvBaseMat();
    oldLeftPos = currentLeftPos;
    oldRightPos = currentRightPos;
}

// returns true if collision occurred
bool coVRNavigationManager::avoidCollision(osg::Vec3 &glideVec)
{
    osg::Vec3 tmp;
    osg::Vec3 oPos[2], Tip[2], nPos[2];
    osg::Vec3 diff;
    osg::Matrix dcs_mat;
    glideVec.set(0, 0, 0);
    osg::Vec3 rightPos(VRViewer::instance()->getSeparation() / 2.0f, 0.0f, 0.0f);
    osg::Vec3 leftPos(-(VRViewer::instance()->getSeparation() / 2.0f), 0.0f, 0.0f);
    osg::Matrix vmat = cover->getViewerMat();
    currentRightPos = vmat.preMult(rightPos);
    currentLeftPos = vmat.preMult(leftPos);

    osg::Matrix currentBase, diffMat;
    currentBase = cover->getBaseMat();
    diffMat = oldInvBaseMatrix * currentBase;

    // left segment
    nPos[0].set(currentLeftPos[0], currentLeftPos[1], currentLeftPos[2]);
    // right segment
    nPos[1].set(currentRightPos[0], currentRightPos[1], currentRightPos[2]);
    oPos[0] = diffMat.preMult(oldLeftPos);
    oPos[1] = diffMat.preMult(oldRightPos);

    diff = nPos[0] - oPos[0];
    float dist = diff.length();
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

    osg::ref_ptr<osg::LineSegment> ray1 = new osg::LineSegment();
    ray1->set(oPos[0], Tip[0]);

    diff = nPos[1] - oPos[1];
    diff *= collisionDist / diff.length();
    Tip[1] = nPos[1] + diff;
    osg::ref_ptr<osg::LineSegment> ray2 = new osg::LineSegment();
    ray2->set(oPos[1], Tip[1]);

    osg::Vec3 hitPoint[2];
    osg::Vec3 hitNormal[2];

    osgUtil::IntersectVisitor visitor;
    visitor.setTraversalMask(Isect::Collision);
    visitor.addLineSegment(ray1.get());
    visitor.addLineSegment(ray2.get());

    cover->getScene()->accept(visitor);

    int num1 = visitor.getNumHits(ray1.get());
    int num2 = visitor.getNumHits(ray2.get());
    if (num1 || num2)
    {
        osgUtil::Hit hitInformation1;
        osgUtil::Hit hitInformation2;
        if (num1)
            hitInformation1 = visitor.getHitList(ray1.get()).front();
        if (num2)
            hitInformation2 = visitor.getHitList(ray2.get()).front();
        osg::Matrix xform;
        osg::Vec3 distVec[2];
        osg::Vec3 glideV[2];
        float dist[2];
        dist[0] = 0;
        dist[1] = 0;
        float cangle;
        int i;
        hitPoint[0] = hitInformation1.getWorldIntersectPoint();
        hitPoint[1] = hitInformation2.getWorldIntersectPoint();
        hitNormal[0] = hitInformation1.getWorldIntersectNormal();
        hitNormal[1] = hitInformation1.getWorldIntersectNormal();
        for (i = 0; i < 2; i++)
        {
            distVec[i] = hitPoint[i] - oPos[i];
            distVec[i] *= 0.9;
            hitPoint[i] = oPos[i] + distVec[i];
            distVec[i] = hitPoint[i] - Tip[i];
            dist[i] = distVec[i].length();
            //fprintf(stderr,"hitPoint: %f %f %f\n",hitPoint[i][0],hitPoint[i][1],hitPoint[i][2]);
            //fprintf(stderr,"Tip: %f %f %f\n",Tip[i][0],Tip[i][1],Tip[i][2]);
            //fprintf(stderr,"oPos: %f %f %f\n",oPos[i][0],oPos[i][1],oPos[i][2]);
            //fprintf(stderr,"nPos: %f %f %f\n",nPos[i][0],nPos[i][1],nPos[i][2]);
            //fprintf(stderr,"distVec: %f %f %f %d\n",distVec[i][0],distVec[i][1],distVec[i][2],i);
            //fprintf(stderr,"hitNormal: %f %f %f %d\n",hitNormal[i][0],hitNormal[i][1],hitNormal[i][2],i);

            tmp = distVec[i];
            tmp.normalize();
            hitNormal[i].normalize();
            cangle = hitNormal[i] * (tmp);
            if (cangle > M_PI_2 || cangle < -M_PI_2)
            {
                hitNormal[i] = -hitNormal[i];
            }
            //fprintf(stderr,"hitNormal: %f %f %f %d\n",hitNormal[i][0],hitNormal[i][1],hitNormal[i][2],i);
            //fprintf(stderr,"tmp: %f %f %f %f\n",tmp[0],tmp[1],tmp[2],cangle);
            hitNormal[i] *= (dist[i]) * cangle;
            glideV[i] = hitNormal[i] - distVec[i];
            // fprintf(stderr,"glideV: %f %f %f %d\n",glideV[i][0],glideV[i][1],glideV[i][2],i);
        }
        // get xform matrix
        dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
        if (dist[0] > dist[1])
        {
            //use dist1
            i = 0;
        }
        else
        {
            i = 1;
        }
        osg::Matrix tmp;
        tmp.makeTranslate(-distVec[i][0], -distVec[i][1], -distVec[i][2]);
        dcs_mat.postMult(tmp);
        glideVec = -glideV[i];
        // set new xform matrix
        VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
        saveCurrentBaseMatAsOldBaseMat();
        coVRCollaboration::instance()->SyncXform();
        return true;
    }
    saveCurrentBaseMatAsOldBaseMat();
    return false;
}

void
coVRNavigationManager::adjustFloorHeight()
{
    if (navMode == Walk)
    {
        doWalkMoveToFloor();
    }
}

void
coVRNavigationManager::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVRNavigationManager::update\n");

    scaleSlider_->setValue(cover->getScale());

    coPointerButton *button = cover->getPointerButton();
    oldHandDir = handDir;
    if (!wiiNav)
    {
        handMat = cover->getPointerMat();
    }
    else
    {
        if (Input::instance()->isHandValid())
            handMat = Input::instance()->getHandMat();
    }
    oldHandPos = handPos;
    handPos = handMat.getTrans();
    handDir[0] = handMat(1, 0);
    handDir[1] = handMat(1, 1);
    handDir[2] = handMat(1, 2);

    // static osg::Vec3Array *coord = new osg::Vec3Array(4*6);
    osg::Matrix dcs_mat, rot_mat, tmpMat;
    osg::Vec3 handRight, oldHandRight;
    osg::Vec3 dirAxis, rollAxis;
    //int collided[6] = {0, 0, 0, 0, 0, 0}; /* front,back,right,left,up,down */
    //static int numframes=0;

    if (doMouseNav)
    {
        mx = mouseX();
        my = mouseY();
        float widthX = mouseWinWidth(), widthY = mouseWinHeight();
        originX = widthX / 2;
        originY = widthY / 2; // verallgemeinern!!!

        dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
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
            fprintf(stderr, "coVRNavigationManager::update: unhandled navigation mode interaction %d\n", navMode);
            break;
        }
    }

    //fprintf(stderr, "coVRNavigationManager: doMouseNav=%d, mouseFlag=%d, navMode=%d, handLocked=%d\n", (int)doMouseNav, (int)mouseFlag, (int)navMode, (int)handLocked);

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
            fprintf(stderr, "coVRNavigationManager::update: unhandled navigation mode %d\n", (int)navMode);
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
            fprintf(stderr, "coVRNavigationManager::update: unhandled navigation mode %d\n", (int)navMode);
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
            fprintf(stderr, "coVRNavigationManager::update: unhandled navigation mode %d\n", (int)navMode);
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
        osg::Matrix relMat = Input::instance()->getRelativeMat();
        coCoord co(relMat);
        osg::Matrix tf = VRSceneGraph::instance()->getTransform()->getMatrix();
        auto tr = applySpeedFactor(relMat.getTrans());

        switch (getMode())
        {
        case Scale:
        {
            double s = pow(1.03, co.hpr[0]);
            doMouseScale(cover->getScale() * s);
            break;
        }
        case XForm:
        {
            osg::Vec3 center = getCenter();

            relMat.makeTranslate(-tr);
            tf *= relMat;

            MAKE_EULER_MAT(relMat, -co.hpr[0], -co.hpr[1], -co.hpr[2]);
            osg::Matrix originTrans, invOriginTrans;
            originTrans.makeTranslate(center); // rotate arround the center of the objects in objectsRoot
            invOriginTrans.makeTranslate(-center);
            relMat = invOriginTrans * relMat * originTrans;
            tf *= relMat;
            break;
        }
        case Fly:
        {
            relMat.setTrans(tr);
            tf *= relMat;
            break;
        }
        case Glide:
        case Walk:
        {
            MAKE_EULER_MAT(relMat, co.hpr[0], 0, 0);
            relMat.setTrans(tr);
            tf *= relMat;
            break;
        }
        default:
        {
            break;
        }
        }

        if (tf != VRSceneGraph::instance()->getTransform()->getMatrix())
        {
            VRSceneGraph::instance()->getTransform()->setMatrix(tf);
            coVRCollaboration::instance()->SyncXform();
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
        && (!cover->isPointerLocked())
        && (!coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM))
        && (!(cover->getMouseButton() && (doMouseNav))))
    {
        dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
        int xMove = (fabs(AnalogX - (float)jsZeroPosX) > (float)jsOffsetX);
        int yMove = (fabs(AnalogY - (float)jsZeroPosY) > (float)jsOffsetY);
        //wenn joystick in der nullstellung ist: zuruecksetzen vo firstTime
        static int firstTime = 1;
        if (!xMove && !yMove)
            firstTime = 1;

        AnalogX = (AnalogX - jsZeroPosX) / 10;
        AnalogY = (AnalogY - jsZeroPosY) / 10;
        float maxAnalogX = (jsXmax - jsZeroPosX) / 10;
        float maxAnalogY = (jsYmax - jsZeroPosY) / 10;

        if ((xMove || yMove) && navMode != Walk && navMode != Fly && navMode != Glide && navMode != ShowName && navMode != Measure && navMode != TraverseInteractors && navMode != SelectInteract)
        {

            //einfache Translation in xy-Ebene
            if ((button->getState() & vruiButtons::ACTION_BUTTON) == 0)
            {
                float maxVelo = 10.0 * driveSpeed;
                osg::Vec3 relVelo, absVelo;
                relVelo[0] = -(maxVelo / maxAnalogX) * AnalogX;
                relVelo[1] = -(maxVelo / maxAnalogY) * AnalogY;
                relVelo[2] = 0.0;
                absVelo = osg::Matrix::transform3x3(handMat, relVelo);
                osg::Matrix tmp;
                tmp.makeTranslate(absVelo[0], absVelo[1], 0.0);
                dcs_mat.postMult(tmp);
            }

            //Rotation um Objekt
            else if (button->getState() & vruiButtons::ACTION_BUTTON)
            {

                static float trX, trY, trZ;
                trX = dcs_mat(3, 0);
                trY = dcs_mat(3, 1); //diese drei sind fuer Rotation um Objektursprung
                trZ = dcs_mat(3, 2);
                float maxRotDegrees = driveSpeed;
                float degreesX = (maxRotDegrees / maxAnalogY) * AnalogY;
                float degreesY = (maxRotDegrees / maxAnalogX) * AnalogX;
                osg::Matrix tmp;
                tmp.makeTranslate(-trX, -trY, -trZ);
                dcs_mat.postMult(tmp);

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
                    osg::Matrix rot;
                    MAKE_EULER_MAT(rot, degreesY, 0.0, 0.0);
                    dcs_mat.mult(dcs_mat, rot);
                }
                if (yMove) // nurRotation um Achse die x entspricht
                {
                    osg::Matrix rot;
                    MAKE_EULER_MAT(rot, 0.0, degreesX, 0.0);
                    dcs_mat.mult(dcs_mat, rot);
                }

                tmp.makeTranslate(trX, trY, trZ);
                dcs_mat.postMult(tmp);
            }
        }

        //drive- mode, mit Translation in Joystick y
        else if ((xMove || yMove) && navMode == Glide)
        //und Rotation um Hand in Joystick x
        {
            float maxVelo = 10.0 * driveSpeed;
            osg::Vec3 relVelo, absVelo;
            relVelo[0] = 0.0;
            relVelo[1] = -(maxVelo / maxAnalogY) * AnalogY;
            relVelo[2] = 0.0;
            absVelo = osg::Matrix::transform3x3(handMat, relVelo);

            //Ende Translation Joystick bottom-top

            //jetzt Rotation Joystick left-right, zunaechst Werte fuer TRansformatio berechnen
            //unten dann zwischen +-handPos[] ausfuehren
            dirAxis[0] = 0.0;
            dirAxis[1] = 0.0;
            dirAxis[2] = 1.0;

            //normieren auf Abstand Objekt- BetrachterHand
            osg::Vec3 objPos;
            objPos = dcs_mat.getTrans();
            osg::Vec3 distance = handPos - objPos;
            //Schaetzwert in der richtigen Groessenordnung
            float distObjHand = distance.length() / 500.0;
            float maxRotDegrees = driveSpeed; //n bisschen allgemeiner waer gut
            float degrees = (maxRotDegrees / (maxAnalogX * distObjHand)) * -AnalogX;

            osg::Matrix tmp;
            tmp.makeTranslate(-handPos[0], -handPos[1], -handPos[2]);
            dcs_mat.postMult(tmp);
            if (xMove)
            {
                tmp.makeRotate(degrees * M_PI / 180.0, dirAxis[0], dirAxis[1], dirAxis[2]);
                dcs_mat.postMult(tmp);
            }
            tmp.makeTranslate(handPos[0], handPos[1], handPos[2]);
            dcs_mat.postMult(tmp);
            if (yMove)
            {
                tmp.makeTranslate(absVelo[0], absVelo[1], 0.0);
                dcs_mat.postMult(tmp);
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

            float maxVelo = 10.0 * driveSpeed;
            osg::Vec3 relVelo, absVelo;
            relVelo[0] = 0.0; //-(maxVelo / maxAnalogX) * AnalogX;
            relVelo[1] = -(maxVelo / maxAnalogY) * AnalogY;
            relVelo[2] = 0.0;
            absVelo = osg::Matrix::transform3x3(handMat, relVelo);
            dcs_mat.postMult(osg::Matrix::translate(absVelo[0], absVelo[1], absVelo[2]));

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
                if (!old_mat.invert(old_mat))
                    fprintf(stderr, "coVRNavigationManager::update old_mat is singular\n");
                old_dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
            }
            // if xform in progress

            //else

            //diese if macht im moment nichts gescheites
            if (button->getState() & vruiButtons::DRIVE_BUTTON)
            {
                handMat.setTrans(0, 0, 0);
                old_mat.setTrans(0, 0, 0);
                old_dcs_mat.setTrans(0, 0, 0);

                osg::Matrix rel_mat; //, dcs_mat;
                rel_mat.mult(old_mat, handMat); //erste handMat * aktualisierte handMat
                dcs_mat.mult(old_dcs_mat, rel_mat);
                //VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
                coVRCollaboration::instance()->SyncXform();
            }

            // calc axis and angle for roll change (from hand roll change)
            handRight.set(handMat(0, 0), handMat(0, 1), handMat(0, 2));
            oldHandRight.set(old_mat(0, 0), old_mat(0, 1), old_mat(0, 2));
            float rollScale;
            float rollAngle;
            rollAxis = handRight ^ oldHandRight;
            rollScale = fabs(rollAxis * (handDir));
            rollAxis = rollAxis * rollScale;
            rollAngle = rollAxis.length();

            // apply roll change
            //cout << "rollAxis " << rollAxis[0] << "  " << rollAxis[1] << "  " << rollAxis[2] << endl;
            if ((rollAxis[0] != 0.0) || (rollAxis[1] != 0.0) || (rollAxis[2] != 0.0))
            {

                rot_mat.makeRotate(rollAngle * M_PI / 180, rollAxis[0], rollAxis[1], rollAxis[2]);
                dcs_mat.mult(dcs_mat, rot_mat);
            }

            //Ziel: aus Handbewegung auf heading schliessen

            static float trFlyX, trFlyY, trFlyZ;
            trFlyX = dcs_mat(3, 0);
            trFlyY = dcs_mat(3, 1); //diese drei sind fuer Rotation um Objektursprung
            trFlyZ = dcs_mat(3, 2);

            //cos(winkel) = a*b / abs(a) * abs(b)

            float lastDegrees = acos(oldHandDir[2] / oldHandDir.length());
            float degrees = acos(handDir[2] / handDir.length());
            //cout << "winkel " << 90.0 / acos(0.0) * degrees << endl;

            //float degreesX = 0.0;
            float degreesY = (90.0 / acos(0.0)) * (degrees - lastDegrees);
            dcs_mat.postMult(osg::Matrix::translate(-trFlyX, -trFlyY, -trFlyZ));

            //rotate
            osg::Matrix rot;
            MAKE_EULER_MAT(rot, degreesY, 0.0, 0.0);
            dcs_mat.mult(dcs_mat, rot);

            /*  osg::Matrix rot;
             MAKE_EULER_MAT(rot,0.0, degreesX, 0.0);
             dcs_mat.mult(dcs_mat, rot);
         */

            osg::Matrix tmp;
            tmp.makeTranslate(trFlyX, trFlyY, trFlyZ);
            dcs_mat.postMult(tmp);
        }

        // nur mal zum ausprobieren, rotation um pointerachse
        else if ((xMove || yMove) && navMode == Walk)
        {
            //normieren auf Abstand Objekt- BetrachterHand
            float maxRotDegrees = driveSpeed;
            float degrees = (maxRotDegrees / maxAnalogX) * -AnalogX;
            dirAxis = handDir;

            if (xMove)
            {
                dcs_mat.postMult(osg::Matrix::rotate(degrees * (M_PI / 180), dirAxis[0], dirAxis[1], dirAxis[2]));
            }
            if (yMove)
            {
            }
        }

        VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
        coVRCollaboration::instance()->SyncXform();
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
                VRSceneGraph::instance()->setScaleFromButton(AnalogY * cover->frameDuration());
                doGuiTranslate(-AnalogX * cover->frameDuration(), 0.0f, 0.0f);
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
                dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
                dcs_mat.postMult(osg::Matrix::translate(-AnalogX * driveSpeed, -AnalogY * driveSpeed, 0.0f));
                VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
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
        osg::Vec3 glideVec;
        while (i < 4 && avoidCollision(glideVec))
        {

            dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();

            osg::Matrix tmp;
            tmp.makeTranslate(glideVec[0], glideVec[1], glideVec[2]);
            dcs_mat.postMult(tmp);

            VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);

            i++;
        }
    }
    else
    {
        jump = true;
    }

    if (navMode == ShowName || navMode == SelectInteract)
        highlightSelectedNode(cover->getIntersectedNode());

    if (coVRConfig::instance()->isMenuModeOn())
    {
        if (button->wasReleased(vruiButtons::MENU_BUTTON))
        {
            setMenuMode(true);
            //menuCallback(coVRNavigationManager::instance(), NULL);
        }
    }

    if ((button->getState() == vruiButtons::MENU_BUTTON) && (menuButtonQuitInterval >= 0.0))
    {
        if (button->wasPressed(vruiButtons::MENU_BUTTON))
        {
            menuButtonStartTime = cover->currentTime();
        }
        else if (menuButtonStartTime + menuButtonQuitInterval < cover->currentTime())
        {
            OpenCOVER::instance()->requestQuit();
        }
    }

    if (getMode() == TraverseInteractors)
    {
        if (button->wasReleased(vruiButtons::INTER_NEXT))
            coVRIntersectionInteractorManager::the()->cycleThroughInteractors();
        else if (button->wasReleased(vruiButtons::INTER_PREV))
            coVRIntersectionInteractorManager::the()->cycleThroughInteractors(false);
    }

    for (std::vector<coMeasurement *>::iterator it = measurements.begin(); it != measurements.end(); it++)
    {
        (*it)->preFrame();
    }
}

void coVRNavigationManager::setNavMode(NavMode mode, bool updateGroup)
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
        coVRSelectionManager::instance()->setSelectionOnOff(mode == Select);
    }
    if (navMode == SelectInteract || mode == SelectInteract)
    {
        toggleSelectInteract(mode == SelectInteract);
    }

    navMode = mode;

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
        fprintf(stderr, "coVRNavigationManager::setNavMode: unknown mode %d\n", (int)navMode);
        break;
    }

    if (mode == NavNone || coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM))
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
            if (!coVRConfig::instance()->mouseTracking())
            {
                coInteractionManager::the()->registerInteraction(interactionA);
            }
        }
        if (interactionA->isRegistered())
        {
            if (coVRConfig::instance()->mouseTracking())
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

    if (coVRConfig::instance()->isMenuModeOn() && oldNavMode == Menu && mode != Menu)
        VRSceneGraph::instance()->setMenuMode(false);

    if (navMode != Measure && oldNavMode == Measure)
    {
        for (std::vector<coMeasurement *>::iterator it = measurements.begin(); it != measurements.end(); it++)
        {
            delete (*it);
        }
        measurements.clear();
    }
}

ui::ButtonGroup *coVRNavigationManager::navGroup() const
{
    return navGroup_;
}

void coVRNavigationManager::toggleCollide(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::toggleCollide %d\n", state);

    collision = state;
    collisionButton_->setState(collision);
}

void coVRNavigationManager::toggleShowName(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::toggleShowName %d\n", state);

    if (state)
    {
        // enable intersection with scene
        coIntersection::instance()->isectAllNodes(true);
    }
    else
    {
        if (cover->debugLevel(4))
            fprintf(stderr, "realtoggle\n");
        nameLabel_->hide();
        nameMenu_->setVisible(false);
        highlightSelectedNode(NULL);
        // disable intersection with scene
        coIntersection::instance()->isectAllNodes(false);
    }
}

void coVRNavigationManager::toggleInteractors(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::toggleInteractors %d\n", state);

    if (state)
    {
        coVRIntersectionInteractorManager::the()->enableCycleThroughInteractors();
    }
    else if (navMode == TraverseInteractors || (oldNavMode == TraverseInteractors && navMode == Menu))
    {
        coVRIntersectionInteractorManager::the()->disableCycleThroughInteractors();
    }
}

void coVRNavigationManager::setMenuMode(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::seMenuMode(%d) old=%d\n", state, oldNavMode);

    if (!coVRConfig::instance()->isMenuModeOn())
        return;

    if (state)
    {
        VRSceneGraph::instance()->setMenuMode(true);
    }
    else
    {
        VRSceneGraph::instance()->setMenuMode(false);

    } /* else if (!state)
   {
      VRSceneGraph::instance()->setMenuMode(true);
      VRSceneGraph::instance()->setMenuMode(false);
   }*/
}

float coVRNavigationManager::getPhiZVerti(float y2, float y1, float x2, float widthX, float widthY)
{
    float phiZMax = 180.0;
    float y = (y2 - y1);
    float phiZ = 4 * pow(x2, 2) * phiZMax * y / (pow(widthX, 2) * widthY);
    if (x2 < 0.0)
        return (phiZ);
    else
        return (-phiZ);
}

float coVRNavigationManager::getPhiZHori(float x2, float x1, float y2, float widthY, float widthX)
{
    float phiZMax = 180.0;
    float x = x2 - x1;
    float phiZ = 4 * pow(y2, 2) * phiZMax * x / (pow(widthY, 2) * widthX);

    if (y2 > 0.0)
        return (phiZ);
    else
        return (-phiZ);
}

float coVRNavigationManager::getPhi(float relCoord1, float width1)
{
    float phi, phiMax = 180.0;
    phi = phiMax * relCoord1 / width1;
    return (phi);
}

void coVRNavigationManager::makeRotate(float heading, float pitch, float roll, int headingBool, int pitchBool, int rollBool)
{
    osg::Matrix actRot = VRSceneGraph::instance()->getTransform()->getMatrix();
    osg::Matrix finalRot;

    if (turntable)
    {
        osg::Matrix newHeadingRot;
        osg::Matrix newPitchRot;

        MAKE_EULER_MAT(newHeadingRot, headingBool * heading, 0.0, 0.0);
        MAKE_EULER_MAT(newPitchRot, 0.0, pitchBool * pitch, 0.0);

        osg::Vec3 rotAxis = osg::Vec3(0.0f, 0.0f, 0.0f);
        rotAxis = rotAxis * osg::Matrix::inverse(actRot); // transform axis into local coordinates
        newHeadingRot = osg::Matrix::translate(-rotAxis) * newHeadingRot * osg::Matrix::translate(rotAxis); // rotate around the axis

        finalRot = newHeadingRot * actRot * newPitchRot; // apply headingRot in local coordinates (we want to rotate according to the objects rotated up vector)
    }
    else
    {
        osg::Matrix newRot;
        MAKE_EULER_MAT(newRot, headingBool * heading, pitchBool * pitch, rollBool * roll / 7.0);
        finalRot.mult(actRot, newRot);
    }

    VRSceneGraph::instance()->getTransform()->setMatrix(finalRot);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::doMouseFly()
{
    osg::Vec3 velDir;

    osg::Matrix dcs_mat;
    dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    float heading = 0.0;
    float pitch = (my - y0) / 300;
    float roll = (mx - x0) / -300;
    osg::Matrix rot;
    MAKE_EULER_MAT(rot, heading, pitch, roll);
    dcs_mat.postMult(rot);
    velDir[0] = 0.0;
    velDir[1] = -1.0;
    velDir[2] = 0.0;
    osg::Matrix tmp;
    tmp.makeTranslate(velDir[0] * currentVelocity, velDir[1] * currentVelocity, velDir[2] * currentVelocity);
    dcs_mat.postMult(tmp);
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::doMouseXform()
{
    osg::Matrix dcs_mat;
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
            cover->setCurrentCursor(osgViewer::GraphicsWindow::CycleCursor);

            float newx, newy; //newz;//relz0
            float heading, pitch, roll, rollVerti, rollHori;

            newx = mouseX() - originX;
            newy = mouseY() - originY;

            //getPhi kann man noch aendern, wenn xy-Rotation nicht gewichtet werden soll
            heading = getPhi(newx - relx0, widthX);
            pitch = -getPhi(newy - rely0, widthY);

            rollVerti = getPhiZVerti(newy, rely0, newx, widthX, widthY);
            rollHori = getPhiZHori(newx, relx0, newy, widthY, widthX);
            roll = (rollVerti + rollHori);

            makeRotate(heading, pitch, roll, 1, 1, 1);

            relx0 = mouseX() - originX;
            rely0 = mouseY() - originY;
        }
        else if (shiftMouseNav || isViewerPosRotation) //Rotation um beliebigen Punkt funktioniert
        {
            cover->setCurrentCursor(osgViewer::GraphicsWindow::CycleCursor);

            osg::Matrix doTrans, rot, doRot, doRotObj;
            dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();

            //Rotation um beliebigen Punkt -> Mauszeiger, Viewer-Position, ...
            doTrans.makeTranslate(-transXRel, (-transYRel), -transZRel);
            doRotObj.mult(dcs_mat, doTrans);
            VRSceneGraph::instance()->getTransform()->setMatrix(doRotObj);
            dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();

            float newx, newy; //newz;//relz0
            float heading, pitch, roll, rollVerti, rollHori;

            newx = mouseX() - originX;
            newy = mouseY() - originY;

            //getPhi kann man noch aendern, wenn xy-Rotation nicht gewichtet werden soll
            heading = getPhi(newx - relx0, widthX);
            pitch = -getPhi(newy - rely0, widthY);

            rollVerti = getPhiZVerti(newy, rely0, newx, widthX, widthY);
            rollHori = getPhiZHori(newx, relx0, newy, widthY, widthX);
            roll = rollVerti + rollHori;

            if (turntable)
            {
                osg::Matrix newHeadingRot;
                osg::Matrix newPitchRot;

                MAKE_EULER_MAT(newHeadingRot, heading, 0.0, 0.0);
                MAKE_EULER_MAT(newPitchRot, 0.0, pitch, 0.0);

                osg::Vec3 rotAxis = osg::Vec3(0.0f, 0.0f, 0.0f);
                rotAxis = rotAxis * osg::Matrix::inverse(dcs_mat); // transform axis into local coordinates
                newHeadingRot = osg::Matrix::translate(-rotAxis) * newHeadingRot * osg::Matrix::translate(rotAxis); // rotate around the axis

                doRot = newHeadingRot * dcs_mat * newPitchRot; // apply headingRot in local coordinates (we want to rotate according to the objects rotated up vector)
            }
            else
            {

                MAKE_EULER_MAT(rot, heading, pitch, roll);
                doRot.mult(dcs_mat, rot);
            }

            relx0 = mouseX() - originX;
            rely0 = mouseY() - originY;
            doTrans.makeTranslate(transXRel, (transYRel), transZRel);
            doRotObj.mult(doRot, doTrans);
            VRSceneGraph::instance()->getTransform()->setMatrix(doRotObj);
        }
        coVRCollaboration::instance()->SyncXform();
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
            cover->setCurrentCursor(osgViewer::GraphicsWindow::HandCursor);

            float newxTrans = mouseX() - originX;
            float newyTrans = mouseY() - originY;
            float xTrans, yTrans;
            xTrans = (newxTrans - relx0) * modifiedHSize / widthX;
            yTrans = (newyTrans - rely0) * modifiedVSize / widthY;
            osg::Matrix actTransState = mat0;
            osg::Matrix trans;
            osg::Matrix doTrans;
            trans.makeTranslate(1.1 * xTrans, 0.0, 1.1 * yTrans);
            //die 1.1 sind geschaetzt, um die Ungenauigkeit des errechneten Faktors auszugleichen
            //es sieht aber nicht so aus, als wuerde man es auf diese
            //Weise perfekt hinbekommen
            doTrans.mult(actTransState, trans);
            VRSceneGraph::instance()->getTransform()->setMatrix(doTrans);
        }

        else if (shiftMouseNav) //Translation in der Tiefe
        {
            cover->setCurrentCursor(osgViewer::GraphicsWindow::UpDownCursor);

            float newzTrans = mouseY() - originY;
            float zTrans;
            zTrans = (newzTrans - rely0) * mouseScreenHeight() * 2 / widthY;
            osg::Matrix actTransState = mat0;
            osg::Matrix trans;
            osg::Matrix doTrans;
            trans.makeTranslate(0.0, 2.0 * zTrans, 0.0);
            doTrans.mult(actTransState, trans);
            VRSceneGraph::instance()->getTransform()->setMatrix(doTrans);
            coVRCollaboration::instance()->SyncXform();
            coVRCollaboration::instance()->SyncScale();
        }
        coVRCollaboration::instance()->SyncXform();
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

void coVRNavigationManager::doMouseScale(float newScaleFactor)
{
    //getting the old translation and adjust it according to the new scale factor
    osg::Vec3 center = mouseNavCenter;
    //calculate the new translation matrix
    //and move the objects by the new translation
    osg::Vec3 delta2 = mat0.getTrans();
    osg::Vec3 delta = delta2 - center;

    delta *= (newScaleFactor / actScaleFactor);
    delta += center;
    delta -= delta2;
    osg::Matrix tmp;
    tmp.makeTranslate(delta[0], delta[1], delta[2]);
    osg::Matrix xform_mat = mat0 * tmp;

    VRSceneGraph::instance()->getTransform()->setMatrix(xform_mat);
    VRSceneGraph::instance()->setScaleFactor(newScaleFactor);
    coVRCollaboration::instance()->SyncXform();
    coVRCollaboration::instance()->SyncScale();
}

void coVRNavigationManager::doMouseScale()
{
    cover->setCurrentCursor(osgViewer::GraphicsWindow::UpDownCursor);
    float ampl = mouseWinHeight()/2;
    float newScaleY = mouseY() - originY;

    //calculate the new scale factor
    float newScaleFactor = actScaleFactor * exp(((rely0 - newScaleY) / ampl));
    doMouseScale(newScaleFactor);
}

void coVRNavigationManager::doMouseWalk()
{
    osg::Vec3 velDir;
    osg::Matrix dcs_mat;
    dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    osg::Matrix tmp;
    osg::Matrix tmp2;

    if (interactionMA->wasStarted())
    {
        tmp.makeTranslate((mx - x0) * driveSpeed * -1, 0, (my - y0) * driveSpeed * -1);
        dcs_mat.postMult(tmp);
    }
    else if (interactionMA->isRunning())
    {
        float angle = (mx - x0) / 300;
        tmp = osg::Matrix::rotate(angle * M_PI / 180, 0.0, 0.0, 1.0);
        osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
        tmp2.makeTranslate(viewerPos);
        tmp.postMult(tmp2);
        tmp2.makeTranslate(-viewerPos);
        tmp.preMult(tmp2);

        dcs_mat.postMult(tmp);
        velDir[0] = 0.0;
        velDir[1] = 1.0;
        velDir[2] = 0.0;
        currentVelocity = (my - y0) * driveSpeed * -0.5;
        tmp.makeTranslate(velDir[0] * currentVelocity, velDir[1] * currentVelocity, velDir[2] * currentVelocity);
        dcs_mat.postMult(tmp);
    }
    else if (interactionMB->wasStarted())
    { // pan
        tmp.makeTranslate((mx - x0) * driveSpeed * -1, 0, (my - y0) * driveSpeed * -1);
        dcs_mat.postMult(tmp);
    }
    else if (interactionMB->isRunning())
    { // pan
        tmp.makeTranslate((mx - x0) * driveSpeed * -1, 0, (my - y0) * driveSpeed * -1);
        dcs_mat.postMult(tmp);
    }
#if 0
   fprintf(stderr, "buttons=0x%x, mouseNav=%d, handLocked=%d\n",
         cover->getMouseButton(), (int)doMouseNav, handLocked);
#endif
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::stopMouseNav()
{
    cover->setCurrentCursor(osgViewer::GraphicsWindow::LeftArrowCursor);

    actScaleFactor = cover->getScale();
    x0 = mx;
    y0 = my;
    currentVelocity = 10;
    relx0 = x0 - originX; //relativ zum Ursprung des Koordinatensystems
    rely0 = y0 - originY; //dito
    coVRCollaboration::instance()->UnSyncXform();
    coVRCollaboration::instance()->UnSyncScale();
}

void coVRNavigationManager::startMouseNav()
{
    shiftMouseNav = shiftEnabled;

    osg::Matrix dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    mat0 = dcs_mat;
    mouseNavCenter = getCenter();

    actScaleFactor = cover->getScale();
    x0 = mx;
    y0 = my;
    //cerr << "mouseNav" << endl;
    currentVelocity = 10;

    relx0 = x0 - originX; //relativ zum Ursprung des Koordinatensystems
    rely0 = y0 - originY; //dito

    osg::Matrix whereIsViewer;
    whereIsViewer = cover->getViewerMat();
    float yValViewer = whereIsViewer(3, 1);
    float yValObject = dcs_mat(3, 1); //was gibt dcs_mat genau an ?? gibt es eine besser geeignete
    float alphaY = fabs(atan(mouseScreenHeight() / (2.0 * yValViewer)));
    modifiedVSize = 2.0 * tan(alphaY) * fabsf(yValObject - yValViewer);
    float alphaX = fabs(atan(mouseScreenWidth() / (2.0 * yValViewer)));
    modifiedHSize = 2.0 * tan(alphaX) * fabsf(yValObject - yValViewer);
    //float newxTrans = relx0;  //declared but never referenced
    //float newyTrans = rely0;  //dito

    osg::Vec3 transRel = cover->getIntersectionHitPointWorld();

    if (isViewerPosRotation)
    {
        osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
        transXRel = viewerPos[0];
        transYRel = viewerPos[1];
        transZRel = viewerPos[2];
    }
    else
    {
        transXRel = transRel[0];
        transYRel = transRel[1];
        transZRel = transRel[2];
    }
}

void coVRNavigationManager::startXform()
{
    old_mat = handMat;
    if (!old_mat.invert(old_mat))
        fprintf(stderr, "coVRNavigationManager::update old_mat is singular\n");
    old_dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    if (!rotationPoint)
    {
        transformVec = cover->getBBox(VRSceneGraph::instance()->getTransform()).center();
        if (coCoviseConfig::isOn("COVER.ScaleWithInteractors", false))
            transformVec = cover->getBBox(VRSceneGraph::instance()->getScaleTransform()).center();
        //fprintf(stderr, "****set TransformVec to BBOx-Center %f %f %f\n",transformVec[0], transformVec[1], transformVec[2]);
    }
    else
    {
        transformVec = rotPointVec * cover->getBaseMat();
        //fprintf(stderr, "****set TransformVec to %f %f %f\n",transformVec[0], transformVec[1], transformVec[2]);
    }
}

void coVRNavigationManager::doXform()
{
    if (wiiNav)
    {
        // do xformRotate oder translate
        if (wiiFlag != 2 && handPos - oldHandPos == Vec3(0, 0, 0))
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
    osg::Matrix rel_mat, dcs_mat;
    rel_mat.mult(old_mat, handMat); //erste handMat * aktualisierte handMat
    dcs_mat.mult(old_dcs_mat, rel_mat);
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);

    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::doXformRotate()
{
    osg::Matrix rel_mat, dcs_mat;
    osg::Matrix rev_mat, trans_mat;
    // zurueck in ursprung verschieben
    trans_mat.makeTranslate(-transformVec);
    rev_mat.mult(old_dcs_mat, trans_mat);
    // erste handMat * aktualisierte handMat
    osg::Matrix oh = old_mat;
    oh.setTrans(0, 0, 0);
    osg::Matrix hm = handMat;
    hm.setTrans(0, 0, 0);
    rel_mat.mult(oh, hm);
    //osg::Quat qrot = rel_mat.getRotate();
    // nur rotieren nicht transformieren
    rel_mat.setTrans(0, 0, 0);
    dcs_mat.mult(rev_mat, rel_mat);
    // zurueck verschieben
    trans_mat.makeTranslate(transformVec);
    dcs_mat.mult(dcs_mat, trans_mat);
    // setzen der transformationsmatrix
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::doXformTranslate()
{
    osg::Matrix rel_mat, dcs_mat;
    rel_mat.mult(old_mat, handMat); //erste handMat * aktualisierte handMat
    rel_mat.makeTranslate(rel_mat.getTrans()[0], rel_mat.getTrans()[1] / 5.0, rel_mat.getTrans()[2]);
    dcs_mat.mult(old_dcs_mat, rel_mat);
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::stopXform()
{
    coVRCollaboration::instance()->UnSyncXform();
    if (wiiNav)
        wiiFlag = 0;
}

void coVRNavigationManager::startScale()
{
    // save start position
    startHandPos = handPos;
    startHandDir = handDir;

    oldDcsScaleFactor = VRSceneGraph::instance()->scaleFactor();

    old_xform_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
}

void coVRNavigationManager::doScale()
{

    float d = handMat(3, 0) - startHandPos[0];

    float dcsScaleFactor = 1.;
    if (d > 0)
        dcsScaleFactor = oldDcsScaleFactor * (1 + (10 * d / cover->getSceneSize()));
    else
        dcsScaleFactor = oldDcsScaleFactor / (1 + (10 * -d / cover->getSceneSize()));
    if (dcsScaleFactor < 0.0f)
        dcsScaleFactor = 0.1;

    /* move XformDCS to keep click position fixed */
    osg::Vec3 delta2 = old_xform_mat.getTrans();
    osg::Vec3 delta = delta2 - startHandPos;
    osg::Vec3 rot;
    if (rotationPoint)
    {
        rot = rotPointVec + cover->getBaseMat().getTrans();
        delta = delta2 - rot;
    }

    delta *= (dcsScaleFactor / oldDcsScaleFactor);
    if (rotationPoint)
        delta += rot;
    else
        delta += startHandPos;
    delta -= delta2;
    osg::Matrix tmp;
    tmp.makeTranslate(delta[0], delta[1], delta[2]);
    osg::Matrix xform_mat = old_xform_mat * tmp;

    VRSceneGraph::instance()->getTransform()->setMatrix(xform_mat);
    VRSceneGraph::instance()->setScaleFactor(dcsScaleFactor);
    coVRCollaboration::instance()->SyncXform();
    coVRCollaboration::instance()->SyncScale();
}

void coVRNavigationManager::stopScale()
{
    coVRCollaboration::instance()->UnSyncXform();
    coVRCollaboration::instance()->UnSyncScale();
}

void coVRNavigationManager::startFly()
{
    old_mat = handMat;
    startHandPos = handPos;
    startHandDir = handDir;
    ignoreCollision = false;
}

void coVRNavigationManager::doFly()
{
    osg::Vec3 dirAxis, handRight, oldHandRight, rollAxis;
    float dirAngle, rollAngle, rollScale;
    osg::Matrix dcs_mat, rot_mat;
    osg::Vec3 delta = startHandPos - handPos;

    /* calc axis and angle for direction change (from hand direction change) */
    dirAxis = handDir ^ startHandDir;
    dirAngle = dirAxis.length() * cover->frameDuration() * rotationSpeed;

    /* calc axis and angle for roll change (from hand roll change) */
    handRight.set(handMat(0, 0), handMat(0, 1), handMat(0, 2));
    oldHandRight.set(old_mat(0, 0), old_mat(0, 1), old_mat(0, 2));
    rollAxis = handRight ^ oldHandRight;
    rollScale = fabs(rollAxis * (handDir));
    rollAxis = rollAxis * rollScale;
    rollAngle = rollAxis.length() * cover->frameDuration() * 50;

    /* get xform matrix, translate to make viewPos the origin */
    dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    osg::Matrix tmp;
    osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
    tmp.makeTranslate(-viewerPos[0], -viewerPos[1], -viewerPos[2]);
    dcs_mat.postMult(tmp);

    /* apply translation */
    dcs_mat.postMult(applySpeedFactor(delta));

    /* apply direction change */
    if ((dirAxis[0] != 0.0) || (dirAxis[1] != 0.0) || (dirAxis[2] != 0.0))
    {
        rot_mat.makeRotate(dirAngle * 3 * M_PI / 180, dirAxis[0], dirAxis[1], dirAxis[2]);
        dcs_mat.mult(dcs_mat, rot_mat);
    }

    /* apply roll change */
    if ((rollAxis[0] != 0.0) || (rollAxis[1] != 0.0) || (rollAxis[2] != 0.0))
    {
        rot_mat.makeRotate(rollAngle * M_PI / 180, rollAxis[0], rollAxis[1], rollAxis[2]);
        dcs_mat.mult(dcs_mat, rot_mat);
    }

    /* undo viewPos translation, set new xform matrix */
    dcs_mat.postMult(osg::Matrix::translate(viewerPos[0], viewerPos[1], viewerPos[2]));
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::stopFly()
{
    coVRCollaboration::instance()->UnSyncXform();
}

void coVRNavigationManager::startDrive()
{
    old_mat = handMat;
    startHandPos = handPos;
    startHandDir = handDir;
    ignoreCollision = false;
}

double coVRNavigationManager::speedFactor(double delta) const
{
    /*
   return pow(navExp, fmax(fabs(delta),0.0)/8.0)
       * delta * driveSpeed * cover->frameDuration();
       */
    return sign(delta) * pow((double)(fabs(delta) / 40.0), (double)navExp) * 40.0
           * driveSpeed * (cover->frameDuration() > 0.05 ? 0.05 : cover->frameDuration());
}

osg::Vec3 coVRNavigationManager::applySpeedFactor(osg::Vec3 vec) const
{
    double l = vec.length();
    double s = speedFactor(l);
    vec.normalize();
    return vec * s;
}

void coVRNavigationManager::doDrive()
{
    /* calc delta vector for translation (hand position relative to click point) */
    osg::Vec3 delta = startHandPos - handPos;

    /* calc axis and angle for direction change (from hand direction change) */

    osg::Vec3 tmpv1 = handDir;
    osg::Vec3 tmpv2 = startHandDir;
    tmpv1[2] = 0.0;
    tmpv2[2] = 0.0;
    osg::Vec3 dirAxis = tmpv1 ^ tmpv2;
    float dirAngle = dirAxis.length() * cover->frameDuration() * rotationSpeed;

    /* get xform matrix, translate to make viewPos the origin */
    osg::Matrix dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
    dcs_mat.postMult(osg::Matrix::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2]));

    /* apply translation */
    dcs_mat.postMult(osg::Matrix::translate(applySpeedFactor(delta)));

    /* apply direction change */
    osg::Matrix rot_mat;
    if ((dirAxis[0] != 0.0) || (dirAxis[1] != 0.0) || (dirAxis[2] != 0.0))
    {
        rot_mat.makeRotate(dirAngle * 3 * M_PI / 180, dirAxis[0], dirAxis[1], dirAxis[2]);
        dcs_mat.mult(dcs_mat, rot_mat);
    }

    /* undo handPos translation, set new xform matrix */
    osg::Matrix tmp;
    tmp.makeTranslate(viewerPos[0], viewerPos[1], viewerPos[2]);
    dcs_mat.postMult(tmp);
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);

    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::stopDrive()
{
    coVRCollaboration::instance()->UnSyncXform();
}

void coVRNavigationManager::startWalk()
{
    startDrive();
    ignoreCollision = false;
    old_mat = handMat;

    /* get xform matrix */
    osg::Matrix TransformMat = VRSceneGraph::instance()->getTransform()->getMatrix();

    osg::Matrix viewer = cover->getViewerMat();
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
    osg::Matrix oldRot, newRot;
    coordOld.makeMat(oldRot);
    coord.makeMat(newRot);
    osg::Matrix diffRot = oldRot.inverse(oldRot) * newRot;
    osg::Vec3 viewerPos = viewer.getTrans();
    osg::Matrix vToOrigin;
    vToOrigin.makeTranslate(-viewerPos);
    osg::Matrix originToV;
    originToV.makeTranslate(viewerPos);

    /* set new xform matrix */
    VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat * vToOrigin * diffRot * originToV);
}

void coVRNavigationManager::doWalk()
{
    doDrive();
}

void coVRNavigationManager::doWalkMoveToFloor()
{
    float floorHeight = VRSceneGraph::instance()->floorHeight();

    //  just adjust height here

    osg::Matrix viewer = cover->getViewerMat();
    osg::Vec3 pos = viewer.getTrans();

    // down segment
    osg::Vec3 p0, q0;
    p0.set(pos[0], pos[1], floorHeight + stepSize);
    q0.set(pos[0], pos[1], floorHeight - stepSize);

    osg::ref_ptr<osg::LineSegment> ray = new osg::LineSegment();
    ray->set(p0, q0);

    // down segment 2
    p0.set(pos[0], pos[1] + 10, floorHeight + stepSize);
    q0.set(pos[0], pos[1] + 10, floorHeight - stepSize);
    osg::ref_ptr<osg::LineSegment> ray2 = new osg::LineSegment();
    ray2->set(p0, q0);

    osgUtil::IntersectVisitor visitor;
    visitor.setTraversalMask(Isect::Walk);
    visitor.addLineSegment(ray.get());
    visitor.addLineSegment(ray2.get());

    VRSceneGraph::instance()->getTransform()->accept(visitor);
    int num1 = visitor.getNumHits(ray.get());
    int num2 = visitor.getNumHits(ray2.get());
    if (num1 || num2)
    {
        osgUtil::Hit hitInformation1;
        osgUtil::Hit hitInformation2;
        if (num1)
            hitInformation1 = visitor.getHitList(ray.get()).front();
        if (num2)
            hitInformation2 = visitor.getHitList(ray2.get()).front();
        if (num1 || num2)
        {
            float dist = 0.0;
			osg::Node *floorNode = NULL;
			osgUtil::Hit *usedHI = NULL;
			if (num1 && !num2)
			{
				dist = hitInformation1.getWorldIntersectPoint()[2] - floorHeight;
				floorNode = hitInformation1.getGeode();
				usedHI = &hitInformation1;

			}
			else if (!num1 && num2)
			{
				dist = hitInformation2.getWorldIntersectPoint()[2] - floorHeight;
				floorNode = hitInformation2.getGeode();
				usedHI = &hitInformation2;
			}
			else if (num1 && num2)
			{
				floorNode = hitInformation1.getGeode();
				dist = hitInformation1.getWorldIntersectPoint()[2] - floorHeight;
				usedHI = &hitInformation1;
				if (fabs(hitInformation2.getWorldIntersectPoint()[2] - floorHeight) < fabs(dist))
				{
					dist = hitInformation2.getWorldIntersectPoint()[2] - floorHeight;
					usedHI = &hitInformation2;
				}
            }

			//  get xform matrix
			osg::Matrix dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();

			if (floorNode == oldFloorNode)
			{
				// we are walking on the same object as last time so move with the object if it is moving
				osg::Matrix modelTransform;
				modelTransform.makeIdentity();
				int on = oldNodePath.size() - 1;
				bool notSamePath = false;
				for (int i = usedHI->getNodePath().size() - 1; i >= 0; i--)
				{
					osg::Node*n = usedHI->getNodePath()[i];
					if (n == cover->getObjectsRoot())
						break;
					osg::MatrixTransform *t = dynamic_cast<osg::MatrixTransform *>(n);
					if (t != NULL)
					{
						modelTransform = modelTransform * t->getMatrix();
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
					oldNodePath = usedHI->getNodePath();
				}
				if(modelTransform != oldFloorMatrix)
				{
					
					//osg::Matrix iT;
					osg::Matrix iS;
					osg::Matrix S;
					osg::Matrix imT;
					//iT.invert_4x4(dcs_mat);
					float sf = cover->getScale();
					S.makeScale(sf, sf, sf);
					sf = 1.0 / sf;
					iS.makeScale(sf, sf, sf);
					imT.invert_4x4(modelTransform);
					dcs_mat = iS *imT*oldFloorMatrix * S * dcs_mat;
					oldFloorMatrix = modelTransform;
					// set new xform matrix
					VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
					// now we have a new base matrix and we have to compute the floor height again, otherwise we will jump up and down
					VRSceneGraph::instance()->getTransform()->accept(visitor);

					int num1 = visitor.getNumHits(ray.get());
					int num2 = visitor.getNumHits(ray2.get());
					if (num1 || num2)
					{
						osgUtil::Hit hitInformation1;
						osgUtil::Hit hitInformation2;
						if (num1)
							hitInformation1 = visitor.getHitList(ray.get()).front();
						if (num2)
							hitInformation2 = visitor.getHitList(ray2.get()).front();
						if (num1 || num2)
						{
							float dist = 0.0;
							osg::Node *floorNode = NULL;
							osgUtil::Hit *usedHI = NULL;
							if (num1 && !num2)
							{
								dist = hitInformation1.getWorldIntersectPoint()[2] - floorHeight;
								floorNode = hitInformation1.getGeode();
								usedHI = &hitInformation1;

							}
							else if (!num1 && num2)
							{
								dist = hitInformation2.getWorldIntersectPoint()[2] - floorHeight;
								floorNode = hitInformation2.getGeode();
								usedHI = &hitInformation2;
							}
							else if (num1 && num2)
							{
								floorNode = hitInformation1.getGeode();
								dist = hitInformation1.getWorldIntersectPoint()[2] - floorHeight;
								usedHI = &hitInformation1;
								if (fabs(hitInformation2.getWorldIntersectPoint()[2] - floorHeight) < fabs(dist))
								{
									dist = hitInformation2.getWorldIntersectPoint()[2] - floorHeight;
									usedHI = &hitInformation2;
								}
							}
						}
					}
				}

				
			}


            //  apply translation , so that isectPt is at floorLevel
            osg::Matrix tmp;
            tmp.makeTranslate(0, 0, -dist);
            dcs_mat.postMult(tmp);

            // set new xform matrix
            VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);


			if ((floorNode != oldFloorNode) && (usedHI != NULL))
			{
				osg::Matrix modelTransform;
				modelTransform.makeIdentity();
				for (int i = usedHI->getNodePath().size() - 1; i >= 0; i--)
				{
					osg::Node*n = usedHI->getNodePath()[i];
					if (n == cover->getObjectsRoot())
						break;
					osg::MatrixTransform *t = dynamic_cast<osg::MatrixTransform *>(n);
					if (t != NULL)
					{
						modelTransform = modelTransform * t->getMatrix();
					}
				}
				oldFloorMatrix = modelTransform;
				oldNodePath = usedHI->getNodePath();
			}

			oldFloorNode = floorNode;

            // do not sync with remote, they will do the same
            // on their side SyncXform();
        }
    }
	else
	{
		oldFloorNode = NULL;
	}
	
    // coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::stopWalk()
{
    stopDrive();
}

void coVRNavigationManager::startShowName()
{
    //fprintf(stderr, "coVRNavigationManager::startShowName\n");
}

void coVRNavigationManager::highlightSelectedNode(osg::Node *selectedNode)
{

    if (selectedNode != oldSelectedNode_)
    {
        coVRSelectionManager::instance()->clearSelection();
        if (selectedNode && (selectedNode->getNumParents() > 0))
        {
            coVRSelectionManager::instance()->setSelectionWire(0);
            coVRSelectionManager::instance()->setSelectionColor(1, 0, 0);
            coVRSelectionManager::instance()->addSelection(selectedNode->getParent(0), selectedNode);
            coVRSelectionManager::instance()->pickedObjChanged();
        }
        oldSelectedNode_ = selectedNode;
    }
}

void coVRNavigationManager::doShowName()
{
    // get the intersected node
    if (cover->getIntersectedNode())
    {
        // position label with name
        nameLabel_->setPosition(cover->getIntersectionHitPointWorld());
        if (cover->getIntersectedNode() != oldShowNamesNode_)
        {
            // get the node name
            string nodeName;
            //char *labelName = NULL;

            // show only node names under objects root
            // so check if node is under objects root
            Node *currentNode = cover->getIntersectedNode();
            while (currentNode != NULL)
            {
                if (currentNode == cover->getObjectsRoot())
                {
                    if (showGeodeName_)
                    {
                        // first look for a node description beginning with _SCGR_
                        std::vector<std::string> dl = cover->getIntersectedNode()->getDescriptions();
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
                            nodeName = cover->getIntersectedNode()->getName();
                            //fprintf(stderr,"taking the node name %s\n", nodeName.c_str());
                        }
                    }
                    else // show name of the dcs above
                    {
                        osg::Node *parentDcs = cover->getIntersectedNode()->getParent(0);
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
                    nodeName = coTranslator::coTranslate(nodeName);

                    //now remove special names
                    /*
               if (std::string::npos != nodeName.find(std::string("FACES")))
               {
                   nodeName = std::string("-");
               }
               */

                    nameLabel_->setString(nodeName.c_str());
                    nameLabel_->show();

                    nameMenu_->setVisible(true);
                    nameButton_->setLabel(nodeName);

                    // delete []labelName;
                    oldShowNamesNode_ = cover->getIntersectedNode();
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
        }
    }
    else
    {
        oldShowNamesNode_ = NULL;
        nameLabel_->hide();
        nameMenu_->setVisible(false);
        nameButton_->setLabel("-");
    }
}

void coVRNavigationManager::stopShowName()
{
    // remove label and highlight
    nameLabel_->hide();
    nameMenu_->setVisible(false);
}

void coVRNavigationManager::startMeasure()
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
        coMeasurement *measurement = new coMeasurement();
        measurement->start();
        measurements.push_back(measurement);
    }
}

void coVRNavigationManager::doMeasure()
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

void coVRNavigationManager::stopMeasure()
{
}

void coVRNavigationManager::toggleSelectInteract(bool state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRNavigationManager::toggleSelectInteract %d\n", state);

    if (state)
    {
        // enable intersection with scene
        coIntersection::instance()->isectAllNodes(true);
    }
    else
    {
        if (cover->debugLevel(4))
            fprintf(stderr, "realtoggle\n");
        highlightSelectedNode(NULL);
        // disable intersection with scene
        coIntersection::instance()->isectAllNodes(false);
    }
}

void coVRNavigationManager::startSelectInteract()
{
    //fprintf(stderr, "coVRNavigationManager::startSelectInteract\n");
    animationWasRunning = coVRAnimationManager::instance()->animationRunning();
    coVRAnimationManager::instance()->enableAnimation(false);
}

void coVRNavigationManager::doSelectInteract()
{
}

void coVRNavigationManager::stopSelectInteract(bool mouse)
{
    coVRAnimationManager::instance()->enableAnimation(animationWasRunning);

    if (!cover->getIntersectedNode())
        return;

    bool changeToInteraction = false;
    Node *currentNode = cover->getIntersectedNode();
    while (currentNode != NULL)
    {
        if (auto ud = currentNode->getUserData())
        {
            if (auto ir = dynamic_cast<InteractorReference *>(ud))
            {
                auto inter = ir->interactor();
                if (inter)
                {
                    std::cerr << "doSelectInteract: requesting interaction for node " << currentNode->getName() << std::endl;
                    changeToInteraction = coVRPluginList::instance()->requestInteraction(inter, cover->getIntersectedNode(), mouse);
                    if (changeToInteraction)
                        break;
                }
            }
        }
        if (currentNode == cover->getObjectsRoot())
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
    }
}

void coVRNavigationManager::wasJumping()
{
    jump = true;
}

void coVRNavigationManager::setDriveSpeed(float speed)
{
    driveSpeed = speed;
    driveSpeedSlider_->setValue(speed);
}

float coVRNavigationManager::getDriveSpeed()
{
    return driveSpeed;
}

bool coVRNavigationManager::isSnapping() const
{
    return snapping;
}

void coVRNavigationManager::enableSnapping(bool val)
{
    snapping = val;
}

bool coVRNavigationManager::isDegreeSnapping() const
{
    return snappingD;
}

void coVRNavigationManager::enableDegreeSnapping(bool val, float degrees)
{
    snapDegrees = degrees;
    snappingD = val;
}

bool coVRNavigationManager::restrictOn() const
{
    return m_restrict;
}

float coVRNavigationManager::snappingDegrees() const
{
    return snapDegrees;
}

void coVRNavigationManager::setStepSize(float stepsize)
{
    stepSize = stepsize;
}

float coVRNavigationManager::getStepSize() const
{
    return stepSize;
}

void coVRNavigationManager::processHotKeys(int keymask)
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
        VRSceneGraph::instance()->viewAll(false);
    }
}

void coVRNavigationManager::setRotationPoint(float x, float y, float z, float size)
{
    //fprintf(stderr, "coVRNavigationManager::setRotationPoint x=%f y=%f z=%f rad=%f\n",x,y,z,size);

    rotationPoint = true;
    rotPointVec = Vec3(x, y, z);
    osg::Matrix m;
    m.makeScale(size, size, size);
    m.setTrans(rotPointVec);
    rotPoint->setMatrix(m);
}

void coVRNavigationManager::setRotationPointVisible(bool visible)
{
    //fprintf(stderr, "coVRNavigationManager::setRotationPointVisibl %d\n", visible);
    rotationPointVisible = visible;
    if (rotationPointVisible)
        VRSceneGraph::instance()->objectsRoot()->addChild(rotPoint);
    else
        VRSceneGraph::instance()->objectsRoot()->removeChild(rotPoint);
}

void coVRNavigationManager::setRotationAxis(float x, float y, float z)
{
    rotationAxis = true;
    rotationVec = Vec3(x, y, z);
}

void coVRNavigationManager::doGuiRotate(float x, float y, float z)
{
    osg::Matrix rel_mat, dcs_mat;
    osg::Matrix rev_mat, rotx, roty, rotz, trans_mat;
    float factor = osg::inDegrees(5.0);
    old_dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
    // zurueck in rotationspunkt verschieben
    if (!rotationPoint && !rotationAxis)
    {
        transformVec = cover->getBBox(VRSceneGraph::instance()->getTransform()).center();
        if (coCoviseConfig::isOn("COVER.ScaleWithInteractors", false))
            transformVec = cover->getBBox(VRSceneGraph::instance()->getScaleTransform()).center();
    }
    else
    {
        transformVec = rotPointVec * cover->getBaseMat();
    }
    if (!rotationAxis)
    {
        trans_mat.makeTranslate(-transformVec);
        rev_mat.mult(old_dcs_mat, trans_mat);
        // rotationsmatrix
        rotx.makeRotate(x * factor, 1.0, 0.0, 0.0);
        roty.makeRotate(y * factor, 0.0, 1.0, 0.0);
        rotz.makeRotate(z * factor, 0.0, 0.0, 1.0);
        rel_mat = rotx * roty * rotz;
        dcs_mat.mult(rev_mat, rel_mat);
        // zurueck verschieben
        trans_mat.makeTranslate(transformVec);
        dcs_mat.mult(dcs_mat, trans_mat);
    }
    else
    {
        rel_mat.makeRotate(x * factor, rotationVec);
        dcs_mat.mult(old_dcs_mat, rel_mat);
    }
    // setzen der transformationsmatrix
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRNavigationManager::doGuiTranslate(float x, float y, float z)
{
    osg::Matrix dcs_mat;
    osg::Matrix doTrans;

    float sceneSize = cover->getSceneSize();
    float scale = 0.05f; // * sqrt(VRSceneGraph::instance()->getScaleTransform()->getMatrix().getScale()[0]);

    float translatefac = sceneSize * scale;
    if (guiTranslateFactor > 0)
        translatefac = translatefac * guiTranslateFactor;

    doTrans.makeTranslate(x * translatefac, y * translatefac, z * translatefac);
    dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();

    dcs_mat = dcs_mat * doTrans;
    VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
}
