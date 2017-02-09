/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRAnimationManager.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRLighting.h>

#include "AKToolbar.h"

#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coLabelSubMenuToolboxItem.h>
#include <OpenVRUI/coIconButtonToolboxItem.h>
#include <OpenVRUI/coIconSubMenuToolboxItem.h>
#include <OpenVRUI/coSliderToolboxItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coIconToggleButtonToolboxItem.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osgUtil/CullVisitor>

#include "AkSubMenu.h"

using namespace osg;
using covise::coCoviseConfig;

AKToolbar *AKToolbar::plugin = NULL;
int AKToolbar::debugLevel_ = 0;

static int numLoops = 0;
static const int NUM_PRELOOP = 20; // loop 20 times before starting up
// need to be sure that the vrui menu is loaded

void AKToolbar::addShortcut(const char *name, string command, bool toggleButton, const char *plugin)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "AKToolbar::addShortcut %s -> %s\n", name, command.c_str());
    char filename[PATH_MAX];
    strcpy(filename, "AKToolbar/");
    strcat(filename, name);
    coToolboxMenuItem *button;
    if (toggleButton)
    {
        button = new coIconToggleButtonToolboxItem(filename);
    }
    else
        button = new coIconButtonToolboxItem(filename);

    if (button)
    {
        // make sure we are informed about clicks
        button->setMenuListener(this);

        // remember command for this button
        shortcutMap_[button].command = command;

        if (!plugin || cover->getPlugin(plugin))
        {
            // CoviseConfig delivers back-to-front, so whe put in at front
            akToolbar_->insert(button, 0);
        }

        // remember command for this button
        shortcutMap_[button].command = command;
        shortcutMap_[button].button = button;

        if (plugin)
            shortcutMap_[button].plugin = plugin;
    }

    shortcutList_.push_back(&shortcutMap_[button]);
}

AKToolbar::AKToolbar()
{
}

bool AKToolbar::init()
{
    if (plugin)
        return false;

    animSlider_ = NULL;

    coverMenuClone_ = NULL;

    if (cover->debugLevel(1))
        fprintf(stderr, "\n--- Initialize plugin AKToolbar\n");

    oldStencil = false;

    defaultStencil = new Stencil();
    // draw only where stencil bit 1 is not set to 1
    defaultStencil->setFunction(Stencil::NOTEQUAL, 1, 1);
    // and don't touch the buffer
    defaultStencil->setOperation(Stencil::KEEP, Stencil::KEEP, Stencil::KEEP);

    menuStencil = new Stencil();
    menuStencil->setFunction(Stencil::ALWAYS, 1, 1);
    menuStencil->setOperation(Stencil::REPLACE, Stencil::REPLACE, Stencil::REPLACE);

    pointerStencil = new Stencil();
    pointerStencil->setFunction(Stencil::ALWAYS, 1, 1);
    pointerStencil->setOperation(Stencil::KEEP, Stencil::KEEP, Stencil::KEEP);

    /// Objects never intersect @@@@@@@@@@@@ Change!
    ///cover->getObjectsRoot()->setTravFuncs(PFTRAV_ISECT,stopTraversal,NULL);

    // AK Toolbar
    akToolbar_ = new coToolboxMenu("AK-VR");
    cover->setToolBar(akToolbar_);

    lastEventTime_ = cover->frameTime();

    // Shortcut buttons
    coCoviseConfig::ScopeEntries shortcutEntries = coCoviseConfig::getScopeEntries("COVER.Plugin.AKToolbar", "ShortCut");
    const char **shortcut = shortcutEntries.getValue();
    while (shortcut && *shortcut)
    {
        std::string entry("COVER.Plugin.AKToolbar.");
        entry += *shortcut;
        ++shortcut;
        if (!*shortcut)
            break;
        string value = coCoviseConfig::getEntry(entry);
        string plugin = coCoviseConfig::getEntry("plugin", entry);

        if (plugin.empty())
        {
            if (0 == strcasecmp(*shortcut, "viewall"))
                addShortcut("viewall", "ViewAll");
            else if (0 == strcasecmp(*shortcut, "xform"))
                addShortcut("xform", "XForm", true);
            else if (0 == strcasecmp(*shortcut, "scale"))
                addShortcut("scale", "Scale", true);
            else if (0 == strcasecmp(*shortcut, "drive"))
                addShortcut("drive", "Drive", true);
            else if (0 == strcasecmp(*shortcut, "fly"))
                addShortcut("fly", "Fly", true);
            else if (0 == strcasecmp(*shortcut, "walk"))
                addShortcut("walk", "Walk", true);
            else if (0 == strcasecmp(*shortcut, "remove"))
                addShortcut("remove", "Remove");
            else if (0 == strcasecmp(*shortcut, "undo"))
                addShortcut("undo", "Undo");
            else if (0 == strcasecmp(*shortcut, "scalePlus"))
                addShortcut("scalePlus", "ScalePlus");
            else if (0 == strcasecmp(*shortcut, "scaleMinus"))
                addShortcut("scaleMinus", "ScaleMinus");
            else if (0 == strcasecmp(*shortcut, "showName"))
                addShortcut("showName", "ShowName", true);
            else if (0 == strcasecmp(*shortcut, "quit"))
                addShortcut("quit", "Quit", true);
            else if (0 == strcasecmp(*shortcut, "toggleAnimation"))
                addShortcut("toggleAnimation", "ToggleAnimation", true);
            else if (0 == strcasecmp(*shortcut, "traverseInteractors"))
                addShortcut("traverseInteractors", "TraverseInteractors", true);
            else if (0 == strcasecmp(*shortcut, "stereoSeparation"))
                addShortcut("stereoSeparation", "StereoSeparation", true);
            else if (0 == strcasecmp(*shortcut, "orthographicProjection"))
                addShortcut("orthographicProjection", "OrthographicProjection", true);
            else if (0 == strcasecmp(*shortcut, "headlight"))
                addShortcut("headlight", "headlight", true);
            else if (0 == strcasecmp(*shortcut, "otherlight"))
                addShortcut("otherlight", "otherlight", true);
            else if (0 == strcasecmp(*shortcut, "spotlight"))
                addShortcut("spotlight", "spotlight", true);
            else if (0 == strcasecmp(*shortcut, "specularlight"))
                addShortcut("specularlight", "specularlight", true);
        }
        else
        {
            string command = coCoviseConfig::getEntry("command", entry);
            if (command.empty())
                command = value;
            string icon = coCoviseConfig::getEntry("icon", entry);
            bool toggleButton = coCoviseConfig::isOn(string("toggleButton"), entry, false);
            addShortcut(icon.c_str(), command, toggleButton, plugin.c_str());
        }
        ++shortcut;
    }

    // animation slider
    if (coCoviseConfig::isOn("COVER.Plugin.AKToolbar.MapAnimationSlider", false))
    {
        animSlider_ = new coSliderToolboxItem("[Time Step]", 1, 1, 1);
        animSlider_->setMenuListener(this);
    }

    // Status button
    stateButton_ = new coIconButtonToolboxItem("AKToolbar/Status");
    if (coCoviseConfig::isOn("COVER.Plugin.AKToolbar.StateButton", true)
        && OpenCOVER::instance()->visPlugin())
        akToolbar_->add(stateButton_);
    stateButton_->setMenuListener(this);
    stateBusy_ = false;
    stateButton_->setActive(!stateBusy_);

    // Debug Level
    debugLevel_ = coCoviseConfig::getInt("COVER.Plugin.AKToolbar.DebugLevel", 0);

    //////////////////////////////////////////////////////////
    //// Master Menu

    coverMenu_ = coCoviseConfig::isOn("COVER.Plugin.AKToolbar.CoverMenu", true);

    /// Clone the Master menu
    coMenu *coverMenu = cover->getMenu();

    coverMenuClone_ = new AkSubMenu(coverMenu, NULL, NULL, akToolbar_);

    if (coverMenu_)
    {
        coverMenuClone_->addToToolbar();
    }

    // prevent double clicks : no click valid in this period
    minTimeBetweenClicks_ = coCoviseConfig::getFloat("COVER.Plugin.AKToolbar.MinClickTime", 0.5);

    //////////////////////////////////////////////////////////
    // position AK-Toolbar and make it visible
    float x, y, z, h, p, r, scale;

    x = coCoviseConfig::getFloat("x", "COVER.Plugin.AKToolbar.Position", -100);
    y = coCoviseConfig::getFloat("y", "COVER.Plugin.AKToolbar.Position", 20);
    z = coCoviseConfig::getFloat("z", "COVER.Plugin.AKToolbar.Position", -50);

    h = coCoviseConfig::getFloat("h", "COVER.Plugin.AKToolbar.Orientation", 0);
    p = coCoviseConfig::getFloat("p", "COVER.Plugin.AKToolbar.Orientation", 90);
    r = coCoviseConfig::getFloat("r", "COVER.Plugin.AKToolbar.Orientation", 0);

    scale = coCoviseConfig::getFloat("COVER.Plugin.AKToolbar.Scale", 0.2);

    int attachment = coUIElement::TOP;
    std::string att = coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
    if (att != "")
    {
        if (!strcasecmp(att.c_str(), "BOTTOM"))
        {
            attachment = coUIElement::BOTTOM;
        }
        else if (!strcasecmp(att.c_str(), "LEFT"))
        {
            attachment = coUIElement::LEFT;
        }
        else if (!strcasecmp(att.c_str(), "RIGHT"))
        {
            attachment = coUIElement::RIGHT;
        }
    }
    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->setAttachment(attachment);

    //float sceneSize = cover->getSceneSize();

    vruiMatrix *mat = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rot = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *trans = vruiRendererInterface::the()->createMatrix();

    rot->makeEuler(h, p, r);
    trans->makeTranslate(x, y, z);
    mat->makeIdentity();
    mat->mult(rot);
    mat->mult(trans);
    akToolbar_->setTransformMatrix(mat);
    akToolbar_->setScale(scale);
    akToolbar_->setVisible(true);
    akToolbar_->fixPos(true);
    akToolbar_->setAttachment(attachment);

    akToolbar_->setMenuListener(this);

    coverMenu->setVisible(false);

    return true;
}

// this is called if the plugin is removed at runtime
AKToolbar::~AKToolbar()
{
    for (ShortcutMap::iterator it = shortcutMap_.begin();
         it != shortcutMap_.end();
         ++it)
    {
        it->second.button->setMenuListener(NULL);
        delete it->second.button;
        it->second.button = NULL;
    }

    //crashes - need to fix the crash
    //not needed for visenso
    //coMenu *coverMenu = cover->getMenu();
    //if(coverMenu)
    //   coverMenu->setVisible(true);
    delete coverMenuClone_;
    coverMenuClone_ = NULL;
    delete animSlider_;
    animSlider_ = NULL;
    delete stateButton_;
    stateButton_ = NULL;
    delete akToolbar_;
    akToolbar_ = NULL;
}

void AKToolbar::focusEvent(bool /*focus*/, coMenu *)
{

    if (cover->debugLevel(6))
    {
        /*  if (focus)
           fprintf(stderr,"AKToolbar::focusEvent: Got Focus");
        else
           fprintf(stderr,"AKToolbar::focusEvent: Lost Focus");*/
    }

    //VRPinboard::mainPinboard->lockInteraction = focus;
}

void AKToolbar::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == stateButton_)
    {
        coVRPluginList::instance()->executeAll();
    }
    if (menuItem == animSlider_)
    {
        coVRAnimationManager::instance()->requestAnimationTime(animSlider_->getValue());
        return;
    }

    // prevent too frequent clicks
    double dblCurrTime = cover->frameTime();
    if (dblCurrTime - lastEventTime_ < minTimeBetweenClicks_)
        return;
    else
        lastEventTime_ = dblCurrTime;

    ShortcutMap::iterator iter = shortcutMap_.find(menuItem);
    if (iter != shortcutMap_.end())
    {
        std::string &command = iter->second.command;
        if (command == "ViewAll" || command == "Undo"
            || command == "ScalePlus" || command == "ScaleMinus" || command == "Quit") // does not work, fake ourself
        {
            cover->callBuiltInFunctionCallback(iter->second.command.c_str());
        }
        else if (command == "headlight")
        {
            if (coVRLighting::instance()->switchHeadlight_)
            {
                coVRLighting::instance()->switchHeadlight_->setState(!coVRLighting::instance()->switchHeadlight_->getState());
                coVRLighting::instance()->menuEvent(coVRLighting::instance()->switchHeadlight_);
            }
        }
        else if (command == "otherlight")
        {
            if (coVRLighting::instance()->switchOtherlights_)
            {
                coVRLighting::instance()->switchOtherlights_->setState(!coVRLighting::instance()->switchOtherlights_->getState());
                coVRLighting::instance()->menuEvent(coVRLighting::instance()->switchOtherlights_);
            }
        }
        else if (command == "spotlight")
        {
            if (coVRLighting::instance()->switchSpotlight_)
            {
                coVRLighting::instance()->switchSpotlight_->setState(!coVRLighting::instance()->switchSpotlight_->getState());
                coVRLighting::instance()->menuEvent(coVRLighting::instance()->switchSpotlight_);
            }
        }
        else if (command == "specularlight")
        {
            if (coVRLighting::instance()->switchSpecularlight_)
            {
                coVRLighting::instance()->switchSpecularlight_->setState(!coVRLighting::instance()->switchSpecularlight_->getState());
                coVRLighting::instance()->menuEvent(coVRLighting::instance()->switchSpecularlight_);
            }
        }
        else
        {
            coMenuItem *item = NULL;
            if (iter->second.observed)
                item = iter->second.observed;
            else
                item = cover->getBuiltInFunctionMenuItem(iter->second.command.c_str());
            if (coCheckboxMenuItem *cb = dynamic_cast<coCheckboxMenuItem *>(item))
            {
                bool oldState = cb->getState();
                cb->setState(!oldState, true, true);
            }
            else if (coButtonMenuItem *bt = dynamic_cast<coButtonMenuItem *>(item))
            {
                bt->doActionRelease();
                if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Menu)
                    coVRNavigationManager::instance()->setMenuMode(false);
            }
        }
    }
}

void AKToolbar::message(int type, int, const void *)
{
    stateBusy_ = (type != 0);
    stateButton_->setActive(!stateBusy_);
}

void AKToolbar::updateAnimationSlider()
{
    if (!animSlider_)
        return;

    coVRAnimationManager *am = coVRAnimationManager::instance();
    if (am->getNumTimesteps() > 1)
    {
        if (akToolbar_->index(animSlider_) == -1)
            akToolbar_->insert(animSlider_, 0);
    }
    else
    {
        akToolbar_->remove(animSlider_);
    }
    double base = am->getTimestepBase();
    double scale = am->getTimestepScale();
    animSlider_->setMin(base);
    animSlider_->setMax(base + (am->getNumTimesteps() - 1) * scale);
    animSlider_->setValue(base + am->getAnimationFrame() * scale);
    animSlider_->setLabel(am->getTimestepUnit());
}

void AKToolbar::updatePlugins()
{
    updateAnimationSlider();

    for (size_t i = 0; i < shortcutList_.size(); ++i)
    {
        Shortcut *sc = shortcutList_[i];
        const std::string &plugin = sc->plugin;
        if (plugin.empty() && sc->command == "ToggleAnimation")
        {
            if (coVRAnimationManager::instance()->getNumTimesteps() > 1)
            {
                sc->observed = coVRAnimationManager::instance()->getMenuButton(sc->command);
            }
            else
            {
                sc->observed = NULL;
            }
            if (sc->observed)
            {
                if (akToolbar_->index(sc->button) == -1)
                {
                    int pos = 0;
                    if (akToolbar_->index(animSlider_) != -1)
                        pos = akToolbar_->index(animSlider_) - 1;
                    akToolbar_->insert(sc->button, pos);
                }
            }
            else
            {
                akToolbar_->remove(sc->button);
            }
        }
        else if (!plugin.empty())
        {
            if (cover->getPlugin(plugin.c_str()))
            {
                sc->observed = cover->getPlugin(plugin.c_str())->getMenuButton(sc->command);
            }
            else
            {
                sc->observed = NULL;
            }
            if (sc->observed)
            {
                if (akToolbar_->index(sc->button) == -1)
                {
                    int pos = 0;
                    if (akToolbar_->index(animSlider_) != -1)
                        pos = akToolbar_->index(animSlider_) + 2;
                    akToolbar_->insert(sc->button, pos);
                }
            }
            else
            {
                akToolbar_->remove(sc->button);
            }
        }
    }
}

void AKToolbar::preFrame()
{
    /// wait NUM_PRELOOP preloop frames and then start up Plugin
    if (!AKToolbar::plugin)
    {
        if (numLoops++ < NUM_PRELOOP)
        {
            return;
        }
        numLoops++;
        AKToolbar::plugin = this;
    }

    // do not use stencil for aktoolbar if stencil-buffer is used for stipple
    bool stipple = false;
    for (int i = 0; i < coVRConfig::instance()->numScreens(); i++)
    {
        stipple = coVRConfig::instance()->channels[i].stereoMode == osg::DisplaySettings::VERTICAL_INTERLACE || coVRConfig::instance()->channels[i].stereoMode == osg::DisplaySettings::HORIZONTAL_INTERLACE || coVRConfig::instance()->channels[i].stereoMode == osg::DisplaySettings::CHECKERBOARD;
        if (stipple)
            break;
    }

    if (coVRConfig::instance()->stencil() != oldStencil && !stipple && false)
    {
        oldStencil = coVRConfig::instance()->stencil();

        ref_ptr<StateSet> sceneStateSet = cover->getScene()->getOrCreateStateSet();
        ref_ptr<StateSet> menuStateSet = cover->getMenuGroup()->getOrCreateStateSet();
        ref_ptr<StateSet> pointerStateSet = cover->getPointer()->getOrCreateStateSet();

        depth = new Depth(Depth::ALWAYS);

        StateAttribute::GLModeValue value;

        if (coVRConfig::instance()->stencil())
        {
            printf("AKToolbar::preFrame info: turn on stenciling\n");
            value = StateAttribute::ON | StateAttribute::OVERRIDE;
        }
        else
        {
            printf("AKToolbar::preFrame info: turn off stenciling\n");
            value = StateAttribute::OFF;
        }

        sceneStateSet->setAttributeAndModes(defaultStencil.get(), value);
        pointerStateSet->setAttributeAndModes(pointerStencil.get(), value);
        pointerStateSet->setNestRenderBins(false);
        pointerStateSet->setRenderBinDetails(40, "RenderBin");
        menuStateSet->setAttributeAndModes(menuStencil.get(), value);
        menuStateSet->setAttributeAndModes(depth.get(), value);
        menuStateSet->setRenderBinDetails(20, "RenderBin");
        menuStateSet->setNestRenderBins(false);

        //FIXME Shouldn't be necessary
        cover->getScene()->setStateSet(sceneStateSet.get());
        cover->getMenuGroup()->setStateSet(menuStateSet.get());
        cover->getPointer()->setStateSet(pointerStateSet.get());
    }

    for (ShortcutMap::iterator iter = shortcutMap_.begin(); iter != shortcutMap_.end(); ++iter)
    {
        coCheckboxMenuItem *cb;
        if (iter->second.observed)
        {
            cb = dynamic_cast<coCheckboxMenuItem *>(iter->second.observed);
        }
        else
        {
            cb = dynamic_cast<coCheckboxMenuItem *>(cover->getBuiltInFunctionMenuItem(iter->second.command.c_str()));
        }
        if (cb)
        {
            coIconToggleButtonToolboxItem *tb = (coIconToggleButtonToolboxItem *)iter->first;
            if (cb->getState() != tb->getState())
                tb->setState(cb->getState());
        }
    }

    updatePlugins();
}

COVERPLUGIN(AKToolbar)
