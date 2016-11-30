/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Sequence>
#include <osgGA/GUIEventAdapter>

#include <config/CoviseConfig.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <net/message.h>
#include <net/message_types.h>
#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRAnimationManager.h"
#include "coVRCollaboration.h"
#include "coVRMSController.h"
#include "OpenCOVER.h"

#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>

using namespace vrui;
using namespace opencover;
using namespace grmsg;
using namespace covise;

coVRAnimationManager::coVRAnimationManager()
    : AnimSliderMin(-25.)
    , AnimSliderMax(25.)
    , timeState(AnimSliderMax) // slider value of animation slider
    , aniDirection(1)
    , numFrames(0)
    , startFrame(0)
    , stopFrame(0)
    , oldFrame(-1)
    , animButton(NULL)
    , animRunning(true)
    , lastAnimationUpdate(0.0)
    , currentAnimationFrame(0)
    , requestedAnimationFrame(-1)
    , timestepScale(1.0)
    , timestepBase(0.0)
    , timestepUnit("Time Step")
{
    initAnimMenu();
    animWheelInteraction = new coTrackerButtonInteraction(coInteraction::Wheel, "Animation", coInteraction::Low);
}

coVRAnimationManager::~coVRAnimationManager()
{
    delete animWheelInteraction;
    delete animSyncItem;
    delete rotateObjectsToggleItem;
    delete animToggleItem;
    delete animSpeedItem;
    delete animForwardItem;
    delete animBackItem;
    delete animFrameItem;
    delete animPingPongItem;
    delete animSyncItem;
    delete animRowMenu;
}

coVRAnimationManager *
coVRAnimationManager::instance()
{
    static coVRAnimationManager *singleton;
    if (!singleton)
        singleton = new coVRAnimationManager;
    return singleton;
}

void coVRAnimationManager::initAnimMenu()
{
    // Create animation menu:

    // anim speed was configurable in 5.2 where animSpeed was a slider
    // for the dial where min=-max we use the greater value as AnimSliderMax
    float animSpeedStartValue = AnimSliderMax * 0.96f; // 24 FPS for 25.0

    // Achtung nach diesem Fehler kann man lange suchen (dr):
    // wenn im config.xml dieser Abschnitt fehlt, dann werden nicht die oben definierten default werte genommen sondern die vom coVRTUI
    float min = coCoviseConfig::getFloat("min", "COVER.AnimationSpeed", AnimSliderMin);
    float max = coCoviseConfig::getFloat("max", "COVER.AnimationSpeed", AnimSliderMax);
    animSpeedStartValue = coCoviseConfig::getFloat("default", "COVER.AnimationSpeed", animSpeedStartValue);

    if (min > max)
        std::swap(min, max);
    AnimSliderMin = min;
    AnimSliderMax = max;

    animToggleItem = new coCheckboxMenuItem("Animate", true);
    animSpeedItem = new coPotiMenuItem("Speed", AnimSliderMin, AnimSliderMax, animSpeedStartValue);
    sendAnimationSpeedMessage();

    animForwardItem = new coButtonMenuItem("Step Forward");
    animBackItem = new coButtonMenuItem("Step Backward");
    animFrameItem = new coSliderMenuItem(timestepUnit, timestepBase, timestepBase, timestepBase);
    animPingPongItem = new coCheckboxMenuItem("Oscillate", false);
    animSyncItem = new coCheckboxMenuItem("Synchronize", false);
    animRowMenu = new coRowMenu("Animation");
    animToggleItem->setMenuListener(this);
    animPingPongItem->setMenuListener(this);
    animSpeedItem->setMenuListener(this);
    animForwardItem->setMenuListener(this);
    animBackItem->setMenuListener(this);
    animFrameItem->setMenuListener(this);
    animSyncItem->setMenuListener(this);

    animFrameItem->setInteger(true);
    animFrameItem->setNumTicks(1);
    animFrameItem->setMin(0);
    animFrameItem->setMax(0);

    animRowMenu->add(animToggleItem);
    animRowMenu->add(animSpeedItem);
    animRowMenu->add(animForwardItem);
    animRowMenu->add(animBackItem);
    animRowMenu->add(animFrameItem);
    animRowMenu->add(animPingPongItem);
    animRowMenu->add(animSyncItem);
    rotateObjectsToggleItem = new coCheckboxMenuItem("RotateObjects", false);
#if 0
   animRowMenu->add(rotateObjectsToggleItem);
#endif
}

void coVRAnimationManager::setOscillate(bool state)
{
    animPingPongItem->setState(state);
}

bool coVRAnimationManager::isOscillating() const
{
    return animPingPongItem->getState();
}

// process key events
bool coVRAnimationManager::keyEvent(int type, int keySym, int mod)
{
    bool handled = false;

    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (!(mod & osgGA::GUIEventAdapter::MODKEY_ALT))
        {
            if (keySym == 'a')
            {
                enableAnimation(!animationRunning());
            }
            else if (keySym == '.')
            {
                requestAnimationFrame(getAnimationFrame() + 1);
            }
            else if (keySym == ',')
            {
                requestAnimationFrame(getAnimationFrame() - 1);
            }
        }
    }
    return handled;
}

void coVRAnimationManager::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAnimationManager::menuEvent\n");

    if (menuItem == animSyncItem)
    {
        cover->sendBinMessage("TIMESTEP_SYNCRONIZE", animSyncItem->getState() ? "1" : "0", 2);
        cover->sendBinMessage("TIMESTEP_ANIMATE", animToggleItem->getState() ? "1" : "0", 2);
    }
    else if (menuItem == animToggleItem)
    {
        if (animRunning != animToggleItem->getState())
            enableAnimation(animToggleItem->getState());
    }
    else if (menuItem == animForwardItem || menuItem == animBackItem)
    {
        if (animRunning)
            enableAnimation(false);

        if (menuItem == animForwardItem)
            requestAnimationFrame(getAnimationFrame() + 1);
        else
            requestAnimationFrame(getAnimationFrame() - 1);
    }
    else if (menuItem == animFrameItem)
    {
       requestAnimationTime(animFrameItem->getValue());
    }
    else if (menuItem == animSpeedItem)
        sendAnimationSpeedMessage();
}

void
coVRAnimationManager::requestAnimationTime(double t)
{
    int step = static_cast<int>((t - timestepBase) / timestepScale + 0.5);
    requestAnimationFrame(step);
}

void
coVRAnimationManager::setRemoteAnimationFrame(int currentFrame)
{
    if (currentFrame >= 0 || currentFrame < numFrames)
    {
        currentAnimationFrame = currentFrame;
        for (unsigned int i = 0; i < listOfSeq.size(); i++)
        {
            unsigned int numChildren = listOfSeq[i]->getNumChildren();
            //listOfSeq[i]->setValue(((unsigned int)currentFrame) < numChildren ? currentFrame : numChildren - 1);
            listOfSeq[i]->setValue(currentFrame % numChildren);
        }
        coVRPluginList::instance()->setTimestep(currentFrame);
        if (animFrameItem && numFrames != 0)
            animFrameItem->setValue(currentFrame);
    }
}

void coVRAnimationManager::setRemoteAnimate(bool state)
{
    animRunning = state;
    animToggleItem->setState(state);
}
void coVRAnimationManager::setRemoteSynchronize(bool state)
{
    animSyncItem->setState(state);
}

void
coVRAnimationManager::requestAnimationFrame(int currentFrame)
{
    if ((currentFrame != oldFrame) && animSyncItem->getState())
    {
        if (animRunning)
        {
            if (coVRCollaboration::instance()->isMaster()) // send update ot others if we are the Master
            {
                char num[100];
                sprintf(num, "%d", currentFrame);
                cover->sendBinMessage("TIMESTEP", num, strlen(num) + 1);
            }
        }
        else
        {
            // send update to other users
            char num[100];
            sprintf(num, "%d", currentFrame);
            cover->sendBinMessage("TIMESTEP", num, strlen(num) + 1);
        }
        oldFrame = currentFrame;
    }

    if (numFrames == 0)
    {
        currentFrame = 0;
    }
    else
    {
        if (currentFrame < 0)
            currentFrame = (currentFrame % numFrames) + numFrames;
        if (numFrames > 0)
            currentFrame %= numFrames;
    }

    if (stopFrame >= startFrame)
        currentFrame = (currentFrame - startFrame + stopFrame - startFrame + 1) % (stopFrame - startFrame + 1) + startFrame;
    else
        currentFrame = startFrame;

    if (requestedAnimationFrame == -1)
    {
        requestedAnimationFrame = currentFrame;
        coVRPluginList::instance()->requestTimestep(currentFrame);
    }
}

void
coVRAnimationManager::setAnimationFrame(int currentFrame)
{
    if (requestedAnimationFrame != -1 && currentFrame != requestedAnimationFrame)
    {
        std::cerr << "setAnimationFrame(" << currentFrame << "), but " << requestedAnimationFrame << " was requested" << std::endl;
    }
    requestedAnimationFrame = -1;
    if (currentAnimationFrame != currentFrame)
    {
        currentAnimationFrame = currentFrame;
        for (unsigned int i = 0; i < listOfSeq.size(); i++)
        {
            unsigned int numChildren = listOfSeq[i]->getNumChildren();
            //listOfSeq[i]->setValue(((unsigned int)currentFrame) < numChildren ? currentFrame : numChildren - 1);
			if (numChildren>0)            
				listOfSeq[i]->setValue(currentFrame % numChildren);
        }
        coVRPluginList::instance()->setTimestep(currentFrame);
        if (animFrameItem && numFrames != 0)
            animFrameItem->setValue(timestepBase + timestepScale * currentFrame);
        lastAnimationUpdate = cover->frameTime();
        sendAnimationStepMessage();
    }
}

void
coVRAnimationManager::updateAnimationFrame()
{
    if (animRunning && (!animSyncItem->getState() || coVRCollaboration::instance()->isMaster()))
    {
        if (!animPingPongItem->getState()) // normal loop mode
        {
            aniDirection = 1;
        }
        else // oscillate mode
        {
            if (currentAnimationFrame >= stopFrame) // check for end of sequence
                aniDirection = -1;
            if (currentAnimationFrame <= startFrame) // check for start of sequence
                aniDirection = 1;
        }

        if (animSpeedItem->getValue() > 0.0)
        {
            if ((cover->frameTime() - lastAnimationUpdate > 1.0 / animSpeedItem->getValue())
                || (animSpeedItem->getValue() > AnimSliderMax - 0.001))
            {
                requestAnimationFrame(currentAnimationFrame + aniDirection);
            }
        }
        else if (animSpeedItem->getValue() < 0.0)
        {
            if ((cover->frameTime() - lastAnimationUpdate > -1.0 / animSpeedItem->getValue())
                || (animSpeedItem->getValue() < AnimSliderMin + 0.001))
            {
                requestAnimationFrame(currentAnimationFrame - aniDirection);
            }
        }
    }
    else
    {
        // wait for plugins to resolve recently requested timestep,
        // which might be different from currentAnimationFrame,
        // so don't: requestAnimationFrame(currentAnimationFrame);
    }
}

float
coVRAnimationManager::getAnimationSpeed()
{
    return animSpeedItem->getValue();
}

void
coVRAnimationManager::setAnimationSpeed(float speed)
{
    if (speed < animSpeedItem->getMin())
        speed = animSpeedItem->getMin();
    if (speed > animSpeedItem->getMax())
        speed = animSpeedItem->getMax();

    animSpeedItem->setValue(speed);
    sendAnimationSpeedMessage();
}

bool
coVRAnimationManager::animationRunning()
{
    return animRunning;
}

void
coVRAnimationManager::enableAnimation(bool state)
{
    animRunning = state;
    animToggleItem->setState(animRunning);
    cover->sendBinMessage("TIMESTEP_ANIMATE", animRunning ? "1" : "0", 2);
    sendAnimationStateMessage();
}

void
coVRAnimationManager::update()
{

    if (animWheelInteraction->wasStarted() || animWheelInteraction->isRunning())
    {
        if (animationRunning())
            enableAnimation(false);

        requestAnimationFrame(getAnimationFrame() + animWheelInteraction->getWheelCount());
    }
    else
    {
        // Set selected animation frame:
        updateAnimationFrame();
    }

#if 0
   // rotate world menu button is checked
   if(rotateObjectsToggleItem->getState())
   {
      static float oldAngle=0;
      float angle=currentFrame*frameAngle;
      float diffAngle = angle - oldAngle;
      oldAngle = angle;
      if(diffAngle>0.0001 || diffAngle < -0.0001)
      {
         // rotate world
         osg::Matrix mat = objectsXformDCS->getMatrix();
         mat.preMult(osg::Matrix::translate(-rotationPoint[0],
                  -rotationPoint[1],
                  -rotationPoint[2]));
         mat.preMult(osg::Matrix::rotate(diffAngle,
                  rotationAxis[0],
                  rotationAxis[1],
                  rotationAxis[2]));
         mat.preMult(osg::Matrix::translate(rotationPoint[0],
                  rotationPoint[1],
                  rotationPoint[2]));
         objectsXformDCS->setMatrix(mat);
      }
   }
#endif
}

void
coVRAnimationManager::forwardCallback(void *sceneGraph, buttonSpecCell *)
{
    coVRAnimationManager *sg = static_cast<coVRAnimationManager *>(sceneGraph);
    if (sg->animationRunning())
        sg->enableAnimation(false);
    sg->requestAnimationFrame(sg->getAnimationFrame() + 1);
}

void
coVRAnimationManager::backwardCallback(void *sceneGraph, buttonSpecCell *)
{
    coVRAnimationManager *sg = static_cast<coVRAnimationManager *>(sceneGraph);
    if (sg->animationRunning())
        sg->enableAnimation(false);
    sg->requestAnimationFrame(sg->getAnimationFrame() - 1);
}

void
coVRAnimationManager::remove_controls()
{
    showAnimMenu(false);
}

void coVRAnimationManager::setNumTimesteps(int t)
{
    numFrames = t;
    if (numFrames == 0)
        numFrames = 1;
    if (animFrameItem)
    {
        animFrameItem->setMax(timestepBase + (t - 1) * timestepScale);
        animFrameItem->setNumTicks(t - 1);
    }

    if (startFrame >= numFrames)
        startFrame = 0;
    stopFrame = numFrames - 1;

    if (numFrames > 1)
    {
        if (!animWheelInteraction->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(animWheelInteraction);
        }
    }
    else
    {
        if (animWheelInteraction->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(animWheelInteraction);
        }
    }
}

void
coVRAnimationManager::add_set_controls()
{
    showAnimMenu(true);
}

void coVRAnimationManager::showAnimMenu(bool visible)
{
    if (visible && animButton == NULL)
    {
        animButton = new coSubMenuItem("Animation...");
        animButton->setMenu(animRowMenu);
        cover->getMenu()->add(animButton);
    }
    else if (!visible && animButton != NULL)
    {
        animButton->closeSubmenu();
        delete animButton;
        animButton = NULL;
    }
}

void
coVRAnimationManager::removeSequence(osg::Sequence *seq)
{
    for (unsigned int i = 0; i < listOfSeq.size(); i++)
    {
        if (listOfSeq[i] == seq)
        {
            removeTimestepProvider(seq);
            for (unsigned int n = i; n < listOfSeq.size() - 1; n++)
                listOfSeq[n] = listOfSeq[n + 1];
            listOfSeq.pop_back();
            break;
        }
    }
}

const std::vector<osg::Sequence *>&
coVRAnimationManager::getSequences() const
{
    return listOfSeq;
}

void
coVRAnimationManager::addSequence(osg::Sequence *seq)
{
    /*if (int(seq->getNumChildren()) > currentAnimationFrame)
        seq->setValue(currentAnimationFrame);
    else
        seq->setValue(seq->getNumChildren());*/
    
    if (seq->getNumChildren()>0)
		seq->setValue(currentAnimationFrame % seq->getNumChildren());
    

    setNumTimesteps(seq->getNumChildren(), seq);
    bool alreadyAdded = false;
    for (unsigned int i = 0;
         i < listOfSeq.size();
         i++)
    {
        if (seq == listOfSeq[i])
        {
            alreadyAdded = true;
            break;
        }
    }
    if (!alreadyAdded)
    {
        listOfSeq.push_back(seq);
    }
}

void coVRAnimationManager::setStartFrame(int frame)
{
    startFrame = frame;
    if (startFrame >= numFrames)
        startFrame = numFrames - 1;
    if (startFrame < 0)
        startFrame = 0;
    if (startFrame > stopFrame)
        stopFrame = startFrame;
}

int coVRAnimationManager::getStartFrame() const
{
    return startFrame;
}

void coVRAnimationManager::setStopFrame(int frame)
{
    stopFrame = frame;
    if (stopFrame >= numFrames)
        stopFrame = numFrames - 1;
    if (stopFrame < 0)
        stopFrame = 0;
    if (startFrame > stopFrame)
        startFrame = stopFrame;
}

int coVRAnimationManager::getStopFrame() const
{
    return stopFrame;
}

// get number of timesteps
int coVRAnimationManager::getNumTimesteps()
{
    return numFrames;
}

// set number of timesteps
void coVRAnimationManager::setNumTimesteps(int t, const void *who)
{
    timestepMap[who] = t;

    int numTimesteps = 0;
    for (TimestepMap::const_iterator it = timestepMap.begin();
         it != timestepMap.end();
         it++)
    {
        if (it->second > numTimesteps)
        {
            numTimesteps = it->second;
        }
    }
    setNumTimesteps(numTimesteps);
    showAnimMenu(numTimesteps > 1);
}

void coVRAnimationManager::removeTimestepProvider(const void *who)
{
    setNumTimesteps(0, who);
    timestepMap.erase(who);
}

void coVRAnimationManager::sendAnimationStateMessage()
{
    // send animation mode to gui
    if (coVRMSController::instance()->isMaster())
    {
        coGRAnimationOnMsg animationModeMsg(animationRunning());
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(animationModeMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void coVRAnimationManager::sendAnimationSpeedMessage()
{
    // send animation speed to gui
    if (coVRMSController::instance()->isMaster())
    {

        coGRSetAnimationSpeedMsg animationSpeedMsg(getAnimationSpeed(), animSpeedItem->getMin(), animSpeedItem->getMax());
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(animationSpeedMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void coVRAnimationManager::sendAnimationStepMessage()
{
    // send animation step to gui
    if (coVRMSController::instance()->isMaster())
    {
        coGRSetTimestepMsg timestepMsg(getAnimationFrame(), getNumTimesteps());
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(timestepMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

coMenuItem *coVRAnimationManager::getMenuButton(const std::string &function)
{
    if (function == "ToggleAnimation")
        return animToggleItem;

    return NULL;
}

void coVRAnimationManager::setTimestepUnit(const char *unit)
{
    timestepUnit = unit;
    animFrameItem->setLabel(unit);
}

void coVRAnimationManager::setTimestepBase(double base)
{
    timestepBase = base;
    bool integer = (timestepBase == static_cast<int>(timestepBase))
                   && (timestepScale == static_cast<int>(timestepScale));
    animFrameItem->setInteger(integer);
    animFrameItem->setMin(timestepBase);
    animFrameItem->setMax(timestepBase + (getNumTimesteps() - 1) * timestepScale);
}

void coVRAnimationManager::setTimestepScale(double scale)
{
    timestepScale = scale;
    bool integer = (timestepBase == static_cast<int>(timestepBase))
                   && (timestepScale == static_cast<int>(timestepScale));
    animFrameItem->setInteger(integer);
    animFrameItem->setMax(timestepBase + (getNumTimesteps() - 1) * timestepScale);
}

std::string coVRAnimationManager::getTimestepUnit() const
{
    return timestepUnit;
}

double coVRAnimationManager::getTimestepBase() const
{
    return timestepBase;
}

double coVRAnimationManager::getTimestepScale() const
{
    return timestepScale;
}
