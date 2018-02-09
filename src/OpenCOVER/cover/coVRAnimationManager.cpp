/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Sequence>
#include <osgGA/GUIEventAdapter>

#include <config/CoviseConfig.h>
#include "coVRAnimationManager.h"
#include "ui/Menu.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Slider.h"
#include <net/message.h>
#include <net/message_types.h>
#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRCollaboration.h"
#include "coVRMSController.h"
#include "OpenCOVER.h"

#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>

using namespace opencover;
using namespace grmsg;
using namespace covise;

coVRAnimationManager *coVRAnimationManager::s_instance;

coVRAnimationManager::coVRAnimationManager()
    : ui::Owner("AnimationManager", cover->ui)
    , AnimSliderMin(-25.)
    , AnimSliderMax(25.)
    , timeState(AnimSliderMax) // slider value of animation slider
    , aniDirection(1)
    , numFrames(0)
    , startFrame(0)
    , stopFrame(0)
    , oldFrame(-1)
    , animRunning(true)
    , lastAnimationUpdate(0.0)
    , currentAnimationFrame(0)
    , requestedAnimationFrame(-1)
    , timestepScale(1.0)
    , timestepBase(0.0)
    , timestepUnit("Time step")
{
    assert(!s_instance);

    initAnimMenu();
    showAnimMenu(false);
}

coVRAnimationManager::~coVRAnimationManager()
{
    s_instance = NULL;
}

coVRAnimationManager * coVRAnimationManager::instance()
{
    if (!s_instance)
        s_instance = new coVRAnimationManager;
    return s_instance;
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

#if 0
   animRowMenu->add(rotateObjectsToggleItem);
#endif
    animRowMenu = new ui::Menu("Animation", this);

    animToggleItem = new ui::Button(animRowMenu, "Animate");
    animToggleItem->setShortcut("a");
    animToggleItem->addShortcut(" ");
    animToggleItem->setCallback([this](bool flag){
        if (animRunning != flag)
            enableAnimation(flag);
    });
    animToggleItem->setState(animRunning);
    animToggleItem->setPriority(ui::Element::Toolbar);
    animToggleItem->setIcon("media-playback-start");

    animStepGroup = new ui::Group(animRowMenu, "TimestepGroup");
    animStepGroup->setText("");

    animFrameItem = new ui::Slider(animStepGroup, "Timestep");
    animFrameItem->setText(timestepUnit);
    animFrameItem->setIntegral(true);
    animFrameItem->setBounds(timestepBase, timestepBase);
    animFrameItem->setValue(timestepBase);
    animFrameItem->setCallback([this](ui::Slider::ValueType val, bool released){
        if (animationRunning())
            enableAnimation(false);
        requestAnimationTime(val);
    });
    animFrameItem->setPriority(ui::Element::Toolbar);

    animBackItem = new ui::Action(animStepGroup, "StepBackward");
    animBackItem->setText("Step backward");
    animBackItem->setShortcut(",");
    animBackItem->addShortcut("Shift+Button:WheelDown");
    animBackItem->addShortcut("Button:WheelLeft");
    animBackItem->setCallback([this](){
        if (animationRunning())
            enableAnimation(false);
        requestAnimationFrame(getAnimationFrame() - 1);
    });
    animBackItem->setPriority(ui::Element::Toolbar);
    animBackItem->setIcon("media-seek-backward");

    animForwardItem = new ui::Action(animStepGroup, "StepForward");
    animForwardItem->setText("Step forward");
    animForwardItem->setShortcut(".");
    animForwardItem->addShortcut("Shift+Button:WheelUp");
    animForwardItem->addShortcut("Button:WheelRight");
    animForwardItem->setCallback([this](){
        if (animationRunning())
            enableAnimation(false);
        requestAnimationFrame(getAnimationFrame() + 1);
    });
    animForwardItem->setPriority(ui::Element::Toolbar);
    animForwardItem->setIcon("media-seek-forward");

    animPingPongItem = new ui::Button(animRowMenu, "Oscillate");
    animPingPongItem->setState(false);
    animSyncItem = new ui::Button(animRowMenu, "Synchronize");
    animSyncItem->setState(false);

    animSpeedItem = new ui::Slider(animRowMenu, "Speed");
    animSpeedItem->setPresentation(ui::Slider::AsDial);
    animSpeedItem->setBounds(AnimSliderMin, AnimSliderMax);
    animSpeedItem->setValue(animSpeedStartValue);
    animSpeedItem->setCallback([this](ui::Slider::ValueType val, bool released){
        setAnimationSpeed(val);
    });

    animLimitGroup = new ui::Group(animRowMenu, "LimitGroup");
    animLimitGroup->setText("");

    animStartItem = new ui::Slider(animLimitGroup, "StartTimestep");
    animStartItem->setText("Start timestep");
    animStartItem->setIntegral(true);
    animStartItem->setBounds(timestepBase, timestepBase);
    animStartItem->setValue(timestepBase);
    animStartItem->setCallback([this](ui::Slider::ValueType val, bool released){
        if (released)
            setStartFrame(val);
    });
    animStartItem->setPriority(ui::Element::Low);

    animStopItem = new ui::Slider(animLimitGroup, "StopTimestep");
    animStopItem->setText("Stop timestep");
    animStopItem->setIntegral(true);
    animStopItem->setBounds(timestepBase, timestepBase);
    animStopItem->setValue(timestepBase);
    animStopItem->setCallback([this](ui::Slider::ValueType val, bool released){
        if (released)
            setStopFrame(val);
    });
    animStopItem->setPriority(ui::Element::Low);
}

void coVRAnimationManager::setOscillate(bool state)
{
    animPingPongItem->setState(state);
}

bool coVRAnimationManager::isOscillating() const
{
    return animPingPongItem->state();
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

bool
coVRAnimationManager::requestAnimationFrame(int currentFrame)
{
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

    bool change = currentAnimationFrame != currentFrame;

    if (requestedAnimationFrame == -1 && change)
    {
        requestedAnimationFrame = currentFrame;
        coVRPluginList::instance()->requestTimestep(currentFrame);
    }

    if ((currentFrame != oldFrame) && animSyncItem->state())
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
        change = true;
    }

    return change;
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

bool
coVRAnimationManager::updateAnimationFrame()
{
    if (animRunning && (!animSyncItem->state() || coVRCollaboration::instance()->isMaster()))
    {
        if (!animPingPongItem->state()) // normal loop mode
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

        if (animSpeedItem->value() > 0.0)
        {
            if ((cover->frameTime() - lastAnimationUpdate > 1.0 / animSpeedItem->value())
                || (animSpeedItem->value() > AnimSliderMax - 0.001))
            {
                return requestAnimationFrame(currentAnimationFrame + aniDirection);
            }
        }
        else if (animSpeedItem->value() < 0.0)
        {
            if ((cover->frameTime() - lastAnimationUpdate > -1.0 / animSpeedItem->value())
                || (animSpeedItem->value() < AnimSliderMin + 0.001))
            {
                return requestAnimationFrame(currentAnimationFrame - aniDirection);
            }
        }
    }
    else
    {
        // wait for plugins to resolve recently requested timestep,
        // which might be different from currentAnimationFrame,
        // so don't: requestAnimationFrame(currentAnimationFrame);
        return false;
    }
    return false;
}

float
coVRAnimationManager::getAnimationSpeed()
{
    return animSpeedItem->value();
}

float
coVRAnimationManager::getCurrentSpeed() const {

    return animSpeedItem->value() * aniDirection;
}

void
coVRAnimationManager::setAnimationSpeed(float speed)
{
    if (speed < animSpeedItem->min())
        speed = animSpeedItem->min();
    if (speed > animSpeedItem->max())
        speed = animSpeedItem->max();

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

bool
coVRAnimationManager::update()
{
    // Set selected animation frame:
    return updateAnimationFrame();

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

void coVRAnimationManager::setNumTimesteps(int t)
{
    numFrames = t;
    if (numFrames == 0)
        numFrames = 1;
    if (animFrameItem)
    {
        animFrameItem->setBounds(timestepBase, timestepBase + (numFrames - 1) * timestepScale);
        //animFrameItem->setNumTicks(numFrames - 1);
        animStartItem->setBounds(timestepBase, timestepBase + (numFrames - 1) * timestepScale);
        animStopItem->setBounds(timestepBase, timestepBase + (numFrames - 1) * timestepScale);
    }

    if (startFrame >= numFrames)
        startFrame = 0;
    animStartItem->setValue(startFrame);
    stopFrame = numFrames - 1;
    animStopItem->setValue(startFrame);
}

void coVRAnimationManager::showAnimMenu(bool visible)
{
    animRowMenu->setVisible(visible);

    animToggleItem->setEnabled(visible);
    animForwardItem->setEnabled(visible);
    animBackItem->setEnabled(visible);
    animFrameItem->setEnabled(visible);
    animStartItem->setEnabled(visible);
    animStopItem->setEnabled(visible);
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
    if (animStartItem->value() != startFrame)
        animStartItem->setValue(startFrame);
    if (animStopItem->value() != stopFrame)
        animStopItem->setValue(stopFrame);
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
    if (animStartItem->value() != startFrame)
        animStartItem->setValue(startFrame);
    if (animStopItem->value() != stopFrame)
        animStopItem->setValue(stopFrame);
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

        coGRSetAnimationSpeedMsg animationSpeedMsg(getAnimationSpeed(), animSpeedItem->min(), animSpeedItem->max());
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

void coVRAnimationManager::setTimestepUnit(const char *unit)
{
    timestepUnit = unit;
    animFrameItem->setText(unit);
}

void coVRAnimationManager::setTimestepBase(double base)
{
    timestepBase = base;
    bool integer = (timestepBase == static_cast<int>(timestepBase))
                   && (timestepScale == static_cast<int>(timestepScale));
    animFrameItem->setIntegral(integer);
    animStartItem->setIntegral(integer);
    animStopItem->setIntegral(integer);
    animFrameItem->setBounds(timestepBase, timestepBase + (getNumTimesteps() - 1) * timestepScale);
    animStartItem->setBounds(timestepBase, timestepBase + (getNumTimesteps() - 1) * timestepScale);
    animStopItem->setBounds(timestepBase, timestepBase + (getNumTimesteps() - 1) * timestepScale);
}

void coVRAnimationManager::setTimestepScale(double scale)
{
    timestepScale = scale;
    bool integer = (timestepBase == static_cast<int>(timestepBase))
                   && (timestepScale == static_cast<int>(timestepScale));
    animFrameItem->setIntegral(integer);
    animStartItem->setIntegral(integer);
    animStopItem->setIntegral(integer);
    animFrameItem->setBounds(1, timestepBase + (getNumTimesteps() - 1) * timestepScale);
    animStartItem->setBounds(1, timestepBase + (getNumTimesteps() - 1) * timestepScale);
    animStopItem->setBounds(1, timestepBase + (getNumTimesteps() - 1) * timestepScale);
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
