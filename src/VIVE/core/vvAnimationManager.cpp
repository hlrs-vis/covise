/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vsg/nodes/Switch.h>

#include <config/CoviseConfig.h>
#include "vvAnimationManager.h"
#include "ui/Menu.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Slider.h"
#include <net/message.h>
#include <net/message_types.h>
#include "vvPluginSupport.h"
#include "vvPluginList.h"
#include "vvCollaboration.h"
#include "vvMSController.h"
#include "vvVIVE.h"

#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>

#include <vrb/client/VRBMessage.h>
//#include <net/message.h>
//#include <net/message_types.h>
using namespace vive;
using namespace grmsg;
using namespace covise;

vvAnimationManager *vvAnimationManager::s_instance;

vvAnimationManager::vvAnimationManager()
: ui::Owner("AnimationManager", vv->ui)
, m_animSliderMin(-25.)
, m_animSliderMax(25.)
, m_numFrames(0)
, m_startFrame(0)
, m_stopFrame(0)
, m_oldFrame(-1)
, m_animRunning(true)
, m_lastAnimationUpdate(0.0)
, m_currentAnimationFrame(-1)
, m_requestedAnimationFrame(-1)
, m_timestepScale(1.0)
, m_timestepBase(0.0)
, m_timestepUnit("Time step")
{
    assert(!s_instance);

    initAnimMenu();
    showAnimMenu(false);
    setAnimationFrame(0);
}

vvAnimationManager::~vvAnimationManager()
{
    s_instance = NULL;
}

vvAnimationManager * vvAnimationManager::instance()
{
    if (!s_instance)
        s_instance = new vvAnimationManager;
    return s_instance;
}

void vvAnimationManager::initAnimMenu()
{
    // Create animation menu:

    // anim speed was configurable in 5.2 where animSpeed was a slider
    // for the dial where min=-max we use the greater value as AnimSliderMax
    float animSpeedStartValue = m_animSliderMax * 0.96f; // 24 FPS for 25.0

    // Achtung nach diesem Fehler kann man lange suchen (dr):
    // wenn im config.xml dieser Abschnitt fehlt, dann werden nicht die oben definierten default werte genommen sondern die vom vvTui
    float min = coCoviseConfig::getFloat("min", "COVER.AnimationSpeed", m_animSliderMin);
    float max = coCoviseConfig::getFloat("max", "COVER.AnimationSpeed", m_animSliderMax);
    animSpeedStartValue = coCoviseConfig::getFloat("default", "COVER.AnimationSpeed", animSpeedStartValue);

    if (min > max)
        std::swap(min, max);
    m_animSliderMin = min;
    m_animSliderMax = max;
    m_configFile = vv->configFile("animationmanager"); 
#if 0
   animRowMenu->add(rotateObjectsToggleItem);
#endif
    animRowMenu = new ui::Menu("Animation", this);

    animToggleItem = std::make_unique<ui::ButtonConfigValue>(animRowMenu, "Animate", m_animRunning, *m_configFile, "", config::Flag::PerModel);
    animToggleItem->ui()->setShortcut("a");
    animToggleItem->ui()->addShortcut(" ");
    animToggleItem->setUpdater([this](){
        if (m_animRunning != animToggleItem->getValue())
            enableAnimation(animToggleItem->getValue());
    });
    animToggleItem->ui()->setPriority(ui::Element::Toolbar);
    animToggleItem->ui()->setIcon("media-playback-start");

    animStepGroup = new ui::Group(animRowMenu, "TimestepGroup");
    animStepGroup->setText("");

    animFrameItem = new ui::Slider(animStepGroup, "Timestep");
    animFrameItem->setValue(m_timestepBase);
    animFrameItem->setText(m_timestepUnit);
    animFrameItem->setIntegral(true);
    animFrameItem->setBounds(m_timestepBase, std::max(m_timestepBase, animFrameItem->value()));
    animFrameItem->setCallback([this](ui::Slider::ValueType val, bool released){
        requestAnimationTime(val);
        m_animationPaused = !released;
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
        requestAnimationFrame(getAnimationFrame() + 1 * m_aniSkip);
    });
    animForwardItem->setPriority(ui::Element::Toolbar);
    animForwardItem->setIcon("media-seek-forward");

    animPingPongItem = std::make_unique<ui::ButtonConfigValue>(animRowMenu, "Oscillate", false, *m_configFile, "", config::Flag::PerModel);
    
    animSpeedItem = std::make_unique<ui::SliderConfigValue>(animRowMenu, "Speed", animSpeedStartValue, *m_configFile, "", config::Flag::PerModel);
    animSpeedItem->ui()->setPresentation(ui::Slider::AsDial);
    animSpeedItem->ui()->setBounds(m_animSliderMin, m_animSliderMax);
    animSpeedItem->setUpdater([this](){
        setAnimationSpeed((float)animSpeedItem->getValue());
    });

    animSkipItem = std::make_unique<ui::SliderConfigValue>(animRowMenu, "Step", 1, *m_configFile, "", config::Flag::PerModel);
    animSkipItem->ui()->setPresentation(ui::Slider::AsSlider);
    animSkipItem->ui()->setIntegral(true);
    animSkipItem->ui()->setBounds(1, std::max(10.0, animSkipItem->getValue()));
    animSkipItem->setUpdater([this](){
        setAnimationSkip((int)animSkipItem->getValue(), true);
    });

    animLimitGroup = new ui::Group(animRowMenu, "LimitGroup");
    animLimitGroup->setText("");

    animStartItem = std::make_unique<ui::SliderConfigValue>(animLimitGroup, "StartTimestep", m_timestepBase, *m_configFile, "", config::Flag::PerModel);
    animStartItem->ui()->setText("Start timestep");
    animStartItem->ui()->setIntegral(true);
    animStartItem->ui()->setBounds(m_timestepBase, m_timestepBase);
    animStartItem->setUpdater([this](){
        if (!animStartItem->ui()->isMoving())
            setStartFrame((int)animStartItem->getValue());
    });
    animStartItem->ui()->setPriority(ui::Element::Low);

    animStopItem = std::make_unique<ui::SliderConfigValue>(animLimitGroup, "StopTimestep", m_timestepBase, *m_configFile, "", config::Flag::PerModel);
    animStopItem->ui()->setText("Stop timestep");
    animStopItem->ui()->setIntegral(true);
    animStopItem->ui()->setBounds(m_timestepBase, m_timestepBase);
    animStopItem->setUpdater([this](){
        if (!animStopItem->ui()->isMoving())
            setStopFrame((int)animStopItem->getValue());
    });
    animStopItem->ui()->setPriority(ui::Element::Low);

    animSyncItem = std::make_unique<ui::ButtonConfigValue>(animRowMenu, "Synchronize", true, *m_configFile, "", config::Flag::PerModel);
    animSyncItem->ui()->setShared(true);
	animSyncItem->setUpdater([this]()
		{
			auto state = animSyncItem->getValue();
            animFrameItem->setShared(state);
			animSpeedItem->ui()->setShared(state);
            animSkipItem->ui()->setShared(state);
			animToggleItem->ui()->setShared(state);
			
		});
    animSyncItem->ui()->callback()(true);
}

void vvAnimationManager::setOscillate(bool state)
{
    animPingPongItem->setValue(state);
}

bool vvAnimationManager::isOscillating() const
{
    return animPingPongItem->getValue();
}

void
vvAnimationManager::requestAnimationTime(double t)
{
    int step = static_cast<int>((t - m_timestepBase) / m_timestepScale + 0.5);
    requestAnimationFrame(step);
}

void
vvAnimationManager::setRemoteAnimationFrame(int currentFrame)
{
    if (currentFrame >= 0 || currentFrame < m_numFrames)
    {
        m_currentAnimationFrame = currentFrame;
        for (unsigned int i = 0; i < m_listOfSeq.size(); i++)
        {
            updateSequence(m_listOfSeq[i], currentFrame);
        }
        vvPluginList::instance()->setTimestep(currentFrame);
        if (animFrameItem && m_numFrames != 0)
            animFrameItem->setValue(currentFrame);
    }
}

void vvAnimationManager::setRemoteAnimate(bool state)
{
    m_animRunning = state;
    animToggleItem->setValue(state);
}
void vvAnimationManager::setRemoteSynchronize(bool state)
{
    animSyncItem->setValue(state);
}

bool
vvAnimationManager::requestAnimationFrame(int currentFrame)
{
    if (m_numFrames == 0)
    {
        currentFrame = 0;
    }
    else
    {
        while (currentFrame < 0)
            currentFrame += m_numFrames;
        currentFrame %= m_numFrames;
    }

    if (m_stopFrame >= m_startFrame)
        currentFrame = (currentFrame - m_startFrame + m_stopFrame - m_startFrame + 1) % (m_stopFrame - m_startFrame + 1) + m_startFrame;
    else
        currentFrame = m_startFrame;

    currentFrame = (currentFrame/m_aniSkip)*m_aniSkip;

    bool change = false;
    if (m_currentAnimationFrame != currentFrame && m_requestedAnimationFrame == -1)
    {
        change = true;
        m_requestedAnimationFrame = currentFrame;
        vvPluginList::instance()->requestTimestep(currentFrame);
    }

    if ((currentFrame != m_oldFrame) && animSyncItem->getValue())
    {
        m_oldFrame = currentFrame;
        change = true;
    }

    if (change)
        m_lastAnimationUpdate = vv->frameTime();

    return change;
}

void
vvAnimationManager::updateSequence(Sequence& seq, int currentFrame)
{
    size_t numChildren = seq.seq->children.size();
    if (currentFrame < numChildren)
    {
        seq.seq->setSingleChildOn(currentFrame);
        return;
    }

    switch(seq.fill)
    {
        case Nothing:
            seq.seq->setAllChildren(false);
            break;
        case Last:
            seq.setValue((int)numChildren-1);
            break;
        case Cycle:
            if (numChildren>0)
                seq.setValue(currentFrame % numChildren);
            else
                seq.seq->setAllChildren(false);
            break;
    }
}

void
vvAnimationManager::setAnimationFrame(int currentFrame)
{
    if (m_requestedAnimationFrame != -1 && currentFrame != m_requestedAnimationFrame)
    {
        std::cerr << "setAnimationFrame(" << currentFrame << "), but " << m_requestedAnimationFrame << " was requested" << std::endl;
    }
    m_requestedAnimationFrame = -1;
    if (m_currentAnimationFrame != currentFrame)
    {
        m_currentAnimationFrame = currentFrame;
        for (unsigned int i = 0; i < m_listOfSeq.size(); i++)
        {
            updateSequence(m_listOfSeq[i], currentFrame);
        }
        vvPluginList::instance()->setTimestep(currentFrame);
        if (animFrameItem && m_numFrames != 0)
            animFrameItem->setValue(m_timestepBase + m_timestepScale * currentFrame);
    }
}

int vvAnimationManager::getNextFrame(int current)
{
    if (current == -1)
        current = m_currentAnimationFrame;

    if (!m_animRunning)
        return current;

    if (animPingPongItem->getValue()) // normal loop mode
    {
        if (current >= m_stopFrame) // check for end of sequence
            m_aniDirection = -1;
        if (current <= m_startFrame) // check for start of sequence
            m_aniDirection = 1;
    }
    else
        m_aniDirection = 1;

    int next = current;
    if (animSpeedItem->getValue() > 0.0)
    {
        next = current + m_aniDirection * m_aniSkip;
    }
    else if (animSpeedItem->getValue() < 0.0)
    {
        next = current - m_aniDirection * m_aniSkip;
    }

    if (m_numFrames == 0)
    {
        next = 0;
    }
    else
    {
        if (next < 0)
            next = (next % m_numFrames) + m_numFrames;
        next %= m_numFrames;
    }

    if (m_stopFrame >= m_startFrame)
        next = (next - m_startFrame + m_stopFrame - m_startFrame + 1) % (m_stopFrame - m_startFrame + 1) + m_startFrame;
    else
        next = m_startFrame;

    next = (next/m_aniSkip)*m_aniSkip;

    return next;
}

bool
vvAnimationManager::updateAnimationFrame()
{
    if (m_animRunning && !m_animationPaused && (!animSyncItem->getValue() || vvCollaboration::instance()->isMaster()))
    {
        auto speed = animSpeedItem->getValue();
        if ((vv->frameTime() - m_lastAnimationUpdate >= 1.0 / std::abs(speed))
            || (speed > 0. && speed > m_animSliderMax - 0.001)
            || (speed < 0. && speed < m_animSliderMin + 0.001)) {
            return requestAnimationFrame(getNextFrame());
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
vvAnimationManager::getAnimationSpeed()
{
    return (float)animSpeedItem->getValue();
}

size_t vvAnimationManager::getAnimationSkip()
{
    return (size_t)animSkipItem->getValue();

}

float
vvAnimationManager::getCurrentSpeed() const {

    return (float)animSpeedItem->getValue() * m_aniDirection;
}

void
vvAnimationManager::setAnimationSpeed(float speed)
{
    if (speed < (float)animSpeedItem->ui()->min())
        speed = (float)animSpeedItem->ui()->min();
    if (speed > (float)animSpeedItem->ui()->max())
        speed = (float)animSpeedItem->ui()->max();

    animSpeedItem->setValue(speed);
}

void vive::vvAnimationManager::setAnimationSpeedMax(float maxSpeed)
{
    animSpeedItem->ui()->setBounds(animSpeedItem->ui()->min(), maxSpeed);
}

void vvAnimationManager::setAnimationSkip(int frames, bool ignoreMax)
{
    if (frames < (int)animSkipItem->ui()->min())
    {
        frames = (int)animSkipItem->ui()->min();
    }
    if (frames > (int)animSkipItem->ui()->max())
    {
        if(ignoreMax)
            setAnimationSkipMax(frames);
        else
            frames = (int)animSkipItem->ui()->max();
    }

    animSkipItem->setValue(frames);
    m_aniSkip = frames;
}

void vive::vvAnimationManager::setAnimationSkipMax(int maxFrames)
{
    animSkipItem->ui()->setBounds(1, maxFrames);
}

bool
vvAnimationManager::animationRunning()
{
    return m_animRunning;
}

void
vvAnimationManager::enableAnimation(bool state)
{
    m_animRunning = state;
    animToggleItem->setValue(m_animRunning);
}

bool
vvAnimationManager::update()
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
         vsg::dmat4 mat = objectsXformDCS->matrix;
         mat.preMult(vsg::dmat4::translate(-rotationPoint[0],
                  -rotationPoint[1],
                  -rotationPoint[2]));
         mat.preMult(vsg::dmat4::rotate(diffAngle,
                  rotationAxis[0],
                  rotationAxis[1],
                  rotationAxis[2]));
         mat.preMult(vsg::dmat4::translate(rotationPoint[0],
                  rotationPoint[1],
                  rotationPoint[2]));
         objectsXformDCS->matrix = (mat);
      }
   }
#endif
}

void vvAnimationManager::setNumTimesteps(int t)
{
    m_numFrames = t;
    if (m_numFrames == 0)
        m_numFrames = 1;
    if (animFrameItem)
    {
        animFrameItem->setBounds(m_timestepBase, m_timestepBase + (m_numFrames - 1) * m_timestepScale);
        //animFrameItem->setNumTicks(numFrames - 1);
        animStartItem->ui()->setBounds(m_timestepBase, m_timestepBase + (m_numFrames - 1) * m_timestepScale);
        animStopItem->ui()->setBounds(m_timestepBase, m_timestepBase + (m_numFrames - 1) * m_timestepScale);
    }

    if (m_startFrame >= m_numFrames)
        m_startFrame = 0;
    animStartItem->setValue(m_startFrame);
    m_stopFrame = m_numFrames - 1;
    animStopItem->setValue(m_startFrame);

    if (m_currentAnimationFrame >= m_numFrames)
    {
        requestAnimationFrame(0);
    }
}

void vvAnimationManager::showAnimMenu(bool visible)
{
    animRowMenu->setVisible(visible);

    animToggleItem->ui()->setEnabled(visible);
    animForwardItem->setEnabled(visible);
    animBackItem->setEnabled(visible);
    animFrameItem->setEnabled(visible);
    animStartItem->ui()->setEnabled(visible);
    animStopItem->ui()->setEnabled(visible);
}

void
vvAnimationManager::removeSequence(vsg::Switch* seq)
{
    for (unsigned int i = 0; i < m_listOfSeq.size(); i++)
    {
        if (m_listOfSeq[i].seq.get() == seq)
        {
            removeTimestepProvider(seq);
            for (unsigned int n = i; n < m_listOfSeq.size() - 1; n++)
                m_listOfSeq[n] = m_listOfSeq[n + 1];
            m_listOfSeq.pop_back();
            break;
        }
    }
}

const std::vector<vvAnimationManager::Sequence>&
vvAnimationManager::getSequences() const
{
    return m_listOfSeq;
}

void
vvAnimationManager::addSequence(vsg::Switch* seq, vvAnimationManager::FillMode mode)
{
    setNumTimesteps((int)seq->children.size(), seq);
    bool alreadyAdded = false;
    for (unsigned int i = 0;
         i < m_listOfSeq.size();
         i++)
    {
        if (seq == m_listOfSeq[i].seq)
        {
            m_listOfSeq[i].fill = mode;
            alreadyAdded = true;
            updateSequence(m_listOfSeq[i], m_currentAnimationFrame);
            break;
        }
    }
    if (!alreadyAdded)
    {
        m_listOfSeq.emplace_back(seq, mode);
        updateSequence(m_listOfSeq.back(), m_currentAnimationFrame);
    }
}

void vvAnimationManager::setStartFrame(int frame)
{
    m_startFrame = frame;
    if (m_startFrame >= m_numFrames)
        m_startFrame = m_numFrames - 1;
    if (m_startFrame < 0)
        m_startFrame = 0;
    if (m_startFrame > m_stopFrame)
        m_stopFrame = m_startFrame;
    if (animStartItem->getValue() != m_startFrame)
        animStartItem->setValue(m_startFrame);
    if (animStopItem->getValue() != m_stopFrame)
        animStopItem->setValue(m_stopFrame);
}

int vvAnimationManager::getStartFrame() const
{
    return m_startFrame;
}

void vvAnimationManager::setStopFrame(int frame)
{
    m_stopFrame = frame;
    if (m_stopFrame >= m_numFrames)
        m_stopFrame = m_numFrames - 1;
    if (m_stopFrame < 0)
        m_stopFrame = 0;
    if (m_startFrame > m_stopFrame)
        m_startFrame = m_stopFrame;
    if (animStartItem->getValue() != m_startFrame)
        animStartItem->setValue(m_startFrame);
    if (animStopItem->getValue() != m_stopFrame)
        animStopItem->setValue(m_stopFrame);
}

int vvAnimationManager::getStopFrame() const
{
    return m_stopFrame;
}

// get number of timesteps
int vvAnimationManager::getNumTimesteps()
{
    return m_numFrames;
}

// set number of timesteps
void vvAnimationManager::setNumTimesteps(int t, const void *who)
{
    m_timestepMap[who] = t;

    int numTimesteps = 0;
    for (TimestepMap::const_iterator it = m_timestepMap.begin();
         it != m_timestepMap.end();
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

// set number of timesteps
void vvAnimationManager::setMaxFrameRate(int t)
{
    if (t > animSpeedItem->ui()->max())
    {
        animSpeedItem->ui()->setBounds(animSpeedItem->ui()->min(),animSpeedItem->ui()->max());
    }
}

void vvAnimationManager::removeTimestepProvider(const void *who)
{
    setNumTimesteps(0, who);
    m_timestepMap.erase(who);
}

void vvAnimationManager::setTimestepUnit(const char *unit)
{
    m_timestepUnit = unit;
    animFrameItem->setText(unit);
}

void vvAnimationManager::setTimestepBase(double base)
{
    m_timestepBase = base;
    bool integer = (m_timestepBase == static_cast<int>(m_timestepBase))
                   && (m_timestepScale == static_cast<int>(m_timestepScale));
    animFrameItem->setIntegral(integer);
    animStartItem->ui()->setIntegral(integer);
    animStopItem->ui()->setIntegral(integer);
    animFrameItem->setBounds(m_timestepBase, m_timestepBase + (getNumTimesteps() - 1) * m_timestepScale);
    animStartItem->ui()->setBounds(m_timestepBase, m_timestepBase + (getNumTimesteps() - 1) * m_timestepScale);
    animStopItem->ui()->setBounds(m_timestepBase, m_timestepBase + (getNumTimesteps() - 1) * m_timestepScale);
}

void vvAnimationManager::setTimestepScale(double scale)
{
    m_timestepScale = scale;
    bool integer = (m_timestepBase == static_cast<int>(m_timestepBase))
                   && (m_timestepScale == static_cast<int>(m_timestepScale));
    animFrameItem->setIntegral(integer);
    animStartItem->ui()->setIntegral(integer);
    animStopItem->ui()->setIntegral(integer);
    animFrameItem->setBounds(1, m_timestepBase + (getNumTimesteps() - 1) * m_timestepScale);
    animStartItem->ui()->setBounds(1, m_timestepBase + (getNumTimesteps() - 1) * m_timestepScale);
    animStopItem->ui()->setBounds(1, m_timestepBase + (getNumTimesteps() - 1) * m_timestepScale);
}

std::string vvAnimationManager::getTimestepUnit() const
{
    return m_timestepUnit;
}

double vvAnimationManager::getTimestepBase() const
{
    return m_timestepBase;
}

double vvAnimationManager::getTimestepScale() const
{
    return m_timestepScale;
}
