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

#include <vrb/client/VRBMessage.h>
//#include <net/message.h>
//#include <net/message_types.h>
using namespace opencover;
using namespace grmsg;
using namespace covise;

coVRAnimationManager *coVRAnimationManager::s_instance;

coVRAnimationManager::coVRAnimationManager()
    : ui::Owner("AnimationManager", cover->ui)
    , m_animSliderMin(-25.)
    , m_animSliderMax(25.)
    , m_timeState(m_animSliderMax) // slider value of animation slider
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
    float animSpeedStartValue = m_animSliderMax * 0.96f; // 24 FPS for 25.0

    // Achtung nach diesem Fehler kann man lange suchen (dr):
    // wenn im config.xml dieser Abschnitt fehlt, dann werden nicht die oben definierten default werte genommen sondern die vom coVRTUI
    float min = coCoviseConfig::getFloat("min", "COVER.AnimationSpeed", m_animSliderMin);
    float max = coCoviseConfig::getFloat("max", "COVER.AnimationSpeed", m_animSliderMax);
    animSpeedStartValue = coCoviseConfig::getFloat("default", "COVER.AnimationSpeed", animSpeedStartValue);

    if (min > max)
        std::swap(min, max);
    m_animSliderMin = min;
    m_animSliderMax = max;
    m_configFile = cover->configFile("animationmanager"); 
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
        setAnimationSpeed(animSpeedItem->getValue());
    });

    animSkipItem = std::make_unique<ui::SliderConfigValue>(animRowMenu, "Step", 1, *m_configFile, "", config::Flag::PerModel);
    animSkipItem->ui()->setPresentation(ui::Slider::AsSlider);
    animSkipItem->ui()->setIntegral(true);
    animSkipItem->ui()->setBounds(1, std::max(10.0, animSkipItem->getValue()));
    animSkipItem->setUpdater([this](){
        setAnimationSkip(animSkipItem->getValue(), true);
    });

    animLimitGroup = new ui::Group(animRowMenu, "LimitGroup");
    animLimitGroup->setText("");

    animStartItem = std::make_unique<ui::SliderConfigValue>(animLimitGroup, "StartTimestep", m_timestepBase, *m_configFile, "", config::Flag::PerModel);
    animStartItem->ui()->setText("Start timestep");
    animStartItem->ui()->setIntegral(true);
    animStartItem->ui()->setBounds(m_timestepBase, m_timestepBase);
    animStartItem->setUpdater([this](){
        if (!animStartItem->ui()->isMoving())
            setStartFrame(animStartItem->getValue());
    });
    animStartItem->ui()->setPriority(ui::Element::Low);

    animStopItem = std::make_unique<ui::SliderConfigValue>(animLimitGroup, "StopTimestep", m_timestepBase, *m_configFile, "", config::Flag::PerModel);
    animStopItem->ui()->setText("Stop timestep");
    animStopItem->ui()->setIntegral(true);
    animStopItem->ui()->setBounds(m_timestepBase, m_timestepBase);
    animStopItem->setUpdater([this](){
        if (!animStopItem->ui()->isMoving())
            setStartFrame(animStopItem->getValue());
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

void coVRAnimationManager::setOscillate(bool state)
{
    animPingPongItem->setValue(state);
}

bool coVRAnimationManager::isOscillating() const
{
    return animPingPongItem->getValue();
}

void
coVRAnimationManager::requestAnimationTime(double t)
{
    int step = static_cast<int>((t - m_timestepBase) / m_timestepScale + 0.5);
    requestAnimationFrame(step);
}

void
coVRAnimationManager::setRemoteAnimationFrame(int currentFrame)
{
    if (currentFrame >= 0 || currentFrame < m_numFrames)
    {
        m_currentAnimationFrame = currentFrame;
        for (unsigned int i = 0; i < m_listOfSeq.size(); i++)
        {
            updateSequence(m_listOfSeq[i], currentFrame);
        }
        coVRPluginList::instance()->setTimestep(currentFrame);
        if (animFrameItem && m_numFrames != 0)
            animFrameItem->setValue(currentFrame);
    }
}

void coVRAnimationManager::setRemoteAnimate(bool state)
{
    m_animRunning = state;
    animToggleItem->setValue(state);
}
void coVRAnimationManager::setRemoteSynchronize(bool state)
{
    animSyncItem->setValue(state);
}

bool
coVRAnimationManager::requestAnimationFrame(int currentFrame)
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
        coVRPluginList::instance()->requestTimestep(currentFrame);
    }

    if ((currentFrame != m_oldFrame) && animSyncItem->getValue())
    {
        m_oldFrame = currentFrame;
        change = true;
    }

    return change;
}

void
coVRAnimationManager::updateSequence(Sequence &seq, int currentFrame)
{
    int numChildren = seq.seq->getNumChildren();
    if (currentFrame < numChildren)
    {
        seq.seq->setValue(currentFrame);
        return;
    }

    switch(seq.fill)
    {
        case Nothing:
            seq.seq->setValue(-1);
            break;
        case Last:
            seq.seq->setValue(numChildren-1);
            break;
        case Cycle:
            if (numChildren>0)
                seq.seq->setValue(currentFrame % numChildren);
            else
                seq.seq->setValue(-1);
            break;
    }
}

void
coVRAnimationManager::setAnimationFrame(int currentFrame)
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
        coVRPluginList::instance()->setTimestep(currentFrame);
        if (animFrameItem && m_numFrames != 0)
            animFrameItem->setValue(m_timestepBase + m_timestepScale * currentFrame);
        m_lastAnimationUpdate = cover->frameTime();
    }
}

int coVRAnimationManager::getNextFrame(int current)
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
coVRAnimationManager::updateAnimationFrame()
{
    if (m_animRunning && !m_animationPaused && (!animSyncItem->getValue() || coVRCollaboration::instance()->isMaster()))
    {
        auto speed = animSpeedItem->getValue();
        if ((cover->frameTime() - m_lastAnimationUpdate >= 1.0 / std::abs(speed))
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
coVRAnimationManager::getAnimationSpeed()
{
    return animSpeedItem->getValue();
}

size_t coVRAnimationManager::getAnimationSkip()
{
    return (size_t)animSkipItem->getValue();

}

float
coVRAnimationManager::getCurrentSpeed() const {

    return animSpeedItem->getValue() * m_aniDirection;
}

void
coVRAnimationManager::setAnimationSpeed(float speed)
{
    if (speed < animSpeedItem->ui()->min())
        speed = animSpeedItem->ui()->min();
    if (speed > animSpeedItem->ui()->max())
        speed = animSpeedItem->ui()->max();

    animSpeedItem->setValue(speed);
}

void opencover::coVRAnimationManager::setAnimationSpeedMax(float maxSpeed)
{
    animSpeedItem->ui()->setBounds(animSpeedItem->ui()->min(), maxSpeed);
}

void coVRAnimationManager::setAnimationSkip(int frames, bool ignoreMax)
{
    if (frames < animSkipItem->ui()->min())
        frames = animSkipItem->ui()->min();
    if (frames > animSkipItem->ui()->max())
    if(ignoreMax)
        setAnimationSkipMax(frames);
    else
        frames = animSkipItem->ui()->max();

    animSkipItem->setValue(frames);
    m_aniSkip = frames;
}

void opencover::coVRAnimationManager::setAnimationSkipMax(int maxFrames)
{
    animSkipItem->ui()->setBounds(1, maxFrames);
}

bool
coVRAnimationManager::animationRunning()
{
    return m_animRunning;
}

void
coVRAnimationManager::enableAnimation(bool state)
{
    m_animRunning = state;
    animToggleItem->setValue(m_animRunning);
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

void coVRAnimationManager::showAnimMenu(bool visible)
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
coVRAnimationManager::removeSequence(osg::Sequence *seq)
{
    for (unsigned int i = 0; i < m_listOfSeq.size(); i++)
    {
        if (m_listOfSeq[i].seq == seq)
        {
            removeTimestepProvider(seq);
            for (unsigned int n = i; n < m_listOfSeq.size() - 1; n++)
                m_listOfSeq[n] = m_listOfSeq[n + 1];
            m_listOfSeq.pop_back();
            break;
        }
    }
}

const std::vector<coVRAnimationManager::Sequence>&
coVRAnimationManager::getSequences() const
{
    return m_listOfSeq;
}

void
coVRAnimationManager::addSequence(osg::Sequence *seq, coVRAnimationManager::FillMode mode)
{
    setNumTimesteps(seq->getNumChildren(), seq);
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

void coVRAnimationManager::setStartFrame(int frame)
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

int coVRAnimationManager::getStartFrame() const
{
    return m_startFrame;
}

void coVRAnimationManager::setStopFrame(int frame)
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

int coVRAnimationManager::getStopFrame() const
{
    return m_stopFrame;
}

// get number of timesteps
int coVRAnimationManager::getNumTimesteps()
{
    return m_numFrames;
}

// set number of timesteps
void coVRAnimationManager::setNumTimesteps(int t, const void *who)
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
void coVRAnimationManager::setMaxFrameRate(int t)
{
    if (t > animSpeedItem->ui()->max())
    {
        animSpeedItem->ui()->setBounds(animSpeedItem->ui()->min(),animSpeedItem->ui()->max());
    }
}

void coVRAnimationManager::removeTimestepProvider(const void *who)
{
    setNumTimesteps(0, who);
    m_timestepMap.erase(who);
}

void coVRAnimationManager::setTimestepUnit(const char *unit)
{
    m_timestepUnit = unit;
    animFrameItem->setText(unit);
}

void coVRAnimationManager::setTimestepBase(double base)
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

void coVRAnimationManager::setTimestepScale(double scale)
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

std::string coVRAnimationManager::getTimestepUnit() const
{
    return m_timestepUnit;
}

double coVRAnimationManager::getTimestepBase() const
{
    return m_timestepBase;
}

double coVRAnimationManager::getTimestepScale() const
{
    return m_timestepScale;
}
