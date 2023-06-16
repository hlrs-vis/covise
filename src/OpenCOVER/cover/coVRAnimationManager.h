/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_ANIMATION_MANAGER_H
#define COVR_ANIMATION_MANAGER_H

/*! \file
 \brief  manage timestep data

 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

namespace opencover
{
namespace ui
{
class Group;
class Menu;
class Action;
class Button;
class Slider;
}
}
#include "ui/Owner.h"
#include "ui/CovconfigLink.h"

#include <util/coExport.h>
#include <map>
#include <vector>
#include <osg/Sequence>

#include <OpenConfig/file.h>
namespace opencover
{
class COVEREXPORT coVRAnimationManager
: public ui::Owner
{
    friend class coVRPluginList;

    static coVRAnimationManager *s_instance;
    coVRAnimationManager();
public:
    ~coVRAnimationManager();
    static coVRAnimationManager *instance();

    //! how to handle missing elements at end of animition sequence
    enum FillMode
    {
        Nothing, //< nothing is shown
        Last, //< last element is shown
        Cycle, //< previous elements are repeated periodically
    };

    struct Sequence
    {
        Sequence(osg::Sequence *seq, FillMode mode=Nothing): seq(seq), fill(mode) {}

        osg::ref_ptr<osg::Sequence> seq;
        FillMode fill = Nothing;
    };

    void setNumTimesteps(int);
    void showAnimMenu(bool visible);

    void addSequence(osg::Sequence *seq, FillMode mode=Nothing);
    void removeSequence(osg::Sequence *seq);

    const std::vector<Sequence> &getSequences() const;

    int getAnimationFrame() const
    {
        return m_currentAnimationFrame;
    }

    int getNextFrame(int current = -1);

    bool requestAnimationFrame(int currentFrame);
    void requestAnimationTime(double t);
    float getAnimationSpeed();
    size_t getAnimationSkip();
    void setAnimationSpeed(float speed);
    void setAnimationSpeedMax(float maxSpeed);
    void setAnimationSkip(int frames, bool ignoreMax = false);
    void setAnimationSkipMax(int maxFrames);
    bool animationRunning();
    void enableAnimation(bool state);
    void setRemoteAnimationFrame(int currentFrame);
    void setRemoteAnimate(bool state);
    void setRemoteSynchronize(bool state);
    void setOscillate(bool state);
    bool isOscillating() const;
    int getStartFrame() const;
    void setStartFrame(int frame);
    int getStopFrame() const;
    void setStopFrame(int frame);
    float getCurrentSpeed() const;

    // get number of timesteps
    int getNumTimesteps();

    // set number of timesteps
    void setNumTimesteps(int, const void *who);
    void setMaxFrameRate(int);

    // remove source of timesteps
    void removeTimestepProvider(const void *who);

    void setTimestepUnit(const char *unit);
    std::string getTimestepUnit() const;
    void setTimestepBase(double base);
    double getTimestepBase() const;
    void setTimestepScale(double scale);
    double getTimestepScale() const;

    bool update();

private:
    void setAnimationFrame(int currentFrame);

    void updateSequence(Sequence &seq, int currentFrame);
    std::vector<Sequence> m_listOfSeq;
    float m_animSliderMin, m_animSliderMax;
    float m_timeState;

    int m_aniDirection = 1; // added for ping pong mode
    int m_aniSkip = 1; // step width for automatic animation

    int m_numFrames;
    int m_startFrame, m_stopFrame;
    float m_frameAngle;
    int m_oldFrame;

    void initAnimMenu();
    std::unique_ptr<opencover::config::File> m_configFile;
    // Animation menu:
    ui::Action *animForwardItem;
    ui::Action *animBackItem;
    ui::Group *animStepGroup;
    ui::Group *animLimitGroup;
    ui::Menu *animRowMenu;

    std::unique_ptr<ui::ButtonConfigValue> animToggleItem;
    ui::Slider *animFrameItem;
    std::unique_ptr<ui::SliderConfigValue> animSpeedItem;
    std::unique_ptr<ui::ButtonConfigValue> rotateObjectsToggleItem;
    std::unique_ptr<ui::ButtonConfigValue> animPingPongItem;
    std::unique_ptr<ui::ButtonConfigValue> animSyncItem;
    std::unique_ptr<ui::SliderConfigValue> animStartItem, animStopItem;
    std::unique_ptr<ui::SliderConfigValue> presentationStep;
    std::unique_ptr<ui::SliderConfigValue> animSkipItem;

    
    bool m_animRunning;
    double m_lastAnimationUpdate;
    int m_currentAnimationFrame, m_requestedAnimationFrame;
    bool updateAnimationFrame();

    typedef std::map<const void *, int> TimestepMap;
    TimestepMap m_timestepMap;

    double m_timestepScale, m_timestepBase;
    std::string m_timestepUnit;
    bool m_animationPaused = false;

};
}
#endif
