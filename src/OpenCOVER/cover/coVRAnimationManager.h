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

namespace vrui
{
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coSliderMenuItem;
class coRowMenu;
class coPotiMenuItem;
class coTrackerButtonInteraction;
}

namespace osg
{
class Sequence;
};

#include <util/coExport.h>
#include <OpenVRUI/coMenu.h>
#include <map>
#include <vector>

namespace opencover
{
class buttonSpecCell;
class COVEREXPORT coVRAnimationManager : public vrui::coMenuListener
{
    friend class coVRPluginList;

    static coVRAnimationManager *s_instance;
    coVRAnimationManager();
public:
    ~coVRAnimationManager();
    static coVRAnimationManager *instance();

    void setNumTimesteps(int);
    void showAnimMenu(bool visible);

    void addSequence(osg::Sequence *seq);
    void removeSequence(osg::Sequence *seq);

    const std::vector<osg::Sequence *> &getSequences() const;

    int getAnimationFrame()
    {
        return currentAnimationFrame;
    };
    void requestAnimationFrame(int currentFrame);
    void requestAnimationTime(double t);
    float getAnimationSpeed();
    void setAnimationSpeed(float speed);
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

    // remove source of timesteps
    void removeTimestepProvider(const void *who);

    void setTimestepUnit(const char *unit);
    std::string getTimestepUnit() const;
    void setTimestepBase(double base);
    double getTimestepBase() const;
    void setTimestepScale(double scale);
    double getTimestepScale() const;

    // process key events
    bool keyEvent(int type, int keySym, int mod);

    void update();

    vrui::coMenuItem *getMenuButton(const std::string &functionName);

private:
    void setAnimationFrame(int currentFrame);

    std::vector<osg::Sequence *> listOfSeq;
    float AnimSliderMin, AnimSliderMax;
    float timeState;
    void remove_controls();

    int aniDirection; // added for ping pong mode

    int numFrames;
    int startFrame, stopFrame;
    float frameAngle;
    int oldFrame;

    vrui::coTrackerButtonInteraction *animWheelInteraction;
    ///< interaction for advancing timesteps
    //with mouse wheel

    void initAnimMenu();

    // Animation menu:
    vrui::coSubMenuItem *animButton;
    vrui::coCheckboxMenuItem *animToggleItem;
    vrui::coPotiMenuItem *animSpeedItem;
    vrui::coButtonMenuItem *animForwardItem;
    vrui::coButtonMenuItem *animBackItem;
    vrui::coSliderMenuItem *animFrameItem;
    vrui::coCheckboxMenuItem *rotateObjectsToggleItem;
    vrui::coCheckboxMenuItem *animPingPongItem;
    vrui::coCheckboxMenuItem *animSyncItem;
    vrui::coRowMenu *animRowMenu;

    bool animRunning;
    double lastAnimationUpdate;
    int currentAnimationFrame, requestedAnimationFrame;
    void updateAnimationFrame();

    void menuEvent(vrui::coMenuItem *);

    void add_set_controls();

    static void forwardCallback(void *sceneGraph, buttonSpecCell *spec);
    static void backwardCallback(void *sceneGraph, buttonSpecCell *spec);
    static void animSpeedCallback(void *sceneGraph, buttonSpecCell *spec);

    typedef std::map<const void *, int> TimestepMap;
    TimestepMap timestepMap;

    void sendAnimationStateMessage();
    void sendAnimationSpeedMessage();
    void sendAnimationStepMessage();

    double timestepScale, timestepBase;
    std::string timestepUnit;
};
}
#endif
