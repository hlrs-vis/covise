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

#include <util/coExport.h>
#include <map>
#include <vector>
#include <osg/Sequence>

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

    int getAnimationFrame()
    {
        return currentAnimationFrame;
    };
    bool requestAnimationFrame(int currentFrame);
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
    std::vector<Sequence> listOfSeq;
    float AnimSliderMin, AnimSliderMax;
    float timeState;

    int aniDirection; // added for ping pong mode

    int numFrames;
    int startFrame, stopFrame;
    float frameAngle;
    int oldFrame;

    void initAnimMenu();

    // Animation menu:
    ui::Button *animToggleItem;
    ui::Slider *animSpeedItem;
    ui::Action *animForwardItem;
    ui::Action *animBackItem;
    ui::Group *animStepGroup;
    ui::Slider *animFrameItem;
    ui::Button *rotateObjectsToggleItem;
    ui::Button *animPingPongItem;
    ui::Button *animSyncItem;
    ui::Group *animLimitGroup;
    ui::Slider *animStartItem, *animStopItem;
    ui::Slider *presentationStep;
    ui::Menu *animRowMenu;

    bool animRunning;
    double lastAnimationUpdate;
    int currentAnimationFrame, requestedAnimationFrame;
    bool updateAnimationFrame();

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
