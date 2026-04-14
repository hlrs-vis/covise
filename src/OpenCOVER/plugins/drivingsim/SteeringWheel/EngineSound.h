/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EngineSound_H
#define _EngineSound_H

#include <string>
#include <audio/Player.h>
#include <cover/coVRTui.h>
using namespace opencover;
using namespace opencover::audio;
using namespace covise;

#define NUM_SPEEDS 30
#define SPEED_STEP 200
#define OVERLAP 20
class SoundStep
{
public:
    SoundStep(Player *p, float umin);
    virtual ~SoundStep();
    void stop();
    void start();
    std::shared_ptr<Source> source;
    float speed;
    bool playing;
    Player *player;
};

class EngineSound : public coTUIListener
{
public:
    Player *player;

    EngineSound(Player *p);
    virtual ~EngineSound();

    void setSpeed(float umin); // 0 == off
    virtual void tabletEvent(coTUIElement *tUItem);
    coTUIEditFloatField *revs;
    coTUIEditFloatField *pitch;
    coTUIButton *up;
    coTUIButton *down;
    int currentSample;

private:
    SoundStep *sounds[NUM_SPEEDS];
    float pitchValues[NUM_SPEEDS];
    float volumeValues[NUM_SPEEDS];
};
#endif
