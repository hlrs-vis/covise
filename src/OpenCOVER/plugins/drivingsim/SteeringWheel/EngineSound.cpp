/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "EngineSound.h"

using namespace vrml;

SoundStep::SoundStep(vrml::Player *p, float umin)
{
    player = p;
    speed = umin;
    if (p != NULL)
    {
        char name[500];
        sprintf(name, "%d.wav", (int)speed);
        Audio *engineAudio = new Audio(name);
        source = player->newSource(engineAudio);
        if (source)
        {
            source->setLoop(true);
            source->play();
            source->setIntensity(0.0);
        }
    }
    playing = false;
}

SoundStep::~SoundStep()
{
    delete source;
}

void SoundStep::stop()
{
    if (playing)
    {
        //source->stop();
        if (source)
            source->setIntensity(0);
        playing = false;
    }
}
void SoundStep::start()
{
    if (!playing)
    {
        //source->play();
        if (source)
            source->setIntensity(1);
        playing = true;
    }
}

EngineSound::EngineSound(vrml::Player *p)
{
    player = p;
    for (int i = 0; i < NUM_SPEEDS; i++)
    {
        sounds[i] = new SoundStep(player, 1000 + i * SPEED_STEP);
    }
    pitchValues[0] = 2600; //1000
    pitchValues[1] = 2600; //1200
    pitchValues[2] = 2500; //1400
    pitchValues[3] = 2000; //1600
    pitchValues[4] = 1800; //1800
    pitchValues[5] = 1400; //2000
    pitchValues[6] = 2000; //2200
    pitchValues[7] = 1900; //2400
    pitchValues[8] = 1600; //2600
    pitchValues[9] = 1500; //2800
    pitchValues[10] = 1700; //3000
    pitchValues[11] = 1800; //3200
    pitchValues[12] = 1300; //3400
    pitchValues[13] = 1900; //3600
    pitchValues[14] = 1100; //3800
    pitchValues[15] = 1000; //4000
    pitchValues[16] = 2000; //4200
    pitchValues[17] = 1000; //4400
    pitchValues[18] = 600; //4600
    pitchValues[19] = 700; //4800
    pitchValues[20] = 700; //5000
    pitchValues[21] = 1100; //5200
    pitchValues[22] = 700; //5400
    pitchValues[23] = 1000; //5600
    pitchValues[24] = 800; //5800
    pitchValues[25] = 1200; //6000
    pitchValues[26] = 300; //6200
    pitchValues[27] = 700; //6400
    pitchValues[28] = 900; //6600
    pitchValues[29] = 300; //6800
    volumeValues[0] = 0.1f; //1000
    volumeValues[1] = 0.1f; //1200
    volumeValues[2] = 0.1f; //1400
    volumeValues[3] = 0.1f; //1600
    volumeValues[4] = 0.1f; //1800
    volumeValues[5] = 0.1f; //2000
    volumeValues[6] = 0.1f; //2200
    volumeValues[7] = 0.1f; //2400
    volumeValues[8] = 0.1f; //2600
    volumeValues[9] = 0.1f; //2800
    volumeValues[10] = 0.1f; //3000
    volumeValues[11] = 0.1f; //3200
    volumeValues[12] = 0.1f; //3400
    volumeValues[13] = 0.1f; //3600
    volumeValues[14] = 0.1f; //3800
    volumeValues[15] = 0.1f; //4000
    volumeValues[16] = 0.1f; //4200
    volumeValues[17] = 0.1f; //4400
    volumeValues[18] = 0.1f; //4600
    volumeValues[19] = 0.1f; //4800
    volumeValues[20] = 0.1f; //5000
    volumeValues[21] = 0.1f; //5200
    volumeValues[22] = 0.1f; //5400
    volumeValues[23] = 0.1f; //5600
    volumeValues[24] = 0.1f; //5800
    volumeValues[25] = 0.1f; //6000
    volumeValues[26] = 0.1f; //6200
    volumeValues[27] = 0.1f; //6400
    volumeValues[28] = 0.1f; //6600
    volumeValues[29] = 0.1f; //6800
    revs = new coTUIEditFloatField("800", coVRTui::instance()->getCOVERTab()->getID());
    revs->setEventListener(this);
    revs->setPos(20, 1);
    revs->setValue(800);
    up = new coTUIButton("up", coVRTui::instance()->getCOVERTab()->getID());
    up->setEventListener(this);
    up->setPos(20, 2);
    down = new coTUIButton("down", coVRTui::instance()->getCOVERTab()->getID());
    down->setEventListener(this);
    down->setPos(20, 3);
    pitch = new coTUIEditFloatField("2000", coVRTui::instance()->getCOVERTab()->getID());
    pitch->setEventListener(this);
    pitch->setPos(20, 4);
    pitch->setValue(2000);
}
void EngineSound::tabletEvent(coTUIElement *TuiItem)
{
    if (TuiItem == up)
    {
        revs->setValue(revs->getValue() + 1);
    }
    if (TuiItem == down)
    {
        revs->setValue(revs->getValue() - 1);
    }
    if (TuiItem == pitch)
    {
        volumeValues[currentSample] = (pitch->getValue());
    }
}

EngineSound::~EngineSound()
{
    for (int i = 0; i < NUM_SPEEDS; i++)
    {
        delete sounds[i];
    }
}

void EngineSound::setSpeed(float s)
{
    if (s > 0 && s < 600)
        s = 600;
    if (s > 0 && s < 900)
        s = revs->getValue();
    if (s > 6800)
        s = 6800;
    //std::cerr << "speed " << s << std::endl;
    for (int i = 0; i < NUM_SPEEDS; i++)
    {
        if (s > (sounds[i]->speed) || sounds[i]->speed - (SPEED_STEP + OVERLAP) >= s)
        {
            sounds[i]->stop();
        }
        else
        {
            sounds[i]->start();
            if (s < sounds[i]->speed - (SPEED_STEP)) // einblenden
            {
                if (sounds[i]->source)
                    sounds[i]->source->setIntensity(1.0 - (((sounds[i]->speed - s - OVERLAP) / SPEED_STEP) * volumeValues[i]) - (((sounds[i]->speed - SPEED_STEP) - s) / (float)OVERLAP));
                //std::cerr << "intensity " << (1.0 - (((sounds[i]->speed-s-OVERLAP)/SPEED_STEP)*volumeValues[i]) - (((sounds[i]->speed - SPEED_STEP)-s)/(float)OVERLAP))<< std::endl;
            }
            else if (s > sounds[i]->speed - (OVERLAP)) // ausblenden
            {
                if (sounds[i]->source)
                    sounds[i]->source->setIntensity(((sounds[i]->speed - s) / (float)OVERLAP) * (1.0 - volumeValues[i]));
                //std::cerr << "intensity " <<((sounds[i]->speed-s)/(float)OVERLAP)*(1.0 - volumeValues[i]) << std::endl;
            }
            else
            {
                if (sounds[i]->source)
                    sounds[i]->source->setIntensity(1.0 - (((sounds[i]->speed - s - OVERLAP) / SPEED_STEP) * volumeValues[i]));
                //std::cerr << "intensity " <<1.0 - (((sounds[i]->speed-s-OVERLAP)/SPEED_STEP)*volumeValues[i]) << std::endl;
            }
            if (sounds[i]->source)
                sounds[i]->source->setPitch(22000.0 + ((s - sounds[i]->speed) / (float)SPEED_STEP) * pitchValues[i]);
            static int oldSample = -1;
            currentSample = i;
            if (currentSample != oldSample)
            {
                pitch->setValue(volumeValues[i]);
            }
            oldSample = currentSample;
        }
    }
}
