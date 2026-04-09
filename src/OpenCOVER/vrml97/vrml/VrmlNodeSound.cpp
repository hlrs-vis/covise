/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSound.cpp
//    contributed by Kumaran Santhanam

#include "VrmlNodeSound.h"
#include "VrmlNodeAudioClip.h"
#include "VrmlNodeType.h"

#include "MathUtils.h"
#include "VrmlScene.h"
#include "System.h"

#include <algorithm>
#ifdef HAVE_AUDIO
#include <audio/Player.h>
#endif

using namespace vrml;
using opencover::audio::Player;

// Sound factory. Add each Sound to the scene for fast access.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSound(scene);
}

// Define the built in VrmlNodeType:: "Sound" fields

void VrmlNodeSound::initFields(VrmlNodeSound *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("direction", node->d_direction),
        exposedField("intensity", node->d_intensity),
        exposedField("location", node->d_location),
        exposedField("maxBack", node->d_maxBack),
        exposedField("maxFront", node->d_maxFront),
        exposedField("minBack", node->d_minBack),
        exposedField("minFront", node->d_minFront),
        exposedField("priority", node->d_priority),
        exposedField("source", node->d_source, [node](auto value)
            { node->source = nullptr; }),
        field("spatialize", node->d_spatialize),
        exposedField("doppler", node->d_doppler));
}

const char *VrmlNodeSound::typeName() { return "Sound"; }

// Should subclass NodeType and have each Sound maintain its own type...
VrmlNodeSound::VrmlNodeSound(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
    , d_direction(0, 0, 1)
    , d_intensity(1)
    , d_maxBack(10)
    , d_maxFront(10)
    , d_minBack(1)
    , d_minFront(1)
    , d_spatialize(true)
    , d_doppler(false)
    , source(nullptr)
{
    forceTraversal(false);
}

VrmlNodeSound::VrmlNodeSound(VrmlNodeSound *sound)
    : VrmlNodeChild(*(VrmlNodeChild *)sound)
    , d_direction(sound->d_direction)
    , d_intensity(sound->d_intensity)
    , d_maxBack(sound->d_maxBack)
    , d_maxFront(sound->d_maxFront)
    , d_minBack(sound->d_minBack)
    , d_minFront(sound->d_minFront)
    , d_spatialize(sound->d_spatialize)
    , d_doppler(sound->d_doppler)
    , source(sound->source)
{
    forceTraversal(false);
}

void VrmlNodeSound::cloneChildren(VrmlNamespace *ns)
{
    if (d_source.get())
    {
        d_source.set(d_source.get()->clone(ns));
        d_source.get()->parentList.push_back(this);
    }
}

void VrmlNodeSound::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_source.get())
        d_source.get()->clearFlags();
}

void VrmlNodeSound::addToScene(VrmlScene *s, const char *relUrl)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_source.get())
        d_source.get()->addToScene(s, relUrl);
    nodeStack.pop_front();
}

void VrmlNodeSound::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_source.get())
        d_source.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

float elliptic_mix(float a, float b, float f)
{
    return a * b / (f * (a - b) + b);
}

void VrmlNodeSound::render(Viewer *viewer)
{
    double timeNow = System::the->time();
    VrmlSFTime now(timeNow);

#ifdef HAVE_AUDIO
    Player *player = System::the->getPlayer();
    if (d_source.get() && player)
    {

        if (d_source.get()->as<VrmlNodeAudioClip>())
        {
            VrmlNodeAudioClip *clip = d_source.get()->as<VrmlNodeAudioClip>();
            if (!source)
            {
                if (clip->getAudio()->numSamples() > 0)
                    source = player->makeSource(clip->getAudio());
            }
            else if (clip->audioLastModified > lastTime)
            {
                // XXX: update source
                source->setAudio(clip->getAudio());
                fprintf(stderr, "source update: lastTime=%f, lastModified=%f\n",
                    lastTime, clip->audioLastModified);
            }

            if (clip->isAudible(now) && source)
            {
                // TODO: what coordinate system is d_location in? can we transform to world coordinates here?
                float x, y, z;
                viewer->getWC(d_location.x(), d_location.y(), d_location.z(), &x, &y, &z);
                // std::cout << "d_location: " << d_location.x() << ", " << d_location.y() << ", " << d_location.z() << ";  WC: " << x << ", " << y << ", " << z << std::endl;
                source->setPosition(x, y, z);

                float intensity = d_intensity.get();
                if (System::the->isCorrectSpatializedAudio() && d_spatialize.get())
                {
                    // Compute the intensity based on the distance and angle to the viewer
                    viewer->getPosition(&x, &y, &z);
                    VrmlSFVec3f toViewer(x, y, z);
                    toViewer.subtract(&d_location); // now we have the vector to the viewer
                    float dist = toViewer.length();
                    toViewer.normalize();
                    d_direction.normalize();
                    float f = toViewer.dot(&d_direction) * 0.5 + 0.5;
                    float rmin = fabs(elliptic_mix(d_minBack.get(), d_minFront.get(), f));
                    float rmax = fabs(elliptic_mix(d_maxBack.get(), d_maxFront.get(), f));

                    intensity *= std::clamp((rmax - dist) / (rmax - rmin), 0.f, 1.f);
                }

                source->setIntensity(intensity);

                if (d_doppler.get())
                {
                    VrmlSFVec3f v = d_location;
                    v.subtract(&lastLocation);
                    v.divide((float)(timeNow - lastTime));
                    source->setVelocity(v.x(), v.y(), v.z());
                }

                if (!source->isPlaying())
                {
                    bool loop = false;
                    if (clip->getField("loop")->toSFBool())
                        loop = clip->getField("loop")->toSFBool()->get();
                    source->setLoop(loop);

                    float pitch = 1.0;
                    if (clip->getField("pitch")->toSFFloat())
                        pitch = clip->getField("pitch")->toSFFloat()->get();
                    source->setPitch(pitch);
                    source->setSpatialize(d_spatialize.get());
                    source->setStart(clip->currentCliptime(now));
                    source->play();
                }
            }
            else
            {
                if (source && source->isPlaying())
                    source->stop();
            }
        }
    }
#endif
    setModified();

    lastLocation = d_location;
    lastTime = timeNow;
}
