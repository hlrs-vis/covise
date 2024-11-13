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

#include "Player.h"

using namespace vrml;

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
                     exposedField("source", node->d_source, [node](auto value){
                        delete node->source;
                        node->source = NULL;
                     }),
                     field("spatialize", node->d_spatialize),
                     exposedField("doppler", node->d_doppler));

}

const char *VrmlNodeSound::name() { return "Sound"; }

// Should subclass NodeType and have each Sound maintain its own type...
VrmlNodeSound::VrmlNodeSound(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_direction(0, 0, 1)
    , d_intensity(1)
    , d_maxBack(10)
    , d_maxFront(10)
    , d_minBack(1)
    , d_minFront(1)
    , d_spatialize(true)
    , d_doppler(false)
    , source(0)
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

VrmlNodeSound::~VrmlNodeSound()
{
    delete source;
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

void VrmlNodeSound::render(Viewer *viewer)
{
    double timeNow = System::the->time();
    VrmlSFTime now(timeNow);

    if (d_source.get() && viewer->getPlayer())
    {
        Player *player = viewer->getPlayer();
        float x, y, z;

        // Is viewer inside the box?
        viewer->getPosition(&x, &y, &z);
        VrmlSFVec3f toViewer(x, y, z);
        toViewer.subtract(&d_location); // now we have the vector to the viewer
        float dist = (float)toViewer.length();
        toViewer.normalize();
        d_direction.normalize();
        // angle between the sound direction and the viewer
        float angle = (float)acos(toViewer.dot(&d_direction));
        //fprintf(stderr,"angle: %f",angle/M_PI*180.0);
        float cang = (float)cos(angle / 2.0);
        float rmin, rmax;
        double intensity;
        rmin = fabs(d_minBack.get() * d_minFront.get() / (cang * cang * (d_minBack.get() - d_minFront.get()) + d_minFront.get()));
        rmax = fabs(d_maxBack.get() * d_maxFront.get() / (cang * cang * (d_maxBack.get() - d_maxFront.get()) + d_maxFront.get()));
        //fprintf(stderr,"rmin: %f rmax: %f",rmin,rmax);
        if (dist <= rmin)
            intensity = 1.0;
        else if (dist > rmax)
            intensity = 0.0;
        else
        {
            intensity = (rmax - dist) / (rmax - rmin);
        }

        if (d_source.get()->as<VrmlNodeAudioClip>())
        {
            VrmlNodeAudioClip *clip = d_source.get()->as<VrmlNodeAudioClip>();
            if (!source)
            {
                if (clip->getAudio()->numSamples() > 0)
                    source = player->newSource(clip->getAudio());
            }
            else if (clip->getAudio()->lastModified() > lastTime)
            {
                // XXX: update source
                source->setAudio(clip->getAudio());
                fprintf(stderr, "source update: lastTime=%f, lastModified=%f\n",
                        lastTime, clip->getAudio()->lastModified());
            }
            if (clip->isAudible(now) && source)
            {
                source->setPositionOC(d_location.x(), d_location.y(), d_location.z());

                //fprintf(stderr, "intens=%f\n", intensity*d_intensity.get());
                if (!System::the->isCorrectSpatializedAudio() && !d_spatialize.get())
                    source->setIntensity(d_intensity.get());
                else
                    source->setIntensity((float)(intensity * d_intensity.get()));

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

                    if (d_spatialize.get())
                        source->setSpatialize(true);
                    else
                        source->setSpatialize(false);

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
    setModified();

    lastLocation = d_location;
    lastTime = timeNow;
}
