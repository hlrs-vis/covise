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

#include "Player.h"

using namespace vrml;

// Sound factory. Add each Sound to the scene for fast access.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSound(scene);
}

// Define the built in VrmlNodeType:: "Sound" fields

VrmlNodeType *VrmlNodeSound::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Sound", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("direction", VrmlField::SFVEC3F);
    t->addExposedField("intensity", VrmlField::SFFLOAT);
    t->addExposedField("location", VrmlField::SFVEC3F);
    t->addExposedField("maxBack", VrmlField::SFFLOAT);
    t->addExposedField("maxFront", VrmlField::SFFLOAT);
    t->addExposedField("minBack", VrmlField::SFFLOAT);
    t->addExposedField("minFront", VrmlField::SFFLOAT);
    t->addExposedField("priority", VrmlField::SFFLOAT);
    t->addExposedField("source", VrmlField::SFNODE);
    t->addField("spatialize", VrmlField::SFBOOL);
    t->addExposedField("doppler", VrmlField::SFBOOL);

    return t;
}

// Should subclass NodeType and have each Sound maintain its own type...

VrmlNodeType *VrmlNodeSound::nodeType() const { return defineType(0); }

VrmlNodeSound::VrmlNodeSound(VrmlScene *scene)
    : VrmlNodeChild(scene)
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

VrmlNode *VrmlNodeSound::cloneMe() const
{
    return new VrmlNodeSound(*this);
}

void VrmlNodeSound::cloneChildren(VrmlNamespace *ns)
{
    if (d_source.get())
    {
        d_source.set(d_source.get()->clone(ns));
        d_source.get()->parentList.push_back(this);
    }
}

VrmlNodeSound *VrmlNodeSound::toSound() const
{
    return (VrmlNodeSound *)this;
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

std::ostream &VrmlNodeSound::printFields(std::ostream &os, int indent)
{
    if (!FPZERO(d_direction.x()) || !FPZERO(d_direction.y()) || !FPEQUAL(d_direction.z(), 1.0))
        PRINT_FIELD(direction);

    // ...

    return os;
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

        if (d_source.get()->toAudioClip())
        {
            VrmlNodeAudioClip *clip = d_source.get()->toAudioClip();
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

// Set the value of one of the node fields/events.
// setField is public so the parser can access it.

void VrmlNodeSound::setField(const char *fieldName,
                             const VrmlField &fieldValue)
{
    if
        TRY_FIELD(direction, SFVec3f)
    else if
        TRY_FIELD(intensity, SFFloat)
    else if
        TRY_FIELD(location, SFVec3f)
    else if
        TRY_FIELD(maxBack, SFFloat)
    else if
        TRY_FIELD(maxFront, SFFloat)
    else if
        TRY_FIELD(minBack, SFFloat)
    else if
        TRY_FIELD(minFront, SFFloat)
    else if
        TRY_FIELD(priority, SFFloat)
    else if
        TRY_SFNODE_FIELD2(source, AudioClip, MovieTexture)
    else if
        TRY_FIELD(spatialize, SFBool)
    else if
        TRY_FIELD(doppler, SFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (!strcmp(fieldName, "source"))
    {
        delete source;
        source = NULL;
    }
}

const VrmlField *VrmlNodeSound::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "direction") == 0)
        return &d_direction;
    else if (strcmp(fieldName, "intensity") == 0)
        return &d_intensity;
    else if (strcmp(fieldName, "location") == 0)
        return &d_location;
    else if (strcmp(fieldName, "maxBack") == 0)
        return &d_maxBack;
    else if (strcmp(fieldName, "maxFront") == 0)
        return &d_maxFront;
    else if (strcmp(fieldName, "minBack") == 0)
        return &d_minBack;
    else if (strcmp(fieldName, "minFront") == 0)
        return &d_minFront;
    else if (strcmp(fieldName, "priority") == 0)
        return &d_priority;
    else if (strcmp(fieldName, "source") == 0)
        return &d_source;
    else if (strcmp(fieldName, "spatialize") == 0)
        return &d_spatialize;
    else if (strcmp(fieldName, "doppler") == 0)
        return &d_doppler;

    return VrmlNodeChild::getField(fieldName);
}
