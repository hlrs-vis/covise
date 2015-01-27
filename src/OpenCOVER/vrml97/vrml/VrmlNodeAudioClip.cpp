/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeAudioClip.cpp
//    contributed by Kumaran Santhanam

#include "VrmlNodeAudioClip.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "MathUtils.h"
#include "Audio.h"
#include "Doc.h"
#include "VrmlNodeSound.h"

using std::cerr;
using std::endl;
using namespace vrml;

// Render

void VrmlNodeAudioClip::update(VrmlSFTime &now)
{
    // If the URL has been modified, update the audio object
    if (d_url_modified)
    {
        Doc relDoc(d_relativeUrl.get());
        if (d_audio->tryURLs(d_url.size(),
                             d_url.get(),
                             &relDoc))
        {
            d_duration.set(d_audio->duration());
            eventOut(now.get(), "duration_changed", d_duration);
        }
        else
        {
            cerr << "Error: couldn't read AudioClip from URL "
                 << d_url << endl;
        }

        d_url_modified = false;
    }

    // If the clip is audible, play it.  Otherwise, stop it.
    if (lastActive != isAudible(now))
    {
        lastActive = isAudible(now);
        d_isActive.set(lastActive);
        eventOut(now.get(), "isActive", d_isActive);
    }

    setModified();
}

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeAudioClip(s); }

// Define the built in VrmlNodeType:: "AudioClip" fields

VrmlNodeType *VrmlNodeAudioClip::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("AudioClip", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("description", VrmlField::SFSTRING);
    t->addExposedField("loop", VrmlField::SFBOOL);
    t->addExposedField("pitch", VrmlField::SFFLOAT);
    t->addExposedField("startTime", VrmlField::SFTIME);
    t->addExposedField("stopTime", VrmlField::SFTIME);
    t->addExposedField("url", VrmlField::MFSTRING);

    t->addEventOut("duration_changed", VrmlField::SFTIME);
    t->addEventOut("isActive", VrmlField::SFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeAudioClip::nodeType() const { return defineType(0); }

VrmlNodeAudioClip::VrmlNodeAudioClip(VrmlScene *scene)
    : VrmlNode(scene)
    , d_pitch(1.0)
    , d_isActive(false)
    , d_audio(new Audio(0))
    , d_url_modified(false)
    , _doc(0)
    , lastActive(true)
{
    if (d_scene)
        d_scene->addAudioClip(this);
}

// Define copy constructor so clones don't share d_audio (for now, anyway...)

VrmlNodeAudioClip::VrmlNodeAudioClip(const VrmlNodeAudioClip &n)
    : VrmlNode(n)
    , d_description(n.d_description)
    , d_loop(n.d_loop)
    , d_pitch(n.d_pitch)
    , d_startTime(n.d_startTime)
    , d_stopTime(n.d_stopTime)
    , d_url(n.d_url)
    , d_duration(n.d_duration)
    , d_isActive(false)
    , d_audio(new Audio(*n.d_audio))
    , d_url_modified(true)
    , _doc(0)
    , lastActive(true)
{
    if (d_scene)
        d_scene->addAudioClip(this);
}

VrmlNodeAudioClip::~VrmlNodeAudioClip()
{
    if (d_scene)
        d_scene->removeAudioClip(this);
    delete d_audio;
}

VrmlNode *VrmlNodeAudioClip::cloneMe() const
{
    return new VrmlNodeAudioClip(*this);
}

void VrmlNodeAudioClip::addToScene(VrmlScene *s, const char *rel)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addAudioClip(this);
    if (d_relativeUrl.get() == NULL)
        d_relativeUrl.set(rel);
}

VrmlNodeAudioClip *VrmlNodeAudioClip::toAudioClip() const
{
    return (VrmlNodeAudioClip *)this;
}

std::ostream &VrmlNodeAudioClip::printFields(std::ostream &os, int indent)
{
    if (d_description.get())
        PRINT_FIELD(description);
    if (d_loop.get())
        PRINT_FIELD(loop);
    if (!FPEQUAL(d_pitch.get(), 1.0))
        PRINT_FIELD(pitch);
    if (!FPZERO(d_startTime.get()))
        PRINT_FIELD(startTime);
    if (!FPZERO(d_stopTime.get()))
        PRINT_FIELD(stopTime);
    if (d_url.size() > 0)
        PRINT_FIELD(url);
    return os;
}

const Audio *VrmlNodeAudioClip::getAudio() const
{
    return d_audio;
}

bool VrmlNodeAudioClip::isAudible(VrmlSFTime &inTime) const
{
    // Determine if this clip should be playing right now

    // If there's no audio or START <= 0, we don't play anything
    if (d_audio == 0 || d_startTime.get() <= 0)
        return false;

    bool audible = false;
    // If START < STOP  and  START <= NOW < STOP
    if (d_stopTime.get() > d_startTime.get())
        audible = (d_startTime.get() <= inTime.get() && inTime.get() < d_stopTime.get());

    // If STOP < START  and  START <= NOW
    else
        audible = (inTime.get() >= d_startTime.get());

    // If the clip is not looping, it's not audible after
    // its duration has expired.
    if (d_loop.get() == false)
        if (inTime.get() - d_startTime.get() > d_audio->duration())
            audible = false;

    return audible;
}

double VrmlNodeAudioClip::currentCliptime(VrmlSFTime &inTime) const
{
    double cliptime = 0.0;
    if (isAudible(inTime))
        cliptime = fmod((inTime.get() - d_startTime.get()) * d_pitch.get(),
                        d_audio->duration());

    return cliptime;
}

// Set the value of one of the node fields.

void VrmlNodeAudioClip::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(description, SFString)
    else if
        TRY_FIELD(loop, SFBool)
    else if
        TRY_FIELD(pitch, SFFloat)
    else if
        TRY_FIELD(startTime, SFTime)
    else if
        TRY_FIELD(stopTime, SFTime)
    else if
        TRY_FIELD(url, MFString)
    else
        VrmlNode::setField(fieldName, fieldValue);

    if (strcmp("url", fieldName) == 0)
    {
        d_url_modified = true;
        setModified();
    }
}

const VrmlField *VrmlNodeAudioClip::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "description") == 0)
        return &d_description;
    else if (strcmp(fieldName, "loop") == 0)
        return &d_loop;
    else if (strcmp(fieldName, "pitch") == 0)
        return &d_pitch;
    else if (strcmp(fieldName, "startTime") == 0)
        return &d_startTime;
    else if (strcmp(fieldName, "stopTime") == 0)
        return &d_stopTime;
    else if (strcmp(fieldName, "url") == 0)
        return &d_url;

    return VrmlNode::getField(fieldName);
}
