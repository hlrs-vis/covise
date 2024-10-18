/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMovieTexture.cpp

#include "VrmlNodeMovieTexture.h"

#include "Image.h"
#include "MathUtils.h"
#include "System.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Viewer.h"
#include "Doc.h"

using std::cerr;
using std::endl;
using namespace vrml;

void VrmlNodeMovieTexture::initFields(VrmlNodeMovieTexture *node, VrmlNodeType *t)
{
    VrmlNodeTexture::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("loop", node->d_loop),
                     exposedField("speed", node->d_speed),
                     exposedField("startTime", node->d_startTime),
                     exposedField("stopTime", node->d_stopTime),
                     exposedField("url", node->d_url),
                     field("repeatS", node->d_repeatS),
                    field("repeatT", node->d_repeatT),
                    field("blendMode", node->d_blendMode),
                    field("environment", node->d_environment),
                    field("anisotropy", node->d_anisotropy),
                    field("filter", node->d_filter));
                    
    if (t)
    {
        t->addEventOut("duration_changed", VrmlField::SFTIME);
        t->addEventOut("isActive", VrmlField::SFBOOL);
    }
}

const char *VrmlNodeMovieTexture::name(){ return "MovieTexture";}

VrmlNodeMovieTexture::VrmlNodeMovieTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene, name())
    , d_loop(false)
    , d_speed(1.0)
    , d_repeatS(true)
    , d_repeatT(true)
    , d_environment(false)
    , d_anisotropy(1)
    , d_filter(0)
    , d_image(0)
    , d_frame(0)
    , d_lastFrame(-1)
    , d_lastFrameTime(-1.0)
    , d_texObject(0)
{
    imageChanged = false;
    mpegStaticLength = 0.;
    std::string lenString = System::the->getConfigEntry("COVER.Plugin.Vrml97.MpgStaticLength");
    if (!lenString.empty())
    {
        sscanf(lenString.c_str(), "%f", &mpegStaticLength);
    }
    if (d_scene)
        d_scene->addMovie(this);
}

VrmlNodeMovieTexture::~VrmlNodeMovieTexture()
{
    if (d_scene)
        d_scene->removeMovie(this);
    delete d_image;
}

VrmlNodeMovieTexture *VrmlNodeMovieTexture::toMovieTexture() const
{
    return (VrmlNodeMovieTexture *)this;
}

void VrmlNodeMovieTexture::addToScene(VrmlScene *s, const char *rel)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addMovie(this);
    VrmlNodeTexture::addToScene(s, rel);
}

void VrmlNodeMovieTexture::update(VrmlSFTime &timeNow)
{
    if (isModified())
    {
        if (d_image)
        {
            const char *imageUrl = d_image->url();
            int imageLen = (int)strlen(imageUrl);
            int i, nUrls = d_url.size();
            for (i = 0; i < nUrls; ++i)
            {
                int len = (int)strlen(d_url[i]);

                if ((strcmp(imageUrl, d_url[i]) == 0) || (imageLen > len && strcmp(imageUrl + imageLen - len, d_url[i]) == 0))
                    break;
            }

            // if (d_image->url() not in d_url list) ...
            if (i == nUrls)
            {
                delete d_image;
                imageChanged = true;
                d_image = 0;
            }
        }
    }

    // Load the movie if needed (should check startTime...)
    if (!d_image && d_url.size() > 0)
    {
        Doc relDoc(d_relativeUrl.get());
        Doc *rel = d_relativeUrl.get() ? &relDoc : d_scene->urlDoc();
        d_image = new Image;
        if (!d_image->tryURLs(d_url.size(), d_url.get(), rel))
            cerr << "Error: couldn't read MovieTexture from URL " << d_url << endl;

        int nFrames = d_image->nFrames();
        d_duration = (nFrames >= 0) ? nFrames : -1;
        eventOut(timeNow.get(), "duration_changed", d_duration);
        d_frame = (d_speed.get() >= 0) ? 0 : nFrames - 1;

        //System::the->debug("MovieTexture.%s loaded %d frames\n", name(), nFrames);
    }
    if (d_image->nc() != -1)
    {
        // No pictures to show
        if (!d_image || d_image->nFrames() == 0 || d_startTime.get() < 0)
            return;

        // Become active at the first tick at or after startTime if either
        // the valid stopTime hasn't passed or we are looping.
        /* if (! d_isActive.get() &&
        d_startTime.get() <= timeNow.get() &&
        ((d_startTime.get() >= d_lastFrameTime)||(d_startTime.get() <= 0)) &&
        ( (d_stopTime.get() < d_startTime.get() || // valid stopTime
      d_stopTime.get() > timeNow.get()) ||    // hasn't passed
     d_loop.get() ))
     */
        if (!d_isActive.get() && d_startTime.get() <= timeNow.get() && ((d_stopTime.get() < d_startTime.get() || d_stopTime.get() > timeNow.get()) || d_loop.get()))
        {
            //System::the->debug("MovieTexture.%s::isActive TRUE\n", name());
            d_isActive.set(true);
            eventOut(timeNow.get(), "isActive", d_isActive);
            d_lastFrameTime = timeNow.get();
            d_frame = (d_speed.get() >= 0) ? 0 : d_image->nFrames() - 1;
            setModified();
        }

        // Check whether stopTime has passed
        else if (d_isActive.get() && ((d_stopTime.get() > d_startTime.get() && d_stopTime.get() <= timeNow.get()) || d_frame < 0))
        {
            //System::the->debug("MovieTexture.%s::isActive FALSE\n", name());
            d_isActive.set(false);
            eventOut(timeNow.get(), "isActive", d_isActive);
            setModified();
        }

        // Check whether the frame should be advanced
        else if (d_isActive.get() && d_lastFrameTime + fabs(d_speed.get()) <= timeNow.get())
        {
            if (d_speed.get() < 0.0)
                --d_frame;
            else
            {
                if (mpegStaticLength > 0.)
                    d_frame = (int)((timeNow.get() - d_startTime.get()) * d_image->nFrames() / mpegStaticLength);
                else
                    ++d_frame;
            }
            // cerr << "movieTexture " << d_frame << " " << d_image->nFrames() <<" "<< d_speed.get() << " " << timeNow.get()-d_startTime.get() << endl;
            //System::the->debug("MovieTexture.%s::frame %d\n", name(), d_frame);
            d_lastFrameTime = timeNow.get();
            setModified();
        }

        // Tell the scene when the next update is needed.
        if (d_isActive.get())
        {
            double d = d_lastFrameTime + fabs(d_speed.get()) - timeNow.get();
            d_scene->setDelta(0.9 * d);
        }
        if (d_image->nc() == 0)
            fprintf(stderr, "oops\n");
    }
    // viewer-> updateMovieParam(d_viewerObject,movpar)
}

// Ignore set_speed when active.

void VrmlNodeMovieTexture::eventIn(double timeStamp,
                                   const char *eventName,
                                   const VrmlField *fieldValue)
{
    const char *origEventName = eventName;
    if (strncmp(eventName, "set_", 4) == 0)
        eventName += 4;

    // Ignore set_speed when active
    if (strcmp(eventName, "speed") == 0)
    {
        if (!d_isActive.get())
        {
            setFieldByName(eventName, *fieldValue);
            eventOut(timeStamp, "speed_changed", *fieldValue);
            movProp.speed = d_speed.get();
            setModified();
        }
    }
    else if (strcmp(eventName, "startTime") == 0)
    {
        if (!d_isActive.get())
        {
            setFieldByName(eventName, *fieldValue);
            eventOut(timeStamp, "start_changed", *fieldValue);
            movProp.start = d_startTime.get();
            setModified();
        }
    }
    else if (strcmp(eventName, "stopTime") == 0)
    {
        if (!d_isActive.get())
        {
            setFieldByName(eventName, *fieldValue);
            eventOut(timeStamp, "stop_changed", *fieldValue);
            movProp.stop = d_stopTime.get();
            setModified();
        }
    }

    // Let the generic code handle the rest.
    else
        VrmlNode::eventIn(timeStamp, origEventName, fieldValue);
}

// Render a frame if there is one available.

void VrmlNodeMovieTexture::render(Viewer *viewer)
{
    //System::the->debug("MovieTexture.%s::render frame %d\n", name(), d_frame);
    //   if ( ! d_image || d_frame < 0 ) return;

    unsigned char *pix = d_image->pixels(d_frame);

    if ((d_frame != d_lastFrame && d_texObject) || imageChanged)
    {
        viewer->removeTextureObject(d_texObject);
        d_texObject = 0;
    }

    if (d_image->nc() == -1)
    {
        // MPEG
        if (d_texObject)
        {
            viewer->insertMovieReference(d_texObject, d_image->nc(), d_environment.get(), getBlendMode());
            System::the->inform("Reference found\n");
        }
        else
        {
            movProp.loop = d_loop.get();
            movProp.speed = d_speed.get();
            movProp.start = d_startTime.get();
            movProp.stop = d_stopTime.get();
            movProp.repeatS = d_repeatS.get();
            movProp.repeatT = d_repeatT.get();
            movProp.mtNode = this;
            System::the->inform("created Movie\n");
            d_texObject = viewer->insertMovieTexture((char *)d_image->pixels(), &movProp, d_image->nc(), false, d_environment.get(), getBlendMode(), d_anisotropy.get(), d_filter.get());
            if (imageChanged)
            {
                viewer->setTextureTransform(NULL, 0, NULL, NULL);
                imageChanged = false;
            }
        }
    }
    else if (!pix)
    {
        d_frame = -1;
    }
    else if (d_texObject)
    {
        viewer->insertTextureReference(d_texObject, d_image->nc(), false, getBlendMode());
        System::the->inform("Reference found\n");
    }
    else
    {
        // Ensure image dimensions are powers of 2 (move to VrmlNodeTexture...)
        static const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
        static const int nSizes = sizeof(sizes) / sizeof(int);
        int w = d_image->w();
        int h = d_image->h();
        int i, j;
        for (i = 0; i < nSizes; ++i)
            if (w < sizes[i])
                break;
        for (j = 0; j < nSizes; ++j)
            if (h < sizes[j])
                break;

        if (i > 0 && j > 0)
        {
            // Always scale images down in size and reuse the same pixel memory.
            if (w>maxTextureSize() || h>maxTextureSize() || (!useTextureNPOT() && (w != sizes[i - 1] || h != sizes[j - 1])))
            {
                cerr << endl << "Scaling texture " << d_image->url() << endl;
                viewer->scaleTexture(w, h, sizes[i - 1], sizes[j - 1],
                                     d_image->nc(), pix);
                d_image->setSize(sizes[i - 1], sizes[j - 1]);
            }

            System::the->inform("num components%d\n", d_image->nc());
            d_texObject = viewer->insertTexture(d_image->w(),
                                                d_image->h(),
                                                d_image->nc(),
                                                d_repeatS.get(),
                                                d_repeatT.get(),
                                                pix,
                                                d_image->url(),
                                                !d_isActive.get(), getBlendMode() != 0);
        }
    }

    d_lastFrame = d_frame;
    clearModified();
}

int VrmlNodeMovieTexture::nComponents()
{
    return d_image ? d_image->nc() : 0;
}

int VrmlNodeMovieTexture::width()
{
    return d_image ? d_image->w() : 0;
}

int VrmlNodeMovieTexture::height()
{
    return d_image ? d_image->h() : 0;
}

int VrmlNodeMovieTexture::nFrames()
{
    return d_image ? d_image->nFrames() : 0;
}

void VrmlNodeMovieTexture::stopped()
{
    double timeNow = System::the->time();
    d_stopTime.set(timeNow);
    eventOut(timeNow, "stopTime_changed", d_stopTime);
    setModified();
}

unsigned char *VrmlNodeMovieTexture::pixels()
{
    return d_image ? d_image->pixels() : 0;
}

// Get the value of a field or eventOut.

const VrmlField *VrmlNodeMovieTexture::getField(const char *fieldName) const
{
    // eventOuts
    if (strcmp(fieldName, "duration") == 0)
        return &d_duration;
    else if (strcmp(fieldName, "isActive") == 0)
        return &d_isActive;

    return VrmlNodeTemplate::getField(fieldName); // Parent class
}

// Set the value of one of the node fields.

bool VrmlNodeMovieTexture::isOnlyGeometry() const
{
    if (!VrmlNodeTexture::isOnlyGeometry())
    {
        return false;
    }
    return false;
}
