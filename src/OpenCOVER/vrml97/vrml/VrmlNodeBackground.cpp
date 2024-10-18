/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBackground.cpp
//

#include "VrmlNodeBackground.h"

#include "Doc.h"
#include "Image.h"
#include "VrmlNodeType.h"
#include "VrmlSFBool.h"
#include "VrmlScene.h"
#include "Viewer.h"

using std::cerr;
using std::endl;
using namespace vrml;

//  Background factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeBackground(scene);
}

// Define the built in VrmlNodeType:: "Background" fields

void VrmlNodeBackground::initFields(VrmlNodeBackground *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    
    initFieldsHelper(node, t,
                     exposedField("groundAngle", node->d_groundAngle),
                     exposedField("groundColor", node->d_groundColor),
                     exposedField("backUrl", node->d_backUrl),
                     exposedField("bottomUrl", node->d_bottomUrl),
                     exposedField("frontUrl", node->d_frontUrl),
                     exposedField("leftUrl", node->d_leftUrl),
                     exposedField("rightUrl", node->d_rightUrl),
                     exposedField("topUrl", node->d_topUrl),
                     exposedField("skyAngle", node->d_skyAngle),
                     exposedField("skyColor", node->d_skyColor));
    
    if (t)
    {

        t->addEventIn("set_bind", VrmlField::SFBOOL);
        t->addEventOut("isBound", VrmlField::SFBOOL);
    }
}

const char *VrmlNodeBackground::name() { return "Background"; }

VrmlNodeBackground::VrmlNodeBackground(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_viewerObject(0)
{
    for (int i = 0; i < 6; ++i)
        d_texPtr[i] = 0;
    if (d_scene)
        d_scene->addBackground(this);
}

VrmlNodeBackground::~VrmlNodeBackground()
{
    if (d_scene)
        d_scene->removeBackground(this);
    // remove d_viewerObject...
}

VrmlNodeBackground *VrmlNodeBackground::toBackground() const
{
    return (VrmlNodeBackground *)this;
}

void VrmlNodeBackground::addToScene(VrmlScene *s, const char *rel)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addBackground(this);
    d_relativeUrl.set(rel);
}

// Load and scale textures as needed.

static Image *getTexture(VrmlMFString &urls,
                         Doc *relative,
                         Image *tex,
                         int thisIndex,
                         Viewer *viewer)
{
    // Check whether the url has already been loaded
    int n = urls.size();
    if (n > 0)
    {
        for (int index = thisIndex - 1; index >= 0; --index)
        {
            const char *currentTex = tex[index].url();
            auto relPath = relative ? relative->urlPath() : "";
            int currentLen = (int)(currentTex ? strlen(currentTex) : 0);
            int relPathLen = relPath.length();
            if (relPathLen >= currentLen)
                relPathLen = 0;

            if (currentTex)
                for (int i = 0; i < n; ++i)
                    if (strcmp(currentTex, urls[i]) == 0 || strcmp(currentTex + relPathLen, urls[i]) == 0)
                        return &tex[index];
        }

        // Have to load it
        if (!tex[thisIndex].tryURLs(n, &urls[0], relative))
            cerr << "Error: couldn't read Background texture from URL "
                 << urls << endl;

        // check whether it needs to be scaled
        else if (tex[thisIndex].pixels() && tex[thisIndex].nc())
        {
            // Ensure the image dimensions are powers of two
            static const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
            static const int nSizes = sizeof(sizes) / sizeof(int);
            int w = tex[thisIndex].w();
            int h = tex[thisIndex].h();
            int i, j;
            for (i = 0; i < nSizes; ++i)
                if (w < sizes[i])
                    break;
            for (j = 0; j < nSizes; ++j)
                if (h < sizes[j])
                    break;

            if (i > 0 && j > 0)
            {
                // Always scale images down in size and reuse the same pixel
                // memory. This can cause some ugliness...
                if (w != sizes[i - 1] || h != sizes[j - 1])
                {
                    //cerr << endl << "Scaling texture " << tex[thisIndex].url() << endl;
                    viewer->scaleTexture(w, h, sizes[i - 1], sizes[j - 1],
                                         tex[thisIndex].nc(),
                                         tex[thisIndex].pixels());
                    tex[thisIndex].setSize(sizes[i - 1], sizes[j - 1]);
                }
            }
        }
    }

    return &tex[thisIndex];
}

// Backgrounds are rendered once per scene at the beginning, not
// when they are traversed by the standard render() method.

void VrmlNodeBackground::renderBindable(Viewer *viewer)
{
#ifdef DEBUG
    cout << "renderBindable obj " << d_viewerObject
         << " mod " << isModified()
         << " skyColors " << d_skyColor
         << " skyAngles " << d_skyAngle << endl;
#endif

    // Background isn't selectable, so don't waste the time.
    if (viewer->getRenderMode() == Viewer::RENDER_MODE_PICK)
        return;

    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }

    if (d_viewerObject)
        viewer->insertReference(d_viewerObject);
    else
    {
        if (isModified() || d_texPtr[0] == 0)
        {
            Doc relDoc(d_relativeUrl.get());
            Doc *rel = d_relativeUrl.get() ? &relDoc : d_scene->urlDoc();
            d_texPtr[0] = getTexture(d_backUrl, rel, d_tex, 0, viewer);
            d_texPtr[1] = getTexture(d_bottomUrl, rel, d_tex, 1, viewer);
            d_texPtr[2] = getTexture(d_frontUrl, rel, d_tex, 2, viewer);
            d_texPtr[3] = getTexture(d_leftUrl, rel, d_tex, 3, viewer);
            d_texPtr[4] = getTexture(d_rightUrl, rel, d_tex, 4, viewer);
            d_texPtr[5] = getTexture(d_topUrl, rel, d_tex, 5, viewer);
        }

        int i, whc[18]; // Width, height, and nComponents for 6 tex
        unsigned char *pixels[6];
        int nPix = 0;

        for (i = 0; i < 6; ++i)
        {
            whc[3 * i + 0] = d_texPtr[i]->w();
            whc[3 * i + 1] = d_texPtr[i]->h();
            whc[3 * i + 2] = d_texPtr[i]->nc();
            pixels[i] = d_texPtr[i]->pixels();
            if (whc[3 * i + 0] > 0 && whc[3 * i + 1] > 0 && whc[3 * i + 2] > 0 && pixels[i])
                ++nPix;
        }

        d_viewerObject = viewer->insertBackground(nGroundAngles(),
                                                  groundAngle(),
                                                  groundColor(),
                                                  nSkyAngles(),
                                                  skyAngle(),
                                                  skyColor(),
                                                  whc,
                                                  (nPix > 0) ? pixels : 0);

        clearModified();
    }
}

void VrmlNodeBackground::eventIn(double timeStamp,
                                 const char *eventName,
                                 const VrmlField *fieldValue)
{
    if (strcmp(eventName, "set_bind") == 0)
    {
        VrmlNodeBackground *current = d_scene->bindableBackgroundTop();
        const VrmlSFBool *b = fieldValue->toSFBool();
        if (!b)
        {
            cerr << "Error: invalid value for Background::set_bind eventIn "
                 << (*fieldValue) << endl;
            return;
        }

        if (b->get()) // set_bind TRUE
        {
            if (this != current)
            {
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(false));
                d_scene->bindablePush(this);
                eventOut(timeStamp, "isBound", VrmlSFBool(true));
            }
        }
        else // set_bind FALSE
        {
            d_scene->bindableRemove(this);
            if (this == current)
            {
                eventOut(timeStamp, "isBound", VrmlSFBool(false));
                current = d_scene->bindableBackgroundTop();
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(true));
            }
        }
    }

    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}
