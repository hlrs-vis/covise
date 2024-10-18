/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeImageTexture.cpp

#include "VrmlNodeImageTexture.h"

#include "Image.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Doc.h"

using std::cerr;
using std::endl;
using namespace vrml;

bool VrmlNodeImageTexture::scaledown = false;

const VrmlMFString &VrmlNodeImageTexture::getUrl() const
{
    return d_url;
}

VrmlNodeImageTexture *VrmlNodeImageTexture::toImageTexture() const
{
    return (VrmlNodeImageTexture *)this;
}


void VrmlNodeImageTexture::initFields(VrmlNodeImageTexture *node, VrmlNodeType *t)
{
    VrmlNodeTexture::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("url", node->d_url, [node](auto f){
                        delete node->d_image;
                        node->d_image = nullptr;
                     }),
                     field("repeatS", node->d_repeatS),
                     field("repeatT", node->d_repeatT),
                     field("environment", node->d_environment),
                     field("blendMode", node->d_blendMode),
                     field("filterMode", node->d_filterMode),
                     field("anisotropy", node->d_anisotropy));
    
    
    if(t)
    {
        initScaling();
    }
}

const char *VrmlNodeImageTexture::name() { return "ImageTexture"; }

void VrmlNodeImageTexture::initScaling()
{
    scaledown = System::the->getConfigState("COVER.Plugin.Vrml97.SmallTextures", false);
    std::string sizeString = System::the->getConfigEntry("COVER.Plugin.Vrml97.MaxTextureSize");
    if (!sizeString.empty())
    {
        int maxSize = 0;
        sscanf(sizeString.c_str(), "%d", &maxSize);
        if (maxSize > 0)
        {
            setMaxTextureSize(maxSize);
        }
    }
}

VrmlNodeImageTexture::VrmlNodeImageTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene, name())
    , d_repeatS(true)
    , d_repeatT(true)
    , d_environment(false)
    , d_filterMode(0)
    , d_anisotropy(1)
    , d_image(0)
    , d_texObject(0)
{
}

VrmlNodeImageTexture::~VrmlNodeImageTexture()
{
    delete d_image;
    // delete d_texObject...
}

void VrmlNodeImageTexture::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_image)
        {
            delete d_image; // URL is the only modifiable bit
            d_image = 0;
        }
        if (d_texObject)
        {
            viewer->removeTextureObject(d_texObject);
            d_texObject = 0;
        }
    }

    // should probably read the image during addToScene...
    // should cache on url so multiple references to the same file are
    // loaded just once... of course world authors should just DEF/USE
    // them...
    if (!d_image && d_url.size() > 0)
    {
        auto relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_image = new Image;
        if (!d_image->tryURLs(d_url.size(), d_url.get(), &relDoc))
            fprintf(stderr, "Warning: couldn't read ImageTexture from URL %s\n", d_url.get(0));
    }

    // Check texture cache
    if (d_texObject && d_image)
    {
        viewer->insertTextureReference(d_texObject, d_image->nc(), d_environment.get(), getBlendMode());
    }
    else
    {
        unsigned char *pix;

        if (d_image && (pix = d_image->pixels()))
        {
            // Ensure the image dimensions are powers of two
            static const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
            static const int nSizes = sizeof(sizes) / sizeof(int);
            int w = d_image->w();
            int h = d_image->h();
            int i, j;
            if (w == 0 && h == 0)
            {
                d_texObject = viewer->insertTexture(d_image->w(),
                                                    d_image->h(),
                                                    d_image->nc(),
                                                    d_repeatS.get(),
                                                    d_repeatT.get(),
                                                    pix,
                                                   *d_url.get(),
                                                    true, d_environment.get(), getBlendMode(), d_anisotropy.get(), d_filterMode.get());
            }
            else
            {
                for (i = 0; i < nSizes; ++i)
                    if (w < sizes[i] && w <= maxTextureSize())
                        break;
                for (j = 0; j < nSizes; ++j)
                    if (h < sizes[j] && h <= maxTextureSize())
                        break;

                if (i > 0 && j > 0)
                {
                    if ((i > 2) && (scaledown))
                        i--;
                    if ((j > 2) && (scaledown))
                        j--;
                    // Always scale images down in size and reuse the same pixel
                    // memory. This can cause some ugliness...
                    if (w>maxTextureSize() || h>maxTextureSize() || scaledown || (!useTextureNPOT() && (w != sizes[i - 1] || h != sizes[j - 1])))
                    {
                        //cerr << endl<< "Scaling texture " << d_image->url()<< endl;
                        viewer->scaleTexture(w, h, sizes[i - 1], sizes[j - 1],
                                             d_image->nc(), pix);
                        d_image->setSize(sizes[i - 1], sizes[j - 1]);
                    }

                    d_texObject = viewer->insertTexture(d_image->w(),
                                                        d_image->h(),
                                                        d_image->nc(),
                                                        d_repeatS.get(),
                                                        d_repeatT.get(),
                                                        pix,
                                                       *d_url.get(),
                                                        true, d_environment.get(), getBlendMode(), d_anisotropy.get(), d_filterMode.get());
                }
            }
        }
    }

    clearModified();
}

int VrmlNodeImageTexture::nComponents()
{
    return d_image ? d_image->nc() : 0;
}

int VrmlNodeImageTexture::width()
{
    return d_image ? d_image->w() : 0;
}

int VrmlNodeImageTexture::height()
{
    return d_image ? d_image->h() : 0;
}

int VrmlNodeImageTexture::nFrames()
{
    return 0;
}

unsigned char *VrmlNodeImageTexture::pixels()
{
    return d_image ? d_image->pixels() : 0;
}
