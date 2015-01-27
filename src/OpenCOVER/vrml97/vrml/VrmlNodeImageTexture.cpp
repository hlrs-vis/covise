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
int maxTextureSize = 65536;
static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeImageTexture(s);
}

const VrmlMFString &VrmlNodeImageTexture::getUrl() const
{
    return d_url;
}

VrmlNodeImageTexture *VrmlNodeImageTexture::toImageTexture() const
{
    return (VrmlNodeImageTexture *)this;
}

// Define the built in VrmlNodeType:: "ImageTexture" fields

VrmlNodeType *VrmlNodeImageTexture::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("ImageTexture", creator);
        scaledown = System::the->getConfigState("COVER.Plugin.Vrml97.SmallTextures", false);
        std::string sizeString = System::the->getConfigEntry("COVER.Plugin.Vrml97.MaxTextureSize");
        if (!sizeString.empty())
        {
            sscanf(sizeString.c_str(), "%d", &maxTextureSize);
        }
        //cerr << "Scaledown" << scaledown << endl;
    }

    VrmlNodeTexture::defineType(t); // Parent class

    t->addExposedField("url", VrmlField::MFSTRING);
    t->addField("repeatS", VrmlField::SFBOOL);
    t->addField("repeatT", VrmlField::SFBOOL);
    t->addField("environment", VrmlField::SFBOOL);
    t->addField("blendMode", VrmlField::SFINT32);
    t->addField("filterMode", VrmlField::SFINT32);
    t->addField("anisotropy", VrmlField::SFINT32);

    return t;
}

VrmlNodeImageTexture::VrmlNodeImageTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene)
    , d_repeatS(true)
    , d_repeatT(true)
    , d_environment(false)
    , d_filterMode(0)
    , d_anisotropy(1)
    , d_image(0)
    , d_texObject(0)
{
}

VrmlNodeType *VrmlNodeImageTexture::nodeType() const { return defineType(0); }

VrmlNodeImageTexture::~VrmlNodeImageTexture()
{
    delete d_image;
    // delete d_texObject...
}

VrmlNode *VrmlNodeImageTexture::cloneMe() const
{
    return new VrmlNodeImageTexture(*this);
}

std::ostream &VrmlNodeImageTexture::printFields(std::ostream &os, int indent)
{
    if (d_url.get())
        PRINT_FIELD(url);
    if (!d_repeatS.get())
        PRINT_FIELD(repeatS);
    if (!d_repeatT.get())
        PRINT_FIELD(repeatT);
    if (d_environment.get())
        PRINT_FIELD(environment);
    return os;
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
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
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
            int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
            int nSizes = sizeof(sizes) / sizeof(int);
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
                                                    true, d_environment.get(), getBlendMode(), d_anisotropy.get(), d_filterMode.get());
            }
            else
            {
                for (i = 0; i < nSizes; ++i)
                    if (w < sizes[i] && w <= maxTextureSize)
                        break;
                for (j = 0; j < nSizes; ++j)
                    if (h < sizes[j] && h <= maxTextureSize)
                        break;

                if (i > 0 && j > 0)
                {
                    if ((i > 2) && (scaledown))
                        i--;
                    if ((j > 2) && (scaledown))
                        j--;
                    // Always scale images down in size and reuse the same pixel
                    // memory. This can cause some ugliness...
                    if (w != sizes[i - 1] || h != sizes[j - 1])
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

// Set the value of one of the node fields.

void VrmlNodeImageTexture::setField(const char *fieldName,
                                    const VrmlField &fieldValue)
{
    if (strcmp(fieldName, "url") == 0)
    {
        delete d_image;
        d_image = 0;
    }

    if
        TRY_FIELD(url, MFString)
    else if
        TRY_FIELD(repeatS, SFBool)
    else if
        TRY_FIELD(repeatT, SFBool)
    else if
        TRY_FIELD(environment, SFBool)
    else if
        TRY_FIELD(blendMode, SFInt)
    else if
        TRY_FIELD(filterMode, SFInt)
    else if
        TRY_FIELD(anisotropy, SFInt)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeImageTexture::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "url") == 0)
        return &d_url;
    else if (strcmp(fieldName, "repeatS") == 0)
        return &d_repeatS;
    else if (strcmp(fieldName, "repeatT") == 0)
        return &d_repeatT;
    else if (strcmp(fieldName, "environment") == 0)
        return &d_environment;
    else if (strcmp(fieldName, "filterMode") == 0)
        return &d_filterMode;
    else if (strcmp(fieldName, "anisotropy") == 0)
        return &d_anisotropy;
    else if (strcmp(fieldName, "blendMode") == 0)
        return &d_blendMode;

    return VrmlNode::getField(fieldName);
}
