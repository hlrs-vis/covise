/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePixelTexture.cpp

#include "VrmlNodePixelTexture.h"

#include "VrmlNodeType.h"
#include "VrmlScene.h"

using std::cerr;
using std::endl;

using namespace vrml;

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodePixelTexture(s);
}

VrmlNodePixelTexture *VrmlNodePixelTexture::toPixelTexture() const
{
    return (VrmlNodePixelTexture *)this;
}

// Define the built in VrmlNodeType:: "PixelTexture" fields

VrmlNodeType *VrmlNodePixelTexture::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("PixelTexture", creator);
    }

    VrmlNodeTexture::defineType(t); // Parent class

    t->addExposedField("image", VrmlField::SFIMAGE);
    t->addField("repeatS", VrmlField::SFBOOL);
    t->addField("repeatT", VrmlField::SFBOOL);
    t->addField("blendMode", VrmlField::SFINT32);
    return t;
}

VrmlNodeType *VrmlNodePixelTexture::nodeType() const { return defineType(0); }

VrmlNodePixelTexture::VrmlNodePixelTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene)
    , d_repeatS(true)
    , d_repeatT(true)
    , d_texObject(0)
{
}

VrmlNodePixelTexture::~VrmlNodePixelTexture()
{
    // viewer->removeTextureObject( d_texObject ); ...
}

VrmlNode *VrmlNodePixelTexture::cloneMe() const
{
    return new VrmlNodePixelTexture(*this);
}

std::ostream &VrmlNodePixelTexture::printFields(std::ostream &os, int indent)
{
    if (!d_repeatS.get())
        PRINT_FIELD(repeatS);
    if (!d_repeatT.get())
        PRINT_FIELD(repeatT);
    if (d_image.width() > 0 && d_image.height() > 0 && d_image.nComponents() > 0 && d_image.pixels() != 0)
        PRINT_FIELD(image);
    return os;
}

void VrmlNodePixelTexture::render(Viewer *viewer)
{
    unsigned char *pixels = d_image.pixels();

    if (isModified())
    {
        if (d_texObject)
        {
            viewer->removeTextureObject(d_texObject);
            d_texObject = 0;
        }
    }

    if (pixels)
    {
        if (d_texObject)
        {
            viewer->insertTextureReference(d_texObject, d_image.nComponents(), false, getBlendMode());
        }
        else
        {
            // Ensure the image dimensions are powers of two
            const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
            const int nSizes = sizeof(sizes) / sizeof(int);
            int w = d_image.width();
            int h = d_image.height();
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
                if (w != sizes[i - 1] || h != sizes[j - 1])
                {
                    cerr << endl << "Scaling texture " << endl;
                    viewer->scaleTexture(w, h, sizes[i - 1], sizes[j - 1],
                                         d_image.nComponents(), pixels);
                    d_image.setSize(sizes[i - 1], sizes[j - 1]);
                }

                d_texObject = viewer->insertTexture(d_image.width(),
                                                    d_image.height(),
                                                    d_image.nComponents(),
                                                    d_repeatS.get(),
                                                    d_repeatT.get(),
                                                    pixels,
                                                    true, (getBlendMode() != 0));
            }
        }
    }

    clearModified();
}

int VrmlNodePixelTexture::nComponents()
{
    return d_image.nComponents();
}

int VrmlNodePixelTexture::width()
{
    return d_image.width();
}

int VrmlNodePixelTexture::height()
{
    return d_image.height();
}

int VrmlNodePixelTexture::nFrames()
{
    return 0;
}

unsigned char *VrmlNodePixelTexture::pixels()
{
    return d_image.pixels();
}

// Set the value of one of the node fields.

void VrmlNodePixelTexture::setField(const char *fieldName,
                                    const VrmlField &fieldValue)
{
    if
        TRY_FIELD(image, SFImage)
    else if
        TRY_FIELD(repeatS, SFBool)
    else if
        TRY_FIELD(repeatT, SFBool)
    else if
        TRY_FIELD(blendMode, SFInt)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodePixelTexture::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "image") == 0)
        return &d_image;
    else if (strcmp(fieldName, "repeatS") == 0)
        return &d_repeatS;
    else if (strcmp(fieldName, "repeatT") == 0)
        return &d_repeatT;
    else if (strcmp(fieldName, "blendMode") == 0)
        return &d_blendMode;

    return VrmlNode::getField(fieldName);
}
