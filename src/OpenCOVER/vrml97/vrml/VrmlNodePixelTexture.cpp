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

// Define the built in VrmlNodeType:: "PixelTexture" fields

void VrmlNodePixelTexture::initFields(VrmlNodePixelTexture *node, VrmlNodeType *t)
{
    VrmlNodeTexture::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("image", node->d_image),
                     field("repeatS", node->d_repeatS),
                     field("repeatT", node->d_repeatT),
                     field("blendMode", node->d_blendMode));
}

const char *VrmlNodePixelTexture::name() { return "PixelTexture"; }

VrmlNodePixelTexture::VrmlNodePixelTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene, name())
    , d_repeatS(true)
    , d_repeatT(true)
    , d_texObject(0)
{
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
            static const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
            static const int nSizes = sizeof(sizes) / sizeof(int);
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
                if (w>maxTextureSize() || h>maxTextureSize() || (!useTextureNPOT() && (w != sizes[i - 1] || h != sizes[j - 1])))
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
                                                    "",
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
