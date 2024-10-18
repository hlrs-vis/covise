/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeImageTexture.h

#ifndef _VRMLNODEIMAGETEXTURE_
#define _VRMLNODEIMAGETEXTURE_

#include "VrmlNodeTexture.h"
#include "VrmlMFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFInt.h"

#include "Viewer.h"

namespace vrml
{

class Image;

class VRMLEXPORT VrmlNodeImageTexture : public VrmlNodeTexture
{

public:
    // Define the fields of ImageTexture nodes
    static void initFields(VrmlNodeImageTexture *node, VrmlNodeType *t);
    static const char *name();

    static void initScaling();
    VrmlNodeImageTexture(VrmlScene *);
    virtual ~VrmlNodeImageTexture();

    virtual void render(Viewer *);

    virtual int nComponents();
    virtual int width();
    virtual int height();
    virtual int nFrames();
    virtual unsigned char *pixels();

    virtual const VrmlMFString &getUrl() const;

    virtual VrmlNodeImageTexture *toImageTexture() const;

    virtual bool getRepeatS() // LarryD Feb18/99
    {
        return d_repeatS.get();
    }
    virtual bool getRepeatT() // LarryD Feb18/99
    {
        return d_repeatT.get();
    }
    char **getURL() // LarryD  Feb18/99
    {
        return d_url.get();
    }

private:
    VrmlMFString d_url;
    VrmlSFBool d_repeatS;
    VrmlSFBool d_repeatT;
    VrmlSFBool d_environment;
    VrmlSFInt d_filterMode;
    VrmlSFInt d_anisotropy;

    Image *d_image;

    Viewer::TextureObject d_texObject;
    static bool scaledown;
};
}
#endif // _VRMLNODEIMAGETEXTURE_
