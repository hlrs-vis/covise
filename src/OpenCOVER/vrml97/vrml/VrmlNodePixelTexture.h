/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePixelTexture.h

#ifndef _VRMLNODEPIXELTEXTURE_
#define _VRMLNODEPIXELTEXTURE_

#include "VrmlNodeTexture.h"
#include "VrmlSFImage.h"
#include "VrmlSFBool.h"
#include "Viewer.h"
#include "VrmlSFInt.h"

namespace vrml
{

class VRMLEXPORT VrmlNodePixelTexture : public VrmlNodeTexture
{

public:
    // Define the fields of PixelTexture nodes
    static void initFields(VrmlNodePixelTexture *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodePixelTexture(VrmlScene *);

    virtual void render(Viewer *);

    virtual int nComponents();
    virtual int width();
    virtual int height();
    virtual int nFrames();
    virtual unsigned char *pixels();

    virtual bool getRepeatS()
    {
        return d_repeatS.get();
    }
    virtual bool getRepeatT()
    {
        return d_repeatT.get();
    }

private:
    VrmlSFImage d_image;
    VrmlSFBool d_repeatS;
    VrmlSFBool d_repeatT;

    Viewer::TextureObject d_texObject;
};
}
#endif // _VRMLNODEPIXELTEXTURE_
