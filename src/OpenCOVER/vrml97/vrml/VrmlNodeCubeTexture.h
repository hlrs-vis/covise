/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCubeTexture.h

#ifndef _VRMLNODECubeTexture_
#define _VRMLNODECubeTexture_

#include "VrmlNodeTexture.h"
#include "VrmlMFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFInt.h"

#include "Viewer.h"

namespace vrml
{

class Image;

class VRMLEXPORT VrmlNodeCubeTexture : public VrmlNodeTexture
{

public:
    // Define the fields of CubeTexture nodes
    static void initFields(VrmlNodeCubeTexture *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCubeTexture(VrmlScene *);
    virtual ~VrmlNodeCubeTexture();

    virtual void render(Viewer *);

    virtual int nComponents();
    virtual int width();
    virtual int height();
    virtual int nFrames();
    virtual unsigned char *pixels();

    virtual const VrmlMFString &getUrl() const;

    virtual VrmlNodeCubeTexture *toCubeTexture() const;

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
        return d_urlXP.get();
    }

private:
    void readCubeTexture(Image*& image, VrmlMFString& texture, const std::string &textureName);

    VrmlMFString d_urlXP;
    VrmlMFString d_urlXN;
    VrmlMFString d_urlYP;
    VrmlMFString d_urlYN;
    VrmlMFString d_urlZP;
    VrmlMFString d_urlZN;
    VrmlSFBool d_repeatS;
    VrmlSFBool d_repeatT;

    Image *d_imageXP;
    Image *d_imageXN;
    Image *d_imageYP;
    Image *d_imageYN;
    Image *d_imageZP;
    Image *d_imageZN;

    Viewer::TextureObject d_texObject;
};
}
#endif // _VRMLNODECubeTexture_
