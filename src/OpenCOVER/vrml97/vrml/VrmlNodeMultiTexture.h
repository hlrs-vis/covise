/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMultiTexture.h

//  X3D MultiTexture

#ifndef _VRMLNODEMULTITEXTURE_
#define _VRMLNODEMULTITEXTURE_

#include "VrmlNodeTexture.h"

#include "VrmlSFFloat.h"
#include "VrmlSFColor.h"
#include "VrmlMFString.h"
#include "VrmlMFNode.h"

#include "Viewer.h"

namespace vrml
{

class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeMultiTexture : public VrmlNodeTexture
{

public:
    // Define the fields of MultiTexture nodes
    static void initFields(VrmlNodeMultiTexture *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeMultiTexture(VrmlScene *);
    virtual ~VrmlNodeMultiTexture();

    virtual void cloneChildren(VrmlNamespace *);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual void clearFlags(); // Clear childrens flags too.

    virtual bool isModified() const;


    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void render(Viewer *);

    virtual VrmlNodeMultiTexture *toMultiTexture() const;

    VrmlNode *texture();
    VrmlNode *textureTransform();

    virtual void setAppearance(VrmlNode *appearance)
    {
        d_appearance = appearance;
    }

    // useless dummies
    virtual int nComponents()
    {
        return 0;
    }
    virtual int width()
    {
        return 0;
    }
    virtual int height()
    {
        return 0;
    }
    virtual int nFrames()
    {
        return 0;
    }
    virtual unsigned char *pixels()
    {
        return 0;
    }

private:
    VrmlSFFloat d_alpha;
    VrmlSFColor d_color;
    VrmlMFString d_function;
    VrmlMFString d_mode;
    VrmlMFString d_source;
    VrmlMFNode d_texture;

    Viewer::TextureObject d_texObject;
    VrmlNode *d_appearance;
};
}
#endif // _VRMLNODEMULTITEXTURE_
