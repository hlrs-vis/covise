/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTexture.h

#ifndef _VRMLNODETEXTURE_
#define _VRMLNODETEXTURE_

#include "VrmlNode.h"
#include "VrmlSFString.h"
#include "VrmlSFInt.h"
#include "VrmlNodeTemplate.h"

namespace vrml
{

class VRMLEXPORT VrmlNodePixelTexture;
class VRMLEXPORT VrmlNodeImageTexture;
class VRMLEXPORT VrmlNodeCubeTexture;
class VRMLEXPORT VrmlNodeMultiTexture;

class VRMLEXPORT VrmlNodeTexture : public VrmlNodeTemplate
{

public:
    
    static void initFields(VrmlNodeTexture *node, VrmlNodeType *t);    
    // Define the fields of Texture nodes
    static VrmlNodeType *defineType(VrmlNodeType *t);

    static void enableTextureNPOT(bool flag);
    static bool useTextureNPOT();
    static void setMaxTextureSize(int size);
    static int maxTextureSize();

    VrmlNodeTexture(VrmlScene *s, const std::string &name);

    virtual VrmlNodeTexture *toTexture() const;

    virtual int nComponents() = 0;
    virtual int width() = 0;
    virtual int height() = 0;
    virtual int nFrames() = 0;
    virtual unsigned char *pixels() = 0;

    //similar to the VrmlNode calls, but placed here, cause they make more sense here.
    virtual VrmlNodePixelTexture *toPixelTexture() const
    {
        return NULL;
    }
    virtual VrmlNodeImageTexture *toImageTexture() const
    {
        return NULL;
    }
    virtual VrmlNodeCubeTexture *toCubeTexture() const
    {
        return NULL;
    }
    virtual VrmlNodeMultiTexture *toMultiTexture() const
    {
        return NULL;
    }

    void addToScene(VrmlScene *scene, const char *relativeUrl)
    {
        d_scene = scene;
        if (d_relativeUrl.get() == NULL)
        {
            d_relativeUrl.set(relativeUrl);
        }
    }

    /* 
        MultiTexture need to overwrite the texture parameters blendMode
        and filterMode (and maybe anisotropy in the future).
        Unfortunalty, the values can not be overwritten directly, otherwise
        USEd texturenodes (like in the following example) would have
        a texture parameter changed by the usage inside a MultiTexture node
        and could be written wrong to a file later like in the following
        example:

        texture MultiTexture {
          mode [ "ADD", "REPLACE" ] # blendModes > 0
          texture [
            DEF REUSED_NODE ImageTexture {
              url "image1.jpg"
              blendMode 0
            }
            ImageTexture {
              url "image2.jpg"
            }
          ]
        }
        ...
        ...
            USE REUSED_NODE

        A file would be written with blendMode > 0 in REUSED_NODE

        Therefore each texture parameter field is accompanied by a
        second varible holding a second value for usage of the MultiTexture
        node
      */

    void setBlendMode(int mode)
    {
        d_blendModeOverwrite = mode;
    }

    int getBlendMode()
    {
        if (d_blendModeOverwrite == -1)
            return d_blendMode.get();
        return d_blendModeOverwrite;
    }

    virtual void setAppearance(VrmlNode *)
    {
    }

protected:
    VrmlSFInt d_blendMode;
    int d_blendModeOverwrite;

    VrmlSFString d_relativeUrl;

private:
    static bool s_useTextureNPOT;
    static int s_maxTextureSize;
};
}
#endif // _VRMLNODETEXTURE_
