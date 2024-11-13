/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCubeTexture.cpp

#include "VrmlNodeCubeTexture.h"

#include "Image.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Doc.h"

using std::cerr;
using std::endl;
using namespace vrml;

const VrmlMFString &VrmlNodeCubeTexture::getUrl() const
{
    return d_urlXP;
}

// Define the built in VrmlNodeType:: "CubeTexture" fields

void VrmlNodeCubeTexture::initFields(VrmlNodeCubeTexture *node, VrmlNodeType *t)
{
    VrmlNodeTexture::initFields(node, t);
    initFieldsHelper(node, t,
                        exposedField("urlXP", node->d_urlXP, [node](auto value){
                            delete node->d_imageXP;
                            node->d_imageXP = nullptr;
                        }),
                        exposedField("urlXN", node->d_urlXN, [node](auto value){
                            delete node->d_imageXN;
                            node->d_imageXN = nullptr;
                        }),
                        exposedField("urlYP", node->d_urlYP, [node](auto value){
                            delete node->d_imageYP;
                            node->d_imageYP = nullptr;
                        }),
                        exposedField("urlYN", node->d_urlYN, [node](auto value){
                            delete node->d_imageYN;
                            node->d_imageYN = nullptr;
                        }),
                        exposedField("urlZP", node->d_urlZP, [node](auto value){
                            delete node->d_imageYN;
                            node->d_imageYN = nullptr;
                        }),
                        exposedField("urlZN", node->d_urlZN, [node](auto value){
                            delete node->d_imageYN;
                            node->d_imageYN = nullptr;
                        }),
                        field("repeatS", node->d_repeatS),
                        field("repeatT", node->d_repeatT),
                        field("blendMode", node->d_blendMode));

}

const char *VrmlNodeCubeTexture::name() { return "CubeTexture"; }


VrmlNodeCubeTexture::VrmlNodeCubeTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene, name())
    , d_repeatS(true)
    , d_repeatT(true)
    , d_imageXP(0)
    , d_imageXN(0)
    , d_imageYP(0)
    , d_imageYN(0)
    , d_imageZP(0)
    , d_imageZN(0)
    , d_texObject(0)
{
}

VrmlNodeCubeTexture::~VrmlNodeCubeTexture()
{
    delete d_imageXP;
    delete d_imageXN;
    delete d_imageYP;
    delete d_imageYN;
    delete d_imageZP;
    delete d_imageZN;
    // delete d_texObject...
}

void VrmlNodeCubeTexture::readCubeTexture(Image* &image,VrmlMFString& texture, const std::string &textureName)
{
    auto relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
    Doc relDoc(relUrl);
    image = new Image;
    if (!image->tryURLs(texture.size(), texture.get(), &relDoc))
        cerr << "Error: couldn't read CubeTexture from " << textureName << " " << texture << endl;
}

void VrmlNodeCubeTexture::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_imageXP)
        {
            delete d_imageXP; // URL is the only modifiable bit
            d_imageXP = 0;
        }
        if (d_imageXN)
        {
            delete d_imageXN; // URL is the only modifiable bit
            d_imageXN = 0;
        }
        if (d_imageYP)
        {
            delete d_imageYP; // URL is the only modifiable bit
            d_imageYP = 0;
        }
        if (d_imageYN)
        {
            delete d_imageYN; // URL is the only modifiable bit
            d_imageYN = 0;
        }
        if (d_imageZP)
        {
            delete d_imageZP; // URL is the only modifiable bit
            d_imageZP = 0;
        }
        if (d_imageZN)
        {
            delete d_imageZN; // URL is the only modifiable bit
            d_imageZN = 0;
        }
        if (d_texObject)
        {
            viewer->removeCubeTextureObject(d_texObject);
            d_texObject = 0;
        }
    }

    // should probably read the image during addToScene...
    // should cache on url so multiple references to the same file are
    // loaded just once... of course world authors should just DEF/USE
    // them...
    if (!d_imageXP && d_urlXP.size() > 0)
    {
        readCubeTexture(d_imageXP,d_urlXP, "urlXP");
    }
    if (!d_imageXN && d_urlXN.size() > 0)
    {
       readCubeTexture(d_imageXN,d_urlXN, "urlXN");
    }
    if (!d_imageYP && d_urlYP.size() > 0)
    {
       readCubeTexture(d_imageYP,d_urlYP, "urlYP");
    }
    if (!d_imageYN && d_urlYN.size() > 0)
    {
       readCubeTexture(d_imageYN,d_urlYN, "urlYN");
    }
    if (!d_imageZP && d_urlZP.size() > 0)
    {
       readCubeTexture(d_imageZP,d_urlZP, "urlZP");
    }
    if (!d_imageZN && d_urlZN.size() > 0)
    {
       readCubeTexture(d_imageZN,d_urlZN, "urlZN");
    }

    // Check texture cache
    if (d_texObject && d_imageXP)
    {
        viewer->insertCubeTextureReference(d_texObject, d_imageXP->nc(), getBlendMode());
    }
    else
    {

        if ((d_imageXP && d_imageXP->pixels()) && (d_imageXN && d_imageXN->pixels()) && (d_imageYP && d_imageYP->pixels()) && (d_imageYN && d_imageYN->pixels()) && (d_imageZP && d_imageZP->pixels()) && (d_imageZN && d_imageZN->pixels()))
        {
            d_texObject = viewer->insertCubeTexture(d_imageXP->w(),
                                                    d_imageXP->h(),
                                                    d_imageXP->nc(),
                                                    d_repeatS.get(),
                                                    d_repeatT.get(),
                                                    d_imageXP->pixels(),
                                                    d_imageXN->pixels(),
                                                    d_imageYP->pixels(),
                                                    d_imageYN->pixels(),
                                                    d_imageZP->pixels(),
                                                    d_imageZN->pixels(),
                                                    true, getBlendMode());
        }
    }

    clearModified();
}

int VrmlNodeCubeTexture::nComponents()
{
    return d_imageXP ? d_imageXP->nc() : 0;
}

int VrmlNodeCubeTexture::width()
{
    return d_imageXP ? d_imageXP->w() : 0;
}

int VrmlNodeCubeTexture::height()
{
    return d_imageXP ? d_imageXP->h() : 0;
}

int VrmlNodeCubeTexture::nFrames()
{
    return 0;
}

unsigned char *VrmlNodeCubeTexture::pixels()
{
    return d_imageXP ? d_imageXP->pixels() : 0;
}
