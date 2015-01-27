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

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeCubeTexture(s);
}

const VrmlMFString &VrmlNodeCubeTexture::getUrl() const
{
    return d_urlXP;
}

VrmlNodeCubeTexture *VrmlNodeCubeTexture::toCubeTexture() const
{
    return (VrmlNodeCubeTexture *)this;
}

// Define the built in VrmlNodeType:: "CubeTexture" fields

VrmlNodeType *VrmlNodeCubeTexture::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("CubeTexture", creator);
    }

    VrmlNodeTexture::defineType(t); // Parent class

    t->addExposedField("urlXP", VrmlField::MFSTRING);
    t->addExposedField("urlXN", VrmlField::MFSTRING);
    t->addExposedField("urlYP", VrmlField::MFSTRING);
    t->addExposedField("urlYN", VrmlField::MFSTRING);
    t->addExposedField("urlZP", VrmlField::MFSTRING);
    t->addExposedField("urlZN", VrmlField::MFSTRING);

    t->addField("repeatS", VrmlField::SFBOOL);
    t->addField("repeatT", VrmlField::SFBOOL);
    t->addField("blendMode", VrmlField::SFINT32);

    return t;
}

VrmlNodeCubeTexture::VrmlNodeCubeTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene)
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

VrmlNodeType *VrmlNodeCubeTexture::nodeType() const { return defineType(0); }

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

VrmlNode *VrmlNodeCubeTexture::cloneMe() const
{
    return new VrmlNodeCubeTexture(*this);
}

std::ostream &VrmlNodeCubeTexture::printFields(std::ostream &os, int indent)
{
    if (d_urlXP.get())
        PRINT_FIELD(urlXP);
    if (d_urlXN.get())
        PRINT_FIELD(urlXN);
    if (d_urlYP.get())
        PRINT_FIELD(urlYP);
    if (d_urlYN.get())
        PRINT_FIELD(urlYN);
    if (d_urlZP.get())
        PRINT_FIELD(urlZP);
    if (d_urlZN.get())
        PRINT_FIELD(urlZN);
    if (!d_repeatS.get())
        PRINT_FIELD(repeatS);
    if (!d_repeatT.get())
        PRINT_FIELD(repeatT);
    return os;
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
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_imageXP = new Image;
        if (!d_imageXP->tryURLs(d_urlXP.size(), d_urlXP.get(), &relDoc))
            cerr << "Error: couldn't read CubeTexture from URLXP " << d_urlXP << endl;
    }
    if (!d_imageXN && d_urlXN.size() > 0)
    {
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_imageXN = new Image;
        if (!d_imageXN->tryURLs(d_urlXN.size(), d_urlXN.get(), &relDoc))
            cerr << "Error: couldn't read CubeTexture from URLXN " << d_urlXN << endl;
    }
    if (!d_imageYP && d_urlYP.size() > 0)
    {
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_imageYP = new Image;
        if (!d_imageYP->tryURLs(d_urlYP.size(), d_urlYP.get(), &relDoc))
            cerr << "Error: couldn't read CubeTexture from URLYP " << d_urlYP << endl;
    }
    if (!d_imageYN && d_urlYN.size() > 0)
    {
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_imageYN = new Image;
        if (!d_imageYN->tryURLs(d_urlYN.size(), d_urlYN.get(), &relDoc))
            cerr << "Error: couldn't read CubeTexture from URLYN " << d_urlYN << endl;
    }
    if (!d_imageZP && d_urlZP.size() > 0)
    {
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_imageZP = new Image;
        if (!d_imageZP->tryURLs(d_urlZP.size(), d_urlZP.get(), &relDoc))
            cerr << "Error: couldn't read CubeTexture from URLZP " << d_urlZP << endl;
    }
    if (!d_imageZN && d_urlZN.size() > 0)
    {
        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        Doc relDoc(relUrl);
        d_imageZN = new Image;
        if (!d_imageZN->tryURLs(d_urlZN.size(), d_urlZN.get(), &relDoc))
            cerr << "Error: couldn't read CubeTexture from URLZN " << d_urlZN << endl;
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

// Set the value of one of the node fields.

void VrmlNodeCubeTexture::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    if (strcmp(fieldName, "urlXP") == 0)
    {
        delete d_imageXP;
        d_imageXP = 0;
    }
    else if (strcmp(fieldName, "urlXN") == 0)
    {
        delete d_imageXN;
        d_imageXN = 0;
    }
    else if (strcmp(fieldName, "urlYP") == 0)
    {
        delete d_imageYP;
        d_imageYP = 0;
    }
    else if (strcmp(fieldName, "urlYN") == 0)
    {
        delete d_imageYN;
        d_imageYN = 0;
    }
    else if (strcmp(fieldName, "urlZP") == 0)
    {
        delete d_imageZP;
        d_imageZP = 0;
    }
    else if (strcmp(fieldName, "urlZN") == 0)
    {
        delete d_imageZN;
        d_imageZN = 0;
    }

    if
        TRY_FIELD(urlXP, MFString)
    else if
        TRY_FIELD(urlXN, MFString)
    else if
        TRY_FIELD(urlYP, MFString)
    else if
        TRY_FIELD(urlYN, MFString)
    else if
        TRY_FIELD(urlZP, MFString)
    else if
        TRY_FIELD(urlZN, MFString)
    else if
        TRY_FIELD(repeatS, SFBool)
    else if
        TRY_FIELD(repeatT, SFBool)
    else if
        TRY_FIELD(blendMode, SFInt)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeCubeTexture::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "urlXP") == 0)
        return &d_urlXP;
    else if (strcmp(fieldName, "urlXN") == 0)
        return &d_urlXN;
    else if (strcmp(fieldName, "urlYP") == 0)
        return &d_urlYP;
    else if (strcmp(fieldName, "urlYN") == 0)
        return &d_urlYN;
    else if (strcmp(fieldName, "urlZP") == 0)
        return &d_urlZP;
    else if (strcmp(fieldName, "urlZN") == 0)
        return &d_urlZN;
    else if (strcmp(fieldName, "repeatS") == 0)
        return &d_repeatS;
    else if (strcmp(fieldName, "repeatT") == 0)
        return &d_repeatT;
    else if (strcmp(fieldName, "blendMode") == 0)
        return &d_blendMode;

    return VrmlNode::getField(fieldName);
}
