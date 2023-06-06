/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "TextureSample.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/ARToolKit.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <osg/StateAttribute>
#include <osg/Image>
#include <osg/Material>
#include <osg/Geode>
#include <osg/Texture2D>

TextureSamplePlugin *plugin = NULL;

class buttonInfo : public coTUIListener
{
public:
    buttonInfo() { sampleNum = 0; };
    virtual ~buttonInfo() {}
    int sampleNum;
    coTUIButton *button;
    virtual void tabletPressEvent(coTUIElement *tUIItem);

private:
};

void TextureSamplePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (pickedObject.get())
    {
        if (tUIItem == applyVideo)
        {
            osg::Geode *geode = dynamic_cast<osg::Geode *>(pickedObject.get());
            if (geode)
            {
                osg::Drawable *drawable = geode->getDrawable(0);
                if (drawable)
                {
                    osg::StateSet *ss = drawable->getOrCreateStateSet();

                    osg::StateAttribute *stateAttrib = ss->getTextureAttribute(0, osg::StateAttribute::TEXTURE);
                    osg::Texture2D *tex0 = dynamic_cast<osg::Texture2D *>(stateAttrib);
                    if (tex0)
                    {
                        int h = 0, w = 0;
                        int i, j;
                        osg::Image *image = tex0->getImage();

                        if (ARToolKit::art->videoWidth > 512)
                            w = 512;
                        else if (ARToolKit::art->videoWidth > 256)
                            w = 256;
                        else if (ARToolKit::art->videoWidth > 128)
                            w = 128;
                        else if (ARToolKit::art->videoWidth > 64)
                            w = 64;
                        else if (ARToolKit::art->videoWidth > 32)
                            w = 32;
                        if (ARToolKit::art->videoHeight > 512)
                            h = 512;
                        else if (ARToolKit::art->videoHeight > 256)
                            h = 256;
                        else if (ARToolKit::art->videoHeight > 128)
                            h = 128;
                        else if (ARToolKit::art->videoHeight > 64)
                            h = 64;
                        else if (ARToolKit::art->videoHeight > 32)
                            h = 32;
                        unsigned char *imageData = new unsigned char[w * h * 3];
                        for (i = 0; i < h; i++)
                            for (j = 0; j < w; j++)
                            {
#ifdef WIN32
                                imageData[(i * w + j) * 3] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 2];
                                imageData[(i * w + j) * 3 + 1] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 1];
                                imageData[(i * w + j) * 3 + 2] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3];
#else
                                imageData[(i * w + j) * 3] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3];
                                imageData[(i * w + j) * 3 + 1] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 1];
                                imageData[(i * w + j) * 3 + 2] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 2];
#endif
                            }

                        GLint internalFormat = GL_RGB8;
                        GLint format = GL_RGB;
                        image->setImage(w, h, 1, internalFormat, format, GL_UNSIGNED_BYTE, imageData, osg::Image::USE_NEW_DELETE);
                    }
                }
            }
        }
    }
}

void TextureSamplePlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == rSlider)
    {
        currentColor[0] = rSlider->getValue();
    }
    if (tUIItem == gSlider)
    {
        currentColor[1] = gSlider->getValue();
    }
    if (tUIItem == bSlider)
    {
        currentColor[2] = bSlider->getValue();
    }
    if (tUIItem == aSlider)
    {
        currentColor[3] = aSlider->getValue();
    }
    if (pickedObject.get())
    {
        osg::Geode *geode = dynamic_cast<osg::Geode *>(pickedObject.get());
        if (geode)
        {
            osg::Drawable *drawable = geode->getDrawable(0);
            if (drawable)
            {
                osg::StateSet *ss = drawable->getOrCreateStateSet();
                osg::StateAttribute *stateAttrib = ss->getAttribute(osg::StateAttribute::MATERIAL);
                osg::Material *mat = dynamic_cast<osg::Material *>(stateAttrib);
                if (mat)
                {
                    mat->setDiffuse(osg::Material::FRONT_AND_BACK, currentColor);
                    if (currentColor[3] < 1.0)
                        ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
                }
            }
        }
    }
}

TextureSamplePlugin::TextureSamplePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool TextureSamplePlugin::init()
{
    if (plugin)
        return false;

    plugin = this;

    textureTab = new coTUITab("TextureSamples", coVRTui::tui->mainFolder->getID());
    VrmlNamespace::addBuiltIn(VrmlNodeTextureSample::defineType());
    applyVideo = new coTUIButton("ApplyVideo", textureTab->getID());
    rSlider = new coTUIFloatSlider("rSlider", textureTab->getID());
    gSlider = new coTUIFloatSlider("gSlider", textureTab->getID());
    bSlider = new coTUIFloatSlider("bSlider", textureTab->getID());
    aSlider = new coTUIFloatSlider("aSlider", textureTab->getID());
    objectNameLabel = new coTUILabel("NoName", textureTab->getID());
    objectNameLabel->setPos(1, 6);

    textureTab->setPos(0, 0);
    applyVideo->setPos(1, 0);
    applyVideo->setEventListener(this);
    rSlider->setRange(0, 1);
    rSlider->setValue(1);
    gSlider->setRange(0, 1);
    gSlider->setValue(1);
    bSlider->setRange(0, 1);
    bSlider->setValue(1);
    aSlider->setRange(0, 1);
    aSlider->setValue(1);

    rSlider->setPos(1, 1);
    rSlider->setEventListener(this);
    gSlider->setPos(1, 2);
    gSlider->setEventListener(this);
    bSlider->setPos(1, 3);
    bSlider->setEventListener(this);
    aSlider->setPos(1, 4);
    aSlider->setEventListener(this);

    currentColor.set(1, 1, 1, 1);

    return true;
}

// this is called if the plugin is removed at runtime
TextureSamplePlugin::~TextureSamplePlugin()
{
    delete textureTab;
}

void
TextureSamplePlugin::preFrame()
{
    if (cover->getIntersectedNode())
    {
        if (cover->getPointerButton()->wasPressed())
        {
            pickedObject = cover->getIntersectedNode();
            objectNameLabel->setLabel(pickedObject->getName().c_str());
        }
    }
}

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeTextureSample.cpp

#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlNodeTexture.h>
#include <vrml97/vrml/VrmlScene.h>

static list<VrmlNodeTextureSample *> textureSamples;
static list<buttonInfo *> ApplyButtons;

// ARSensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeTextureSample(scene);
}

// Define the built in VrmlNodeType:: "ARSensor" fields

VrmlNodeType *VrmlNodeTextureSample::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("TextureSample", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("sampleNum", VrmlField::SFINT32);
    t->addField("repeatS", VrmlField::SFBOOL);
    t->addField("repeatT", VrmlField::SFBOOL);
    t->addField("environment", VrmlField::SFBOOL);
    t->addField("blendMode", VrmlField::SFINT32);
    t->addField("filterMode", VrmlField::SFINT32);
    t->addField("anisotropy", VrmlField::SFINT32);

    return t;
}

VrmlNodeType *VrmlNodeTextureSample::nodeType() const
{
    return defineType(0);
}

VrmlNodeTextureSample::VrmlNodeTextureSample(VrmlScene *scene)
    : VrmlNodeTexture(scene)
    , d_sampleNum(0)
    , d_blendMode(0)
    , d_repeatS(true)
    , d_repeatT(true)
    , d_environment(false)
    , d_anisotropy(1)
    , d_filterMode(0)
    , d_texObject(0)
{
    pix = new unsigned char[width() * height() * 3];
    addSample(this);
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeTextureSample::VrmlNodeTextureSample(const VrmlNodeTextureSample &n)
    : VrmlNodeTexture(n.d_scene)
    , d_sampleNum(n.d_sampleNum)
    , d_blendMode(n.d_blendMode)
    , d_repeatS(n.d_repeatS)
    , d_repeatT(n.d_repeatT)
    , d_environment(n.d_environment)
    , d_anisotropy(n.d_anisotropy)
    , d_filterMode(n.d_filterMode)
    , d_texObject(n.d_texObject)
{
    pix = new unsigned char[width() * height() * 3];
    addSample(this);
}

VrmlNodeTextureSample::~VrmlNodeTextureSample()
{
    delete[] pix;
    removeSample(this);
}

VrmlNode *VrmlNodeTextureSample::cloneMe() const
{
    return new VrmlNodeTextureSample(*this);
}

VrmlNodeTextureSample *VrmlNodeTextureSample::toTextureSample() const
{
    return (VrmlNodeTextureSample *)this;
}

void VrmlNodeTextureSample::render(Viewer *viewer)
{

    // Check texture cache
    if (d_texObject)
    {
        viewer->insertTextureReference(d_texObject, 3, d_environment.get(), d_blendMode.get());
    }
    else
    {
        if (width())
        {

            d_texObject = viewer->insertTexture(width(),
                                                height(),
                                                3,
                                                d_repeatS.get(),
                                                d_repeatT.get(),
                                                pix,
                                                "",
                                                true, d_environment.get(), d_blendMode.get(), d_anisotropy.get(), d_filterMode.get());
        }
    }

    clearModified();
}

int VrmlNodeTextureSample::nComponents()
{
    return 3;
}

int VrmlNodeTextureSample::width()
{
    if (ARToolKit::art->videoWidth > 512)
        return 512;
    if (ARToolKit::art->videoWidth > 256)
        return 256;
    if (ARToolKit::art->videoWidth > 128)
        return 128;
    if (ARToolKit::art->videoWidth > 64)
        return 64;
    if (ARToolKit::art->videoWidth > 32)
        return 32;
    return 0;
}

int VrmlNodeTextureSample::height()
{
    if (ARToolKit::art->videoHeight > 512)
        return 512;
    if (ARToolKit::art->videoHeight > 256)
        return 256;
    if (ARToolKit::art->videoHeight > 128)
        return 128;
    if (ARToolKit::art->videoHeight > 64)
        return 64;
    if (ARToolKit::art->videoHeight > 32)
        return 32;
    return 0;
}

int VrmlNodeTextureSample::nFrames()
{
    return 0;
}

unsigned char *VrmlNodeTextureSample::pixels()
{
    return pix;
}

ostream &VrmlNodeTextureSample::printFields(ostream &os, int indent)
{
    if (!d_sampleNum.get())
        PRINT_FIELD(sampleNum);
    if (!d_repeatS.get())
        PRINT_FIELD(repeatS);
    if (!d_repeatT.get())
        PRINT_FIELD(repeatT);
    if (d_environment.get())
        PRINT_FIELD(environment);
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeTextureSample::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_FIELD(sampleNum, SFInt)
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
        VrmlNodeTexture::setField(fieldName, fieldValue);
    newButton(d_sampleNum.get());
}

const VrmlField *VrmlNodeTextureSample::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "sampleNum") == 0)
        return &d_sampleNum;
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
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void buttonInfo::tabletPressEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
    for (list<VrmlNodeTextureSample *>::iterator it = textureSamples.begin(); it != textureSamples.end(); it++)
    {
        if ((*it)->number() == sampleNum)
        {
            int h, w;
            int i, j;
            unsigned char *pix = (*it)->pixels();
            h = (*it)->height();
            w = (*it)->width();
            for (i = 0; i < h; i++)
                for (j = 0; j < w; j++)
                {
#ifdef WIN32
                    pix[(i * w + j) * 3] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 2];
                    pix[(i * w + j) * 3 + 1] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 1];
                    pix[(i * w + j) * 3 + 2] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3];
#else
                    pix[(i * w + j) * 3] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3];
                    pix[(i * w + j) * 3 + 1] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 1];
                    pix[(i * w + j) * 3 + 2] = ARToolKit::art->videoData[(i * ARToolKit::art->videoWidth + j) * 3 + 2];
#endif
                }
            (*it)->updateTexture();
        }
    }
}

void VrmlNodeTextureSample::updateTexture()
{
    d_texObject = 0;
    setModified();
}

void VrmlNodeTextureSample::addSample(VrmlNodeTextureSample *node)
{
    textureSamples.push_front(node);
    newButton(node->d_sampleNum.get());
}

void VrmlNodeTextureSample::newButton(int n)
{
    int i = 0;
    for (list<buttonInfo *>::iterator it = ApplyButtons.begin(); it != ApplyButtons.end(); it++)
    {
        if ((*it)->sampleNum == n)
        {
            return;
        }
        i++;
    }
    char name[500];
    sprintf(name, "Texture %d", n);
    coTUIButton *button = new coTUIButton(name, plugin->textureTab->getID());
    buttonInfo *bi = new buttonInfo();
    bi->button = button;
    bi->sampleNum = n;
    button->setPos(0, n);
    button->setEventListener(bi);
    ApplyButtons.push_front(bi);
}

void VrmlNodeTextureSample::removeSample(VrmlNodeTextureSample *node)
{
    textureSamples.remove(node);
}

COVERPLUGIN(TextureSamplePlugin)
