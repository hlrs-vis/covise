/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TEMPLATE_PLUGIN_H
#define _TEMPLATE_PLUGIN_H
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
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include "vrml97/vrml/VrmlNodeTexture.h"
#include "vrml97/vrml/VrmlMFString.h"
#include "vrml97/vrml/VrmlSFBool.h"
#include "vrml97/vrml/VrmlSFInt.h"

#include "vrml97/vrml/Viewer.h"
#include "cover/coTabletUI.h"

class PLUGINEXPORT VrmlNodeTextureSample : public VrmlNodeTexture
{

public:
    // Define the fields of ARSensor nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTextureSample(VrmlScene *scene = 0);
    VrmlNodeTextureSample(const VrmlNodeTextureSample &n);
    virtual ~VrmlNodeTextureSample();

    void render(Viewer *viewer);
    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeTextureSample *toTextureSample() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;
    virtual int nComponents();
    virtual int width();
    virtual int height();
    virtual int nFrames();
    virtual int number()
    {
        return d_sampleNum.get();
    };
    virtual unsigned char *pixels();
    void updateTexture();

    virtual bool getRepeatS() // LarryD Feb18/99
    {
        return d_repeatS.get();
    }
    virtual bool getRepeatT() // LarryD Feb18/99
    {
        return d_repeatT.get();
    }

private:
    // Fields
    VrmlSFInt d_sampleNum;
    VrmlSFInt d_blendMode;
    VrmlSFBool d_repeatS;
    VrmlSFBool d_repeatT;
    VrmlSFBool d_environment;
    VrmlSFInt d_anisotropy;
    VrmlSFInt d_filterMode;
    Viewer::TextureObject d_texObject;
    unsigned char *pix;

    static void addSample(VrmlNodeTextureSample *node);
    static void removeSample(VrmlNodeTextureSample *node);
    static void newButton(int i);
};

class TextureSamplePlugin : public coVRPlugin, public coTUIListener
{
public:
    TextureSamplePlugin();
    virtual ~TextureSamplePlugin();
    bool init();

    // this will be called in PreFrame
    void preFrame();

    coTUITab *textureTab;
    coTUIButton *applyVideo;
    coTUIFloatSlider *rSlider;
    coTUIFloatSlider *gSlider;
    coTUIFloatSlider *bSlider;
    coTUIFloatSlider *aSlider;
    coTUILabel *objectNameLabel;
    osg::Vec4 currentColor;
    osg::ref_ptr<osg::Node> pickedObject;
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

private:
};
#endif
