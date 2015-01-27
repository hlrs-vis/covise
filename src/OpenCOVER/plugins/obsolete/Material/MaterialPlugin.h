/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TEMPLATE_PLUGIN_H
#define _TEMPLATE_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2005 HLRS  **
 **                                                                          **
 ** Description: Material Plugin											 **
 ** to change the diffuse, ambient and specular color of a picked object     **
 **                                                                          **
 ** Author: A.Brestrich		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Jan-05  v1	    				       		                            **
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

#include <util/coTabletUIMessages.h>

class TexVisitor : public osg::NodeVisitor
{
public:
    TexVisitor(TraversalMode tm, coTUITextureTab *TextureTab);
    void apply(osg::Node &node);
    void clearImageList()
    {
        imageList.clear();
    };

protected:
    coTUITextureTab *texTab;
    std::vector<osg::Image *> imageList;
};

class MaterialPlugin : public coVRPlugin, public coTUIListener, public coSelectionListener
{
public:
    MaterialPlugin();
    virtual ~MaterialPlugin();
    bool init();
    virtual bool selectionChanged();
    virtual bool pickedObjChanged();

    void addNode(osg::Node *node, RenderObject *obj);

    //_____________________________the plugin tab__________________________________________________________
    coTUITabFolder *tabFolder;
    coTUITab *textureTab;

    //_____________________________the color tabs__________________________________________________________
    coTUIColorTab *ambientTab;
    coTUIColorTab *diffuseTab;
    coTUIColorTab *specularTab;
    coTUIColorTab *emissiveTab;
    //_____________________________the texture tab________________________________________________________
    coTUITextureTab *texTab;

    coTUILabel *objectNameLabel;
    //coTUIToggleButton *linkButton;

    //_____________________________colors__________________________________________________________
    osg::Vec4 currentDiffuseColor;
    osg::Vec4 currentSpecularColor;
    osg::Vec4 currentAmbientColor;
    osg::Vec4 currentEmissiveColor;

    TexVisitor *vis;

    bool linked;

    osg::ref_ptr<osg::Node> pickedObject;

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

private:
};
#endif
