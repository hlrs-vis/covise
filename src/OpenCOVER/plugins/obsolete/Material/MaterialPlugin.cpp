/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Material Plugin					                           **
 **                                                                          **
 **                                                                          **
 ** Author: A.Brestrich		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Jan-05  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
 \****************************************************************************/

#include "MaterialPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <osg/StateAttribute>
#include <osg/Image>
#include <osg/Material>
#include <osg/Geode>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/BlendFunc>
#include <osg/Texture2D>

using namespace osg;

class buttonInfo : public coTUIListener
{
public:
    buttonInfo() { sampleNum = 0; };
    virtual ~buttonInfo();
    int sampleNum;
    coTUIButton *button;
    virtual void tabletPressEvent(coTUIElement *tUIItem);

private:
};

MaterialPlugin::MaterialPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool MaterialPlugin::init()
{
    textureTab = new coTUITab("Material", coVRTui::tui->mainFolder->getID());
    textureTab->setPos(0, 0);

    tabFolder = new coTUITabFolder("Folder", textureTab->getID());
    tabFolder->setPos(0, 0);

    objectNameLabel = new coTUILabel("No Object picked", textureTab->getID());
    objectNameLabel->setPos(0, 3);

    ambientTab = new coTUIColorTab("Ambient", tabFolder->getID());
    specularTab = new coTUIColorTab("Specular", tabFolder->getID());
    emissiveTab = new coTUIColorTab("Emissive", tabFolder->getID());
    texTab = new coTUITextureTab("Texture", tabFolder->getID());
    diffuseTab = new coTUIColorTab("Diffuse", tabFolder->getID());

    ambientTab->setPos(1, 0);
    specularTab->setPos(2, 0);
    emissiveTab->setPos(3, 0);
    texTab->setPos(4, 0);
    diffuseTab->setPos(0, 0);

    texTab->setEventListener(this);
    ambientTab->setEventListener(this);
    specularTab->setEventListener(this);
    emissiveTab->setEventListener(this);
    diffuseTab->setEventListener(this);

    vis = new TexVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN, texTab);

    coVRSelectionManager::instance()->addListener(this);

    return true;
}

// this is called if the plugin is removed at runtime
MaterialPlugin::~MaterialPlugin()
{
    delete objectNameLabel;
    delete ambientTab;
    delete specularTab;
    delete emissiveTab;
    delete diffuseTab;
    delete texTab;
    delete tabFolder;
    delete textureTab;
    delete vis;

    coVRSelectionManager::instance()->removeListener(this);
}

bool MaterialPlugin::selectionChanged()
{
    std::list<osg::ref_ptr<osg::Node> > selectedNodeList = coVRSelectionManager::instance()->getSelectionList();
    std::list<osg::ref_ptr<osg::Node> >::iterator nodeIter = selectedNodeList.end();
    if (selectedNodeList.size() == 0)
    {
        pickedObject = NULL;
        objectNameLabel->setLabel("NONE");
    }
    else
    {
        nodeIter--;
        //selectedNodesParent = (*parentIter).get();

        pickedObject = (*nodeIter).get();

        objectNameLabel->setLabel(pickedObject->getName().c_str());
        if (pickedObject.get())
        {
            Geode *geode = dynamic_cast<Geode *>(pickedObject.get());
            if (geode)
            {
                texTab->setCurrentNode(geode);
                Drawable *drawable = geode->getDrawable(0);

                if (drawable)
                {
                    StateSet *ss = drawable->getOrCreateStateSet();
                    StateAttribute *stateAttrib;
                    for (int textureNumber = 0; textureNumber < 21; textureNumber++)
                    {
                        int texGenMode = TEX_GEN_NONE;
                        int texEnvMode = TEX_ENV_MODULATE;
                        StateAttribute *stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXTURE);
                        if (stateAttrib)
                        {
                            stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXGEN);
                            if (stateAttrib)
                            {
                                TexGen *texGen = dynamic_cast<TexGen *>(stateAttrib);
                                if (texGen)
                                {
                                    switch (texGen->getMode())
                                    {
                                    case TexGen::OBJECT_LINEAR:
                                        texGenMode = TEX_GEN_OBJECT_LINEAR;
                                        break;
                                    case TexGen::EYE_LINEAR:
                                        texGenMode = TEX_GEN_EYE_LINEAR;
                                        break;
                                    case TexGen::SPHERE_MAP:
                                        texGenMode = TEX_GEN_SPHERE_MAP;
                                        break;
                                    case TexGen::NORMAL_MAP:
                                        texGenMode = TEX_GEN_NORMAL_MAP;
                                        break;
                                    case TexGen::REFLECTION_MAP:
                                        texGenMode = TEX_GEN_REFLECTION_MAP;
                                        break;
                                    }
                                }
                            }
                            stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXENV);
                            if (stateAttrib)
                            {
                                TexEnv *texEnv = dynamic_cast<TexEnv *>(stateAttrib);
                                if (texEnv)
                                {
                                    switch (texEnv->getMode())
                                    {
                                    case TexEnv::DECAL:
                                        texEnvMode = TEX_ENV_DECAL;
                                        break;
                                    case TexEnv::MODULATE:
                                        texEnvMode = TEX_ENV_MODULATE;
                                        break;
                                    case TexEnv::BLEND:
                                        texEnvMode = TEX_ENV_BLEND;
                                        break;
                                    case TexEnv::REPLACE:
                                        texEnvMode = TEX_ENV_REPLACE;
                                        break;
                                    case TexEnv::ADD:
                                        texEnvMode = TEX_ENV_ADD;
                                        break;
                                    default:
                                        texEnvMode = TEX_ENV_MODULATE;
                                    }
                                }
                            }
                            texTab->setTexture(textureNumber, texEnvMode, texGenMode);
                        }
                    }
                    stateAttrib = ss->getAttribute(StateAttribute::MATERIAL);
                    Material *mat = dynamic_cast<Material *>(stateAttrib);
                    //_________________________________Set the colors__________________________________________________
                    if (mat)
                    {
                        //_____________________________Set the Diffuse color__________________________________________________________
                        currentDiffuseColor = mat->getDiffuse(Material::FRONT);

                        diffuseTab->setColor(currentDiffuseColor[0],
                                             currentDiffuseColor[1],
                                             currentDiffuseColor[2],
                                             currentDiffuseColor[3]);
                        //_____________________________Set the Specular color__________________________________________________________
                        currentSpecularColor = mat->getSpecular(Material::FRONT);

                        specularTab->setColor(currentSpecularColor[0],
                                              currentSpecularColor[1],
                                              currentSpecularColor[2],
                                              currentSpecularColor[3]);
                        //_____________________________Set the Ambient color________________________________________________
                        currentAmbientColor = mat->getAmbient(Material::FRONT);

                        ambientTab->setColor(currentAmbientColor[0],
                                             currentAmbientColor[1],
                                             currentAmbientColor[2],
                                             currentAmbientColor[3]);
                        //_____________________________Set the Emission color________________________________________________
                        currentEmissiveColor = mat->getEmission(Material::FRONT);

                        emissiveTab->setColor(currentEmissiveColor[0],
                                              currentEmissiveColor[1],
                                              currentEmissiveColor[2],
                                              currentEmissiveColor[3]);
                    }
                }
            }
        }
    }
    return true;
}

bool MaterialPlugin::pickedObjChanged()
{
    return true;
}

void MaterialPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    if (tUIItem == texTab)
    {
        vis->apply(*cover->getObjectsRoot());
        vis->clearImageList();
    }
}

void MaterialPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == texTab)
    {
        Geode *geode = (osg::Geode *)texTab->getChangedNode();
        if (geode)
        {
            int intTexFormat = GL_RGBA8;
            int type = GL_UNSIGNED_BYTE;
            int pixelFormat = GL_RGB;

            if (texTab->getDepth() == 24)
            {
                pixelFormat = GL_RGB;
                intTexFormat = GL_RGB8;
            }
            else if (texTab->getDepth() == 32)
            {
                pixelFormat = GL_RGBA;
                intTexFormat = GL_RGBA8;
            }
            else if (texTab->getDepth() == 16)
            {
                pixelFormat = GL_LUMINANCE_ALPHA;
                intTexFormat = GL_RGBA8;
            }
            else if (texTab->getDepth() == 8)
            {
                pixelFormat = GL_LUMINANCE;
                intTexFormat = GL_RGB8;
            }
            Drawable *drawable = geode->getDrawable(0);
            if (drawable)
            {
                StateSet *ss = drawable->getOrCreateStateSet();
                Texture2D *texture;
                Image *image;
                TexGen *texGen = NULL;
                TexEnv *texEnv = NULL;
                int texNumber = texTab->getTextureNumber();
                int texMode = texTab->getTextureMode();
                int texGenMode = texTab->getTextureTexGenMode();
                texture = dynamic_cast<Texture2D *>(ss->getTextureAttribute(texNumber, StateAttribute::TEXTURE));

                StateAttribute *stateAttrib = ss->getTextureAttribute(texNumber, StateAttribute::TEXENV);
                if (stateAttrib)
                {
                    texEnv = dynamic_cast<TexEnv *>(stateAttrib);
                }
                stateAttrib = ss->getTextureAttribute(texNumber, StateAttribute::TEXGEN);
                if (stateAttrib)
                {
                    texGen = dynamic_cast<TexGen *>(stateAttrib);
                }
                if (texEnv == NULL)
                    texEnv = new TexEnv;
                if (texGen == NULL)
                    texGen = new TexGen;
                if (texture)
                {
                    image = texture->getImage();
                    if (!image)
                        image = new Image;
                }
                else
                {
                    texture = new Texture2D;
                    image = new Image;

                    texture->setDataVariance(Object::DYNAMIC);
                    texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
                    texture->setFilter(Texture::MAG_FILTER, Texture::LINEAR);
                    texture->setWrap(Texture::WRAP_S, Texture::REPEAT);
                    texture->setWrap(Texture::WRAP_T, Texture::REPEAT);
                    ss->setTextureAttributeAndModes(texNumber, texture, StateAttribute::ON);
                    texGenMode = TEX_GEN_OBJECT_LINEAR;
                }
                switch (texGenMode)
                {
                case TEX_GEN_OBJECT_LINEAR:
                    texGen->setMode(TexGen::OBJECT_LINEAR);
                    break;
                case TEX_GEN_EYE_LINEAR:
                    texGen->setMode(TexGen::EYE_LINEAR);
                    break;
                case TEX_GEN_SPHERE_MAP:
                    texGen->setMode(TexGen::SPHERE_MAP);
                    break;
                case TEX_GEN_NORMAL_MAP:
                    texGen->setMode(TexGen::NORMAL_MAP);
                    break;
                case TEX_GEN_REFLECTION_MAP:
                    texGen->setMode(TexGen::REFLECTION_MAP);
                    break;
                }
                switch (texMode)
                {
                case TEX_ENV_DECAL:
                    texEnv->setMode(TexEnv::DECAL);
                    break;
                case TEX_ENV_MODULATE:
                    texEnv->setMode(TexEnv::MODULATE);
                    break;
                case TEX_ENV_BLEND:
                    texEnv->setMode(TexEnv::BLEND);
                    break;
                case TEX_ENV_REPLACE:
                    texEnv->setMode(TexEnv::REPLACE);
                    break;
                case TEX_ENV_ADD:
                    texEnv->setMode(TexEnv::ADD);
                    break;
                default:
                    texEnv->setMode(TexEnv::DECAL);
                }
                if (texTab->hasAlpha())
                {
                    ss->setRenderingHint(StateSet::TRANSPARENT_BIN);
                    ss->setMode(GL_BLEND /*StateAttribute::BLENDFUNC*/, StateAttribute::ON);
                    BlendFunc *blendFunc = new BlendFunc();
                    blendFunc->setFunction(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
                    ss->setAttributeAndModes(blendFunc, StateAttribute::ON);
                }
                if (texGenMode != TEX_GEN_NONE)
                    ss->setTextureAttributeAndModes(texNumber, texGen, StateAttribute::ON);
                else
                    ss->setTextureAttributeAndModes(texNumber, texGen, StateAttribute::OFF);
                ss->setTextureAttributeAndModes(texNumber, texEnv, StateAttribute::ON);
                image->setImage(texTab->getWidth(),
                                texTab->getHeight(),
                                texTab->getDepth(),
                                intTexFormat,
                                pixelFormat,
                                type,
                                texTab->getData(),
                                Image::USE_NEW_DELETE);
                texture->setImage(image);
                drawable->setStateSet(ss);
            }
        }
    }
}

void MaterialPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (pickedObject.get())
    {
        if (tUIItem == diffuseTab)
        {
            currentDiffuseColor[0] = diffuseTab->getRed();
            currentDiffuseColor[1] = diffuseTab->getGreen();
            currentDiffuseColor[2] = diffuseTab->getBlue();
            currentDiffuseColor[3] = diffuseTab->getAlpha();
        }
        else if (tUIItem == specularTab)
        {
            currentSpecularColor[0] = specularTab->getRed();
            currentSpecularColor[1] = specularTab->getGreen();
            currentSpecularColor[2] = specularTab->getBlue();
            currentSpecularColor[3] = specularTab->getAlpha();
        }
        else if (tUIItem == ambientTab)
        {
            currentAmbientColor[0] = ambientTab->getRed();
            currentAmbientColor[1] = ambientTab->getGreen();
            currentAmbientColor[2] = ambientTab->getBlue();
            currentAmbientColor[3] = ambientTab->getAlpha();
        }
        else if (tUIItem == emissiveTab)
        {
            currentEmissiveColor[0] = emissiveTab->getRed();
            currentEmissiveColor[1] = emissiveTab->getGreen();
            currentEmissiveColor[2] = emissiveTab->getBlue();
            currentEmissiveColor[3] = emissiveTab->getAlpha();
        }
        Geode *geode = dynamic_cast<Geode *>(pickedObject.get());
        if (geode)
        {
            Drawable *drawable = geode->getDrawable(0);
            if (drawable)
            {
                StateSet *ss = drawable->getOrCreateStateSet();
                StateAttribute *stateAttrib = ss->getAttribute(StateAttribute::MATERIAL);
                Material *mat = dynamic_cast<Material *>(stateAttrib);
                if (mat)
                {
                    mat->setDiffuse(Material::FRONT_AND_BACK, currentDiffuseColor);
                    mat->setSpecular(Material::FRONT_AND_BACK, currentSpecularColor);
                    mat->setAmbient(Material::FRONT_AND_BACK, currentAmbientColor);
                    mat->setEmission(Material::FRONT_AND_BACK, currentEmissiveColor);

                    if (currentDiffuseColor[3] < 1.0)
                        ss->setRenderingHint(StateSet::TRANSPARENT_BIN);
                }
            }
        }
    }
}

TexVisitor::TexVisitor(TraversalMode tm, coTUITextureTab *textureTab)
    : NodeVisitor(tm)
{
    texTab = textureTab;
}

void TexVisitor::apply(Node &node)
{
    Geode *geode = dynamic_cast<Geode *>(&node);
    if (geode)
    {
        Drawable *drawable = geode->getDrawable(0);
        if (drawable)
        {
            StateSet *ss = drawable->getOrCreateStateSet();
            for (int textureNumber = 0; textureNumber < 21; textureNumber++)
            {
                StateAttribute *stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXTURE);
                if (stateAttrib)
                {
                    Texture2D *texture = dynamic_cast<Texture2D *>(stateAttrib);
                    if (texture)
                    {
                        Image *image = texture->getImage();
                        if (image)
                        {
                            if (std::find(imageList.begin(), imageList.end(), image) == imageList.end())
                            {
                                imageList.push_back(image);
                                texTab->setTexture(image->t(),
                                                   image->s(),
                                                   image->getPixelSizeInBits(),
                                                   image->getImageSizeInBytes(),
                                                   reinterpret_cast<char *>(image->data()));
                            }
                        }
                    }
                }
            }
        }
    }
    traverse(node);
    texTab->finishedTraversing();
}

void MaterialPlugin::addNode(Node *node, const RenderObject *obj)
{
    (void)node;
    (void)obj;
    TexVisitor *vis = new TexVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN, texTab);
    vis->apply(*cover->getObjectsRoot());
}

COVERPLUGIN(MaterialPlugin)
