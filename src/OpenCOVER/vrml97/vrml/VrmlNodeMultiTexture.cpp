/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMultiTexture.cpp

#include "VrmlNodeMultiTexture.h"

#include "VrmlNodeTexture.h"

#include "Image.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Viewer.h"
#include "Doc.h"

#include "VrmlSFNode.h"

#include "VrmlNodeAppearance.h"
#include "VrmlNodeMultiTextureTransform.h"

#include "MathUtils.h"

using std::cerr;
using std::endl;
using namespace vrml;

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeMultiTexture(s);
}

void VrmlNodeMultiTexture::initFields(VrmlNodeMultiTexture *node, VrmlNodeType *t)
{
    VrmlNodeTexture::initFields(node, t);

    initFieldsHelper(node, t,
                     exposedField("alpha", node->d_alpha, [](auto value){
                            cerr << "Sorry: MultiTexture.alpha is not supported yet" << endl;
                     }),
                     exposedField("color", node->d_color),
                     exposedField("function", node->d_function, [](auto value){
                            cerr << "Sorry: MultiTexture.function is not supported yet" << endl;
                     }),
                     exposedField("mode", node->d_mode),
                     exposedField("source", node->d_source, [](auto value){
                            cerr << "Sorry: MultiTexture.source is not supported yet" << endl;
                     }),
                     exposedField("texture", node->d_texture));
}

const char *VrmlNodeMultiTexture::name() { return "MultiTexture"; }

VrmlNodeMultiTexture::VrmlNodeMultiTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene, name())
    , d_alpha(1)
    , d_color(1, 1, 1)
    , d_function()
    , d_mode()
    , d_source()
    , d_texture()
{
    d_appearance = NULL;
}

VrmlNodeMultiTexture::~VrmlNodeMultiTexture()
{
    while (d_texture.size())
    {
        if (d_texture[0])
        {
            d_texture.removeNode(d_texture[0]);
        }
    }
}

void VrmlNodeMultiTexture::render(Viewer *viewer)
{
    viewer->enableLighting(false); // turn lighting off for this object

    int numTextures = 0;
    for (int i = 0; i < d_texture.size(); i++)
    {
        if (d_texture.get(i) && d_texture.get(i)->as<VrmlNodeTexture>())
            numTextures++;
    }

    viewer->setNumTextures(numTextures);

    for (int i = 0; i < d_texture.size(); i++)
    {
        viewer->textureNumber = i;

        VrmlNodeTexture *t = d_texture.get(i) ? d_texture.get(i)->as<VrmlNodeTexture>() : 0;

        if (t)
        {
            // search in a MultiTextureTransform of the parent scenegraph
            bool foundTextureTransform = false;
            if (d_appearance)
            {
                VrmlNodeAppearance *appearance = d_appearance->as<VrmlNodeAppearance>();
                if (strcmp(appearance->nodeType()->getName(), "Appearance") == 0)
                {
                    if (appearance->textureTransform())
                    {
                        if (strcmp(appearance->textureTransform()->nodeType()->getName(),
                                   "MultiTextureTransform") == 0)
                        {
                            foundTextureTransform = true;
                            VrmlNodeMultiTextureTransform *mtexTrans = appearance->textureTransform()->as<VrmlNodeMultiTextureTransform>();
                            mtexTrans->render(viewer, i);
                        }
                        else if (strcmp(appearance->textureTransform()->nodeType()->getName(),
                                        "TextureTransform") == 0)
                        {
                            foundTextureTransform = true;
                            appearance->textureTransform()->render(viewer);
                        }
                    }
                }
            }
            if (!foundTextureTransform)
                viewer->setTextureTransform(0, 0, 0, 0);

            if (i >= d_mode.size())
                t->setBlendMode(viewer->getBlendModeForVrmlNode("MODULATE"));
            else
            {
                if (strcmp(d_mode[i], "SELECTARG2") == 0)
                    viewer->setColor(d_color.r(), d_color.g(), d_color.b());
                t->setBlendMode(viewer->getBlendModeForVrmlNode(d_mode[i]));
            }

            t->render(viewer);
        }
    }

    clearModified();
}

void VrmlNodeMultiTexture::cloneChildren(VrmlNamespace *ns)
{
    // Replace references with clones
    int n = d_texture.size();
    VrmlNode **kids = d_texture.get();
    for (int i = 0; i < n; ++i)
    {
        if (!kids[i])
            continue;
        VrmlNode *newKid = kids[i]->clone(ns)->reference();
        kids[i]->dereference();
        kids[i] = newKid;
        kids[i]->parentList.push_back(this);
    }
}

bool VrmlNodeMultiTexture::isModified() const
{

    int n = d_texture.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_texture[i] == NULL)
            continue;
        if (d_texture[i]->isModified())
            return true;
    }
    return (d_modified);
}

void VrmlNodeMultiTexture::clearFlags()
{
    VrmlNode::clearFlags();
    int n = d_texture.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_texture[i] == NULL)
            continue;
        d_texture[i]->clearFlags();
    }
}

void VrmlNodeMultiTexture::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    int n = d_texture.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_texture[i] == NULL)
            continue;
        d_texture[i]->addToScene(s, rel);
    }
    nodeStack.pop_front();
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeMultiTexture::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns); // Copy my routes

    // Copy subnode routes
    int n = d_texture.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_texture[i] == NULL)
            continue;
        d_texture[i]->copyRoutes(ns);
    }
    nodeStack.pop_front();
}
