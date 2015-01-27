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

VrmlNodeMultiTexture *VrmlNodeMultiTexture::toMultiTexture() const
{
    return (VrmlNodeMultiTexture *)this;
}

// Define the built in VrmlNodeType:: "MultiTexture" fields

VrmlNodeType *VrmlNodeMultiTexture::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MultiTexture", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("alpha", VrmlField::SFFLOAT);
    t->addExposedField("color", VrmlField::SFCOLOR);
    t->addExposedField("function", VrmlField::MFSTRING);
    t->addExposedField("mode", VrmlField::MFSTRING);
    t->addExposedField("source", VrmlField::MFSTRING);
    t->addExposedField("texture", VrmlField::MFNODE);

    return t;
}

VrmlNodeMultiTexture::VrmlNodeMultiTexture(VrmlScene *scene)
    : VrmlNodeTexture(scene)
    , d_alpha(1)
    , d_color(1, 1, 1)
    , d_function()
    , d_mode()
    , d_source()
    , d_texture()
{
    d_appearance = NULL;
}

VrmlNodeType *VrmlNodeMultiTexture::nodeType() const { return defineType(0); }

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

VrmlNode *VrmlNodeMultiTexture::cloneMe() const
{
    return new VrmlNodeMultiTexture(*this);
}

std::ostream &VrmlNodeMultiTexture::printFields(std::ostream &os, int indent)
{
    if (!FPEQUAL(d_alpha.get(), 1))
        PRINT_FIELD(alpha);
    if (!FPEQUAL(d_color.r(), 1) || !FPEQUAL(d_color.g(), 1) || !FPEQUAL(d_color.b(), 1))
        PRINT_FIELD(color);
    if (d_function.get())
        PRINT_FIELD(function);
    if (d_mode.get())
        PRINT_FIELD(mode);
    if (d_source.get())
        PRINT_FIELD(source);
    if (d_texture.size() > 0)
        PRINT_FIELD(texture);
    return os;
}

void VrmlNodeMultiTexture::render(Viewer *viewer)
{
    viewer->enableLighting(false); // turn lighting off for this object

    int numTextures = 0;
    for (int i = 0; i < d_texture.size(); i++)
    {
        if (d_texture.get(i) && d_texture.get(i)->toTexture())
            numTextures++;
    }

    viewer->setNumTextures(numTextures);

    for (int i = 0; i < d_texture.size(); i++)
    {
        viewer->textureNumber = i;

        VrmlNodeTexture *t = d_texture.get(i) ? d_texture.get(i)->toTexture() : 0;

        if (t)
        {
            // search in a MultiTextureTransform of the parent scenegraph
            bool foundTextureTransform = false;
            if (d_appearance)
            {
                VrmlNodeAppearance *appearance = d_appearance->toAppearance();
                if (strcmp(appearance->nodeType()->getName(), "Appearance") == 0)
                {
                    if (appearance->textureTransform())
                    {
                        if (strcmp(appearance->textureTransform()->nodeType()->getName(),
                                   "MultiTextureTransform") == 0)
                        {
                            foundTextureTransform = true;
                            VrmlNodeMultiTextureTransform *mtexTrans = appearance->textureTransform()->toMultiTextureTransform();
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

// Set the value of one of the node fields.

void VrmlNodeMultiTexture::setField(const char *fieldName,
                                    const VrmlField &fieldValue)
{
    if
        TRY_FIELD(alpha, SFFloat)
    else if
        TRY_FIELD(color, SFColor)
    else if
        TRY_FIELD(function, MFString)
    else if
        TRY_FIELD(mode, MFString)
    else if
        TRY_FIELD(source, MFString)
    else if
        TRY_FIELD(texture, MFNode)
    else
        VrmlNode::setField(fieldName, fieldValue);

    if ((strcmp(fieldName, "alpha") == 0) || (strcmp(fieldName, "function") == 0) || (strcmp(fieldName, "source") == 0))
        cerr << "Sorry: MultiTexture." << fieldName << " is not supported yet" << endl;
}

const VrmlField *VrmlNodeMultiTexture::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "alpha") == 0)
        return &d_alpha;
    else if (strcmp(fieldName, "color") == 0)
        return &d_color;
    else if (strcmp(fieldName, "function") == 0)
        return &d_function;
    else if (strcmp(fieldName, "mode") == 0)
        return &d_mode;
    else if (strcmp(fieldName, "source") == 0)
        return &d_source;
    else if (strcmp(fieldName, "texture") == 0)
        return &d_texture;

    return VrmlNode::getField(fieldName);
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
