/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeAppearance.cpp
//

#include "VrmlNodeAppearance.h"
#include "VrmlNodeType.h"

#include "Viewer.h"
#include "VrmlNodeMaterial.h"
#include "VrmlNodeTexture.h"
#include "VrmlScene.h"
#include "Doc.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeAppearance(scene);
}

// Define the built in VrmlNodeType:: "Appearance" fields

VrmlNodeType *VrmlNodeAppearance::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Appearance", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("material", VrmlField::SFNODE);
    t->addExposedField("texture", VrmlField::SFNODE);
    t->addExposedField("textureTransform", VrmlField::SFNODE);

    // additional fields for multi-texturing
    t->addExposedField("texture2", VrmlField::SFNODE);
    t->addExposedField("textureTransform2", VrmlField::SFNODE);
    t->addExposedField("texture3", VrmlField::SFNODE);
    t->addExposedField("textureTransform3", VrmlField::SFNODE);
    t->addExposedField("texture4", VrmlField::SFNODE);
    t->addExposedField("textureTransform4", VrmlField::SFNODE);
    t->addExposedField("texture5", VrmlField::SFNODE);
    t->addExposedField("textureTransform5", VrmlField::SFNODE);
    t->addExposedField("texture6", VrmlField::SFNODE);
    t->addExposedField("textureTransform6", VrmlField::SFNODE);
    t->addExposedField("texture7", VrmlField::SFNODE);
    t->addExposedField("textureTransform7", VrmlField::SFNODE);
    t->addExposedField("texture8", VrmlField::SFNODE);
    t->addExposedField("textureTransform8", VrmlField::SFNODE);
    t->addExposedField("texture9", VrmlField::SFNODE);
    t->addExposedField("textureTransform9", VrmlField::SFNODE);
    t->addExposedField("texture10", VrmlField::SFNODE);
    t->addExposedField("textureTransform10", VrmlField::SFNODE);

    return t;
}

VrmlNodeType *VrmlNodeAppearance::nodeType() const { return defineType(0); }

VrmlNodeAppearance::VrmlNodeAppearance(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
}

VrmlNodeAppearance::~VrmlNodeAppearance()
{
}

VrmlNode *VrmlNodeAppearance::cloneMe() const
{
    return new VrmlNodeAppearance(*this);
}

void VrmlNodeAppearance::cloneChildren(VrmlNamespace *ns)
{
    // Replace references with clones
    if (d_material.get())
    {
        d_material.set(d_material.get()->clone(ns));
        d_material.get()->parentList.push_back(this);
    }
    if (d_texture.get())
    {
        d_texture.set(d_texture.get()->clone(ns));
        d_texture.get()->parentList.push_back(this);
    }
    if (d_textureTransform.get())
    {
        d_textureTransform.set(d_textureTransform.get()->clone(ns));
        d_textureTransform.get()->parentList.push_back(this);
    }

    // additional fields for multi-texturing
    if (d_texture2.get())
    {
        d_texture2.set(d_texture2.get()->clone(ns));
        d_texture2.get()->parentList.push_back(this);
    }
    if (d_textureTransform2.get())
    {
        d_textureTransform2.set(d_textureTransform2.get()->clone(ns));
        d_textureTransform2.get()->parentList.push_back(this);
    }
    if (d_texture3.get())
    {
        d_texture3.set(d_texture3.get()->clone(ns));
        d_texture3.get()->parentList.push_back(this);
    }
    if (d_textureTransform3.get())
    {
        d_textureTransform3.set(d_textureTransform3.get()->clone(ns));
        d_textureTransform3.get()->parentList.push_back(this);
    }
    if (d_texture4.get())
    {
        d_texture4.set(d_texture4.get()->clone(ns));
        d_texture4.get()->parentList.push_back(this);
    }
    if (d_textureTransform4.get())
    {
        d_textureTransform4.set(d_textureTransform4.get()->clone(ns));
        d_textureTransform4.get()->parentList.push_back(this);
    }
    if (d_texture5.get())
    {
        d_texture5.set(d_texture5.get()->clone(ns));
        d_texture5.get()->parentList.push_back(this);
    }
    if (d_textureTransform5.get())
    {
        d_textureTransform5.set(d_textureTransform5.get()->clone(ns));
        d_textureTransform5.get()->parentList.push_back(this);
    }
    if (d_texture6.get())
    {
        d_texture6.set(d_texture6.get()->clone(ns));
        d_texture6.get()->parentList.push_back(this);
    }
    if (d_textureTransform6.get())
    {
        d_textureTransform6.set(d_textureTransform6.get()->clone(ns));
        d_textureTransform6.get()->parentList.push_back(this);
    }
    if (d_texture7.get())
    {
        d_texture7.set(d_texture7.get()->clone(ns));
        d_texture7.get()->parentList.push_back(this);
    }
    if (d_textureTransform7.get())
    {
        d_textureTransform7.set(d_textureTransform7.get()->clone(ns));
        d_textureTransform7.get()->parentList.push_back(this);
    }
    if (d_texture8.get())
    {
        d_texture8.set(d_texture8.get()->clone(ns));
        d_texture8.get()->parentList.push_back(this);
    }
    if (d_textureTransform8.get())
    {
        d_textureTransform8.set(d_textureTransform8.get()->clone(ns));
        d_textureTransform8.get()->parentList.push_back(this);
    }
    if (d_texture9.get())
    {
        d_texture9.set(d_texture9.get()->clone(ns));
        d_texture9.get()->parentList.push_back(this);
    }
    if (d_textureTransform9.get())
    {
        d_textureTransform9.set(d_textureTransform9.get()->clone(ns));
        d_textureTransform9.get()->parentList.push_back(this);
    }
    if (d_texture10.get())
    {
        d_texture10.set(d_texture10.get()->clone(ns));
        d_texture10.get()->parentList.push_back(this);
    }
    if (d_textureTransform10.get())
    {
        d_textureTransform10.set(d_textureTransform10.get()->clone(ns));
        d_textureTransform10.get()->parentList.push_back(this);
    }
}

VrmlNodeAppearance *VrmlNodeAppearance::toAppearance() const
{
    return (VrmlNodeAppearance *)this;
}

bool VrmlNodeAppearance::isModified() const
{
    return (d_modified || (d_material.get() && d_material.get()->isModified()) || (d_texture.get() && d_texture.get()->isModified()) || (d_textureTransform.get() && d_textureTransform.get()->isModified()) || // additional fields for multi-texturing
            (d_texture2.get() && d_texture2.get()->isModified()) || (d_textureTransform2.get() && d_textureTransform2.get()->isModified()) || // additional fields for multi-texturing
            (d_texture3.get() && d_texture3.get()->isModified()) || (d_textureTransform3.get() && d_textureTransform3.get()->isModified()) || // additional fields for multi-texturing
            (d_texture4.get() && d_texture4.get()->isModified()) || (d_textureTransform4.get() && d_textureTransform4.get()->isModified()) || // additional fields for multi-texturing
            (d_texture5.get() && d_texture5.get()->isModified()) || (d_textureTransform5.get() && d_textureTransform5.get()->isModified()) || // additional fields for multi-texturing
            (d_texture6.get() && d_texture6.get()->isModified()) || (d_textureTransform6.get() && d_textureTransform6.get()->isModified()) || // additional fields for multi-texturing
            (d_texture7.get() && d_texture7.get()->isModified()) || (d_textureTransform7.get() && d_textureTransform7.get()->isModified()) || // additional fields for multi-texturing
            (d_texture8.get() && d_texture8.get()->isModified()) || (d_textureTransform8.get() && d_textureTransform8.get()->isModified()) || // additional fields for multi-texturing
            (d_texture9.get() && d_texture9.get()->isModified()) || (d_textureTransform9.get() && d_textureTransform9.get()->isModified()) || // additional fields for multi-texturing
            (d_texture10.get() && d_texture10.get()->isModified()) || (d_textureTransform10.get() && d_textureTransform10.get()->isModified()));
}

void VrmlNodeAppearance::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_material.get())
        d_material.get()->clearFlags();
    if (d_texture.get())
        d_texture.get()->clearFlags();
    if (d_textureTransform.get())
        d_textureTransform.get()->clearFlags();

    // additional fields for multi-texturing
    if (d_texture2.get())
        d_texture2.get()->clearFlags();
    if (d_textureTransform2.get())
        d_textureTransform2.get()->clearFlags();
    if (d_texture3.get())
        d_texture3.get()->clearFlags();
    if (d_textureTransform3.get())
        d_textureTransform3.get()->clearFlags();
    if (d_texture4.get())
        d_texture4.get()->clearFlags();
    if (d_textureTransform4.get())
        d_textureTransform4.get()->clearFlags();
    if (d_texture5.get())
        d_texture5.get()->clearFlags();
    if (d_textureTransform5.get())
        d_textureTransform5.get()->clearFlags();
    if (d_texture6.get())
        d_texture6.get()->clearFlags();
    if (d_textureTransform6.get())
        d_textureTransform6.get()->clearFlags();
    if (d_texture7.get())
        d_texture7.get()->clearFlags();
    if (d_textureTransform7.get())
        d_textureTransform7.get()->clearFlags();
    if (d_texture8.get())
        d_texture8.get()->clearFlags();
    if (d_textureTransform8.get())
        d_textureTransform8.get()->clearFlags();
    if (d_texture9.get())
        d_texture9.get()->clearFlags();
    if (d_textureTransform9.get())
        d_textureTransform9.get()->clearFlags();
    if (d_texture10.get())
        d_texture10.get()->clearFlags();
    if (d_textureTransform10.get())
        d_textureTransform10.get()->clearFlags();
}

void VrmlNodeAppearance::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    if (d_relativeUrl.get() == NULL)
        d_relativeUrl.set(rel);
    d_scene = s;
    if (d_material.get())
        d_material.get()->addToScene(s, rel);
    if (d_texture.get())
        d_texture.get()->addToScene(s, rel);
    if (d_textureTransform.get())
        d_textureTransform.get()->addToScene(s, rel);

    // additional fields for multi-texturing
    if (d_texture2.get())
        d_texture2.get()->addToScene(s, rel);
    if (d_textureTransform2.get())
        d_textureTransform2.get()->addToScene(s, rel);
    if (d_texture3.get())
        d_texture3.get()->addToScene(s, rel);
    if (d_textureTransform3.get())
        d_textureTransform3.get()->addToScene(s, rel);
    if (d_texture4.get())
        d_texture4.get()->addToScene(s, rel);
    if (d_textureTransform4.get())
        d_textureTransform4.get()->addToScene(s, rel);
    if (d_texture5.get())
        d_texture5.get()->addToScene(s, rel);
    if (d_textureTransform5.get())
        d_textureTransform5.get()->addToScene(s, rel);
    if (d_texture6.get())
        d_texture6.get()->addToScene(s, rel);
    if (d_textureTransform6.get())
        d_textureTransform6.get()->addToScene(s, rel);
    if (d_texture7.get())
        d_texture7.get()->addToScene(s, rel);
    if (d_textureTransform7.get())
        d_textureTransform7.get()->addToScene(s, rel);
    if (d_texture8.get())
        d_texture8.get()->addToScene(s, rel);
    if (d_textureTransform8.get())
        d_textureTransform8.get()->addToScene(s, rel);
    if (d_texture9.get())
        d_texture9.get()->addToScene(s, rel);
    if (d_textureTransform9.get())
        d_textureTransform9.get()->addToScene(s, rel);
    if (d_texture10.get())
        d_texture10.get()->addToScene(s, rel);
    if (d_textureTransform10.get())
        d_textureTransform10.get()->addToScene(s, rel);
    nodeStack.pop_front();
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeAppearance::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns); // Copy my routes

    // Copy subnode routes
    if (d_material.get())
        d_material.get()->copyRoutes(ns);
    if (d_texture.get())
        d_texture.get()->copyRoutes(ns);
    if (d_textureTransform.get())
        d_textureTransform.get()->copyRoutes(ns);

    // additional fields for multi-texturing
    if (d_texture2.get())
        d_texture2.get()->copyRoutes(ns);
    if (d_textureTransform2.get())
        d_textureTransform2.get()->copyRoutes(ns);
    if (d_texture3.get())
        d_texture3.get()->copyRoutes(ns);
    if (d_textureTransform3.get())
        d_textureTransform3.get()->copyRoutes(ns);
    if (d_texture4.get())
        d_texture4.get()->copyRoutes(ns);
    if (d_textureTransform4.get())
        d_textureTransform4.get()->copyRoutes(ns);
    if (d_texture5.get())
        d_texture5.get()->copyRoutes(ns);
    if (d_textureTransform5.get())
        d_textureTransform5.get()->copyRoutes(ns);
    if (d_texture6.get())
        d_texture6.get()->copyRoutes(ns);
    if (d_textureTransform6.get())
        d_textureTransform6.get()->copyRoutes(ns);
    if (d_texture7.get())
        d_texture7.get()->copyRoutes(ns);
    if (d_textureTransform7.get())
        d_textureTransform7.get()->copyRoutes(ns);
    if (d_texture8.get())
        d_texture8.get()->copyRoutes(ns);
    if (d_textureTransform8.get())
        d_textureTransform8.get()->copyRoutes(ns);
    if (d_texture9.get())
        d_texture9.get()->copyRoutes(ns);
    if (d_textureTransform9.get())
        d_textureTransform9.get()->copyRoutes(ns);
    if (d_texture10.get())
        d_texture10.get()->copyRoutes(ns);
    if (d_textureTransform10.get())
        d_textureTransform10.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

std::ostream &VrmlNodeAppearance::printFields(std::ostream &os, int indent)
{
    if (d_material.get())
        PRINT_FIELD(material);
    if (d_texture.get())
        PRINT_FIELD(texture);
    if (d_textureTransform.get())
        PRINT_FIELD(textureTransform);

    // additional fields for multi-texturing
    if (d_texture2.get())
        PRINT_FIELD(texture2);
    if (d_textureTransform2.get())
        PRINT_FIELD(textureTransform2);
    if (d_texture3.get())
        PRINT_FIELD(texture3);
    if (d_textureTransform3.get())
        PRINT_FIELD(textureTransform3);
    if (d_texture4.get())
        PRINT_FIELD(texture4);
    if (d_textureTransform4.get())
        PRINT_FIELD(textureTransform4);
    if (d_texture5.get())
        PRINT_FIELD(texture5);
    if (d_textureTransform5.get())
        PRINT_FIELD(textureTransform5);
    if (d_texture6.get())
        PRINT_FIELD(texture6);
    if (d_textureTransform6.get())
        PRINT_FIELD(textureTransform6);
    if (d_texture7.get())
        PRINT_FIELD(texture7);
    if (d_textureTransform7.get())
        PRINT_FIELD(textureTransform7);
    if (d_texture8.get())
        PRINT_FIELD(texture8);
    if (d_textureTransform8.get())
        PRINT_FIELD(textureTransform8);
    if (d_texture9.get())
        PRINT_FIELD(texture9);
    if (d_textureTransform9.get())
        PRINT_FIELD(textureTransform9);
    if (d_texture10.get())
        PRINT_FIELD(texture10);
    if (d_textureTransform10.get())
        PRINT_FIELD(textureTransform10);

    return os;
}

void VrmlNodeAppearance::render(Viewer *viewer)
{
    VrmlNodeMaterial *m = d_material.get() ? d_material.get()->toMaterial() : 0;
    VrmlNodeTexture *t = d_texture.get() ? d_texture.get()->toTexture() : 0;

    // additional fields for multi-texturing
    VrmlNodeTexture *t2 = d_texture2.get() ? d_texture2.get()->toTexture() : 0;
    VrmlNodeTexture *t3 = d_texture3.get() ? d_texture3.get()->toTexture() : 0;
    VrmlNodeTexture *t4 = d_texture4.get() ? d_texture4.get()->toTexture() : 0;
    VrmlNodeTexture *t5 = d_texture5.get() ? d_texture5.get()->toTexture() : 0;
    VrmlNodeTexture *t6 = d_texture6.get() ? d_texture6.get()->toTexture() : 0;
    VrmlNodeTexture *t7 = d_texture7.get() ? d_texture7.get()->toTexture() : 0;
    VrmlNodeTexture *t8 = d_texture8.get() ? d_texture8.get()->toTexture() : 0;
    VrmlNodeTexture *t9 = d_texture9.get() ? d_texture9.get()->toTexture() : 0;
    VrmlNodeTexture *t10 = d_texture10.get() ? d_texture10.get()->toTexture() : 0;

    if (m)
    {
        float trans = m->transparency();
        float *diff = m->diffuseColor();
        float diffuse[3] = { diff[0], diff[1], diff[2] };
        //int nTexComponents = t ? t->nComponents() : 0;
        //if (nTexComponents == 2 || nTexComponents == 4)
        //   trans = 0.0;
        //if (nTexComponents >= 3)
        //	diffuse[0] = diffuse[1] = diffuse[2] = 1.0;

        const char *relUrl = d_relativeUrl.get() ? d_relativeUrl.get() : d_scene->urlDoc()->url();
        viewer->setNameModes(m->name(), relUrl);
        viewer->setMaterial(m->ambientIntensity(),
                            diffuse,
                            m->emissiveColor(),
                            m->shininess(),
                            m->specularColor(),
                            trans);

        m->clearModified();
    }
    else
    {
        viewer->setColor(1.0, 1.0, 1.0); // default color
        viewer->enableLighting(false); // turn lighting off for this object
    }

    int numTextures = 0;
    if (t10)
    {
        numTextures = 10;
    }
    else if (t9)
    {
        numTextures = 9;
    }
    else if (t8)
    {
        numTextures = 8;
    }
    else if (t7)
    {
        numTextures = 7;
    }
    else if (t6)
    {
        numTextures = 6;
    }
    else if (t5)
    {
        numTextures = 5;
    }
    else if (t4)
    {
        numTextures = 4;
    }
    else if (t3)
    {
        numTextures = 3;
    }
    else if (t2)
    {
        numTextures = 2;
    }
    else if (t)
    {
        numTextures = 1;
    }

    viewer->setNumTextures(numTextures);

    if (t)
    {
        // first texture
        viewer->textureNumber = 0;

        if (d_textureTransform.get())
            d_textureTransform.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t->setAppearance(this);
        t->render(viewer);
    }

    // additional fields for multi-texturing
    // MAYBE something must be done here for multi-texturing to work correctly
    if (t2)
    {
        // second texture
        viewer->textureNumber = 1;

        if (d_textureTransform2.get())
            d_textureTransform2.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t2->render(viewer);
    }
    if (t3)
    {
        // third texture
        viewer->textureNumber = 2;

        if (d_textureTransform3.get())
            d_textureTransform3.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t3->render(viewer);
    }
    if (t4)
    {
        // fourth texture
        viewer->textureNumber = 3;

        if (d_textureTransform4.get())
            d_textureTransform4.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t4->render(viewer);
    }
    if (t5)
    {
        // fourth texture
        viewer->textureNumber = 4;

        if (d_textureTransform5.get())
            d_textureTransform5.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t5->render(viewer);
    }
    if (t6)
    {
        // fourth texture
        viewer->textureNumber = 5;

        if (d_textureTransform6.get())
            d_textureTransform6.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t6->render(viewer);
    }
    if (t7)
    {
        // fourth texture
        viewer->textureNumber = 6;

        if (d_textureTransform7.get())
            d_textureTransform7.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t7->render(viewer);
    }
    if (t8)
    {
        // fourth texture
        viewer->textureNumber = 7;

        if (d_textureTransform8.get())
            d_textureTransform8.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t8->render(viewer);
    }
    if (t9)
    {
        // fourth texture
        viewer->textureNumber = 8;

        if (d_textureTransform9.get())
            d_textureTransform9.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t9->render(viewer);
    }
    if (t10)
    {
        // fourth texture
        viewer->textureNumber = 9;

        if (d_textureTransform10.get())
            d_textureTransform10.get()->render(viewer);
        else
            viewer->setTextureTransform(0, 0, 0, 0);

        t10->render(viewer);
    }

    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeAppearance::setField(const char *fieldName,
                                  const VrmlField &fieldValue)
{
    if
        TRY_SFNODE_FIELD(material, Material)
    else if
        TRY_SFNODE_FIELD(texture, Texture)
    else if
        TRY_SFNODE_FIELD2(textureTransform, TextureTransform, MultiTextureTransform)
    // additional fields for multi-texturing
    else if
        TRY_SFNODE_FIELD(texture2, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform2, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture3, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform3, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture4, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform4, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture5, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform5, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture6, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform6, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture7, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform7, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture8, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform8, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture9, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform9, TextureTransform)
    else if
        TRY_SFNODE_FIELD(texture10, Texture)
    else if
        TRY_SFNODE_FIELD(textureTransform10, TextureTransform)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeAppearance::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "material") == 0)
        return &d_material;
    else if (strcmp(fieldName, "texture") == 0)
        return &d_texture;
    else if (strcmp(fieldName, "textureTransform") == 0)
        return &d_textureTransform;
    // additional fields for multi-texturing
    else if (strcmp(fieldName, "texture2") == 0)
        return &d_texture2;
    else if (strcmp(fieldName, "textureTransform2") == 0)
        return &d_textureTransform2;
    else if (strcmp(fieldName, "texture3") == 0)
        return &d_texture3;
    else if (strcmp(fieldName, "textureTransform3") == 0)
        return &d_textureTransform3;
    else if (strcmp(fieldName, "texture4") == 0)
        return &d_texture4;
    else if (strcmp(fieldName, "textureTransform4") == 0)
        return &d_textureTransform4;
    else if (strcmp(fieldName, "texture5") == 0)
        return &d_texture5;
    else if (strcmp(fieldName, "textureTransform5") == 0)
        return &d_textureTransform5;
    else if (strcmp(fieldName, "texture6") == 0)
        return &d_texture6;
    else if (strcmp(fieldName, "textureTransform6") == 0)
        return &d_textureTransform6;
    else if (strcmp(fieldName, "texture7") == 0)
        return &d_texture7;
    else if (strcmp(fieldName, "textureTransform7") == 0)
        return &d_textureTransform7;
    else if (strcmp(fieldName, "texture8") == 0)
        return &d_texture8;
    else if (strcmp(fieldName, "textureTransform8") == 0)
        return &d_textureTransform8;
    else if (strcmp(fieldName, "texture9") == 0)
        return &d_texture9;
    else if (strcmp(fieldName, "textureTransform9") == 0)
        return &d_textureTransform9;
    else if (strcmp(fieldName, "texture10") == 0)
        return &d_texture10;
    else if (strcmp(fieldName, "textureTransform10") == 0)
        return &d_textureTransform10;

    return VrmlNodeChild::getField(fieldName);
}
