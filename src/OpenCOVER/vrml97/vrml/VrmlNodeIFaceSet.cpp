/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIFaceSet.cpp

#include "VrmlNodeIFaceSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeTextureCoordinate.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

void VrmlNodeIFaceSet::initFields(VrmlNodeIFaceSet *node, VrmlNodeType *t)
{
    VrmlNodeIndexedSet::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     field("ccw", node->d_ccw),
                     field("convex", node->d_convex),
                     field("creaseAngle", node->d_creaseAngle),
                     exposedField("normal", node->d_normal),
                     field("normalIndex", node->d_normalIndex),
                     field("normalPerVertex", node->d_normalPerVertex),
                     field("solid", node->d_solid));

    for (size_t i = 0; i < MAX_TEXCOORDS; i++)
    {
        std::string suffix = i == 0 ? std::string() : std::to_string(i + 1);
        initFieldsHelper(node, t,
                         exposedField("texCoord" + suffix, node->d_texCoords[i]),
                         field("texCoordIndex" + suffix, node->d_texCoordIndices[i]));
        if(t)
        {
            t->addEventIn(("set_texCoordIndex" + suffix).c_str(), VrmlField::MFINT32);
        }
    }
    
    if(t)
    {
        t->addEventIn("set_normalIndex", VrmlField::MFINT32);
    }
}


const char *VrmlNodeIFaceSet::name() { return "IndexedFaceSet"; }


VrmlNodeIFaceSet::VrmlNodeIFaceSet(VrmlScene *scene)
    : VrmlNodeIndexedSet(scene, name())
    , d_ccw(true)
    , d_convex(true)
    , d_creaseAngle(System::the->defaultCreaseAngle())
    , d_normalPerVertex(true)
    , d_solid(true)
{
}

void VrmlNodeIFaceSet::cloneChildren(VrmlNamespace *ns)
{
    if (d_color.get())
    {
        d_color.set(d_color.get()->clone(ns));
        d_color.get()->parentList.push_back(this);
    }
    if (d_coord.get())
    {
        d_coord.set(d_coord.get()->clone(ns));
        d_coord.get()->parentList.push_back(this);
    }
    if (d_normal.get())
    {
        d_normal.set(d_normal.get()->clone(ns));
        d_normal.get()->parentList.push_back(this);
    }

    for (size_t i = 0; i < MAX_TEXCOORDS; i++)
    {
        if(d_texCoords[i].get())
        {
            d_texCoords[i].set(d_texCoords[i].get()->clone(ns));
            d_texCoords[i].get()->parentList.push_back(this);
        }
    }
}

bool VrmlNodeIFaceSet::isModified() const
{
    bool retval = (d_modified
            || (d_color.get() && d_color.get()->isModified())
            || (d_coord.get() && d_coord.get()->isModified())
            || (d_normal.get() && d_normal.get()->isModified()));

            for (size_t i = 0; i < MAX_TEXCOORDS; i++)
            {
                if(d_texCoords[i].get() && d_texCoords[i].get()->isModified())
                {
                    return true;
                }
            }
    return retval;
}

void VrmlNodeIFaceSet::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_color.get())
        d_color.get()->clearFlags();
    if (d_coord.get())
        d_coord.get()->clearFlags();
    if (d_normal.get())
        d_normal.get()->clearFlags();
    for (size_t i = 0; i < MAX_TEXCOORDS; i++)
    {
        if(d_texCoords[i].get())
            d_texCoords[i].get()->clearFlags();
    }
}

void VrmlNodeIFaceSet::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_color.get())
        d_color.get()->addToScene(s, rel);
    if (d_coord.get())
        d_coord.get()->addToScene(s, rel);
    if (d_normal.get())
        d_normal.get()->addToScene(s, rel);
    
    for (size_t i = 0; i < MAX_TEXCOORDS; i++)
    {
        if(d_texCoords[i].get())
            d_texCoords[i].get()->addToScene(s, rel);
    }

    nodeStack.pop_front();
}

void VrmlNodeIFaceSet::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_color.get())
        d_color.get()->copyRoutes(ns);
    if (d_coord.get())
        d_coord.get()->copyRoutes(ns);
    if (d_normal.get())
        d_normal.get()->copyRoutes(ns);
    
    for (size_t i = 0; i < MAX_TEXCOORDS; i++)
    {
        if(d_texCoords[i].get())
            d_texCoords[i].get()->copyRoutes(ns);
    }
    
    nodeStack.pop_front();
}

Viewer::Object VrmlNodeIFaceSet::insertGeometry(Viewer *viewer)
{
    Viewer::Object obj;

    unsigned int optMask = 0;

    if (d_ccw.get())
        optMask |= Viewer::MASK_CCW;
    if (d_convex.get())
        optMask |= Viewer::MASK_CONVEX;
    if (d_solid.get())
        optMask |= Viewer::MASK_SOLID;
    if (d_colorPerVertex.get())
        optMask |= Viewer::MASK_COLOR_PER_VERTEX;
    if (d_normalPerVertex.get())
        optMask |= Viewer::MASK_NORMAL_PER_VERTEX;

    obj = VrmlNodeColoredSet::insertGeometry(viewer, optMask, d_coordIndex,
                                             d_colorIndex, d_creaseAngle,
                                             d_normal, d_normalIndex,
                                             d_texCoords[0], d_texCoordIndices[0],
                                             d_texCoords[1], d_texCoordIndices[1],
                                             d_texCoords[2], d_texCoordIndices[2],
                                             d_texCoords[3], d_texCoordIndices[3]);

    if (d_color.get())
        d_color.get()->clearModified();
    if (d_coord.get())
        d_coord.get()->clearModified();
    if (d_normal.get())
        d_normal.get()->clearModified();
    
    for (size_t i = 0; i < MAX_TEXCOORDS; i++)
    {
        if(d_texCoords[i].get())
            d_texCoords[i].get()->clearModified();
    }
    return obj;
}

VrmlNode *VrmlNodeIFaceSet::getNormal()
{
    return d_normal.get();
}

const VrmlMFInt &VrmlNodeIFaceSet::getNormalIndex() const
{
    return d_normalIndex;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord()
{
    return d_texCoords[0].get();
}

const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex() const
{
    return d_texCoordIndices[0];
}

// additional fields for multi-texturing
VrmlNode *VrmlNodeIFaceSet::getTexCoord2()
{
    return d_texCoords[1].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex2() const
{
    return d_texCoordIndices[1];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord3()
{
    return d_texCoords[2].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex3() const
{
    return d_texCoordIndices[2];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord4()
{
    return d_texCoords[3].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex4() const
{
    return d_texCoordIndices[3];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord5()
{
    return d_texCoords[4].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex5() const
{
    return d_texCoordIndices[4];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord6()
{
    return d_texCoords[5].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex6() const
{
    return d_texCoordIndices[5];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord7()
{
    return d_texCoords[6].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex7() const
{
    return d_texCoordIndices[6];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord8()
{
    return d_texCoords[7].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex8() const
{
    return d_texCoordIndices[7];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord9()
{
    return d_texCoords[8].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex9() const
{
    return d_texCoordIndices[8];
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord10()
{
    return d_texCoords[9].get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex10() const
{
    return d_texCoordIndices[9];
}

VrmlNodeIFaceSet *VrmlNodeIFaceSet::toIFaceSet() const
{
    return (VrmlNodeIFaceSet *)this;
}
