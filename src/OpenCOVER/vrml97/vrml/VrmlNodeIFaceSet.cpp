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

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeIFaceSet(s); }

// Define the built in VrmlNodeType:: "IndexedFaceSet" fields

VrmlNodeType *VrmlNodeIFaceSet::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("IndexedFaceSet", creator);
    }

    VrmlNodeIndexedSet::defineType(t); // Parent class

    t->addEventIn("set_normalIndex", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex", VrmlField::MFINT32);
    // additional fields for multi-texturing
    t->addEventIn("set_texCoordIndex2", VrmlField::MFINT32);
    // additional fields for multi-texturing
    t->addEventIn("set_texCoordIndex3", VrmlField::MFINT32);
    // additional fields for multi-texturing
    t->addEventIn("set_texCoordIndex4", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex5", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex6", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex7", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex8", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex9", VrmlField::MFINT32);
    t->addEventIn("set_texCoordIndex10", VrmlField::MFINT32);
    t->addExposedField("normal", VrmlField::SFNODE);
    t->addExposedField("texCoord", VrmlField::SFNODE);
    // additional fields for multi-texturing
    t->addExposedField("texCoord2", VrmlField::SFNODE);
    // additional fields for multi-texturing
    t->addExposedField("texCoord3", VrmlField::SFNODE);
    // additional fields for multi-texturing
    t->addExposedField("texCoord4", VrmlField::SFNODE);
    t->addExposedField("texCoord5", VrmlField::SFNODE);
    t->addExposedField("texCoord6", VrmlField::SFNODE);
    t->addExposedField("texCoord7", VrmlField::SFNODE);
    t->addExposedField("texCoord8", VrmlField::SFNODE);
    t->addExposedField("texCoord9", VrmlField::SFNODE);
    t->addExposedField("texCoord10", VrmlField::SFNODE);
    t->addField("ccw", VrmlField::SFBOOL);
    t->addField("convex", VrmlField::SFBOOL);
    t->addField("creaseAngle", VrmlField::SFFLOAT);
    t->addField("normalIndex", VrmlField::MFINT32);
    t->addField("normalPerVertex", VrmlField::SFBOOL);
    t->addField("solid", VrmlField::SFBOOL);
    t->addField("texCoordIndex", VrmlField::MFINT32);
    // additional fields for multi-texturing
    t->addField("texCoordIndex2", VrmlField::MFINT32);
    // additional fields for multi-texturing
    t->addField("texCoordIndex3", VrmlField::MFINT32);
    // additional fields for multi-texturing
    t->addField("texCoordIndex4", VrmlField::MFINT32);
    t->addField("texCoordIndex5", VrmlField::MFINT32);
    t->addField("texCoordIndex6", VrmlField::MFINT32);
    t->addField("texCoordIndex7", VrmlField::MFINT32);
    t->addField("texCoordIndex8", VrmlField::MFINT32);
    t->addField("texCoordIndex9", VrmlField::MFINT32);
    t->addField("texCoordIndex10", VrmlField::MFINT32);

    return t;
}

VrmlNodeType *VrmlNodeIFaceSet::nodeType() const { return defineType(0); }

VrmlNodeIFaceSet::VrmlNodeIFaceSet(VrmlScene *scene)
    : VrmlNodeIndexedSet(scene)
    , d_ccw(true)
    , d_convex(true)
    , d_creaseAngle(System::the->defaultCreaseAngle())
    , d_normalPerVertex(true)
    , d_solid(true)
{
}

VrmlNodeIFaceSet::~VrmlNodeIFaceSet()
{
}

VrmlNode *VrmlNodeIFaceSet::cloneMe() const
{
    return new VrmlNodeIFaceSet(*this);
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
    if (d_texCoord.get())
    {
        d_texCoord.set(d_texCoord.get()->clone(ns));
        d_texCoord.get()->parentList.push_back(this);
    }
    // additional fields for multi-texturing
    if (d_texCoord2.get())
    {
        d_texCoord2.set(d_texCoord2.get()->clone(ns));
        d_texCoord2.get()->parentList.push_back(this);
    }
    if (d_texCoord3.get())
    {
        d_texCoord3.set(d_texCoord3.get()->clone(ns));
        d_texCoord3.get()->parentList.push_back(this);
    }
    if (d_texCoord4.get())
    {
        d_texCoord4.set(d_texCoord4.get()->clone(ns));
        d_texCoord4.get()->parentList.push_back(this);
    }
    if (d_texCoord5.get())
    {
        d_texCoord5.set(d_texCoord5.get()->clone(ns));
        d_texCoord5.get()->parentList.push_back(this);
    }
    if (d_texCoord6.get())
    {
        d_texCoord6.set(d_texCoord6.get()->clone(ns));
        d_texCoord6.get()->parentList.push_back(this);
    }
    if (d_texCoord7.get())
    {
        d_texCoord7.set(d_texCoord7.get()->clone(ns));
        d_texCoord7.get()->parentList.push_back(this);
    }
    if (d_texCoord8.get())
    {
        d_texCoord8.set(d_texCoord8.get()->clone(ns));
        d_texCoord8.get()->parentList.push_back(this);
    }
    if (d_texCoord9.get())
    {
        d_texCoord9.set(d_texCoord9.get()->clone(ns));
        d_texCoord9.get()->parentList.push_back(this);
    }
    if (d_texCoord10.get())
    {
        d_texCoord10.set(d_texCoord10.get()->clone(ns));
        d_texCoord10.get()->parentList.push_back(this);
    }
}

bool VrmlNodeIFaceSet::isModified() const
{
    return (d_modified || (d_color.get() && d_color.get()->isModified()) || (d_coord.get() && d_coord.get()->isModified()) || (d_normal.get() && d_normal.get()->isModified()) || (d_texCoord.get() && d_texCoord.get()->isModified()) ||
            // additional fields for multi-texturing
            (d_texCoord2.get() && d_texCoord2.get()->isModified()) || (d_texCoord3.get() && d_texCoord3.get()->isModified()) || (d_texCoord4.get() && d_texCoord4.get()->isModified()) || (d_texCoord5.get() && d_texCoord5.get()->isModified()) || (d_texCoord6.get() && d_texCoord6.get()->isModified()) || (d_texCoord7.get() && d_texCoord7.get()->isModified()) || (d_texCoord8.get() && d_texCoord8.get()->isModified()) || (d_texCoord9.get() && d_texCoord9.get()->isModified()) || (d_texCoord10.get() && d_texCoord10.get()->isModified()));
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
    if (d_texCoord.get())
        d_texCoord.get()->clearFlags();
    // additional fields for multi-texturing
    if (d_texCoord2.get())
        d_texCoord2.get()->clearFlags();
    if (d_texCoord3.get())
        d_texCoord3.get()->clearFlags();
    if (d_texCoord4.get())
        d_texCoord4.get()->clearFlags();
    if (d_texCoord5.get())
        d_texCoord5.get()->clearFlags();
    if (d_texCoord6.get())
        d_texCoord6.get()->clearFlags();
    if (d_texCoord7.get())
        d_texCoord7.get()->clearFlags();
    if (d_texCoord8.get())
        d_texCoord8.get()->clearFlags();
    if (d_texCoord9.get())
        d_texCoord9.get()->clearFlags();
    if (d_texCoord10.get())
        d_texCoord10.get()->clearFlags();
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
    if (d_texCoord.get())
        d_texCoord.get()->addToScene(s, rel);
    // additional fields for multi-texturing
    if (d_texCoord2.get())
        d_texCoord2.get()->addToScene(s, rel);
    if (d_texCoord3.get())
        d_texCoord3.get()->addToScene(s, rel);
    if (d_texCoord4.get())
        d_texCoord4.get()->addToScene(s, rel);
    if (d_texCoord5.get())
        d_texCoord5.get()->addToScene(s, rel);
    if (d_texCoord6.get())
        d_texCoord6.get()->addToScene(s, rel);
    if (d_texCoord7.get())
        d_texCoord7.get()->addToScene(s, rel);
    if (d_texCoord8.get())
        d_texCoord8.get()->addToScene(s, rel);
    if (d_texCoord9.get())
        d_texCoord9.get()->addToScene(s, rel);
    if (d_texCoord10.get())
        d_texCoord10.get()->addToScene(s, rel);
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
    if (d_texCoord.get())
        d_texCoord.get()->copyRoutes(ns);
    // additional fields for multi-texturing
    if (d_texCoord2.get())
        d_texCoord2.get()->copyRoutes(ns);
    if (d_texCoord3.get())
        d_texCoord3.get()->copyRoutes(ns);
    if (d_texCoord4.get())
        d_texCoord4.get()->copyRoutes(ns);
    if (d_texCoord5.get())
        d_texCoord5.get()->copyRoutes(ns);
    if (d_texCoord6.get())
        d_texCoord6.get()->copyRoutes(ns);
    if (d_texCoord7.get())
        d_texCoord7.get()->copyRoutes(ns);
    if (d_texCoord8.get())
        d_texCoord8.get()->copyRoutes(ns);
    if (d_texCoord9.get())
        d_texCoord9.get()->copyRoutes(ns);
    if (d_texCoord10.get())
        d_texCoord10.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

std::ostream &VrmlNodeIFaceSet::printFields(std::ostream &os, int indent)
{
    if (!d_ccw.get())
        PRINT_FIELD(ccw);
    if (!d_convex.get())
        PRINT_FIELD(convex);
    if (!d_normalPerVertex.get())
        PRINT_FIELD(normalPerVertex);
    if (!d_solid.get())
        PRINT_FIELD(solid);

    if (d_creaseAngle.get() != 0.0)
        PRINT_FIELD(creaseAngle);
    if (d_normal.get())
        PRINT_FIELD(normal);
    if (d_normalIndex.size() > 0)
        PRINT_FIELD(normalIndex);
    if (d_color.get())
        PRINT_FIELD(color);
    if (d_colorIndex.size() > 0)
        PRINT_FIELD(colorIndex);
    if (d_texCoord.get())
        PRINT_FIELD(texCoord);
    if (d_texCoordIndex.size() > 0)
        PRINT_FIELD(texCoordIndex);

    // additional fields for multi-texturing
    if (d_texCoord2.get())
        PRINT_FIELD(texCoord2);
    if (d_texCoordIndex2.size() > 0)
        PRINT_FIELD(texCoordIndex2);
    if (d_texCoord3.get())
        PRINT_FIELD(texCoord3);
    if (d_texCoordIndex3.size() > 0)
        PRINT_FIELD(texCoordIndex3);
    if (d_texCoord4.get())
        PRINT_FIELD(texCoord4);
    if (d_texCoordIndex4.size() > 0)
        PRINT_FIELD(texCoordIndex4);
    if (d_texCoord5.get())
        PRINT_FIELD(texCoord5);
    if (d_texCoordIndex5.size() > 0)
        PRINT_FIELD(texCoordIndex5);
    if (d_texCoord6.get())
        PRINT_FIELD(texCoord6);
    if (d_texCoordIndex6.size() > 0)
        PRINT_FIELD(texCoordIndex6);
    if (d_texCoord7.get())
        PRINT_FIELD(texCoord7);
    if (d_texCoordIndex7.size() > 0)
        PRINT_FIELD(texCoordIndex7);
    if (d_texCoord8.get())
        PRINT_FIELD(texCoord8);
    if (d_texCoordIndex8.size() > 0)
        PRINT_FIELD(texCoordIndex8);
    if (d_texCoord9.get())
        PRINT_FIELD(texCoord9);
    if (d_texCoordIndex9.size() > 0)
        PRINT_FIELD(texCoordIndex9);
    if (d_texCoord10.get())
        PRINT_FIELD(texCoord10);
    if (d_texCoordIndex10.size() > 0)
        PRINT_FIELD(texCoordIndex10);

    VrmlNodeIndexedSet::printFields(os, indent);

    return os;
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
                                             d_texCoord, d_texCoordIndex,
                                             d_texCoord2, d_texCoordIndex2,
                                             d_texCoord3, d_texCoordIndex3,
                                             d_texCoord4, d_texCoordIndex4);

    if (d_color.get())
        d_color.get()->clearModified();
    if (d_coord.get())
        d_coord.get()->clearModified();
    if (d_normal.get())
        d_normal.get()->clearModified();
    if (d_texCoord.get())
        d_texCoord.get()->clearModified();
    if (d_texCoord2.get())
        d_texCoord2.get()->clearModified();
    if (d_texCoord3.get())
        d_texCoord3.get()->clearModified();
    if (d_texCoord4.get())
        d_texCoord4.get()->clearModified();
    if (d_texCoord5.get())
        d_texCoord5.get()->clearModified();
    if (d_texCoord6.get())
        d_texCoord6.get()->clearModified();
    if (d_texCoord7.get())
        d_texCoord7.get()->clearModified();
    if (d_texCoord8.get())
        d_texCoord8.get()->clearModified();
    if (d_texCoord9.get())
        d_texCoord9.get()->clearModified();
    if (d_texCoord10.get())
        d_texCoord10.get()->clearModified();

    return obj;
}

// Set the value of one of the node fields.

void VrmlNodeIFaceSet::setField(const char *fieldName,
                                const VrmlField &fieldValue)
{
    if
        TRY_FIELD(ccw, SFBool)
    else if
        TRY_FIELD(convex, SFBool)
    else if
        TRY_FIELD(creaseAngle, SFFloat)
    else if
        TRY_SFNODE_FIELD(normal, Normal)
    else if
        TRY_FIELD(normalIndex, MFInt)
    else if
        TRY_FIELD(normalPerVertex, SFBool)
    else if
        TRY_FIELD(solid, SFBool)
    else if
        TRY_SFNODE_FIELD3(texCoord, TextureCoordinate, MultiTextureCoordinate, TextureCoordinateGenerator)
    // additional fields for multi-texturing
    else if
        TRY_SFNODE_FIELD(texCoord2, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord3, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord4, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord5, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord6, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord7, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord8, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord9, TextureCoordinate)
    else if
        TRY_SFNODE_FIELD(texCoord10, TextureCoordinate)
    else if
        TRY_FIELD(texCoordIndex, MFInt)
    // additional fields for multi-texturing
    else if
        TRY_FIELD(texCoordIndex2, MFInt)
    else if
        TRY_FIELD(texCoordIndex3, MFInt)
    else if
        TRY_FIELD(texCoordIndex4, MFInt)
    else if
        TRY_FIELD(texCoordIndex5, MFInt)
    else if
        TRY_FIELD(texCoordIndex6, MFInt)
    else if
        TRY_FIELD(texCoordIndex7, MFInt)
    else if
        TRY_FIELD(texCoordIndex8, MFInt)
    else if
        TRY_FIELD(texCoordIndex9, MFInt)
    else if
        TRY_FIELD(texCoordIndex10, MFInt)
    else if
        TRY_FIELD(colorIndex, MFInt)
    else if
        TRY_FIELD(colorPerVertex, SFBool)
    else
        VrmlNodeIndexedSet::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeIFaceSet::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "ccw") == 0)
        return &d_ccw;
    else if (strcmp(fieldName, "convex") == 0)
        return &d_convex;
    else if (strcmp(fieldName, "creaseAngle") == 0)
        return &d_creaseAngle;
    else if (strcmp(fieldName, "normal") == 0)
        return &d_normal;
    else if (strcmp(fieldName, "normalIndex") == 0)
        return &d_normalIndex;
    else if (strcmp(fieldName, "normalPerVertex") == 0)
        return &d_normalPerVertex;
    else if (strcmp(fieldName, "solid") == 0)
        return &d_solid;
    else if (strcmp(fieldName, "normalIndex") == 0)
        return &d_normalIndex;
    else if (strcmp(fieldName, "texCoord") == 0)
        return &d_texCoord;
    else if (strcmp(fieldName, "texCoordIndex") == 0)
        return &d_texCoordIndex;
    // additional fields for multi-texturing
    else if (strcmp(fieldName, "texCoord2") == 0)
        return &d_texCoord2;
    else if (strcmp(fieldName, "texCoordIndex2") == 0)
        return &d_texCoordIndex2;
    else if (strcmp(fieldName, "texCoord3") == 0)
        return &d_texCoord3;
    else if (strcmp(fieldName, "texCoordIndex3") == 0)
        return &d_texCoordIndex3;
    else if (strcmp(fieldName, "texCoord4") == 0)
        return &d_texCoord4;
    else if (strcmp(fieldName, "texCoordIndex4") == 0)
        return &d_texCoordIndex4;
    else if (strcmp(fieldName, "texCoord5") == 0)
        return &d_texCoord5;
    else if (strcmp(fieldName, "texCoordIndex5") == 0)
        return &d_texCoordIndex5;
    else if (strcmp(fieldName, "texCoord6") == 0)
        return &d_texCoord6;
    else if (strcmp(fieldName, "texCoordIndex6") == 0)
        return &d_texCoordIndex6;
    else if (strcmp(fieldName, "texCoord7") == 0)
        return &d_texCoord7;
    else if (strcmp(fieldName, "texCoordIndex7") == 0)
        return &d_texCoordIndex7;
    else if (strcmp(fieldName, "texCoord8") == 0)
        return &d_texCoord8;
    else if (strcmp(fieldName, "texCoordIndex8") == 0)
        return &d_texCoordIndex8;
    else if (strcmp(fieldName, "texCoord9") == 0)
        return &d_texCoord9;
    else if (strcmp(fieldName, "texCoordIndex9") == 0)
        return &d_texCoordIndex9;
    else if (strcmp(fieldName, "texCoord10") == 0)
        return &d_texCoord10;
    else if (strcmp(fieldName, "texCoordIndex10") == 0)
        return &d_texCoordIndex10;
    else if (strcmp(fieldName, "colorIndex") == 0)
        return &d_colorIndex;
    else if (strcmp(fieldName, "colorPerVertex") == 0)
        return &d_colorPerVertex;

    return VrmlNodeIndexedSet::getField(fieldName);
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
    return d_texCoord.get();
}

const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex() const
{
    return d_texCoordIndex;
}

// additional fields for multi-texturing
VrmlNode *VrmlNodeIFaceSet::getTexCoord2()
{
    return d_texCoord2.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex2() const
{
    return d_texCoordIndex2;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord3()
{
    return d_texCoord3.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex3() const
{
    return d_texCoordIndex3;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord4()
{
    return d_texCoord4.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex4() const
{
    return d_texCoordIndex4;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord5()
{
    return d_texCoord5.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex5() const
{
    return d_texCoordIndex5;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord6()
{
    return d_texCoord6.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex6() const
{
    return d_texCoordIndex6;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord7()
{
    return d_texCoord7.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex7() const
{
    return d_texCoordIndex7;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord8()
{
    return d_texCoord8.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex8() const
{
    return d_texCoordIndex8;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord9()
{
    return d_texCoord9.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex9() const
{
    return d_texCoordIndex9;
}

VrmlNode *VrmlNodeIFaceSet::getTexCoord10()
{
    return d_texCoord10.get();
}
const VrmlMFInt &VrmlNodeIFaceSet::getTexCoordIndex10() const
{
    return d_texCoordIndex10;
}

VrmlNodeIFaceSet *VrmlNodeIFaceSet::toIFaceSet() const
{
    return (VrmlNodeIFaceSet *)this;
}
