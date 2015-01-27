/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeWave.cpp
//

#include "VrmlNodeWave.h"
#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeWave(scene);
}

// Define the built in VrmlNodeType:: "Wave" fields

VrmlNodeType *VrmlNodeWave::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Wave", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("fraction", VrmlField::SFFLOAT);
    t->addExposedField("freq1", VrmlField::SFFLOAT);
    t->addExposedField("height1", VrmlField::SFFLOAT);
    t->addExposedField("damping1", VrmlField::SFFLOAT);
    t->addExposedField("dir1", VrmlField::SFVEC3F);
    t->addExposedField("freq2", VrmlField::SFFLOAT);
    t->addExposedField("height2", VrmlField::SFFLOAT);
    t->addExposedField("damping2", VrmlField::SFFLOAT);
    t->addExposedField("dir2", VrmlField::SFVEC3F);
    t->addExposedField("speed1", VrmlField::SFFLOAT);
    t->addExposedField("speed2", VrmlField::SFFLOAT);
    t->addExposedField("coeffSin", VrmlField::SFROTATION);
    t->addExposedField("coeffCos", VrmlField::SFROTATION);
    t->addExposedField("fileName", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeWave::nodeType() const { return defineType(0); }

VrmlNodeWave::VrmlNodeWave(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_fraction(0)
    , d_speed1(1.0f)
    , d_speed2(1.0f)
    , d_freq1(0.18f)
    , d_height1(0.1f)
    , d_damping1(1.0f)
    , d_dir1(1, 1, 0)
    , d_freq2(3.0f)
    , d_height2(0.1f)
    , d_damping2(1.0f)
    , d_dir2(1.0f, 0)
    , d_coeffSin(1.0f, -1.0f / 6.0f, 1.0f / 120.0f, -1.0f / 5040.0f)
    , d_coeffCos(1.0f, -1.0f / 2.0f, 1.0f / 24.0f, -1.0f / 720.0f)
    , d_fileName("cg_water.cg")
{
}

VrmlNodeWave::~VrmlNodeWave()
{
}

VrmlNode *VrmlNodeWave::cloneMe() const
{
    return new VrmlNodeWave(*this);
}

VrmlNodeWave *VrmlNodeWave::toWave() const
{
    return (VrmlNodeWave *)this;
}

void VrmlNodeWave::addToScene(VrmlScene *, const char * /*rel*/)
{
}

std::ostream &VrmlNodeWave::printFields(std::ostream &os, int indent)
{
    if (d_fileName.get())
        PRINT_FIELD(fileName);

    return os;
}

void VrmlNodeWave::render(Viewer *viewer)
{
    float dir1[3] = { d_dir1.x(), d_dir1.y(), d_dir1.z() };
    float dir2[3] = { d_dir2.x(), d_dir2.y(), d_dir2.z() };
    //cerr << "Fraction: " << d_fraction.get() << endl;
    viewer->insertWave(d_fraction.get(),
                       d_speed1.get(),
                       d_speed2.get(),
                       d_freq1.get(),
                       d_height1.get(),
                       d_damping1.get(),
                       dir1,
                       d_freq2.get(),
                       d_height2.get(),
                       d_damping2.get(),
                       dir2,
                       d_coeffSin.get(),
                       d_coeffCos.get(),
                       d_fileName.get());

    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeWave::setField(const char *fieldName,
                            const VrmlField &fieldValue)
{
    if
        TRY_FIELD(fraction, SFFloat)
    else if
        TRY_FIELD(speed1, SFFloat)
    else if
        TRY_FIELD(speed2, SFFloat)
    else if
        TRY_FIELD(freq1, SFFloat)
    else if
        TRY_FIELD(height1, SFFloat)
    else if
        TRY_FIELD(damping1, SFFloat)
    else if
        TRY_FIELD(dir1, SFVec3f)
    else if
        TRY_FIELD(freq2, SFFloat)
    else if
        TRY_FIELD(height2, SFFloat)
    else if
        TRY_FIELD(damping2, SFFloat)
    else if
        TRY_FIELD(dir2, SFVec3f)
    else if
        TRY_FIELD(coeffSin, SFRotation)
    else if
        TRY_FIELD(coeffCos, SFRotation)
    else if
        TRY_FIELD(fileName, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeWave::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "fraction") == 0)
        return &d_fraction;
    else if (strcmp(fieldName, "speed1") == 0)
        return &d_speed1;
    else if (strcmp(fieldName, "speed2") == 0)
        return &d_speed2;
    else if (strcmp(fieldName, "freq1") == 0)
        return &d_freq1;
    else if (strcmp(fieldName, "height1") == 0)
        return &d_height1;
    else if (strcmp(fieldName, "damping1") == 0)
        return &d_damping1;
    else if (strcmp(fieldName, "dir1") == 0)
        return &d_dir1;
    else if (strcmp(fieldName, "freq2") == 0)
        return &d_freq2;
    else if (strcmp(fieldName, "height2") == 0)
        return &d_height2;
    else if (strcmp(fieldName, "damping2") == 0)
        return &d_damping2;
    else if (strcmp(fieldName, "dir2") == 0)
        return &d_dir2;
    else if (strcmp(fieldName, "coeffSin") == 0)
        return &d_coeffSin;
    else if (strcmp(fieldName, "coeffCos") == 0)
        return &d_coeffCos;
    else if (strcmp(fieldName, "fileName") == 0)
        return &d_fileName;

    return VrmlNodeChild::getField(fieldName);
}
