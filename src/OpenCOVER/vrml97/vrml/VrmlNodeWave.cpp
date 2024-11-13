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

void VrmlNodeWave::initFields(VrmlNodeWave *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("fraction", node->d_fraction),
                     exposedField("speed1", node->d_speed1),
                     exposedField("speed2", node->d_speed2),
                     exposedField("freq1", node->d_freq1),
                     exposedField("height1", node->d_height1),
                     exposedField("damping1", node->d_damping1),
                     exposedField("dir1", node->d_dir1),
                     exposedField("freq2", node->d_freq2),
                     exposedField("height2", node->d_height2),
                     exposedField("damping2", node->d_damping2),
                     exposedField("dir2", node->d_dir2),
                     exposedField("coeffSin", node->d_coeffSin),
                     exposedField("coeffCos", node->d_coeffCos),
                     exposedField("fileName", node->d_fileName));
}

const char *VrmlNodeWave::name() { return "Wave"; }

VrmlNodeWave::VrmlNodeWave(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
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

void VrmlNodeWave::addToScene(VrmlScene *, const char * /*rel*/)
{
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
