/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeGeoreferenced.cpp

#include "VrmlNodeGeoreferenced.h"

#include "vrml97/vrml/VrmlNodeType.h"

#include <geodata/GeoData.h>

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeGeoreferenced(s); }

// Define the built in VrmlNodeType:: "Transform" fields

void VrmlNodeGeoreferenced::initFields(VrmlNodeGeoreferenced *node, VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
        exposedField("geolocation", &VrmlNodeGeoreferenced::d_geolocation),
        exposedField("fixAxes", &VrmlNodeGeoreferenced::d_fixAxes));
}

const char *VrmlNodeGeoreferenced::typeName() { return "Georeferenced"; }

VrmlNodeGeoreferenced::VrmlNodeGeoreferenced(VrmlScene *scene)
    : VrmlNodeGroup(scene, typeName())
    , d_geolocation(0.0, 0.0, 0.0)
    , d_xformObject(0)
{
    d_modified = true;

    m_connection = opencover::GeoData::instance()->projectionChanged.connect([&]()
        { setModified(); });
}

VrmlNodeGeoreferenced::VrmlNodeGeoreferenced(const VrmlNodeGeoreferenced &other)
    : VrmlNodeGroup(other)
    , d_geolocation(0.0, 0.0, 0.0)
    , d_xformObject(0)
{
    d_modified = true;

    m_connection = opencover::GeoData::instance()->projectionChanged.connect([&]()
        { setModified(); });
}

void VrmlNodeGeoreferenced::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_xformObject && isModified())
    {
        viewer->removeObject(d_xformObject);
        d_xformObject = 0;
    }
    checkAndRemoveNodes(viewer);
    if (d_xformObject)
    {
        viewer->insertReference(d_xformObject);
    }
    else if (d_children.size() > 0)
    {
        d_xformObject = viewer->beginObject(name(), 0, this);

        auto t = getTranslation();
        auto r = getRotation();
        float translation[] { t.x(), t.y(), t.z() };
        float rotation[] { (float)r.x(), (float)r.y(), (float)r.z(), (float)r.w() };
        float scale[] { 1, 1, 1 };
        float center[] { 0, 0, 0 };
        float zeroRotation[] { 1, 0, 0, 0 };

        // Apply transforms
        viewer->setTransform(center,
            rotation,
            scale,
            zeroRotation,
            translation,
            d_modified);

        // Render children
        VrmlNodeGroup::render(viewer);

        // Reverse transforms (for immediate mode/no matrix stack renderer)
        viewer->unsetTransform(center,
            rotation,
            scale,
            zeroRotation,
            translation);

        viewer->endObject();
    }

    clearModified();
}

osg::Vec3 VrmlNodeGeoreferenced::getTranslation()
{
    osg::Vec3 t(d_geolocation.x(), d_geolocation.y(), d_geolocation.z());
    auto r = t * opencover::GeoData::instance()->projectTransform();
    return osg::Vec3(r.x(), r.z(), -r.y()); // vrml coordinates are flipped :)
}

osg::Quat VrmlNodeGeoreferenced::getRotation()
{
    if (!d_fixAxes.get())
    {
        return osg::Quat(1, 0, 0, 0);
    }
    else
    {
        return osg::Quat(1, 0, 0, -M_PI / 2);
    }
}

void VrmlNodeGeoreferenced::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;

    int i, n = d_children.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_children[i];
        if (kid)
            kid->accumulateTransform(this);
    }
}

void VrmlNodeGeoreferenced::inverseTransform(Viewer *viewer)
{
    VrmlNode *parentTransform = getParentTransform();
    if (parentTransform)
        parentTransform->inverseTransform(viewer);

    auto t = getTranslation();
    auto r = getRotation();
    float trans[] { -t.x(), -t.y(), -t.z() };
    float rotation[] { (float)r.x(), (float)r.y(), (float)r.z(), (float)r.w() };
    float zeroRotation[] { 1, 0, 0, 0 };
    float scale[] = { 1, 1, 1 };
    float zero[] = { 0, 0, 0 };

    // Apply transforms (this may need to be broken into separate
    // calls to perform the ops in reverse order...)
    viewer->setTransform(zero,
        rotation,
        scale,
        zeroRotation,
        trans, true);
}

void VrmlNodeGeoreferenced::inverseTransform(double *m)
{
    VrmlNode *parentTransform = getParentTransform();
    osg::Matrixd matrix;
    matrix.makeIdentity();

    if (parentTransform)
    {
        parentTransform->inverseTransform(matrix.ptr());
    }

    // Invert this transform
    auto t = getTranslation();
    auto r = getRotation();

    matrix.preMult(osg::Matrixd::translate(t));
    matrix.preMult(osg::Matrixd::rotate(r));
    memcpy(m, matrix.ptr(), 16 * sizeof(double));
}
