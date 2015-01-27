/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ChargedPoint.h"
#include <iostream>
#include <algorithm>

using namespace std;

ChargedPoint::ChargedPoint(std::string name, float initialCharge)
    : ChargedObject(TYPE_POINT, name, initialCharge)
{
    // interactor
    createGeometry();
}

ChargedPoint::~ChargedPoint()
{
}

void ChargedPoint::activeStateChanged()
{
}

void ChargedPoint::createGeometry()
{
    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());
    // create new goemtry
    geometryNode = new osg::Geode();
    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0.0, 0.0, 0.0), 0.075);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    ((osg::Geode *)this->geometryNode.get())->addDrawable(mySphereDrawable);
    ((osg::Geode *)this->geometryNode.get())->getOrCreateStateSet()->setAttribute(objectMaterial.get(), osg::StateAttribute::ON /*PROTECTED*/);
    // add
    scaleTransform->addChild(geometryNode.get());
}

///////////////////////////////////////////////////////////////////////////////

osg::Vec3 ChargedPoint::getFieldAt(osg::Vec3 point)
{
    osg::Vec3 vector = point - position;
    float len = vector.length();
    len = max(len, MINIMUM_DISTANCE);
    return vector * ((charge * EPSILON_POINT) / (len * len * len));
}

float ChargedPoint::getPotentialAt(osg::Vec3 point)
{
    osg::Vec3 vector = point - position;
    float len = vector.length();
    len = max(len, MINIMUM_DISTANCE);
    return (charge * EPSILON_POINT) / len;
}

osg::Vec4 ChargedPoint::getFieldAndPotentialAt(osg::Vec3 point)
{
    osg::Vec3 vector = point - position;
    float len = vector.length();
    len = max(len, MINIMUM_DISTANCE);
    return osg::Vec4(
        vector * ((charge * EPSILON_POINT) / (len * len * len)),
        (charge * EPSILON_POINT) / len);
}
