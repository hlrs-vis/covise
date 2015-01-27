/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                                        **
 ** Description: coArrow                                                   **
 **              Draws an arrow with radius and length of the shaft        **
 **               direction is the z axis                                  **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "coArrow.h"

using namespace opencover;

coArrow::coArrow(float radius, float height, bool originAtTip, bool draw)
    : drawn(false)
{
    cone = new osg::Cone();
    coneDraw = new osg::ShapeDrawable();

    cylinder = new osg::Cylinder();
    cylinderDraw = new osg::ShapeDrawable();

    hints = new osg::TessellationHints();
    hints->setDetailRatio(0.2f);

    material = new osg::Material();

    if (draw)
    {
        if (originAtTip)
        {
            cone->set(osg::Vec3(0.0, 0.0, -height * 0.4), radius, height * 0.5);
            cylinder->set(osg::Vec3(0.0, 0.0, -height * 0.8), radius / 3.0, height);
        }
        else
        {
            cone->set(osg::Vec3(0.0, 0.0, height * 0.6), radius, height * 0.5);
            cylinder->set(osg::Vec3(0.0, 0.0, height * 0.2), radius / 5.0, height);
        }

        coneDraw->setShape(cone);
        coneDraw->setTessellationHints(hints);
        addDrawable(coneDraw);

        cylinderDraw->setShape(cylinder);
        cylinderDraw->setTessellationHints(hints);
        addDrawable(cylinderDraw);

        stateSet = coneDraw->getOrCreateStateSet();
        stateSet->setAttributeAndModes(material);

        coneDraw->setStateSet(stateSet);
        cylinderDraw->setStateSet(stateSet);

        coneDraw->setUseDisplayList(false);
        cylinderDraw->setUseDisplayList(false);

        drawn = true;
    }
}

void coArrow::drawArrow(osg::Vec3 base, float radius, float length)
{
    //fprintf(stderr,"coArrow::drawArrow base = (%f %f %f) radius = %f length = %f \n", base.x(), base.y(), base.z(), radius, length);

    float coneRadius = radius * 4.0f;
    float coneLength = coneRadius * 4.0f;
    float cylinderLength = length - coneLength;

    // draw the cylinder of the vector
    // the osg cylinders center is its center of mass
    // (offset factor: 0.5)
    cylinder->set(osg::Vec3(base.x(), base.y(), base.z() + length * 0.5f - coneLength * 0.5), radius, cylinderLength);

    // draw the cone of the vector
    // the osg cones center is its center of mass
    // (offset factor: 0.25)
    cone->set(osg::Vec3(base.x(), base.y(), base.z() + cylinderLength + coneLength * 0.25f), coneRadius, coneLength);

    if (!drawn)
    {
        cylinderDraw->setShape(cylinder);
        cylinderDraw->setTessellationHints(hints);
        addDrawable(cylinderDraw);

        coneDraw->setShape(cone);
        coneDraw->setTessellationHints(hints);
        addDrawable(coneDraw);

        stateSet = coneDraw->getOrCreateStateSet();
        stateSet->setAttributeAndModes(material);
        coneDraw->setStateSet(stateSet);
        cylinderDraw->setStateSet(stateSet);
        coneDraw->setUseDisplayList(false);
        cylinderDraw->setUseDisplayList(false);

        drawn = true;
    }
}

void coArrow::setVisible(bool visible)
{
    //fprintf(stderr,"coArrow::setVisible %d\n", visible );

    if (visible)
        this->setNodeMask(0xffffffff);
    else
        this->setNodeMask(0x00000000);
}

void coArrow::setColor(osg::Vec4 color)
{
    _color = color;
    material->setDiffuse(osg::Material::FRONT_AND_BACK, _color);
}

void coArrow::setAmbient(osg::Vec4 ambient)
{
    _ambient = ambient;
    material->setAmbient(osg::Material::FRONT_AND_BACK, _ambient);
}

coArrow::~coArrow()
{
    removeDrawable(coneDraw);
    removeDrawable(cylinderDraw);
}
