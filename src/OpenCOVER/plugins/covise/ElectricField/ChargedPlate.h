/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHARGED_PLATE_H
#define _CHARGED_PLATE_H

#include "ChargedObject.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>

const int NUM_APPROXIMATION_POINTS = 50;

class ChargedPlate : public ChargedObject
{
public:
    ChargedPlate(std::string name, float initialCharge);
    virtual ~ChargedPlate();

    virtual osg::Vec3 getFieldAt(osg::Vec3 point);
    virtual float getPotentialAt(osg::Vec3 point);
    virtual osg::Vec4 getFieldAndPotentialAt(osg::Vec3 point);

    void correctField(osg::Vec3 point, osg::Vec3 &field);
    void correctPotential(osg::Vec3 point, float &potential);
    void correctFieldAndPotential(osg::Vec3 point, osg::Vec4 &fieldAndPotential);

    void setOtherPlate(ChargedPlate *plate);

    void setCharge(float charge);

    void setRadius(float radius, bool fromUser = false);
    float getRadius()
    {
        return radius;
    };

    void menuEvent(coMenuItem *menuItem);
    void menuReleaseEvent(coMenuItem *menuItem);
    void guiParamChanged(GuiParam *guiParam);

protected:
    virtual void createGeometry(); // interactor

private:
    ChargedPlate *otherPlate;

    void activeStateChanged();

    float radius;

    GuiParamFloat *p_radius;

    osg::ref_ptr<osg::ShapeDrawable> plateDrawable;
    osg::ref_ptr<osg::Cylinder> plateCylinder;

    //coSliderMenuItem* menuItemRadius;
};

#endif
