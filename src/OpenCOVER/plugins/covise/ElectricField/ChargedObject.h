/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHARGED_OBJECT_H
#define _CHARGED_OBJECT_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>

#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include "cover/coVRLabel.h"

#include <PluginUtil/GenericGuiObject.h>
using namespace opencover;
using namespace vrui;

const float PI = 3.141592;
const float EPSILON_0 = 8.85418781762;
const float EPSILON_R = 1.00059;
const float EPSILON_POINT = 1.0 / (4.0 * PI * EPSILON_0 * EPSILON_R);

const unsigned int TYPE_POINT = 1;
const unsigned int TYPE_PLATE = 2;

class ChargedObject : public coMenuListener, public GenericGuiObject, public coVR3DTransInteractor
{
public:
    ChargedObject(unsigned int type_id, std::string name, float initialCharge);
    virtual ~ChargedObject();

    virtual void preFrame();

    void setPosition(osg::Vec3 position);
    osg::Vec3 getPosition()
    {
        return position;
    };

    virtual void setCharge(float charge);
    float getCharge()
    {
        return charge;
    };

    void setActive(bool active);
    bool isActive()
    {
        return active;
    };

    unsigned int getTypeId()
    {
        return type_id;
    };

    void setLabelVisibility(bool v);

    virtual osg::Vec3 getFieldAt(osg::Vec3 point) = 0;
    virtual float getPotentialAt(osg::Vec3 point) = 0;
    virtual osg::Vec4 getFieldAndPotentialAt(osg::Vec3 point) = 0;

    virtual void menuEvent(coMenuItem *menuItem);
    virtual void menuReleaseEvent(coMenuItem *menuItem);
    virtual void guiParamChanged(GuiParam *guiParam);

protected:
    virtual void activeStateChanged() = 0;

    osg::Vec3 position;
    bool active;
    float charge;

    osg::ref_ptr<osg::Material> objectMaterial;
    bool changedFromUser_;

private:
    void setColorAccordingToCharge();
    void adaptPositionToGrid();

    unsigned int type_id;

    GuiParamBool *p_active;
    GuiParamFloat *p_charge;
    GuiParamVec3 *p_position;

    coVR3DTransInteractor *pickInteractor;

    coLabelMenuItem *menuItemSeparator;
    coLabelMenuItem *menuItemCaption;
    coButtonMenuItem *menuItemDelete;
    coSliderMenuItem *menuItemCharge;

    coVRLabel *label;
};

#endif
