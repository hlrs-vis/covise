/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TARGET_H
#define _TARGET_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>

#include <PluginUtil/GenericGuiObject.h>

#include <iostream>
#include <vector>

using namespace opencover;

class Target : public GenericGuiObject
{
public:
    Target(osg::ref_ptr<osg::Group> parent);
    ~Target();

protected:
    void guiParamChanged(GuiParam *guiParam);

private:
    void setVisible(bool visible);
    void setPosition(osg::Vec3 pos);

    osg::ref_ptr<osg::Group> parentNode;

    GuiParamBool *p_visible;
    GuiParamVec3 *p_position;

    osg::ref_ptr<osg::Material> material;

    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::ShapeDrawable> drawable;
    osg::ref_ptr<osg::Sphere> geometry;
};

#endif
