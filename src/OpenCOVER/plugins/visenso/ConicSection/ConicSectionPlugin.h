/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: ConicSectionPlugin                                          **
 **              for VR4Schule mathematics                                 **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _CONIC_SECTION_PLUGIN_H
#define _CONIC_SECTION_PLUGIN_H

#include <osg/BoundingBox>

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <config/CoviseConfig.h>

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>

namespace opencover
{
class coPlane;
}
namespace vrui
{
class coRowMenu;
}

using namespace vrui;
using namespace opencover;

class ConicSectionPlugin : public coVRPlugin, public coMenuListener
{
public:
    // variables of class
    static ConicSectionPlugin *plugin;

    //inherit from coVRPlugin or MenuListener
    virtual bool init();
    virtual void preFrame();
    virtual void menuEvent(coMenuItem *menuItem);

    // constructor destructor
    ConicSectionPlugin();
    virtual ~ConicSectionPlugin();

protected:
    void makeMenu();

private:
    std::string calculateSection(osg::Vec4 eq);
    std::string sectionString(osg::Vec4 eq);
    void drawPlane(osg::Vec4 eq);
    bool isNull(float f);

    // menu
    coRowMenu *conicMenu_;
    coCheckboxMenuItem *showClipplane_;
    coLabelMenuItem *sectionPlaneEquation_;
    coLabelMenuItem *sectionType_;
    coLabelMenuItem *sectionEquation_;

    // gemoetry for cones
    osg::ref_ptr<osg::MatrixTransform> Cone_;
    osg::ref_ptr<osg::MatrixTransform> topConeTransform_;
    osg::ref_ptr<osg::Cone> topCone_;
    osg::ref_ptr<osg::Cone> bottomCone_;
    osg::ref_ptr<osg::ShapeDrawable> topConeDraw_;
    osg::ref_ptr<osg::ShapeDrawable> bottomConeDraw_;
    osg::ref_ptr<osg::Geode> topConeGeode_;
    osg::ref_ptr<osg::Geode> bottomConeGeode_;

    // geometry for plane
    osg::ref_ptr<osg::Geometry> plane_;
    osg::ref_ptr<osg::Geode> planeGeode_;
    osg::ref_ptr<osg::ShapeDrawable> planeDraw_;
    osg::ref_ptr<osg::StateSet> stateSet_;
    osg::ref_ptr<osg::Material> material_;
    osg::Vec3 drawPoints_[8];
    osg::Vec4 color_;

    // old plane
    osg::Vec4 oldPlane;
    coPlane *helperPlane_;
    osg::ref_ptr<osg::Vec3Array> polyCoords_;
    osg::ref_ptr<osg::Vec3Array> polyNormal_;
};

#endif
