/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_H
#define _TRACER_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/LineWidth>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/GenericGuiObject.h>

#include <CovisePluginUtil/SmokeGeneratorSolutions.h>
#include "../../general/Tracer/TracerPlane.h"

using namespace opencover;
using namespace vrui;

class Tracer : public GenericGuiObject, public coMenuListener
{
public:
    Tracer();
    virtual ~Tracer();

    void preFrame();
    void update();

protected:
    void menuEvent(coMenuItem *menuItem);
    void guiParamChanged(GuiParam *guiParam);

private:
    GuiParamBool *p_visible;
    GuiParamMatrix *p_matrix;

    coCheckboxMenuItem *menuItemVisible;

    coVR3DTransRotInteractor *interactor;
    SmokeGeneratorSolutions solutions_;
    void displaySmoke();
    osg::ref_ptr<osg::Geode> smokeGeode_;
    osg::ref_ptr<osg::Geometry> smokeGeometry_;
    osg::ref_ptr<osg::Vec4Array> smokeColor_;
    osg::ref_ptr<osg::Vec3Array> coordLine_;
    osg::ref_ptr<osg::Vec3Array> coordPoly_;
    osg::ref_ptr<osg::Vec3Array> polyNormal_;
    osg::ref_ptr<osg::Geode> geometryNode; ///< Geometry node (plane)
    osg::ref_ptr<osg::Geometry> geometryLine_; ///< Geometry object (plane)
    osg::ref_ptr<osg::Geometry> geometryPoly_; ///< Geometry object (plane)
};

#endif
