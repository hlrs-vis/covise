/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NEURO_PLUGIN_H
#define NEURO_PLUGIN_H

#include <memory>

#include <osg/ref_ptr>
#include <osg/Vec3>

#include <virvo/math/axis.h>

#include <cover/coVRPlugin.h>

#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>

#include <PluginUtil/coVR1DTransInteractor.h>

class vvVolDesc;

namespace osg
{
class MatrixTransform;
class PositionAttitudeTransform;
}

class NeuroPlugin : public opencover::coVRPlugin
                  , public vrui::coMenuListener
{
public:

    // typedefs
    using vruiCheckBox    = std::unique_ptr<vrui::coCheckboxMenuItem>;
    using vruiMenu        = std::unique_ptr<vrui::coMenu>;
    using vruiRadioButton = std::unique_ptr<vrui::coCheckboxMenuItem>;
    using vruiRadioGroup  = std::unique_ptr<vrui::coCheckboxGroup>;
    using vruiSlider      = std::unique_ptr<vrui::coSliderMenuItem>;
    using vruiSubMenu     = std::unique_ptr<vrui::coSubMenuItem>;

public:

    NeuroPlugin();
    ~NeuroPlugin();

    bool init();
    void preFrame();

private:
    const vvVolDesc *volDesc_;
    osg::ref_ptr<osg::MatrixTransform> transform_;

    osg::ref_ptr<osg::PositionAttitudeTransform> transYZ_;
    osg::ref_ptr<osg::PositionAttitudeTransform> transXZ_;
    osg::ref_ptr<osg::PositionAttitudeTransform> transXY_;

    osg::ref_ptr<osg::Geometry> quadYZ_;
    osg::ref_ptr<osg::Geometry> quadXZ_;
    osg::ref_ptr<osg::Geometry> quadXY_;

    // Transform vertex with the matrix used to reorient the volume.
    osg::Vec3 applyTrans(const osg::Vec3& v);

    // Add slice quads (geometry only) to the scene graph.
    void addSliceGeometry(virvo::cartesian_axis<3> axis);

    // Set slice quads' texture property.
    void setSliceTexture(virvo::cartesian_axis<3> axis, int sliceNum);

    // Update position of slice interactor.
    void updateInteractorPos(virvo::cartesian_axis<3> axis, int sliceNum);

    // Determine current slice based on interactor position.
    int getSlicePos(virvo::cartesian_axis<3> axis);

    std::unique_ptr<opencover::coVR1DTransInteractor> interactorYZ_;
    std::unique_ptr<opencover::coVR1DTransInteractor> interactorXZ_;
    std::unique_ptr<opencover::coVR1DTransInteractor> interactorXY_;

    bool repaintSlices_;
    int slicePosX_;
    int slicePosY_;
    int slicePosZ_;

    // Minimum intensity value for ortho slices.
    float minSliceIntensity_;
    // Maximum intensity value for ortho slices.
    float maxSliceIntensity_;

    // Minimum voxel value [0..1].
    float minVoxel_;
    // Maximum voxel value [0..1].
    float maxVoxel_;

    //--- VRUI --------------------------------------------

    struct
    {
        vruiMenu    mainMenu;
        vruiSubMenu mainMenuEntry;

        vruiMenu    sliceMenu;
        vruiSubMenu sliceMenuEntry;
        vruiSlider  minIntensitySlider;
        vruiSlider  maxIntensitySlider;
    } vrui;

    void menuEvent(vrui::coMenuItem *item);
};

#endif
