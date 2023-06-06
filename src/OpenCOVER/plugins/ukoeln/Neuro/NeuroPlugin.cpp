/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <vector>

#include <osg/Geometry>
#include <osg/Image>
#include <osg/io_utils>
#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/TexMat>
#include <osg/Texture2D>

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>

#include <virvo/osg/VolumeDrawable.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvvoldesc.h>

#include "NeuroPlugin.h"

using namespace covise;
using namespace opencover;
using namespace vrui;

osg::Geometry *createTexturedQuadGeometry_(const osg::Vec3& corner,
                                           const osg::Vec3& widthVec,
                                           const osg::Vec3& heightVec,
                                           const osg::Vec2& tc1,
                                           const osg::Vec2& tc2,
                                           const osg::Vec2& tc3,
                                           const osg::Vec2& tc4)
{
    osg::Geometry *retval = new osg::Geometry;

    // vertices
    osg::Vec3Array* verts = new osg::Vec3Array(4);
    (*verts)[0] = corner + heightVec;
    (*verts)[1] = corner;
    (*verts)[2] = corner + widthVec;
    (*verts)[3] = corner + widthVec + heightVec;
    retval->setVertexArray(verts);

    // textures coordinates
    osg::Vec2Array* texcoords = new osg::Vec2Array(4);
    (*texcoords)[0] = tc1;
    (*texcoords)[1] = tc2;
    (*texcoords)[2] = tc3;
    (*texcoords)[3] = tc4;
    retval->setTexCoordArray(0, texcoords);

    osg::Vec4Array* col = new osg::Vec4Array(1);
    (*col)[0] = osg::Vec4(1., 1., 1., 1.);
    retval->setColorArray(col, osg::Array::BIND_OVERALL);

    osg::Vec3Array* normals = new osg::Vec3Array(1);
    (*normals)[0] = widthVec ^ heightVec;
    (*normals)[0].normalize();
    retval->setNormalArray(normals, osg::Array::BIND_OVERALL);

    retval->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));

    return retval;
}

class FindVolumeVisitor : public osg::NodeVisitor
{
public:
    using osg::NodeVisitor::apply;

public:
    FindVolumeVisitor(TraversalMode tm = TRAVERSE_ALL_CHILDREN)
        : osg::NodeVisitor(tm)
        , volDesc_(nullptr)
    {
    }

    void apply(osg::Node &node)
    {
        if (auto geode = dynamic_cast<osg::Geode *>(&node))
        {
            for (size_t i = 0; i < geode->getNumDrawables(); ++i)
            {
                auto drawable = geode->getDrawable(i);
                if (auto vd = dynamic_cast<virvo::VolumeDrawable *>(drawable))
                {
                    volDesc_ = vd->getVolumeDescription();
                    transform_ = dynamic_cast<osg::MatrixTransform *>(node.getParent(0));
                }
            }
        }

        osg::NodeVisitor::traverse(node);
    }

    const vvVolDesc *getVolDesc() const
    {
        return volDesc_;
    }

    osg::ref_ptr<osg::MatrixTransform> getTransform() const
    {
        return transform_;
    }

private:
    // Virvo volume description.
    vvVolDesc *volDesc_;
    // Transform node hanging above volume drawable's geode.
    osg::ref_ptr<osg::MatrixTransform> transform_;

};

NeuroPlugin::NeuroPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, volDesc_(nullptr)
, repaintSlices_(true)
, slicePosX_(-1)
, slicePosY_(-1)
, slicePosZ_(-1)
, minSliceIntensity_(.2f)
, maxSliceIntensity_(1.f)
{
    vrui.mainMenuEntry.reset(new coSubMenuItem("Neuro..."));
    opencover::cover->getMenu()->add(vrui.mainMenuEntry.get());
    vrui.mainMenu.reset(new coRowMenu("Neuro", cover->getMenu()));
    vrui.mainMenuEntry->setMenu(vrui.mainMenu.get());

    vrui.sliceMenuEntry.reset(new coSubMenuItem("Slices..."));
    vrui.mainMenu->add(vrui.sliceMenuEntry.get());

    vrui.sliceMenu.reset(new coRowMenu("Slices", vrui.mainMenu.get()));
    vrui.sliceMenuEntry->setMenu(vrui.sliceMenu.get());

    vrui.minIntensitySlider.reset(new coSliderMenuItem("Minimum intensity", .0f, 1.f, .2f));
    vrui.minIntensitySlider->setInteger(false);
    vrui.minIntensitySlider->setMenuListener(this);
    vrui.sliceMenu->add(vrui.minIntensitySlider.get());

    vrui.maxIntensitySlider.reset(new coSliderMenuItem("Maximum intensity", .0f, 1.f, 1.f));
    vrui.maxIntensitySlider->setInteger(false);
    vrui.maxIntensitySlider->setMenuListener(this);
    vrui.sliceMenu->add(vrui.maxIntensitySlider.get());
}

NeuroPlugin::~NeuroPlugin()
{
    if (transform_ != nullptr)
    {
        transform_->removeChild(transYZ_);
        transform_->removeChild(transXZ_);
        transform_->removeChild(transXY_);
    }
}

bool NeuroPlugin::init()
{
    return true;
}

void NeuroPlugin::preFrame()
{
    if (volDesc_ == nullptr)
    {
        // Need to initially find the volume drawable.
        FindVolumeVisitor visitor;
        opencover::cover->getObjectsRoot()->accept(visitor); 
        volDesc_ = visitor.getVolDesc();
        transform_ = visitor.getTransform();

        if (volDesc_ != nullptr && transform_ != nullptr)
        {
            volDesc_->findMinMax(0, minVoxel_, maxVoxel_);
            maxVoxel_ /= volDesc_->getValueRange();

            addSliceGeometry(virvo::cartesian_axis<3>::X);
            addSliceGeometry(virvo::cartesian_axis<3>::Y);
            addSliceGeometry(virvo::cartesian_axis<3>::Z);

            auto size = volDesc_->getSize();
            auto size2 = size * .5f;

            // default size for all interactors
            float interSize = -1.f;
            // if defined, COVER.IconSize overrides the default
            interSize = coCoviseConfig::getFloat("COVER.IconSize", interSize);
            // if defined, COVERConfigCuttingSurfacePlugin.IconSize overrides both
            interSize = coCoviseConfig::getFloat("COVER.Plugin.Cuttingsurface.IconSize", interSize);

            // X interactor
            interactorYZ_.reset(new coVR1DTransInteractor(applyTrans(osg::Vec3()), applyTrans(osg::Vec3(1., 0., 0.)),
                                                          interSize, vrui::coInteraction::ButtonA,
                                                          "hand", "YZ",
                                                          vrui::coInteraction::Medium));
            interactorYZ_->enableIntersection();
            updateInteractorPos(virvo::cartesian_axis<3>::X, 0);


            // Y interactor
            interactorXZ_.reset(new coVR1DTransInteractor(applyTrans(osg::Vec3()), applyTrans(osg::Vec3(0., 1., 0.)),
                                                          interSize, vrui::coInteraction::ButtonA,
                                                          "hand", "XZ",
                                                          vrui::coInteraction::Medium));
            interactorXZ_->enableIntersection();
            updateInteractorPos(virvo::cartesian_axis<3>::Y, 0);


            // Z interactor
            interactorXY_.reset(new coVR1DTransInteractor(applyTrans(osg::Vec3()), applyTrans(osg::Vec3(0., 0., 1.)),
                                                          interSize, vrui::coInteraction::ButtonA,
                                                          "hand", "XY",
                                                          vrui::coInteraction::Medium));
            interactorXY_->enableIntersection();
            updateInteractorPos(virvo::cartesian_axis<3>::Z, 0);

            repaintSlices_ = true;
        }
    }

    if (volDesc_ != nullptr)
    {
        // Repaint slices.
        if (repaintSlices_ || getSlicePos(virvo::cartesian_axis<3>::X) != slicePosX_)
        {
            slicePosX_ = getSlicePos(virvo::cartesian_axis<3>::X);//std::cout << "slicePosX: " << slicePosX_ << '\n';
            osg::Vec3 pos = transYZ_->getPosition();
            pos[0] = applyTrans(interactorYZ_->getPosition())[0];
            transYZ_->setPosition(pos);
            setSliceTexture(virvo::cartesian_axis<3>::X, slicePosX_);
        }

        if (repaintSlices_ || getSlicePos(virvo::cartesian_axis<3>::Y) != slicePosY_)
        {
            slicePosY_ = getSlicePos(virvo::cartesian_axis<3>::Y);//std::cout << "slicePosY: " << slicePosY_ << '\n';
            osg::Vec3 pos = transXZ_->getPosition();
            pos[1] = applyTrans(interactorXZ_->getPosition())[1];
            transXZ_->setPosition(pos);
            setSliceTexture(virvo::cartesian_axis<3>::Y, slicePosY_);
        }

        if (repaintSlices_ || getSlicePos(virvo::cartesian_axis<3>::Z) != slicePosZ_)
        {
            slicePosZ_ = getSlicePos(virvo::cartesian_axis<3>::Z);//std::cout << "slicePosZ: " << slicePosZ_ << '\n';
            osg::Vec3 pos = transXY_->getPosition();
            pos[2] = applyTrans(interactorXY_->getPosition())[2];
            transXY_->setPosition(pos);
            setSliceTexture(virvo::cartesian_axis<3>::Z, slicePosZ_);
        }

        repaintSlices_ = false;
    }
}

osg::Vec3 NeuroPlugin::applyTrans(const osg::Vec3& v)
{
    if (transform_ != nullptr)
        return transform_->getMatrix() * v;
    else
        return v;
}

void NeuroPlugin::addSliceGeometry(virvo::cartesian_axis<3> axis)
{
    auto size = volDesc_->getSize();
    auto size2 = size * .5f;

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;

    if (axis == virvo::cartesian_axis<3>::X)
    {
        quadYZ_ = createTexturedQuadGeometry_(osg::Vec3(0., -size2.y, -size2.z),
                                              osg::Vec3(0., size.y, 0.),
                                              osg::Vec3(0., 0., size.z),
                                              osg::Vec2(0., 0.), osg::Vec2(1., 0.), osg::Vec2(1., 1.), osg::Vec2(0., 1.));
        transYZ_ = new osg::PositionAttitudeTransform;

        osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
        auto state = quadYZ_->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0, tex);

        geode->addDrawable(quadYZ_);
        transYZ_->addChild(geode);
        transform_->addChild(transYZ_);
    }
    else if (axis == virvo::cartesian_axis<3>::Y)
    {
        quadXZ_ = osg::createTexturedQuadGeometry(osg::Vec3(-size2.x, 0., -size2.z),
                                                  osg::Vec3(size.x, 0., 0.),
                                                  osg::Vec3(0., 0., size.z));
        transXZ_ = new osg::PositionAttitudeTransform;

        osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
        auto state = quadXZ_->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0, tex);

        geode->addDrawable(quadXZ_);
        transXZ_->addChild(geode);
        transform_->addChild(transXZ_);
    }
    else if (axis == virvo::cartesian_axis<3>::Z)
    {
        quadXY_ = osg::createTexturedQuadGeometry(osg::Vec3(-size2.x, -size2.y, 0.),
                                                  osg::Vec3(size.x, 0., 0.),
                                                  osg::Vec3(0., size.y, 0.));
        transXY_ = new osg::PositionAttitudeTransform;

        osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
        auto state = quadXY_->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0, tex);

        geode->addDrawable(quadXY_);
        transXY_->addChild(geode);
        transform_->addChild(transXY_);
    }
}

void NeuroPlugin::setSliceTexture(virvo::cartesian_axis<3> axis, int sliceNum)
{
    size_t width;
    size_t height;
    size_t slices;

    vvTransFunc tf;
    float lo = minSliceIntensity_;
    float hi = maxSliceIntensity_;
    tf._widgets.push_back(new vvTFColor(vvColor(lo, lo, lo), minVoxel_));
    tf._widgets.push_back(new vvTFColor(vvColor(hi, hi, hi), maxVoxel_));

    volDesc_->getVolumeSize(axis, width, height, slices);
    sliceTextures[(int)axis].resize(width * height * 3);
    auto& texture = sliceTextures[(int)axis];
    volDesc_->makeSliceImage(volDesc_->getCurrentFrame(), axis, slices - sliceNum - 1, texture.data(), &tf);

    osg::ref_ptr<osg::Image> img = new osg::Image;
    img->setOrigin(osg::Image::TOP_LEFT);
    img->allocateImage(width, height, 1, GL_RGB, GL_UNSIGNED_BYTE);
    img->setImage(width, height, 1, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, texture.data(), osg::Image::NO_DELETE);

    if (axis == virvo::cartesian_axis<3>::X)
    {
        img->flipVertical();
    }

    if (axis == virvo::cartesian_axis<3>::Z)
    {
        img->flipVertical();
    }

    if (axis == virvo::cartesian_axis<3>::X)
    {
        auto state = quadYZ_->getOrCreateStateSet();
        auto tex = dynamic_cast<osg::Texture2D *>(state->getTextureAttribute(0, osg::StateAttribute::TEXTURE));

        if (tex != nullptr)
            tex->setImage(img);
    }
    else if (axis == virvo::cartesian_axis<3>::Y)
    {
        auto state = quadXZ_->getOrCreateStateSet();
        auto tex = dynamic_cast<osg::Texture2D *>(state->getTextureAttribute(0, osg::StateAttribute::TEXTURE));

        if (tex != nullptr)
            tex->setImage(img);
    }
    else if (axis == virvo::cartesian_axis<3>::Z)
    {
        auto state = quadXY_->getOrCreateStateSet();
        auto tex = dynamic_cast<osg::Texture2D *>(state->getTextureAttribute(0, osg::StateAttribute::TEXTURE));

        if (tex != nullptr)
            tex->setImage(img);
    }
}

void NeuroPlugin::updateInteractorPos(virvo::cartesian_axis<3> axis, int sliceNum)
{
    auto size = volDesc_->getSize();
    auto size2 = size * .5f;

    if (axis == virvo::cartesian_axis<3>::X)
    {
        interactorYZ_->updateTransform(applyTrans(osg::Vec3(0., -size2.y, -size2.z)));
    }
    else if (axis == virvo::cartesian_axis<3>::Y)
    {
        interactorXZ_->updateTransform(applyTrans(osg::Vec3(-size2.x, 0., -size2.z)));
    }
    else if (axis == virvo::cartesian_axis<3>::Z)
    {
        interactorXY_->updateTransform(applyTrans(osg::Vec3(-size2.x, -size2.y, 0.)));
    }
}

int NeuroPlugin::getSlicePos(virvo::cartesian_axis<3> axis)
{
    auto size = volDesc_->getSize();
    auto size2 = size * .5f;

    if (axis == virvo::cartesian_axis<3>::X)
    {
        osg::Vec3 pos = applyTrans(interactorYZ_->getPosition()) + osg::Vec3(size2.x, size2.y, size2.z);
        float posx = (pos.x() / size.x) * volDesc_->vox[0];
        return ts_clamp(static_cast<int>(posx), 0, (int)volDesc_->vox[0]-1);
    }
    else if (axis == virvo::cartesian_axis<3>::Y)
    {
        osg::Vec3 pos = applyTrans(interactorXZ_->getPosition()) + osg::Vec3(size2.x, size2.y, size2.z);
        float posy = (pos.y() / size.y) * volDesc_->vox[1];
        return ts_clamp(static_cast<int>(posy), 0, (int)volDesc_->vox[1]-1);
    }
    else if (axis == virvo::cartesian_axis<3>::Z)
    {
        osg::Vec3 pos = applyTrans(interactorXY_->getPosition()) + osg::Vec3(size2.x, size2.y, size2.z);
        float posz = (pos.z() / size.z) * volDesc_->vox[2];
        return ts_clamp(static_cast<int>(posz), 0, (int)volDesc_->vox[2]-1);
    }

    assert(0);

    return -1;
}

void NeuroPlugin::menuEvent(vrui::coMenuItem *item)
{
    // slice menu
    if (item == vrui.minIntensitySlider.get())
    {
        minSliceIntensity_ = vrui.minIntensitySlider->getValue();
        repaintSlices_ = true;
    }
    else if (item == vrui.maxIntensitySlider.get())
    {
        maxSliceIntensity_ = vrui.maxIntensitySlider->getValue();
        repaintSlices_ = true;
    }
}

COVERPLUGIN(NeuroPlugin)
