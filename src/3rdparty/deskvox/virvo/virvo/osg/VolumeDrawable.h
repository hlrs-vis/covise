// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_OSG_VOLUME_DRAWABLE
#define VV_OSG_VOLUME_DRAWABLE

#include <string>
#include <osg/Drawable>
#include <osg/Geode>
#include <virvo/math/math.h>
#include <virvo/texture/forward.h>
#include <virvo/vvvecmath.h>
#include <virvo/vvrenderer.h>

#include "vvosgexport.h"

namespace osg
{
class State;
}

class vvVolDesc;
class vvTransFunc;

namespace virvo {

class VIRVOOSGEXPORT VolumeDrawable : public osg::Drawable
{
public:
    VolumeDrawable(std::string rendererName = "", std::string voxType = "");
    virtual ~VolumeDrawable();

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

    void setParameter(vvRenderState::ParameterType param, const vvParam &newValue);

    void setCurrentFrame(int);
    int getCurrentFrame() const;
    int getNumFrames() const;
    void setViewDirection(const osg::Vec3 &);
    void setObjectDirection(const osg::Vec3 &);

    void setClipDirection(const osg::Vec3 &);
    void setClipPoint(const osg::Vec3 &);
    void setClipping(bool enable);
    bool getClipping() const;
    void setSingleSliceClipping(bool enable);

    void setROIPosition(const osg::Vec3 &);
    void setROISize(float);
    void setROISelected(bool value);
    osg::Vec3 getROIPosition() const;
    float getROISize() const;

    void setChannelWeights(std::vector<float> const& cw);
    void setUseChannelWeights(bool enable);
    bool getUseChannelWeights() const;

    void setPosition(const osg::Vec3 &);
    osg::Vec3 getPosition() const;
    osg::Vec3 getCenter() const;
    void getBoundingBox(osg::Vec3 *min, osg::Vec3 *max) const;
    vvVolDesc *getVolumeDescription() const;
    void setVolumeDescription(vvVolDesc *vd);
    bool getInstantMode() const;
    bool have3DTextures() const;
    void setPreintegration(bool value);
    void setLighting(bool value);
    void setInterpolation(bool value);
    bool getInterpolation() const;
    void setBoundaries(bool value);
    void setBoundariesActive(bool value);
    void setQuality(float quality);
    float getQuality() const;
    void enableFlatDisplay(bool enable);

    void setTransferFunction(const vvTransFunc &tf, int chan = 0);
    void setTransferFunctions(const std::vector<vvTransFunc> &tf);

    void mapTransferFunctionFrom01(int chan = 0);
    void mapTransferFunctionsFrom01();

    void setShader(int num);
    int getShader() const;

    enum BlendMode
    {
        AlphaBlend,
        MinimumIntensity,
        MaximumIntensity
    };

    void setBlendMode(BlendMode mode);
    BlendMode getBlendMode() const;

    int getMaxClipPlanes() const;

private:
    virtual osg::Object *cloneType() const
    {
        return new VolumeDrawable();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new VolumeDrawable(*this, copyop);
    }

    VolumeDrawable(const VolumeDrawable &, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    void init(std::string rendererName = "", std::string voxType = "");

    mutable vvVolDesc *vd;
    struct ContextState
    {
        ContextState();
        ~ContextState();
        vvRenderer *renderer;
        std::vector<vvRenderState::ParameterType> parameterChanges;
        bool applyTF;
    };

    mutable std::vector<ContextState *> contextState;
    mutable int currentFrame;
    virvo::vec3f viewDir;
    virvo::vec3f objDir;
    bool preint;
    bool lighting;
    bool selected;
    virvo::tex_filter_mode interpolation;
    bool culled;
    vvRenderState renderState;
    bool flatDisplay;
    int shader;
    BlendMode blendMode;
    std::string rendererName;
    std::string geoType;
    std::string voxType;
};

} // namespace virvo

#endif
