/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_VOLUME_DRAWABLE
#define CO_VOLUME_DRAWABLE

#include <osg/Drawable>
#include <osg/Geode>
#include <virvo/math/math.h>
#include <virvo/texture/forward.h>
#include <virvo/vvvecmath.h>
#include <virvo/vvrenderer.h>

namespace osg
{
class State;
};

class vvVolDesc;
class vvTransFunc;

class coVolumeDrawable : public osg::Drawable
{
public:
    coVolumeDrawable();
    virtual ~coVolumeDrawable();

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

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
    void setBoundaries(bool value);
    void setBoundariesActive(bool value);
    void setQuality(float quality);
    float getQuality() const;
    bool wasCulled() const;
    void enableFlatDisplay(bool enable);

    void setTransferFunction(const vvTransFunc &tf, int chan = 0);
    void setTransferFunctions(const std::vector<vvTransFunc> &tf);

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

private:
    virtual osg::Object *cloneType() const
    {
        return new coVolumeDrawable();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new coVolumeDrawable(*this, copyop);
    }

    coVolumeDrawable(const coVolumeDrawable &, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    void init();

    void setParameter(vvRenderState::ParameterType param, const vvParam &newValue);

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
#endif
