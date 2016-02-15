/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32 // might be required for all 64 bit OSes
#define VV_HAVE_LLONG 1
#define VV_HAVE_ULLONG 1
#endif

#include "coVolumeDrawable.h"
#include <virvo/vvrendererfactory.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvtexrend.h>

#include <iostream>
#include <osg/Version>

using namespace osg;
using namespace std;

#undef VERBOSE

coVolumeDrawable::ContextState::ContextState()
    : renderer(NULL)
    , applyTF(false)
{
}

coVolumeDrawable::ContextState::~ContextState()
{
    delete renderer;
}

coVolumeDrawable::coVolumeDrawable(std::string rendererName, std::string voxType)
{
#ifdef VERBOSE
    cerr << "coVolumeDrawable::<init> warn: empty constructor called" << endl;
#endif
    init(rendererName, voxType);
}

coVolumeDrawable::coVolumeDrawable(const coVolumeDrawable &drawable,
                                   const osg::CopyOp &copyop)
    : Drawable(drawable, copyop)
{
#ifdef VERBOSE
    cerr << "coVolumeDrawable::<init> copying" << endl;
#endif
    init();
}

void coVolumeDrawable::init(std::string rendererName, std::string voxType)
{
    if (geoType.empty())
        geoType = rendererName;

    vd = NULL;
    setSupportsDisplayList(false);
    setParameter(vvRenderState::VV_BOUNDARIES, false);
    preint = false;
    lighting = false;
    interpolation = virvo::Linear;
    selected = false;
    flatDisplay = false;
    blendMode = AlphaBlend;

    setParameter(vvRenderState::VV_ROI_POS, virvo::vec3f(0., 0., 0.));
    setParameter(vvRenderState::VV_ROI_SIZE, virvo::vec3f(0., 0., 0.));
    setParameter(vvRenderState::VV_IS_ROI_USED, false);
    setParameter(vvRenderState::VV_MIP_MODE, 0);

    shader = 0;
    currentFrame = 0;
}

void coVolumeDrawable::setParameter(const vvRenderState::ParameterType param, const vvParam &newValue)
{
    renderState.setParameter(param, newValue);
    for (size_t i = 0; i < contextState.size(); ++i)
    {
        contextState[i]->parameterChanges.push_back(param);
    }
}

coVolumeDrawable::~coVolumeDrawable()
{
#ifdef VERBOSE
    cerr << "coVolumeDrawable::<dtor>: this=" << this << endl;
#endif
    contextState.clear();
    delete vd;
}

void coVolumeDrawable::drawImplementation(RenderInfo &renderInfo) const
{
    vvDebugMsg::msg(3, "coVolumeDrawable::drawImplementation()");

    const unsigned ctx = renderInfo.getState()->getContextID();
    while (ctx >= contextState.size())
    {
        // this will delete the old renderer contextState.resize(ctx+1);
        ContextState *nc = new ContextState;
        contextState.push_back(nc);
    }
    vvRenderer *&renderer = contextState[ctx]->renderer;

    if (vd && !renderer)
    {
        renderer = vvRendererFactory::create(vd, renderState, geoType.c_str(), voxType.c_str());
    }

    // if a renderer exists, process regular rendering procedure
    if (renderer)
    {
        for (std::vector<vvRenderState::ParameterType>::iterator it = contextState[ctx]->parameterChanges.begin();
             it != contextState[ctx]->parameterChanges.end();
             ++it)
        {
            renderer->setParameter(*it, renderState.getParameter(*it));
        }
        contextState[ctx]->parameterChanges.clear();
        if (contextState[ctx]->applyTF)
        {
            renderer->updateTransferFunction();
            contextState[ctx]->applyTF = false;
        }

        ref_ptr<StateSet> currentState = new StateSet;
        renderInfo.getState()->captureCurrentState(*currentState);
        renderInfo.getState()->pushStateSet(currentState.get());

        renderer->setCurrentFrame(currentFrame);
        //renderer->setQuality(quality);
        renderer->setViewingDirection(viewDir);
        renderer->setObjectDirection(objDir);

        renderer->renderFrame();

        renderInfo.getState()->popStateSet();
    }
    else
    {
        //cerr << "vd==NULL" << endl;
    }
}

void coVolumeDrawable::setCurrentFrame(int frame)
{
    currentFrame = frame;
}

int coVolumeDrawable::getCurrentFrame() const
{
    return currentFrame;
}

int coVolumeDrawable::getNumFrames() const
{
    if (vd)
    {
        return vd->frames;
    }

    return 0;
}

void coVolumeDrawable::setViewDirection(const osg::Vec3 &dir)
{
    viewDir = virvo::vec3f(dir[0], dir[1], dir[2]);
}

void coVolumeDrawable::setObjectDirection(const osg::Vec3 &dir)
{
    objDir = virvo::vec3f(dir[0], dir[1], dir[2]);
}

void coVolumeDrawable::setClipDirection(const osg::Vec3 &dir)
{
    setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL, virvo::vec3f(-dir[0], -dir[1], -dir[2]));
}

void coVolumeDrawable::setClipPoint(const osg::Vec3 &point)
{
    setParameter(vvRenderState::VV_CLIP_PLANE_POINT, virvo::vec3f(point[0], point[1], point[2]));
}

void coVolumeDrawable::setClipping(bool enable)
{
    setParameter(vvRenderState::VV_CLIP_MODE, static_cast<unsigned int>(enable));
}

bool coVolumeDrawable::getClipping() const
{
    return renderState.getParameter(vvRenderState::VV_CLIP_MODE);
}

void coVolumeDrawable::setSingleSliceClipping(bool enable)
{
    setParameter(vvRenderState::VV_CLIP_SINGLE_SLICE, enable);
}

void coVolumeDrawable::setROIPosition(const osg::Vec3 &pos)
{
#ifdef VERBOSE
    cerr << "ROI pos: " << pos[0] << " " << pos[1] << " " << pos[2] << endl;
#endif
    setParameter(vvRenderState::VV_ROI_POS, virvo::vec3f(pos[0], pos[1], pos[2]));
}

void coVolumeDrawable::setROISize(float size)
{
#ifdef VERBOSE
    cerr << "ROI size: " << size << endl;
#endif
    setParameter(vvRenderState::VV_ROI_SIZE, virvo::vec3f(size, size, size));
    setParameter(vvRenderState::VV_IS_ROI_USED, (size > 0.f));
}

float coVolumeDrawable::getROISize() const
{
    return renderState.getParameter(vvRenderState::VV_ROI_SIZE).asVec3f()[0];
}

osg::Vec3 coVolumeDrawable::getROIPosition() const
{
    virvo::vec3f roiPos = renderState.getParameter(vvRenderState::VV_ROI_POS);
    return osg::Vec3(roiPos[0], roiPos[1], roiPos[2]);
}

osg::Vec3 coVolumeDrawable::getPosition() const
{
    if (vd)
    {
        return Vec3(vd->pos[0], vd->pos[1], vd->pos[2]);
    }
    else
        return Vec3(0., 0., 0.);
}

void coVolumeDrawable::setPosition(const osg::Vec3 &pos)
{
    if (vd)
    {
        vd->pos = virvo::vec3f(pos[0], pos[1], pos[2]);
    }
}

void coVolumeDrawable::getBoundingBox(osg::Vec3 *min, osg::Vec3 *max) const
{
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    const BoundingBox &bb = computeBoundingBox();
#else
    const BoundingBox &bb = getBound();
#endif
    *min = bb._min;
    *max = bb._max;
}

vvVolDesc *coVolumeDrawable::getVolumeDescription() const
{
    return vd;
}

void coVolumeDrawable::setVolumeDescription(vvVolDesc *v)
{
    for (unsigned int i = 0; i < contextState.size(); i++)
        delete contextState[i];
    contextState.clear();

    if (vd != v)
    {
        delete vd;
    }

#ifdef VERBOSE
    fprintf(stderr, "setVolumeDescription: voldesc = %p\n", v);
#endif

    vd = v;

    if (vd && vd->tf[0].isEmpty())
    {
        vd->tf[0].setDefaultColors(vd->chan == 1 ? 0 : 3, 0., 1.);
        vd->tf[0].setDefaultAlpha(0, 0., 1.);
    }

    dirtyBound();

    if (vd)
    {
        osg::Vec3 diag = osg::Vec3(vd->vox[0] * vd->dist[0], vd->vox[1] * vd->dist[2], vd->vox[2] * vd->dist[2]);
        setInitialBound(BoundingBox(getPosition() - diag * .5, getPosition() + diag * .5));
    }
    else
    {
        setInitialBound(BoundingBox(Vec3(0., 0., 0.), Vec3(0., 0., 0.)));
    }
}

bool coVolumeDrawable::getInstantMode() const
{
    if (contextState.size() > 0)
    {
        vvRenderer *&renderer = contextState[0]->renderer;
        if (renderer)
        {
            return renderer->instantClassification();
        }
        else
        {
#ifdef VERBOSE
            fprintf(stderr, "instant false\n");
#endif
            return false;
        }
    }
    else
    {
        return true;
    }
}

void coVolumeDrawable::setPreintegration(bool val)
{
    preint = val;
    setParameter(vvRenderer::VV_PREINT, preint);
}

void coVolumeDrawable::setLighting(bool val)
{
    lighting = val;
    setParameter(vvRenderer::VV_LIGHTING, lighting);
}

void coVolumeDrawable::setROISelected(bool val)
{
    selected = val;
    vvColor col(1.f, 1.f, 1.f);
    if (selected)
        col = vvColor(1.f, 0.f, 0.f);
    setParameter(vvRenderState::VV_PROBE_COLOR, col);
}

void coVolumeDrawable::setInterpolation(bool val)
{
    interpolation = val ? virvo::Linear : virvo::Nearest;
    setParameter(vvRenderer::VV_SLICEINT, interpolation);
    //setParameter(vvRenderer::VV_WARPINT, val ? 1.0f : 0.0f);
}

void coVolumeDrawable::setBoundaries(bool val)
{
    setParameter(vvRenderState::VV_BOUNDARIES, val);
}

void coVolumeDrawable::setBoundariesActive(bool val)
{
    vvColor col(1.f, 1.f, 1.f);
    if (val)
        col = vvColor(1.f, 1.f, 0.f);
    setParameter(vvRenderState::VV_BOUND_COLOR, col);
}

void coVolumeDrawable::setQuality(float q)
{
    setParameter(vvRenderState::VV_QUALITY, q);
}

float coVolumeDrawable::getQuality() const
{
    return renderState.getParameter(vvRenderState::VV_QUALITY);
}

bool coVolumeDrawable::have3DTextures() const
{
    if (contextState.size() > 0)
    {
        vvRenderer *&renderer = contextState[0]->renderer;
        vvTexRend *texrend = dynamic_cast<vvTexRend *>(renderer);
        if (texrend)
        {
            return true;
        }
    }

    return false;
}

bool coVolumeDrawable::wasCulled() const
{
    return false;
}

void coVolumeDrawable::setTransferFunctions(const std::vector<vvTransFunc> &tf)
{
    if (vd)
    {
        vd->tf = tf;
        typedef std::vector<vvTFWidget *> Widgets;

        for (size_t chan = 0; chan < vd->tf.size(); ++chan)
        {
            for (Widgets::iterator it = vd->tf[chan]._widgets.begin();
                 it != vd->tf[chan]._widgets.end();
                 ++it)
            {
                vvTFWidget *w = *it;
                w->mapFrom01(vd->real[chan][0], vd->real[chan][1]);
            }
        }
    }

    for (unsigned int i = 0; i < contextState.size(); i++)
    {
        contextState[i]->applyTF = true;
    }
}

void coVolumeDrawable::setTransferFunction(const vvTransFunc &tf, int chan)
{
    if (vd)
    {
        if (vd->tf.size() <= chan)
            vd->tf.resize(chan + 1);
        vd->tf[chan] = tf;
        typedef std::vector<vvTFWidget *> Widgets;
        for (Widgets::iterator it = vd->tf[chan]._widgets.begin();
             it != vd->tf[chan]._widgets.end();
             ++it)
        {
            vvTFWidget *w = *it;
            w->mapFrom01(vd->real[chan][0], vd->real[chan][1]);
        }
    }

    for (unsigned int i = 0; i < contextState.size(); i++)
    {
        contextState[i]->applyTF = true;
    }
}

void coVolumeDrawable::enableFlatDisplay(bool enable)
{
    flatDisplay = enable;
    setParameter(vvTexRend::VV_SLICEORIENT, (int)flatDisplay);
}

void coVolumeDrawable::setShader(int num)
{
    shader = num;
    setParameter(vvRenderState::VV_PIX_SHADER, shader);
}

int coVolumeDrawable::getShader() const
{
    return shader;
}

void coVolumeDrawable::setBlendMode(BlendMode mode)
{
    blendMode = mode;
    if (blendMode == AlphaBlend)
        setParameter(vvRenderState::VV_MIP_MODE, 0);
    else if (blendMode == MaximumIntensity)
        setParameter(vvRenderState::VV_MIP_MODE, 1);
    else if (blendMode == MinimumIntensity)
        setParameter(vvRenderState::VV_MIP_MODE, 2);
}

coVolumeDrawable::BlendMode coVolumeDrawable::getBlendMode() const
{
    return blendMode;
}
