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


#ifdef WIN32 // might be required for all 64 bit OSes
#define VV_HAVE_LLONG 1
#define VV_HAVE_ULLONG 1
#endif

#include "VolumeDrawable.h"
#include <virvo/vvclipobj.h>
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

namespace virvo {

VolumeDrawable::ContextState::ContextState()
    : renderer(NULL)
    , applyTF(false)
{
}

VolumeDrawable::ContextState::~ContextState()
{
    delete renderer;
}

VolumeDrawable::VolumeDrawable(std::string rendererName, std::string voxType)
{
#ifdef VERBOSE
    cerr << "VolumeDrawable::<init> warn: empty constructor called" << endl;
#endif
    init(rendererName, voxType);
}

VolumeDrawable::VolumeDrawable(const VolumeDrawable &drawable,
                                   const osg::CopyOp &copyop)
    : Drawable(drawable, copyop)
{
#ifdef VERBOSE
    cerr << "VolumeDrawable::<init> copying" << endl;
#endif
    init();
}

void VolumeDrawable::init(std::string rendererName, std::string voxType)
{
    if (geoType.empty())
        geoType = rendererName;

    vd = NULL;
    setSupportsDisplayList(false);
    if (voxType == "rgba")
        setParameter(vvRenderState::VV_POST_CLASSIFICATION, false);
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

void VolumeDrawable::setParameter(const vvRenderState::ParameterType param, const vvParam &newValue)
{
    renderState.setParameter(param, newValue);
    for (size_t i = 0; i < contextState.size(); ++i)
    {
        contextState[i]->parameterChanges.push_back(param);
    }
}

VolumeDrawable::~VolumeDrawable()
{
#ifdef VERBOSE
    cerr << "VolumeDrawable::<dtor>: this=" << this << endl;
#endif
    contextState.clear();
    delete vd;
}

void VolumeDrawable::drawImplementation(RenderInfo &renderInfo) const
{
    vvDebugMsg::msg(3, "VolumeDrawable::drawImplementation()");

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

void VolumeDrawable::setCurrentFrame(int frame)
{
    currentFrame = frame;
}

int VolumeDrawable::getCurrentFrame() const
{
    return currentFrame;
}

int VolumeDrawable::getNumFrames() const
{
    if (vd)
    {
        return vd->frames;
    }

    return 0;
}

void VolumeDrawable::setViewDirection(const osg::Vec3 &dir)
{
    viewDir = virvo::vec3f(dir[0], dir[1], dir[2]);
}

void VolumeDrawable::setObjectDirection(const osg::Vec3 &dir)
{
    objDir = virvo::vec3f(dir[0], dir[1], dir[2]);
}

void VolumeDrawable::setClipDirection(const osg::Vec3 &dir)
{
    setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL, virvo::vec3f(-dir[0], -dir[1], -dir[2]));
}

void VolumeDrawable::setClipPoint(const osg::Vec3 &point)
{
    setParameter(vvRenderState::VV_CLIP_PLANE_POINT, virvo::vec3f(point[0], point[1], point[2]));
}

void VolumeDrawable::setClipping(bool enable)
{
    setParameter(vvRenderState::VV_CLIP_MODE, enable);
}

bool VolumeDrawable::getClipping() const
{
    return renderState.getParameter(vvRenderState::VV_CLIP_MODE);
}

void VolumeDrawable::setSingleSliceClipping(bool enable)
{
    setParameter(vvRenderState::VV_CLIP_SINGLE_SLICE, enable);
}

void VolumeDrawable::setROIPosition(const osg::Vec3 &pos)
{
#ifdef VERBOSE
    cerr << "ROI pos: " << pos[0] << " " << pos[1] << " " << pos[2] << endl;
#endif
    setParameter(vvRenderState::VV_ROI_POS, virvo::vec3f(pos[0], pos[1], pos[2]));
}

void VolumeDrawable::setROISize(float size)
{
#ifdef VERBOSE
    cerr << "ROI size: " << size << endl;
#endif
    setParameter(vvRenderState::VV_ROI_SIZE, virvo::vec3f(size, size, size));
    setParameter(vvRenderState::VV_IS_ROI_USED, (size > 0.f));
}

float VolumeDrawable::getROISize() const
{
    return renderState.getParameter(vvRenderState::VV_ROI_SIZE).asVec3f()[0];
}

osg::Vec3 VolumeDrawable::getROIPosition() const
{
    virvo::vec3f roiPos = renderState.getParameter(vvRenderState::VV_ROI_POS);
    return osg::Vec3(roiPos[0], roiPos[1], roiPos[2]);
}

void VolumeDrawable::setChannelWeights(std::vector<float> const& cw)
{
    vd->channelWeights = cw;
}

void VolumeDrawable::setUseChannelWeights(bool enable)
{
    setParameter(vvRenderState::VV_CHANNEL_WEIGHTS, enable);
}

bool VolumeDrawable::getUseChannelWeights() const
{
    return renderState.getParameter(vvRenderState::VV_CHANNEL_WEIGHTS);
}

osg::Vec3 VolumeDrawable::getPosition() const
{
    if (vd)
    {
        return osg::Vec3(vd->pos[0], vd->pos[1], vd->pos[2]);
    }
    else
        return osg::Vec3(0., 0., 0.);
}

void VolumeDrawable::setPosition(const osg::Vec3 &pos)
{
    if (vd)
    {
        vd->pos = virvo::vec3f(pos[0], pos[1], pos[2]);
    }
}

void VolumeDrawable::getBoundingBox(osg::Vec3 *min, osg::Vec3 *max) const
{
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    const BoundingBox &bb = getInitialBound();
#else
    const BoundingBox &bb = getBound();
#endif
    *min = bb._min;
    *max = bb._max;
}

vvVolDesc *VolumeDrawable::getVolumeDescription() const
{
    return vd;
}

void VolumeDrawable::setVolumeDescription(vvVolDesc *v)
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
        vd->tf[0].setDefaultColors(vd->getChan() == 1 ? 0 : 3, 0., 1.);
        vd->tf[0].setDefaultAlpha(0, 0., 1.);
    }

    dirtyBound();

    if (vd)
    {
        osg::Vec3 offset(vd->vox[0] * vd->getDist()[0] * vd->_scale * 0.5f,
                vd->vox[1] * vd->getDist()[1] * vd->_scale * 0.5f,
                vd->vox[2] * vd->getDist()[2] * vd->_scale * 0.5f);
        setInitialBound(BoundingBox(getPosition() - offset, getPosition() + offset));
    }
    else
    {
        setInitialBound(BoundingBox(osg::Vec3(0., 0., 0.), osg::Vec3(0., 0., 0.)));
    }
}

bool VolumeDrawable::getInstantMode() const
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

void VolumeDrawable::setPreintegration(bool val)
{
    preint = val;
    setParameter(vvRenderer::VV_PREINT, preint);
}

void VolumeDrawable::setLighting(bool val)
{
    lighting = val;
    setParameter(vvRenderer::VV_LIGHTING, lighting);
}

void VolumeDrawable::setROISelected(bool val)
{
    selected = val;
    vvColor col(1.f, 1.f, 1.f);
    if (selected)
        col = vvColor(1.f, 0.f, 0.f);
    setParameter(vvRenderState::VV_PROBE_COLOR, col);
}

void VolumeDrawable::setInterpolation(bool val)
{
    interpolation = val ? virvo::Linear : virvo::Nearest;
    setParameter(vvRenderer::VV_SLICEINT, interpolation);
    //setParameter(vvRenderer::VV_WARPINT, val ? 1.0f : 0.0f);
}

bool VolumeDrawable::getInterpolation() const
{
    int i = renderState.getParameter(vvRenderState::VV_SLICEINT).asInt();
    return static_cast<virvo::tex_filter_mode>(i) == Linear;
}

void VolumeDrawable::setBoundaries(bool val)
{
    setParameter(vvRenderState::VV_BOUNDARIES, val);
}

void VolumeDrawable::setBoundariesActive(bool val)
{
    vvColor col(1.f, 1.f, 1.f);
    if (val)
        col = vvColor(1.f, 1.f, 0.f);
    setParameter(vvRenderState::VV_BOUND_COLOR, col);
}

void VolumeDrawable::setQuality(float q)
{
    setParameter(vvRenderState::VV_QUALITY, q);
}

float VolumeDrawable::getQuality() const
{
    return renderState.getParameter(vvRenderState::VV_QUALITY);
}

bool VolumeDrawable::have3DTextures() const
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

void VolumeDrawable::setTransferFunctions(const std::vector<vvTransFunc> &tf)
{
    if (vd)
    {
        vd->tf = tf;
    }

    for (unsigned int i = 0; i < contextState.size(); i++)
    {
        contextState[i]->applyTF = true;
    }
}

void VolumeDrawable::setTransferFunction(const vvTransFunc &tf, int chan)
{
    if (vd)
    {
        if (static_cast<int>(vd->tf.size()) <= chan)
            vd->tf.resize(chan + 1);
        vd->tf[chan] = tf;
    }

    for (unsigned int i = 0; i < contextState.size(); i++)
    {
        contextState[i]->applyTF = true;
    }
}

void VolumeDrawable::mapTransferFunctionFrom01(int chan)
{
    if (vd)
    {
        typedef std::vector<vvTFWidget *> Widgets;
        for (Widgets::iterator it = vd->tf[chan]._widgets.begin();
             it != vd->tf[chan]._widgets.end();
             ++it)
        {
            vvTFWidget *w = *it;
            w->mapFrom01(vd->range(chan)[0], vd->range(chan)[1]);
        }
    }
}

void VolumeDrawable::mapTransferFunctionsFrom01()
{
    if (vd)
    {
        typedef std::vector<vvTFWidget *> Widgets;

        for (size_t chan = 0; chan < vd->tf.size(); ++chan)
        {
            for (Widgets::iterator it = vd->tf[chan]._widgets.begin();
                 it != vd->tf[chan]._widgets.end();
                 ++it)
            {
                vvTFWidget *w = *it;
                w->mapFrom01(vd->range(chan)[0], vd->range(chan)[1]);
            }
        }
    }
}

void VolumeDrawable::enableFlatDisplay(bool enable)
{
    flatDisplay = enable;
    setParameter(vvTexRend::VV_SLICEORIENT, (int)flatDisplay);
}

void VolumeDrawable::setShader(int num)
{
    shader = num;
    setParameter(vvRenderState::VV_PIX_SHADER, shader);
}

int VolumeDrawable::getShader() const
{
    return shader;
}

void VolumeDrawable::setBlendMode(BlendMode mode)
{
    blendMode = mode;
    if (blendMode == AlphaBlend)
        setParameter(vvRenderState::VV_MIP_MODE, 0);
    else if (blendMode == MaximumIntensity)
        setParameter(vvRenderState::VV_MIP_MODE, 1);
    else if (blendMode == MinimumIntensity)
        setParameter(vvRenderState::VV_MIP_MODE, 2);
}

VolumeDrawable::BlendMode VolumeDrawable::getBlendMode() const
{
    return blendMode;
}

int VolumeDrawable::getMaxClipPlanes() const
{
    int result = 0;

    if (contextState.size() > 0)
    {
        vvRenderer *&renderer = contextState[0]->renderer;

        boost::shared_ptr<vvClipPlane> plane = vvClipPlane::create();

        typedef vvRenderState::ParameterType PT;

        for (PT i = vvRenderState::VV_CLIP_OBJ0;
                i != vvRenderState::VV_CLIP_OBJ_LAST;
                i = PT(i + 1))
        {
            if (renderer->checkParameter(i, plane))
            {
                ++result;
            }
        }
    }

    return result;
}

} // namespace virvo
