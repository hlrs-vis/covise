/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2006 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <sstream>
#include <iomanip>
#include <stdio.h>

#include <osg/io_utils>

#include <osg/MatrixTransform>

#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>

#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osg/Geometry>
#include "coVRStatsDisplay.h"
#include "coVRFileManager.h"
#include <config/CoviseConfig.h>
#include <osg/Version>

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
#define getBound getBoundingBox
#endif

using namespace opencover;

namespace {

enum Accum {
    AccumMin,
    AccumMax,
    AccumJitter,
    AccumMean,
    AccumMedian,
    AccumAverage,
    AccumAverageInverse,
    AccumNewest, // keep last before the print-only values
    // the following values are just for displaying in AveragedValueTextDrawCallback
    AccumMeanJitter,
    AccumMedianJitter,
    AccumAverageJitter,
    AccumMinMax,
};

bool getAttribute(osg::Stats *stats, unsigned int startFrameNumber, unsigned int endFrameNumber, const std::string& attributeName, Accum accum, double& value)
{
    if (endFrameNumber<startFrameNumber)
    {
        std::swap(endFrameNumber, startFrameNumber);
    }

    value = 0.0;
    double vmin = std::numeric_limits<double>::max();
    double vmax = std::numeric_limits<double>::lowest();

    std::vector<double> vals;

    int numFound = 0;
    for(unsigned int i = startFrameNumber; i<=endFrameNumber; ++i)
    {
        double v = 0.0;
        if (!stats->getAttribute(i,attributeName, v))
            continue;

        ++numFound;
        if (v < vmin)
            vmin = v;
        if (v > vmax)
            vmax = v;

        switch (accum) {
        case AccumMeanJitter:
        case AccumMedianJitter:
        case AccumAverageJitter:
        case AccumMinMax:
            assert("invalid for AccumMeanJitter and AccumMedianJitter" == 0);
            break;
        case AccumAverage:
            value += v;
            break;
        case AccumAverageInverse:
            value += 1./v;
            break;
        case AccumNewest:
            value = v;
            break;
        case AccumMedian:
            vals.push_back(v);
            break;
        default:
            break;
        }
    }

    if (numFound == 0)
        return false;

    switch (accum) {
    case AccumAverageInverse:
        value = numFound/value;
        break;
    case AccumAverage:
        value /= numFound;
        break;
    case AccumMin:
        value = vmin;
        break;
    case AccumMax:
        value = vmax;
        break;
    case AccumJitter:
        value = (vmax-vmin)*0.5;
        break;
    case AccumMean:
        value = (vmin+vmax)*0.5;
        break;
    case AccumMedian:
        assert(numFound == vals.size());
        std::sort(vals.begin(), vals.end());
        value = vals[vals.size()/2];
        break;
    default:
        break;
    }

    return true;
}

bool getAttribute(osg::Stats *stats, const std::string& attributeName, Accum accum, double& value)
{
    return getAttribute(stats, stats->getEarliestFrameNumber(), stats->getLatestFrameNumber(), attributeName, accum, value);
}

int getSlowestFrame(osg::Stats *stats, int frameNumber)
{
    static int lastFrame = -1;
    static int cachedSlowest = -1;

    if (lastFrame == frameNumber)
        return cachedSlowest;

    cachedSlowest = -1;

    int start = stats->getEarliestFrameNumber(), last = stats->getLatestFrameNumber();
    double duration = -1.;
    for (int f=start; f<last; ++f)
    {
        double d = 0.;
        if (!stats->getAttribute(f, "Frame duration", d))
            continue;
        if (d > duration)
        {
            duration = d;
            cachedSlowest = f;
        }
    }

    return cachedSlowest;
}

} // anonymous namespace

coVRStatsDisplay::coVRStatsDisplay()
    : _statsType(NO_STATS)
    , _initialized(false)
    , _threadingModel(osgViewer::ViewerBase::SingleThreaded)
    , _frameRateChildNum(0)
    , _threadingModelChildNum(0)
    , _gpuMemChildNum(0)
    , _gpuPCIeChildNum(0)
    , _gpuClockChildNum(0)
    , _gpuUtilChildNum(0)
    , _rhrFpsChildNum(0)
    , _rhrBandwidthChildNum(0)
    , _rhrSkippedChildNum(0)
    , _viewerChildNum(0)
    , _gpuChildNum(0)
    , _cameraSceneChildNum(0)
    , _viewerSceneChildNum(0)
    , _numBlocks(8)
    , _blockMultiplier(10000.0)
    , _statsWidth(1280.0f)
    , _statsHeight(1024.0f)
{
    _camera = new osg::Camera;
    _camera->setName("Statistics");
    _camera->setRenderer(new osgViewer::Renderer(_camera.get()));
    _camera->setProjectionResizePolicy(osg::Camera::FIXED);
}

void coVRStatsDisplay::showStats(int whichStats, osgViewer::ViewerBase *viewer)
{

    if (viewer && _threadingModelText.valid())
    {
        updateThreadingModelText(viewer->getThreadingModel());
    }

    if (!_initialized)
    {
        setUpHUDCamera(viewer);
        setUpScene(viewer);
    }
    _statsType = whichStats;
    if (_statsType == LAST)
        _statsType = NO_STATS;
    osgViewer::ViewerBase::Cameras cameras;
    viewer->getCameras(cameras);

    viewer->getViewerStats()->collectStats("frame_rate", false);
    viewer->getViewerStats()->collectStats("isect", false);
    viewer->getViewerStats()->collectStats("plugin", false);
    viewer->getViewerStats()->collectStats("opencover", false);
    viewer->getViewerStats()->collectStats("sync", false);
    viewer->getViewerStats()->collectStats("swap", false);
    viewer->getViewerStats()->collectStats("finish", false);
    viewer->getViewerStats()->collectStats("update", false);

    for (osgViewer::ViewerBase::Cameras::iterator itr = cameras.begin();
         itr != cameras.end();
         ++itr)
    {
        osg::Stats *stats = (*itr)->getStats();
        if (stats)
        {
            stats->collectStats("rendering", false);
            stats->collectStats("gpu", false);
            stats->collectStats("scene", false);
        }
    }

    viewer->getViewerStats()->collectStats("scene", false);

    _camera->setNodeMask(0x0);
    _switch->setAllChildrenOff();
    switch (_statsType)
    {
    case (NO_STATS):
    {
        break;
    }
    case (VIEWER_SCENE_STATS):
    {
        _camera->setNodeMask(0xffffffff);
        _switch->setValue(_viewerSceneChildNum, true);

        viewer->getViewerStats()->collectStats("scene", true);
    }
    case (CAMERA_SCENE_STATS):
    {
        _camera->setNodeMask(0xffffffff);
        _switch->setValue(_cameraSceneChildNum, true);

        for (osgViewer::ViewerBase::Cameras::iterator itr = cameras.begin();
             itr != cameras.end();
             ++itr)
        {
            osg::Stats *stats = (*itr)->getStats();
            if (stats)
            {
                stats->collectStats("scene", true);
            }
        }
    }
    case (VIEWER_STATS):
    {
        osgViewer::ViewerBase::Scenes scenes;
        viewer->getScenes(scenes);
        for (osgViewer::ViewerBase::Scenes::iterator itr = scenes.begin();
             itr != scenes.end();
             ++itr)
        {
            osgViewer::Scene *scene = *itr;
            osgDB::DatabasePager *dp = scene->getDatabasePager();
            if (dp && dp->isRunning())
            {
                dp->resetStats();
            }
        }

        viewer->getViewerStats()->collectStats("isect", true);
        viewer->getViewerStats()->collectStats("plugin", true);
        viewer->getViewerStats()->collectStats("opencover", true);
        viewer->getViewerStats()->collectStats("sync", true);
        viewer->getViewerStats()->collectStats("swap", true);
        viewer->getViewerStats()->collectStats("finish", true);
        viewer->getViewerStats()->collectStats("update", true);

        for (osgViewer::ViewerBase::Cameras::iterator itr = cameras.begin();
             itr != cameras.end();
             ++itr)
        {
            if ((*itr)->getStats())
                (*itr)->getStats()->collectStats("rendering", true);
            if ((*itr)->getStats())
                (*itr)->getStats()->collectStats("gpu", true);
        }

        _camera->setNodeMask(0xffffffff);
        _switch->setValue(_viewerChildNum, true);
        _switch->setValue(_gpuChildNum, _statsType==VIEWER_STATS && _gpuStats);
        _switch->setValue(_threadingModelChildNum, true);
    }
    case (FRAME_RATE):
    {
        viewer->getViewerStats()->collectStats("frame_rate", true);

        _camera->setNodeMask(0xffffffff);
        _switch->setValue(_frameRateChildNum, true);

        auto stats = viewer->getViewerStats();
        unsigned int first = stats->getEarliestFrameNumber(), last = stats->getLatestFrameNumber();
        double dummy;
        if (_gpuStats)
        {
            _switch->setValue(_gpuUtilChildNum, true);
            _switch->setValue(_gpuPCIeChildNum, true);
            _switch->setValue(_gpuClockChildNum, true);
            _switch->setValue(_gpuMemChildNum, true);
        }
        if (_rhrStats)
        {
            _switch->setValue(_rhrFpsChildNum, true);
            _switch->setValue(_rhrDelayChildNum, true);
            _switch->setValue(_rhrBandwidthChildNum, true);
            _switch->setValue(_rhrSkippedChildNum, true);
            _switch->setValue(_threadingModelChildNum, true);
        }
    }
    default:
        break;
    }
}

void coVRStatsDisplay::enableGpuStats(bool enable, const std::string &devname)
{
    _gpuStats = enable;
    _gpuName = devname;
}

void coVRStatsDisplay::enableRhrStats(bool enable)
{
    _rhrStats = enable;
}

void coVRStatsDisplay::enableFinishStats(bool enable)
{
    _finishStats = enable;
}

void coVRStatsDisplay::enableSyncStats(bool enable)
{
    _syncStats = enable;
}

void coVRStatsDisplay::updateThreadingModelText(osgViewer::ViewerBase::ThreadingModel tm)
{
    _threadingModel = tm;

    if (!_threadingModelText.valid())
        return;

    switch (_threadingModel)
    {
    case (osgViewer::Viewer::SingleThreaded):
        _threadingModelText->setText("ThreadingModel: SingleThreaded", osgText::String::ENCODING_UTF8);
        break;
    case (osgViewer::Viewer::CullDrawThreadPerContext):
        _threadingModelText->setText("ThreadingModel: CullDrawThreadPerContext", osgText::String::ENCODING_UTF8);
        break;
    case (osgViewer::Viewer::DrawThreadPerContext):
        _threadingModelText->setText("ThreadingModel: DrawThreadPerContext", osgText::String::ENCODING_UTF8);
        break;
    case (osgViewer::Viewer::CullThreadPerCameraDrawThreadPerContext):
        _threadingModelText->setText("ThreadingModel: CullThreadPerCameraDrawThreadPerContext", osgText::String::ENCODING_UTF8);
        break;
    case (osgViewer::Viewer::AutomaticSelection):
        _threadingModelText->setText("ThreadingModel: AutomaticSelection", osgText::String::ENCODING_UTF8);
        break;
    default:
        _threadingModelText->setText("ThreadingModel: unknown", osgText::String::ENCODING_UTF8);
        break;
    }
}

void coVRStatsDisplay::reset()
{
    _initialized = false;
    _camera->setGraphicsContext(0);
    _camera->removeChildren(0, _camera->getNumChildren());
}

void coVRStatsDisplay::setUpHUDCamera(osgViewer::ViewerBase *viewer)
{
    osgViewer::GraphicsWindow *window = dynamic_cast<osgViewer::GraphicsWindow *>(_camera->getGraphicsContext());

    if (!window)
    {
        osgViewer::Viewer::Windows windows;
        viewer->getWindows(windows);

        if (windows.empty())
            return;

        window = windows.front();
    }

    _camera->setGraphicsContext(window);

    _camera->setViewport(0, 0, window->getTraits()->width, window->getTraits()->height);
    _camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);

    _camera->setRenderOrder(osg::Camera::POST_RENDER, 10);

    _camera->setProjectionMatrix(osg::Matrix::ortho2D(0.0, _statsWidth, 0.0, _statsHeight));
    _camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    _camera->setViewMatrix(osg::Matrix::identity());

    // only clear the depth buffer
    _camera->setClearMask(0);

    _camera->setRenderer(new osgViewer::Renderer(_camera.get()));

    _initialized = true;
}

// Drawcallback to draw averaged attribute
struct AveragedValueTextDrawCallback : public virtual osg::Drawable::DrawCallback
{
    AveragedValueTextDrawCallback(osg::Stats *stats, const std::string &name, Accum accum=AccumAverage, double multiplier = 1.0)
        : _stats(stats)
        , _attributeName(name)
        , _accum(accum)
        , _multiplier(multiplier)
        , _tickLastUpdated(0)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        osgText::Text *text = (osgText::Text *)drawable;

        osg::Timer_t tick = osg::Timer::instance()->tick();
        double delta = osg::Timer::instance()->delta_m(_tickLastUpdated, tick);

        if (_accum > AccumNewest)
        {
            _tickLastUpdated = tick;
            Accum first = AccumMean, second = AccumJitter;
            switch (_accum) {
            case AccumMeanJitter:
                first = AccumMean;
                second = AccumJitter;
                break;
            case AccumMedianJitter:
                first = AccumMedian;
                second = AccumJitter;
                break;
            case AccumAverageJitter:
                first = AccumAverage;
                second = AccumJitter;
                break;
            case AccumMinMax:
                first = AccumMin;
                second = AccumMax;
                break;
            default:
                break;
            }

            double val1, val2;
            if (getAttribute(_stats, _attributeName, first, val1)
                && getAttribute(_stats, _attributeName, second, val2))
            {
                sprintf(_tmpText, "%3.1f%s%3.1f", val1 * _multiplier, _accum==AccumMinMax ? "-" : "±", val2 * _multiplier);
                text->setText(_tmpText, osgText::String::ENCODING_UTF8);
            }
            else
            {
                text->setText("", osgText::String::ENCODING_UTF8);
            }

        }
        else if (delta > 50) // update every 50ms
        {
            _tickLastUpdated = tick;
            double value;
            if (getAttribute(_stats, _attributeName, _accum, value))
            {
                sprintf(_tmpText, "%4.2f", value * _multiplier);
                text->setText(_tmpText, osgText::String::ENCODING_UTF8);
            }
            else
            {
                text->setText("", osgText::String::ENCODING_UTF8);
            }
        }
        text->drawImplementation(renderInfo);
    }

    osg::ref_ptr<osg::Stats> _stats;
    std::string _attributeName;
    Accum _accum;
    double _multiplier;
    mutable char _tmpText[128];
    mutable osg::Timer_t _tickLastUpdated;
};

struct CameraSceneStatsTextDrawCallback : public virtual osg::Drawable::DrawCallback
{
    CameraSceneStatsTextDrawCallback(osg::Camera *camera, int cameraNumber)
        : _camera(camera)
        , _tickLastUpdated(0)
        , _cameraNumber(cameraNumber)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        if (!_camera)
            return;

        osgText::Text *text = (osgText::Text *)drawable;

        osg::Timer_t tick = osg::Timer::instance()->tick();
        double delta = osg::Timer::instance()->delta_m(_tickLastUpdated, tick);

        if (delta > 100) // update every 100ms
        {
            _tickLastUpdated = tick;
            std::ostringstream viewStr;
            viewStr.clear();

            osg::Stats *stats = _camera->getStats();
            osgViewer::Renderer *renderer = dynamic_cast<osgViewer::Renderer *>(_camera->getRenderer());

            if (stats && renderer)
            {
                viewStr.setf(std::ios::left, std::ios::adjustfield);
                viewStr.width(14);
                // Used fixed formatting, as scientific will switch to "...e+.." notation for
                // large numbers of vertices/drawables/etc.
                viewStr.setf(std::ios::fixed);
                viewStr.precision(0);

                viewStr << std::setw(1) << "#" << _cameraNumber << std::endl;

                // Camera name
                if (!_camera->getName().empty())
                    viewStr << _camera->getName();
                viewStr << std::endl;

                int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();
                if (!(renderer->getGraphicsThreadDoesCull()))
                {
                    --frameNumber;
                }

#define STATS_ATTRIBUTE(str)                           \
    if (stats->getAttribute(frameNumber, str, value))  \
        viewStr << std::setw(8) << value << std::endl; \
    else                                               \
        viewStr << std::setw(8) << "." << std::endl;

                double value = 0.0;

                STATS_ATTRIBUTE("Visible number of lights")
                STATS_ATTRIBUTE("Visible number of render bins")
                STATS_ATTRIBUTE("Visible depth")
                STATS_ATTRIBUTE("Visible number of materials")
                STATS_ATTRIBUTE("Visible number of impostors")
                STATS_ATTRIBUTE("Visible number of drawables")
                STATS_ATTRIBUTE("Visible vertex count")

                STATS_ATTRIBUTE("Visible number of GL_POINTS")
                STATS_ATTRIBUTE("Visible number of GL_LINES")
                STATS_ATTRIBUTE("Visible number of GL_LINE_STRIP")
                STATS_ATTRIBUTE("Visible number of GL_LINE_LOOP")
                STATS_ATTRIBUTE("Visible number of GL_TRIANGLES")
                STATS_ATTRIBUTE("Visible number of GL_TRIANGLE_STRIP")
                STATS_ATTRIBUTE("Visible number of GL_TRIANGLE_FAN")
                STATS_ATTRIBUTE("Visible number of GL_QUADS")
                STATS_ATTRIBUTE("Visible number of GL_QUAD_STRIP")
                STATS_ATTRIBUTE("Visible number of GL_POLYGON")

                text->setText(viewStr.str(), osgText::String::ENCODING_UTF8);
            }
        }
        text->drawImplementation(renderInfo);
    }

    osg::observer_ptr<osg::Camera> _camera;
    mutable osg::Timer_t _tickLastUpdated;
    int _cameraNumber;
};

struct ViewSceneStatsTextDrawCallback : public virtual osg::Drawable::DrawCallback
{
    ViewSceneStatsTextDrawCallback(osgViewer::View *view, int viewNumber)
        : _view(view)
        , _tickLastUpdated(0)
        , _viewNumber(viewNumber)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        if (!_view)
            return;

        osgText::Text *text = (osgText::Text *)drawable;

        osg::Timer_t tick = osg::Timer::instance()->tick();
        double delta = osg::Timer::instance()->delta_m(_tickLastUpdated, tick);

        if (delta > 200) // update every 100ms
        {
            _tickLastUpdated = tick;
            osg::Stats *stats = _view->getStats();
            if (stats)
            {
                std::ostringstream viewStr;
                viewStr.clear();
                viewStr.setf(std::ios::left, std::ios::adjustfield);
                viewStr.width(20);
                viewStr.setf(std::ios::fixed);
                viewStr.precision(0);

                viewStr << std::setw(1) << "#" << _viewNumber;

                // View name
                if (!_view->getName().empty())
                    viewStr << ": " << _view->getName();
                viewStr << std::endl;

                int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();
                // if (!(renderer->getGraphicsThreadDoesCull()))
                {
                    --frameNumber;
                }

#define STATS_ATTRIBUTE_PAIR(str1, str2)               \
    if (stats->getAttribute(frameNumber, str1, value)) \
        viewStr << std::setw(9) << value;              \
    else                                               \
        viewStr << std::setw(9) << ".";                \
    if (stats->getAttribute(frameNumber, str2, value)) \
        viewStr << std::setw(9) << value << std::endl; \
    else                                               \
        viewStr << std::setw(9) << "." << std::endl;

                double value = 0.0;

                // header
                viewStr << std::setw(9) << "Unique" << std::setw(9) << "Instance" << std::endl;

                STATS_ATTRIBUTE_PAIR("Number of unique StateSet", "Number of instanced Stateset")
                STATS_ATTRIBUTE_PAIR("Number of unique Group", "Number of instanced Group")
                STATS_ATTRIBUTE_PAIR("Number of unique Transform", "Number of instanced Transform")
                STATS_ATTRIBUTE_PAIR("Number of unique LOD", "Number of instanced LOD")
                STATS_ATTRIBUTE_PAIR("Number of unique Switch", "Number of instanced Switch")
                STATS_ATTRIBUTE_PAIR("Number of unique Geode", "Number of instanced Geode")
                STATS_ATTRIBUTE_PAIR("Number of unique Drawable", "Number of instanced Drawable")
                STATS_ATTRIBUTE_PAIR("Number of unique Geometry", "Number of instanced Geometry")
                STATS_ATTRIBUTE_PAIR("Number of unique Vertices", "Number of instanced Vertices")
                STATS_ATTRIBUTE_PAIR("Number of unique Primitives", "Number of instanced Primitives")

                text->setText(viewStr.str(), osgText::String::ENCODING_UTF8);
            }
            else
            {
                osg::notify(osg::WARN) << std::endl << "No valid view to collect scene stats from" << std::endl;

                text->setText("", osgText::String::ENCODING_UTF8);
            }
        }
        text->drawImplementation(renderInfo);
    }

    osg::observer_ptr<osgViewer::View> _view;
    mutable osg::Timer_t _tickLastUpdated;
    int _viewNumber;
};

struct BlockDrawCallback : public virtual osg::Drawable::DrawCallback
{
    BlockDrawCallback(coVRStatsDisplay *statsHandler, float xPos, osg::Stats *viewerStats, osg::Stats *stats, const std::string &beginName, const std::string &endName, int frameDelta, int numFrames)
        : _statsHandler(statsHandler)
        , _xPos(xPos)
        , _viewerStats(viewerStats)
        , _stats(stats)
        , _beginName(beginName)
        , _endName(endName)
        , _frameDelta(frameDelta)
        , _numFrames(numFrames)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        osg::Geometry *geom = (osg::Geometry *)drawable;
        osg::Vec3Array *vertices = (osg::Vec3Array *)geom->getVertexArray();

        int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

        int startFrame = frameNumber + _frameDelta - _numFrames + 1;
        int endFrame = frameNumber + _frameDelta;
        double referenceTime;
        if (!_viewerStats->getAttribute(startFrame, "Reference time", referenceTime))
        {
            return;
        }

        unsigned int vi = 0;
        double beginValue, endValue;
        for (int i = startFrame; i <= endFrame; ++i)
        {
            if (_stats->getAttribute(i, _beginName, beginValue) && _stats->getAttribute(i, _endName, endValue))
            {
                (*vertices)[vi++].x() = _xPos + (beginValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (beginValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (endValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (endValue - referenceTime) * _statsHandler->getBlockMultiplier();
            }
        }
		vertices->dirty();

        drawable->drawImplementation(renderInfo);
    }

    coVRStatsDisplay *_statsHandler;
    float _xPos;
    osg::ref_ptr<osg::Stats> _viewerStats;
    osg::ref_ptr<osg::Stats> _stats;
    std::string _beginName;
    std::string _endName;
    int _frameDelta;
    int _numFrames;
};

struct SlowestBlockDrawCallback : public virtual osg::Drawable::DrawCallback
{
    SlowestBlockDrawCallback(coVRStatsDisplay *statsHandler, float xPos, osg::Stats *viewerStats, osg::Stats *stats, const std::string &beginName, const std::string &endName)
        : _statsHandler(statsHandler)
        , _xPos(xPos)
        , _viewerStats(viewerStats)
        , _stats(stats)
        , _beginName(beginName)
        , _endName(endName)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        osg::Geometry *geom = (osg::Geometry *)drawable;
        osg::Vec3Array *vertices = (osg::Vec3Array *)geom->getVertexArray();

        int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();
        int slowest = getSlowestFrame(_viewerStats, frameNumber);
        double referenceTime;
        if (!_viewerStats->getAttribute(slowest, "Reference time", referenceTime))
        {
            return;
        }

        unsigned int vi = 0;
        double beginValue, endValue;
        if (slowest >= 0)
        {
            if (_stats->getAttribute(slowest, _beginName, beginValue) && _stats->getAttribute(slowest, _endName, endValue))
            {
                (*vertices)[vi++].x() = _xPos + (beginValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (beginValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (endValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (endValue - referenceTime) * _statsHandler->getBlockMultiplier();
            }
        }

        vertices->dirty();

        drawable->drawImplementation(renderInfo);
    }

    coVRStatsDisplay *_statsHandler;
    float _xPos;
    osg::ref_ptr<osg::Stats> _viewerStats;
    osg::ref_ptr<osg::Stats> _stats;
    std::string _beginName;
    std::string _endName;
    int _numFrames;
};

osg::Geometry *coVRStatsDisplay::createBackgroundRectangle(const osg::Vec3 &pos, const float width, const float height, osg::Vec4 &color)
{
    osg::StateSet *ss = new osg::StateSet;

    osg::Geometry *geometry = new osg::Geometry;

    geometry->setUseDisplayList(false);
    geometry->setStateSet(ss);

    osg::Vec3Array *vertices = new osg::Vec3Array;
    geometry->setVertexArray(vertices);

    vertices->push_back(osg::Vec3(pos.x(), pos.y(), 0));
    vertices->push_back(osg::Vec3(pos.x(), pos.y() - height, 0));
    vertices->push_back(osg::Vec3(pos.x() + width, pos.y() - height, 0));
    vertices->push_back(osg::Vec3(pos.x() + width, pos.y(), 0));

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(color);
    geometry->setColorArray(colors);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::DrawElementsUInt *base = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    base->push_back(0);
    base->push_back(1);
    base->push_back(2);
    base->push_back(3);

    geometry->addPrimitiveSet(base);

    return geometry;
}

struct StatsGraph : public osg::MatrixTransform
{
    StatsGraph(osg::Vec3 pos, float width, float height, int stackHeight=0)
        : _pos(pos)
        , _width(width)
        , _height(height)
        , _stackHeight(stackHeight)
        , _statsGraphGeode(new osg::Geode)
    {
        _pos -= osg::Vec3(0, height, 0.1);
        setMatrix(osg::Matrix::translate(_pos));
        addChild(_statsGraphGeode.get());

        _statsGraphGeode->addDrawable(new Ticker(this, _width, _height, osg::Vec4(1,1,1,1)));
    }

    void addStatGraph(osg::Stats *viewerStats, osg::Stats *stats, const osg::Vec4 &color, float base, float max, const std::string &nameBegin, Accum accum=AccumAverage, const std::string &nameEnd = "")
    {
        if (base > _stackHeight)
            base = _stackHeight;
        _statsGraphGeode->addDrawable(new Graph(this, base*0.05*_height, _width, _height*(1.-0.05*_stackHeight), viewerStats, stats, color, max, nameBegin, accum, nameEnd));
        ++_numGraphs;
    }

    int _frameNumber = 0;
    osg::Vec3 _pos;
    float _width;
    float _height;
    int _numGraphs = 0;
    int _stackHeight = 0;
    static constexpr float increment = 1.9;

    osg::ref_ptr<osg::Geode> _statsGraphGeode;

protected:
    struct Ticker : public osg::Geometry
    {
        Ticker(struct StatsGraph *graph, float width, float height,
              const osg::Vec4 &color)
        {
            setUseDisplayList(false);

            setVertexArray(new osg::Vec3Array);

            osg::Vec4Array *colors = new osg::Vec4Array;
            colors->push_back(color);
            setColorArray(colors);
            setColorBinding(osg::Geometry::BIND_OVERALL);

            osg::StateSet *s = getOrCreateStateSet();
            s->setAttribute(new osg::LineWidth(2.), osg::StateAttribute::ON);

            setDrawCallback(new TickerUpdateCallback(graph, width, height));
        }
    };
    struct TickerUpdateCallback : public osg::Drawable::DrawCallback
    {
        TickerUpdateCallback(StatsGraph *graph, float width, float height)
            : _width((unsigned int)width)
            , _height((unsigned int)height)
            , _curX(0.f)
            , _graph(graph)
        {
        }

        virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
        {
            osg::Geometry *geometry = const_cast<osg::Geometry *>(drawable->asGeometry());
            if (!geometry)
                return;
            osg::Vec3Array *vertices = dynamic_cast<osg::Vec3Array *>(geometry->getVertexArray());
            if (!vertices)
                return;


            double t = renderInfo.getState()->getFrameStamp()->getReferenceTime();
            times.push_back(t);

            // One vertex per pixel in X.
            unsigned width = _width/increment;
            if (times.size() > width)
            {
                unsigned int excedent = times.size() - width;
                times.erase(times.begin(), times.begin() + excedent);
            }

            if (trunc(t) > _last)
            {
                _last = t;
                vertices->push_back(osg::Vec3(_curX,  0.00f*float(_height), 0));
                vertices->push_back(osg::Vec3(_curX, -0.03f*float(_height), 0));
            }

            if (vertices->size() > width)
            {
                unsigned int excedent = vertices->size() - width;
                excedent &= ~1;
                vertices->erase(vertices->begin(), vertices->begin() + excedent);
            }

            // Create primitive set if none exists.
            if (geometry->getNumPrimitiveSets() == 0)
                geometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 0));

            // Update primitive set.
            osg::DrawArrays *drawArrays = dynamic_cast<osg::DrawArrays *>(geometry->getPrimitiveSet(0));
            if (!drawArrays)
                return;
            drawArrays->setFirst(0);
            drawArrays->setCount(vertices->size());

            _curX += increment;

            geometry->dirtyBound();

            vertices->dirty();

            drawable->drawImplementation(renderInfo);
        }

        const unsigned int _width;
        const unsigned int _height;
        mutable float _curX;
        mutable double _last = -1.;
        mutable std::vector<double> times;
        StatsGraph *_graph = nullptr;
    };

    struct Graph : public osg::Geometry
    {
        Graph(struct StatsGraph *graph, float base, float width, float height, osg::Stats *viewerStats, osg::Stats *stats,
              const osg::Vec4 &color, float max, const std::string &nameBegin, Accum accum=AccumAverage, const std::string &nameEnd = "")
        {
            setUseDisplayList(false);

            setVertexArray(new osg::Vec3Array);

            osg::Vec4Array *colors = new osg::Vec4Array;
            colors->push_back(color);
            setColorArray(colors);
            setColorBinding(osg::Geometry::BIND_OVERALL);

            setDrawCallback(new GraphUpdateCallback(graph, base, width, height, viewerStats, stats, max, nameBegin, accum, nameEnd));
        }
    };

    struct GraphUpdateCallback : public osg::Drawable::DrawCallback
    {
        GraphUpdateCallback(StatsGraph *graph, float base, float width, float height, osg::Stats *viewerStats, osg::Stats *stats,
                            float max, const std::string &nameBegin, Accum accum, const std::string &nameEnd = "")
            : _base(base)
            , _width((unsigned int)width)
            , _height((unsigned int)height)
            , _curX(0.f)
            , _viewerStats(viewerStats)
            , _stats(stats)
            , _max(max)
            , _accum(accum)
            , _nameBegin(nameBegin)
            , _nameEnd(nameEnd)
            , _graph(graph)
        {
        }

        virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
        {
            osg::Geometry *geometry = const_cast<osg::Geometry *>(drawable->asGeometry());
            if (!geometry)
                return;
            osg::Vec3Array *vertices = dynamic_cast<osg::Vec3Array *>(geometry->getVertexArray());
            if (!vertices)
                return;

            int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

            // Get stats
            double value;
            if (_nameEnd.empty())
            {
                if (!getAttribute(_stats, _nameBegin, _accum, value))
                {
                    value = 0.0;
                }
            }
            else
            {
                double beginValue, endValue;
                if (_stats->getAttribute(frameNumber, _nameBegin, beginValue) && _stats->getAttribute(frameNumber, _nameEnd, endValue))
                {
                    value = endValue - beginValue;
                }
                else
                {
                    value = 0.0;
                }
            }

            // Add new vertex for this frame.
            value = osg::clampTo(value, 0.0, double(_max));
            vertices->push_back(osg::Vec3(_curX, _base + float(_height) / _max * value, 0));

            // One vertex per pixel in X.
            unsigned width = _width/increment;
            if (vertices->size() > width)
            {
                unsigned int excedent = vertices->size() - width;
                vertices->erase(vertices->begin(), vertices->begin() + excedent);

                // Make the graph scroll when there is enough data.
                // Note: We check the frame number so that even if we have
                // many graphs, the transform is translated only once per
                // frame.
                if (_graph->_frameNumber != frameNumber)
                {
                    // We know the exact layout of this part of the scene
                    // graph, so this is OK...
                    osg::MatrixTransform *transform = geometry->getParent(0)->getParent(0)->asTransform()->asMatrixTransform();
                    if (transform)
                    {
                        transform->setMatrix(osg::Matrix::translate(_graph->_pos)*osg::Matrix::translate(osg::Vec3(-vertices->at(0)[0], 0, 0)));

                    }
                }
            }
            else
            {
                // Create primitive set if none exists.
                if (geometry->getNumPrimitiveSets() == 0)
                    geometry->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, 0));

                // Update primitive set.
                osg::DrawArrays *drawArrays = dynamic_cast<osg::DrawArrays *>(geometry->getPrimitiveSet(0));
                if (!drawArrays)
                    return;
                drawArrays->setFirst(0);
                drawArrays->setCount(vertices->size());
            }

            _curX += increment;
            _graph->_frameNumber = frameNumber;

            geometry->dirtyBound();

			vertices->dirty();

            drawable->drawImplementation(renderInfo);
        }

        float _base = 0.f;
        const unsigned int _width;
        const unsigned int _height;
        mutable float _curX;
        osg::Stats *_viewerStats;
        osg::Stats *_stats;
        const float _max;
        Accum _accum;
        bool _average;
        bool _averageInInverseSpace;
        const std::string _nameBegin;
        const std::string _nameEnd;
        StatsGraph *_graph = nullptr;
    };
};

osg::Geometry *coVRStatsDisplay::createGeometry(const osg::Vec3 &pos, float height, const osg::Vec4 &colour, unsigned int numBlocks)
{
    osg::Geometry *geometry = new osg::Geometry;

    geometry->setUseDisplayList(false);

    osg::Vec3Array *vertices = new osg::Vec3Array;
    geometry->setVertexArray(vertices);
    vertices->reserve(numBlocks * 4);

    for (unsigned int i = 0; i < numBlocks; ++i)
    {
        vertices->push_back(pos + osg::Vec3(i * 20, height, 0.0));
        vertices->push_back(pos + osg::Vec3(i * 20, 0.0, 0.0));
        vertices->push_back(pos + osg::Vec3(i * 20 + 10.0, 0.0, 0.0));
        vertices->push_back(pos + osg::Vec3(i * 20 + 10.0, height, 0.0));
    }

    osg::Vec4Array *colours = new osg::Vec4Array;
    colours->push_back(colour);
    geometry->setColorArray(colours);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    geometry->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, numBlocks * 4));

    return geometry;
}

struct FrameMarkerDrawCallback : public virtual osg::Drawable::DrawCallback
{
    FrameMarkerDrawCallback(coVRStatsDisplay *statsHandler, float xPos, osg::Stats *viewerStats, int frameDelta, int numFrames)
        : _statsHandler(statsHandler)
        , _xPos(xPos)
        , _viewerStats(viewerStats)
        , _frameDelta(frameDelta)
        , _numFrames(numFrames)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        osg::Geometry *geom = (osg::Geometry *)drawable;
        osg::Vec3Array *vertices = (osg::Vec3Array *)geom->getVertexArray();

        int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

        int startFrame = frameNumber + _frameDelta - _numFrames + 1;
        int endFrame = frameNumber + _frameDelta;
        double referenceTime;
        if (!_viewerStats->getAttribute(startFrame, "Reference time", referenceTime))
        {
            return;
        }

        unsigned int vi = 0;
        double currentReferenceTime;
        for (int i = startFrame; i <= endFrame; ++i)
        {
            if (_viewerStats->getAttribute(i, "Reference time", currentReferenceTime))
            {
                (*vertices)[vi++].x() = _xPos + (currentReferenceTime - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (currentReferenceTime - referenceTime) * _statsHandler->getBlockMultiplier();
            }
        }

		vertices->dirty();

        drawable->drawImplementation(renderInfo);
    }

    coVRStatsDisplay *_statsHandler;
    float _xPos;
    osg::ref_ptr<osg::Stats> _viewerStats;
    std::string _endName;
    int _frameDelta;
    int _numFrames;
};

struct SlowFrameMarkerDrawCallback : public virtual osg::Drawable::DrawCallback
{
    SlowFrameMarkerDrawCallback(coVRStatsDisplay *statsHandler, float xPos, osg::Stats *viewerStats)
        : _statsHandler(statsHandler)
        , _xPos(xPos)
        , _viewerStats(viewerStats)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        osg::Geometry *geom = (osg::Geometry *)drawable;
        osg::Vec3Array *vertices = (osg::Vec3Array *)geom->getVertexArray();

        int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

        int slowest = getSlowestFrame(_viewerStats, frameNumber);
        double referenceTime;
        if (!_viewerStats->getAttribute(slowest, "Reference time", referenceTime))
        {
            return;
        }

        unsigned int vi = 0;
        double currentReferenceTime;
        if (_viewerStats->getAttribute(slowest+1, "Reference time", currentReferenceTime))
        {
            (*vertices)[vi++].x() = _xPos + (currentReferenceTime - referenceTime) * _statsHandler->getBlockMultiplier();
            (*vertices)[vi++].x() = _xPos + (currentReferenceTime - referenceTime) * _statsHandler->getBlockMultiplier();
        }

        vertices->dirty();

        drawable->drawImplementation(renderInfo);
    }

    coVRStatsDisplay *_statsHandler;
    float _xPos;
    osg::ref_ptr<osg::Stats> _viewerStats;
    std::string _endName;
};

struct PagerCallback : public virtual osg::NodeCallback
{

    PagerCallback(osgDB::DatabasePager *dp,
                  osgText::Text *minValue,
                  osgText::Text *maxValue,
                  osgText::Text *averageValue,
                  osgText::Text *filerequestlist,
                  osgText::Text *compilelist,
                  double multiplier)
        : _dp(dp)
        , _minValue(minValue)
        , _maxValue(maxValue)
        , _averageValue(averageValue)
        , _filerequestlist(filerequestlist)
        , _compilelist(compilelist)
        , _multiplier(multiplier)
    {
    }

    virtual void operator()(osg::Node *node, osg::NodeVisitor *nv)
    {
        if (_dp.valid())
        {
            double value = _dp->getAverageTimeToMergeTiles();
            if (value >= 0.0 && value <= 1000)
            {
                sprintf(_tmpText, "%4.0f", value * _multiplier);
                _averageValue->setText(_tmpText, osgText::String::ENCODING_UTF8);
            }
            else
            {
                _averageValue->setText("", osgText::String::ENCODING_UTF8);
            }

            value = _dp->getMinimumTimeToMergeTile();
            if (value >= 0.0 && value <= 1000)
            {
                sprintf(_tmpText, "%4.0f", value * _multiplier);
                _minValue->setText(_tmpText, osgText::String::ENCODING_UTF8);
            }
            else
            {
                _minValue->setText("", osgText::String::ENCODING_UTF8);
            }

            value = _dp->getMaximumTimeToMergeTile();
            if (value >= 0.0 && value <= 1000)
            {
                sprintf(_tmpText, "%4.0f", value * _multiplier);
                _maxValue->setText(_tmpText, osgText::String::ENCODING_UTF8);
            }
            else
            {
                _maxValue->setText("", osgText::String::ENCODING_UTF8);
            }

            sprintf(_tmpText, "%4d", _dp->getFileRequestListSize());
            _filerequestlist->setText(_tmpText, osgText::String::ENCODING_UTF8);

            sprintf(_tmpText, "%4d", _dp->getDataToCompileListSize());
            _compilelist->setText(_tmpText, osgText::String::ENCODING_UTF8);
        }

        traverse(node, nv);
    }

    osg::observer_ptr<osgDB::DatabasePager> _dp;

    osg::ref_ptr<osgText::Text> _minValue;
    osg::ref_ptr<osgText::Text> _maxValue;
    osg::ref_ptr<osgText::Text> _averageValue;
    osg::ref_ptr<osgText::Text> _filerequestlist;
    osg::ref_ptr<osgText::Text> _compilelist;
    double _multiplier;
    char _tmpText[128];
    osg::Timer_t _tickLastUpdated;
};

osg::Geometry *coVRStatsDisplay::createFrameMarkers(const osg::Vec3 &pos, float height, const osg::Vec4 &colour, unsigned int numBlocks)
{
    osg::Geometry *geometry = new osg::Geometry;

    geometry->setUseDisplayList(false);

    osg::Vec3Array *vertices = new osg::Vec3Array;
    geometry->setVertexArray(vertices);
    vertices->reserve(numBlocks * 2);

    for (unsigned int i = 0; i < numBlocks; ++i)
    {
        vertices->push_back(pos + osg::Vec3(double(i) * _blockMultiplier * 0.01, height, 0.0));
        vertices->push_back(pos + osg::Vec3(double(i) * _blockMultiplier * 0.01, 0.0, 0.0));
    }

    osg::Vec4Array *colours = new osg::Vec4Array;
    colours->push_back(colour);
    geometry->setColorArray(colours);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    geometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, numBlocks * 2));

    return geometry;
}

osg::Geometry *coVRStatsDisplay::createTick(const osg::Vec3 &pos, float height, const osg::Vec4 &colour, unsigned int numTicks)
{
    osg::Geometry *geometry = new osg::Geometry;

    geometry->setUseDisplayList(false);

    osg::Vec3Array *vertices = new osg::Vec3Array;
    geometry->setVertexArray(vertices);
    vertices->reserve(numTicks * 2);

    for (unsigned int i = 0; i < numTicks; ++i)
    {
        float tickHeight = (i % 10) ? height : height * 2.0;
        vertices->push_back(pos + osg::Vec3(double(i) * _blockMultiplier * 0.001, tickHeight, 0.0));
        vertices->push_back(pos + osg::Vec3(double(i) * _blockMultiplier * 0.001, 0.0, 0.0));
    }

    osg::Vec4Array *colours = new osg::Vec4Array;
    colours->push_back(colour);
    geometry->setColorArray(colours);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    geometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, numTicks * 2));

    return geometry;
}

void coVRStatsDisplay::setUpScene(osgViewer::ViewerBase *viewer)
{
    _switch = new osg::Switch;

    _camera->addChild(_switch.get());

    osg::StateSet *stateset = _switch->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
    stateset->setAttribute(new osg::PolygonMode(), osg::StateAttribute::PROTECTED);

    std::string font = coVRFileManager::instance()->getFontFile(NULL);

    // collect all the relevant cameras
    osgViewer::ViewerBase::Cameras validCameras;
    viewer->getCameras(validCameras);

    osgViewer::ViewerBase::Cameras cameras;
    for (osgViewer::ViewerBase::Cameras::iterator itr = validCameras.begin();
         itr != validCameras.end();
         ++itr)
    {
        if ((*itr)->getStats())
        {
            cameras.push_back(*itr);
        }
    }

    // check for query time support
    unsigned int numCamerasWithTimerQuerySupport = 0;
   /* for (osgViewer::ViewerBase::Cameras::iterator citr = cameras.begin();
         citr != cameras.end();
         ++citr)
    {
        if ((*citr)->getGraphicsContext())
        {
            unsigned int contextID = (*citr)->getGraphicsContext()->getState()->getContextID();
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
            const osg::ref_ptr<osg::GLExtensions> extensions = new osg::GLExtensions(contextID);
            if (extensions && extensions->isTimerQuerySupported)
#else
            const osg::ref_ptr<osg::Drawable::Extensions> extensions = osg::Drawable::getExtensions(contextID, false);
            if (extensions && extensions->isTimerQuerySupported())
#endif
            {
                ++numCamerasWithTimerQuerySupport;
            }
        }
    }*/

    bool acquireGPUStats = numCamerasWithTimerQuerySupport == cameras.size();
    acquireGPUStats = true;

    float leftPos = covise::coCoviseConfig::getFloat("leftPos", "COVER.Stats", 10.0f);
    float startBlocks = 150.0f;
    float characterSize = 20.0f;
    float space = covise::coCoviseConfig::getFloat("space", "COVER.Stats", characterSize*0.3f);

    osg::Vec3 pos(leftPos, _statsHeight - 24.0f, 0.0f);

    osg::Vec4 colorFR(1.0f, 1.0f, 1.0f, 1.0f);
    osg::Vec4 colorFRAlpha(1.0f, 1.0f, 1.0f, 0.5f);
    osg::Vec4 colorMaxFR(1.0f, 0.0f, 1.0f, 1.0f);
    osg::Vec4 colorMaxFRAlpha(1.0f, 0.0f, 1.0f, 0.5f);
    osg::Vec4 colorUpdate(0.0f, 1.0f, 0.0f, 1.0f);
    osg::Vec4 colorUpdateAlpha(0.0f, 1.0f, 0.0f, 0.5f);
    osg::Vec4 colorSync(1.0f, 0.0f, 0.0f, 1.0f);
    osg::Vec4 colorSyncAlpha(1.0f, 0.0f, 0.0f, 0.5f);
    osg::Vec4 colorSwap(0.5f, 1.0f, 0.5f, 1.0f);
    osg::Vec4 colorSwapAlpha(0.5f, 1.0f, 0.5f, 0.5f);
    osg::Vec4 colorFinish(0.5f, 1.0f, 0.0f, 1.0f);
    osg::Vec4 colorFinishAlpha(0.5f, 1.0f, 0.0f, 0.5f);
    osg::Vec4 colorEvent(0.0f, 1.0f, 0.5f, 1.0f);
    osg::Vec4 colorEventAlpha(0.0f, 1.0f, 0.5f, 0.5f);
    osg::Vec4 colorIsect(0.0f, 0.5f, 0.8f, 1.0f);
    osg::Vec4 colorIsectAlpha(0.0f, 0.5f, 0.8f, 0.5f);
    osg::Vec4 colorPlugin(0.0f, 0.2f, 1.0f, 1.0f);
    osg::Vec4 colorPluginAlpha(0.0f, 0.2f, 1.0f, 0.5f);
    osg::Vec4 colorCover(0.0f, 0.5f, 1.0f, 1.0f);
    osg::Vec4 colorCoverAlpha(0.0f, 0.5f, 1.0f, 0.5f);
    osg::Vec4 colorCull(0.0f, 1.0f, 1.0f, 1.0f);
    osg::Vec4 colorCullAlpha(0.0f, 1.0f, 1.0f, 0.5f);
    osg::Vec4 colorDraw(1.0f, 1.0f, 0.0f, 1.0f);
    osg::Vec4 colorDrawAlpha(1.0f, 1.0f, 0.0f, 0.5f);
    osg::Vec4 colorGPU(1.0f, 0.5f, 0.0f, 1.0f);
    osg::Vec4 colorGPUAlpha(1.0f, 0.5f, 0.0f, 0.5f);

    osg::Vec4 colorGpuUtil(1.f, 1.f, 1.f, 1.f);
    osg::Vec4 colorGpuMemClock(1.f, 1.f, 0.f, 1.f);
    osg::Vec4 colorGpuClock(1.f, 0.5f, 0.f, 1.f);
    osg::Vec4 colorGpuPCIe(0.f, 1.f, 1.f, 1.f);
    osg::Vec4 colorGpuMem(1.f, 1.f, 1.f, 1.f);

    osg::Vec4 colorDP(1.0f, 1.0f, 0.5f, 1.0f);

    // frame rate stats
    {
        osg::Geode *geode = new osg::Geode();
        _frameRateChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> frameRateLabel = new osgText::Text;
        geode->addDrawable(frameRateLabel.get());
        frameRateLabel->setColor(colorFR);
        frameRateLabel->setFont(font);
        frameRateLabel->setCharacterSize(characterSize);
        frameRateLabel->setPosition(pos);
        frameRateLabel->setText("Frames/s:X", osgText::String::ENCODING_UTF8);
        pos.x() = frameRateLabel->getBound().xMax();
        frameRateLabel->setText("Frames/s: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> frameRateValue = new osgText::Text;
        geode->addDrawable(frameRateValue.get());
        frameRateValue->setColor(colorFR);
        frameRateValue->setFont(font);
        frameRateValue->setCharacterSize(characterSize);
        frameRateValue->setPosition(pos);
        frameRateValue->setText("7777.77", osgText::String::ENCODING_UTF8);
        frameRateValue->setDrawCallback(new AveragedValueTextDrawCallback(viewer->getViewerStats(), "Frame rate", AccumAverageInverse, 1.0));
        pos.x() = frameRateValue->getBound().xMax();

        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        geode->addDrawable(label.get());
        label->setColor(colorFR);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos);
        label->setText("ms/F:X", osgText::String::ENCODING_UTF8);
        pos.x() = label->getBound().xMax();
        label->setText("ms/F: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text;
        geode->addDrawable(value.get());
        value->setColor(colorFR);
        value->setFont(font);
        value->setCharacterSize(characterSize);
        value->setPosition(pos);
        value->setText("777.7/777.7", osgText::String::ENCODING_UTF8);
        value->setDrawCallback(new AveragedValueTextDrawCallback(viewer->getViewerStats(), "Frame duration", AccumMeanJitter, 1000.0));
        pos.x() = value->getBound().xMax();

        pos.x() += space;
    }

    // GPU utilization
    {
        osg::Geode *geode = new osg::Geode();
        _gpuUtilChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        geode->addDrawable(label.get());

        label->setColor(colorGpuUtil);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos);
        label->setText("Util:X", osgText::String::ENCODING_UTF8);
        pos.x() = label->getBound().xMax();
        label->setText("Util: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text;
        geode->addDrawable(value.get());

        value->setColor(colorGpuUtil);
        value->setFont(font);
        value->setCharacterSize(characterSize);
        value->setPosition(pos);
        value->setText("7777.77", osgText::String::ENCODING_UTF8);

        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "GPU Utilization");
        value->setDrawCallback(cb);

        pos.x() = value->getBound().xMax();
        pos.x() += space;
    }

    // PCIe rx rate
    {
        osg::Geode *geode = new osg::Geode();
        _gpuPCIeChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        geode->addDrawable(label.get());

        label->setColor(colorGpuPCIe);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos);
        label->setText("PCIe:X", osgText::String::ENCODING_UTF8);
        pos.x() = label->getBound().xMax();
        label->setText("PCIe: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text;
        geode->addDrawable(value.get());

        value->setColor(colorGpuPCIe);
        value->setFont(font);
        value->setCharacterSize(characterSize);
        value->setPosition(pos);
        value->setText("7.77", osgText::String::ENCODING_UTF8);
        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "GPU PCIe rx KB/s", AccumAverageInverse, 1./1024/1024);
        value->setDrawCallback(cb);
        pos.x() = value->getBound().xMax();

        osg::ref_ptr<osgText::Text> sep = new osgText::Text;
        geode->addDrawable(sep.get());
        sep->setColor(colorGpuPCIe);
        sep->setFont(font);
        sep->setCharacterSize(characterSize);
        sep->setPosition(pos);
        sep->setText("/", osgText::String::ENCODING_UTF8);
        pos.x() = sep->getBound().xMax();

        osg::ref_ptr<osgText::Text> value2 = new osgText::Text;
        geode->addDrawable(value2.get());
        value2->setColor(colorGpuPCIe);
        value2->setFont(font);
        value2->setCharacterSize(characterSize);
        value2->setPosition(pos);
        value2->setText("7.77", osgText::String::ENCODING_UTF8);
        pos.x() = value2->getBound().xMax();

        auto cb2 = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "GPU PCIe tx KB/s", AccumAverageInverse, 1./1024/1024);
        value2->setDrawCallback(cb2);

        pos.x() += space;
    }

    // GPU clock
    {
        osg::Geode *geode = new osg::Geode();
        _gpuClockChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        geode->addDrawable(label.get());
        label->setColor(colorGpuClock);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos);
        label->setText("GPU/Mem:X", osgText::String::ENCODING_UTF8);
        pos.x() = label->getBound().xMax();
        label->setText("GPU/Mem: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text;
        geode->addDrawable(value.get());
        value->setColor(colorGpuClock);
        value->setFont(font);
        value->setCharacterSize(characterSize);
        value->setPosition(pos);
        value->setText("7777.77", osgText::String::ENCODING_UTF8);
        pos.x() = value->getBound().xMax();
        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "GPU Clock MHz");
        value->setDrawCallback(cb);

        osg::ref_ptr<osgText::Text> sep = new osgText::Text;
        geode->addDrawable(sep.get());
        sep->setColor(colorGpuClock);
        sep->setFont(font);
        sep->setCharacterSize(characterSize);
        sep->setPosition(pos);
        sep->setText("/", osgText::String::ENCODING_UTF8);
        pos.x() = sep->getBound().xMax();

        osg::ref_ptr<osgText::Text> value2 = new osgText::Text;
        geode->addDrawable(value2.get());
        value2->setColor(colorGpuMemClock);
        value2->setFont(font);
        value2->setCharacterSize(characterSize);
        value2->setPosition(pos);
        value2->setText("7777.77", osgText::String::ENCODING_UTF8);
        pos.x() = value2->getBound().xMax();
        auto cb2 = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "GPU Mem Clock MHz");
        value2->setDrawCallback(cb2);

        osg::ref_ptr<osgText::Text> label2 = new osgText::Text;
        geode->addDrawable(label2.get());
        label2->setColor(colorGpuMemClock);
        label2->setFont(font);
        label2->setCharacterSize(characterSize);
        label2->setPosition(pos);
        label2->setText(" MHz", osgText::String::ENCODING_UTF8);
        pos.x() = label2->getBound().xMax();

        pos.x() += space;
    }

    // used GPU memory
    {
        osg::Geode *geode = new osg::Geode();
        _gpuMemChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

#if 0
        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        geode->addDrawable(label.get());
        label->setColor(colorGpuMem);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos);
        label->setText("Used:X", osgText::String::ENCODING_UTF8);
        pos.x() = label->getBound().xMax();
        label->setText("Used: ", osgText::String::ENCODING_UTF8);
#endif

        osg::ref_ptr<osgText::Text> value = new osgText::Text;
        geode->addDrawable(value.get());
        value->setColor(colorGpuMem);
        value->setFont(font);
        value->setCharacterSize(characterSize);
        value->setPosition(pos);
        value->setText("77.77", osgText::String::ENCODING_UTF8);
        pos.x() = value->getBound().xMax();
        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "GPU Mem Used", AccumMax, 1./1024/1024/1024);
        value->setDrawCallback(cb);

        osg::ref_ptr<osgText::Text> label2 = new osgText::Text;
        geode->addDrawable(label2.get());
        label2->setColor(colorGpuMem);
        label2->setFont(font);
        label2->setCharacterSize(characterSize);
        label2->setPosition(pos);
        label2->setText(" GB", osgText::String::ENCODING_UTF8);
        pos.x() = label2->getBound().xMax();

        pos.x() += space;
    }

    // next line
    pos.y() -= characterSize * 1.5f;
    pos.x() = leftPos;

    // threading model
    {
        osg::Geode *geode = new osg::Geode();

        _threadingModelText = new osgText::Text;
        _threadingModelChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        geode->addDrawable(_threadingModelText.get());

        _threadingModelText->setColor(colorFR);
        _threadingModelText->setFont(font);
        _threadingModelText->setCharacterSize(characterSize);
        _threadingModelText->setPosition(pos);

        //updateThreadingModelText(osgViewer::Viewer::CullThreadPerCameraDrawThreadPerContext);
        updateThreadingModelText(viewer->getThreadingModel());
        pos.x() = _threadingModelText->getBound().xMax();
        pos.x() += 3.*space;

        updateThreadingModelText(viewer->getThreadingModel());
    }

    // remote FPS
    {
        osg::Geode *geode = new osg::Geode();
        _rhrFpsChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> rhrFpsLabel = new osgText::Text;
        geode->addDrawable(rhrFpsLabel.get());

        rhrFpsLabel->setColor(colorFR);
        rhrFpsLabel->setFont(font);
        rhrFpsLabel->setCharacterSize(characterSize);
        rhrFpsLabel->setPosition(pos);
        rhrFpsLabel->setText("RHR FPS:X", osgText::String::ENCODING_UTF8);
        pos.x() = rhrFpsLabel->getBound().xMax();
        rhrFpsLabel->setText("RHR FPS: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> rhrFpsValue = new osgText::Text;
        geode->addDrawable(rhrFpsValue.get());

        rhrFpsValue->setColor(colorFR);
        rhrFpsValue->setFont(font);
        rhrFpsValue->setCharacterSize(characterSize);
        rhrFpsValue->setPosition(pos);
        rhrFpsValue->setText("777.77", osgText::String::ENCODING_UTF8);

        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "RHR FPS", AccumAverageInverse, 1.);
        rhrFpsValue->setDrawCallback(cb);

        pos.x() = rhrFpsValue->getBound().xMax();
        pos.x() += space;
    }

    // remote render latency
    {
        osg::Geode *geode = new osg::Geode();
        _rhrDelayChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> rhrDelayLabel = new osgText::Text;
        geode->addDrawable(rhrDelayLabel.get());

        rhrDelayLabel->setColor(colorFR);
        rhrDelayLabel->setFont(font);
        rhrDelayLabel->setCharacterSize(characterSize);
        rhrDelayLabel->setPosition(pos);
        rhrDelayLabel->setText("Delay (s):X", osgText::String::ENCODING_UTF8);
        pos.x() = rhrDelayLabel->getBound().xMax();
        rhrDelayLabel->setText("Delay (s): ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> rhrDelayValue = new osgText::Text;
        geode->addDrawable(rhrDelayValue.get());

        rhrDelayValue->setColor(colorFR);
        rhrDelayValue->setFont(font);
        rhrDelayValue->setCharacterSize(characterSize);
        rhrDelayValue->setPosition(pos);
        rhrDelayValue->setText("7.777", osgText::String::ENCODING_UTF8);

        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "RHR Delay", AccumMax, 1.);
        rhrDelayValue->setDrawCallback(cb);

        pos.x() = rhrDelayValue->getBound().xMax();
        pos.x() += space;
    }

    // remote render bandwidth
    {
        osg::Geode *geode = new osg::Geode();
        _rhrBandwidthChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> rhrBandwidthLabel = new osgText::Text;
        geode->addDrawable(rhrBandwidthLabel.get());

        rhrBandwidthLabel->setColor(colorFR);
        rhrBandwidthLabel->setFont(font);
        rhrBandwidthLabel->setCharacterSize(characterSize);
        rhrBandwidthLabel->setPosition(pos);
        rhrBandwidthLabel->setText("MB/s:X", osgText::String::ENCODING_UTF8);
        pos.x() = rhrBandwidthLabel->getBound().xMax();
        rhrBandwidthLabel->setText("MB/s: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> rhrBandwidthValue = new osgText::Text;
        geode->addDrawable(rhrBandwidthValue.get());

        rhrBandwidthValue->setColor(colorFR);
        rhrBandwidthValue->setFont(font);
        rhrBandwidthValue->setCharacterSize(characterSize);
        rhrBandwidthValue->setPosition(pos);
        rhrBandwidthValue->setText("777.77", osgText::String::ENCODING_UTF8);

        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "RHR Bps", AccumAverageInverse, 1./1024/1024);
        rhrBandwidthValue->setDrawCallback(cb);

        pos.x() = rhrBandwidthValue->getBound().xMax();
        pos.x() += space;
    }

    // skipped remote frames
    {
        osg::Geode *geode = new osg::Geode();
        _rhrSkippedChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        geode->addDrawable(label.get());

        label->setColor(colorFR);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos);
        label->setText("Skipped/s:X", osgText::String::ENCODING_UTF8);
        pos.x() = label->getBound().xMax();
        label->setText("Skipped/s: ", osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text;
        geode->addDrawable(value.get());

        value->setColor(colorFR);
        value->setFont(font);
        value->setCharacterSize(characterSize);
        value->setPosition(pos);
        value->setText("77.77", osgText::String::ENCODING_UTF8);

        auto cb = new AveragedValueTextDrawCallback(viewer->getViewerStats(), "RHR Skipped Frames", AccumMax, 1./1024/1024);
        value->setDrawCallback(cb);

        pos.x() = value->getBound().xMax();
        pos.x() += space;
    }

    // next line
    pos.y() -= characterSize * 1.5f;

    osg::Vec4 backgroundColor(0.0, 0.0, 0.0f, 0.3);
    osg::Vec4 staticTextColor(1.0, 1.0, 0.0f, 1.0);
    osg::Vec4 dynamicTextColor(1.0, 1.0, 1.0f, 1.0);
    float backgroundMargin = 5;
    float backgroundSpacing = 3;

#define ADDBLOCK(viewerStats, stats, text, prefix, color) \
    { \
            pos.x() = leftPos; \
\
            osg::ref_ptr<osgText::Text> label = new osgText::Text; \
            geode->addDrawable(label.get()); \
\
            label->setColor(color); \
            label->setFont(font); \
            label->setCharacterSize(characterSize); \
            label->setPosition(pos); \
            label->setText(text, osgText::String::ENCODING_UTF8); \
\
            pos.x() = label->getBound().xMax(); \
\
            osg::ref_ptr<osgText::Text> value = new osgText::Text; \
            geode->addDrawable(value.get()); \
\
            value->setColor(color); \
            value->setFont(font); \
            value->setCharacterSize(characterSize); \
            value->setPosition(pos); \
            value->setText("0.0", osgText::String::ENCODING_UTF8); \
\
            value->setDrawCallback(new AveragedValueTextDrawCallback(stats, prefix " time taken", AccumAverage, 1000.0)); \
\
            pos.x() = startBlocks; \
            osg::Geometry *geometry = createGeometry(pos, characterSize * 0.8, color##Alpha, _numBlocks); \
            geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, prefix " begin time", prefix " end time", -1, _numBlocks)); \
            geode->addDrawable(geometry); \
\
            pos.x() = startBlocks; \
            osg::Geometry *geo = createGeometry(pos, characterSize*0.2, color, 1); \
            geo->setDrawCallback(new SlowestBlockDrawCallback(this, startBlocks, viewerStats, stats, prefix " begin time", prefix " end time")); \
            geode->addDrawable(geo); \
\
            pos.y() -= characterSize * 1.5f; \
}

    // viewer stats
    {
        osg::Group *group = new osg::Group;
        _viewerChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode *geode = new osg::Geode();
        group->addChild(geode);

        float topOfViewerStats = pos.y() + characterSize;

        geode->addDrawable(createBackgroundRectangle(
            pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
            _statsWidth - 2 * backgroundMargin,
            (3 + 4.5 * cameras.size()) * characterSize + 2 * backgroundMargin,
            backgroundColor));

        ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "COVER: ", "opencover", colorCover)
        ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "  Isect: ", "Isect", colorIsect)
        ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "  Plugins: ", "Plugin", colorPlugin)
#if 0
        ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "    Preframe: ", "preframe", colorUpdate)
#endif
        ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "Update: ", "Update traversal", colorUpdate)
        if (_finishStats)
        {
            ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "Finish: ", "finish", colorUpdate)
        }
        ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "Swap: ", "swap", colorSwap)
        if (_syncStats)
        {
            ADDBLOCK(viewer->getViewerStats(), viewer->getViewerStats(), "Sync: ", "sync", colorUpdate)
        }
        pos.x() = leftPos;

        // add camera stats
        for (osgViewer::ViewerBase::Cameras::iterator citr = cameras.begin();
             citr != cameras.end();
             ++citr)
        {
            group->addChild(createCameraTimeStats(font, pos, startBlocks, acquireGPUStats, characterSize, viewer->getViewerStats(), *citr));
        }

        // add frame ticks
        {
            osg::Geode *geode = new osg::Geode;
            group->addChild(geode);

            osg::Vec4 colourTicksAlpha(1.0f, 1.0f, 1.0f, 0.5f);
            osg::Vec4 colourTicks(1.0f, 1.0f, 1.0f, 1.0f);

            pos.x() = startBlocks;
            pos.y() += characterSize;
            float height = topOfViewerStats - pos.y();

            osg::Geometry *ticks = createTick(pos, 5.0f, colourTicksAlpha, 100);
            geode->addDrawable(ticks);

            osg::Geometry *frameMarkers = createFrameMarkers(pos, height, colourTicksAlpha, _numBlocks + 1);
            frameMarkers->setDrawCallback(new FrameMarkerDrawCallback(this, startBlocks, viewer->getViewerStats(), 0, _numBlocks + 1));
            geode->addDrawable(frameMarkers);

            pos.x() = leftPos;

            osg::Geometry *slowTick = createTick(pos, 5.0f, colourTicks, 100);
            geode->addDrawable(slowTick);

            osg::Geometry *slowFrameMarkers = createFrameMarkers(pos, height, colourTicks, 1);
            slowFrameMarkers->setDrawCallback(new SlowFrameMarkerDrawCallback(this, startBlocks, viewer->getViewerStats()));
            geode->addDrawable(slowFrameMarkers);

            pos.x() = leftPos;
        }

        const float MaxTime = 0.100;

        // Stats line graph
        {
            pos.y() -= (backgroundSpacing + 2 * backgroundMargin);
            float width = _statsWidth - 4 * backgroundMargin;
            float height = 10 * characterSize;

            // Create a stats graph and add any stats we want to track with it.
            StatsGraph *statsGraph = new StatsGraph(pos, width, height, 0);
            group->addChild(statsGraph);

            statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorFR, 5, MaxTime, "Frame duration", AccumNewest);
            //statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorMaxFR, 5, MaxTime*5.f, "Frame duration", AccumMax);
            //statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorIsect, 0, MaxTime, "Isect time taken", AccumNewest);
            statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorPlugin, 1, MaxTime, "Plugin time taken", AccumNewest);
            statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorCover, 1, MaxTime, "opencover time taken", AccumNewest);
            if (_syncStats)
                statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorSync, 2, MaxTime, "sync time taken", AccumNewest);
            statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorSwap, 3, MaxTime, "swap time taken", AccumNewest);
            if (_finishStats)
                statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorFinish, 4, MaxTime, "finish time taken", AccumNewest);

            for (osgViewer::ViewerBase::Cameras::iterator citr = cameras.begin();
                 citr != cameras.end();
                 ++citr)
            {
                statsGraph->addStatGraph(viewer->getViewerStats(), (*citr)->getStats(), colorCull, MaxTime, 0, "Cull traversal time taken", AccumNewest);
                statsGraph->addStatGraph(viewer->getViewerStats(), (*citr)->getStats(), colorDraw, MaxTime, 0, "Draw traversal time taken", AccumNewest);
                statsGraph->addStatGraph(viewer->getViewerStats(), (*citr)->getStats(), colorGPU, MaxTime, 0, "GPU draw time taken", AccumNewest);
            }

            geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, backgroundMargin, 0),
                                                         width + 2 * backgroundMargin,
                                                         height + 2 * backgroundMargin,
                                                         backgroundColor));

            pos.x() = leftPos;
            pos.y() -= height + 2 * backgroundMargin;
        }

        // Databasepager stats
        osgViewer::ViewerBase::Scenes scenes;
        viewer->getScenes(scenes);
        for (osgViewer::ViewerBase::Scenes::iterator itr = scenes.begin();
             itr != scenes.end();
             ++itr)
        {
            osgViewer::Scene *scene = *itr;
            osgDB::DatabasePager *dp = scene->getDatabasePager();
            if (dp && dp->isRunning())
            {
                pos.y() -= (characterSize + backgroundSpacing);

                geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                             _statsWidth - 2 * backgroundMargin,
                                                             characterSize + 2 * backgroundMargin,
                                                             backgroundColor));

                osg::ref_ptr<osgText::Text> averageLabel = new osgText::Text;
                geode->addDrawable(averageLabel.get());

                averageLabel->setColor(colorDP);
                averageLabel->setFont(font);
                averageLabel->setCharacterSize(characterSize);
                averageLabel->setPosition(pos);
                averageLabel->setText("DatabasePager time to merge new tiles - average: ", osgText::String::ENCODING_UTF8);

                pos.x() = averageLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> averageValue = new osgText::Text;
                geode->addDrawable(averageValue.get());

                averageValue->setColor(colorDP);
                averageValue->setFont(font);
                averageValue->setCharacterSize(characterSize);
                averageValue->setPosition(pos);
                averageValue->setText("1000", osgText::String::ENCODING_UTF8);

                pos.x() = averageValue->getBound().xMax() + 2.0f * characterSize;

                osg::ref_ptr<osgText::Text> minLabel = new osgText::Text;
                geode->addDrawable(minLabel.get());

                minLabel->setColor(colorDP);
                minLabel->setFont(font);
                minLabel->setCharacterSize(characterSize);
                minLabel->setPosition(pos);
                minLabel->setText("min: ", osgText::String::ENCODING_UTF8);

                pos.x() = minLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> minValue = new osgText::Text;
                geode->addDrawable(minValue.get());

                minValue->setColor(colorDP);
                minValue->setFont(font);
                minValue->setCharacterSize(characterSize);
                minValue->setPosition(pos);
                minValue->setText("1000", osgText::String::ENCODING_UTF8);

                pos.x() = minValue->getBound().xMax() + 2.0f * characterSize;

                osg::ref_ptr<osgText::Text> maxLabel = new osgText::Text;
                geode->addDrawable(maxLabel.get());

                maxLabel->setColor(colorDP);
                maxLabel->setFont(font);
                maxLabel->setCharacterSize(characterSize);
                maxLabel->setPosition(pos);
                maxLabel->setText("max: ", osgText::String::ENCODING_UTF8);

                pos.x() = maxLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> maxValue = new osgText::Text;
                geode->addDrawable(maxValue.get());

                maxValue->setColor(colorDP);
                maxValue->setFont(font);
                maxValue->setCharacterSize(characterSize);
                maxValue->setPosition(pos);
                maxValue->setText("1000", osgText::String::ENCODING_UTF8);

                pos.x() = maxValue->getBound().xMax();

                osg::ref_ptr<osgText::Text> requestsLabel = new osgText::Text;
                geode->addDrawable(requestsLabel.get());

                requestsLabel->setColor(colorDP);
                requestsLabel->setFont(font);
                requestsLabel->setCharacterSize(characterSize);
                requestsLabel->setPosition(pos);
                requestsLabel->setText("requests: ", osgText::String::ENCODING_UTF8);

                pos.x() = requestsLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> requestList = new osgText::Text;
                geode->addDrawable(requestList.get());

                requestList->setColor(colorDP);
                requestList->setFont(font);
                requestList->setCharacterSize(characterSize);
                requestList->setPosition(pos);
                requestList->setText("0", osgText::String::ENCODING_UTF8);

                pos.x() = requestList->getBound().xMax() + 2.0f * characterSize;
                ;

                osg::ref_ptr<osgText::Text> compileLabel = new osgText::Text;
                geode->addDrawable(compileLabel.get());

                compileLabel->setColor(colorDP);
                compileLabel->setFont(font);
                compileLabel->setCharacterSize(characterSize);
                compileLabel->setPosition(pos);
                compileLabel->setText("tocompile: ", osgText::String::ENCODING_UTF8);

                pos.x() = compileLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> compileList = new osgText::Text;
                geode->addDrawable(compileList.get());

                compileList->setColor(colorDP);
                compileList->setFont(font);
                compileList->setCharacterSize(characterSize);
                compileList->setPosition(pos);
                compileList->setText("0", osgText::String::ENCODING_UTF8);

                pos.x() = maxLabel->getBound().xMax();

                geode->setCullCallback(new PagerCallback(dp, minValue.get(), maxValue.get(), averageValue.get(), requestList.get(), compileList.get(), 1000.0));
            }

            pos.x() = leftPos;
        }
    }

    auto opos = pos;
    // another stats line for gpu stats
    {
        osg::Group *group = new osg::Group;
        _gpuChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode *geode = new osg::Geode();
        group->addChild(geode);

        // Another stats line graph for GPU stats
        pos.y() -= (backgroundSpacing + 2 * backgroundMargin);
        float width = _statsWidth - 4 * backgroundMargin;
        float height = 10 * characterSize;

        // Create a stats graph and add any stats we want to track with it.
        StatsGraph *statsGraph = new StatsGraph(pos, width, height, 0);
        group->addChild(statsGraph);

        statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorGpuClock, 0., 1., "GPU Clock Rate", AccumNewest);
        statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorGpuMemClock, 0., 1., "GPU Mem Clock Rate", AccumNewest);
        statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorGpuUtil, 0., 1., "GPU Utilization", AccumNewest);
        statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorGpuPCIe, 0., 4*1024*1024, "GPU PCIe rx KB/s", AccumNewest);

        geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, backgroundMargin, 0),
                                                     width + 2 * backgroundMargin,
                                                     height + 2 * backgroundMargin,
                                                     backgroundColor));

            pos.x() = leftPos;
            pos.y() -= height + 2 * backgroundMargin;
    }
    pos = opos;

    // Camera scene stats
    {
        pos.y() -= (characterSize + backgroundSpacing + 2 * backgroundMargin);

        osg::Group *group = new osg::Group;
        _cameraSceneChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode *geode = new osg::Geode();
        geode->setCullingActive(false);
        group->addChild(geode);
        geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                     7 * characterSize + 2 * backgroundMargin,
                                                     19 * characterSize + 2 * backgroundMargin,
                                                     backgroundColor));

        // Camera scene & primitive stats static text
        osg::ref_ptr<osgText::Text> camStaticText = new osgText::Text;
        geode->addDrawable(camStaticText.get());
        camStaticText->setColor(staticTextColor);
        camStaticText->setFont(font);
        camStaticText->setCharacterSize(characterSize);
        camStaticText->setPosition(pos);

        std::ostringstream viewStr;
        viewStr.clear();
        viewStr.setf(std::ios::left, std::ios::adjustfield);
        viewStr.width(14);
        viewStr << "Camera" << std::endl;
        viewStr << "" << std::endl; // placeholder for Camera name
        viewStr << "Lights" << std::endl;
        viewStr << "Bins" << std::endl;
        viewStr << "Depth" << std::endl;
        viewStr << "Matrices" << std::endl;
        viewStr << "Imposters" << std::endl;
        viewStr << "Drawables" << std::endl;
        viewStr << "Vertices" << std::endl;
        viewStr << "Points" << std::endl;
        viewStr << "Lines" << std::endl;
        viewStr << "Line strips" << std::endl;
        viewStr << "Line loops" << std::endl;
        viewStr << "Triangles" << std::endl;
        viewStr << "Tri. strips" << std::endl;
        viewStr << "Tri. fans" << std::endl;
        viewStr << "Quads" << std::endl;
        viewStr << "Quad strips" << std::endl;
        viewStr << "Polygons" << std::endl;
        viewStr.setf(std::ios::right, std::ios::adjustfield);
        camStaticText->setText(viewStr.str(), osgText::String::ENCODING_UTF8);

        // Move camera block to the right
        pos.x() += 7 * characterSize + 2 * backgroundMargin + backgroundSpacing;

        // Add camera scene stats, one block per camera
        int cameraCounter = 0;
        for (osgViewer::ViewerBase::Cameras::iterator citr = cameras.begin(); citr != cameras.end(); ++citr)
        {
            geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                         5 * characterSize + 2 * backgroundMargin,
                                                         19 * characterSize + 2 * backgroundMargin,
                                                         backgroundColor));

            // Camera scene stats
            osg::ref_ptr<osgText::Text> camStatsText = new osgText::Text;
            geode->addDrawable(camStatsText.get());

            camStatsText->setColor(dynamicTextColor);
            camStatsText->setFont(font);
            camStatsText->setCharacterSize(characterSize);
            camStatsText->setPosition(pos);
            camStatsText->setText("", osgText::String::ENCODING_UTF8);
            camStatsText->setDrawCallback(new CameraSceneStatsTextDrawCallback(*citr, cameraCounter));

            // Move camera block to the right
            pos.x() += 5 * characterSize + 2 * backgroundMargin + backgroundSpacing;
            cameraCounter++;
        }
    }

    // Viewer scene stats
    {
        osg::Group *group = new osg::Group;
        _viewerSceneChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode *geode = new osg::Geode();
        geode->setCullingActive(false);
        group->addChild(geode);

        geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                     6 * characterSize + 2 * backgroundMargin,
                                                     12 * characterSize + 2 * backgroundMargin,
                                                     backgroundColor));

        // View scene stats static text
        osg::ref_ptr<osgText::Text> camStaticText = new osgText::Text;
        geode->addDrawable(camStaticText.get());
        camStaticText->setColor(staticTextColor);
        camStaticText->setFont(font);
        camStaticText->setCharacterSize(characterSize);
        camStaticText->setPosition(pos);

        std::ostringstream viewStr;
        viewStr.clear();
        viewStr.setf(std::ios::left, std::ios::adjustfield);
        viewStr.width(14);
        viewStr << "View" << std::endl;
        viewStr << " " << std::endl;
        viewStr << "Stateset" << std::endl;
        viewStr << "Group" << std::endl;
        viewStr << "Transform" << std::endl;
        viewStr << "LOD" << std::endl;
        viewStr << "Switch" << std::endl;
        viewStr << "Geode" << std::endl;
        viewStr << "Drawable" << std::endl;
        viewStr << "Geometry" << std::endl;
        viewStr << "Vertices" << std::endl;
        viewStr << "Primitives" << std::endl;
        viewStr.setf(std::ios::right, std::ios::adjustfield);
        camStaticText->setText(viewStr.str(), osgText::String::ENCODING_UTF8);

        // Move viewer block to the right
        pos.x() += 6 * characterSize + 2 * backgroundMargin + backgroundSpacing;

        std::vector<osgViewer::View *> views;
        viewer->getViews(views);

        std::vector<osgViewer::View *>::iterator it;
        int viewCounter = 0;
        for (it = views.begin(); it != views.end(); ++it)
        {
            geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                         10 * characterSize + 2 * backgroundMargin,
                                                         12 * characterSize + 2 * backgroundMargin,
                                                         backgroundColor));

            // Text for scene statistics
            osgText::Text *text = new osgText::Text;
            geode->addDrawable(text);

            text->setColor(dynamicTextColor);
            text->setFont(font);
            text->setCharacterSize(characterSize);
            text->setPosition(pos);
            text->setDrawCallback(new ViewSceneStatsTextDrawCallback(*it, viewCounter));

            pos.x() += 10 * characterSize + 2 * backgroundMargin + backgroundSpacing;
            viewCounter++;
        }
    }
}

osg::Node *coVRStatsDisplay::createCameraTimeStats(const std::string &font, osg::Vec3 &pos, float startBlocks, bool acquireGPUStats, float characterSize, osg::Stats *viewerStats, osg::Camera *camera)
{
    osg::Stats *stats = camera->getStats();
    if (!stats)
        return 0;

    osg::Group *group = new osg::Group;

    osg::Geode *geode = new osg::Geode();
    group->addChild(geode);

    float leftPos = pos.x();

    osg::Vec4 colorCull(0.0f, 1.0f, 1.0f, 1.0f);
    osg::Vec4 colorCullAlpha(0.0f, 1.0f, 1.0f, 0.5f);
    osg::Vec4 colorDraw(1.0f, 1.0f, 0.0f, 1.0f);
    osg::Vec4 colorDrawAlpha(1.0f, 1.0f, 0.0f, 0.5f);
    osg::Vec4 colorGPU(1.0f, 0.5f, 0.0f, 1.0f);
    osg::Vec4 colorGPUAlpha(1.0f, 0.5f, 0.0f, 0.5f);

    ADDBLOCK(viewerStats, stats, "Cull: ", "Cull traversal", colorCull)
    ADDBLOCK(viewerStats, stats, "Draw: ", "Draw traversal", colorDraw)

    if (acquireGPUStats)
    {
        ADDBLOCK(viewerStats, stats, "GPU: ", "GPU draw", colorGPU)
    }

    pos.x() = leftPos;

    return group;
}

void coVRStatsDisplay::getUsage(osg::ApplicationUsage &usage) const
{
    usage.addKeyboardMouseBinding("s", "On screen stats.");
    usage.addKeyboardMouseBinding("S", "Output stats to console.");
}
