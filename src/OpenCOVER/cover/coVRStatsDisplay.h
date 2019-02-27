/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_coVRStatsDisplay_h
#define CO_coVRStatsDisplay_h

#include <util/coTypes.h>
#include <osg/AnimationPath>
#include <osgText/Text>
#include <osgGA/GUIEventHandler>
#include <osgGA/AnimationPathManipulator>

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>

//#include <osgDB/fstream>
namespace opencover
{
class COVEREXPORT coVRStatsDisplay
{
    friend class VRViewer;
public:
    coVRStatsDisplay();
    virtual ~coVRStatsDisplay(){};

    enum StatsType
    {
        NO_STATS = 0,
        FRAME_RATE = 1,
        VIEWER_STATS = 2,
        CAMERA_SCENE_STATS = 3,
        VIEWER_SCENE_STATS = 4,
        LAST = 5
    };

    double getBlockMultiplier() const
    {
        return _blockMultiplier;
    }

    void reset();

    osg::Camera *getCamera()
    {
        return _camera.get();
    }
    const osg::Camera *getCamera() const
    {
        return _camera.get();
    }

    void showStats(int whichStats, osgViewer::ViewerBase *viewer);

    void enableGpuStats(bool enable);
    void enableRhrStats(bool enable);
    void enableFinishStats(bool enable);
    void enableSyncStats(bool enable);

    /** Get the keyboard and mouse usage of this manipulator.*/
    virtual void getUsage(osg::ApplicationUsage &usage) const;

protected:
    void setUpHUDCamera(osgViewer::ViewerBase *viewer);

    osg::Geometry *createBackgroundRectangle(const osg::Vec3 &pos, const float width, const float height, osg::Vec4 &color);

    osg::Geometry *createGeometry(const osg::Vec3 &pos, float height, const osg::Vec4 &colour, unsigned int numBlocks);

    osg::Geometry *createFrameMarkers(const osg::Vec3 &pos, float height, const osg::Vec4 &colour, unsigned int numBlocks);

    osg::Geometry *createTick(const osg::Vec3 &pos, float height, const osg::Vec4 &colour, unsigned int numTicks);

    osg::Node *createCameraTimeStats(const std::string &font, osg::Vec3 &pos, float startBlocks, bool acquireGPUStats, float characterSize, osg::Stats *viewerStats, osg::Camera *camera);

    void setUpScene(osgViewer::ViewerBase *viewer);

    void updateThreadingModelText(osgViewer::ViewerBase::ThreadingModel tm);

    int _statsType;

    bool _initialized;
    osg::ref_ptr<osg::Camera> _camera;

    osg::ref_ptr<osg::Switch> _switch;

    osgViewer::ViewerBase::ThreadingModel _threadingModel;
    osg::ref_ptr<osgText::Text> _threadingModelText;

    bool _finishStats = true;
    bool _syncStats = true;
    bool _gpuStats = false;
    bool _rhrStats = false;
    unsigned int _frameRateChildNum;
    unsigned int _gpuMemChildNum;
    unsigned int _rhrFpsChildNum;
    unsigned int _rhrBandwidthChildNum;
    unsigned int _rhrDelayChildNum;
    unsigned int _viewerChildNum;
    unsigned int _cameraSceneChildNum;
    unsigned int _viewerSceneChildNum;
    unsigned int _numBlocks;
    double _blockMultiplier;

    float _statsWidth;
    float _statsHeight;
};
}
#endif
