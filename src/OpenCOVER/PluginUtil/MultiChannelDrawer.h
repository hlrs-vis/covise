/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MULTICHANNELDRAWER_H
#define MULTICHANNELDRAWER_H

#include <osg/Matrix>
#include <osg/Array>
#include <osg/Geode>
#include <osg/Camera>
#include <osg/TextureRectangle>

#include <util/coExport.h>

#include <memory>
#include <vector>

namespace opencover
{

class MultiChannelDrawer;
struct ViewChannelData;

enum ViewEye {
    Middle,
    Left,
    Right,
    Invalid,
};

// Store data associated with one view (image rendered for a viewport)
struct ViewData {
    static const int NumImages = 1;

    ViewEye eye = Invalid;
    int viewNum = -1;
    int geoWidth = 0, geoHeight = 0;
    int width[NumImages], height[NumImages];
    int depthWidth[NumImages], depthHeight[NumImages];
    GLenum colorFormat[NumImages], depthFormat[NumImages];
    osg::Matrix imgProj, imgView, imgModel;
    osg::Matrix newProj, newView, newModel;

    // geometry for mapping depth image
    osg::ref_ptr<osg::TextureRectangle> colorTex;
    osg::ref_ptr<osg::TextureRectangle> depthTex;
    osg::ref_ptr<osg::Image> colorImg[NumImages];
    osg::ref_ptr<osg::Image> depthImg[NumImages];
    osg::ref_ptr<osg::Vec2Array> texcoord;
    osg::ref_ptr<osg::Geometry> fixedGeo;
    osg::ref_ptr<osg::Geometry> reprojGeo;
    osg::ref_ptr<osg::Vec2Array> pointCoord, quadCoord;
    osg::ref_ptr<osg::DrawArrays> pointArr, quadArr;
    osg::ref_ptr<osg::Uniform> size;
    osg::ref_ptr<osg::Uniform> pixelOffset;
    osg::ref_ptr<osg::Uniform> withNeighbors;
    osg::ref_ptr<osg::Uniform> withHoles;
    osg::ref_ptr<osg::Program> reprojConstProgram;
    osg::ref_ptr<osg::Program> reprojAdaptProgram;
    osg::ref_ptr<osg::Program> reprojMeshProgram;

    std::vector<ViewChannelData *> viewChan;

    ViewData(int view=-1)
        : viewNum(view)
    {
        for (int i=0; i<NumImages; ++i) {
            width[i] = 0;
            height[i] = 0;
            depthWidth[i] = 0;
            depthHeight[i] = 0;
            colorFormat[i] = 0;
            depthFormat[i] = 0;
        }
    }

    ~ViewData();
};

//! store data associated with one channel (output viewport)
struct ChannelData {
    MultiChannelDrawer *drawer = nullptr;
    int channelNum = -1;
    ViewEye eye = Middle;
    bool second;
    int width;
    int height;
    osg::Matrix curProj, curView, curModel;

    // geometry for mapping depth image
    osg::ref_ptr<osg::Camera> camera;
    osg::ref_ptr<osg::Group> scene;

    std::vector<std::shared_ptr<ViewChannelData>> viewChan;

    ChannelData(MultiChannelDrawer *drawer, int channel)
        : drawer(drawer)
        , channelNum(channel)
        , eye(Middle)
        , second(false)
        , width(0)
        , height(0)
    {
    }

    ~ChannelData();

    void addView(std::shared_ptr<ViewData> vd);
    void enableView(std::shared_ptr<ViewData> vd, bool enable);
    void clearViews();
    void updateViews();
};

//! data for rendering a View into a Channel
struct ViewChannelData {

    ViewChannelData(std::shared_ptr<ViewData> view, ChannelData *chan);
    ~ViewChannelData();
    void setThis(std::shared_ptr<ViewChannelData> vcd);

    osg::ref_ptr<osg::Geometry> fixedGeo;
    osg::ref_ptr<osg::Geometry> reprojGeo;
    osg::ref_ptr<osg::Uniform> reprojMat;
    osg::ref_ptr<osg::StateSet> state;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Drawable::DrawCallback> drawCallback;

    ChannelData *chan = nullptr;
    std::shared_ptr<ViewData> view;

    void update();

};

class PLUGIN_UTILEXPORT MultiChannelDrawer: public osg::Camera {
    friend struct SingleScreenCB;
public:
   typedef opencover::ChannelData ChannelData;
   typedef opencover::ViewData ViewData;
   typedef opencover::ViewEye ViewEye;

    MultiChannelDrawer(bool flipped=false, bool useCuda=false);
    ~MultiChannelDrawer();
    int numViews() const;
    void update(); //! to be called each frame, updates current matrices
    const osg::Matrix &modelMatrix(int idx) const;
    const osg::Matrix &viewMatrix(int idx) const;
    const osg::Matrix &projectionMatrix(int idx) const;

    //! reprojection mode
    enum Mode {
        AsIs, //< as is, without reprojection
        Reproject, //< reproject every pixel as single pixel-sized point
        ReprojectAdaptive, //< reproject pixels and adapt their size based on viewer distance and reprojection matrix
        ReprojectAdaptiveWithNeighbors, //< reproject pixels and adapt their size so that gaps to neighbor pixels are filled
        ReprojectMesh, //< reproject as rectilinear mesh
        ReprojectMeshWithHoles //< reprjoct as rectilinear mesh, but keep holes where pixels become heavily deformed
   };
   Mode mode() const;
   void setMode(Mode mode);

   enum ViewSelection {
       Same, //< only render the view created for this viewport
       MatchingEye, //< render all views for the same (left/right) eye
       All, //< render all views
   };

   //! whether all available views should be rendered
   ViewSelection viewsToRender() const;
   //! return whether all available views should be rendered
   void setViewsToRender(ViewSelection views);
   //! set number of views to render, -1: one view/channel/stereo eye
   void setNumViews(int nv=-1);
   //! configure for which eye this a view is intended
   void setViewEye(int view, ViewEye eye);

   //! from now on, draw with current RGBA and depth data for all views
   void swapFrame();
   //! set matrices corresponding to RGBA and depth data for view idx
   void updateMatrices(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj);
   //! resize view idx
   void resizeView(int idx, int w, int h, GLenum depthFormat=0, GLenum colorFormat=GL_UNSIGNED_BYTE);
   //! get a pointer that can be retained
   std::shared_ptr<ViewData> getViewData(int idx);
   //! access RGBA data for view idx
   unsigned char *rgba(int idx) const;
   //! access depth data for view idx
   unsigned char *depth(int idx) const;
   //! fill color array with all zeros
   void clearColor(int idx);
   //! fill depth array with all ones
   void clearDepth(int idx);

private:
   void initViewData(ViewData &cd);
   void initChannelData(ChannelData &cd);
   void createGeometry(ViewData &cd);
   void createGeometry(ChannelData &cd);
   void clearViewData();
   std::vector<std::shared_ptr<ViewData>> m_viewData;
   std::vector<std::shared_ptr<ChannelData>> m_channelData;
   bool m_flipped;
   Mode m_mode;
   ViewSelection m_viewsToRender = Same;
   int renderTex=0, writeTex=0;
   struct AvailableEyes {
       bool middle=false;
       bool left=false;
       bool right=false;
   };
   AvailableEyes m_availableEyes;
   bool haveEye(ViewEye eye);

   const bool m_useCuda;
   void updateGeoForView(ViewData &vd);
   int m_numViews = 0;
};

}
#endif
