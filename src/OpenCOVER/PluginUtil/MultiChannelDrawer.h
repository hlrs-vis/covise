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

namespace opencover
{

//! store data associated with one channel
struct ChannelData {
    int channelNum;
    bool second;
    int frameNum;
    int width;
    int height;
    int depthWidth=0, depthHeight=0;
    GLenum colorFormat;
    GLenum depthFormat;
    osg::Matrix curProj, curView, curModel;
    osg::Matrix imgProj, imgView, imgModel;
    osg::Matrix newProj, newView, newModel;

    // geometry for mapping depth image
    osg::ref_ptr<osg::TextureRectangle> colorTex;
    osg::ref_ptr<osg::TextureRectangle> depthTex;
    osg::ref_ptr<osg::Vec2Array> texcoord;
    osg::ref_ptr<osg::Geometry> fixedGeo;
    osg::ref_ptr<osg::Geometry> reprojGeo;
    osg::ref_ptr<osg::Geometry> meshGeo;
    osg::ref_ptr<osg::Vec2Array> pointCoord, quadCoord;
    osg::ref_ptr<osg::DrawArrays> pointArr, quadArr;
    osg::ref_ptr<osg::Uniform> size;
    osg::ref_ptr<osg::Uniform> pixelOffset;
    osg::ref_ptr<osg::Uniform> withNeighbors;
    osg::ref_ptr<osg::Uniform> withHoles;
    osg::ref_ptr<osg::Uniform> reprojMat;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::MatrixTransform> scene;
    osg::ref_ptr<osg::Camera> camera;
    osg::ref_ptr<osg::Program> reprojConstProgram;
    osg::ref_ptr<osg::Program> reprojAdaptProgram;
    osg::ref_ptr<osg::Program> reprojMeshProgram;

    ChannelData(int channel=-1)
        : channelNum(channel)
        , second(false)
        , frameNum(0)
        , width(0)
        , height(0)
        , colorFormat(0)
        , depthFormat(0)
    {
    }
};

class PLUGIN_UTILEXPORT MultiChannelDrawer: public osg::Camera {
public:
   typedef opencover::ChannelData ChannelData;

    MultiChannelDrawer(bool flipped=false, bool useCuda=false);
    ~MultiChannelDrawer();
    int numViews() const;
    void update(); //! to be called each frame, updates current matrices
    const osg::Matrix &modelMatrix(int idx) const;
    const osg::Matrix &viewMatrix(int idx) const;
    const osg::Matrix &projectionMatrix(int idx) const;

    //! render mode
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

   //! from now on, draw with current RGBA and depth data for all views
   void swapFrame();
   //! set matrices corresponding to RGBA and depth data for view idx
   void updateMatrices(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj);
   //! resize view idx
   void resizeView(int idx, int w, int h, GLenum depthFormat=0, GLenum colorFormat=GL_UNSIGNED_BYTE);
   //! set matrices for which view idx shall be reprojected (mode != AsIs)
   void reproject(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj);
   //! set matrices from COVER for all views
   void reproject();
   //! access RGBA data for view idx
   unsigned char *rgba(int idx) const;
   //! access depth data for view idx
   unsigned char *depth(int idx) const;
   //! fill color array with all zeros
   void clearColor(int idx);
   //! fill depth array with all ones
   void clearDepth(int idx);

private:
   void initChannelData(ChannelData &cd);
   void createGeometry(ChannelData &cd);
   void clearChannelData();
   std::vector<ChannelData> m_channelData;
   bool m_flipped;
   Mode m_mode;

   const bool m_useCuda;
};

}
#endif
