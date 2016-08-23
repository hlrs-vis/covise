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
    osg::Matrix curProj, curView, curModel;
    osg::Matrix newProj, newView, newModel;

    // geometry for mapping depth image
    osg::ref_ptr<osg::TextureRectangle> colorTex;
    osg::ref_ptr<osg::TextureRectangle> depthTex;
    osg::ref_ptr<osg::Vec2Array> texcoord;
    osg::ref_ptr<osg::Geometry> fixedGeo;
    osg::ref_ptr<osg::Geometry> reprojGeo;
    osg::ref_ptr<osg::Vec2Array> coord;
    osg::ref_ptr<osg::Uniform> size;
    osg::ref_ptr<osg::Uniform> pixelOffset;
    osg::ref_ptr<osg::Uniform> withNeighbors;
    osg::ref_ptr<osg::Uniform> reprojMat;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::MatrixTransform> scene;
    osg::ref_ptr<osg::Camera> camera;
    osg::ref_ptr<osg::Program> reprojConstProgram;
    osg::ref_ptr<osg::Program> reprojAdaptProgram;

    ChannelData(int channel=-1)
        : channelNum(channel)
        , second(false)
    {
    }
};

class PLUGIN_UTILEXPORT MultiChannelDrawer: public osg::Camera {
public:
   typedef opencover::ChannelData ChannelData;

    MultiChannelDrawer(int numChannels, bool flipped=false);
    ~MultiChannelDrawer();

   void switchReprojection(bool reproj);
   void switchAdaptivePointSize(bool adapt, bool withNeighbors=true);
   void initChannelData(ChannelData &cd);
   void createGeometry(ChannelData &cd);
   void clearChannelData();
   void swapFrame();
   void updateMatrices(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj);
   void resizeView(int idx, int w, int h, GLenum depthFormat=0);
   void reproject(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj);
   unsigned char *rgba(int idx) const;
   unsigned char *depth(int idx) const;

   std::vector<ChannelData> m_channelData;
   bool m_flipped;
};

}
#endif
