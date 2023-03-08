/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <memory>
#include <vector>
#include <osg/BoundingSphere>
#include <osg/Geometry>
#include <PluginUtil/MultiChannelDrawer.h>
#include <anari/anari.h>
#include "asg.h"

class Renderer
{
public:
    typedef std::shared_ptr<Renderer> SP;

    Renderer();
   ~Renderer();

    void loadScene(std::string fileName);
    void unloadScene(std::string fileName);

    void loadVolumeRAW(std::string fileName);
    void unloadVolumeRAW(std::string fileName);

    void loadVolume(const void *data, int sizeX, int sizeY, int sizeZ, int bpc,
                    float minValue = 0.f, float maxValue = 1.f);
    void unloadVolume();

    void setRendererType(std::string type);

    std::vector<std::string> getRendererTypes();

    void expandBoundingSphere(osg::BoundingSphere &bs);

    void renderFrame(osg::RenderInfo &info);
    void renderFrame(osg::RenderInfo &info, unsigned chan);

private:
    osg::ref_ptr<opencover::MultiChannelDrawer> multiChannelDrawer{nullptr};
    struct ChannelInfo {
        int width=1, height=1;
        GLenum colorFormat=GL_FLOAT;
        GLenum depthFormat=GL_UNSIGNED_BYTE;
        osg::Matrix mv, pr;
    };
    std::vector<ChannelInfo> channelInfos;

    struct {
        std::string libtype = "environment";
        std::string devtype = "default";
        std::string renderertype = "default";
        ANARILibrary library{nullptr};
        ANARIDevice device{nullptr};
        ANARIRenderer renderer{nullptr};
        ANARIWorld world{nullptr};
        ANARILight headLight{nullptr};
        ASGObject root{nullptr};
        ASGObject meshes{nullptr};
        ASGStructuredVolume volume{nullptr};
        ASGLookupTable1D lut{nullptr};
        std::vector<ANARICamera> cameras;
        std::vector<ANARIFrame> frames;
    } anari;

    void initDevice();
    void initFrames();
    void initScene();
    void initVolume();

    struct {
        std::string value;
        bool changed = false;
    } fileName;

    struct {
        const void *data;
        int sizeX, sizeY, sizeZ;
        int bpc;
        float minValue, maxValue;

        std::vector<float> voxels;
        std::vector<float> rgbLUT;
        std::vector<float> alphaLUT;

        bool changed = false;
        bool deleteData = false;
    } volumeData;
};


