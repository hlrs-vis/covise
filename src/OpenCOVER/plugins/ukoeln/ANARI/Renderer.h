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
#ifdef HAVE_HDF5
#include "readFlash.h"
#endif
#include "readVTK.h"

class Renderer
{
public:
    typedef std::shared_ptr<Renderer> SP;

    Renderer();
   ~Renderer();

    void loadMesh(std::string fileName);
    void unloadMesh(std::string fileName);

    void loadVolumeRAW(std::string fileName);
    void unloadVolumeRAW(std::string fileName);

    void loadVolume(const void *data, int sizeX, int sizeY, int sizeZ, int bpc,
                    float minValue = 0.f, float maxValue = 1.f);
    void unloadVolume();

    void loadFLASH(std::string fileName);
    void unloadFLASH(std::string fileName);

    void loadUMesh(const float *vertexPosition, const uint64_t *cellIndex, const uint64_t *index,
                   const float *vertexData, size_t numCells, size_t numIndices, size_t numVerts,
                   float minValue = 0.f, float maxValue = 1.f);
    void unloadUMesh();

    void loadUMeshVTK(std::string fileName);
    void unloadUMeshVTK(std::string fileName);

    void setRendererType(std::string type);

    std::vector<std::string> getRendererTypes();

    void setPixelSamples(int spp);

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
        ASGStructuredVolume structuredVolume{nullptr};
        struct {
            ANARIVolume volume{nullptr};
            ANARISpatialField field{nullptr};
        } amrVolume;
        struct {
            ANARIVolume volume{nullptr};
            ANARISpatialField field{nullptr};
        } unstructuredVolume;
        ASGLookupTable1D lut{nullptr};
        std::vector<ANARICamera> cameras;
        std::vector<ANARIFrame> frames;
    } anari;

    void initDevice();
    void initFrames();
    void initMesh();
    void initStructuredVolume();
    void initAMRVolume();
    void initUnstructuredVolume();

    struct {
        std::string fileName;
        bool changed = false;
    } meshData;

    struct {
        std::string fileName;

        const void *data;
        int sizeX, sizeY, sizeZ;
        int bpc;
        float minValue, maxValue;

        std::vector<float> voxels;
        std::vector<float> rgbLUT;
        std::vector<float> alphaLUT;

        bool changed = false;
        bool deleteData = false;
    } structuredVolumeData;

#ifdef HAVE_HDF5
    struct {
        std::string fileName;
        FlashReader flashReader;

        AMRField data;
        float minValue, maxValue;
        std::vector<float> rgbLUT;
        std::vector<float> alphaLUT;

        bool changed = false;
    } amrVolumeData;
#endif

    struct {
        std::string fileName;
#ifdef HAVE_VTK
        VTKReader vtkReader;
#endif

        UnstructuredField data;
        float minValue, maxValue;
        std::vector<float> rgbLUT;
        std::vector<float> alphaLUT;

        bool changed = false;
    } unstructuredVolumeData;
    
    int spp{1};
};


