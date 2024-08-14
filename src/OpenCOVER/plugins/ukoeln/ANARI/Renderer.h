/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <memory>
#include <vector>
#include <thread>
#ifdef ANARI_PLUGIN_HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <osg/BoundingSphere>
#include <osg/Geometry>
#include <cover/coVRLighting.h>
#include <PluginUtil/MultiChannelDrawer.h>
#include <anari/anari.h>
#include <anari/anari_cpp/ext/glm.h>
#include "asg.h"
#ifdef HAVE_HDF5
#include "readFlash.h"
#endif
#include "readUMesh.h"
#include "readVTK.h"
#include "ui_anari.h"
#include "Projection.h"
#ifdef ANARI_PLUGIN_HAVE_RR
#include <MiniRR.h>
#endif

class Renderer
{
public:
    typedef std::shared_ptr<Renderer> SP;

    typedef glm::vec4 ClipPlane;

    struct Light
    {
        explicit Light(const opencover::coVRLighting::Light &l) : coLight(l), updated(true)
        {}

        osg::Vec4 prevPos; // subject to the previous light node xform!
        opencover::coVRLighting::Light coLight;
        bool updated{false};
    };

    Renderer();
   ~Renderer();

    void init();

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
                   const uint8_t *type, const float *vertexData, size_t numCells, size_t numIndices,
                   size_t numVerts, float minValue = 0.f, float maxValue = 1.f);
    void unloadUMesh();

    // .umesh file format:
    void loadUMeshFile(std::string fileName);
    void unloadUMeshFile(std::string fileName);

    // (optional) scalar data for umeshes
    void loadUMeshScalars(std::string fileName);
    void unloadUMeshScalars(std::string fileName);

    void loadUMeshVTK(std::string fileName);
    void unloadUMeshVTK(std::string fileName);
    
    void loadPointCloud(std::string fileName);
    void unloadPointCloud(std::string fileName);

    void loadHDRI(std::string fileName);
    void unloadHDRI(std::string fileName);

    void setRendererType(std::string type);

    std::vector<std::string> getRendererTypes();

    std::vector<ui_anari::ParameterList> &getRendererParameters();

    void setParameter(std::string name, bool value);
    void setParameter(std::string name, int value);
    void setParameter(std::string name, float value);

    void expandBoundingSphere(osg::BoundingSphere &bs);

    void updateLights(const osg::Matrix &modelMat);

    void setClipPlanes(const std::vector<ClipPlane> &planes);

    // volume debug mode where MPI rank IDs are assigned random colors
    void setColorRanks(bool value);

    void wait();

    void renderFrame();
    void renderFrame(unsigned chan);

    void drawFrame();
    void drawFrame(unsigned chan);

    int mpiRank{0};
    int mpiSize{1};
    int mainRank{0};
    int displayRank{0};

private:
    osg::ref_ptr<opencover::MultiChannelDrawer> multiChannelDrawer{nullptr};
    struct ChannelInfo {
        struct {
            int width=1, height=1;
            GLenum colorFormat=GL_FLOAT;
            GLenum depthFormat=GL_UNSIGNED_BYTE;
            bool resized=false;
            bool updated=false;
        } frame;
        glm::mat4 mm, mv, pr;
        glm::vec3 eye, dir, up;
        float fovy, aspect;
        glm::box2 imgRegion;
    };
    std::vector<ChannelInfo> channelInfos;
    int numChannels{0};

    struct {
        std::string libtype = "environment";
        std::string devtype = "default";
        std::string renderertype = "default";
        ANARILibrary library{nullptr};
        ANARIDevice device{nullptr};
        ANARIRenderer renderer{nullptr};
        ANARIWorld world{nullptr};
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
        std::vector<ANARILight> lights;
        std::vector<ANARICamera> cameras;
        std::vector<ANARIFrame> frames;
        struct {
          bool enabled{false};
#ifdef ANARI_PLUGIN_HAVE_CUDA
          cudaStream_t copyStream{0};
#endif
        } cudaInterop;
    } anari;

    void initMPI();
    void initRR();
    void initChannels();
    void initDevice();
    void initFrames();
    void initWorld();
    void initMesh();
    void initPointClouds();
    void initStructuredVolume();
    void initAMRVolume();
    void initUnstructuredVolume();
    void initClipPlanes();
    void initHDRI();
    void initTransFunc();

    enum ReaderType { FLASH, VTK, UMESH, UNKNOWN };

    bool colorByRank{false};

    void generateTransFunc();

    struct AABB {
        AABB() {
            data[0] =  1e30f;
            data[1] =  1e30f;
            data[2] =  1e30f;
            data[3] = -1e30f;
            data[4] = -1e30f;
            data[5] = -1e30f;
        }

        AABB &extend(const AABB &other) {
            data[0] = fminf(data[0], other.data[0]);
            data[1] = fminf(data[1], other.data[1]);
            data[2] = fminf(data[2], other.data[2]);
            data[3] = fmaxf(data[3], other.data[3]);
            data[4] = fmaxf(data[4], other.data[4]);
            data[5] = fmaxf(data[5], other.data[5]);
            return *this;
        }
        float data[6];
    };
    struct {
        AABB local;
        AABB global;
        bool updated = false;
    } bounds;

    struct {
        std::string fileName;
        bool updated = false;
    } meshData;

    struct {
        std::vector<std::string> fileNames;
        bool updated = false;
    } pointCloudData;

    struct {
        std::string fileName;

        const void *data;
        int sizeX, sizeY, sizeZ;
        int bpc;
        float minValue, maxValue;

        std::vector<float> voxels;
        std::vector<float> rgbLUT;
        std::vector<float> alphaLUT;

        bool updated = false;
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

        bool updated = false;
    } amrVolumeData;
#endif

    struct {
        std::string fileName;
        ReaderType readerType{UNKNOWN};
#ifdef HAVE_VTK
        VTKReader vtkReader;
#endif
#ifdef HAVE_UMESH
        UMeshReader umeshReader;
        typedef struct { std::string fileName; int fieldID; } UMeshScalarFile;
        // *optional* list of file names providing umesh scalars; in this case,
        // these files overwrite the umesh's perVertex (if it exists)
        std::vector<UMeshScalarFile> umeshScalarFiles;
#endif

        UnstructuredField data;
        float minValue, maxValue;
        std::vector<float> rgbLUT;
        std::vector<float> alphaLUT;

        bool updated = false;
    } unstructuredVolumeData;

    struct {
        std::vector<glm::vec3> colors;
        std::vector<float> opacities;
        bool updated = false;
    } transFunc;

    struct {
        std::vector<Light> data;

        bool updated = false;
    } lights;

    struct {
        std::vector<ClipPlane> data;

        bool updated = false;
    } clipPlanes;

    struct {
        std::vector<glm::vec3> pixels;
        unsigned width, height;

        bool updated = false;
    } hdri;

    std::vector<std::string> rendererTypes;
    std::vector<ui_anari::ParameterList> rendererParameters;

#ifdef ANARI_PLUGIN_HAVE_RR
    // thread to process events on that aren't executed in lockstep
    std::thread remoteThread;
    std::shared_ptr<minirr::MiniRR> rr;
#endif
    std::vector<uint32_t> imageBuffer;
    bool isClient{false};
    bool isServer{false};
};


