/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdio>
#include <iostream>
#include <sstream>
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/glm.h>
#include <osg/io_utils>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#include "Projection.h"
#include "Renderer.h"

using namespace covise;
using namespace opencover;


void statusFunc(const void *userData,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
    (void)userData;
    if (severity == ANARI_SEVERITY_FATAL_ERROR)
        fprintf(stderr, "[FATAL] %s\n", message);
    else if (severity == ANARI_SEVERITY_ERROR)
        fprintf(stderr, "[ERROR] %s\n", message);
    else if (severity == ANARI_SEVERITY_WARNING)
        fprintf(stderr, "[WARN ] %s\n", message);
    else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
        fprintf(stderr, "[PERF ] %s\n", message);
    else if (severity == ANARI_SEVERITY_INFO)
        fprintf(stderr, "[INFO] %s\n", message);
}

static std::string getExt(const std::string &fileName)
{
    int pos = fileName.rfind('.');
    if (pos == fileName.npos)
        return "";
    return fileName.substr(pos);
}

static std::vector<std::string> string_split(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}



Renderer::Renderer()
{
    initDevice();

    if (!anari.device)
        throw std::runtime_error("Could not init ANARI device");
}

Renderer::~Renderer()
{
    cover->getScene()->removeChild(multiChannelDrawer);
}

void Renderer::loadMesh(std::string fn)
{
    // deferred!
    meshData.fileName = fn;
    meshData.changed = true;
}

void Renderer::unloadMesh(std::string fn)
{
    // NO!
}

void Renderer::loadVolume(const void *data, int sizeX, int sizeY, int sizeZ, int bpc,
                          float minValue, float maxValue)
{
    // deferred!
    structuredVolumeData.data = data;
    structuredVolumeData.sizeX = sizeX;
    structuredVolumeData.sizeY = sizeY;
    structuredVolumeData.sizeZ = sizeZ;
    structuredVolumeData.bpc = bpc;
    structuredVolumeData.minValue = minValue;
    structuredVolumeData.maxValue = maxValue;
    structuredVolumeData.changed = true;
}

void Renderer::unloadVolume()
{
    // NO!
}

void Renderer::loadFLASH(std::string fn)
{
#ifdef HAVE_HDF5
    // deferred!
    amrVolumeData.fileName = fn;
    amrVolumeData.changed = true;

    if (amrVolumeData.flashReader.open(fn.c_str())) {
        amrVolumeData.data = amrVolumeData.flashReader.getField(0);
    }
#endif
}

void Renderer::unloadFLASH(std::string fn)
{
    // NO!
}

void Renderer::loadUMesh(const float *vertexPosition, const uint64_t *cellIndex, const uint64_t *index,
                         const uint8_t *cellType, const float *vertexData, size_t numCells, size_t numIndices,
                         size_t numVerts, float minValue, float maxValue)
{
    // deferred!
    unstructuredVolumeData.data.vertexPosition.resize(numVerts*3);
    memcpy(unstructuredVolumeData.data.vertexPosition.data(),vertexPosition,numVerts*3*sizeof(float));

    unstructuredVolumeData.data.cellIndex.resize(numCells);
    memcpy(unstructuredVolumeData.data.cellIndex.data(),cellIndex,numCells*sizeof(uint64_t));

    unstructuredVolumeData.data.index.resize(numIndices);
    memcpy(unstructuredVolumeData.data.index.data(),index,numIndices*sizeof(uint64_t));

    unstructuredVolumeData.data.cellType.resize(numCells);
    memcpy(unstructuredVolumeData.data.cellType.data(),cellType,numCells*sizeof(uint64_t));

    unstructuredVolumeData.data.vertexData.resize(numVerts);
    memcpy(unstructuredVolumeData.data.vertexData.data(),vertexData,numVerts*sizeof(float));

    unstructuredVolumeData.data.dataRange.x = unstructuredVolumeData.minValue = minValue;
    unstructuredVolumeData.data.dataRange.y = unstructuredVolumeData.maxValue = maxValue;
    unstructuredVolumeData.changed = true;
}

void Renderer::unloadUMesh()
{
    // NO!
}

void Renderer::loadUMeshVTK(std::string fn)
{
#ifdef HAVE_VTK
    // deferred!
    unstructuredVolumeData.fileName = fn;
    unstructuredVolumeData.changed = true;

    if (unstructuredVolumeData.vtkReader.open(fn.c_str())) {
        unstructuredVolumeData.data = unstructuredVolumeData.vtkReader.getField(0);
    }
#endif
}

void Renderer::unloadUMeshVTK(std::string fn)
{
    // NO!
}


void Renderer::setRendererType(std::string type)
{
    if (anari.frames.empty())
        return;

    anari.renderertype = type;
    for (size_t i=0; i<channelInfos.size(); ++i) {
        // Causes frame re-initialization
        channelInfos[i].width = 1;
        channelInfos[i].height = 1;
        channelInfos[i].mv = osg::Matrix::identity();
        channelInfos[i].pr = osg::Matrix::identity();
    }
    initFrames();
    //anariRelease(anari.device, anari.renderer);
    //anari.renderer = nullptr;
    //anari.frames.clear();
}

std::vector<std::string> Renderer::getRendererTypes()
{
    std::vector<std::string> result;
    const char * const *rendererSubtypes = nullptr;
    anariGetProperty(anari.device, anari.device, "subtypes.renderer", ANARI_STRING_LIST,
                     &rendererSubtypes, sizeof(rendererSubtypes), ANARI_WAIT);
    if (rendererSubtypes != nullptr) {

        while (const char* rendererType = *rendererSubtypes++) {
            result.push_back(rendererType);
        }
    }

    if (result.empty()) {
        // If the device does not support the "subtypes.renderer" property,
        // try to obtain the renderer types from the library directly
        const char** deviceSubtypes = anariGetDeviceSubtypes(anari.library);
        if (deviceSubtypes != nullptr) {
            while (const char* dstype = *deviceSubtypes++) {
                const char** rendererTypes = anariGetObjectSubtypes(anari.device, ANARI_RENDERER);
                while (rendererTypes && *rendererTypes) {
                    const char* rendererType = *rendererTypes++;
                    result.push_back(rendererType);
                }
            }
        }
    }

    if (result.empty())
        result.push_back("default");

    return result;
}

void Renderer::setPixelSamples(int spp)
{
    this->spp = spp;

    if (!anari.renderer)
        return;

    anariSetParameter(anari.device, anari.renderer, "pixelSamples", ANARI_INT32, &spp);
    anariCommitParameters(anari.device, anari.renderer);
}

void Renderer::loadVolumeRAW(std::string fn)
{
    // deferred!

    // parse dimensions
    std::vector<std::string> strings = string_split(fn, '_');

    int sizeX=0, sizeY=0, sizeZ=0;
    for (auto str : strings) {
        int res = sscanf(str.c_str(), "%dx%dx%dx", &sizeX, &sizeY, &sizeZ);
        if (res == 3)
            break;
    }

    size_t numVoxels = sizeX*size_t(sizeY)*sizeZ;
    if (numVoxels == 0)
        return;

    structuredVolumeData.fileName = fn;
    structuredVolumeData.data = new uint8_t[numVoxels];
    structuredVolumeData.sizeX = sizeX;
    structuredVolumeData.sizeY = sizeY;
    structuredVolumeData.sizeZ = sizeZ;
    structuredVolumeData.bpc = 1;// !
    structuredVolumeData.minValue = 0.f;
    structuredVolumeData.maxValue = 1.f;
    structuredVolumeData.changed = true;
    structuredVolumeData.deleteData = true;

    FILE *file = fopen(fn.c_str(), "rb");
    size_t res = fread((void *)structuredVolumeData.data, numVoxels, 1, file);
    if (res != 1) {
        printf("Error reading %" PRIu64 " voxels with fread(). Result was %" PRIu64 "\n",
               (uint64_t)numVoxels, (uint64_t)res);
    }
    fclose(file);
}

void Renderer::unloadVolumeRAW(std::string fn)
{
    // NO!
}

void Renderer::expandBoundingSphere(osg::BoundingSphere &bs)
{
    if (!anari.world)
        return;

    float bounds[6] = { 1e30f, 1e30f, 1e30f,
                       -1e30f,-1e30f,-1e30f };

    // e.g., set when AMR data was loaded!
    anariGetProperty(anari.device, anari.world, "bounds", ANARI_FLOAT32_BOX3, &bounds, sizeof(bounds), ANARI_WAIT);

    if (anari.meshes) {
        asgComputeBounds(anari.meshes,
                         &bounds[0],&bounds[1],&bounds[2],
                         &bounds[3],&bounds[4],&bounds[5]);
    }

    if (anari.structuredVolume) {
        // asgComputeBounds doesn't work for volumes yet...
        bounds[0] = fminf(bounds[0], 0.f);
        bounds[1] = fminf(bounds[1], 0.f);
        bounds[2] = fminf(bounds[2], 0.f);
        bounds[3] = fmaxf(bounds[3], structuredVolumeData.sizeX);
        bounds[4] = fmaxf(bounds[4], structuredVolumeData.sizeY);
        bounds[5] = fmaxf(bounds[5], structuredVolumeData.sizeZ);
    }

    osg::Vec3f minCorner(bounds[0],bounds[1],bounds[2]);
    osg::Vec3f maxCorner(bounds[3],bounds[4],bounds[5]);

    osg::Vec3f center = (minCorner+maxCorner)*.5f;
    float radius = (center-minCorner).length();
    bs.set(center, radius);
}

void Renderer::updateLights(const osg::Matrix &modelMat)
{
    anari.lights.clear();
    for (size_t l=0; l<opencover::coVRLighting::instance()->lightList.size(); ++l) {
        auto &light = opencover::coVRLighting::instance()->lightList[l];
        if (light.on) {
            ANARILight al;
            osg::Light *osgLight = light.source->getLight();
            osg::NodePath np;
            np.push_back(light.root);
            np.push_back(light.source);
            osg::Vec4 pos = osgLight->getPosition();
            pos = pos * osg::Matrix::inverse(modelMat);

            if (pos.w() == 0.f) {
                al = anariNewLight(anari.device,"directional");
                anariSetParameter(anari.device, al, "direction",
                                  ANARI_FLOAT32_VEC3, pos.ptr());
            } else {
                al = anariNewLight(anari.device,"point");
                anariSetParameter(anari.device, al, "position",
                                  ANARI_FLOAT32_VEC3, pos.ptr());
            }

            anariCommitParameters(anari.device, al);

            anari.lights.push_back(al);
        }
    }

    ANARIArray1D anariLights = anariNewArray1D(anari.device, anari.lights.data(), 0, 0,
                                               ANARI_LIGHT, anari.lights.size());
    anariSetParameter(anari.device, anari.world, "light", ANARI_ARRAY1D, &anariLights);
    anariCommitParameters(anari.device, anari.world);

    anariRelease(anari.device, anariLights);
}

void Renderer::renderFrame()
{
    int numChannels = coVRConfig::instance()->numChannels();
    if (!multiChannelDrawer) {
        multiChannelDrawer = new MultiChannelDrawer(false, false);
        multiChannelDrawer->setMode(MultiChannelDrawer::AsIs);
        cover->getScene()->addChild(multiChannelDrawer);
        channelInfos.resize(numChannels);
    }

    if (anari.frames.empty())
        initFrames();

    if (anari.frames.empty()) // init failed!
        return;

    if (meshData.changed) {
        initMesh();
        meshData.changed = false;
    }

    if (structuredVolumeData.changed) {
        initStructuredVolume();
        structuredVolumeData.changed = false;
    }

#ifdef HAVE_HDF5
    if (amrVolumeData.changed) {
        initAMRVolume();
        amrVolumeData.changed = false;
    }
#endif

    if (unstructuredVolumeData.changed) {
        initUnstructuredVolume();
        unstructuredVolumeData.changed = false;
    }

    for (unsigned chan=0; chan<numChannels; ++chan) {
        renderFrame(chan);
    }
}

void Renderer::renderFrame(unsigned chan)
{
    multiChannelDrawer->update();

    auto cam = coVRConfig::instance()->channels[chan].camera;
    auto vp = cam->getViewport();
    int width = vp->width();
    int height = vp->height();

    if (channelInfos[chan].width != width || channelInfos[chan].height != height) {
        channelInfos[chan].width = width;
        channelInfos[chan].height = height;
        multiChannelDrawer->resizeView(chan, width, height,
                                       channelInfos[chan].colorFormat,
                                       channelInfos[chan].depthFormat);
        multiChannelDrawer->clearColor(chan);
        multiChannelDrawer->clearDepth(chan);

        unsigned imgSize[] = {(unsigned)width,(unsigned)height};
        anariSetParameter(anari.device, anari.frames[chan], "size", ANARI_UINT32_VEC2, imgSize);
        anariCommitParameters(anari.device, anari.frames[chan]);
    }

    osg::Matrix mv = multiChannelDrawer->modelMatrix(chan) * multiChannelDrawer->viewMatrix(chan);
    osg::Matrix pr = multiChannelDrawer->projectionMatrix(chan);

    glm::mat4 glmv, glpr;
    // glm matrices are column-major, osg matrices are row-major!
    glmv[0] = glm::vec4(mv(0,0), mv(0,1), mv(0,2), mv(0,3));
    glmv[1] = glm::vec4(mv(1,0), mv(1,1), mv(1,2), mv(1,3));
    glmv[2] = glm::vec4(mv(2,0), mv(2,1), mv(2,2), mv(2,3));
    glmv[3] = glm::vec4(mv(3,0), mv(3,1), mv(3,2), mv(3,3));

    glpr[0] = glm::vec4(pr(0,0), pr(0,1), pr(0,2), pr(0,3));
    glpr[1] = glm::vec4(pr(1,0), pr(1,1), pr(1,2), pr(1,3));
    glpr[2] = glm::vec4(pr(2,0), pr(2,1), pr(2,2), pr(2,3));
    glpr[3] = glm::vec4(pr(3,0), pr(3,1), pr(3,2), pr(3,3));

    glm::vec3 eye, dir, up;
    float fovy, aspect;
    glm::box2 imgRegion;
    offaxisStereoCameraFromTransform(
        inverse(glpr), inverse(glmv), eye, dir, up, fovy, aspect, imgRegion);

    if (channelInfos[chan].mv != mv || channelInfos[chan].pr != pr) {
        channelInfos[chan].mv = mv;
        channelInfos[chan].pr = pr;

        anariSetParameter(anari.device, anari.cameras[chan], "fovy", ANARI_FLOAT32, &fovy);
        anariSetParameter(anari.device, anari.cameras[chan], "aspect", ANARI_FLOAT32, &aspect);
        anariSetParameter(anari.device, anari.cameras[chan], "position", ANARI_FLOAT32_VEC3, &eye.x);
        anariSetParameter(anari.device, anari.cameras[chan], "direction", ANARI_FLOAT32_VEC3, &dir.x);
        anariSetParameter(anari.device, anari.cameras[chan], "up", ANARI_FLOAT32_VEC3, &up.x);
        anariSetParameter(anari.device, anari.cameras[chan], "imageRegion", ANARI_FLOAT32_BOX2, &imgRegion.min);
        anariCommitParameters(anari.device, anari.cameras[chan]);
    }

    updateLights(multiChannelDrawer->modelMatrix(chan));

    anariRenderFrame(anari.device, anari.frames[chan]);
    anariFrameReady(anari.device, anari.frames[chan], ANARI_WAIT);

    uint32_t widthOUT;
    uint32_t heightOUT;
    ANARIDataType typeOUT;
    const uint32_t *fbPointer = (const uint32_t *)anariMapFrame(anari.device, anari.frames[chan],
                                                                "channel.color",
                                                                &widthOUT,
                                                                &heightOUT,
                                                                &typeOUT);
    memcpy((uint32_t *)multiChannelDrawer->rgba(chan), fbPointer,
           widthOUT*heightOUT*anari::sizeOf(typeOUT));
    anariUnmapFrame(anari.device, anari.frames[chan], "channel.color");

    const float *dbPointer = (const float *)anariMapFrame(anari.device, anari.frames[chan],
                                                          "channel.depth",
                                                          &widthOUT,
                                                          &heightOUT,
                                                          &typeOUT);
    std::vector<float> dbXformed(widthOUT*heightOUT);
    transformDepthFromWorldToGL(dbPointer, dbXformed.data(), eye, dir, up, fovy,
                                aspect, imgRegion, glmv, glpr, widthOUT, heightOUT);
    memcpy((float *)multiChannelDrawer->depth(chan), dbXformed.data(),
           widthOUT*heightOUT*anari::sizeOf(typeOUT));

    anariUnmapFrame(anari.device, anari.frames[chan], "channel.depth");

    multiChannelDrawer->swapFrame();
}

void Renderer::initDevice()
{
    bool libraryEntryExists = false;
    anari.libtype = covise::coCoviseConfig::getEntry(
        "value",
        "COVER.Plugin.ANARI.Library",
        "environment",
        &libraryEntryExists
    );

    bool hostnameEntryExists = false;
    std::string hostname = covise::coCoviseConfig::getEntry(
        "hostname",
        "COVER.Plugin.ANARI.RemoteServer",
        "localhost",
        &hostnameEntryExists
    );

    bool portEntryExists = false;
    unsigned short port = covise::coCoviseConfig::getInt(
        "port",
        "COVER.Plugin.ANARI.RemoteServer",
        31050,
        &portEntryExists
    );

    anari.library = anariLoadLibrary(anari.libtype.c_str(), statusFunc);
    if (!anari.library) return;
    anari.device = anariNewDevice(anari.library, anari.devtype.c_str());
    if (!anari.device) return;
    if (anari.libtype == "remote") {
        if (hostnameEntryExists)
            anariSetParameter(anari.device, anari.device, "server.hostname", ANARI_STRING,
                              hostname.c_str());

        if (portEntryExists)
            anariSetParameter(anari.device, anari.device, "server.port", ANARI_UINT16, &port);
    }
    anariCommitParameters(anari.device, anari.device);
    anari.world = anariNewWorld(anari.device);
}

void Renderer::initFrames()
{
    anari.renderer = anariNewRenderer(anari.device, anari.renderertype.c_str());

    anariSetParameter(anari.device, anari.renderer, "pixelSamples", ANARI_INT32, &spp);

    float r = coCoviseConfig::getFloat("r", "COVER.Background", 0.0f);
    float g = coCoviseConfig::getFloat("g", "COVER.Background", 0.0f);
    float b = coCoviseConfig::getFloat("b", "COVER.Background", 0.0f);
    float bgcolor[] = {r,g,b,1.f};

    anariSetParameter(anari.device, anari.renderer, "backgroundColor", ANARI_FLOAT32_VEC4,
                      bgcolor);
    anariCommitParameters(anari.device, anari.renderer);

    int numChannels = coVRConfig::instance()->numChannels();
    anari.frames.resize(numChannels);
    anari.cameras.resize(numChannels);
    for (unsigned chan=0; chan<numChannels; ++chan) {
        ANARIFrame &frame = anari.frames[chan];
        ANARICamera &camera = anari.cameras[chan];

        frame = anariNewFrame(anari.device);
        anariSetParameter(anari.device, frame, "world", ANARI_WORLD, &anari.world);

        ANARIDataType fbFormat = ANARI_UFIXED8_RGBA_SRGB;
        ANARIDataType dbFormat = ANARI_FLOAT32;
        anariSetParameter(anari.device, frame, "channel.color", ANARI_DATA_TYPE, &fbFormat);
        anariSetParameter(anari.device, frame, "channel.depth", ANARI_DATA_TYPE, &dbFormat);
        anariSetParameter(anari.device, frame, "renderer", ANARI_RENDERER, &anari.renderer);

        camera = anariNewCamera(anari.device, "perspective");
        anariSetParameter(anari.device, frame, "camera", ANARI_CAMERA, &camera);
        anariCommitParameters(anari.device, frame);
    }
}

// Scene loading

#define ASG_SAFE_CALL(X) X // TODO!

void Renderer::initMesh()
{
    const char *fileName = meshData.fileName.c_str();

    if (!anari.root)
        anari.root = asgNewObject();

    anari.meshes = asgNewObject();

    // Load from file
    std::string ext = getExt(fileName);
    if (ext==".pbf" || ext==".pbrt")
        ASG_SAFE_CALL(asgLoadPBRT(anari.meshes, fileName, 0));
    else
        ASG_SAFE_CALL(asgLoadASSIMP(anari.meshes, fileName, 0));

    ASG_SAFE_CALL(asgObjectAddChild(anari.root, anari.meshes));

    // Build up ANARI world
    ASGBuildWorldFlags_t flags = ASG_BUILD_WORLD_FLAG_GEOMETRIES |
                                 ASG_BUILD_WORLD_FLAG_MATERIALS  |
                                 ASG_BUILD_WORLD_FLAG_TRANSFORMS;
    ASG_SAFE_CALL(asgBuildANARIWorld(anari.root, anari.device, anari.world, flags, 0));

    anariCommitParameters(anari.device, anari.world);
}

void Renderer::initStructuredVolume()
{
    if (structuredVolumeData.bpc == 1) {
        // Convert to float..
        structuredVolumeData.voxels.resize(
            structuredVolumeData.sizeX*size_t(structuredVolumeData.sizeY)*structuredVolumeData.sizeZ);

        for (size_t i=0; i<structuredVolumeData.voxels.size(); ++i) {
            structuredVolumeData.voxels[i] = ((const uint8_t *)structuredVolumeData.data)[i]/255.999f;
        }

        if (!anari.root)
            anari.root = asgNewObject();

        anari.structuredVolume = asgNewStructuredVolume(structuredVolumeData.voxels.data(),
                                                        structuredVolumeData.sizeX,
                                                        structuredVolumeData.sizeY,
                                                        structuredVolumeData.sizeZ,
                                                        ASG_DATA_TYPE_FLOAT32, nullptr);
        ASG_SAFE_CALL(asgStructuredVolumeSetRange(anari.structuredVolume,
                                                  structuredVolumeData.minValue,
                                                  structuredVolumeData.maxValue));

        structuredVolumeData.rgbLUT.resize(15);
        structuredVolumeData.alphaLUT.resize(5);

        anari.lut = asgNewLookupTable1D(structuredVolumeData.rgbLUT.data(),
                                        structuredVolumeData.alphaLUT.data(),
                                        structuredVolumeData.alphaLUT.size(),
                                        nullptr);
        ASG_SAFE_CALL(asgMakeDefaultLUT1D(anari.lut, ASG_LUT_ID_DEFAULT_LUT));
        ASG_SAFE_CALL(asgStructuredVolumeSetLookupTable1D(anari.structuredVolume, anari.lut));

        ASG_SAFE_CALL(asgObjectAddChild(anari.root, anari.structuredVolume));

        ASGBuildWorldFlags_t flags = ASG_BUILD_WORLD_FLAG_VOLUMES |
                                     ASG_BUILD_WORLD_FLAG_LUTS;
        ASG_SAFE_CALL(asgBuildANARIWorld(anari.root, anari.device, anari.world, flags, 0));

        anariCommitParameters(anari.device, anari.world);
    }

    if (structuredVolumeData.deleteData) {
        switch (structuredVolumeData.bpc) {
            case 1:
                delete[] (uint8_t *)structuredVolumeData.data;
                break;

            case 2:
                delete[] (uint16_t *)structuredVolumeData.data;
                break;

            case 4:
                delete[] (float *)structuredVolumeData.data;
                break;
        }
        structuredVolumeData.deleteData = false;
    }
}

void Renderer::initAMRVolume()
{
#ifdef HAVE_HDF5
    anari.amrVolume.field = anariNewSpatialField(anari.device, "amr");
    // TODO: "amr" field is an extension - check if it is supported!
    auto &data = amrVolumeData.data;
    std::vector<anari::Array3D> blockDataV(data.blockData.size());
    for (size_t i = 0; i < data.blockData.size(); ++i) {
        blockDataV[i] = anari::newArray3D(anari.device, data.blockData[i].values.data(),
                                          data.blockData[i].dims[0],
                                          data.blockData[i].dims[1],
                                          data.blockData[i].dims[2]);
    }
    printf("Array sizes:\n");
    printf("    'cellWidth'  : %zu\n", data.cellWidth.size());
    printf("    'blockBounds': %zu\n", data.blockBounds.size());
    printf("    'blockLevel' : %zu\n", data.blockLevel.size());
    printf("    'blockData'  : %zu\n", blockDataV.size());

    anari::setParameterArray1D(anari.device, anari.amrVolume.field, "cellWidth", ANARI_FLOAT32,
                               data.cellWidth.data(), data.cellWidth.size());
    anari::setParameterArray1D(anari.device, anari.amrVolume.field, "block.bounds", ANARI_INT32_BOX3,
                               data.blockBounds.data(),
                               data.blockBounds.size());
    anari::setParameterArray1D(anari.device, anari.amrVolume.field, "block.level", ANARI_INT32,
                               data.blockLevel.data(), data.blockLevel.size());
    anari::setParameterArray1D(anari.device, anari.amrVolume.field, "block.data", ANARI_ARRAY1D,
                               blockDataV.data(), blockDataV.size());

    for (auto a : blockDataV)
        anari::release(anari.device, a);

    anariCommitParameters(anari.device, anari.amrVolume.field);

    amrVolumeData.minValue = data.voxelRange.x;
    amrVolumeData.maxValue = data.voxelRange.y;

    anari.amrVolume.volume = anari::newObject<anari::Volume>(anari.device, "transferFunction1D");
    anari::setParameter(anari.device, anari.amrVolume.volume, "field", anari.amrVolume.field);

    {
        std::vector<glm::vec3> colors;
        std::vector<float> opacities;

        colors.emplace_back(0.f, 0.f, 1.f);
        colors.emplace_back(0.f, 1.f, 0.f);
        colors.emplace_back(1.f, 0.f, 0.f);

        opacities.emplace_back(0.f);
        opacities.emplace_back(1.f);

        anari::setAndReleaseParameter(
            anari.device, anari.amrVolume.volume, "color",
            anari::newArray1D(anari.device, colors.data(), colors.size()));
        anari::setAndReleaseParameter(
            anari.device, anari.amrVolume.volume, "opacity",
            anari::newArray1D(anari.device, opacities.data(), opacities.size()));
        anariSetParameter(anari.device, anari.amrVolume.volume, "valueRange", ANARI_FLOAT32_BOX1,
                          &data.voxelRange);
    }

    anari::commitParameters(anari.device, anari.amrVolume.volume);

    anari::setAndReleaseParameter(anari.device, anari.world, "volume",
                                  anari::newArray1D(anari.device, &anari.amrVolume.volume));
    anariRelease(anari.device, anari.amrVolume.volume);
    anariCommitParameters(anari.device, anari.world);
#endif
}

void Renderer::initUnstructuredVolume()
{
    anari.unstructuredVolume.field = anari::newObject<anari::SpatialField>(anari.device, "unstructured");
    // TODO: "unstructured" field is an extension - check if it is supported!
    auto &data = unstructuredVolumeData.data;
    printf("Array sizes:\n");
    printf("    'vertexPosition': %zu\n", data.vertexPosition.size());
    printf("    'vertexData'    : %zu\n", data.vertexData.size());
    printf("    'index'         : %zu\n", data.index.size());
    printf("    'cellIndex'     : %zu\n", data.cellIndex.size());

    anari::setParameterArray1D(anari.device, anari.unstructuredVolume.field, "vertex.position",
            ANARI_FLOAT32_VEC3, data.vertexPosition.data(), data.vertexPosition.size());
    anari::setParameterArray1D(anari.device, anari.unstructuredVolume.field, "vertex.data",
            ANARI_FLOAT32, data.vertexData.data(), data.vertexData.size());
    anari::setParameterArray1D(anari.device, anari.unstructuredVolume.field, "index",
            ANARI_UINT64, data.index.data(), data.index.size());
    anari::setParameter(anari.device, anari.unstructuredVolume.field, "indexPrefixed",
                        ANARI_BOOL, &data.indexPrefixed);
    anari::setParameterArray1D(anari.device, anari.unstructuredVolume.field, "cell.index",
            ANARI_UINT64, data.cellIndex.data(), data.cellIndex.size());
    anari::setParameterArray1D(anari.device, anari.unstructuredVolume.field, "cell.type",
                               ANARI_UINT8, data.cellType.data(), data.cellType.size());

    anari::commitParameters(anari.device, anari.unstructuredVolume.field);

    anari.unstructuredVolume.volume = anari::newObject<anari::Volume>(anari.device, "transferFunction1D");
    anari::setParameter(anari.device, anari.unstructuredVolume.volume, "field", anari.unstructuredVolume.field);

    {
        std::vector<glm::vec3> colors;
        std::vector<float> opacities;

        colors.emplace_back(0.f, 0.f, 1.f);
        colors.emplace_back(0.f, 1.f, 0.f);
        colors.emplace_back(1.f, 0.f, 0.f);

        opacities.emplace_back(0.f);
        opacities.emplace_back(1.f);

        anari::setAndReleaseParameter(
            anari.device, anari.unstructuredVolume.volume, "color",
            anari::newArray1D(anari.device, colors.data(), colors.size()));
        anari::setAndReleaseParameter(
            anari.device, anari.unstructuredVolume.volume, "opacity",
            anari::newArray1D(anari.device, opacities.data(), opacities.size()));
        anariSetParameter(anari.device, anari.unstructuredVolume.volume, "valueRange", ANARI_FLOAT32_BOX1,
                          &data.dataRange);
    }

    anari::commitParameters(anari.device, anari.unstructuredVolume.volume);

    anari::setAndReleaseParameter(anari.device, anari.world, "volume",
                                  anari::newArray1D(anari.device, &anari.unstructuredVolume.volume));
    anariRelease(anari.device, anari.unstructuredVolume.volume);
    anariCommitParameters(anari.device, anari.world);
}


