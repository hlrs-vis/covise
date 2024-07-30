/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdio>
#include <iostream>
#include <sstream>
#ifdef ANARI_PLUGIN_HAVE_MPI
#include <mpi.h>
#endif
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/glm.h>
#include <osg/io_utils>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#ifdef ANARI_PLUGIN_HAVE_CUDA
#include <PluginUtil/CudaSafeCall.h>
#endif
#include "generateRandomSpheres.h"
#include "readPTS.h"
#include "readPLY.h"
#include "Projection.h"
#include "Renderer.h"
#include "hdri.h"

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


static bool deviceHasExtension(anari::Library library,
    const std::string &deviceSubtype,
    const std::string &extName)
{
    const char **extensions =
        anariGetDeviceExtensions(library, deviceSubtype.c_str());

    for (; *extensions; extensions++) {
        if (*extensions == extName)
            return true;
    }
    return false;
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

inline glm::mat4 osg2glm(const osg::Matrix &m)
{
    glm::mat4 res;
    // glm matrices are column-major, osg matrices are row-major!
    res[0] = glm::vec4(m(0,0), m(0,1), m(0,2), m(0,3));
    res[1] = glm::vec4(m(1,0), m(1,1), m(1,2), m(1,3));
    res[2] = glm::vec4(m(2,0), m(2,1), m(2,2), m(2,3));
    res[3] = glm::vec4(m(3,0), m(3,1), m(3,2), m(3,3));
    return res;
}

inline osg::Matrix glm2osg(const glm::mat4 &m)
{
    osg::Matrix res;
    // glm matrices are column-major, osg matrices are row-major!
    res(0,0) = m[0].x; res(0,1) = m[0].y; res(0,2) = m[0].z; res(0,3) = m[0].w;
    res(1,0) = m[1].x; res(1,1) = m[1].y; res(1,2) = m[1].z; res(1,3) = m[1].w;
    res(2,0) = m[2].x; res(2,1) = m[2].y; res(2,2) = m[2].z; res(2,3) = m[2].w;
    res(3,0) = m[3].x; res(3,1) = m[3].y; res(3,2) = m[3].z; res(3,3) = m[3].w;
    return res;
}

Renderer::Renderer()
{
    initDevice();

    if (!anari.device)
        throw std::runtime_error("Could not init ANARI device");

    // Try loading HDRI image from config
    bool hdriEntryExists = false;
    std::string hdriName  = covise::coCoviseConfig::getEntry(
        "value",
        "COVER.Plugin.ANARI.hdri",
        &hdriEntryExists
    );

    if (hdriEntryExists) {
        loadHDRI(hdriName);
    }
}

Renderer::~Renderer()
{
#ifdef ANARI_PLUGIN_HAVE_CUDA
    if (anari.cudaInterop.enabled) {
        CUDA_SAFE_CALL(cudaStreamDestroy(anari.cudaInterop.copyStream));
    }
#endif
    if (multiChannelDrawer) {
        cover->getScene()->removeChild(multiChannelDrawer);
    }
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

void Renderer::loadUMeshFile(std::string fn)
{
#ifdef HAVE_UMESH
    // deferred!
    if (unstructuredVolumeData.umeshReader.open(fn.c_str())) {
        unstructuredVolumeData.fileName = fn;
        unstructuredVolumeData.readerType = UMESH;
        unstructuredVolumeData.changed = true;
    }
#endif
}

void Renderer::unloadUMeshFile(std::string fn)
{
    // NO!
}

void Renderer::loadUMeshScalars(std::string fn)
{
#ifdef HAVE_UMESH
    unstructuredVolumeData.umeshScalarFiles.push_back({fn, 0});
#endif
}

void Renderer::unloadUMeshScalars(std::string fn)
{
    // NO!
}

void Renderer::loadUMeshVTK(std::string fn)
{
#ifdef HAVE_VTK
    // deferred!
    if (unstructuredVolumeData.vtkReader.open(fn.c_str())) {
        unstructuredVolumeData.fileName = fn;
        unstructuredVolumeData.readerType = VTK;
        unstructuredVolumeData.changed = true;
    }
#endif
}

void Renderer::unloadUMeshVTK(std::string fn)
{
    // NO!
}

void Renderer::loadPointCloud(std::string fn)
{
    // deferred!
    pointCloudData.fileNames.push_back(fn);
    pointCloudData.changed = true;
}

void Renderer::unloadPointCloud(std::string fn)
{
    // NO!
}

void Renderer::loadHDRI(std::string fn)
{
    HDRI img;
    img.load(fn);
    hdri.pixels.resize(img.width * img.height);
    hdri.width = img.width;
    hdri.height = img.height;
    if (img.numComponents == 3) {
      memcpy(hdri.pixels.data(), img.pixel.data(), sizeof(hdri.pixels[0]) * hdri.pixels.size());
    } else if (img.numComponents == 4) {
      for (size_t i = 0; i < img.pixel.size(); i += 4) {
        hdri.pixels[i / 4] = glm::vec3(img.pixel[i], img.pixel[i + 1], img.pixel[i + 2]);
      }
    }
    hdri.changed = true;
}

void Renderer::unloadHDRI(std::string fn)
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
    if (rendererTypes.empty()) {
        const char * const *rendererSubtypes = nullptr;
        anariGetProperty(anari.device, anari.device, "subtypes.renderer", ANARI_STRING_LIST,
                         &rendererSubtypes, sizeof(rendererSubtypes), ANARI_WAIT);
        if (rendererSubtypes != nullptr) {

            while (const char* rendererType = *rendererSubtypes++) {
                rendererTypes.push_back(rendererType);
            }
        }

        if (rendererTypes.empty()) {
            // If the device does not support the "subtypes.renderer" property,
            // try to obtain the renderer types from the library directly
            const char** deviceSubtypes = anariGetDeviceSubtypes(anari.library);
            if (deviceSubtypes != nullptr) {
                while (const char* dstype = *deviceSubtypes++) {
                    const char** rt = anariGetObjectSubtypes(anari.device, ANARI_RENDERER);
                    while (rt && *rt) {
                        const char* rendererType = *rt++;
                        rendererTypes.push_back(rendererType);
                    }
                }
            }
        }
    }

    if (rendererTypes.empty())
        rendererTypes.push_back("default");

    return rendererTypes;
}

std::vector<ui_anari::ParameterList> &Renderer::getRendererParameters()
{
    if (rendererParameters.empty()) {
        auto r_subtypes = getRendererTypes();
        for (auto subtype : r_subtypes) {
            auto parameters =
                ui_anari::parseParameters(anari.device, ANARI_RENDERER, subtype.c_str());
            rendererParameters.push_back(parameters);
        }
    }

    return rendererParameters;
}

void Renderer::setParameter(std::string name, bool value)
{
    anari::setParameter(anari.device, anari.renderer, name.c_str(), value);
    anariCommitParameters(anari.device, anari.renderer);
}

void Renderer::setParameter(std::string name, int value)
{
    anari::setParameter(anari.device, anari.renderer, name.c_str(), value);
    anariCommitParameters(anari.device, anari.renderer);
}

void Renderer::setParameter(std::string name, float value)
{
    anari::setParameter(anari.device, anari.renderer, name.c_str(), value);
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

#ifdef ANARI_PLUGIN_HAVE_MPI
    if (mpiSize > 1) {
        float localBounds[6];
        for (int i=0; i<6; ++i) {
            localBounds[i] = bounds[i];
        }
        float globalBounds[6];
        MPI_Allreduce(&localBounds[0], &globalBounds[0], 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localBounds[1], &globalBounds[1], 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localBounds[2], &globalBounds[2], 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localBounds[3], &globalBounds[3], 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localBounds[4], &globalBounds[4], 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localBounds[5], &globalBounds[5], 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        for (int i=0; i<6; ++i) {
            bounds[i] = globalBounds[i];
        }
    }
#endif

    osg::Vec3f minCorner(bounds[0],bounds[1],bounds[2]);
    osg::Vec3f maxCorner(bounds[3],bounds[4],bounds[5]);

    osg::Vec3f center = (minCorner+maxCorner)*.5f;
    float radius = (center-minCorner).length();
    bs.set(center, radius);
}

void Renderer::updateLights(const osg::Matrix &modelMat)
{
    std::vector<Light> newLights;

    // assemble new light list
    for (size_t l=0; l<opencover::coVRLighting::instance()->lightList.size(); ++l) {
        auto &light = opencover::coVRLighting::instance()->lightList[l];
        if (light.on) {
            newLights.push_back(light);
        }
    }

    // check if lights have updated
    if (newLights.size() != lights.data.size()) {
        lights.data = newLights;
        lights.changed = true;
    } else {
        for (size_t l=0; l<newLights.size(); ++l) {
            if (newLights[l].source != lights.data[l].source ||
                newLights[l].root != lights.data[l].root) {
                lights.data = newLights;
                lights.changed = true;
                break;
            }
        }
    }

    if (lights.changed || hdri.changed) {
        anari.lights.clear();

        if (!hdri.pixels.empty()) {
            ANARILight al = anariNewLight(anari.device,"hdri");

            ANARIArray2D radiance = anariNewArray2D(anari.device,hdri.pixels.data(),0,0,
                                                    ANARI_FLOAT32_VEC3,hdri.width,hdri.height);
            anariSetParameter(anari.device, al, "radiance", ANARI_ARRAY2D, &radiance);

            anariCommitParameters(anari.device, al);

            anariRelease(anari.device, radiance);

            anari.lights.push_back(al);
        }

        for (size_t l=0; l<lights.data.size(); ++l) {
            ANARILight al;
            auto &light = lights.data[l];
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
        ANARIArray1D anariLights = anariNewArray1D(anari.device, anari.lights.data(), 0, 0,
                                                   ANARI_LIGHT, anari.lights.size());
        anariSetParameter(anari.device, anari.world, "light", ANARI_ARRAY1D, &anariLights);
        anariCommitParameters(anari.device, anari.world);

        anariRelease(anari.device, anariLights);

        lights.changed = false;
        hdri.changed = false;
    }

    // Even if the light _configuration_ hasn't change, proactively
    // update the light positions anyway (as this is cheap)
    for (size_t l=0; l<lights.data.size(); ++l) {
        ANARILight al = anari.lights[l];
        auto &light = lights.data[l];
        osg::Light *osgLight = light.source->getLight();
        osg::NodePath np;
        np.push_back(light.root);
        np.push_back(light.source);
        osg::Vec4 pos = osgLight->getPosition();
        pos = pos * osg::Matrix::inverse(modelMat);
        if (pos.w() == 0.f) {
            anariSetParameter(anari.device, al, "direction",
                              ANARI_FLOAT32_VEC3, pos.ptr());
        } else {
            anariSetParameter(anari.device, al, "position",
                              ANARI_FLOAT32_VEC3, pos.ptr());
        }

        anariCommitParameters(anari.device, al);
    }
}

void Renderer::setClipPlanes(const std::vector<Renderer::ClipPlane> &planes)
{
    bool doUpdate =  false;
    if (planes.size() != clipPlanes.data.size()) {
        doUpdate = true;
    } else {
        for (size_t i = 0; i < planes.size(); ++i) {
            for (int c = 0; c < 4; ++c) {
                if (clipPlanes.data[i][c] != planes[i][c]) {
                    doUpdate = true;
                    break;
                }

                if (doUpdate)
                    break;
            }
        }
    }

    if (doUpdate) {
        clipPlanes.data.clear();
        clipPlanes.data.insert(clipPlanes.data.begin(), planes.begin(), planes.end());
        clipPlanes.changed = true;
    }
}

void Renderer::renderFrame()
{
    const bool isDisplayRank = mpiRank==displayRank;

    int numChannels = coVRConfig::instance()->numChannels();
    channelInfos.resize(numChannels);

    if (isDisplayRank && !multiChannelDrawer) {
#ifdef ANARI_PLUGIN_HAVE_CUDA
        multiChannelDrawer = new MultiChannelDrawer(false, anari.cudaInterop.enabled);
#else
        multiChannelDrawer = new MultiChannelDrawer(false, false);
#endif
        multiChannelDrawer->setMode(MultiChannelDrawer::AsIs);
        cover->getScene()->addChild(multiChannelDrawer);
    }

    if (anari.frames.empty())
        initFrames();

    if (anari.frames.empty()) // init failed!
        return;

    if (meshData.changed) {
        initMesh();
        meshData.changed = false;
    }

    // if (pointCloudData.fileNames.empty()) {
    //      pointCloudData.fileNames.push_back("random");
    //      pointCloudData.changed = true;
    // }
    if (pointCloudData.changed) {
        initPointClouds();
        pointCloudData.changed = false;
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

    if (clipPlanes.changed) {
        initClipPlanes();
        clipPlanes.changed = false;
    }

    for (unsigned chan=0; chan<numChannels; ++chan) {
        renderFrame(chan);
    }
}

void Renderer::renderFrame(unsigned chan)
{
    const bool isDisplayRank = mpiRank==displayRank;
    if (isDisplayRank) {
        multiChannelDrawer->update();

        auto cam = coVRConfig::instance()->channels[chan].camera;
        auto vp = cam->getViewport();
        int width = vp->width();
        int height = vp->height();

#ifdef ANARI_PLUGIN_HAVE_MPI
        if (mpiSize > 1) {
            MPI_Bcast(&width, 1, MPI_INT, displayRank, MPI_COMM_WORLD);
            MPI_Bcast(&height, 1, MPI_INT, displayRank, MPI_COMM_WORLD);
        }
#endif

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

        glm::mat4 glmm = osg2glm(multiChannelDrawer->modelMatrix(chan));
        glm::mat4 glmv = osg2glm(mv);
        glm::mat4 glpr = osg2glm(pr);

#ifdef ANARI_PLUGIN_HAVE_MPI
        if (mpiSize > 1) {
            MPI_Bcast(&glmm, sizeof(glmm), MPI_BYTE, displayRank, MPI_COMM_WORLD);
            MPI_Bcast(&glmv, sizeof(glmv), MPI_BYTE, displayRank, MPI_COMM_WORLD);
            MPI_Bcast(&glpr, sizeof(glpr), MPI_BYTE, displayRank, MPI_COMM_WORLD);
        }
#endif

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

#ifdef ANARI_PLUGIN_HAVE_MPI
        if (mpiSize > 1)
            MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef ANARI_PLUGIN_HAVE_CUDA
        if (anari.cudaInterop.enabled) {
            uint32_t widthOUT;
            uint32_t heightOUT;
            ANARIDataType typeOUT;
            const uint32_t *fbPointer = (const uint32_t *)anariMapFrame(anari.device, anari.frames[chan],
                                                                        "channel.colorGPU",
                                                                        &widthOUT,
                                                                        &heightOUT,
                                                                        &typeOUT);
            CUDA_SAFE_CALL(cudaMemcpyAsync((uint32_t *)multiChannelDrawer->rgba(chan), fbPointer,
                   widthOUT*heightOUT*anari::sizeOf(typeOUT), cudaMemcpyDeviceToDevice,
                   anari.cudaInterop.copyStream));
            CUDA_SAFE_CALL(cudaStreamSynchronize(anari.cudaInterop.copyStream));
            anariUnmapFrame(anari.device, anari.frames[chan], "channel.colorGPU");

            const float *dbPointer = (const float *)anariMapFrame(anari.device, anari.frames[chan],
                                                                  "channel.depthGPU",
                                                                  &widthOUT,
                                                                  &heightOUT,
                                                                  &typeOUT);
            float *dbXformed;
            CUDA_SAFE_CALL(cudaMalloc(&dbXformed, widthOUT*heightOUT*sizeof(float)));
            transformDepthFromWorldToGL_CUDA(dbPointer, dbXformed, eye, dir, up, fovy,
                                        aspect, imgRegion, glmv, glpr, widthOUT, heightOUT);
            CUDA_SAFE_CALL(cudaMemcpyAsync((float *)multiChannelDrawer->depth(chan), dbXformed,
                   widthOUT*heightOUT*anari::sizeOf(typeOUT), cudaMemcpyDeviceToDevice,
                   anari.cudaInterop.copyStream));
            CUDA_SAFE_CALL(cudaStreamSynchronize(anari.cudaInterop.copyStream));
            CUDA_SAFE_CALL(cudaFree(dbXformed));

            anariUnmapFrame(anari.device, anari.frames[chan], "channel.depthGPU");
        } else {
#endif
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
#ifdef ANARI_PLUGIN_HAVE_CUDA
        }
#endif

        multiChannelDrawer->swapFrame();
    }
#ifdef ANARI_PLUGIN_HAVE_MPI
    else {
        // non-display rank:
        int width, height;
        MPI_Bcast(&width, 1, MPI_INT, displayRank, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, displayRank, MPI_COMM_WORLD);

        if (channelInfos[chan].width != width || channelInfos[chan].height != height) {
            channelInfos[chan].width = width;
            channelInfos[chan].height = height;

            unsigned imgSize[] = {(unsigned)width,(unsigned)height};
            anariSetParameter(anari.device, anari.frames[chan], "size", ANARI_UINT32_VEC2, imgSize);
            anariCommitParameters(anari.device, anari.frames[chan]);
        }

        glm::mat4 glmm, glmv, glpr;
        MPI_Bcast(&glmm, sizeof(glmm), MPI_BYTE, displayRank, MPI_COMM_WORLD);
        MPI_Bcast(&glmv, sizeof(glmv), MPI_BYTE, displayRank, MPI_COMM_WORLD);
        MPI_Bcast(&glpr, sizeof(glpr), MPI_BYTE, displayRank, MPI_COMM_WORLD);

        glm::vec3 eye, dir, up;
        float fovy, aspect;
        glm::box2 imgRegion;
        offaxisStereoCameraFromTransform(
            inverse(glpr), inverse(glmv), eye, dir, up, fovy, aspect, imgRegion);

        osg::Matrix mv = glm2osg(glmv);
        osg::Matrix pr = glm2osg(glpr);
        // glm matrices are column-major, osg matrices are row-major!
        mv(0,0) = glmv[0].x; mv(0,1) = glmv[0].y; mv(0,2) = glmv[0].z; mv(0,3) = glmv[0].w;
        mv(1,0) = glmv[1].x; mv(1,1) = glmv[1].y; mv(1,2) = glmv[1].z; mv(1,3) = glmv[1].w;
        mv(2,0) = glmv[2].x; mv(2,1) = glmv[2].y; mv(2,2) = glmv[2].z; mv(2,3) = glmv[2].w;
        mv(3,0) = glmv[3].x; mv(3,1) = glmv[3].y; mv(3,2) = glmv[3].z; mv(3,3) = glmv[3].w;

        pr(0,0) = glpr[0].x; pr(0,1) = glpr[0].y; pr(0,2) = glpr[0].z; pr(0,3) = glpr[0].w;
        pr(1,0) = glpr[1].x; pr(1,1) = glpr[1].y; pr(1,2) = glpr[1].z; pr(1,3) = glpr[1].w;
        pr(2,0) = glpr[2].x; pr(2,1) = glpr[2].y; pr(2,2) = glpr[2].z; pr(2,3) = glpr[2].w;
        pr(3,0) = glpr[3].x; pr(3,1) = glpr[3].y; pr(3,2) = glpr[3].z; pr(3,3) = glpr[3].w;

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

        updateLights(glm2osg(glmm));
        anariRenderFrame(anari.device, anari.frames[chan]);
        anariFrameReady(anari.device, anari.frames[chan], ANARI_WAIT);

        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
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
#ifdef ANARI_PLUGIN_HAVE_CUDA
    anari.cudaInterop.enabled
        = deviceHasExtension(anari.library, "default", "ANARI_VISRTX_CUDA_OUTPUT_BUFFERS");
    if (anari.cudaInterop.enabled) {
        CUDA_SAFE_CALL(cudaStreamCreate(&anari.cudaInterop.copyStream));
    }
#endif
}

void Renderer::initFrames()
{
    anari.renderer = anariNewRenderer(anari.device, anari.renderertype.c_str());

    float r = coCoviseConfig::getFloat("r", "COVER.Background", 0.0f);
    float g = coCoviseConfig::getFloat("g", "COVER.Background", 0.0f);
    float b = coCoviseConfig::getFloat("b", "COVER.Background", 0.0f);
    float bgcolor[] = {r,g,b,1.f};

    anariSetParameter(anari.device, anari.renderer, "background", ANARI_FLOAT32_VEC4,
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

void Renderer::initPointClouds()
{
    auto surfaceArray
        = anari::newArray1D(anari.device, ANARI_SURFACE, pointCloudData.fileNames.size());

    bool radiusEntryExists = false;
    float radius  = covise::coCoviseConfig::getFloat(
        "radius",
        "COVER.Plugin.ANARI.PointCloud",
        0.1f,
        &radiusEntryExists
    );

    auto *s = anari::map<anari::Surface>(anari.device, surfaceArray);
    for (size_t i = 0; i < pointCloudData.fileNames.size(); ++i) {
        std::string fn(pointCloudData.fileNames[i]);
        // TODO: import
        if (fn == "random") {
            auto surface = generateRandomSpheres(anari.device, glm::vec3(0.f));
            s[i] = surface;
            anariRelease(anari.device, surface);
        } else if (getExt(fn)==".pts") {
            auto surface = readPTS(anari.device, fn, radius);
            s[i] = surface;
	        anariRelease(anari.device, surface);
	    } else if (getExt(fn)==".ply") {
            auto surface = readPLY(anari.device, fn, radius);
            s[i] = surface;
	        anariRelease(anari.device, surface);
	    }
    }
    anari::unmap(anari.device, surfaceArray);
    anari::setAndReleaseParameter(anari.device, anari.world, "surface", surfaceArray);

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
    if (unstructuredVolumeData.readerType == UMESH) {
#ifdef HAVE_UMESH
        for (const auto &sf : unstructuredVolumeData.umeshScalarFiles) {
            unstructuredVolumeData.umeshReader.addFieldFromFile(sf.fileName.c_str(), sf.fieldID);
        }
        unstructuredVolumeData.data = unstructuredVolumeData.umeshReader.getField(0);
#else
        return;
#endif
    } else if (unstructuredVolumeData.readerType == VTK) {
#ifdef HAVE_VTK
        unstructuredVolumeData.data = unstructuredVolumeData.vtkReader.getField(0);
#else
        return;
#endif
    }
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

void Renderer::initClipPlanes()
{
    if (!anari.renderer)
        return;

    if (clipPlanes.data.empty()) {
        anari::unsetParameter(anari.device, anari.renderer, "clipPlane");
    } else {
        anari::setAndReleaseParameter(
            anari.device, anari.renderer, "clipPlane",
            anari::newArray1D(anari.device, clipPlanes.data.data(), clipPlanes.data.size()));
    }

    anari::commitParameters(anari.device, anari.renderer);
}


