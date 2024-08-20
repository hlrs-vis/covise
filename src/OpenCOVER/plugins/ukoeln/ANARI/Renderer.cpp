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
//#include <glm/gtx/string_cast.hpp>
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

enum FileType {
    OBJ,
    UMesh,
    UMeshScalars,
    VTU,
    VTK,
    Unknown,
    // Keep last:
    FileTypeCount,
};

inline
FileType getFileType(const std::string &fileName)
{
    auto ext = getExt(fileName);
    if (ext == ".obj") return OBJ;
    else if (ext == ".umesh") return UMesh;
    else if (ext == ".scalars") return UMeshScalars;
    else if (ext == ".vtu") return VTU;
    else if (ext == ".vtk") return VTK;
    else return Unknown;
}

inline
int getID(std::string fileName) {
    // check if we know this file name:
    static std::map<std::string,int> knownFileNames[FileTypeCount];
    FileType type = getFileType(fileName);

    auto it = knownFileNames[type].find(fileName);
    if (it != knownFileNames[type].end()) {
      return it->second;
    }

    // file name is unknown
    static int nextID[FileTypeCount] = { 0 };
    int ID = nextID[type]++;
    knownFileNames[type][fileName] = ID;
    return ID;
}

struct Slot {
    Slot() = default;
    Slot(std::string fileName, int mpiSize) {
        int ID = getID(fileName);
        // round-robin:
        mpiRank = ID%mpiSize;
    }
    int mpiRank{0};
    int timeStep{0};
};


static bool deviceHasExtension(anari::Library library,
    const std::string &deviceSubtype,
    const std::string &extName)
{
    const char **extensions =
        anariGetDeviceExtensions(library, deviceSubtype.c_str());

    if (!extensions)
        return false;

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

inline glm::vec3 randomColor(unsigned idx)
{
  unsigned int r = (unsigned int)(idx*13*17 + 0x234235);
  unsigned int g = (unsigned int)(idx*7*3*5 + 0x773477);
  unsigned int b = (unsigned int)(idx*11*19 + 0x223766);
  return glm::vec3((r&255)/255.f,
                   (g&255)/255.f,
                   (b&255)/255.f);
}

Renderer::Renderer()
{
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

void Renderer::init()
{
    initMPI();
    initRR();
    initChannels();
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

    // generate default TF:
    generateTransFunc();
}

void Renderer::loadMesh(std::string fn)
{
    Slot slot(fn, mpiSize);
    if (slot.mpiRank != mpiRank) {
        bounds.updated = true; // all ranks participate in Bcast
        return;
    }

    // deferred!
    meshData.fileName = fn;
    meshData.updated = true;
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
    structuredVolumeData.updated = true;
}

void Renderer::unloadVolume()
{
    // NO!
}

void Renderer::loadFLASH(std::string fn)
{
#ifdef HAVE_HDF5
    Slot slot(fn, mpiSize);
    if (slot.mpiRank != mpiRank) {
        bounds.updated = true; // all ranks participate in Bcast
        return;
    }

    // deferred!
    amrVolumeData.fileName = fn;
    amrVolumeData.updated = true;

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
    unstructuredVolumeData.updated = true;
}

void Renderer::unloadUMesh()
{
    // NO!
}

void Renderer::loadUMeshFile(std::string fn)
{
#ifdef HAVE_UMESH
    Slot slot(fn, mpiSize);
    if (slot.mpiRank != mpiRank) {
        bounds.updated = true; // all ranks participate in Bcast
        return;
    }

    // deferred!
    if (unstructuredVolumeData.umeshReader.open(fn.c_str())) {
        unstructuredVolumeData.fileName = fn;
        unstructuredVolumeData.readerType = UMESH;
        unstructuredVolumeData.updated = true;
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
    Slot slot(fn, mpiSize);
    if (slot.mpiRank != mpiRank) {
        bounds.updated = true; // all ranks participate in Bcast
        return;
    }

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
        unstructuredVolumeData.updated = true;
    }
#endif
}

void Renderer::unloadUMeshVTK(std::string fn)
{
    // NO!
}

void Renderer::loadPointCloud(std::string fn)
{
    Slot slot(fn, mpiSize);
    if (slot.mpiRank != mpiRank) {
        bounds.updated = true; // all ranks participate in Bcast
        return;
    }

    // deferred!
    pointCloudData.fileNames.push_back(fn);
    pointCloudData.updated = true;
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
    hdri.updated = true;
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
        channelInfos[i].frame.width = 1;
        channelInfos[i].frame.height = 1;
        channelInfos[i].mv = glm::mat4();
        channelInfos[i].pr = glm::mat4();
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
    structuredVolumeData.updated = true;
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
    AABB bounds = this->bounds.global;

    osg::Vec3f minCorner(bounds.data[0],bounds.data[1],bounds.data[2]);
    osg::Vec3f maxCorner(bounds.data[3],bounds.data[4],bounds.data[5]);

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
            newLights.push_back(Light(light));
        }
    }

    // check if lights have updated
    if (newLights.size() != lights.data.size()) {
        lights.data = newLights;
        lights.updated = true;
    } else {
        for (size_t l=0; l<newLights.size(); ++l) {
            if (newLights[l].coLight.source != lights.data[l].coLight.source ||
                newLights[l].coLight.root != lights.data[l].coLight.root) {
                lights.data = newLights;
                lights.updated = true;
                break;
            }
        }
    }

    if (lights.updated || hdri.updated) {
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
            osg::Light *osgLight = light.coLight.source->getLight();
            osg::NodePath np;
            np.push_back(light.coLight.root);
            np.push_back(light.coLight.source);
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

        lights.updated = false;
        hdri.updated = false;
    }

    // Even if the light _configuration_ hasn't change, the light properties
    // themselves might have

    for (size_t l=0; l<lights.data.size(); ++l) {

        osg::Light *oldLight = lights.data[l].coLight.source->getLight();
        osg::Light *newLight = newLights[l].coLight.source->getLight();

        osg::NodePath np;
        np.push_back(newLights[l].coLight.root);
        np.push_back(newLights[l].coLight.source);

        osg::Vec4 pos = newLights[l].coLight.source->getLight()->getPosition();
        pos = pos * osg::Matrix::inverse(modelMat);

        if (*oldLight != *newLight || lights.data[l].prevPos != pos) {
            lights.data[l].updated = true;
        }
    }

    // Update light properties:
    for (size_t l=0; l<lights.data.size(); ++l) {

        if (!lights.data[l].updated) continue;

        auto &light = lights.data[l].coLight;
        osg::Light *osgLight = light.source->getLight();
        osg::NodePath np;
        np.push_back(light.root);
        np.push_back(light.source);
        osg::Vec4 pos = osgLight->getPosition();
        pos = pos * osg::Matrix::inverse(modelMat);

        ANARILight al = anari.lights[l];
        if (pos.w() == 0.f) {
            anariSetParameter(anari.device, al, "direction",
                              ANARI_FLOAT32_VEC3, pos.ptr());
        } else {
            anariSetParameter(anari.device, al, "position",
                              ANARI_FLOAT32_VEC3, pos.ptr());
        }

        anariCommitParameters(anari.device, al);

        lights.data[l].prevPos = pos;
        lights.data[l].updated = false;
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
        clipPlanes.updated = true;
    }
}

void Renderer::setTransFunc(const glm::vec3 *rgb, unsigned numRGB,
                            const float *opacity, unsigned numOpacity)
{
    transFunc.colors.resize(numRGB);
    transFunc.opacities.resize(numOpacity);

    memcpy(transFunc.colors.data(),
           rgb,
           transFunc.colors.size()*sizeof(transFunc.colors[0]));

    memcpy(transFunc.opacities.data(),
           opacity,
           transFunc.opacities.size()*sizeof(transFunc.opacities[0]));
    
    transFunc.updated = true;

#ifdef ANARI_PLUGIN_HAVE_RR
    objectUpdates |= RR_TRANSFUNC_UPDATED;
#endif
}

void Renderer::setColorRanks(bool value)
{
    colorByRank = value;
    generateTransFunc();
    transFunc.updated = true;

#ifdef ANARI_PLUGIN_HAVE_RR
    objectUpdates |= RR_TRANSFUNC_UPDATED;
#endif
}

void Renderer::wait()
{
#ifdef ANARI_PLUGIN_HAVE_MPI
    if (mpiSize > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

void Renderer::renderFrame()
{
    const bool isDisplayRank = mpiRank==displayRank;

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

    if (meshData.updated) {
        initMesh();
        meshData.updated = false;
    }

    // if (pointCloudData.fileNames.empty()) {
    //      pointCloudData.fileNames.push_back("random");
    //      pointCloudData.updated = true;
    // }
    if (pointCloudData.updated) {
        initPointClouds();
        pointCloudData.updated = false;
    }

    if (structuredVolumeData.updated) {
        initStructuredVolume();
        structuredVolumeData.updated = false;
    }

#ifdef HAVE_HDF5
    if (amrVolumeData.updated) {
        initAMRVolume();
        amrVolumeData.updated = false;
    }
#endif

    if (unstructuredVolumeData.updated) {
        initUnstructuredVolume();
        unstructuredVolumeData.updated = false;
    }

    if (transFunc.updated) {
        initTransFunc();
        transFunc.updated = false;
    }

    if (clipPlanes.updated) {
        initClipPlanes();
        clipPlanes.updated = false;
    }

    if (bounds.updated) {
#ifdef ANARI_PLUGIN_HAVE_MPI
        if (mpiSize > 1) {
            MPI_Allreduce(&bounds.local.data[0], &bounds.global.data[0], 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&bounds.local.data[1], &bounds.global.data[1], 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&bounds.local.data[2], &bounds.global.data[2], 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&bounds.local.data[3], &bounds.global.data[3], 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&bounds.local.data[4], &bounds.global.data[4], 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&bounds.local.data[5], &bounds.global.data[5], 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        } else
#endif
        {
            memcpy(&bounds.global.data[0],
                   &bounds.local.data[0],
                   sizeof(bounds.local.data));
        }

#ifdef ANARI_PLUGIN_HAVE_RR
        objectUpdates |= RR_BOUNDS_UPDATED;
#endif

        bounds.updated = false;
    }

    for (unsigned chan=0; chan<numChannels; ++chan) {
        renderFrame(chan);
    }

    wait();
}

void Renderer::renderFrame(unsigned chan)
{
    const bool isDisplayRank = mpiRank==displayRank;
    ChannelInfo &info = channelInfos[chan];

    int width=info.frame.width, height=info.frame.height;
    glm::mat4 mm=info.mm, mv=info.mv, pr=info.pr, vv{};
    if (isDisplayRank) {
        multiChannelDrawer->update();

        auto cam = coVRConfig::instance()->channels[chan].camera;
        auto vp = cam->getViewport();
        width = vp->width();
        height = vp->height();
    
        if (info.frame.width != width || info.frame.height != height) {
            objectUpdates |= RR_VIEWPORT_UPDATED;
        }

        mm = osg2glm(multiChannelDrawer->modelMatrix(chan));
        vv = osg2glm(multiChannelDrawer->viewMatrix(chan));
        mv = osg2glm(multiChannelDrawer->modelMatrix(chan) * multiChannelDrawer->viewMatrix(chan));
        pr = osg2glm(multiChannelDrawer->projectionMatrix(chan));

        if (info.mv != mv || info.pr != pr) {
            objectUpdates |= RR_CAMERA_UPDATED;
        }
    }

#ifdef ANARI_PLUGIN_HAVE_RR
    if (isClient) {
        // Exchange what updated:
        uint64_t myObjectUpdates = objectUpdates;
        uint64_t peerObjectUpdates{0x0};
        rr->sendObjectUpdates(objectUpdates);
        rr->recvObjectUpdates(peerObjectUpdates);

        // Send ours:
        if (myObjectUpdates & RR_VIEWPORT_UPDATED)
        {
            minirr::Viewport rrViewport;
            rrViewport.width = width;
            rrViewport.height = height;
            rr->sendViewport(rrViewport);
        }

        if (myObjectUpdates & RR_CAMERA_UPDATED)
        {
            minirr::Camera rrCamera;
            std::memcpy(rrCamera.modelMatrix, &mm[0], sizeof(rrCamera.modelMatrix));
            std::memcpy(rrCamera.viewMatrix, &vv[0], sizeof(rrCamera.viewMatrix));
            std::memcpy(rrCamera.projMatrix, &pr[0], sizeof(rrCamera.projMatrix));
            rr->sendCamera(rrCamera);
        }
    
        if (myObjectUpdates & RR_TRANSFUNC_UPDATED)
        {
            minirr::Transfunc rrTransfunc;
            rrTransfunc.rgb = (float *)transFunc.colors.data();
            rrTransfunc.alpha = (float *)transFunc.opacities.data();
            rrTransfunc.numRGB = transFunc.colors.size();
            rrTransfunc.numAlpha = transFunc.opacities.size();
            rr->sendTransfunc(rrTransfunc);
            // TODO: ranges, scale
        }

        // Recv peer:
        if (peerObjectUpdates & RR_BOUNDS_UPDATED)
        {
            minirr::AABB rrBounds;
            rr->recvBounds(rrBounds);
            std::memcpy(&bounds.global.data[0], &rrBounds[0], sizeof(rrBounds));
            bounds.global.data[0] = rrBounds[0];
        }
    }

    if (isServer) {
        // Exchange what updated:
        uint64_t myObjectUpdates = objectUpdates;
        uint64_t peerObjectUpdates{0x0};
        rr->recvObjectUpdates(peerObjectUpdates);
        rr->sendObjectUpdates(myObjectUpdates);

        // Recv peer:
        if (peerObjectUpdates & RR_VIEWPORT_UPDATED)
        {
            minirr::Viewport rrViewport;
            rr->recvViewport(rrViewport);
            width = rrViewport.width;
            height = rrViewport.height;
        }

        if (peerObjectUpdates & RR_CAMERA_UPDATED)
        {
            minirr::Camera rrCamera;
            rr->recvCamera(rrCamera);
            std::memcpy(&mm[0], rrCamera.modelMatrix, sizeof(rrCamera.modelMatrix));
            std::memcpy(&vv[0], rrCamera.viewMatrix, sizeof(rrCamera.viewMatrix));
            std::memcpy(&pr[0], rrCamera.projMatrix, sizeof(rrCamera.projMatrix));
            mv = vv*mm;
        }

        if (peerObjectUpdates & RR_TRANSFUNC_UPDATED)
        {
            minirr::Transfunc rrTransfunc;
            rr->recvTransfunc(rrTransfunc);
            transFunc.colors.resize(rrTransfunc.numRGB);
            std::memcpy((float *)transFunc.colors.data(), rrTransfunc.rgb,
                sizeof(transFunc.colors[0])*transFunc.colors.size());
            transFunc.opacities.resize(rrTransfunc.numAlpha);
            std::memcpy((float *)transFunc.opacities.data(), rrTransfunc.alpha,
                sizeof(transFunc.opacities[0])*transFunc.opacities.size());
            // TODO: ranges, scale
   
            // consume on next renderFrame:
            transFunc.updated = true;
        }
   
        // Send ours:
        if (myObjectUpdates & RR_BOUNDS_UPDATED)
        {
            minirr::AABB rrBounds;
            std::memcpy(&rrBounds[0], &bounds.global.data[0], sizeof(rrBounds));
            rr->sendBounds(rrBounds);
        }
    }

    objectUpdates = 0x0;
#endif

#ifdef ANARI_PLUGIN_HAVE_MPI
    if (mpiSize > 1) {
        MPI_Bcast(&width, 1, MPI_INT, mainRank, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, mainRank, MPI_COMM_WORLD);
        MPI_Bcast(&mm, sizeof(mm), MPI_BYTE, mainRank, MPI_COMM_WORLD);
        MPI_Bcast(&mv, sizeof(mv), MPI_BYTE, mainRank, MPI_COMM_WORLD);
        MPI_Bcast(&pr, sizeof(pr), MPI_BYTE, mainRank, MPI_COMM_WORLD);
    }
#endif

    if (info.frame.width != width || info.frame.height != height) {
        info.frame.width = width;
        info.frame.height = height;
        info.frame.resized = true;

        unsigned imgSize[] = {(unsigned)width,(unsigned)height};
        anariSetParameter(anari.device, anari.frames[chan], "size", ANARI_UINT32_VEC2, imgSize);
        anariCommitParameters(anari.device, anari.frames[chan]);
    }

    if (info.mv != mv || info.pr != pr) {
        info.mv = mv;
        info.pr = pr;

        offaxisStereoCameraFromTransform(
            inverse(pr), inverse(mv), info.eye, info.dir, info.up, info.fovy, info.aspect, info.imgRegion);

        anariSetParameter(anari.device, anari.cameras[chan], "fovy", ANARI_FLOAT32, &info.fovy);
        anariSetParameter(anari.device, anari.cameras[chan], "aspect", ANARI_FLOAT32, &info.aspect);
        anariSetParameter(anari.device, anari.cameras[chan], "position", ANARI_FLOAT32_VEC3, &info.eye.x);
        anariSetParameter(anari.device, anari.cameras[chan], "direction", ANARI_FLOAT32_VEC3, &info.dir.x);
        anariSetParameter(anari.device, anari.cameras[chan], "up", ANARI_FLOAT32_VEC3, &info.up.x);
        anariSetParameter(anari.device, anari.cameras[chan], "imageRegion", ANARI_FLOAT32_BOX2, &info.imgRegion.min);
        anariCommitParameters(anari.device, anari.cameras[chan]);
    }

    if (info.mm != mm) {
        info.mm = mm;
        updateLights(glm2osg(mm));
    }

    anariRenderFrame(anari.device, anari.frames[chan]);
    anariFrameReady(anari.device, anari.frames[chan], ANARI_WAIT);

#ifdef ANARI_PLUGIN_HAVE_RR
    if (isClient) {
        imageBuffer.resize(info.frame.width*info.frame.height);
        rr->recvImage(imageBuffer.data(), info.frame.width, info.frame.height);
    }

    if (isServer) {
        uint32_t widthOUT;
        uint32_t heightOUT;
        ANARIDataType typeOUT;
        const uint32_t *fbPointer = (const uint32_t *)anariMapFrame(anari.device, anari.frames[chan],
                                                                    "channel.color",
                                                                    &widthOUT,
                                                                    &heightOUT,
                                                                    &typeOUT);
        rr->sendImage(fbPointer, widthOUT, heightOUT);
    }
#endif

    // trigger redraw:
    if (isDisplayRank) {
        info.frame.updated = true;
    }
}

void Renderer::drawFrame()
{
    for (unsigned chan=0; chan<numChannels; ++chan) {
        drawFrame(chan);
    }
}

void Renderer::drawFrame(unsigned chan)
{
    const bool isDisplayRank = mpiRank==displayRank;
    if (!isDisplayRank)
        return;

    ChannelInfo &info = channelInfos[chan];
    if (info.frame.resized) {
        multiChannelDrawer->resizeView(
            chan, info.frame.width, info.frame.height, info.frame.colorFormat, info.frame.depthFormat);
        multiChannelDrawer->clearColor(chan);
        multiChannelDrawer->clearDepth(chan);
        info.frame.resized = false;
    }

    if (!info.frame.updated)
        return;

    if (isClient) {
        memcpy((uint32_t *)multiChannelDrawer->rgba(chan), imageBuffer.data(),
               sizeof(imageBuffer[0]) * imageBuffer.size());
#ifdef ANARI_PLUGIN_HAVE_CUDA
    } else if (anari.cudaInterop.enabled) {
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
        transformDepthFromWorldToGL_CUDA(dbPointer, dbXformed, info.eye, info.dir, info.up, info.fovy,
                                    info.aspect, info.imgRegion, info.mv, info.pr, widthOUT, heightOUT);
        CUDA_SAFE_CALL(cudaMemcpyAsync((float *)multiChannelDrawer->depth(chan), dbXformed,
               widthOUT*heightOUT*anari::sizeOf(typeOUT), cudaMemcpyDeviceToDevice,
               anari.cudaInterop.copyStream));
        CUDA_SAFE_CALL(cudaStreamSynchronize(anari.cudaInterop.copyStream));
        CUDA_SAFE_CALL(cudaFree(dbXformed));

        anariUnmapFrame(anari.device, anari.frames[chan], "channel.depthGPU");
#endif
    } else {
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
        transformDepthFromWorldToGL(dbPointer, dbXformed.data(), info.eye, info.dir, info.up, info.fovy,
                                    info.aspect, info.imgRegion, info.mv, info.pr, widthOUT, heightOUT);
        memcpy((float *)multiChannelDrawer->depth(chan), dbXformed.data(),
               widthOUT*heightOUT*anari::sizeOf(typeOUT));

        anariUnmapFrame(anari.device, anari.frames[chan], "channel.depth");
    }

    multiChannelDrawer->swapFrame();

    // frame was consumed:
    info.frame.updated = false;
}

void Renderer::initMPI()
{
#ifdef ANARI_PLUGIN_HAVE_MPI
    int mpiInitCalled = 0;
    MPI_Initialized(&mpiInitCalled);

    if (mpiInitCalled) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    }

    // can be set to a value outside of the range in
    // the config for headless clusters:
    bool displayRankEntryExists = false;
    displayRank  = covise::coCoviseConfig::getInt(
        "displayRank",
        "COVER.Plugin.ANARI.Cluster",
        0,
        &displayRankEntryExists
    );
#endif
}

void Renderer::initRR()
{
#ifdef ANARI_PLUGIN_HAVE_RR
    bool modeEntryExists = false;
    std::string mode = covise::coCoviseConfig::getEntry(
        "mode",
        "COVER.Plugin.ANARI.RR",
        "",
        &modeEntryExists
    );

    bool hostnameEntryExists = false;
    std::string hostname = covise::coCoviseConfig::getEntry(
        "hostname",
        "COVER.Plugin.ANARI.RR",
        "localhost",
        &hostnameEntryExists
    );

    bool portEntryExists = false;
    unsigned short port  = covise::coCoviseConfig::getInt(
        "port",
        "COVER.Plugin.ANARI.RR",
        31050,
        &portEntryExists
    );

    // TODO:
    isServer = mpiRank == mainRank && modeEntryExists && mode == "server";
    isClient = modeEntryExists && mode == "client";

    rr = std::make_shared<minirr::MiniRR>();

    if (isClient) {
        std::cout << "ANARI.RR.mode is 'client'\n";
        rr->initAsClient(hostname, port);
        rr->run();
    }

    if (isServer) {
        std::cout << "ANARI.RR.mode is 'server'\n";
        rr->initAsServer(port);
        rr->run();
    }
#endif
}

void Renderer::initChannels()
{
    const bool isDisplayRank = mpiRank==displayRank;

    if (isDisplayRank) {
        numChannels = coVRConfig::instance()->numChannels();
    }

#ifdef ANARI_PLUGIN_HAVE_RR
    if (isClient) {
        rr->sendNumChannels(numChannels);
    }

    if (isServer) {
        rr->recvNumChannels(numChannels);
    }
#endif

#ifdef ANARI_PLUGIN_HAVE_MPI
    if (mpiSize > 1) {
        MPI_Bcast(&numChannels, 1, MPI_INT, mainRank, MPI_COMM_WORLD);
    }
#endif

    channelInfos.resize(numChannels);
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
    anari.world = anari::newObject<anari::World>(anari.device);
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

    AABB bounds;
    asgComputeBounds(anari.meshes,
                     &bounds.data[0],&bounds.data[1],&bounds.data[2],
                     &bounds.data[3],&bounds.data[4],&bounds.data[5]);
    this->bounds.local.extend(bounds);
    this->bounds.updated = true;
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

    AABB bounds;
    anariGetProperty(
        anari.device, anari.world, "bounds", ANARI_FLOAT32_BOX3, &bounds.data, sizeof(bounds.data), ANARI_WAIT);
    this->bounds.local.extend(bounds);
    this->bounds.updated = true;
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

        // asgComputeBounds doesn't work for volumes yet...
        AABB bounds;
        bounds.data[0] = 0.f;
        bounds.data[1] = 0.f;
        bounds.data[2] = 0.f;
        bounds.data[3] = structuredVolumeData.sizeX;
        bounds.data[4] = structuredVolumeData.sizeY;
        bounds.data[5] = structuredVolumeData.sizeZ;
        this->bounds.local.extend(bounds);
        this->bounds.updated = true;
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

    initTransFunc();

    anariSetParameter(anari.device, anari.amrVolume.volume, "valueRange", ANARI_FLOAT32_BOX1,
                      &data.voxelRange);

    anari::commitParameters(anari.device, anari.amrVolume.volume);

    anari::setAndReleaseParameter(anari.device, anari.world, "volume",
                                  anari::newArray1D(anari.device, &anari.amrVolume.volume));
    anariRelease(anari.device, anari.amrVolume.volume);
    anariCommitParameters(anari.device, anari.world);

    AABB bounds;
    anariGetProperty(
        anari.device, anari.world, "bounds", ANARI_FLOAT32_BOX3, &bounds.data, sizeof(bounds.data), ANARI_WAIT);
    this->bounds.local.extend(bounds);
    this->bounds.updated = true;
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
    anari::setParameter(anari.device, anari.unstructuredVolume.volume, "id", (unsigned)mpiRank);

    initTransFunc();

    anariSetParameter(anari.device, anari.unstructuredVolume.volume, "valueRange", ANARI_FLOAT32_BOX1,
                      &data.dataRange);

    anari::commitParameters(anari.device, anari.unstructuredVolume.volume);

    anari::setAndReleaseParameter(anari.device, anari.world, "volume",
                                  anari::newArray1D(anari.device, &anari.unstructuredVolume.volume));
    anariRelease(anari.device, anari.unstructuredVolume.volume);
    anariCommitParameters(anari.device, anari.world);

    AABB bounds;
    anariGetProperty(
        anari.device, anari.world, "bounds", ANARI_FLOAT32_BOX3, &bounds.data, sizeof(bounds.data), ANARI_WAIT);
    this->bounds.local.extend(bounds);
    this->bounds.updated = true;
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

void Renderer::generateTransFunc()
{
    transFunc.colors.clear();
    transFunc.opacities.clear();

    if (colorByRank) {
        auto c = randomColor(mpiRank);
        transFunc.colors.emplace_back(c.x, c.y, c.z);
    } else {
        // dflt. color map:
        transFunc.colors.emplace_back(0.f, 0.f, 1.f);
        transFunc.colors.emplace_back(0.f, 1.f, 0.f);
        transFunc.colors.emplace_back(1.f, 0.f, 0.f);
    }

    transFunc.opacities.emplace_back(0.f);
    transFunc.opacities.emplace_back(1.f);
}

void Renderer::initTransFunc()
{
    if (anari.amrVolume.volume)
    {
        anari::setAndReleaseParameter(
            anari.device, anari.amrVolume.volume, "color",
            anari::newArray1D(anari.device, transFunc.colors.data(), transFunc.colors.size()));
        anari::setAndReleaseParameter(
            anari.device, anari.amrVolume.volume, "opacity",
            anari::newArray1D(anari.device, transFunc.opacities.data(), transFunc.opacities.size()));
        anari::commitParameters(anari.device, anari.amrVolume.volume);
    }

    if (anari.unstructuredVolume.volume)
    {
        anari::setAndReleaseParameter(
            anari.device, anari.unstructuredVolume.volume, "color",
            anari::newArray1D(anari.device, transFunc.colors.data(), transFunc.colors.size()));
        anari::setAndReleaseParameter(
            anari.device, anari.unstructuredVolume.volume, "opacity",
            anari::newArray1D(anari.device, transFunc.opacities.data(), transFunc.opacities.size()));
        anari::commitParameters(anari.device, anari.unstructuredVolume.volume);
    }
}


