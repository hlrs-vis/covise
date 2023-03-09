#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <anari/anari.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! @file asg.h
 ANARI Scene Graph (ASG) C99 API */

#define ASGAPI /*TODO*/

typedef uint8_t ASGBool_t;
#define ASG_FALSE 0
#define ASG_TRUE  1

typedef int ASGError_t;
#define ASG_ERROR_NO_ERROR                  0
#define ASG_ERROR_INVALID_CHILD_ID          5
#define ASG_ERROR_INVALID_PARENT_ID         6
#define ASG_ERROR_INVALID_PATH_ID           7
#define ASG_ERROR_INSUFFICIENT_ARRAY_SIZE   10
#define ASG_ERROR_PARAM_NOT_FOUND           50
#define ASG_ERROR_MISSING_FILE_HANDLER      100
#define ASG_ERROR_FILE_IO_ERROR             150
#define ASG_ERROR_INVALID_LUT_ID            700

typedef int ASGType_t;
#define ASG_TYPE_OBJECT                     0
#define ASG_TYPE_TRIANGLE_GEOMETRY          1000
#define ASG_TYPE_SPHERE_GEOMETRY            1001
#define ASG_TYPE_CYLINDER_GEOMETRY          1004
#define ASG_TYPE_MATERIAL                   1010
#define ASG_TYPE_LIGHT                      1020
#define ASG_TYPE_SURFACE                    1030
#define ASG_TYPE_LOOKUP_TABLE1D             1040
#define ASG_TYPE_STRUCTURED_VOLUME          1050
#define ASG_TYPE_TRANSFORM                  1060
#define ASG_TYPE_SELECT                     1070
#define ASG_TYPE_CAMERA                     1080

typedef int ASGDataType_t;
#define ASG_DATA_TYPE_INT8                  0
#define ASG_DATA_TYPE_INT8_VEC1             0
#define ASG_DATA_TYPE_INT8_VEC2             1
#define ASG_DATA_TYPE_INT8_VEC3             2
#define ASG_DATA_TYPE_INT8_VEC4             3

#define ASG_DATA_TYPE_INT16                 10
#define ASG_DATA_TYPE_INT16_VEC1            10
#define ASG_DATA_TYPE_INT16_VEC2            11
#define ASG_DATA_TYPE_INT16_VEC3            12
#define ASG_DATA_TYPE_INT16_VEC4            13

#define ASG_DATA_TYPE_INT32                 20
#define ASG_DATA_TYPE_INT32_VEC1            20
#define ASG_DATA_TYPE_INT32_VEC2            21
#define ASG_DATA_TYPE_INT32_VEC3            22
#define ASG_DATA_TYPE_INT32_VEC4            23

#define ASG_DATA_TYPE_INT64                 30
#define ASG_DATA_TYPE_INT64_VEC1            30
#define ASG_DATA_TYPE_INT64_VEC2            31
#define ASG_DATA_TYPE_INT64_VEC3            32
#define ASG_DATA_TYPE_INT64_VEC4            33

#define ASG_DATA_TYPE_UINT8                 100
#define ASG_DATA_TYPE_UINT8_VEC1            100
#define ASG_DATA_TYPE_UINT8_VEC2            101
#define ASG_DATA_TYPE_UINT8_VEC3            102
#define ASG_DATA_TYPE_UINT8_VEC4            103

#define ASG_DATA_TYPE_UINT16                110
#define ASG_DATA_TYPE_UINT16_VEC1           110
#define ASG_DATA_TYPE_UINT16_VEC2           111
#define ASG_DATA_TYPE_UINT16_VEC3           112
#define ASG_DATA_TYPE_UINT16_VEC4           113

#define ASG_DATA_TYPE_UINT32                120
#define ASG_DATA_TYPE_UINT32_VEC1           120
#define ASG_DATA_TYPE_UINT32_VEC2           121
#define ASG_DATA_TYPE_UINT32_VEC3           122
#define ASG_DATA_TYPE_UINT32_VEC4           123

#define ASG_DATA_TYPE_UINT64                130
#define ASG_DATA_TYPE_UINT64_VEC1           130
#define ASG_DATA_TYPE_UINT64_VEC2           131
#define ASG_DATA_TYPE_UINT64_VEC3           132
#define ASG_DATA_TYPE_UINT64_VEC4           133

#define ASG_DATA_TYPE_FLOAT32               230
#define ASG_DATA_TYPE_FLOAT32_VEC1          230
#define ASG_DATA_TYPE_FLOAT32_VEC2          231
#define ASG_DATA_TYPE_FLOAT32_VEC3          232
#define ASG_DATA_TYPE_FLOAT32_VEC4          233

#define ASG_DATA_TYPE_FLOAT64               300

#define ASG_DATA_TYPE_HANDLE                666

typedef int ASGVisitorTraversalType_t;
#define ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN 0
#define ASG_VISITOR_TRAVERSAL_TYPE_PARENTS  1

typedef int ASGMatrixFormat_t;
#define ASG_MATRIX_FORMAT_COL_MAJOR         0
#define ASG_MATRIX_FORMAT_ROW_MAJOR         1

typedef int ASGLutID;
#define ASG_LUT_ID_DEFAULT_LUT              0

#define ASG_IO_FLAG_RESAMPLE_VOLUME_DIMS    1
#define ASG_IO_FLAG_RESAMPLE_VOXEL_TYPE     2

typedef uint64_t ASGBuildWorldFlags_t;
#define ASG_BUILD_WORLD_FLAG_FULL_REBUILD   0xFFFFFFFFFFFFFFFFULL
#define ASG_BUILD_WORLD_FLAG_GEOMETRIES     0x0000000000000001ULL
#define ASG_BUILD_WORLD_FLAG_VOLUMES        0x0000000000000002ULL
#define ASG_BUILD_WORLD_FLAG_MATERIALS      0x0000000000000004ULL
#define ASG_BUILD_WORLD_FLAG_LIGHTS         0x0000000000000008ULL
#define ASG_BUILD_WORLD_FLAG_TRANSFORMS     0x0000000000000010ULL
#define ASG_BUILD_WORLD_FLAG_LUTS           0x0000000000000020ULL

#ifdef __cplusplus
#define ASG_DFLT_PARAM(P) =P
#else
#define ASG_DFLT_PARAM(P)
#endif

struct _ASGObject;
struct _ASGVisitor;

typedef void (*ASGVisitFunc)(struct _ASGVisitor*, struct _ASGObject*, void*);
typedef void (*ASGFreeFunc)(void*);

typedef struct _ASGVisitor *ASGVisitor;

/* ========================================================
 * ASGParam
 * ========================================================*/

typedef struct {
    const char* name;
    ASGDataType_t type;
    struct { uint32_t r0,r1,r2,r3; } value;
} ASGParam;


/* ========================================================
 * ASGObject
 * ========================================================*/

typedef struct _ASGObject *ASGObject, *ASGGeometry, *ASGTriangleGeometry,
    *ASGSphereGeometry, *ASGCylinderGeometry, *ASGMaterial, *ASGLight, *ASGSurface,
    *ASGSampler2D, *ASGLookupTable1D, *ASGStructuredVolume, *ASGTransform, *ASGSelect,
    *ASGCamera;


/* ========================================================
 * API functions
 * ========================================================*/

//Helpers
ASGAPI size_t asgSizeOfDataType(ASGDataType_t type);

/*! @defgroup ASGParam Parametric objects
  ASG follows the principle that different rendering APIs expect different
  parameters for defining objects such as lights, cameras, materials, etc. As
  an example, a classic, fixed-function oriented rasterization-centric ANARI
  device might support material and lighting models that align with the
  traditional Blinn-Phong model and delta light sources, while a ray tracing
  engine will support physically based materials and lights. ASG acknowledges
  this and allows the user to specify named parameters of different scalar,
  vector, or sampler types to allow for highest flexibility. ASGParam serves as
  a base type for such specialized parametric object types */
/*! @{*/
ASGAPI ASGParam asgParam1i(const char* name, int v1);
ASGAPI ASGParam asgParam2i(const char* name, int v1, int v2);
ASGAPI ASGParam asgParam3i(const char* name, int v1, int v2, int v3);
ASGAPI ASGParam asgParam4i(const char* name, int v1, int v2, int v3, int v4);
ASGAPI ASGParam asgParam2iv(const char* name, int* v);
ASGAPI ASGParam asgParam3iv(const char* name, int* v);
ASGAPI ASGParam asgParam4iv(const char* name, int* v);

ASGAPI ASGParam asgParam1f(const char* name, float v1);
ASGAPI ASGParam asgParam2f(const char* name, float v1, float v2);
ASGAPI ASGParam asgParam3f(const char* name, float v1, float v2, float v3);
ASGAPI ASGParam asgParam4f(const char* name, float v1, float v2, float v3, float v4);
ASGAPI ASGParam asgParam2fv(const char* name, float* v);
ASGAPI ASGParam asgParam3fv(const char* name, float* v);
ASGAPI ASGParam asgParam4fv(const char* name, float* v);

ASGAPI ASGParam asgParamSampler2D(const char* name, ASGSampler2D samp);

ASGAPI ASGError_t asgParamGetValue(ASGParam param, void* mem);
/*! @}*/

/*! #defgroup ASGObject Ref-counted objects
  These are the base type that scene graphs are composed of; objects follow the
  same ref counting semantics as ANARI. Every object in the scene graph derives
  from this common object type. Constructor functions are denoted asgNewXXX(),
  where XXX is the name of the type; this naming scheme indicates that the type
  is derived from ASGObject */
/*! @{*/
ASGAPI ASGObject asgNewObject();
ASGAPI ASGError_t asgRelease(ASGObject obj);
ASGAPI ASGError_t asgRetain(ASGObject obj);
ASGAPI ASGError_t asgGetType(ASGObject obj, ASGType_t* type);
ASGAPI ASGError_t asgObjectSetName(ASGObject obj, const char* name);
ASGAPI ASGError_t asgObjectGetName(ASGObject obj, const char** name);
ASGAPI ASGError_t asgObjectAddChild(ASGObject obj, ASGObject child);
ASGAPI ASGError_t asgObjectSetChild(ASGObject obj, int childID, ASGObject child);
ASGAPI ASGError_t asgObjectGetChild(ASGObject obj, int childID, ASGObject* child);
ASGAPI ASGError_t asgObjectGetChildren(ASGObject obj, ASGObject* children,
                                       int* numChildren);
ASGAPI ASGError_t asgObjectRemoveChild(ASGObject obj, ASGObject child);
ASGAPI ASGError_t asgObjectRemoveChildAt(ASGObject obj, int childID);
ASGAPI ASGError_t asgObjectGetParent(ASGObject obj, int parentID, ASGObject* parent);
ASGAPI ASGError_t asgObjectGetParents(ASGObject obj, ASGObject* parents,
                                      int* numParents);
ASGAPI ASGError_t asgObjectGetChildPaths(ASGObject obj, ASGObject target,
                                         ASGObject** paths, int** pathLengths,
                                         int* numPaths);
ASGAPI ASGError_t asgObjectGetParentPaths(ASGObject obj, ASGObject target,
                                          ASGObject** paths, int** pathLengths,
                                          int* numPaths);
ASGAPI ASGError_t asgObjectAccept(ASGObject obj, ASGVisitor visitor);
/*! @}*/

/*! @defgroup ASGVisitor Visitor */
/*! @{*/
ASGAPI ASGVisitor asgCreateVisitor(void (*visitFunc)(ASGVisitor, ASGObject, void*),
                                   void* userData,
                                   ASGVisitorTraversalType_t traversalType);
ASGAPI ASGError_t asgDestroyVisitor(ASGVisitor visitor);
ASGAPI ASGError_t asgVisitorApply(ASGVisitor visitor, ASGObject obj);
/*! @}*/

/*! @defgroup ASGSelect Select nodes
  Select nodes are group nodes that store a visibility flag for each child node
  ANARI groups/worlds */
/*! @{*/

/*! Construct select node. Subtrees that are set to invisible are later culled
  by visitors.@param defaultVisibility Visibility assigned to newly added
  children */
ASGAPI ASGSelect asgNewSelect(ASGBool_t defaultVisibility ASG_DFLT_PARAM(ASG_TRUE));
ASGAPI ASGError_t asgSelectSetDefaultVisibility(ASGSelect select,
                                                ASGBool_t defaultVisibility);
ASGAPI ASGError_t asgSelectGetDefaultVisibility(ASGSelect select,
                                                ASGBool_t* defaultVisibility);
ASGAPI ASGError_t asgSelectSetChildVisible(ASGSelect select, int childID,
                                           ASGBool_t visible);
ASGAPI ASGError_t asgSelectGetChildVisible(ASGSelect select, int childID,
                                           ASGBool_t*visible);
/*! @}*/

/*! @defgroup ASGCamera Cameras */
/*! @{*/
ASGAPI ASGCamera asgNewCamera(const char* cameraType);
ASGAPI ASGError_t asgCameraGetType(ASGCamera camera, const char** cameraType);
ASGAPI ASGError_t asgCameraSetParam(ASGCamera camera, ASGParam param);
ASGAPI ASGError_t asgCameraGetParam(ASGCamera camera, const char* paramName,
                                    ASGParam* param);
/*! @}*/

/*! @defgroup ASGMaterial Materials
  Materials intentionally are simple collections of ASGParam's; the
  application, and also the ANARI implementation, are relatively free to
  interpret these in any way imaginable. Materials are directly placed in the
  scene graph where they affect all the surfaces etc. underneath */
/*! @{*/
ASGAPI ASGMaterial asgNewMaterial(const char* materialType);
ASGAPI ASGError_t asgMaterialGetType(ASGMaterial material, const char** materialType);
ASGAPI ASGError_t asgMaterialSetParam(ASGMaterial material, ASGParam param);
ASGAPI ASGError_t asgMaterialGetParam(ASGMaterial material, const char* paramName,
                                      ASGParam* param);
/*! @}*/

/*! @defgroup ASGLight Lights
  Similar to materials, lights are just collecionts of ASGParam's. This allows
  the user to define more complicated lights such as area lights, HDRI, etc. */
/*! @{*/
ASGAPI ASGLight asgNewLight(const char* lightType);
ASGAPI ASGError_t asgLightGetType(ASGLight light, const char** lightType);
ASGAPI ASGError_t asgLightSetParam(ASGLight light, ASGParam param);
ASGAPI ASGError_t asgLightGetParam(ASGLight light, const char* paramName,
                                   ASGParam* param);
/*! @}*/

// Geometries
ASGAPI ASGTriangleGeometry asgNewTriangleGeometry(float* vertices, float* vertexNormals,
                                                  float* vertexColors,
                                                  uint32_t numVertices, uint32_t* indices,
                                                  uint32_t numIncidices,
                                                  ASGFreeFunc freeVertices
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeNormals
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeColors
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeIndices
                                                  ASG_DFLT_PARAM(NULL));
// TODO: special handling for 64-bit triangle indices (asgNewTriangleGeometry64?)
ASGAPI ASGError_t asgTriangleGeometryGetVertices(ASGTriangleGeometry geom,
                                                 float** vertices);
ASGAPI ASGError_t asgTriangleGeometryGetVertexNormals(ASGTriangleGeometry geom,
                                                      float** vertexNormals);
ASGAPI ASGError_t asgTriangleGeometryGetVertexColors(ASGTriangleGeometry geom,
                                                     float** vertexColors);
ASGAPI ASGError_t asgTriangleGeometryGetNumVertices(ASGTriangleGeometry geom,
                                                    uint32_t* numVertices);
ASGAPI ASGError_t asgTriangleGeometryGetIndices(ASGTriangleGeometry geom,
                                                uint32_t** indices);
ASGAPI ASGError_t asgTriangleGeometryGetNumIndices(ASGTriangleGeometry geom,
                                                   uint32_t* numIndices);

ASGAPI ASGSphereGeometry asgNewSphereGeometry(float* vertices, float* radii,
                                              float* vertexColors, uint32_t numVertices,
                                              uint32_t* indices, uint32_t numIndices,
                                              float defaultRadius ASG_DFLT_PARAM(1.f),
                                              ASGFreeFunc freeVertices
                                              ASG_DFLT_PARAM(NULL),
                                              ASGFreeFunc freeRadii ASG_DFLT_PARAM(NULL),
                                              ASGFreeFunc freeColors
                                              ASG_DFLT_PARAM(NULL),
                                              ASGFreeFunc freeIndices
                                              ASG_DFLT_PARAM(NULL));

ASGAPI ASGCylinderGeometry asgNewCylinderGeometry(float* vertices, float* radii,
                                                  float* vertexColors, uint8_t* caps,
                                                  uint32_t numVertices,
                                                  uint32_t* indices, uint32_t numIndices,
                                                  float defaultRadius ASG_DFLT_PARAM(1.f),
                                                  ASGFreeFunc freeVertices
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeRadii
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeColors
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeCaps
                                                  ASG_DFLT_PARAM(NULL),
                                                  ASGFreeFunc freeIndices
                                                  ASG_DFLT_PARAM(NULL));

ASGAPI ASGError_t asgGeometryComputeBounds(ASGGeometry geom,
                                           float* minX, float* minY, float* minZ,
                                           float* maxX, float* maxY, float* maxZ);

// Surface
ASGAPI ASGSurface asgNewSurface(ASGGeometry geom, ASGMaterial mat);
ASGAPI ASGGeometry asgSurfaceGetGeometry(ASGSurface surf, ASGGeometry* geom);
ASGAPI ASGMaterial asgSurfaceGetMaterial(ASGSurface surf, ASGMaterial* mat);

// Transform
ASGAPI ASGTransform asgNewTransform(float initialMatrix[12], ASGMatrixFormat_t format
                                    ASG_DFLT_PARAM(ASG_MATRIX_FORMAT_COL_MAJOR));
ASGAPI ASGError_t asgTransformSetMatrix(ASGTransform trans, float matrix[12]);
ASGAPI ASGError_t asgTransformGetMatrix(ASGTransform trans, float matrix[12]);
ASGAPI ASGError_t asgTransformRotate(ASGTransform trans, float axis[3],
                                     float angleInRadians);
ASGAPI ASGError_t asgTransformTranslate(ASGTransform trans, float xyz[3]);

// RGBA luts
ASGAPI ASGLookupTable1D asgNewLookupTable1D(float* rgb, float* alpha, int32_t numEntries,
                                            ASGFreeFunc freeRGB ASG_DFLT_PARAM(NULL),
                                            ASGFreeFunc freeAlpha ASG_DFLT_PARAM(NULL));
ASGAPI ASGError_t asgLookupTable1DGetRGB(ASGLookupTable1D lut, float** rgb);
ASGAPI ASGError_t asgLookupTable1DGetAlpha(ASGLookupTable1D lut, float** alpha);
ASGAPI ASGError_t asgLookupTable1DGetNumEntries(ASGLookupTable1D lut,
                                                int32_t* numEntries);

// Volumes
ASGAPI ASGStructuredVolume asgNewStructuredVolume(void* data, int32_t width,
                                                  int32_t height, int32_t depth,
                                                  ASGDataType_t type,
                                                  ASGFreeFunc freeData
                                                  ASG_DFLT_PARAM(NULL));
ASGAPI ASGError_t asgStructuredVolumeGetData(ASGStructuredVolume vol, void** data);
ASGAPI ASGError_t asgStructuredVolumeGetDims(ASGStructuredVolume vol, int32_t* width,
                                             int32_t* height, int32_t* depth);
ASGAPI ASGError_t asgStructuredVolumeGetDatatype(ASGStructuredVolume vol,
                                                 ASGDataType_t* type);
ASGAPI ASGError_t asgStructuredVolumeSetRange(ASGStructuredVolume vol, float rangeMin,
                                              float rangeMax);
ASGAPI ASGError_t asgStructuredVolumeGetRange(ASGStructuredVolume vol, float* rangeMin,
                                              float* rangeMax);
ASGAPI ASGError_t asgStructuredVolumeSetDist(ASGStructuredVolume, float distX,
                                             float distY, float distZ);
ASGAPI ASGError_t asgStructuredVolumeGetDist(ASGStructuredVolume, float* distX,
                                             float* distY, float* distZ);
ASGAPI ASGError_t asgStructuredVolumeSetLookupTable1D(ASGStructuredVolume vol,
                                                      ASGLookupTable1D lut);
ASGAPI ASGError_t asgStructuredVolumeGetLookupTable1D(ASGStructuredVolume vol,
                                                      ASGLookupTable1D* lut);

/*! @defgroup IO I/O */
/*! @{*/
ASGAPI ASGError_t asgLoadASSIMP(ASGObject obj, const char* fileName, uint64_t flags);
ASGAPI ASGError_t asgLoadPBRT(ASGObject obj, const char* fileName, uint64_t flags);
ASGAPI ASGError_t asgLoadVOLKIT(ASGStructuredVolume vol, const char* fileName,
                                uint64_t flags);
/*! @}*/

// Procedural volumes, builtin materials, delta lights, RGBA LUTs, etc.
ASGAPI ASGError_t asgMakeMarschnerLobb(ASGStructuredVolume vol);
ASGAPI ASGError_t asgMakeDefaultLUT1D(ASGLookupTable1D lut, ASGLutID lutID);
ASGAPI ASGError_t asgMakeMatte(ASGMaterial* material, float kd[3],
                               ASGSampler2D mapKD ASG_DFLT_PARAM(NULL));
ASGAPI ASGError_t asgMakePointLight(ASGLight* light, float position[3], float color[3],
                                    float intensity ASG_DFLT_PARAM(1.f));

// Builtin visitors / routines that traverse the whole graph

ASGAPI ASGError_t asgComputeBounds(ASGObject obj, float* minX, float* minY, float* minZ,
                                   float* maxX, float* maxY, float* maxZ,
                                   uint64_t nodeMask ASG_DFLT_PARAM(0));

ASGAPI ASGError_t asgPickObject(ASGObject obj, ASGCamera camera, uint32_t x, uint32_t y,
                                uint32_t frameSizeX, uint32_t frameSizeY,
                                ASGObject* pickedObject,
                                uint64_t nodeMask ASG_DFLT_PARAM(0));

/*! Build ANARI world from ASG subgraph
  Visits the subgraph induced by @param obj and updates the ANARI world
  accordingly. The routine tries to only update those nodes that have the dirty
  flag set */
ASGAPI ASGError_t asgBuildANARIWorld(ASGObject obj, ANARIDevice device, ANARIWorld world,
                                     ASGBuildWorldFlags_t flags
                                     ASG_DFLT_PARAM(ASG_BUILD_WORLD_FLAG_FULL_REBUILD),
                                     uint64_t nodeMask ASG_DFLT_PARAM(0));

#ifdef __cplusplus
}
#endif


