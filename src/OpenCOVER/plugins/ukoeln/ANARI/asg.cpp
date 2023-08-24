#include <assert.h>
#include <float.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#if ASG_HAVE_ASSIMP
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
//#include <assimp/Matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#endif
#if ASG_HAVE_PBRT_PARSER
#include <pbrtParser/Scene.h>
#endif
#if ASG_HAVE_VOLKIT
#include <vkt/InputStream.hpp>
#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/VolumeFile.hpp>
#endif
// For picking, currently using a Visionaray LBVH that is built lazily
// when the object is first tested for intersection. Removing that
// (and the whole dependency against that lib) as soon as ANARI supports
// picking natively
#ifndef ASG_USE_VISIONARAY_FOR_PICKING
#define ASG_USE_VISIONARAY_FOR_PICKING 1
#endif
#if ASG_USE_VISIONARAY_FOR_PICKING
#include <visionaray/bvh.h>
#include <visionaray/traverse.h>
#endif
#include "asg.h"
#include "linalg.hpp"

// ========================================================
// Data types
// ========================================================

struct TypeInfo {
    ASGDataType_t type;
    ANARIDataType anariType;
    size_t size;
};

static const TypeInfo g_typeInfo[] = {
    { ASG_DATA_TYPE_INT8,        ANARI_INT8,           1  },
    { ASG_DATA_TYPE_INT8_VEC1,   ANARI_INT8,           1  },
    { ASG_DATA_TYPE_INT8_VEC2,   ANARI_INT8_VEC2,      2  },
    { ASG_DATA_TYPE_INT8_VEC3,   ANARI_INT8_VEC3,      3  },
    { ASG_DATA_TYPE_INT8_VEC4,   ANARI_INT8_VEC4,      4  },

    { ASG_DATA_TYPE_INT16,       ANARI_INT16,          2  },
    { ASG_DATA_TYPE_INT16_VEC1,  ANARI_INT16,          2  },
    { ASG_DATA_TYPE_INT16_VEC2,  ANARI_INT16_VEC2,     4  },
    { ASG_DATA_TYPE_INT16_VEC3,  ANARI_INT16_VEC3,     6  },
    { ASG_DATA_TYPE_INT16_VEC4,  ANARI_INT16_VEC4,     8  },

    { ASG_DATA_TYPE_INT32,       ANARI_INT32,          4  },
    { ASG_DATA_TYPE_INT32_VEC1,  ANARI_INT32,          4  },
    { ASG_DATA_TYPE_INT32_VEC2,  ANARI_INT32_VEC2,     8  },
    { ASG_DATA_TYPE_INT32_VEC3,  ANARI_INT32_VEC3,    12  },
    { ASG_DATA_TYPE_INT32_VEC4,  ANARI_INT32_VEC4,    16  },

    { ASG_DATA_TYPE_INT64,       ANARI_INT64,          8  },
    { ASG_DATA_TYPE_INT64_VEC1,  ANARI_INT64,          8  },
    { ASG_DATA_TYPE_INT64_VEC2,  ANARI_INT64_VEC2,    16  },
    { ASG_DATA_TYPE_INT64_VEC3,  ANARI_INT64_VEC3,    24  },
    { ASG_DATA_TYPE_INT64_VEC4,  ANARI_INT64_VEC4,    32  },

    { ASG_DATA_TYPE_UINT8,       ANARI_UINT8,          1  },
    { ASG_DATA_TYPE_UINT8_VEC1,  ANARI_UINT8,          1  },
    { ASG_DATA_TYPE_UINT8_VEC2,  ANARI_UINT8_VEC2,     2  },
    { ASG_DATA_TYPE_UINT8_VEC3,  ANARI_UINT8_VEC3,     3  },
    { ASG_DATA_TYPE_UINT8_VEC4,  ANARI_UINT8_VEC4,     4  },

    { ASG_DATA_TYPE_UINT16,      ANARI_UINT16,         2  },
    { ASG_DATA_TYPE_UINT16_VEC1, ANARI_UINT16,         2  },
    { ASG_DATA_TYPE_UINT16_VEC2, ANARI_UINT16_VEC2,    4  },
    { ASG_DATA_TYPE_UINT16_VEC3, ANARI_UINT16_VEC3,    6  },
    { ASG_DATA_TYPE_UINT16_VEC4, ANARI_UINT16_VEC4,    8  },

    { ASG_DATA_TYPE_UINT32,      ANARI_UINT32,         4  },
    { ASG_DATA_TYPE_UINT32_VEC1, ANARI_UINT32,         4  },
    { ASG_DATA_TYPE_UINT32_VEC2, ANARI_UINT32_VEC2,    8  },
    { ASG_DATA_TYPE_UINT32_VEC3, ANARI_UINT32_VEC3,   12  },
    { ASG_DATA_TYPE_UINT32_VEC4, ANARI_UINT32_VEC4,   16  },

    { ASG_DATA_TYPE_UINT64,      ANARI_UINT64,         8  },
    { ASG_DATA_TYPE_UINT64_VEC1, ANARI_UINT64,         8  },
    { ASG_DATA_TYPE_UINT64_VEC2, ANARI_UINT64_VEC2,   16  },
    { ASG_DATA_TYPE_UINT64_VEC3, ANARI_UINT64_VEC3,   24  },
    { ASG_DATA_TYPE_UINT64_VEC4, ANARI_UINT64_VEC4,   32  },

    { ASG_DATA_TYPE_FLOAT32,      ANARI_FLOAT32,       4  },
    { ASG_DATA_TYPE_FLOAT32_VEC1, ANARI_FLOAT32,       4  },
    { ASG_DATA_TYPE_FLOAT32_VEC2, ANARI_FLOAT32_VEC2,  8  },
    { ASG_DATA_TYPE_FLOAT32_VEC3, ANARI_FLOAT32_VEC3, 12  },
    { ASG_DATA_TYPE_FLOAT32_VEC4, ANARI_FLOAT32_VEC4, 16  },

    { ASG_DATA_TYPE_HANDLE, ANARI_OBJECT /*TODO!!*/,  sizeof(void*)   },
};

size_t asgSizeOfDataType(ASGDataType_t type)
{
    auto begin = g_typeInfo;
    auto end = g_typeInfo + sizeof(g_typeInfo)/sizeof(TypeInfo);
    auto it = std::find_if(begin,end,[type](TypeInfo info)
                                     { return info.type==type; });

    if (it != end)
        return it->size;

    return size_t(-1);
}

// ========================================================
// ASGParam
// ========================================================

#define PARAM_DEF1(TYPE,TYPENAME,MNEMONIC)                                              \
ASGParam asgParam1##MNEMONIC(const char* name, TYPE v1) {                               \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME;                                                              \
    std::memcpy(&param.value,&v1,asgSizeOfDataType(TYPENAME));                          \
    return param;                                                                       \
}

#define PARAM_DEF2(TYPE,TYPENAME,MNEMONIC)                                              \
ASGParam asgParam2##MNEMONIC(const char* name, TYPE v1, TYPE v2) {                      \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME##_VEC2;                                                       \
    size_t sodt = asgSizeOfDataType(TYPENAME);                                          \
    std::memcpy(&param.value,&v1,sodt);                                                 \
    std::memcpy((char*)(&param.value)+sodt,&v2,sodt);                                   \
    return param;                                                                       \
}

#define PARAM_DEF3(TYPE,TYPENAME,MNEMONIC)                                              \
ASGParam asgParam3##MNEMONIC(const char* name, TYPE v1, TYPE v2, TYPE v3) {             \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME##_VEC3;                                                       \
    size_t sodt = asgSizeOfDataType(TYPENAME);                                          \
    std::memcpy(&param.value,&v1,sodt);                                                 \
    std::memcpy((char*)(&param.value)+sodt,&v2,sodt);                                   \
    std::memcpy((char*)(&param.value)+2*sodt,&v3,sodt);                                 \
    return param;                                                                       \
}

#define PARAM_DEF4(TYPE,TYPENAME,MNEMONIC)                                              \
ASGParam asgParam4##MNEMONIC(const char* name, TYPE v1, TYPE v2, TYPE v3, TYPE v4) {    \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME##_VEC4;                                                       \
    size_t sodt = asgSizeOfDataType(TYPENAME);                                          \
    std::memcpy(&param.value,&v1,sodt);                                                 \
    std::memcpy((char*)(&param.value)+sodt,&v2,sodt);                                   \
    std::memcpy((char*)(&param.value)+2*sodt,&v3,sodt);                                 \
    std::memcpy((char*)(&param.value)+3*sodt,&v4,sodt);                                 \
    return param;                                                                       \
}

#define PARAM_DEF2V(TYPE,TYPENAME,MNEMONIC)                                             \
ASGParam asgParam2##MNEMONIC(const char* name, TYPE* v) {                               \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME;                                                              \
    std::memcpy(&param.value,v,asgSizeOfDataType(TYPENAME));                            \
    return param;                                                                       \
}

#define PARAM_DEF3V(TYPE,TYPENAME,MNEMONIC)                                             \
ASGParam asgParam3##MNEMONIC(const char* name, TYPE* v) {                               \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME;                                                              \
    std::memcpy(&param.value,v,asgSizeOfDataType(TYPENAME));                            \
    return param;                                                                       \
}

#define PARAM_DEF4V(TYPE,TYPENAME,MNEMONIC)                                             \
ASGParam asgParam4##MNEMONIC(const char* name, TYPE* v) {                               \
    ASGParam param;                                                                     \
    param.name = name;                                                                  \
    param.type = TYPENAME;                                                              \
    std::memcpy(&param.value,v,asgSizeOfDataType(TYPENAME));                            \
    return param;                                                                       \
}

PARAM_DEF1(int,ASG_DATA_TYPE_INT32,i)
PARAM_DEF2(int,ASG_DATA_TYPE_INT32,i)
PARAM_DEF3(int,ASG_DATA_TYPE_INT32,i)
PARAM_DEF4(int,ASG_DATA_TYPE_INT32,i)
PARAM_DEF2V(int,ASG_DATA_TYPE_INT32_VEC2,iv)
PARAM_DEF3V(int,ASG_DATA_TYPE_INT32_VEC3,iv)
PARAM_DEF4V(int,ASG_DATA_TYPE_INT32_VEC3,iv)

PARAM_DEF1(float,ASG_DATA_TYPE_FLOAT32,f)
PARAM_DEF2(float,ASG_DATA_TYPE_FLOAT32,f)
PARAM_DEF3(float,ASG_DATA_TYPE_FLOAT32,f)
PARAM_DEF4(float,ASG_DATA_TYPE_FLOAT32,f)
PARAM_DEF2V(float,ASG_DATA_TYPE_FLOAT32_VEC2,fv)
PARAM_DEF3V(float,ASG_DATA_TYPE_FLOAT32_VEC3,fv)
PARAM_DEF4V(float,ASG_DATA_TYPE_FLOAT32_VEC3,fv)

ASGParam asgParamSampler2D(const char* name, ASGSampler2D samp)
{
    ASGParam param;
    param.name = name;
    param.type = ASG_DATA_TYPE_HANDLE;
    // TODO: this stuff here needs validation...
    if (samp == nullptr)
        std::memset(&param.value,0,sizeof(param.value));
    else
        std::memcpy(&param.value,samp,asgSizeOfDataType(ASG_DATA_TYPE_HANDLE));
    return param;
}

ASGError_t asgParamGetValue(ASGParam param, void* mem)
{
    std::memcpy(mem,&param.value,asgSizeOfDataType(param.type));
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Params - classes that support params inherit from this!
// ========================================================

struct Params {
    std::vector<ASGParam> params;

    void setParam(ASGParam param)
    {
        auto it = std::find_if(params.begin(),params.end(),
                               [param](ASGParam p) {
                                    size_t n=std::max(strlen(param.name),strlen(p.name));
                                    return strncmp(param.name,p.name,n)==0;
                               });

        if (it == params.end()) {
            params.push_back(param);
        } else {
            *it = param;
        }
    };

    bool getParam(const char* paramName, ASGParam* param)
    {
        auto it = std::find_if(params.begin(),params.end(),
                               [paramName](ASGParam p) {
                                    size_t n=std::max(strlen(paramName),strlen(p.name));
                                    return strncmp(paramName,p.name,n)==0;
                               });

        if (it != params.end()) {
            ASGParam p = *it;
            std::memcpy(param,&p,sizeof(ASGParam));
            return true;
        } else {
            return false;
        }
    }
};

// ========================================================
// Ref-counted objects
// ========================================================

typedef void _ASGImpl;
typedef _ASGImpl *ASGImpl;

struct _ASGObject {
    // Node interface
    ASGType_t type;
    std::string name;
    std::vector<_ASGObject*> children;
    std::vector<_ASGObject*> parents;

    void (*addChild)(_ASGObject*, _ASGObject*);

    // Ref count (TODO!)

    // Visitable interface
    void (*accept)(_ASGObject*, ASGVisitor);

    // Release function
    void (*release)(_ASGObject*);

    // Private implementation
    ASGImpl impl;

    // Used by visitors to, e.g., cull nodes
    uint64_t mask;
};

struct _ASGVisitor {
    ASGVisitFunc visit;
    void* userData;
    ASGVisitorTraversalType_t traversalType;

    void (*apply)(ASGVisitor, ASGObject); // TODO: C++ ...

    // Used by *internal* visit functions to determine if nodes are visible
    ASGBool_t visible;
};

void _asgAccept(struct _ASGObject* _obj, ASGVisitor _visitor)
{
    _visitor->visit(_visitor, _obj, _visitor->userData);
}

void _asgApply(ASGVisitor _visitor, struct _ASGObject* _obj)
{
    ASGObject obj = (ASGObject)_obj;

    if (_visitor->traversalType==ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN) {
        for (unsigned i=0; i<obj->children.size(); ++i) {
            ASGObject child = obj->children[i];
            child->accept(child,_visitor);
        }
    } else {
        for (unsigned i=0; i<obj->parents.size(); ++i) {
            ASGObject parent = obj->parents[i];
            parent->accept(parent,_visitor);
        }
    }
}

// For selects
void _asgApplyVisible(ASGVisitor _visitor, struct _ASGObject* _obj, ASGBool_t* visibility)
{
    if (visibility != nullptr) {
        ASGObject obj = (ASGObject)_obj;

        if (_visitor->traversalType==ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN) {
            for (unsigned i=0; i<obj->children.size(); ++i) {
                ASGBool_t prev = _visitor->visible;
                _visitor->visible = visibility[i];
                ASGObject child = obj->children[i];
                child->accept(child,_visitor);
                _visitor->visible = prev;
            }
        } else {
            for (unsigned i=0; i<obj->parents.size(); ++i) {
                ASGBool_t prev = _visitor->visible;
                _visitor->visible = visibility[i];
                ASGObject parent = obj->parents[i];
                parent->accept(parent,_visitor);
                _visitor->visible = prev;
            }
        }
    } else {
        _asgApply(_visitor,_obj);
    }
}

ASGObject asgNewObject()
{
    ASGObject obj = new _ASGObject;
    obj->type = ASG_TYPE_OBJECT;
    obj->accept = _asgAccept;
    obj->addChild = [](ASGObject obj, ASGObject child) {
        obj->children.push_back(child);
        child->parents.push_back(obj);
    };
    obj->release = NULL;
    obj->impl = NULL;
    obj->mask = 0ULL;
    return obj;
}

ASGError_t asgGetType(ASGObject obj, ASGType_t* type)
{
    *type = obj->type;

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgRelease(ASGObject obj)
{
    if (obj->release != NULL)
        obj->release(obj);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgRetain(ASGObject obj)
{
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectSetName(ASGObject obj, const char* name)
{
    obj->name = std::string(name);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectGetName(ASGObject obj, const char** name)
{
    *name = obj->name.c_str();

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectAddChild(ASGObject obj, ASGObject child)
{
    obj->addChild(obj,child); // TODO: why do we need this indirection here!?
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectSetChild(ASGObject obj, int childID, ASGObject child)
{
    if (childID >= obj->children.size())
        return ASG_ERROR_INVALID_CHILD_ID;

    // TODO: implement that through forests!
    obj->children[childID] = child;
    // TODO: check if we are parent already?!
    child->parents.push_back(obj);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectGetChild(ASGObject obj, int childID, ASGObject* child)
{
    if (childID >= obj->children.size())
        return ASG_ERROR_INVALID_CHILD_ID;

    *child = obj->children[childID];

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectGetChildren(ASGObject obj, ASGObject* children, int* numChildren)
{
    *numChildren = (int)obj->children.size();
    if (children == NULL) {
        return ASG_ERROR_NO_ERROR;
    } else {
        std::memcpy(children,obj->children.data(),obj->children.size()*sizeof(ASGObject));
        return ASG_ERROR_NO_ERROR;
    }
}

ASGError_t asgObjectRemoveChild(ASGObject obj, ASGObject child)
{
    // from the child that we're going to remove, first remove obj as a parent
    std::vector<ASGObject>& parents = child->parents;
    parents.erase(std::remove_if(parents.begin(),parents.end(),
                                 [obj](ASGObject parent) { return parent==obj; }),
                  parents.end());

    // Now remove the child from the list of children (however unlikely, we _might_
    // be storing multiple references to that child..
    std::vector<ASGObject>& children = obj->children;
    children.erase(std::remove_if(children.begin(),children.end(),
                                  [child](ASGObject c) { return c==child; }),
                  children.end());

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectRemoveChildAt(ASGObject obj, int childID)
{
    if (childID >= obj->children.size())
        return ASG_ERROR_INVALID_CHILD_ID;

    // from the child that we're going to remove, first remove obj as a parent
    std::vector<ASGObject>& parents = obj->children[childID]->parents;
    parents.erase(std::remove_if(parents.begin(),parents.end(),
                                 [obj](ASGObject parent) { return parent==obj; }),
                  parents.end());

    // Now remove the child from the list of children
    obj->children.erase(obj->children.begin()+childID);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectGetParent(ASGObject obj, int parentID, ASGObject* parent)
{
    if (parentID >= obj->parents.size())
        return ASG_ERROR_INVALID_PARENT_ID;

    *parent = obj->parents[parentID];

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectGetParents(ASGObject obj, ASGObject* parents, int* numParents)
{
    *numParents = (int)obj->parents.size();
    if (parents == NULL) {
        return ASG_ERROR_NO_ERROR;
    } else {
        std::memcpy(parents,obj->parents.data(),obj->parents.size()*sizeof(ASGObject));
        return ASG_ERROR_NO_ERROR;
    }
}

// Assemble node paths

ASGError_t getPaths(ASGObject obj, ASGObject target, int traversalOrder,
                    ASGObject** paths, int** pathLengths, int* numPaths)
{
    if (paths == NULL && *numPaths == 0) {  // count paths

        struct TraversalData {
            ASGObject target;
            int* numPaths;
        };

        TraversalData data { target, numPaths };

        ASGVisitor visitor = asgCreateVisitor(
            [](ASGVisitor self, ASGObject obj, void* userData) {
                TraversalData* data = (TraversalData*)userData;

                if (obj==data->target)
                    (*data->numPaths)++;

                asgVisitorApply(self,obj);
            },&data,traversalOrder);

        asgObjectAccept(obj,visitor);
        asgDestroyVisitor(visitor);
    } else if (paths == NULL) {             // count lengths of all paths

        struct TraversalData {
            ASGObject from;
            ASGObject target;
            int traversalOrder;
            int** pathLengths;
            const int* numPaths;
            int currentPath;
        };

        TraversalData data { obj, target, traversalOrder, pathLengths, numPaths, 0 };

        ASGVisitor visitor = asgCreateVisitor(
            [](ASGVisitor self, ASGObject obj, void* userData) {
                TraversalData* data = (TraversalData*)userData;

                if ((obj != data->target) &&
                    ((data->traversalOrder==ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN
                        && obj->children.empty()) ||
                     (data->traversalOrder==ASG_VISITOR_TRAVERSAL_TYPE_PARENTS
                        && obj->parents.empty())))
                {
                    // Dead-end, reset node count
                    *data->pathLengths[data->currentPath] = 0;
                } else if (obj==data->target) {
                    (*data->pathLengths[data->currentPath])++;
                    data->currentPath++;
                } else if (obj!=data->from) {
                    (*data->pathLengths[data->currentPath])++;
                }

                if (data->currentPath < *data->numPaths)
                    asgVisitorApply(self,obj);

            },&data,traversalOrder);

        asgObjectAccept(obj,visitor);
        asgDestroyVisitor(visitor);
    } else {                                // assemble paths

        struct TraversalData {
            ASGObject from;
            ASGObject target;
            ASGObject** paths;
            int traversalOrder;
            int** pathLengths;
            const int* numPaths;
            std::vector<ASGObject> onePath;
            int currentPath;
        };

        std::vector<ASGObject> onePath;
        TraversalData data { obj, target, paths, traversalOrder, pathLengths, numPaths,
                             onePath, 0 };

        ASGVisitor visitor = asgCreateVisitor(
            [](ASGVisitor self, ASGObject obj, void* userData) {
                TraversalData* data = (TraversalData*)userData;

                if ((obj != data->target) &&
                    ((data->traversalOrder==ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN
                        && obj->children.empty()) ||
                     (data->traversalOrder==ASG_VISITOR_TRAVERSAL_TYPE_PARENTS
                        && obj->parents.empty())))
                {
                    // Dead-end, clear path
                    data->onePath.clear();
                } else if (obj==data->target) {
                    data->onePath.push_back(obj);
                    if ((int)data->onePath.size()
                            == *data->pathLengths[data->currentPath]) {
                        memcpy(data->paths[data->currentPath],data->onePath.data(),
                               data->onePath.size()*sizeof(ASGObject));
                        data->onePath.clear();
                    } else {
                        // TODO: problem!
                    }
                    data->currentPath++;
                } else if (obj!=data->from) {
                    data->onePath.push_back(obj);
                }

                if (data->currentPath < *data->numPaths)
                    asgVisitorApply(self,obj);

            },&data,traversalOrder);

        asgObjectAccept(obj,visitor);
        asgDestroyVisitor(visitor);
    }

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgObjectGetChildPaths(ASGObject obj, ASGObject target, ASGObject** paths,
                                  int** pathLengths, int* numPaths)
{
    return getPaths(obj,target,ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN,paths,pathLengths,
                    numPaths);
}

ASGError_t asgObjectGetParentPaths(ASGObject obj, ASGObject target, ASGObject** paths,
                                   int** pathLengths, int* numPaths)
{
    return getPaths(obj,target,ASG_VISITOR_TRAVERSAL_TYPE_PARENTS,paths,pathLengths,
                    numPaths);
}

ASGError_t asgObjectAccept(ASGObject obj, ASGVisitor visitor)
{
    obj->accept(obj,visitor);
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Visitor
// ========================================================

ASGVisitor asgCreateVisitor(void (*visitFunc)(ASGVisitor, ASGObject, void*),
                            void* userData,
                            ASGVisitorTraversalType_t traversalType)
{
    ASGVisitor visitor = new _ASGVisitor;
    visitor->visit = (ASGVisitFunc)visitFunc;
    visitor->userData = userData;
    visitor->traversalType = traversalType;
    visitor->apply = _asgApply;
    visitor->visible = ASG_TRUE;
    return visitor;
}

ASGError_t asgDestroyVisitor(ASGVisitor visitor)
{
    delete visitor;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgVisitorApply(ASGVisitor visitor, ASGObject obj)
{
    visitor->apply(visitor,obj);
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Select
// ========================================================

struct Select {
    ASGBool_t defaultVisibility;
    std::vector<ASGBool_t> visibility; // per child node
};

ASGSelect asgNewSelect(ASGBool_t defaultVisibility)
{
    ASGSelect select = (ASGSelect)asgNewObject();
    select->type = ASG_TYPE_SELECT;
    select->impl = new Select;
    ((Select*)select->impl)->defaultVisibility = defaultVisibility;

    // Overwrite to add the child node, and to set the
    // visibility accordingly
    // TODO: ASGObject is a candidate for a class with virtual members :-)
    select->addChild = [](ASGObject obj, ASGObject child) {
        ASGSelect select = (ASGSelect)obj;
        select->children.push_back(child);
        child->parents.push_back(obj);
        ((Select*)select->impl)->visibility.push_back(
                ((Select*)select->impl)->defaultVisibility);
    };
    return select;
}

ASGError_t asgSelectSetDefaultVisibility(ASGSelect select, ASGBool_t defaultVisibility)
{
    ((Select*)select->impl)->defaultVisibility = defaultVisibility;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgSelectGetDefaultVisibility(ASGSelect select, ASGBool_t* defaultVisibility)
{
    *defaultVisibility = ((Select*)select->impl)->defaultVisibility;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgSelectSetChildVisible(ASGSelect select, int childID, ASGBool_t visible)
{
    ((Select*)select->impl)->visibility[childID] = visible;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgSelectGetChildVisible(ASGSelect select, int childID, ASGBool_t* visible)
{
    *visible = ((Select*)select->impl)->visibility[childID];
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Camera
// ========================================================

struct Camera : Params {
    std::string type;
};

ASGCamera asgNewCamera(const char* cameraType)
{
    ASGCamera camera = (ASGCamera)asgNewObject();
    camera->type = ASG_TYPE_CAMERA;
    camera->impl = new Camera;
    ((Camera*)camera->impl)->type = cameraType;
    return camera;
}

ASGError_t asgCameraGetType(ASGCamera camera, const char** cameraType)
{
    *cameraType = ((Camera*)camera->impl)->type.c_str();
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgCameraSetParam(ASGCamera camera, ASGParam param)
{
    ((Camera*)camera->impl)->setParam(param);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgCameraGetParam(ASGCamera camera, const char* paramName, ASGParam* param)
{
    bool found = ((Params*)camera->impl)->getParam(paramName,param);

    if (found)
        return ASG_ERROR_NO_ERROR;
    else
        return ASG_ERROR_PARAM_NOT_FOUND;
}

// ========================================================
// Material
// ========================================================

struct Material : Params {
    std::string type;
    // Exclusively used by ANARI build visitors
    ANARIMaterial anariMaterial = NULL;
};

ASGMaterial asgNewMaterial(const char* materialType)
{
    ASGMaterial material = (ASGMaterial)asgNewObject();
    material->type = ASG_TYPE_MATERIAL;
    material->impl = new Material;
    ((Material*)material->impl)->type = materialType;
    return material;
}

ASGError_t asgMaterialGetType(ASGMaterial material, const char** materialType)
{
    *materialType = ((Material*)material->impl)->type.c_str();
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgMaterialSetParam(ASGMaterial material, ASGParam param)
{
    ((Params*)material->impl)->setParam(param);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgMaterialGetParam(ASGMaterial material, const char* paramName,
                               ASGParam* param)
{
    bool found = ((Params*)material->impl)->getParam(paramName,param);

    if (found)
        return ASG_ERROR_NO_ERROR;
    else
        return ASG_ERROR_PARAM_NOT_FOUND;
}

// ========================================================
// Light
// ========================================================

struct Light : Params {
    std::string type;
    // Exclusively used by ANARI build visitors
    ANARILight anariLight = NULL;
};

ASGLight asgNewLight(const char* lightType)
{
    ASGLight light = (ASGLight)asgNewObject();
    light->type = ASG_TYPE_LIGHT;
    light->impl = new Light;
    ((Light*)light->impl)->type = lightType;
    return light;
}

ASGError_t asgLightGetType(ASGLight light, const char** lightType)
{
    *lightType = ((Light*)light->impl)->type.c_str();
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgLightSetParam(ASGLight light, ASGParam param)
{
    ((Params*)light->impl)->setParam(param);

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgLightGetParam(ASGLight light, const char* paramName, ASGParam* param)
{
    bool found = ((Params*)light->impl)->getParam(paramName,param);

    if (found)
        return ASG_ERROR_NO_ERROR;
    else
        return ASG_ERROR_PARAM_NOT_FOUND;
}

// ========================================================
// Geometries
// ========================================================

struct TriangleGeom {
    float* vertices;
    float* vertexNormals;
    float* vertexColors;
    uint32_t numVertices;
    uint32_t* indices;
    uint32_t numIndices;
    float* verticesUserPtr;
    float* vertexNormalsUserPtr;
    float* vertexColorsUserPtr;
    uint32_t* indicesUserPtr;
    ASGFreeFunc freeVertices;
    ASGFreeFunc freeNormals;
    ASGFreeFunc freeColors;
    ASGFreeFunc freeIndices;
    // Exclusively used by ANARI build visitors
    ANARIGeometry anariGeometry = NULL;
#if ASG_USE_VISIONARAY_FOR_PICKING
    visionaray::index_bvh<visionaray::basic_triangle<3,float>> bvh;
#endif
};

void _asgReleaseTriangleGeometry(ASGTriangleGeometry geom)
{
}

ASGTriangleGeometry asgNewTriangleGeometry(float* vertices, float* vertexNormals,
                                           float* vertexColors, uint32_t numVertices,
                                           uint32_t* indices, uint32_t numIndices,
                                           ASGFreeFunc freeVertices,
                                           ASGFreeFunc freeNormals,
                                           ASGFreeFunc freeColors,
                                           ASGFreeFunc freeIndices)
{
    ASGTriangleGeometry geom = (ASGTriangleGeometry)asgNewObject();
    geom->type = ASG_TYPE_TRIANGLE_GEOMETRY;
    geom->release = _asgReleaseTriangleGeometry;
    geom->impl = new TriangleGeom;
    ((TriangleGeom*)geom->impl)->vertices = vertices;
    ((TriangleGeom*)geom->impl)->vertexNormals = vertexNormals;
    ((TriangleGeom*)geom->impl)->vertexColors = vertexColors;
    ((TriangleGeom*)geom->impl)->numVertices = numVertices;
    ((TriangleGeom*)geom->impl)->indices = indices;
    ((TriangleGeom*)geom->impl)->numIndices = numIndices;
    ((TriangleGeom*)geom->impl)->verticesUserPtr = vertices;
    ((TriangleGeom*)geom->impl)->vertexNormalsUserPtr = vertexNormals;
    ((TriangleGeom*)geom->impl)->vertexColorsUserPtr = vertexColors;
    ((TriangleGeom*)geom->impl)->indicesUserPtr = indices;
    ((TriangleGeom*)geom->impl)->freeVertices = freeVertices;
    ((TriangleGeom*)geom->impl)->freeNormals = freeNormals;
    ((TriangleGeom*)geom->impl)->freeColors = freeColors;
    ((TriangleGeom*)geom->impl)->freeIndices = freeIndices;
    return geom;
}

ASGError_t asgTriangleGeometryGetVertices(ASGTriangleGeometry geom, float** vertices)
{
    *vertices = ((TriangleGeom*)geom->impl)->vertices;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTriangleGeometryGetVertexNormals(ASGTriangleGeometry geom,
                                               float** vertexNormals)
{
    *vertexNormals = ((TriangleGeom*)geom->impl)->vertexNormals;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTriangleGeometryGetVertexColors(ASGTriangleGeometry geom,
                                              float** vertexColors)
{
    *vertexColors = ((TriangleGeom*)geom->impl)->vertexColors;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTriangleGeometryGetNumVertices(ASGTriangleGeometry geom,
                                             uint32_t* numVertices)
{
    *numVertices = ((TriangleGeom*)geom->impl)->numVertices;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTriangleGeometryGetIndices(ASGTriangleGeometry geom, uint32_t** indices)
{
    *indices = ((TriangleGeom*)geom->impl)->indices;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTriangleGeometryGetNumIndices(ASGTriangleGeometry geom,
                                            uint32_t* numIndices)
{
    *numIndices = ((TriangleGeom*)geom->impl)->numIndices;
    return ASG_ERROR_NO_ERROR;
}

struct SphereGeom {
    float* vertices;
    float* radii;
    float* vertexColors;
    uint32_t numVertices;
    uint32_t* indices;
    uint32_t numIndices;
    float* verticesUserPtr;
    float* radiiUserPtr;
    float* vertexColorsUserPtr;
    uint32_t* indicesUserPtr;
    float defaultRadius;
    ASGFreeFunc freeVertices;
    ASGFreeFunc freeRadii;
    ASGFreeFunc freeColors;
    ASGFreeFunc freeIndices;
    // Exclusively used by ANARI build visitors
    ANARIGeometry anariGeometry = NULL;
};

ASGSphereGeometry asgNewSphereGeometry(float* vertices, float* radii,
                                       float* vertexColors, uint32_t numVertices,
                                       uint32_t* indices, uint32_t numIndices,
                                       float defaultRadius, ASGFreeFunc freeVertices,
                                       ASGFreeFunc freeRadii, ASGFreeFunc freeColors,
                                       ASGFreeFunc freeIndices)
{
    ASGSphereGeometry geom = (ASGSphereGeometry)asgNewObject();
    geom->type = ASG_TYPE_SPHERE_GEOMETRY;
    geom->impl = new SphereGeom;
    ((SphereGeom*)geom->impl)->vertices = vertices;
    ((SphereGeom*)geom->impl)->radii = radii;
    ((SphereGeom*)geom->impl)->vertexColors = vertexColors;
    ((SphereGeom*)geom->impl)->numVertices = numVertices;
    ((SphereGeom*)geom->impl)->indices = indices;
    ((SphereGeom*)geom->impl)->numIndices = numIndices;
    ((SphereGeom*)geom->impl)->verticesUserPtr = vertices;
    ((SphereGeom*)geom->impl)->radiiUserPtr = radii;
    ((SphereGeom*)geom->impl)->vertexColorsUserPtr = vertexColors;
    ((SphereGeom*)geom->impl)->indices = indices;
    ((SphereGeom*)geom->impl)->defaultRadius = defaultRadius;
    ((SphereGeom*)geom->impl)->freeVertices = freeVertices;
    ((SphereGeom*)geom->impl)->freeRadii = freeRadii;
    ((SphereGeom*)geom->impl)->freeColors = freeColors;
    ((SphereGeom*)geom->impl)->freeIndices = freeIndices;
    return geom;
}

struct CylinderGeom {
    float* vertices;
    float* radii;
    float* vertexColors;
    uint8_t* caps;
    uint32_t numVertices;
    uint32_t* indices;
    uint32_t numIndices;
    float* verticesUserPtr;
    float* radiiUserPtr;
    float* vertexColorsUserPtr;
    uint8_t* capsUserPtr;
    uint32_t* indicesUserPtr;
    float defaultRadius;
    ASGFreeFunc freeVertices;
    ASGFreeFunc freeRadii;
    ASGFreeFunc freeColors;
    ASGFreeFunc freeCaps;
    ASGFreeFunc freeIndices;
    // Exclusively used by ANARI build visitors
    ANARIGeometry anariGeometry = NULL;
};

ASGAPI ASGCylinderGeometry asgNewCylinderGeometry(float* vertices, float* radii,
                                                  float* vertexColors, uint8_t* caps,
                                                  uint32_t numVertices,
                                                  uint32_t* indices, uint32_t numIndices,
                                                  float defaultRadius,
                                                  ASGFreeFunc freeVertices,
                                                  ASGFreeFunc freeRadii,
                                                  ASGFreeFunc freeColors,
                                                  ASGFreeFunc freeCaps,
                                                  ASGFreeFunc freeIndices)
{
    ASGCylinderGeometry geom = (ASGCylinderGeometry)asgNewObject();
    geom->type = ASG_TYPE_CYLINDER_GEOMETRY;
    geom->impl = new CylinderGeom;
    ((CylinderGeom*)geom->impl)->vertices = vertices;
    ((CylinderGeom*)geom->impl)->radii = radii;
    ((CylinderGeom*)geom->impl)->vertexColors = vertexColors;
    ((CylinderGeom*)geom->impl)->caps = caps;
    ((CylinderGeom*)geom->impl)->numVertices = numVertices;
    ((CylinderGeom*)geom->impl)->indices = indices;
    ((CylinderGeom*)geom->impl)->numIndices = numIndices;
    ((CylinderGeom*)geom->impl)->verticesUserPtr = vertices;
    ((CylinderGeom*)geom->impl)->radiiUserPtr = radii;
    ((CylinderGeom*)geom->impl)->vertexColorsUserPtr = vertexColors;
    ((CylinderGeom*)geom->impl)->capsUserPtr = caps;
    ((CylinderGeom*)geom->impl)->indices = indices;
    ((CylinderGeom*)geom->impl)->defaultRadius = defaultRadius;
    ((CylinderGeom*)geom->impl)->freeVertices = freeVertices;
    ((CylinderGeom*)geom->impl)->freeRadii = freeRadii;
    ((CylinderGeom*)geom->impl)->freeColors = freeColors;
    ((CylinderGeom*)geom->impl)->freeCaps = freeCaps;
    ((CylinderGeom*)geom->impl)->freeIndices = freeIndices;
    return geom;
}

ASGError_t asgGeometryComputeBounds(ASGGeometry geom,
                                    float* minX, float* minY, float* minZ,
                                    float* maxX, float* maxY, float* maxZ)
{
    *minX = *minY = *minZ =  FLT_MAX;
    *maxX = *maxY = *maxZ = -FLT_MAX;

    if (geom->type == ASG_TYPE_TRIANGLE_GEOMETRY) {

        TriangleGeom* tg = (TriangleGeom*)geom->impl;

        if (tg->indices != nullptr && tg->numIndices > 0) {
            for (unsigned i=0; i<tg->numIndices; ++i) {
                for (unsigned j=0; j<3; ++j) {
                    unsigned index = i * 3 + j;
                    asg::Vec3f v = {
                        tg->vertices[tg->indices[index]*3],
                        tg->vertices[tg->indices[index]*3+1],
                        tg->vertices[tg->indices[index]*3+2]
                    };

                    *minX = fminf(*minX,v.x);
                    *minY = fminf(*minY,v.y);
                    *minZ = fminf(*minZ,v.z);

                    *maxX = fmaxf(*maxX,v.x);
                    *maxY = fmaxf(*maxY,v.y);
                    *maxZ = fmaxf(*maxZ,v.z);
                }
            }
        } else {
            // No indices, so just insert the verts directly
            for (unsigned i=0; i<tg->numVertices; ++i) {
                asg::Vec3f v = {
                    tg->vertices[i*3],
                    tg->vertices[i*3+1],
                    tg->vertices[i*3+2]
                };

                *minX = fminf(*minX,v.x);
                *minY = fminf(*minY,v.y);
                *minZ = fminf(*minZ,v.z);

                *maxX = fmaxf(*maxX,v.x);
                *maxY = fmaxf(*maxY,v.y);
                *maxZ = fmaxf(*maxZ,v.z);
            }
        }
    } else if (geom->type == ASG_TYPE_SPHERE_GEOMETRY) {

        SphereGeom* sg = (SphereGeom*)geom->impl;

        if (sg->indices != nullptr && sg->numIndices > 0) {
            for (unsigned i=0; i<sg->numIndices; ++i) {
                asg::Vec3f v = {
                    sg->vertices[sg->indices[i]*3],
                    sg->vertices[sg->indices[i]*3+1],
                    sg->vertices[sg->indices[i]*3+2]
                };

                asg::Vec3f v1 = v - sg->radii[sg->indices[i]];
                asg::Vec3f v2 = v + sg->radii[sg->indices[i]];

                *minX = fminf(*minX,v1.x);
                *minY = fminf(*minY,v1.y);
                *minZ = fminf(*minZ,v1.z);

                *maxX = fmaxf(*maxX,v1.x);
                *maxY = fmaxf(*maxY,v1.y);
                *maxZ = fmaxf(*maxZ,v1.z);

                *minX = fminf(*minX,v2.x);
                *minY = fminf(*minY,v2.y);
                *minZ = fminf(*minZ,v2.z);

                *maxX = fmaxf(*maxX,v2.x);
                *maxY = fmaxf(*maxY,v2.y);
                *maxZ = fmaxf(*maxZ,v2.z);
            }
        } else {
            // No indices, so just insert the verts directly
            for (unsigned i=0; i<sg->numVertices; ++i) {
                asg::Vec3f v = {
                    sg->vertices[i*3],
                    sg->vertices[i*3+1],
                    sg->vertices[i*3+2]
                };

                asg::Vec3f v1 = v - sg->radii[i];
                asg::Vec3f v2 = v + sg->radii[i];

                *minX = fminf(*minX,v1.x);
                *minY = fminf(*minY,v1.y);
                *minZ = fminf(*minZ,v1.z);

                *maxX = fmaxf(*maxX,v1.x);
                *maxY = fmaxf(*maxY,v1.y);
                *maxZ = fmaxf(*maxZ,v1.z);

                *minX = fminf(*minX,v2.x);
                *minY = fminf(*minY,v2.y);
                *minZ = fminf(*minZ,v2.z);

                *maxX = fmaxf(*maxX,v2.x);
                *maxY = fmaxf(*maxY,v2.y);
                *maxZ = fmaxf(*maxZ,v2.z);
            }
        }
    } else if (geom->type == ASG_TYPE_CYLINDER_GEOMETRY) {

        CylinderGeom* cg = (CylinderGeom*)geom->impl;

        if (cg->indices != nullptr && cg->numIndices > 0) {
            for (unsigned i=0; i<cg->numIndices; ++i) {

                asg::Vec3f v1 = {
                    cg->vertices[cg->indices[i*2]*3],
                    cg->vertices[cg->indices[i*2]*3+1],
                    cg->vertices[cg->indices[i*2]*3+2]
                };

                asg::Vec3f v2 = {
                    cg->vertices[cg->indices[i*2+1]*3],
                    cg->vertices[cg->indices[i*2+1]*3+1],
                    cg->vertices[cg->indices[i*2+1]*3+2]
                };

                float radius = cg->radii[i];

                asg::Vec3f a = v2 - v1;
                float aa = dot(a,a);
                asg::Vec3f e = {
                    radius * sqrtf(1.f - a.x * a.x / aa),
                    radius * sqrtf(1.f - a.y * a.y / aa),
                    radius * sqrtf(1.f - a.z * a.z / aa)
                };

                asg::Vec3f pa = {
                    fminf(v1.x-e.x,v2.x-e.x),
                    fminf(v1.y-e.y,v2.y-e.y),
                    fminf(v1.z-e.z,v2.z-e.z)
                };

                asg::Vec3f pb = {
                    fmaxf(v1.x+e.x,v2.x+e.x),
                    fmaxf(v1.y+e.y,v2.y+e.y),
                    fmaxf(v1.z+e.z,v2.z+e.z)
                };

                v1 = pa;
                v2 = pb;

                *minX = fminf(*minX,v1.x);
                *minY = fminf(*minY,v1.y);
                *minZ = fminf(*minZ,v1.z);

                *maxX = fmaxf(*maxX,v1.x);
                *maxY = fmaxf(*maxY,v1.y);
                *maxZ = fmaxf(*maxZ,v1.z);

                *minX = fminf(*minX,v2.x);
                *minY = fminf(*minY,v2.y);
                *minZ = fminf(*minZ,v2.z);

                *maxX = fmaxf(*maxX,v2.x);
                *maxY = fmaxf(*maxY,v2.y);
                *maxZ = fmaxf(*maxZ,v2.z);
            }
        } else {
            // No indices, so just insert the verts directly
            for (unsigned i=0; i<cg->numVertices; ++i) {
                asg::Vec3f v1 = {
                    cg->vertices[i*6],
                    cg->vertices[i*6+1],
                    cg->vertices[i*6+2]
                };

                asg::Vec3f v2 = {
                    cg->vertices[i*6+3],
                    cg->vertices[i*6+4],
                    cg->vertices[i*6+5]
                };

                float radius = cg->radii[i];

                asg::Vec3f a = v2 - v1;
                float aa = dot(a,a);
                asg::Vec3f e = {
                    radius * sqrtf(1.f - a.x * a.x / aa),
                    radius * sqrtf(1.f - a.y * a.y / aa),
                    radius * sqrtf(1.f - a.z * a.z / aa)
                };

                asg::Vec3f pa = {
                    fminf(v1.x-e.x,v2.x-e.x),
                    fminf(v1.y-e.y,v2.y-e.y),
                    fminf(v1.z-e.z,v2.z-e.z)
                };

                asg::Vec3f pb = {
                    fmaxf(v1.x+e.x,v2.x+e.x),
                    fmaxf(v1.y+e.y,v2.y+e.y),
                    fmaxf(v1.z+e.z,v2.z+e.z)
                };

                v1 = pa;
                v2 = pb;

                *minX = fminf(*minX,v1.x);
                *minY = fminf(*minY,v1.y);
                *minZ = fminf(*minZ,v1.z);

                *maxX = fmaxf(*maxX,v1.x);
                *maxY = fmaxf(*maxY,v1.y);
                *maxZ = fmaxf(*maxZ,v1.z);

                *minX = fminf(*minX,v2.x);
                *minY = fminf(*minY,v2.y);
                *minZ = fminf(*minZ,v2.z);

                *maxX = fmaxf(*maxX,v2.x);
                *maxY = fmaxf(*maxY,v2.y);
                *maxZ = fmaxf(*maxZ,v2.z);
            }
        }
    }

    return ASG_ERROR_NO_ERROR;
}


// ========================================================
// Surface
// ========================================================

struct Surface {
    ASGGeometry geometry;
    ASGMaterial material;
    // Exclusively used by ANARI build visitors
    ANARISurface anariSurface = NULL;
};

ASGSurface asgNewSurface(ASGGeometry geom, ASGMaterial mat)
{
    ASGSurface surf = (ASGSurface)asgNewObject();
    surf->type = ASG_TYPE_SURFACE;
    surf->impl = new Surface;
    ((Surface*)surf->impl)->geometry = geom;
    ((Surface*)surf->impl)->material = mat;
    return surf;
}

ASGGeometry asgSurfaceGetGeometry(ASGSurface surf, ASGGeometry* geom)
{
    *geom = ((Surface*)surf->impl)->geometry;
    return ASG_ERROR_NO_ERROR;
}

ASGMaterial asgSurfaceGetMaterial(ASGSurface surf, ASGMaterial* mat)
{
    *mat = ((Surface*)surf->impl)->material;
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Transform
// ========================================================

struct Transform {
    float matrix[12];
    // Exclusively used by ANARI build visitors
    ANARIInstance anariInstance = NULL;
};

ASGTransform asgNewTransform(float initialMatrix[12], ASGMatrixFormat_t format)
{
    ASGTransform trans = (ASGTransform)asgNewObject();
    trans->type = ASG_TYPE_TRANSFORM;
    trans->impl = new Transform;
    std::memcpy(((Transform*)trans->impl)->matrix,initialMatrix,12*sizeof(float));
    return trans;
}

ASGError_t asgTransformSetMatrix(ASGTransform trans, float matrix[12])
{
    std::memcpy(((Transform*)trans->impl)->matrix,(const void*)matrix,12*sizeof(float));
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTransformGetMatrix(ASGTransform trans, float matrix[12])
{
    std::memcpy(matrix,((Transform*)trans->impl)->matrix,12*sizeof(float));
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTransformRotate(ASGTransform trans, float axis[3], float angleInRadians)
{
    asg::Mat3f rot = asg::makeRotation({axis[0],axis[1],axis[2]},angleInRadians);
    asg::Mat3f current;
    std::memcpy((void*)&current,((Transform*)trans->impl)->matrix,9*sizeof(float));
    asg::Mat3f newMatrix = current * rot;
    std::memcpy(((Transform*)trans->impl)->matrix,(const void*)&newMatrix,
                9*sizeof(float));
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgTransformTranslate(ASGTransform trans, float xyz[3])
{
    ((Transform*)trans->impl)->matrix[ 9] += xyz[0];
    ((Transform*)trans->impl)->matrix[10] += xyz[1];
    ((Transform*)trans->impl)->matrix[11] += xyz[2];
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// RGBA luts
// ========================================================

struct LUT1D {
    float* rgb;
    float* alpha;
    int32_t numEntries;
    float* rgbUserPtr;
    float* alphaUserPtr;
    ASGFreeFunc freeRGB;
    ASGFreeFunc freeAlpha;
};

ASGLookupTable1D asgNewLookupTable1D(float* rgb, float* alpha, int32_t numEntries,
                                     ASGFreeFunc freeRGB, ASGFreeFunc freeAlpha)
{
    ASGLookupTable1D lut = (ASGLookupTable1D)asgNewObject();
    lut->type = ASG_TYPE_LOOKUP_TABLE1D;
    lut->impl = new LUT1D;
    ((LUT1D*)lut->impl)->rgb = rgb;
    ((LUT1D*)lut->impl)->alpha = alpha;
    ((LUT1D*)lut->impl)->numEntries = numEntries;
    ((LUT1D*)lut->impl)->rgbUserPtr = rgb;
    ((LUT1D*)lut->impl)->alphaUserPtr = alpha;
    ((LUT1D*)lut->impl)->freeRGB = freeRGB;
    ((LUT1D*)lut->impl)->freeAlpha = freeAlpha;
    return lut;
}

ASGError_t asgLookupTable1DGetRGB(ASGLookupTable1D lut, float** rgb)
{
    *rgb = ((LUT1D*)lut->impl)->rgb;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgLookupTable1DGetAlpha(ASGLookupTable1D lut, float** alpha)
{
    *alpha = ((LUT1D*)lut->impl)->alpha;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgLookupTable1DGetNumEntries(ASGLookupTable1D lut, int32_t* numEntries)
{
    *numEntries = ((LUT1D*)lut->impl)->numEntries;
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Volumes
// ========================================================

struct StructuredVolume {
    void* data;
    int32_t width, height, depth;
    ASGDataType_t type;
    float rangeMin, rangeMax;
    float distX, distY, distZ;
    ASGLookupTable1D lut1D;
    void* userPtr;
    ASGFreeFunc freeData;
    // Exclusively used by ANARI build visitors
    ANARIVolume anariVolume = NULL;
    ANARISpatialField anariSpatialField = NULL;
};

ASGStructuredVolume asgNewStructuredVolume(void* data, int32_t width, int32_t height,
                                          int32_t depth, ASGDataType_t type,
                                          ASGFreeFunc freeData)
{
    ASGStructuredVolume vol = (ASGStructuredVolume)asgNewObject();
    vol->type = ASG_TYPE_STRUCTURED_VOLUME;
    vol->impl = new StructuredVolume;
    ((StructuredVolume*)vol->impl)->data = data;
    ((StructuredVolume*)vol->impl)->width = width;
    ((StructuredVolume*)vol->impl)->height = height;
    ((StructuredVolume*)vol->impl)->depth = depth;
    ((StructuredVolume*)vol->impl)->type = type;
    ((StructuredVolume*)vol->impl)->rangeMin = 0.f;
    ((StructuredVolume*)vol->impl)->rangeMax = 1.f;
    ((StructuredVolume*)vol->impl)->distX = 1.f;
    ((StructuredVolume*)vol->impl)->distY = 1.f;
    ((StructuredVolume*)vol->impl)->distY = 1.f;
    ((StructuredVolume*)vol->impl)->lut1D = NULL;
    ((StructuredVolume*)vol->impl)->userPtr = data;
    ((StructuredVolume*)vol->impl)->freeData = freeData;
    return vol;
}

ASGError_t asgStructuredVolumeGetData(ASGStructuredVolume vol, void** data)
{
    *data = ((StructuredVolume*)vol->impl)->data;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeGetDims(ASGStructuredVolume vol, int32_t* width,
                                      int32_t* height, int32_t* depth)
{
    *width = ((StructuredVolume*)vol->impl)->width;
    *height = ((StructuredVolume*)vol->impl)->height;
    *depth = ((StructuredVolume*)vol->impl)->depth;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeGetType(ASGStructuredVolume vol, ASGType_t* type)
{
    *type = ((StructuredVolume*)vol->impl)->type;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeSetRange(ASGStructuredVolume vol, float rangeMin,
                                       float rangeMax)
{
    ((StructuredVolume*)vol->impl)->rangeMin = rangeMin;
    ((StructuredVolume*)vol->impl)->rangeMax = rangeMax;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeGetRange(ASGStructuredVolume vol, float* rangeMin,
                                       float* rangeMax)
{
    *rangeMin = ((StructuredVolume*)vol->impl)->rangeMin;
    *rangeMax = ((StructuredVolume*)vol->impl)->rangeMax;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeSetDist(ASGStructuredVolume vol, float distX, float distY,
                                      float distZ)
{
    ((StructuredVolume*)vol->impl)->distX = distX;
    ((StructuredVolume*)vol->impl)->distY = distY;
    ((StructuredVolume*)vol->impl)->distZ = distZ;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeGetDist(ASGStructuredVolume vol, float* distX, float* distY,
                                      float* distZ)
{
    *distX = ((StructuredVolume*)vol->impl)->distX;
    *distY = ((StructuredVolume*)vol->impl)->distY;
    *distZ = ((StructuredVolume*)vol->impl)->distZ;
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeSetLookupTable1D(ASGStructuredVolume vol,
                                               ASGLookupTable1D lut)
{
    StructuredVolume* impl = (StructuredVolume*)vol->impl;
    if (impl->lut1D != NULL)
        asgRelease(impl->lut1D);

    impl->lut1D = lut;
    asgRetain(lut);
    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgStructuredVolumeGetLookupTable1D(ASGStructuredVolume vol,
                                               ASGLookupTable1D* lut)
{
    *lut = ((StructuredVolume*)vol->impl)->lut1D;
    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// I/O
// ========================================================

#if ASG_HAVE_ASSIMP
namespace assimp {

    typedef uint64_t VisitFlags;

    VisitFlags FLAG_GEOMETRY         = 1;
    VisitFlags FLAG_ACCUM_TRANSFORMS = 2;

    void Visit(aiNode* node, const aiScene* scene, ASGObject obj,
               const std::vector<ASGMaterial>& materials,
               const std::vector<ASGLight>& lights,
               aiMatrix4x4 accTransform, VisitFlags flags)
    {
        aiMatrix4x4 transform;

        if (node->mNumMeshes > 0) {
            if (flags & FLAG_GEOMETRY) {
                for (unsigned i=0; i<node->mNumMeshes; ++i) {
                    unsigned meshID = node->mMeshes[i];
                    aiMesh* mesh = scene->mMeshes[meshID];

                    float* vertices = (float*)malloc(mesh->mNumVertices*3*sizeof(float));

                    float* vertexNormals = NULL;
                    if (mesh->HasNormals())
                        vertexNormals
                                = (float*)malloc(mesh->mNumVertices*3*sizeof(float));

                    float* vertexColors = NULL;
                    if (mesh->mColors[0] != NULL) // TODO:(?) AI_MAX_NUMBER_OF_COLOR_SETS
                        vertexColors
                                = (float*)malloc(mesh->mNumVertices*4*sizeof(float));

                    for (unsigned j=0; j<mesh->mNumVertices; ++j) {
                        aiVector3D v = mesh->mVertices[j];

                        if (flags & FLAG_ACCUM_TRANSFORMS) {
                            v *= accTransform;
                        }

                        vertices[j*3] = v.x;
                        vertices[j*3+1] = v.y;
                        vertices[j*3+2] = v.z;

                        if (vertexNormals != NULL) {
                            aiVector3D vn = mesh->mNormals[j];

                            // TODO: accum. inverse-transpose, too
                            if (flags & FLAG_ACCUM_TRANSFORMS) {
                                //vn *= accTransform;
                            }

                            vertexNormals[j*3] = vn.x;
                            vertexNormals[j*3+1] = vn.y;
                            vertexNormals[j*3+2] = vn.z;
                        }

                        if (vertexColors != NULL) {
                            aiColor4D vc = mesh->mColors[0][j];
                            vertexColors[j*4] = vc.r;
                            vertexColors[j*4+1] = vc.g;
                            vertexColors[j*4+2] = vc.b;
                            vertexColors[j*4+3] = vc.a;
                        }
                    }

                    uint32_t* indices = NULL;
                    uint32_t numIndices = 0;
                    if (mesh->HasFaces()) {
                        numIndices = mesh->mNumFaces;
                        // The following is valid as we're importing with triangulation
                        indices = (uint32_t*)malloc(numIndices*3*sizeof(uint32_t));
                        for (unsigned j=0; j<mesh->mNumFaces; ++j) {
                            indices[j*3] = mesh->mFaces[j].mIndices[0];
                            indices[j*3+1] = mesh->mFaces[j].mIndices[1];
                            indices[j*3+2] = mesh->mFaces[j].mIndices[2];
                        }
                    }

                    ASGMaterial mat = nullptr;

                    unsigned materialID = mesh->mMaterialIndex;

                    if (materialID < materials.size()) { // TODO: error handling
                        mat = materials[materialID];
                    }

                    ASGTriangleGeometry geom = asgNewTriangleGeometry(vertices,
                                                                      vertexNormals,
                                                                      vertexColors,
                                                                      mesh->mNumVertices,
                                                                      indices,
                                                                      numIndices,
                                                                      free,free,
                                                                      free,free);

                    ASGSurface surf = asgNewSurface(geom,mat);

                    asgObjectAddChild(obj,surf);
                }
                transform = aiMatrix4x4(); // identity
            }
        }

        if (flags & FLAG_ACCUM_TRANSFORMS) {
            transform = node->mTransformation * accTransform;
        }

        // Light sources have an associated node with the same name as the
        // light source (!), whose transform is applied to the light's position
        auto it = std::find_if(lights.begin(),lights.end(),
                               [&node](const ASGLight& light) {
                                   const char* name;
                                   asgObjectGetName(light,&name);
                                   return std::string(node->mName.C_Str())
                                                == std::string(name);
                               });

        if (it != lights.end()) {
            if (flags & FLAG_ACCUM_TRANSFORMS) {// flatten
                float position[3] {0.f,0.f,0.f};
                ASGParam positionParam;
                asgLightGetParam(*it,"position",&positionParam);

                aiVector3D v = {position[0],position[1],position[2]};
                v *= transform;
                position[0] = v.x;
                position[1] = v.y;
                position[2] = v.z;

                // TODO: this only works if there's *one* instance of this light!
                asgLightSetParam(*it,asgParam3fv("position",position));
                asgObjectAddChild(obj,*it);
            }
        }

        for (unsigned i=0; i<node->mNumChildren; ++i) {
            if (flags & FLAG_ACCUM_TRANSFORMS) // flatten
                Visit(node->mChildren[i],scene,obj,materials,lights,transform,flags);
        }
    }
} // ::assimp
#endif

ASGError_t asgLoadASSIMP(ASGObject obj, const char* fileName, uint64_t flags)
{
#if ASG_HAVE_ASSIMP
    Assimp::DefaultLogger::create("",Assimp::Logger::VERBOSE);

    static Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(fileName,
                                             aiProcess_Triangulate |
                                             //aiProcess_JoinIdenticalVertices  |
                                             aiProcess_SortByPType);

    if (scene == nullptr) {
        Assimp::DefaultLogger::get()->error(importer.GetErrorString());
        return ASG_ERROR_FILE_IO_ERROR;
    }

    std::vector<ASGMaterial> materials;
    for (unsigned i=0; i<scene->mNumMaterials; ++i) {
        ASGMaterial mat = asgNewMaterial("");

        aiMaterial* assimpMAT = scene->mMaterials[i];
        aiColor3D col;
        assimpMAT->Get(AI_MATKEY_COLOR_DIFFUSE,col);
        aiString name;
        assimpMAT->Get(AI_MATKEY_NAME,name);

        float kd[3] = {col.r,col.g,col.b};

        asgMakeMatte(&mat,kd,NULL);
        asgObjectSetName(mat,name.C_Str());
        materials.push_back(mat);
    }

    std::vector<ASGLight> lights;
    for (unsigned i=0; i<scene->mNumLights; ++i) {
        aiLight* assimpLight = scene->mLights[i];

        switch (assimpLight->mType) {
            case aiLightSource_POINT: {
                ASGLight light = asgNewLight("");

                float pos[3] = {assimpLight->mPosition.x,
                                assimpLight->mPosition.y,
                                assimpLight->mPosition.z};
                float color[3] = {assimpLight->mColorDiffuse.r,
                                  assimpLight->mColorDiffuse.g,
                                  assimpLight->mColorDiffuse.b};

                aiString name = assimpLight->mName;

                asgMakePointLight(&light,pos,color);
                asgObjectSetName(light,name.C_Str());

                lights.push_back(light);

                break;
            }

            default: {
                break;
            }
        }
    }

    assimp::VisitFlags vflags = assimp::FLAG_GEOMETRY | assimp::FLAG_ACCUM_TRANSFORMS;
    assimp::Visit(scene->mRootNode,scene,obj,materials,lights,aiMatrix4x4(),vflags);

    return ASG_ERROR_NO_ERROR;
#else
    return ASG_ERROR_MISSING_FILE_HANDLER;
#endif
}

#if ASG_HAVE_PBRT_PARSER
namespace pbrt {

    void Visit(Object::SP pbrtOBJ, ASGObject obj,
               std::unordered_map<pbrt::Shape::SP,ASGGeometry>& shape2geom,
               std::unordered_map<pbrt::Material::SP,ASGMaterial>& mat2mat)
    {
        for (auto shape : pbrtOBJ->shapes) {

            ASGSurface surf = NULL;
            ASGGeometry geom = NULL;
            ASGMaterial mat = NULL;

            auto it = shape2geom.find(shape);

            if (it != shape2geom.end())
                geom = it->second;
            else {
                if (auto sphere = std::dynamic_pointer_cast<Sphere>(shape)) {

                } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(shape)) {

                    float* vertex
                        = (float*)malloc(mesh->vertex.size()*3*sizeof(float));
                    memcpy(vertex,mesh->vertex.data(),mesh->vertex.size()*3*sizeof(float));

                    uint32_t* index
                        = (uint32_t*)malloc(mesh->index.size()*3*sizeof(uint32_t));

                    // int to uint
                    for (size_t i=0; i<mesh->index.size(); ++i) {
                        index[i*3]   = (uint32_t)mesh->index[i].x;
                        index[i*3+1] = (uint32_t)mesh->index[i].y;
                        index[i*3+2] = (uint32_t)mesh->index[i].z;
                    }

                    geom = asgNewTriangleGeometry(vertex,NULL,NULL,mesh->vertex.size(),
                                                  index,mesh->index.size(),free,free,
                                                  free,free);

                    shape2geom.insert({shape,geom});
                }
            }

            auto mit = mat2mat.find(shape->material);

            if (mit != mat2mat.end())
                mat = mit->second;
            else {
                if (auto m = std::dynamic_pointer_cast<MatteMaterial>(shape->material)) {
                    mat = asgNewMaterial("");
                    float kd[3] = {m->kd.x,m->kd.y,m->kd.z};
                    asgMakeMatte(&mat,kd);
                    asgObjectSetName(mat,m->name.c_str());
                }
            }

            if (geom != nullptr) {
                surf = asgNewSurface(geom,mat);
                asgObjectAddChild(obj,surf);
            }
        }

        for (auto inst : pbrtOBJ->instances) {
            float mat[12] = {
                inst->xfm.l.vx.x,inst->xfm.l.vx.y,inst->xfm.l.vx.z,
                inst->xfm.l.vy.x,inst->xfm.l.vy.y,inst->xfm.l.vy.z,
                inst->xfm.l.vz.x,inst->xfm.l.vz.y,inst->xfm.l.vz.z,
                inst->xfm.p.x,inst->xfm.p.y,inst->xfm.p.z
            };

            ASGTransform trans = asgNewTransform(mat);

            Visit(inst->object,trans,shape2geom,mat2mat);

            asgObjectAddChild(obj,trans);
        }

        for (auto ls : pbrtOBJ->lightSources) {

        }
    }
} // ::pbrt
#endif

ASGError_t asgLoadPBRT(ASGObject obj, const char* fileName, uint64_t flags)
{
#if ASG_HAVE_PBRT_PARSER
    using namespace pbrt;

    Scene::SP scene;

    // get file extension
    std::string fn(fileName);
    int pos = fn.rfind('.');
    if (pos == fn.npos)
        return ASG_ERROR_FILE_IO_ERROR;
    std::string ext = fn.substr(pos);

    try {
        if (ext==".pbf")
            scene = Scene::loadFrom(fileName);
        else if (ext==".pbrt")
            scene = importPBRT(fileName);
        else
            return ASG_ERROR_FILE_IO_ERROR;
    } catch (std::runtime_error& e) {
        return ASG_ERROR_FILE_IO_ERROR;
    }

    // Keep track of the shapes we found and the geometries we
    // already created
    std::unordered_map<pbrt::Shape::SP,ASGGeometry> shape2geom;
    std::unordered_map<pbrt::Material::SP,ASGMaterial> mat2mat;
    Visit(scene->world,obj,shape2geom,mat2mat);

    return ASG_ERROR_NO_ERROR;
#else
    return ASG_ERROR_MISSING_FILE_HANDLER;
#endif
}

ASGError_t asgLoadVOLKIT(ASGStructuredVolume vol, const char* fileName, uint64_t flags)
{
#if ASG_HAVE_VOLKIT
    vkt::VolumeFile file(fileName, vkt::OpenMode::Read);

    vkt::VolumeFileHeader hdr = file.getHeader();

    if (!hdr.isStructured)
        return ASG_ERROR_FILE_IO_ERROR;

    vkt::Vec3i dims = hdr.dims;
    if (dims.x * dims.y * dims.z < 1)
        return ASG_ERROR_FILE_IO_ERROR;

    vkt::DataFormat dataFormat = hdr.dataFormat;
    if (dataFormat == vkt::DataFormat::Unspecified)
    {
        // just guessing here..
        dataFormat = vkt::DataFormat::UInt8;
    }

    vkt::StructuredVolume inputVolume(dims.x, dims.y, dims.z, dataFormat);
    vkt::InputStream is(file);
    is.read(inputVolume);

    StructuredVolume* impl = (StructuredVolume*)vol->impl;

    vkt::StructuredVolume volkitVolume = inputVolume;
    if (flags & ASG_IO_FLAG_RESAMPLE_VOXEL_TYPE) {
        vkt::DataFormat destDataFormat = vkt::DataFormat::UInt8;
        if (impl->type == ASG_DATA_TYPE_UINT8)
            destDataFormat = vkt::DataFormat::UInt16;
        else if (impl->type == ASG_DATA_TYPE_UINT16)
            destDataFormat = vkt::DataFormat::UInt16;
        else if (impl->type == ASG_DATA_TYPE_FLOAT32)
            destDataFormat = vkt::DataFormat::Float32;
        else
            return ASG_ERROR_FILE_IO_ERROR;

        if (destDataFormat != dataFormat) {
            volkitVolume = vkt::StructuredVolume(dims.x, dims.y, dims.z, destDataFormat);
            vkt::Resample(volkitVolume,inputVolume,vkt::FilterMode::Nearest);
            hdr.dataFormat = dataFormat = destDataFormat;
        }
    }
    // TODO: same for volume dims

    size_t bpv(-1);
    if (dataFormat == vkt::DataFormat::UInt8)
        bpv = 1;
    else if (dataFormat == vkt::DataFormat::UInt16)
        bpv = 2;
    else if (dataFormat == vkt::DataFormat::Float32)
        bpv = 4;
    else
        return ASG_ERROR_FILE_IO_ERROR;

    asgRelease(vol);

    impl->data = malloc(dims.x * dims.y * dims.z * bpv);
    std::memcpy(impl->data,volkitVolume.getData(),dims.x * dims.y * dims.z * bpv);
    impl->width = dims.x;
    impl->height = dims.y;
    impl->depth = dims.z;
    impl->type = bpv==1 ? ASG_DATA_TYPE_UINT8
                        : bpv==2 ? ASG_DATA_TYPE_UINT16 : ASG_DATA_TYPE_FLOAT32;
    impl->rangeMin = volkitVolume.getVoxelMapping().x;
    impl->rangeMax = volkitVolume.getVoxelMapping().y;
    impl->distX = volkitVolume.getDist().x;
    impl->distY = volkitVolume.getDist().x;
    impl->distZ = volkitVolume.getDist().z;
    impl->lut1D = NULL; // TODO: file might actually have a lut
    impl->userPtr = impl->data;
    impl->freeData = free;
    return ASG_ERROR_NO_ERROR;
#else
    return ASG_ERROR_MISSING_FILE_HANDLER;
#endif
}

// ========================================================
// Procedural
// ========================================================

ASGError_t asgMakeMarschnerLobb(ASGStructuredVolume vol)
{
    StructuredVolume* impl = (StructuredVolume*)vol->impl;
    int32_t volDims[3] = {
        impl->width,
        impl->height,
        impl->depth,
    };

    float fM = 6.f, a = .25f;
    auto rho = [=](float r) {
        return cosf(2.f*M_PI*fM*cosf(M_PI*r/2.f));
    };
    for (int z=0; z<volDims[2]; ++z) {
        for (int y=0; y<volDims[1]; ++y) {
            for (int x=0; x<volDims[0]; ++x) {

                float xx = (float)x/volDims[0]*2.f-1.f;
                float yy = (float)y/volDims[1]*2.f-1.f;
                float zz = (float)z/volDims[2]*2.f-1.f;
                unsigned linearIndex = volDims[0]*volDims[1]*z
                    + volDims[0]*y
                    + x;

                float val = (1.f-sinf(M_PI*zz/2.f) + a * (1+rho(sqrtf(xx*xx+yy*yy))))
                                            / (2.f*(1.f+a));

                assert(impl->type==ASG_DATA_TYPE_FLOAT32);
                ((float*)impl->data)[linearIndex] = val;
            }
        }
    }

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgMakeDefaultLUT1D(ASGLookupTable1D lut, ASGLutID lutID)
{
    switch (lutID)
    {
        case ASG_LUT_ID_DEFAULT_LUT: {
            int numEntries = ((LUT1D*)lut->impl)->numEntries;
            assert(numEntries == 5);

            float rgba[] = {
                    1.f, 1.f, 1.f, .005f,
                    0.f, .1f, .1f, .25f,
                    .5f, .5f, .7f, .5f,
                    .7f, .7f, .07f, .75f,
                    1.f, .3f, .3f, 1.f
                    };
            for (int i=0; i<numEntries; ++i) {
                ((LUT1D*)lut->impl)->rgb[i*3] = rgba[i*4];
                ((LUT1D*)lut->impl)->rgb[i*3+1] = rgba[i*4+1];
                ((LUT1D*)lut->impl)->rgb[i*3+2] = rgba[i*4+2];
                ((LUT1D*)lut->impl)->alpha[i] = rgba[i*4+3];
            }
            break;
        }

        default: return ASG_ERROR_INVALID_LUT_ID;
    }

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgMakeMatte(ASGMaterial* material, float kd[3], ASGSampler2D mapKD)
{
    asgRelease(*material);
    *material = asgNewMaterial("matte");

    asgMaterialSetParam(*material,asgParam3fv("kd",kd));
    asgMaterialSetParam(*material,asgParamSampler2D("mapKD",mapKD));

    return ASG_ERROR_NO_ERROR;
}

ASGError_t asgMakePointLight(ASGLight* light, float position[3], float color[3],
                             float intensity)
{
    asgRelease(*light);
    *light = asgNewLight("point");

    asgLightSetParam(*light,asgParam3fv("position",position));
    asgLightSetParam(*light,asgParam3fv("color",color));
    asgLightSetParam(*light,asgParam1f("intensity",intensity));

    return ASG_ERROR_NO_ERROR;
}

// ========================================================
// Builtin visitors
// ========================================================

// Compute bounds

struct Bounds {
    asg::Vec3f minVal, maxVal;
    std::vector<asg::Mat4x3f> transStack;
};

static void visitBounds(ASGVisitor self, ASGObject obj, void* userData) {

    ASGType_t t;
    asgGetType(obj,&t);

    Bounds* bounds = (Bounds*)userData;

    switch (t)
    {
        case ASG_TYPE_TRANSFORM: {
            Transform* trans = (Transform*)obj->impl;

            asg::Mat4x3f mat4x3;
            std::memcpy(&mat4x3,trans->matrix,sizeof(mat4x3));
            bounds->transStack.push_back(mat4x3);

            asgVisitorApply(self,obj);

            bounds->transStack.pop_back();

            break;
        }

        case ASG_TYPE_SURFACE: {
            Surface* surf = (Surface*)obj->impl;
            assert(surf->geometry != nullptr);

            if (surf->geometry->type == ASG_TYPE_TRIANGLE_GEOMETRY) {

                TriangleGeom* geom = (TriangleGeom*)surf->geometry->impl;

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    for (unsigned i=0; i<geom->numIndices; ++i) {
                        for (unsigned j=0; j<3; ++j) {
                            unsigned index = i * 3 + j;
                            asg::Vec3f v = {
                                geom->vertices[geom->indices[index]*3],
                                geom->vertices[geom->indices[index]*3+1],
                                geom->vertices[geom->indices[index]*3+2]
                            };

                            for (asg::Mat4x3f trans : bounds->transStack) {
                                v = trans * asg::Vec4f{v.x,v.y,v.z,1.f};
                            }

                            bounds->minVal = min(bounds->minVal,v);
                            bounds->maxVal = max(bounds->maxVal,v);
                        }
                    }
                } else {
                    // No indices, so just insert the verts directly
                    for (unsigned i=0; i<geom->numVertices; ++i) {
                        asg::Vec3f v = {
                            geom->vertices[i*3],
                            geom->vertices[i*3+1],
                            geom->vertices[i*3+2]
                        };

                        for (asg::Mat4x3f trans : bounds->transStack) {
                            v = trans * asg::Vec4f{v.x,v.y,v.z,1.f};
                        }

                        bounds->minVal = min(bounds->minVal,v);
                        bounds->maxVal = max(bounds->maxVal,v);
                    }
                }
            } else if (surf->geometry->type == ASG_TYPE_SPHERE_GEOMETRY) {

                SphereGeom* geom = (SphereGeom*)surf->geometry->impl;

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    for (unsigned i=0; i<geom->numIndices; ++i) {
                        asg::Vec3f v = {
                            geom->vertices[geom->indices[i]*3],
                            geom->vertices[geom->indices[i]*3+1],
                            geom->vertices[geom->indices[i]*3+2]
                        };

                        asg::Vec3f v1 = v - geom->radii[geom->indices[i]];
                        asg::Vec3f v2 = v + geom->radii[geom->indices[i]];

                        for (asg::Mat4x3f trans : bounds->transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds->minVal = min(bounds->minVal,v1);
                        bounds->maxVal = max(bounds->maxVal,v1);

                        bounds->minVal = min(bounds->minVal,v2);
                        bounds->maxVal = max(bounds->maxVal,v2);
                    }
                } else {
                    // No indices, so just insert the verts directly
                    for (unsigned i=0; i<geom->numVertices; ++i) {
                        asg::Vec3f v = {
                            geom->vertices[i*3],
                            geom->vertices[i*3+1],
                            geom->vertices[i*3+2]
                        };

                        asg::Vec3f v1 = v - geom->radii[i];
                        asg::Vec3f v2 = v + geom->radii[i];

                        for (asg::Mat4x3f trans : bounds->transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds->minVal = min(bounds->minVal,v1);
                        bounds->maxVal = max(bounds->maxVal,v1);

                        bounds->minVal = min(bounds->minVal,v2);
                        bounds->maxVal = max(bounds->maxVal,v2);
                    }
                }
            } else if (surf->geometry->type == ASG_TYPE_CYLINDER_GEOMETRY) {

                CylinderGeom* geom = (CylinderGeom*)surf->geometry->impl;

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    for (unsigned i=0; i<geom->numIndices; ++i) {

                        asg::Vec3f v1 = {
                            geom->vertices[geom->indices[i*2]*3],
                            geom->vertices[geom->indices[i*2]*3+1],
                            geom->vertices[geom->indices[i*2]*3+2]
                        };

                        asg::Vec3f v2 = {
                            geom->vertices[geom->indices[i*2+1]*3],
                            geom->vertices[geom->indices[i*2+1]*3+1],
                            geom->vertices[geom->indices[i*2+1]*3+2]
                        };

                        float radius = geom->radii[i];

                        asg::Vec3f a = v2 - v1;
                        float aa = dot(a,a);
                        asg::Vec3f e = {
                            radius * sqrtf(1.f - a.x * a.x / aa),
                            radius * sqrtf(1.f - a.y * a.y / aa),
                            radius * sqrtf(1.f - a.z * a.z / aa)
                        };

                        asg::Vec3f pa = {
                            fminf(v1.x-e.x,v2.x-e.x),
                            fminf(v1.y-e.y,v2.y-e.y),
                            fminf(v1.z-e.z,v2.z-e.z)
                        };

                        asg::Vec3f pb = {
                            fmaxf(v1.x+e.x,v2.x+e.x),
                            fmaxf(v1.y+e.y,v2.y+e.y),
                            fmaxf(v1.z+e.z,v2.z+e.z)
                        };

                        v1 = pa;
                        v2 = pb;

                        for (asg::Mat4x3f trans : bounds->transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds->minVal = min(bounds->minVal,v1);
                        bounds->maxVal = max(bounds->maxVal,v1);

                        bounds->minVal = min(bounds->minVal,v2);
                        bounds->maxVal = max(bounds->maxVal,v2);
                    }
                } else {
                    // No indices, so just insert the verts directly
                    for (unsigned i=0; i<geom->numVertices; ++i) {

                        asg::Vec3f v1 = {
                            geom->vertices[i*6],
                            geom->vertices[i*6+1],
                            geom->vertices[i*6+2]
                        };

                        asg::Vec3f v2 = {
                            geom->vertices[i*6+3],
                            geom->vertices[i*6+4],
                            geom->vertices[i*6+5]
                        };

                        float radius = geom->radii[i];

                        asg::Vec3f a = v2 - v1;
                        float aa = dot(a,a);
                        asg::Vec3f e = {
                            radius * sqrtf(1.f - a.x * a.x / aa),
                            radius * sqrtf(1.f - a.y * a.y / aa),
                            radius * sqrtf(1.f - a.z * a.z / aa)
                        };

                        asg::Vec3f pa = {
                            fminf(v1.x-e.x,v2.x-e.x),
                            fminf(v1.y-e.y,v2.y-e.y),
                            fminf(v1.z-e.z,v2.z-e.z)
                        };

                        asg::Vec3f pb = {
                            fmaxf(v1.x+e.x,v2.x+e.x),
                            fmaxf(v1.y+e.y,v2.y+e.y),
                            fmaxf(v1.z+e.z,v2.z+e.z)
                        };

                        v1 = pa;
                        v2 = pb;

                        for (asg::Mat4x3f trans : bounds->transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds->minVal = min(bounds->minVal,v1);
                        bounds->maxVal = max(bounds->maxVal,v1);

                        bounds->minVal = min(bounds->minVal,v2);
                        bounds->maxVal = max(bounds->maxVal,v2);
                    }
                }
            }

            break;
        }

        case ASG_TYPE_STRUCTURED_VOLUME: {
            // TODO!
            break;
        }
        case ASG_TYPE_OBJECT:
            // fall-through
        default: {
            asgVisitorApply(self,obj);
            break;
        }
    }
}

ASGError_t asgComputeBounds(ASGObject obj, float* minX, float* minY, float* minZ,
                            float* maxX, float* maxY, float* maxZ, uint64_t nodeMask)
{
    Bounds bounds;
    bounds.minVal = {+FLT_MAX,+FLT_MAX,+FLT_MAX};
    bounds.maxVal = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
    ASGVisitor visitor = asgCreateVisitor(visitBounds,&bounds,
                                          ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN);
    asgObjectAccept(obj,visitor);
    asgDestroyVisitor(visitor);

    *minX = bounds.minVal.x; *minY = bounds.minVal.y; *minZ = bounds.minVal.z;
    *maxX = bounds.maxVal.x; *maxY = bounds.maxVal.y; *maxZ = bounds.maxVal.z;

    return ASG_ERROR_NO_ERROR;
}


// Pick object

struct PickRecord {
    ASGObject handle;
    asg::Vec3f rayOri, rayDir;
    float maxT;
    std::vector<asg::Mat4x3f> transStack;
};

static void pickObject(ASGVisitor self, ASGObject obj, void* userData) {

    ASGType_t t;
    asgGetType(obj,&t);

    PickRecord* pr = (PickRecord*)userData;

    switch (t)
    {
        case ASG_TYPE_TRANSFORM: {
            Transform* trans = (Transform*)obj->impl;

            asg::Mat4x3f mat4x3;
            std::memcpy(&mat4x3,trans->matrix,sizeof(mat4x3));
            pr->transStack.push_back(mat4x3);

            asgVisitorApply(self,obj);

            pr->transStack.pop_back();

            break;
        }

        case ASG_TYPE_SURFACE: {
            Surface* surf = (Surface*)obj->impl;
            assert(surf->geometry != nullptr);

            Bounds bounds;
            bounds.minVal = {+FLT_MAX,+FLT_MAX,+FLT_MAX};
            bounds.maxVal = {-FLT_MAX,-FLT_MAX,-FLT_MAX};

            if (surf->geometry->type == ASG_TYPE_TRIANGLE_GEOMETRY) {

                TriangleGeom* geom = (TriangleGeom*)surf->geometry->impl;

                if (geom->indices != nullptr && geom->numIndices > 0) {
#if ASG_USE_VISIONARAY_FOR_PICKING
                    using namespace visionaray;
                    if (geom->bvh.num_nodes() == 0) {
                        std::vector<basic_triangle<3,float>> tris;

                        for (unsigned i=0; i<geom->numIndices; ++i) {
                            vec3 v1 = {
                                geom->vertices[geom->indices[i*3]*3],
                                geom->vertices[geom->indices[i*3]*3+1],
                                geom->vertices[geom->indices[i*3]*3+2]
                            };
                            vec3 v2 = {
                                geom->vertices[geom->indices[i*3+1]*3],
                                geom->vertices[geom->indices[i*3+1]*3+1],
                                geom->vertices[geom->indices[i*3+1]*3+2]
                            };
                            vec3 v3 = {
                                geom->vertices[geom->indices[i*3+2]*3],
                                geom->vertices[geom->indices[i*3+2]*3+1],
                                geom->vertices[geom->indices[i*3+2]*3+2]
                            };

                            basic_triangle<3,float> tri;
                            tri.v1 = v1; tri.e1 = v2-v1; tri.e2 = v3-v1;
                            tri.prim_id = i;
                            tris.push_back(tri);
                        }

                        lbvh_builder builder;
                        geom->bvh = builder.build(index_bvh<basic_triangle<3,float>>{},
                                                  tris.data(),tris.size());
                    }

                    mat3x3 m3 = mat3x3::identity();
                    vec3 t(0.f);
                    for (asg::Mat4x3f trans : pr->transStack) {
                        mat3x3 m33(*((vec3*)&trans.col0),*((vec3*)&trans.col1),*((vec3*)&trans.col2));
                        vec3 tt(*(vec3*)&trans.col3);
                        m3 = m33 * m3;
                        t += tt;
                    }
                    auto inst = geom->bvh.inst({m3,t});
                    basic_ray<float> r;
                    r.ori = vec3f(pr->rayOri.x,pr->rayOri.y,pr->rayOri.z);
                    r.dir = vec3f(pr->rayDir.x,pr->rayDir.y,pr->rayDir.z);
                    r.tmin = 0.f;
                    r.tmax = pr->maxT;
                    auto hr = closest_hit(r,&inst,&inst+1);
                    if (hr.t < pr->maxT) {
                        pr->handle = obj;
                        pr->maxT = hr.t;
                    }
#else
                    for (unsigned i=0; i<geom->numIndices; ++i) {
                        for (unsigned j=0; j<3; ++j) {
                            unsigned index = i * 3 + j;
                            asg::Vec3f v = {
                                geom->vertices[geom->indices[index]*3],
                                geom->vertices[geom->indices[index]*3+1],
                                geom->vertices[geom->indices[index]*3+2]
                            };

                            for (asg::Mat4x3f trans : pr->transStack) {
                                v = trans * asg::Vec4f{v.x,v.y,v.z,1.f};
                            }

                            bounds.minVal = min(bounds.minVal,v);
                            bounds.maxVal = max(bounds.maxVal,v);
                        }
                    }
#endif
                } else {
                    // No indices, so just insert the verts directly
                    for (unsigned i=0; i<geom->numVertices; ++i) {
                        asg::Vec3f v = {
                            geom->vertices[i*3],
                            geom->vertices[i*3+1],
                            geom->vertices[i*3+2]
                        };

                        for (asg::Mat4x3f trans : pr->transStack) {
                            v = trans * asg::Vec4f{v.x,v.y,v.z,1.f};
                        }

                        bounds.minVal = min(bounds.minVal,v);
                        bounds.maxVal = max(bounds.maxVal,v);
                    }
                }
            } else if (surf->geometry->type == ASG_TYPE_SPHERE_GEOMETRY) {

                SphereGeom* geom = (SphereGeom*)surf->geometry->impl;

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    for (unsigned i=0; i<geom->numIndices; ++i) {
                        asg::Vec3f v = {
                            geom->vertices[geom->indices[i]*3],
                            geom->vertices[geom->indices[i]*3+1],
                            geom->vertices[geom->indices[i]*3+2]
                        };

                        asg::Vec3f v1 = v - geom->radii[geom->indices[i]];
                        asg::Vec3f v2 = v + geom->radii[geom->indices[i]];

                        for (asg::Mat4x3f trans : bounds.transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds.minVal = min(bounds.minVal,v1);
                        bounds.maxVal = max(bounds.maxVal,v1);

                        bounds.minVal = min(bounds.minVal,v2);
                        bounds.maxVal = max(bounds.maxVal,v2);
                    }
                } else {
                    // No indices, so just insert the verts directly
                    for (unsigned i=0; i<geom->numVertices; ++i) {
                        asg::Vec3f v = {
                            geom->vertices[i*3],
                            geom->vertices[i*3+1],
                            geom->vertices[i*3+2]
                        };

                        asg::Vec3f v1 = v - geom->radii[i];
                        asg::Vec3f v2 = v + geom->radii[i];

                        for (asg::Mat4x3f trans : bounds.transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds.minVal = min(bounds.minVal,v1);
                        bounds.maxVal = max(bounds.maxVal,v1);

                        bounds.minVal = min(bounds.minVal,v2);
                        bounds.maxVal = max(bounds.maxVal,v2);
                    }
                }
            } else if (surf->geometry->type == ASG_TYPE_CYLINDER_GEOMETRY) {

                CylinderGeom* geom = (CylinderGeom*)surf->geometry->impl;

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    for (unsigned i=0; i<geom->numIndices; ++i) {

                        asg::Vec3f v1 = {
                            geom->vertices[geom->indices[i*2]*3],
                            geom->vertices[geom->indices[i*2]*3+1],
                            geom->vertices[geom->indices[i*2]*3+2]
                        };

                        asg::Vec3f v2 = {
                            geom->vertices[geom->indices[i*2+1]*3],
                            geom->vertices[geom->indices[i*2+1]*3+1],
                            geom->vertices[geom->indices[i*2+1]*3+2]
                        };

                        float radius = geom->radii[i];

                        asg::Vec3f a = v2 - v1;
                        float aa = dot(a,a);
                        asg::Vec3f e = {
                            radius * sqrtf(1.f - a.x * a.x / aa),
                            radius * sqrtf(1.f - a.y * a.y / aa),
                            radius * sqrtf(1.f - a.z * a.z / aa)
                        };

                        asg::Vec3f pa = {
                            fminf(v1.x-e.x,v2.x-e.x),
                            fminf(v1.y-e.y,v2.y-e.y),
                            fminf(v1.z-e.z,v2.z-e.z)
                        };

                        asg::Vec3f pb = {
                            fmaxf(v1.x+e.x,v2.x+e.x),
                            fmaxf(v1.y+e.y,v2.y+e.y),
                            fmaxf(v1.z+e.z,v2.z+e.z)
                        };

                        v1 = pa;
                        v2 = pb;

                        for (asg::Mat4x3f trans : bounds.transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds.minVal = min(bounds.minVal,v1);
                        bounds.maxVal = max(bounds.maxVal,v1);

                        bounds.minVal = min(bounds.minVal,v2);
                        bounds.maxVal = max(bounds.maxVal,v2);
                    }
                } else {
                    // No indices, so just insert the verts directly
                    for (unsigned i=0; i<geom->numVertices; ++i) {

                        asg::Vec3f v1 = {
                            geom->vertices[i*6],
                            geom->vertices[i*6+1],
                            geom->vertices[i*6+2]
                        };

                        asg::Vec3f v2 = {
                            geom->vertices[i*6+3],
                            geom->vertices[i*6+4],
                            geom->vertices[i*6+5]
                        };

                        float radius = geom->radii[i];

                        asg::Vec3f a = v2 - v1;
                        float aa = dot(a,a);
                        asg::Vec3f e = {
                            radius * sqrtf(1.f - a.x * a.x / aa),
                            radius * sqrtf(1.f - a.y * a.y / aa),
                            radius * sqrtf(1.f - a.z * a.z / aa)
                        };

                        asg::Vec3f pa = {
                            fminf(v1.x-e.x,v2.x-e.x),
                            fminf(v1.y-e.y,v2.y-e.y),
                            fminf(v1.z-e.z,v2.z-e.z)
                        };

                        asg::Vec3f pb = {
                            fmaxf(v1.x+e.x,v2.x+e.x),
                            fmaxf(v1.y+e.y,v2.y+e.y),
                            fmaxf(v1.z+e.z,v2.z+e.z)
                        };

                        v1 = pa;
                        v2 = pb;

                        for (asg::Mat4x3f trans : bounds.transStack) {
                            v1 = trans * asg::Vec4f{v1.x,v1.y,v1.z,1.f};
                            v2 = trans * asg::Vec4f{v2.x,v2.y,v2.z,1.f};
                        }

                        bounds.minVal = min(bounds.minVal,v1);
                        bounds.maxVal = max(bounds.maxVal,v1);

                        bounds.minVal = min(bounds.minVal,v2);
                        bounds.maxVal = max(bounds.maxVal,v2);
                    }
                }
            }

#if !ASG_USE_VISIONARAY_FOR_PICKING
            // TODO: we'll later let ANARI handle this, when picking
            // is specified - and implemented by the devices

            asg::Vec3f rayDirInv = 1.f/pr->rayDir;

            asg::Vec3f t1 = (bounds.minVal - pr->rayOri) * rayDirInv; 
            asg::Vec3f t2 = (bounds.maxVal - pr->rayOri) * rayDirInv; 

            float tnear = std::max(std::min(t1.x,t2.x),std::max(std::min(t1.y,t2.y),std::min(t1.z,t2.z)));
            float tfar  = std::min(std::max(t1.x,t2.x),std::min(std::max(t1.y,t2.y),std::max(t1.z,t2.z)));

            float t = tnear < tfar ? tnear : FLT_MAX;

            if (t < pr->maxT) {
                pr->handle = obj;
                pr->maxT = t;
            }
#endif

            break;
        }
        case ASG_TYPE_OBJECT:
            // fall-through
        default: {
            asgVisitorApply(self,obj);
            break;
        }
    }
}

ASGError_t asgPickObject(ASGObject obj, ASGCamera camera, uint32_t x, uint32_t y,
                         uint32_t frameSizeX, uint32_t frameSizeY,
                         ASGObject* pickedObject, uint64_t nodeMask)
{
    PickRecord pr;
    pr.handle = nullptr;

    // TODO: this requires knowledge of the parameters stored by cams..
    // should really have ANARI handle picking, hope they'll specify that soon :)

    // TODO: return errors if these parameters cannot be retrieved
    float aspect = 0.f;
    float position[3] {0.f,0.f,0.f};
    float direction[3] {0.f,0.f,0.f};
    float upv[3] {0.f,0.f,0.f};
    ASGParam aspectParam, positionParam, directionParam, upParam;
    asgCameraGetParam(camera,"aspect",&aspectParam);
    asgParamGetValue(aspectParam,&aspect);
    asgCameraGetParam(camera,"position",&positionParam);
    asgParamGetValue(positionParam,position);
    asgCameraGetParam(camera,"direction",&directionParam);
    asgParamGetValue(directionParam,direction);
    asgCameraGetParam(camera,"up",&upParam);
    asgParamGetValue(upParam,upv);
    float fovy = M_PI/3.f; // TODO: that's just the dflt..

    asg::Vec3f eye, center, up;
    eye.x = position[0]; eye.y = position[1]; eye.z = position[2];
    center.x = eye.x+direction[0]; center.y = eye.y+direction[1]; center.z = eye.z+direction[2];
    up.x = upv[0]; up.y = upv[1]; up.z = upv[2];
    asg::Vec3f f = normalize(eye-center);
    asg::Vec3f s = normalize(cross(up,f));
    asg::Vec3f u = cross(f,s);

    asg::Vec3f U = s * std::tan(fovy*.5f) * aspect;
    asg::Vec3f V = u * std::tan(fovy*.5f);
    asg::Vec3f W = -f;

    float xf = 2.f * (x+.5f) / frameSizeX - 1.f;
    float yf = 2.f * (y+.5f) / frameSizeY - 1.f;

    pr.rayOri = eye;
    pr.rayDir = normalize(U*xf + V*yf + W);

    pr.maxT = FLT_MAX;
    ASGVisitor visitor = asgCreateVisitor(pickObject,&pr,
                                          ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN);
    asgObjectAccept(obj,visitor);
    asgDestroyVisitor(visitor);

    *pickedObject = pr.handle;

    return ASG_ERROR_NO_ERROR;
}


// Build ANARI world

struct ANARI {
    ANARIDevice device;
    ASGBuildWorldFlags_t flags;
    std::vector<ANARIInstance> instances;
    std::vector<ANARISurface> surfaces;
    std::vector<ANARIVolume> volumes;
    std::vector<ANARILight> lights;
    asg::Mat4x3f trans;
};

template <typename GroupNode>
void setANARIEntities(GroupNode groupNode, ANARI& anari)
{
    anariUnsetParameter(anari.device,groupNode,"surface");
    anariUnsetParameter(anari.device,groupNode,"volume");

    if (anari.flags & ASG_BUILD_WORLD_FLAG_LIGHTS)
        anariUnsetParameter(anari.device,groupNode,"light");

    if (anari.surfaces.size() > 0) {
        ANARIArray1D surfaces = anariNewArray1D(anari.device,anari.surfaces.data(),0,0,
                                                ANARI_SURFACE,anari.surfaces.size());
        anariSetParameter(anari.device,groupNode,"surface",ANARI_ARRAY1D,&surfaces);
        anariRelease(anari.device,surfaces);
    }

    if (anari.volumes.size() > 0) {
        ANARIArray1D volumes = anariNewArray1D(anari.device,anari.volumes.data(),0,0,
                                               ANARI_VOLUME,anari.volumes.size());
        anariSetParameter(anari.device,groupNode,"volume",ANARI_ARRAY1D,&volumes);
        anariRelease(anari.device,volumes);
    }

    if ((anari.flags & ASG_BUILD_WORLD_FLAG_LIGHTS) && anari.lights.size() > 0) {
        ANARIArray1D lights = anariNewArray1D(anari.device,anari.lights.data(),0,0,
                                              ANARI_LIGHT,anari.lights.size());
        anariSetParameter(anari.device,groupNode,"light",ANARI_ARRAY1D,&lights);
        anariRelease(anari.device,lights);
    }

    if (std::is_same<GroupNode,ANARIWorld>::value) {
        anariUnsetParameter(anari.device,groupNode,"instance");

        if (anari.instances.size() > 0) {
            ANARIArray1D instances = anariNewArray1D(anari.device,anari.instances.data(),
                                                     0,0,ANARI_INSTANCE,
                                                     anari.instances.size());
            anariSetParameter(anari.device,groupNode,"instance",ANARI_ARRAY1D,&instances);
            anariRelease(anari.device,instances);
        }
    }
}

static void visitANARIWorld(ASGVisitor self, ASGObject obj, void* userData) {

    ANARI* anari = (ANARI*)userData;
    ASGType_t t;
    asgGetType(obj,&t);

    switch (t)
    {
        case ASG_TYPE_SELECT: {
            _asgApplyVisible(self,obj,((Select*)obj->impl)->visibility.data());
            break;
        }

        case ASG_TYPE_TRANSFORM: {
            Transform* trans = (Transform*)obj->impl;

            if (anari->flags & ASG_BUILD_WORLD_FLAG_TRANSFORMS) {

                // TODO: check if we already found the surface being instanced;
                // this isn't supported by the current implementation yet
                std::vector<ANARIInstance> instances; // append, will be added in the end
                std::vector<ANARISurface> surfaces(anari->surfaces);
                std::vector<ANARIVolume> volumes(anari->volumes);
                std::vector<ANARILight> lights(anari->lights);

                anari->surfaces.clear();
                anari->volumes.clear();
                anari->lights.clear();

                asg::Mat4x3f prevTrans = anari->trans;
                asg::Mat3f A = *(asg::Mat3f*)&anari->trans;
                asg::Mat3f B{{trans->matrix[0],trans->matrix[1],trans->matrix[2]},
                             {trans->matrix[3],trans->matrix[4],trans->matrix[5]},
                             {trans->matrix[6],trans->matrix[7],trans->matrix[8]}};
                asg::Mat3f C = A * B;
                anari->trans.col0 = C.col0;
                anari->trans.col1 = C.col1;
                anari->trans.col2 = C.col2;
                anari->trans.col3 += asg::Vec3f{trans->matrix[ 9],
                                                trans->matrix[10],
                                                trans->matrix[11]};

                asgVisitorApply(self,obj);

                bool notEmpty = anari->surfaces.size() || anari->volumes.size()
                                    || anari->lights.size();

                ANARIGroup anariGroup = anariNewGroup(anari->device);
                setANARIEntities(anariGroup,*anari);
                anariCommitParameters(anari->device,anariGroup);

                anari->surfaces = std::move(surfaces);
                anari->volumes = std::move(volumes);
                anari->lights = std::move(lights);
                // instances are special b/c they're added to the world
                anari->instances.insert(anari->instances.end(),instances.begin(),
                                        instances.end());

                if (trans->anariInstance == nullptr)
                    trans->anariInstance = anariNewInstance(anari->device, "transform");

                anariSetParameter(anari->device,trans->anariInstance,"group",
                                  ANARI_GROUP,&anariGroup);
                anariSetParameter(anari->device,trans->anariInstance,"transform",
                                  ANARI_FLOAT32_MAT3x4,(float*)&anari->trans);

                anariCommitParameters(anari->device,trans->anariInstance);

                anariRelease(anari->device,anariGroup);

                if (self->visible && notEmpty)
                    anari->instances.push_back(trans->anariInstance);

                anari->trans = prevTrans;
            }

            break;
        }

        case ASG_TYPE_LIGHT: {
            Light* light = (Light*)obj->impl;

            if (anari->flags & ASG_BUILD_WORLD_FLAG_LIGHTS) {
                if (strncmp(light->type.c_str(),"point",5)==0) {
                    float color[3] {0.f,0.f,0.f};
                    float position[3] {0.f,0.f,0.f};
                    ASGParam colorParam, positionParam;
                    asgLightGetParam(obj,"color",&colorParam);
                    asgLightGetParam(obj,"position",&positionParam);
                    asgParamGetValue(colorParam,color);
                    asgParamGetValue(positionParam,position);

                    anariRelease(anari->device, light->anariLight);

                    light->anariLight = anariNewLight(anari->device,"point");

                    anariSetParameter(anari->device,light->anariLight,"color",
                                      ANARI_FLOAT32_VEC3,color);
                    anariSetParameter(anari->device,light->anariLight,"position",
                                      ANARI_FLOAT32_VEC3,position);

                    anariCommitParameters(anari->device,light->anariLight);
                }

                if (self->visible)
                    anari->lights.push_back(light->anariLight);
            }

            break;
        }

        case ASG_TYPE_SURFACE: {
            Surface* surf = (Surface*)obj->impl;

            assert(surf->geometry != nullptr);

            ANARIGeometry geomHandle = nullptr;

            if ((anari->flags & ASG_BUILD_WORLD_FLAG_GEOMETRIES) &&
                surf->geometry->type == ASG_TYPE_TRIANGLE_GEOMETRY) {

                TriangleGeom* geom = (TriangleGeom*)surf->geometry->impl;

                if (geom->anariGeometry == nullptr)
                    geom->anariGeometry
                        = anariNewGeometry(anari->device,"triangle");

                geomHandle = geom->anariGeometry;

                ANARIArray1D vertexPosition = anariNewArray1D(anari->device,
                                                              geom->vertices,
                                                              0,0,ANARI_FLOAT32_VEC3,
                                                              geom->numVertices);

                anariSetParameter(anari->device,geom->anariGeometry,"vertex.position",
                                  ANARI_ARRAY1D,&vertexPosition);

                anariRelease(anari->device,vertexPosition);

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    ANARIArray1D primitiveIndex = anariNewArray1D(anari->device,
                                                                  geom->indices,
                                                                  0,0,
                                                                  ANARI_UINT32_VEC3,
                                                                  geom->numIndices);
                    anariSetParameter(anari->device,geom->anariGeometry,
                                      "primitive.index",
                                      ANARI_ARRAY1D,&primitiveIndex);

                    anariRelease(anari->device,primitiveIndex);
                }

                anariCommitParameters(anari->device,geom->anariGeometry);
            } else if ((anari->flags & ASG_BUILD_WORLD_FLAG_GEOMETRIES) &&
                surf->geometry->type == ASG_TYPE_SPHERE_GEOMETRY) {

                SphereGeom* geom = (SphereGeom*)surf->geometry->impl;

                if (geom->anariGeometry == nullptr)
                    geom->anariGeometry
                        = anariNewGeometry(anari->device,"sphere");

                geomHandle = geom->anariGeometry;

                ANARIArray1D vertexPosition = anariNewArray1D(anari->device,
                                                              geom->vertices,
                                                              0,0,ANARI_FLOAT32_VEC3,
                                                              geom->numVertices);

                ANARIArray1D vertexRadius = anariNewArray1D(anari->device,
                                                            geom->radii,
                                                            0,0,ANARI_FLOAT32,
                                                            geom->numVertices);

                anariSetParameter(anari->device,geom->anariGeometry,"vertex.position",
                                  ANARI_ARRAY1D,&vertexPosition);

                anariSetParameter(anari->device,geom->anariGeometry,"vertex.radius",
                                  ANARI_ARRAY1D,&vertexRadius);

                anariRelease(anari->device,vertexPosition);
                anariRelease(anari->device,vertexRadius);

                if (geom->vertexColors) {
                    // TODO: support all color types
                    ANARIArray1D vertexColor = anariNewArray1D(anari->device,
                                                               geom->vertexColors,
                                                               0,0,ANARI_FLOAT32_VEC4,
                                                               geom->numVertices);

                    anariSetParameter(anari->device,geom->anariGeometry,"vertex.color",
                                      ANARI_ARRAY1D,&vertexColor);

                    anariRelease(anari->device,vertexColor);
                }

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    ANARIArray1D primitiveIndex = anariNewArray1D(anari->device,
                                                                  geom->indices,
                                                                  0,0,
                                                                  ANARI_UINT32_VEC3,
                                                                  geom->numIndices);
                    anariSetParameter(anari->device,geom->anariGeometry,
                                      "primitive.index",
                                      ANARI_ARRAY1D,&primitiveIndex);

                    anariRelease(anari->device,primitiveIndex);
                }

                anariCommitParameters(anari->device,geom->anariGeometry);
            } else if ((anari->flags & ASG_BUILD_WORLD_FLAG_GEOMETRIES) &&
                surf->geometry->type == ASG_TYPE_CYLINDER_GEOMETRY) {

                CylinderGeom* geom = (CylinderGeom*)surf->geometry->impl;

                if (geom->anariGeometry == nullptr)
                    geom->anariGeometry
                        = anariNewGeometry(anari->device,"cylinder");

                geomHandle = geom->anariGeometry;

                ANARIArray1D vertexPosition = anariNewArray1D(anari->device,
                                                              geom->vertices,
                                                              0,0,ANARI_FLOAT32_VEC3,
                                                              geom->numVertices);

                ANARIArray1D primitiveRadius = anariNewArray1D(anari->device,
                                                               geom->radii,
                                                               0,0,ANARI_FLOAT32,
                                                               geom->numVertices);

                anariSetParameter(anari->device,geom->anariGeometry,"vertex.position",
                                  ANARI_ARRAY1D,&vertexPosition);

                anariSetParameter(anari->device,geom->anariGeometry,"primitive.radius",
                                  ANARI_ARRAY1D,&primitiveRadius);

                anariRelease(anari->device,vertexPosition);
                anariRelease(anari->device,primitiveRadius);

                if (geom->vertexColors) {
                    // TODO: support all color types
                    ANARIArray1D vertexColor = anariNewArray1D(anari->device,
                                                               geom->vertexColors,
                                                               0,0,ANARI_FLOAT32_VEC4,
                                                               geom->numVertices);

                    anariSetParameter(anari->device,geom->anariGeometry,"vertex.color",
                                      ANARI_ARRAY1D,&vertexColor);

                    anariRelease(anari->device,vertexColor);
                }

                if (geom->indices != nullptr && geom->numIndices > 0) {
                    ANARIArray1D primitiveIndex = anariNewArray1D(anari->device,
                                                                  geom->indices,
                                                                  0,0,
                                                                  ANARI_UINT32_VEC3,
                                                                  geom->numIndices);
                    anariSetParameter(anari->device,geom->anariGeometry,
                                      "primitive.index",
                                      ANARI_ARRAY1D,&primitiveIndex);

                    anariRelease(anari->device,primitiveIndex);
                }

                anariCommitParameters(anari->device,geom->anariGeometry);
            }

            if ((anari->flags & ASG_BUILD_WORLD_FLAG_MATERIALS)
                    && surf->material != nullptr) {
                Material* mat = (Material*)surf->material->impl;

                if (strncmp(mat->type.c_str(),"matte",5)==0) {
                    ASGParam kdParam;
                    ASGError_t res = asgMaterialGetParam(surf->material,"kd",
                                                         &kdParam);
                    if (res != ASG_ERROR_PARAM_NOT_FOUND) {
                        if (mat->anariMaterial == nullptr)
                            mat->anariMaterial = anariNewMaterial(anari->device, "matte");

                        float kd[3];
                        asgParamGetValue(kdParam,kd);
                        //std::cout << kd[0] << ' ' << kd[1] << ' ' << kd[2] << '\n';

                        anariSetParameter(anari->device,mat->anariMaterial,"color",
                                          ANARI_FLOAT32_VEC3,kd);
                        anariCommitParameters(anari->device,mat->anariMaterial);
                    }
                }
            }

            if (surf->anariSurface == nullptr)
                surf->anariSurface = anariNewSurface(anari->device);

            if (geomHandle == nullptr) {
                // That means we didn't process geometries, so at least
                // need to obtain the handle
                switch (surf->geometry->type) {

                    case ASG_TYPE_TRIANGLE_GEOMETRY: {
                        TriangleGeom* geom = (TriangleGeom*)surf->geometry->impl;
                        geomHandle = geom->anariGeometry;
                        break;
                    }

                    case ASG_TYPE_SPHERE_GEOMETRY: {
                        SphereGeom* geom = (SphereGeom*)surf->geometry->impl;
                        geomHandle = geom->anariGeometry;
                        break;
                    }

                    case ASG_TYPE_CYLINDER_GEOMETRY: {
                        CylinderGeom* geom = (CylinderGeom*)surf->geometry->impl;
                        geomHandle = geom->anariGeometry;
                        break;
                    }
                }
            }

            anariSetParameter(anari->device,surf->anariSurface,"geometry",
                              ANARI_GEOMETRY,&geomHandle);
            if (surf->material != nullptr
                 && ((Material*)(surf->material->impl))->anariMaterial != nullptr) {
                Material* mat = (Material*)surf->material->impl;
                anariSetParameter(anari->device,surf->anariSurface,"material",
                                  ANARI_MATERIAL,&mat->anariMaterial);
            }
            anariCommitParameters(anari->device,surf->anariSurface);

            if (self->visible)
                anari->surfaces.push_back(surf->anariSurface);

            break;
        }

        case ASG_TYPE_STRUCTURED_VOLUME: {
            StructuredVolume* vol = (StructuredVolume*)obj->impl;

            if (vol->anariVolume == nullptr)
                vol->anariVolume = anariNewVolume(anari->device,"transferFunction1D");

            if (anari->flags & ASG_BUILD_WORLD_FLAG_VOLUMES) {

                assert(vol->type == ASG_DATA_TYPE_FLOAT32); // TODO!

                int volDims[] = {
                    vol->width,
                    vol->height,
                    vol->depth,
                };

                anariRelease(anari->device, vol->anariSpatialField);

                ANARIArray3D scalar = anariNewArray3D(anari->device,vol->data,
                                                      0,0,ANARI_FLOAT32,
                                                      volDims[0],volDims[1],volDims[2]);

                vol->anariSpatialField = anariNewSpatialField(anari->device,
                                                              "structuredRegular");
                anariSetParameter(anari->device,vol->anariSpatialField,"data",
                                  ANARI_ARRAY3D,&scalar);
                const char* filter = "linear";
                anariSetParameter(anari->device,vol->anariSpatialField,"filter",
                                  ANARI_STRING,filter);
                anariCommitParameters(anari->device, vol->anariSpatialField);

                float valueRange[2] = {0.f,1.f};
                //anariSetParameter(anari->device,vol->anariVolume,"valueRange",ANARI_FLOAT32_BOX1,&valueRange);

                anariSetParameter(anari->device,vol->anariVolume,"field",
                                  ANARI_SPATIAL_FIELD, &vol->anariSpatialField);

                anariCommitParameters(anari->device, vol->anariVolume);

                anariRelease(anari->device, scalar);
            }

            if (anari->flags & ASG_BUILD_WORLD_FLAG_LUTS) {
                float* rgb;
                float* alpha;
                int32_t numEntries;

                if (vol->lut1D == NULL) {
                    numEntries = 5;
                    rgb = (float*)malloc(numEntries*3*sizeof(float));
                    alpha = (float*)malloc(numEntries*sizeof(float));
                    vol->lut1D = asgNewLookupTable1D(rgb,alpha,numEntries,free);
                    asgMakeDefaultLUT1D(vol->lut1D,ASG_LUT_ID_DEFAULT_LUT);
                } else {
                    asgLookupTable1DGetRGB(vol->lut1D, &rgb);
                    asgLookupTable1DGetAlpha(vol->lut1D, &alpha);
                    asgLookupTable1DGetNumEntries(vol->lut1D, &numEntries);
                }

                ANARIArray1D anariColor = anariNewArray1D(anari->device, rgb, 0, 0, ANARI_FLOAT32_VEC3, numEntries);
                ANARIArray1D anariOpacity = anariNewArray1D(anari->device, alpha, 0, 0, ANARI_FLOAT32, numEntries);

                anariSetParameter(anari->device, vol->anariVolume, "color", ANARI_ARRAY1D, &anariColor);
                anariSetParameter(anari->device, vol->anariVolume, "opacity", ANARI_ARRAY1D, &anariOpacity);

                anariCommitParameters(anari->device, vol->anariVolume);

                anariRelease(anari->device, anariColor);
                anariRelease(anari->device, anariOpacity);
            }

            if (self->visible)
                anari->volumes.push_back(vol->anariVolume);

            break;
        }
        case ASG_TYPE_OBJECT:
            // fall-through
        default: {
            asgVisitorApply(self,obj);
            break;
        }
    }
}

ASGError_t asgBuildANARIWorld(ASGObject obj, ANARIDevice device, ANARIWorld world,
                              ASGBuildWorldFlags_t flags, uint64_t nodeMask)
{
    ANARI anari;
    anari.device = device;
    anari.flags = flags;
    anari.trans = asg::makeIdentity<asg::Mat4x3f>();
    ASGVisitor visitor = asgCreateVisitor(visitANARIWorld,&anari,
                                          ASG_VISITOR_TRAVERSAL_TYPE_CHILDREN);
    asgObjectAccept(obj,visitor);
    asgDestroyVisitor(visitor);

    setANARIEntities(world,anari);

    return ASG_ERROR_NO_ERROR;
}


