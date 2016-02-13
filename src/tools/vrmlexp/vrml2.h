/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: vrml2.h

	DESCRIPTION:  VRML 2.0 file export class defs

	CREATED BY: Scott Morrison

	HISTORY: created June, 1996

 *>	Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "iparamb2.h"
#include <string>
#include <vector>
#include "tabletui.h"

#if MAX_PRODUCT_VERSION_MAJOR > 14 && ! defined FASTIO
#include "maxtextfile.h"
#define MAXSTREAMDECL MaxSDK::Util::TextFile::Writer
#define MSTREAMPRINTF mStream.Printf( _T
#else
#define MAXSTREAMDECL FILE *
#define MSTREAMPRINTF fprintf((mStream), 
#endif

#define RUN_BY_PROX_SENSOR 1
#define RUN_BY_TIME_SENSOR 2
#define RUN_BY_TOUCH_SENSOR 4
#define RUN_BY_ONOFF_SENSOR 8
#define RUN_BY_COVER_SENSOR 16
#define RUN_BY_TABLETUI_SENSOR 32
#define RUN_BY_ANY_SENSOR 63
#define RUN_BY_SWITCH_SENSOR 64

#define NO_EFFECT 0
#define BUMP_MAPPING 1
#define BUMP_MAPPING_ENV 2
#define BUMP_MAPPING_CUBE 3
#define NUM_EFFECTS 4

// Type of parameters in an acubic map
enum
{
    acubic_size,
    acubic_blur,
    acubic_bluroffset,
    acubic_near,
    acubic_far,
    acubic_source,
    acubic_useatmospheric,
    acubic_applyblur,
    acubic_frametype,
    acubic_nthframe,
    acubic_bitmap_names,
    acubic_outputname,
};

#define ENTER_FIELD 1
#define EXIT_FIELD 2

class ARSensorObject;
class MultiTouchSensorObject;
class COVERObject;
class TabletUIObject;
class TabletUIElement;
class TabletUIElementList;
class Cal3DObject;

struct AnimRoute
{
    AnimRoute()
    {
        mToNode = NULL;
    }
    AnimRoute(const TCHAR *from, const TCHAR *to, INode *fromNode, INode *node, int field, int tuiElementType)
    {
        mFromName = from;
        mToName = to;
        mToNode = node;
        mFromNode = fromNode;
        mField = field;
        mTuiElementType = tuiElementType;
    }
    TSTR mFromName; // route nodes from animate trigger
    TSTR mToName; // route anim trigger to
    INode *mToNode; // INode for route to
    INode *mFromNode; // INode for route from
    int mField; // Field to route from
    int mTuiElementType;
};

struct InterpRoute
{
    InterpRoute()
    {
        mType = 0;
    }
    InterpRoute(const TCHAR *interp, int type, const TCHAR *nodeName, INode *node)
    {
        mInterp = interp;
        mType = type;
        mNodeName = nodeName;
        mNode = node;
    }

    TSTR mInterp;
    int mType;
    TSTR mNodeName;
    INode *mNode;
};

// Object hash table for instancing

struct ObjectBucket
{
    ObjectBucket(Object *o)
    {
        obj = o;
        objectUsed = FALSE;
        hasName = FALSE;
        hasInstName = FALSE;
        next = NULL;
        numInstances = 0;
    }
    ~ObjectBucket()
    {
        delete next;
    }
    Object *obj;
    BOOL objectUsed;
    BOOL hasName;
    BOOL hasInstName;
    TSTR name;
    TSTR instName;
    BOOL instMirrored;
    ObjectBucket *next;
    int numInstances;
};

#define OBJECT_HASH_TABLE_SIZE 1001

class ShaderEffect
{
public:
    ShaderEffect()
    {
        name = _T("");
        paramValues = _T("");
    }
    ShaderEffect(TSTR &n)
    {
        name = n;
        paramValues = _T("");
    }
    ShaderEffect(TCHAR *n)
    {
        name = n;
        paramValues = _T("");
    }
    ~ShaderEffect()
    {
    }
    TSTR getName()
    {
        return name;
    };
    TSTR getParamValues()
    {
        return paramValues;
    };
    void setParamValues(TSTR &v)
    {
        paramValues = v;
    };

private:
    TSTR name;
    TSTR paramValues;
};
class ObjectHashTable
{
public:
    ObjectHashTable()
    {
        mTable.SetCount(OBJECT_HASH_TABLE_SIZE);
        for (int i = 0; i < OBJECT_HASH_TABLE_SIZE; i++)
            mTable[i] = NULL;
    }
    ~ObjectHashTable()
    {
        for (int i = 0; i < OBJECT_HASH_TABLE_SIZE; i++)
            delete mTable[i];
    }

    ObjectBucket *AddObject(Object *obj, bool countInstances = false);

private:
    Tab<ObjectBucket *> mTable;
};

struct SensorBucket
{
    SensorBucket(INode *node)
    {
        mNode = node;
        mSensors = NULL;
        mTUIElems = NULL;
        mNext = NULL;
    }
    ~SensorBucket()
    {
        delete mSensors;
        delete mTUIElems;
    }
    INode *mNode;
    INodeList *mSensors;
    TabletUIElementList *mTUIElems;
    SensorBucket *mNext;
};

#define SENSOR_HASH_TABLE_SIZE 97

class SensorHashTable
{
public:
    SensorHashTable()
    {
        mTable.SetCount(SENSOR_HASH_TABLE_SIZE);
        for (int i = 0; i < SENSOR_HASH_TABLE_SIZE; i++)
            mTable[i] = NULL;
    }
    ~SensorHashTable()
    {
        for (int i = 0; i < SENSOR_HASH_TABLE_SIZE; i++)
            delete mTable[i];
    }

    void AddSensor(INode *node, INode *sensor, TabletUIElement *el);
    SensorBucket *FindSensor(INode *node);

private:
    Tab<SensorBucket *> mTable;
};

// Hash table of strings
struct textureTableString
{
    const TCHAR *textureName;
    int index;
};

#define TEXTURE_HASH_TABLE_SIZE 511

class TextureTable
{
public:
    TextureTable(){};

    ~TextureTable()
    {
        for (int i = 0; i < TEXTURE_HASH_TABLE_SIZE; i++)
            for (int j = 0; j < mTexture[i].Count(); j++)
                delete mTexture[i][j];
    }

    Tab<const TCHAR *> const &Find(textureTableString *s) const
    {
        s->index = hash(s->textureName);
        return mTexture[s->index];
    }

    void Add(textureTableString *s)
    {
        mTexture[s->index].Append(1, &s->textureName);
    };

private:
    int hash(const TCHAR *textureName) const
    {
        assert((textureName != 0) && (textureName[0] != 0));
        unsigned index = textureName[0];
        for (int i = 1; textureName[i] != 0; ++i)
        {
            index = (index << 4) + textureName[i];
        }
        return index % TEXTURE_HASH_TABLE_SIZE;
    }

    Tab<const TCHAR *> mTexture[TEXTURE_HASH_TABLE_SIZE];
};

class VRML2Export
{
public:
    VRML2Export();
    ~VRML2Export();

    void askForConfirmation();
    BOOL mReplace; // true if user wants current file to be replaced
    BOOL mReplaceAll; // true if user wants all files to be replaced
    BOOL mSkipAll; // true if user wants all files to be replaced
    TSTR sourceFile;
    TSTR destFile;
    int effect;

    int DoExport(const TCHAR *name, Interface *i, VRBLExport *exp);
#ifdef _LEC_
    int DoFBExport(const TCHAR *name, Interface *i, VRBLExport *exp, int frame, TimeValue time);
#endif

    BOOL processTexture(TSTR bitmapFile, TSTR &fileName, TSTR &url);
    inline BOOL GetGenNormals()
    {
        return mGenNormals;
    }
    inline void SetGenNormals(BOOL gen)
    {
        mGenNormals = gen;
    }
    inline BOOL GetDefUse()
    {
        return mDefUse;
    }
    inline void SetDefUse(BOOL gen)
    {
        mDefUse = gen;
    }
    inline BOOL GetIndent()
    {
        return mIndent;
    }
    inline void SetIndent(BOOL in)
    {
        mIndent = in;
    }
    inline ExportType GetExportType()
    {
        return mType;
    }
    inline void SetExportType(ExportType t)
    {
        mType = t;
    }
    inline Interface *GetIP()
    {
        return mIp;
    }
    inline INode *GetCamera()
    {
        return mCamera;
    }
    inline void SetCamera(INode *cam)
    {
        mCamera = cam;
    }
    inline void SetUsePrefix(BOOL u)
    {
        mUsePrefix = u;
    }
    inline BOOL GetUsePrefix()
    {
        return mUsePrefix;
    }
    inline void SetUrlPrefix(TSTR &s)
    {
        mUrlPrefix = s;
    }
    inline TSTR &GetUrlPrefix()
    {
        return mUrlPrefix;
    }
    inline void SetFields(BOOL f)
    {
        mGenFields = f;
    }
    inline BOOL GetFields()
    {
        return mGenFields;
    }
    inline BOOL IsVRML2()
    {
        return TRUE;
    }
    inline BOOL GetZUp()
    {
        return mZUp;
    }
    inline void SetZUp(BOOL zup)
    {
        mZUp = zup;
    }
    inline int GetDigits()
    {
        return mDigits;
    }
    inline void SetDigits(int n)
    {
        mDigits = n;
    }
    inline BOOL GetCoordInterp()
    {
        return mCoordInterp;
    }
#ifdef _LEC_
    inline BOOL GetFlipBook()
    {
        return mFlipBook;
    }
#endif
    inline void SetCoordInterp(BOOL ci)
    {
        mZUp = ci;
    }
    inline BOOL GetPreLight()
    {
        return mPreLight;
    }
    inline void SetPreLight(BOOL i)
    {
        mPreLight = i;
    }
    //    inline void SetPolygonType(int type)    { mPolygonType = type; }

    inline BOOL GetTformSample()
    {
        return mTformSample;
    }
    inline void SetTformSample(BOOL b)
    {
        mTformSample = b;
    }
    inline int GetTformSampleRate()
    {
        return mTformSampleRate;
    }
    inline void SetTformSampleRate(int rate)
    {
        mTformSampleRate = rate;
    }

    inline BOOL GetCoordSample()
    {
        return mCoordSample;
    }
    inline void SetCoordSample(BOOL b)
    {
        mCoordSample = b;
    }
    inline int GetCoordSampleRate()
    {
        return mCoordSampleRate;
    }
    inline void SetCoordSampleRate(int rate)
    {
        mCoordSampleRate = rate;
    }
    inline TSTR &GetInfo()
    {
        return mInfo;
    }
    inline void SetInfo(TCHAR *info)
    {
        mInfo = info;
    }
    inline TSTR &GetTitle()
    {
        return mInfo;
    }
    inline void SetTitle(TCHAR *title)
    {
        mTitle = title;
    }
    inline BOOL GetExportHidden()
    {
        return mExportHidden;
    }
    inline void SetExportHidden(BOOL eh)
    {
        mExportHidden = eh;
    }
    inline bool isChildSelected(INode *node)
    {
        if (node->Selected())
            return true;
        int n = node->NumberOfChildren();
        for (int i = 0; i < n; i++)
            if (isChildSelected(node->GetChildNode(i)))
                return true;
        return false;
    };
    inline bool isChildVisible(INode *node)
    {
        int n = node->NumberOfChildren();
        for (int i = 0; i < n; i++)
            if (isChildVisible(node->GetChildNode(i)))
                return true;
        if (node->IsHidden())
            return false;
        return true;
    };
    inline bool doExport(INode *node)
    {
        if (mExportSelected)
        {
            if (isChildSelected(node))
            {
                if (mExportHidden)
                    return true;
                if (node->IsHidden() && !isChildVisible(node))
                    return false;
                return true;
            }
            else
            {
                // check, if any child is selected
                return false;
            }
        }
        else
        {
            if (!mExportHidden && (node->IsHidden() && !isChildVisible(node)))
                return false;
        }
        return true;
    };

    Interface *mIp; // MAX interface pointer

private:
    TCHAR *point(Point3 &p);
    TCHAR *scalePoint(Point3 &p);
    TCHAR *normPoint(Point3 &p);
    TCHAR *axisPoint(Point3 &p, float ang);
    TCHAR *texture(UVVert &uv);
    TCHAR *color(Color &c);
    TCHAR *color(Point3 &c);
    TCHAR *floatVal(float f);

    // VRML Output routines
    void Indent(int level);
    size_t MaybeNewLine(size_t width, int level);
    void StartNode(INode *node, int level, BOOL outputName, Object *);
    void EndNode(INode *node, Object *obj, int level, BOOL lastChild);
    BOOL IsBBoxTrigger(INode *node);
    BOOL OutputNodeTransform(INode *node, int level, BOOL mirrored);
    void OutputMultiMtl(Mtl *mtl, int level);
    BOOL OutputMaterial(INode *node, BOOL &isWire, BOOL &twoSided, int level,
                        int textureNum);
    TCHAR *textureName(const TCHAR *name, int blendMode, bool environment = false);
    void OutputPolyShapeObject(INode *node, PolyShape &shape, int level);
    BOOL HasTexture(INode *node, BOOL &isWire);
    TSTR PrefixUrl(TSTR &fileName);
    TextureDesc *GetMtlTex(Mtl *mtl, BOOL &isWire, int mapChannel = 1, int askForSubTexture = 0);
    TextureDesc *GetMatTex(INode *node, BOOL &isWire, int mapChannel = 1, int askForSubTexture = 0);
    void GetTextures(Mtl *mtl, BOOL &isWire, int &numTexDesks, TextureDesc **tds);
    void OutputNormalIndices(Mesh &mesh, NormalTable *normTab, int level,
                             int textureNum);
    NormalTable *OutputNormals(Mesh &mesh, int level);
    BOOL OutputMaxLOD(INode *node, Object *obj, int level, int numLevels, float *distances, INode **children, int numChildren, BOOL mirrored);
    void OutputTriObject(INode *node, TriObject *obj, BOOL multiMat,
                         BOOL isWire, BOOL twoSided, int level,
                         int textureNum, BOOL pMirror);

    BOOL hasMaterial(TriObject *obj, int textureNum);
    void OutputPolygonObject(INode *node, TriObject *obj, BOOL multiMat,
                             BOOL isWire, BOOL twoSided, int level, int textureNum,
                             BOOL pMirror);
    BOOL isVrmlObject(INode *node, Object *obj, INode *parent, bool hastVisController);
    BOOL ChildIsAnimated(INode *node);
    BOOL ObjIsAnimated(Object *obj);
    BOOL ObjIsPrim(INode *node, Object *obj);
    void VrmlOutObject(INode *node, INode *parent, Object *obj, int level,
                       BOOL mirrored);
    BOOL VrmlOutSphereTest(INode *node, Object *obj);
    BOOL VrmlOutSphere(INode *node, Object *obj, int level);
    BOOL VrmlOutCylinder(INode *node, Object *obj, int level);
    BOOL VrmlOutCone(INode *node, Object *obj, int level);
    BOOL VrmlOutCube(INode *node, Object *obj, int level);
    BOOL VrmlOutCamera(INode *node, Object *obj, int level);
    BOOL VrmlOutSound(INode *node, SoundObject *obj, int level);
    void VrmlOutSwitchCamera(INode *node, INode *sw, int level);
    int VrmlOutSwitch(INode *node, int level);
    void TouchSensorMovieScript(TCHAR *objName, int level);
    void SensorBindScript(const TCHAR *objName, const TCHAR *name, int level, INode *node = NULL, INode *child = NULL, int type = 0);
    BOOL VrmlOutTouchSensor(INode *node, int level);
    BOOL VrmlOutARSensor(INode *node, int level);
    BOOL VrmlOutMTSensor(INode *node, int level);
    BOOL VrmlOutCOVER(INode *node, int level);
    void BindCamera(INode *node, INode *child, TCHAR *vrmlObjName, int type, int level);
    BOOL AddChildObjRoutes(INode *node, INode *animNode, Tab<Class_ID> childClass, INode *otop, TCHAR *vrmlObjName, int type1, int type2, int level, bool movie);
    BOOL VrmlOutProxSensor(INode *node, ProxSensorObject *obj, int level);
    BOOL VrmlOutBillboard(INode *node, Object *obj, int level);
    void VrmlOutTimeSensor(INode *node, TimeSensorObject *obj, int level);
    //void VrmlAnchorHeader(INode* node, MrBlueObject* obj,
    //                      VRBL_TriggerType type, BOOL fromParent, int level);
    //BOOL VrmlOutMrBlue(INode* node, INode* parent, MrBlueObject* obj,
    //                   int* level, BOOL fromParent);
    BOOL VrmlOutInline(VRMLInsObject *obj, int level);
    BOOL VrmlOutCOVISEObject(VRMLCOVISEObjectObject *obj, int level);
    BOOL VrmlOutCal3D(Cal3DObject *obj, int level);

    BOOL VrmlOutLOD(INode *node, LODObject *obj, int level, BOOL mirrored);
    void VrmlOutCoordinateInterpolator(INode *node, Object *obj, int level,
                                       BOOL pMirror);
    BOOL VrmlOutCylinderTest(INode *node, Object *obj);
    BOOL VrmlOutCylinderTform(INode *node, Object *obj, int level,
                              BOOL mirrored);
    BOOL VrmlOutConeTest(INode *node, Object *obj);
    BOOL VrmlOutConeTform(INode *node, Object *obj, int level, BOOL mirrored);
    BOOL VrmlOutCubeTest(INode *node, Object *obj);
    BOOL VrmlOutCubeTform(INode *node, Object *obj, int level, BOOL mirrored);
    BOOL VrmlOutSpecialTform(INode *node, Object *obj, int level,
                             BOOL mirrored);
    BOOL VrmlOutSpecial(INode *node, INode *parent, Object *obj, int level,
                        BOOL mirrored);
    //int StartMrBlueHelpers(INode* node, int level);
    //void EndMrBlueNode(INode* childNode, int& level, BOOL fromParent);
    //void EndMrBlueHelpers(INode* node, int level);
    BOOL VrmlOutPointLight(INode *node, LightObject *light, int level);
    BOOL VrmlOutDirectLight(INode *node, LightObject *light, int level);
    BOOL VrmlOutSpotLight(INode *node, LightObject *light, int level);
    BOOL VrmlOutTopPointLight(INode *node, LightObject *light);
    BOOL VrmlOutTopDirectLight(INode *node, LightObject *light);
    BOOL VrmlOutTopSpotLight(INode *node, LightObject *light);
    void OutputTopLevelLight(INode *node, LightObject *light);
    void WriteControllerData(INode *node,
                             Tab<TimeValue> &posTimes, Tab<Point3> &posKeys,
                             Tab<TimeValue> &rotTimes, Tab<AngAxis> &rotKeys,
                             Tab<TimeValue> &sclTImes, Tab<ScaleValue> &sclKeys,
                             int type, int level);
    void WriteAllControllerData(INode *node, int type, int level,
                                Control *lc);

    void WriteVisibilityData(INode *node, int level);
    BOOL IsLight(INode *node);
    BOOL IsCamera(INode *node);
    BOOL IsAudio(INode *node);
    Control *GetLightColorControl(INode *node);
    void VrmlOutControllers(INode *node, int level);
    void ScanSceneGraph1();
    void ScanSceneGraph2();
    void ComputeWorldBoundBox(INode *node, ViewExp *vpt);
    bool OutputSwitches(INode *node, int level);
    INode *isSwitched(INode *node, INode *firstNode = NULL);
    void OutputTouchSensors(INode *node, int level);
    void OutputARSensors(INode *node, int level);
    void OutputMTSensors(INode *node, int level);
    bool OutputTabletUIScripts(INode *node, int level);
    BOOL VrmlOutARSensor(INode *node, ARSensorObject *obj, int level);
    BOOL VrmlOutMTSensor(INode *node, MultiTouchSensorObject *obj, int level);
    BOOL VrmlOutCOVER(INode *node, COVERObject *obj, int level);
    void VrmlOutTUI(INode *node, TabletUIElement *el, int level);
    BOOL VrmlOutTUIElement(TabletUIElement *el, INode *node, int level);
    BOOL VrmlOutTUIButton(TabletUIElement *el, INode *node, int level);
    void TraverseNode(INode *node);
    BOOL ObjectIsLODRef(INode *node);
    void VrmlOutTopLevelCamera(int level, INode *node, BOOL topLevel);
    void VrmlOutTopLevelNavInfo(int level, INode *node, BOOL topLevel);
    void VrmlOutTopLevelBackground(int level, INode *node, BOOL topLevel);
    void VrmlOutTopLevelFog(int level, INode *node, BOOL topLevel);
    void VrmlOutTopLevelSky(int level, INode *node, BOOL topLevel);
    void VrmlOutInitializeAudioClip(int level, INode *node);
    void VrmlOutAudioClip(int level, INode *node);
    void VrmlOutFileInfo();
    void VrmlOutWorldInfo();
    void VrmlOutGridHelpers(INode *);

    int StartAnchor(INode *node, int &level);
    void VrmlOutNode(INode *node, INode *parent, int level, BOOL isLOD,
                     BOOL lastChild, BOOL mirrored);
    void InitInterpolators(INode *node);
    void AddInterpolator(const TCHAR *interp, int type, const TCHAR *name, INode *node);
    void WriteInterpolatorRoutes(int level);
    void AddCameraAnimRoutes(TCHAR *vrmlObjName, INode *fromNode, INode *top, int field = 0);
    void AddAnimRoute(const TCHAR *from, const TCHAR *to, INode *fromNode, INode *node, int field = 0, int tuiElementType = 0);
    int NodeNeedsTimeSensor(INode *node);
    void WriteAnimRoutes();
    void WriteScripts();
    TCHAR *VrmlParent(INode *node);
    BOOL IsAimTarget(INode *node);
    void GenerateUniqueNodeNames(INode *node);
    TCHAR *isMovie(const TCHAR *);
    BOOL ObjectIsReferenced(INode *lodNode, INode *node);
    void VrmlOutSwitchScript(INode *node);

    MAXSTREAMDECL mStream; // The file mStream to write
    TCHAR *mFilename; // The export .WRL filename
    BOOL mGenNormals; // Generate normals in the VRML file
    BOOL mIndent; // Should we indent?
    INodeList *mLodList; // List of LOD objects in the scene
    INodeList *mTimerList; // List of TimeSensor Nodes in the scene
    INodeList *mTabletUIList; // List of TabletUI Nodes in the scene
    INodeList *mScriptsList; // List of Script Nodes in the scene
    ExportType mType; // Language to export (VRML, VRML, ...)
    INode *mCamera; // Initial camera;
    INode *mNavInfo; // Initial Navigation Info;
    INode *mBackground; // Initial Background node
    INode *mFog; // Initial Fog node
    INode *mSky; // Initial Sky node
    BOOL mUsePrefix; // Use URL Prefix
    TSTR mUrlPrefix; // The URL prefix
    BOOL mGenFields; // Generate "fields" statements
    BOOL mHadAnim; // File has animation data
    TimeValue mStart; // First frame of the animation
    TSTR mTimer; // Name of active TimeSensor
    Tab<TSTR> mInterps; // Interpolators that need ROUTE statements
    Tab<int> mInterpTypes; // Type of interpolator nodes
    Tab<TSTR> mInterpNodes; // Nodes for interpolators
    float mCycleInterval; // Length of animation in seconds
    Tab<InterpRoute> mInterpRoutes; // Routes for Intpolator nodes
    Tab<AnimRoute> mAnimRoutes; // route nodes from anim
    BOOL mZUp; // Z axis if true, Y axis otherwise
    int mDigits; // Digits of precision on output
    BOOL mCoordInterp; // Generate coordinate interpolators
#ifdef _LEC_
    BOOL mFlipBook; // Generate one VRML file per frame (LEC request)
#endif
    BOOL mTformSample; // TRUE for once per frame
    int mTformSampleRate; // Custom sample rate
    BOOL mCoordSample; // TRUE for once per frame
    int mCoordSampleRate; // Custom sample rate
    ObjectHashTable mObjTable; // Hash table of all objects in the scene
    SensorHashTable mSensorTable; // Hash table of all TouchSensor and Anchors
    TextureTable mTextureTable; // all textures and movie textures
    Box3 mBoundBox; // Bounding box for the whole scene
    TSTR mTitle; // Title of world
    TSTR mInfo; // Info for world
    BOOL mExportHidden; // Export hidden objects
    BOOL mPrimitives; // Create VRML primitves
    BOOL mHasLights; // TRUE iff scene has lights
    BOOL mHasNavInfo; // TRUE iff scene has NavigationInfo
    int mPolygonType; // 0 triangles, 1 quads, 2 ngons
    NodeTable mNodes; // hash table of all nodes name in the scene
    BOOL mEnableProgressBar; // this is used by the progress bar
    BOOL mPreLight; // should we calculate the color per vertex
    BOOL mCPVSource; // 1 if MAX's; 0 if should we need to calculate the color per vertex
    CallbackTable *mCallbacks; // export callback methods
    BOOL mDefUse; // should we defuse the file
    BOOL mExpLights; // should we export lights
    BOOL mCopyTextures; // should we export lights
    BOOL mForceWhite; // should force textured material to white
    BOOL mExportSelected; // true if in exportSelected mode
    BOOL mExportOccluders; // export lines as occluders
    BOOL haveDiffuseMap; // does the Material have a diffuseMap? if so, set Color to 1 1 1
    Tab<INode *> switchObjects;
    int numSwitchObjects;
};

float GetLosProxDist(INode *node, TimeValue t);
Point3 GetLosVector(INode *node, TimeValue t);

int reduceAngAxisKeys(Tab<TimeValue> &times, Tab<AngAxis> &points, float eps);
int reducePoint3Keys(Tab<TimeValue> &times, Tab<Point3> &points, float eps);
int reduceScaleValueKeys(Tab<TimeValue> &times, Tab<ScaleValue> &points,
                         float eps);
void CommaScan(TCHAR *buf);
