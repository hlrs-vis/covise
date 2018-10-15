/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: cal3d.h

    DESCRIPTION:  A Cal3D helper implementation
 
    CREATED BY: Uwe Woessner
  
    HISTORY: created 25 Apr. 2011
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef _CAL3D_H
#define _CAL3D_H

#ifndef NO_CAL3D
#include "cal3d/cal3d.h"

class Cal3DCoreHelper
{
    // misc
public:
    static const int STATE_IDLE;
    static const int STATE_FANCY;
    static const int STATE_MOTION;

    // member variables
protected:
    int m_state;
    CalCoreModel *m_calCoreModel;
    CalModel *m_calModel;
    int m_animationId[50];
    int m_animationCount;
    int m_meshId[32];
    int m_meshCount;
    //GLuint m_textureId[32];
    int m_textureCount;
    float m_motionBlend[3];
    float m_renderScale;
    float m_lodLevel;
    std::string m_path;
    std::string m_name;
    std::wstring m_VrmlName;
    bool written;

    // constructors/destructor
public:
    Cal3DCoreHelper();
    virtual ~Cal3DCoreHelper();

    // member functions
public:
    void executeAction(int action);
    float getLodLevel();
    void getMotionBlend(float *pMotionBlend);
    float getRenderScale();
    int getState();
    bool loadCfg(const std::string &strFilename);
    void onRender();
    void onShutdown();
    void onUpdate(float elapsedSeconds);
    void setLodLevel(float lodLevel);
    void setMotionBlend(float *pMotionBlend, float delay);
    void setState(int state, float delay);
    void setPath(const std::string &strPath);
    const std::string &getName()
    {
        return m_name;
    };
    void buildMesh(Mesh &mesh, float scale, TimeValue t);
    void clearWritten()
    {
        written = false;
    };
    void setWritten()
    {
        written = true;
    };
    bool wasWritten()
    {
        return written;
    };
    const std::wstring &getVRMLName()
    {
        return m_VrmlName;
    };

protected:
    //GLuint loadTexture(const std::string& strFilename);
    void renderSkeleton();
    void renderBoundingBox();
    TimeValue oldT;
};

class Cal3DCoreHelpers
{
public:
    Cal3DCoreHelpers(){};
    ~Cal3DCoreHelpers();
    Cal3DCoreHelper *getCoreHelper(const std::string &name);
    Cal3DCoreHelper *addHelper(const std::string &name);
    void clearWritten();
    std::list<Cal3DCoreHelper *> cores;
};
#define CAL3D_CLASS_ID1 0xACA3567
#define CAL3D_CLASS_ID2 0x0

extern ClassDesc *GetCal3DDesc();

class Cal3DCreateCallBack;

class Cal3DObject : public HelperObject
{
    friend class Cal3DCreateCallBack;
    friend class Cal3DObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(Cal3DObject *ob);
    Cal3DCoreHelper *coreHelper;
    float scale;

public:
    TSTR cal3d_cfg;
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;
    static ISpinnerControl *sizeSpin;
    static ISpinnerControl *animationIDSpin;
    static ISpinnerControl *actionIDSpin;
    static Cal3DCoreHelpers *cores;

    Cal3DCoreHelper *getCoreHelper()
    {
        return coreHelper;
    };
    float GetSize(void)
    {
        return 1;
    }
	void setURL(const std::string &);
	void setURL(const std::wstring &);
    void MakeQuad(int *f, int a, int b, int c, int d, int vab, int vbc, int vcd, int vda);

    BOOL needsScript; // Do we need to generate a script node?

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif
    static IObjParam *iObjParams;

    INode *audioClip;

    Mesh mesh;
    void BuildMesh(TimeValue t);

    CommandMode *previousMode;

    static ICustButton *Cal3DPickButton;
    IParamBlock *pblock;
    static IParamMap *pmapParam;

    Cal3DObject();
    ~Cal3DObject();
    void DrawEllipsoids(TimeValue t, INode *node, GraphicsWindow *gw);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif

    // From BaseObject
    void GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm);
    int HitTest(TimeValue t, INode *inode, int type, int crossing,
                int flags, IPoint2 *p, ViewExp *vpt);
    int Display(TimeValue t, INode *inode, ViewExp *vpt, int flags);
    CreateMouseCallBack *GetCreateMouseCallBack();
    void BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev);
    void EndEditParams(IObjParam *ip, ULONG flags, Animatable *next);

#if MAX_PRODUCT_VERSION_MAJOR > 14
    virtual const
#else
    virtual
#endif
        MCHAR *
        GetObjectName()
    {
        return GetString(IDS_CAL3D);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_CAL3D);
    }
    Interval ObjectValidity();
    Interval ObjectValidity(TimeValue time);
    int DoOwnSelectHilite()
    {
        return 1;
    }

    void GetWorldBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);
    void GetLocalBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);

    // Specific to Cal3D object

    TSTR &GetUrl(void)
    {
        return cal3d_cfg;
    }

    // Animatable methods
    void DeleteThis()
    {
        delete this;
    }
    Class_ID ClassID()
    {
        return Class_ID(CAL3D_CLASS_ID1,
                        CAL3D_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_CAL3D_CLASS));
    }
    int IsKeyable()
    {
        return 1;
    }
    LRESULT CALLBACK TrackViewWinProc(HWND hwnd, UINT message,
                                      WPARAM wParam, LPARAM lParam)
    {
        return 0;
    }

    int NumRefs()
    {
        return 1;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    // IO
    IOResult Save(ISave *isave);
    IOResult Load(ILoad *iload);
};

#define PB_CAL_SIZE 0
#define PB_CAL_ANIM 1
#define PB_CAL_ACTION 2
#define PB_CAL_LENGTH 3

#endif

#endif