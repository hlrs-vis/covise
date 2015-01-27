/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: MultiTouchSensor.h

    DESCRIPTION:  Defines a VRML 2.0 MultiTouchSensor helper

    CREATED BY: Uwe Woessner

    HISTORY: created 3.4.2003
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __MultiTouchSensor__H__

#define __MultiTouchSensor__H__

#define MultiTouchSensor_CLASS_ID1 0x72ea4542
#define MultiTouchSensor_CLASS_ID2 0xFEE2BBAD

#define MultiTouchSensorClassID Class_ID(MultiTouchSensor_CLASS_ID1, MultiTouchSensor_CLASS_ID2)

//extern ClassDesc* GetMultiTouchSensorDesc();

class MultiTouchSensorCreateCallBack;
class MultiTouchSensorObjPick;

class MultiTouchSensorObj
{
public:
    INode *node;
    TSTR listStr;
    void ResetStr(void)
    {
        if (node)
            listStr.printf(_T("%s"), node->GetName());
        else
            listStr.printf(_T("%s"), _T("NO_NAME"));
    }
    MultiTouchSensorObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class MultiTouchSensorObject : public HelperObject
{
    friend class MultiTouchSensorCreateCallBack;
    friend class MultiTouchSensorObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);

public:
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;
    BOOL needsScript; // Do we need to generate a script node?
    BOOL surfaceMode;

    static Quat surfaceRot;
    static Point3 surfaceMin;
    static float surfaceSize;

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif
    float radius;
    static IObjParam *iObjParams;

    Mesh mesh;
    void BuildMesh(TimeValue t);

    CommandMode *previousMode;

    static ICustButton *MultiTouchSensorPickButton;
    static ICustButton *TrackedObjectPickButton;

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    INode *triggerObject;

    TSTR MarkerName;

    MultiTouchSensorObject();
    ~MultiTouchSensorObject();

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

    int NumSubObjTypes();
    ISubObjType *GetSubObjType(int i);

    void GetSubObjectCenters(SubObjAxisCallback *cb, TimeValue t, INode *node, ModContext *mc);
    void GetSubObjectTMs(SubObjAxisCallback *cb, TimeValue t, INode *node, ModContext *mc);
    //int SubObjectIndex(HitRecord *hitRec);

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
        return GetString(IDS_MT_SENSOR);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_MT_SENSOR);
    }
    Interval ObjectValidity();
    Interval ObjectValidity(TimeValue time);
    int DoOwnSelectHilite()
    {
        return 1;
    }

    void GetWorldBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);
    void GetLocalBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);

    // Animatable methods
    void DeleteThis()
    {
        delete this;
    }
    Class_ID ClassID()
    {
        return Class_ID(MultiTouchSensor_CLASS_ID1,
                        MultiTouchSensor_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = GetString(IDS_MT_SENSOR_CLASS);
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
    void enterSurfaceMode();
    void exitSurfaceMode();

    void Transform(TimeValue t, Matrix3 &partm, Matrix3 tmAxis,
                   BOOL localOrigin, Matrix3 xfrm, int type);

    void Move(TimeValue t, Matrix3 &partm, Matrix3 &tmAxis, Point3 &val, BOOL localOrigin = FALSE);
    void Rotate(TimeValue t, Matrix3 &partm, Matrix3 &tmAxis, Quat &val, BOOL localOrigin = FALSE);
    void Scale(TimeValue t, Matrix3 &partm, Matrix3 &tmAxis, Point3 &val, BOOL localOrigin = FALSE);
    void TransformStart(TimeValue t);
    void TransformHoldingFinish(TimeValue t);
    void TransformFinish(TimeValue t);
    void TransformCancel(TimeValue t);

    int NumRefs()
    {
        return 2;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    IOResult Load(ILoad *iload);
    IOResult Save(ISave *isave);

    void ActivateSubobjSel(int level, XFormModes &modes);

private:
    float size;
    MoveModBoxCMode *moveMode;
    RotateModBoxCMode *rotMode;
    UScaleModBoxCMode *uscaleMode;
};

class TransformPlaneRestore : public RestoreObj
{
public:
    Point3 oldSurfaceMin, newSurfaceMin;
    Quat oldSurfaceRot, newSurfaceRot;
    float oldSurfaceSize, newSurfaceSize;
    MultiTouchSensorObject *em;
    TransformPlaneRestore(MultiTouchSensorObject *emm);
    void Restore(int isUndo);
    void Redo();
    int Size()
    {
        return 2 * (sizeof(Point3) + sizeof(Quat) + sizeof(float))
               + sizeof(MultiTouchSensorObject *);
    }
    TSTR Description()
    {
        return TSTR(_T("Surface Plane move"));
    }
};

#define PB_MT_SIZE 0
#define PB_MT_ENABLED 1
#define PB_MT_FREEZE 2
#define PB_MT_MINX 3
#define PB_MT_MINY 4
#define PB_MT_MINZ 5
#define PB_MT_SIZEX 6
#define PB_MT_SIZEY 7
#define PB_MT_SIZEZ 8
#define PB_MT_ORIH 9
#define PB_MT_ORIP 10
#define PB_MT_ORIR 11
#define PB_MT_IPX 12
#define PB_MT_IPY 13
#define PB_MT_IPZ 14
#define PB_MT_POS 15
#define PB_MT_ORI 16
#define PB_MT_LENGTH 17

#endif
