/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: arsensor.h

    DESCRIPTION:  Defines a VRML 2.0 ARSensor helper

    CREATED BY: Uwe Woessner

    HISTORY: created 3.4.2003
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __ARSensor__H__

#define __ARSensor__H__

#define ARSensor_CLASS_ID1 0x73fa3442
#define ARSensor_CLASS_ID2 0xFEE2BDAD

#define ARSensorClassID Class_ID(ARSensor_CLASS_ID1, ARSensor_CLASS_ID2)

extern ClassDesc *GetARSensorDesc();

class ARSensorCreateCallBack;
class ARSensorObjPick;

class ARSensorObj
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
    ARSensorObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class ARSensorObject : public HelperObject
{
    friend class ARSensorCreateCallBack;
    friend class ARSensorObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);

public:
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;
    BOOL needsScript; // Do we need to generate a script node?

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

    static ICustButton *ARSensorPickButton;
    static ICustButton *TrackedObjectPickButton;

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    INode *triggerObject;

    TSTR MarkerName;

    ARSensorObject();
    ~ARSensorObject();

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
        return GetString(IDS_AR_SENSOR);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_AR_SENSOR);
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
        return Class_ID(ARSensor_CLASS_ID1,
                        ARSensor_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = GetString(IDS_AR_SENSOR_CLASS);
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
        return 2;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    IOResult Load(ILoad *iload);
    IOResult Save(ISave *isave);
};

#define PB_AR_SIZE 0
#define PB_AR_ENABLED 1
#define PB_AR_FREEZE 2
#define PB_AR_HEADING_ONLY 3
#define PB_AR_MINX 4
#define PB_AR_MINY 5
#define PB_AR_MINZ 6
#define PB_AR_MAXX 7
#define PB_AR_MAXY 8
#define PB_AR_MAXZ 9
#define PB_AR_IPX 10
#define PB_AR_IPY 11
#define PB_AR_IPZ 12
#define PB_AR_POS 13
#define PB_AR_ORI 14
#define PB_AR_CURRENT_CAMERA 15
#define PB_AR_LENGTH 16

#endif
