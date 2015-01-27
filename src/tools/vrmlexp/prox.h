/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: prox.h
 
    DESCRIPTION:  Defines a VRML 2.0 ProximitySensor helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 5 Sept. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __ProxSensor__H__

#define __ProxSensor__H__

#define ProxSensor_CLASS_ID1 0x21fa3942
#define ProxSensor_CLASS_ID2 0xF002BD7D

#define ProxSensorClassID Class_ID(ProxSensor_CLASS_ID1, ProxSensor_CLASS_ID2)

extern ClassDesc *GetProxSensorDesc();

class ProxSensorCreateCallBack;
class ProxSensorObjPick;

class ProxSensorObj
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
    ProxSensorObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class ProxSensorObject : public HelperObject
{
    friend class ProxSensorCreateCallBack;
    friend class ProxSensorObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(ProxSensorObject *ob);

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

    Tab<ProxSensorObj *> objects;
    Tab<ProxSensorObj *> objectsExit;
    CommandMode *previousMode;

    static ICustButton *ProxSensorPickButton;
    static ICustButton *ProxSensorPickExitButton;
    IParamBlock *pblock;
    static IParamMap *pmapParam;

    ProxSensorObject();
    ~ProxSensorObject();

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
        return GetString(IDS_PROX_SENSOR);
    }

    Tab<ProxSensorObj *> GetObjects()
    {
        return objects;
    }
    Tab<ProxSensorObj *> GetExitObjects()
    {
        return objectsExit;
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_PROX_SENSOR);
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
        return Class_ID(ProxSensor_CLASS_ID1,
                        ProxSensor_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = GetString(IDS_PROX_SENSOR_CLASS);
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
        return objects.Count() + 1 + objectsExit.Count();
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    //    IOResult Load(ILoad *iload) ;
};

#define PB_PS_HEIGHT 0
#define PB_PS_WIDTH 1
#define PB_PS_LENGTH 2
#define PB_PS_ENABLED 3
#define PB_PS_NUMOBJS 4
#define PB_PS_NUMOBJS_EXIT 5
#define PB_PS_PB_LENGTH 6

#endif
