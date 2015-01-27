/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: timer.h
 
    DESCRIPTION:  Defines a VRML 2.0 TimeSensor helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 29 Aug. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __TimeSensor__H__

#define __TimeSensor__H__

#define TimeSensor_CLASS_ID1 0xACAD3442
#define TimeSensor_CLASS_ID2 0xF00BAD

#define TimeSensorClassID Class_ID(TimeSensor_CLASS_ID1, TimeSensor_CLASS_ID2)

extern ClassDesc *GetTimeSensorDesc();

class TimeSensorCreateCallBack;
class TimeSensorObjPick;

class TimeSensorObj
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
    TimeSensorObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class TimeSensorObject : public HelperObject
{
    friend class TimeSensorCreateCallBack;
    friend class TimeSensorObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(TimeSensorObject *ob);

public:
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;
    BOOL needsScript; // Do we need to generate a script node?
    BOOL vrmlWritten;

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

    Tab<TimeSensorObj *> TimeSensorObjects;
    CommandMode *previousMode;

    static ICustButton *TimeSensorPickButton;
    IParamBlock *pblock;
    static IParamMap *pmapParam;

    TimeSensorObject();
    ~TimeSensorObject();

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
        return GetString(IDS_TIME_SENSOR);
    }

    Tab<TimeSensorObj *> GetTimeSensorObjects()
    {
        return TimeSensorObjects;
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_TIME_SENSOR);
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
        return Class_ID(TimeSensor_CLASS_ID1,
                        TimeSensor_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_TIME_SENSOR_CLASS));
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
        return TimeSensorObjects.Count() + 1;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    IOResult Load(ILoad *iload);
};

#define PB_SIZE 0
#define PB_LOOP 1
#define PB_START_TIME 2
#define PB_STOP_TIME 3
#define PB_NUMOBJS 4
#define PB_START_ON_LOAD 5
#define PB_CYCLEINTERVAL 6
#define PB_LENGTH 7

#endif
