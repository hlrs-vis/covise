/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: timeSteps.h
 
    DESCRIPTION:  Defines a VRML 2.0 Timesteps helper
 
    CREATED BY: Uwe Woessner
 
    HISTORY: created 29 Aug. 2026
 
 *> Copyright (c) 2026, All Rights Reserved.
 **********************************************************************/

#ifndef __Timesteps__H__

#define __Timesteps__H__

#define Timesteps_CLASS_ID1 0xACDD3422
#define Timesteps_CLASS_ID2 0xF01BAD

#define TimestepsClassID Class_ID(Timesteps_CLASS_ID1, Timesteps_CLASS_ID2)

extern ClassDesc *GetTimestepsDesc();

class TimestepsCreateCallBack;
class TimestepsObjPick;

class TimestepsObj
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
    TimestepsObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class TimestepsObject : public HelperObject
{
    friend class TimestepsCreateCallBack;
    friend class TimestepsObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(TimestepsObject *ob);

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

    Tab<TimestepsObj *> TimestepsObjects;
    CommandMode *previousMode;

    static ICustButton *TimestepsPickButton;
    IParamBlock *pblock;
    static IParamMap *pmapParam;

    TimestepsObject();
    ~TimestepsObject();

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

#if MAX_PRODUCT_VERSION_MAJOR > 23
    const TCHAR* GetObjectName(bool localized) const override { return localized ? GetString(IDS_TIME_SENSOR) : _T("Timesteps"); }
#else

#if MAX_PRODUCT_VERSION_MAJOR > 14
    virtual const
#else
    virtual
#endif
        MCHAR*
        GetObjectName()
    {
        return GetString(IDS_TIME_SENSOR);
    }
#endif

    Tab<TimestepsObj *> GetTimestepsObjects()
    {
        return TimestepsObjects;
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
        return Class_ID(Timesteps_CLASS_ID1,
                        Timesteps_CLASS_ID2);
    }
#if MAX_PRODUCT_VERSION_MAJOR > 23
    void GetClassName(MSTR& s, bool localized) const override { s = localized ? GetString(IDS_TIME_SENSOR_CLASS) : _T("Timesteps"); }
#else

    void GetClassName(TSTR& s)
    {
        s = GetString(IDS_TIME_SENSOR_CLASS);
    }
#endif
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
        return TimestepsObjects.Count() + 1;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    IOResult Load(ILoad *iload);
};

#define PB_SIZE 0
#define PB_LOOP 1
#define PB_NUMTIMESTEPS 2
#define PB_TIMESTEPS 3
#define PB_NUMOBJS 4
#define PB_TS_SPEED 5
#define PB_LENGTH 6

#endif
