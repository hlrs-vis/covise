/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: navinfo.h
 
    DESCRIPTION:  Defines a NavigationInformation VRML 2.0 helper object
 
    CREATED BY: Scott Morrison
 
    HISTORY: created August 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __NavInfo__H__

#define __NavInfo__H__

#define NavInfo_CLASS_ID1 0xACAD3442
#define NavInfo_CLASS_ID2 0xF00DBAD

#define NavInfoClassID Class_ID(NavInfo_CLASS_ID1, NavInfo_CLASS_ID2)

extern ClassDesc *GetNavInfoDesc();

class NavInfoCreateCallBack;

class NavInfoObject : public HelperObject
{
    friend class NavInfoCreateCallBack;
    friend class NavInfoObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(NavInfoObject *ob);

public:
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;

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

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    NavInfoObject();
    ~NavInfoObject();

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
        return GetString(IDS_NAV_INFO);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_NAV_INFO);
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
        return Class_ID(NavInfo_CLASS_ID1,
                        NavInfo_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_NAV_INFO_CLASS));
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

    IOResult Load(ILoad *iload);
};

#define PB_TYPE 0
#define PB_HEADLIGHT 1
#define PB_VIS_LIMIT 2
#define PB_SPEED 3
#define PB_COLLISION 4
#define PB_TERRAIN 5
#define PB_STEP 6
#define PB_NI_SIZE 7
#define PB_NI_SCALE 8
#define PB_NI_NEAR 9
#define PB_NI_FAR 10
#define PB_NA_LENGTH 11

#endif
