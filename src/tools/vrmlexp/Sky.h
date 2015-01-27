/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: sky.h
 
    DESCRIPTION:  VRML 2.0 Sky helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 28 Aug. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __Sky__H__

#define __Sky__H__

#define Sky_CLASS_ID1 0xACBD3646
#define Sky_CLASS_ID2 0xF46DBFD

#define SkyClassID Class_ID(Sky_CLASS_ID1, Sky_CLASS_ID2)

extern ClassDesc *GetSkyDesc();

class SkyCreateCallBack;

class SkyObject : public HelperObject
{
    friend class SkyCreateCallBack;
    friend class SkyObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(SkyObject *ob);

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

    SkyObject();
    ~SkyObject();

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
        return GetString(IDS_SKY);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_SKY);
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
        return Class_ID(Sky_CLASS_ID1,
                        Sky_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_SKY_CLASS));
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
};

#define PB_SKY_ENABLED 0
#define PB_SKY_CURRENTTIME 1
#define PB_SKY_TIMELAPSE 2
#define PB_SKY_YEAR 3
#define PB_SKY_MONTH 4
#define PB_SKY_DAY 5
#define PB_SKY_HOUR 6
#define PB_SKY_MINUTE 7
#define PB_SKY_LATITUDE 8
#define PB_SKY_LONGITUDE 9
#define PB_SKY_ALTITUDE 10
#define PB_SKY_RADIUS 11
#define PB_SKY_SIZE 12
#define PB_SKY_LENGTH 13

#endif
