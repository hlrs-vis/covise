/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: anchor.h
 
    DESCRIPTION:  Defines a VRML 2.0 Anchor helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 17 Sept. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __Anchor__H__

#define __Anchor__H__

#define Anchor_CLASS_ID1 0x7ef53442
#define Anchor_CLASS_ID2 0xF002Bd37

#define AnchorClassID Class_ID(Anchor_CLASS_ID1, Anchor_CLASS_ID2)

extern ClassDesc *GetAnchorDesc();

class AnchorCreateCallBack;

class AnchorObject : public HelperObject
{
    friend class AnchorCreateCallBack;
    friend class AnchorObjPick;
#if MAX_PRODUCT_VERSION_MAJOR > 8
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
#else
    friend BOOL CALLBACK RollupDialogProc(HWND hDlg, UINT message,
#endif
                                             WPARAM wParam, LPARAM lParam);

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
    static IObjParam *iObjParams;

    Mesh mesh;
    void BuildMesh(TimeValue t);

    CommandMode *previousMode;

    static ICustButton *AnchorPickButton;
    static ICustButton *ParentPickButton;

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    INode *triggerObject;
    INode *cameraObject;
    TSTR description, URL, parameter;
    BOOL isJump;

    AnchorObject();
    ~AnchorObject();

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
        return GetString(IDS_ANCHOR);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_ANCHOR);
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
        return Class_ID(Anchor_CLASS_ID1,
                        Anchor_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_ANCHOR_CLASS));
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
        return 3;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);

    IOResult Load(ILoad *iload);
    IOResult Save(ISave *iload);
};

#define PB_AN_SIZE 0
#define PB_AN_TYPE 1
#define PB_AN_LENGTH 2

#endif
