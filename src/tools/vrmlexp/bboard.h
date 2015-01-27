/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: bboard.h
 
    DESCRIPTION:  Defines a Billboard VRML 2.0 helper object
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 29 Feb. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __Billboard__H__

#define __Billboard__H__

#define Billboard_CLASS_ID1 0xABBD3442
#define Billboard_CLASS_ID2 0xFBBDBAD

#define BillboardClassID Class_ID(Billboard_CLASS_ID1, Billboard_CLASS_ID2)

extern ClassDesc *GetBillboardDesc();

class BillboardCreateCallBack;

class IBillboard : public FPMixinInterface
{
public:
    virtual void SetSize(float value, TimeValue time) = 0;
    virtual float GetSize(TimeValue time, Interval &valid = FOREVER) const = 0;

    // These methods allow for specifing and retrieving at once (simultaneously)
    // the on\off state of all light sources of a luminaire
    virtual void SetScreenAlign(int onOff, TimeValue &time) = 0;
    virtual int GetScreenAlign(TimeValue &time, Interval &valid = FOREVER) const = 0;
    // Function Publishing Methods IDs
    enum
    {
        kBILL_GET_SIZE,
        kBILL_SET_SIZE,
        kBILL_GET_SCREEN_ALIGN,
        kBILL_SET_SCREEN_ALIGN,
    };
};

#define BILLBOARDOBJECT_INTERFACE Interface_ID(0x7e631ff5, 0x7163389d)

class BillboardObject : public HelperObject, public IBillboard
{
    friend class BillboardCreateCallBack;
    friend class BillboardObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(BillboardObject *ob);

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

    static FPInterfaceDesc mInterfaceDesc;
    FPInterfaceDesc *GetDesc()
    {
        return &mInterfaceDesc;
    }
    BaseInterface *GetInterface(Interface_ID id);

    BillboardObject();
    ~BillboardObject();

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
        return GetString(IDS_BILLBOARD);
    }

    ParamDimension *GetParameterDim(int pbIndex);
    TSTR GetParameterName(int pbIndex);

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_BILLBOARD);
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
        return Class_ID(Billboard_CLASS_ID1,
                        Billboard_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = GetString(IDS_BILLBOARD_CLASS);
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
    virtual void SetSize(float value, TimeValue time);
    virtual float GetSize(TimeValue time, Interval &valid = FOREVER) const;

    virtual void SetScreenAlign(int onOff, TimeValue &time);
    virtual int GetScreenAlign(TimeValue &time, Interval &valid = FOREVER) const;
    BEGIN_FUNCTION_MAP
    PROP_TFNS(kBILL_GET_SIZE, GetSize, kBILL_SET_SIZE, SetSize, TYPE_FLOAT);
    PROP_TFNS(kBILL_GET_SCREEN_ALIGN, GetScreenAlign, kBILL_SET_SCREEN_ALIGN, SetScreenAlign, TYPE_BOOL);
    END_FUNCTION_MAP
};

#define PB_BB_SIZE 0
#define PB_BB_SCREEN_ALIGN 1
#define PB_BB_LENGTH 2

#endif
