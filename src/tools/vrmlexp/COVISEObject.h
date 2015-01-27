/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: COVISEObject.h
 
    DESCRIPTION:  Defines a VRML COVISEObject Helper Class
 
    CREATED BY: Uwe Woessner
 
    HISTORY: 28.5.2014
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __VRML_COVISEOOBJECT__H__

#define __VRML_COVISEOOBJECT__H__

#define VRML_COVISEOOBJECT_CLASS_ID1 0xADAD3577
#define VRML_COVISEOOBJECT_CLASS_ID2 0x0421342

extern ClassDesc *GetVRMLCOVISEObjectDesc();

class VRMLCOVISEObjectCreateCallBack;

class VRMLCOVISEObjectObject : public HelperObject
{
    friend class VRMLCOVISEObjectCreateCallBack;
    friend INT_PTR CALLBACK VRMLCOVISEObjectRollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam);

    // Class vars
    static HWND hRollup;
    static ISpinnerControl *sizeSpin;

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
    void MakeQuad(int *f, int a, int b, int c, int d, int vab, int vbc, int vcd, int vda);
    void BuildMesh(void);
    TSTR insURL;
    BOOL useSize;

public:
    VRMLCOVISEObjectObject();
    ~VRMLCOVISEObjectObject();

#if MAX_PRODUCT_VERSION_MAJOR > 8
    RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif

    // From BaseObject
    void GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm);
    int HitTest(TimeValue t, INode *inode, int type, int crossing, int flags, IPoint2 *p, ViewExp *vpt);
    //	void Snap(TimeValue t, INode* inode, SnapInfo *snap, IPoint2 *p, ViewExp *vpt);
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
        return GetString(IDS_COVISEOBJECT);
    }

    void SetSize(float r);
    float GetSize(void)
    {
        return radius;
    }
    void SetUseSize(int v)
    {
        useSize = v;
    }
    BOOL GetUseSize(void)
    {
        return useSize;
    }

    TSTR &GetUrl(void)
    {
        return insURL;
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_COVISEOBJECT);
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
        return Class_ID(VRML_COVISEOOBJECT_CLASS_ID1,
                        VRML_COVISEOOBJECT_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_COVISEOBJECT_CLASS));
    }
    int IsKeyable()
    {
        return 1;
    }
    LRESULT CALLBACK TrackViewWinProc(HWND hwnd, UINT message,
                                      WPARAM wParam, LPARAM lParam)
    {
        return (0);
    }

    // IO
    IOResult Save(ISave *isave);
    IOResult Load(ILoad *iload);
};

#endif
