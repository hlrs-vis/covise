/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: target.h

	DESCRIPTION:  Defines a Target Object Class

	CREATED BY: Dan Silva

	HISTORY: created 11 January 1995

 *>	Copyright (c) 1994, All Rights Reserved.
 **********************************************************************/

#ifndef __TARGET__H__

#define __TARGET__H__

class TargetObject : public GeomObject
{
    friend class TargetObjectCreateCallBack;
    friend BOOL CALLBACK TargetParamDialogProc(HWND hDlg, UINT message,
                                               WPARAM wParam, LPARAM lParam);

    // Mesh cache
    static HWND hSimpleCamParams;
    static IObjParam *iObjParams;
    static Mesh mesh;
    static int meshBuilt;

    void GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm);
    void BuildMesh();

    //  inherited virtual methods for Reference-management
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);

public:
    TargetObject();

    //  inherited virtual methods:

    // From BaseObject
    int HitTest(TimeValue t, INode *inode, int type, int crossing, int flags, IPoint2 *p, ViewExp *vpt);
    void Snap(TimeValue t, INode *inode, SnapInfo *snap, IPoint2 *p, ViewExp *vpt);
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
        return GetString(IDS_DB_TARGET);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_DB_TARGET);
    }
    ObjectHandle ApplyTransform(Matrix3 &matrix);
    int UsesWireColor()
    {
        return 0;
    }
    int IsRenderable()
    {
        return 0;
    }

    // From GeomObject
    int IntersectRay(TimeValue t, Ray &r, float &at);
    ObjectHandle CreateTriObjRep(TimeValue t); // for rendering, also for deformation
    void GetWorldBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);
    void GetLocalBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);
    void GetDeformBBox(TimeValue t, Box3 &box, Matrix3 *tm, BOOL useSel);

    // Animatable methods
    void DeleteThis()
    {
        delete this;
    }
    Class_ID ClassID()
    {
        return Class_ID(TARGET_CLASS_ID, 0);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_DB_TARGET));
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

    // From ref.h
    RefTargetHandle Clone(RemapDir &remap = NoRemap());

    // IO
    IOResult Save(ISave *isave);
    IOResult Load(ILoad *iload);
};

#endif
