/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: thermal.h
 
    DESCRIPTION:  Defines a VRML 2.0 Thermal node helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 29 Feb. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __Thermal__H__

#define __Thermal__H__

#define Thermal_CLASS_ID1 0xAC92443
#define Thermal_CLASS_ID2 0xF83deAE

#define ThermalClassID Class_ID(Thermal_CLASS_ID1, Thermal_CLASS_ID2)

extern ClassDesc *GetThermalDesc();

class ThermalCreateCallBack;

class ThermalObject : public HelperObject
{
    friend class ThermalCreateCallBack;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(ThermalObject *ob);

public:
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;
    static ISpinnerControl *minBackSpin;
    static ISpinnerControl *maxBackSpin;
    static ISpinnerControl *minFrontSpin;
    static ISpinnerControl *maxFrontSpin;
    static ISpinnerControl* heightSpin;
    static ISpinnerControl* turbulenceSpin;
    static ISpinnerControl* vxSpin;
    static ISpinnerControl* vySpin;
    static ISpinnerControl* vzSpin;

    BOOL needsScript; // Do we need to generate a script node?

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif
    static IObjParam *iObjParams;

    INode *audioClip;

    Mesh mesh;
    void BuildMesh(TimeValue t);


    IParamBlock *pblock;
    static IParamMap *pmapParam;

    ThermalObject();
    ~ThermalObject();
    void DrawEllipsoids(TimeValue t, INode *node, GraphicsWindow *gw);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif

    // From BaseObject
    void GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm);
    int Display(TimeValue t, INode *inode, ViewExp *vpt, int flags);
    CreateMouseCallBack *GetCreateMouseCallBack();
    void BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev);
    void EndEditParams(IObjParam *ip, ULONG flags, Animatable *next);

#if MAX_PRODUCT_VERSION_MAJOR > 23
    const TCHAR* GetObjectName(bool localized) const override { return localized ? GetString(IDS_THERMAL) : _T("Thermal"); }
#else

#if MAX_PRODUCT_VERSION_MAJOR > 14
    virtual const
#else
    virtual
#endif
        MCHAR*
        GetObjectName()
    {
        return GetString(IDS_THERMAL);
    }
#endif

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_THERMAL);
    }
    Interval ObjectValidity();
    Interval ObjectValidity(TimeValue time);
    int DoOwnSelectHilite()
    {
        return 1;
    }

    void GetWorldBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);
    void GetLocalBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);

    // Specific to thermal object
    float GetHeight(TimeValue t, Interval& valid = Interval(0, 0));
    float GetTurbulence(TimeValue t, Interval& valid = Interval(0, 0));
    float GetVX(TimeValue t, Interval& valid = Interval(0, 0));
    float GetVY(TimeValue t, Interval& valid = Interval(0, 0));
    float GetVZ(TimeValue t, Interval& valid = Interval(0, 0));
    float GetMinBack(TimeValue t, Interval& valid = Interval(0, 0));
    float GetMaxBack(TimeValue t, Interval &valid = Interval(0, 0));
    float GetMinFront(TimeValue t, Interval &valid = Interval(0, 0));
    float GetMaxFront(TimeValue t, Interval &valid = Interval(0, 0));
    void SetHeight(TimeValue t, float f);
    void SetTurbulence(TimeValue t, float f);
    void SetVX(TimeValue t, float f);
    void SetVY(TimeValue t, float f);
    void SetVZ(TimeValue t, float f);
    void SetMinBack(TimeValue t, float f);
    void SetMaxBack(TimeValue t, float f);
    void SetMinFront(TimeValue t, float f);
    void SetMaxFront(TimeValue t, float f);

    // Animatable methods
    void DeleteThis()
    {
        delete this;
    }
    Class_ID ClassID()
    {
        return Class_ID(Thermal_CLASS_ID1,
                        Thermal_CLASS_ID2);
    }
#if MAX_PRODUCT_VERSION_MAJOR > 23
    void GetClassName(MSTR& s, bool localized) const override { s = localized ? GetString(IDS_THERMAL_CLASS) : _T("Thermal"); }
#else

    void GetClassName(TSTR& s)
    {
        s = GetString(IDS_THERMAL_CLASS);
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
        return 2;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
};

#define PB_THERMAL_SIZE 0
#define PB_THERMAL_TURBULENCE 1
#define PB_THERMAL_MAX_BACK 2
#define PB_THERMAL_MIN_BACK 3
#define PB_THERMAL_MAX_FRONT 4
#define PB_THERMAL_MIN_FRONT 5
#define PB_THERMAL_VX 6
#define PB_THERMAL_VY 7
#define PB_THERMAL_VZ 8
#define PB_THERMAL_HEIGHT 9
#define PB_THERMAL_LENGTH 10

#endif
