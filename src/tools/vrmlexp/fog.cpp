/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: fog.cpp

    DESCRIPTION:  A VRML 2.0 Fog helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 28 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "fog.h"

//------------------------------------------------------

class FogClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new FogObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_FOG_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Fog_CLASS_ID1,
                                         Fog_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static FogClassDesc FogDesc;

ClassDesc *GetFogDesc() { return &FogDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

HWND FogObject::hRollup = NULL;
int FogObject::dlgPrevSel = -1;

TCHAR *fogTypes[] = { _T("EXPONENTIAL"), _T("LINEAR") };

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     FogObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        HWND cb = GetDlgItem(hDlg, IDC_FOG_COMBO);
        int i;
        // gdf this prevents extra entries if the user cancel during create
        ComboBox_ResetContent(cb);
        for (i = 0; i < 2; i++)
            ComboBox_AddString(cb, fogTypes[i]);
        int type;
        th->pblock->GetValue(PB_TYPE, th->iObjParams->GetTime(),
                             type, FOREVER);
        ComboBox_SelectString(cb, 0, fogTypes[type]);
        return TRUE;
    }

    case WM_COMMAND:
        switch (HIWORD(wParam))
        {
        case LBN_SELCHANGE:
            HWND cb = GetDlgItem(hDlg, IDC_FOG_COMBO);
            int curSel = ComboBox_GetCurSel(cb);
            th->pblock->SetValue(PB_TYPE, th->iObjParams->GetTime(), curSel);
            return TRUE;
        }

    default:
        return FALSE;
    }

    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Color
    ParamUIDesc(PB_COLOR,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH),
    ParamUIDesc(
        PB_VIS_RANGE,
        EDITTYPE_UNIVERSE,
        IDC_VR_EDIT, IDC_VR_SPIN,
        0.0f, 100000.0f,
        SPIN_AUTOSCALE),

    ParamUIDesc(
        PB_FOG_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 100000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 3

static ParamBlockDescID descVer0[] = {
    { TYPE_INT, NULL, FALSE, 0 },
    { TYPE_RGBA, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 }
};

// Current version
// static ParamVersionDesc curVersion(descVer0, PB_FOG_LENGTH, 0);
#define CURRENT_VERSION 0

class FogParamDlgProc : public ParamMapUserDlgProc
{
public:
    FogObject *ob;

    FogParamDlgProc(FogObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR FogParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                 UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *FogObject::pmapParam = NULL;

void
FogObject::BeginEditParams(IObjParam *ip, ULONG flags,
                           Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last Fog created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_FOG),
                                     _T("Fog" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new FogParamDlgProc(this));
    }
}

void
FogObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

FogObject::FogObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_FOG_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_TYPE, 0, 0);
    pb->SetValue(PB_COLOR, 0, Point3(1, 1, 1));
    pb->SetValue(PB_VIS_RANGE, 0, 0.0f);
    pb->SetValue(PB_FOG_SIZE, 1, 1.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

FogObject::~FogObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *FogObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult FogObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                      PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult FogObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                      PartID &partID, RefMessage message)
#endif
{
    //     int i;
    //     switch (message) {
    //     }
    return REF_SUCCEED;
}

RefTargetHandle
FogObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    return NULL;
}

void
FogObject::SetReference(int ind, RefTargetHandle rtarg)
{
    pblock = (IParamBlock *)rtarg;
}

ObjectState
FogObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
FogObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
FogObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
FogObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                            Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
FogObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                            Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
    GetMat(t, inode, vpt, tm);

    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
}

void
FogObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_FOG_SIZE, t, size, FOREVER);
#include "fogob.cpp"
    mesh.buildBoundingBox();
}

int
FogObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_FOG_SIZE, t, radius, FOREVER);
    if (radius <= 0.0)
        return 0;
    BuildMesh(t);
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();

    DWORD rlim = gw->getRndLimits();
    gw->setRndLimits(GW_WIREFRAME | GW_EDGES_ONLY | GW_BACKCULL);
    GetMat(t, inode, vpt, m);
    gw->setTransform(m);
    if (inode->Selected())
        gw->setColor(LINE_COLOR, 1.0f, 1.0f, 1.0f);
    else if (!inode->IsFrozen())
        gw->setColor(LINE_COLOR, 1.0f, 0.0f, 0.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
FogObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
                   int flags, IPoint2 *p, ViewExp *vpt)
{
    HitRegion hitRegion;
    DWORD savedLimits;
    int res = FALSE;
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();
    MakeHitRegion(hitRegion, type, crossing, 4, p);
    gw->setRndLimits(((savedLimits = gw->getRndLimits()) | GW_PICK) & ~GW_ILLUM);
    GetMat(t, inode, vpt, m);
    gw->setTransform(m);
    gw->clearHitCode();
    if (mesh.select(gw, mtl, &hitRegion, flags & HIT_ABORTONHIT))
        return TRUE;
    gw->setRndLimits(savedLimits);
    return res;
}

class FogCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    FogObject *fogObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(FogObject *obj) { fogObject = obj; }
};

int
FogCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
                        IPoint2 m, Matrix3 &mat)
{
    Point3 p1, center;

    switch (msg)
    {
    case MOUSE_POINT:
    case MOUSE_MOVE:
        switch (point)
        {
        case 0: // only happens with MOUSE_POINT msg
            sp0 = m;
            p0 = vpt->SnapPoint(m, m, NULL, SNAP_IN_PLANE);
            mat.SetTrans(p0);
            break;
        case 1:
            mat.IdentityMatrix();
            p1 = vpt->SnapPoint(m, m, NULL, SNAP_IN_PLANE);
            mat.SetTrans(p0);
            float radius = Length(p1 - p0);
            fogObject->pblock->SetValue(PB_FOG_SIZE,
                                        fogObject->iObjParams->GetTime(), radius);
            fogObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(fogObject->iObjParams->SnapAngle(ang));
            }

            if (msg == MOUSE_POINT)
            {
                return (Length(m - sp0) < 3) ? CREATE_ABORT : CREATE_STOP;
            }
            break;
        }
        break;
    case MOUSE_ABORT:
        return CREATE_ABORT;
    }

    return TRUE;
}

// A single instance of the callback object.
static FogCreateCallBack FogCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
FogObject::GetCreateMouseCallBack()
{
    FogCreateCB.SetObj(this);
    return (&FogCreateCB);
}

RefTargetHandle
FogObject::Clone(RemapDir &remap)
{
    FogObject *ni = new FogObject();
    ni->ReplaceReference(0, pblock->Clone(remap));
    BaseClone(this, ni, remap);
    return ni;
}
