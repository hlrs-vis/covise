/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: navinfo.cpp

    DESCRIPTION:  A VRML Navigation Info Helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 19 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "navinfo.h"

//------------------------------------------------------

class NavInfoClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new NavInfoObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_NAV_INFO_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(NavInfo_CLASS_ID1,
                                         NavInfo_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static NavInfoClassDesc NavInfoDesc;

ClassDesc *GetNavInfoDesc() { return &NavInfoDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

HWND NavInfoObject::hRollup = NULL;
int NavInfoObject::dlgPrevSel = -1;

TCHAR *navTypes[] = { _T("WALK"), _T("EXAMINE"), _T("FLY"), _T("DRIVE"), _T("NONE") };

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     NavInfoObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        HWND cb = GetDlgItem(hDlg, IDC_TYPE);
        int i;
        for (i = 0; i < 5; i++)
            ComboBox_AddString(cb, navTypes[i]);
        int type;
        th->pblock->GetValue(PB_TYPE, th->iObjParams->GetTime(), type, FOREVER);
        if (type < 0 || type > 4)
            type = 0;
        ComboBox_SelectString(cb, 0, navTypes[type]);
        return TRUE;
    }

    case WM_COMMAND:
        switch (HIWORD(wParam))
        {
        case LBN_SELCHANGE:
            HWND cb = GetDlgItem(hDlg, IDC_TYPE);
            int curSel = ComboBox_GetCurSel(cb);
            if (curSel < 0 || curSel > 4)
                curSel = 0;
            th->pblock->SetValue(PB_TYPE, th->iObjParams->GetTime(), curSel);
            return TRUE;
        }

    default:
        return FALSE;
    }

    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Collision
    ParamUIDesc(
        PB_COLLISION,
        EDITTYPE_UNIVERSE,
        IDC_COLLISION_EDIT, IDC_COLLISION_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Size
    ParamUIDesc(
        PB_NI_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Terrain
    ParamUIDesc(
        PB_TERRAIN,
        EDITTYPE_UNIVERSE,
        IDC_TERRAIN_EDIT, IDC_TERRAIN_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Step Height
    ParamUIDesc(
        PB_STEP,
        EDITTYPE_UNIVERSE,
        IDC_STEP_EDIT, IDC_STEP_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Loop
    ParamUIDesc(PB_HEADLIGHT, TYPE_SINGLECHEKBOX, IDC_HEADLIGHT),

    // Speed
    ParamUIDesc(
        PB_SPEED,
        EDITTYPE_UNIVERSE,
        IDC_SPEED_EDIT, IDC_SPEED_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Visibilit Limit
    ParamUIDesc(
        PB_VIS_LIMIT,
        EDITTYPE_UNIVERSE,
        IDC_VIS_LIMIT_EDIT, IDC_VIS_LIMIT_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // SCALE (COVER Extension)
    ParamUIDesc(
        PB_NI_SCALE,
        EDITTYPE_UNIVERSE,
        IDC_SCALE_EDIT, IDC_SCALE_SPIN,
        0.0f, 1000000.0f,
        SPIN_AUTOSCALE),

    // near
    ParamUIDesc(
        PB_NI_NEAR,
        EDITTYPE_UNIVERSE,
        IDC_NEAR_EDIT, IDC_NEAR_SPIN,
        0.0f, 100000000.0f,
        SPIN_AUTOSCALE),
    // far
    ParamUIDesc(
        PB_NI_FAR,
        EDITTYPE_UNIVERSE,
        IDC_FAR_EDIT, IDC_FAR_SPIN,
        0.0f, 100000000.0f,
        SPIN_AUTOSCALE),
};

#define PARAMDESC_LENGTH 10

static ParamBlockDescID descVer0[] = {
    { TYPE_INT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 },
    { TYPE_FLOAT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
};

static ParamBlockDescID descVer1[] = {
    { TYPE_INT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 },
    { TYPE_FLOAT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
    { TYPE_FLOAT, NULL, FALSE, 7 },
    { TYPE_FLOAT, NULL, FALSE, 8 },
};

static ParamBlockDescID descVer2[] = {
    { TYPE_INT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 },
    { TYPE_FLOAT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
    { TYPE_FLOAT, NULL, FALSE, 7 },
    { TYPE_FLOAT, NULL, FALSE, 8 },
    { TYPE_FLOAT, NULL, FALSE, 9 },
    { TYPE_FLOAT, NULL, FALSE, 10 },
};

#define NUM_OLD_VERSIONS 2

static ParamVersionDesc versions[] = {
    ParamVersionDesc(descVer0, 7, 0),
    ParamVersionDesc(descVer1, 9, 1),
};

// Current version
#define CURRENT_VERSION 2
static ParamVersionDesc curVersion(descVer2, PB_NA_LENGTH, CURRENT_VERSION);

class NavInfoParamDlgProc : public ParamMapUserDlgProc
{
public:
    NavInfoObject *ob;

    NavInfoParamDlgProc(NavInfoObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR NavInfoParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                     UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *NavInfoObject::pmapParam = NULL;

IOResult
NavInfoObject::Load(ILoad *iload)
{
    iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                       NUM_OLD_VERSIONS,
                                                       &curVersion, this, 0));
    return IO_OK;
}

void
NavInfoObject::BeginEditParams(IObjParam *ip, ULONG flags,
                               Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last NavInfo created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_NAV_INFO),
                                     _T("NavigationInfo" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new NavInfoParamDlgProc(this));
    }
}

void
NavInfoObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

NavInfoObject::NavInfoObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer2, PB_NA_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_TYPE, 0, 0);
    pb->SetValue(PB_HEADLIGHT, 0, TRUE);
    pb->SetValue(PB_VIS_LIMIT, 0, 0.0f);
    pb->SetValue(PB_SPEED, 0, 1.0f);
    pb->SetValue(PB_COLLISION, 0, 0.25f);
    pb->SetValue(PB_TERRAIN, 0, 1.6f);
    pb->SetValue(PB_STEP, 0, 0.75f);
    pb->SetValue(PB_NI_SCALE, 0, 0.0f);
    pb->SetValue(PB_NI_NEAR, 0, 0.0f);
    pb->SetValue(PB_NI_FAR, 0, 0.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

NavInfoObject::~NavInfoObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *NavInfoObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult NavInfoObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                          PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult NavInfoObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                          PartID &partID, RefMessage message)
#endif
{
    //     int i;
    //     switch (message) {
    //     }
    return REF_SUCCEED;
}

RefTargetHandle
NavInfoObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    return NULL;
}

void
NavInfoObject::SetReference(int ind, RefTargetHandle rtarg)
{
    pblock = (IParamBlock *)rtarg;
}

ObjectState
NavInfoObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
NavInfoObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
NavInfoObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
NavInfoObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                Box3 &box)
{
    BuildMesh(t);
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
NavInfoObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
    GetMat(t, inode, vpt, tm);

    BuildMesh(t);
    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
}

void
NavInfoObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_NI_SIZE, t, size, FOREVER);
#include "niob.cpp"
    mesh.buildBoundingBox();
}

int
NavInfoObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_NI_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 0.0f, 0.0f, 1.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
NavInfoObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class NavInfoCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    NavInfoObject *navInfoObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(NavInfoObject *obj) { navInfoObject = obj; }
};

int
NavInfoCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            navInfoObject->pblock->SetValue(PB_NI_SIZE,
                                            navInfoObject->iObjParams->GetTime(), radius);
            navInfoObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(navInfoObject->iObjParams->SnapAngle(ang));
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
static NavInfoCreateCallBack NavInfoCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
NavInfoObject::GetCreateMouseCallBack()
{
    NavInfoCreateCB.SetObj(this);
    return (&NavInfoCreateCB);
}

RefTargetHandle
NavInfoObject::Clone(RemapDir &remap)
{
    NavInfoObject *ni = new NavInfoObject();
    ni->ReplaceReference(0, pblock->Clone(remap));
    BaseClone(this, ni, remap);
    return ni;
}
