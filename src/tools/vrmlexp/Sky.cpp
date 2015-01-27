/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: sky.cpp

    DESCRIPTION:  A VRML 2.0 Sky helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 28 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "sky.h"

//------------------------------------------------------

class SkyClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new SkyObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_SKY_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Sky_CLASS_ID1,
                                         Sky_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static SkyClassDesc SkyDesc;

ClassDesc *GetSkyDesc() { return &SkyDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

HWND SkyObject::hRollup = NULL;
int SkyObject::dlgPrevSel = -1;

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     SkyObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        return TRUE;
    }

    case WM_COMMAND:
    // switch(HIWORD(wParam)) {
    // }

    default:
        return FALSE;
    }

    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Color

    ParamUIDesc(
        PB_SKY_ENABLED,
        TYPE_SINGLECHEKBOX,
        IDC_ENABLED),
    ParamUIDesc(
        PB_SKY_CURRENTTIME,
        TYPE_SINGLECHEKBOX,
        IDC_CURRENTTIME),
    ParamUIDesc(
        PB_SKY_TIMELAPSE,
        TYPE_SINGLECHEKBOX,
        IDC_TIMELAPSE),
    ParamUIDesc(
        PB_SKY_YEAR,
        EDITTYPE_INT,
        IDC_YEAR_EDIT, IDC_YEAR_SPIN,
        1900, 3000,
        1),
    ParamUIDesc(
        PB_SKY_MONTH,
        EDITTYPE_INT,
        IDC_MONTH_EDIT, IDC_MONTH_SPIN,
        1, 12,
        1),
    ParamUIDesc(
        PB_SKY_DAY,
        EDITTYPE_INT,
        IDC_DAY_EDIT, IDC_DAY_SPIN,
        1, 31,
        1),
    ParamUIDesc(
        PB_SKY_HOUR,
        EDITTYPE_INT,
        IDC_HOUR_EDIT, IDC_HOUR_SPIN,
        1, 24,
        1),
    ParamUIDesc(
        PB_SKY_MINUTE,
        EDITTYPE_INT,
        IDC_MINUTE_EDIT, IDC_MINUTE_SPIN,
        1, 60,
        1),
    ParamUIDesc(
        PB_SKY_LATITUDE,
        EDITTYPE_FLOAT,
        IDC_LATITUDE_EDIT, IDC_LATITUDE_SPIN,
        -90, 90,
        SPIN_AUTOSCALE),
    ParamUIDesc(
        PB_SKY_LONGITUDE,
        EDITTYPE_FLOAT,
        IDC_LONGITUDE_EDIT, IDC_LONGITUDE_SPIN,
        -180, 180,
        SPIN_AUTOSCALE),
    ParamUIDesc(
        PB_SKY_ALTITUDE,
        EDITTYPE_UNIVERSE,
        IDC_ALTITUDE_EDIT, IDC_ALTITUDE_SPIN,
        0, 10000,
        SPIN_AUTOSCALE),
    ParamUIDesc(
        PB_SKY_RADIUS,
        EDITTYPE_UNIVERSE,
        IDC_RADIUS_EDIT, IDC_RADIUS_SPIN,
        0, 1000000,
        SPIN_AUTOSCALE),
    ParamUIDesc(
        PB_SKY_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 100000.0f,
        SPIN_AUTOSCALE)

};

#define PARAMDESC_LENGTH PB_SKY_LENGTH

static ParamBlockDescID descVer0[] = {
    { TYPE_BOOL, NULL, FALSE, 0 },
    { TYPE_BOOL, NULL, FALSE, 1 },
    { TYPE_BOOL, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_INT, NULL, FALSE, 4 },
    { TYPE_INT, NULL, FALSE, 5 },
    { TYPE_INT, NULL, FALSE, 6 },
    { TYPE_INT, NULL, FALSE, 7 },
    { TYPE_FLOAT, NULL, FALSE, 8 },
    { TYPE_FLOAT, NULL, FALSE, 9 },
    { TYPE_FLOAT, NULL, FALSE, 10 },
    { TYPE_FLOAT, NULL, FALSE, 11 },
    { TYPE_FLOAT, NULL, FALSE, 12 },
};

// Current version
// static ParamVersionDesc curVersion(descVer0, PB_SKY_LENGTH, 0);
#define CURRENT_VERSION 0

class SkyParamDlgProc : public ParamMapUserDlgProc
{
public:
    SkyObject *ob;

    SkyParamDlgProc(SkyObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR SkyParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                 UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *SkyObject::pmapParam = NULL;

void
SkyObject::BeginEditParams(IObjParam *ip, ULONG flags,
                           Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last Sky created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_SKY),
                                     _T("Sky" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new SkyParamDlgProc(this));
    }
}

void
SkyObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

SkyObject::SkyObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_SKY_LENGTH,
                                           CURRENT_VERSION);

    pb->SetValue(PB_SKY_ENABLED, 0, 1);
    pb->SetValue(PB_SKY_CURRENTTIME, 0, 1);
    pb->SetValue(PB_SKY_TIMELAPSE, 0, 0);
    pb->SetValue(PB_SKY_YEAR, 0, 2005);
    pb->SetValue(PB_SKY_MONTH, 0, 4);
    pb->SetValue(PB_SKY_DAY, 0, 9);
    pb->SetValue(PB_SKY_HOUR, 0, 13);
    pb->SetValue(PB_SKY_MINUTE, 0, 12);
    pb->SetValue(PB_SKY_LATITUDE, 0, 48.6f);
    pb->SetValue(PB_SKY_LONGITUDE, 0, 9.0008f);
    pb->SetValue(PB_SKY_ALTITUDE, 0, 300.0f);
    pb->SetValue(PB_SKY_RADIUS, 0, 8000.0f);
    pb->SetValue(PB_SKY_SIZE, 1, 1.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

SkyObject::~SkyObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *SkyObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult SkyObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                      PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult SkyObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                      PartID &partID, RefMessage message)
#endif
{
    //     int i;
    //     switch (message) {
    //     }
    return REF_SUCCEED;
}

RefTargetHandle
SkyObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    return NULL;
}

void
SkyObject::SetReference(int ind, RefTargetHandle rtarg)
{
    pblock = (IParamBlock *)rtarg;
}

ObjectState
SkyObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
SkyObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
SkyObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
SkyObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                            Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
SkyObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
SkyObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_SKY_SIZE, t, size, FOREVER);
#include "skyob.cpp"
    mesh.buildBoundingBox();
}

int
SkyObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_SKY_SIZE, t, radius, FOREVER);
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
SkyObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class SkyCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    SkyObject *skyObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(SkyObject *obj) { skyObject = obj; }
};

int
SkyCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            skyObject->pblock->SetValue(PB_SKY_SIZE,
                                        skyObject->iObjParams->GetTime(), radius);
            skyObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(skyObject->iObjParams->SnapAngle(ang));
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
static SkyCreateCallBack SkyCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
SkyObject::GetCreateMouseCallBack()
{
    SkyCreateCB.SetObj(this);
    return (&SkyCreateCB);
}

RefTargetHandle
SkyObject::Clone(RemapDir &remap)
{
    SkyObject *ni = new SkyObject();
    ni->ReplaceReference(0, pblock->Clone(remap));
    BaseClone(this, ni, remap);
    return ni;
}
