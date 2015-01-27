/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: arsensor.cpp

DESCRIPTION:  A VRML ARSensor Helper

CREATED BY: Uwe Woessner

HISTORY: created 3.4.2003

*> Copyright (c) 1996, All Rights Reserved.
**********************************************************************/

#include "vrml.h"
#include "arsensor.h"

//------------------------------------------------------

class ARSensorClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new ARSensorObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_AR_SENSOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(ARSensor_CLASS_ID1, ARSensor_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static ARSensorClassDesc ARSensorDesc;

ClassDesc *GetARSensorDesc() { return &ARSensorDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *ARSensorObject::TrackedObjectPickButton = NULL;

HWND ARSensorObject::hRollup = NULL;
int ARSensorObject::dlgPrevSel = -1;

class SensorTargetPick : public PickModeCallback
{
    ARSensorObject *parent;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetARSensor(ARSensorObject *l) { parent = l; }
};

//static SensorTargetPick    theParentPick;
#define PARENT_PICK_MODE 1
#define AR_PICK_MODE 2

static SensorTargetPick thePPick;
static int pickMode = 0;
static CommandMode *lastMode = NULL;

static void
SetPickMode(PickModeCallback *p, int w = 0)
{
    if (pickMode || !p)
    {
        pickMode = 0;
        GetCOREInterface()->PushCommandMode(lastMode);
        lastMode = NULL;
        GetCOREInterface()->ClearPickMode();
    }
    else
    {
        pickMode = w;
        lastMode = GetCOREInterface()->GetCommandMode();
        // thePick.SetARSensor(o);
        GetCOREInterface()->SetPickMode(p);
    }
}

BOOL
SensorTargetPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                          int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(ARSensor_CLASS_ID1, ARSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
SensorTargetPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
SensorTargetPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
SensorTargetPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL && parent->ReplaceReference(1, node) == REF_SUCCEED)
    {

        SetPickMode(NULL);
        parent->TrackedObjectPickButton->SetCheck(FALSE);
        HWND hw = parent->hRollup;
        Static_SetText(GetDlgItem(hw, IDC_TRIGGER_OBJ),
                       parent->triggerObject->GetName());
        return FALSE;
    }
    return FALSE;
}

HCURSOR
SensorTargetPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

class ARSensorObjPick : public PickModeCallback
{
    ARSensorObject *arSensor;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetARSensor(ARSensorObject *l) { arSensor = l; }
};

static ARSensorObjPick theARSPick;

BOOL
ARSensorObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                         int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(ARSensor_CLASS_ID1, ARSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
ARSensorObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_ARSensor_PICK_MODE));
}

void
ARSensorObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
ARSensorObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
    }
    return FALSE;
}

HCURSOR
ARSensorObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     ARSensorObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:

        th->TrackedObjectPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK_PARENT));
        th->TrackedObjectPickButton->SetType(CBT_CHECK);
        th->TrackedObjectPickButton->SetButtonDownNotify(TRUE);
        th->TrackedObjectPickButton->SetHighlightColor(GREEN_WASH);
        th->TrackedObjectPickButton->SetCheck(FALSE);

        // Now we need to fill in the list box IDC_LIST
        th->hRollup = hDlg;

        th->dlgPrevSel = -1;
        if (th->triggerObject)
            Static_SetText(GetDlgItem(hDlg, IDC_TRIGGER_OBJ),
                           th->triggerObject->GetName());

        SendMessage(GetDlgItem(hDlg, IDC_MARKER_NAME), WM_SETTEXT, 0, (LPARAM)th->MarkerName.data());
        EnableWindow(GetDlgItem(hDlg, IDC_MARKER_NAME), TRUE);
        if (pickMode)
            SetPickMode(NULL);
        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(NULL);
        ReleaseICustButton(th->TrackedObjectPickButton);
        return FALSE;

    case WM_MOUSEACTIVATE:
        return FALSE;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MOUSEMOVE:
        return FALSE;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_MARKER_NAME:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                int len = (int)SendDlgItemMessage(hDlg, IDC_MARKER_NAME, WM_GETTEXTLENGTH, 0, 0);
                TSTR temp;
                temp.Resize(len + 1);
                SendDlgItemMessage(hDlg, IDC_MARKER_NAME, WM_GETTEXT, len + 1, (LPARAM)temp.data());
                th->MarkerName = temp;
                break;
            }
            break;
        case IDC_PICK_PARENT: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                thePPick.SetARSensor(th);
                SetPickMode(&thePPick, PARENT_PICK_MODE);
                break;
            }
            break;
        }
        return FALSE;
    default:
        return FALSE;
    }
}

static ParamUIDesc descParam[] = {
    // Size
    ParamUIDesc(
        PB_AR_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Enabled
    ParamUIDesc(PB_AR_ENABLED, TYPE_SINGLECHEKBOX, IDC_ENABLE),
    // Freeze
    ParamUIDesc(PB_AR_FREEZE, TYPE_SINGLECHEKBOX, IDC_FREEZE),
    // Heading Only
    ParamUIDesc(PB_AR_HEADING_ONLY, TYPE_SINGLECHEKBOX, IDC_HEADING_ONLY),
    ParamUIDesc(PB_AR_CURRENT_CAMERA, TYPE_SINGLECHEKBOX, IDC_CURRENT_CAMERA),
    // MINX
    ParamUIDesc(
        PB_AR_MINX,
        EDITTYPE_UNIVERSE,
        IDC_MX_EDIT, IDC_MX_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MINY
    ParamUIDesc(
        PB_AR_MINY,
        EDITTYPE_UNIVERSE,
        IDC_MY_EDIT, IDC_MY_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MINZ
    ParamUIDesc(
        PB_AR_MINZ,
        EDITTYPE_UNIVERSE,
        IDC_MZ_EDIT, IDC_MZ_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MAXX
    ParamUIDesc(
        PB_AR_MAXX,
        EDITTYPE_UNIVERSE,
        IDC_MAX_EDIT, IDC_MAX_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MAXY
    ParamUIDesc(
        PB_AR_MAXY,
        EDITTYPE_UNIVERSE,
        IDC_MAY_EDIT, IDC_MAY_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MAXZ
    ParamUIDesc(
        PB_AR_MAXZ,
        EDITTYPE_UNIVERSE,
        IDC_MAZ_EDIT, IDC_MAZ_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // IPX
    ParamUIDesc(
        PB_AR_IPX,
        EDITTYPE_UNIVERSE,
        IDC_IPX_EDIT, IDC_IPX_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // IPY
    ParamUIDesc(
        PB_AR_IPY,
        EDITTYPE_UNIVERSE,
        IDC_IPY_EDIT, IDC_IPY_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // IPZ
    ParamUIDesc(
        PB_AR_IPZ,
        EDITTYPE_UNIVERSE,
        IDC_IPZ_EDIT, IDC_IPZ_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // ORI
    ParamUIDesc(
        PB_AR_ORI,
        EDITTYPE_UNIVERSE,
        IDC_ORI_EDIT, IDC_ORI_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),
    // POS_T
    ParamUIDesc(
        PB_AR_POS,
        EDITTYPE_UNIVERSE,
        IDC_POS_EDIT, IDC_POS_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 16

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_FLOAT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
    { TYPE_FLOAT, NULL, FALSE, 7 },
    { TYPE_FLOAT, NULL, FALSE, 8 },
    { TYPE_FLOAT, NULL, FALSE, 9 },
    { TYPE_FLOAT, NULL, FALSE, 10 },
    { TYPE_FLOAT, NULL, FALSE, 11 },
    { TYPE_FLOAT, NULL, FALSE, 12 },
    { TYPE_FLOAT, NULL, FALSE, 13 },
    { TYPE_FLOAT, NULL, FALSE, 14 },
    { TYPE_FLOAT, NULL, FALSE, 15 },
    { TYPE_INT, NULL, FALSE, 16 },
};

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_AR_LENGTH, CURRENT_VERSION);

class ARSensorParamDlgProc : public ParamMapUserDlgProc
{
public:
    ARSensorObject *ob;

    ARSensorParamDlgProc(ARSensorObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR ARSensorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                      UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *ARSensorObject::pmapParam = NULL;

#if 0
IOResult
ARSensorObject::Load(ILoad *iload) 
{
   iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
      NUM_OLD_VERSIONS,
      &curVersion,this,0));
   return IO_OK;
}

#endif

void
ARSensorObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {
        // Left over from last ARSensor created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_AR_SENSOR),
                                     _T("AR Sensor" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new ARSensorParamDlgProc(this));
    }
}

void
ARSensorObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

ARSensorObject::ARSensorObject()
    : HelperObject()
{
    pblock = NULL;
    previousMode = NULL;
    triggerObject = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_AR_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_AR_SIZE, 0, 0.0f);
    pb->SetValue(PB_AR_ENABLED, 0, TRUE);
    pb->SetValue(PB_AR_FREEZE, 0, TRUE);
    pb->SetValue(PB_AR_HEADING_ONLY, 0, TRUE);
    pb->SetValue(PB_AR_MINX, 0, -1.0f);
    pb->SetValue(PB_AR_MINY, 0, -1.0f);
    pb->SetValue(PB_AR_MINZ, 0, -1.0f);
    pb->SetValue(PB_AR_MAXX, 0, -1.0f);
    pb->SetValue(PB_AR_MAXY, 0, -1.0f);
    pb->SetValue(PB_AR_MAXZ, 0, -1.0f);
    pb->SetValue(PB_AR_IPX, 0, 10000.0f);
    pb->SetValue(PB_AR_IPY, 0, 10000.0f);
    pb->SetValue(PB_AR_IPZ, 0, 10000.0f);
    pb->SetValue(PB_AR_POS, 0, 0.3f);
    pb->SetValue(PB_AR_ORI, 0, 15.0f);
    pb->SetValue(PB_AR_CURRENT_CAMERA, 0, TRUE);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

ARSensorObject::~ARSensorObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *ARSensorObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult ARSensorObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                           PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult ARSensorObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                           PartID &partID, RefMessage message)
#endif
{
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        if (hTarget == triggerObject)
            triggerObject = NULL;
        break;
    case REFMSG_NODE_NAMECHANGE:
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
ARSensorObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return triggerObject;
    return NULL;
}

void
ARSensorObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        return;
    }
    if (ind == 1)
    {
        triggerObject = (INode *)rtarg;
        return;
    }
}

ObjectState
ARSensorObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
ARSensorObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
ARSensorObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
ARSensorObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                 Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
ARSensorObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                 Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
    mesh.buildBoundingBox();
    GetMat(t, inode, vpt, tm);

    BuildMesh(t);
    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
}

void
ARSensorObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_AR_SIZE, t, size, FOREVER);
#include "arsensorob.cpp"
}

int
ARSensorObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_AR_SIZE, t, radius, FOREVER);
    if (radius <= 0.0)
        return 0;
    BuildMesh(t);
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();

    DWORD rlim = gw->getRndLimits();
    gw->setRndLimits(GW_WIREFRAME | GW_EDGES_ONLY | GW_BACKCULL);
    //gw->setRndLimits(GW_BACKCULL);
    GetMat(t, inode, vpt, m);
    gw->setTransform(m);
    if (inode->Selected())
        gw->setColor(LINE_COLOR, 1.0f, 1.0f, 1.0f);
    else if (!inode->IsFrozen())
        gw->setColor(LINE_COLOR, 0.4f, 1.0f, 0.6f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
ARSensorObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class ARSensorCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    ARSensorObject *arSensorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(ARSensorObject *obj) { arSensorObject = obj; }
};

int
ARSensorCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            arSensorObject->pblock->SetValue(PB_AR_SIZE,
                                             arSensorObject->iObjParams->GetTime(), radius);
            arSensorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(arSensorObject->iObjParams->SnapAngle(ang));
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
static ARSensorCreateCallBack ARSensorCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
ARSensorObject::GetCreateMouseCallBack()
{
    ARSensorCreateCB.SetObj(this);
    return (&ARSensorCreateCB);
}

#define NAME_CHUNK 0xad30

IOResult
ARSensorObject::Save(ISave *isave)
{
    isave->BeginChunk(NAME_CHUNK);
    isave->WriteCString(MarkerName.data());
    isave->EndChunk();

    return IO_OK;
}

IOResult
ARSensorObject::Load(ILoad *iload)
{
    TCHAR *txt;

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case NAME_CHUNK:
            iload->ReadCStringChunk(&txt);
            MarkerName = txt;
            break;

        default:
            break;
        }
        iload->CloseChunk();
    }
    return IO_OK;
}

RefTargetHandle
ARSensorObject::Clone(RemapDir &remap)
{
    ARSensorObject *ts = new ARSensorObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    if (triggerObject)
    {
        if (remap.FindMapping(triggerObject))
            ts->ReplaceReference(1, remap.FindMapping(triggerObject));
        else
            ts->ReplaceReference(1, triggerObject);
    }
    ts->MarkerName = MarkerName;
    BaseClone(this, ts, remap);
    return ts;
}
