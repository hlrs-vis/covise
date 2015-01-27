/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: MultiTouchSensor.cpp

DESCRIPTION:  A VRML MultiTouchSensor Helper

CREATED BY: Uwe Woessner

HISTORY: created 3.4.2003

*> Copyright (c) 1996, All Rights Reserved.
**********************************************************************/

#include "vrml.h"
#include "MultiTouchSensor.h"

//------------------------------------------------------

class MultiTouchSensorClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new MultiTouchSensorObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_MT_SENSOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(MultiTouchSensor_CLASS_ID1, MultiTouchSensor_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static MultiTouchSensorClassDesc MultiTouchSensorDesc;

ClassDesc *GetMultiTouchSensorDesc() { return &MultiTouchSensorDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *MultiTouchSensorObject::TrackedObjectPickButton = NULL;

HWND MultiTouchSensorObject::hRollup = NULL;
int MultiTouchSensorObject::dlgPrevSel = -1;

bool CheckNodeSelection(Interface *ip, INode *inode)
{
    if (!ip)
        return FALSE;
    if (!inode)
        return FALSE;
    int i, nct = ip->GetSelNodeCount();
    for (i = 0; i < nct; i++)
        if (ip->GetSelNode(i) == inode)
            return TRUE;
    return FALSE;
}

class MTSensorTargetPick : public PickModeCallback
{
    MultiTouchSensorObject *parent;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetMultiTouchSensor(MultiTouchSensorObject *l) { parent = l; }
};

//static MTSensorTargetPick    theParentPick;
#define PARENT_PICK_MODE 1
#define AR_PICK_MODE 2

static MTSensorTargetPick thePPick;
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
        // thePick.SetMultiTouchSensor(o);
        GetCOREInterface()->SetPickMode(p);
    }
}

BOOL
MTSensorTargetPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                            int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(MultiTouchSensor_CLASS_ID1, MultiTouchSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
MTSensorTargetPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
MTSensorTargetPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
MTSensorTargetPick::Pick(IObjParam *ip, ViewExp *vpt)
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
MTSensorTargetPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

class MultiTouchSensorObjPick : public PickModeCallback
{
    MultiTouchSensorObject *MultiTouchSensor;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetMultiTouchSensor(MultiTouchSensorObject *l) { MultiTouchSensor = l; }
};

static MultiTouchSensorObjPick theARSPick;

BOOL
MultiTouchSensorObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                                 int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(MultiTouchSensor_CLASS_ID1, MultiTouchSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
MultiTouchSensorObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_MultiTouchSensor_PICK_MODE));
}

void
MultiTouchSensorObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
MultiTouchSensorObjPick::Pick(IObjParam *ip, ViewExp *vpt)
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
MultiTouchSensorObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     MultiTouchSensorObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;
    switch (message)
    {
    case WM_INITDIALOG:

        ICustButton *but;
        but = GetICustButton(GetDlgItem(hDlg, IDC_SURFACE_MODE));
        but->SetType(CBT_CHECK);
        but->SetCheck(th->surfaceMode);
        but->SetHighlightColor(GREEN_WASH);
        ReleaseICustButton(but);

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
        case IDC_SURFACE_MODE:
            if (th->surfaceMode)
                th->exitSurfaceMode();
            else
                th->enterSurfaceMode();
            break;
        case IDC_PICK_PARENT: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                thePPick.SetMultiTouchSensor(th);
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
        PB_MT_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Enabled
    ParamUIDesc(PB_MT_ENABLED, TYPE_SINGLECHEKBOX, IDC_ENABLE),
    // Freeze
    ParamUIDesc(PB_MT_FREEZE, TYPE_SINGLECHEKBOX, IDC_FREEZE),
    // MINX
    ParamUIDesc(
        PB_MT_MINX,
        EDITTYPE_UNIVERSE,
        IDC_MX_EDIT, IDC_MX_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MINY
    ParamUIDesc(
        PB_MT_MINY,
        EDITTYPE_UNIVERSE,
        IDC_MY_EDIT, IDC_MY_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // MINZ
    ParamUIDesc(
        PB_MT_MINZ,
        EDITTYPE_UNIVERSE,
        IDC_MZ_EDIT, IDC_MZ_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // SIZEX
    ParamUIDesc(
        PB_MT_SIZEX,
        EDITTYPE_UNIVERSE,
        IDC_MAX_EDIT, IDC_MAX_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // SIZEY
    ParamUIDesc(
        PB_MT_SIZEY,
        EDITTYPE_UNIVERSE,
        IDC_MAY_EDIT, IDC_MAY_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // SIZEZ
    ParamUIDesc(
        PB_MT_SIZEZ,
        EDITTYPE_UNIVERSE,
        IDC_MAZ_EDIT, IDC_MAZ_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // ORIH
    ParamUIDesc(
        PB_MT_ORIH,
        EDITTYPE_UNIVERSE,
        IDC_H_EDIT, IDC_H_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // ORIP
    ParamUIDesc(
        PB_MT_ORIP,
        EDITTYPE_UNIVERSE,
        IDC_P_EDIT, IDC_P_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // ORIR
    ParamUIDesc(
        PB_MT_ORIR,
        EDITTYPE_UNIVERSE,
        IDC_R_EDIT, IDC_R_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // IPX
    ParamUIDesc(
        PB_MT_IPX,
        EDITTYPE_UNIVERSE,
        IDC_IPX_EDIT, IDC_IPX_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // IPY
    ParamUIDesc(
        PB_MT_IPY,
        EDITTYPE_UNIVERSE,
        IDC_IPY_EDIT, IDC_IPY_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // IPZ
    ParamUIDesc(
        PB_MT_IPZ,
        EDITTYPE_UNIVERSE,
        IDC_IPZ_EDIT, IDC_IPZ_SPIN,
        -100000.0f, 100000.0f,
        SPIN_AUTOSCALE),
    // ORI
    ParamUIDesc(
        PB_MT_ORI,
        EDITTYPE_UNIVERSE,
        IDC_ORI_EDIT, IDC_ORI_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),
    // POS_T
    ParamUIDesc(
        PB_MT_POS,
        EDITTYPE_UNIVERSE,
        IDC_POS_EDIT, IDC_POS_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH PB_MT_LENGTH

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
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
    { TYPE_FLOAT, NULL, FALSE, 15 },
    { TYPE_FLOAT, NULL, FALSE, 16 },
};

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_MT_LENGTH, CURRENT_VERSION);

class MultiTouchSensorParamDlgProc : public ParamMapUserDlgProc
{
public:
    MultiTouchSensorObject *ob;

    MultiTouchSensorParamDlgProc(MultiTouchSensorObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR MultiTouchSensorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                              UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

void MultiTouchSensorObject::enterSurfaceMode()
{
    surfaceMode = true;
}

void MultiTouchSensorObject::exitSurfaceMode()
{
    surfaceMode = false;
}

IParamMap *MultiTouchSensorObject::pmapParam = NULL;

#if 0
IOResult
MultiTouchSensorObject::Load(ILoad *iload) 
{
   iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
      NUM_OLD_VERSIONS,
      &curVersion,this,0));
   return IO_OK;
}

#endif

void
MultiTouchSensorObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                        Animatable *prev)
{
    iObjParams = ip;

    // Create sub object editing modes.

    moveMode = new MoveModBoxCMode(this, ip);

    rotMode = new RotateModBoxCMode(this, ip);

    uscaleMode = new UScaleModBoxCMode(this, ip);

    if (pmapParam)
    {
        // Left over from last MultiTouchSensor created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_MT_SENSOR),
                                     _T("MT Sensor" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new MultiTouchSensorParamDlgProc(this));
    }
}

void MultiTouchSensorObject::ActivateSubobjSel(int level, XFormModes &modes)
{
    modes = XFormModes(moveMode, rotMode, uscaleMode, NULL, NULL, NULL);
    if (level == 0)
    {
        exitSurfaceMode();
    }
    else
    {
        enterSurfaceMode();
    }
}

void
MultiTouchSensorObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    ip->DeleteMode(moveMode);
    ip->DeleteMode(rotMode);
    ip->DeleteMode(uscaleMode);
    if (moveMode)
        delete moveMode;
    moveMode = NULL;
    if (rotMode)
        delete rotMode;
    rotMode = NULL;
    if (uscaleMode)
        delete uscaleMode;
    uscaleMode = NULL;
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

MultiTouchSensorObject::MultiTouchSensorObject()
    : HelperObject()
{
    pblock = NULL;
    previousMode = NULL;
    triggerObject = NULL;
    surfaceMode = false;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_MT_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_MT_SIZE, 0, 0.0f);
    pb->SetValue(PB_MT_ENABLED, 0, TRUE);
    pb->SetValue(PB_MT_FREEZE, 0, TRUE);
    pb->SetValue(PB_MT_MINX, 0, -1.0f);
    pb->SetValue(PB_MT_MINY, 0, -1.0f);
    pb->SetValue(PB_MT_MINZ, 0, -1.0f);
    pb->SetValue(PB_MT_SIZEX, 0, 40.0f);
    pb->SetValue(PB_MT_SIZEY, 0, 30.0f);
    pb->SetValue(PB_MT_SIZEZ, 0, 1.0f);
    pb->SetValue(PB_MT_ORIH, 0, 0.0f);
    pb->SetValue(PB_MT_ORIP, 0, 0.0f);
    pb->SetValue(PB_MT_ORIR, 0, 0.0f);
    pb->SetValue(PB_MT_IPX, 0, 10000.0f);
    pb->SetValue(PB_MT_IPY, 0, 10000.0f);
    pb->SetValue(PB_MT_IPZ, 0, 10000.0f);
    pb->SetValue(PB_MT_POS, 0, 0.3f);
    pb->SetValue(PB_MT_ORI, 0, 15.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

MultiTouchSensorObject::~MultiTouchSensorObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *MultiTouchSensorObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult MultiTouchSensorObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                                   PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult MultiTouchSensorObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
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
MultiTouchSensorObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return triggerObject;
    return NULL;
}

void
MultiTouchSensorObject::SetReference(int ind, RefTargetHandle rtarg)
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
MultiTouchSensorObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
MultiTouchSensorObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
MultiTouchSensorObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
MultiTouchSensorObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                         Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    //box = mesh.getBoundingBox();
    Point3 c(0, 0, 0);
    box.MakeCube(c, 1.3f * size);
}

void
MultiTouchSensorObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                         Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
    mesh.buildBoundingBox();
    GetMat(t, inode, vpt, tm);

    Point3 c(0, 0, 0);
    box.MakeCube(tm * c, 1.3f * size);

    if (!iObjParams)
        return;
    if (surfaceMode && CheckNodeSelection(iObjParams, inode))
    {

        Matrix3 rotMatrix;
        surfaceRot.MakeMatrix(rotMatrix);
        rotMatrix.SetTrans(surfaceMin);
        float sizex = 1, sizey = 1, sizez = 1;
        pblock->GetValue(PB_MT_SIZEX, t, sizex, FOREVER);
        pblock->GetValue(PB_MT_SIZEY, t, sizey, FOREVER);
        pblock->GetValue(PB_MT_SIZEZ, t, sizez, FOREVER);

        box += Point3(0.0f, 0.0f, 0.0f) * rotMatrix;
        box += Point3(sizex, 0.0f, 0.0f) * rotMatrix;
        box += Point3(sizex, sizey, 0.0f) * rotMatrix;
        box += Point3(0.0f, sizey, 0.0f) * rotMatrix;
    }
}

Quat MultiTouchSensorObject::surfaceRot(0.0f, 0.0f, 0.0f, 1.0f);
Point3 MultiTouchSensorObject::surfaceMin(0.0f, 0.0f, 0.0f);
float MultiTouchSensorObject::surfaceSize = 1.0f;

void
MultiTouchSensorObject::BuildMesh(TimeValue t)
{
    pblock->GetValue(PB_MT_SIZE, t, size, FOREVER);
#include "MultiTouchSensorIcon.cpp"
}

int
MultiTouchSensorObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_MT_SIZE, t, radius, FOREVER);
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
    m.IdentityMatrix();
    gw->setTransform(m);
    if (!iObjParams)
        return 0;
    if (surfaceMode && CheckNodeSelection(iObjParams, inode))
    {
        // Draw rectangle representing slice plane.
        gw->setColor(LINE_COLOR, GetUIColor(COLOR_SEL_GIZMOS));

        Point3 rp[5];
        Matrix3 rotMatrix;

        float sizex = 1, sizey = 1, sizez = 1;
        pblock->GetValue(PB_MT_MINX, t, sizex, FOREVER);
        pblock->GetValue(PB_MT_MINY, t, sizey, FOREVER);
        pblock->GetValue(PB_MT_MINZ, t, sizez, FOREVER);
        surfaceMin.Set(sizex, sizey, sizez);
        pblock->GetValue(PB_MT_ORIH, t, sizex, FOREVER);
        pblock->GetValue(PB_MT_ORIP, t, sizey, FOREVER);
        pblock->GetValue(PB_MT_ORIR, t, sizez, FOREVER);
        surfaceRot.SetEuler((float)((sizex / 180.0) * PI), (float)((sizey / 180.0) * PI), (float)((sizez / 180.0) * PI));
        surfaceRot.MakeMatrix(rotMatrix);
        rotMatrix.SetTrans(surfaceMin);
        pblock->GetValue(PB_MT_SIZEX, t, sizex, FOREVER);
        pblock->GetValue(PB_MT_SIZEY, t, sizey, FOREVER);
        pblock->GetValue(PB_MT_SIZEZ, t, sizez, FOREVER);

        rp[0] = Point3(0.0f, 0.0f, 0.0f) * rotMatrix;
        rp[1] = Point3(sizex, 0.0f, 0.0f) * rotMatrix;
        rp[2] = Point3(sizex, sizey, 0.0f) * rotMatrix;
        rp[3] = Point3(0.0f, sizey, 0.0f) * rotMatrix;
        gw->polyline(4, rp, NULL, NULL, TRUE, NULL);
    }
    return (0);
}

static GenSubObjType SOT_Surface(1);

int MultiTouchSensorObject::NumSubObjTypes()
{
    return 1;
}

ISubObjType *MultiTouchSensorObject::GetSubObjType(int i)
{

    static bool initialized = false;
    if (!initialized)
    {
        initialized = true;
        SOT_Surface.SetName(GetString(IDS_MT_SURFACE));
    }

    switch (i)
    {
    case -1:
        if (GetSubObjectLevel() > 0)
            return GetSubObjType(GetSubObjectLevel() - 1);
        break;
    case 0:
        return &SOT_Surface;
    }
    return NULL;
}

void MultiTouchSensorObject::GetSubObjectCenters(SubObjAxisCallback *cb, TimeValue t, INode *node, ModContext *mc)
{
    Matrix3 tm = node->GetObjectTM(t);

    if (surfaceMode)
    {

        float minx = 1, miny = 1, minz = 1;
        pblock->GetValue(PB_MT_MINX, t, minx, FOREVER);
        pblock->GetValue(PB_MT_MINY, t, miny, FOREVER);
        pblock->GetValue(PB_MT_MINZ, t, minz, FOREVER);
        cb->Center(Point3(minx, miny, minz), 0);
        return;
    }
    cb->Center(tm.GetTrans(), 0);
}

void MultiTouchSensorObject::GetSubObjectTMs(SubObjAxisCallback *cb,
                                             TimeValue t, INode *node, ModContext *mc)
{
    if (surfaceMode)
    {
        Matrix3 rotMatrix;
        rotMatrix.IdentityMatrix();
        float sizex = 1, sizey = 1, sizez = 1;
        pblock->GetValue(PB_MT_ORIH, t, sizex, FOREVER);
        pblock->GetValue(PB_MT_ORIP, t, sizey, FOREVER);
        pblock->GetValue(PB_MT_ORIR, t, sizez, FOREVER);
        surfaceRot.SetEuler((float)(sizex / 180.0 * PI), (float)(sizey / 180.0 * PI), (float)(sizez / 180.0 * PI));
        surfaceRot.MakeMatrix(rotMatrix);
        pblock->GetValue(PB_MT_SIZEX, t, sizex, FOREVER);
        pblock->GetValue(PB_MT_SIZEY, t, sizey, FOREVER);
        pblock->GetValue(PB_MT_SIZEZ, t, sizez, FOREVER);

        cb->TM(rotMatrix, 0);
        return;
    }

    Matrix3 tm = node->GetObjectTM(t);
    cb->TM(tm, 0);
}

void MultiTouchSensorObject::Transform(TimeValue t, Matrix3 &partm, Matrix3 tmAxis,
                                       BOOL localOrigin, Matrix3 xfrm, int type)
{
    if (!iObjParams)
        return;

    if (surfaceMode)
    {
        // Special case -- just transform slicing plane.
        theHold.Put(new TransformPlaneRestore(this));
        Matrix3 tm = partm * Inverse(tmAxis);
        Matrix3 itm = Inverse(tm);
        Matrix3 myxfm = tm * xfrm * itm;
        Point3 myTrans, myScale;
        Quat myRot;
        DecomposeMatrix(myxfm, myTrans, myRot, myScale);
        float factor;
        switch (type)
        {
        case 0:
            surfaceMin += myTrans;
            pblock->SetValue(PB_MT_MINX, t, surfaceMin[0]);
            pblock->SetValue(PB_MT_MINY, t, surfaceMin[1]);
            pblock->SetValue(PB_MT_MINZ, t, surfaceMin[2]);
            break;
        case 1:
            surfaceRot *= myRot;
            float h, p, r;
            surfaceRot.GetEuler(&h, &p, &r);
            pblock->SetValue(PB_MT_ORIH, t, (float)(h * 180.0 / PI));
            pblock->SetValue(PB_MT_ORIP, t, (float)(p * 180.0 / PI));
            pblock->SetValue(PB_MT_ORIR, t, (float)(r * 180.0 / PI));
            break;
        case 2:
            factor = (float)exp(log(myScale[0] * myScale[1] * myScale[2]) / 3.0);
            surfaceSize *= factor;
            pblock->SetValue(PB_MT_SIZEX, t, surfaceSize * 40);
            pblock->SetValue(PB_MT_SIZEY, t, surfaceSize * 30);
            break;
        }

        pmapParam->Invalidate();
        NotifyDependents(FOREVER, PART_DISPLAY, REFMSG_CHANGE);
        iObjParams->RedrawViews(iObjParams->GetTime());
        return;
    }
}

void MultiTouchSensorObject::Move(TimeValue t, Matrix3 &partm, Matrix3 &tmAxis, Point3 &val, BOOL localOrigin)
{
    Transform(t, partm, tmAxis, localOrigin, TransMatrix(val), 0);
}

void MultiTouchSensorObject::Rotate(TimeValue t, Matrix3 &partm, Matrix3 &tmAxis, Quat &val, BOOL localOrigin)
{
    Matrix3 mat;
    val.MakeMatrix(mat);
    Transform(t, partm, tmAxis, localOrigin, mat, 1);
}

void MultiTouchSensorObject::Scale(TimeValue t, Matrix3 &partm, Matrix3 &tmAxis, Point3 &val, BOOL localOrigin)
{
    Transform(t, partm, tmAxis, localOrigin, ScaleMatrix(val), 2);
}

void MultiTouchSensorObject::TransformStart(TimeValue t)
{
    if (!iObjParams)
        return;
    iObjParams->LockAxisTripods(TRUE);
}

void MultiTouchSensorObject::TransformHoldingFinish(TimeValue t)
{
    if (!iObjParams)
        return;
}

void MultiTouchSensorObject::TransformFinish(TimeValue t)
{
    if (!iObjParams)
        return;
    iObjParams->LockAxisTripods(FALSE);
}

void MultiTouchSensorObject::TransformCancel(TimeValue t)
{
    if (!iObjParams)
        return;
    iObjParams->LockAxisTripods(FALSE);
}

int
MultiTouchSensorObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class MultiTouchSensorCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    MultiTouchSensorObject *MTSO;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(MultiTouchSensorObject *obj) { MTSO = obj; }
};

int
MultiTouchSensorCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            MTSO->pblock->SetValue(PB_MT_SIZE,
                                   MTSO->iObjParams->GetTime(), radius);
            MTSO->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(MTSO->iObjParams->SnapAngle(ang));
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
static MultiTouchSensorCreateCallBack MultiTouchSensorCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
MultiTouchSensorObject::GetCreateMouseCallBack()
{
    MultiTouchSensorCreateCB.SetObj(this);
    return (&MultiTouchSensorCreateCB);
}

#define NAME_CHUNK 0xad30

IOResult
MultiTouchSensorObject::Save(ISave *isave)
{
    isave->BeginChunk(NAME_CHUNK);
    isave->WriteCString(MarkerName.data());
    isave->EndChunk();

    return IO_OK;
}

IOResult
MultiTouchSensorObject::Load(ILoad *iload)
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
MultiTouchSensorObject::Clone(RemapDir &remap)
{
    MultiTouchSensorObject *ts = new MultiTouchSensorObject();
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

TransformPlaneRestore::TransformPlaneRestore(MultiTouchSensorObject *emm)
{
    em = emm;
    oldSurfaceMin = em->surfaceMin;
    oldSurfaceRot = em->surfaceRot;
    oldSurfaceSize = em->surfaceSize;
}

void TransformPlaneRestore::Restore(int isUndo)
{
    newSurfaceMin = em->surfaceMin;
    newSurfaceRot = em->surfaceRot;
    newSurfaceSize = em->surfaceSize;
    em->surfaceMin = oldSurfaceMin;
    em->surfaceRot = oldSurfaceRot;
    em->surfaceSize = oldSurfaceSize;
    em->NotifyDependents(FOREVER, PART_DISPLAY, REFMSG_CHANGE);
}

void TransformPlaneRestore::Redo()
{
    em->surfaceMin = newSurfaceMin;
    em->surfaceRot = newSurfaceRot;
    em->surfaceSize = newSurfaceSize;
    em->NotifyDependents(FOREVER, PART_DISPLAY, REFMSG_CHANGE);
}
