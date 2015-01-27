/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: anchor.cpp

    DESCRIPTION:  A VRML Anchor helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 17 Sept, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "anchor.h"
#include "bookmark.h"

//------------------------------------------------------

class AnchorClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new AnchorObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_ANCHOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Anchor_CLASS_ID1,
                                         Anchor_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static AnchorClassDesc AnchorDesc;

ClassDesc *GetAnchorDesc() { return &AnchorDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *AnchorObject::ParentPickButton = NULL;

HWND AnchorObject::hRollup = NULL;
int AnchorObject::dlgPrevSel = -1;

class TriggerPick : public PickModeCallback
{
    AnchorObject *parent;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetAnchor(AnchorObject *l) { parent = l; }
};

//static TriggerPick theParentPick;
static TriggerPick thePick;
static BOOL pickMode = FALSE;
static CommandMode *lastMode = NULL;

static void
SetPickMode(AnchorObject *o)
{
    if (pickMode || !o)
    {
        pickMode = FALSE;
        GetCOREInterface()->PushCommandMode(lastMode);
        lastMode = NULL;
        GetCOREInterface()->ClearPickMode();
    }
    else
    {
        pickMode = TRUE;
        lastMode = GetCOREInterface()->GetCommandMode();
        thePick.SetAnchor(o);
        GetCOREInterface()->SetPickMode(&thePick);
    }
}

BOOL
TriggerPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                     int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(Anchor_CLASS_ID1, Anchor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
TriggerPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
TriggerPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
TriggerPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        RefResult ret = parent->ReplaceReference(1, node);

        SetPickMode(NULL);
        //parent->iObjParams->SetCommandMode(parent->previousMode);
        //parent->previousMode = NULL;
        parent->ParentPickButton->SetCheck(FALSE);
        HWND hw = parent->hRollup;
        Static_SetText(GetDlgItem(hw, IDC_TRIGGER_OBJ),
                       parent->triggerObject->GetName());
        return FALSE;
    }
    return FALSE;
}

HCURSOR
TriggerPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

static void
GetCameras(INode *inode, Tab<INode *> *list)
{
    const ObjectState &os = inode->EvalWorldState(0);
    Object *ob = os.obj;
    if (ob != NULL)
    {
        if (ob->SuperClassID() == CAMERA_CLASS_ID)
        {
            list->Append(1, &inode);
        }
    }
    int count = inode->NumberOfChildren();
    for (int i = 0; i < count; i++)
        GetCameras(inode->GetChildNode(i), list);
}

BOOL CALLBACK
    AnchorDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                  AnchorObject *th)
{
    TCHAR text[MAX_PATH];
    int c, camIndex, i, type;
    HWND cb;
    Tab<INode *> cameras;

    switch (message)
    {
    case WM_INITDIALOG:

        th->ParentPickButton = GetICustButton(GetDlgItem(hDlg,
                                                         IDC_PICK_PARENT));
        th->ParentPickButton->SetType(CBT_CHECK);
        th->ParentPickButton->SetButtonDownNotify(TRUE);
        th->ParentPickButton->SetHighlightColor(GREEN_WASH);
        th->ParentPickButton->SetCheck(FALSE);

        th->dlgPrevSel = -1;
        th->hRollup = hDlg;
        if (th->triggerObject)
            Static_SetText(GetDlgItem(hDlg, IDC_TRIGGER_OBJ),
                           th->triggerObject->GetName());
        th->pblock->GetValue(PB_AN_TYPE, th->iObjParams->GetTime(),
                             type, FOREVER);
        th->isJump = type == 0;
        EnableWindow(GetDlgItem(hDlg, IDC_ANCHOR_URL), th->isJump);
        EnableWindow(GetDlgItem(hDlg, IDC_PARAMETER), th->isJump);
        EnableWindow(GetDlgItem(hDlg, IDC_BOOKMARKS), th->isJump);
        EnableWindow(GetDlgItem(hDlg, IDC_CAMERA), !th->isJump);
        GetCameras(th->iObjParams->GetRootNode(), &cameras);
        c = cameras.Count();
        cb = GetDlgItem(hDlg, IDC_CAMERA);
        camIndex = -1;
        for (i = 0; i < c; i++)
        {
            // add the name to the list
            TSTR name = cameras[i]->GetName();
            int ind = ComboBox_AddString(cb, name.data());
            ComboBox_SetItemData(cb, ind, cameras[i]);
            if (cameras[i] == th->cameraObject)
                camIndex = i;
        }
        if (camIndex != -1)
            ComboBox_SelectString(cb, 0, cameras[camIndex]->GetName());

        Edit_SetText(GetDlgItem(hDlg, IDC_DESC), th->description.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_ANCHOR_URL), th->URL.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_PARAMETER), th->parameter.data());

        if (pickMode)
            SetPickMode(th);

        return TRUE;

    case WM_DESTROY:

        if (pickMode)
            SetPickMode(th);
        //th->iObjParams->ClearPickMode();
        //th->previousMode = NULL;
        ReleaseICustButton(th->ParentPickButton);
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
        case IDC_BOOKMARKS:
        {
            // do bookmarks
            TSTR url, cam, desc;
            if (GetBookmarkURL(th->iObjParams, &url, &cam, &desc))
            {
                // get the new URL information;
                Edit_SetText(GetDlgItem(hDlg, IDC_ANCHOR_URL), url.data());
                Edit_SetText(GetDlgItem(hDlg, IDC_DESC), desc.data());
            }
        }
        break;
        case IDC_CAMERA:
            if (HIWORD(wParam) == CBN_SELCHANGE)
            {
                cb = GetDlgItem(hDlg, IDC_CAMERA);
                int sel = ComboBox_GetCurSel(cb);
                INode *rtarg;
                rtarg = (INode *)ComboBox_GetItemData(cb, sel);
                th->ReplaceReference(2, rtarg);
            }
            break;
        case IDC_HYPERLINK:
            th->isJump = IsDlgButtonChecked(hDlg, IDC_HYPERLINK);
            EnableWindow(GetDlgItem(hDlg, IDC_ANCHOR_URL), th->isJump);
            EnableWindow(GetDlgItem(hDlg, IDC_PARAMETER), th->isJump);
            EnableWindow(GetDlgItem(hDlg, IDC_BOOKMARKS), th->isJump);
            EnableWindow(GetDlgItem(hDlg, IDC_CAMERA), !th->isJump);
            break;
        case IDC_SET_CAMERA:
            th->isJump = !IsDlgButtonChecked(hDlg, IDC_SET_CAMERA);
            EnableWindow(GetDlgItem(hDlg, IDC_ANCHOR_URL), th->isJump);
            EnableWindow(GetDlgItem(hDlg, IDC_PARAMETER), th->isJump);
            EnableWindow(GetDlgItem(hDlg, IDC_BOOKMARKS), th->isJump);
            EnableWindow(GetDlgItem(hDlg, IDC_CAMERA), !th->isJump);
            break;
        case IDC_PICK_PARENT: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                SetPickMode(th);
                /*
                if (th->previousMode) {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                } else {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    theParentPick.SetAnchor(th);
                    th->iObjParams->SetPickMode(&theParentPick);
                }
                */
                break;
            }
            return TRUE;
        case IDC_ANCHOR_URL:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_ANCHOR_URL),
                             text, MAX_PATH);
                th->URL = text;
            }
            break;
        case IDC_DESC:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_DESC),
                             text, MAX_PATH);
                th->description = text;
            }
            break;
        case IDC_PARAMETER:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_PARAMETER),
                             text, MAX_PATH);
                th->parameter = text;
            }
            break;
        default:
            return FALSE;
        }
    }
    return FALSE;
}

static int buttonIds[] = { IDC_HYPERLINK, IDC_SET_CAMERA };

static ParamUIDesc descParam[] = {
    // Size
    ParamUIDesc(
        PB_AN_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Type
    ParamUIDesc(PB_AN_TYPE, TYPE_RADIO, buttonIds, 2),
};

#define PARAMDESC_LENGTH 2

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
};

//static ParamVersionDesc versions[] = {
//  ParamVersionDesc(descVer0,5,0),
//};

//#define NUM_OLD_VERSIONS 1

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_AN_LENGTH, CURRENT_VERSION);

class AnchorParamDlgProc : public ParamMapUserDlgProc
{
public:
    AnchorObject *ob;

    AnchorParamDlgProc(AnchorObject *o) { ob = o; }
#if MAX_PRODUCT_VERSION_MAJOR > 8
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
#else
    BOOL DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
#endif
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

#if MAX_PRODUCT_VERSION_MAJOR > 8
INT_PTR AnchorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
#else
BOOL AnchorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
#endif
                                    UINT msg, WPARAM wParam, LPARAM lParam)
{
    return AnchorDlgProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *AnchorObject::pmapParam = NULL;

void
AnchorObject::BeginEditParams(IObjParam *ip, ULONG flags,
                              Animatable *prev)
{
    iObjParams = ip;
    if (pmapParam)
    {
        // Left over from last TouchSensor created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_ANCHOR),
#if MAX_PRODUCT_VERSION_MAJOR > 8
                                     GetString(IDS_ANCHOR_CLASS),
#else
                                     _T("Anchor" /*JP_LOC*/),
#endif
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new AnchorParamDlgProc(this));
    }
}

void
AnchorObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

AnchorObject::AnchorObject()
    : HelperObject()
{
    pblock = NULL;
    previousMode = NULL;
    triggerObject = NULL;
    cameraObject = NULL;
    isJump = TRUE;
#if MAX_PRODUCT_VERSION_MAJOR > 8
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_AN_LENGTH,
#else
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_AN_LENGTH,
#endif
                                           CURRENT_VERSION);
    pb->SetValue(PB_AN_TYPE, 0, 0);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

AnchorObject::~AnchorObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *AnchorObject::iObjParams;

// This is only called if the object MAKES references to other things.

#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult AnchorObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                         PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult AnchorObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                         PartID &partID, RefMessage message)
#endif
{
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        if (hTarget == triggerObject)
            triggerObject = NULL;
        if (hTarget == cameraObject)
            cameraObject = NULL;
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
AnchorObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return triggerObject;
    if (ind == 2)
        return cameraObject;
    return NULL;
}

void
AnchorObject::SetReference(int ind, RefTargetHandle rtarg)
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
    if (ind == 2)
    {
        cameraObject = (INode *)rtarg;
        return;
    }
    return;
}

ObjectState
AnchorObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
AnchorObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
AnchorObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
AnchorObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                               Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
AnchorObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                               Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
#if MAX_PRODUCT_VERSION_MAJOR > 8
#else
    mesh.buildBoundingBox();
#endif
    GetMat(t, inode, vpt, tm);

    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
}

void
AnchorObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_AN_SIZE, t, size, FOREVER);
#include "anchorob.cpp"
}

int
AnchorObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_AN_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 0.4f, 0.0f, 0.6f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
AnchorObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class AnchorCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    AnchorObject *anchorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(AnchorObject *obj) { anchorObject = obj; }
};

int
AnchorCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            anchorObject->pblock->SetValue(PB_AN_SIZE,
                                           anchorObject->iObjParams->GetTime(), radius);
            anchorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(anchorObject->iObjParams->SnapAngle(ang));
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
static AnchorCreateCallBack AnchorCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
AnchorObject::GetCreateMouseCallBack()
{
    AnchorCreateCB.SetObj(this);
    return (&AnchorCreateCB);
}

#define DESC_CHUNK 0xad30
#define URL_CHUNK 0xad31
#define PARAM_CHUNK 0xad32

IOResult
AnchorObject::Save(ISave *isave)
{
    isave->BeginChunk(DESC_CHUNK);
    isave->WriteCString(description.data());
    isave->EndChunk();

    isave->BeginChunk(URL_CHUNK);
    isave->WriteCString(URL.data());
    isave->EndChunk();

    isave->BeginChunk(PARAM_CHUNK);
    isave->WriteCString(parameter.data());
    isave->EndChunk();

    return IO_OK;
}

IOResult
AnchorObject::Load(ILoad *iload)
{
    TCHAR *txt;

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case DESC_CHUNK:
            iload->ReadCStringChunk(&txt);
            description = txt;
            break;

        case URL_CHUNK:
            iload->ReadCStringChunk(&txt);
            URL = txt;
            break;

        case PARAM_CHUNK:
            iload->ReadCStringChunk(&txt);
            parameter = txt;
            break;

        default:
            break;
        }
        iload->CloseChunk();
    }
    return IO_OK;
}

RefTargetHandle
AnchorObject::Clone(RemapDir &remap)
{
    AnchorObject *ts = new AnchorObject();
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ts->ReplaceReference(0, remap.CloneRef(pblock));
    ts->ReplaceReference(1, triggerObject);
    ts->ReplaceReference(2, cameraObject);
#else
    ts->ReplaceReference(0, pblock->Clone(remap));
    if (remap.FindMapping(triggerObject))
        ts->ReplaceReference(1, remap.FindMapping(triggerObject));
    else
        ts->ReplaceReference(1, triggerObject);
    if (remap.FindMapping(cameraObject))
        ts->ReplaceReference(2, remap.FindMapping(cameraObject));
    else
        ts->ReplaceReference(2, cameraObject);
#endif
    ts->description = description;
    ts->URL = URL;
    ts->parameter = parameter;
    BaseClone(this, ts, remap);
    return ts;
}
