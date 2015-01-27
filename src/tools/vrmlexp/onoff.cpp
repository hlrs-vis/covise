/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: onoff.cpp

    DESCRIPTION:  A VRML Onoff Sensor Helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 4 Sept, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "onoff.h"

//------------------------------------------------------

class OnOffSwitchClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new OnOffSwitchObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_ONOFF_SWITCH_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(OnOffSwitch_CLASS_ID1, OnOffSwitch_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static OnOffSwitchClassDesc OnOffSwitchDesc;

ClassDesc *GetOnOffSwitchDesc() { return &OnOffSwitchDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *OnOffSwitchObject::OnPickButton = NULL;
ICustButton *OnOffSwitchObject::OffPickButton = NULL;

HWND OnOffSwitchObject::hRollup = NULL;

class ObjPick : public PickModeCallback
{
    OnOffSwitchObject *targetObject;
    int onOrOff;

public:
    ObjPick(int i);
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetOnOffSwitch(OnOffSwitchObject *l) { targetObject = l; }
};
ObjPick::ObjPick(int i)
{
    onOrOff = i;
}

//static ObjPick    theParentPick;
#define ON_PICK_MODE 1
#define OFF_PICK_MODE 2

static ObjPick theOnPick(1);
static ObjPick theOffPick(0);

static PickModeCallback *lastPick = NULL;
static void
SetPickMode(PickModeCallback *p)
{
    static CommandMode *lastMode = NULL;
    if (p)
    {
        if (lastPick != p)
        {
            if (lastPick)
            {
                GetCOREInterface()->PushCommandMode(lastMode);
                GetCOREInterface()->ClearPickMode();
            }
            lastMode = GetCOREInterface()->GetCommandMode();
            GetCOREInterface()->SetPickMode(p);
            lastPick = p;
        }
    }
    else
    {
        lastPick = NULL;
        GetCOREInterface()->PushCommandMode(lastMode);
        lastMode = NULL;
        GetCOREInterface()->ClearPickMode();
    }
}

BOOL
ObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                 int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(OnOffSwitch_CLASS_ID1, OnOffSwitch_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
ObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
ObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
ObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        HWND hw = targetObject->hRollup;
        if (onOrOff)
        {
            if (targetObject->ReplaceReference(1, node) == REF_SUCCEED)
            {
                Static_SetText(GetDlgItem(hw, IDC_ON_OBJ),
                               targetObject->onObject->GetName());
            }
        }
        else
        {
            if (targetObject->ReplaceReference(2, node) == REF_SUCCEED)
            {
                Static_SetText(GetDlgItem(hw, IDC_OFF_OBJ),
                               targetObject->offObject->GetName());
            }
        }
        targetObject->OnPickButton->SetCheck(FALSE);
        targetObject->OffPickButton->SetCheck(FALSE);
        if (targetObject->onObject == NULL)
        {
            SetPickMode(&theOnPick);
            targetObject->OnPickButton->SetCheck(TRUE);
        }
        else if (targetObject->offObject == NULL)
        {
            SetPickMode(&theOffPick);
            targetObject->OffPickButton->SetCheck(TRUE);
        }
        else
        {
            SetPickMode(NULL);
        }
    }
    return FALSE;
}

HCURSOR
ObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     OnOffSwitchObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:

        th->OnPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK_ON));
        th->OnPickButton->SetType(CBT_CHECK);
        th->OnPickButton->SetButtonDownNotify(TRUE);
        th->OnPickButton->SetHighlightColor(GREEN_WASH);
        th->OnPickButton->SetCheck(FALSE);

        th->OffPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK_OFF));
        th->OffPickButton->SetType(CBT_CHECK);
        th->OffPickButton->SetButtonDownNotify(TRUE);
        th->OffPickButton->SetHighlightColor(GREEN_WASH);
        th->OffPickButton->SetCheck(FALSE);

        // Now we need to fill in the list box IDC_LIST
        th->hRollup = hDlg;

        //        EnableWindow(GetDlgItem(hDlg, IDC_DEL),
        //                     (th->objects.Count() > 0));
        if (th->onObject)
            Static_SetText(GetDlgItem(hDlg, IDC_ON_OBJ),
                           th->onObject->GetName());
        if (th->offObject)
            Static_SetText(GetDlgItem(hDlg, IDC_OFF_OBJ),
                           th->offObject->GetName());

        if (lastPick)
            SetPickMode(NULL);
        return TRUE;

    case WM_DESTROY:
        if (lastPick)
            SetPickMode(NULL);
        // th->iObjParams->ClearPickMode();
        // th->previousMode = NULL;
        ReleaseICustButton(th->OnPickButton);
        ReleaseICustButton(th->OffPickButton);
        return FALSE;

    case WM_MOUSEACTIVATE:
        //        th->iObjParams->RealizeParamPanel();
        return FALSE;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MOUSEMOVE:
        //        th->iObjParams->RollupMouseMessage(hDlg,message,wParam,lParam);
        return FALSE;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_PICK_ON: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (lastPick == &theOffPick)
                {
                    SetPickMode(NULL);
                    th->OffPickButton->SetCheck(FALSE);
                }
                theOnPick.SetOnOffSwitch(th);
                theOffPick.SetOnOffSwitch(th);
                SetPickMode(&theOnPick);
                break;
            }
            break;
        case IDC_PICK_OFF: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (lastPick == &theOnPick)
                {
                    SetPickMode(NULL);
                    th->OnPickButton->SetCheck(FALSE);
                }
                theOnPick.SetOnOffSwitch(th);
                theOffPick.SetOnOffSwitch(th);
                SetPickMode(&theOffPick);
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
        PB_ONOFF_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 1

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
};

//static ParamVersionDesc versions[] = {
//  ParamVersionDesc(descVer0,5,0),
//};

//#define NUM_OLD_VERSIONS 1

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_ONOFF_LENGTH, CURRENT_VERSION);

class OnOffSwitchParamDlgProc : public ParamMapUserDlgProc
{
public:
    OnOffSwitchObject *ob;

    OnOffSwitchParamDlgProc(OnOffSwitchObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR OnOffSwitchParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                         UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *OnOffSwitchObject::pmapParam = NULL;

#if 0
IOResult
OnOffSwitchObject::Load(ILoad *iload) 
{
  iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                     NUM_OLD_VERSIONS,
                                                     &curVersion,this,0));
  return IO_OK;
}

#endif

void
OnOffSwitchObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                   Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {
        // Left over from last OnOffSwitch created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_ONOFF_SWITCH),
                                     _T("Onoff Sensor" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new OnOffSwitchParamDlgProc(this));
    }
}

void
OnOffSwitchObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

OnOffSwitchObject::OnOffSwitchObject()
    : HelperObject()
{
    pblock = NULL;
    previousMode = NULL;
    onObject = NULL;
    offObject = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_ONOFF_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_ONOFF_SIZE, 0, 0.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

OnOffSwitchObject::~OnOffSwitchObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *OnOffSwitchObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult OnOffSwitchObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                              PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult OnOffSwitchObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                              PartID &partID, RefMessage message)
#endif
{
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        // Find the ID on the list and call ResetStr
        if (hTarget == onObject)
            onObject = NULL;
        if (hTarget == offObject)
            offObject = NULL;
        break;
    case REFMSG_NODE_NAMECHANGE:
        // Find the ID on the list and call ResetStr

        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
OnOffSwitchObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return onObject;
    if (ind == 2)
        return offObject;
    return NULL;
}

void
OnOffSwitchObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        return;
    }
    if (ind == 1)
    {
        onObject = (INode *)rtarg;
        return;
    }
    if (ind == 2)
    {
        offObject = (INode *)rtarg;
        return;
    }
}

ObjectState
OnOffSwitchObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
OnOffSwitchObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
OnOffSwitchObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
OnOffSwitchObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                    Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
OnOffSwitchObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
OnOffSwitchObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_ONOFF_SIZE, t, size, FOREVER);
#include "onoffob.cpp"
    mesh.buildBoundingBox();
}

int
OnOffSwitchObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_ONOFF_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 0.4f, 1.0f, 0.6f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
OnOffSwitchObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class OnOffSwitchCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    OnOffSwitchObject *onoffSensorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(OnOffSwitchObject *obj) { onoffSensorObject = obj; }
};

int
OnOffSwitchCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            onoffSensorObject->pblock->SetValue(PB_ONOFF_SIZE,
                                                onoffSensorObject->iObjParams->GetTime(), radius);
            onoffSensorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(onoffSensorObject->iObjParams->SnapAngle(ang));
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
static OnOffSwitchCreateCallBack OnOffSwitchCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
OnOffSwitchObject::GetCreateMouseCallBack()
{
    OnOffSwitchCreateCB.SetObj(this);
    return (&OnOffSwitchCreateCB);
}

RefTargetHandle
OnOffSwitchObject::Clone(RemapDir &remap)
{
    OnOffSwitchObject *ts = new OnOffSwitchObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    if (onObject)
    {
        if (remap.FindMapping(onObject))
            ts->ReplaceReference(1, remap.FindMapping(onObject));
        else
            ts->ReplaceReference(1, onObject);
    }
    if (offObject)
    {
        if (remap.FindMapping(offObject))
            ts->ReplaceReference(2, remap.FindMapping(offObject));
        else
            ts->ReplaceReference(2, offObject);
    }

    BaseClone(this, ts, remap);
    return ts;
}
