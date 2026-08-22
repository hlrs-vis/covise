/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: timeSteps.cpp
 
    DESCRIPTION:  Defines a VRML 2.0 Timesteps helper
 
    CREATED BY: Uwe Woessner
 
    HISTORY: created 29 Aug. 2026
 
 *> Copyright (c) 2026, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "timeSteps.h"

//------------------------------------------------------

class TimestepsClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new TimestepsObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_TIMESTEPS_SENSOR_CLASS); }
    const TCHAR* NonLocalizedClassName() { return _T("Timesteps"); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Timesteps_CLASS_ID1,
                                         Timesteps_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static TimestepsClassDesc TimestepsDesc;

ClassDesc *GetTimestepsDesc() { return &TimestepsDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *TimestepsObject::TimestepsPickButton = NULL;

HWND TimestepsObject::hRollup = NULL;
int TimestepsObject::dlgPrevSel = -1;

class TimestepsObjPick : public PickModeCallback
{
    TimestepsObject *Timesteps;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetTimesteps(TimestepsObject *l) { Timesteps = l; }
};

static TimestepsObjPick thePick;
static BOOL pickMode = FALSE;
static CommandMode *lastMode = NULL;

static void
SetPickMode(TimestepsObject *tso)
{
    if (pickMode)
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
        thePick.SetTimesteps(tso);
        GetCOREInterface()->SetPickMode(&thePick);
    }
}

BOOL
TimestepsObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                           int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(Timesteps_CLASS_ID1, Timesteps_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
TimestepsObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_TimeSteps_PICK_MODE));
}

void
TimestepsObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
TimestepsObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < Timesteps->TimestepsObjects.Count(); i++)
        {
            if (Timesteps->TimestepsObjects[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        TimestepsObj *obj = new TimestepsObj(node);
        int id = Timesteps->TimestepsObjects.Append(1, &obj);
        Timesteps->pblock->SetValue(PB_NUMOBJS,
                                     Timesteps->iObjParams->GetTime(),
                                     Timesteps->TimestepsObjects.Count());

#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = Timesteps->ReplaceReference(id + 1, node);
#else
        RefResult ret = Timesteps->MakeRefByID(FOREVER, id + 1, node);
#endif

        HWND hw = Timesteps->hRollup;
        int ind = (int)SendMessage(GetDlgItem(hw, IDC_TimeSteps_LIST),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hw, IDC_TimeSteps_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        EnableWindow(GetDlgItem(hw, IDC_TimeSteps_DEL),
                     Timesteps->TimestepsObjects.Count() > 0);
    }
    return FALSE;
}

HCURSOR
TimestepsObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(TimestepsObject *ob)
{
    if (ob && ob->hRollup)
    {
        int count, i;

        count = (int)SendDlgItemMessage(ob->hRollup, IDC_TimeSteps_LIST,
                                        LB_GETCOUNT, 0, 0);

        // First remove any objects on the list
        for (i = count - 1; i >= 0; i--)
            SendDlgItemMessage(ob->hRollup, IDC_TimeSteps_LIST,
                               LB_DELETESTRING, (WPARAM)i, 0);

        for (i = 0; i < ob->TimestepsObjects.Count(); i++)
        {
            TimestepsObj *obj = ob->TimestepsObjects[i];
            obj->ResetStr(); // Make sure we're up to date

            // for now just load the name, we might want to add
            // the frame range as some point
            int ind = (int)SendMessage(GetDlgItem(ob->hRollup, IDC_TimeSteps_LIST),
                                       LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(ob->hRollup, IDC_TimeSteps_LIST),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        }
    }
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     TimestepsObject *th)
{
    int loop = FALSE; // used to test for IDC_LOOP

    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
        //        SetDlgFont( hDlg, th->iObjParams->GetAppHFont() );

        th->TimestepsPickButton = GetICustButton(GetDlgItem(hDlg, IDC_TimeSteps_PICK));
        th->TimestepsPickButton->SetType(CBT_CHECK);
        th->TimestepsPickButton->SetButtonDownNotify(TRUE);
        th->TimestepsPickButton->SetHighlightColor(GREEN_WASH);
        th->TimestepsPickButton->SetCheck(FALSE);

        // only enable IDC_START_ON_LOAD if IDC_LOOP is checked
        th->pblock->GetValue(PB_LOOP, th->iObjParams->GetTime(), loop, FOREVER);
        EnableWindow(GetDlgItem(hDlg, IDC_START_ON_LOAD), loop);

        // Now we need to fill in the list box IDC_TimeSteps_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        //        EnableWindow(GetDlgItem(hDlg, IDC_TimeSteps_DEL),
        //                     (th->TimestepsObjects.Count() > 0));
        th->dlgPrevSel = -1;

        if (pickMode)
            SetPickMode(th);

        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(th);
        //th->iObjParams->ClearPickMode();
        //th->previousMode = NULL;
        ReleaseICustButton(th->TimestepsPickButton);
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
        case IDC_LOOP: // only enable IDC_START_ON_LOAD if IDC_LOOP is checked
            if (!IsDlgButtonChecked(hDlg, IDC_LOOP))
            {
            }
            break;
        case IDC_TimeSteps_PICK: // Pick an object from the scene
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
                    thePick.SetTimesteps(th);
                    th->iObjParams->SetPickMode(&thePick);
                }
                */
                break;
            }
            break;
        case IDC_TimeSteps_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_TimeSteps_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                TimestepsObj *obj = (TimestepsObj *)
                    SendDlgItemMessage(hDlg, IDC_TimeSteps_LIST,
                                       LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->TimestepsObjects.Count(); i++)
                {
                    if (obj == th->TimestepsObjects[i])
                    {
                        // remove the item from the list
                        SendDlgItemMessage(hDlg, IDC_TimeSteps_LIST,
                                           LB_DELETESTRING,
                                           (WPARAM)index, 0);
                        th->dlgPrevSel = -1;
                        // remove the object from the table
                        th->DeleteReference(i + 1);
                        th->TimestepsObjects.Delete(i, 1);
                        th->pblock->SetValue(PB_NUMOBJS,
                                             th->iObjParams->GetTime(),
                                             th->TimestepsObjects.Count());
                        break;
                    }
                }
                EnableWindow(GetDlgItem(hDlg, IDC_TimeSteps_DEL),
                             (th->TimestepsObjects.Count() > 0));
                if (th->TimestepsObjects.Count() <= 0)
                {
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
            }
        }
        break;
        case IDC_TimeSteps_LIST:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_TimeSteps_LIST),
                                           LB_GETCURSEL, 0, 0);
                if (th->dlgPrevSel != -1)
                {
                    // save any editing
                    TimestepsObj *obj = (TimestepsObj *)
                        SendDlgItemMessage(hDlg, IDC_TimeSteps_LIST,
                                           LB_GETITEMDATA, th->dlgPrevSel, 0);
                    obj->ResetStr();
                    SendMessage(GetDlgItem(hDlg, IDC_TimeSteps_LIST),
                                LB_DELETESTRING, th->dlgPrevSel, 0);
                    int ind = (int)SendMessage(GetDlgItem(hDlg,
                                                          IDC_TimeSteps_LIST),
                                               LB_ADDSTRING, 0,
                                               (LPARAM)obj->listStr.data());
                    SendMessage(GetDlgItem(hDlg, IDC_TimeSteps_LIST),
                                LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                    SendMessage(GetDlgItem(hDlg, IDC_TimeSteps_LIST),
                                LB_SETCURSEL, sel, 0);
                }
                th->dlgPrevSel = sel;
                if (sel >= 0)
                {
                    TimestepsObj *obj = (TimestepsObj *)
                        SendDlgItemMessage(hDlg, IDC_TimeSteps_LIST,
                                           LB_GETITEMDATA, sel, 0);
                    assert(obj);
                }
                else
                {
                }
                th->iObjParams->RedrawViews(th->iObjParams->GetTime());
            }
            break;
            case LBN_SELCANCEL:
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
        PB_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_TimeSteps_SIZE, IDC_TimeSteps_SIZE_SPINNER,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Loop
    ParamUIDesc(PB_LOOP, TYPE_SINGLECHEKBOX, IDC_LOOP),


    // num timesteps
    ParamUIDesc(
        PB_NUMTIMESTEPS,
        EDITTYPE_INT,
        IDC_NT_EDIT, IDC_NT_SPIN,
        0, 10,
        100000),

    // Speed
    ParamUIDesc(
        PB_TS_SPEED,
        EDITTYPE_FLOAT,
        IDC_MF_EDIT, IDC_MF_SPIN,
        0, 300,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 4


static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_INT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_INT, NULL, FALSE, 6 },
};

static ParamVersionDesc versions[] = {
    ParamVersionDesc(descVer0, 6, 0),
};

#define NUM_OLD_VERSIONS 0

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_LENGTH, CURRENT_VERSION);

class TimestepsParamDlgProc : public ParamMapUserDlgProc
{
public:
    TimestepsObject *ob;

    TimestepsParamDlgProc(TimestepsObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR TimestepsParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                        UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *TimestepsObject::pmapParam = NULL;

IOResult
TimestepsObject::Load(ILoad *iload)
{
    iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                       NUM_OLD_VERSIONS,
                                                       &curVersion, this, 0));
    return IO_OK;
}

void
TimestepsObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                  Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last Timesteps created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_TimeSteps),
                                     _T("Time Steps" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new TimestepsParamDlgProc(this));
    }
}

void
TimestepsObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

TimestepsObject::TimestepsObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_LENGTH,
                                           CURRENT_VERSION);
    TimeValue duration = TheManager->Max()->GetAnimRange().Duration();
    pb->SetValue(PB_SIZE, 0, 0.0f);
    pb->SetValue(PB_LOOP, 0, FALSE);
    pb->SetValue(PB_NUMTIMESTEPS, 0, (int)duration);
    pb->SetValue(PB_TS_SPEED, 0, (float)25.0);
    pb->SetValue(PB_NUMOBJS, 0, 0);
    ReplaceReference(0, pb);
    assert(pblock);
    previousMode = NULL;
    TimestepsObjects.SetCount(0);
    BuildObjectList(this);

    vrmlWritten = false;
}

TimestepsObject::~TimestepsObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < TimestepsObjects.Count(); i++)
    {
        TimestepsObj *obj = TimestepsObjects[i];
        delete obj;
    }
}

IObjParam *TimestepsObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult TimestepsObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult TimestepsObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message)
#endif
{
    int i;
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < TimestepsObjects.Count(); i++)
        {
            if (TimestepsObjects[i]->node == hTarget)
            {
                TimestepsObjects.Delete(i, 1);
                // Do I need to remove the reference? FIXME
                int numObjs;
                pblock->GetValue(PB_NUMOBJS, 0, numObjs,
                                 FOREVER);
                numObjs--;
                pblock->SetValue(PB_NUMOBJS, 0, numObjs);
            }
        }
        break;
    case REFMSG_NODE_NAMECHANGE:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < TimestepsObjects.Count(); i++)
        {
            if (TimestepsObjects[i]->node == hTarget)
            {
                // Found it
                TimestepsObjects[i]->ResetStr();
                break;
            }
        }
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
TimestepsObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    if (ind > TimestepsObjects.Count())
        return NULL;

    if (TimestepsObjects[ind - 1] == NULL)
        return NULL;
    return TimestepsObjects[ind - 1]->node;
}

void
TimestepsObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        if (pblock)
        {
            int numObjs;
            pblock->GetValue(PB_NUMOBJS, 0, numObjs,
                             FOREVER);
            if (TimestepsObjects.Count() == 0)
            {
                TimestepsObjects.SetCount(numObjs);
                for (int i = 0; i < numObjs; i++)
                    TimestepsObjects[i] = new TimestepsObj();
            }
        }
        return;
    }
    else if (ind > TimestepsObjects.Count())
        return;

    TimestepsObjects[ind - 1]->node = (INode *)rtarg;
    TimestepsObjects[ind - 1]->ResetStr();
}

ObjectState
TimestepsObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
TimestepsObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
TimestepsObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
TimestepsObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                   Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
TimestepsObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
TimestepsObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_SIZE, t, size, FOREVER);
#include "clockob.cpp"
    mesh.buildBoundingBox();
}

int
TimestepsObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 0.0f, 1.0f, 0.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
TimestepsObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class TimestepsCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    TimestepsObject *TSO;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(TimestepsObject *obj) { TSO = obj; }
};

int
TimestepsCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            TSO->pblock->SetValue(PB_SIZE,
                TSO->iObjParams->GetTime(), radius);
            TSO->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(TSO->iObjParams->SnapAngle(ang));
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
static TimestepsCreateCallBack TimestepsCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
TimestepsObject::GetCreateMouseCallBack()
{
    TimestepsCreateCB.SetObj(this);
    return (&TimestepsCreateCB);
}

RefTargetHandle
TimestepsObject::Clone(RemapDir &remap)
{
    TimestepsObject *ts = new TimestepsObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    ts->TimestepsObjects.SetCount(TimestepsObjects.Count());
    for (int i = 0; i < TimestepsObjects.Count(); i++)
    {
        if (remap.FindMapping(TimestepsObjects[i]->node))
            ts->ReplaceReference(i + 1, remap.FindMapping(TimestepsObjects[i]->node));
        else
            ts->ReplaceReference(i + 1, TimestepsObjects[i]->node);
    }

    BaseClone(this, ts, remap);
    return ts;
}
