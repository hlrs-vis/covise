/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: timer.cpp

    DESCRIPTION:  A VRML Time Sensor Helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 15 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "timer.h"

//------------------------------------------------------

class TimeSensorClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new TimeSensorObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_TIME_SENSOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(TimeSensor_CLASS_ID1,
                                         TimeSensor_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static TimeSensorClassDesc TimeSensorDesc;

ClassDesc *GetTimeSensorDesc() { return &TimeSensorDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *TimeSensorObject::TimeSensorPickButton = NULL;

HWND TimeSensorObject::hRollup = NULL;
int TimeSensorObject::dlgPrevSel = -1;

class TimeSensorObjPick : public PickModeCallback
{
    TimeSensorObject *timeSensor;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetTimeSensor(TimeSensorObject *l) { timeSensor = l; }
};

static TimeSensorObjPick thePick;
static BOOL pickMode = FALSE;
static CommandMode *lastMode = NULL;

static void
SetPickMode(TimeSensorObject *tso)
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
        thePick.SetTimeSensor(tso);
        GetCOREInterface()->SetPickMode(&thePick);
    }
}

BOOL
TimeSensorObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                           int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(TimeSensor_CLASS_ID1, TimeSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
TimeSensorObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_TimeSensor_PICK_MODE));
}

void
TimeSensorObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
TimeSensorObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < timeSensor->TimeSensorObjects.Count(); i++)
        {
            if (timeSensor->TimeSensorObjects[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        TimeSensorObj *obj = new TimeSensorObj(node);
        int id = timeSensor->TimeSensorObjects.Append(1, &obj);
        timeSensor->pblock->SetValue(PB_NUMOBJS,
                                     timeSensor->iObjParams->GetTime(),
                                     timeSensor->TimeSensorObjects.Count());

#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = timeSensor->ReplaceReference(id + 1, node);
#else
        RefResult ret = timeSensor->MakeRefByID(FOREVER, id + 1, node);
#endif

        HWND hw = timeSensor->hRollup;
        int ind = (int)SendMessage(GetDlgItem(hw, IDC_TimeSensor_LIST),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hw, IDC_TimeSensor_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        EnableWindow(GetDlgItem(hw, IDC_TimeSensor_DEL),
                     timeSensor->TimeSensorObjects.Count() > 0);
    }
    return FALSE;
}

HCURSOR
TimeSensorObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(TimeSensorObject *ob)
{
    if (ob && ob->hRollup)
    {
        int count, i;

        count = (int)SendDlgItemMessage(ob->hRollup, IDC_TimeSensor_LIST,
                                        LB_GETCOUNT, 0, 0);

        // First remove any objects on the list
        for (i = count - 1; i >= 0; i--)
            SendDlgItemMessage(ob->hRollup, IDC_TimeSensor_LIST,
                               LB_DELETESTRING, (WPARAM)i, 0);

        for (i = 0; i < ob->TimeSensorObjects.Count(); i++)
        {
            TimeSensorObj *obj = ob->TimeSensorObjects[i];
            obj->ResetStr(); // Make sure we're up to date

            // for now just load the name, we might want to add
            // the frame range as some point
            int ind = (int)SendMessage(GetDlgItem(ob->hRollup, IDC_TimeSensor_LIST),
                                       LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(ob->hRollup, IDC_TimeSensor_LIST),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        }
    }
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     TimeSensorObject *th)
{
    int loop = FALSE; // used to test for IDC_LOOP

    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
        //        SetDlgFont( hDlg, th->iObjParams->GetAppHFont() );

        th->TimeSensorPickButton = GetICustButton(GetDlgItem(hDlg, IDC_TimeSensor_PICK));
        th->TimeSensorPickButton->SetType(CBT_CHECK);
        th->TimeSensorPickButton->SetButtonDownNotify(TRUE);
        th->TimeSensorPickButton->SetHighlightColor(GREEN_WASH);
        th->TimeSensorPickButton->SetCheck(FALSE);

        // only enable IDC_START_ON_LOAD if IDC_LOOP is checked
        th->pblock->GetValue(PB_LOOP, th->iObjParams->GetTime(), loop, FOREVER);
        EnableWindow(GetDlgItem(hDlg, IDC_START_ON_LOAD), loop);

        // Now we need to fill in the list box IDC_TimeSensor_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        //        EnableWindow(GetDlgItem(hDlg, IDC_TimeSensor_DEL),
        //                     (th->TimeSensorObjects.Count() > 0));
        th->dlgPrevSel = -1;

        if (pickMode)
            SetPickMode(th);

        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(th);
        //th->iObjParams->ClearPickMode();
        //th->previousMode = NULL;
        ReleaseICustButton(th->TimeSensorPickButton);
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
                CheckDlgButton(hDlg, IDC_START_ON_LOAD, 0);
                th->pblock->SetValue(PB_START_ON_LOAD, th->iObjParams->GetTime(), FALSE);
            }
            EnableWindow(GetDlgItem(hDlg, IDC_START_ON_LOAD), IsDlgButtonChecked(hDlg, IDC_LOOP));
            break;
        case IDC_TimeSensor_PICK: // Pick an object from the scene
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
                    thePick.SetTimeSensor(th);
                    th->iObjParams->SetPickMode(&thePick);
                }
                */
                break;
            }
            break;
        case IDC_TimeSensor_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_TimeSensor_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                TimeSensorObj *obj = (TimeSensorObj *)
                    SendDlgItemMessage(hDlg, IDC_TimeSensor_LIST,
                                       LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->TimeSensorObjects.Count(); i++)
                {
                    if (obj == th->TimeSensorObjects[i])
                    {
                        // remove the item from the list
                        SendDlgItemMessage(hDlg, IDC_TimeSensor_LIST,
                                           LB_DELETESTRING,
                                           (WPARAM)index, 0);
                        th->dlgPrevSel = -1;
                        // remove the object from the table
                        th->DeleteReference(i + 1);
                        th->TimeSensorObjects.Delete(i, 1);
                        th->pblock->SetValue(PB_NUMOBJS,
                                             th->iObjParams->GetTime(),
                                             th->TimeSensorObjects.Count());
                        break;
                    }
                }
                EnableWindow(GetDlgItem(hDlg, IDC_TimeSensor_DEL),
                             (th->TimeSensorObjects.Count() > 0));
                if (th->TimeSensorObjects.Count() <= 0)
                {
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
            }
        }
        break;
        case IDC_TimeSensor_LIST:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_TimeSensor_LIST),
                                           LB_GETCURSEL, 0, 0);
                if (th->dlgPrevSel != -1)
                {
                    // save any editing
                    TimeSensorObj *obj = (TimeSensorObj *)
                        SendDlgItemMessage(hDlg, IDC_TimeSensor_LIST,
                                           LB_GETITEMDATA, th->dlgPrevSel, 0);
                    obj->ResetStr();
                    SendMessage(GetDlgItem(hDlg, IDC_TimeSensor_LIST),
                                LB_DELETESTRING, th->dlgPrevSel, 0);
                    int ind = (int)SendMessage(GetDlgItem(hDlg,
                                                          IDC_TimeSensor_LIST),
                                               LB_ADDSTRING, 0,
                                               (LPARAM)obj->listStr.data());
                    SendMessage(GetDlgItem(hDlg, IDC_TimeSensor_LIST),
                                LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                    SendMessage(GetDlgItem(hDlg, IDC_TimeSensor_LIST),
                                LB_SETCURSEL, sel, 0);
                }
                th->dlgPrevSel = sel;
                if (sel >= 0)
                {
                    TimeSensorObj *obj = (TimeSensorObj *)
                        SendDlgItemMessage(hDlg, IDC_TimeSensor_LIST,
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
        IDC_TimeSensor_SIZE, IDC_TimeSensor_SIZE_SPINNER,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Loop
    ParamUIDesc(PB_LOOP, TYPE_SINGLECHEKBOX, IDC_LOOP),

    // Start On Load
    ParamUIDesc(PB_START_ON_LOAD, TYPE_SINGLECHEKBOX, IDC_START_ON_LOAD),

    // Start Time
    ParamUIDesc(
        PB_START_TIME,
        EDITTYPE_TIME,
        IDC_START_EDIT, IDC_START_SPIN,
        -999999999.0f, 999999999.0f,
        10.0f),

    // Stop Time
    ParamUIDesc(
        PB_STOP_TIME,
        EDITTYPE_TIME,
        IDC_STOP_EDIT, IDC_STOP_SPIN,
        -999999999.0f, 999999999.0f,
        10.0f),

    // CycleInterval
    ParamUIDesc(
        PB_CYCLEINTERVAL,
        EDITTYPE_FLOAT,
        IDC_TimeSensor_CYCLE, IDC_TimeSensor_CYCLE_SPIN,
        0.0f, 315360000.0f,
        SPIN_AUTOSCALE),
};

#define PARAMDESC_LENGTH 6

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_INT, NULL, FALSE, 4 }
};

static ParamBlockDescID descVer1[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_INT, NULL, FALSE, 4 },
    { TYPE_INT, NULL, FALSE, 5 },
};

static ParamBlockDescID descVer2[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_INT, NULL, FALSE, 4 },
    { TYPE_INT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
};

static ParamVersionDesc versions[] = {
    ParamVersionDesc(descVer0, 5, 0),
    ParamVersionDesc(descVer1, 6, 1),
    ParamVersionDesc(descVer2, 7, 2),
};

#define NUM_OLD_VERSIONS 2

#define CURRENT_VERSION 2
// Current version
static ParamVersionDesc curVersion(descVer2, PB_LENGTH, CURRENT_VERSION);

class TimeSensorParamDlgProc : public ParamMapUserDlgProc
{
public:
    TimeSensorObject *ob;

    TimeSensorParamDlgProc(TimeSensorObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR TimeSensorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                        UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *TimeSensorObject::pmapParam = NULL;

IOResult
TimeSensorObject::Load(ILoad *iload)
{
    iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                       NUM_OLD_VERSIONS,
                                                       &curVersion, this, 0));
    return IO_OK;
}

void
TimeSensorObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                  Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last TimeSensor created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_TimeSensor),
                                     _T("Time Sensor" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new TimeSensorParamDlgProc(this));
    }
}

void
TimeSensorObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

TimeSensorObject::TimeSensorObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer2, PB_LENGTH,
                                           CURRENT_VERSION);
    TimeValue start = TheManager->Max()->GetAnimRange().Start();
    TimeValue end = TheManager->Max()->GetAnimRange().End();
    float cycleInterval = (end - start) / ((float)GetTicksPerFrame() * GetFrameRate());
    pb->SetValue(PB_SIZE, 0, 0.0f);
    pb->SetValue(PB_LOOP, 0, FALSE);
    pb->SetValue(PB_START_ON_LOAD, 0, FALSE);
    pb->SetValue(PB_START_TIME, 0, start);
    pb->SetValue(PB_STOP_TIME, 0, end);
    pb->SetValue(PB_NUMOBJS, 0, 0);
    pb->SetValue(PB_CYCLEINTERVAL, 0, cycleInterval);
    ReplaceReference(0, pb);
    assert(pblock);
    previousMode = NULL;
    TimeSensorObjects.SetCount(0);
    BuildObjectList(this);

    vrmlWritten = false;
}

TimeSensorObject::~TimeSensorObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < TimeSensorObjects.Count(); i++)
    {
        TimeSensorObj *obj = TimeSensorObjects[i];
        delete obj;
    }
}

IObjParam *TimeSensorObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult TimeSensorObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult TimeSensorObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message)
#endif
{
    int i;
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < TimeSensorObjects.Count(); i++)
        {
            if (TimeSensorObjects[i]->node == hTarget)
            {
                TimeSensorObjects.Delete(i, 1);
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
        for (i = 0; i < TimeSensorObjects.Count(); i++)
        {
            if (TimeSensorObjects[i]->node == hTarget)
            {
                // Found it
                TimeSensorObjects[i]->ResetStr();
                break;
            }
        }
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
TimeSensorObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    if (ind > TimeSensorObjects.Count())
        return NULL;

    if (TimeSensorObjects[ind - 1] == NULL)
        return NULL;
    return TimeSensorObjects[ind - 1]->node;
}

void
TimeSensorObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        if (pblock)
        {
            int numObjs;
            pblock->GetValue(PB_NUMOBJS, 0, numObjs,
                             FOREVER);
            if (TimeSensorObjects.Count() == 0)
            {
                TimeSensorObjects.SetCount(numObjs);
                for (int i = 0; i < numObjs; i++)
                    TimeSensorObjects[i] = new TimeSensorObj();
            }
        }
        return;
    }
    else if (ind > TimeSensorObjects.Count())
        return;

    TimeSensorObjects[ind - 1]->node = (INode *)rtarg;
    TimeSensorObjects[ind - 1]->ResetStr();
}

ObjectState
TimeSensorObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
TimeSensorObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
TimeSensorObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
TimeSensorObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                   Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
TimeSensorObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
TimeSensorObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_SIZE, t, size, FOREVER);
#include "clockob.cpp"
    mesh.buildBoundingBox();
}

int
TimeSensorObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
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
TimeSensorObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class TimeSensorCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    TimeSensorObject *timeSensorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(TimeSensorObject *obj) { timeSensorObject = obj; }
};

int
TimeSensorCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            timeSensorObject->pblock->SetValue(PB_SIZE,
                                               timeSensorObject->iObjParams->GetTime(), radius);
            timeSensorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(timeSensorObject->iObjParams->SnapAngle(ang));
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
static TimeSensorCreateCallBack TimeSensorCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
TimeSensorObject::GetCreateMouseCallBack()
{
    TimeSensorCreateCB.SetObj(this);
    return (&TimeSensorCreateCB);
}

RefTargetHandle
TimeSensorObject::Clone(RemapDir &remap)
{
    TimeSensorObject *ts = new TimeSensorObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    ts->TimeSensorObjects.SetCount(TimeSensorObjects.Count());
    for (int i = 0; i < TimeSensorObjects.Count(); i++)
    {
        if (remap.FindMapping(TimeSensorObjects[i]->node))
            ts->ReplaceReference(i + 1, remap.FindMapping(TimeSensorObjects[i]->node));
        else
            ts->ReplaceReference(i + 1, TimeSensorObjects[i]->node);
    }

    BaseClone(this, ts, remap);
    return ts;
}
