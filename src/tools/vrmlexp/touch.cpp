/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: touch.cpp

    DESCRIPTION:  A VRML Touch Sensor Helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 4 Sept, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "touch.h"

//------------------------------------------------------

class TouchSensorClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new TouchSensorObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_TOUCH_SENSOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(TouchSensor_CLASS_ID1, TouchSensor_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static TouchSensorClassDesc TouchSensorDesc;

ClassDesc *GetTouchSensorDesc() { return &TouchSensorDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *TouchSensorObject::ParentPickButton = NULL;

HWND TouchSensorObject::hRollup = NULL;
int TouchSensorObject::dlgPrevSel = -1;

class ParentObjPick : public PickModeCallback
{
    TouchSensorObject *parent;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetTouchSensor(TouchSensorObject *l) { parent = l; }
};

//static ParentObjPick    theParentPick;
#define PARENT_PICK_MODE 1
#define TOUCH_PICK_MODE 2

static ParentObjPick thePPick;
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
        // thePick.SetTouchSensor(o);
        GetCOREInterface()->SetPickMode(p);
    }
}

BOOL
ParentObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                       int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(TouchSensor_CLASS_ID1, TouchSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
ParentObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
ParentObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
ParentObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL && parent->ReplaceReference(1, node) == REF_SUCCEED)
    {

        SetPickMode(NULL);
        parent->ParentPickButton->SetCheck(FALSE);
        HWND hw = parent->hRollup;
        Static_SetText(GetDlgItem(hw, IDC_TRIGGER_OBJ),
                       parent->triggerObject->GetName());
        return FALSE;
    }
    return FALSE;
}

HCURSOR
ParentObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

ICustButton *TouchSensorObject::TouchSensorPickButton = NULL;

class TouchSensorObjPick : public PickModeCallback
{
    TouchSensorObject *touchSensor;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetTouchSensor(TouchSensorObject *l) { touchSensor = l; }
};

// static TouchSensorObjPick thePick;
static TouchSensorObjPick theTSPick;

BOOL
TouchSensorObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                            int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(TouchSensor_CLASS_ID1, TouchSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
TouchSensorObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_TouchSensor_PICK_MODE));
}

void
TouchSensorObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
TouchSensorObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < touchSensor->objects.Count(); i++)
        {
            if (touchSensor->objects[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        // Don't allow a loop.  001129  --prs.
        if (node->TestForLoop(FOREVER, touchSensor) != REF_SUCCEED)
            return FALSE;

        TouchSensorObj *obj = new TouchSensorObj(node);
        int id = touchSensor->objects.Append(1, &obj);
        touchSensor->pblock->SetValue(PB_TS_NUMOBJS,
                                      touchSensor->iObjParams->GetTime(),
                                      touchSensor->objects.Count());

#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = touchSensor->ReplaceReference(id + 2, node);
#else
        RefResult ret = touchSensor->MakeRefByID(FOREVER, id + 2, node);
#endif

        HWND hw = touchSensor->hRollup;
        int ind = (int)SendMessage(GetDlgItem(hw, IDC_LIST),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hw, IDC_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        EnableWindow(GetDlgItem(hw, IDC_DEL),
                     touchSensor->objects.Count() > 0);
    }
    return FALSE;
}

HCURSOR
TouchSensorObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(TouchSensorObject *ob)
{
    if (ob && ob->hRollup)
    {
        int count, i;

        count = (int)SendDlgItemMessage(ob->hRollup, IDC_LIST,
                                        LB_GETCOUNT, 0, 0);

        // First remove any objects on the list
        for (i = count - 1; i >= 0; i--)
            SendDlgItemMessage(ob->hRollup, IDC_LIST,
                               LB_DELETESTRING, (WPARAM)i, 0);

        for (i = 0; i < ob->objects.Count(); i++)
        {
            TouchSensorObj *obj = ob->objects[i];
            obj->ResetStr(); // Make sure we're up to date

            // for now just load the name, we might want to add
            // the frame range as some point
            int ind = (int)SendMessage(GetDlgItem(ob->hRollup, IDC_LIST),
                                       LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(ob->hRollup, IDC_LIST),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        }
    }
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     TouchSensorObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
        //        SetDlgFont( hDlg, th->iObjParams->GetAppHFont() );

        th->TouchSensorPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK));
        th->TouchSensorPickButton->SetType(CBT_CHECK);
        th->TouchSensorPickButton->SetButtonDownNotify(TRUE);
        th->TouchSensorPickButton->SetHighlightColor(GREEN_WASH);
        th->TouchSensorPickButton->SetCheck(FALSE);

        th->ParentPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK_PARENT));
        th->ParentPickButton->SetType(CBT_CHECK);
        th->ParentPickButton->SetButtonDownNotify(TRUE);
        th->ParentPickButton->SetHighlightColor(GREEN_WASH);
        th->ParentPickButton->SetCheck(FALSE);

        // Now we need to fill in the list box IDC_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        //        EnableWindow(GetDlgItem(hDlg, IDC_DEL),
        //                     (th->objects.Count() > 0));
        th->dlgPrevSel = -1;
        if (th->triggerObject)
            Static_SetText(GetDlgItem(hDlg, IDC_TRIGGER_OBJ),
                           th->triggerObject->GetName());

        if (pickMode)
            SetPickMode(NULL);
        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(NULL);
        ReleaseICustButton(th->TouchSensorPickButton);
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
        case IDC_PICK: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (pickMode == PARENT_PICK_MODE)
                {
                    SetPickMode(NULL);
                    th->ParentPickButton->SetCheck(FALSE);
                }
                theTSPick.SetTouchSensor(th);
                SetPickMode(&theTSPick, TOUCH_PICK_MODE);
                /*
                if (th->previousMode) {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                } else {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    thePick.SetTouchSensor(th);
                    th->iObjParams->SetPickMode(&thePick);
                }
                */
                break;
            }
            break;
        case IDC_PICK_PARENT: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (pickMode == TOUCH_PICK_MODE)
                {
                    SetPickMode(NULL);
                    th->TouchSensorPickButton->SetCheck(FALSE);
                }
                thePPick.SetTouchSensor(th);
                SetPickMode(&thePPick, PARENT_PICK_MODE);
                break;
            }
            break;
        case IDC_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                TouchSensorObj *obj = (TouchSensorObj *)
                    SendDlgItemMessage(hDlg, IDC_LIST,
                                       LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->objects.Count(); i++)
                {
                    if (obj == th->objects[i])
                    {
                        // remove the item from the list
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_DELETESTRING,
                                           (WPARAM)index, 0);
                        th->dlgPrevSel = -1;
                        // Remove the reference to obj->node
                        th->DeleteReference(i + 2);
                        // remove the object from the table
                        th->objects.Delete(i, 1);
                        th->pblock->SetValue(PB_TS_NUMOBJS,
                                             th->iObjParams->GetTime(),
                                             th->objects.Count());
                        break;
                    }
                }
                EnableWindow(GetDlgItem(hDlg, IDC_DEL),
                             (th->objects.Count() > 0));
                if (th->objects.Count() <= 0)
                {
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
            }
        }
        break;
        case IDC_LIST:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                           LB_GETCURSEL, 0, 0);
                if (th->dlgPrevSel != -1)
                {
                    // save any editing
                    TouchSensorObj *obj = (TouchSensorObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_GETITEMDATA, th->dlgPrevSel, 0);
                    obj->ResetStr();
                    SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                LB_DELETESTRING, th->dlgPrevSel, 0);
                    int ind = (int)SendMessage(GetDlgItem(hDlg,
                                                          IDC_LIST),
                                               LB_ADDSTRING, 0,
                                               (LPARAM)obj->listStr.data());
                    SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                    SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                LB_SETCURSEL, sel, 0);
                }
                th->dlgPrevSel = sel;
                if (sel >= 0)
                {
                    TouchSensorObj *obj = (TouchSensorObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST,
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
        PB_TS_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Loop
    ParamUIDesc(PB_TS_ENABLED, TYPE_SINGLECHEKBOX, IDC_ENABLE),

};

#define PARAMDESC_LENGTH 2

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
};

//static ParamVersionDesc versions[] = {
//  ParamVersionDesc(descVer0,5,0),
//};

//#define NUM_OLD_VERSIONS 1

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_TS_LENGTH, CURRENT_VERSION);

class TouchSensorParamDlgProc : public ParamMapUserDlgProc
{
public:
    TouchSensorObject *ob;

    TouchSensorParamDlgProc(TouchSensorObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR TouchSensorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                         UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *TouchSensorObject::pmapParam = NULL;

#if 0
IOResult
TouchSensorObject::Load(ILoad *iload) 
{
  iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                     NUM_OLD_VERSIONS,
                                                     &curVersion,this,0));
  return IO_OK;
}

#endif

void
TouchSensorObject::BeginEditParams(IObjParam *ip, ULONG flags,
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
                                     MAKEINTRESOURCE(IDD_TOUCH_SENSOR),
                                     GetString(IDS_TOUCH_SENSOR_CLASS),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new TouchSensorParamDlgProc(this));
    }
}

void
TouchSensorObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

TouchSensorObject::TouchSensorObject()
    : HelperObject()
{
    pblock = NULL;
    triggerObject = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_TS_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_TS_SIZE, 0, 0.0f);
    pb->SetValue(PB_TS_ENABLED, 0, TRUE);
    pb->SetValue(PB_TS_NUMOBJS, 0, 0);
    ReplaceReference(0, pb);
    assert(pblock);
    previousMode = NULL;
    triggerObject = NULL;
    objects.SetCount(0);
    BuildObjectList(this);
}

TouchSensorObject::~TouchSensorObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < objects.Count(); i++)
    {
        TouchSensorObj *obj = objects[i];
        delete obj;
    }
}

IObjParam *TouchSensorObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult TouchSensorObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                              PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult TouchSensorObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                              PartID &partID, RefMessage message)
#endif
{
    int i;
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < objects.Count(); i++)
        {
            if (objects[i]->node == hTarget)
            {
                // Do I need to remove the reference? FIXME
                objects.Delete(i, 1);
            }
            int numObjs;
            pblock->GetValue(PB_TS_NUMOBJS, 0, numObjs,
                             FOREVER);
            numObjs--;
            pblock->SetValue(PB_TS_NUMOBJS, 0, numObjs);
        }
        if (hTarget == triggerObject)
            triggerObject = NULL;
        break;
    case REFMSG_NODE_NAMECHANGE:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < objects.Count(); i++)
        {
            if (objects[i]->node == hTarget)
            {
                // Found it
                objects[i]->ResetStr();
                break;
            }
        }
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
TouchSensorObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return triggerObject;
    if (ind - 1 > objects.Count())
        return NULL;

    if (objects[ind - 2] == NULL)
        return NULL;
    return objects[ind - 2]->node;
}

void
TouchSensorObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        if (pblock)
        {
            int numObjs;
            pblock->GetValue(PB_TS_NUMOBJS, 0, numObjs,
                             FOREVER);
            if (objects.Count() == 0)
            {
                objects.SetCount(numObjs);
                for (int i = 0; i < numObjs; i++)
                    objects[i] = new TouchSensorObj();
            }
        }
        return;
    }
    if (ind == 1)
    {
        triggerObject = (INode *)rtarg;
        return;
    }
    if (ind - 1 > objects.Count())
        return;

    objects[ind - 2]->node = (INode *)rtarg;
    objects[ind - 2]->ResetStr();
}

ObjectState
TouchSensorObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
TouchSensorObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
TouchSensorObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
TouchSensorObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                    Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
TouchSensorObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
TouchSensorObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_TS_SIZE, t, size, FOREVER);
#include "touchob.cpp"
    mesh.buildBoundingBox();
}

int
TouchSensorObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_TS_SIZE, t, radius, FOREVER);
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
TouchSensorObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class TouchSensorCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    TouchSensorObject *touchSensorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(TouchSensorObject *obj) { touchSensorObject = obj; }
};

int
TouchSensorCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            touchSensorObject->pblock->SetValue(PB_TS_SIZE,
                                                touchSensorObject->iObjParams->GetTime(), radius);
            touchSensorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(touchSensorObject->iObjParams->SnapAngle(ang));
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
static TouchSensorCreateCallBack TouchSensorCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
TouchSensorObject::GetCreateMouseCallBack()
{
    TouchSensorCreateCB.SetObj(this);
    return (&TouchSensorCreateCB);
}

RefTargetHandle
TouchSensorObject::Clone(RemapDir &remap)
{
    TouchSensorObject *ts = new TouchSensorObject();
    ts->ReplaceReference(0, remap.CloneRef(pblock));
    ts->objects.SetCount(objects.Count());
    if (triggerObject)
    {
        if (remap.FindMapping(triggerObject))
            ts->ReplaceReference(1, remap.FindMapping(triggerObject));
        else
            ts->ReplaceReference(1, triggerObject);
    }
    for (int i = 0; i < objects.Count(); i++)
    {
        if (remap.FindMapping(objects[i]->node))
            ts->ReplaceReference(i + 2, remap.FindMapping(objects[i]->node));
        else
            ts->ReplaceReference(i + 2, objects[i]->node);
    }

    BaseClone(this, ts, remap);
    return ts;
}
