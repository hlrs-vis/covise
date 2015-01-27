/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: switch.cpp

    DESCRIPTION:  A VRML Touch Sensor Helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 4 Sept, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

//TODO remove REF 1 (was triggerObject)

#include "vrml.h"
#include "switch.h"

//------------------------------------------------------

class SwitchClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new SwitchObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_SWITCH_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Switch_CLASS_ID1, Switch_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static SwitchClassDesc SwitchDesc;

ClassDesc *GetSwitchDesc() { return &SwitchDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *SwitchObject::ParentPickButton = NULL;

HWND SwitchObject::hRollup = NULL;
int SwitchObject::dlgPrevSel = -1;

class SwitchParentObjPick : public PickModeCallback
{
    SwitchObject *parent;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetSwitch(SwitchObject *l) { parent = l; }
};

//static SwitchParentObjPick    theParentPick;
#define PARENT_PICK_MODE 1
#define TOUCH_PICK_MODE 2

static SwitchParentObjPick thePPick;
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
        // thePick.SetSwitch(o);
        GetCOREInterface()->SetPickMode(p);
    }
}

BOOL
SwitchParentObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                             int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(Switch_CLASS_ID1, Switch_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
SwitchParentObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
SwitchParentObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
SwitchParentObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL && parent->ReplaceReference(1, node) == REF_SUCCEED)
    {

        SetPickMode(NULL);
        // parent->iObjParams->SetCommandMode(parent->previousMode);
        // parent->previousMode = NULL;
        /* parent->ParentPickButton->SetCheck(FALSE);
        HWND hw = parent->hRollup;
        Static_SetText(GetDlgItem(hw,IDC_TRIGGER_OBJ),
                       parent->triggerObject->GetName());*/
        return FALSE;
    }
    return FALSE;
}

HCURSOR
SwitchParentObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

ICustButton *SwitchObject::SwitchPickButton = NULL;

class SwitchObjPick : public PickModeCallback
{
    SwitchObject *switchSensor;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetSwitch(SwitchObject *l) { switchSensor = l; }
};

// static SwitchObjPick thePick;
static SwitchObjPick theTSPick;

BOOL
SwitchObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                       int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(Switch_CLASS_ID1, Switch_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
SwitchObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_Switch_PICK_MODE));
}

void
SwitchObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
SwitchObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < switchSensor->objects.Count(); i++)
        {
            if (switchSensor->objects[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        // Don't allow a loop.  001129  --prs.
        if (node->TestForLoop(FOREVER, switchSensor) != REF_SUCCEED)
            return FALSE;

        SwitchObj *obj = new SwitchObj(node);
        int id = switchSensor->objects.Append(1, &obj);
        switchSensor->pblock->SetValue(PB_S_NUMOBJS,
                                       switchSensor->iObjParams->GetTime(),
                                       switchSensor->objects.Count());

#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = switchSensor->ReplaceReference(id + 2, node);
#else
        RefResult ret = switchSensor->MakeRefByID(FOREVER, id + 2, node);
#endif

        HWND hw = switchSensor->hRollup;
        int ind = (int)SendMessage(GetDlgItem(hw, IDC_LIST),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hw, IDC_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        EnableWindow(GetDlgItem(hw, IDC_DEL),
                     switchSensor->objects.Count() > 0);
    }
    return FALSE;
}

HCURSOR
SwitchObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(SwitchObject *ob)
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
            SwitchObj *obj = ob->objects[i];
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
                     SwitchObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
        //        SetDlgFont( hDlg, th->iObjParams->GetAppHFont() );

        th->SwitchPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK));
        th->SwitchPickButton->SetType(CBT_CHECK);
        th->SwitchPickButton->SetButtonDownNotify(TRUE);
        th->SwitchPickButton->SetHighlightColor(GREEN_WASH);
        th->SwitchPickButton->SetCheck(FALSE);

        /* th->ParentPickButton = GetICustButton(GetDlgItem(hDlg,IDC_PICK_PARENT));
        th->ParentPickButton->SetType(CBT_CHECK);
        th->ParentPickButton->SetButtonDownNotify(TRUE);
        th->ParentPickButton->SetHighlightColor(GREEN_WASH);
        th->ParentPickButton->SetCheck(FALSE);*/

        // Now we need to fill in the list box IDC_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        //        EnableWindow(GetDlgItem(hDlg, IDC_DEL),
        //                     (th->objects.Count() > 0));
        th->dlgPrevSel = -1;
        /*if (th->triggerObject)
            Static_SetText(GetDlgItem(hDlg,IDC_TRIGGER_OBJ),
                           th->triggerObject->GetName());*/

        if (pickMode)
            SetPickMode(NULL);
        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(NULL);
        // th->iObjParams->ClearPickMode();
        // th->previousMode = NULL;
        ReleaseICustButton(th->SwitchPickButton);
        ReleaseICustButton(th->ParentPickButton);
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
                theTSPick.SetSwitch(th);
                SetPickMode(&theTSPick, TOUCH_PICK_MODE);
                /*
                if (th->previousMode) {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                } else {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    thePick.SetSwitch(th);
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
                    th->SwitchPickButton->SetCheck(FALSE);
                }
                thePPick.SetSwitch(th);
                SetPickMode(&thePPick, PARENT_PICK_MODE);
                /*
                if (th->previousMode) {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                } else {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    theParentPick.SetSwitch(th);
                    th->iObjParams->SetPickMode(&theParentPick);
                }
                */
                break;
            }
            break;
        case IDC_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                SwitchObj *obj = (SwitchObj *)
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
                        th->pblock->SetValue(PB_S_NUMOBJS,
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
        case IDC_MOVEUP:
        {
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                SwitchObj *obj = (SwitchObj *)
                    SendDlgItemMessage(hDlg, IDC_LIST,
                                       LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->objects.Count(); i++)
                {
                    if (obj == th->objects[i])
                    {
                        if ((index > 0) && (i > 0))
                        {
                            SwitchObj *oldObj = (SwitchObj *)SendMessage(GetDlgItem(hDlg, IDC_LIST), LB_GETITEMDATA,
                                                                         (WPARAM)index - 1, 0);
                            th->objects.Delete(i, 1);
                            th->objects.Insert(i - 1, 1, &obj);
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_DELETESTRING,
                                               (WPARAM)index, 0);
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_INSERTSTRING,
                                               (WPARAM)index - 1,
                                               (LPARAM)obj->listStr.data());
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETITEMDATA, (WPARAM)index - 1, (LPARAM)obj);
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETITEMDATA, (WPARAM)index, (LPARAM)oldObj);

#if MAX_PRODUCT_VERSION_MAJOR > 8
                            RefResult ret = th->ReplaceReference(i + 2, oldObj->node);
                            ret = th->ReplaceReference(i + 1, obj->node);
#else
                            RefResult ret = switchSensor->MakeRefByID(FOREVER, i + 2, oldObj->node);
                            ret = switchSensor->MakeRefByID(FOREVER, i + 1, obj->node);
#endif
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETCURSEL, (WPARAM)index - 1, 0);
                            th->dlgPrevSel = index - 1;

                            obj = (SwitchObj *)
                                SendDlgItemMessage(hDlg, IDC_LIST,
                                                   LB_GETITEMDATA, index - 1, 0);
                            assert(obj);
                        }
                    }
                }
            }
        }
        break;
        case IDC_MOVEDOWN:
        {
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                SwitchObj *obj = (SwitchObj *)
                    SendDlgItemMessage(hDlg, IDC_LIST,
                                       LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->objects.Count(); i++)
                {
                    if (obj == th->objects[i])
                    {
                        if (i < th->objects.Count() - 1)
                        {
                            SwitchObj *oldObj = (SwitchObj *)SendMessage(GetDlgItem(hDlg, IDC_LIST), LB_GETITEMDATA,
                                                                         (WPARAM)index + 1, 0);
                            th->objects.Delete(i, 1);
                            th->objects.Insert(i + 1, 1, &obj);
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_DELETESTRING,
                                               (WPARAM)index, 0);
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_INSERTSTRING,
                                               (WPARAM)index + 1,
                                               (LPARAM)obj->listStr.data());
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETITEMDATA, (WPARAM)index + 1, (LPARAM)obj);
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETITEMDATA, (WPARAM)index, (LPARAM)oldObj);

#if MAX_PRODUCT_VERSION_MAJOR > 8
                            RefResult ret = th->ReplaceReference(i + 2, oldObj->node);
                            ret = th->ReplaceReference(i + 3, obj->node);
#else
                            RefResult ret = switchSensor->MakeRefByID(FOREVER, i + 2, oldObj->node);
                            ret = switchSensor->MakeRefByID(FOREVER, i + 3, obj->node);
#endif
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETCURSEL, (WPARAM)index + 1, 0);
                            th->dlgPrevSel = index + 1;

                            obj = (SwitchObj *)
                                SendDlgItemMessage(hDlg, IDC_LIST,
                                                   LB_GETITEMDATA, index + 1, 0);
                            assert(obj);
                        }
                        break;
                    }
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
                /*               if (th->dlgPrevSel != -1) {
                    // save any editing
                    SwitchObj *obj = (SwitchObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_GETITEMDATA, th->dlgPrevSel, 0);
                    obj->ResetStr();
                    SendMessage(GetDlgItem(hDlg,IDC_LIST),
                                LB_DELETESTRING, th->dlgPrevSel, 0);
                    int ind = SendMessage(GetDlgItem(hDlg,
                                                     IDC_LIST),
                                          LB_ADDSTRING, 0,
                                          (LPARAM)obj->listStr.data());
                    SendMessage(GetDlgItem(hDlg,IDC_LIST),
                                LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                    SendMessage(GetDlgItem(hDlg,IDC_LIST),
                                LB_SETCURSEL, sel, 0);
                }
 */ th->dlgPrevSel = sel;
                if (sel >= 0)
                {
                    SwitchObj *obj = (SwitchObj *)
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
        PB_S_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // default
    ParamUIDesc(
        PB_S_DEFAULT,
        EDITTYPE_INT,
        IDC_DEFAULT_EDIT, IDC_DEFAULT_SPIN,
        -1, 100,
        1),
    // default
    ParamUIDesc(
        PB_S_ALLOW_NONE,
        TYPE_SINGLECHEKBOX,
        IDC_ALLOW_NONE),

};

#define PARAMDESC_LENGTH 3

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
};

//static ParamVersionDesc versions[] = {
//  ParamVersionDesc(descVer0,5,0),
//};

//#define NUM_OLD_VERSIONS 1

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_S_LENGTH, CURRENT_VERSION);

class SwitchParamDlgProc : public ParamMapUserDlgProc
{
public:
    SwitchObject *ob;

    SwitchParamDlgProc(SwitchObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR SwitchParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                    UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *SwitchObject::pmapParam = NULL;

#if 0
IOResult
SwitchObject::Load(ILoad *iload) 
{
  iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                     NUM_OLD_VERSIONS,
                                                     &curVersion,this,0));
  return IO_OK;
}

#endif

void
SwitchObject::BeginEditParams(IObjParam *ip, ULONG flags,
                              Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {
        // Left over from last Switch created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_SWITCH),
                                     _T("Switch" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new SwitchParamDlgProc(this));
    }
}

void
SwitchObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

SwitchObject::SwitchObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_S_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_SIZE, 0, 0.0f);
    pb->SetValue(PB_S_NUMOBJS, 0, 0);
    pb->SetValue(PB_S_DEFAULT, 0, -1);
    pb->SetValue(PB_S_ALLOW_NONE, 0, TRUE);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
    previousMode = NULL;
    // triggerObject = NULL;
    objects.SetCount(0);
    BuildObjectList(this);

    needsScript = true;
}

SwitchObject::~SwitchObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < objects.Count(); i++)
    {
        SwitchObj *obj = objects[i];
        delete obj;
    }
}

IObjParam *SwitchObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult SwitchObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                         PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult SwitchObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
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
            pblock->GetValue(PB_S_NUMOBJS, 0, numObjs,
                             FOREVER);
            numObjs--;
            pblock->SetValue(PB_S_NUMOBJS, 0, numObjs);
        }
        /*   if (hTarget == triggerObject)
            triggerObject = NULL;*/
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
SwitchObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return NULL;
    if (ind - 1 > objects.Count())
        return NULL;

    if (objects[ind - 2] == NULL)
        return NULL;
    return objects[ind - 2]->node;
}

void
SwitchObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        if (pblock)
        {
            int numObjs;
            pblock->GetValue(PB_S_NUMOBJS, 0, numObjs,
                             FOREVER);
            if (objects.Count() == 0)
            {
                objects.SetCount(numObjs);
                for (int i = 0; i < numObjs; i++)
                    objects[i] = new SwitchObj();
            }
        }
        return;
    }
    if (ind == 1)
    {
        //triggerObject = (INode*) rtarg;
        return;
    }
    if (ind - 1 > objects.Count())
        return;

    objects[ind - 2]->node = (INode *)rtarg;
    objects[ind - 2]->ResetStr();

    for (int i = 0; i < objects.Count(); i++)
        if (objects[i]->node == NULL)
            return;
    NotifyDependents(FOREVER, PART_ALL, TARGETMSG_LOADFINISHED);
}

ObjectState
SwitchObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
SwitchObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
SwitchObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
SwitchObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                               Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
SwitchObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
SwitchObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_S_SIZE, t, size, FOREVER);
#include "switchob.cpp"
    mesh.buildBoundingBox();
}

int
SwitchObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_S_SIZE, t, radius, FOREVER);
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
SwitchObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class SwitchCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    SwitchObject *switchSensorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(SwitchObject *obj) { switchSensorObject = obj; }
};

int
SwitchCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            switchSensorObject->pblock->SetValue(PB_S_SIZE,
                                                 switchSensorObject->iObjParams->GetTime(), radius);
            switchSensorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(switchSensorObject->iObjParams->SnapAngle(ang));
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
static SwitchCreateCallBack SwitchCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
SwitchObject::GetCreateMouseCallBack()
{
    SwitchCreateCB.SetObj(this);
    return (&SwitchCreateCB);
}

RefTargetHandle
SwitchObject::Clone(RemapDir &remap)
{
    SwitchObject *ts = new SwitchObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    ts->objects.SetCount(objects.Count());
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
