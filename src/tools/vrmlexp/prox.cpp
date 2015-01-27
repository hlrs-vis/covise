/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: prox.cpp

DESCRIPTION:  A VRML ProximitySensor Helper

CREATED BY: Scott Morrison

HISTORY: created 4 Sept, 1996

*> Copyright (c) 1996, All Rights Reserved.
**********************************************************************/

#include "vrml.h"
#include "prox.h"

//------------------------------------------------------

class ProxSensorClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new ProxSensorObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_PROX_SENSOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(ProxSensor_CLASS_ID1,
                                         ProxSensor_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static ProxSensorClassDesc ProxSensorDesc;

ClassDesc *GetProxSensorDesc() { return &ProxSensorDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *ProxSensorObject::ProxSensorPickButton = NULL;
ICustButton *ProxSensorObject::ProxSensorPickExitButton = NULL;

HWND ProxSensorObject::hRollup = NULL;
int ProxSensorObject::dlgPrevSel = -1;

class ProxSensorObjPick : public PickModeCallback
{
    ProxSensorObject *proxSensor;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetProxSensor(ProxSensorObject *l) { proxSensor = l; }
};

//static ProxSensorObjPick thePick;
static ProxSensorObjPick thePick;
static int pickMode = 0;
static CommandMode *lastMode = NULL;

static void
SetPickMode(ProxSensorObject *o, int whichList = 0)
{
    if (whichList == 0 || !o)
    {
        pickMode = 0;
        if (lastMode)
        {
            GetCOREInterface()->PushCommandMode(lastMode);
            lastMode = NULL;
            GetCOREInterface()->ClearPickMode();
        }
    }
    else
    {
        pickMode = whichList;
        if (!lastMode)
        {
            lastMode = GetCOREInterface()->GetCommandMode();
            thePick.SetProxSensor(o);
            GetCOREInterface()->SetPickMode(&thePick);
        }
    }
}

BOOL
ProxSensorObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                           int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(ProxSensor_CLASS_ID1, ProxSensor_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
ProxSensorObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_TouchSensor_PICK_MODE));
}

void
ProxSensorObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
ProxSensorObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < proxSensor->objects.Count(); i++)
        {
            if (proxSensor->objects[i]->node == node)
                return FALSE; // Can't click those we already have
        }
        for (int i = 0; i < proxSensor->objectsExit.Count(); i++)
        {
            if (proxSensor->objectsExit[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        // Don't allow a loop.  001129  --prs.
        if (node->TestForLoop(FOREVER, proxSensor) != REF_SUCCEED)
            return FALSE;

        ProxSensorObj *obj = new ProxSensorObj(node);
        RefResult ret;
        HWND hw = proxSensor->hRollup;
        int ind;
        if (pickMode == IDC_LIST_EXIT)
        {
            int id = proxSensor->objectsExit.Append(1, &obj);
            proxSensor->pblock->SetValue(PB_PS_NUMOBJS_EXIT,
                                         proxSensor->iObjParams->GetTime(),
                                         proxSensor->objectsExit.Count());
#if MAX_PRODUCT_VERSION_MAJOR > 8
            ret = proxSensor->ReplaceReference(id + 1 + proxSensor->objects.Count(), node);
#else
            ret = proxSensor->MakeRefByID(FOREVER, id + 1 + proxSensor->objects.Count(), node);
#endif
            ind = (int)SendMessage(GetDlgItem(hw, IDC_LIST_EXIT),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(hw, IDC_LIST_EXIT),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
            EnableWindow(GetDlgItem(hw, IDC_DEL_EXIT),
                         proxSensor->objects.Count() > 0);
        }
        else
        {
            int id = proxSensor->objects.Append(1, &obj);
            proxSensor->pblock->SetValue(PB_PS_NUMOBJS,
                                         proxSensor->iObjParams->GetTime(),
                                         proxSensor->objects.Count());
#if MAX_PRODUCT_VERSION_MAJOR > 8
            ret = proxSensor->ReplaceReference(id + 1, node);
#else
            ret = proxSensor->MakeRefByID(FOREVER, id + 1, node);
#endif
            ind = (int)SendMessage(GetDlgItem(hw, IDC_LIST),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(hw, IDC_LIST),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
            EnableWindow(GetDlgItem(hw, IDC_DEL),
                         proxSensor->objects.Count() > 0);
        }
    }
    return FALSE;
}

HCURSOR
ProxSensorObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(ProxSensorObject *ob)
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
            ProxSensorObj *obj = ob->objects[i];
            obj->ResetStr(); // Make sure we're up to date

            // for now just load the name, we might want to add
            // the frame range as some point
            int ind = (int)SendMessage(GetDlgItem(ob->hRollup, IDC_LIST),
                                       LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(ob->hRollup, IDC_LIST),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        }

        count = (int)SendDlgItemMessage(ob->hRollup, IDC_LIST_EXIT,
                                        LB_GETCOUNT, 0, 0);

        // First remove any objects on the list
        for (i = count - 1; i >= 0; i--)
            SendDlgItemMessage(ob->hRollup, IDC_LIST_EXIT,
                               LB_DELETESTRING, (WPARAM)i, 0);

        for (i = 0; i < ob->objectsExit.Count(); i++)
        {
            ProxSensorObj *obj = ob->objectsExit[i];
            obj->ResetStr(); // Make sure we're up to date

            // for now just load the name, we might want to add
            // the frame range as some point
            int ind = (int)SendMessage(GetDlgItem(ob->hRollup, IDC_LIST_EXIT),
                                       LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(ob->hRollup, IDC_LIST_EXIT),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        }
    }
}

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     ProxSensorObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:

        th->ProxSensorPickButton = GetICustButton(GetDlgItem(hDlg,
                                                             IDC_PICK));
        th->ProxSensorPickButton->SetType(CBT_CHECK);
        th->ProxSensorPickButton->SetButtonDownNotify(TRUE);
        th->ProxSensorPickButton->SetHighlightColor(GREEN_WASH);
        th->ProxSensorPickButton->SetCheck(FALSE);

        th->ProxSensorPickExitButton = GetICustButton(GetDlgItem(hDlg,
                                                                 IDC_PICK_EXIT));
        th->ProxSensorPickExitButton->SetType(CBT_CHECK);
        th->ProxSensorPickExitButton->SetButtonDownNotify(TRUE);
        th->ProxSensorPickExitButton->SetHighlightColor(GREEN_WASH);
        th->ProxSensorPickExitButton->SetCheck(FALSE);

        // Now we need to fill in the list box IDC_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        //        EnableWindow(GetDlgItem(hDlg, IDC_DEL),
        //                     (th->objects.Count() > 0));
        th->dlgPrevSel = -1;

        if (pickMode)
            SetPickMode(th);

        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(th);
        //th->iObjParams->ClearPickMode();
        //th->previousMode = NULL;
        ReleaseICustButton(th->ProxSensorPickButton);
        ReleaseICustButton(th->ProxSensorPickExitButton);
        return FALSE;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_PICK: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (th->ProxSensorPickExitButton->IsChecked())
                {
                    SetPickMode(NULL, 0);
                    th->ProxSensorPickExitButton->SetCheck(FALSE);
                }
                if (!th->ProxSensorPickButton->IsChecked())
                    SetPickMode(th, IDC_LIST);
                else
                    SetPickMode(NULL, 0);

                break;
            }
            break;
        case IDC_PICK_EXIT: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (th->ProxSensorPickButton->IsChecked())
                {
                    SetPickMode(NULL, 0);
                    th->ProxSensorPickButton->SetCheck(FALSE);
                }
                if (!th->ProxSensorPickExitButton->IsChecked())
                    SetPickMode(th, IDC_LIST_EXIT);
                else
                    SetPickMode(NULL, 0);
                break;
            }
            break;
        case IDC_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                ProxSensorObj *obj = (ProxSensorObj *)
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
                        // remove the object from the table
                        th->DeleteReference(i + 1);
                        th->objects.Delete(i, 1);
                        th->pblock->SetValue(PB_PS_NUMOBJS,
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
        case IDC_DEL_EXIT:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST_EXIT),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                ProxSensorObj *obj = (ProxSensorObj *)
                    SendDlgItemMessage(hDlg, IDC_LIST_EXIT,
                                       LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->objectsExit.Count(); i++)
                {
                    if (obj == th->objectsExit[i])
                    {
                        // remove the item from the list
                        SendDlgItemMessage(hDlg, IDC_LIST_EXIT,
                                           LB_DELETESTRING,
                                           (WPARAM)index, 0);
                        th->dlgPrevSel = -1;
                        // remove the object from the table
                        th->DeleteReference(i + 1);
                        th->objectsExit.Delete(i, 1);
                        th->pblock->SetValue(PB_PS_NUMOBJS,
                                             th->iObjParams->GetTime(),
                                             th->objectsExit.Count());
                        break;
                    }
                }
                EnableWindow(GetDlgItem(hDlg, IDC_DEL_EXIT),
                             (th->objectsExit.Count() > 0));
                if (th->objectsExit.Count() <= 0)
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
                    ProxSensorObj *obj = (ProxSensorObj *)
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
                    ProxSensorObj *obj = (ProxSensorObj *)
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
        case IDC_LIST_EXIT:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST_EXIT),
                                           LB_GETCURSEL, 0, 0);
                if (th->dlgPrevSel != -1)
                {
                    // save any editing
                    ProxSensorObj *obj = (ProxSensorObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST_EXIT,
                                           LB_GETITEMDATA, th->dlgPrevSel, 0);
                    obj->ResetStr();
                    SendMessage(GetDlgItem(hDlg, IDC_LIST_EXIT),
                                LB_DELETESTRING, th->dlgPrevSel, 0);
                    int ind = (int)SendMessage(GetDlgItem(hDlg,
                                                          IDC_LIST_EXIT),
                                               LB_ADDSTRING, 0,
                                               (LPARAM)obj->listStr.data());
                    SendMessage(GetDlgItem(hDlg, IDC_LIST_EXIT),
                                LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                    SendMessage(GetDlgItem(hDlg, IDC_LIST_EXIT),
                                LB_SETCURSEL, sel, 0);
                }
                th->dlgPrevSel = sel;
                if (sel >= 0)
                {
                    ProxSensorObj *obj = (ProxSensorObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST_EXIT,
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
    // Length
    ParamUIDesc(
        PB_PS_LENGTH,
        EDITTYPE_UNIVERSE,
        IDC_LENGTH_EDIT, IDC_LENGTH_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Width
    ParamUIDesc(
        PB_PS_WIDTH,
        EDITTYPE_UNIVERSE,
        IDC_WIDTH_EDIT, IDC_WIDTH_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Height
    ParamUIDesc(
        PB_PS_HEIGHT,
        EDITTYPE_UNIVERSE,
        IDC_HEIGHT_EDIT, IDC_HEIGHT_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Enabled
    ParamUIDesc(PB_PS_ENABLED, TYPE_SINGLECHEKBOX, IDC_ENABLE),

};

#define PARAMDESC_LENGTH 4

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_FLOAT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
    { TYPE_INT, NULL, FALSE, 4 },
    { TYPE_INT, NULL, FALSE, 5 },
};

//static ParamVersionDesc versions[] = {
//  ParamVersionDesc(descVer0,5,0),
//};

//#define NUM_OLD_VERSIONS 1

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_PS_LENGTH, CURRENT_VERSION);

class ProxSensorParamDlgProc : public ParamMapUserDlgProc
{
public:
    ProxSensorObject *ob;

    ProxSensorParamDlgProc(ProxSensorObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR ProxSensorParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                        UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *ProxSensorObject::pmapParam = NULL;

#if 0
IOResult
ProxSensorObject::Load(ILoad *iload) 
{
   iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
      NUM_OLD_VERSIONS,
      &curVersion,this,0));
   return IO_OK;
}

#endif

void
ProxSensorObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                  Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {
        // Left over from last ProxSensor created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_PROX_SENSOR), _T("Prox Sensor"),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new ProxSensorParamDlgProc(this));
    }
}

void
ProxSensorObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

ProxSensorObject::ProxSensorObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_PS_PB_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_PS_LENGTH, 0, 0.0f);
    pb->SetValue(PB_PS_WIDTH, 0, 0.0f);
    pb->SetValue(PB_PS_HEIGHT, 0, 0.0f);
    pb->SetValue(PB_PS_ENABLED, 0, TRUE);
    pb->SetValue(PB_PS_NUMOBJS, 0, 0);
    pb->SetValue(PB_PS_NUMOBJS_EXIT, 0, 0);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
    previousMode = NULL;
    objects.SetCount(0);
    BuildObjectList(this);
}

ProxSensorObject::~ProxSensorObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < objects.Count(); i++)
    {
        ProxSensorObj *obj = objects[i];
        delete obj;
    }
    for (int i = 0; i < objectsExit.Count(); i++)
    {
        ProxSensorObj *obj = objectsExit[i];
        delete obj;
    }
}

IObjParam *ProxSensorObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult ProxSensorObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult ProxSensorObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
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
                int numObjs;
                pblock->GetValue(PB_PS_NUMOBJS, 0, numObjs,
                                 FOREVER);
                numObjs--;
                pblock->SetValue(PB_PS_NUMOBJS, 0, numObjs);
            }
        }
        for (i = 0; i < objectsExit.Count(); i++)
        {
            if (objectsExit[i]->node == hTarget)
            {
                // Do I need to remove the reference? FIXME
                objectsExit.Delete(i, 1);
                int numObjs;
                pblock->GetValue(PB_PS_NUMOBJS_EXIT, 0, numObjs,
                                 FOREVER);
                numObjs--;
                pblock->SetValue(PB_PS_NUMOBJS_EXIT, 0, numObjs);
            }
        }
        break;
    case REFMSG_NODE_NAMECHANGE:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < objectsExit.Count(); i++)
        {
            if (objectsExit[i]->node == hTarget)
            {
                // Found it
                objectsExit[i]->ResetStr();
                break;
            }
        }
        for (i = 0; i < objects.Count(); i++)
        {
            if (objectsExit[i]->node == hTarget)
            {
                // Found it
                objectsExit[i]->ResetStr();
                break;
            }
        }
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
ProxSensorObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    if (ind > objects.Count() + objectsExit.Count())
        return NULL;

    if (ind > objects.Count())
    {
        if (objectsExit[ind - objects.Count() - 1] == NULL)
            return NULL;
        return objectsExit[ind - objects.Count() - 1]->node;
    }
    if (objects[ind - 1] == NULL)
        return NULL;
    return objects[ind - 1]->node;
}

void
ProxSensorObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        if (pblock)
        {
            int numObjs;
            pblock->GetValue(PB_PS_NUMOBJS, 0, numObjs,
                             FOREVER);
            if (objects.Count() == 0)
            {
                objects.SetCount(numObjs);
                for (int i = 0; i < numObjs; i++)
                    objects[i] = new ProxSensorObj();
            }
            pblock->GetValue(PB_PS_NUMOBJS_EXIT, 0, numObjs,
                             FOREVER);
            if (objectsExit.Count() == 0)
            {
                objectsExit.SetCount(numObjs);
                for (int i = 0; i < numObjs; i++)
                    objectsExit[i] = new ProxSensorObj();
            }
        }
        return;
    }
    else if (ind > objects.Count())
    {
        if (ind > objects.Count() + objectsExit.Count())
        {
            return;
        }
        objectsExit[ind - 1 - objects.Count()]->node = (INode *)rtarg;
        objectsExit[ind - 1 - objects.Count()]->ResetStr();
    }
    else
    {

        objects[ind - 1]->node = (INode *)rtarg;
        objects[ind - 1]->ResetStr();
    }
}

ObjectState
ProxSensorObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
ProxSensorObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
ProxSensorObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
ProxSensorObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                   Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
ProxSensorObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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

static void
MakeQuad(Face *f, int a, int b, int c, int d, int sg)
{
    f[0].setVerts(a, b, c);
    f[0].setSmGroup(sg);
    f[0].setEdgeVisFlags(1, 1, 0);
    f[1].setVerts(c, d, a);
    f[1].setSmGroup(sg);
    f[1].setEdgeVisFlags(1, 1, 0);
}

void
ProxSensorObject::BuildMesh(TimeValue t)
{
    int nverts = 8;
    int nfaces = 12;
    Point3 va;
    Point3 vb;

    float length, width, height;
    pblock->GetValue(PB_PS_LENGTH, t, length, FOREVER);
    pblock->GetValue(PB_PS_WIDTH, t, width, FOREVER);
    pblock->GetValue(PB_PS_HEIGHT, t, height, FOREVER);
    float x = width / 2.0f;
    float y = length / 2.0f;
    float z = height;
    va = Point3(-x, -y, 0.0f);
    vb = Point3(x, y, z);

    mesh.setNumVerts(nverts);
    mesh.setNumFaces(nfaces);

    mesh.setVert(0, Point3(va.x, va.y, va.z));
    mesh.setVert(1, Point3(vb.x, va.y, va.z));
    mesh.setVert(2, Point3(va.x, vb.y, va.z));
    mesh.setVert(3, Point3(vb.x, vb.y, va.z));
    mesh.setVert(4, Point3(va.x, va.y, vb.z));
    mesh.setVert(5, Point3(vb.x, va.y, vb.z));
    mesh.setVert(6, Point3(va.x, vb.y, vb.z));
    mesh.setVert(7, Point3(vb.x, vb.y, vb.z));

    MakeQuad(&(mesh.faces[0]), 0, 2, 3, 1, 1);
    MakeQuad(&(mesh.faces[2]), 2, 0, 4, 6, 2);
    MakeQuad(&(mesh.faces[4]), 3, 2, 6, 7, 4);
    MakeQuad(&(mesh.faces[6]), 1, 3, 7, 5, 8);
    MakeQuad(&(mesh.faces[8]), 0, 1, 5, 4, 16);
    MakeQuad(&(mesh.faces[10]), 4, 5, 7, 6, 32);
    mesh.InvalidateGeomCache();
    mesh.EnableEdgeList(1);
    mesh.buildBoundingBox();
}

int
ProxSensorObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    BuildMesh(t);
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();

    DWORD rlim = gw->getRndLimits();
    gw->setRndLimits(GW_WIREFRAME | GW_BACKCULL);
    GetMat(t, inode, vpt, m);
    gw->setTransform(m);
    if (inode->Selected())
        gw->setColor(LINE_COLOR, 1.0f, 1.0f, 1.0f);
    else if (!inode->IsFrozen())
        gw->setColor(LINE_COLOR, 1.0f, 0.5f, 0.5f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
ProxSensorObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class ProxSensorCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0, sp1;
    Point3 p0, p1;
    ProxSensorObject *ob;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(ProxSensorObject *obj) { ob = obj; }
};

int ProxSensorCreateCallBack::proc(ViewExp *vpt, int msg, int point,
                                   int flags, IPoint2 m, Matrix3 &mat)
{
    Point3 d;
    if (msg == MOUSE_POINT || msg == MOUSE_MOVE)
    {
        switch (point)
        {
        case 0:
            sp0 = m;
            ob->pblock->SetValue(PB_PS_WIDTH, 0, 0.0f);
            ob->pblock->SetValue(PB_PS_LENGTH, 0, 0.0f);
            ob->pblock->SetValue(PB_PS_HEIGHT, 0, 0.0f);

            p0 = vpt->SnapPoint(m, m, NULL, SNAP_IN_PLANE);
            p1 = p0 + Point3(.01, .01, .01);
            mat.SetTrans(float(.5) * (p0 + p1));
            break;
        case 1:
            sp1 = m;
            p1 = vpt->SnapPoint(m, m, NULL, SNAP_IN_PLANE);
            p1.z = p0.z + (float).01;
            mat.SetTrans(float(.5) * (p0 + p1));
            d = p1 - p0;

            ob->pblock->SetValue(PB_PS_WIDTH, 0, float(fabs(d.x)));
            ob->pblock->SetValue(PB_PS_LENGTH, 0, float(fabs(d.y)));
            ob->pblock->SetValue(PB_PS_HEIGHT, 0, float(fabs(d.z)));
            ob->pmapParam->Invalidate();

            if (msg == MOUSE_POINT && (Length(sp1 - sp0) < 3 || Length(d) < 0.1f))
            {
                return CREATE_ABORT;
            }
            break;
        case 2:
            p1.z = vpt->SnapLength(vpt->GetCPDisp(p1, Point3(0, 0, 1), sp1, m));

            d = p1 - p0;

            ob->pblock->SetValue(PB_PS_WIDTH, 0, float(fabs(d.x)));
            ob->pblock->SetValue(PB_PS_LENGTH, 0, float(fabs(d.y)));
            ob->pblock->SetValue(PB_PS_HEIGHT, 0, float(fabs(d.z)));
            ob->pmapParam->Invalidate();

            if (msg == MOUSE_POINT)
            {
                return CREATE_STOP;
            }
            break;
        }
    }
    else if (msg == MOUSE_ABORT)
    {
        return CREATE_ABORT;
    }

    return TRUE;
}

// A single instance of the callback object.
static ProxSensorCreateCallBack ProxSensorCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
ProxSensorObject::GetCreateMouseCallBack()
{
    ProxSensorCreateCB.SetObj(this);
    return (&ProxSensorCreateCB);
}

RefTargetHandle
ProxSensorObject::Clone(RemapDir &remap)
{
    ProxSensorObject *ts = new ProxSensorObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    ts->objects.SetCount(objects.Count());
    for (int i = 0; i < objects.Count(); i++)
    {
        if (remap.FindMapping(objects[i]->node))
            ts->ReplaceReference(i + 1, remap.FindMapping(objects[i]->node));
        else
            ts->ReplaceReference(i + 1, objects[i]->node);
    }

    BaseClone(this, ts, remap);
    return ts;
}
