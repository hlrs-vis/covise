/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: cover.cpp

    DESCRIPTION:  A VRML COVER Helper
 
    CREATED BY: Uwe Woessner
  
    HISTORY: created 3.4.2003
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "cover.h"

//------------------------------------------------------

class COVERClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new COVERObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_COVER_SENSOR_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(COVER_CLASS_ID1, COVER_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static COVERClassDesc COVERDesc;

ClassDesc *GetCOVERDesc() { return &COVERDesc; }
TCHAR *somekeys[] = { _T("1"), _T("2"), _T("3"), _T("4"), _T("5"), _T("6"), _T("7"), _T("8"), _T("9"), _T("0"), _T("F1"), _T("F2"), _T("F3"), _T("F4"), _T("F5"), _T("F6"), _T("F7"), _T("F8"), _T("F9"), _T("F10"), _T("F11"), _T("F12"), _T("a"), _T("b"), _T("c"), _T("d"), _T("e"), _T("f"), _T("g"), _T("h"), _T("i"), _T("j"), _T("k"), _T("l"), _T("m"), _T("n"), _T("o"), _T("p"), _T("q"), _T("r"), _T("s"), _T("t"), _T("u"), _T("v"), _T("w"), _T("x"), _T("y"), _T("z"), _T("A"), _T("B"), _T("C"), _T("D"), _T("E"), _T("F"), _T("G"), _T("H"), _T("I"), _T("J"), _T("K"), _T("L"), _T("M"), _T("N"), _T("O"), _T("P"), _T("Q"), _T("R"), _T("S"), _T("T"), _T("U"), _T("V"), _T("W"), _T("X"), _T("Y"), _T("Z"), _T("!"), _T("<"), _T("$"), _T("-"), _T("&"), _T("/"), _T("("), _T(")"), _T("="), _T("?"), _T("ß"), _T("+"), _T("*"), _T("~"), _T("#"), _T("0"), _T("1"), _T("2"), _T("3"), _T("4"), _T("5"), _T("6"), _T("7"), _T("8"), _T("9"), _T("0"), _T("1"), _T("2"), _T("3"), _T("4"), _T("5"), _T("6"), _T("7"), _T("8"), _T("9"), _T("0") };
int COVERObj::KeyIndex = 0;
// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *COVERObject::TrackedObjectPickButton = NULL;
ICustButton *COVERObject::KeyboardPickButton = NULL;

HWND COVERObject::hRollup = NULL;
int COVERObject::dlgPrevSel = -1;

class AvaratTargetPick : public PickModeCallback
{
    COVERObject *parent;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetCOVER(COVERObject *l) { parent = l; }
};

//static AvaratTargetPick    theParentPick;
#define PARENT_PICK_MODE 1
#define COVER_PICK_MODE 2

static AvaratTargetPick thePPick;
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
        // thePick.SetCOVER(o);
        GetCOREInterface()->SetPickMode(p);
    }
}

BOOL
AvaratTargetPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                          int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(COVER_CLASS_ID1, COVER_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
AvaratTargetPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_TRIGGER));
}

void
AvaratTargetPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
AvaratTargetPick::Pick(IObjParam *ip, ViewExp *vpt)
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
AvaratTargetPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

class COVERObjPick : public PickModeCallback
{
    COVERObject *cover;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetCOVER(COVERObject *l) { cover = l; }
};

static COVERObjPick theCOVERPick;

BOOL
COVERObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                      int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(COVER_CLASS_ID1, COVER_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
COVERObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_COVER_PICK_MODE));
}

void
COVERObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
COVERObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < cover->objects.Count(); i++)
        {
            if (cover->objects[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        // Don't allow a loop.  001129  --prs.
        if (node->TestForLoop(FOREVER, cover) != REF_SUCCEED)
            return FALSE;

        COVERObj *obj = new COVERObj(node);
        int id = cover->objects.Append(1, &obj);
        cover->pblock->SetValue(PB_COVER_NUMOBJS,
                                cover->iObjParams->GetTime(),
                                cover->objects.Count());

#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = cover->ReplaceReference(id + 2, node);
#else
        RefResult ret = cover->MakeRefByID(FOREVER, id + 2, node);
#endif

        HWND hw = cover->hRollup;
        int ind = (int)SendMessage(GetDlgItem(hw, IDC_LIST),
                                   LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hw, IDC_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        EnableWindow(GetDlgItem(hw, IDC_DEL),
                     cover->objects.Count() > 0);
    }
    return FALSE;
}

HCURSOR
COVERObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(COVERObject *ob)
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
            COVERObj *obj = ob->objects[i];
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
                     COVERObject *th)
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

        th->KeyboardPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK));
        th->KeyboardPickButton->SetType(CBT_CHECK);
        th->KeyboardPickButton->SetButtonDownNotify(TRUE);
        th->KeyboardPickButton->SetHighlightColor(GREEN_WASH);
        th->KeyboardPickButton->SetCheck(FALSE);

        // Now we need to fill in the list box IDC_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        th->dlgPrevSel = -1;
        if (th->triggerObject)
            Static_SetText(GetDlgItem(hDlg, IDC_TRIGGER_OBJ),
                           th->triggerObject->GetName());

        //SendMessage(GetDlgItem(hDlg,IDC_KEY_CODE), WM_SETTEXT, 0, (LPARAM)th->KeysString.data());
        EnableWindow(GetDlgItem(hDlg, IDC_KEY_CODE), TRUE);
        if (pickMode)
            SetPickMode(NULL);
        return TRUE;

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(NULL);
        ReleaseICustButton(th->TrackedObjectPickButton);
        ReleaseICustButton(th->KeyboardPickButton);
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
                    th->KeyboardPickButton->SetCheck(FALSE);
                }
                theCOVERPick.SetCOVER(th);
                SetPickMode(&theCOVERPick, COVER_PICK_MODE);
                break;
            }
            break;
        case IDC_KEY_CODE:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                int len = (int)SendDlgItemMessage(hDlg, IDC_KEY_CODE, WM_GETTEXTLENGTH, 0, 0);
                TSTR temp;
                temp.Resize(len + 1);
                SendDlgItemMessage(hDlg, IDC_KEY_CODE, WM_GETTEXT, len + 1, (LPARAM)temp.data());
                int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                             LB_GETCURSEL, 0, 0);
                if (index != LB_ERR)
                {
                    COVERObj *obj = (COVERObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_GETITEMDATA, index, 0);
                    for (int i = 0; i < th->objects.Count(); i++)
                    {
                        if (obj == th->objects[i])
                        {
                            obj->keyStr = temp;
                            obj->ResetStr();

                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_DELETESTRING, index, 0);
                            int ind = (int)SendMessage(GetDlgItem(hDlg,
                                                                  IDC_LIST),
                                                       LB_ADDSTRING, 0,
                                                       (LPARAM)obj->listStr.data());
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                            SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                        LB_SETCURSEL, ind, 0);
                            /*SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_DELETESTRING,
                                           (WPARAM) index, 0);
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                          LB_INSERTSTRING, index,
                                          (LPARAM)obj->listStr.data());
                    SendDlgItemMessage(hDlg,IDC_LIST,
                                LB_SETCURSEL, index, 0);

*/

                            break;
                        }
                    }
                }
                //th->KeysString = temp;
                break;
            }
            break;
        case IDC_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                         LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                COVERObj *obj = (COVERObj *)
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
                        th->pblock->SetValue(PB_COVER_NUMOBJS,
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
                    COVERObj *obj = (COVERObj *)
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
                    COVERObj *obj = (COVERObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_GETITEMDATA, sel, 0);
                    assert(obj);
                    SendMessage(GetDlgItem(hDlg, IDC_KEY_CODE), WM_SETTEXT, 0, (LPARAM)obj->keyStr.data());
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
        case IDC_PICK_PARENT: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                thePPick.SetCOVER(th);
                SetPickMode(&thePPick, PARENT_PICK_MODE);
                break;
            }
            break;
        }
        return FALSE;
    default:
        return FALSE;
    }
    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Size
    ParamUIDesc(
        PB_COVER_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 1

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
};

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_COVER_LENGTH, CURRENT_VERSION);

class COVERParamDlgProc : public ParamMapUserDlgProc
{
public:
    COVERObject *ob;

    COVERParamDlgProc(COVERObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR COVERParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                   UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *COVERObject::pmapParam = NULL;

#if 0
IOResult
COVERObject::Load(ILoad *iload) 
{
  iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
                                                     NUM_OLD_VERSIONS,
                                                     &curVersion,this,0));
  return IO_OK;
}

#endif

void
COVERObject::BeginEditParams(IObjParam *ip, ULONG flags,
                             Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {
        // Left over from last COVER created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_COVER),
                                     _T("COVER" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new COVERParamDlgProc(this));
    }
}

void
COVERObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
    //    iObjParams = NULL;
}

COVERObject::COVERObject()
    : HelperObject()
{
    pblock = NULL;
    previousMode = NULL;
    triggerObject = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_COVER_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_COVER_SIZE, 0, 0.0f);
    pb->SetValue(PB_COVER_NUMOBJS, 0, 0);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
    objects.SetCount(0);
    BuildObjectList(this);
}

COVERObject::~COVERObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < objects.Count(); i++)
    {
        COVERObj *obj = objects[i];
        delete obj;
    }
}

IObjParam *COVERObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult COVERObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult COVERObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
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
            pblock->GetValue(PB_COVER_NUMOBJS, 0, numObjs,
                             FOREVER);
            numObjs--;
            pblock->SetValue(PB_COVER_NUMOBJS, 0, numObjs);
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
COVERObject::GetReference(int ind)
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
COVERObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
        if (pblock)
        {
            int numObjs;
            pblock->GetValue(PB_COVER_NUMOBJS, 0, numObjs,
                             FOREVER);
            TSTR *keys = new TSTR[numObjs];
            int numkeys = 0;
            int starti = 0;
            int endi = 0;
            while ((numkeys < numObjs) && (starti < KeysString.Length()))
            {
                if (KeysString[starti] == ' ')
                {
                    endi = starti + 2;
                    while ((endi < KeysString.Length()) && (KeysString[endi] != ' '))
                    {
                        endi++;
                    }
                    keys[numkeys] = KeysString.Substr(starti + 1, (endi - starti) - 1);
                    numkeys++;
                }
                starti++;
            }
            if (objects.Count() == 0)
            {
                objects.SetCount(numObjs);

                for (int i = 0; i < numObjs; i++)
                {
                    objects[i] = new COVERObj();
                    if (i < numkeys)
                        objects[i]->keyStr = keys[i];
                }
            }
            delete[] keys;
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
COVERObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
COVERObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
COVERObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
COVERObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
COVERObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
    GetMat(t, inode, vpt, tm);

    BuildMesh(t);
    mesh.buildBoundingBox();
    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
}

void
COVERObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_COVER_SIZE, t, size, FOREVER);
#include "coverob.cpp"
}

int
COVERObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_COVER_SIZE, t, radius, FOREVER);
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
COVERObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class COVERCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    COVERObject *coverObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(COVERObject *obj) { coverObject = obj; }
};

int
COVERCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            coverObject->pblock->SetValue(PB_COVER_SIZE,
                                          coverObject->iObjParams->GetTime(), radius);
            coverObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(coverObject->iObjParams->SnapAngle(ang));
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
static COVERCreateCallBack COVERCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
COVERObject::GetCreateMouseCallBack()
{
    COVERCreateCB.SetObj(this);
    return (&COVERCreateCB);
}

#define NAME_CHUNK 0xad30

IOResult
COVERObject::Save(ISave *isave)
{
    isave->BeginChunk(NAME_CHUNK);
    KeysString = _T(" ");
    for (int i = 0; i < objects.Count(); i++)
    {
        KeysString.Append(objects[i]->keyStr);
        KeysString.Append(_T(" "));
    }
    isave->WriteCString(KeysString.data());
    isave->EndChunk();

    return IO_OK;
}

IOResult
COVERObject::Load(ILoad *iload)
{
    TCHAR *txt;

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case NAME_CHUNK:
            iload->ReadCStringChunk(&txt);
            KeysString = txt;
            break;

        default:
            break;
        }
        iload->CloseChunk();
    }
    return IO_OK;
}

RefTargetHandle
COVERObject::Clone(RemapDir &remap)
{
    COVERObject *ts = new COVERObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    ts->objects.SetCount(objects.Count());
    ts->ReplaceReference(1, triggerObject);
    ts->KeysString = KeysString;
    for (int i = 0; i < objects.Count(); i++)
        ts->ReplaceReference(i + 2, objects[i]->node);
    BaseClone(this, ts, remap);
    return ts;
}
