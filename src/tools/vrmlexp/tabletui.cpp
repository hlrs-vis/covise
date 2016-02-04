/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: tabletui.cpp

DESCRIPTION:  Defines a VRML 2.0 TabletUI helper

CREATED BY: Uwe Woessner

HISTORY: created 4 Apr. 2004

*> Copyright (c) 1996, All Rights Reserved.
**********************************************************************/

#include "coTabletUI.h"
#include "vrml.h"
#include "timer.h"
#include "switch.h"
#include "tabletui.h"
#include "3dsmaxport.h"

extern TCHAR *VRMLName(const TCHAR *name);

#if MAX_PRODUCT_VERSION_MAJOR > 14
#define STRTOUTF8(x) x.ToUTF8().data()
#else
#define STRTOUTF8(x) x
#endif

#if MAX_PRODUCT_VERSION_MAJOR > 14 && ! defined FASTIO
#define MSTREAMPRINTF mStream.Printf( _T

#define PRINT_HEADER(stream, val1, val2) (stream.Printf(_T("DEF %s %s {\n"), val1, val2))
#define PRINT_TAIL(stream) (stream.Printf(_T("}\n\n")))
#define PRINT_STRING(stream, val1, val2) (stream.Printf(_T("%s \"%s\" \n"), val1, val2))
#else
#define MSTREAMPRINTF fprintf((mStream),

#define PRINT_HEADER(stream, val1, val2) (fprintf(stream, ("DEF %s %s {\n"), val1, val2))
#define PRINT_TAIL(stream) (fprintf(stream, ("}\n\n")))
#define PRINT_STRING(stream, val1, val2) (fprintf(stream, ("%s \"%s\" \n"), val1, val2))
#endif

//------------------------------------------------------

class TabletUIClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new TabletUIObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_TABLETUI_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(TabletUI_CLASS_ID1, TabletUI_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static TabletUIClassDesc TabletUIDesc;
TabletUIObject *theTabletUIObject;

ClassDesc *GetTabletUIDesc() { return &TabletUIDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

HWND TabletUIObject::hRollup = NULL;
TabletUIElement *TabletUIObject::dlgPrevSel = NULL;

#define TOUCH_PICK_MODE 1

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
        // thePick.SetTabletUI(o);
        GetCOREInterface()->SetPickMode(p);
    }
}

ICustButton *TabletUIObject::TabletUIPickButton = NULL;

class TabletUIObjPick : public PickModeCallback
{
    TabletUIObject *th;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetTabletUI(TabletUIObject *l) { th = l; }
};

static TabletUIObjPick theTSPick;

BOOL
TabletUIObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                         int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(TabletUI_CLASS_ID1, TabletUI_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
TabletUIObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_TabletUI_PICK_MODE));
}

void
TabletUIObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
TabletUIObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    node = vpt->GetClosestHit();

    HWND hw = th->hRollup;
    // Selected Element
    HTREEITEM selectTreeItem = NULL;
    selectTreeItem = (HTREEITEM)SendDlgItemMessage(hw, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
    if (selectTreeItem != NULL)
    {
        TabletUIElement *el = th->getTreeElement(selectTreeItem);

        if (el->hasObjects && (node != NULL) &&
            /*!((el->type == TabletUIElement::TUIFloatSlider) && ((node->EvalWorldState(0).obj->SuperClassID() == HELPER_CLASS_ID) && !(node->EvalWorldState(0).obj->ClassID() == TimeSensorClassID))) &&*/
            !((el->type == TabletUIElement::TUIToggleButton)
              && ((node->EvalWorldState(0).obj->ClassID() != TimeSensorClassID) && (node->EvalWorldState(0).obj->ClassID() != SwitchClassID)))
            && !((el->type == TabletUIElement::TUIComboBox) && !(node->EvalWorldState(0).obj->ClassID() == SwitchClassID)))
        {

            // Check to see if we have a reference to this object already
            for (int i = 0; i < el->objects.Count(); i++)
            {
                if (el->objects[i]->node == node)
                    return FALSE; // Can't click those we already have
            }

            // Don't allow a loop.  001129  --prs.
            if (node->TestForLoop(FOREVER, th) != REF_SUCCEED)
                return FALSE;

            TabletUIObj *obj = new TabletUIObj(node);
            el->objects.Append(1, &obj);
            int numObjs;
            th->pTabletUIBlock->GetValue(PB_TUI_NUMOBJS, 0, numObjs, FOREVER);
            th->pTabletUIBlock->SetValue(PB_TUI_NUMOBJS, th->iObjParams->GetTime(), ++numObjs);

            th->UpdateRefList();

            int ind = (int)SendMessage(GetDlgItem(hw, IDC_LIST),
                                       LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
            if (el->type == TabletUIElement::TUIComboBox)
            {
                TUIParamComboBox *tuicombo = static_cast<TUIParamComboBox *>(el->paramRollout);
                tuicombo->AddSwitch(node);
            }
            /*     for ( int i = 0; i < node->NumSubs(); i++)
            SendMessage(GetDlgItem(hw,IDC_LIST),
            LB_ADDSTRING, 0, (LPARAM)(TCHAR*)node->SubAnimName(i));*/
            SendMessage(GetDlgItem(hw, IDC_LIST),
                        LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
            EnableWindow(GetDlgItem(hw, IDC_DEL),
                         el->objects.Count() > 0);
        }
        return FALSE;
    }
    return FALSE;
}

HCURSOR
TabletUIObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
TabletUIElement::BuildObjectList()
{
    int count, i;

    count = (int)SendDlgItemMessage(myObject->hRollup, IDC_LIST,
                                    LB_GETCOUNT, 0, 0);

    // First remove any objects on the list
    for (i = count - 1; i >= 0; i--)
        SendDlgItemMessage(myObject->hRollup, IDC_LIST,
                           LB_DELETESTRING, (WPARAM)i, 0);

    for (i = 0; i < objects.Count(); i++)
    {
        TabletUIObj *obj = objects[i];
        obj->ResetStr(); // Make sure we're up to date

        int ind = (int)SendMessage(GetDlgItem(myObject->hRollup, IDC_LIST),
                                   LB_ADDSTRING, 0,
                                   (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(myObject->hRollup, IDC_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
    }

    dlgObjPrevSel = -1;
}

void TabletUIObject::BuildElementList()
{
    if (hRollup)
    {
        updateElementTree(NULL);

        for (int i = 0; i < elements.Count(); i++)
        {
            TabletUIElement *el = elements[i];
            el->BuildObjectList();
        }

        EnableWindow(GetDlgItem(hRollup, IDC_TUIELEMENT_DEL),
                     (elements.Count() > 0));
    }
}

typedef struct
{
    TCHAR *elementName;
    TCHAR *shortName;
    bool implemented;
} elementType;

elementType elementTypes[] = {
    { _T("TUIButton"), _T("BTN"), true },
    { _T("TUIComboBox"), _T("CBx"), true },
    { _T("TUIEditField"), _T("EF"), false },
    { _T("TUIEditFloatField"), _T("EFl"), false },
    { _T("TUIEditIntField"), _T("EFi"), false },
    { _T("TUIFloatSlider"), _T("FSl"), true },
    { _T("TUIFrame"), _T("Fr"), true },
    { _T("TUILabel"), _T("LB"), true },
    { _T("TUIListBox"), _T("Lis"), false },
    { _T("TUIMessageBox"), _T("MB"), false },
    { _T("TUISlider"), _T("Sl"), false },
    { _T("TUISpinEditfield"), _T("SEd"), false },
    { _T("TUISplitter"), _T("Spl"), true },
    { _T("TUITab"), _T("Tab"), true },
    { _T("TUITabFolder"), _T("TF"), true },
    { _T("TUIToggleButton"), _T("TB"), true }
};

HWND TabletUIObject::ComboBox()
{
    HWND cb = GetDlgItem(hRollup, IDC_COMBO1);
    ComboBox_ResetContent(cb);

    int i = 0;
    while ((i < elements.Count()) && (abs(elements[i]->type) != TabletUIElement::TUITab))
        i++;

    if (i == elements.Count())
    {
        ComboBox_AddString(cb, elementTypes[TabletUIElement::TUITab].elementName);
        ComboBox_AddString(cb, elementTypes[TabletUIElement::TUITabFolder].elementName);
    }
    else
        for (i = 0; i < 16; i++)
            if (elementTypes[i].implemented)
                ComboBox_AddString(cb, elementTypes[i].elementName);

    return cb;
}

Tab<int> getValidTypes(int type)
{
    Tab<int> types;

    switch (abs(type))
    {
    case TabletUIElement::TUISplitter:
    {
        int ptypes[] = { TabletUIElement::TUIFrame, TabletUIElement::TUITab };
        for (int i = 0; i < sizeof(ptypes) / sizeof(int); i++)
            types.Append(1, &ptypes[i]);
    }
    break;
    case TabletUIElement::TUITab:
    {
        int ptypes[] = { TabletUIElement::TUIFrame, TabletUIElement::TUISplitter, TabletUIElement::TUITabFolder };
        for (int i = 0; i < sizeof(ptypes) / sizeof(int); i++)
            types.Append(1, &ptypes[i]);
    }
    break;
    case TabletUIElement::TUITabFolder:
    {
        int ptypes[] = { TabletUIElement::TUIFrame, TabletUIElement::TUISplitter };
        for (int i = 0; i < sizeof(ptypes) / sizeof(int); i++)
            types.Append(1, &ptypes[i]);
    }
    break;
    default:
    {
        int ptypes[] = { TabletUIElement::TUIFrame, TabletUIElement::TUISplitter, TabletUIElement::TUITab };
        for (int i = 0; i < sizeof(ptypes) / sizeof(int); i++)
            types.Append(1, &ptypes[i]);
    }
    }

    return types;
}

void TabletUIObject::ParentComboBox(int type, TabletUIElement *el)
{
    Tab<int> validType = getValidTypes(type);
    Tab<int> parentType;
    if (el != NULL)
        if (el->parent != NULL)
            parentType = getValidTypes(el->parent->type);
        else
        {
            int ptypes[] = { TabletUIElement::TUITabFolder, TabletUIElement::TUITab };
            for (int i = 0; i < sizeof(ptypes) / sizeof(int); i++)
                parentType.Append(1, &ptypes[i]);
        }

    HWND pcb = GetDlgItem(hRollup, IDC_PARENT_COMBOBOX);

    ComboBox_ResetContent(pcb);
    int first = -1;
    for (int j = 0; j < validType.Count(); j++)
    {
        if (validType[j] == TabletUIElement::TUITabFolder)
            ComboBox_InsertString(pcb, 0, _T("MainFolder"));

        for (int i = elements.Count() - 1; i >= 0; i--)
            if ((elements[i] != el) && (abs(elements[i]->type) == validType[j]))
            {
                int k = 0;
                bool childFound = false;
                if ((el != NULL) && (childFound = el->searchChild(elements[i], false)))
                    for (; k < parentType.Count(); k++)
                        if (abs(elements[i]->type) == parentType[k])
                            break;

                if (!childFound || (k != parentType.Count()))
                {
                    int index = ComboBox_InsertString(pcb, -1, elements[i]->name);
                    TabletUIElement *el = elements[i];
                    ComboBox_SetItemData(pcb, index, (LPARAM)el);
                    first = i;
                }
            }

        if (first >= 0)
            ComboBox_SelectString(pcb, 0, elements[first]->name);
    }
}

void TabletUIObject::deselectElement(TabletUIElement *el)
{

    ICustEdit *edit = GetICustEdit(GetDlgItem(hRollup, IDC_TUIELEMENTNAME_EDIT));

    if (edit != NULL)
    {
        if (el != NULL)
        {
            edit->SetText(el->name.data());
            for (int i = el->objects.Count() - 1; i >= 0; i--)
            {
                // remove all objects
                SendDlgItemMessage(hRollup, IDC_LIST,
                                   LB_DELETESTRING,
                                   (WPARAM)i, 0);
            }
            if (el->paramRollout)
                if (el->paramRollout->pTUIParamMap != NULL)
                    el->paramRollout->EndEditParams(iObjParams, NULL);
        }
        else
            edit->SetText(_T(""));

        ReleaseICustEdit(edit);
    }
    dlgPrevSel = NULL;
}

void TabletUIObject::selectElement(TabletUIElement *el)
{
    assert(el);

    ICustEdit *edit = GetICustEdit(GetDlgItem(hRollup, IDC_TUIELEMENTNAME_EDIT));
    edit->SetText(el->name.data());
    ReleaseICustEdit(edit);

    ParentComboBox(el->type, el);
    HWND pcb = GetDlgItem(hRollup, IDC_PARENT_COMBOBOX);
    if (el->parent != NULL)
        ComboBox_SelectString(pcb, 0, el->parent->name);
    else
        ComboBox_SelectString(pcb, 0, _T("MainFolder"));

    for (int i = 0; i < el->objects.Count(); i++)
    {
        TabletUIObj *obj = el->objects[i];
        obj->ResetStr(); // Make sure we're up to date

        int ind = (int)SendMessage(GetDlgItem(hRollup, IDC_LIST),
                                   LB_ADDSTRING, 0,
                                   (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hRollup, IDC_LIST),
                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
    }

    if (el->paramRollout)
    {
        el->paramRollout->myElem = el;
        el->paramRollout->BeginEditParams(iObjParams, NULL);
    }

    dlgPrevSel = el;
    EnableWindow(GetDlgItem(hRollup, IDC_DEL),
                 el->objects.Count() > 0);
    EnableWindow(GetDlgItem(hRollup, IDC_PICK), el->hasObjects);
}

void TabletUIElement::delObjects(HWND hDlg, int elIndex)
{

    int numObjs;
    myObject->pTabletUIBlock->GetValue(PB_TUI_NUMOBJS,
                                       0, numObjs, FOREVER);
    if (objects.Count() > 0)
    {

        numObjs -= objects.Count();
        myObject->pTabletUIBlock->SetValue(PB_TUI_NUMOBJS,
                                           myObject->iObjParams->GetTime(),
                                           numObjs);

        // remove all objects
        for (int j = objects.Count() - 1; j >= 0; j--)
        {
            SendDlgItemMessage(hDlg, IDC_LIST,
                               LB_DELETESTRING,
                               (WPARAM)j, 0);
            // remove the object from the table
            TabletUIObj *obj = objects[j];
            objects.Delete(j, 1);
            delete obj;
            obj = NULL;
        }
    }
}

void TabletUIObject::delElement(int index, TabletUIElement *el)
{

    // Delete Children
    for (int i = 0; i < el->children.Count();)
        for (int j = 0; j < elements.Count(); j++)
            if (el->children[i] == elements[j])
            {
                delElement(j, elements[j]);
                break;
            }

    //Delete Element
    el->delObjects(hRollup, index);
    TabletUIElement *parent = el->parent;
    if (parent != NULL)
        for (int l = 0; l < parent->children.Count(); l++)
            if (parent->children[l] == el)
                parent->children.Delete(l, 1);

    int numElems;
    pTabletUIBlock->GetValue(PB_TUI_NUMELEMS,
                             0, numElems, FOREVER);

    pTabletUIBlock->SetValue(PB_TUI_NUMELEMS,
                             theTabletUIObject->iObjParams->GetTime(),
                             numElems - 1);

    for (int k = 0; k < elements.Count(); k++)
        if (elements[k] == el)
        {
            elements.Delete(k, 1);
            break;
        }
    delete el;
    el = NULL;
}

void TabletUIObject::delReference(int index, TabletUIElement *el)
{

    // remove references to Children
    for (int i = 0; i < el->children.Count(); i++)
        for (int j = 0; j < elements.Count(); j++)
            if (el->children[i] == elements[j])
            {
                delReference(j, elements[j]);
                break;
            }

    //remove References to objects and Element
    int objIndex = 1;
    int k = 0;
    while (k < index)
        objIndex += elements[k++]->objects.Count();

    // Remove the reference to obj->node
    for (int l = el->objects.Count() - 1; l >= 0; l--)
        ReplaceReference(objIndex + l, NULL);

    pTabletUIBlock->GetValue(PB_TUI_NUMOBJS, 0, k, FOREVER);
    if (el->paramRollout != NULL)
        ReplaceReference(1 + k + index, NULL);
}

TabletUIElement *TabletUIObject::addTUIElement(TabletUIObject *th, int selection, TabletUIElement *parentEl, TCHAR *name)
{

    TCHAR *tuielem = new TCHAR[100];

    if (name == NULL)
    {
        unsigned int nr = 1;
        int i;
        do
        {
            _stprintf(tuielem, _T("%s%02d"), elementTypes[selection].elementName, nr++);
            for (i = 0; i < th->elements.Count(); i++)
            {
                if (_tcscmp(th->elements[i]->name, tuielem) == 0)
                    break;
            }

        } while (i != th->elements.Count());
    }
    else
        _tcscpy(tuielem, name);

    TabletUIElement *el = new TabletUIElement(th, tuielem, selection, 0);

    if (el != NULL)
    {
        el->parent = parentEl;
        if (parentEl)
            parentEl->children.Append(1, &el);

        th->addcoTabletUIElement(el);

        return el;
    }
    else
        return NULL;
}

void TabletUIObject::addTreeItem(TabletUIElement *el, int neighbor, HWND hDlg)
{
    TVINSERTSTRUCT treeStruct;

    if (el->parent == NULL)
    {
        treeStruct.hInsertAfter = TVI_ROOT;
        treeStruct.hParent = NULL;
    }
    else
    {
        if (neighbor > 0)
            treeStruct.hInsertAfter = el->parent->children[0]->treeItem;
        else
            treeStruct.hInsertAfter = TVI_FIRST;
        treeStruct.hParent = el->parent->treeItem;
    }

    treeStruct.item.mask = TVIF_TEXT;
    treeStruct.item.pszText = (TCHAR *)el->name.data();

    el->treeItem = (HTREEITEM)SendMessage(hDlg, TVM_INSERTITEM, 0, (LPARAM)&treeStruct);
    TreeView_SetIndent(hDlg, 1);
}

void TabletUIObject::clearTree(HWND hDlg)
{
    int TreeCount = TreeView_GetCount(hDlg);
    for (int i = 0; i <= TreeCount; i++)
        TreeView_DeleteAllItems(hDlg);

    for (int i = 0; i < elements.Count(); i++)
        if (elements[i]->treeItem != NULL)
            elements[i]->treeItem = NULL;
};

void TabletUIObject::updateElementTree(TabletUIElement *elSelect)
{
    HWND hDlg = GetDlgItem(hRollup, IDC_TREE1);
    clearTree(hDlg);

    for (int i = 0; i < elements.Count(); i++)
        if (elements[i]->parent == NULL)
        {
            TabletUIElement *root = elements[i];
            addTreeItem(root, 0, hDlg);
            TabletUIElement *el = root;
            int j = 0;
            while (!((el == root) && (j == el->children.Count())))
            {
                if (j < el->children.Count())
                {
                    el = el->children[j];
                    addTreeItem(el, j - 1, hDlg);
                }
                else
                {
                    el = el->parent;
                }
                j = 0;
                while ((j < el->children.Count()) && (el->children[j]->treeItem != NULL))
                    j++;
            }
        }

    if (elSelect != NULL)
        TreeView_SelectItem(hDlg, elSelect->treeItem);
    else
    {
        TreeView_SelectItem(hDlg, NULL);
        if (dlgPrevSel == NULL)
            deselectElement(NULL);
    }
}

TabletUIElement *TabletUIObject::getTreeElement(HTREEITEM selectTreeItem)
{
    int i = 0;
    for (; i < elements.Count(); i++)
        if (elements[i]->treeItem == selectTreeItem)
            return elements[i];

    return NULL;
}

bool TabletUIElement::searchChild(TabletUIElement *searchElem, bool remove)
{
    for (int i = 0; i < children.Count(); i++)
    {
        TabletUIElement *el = children[i];
        if (el == searchElem)
        {
            if (remove)
                children.Delete(i, 1);
            return true;
        }
        else if (el->searchChild(searchElem, remove))
            return true;
    }
    return false;
}

static WNDPROC ParentSelWndProc = NULL;

static LRESULT CALLBACK ParentSelSubWndProc(
    HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_SETFOCUS:
        DisableAccelerators();
        if (theTabletUIObject->iObjParams)
            theTabletUIObject->iObjParams->UnRegisterDlgWnd(theTabletUIObject->hRollup);
        break;
    case WM_KILLFOCUS:
        EnableAccelerators();
        if (theTabletUIObject->iObjParams)
            theTabletUIObject->iObjParams->RegisterDlgWnd(theTabletUIObject->hRollup);
        break;

    case WM_CHAR:
        if (wParam == 13)
        {
            TCHAR buf[256];
            HWND hCombo = GetParent(hWnd);
            LRESULT res;
            GetWindowText(hWnd, buf, 256);
            if (CB_ERR != (res = SendMessage(hCombo, CB_FINDSTRINGEXACT, 0, (LPARAM)buf)))
            {
                // String is already in the list.
                SendMessage(hCombo, CB_SETCURSEL, res, 0);
                SendMessage(GetParent(hCombo), WM_COMMAND,
                            MAKEWPARAM(GetWindowLongPtr(hCombo, GWLP_ID), CBN_SELCHANGE),
                            (LPARAM)hCombo);
            }
            else
            {
                SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)buf);
                SendMessage(hWnd, EM_SETSEL, 0, (WPARAM)((INT)-1));
                TabletUIElement *elnew = theTabletUIObject->addTUIElement(theTabletUIObject, -(TabletUIElement::TUITab), NULL, buf);
                theTabletUIObject->updateElementTree(elnew);

                HWND cb = GetDlgItem(theTabletUIObject->hRollup, IDC_COMBO1);
                if (ComboBox_GetCount(cb) == 2)
                {
                    ComboBox_ResetContent(cb);
                    for (int i = 0; i < 16; i++)
                        if (elementTypes[i].implemented)
                            ComboBox_AddString(cb, elementTypes[i].elementName);
                }
            }
            return 0;
        }
        break;
    }
    return CallWindowProc(ParentSelWndProc, hWnd, message, wParam, lParam);
}

static BOOL CALLBACK EnumChildren(HWND hwnd, LPARAM lParam)
{
    ParentSelWndProc = DLSetWindowLongPtr(hwnd, ParentSelSubWndProc);
    return FALSE;
}

void SubClassParentSel(HWND hParentSel)
{
    EnumChildWindows(hParentSel, EnumChildren, 0);
}

HWND hEdit;

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     TabletUIObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        th->hRollup = hDlg;
        HWND cb = th->ComboBox();

        int type;
        th->pTabletUIBlock->GetValue(PB_TUI_TYPE, th->iObjParams->GetTime(),
                                     type, FOREVER);
        ComboBox_SelectString(cb, 0, elementTypes[type].elementName);

        if (ParentSelWndProc == NULL)
            SubClassParentSel(GetDlgItem(hDlg, IDC_PARENT_COMBOBOX));

        ICustEdit *edit = GetICustEdit(GetDlgItem(hDlg, IDC_TUIELEMENTNAME_EDIT));
        edit->WantReturn(TRUE);
        ReleaseICustEdit(edit);

        th->TabletUIPickButton = GetICustButton(GetDlgItem(hDlg, IDC_PICK));
        th->TabletUIPickButton->SetType(CBT_CHECK);
        th->TabletUIPickButton->SetButtonDownNotify(TRUE);
        th->TabletUIPickButton->SetHighlightColor(GREEN_WASH);
        th->TabletUIPickButton->SetCheck(FALSE);

        // Now we need to fill in the list box IDC_LIST
        th->BuildElementList();
        th->updateTabletUI();

        HWND lb = GetDlgItem(hDlg, IDC_LIST);
        ListBox_ResetContent(lb);
        EnableWindow(GetDlgItem(hDlg, IDC_DEL),
                     false);
        th->dlgPrevSel = NULL;

        //     th->updateTUI();
        if (pickMode)
            SetPickMode(NULL);
        return TRUE;
    }

    case WM_DESTROY:
        if (pickMode)
            SetPickMode(NULL);

        ReleaseICustButton(th->TabletUIPickButton);
        return FALSE;

    case WM_MOUSEACTIVATE:
        return FALSE;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:

    case WM_MOUSEMOVE:
        return FALSE;

    case WM_CHAR:
        return FALSE;

    case WM_NOTIFY:
        switch (LOWORD(wParam))
        {
        case IDC_TREE1:
        {
            if (((LPNMHDR)lParam)->code == TVN_SELCHANGED)
            {
                HTREEITEM selectTreeItem = NULL;
                selectTreeItem = (HTREEITEM)SendDlgItemMessage(hDlg, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
                if (selectTreeItem != NULL)
                {
                    th->deselectElement(th->dlgPrevSel);

                    TabletUIElement *el = th->getTreeElement(selectTreeItem);
                    th->selectElement(el);
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
                else
                    th->deselectElement(th->dlgPrevSel);
            }
        }
        break;
        }

        return FALSE;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_TUIELEMENTNAME_EDIT:
            if (HIWORD(wParam) == EN_CHANGE)
            {
                ICustEdit *edit = GetICustEdit(GetDlgItem(hDlg, IDC_TUIELEMENTNAME_EDIT));
                TCHAR buf[256];
                edit->GetText(buf, 256);

                if (edit->GotReturn())
                {
                    HTREEITEM selectTreeItem = NULL;
                    selectTreeItem = (HTREEITEM)SendDlgItemMessage(hDlg, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
                    if (selectTreeItem != NULL)
                    {
                        TCHAR text[MAX_PATH];
                        edit->GetText(text, MAX_PATH);
                        TabletUIElement *el = th->getTreeElement(selectTreeItem);
                        if (el != NULL)
                        {
                            el->name = text;
                            th->updateElementTree(el);
                            th->updateTabletUI();
                        }
                    }
                }

                ReleaseICustEdit(edit);
            }
            break;

        case IDC_PARENT_COMBOBOX:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case LBN_SELCHANGE:
            {
                HTREEITEM selectTreeItem = NULL;
                selectTreeItem = (HTREEITEM)SendDlgItemMessage(hDlg, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
                if (selectTreeItem != NULL)
                {

                    th->clearTabletUI();
                    TabletUIElement *el = th->getTreeElement(selectTreeItem);

                    if (el != NULL)
                    {
                        TabletUIElement *oldParent = NULL;
                        if (el->parent != NULL)
                        {
                            oldParent = el->parent;
                            for (int i = 0; i < el->parent->children.Count(); i++)
                                if (el == el->parent->children[i])
                                {
                                    el->parent->children.Delete(i, 1);
                                    break;
                                }
                            el->parent = NULL;
                        }

                        int j = (int)SendMessage(GetDlgItem(hDlg, IDC_PARENT_COMBOBOX),
                                                 CB_GETCURSEL, 0, 0);
                        if (j != LB_ERR)
                        {
                            TabletUIElement *elParent = (TabletUIElement *)ComboBox_GetItemData(GetDlgItem(hDlg, IDC_PARENT_COMBOBOX), j);

                            if (elParent != NULL)
                            {
                                if (el->searchChild(elParent, true))
                                {
                                    elParent->parent = oldParent;
                                    if (oldParent != NULL)
                                        oldParent->children.Append(1, &elParent);
                                }

                                el->parent = elParent;
                                elParent->children.Append(1, &el);
                            }
                        }

                        th->updateElementTree(el);
                        th->updateTabletUI();
                    }
                }
            }
            break;
            }
            break;
        case IDC_COMBO1:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                HWND cb = GetDlgItem(hDlg, IDC_COMBO1);
                int sel = ComboBox_GetCurSel(cb);
                if (ComboBox_GetCount(cb) == 2)
                    sel += 13;
                else
                {
                    int index = 0;
                    for (int i = 0; i < 16; i++)
                        if ((index == sel) && elementTypes[i].implemented)
                        {
                            sel = i;
                            break;
                        }
                        else if (elementTypes[i].implemented)
                            index++;
                }

                TreeView_Select(GetDlgItem(hDlg, IDC_TREE1), NULL, TVGN_CARET);
                th->ParentComboBox(sel, NULL);
            }
            break;
            }
            break;
        case IDC_TUIELEMENT_CREATE:
        {

            HWND cb = GetDlgItem(hDlg, IDC_COMBO1);
            int sel = ComboBox_GetCurSel(cb);
            if (ComboBox_GetCount(cb) == 2)
            {
                sel += 13;
                if (sel == 13)
                {
                    ComboBox_ResetContent(cb);
                    for (int i = 0; i < 16; i++)
                        if (elementTypes[i].implemented)
                            ComboBox_AddString(cb, elementTypes[i].elementName);
                }
            }
            else
            {
                int index = 0;
                for (int i = 0; i < 16; i++)
                    if ((index == sel) && elementTypes[i].implemented)
                    {
                        sel = i;
                        break;
                    }
                    else if (elementTypes[i].implemented)
                        index++;
            }

            TabletUIElement *elParent = NULL;
            int j = (int)SendMessage(GetDlgItem(hDlg, IDC_PARENT_COMBOBOX),
                                     CB_GETCURSEL, 0, 0);
            if (j != LB_ERR)
                elParent = (TabletUIElement *)ComboBox_GetItemData(GetDlgItem(hDlg, IDC_PARENT_COMBOBOX), j);
            TabletUIElement *el = th->addTUIElement(th, sel, elParent, NULL);

            th->updateElementTree(el);
            EnableWindow(GetDlgItem(hDlg, IDC_TUIELEMENT_DEL),
                         (th->elements.Count() > 0));

            return TRUE;
        }
        case IDC_RECONNECT:
        {
            if (coTabletUI::tUI)
                coTabletUI::tUI->tryConnect();
            break;
        }
        break;
        case IDC_PICK: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                theTSPick.SetTabletUI(th);
                SetPickMode(&theTSPick, TOUCH_PICK_MODE);
                break;
            }
            break;

        case IDC_DEL:
        { // Delete the object from the list
            HTREEITEM selectTreeItem = NULL;
            selectTreeItem = (HTREEITEM)SendDlgItemMessage(hDlg, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
            if (selectTreeItem != NULL)
            {
                TabletUIElement *el = th->getTreeElement(selectTreeItem);
                int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                             LB_GETCURSEL, 0, 0);
                if (index != LB_ERR)
                {
                    TabletUIObj *obj = (TabletUIObj *)
                        SendDlgItemMessage(hDlg, IDC_LIST,
                                           LB_GETITEMDATA, index, 0);

                    int objIndex = 1;
                    int j = 0;
                    while ((j != th->elements.Count()) && (th->elements[j] != el))
                    {
                        objIndex += th->elements[j++]->objects.Count();
                    }

                    for (int i = 0; i < el->objects.Count(); i++)
                    {
                        if (obj == el->objects[i])
                        {
                            // remove the item from the list
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_DELETESTRING,
                                               (WPARAM)index, 0);
                            el->dlgObjPrevSel = -1;
                            if (el->type == TabletUIElement::TUIComboBox)
                            {
                                TUIParamComboBox *tuicombo = static_cast<TUIParamComboBox *>(el->paramRollout);
                                if (tuicombo->comboObjects.size() != 0)
                                    tuicombo->DelSwitch(obj->node);
                            }

                            // Remove the reference to obj->node
                            th->ReplaceReference(objIndex + i, NULL);

                            // remove the object from the table
                            el->objects.Delete(i, 1);
                            int numObjs;
                            th->pTabletUIBlock->GetValue(PB_TUI_NUMOBJS,
                                                         0, numObjs, FOREVER);
                            th->pTabletUIBlock->SetValue(PB_TUI_NUMOBJS,
                                                         th->iObjParams->GetTime(),
                                                         numObjs - 1);
                            // Remove the last reference from the list and update
                            th->UpdateRefList();
                            break;
                        }
                    }
                    EnableWindow(GetDlgItem(hDlg, IDC_DEL),
                                 (el->objects.Count() > 0));
                    if (el->objects.Count() <= 0)
                    {
                        th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                    }
                }
            }
        }
        break;
        case IDC_TUIELEMENT_DEL:
        { // Delete the object from the list
            HTREEITEM selectTreeItem = NULL;
            selectTreeItem = (HTREEITEM)SendDlgItemMessage(hDlg, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
            if (selectTreeItem != NULL)
            {
                th->clearTabletUI();

                TabletUIElement *el = th->getTreeElement(selectTreeItem);

                for (int i = 0; i < th->elements.Count(); i++)
                {
                    if (el == th->elements[i])
                    {
                        el->dlgObjPrevSel = -1;
                        if (el->paramRollout)
                            el->paramRollout->EndEditParams(th->iObjParams, NULL);

                        th->delReference(i, el);
                        th->delElement(i, el);
                        th->UpdateRefList();
                        th->ComboBox();
                        break;
                    }
                }

                th->dlgPrevSel = NULL;
                th->updateElementTree(NULL);

                EnableWindow(GetDlgItem(hDlg, IDC_TUIELEMENT_DEL),
                             (th->elements.Count() > 0));
                if (th->elements.Count() <= 0)
                {
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
                th->updateTabletUI();
            }
        }
        break;

        case IDC_LIST:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                HTREEITEM selectTreeItem = NULL;
                selectTreeItem = (HTREEITEM)SendDlgItemMessage(hDlg, IDC_TREE1, TVM_GETNEXTITEM, TVGN_CARET, (LPARAM)selectTreeItem);
                if (selectTreeItem != NULL)
                {
                    TabletUIElement *el = th->getTreeElement(selectTreeItem);
                    int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                               LB_GETCURSEL, 0, 0);
                    if (el->dlgObjPrevSel != -1)
                    {
                        // save any editing
                        TabletUIObj *obj = (TabletUIObj *)
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_GETITEMDATA, el->dlgObjPrevSel, 0);
                        obj->ResetStr();
                        SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                    LB_DELETESTRING, el->dlgObjPrevSel, 0);
                        int ind = (int)SendMessage(GetDlgItem(hDlg,
                                                              IDC_LIST),
                                                   LB_ADDSTRING, 0,
                                                   (LPARAM)obj->listStr.data());
                        SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                    LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
                        SendMessage(GetDlgItem(hDlg, IDC_LIST),
                                    LB_SETCURSEL, sel, 0);
                    }
                    el->dlgObjPrevSel = sel;
                    if (sel >= 0)
                    {
                        TabletUIObj *obj = (TabletUIObj *)
                            SendDlgItemMessage(hDlg, IDC_LIST,
                                               LB_GETITEMDATA, sel, 0);
                        assert(obj);
                    }
                    else
                    {
                    }
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
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
    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Size
    ParamUIDesc(
        PB_TUI_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_SIZE_EDIT, IDC_SIZE_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 1

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 1 },
};

#define CURRENT_VERSION 0
// Current version
static ParamVersionDesc curVersion(descVer0, PB_TUI_LENGTH, CURRENT_VERSION);

class TabletUIParamDlgProc : public ParamMapUserDlgProc
{
public:
    TabletUIObject *ob;

    TabletUIParamDlgProc(TabletUIObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR TabletUIParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                      UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *TabletUIObject::pmapParam = NULL;

#if 0
IOResult
TabletUIObject::Load(ILoad *iload) 
{
   iload->RegisterPostLoadCallback(new ParamBlockPLCB(versions,
      NUM_OLD_VERSIONS,
      &curVersion,this,0));
   return IO_OK;
}

#endif

void
TabletUIObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {
        // Left over from last TabletUI created
        pmapParam->SetParamBlock(pTabletUIBlock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pTabletUIBlock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_TABLETUI),
                                     GetString(IDS_TABLETUI_CLASS),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new TabletUIParamDlgProc(this));
    }
}

void
TabletUIObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }

    for (int i = 0; i < elements.Count(); i++)
    {
        TabletUIElement *el = elements[i];
        if (el->paramRollout)
            el->paramRollout->EndEditParams(this->iObjParams, NULL);
    }

    clearTabletUI();
}

TabletUIElement::TabletUIElement(TabletUIObject *obj, const TCHAR *newName, int select, int numObjs)
{

    name = newName;
    hasObjects = true;
    myObject = obj;
    loaded = true;

    int objCount;
    myObject->pTabletUIBlock->GetValue(PB_TUI_NUMOBJS, 0, objCount, FOREVER);
    if (numObjs == 0)
        objects.SetCount(0);
    else
        for (int j = 0; j < numObjs; j++)
        {
            TabletUIObj *obj = new TabletUIObj();
            objects.Append(1, &obj);
            objCount++;
        }
    myObject->pTabletUIBlock->SetValue(PB_TUI_NUMOBJS, 0, objCount);

    dlgObjPrevSel = -1;

    paramRollout = NULL;
    myTuiElem = NULL;
    treeItem = NULL;
    TabletUIElement *el = this;

    int id = myObject->elements.Append(1, &el);
    int numElems;
    myObject->pTabletUIBlock->GetValue(PB_TUI_NUMELEMS, 0, numElems, FOREVER);
    myObject->pTabletUIBlock->SetValue(PB_TUI_NUMELEMS, 0, ++numElems);

    switch (abs(select))
    {
    case TUIButton:
        paramRollout = new TUIParamButton;
        break;
    case TUIComboBox:
        paramRollout = new TUIParamComboBox;
        loaded = false;
        break;
    case TUIFloatSlider:
        paramRollout = new TUIParamFloatSlider;
        break;
    case TUIFrame:
        paramRollout = new TUIParamFrame;
        hasObjects = false;
        break;
    case TUILabel:
        paramRollout = new TUIParamLabel;
        hasObjects = false;
        break;
    case TUISpinEditfield:
        paramRollout = new TUIParamSpinEditField;
        break;
    case TUISplitter:
        paramRollout = new TUIParamSplitter;
        hasObjects = false;
        break;
    case TUITab:
    case TUITabFolder:
        hasObjects = false;
        break;
    case TUIToggleButton:
        paramRollout = new TUIParamToggleButton;
        break;
    default:
        break;
    }

    if (paramRollout != NULL)
        paramRollout->myElem = this;
    myObject->UpdateRefList();
    type = select;
}

TabletUIElement::~TabletUIElement()
{
    for (int i = 0; i < objects.Count(); i++)
    {
        TabletUIObj *obj = objects[i];
        delete obj;
    }

    if (paramRollout)
        delete paramRollout;
    if (myTuiElem)
    {
        delete myTuiElem;
        myTuiElem = NULL;
    }
}

TabletUIObject::TabletUIObject()
    : HelperObject()
{
    pTabletUIBlock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_TUI_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_TUI_SIZE, 0, 0.0f);
    pb->SetValue(PB_TUI_NUMOBJS, 0, 0);
    pb->SetValue(PB_TUI_NUMELEMS, 0, 0);
    pb->SetValue(PB_TUI_TYPE, 0, 0);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif

    assert(pTabletUIBlock);
    elements.SetCount(0);
    BuildElementList();

    oldType = -1;
    onload = false;

    theTabletUIObject = this;
}

TabletUIObject::~TabletUIObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < elements.Count(); i++)
    {
        TabletUIElement *el = elements[i];
        if (el->paramRollout)
            el->paramRollout->EndEditParams(this->iObjParams, NULL);
        delete el;
    }
}

bool TabletUIElement::checkElementLoaded()
{
    for (int j = 0; j < objects.Count(); j++)
        if (objects[j]->node == NULL)
            return FALSE;

    for (int j = 0; j < objects.Count(); j++)
    {
        if (paramRollout != NULL)
            if (!paramRollout->ReferenceLoad())
                return FALSE;
    }

    return TRUE;
}

int TabletUIObject::NumRefs()
{
    int count = 1;
    for (int i = 0; i < elements.Count(); i++)
        count += elements[i]->objects.Count();

    count += elements.Count();

    return count;
}

void TabletUIObject::UpdateRefList()
{
    int index = 1;
    for (int i = 0; i < elements.Count(); i++)
        for (int j = 0; j < elements[i]->objects.Count(); j++)
        {
#if MAX_PRODUCT_VERSION_MAJOR > 8
            RefResult ret = ReplaceReference(index++, elements[i]->objects[j]->node);
#else
            RefResult ret = MakeRefByID(FOREVER, index++, elements[i]->objects[j]->node);
#endif
        }

    for (int i = 0; i < elements.Count(); i++)
    {
        if (elements[i]->paramRollout != NULL)
#if MAX_PRODUCT_VERSION_MAJOR > 8
            RefResult ret = ReplaceReference(index, elements[i]->paramRollout->pTUIParamBlock);
#else
            RefResult ret = MakeRefByID(FOREVER, index, elements[i]->paramRollout->pTUIParamBlock);
#endif
        index++;
    }
}

IObjParam *TabletUIObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult TabletUIObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                           PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult TabletUIObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                           PartID &partID, RefMessage message)
#endif
{
    int i;
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
    {
        int numObjs;
        pTabletUIBlock->GetValue(PB_TUI_NUMOBJS, 0, numObjs, FOREVER);
        // Find the ID on the list and call ResetStr
        for (i = 0; i < elements.Count(); i++)
            for (int j = 0; j < elements[i]->objects.Count(); j++)
                if (elements[i]->objects[j]->node == hTarget)
                {
                    TabletUIObj *obj = elements[i]->objects[j];
                    delete obj;
                    elements[i]->objects.Delete(j, 1);
                    pTabletUIBlock->SetValue(PB_TUI_NUMOBJS, 0, --numObjs);
                    break;
                }
        UpdateRefList();
    }

    break;
    case REFMSG_NODE_NAMECHANGE:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < elements.Count(); i++)
        {
            for (int j = 0; j < elements[i]->objects.Count(); j++)
                if (elements[i]->objects[j]->node == hTarget)
                {
                    elements[i]->objects[j]->ResetStr();
                    break;
                }
        }
        break;
    case TARGETMSG_LOADFINISHED:
    {
        for (int i = 0; i < elements.Count(); i++)
            if (!elements[i]->loaded)
                elements[i]->loaded = elements[i]->checkElementLoaded();
    }
    break;
    }
    //   updateTUI();
    return REF_SUCCEED;
}

RefTargetHandle
TabletUIObject::GetReference(int ind)
{
    if (ind == 0)
        return pTabletUIBlock;

    int numObjs = 0;
    int numold;
    for (int i = 0; i < elements.Count(); i++)
    {
        numold = numObjs;
        numObjs += elements[i]->objects.Count();
        if (numObjs > ind - 1)
        {
            return (elements[i]->objects[ind - 1 - numold]->node);
            break;
        }
    }

    if ((elements[ind - numObjs - 1] == NULL) || (elements[ind - numObjs - 1]->paramRollout == NULL))
        return NULL;
    return elements[ind - numObjs - 1]->paramRollout->pTUIParamBlock;
}

void
TabletUIObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pTabletUIBlock = (IParamBlock *)rtarg;
        if (pTabletUIBlock)
        {
            int numobjs;
            pTabletUIBlock->GetValue(PB_TUI_NUMOBJS, 0, numobjs,
                                     FOREVER);

            for (int i = 0; i < elements.Count(); i++)
                for (int j = 0; j < elements[i]->objects.Count(); j++)
                    if (!elements[i]->objects[j])
                        elements[i]->objects[j] = new TabletUIObj();
        }
        return;
    }

    int numObjs = 0;
    int numold;
    for (int i = 0; i < elements.Count(); i++)
    {
        numold = numObjs;
        numObjs += elements[i]->objects.Count();
        if (numObjs > ind - 1)
        {
            elements[i]->objects[ind - 1 - numold]->node = (INode *)rtarg;
            elements[i]->objects[ind - 1 - numold]->ResetStr();
            if (!elements[i]->loaded && (elements[i]->paramRollout != NULL))
                if (elements[i]->paramRollout->pTUIParamBlock != NULL)
                    elements[i]->loaded = elements[i]->checkElementLoaded();
            return;
        }
    }

    if (ind - numObjs > elements.Count())
        return;

    if (elements[ind - numObjs - 1]->paramRollout != NULL)
    {
        elements[ind - numObjs - 1]->paramRollout->pTUIParamBlock = (IParamBlock *)rtarg;
        if (!elements[ind - numObjs - 1]->loaded)
            elements[ind - numObjs - 1]->loaded = elements[ind - numObjs - 1]->checkElementLoaded();
    }
}

ObjectState
TabletUIObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
TabletUIObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
TabletUIObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
TabletUIObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                 Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
TabletUIObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
TabletUIObject::BuildMesh(TimeValue t)
{
    float size;
    pTabletUIBlock->GetValue(PB_TUI_SIZE, t, size, FOREVER);
#include "tabletuiob.cpp"
    mesh.buildBoundingBox();
}

int
TabletUIObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pTabletUIBlock->GetValue(PB_TUI_SIZE, t, radius, FOREVER);
    if (radius <= 0.0)
        return 0;
    BuildMesh(t);
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();

    DWORD rlim = gw->getRndLimits();
    gw->setRndLimits(GW_WIREFRAME | GW_EDGES_ONLY | GW_BACKCULL);
    //   gw->setRndLimits(GW_POLY_EDGES|GW_BACKCULL);
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
TabletUIObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class TabletUICreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    TabletUIObject *tabletuiSensorObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(TabletUIObject *obj) { tabletuiSensorObject = obj; }
};

int
TabletUICreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            tabletuiSensorObject->pTabletUIBlock->SetValue(PB_TUI_SIZE,
                                                           tabletuiSensorObject->iObjParams->GetTime(), radius);
            tabletuiSensorObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(tabletuiSensorObject->iObjParams->SnapAngle(ang));
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
static TabletUICreateCallBack TabletUICreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
TabletUIObject::GetCreateMouseCallBack()
{
    TabletUICreateCB.SetObj(this);
    return (&TabletUICreateCB);
}

#define COUNT_CHUNK 0xad30
#define DATA_CHUNK 0xad31
#define NAME_CHUNK 0xad32
#define COMBO_NAME_CHUNK 0xad33
#define COMBO_NODENAME_CHUNK 0xad34
#define COMBO_OBJECTSBEGIN_CHUNK 0xad35
#define COMBO_OBJECTSEND_CHUNK 0xad36
#define SHADER_PARAM_CHUNK 0xad37

IOResult
TabletUIObject::Save(ISave *isave)
{
    ULONG written;
    int count;

    isave->BeginChunk(COUNT_CHUNK);
    count = elements.Count();
    isave->Write(&count, sizeof(int), &written);
    isave->EndChunk();

    for (int i = 0; i < elements.Count(); i++)
    {
        isave->BeginChunk(DATA_CHUNK);
        count = elements[i]->objects.Count();
        isave->Write(&count, sizeof(int), &written);
        isave->Write(&elements[i]->type, sizeof(int), &written);

        int parent_id;
        if (elements[i]->parent == NULL)
            parent_id = 0;
        else
        {
            int j = 0;
            while ((j < elements.Count()) && (elements[i]->parent != elements[j]))
                j++;
            parent_id = j + 1;
        }
        isave->Write(&parent_id, sizeof(int), &written);
        isave->EndChunk();

        isave->BeginChunk(NAME_CHUNK);
        IOResult res = isave->WriteCString(elements[i]->name.data());
        if (res != IO_OK)
            fprintf(stderr, "Error writing string");
        isave->EndChunk();

        isave->BeginChunk(SHADER_PARAM_CHUNK);
        res = isave->WriteCString(elements[i]->shaderParam.data());
        if (res != IO_OK)
            fprintf(stderr, "Error writing shaderParam");
        isave->EndChunk();

        if (elements[i]->paramRollout != NULL)
            elements[i]->paramRollout->Save(isave);
    }

    return IO_OK;
}

IOResult
TabletUIObject::Load(ILoad *iload)
{
    ULONG read;
    int count, type = 0, numObjs = 0;
    Tab<int> parent_id;
    TCHAR *buf;
    TSTR name;
    TabletUIElement *el;

    onload = true;

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case COUNT_CHUNK:
            iload->Read(&count, sizeof(int), &read);
            break;

        case DATA_CHUNK:
        {

            int id;

            iload->Read(&numObjs, sizeof(int), &read);
            iload->Read(&type, sizeof(int), &read);
            iload->Read(&id, sizeof(int), &read);
            parent_id.Append(1, &id);
        }
        break;

        case NAME_CHUNK:
        {

            IOResult res = iload->ReadCStringChunk(&buf);
            name = buf;

            el = new TabletUIElement(this, name.data(), type, numObjs);
        }
        break;
        case SHADER_PARAM_CHUNK:
        {

            IOResult res = iload->ReadCStringChunk(&buf);
            el->shaderParam = buf;
        }
        break;

        case COMBO_OBJECTSBEGIN_CHUNK:

            if (el->paramRollout != NULL)
                el->paramRollout->Load(iload);

            break;

        default:
            break;
        }
        iload->CloseChunk();
    }

    for (int i = 0; i < elements.Count(); i++)
    {
        if (parent_id[i] != 0)
        {
            elements[i]->parent = elements[parent_id[i] - 1];
            elements[parent_id[i] - 1]->children.Append(1, &elements[i]);
        }
        else
            elements[i]->parent = NULL;
    }

    //   updateTUI();
    onload = false;
    return IO_OK;
}

RefTargetHandle
TabletUIObject::Clone(RemapDir &remap)
{
    TabletUIObject *ts = new TabletUIObject();
    ts->ReplaceReference(0, pTabletUIBlock->Clone(remap));
    if (iObjParams != NULL)
        ts->iObjParams = (IObjParam *)iObjParams->CloneInterface();

    int objCount = 0;

    for (int i = 0; i < elements.Count(); i++)
    {
        TabletUIElement *el = new TabletUIElement(ts, elements[i]->name.data(), 100, elements[i]->objects.Count());
        el->shaderParam = elements[i]->name;

        RefTargetHandle tp = NULL;
        switch (abs(elements[i]->type))
        {
        case (TabletUIElement::TUIButton):
            tp = static_cast<TUIParamButton *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUIComboBox):
            tp = static_cast<TUIParamComboBox *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUIFloatSlider):
            tp = static_cast<TUIParamFloatSlider *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUIFrame):
            tp = static_cast<TUIParamFrame *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUILabel):
            tp = static_cast<TUIParamLabel *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUISpinEditfield):
            tp = static_cast<TUIParamSpinEditField *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUISplitter):
            tp = static_cast<TUIParamSplitter *>(elements[i]->paramRollout)->Clone(remap);
            break;
        case (TabletUIElement::TUIToggleButton):
            tp = static_cast<TUIParamToggleButton *>(elements[i]->paramRollout)->Clone(remap);
            break;
        default:
            break;
        }
        el->paramRollout = dynamic_cast<TUIParam *>(tp);
        if (el->paramRollout != NULL)
            el->paramRollout->myElem = el;
        el->type = elements[i]->type;

        for (int j = 0; j < elements[i]->objects.Count(); j++)
        {
            ts->elements[i]->objects[j]->node = elements[i]->objects[j]->node;
            ts->elements[i]->objects[j]->ResetStr();
            if (remap.FindMapping(elements[i]->objects[j]->node))
                ts->ReplaceReference(++objCount, remap.FindMapping(elements[i]->objects[j]->node));
            else
                ts->ReplaceReference(++objCount, elements[i]->objects[j]->node);
        }
    }

    int elemCount = objCount;
    for (int i = 0; i < elements.Count(); i++)
    {
        if (elements[i]->parent != NULL)
        {
            for (int k = 0; k < elements.Count(); k++)
                if (elements[k] == elements[i]->parent)
                {

                    ts->elements[i]->parent = ts->elements[k];
                    ts->elements[i]->parent->children.Append(1, &ts->elements[i]);
                    break;
                }
        }
        else
            ts->elements[i]->parent = NULL;

        if (ts->elements[i]->paramRollout != NULL)
#if MAX_PRODUCT_VERSION_MAJOR > 8
            RefResult ret = ts->ReplaceReference(++elemCount, ts->elements[i]->paramRollout->pTUIParamBlock);
#else
            RefResult ret = MakeRefByID(FOREVER, ++elemCount, ts->elements[i]->paramRollout->pTUIParamBlock);
#endif
        else
            elemCount++;
    }

    ts->pTabletUIBlock->SetValue(PB_TUI_NUMOBJS, 0, objCount);
    ts->pTabletUIBlock->SetValue(PB_TUI_NUMELEMS, 0, elemCount);

    BaseClone(this, ts, remap);
    clearTabletUI();
    ts->updateTabletUI();
    ts->updateElementTree(NULL);
    return ts;
}

void TabletUIObject::clearTabletUI()
{
    for (int i = 0; i < elements.Count(); i++)
    {
        if (elements[i]->myTuiElem != NULL)
        {
            delete elements[i]->myTuiElem;
            elements[i]->myTuiElem = NULL;
        }
    }

    delete coTabletUI::tUI;
    coTabletUI::tUI = NULL;
}

void TabletUIObject::updateTabletUI()
{
    clearTabletUI();

    for (int i = 0; i < elements.Count(); i++)
        if (elements[i]->parent == NULL)
        {
            TabletUIElement *root = elements[i];
            addcoTabletUIElement(root);
            TabletUIElement *el = root;
            int j = 0;
            while (!((el == root) && (j == el->children.Count())))
            {
                if (j < el->children.Count())
                {
                    el = el->children[j];
                    addcoTabletUIElement(el);
                }
                else
                {
                    el = el->parent;
                }
                j = 0;
                while ((j < el->children.Count()) && (el->children[j]->myTuiElem != NULL))
                    j++;
            }
        }
}

void TabletUIObject::tabletUIComboEntry(coTUIComboBox *tuiComboBox, TUIParamComboBox *paramComboBox)
{
    int emptyChecked = 0;
    paramComboBox->pTUIParamBlock->GetValue(PB_S_MAX, iObjParams->GetTime(),
                                            emptyChecked, FOREVER);
    TSTR tmpS = paramComboBox->emptyName;

    if (emptyChecked == 1)
        tuiComboBox->addEntry(STRTOUTF8(tmpS));

    multimap<int, ComboBoxObj *>::iterator it = paramComboBox->comboObjects.begin();
    int index;

    while (it != paramComboBox->comboObjects.end())
    {
        index = (*it).first;
        TSTR name = (*it).second->comboBoxName;
        name += _T("(");

        do
        {
            name += (*it).second->listStr;
            name += _T(" ");
            it++;
        } while ((it != paramComboBox->comboObjects.end()) && ((*it).first == index));

        name += _T(")");
        TSTR tmpS = name;
        tuiComboBox->addEntry(STRTOUTF8(tmpS));
    }

    int curSel = 0;
    if (iObjParams != NULL)
        paramComboBox->pTUIParamBlock->GetValue(PB_S_MIN, iObjParams->GetTime(),
                                                curSel, FOREVER);
    else
        paramComboBox->pTUIParamBlock->GetValue(PB_S_MIN, 0, curSel, FOREVER);
    tuiComboBox->setSelectedEntry(curSel);
}

void TabletUIObject::addcoTabletUIElement(TabletUIElement *el)
{
    if (coTabletUI::tUI == NULL)
        coTabletUI::tUI = new coTabletUI();

    int xp = 0, yp = 0, min = 0, max = 0, value = 0, val = 0;
    float fmin = 0.0, fmax = 0.0, fvalue = 0.0;

    int type = abs(el->type);
    int i = 0;
    int parentID = 0;

    if ((el->parent != NULL) && (el->parent->myTuiElem != NULL))
        parentID = el->parent->myTuiElem->getID();
    else
        parentID = coTabletUI::tUI->mainFolder->getID();

    coTUIElement *te = NULL;
    TSTR tmpS = el->name;
    if (type == TabletUIElement::TUIButton)
    {
        coTUIButton *tuiButton = new coTUIButton(STRTOUTF8(tmpS), parentID);
        te = dynamic_cast<coTUIElement *>(tuiButton);
    }
    else if (type == TabletUIElement::TUIComboBox)
    {
        coTUIComboBox *tuiComboBox = new coTUIComboBox(STRTOUTF8(tmpS), parentID);
        te = dynamic_cast<coTUIElement *>(tuiComboBox);
        TUIParamComboBox *paramComboBox = static_cast<TUIParamComboBox *>(el->paramRollout);
        tabletUIComboEntry(tuiComboBox, paramComboBox);

        if (iObjParams != NULL)
            el->paramRollout->pTUIParamBlock->GetValue(PB_S_MIN, iObjParams->GetTime(),
                                                       val, FOREVER);
        else
            el->paramRollout->pTUIParamBlock->GetValue(PB_S_MIN, 0, val, FOREVER);
        if ((int)paramComboBox->comboObjects.size() > val)
            tuiComboBox->setSelectedEntry(val);
    }
    else if (type == TabletUIElement::TUIEditField)
        te = (coTUIElement *)new coTUIEditField(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUIEditFloatField)
        te = (coTUIElement *)new coTUIEditFloatField(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUIEditIntField)
        te = (coTUIElement *)new coTUIEditIntField(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUIFloatSlider)
    {
        el->paramRollout->ParamBlockGetValues(&fmin, &fmax, &fvalue);
        if (iObjParams != NULL)
            el->paramRollout->pTUIParamBlock->GetValue(PB_S_VAL, iObjParams->GetTime(),
                                                       val, FOREVER);
        else
            el->paramRollout->pTUIParamBlock->GetValue(PB_S_VAL, 0, val, FOREVER);
        coTUIFloatSlider *tuiFloatSlider = new coTUIFloatSlider(STRTOUTF8(tmpS), parentID);
        tuiFloatSlider->setMin(fmin);
        tuiFloatSlider->setMax(fmax);
        tuiFloatSlider->setValue(fvalue);
        if (val == 0)
            tuiFloatSlider->setOrientation(coTUIFloatSlider::Vertical);
        else
            tuiFloatSlider->setOrientation(coTUIFloatSlider::Horizontal);
        te = dynamic_cast<coTUIElement *>(tuiFloatSlider);
    }
    else if (type == TabletUIElement::TUIFrame)
    {
        coTUIFrame *tuiFrame = new coTUIFrame(STRTOUTF8(tmpS), parentID);
        static_cast<TUIParamFrame *>(el->paramRollout)->setValues(tuiFrame);
        te = dynamic_cast<coTUIElement *>(tuiFrame);
    }
    else if (type == TabletUIElement::TUILabel)
    {
        coTUILabel *tuiLabel = new coTUILabel(STRTOUTF8(tmpS), parentID);
        te = dynamic_cast<coTUILabel *>(tuiLabel);
    }
    else if (type == TabletUIElement::TUIListBox)
        te = (coTUIElement *)new coTUIListBox(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUIMessageBox)
        te = (coTUIElement *)new coTUIMessageBox(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUISlider)
        te = (coTUIElement *)new coTUISlider(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUISpinEditfield)
        te = (coTUIElement *)new coTUISpinEditfield(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUISplitter)
    {
        coTUISplitter *tuiSplitter = new coTUISplitter(STRTOUTF8(tmpS), parentID);
        static_cast<TUIParamSplitter *>(el->paramRollout)->setValues(tuiSplitter);
        te = dynamic_cast<coTUIElement *>(tuiSplitter);
    }
    else if (type == TabletUIElement::TUITab)
        te = (coTUIElement *)new coTUITab(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUITabFolder)
        te = (coTUIElement *)new coTUITabFolder(STRTOUTF8(tmpS), parentID);
    else if (type == TabletUIElement::TUIToggleButton)
    {
        coTUIToggleButton *tuiToggleButton = new coTUIToggleButton(STRTOUTF8(tmpS), parentID);
        te = dynamic_cast<coTUIElement *>(tuiToggleButton);
    }

    if (te != NULL)
    {
        if (el->paramRollout != NULL)
            el->paramRollout->ParamBlockGetPos(&xp, &yp);
        el->myTuiElem = te;
        te->setPos(xp, yp);
        te->setLabel(STRTOUTF8(tmpS));
    }
}

// Indent to the given level.
void TabletUIElement::Indent(MAXSTREAM mStream, int level)
{
    if (level < 0)
        return;
    for (; level; level--)
     MSTREAMPRINTF  ("  "));
}

void TabletUIElement::Print(MAXSTREAM mStream)
{
    PRINT_HEADER(mStream, VRMLName(name.data()), elementTypes[type].elementName);
    Indent(mStream, 2);
    PRINT_STRING(mStream, _T("elementName"), name.data());
    Indent(mStream, 2);
    if (shaderParam.length() > 0)
    {
        PRINT_STRING(mStream, _T("shaderParam"), shaderParam.data());
        Indent(mStream, 2);
    }
    if (parent == NULL)
        PRINT_STRING(mStream, _T("parent"), _T("MainFolder"));
    else
        PRINT_STRING(mStream, _T("parent"), parent->name.data());

    if (paramRollout != NULL)
    {
        paramRollout->PrintAdditional(mStream);
        PRINT_TAIL(mStream);
        paramRollout->PrintScript(mStream, name.data(), 0);
    }
    else
        PRINT_TAIL(mStream);
}
