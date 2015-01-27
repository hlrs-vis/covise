/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: tabletui.h
 
    DESCRIPTION:  Defines a VRML 2.0 TabletUI helper
 
    CREATED BY: Uwe Woessner
 
    HISTORY: created 4 Apr. 2004
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "TUIParam.h"

#ifndef __TabletUI__H__

#define __TabletUI__H__

#define TabletUI_CLASS_ID1 0x74fb3452
#define TabletUI_CLASS_ID2 0xF412BDDD

#define TabletUIClassID Class_ID(TabletUI_CLASS_ID1, TabletUI_CLASS_ID2)

extern ClassDesc *GetTabletUIDesc();

class TabletUICreateCallBack;
class TabletUIObjPick;
class TabletUIObject;
class coTUIElement;
class coTUIComboBox;

class TabletUIElementList
{
public:
    TabletUIElementList(TabletUIElement *tuielem)
    {
        elem = tuielem;
        next = NULL;
    }
    ~TabletUIElementList()
    {
        delete next;
    }
    TabletUIElementList *AddElem(TabletUIElement *tuielem)
    {
        TabletUIElementList *el = new TabletUIElementList(tuielem);
        el->next = this;
        return el;
    }
    BOOL NodeInList(TabletUIElement *tuielem)
    {
        for (TabletUIElementList *el = this; el; el = el->next)
            if (el->elem == tuielem)
                return TRUE;
        return FALSE;
    }
    TabletUIElement *GetElem()
    {
        return elem;
    }
    TabletUIElementList *GetNext()
    {
        return next;
    }

private:
    TabletUIElement *elem;
    TabletUIElementList *next;
};

class TabletUIObj
{
public:
    INode *node;
    TSTR listStr;

    void ResetStr(void)
    {
        if (node)
            listStr.printf(_T("%s"), node->GetName());
        else
            listStr.printf(_T("%s"), _T("NO_NAME"));
    }
    TabletUIObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class TabletUIElement
{
    friend class TabletUIObjPick;
    friend class TabletUIObject;
    friend BOOL CALLBACK RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                                          TabletUIObject *th);

    HTREEITEM treeItem;
    int dlgObjPrevSel;
    bool hasObjects;

public:
    enum types
    {
        TUIButton,
        TUIComboBox,
        TUIEditField,
        TUIEditFloatField,
        TUIEditIntField,
        TUIFloatSlider,
        TUIFrame,
        TUILabel,
        TUIListBox,
        TUIMessageBox,
        TUISlider,
        TUISpinEditfield,
        TUISplitter,
        TUITab,
        TUITabFolder,
        TUIToggleButton
    };

    TabletUIObject *myObject;
    int type;
    TSTR name;
    TSTR shaderParam;
    BOOL loaded;
    TabletUIElement *parent;
    Tab<TabletUIObj *> objects;
    Tab<TabletUIElement *> children;
    coTUIElement *myTuiElem;
    TUIParam *paramRollout;

    TabletUIElement(TabletUIObject *obj, const TCHAR *name, int select, int numObjs);
    ~TabletUIElement();

    void BuildObjectList();
    void delObjects(HWND hDlg, int elIndex);
    void Print(MAXSTREAM mStream);
    void Indent(MAXSTREAM mStream, int level);
    bool checkElementLoaded();
    bool searchChild(TabletUIElement *searchElem, bool remove);
};

class TabletUIObject : public HelperObject
{
    friend class TabletUICreateCallBack;
    friend class TabletUIObjPick;
    friend class TabletUIElement;
    friend BOOL CALLBACK RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                                          TabletUIObject *th);
    friend static LRESULT CALLBACK ParentSelSubWndProc(
        HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

    static HWND hRollup;
    static TabletUIElement *dlgPrevSel;
    bool onload;
    float radius;
    BOOL needsScript; // Do we need to generate a script node?
    Mesh mesh;
    static ICustButton *TabletUIPickButton;
    IParamBlock *pTabletUIBlock;
    static IParamMap *pmapParam;

public:
    // Class vars
    static IObjParam *iObjParams;

    void BuildElementList();
    void BuildMesh(TimeValue t);

    Tab<TabletUIElement *> elements;

    TabletUIObject();
    ~TabletUIObject();

    TabletUIElement *addTUIElement(TabletUIObject *th, int selection, TabletUIElement *el, TCHAR *name);
    void selectElement(TabletUIElement *el);
    void deselectElement(TabletUIElement *el);
    void delElement(int index, TabletUIElement *el);
    void delReference(int index, TabletUIElement *el);
    void clearTabletUI();
    void updateTabletUI();
    void addcoTabletUIElement(TabletUIElement *el);
    void tabletUIComboEntry(coTUIComboBox *tuiComboBox, TUIParamComboBox *paramComboBox);
    void UpdateRefList();
    void updateElementTree(TabletUIElement *el);
    void addTreeItem(TabletUIElement *el, int neighbor, HWND hDlg);
    void clearTree(HWND hDlg);
    TabletUIElement *getTreeElement(HTREEITEM treeItem);
    void ParentComboBox(int type, TabletUIElement *el);
    HWND ComboBox();

    int oldType;
#if MAX_PRODUCT_VERSION_MAJOR > 8
    RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif

    // From BaseObject
    void GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm);
    int HitTest(TimeValue t, INode *inode, int type, int crossing,
                int flags, IPoint2 *p, ViewExp *vpt);
    int Display(TimeValue t, INode *inode, ViewExp *vpt, int flags);
    CreateMouseCallBack *GetCreateMouseCallBack();
    void BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev);
    void EndEditParams(IObjParam *ip, ULONG flags, Animatable *next);

#if MAX_PRODUCT_VERSION_MAJOR > 14
    virtual const
#else
    virtual
#endif
        MCHAR *
        GetObjectName()
    {
        return GetString(IDS_TABLETUI);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_TABLETUI);
    }
    Interval ObjectValidity();
    Interval ObjectValidity(TimeValue time);
    int DoOwnSelectHilite()
    {
        return 1;
    }

    void GetWorldBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);
    void GetLocalBoundBox(TimeValue t, INode *mat, ViewExp *vpt, Box3 &box);

    // Animatable methods
    void DeleteThis()
    {
        delete this;
    }
    Class_ID ClassID()
    {
        return Class_ID(TabletUI_CLASS_ID1,
                        TabletUI_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_TABLETUI_CLASS));
    }
    int IsKeyable()
    {
        return 1;
    }
    LRESULT CALLBACK TrackViewWinProc(HWND hwnd, UINT message,
                                      WPARAM wParam, LPARAM lParam)
    {
        return 0;
    }

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif

    int NumRefs();
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);

    IOResult Save(ISave *isave);
    IOResult Load(ILoad *iload);
};

#define PB_TUI_SIZE 0
#define PB_TUI_NUMOBJS 2
#define PB_TUI_TYPE 1
#define PB_TUI_NUMELEMS 3
#define PB_TUI_LENGTH 4

#endif
