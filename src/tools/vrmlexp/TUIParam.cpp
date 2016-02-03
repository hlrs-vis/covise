/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE..........: TUIParam.cpp

	DESCRIPTION...: Bitmap Path Editor

	CREATED BY....: Christer Janson - Kinetix

	HISTORY.......: Created Thursday, October 16, 1997

 *>	Copyright (c) 1998, All Rights Reserved.
 **********************************************************************/
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers

#include "tabletui.h"
#include <iassembly.h>
#include <iassemblymgr.h>
#include "vrml.h"
#include "3dsmaxport.h"
#include "coTabletUI.h"
#include "switch.h"

//HINSTANCE		hInstance;
static TCHAR *useFolder;

#define DX_RENDER_PARAMBLOCK 2
#define DX_RENDERBITMAP_PARAMID 2

#if MAX_PRODUCT_VERSION_MAJOR > 14 && ! defined FASTIO
#define STREAMPRINTF stream.Printf(
#define MSTREAMPRINTF  mStream.Printf( _T
#else
#define STREAMPRINTF fprintf((stream),
#define MSTREAMPRINTF  fprintf((mStream),
#endif
#if MAX_PRODUCT_VERSION_MAJOR > 14 && ! defined FASTIO
//Print Macros
#define PRINT_POS(stream, val1, val2) (stream.Printf(_T("pos %d %d \n"), val1, val2))
#define PRINT_IVALUE(stream, val1, val2) (stream.Printf(_T("%s %d\n"), val1, val2))
#define PRINT_FVALUE(stream, val1, val2) (stream.Printf(_T("%s %s\n"), val1, floatVal(val2)))
#define PRINT_STR(stream, val1, val2) (stream.Printf(_T("%s %s\n"), val1, val2))
#else
#define PRINT_POS(stream, val1, val2) (fprintf((stream), ("pos %d %d \n"), val1, val2))
#define PRINT_IVALUE(stream, val1, val2) (fprintf((stream), ("%s %d\n"), val1, val2))
#define PRINT_FVALUE(stream, val1, val2) (fprintf((stream), ("%s %s\n"), val1, floatVal(val2)))
#define PRINT_STR(stream, val1, val2) (fprintf((stream), ("%s %s\n"), val1, val2))
#endif

inline float
round(float f)
{
    if (f < 0.0f)
    {
        if (f > -1.0e-5)
            return 0.0f;
        return f;
    }
    if (f < 1.0e-5)
        return 0.0f;
    return f;
}
static void
CommaScan(TCHAR *buf)
{
    for (; *buf; buf++)
        if (*buf == ',')
            *buf = '.';
}
static TCHAR *floatVal(float f)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg"), 8);
    _stprintf(buf, format, round(f));
    CommaScan(buf);
    return buf;
}

extern TCHAR *elementTypes[];
typedef struct
{
    coTUIFrame::styles numStyle;
    TCHAR *strStyle;
} frameStyle;
frameStyle frameStyles[] = { { coTUIFrame::Plain, _T("Plain") }, { coTUIFrame::Raised, _T("Raised") }, { coTUIFrame::Sunken, _T("Sunken") }, { coTUIFrame::Plain, _T("") } };
typedef struct
{
    coTUIFrame::shapes numShape;
    TCHAR *strShape;
} frameShape;
frameShape frameShapes[] = { { coTUIFrame::NoFrame, _T("NoFrame") }, { coTUIFrame::Box, _T("Box") }, { coTUIFrame::Panel, _T("Panel") }, { coTUIFrame::WinPanel, _T("WinPanel") }, { coTUIFrame::HLine, _T("HLine") }, { coTUIFrame::VLine, _T("VLine") }, { coTUIFrame::StyledPanel, _T("StyledPanel") }, { coTUIFrame::NoFrame, _T("") } };

static TUIParam *theTUIParam;
HWND TUIParam::hRollup = NULL;

class TUIParamClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return theTUIParam;
    }
    const TCHAR *ClassName() { return GetString(IDS_TUIPARAM_EDITOR); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(TUIPARAM_CLASS_ID1, TUIPARAM_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static TUIParamClassDesc TUIParamDesc;
ClassDesc *GetTUIParamDesc() { return &TUIParamDesc; }

void TUIParam::rePos()
{
    int xp, yp;
    if (myElem->myObject->iObjParams != NULL)
    {
        pTUIParamBlock->GetValue(PB_S_POSX, myElem->myObject->iObjParams->GetTime(),
                                 xp, FOREVER);
        pTUIParamBlock->GetValue(PB_S_POSY, myElem->myObject->iObjParams->GetTime(),
                                 yp, FOREVER);
    }
    else
    {
        pTUIParamBlock->GetValue(PB_S_POSX, 0, xp, FOREVER);
        pTUIParamBlock->GetValue(PB_S_POSY, 0, yp, FOREVER);
    }
    myElem->myTuiElem->setPos(xp, yp);
}

static WNDPROC DefaultSelWndProc = NULL;

static LRESULT CALLBACK DefaultSelSubWndProc(
    HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_SETFOCUS:
        DisableAccelerators();
        if (theTUIParam->ip)
            theTUIParam->ip->UnRegisterDlgWnd(theTUIParam->pTUIParamMap->GetHWnd());
        break;
    case WM_KILLFOCUS:
        EnableAccelerators();
        if (theTUIParam->ip)
            theTUIParam->ip->RegisterDlgWnd(theTUIParam->pTUIParamMap->GetHWnd());
        break;

    case WM_CHAR:
        if (wParam == 13)
        {
            TCHAR buf[256];
            HWND hCombo = GetParent(hWnd);
            LRESULT res;
            GetWindowText(hWnd, buf, 256);
            if (CB_ERR == (res = SendMessage(hCombo, CB_FINDSTRINGEXACT, 0, (LPARAM)buf)))
            {
                // String is not already in the list.
                int curSel;
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->GetValue(PB_S_MIN, theTUIParam->myElem->myObject->iObjParams->GetTime(),
                                                          curSel, FOREVER);
                else
                    theTUIParam->pTUIParamBlock->GetValue(PB_S_MIN, 0, curSel, FOREVER);

                TUIParamComboBox *tuiComboBox = static_cast<TUIParamComboBox *>(theTUIParam);
                pair<multimap<int, ComboBoxObj *>::iterator, multimap<int, ComboBoxObj *>::iterator> indexRange;
                multimap<int, ComboBoxObj *>::iterator it;
                indexRange = tuiComboBox->comboObjects.equal_range(curSel);
                for (it = indexRange.first; it != indexRange.second; it++)
                    (*it).second->comboBoxName = buf;
                tuiComboBox->UpdateComboBox(curSel);
                theTUIParam->myElem->myObject->updateTabletUI();
            }
            return 0;
        }
        break;
    }
    return CallWindowProc(DefaultSelWndProc, hWnd, message, wParam, lParam);
}

static BOOL CALLBACK EnumChildren(HWND hwnd, LPARAM lParam)
{
    DefaultSelWndProc = DLSetWindowLongPtr(hwnd, DefaultSelSubWndProc);
    return FALSE;
}

void SubClassDefaultSel(HWND hDefaultSel)
{
    EnumChildWindows(hDefaultSel, EnumChildren, 0);
}

static INT_PTR CALLBACK TUIParamDlgProc(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    int val;
    float fval;

    switch (msg)
    {
    case WM_INITDIALOG:
    {
        theTUIParam->hRollup = hWnd;
        theTUIParam->Init(hWnd);
    }
    break;

    case WM_DESTROY:
        theTUIParam->Destroy(hWnd);
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_CLOSEBUTTON:
            theTUIParam->iu->CloseUtility();
            break;
        case IDC_CHECKDEP:
            //					theTUIParam.DoDialog();
            break;
        case IDC_POSX_EDIT:
        case IDC_POSY_EDIT:
            theTUIParam->rePos();
            break;
        case IDC_TUIMIN_EDIT:
            if (theTUIParam->myElem->myObject->iObjParams != NULL)
                theTUIParam->pTUIParamBlock->GetValue(PB_S_MIN, theTUIParam->myElem->myObject->iObjParams->GetTime(),
                                                      fval, FOREVER);
            else
                theTUIParam->pTUIParamBlock->GetValue(PB_S_MIN, 0, fval, FOREVER);
            static_cast<coTUIFloatSlider *>(theTUIParam->myElem->myTuiElem)->setMin(fval);
            break;
        case IDC_ShaderUniformEDIT:
        {

            ICustEdit *edit = GetICustEdit(GetDlgItem(hWnd, IDC_ShaderUniformEDIT));
            TCHAR buf[256];
            edit->GetText(buf, 256);
            theTUIParam->myElem->shaderParam = buf;
        }
        break;

        case IDC_TUIMAX_EDIT:
            if (theTUIParam->myElem->myObject->iObjParams != NULL)
                theTUIParam->pTUIParamBlock->GetValue(PB_S_MAX, theTUIParam->myElem->myObject->iObjParams->GetTime(),
                                                      fval, FOREVER);
            else
                theTUIParam->pTUIParamBlock->GetValue(PB_S_MAX, 0, fval, FOREVER);
            static_cast<coTUIFloatSlider *>(theTUIParam->myElem->myTuiElem)->setMax(fval);
            break;
        case IDC_TUIVALUE_EDIT:
            if (theTUIParam->myElem->myObject->iObjParams != NULL)
                theTUIParam->pTUIParamBlock->GetValue(PB_S_VALUE, theTUIParam->myElem->myObject->iObjParams->GetTime(),
                                                      fval, FOREVER);
            else
                theTUIParam->pTUIParamBlock->GetValue(PB_S_VALUE, 0, fval, FOREVER);
            static_cast<coTUIFloatSlider *>(theTUIParam->myElem->myTuiElem)->setValue(fval);
            break;

        case IDC_TUIORIENTATION_SLIDER:
            if (IsDlgButtonChecked(hWnd, IDC_TUIORIENTATION_SLIDER))
            {
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VAL, theTUIParam->myElem->myObject->iObjParams->GetTime(), coTUIFloatSlider::Vertical);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VAL, 0, coTUIFloatSlider::Vertical);
                static_cast<coTUIFloatSlider *>(theTUIParam->myElem->myTuiElem)->setOrientation(coTUIFloatSlider::Vertical);
            }
            else
            {
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VAL, theTUIParam->myElem->myObject->iObjParams->GetTime(), coTUIFloatSlider::Horizontal);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VAL, 0, coTUIFloatSlider::Horizontal);
                static_cast<coTUIFloatSlider *>(theTUIParam->myElem->myTuiElem)->setOrientation(coTUIFloatSlider::Horizontal);
            }
            break;

        case IDC_TUISHAPE_COMBO:
            if (HIWORD(wParam) == LBN_SELCHANGE)
            {
                HWND cb = GetDlgItem(hWnd, IDC_TUISHAPE_COMBO);
                int curSel = ComboBox_GetCurSel(cb);
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_MIN, theTUIParam->myElem->myObject->iObjParams->GetTime(), curSel);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_MIN, 0, curSel);
                static_cast<coTUIFrame *>(theTUIParam->myElem->myTuiElem)->setShape(frameShapes[curSel].numShape);
            }
            break;

        case IDC_TUISTYLE_COMBO:
            if (HIWORD(wParam) == LBN_SELCHANGE)
            {
                HWND cb = GetDlgItem(hWnd, IDC_TUISTYLE_COMBO);
                int curSel = ComboBox_GetCurSel(cb);
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_MAX, theTUIParam->myElem->myObject->iObjParams->GetTime(), curSel);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_MAX, 0, curSel);
                static_cast<coTUIFrame *>(theTUIParam->myElem->myTuiElem)->setStyle(frameStyles[curSel].numStyle);
            }
            break;
        case IDC_TUIORIENTATION_SPLITTER:
            if (IsDlgButtonChecked(hWnd, IDC_TUIORIENTATION_SPLITTER))
            {
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VALUE, theTUIParam->myElem->myObject->iObjParams->GetTime(), coTUISplitter::Vertical);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VALUE, 0, coTUISplitter::Vertical);
                static_cast<coTUISplitter *>(theTUIParam->myElem->myTuiElem)->setOrientation(coTUISplitter::Vertical);
            }
            else
            {
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VALUE, theTUIParam->myElem->myObject->iObjParams->GetTime(), coTUISplitter::Horizontal);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_VALUE, 0, coTUISplitter::Horizontal);
                static_cast<coTUISplitter *>(theTUIParam->myElem->myTuiElem)->setOrientation(coTUISplitter::Horizontal);
            }
            break;
        case IDC_TUIEMPTY_CHECK:
        {
            val = IsDlgButtonChecked(hWnd, IDC_TUIEMPTY_CHECK);
            EnableWindow(GetDlgItem(hWnd, IDC_TUIEMPTY_EDIT), val);

            HWND cb = GetDlgItem(hWnd, IDC_TUIDEFAULT_COMBO);
            int curSel = ComboBox_GetCurSel(cb);
            if (val == 1)
                curSel++;
            else
                (curSel == 0) ? 0 : curSel--;
            if (theTUIParam->myElem->myObject->iObjParams != NULL)
            {
                theTUIParam->pTUIParamBlock->SetValue(PB_S_MAX, theTUIParam->myElem->myObject->iObjParams->GetTime(), val);
                theTUIParam->pTUIParamBlock->SetValue(PB_S_MIN, theTUIParam->myElem->myObject->iObjParams->GetTime(), curSel);
            }
            else
            {
                theTUIParam->pTUIParamBlock->SetValue(PB_S_MAX, 0, val);
                theTUIParam->pTUIParamBlock->SetValue(PB_S_MIN, 0, curSel);
            }

            static_cast<TUIParamComboBox *>(theTUIParam)->UpdateComboBox(curSel);

            theTUIParam->myElem->myObject->updateTabletUI();
        }
        break;
        case IDC_TUIDEFAULT_COMBO:
        {
            HWND cb = GetDlgItem(hWnd, IDC_TUIDEFAULT_COMBO);
            int curSel = ComboBox_GetCurSel(cb);
            if (curSel != -1)
            {
                if (theTUIParam->myElem->myObject->iObjParams != NULL)
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_MIN, theTUIParam->myElem->myObject->iObjParams->GetTime(), curSel);
                else
                    theTUIParam->pTUIParamBlock->SetValue(PB_S_MIN, 0, curSel);
                static_cast<coTUIComboBox *>(theTUIParam->myElem->myTuiElem)->setSelectedEntry(curSel);
                theTUIParam->myElem->myObject->updateTabletUI();
            }
        }
        break;
        case IDC_TUIEMPTY_EDIT:
            if (HIWORD(wParam) == EN_CHANGE)
            {
                ICustEdit *edit = GetICustEdit(GetDlgItem(hWnd, IDC_TUIEMPTY_EDIT));
                TCHAR buf[256];
                edit->GetText(buf, 256);

                if (edit->GotReturn())
                {
                    TUIParamComboBox *paramComboBox = static_cast<TUIParamComboBox *>(theTUIParam);
                    paramComboBox->emptyName = buf;
                    HWND cb = GetDlgItem(hWnd, IDC_TUIDEFAULT_COMBO);
                    int curSel = ComboBox_GetCurSel(cb);
                    paramComboBox->UpdateComboBox(curSel);
                    theTUIParam->myElem->myObject->updateTabletUI();
                }

                ReleaseICustEdit(edit);
            }
            break;
        }
        break;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MOUSEMOVE:
        theTUIParam->ip->RollupMouseMessage(hWnd, msg, wParam, lParam);
        break;

    case CC_SPINNER_CHANGE:
        if (HIWORD(wParam))
            switch (LOWORD(wParam))
            {
            case IDC_POSX_SPIN:
            case IDC_POSY_SPIN:
                theTUIParam->rePos();
                break;
            };
        break;
    }
    return TRUE;
}

#define CURRENT_VERSION 0

// Current version
//static ParamVersionDesc curVersion(descVer0, PARAMBLOCK_LENGTH, CURRENT_VERSION);

class ParamDlgProc : public ParamMapUserDlgProc
{
public:
    TUIParam *ob;

    ParamDlgProc(TUIParam *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR ParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                              UINT msg, WPARAM wParam, LPARAM lParam)
{
    return TUIParamDlgProc(hWnd, msg, wParam, lParam);
}

IParamMap *TUIParam::pTUIParamMap = NULL;

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

TUIParam::~TUIParam()
{
}

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult TUIParam::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                     PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult TUIParam::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                     PartID &partID, RefMessage message)
#endif
{
    return REF_SUCCEED;
}

RefTargetHandle
TUIParam::GetReference(int ind)
{
    if (ind == 0)
        return pTUIParamBlock;

    return NULL;
}

void
TUIParam::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pTUIParamBlock = (IParamBlock *)rtarg;
    }
}

void TUIParam::EndEditParams(Interface *ip, IUtil *iu)
{
    this->iu = NULL;
    this->ip = NULL;

    if (pTUIParamMap)
        DestroyCPParamMap(pTUIParamMap);
    pTUIParamMap = NULL;
}

void TUIParam::ParamBlockGetPos(int *posx, int *posy)
{
    if (pTUIParamBlock != NULL)
    {
        if (myElem->myObject->iObjParams != NULL)
        {
            pTUIParamBlock->GetValue(PB_S_POSX, myElem->myObject->iObjParams->GetTime(),
                                     *posx, FOREVER);
            pTUIParamBlock->GetValue(PB_S_POSY, myElem->myObject->iObjParams->GetTime(),
                                     *posy, FOREVER);
        }
        else
        {
            pTUIParamBlock->GetValue(PB_S_POSX, 0, *posx, FOREVER);
            pTUIParamBlock->GetValue(PB_S_POSY, 0, *posy, FOREVER);
        }
    }
}

void TUIParam::Destroy(HWND hWnd)
{
}

ObjectState
TUIParam::Eval(TimeValue time)
{
    return ObjectState(this);
}

void printPos(IParamBlock *pb, TabletUIObject *th, MAXSTREAM mStream)
{
    int xp, yp;

    pb->GetValue(PB_S_POSX, 0,
                 xp, FOREVER);
    pb->GetValue(PB_S_POSY, 0,
                 yp, FOREVER);
    PRINT_POS(mStream, xp, yp);
}

TUIParam::TUIParam()
{
    iu = NULL;
    ip = NULL;
    //   DialogType = type;

    pTUIParamBlock = NULL;
}

TUIParamFloatSlider::TUIParamFloatSlider()
    : TUIParam()
{

    ParamBlockDescID descVer0[] = {
        { TYPE_INT, NULL, FALSE, 0 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
    };

    IParamBlock *pb = CreateParameterBlock(descVer0, PARAMBLOCK_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_POSX, 0, 0);
    pb->SetValue(PB_S_POSY, 0, 0);
    pb->SetValue(PB_S_MIN, 0, 0);
    pb->SetValue(PB_S_MAX, 0, 1.0f);
    pb->SetValue(PB_S_VALUE, 0, 0.5f);
    pb->SetValue(PB_S_VAL, 0, coTUISplitter::Horizontal);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pTUIParamBlock);
}

void TUIParamFloatSlider::BeginEditParams(Interface *ip, IUtil *iu)
{
    ParamUIDesc descParam[] = {
        // Size
        ParamUIDesc(
            PB_S_POSX,
            EDITTYPE_INT,
            IDC_POSX_EDIT, IDC_POSX_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_POSY,
            EDITTYPE_INT,
            IDC_POSY_EDIT, IDC_POSY_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_MIN,
            EDITTYPE_FLOAT,
            IDC_TUIMIN_EDIT, IDC_TUIMIN_SPIN,
            -10000000.0, 10000000.0,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_MAX,
            EDITTYPE_FLOAT,
            IDC_TUIMAX_EDIT, IDC_TUIMAX_SPIN,
            -10000000.0, 10000000.0,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_VALUE,
            EDITTYPE_FLOAT,
            IDC_TUIVALUE_EDIT, IDC_TUIVALUE_SPIN,
            -10000000.0, 10000000.0,
            SPIN_AUTOSCALE),
    };

    theTUIParam = this;
    this->iu = iu;
    this->ip = ip;

    if (pTUIParamMap)
    {
        // Left over from last TabletUI created
        pTUIParamMap->SetParamBlock(pTUIParamBlock);
    }
    else
    {

        // Gotta make a new one.

        pTUIParamMap = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                        pTUIParamBlock,
                                        ip,
                                        hInstance,
                                        MAKEINTRESOURCE(IDD_TUISLIDER),
                                        GetString(IDS_TUIPARAM_EDITOR),
                                        0);
    }

    if (pTUIParamMap)
    {
        // A callback for dialog
        pTUIParamMap->SetUserDlgProc(new ParamDlgProc(this));
    }
}

RefTargetHandle
TUIParamFloatSlider::Clone(RemapDir &remap)
{
    TUIParamFloatSlider *tp = new TUIParamFloatSlider();
    tp->ReplaceReference(0, pTUIParamBlock->Clone(remap));

    BaseClone(this, tp, remap);
    return tp;
}

void TUIParamFloatSlider::Init(HWND hWnd)
{
    int i = 0;

    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_VAL, myElem->myObject->iObjParams->GetTime(),
                                 i, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_VAL, 0, i, FOREVER);
    CheckDlgButton(hWnd, IDC_TUIORIENTATION_SLIDER, i - 1);

    ICustEdit *edit = GetICustEdit(GetDlgItem(hWnd, IDC_ShaderUniformEDIT));
    edit->SetText(myElem->shaderParam.data());
}

void TUIParamFloatSlider::PrintAdditional(MAXSTREAM mStream)
{
    int val;
    float value;
    IParamBlock *pb = myElem->paramRollout->pTUIParamBlock;

    myElem->Indent(mStream, 2);
    printPos(pTUIParamBlock, myElem->myObject, mStream);

    //   pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(), val, FOREVER);
    pTUIParamBlock->GetValue(PB_S_MIN, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_FVALUE(mStream, _T("min"), value);

    pTUIParamBlock->GetValue(PB_S_MAX, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_FVALUE(mStream, _T("max"), value);

    pTUIParamBlock->GetValue(PB_S_VALUE, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_FVALUE(mStream, _T("value"), value);

    pTUIParamBlock->GetValue(PB_S_VAL, 0,
                             val, FOREVER);
    myElem->Indent(mStream, 2);
    if (val == 1)
        PRINT_STR(mStream, _T("orientation"), _T("\"horizontal\""));
    else
        PRINT_STR(mStream, _T("orientation"), _T("\"vertical\""));
}

void TUIParamFloatSlider::PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval)
{
    myElem->Indent(mStream, 0);

   MSTREAMPRINTF  ("DEF %s-SCRIPT Script { \n"),objname);
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventIn SFFloat value\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventOut SFFloat value_changed\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("field SFNode fSlider USE %s\n"), objname);
   myElem->Indent(mStream, 1);

   MSTREAMPRINTF  ("url \"javascript:\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("function value(k) {\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("value_changed = (k-fSlider.min)/(fSlider.max-fSlider.min);\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("}\"\n"));
   MSTREAMPRINTF  ("}\n\n"));

   myElem->Indent(mStream, 0);
   MSTREAMPRINTF  ("ROUTE %s.value_changed TO %s-SCRIPT.value\n\n"),objname,objname);
}

TUIParamSpinEditField::TUIParamSpinEditField()
    : TUIParam()
{

    ParamBlockDescID descVer0[] = {
        { TYPE_INT, NULL, FALSE, 0 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
    };

    IParamBlock *pb = CreateParameterBlock(descVer0, PARAMBLOCK_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_POSX, 0, 0);
    pb->SetValue(PB_S_POSY, 0, 0);
    pb->SetValue(PB_S_MIN, 0, 0);
    pb->SetValue(PB_S_MAX, 0, 20);
    pb->SetValue(PB_S_VALUE, 0, 5);
    pb->SetValue(PB_S_VAL, 0, 5);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pTUIParamBlock);
}

void TUIParamSpinEditField::BeginEditParams(Interface *ip, IUtil *iu)
{
    ParamUIDesc descParam[] = {
        // Size
        ParamUIDesc(
            PB_S_POSX,
            EDITTYPE_INT,
            IDC_POSX_EDIT, IDC_POSX_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_POSY,
            EDITTYPE_INT,
            IDC_POSY_EDIT, IDC_POSY_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_MIN,
            EDITTYPE_INT,
            IDC_TUIMIN_EDIT, IDC_TUIMIN_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_MAX,
            EDITTYPE_INT,
            IDC_TUIMAX_EDIT, IDC_TUIMAX_SPIN,
            0, 30,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_VALUE,
            EDITTYPE_INT,
            IDC_TUIVALUE_EDIT, IDC_TUIVALUE_SPIN,
            0, 30,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_VAL,
            EDITTYPE_INT,
            IDC_TUIVAL_EDIT, IDC_TUIVAL_SPIN,
            0, 30,
            SPIN_AUTOSCALE),

    };

    theTUIParam = this;
    this->iu = iu;
    this->ip = ip;

    if (pTUIParamMap)
    {
        // Left over from last TabletUI created
        pTUIParamMap->SetParamBlock(pTUIParamBlock);
    }
    else
    {

        // Gotta make a new one.

        pTUIParamMap = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                        pTUIParamBlock,
                                        ip,
                                        hInstance,
                                        MAKEINTRESOURCE(IDD_TUISLIDER), // was spinedit
                                        GetString(IDS_TUIPARAM_EDITOR),
                                        0);
    }

    if (pTUIParamMap)
    {
        // A callback for dialog
        pTUIParamMap->SetUserDlgProc(new ParamDlgProc(this));
    }
}

RefTargetHandle
TUIParamSpinEditField::Clone(RemapDir &remap)
{
    TUIParamSpinEditField *tp = new TUIParamSpinEditField();
    tp->ReplaceReference(0, pTUIParamBlock->Clone(remap));

    BaseClone(this, tp, remap);
    return tp;
}

void TUIParamSpinEditField::PrintAdditional(MAXSTREAM mStream)
{
}

TUIParamButton::TUIParamButton()
    : TUIParam()
{

    ParamBlockDescID descVer0[] = {
        { TYPE_INT, NULL, FALSE, 0 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
    };

    IParamBlock *pb = CreateParameterBlock(descVer0, 2,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_POSX, 0, 0);
    pb->SetValue(PB_S_POSY, 0, 0);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pTUIParamBlock);
}

void TUIParamButton::BeginEditParams(Interface *ip, IUtil *iu)
{
    ParamUIDesc descParam[] = {
        // Size
        ParamUIDesc(
            PB_S_POSX,
            EDITTYPE_INT,
            IDC_POSX_EDIT, IDC_POSX_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_POSY,
            EDITTYPE_INT,
            IDC_POSY_EDIT, IDC_POSY_SPIN,
            0, 10,
            SPIN_AUTOSCALE),

    };

    theTUIParam = this;
    this->iu = iu;
    this->ip = ip;

    if (pTUIParamMap)
    {
        // Left over from last TabletUI created
        pTUIParamMap->SetParamBlock(pTUIParamBlock);
    }
    else
    {

        // Gotta make a new one.

        pTUIParamMap = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                        pTUIParamBlock,
                                        ip,
                                        hInstance,
                                        MAKEINTRESOURCE(IDD_TUIBUTTON),
                                        GetString(IDS_TUIPARAM_EDITOR),
                                        0);
    }

    if (pTUIParamMap)
    {
        // A callback for dialog
        pTUIParamMap->SetUserDlgProc(new ParamDlgProc(this));
    }
}

RefTargetHandle
TUIParamButton::Clone(RemapDir &remap)
{
    TUIParamButton *tp = new TUIParamButton();
    tp->ReplaceReference(0, pTUIParamBlock->Clone(remap));

    BaseClone(this, tp, remap);
    return tp;
}

void TUIParamButton::PrintAdditional(MAXSTREAM mStream)
{
    IParamBlock *pb = myElem->paramRollout->pTUIParamBlock;

    myElem->Indent(mStream, 2);
    printPos(pTUIParamBlock, myElem->myObject, mStream);
}

void TUIParamToggleButton::PrintAdditional(MAXSTREAM mStream)
{

    myElem->Indent(mStream, 2);
    printPos(pTUIParamBlock, myElem->myObject, mStream);
    if (myElem->objects.Count() && myElem->objects[0]->node)
    { // if we have a switch, then look for its default value and set our state accordingly

        Object *obj = myElem->objects[0]->node->EvalWorldState(0).obj;
        Class_ID id;
        if (obj)
        {
            id = obj->ClassID();

            if (id == SwitchClassID)
            {
                SwitchObject *so = (SwitchObject *)obj;
                int defaultValue;
                so->pblock->GetValue(PB_S_DEFAULT, 0, defaultValue, FOREVER);
                myElem->Indent(mStream, 2);
                if (defaultValue == -1)
                {
                                        MSTREAMPRINTF  ("state FALSE\n"));
                }
                else
                {
                                        MSTREAMPRINTF  ("state TRUE\n"));
                }
            }
        }
    }
}

void TUIParamToggleButton::PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval)
{
    myElem->Indent(mStream, 0);

   MSTREAMPRINTF  ("DEF %s-SCRIPT Script { \n"),objname);
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventIn SFBool state\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventOut SFTime startTime_changed\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventOut SFTime stopTime_changed\n"));
   myElem->Indent(mStream, 1);

   bool oldState = false;
   if (myElem->objects.Count() && myElem->objects[0]->node)
   { // if we have a switch, then look for its default value and set our state accordingly

       Object *obj = myElem->objects[0]->node->EvalWorldState(0).obj;
       Class_ID id;
       if (obj)
       {
           id = obj->ClassID();

           if (id == SwitchClassID)
           {
               SwitchObject *so = (SwitchObject *)obj;
               int defaultValue;
               so->pblock->GetValue(PB_S_DEFAULT, 0, defaultValue, FOREVER);
               myElem->Indent(mStream, 2);
               if (defaultValue != -1)
               {
                   oldState = true;
               }
           }
       }
   }
   if (oldState)
   {
           MSTREAMPRINTF  ("field SFBool oldstate TRUE\n"), cycleInterval);
   }
   else
   {
           MSTREAMPRINTF  ("field SFBool oldstate FALSE\n"), cycleInterval);
   }
   myElem->Indent(mStream, 1);

   MSTREAMPRINTF  ("url \"javascript:\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("function state(k,t) {\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("if (k != oldstate)\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("{\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("if (k) startTime_changed = t;\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("else stopTime_changed = t;\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("oldstate = k;\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("}\n"));

   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("}\"\n"));
   MSTREAMPRINTF  ("}\n\n"));

   myElem->Indent(mStream, 0);
   MSTREAMPRINTF  ("ROUTE %s.state TO %s-SCRIPT.state\n\n"),myElem->name.data(),objname);
}

void TUIParamToggleButton::PrintObjects(MAXSTREAM mStream, TabletUIObj *obj)
{
   MSTREAMPRINTF  ("\n\n"));

   myElem->Indent(mStream, 0);

   MSTREAMPRINTF  ("DEF %s-SCRIPT Script { \n"), obj->listStr);
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventIn SFTime stopTime\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventIn SFTime fractionChanged\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventOut SFFloat newFraction\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventOut SFTime timerStop\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("eventOut SFBool toggleOn\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("field SFFloat oldStopFraction 0.0\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("field SFFloat newStopFraction 0.0\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("field SFNode node USE %s-TIMER\n\n"),obj->listStr);
   myElem->Indent(mStream, 1);

   MSTREAMPRINTF  ("url \"javascript:\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("toggleOn = TRUE;\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("function fractionChanged(k,t) {\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("if (oldStopFraction != 0)\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("{\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("if (oldStopFraction + k > 1)\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("{\n"));
   myElem->Indent(mStream, 4);
   MSTREAMPRINTF  ("oldStopFraction = -k;\n"));
   myElem->Indent(mStream, 4);
   MSTREAMPRINTF  ("if (!node.loop)\n"));
   myElem->Indent(mStream, 4);
   MSTREAMPRINTF  ("{\n"));
   myElem->Indent(mStream, 5);
   MSTREAMPRINTF  ("timerStop = t;\n"));
   myElem->Indent(mStream, 5);
   MSTREAMPRINTF  ("toggleOn = FALSE;\n"));
   myElem->Indent(mStream, 4);
   MSTREAMPRINTF  ("}\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("}\n"));
   myElem->Indent(mStream, 3);
   MSTREAMPRINTF  ("else if (oldStopFraction + k < 0) oldStopFraction = 1+oldStopFraction;\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("}\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("else if ((k == 1) && !node.loop) toggleOn = FALSE;\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("newFraction = oldStopFraction + k;\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("newStopFraction = newFraction;\n"));
   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("}\n\n"));

   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("function stopTime(k,t) {\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("if (newStopFraction == 1) oldStopFraction = 0;\n"));
   myElem->Indent(mStream, 2);
   MSTREAMPRINTF  ("else oldStopFraction = newStopFraction;\n"));

   myElem->Indent(mStream, 1);
   MSTREAMPRINTF  ("}\"\n"));
   MSTREAMPRINTF  ("}\n\n"));
}

TUIParamFrame::TUIParamFrame()
    : TUIParam()
{

    ParamBlockDescID descVer0[] = {
        { TYPE_INT, NULL, FALSE, 0 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
    };

    IParamBlock *pb = CreateParameterBlock(descVer0, PARAMBLOCK_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_POSX, 0, 0);
    pb->SetValue(PB_S_POSY, 0, 0);
    pb->SetValue(PB_S_MIN, 0, 1);
    pb->SetValue(PB_S_MAX, 0, 1);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pTUIParamBlock);
}

void TUIParamFrame::BeginEditParams(Interface *ip, IUtil *iu)
{
    ParamUIDesc descParam[] = {
        // Size
        ParamUIDesc(
            PB_S_POSX,
            EDITTYPE_INT,
            IDC_POSX_EDIT, IDC_POSX_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_POSY,
            EDITTYPE_INT,
            IDC_POSY_EDIT, IDC_POSY_SPIN,
            0, 10,
            SPIN_AUTOSCALE),

    };

    theTUIParam = this;
    this->iu = iu;
    this->ip = ip;

    if (pTUIParamMap)
    {
        // Left over from last TabletUI created
        pTUIParamMap->SetParamBlock(pTUIParamBlock);
    }
    else
    {

        // Gotta make a new one.

        pTUIParamMap = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                        pTUIParamBlock,
                                        ip,
                                        hInstance,
                                        MAKEINTRESOURCE(IDD_TUIFRAME),
                                        GetString(IDS_TUIPARAM_EDITOR),
                                        0);
    }

    if (pTUIParamMap)
    {
        // A callback for dialog
        pTUIParamMap->SetUserDlgProc(new ParamDlgProc(this));
    }
}

RefTargetHandle
TUIParamFrame::Clone(RemapDir &remap)
{
    TUIParamFrame *tp = new TUIParamFrame();
    tp->ReplaceReference(0, pTUIParamBlock->Clone(remap));

    BaseClone(this, tp, remap);
    return tp;
}

void TUIParamFrame::Init(HWND hWnd)
{
    HWND cb = GetDlgItem(hWnd, IDC_TUISHAPE_COMBO);
    ComboBox_ResetContent(cb);
    int i = 0;
    while (_tcslen(frameShapes[i].strShape) > 0)
        ComboBox_AddString(cb, frameShapes[i++].strShape);
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 i, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MIN, 0, i, FOREVER);
    ComboBox_SelectString(cb, 0, frameShapes[i].strShape);

    cb = GetDlgItem(hWnd, IDC_TUISTYLE_COMBO);
    ComboBox_ResetContent(cb);
    i = 0;
    while (_tcslen(frameStyles[i].strStyle) > 0)
        ComboBox_AddString(cb, frameStyles[i++].strStyle);
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 i, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MAX, 0, i, FOREVER);
    ComboBox_SelectString(cb, 0, frameStyles[i].strStyle);
}

void TUIParamFrame::PrintAdditional(MAXSTREAM mStream)
{
    int value = 0;

    myElem->Indent(mStream, 2);
    printPos(pTUIParamBlock, myElem->myObject, mStream);

    pTUIParamBlock->GetValue(PB_S_MIN, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_IVALUE(mStream, _T("shape"), frameShapes[value].numShape);

    pTUIParamBlock->GetValue(PB_S_MAX, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_IVALUE(mStream, _T("style"), frameStyles[value].numStyle);
}

void TUIParamFrame::setValues(coTUIFrame *tf)
{
    int value1, value2;

    if (myElem->myObject->iObjParams != NULL)
    {
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 value1, FOREVER);
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 value2, FOREVER);
    }
    else
    {
        pTUIParamBlock->GetValue(PB_S_MIN, 0, value1, FOREVER);
        pTUIParamBlock->GetValue(PB_S_MAX, 0, value2, FOREVER);
    }
    tf->setShape(frameShapes[value1].numShape);
    tf->setStyle(frameStyles[value2].numStyle);
}

TUIParamSplitter::TUIParamSplitter()
    : TUIParam()
{

    ParamBlockDescID descVer0[] = {
        { TYPE_INT, NULL, FALSE, 0 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_BOOL, NULL, FALSE, 1 },
    };

    IParamBlock *pb = CreateParameterBlock(descVer0, PARAMBLOCK_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_POSX, 0, 0);
    pb->SetValue(PB_S_POSY, 0, 0);
    pb->SetValue(PB_S_MIN, 0, 1);
    pb->SetValue(PB_S_MAX, 0, 1);
    pb->SetValue(PB_S_VALUE, 0, coTUISplitter::Horizontal);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pTUIParamBlock);
}

void TUIParamSplitter::BeginEditParams(Interface *ip, IUtil *iu)
{
    ParamUIDesc descParam[] = {
        // Size
        ParamUIDesc(
            PB_S_POSX,
            EDITTYPE_INT,
            IDC_POSX_EDIT, IDC_POSX_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_POSY,
            EDITTYPE_INT,
            IDC_POSY_EDIT, IDC_POSY_SPIN,
            0, 10,
            SPIN_AUTOSCALE),

    };

    theTUIParam = this;
    this->iu = iu;
    this->ip = ip;

    if (pTUIParamMap)
    {
        // Left over from last TabletUI created
        pTUIParamMap->SetParamBlock(pTUIParamBlock);
    }
    else
    {

        // Gotta make a new one.

        pTUIParamMap = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                        pTUIParamBlock,
                                        ip,
                                        hInstance,
                                        MAKEINTRESOURCE(IDD_TUISPLITTER),
                                        GetString(IDS_TUIPARAM_EDITOR),
                                        0);
    }

    if (pTUIParamMap)
    {
        // A callback for dialog
        pTUIParamMap->SetUserDlgProc(new ParamDlgProc(this));
    }
}

RefTargetHandle
TUIParamSplitter::Clone(RemapDir &remap)
{
    TUIParamSplitter *tp = new TUIParamSplitter();
    tp->ReplaceReference(0, pTUIParamBlock->Clone(remap));

    BaseClone(this, tp, remap);
    return tp;
}

void TUIParamSplitter::Init(HWND hWnd)
{
    HWND cb1 = GetDlgItem(hWnd, IDC_TUISHAPE_COMBO);
    ComboBox_ResetContent(cb1);
    int i = 0;
    while (_tcslen(frameShapes[i].strShape) > 0)
        ComboBox_AddString(cb1, frameShapes[i++].strShape);

    HWND cb2 = GetDlgItem(hWnd, IDC_TUISTYLE_COMBO);
    ComboBox_ResetContent(cb2);
    int j = 0;
    while (_tcslen(frameStyles[j].strStyle) > 0)
        ComboBox_AddString(cb2, frameStyles[j++].strStyle);

    int k = 0;
    if (myElem->myObject->iObjParams != NULL)
    {
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 i, FOREVER);
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 j, FOREVER);
        pTUIParamBlock->GetValue(PB_S_VALUE, myElem->myObject->iObjParams->GetTime(),
                                 k, FOREVER);
    }
    else
    {
        pTUIParamBlock->GetValue(PB_S_MIN, 0, i, FOREVER);
        pTUIParamBlock->GetValue(PB_S_MAX, 0, j, FOREVER);
        pTUIParamBlock->GetValue(PB_S_VALUE, 0, k, FOREVER);
    }

    ComboBox_SelectString(cb1, 0, frameShapes[i].strShape);
    ComboBox_SelectString(cb2, 0, frameStyles[j].strStyle);
    CheckDlgButton(hWnd, IDC_TUIORIENTATION_SPLITTER, k - 1);
}

void TUIParamSplitter::PrintAdditional(MAXSTREAM mStream)
{
    int value = 0;

    myElem->Indent(mStream, 2);
    printPos(pTUIParamBlock, myElem->myObject, mStream);

    pTUIParamBlock->GetValue(PB_S_MIN, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_IVALUE(mStream, _T("shape"), frameShapes[value].numShape);

    pTUIParamBlock->GetValue(PB_S_MAX, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_IVALUE(mStream, _T("style"), frameStyles[value].numStyle);

    pTUIParamBlock->GetValue(PB_S_VALUE, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    PRINT_IVALUE(mStream, _T("orientation"), value);
}

void TUIParamSplitter::setValues(coTUISplitter *ts)
{
    int value1, value2, value3;

    if (myElem->myObject->iObjParams != NULL)
    {
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 value1, FOREVER);
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 value2, FOREVER);
        pTUIParamBlock->GetValue(PB_S_VALUE, myElem->myObject->iObjParams->GetTime(),
                                 value3, FOREVER);
    }
    else
    {
        pTUIParamBlock->GetValue(PB_S_MIN, 0, value1, FOREVER);
        pTUIParamBlock->GetValue(PB_S_MAX, 0, value2, FOREVER);
        pTUIParamBlock->GetValue(PB_S_VALUE, 0, value3, FOREVER);
    }

    ts->setShape(frameShapes[value1].numShape);
    ts->setStyle(frameStyles[value2].numStyle);
    ts->setOrientation(value3);
}

ComboBoxObj::ComboBoxObj(INode *swNode, INode *inode, TSTR &name)
{
    switchNode = swNode;
    node = inode;
    comboBoxName = listStr = name;
}

TUIParamComboBox::TUIParamComboBox()
    : TUIParam()
{
    emptyName = _T("NONE");

    ParamBlockDescID descVer0[] = {
        { TYPE_INT, NULL, FALSE, 0 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 },
        { TYPE_INT, NULL, FALSE, 1 }
    };

    IParamBlock *pb = CreateParameterBlock(descVer0, PARAMBLOCK_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_S_POSX, 0, 0);
    pb->SetValue(PB_S_POSY, 0, 0);
    pb->SetValue(PB_S_MIN, 0, 0);
    pb->SetValue(PB_S_MAX, 0, 0);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pTUIParamBlock);
}

void TUIParamComboBox::BeginEditParams(Interface *ip, IUtil *iu)
{

    ParamUIDesc descParam[] = {
        // Size
        ParamUIDesc(
            PB_S_POSX,
            EDITTYPE_INT,
            IDC_POSX_EDIT, IDC_POSX_SPIN,
            0, 10,
            SPIN_AUTOSCALE),
        ParamUIDesc(
            PB_S_POSY,
            EDITTYPE_INT,
            IDC_POSY_EDIT, IDC_POSY_SPIN,
            0, 10,
            SPIN_AUTOSCALE),

    };

    theTUIParam = this;
    this->iu = iu;
    this->ip = ip;

    if (pTUIParamMap)
    {
        // Left over from last TabletUI created
        pTUIParamMap->SetParamBlock(pTUIParamBlock);
    }
    else
    {

        // Gotta make a new one.

        pTUIParamMap = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                        pTUIParamBlock,
                                        ip,
                                        hInstance,
                                        MAKEINTRESOURCE(IDD_TUICOMBOBOX),
                                        GetString(IDS_TUIPARAM_EDITOR),
                                        0);
    }

    if (pTUIParamMap)
    {
        // A callback for dialog
        pTUIParamMap->SetUserDlgProc(new ParamDlgProc(this));
    }
}

TUIParamComboBox::~TUIParamComboBox()
{
    DeleteAllRefsFromMe();
    multimap<int, ComboBoxObj *>::iterator it;
    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
    {
        ComboBoxObj *obj = (*it).second;
        delete obj;
    }
    comboObjects.clear();
}

multimap<int, ComboBoxObj *>::iterator TUIParamComboBox::AddObject(int index, TSTR name)
{
    ComboBoxObj *obj = new ComboBoxObj(NULL, NULL, name);
    multimap<int, ComboBoxObj *>::iterator it = comboObjects.insert(pair<int, ComboBoxObj *>(index, obj));

    return it;
}

RefTargetHandle
TUIParamComboBox::Clone(RemapDir &remap)
{
    TUIParamComboBox *tp = new TUIParamComboBox();
    tp->ReplaceReference(0, pTUIParamBlock->Clone(remap));

    BaseClone(this, tp, remap);
    return tp;
}

void TUIParamComboBox::Init(HWND hWnd)
{

    UpdateComboObjects();

    int curSel;
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 curSel, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MIN, 0, curSel, FOREVER);
    UpdateComboBox(curSel);
    if (DefaultSelWndProc == NULL)
        SubClassDefaultSel(GetDlgItem(hWnd, IDC_TUIDEFAULT_COMBO));

    myElem->myObject->updateTabletUI();

    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 curSel, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MAX, 0, curSel, FOREVER);
    CheckDlgButton(hWnd, IDC_TUIEMPTY_CHECK, curSel);

    ICustEdit *edit = GetICustEdit(GetDlgItem(hWnd, IDC_TUIEMPTY_EDIT));
    if (edit != NULL)
    {
        edit->SetText(emptyName);
        edit->Enable(curSel);
        edit->WantReturn(TRUE);
        ReleaseICustEdit(edit);
    }
}

void TUIParamComboBox::UpdateComboObjects()
{
    multimap<int, ComboBoxObj *>::iterator it;
    multimap<int, ComboBoxObj *> tmpMap;
    Tab<bool> hitObj;
    Tab<INode *> hitNodes; /* check for duplicate nodes in switches */
    int number = 1;
    bool init = FALSE;

    hitObj.Append((int)comboObjects.size(), &init);
    for (int j = 0; j < myElem->objects.Count(); j++)
    {
        if (myElem->objects[j]->node != NULL)
        {
            SwitchObject *swObj = (SwitchObject *)myElem->objects[j]->node->EvalWorldState(0).obj;
            number += swObj->objects.Count() + 1;

            for (int i = 0; i < swObj->objects.Count(); i++)
            {
                int k = 0;
                int l = 0;
                for (l = 0; l < hitNodes.Count(); l++)
                    if (swObj->objects[i]->node == hitNodes[l])
                        break;

                if (l == hitNodes.Count())
                {
                    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
                        if ((myElem->objects[j]->node == (*it).second->switchNode) && (swObj->objects[i]->node == (*it).second->node))
                        {
                            multimap<int, ComboBoxObj *>::iterator el = tmpMap.insert(pair<int, ComboBoxObj *>(i, (*it).second));
                            (*el).second->listStr = swObj->objects[i]->listStr;
                            hitObj[k] = TRUE;
                            break;
                        }
                        else
                            k++;

                    if (it == comboObjects.end())
                    {
                        ComboBoxObj *comboObj = new ComboBoxObj(myElem->objects[j]->node, swObj->objects[i]->node, swObj->objects[i]->listStr);
                        tmpMap.insert(pair<int, ComboBoxObj *>(i, comboObj));
                    }

                    hitNodes.Append(1, &swObj->objects[i]->node);
                }
            }
        }
    }

    hitNodes.Delete(0, hitNodes.Count());

    int k = 0;
    size_t objCount = comboObjects.count(0) + 1;
    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
    {
        if (!hitObj[k])
        {
            if ((*it).first == 0)
            {
#if MAX_PRODUCT_VERSION_MAJOR > 8
                RefResult ret = ReplaceReference(k + 1, NULL);
#else
                RefResult ret = MakeRefByID(FOREVER, k + 1, NULL);
#endif
            }
            size_t index = objCount + k;
#if MAX_PRODUCT_VERSION_MAJOR > 8
            RefResult ret = ReplaceReference((int)index, NULL);
#else
            RefResult ret = MakeRefByID(FOREVER, index, NULL);
#endif
            delete (*it).second;
        }
        k++;
    }
    hitObj.Delete(0, (int)comboObjects.size());

    if (myElem->objects.Count() > 0)
    {
        comboObjects.clear();
        comboObjects = tmpMap;
    }
    UpdateRefList();
}

void TUIParamComboBox::InitDefaultChoice()
{
    HWND cb = GetDlgItem(theTUIParam->hRollup, IDC_TUIDEFAULT_COMBO);
    ComboBox_ResetContent(cb);
}

void TUIParamComboBox::UpdateComboBox(int selection)
{
    HWND cb = GetDlgItem(theTUIParam->hRollup, IDC_TUIDEFAULT_COMBO);
    ComboBox_ResetContent(cb);

    int emptyChecked = 0;
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 emptyChecked, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MAX, 0, emptyChecked, FOREVER);
    if (emptyChecked == 1)
        ComboBox_AddString(cb, emptyName);

    multimap<int, ComboBoxObj *>::iterator it = comboObjects.begin();
    int index;

    while (it != comboObjects.end())
    {
        index = (*it).first;
        TSTR name = (*it).second->comboBoxName;
        name += _T("(");

        do
        {
            name += (*it).second->listStr;
            name += _T(" ");
            it++;
        } while ((it != comboObjects.end()) && ((*it).first == index));

        name += _T(")");
        ComboBox_AddString(cb, name);
    }

    ComboBox_SetCurSel(cb, selection);
}

void TUIParamComboBox::UpdateRefList()
{

    multimap<int, ComboBoxObj *>::iterator it = comboObjects.begin();
    int i = 1;

    while ((it != comboObjects.end()) && ((*it).first == 0))
#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = ReplaceReference(i++, (*it++).second->switchNode);
#else
        RefResult ret = MakeRefByID(FOREVER, i++, (*it++).second->switchNode);
#endif

    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = ReplaceReference(i++, (*it).second->node);
#else
        RefResult ret = MakeRefByID(FOREVER, i++, (*it).second->node);
#endif
}

void TUIParamComboBox::AddSwitch(INode *addSwitch)
{
    Tab<INode *> hitNodes; /* check for duplicate nodes in switches */
    multimap<int, ComboBoxObj *>::iterator it;

    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
        hitNodes.Append(1, &(*it).second->node);

    SwitchObject *swObj = (SwitchObject *)addSwitch->EvalWorldState(0).obj;
    for (int i = 0; i < swObj->objects.Count(); i++)
    {
        int j;
        for (j = 0; j < hitNodes.Count(); j++)
            if (swObj->objects[i]->node == hitNodes[j])
                break;

        if (j == hitNodes.Count())
        {
            ComboBoxObj *comboObj = new ComboBoxObj(addSwitch, swObj->objects[i]->node, swObj->objects[i]->listStr);
            comboObjects.insert(pair<int, ComboBoxObj *>(i, comboObj));
        }
    }

    UpdateRefList();
    int curSel;
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 curSel, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MIN, 0, curSel, FOREVER);
    UpdateComboBox(curSel);
    myElem->myObject->updateTabletUI();
    static_cast<coTUIComboBox *>(myElem->myTuiElem)->setSelectedEntry(curSel);
}

void TUIParamComboBox::DelSwitch(INode *delSwitch)
{
    int oldRefs = NumRefs();
    int curSel;
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                 curSel, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MIN, 0,
                                 curSel, FOREVER);
    multimap<int, ComboBoxObj *>::iterator it;
    int numObjs = 1;
    int numSwitches = 1;

    for (it = comboObjects.begin(); it != comboObjects.end();)
        if ((*it).second->switchNode == delSwitch)
        {
            if ((*it).first == curSel)
            {
                curSel = 0;
                if (myElem->myObject->iObjParams != NULL)
                    pTUIParamBlock->SetValue(PB_S_MIN, theTUIParam->myElem->myObject->iObjParams->GetTime(), curSel);
                else
                    pTUIParamBlock->SetValue(PB_S_MIN, 0, curSel);
            }

            if ((*it).first == 0)
            {
#if MAX_PRODUCT_VERSION_MAJOR > 8
                RefResult ret = ReplaceReference(numSwitches, NULL);
#else
                RefResult ret = MakeRefByID(FOREVER, numSwitches, NULL);
#endif
            }
            int index = (int)comboObjects.count(0) + numObjs;
#if MAX_PRODUCT_VERSION_MAJOR > 8
            RefResult ret = ReplaceReference(index, NULL);
#else
            RefResult ret = MakeRefByID(FOREVER, index, NULL);
#endif
            delete (*it).second;
            it = comboObjects.erase(it);
        }
        else
        {
            it++;
            numObjs++;
            numSwitches++;
        }

    UpdateRefList();
    UpdateComboBox(curSel);
    myElem->myObject->updateTabletUI();
    static_cast<coTUIComboBox *>(myElem->myTuiElem)->setSelectedEntry(curSel);
}

void TUIParamComboBox::PrintAdditional(MAXSTREAM mStream)
{
    int value = 0;

    myElem->Indent(mStream, 2);
    printPos(pTUIParamBlock, myElem->myObject, mStream);

    pTUIParamBlock->GetValue(PB_S_MAX, 0,
                             value, FOREVER);
    myElem->Indent(mStream, 2);
    if (value == 0)
        PRINT_STR(mStream, _T("withNone"), _T("FALSE"));
    else
        PRINT_STR(mStream, _T("withNone"), _T("TRUE"));

    myElem->Indent(mStream, 2);
    if (value == 0) MSTREAMPRINTF  ("items ["));
    else MSTREAMPRINTF  ("items [ \"%s\","), emptyName);
    multimap<int, ComboBoxObj *>::iterator it;
    int index = 0;
    while ((it = comboObjects.find(index++)) != comboObjects.end())
      MSTREAMPRINTF  (" \"%s\","),(*it).second->comboBoxName);
   MSTREAMPRINTF  ("]\n"));

   pTUIParamBlock->GetValue(PB_S_MIN, 0,
                            value, FOREVER);
   myElem->Indent(mStream, 2);
   PRINT_IVALUE(mStream, _T("defaultChoice"), value);
}

int TUIParamComboBox::NumRefs()
{
    int number = (int)(comboObjects.size() + comboObjects.count(0)) + 1;
    return number;
}

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult TUIParamComboBox::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult TUIParamComboBox::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message)
#endif
{

    switch (message)
    {
    case REFMSG_TARGET_DELETED:
    {
        int oldRefs = NumRefs();
        map<INode *, int> indexMap;
        map<INode *, int>::iterator indexIt;
        // Find the ID on the list and call ResetStr
        multimap<int, ComboBoxObj *> tmpMap;
        multimap<int, ComboBoxObj *>::iterator it;
        int number = 1;
        int refIdSwitch = 1;
        int refIdObj = 1;
        int switchCount = (int)comboObjects.count(0);

        for (it = comboObjects.begin(); it != comboObjects.end(); it++)
        {
            if (((*it).second->switchNode != hTarget) && ((*it).second->node != hTarget))
            {
                pair<map<INode *, int>::iterator, bool> ret;
                ret = indexMap.insert(pair<INode *, int>((*it).second->switchNode, 0));
                if (ret.second)
                {
                    number++;
                    refIdSwitch++;
                }
                else
                    ret.first->second = ret.first->second + 1;
                tmpMap.insert(pair<int, ComboBoxObj *>(ret.first->second, (*it).second));
                number++;
                refIdObj++;
            }
            else if ((*it).second->switchNode == hTarget)
            {
                if ((*it).first == 0)
                {
                    (*it).second->switchNode = (INode *)NULL;
                    refIdSwitch++;
                }
#if MAX_PRODUCT_VERSION_MAJOR > 8
                RefResult ret = ReplaceReference(switchCount + refIdObj, NULL);
#else
                RefResult ret = MakeRefByID(FOREVER, refIdSwitch + refIdObj, NULL);
#endif
                refIdObj++;
            }
            else
            {
                if ((*it).first == 0)
                {
#if MAX_PRODUCT_VERSION_MAJOR > 8
                    RefResult ret = ReplaceReference(refIdSwitch, NULL);
#else
                    RefResult ret = MakeRefByID(FOREVER, refIdSwitch, NULL);
#endif

                    refIdSwitch++;
                }

                (*it).second->node = (INode *)NULL;
                refIdObj++;
            }
        }

        comboObjects.clear();
        comboObjects = tmpMap;
        indexMap.clear();

        UpdateRefList();
    }
    break;
    case REFMSG_NODE_NAMECHANGE:
    {
        // Find the ID on the list and call ResetStr
        multimap<int, ComboBoxObj *>::iterator it;
        for (it = comboObjects.begin(); it != comboObjects.end(); it++)
            if ((*it).second->node == hTarget)
                (*it).second->listStr.printf(_T("%s"), (*it).second->node->GetName());
    }
    break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
TUIParamComboBox::GetReference(int ind)
{
    if (ind == 0)
        return pTUIParamBlock;

    int i = 0;
    multimap<int, ComboBoxObj *>::iterator it = comboObjects.begin();
    while ((it != comboObjects.end()) && ((*it).first == 0) && (i < ind - 1))
    {
        it++;
        i++;
    }
    if (((it != comboObjects.end()) && (*it).first == 0) && (i == ind - 1))
        return (*it).second->switchNode;

    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
        if (i == ind - 1)
            return (*it).second->node;
        else
            i++;

    return NULL;
}

void
TUIParamComboBox::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pTUIParamBlock = (IParamBlock *)rtarg;
        return;
    }

    int i = 0;
    multimap<int, ComboBoxObj *>::iterator it = comboObjects.begin();
    while ((it != comboObjects.end()) && ((*it).first == 0) && (i < ind - 1))
    {
        i++;
        it++;
    }
    if ((it != comboObjects.end()) && ((*it).first == 0) && (i == ind - 1))
    {
        (*it).second->switchNode = (INode *)rtarg;
        return;
    }

    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
        if (i == ind - 1)
        {
            (*it).second->node = (INode *)rtarg;
            return;
        }
        else
            i++;
}

bool TUIParamComboBox::ReferenceLoad()
{
    pair<multimap<int, ComboBoxObj *>::iterator, multimap<int, ComboBoxObj *>::iterator> indexRange;
    multimap<int, ComboBoxObj *>::iterator it;
    Tab<int> nodeCount;

    for (int j = 0; j < myElem->objects.Count(); j++)
    {
        if (myElem->objects[j]->node != NULL)
        {
            SwitchObject *swObj = (SwitchObject *)myElem->objects[j]->node->EvalWorldState(0).obj;
            if (swObj->objects.Count() == 0)
                return false;

            for (int i = 0; i < swObj->objects.Count(); i++)
            {
                if (swObj->objects[i]->node == NULL)
                    return false;
                indexRange = comboObjects.equal_range(i);
                it = indexRange.first;
                int k = 0;
                if (nodeCount.Count() <= i)
                    nodeCount.Append(1, &k);
                while ((it != indexRange.second) && (k++ < nodeCount[i]))
                    it++;
                if (it != indexRange.second)
                {
                    (*it).second->switchNode = myElem->objects[j]->node;
                    (*it).second->node = swObj->objects[i]->node;
                }

                nodeCount[i]++;
            }
        }
    }
    UpdateRefList();

    return true;
}

#define COMBO_NAME_CHUNK 0xad33
#define COMBO_NODENAME_CHUNK 0xad34
#define COMBO_OBJECTSBEGIN_CHUNK 0xad35
#define COMBO_OBJECTSEND_CHUNK 0xad36
#define COMBO_INDEX_CHUNK 0xad37
#define COMBO_EMPTYNAME_CHUNK 0xad38

IOResult
TUIParamComboBox::Save(ISave *isave)
{

    ULONG written;
    isave->BeginChunk(COMBO_OBJECTSBEGIN_CHUNK);
    int emptyChecked = 0;
    if (myElem->myObject->iObjParams != NULL)
        pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                 emptyChecked, FOREVER);
    else
        pTUIParamBlock->GetValue(PB_S_MAX, 0, emptyChecked, FOREVER);
    isave->Write(&emptyChecked, sizeof(int), &written);
    isave->EndChunk();

    isave->BeginChunk(COMBO_EMPTYNAME_CHUNK);
    isave->WriteCString(emptyName);
    isave->EndChunk();

    multimap<int, ComboBoxObj *>::iterator it;
    for (it = comboObjects.begin(); it != comboObjects.end(); it++)
    {
        isave->BeginChunk(COMBO_INDEX_CHUNK);
        isave->Write(&(*it).first, sizeof(int), &written);
        isave->EndChunk();
        isave->BeginChunk(COMBO_NAME_CHUNK);
        isave->WriteCString((*it).second->comboBoxName);
        isave->EndChunk();
        isave->BeginChunk(COMBO_NODENAME_CHUNK);
        isave->WriteCString((*it).second->listStr);
        isave->EndChunk();
    }

    isave->BeginChunk(COMBO_OBJECTSEND_CHUNK);
    isave->EndChunk();

    return IO_OK;
}

IOResult
TUIParamComboBox::Load(ILoad *iload)
{
    TCHAR *buf;
    TSTR name;
    ULONG read;
    multimap<int, ComboBoxObj *>::iterator it;

    int emptyChecked;
    iload->Read(&emptyChecked, sizeof(int), &read);
    iload->CloseChunk();

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case COMBO_EMPTYNAME_CHUNK:
        {
            IOResult res = iload->ReadCStringChunk(&buf);
            name = buf;
            emptyName = name;
        }
        break;

        case COMBO_INDEX_CHUNK:
        {
            int index;
            iload->Read(&index, sizeof(int), &read);

            it = AddObject(index, _T(""));
        }
        break;

        case COMBO_NAME_CHUNK:
        {
            IOResult res = iload->ReadCStringChunk(&buf);
            name = buf;

            (*it).second->comboBoxName = name;
        }
        break;
        case COMBO_NODENAME_CHUNK:
        {
            IOResult res = iload->ReadCStringChunk(&buf);
            name = buf;

            (*it).second->listStr = name;
        }
        break;
        case COMBO_OBJECTSEND_CHUNK:
            return IO_OK;
            break;

        default:
            break;
        }
        iload->CloseChunk();
    }

    return IO_OK;
}
