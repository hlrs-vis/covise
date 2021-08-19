/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: thermal.cpp

    DESCRIPTION:  A VRML 2.0 Thermal Helper object
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 29 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "thermal.h"

#define SEGMENTS 32

ISpinnerControl* ThermalObject::heightSpin;
ISpinnerControl* ThermalObject::turbulenceSpin;
ISpinnerControl* ThermalObject::vxSpin;
ISpinnerControl* ThermalObject::vySpin;
ISpinnerControl* ThermalObject::vzSpin;
ISpinnerControl* ThermalObject::minBackSpin;
ISpinnerControl *ThermalObject::maxBackSpin;
ISpinnerControl *ThermalObject::minFrontSpin;
ISpinnerControl *ThermalObject::maxFrontSpin;

//------------------------------------------------------

class ThermalClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new ThermalObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_THERMAL_CLASS); }
    const TCHAR* NonLocalizedClassName() { return _T("Thermal"); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Thermal_CLASS_ID1,
                                         Thermal_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static ThermalClassDesc ThermalDesc;

ClassDesc *GetThermalDesc() { return &ThermalDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;


HWND ThermalObject::hRollup = NULL;
int ThermalObject::dlgPrevSel = -1;



#define RELEASE_SPIN(x)         \
    if (th->x)                  \
    {                           \
        ReleaseISpinner(th->x); \
        th->x = NULL;           \
    }
#define RELEASE_BUT(x)             \
    if (th->x)                     \
    {                              \
        ReleaseICustButton(th->x); \
        th->x = NULL;              \
    }

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     ThermalObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        float minF, minB, maxF, maxB, turbulence,vx,vy,vz,height;
        TimeValue t = 0;

        th->pblock->GetValue(PB_THERMAL_MIN_FRONT, t, minF, FOREVER);
        th->pblock->GetValue(PB_THERMAL_MIN_BACK, t, minB, FOREVER);
        th->pblock->GetValue(PB_THERMAL_MAX_FRONT, t, maxF, FOREVER);
        th->pblock->GetValue(PB_THERMAL_MAX_BACK, t, maxB, FOREVER);
        th->pblock->GetValue(PB_THERMAL_TURBULENCE, t, turbulence, FOREVER);
        th->pblock->GetValue(PB_THERMAL_VX, t, vx, FOREVER);
        th->pblock->GetValue(PB_THERMAL_VY, t, vy, FOREVER);
        th->pblock->GetValue(PB_THERMAL_VZ, t, vz, FOREVER);
        th->pblock->GetValue(PB_THERMAL_HEIGHT, t, height, FOREVER);
        // Now we need to fill in the list box IDC_Thermal_LIST
        th->hRollup = hDlg;

        return TRUE;
    }
    case WM_DESTROY:
        // th->iObjParams->ClearPickMode();
        // th->previousMode = NULL;
        RELEASE_SPIN(minBackSpin);
        RELEASE_SPIN(maxBackSpin);
        RELEASE_SPIN(minFrontSpin);
        RELEASE_SPIN(maxFrontSpin);

        return FALSE;

    case CC_SPINNER_BUTTONDOWN:
    {
        int id = LOWORD(wParam);
        switch (id)
        {
        case IDC_MIN_BACK_SPIN:
        case IDC_MAX_BACK_SPIN:
        case IDC_MIN_FRONT_SPIN:
        case IDC_MAX_FRONT_SPIN:
            theHold.Begin();
            return TRUE;
        default:
            return FALSE;
        }
    }
    break;

    case CC_SPINNER_BUTTONUP:
    {
        if (!HIWORD(wParam))
        {
            theHold.Cancel();
            break;
        }
        int id = LOWORD(wParam);
        switch (id)
        {
        case IDC_MIN_BACK_SPIN:
        case IDC_MAX_BACK_SPIN:
        case IDC_MIN_FRONT_SPIN:
        case IDC_MAX_FRONT_SPIN:
            theHold.Accept(GetString(IDS_DS_PARAMCHG));
            return TRUE;
        default:
            return FALSE;
        }
    }

    case CC_SPINNER_CHANGE:
    {
        float minB, maxB, minF, maxF;
        int id = LOWORD(wParam);
        TimeValue t = 0; // not really needed...yet

        switch (id)
        {
        case IDC_MIN_BACK_SPIN:
        case IDC_MAX_BACK_SPIN:
        case IDC_MIN_FRONT_SPIN:
        case IDC_MAX_FRONT_SPIN:
            if (!HIWORD(wParam))
                theHold.Begin();

            minB = th->minBackSpin->GetFVal();
            maxB = th->maxBackSpin->GetFVal();
            minF = th->minFrontSpin->GetFVal();
            maxF = th->maxFrontSpin->GetFVal();

            if (minB > minF)
            {
                if (id == IDC_MIN_BACK_SPIN)
                    minF = minB;
                else if (id == IDC_MIN_FRONT_SPIN)
                    minB = minF;
            }
            if (maxB > maxF)
            {
                if (id == IDC_MAX_BACK_SPIN)
                    maxF = maxB;
                else if (id == IDC_MAX_FRONT_SPIN)
                    maxB = maxF;
            }
            if (minB > maxB)
            {
                if (id == IDC_MIN_BACK_SPIN)
                    maxB = minB;
                else
                    minB = maxB;
            }
            if (minF > maxF)
            {
                if (id == IDC_MIN_FRONT_SPIN)
                    maxF = minF;
                else
                    minF = maxF;
            }

            th->minBackSpin->SetValue(minB, FALSE);
            th->maxBackSpin->SetValue(maxB, FALSE);
            th->minFrontSpin->SetValue(minF, FALSE);
            th->maxFrontSpin->SetValue(maxF, FALSE);
            th->SetMinBack(t, minB);
            th->SetMaxBack(t, maxB);
            th->SetMinFront(t, minF);
            th->SetMaxFront(t, maxF);

            if (!HIWORD(wParam))
                theHold.Accept(GetString(IDS_DS_PARAMCHG));
            return TRUE;
        default:
            return FALSE;
        }
    }

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_AUDIO_CLIP_PICK: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                /*
                if (th->previousMode) {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                } else {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    thePick.SetThermal(th);
                    th->iObjParams->SetPickMode(&thePick);
                }
                */
                return TRUE;
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
    // Turbulence
    ParamUIDesc(
        PB_THERMAL_TURBULENCE,
        EDITTYPE_UNIVERSE,
        IDC_TURBULENCE_EDIT, IDC_TURBULENCE_SPIN,
        0.0f, 10.0f,
        0.1f),

    // VX
    ParamUIDesc(
        PB_THERMAL_VX,
        EDITTYPE_UNIVERSE,
        IDC_VX_EDIT, IDC_VX_SPIN,
        0.0f, 100.0f,
        1.0f),
    // VY
    ParamUIDesc(
        PB_THERMAL_VY,
        EDITTYPE_UNIVERSE,
        IDC_VY_EDIT, IDC_VY_SPIN,
        0.0f, 100.0f,
        1.0f),
    // VX
    ParamUIDesc(
        PB_THERMAL_VZ,
        EDITTYPE_UNIVERSE,
        IDC_VZ_EDIT, IDC_VZ_SPIN,
        0.0f, 100.0f,
        1.0f),

    // HEIGHT
    ParamUIDesc(
        PB_THERMAL_HEIGHT,
        EDITTYPE_UNIVERSE,
        IDC_HEIGHT_EDIT, IDC_HEIGHT_SPIN,
        0.0f, 1000.0f,
        10.0f),

    // Min Back
    ParamUIDesc(
        PB_THERMAL_MIN_BACK,
        EDITTYPE_UNIVERSE,
        IDC_MIN_BACK_EDIT, IDC_MIN_BACK_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Max Back
    ParamUIDesc(
        PB_THERMAL_MAX_BACK,
        EDITTYPE_UNIVERSE,
        IDC_MAX_BACK_EDIT, IDC_MAX_BACK_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Min Front
    ParamUIDesc(
        PB_THERMAL_MIN_FRONT,
        EDITTYPE_UNIVERSE,
        IDC_MIN_FRONT_EDIT, IDC_MIN_FRONT_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Max Front
    ParamUIDesc(
        PB_THERMAL_MAX_FRONT,
        EDITTYPE_UNIVERSE,
        IDC_MAX_FRONT_EDIT, IDC_MAX_FRONT_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Icon Size
    ParamUIDesc(
        PB_THERMAL_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 10

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_FLOAT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 },
    { TYPE_FLOAT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
    { TYPE_FLOAT, NULL, FALSE, 7 },
    { TYPE_FLOAT, NULL, FALSE, 8 },
    { TYPE_FLOAT, NULL, FALSE, 9 },
};

#define CURRENT_VERSION 0

class ThermalParamDlgProc : public ParamMapUserDlgProc
{
public:
    ThermalObject *ob;

    ThermalParamDlgProc(ThermalObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR ThermalParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                   UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *ThermalObject::pmapParam = NULL;

float ThermalObject::GetHeight(TimeValue t, Interval& valid)
{
    Interval iv;

    float h;
    pblock->GetValue(PB_THERMAL_MIN_BACK, t, h, valid);
    return h;
}
float ThermalObject::GetTurbulence(TimeValue t, Interval& valid)
{
    Interval iv;

    float h;
    pblock->GetValue(PB_THERMAL_TURBULENCE, t, h, valid);
    return h;
}
float ThermalObject::GetVX(TimeValue t, Interval& valid)
{
    Interval iv;

    float h;
    pblock->GetValue(PB_THERMAL_VX, t, h, valid);
    return h;
}
float ThermalObject::GetVY(TimeValue t, Interval& valid)
{
    Interval iv;

    float h;
    pblock->GetValue(PB_THERMAL_VY, t, h, valid);
    return h;
}
float ThermalObject::GetVZ(TimeValue t, Interval& valid)
{
    Interval iv;

    float h;
    pblock->GetValue(PB_THERMAL_VZ, t, h, valid);
    return h;
}
float ThermalObject::GetMinBack(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_THERMAL_MIN_BACK, t, f, valid);
    pblock->GetValue(PB_THERMAL_MAX_BACK, t, g, valid);
    if (g < f)
        return g;
    return f;
}

float ThermalObject::GetMaxBack(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_THERMAL_MAX_BACK, t, f, valid);
    pblock->GetValue(PB_THERMAL_MIN_BACK, t, g, valid);
    if (g > f)
        return g;
    return f;
}

float ThermalObject::GetMinFront(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_THERMAL_MIN_FRONT, t, f, valid);
    pblock->GetValue(PB_THERMAL_MAX_FRONT, t, g, valid);
    if (g < f)
        return g;
    return f;
}

float ThermalObject::GetMaxFront(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_THERMAL_MAX_FRONT, t, f, valid);
    pblock->GetValue(PB_THERMAL_MIN_FRONT, t, g, valid);
    if (g > f)
        return g;
    return f;
}

void ThermalObject::SetHeight(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_HEIGHT, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}
void ThermalObject::SetTurbulence(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_TURBULENCE, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}
void ThermalObject::SetVX(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_VX, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}
void ThermalObject::SetVY(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_VY, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}
void ThermalObject::SetVZ(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_VZ, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}
void ThermalObject::SetMinBack(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_MIN_BACK, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void ThermalObject::SetMaxBack(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_MAX_BACK, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void ThermalObject::SetMinFront(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_MIN_FRONT, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void ThermalObject::SetMaxFront(TimeValue t, float f)
{
    pblock->SetValue(PB_THERMAL_MAX_FRONT, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void
ThermalObject::BeginEditParams(IObjParam *ip, ULONG flags,
                             Animatable *prev)
{
    iObjParams = ip;
    TimeValue t = ip->GetTime(); // not really needed...yet

    if (pmapParam)
    {

        // Left over from last Thermal created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_THERMAL),
                                     _T("Thermal" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new ThermalParamDlgProc(this));
    }

    turbulenceSpin = GetISpinner(GetDlgItem(hRollup, IDC_TURBULENCE_SPIN));
    vxSpin = GetISpinner(GetDlgItem(hRollup, IDC_VX_SPIN));
    vySpin = GetISpinner(GetDlgItem(hRollup, IDC_VY_SPIN));
    vzSpin = GetISpinner(GetDlgItem(hRollup, IDC_VZ_SPIN));
    minBackSpin = GetISpinner(GetDlgItem(hRollup, IDC_MIN_BACK_SPIN));
    maxBackSpin = GetISpinner(GetDlgItem(hRollup, IDC_MAX_BACK_SPIN));
    minFrontSpin = GetISpinner(GetDlgItem(hRollup, IDC_MIN_FRONT_SPIN));
    maxFrontSpin = GetISpinner(GetDlgItem(hRollup, IDC_MAX_FRONT_SPIN));
    heightSpin = GetISpinner(GetDlgItem(hRollup, IDC_HEIGHT_SPIN));

    turbulenceSpin->SetLimits(0.0f, 10.0f, FALSE);
    turbulenceSpin->SetValue(GetTurbulence(t), FALSE);
    turbulenceSpin->SetScale(1.0f);
    turbulenceSpin->LinkToEdit(GetDlgItem(hRollup, IDC_TURBULENCE_EDIT), EDITTYPE_UNIVERSE);

    vxSpin->SetLimits(0.0f, 100.0f, FALSE);
    vxSpin->SetValue(GetVX(t), FALSE);
    vxSpin->SetScale(1.0f);
    vxSpin->LinkToEdit(GetDlgItem(hRollup, IDC_VX_EDIT), EDITTYPE_UNIVERSE);
    vySpin->SetLimits(0.0f, 100.0f, FALSE);
    vySpin->SetValue(GetVY(t), FALSE);
    vySpin->SetScale(1.0f);
    vySpin->LinkToEdit(GetDlgItem(hRollup, IDC_VY_EDIT), EDITTYPE_UNIVERSE);
    vzSpin->SetLimits(0.0f, 100.0f, FALSE);
    vzSpin->SetValue(GetVZ(t), FALSE);
    vzSpin->SetScale(1.0f);
    vzSpin->LinkToEdit(GetDlgItem(hRollup, IDC_VZ_EDIT), EDITTYPE_UNIVERSE);

    minBackSpin->SetLimits(0.0f, 10000.0f, FALSE);
    minBackSpin->SetValue(GetMinBack(t), FALSE);
    minBackSpin->SetScale(1.0f);
    minBackSpin->LinkToEdit(GetDlgItem(hRollup, IDC_MIN_BACK_EDIT), EDITTYPE_UNIVERSE);
    maxBackSpin->SetLimits(0.0f, 10000.0f, FALSE);
    maxBackSpin->SetValue(GetMaxBack(t), FALSE);
    maxBackSpin->SetScale(1.0f);
    maxBackSpin->LinkToEdit(GetDlgItem(hRollup, IDC_MAX_BACK_EDIT), EDITTYPE_UNIVERSE);
    minFrontSpin->SetLimits(0.0f, 10000.0f, FALSE);
    minFrontSpin->SetValue(GetMinFront(t), FALSE);
    minFrontSpin->SetScale(1.0f);
    minFrontSpin->LinkToEdit(GetDlgItem(hRollup, IDC_MIN_FRONT_EDIT), EDITTYPE_UNIVERSE);
    maxFrontSpin->SetLimits(0.0f, 10000.0f, FALSE);
    maxFrontSpin->SetValue(GetMaxFront(t), FALSE);
    maxFrontSpin->SetScale(1.0f);
    maxFrontSpin->LinkToEdit(GetDlgItem(hRollup, IDC_MAX_FRONT_EDIT), EDITTYPE_UNIVERSE);

    heightSpin->SetLimits(0.0f, 100.0f, FALSE);
    heightSpin->SetValue(GetHeight(t), FALSE);
    heightSpin->SetScale(1.0f);
    heightSpin->LinkToEdit(GetDlgItem(hRollup, IDC_HEIGHT_EDIT), EDITTYPE_UNIVERSE);
}

void
ThermalObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

ThermalObject::ThermalObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_THERMAL_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_THERMAL_SIZE, 0, 0.0f);
    pb->SetValue(PB_THERMAL_TURBULENCE, 0, 0.0f);
    pb->SetValue(PB_THERMAL_MAX_BACK, 0, 70.0f);
    pb->SetValue(PB_THERMAL_MIN_BACK, 0, 50.0f);
    pb->SetValue(PB_THERMAL_MAX_FRONT, 0, 70.0f);
    pb->SetValue(PB_THERMAL_MIN_FRONT, 0, 50.0f);
    pb->SetValue(PB_THERMAL_HEIGHT, 0, 100.0f);
    pb->SetValue(PB_THERMAL_VX, 0, 0.0f);
    pb->SetValue(PB_THERMAL_VY, 0, 6.0f);
    pb->SetValue(PB_THERMAL_VZ, 0, 0.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

ThermalObject::~ThermalObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *ThermalObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult ThermalObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult ThermalObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message)
#endif
{
    return REF_SUCCEED;
}

RefTargetHandle
ThermalObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;

    return NULL;
}

void
ThermalObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
    }
}

ObjectState
ThermalObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
ThermalObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
ThermalObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
GetEllipsePoints(float front, float back, float height, Point3 *ellipse)
{
    float a = (back + front) / 2.0f;
    float c = (front - back) / 2.0f;
    float b = (float)sqrt(a * a - c * c);
    int i;
    float delTheta = (2 * PI) / float(SEGMENTS);
    float theta;
    for (i = 0, theta = 0.0f; i < SEGMENTS; i++, theta += delTheta)
    {
        ellipse[i].x = b * float(cos(theta));
        ellipse[i].y = -(a * float(sin(theta)) + c);
        ellipse[i].z = 0.0f;
    }
    for (i = 0; i < SEGMENTS; i++)
    {
        ellipse[SEGMENTS + i].x = ellipse[i].x;
        ellipse[SEGMENTS + i].y = ellipse[i].y;
        ellipse[SEGMENTS + i].z = height;
    }
}

void
ThermalObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
    if (inode->Selected())
    {
        float minFront, minBack, maxFront, maxBack, height;
        pblock->GetValue(PB_THERMAL_MIN_FRONT, t, minFront, FOREVER);
        pblock->GetValue(PB_THERMAL_MIN_BACK, t, minBack, FOREVER);
        pblock->GetValue(PB_THERMAL_MAX_FRONT, t, maxFront, FOREVER);
        pblock->GetValue(PB_THERMAL_MAX_BACK, t, maxBack, FOREVER);
        pblock->GetValue(PB_THERMAL_HEIGHT, t, height, FOREVER);

        Point3 ellipse[2 * SEGMENTS];
        GetEllipsePoints(minFront, minBack, height, ellipse);
        int i;
        for (i = 0; i < 2 * SEGMENTS; i++)
        {
            box += ellipse[i];
        }
        GetEllipsePoints(maxFront, maxBack, height, ellipse);
        for (i = 0; i < 2 * SEGMENTS; i++)
        {
            box += ellipse[i];
        }
    }
}

void
ThermalObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    Matrix3 tm;
    BuildMesh(t); // 000829  --prs.
    GetMat(t, inode, vpt, tm);

    int nv = mesh.getNumVerts();
    GetLocalBoundBox(t, inode, vpt, box);
    box = box * tm;
    //    for (int i=0; i<nv; i++)
    //        box += tm*mesh.getVert(i);
}

void
ThermalObject::BuildMesh(TimeValue t)
{
    float length;
    pblock->GetValue(PB_THERMAL_SIZE, t, length, FOREVER);
    mesh.setNumVerts(11);
    mesh.setNumFaces(5);
    mesh.setVert(0, Point3(-length, 0.0f, length));
    mesh.setVert(1, Point3(length, 0.0f, length));
    mesh.setVert(2, Point3(length, 0.0f, -length));
    mesh.setVert(3, Point3(-length, 0.0f, -length));
    mesh.setVert(4, Point3(0.0f, 0.0f, 0.0f));
    mesh.setVert(5, Point3(0.0f, -3 * length, 0.0f));
    mesh.setVert(6, Point3(0.0f, 0.0f, 0.0f));
    length *= 0.3f;
    mesh.setVert(7, Point3(-length, -7.0f * length, length));
    mesh.setVert(8, Point3(length, -7.0f * length, length));
    mesh.setVert(9, Point3(length, -7.0f * length, -length));
    mesh.setVert(10, Point3(-length, -7.0f * length, -length));

    mesh.faces[0].setEdgeVisFlags(1, 0, 1);
    mesh.faces[0].setSmGroup(1);
    mesh.faces[0].setVerts(0, 1, 3);

    mesh.faces[1].setEdgeVisFlags(1, 1, 0);
    mesh.faces[1].setSmGroup(1);
    mesh.faces[1].setVerts(1, 2, 3);

    mesh.faces[2].setEdgeVisFlags(1, 1, 0);
    mesh.faces[2].setSmGroup(1);
    mesh.faces[2].setVerts(4, 5, 6);

    mesh.faces[3].setEdgeVisFlags(0, 1, 1);
    mesh.faces[3].setSmGroup(1);
    mesh.faces[3].setVerts(7, 9, 5);

    mesh.faces[4].setEdgeVisFlags(0, 1, 1);
    mesh.faces[4].setSmGroup(1);
    mesh.faces[4].setVerts(10, 8, 5);

    mesh.InvalidateGeomCache();
    mesh.EnableEdgeList(1);
    mesh.buildBoundingBox();
}

void
ThermalObject::DrawEllipsoids(TimeValue t, INode *inode, GraphicsWindow *gw)
{
    Matrix3 tm = inode->GetObjectTM(t);
    gw->setTransform(tm);
    float minFront, minBack, maxFront, maxBack, height;
    pblock->GetValue(PB_THERMAL_MIN_FRONT, t, minFront, FOREVER);
    pblock->GetValue(PB_THERMAL_MIN_BACK, t, minBack, FOREVER);
    pblock->GetValue(PB_THERMAL_MAX_FRONT, t, maxFront, FOREVER);
    pblock->GetValue(PB_THERMAL_MAX_BACK, t, maxBack, FOREVER);
    pblock->GetValue(PB_THERMAL_HEIGHT, t, height, FOREVER);

    Point3 ellipse[2 * SEGMENTS + 1];

    // Draw the Min ellipsoid in Blue
    gw->setColor(LINE_COLOR, 0.0f, 0.0f, 1.0f);
    GetEllipsePoints(minFront, minBack, height, ellipse);
    gw->polyline(SEGMENTS, ellipse, NULL, NULL, TRUE, NULL);
    gw->polyline(SEGMENTS, ellipse + SEGMENTS, NULL, NULL, TRUE, NULL);
    Point3 ellipseLine[2 * 4];
    ellipseLine[0] = ellipse[0];
    ellipseLine[1] = ellipse[SEGMENTS];
    ellipseLine[2] = ellipse[0 + SEGMENTS / 2];
    ellipseLine[3] = ellipse[SEGMENTS + SEGMENTS / 2];
    gw->polyline(2, ellipseLine, NULL, NULL, TRUE, NULL);
    gw->polyline(2, ellipseLine + 2, NULL, NULL, TRUE, NULL);

    // Draw the Max ellipsoid in Red
    gw->setColor(LINE_COLOR, 1.0f, 0.0f, 0.0f);
    GetEllipsePoints(maxFront, maxBack, height, ellipse);
    gw->polyline(SEGMENTS, ellipse, NULL, NULL, TRUE, NULL);
    gw->polyline(SEGMENTS, ellipse + SEGMENTS, NULL, NULL, TRUE, NULL);

    ellipseLine[0] = ellipse[0];
    ellipseLine[1] = ellipse[SEGMENTS];
    ellipseLine[2] = ellipse[0 + SEGMENTS / 2];
    ellipseLine[3] = ellipse[SEGMENTS + SEGMENTS / 2];
    gw->polyline(2, ellipseLine, NULL, NULL, TRUE, NULL);
    gw->polyline(2, ellipseLine+2, NULL, NULL, TRUE, NULL);
}

int
ThermalObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_THERMAL_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 0.5f, 0.3f, 1.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    if (inode->Selected())
    {
        DrawEllipsoids(t, inode, gw);
    }

    gw->setRndLimits(rlim);
    return (0);
}

class ThermalCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    ThermalObject *thermalObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(ThermalObject *obj) { thermalObject = obj; }
};

int
ThermalCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            thermalObject->pblock->SetValue(PB_THERMAL_SIZE,
                                          thermalObject->iObjParams->GetTime(), radius);
            thermalObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(thermalObject->iObjParams->SnapAngle(ang));
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
static ThermalCreateCallBack ThermalCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
ThermalObject::GetCreateMouseCallBack()
{
    ThermalCreateCB.SetObj(this);
    return (&ThermalCreateCB);
}

RefTargetHandle
ThermalObject::Clone(RemapDir &remap)
{
    ThermalObject *ts = new ThermalObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    BaseClone(this, ts, remap);
    return ts;
}
