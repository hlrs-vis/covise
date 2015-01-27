/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: backgrnd.cpp

    DESCRIPTION:  A VRML Background helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 26 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "backgrnd.h"

//------------------------------------------------------

class BackgroundClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new BackgroundObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_BACKGROUND_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Background_CLASS_ID1,
                                         Background_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static BackgroundClassDesc BackgroundDesc;

ClassDesc *GetBackgroundDesc() { return &BackgroundDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

HWND BackgroundObject::hRollup = NULL;
int BackgroundObject::dlgPrevSel = -1;

INT_PTR CALLBACK
    BackgroundImageDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    TCHAR text[MAX_PATH];
    BackgroundObject *th = (BackgroundObject *)GetWindowLongPtr(hDlg, GWLP_USERDATA);

    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        th = (BackgroundObject *)lParam;
        SetWindowLongPtr(hDlg, GWLP_USERDATA, (LONG)th);
        Edit_SetText(GetDlgItem(hDlg, IDC_BACK), th->back.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_BOTTOM), th->bottom.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_FRONT), th->front.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_LEFT), th->left.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_RIGHT), th->right.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_TOP), th->top.data());
        return TRUE;
    }

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_BACK:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_BACK), text, MAX_PATH);
                th->back = text;
            }
            break;
        case IDC_BOTTOM:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_BOTTOM), text, MAX_PATH);
                th->bottom = text;
            }
            break;
        case IDC_FRONT:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_FRONT), text, MAX_PATH);
                th->front = text;
            }
            break;
        case IDC_LEFT:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_LEFT), text, MAX_PATH);
                th->left = text;
            }
            break;
        case IDC_RIGHT:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_RIGHT), text, MAX_PATH);
                th->right = text;
            }
            break;
        case IDC_TOP:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_TOP), text, MAX_PATH);
                th->top = text;
            }
            break;
        }
        return TRUE;
    default:
        return FALSE;
    }

    return FALSE;
}

static int buttonIds[] = { IDC_1_COLOR, IDC_2_COLOR, IDC_3_COLOR };

static ParamUIDesc descSkyParam[] = {
    // Number of colors
    ParamUIDesc(PB_SKY_NUM_COLORS,
                TYPE_RADIO, buttonIds, 3),

    // Color 1
    ParamUIDesc(PB_SKY_COLOR1,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH_1),

    // Color 2
    ParamUIDesc(PB_SKY_COLOR2,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH_2),

    // Color 2 angle
    ParamUIDesc(
        PB_SKY_COLOR2_ANGLE,
        EDITTYPE_FLOAT,
        IDC_COLOR_2_ANGLE_EDIT, IDC_COLOR_2_ANGLE_SPIN,
        0.0f, 180.0f,
        1.0f, stdAngleDim),

    // Color 3
    ParamUIDesc(PB_SKY_COLOR3,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH_3),

    // Color 3 angle
    ParamUIDesc(
        PB_SKY_COLOR3_ANGLE,
        EDITTYPE_FLOAT,
        IDC_COLOR_3_ANGLE_EDIT, IDC_COLOR_3_ANGLE_SPIN,
        0.0f, 180.0f,
        1.0f, stdAngleDim),

    // Size
    ParamUIDesc(
        PB_BG_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_SKY_LENGTH 7

static ParamUIDesc descGroundParam[] = {
    // Number of colors
    ParamUIDesc(PB_GROUND_NUM_COLORS,
                TYPE_RADIO, buttonIds, 3),

    // Color 1
    ParamUIDesc(PB_GROUND_COLOR1,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH_1),

    // Color 2
    ParamUIDesc(PB_GROUND_COLOR2,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH_2),

    // Color 2 angle
    ParamUIDesc(
        PB_GROUND_COLOR2_ANGLE,
        EDITTYPE_FLOAT,
        IDC_COLOR_2_ANGLE_EDIT, IDC_COLOR_2_ANGLE_SPIN,
        0.0f, 180.0f,
        1.0f, stdAngleDim),

    // Color 3
    ParamUIDesc(PB_GROUND_COLOR3,
                TYPE_COLORSWATCH, IDC_COLOR_SWATCH_3),

    // Color 3 angle
    ParamUIDesc(
        PB_GROUND_COLOR3_ANGLE,
        EDITTYPE_FLOAT,
        IDC_COLOR_3_ANGLE_EDIT, IDC_COLOR_3_ANGLE_SPIN,
        0.0f, 180.0f,
        1.0f, stdAngleDim),

};

#define PARAMDESC_GROUND_LENGTH 6

static ParamBlockDescID descVer0[] = {
    { TYPE_INT, NULL, FALSE, 0 },
    { TYPE_RGBA, NULL, FALSE, 1 },
    { TYPE_RGBA, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 },
    { TYPE_RGBA, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_INT, NULL, FALSE, 6 },
    { TYPE_RGBA, NULL, FALSE, 7 },
    { TYPE_RGBA, NULL, FALSE, 8 },
    { TYPE_FLOAT, NULL, FALSE, 9 },
    { TYPE_RGBA, NULL, FALSE, 10 },
    { TYPE_FLOAT, NULL, FALSE, 11 },
    { TYPE_FLOAT, NULL, FALSE, 12 },
};

// Current version
static ParamVersionDesc curVersion(descVer0, PB_BG_LENGTH, 0);
#define CURRENT_VERSION 0

IParamMap *BackgroundObject::skyParam = NULL;
IParamMap *BackgroundObject::groundParam = NULL;

static HWND imgDlg;

void
BackgroundObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                  Animatable *prev)
{
    iObjParams = ip;

    if (skyParam)
    {

        // Left over from last Background created
        skyParam->SetParamBlock(pblock);
        groundParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        skyParam = CreateCPParamMap(descSkyParam, PARAMDESC_SKY_LENGTH,
                                    pblock,
                                    ip,
                                    hInstance,
                                    MAKEINTRESOURCE(IDD_BACKGROUND_COLORS),
                                    GetString(IDS_SKY_COLORS),
                                    0);
        groundParam = CreateCPParamMap(descGroundParam, PARAMDESC_GROUND_LENGTH,
                                       pblock,
                                       ip,
                                       hInstance,
                                       MAKEINTRESOURCE(IDD_BACKGROUND_COLORS1),
                                       GetString(IDS_GROUND_COLORS),
                                       APPENDROLL_CLOSED);
        // imageDlg = ip->AddRollupPage(hInstance,
        imgDlg = GetCOREInterface()->AddRollupPage(hInstance,
                                                   MAKEINTRESOURCE(IDD_BACKGROUND_IMAGES),
                                                   BackgroundImageDlgProc,
                                                   GetString(IDS_IMAGES),
                                                   (LPARAM) this,
                                                   APPENDROLL_CLOSED);
    }
}

void
BackgroundObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (skyParam)
            DestroyCPParamMap(skyParam);
        skyParam = NULL;
        if (groundParam)
            DestroyCPParamMap(groundParam);
        groundParam = NULL;
        // if (imageDlg) ip->DeleteRollupPage(imageDlg);
        if (imgDlg)
            GetCOREInterface()->DeleteRollupPage(imgDlg);
        imgDlg = NULL;
    }
}

BackgroundObject::BackgroundObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_BG_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_SKY_NUM_COLORS, 0, 0);
    pb->SetValue(PB_SKY_COLOR1, 0, Point3(0.0f, 0.0f, 1.0f));
    pb->SetValue(PB_SKY_COLOR2, 0, Point3(1.0f, 0.0f, 0.0f));
    pb->SetValue(PB_SKY_COLOR3, 0, Point3(0.0f, 1.0f, 0.0f));
    pb->SetValue(PB_SKY_COLOR2_ANGLE, 0, float(PI / 4.0));
    pb->SetValue(PB_SKY_COLOR3_ANGLE, 0, float(PI / 2.0));
    pb->SetValue(PB_GROUND_NUM_COLORS, 0, 0);
    pb->SetValue(PB_GROUND_COLOR1, 0, Point3(0.7f, 0.4f, 0.3f));
    pb->SetValue(PB_GROUND_COLOR2, 0, Point3(1.0f, 0.0f, 0.0f));
    pb->SetValue(PB_GROUND_COLOR3, 0, Point3(0.0f, 1.0f, 0.0f));
    pb->SetValue(PB_GROUND_COLOR2_ANGLE, 0, float(PI / 4.0));
    pb->SetValue(PB_GROUND_COLOR3_ANGLE, 0, float(PI / 2.0));
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

BackgroundObject::~BackgroundObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *BackgroundObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult BackgroundObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult BackgroundObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                             PartID &partID, RefMessage message)
#endif
{
    //     int i;
    //     switch (message) {
    //     }
    return REF_SUCCEED;
}

RefTargetHandle
BackgroundObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    return NULL;
}

void
BackgroundObject::SetReference(int ind, RefTargetHandle rtarg)
{
    pblock = (IParamBlock *)rtarg;
}

ObjectState
BackgroundObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
BackgroundObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
BackgroundObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
BackgroundObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                   Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
BackgroundObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
BackgroundObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_BG_SIZE, t, size, FOREVER);
#include "bgob.cpp"
    mesh.buildBoundingBox();
}

int
BackgroundObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_BG_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 1.0f, 1.0f, 0.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
BackgroundObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class BackgroundCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    BackgroundObject *backgroundObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(BackgroundObject *obj) { backgroundObject = obj; }
};

int
BackgroundCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            backgroundObject->pblock->SetValue(PB_BG_SIZE,
                                               backgroundObject->iObjParams->GetTime(), radius);
            backgroundObject->skyParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(backgroundObject->iObjParams->SnapAngle(ang));
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
static BackgroundCreateCallBack BackgroundCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
BackgroundObject::GetCreateMouseCallBack()
{
    BackgroundCreateCB.SetObj(this);
    return (&BackgroundCreateCB);
}

#define BACK_CHUNK 0xad00
#define BOTTOM_CHUNK 0xad01
#define FRONT_CHUNK 0xad02
#define LEFT_CHUNK 0xad03
#define RIGHT_CHUNK 0xad04
#define TOP_CHUNK 0xad05

IOResult
BackgroundObject::Save(ISave *isave)
{
    isave->BeginChunk(BACK_CHUNK);
    isave->WriteCString(back.data());
    isave->EndChunk();

    isave->BeginChunk(BOTTOM_CHUNK);
    isave->WriteCString(bottom.data());
    isave->EndChunk();

    isave->BeginChunk(FRONT_CHUNK);
    isave->WriteCString(front.data());
    isave->EndChunk();

    isave->BeginChunk(LEFT_CHUNK);
    isave->WriteCString(left.data());
    isave->EndChunk();

    isave->BeginChunk(RIGHT_CHUNK);
    isave->WriteCString(right.data());
    isave->EndChunk();

    isave->BeginChunk(TOP_CHUNK);
    isave->WriteCString(top.data());
    isave->EndChunk();

    return IO_OK;
}

IOResult
BackgroundObject::Load(ILoad *iload)
{
    TCHAR *txt;

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case BACK_CHUNK:
            iload->ReadCStringChunk(&txt);
            back = txt;
            break;

        case BOTTOM_CHUNK:
            iload->ReadCStringChunk(&txt);
            bottom = txt;
            break;

        case FRONT_CHUNK:
            iload->ReadCStringChunk(&txt);
            front = txt;
            break;

        case LEFT_CHUNK:
            iload->ReadCStringChunk(&txt);
            left = txt;
            break;

        case RIGHT_CHUNK:
            iload->ReadCStringChunk(&txt);
            right = txt;
            break;

        case TOP_CHUNK:
            iload->ReadCStringChunk(&txt);
            top = txt;
            break;
        default:
            break;
        }
        iload->CloseChunk();
    }
    return IO_OK;
}

RefTargetHandle
BackgroundObject::Clone(RemapDir &remap)
{
    BackgroundObject *bg = new BackgroundObject();
    bg->ReplaceReference(0, pblock->Clone(remap));
    bg->back = back;
    bg->bottom = bottom;
    bg->front = front;
    bg->left = left;
    bg->right = right;
    bg->top = top;
    BaseClone(this, bg, remap);
    return bg;
}
