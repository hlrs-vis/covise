/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: bboard.cpp

    DESCRIPTION:  A VRML Billboard VRML 2.0 helper object
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 18 Sept, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "bboard.h"
#include "iFnPub.h"

//------------------------------------------------------

class BillboardClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new BillboardObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_BILLBOARD_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Billboard_CLASS_ID1,
                                         Billboard_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static BillboardClassDesc BillboardDesc;

ClassDesc *GetBillboardDesc() { return &BillboardDesc; }
//===========================================================================
// IBillboard Interface descriptor
//===========================================================================
// Assembly Manager interface instance and descriptor
FPInterfaceDesc BillboardObject::mInterfaceDesc(
    BILLBOARDOBJECT_INTERFACE, // Interface id
    _T("IBillboard"), // Interface name used by maxscript - don't localize it!
    0, // Res ID of description string
    &BillboardDesc, // Class descriptor
    FP_MIXIN,

    // - Methods -

    // - Properties -
    properties,

    IBillboard::kBILL_GET_SIZE, IBillboard::kBILL_SET_SIZE, _T("Size"), 0, TYPE_FLOAT,
    IBillboard::kBILL_GET_SCREEN_ALIGN, IBillboard::kBILL_SET_SCREEN_ALIGN, _T("ScreenAlign"), 0, TYPE_BOOL,
#if MAX_PRODUCT_VERSION_MAJOR > 14
    p_end
#else
    end
#endif
    );

//---------------------------------------------------------------------------
//
BaseInterface *BillboardObject::GetInterface(Interface_ID id)
{
    if (id == BILLBOARDOBJECT_INTERFACE)
        return static_cast<IBillboard *>(this);
    else
        return HelperObject::GetInterface(id);
}
// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

HWND BillboardObject::hRollup = NULL;
int BillboardObject::dlgPrevSel = -1;

static ParamUIDesc descParam[] = {
    // Size
    ParamUIDesc(
        PB_BB_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 1000.0f,
        SPIN_AUTOSCALE),

    // Loop
    ParamUIDesc(PB_BB_SCREEN_ALIGN, TYPE_SINGLECHEKBOX, IDC_ALIGN),
};

ParamDimension *BillboardObject::GetParameterDim(int pbIndex)
{
    switch (pbIndex)
    {
    case PB_BB_SIZE:
        return stdWorldDim;
    case PB_BB_SCREEN_ALIGN:
        return defaultDim;
    default:
        return defaultDim;
    }
}

TSTR BillboardObject::GetParameterName(int pbIndex)
{
    switch (pbIndex)
    {
    case PB_BB_SIZE:
        return TSTR(GetString(IDS_BB_SIZE));
    case PB_BB_SCREEN_ALIGN:
        return GetString(IDS_BB_SCREEN_ALIGN);
    default:
        return TSTR(_T(""));
    }
}

#define PARAMDESC_LENGTH 2

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
};

#define NUM_OLD_VERSIONS 0

// Current version
#define CURRENT_VERSION 0

IParamMap *BillboardObject::pmapParam = NULL;

void
BillboardObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                 Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last Billboard created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_BBOARD),
                                     _T("Billboard" /*JP_LOC*/),
                                     0);
    }
}

void
BillboardObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

BillboardObject::BillboardObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_BB_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_BB_SCREEN_ALIGN, 0, FALSE);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
}

BillboardObject::~BillboardObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *BillboardObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult BillboardObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                            PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult BillboardObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                            PartID &partID, RefMessage message)
#endif
{
    return REF_SUCCEED;
}

RefTargetHandle
BillboardObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    return NULL;
}

void
BillboardObject::SetReference(int ind, RefTargetHandle rtarg)
{
    pblock = (IParamBlock *)rtarg;
}

ObjectState
BillboardObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
BillboardObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
BillboardObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{ // The affine TM transforms from world coords to view coords
    if (vpt->IsActive())
    {
        Matrix3 ntm, ptm, newptm, aTM, coordSysTM;
        // so we need the inverse of this matrix
        ntm = inode->GetNodeTM(t);
        ptm = inode->GetParentTM(t);

        vpt->GetAffineTM(aTM);
        coordSysTM = Inverse(aTM);
        // The Z axis of this matrix is the view direction.
        Point3 viewDir = coordSysTM.GetRow(2);
        Point3 viewDirn = Normalize(viewDir);
        Point3 parentDir = ptm.GetRow(2);
        Point3 parentDirn = Normalize(parentDir);
        float angle = DotProd(parentDirn, viewDirn);
        if (angle > 0.9 || angle < -0.9)
        {
            inode->SetNodeTM(t, ntm);
        }
        else
        {
            int screenAlign = 0;
            pblock->GetValue(PB_BB_SCREEN_ALIGN, t, screenAlign, FOREVER);
            if (screenAlign)
            {

                Point3 viewX = coordSysTM.GetRow(0);
                Point3 viewXn = Normalize(viewX);
                Point3 viewY = coordSysTM.GetRow(1);
                Point3 viewYn = (Normalize(viewY));
                newptm = ntm;
                newptm.SetRow(0, viewXn);
                newptm.SetRow(1, -viewDirn);
                newptm.SetRow(2, viewYn);
                inode->SetNodeTM(t, newptm);
            }
            else
            {
                Point3 newx = CrossProd(parentDirn, viewDirn);
                Point3 newy = CrossProd(parentDirn, newx);
                newptm = ntm;
                newptm.SetRow(0, newx);
                newptm.SetRow(1, newy);
                newptm.SetRow(2, parentDirn);
                inode->SetNodeTM(t, newptm);
            }
        }
    }

    tm = inode->GetObjectTM(t);
}

void
BillboardObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                  Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
}

void
BillboardObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
BillboardObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_BB_SIZE, t, size, FOREVER);
#include "bbob.cpp"
    mesh.buildBoundingBox();
}

int
BillboardObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_BB_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 0.5f, 0.5f, 1.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
BillboardObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class BillboardCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    BillboardObject *billboardObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(BillboardObject *obj) { billboardObject = obj; }
};

int
BillboardCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            billboardObject->pblock->SetValue(PB_BB_SIZE,
                                              billboardObject->iObjParams->GetTime(), radius);
            billboardObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(billboardObject->iObjParams->SnapAngle(ang));
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
static BillboardCreateCallBack BillboardCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
BillboardObject::GetCreateMouseCallBack()
{
    BillboardCreateCB.SetObj(this);
    return (&BillboardCreateCB);
}

RefTargetHandle
BillboardObject::Clone(RemapDir &remap)
{
    BillboardObject *ni = new BillboardObject();
    ni->ReplaceReference(0, pblock->Clone(remap));
    BaseClone(this, ni, remap);
    return ni;
}

void BillboardObject::SetSize(float value, TimeValue time)
{
    DbgAssert(pblock != NULL);
    pblock->SetValue(PB_BB_SIZE, time, value);
    if (pmapParam)
        pmapParam->Invalidate();
}

float BillboardObject::GetSize(TimeValue time, Interval &valid) const
{
    DbgAssert(pblock != NULL);
    float value = 0.0f;
    pblock->GetValue(PB_BB_SIZE, time, value, valid);
    return value;
}
void BillboardObject::SetScreenAlign(int onOff, TimeValue &time)
{
    DbgAssert(pblock != NULL);
    pblock->SetValue(PB_BB_SCREEN_ALIGN, time, onOff);
    if (pmapParam)
        pmapParam->Invalidate();
}
int BillboardObject::GetScreenAlign(TimeValue &time, Interval &valid) const
{
    DbgAssert(pblock != NULL);
    int value = false;
    pblock->GetValue(PB_BB_SCREEN_ALIGN, time, value, valid);
    return value;
}