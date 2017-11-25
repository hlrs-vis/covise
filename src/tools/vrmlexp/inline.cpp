/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: vrml_ins.cpp

    DESCRIPTION:  A VRML Insert helper implementation
 
    CREATED BY: Charles Thaeler
  
    HISTORY: created 6 Mar. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "bookmark.h"
#include "inline.h"

#include "3dsmaxport.h"

// Parameter block indices
#define PB_LENGTH 0

//------------------------------------------------------

class VRMLInsertClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE) { return new VRMLInsObject; }
    const TCHAR *ClassName() { return GetString(IDS_INLINE_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(VRML_INS_CLASS_ID1,
                                         VRML_INS_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static VRMLInsertClassDesc vrmlInsertDesc;

ClassDesc *GetVRMLInsertDesc() { return &vrmlInsertDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ISpinnerControl *VRMLInsObject::sizeSpin = NULL;

HWND VRMLInsObject::hRollup = NULL;

INT_PTR CALLBACK
    VRMLInsRollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    //VRMLInsObject *th = (VRMLInsObject *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
	VRMLInsObject *th = DLGetWindowLongPtr<VRMLInsObject *>(hDlg);
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        th = (VRMLInsObject *)lParam;
        BOOL usingsize = th->GetUseSize();
        //SetWindowLongPtr(hDlg, GWLP_USERDATA, (LONG)th);
		DLSetWindowLongPtr(hDlg, th);
        SetDlgFont(hDlg, th->iObjParams->GetAppHFont());

        th->sizeSpin = GetISpinner(GetDlgItem(hDlg, IDC_INS_SIZE_SPINNER));
        th->sizeSpin->SetLimits(0, 999999, FALSE);
        th->sizeSpin->SetScale(1.0f);
        th->sizeSpin->SetValue(th->GetSize(), FALSE);
        th->sizeSpin->LinkToEdit(GetDlgItem(hDlg, IDC_INS_SIZE), EDITTYPE_POS_FLOAT);
        EnableWindow(GetDlgItem(hDlg, IDC_INS_SIZE), TRUE);
        EnableWindow(GetDlgItem(hDlg, IDC_INS_SIZE_SPINNER), TRUE);

        SendMessage(GetDlgItem(hDlg, IDC_INS_URL), WM_SETTEXT, 0, (LPARAM)(LPCTSTR)th->insURL.data());
        EnableWindow(GetDlgItem(hDlg, IDC_INS_URL), TRUE);

        CheckDlgButton(hDlg, IDC_INS_BBOX_SIZE, usingsize);
        CheckDlgButton(hDlg, IDC_INS_BBOX_DEF, !usingsize);
    }
        return TRUE;

    case WM_DESTROY:
        ReleaseISpinner(th->sizeSpin);
        return FALSE;

    case CC_SPINNER_CHANGE:
        switch (LOWORD(wParam))
        {
        case IDC_INS_SIZE_SPINNER:
            th->SetSize(th->sizeSpin->GetFVal());
            th->iObjParams->RedrawViews(th->iObjParams->GetTime(), REDRAW_INTERACTIVE);
            break;
        }
        return TRUE;

    case CC_SPINNER_BUTTONUP:
        th->iObjParams->RedrawViews(th->iObjParams->GetTime(), REDRAW_END);
        return TRUE;

    case WM_MOUSEACTIVATE:
        th->iObjParams->RealizeParamPanel();
        return FALSE;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MOUSEMOVE:
        th->iObjParams->RollupMouseMessage(hDlg, message, wParam, lParam);
        return FALSE;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_BOOKMARKS:
            // do bookmarks
            if (GetBookmarkURL(th->iObjParams, &th->insURL, NULL, NULL))
            {
                // get the new URL information;
                SendMessage(GetDlgItem(hDlg, IDC_INS_URL), WM_SETTEXT, 0,
                            (LPARAM)(LPCTSTR)th->insURL.data());
            }
            break;
        case IDC_INS_URL:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                int len = (int)SendDlgItemMessage(hDlg, IDC_INS_URL, WM_GETTEXTLENGTH, 0, 0);
                TSTR temp;
                temp.Resize(len + 1);
                SendDlgItemMessage(hDlg, IDC_INS_URL, WM_GETTEXT, len + 1, (LPARAM)temp.data());
                th->insURL = temp;
                break;
            }
            break;
        case IDC_INS_BBOX_SIZE:
        case IDC_INS_BBOX_DEF:
            th->SetUseSize(IsDlgButtonChecked(hDlg, IDC_INS_BBOX_SIZE));
            break;
        }
        return FALSE;

    default:
        return FALSE;
    }
}

void
VRMLInsObject::BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    iObjParams = ip;

    if (!hRollup)
    {
        hRollup = ip->AddRollupPage(
            hInstance,
            MAKEINTRESOURCE(IDD_INS),
            VRMLInsRollupDialogProc,
            GetString(IDS_VRML_INS_TITLE),
            (LPARAM) this);

        ip->RegisterDlgWnd(hRollup);
    }
    else
    {
        //SetWindowLongPtr(hRollup, GWLP_USERDATA, (LONG_PTR) this);
		DLSetWindowLongPtr(hRollup, this);

        // Init the dialog to our values.
    }
}

void
VRMLInsObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (hRollup)
        {
            ip->UnRegisterDlgWnd(hRollup);
            ip->DeleteRollupPage(hRollup);
            hRollup = NULL;
        }
    }
    else
    {
        if (hRollup)
			DLSetWindowLongPtr(hRollup, this);
            //SetWindowLongPtr(hRollup, GWLP_USERDATA, 0);
    }

    iObjParams = NULL;
}

VRMLInsObject::VRMLInsObject()
    : HelperObject()
{
    radius = 0.0f;
    useSize = TRUE;
    // Initialize the object from the dlg versions
}

VRMLInsObject::~VRMLInsObject()
{
}

void
VRMLInsObject::SetSize(float r)
{
    radius = r;
    NotifyDependents(FOREVER, PART_OBJ, REFMSG_CHANGE);
}

IObjParam *VRMLInsObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult VRMLInsObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                          PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult VRMLInsObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                          PartID &partID, RefMessage message)
#endif
{
    switch (message)
    {
    case REFMSG_CHANGE:
        // UpdateUI(iObjParams->GetTime());
        break;

    case REFMSG_GET_PARAM_DIM:
    {
        GetParamDim *gpd = (GetParamDim *)partID;
        switch (gpd->index)
        {
        case 0:
            gpd->dim = stdWorldDim;
            break;
        }
        return REF_HALT;
    }

    case REFMSG_GET_PARAM_NAME:
    {
        GetParamName *gpn = (GetParamName *)partID;
        switch (gpn->index)
        {
        case 0:
            // gpn->name = TSTR(GetResString(IDS_DB_TAPE_LENGTH));
            break;
        }
        return REF_HALT;
    }
    }
    return (REF_SUCCEED);
}

ObjectState
VRMLInsObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
VRMLInsObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    // UpdateUI(time);
    return ivalid;
}

void
VRMLInsObject::GetMat(TimeValue t, INode *inode, ViewExp& vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
#ifdef FIXED_SIZE
	if (!vpt.IsAlive())
	{
		tm.Zero();
		return;
    }
    float scaleFactor = vpt->GetVPWorldWidth(tm.GetTrans()) / (float)360.0;
    if (scaleFactor != 1.0f)
        tm.Scale(Point3(scaleFactor, scaleFactor, scaleFactor));
#endif
}

void
VRMLInsObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt, Box3 &box)
{
	if (!vpt || !vpt->IsAlive())
	{
		// why are we here?
		box.Init();
		return;
		}
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
#ifdef FIXED_SIZE
    float scaleFactor = vpt->GetVPWorldWidth(m.GetTrans()) / (float)360.0;
    box.Scale(scaleFactor);
#endif
}

void
VRMLInsObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt, Box3 &box)
{
	if (!vpt || !vpt->IsAlive())
	{
		// why are we here?
		box.Init();
		return;
	}
    Matrix3 tm;
    BuildMesh(); // 000829  --prs.
    GetMat(t, inode, *vpt, tm);

    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
}

void
VRMLInsObject::MakeQuad(int *f, int a, int b, int c, int d, int vab, int vbc, int vcd, int vda)
{
    mesh.faces[*f].setVerts(a, b, c); // back Face
    mesh.faces[*f].setEdgeVisFlags(vab, vbc, 0);
    mesh.faces[(*f)++].setSmGroup(0);

    mesh.faces[*f].setVerts(c, d, a);
    mesh.faces[*f].setEdgeVisFlags(vcd, vda, 0);
    mesh.faces[(*f)++].setSmGroup(0);
}

void
VRMLInsObject::BuildMesh()
{
    float r = radius,
          r2 = r / 3.0f;

    mesh.setNumVerts(28);
    mesh.setNumFaces(52);

    int v = 0;
    mesh.setVert(v++, Point3(-r2, r2, r2)); //0 -- back of center cube of the plus
    mesh.setVert(v++, Point3(-r2, r2, -r2));
    mesh.setVert(v++, Point3(r2, r2, -r2));
    mesh.setVert(v++, Point3(r2, r2, r2));

    mesh.setVert(v++, Point3(-r2, -r2, r2)); //4 -- front of center cube of the plus
    mesh.setVert(v++, Point3(-r2, -r2, -r2));
    mesh.setVert(v++, Point3(r2, -r2, -r2));
    mesh.setVert(v++, Point3(r2, -r2, r2));

    mesh.setVert(v++, Point3(-r2, -r, r2)); //8 -- front of the plus
    mesh.setVert(v++, Point3(-r2, -r, -r2));
    mesh.setVert(v++, Point3(r2, -r, -r2));
    mesh.setVert(v++, Point3(r2, -r, r2));

    mesh.setVert(v++, Point3(-r, r2, r2)); //12 -- left end
    mesh.setVert(v++, Point3(-r, r2, -r2));
    mesh.setVert(v++, Point3(-r, -r2, -r2));
    mesh.setVert(v++, Point3(-r, -r2, r2));

    mesh.setVert(v++, Point3(r, r2, r2)); //16 -- right end
    mesh.setVert(v++, Point3(r, r2, -r2));
    mesh.setVert(v++, Point3(r, -r2, -r2));
    mesh.setVert(v++, Point3(r, -r2, r2));

    mesh.setVert(v++, Point3(-r2, r2, r)); //20 -- top end
    mesh.setVert(v++, Point3(r2, r2, r));
    mesh.setVert(v++, Point3(r2, -r2, r));
    mesh.setVert(v++, Point3(-r2, -r2, r));

    mesh.setVert(v++, Point3(-r, r, -r)); //24 -- bottom end
    mesh.setVert(v++, Point3(r, r, -r));
    mesh.setVert(v++, Point3(r, -r, -r));
    mesh.setVert(v++, Point3(-r, -r, -r));

    /* Now the Faces */
    int f = 0;
    // TOP
    MakeQuad(&f, 23, 22, 21, 20, 1, 1, 1, 1); // Top
    MakeQuad(&f, 7, 22, 23, 4, 1, 0, 0, 1); // Front
    MakeQuad(&f, 3, 21, 22, 7, 1, 0, 0, 1); // Right
    MakeQuad(&f, 0, 20, 21, 3, 1, 0, 0, 0); // Back
    MakeQuad(&f, 4, 23, 20, 0, 1, 0, 0, 1); // Left

    // FRONT
    MakeQuad(&f, 8, 9, 10, 11, 1, 1, 1, 1); // End
    MakeQuad(&f, 4, 8, 11, 7, 1, 0, 0, 0); // Top
    MakeQuad(&f, 7, 11, 10, 6, 1, 0, 0, 1); // Right
    MakeQuad(&f, 6, 10, 9, 5, 1, 0, 1, 1); // Bottom
    MakeQuad(&f, 5, 9, 8, 4, 1, 0, 0, 1); // Left

    // LEFT
    MakeQuad(&f, 12, 13, 14, 15, 1, 1, 1, 1); // End
    MakeQuad(&f, 0, 12, 15, 4, 1, 0, 0, 0); // Top
    MakeQuad(&f, 4, 15, 14, 5, 1, 0, 0, 0); // Right
    MakeQuad(&f, 5, 14, 13, 1, 1, 0, 0, 1); // Bottom
    MakeQuad(&f, 1, 13, 12, 0, 1, 0, 0, 0); // Left

    // BACK
    MakeQuad(&f, 3, 2, 1, 0, 0, 1, 0, 0); // Left

    // RIGHT
    MakeQuad(&f, 19, 18, 17, 16, 1, 1, 1, 1); // End
    MakeQuad(&f, 7, 19, 16, 3, 1, 0, 0, 0); // Top
    MakeQuad(&f, 3, 16, 17, 2, 1, 0, 0, 0); // Right
    MakeQuad(&f, 2, 17, 18, 6, 1, 0, 0, 1); // Bottom
    MakeQuad(&f, 6, 18, 19, 7, 1, 0, 0, 0); // Left

    // BASE
    MakeQuad(&f, 24, 25, 26, 27, 1, 1, 1, 1); // Bottom
    MakeQuad(&f, 5, 27, 26, 6, 1, 0, 0, 0); // Front
    MakeQuad(&f, 6, 26, 25, 2, 1, 0, 0, 0); // Right
    MakeQuad(&f, 2, 25, 24, 1, 1, 0, 0, 0); // Back
    MakeQuad(&f, 1, 24, 27, 5, 1, 0, 0, 0); // Left

    mesh.InvalidateGeomCache();
    mesh.EnableEdgeList(1);
    mesh.buildBoundingBox();
}

int
VRMLInsObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
	if (!vpt || !vpt->IsAlive())
	{
		// why are we here?
		DbgAssert(!"Doing Display() on invalid view port!");
		return FALSE;
	}
    if (radius <= 0.0)
        return 0;
    BuildMesh();
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();

    DWORD rlim = gw->getRndLimits();
    gw->setRndLimits(GW_WIREFRAME | GW_EDGES_ONLY | GW_BACKCULL | (rlim&GW_Z_BUFFER));
	GetMat(t, inode, *vpt, m);
    gw->setTransform(m);
    if (inode->Selected())
        gw->setColor(LINE_COLOR, 1.0f, 1.0f, 1.0f);
    else if (!inode->IsFrozen())
        gw->setColor(LINE_COLOR, 0.0f, 1.0f, 0.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);
    gw->setRndLimits(rlim);
    return (0);
}

int
VRMLInsObject::HitTest(TimeValue t, INode *inode, int type, int crossing, int flags, IPoint2 *p, ViewExp *vpt)
{
	if (!vpt || !vpt->IsAlive())
	{
		// why are we here?
		DbgAssert(!"Doing HitTest() on invalid view port!");
		return FALSE;
	}
    HitRegion hitRegion;
    DWORD savedLimits;
    int res = FALSE;
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();
    MakeHitRegion(hitRegion, type, crossing, 4, p);
    gw->setRndLimits(((savedLimits = gw->getRndLimits()) | GW_PICK) & ~GW_ILLUM);
    GetMat(t, inode, *vpt, m);
    gw->setTransform(m);
    gw->clearHitCode();
    if (mesh.select(gw, mtl, &hitRegion, flags & HIT_ABORTONHIT))
        return TRUE;
    gw->setRndLimits(savedLimits);
    return res;
}

class VRMLInsCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    VRMLInsObject *vrmlInsObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(VRMLInsObject *obj) { vrmlInsObject = obj; }
};

int
VRMLInsCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m, Matrix3 &mat)
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
            vrmlInsObject->radius = Length(p1 - p0);
            if (vrmlInsObject->sizeSpin)
                vrmlInsObject->sizeSpin->SetValue(vrmlInsObject->radius, FALSE);
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(vrmlInsObject->iObjParams->SnapAngle(ang));
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
static VRMLInsCreateCallBack vrmlInsCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
VRMLInsObject::GetCreateMouseCallBack()
{
    vrmlInsCreateCB.SetObj(this);
    return (&vrmlInsCreateCB);
}

// IO
#define VRML_INS_SIZE_CHUNK 0xacb0
#define VRML_INS_URL_CHUNK 0xacb1
#define VRML_INS_BBOX_CHUNK 0xacb2

IOResult
VRMLInsObject::Save(ISave *isave)
{
    ULONG written;

    isave->BeginChunk(VRML_INS_SIZE_CHUNK);
    isave->Write(&radius, sizeof(float), &written);
    isave->EndChunk();

    isave->BeginChunk(VRML_INS_URL_CHUNK);
#ifdef _UNICODE
    isave->WriteWString(insURL.data());
#else
    isave->WriteCString(insURL.data());
#endif
    isave->EndChunk();

    isave->BeginChunk(VRML_INS_BBOX_CHUNK);
    isave->Write(&useSize, sizeof(int), &written);
    isave->EndChunk();

    return IO_OK;
}

IOResult
VRMLInsObject::Load(ILoad *iload)
{
    ULONG nread;
    IOResult res;

    while (IO_OK == (res = iload->OpenChunk()))
    {
        switch (iload->CurChunkID())
        {
        case VRML_INS_SIZE_CHUNK:
            iload->Read(&radius, sizeof(float), &nread);
            break;
        case VRML_INS_URL_CHUNK:
        {
            TCHAR *n;
#ifdef _UNICODE
            iload->ReadWStringChunk(&n);
#else
            iload->ReadCStringChunk(&n);
#endif
            insURL = n;
            break;
        }
        case VRML_INS_BBOX_CHUNK:
        {
            iload->Read((int *)&useSize, sizeof(int), &nread);
            break;
        }
        }
        iload->CloseChunk();
        if (res != IO_OK)
            return res;
    }
    return IO_OK;
}

RefTargetHandle
VRMLInsObject::Clone(RemapDir &remap)
{
    VRMLInsObject *vi = new VRMLInsObject();
    vi->radius = radius;
    vi->insURL = insURL;
    vi->useSize = useSize;
    BaseClone(this, vi, remap);
    return vi;
}
