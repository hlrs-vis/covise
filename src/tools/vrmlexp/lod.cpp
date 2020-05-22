/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: lod.cpp

DESCRIPTION:  A VRML Level Of Detail helper implementation

CREATED BY: Charles Thaeler

HISTORY: created 29 Feb. 1996

*> Copyright (c) 1996, All Rights Reserved.
**********************************************************************/

#include "vrml.h"
#include "lod.h"

// Parameter block indices
#define PB_LENGTH 0

//------------------------------------------------------

class LODClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE) { return new LODObject; }
    const TCHAR *ClassName() { return GetString(IDS_LOD_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(LOD_CLASS_ID1,
                                         LOD_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static LODClassDesc lodDesc;

ClassDesc *GetLODDesc() { return &lodDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ISpinnerControl *LODObject::sizeSpin = NULL;
ISpinnerControl *LODObject::distSpin = NULL;
ICustButton *LODObject::lodPickButton = NULL;

HWND LODObject::hRollup = NULL;
int LODObject::dlgPrevSel = -1;

class LODObjPick : public PickModeCallback
{
    LODObject *lod;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetLOD(LODObject *l) { lod = l; }
};

static LODObjPick thePick;

BOOL
LODObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    if ((obj->SuperClassID() == HELPER_CLASS_ID && obj->ClassID() == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2)))
        return FALSE;
    return TRUE;
}

void
LODObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_LOD_PICK_MODE));
}

void
LODObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
LODObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
        // Check to see if we have a reference to this object already
        for (int i = 0; i < lod->lodObjects.Count(); i++)
        {
            if (lod->lodObjects[i]->node == node)
                return FALSE; // Can't click those we already have
        }

        LODObj *obj = new LODObj(node);
        int id = lod->lodObjects.Append(1, &obj);

#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = lod->ReplaceReference(id, node);
#else
        RefResult ret = lod->MakeRefByID(FOREVER, id, node);
#endif

        HWND hw = lod->hRollup;
        int ind = (int)SendMessage(GetDlgItem(hw, IDC_LOD_LIST), LB_ADDSTRING, 0, (LPARAM)obj->listStr.data());
        SendMessage(GetDlgItem(hw, IDC_LOD_LIST), LB_SETITEMDATA, (WPARAM)ind, (LPARAM)obj);
        EnableWindow(GetDlgItem(hw, IDC_LOD_DEL), (lod->lodObjects.Count() > 0));
    }
    return FALSE;
}

HCURSOR
LODObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

void
BuildObjectList(LODObject *ob)
{
    if (ob && ob->hRollup)
    {
        int count, i;

        count = (int)SendDlgItemMessage(ob->hRollup, IDC_LOD_LIST, LB_GETCOUNT, 0, 0);

        // First remove any objects on the list
        for (i = count - 1; i >= 0; i--)
            SendDlgItemMessage(ob->hRollup, IDC_LOD_LIST, LB_DELETESTRING, (WPARAM)i, 0);

        for (i = 0; i < ob->lodObjects.Count(); i++)
        {
            LODObj *obj = ob->lodObjects[i];
            obj->ResetStr(); // Make sure we're up to date

            // for now just load the name, we might want to add the frame range as some point
            int ind = (int)SendMessage(GetDlgItem(ob->hRollup, IDC_LOD_LIST), LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(ob->hRollup, IDC_LOD_LIST), LB_SETITEMDATA,
                        (WPARAM)ind, (LPARAM)obj);
        }
    }
}

INT_PTR CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    LODObject *th = (LODObject *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        th = (LODObject *)lParam;
        SetWindowLongPtr(hDlg, GWLP_USERDATA, (LONG_PTR)th);
        SetDlgFont(hDlg, th->iObjParams->GetAppHFont());

        th->sizeSpin = GetISpinner(GetDlgItem(hDlg, IDC_LOD_SIZE_SPINNER));
        th->sizeSpin->SetLimits(0, 999999, FALSE);
        th->sizeSpin->SetScale(1.0f);
        th->sizeSpin->SetValue(th->GetSize(), FALSE);
        th->sizeSpin->LinkToEdit(GetDlgItem(hDlg, IDC_LOD_SIZE), EDITTYPE_POS_FLOAT);
        EnableWindow(GetDlgItem(hDlg, IDC_LOD_SIZE), TRUE);
        EnableWindow(GetDlgItem(hDlg, IDC_LOD_SIZE_SPINNER), TRUE);

        th->distSpin = GetISpinner(GetDlgItem(hDlg, IDC_LOD_DIST_SPINNER));
        th->distSpin->SetLimits(0, 999999, FALSE);
        th->distSpin->SetScale(1.0f);
        th->distSpin->SetValue(0, FALSE);
        th->distSpin->LinkToEdit(GetDlgItem(hDlg, IDC_LOD_DIST), EDITTYPE_POS_FLOAT);
        // Disable till there is a selected object on the list
        EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST), FALSE);
        EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST_SPINNER), FALSE);

        th->lodPickButton = GetICustButton(GetDlgItem(hDlg, IDC_LOD_PICK));
        th->lodPickButton->SetType(CBT_CHECK);
        th->lodPickButton->SetButtonDownNotify(TRUE);
        th->lodPickButton->SetHighlightColor(GREEN_WASH);

        // Now we need to fill in the list box IDC_LOD_LIST
        th->hRollup = hDlg;
        BuildObjectList(th);

        EnableWindow(GetDlgItem(hDlg, IDC_LOD_DEL), (th->lodObjects.Count() > 0));
        th->dlgPrevSel = -1;
    }
        return TRUE;

    case WM_DESTROY:
        th->iObjParams->ClearPickMode();
        th->previousMode = NULL;
        ReleaseISpinner(th->sizeSpin);
        ReleaseISpinner(th->distSpin);
        ReleaseICustButton(th->lodPickButton);
        return FALSE;

    case CC_SPINNER_CHANGE:
        switch (LOWORD(wParam))
        {
        case IDC_LOD_SIZE_SPINNER:
            th->SetSize(th->sizeSpin->GetFVal());
            th->iObjParams->RedrawViews(th->iObjParams->GetTime(), REDRAW_INTERACTIVE);
            break;
        case IDC_LOD_DIST_SPINNER:
            th->SetCurDist(th->distSpin->GetFVal());
            th->iObjParams->RedrawViews(th->iObjParams->GetTime(), REDRAW_INTERACTIVE);
            int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_GETCURSEL, 0, 0);
            LODObj *obj = (LODObj *)SendDlgItemMessage(hDlg,
                                                       IDC_LOD_LIST, LB_GETITEMDATA, sel, 0);
            obj->dist = th->dlgCurSelDist;
            obj->ResetStr();
            SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_DELETESTRING,
                        sel, 0);
            int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_ADDSTRING, 0,
                                       (LPARAM)obj->listStr.data());
            SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_SETITEMDATA,
                        (WPARAM)ind, (LPARAM)obj);
            SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_SETCURSEL, (WPARAM)ind, 0);
            break;
        }
        return TRUE;
    case CC_SPINNER_BUTTONDOWN:
        th->iObjParams->RedrawViews(th->iObjParams->GetTime(), REDRAW_BEGIN);
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
        case IDC_LOD_PICK: // Pick an object from the scene
            // Set the pickmode...
            switch (HIWORD(wParam))
            {
            case BN_BUTTONDOWN:
                if (th->previousMode)
                {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                }
                else
                {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    thePick.SetLOD(th);
                    th->iObjParams->SetPickMode(&thePick);
                }
                break;
            }
            break;
        case IDC_LOD_DEL:
        { // Delete the object from the list
            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_GETCURSEL, 0, 0);
            if (index != LB_ERR)
            {
                LODObj *obj = (LODObj *)SendDlgItemMessage(hDlg,
                                                           IDC_LOD_LIST, LB_GETITEMDATA, index, 0);
                for (int i = 0; i < th->lodObjects.Count(); i++)
                {
                    if (obj == th->lodObjects[i])
                    {
                        // remove the item from the list
                        SendDlgItemMessage(hDlg, IDC_LOD_LIST, LB_DELETESTRING,
                                           (WPARAM)index, 0);
                        th->dlgPrevSel = -1;
                        // Remove the reference to obj->node
                        th->DeleteReference(th->FindRef((RefTargetHandle)obj->node));
                        // remove the object from the table
                        th->lodObjects.Delete(i, 1);
                        break;
                    }
                }
                EnableWindow(GetDlgItem(hDlg, IDC_LOD_DEL), (th->lodObjects.Count() > 0));
                if (th->lodObjects.Count() <= 0)
                {
                    th->SetCurDist(-1.0f);
                    th->iObjParams->RedrawViews(th->iObjParams->GetTime());
                }
            }
        }
        break;
        case IDC_LOD_LIST:
            switch (HIWORD(wParam))
            {
            case LBN_SELCHANGE:
            {
                int sel = (int)SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_GETCURSEL, 0, 0);
                if (th->dlgPrevSel != -1)
                {
                    // save any editing
                    LODObj *obj = (LODObj *)SendDlgItemMessage(hDlg,
                                                               IDC_LOD_LIST, LB_GETITEMDATA, th->dlgPrevSel, 0);
                    obj->dist = th->distSpin->GetFVal();
                    obj->ResetStr();
                    SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_DELETESTRING,
                                th->dlgPrevSel, 0);
                    int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_ADDSTRING, 0,
                                               (LPARAM)obj->listStr.data());
                    SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_SETITEMDATA,
                                (WPARAM)ind, (LPARAM)obj);
                    SendMessage(GetDlgItem(hDlg, IDC_LOD_LIST), LB_SETCURSEL, sel, 0);
                }
                th->dlgPrevSel = sel;
                if (sel >= 0)
                {
                    LODObj *obj = (LODObj *)SendDlgItemMessage(hDlg,
                                                               IDC_LOD_LIST, LB_GETITEMDATA, sel, 0);
                    assert(obj);

                    th->distSpin->SetValue(obj->dist, TRUE);
                    th->SetCurDist(obj->dist);
                    EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST), TRUE);
                    EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST_SPINNER), TRUE);
                }
                else
                {
                    EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST), FALSE);
                    EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST_SPINNER), FALSE);
                    th->SetCurDist(-1.0f);
                }
                th->iObjParams->RedrawViews(th->iObjParams->GetTime());
            }
            break;
            case LBN_SELCANCEL:
                EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST), FALSE);
                EnableWindow(GetDlgItem(hDlg, IDC_LOD_DIST_SPINNER), FALSE);
                break;
            }
            break;
        }
        return FALSE;

    default:
        return FALSE;
    }
}

void
LODObject::BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    iObjParams = ip;

    if (!hRollup)
    {
        hRollup = ip->AddRollupPage(
            hInstance,
            MAKEINTRESOURCE(IDD_LOD),
            RollupDialogProc,
            GetString(IDS_LOD_TITLE),
            (LPARAM) this);

        ip->RegisterDlgWnd(hRollup);
    }
    else
    {
        SetWindowLongPtr(hRollup, GWLP_USERDATA, (LONG_PTR) this);
    }
    dlgCurSelDist = -1.0f; // Start with no visible distance sphere
}

void
LODObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
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
            SetWindowLongPtr(hRollup, GWLP_USERDATA, 0);
    }

    iObjParams = NULL;
    dlgCurSelDist = -1.0f; // End with no visible distance sphere
}

LODObject::LODObject()
    : HelperObject()
{
    dlgCurSelDist = -1.0f;
    previousMode = NULL;
    BuildObjectList(this);
}

LODObject::~LODObject()
{
    DeleteAllRefsFromMe();
    for (int i = 0; i < lodObjects.Count(); i++)
    {
        LODObj *obj = lodObjects[0];
        lodObjects.Delete(0, 1);
        delete obj;
    }
}

void
LODObject::SetSize(float r)
{
    radius = r;
    NotifyDependents(FOREVER, PART_OBJ, REFMSG_CHANGE);
}

void
LODObject::SetCurDist(float d)
{
    dlgCurSelDist = d;
    NotifyDependents(FOREVER, PART_OBJ, REFMSG_CHANGE);
}

IObjParam *LODObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult LODObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                      PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult LODObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                      PartID &partID, RefMessage message)
#endif
{
    int i;
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < lodObjects.Count(); i++)
        {
            if (lodObjects[i]->node == hTarget)
            {
                // Do I need to remove the reference? FIXME
                lodObjects.Delete(i, 1);
            }
        }
        break;
    case REFMSG_NODE_NAMECHANGE:
        // Find the ID on the list and call ResetStr
        for (i = 0; i < lodObjects.Count(); i++)
        {
            if (lodObjects[i]->node == hTarget)
            {
                // Found it
                lodObjects[i]->ResetStr();
                break;
            }
        }
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
LODObject::GetReference(int ind)
{
    if (ind >= lodObjects.Count())
        return NULL;

    return lodObjects[ind]->node;
}

void
LODObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind >= lodObjects.Count())
        return;

    lodObjects[ind]->node = (INode *)rtarg;
    lodObjects[ind]->ResetStr();
}

ObjectState
LODObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
LODObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
LODObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
LODObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt, Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
    if (dlgCurSelDist >= 0.0f)
    {
        Point3 x[32], y[32], z[32];
        GetDistPoints(dlgCurSelDist, x, y, z);
        for (int i = 0; i < 32; i++)
        {
            box += x[i];
            box += y[i];
            box += z[i];
        }
    }
}

void
LODObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt, Box3 &box)
{
    Matrix3 tm;
    BuildMesh(); // 000829  --prs.
    mesh.buildBoundingBox();

    GetMat(t, inode, vpt, tm);

    int nv = mesh.getNumVerts();
    box.Init();
    for (int i = 0; i < nv; i++)
        box += tm * mesh.getVert(i);
    if (dlgCurSelDist >= 0.0f)
    {
        Point3 x[32], y[32], z[32];
        GetDistPoints(dlgCurSelDist, x, y, z);
        for (int i = 0; i < 32; i++)
        {
            box += tm * x[i];
            box += tm * y[i];
            box += tm * z[i];
        }
    }
}

void
LODObject::BuildMesh()
{
    float size = radius;
#include "lodob.cpp"
}

void
LODObject::GetDistPoints(float radius, Point3 *x, Point3 *y, Point3 *z)
{
    float dang = PI / (2.0f * 8.0f);
    float ang = 0.0f;
    for (int i = 0; i < 32; i++)
    {
        z[i].x = x[i].y = y[i].x = radius * (float)cos(ang);
        z[i].y = x[i].z = y[i].z = radius * (float)sin(ang);
        z[i].z = x[i].x = y[i].y = 0.0f;
        ang += dang;
    }
}

void
LODObject::DrawDistSphere(TimeValue t, INode *inode, GraphicsWindow *gw)
{
    Matrix3 tm = inode->GetObjectTM(t);
    gw->setTransform(tm);

    Point3 x[33], y[33], z[33];
    GetDistPoints(dlgCurSelDist, x, y, z);

    if (!inode->IsFrozen())
        gw->setColor(LINE_COLOR, 0.0f, 0.0f, 1.0f);
    gw->polyline(32, x, NULL, NULL, TRUE, NULL);
    gw->polyline(32, y, NULL, NULL, TRUE, NULL);
    gw->polyline(32, z, NULL, NULL, TRUE, NULL);
}

int
LODObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    if (radius <= 0.0)
        return 0;
    BuildMesh();
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
        gw->setColor(LINE_COLOR, 0.0f, 1.0f, 1.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);
    if (inode->Selected() && dlgCurSelDist >= 0.0f)
        DrawDistSphere(t, inode, gw);

    gw->setRndLimits(rlim);
    return (0);
}

int
LODObject::HitTest(TimeValue t, INode *inode, int type, int crossing, int flags, IPoint2 *p, ViewExp *vpt)
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

class LODCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    LODObject *lodObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(LODObject *obj) { lodObject = obj; }
};

int
LODCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m, Matrix3 &mat)
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
            lodObject->radius = Length(p1 - p0);
            if (lodObject->sizeSpin)
                lodObject->sizeSpin->SetValue(lodObject->radius, FALSE);
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(lodObject->iObjParams->SnapAngle(ang));
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
static LODCreateCallBack lodCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
LODObject::GetCreateMouseCallBack()
{
    lodCreateCB.SetObj(this);
    return (&lodCreateCB);
}

// IO
#define LOD_SIZE_CHUNK 0xaca0
#define LOD_OBJ_DIST_CHUNK 0xaca1

IOResult
LODObject::Save(ISave *isave)
{
    ULONG written;

    isave->BeginChunk(LOD_SIZE_CHUNK);
    isave->Write(&radius, sizeof(float), &written);
    isave->EndChunk();

    int c = lodObjects.Count();
    if (c > 0)
    {
        for (int i = 0; i < c; i++)
        {
            float dist = lodObjects[i]->dist;
            written = 0;
            isave->BeginChunk(LOD_OBJ_DIST_CHUNK);
            isave->Write(&dist, sizeof(float), &written);
            isave->EndChunk();
        }
    }
    return IO_OK;
}

IOResult
LODObject::Load(ILoad *iload)
{
    ULONG nread;
    IOResult res;
    float dist;
    LODObj *obj;

    while (IO_OK == (res = iload->OpenChunk()))
    {
        switch (iload->CurChunkID())
        {
        case LOD_SIZE_CHUNK:
            iload->Read(&radius, sizeof(float), &nread);
            break;
        case LOD_OBJ_DIST_CHUNK:
            iload->Read(&dist, sizeof(float), &nread);
            obj = new LODObj(NULL, dist);
            lodObjects.Append(1, &obj);
            break;
        }
        iload->CloseChunk();
        if (res != IO_OK)
            return res;
    }
    return IO_OK;
}

RefTargetHandle
LODObject::Clone(RemapDir &remap)
{
    LODObject *ts = new LODObject();
    ts->lodObjects.SetCount(lodObjects.Count());
    ts->radius = radius;
    for (int i = 0; i < lodObjects.Count(); i++)
    {
        ts->lodObjects[i] = new LODObj;
        ts->lodObjects[i]->dist = lodObjects[i]->dist;
        if (remap.FindMapping(lodObjects[i]->node))
            ts->ReplaceReference(i, remap.FindMapping(lodObjects[i]->node));
        else
            ts->ReplaceReference(i, lodObjects[i]->node);
    }
    BaseClone(this, ts, remap);
    return ts;
}
