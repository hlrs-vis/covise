/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: cooout.cpp

	DESCRIPTION:  A utility that outputs an object in C++ code

	CREATED BY: Scott Morrison

	HISTORY: created Spetember 9 1996

 *>	Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
//#ifdef _DEBUG

extern TCHAR *GetString(int id);

#define CPP_OUT_CLASS_ID 0x8f7ce9ea

class CppOut : public UtilityObj
{
public:
    IUtil *iu;
    Interface *ip;
    HWND hPanel;
    ICustButton *iPick;

    CppOut();
    void BeginEditParams(Interface *ip, IUtil *iu);
    void EndEditParams(Interface *ip, IUtil *iu);
    void DeleteThis() {}

    void Init(HWND hWnd);
    void Destroy(HWND hWnd);

    void OutputObject(INode *node, TCHAR *fname);
};
static CppOut theCppOut;

class CppOutClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE) { return &theCppOut; }
    const TCHAR *ClassName() { return GetString(IDS_RB_CPPOBJECTOUT); }
    SClass_ID SuperClassID() { return UTILITY_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(CPP_OUT_CLASS_ID, 0); }
    const TCHAR *Category() { return _T(""); }
};

static CppOutClassDesc cppOutDesc;
ClassDesc *GetCppOutDesc() { return &cppOutDesc; }

class CppOutPickNodeCallback : public PickNodeCallback
{
public:
    BOOL Filter(INode *node);
};

BOOL
CppOutPickNodeCallback::Filter(INode *node)
{
    ObjectState os = node->EvalWorldState(theCppOut.ip->GetTime());
    if (os.obj->SuperClassID() == GEOMOBJECT_CLASS_ID && os.obj->IsRenderable())
        return TRUE;
    else
        return FALSE;
}

static CppOutPickNodeCallback thePickFilt;

class CppOutPickModeCallback : public PickModeCallback
{
public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip) { theCppOut.iPick->SetCheck(TRUE); }
    void ExitMode(IObjParam *ip) { theCppOut.iPick->SetCheck(FALSE); }

    PickNodeCallback *GetFilter() { return &thePickFilt; }
    BOOL RightClick(IObjParam *ip, ViewExp *vpt) { return TRUE; }
};

static CppOutPickModeCallback thePickMode;

BOOL CppOutPickModeCallback::HitTest(
    IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags)
{
    return ip->PickNode(hWnd, m, &thePickFilt) ? TRUE : FALSE;
}

BOOL
CppOutPickModeCallback::Pick(IObjParam *ip, ViewExp *vpt)
{
    INode *node = vpt->GetClosestHit();
    if (node)
    {
        static TCHAR fname[256] = { '\0' };
        OPENFILENAME ofn;
        memset(&ofn, 0, sizeof(ofn));
        FilterList fl;
        fl.Append(GetString(IDS_RB_CPPFILES));
        fl.Append(_T("*.cpp"));
        TSTR title = GetString(IDS_RB_SAVEOBJECT);

        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = theCppOut.hPanel;
        ofn.lpstrFilter = fl;
        ofn.lpstrFile = fname;
        ofn.nMaxFile = 256;
        ofn.lpstrInitialDir = ip->GetDir(APP_EXPORT_DIR);
        ofn.Flags = OFN_HIDEREADONLY | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
        ofn.lpstrDefExt = _T("asc");
        ofn.lpstrTitle = title;

    tryAgain:
        if (GetSaveFileName(&ofn))
        {
            if (DoesFileExist(fname))
            {
                TSTR buf1;
                TSTR buf2 = GetString(IDS_RB_SAVEOBJECT);
                buf1.printf(GetString(IDS_RB_FILEEXISTS), fname);
                if (IDYES != MessageBox(
                                 theCppOut.hPanel,
                                 buf1, buf2, MB_YESNO | MB_ICONQUESTION))
                {
                    goto tryAgain;
                }
            }
            theCppOut.OutputObject(node, fname);
        }
    }
    return TRUE;
}

static INT_PTR CALLBACK
    CppOutDlgProc(
        HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_INITDIALOG:
        theCppOut.Init(hWnd);
        break;

    case WM_DESTROY:
        theCppOut.Destroy(hWnd);
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDOK:
            theCppOut.iu->CloseUtility();
            break;

        case IDC_CPPOUT_PICK:
            theCppOut.ip->SetPickMode(&thePickMode);
            break;
        }
        break;

    default:
        return FALSE;
    }
    return TRUE;
}

CppOut::CppOut()
{
    iu = NULL;
    ip = NULL;
    hPanel = NULL;
    iPick = NULL;
}

extern HINSTANCE hInstance;

void
CppOut::BeginEditParams(Interface *ip, IUtil *iu)
{
    this->iu = iu;
    this->ip = ip;
    hPanel = ip->AddRollupPage(
        hInstance,
        MAKEINTRESOURCE(IDD_CPPOUT_PANEL),
        CppOutDlgProc,
        GetString(IDS_RB_CPPOBJECTOUT),
        0);
}

void
CppOut::EndEditParams(Interface *ip, IUtil *iu)
{
    ip->ClearPickMode();
    this->iu = NULL;
    this->ip = NULL;
    if (hPanel)
        ip->DeleteRollupPage(hPanel);
    hPanel = NULL;
}

void
CppOut::Init(HWND hWnd)
{
    iPick = GetICustButton(GetDlgItem(hWnd, IDC_CPPOUT_PICK));
    iPick->SetType(CBT_CHECK);
    iPick->SetHighlightColor(GREEN_WASH);
}

void
CppOut::Destroy(HWND hWnd)
{
    ReleaseICustButton(iPick);
    iPick = NULL;
}

class NullView : public View
{
public:
    Point2 ViewToScreen(Point3 p) { return Point2(p.x, p.y); }
    NullView()
    {
        worldToView.IdentityMatrix();
        screenW = 640.0f;
        screenH = 480.0f;
    }
};

void
CppOut::OutputObject(INode *node, TCHAR *fname)
{
    ObjectState os = node->EvalWorldState(theCppOut.ip->GetTime());
    assert(os.obj->SuperClassID() == GEOMOBJECT_CLASS_ID);
    BOOL needDel;
    NullView nullView;
    Mesh *mesh = ((GeomObject *)os.obj)->GetRenderMesh(ip->GetTime(), node, nullView, needDel);
    if (!mesh)
        return;

    FILE *file = _tfopen(fname, _T("wt"));

    float maxLen = 0.0f, len;
    int i;
    Matrix3 tm = node->GetObjTMAfterWSM(theCppOut.ip->GetTime());
    AffineParts parts;
    decomp_affine(tm, &parts);

    if (file)
    {
        Box3 bb = mesh->getBoundingBox() * tm;
        Point3 center = bb.Center();
        for (i = 0; i < mesh->getNumVerts(); i++)
        {
            Point3 v = tm * mesh->verts[i] - center;
            len = Length(v);
            if (v[0] > maxLen)
                maxLen = v[0];
            if (v[1] > maxLen)
                maxLen = v[1];
            if (v[2] > maxLen)
                maxLen = v[2];
        }
        maxLen *= 2.0;
        fprintf(file, "    mesh.setNumVerts(%d);\n", mesh->getNumVerts());
        fprintf(file, "    mesh.setNumFaces(%d);\n", mesh->getNumFaces());
        for (i = 0; i < mesh->getNumVerts(); i++)
        {
            //Point3 v = (tm  * mesh->verts[i] - center) / maxLen;
            Point3 v = (tm * mesh->verts[i]);
            fprintf(file, "    mesh.setVert(%d, size * Point3(%f,%f,%f));\n",
                    i, v.x, v.y, v.z);
        }

        for (i = 0; i < mesh->getNumFaces(); i++)
        {
            if (parts.f < 0.0f)
                fprintf(file, "    mesh.faces[%d].setVerts(%d,%d,%d);\n",
                        i, mesh->faces[i].v[1], mesh->faces[i].v[0],
                        mesh->faces[i].v[2]);
            else
                fprintf(file, "    mesh.faces[%d].setVerts(%d,%d,%d);\n",
                        i, mesh->faces[i].v[0], mesh->faces[i].v[1],
                        mesh->faces[i].v[2]);
            if (parts.f < 0.0f)
                fprintf(file, "    mesh.faces[%d].setEdgeVisFlags(%d,%d,%d);\n",
                        i,
                        mesh->faces[i].getEdgeVis(0) ? 1 : 0,
                        mesh->faces[i].getEdgeVis(2) ? 1 : 0,
                        mesh->faces[i].getEdgeVis(1) ? 1 : 0);
            else
                fprintf(file, "    mesh.faces[%d].setEdgeVisFlags(%d,%d,%d);\n",
                        i,
                        mesh->faces[i].getEdgeVis(0) ? 1 : 0,
                        mesh->faces[i].getEdgeVis(1) ? 1 : 0,
                        mesh->faces[i].getEdgeVis(2) ? 1 : 0);
            fprintf(file, "    mesh.faces[%d].setSmGroup(%x);\n",
                    i, mesh->faces[i].smGroup);
        }

        fclose(file);
    }
    if (needDel)
        delete mesh;
}

//#endif
