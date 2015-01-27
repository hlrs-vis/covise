/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: audio.cpp

    DESCRIPTION:  A VRML 2.0 AutoClip helper
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 29 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "audio.h"

//------------------------------------------------------

class AudioClipClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new AudioClipObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_AUDIO_CLIP_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(AudioClip_CLASS_ID1,
                                         AudioClip_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static AudioClipClassDesc AudioClipDesc;

ClassDesc *GetAudioClipDesc() { return &AudioClipDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

BOOL CALLBACK
    AudioClipDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     AudioClipObject *th)
{
    TCHAR text[MAX_PATH];

    switch (message)
    {
    case WM_INITDIALOG:
    {
        Edit_SetText(GetDlgItem(hDlg, IDC_URL), th->url.data());
        Edit_SetText(GetDlgItem(hDlg, IDC_DESC), th->desc.data());
        return FALSE;
    }

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_URL:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_URL), text, MAX_PATH);
                th->url = text;
                break;
            }
            break;
        case IDC_DESC:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                Edit_GetText(GetDlgItem(hDlg, IDC_DESC), text, MAX_PATH);
                th->desc = text;
                break;
            }
            break;
        }
        break;
    default:
        return FALSE;
    }

    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Loop
    ParamUIDesc(PB_AC_LOOP,
                TYPE_SINGLECHEKBOX, IDC_LOOP),

    // Loop
    ParamUIDesc(PB_AC_START,
                TYPE_SINGLECHEKBOX, IDC_START_ON_LOAD),

    // Pitch
    ParamUIDesc(
        PB_AC_PITCH,
        EDITTYPE_FLOAT,
        IDC_PITCH_EDIT, IDC_PITCH_SPIN,
        0.0f, 1.0f,
        0.1f),

    // Pitch
    ParamUIDesc(
        PB_AC_SIZE,
        EDITTYPE_FLOAT,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 10000.0f,
        0.1f),

};

#define PARAMDESC_LENGTH 4

static ParamBlockDescID descVer0[] = {
    { TYPE_INT, NULL, FALSE, 0 },
    { TYPE_FLOAT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_INT, NULL, FALSE, 3 },
};

#define CURRENT_VERSION 0

class AudioClipParamDlgProc : public ParamMapUserDlgProc
{
public:
    AudioClipObject *ob;

    AudioClipParamDlgProc(AudioClipObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR AudioClipParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                       UINT msg, WPARAM wParam, LPARAM lParam)
{
    return AudioClipDlgProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *AudioClipObject::pmapParam = NULL;

void
AudioClipObject::BeginEditParams(IObjParam *ip, ULONG flags,
                                 Animatable *prev)
{
    iObjParams = ip;

    if (pmapParam)
    {

        // Left over from last AudioClip created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_AUDIO_CLIP),
                                     _T("Audio Clip" /*JP_LOC*/),
                                     0);
    }
    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new AudioClipParamDlgProc(this));
    }
}

void
AudioClipObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

AudioClipObject::AudioClipObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_AC_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_AC_LOOP, 0, FALSE);
    pb->SetValue(PB_AC_PITCH, 0, 1.0f);
    pb->SetValue(PB_AC_START, 0, FALSE);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);

    written = 0;
}

AudioClipObject::~AudioClipObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *AudioClipObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult AudioClipObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                            PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult AudioClipObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                            PartID &partID, RefMessage message)
#endif
{
    //     int i;
    //     switch (message) {
    //     }
    return REF_SUCCEED;
}

RefTargetHandle
AudioClipObject::GetReference(int ind)
{
    if (ind == 0)
        return (RefTargetHandle)pblock;
    return NULL;
}

void
AudioClipObject::SetReference(int ind, RefTargetHandle rtarg)
{
    pblock = (IParamBlock *)rtarg;
}

ObjectState
AudioClipObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
AudioClipObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
AudioClipObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
AudioClipObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                                  Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    box = mesh.getBoundingBox();
}

void
AudioClipObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
AudioClipObject::BuildMesh(TimeValue t)
{
    float size;
    pblock->GetValue(PB_AC_SIZE, t, size, FOREVER);
#include "acob.cpp"

    mesh.buildBoundingBox();
}

int
AudioClipObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_AC_SIZE, t, radius, FOREVER);
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
        gw->setColor(LINE_COLOR, 1.0f, 0.0f, 1.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);

    gw->setRndLimits(rlim);
    return (0);
}

int
AudioClipObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class AudioClipCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    AudioClipObject *audioClipObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(AudioClipObject *obj) { audioClipObject = obj; }
};

int
AudioClipCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            audioClipObject->pblock->SetValue(PB_AC_SIZE,
                                              audioClipObject->iObjParams->GetTime(), radius);
            audioClipObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(audioClipObject->iObjParams->SnapAngle(ang));
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
static AudioClipCreateCallBack AudioClipCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
AudioClipObject::GetCreateMouseCallBack()
{
    AudioClipCreateCB.SetObj(this);
    return (&AudioClipCreateCB);
}

#define DESC_CHUNK 0xad30
#define URL_CHUNK 0xad31
#define TOP_CHUNK 0xad05

IOResult
AudioClipObject::Save(ISave *isave)
{
    isave->BeginChunk(DESC_CHUNK);
    isave->WriteCString(desc.data());
    isave->EndChunk();

    isave->BeginChunk(URL_CHUNK);
    isave->WriteCString(url.data());
    isave->EndChunk();

    return IO_OK;
}

IOResult
AudioClipObject::Load(ILoad *iload)
{
    TCHAR *txt;

    while (iload->OpenChunk() == IO_OK)
    {
        switch (iload->CurChunkID())
        {
        case DESC_CHUNK:
            iload->ReadCStringChunk(&txt);
            desc = txt;
            break;

        case URL_CHUNK:
            iload->ReadCStringChunk(&txt);
            url = txt;
            break;

        default:
            break;
        }
        iload->CloseChunk();
    }
    return IO_OK;
}

RefTargetHandle
AudioClipObject::Clone(RemapDir &remap)
{
    AudioClipObject *ac = new AudioClipObject();
    ac->ReplaceReference(0, pblock->Clone(remap));
    ac->desc = desc;
    ac->url = url;
    BaseClone(this, ac, remap);
    return ac;
}
