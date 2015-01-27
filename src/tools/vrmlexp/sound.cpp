/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: sound.cpp

    DESCRIPTION:  A VRML 2.0 Sound Helper object
 
    CREATED BY: Scott Morrison
  
    HISTORY: created 29 Aug, 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "sound.h"
#include "audio.h"

#define SEGMENTS 32

ISpinnerControl *SoundObject::minBackSpin;
ISpinnerControl *SoundObject::maxBackSpin;
ISpinnerControl *SoundObject::minFrontSpin;
ISpinnerControl *SoundObject::maxFrontSpin;

//------------------------------------------------------

class SoundClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new SoundObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_SOUND_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(Sound_CLASS_ID1,
                                         Sound_CLASS_ID2); }
    const TCHAR *Category() { return _T("VRML97"); }
};

static SoundClassDesc SoundDesc;

ClassDesc *GetSoundDesc() { return &SoundDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *SoundObject::SoundPickButton = NULL;

HWND SoundObject::hRollup = NULL;
int SoundObject::dlgPrevSel = -1;

class SoundObjPick : public PickModeCallback
{
    SoundObject *sound;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetSound(SoundObject *l) { sound = l; }
};

//static SoundObjPick thePick;
static SoundObjPick thePick;
static BOOL pickMode = FALSE;
static CommandMode *lastMode = NULL;

static void
SetPickMode(SoundObject *o)
{
    if (pickMode)
    {
        pickMode = FALSE;
        GetCOREInterface()->PushCommandMode(lastMode);
        lastMode = NULL;
        GetCOREInterface()->ClearPickMode();
    }
    else
    {
        pickMode = TRUE;
        lastMode = GetCOREInterface()->GetCommandMode();
        thePick.SetSound(o);
        GetCOREInterface()->SetPickMode(&thePick);
    }
}

BOOL
SoundObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                      int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    return obj->ClassID() == AudioClipClassID;
}

void
SoundObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_AUDIOCLIP));
}

void
SoundObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
SoundObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = sound->ReplaceReference(1, node);
#else
        RefResult ret = sound->MakeRefByID(FOREVER, 1, node);
#endif

        SetPickMode(NULL);
        // sound->iObjParams->SetCommandMode(sound->previousMode);
        // sound->previousMode = NULL;
        sound->SoundPickButton->SetCheck(FALSE);
        HWND hw = sound->hRollup;
        Static_SetText(GetDlgItem(hw, IDC_NAME), sound->audioClip->GetName());
        return FALSE;
    }
    return FALSE;
}

HCURSOR
SoundObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

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
                     SoundObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        float minF, minB, maxF, maxB;
        TimeValue t = 0;

        th->SoundPickButton = GetICustButton(GetDlgItem(hDlg,
                                                        IDC_AUDIO_CLIP_PICK));
        th->SoundPickButton->SetType(CBT_CHECK);
        th->SoundPickButton->SetButtonDownNotify(TRUE);
        th->SoundPickButton->SetHighlightColor(GREEN_WASH);
        th->SoundPickButton->SetCheck(FALSE);

        th->pblock->GetValue(PB_SND_MIN_FRONT, t, minF, FOREVER);
        th->pblock->GetValue(PB_SND_MIN_BACK, t, minB, FOREVER);
        th->pblock->GetValue(PB_SND_MAX_FRONT, t, maxF, FOREVER);
        th->pblock->GetValue(PB_SND_MAX_BACK, t, maxB, FOREVER);
        /*
        th->minBackSpin->SetValue(minB, FALSE);
        th->maxBackSpin->SetValue(maxB, FALSE);
        th->minFrontSpin->SetValue(minF, FALSE);
        th->maxFrontSpin->SetValue(maxF, FALSE);
*/

        // Now we need to fill in the list box IDC_Sound_LIST
        th->hRollup = hDlg;

        if (th->audioClip)
            Static_SetText(GetDlgItem(hDlg, IDC_NAME),
                           th->audioClip->GetName());

        if (pickMode)
            SetPickMode(th);

        return TRUE;
    }
    case WM_DESTROY:
        if (pickMode)
            SetPickMode(th);
        // th->iObjParams->ClearPickMode();
        // th->previousMode = NULL;
        RELEASE_SPIN(minBackSpin);
        RELEASE_SPIN(maxBackSpin);
        RELEASE_SPIN(minFrontSpin);
        RELEASE_SPIN(maxFrontSpin);

        RELEASE_BUT(SoundPickButton);
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
                SetPickMode(th);
                /*
                if (th->previousMode) {
                    // reset the command mode
                    th->iObjParams->SetCommandMode(th->previousMode);
                    th->previousMode = NULL;
                } else {
                    th->previousMode = th->iObjParams->GetCommandMode();
                    thePick.SetSound(th);
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
    // Intensity
    ParamUIDesc(
        PB_SND_INTENSITY,
        EDITTYPE_UNIVERSE,
        IDC_INTENSITY_EDIT, IDC_INTENSITY_SPIN,
        0.0f, 1.0f,
        0.1f),

    // Priority
    ParamUIDesc(
        PB_SND_PRIORITY,
        EDITTYPE_UNIVERSE,
        IDC_PRIORITY_EDIT, IDC_PRIORITY_SPIN,
        0.0f, 1.0f,
        0.1f),

    // Spatialize
    ParamUIDesc(PB_SND_SPATIALIZE, TYPE_SINGLECHEKBOX, IDC_SPATIALIZE),

    // Min Back
    ParamUIDesc(
        PB_SND_MIN_BACK,
        EDITTYPE_UNIVERSE,
        IDC_MIN_BACK_EDIT, IDC_MIN_BACK_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Max Back
    ParamUIDesc(
        PB_SND_MAX_BACK,
        EDITTYPE_UNIVERSE,
        IDC_MAX_BACK_EDIT, IDC_MAX_BACK_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Min Front
    ParamUIDesc(
        PB_SND_MIN_FRONT,
        EDITTYPE_UNIVERSE,
        IDC_MIN_FRONT_EDIT, IDC_MIN_FRONT_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Max Front
    ParamUIDesc(
        PB_SND_MAX_FRONT,
        EDITTYPE_UNIVERSE,
        IDC_MAX_FRONT_EDIT, IDC_MAX_FRONT_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

    // Icon Size
    ParamUIDesc(
        PB_SND_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_ICON_EDIT, IDC_ICON_SPIN,
        0.0f, 10000.0f,
        SPIN_AUTOSCALE),

};

#define PARAMDESC_LENGTH 8

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_FLOAT, NULL, FALSE, 1 },
    { TYPE_FLOAT, NULL, FALSE, 2 },
    { TYPE_FLOAT, NULL, FALSE, 3 },
    { TYPE_FLOAT, NULL, FALSE, 4 },
    { TYPE_FLOAT, NULL, FALSE, 5 },
    { TYPE_FLOAT, NULL, FALSE, 6 },
    { TYPE_INT, NULL, FALSE, 7 },
};

#define CURRENT_VERSION 0

class SoundParamDlgProc : public ParamMapUserDlgProc
{
public:
    SoundObject *ob;

    SoundParamDlgProc(SoundObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR SoundParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                   UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *SoundObject::pmapParam = NULL;

float SoundObject::GetMinBack(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_SND_MIN_BACK, t, f, valid);
    pblock->GetValue(PB_SND_MAX_BACK, t, g, valid);
    if (g < f)
        return g;
    return f;
}

float SoundObject::GetMaxBack(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_SND_MAX_BACK, t, f, valid);
    pblock->GetValue(PB_SND_MIN_BACK, t, g, valid);
    if (g > f)
        return g;
    return f;
}

float SoundObject::GetMinFront(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_SND_MIN_FRONT, t, f, valid);
    pblock->GetValue(PB_SND_MAX_FRONT, t, g, valid);
    if (g < f)
        return g;
    return f;
}

float SoundObject::GetMaxFront(TimeValue t, Interval &valid)
{
    Interval iv;

    float f, g;
    pblock->GetValue(PB_SND_MAX_FRONT, t, f, valid);
    pblock->GetValue(PB_SND_MIN_FRONT, t, g, valid);
    if (g > f)
        return g;
    return f;
}

void SoundObject::SetMinBack(TimeValue t, float f)
{
    pblock->SetValue(PB_SND_MIN_BACK, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void SoundObject::SetMaxBack(TimeValue t, float f)
{
    pblock->SetValue(PB_SND_MAX_BACK, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void SoundObject::SetMinFront(TimeValue t, float f)
{
    pblock->SetValue(PB_SND_MIN_FRONT, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void SoundObject::SetMaxFront(TimeValue t, float f)
{
    pblock->SetValue(PB_SND_MAX_FRONT, t, f);
    NotifyDependents(FOREVER, PART_ALL, REFMSG_CHANGE);
}

void
SoundObject::BeginEditParams(IObjParam *ip, ULONG flags,
                             Animatable *prev)
{
    iObjParams = ip;
    TimeValue t = ip->GetTime(); // not really needed...yet

    if (pmapParam)
    {

        // Left over from last Sound created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_SOUND),
                                     _T("Sound" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new SoundParamDlgProc(this));
    }

    minBackSpin = GetISpinner(GetDlgItem(hRollup, IDC_MIN_BACK_SPIN));
    maxBackSpin = GetISpinner(GetDlgItem(hRollup, IDC_MAX_BACK_SPIN));
    minFrontSpin = GetISpinner(GetDlgItem(hRollup, IDC_MIN_FRONT_SPIN));
    maxFrontSpin = GetISpinner(GetDlgItem(hRollup, IDC_MAX_FRONT_SPIN));

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
}

void
SoundObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

SoundObject::SoundObject()
    : HelperObject()
{
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_SND_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_SND_SIZE, 0, 0.0f);
    pb->SetValue(PB_SND_SPATIALIZE, 0, TRUE);
    pb->SetValue(PB_SND_MAX_BACK, 0, 10.0f);
    pb->SetValue(PB_SND_MIN_BACK, 0, 1.0f);
    pb->SetValue(PB_SND_MAX_FRONT, 0, 10.0f);
    pb->SetValue(PB_SND_MIN_FRONT, 0, 1.0f);
    pb->SetValue(PB_SND_PRIORITY, 0, 0.0f);
    pb->SetValue(PB_SND_INTENSITY, 0, 1.0f);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
    previousMode = NULL;
    audioClip = NULL;
}

SoundObject::~SoundObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *SoundObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult SoundObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult SoundObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message)
#endif
{
    // FIXME handle these messages
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        if (hTarget == (RefTargetHandle)audioClip)
            audioClip = NULL;
        break;
    case REFMSG_NODE_NAMECHANGE:
        if (hTarget == (RefTargetHandle)audioClip && hRollup)
            Static_SetText(GetDlgItem(hRollup, IDC_NAME),
                           audioClip->GetName());
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
SoundObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;
    if (ind == 1)
        return audioClip;

    return NULL;
}

void
SoundObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
    }
    else if (ind == 1)
        audioClip = (INode *)rtarg;
}

ObjectState
SoundObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
SoundObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
SoundObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
GetEllipsePoints(float front, float back, Point3 *ellipse)
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
        ellipse[SEGMENTS + i].x = ellipse[i].z;
        ellipse[SEGMENTS + i].y = ellipse[i].y;
        ellipse[SEGMENTS + i].z = ellipse[i].x;
    }
}

void
SoundObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    Matrix3 m = inode->GetObjectTM(t);
    BuildMesh(t);
    box = mesh.getBoundingBox();
    if (inode->Selected())
    {
        float minFront, minBack, maxFront, maxBack;
        pblock->GetValue(PB_SND_MIN_FRONT, t, minFront, FOREVER);
        pblock->GetValue(PB_SND_MIN_BACK, t, minBack, FOREVER);
        pblock->GetValue(PB_SND_MAX_FRONT, t, maxFront, FOREVER);
        pblock->GetValue(PB_SND_MAX_BACK, t, maxBack, FOREVER);

        Point3 ellipse[2 * SEGMENTS];
        GetEllipsePoints(minFront, minBack, ellipse);
        int i;
        for (i = 0; i < 2 * SEGMENTS; i++)
        {
            box += ellipse[i];
        }
        GetEllipsePoints(maxFront, maxBack, ellipse);
        for (i = 0; i < 2 * SEGMENTS; i++)
        {
            box += ellipse[i];
        }
    }
}

void
SoundObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
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
SoundObject::BuildMesh(TimeValue t)
{
    float length;
    pblock->GetValue(PB_SND_SIZE, t, length, FOREVER);
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
SoundObject::DrawEllipsoids(TimeValue t, INode *inode, GraphicsWindow *gw)
{
    Matrix3 tm = inode->GetObjectTM(t);
    gw->setTransform(tm);
    float minFront, minBack, maxFront, maxBack;
    pblock->GetValue(PB_SND_MIN_FRONT, t, minFront, FOREVER);
    pblock->GetValue(PB_SND_MIN_BACK, t, minBack, FOREVER);
    pblock->GetValue(PB_SND_MAX_FRONT, t, maxFront, FOREVER);
    pblock->GetValue(PB_SND_MAX_BACK, t, maxBack, FOREVER);

    Point3 ellipse[2 * SEGMENTS + 1];

    // Draw the Min ellipsoid in Blue
    gw->setColor(LINE_COLOR, 0.0f, 0.0f, 1.0f);
    GetEllipsePoints(minFront, minBack, ellipse);
    gw->polyline(SEGMENTS, ellipse, NULL, NULL, TRUE, NULL);
    gw->polyline(SEGMENTS, ellipse + SEGMENTS, NULL, NULL, TRUE, NULL);

    // Draw the Max ellipsoid in Red
    gw->setColor(LINE_COLOR, 1.0f, 0.0f, 0.0f);
    GetEllipsePoints(maxFront, maxBack, ellipse);
    gw->polyline(SEGMENTS, ellipse, NULL, NULL, TRUE, NULL);
    gw->polyline(SEGMENTS, ellipse + SEGMENTS, NULL, NULL, TRUE, NULL);
}

int
SoundObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    float radius;
    pblock->GetValue(PB_SND_SIZE, t, radius, FOREVER);
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

int
SoundObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
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

class SoundCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    SoundObject *soundObject;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(SoundObject *obj) { soundObject = obj; }
};

int
SoundCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags,
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
            soundObject->pblock->SetValue(PB_SND_SIZE,
                                          soundObject->iObjParams->GetTime(), radius);
            soundObject->pmapParam->Invalidate();
            if (flags & MOUSE_CTRL)
            {
                float ang = (float)atan2(p1.y - p0.y, p1.x - p0.x);
                mat.PreRotateZ(soundObject->iObjParams->SnapAngle(ang));
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
static SoundCreateCallBack SoundCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
SoundObject::GetCreateMouseCallBack()
{
    SoundCreateCB.SetObj(this);
    return (&SoundCreateCB);
}

RefTargetHandle
SoundObject::Clone(RemapDir &remap)
{
    SoundObject *ts = new SoundObject();
    ts->ReplaceReference(0, pblock->Clone(remap));
    if (audioClip)
    {
        if (remap.FindMapping(audioClip))
            ts->ReplaceReference(1, remap.FindMapping(audioClip));
        else
            ts->ReplaceReference(1, audioClip);
    }
    BaseClone(this, ts, remap);
    return ts;
}
