/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: onoff.h
 
    DESCRIPTION:  Defines a VRML 2.0 OnOffSwitch helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 4 Sept. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __OnOffSwitch__H__

#define __OnOffSwitch__H__

#define OnOffSwitch_CLASS_ID1 0x73fa3443
#define OnOffSwitch_CLASS_ID2 0xF002BDAE

#define OnOffSwitchClassID Class_ID(OnOffSwitch_CLASS_ID1, OnOffSwitch_CLASS_ID2)

extern ClassDesc *GetOnOffSwitchDesc();

class OnOffSwitchCreateCallBack;
class OnOffSwitchObjPick;

class OnOffSwitchObj
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
    OnOffSwitchObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class OnOffSwitchObject : public HelperObject
{
    friend class OnOffSwitchCreateCallBack;
    friend class OnOffSwitchObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);

public:
    // Class vars
    static HWND hRollup;
    static int dlgPrevSel;
    BOOL needsScript; // Do we need to generate a script node?

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif
    float radius;
    static IObjParam *iObjParams;

    Mesh mesh;
    void BuildMesh(TimeValue t);

    CommandMode *previousMode;

    static ICustButton *OffPickButton;
    static ICustButton *OnPickButton;

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    INode *onObject;
    INode *offObject;

    OnOffSwitchObject();
    ~OnOffSwitchObject();

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
        return GetString(IDS_ONOFF_SWITCH);
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_ONOFF_SWITCH);
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
        return Class_ID(OnOffSwitch_CLASS_ID1,
                        OnOffSwitch_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = GetString(IDS_ONOFF_SWITCH_CLASS);
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

    int NumRefs()
    {
        return 3;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
};

#define PB_ONOFF_SIZE 0
#define PB_ONOFF_LENGTH 1

#endif
