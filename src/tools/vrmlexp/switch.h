/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: switch.h
 
    DESCRIPTION:  Defines a VRML 2.0 Switch helper
 
    CREATED BY: Scott Morrison
 
    HISTORY: created 4 Sept. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __Switch__H__

#define __Switch__H__

#define Switch_CLASS_ID1 0x73fb3452
#define Switch_CLASS_ID2 0xF402BDAD

#define SwitchClassID Class_ID(Switch_CLASS_ID1, Switch_CLASS_ID2)

#define TARGETMSG_LOADFINISHED 0x00010001

extern ClassDesc *GetSwitchDesc();

class SwitchCreateCallBack;
class SwitchObjPick;

class SwitchObj
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
    SwitchObj(INode *n = NULL)
    {
        node = n;
        ResetStr();
    }
};

class SwitchObject : public HelperObject
{
    friend class SwitchCreateCallBack;
    friend class SwitchObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(SwitchObject *ob);

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

    Tab<SwitchObj *> objects;
    CommandMode *previousMode;

    static ICustButton *SwitchPickButton;
    static ICustButton *ParentPickButton;

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    //INode* triggerObject;

    SwitchObject();
    ~SwitchObject();

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
        return GetString(IDS_SWITCH);
    }

    Tab<SwitchObj *> GetObjects()
    {
        return objects;
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_SWITCH);
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
        return Class_ID(Switch_CLASS_ID1,
                        Switch_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = TSTR(GetString(IDS_SWITCH_CLASS));
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
        return objects.Count() + 2;
    }
    RefTargetHandle GetReference(int i);
    void SetReference(int i, RefTargetHandle rtarg);
    //    IOResult Load(ILoad *iload) ;
};

#define PB_S_SIZE 0
#define PB_S_DEFAULT 1
#define PB_S_ALLOW_NONE 2
#define PB_S_NUMOBJS 3
#define PB_S_LENGTH 4

#endif
