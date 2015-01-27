/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: cover.h

    DESCRIPTION:  Defines a VRML 2.0 COVER helper

    CREATED BY: Uwe Woessner

    HISTORY: created 3.4.2003
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef __COVER__H__

#define __COVER__H__

#define COVER_CLASS_ID1 0x73fa4142
#define COVER_CLASS_ID2 0xFDE4BDAD

#define COVERClassID Class_ID(COVER_CLASS_ID1, COVER_CLASS_ID2)

extern ClassDesc *GetCOVERDesc();

class COVERCreateCallBack;
class COVERObjPick;

extern TCHAR *somekeys[];

class COVERObj
{
    static int KeyIndex;

public:
    INode *node;
    TSTR listStr;
    TSTR keyStr;
    void ResetStr(void)
    {
        if (node)
            listStr.printf(_T("%s: %s"), keyStr, node->GetName());
        else
            listStr.printf(_T("%s: %s"), keyStr, _T("NO_NAME"));
    }
    COVERObj(INode *n = NULL)
    {
        node = n;
        keyStr = somekeys[KeyIndex % 100];
        KeyIndex++;
        ResetStr();
    }
};

class COVERObject : public HelperObject
{
    friend class COVERCreateCallBack;
    friend class COVERObjPick;
    friend INT_PTR CALLBACK RollupDialogProc(HWND hDlg, UINT message,
                                             WPARAM wParam, LPARAM lParam);
    friend void BuildObjectList(COVERObject *ob);

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

    Tab<COVERObj *> objects;
    CommandMode *previousMode;

    static ICustButton *TrackedObjectPickButton;
    static ICustButton *KeyboardPickButton;

    IParamBlock *pblock;
    static IParamMap *pmapParam;

    INode *triggerObject;

    TSTR KeysString;
    Tab<TSTR> Keys;

    COVERObject();
    ~COVERObject();

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
        return GetString(IDS_COVER_SENSOR);
    }

    Tab<COVERObj *> GetObjects()
    {
        return objects;
    }

    // From Object
    ObjectState Eval(TimeValue time);
    void InitNodeName(TSTR &s)
    {
        s = GetString(IDS_COVER_SENSOR);
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
        return Class_ID(COVER_CLASS_ID1,
                        COVER_CLASS_ID2);
    }
    void GetClassName(TSTR &s)
    {
        s = GetString(IDS_COVER_SENSOR_CLASS);
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
    IOResult Load(ILoad *iload);
    IOResult Save(ISave *isave);
};

#define PB_COVER_SIZE 0
#define PB_COVER_NUMOBJS 1
#define PB_COVER_LENGTH 2

#endif
