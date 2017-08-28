/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: TUIParam.h

	DESCRIPTION:	Template Utility

	CREATED BY:

	HISTORY:

 *>	Copyright (c) 1997, All Rights Reserved.
 **********************************************************************/

#ifndef __REFCHECK__H
#define __REFCHECK__H

#define TUIPARAM_CLASS_ID1 0x74fb3451
#define TUIPARAM_CLASS_ID2 0xF412BDDE

#define TUIPARAM_ClassID Class_ID(TUIPARAM_CLASS_ID1, TUIPARAM_CLASS_ID2)

#include "Max.h"
#include "resource.h"
#include "utilapi.h"
#include "istdplug.h"
#include <io.h>
#include <map>

#if MAX_PRODUCT_VERSION_MAJOR > 14 && ! defined FASTIO
#include "maxtextfile.h"
#include "BufferedStream.h"
#define MAXSTREAM BufferedStream &
//#define MAXSTREAM MaxSDK::Util::TextFile::Writer &
#else
#define MAXSTREAM FILE *
#endif

#include "iparamb2.h"
#include "iparamm2.h"

using namespace std;

#define REFCHECK_CLASS_ID Class_ID(0xa7d423ed, 0x64de98f9)

extern ClassDesc *GetTUIParamDesc();
//extern HINSTANCE	hInstance;
extern TCHAR *GetString(int id);

#define COPYWARN_YES 0x00
#define COPYWARN_YESTOALL 0x01
#define COPYWARN_NO 0x02
#define COPYWARN_NOTOALL 0x03

#define PB_S_POSX 0
#define PB_S_POSY 1
#define PB_S_MIN 2
#define PB_S_MAX 3
#define PB_S_VALUE 4
#define PB_S_VAL 5
#define PB_S_DATA 6

/**
 * The NameEnumCallBack used to find all Light Dist. files.
 */

class TabletUIObj;
class TabletUIElement;
class coTUIFrame;
class coTUISplitter;
class INODE;

class ComboBoxObj
{
public:
    INode *switchNode;
    INode *node;
    TSTR listStr;
    TSTR comboBoxName;

    ComboBoxObj(INode *swNode, INode *inode, TSTR &name);
    ~ComboBoxObj(){};
};

class TUIParam : public HelperObject
{

    static ParamUIDesc descParam[];
    static ParamBlockDescID descVer0[];

public:
    IUtil *iu;
    Interface *ip;
    TabletUIElement *myElem;
    static HWND hRollup;

    IParamBlock *pTUIParamBlock;
    static IParamMap *pTUIParamMap;

    TUIParam();
    ~TUIParam();

    virtual void PrintAdditional(MAXSTREAM mStream) = 0;
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval) = 0;

    virtual void BeginEditParams(Interface *ip, IUtil *iu) = 0;
    virtual void Init(HWND hWnd) = 0;
    void EndEditParams(Interface *ip, IUtil *iu);
    void DeleteThis()
    {
    }
    void ParamBlockGetPos(int *posx, int *posy);
    void rePos();
    template <class T>
    void ParamBlockGetValues(T *min, T *max, T *value)
    {
        if (myElem->myObject->iObjParams != NULL)
        {
            pTUIParamBlock->GetValue(PB_S_MIN, myElem->myObject->iObjParams->GetTime(),
                                     *min, FOREVER);
            pTUIParamBlock->GetValue(PB_S_MAX, myElem->myObject->iObjParams->GetTime(),
                                     *max, FOREVER);
            pTUIParamBlock->GetValue(PB_S_VALUE, myElem->myObject->iObjParams->GetTime(),
                                     *value, FOREVER);
        }
        else
        {
            pTUIParamBlock->GetValue(PB_S_MIN, 0, *min, FOREVER);
            pTUIParamBlock->GetValue(PB_S_MAX, 0, *max, FOREVER);
            pTUIParamBlock->GetValue(PB_S_VALUE, 0, *value, FOREVER);
        }
    };

    void Destroy(HWND hWnd);

    CreateMouseCallBack *GetCreateMouseCallBack()
    {
        return (NULL);
    };
    ObjectState Eval(TimeValue time);

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif

    virtual int NumRefs()
    {
        return 1;
    };
    virtual RefTargetHandle GetReference(int i);
    virtual void SetReference(int i, RefTargetHandle rtarg);

#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir()) = 0;
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap()) = 0;
#endif

    virtual IOResult Save(ISave *isave)
    {
        return IO_OK;
    };
    virtual IOResult Load(ILoad *iload)
    {
        return IO_OK;
    };

    virtual bool ReferenceLoad()
    {
        return true;
    };
};

class TUIParamFloatSlider : public TUIParam
{

public:
    static const int PARAMDESC_LENGTH = 5;
    static const int PARAMBLOCK_LENGTH = 6;
    TUIParamFloatSlider();

    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval);
    virtual void BeginEditParams(Interface *ip, IUtil *iu);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif
    virtual void Init(HWND hWnd);
};

class TUIParamSpinEditField : public TUIParam
{

public:
    static const int PARAMDESC_LENGTH = 6;
    static const int PARAMBLOCK_LENGTH = 6;
    TUIParamSpinEditField();

    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval){};
    virtual void BeginEditParams(Interface *ip, IUtil *iu);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif
    virtual void Init(HWND hWnd){};
};

class TUIParamButton : public TUIParam
{

public:
    static const int PARAMDESC_LENGTH = 2;
    static const int PARAMBLOCK_LENGTH = 2;
    TUIParamButton();

    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval){};
    virtual void BeginEditParams(Interface *ip, IUtil *iu);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif
    virtual void Init(HWND hWnd){};
};

class TUIParamToggleButton : public TUIParamButton
{

public:
    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval);
    void PrintObjects(MAXSTREAM mStream, TabletUIObj *obj);
};

class TUIParamLabel : public TUIParamButton
{

public:
};

class TUIParamFrame : public TUIParam
{

public:
    static const int PARAMDESC_LENGTH = 2;
    static const int PARAMBLOCK_LENGTH = 4;
    TUIParamFrame();

    virtual void setValues(coTUIFrame *tf);
    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval){};
    virtual void BeginEditParams(Interface *ip, IUtil *iu);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif
    virtual void Init(HWND hWnd);
};

class TUIParamSplitter : public TUIParam
{

public:
    static const int PARAMDESC_LENGTH = 2;
    static const int PARAMBLOCK_LENGTH = 5;
    TUIParamSplitter();

    virtual void setValues(coTUISplitter *tf);
    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval){};
    virtual void BeginEditParams(Interface *ip, IUtil *iu);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif
    virtual void Init(HWND hWnd);
};

class TUIParamComboBox : public TUIParam
{

public:
    static const int PARAMDESC_LENGTH = 2;
    static const int PARAMBLOCK_LENGTH = 4;

    multimap<int, ComboBoxObj *> comboObjects;
    TSTR emptyName;

    TUIParamComboBox();
    ~TUIParamComboBox();

    virtual void PrintAdditional(MAXSTREAM mStream);
    virtual void PrintScript(MAXSTREAM mStream, TSTR objname, float cycleInterval){};
    virtual void BeginEditParams(Interface *ip, IUtil *iu);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    virtual RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#else
    virtual RefTargetHandle Clone(RemapDir &remap = NoRemap());
#endif
    virtual void Init(HWND hWnd);

    void InitDefaultChoice();
    void UpdateComboObjects();
    void UpdateComboBox(int selection);
    multimap<int, ComboBoxObj *>::iterator AddObject(int index, TSTR name);
    void UpdateRefList();
    void AddSwitch(INode *addSwitch);
    void DelSwitch(INode *delSwitch);

#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif
    virtual int NumRefs();
    virtual RefTargetHandle GetReference(int i);
    virtual void SetReference(int i, RefTargetHandle rtarg);

    virtual IOResult Save(ISave *isave);
    virtual IOResult Load(ILoad *iload);

    virtual bool ReferenceLoad();
};

#endif // __REFCHECK__H
