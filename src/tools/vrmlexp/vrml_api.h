/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: vrml_api.h

	DESCRIPTION:  simple interface into the vrml exporter

	CREATED BY: greg finch

	HISTORY: created 1 may, 1997

 *>	Copyright (c) 1997, All Rights Reserved.
 **********************************************************************/

#ifndef __VRML_API__H__
#define __VRML_API__H__

struct PreSceneParam
{
    int version; // just in case we rev. this thing
    int indent; // indent level
    Interface *i; // MAX's interface pointer
    TCHAR *fName; // the name of the output file
};

struct PostSceneParam
{
    int version; // just in case we rev. this thing
    int indent; // indent level
    Interface *i; // MAX's interface pointer
    TCHAR *fName; // the name of the output file
};

struct PreNodeParam
{
    int version; // just in case we rev. this thing
    int indent; // indent level
    Interface *i; // MAX's interface pointer
    TCHAR *fName; // the name of the output file
    INode *node; // the node
};

struct PostNodeParam
{
    int version; // just in case we rev. this thing
    int indent; // indent level
    Interface *i; // MAX's interface pointer
    TCHAR *fName; // the name of the output file
    INode *node; // the node
};

typedef int(FAR WINAPI *DllPreScene)(PreSceneParam *p);
typedef void(FAR WINAPI *DllPostScene)(PostSceneParam *p);
typedef int(FAR WINAPI *DllPostNode)(PostNodeParam *p);
typedef int(FAR WINAPI *DllPreNode)(PreNodeParam *p);

// Export Callback Support
#define PreSceneCallback (1 << 0)
#define PostSceneCallback (1 << 1)
#define PreNodeCallback (1 << 2)
#define PostNodeCallback (1 << 3)

#define WroteNodeFailed 0
#define WroteNode (1 << 0)
#define WroteNodeChildren (1 << 1)

class CallbackTable
{
public:
    CallbackTable();
    ~CallbackTable();

    int GetKeyCount();
    TCHAR *GetKey(int i);

    int GetDllCount();
    TCHAR *GetDll(int i);

    int GetDLLHandleCount();
    HMODULE GetDLLHandle(int i);

    int GetPreSceneCount();
    DllPreScene GetPreScene(int i);

    int GetPostSceneCount();
    DllPostScene GetPostScene(int i);

    int GetPreNodeCount();
    DllPreNode GetPreNode(int i);

    int GetPostNodeCount();
    DllPostNode GetPostNode(int i);

    BOOL GetCallbackMethods(Interface *ip);

private:
    int AddKey(TCHAR *s);
    int AddDll(TCHAR *s);
    int AddDLLHandle(HMODULE h);
    int AddPreScene(DllPreScene p);
    int AddPostScene(DllPostScene p);
    int AddPreNode(DllPreNode p);
    int AddPostNode(DllPostNode p);

    Tab<HMODULE> mHLibInst;
    Tab<TCHAR *> mDllKeys;
    Tab<TCHAR *> mDlls;
    Tab<DllPreScene> mPreScene;
    Tab<DllPostScene> mPostScene;
    Tab<DllPreNode> mPreNode;
    Tab<DllPostNode> mPostNode;
};

#endif