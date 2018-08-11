/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: vrml_api.cpp

	DESCRIPTION:  simple interface into the vrml exporter

	CREATED BY: greg finch

	HISTORY: created 1 may, 1997

 *>	Copyright (c) 1997, All Rights Reserved.
 **********************************************************************/
#include "vrml.h"
#include "vrml_api.h"

CallbackTable::~CallbackTable()
{
    mDllKeys.Delete(0, mDllKeys.Count());
    mDllKeys.Shrink();
    mDlls.Delete(0, mDlls.Count());
    mDlls.Shrink();
    mPreScene.Delete(0, mPreScene.Count());
    mPreScene.Shrink();
    mPostScene.Delete(0, mPostScene.Count());
    mPostScene.Shrink();
    mPreNode.Delete(0, mPreNode.Count());
    mPreNode.Shrink();
    mPostNode.Delete(0, mPostNode.Count());
    mPostNode.Shrink();

    for (int i = 0; i < mHLibInst.Count(); i++)
    {
        FreeLibrary((HMODULE)mHLibInst[i]);
        mHLibInst[i] = NULL;
    }
    mHLibInst.Delete(0, mHLibInst.Count());
    mHLibInst.Shrink();
}

CallbackTable::CallbackTable() {}

int
CallbackTable::AddKey(TCHAR *s)
{
    return mDllKeys.Insert(mDllKeys.Count(), 1, &s);
}

int
CallbackTable::GetKeyCount()
{
    return mDllKeys.Count();
}

TCHAR *
CallbackTable::GetKey(int i)
{
    return mDllKeys[i];
}

int
CallbackTable::AddDll(TCHAR *s)
{
    return mDlls.Insert(mDlls.Count(), 1, &s);
}

int
CallbackTable::GetDllCount()
{
    return mDlls.Count();
}

TCHAR *
CallbackTable::GetDll(int i)
{
    return mDlls[i];
}

int
CallbackTable::AddPreScene(DllPreScene p)
{
    return mPreScene.Insert(mPreScene.Count(), 1, &p);
}

int
CallbackTable::GetPreSceneCount()
{
    return mPreScene.Count();
}

DllPreScene
CallbackTable::GetPreScene(int i)
{
    return mPreScene[i];
}

int
CallbackTable::AddPostScene(DllPostScene p)
{
    return mPostScene.Insert(mPostScene.Count(), 1, &p);
}

int
CallbackTable::GetPostSceneCount()
{
    return mPostScene.Count();
}

DllPostScene
CallbackTable::GetPostScene(int i)
{
    return mPostScene[i];
}

int
CallbackTable::AddPreNode(DllPreNode p)
{
    return mPreNode.Insert(mPreNode.Count(), 1, &p);
}

int
CallbackTable::GetPreNodeCount()
{
    return mPreNode.Count();
}

DllPreNode
CallbackTable::GetPreNode(int i)
{
    return mPreNode[i];
}

int
CallbackTable::AddPostNode(DllPostNode p)
{
    return mPostNode.Insert(mPostNode.Count(), 1, &p);
}

int
CallbackTable::GetPostNodeCount()
{
    return mPostNode.Count();
}

DllPostNode
CallbackTable::GetPostNode(int i)
{
    return mPostNode[i];
}

int
CallbackTable::AddDLLHandle(HMODULE h)
{
    return mHLibInst.Insert(mHLibInst.Count(), 1, &h);
}

int
CallbackTable::GetDLLHandleCount()
{
    return mHLibInst.Count();
}

HMODULE
CallbackTable::GetDLLHandle(int i)
{
    return mHLibInst[i];
}

BOOL
CallbackTable::GetCallbackMethods(Interface *ip)
{
    int i;
    LPCTSTR lpSection = _T("Dlls");
    LPCTSTR lpKey = NULL;
    LPCTSTR lpDefault = _T("none");
    TCHAR lpBuf[MAX_PATH];
    DWORD nSize = sizeof(lpBuf);

    const TCHAR *maxPlugCfgPath = ip->GetDir(APP_PLUGCFG_DIR);
    TCHAR lpINIFileName[MAX_PATH];
    _stprintf(lpINIFileName, _T("%s\\%s"), maxPlugCfgPath, _T("vrmlexp.ini"));

    // see if the INI files exists.
    WIN32_FIND_DATA file;
    HANDLE findhandle = FindFirstFile(lpINIFileName, &file);
    FindClose(findhandle);
    if (findhandle == INVALID_HANDLE_VALUE)
    {
        /*  if the .ini file is missing assume they aren't using the API
        TSTR buf = "couldn't find INI file";
	    MessageBox(GetActiveWindow(), buf, " ", MB_OK|MB_TASKMODAL);
        */
        return FALSE;
    }

    // get dllKeys
    MaxSDK::Util::GetPrivateProfileString(lpSection, lpKey, lpDefault, (LPTSTR)lpBuf, nSize, (LPCTSTR)lpINIFileName);
    TCHAR *tmpPtr = lpBuf;
    while (tmpPtr[0] != '\0')
    {
        TCHAR *dllKey = (TCHAR *)malloc(_tcslen(tmpPtr) * sizeof(TCHAR));
        _tcscpy(dllKey, tmpPtr);
        AddKey(dllKey);
        tmpPtr = _tcschr(tmpPtr, '\0') + 1;
    }

    // check validity of INI file
    if (!lstrcmp(lpBuf, _T("none")))
    {
        TSTR buf = _T("couldn't find [dlls] section in INI file");
        MessageBox(GetActiveWindow(), buf, _T(" "), MB_OK | MB_TASKMODAL);
        return FALSE;
    }

    // get dlls
    for (i = 0; i < GetKeyCount(); i++)
    {
        MaxSDK::Util::GetPrivateProfileString(lpSection, GetKey(i), lpDefault, (LPTSTR)lpBuf, nSize, lpINIFileName);
        TCHAR *dllPtr = (TCHAR *)malloc(_tcslen(lpBuf) * sizeof(TCHAR));
        _tcscpy(dllPtr, lpBuf);
        AddDll(dllPtr);
    }

    // load the dlls
    for (i = GetDllCount(); i--;)
    {
        HINSTANCE libInst = (HINSTANCE)LoadLibraryEx(GetDll(i), NULL, 0);
        if (libInst)
        {
            AddDLLHandle(libInst);
            FARPROC lpCallbackType = NULL;
            FARPROC lpProc = NULL;
            TCHAR lpStr[64];
            __int64 CallbackType;

            lpCallbackType = GetProcAddress((HMODULE)libInst, (LPCSTR) "ExportLibSupport");
            if (lpCallbackType)
            {
                CallbackType = (*(lpCallbackType))();
            }

            // Export Lib support for pre-scene export
            if (PreSceneCallback & CallbackType)
            {
                _tcscpy(lpStr, _T("PreSceneExport"));
                DllPreScene lpProc = (DllPreScene)GetProcAddress((HMODULE)libInst, (LPCSTR)lpStr);
                if (!lpProc)
                {
                    TCHAR buf[256];
                    _stprintf(buf, _T("method %s not implimented"), lpStr);
                    MessageBox(GetActiveWindow(), buf, _T("Export Lib"), MB_OK | MB_TASKMODAL);
                }
                AddPreScene(lpProc);
            }

            // Export Lib support for post-scene export
            if (PostSceneCallback & CallbackType)
            {
                _tcscpy(lpStr, _T("PostSceneExport"));
                DllPostScene lpProc = (DllPostScene)GetProcAddress((HMODULE)libInst, (LPCSTR)lpStr);
                if (!lpProc)
                {
                    TCHAR buf[256];
                    _stprintf(buf, _T("method %s not implimented"), lpStr);
                    MessageBox(GetActiveWindow(), buf, _T("Export Lib"), MB_OK | MB_TASKMODAL);
                }
                AddPostScene(lpProc);
            }

            // Export Lib support for pre-node export
            if (PreNodeCallback & CallbackType)
            {
                _tcscpy(lpStr, _T("PreNodeExport"));
                DllPreNode lpProc = (DllPreNode)GetProcAddress((HMODULE)libInst, (LPCSTR)lpStr);
                if (!lpProc)
                {
                    TCHAR buf[256];
                    _stprintf(buf, _T("method %s not implimented"), lpStr);
                    MessageBox(GetActiveWindow(), buf, _T("Export Lib"), MB_OK | MB_TASKMODAL);
                }
                AddPreNode(lpProc);
            }

            // Export Lib support for post-node export
            if (PostNodeCallback & CallbackType)
            {
                _tcscpy(lpStr, _T("PostNodeExport"));
                DllPostNode lpProc = (DllPostNode)GetProcAddress((HMODULE)libInst, (LPCSTR)lpStr);
                if (!lpProc)
                {
                    TCHAR buf[256];
                    _stprintf(buf, _T("method %s not implimented"), lpStr);
                    MessageBox(GetActiveWindow(), buf, _T("Export Lib"), MB_OK | MB_TASKMODAL);
                }
                AddPostNode(lpProc);
            }
        }
        else
        {
            TCHAR buf[256];
            _stprintf(buf, _T("Lib [%s] Load Failed"), GetDll(i));
            MessageBox(GetActiveWindow(), buf, _T("Export Lib"), MB_OK | MB_TASKMODAL);
        }
    }

    return TRUE;
}
