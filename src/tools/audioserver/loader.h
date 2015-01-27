/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//////////////////////////////////////////////////////////////////////////////
//
// loader_example\loader.h
//
// Simple implementation of the DirectMusic loader.
//
// Copyright (c) Microsoft Corporation. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include <windows.h>
#include <dmusici.h>

class CMyIStream : public IStream, public IDirectMusicGetLoader
{
public:
    CMyIStream();
    ~CMyIStream();

    HRESULT Attach(LPCTSTR tzFile, IDirectMusicLoader *pLoader);
    void Detach();

    // IUnknown methods
    //
    STDMETHODIMP QueryInterface(REFIID iid, void **ppv);
    STDMETHODIMP_(ULONG) AddRef();
    STDMETHODIMP_(ULONG) Release();

    // IDirectMusicGetLoader
    //
    STDMETHODIMP GetLoader(IDirectMusicLoader **ppLoader);

    // IStream methods
    //
    STDMETHODIMP Read(void *pv, ULONG cb, ULONG *pcb);
    STDMETHODIMP Seek(LARGE_INTEGER dlibMove, DWORD dwOrigin, ULARGE_INTEGER *plibNewPosition);
    STDMETHODIMP Clone(IStream **ppstm);
    STDMETHODIMP Write(const void *pv, ULONG cb, ULONG *pcb);
    STDMETHODIMP SetSize(ULARGE_INTEGER libNewSize);
    STDMETHODIMP CopyTo(IStream *pstm, ULARGE_INTEGER cb, ULARGE_INTEGER *pcbRead, ULARGE_INTEGER *pcbWritten);
    STDMETHODIMP Commit(DWORD grfCommitFlags);
    STDMETHODIMP Revert(void);
    STDMETHODIMP LockRegion(ULARGE_INTEGER libOffset, ULARGE_INTEGER cb, DWORD dwLockType);
    STDMETHODIMP UnlockRegion(ULARGE_INTEGER libOffset, ULARGE_INTEGER cb, DWORD dwLockType);
    STDMETHODIMP Stat(STATSTG *pstatstg, DWORD grfStatFlag);

private:
    LONG m_cRef; // COM Reference count
    IDirectMusicLoader *m_pLoader; // Owning loader object
    HANDLE m_hFile; // Open file object
    TCHAR m_tzFile[MAX_PATH]; // For clone method
};

class CObjectRef
{
public:
    CObjectRef()
    {
        m_pNext = NULL;
        m_pObject = NULL;
        m_pwsFileName = NULL;
    };
    ~CObjectRef()
    {
        delete[] m_pwsFileName;
    };
    CObjectRef *m_pNext;
    WCHAR *m_pwsFileName;
    GUID m_guidObject;
    IDirectMusicObject *m_pObject;
};

class C_DMLoader : public IDirectMusicLoader
{
public:
    C_DMLoader();
    ~C_DMLoader();
    HRESULT Init();

    // IUnknown methods
    //
    STDMETHODIMP QueryInterface(REFIID iid, void **ppv);
    STDMETHODIMP_(ULONG) AddRef();
    STDMETHODIMP_(ULONG) Release();

    // IDirectMusicLoader methods
    //
    STDMETHODIMP GetObject(LPDMUS_OBJECTDESC pDesc, REFIID riid, LPVOID FAR *ppv);
    STDMETHODIMP SetObject(LPDMUS_OBJECTDESC pDesc);
    STDMETHODIMP SetSearchDirectory(REFGUID rguidClass, WCHAR *pwzPath, BOOL fClear);
    STDMETHODIMP ScanDirectory(REFGUID rguidClass, WCHAR *pwzFileExtension, WCHAR *pwzScanFileName);
    STDMETHODIMP CacheObject(IDirectMusicObject *pObject);
    STDMETHODIMP ReleaseObject(IDirectMusicObject *pObject);
    STDMETHODIMP ClearCache(REFGUID rguidClass);
    STDMETHODIMP EnableCache(REFGUID rguidClass, BOOL fEnable);
    STDMETHODIMP EnumObject(REFGUID rguidClass, DWORD dwIndex, LPDMUS_OBJECTDESC pDesc);

private:
    LONG m_cRef; // COM Reference count
    WCHAR m_wzSearchPath[MAX_PATH];
    CObjectRef *m_pObjectList; // List of already loaded objects (the cache.)
};
