/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//////////////////////////////////////////////////////////////////////////////
//
// Istream.cpp
//
// Demonstrates implementing IStream and IDirectMusicloader for use by a
// custom DirectMusic loader. This implementation simply wraps the Win32
// file system, but the data can come from anywhere.
//
// Copyright (c) Microsoft Corporation. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "precomp.h"

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::CMyIStream
//
CMyIStream::CMyIStream()
    : m_cRef(1)
    , m_pLoader(NULL)
    , m_hFile(INVALID_HANDLE_VALUE)
{
    m_tzFile[0] = _T('\0');
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::~CMyIStream
//
CMyIStream::~CMyIStream()
{
    Detach();
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Attach
//
// Attach is called by the loader to open a file on this stream. A pointer
// to the loader is also cached.
//
// An IStream implementation to be used with the loader must also
// implement IDirectMusicLoader. This interface's one method, GetLoader, enables
// the system to load referenced content automatically.
//
HRESULT CMyIStream::Attach(LPCTSTR tzFile, IDirectMusicLoader *pLoader)
{
    Detach();

    m_hFile = CreateFile(
        tzFile,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL, // Security attributes
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL); // hTemplateFile

    if (m_hFile == INVALID_HANDLE_VALUE)
    {
        return E_FAIL;
    }

    m_pLoader = pLoader;
    m_pLoader->AddRef();

    _tcscpy(m_tzFile, tzFile);

    return S_OK;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Detach
//
// Close the file in preparation for another Attach or on final Release.
//
void CMyIStream::Detach()
{
    RELEASE(m_pLoader);

    if (m_hFile != INVALID_HANDLE_VALUE)
    {
        CloseHandle(m_hFile);
    }

    m_tzFile[0] = _T('\0');
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::QueryInterface
//
STDMETHODIMP CMyIStream::QueryInterface(REFIID iid, void **ppv)
{
    *ppv = NULL;

    if (iid == IID_IUnknown || iid == IID_ISequentialStream)
    {
        // ISequentialStream is the base class for IStream, which
        // provides only sequential file I/O.
        //
        *ppv = (void *)static_cast<ISequentialStream *>(this);
    }
    else if (iid == IID_IStream)
    {
        // IStream adds the concept of a file pointer to the
        // sequential stream.
        //
        *ppv = (void *)static_cast<IStream *>(this);
    }
    else if (iid == IID_IDirectMusicGetLoader)
    {
        // This is a DirectMusic specific interface to get back
        // the loader that created this IStream.
        //
        *ppv = (void *)static_cast<IDirectMusicGetLoader *>(this);
    }
    else
    {
        return E_NOINTERFACE;
    }

    AddRef();
    return S_OK;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::AddRef
//
STDMETHODIMP_(ULONG) CMyIStream::AddRef()
{
    return InterlockedIncrement(&m_cRef);
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Release
//
STDMETHODIMP_(ULONG) CMyIStream::Release()
{
    if (InterlockedDecrement(&m_cRef) == 0)
    {
        delete this;
        return 0;
    }

    return m_cRef;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::GetLoader
//
// Return the loader which created this stream. Under the rules of COM, we
// have to AddRef the interface pointer first.
//
STDMETHODIMP CMyIStream::GetLoader(IDirectMusicLoader **ppLoader)
{
    m_pLoader->AddRef();
    *ppLoader = m_pLoader;

    return S_OK;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Read
//
// Wraps a read call. The call is considered to have failed if not
// all the data could be read.
//
STDMETHODIMP CMyIStream::Read(void *pv, ULONG cb, ULONG *pcb)
{
    ULONG cbRead;

    if (m_hFile == INVALID_HANDLE_VALUE)
    {
        return E_FAIL;
    }

    if (pcb == NULL)
    {
        pcb = &cbRead;
    }

    if (!ReadFile(m_hFile, pv, cb, pcb, NULL))
    {
        return E_FAIL;
    }
    else if (*pcb != cb)
    {
        return E_FAIL;
    }

    return S_OK;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Seek
//
// Move the file pointer.
//
// Be very careful here. The file pointer is 64 bits. If you are implementing
// an IStream over a medium or file system that only uses a 32-bit file pointer,
// you will have to handle negative offsets properly when casting the file
// pointer to 32 bits. DirectMusic will make calls to move the file pointer by
// a relative offset of a negative number, and improper handling of this case
// can cause loads to fail in mysterious ways.
//
STDMETHODIMP CMyIStream::Seek(LARGE_INTEGER dlibMove, DWORD dwOrigin, ULARGE_INTEGER *plibNewPosition)
{
    if (m_hFile == INVALID_HANDLE_VALUE)
    {
        return E_FAIL;
    }

    LARGE_INTEGER liNewPos;

    liNewPos.HighPart = dlibMove.HighPart;

    liNewPos.LowPart = SetFilePointer(m_hFile, dlibMove.LowPart, &liNewPos.HighPart, dwOrigin);

    if (liNewPos.LowPart == 0xFFFFFFFF && GetLastError() != NO_ERROR)
    {
        return E_FAIL;
    }

    if (plibNewPosition)
    {
        plibNewPosition->QuadPart = liNewPos.QuadPart;
    }

    return S_OK;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Clone
//
// Create a duplicate stream.
//
// Two things to be careful of here are the reference counts (the new object
// must have a reference count of 1) and to properly clone the file pointer.
//
STDMETHODIMP CMyIStream::Clone(IStream **ppstm)
{
    CMyIStream *pOther = new CMyIStream;

    if (pOther == NULL)
    {
        return E_OUTOFMEMORY;
    }

    if (m_hFile != INVALID_HANDLE_VALUE)
    {
        ULARGE_INTEGER ullCurrentPosition;

        HRESULT hr = pOther->Attach(m_tzFile, m_pLoader);

        if (SUCCEEDED(hr))
        {
            LARGE_INTEGER liZero = { 0 };

            hr = Seek(liZero, STREAM_SEEK_CUR, &ullCurrentPosition);
        }

        if (SUCCEEDED(hr))
        {
            LARGE_INTEGER liNewPosition;

            liNewPosition.QuadPart = ullCurrentPosition.QuadPart;

            hr = pOther->Seek(liNewPosition, STREAM_SEEK_SET, NULL);
        }

        if (FAILED(hr))
        {
            RELEASE(pOther);
            return hr;
        }
    }

    *ppstm = static_cast<IStream *>(pOther);

    return S_OK;
}

//============================================================================
//
// IStream methods the DirectMusic loader doesn't use.
//
//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Write
//
STDMETHODIMP CMyIStream::Write(const void *pv, ULONG cb, ULONG *pcb)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::SetSize
//
STDMETHODIMP CMyIStream::SetSize(ULARGE_INTEGER libNewSize)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::CopyTo
//
STDMETHODIMP CMyIStream::CopyTo(IStream *pstm, ULARGE_INTEGER cb, ULARGE_INTEGER *pcbRead, ULARGE_INTEGER *pcbWritten)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Commit
//
STDMETHODIMP CMyIStream::Commit(DWORD grfCommitFlags)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Revert
//
STDMETHODIMP CMyIStream::Revert(void)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::LockRegion
//
STDMETHODIMP CMyIStream::LockRegion(ULARGE_INTEGER libOffset, ULARGE_INTEGER cb, DWORD dwLockType)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::UnlockRegion
//
STDMETHODIMP CMyIStream::UnlockRegion(ULARGE_INTEGER libOffset, ULARGE_INTEGER cb, DWORD dwLockType)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// CMyIStream::Stat
//
STDMETHODIMP CMyIStream::Stat(STATSTG *pstatstg, DWORD grfStatFlag)
{
    return E_NOTIMPL;
}
