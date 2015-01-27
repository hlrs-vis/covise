/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/////////////////////////////////////////////////////////////////////////////
//
// Loader.cpp
//
// Simple implementation of the DirectMusic loader.
//
// Copyright (c) Microsoft Corporation. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "precomp.h"

static MUSIC_TIME MusicTimeFromWav(IStream *pStream);

//////////////////////////////////////////////////////////////////////////////
//
// C_DMLoader::C_DMLoader
//
C_DMLoader::C_DMLoader()
    : m_cRef(1)
    , m_pObjectList(NULL)
{
    // Default to the current directory.

    wcscpy(m_wzSearchPath, L".\\");
}

//////////////////////////////////////////////////////////////////////////////
//
// C_DMLoader::~C_DMLoader
//
C_DMLoader::~C_DMLoader()
{
    while (m_pObjectList)
    {
        CObjectRef *pObject = m_pObjectList;
        m_pObjectList = pObject->m_pNext;
        if (pObject->m_pObject)
        {
            pObject->m_pObject->Release();
        }
        delete pObject;
    }
}

//////////////////////////////////////////////////////////////////////////////
//
// C_DMLoader::QueryInterface
//
STDMETHODIMP C_DMLoader::QueryInterface(REFIID iid, void **ppv)
{
    *ppv = NULL;

    if (iid == IID_IUnknown || iid == IID_IDirectMusicLoader)
    {
        *ppv = (void *)static_cast<IDirectMusicLoader *>(this);
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
// C_DMLoader::AddRef
//
STDMETHODIMP_(ULONG) C_DMLoader::AddRef()
{
    return InterlockedIncrement(&m_cRef);
}

//////////////////////////////////////////////////////////////////////////////
//
// C_DMLoader::Release
//
STDMETHODIMP_(ULONG) C_DMLoader::Release()
{
    LONG ulCount = InterlockedDecrement(&m_cRef);
    if (ulCount <= 0)
    {
        delete this;
    }
    return (ULONG)ulCount;
}

//////////////////////////////////////////////////////////////////////////////
//
// C_DMLoader::GetObject
//
// This is the workhorse method of the loader. The DMUS_OBJECTDESC structure
// describes the object to be loaded. There are basically three cases that
// must be handled, based on the following three flags in dwValidData:
//
// 1. DMUS_OBJ_OBJECT. This indicates that the content
//    is being referenced by GUID. The only time you *must* pay attention
//    to this type of load is GUID_DefaultGMCollection. Here we satisfy
//    this request with a cached pointer to GM.DLS.
//
// 2. DMUS_OBJ_FILENAME. This is a request to load by filename. Note that
//    referenced content is loaded by just the filename, so you must have
//    a scheme for locating the directory the content is in. The default
//    DirectMusic loader uses SetSearchDirectory to accomplish this; however,
//    you can find the content in any way you like.
//
// 3. DMUS_OBJ_STREAM. You will receive this load request when content
//    is embedded in a container object within another file. The key
//    thing to note here is that embedded content is loaded recursively,
//    and the caller expects the stream pointer to be unchanged on return
//    from the nested load. Therefore you must clone the stream and load on
//    the cloned stream. You cannot be guaranteed that Load will restore the
//    stream pointer on all media types.
//
//
STDMETHODIMP C_DMLoader::GetObject(LPDMUS_OBJECTDESC pDesc, REFIID riid, LPVOID FAR *ppv)
{
    USES_CONVERSION;

    HRESULT hr;
    IDirectMusicObject *pObject = NULL;
    MUSIC_TIME mt = 0;
    //	char msg[MAX_PATH];

    // First check whether the object has already been loaded.
    // In this case, we are looking for only the object GUID or file name.
    // If the loader sees that the object is already loaded, it
    // returns a pointer to it and increments the reference.
    // It is very important to keep the previously loaded objects
    // cached in this way. Otherwise, objects like DLS collections will get loaded
    // multiple times with a very great expense in memory and efficiency.
    // This is primarily an issue when objects reference each other: for example,
    // when segments reference style and collection objects.

    CObjectRef *pObjectRef = NULL;

    // Scan by GUID.

    if (pDesc->dwValidData & DMUS_OBJ_OBJECT)
    {
        for (pObjectRef = m_pObjectList; pObjectRef; pObjectRef = pObjectRef->m_pNext)
        {
            if (pDesc->guidObject == pObjectRef->m_guidObject)
            {
                break;
            }
        }
    }

    // Scan by file name.

    else if (pDesc->dwValidData & DMUS_OBJ_FILENAME)
    {
        for (pObjectRef = m_pObjectList; pObjectRef; pObjectRef = pObjectRef->m_pNext)
        {
            if (pObjectRef->m_pwsFileName && !wcscmp(pDesc->wszFileName, pObjectRef->m_pwsFileName))
            {
                break;
            }
        }
    }

    // If the object was found, make sure it supports the requested
    // interface, and add a reference, but don't reload it.

    if (pObjectRef)
    {
        hr = E_FAIL;
        if (pObjectRef->m_pObject)
        {
            hr = pObjectRef->m_pObject->QueryInterface(riid, ppv);
        }
        return hr;
    }

    // We didn't find it in the cache, so now we must create it.
    // First the given class must be creatable and must support IDirectMusicObject.

    hr = CoCreateInstance(pDesc->guidClass, NULL, CLSCTX_INPROC_SERVER, IID_IDirectMusicObject, (void **)&pObject);
    if (FAILED(hr))
    {
        return hr;
    }

    if (pDesc->dwValidData & DMUS_OBJ_FILENAME)
    {
        // Load from a file. Make a stream based on the file and load from
        // it.
        //
        // Though we RELEASE the stream pointer at the end, DirectMusic may
        // hold on to the stream if needed (such as for DLS collections or
        // streamed waves).
        //
        WCHAR wzFileName[MAX_PATH];
        WCHAR wzExt[_MAX_EXT];

        if (pDesc->dwValidData & DMUS_OBJ_FULLPATH)
        {
            _wmakepath(wzFileName, NULL, NULL, pDesc->wszFileName, NULL);
        }
        else if (pDesc->dwValidData & DMUS_OBJ_FILENAME)
        {
            _wmakepath(wzFileName, NULL, m_wzSearchPath, pDesc->wszFileName, NULL);
        }
        _wsplitpath(wzFileName, NULL, NULL, NULL, wzExt);
        //		wcstombs( msg, wzFileName, wcslen(wzFileName));
        //		MessageBox(NULL, msg,"wzFileName", MB_OK);

        CMyIStream *pStream = new CMyIStream;
        if (pStream == NULL)
        {
            hr = E_OUTOFMEMORY;
        }

        if (SUCCEEDED(hr))
        {
            hr = pStream->Attach(W2CT(wzFileName), this);
        }

        if (SUCCEEDED(hr))
        {
            if ((0 == _wcsicmp(wzExt, L".wav")) && (riid == IID_IDirectMusicSegment8))
                // This is a workaround for a missing feature in DirectX 8.0
                // See the comments for MusicTimeFromWav.
                //
                mt = MusicTimeFromWav(pStream);
        }

        IPersistStream *pPersistStream = NULL;
        if (SUCCEEDED(hr))
        {
            hr = pObject->QueryInterface(IID_IPersistStream, (void **)&pPersistStream);
        }

        if (SUCCEEDED(hr))
        {
            hr = pPersistStream->Load(pStream);
        }

        RELEASE(pStream);
        RELEASE(pPersistStream);
    }
    else if (pDesc->dwValidData & DMUS_OBJ_STREAM)
    {
        // Loading by stream.
        //
        IStream *pClonedStream = NULL;
        IPersistStream *pPersistStream = NULL;

        hr = pObject->QueryInterface(IID_IPersistStream, (void **)&pPersistStream);
        if (SUCCEEDED(hr))
        {
            hr = pDesc->pStream->Clone(&pClonedStream);
        }

        if (SUCCEEDED(hr))
        {
            hr = pPersistStream->Load(pClonedStream);
        }

        RELEASE(pPersistStream);
        RELEASE(pClonedStream);
    }
    else
    {
        // No way we understand to reference the object.
        //
        hr = E_FAIL;
    }

    if (SUCCEEDED(hr))
    {
        // If we succeeded in loading it, keep a pointer to it,
        // AddRef it, and keep the GUID for finding it next time.
        // To get the GUID, call ParseDescriptor on the object and
        // it will fill in the fields it knows about, including the
        // GUID.
        // Note that this only applies to wave, DLS, and style objects.
        // You may wish to add or remove other object types.

        if ((pDesc->guidClass == CLSID_DirectMusicStyle) || (pDesc->guidClass == CLSID_DirectSoundWave) || (pDesc->guidClass == CLSID_DirectMusicCollection))
        {
            DMUS_OBJECTDESC DESC;
            memset((void *)&DESC, 0, sizeof(DESC));
            DESC.dwSize = sizeof(DMUS_OBJECTDESC);
            pObject->GetDescriptor(&DESC);
            if ((DESC.dwValidData & DMUS_OBJ_OBJECT) || (pDesc->dwValidData & DMUS_OBJ_FILENAME))
            {
                CObjectRef *pObjectRef = new CObjectRef;
                if (pObjectRef)
                {
                    pObjectRef->m_guidObject = DESC.guidObject;
                    if (pDesc->dwValidData & DMUS_OBJ_FILENAME)
                    {
                        pObjectRef->m_pwsFileName = new WCHAR[wcslen(pDesc->wszFileName) + 1];
                        wcscpy(pObjectRef->m_pwsFileName, pDesc->wszFileName);
                    }
                    pObjectRef->m_pNext = m_pObjectList;
                    m_pObjectList = pObjectRef;
                    pObjectRef->m_pObject = pObject;
                    pObject->AddRef();
                }
            }
        }
        hr = pObject->QueryInterface(riid, ppv);
    }

    if (SUCCEEDED(hr) && riid == IID_IDirectMusicSegment8 && mt)
    {
        IDirectMusicSegment *pSegment = (IDirectMusicSegment8 *)(*ppv);

        hr = pSegment->SetLength(mt);
    }

    RELEASE(pObject);

    return hr;
}

//////////////////////////////////////////////////////////////////////////////
//
// C_DMLoader::SetSearchDirectory
//
// We only implement this by convention for the example. If you have another
// method of locating your content (for example, the filename indicates which
// file system the content is embedded in) you can use that instead.

STDMETHODIMP C_DMLoader::SetSearchDirectory(REFCLSID rguidClass, WCHAR *pwzPath, BOOL fClear)
{
    wcscpy(m_wzSearchPath, pwzPath);

    return S_OK;
}

//----------------------------------------------------------------------------
//
// Methods you don't need to implement.

STDMETHODIMP C_DMLoader::SetObject(LPDMUS_OBJECTDESC pDESC)
{
    return E_NOTIMPL;
}

STDMETHODIMP C_DMLoader::ScanDirectory(REFCLSID rguidClass, WCHAR *pszFileExtension, WCHAR *pszCacheFileName)
{
    return E_NOTIMPL;
}

STDMETHODIMP C_DMLoader::CacheObject(IDirectMusicObject *pObject)
{
    return E_NOTIMPL;
}

STDMETHODIMP C_DMLoader::ReleaseObject(IDirectMusicObject *pObject)
{
    return E_NOTIMPL;
}

STDMETHODIMP C_DMLoader::ClearCache(REFCLSID rguidClass)
{
    return E_NOTIMPL;
}

STDMETHODIMP C_DMLoader::EnableCache(REFCLSID rguidClass, BOOL fEnable)
{
    return E_NOTIMPL;
}

STDMETHODIMP C_DMLoader::EnumObject(REFCLSID rguidClass, DWORD dwIndex, LPDMUS_OBJECTDESC pDESC)
{
    return E_NOTIMPL;
}

//////////////////////////////////////////////////////////////////////////////
//
// MusicTimeFromWav
//
// Determine how long a WAV file is in music time, based on the default
// tempo of 120 bpm.
//
// When a WAV file is loaded as a segment, the segment gets
// a music time length of 1. If the WAV file is converted to a segment in
// DirectMusic Producer first, the music time will be correct; however, this
// isn't always feasible if you have a lot of legacy content. If you are trying
// to queue segments back-to-back, you need to have the length of the segment
// available.
//
// This function is a workaround for that problem.
//
// This is basically a quick RIFF parser that determines the length in samples
// of the wave data, and then converts that to music time based on a tempo
// of 120 bpm. If you need more resolution than you can get out of 120 bpm
// (about 0.65 ms per tick) you can increase the tempo.
//
// If parsing a wave file each time you load it doesn't appeal to you, you
// could store this information elsewhere and have the loader look it up
// by filename.

MUSIC_TIME MusicTimeFromWav(IStream *pStream)
{
#pragma pack(1)
    struct
    {
        struct
        {
            DWORD dwType;
            DWORD cb;
        } riffChunkHeader;
        DWORD dwSubType;
    } riffHeader;
#pragma pack()

    LARGE_INTEGER liZero;
    ULARGE_INTEGER uliStart;
    unsigned long length;
    ULONG cbRead;
    char msg[MSG_LEN];

    liZero.QuadPart = 0;
    pStream->Seek(liZero, STREAM_SEEK_CUR, &uliStart);

    pStream->Read(&riffHeader, sizeof(riffHeader), &cbRead);

    bool fGotHeader = false;
    bool fGotFact = false;
    bool fGotData = false;

    PCMWAVEFORMAT wf = { 0 };
    DWORD dwSamplesFromFact = 0;
    DWORD cbData = 0;

    for (;;)
    {
        if (FAILED(pStream->Read(&riffHeader, sizeof(riffHeader.riffChunkHeader), &cbRead))
            || cbRead != sizeof(riffHeader.riffChunkHeader))
        {
            break;
        }

        ULARGE_INTEGER uli;

        pStream->Seek(liZero, STREAM_SEEK_CUR, &uli);

        switch (riffHeader.riffChunkHeader.dwType)
        {
        case ' tmf':
            pStream->Read(&wf, sizeof(wf), &cbRead);
            fGotHeader = true;
            break;

        case 'tcaf':
            pStream->Read(&dwSamplesFromFact, sizeof(dwSamplesFromFact), &cbRead);
            fGotFact = true;
            break;

        case 'atad':
            cbData = riffHeader.riffChunkHeader.cb;
            fGotData = true;
            break;
        }

        if (fGotHeader)
        {
            if (wf.wf.wFormatTag == WAVE_FORMAT_PCM)
            {
                if (fGotData)
                {
                    break;
                }
            }
            else
            {
                if (fGotFact)
                {
                    break;
                }
            }
        }

        LARGE_INTEGER li;

        li.QuadPart = uli.QuadPart + (riffHeader.riffChunkHeader.cb + 1) & ~1;
        pStream->Seek(li, STREAM_SEEK_SET, NULL);
    }

    if (!fGotHeader)
    {
        // Not a wave file
        MessageBox(NULL, "Not a WAV file!", "", MB_OK);

        //
        return 1;
    }

    LARGE_INTEGER liRestore;

    liRestore.QuadPart = uliStart.QuadPart;
    pStream->Seek(liRestore, STREAM_SEEK_SET, NULL);

    DWORD ticks = DMUS_PPQ;

    if (wf.wf.wFormatTag == WAVE_FORMAT_PCM)
    {
        if (!fGotData)
        {
            // Corrupt: no data
            MessageBox(NULL, "Corrupt, no data!", "", MB_OK);
            //
            return 1;
        }

        DWORD cbSamples;
        if (wf.wf.nChannels * wf.wBitsPerSample > 0)
        {
            cbSamples = cbData / (wf.wf.nChannels * (wf.wBitsPerSample / 8));
        }
        else
        {
            cbSamples = 0;
            sprintf(msg, "nChannels or wBitsPerSample == 0, invalid audio data!!!");
            AddLogMsg(msg);
        }

        // At 120 bpm (2 beats per second) and 768 music-time ticks per beat,
        // we have 1536 ticks per second.
        //
        length = (DWORD)(((double)cbSamples * 2 * ticks) / (double)wf.wf.nSamplesPerSec);

        sprintf(msg, "nChannels = %d, wBitsPerSample = %d, cbData = %ld, cbSamples = %ld, nSamplesPerSec = %ld, length = %ld",
                wf.wf.nChannels, wf.wBitsPerSample,
                cbData, cbSamples, wf.wf.nSamplesPerSec, length);
        AddLogMsg(msg);

        return length;
    }

    // Compressed format.
    //
    if (!fGotFact)
    {
        // Compressed wave has no FACT chunk.
        AddLogMsg("Compressed file!");
        //
        return 1;
    }
    length = (dwSamplesFromFact * 2 * ticks) / wf.wf.nSamplesPerSec;

    sprintf(msg, "nChannels = %d, wBitsPerSample = %d, cbData = %ld, nSamplesPerSec = %ld, length = %ld",
            wf.wf.nChannels, wf.wBitsPerSample,
            cbData, wf.wf.nSamplesPerSec, length);
    AddLogMsg(msg);

    return length;
}
