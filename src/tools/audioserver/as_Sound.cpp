/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "as_Sound.h"
#include "common.h"
#include "loader.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <dmusici.h>
#include "resource.h"

const double pi = 3.1415926535897932;

as_Sound::as_Sound(char *fileName)
{
    long rc = 0;
    long len = 0;
    HRESULT hr;
    WCHAR wChar[MAX_PATH];

    if (NULL == fileName)
        return;

    pSegment = NULL;
    pState = NULL;
    pDS3DBuffer = NULL;
    pAudioPath = NULL;
    memset(&dmod, 0, sizeof(DMUS_OBJECTDESC));
    gain = 1.0f;
    Direction[0] = 0.0f;
    Direction[1] = 0.0f;
    Direction[2] = 0.0f;
    Position[0] = 0.0f;
    Position[1] = 0.0f;
    Position[2] = 0.0f;

    // Create a 3D audiopath with a 3d buffer.
    // We can then play all segments into this buffer and directly control its 3D parameters.
    hr = g_pPerformance->CreateStandardAudioPath(
        DMUS_APATH_DYNAMIC_3D,
        1,
        TRUE,
        &(this->pAudioPath));
    if (FAILED(hr))
    {
        char msg[MSG_LEN];
        sprintf(msg, "Could not create standard audio path, error 0x%08lX", hr);
        AddLogMsg(msg);
        return;
    }

    // Get the 3D buffer in the audio path.
    hr = this->pAudioPath->GetObjectInPath(
        0,
        DMUS_PATH_BUFFER,
        0,
        GUID_NULL,
        0,
        IID_IDirectSound3DBuffer,
        (LPVOID *)&this->pDS3DBuffer);
    if (FAILED(hr))
    {
        AddLogMsg("Error: GetObjectInPath");
        return;
    }
    this->pAudioPath->SetVolume(0, 0);

    dmod.dwSize = sizeof(DMUS_OBJECTDESC);
    dmod.dwValidData = DMUS_OBJ_CLASS | DMUS_OBJ_FILENAME | DMUS_OBJ_FULLPATH;
    dmod.guidClass = CLSID_DirectMusicSegment;

    len = strlen(fileName);
    rc = mbstowcs(wChar, fileName, len + 1);
    wcscpy(dmod.wszFileName, wChar);

    hr = g_Loader->GetObject(&dmod, IID_IDirectMusicSegment8, (void **)&pSegment);
    if (FAILED(hr))
    {
        AddLogMsg("Could not get object: IID_IDirectMusicSegment8");
        return;
    }
    hr = pSegment->Download(g_pPerformance);
    if (FAILED(hr))
    {
        AddLogMsg("Could not download segment");
        return;
    }
    this->status = STATUS_INITIAL;
}

as_Sound::~as_Sound(void)
{
    RELEASE(pSegment);
    memset(&dmod, 0, sizeof(dmod));
    pSegment = NULL;
    pState = NULL;
    status = -1;
}

void as_Sound::Play()
{
    HRESULT hr;
    hr = g_pPerformance->PlaySegmentEx(
        pSegment, // Segment to play.
        NULL, // Used for songs; not implemented.
        NULL, // For transitions.
        DMUS_SEGF_REFTIME | DMUS_SEGF_SECONDARY, // Flags.
        0, // Start time; 0 is immediate.
        &(this->pState), // Pointer that receives segment state.
        NULL, // Object to stop.
        pAudioPath // Audiopath.
        );
    if (FAILED(hr))
    {
        char msg[MSG_LEN];
        sprintf(msg, "Could not play segment, error 0x%08lX", hr);
        AddLogMsg(msg);
        return;
    }
    AddLogMsg("Play");
    hr = pSegment->AddNotificationType(GUID_NOTIFICATION_SEGMENT);
    this->playing = true;
    this->status = STATUS_PLAYING;
}

void as_Sound::Stop()
{
    HRESULT hr;
    hr = g_pPerformance->Stop(pSegment, NULL, 0, 0);
    if (FAILED(hr))
    {
        char msg[MSG_LEN];
        sprintf(msg, "Could not stop segment, error 0x%08lX", hr);
        AddLogMsg(msg);
        return;
    }
    AddLogMsg("Stop");
    this->status = STATUS_STOPPED;
    this->playing = false;
}

void as_Sound::Play(float starttime)
{
    HRESULT hr;

    MUSIC_TIME mtLength;
    MUSIC_TIME mtStart = TimeToMusicTime(starttime);
    char msg[MSG_LEN];

    hr = pSegment->GetLength(&mtLength);

    hr = pSegment->AddNotificationType(GUID_NOTIFICATION_SEGMENT);

    if (true == looping)
    {
        hr = pSegment->SetRepeats(DMUS_SEG_REPEAT_INFINITE);
    }
    else
    {
        hr = pSegment->SetRepeats(0);
    }
    if (S_OK != hr)
    {
        char msg[128];
        sprintf(msg, "SetRepeats error 0x%08lX", hr);
        AddLogMsg(msg);
    }

    hr = pSegment->SetStartPoint(mtStart);
    if (S_OK != hr)
    {
        char msg[128];
        sprintf(msg, "SetStartPoint error 0x%08lX", hr);
        AddLogMsg(msg);
        hr = S_OK;
    }
    /*
      hr = pSegment->SetLoopPoints(mtStart, mtStop);
      if (S_OK != hr) {
         char msg[128];
         sprintf(msg, "SetLoopPoints error 0x%08lX", hr);
         AddLogMsg(msg);
      }
   */
    // Start the first segment going.
    //
    if (SUCCEEDED(hr))
    {
        /*
        hr = g_pPerformance->PlaySegment(pSegment, 0, 0, &pState);
      if (S_OK != hr) {
         char msg[128];
         sprintf(msg, "PlaySegment error 0x%08lX", hr);
      AddLogMsg(msg);
      }
      */
        hr = g_pPerformance->PlaySegmentEx(
            pSegment, // Segment to play.
            NULL, // Used for songs; not implemented.
            NULL, // For transitions.
            DMUS_SEGF_REFTIME | DMUS_SEGF_SECONDARY, // Flags.
            0, // Start time; 0 is immediate.
            &(this->pState), // Pointer that receives segment state.
            NULL, // Object to stop.
            pAudioPath // Audiopath.
            );
        if (FAILED(hr))
        {
            char msg[MSG_LEN];
            sprintf(msg, "Could not play segment, error 0x%08lX", hr);
            AddLogMsg(msg);
            return;
        }
        sprintf(msg, "Play(starttime = %f)", starttime);
        AddLogMsg(msg);
        this->playing = true;
        this->status = STATUS_PLAYING;
    }
}

void as_Sound::Play(float starttime, float stoptime)
{
    HRESULT hr;

    MUSIC_TIME mtLength;
    MUSIC_TIME mtStart = TimeToMusicTime(starttime);
    MUSIC_TIME mtStop = TimeToMusicTime(stoptime);
    REFERENCE_TIME rtTime;

    hr = pSegment->GetLength(&mtLength);

    hr = g_pPerformance->MusicToReferenceTime(
        mtStop,
        &rtTime);
    if (S_OK != hr)
    {
        char msg[128];
        sprintf(msg, "MusicToReferenceTime error 0x%08lX", hr);
        AddLogMsg(msg);
    }

    hr = pSegment->AddNotificationType(GUID_NOTIFICATION_SEGMENT);

    if (true == looping)
    {
        hr = pSegment->SetRepeats(DMUS_SEG_REPEAT_INFINITE);
    }
    else
    {
        hr = pSegment->SetRepeats(0);
    }
    if (S_OK != hr)
    {
        char msg[128];
        sprintf(msg, "SetRepeats error 0x%08lX", hr);
        AddLogMsg(msg);
    }

    hr = pSegment->SetStartPoint(mtStart);
    if (S_OK != hr)
    {
        char msg[128];
        sprintf(msg, "SetStartPoint error 0x%08lX", hr);
        AddLogMsg(msg);
        hr = S_OK;
    }
    /*
      hr = pSegment->SetLoopPoints(mtStart, mtStop);
      if (S_OK != hr) {
         char msg[128];
         sprintf(msg, "SetLoopPoints error 0x%08lX", hr);
         AddLogMsg(msg);
      }
   */
    // Start the segment.
    //
    if (SUCCEEDED(hr))
    {
        char msg[MSG_LEN];
        /*
              hr = g_pPerformance->PlaySegment(pSegment, 0, 0, &pState);
            if (S_OK != hr) {
               char msg[128];
               sprintf(msg, "PlaySegment error 0x%08lX", hr);
               AddLogMsg(msg);
            }
      */
        hr = g_pPerformance->PlaySegmentEx(
            pSegment, // Segment to play.
            NULL, // Used for songs; not implemented.
            NULL, // For transitions.
            DMUS_SEGF_REFTIME | DMUS_SEGF_SECONDARY, // Flags.
            0, // Start time; 0 is immediate.
            &(this->pState), // Pointer that receives segment state.
            NULL, // Object to stop.
            pAudioPath // Audiopath.
            );
        if (FAILED(hr))
        {
            char msg[MSG_LEN];
            sprintf(msg, "Could not play segment, error 0x%08lX", hr);
            AddLogMsg(msg);
            return;
        }

        // Stop the segment.
        //
        /*
              hr = g_pPerformance->Stop(pSegment, 0, mtStop, DMUS_SEGF_REFTIME);
            if (S_OK != hr) {
               char msg[128];
               sprintf(msg, "Stop error 0x%08lX", hr);
               AddLogMsg(msg);
            }
      */
        hr = g_pPerformance->StopEx(this->pSegment, rtTime, DMUS_SEGF_REFTIME);
        if (S_OK != hr)
        {
            char msg[128];
            sprintf(msg, "Stop error 0x%08lX", hr);
            AddLogMsg(msg);
        }
        this->playing = true;
        this->status = STATUS_PLAYING;
        sprintf(msg, "Play(starttime = %5.2f, stoptime = %5.2f)", starttime, stoptime);
        AddLogMsg(msg);
    }
}

void as_Sound::PlayLooping()
{
    looping = true;
    Play();
}

void as_Sound::SetPosition(float x, float y, float z)
{
    this->Position[0] = x;
    this->Position[1] = y;
    this->Position[2] = z;

    /* Add setting of AudioPath / 3D buffer parameters here */

    this->pDS3DBuffer->SetPosition(x, y, z, DS3D_IMMEDIATE);
}

void as_Sound::SetGain(float gain)
{
    // set volume in hundredth of Decibel from 0 (full volume) to -9600 (silence)
    long volume;
    HRESULT hr;

    if (gain > 1.0f)
        gain = 1.0f;

    //	volume = (long)(-(1.0f-gain)*9600.0f);		// wrong calculation

    // 1.0   -> -0 db
    // 0.5   -> -1 db
    // 0.25  -> -2 db
    // 0.125 -> -3 db

    if (0.0f == gain)
        volume = -9600;
    else
        volume = (long)(1000 * log(gain) / log(2.0));

#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "Volume = %ld (%5.2f)", volume, gain);
    AddLogMsg(msg);
#endif

    hr = this->pAudioPath->SetVolume(volume, 0);
    if (FAILED(hr))
    {
        char msg[MSG_LEN];
        sprintf(msg, "Could not set AudioPath volume, error 0x%08lX", hr);
        AddLogMsg(msg);
        return;
    }
}

void as_Sound::SetPitch(float pitch)
{
    // set volume in hundredth of Decibel from 0 (full volume) to -9600 (silence)
    long volume;
    HRESULT hr;

    // Get the 3D buffer in the audio path.
    IDirectSoundBuffer *myBuffer = NULL;
    hr = this->pAudioPath->GetObjectInPath(
        0,
        DMUS_PATH_BUFFER,
        0,
        GUID_NULL,
        0,
        IID_IDirectSoundBuffer,
        (LPVOID *)&myBuffer);
    if (FAILED(hr))
    {
        AddLogMsg("Error: GetObjectInPath");
        return;
    }
    //this->pDS3DBuffer->QueryInterface(IID_IDirectSoundBuffer,(LPVOID *)&myBuffer);
    if (myBuffer)
    {
        myBuffer->SetFrequency(pitch);
    }
    /*
   if (true == looping)
   {
	  
   MUSIC_TIME mtLength;
	   hr = pSegment->GetLength(&mtLength);
      hr = pSegment->SetLoopPoints(0, mtLength/44000*22000);
      if (S_OK != hr) {
         char msg[128];
         sprintf(msg, "SetLoopPoints error 0x%08lX", hr);
         AddLogMsg(msg);
      }
   
      this->pSegment->SetRepeats(DMUS_SEG_REPEAT_INFINITE);
   }
   else
      this->pSegment->SetRepeats(0);*/
}

void as_Sound::SetDirection(float x, float y, float z)
{
    this->Direction[0] = x;
    this->Direction[1] = y;
    this->Direction[2] = z;

    this->pDS3DBuffer->SetConeOrientation(x, y, z, DS3D_IMMEDIATE);
#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetDirection x=%3.2f, y=%3.2f, z=%3.2f", x, y, z);
    AddLogMsg(msg);
#endif
}

void as_Sound::SetDirection(float angle)
{
    float x, y, z;

    //	x = (float)sin(pi*angle/180);
    y = 0.0;
    //	z = (float)cos(pi*angle/180);
    x = (float)sin(angle);
    y = 0.0;
    z = (float)cos(angle);

    this->Direction[0] = x;
    this->Direction[1] = y;
    this->Direction[2] = z;

    this->pDS3DBuffer->SetConeOrientation(x, y, z, DS3D_IMMEDIATE);

#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetDirection angle = %f (x=%3.2f, y=%3.2f, z=%3.2f)", angle, x, y, z);
    AddLogMsg(msg);
#endif
}

void as_Sound::SetDirectionRelative(float angle)
{
    float x, y, z;

    //	x = (float)sin(pi*angle/180);
    //	z = (float)cos(pi*angle/180);
    x = (float)sin(angle);
    y = 0.0;
    z = (float)cos(angle);

    this->Position[0] = x;
    this->Position[1] = y;
    this->Position[2] = z;

    this->pDS3DBuffer->SetPosition(x, y, z, DS3D_IMMEDIATE);
#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetDirectionRelative angle = %f (x=%3.2f, y=%3.2f, z=%3.2f)", angle, x, y, z);
    AddLogMsg(msg);
#endif
}

void as_Sound::SetDirectionRelative(float x, float y, float z)
{
    this->Position[0] = x;
    this->Position[1] = y;
    this->Position[2] = z;

    this->pDS3DBuffer->SetPosition(x, y, z, DS3D_IMMEDIATE);
#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetDirectionRelative x=%3.2f, y=%3.2f, z=%3.2f", x, y, z);
    AddLogMsg(msg);
#endif
}

void as_Sound::SetLooping(bool enable)
{
    this->looping = enable;
    this->pSegment->SetLoopPoints(0, 0);
    if (true == enable)
        this->pSegment->SetRepeats(DMUS_SEG_REPEAT_INFINITE);
    else
        this->pSegment->SetRepeats(0);
#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetLooping %d", enable);
    AddLogMsg(msg);
#endif
}

void as_Sound::SetVelocity(float velocity)
{
    float x, y, z;
    D3DVECTOR vOrientation;
    this->pDS3DBuffer->GetConeOrientation(&vOrientation);
    x = vOrientation.x * velocity;
    y = vOrientation.y * velocity;
    z = vOrientation.z * velocity;
    this->pDS3DBuffer->SetVelocity(x, y, z, DS3D_IMMEDIATE);
}

void as_Sound::SetVelocity(float x, float y, float z)
{
    this->pDS3DBuffer->SetVelocity(x, y, z, DS3D_IMMEDIATE);
}

long as_Sound::GetStatus()
{
    return this->status;
}

void as_Sound::GetPosition(D3DVECTOR *position)
{
    position->x = this->Position[0];
    position->y = this->Position[1];
    position->z = this->Position[2];
}

void as_Sound::GetDirection(D3DVECTOR *direction)
{
    direction->x = this->Direction[0];
    direction->y = this->Direction[1];
    direction->z = this->Direction[2];
}

void *as_Sound::GetSegment()
{
    return this->pSegment;
}

void as_Sound::SetStatus(long newstatus)
{
    this->status = newstatus;
}

bool as_Sound::IsPlaying()
{
    return this->playing;
}

void as_Sound::SetDirectionRelative(float angle, long color)
{
    float x, y, z;

    //	x = (float)sin(pi*angle/180);
    y = 0.0;
    //	z = (float)cos(pi*angle/180);
    x = (float)sin(angle);
    y = 0.0;
    z = (float)cos(angle);

    this->Position[0] = x;
    this->Position[1] = y;
    this->Position[2] = z;

    this->pDS3DBuffer->SetPosition(x, y, z, DS3D_IMMEDIATE);
#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetDirectionRelative angle=%f (x=%3.2f, y=%3.2f, z=%3.2f), color=%08lXh",
            angle, x, y, z, color);
    AddLogMsg(msg);
#endif
    this->updateGridColored();
}

void as_Sound::SetDirectionRelative(float x, float y, float z, long color)
{
    this->Position[0] = x;
    this->Position[1] = y;
    this->Position[2] = z;

    this->pDS3DBuffer->SetPosition(x, y, z, DS3D_IMMEDIATE);
#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "SetDirectionRelative x=%3.2f, y=%3.2f, z=%3.2f, color=%08lXh",
            x, y, z, color);
    AddLogMsg(msg);
#endif
    this->updateGridColored();
}

void as_Sound::CommitDeferredSetings()
{
    IDirectSound3DListener8 *pListener3D;

    if (NULL != this->pAudioPath)
    {
        this->pAudioPath->GetObjectInPath(
            0,
            DMUS_PATH_PRIMARY_BUFFER,
            0,
            GUID_All_Objects,
            0,
            IID_IDirectSound3DListener,
            (void **)&pListener3D);

        pListener3D->CommitDeferredSettings();
    }
}

void as_Sound::updateGridColored()
{
    HWND hWndGrid;
    HDC hDC;
    RECT rc;

    //    static LONG s_lPixel[5] = { CLR_INVALID, CLR_INVALID, CLR_INVALID, CLR_INVALID, CLR_INVALID };
    static LONG s_lX = 0;
    static LONG s_lY = 0;
    COLORREF color = 0;
    COLORREF darkColor = 0;

    if (NULL == hWndGridView)
        return;

    hWndGrid = GetDlgItem(hWndGridView, IDC_GRID);
    hDC = GetDC(hWndGrid);

    // Don't update the grid if a WM_PAINT will be called soon
    BOOL bUpdateInProgress = GetUpdateRect(hWndGridView, NULL, FALSE);
    if (bUpdateInProgress)
    {
        return;
    }

    color = this->color;
    darkColor = RGB(GetRValue(color) * 0.5, GetGValue(color) * 0.5, GetBValue(color) * 0.5);

    s_lX = this->oldGridPosX;
    s_lY = this->oldGridPosY;

    // Draw a crosshair object darker than normal the trail
    SetPixel(hDC, s_lX - 1, s_lY + 0, darkColor);
    SetPixel(hDC, s_lX + 0, s_lY - 1, darkColor);
    SetPixel(hDC, s_lX + 0, s_lY + 0, darkColor);
    SetPixel(hDC, s_lX + 0, s_lY + 1, darkColor);
    SetPixel(hDC, s_lX + 1, s_lY + 0, darkColor);

    // Convert the world space x,y coordinates to pixel coordinates
    GetClientRect(hWndGrid, &rc);

    // attention: x, z are horizontal, y is vertical axis !!!

    s_lX = (LONG)(((rc.right - rc.left) / 2) + 50 * (this->Position[0]));
    s_lY = (LONG)(((rc.bottom - rc.top) / 2) - 50 * (this->Position[2]));

    // Draw a crosshair object in light pixels
    SetPixel(hDC, s_lX - 1, s_lY + 0, color);
    SetPixel(hDC, s_lX + 0, s_lY - 1, color);
    SetPixel(hDC, s_lX + 0, s_lY + 0, color);
    SetPixel(hDC, s_lX + 0, s_lY + 1, color);
    SetPixel(hDC, s_lX + 1, s_lY + 0, color);

    ReleaseDC(hWndGrid, hDC);

    this->oldGridPosX = s_lX;
    this->oldGridPosY = s_lY;
}

void as_Sound::SetColor(unsigned long color)
{
    this->color = color;
}

unsigned long as_Sound::GetColor()
{
    return this->color;
}
