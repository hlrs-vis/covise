/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: dllmain.cpp

	DESCRIPTION:   DLL implementation of primitives

	CREATED BY: Charles Thaeler

        BASED on helpers.cpp

	HISTORY: created 12 February 1996

 *>	Copyright (c) 1994, All Rights Reserved.
 **********************************************************************/

#include "coTabletUI.h"
#include "vrml.h"

#include <iparamb2.h>
//extern ClassDesc* GetMrBlueDesc();
extern ClassDesc *GetLODDesc();
extern ClassDesc *GetVRBLDesc();
extern ClassDesc *GetVRMLCOVISEObjectDesc();
extern ClassDesc *GetVRMLInsertDesc();
extern ClassDesc *GetVRMLMtlDesc();
extern ClassDesc *GetOmniLightDesc();
extern ClassDesc *GetTSpotLightDesc();
extern ClassDesc *GetDirLightDesc();
extern ClassDesc *GetFSpotLightDesc();
extern ClassDesc *GetPolyCounterDesc();
extern ClassDesc *GetTimeSensorDesc();
extern ClassDesc *GetNavInfoDesc();
extern ClassDesc *GetBackgroundDesc();
extern ClassDesc *GetFogDesc();
extern ClassDesc *GetScriptDesc();
extern ClassDesc *GetSkyDesc();
extern ClassDesc *GetAudioClipDesc();
extern ClassDesc *GetSoundDesc();
extern ClassDesc *GetTouchSensorDesc();
extern ClassDesc *GetSwitchDesc();
extern ClassDesc *GetProxSensorDesc();
extern ClassDesc *GetAnchorDesc();
extern ClassDesc *GetBillboardDesc();
extern ClassDesc *GetARSensorDesc();
extern ClassDesc *GetCOVERDesc();
extern ClassDesc *GetCppOutDesc();
extern ClassDesc *GetOnOffSwitchDesc();
extern ClassDesc *GetTabletUIDesc();
extern ClassDesc *GetResetPivotDesc();
extern ClassDesc *GetMultiTouchSensorDesc();
extern ClassDesc *GetCal3DDesc();

HINSTANCE hInstance;
int controlsInit = FALSE;

TCHAR
*GetString(int id)
{
    static TCHAR buf[256];

    if (hInstance)
        return LoadString(hInstance, id, buf, sizeof(buf)) ? buf : NULL;
    return NULL;
}

/** public functions **/
BOOL WINAPI
    DllMain(HINSTANCE hinstDLL, ULONG fdwReason, LPVOID lpvReserved)
{
    hInstance = hinstDLL;

    if (!controlsInit)
    {
        controlsInit = TRUE;
#if (MAX_RELEASE < MAX_RELEASE_R14)
        // jaguar controls
        InitCustomControls(hInstance);

        // initialize Chicago controls
        InitCommonControls();
#endif

        SetTimer(NULL, 0, 5000, coTabletUI::timerCallback);
    }

    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hInstance);
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        break;
    }
    return (TRUE);
}

//------------------------------------------------------
// This is the interface to MAX:
//------------------------------------------------------

__declspec(dllexport) const TCHAR *LibDescription()
{
    return GetString(IDS_LIBDESCRIPTION);
}

#ifndef NO_UTILITY_POLYGONCOUNTER // russom - 12/04/01
#define NUM_BASE_CLASSES 25
#else
#define NUM_BASE_CLASSES 24
#endif

#ifdef _DEBUG
#define NUM_CLASSES (NUM_BASE_CLASSES + 1)
#else
#define NUM_CLASSES NUM_BASE_CLASSES
#endif

/// MUST CHANGE THIS NUMBER WHEN ADD NEW CLASS
__declspec(dllexport) int LibNumberClasses()
{
    return NUM_CLASSES;
}

__declspec(dllexport) ClassDesc *LibClassDesc(int i)
{
    switch (i)
    {
    case 0:
        return GetAnchorDesc();
    case 1:
        return GetTouchSensorDesc();
    case 2:
        return GetProxSensorDesc();
    case 3:
        return GetTimeSensorDesc();
    case 4:
        return GetNavInfoDesc();
    case 5:
        return GetBackgroundDesc();
    case 6:
        return GetFogDesc();
    case 7:
        return GetAudioClipDesc();
    case 8:
        return GetSoundDesc();
    case 9:
        return GetBillboardDesc();
    case 10:
        return GetLODDesc();
    case 11:
        return GetVRBLDesc();
    case 12:
        return GetVRMLInsertDesc();
    case 13:
        return GetARSensorDesc();
    case 14:
        return GetCOVERDesc();
    case 15:
        return GetSwitchDesc();
    case 16:
        return GetOnOffSwitchDesc();
    case 17:
        return GetTabletUIDesc();
    case 18:
        return GetResetPivotDesc();
    case 19:
        return GetSkyDesc();
    case 20:
        return GetScriptDesc();
    case 21:
        return GetMultiTouchSensorDesc();
    case 22:
        return GetCal3DDesc();
    case 23:
        return GetVRMLCOVISEObjectDesc();
#ifndef NO_UTILITY_POLYGONCOUNTER // russom - 12/04/01
    case 24:
        return GetPolyCounterDesc();
#endif
//case 15: return GetMrBlueDesc();
#ifdef _DEBUG
    case NUM_BASE_CLASSES:
        return GetCppOutDesc();
#endif * /
    default:
        return 0;
    }
}

// Return version so can detect obsolete DLLs -- NOTE THIS IS THE API VERSION NUMBER
//                                               NOT THE VERSION OF THE DLL.
__declspec(dllexport) ULONG
    LibVersion() { return VERSION_3DSMAX; }

// Let the plug-in register itself for deferred loading
__declspec(dllexport) ULONG CanAutoDefer()
{
    return 1;
}
