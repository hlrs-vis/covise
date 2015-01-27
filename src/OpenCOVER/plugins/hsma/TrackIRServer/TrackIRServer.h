/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:   TrackIRServer\TrackIRServer.h
//
// author:   Peter Gehrt p.gehrt@hs-mannheim.de
//         Hochschule Mannheim - Virtual Reality Center
//
//requirements: optitrackuuid.lib (OptiTrack SDK) and WS2_32.lib
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <atlbase.h> // ATL COM library for camera connection points
#include <atlcom.h>
#include <conio.h>
#include <optitrack.h>
#import <optitrack.tlb>
#include <Winsock2.h>

#define CALL(x) _cprintf("%s: " #x "\n", ((x) == S_OK) ? "passed" : "FAILED");

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Opti track client module and ATL initialization </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
class COptiTrackClientModule : public CAtlExeModuleT<COptiTrackClientModule>
{
public:
    HRESULT PreMessageLoop(int) throw();
    HRESULT PostMessageLoop() throw();
    inline static HRESULT InitializeCom() throw()
    {
        return CoInitialize(NULL);
    }
    inline static HRESULT UnInitializeCom() throw()
    {
        CoUninitialize();
        return S_OK;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Camera collection events. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
class CameraCollectionEvents : public IDispEventImpl<0, CameraCollectionEvents, &DIID__INPCameraCollectionEvents, &LIBID_OptiTrack, 1, 0>
{
public:
    STDMETHOD_(void, OnDeviceRemoval)(INPCamera *pCamera)
    {
        HandleDeviceRemoval(pCamera);
    }
    void HandleDeviceRemoval(INPCamera *pCamera)
    {
        _cprintf("Camera removed\n");
    }
    BEGIN_SINK_MAP(CameraCollectionEvents)
    SINK_ENTRY_EX(0, DIID__INPCameraCollectionEvents, 1, OnDeviceRemoval)
    END_SINK_MAP()
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Camera events. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
class CameraEvents : public IDispEventImpl<0, CameraEvents, &DIID__INPCameraEvents, &LIBID_OptiTrack, 1, 0>
{
public:
    STDMETHOD_(void, OnFrameAvailable)(INPCamera *pCamera)
    {
        HandleFrameAvailable(pCamera);
    }
    void HandleFrameAvailable(INPCamera *pCamera);
    BEGIN_SINK_MAP(CameraEvents)
    SINK_ENTRY_EX(0, DIID__INPCameraEvents, 1, OnFrameAvailable)
    END_SINK_MAP()
};

////////////////////////////////////////////////////////////////////////////////////////////////////
CameraCollectionEvents collectionEvents;
CameraEvents cameraEvents;
CComPtr<INPCamera> camera;
CComPtr<INPCameraCollection> collection;
CComPtr<INPVector> vector;
CComPtr<INPSmoothing> smoothingYaw;
CComPtr<INPSmoothing> smoothingPitch;
CComPtr<INPSmoothing> smoothingRoll;
CComPtr<INPSmoothing> smoothingX;
CComPtr<INPSmoothing> smoothingY;
CComPtr<INPSmoothing> smoothingZ;
WSADATA wsa;
VARIANT varX, varY, varZ, varYaw, varPitch, varRoll, varEmpty, voption, deltaX, deltaY, deltaZ, deltaYaw, deltaPitch, deltaRoll;

int mySocket;
struct sockaddr_in addr;
struct hostent *host;
int counter;
unsigned int ausgabeCounter = 1;
bool flag;

COptiTrackClientModule theModule;
extern "C" int WINAPI WinMain(HINSTANCE, HINSTANCE, LPTSTR, int nShowCmd)
{
    return theModule.WinMain(nShowCmd);
}