/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:   TrackIRServer\TrackIRServer.cpp
//
// summary:    See attached readme
//
// author:   Peter Gehrt p.gehrt@hs-mannheim.de
//         Virtual Reality Center - Hochschule Mannheim - University of Applied Sciences, 2010
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "TrackIRServer.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Initialize network socket and connection to TrackIRPlugin. </summary>
///
/// <returns>   S_OK or ERROR </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
HRESULT startNetwork()
{
    flag = true;
    mySocket = socket(PF_INET, SOCK_STREAM, 0);
    if (mySocket == -1)
        _cprintf("socket() failed");
    host = gethostbyname("localhost");
    if (!host)
        _cprintf("gethostbyname() failed");
    addr.sin_addr = *(struct in_addr *)host->h_addr;
    addr.sin_port = htons(7050); //ssh port, for debug use only!
    addr.sin_family = AF_INET; // UDP connection
    // waiting for connection
    while (connect(mySocket, (struct sockaddr *)&addr, sizeof(addr)) == -1)
    {
        _cprintf("Please start TrackIRPlugin \r");
    }
    _cprintf("connected\n");
    return S_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Stop network and close socket.</summary>
///
/// <returns>   S_OK or ERROR </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
HRESULT stopNetwork()
{
    closesocket(mySocket);
    return S_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Send tracking data to open socket. </summary>
///
/// <param name="sock">      The socket. </param>
/// <param name="request">   The request. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
void send_TrackingData(const int sock, const char *request)
{
    if (send(mySocket, request, strlen(request), 0) <= 0)
    {
        if (flag) //if failed, try to reconnect
        {
            _cprintf("\n\nConnection lost... restarting Server\n");
            flag = false;
            stopNetwork();
            Sleep(1000);
            startNetwork();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Pre message loop starts server</summary>
///
/// <param name="">   . </param>
///
/// <returns>   . </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

HRESULT COptiTrackClientModule::PreMessageLoop(int)
{
    flag = true;
    AllocConsole();
    vector.CoCreateInstance(CLSID_NPVector);
    collection.CoCreateInstance(CLSID_NPCameraCollection);
    collectionEvents.DispEventAdvise(collection);
    _cprintf("Starting TrackIR Server ...\n");
    // get all cameras and open the first one
    collection->Enum();
    collection->Item(0, &camera);
    cameraEvents.DispEventAdvise(camera);
    camera->Open();
    camera->Start();
    // get some camera information
    long serial, width, height, model, revision, rate;
    camera->get_SerialNumber(&serial);
    camera->get_Width(&width);
    camera->get_Height(&height);
    camera->get_Model(&model);
    camera->get_Revision(&revision);
    camera->get_FrameRate(&rate);
    _cprintf("Kamera:%d Modell:0x%8x Auflösung:%dx%d Revision:0x%8x Wiederholrate:%d\n", serial, model, width, height, revision, rate);
    // initialize windows sockets and network
    if (WSAStartup(MAKEWORD(1, 1), &wsa))
        return EXIT_FAILURE;
    return startNetwork();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Post message loop stops TrackIRServer and close socket. </summary>
///
/// <returns>   S_OK or ERROR. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

HRESULT COptiTrackClientModule::PostMessageLoop()
{
    CALL(camera->Stop());
    CALL(camera->Close());
    CALL(cameraEvents.DispEventUnadvise(camera));
    CALL(collectionEvents.DispEventUnadvise(collection));
    camera.Release();
    collection.Release();
    vector.Release();
    FreeConsole();
    return stopNetwork();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Handle frame available,  calculate vectors and send new trackingdata to TrackIRPlugin. </summary>
///
/// <param name="pCamera">   [in,out] If not null, the camera. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
void CameraEvents::HandleFrameAvailable(INPCamera *pCamera)
{
    CComPtr<INPCameraFrame> frame;
    // timeout: 5ms
    pCamera->GetFrame(2, &frame);
    vector->Update(pCamera, frame);
    // free frame as soon as possible!
    frame->Free();
    // extract 6DOF values
    vector->get_Yaw(&varYaw);
    vector->get_Pitch(&varPitch);
    vector->get_Roll(&varRoll);
    vector->get_X(&varX);
    vector->get_Y(&varY);
    vector->get_Z(&varZ);
    // smooth the values
    VariantInit(&varEmpty);
    smoothingYaw.CoCreateInstance(CLSID_NPSmoothing);
    smoothingPitch.CoCreateInstance(CLSID_NPSmoothing);
    smoothingRoll.CoCreateInstance(CLSID_NPSmoothing);
    smoothingX.CoCreateInstance(CLSID_NPSmoothing);
    smoothingY.CoCreateInstance(CLSID_NPSmoothing);
    smoothingZ.CoCreateInstance(CLSID_NPSmoothing);
    smoothingYaw->Update(varYaw, varEmpty);
    smoothingPitch->Update(varPitch, varEmpty);
    smoothingRoll->Update(varRoll, varEmpty);
    smoothingX->Update(varX, varEmpty);
    smoothingY->Update(varY, varEmpty);
    smoothingZ->Update(varZ, varEmpty);
    // get the smoothed data
    smoothingYaw->get_X(&varYaw);
    smoothingPitch->get_X(&varPitch);
    smoothingRoll->get_X(&varRoll);
    smoothingX->get_X(&varX);
    smoothingY->get_X(&varY);
    smoothingZ->get_X(&varZ);
    //debug output every 500 frames
    if (!(ausgabeCounter++ % 500))
        _cprintf("x=%.1f  y=%.1f  z=%.1f   yaw=%.1f  pitch=%.1f  roll=%.1f \n", varX.dblVal, varY.dblVal, varZ.dblVal, varYaw.dblVal, varPitch.dblVal, varRoll.dblVal);
    char tracking_data[100];
    sprintf(tracking_data, "%.2f;%.2f;%.2f;%.2f;%.2f;%.2f;$$$\n", varX.dblVal, varY.dblVal, varZ.dblVal, varYaw.dblVal, varPitch.dblVal, varRoll.dblVal);
    send_TrackingData(mySocket, tracking_data);
}