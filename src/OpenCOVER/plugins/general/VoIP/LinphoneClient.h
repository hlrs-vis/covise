/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LINPHONECLIENT_H
#define _LINPHONECLIENT_H

#include <string>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <functional>

#include <linphone/core.h>

// ----------------------------------------------------------------------------
// enum class LinphoneClientState
// ----------------------------------------------------------------------------
enum class LinphoneClientState
{
    off,             // init state 
    halted,
    started,
    configured,
    offline,
    registering,
    registered,
    callIncoming,
    callInit,
    callInProgress,
    callRinging,
    callFailed,
    callConnected,
    callStreaming,
    callEnded
};

// ----------------------------------------------------------------------------
// class LinphoneClient
// ----------------------------------------------------------------------------
class LinphoneClient
{
public:

    //! constructor, set global callbacks, init state
    LinphoneClient();

    //! destructor, wait for thread to terminate, release linphone core
    ~LinphoneClient();

    //! sets the callback handler
    void addHandler(std::function<void (LinphoneClientState, LinphoneClientState)>* handler);
    
    //! create new thread, start core iterator
    void startCoreIterator();

    //! initiate registration
    bool doRegistration(std::string sipaddress, std::string password);

    //! initiate unregistration
    void doUnregistration();
    
    //! check if audio is enabled within a call
    bool callAudioIsEnabled();
    
    //! check if video is enabled within a call
    bool callVideoIsEnabled();
    
    //! check if camera stream is allowed to be sent within a call
    bool callCameraIsEnabled();

    //! gets the name of the currently assigned sound device for playback
    const char* getPlaybackDeviceName();

    //! gets the list of the available sound devices which can capture
    std::vector<std::string> getCaptureSoundDevicesList();

    //! gets the list of the available sound devices which can playback sound
    std::vector<std::string> getPlaybackSoundDevicesList();

    //! gets the name of the currently capture sound device
    std::string getCurrentCaptureSoundDevice();
    
    //! gets the name of the currently playback sound device
    std::string getCurrentPlaybackSoundDevice();
    
    //! gets the name of the currently ringer sound device
    std::string getCurrentRingerSoundDevice();
    
    //! gets the name of the currently media sound device
    //std::string getCurrentMediaSoundDevice();

    //! tells wether microphone is enabled or not
    bool getMicrophoneIsEnabled();
    
    //! check whether the device has a hardware echo canceller
    //bool getEchoCancellerAvailable();

    //! returns true if echo cancellation is enabled
    bool getEchoCancellationIsEnabled();

    //! returns true if echo limiter is enabled
    bool getEchoLimiterIsEnabled();

    //! tells whether the audio adaptive jitter compensation is enabled
    bool getAudioJitterCompensation();
    
    //! get microphone gain in db
    float getMicGain();

    //! get playback gain in db before entering sound card
    float getPlaybackGain();

    //! tells whether video capture is enabled
    bool getVideoCaptureEnabled();

    //! tells whether video display is enabled
    bool getVideoDisplayEnabled();
    
    //! tells whether video preview is enabled
    bool getVideoPreviewEnabled();
    
    //! get the default policy for acceptance of incoming video
    bool getAutoAcceptVideo();
    
    //! get the default policy for initiating video
    bool getAutoInitiateVideo();
    
    //! tells whether the video adaptive jitter compensation is enabled
    bool getVideoJitterCompensation();

    //! gets a list of the available video capture devices
    std::vector<std::string> getVideoCaptureDevicesList();

    //! returns the name of the currently active video device    
    std::string getCurrentVideoCaptureDevice();
    
    //! get microphone muted state
    bool getCallMicrophoneMuted();
    
    //! get speaker muted state
    bool getCallSpeakerMuted();
    
    //! self view during call is enabled
    bool getCallSelfViewEnabled();
    
    //! camera stream are allowed to be send to remote
    bool getCallCameraEnabled();

    //! easier readability for debugging purposes
    std::string getStateString(LinphoneClientState);

    //! initiate call to given sip address, keep valid reference to call ptr
    bool initiateCall(std::string);
    
protected:

    std::vector< std::function<void (LinphoneClientState, LinphoneClientState)> * > handlerList;
    
    LinphoneCore *lc = NULL;
    LinphoneCall *call = NULL;
    
    LinphoneCoreVTable vtable = {0};
    LinphoneProxyConfig* proxy_cfg = NULL;
    LinphoneAddress *from = NULL;
    LinphoneAuthInfo *info = NULL;

    const char* identity;
    const char* password;
    const char* server_addr;
    
    volatile bool stopThread = false;
    std::thread thdFunction;

    bool oneTime = false;  // TODO: remove

    const unsigned int timeout = 10; // * intervall ms
    const unsigned int intervall = 500; // time intervall core iterator polls
    
    //! linphoneclient main loop
    void thdMain();
};

// ----------------------------------------------------------------------------

#endif
