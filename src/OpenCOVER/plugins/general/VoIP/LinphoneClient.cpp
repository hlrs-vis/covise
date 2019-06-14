/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "LinphoneClient.h"

#include <linphone/core.h>

#include <iostream>
#include <signal.h>
#include <thread>
#include <chrono>

using namespace std;

// ----------------------------------------------------------------------------

LinphoneClientState lp_state     = LinphoneClientState::off; // FIXME non global
LinphoneClientState lp_state_old = LinphoneClientState::off; // FIXME non global         

// ----------------------------------------------------------------------------
// linphone callback
// notifies global state changes
// ----------------------------------------------------------------------------
static void global_state_changed(struct _LinphoneCore *lc,
                                 LinphoneGlobalState gstate,
                                 const char* message)
{
#ifdef VOIP_DEBUG
    cout << "cb: global_state_changed  " << endl;
    cout << "    state : " << linphone_global_state_to_string(gstate) << endl;
    cout << "    msg : " << message << endl;
#endif

    switch (gstate)
    {
    case LinphoneGlobalShutdown:
    case LinphoneGlobalOff:
    {
        lp_state = LinphoneClientState::halted;
        break;
    }
    case LinphoneGlobalStartup:
    {
        lp_state = LinphoneClientState::started;
        break;
    }
    case LinphoneGlobalOn:
    {
        lp_state = LinphoneClientState::offline;
        break;
    }
    case LinphoneGlobalConfiguring:
    {
        lp_state = LinphoneClientState::configured;
        break;
    }
    default:
    {
        cout << "!!! info : unhandled notification "
             << linphone_global_state_to_string(gstate)
             << " (" << gstate << ")" << endl;
    }
    }
}

// ----------------------------------------------------------------------------
// linphone callback
// notifies registration state changes
// ----------------------------------------------------------------------------
static void registration_state_changed(struct _LinphoneCore* lc,
                                       LinphoneProxyConfig* cfg,
                                       LinphoneRegistrationState cstate,
                                       const char* message)
{
#ifdef VOIP_DEBUG
    cout << "cb: registration_state_changed  " << endl;
    cout << "    id : " << linphone_proxy_config_get_identity(cfg) << endl;
    cout << "    addr : " << linphone_proxy_config_get_addr(cfg) << endl;
    cout << "    state : " << linphone_registration_state_to_string(cstate) << endl;
    cout << "    msg : " << message << endl;
#endif

    switch (cstate)
    {
    case LinphoneRegistrationProgress:
    {
        lp_state = LinphoneClientState::registering;
        break;
    }
    case LinphoneRegistrationOk:
    {
        lp_state = LinphoneClientState::registered;
        break;
    }
    case LinphoneRegistrationCleared:
    case LinphoneRegistrationFailed:
    case LinphoneRegistrationNone:
    {
        lp_state = LinphoneClientState::offline;
        break;
    }
    default:
    {
        cout << "!!! info : unhandled notification "
             << linphone_registration_state_to_string(cstate)
             << " (" << cstate << ")" << endl;
    }
    }
}

// ----------------------------------------------------------------------------
// linphone callback
// notifies call state changes
// ----------------------------------------------------------------------------
static void call_state_changed(struct _LinphoneCore* lc,
                               LinphoneCall* call,
                               LinphoneCallState cstate,
                               const char* message)
{
#ifdef VOIP_DEBUG
    cout << "cb: call_state_changed  " << endl;
    cout << "    state : " << linphone_call_state_to_string(cstate) << endl;
    cout << "    msg : " << message << endl;
#endif

    switch (cstate)
    {
    case LinphoneCallOutgoingInit:
    {
        lp_state = LinphoneClientState::callInit;
        break;
    }
    case LinphoneCallOutgoingRinging:
    {
        lp_state = LinphoneClientState::callRinging;
        break;
    }
    case LinphoneCallOutgoingProgress:
    {
        lp_state = LinphoneClientState::callInProgress;
        break;
    }
    case LinphoneCallConnected:
    {
        lp_state = LinphoneClientState::callConnected;
        break;
    }
    case LinphoneCallStreamsRunning:
    {
        lp_state = LinphoneClientState::callStreaming;
        break;
    }
    case LinphoneCallPaused:
    {
        lp_state = LinphoneClientState::callPaused;
        break;
    }
    case LinphoneCallResuming:
    {
        lp_state = LinphoneClientState::callResuming;
        break;
    }
    case LinphoneCallEnd:
    {
        lp_state = LinphoneClientState::callEnded;
        break;
    }
    case LinphoneCallError:
    {
        lp_state = LinphoneClientState::callFailed;
        break;
    }
    case LinphoneCallReleased:
    {
        lp_state = LinphoneClientState::registered;
        break;
    }
    case LinphoneCallIncomingReceived:
    {
        lp_state = LinphoneClientState::callIncoming;
        break;
    }
    default:
    {
        cout << "!!! info : unhandled notification "
             << linphone_call_state_to_string(cstate)
             << " (" << cstate << ")" << endl;
    }
    }
}

// ----------------------------------------------------------------------------
// linphone callback
// ask the application some authentication information
// ----------------------------------------------------------------------------
static void authentication_requested(struct _LinphoneCore* lc,
                                     LinphoneAuthInfo* ainfo,
                                     LinphoneAuthMethod method)
{
    cout << "cb: authentication_requested" << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// a message is received, can be text or external body
// ----------------------------------------------------------------------------
static void message_received(struct _LinphoneCore* lc,
                             LinphoneChatRoom* croom,
                             LinphoneChatMessage* chatmsg)
{
    cout << "cb: message_received" << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// notify received presence events
// ----------------------------------------------------------------------------
static void notify_presence_received(struct _LinphoneCore* lc,
                                     LinphoneFriend* lpfriend)
{
    cout << "cb: notify_presence_received" << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// notifies an incoming informational message received
// ----------------------------------------------------------------------------
static void info_received(struct _LinphoneCore* lc,
                          LinphoneCall* call,
                          const LinphoneInfoMessage* infomsg)
{
    cout << "cb: info_received" << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// notifies subscription state change
// ----------------------------------------------------------------------------
static void subscription_state_changed(struct _LinphoneCore* lc,
                                       LinphoneEvent* event,
                                       LinphoneSubscriptionState substate)
{
    cout << "cb: subscription_state_changed" << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// notifies a an event notification, see linphone_core_subscribe()
// ----------------------------------------------------------------------------
static void notify_received(struct _LinphoneCore* lc,
                            LinphoneEvent* event,
                            const char* message,
                            const LinphoneContent* content)
{
    cout << "cb: notify_received" << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// notifies configuring status changes 
// ----------------------------------------------------------------------------
static void configuring_status(struct _LinphoneCore* lc,
                               LinphoneConfiguringState cstate,
                               const char* message)
{
    cout << "cb: configuring_status  " << endl;
}

// ----------------------------------------------------------------------------
// linphone callback
// linphonecall callbacks
// ----------------------------------------------------------------------------
static void cb_call_video_frame_decoded(LinphoneCall *call)
{
    cout << "cb (CALL): video frame decoded " << endl;

    cout << "!!! 1" << endl;

    const LinphoneCallParams* params = linphone_call_get_params(call);

    cout << "!!! 2" << endl;

//    const LinphoneVideoDefinition* videodef = linphone_call_params_get_received_video_definition(params);

    cout << linphone_call_params_get_received_framerate(params) << endl;
    cout << linphone_call_params_get_sent_framerate(params) << endl;

    // if (videodef == NULL) return;
    
    // cout << "height: " << linphone_video_definition_get_height(videodef) << endl;
    
//    linphone_call_request_notify_next_video_frame_decoded(call);
/*                                                   
    if (cstate == LinphoneCallUpdatedByRemote)
    {
        LinphoneCallParams *params = linphone_core_create_call_params(lc, call);
        linphone_call_params_enable_video(params, TRUE);
        ms_message (" New state LinphoneCallUpdatedByRemote on call [%p], accepting with video on",call);
        BC_ASSERT_NOT_EQUAL(linphone_call_accept_update(call, params), 0, int, "%i");
        linphone_call_params_unref(params);
    }
*/
}

// ------------------------------------------------------------------------
//! constructor, set global callbacks, init state
// ------------------------------------------------------------------------
LinphoneClient::LinphoneClient()
    : thdFunction()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::LinphoneClient" << endl;
#endif
    
    vtable.registration_state_changed = registration_state_changed;
    vtable.global_state_changed = global_state_changed;
    vtable.call_state_changed = call_state_changed;
    vtable.configuring_status = configuring_status;
    vtable.notify_received = notify_received;
    vtable.subscription_state_changed = subscription_state_changed;
    vtable.info_received = info_received;
    vtable.notify_presence_received = notify_presence_received;
    vtable.message_received = message_received;
    vtable.authentication_requested = authentication_requested;
    
    lc = linphone_core_new(&vtable,NULL,NULL,NULL);
    linphone_core_set_user_agent(lc, "OpenCOVER", "0");

    proxy_cfg = linphone_proxy_config_new();
    
    lp_state = LinphoneClientState::offline;
}

// ------------------------------------------------------------------------
//! destructor, wait for thread to terminate, release linphone core
// ------------------------------------------------------------------------
LinphoneClient::~LinphoneClient()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::~LinphoneClient" << endl;
#endif
            
    stopThread = true;
    
    if (thdFunction.joinable())
    {
        thdFunction.join();
    }
    
    cout << "shutting down ..." << endl;
    linphone_core_destroy(lc);
    cout << "exit" << endl;
}

// ------------------------------------------------------------------------                                                      
//! sets the callback handler
// ------------------------------------------------------------------------                                                      
void LinphoneClient::addHandler(std::function<void (LinphoneClientState, LinphoneClientState)>* handler)
{
    handlerList.push_back(handler);
}

// ------------------------------------------------------------------------
//! sets the UDP port range for audio streaming
// ------------------------------------------------------------------------
void LinphoneClient::setAudioPortRange(unsigned int portMin, unsigned int portMax)
{
    linphone_core_set_audio_port_range(lc, portMin, portMax);
}

// ------------------------------------------------------------------------
//! sets the UDP port range for video streaming
// ------------------------------------------------------------------------
void LinphoneClient::setVideoPortRange(unsigned int portMin, unsigned int portMax)
{
    linphone_core_set_video_port_range(lc, portMin, portMax);
}

// ------------------------------------------------------------------------
//! create new thread, start core iterator                                                                                       
// ------------------------------------------------------------------------
void LinphoneClient::startCoreIterator()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::start()" << endl;
#endif
    
    thdFunction = std::thread(&LinphoneClient::thdMain, this);
}

// ------------------------------------------------------------------------                                                      
//! initiate register/login to sip server with given credentials                                                                 
// ------------------------------------------------------------------------                                                      
bool LinphoneClient::doRegistration(std::string identity, std::string password)
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::doRegistration()" << endl;
#endif

    from = linphone_address_new(identity.c_str());

    if (from == NULL)
    {
        cout << "LinphoneClient error: identity " << identity << " not valid" << endl;
        return false;
    }

#ifdef VOIP_DEBUG
    cout << "create authentication" << endl;
#endif

    info = linphone_auth_info_new(linphone_address_get_username(from),
                                  NULL, password.c_str(), NULL, NULL, NULL);
    linphone_core_add_auth_info(lc, info);
    
#ifdef VOIP_DEBUG
    cout << "configure proxy" << endl;
#endif
    
    linphone_proxy_config_set_identity(proxy_cfg, identity.c_str()); // set identity with user name and domain
    server_addr = linphone_address_get_domain(from); // extract domain address from identity
    linphone_proxy_config_set_server_addr(proxy_cfg, server_addr); // we assume domain = proxy server address

    linphone_proxy_config_enable_register(proxy_cfg,TRUE); // activate registration for this proxy config
    linphone_address_unref(from); // release resource
    
    linphone_core_add_proxy_config(lc,proxy_cfg); // add proxy config to linphone core
    linphone_core_set_default_proxy(lc,proxy_cfg); // set to default proxy
    
    linphone_core_iterate(lc); // first iterate initiates registration
    
    return true;
}

// ------------------------------------------------------------------------                                                      
//! initiate unregistration
// ------------------------------------------------------------------------                                                      
void LinphoneClient::doUnregistration()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::doUnregistration()" << endl;
#endif
    
    proxy_cfg = linphone_core_get_default_proxy_config(lc); // get default proxy config                                      
    linphone_proxy_config_edit(proxy_cfg); // start editing proxy configuration                                              
    linphone_proxy_config_enable_register(proxy_cfg,FALSE); // de-activate registration for this proxy config                
    linphone_proxy_config_done(proxy_cfg); // initiate REGISTER with expire = 0                                              

    while(linphone_proxy_config_get_state(proxy_cfg) !=  LinphoneRegistrationCleared)
    {
        cout << "try unreg" << endl;
        linphone_core_iterate(lc);
        ms_usleep(50000);
    }
}

// ------------------------------------------------------------------------                                                      
//! check if audio is enabled within a call                                                                                      
// ------------------------------------------------------------------------                                                      
bool LinphoneClient::callAudioIsEnabled()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::audioEnabled()" << endl;
#endif
    
    if (!call) return false;
    
    const LinphoneCallParams* params = linphone_call_get_remote_params(call);
    
    return linphone_call_params_audio_enabled(params);
}

// ------------------------------------------------------------------------                                                      
//! check if video is enabled within a call                                                                                      
// ------------------------------------------------------------------------                                                      
bool LinphoneClient::callVideoIsEnabled()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::videoEnabled()" << endl;
#endif
    
    if (!call) return false;
    
    const LinphoneCallParams* params = linphone_call_get_remote_params(call);
    
    return linphone_call_params_video_enabled(params);
}

// ------------------------------------------------------------------------                                                      
//! check if camera stream is allowed to be sent within a call
// ------------------------------------------------------------------------                                                      
bool LinphoneClient::callCameraIsEnabled()
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::cameraEnabled()" << endl;
#endif
    
    if (!call) return false;
    
    return linphone_call_camera_enabled(call);
}

// ------------------------------------------------------------------------                                                      
//! gets the name of the currently assigned sound device for playback
// ------------------------------------------------------------------------                                                      
const char* LinphoneClient::getPlaybackDeviceName()
{
    return linphone_core_get_playback_device(lc);
}

// ------------------------------------------------------------------------                                                      
//! Gets the list of the available sound devices which can capture sound
// ------------------------------------------------------------------------                                                      
vector<string> LinphoneClient::getCaptureSoundDevicesList()
{
    vector<string> vecList;

    const char** list = linphone_core_get_sound_devices(lc);
    
    while(*list != NULL)
    {
        if (linphone_core_sound_device_can_capture(lc, *list))
        {
            vecList.push_back(std::string(*list));
        }
        
        ++list;
    }

    return vecList;
}

// ------------------------------------------------------------------------                                                      
//! Gets the list of the available sound devices which can playback sound
// ------------------------------------------------------------------------                                                      
vector<string> LinphoneClient::getPlaybackSoundDevicesList()
{
    vector<string> vecList;

    const char** list = linphone_core_get_sound_devices(lc);
    
    while(*list != NULL)
    {
        if (linphone_core_sound_device_can_playback(lc, *list))
        {
            vecList.push_back(std::string(*list));
        }
        
        ++list;
    }

    return vecList;
}

// ------------------------------------------------------------------------
//! gets the name of the currently capture sound device
// ------------------------------------------------------------------------
std::string LinphoneClient::getCurrentCaptureSoundDevice()
{
    return string(linphone_core_get_capture_device(lc));
}

// ------------------------------------------------------------------------
//! gets the name of the currently playback sound device
// ------------------------------------------------------------------------
std::string LinphoneClient::getCurrentPlaybackSoundDevice()
{
    return string(linphone_core_get_playback_device(lc));
}

// ------------------------------------------------------------------------
//! gets the name of the currently ringer sound device
// ------------------------------------------------------------------------
std::string LinphoneClient::getCurrentRingerSoundDevice()
{
    return string(linphone_core_get_ringer_device(lc));
}

// ------------------------------------------------------------------------
//! gets the name of the currently media sound device
// FIXME: not available in 3.12 (?)
// ------------------------------------------------------------------------
//std::string LinphoneClient::getCurrentMediaSoundDevice()
//{
//    return string(linphone_core_get_media_device(lc));
//}

// ------------------------------------------------------------------------
//! tells wether microphone is enabled or not
// ------------------------------------------------------------------------
bool LinphoneClient::getMicrophoneIsEnabled()
{
    return linphone_core_mic_enabled(lc);
}

// ------------------------------------------------------------------------
//! check whether the device has a hardware echo canceller
// FIXME: not available in 3.12 (?)
// ------------------------------------------------------------------------
//bool LinphoneClient::getEchoCancellerAvailable()
//{
//    return linphone_core_has_builtin_echo_canceller(lc);
//}

// ------------------------------------------------------------------------
//! returns true if echo cancellation is enabled
// ------------------------------------------------------------------------
bool LinphoneClient::getEchoCancellationIsEnabled()
{
    return linphone_core_echo_cancellation_enabled(lc);
}

// ------------------------------------------------------------------------
//! returns true if echo limiter is enabled
// ------------------------------------------------------------------------
bool LinphoneClient::getEchoLimiterIsEnabled()
{
    return linphone_core_echo_limiter_enabled(lc);
}

// ------------------------------------------------------------------------
//! tells whether the audio adaptive jitter compensation is enabled
// ------------------------------------------------------------------------
bool LinphoneClient::getAudioJitterCompensation()
{
    return linphone_core_audio_adaptive_jittcomp_enabled(lc);
}

// ------------------------------------------------------------------------
//! get microphone gain in db
// ------------------------------------------------------------------------
float LinphoneClient::getMicGain()
{
    return linphone_core_get_mic_gain_db(lc);
}

// ------------------------------------------------------------------------
//! get playback gain in db before entering sound card.
// ------------------------------------------------------------------------
float LinphoneClient::getPlaybackGain()
{
    return linphone_core_get_playback_gain_db(lc);
}

// ------------------------------------------------------------------------
//! tells whether video capture is enabled.
// ------------------------------------------------------------------------
bool LinphoneClient::getVideoCaptureEnabled()
{
    return linphone_core_video_capture_enabled(lc);
}

// ------------------------------------------------------------------------
//! Tells whether video display is enabled.
// ------------------------------------------------------------------------
bool LinphoneClient::getVideoDisplayEnabled()
{
    return linphone_core_video_display_enabled(lc);
}

// ------------------------------------------------------------------------
//! tells whether video preview is enabled
// ------------------------------------------------------------------------
bool LinphoneClient::getVideoPreviewEnabled()
{
    return linphone_core_video_preview_enabled(lc);
}

// ------------------------------------------------------------------------
//! get the default policy for acceptance of incoming video
// ------------------------------------------------------------------------
bool LinphoneClient::getAutoAcceptVideo()
{
    LinphoneVideoActivationPolicy* policy = linphone_core_get_video_activation_policy(lc);
    return linphone_video_activation_policy_get_automatically_initiate(policy);
}

// ------------------------------------------------------------------------
//! get the default policy for initiating video
// ------------------------------------------------------------------------
bool LinphoneClient::getAutoInitiateVideo()
{
    LinphoneVideoActivationPolicy* policy = linphone_core_get_video_activation_policy(lc);
    return linphone_video_activation_policy_get_automatically_accept(policy);
}

// ------------------------------------------------------------------------
//! tells whether the video adaptive jitter compensation is enabled
// ------------------------------------------------------------------------
bool LinphoneClient::getVideoJitterCompensation()
{
    return linphone_core_video_adaptive_jittcomp_enabled(lc);
}

// ------------------------------------------------------------------------                                                      
//! gets a list of the available video capture devices
// ------------------------------------------------------------------------                                                      
vector<string> LinphoneClient::getVideoCaptureDevicesList()
{
    vector<string> vecList;

    const char** list = linphone_core_get_video_devices(lc);
    
    while(*list != NULL)
    {
        vecList.push_back(std::string(*list));
        ++list;
    }

    return vecList;
}

// ------------------------------------------------------------------------
//! returns the name of the currently active video device
// ------------------------------------------------------------------------
string LinphoneClient::getCurrentVideoCaptureDevice()
{
    return linphone_core_get_video_device(lc);
}

// ------------------------------------------------------------------------
//! get microphone muted state
//! \todo linphone_call_get_microphone_muted not found in 3.12 ?
// ------------------------------------------------------------------------
bool LinphoneClient::getCallMicrophoneMuted()
{
    return linphone_core_is_mic_muted(lc);
}

// ------------------------------------------------------------------------
//! get speaker muted state
//! \fixme workaround for missing linphone_call_get_speaker_muted in 3.12
// ------------------------------------------------------------------------
bool LinphoneClient::getCallSpeakerMuted()
{
    if (call != NULL)
    {
        float gain = linphone_call_get_speaker_volume_gain(call);
        return (gain < 0.01);
    }
    else
    {
        return false;
    }
}

// ------------------------------------------------------------------------
//! self view during call is enabled
// ------------------------------------------------------------------------
bool LinphoneClient::getCallSelfViewEnabled()
{
    return linphone_core_self_view_enabled(lc);
}

// ------------------------------------------------------------------------
//! camera stream are allowed to be send to remote
//! \todo linphone_call_camera_enabled crashes ?
// ------------------------------------------------------------------------
bool LinphoneClient::getCallCameraEnabled()
{
    LinphoneCall* call = linphone_core_get_current_call(lc);

    //linphone_call_camera_enabled(call);

    return false; 
}

// ------------------------------------------------------------------------
//! get current linphone state
// ------------------------------------------------------------------------
LinphoneClientState LinphoneClient::getCurrentState()
{
    return lp_state;
}

// ------------------------------------------------------------------------
//! easier readability for debugging purposes
// ------------------------------------------------------------------------
std::string LinphoneClient::getStateString(LinphoneClientState state)
{
    std::string rstrg = "unknown";
    
    switch (state)
    {
    case LinphoneClientState::off:
    {
        rstrg = "off";
        break;
    }
    case LinphoneClientState::halted:
    {
        rstrg = "halted";
        break;
    }
    case LinphoneClientState::started:
    {
        rstrg = "started";
        break;
    }
    case LinphoneClientState::configured:
    {
        rstrg = "configured";
        break;
    }
    case LinphoneClientState::offline:
    {
        rstrg = "offline";
        break;
    }
    case LinphoneClientState::registering:
    {
        rstrg = "registering";
        break;
    }
    case LinphoneClientState::registered:
    {
        rstrg = "registered";
        break;
    }
    case LinphoneClientState::callIncoming:
    {
        rstrg = "callIncoming";
        break;
    }
    case LinphoneClientState::callPaused:
    {
        rstrg = "callPaused";
        break;
    }
    case LinphoneClientState::callResuming:
    {
        rstrg = "callResuming";
        break;
    }
    case LinphoneClientState::callInit:
    {
        rstrg = "callInit";
        break;
    }
    case LinphoneClientState::callInProgress:
    {
        rstrg = "callInProgress";
        break;
    }
    case LinphoneClientState::callRinging:
    {
        rstrg = "callRinging";
        break;
    }
    case LinphoneClientState::callFailed:
    {
        rstrg = "callFailed";
        break;
    }
    case LinphoneClientState::callConnected:
    {
        rstrg = "callConnected";
        break;
    }
    case LinphoneClientState::callStreaming:
    {
        rstrg = "callStreaming";
        break;
    }
    case LinphoneClientState::callEnded:
    {
        rstrg = "callEnded";
        break;
    }
    }

    return rstrg;
}   

// ------------------------------------------------------------------------
//! initiate call to given sip address, keep valid reference to call ptr
// ------------------------------------------------------------------------
bool LinphoneClient::initiateCall(std::string dest)
{
#ifdef VOIP_DEBUG
    cout << "LinphoneClient::initiateCall()" << endl;
#endif
    
    call = linphone_core_invite(lc, dest.c_str());

    if (call == NULL)
    {
        cout << "error: could not place call to " << dest << endl;
        return false;
    }
    else
    {
        cout << "calling " << dest << " ..." << endl;
    }
    
    linphone_call_ref(call);
    return true;
}

// ------------------------------------------------------------------------
//! hang up current call
// ------------------------------------------------------------------------
void LinphoneClient::hangUpCall()
{
    if (call != NULL)
    {
        linphone_call_terminate(call);
        linphone_call_unref(call);        
    }
}

// ------------------------------------------------------------------------
//! pause or resume current call
// ------------------------------------------------------------------------
void LinphoneClient::pauseCall(bool pause)
{
    if (call != NULL)
    {
        if (pause)
        {
            linphone_call_pause(call);
        }
        else
        {
            linphone_call_resume(call);
        }
    }
}

// ------------------------------------------------------------------------
//! mute mic while in call
//! \fixme linphone_core_mute_mic is deprecated but
//!        linphone_call_set_microphone_muted not implemented in 3.12.0-3
// ------------------------------------------------------------------------
void LinphoneClient::setMicMute(bool mute)
{
    linphone_core_mute_mic(lc, mute);
    
    // if (call != NULL)
    // {
    //     linphone_call_set_microphone_muted(call, mute);
    // }
}

// ------------------------------------------------------------------------
//! mute speaker while in call
//! \fixme linphone_call_set_speaker_muted  not implemented in 3.12.0-3
// ------------------------------------------------------------------------
void LinphoneClient::setSpeakerMute(bool mute)
{
    if (call != NULL)
    {
        //linphone_call_set_speaker_muted(call, mute);
        if (mute)
        {
            savedSpeakerGain = linphone_call_get_speaker_volume_gain(call);
            linphone_call_set_speaker_volume_gain(call, 0.0);
        }
        else
        {
            linphone_call_set_speaker_volume_gain(call, savedSpeakerGain);
        }
    }
}

// ------------------------------------------------------------------------
//! enable mic when starting a call
// ------------------------------------------------------------------------
void LinphoneClient::setMicrophoneEnabled(bool enable)
{
    linphone_core_enable_mic(lc, enable);
}

// ------------------------------------------------------------------------
//! enable  camera when starting a call
// ------------------------------------------------------------------------
void LinphoneClient::setCallCameraEnabled(bool enable)
{
    if (call != NULL)
    {
        linphone_call_enable_camera(call, enable);
    }
}

// ------------------------------------------------------------------------
//! enable self view/pip in video window
// ------------------------------------------------------------------------
void LinphoneClient::setCallSelfViewEnabled(bool enable)
{
    linphone_core_enable_self_view(lc, enable);
}

// ------------------------------------------------------------------------
//! get no. of calls
// ------------------------------------------------------------------------
int LinphoneClient::getNoOfCalls()
{
    return linphone_core_get_calls_nb(lc);
}


// ------------------------------------------------------------------------
//! LinphoneClient main loop
// ------------------------------------------------------------------------
void LinphoneClient::thdMain()
{
#ifdef VOIP_DEBUG            
    cout << "LinphoneClient::thdMain()" << endl;
#endif

    unsigned int timer = 0;
    
    while(!stopThread)
    {
        linphone_core_iterate(lc);

#ifdef VOIP_DEBUG            
        cout << "lp_state = " << static_cast<int>(lp_state)<< endl;
#endif

        // call external stateChanged handlers

        if (lp_state_old != lp_state)
        {
            for (auto& handler : handlerList)
            {
                (*handler)(lp_state_old, lp_state);
            }
            lp_state_old = lp_state;
        }
        
        // timeouts

        if (lp_state == LinphoneClientState::registering)
        {
            if (timer >= timeout)
            {
                lp_state = LinphoneClientState::offline;
            }
            else
            {
                ++timer;
            }
        }

        if (lp_state == LinphoneClientState::registered)
        {
            timer = 0;
        }
                
        // handle events
        
        if (lp_state == LinphoneClientState::callIncoming)
        {
            // TODO
        }

        if ((lp_state == LinphoneClientState::callStreaming) && (oneTime == false))
        {
            // TODO
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(intervall));
    }
}

// ------------------------------------------------------------------------
