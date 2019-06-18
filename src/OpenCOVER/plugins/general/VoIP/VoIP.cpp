/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "VoIP.h"

#include <cover/coVRTui.h>

VoIPPlugin *VoIPPlugin::plugin = NULL;

// ----------------------------------------------------------------------------

using namespace covise;
using namespace opencover;
using namespace std;

// ----------------------------------------------------------------------------
VoIPPlugin::VoIPPlugin()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::VoIPPlugin()" << endl;
#endif
        
    server.sipaddress = coCoviseConfig::getEntry("sipaddress", "COVER.Plugin.VoIP.SIPServer", "sip:ip");
    server.transport = coCoviseConfig::getEntry("transport", "COVER.Plugin.VoIP.SIPServer", "udp");

    server.portMinAudio = coCoviseConfig::getInt("min_port", "COVER.Plugin.VoIP.AudioRTPPorts", 10000);
    server.portMaxAudio = coCoviseConfig::getInt("max_port", "COVER.Plugin.VoIP.AudioRTPPorts", 14999);
    server.portMinVideo = coCoviseConfig::getInt("min_port", "COVER.Plugin.VoIP.VideoRTPPorts", 15000);
    server.portMaxVideo = coCoviseConfig::getInt("max_port", "COVER.Plugin.VoIP.VideoRTPPorts", 20000);
        
    identity.sipaddress = coCoviseConfig::getEntry("sipaddress", "COVER.Plugin.VoIP.Identity", "sip:name@ip");
    identity.localport = coCoviseConfig::getInt("localport", "COVER.Plugin.VoIP.Identity", 5060);
    identity.displayname = coCoviseConfig::getEntry("displayname", "COVER.Plugin.VoIP.Identity", "displayname");
    identity.username = coCoviseConfig::getEntry("username", "COVER.Plugin.VoIP.Identity", "username");
    identity.password = coCoviseConfig::getEntry("password", "COVER.Plugin.VoIP.Identity", "password");

    std::vector<std::string> contactsRead = coCoviseConfig::getScopeNames("COVER.Plugin.VoIP", "contact");

    for(std::vector<std::string>::iterator it = contactsRead.begin(); it != contactsRead.end(); ++it) 
    {
        SIPIdentity newContact;

        // FIXME: getScopeNames reads wrong character set ?
        string s = *it;
        std::replace(s.begin(), s.end(), '|', ':');
        std::replace(s.begin(), s.end(), '_', '.'); 

        newContact.sipaddress = s;
        contacts.push_back(newContact);
    }

    lpc = new LinphoneClient();

    handler = std::bind(&VoIPPlugin::msgCallback, this, std::placeholders::_1, std::placeholders::_2);
    lpc->addHandler(&handler);

    // configure linphone
    
    lpc->setAudioPortRange(server.portMinAudio, server.portMaxAudio);
    lpc->setVideoPortRange(server.portMinVideo, server.portMaxVideo);
    
    lpc->startCoreIterator();
}

// ----------------------------------------------------------------------------
VoIPPlugin::~VoIPPlugin()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::~VoIPPlugin()" << endl;
#endif

    if (lpc->getCurrentState() >= LinphoneClientState::registered)
    {
        lpc->doUnregistration();
    }
    
    delete lpc;
}

// ----------------------------------------------------------------------------
VoIPPlugin* VoIPPlugin::instance()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::instance()" << endl;
#endif
    
    return plugin;
}

// ----------------------------------------------------------------------------
bool VoIPPlugin::init()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::init()" << endl;
#endif
    
    createMenu();

    bool exists;

    if (coCoviseConfig::isOn("autoregistration", "COVER.Plugin.VoIP.SIPServer", false, &exists) == true)
    {
        bool success = lpc->doRegistration(identity.sipaddress, identity.password);

        // here all we know is that the initiation was successful or was not
        if (success)
        {
            menuCheckboxRegister->setState(true);
            menuCheckboxRegister->setLabel("Registering ...");
        }
        else
        {
            menuCheckboxRegister->setState(false);
        }
    }
    
    return true;
}

// ----------------------------------------------------------------------------
void VoIPPlugin::msgCallback(LinphoneClientState oldState, LinphoneClientState currentState)
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::msgCallback()" << endl;
    
    cout << "******** linphoneclient state change *******" << endl;
    cout << "old was : " << lpc->getStateString(oldState) << endl;
    cout << "current : " << lpc->getStateString(currentState) << endl;
    cout << "********************************************" << endl;
#endif

    if (oldState == LinphoneClientState::registering)
    {
        if (currentState == LinphoneClientState::registered)
        {
            menuCheckboxRegister->setState(true);
            menuCheckboxRegister->setLabel("Registered");
        }
        else
        {
            menuCheckboxRegister->setState(false);
            menuCheckboxRegister->setLabel("Register");
        }
    }
    else if (oldState == LinphoneClientState::registered)
    {
        if (currentState == LinphoneClientState::callInProgress)
        {
            menuLabelCallState->setLabel("Call: in progress");
        }
        else
        {
            menuLabelCallState->setLabel("Call: initiating failed");
        }
    }
    else if (oldState == LinphoneClientState::callInProgress)
    {
        if (currentState == LinphoneClientState::callRinging)
        {
            menuLabelCallState->setLabel("Call: ringing ...");
            updateMenu();
        }
        else
        {
            menuLabelCallState->setLabel("Call: call failed");
        }
    }
    else if (oldState == LinphoneClientState::callRinging)
    {
        if (currentState == LinphoneClientState::callConnected)
        {
            menuLabelCallState->setLabel("Call: connected");
        }
        else if (currentState == LinphoneClientState::callStreaming)
        {
            menuLabelCallState->setLabel("Call: streaming");
        }
        else
        {
            menuLabelCallState->setLabel("Call: call failed");
        }
        updateMenu();
    }
    else if (oldState == LinphoneClientState::callConnected)
    {
        if (currentState == LinphoneClientState::callStreaming)
        {
            menuLabelCallState->setLabel("Call: streaming");
        }
        else
        {
            menuLabelCallState->setLabel("Call: call failed");
        }
    }
    else if (oldState == LinphoneClientState::callStreaming)
    {
        if (currentState == LinphoneClientState::callPaused)
        {
            menuLabelCallState->setLabel("Call: paused");
        }
        else if (currentState == LinphoneClientState::callEnded)
        {
            menuLabelCallState->setLabel("Call: ended");
        }
        else
        {
            menuLabelCallState->setLabel("Call: call failed");
        }
    }
    else if (oldState == LinphoneClientState::callPaused)
    {
        if (currentState == LinphoneClientState::callResuming)
        {
            menuLabelCallState->setLabel("Call: resuming");
        }
    }
    else if (oldState == LinphoneClientState::callResuming)
    {
        if (currentState == LinphoneClientState::callStreaming)
        {
            menuLabelCallState->setLabel("Call: streaming");
        }
    }
    else if (oldState == LinphoneClientState::callEnded)
    {
        if (currentState == LinphoneClientState::registered)
        {
            menuLabelCallState->setLabel("Call: -");
        }
        updateMenu();
    }
}

// ----------------------------------------------------------------------------
void VoIPPlugin::menuEvent(vrui::coMenuItem *aButton)
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::menuEvent()" << endl;
#endif

    if (aButton == menuCheckboxRegister)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);

        if (cbx->getState() == true)
        {
            bool success = lpc->doRegistration(identity.sipaddress, identity.password);

            // here all we know is that the initiation was successful or was not
            if (success)
            {
                cbx->setLabel("Registering ...");
            }
            else
            {
                cbx->setState(false);
            }
        }
        else
        {
            cbx->setLabel("Register");
            lpc->doUnregistration();
        }
    }
    else if (aButton == menuButtonHangUp)
    {
        lpc->hangUpCall();
    }
    else if (aButton == menuCheckboxPause)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
        lpc->pauseCall(cbx->getState());
    }
    else if (aButton == menuCheckboxMicMute)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
        lpc->setMicMute(cbx->getState());
    }
    else if (aButton == menuCheckboxSpkrMute)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
        lpc->setSpeakerMute(cbx->getState());
    }
    else if (aButton == menuCheckboxMicEnabled)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
        lpc->setMicrophoneEnabled(cbx->getState());
    }
    else if (aButton == menuCheckboxEnableCamera)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
        lpc->setCallCameraEnabled(cbx->getState());
    }
    else if (aButton == menuCheckboxSelfViewEnabled)
    {
        vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
        lpc->setCallSelfViewEnabled(cbx->getState());
    }
    else if (aButton == menuPotiMicGain)
    {
        vrui::coPotiMenuItem* pot = static_cast<vrui::coPotiMenuItem*>(aButton);
        cout << pot->getValue() << endl;
    }
    else if (aButton == menuPotiPlaybackGain)
    {
        vrui::coPotiMenuItem* pot = static_cast<vrui::coPotiMenuItem*>(aButton);
        cout << pot->getValue() << endl;
    }
    else 
    {
        std::vector<SIPIdentity>::iterator contact = contacts.begin();
        std::vector<vrui::coButtonMenuItem*>::iterator it = menuContacts.begin();

        while ((it != menuContacts.end()) && (aButton != (*it)))
        {
            ++it;
            ++contact;
        }

        if (it != menuContacts.end())
        {
            lpc->initiateCall((*contact).sipaddress);

            menuLabelCallState->setLabel("Call: in progress");
            menuLabelCallNameOfPartner->setLabel("Addr: " + (*contact).sipaddress);
        }
    }
}

// ----------------------------------------------------------------------------
void VoIPPlugin::createMenu()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::createMenu()" << endl;
#endif

    // contacts menu
    
    menuContactList = new vrui::coRowMenu("Contact List");
    
    for(std::vector<SIPIdentity>::iterator it = contacts.begin(); it != contacts.end(); ++it) 
    {
        vrui::coButtonMenuItem* menuButtonContact = new vrui::coButtonMenuItem((*it).sipaddress);

        menuButtonContact->setMenuListener(this);
        menuContactList->add(menuButtonContact);
        menuContacts.push_back(menuButtonContact);
    }

    // call menu
    
    menuCall =  new vrui::coRowMenu("Call");

    menuLabelCallState = new vrui::coLabelMenuItem("Call: -");
    menuCall->add(menuLabelCallState);
    menuLabelCallNameOfPartner = new vrui::coLabelMenuItem("Addr: -");
    menuCall->add(menuLabelCallNameOfPartner);
    menuButtonHangUp = new vrui::coButtonMenuItem("Hang Up");
    menuButtonHangUp->setMenuListener(this);
    menuCall->add(menuButtonHangUp);
    menuCheckboxPause = new vrui::coCheckboxMenuItem("Pause", false);
    menuCheckboxPause->setMenuListener(this);
    menuCall->add(menuCheckboxPause);
    menuCheckboxMicMute = new vrui::coCheckboxMenuItem("Microphone Muted",
                                                       lpc->getCallMicrophoneMuted());
    menuCheckboxMicMute->setMenuListener(this);
    menuCall->add(menuCheckboxMicMute);
    menuCheckboxMicEnabled = new vrui::coCheckboxMenuItem("Mic Enabled",
                                                          lpc->getMicrophoneIsEnabled());
    menuCheckboxMicEnabled->setMenuListener(this);
    menuCall->add(menuCheckboxMicEnabled);
    menuCheckboxSpkrMute = new vrui::coCheckboxMenuItem("Speaker Muted",
                                                        lpc->getCallSpeakerMuted());
    menuCheckboxSpkrMute->setMenuListener(this);
    menuCall->add(menuCheckboxSpkrMute);
    menuCheckboxEnableCamera = new vrui::coCheckboxMenuItem("Enable Camera",
                                                            lpc->getCallCameraEnabled());
    menuCheckboxEnableCamera->setMenuListener(this);
    menuCall->add(menuCheckboxEnableCamera);
    menuCheckboxSelfViewEnabled = new vrui::coCheckboxMenuItem("Self View Enabled",
                                                               lpc->getCallSelfViewEnabled());
    menuCheckboxSelfViewEnabled->setMenuListener(this);
    menuCall->add(menuCheckboxSelfViewEnabled);

    // mixer menu
    
    menuMixer =  new vrui::coRowMenu("Audio Mixer"); 
    
    menuPotiMicGain = new vrui::coPotiMenuItem("Mic Gain (dB)", -100.0, 100.0, lpc->getMicGain());
    menuPotiMicGain->setMenuListener(this);
    menuPotiMicGain->setIncrement(1.0);
    menuPotiMicGain->setValue(0.0);
    menuPotiMicGain->setInteger(true);
    menuMixer->add(menuPotiMicGain);
    //menuPotiMicLevel = new vrui::coPotiMenuItem("Mic Level", 0.0, 100.0, 75.0);
    //menuMixer->add(menuPotiMicLevel);
    menuPotiPlaybackGain = new vrui::coPotiMenuItem("Playback Gain (dB)", -100.0, 100.0, lpc->getPlaybackGain());
    menuPotiPlaybackGain->setMenuListener(this);
    menuPotiPlaybackGain->setIncrement(1.0);
    menuPotiPlaybackGain->setValue(0.0);
    menuPotiPlaybackGain->setInteger(true);
    menuMixer->add(menuPotiPlaybackGain);
    //menuPotiVolLevel = new vrui::coPotiMenuItem("Volume Level", 0.0, 100.0, 75.0);
    //menuMixer->add(menuPotiVolLevel);

    // audio menu
    
    menuAudio =  new vrui::coRowMenu("Audio Settings");

    menuCheckboxEchoCancellationEnabled = new vrui::coCheckboxMenuItem("Echo Cancellation Enabled",
                                                                       lpc->getEchoCancellationIsEnabled());
    menuAudio->add(menuCheckboxEchoCancellationEnabled);
    menuCheckboxEchoLimiterEnabled = new vrui::coCheckboxMenuItem("Echo Limiter Enabled",
                                                                  lpc->getEchoLimiterIsEnabled());
    menuAudio->add(menuCheckboxEchoLimiterEnabled);
    menuCheckboxAudioJitterCompensation = new vrui::coCheckboxMenuItem("Jitter Compensation",
                                                                       lpc->getAudioJitterCompensation());
    menuAudio->add(menuCheckboxAudioJitterCompensation);
    
    vector<string> vecCaptureSoundDevices = lpc->getCaptureSoundDevicesList();
    vector<string> vecPlaybackSoundDevices = lpc->getPlaybackSoundDevicesList();

    menuLabelCaptureDeviceList = new vrui::coLabelMenuItem("Capture Devices");
    menuAudio->add(menuLabelCaptureDeviceList);
    
    for(vector<string>::iterator it = vecCaptureSoundDevices.begin(); it != vecCaptureSoundDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;
        if (lpc->getCurrentCaptureSoundDevice() == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuAudio->add(menuCheckboxDevice);
        menuCaptureDevices.push_back(menuCheckboxDevice);
    }
    
    menuLabelPlaybackDeviceList = new vrui::coLabelMenuItem("Playback Devices");
    menuAudio->add(menuLabelPlaybackDeviceList);

    for(vector<string>::iterator it = vecPlaybackSoundDevices.begin(); it != vecPlaybackSoundDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;
        if (lpc->getCurrentPlaybackSoundDevice() == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuAudio->add(menuCheckboxDevice);
        menuPlaybackDevices.push_back(menuCheckboxDevice);
    }

    menuLabelRingerDeviceList = new vrui::coLabelMenuItem("Ringer Devices");;
    menuAudio->add(menuLabelRingerDeviceList);

    for(vector<string>::iterator it = vecPlaybackSoundDevices.begin(); it != vecPlaybackSoundDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;
        if (lpc->getCurrentRingerSoundDevice() == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuAudio->add(menuCheckboxDevice);
        menuRingerDevices.push_back(menuCheckboxDevice);
    }

    // video settings
    
    menuVideo =  new vrui::coRowMenu("Video Settings"); 

    menuCheckboxVideoCaptureEnabled = new vrui::coCheckboxMenuItem("Video Capture Enabled",
                                                                   lpc->getVideoCaptureEnabled());
    menuVideo->add(menuCheckboxVideoCaptureEnabled);
    menuCheckboxVideoDisplayEnabled = new vrui::coCheckboxMenuItem("Video Display Enabled",
                                                                   lpc->getVideoDisplayEnabled());
    menuVideo->add(menuCheckboxVideoDisplayEnabled);
    menuCheckboxVideoPreviewEnabled = new vrui::coCheckboxMenuItem("Video Preview Enabled",
                                                                   lpc->getVideoPreviewEnabled());
    menuVideo->add(menuCheckboxVideoPreviewEnabled);
    menuCheckboxAutoAcceptVideo = new vrui::coCheckboxMenuItem("Auto Accept Video",
                                                               lpc->getAutoAcceptVideo());
    menuVideo->add(menuCheckboxAutoAcceptVideo);
    menuCheckboxAutoInitiateVideo = new vrui::coCheckboxMenuItem("Auto Initiate Video",
                                                                 lpc->getAutoInitiateVideo());
    menuVideo->add(menuCheckboxAutoInitiateVideo);
    menuCheckboxVideoJitterCompensation = new vrui::coCheckboxMenuItem("Jitter Compensation",
                                                                       lpc->getVideoJitterCompensation());
    menuVideo->add(menuCheckboxVideoJitterCompensation);
  
    menuLabelVideoDeviceList = new vrui::coLabelMenuItem("Video Capture Devices");
    menuVideo->add(menuLabelVideoDeviceList);

    vector<string> vecVideoCaptureDevices = lpc->getVideoCaptureDevicesList();
    
    for(vector<string>::iterator it = vecVideoCaptureDevices.begin(); it != vecVideoCaptureDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;

        if (lpc->getCurrentVideoCaptureDevice() == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuVideo->add(menuCheckboxDevice);
        menuVideoCaptureDevices.push_back(menuCheckboxDevice);
    }
    
    // plugin main menu
    
    menuLabelSIPAddress = new vrui::coLabelMenuItem(identity.sipaddress);
    menuCheckboxRegister = new vrui::coCheckboxMenuItem("Register", false); 
    menuCheckboxRegister->setMenuListener(this);
    
    menuContactListItem = new vrui::coSubMenuItem("Contact List..."); 
    menuContactListItem->setMenu(menuContactList);

    menuCallItem = new vrui::coSubMenuItem("Call..."); 
    menuCallItem->setMenu(menuCall);

    menuMixerItem = new vrui::coSubMenuItem("Audio Mixer..."); 
    menuMixerItem->setMenu(menuMixer);
    
    menuAudioItem = new vrui::coSubMenuItem("Audio Settings..."); 
    menuAudioItem->setMenu(menuAudio);
    
    menuVideoItem = new vrui::coSubMenuItem("Video Settings..."); 
    menuVideoItem->setMenu(menuVideo);
    
    menuVoIP = new vrui::coRowMenu("VoIP");
    menuVoIP->add(menuLabelSIPAddress);
    menuVoIP->add(menuCheckboxRegister);
    menuVoIP->add(menuContactListItem);
    menuVoIP->add(menuCallItem);
    menuVoIP->add(menuMixerItem);
    menuVoIP->add(menuAudioItem);
    menuVoIP->add(menuVideoItem);
    
    menuMainItem = new vrui::coSubMenuItem("VoIP");
    menuMainItem->setMenu(menuVoIP);
    
    cover->getMenu()->add(menuMainItem);

    updateMenu();
}

// ----------------------------------------------------------------------------
void VoIPPlugin::updateMenu()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::updateMenu()" << endl;
#endif

    if (lpc->getNoOfCalls() > 0)
    {
        menuButtonHangUp->setActive(true);
        menuCheckboxPause->setActive(true);
        menuCheckboxSpkrMute->setActive(true);
        menuCheckboxEnableCamera->setActive(true);
    }
    else
    {
        menuButtonHangUp->setActive(false);
        menuCheckboxPause->setActive(false);
        menuCheckboxSpkrMute->setActive(false);
        menuCheckboxEnableCamera->setActive(false);
    }
}

// ----------------------------------------------------------------------------
void VoIPPlugin::destroyMenu()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::destroyMenu()" << endl;
#endif
    
    delete menuMainItem;
    delete menuVoIP;
    delete menuVideoItem;
    delete menuAudioItem;
    delete menuCallItem;
    delete menuContactListItem;
    delete menuCheckboxRegister;
    delete menuLabelSIPAddress;
    delete menuVideo;
    delete menuAudio;
    delete menuCheckboxMicMute;
    delete menuCheckboxPause;
    delete menuButtonHangUp;
    delete menuCall;
    delete menuContactList;
    delete menuPotiMicLevel;
    delete menuPotiVolLevel;
    delete menuPotiMicGain;
    delete menuPotiPlaybackGain;    
    //delete menuCheckboxEnableAudio;
    //delete menuCheckboxEnableVideo;
    delete menuCheckboxEnableCamera;
    delete menuCheckboxSelfViewEnabled;

    delete menuCheckboxVideoCaptureEnabled;
    delete menuCheckboxVideoDisplayEnabled;
    delete menuCheckboxVideoPreviewEnabled;
    delete menuCheckboxAutoAcceptVideo;
    delete menuCheckboxAutoInitiateVideo;
    delete menuCheckboxVideoJitterCompensation;
    
    delete menuCheckboxMicEnabled;
    delete menuCheckboxEchoCancellationEnabled;
    delete menuCheckboxEchoLimiterEnabled;
    delete menuCheckboxAudioJitterCompensation;
    delete menuLabelCaptureDeviceList;
    delete menuLabelPlaybackDeviceList;
    delete menuLabelRingerDeviceList;
    delete menuLabelVideoDeviceList;
    delete menuLabelCallNameOfPartner;
    delete menuLabelCallState;
    delete menuCheckboxSpkrMute;
    
    for(vector<vrui::coCheckboxMenuItem*>::iterator it = menuCaptureDevices.begin();
        it != menuCaptureDevices.end(); ++it) 
    {
        delete *it;
    }
    
    for(vector<vrui::coCheckboxMenuItem*>::iterator it = menuPlaybackDevices.begin();
        it != menuPlaybackDevices.end(); ++it) 
    {
        delete *it;
    }
    
    for(vector<vrui::coCheckboxMenuItem*>::iterator it = menuRingerDevices.begin();
        it != menuRingerDevices.end(); ++it) 
    {
        delete *it;
    }
    
    for(vector<vrui::coCheckboxMenuItem*>::iterator it = menuMediaDevices.begin();
        it != menuMediaDevices.end(); ++it) 
    {
        delete *it;
    }

    for(vector<vrui::coCheckboxMenuItem*>::iterator it = menuVideoCaptureDevices.begin();
        it != menuVideoCaptureDevices.end(); ++it) 
    {
        delete *it;
    }
        
    for(vector<vrui::coButtonMenuItem*>::iterator it = menuContacts.begin(); it != menuContacts.end(); ++it) 
    {
        delete *it;
    }
}

// ----------------------------------------------------------------------------
void VoIPPlugin::potiValueChanged(float oldValue, float newValue, vrui::coValuePoti *poti, int context)
{
}

COVERPLUGIN(VoIPPlugin)
