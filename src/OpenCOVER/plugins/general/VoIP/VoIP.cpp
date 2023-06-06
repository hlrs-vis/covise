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
: coVRPlugin(COVER_PLUGIN_NAME)
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

    vector<string> contactsRead = coCoviseConfig::getScopeNames("COVER.Plugin.VoIP","Contact");

    coCoviseConfig::ScopeEntries keysEntries = coCoviseConfig::getScopeEntries("COVER.Plugin.VoIP","Contact");

    const char **keys = keysEntries.getValue();
    
    if (keys != NULL)
    {
        int i = 0;
        while (keys[i] != NULL)
        {
            SIPIdentity newContact;
            newContact.sipaddress = string(keys[++i]);
            contacts.push_back(newContact);
            ++i;
        }
    }
    
    nodNotifyQuestion = new NotifyDialog;
    
    if (coVRMSController::instance()->isMaster())
    {
        lpc = new LinphoneClient();
        
        handler = std::bind(&VoIPPlugin::msgCallback, this, std::placeholders::_1, std::placeholders::_2);
        lpc->addHandler(&handler);
        
        notifier = std::bind(&VoIPPlugin::msgNotify, this, std::placeholders::_1);
        lpc->addNotifier(&notifier);
        
        // configure linphone
        
        lpc->setAudioPortRange(server.portMinAudio, server.portMaxAudio);
        lpc->setVideoPortRange(server.portMinVideo, server.portMaxVideo);
        
        lpc->startCoreIterator();
    }
}

// ----------------------------------------------------------------------------
VoIPPlugin::~VoIPPlugin()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::~VoIPPlugin()" << endl;
#endif

    if (coVRMSController::instance()->isMaster())
    {
        if (lpc->getCurrentState() >= LinphoneClientState::registered)
        {
            lpc->doUnregistration();
        }
    
        delete lpc;
    }
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
    
    return true;
}

// ----------------------------------------------------------------------------
//! postInit()
//! is called once by the first update() call within this plugin
// ----------------------------------------------------------------------------
bool VoIPPlugin::postInit()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::postInit()" << endl;
#endif

    createMenu();

    bool exists;
    
    if (coCoviseConfig::isOn("autoregistration", "COVER.Plugin.VoIP.SIPServer", false, &exists) == true)
    {
        bool bSuccess;

        if (coVRMSController::instance()->isMaster())
        {
            bSuccess = lpc->doRegistration(identity.sipaddress, identity.password);
        }
        bSuccess = coVRMSController::instance()->syncBool(&bSuccess);

        // here all we know is that the initiation was successful or was not
        if (bSuccess)
        {
            syncMenu.bMCBRegister = true;
            sprintf(syncMenu.strMCBRegister, "%s", "Registering ...");
            menuCheckboxRegister->setState(syncMenu.bMCBRegister);
            menuCheckboxRegister->setLabel(syncMenu.strMCBRegister);
        }
        else
        {
            syncMenu.bMCBRegister = false;
            menuCheckboxRegister->setState(syncMenu.bMCBRegister);
        }
        requestUpdate();
    }

    updateMenu();

    return true;
}

// ----------------------------------------------------------------------------
//! update()
//! is called in sync with slaves
//! runs postInit() once at the very beginning
//! syncs RequestUpdate from master to slaves, so something had been chenged
// ----------------------------------------------------------------------------
bool VoIPPlugin::update()
{
// #ifdef VOIP_DEBUG
//     cout << "VoIPPlugin::update()" << endl;
// #endif

    if (bPostInitCompleted == false)
    {
        bPostInitCompleted = postInit();
    }

    char c = bRequestUpdate ? 1 : 0;
    coVRMSController::instance()->syncData(&c, 1);
    bRequestUpdate = (c != 0);

    if (bRequestUpdate == true)
    {
        cout << "VoIPPlugin::update() requested sync" << endl;
        updateMenu();
        bRequestUpdate = false;
        return true;
    }
    else
    {
        return true;
    }
}

// ----------------------------------------------------------------------------
//! msgNotify()
//! callback function, called by lpc instance
//! should never happen to be called by slave nodes
// ----------------------------------------------------------------------------
void VoIPPlugin::msgNotify(std::string strNotification)
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::msgNotify()" << endl;

    assert(coVRMSController::instance()->isMaster() == true);
    assert(lpc != nullptr);
    assert(strNotification.size() < SYNCBUFFER_SIZE);
#endif

    if (lpc->getCurrentState() == LinphoneClientState::callIncoming)
    {
        if (bNotified == false)
        {
            sprintf(syncMenu.strNODQuestion1, "%s", strNotification.c_str());
            sprintf(syncMenu.strNODQuestion2, "%s", "accept");
            sprintf(syncMenu.strNODQuestion3, "%s", "reject");

            string strTemp =  "Addr: " + lpc->getCurrentRemoteAddress();
            sprintf(syncMenu.strMLBCallNameOfPartner, "%s", strTemp.c_str());

            syncMenu.bNODNotifyQuestionShow  = true;

            requestUpdate();
            
            bNotified = true;
        }
        else
        {
            if (nodNotifyQuestion->getSelection() == "accept")
            {
                lpc->acceptIncomingCall();
            }
            else if (nodNotifyQuestion->getSelection() == "reject")
            {
                lpc->rejectIncomingCall();
            }
        }
    }
}

// ----------------------------------------------------------------------------
//! msgCallback()
//! callback function, called by lpc instance
//! should never happen to be called by slave nodes
// ----------------------------------------------------------------------------
void VoIPPlugin::msgCallback(LinphoneClientState oldState, LinphoneClientState currentState)
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::msgCallback()" << endl;
    
    cout << "******** linphoneclient state change *******" << endl;
    cout << "old was : " << lpc->getStateString(oldState) << endl;
    cout << "current : " << lpc->getStateString(currentState) << endl;
    cout << "********************************************" << endl;

    assert(coVRMSController::instance()->isMaster() == true);
    assert(lpc != nullptr);
#endif

    if (oldState == LinphoneClientState::registering)
    {
        if (currentState == LinphoneClientState::registered)
        {
            syncMenu.bMCBRegister = true;
            sprintf(syncMenu.strMCBRegister, "%s", "Registered");
        }
        else
        {
            syncMenu.bMCBRegister = false;
            sprintf(syncMenu.strMCBRegister, "%s", "Register");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::registered)
    {
        if (currentState == LinphoneClientState::callInProgress)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: in progress");
        }
        else if (currentState == LinphoneClientState::callIncoming)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: incoming");
        }
        else
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: initiating failed");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callIncoming)
    {
        if (currentState == LinphoneClientState::callEnded)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: ended");
        }
        else if (currentState == LinphoneClientState::callStreaming)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: streaming");
        }
        else
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: call failed");
        }
        requestUpdate();
    }

    else if (oldState == LinphoneClientState::callInProgress)
    {
        if (currentState == LinphoneClientState::callRinging)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: ringing ...");
        }
        else
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: call failed");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callRinging)
    {
        if (currentState == LinphoneClientState::callConnected)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: connected");
        }
        else if (currentState == LinphoneClientState::callStreaming)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: streaming");
        }
        else
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: call failed");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callConnected)
    {
        if (currentState == LinphoneClientState::callStreaming)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: streaming");
        }
        else
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: call failed");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callStreaming)
    {
        if (currentState == LinphoneClientState::callPaused)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: paused");
        }
        else if (currentState == LinphoneClientState::callEnded)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: ended");
        }
        else
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: call failed");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callPaused)
    {
        if (currentState == LinphoneClientState::callResuming)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: resuming");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callResuming)
    {
        if (currentState == LinphoneClientState::callStreaming)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: streaming");
        }
        requestUpdate();
    }
    else if (oldState == LinphoneClientState::callEnded)
    {
        if (currentState == LinphoneClientState::registered)
        {
            sprintf(syncMenu.strMLBCallState, "%s", "Call: -");
        }
        requestUpdate();
    }
}

// ----------------------------------------------------------------------------
//! createMenu()
//! is called once by postInit()
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

    sprintf(syncMenu.strMLBCallState, "%s", "Call: -");
    menuLabelCallState = new vrui::coLabelMenuItem(syncMenu.strMLBCallState);
    menuCall->add(menuLabelCallState);

    sprintf(syncMenu.strMLBCallNameOfPartner, "%s", "Addr: -");
    menuLabelCallNameOfPartner = new vrui::coLabelMenuItem(syncMenu.strMLBCallNameOfPartner);
    menuCall->add(menuLabelCallNameOfPartner);

    menuButtonHangUp = new vrui::coButtonMenuItem("Hang Up");
    menuButtonHangUp->setMenuListener(this);
    menuCall->add(menuButtonHangUp);

    menuCheckboxPause = new vrui::coCheckboxMenuItem("Pause", false);
    menuCheckboxPause->setMenuListener(this);
    menuCall->add(menuCheckboxPause);

    bool bState;

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getCallMicrophoneMuted();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxMicMute = new vrui::coCheckboxMenuItem("Microphone Muted", bState);
    menuCheckboxMicMute->setMenuListener(this);
    menuCall->add(menuCheckboxMicMute);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getMicrophoneIsEnabled();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxMicEnabled = new vrui::coCheckboxMenuItem("Mic Enabled", bState);
    menuCheckboxMicEnabled->setMenuListener(this);
    menuCall->add(menuCheckboxMicEnabled);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getCallSpeakerMuted();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxSpkrMute = new vrui::coCheckboxMenuItem("Speaker Muted", bState);
    menuCheckboxSpkrMute->setMenuListener(this);
    menuCall->add(menuCheckboxSpkrMute);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getCallCameraEnabled();
    }    
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxEnableCamera = new vrui::coCheckboxMenuItem("Enable Camera", bState);
    menuCheckboxEnableCamera->setMenuListener(this);
    menuCall->add(menuCheckboxEnableCamera);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getCallSelfViewEnabled();
    }    
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxSelfViewEnabled = new vrui::coCheckboxMenuItem("Self View Enabled", bState);
    menuCheckboxSelfViewEnabled->setMenuListener(this);
    menuCall->add(menuCheckboxSelfViewEnabled);

    // mixer menu
    
    menuMixer =  new vrui::coRowMenu("Audio Mixer"); 

    float fValue;
    if (coVRMSController::instance()->isMaster())
    {
        fValue = lpc->getMicGain();
    }    
    coVRMSController::instance()->syncData(&fValue, sizeof(fValue));
    menuPotiMicGain = new vrui::coPotiMenuItem("Mic Gain (dB)", -100.0, 100.0, fValue);
    menuPotiMicGain->setMenuListener(this);
    menuPotiMicGain->setIncrement(1.0);
    menuPotiMicGain->setInteger(true);
    menuPotiMicGain->setValue(0.0);
    menuMixer->add(menuPotiMicGain);

    // vvv not available in linphone sdk 3.12
    //menuPotiMicLevel = new vrui::coPotiMenuItem("Mic Level", 0.0, 100.0, 75.0);
    //menuMixer->add(menuPotiMicLevel);

    if (coVRMSController::instance()->isMaster())
    {
        fValue = lpc->getPlaybackGain();
    }
    coVRMSController::instance()->syncData(&fValue, sizeof(fValue));
    menuPotiPlaybackGain = new vrui::coPotiMenuItem("Playback Gain (dB)", -100.0, 100.0, fValue);
    menuPotiPlaybackGain->setMenuListener(this);
    menuPotiPlaybackGain->setIncrement(1.0);
    menuPotiPlaybackGain->setValue(0.0);
    menuPotiPlaybackGain->setInteger(true);
    menuMixer->add(menuPotiPlaybackGain);

    // vvv not available in linphone sdk 3.12
    //menuPotiVolLevel = new vrui::coPotiMenuItem("Volume Level", 0.0, 100.0, 75.0);
    //menuMixer->add(menuPotiVolLevel);

    // audio menu
    
    menuAudio =  new vrui::coRowMenu("Audio Settings");

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getEchoCancellationIsEnabled();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxEchoCancellationEnabled = new vrui::coCheckboxMenuItem("Echo Cancellation Enabled", bState);
    menuCheckboxEchoCancellationEnabled->setMenuListener(this);
    menuAudio->add(menuCheckboxEchoCancellationEnabled);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getEchoLimiterIsEnabled();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxEchoLimiterEnabled = new vrui::coCheckboxMenuItem("Echo Limiter Enabled", bState);
    menuCheckboxEchoCancellationEnabled->setMenuListener(this);
    menuAudio->add(menuCheckboxEchoLimiterEnabled);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getAudioJitterCompensation();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxAudioJitterCompensation = new vrui::coCheckboxMenuItem("Jitter Compensation", bState);
    menuCheckboxAudioJitterCompensation->setMenuListener(this);
    menuAudio->add(menuCheckboxAudioJitterCompensation);

    string strCurrentCaptureSoundDevice = "not available";
    string strCurrentPlaybackSoundDevice = "not available";
    string strCurrentRingerSoundDevice = "not available";

    const int BUFFER_SIZE = 10;
    const int BUFFER_LENGTH = 128;
    char buffer_CAP[BUFFER_SIZE][BUFFER_LENGTH];
    char buffer_PLB[BUFFER_SIZE][BUFFER_LENGTH];
    char bufferStr[BUFFER_SIZE];
    int vec_size_CAP = 0;
    int vec_size_PLB = 0;
    int index;
    
    if ((vecCaptureSoundDevices.size() > BUFFER_SIZE) ||
        (vecPlaybackSoundDevices.size() > BUFFER_SIZE))
    {
        cout << "WARNING (VoIP): sync buffer too small" << endl;
    }
    
    if (coVRMSController::instance()->isMaster())
    {
        vecCaptureSoundDevices = lpc->getCaptureSoundDevicesList();
        vecPlaybackSoundDevices = lpc->getPlaybackSoundDevicesList();

        strCurrentCaptureSoundDevice = lpc->getCurrentCaptureSoundDevice();
        strCurrentPlaybackSoundDevice = lpc->getCurrentPlaybackSoundDevice();
        strCurrentRingerSoundDevice = lpc->getCurrentRingerSoundDevice();

        std::vector<std::string>::iterator forIt = vecCaptureSoundDevices.begin();
        vec_size_CAP = vecCaptureSoundDevices.size();
        index = 0;
        while (forIt != vecCaptureSoundDevices.end())
        {
            (*forIt).copy(buffer_CAP[index], (*forIt).size());
            buffer_CAP[index][(*forIt).size()] = '\0';
            ++forIt;
            ++index;
        }

        forIt = vecPlaybackSoundDevices.begin();
        vec_size_PLB = vecPlaybackSoundDevices.size();
        index = 0;
        while (forIt != vecPlaybackSoundDevices.end())
        {
            (*forIt).copy(buffer_PLB[index], (*forIt).size());
            buffer_PLB[index][(*forIt).size()] = '\0';
            ++forIt;
            ++index;
        }

        coVRMSController::instance()->sendSlaves((char *)&buffer_PLB, sizeof(buffer_PLB));
        coVRMSController::instance()->sendSlaves((char *)&vec_size_PLB, sizeof(vec_size_PLB));
        coVRMSController::instance()->sendSlaves((char *)&buffer_CAP, sizeof(buffer_CAP));
        coVRMSController::instance()->sendSlaves((char *)&vec_size_CAP, sizeof(vec_size_CAP));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&buffer_PLB, sizeof(buffer_PLB));
        coVRMSController::instance()->readMaster((char *)&vec_size_PLB, sizeof(vec_size_PLB));
        coVRMSController::instance()->readMaster((char *)&buffer_CAP, sizeof(buffer_CAP));
        coVRMSController::instance()->readMaster((char *)&vec_size_CAP, sizeof(vec_size_CAP));
    }

    vecCaptureSoundDevices.clear();
    for (int i = 0; i < vec_size_CAP; ++i)
    {
        std::string s(buffer_CAP[i]);
        vecCaptureSoundDevices.push_back(s);
    }

    vecPlaybackSoundDevices.clear();
    for (int i = 0; i < vec_size_PLB; ++i)
    {
        std::string s(buffer_PLB[i]);
        vecPlaybackSoundDevices.push_back(s);
    }

    strCurrentPlaybackSoundDevice.copy(bufferStr, strCurrentPlaybackSoundDevice.size());
    bufferStr[strCurrentPlaybackSoundDevice.size()] = '\0';
    coVRMSController::instance()->syncData(&bufferStr, BUFFER_SIZE);
    strCurrentPlaybackSoundDevice = string(bufferStr);
            
    strCurrentCaptureSoundDevice.copy(bufferStr, strCurrentCaptureSoundDevice.size());
    bufferStr[strCurrentCaptureSoundDevice.size()] = '\0';
    coVRMSController::instance()->syncData(&bufferStr, BUFFER_SIZE);
    strCurrentCaptureSoundDevice = string(bufferStr);

    strCurrentRingerSoundDevice.copy(bufferStr, strCurrentRingerSoundDevice.size());
    bufferStr[strCurrentRingerSoundDevice.size()] = '\0';
    coVRMSController::instance()->syncData(&bufferStr, BUFFER_SIZE);
    strCurrentRingerSoundDevice = string(bufferStr);
    
    menuLabelCaptureDeviceList = new vrui::coLabelMenuItem("Capture Devices");
    menuAudio->add(menuLabelCaptureDeviceList);
    
    for(vector<string>::iterator it = vecCaptureSoundDevices.begin(); it != vecCaptureSoundDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;
        if (strCurrentCaptureSoundDevice == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuCheckboxDevice->setMenuListener(this);
        menuAudio->add(menuCheckboxDevice);
        menuCheckboxDevice->setMenuListener(this);
        menuCaptureDevices.push_back(menuCheckboxDevice);
    }
    
    menuLabelPlaybackDeviceList = new vrui::coLabelMenuItem("Playback Devices");
    menuAudio->add(menuLabelPlaybackDeviceList);

    for(vector<string>::iterator it = vecPlaybackSoundDevices.begin(); it != vecPlaybackSoundDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;
        if (strCurrentPlaybackSoundDevice == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuCheckboxDevice->setMenuListener(this);
        menuAudio->add(menuCheckboxDevice);
        menuPlaybackDevices.push_back(menuCheckboxDevice);
    }

    menuLabelRingerDeviceList = new vrui::coLabelMenuItem("Ringer Devices");;
    menuAudio->add(menuLabelRingerDeviceList);

    for(vector<string>::iterator it = vecPlaybackSoundDevices.begin(); it != vecPlaybackSoundDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;
        if (strCurrentRingerSoundDevice == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuCheckboxDevice->setMenuListener(this);
        menuAudio->add(menuCheckboxDevice);
        menuRingerDevices.push_back(menuCheckboxDevice);
    }

    // video settings
    
    menuVideo =  new vrui::coRowMenu("Video Settings"); 

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getVideoCaptureEnabled();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxVideoCaptureEnabled = new vrui::coCheckboxMenuItem("Video Capture Enabled", bState);
    menuCheckboxVideoCaptureEnabled->setMenuListener(this);
    menuVideo->add(menuCheckboxVideoCaptureEnabled);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getVideoDisplayEnabled();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxVideoDisplayEnabled = new vrui::coCheckboxMenuItem("Video Display Enabled", bState);
    menuCheckboxVideoDisplayEnabled->setMenuListener(this);
    menuVideo->add(menuCheckboxVideoDisplayEnabled);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getVideoPreviewEnabled();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxVideoPreviewEnabled = new vrui::coCheckboxMenuItem("Video Preview Enabled",bState);
    menuCheckboxVideoPreviewEnabled->setMenuListener(this);
    menuVideo->add(menuCheckboxVideoPreviewEnabled);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getAutoAcceptVideo();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxAutoAcceptVideo = new vrui::coCheckboxMenuItem("Auto Accept Video", bState);
    menuCheckboxAutoAcceptVideo->setMenuListener(this);
    menuVideo->add(menuCheckboxAutoAcceptVideo);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getAutoInitiateVideo();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxAutoInitiateVideo = new vrui::coCheckboxMenuItem("Auto Initiate Video", bState);
    menuCheckboxAutoInitiateVideo->setMenuListener(this);
    menuVideo->add(menuCheckboxAutoInitiateVideo);

    if (coVRMSController::instance()->isMaster())
    {
        bState = lpc->getVideoJitterCompensation();
    }
    bState = coVRMSController::instance()->syncBool(&bState);
    menuCheckboxVideoJitterCompensation = new vrui::coCheckboxMenuItem("Jitter Compensation", bState);
    menuCheckboxVideoJitterCompensation->setMenuListener(this);
    menuVideo->add(menuCheckboxVideoJitterCompensation);
    
    menuLabelVideoDeviceList = new vrui::coLabelMenuItem("Video Capture Devices");
    menuVideo->add(menuLabelVideoDeviceList);

    string strCurrentVideoCaptureDevice = "not available";

    if (coVRMSController::instance()->isMaster())
    {
        vecVideoCaptureDevices = lpc->getVideoCaptureDevicesList();
        strCurrentVideoCaptureDevice = lpc->getCurrentVideoCaptureDevice();

        std::vector<std::string>::iterator forIt = vecVideoCaptureDevices.begin();
        vec_size_CAP = vecVideoCaptureDevices.size();
        index = 0;
        while (forIt != vecVideoCaptureDevices.end())
        {
            (*forIt).copy(buffer_CAP[index], (*forIt).size());
            buffer_CAP[index][(*forIt).size()] = '\0';
            ++forIt;
            ++index;
        }

        coVRMSController::instance()->sendSlaves((char *)&buffer_CAP, sizeof(buffer_CAP));
        coVRMSController::instance()->sendSlaves((char *)&vec_size_CAP, sizeof(vec_size_CAP));

    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&buffer_CAP, sizeof(buffer_CAP));
        coVRMSController::instance()->readMaster((char *)&vec_size_CAP, sizeof(vec_size_CAP));

    }

    vecVideoCaptureDevices.clear();
    for (int i = 0; i < vec_size_CAP; ++i)
    {
        std::string s(buffer_CAP[i]);
        vecVideoCaptureDevices.push_back(s);
    }

    strCurrentVideoCaptureDevice.copy(bufferStr, strCurrentVideoCaptureDevice.size());
    bufferStr[strCurrentVideoCaptureDevice.size()] = '\0';
    coVRMSController::instance()->syncData(&bufferStr, BUFFER_SIZE);
    strCurrentVideoCaptureDevice = string(bufferStr);

    for(vector<string>::iterator it = vecVideoCaptureDevices.begin(); it != vecVideoCaptureDevices.end(); ++it) 
    {
        vrui::coCheckboxMenuItem* menuCheckboxDevice;

        if (strCurrentVideoCaptureDevice == *it)
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, true);
        }
        else
        {
            menuCheckboxDevice = new vrui::coCheckboxMenuItem(*it, false);
        }
        menuCheckboxDevice->setMenuListener(this);
        menuVideo->add(menuCheckboxDevice);
        menuVideoCaptureDevices.push_back(menuCheckboxDevice);
    }
    
    // plugin main menu
    
    menuLabelSIPAddress = new vrui::coLabelMenuItem(identity.sipaddress);
    sprintf(syncMenu.strMCBRegister, "%s", "Register");
    menuCheckboxRegister = new vrui::coCheckboxMenuItem(syncMenu.strMCBRegister, false); 
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
}

// ----------------------------------------------------------------------------
void VoIPPlugin::updateMenu()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::updateMenu()" << endl;
#endif

    cout << "SIZE: " << sizeof(syncMenu) << endl;


    int iNoOfCalls = 0;
    if (coVRMSController::instance()->isMaster())
    {
        iNoOfCalls = lpc->getNoOfCalls();
        coVRMSController::instance()->sendSlaves((char *)&syncMenu, sizeof(syncMenu));
        coVRMSController::instance()->sendSlaves((char *)&iNoOfCalls, sizeof(iNoOfCalls));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&syncMenu, sizeof(syncMenu));
        coVRMSController::instance()->readMaster((char *)&iNoOfCalls, sizeof(iNoOfCalls));
    }
    
    nodNotifyQuestion->setText(syncMenu.strNODQuestion1,
                               syncMenu.strNODQuestion2,
                               syncMenu.strNODQuestion3);
    menuLabelCallNameOfPartner->setLabel(syncMenu.strMLBCallNameOfPartner);
    menuCheckboxRegister->setState(syncMenu.bMCBRegister);
    menuCheckboxRegister->setLabel(syncMenu.strMCBRegister);
    menuLabelCallState->setLabel(syncMenu.strMLBCallState);
    
    if (syncMenu.bNODNotifyQuestionShow == true)
    {
        nodNotifyQuestion->show();
        syncMenu.bNODNotifyQuestionShow = false;
    }
    
    if (iNoOfCalls > 0)
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

#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::updateMenu() update done" << endl;
#endif
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
            bool bSuccess;
            if (coVRMSController::instance()->isMaster())
            {
                bSuccess = lpc->doRegistration(identity.sipaddress, identity.password);
            }
            bSuccess = coVRMSController::instance()->syncBool(&bSuccess);

            // here all we know is that the initiation was successful or was not
            if (bSuccess)
            {                                                      
                sprintf(syncMenu.strMCBRegister, "%s", "Registering ...");
                cbx->setLabel(syncMenu.strMCBRegister);
                syncMenu.bMCBRegister = true;
                cbx->setState(syncMenu.bMCBRegister);
            }
            else
            {
                syncMenu.bMCBRegister = false;
                cbx->setState(syncMenu.bMCBRegister);
            }
            requestUpdate();
        }
        else
        {
            sprintf(syncMenu.strMCBRegister, "%s", "Register");
            cbx->setLabel(syncMenu.strMCBRegister);
            syncMenu.bMCBRegister = false;
            requestUpdate();
            
            if (coVRMSController::instance()->isMaster())
            {
                lpc->doUnregistration();
            }
        }
    }
    else if (aButton == menuButtonHangUp)
    {
        if (coVRMSController::instance()->isMaster())
        {
            lpc->hangUpCall();
        }
    }
    else if (aButton == menuCheckboxPause)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->pauseCall(cbx->getState());
        }
    }
    else if (aButton == menuCheckboxMicMute)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setMicMute(cbx->getState());
        }
    }
    else if (aButton == menuCheckboxSpkrMute)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setSpeakerMute(cbx->getState());
        }
    }
    else if (aButton == menuCheckboxMicEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setMicrophoneEnabled(cbx->getState());
        }
    }
    else if (aButton == menuCheckboxEnableCamera)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setCallCameraEnabled(cbx->getState());
        }
    }
    else if (aButton == menuCheckboxSelfViewEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setCallSelfViewEnabled(cbx->getState());
        }
    }
    else if (aButton == menuPotiMicGain)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coPotiMenuItem* pot = static_cast<vrui::coPotiMenuItem*>(aButton);
            lpc->setMicGain(pot->getValue());
        }
    }
    else if (aButton == menuPotiPlaybackGain)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coPotiMenuItem* pot = static_cast<vrui::coPotiMenuItem*>(aButton);
            lpc->setPlaybackGain(pot->getValue());
        }
    }
    else if(aButton == menuCheckboxEchoCancellationEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setEchoCancellation(cbx->getState());
        }
    }

    else if(aButton == menuCheckboxEchoLimiterEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setEchoLimiter(cbx->getState());
        }
    }
    
    else if(aButton == menuCheckboxAudioJitterCompensation)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setAudioJitterCompensation(cbx->getState());
        }
    }
    else if(aButton == menuCheckboxVideoCaptureEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setVideoCaptureEnabled(cbx->getState());
        }
    }
    else if(aButton == menuCheckboxVideoDisplayEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setVideoDisplayEnabled(cbx->getState());
        }
    }
    else if(aButton == menuCheckboxVideoPreviewEnabled)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setVideoPreviewEnabled(cbx->getState());
        }
    }
    else if(aButton == menuCheckboxAutoAcceptVideo)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setAutoAcceptVideo(cbx->getState());
        }
    }
    else if(aButton == menuCheckboxAutoInitiateVideo)
    {
        if (coVRMSController::instance()->isMaster())
        {
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->setAutoInitiateVideo(cbx->getState());
        }
    }
    else if(aButton == menuCheckboxVideoJitterCompensation)
    {
        if (coVRMSController::instance()->isMaster())
        {       
            vrui::coCheckboxMenuItem* cbx = static_cast<vrui::coCheckboxMenuItem*>(aButton);
            lpc->getVideoJitterCompensation(cbx->getState());
        }
    }
    else 
    {
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
                if (coVRMSController::instance()->isMaster())
                {       
                    lpc->initiateCall((*contact).sipaddress);
                }

                sprintf(syncMenu.strMLBCallState, "%s", "Call: in progress");
                menuLabelCallState->setLabel(syncMenu.strMLBCallState);

                string strTemp =  "Addr: " + (*contact).sipaddress;
                sprintf(syncMenu.strMLBCallNameOfPartner, "%s", strTemp.c_str());
                menuLabelCallNameOfPartner->setLabel(syncMenu.strMLBCallNameOfPartner);
            }
        }
        
        {
            std::vector<vrui::coCheckboxMenuItem*>::iterator it = menuPlaybackDevices.begin();
            std::vector<std::string>::iterator devName = vecPlaybackSoundDevices.begin();

            while ((it != menuPlaybackDevices.end()) && (aButton != (*it)))
            {
                ++it;
                ++devName;
            }
            
            if (it != menuPlaybackDevices.end())
            {
                // uncheck every device
                for(std::vector<vrui::coCheckboxMenuItem*>::iterator forIt = menuPlaybackDevices.begin();
                    forIt != menuPlaybackDevices.end();
                    ++forIt) 
                {
                    (*forIt)->setState(false);
                }
                
                // check selected
                (*it)->setState(true);

                if (coVRMSController::instance()->isMaster())
                {
                    lpc->setCurrentPlaybackSoundDevice(*devName);
                }
            }
        }

        {
            std::vector<vrui::coCheckboxMenuItem*>::iterator it = menuCaptureDevices.begin();
            std::vector<std::string>::iterator devName = vecCaptureSoundDevices.begin();

            while ((it != menuCaptureDevices.end()) && (aButton != (*it)))
            {
                ++it;
                ++devName;
            }
            
            if (it != menuCaptureDevices.end())
            {
                // uncheck every device
                for(std::vector<vrui::coCheckboxMenuItem*>::iterator forIt = menuCaptureDevices.begin();
                    forIt != menuCaptureDevices.end();
                    ++forIt) 
                {
                    (*forIt)->setState(false);
                }
                
                // check selected
                (*it)->setState(true);

                if (coVRMSController::instance()->isMaster())
                {
                    lpc->setCurrentCaptureSoundDevice(*devName);
                }
            }
        }
        
        {
            std::vector<vrui::coCheckboxMenuItem*>::iterator it = menuVideoCaptureDevices.begin();
            std::vector<std::string>::iterator devName = vecVideoCaptureDevices.begin();

            while ((it != menuVideoCaptureDevices.end()) && (aButton != (*it)))
            {
                ++it;
                ++devName;
            }

            if (it != menuVideoCaptureDevices.end())
            {
                // uncheck every device
                for(std::vector<vrui::coCheckboxMenuItem*>::iterator forIt = menuVideoCaptureDevices.begin();
                    forIt != menuVideoCaptureDevices.end();
                    ++forIt) 
                {
                    (*forIt)->setState(false);
                }
                
                // check selected
                (*it)->setState(true);

                if (coVRMSController::instance()->isMaster())
                {
                    lpc->setCurrentVideoCaptureDevice(*devName);
                }
            }
        }
        
        {
            std::vector<vrui::coCheckboxMenuItem*>::iterator it = menuRingerDevices.begin();
            std::vector<std::string>::iterator devName = vecPlaybackSoundDevices.begin();

            while ((it != menuRingerDevices.end()) && (aButton != (*it)))
            {
                ++it;
                ++devName;
            }
            
            if (it != menuRingerDevices.end())
            {
                // uncheck every device
                for(std::vector<vrui::coCheckboxMenuItem*>::iterator forIt = menuRingerDevices.begin();
                    forIt != menuRingerDevices.end();
                    ++forIt) 
                {
                    (*forIt)->setState(false);
                }
                
                // check selected
                (*it)->setState(true);

                if (coVRMSController::instance()->isMaster())
                {
                    lpc->setCurrentRingerSoundDevice(*devName);
                }
            }
        } 
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
