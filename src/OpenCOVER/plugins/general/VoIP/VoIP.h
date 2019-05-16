/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VOIP_PLUGIN_H
#define _VOIP_PLUGIN_H

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <config/CoviseConfig.h>

#include <string>
#include <vector>
#include <functional>

#include "LinphoneClient.h"

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
class VoIPPlugin : public opencover::coVRPlugin,
                   public vrui::coMenuListener,
                   public opencover::coTUIListener
{
public:

    struct SIPServer
    {
        std::string sipaddress = "";
        std::string transport = "";
    };

    struct SIPIdentity
    {
        std::string sipaddress = "";
        int localport = 0;
        std::string displayname = "";
        std::string username="";
        std::string password="";
    };

    VoIPPlugin();
    ~VoIPPlugin();

    static VoIPPlugin* instance();

    virtual bool init();
    void msgCallback(LinphoneClientState oldState, LinphoneClientState currentState);
    virtual void menuEvent(vrui::coMenuItem *aButton);
                       
protected:

    void createMenu();
    void destroyMenu();

    static VoIPPlugin* plugin;
    std::function<void (LinphoneClientState, LinphoneClientState)> handler;
    
    vrui::coSubMenuItem* menuMainItem = nullptr;
    vrui::coSubMenuItem* menuAudioItem = nullptr;
    vrui::coSubMenuItem* menuVideoItem = nullptr;
    vrui::coSubMenuItem* menuCallItem = nullptr;
    vrui::coSubMenuItem* menuMixerItem = nullptr;
    vrui::coSubMenuItem* menuContactListItem = nullptr;
    vrui::coLabelMenuItem* menuLabelSIPAddress = nullptr;
    vrui::coLabelMenuItem* menuLabelCallNameOfPartner = nullptr;
    vrui::coLabelMenuItem* menuLabelCallState = nullptr;
    vrui::coRowMenu* menuVoIP = nullptr;
    vrui::coRowMenu* menuAudio = nullptr;
    vrui::coRowMenu* menuVideo = nullptr;
    vrui::coRowMenu* menuMixer = nullptr;
    vrui::coRowMenu* menuCall = nullptr;
    vrui::coRowMenu* menuContactList = nullptr;
    vrui::coButtonMenuItem* menuButtonHangUp = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxRegister = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxPause = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxMicMute = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxSpkrMute = nullptr;
    vrui::coPotiMenuItem* menuPotiMicLevel = nullptr;
    vrui::coPotiMenuItem* menuPotiVolLevel = nullptr;
    vrui::coPotiMenuItem* menuPotiMicGain = nullptr;
    vrui::coPotiMenuItem* menuPotiPlaybackGain = nullptr;
    //vrui::coCheckboxMenuItem* menuCheckboxEnableAudio = nullptr;
    //vrui::coCheckboxMenuItem* menuCheckboxEnableVideo = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxEnableCamera = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxMicEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxSelfViewEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxEchoCancellationEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxEchoLimiterEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxAudioJitterCompensation = nullptr;
    vrui::coLabelMenuItem* menuLabelCaptureDeviceList = nullptr;
    vrui::coLabelMenuItem* menuLabelPlaybackDeviceList = nullptr;
    vrui::coLabelMenuItem* menuLabelRingerDeviceList = nullptr;
    vrui::coLabelMenuItem* menuLabelVideoDeviceList = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxVideoCaptureEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxVideoDisplayEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxVideoPreviewEnabled = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxAutoAcceptVideo = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxAutoInitiateVideo = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxVideoJitterCompensation = nullptr;
    
    std::vector<vrui::coCheckboxMenuItem*> menuCaptureDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuPlaybackDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuRingerDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuMediaDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuVideoCaptureDevices;
    
    std::vector<vrui::coButtonMenuItem*> menuContacts;

    SIPServer server;
    SIPIdentity identity;
    std::vector<SIPIdentity> contacts;

    LinphoneClient* lpc;
};

// ----------------------------------------------------------------------------

#endif
