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
#include "NotifyDialog.h"

// ----------------------------------------------------------------------------

#define SYNCBUFFER_SIZE 64

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
class VoIPPlugin : public opencover::coVRPlugin,
                   public vrui::coMenuListener,
                   public opencover::coTUIListener,
                   public vrui::coValuePotiActor
{
public:

    struct SIPServer
    {
        std::string sipaddress = "";
        std::string transport = "";
        int portMinAudio = 0;
        int portMaxAudio = 0;
        int portMinVideo = 0;
        int portMaxVideo = 0;
    };

    struct SIPIdentity
    {
        std::string sipaddress = "";
        int localport = 0;
        std::string displayname = "";
        std::string username="";
        std::string password="";
    };

    struct SyncMenu
    {
        char strNODQuestion1[SYNCBUFFER_SIZE] = {0}; // nodNotifyQuestion->setText()
        char strNODQuestion2[SYNCBUFFER_SIZE] = {0};
        char strNODQuestion3[SYNCBUFFER_SIZE] = {0};

        char strMLBCallNameOfPartner[SYNCBUFFER_SIZE] = {0}; // menuLabelCallNameOfPartner->setLabel()
        char strMCBRegister[SYNCBUFFER_SIZE] = {0}; // menuCheckboxRegister->setLabel()
        char strMLBCallState[SYNCBUFFER_SIZE] = {0}; // menuLabelCallState->setLabel()
        
        bool bNODNotifyQuestionShow  = false; // nodNotifyQuestion->show()
        bool bMCBRegister = false; //  menuCheckboxRegister->setState()
    };
    
    VoIPPlugin();
    ~VoIPPlugin();

    static VoIPPlugin* instance();

    virtual bool init();
    virtual void menuEvent(vrui::coMenuItem *aButton);

    void msgCallback(LinphoneClientState oldState, LinphoneClientState currentState);
    void msgNotify(std::string);

    bool update();
    
protected:

    static VoIPPlugin* plugin;

    // tools to sync master/slaves
    
    SyncMenu syncMenu;
    bool bRequestUpdate = false;
    void requestUpdate() { bRequestUpdate = true; }

    bool bPostInitCompleted = false;
    bool postInit();

    // VR menu
    
    void createMenu();
    void updateMenu();
    void destroyMenu();

    void potiValueChanged(float oldValue, float newValue, vrui::coValuePoti *poti, int context = -1);

    std::function<void (LinphoneClientState, LinphoneClientState)> handler;
    std::function<void (std::string)> notifier;

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

    opencover::NotifyDialog* nodNotifyQuestion = nullptr;
    
    std::vector<vrui::coCheckboxMenuItem*> menuCaptureDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuPlaybackDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuRingerDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuMediaDevices;
    std::vector<vrui::coCheckboxMenuItem*> menuVideoCaptureDevices;
    
    std::vector<vrui::coButtonMenuItem*> menuContacts;

    std::vector<std::string> vecCaptureSoundDevices;
    std::vector<std::string> vecPlaybackSoundDevices;
    std::vector<std::string> vecVideoCaptureDevices;
    
    SIPServer server;
    SIPIdentity identity;
    std::vector<SIPIdentity> contacts;

    bool bNotified = false;
    
    LinphoneClient* lpc = nullptr;
};

// ----------------------------------------------------------------------------

#endif
