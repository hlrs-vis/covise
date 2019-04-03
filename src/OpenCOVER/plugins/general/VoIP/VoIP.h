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

#include <linphone/core.h>

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
class VoIPPlugin : public opencover::coVRPlugin,
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

protected:

    void createMenu();
    void destroyMenu();

    static VoIPPlugin* plugin;

    vrui::coSubMenuItem* menuMainItem = nullptr;
    vrui::coSubMenuItem* menuAudioItem = nullptr;
    vrui::coSubMenuItem* menuVideoItem = nullptr;
    vrui::coSubMenuItem* menuCallItem = nullptr;
    vrui::coSubMenuItem* menuContactListItem = nullptr;
    vrui::coLabelMenuItem* menuLabelSIPAddress = nullptr;
    vrui::coRowMenu* menuVoIP = nullptr;
    vrui::coRowMenu* menuAudio = nullptr;
    vrui::coRowMenu* menuVideo = nullptr;
    vrui::coRowMenu* menuCall = nullptr;
    vrui::coRowMenu* menuContactList = nullptr;
    vrui::coButtonMenuItem* menuButtonHangUp = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxRegister = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxPause = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxMute = nullptr;
    vrui::coPotiMenuItem* menuPotiMicLevel = nullptr;
    vrui::coPotiMenuItem* menuPotiVolLevel = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxEnableAudio = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxEnableVideo = nullptr;
    vrui::coCheckboxMenuItem* menuCheckboxEnableCamera = nullptr;
    
    std::vector<vrui::coButtonMenuItem*> menuContacts;

    SIPServer server;
    SIPIdentity identity;
    std::vector<SIPIdentity> contacts;
};

// ----------------------------------------------------------------------------

#endif
