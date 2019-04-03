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
    
    identity.sipaddress = coCoviseConfig::getEntry("sipaddress", "COVER.Plugin.VoIP.Identity", "sip:name@ip");
    identity.localport =coCoviseConfig::getInt("localport", "COVER.Plugin.VoIP.Identity", 5060);
    identity.displayname = coCoviseConfig::getEntry("displayname", "COVER.Plugin.VoIP.Identity", "displayname");
    identity.username = coCoviseConfig::getEntry("username", "COVER.Plugin.VoIP.Identity", "username");
    identity.password = coCoviseConfig::getEntry("password", "COVER.Plugin.VoIP.Identity", "password");
    
    std::vector<std::string> contactsRead = coCoviseConfig::getScopeNames("COVER.Plugin.VoIP", "contact");
    for(std::vector<std::string>::iterator it = contactsRead.begin(); it != contactsRead.end(); ++it) 
    {
        SIPIdentity newContact;
        newContact.sipaddress = *it;
        contacts.push_back(newContact);
    }
}

// ----------------------------------------------------------------------------
VoIPPlugin::~VoIPPlugin()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::~VoIPPlugin()" << endl;
#endif
    
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
    
    return true;
}

// ----------------------------------------------------------------------------
void VoIPPlugin::createMenu()
{
#ifdef VOIP_DEBUG
    cout << "VoIPPlugin::createMenu()" << endl;
#endif
    
    menuContactList = new vrui::coRowMenu("Contact List");
    
    for(std::vector<SIPIdentity>::iterator it = contacts.begin(); it != contacts.end(); ++it) 
    {
        vrui::coButtonMenuItem* menuButtonContact = new vrui::coButtonMenuItem((*it).sipaddress);
        menuContactList->add(menuButtonContact);
        menuContacts.push_back(menuButtonContact);
    }
    
    menuCall =  new vrui::coRowMenu("Call");
    
    menuButtonHangUp = new vrui::coButtonMenuItem("Hang Up");
    menuCall->add(menuButtonHangUp);
    menuCheckboxPause = new vrui::coCheckboxMenuItem("Pause", false);
    menuCall->add(menuCheckboxPause);
    menuCheckboxMute = new vrui::coCheckboxMenuItem("Mic Mute", false);
    menuCall->add(menuCheckboxMute);
    
    menuAudio =  new vrui::coRowMenu("Audio Settings");
    
    menuCheckboxEnableAudio = new vrui::coCheckboxMenuItem("Enable Audio", true);
    menuAudio->add(menuCheckboxEnableAudio);
    menuPotiMicLevel = new vrui::coPotiMenuItem("Mic Level", 0.0, 100.0, 75.0);
    menuAudio->add(menuPotiMicLevel);
    menuPotiVolLevel = new vrui::coPotiMenuItem("Volume Level", 0.0, 100.0, 75.0);
    menuAudio->add(menuPotiVolLevel);
    
    menuVideo =  new vrui::coRowMenu("Video Settings"); 
    
    menuCheckboxEnableVideo = new vrui::coCheckboxMenuItem("Enable Video", true);
    menuVideo->add(menuCheckboxEnableVideo);
    menuCheckboxEnableCamera = new vrui::coCheckboxMenuItem("Enable Camera", true);
    menuVideo->add(menuCheckboxEnableCamera);
    
    menuLabelSIPAddress = new vrui::coLabelMenuItem(identity.sipaddress);
    menuCheckboxRegister = new vrui::coCheckboxMenuItem("Register", false); 
    
    menuContactListItem = new vrui::coSubMenuItem("Contact List..."); 
    menuContactListItem->setMenu(menuContactList);
    
    menuCallItem = new vrui::coSubMenuItem("Call..."); 
    menuCallItem->setMenu(menuCall);
    
    menuAudioItem = new vrui::coSubMenuItem("Audio Settings..."); 
    menuAudioItem->setMenu(menuAudio);
    
    menuVideoItem = new vrui::coSubMenuItem("Video Settings..."); 
    menuVideoItem->setMenu(menuVideo);
    
    menuVoIP = new vrui::coRowMenu("VoIP");
    menuVoIP->add(menuLabelSIPAddress);
    menuVoIP->add(menuCheckboxRegister);
    menuVoIP->add(menuContactListItem);
    menuVoIP->add(menuCallItem);
    menuVoIP->add(menuAudioItem);
    menuVoIP->add(menuVideoItem);
    
    menuMainItem = new vrui::coSubMenuItem("VoIP");
    menuMainItem->setMenu(menuVoIP);
    
    cover->getMenu()->add(menuMainItem);
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
    delete menuCheckboxMute;
    delete menuCheckboxPause;
    delete menuButtonHangUp;
    delete menuCall;
    delete menuContactList;
    delete menuPotiMicLevel;
    delete menuPotiVolLevel;
    delete menuCheckboxEnableAudio;
    delete menuCheckboxEnableVideo;
    delete menuCheckboxEnableCamera;
    
    for(vector<vrui::coButtonMenuItem*>::iterator it = menuContacts.begin(); it != menuContacts.end(); ++it) 
    {
        delete *it;
    }
}

// ----------------------------------------------------------------------------

COVERPLUGIN(VoIPPlugin)
