/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
namespace covise
{
class Message;
class UdpMessage;

}

#include "ui/Owner.h"
#include "vvMessageSender.h"

#include <net/message_types.h>
#include <vrb/SessionID.h>
#include <vrb/client/ClientRegistryClass.h>
#include <vrb/client/SharedState.h>

#include <map>
namespace vrui
{
class coNavInteraction;
}
namespace vrb {
class SharedStateManager;
class VrbClientRegistry;
}
namespace vive
{
class VRBData;
class IData;
class vvPartner;
class VrbMenu;
namespace ui
{
class Owner;
class Group;
class FileBrowser;
class Action;
class SelectionList;
}
class VVCORE_EXPORT vvCommunication: public vrb::regClassObserver, public vive::vvMessageSender
{
public:
	void init();
    static vvCommunication *instance();

    ~vvCommunication();
    void processARVideoFrame(const char * key, const char * tmp);
    void processVRBMessage(covise::TokenBuffer &tb);

    bool collaborative(); // returns true, if in collaborative mode
    bool isMaster(); // returns true, if we are master

    static const char *getHostname();
    static const char *getHostaddress();
    static std::string getUsername();
    int getID();
    const vrb::SessionID &getPrivateSessionID() const;
    const vrb::SessionID &getSessionID() const;
    const vrb::SessionID &getUsedSessionID() const;

    void setSessionID(const vrb::SessionID &id);

    void saveSessionFile(covise::TokenBuffer &tb);
    void loadSessionFile(const std::string &fileName);

    int getNumberOfPartners();
    void setFBData(IData *data);
    void handleVRB(const covise::Message &msg);
    void handleUdp(covise::UdpMessage *msg);
    virtual void update(vrb::clientRegClass *theChangedClass);

    void becomeMaster();
    covise::Message *waitForMessage(int messageType);
    std::unique_ptr<vrb::VrbClientRegistry> registry;
	std::unique_ptr<vrb::SharedStateManager> sharedStateManager;
    enum class Notification{
        Connected, Disconnected, SessionChanged, PartnerJoined, PartnerLeft
    };
    void subscribeNotification(Notification type, const std::function<void(void)> &function);
	//set link to covise plugin function to get message from covise socket
	void setWaitMessagesCallback(std::function<std::vector<covise::Message*>(void)> cb);
	//set link to covise plugin function to handle a covise message
	void setHandleMessageCallback(std::function<void(covise::Message *)> cb);

	std::vector<covise::Message*> waitCoviseMessages();
	void handleCoviseMessage(covise::Message* m);
	//called from vvFileManager to make sure vv->filemenue is initialized
	void initVrbFileMenu();
private:
    static vvCommunication *s_instance;

	vrui::coNavInteraction* remoteNavInteraction = nullptr;
    int randomID = 0;
    bool ignoreRemoteTransform = false;
    std::map<int, VRBData *> mfbData;
    std::unique_ptr<VrbMenu> m_vrbMenu;
    vrb::SessionID m_privateSessionID;
	//covise plugin callbacks
	std::map<Notification, std::vector<std::function<void(void)>>> notificationSubscriptions;
	std::function <std::vector<covise::Message*>(void)> waitMessagesCallback;
	std::function<void(covise::Message*)> handleMessageCallback;
    vvCommunication();
	//inform interested parties about connention to vrb or covise
	void connected();
	//inform interested parties about disconnection from vrb or covise
	void disconnected();
    void toggleClientState(bool state);
    vvPartner *me();
    const vvPartner *me() const;
    void callSubscriptions(Notification type);

};
}

