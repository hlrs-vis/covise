/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRCOMMUNICATION_H
#define COVRCOMMUNICATION_H

/*! \file
 \brief  handle collaborative messages

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2001
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   July 2001
 */

namespace covise
{
class Message;
class UdpMessage;

}

#include "ui/Owner.h"
#include "coVRMessageSender.h"

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
namespace opencover
{
class VRBData;
class IData;
class coVRPartner;
class VrbMenu;
namespace ui
{
class Owner;
class Group;
class FileBrowser;
class Action;
class SelectionList;
};
class COVEREXPORT coVRCommunication: public vrb::regClassObserver, public opencover::coVRMessageSender
{
public:
	void init();
    static coVRCommunication *instance();

    ~coVRCommunication();
    void processRenderMessage(const char * key, const char * tmp);
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
    void handleVRB(covise::Message *msg);
	void handleUdp(covise::UdpMessage* msg);
    void setCurrentFile(const char *filename);
    virtual void update(vrb::clientRegClass *theChangedClass);

    void becomeMaster();
    covise::Message *waitForMessage(int messageType);
    std::unique_ptr<vrb::VrbClientRegistry> registry;
	std::unique_ptr<vrb::SharedStateManager> sharedStateManager;

	//add callback that is called after first contact with vrb
	void addOnConnectCallback(std::function<void(void)> function);
	//add callback that is called when vrb disconnects
	void addOnDisconnectCallback(std::function<void(void)> function);
	//set link to covise plugin function to get message from covise socket
	void setWaitMessagesCallback(std::function<std::vector<covise::Message*>(void)> cb);
	//set link to covise plugin function to handle a covise message
	void setHandleMessageCallback(std::function<void(covise::Message *)> cb);

	std::vector<covise::Message*> waitCoviseMessages();
	void handleCoviseMessage(covise::Message* m);
	//called from coVRFileManager to make sure cover->filemenue is initialized
	void initVrbFileMenu();
private:
    static coVRCommunication *s_instance;

	vrui::coNavInteraction* remoteNavInteraction = nullptr;;
    int randomID = 0;
    bool ignoreRemoteTransform = false;
    std::map<int, VRBData *> mfbData;
    std::unique_ptr<VrbMenu> m_vrbMenu;
    vrb::SessionID m_privateSessionID;
	std::vector<std::function<void(void)>> onConnectCallbacks;
	std::vector<std::function<void(void)>> onDisconnectCallbacks;
	//covise plugin callbacks
	std::function <std::vector<covise::Message*>(void)> waitMessagesCallback;
	std::function<void(covise::Message*)> handleMessageCallback;
    coVRCommunication();
	//inform interested parties about connention to vrb or covise
	void connected();
	//inform interested parties about disconnection from vrb or covise
	void disconnected();
    void toggleClientState(bool state);
    coVRPartner *me();
    const coVRPartner *me() const;

};
}
#endif
