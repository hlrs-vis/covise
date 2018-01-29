/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Office_PLUGIN_H
#define _Office_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Office Plugin (connection to Microsoft Office)              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <net/covise_connect.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Group>
#include <stack>
#include <map>
#include <cover/ui/Owner.h>

namespace opencover
{
namespace ui
{
class Menu;
class Label;
class Group;
class Button;
class Input;
}
}

using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class OfficeConnection: public ui::Owner
{
public:
    std::string applicationType;
    std::string productName;
    OfficeConnection(ServerConnection *toOffice);
    ~OfficeConnection();
    ServerConnection *toOffice = nullptr;
    const ServerConnection *ServerOc = nullptr;
    void sendMessage(Message &m);
    void handleMessage(Message *m);
private:
    
    ui::Group *myFrame = nullptr;
    ui::Input *commandLine = nullptr;
    ui::Label *lastMessage = nullptr;
};

class officeList: public std::list<OfficeConnection *>
{
    Message *msg = nullptr;
    bool deletedConnection;
public:
    officeList();
    virtual ~officeList();
    void destroy(OfficeConnection *);
    void sendMessage(std::string product, Message &m);
    void checkAndHandleMessages();
};


class OfficePlugin : public coVRPlugin, public ui::Owner
{
public:
   
    enum MessageTypes
    {
        MSG_String = 700,
        MSG_ApplicationType = 701,
        MSG_PNGSnapshot = 702,
    };
    OfficePlugin();
    ~OfficePlugin();
    virtual bool init() override;
    static OfficePlugin *instance();

    // this will be called in PreFrame
    void preFrame() override;

    void destroyMenu();
    void createMenu();

    void sendMessage(Message &m);
    
    void message(int toWhom, int type, int len, const void *buf) override;
    void handleMessage(OfficeConnection *oc, Message *m);
    ui::Menu *menu = nullptr;;
    officeList officeConnections;
protected:
    static OfficePlugin *plugin;
    //coButtonMenuItem *addCameraButton;

    ServerConnection *serverConn = nullptr;
};
#endif
