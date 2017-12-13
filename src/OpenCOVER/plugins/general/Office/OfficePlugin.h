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
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Group>
#include <stack>
#include <map>
#include <cover/coTabletUI.h>
#include <OpenVRUI/sginterface/vruiActionUserData.h>
// for AnnotationMessage:
#include <../../general/Annotation/AnnotationPlugin.h>

namespace vrui
{
class coCheckboxMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxGroup;
class coButtonMenuItem;
}

class OfficePlugin;

using namespace vrui;
using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class OfficeConnection: public coTUIListener
{
public:
    std::string applicationType;
    std::string productName;
    OfficeConnection(ServerConnection *toOffice);
    OfficeConnection(OfficeConnection *ServerOc);
    ~OfficeConnection();
    ServerConnection *toOffice;
    OfficeConnection *ServerOc;
    void sendMessage(Message &m);
    void handleMessage(Message *m);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
private:
    
    coTUILabel *productLabel;
    coTUIFrame *myFrame;
    coTUIEditField *commandLine;
    coTUILabel *lastMessage;
};
class officeList: public std::list<OfficeConnection *>
{
    Message *msg;
    bool deletedConnection;
public:
    officeList();
    virtual ~officeList();
    void destroy(OfficeConnection *);
    void sendMessage(std::string product, Message &m);
    void checkAndHandleMessages();
};


class OfficePlugin : public coVRPlugin, public coMenuListener, public coTUIListener
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
    virtual bool init();
    static OfficePlugin *instance()
    {
        return plugin;
    };

    // this will be called in PreFrame
    void preFrame();

    void destroyMenu();
    void createMenu();
    virtual void menuEvent(coMenuItem *aButton);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    coTUITab *revitTab;
    void sendMessage(Message &m);
    
    void message(int toWhom, int type, int len, const void *buf);
    void handleMessage(OfficeConnection *oc, Message *m);
    coTUITab *officeTab;
    officeList officeConnections;
protected:
    static OfficePlugin *plugin;
    coSubMenuItem *OfficeButton;
    //coButtonMenuItem *addCameraButton;

    ServerConnection *serverConn;

    Message *msg;
};
#endif
