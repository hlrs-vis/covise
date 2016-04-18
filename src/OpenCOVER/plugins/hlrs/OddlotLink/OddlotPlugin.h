/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Oddlot_PLUGIN_H
#define _Oddlot_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Oddlot Plugin (connection to the OpenDrive Road Editor)     **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Apr-16  v1	    				       		                             **
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
#include "oddlotMessageTypes.h"

namespace vrui
{
class coCheckboxMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxGroup;
class coButtonMenuItem;
}

class OddlotPlugin;
class OddlotParameter;

using namespace vrui;
using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class OddlotPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{
public:

    
    OddlotPlugin();
    ~OddlotPlugin();
    virtual bool init();
    static OddlotPlugin *instance()
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
    
    void message(int type, int len, const void *buf);
protected:
    static OddlotPlugin *plugin;

    ServerConnection *serverConn;
    ServerConnection *toOddlot;
    void handleMessage(Message *m);
    Message *msg;
};
#endif
