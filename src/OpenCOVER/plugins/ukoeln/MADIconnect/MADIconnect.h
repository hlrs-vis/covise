/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MADI_CONNECT_PLUGIN_H
#define MADI_CONNECT_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: MADiConnect OpenCOVER Plugin (connects to MADI       )      **
 **                                                                          **
 **                                                                          **
 ** Author: D. Wickeroth                                                     **
 **                                                                          **
 ** History:                                                                 **
 ** July 2025  v1  				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/Geode>

#include <net/covise_connect.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Label.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Group.h>
#include <cover/ui/EditField.h>
#include <cover/ui/SelectionList.h>

using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class MADIconnect : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    enum MessageTypes
	{
		MSG_LOAD_NEURONS = 4700,
        MSG_SHOW_NEURONS = 4701,
        MSG_HIDE_NEURONS = 4702,
        MSG_COLOR_NEURONS = 4703,
        MSG_LOAD_VOLUME = 4704,
        MSG_SHOW_VOLUME = 4705,
        MSG_HIDE_VOLUME = 4706,
        MSG_VIEW_ALL = 4707,
        MSG_TEST = 4708,
    };

    MADIconnect();
    ~MADIconnect() override;

    static MADIconnect *instance()
    {
        return plugin;
    };
    virtual bool destroy();
    bool update() override;
	void preFrame() override;

    bool sendMessage(Message &m);

    ui::Menu *madiMenu = nullptr;
    ui::Action *testAction = nullptr;

private:    
    std::string dataPath;
    std::map<std::string, osg::ref_ptr<osg::Switch>> loadedNeurons;
    void sendTestMessage();
    bool showNeuron(const std::string &neuronName, bool show);

protected:
    static MADIconnect *plugin;

    ServerConnection *serverConn = nullptr;
    std::unique_ptr<ServerConnection> toMADI = nullptr;

    void handleMessage(Message *m);
    Message *msg = nullptr;
};
#endif
