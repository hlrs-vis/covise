/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: MADIconnect OpenCOVER Plugin (connects to MADI)              **
 **                                                                          **
 **                                                                          **
 ** Author: D. Wickeroth                                                     **
 **                                                                          **
 ** History:                                                                 **
 ** July 2025  v1                                                           **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "MADIconnect.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>

#include <config/CoviseConfig.h>

#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>

using namespace opencover;
using namespace std;
using covise::TokenBuffer;
using covise::coCoviseConfig;

MADIconnect::MADIconnect()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "MADIconnect\n");

    int port = coCoviseConfig::getInt("port", "COVER.Plugin.MADIconnect", 33333);    

    serverConn = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        cout << "MADI::Creating server connection on port " << port << endl;

        serverConn = new ServerConnection(port, 4711, Message::UNDEFINED);
        if (!serverConn->getSocket())
        {
            cout << "MADI::tried to open server Port " << port << endl;
            cout << "MADI::Creation of server failed!" << endl;
            cout << "MADI::Port-Binding failed! Port already bound?" << endl;
            delete serverConn;
            serverConn = NULL;
        }
        else
        {
            cover->watchFileDescriptor(serverConn->getSocket()->get_id());
        }
    }

    struct linger linger;
	linger.l_onoff = 0;
	linger.l_linger = 0;
	cout << "MADI::Set socket options..." << endl;
	if (serverConn)
	{
		setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

		cout << "MADI::Set server to listen mode..." << endl;
		serverConn->listen();
		if (!serverConn->is_connected()) // could not open server port
		{
			fprintf(stderr, "MADI::Could not open server port %d\n", port);
			delete serverConn;
			serverConn = NULL;
		}
	}

    // Create a unit cube to visualize the plugin    
    osg::Box *unitCube = new osg::Box(osg::Vec3(0, 0, 0), 1000.0f);
    osg::ShapeDrawable *unitCubeDrawable = new osg::ShapeDrawable(unitCube);

    // Declare a instance of the geode class:
    basicShapesGeode = new osg::Geode();
    basicShapesGeode->setName("MADIconnect");

    osg::Vec4 _color;
    _color.set(0.0, 0.0, 1.0, 1.0);
    unitCubeDrawable->setColor(_color);
    unitCubeDrawable->setUseDisplayList(false);

    // Add the unit cube drawable to the geode:
    basicShapesGeode->addDrawable(unitCubeDrawable);

    cover->getObjectsRoot()->addChild(basicShapesGeode.get());
}

bool MADIconnect::destroy()
{
    cover->getObjectsRoot()->removeChild(basicShapesGeode.get());
    return true;
}

// this is called if the plugin is removed at runtime
MADIconnect::~MADIconnect()
{
    fprintf(stderr, "Goodbye MADI\n");
    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
	delete serverConn;
	serverConn = NULL;
}

COVERPLUGIN(MADIconnect)
