/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2007/8 HLRS **
 **                                                                           **
 ** Description: ACInterfacePlugin											            **
 **																		                     **
 **                                                                           **
 ** Author: Mario Baalcke	                                                   **
 **                                                                           **
 **                                                                           **
\****************************************************************************/
#include "ACInterfacePlugin.h"
#include <cover/coVRPluginSupport.h>
#include <net/message.h>

#include "gsoap_stubs/ws_OpenCOVEROpenCOVERProxy.h"

#include "gsoap_stubs/ws_OpenCOVER.nsmap"

#include <PluginUtil/PluginMessageTypes.h>

#include <stdsoap2.cpp>

ACInterfacePlugin *ACInterfacePlugin::plugin = 0;

bool ACInterfacePlugin::pickedObjChanged()
{

    return true;
}

bool ACInterfacePlugin::selectionChanged()
{
    std::list<osg::ref_ptr<osg::Node> > nodes = coVRSelectionManager::instance()->getSelectionList();

    if (nodes.empty())
        return true;

    osg::ref_ptr<osg::Node> node = nodes.front();

    ns1__sendMessage query;
    ns1__sendMessageResponse response;

    std::string list = coVRSelectionManager::generateNames(node.get());

    std::string body = "selected " + list; //node->getName();

    query.message = const_cast<char *>(body.c_str());

    service->sendMessage(&query, &response);

    return true;
}

ACInterfacePlugin::ACInterfacePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, selectionManager(0)
, service(0)
{
}

bool ACInterfacePlugin::init()
{
    if (plugin)
        return false;

    plugin = this;

    selectionManager = coVRSelectionManager::instance();
    selectionManager->addListener(this);

    service = new OpenCOVERProxy();
    service->OpenCOVERProxy_init(SOAP_IO_FLUSH, SOAP_IO_FLUSH);
    service->encodingStyle = 0;
    soap_set_omode(service, SOAP_XML_INDENT);

    service->soap_endpoint = soap_strdup(service, "http://localhost:32091/");

    return true;
}

// this is called if the plugin is removed at runtime
ACInterfacePlugin::~ACInterfacePlugin()
{
    selectionManager->removeListener(this);
}

void ACInterfacePlugin::preFrame()
{
}

void ACInterfacePlugin::message(int type, int len, const void *buf)
{

    if (service == 0)
        return;

    switch (type)
    {
    case PluginMessageTypes::HLRS_ACInterfaceSnapshotPath:
    {
        TokenBuffer tb((const char *)buf, len);
        //char * path;
        std::string path;

        tb >> path;

        fprintf(stderr,
                "ACInterfacePlugin::message() info: Got 'snapshot' message with path '%s'\n",
                path.c_str());

        ns1__sendMessage query;
        ns1__sendMessageResponse response;

        std::string body = "<screenshot><path>" + path + "</path></screenshot>";

        query.message = const_cast<char *>(body.c_str());

        service->sendMessage(&query, &response);

        break;
    }

    case PluginMessageTypes::HLRS_ACInterfaceModelLoadedPath:
    {
        TokenBuffer tb((const char *)buf, len);
        std::string path;

        tb >> path;

        fprintf(stderr,
                "ACInterfacePlugin::message() info: Got 'loaded' message with path '%s'\n",
                path.c_str());

        ns1__sendMessage query;
        ns1__sendMessageResponse response;

        std::string body = "<loaded>" + path + "</loaded>";

        query.message = const_cast<char *>(body.c_str());

        service->sendMessage(&query, &response);

        break;
    }

    default:
        break;
    }
}

COVERPLUGIN(ACInterfacePlugin);
