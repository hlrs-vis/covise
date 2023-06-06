/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "TUIWebviewPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>

using namespace opencover;

WebviewPlugin::WebviewPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

// this is called if the plugin is removed at runtime
WebviewPlugin::~WebviewPlugin()
{
}

// here we get the size and the current center of the cube
void
WebviewPlugin::newInteractor(const RenderObject *container, coInteractor *i)
{
    (void)container;
    (void)i;
    fprintf(stderr, "WebviewPlugin::newInteractor\n");
}

void WebviewPlugin::addObject(const RenderObject *container,
                               osg::Group *root,
                               const RenderObject *obj, const RenderObject *normObj,
                               const RenderObject *colorObj, const RenderObject *texObj)
{
    (void)container;
    (void)obj;
    (void)normObj;
    (void)colorObj;
    (void)texObj;
    (void)root;
    fprintf(stderr, "WebviewPlugin::addObject\n");
}

void
WebviewPlugin::removeObject(const char *objName, bool replace)
{
    (void)objName;
    (void)replace;
    fprintf(stderr, "WebviewPlugin::removeObject\n");
}

void
WebviewPlugin::preFrame()
{
}

bool WebviewPlugin::init()
{
    WebviewTab = new coTUITab("Webview", coVRTui::instance()->mainFolder->getID());
    WebviewTab->setPos(0, 0);

    const std::string& n = "WebviewTest";
    int pID = WebviewTab->getID();
    Webview = new coTUIWebview(n, pID);
    Webview->setEventListener(this);
    return true;

}

bool WebviewPlugin::update() ///trigger new rendering of the window
{
    if(cover->frameTime()-lastChangeTime>=7) ///frameTime returns number of seconds
    {
        lastChangeTime = cover->frameTime(); ///time for url chancing is set to current time
        std::string url1 = ("http://www.9gag.com");
        std::string url2 = ("http://www.google.com");
        if(urlSwitch)
            Webview->setURL(url1); ///sends url1 to TUI
        else
            Webview->setURL(url2);
        urlSwitch = !urlSwitch;
        return true;
    }
    return false;
}

void WebviewPlugin::tabletEvent(coTUIElement* tUIItem)
{
    if (tUIItem == Webview)
    {
        //getLoadedURL
        cerr << "tUIItem == Webview" << endl;
        Webview->coTUIWebview::doSomething();
    }
}
COVERPLUGIN(WebviewPlugin)
