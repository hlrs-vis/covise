/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: create windows with Qt
 **                                                                          **
 **                                                                          **
 ** Author: Martin Aumüller <aumueller@hlrs.de>
 **                                                                          **
\****************************************************************************/

#include "WindowTypeMesa.h"

#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

using namespace opencover;

WindowTypeMesaPlugin::WindowTypeMesaPlugin()
{
    fprintf(stderr, "WindowTypeMesaPlugin::WindowTypeMesaPlugin\n");
}

// this is called if the plugin is removed at runtime
WindowTypeMesaPlugin::~WindowTypeMesaPlugin()
{
    fprintf(stderr, "WindowTypeMesaPlugin::~WindowTypeMesaPlugin\n");
}

bool WindowTypeMesaPlugin::destroy()
{
    while (!m_windows.empty())
    {
        windowDestroy(m_windows.begin()->second.index);
    }
    return true;
}

bool WindowTypeMesaPlugin::update()
{
    return true;
}

bool WindowTypeMesaPlugin::windowCreate(int i)
{
    auto &conf = *coVRConfig::instance();
    auto it = m_windows.find(i);
    if (it != m_windows.end())
    {
        std::cerr << "WindowTypeQt: already managing window no. " << i << std::endl;
        return false;
    }

    auto &win = m_windows[i];
    win.index = i;

    win.buffer = new char[conf.windows[i].sx*conf.windows[i].sy*4];
    win.context = OSMesaCreateContext(GL_RGBA, NULL);
OSMesaMakeCurrent(win.context, win.buffer, GL_UNSIGNED_BYTE, conf.windows[i].sx, conf.windows[i].sy);

    coVRConfig::instance()->windows[i].window = new osgViewer::GraphicsWindowEmbedded(0,0,conf.windows[i].sx, conf.windows[i].sy);
    coVRConfig::instance()->windows[i].context = coVRConfig::instance()->windows[i].window;
    //std::cerr << "window " << i << ": ctx=" << coVRConfig::instance()->windows[i].context << std::endl;

    return true;
}

void WindowTypeMesaPlugin::windowCheckEvents(int num)
{
}

void WindowTypeMesaPlugin::windowUpdateContents(int num)
{
}

void WindowTypeMesaPlugin::windowDestroy(int num)
{
    auto it = m_windows.find(num);
    if (it == m_windows.end())
    {
        std::cerr << "WindowTypeQt: window no. " << num << " not managed by this plugin" << std::endl;
        return;
    }

    auto &conf = *coVRConfig::instance();
    conf.windows[num].context = nullptr;
    conf.windows[num].windowPlugin = nullptr;
    conf.windows[num].window = nullptr;

    auto &win = it->second;
    m_windows.erase(it);

}

COVERPLUGIN(WindowTypeMesaPlugin)
