/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/common.h>
#include "vvWindows.h"

#include <config/CoviseConfig.h>

#include <vsg/core/Version.h>
#include <vsg/io/Logger.h>

#include "vvConfig.h"
#include "vvVIVE.h"

 #include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Menu.h"
#include "ui/View.h"

#include "vvPluginSupport.h"
#include "vvPluginList.h"
#include "vvMSController.h"
#include "vvViewer.h"

using namespace vive;

namespace {

    std::string pluginName(const std::string& windowType)
    {
        std::string p = windowType;
        p[0] = std::toupper(p[0]);
        p = "WindowType" + p;
        return p;
    }

}

vvWindows* vvWindows::s_instance = NULL;
vvWindows* vvWindows::instance()
{
    if (!s_instance)
        s_instance = new vvWindows;
    return s_instance;
}

vvWindows::vvWindows()
    : origVSize(NULL)
    , origHSize(NULL)
    , _firstTimeEmbedded(false)
{
    assert(!s_instance);
    vsg::debug("\nnew vvWindows\n");
    origVSize = NULL;
    origHSize = NULL;

    
    if (vv->viewOptionsMenu)
    {
        auto fs = new ui::Button(vv->viewOptionsMenu, "FullScreen");
        fs->setText("Full screen");
        fs->setState(false);
        fs->addShortcut("Ctrl+f");
        fs->addShortcut("Alt+f");
        fs->setCallback([this](bool state) {
            vvWindows::instance()->makeFullScreen(state);
            });
        fs->setVisible(false, ui::View::VR);
        m_fullScreenButton = fs;

        auto lfs = new ui::Action(vv->viewOptionsMenu, "LeaveFullScreen");
        lfs->setText("Exit full screen");
        lfs->setShortcut("Escape");
        lfs->setCallback([this]() {
            vvWindows::instance()->makeFullScreen(false);
            });
        lfs->setVisible(false);
    }
}

vvWindows::~vvWindows()
{ // don't use singletons in destructors
    destroy();
    delete[] origVSize;
    delete[] origHSize;

    s_instance = NULL;
}

bool
vvWindows::config()
{
   vsg::debug("\nvvWindows::config:", vvConfig::instance()->numWindows(), "windows\n");

    // load plugins for all window types on master and all slaves
    std::set<std::string> windowTypes;
    auto& conf = *vvConfig::instance();
    for (int i = 0; i < conf.numWindows(); i++)
    {
        auto type = conf.windows[i].type;
        if (!type.empty())
            windowTypes.insert(type);
    }
    if (vvMSController::instance()->isMaster())
    {
        vvMSController::SlaveData sdCount(sizeof(int));
        vvMSController::instance()->readSlaves(&sdCount);
        for (int s = 0; s < vvMSController::instance()->clusterSize() - 1; ++s)
        {
            int& count = *static_cast<int*>(sdCount.data[s]);
            for (int i = 0; i < count; ++i)
            {
                std::string t;
                vvMSController::instance()->readSlave(s, t);
                windowTypes.insert(t);
            }
        }
        size_t count = windowTypes.size();
        vvMSController::instance()->sendSlaves(&count, sizeof(count));
        for (auto type : windowTypes)
        {
            auto plugName = pluginName(type);
            plugName = vvMSController::instance()->syncString(plugName);
            vvPluginList::instance()->addPlugin(plugName.c_str(), vvPluginList::Window);
        }
    }
    else
    {
        size_t count = windowTypes.size();
        vvMSController::instance()->sendMaster(&count, sizeof(count));
        for (auto t : windowTypes)
            vvMSController::instance()->sendMaster(t);
        vvMSController::instance()->readMaster(&count, sizeof(count));
        for (int i = 0; i < count; ++i)
        {
            std::string plugName;
            plugName = vvMSController::instance()->syncString(plugName);
            vvPluginList::instance()->addPlugin(plugName.c_str(), vvPluginList::Window);
        }
    }
    

    origVSize = new int[vvConfig::instance()->numWindows()];
    origHSize = new int[vvConfig::instance()->numWindows()];
    for (int i = 0; i < vvConfig::instance()->numWindows(); i++)
    {
        origVSize[i] = -1;
        origHSize[i] = -1;
        if (!createWin(i))
            return false;
        //XXX ia->addRenderSurface(vv->windows[i].rs);
    }
    
    return true;
}

bool vvWindows::unconfig()
{
    auto& conf = *vvConfig::instance();

    for (int i = 0; i < conf.numWindows(); i++)
    {
        origVSize[i] = -1;
        origHSize[i] = -1;

        if (!destroyWin(i))
            return false;

        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowDestroy(i);
    }

    oldWidth.clear();
    oldHeight.clear();
    origWidth.clear();
    origHeight.clear();
    aspectRatio.clear();
    
    return true;
}

bool vvWindows::isFullScreen() const
{
    return m_fullscreen;
}

void vvWindows::makeFullScreen(bool state)
{
    m_fullscreen = state;
     if (m_fullScreenButton)
        m_fullScreenButton->setState(state);

    auto& conf = *vvConfig::instance();
    for (int i = 0; i < conf.numWindows(); ++i)
    {
        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowFullScreen(i, state);
    }
}

void vvWindows::destroy()
{
    
    unconfig();
    vvPluginList::instance()->unloadAllPlugins(vvPluginList::Window);
    
}

void
vvWindows::update()
{
    
    auto& conf = *vvConfig::instance();
    const auto& numWin = conf.numWindows();
    for (int i = 0; i < conf.numWindows(); ++i)
    {
        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowCheckEvents(i);
    }

    if (vvConfig::instance()->numWindows() <= 0 || vvConfig::instance()->numScreens() <= 0)
        return;
    // resize windows, ignore in multiple windows mode
    if (vvConfig::instance()->numWindows() > 1)
        return;

    if (oldWidth.size() < numWin) {
        oldWidth.resize(numWin, -1);
        oldHeight.resize(numWin, -1);
        aspectRatio.resize(numWin, -1.f);
        origWidth.resize(numWin, -1);
        origHeight.resize(numWin, -1);
    }

    for (int win = 0; win < numWin; ++win)
    {
        //static int oldX=-1;
        const vsg::ref_ptr<vsg::WindowTraits> traits = vvConfig::instance()->windows[win].window->traits();
        if (!traits.get())
            return;
        

        auto& screen = vvConfig::instance()->screens[0];
        float initialWidth = screen.hsize;
        float initialHeight = screen.vsize;

        int currentW = traits->width;
        int currentH = traits->height;

        if (oldWidth[win] == -1)
        {
            oldWidth[win] = currentW;
            oldHeight[win] = currentH;
            origWidth[win] = currentW;
            origHeight[win] = currentH;
            if (screen.hsize <= 0)
            {
                aspectRatio[win] = 1;
            }
            else
            {
                aspectRatio[win] = screen.configuredVsize / screen.configuredHsize;
            }
        }

        if (oldWidth[win] != currentW || oldHeight[win] != currentH || _firstTimeEmbedded)
        {
            if ((vvVIVE::instance()->parentWindow) && (vvConfig::instance()->windows[win].embedded))
            {
                float width = (float)origHSize[0];
                float height = (float)origVSize[0];
                if (width > 0 && height > 0)
                {
                    float vsize = screen.hsize * (height / width);
                    screen.vsize = vsize;
                }
                _firstTimeEmbedded = false;
            }
            else if (origWidth[win] != 0)
            {
                // change height or width so that configured screen area remains visible
                if (currentW > 0 && currentH > aspectRatio[win] * currentW)
                {
                    screen.hsize = screen.configuredHsize;
                    screen.vsize = (((screen.configuredHsize * (aspectRatio[win])) / origHeight[win]) * currentH) * origWidth[win] / currentW;
                }
                else
                {
                    screen.hsize = (((screen.configuredVsize / (aspectRatio[win])) / origWidth[win]) * currentW) * origHeight[win] / currentH;
                    screen.vsize = screen.configuredVsize;
                }
            }
            vvConfig::instance()->windows[win].sx = currentW;
            vvConfig::instance()->windows[win].sy = currentH;
            oldWidth[win] = currentW;
            oldHeight[win] = currentH;
        }
    }
    
}

void
vvWindows::updateContents()
{
    
    auto& conf = *vvConfig::instance();
    for (int i = 0; i < vvConfig::instance()->numWindows(); ++i)
    {
        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowUpdateContents(i);
    }
    
}

/*************************************************************************/
bool
vvWindows::createWin(int i)
{
    vsg::debug("vvWindows::createWin", i);
   

    auto& conf = *vvConfig::instance();
    auto& win = conf.windows[i];

    if (conf.windows[i].stereo && !conf.windows[i].type.empty())
    {
        std::cerr << "vvWindows: ignoring window type " << conf.windows[i].type << ", because stereo visual was requested" << std::endl;
        conf.windows[i].type.clear();
    }

    if (!conf.windows[i].type.empty())
    {
        auto windowPlug = vvPluginList::instance()->getPlugin(pluginName(conf.windows[i].type).c_str());
        if (windowPlug)
        {
            if (windowPlug->windowCreate(i))
            {
                conf.windows[i].windowPlugin = windowPlug;
            }
            else
            {
                std::cerr << "vvWindows: plugin failed to create window " << i << " of type " << conf.windows[i].type << std::endl;
                conf.windows[i].type.clear();
            }
        }
        else
        {
            std::cerr << "vvWindows: no plugin for window " << i << " of type " << conf.windows[i].type << std::endl;
            conf.windows[i].type.clear();
        }
    }

    windowStruct& ws = vvConfig::instance()->windows[i];
    if (conf.windows[i].type.empty())
    {
        auto traits = vsg::WindowTraits::create();
        traits->windowTitle = "VIVE";
        
        traits->fullscreen = false;
        traits->x = ws.ox;
        traits->y = ws.oy;
        traits->width = ws.sx;
        traits->height =ws.sy;
        traits->decoration = ws.decoration;
        if (traits->decoration == false)
        {
            traits->overrideRedirect = true;
        }
        // = ws.resize;
        //traits->pbuffer = ws.pbuffer;

        if ((vvVIVE::instance()->parentWindow) && (ws.embedded))
        {
            //traits->pbuffer = false;
            //_eventReceiver = new vvEventReceiver(7878, 0);
            _firstTimeEmbedded = true;
            /*
#if defined(WIN32)
            traits->nativeWindow = new osgViewer::GraphicsWindowWin32::WindowData(vvVIVE::instance()->parentWindow);
#elif defined(__APPLE__) && !defined(USE_X11)
            traits->inheritedWindowData = new osgViewer::GraphicsWindowCocoa::WindowData(vvVIVE::instance()->parentWindow);
#else
#if defined(USE_X11)
            traits->inheritedWindowData = new osgViewer::GraphicsWindowX11::WindowData(vvVIVE::instance()->parentWindow);
#endif
#endif*/
        }

        if (traits->nativeWindow.has_value())
            traits->decoration = false;

        if (traits->decoration == false)
        {
            traits->overrideRedirect = true;
        }

        const int pipeNum = ws.pipeNum;
        if (ws.screenNum >= 0)
        {
            traits->screenNum = ws.screenNum;
        }
        if (vvConfig::instance()->useDisplayVariable() || vvConfig::instance()->pipes[pipeNum].useDISPLAY)
        {
            traits->display = ":0.0";
            traits->screenNum = 0;
            if (const char* disp = getenv("DISPLAY"))
            {
                traits->display = disp;
            }
        }
        else
        {
            if (ws.screenNum >= 0)
            {
                traits->screenNum = ws.screenNum;
            }
            traits->display = vsg::make_string(vvConfig::instance()->pipes[pipeNum].x11DisplayHost,":", vvConfig::instance()->pipes[pipeNum].x11DisplayNum,".", vvConfig::instance()->pipes[pipeNum].x11ScreenNum);
        }

        /*
        if (vvConfig::instance()->doMultisample())
        {
            traits->samples = vvConfig::instance()->getMultisampleSamples();
            traits->sampleBuffers = vvConfig::instance()->getMultisampleSampleBuffers();
        }*/

        ws.doublebuffer = true;

        ws.window = vsg::Window::create(traits);
        ws.window->clearColor().set(0, 0, 0, 1);


        vvViewer::instance()->addWindow(ws.window);
    }

    if (!ws.window.get())
    {
        return false;
    }
    bool syncToVBlankConfigured = false;
    bool syncToVBlank = covise::coCoviseConfig::isOn("COVER.SyncToVBlank", false, &syncToVBlankConfigured);
    if (syncToVBlankConfigured)
    {
        if (auto win = ws.window)
        {
            fprintf(stderr, "TODO implement sync to VBlank syncToVBlank=%d\n", (int)syncToVBlank);
            //win->setSyncToVBlank(syncToVBlank);
        }
    }


    if (vv->debugLevel(4))
        fprintf(stderr, "vvWindows::createWin %d --- finished\n", i);
    return true;
}

bool vvWindows::destroyWin(int i)
{
    auto& conf = *vvConfig::instance();
    auto& win = conf.windows[i];

    win.window = nullptr;
    return true;
}
