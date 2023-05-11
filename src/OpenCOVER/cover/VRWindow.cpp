/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRWindow.C 				*
 *									*
 *	Description		window class for opening up to 		*
 *				4 stereo windows			*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			20.07.2004				*
 *									*
 ************************************************************************/

#include <util/common.h>
#include "VRWindow.h"
#include "coCommandLine.h"

#include <config/CoviseConfig.h>
#include "coVRConfig.h"
#include "OpenCOVER.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Menu.h"
#include "ui/View.h"

#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRMSController.h"
#include <osgViewer/GraphicsWindow>
#if defined(WIN32)
#include <osgViewer/api/Win32/GraphicsWindowWin32>
#elif defined(__APPLE__) && !defined(USE_X11)
#include <osgViewer/api/Cocoa/GraphicsWindowCocoa>
#else
#if defined(USE_X11)
#include <osgViewer/api/X11/GraphicsWindowX11>
#endif
#endif

using namespace opencover;

namespace {

std::string pluginName(const std::string &windowType)
{
    std::string p= windowType;
    p[0] = std::toupper(p[0]);
    p = "WindowType"+p;
    return p;
}

}

VRWindow *VRWindow::s_instance = NULL;
VRWindow *VRWindow::instance()
{
    if (!s_instance)
        s_instance = new VRWindow;
    return s_instance;
}

VRWindow::VRWindow()
    : origVSize(NULL)
    , origHSize(NULL)
    , _firstTimeEmbedded(false)
    , _eventReceiver(NULL)
{
    assert(!s_instance);
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "\nnew VRWindow\n");
    }
    origVSize = NULL;
    origHSize = NULL;


    if (cover->viewOptionsMenu)
    {
        auto fs = new ui::Button(cover->viewOptionsMenu, "FullScreen");
        fs->setText("Full screen");
        fs->setState(false);
        fs->addShortcut("Ctrl+f");
        fs->addShortcut("Alt+f");
        fs->setCallback([this](bool state){
            VRWindow::instance()->makeFullScreen(state);
        });
        fs->setVisible(false, ui::View::VR);
        m_fullScreenButton = fs;

        auto lfs = new ui::Action(cover->viewOptionsMenu, "LeaveFullScreen");
        lfs->setText("Exit full screen");
        lfs->setShortcut("Escape");
        lfs->setCallback([this](){
            VRWindow::instance()->makeFullScreen(false);
        });
        lfs->setVisible(false);
    }
}

VRWindow::~VRWindow()
{ // don't use singletons in destructors
    destroy();
    delete[] origVSize;
    delete[] origHSize;

    s_instance = NULL;
}

bool
VRWindow::config()
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\nVRWindow::config: %d windows\n", coVRConfig::instance()->numWindows());
    }

    // load plugins for all window types on master and all slaves
    std::set<std::string> windowTypes;
    auto &conf = *coVRConfig::instance();
    for (int i = 0; i < conf.numWindows(); i++)
    {
        auto type = conf.windows[i].type;
        if (!type.empty())
            windowTypes.insert(type);
    }

    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::SlaveData sdCount(sizeof(int));
        coVRMSController::instance()->readSlaves(&sdCount);
        for (int s=0; s<coVRMSController::instance()->clusterSize()-1; ++s)
        {
            int &count = *static_cast<int *>(sdCount.data[s]);
            for (int i=0; i<count; ++i)
            {
                std::string t;
                coVRMSController::instance()->readSlave(s, t);
                windowTypes.insert(t);
            }
        }
        int count = windowTypes.size();
        coVRMSController::instance()->sendSlaves(&count, sizeof(count));
        for (auto type: windowTypes)
        {
            auto plugName = pluginName(type);
            plugName = coVRMSController::instance()->syncString(plugName);
            coVRPluginList::instance()->addPlugin(plugName.c_str(), coVRPluginList::Window);
        }
    }
    else
    {
        int count = windowTypes.size();
        coVRMSController::instance()->sendMaster(&count, sizeof(count));
        for (auto t: windowTypes)
            coVRMSController::instance()->sendMaster(t);
        coVRMSController::instance()->readMaster(&count, sizeof(count));
        for (int i=0; i<count; ++i)
        {
            std::string plugName;
            plugName = coVRMSController::instance()->syncString(plugName);
            coVRPluginList::instance()->addPlugin(plugName.c_str(), coVRPluginList::Window);
        }
    }

    origVSize = new int[coVRConfig::instance()->numWindows()];
    origHSize = new int[coVRConfig::instance()->numWindows()];
    for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
    {
        origVSize[i] = -1;
        origHSize[i] = -1;
        if (!createWin(i))
            return false;
        //XXX ia->addRenderSurface(cover->windows[i].rs);
    }

    return true;
}

bool VRWindow::unconfig()
{
    auto &conf = *coVRConfig::instance();

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

bool VRWindow::isFullScreen() const
{
    return m_fullscreen;
}

void VRWindow::makeFullScreen(bool state)
{
    m_fullscreen = state;
    if (m_fullScreenButton)
        m_fullScreenButton->setState(state);

    auto &conf = *coVRConfig::instance();
    for (int i=0; i<conf.numWindows(); ++i)
    {
        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowFullScreen(i, state);
    }
}

void VRWindow::destroy()
{
    unconfig();
    coVRPluginList::instance()->unloadAllPlugins(coVRPluginList::Window);
}

void
VRWindow::update()
{
    auto &conf = *coVRConfig::instance();
    const auto &numWin = conf.numWindows();
    for (int i=0; i<conf.numWindows(); ++i)
    {
        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowCheckEvents(i);
    }

    if (coVRConfig::instance()->numWindows() <= 0 || coVRConfig::instance()->numScreens() <= 0)
        return;
    // resize windows, ignore in multiple windows mode
    if (coVRConfig::instance()->numWindows() > 1)
        return;

    if (oldWidth.size() < numWin) {
        oldWidth.resize(numWin, -1);
        oldHeight.resize(numWin, -1);
        aspectRatio.resize(numWin, -1.f);
        origWidth.resize(numWin, -1);
        origHeight.resize(numWin, -1);
    }

    for (int win=0; win<numWin; ++win)
    {
        //static int oldX=-1;
        const osg::GraphicsContext::Traits *traits = NULL;
        if (coVRConfig::instance()->windows[win].context)
            traits = coVRConfig::instance()->windows[win].context->getTraits();
        if (!traits)
            return;
        /*
        int currentX = traits->x;
        if(currentX!=oldX)
        {
            cerr<< "X pos: " << currentX << endl;
        }
        */

        auto &screen = coVRConfig::instance()->screens[0];
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
                aspectRatio[win] = screen.configuredVsize/screen.configuredHsize;
            }
        }

        if (oldWidth[win] != currentW || oldHeight[win] != currentH || _firstTimeEmbedded)
        {
            if ((OpenCOVER::instance()->parentWindow) && (coVRConfig::instance()->windows[win].embedded))
            {
                float width = (float)origHSize[0];
                float height = (float)origVSize[0];
                if(width >0 && height > 0)
                {
                    float vsize = screen.hsize * (height / width);
                    screen.vsize = vsize;
                }
                _firstTimeEmbedded = false;
            }
            else if (origWidth[win] != 0)
            {
                // change height or width so that configured screen area remains visible
                if (currentW > 0 && currentH > aspectRatio[win]*currentW)
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
            coVRConfig::instance()->windows[win].sx = currentW;
            coVRConfig::instance()->windows[win].sy = currentH;
            oldWidth[win] = currentW;
            oldHeight[win] = currentH;
        }
    }
}

void
VRWindow::updateContents()
{
    auto &conf = *coVRConfig::instance();
    for (int i=0; i<coVRConfig::instance()->numWindows(); ++i)
    {
        if (conf.windows[i].windowPlugin)
            conf.windows[i].windowPlugin->windowUpdateContents(i);
    }
}

/*************************************************************************/
bool
VRWindow::createWin(int i)
{
    if (cover->debugLevel(4))
    {
        fprintf(stderr, "VRWindow::createWin %d\n", i);
    }

    auto &conf = *coVRConfig::instance();
    auto &win = conf.windows[i];
    /*   int multisample_feature, stereo_feature;

      /// colle
      static int  frameBufAttr[64];
      int pos = 0;

      /// open a pfPipeWindow*/

    bool opengl3 = false;
    if(!coVRConfig::instance()->glVersion.empty())
    {
        const double ver = atof(coVRConfig::instance()->glVersion.c_str());
        if (ver >= 3.0)
            opengl3 = true;
        opengl3 = covise::coCoviseConfig::isOn("COVER.WindowConfig.OpenGL3", opengl3);
        if (cover->debugLevel(1))
        {
            std::cerr << "creating window " << i << " with GL context version " << coVRConfig::instance()->glVersion << ", OpenGL3=" << opengl3 << std::endl;
        }
    }

    if (conf.windows[i].stereo && !conf.windows[i].type.empty())
    {
        std::cerr << "VRWindow: ignoring window type " << conf.windows[i].type << ", because stereo visual was requested" << std::endl;
        conf.windows[i].type.clear();
    }

    if (!conf.windows[i].type.empty())
    {
        auto windowPlug = coVRPluginList::instance()->getPlugin(pluginName(conf.windows[i].type).c_str());
        if (windowPlug)
        {
            if (windowPlug->windowCreate(i))
            {
                conf.windows[i].windowPlugin = windowPlug;
            }
            else
            {
                std::cerr << "VRWindow: plugin failed to create window " << i << " of type " << conf.windows[i].type << std::endl;
                conf.windows[i].type.clear();
            }
        }
        else
        {
            std::cerr << "VRWindow: no plugin for window " << i << " of type " << conf.windows[i].type << std::endl;
            conf.windows[i].type.clear();
        }
    }

    if (conf.windows[i].type.empty())
    {
        osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
        traits->x = coVRConfig::instance()->windows[i].ox;
        traits->y = coVRConfig::instance()->windows[i].oy;
        traits->width = coVRConfig::instance()->windows[i].sx;
        traits->height = coVRConfig::instance()->windows[i].sy;
        traits->windowDecoration = coVRConfig::instance()->windows[i].decoration;
        if(traits->windowDecoration == false)
        {
            traits->overrideRedirect = true;
        }
        traits->supportsResize = coVRConfig::instance()->windows[i].resize;
        traits->pbuffer = coVRConfig::instance()->windows[i].pbuffer;
        if(!coVRConfig::instance()->glVersion.empty())
        {
            traits->glContextVersion = coVRConfig::instance()->glVersion;
        }
        if (!coVRConfig::instance()->glProfileMask.empty())
        {
            traits->glContextProfileMask = std::stoi(coVRConfig::instance()->glProfileMask);
        }
        if (!coVRConfig::instance()->glContextFlags.empty())
        {
            traits->glContextFlags = std::stoi(coVRConfig::instance()->glContextFlags);
        }
        traits->windowName = "OpenCOVER";

        if ((OpenCOVER::instance()->parentWindow) && (coVRConfig::instance()->windows[i].embedded))
        {
            traits->pbuffer = false;
            _eventReceiver = new EventReceiver(7878, 0);
            _firstTimeEmbedded = true;
#if defined(WIN32)
            traits->inheritedWindowData = new osgViewer::GraphicsWindowWin32::WindowData(OpenCOVER::instance()->parentWindow);
#elif defined(__APPLE__) && !defined(USE_X11)
            traits->inheritedWindowData = new osgViewer::GraphicsWindowCocoa::WindowData(OpenCOVER::instance()->parentWindow);
#else
#if defined(USE_X11)
            traits->inheritedWindowData = new osgViewer::GraphicsWindowX11::WindowData(OpenCOVER::instance()->parentWindow);
#endif
#endif
        }

        if (traits->inheritedWindowData != NULL)
            traits->windowDecoration = false;

        if(traits->windowDecoration == false)
        {
            traits->overrideRedirect = true;
        }

        const int pipeNum = coVRConfig::instance()->windows[i].pipeNum;
        if (coVRConfig::instance()->windows[i].screenNum >= 0)
        {
            traits->screenNum = coVRConfig::instance()->windows[i].screenNum;
        }
        if (coVRConfig::instance()->useDisplayVariable() || coVRConfig::instance()->pipes[pipeNum].useDISPLAY)
        {
            traits->displayNum = 0;
            traits->screenNum = 0;
            if (const char *disp = getenv("DISPLAY"))
            {
                char *display=new char[strlen(disp)+1];
                strcpy(display,disp);
                const char *host = display;
                char *p = strchr(display, ':');
                if (p)
                {
                    *p = '\0';
                    ++p;
                    char *server = p;
                    const char *screen = NULL;
                    p = strchr(server, '.');
                    if (p)
                    {
                        *p = '\0';
                        ++p;
                        screen = p;
                    }
                    if (host && strlen(host) > 0)
                        traits->hostName = host;
                    traits->displayNum = server ? atoi(server) : 0;
                    traits->screenNum = screen ? atoi(screen) : 0;
                }
                delete[] display;
            }
        }
        else
        {
            const std::string &host = coVRConfig::instance()->pipes[pipeNum].x11DisplayHost;
            if (!host.empty())
                traits->hostName = host;
            traits->displayNum = coVRConfig::instance()->pipes[pipeNum].x11DisplayNum;
            traits->screenNum = coVRConfig::instance()->pipes[pipeNum].x11ScreenNum;
            if (coVRConfig::instance()->windows[i].screenNum >= 0)
            {
                traits->screenNum = coVRConfig::instance()->windows[i].screenNum;
            }

            // if possible, share graphics context with other windows
            for (int j=0; j<i; ++j)
            {
                const windowStruct &wj = coVRConfig::instance()->windows[j];
                const windowStruct &wi = coVRConfig::instance()->windows[i];
                if (wj.pipeNum == wi.pipeNum)
                {
                    traits->sharedContext = wj.context;
                    break;
                }
            }
        }
        //traits->alpha = 8;
        traits->stencil = coVRConfig::instance()->numStencilBits();
        traits->doubleBuffer = true;
        traits->quadBufferStereo = coVRConfig::instance()->windows[i].stereo;

        if (coVRConfig::instance()->doMultisample())
        {
            traits->samples = coVRConfig::instance()->getMultisampleSamples();
            traits->sampleBuffers = coVRConfig::instance()->getMultisampleSampleBuffers();
        }

        coVRConfig::instance()->windows[i].context = osg::GraphicsContext::createGraphicsContext(traits.get());
        if (coVRConfig::instance()->windows[i].context == NULL)
        {
            cerr << "No valid Pixel Format found, trying without multisample Buffer" << endl;
            traits->sampleBuffers = 0;
            coVRConfig::instance()->windows[i].context = osg::GraphicsContext::createGraphicsContext(traits.get());
            if (coVRConfig::instance()->windows[i].context == NULL)
            {
                cerr << "No valid Pixel Format found, trying without stencil" << endl;
                traits->stencil = 0;
                coVRConfig::instance()->windows[i].context = osg::GraphicsContext::createGraphicsContext(traits.get());
                if (coVRConfig::instance()->windows[i].context == NULL)
                {
                    cerr << "No valid Pixel Format found, trying without alpha" << endl;
                    traits->alpha = 0;
                    coVRConfig::instance()->windows[i].context = osg::GraphicsContext::createGraphicsContext(traits.get());
                }
            }
        }
        coVRConfig::instance()->windows[i].doublebuffer = traits->doubleBuffer;
    }

    if (!coVRConfig::instance()->windows[i].context)
    {
        return false;
    }
    if (!coVRConfig::instance()->windows[i].context->valid())
    {
        cerr << "No valid GL context created" << endl;
        return false;
    }
    coVRConfig::instance()->windows[i].window = dynamic_cast<osgViewer::GraphicsWindow *>(coVRConfig::instance()->windows[i].context.get());
    bool syncToVBlankConfigured = false;
    bool syncToVBlank = covise::coCoviseConfig::isOn("COVER.SyncToVBlank", false, &syncToVBlankConfigured);
    if (syncToVBlankConfigured)
    {
        if (auto win = coVRConfig::instance()->windows[i].window)
        {
	fprintf(stderr,"syncToVBlank=%d\n",(int)syncToVBlank);
            win->setSyncToVBlank(syncToVBlank);
        }
    }

    if (opengl3)
    {
    // for non GL3/GL4 and non GLES2 platforms we need enable the osg_ uniforms that the shaders will use,
        // you don't need thse two lines on GL3/GL4 and GLES2 specific builds as these will be enable by default.
    coVRConfig::instance()->windows[i].context->getState()->setUseModelViewAndProjectionUniforms(true);
    //coVRConfig::instance()->windows[i].context->getState()->setUseVertexAttributeAliasing(true);
    }

#if 0
   // if the config file contains the keywords VisualID
   // we use directly the id to configure the framebuffer
   int visId=coVRConfig::instance()->windows[i].visualId;
   char entry[200];
   sprintf(&entry[0], "COVER.WindowConfig:%d.VisualID", i);
   if (visId == -1)
   {
     visId = covise::coCoviseConfig::getInt(entry, coVRConfig::instance()->windows[i].visualId);
#ifdef __sgi
     visId += 134*coVRConfig::instance()->windows[i].pipeNum;
#endif
   }
   else
     visId = coVRConfig::instance()->windows[i].visualId;

   if(visId != -1)
   {
      Producer::VisualChooser *visualChooser = coVRConfig::instance()->windows[i].rs->getVisualChooser();
      if(!visualChooser)
      {
         visualChooser = new Producer::VisualChooser;
         coVRConfig::instance()->windows[i].rs->setVisualChooser(visualChooser);
      }

      fprintf(stderr,"Configuring render surface %d with visual: dec:%d hex:%x\n",i, visId, visId);
      coVRConfig::instance()->windows[i].rs->getVisualChooser()->setVisualID(visId);
   }
#endif
    if (cover->debugLevel(4))
        fprintf(stderr, "VRWindow::createWin %d --- finished\n", i);
    //sleep(200);
    return true;
}

    bool VRWindow::destroyWin(int i)
    {
        auto &conf = *coVRConfig::instance();
        auto &win = conf.windows[i];

        win.window = nullptr;
        if (win.context)
        {
            win.context->close();
            win.context = nullptr;
        }

        return true;
}
