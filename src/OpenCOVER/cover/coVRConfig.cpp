/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <cctype>
#include <config/CoviseConfig.h>
#include <config/coConfigConstants.h>
#include <util/string_util.h>
#include <util/unixcompat.h>
#include "coVRNavigationManager.h"
#include "coCoverConfig.h"
#include "coVRConfig.h"
#include "coVRMSController.h"
#include "input/input.h"
#include "coVRStatsDisplay.h"

using std::cerr;
using std::endl;
using namespace covise;
using namespace opencover;

coVRConfig *coVRConfig::s_instance = NULL;

coVRConfig *coVRConfig::instance()
{
    if (!s_instance)
        s_instance = new coVRConfig;
    return s_instance;
}

float coVRConfig::getSceneSize() const
{
    return m_sceneSize;
}

int coVRConfig::parseStereoMode(const char *modeName, bool *stereo)
{
    bool st = true;

    int stereoMode = osg::DisplaySettings::ANAGLYPHIC;
    if (modeName)
    {
        if (strcasecmp(modeName, "ANAGLYPHIC") == 0)
            stereoMode = osg::DisplaySettings::ANAGLYPHIC;
        else if (strcasecmp(modeName, "QUAD_BUFFER") == 0)
            stereoMode = osg::DisplaySettings::QUAD_BUFFER;
        else if (strcasecmp(modeName, "HORIZONTAL_SPLIT") == 0)
            stereoMode = osg::DisplaySettings::HORIZONTAL_SPLIT;
        else if (strcasecmp(modeName, "VERTICAL_SPLIT") == 0)
            stereoMode = osg::DisplaySettings::VERTICAL_SPLIT;
        else if (strcasecmp(modeName, "RIGHT_EYE") == 0)
            stereoMode = osg::DisplaySettings::RIGHT_EYE;
        else if (strcasecmp(modeName, "RIGHT") == 0)
            stereoMode = osg::DisplaySettings::RIGHT_EYE;
        else if (strcasecmp(modeName, "LEFT") == 0)
            stereoMode = osg::DisplaySettings::LEFT_EYE;
        else if (strcasecmp(modeName, "LEFT_EYE") == 0)
            stereoMode = osg::DisplaySettings::LEFT_EYE;
        else if (strcasecmp(modeName, "STIPPLE") == 0)
            stereoMode = osg::DisplaySettings::VERTICAL_INTERLACE;
        else if (strcasecmp(modeName, "VERTICAL_INTERLACE") == 0)
            stereoMode = osg::DisplaySettings::VERTICAL_INTERLACE;
        else if (strcasecmp(modeName, "HORIZONTAL_INTERLACE") == 0)
            stereoMode = osg::DisplaySettings::HORIZONTAL_INTERLACE;
        else if (strcasecmp(modeName, "CHECKERBOARD") == 0)
            stereoMode = osg::DisplaySettings::CHECKERBOARD;
        else if (strcasecmp(modeName, "MONO") == 0
                || strcasecmp(modeName, "MIDDLE") == 0
                || strcasecmp(modeName, "NONE") == 0
                || strcasecmp(modeName, "") == 0)
        {
            st = false;
            stereoMode = osg::DisplaySettings::LEFT_EYE;
        }
        else
        {
            cerr << "Unknown stereo mode \"" << modeName << "\"" << endl;
        }
    }

    if (stereo)
    {
        *stereo = st;
    }
    return stereoMode;
}

bool coVRConfig::requiresTwoViewpoints(int stereomode)
{
    using osg::DisplaySettings;

    switch (stereomode) {
    case DisplaySettings::LEFT_EYE:
    case DisplaySettings::RIGHT_EYE:
        return false;
    }

    return true;
}

coVRConfig::coVRConfig()
    : m_useDISPLAY(false)
    , m_useVirtualGL(false)
    , m_orthographic(false)
    , m_useWiiMote(false)
    , m_useWiiNavVisenso(false)
    , m_flatDisplay(false)
    , m_continuousRendering(false)
{
    assert(!s_instance);

    /// path for the viewpoint file: initialized by 1st param() call

    if (const char *dl = getenv("COVER_DEBUG"))
    {
        m_dLevel = atoi(dl);
        std::cerr << "setting debug level from environment COVER_DEBUG to " << m_dLevel << std::endl;
    }
    else
    {
        m_dLevel = coCoviseConfig::getInt("COVER.DebugLevel", 0);
    }
    
    int hsize, vsize, x, y, z;
    m_passiveStereo = false;

    constFrameTime = 0.1;
    constantFrameRate = false;
    float frameRate = coCoviseConfig::getInt("COVER.ConstantFrameRate", 0);
    if (frameRate > 0)
    {
        constantFrameRate = true;
        constFrameTime = 1.0f / frameRate;
    }
    m_continuousRendering = coCoviseConfig::isOn("COVER.ContinuousRendering", m_continuousRendering);
    m_lockToCPU = coCoviseConfig::getInt("COVER.LockToCPU", -1);
    m_freeze = coCoviseConfig::isOn("COVER.Freeze", false); // don't freeze by default
    m_sceneSize = coCoviseConfig::getFloat("COVER.SceneSize", 2000.0);
    m_farClip = coCoviseConfig::getFloat("COVER.Far", 10000000);
    m_nearClip = coCoviseConfig::getFloat("COVER.Near", 10.0f);
    int numScreens = coConfigConstants::getShmGroupRootRank()<0 || coConfigConstants::getRank()==coConfigConstants::getShmGroupRootRank() ? 1 : 0;
    numScreens = coCoviseConfig::getInt("COVER.NumScreens", numScreens);
    if (numScreens < 0)
    {
        std::cerr << "COVER.NumScreens cannot be < 0" << std::endl;
        exit(1);
    }
    if (numScreens > 50)
    {
	std::cerr << "COVER.NumScreens cannot be > 50" << std::endl;
	exit(1);
    }
    screens.resize(numScreens);

    const int numChannels = coCoviseConfig::getInt("COVER.NumChannels", numScreens); // normally numChannels == numScreens, only if we use PBOs, it might be equal to the number of PBOs
    channels.resize(numChannels);
    
    const int numWindows = coCoviseConfig::getInt("COVER.NumWindows", numScreens);
    if (numWindows < 0)
    {
	std::cerr << "COVER.NumWindows cannot be < 0" << std::endl;
	exit(1);
    }
    if (numWindows > 50)
    {
	std::cerr << "COVER.NumWindows cannot be > 50" << std::endl;
	exit(1);
    }
    windows.resize(numWindows);

    const int numViewports = coCoviseConfig::getInt("COVER.NumViewports", numChannels); // normally this is equal to the number of Channels
    if (numViewports < 0)
    {
	std::cerr << "COVER.NumViewports cannot be negative" << std::endl;
	exit(1);
    }
    viewports.resize(numViewports);

    const int numBlendingTextures = coCoviseConfig::getInt("COVER.NumBlendingTextures", 0); 
    if (numBlendingTextures < 0)
    {
	std::cerr << "COVER.NumBlendingTextures cannot be negative" << std::endl;
	exit(1);
    }
    blendingTextures.resize(numBlendingTextures);

    const int numPBOs = coCoviseConfig::getInt("COVER.NumPBOs", 0);
    if (numPBOs < 0)
    {
	std::cerr << "COVER.NumPBOs cannot be negative" << std::endl;
	exit(1);
    }
    PBOs.resize(numPBOs);

    const int numPipes = coCoviseConfig::getInt("COVER.NumPipes", 1);
    if (numPipes < 1)
    {
	std::cerr << "COVER.NumPipes cannot be < 1" << std::endl;
	exit(1);
    }
    if (numPipes > 50)
    {
	std::cerr << "COVER.NumPipes cannot be > 50" << std::endl;
	exit(1);
    }
    pipes.resize(numPipes);

    glVersion = coCoviseConfig::getEntry("COVER.GLVersion");
    glProfileMask = coCoviseConfig::getEntry("COVER.GLProfileMast");
    glContextFlags = coCoviseConfig::getEntry("COVER.GLContextFlags");
    m_stencil = coCoviseConfig::isOn("COVER.Stencil", true);
    m_stencilBits = coCoviseConfig::getInt("COVER.StencilBits", 1);
    m_stereoSeparation = 64.0f;
    string line = coCoviseConfig::getEntry("separation", "COVER.Stereo");
    if (!line.empty())
    {
        if (strncmp(line.c_str(), "AUTO", 4) == 0)
        {
            m_stereoSeparation = 1000;
        }
        else
        {
            if (sscanf(line.c_str(), "%f", &m_stereoSeparation) != 1)
            {
                cerr << "CoVRConfig sscanf failed stereosep" << endl;
            }
        }
    }

    m_monoView = MONO_MIDDLE;

    m_useDisplayLists = coCoviseConfig::isOn("COVER.UseDisplayLists", false);
    m_useVBOs = coCoviseConfig::isOn("COVER.UseVertexBufferObjects", !m_useDisplayLists);

    multisample = coCoviseConfig::isOn("COVER.Multisample", false);
    multisampleInvert = coCoviseConfig::isOn(std::string("invert"), std::string("COVER.Multisample"), false);
    multisampleSamples = coCoviseConfig::getInt("numSamples", "COVER.Multisample", 2);
    multisampleSampleBuffers = coCoviseConfig::getInt("numBuffers", "COVER.Multisample", 2);
    multisampleCoverage = coCoviseConfig::getFloat("sampleCoverage", "COVER.Multisample", 1.0);

    std::string msMode = coCoviseConfig::getEntry("mode", "COVER.Multisample", "FASTEST");

    if (msMode == "FASTEST")
    {
        multisampleMode = osg::Multisample::FASTEST;
    }
    else if (msMode == "NICEST")
    {
        multisampleMode = osg::Multisample::NICEST;
    }
    else
    {
        multisampleMode = osg::Multisample::DONT_CARE;
    }

    m_useWiiMote = coCoviseConfig::isOn("COVER.Input.WiiMote", false);
    m_useWiiNavVisenso = coCoviseConfig::isOn("COVER.Input.WiiNavigationVisenso", false);
    m_menuModeOn = coCoviseConfig::isOn("COVER.MenuMode", false);
    m_coloringSceneInMenuMode = coCoviseConfig::isOn("COVER.MenuMode.Coloring", true);

    std::string entry = coCoviseConfig::getEntry("COVER.MonoView");
    if (!entry.empty())
    {
        if (strcasecmp(entry.c_str(), "LEFT") == 0)
            m_monoView = MONO_LEFT;
        if (strcasecmp(entry.c_str(), "RIGHT") == 0)
            m_monoView = MONO_RIGHT;
        if (strcasecmp(entry.c_str(), "NONE") == 0)
            m_monoView = MONO_NONE;
    }
    entry = coCoviseConfig::getEntry("COVER.StereoMode");
    m_stereoMode = parseStereoMode(entry.c_str());

    m_envMapMode = FIXED_TO_VIEWER;
    entry = coCoviseConfig::getEntry("COVER.EnvMapMode");
    if (!entry.empty())
    {
        if (strcasecmp(entry.c_str(), "fixedToViewer") == 0)
            m_envMapMode = FIXED_TO_VIEWER;
        if (strcasecmp(entry.c_str(), "fixedToObjectsRoot") == 0)
            m_envMapMode = FIXED_TO_OBJROOT;
        if (strcasecmp(entry.c_str(), "fixedToViewerFront") == 0)
            m_envMapMode = FIXED_TO_VIEWER_FRONT;
        if (strcasecmp(entry.c_str(), "fixedToObjectsRootFront") == 0)
            m_envMapMode = FIXED_TO_OBJROOT_FRONT;
        if (strcasecmp(entry.c_str(), "none") == 0)
            m_envMapMode = NONE;
    }

    m_LODScale = coCoviseConfig::getFloat("COVER.LODScale", 1.0);
    m_worldAngle = coCoviseConfig::getFloat("COVER.WorldAngle", 0.);

    drawStatistics = coCoviseConfig::isOn("COVER.Statistics", false) ? coVRStatsDisplay::VIEWER_STATS : coVRStatsDisplay::NO_STATS;
    HMDMode = coCoviseConfig::isOn("mode", std::string("COVER.HMD"), false);
    HMDViewingAngle = coCoviseConfig::getFloat("angle", "COVER.HMD", 60.0f);

    // tracked HMD
    trackedHMD = coCoviseConfig::isOn("tracked", std::string("COVER.HMD"), false);

    if (debugLevel(2))
        fprintf(stderr, "\nnew coVRConfig\n");

    m_passiveStereo = false;
    m_flatDisplay = true;
    for (size_t i = 0; i < screens.size(); i++)
    {

        float h, p, r;
        
        char str[200];
        sprintf(str, "COVER.ScreenConfig.Screen:%d", (int)i);
        bool state = coCoverConfig::getScreenConfigEntry(i, screens[i].name, &hsize, &vsize, &x, &y, &z, &h, &p, &r);
        if (!state)
        {
            cerr << "Exiting because of erroneous ScreenConfig entry." << endl;
            exit(-1);
        }
        else
        {
            
            screens[i].render = coCoviseConfig::isOn("render", str, true);
            screens[i].hsize = (float)hsize;
            screens[i].configuredHsize = screens[i].hsize;
            screens[i].vsize = (float)vsize;
            screens[i].configuredVsize = screens[i].vsize;
            screens[i].xyz.set((float)x, (float)y, (float)z);
            screens[i].hpr.set(h, p, r);
        }
        if ((i == 1) && (screens[0].name == screens[1].name))
        {
            m_passiveStereo = true;
        }

        // this check is too simple
        if (screens[i].hpr != screens[0].hpr)
        {
            m_flatDisplay = false;
        }
        screens[i].lTan = -1;
        screens[i].rTan = -1;
        screens[i].tTan = -1;
        screens[i].bTan = -1; // left, right, top bottom field of views, default/not set is -1
    }

    for (size_t i = 0; i < pipes.size(); i++)
    {
        char str[200];
        sprintf(str, "COVER.PipeConfig.Pipe:%d", (int)i);
        pipes[i].x11DisplayNum = coCoviseConfig::getInt("server", str, 0);
        pipes[i].x11ScreenNum = coCoviseConfig::getInt("screen", str, 0);
        pipes[i].x11DisplayHost = coCoviseConfig::getEntry("host", str, "");
        bool present = false;
        pipes[i].useDISPLAY = coCoviseConfig::isOn("useDISPLAY", str, false, &present);
    }

    for (size_t i = 0; i < windows.size(); i++)
    {
        auto &w = windows[i];
        w.window = NULL;

        char str[200];
        sprintf(str, "COVER.WindowConfig.Window:%d", (int)i);

        w.name = coCoviseConfig::getEntry("comment", str, "COVER");
        w.pipeNum = coCoviseConfig::getInt("pipeIndex", str, 0);
        w.ox = coCoviseConfig::getInt("left", str, 0);
        w.oy = coCoviseConfig::getInt("top", str, 0);
        w.sx = coCoviseConfig::getInt("width", str, 1024);
        w.sy = coCoviseConfig::getInt("height", str, 768);
        bool have_bottom = false;
        coCoviseConfig::getInt("bottom", str, 0, &have_bottom);
        if (have_bottom)
            printf("bottom is ignored in %s, please use top\n", str);
        w.decoration = coCoviseConfig::isOn("decoration", std::string(str), false);
        w.resize = coCoviseConfig::isOn("resize", str, true);
        w.stereo = coCoviseConfig::isOn("stereo", std::string(str), false);
        w.embedded = coCoviseConfig::isOn("embedded", std::string(str), false);
        w.pbuffer = coCoviseConfig::isOn("pbuffer", std::string(str), false);
#if USE_OSMESA
        w.type = coCoviseConfig::getEntry("type", std::string(str), "Mesa");
#else
        w.type = coCoviseConfig::getEntry("type", std::string(str), "");
#endif
        std::transform(w.type.begin(), w.type.end(), w.type.begin(), ::tolower);
        w.swapGroup = coCoviseConfig::getInt("swapGroup", str, -1);
        w.swapBarrier = coCoviseConfig::getInt("swapBarrier", str, -1);
        w.screenNum = coCoviseConfig::getInt("screen", str, -1);
    }

    m_stereoState = false;
    for (size_t i = 0; i < channels.size(); i++)
    {
        std::string stereoM;

        char str[200];
        sprintf(str, "COVER.ChannelConfig.Channel:%d", (int)i);
        std::string s = coCoviseConfig::getEntry("comment", str, "NoNameChannel");
        channels[i].name = s;
        stereoM = coCoviseConfig::getEntry("stereoMode", str);
        if (!stereoM.empty())
        {
            channels[i].stereoMode = parseStereoMode(stereoM.c_str(), &channels[i].stereo);
        }
        else
        {
            if (m_passiveStereo)
            {
                if (i % 2)
                {
                    channels[i].stereoMode = osg::DisplaySettings::RIGHT_EYE;
                }
                else
                {
                    channels[i].stereoMode = osg::DisplaySettings::LEFT_EYE;
                }
                channels[i].stereo = true;
            }
            else
            {
                channels[i].stereoMode = m_stereoMode;
            }
        }

        if (channels[i].stereo)
            m_stereoState = true;

        if (channels[i].stereoMode == osg::DisplaySettings::VERTICAL_INTERLACE || channels[i].stereoMode == osg::DisplaySettings::HORIZONTAL_INTERLACE || channels[i].stereoMode == osg::DisplaySettings::CHECKERBOARD)
        {
            m_stencil = true;
        }

        bool exists = false;
        channels[i].fixedViewer = coCoviseConfig::isOn("fixedViewer", str, false, &exists);
        channels[i].stereoOffset = coCoviseConfig::getFloat("stereoOffset", str, 0.f);
        
        channels[i].PBONum = coCoviseConfig::getInt("PBOIndex", str, -1);
        if(channels[i].PBONum == -1)
        {
            channels[i].viewportNum = coCoviseConfig::getInt("viewportIndex", str, i);
        }
        else
        {
            channels[i].viewportNum = -1;
        }
        channels[i].screenNum = coCoviseConfig::getInt("screenIndex", str, i);
        if (channels[i].screenNum >= screens.size())
        {
            std::cerr << "screenIndex " << channels[i].screenNum << " for channel " << i << " out of range (max: " << screens.size()-1 << ")" << std::endl;
            exit(1);
        }
    }
    m_stereoState = coCoviseConfig::isOn("COVER.Stereo", m_stereoState);

    for (size_t i = 0; i < PBOs.size(); i++)
    {
        std::string stereoM;

        char str[200];
        sprintf(str, "COVER.PBOConfig.PBO:%d", (int)i);
        
        PBOs[i].PBOsx = coCoviseConfig::getInt("PBOSizeX", str, -1);
        PBOs[i].PBOsx = coCoviseConfig::getInt("width", str, PBOs[i].PBOsx);
        PBOs[i].PBOsy = coCoviseConfig::getInt("PBOSizeY", str, -1);
        PBOs[i].PBOsy = coCoviseConfig::getInt("height", str, PBOs[i].PBOsy);
        PBOs[i].windowNum = coCoviseConfig::getInt("windowIndex", str, -1);
    }

    for (size_t i = 0; i < viewports.size(); i++)
    {
        std::string stereoM;

        char str[200];
        sprintf(str, "COVER.ViewportConfig.Viewport:%d", (int)i);
        viewportStruct &vp = viewports[i];
        std::string mode = coCoviseConfig::getEntry("mode", str, "");
        mode = toLower(mode);
        if (mode.empty() || mode == "channel")
        {
            vp.mode = viewportStruct::Channel;
        }
        else if (mode == "pbo")
        {
            vp.mode = viewportStruct::PBO;
        }
        else if (mode == "tridelityml")
        {
            vp.mode = viewportStruct::TridelityML;
        }
        else if (mode == "tridelitymv")
        {
            vp.mode = viewportStruct::TridelityMV;
        }
        else
        {
            std::cerr << "cannot parse viewport mode \"" << mode << "\" for viewport " << i << std::endl;
            return;
        }
        bool exists=false;
        vp.window = coCoviseConfig::getInt("windowIndex", str, -1, &exists);
        if(!exists)
        {
            // no viewport config, check for values in channelConfig for backward compatibility

            sprintf(str, "COVER.ChannelConfig.Channel:%d", (int)i);
            vp.window = coCoviseConfig::getInt("windowIndex", str, -1,&exists);
            if (!exists)
            {
                std::cerr << "no ChannelConfig for channel " << i << std::endl;
                exit(1);
            }
        }

        vp.PBOnum = coCoviseConfig::getInt("PBOIndex", str, -1);
        switch(vp.mode)
        {
            case viewportStruct::Channel:
            case viewportStruct::PBO:
                {
                    if (vp.PBOnum >= 0)
                    {
                        vp.mode = viewportStruct::PBO;
                    }
                }
                break;
            case viewportStruct::TridelityML:
            case viewportStruct::TridelityMV:
                {
                    if (vp.PBOnum >= 0)
                    {
                        std::cerr << "PBOIndex given for viewport " << i << " with incompatible mode " << vp.mode << ", use pboList" << std::endl;
                        return;
                    }

                    std::string pbolist = coCoviseConfig::getEntry("pboList", str, "");
                    std::vector<string> pbos = split(pbolist,',');
                    for (size_t i=0; i<pbos.size(); ++i)
                    {
                        int p=0;
                        std::stringstream s(pbos[i]);
                        s >> p;
                        vp.pbos.push_back(p);
                    }
                    if (vp.pbos.size() != 5)
                    {
                        std::cerr << "need exactly 5 PBOs for TridelityML/TridelityMV mode, but " << vp.pbos.size() << " given" << std::endl;
                        return;
                    }
                }
                break;
        }

        vp.viewportXMin = coCoviseConfig::getFloat("left", str, 0);
        vp.viewportYMin = coCoviseConfig::getFloat("bottom", str, 0);

        if (vp.window >= 0)
        {
            if (vp.viewportXMin > 1.0)
            {
                vp.viewportXMin = vp.viewportXMin / ((float)(windows[vp.window].sx));
            }
            if (vp.viewportYMin > 1.0)
            {
                vp.viewportYMin = vp.viewportYMin / ((float)(windows[vp.window].sy));
            }
        }

        vp.viewportXMax = coCoviseConfig::getFloat("right", str, -1);
        vp.viewportYMax = coCoviseConfig::getFloat("top", str, -1);
        if (vp.viewportXMax < 0)
        {
            float w,h;
            w = coCoviseConfig::getFloat("width", str, 1.0);
            h = coCoviseConfig::getFloat("height", str, 1.0);
            if (vp.window >= 0)
            {
                if (w > 1.0)
                    vp.viewportXMax = vp.viewportXMin + (w / ((float)(windows[vp.window].sx)));
                else
                    vp.viewportXMax = vp.viewportXMin + w;
                if (h > 1.0)
                    vp.viewportYMax = vp.viewportYMin + (h / ((float)(windows[vp.window].sy)));
                else
                    vp.viewportYMax = vp.viewportYMin + h;
            }
        }
        else
        {
            if (vp.window >= 0)
            {
                if (vp.viewportXMax > 1.0)
                    vp.viewportXMax = vp.viewportXMax / ((float)(windows[vp.window].sx));
                if (vp.viewportYMax > 1.0)
                    vp.viewportYMax = vp.viewportYMax / ((float)(windows[vp.window].sy));
            }
        }

        vp.sourceXMin = coCoviseConfig::getFloat("sourceLeft", str, 0);
        vp.sourceYMin = coCoviseConfig::getFloat("sourceBottom", str, 0);

        if (vp.PBOnum >= 0)
        {
            if (vp.sourceXMin > 1.0)
            {
                vp.sourceXMin = vp.sourceXMin / ((float)(PBOs[vp.PBOnum].PBOsx));
            }
            if (vp.sourceYMin > 1.0)
            {
                vp.sourceYMin = vp.sourceYMin / ((float)(PBOs[vp.PBOnum].PBOsy));
            }
        }

        vp.sourceXMax = coCoviseConfig::getFloat("sourceRight", str, -1);
        vp.sourceYMax = coCoviseConfig::getFloat("sourceTop", str, -1);
        if (vp.sourceXMax < 0)
        {
            float w,h;
            w = coCoviseConfig::getFloat("sourceWidth", str, 1.0);
            h = coCoviseConfig::getFloat("sourceHeight", str, 1.0);
            if (vp.PBOnum >= 0)
            {
                if (w > 1.0)
                    vp.sourceXMax = vp.sourceXMin + (w / ((float)(PBOs[vp.PBOnum].PBOsx)));
                else
                    vp.sourceXMax = vp.sourceXMin + w;
                if (h > 1.0)
                    vp.sourceYMax = vp.sourceYMin + (h / ((float)(PBOs[vp.PBOnum].PBOsy)));
                else
                    vp.sourceYMax = vp.sourceYMin + h;
            }
        }
        else
        {
            if (vp.PBOnum >= 0)
            {
                if (vp.sourceXMax > 1.0)
                    vp.sourceXMax = vp.sourceXMax / ((float)(PBOs[vp.PBOnum].PBOsx));
                if (vp.sourceYMax > 1.0)
                    vp.sourceYMax = vp.sourceYMax / ((float)(PBOs[vp.PBOnum].PBOsy));
            }
        }

        vp.distortMeshName = coCoviseConfig::getEntry("distortMesh", str, "");
        vp.blendingTextureName = coCoviseConfig::getEntry("blendingTexture", str, "");

    }
    
    for (size_t i = 0; i < blendingTextures.size(); i++)
    {
        char str[200];
        sprintf(str, "COVER.BlendingTextureConfig.BlendingTexture:%d", (int)i);
        blendingTextureStruct &bt = blendingTextures[i];
        bool exists=false;
        bt.window = coCoviseConfig::getInt("windowIndex", str, -1,&exists);
        
        bt.viewportXMin = coCoviseConfig::getFloat("left", str, 0);
        bt.viewportYMin = coCoviseConfig::getFloat("bottom", str, 0);

        if (bt.viewportXMin > 1.0)
        {
            bt.viewportXMin = bt.viewportXMin / ((float)(windows[bt.window].sx));
        }
        if (bt.viewportYMin > 1.0)
        {
            bt.viewportYMin = bt.viewportYMin / ((float)(windows[bt.window].sy));
        }

        bt.viewportXMax = coCoviseConfig::getFloat("right", str, -1);
        bt.viewportYMax = coCoviseConfig::getFloat("top", str, -1);
        if (bt.viewportXMax < 0)
        {
            float w,h;
            w = coCoviseConfig::getFloat("width", str, 1024);
            h = coCoviseConfig::getFloat("height", str, 768);
            if (w > 1.0)
                bt.viewportXMax = bt.viewportXMin + (w / ((float)(windows[bt.window].sx)));
            else
                bt.viewportXMax = bt.viewportXMin + w;
            if (h > 1.0)
                bt.viewportYMax = bt.viewportYMin + (h / ((float)(windows[bt.window].sy)));
            else
                bt.viewportYMax = bt.viewportYMin + h;
        }
        else
        {
            if (bt.viewportXMax > 1.0)
                bt.viewportXMax = bt.viewportXMax / ((float)(windows[bt.window].sx));
            if (bt.viewportYMax > 1.0)
                bt.viewportYMax = bt.viewportYMax / ((float)(windows[bt.window].sy));
        }

        bt.blendingTextureName = coCoviseConfig::getEntry("blendingTexture", str, "");

    }

    std::string lang = coCoviseConfig::getEntry("value", "COVER.Menu.Language", "ENGLISH");
    if (lang == "GERMAN")
        m_language = GERMAN;
    else
        m_language = ENGLISH;

    if (debugLevel(2))
    {
        std::cerr << "configured with "
            << this->numScreens() << " screens, "
            << this->numWindows() << " windows, "
            << this->numChannels() << " channels, "
            << this->numViewports() << " viewports, "
            << this->numBlendingTextures() << " blending textures, "
            << this->numPBOs() << " PBOs."
            << std::endl;
    }
}

coVRConfig::~coVRConfig()
{
    if (debugLevel(2))
        fprintf(stderr, "delete coVRConfig\n");

    viewports.clear();
    blendingTextures.clear();
    PBOs.clear();
    channels.clear();
    screens.clear();
    windows.clear();
    pipes.clear();

    s_instance = NULL;
}

bool
coVRConfig::debugLevel(int level) const
{
    if (level <= m_dLevel)
        return true;
    else
        return false;
}

int
coVRConfig::getDebugLevel() const
{
    return m_dLevel;
}

void
coVRConfig::setDebugLevel(int level)
{
    m_dLevel = level;
}

bool coVRConfig::mouseNav() const
{
    return Input::instance()->hasMouse();
}

bool coVRConfig::has6DoFInput() const
{
    for (int i=0; i<Input::instance()->getNumPersons(); ++i)
    {
        const Person *p = Input::instance()->getPerson(i);
        if (p->hasHand(0))
            return true;
        if (p->hasRelative() && p->getRelative()->is6Dof())
            return true;
    }
    return false;
}

bool coVRConfig::mouseTracking() const
{
    return !Input::instance()->isTrackingOn() && mouseNav();
}

bool coVRConfig::useWiiMote() const
{
    return m_useWiiMote;
}

bool coVRConfig::useWiiNavigationVisenso() const
{
    return m_useWiiNavVisenso;
}

bool coVRConfig::isMenuModeOn() const
{
    return m_menuModeOn;
}

bool coVRConfig::colorSceneInMenuMode() const
{
    return m_coloringSceneInMenuMode;
}

bool coVRConfig::haveFlatDisplay() const
{
    return m_flatDisplay;
}

bool coVRConfig::useDisplayLists() const
{
    return m_useDisplayLists;
}

bool coVRConfig::useVBOs() const
{
    return m_useVBOs;
}

float coVRConfig::worldAngle() const
{
    return m_worldAngle;
}

int coVRConfig::lockToCPU() const
{
    return m_lockToCPU;
}
int coVRConfig::numScreens() const
{
    return screens.size();
}
int coVRConfig::numViewports() const
{
    return viewports.size();
}
int coVRConfig::numBlendingTextures() const
{
    return blendingTextures.size();
}
int coVRConfig::numChannels() const
{
    return channels.size();
}

int coVRConfig::numPBOs() const
{
    return PBOs.size();
}

int coVRConfig::numWindows() const
{
    return windows.size();
}

bool coVRConfig::frozen() const
{
    return m_freeze;
}

void coVRConfig::setFrozen(bool state)
{
    m_freeze = state;
}

bool coVRConfig::orthographic() const
{
    return m_orthographic;
}

void coVRConfig::setOrthographic(bool state)
{
    m_orthographic = state;
}

coVRConfig::MonoViews coVRConfig::monoView() const
{
    return m_monoView;
}

bool coVRConfig::stereoState() const
{
    return m_stereoState;
}

float coVRConfig::stereoSeparation() const
{
    return m_stereoSeparation;
}

// get number of requested stencil bits (default = 1)
int coVRConfig::numStencilBits() const
{
    if (!m_stencil)
        return 0;
    return m_stencilBits;
}

float coVRConfig::nearClip() const
{
    return m_nearClip;
}

float coVRConfig::farClip() const
{
    return m_farClip;
}

float coVRConfig::getLODScale() const
{
    return m_LODScale;
}

void coVRConfig::setLODScale(float s)
{
    m_LODScale = s;
}

bool coVRConfig::stencil() const
{
    return m_stencil;
}

int coVRConfig::stereoMode() const
{
    return m_stereoMode;
}

void coVRConfig::setFrameRate(float fr)
{
    if (fr > 0)
    {
        constantFrameRate = true;
        constFrameTime = 1.0f / fr;
    }
    else
        constantFrameRate = false;
}

float coVRConfig::frameRate() const
{
    if (constantFrameRate)
        return 1.0f / constFrameTime;
    else
        return 0.f;
}

bool coVRConfig::continuousRendering() const
{
    if (coVRMSController::instance()->isCluster())
        return true;

    return m_continuousRendering;
}
