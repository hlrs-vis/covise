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
//#include "vvNavigationManager.h"
#include "vvVIVEConfig.h"
#include "vvConfig.h"
#include "vvMSController.h"
//#include "input/input.h"
//#include "vvStatsDisplay.h"

using std::cerr;
using std::endl;
using namespace covise;
using namespace vive;

vvConfig *vvConfig::s_instance = NULL;

vvConfig *vvConfig::instance()
{
    if (!s_instance)
        s_instance = new vvConfig;
    return s_instance;
}

void vvConfig::destroy()
{
    delete s_instance;
    s_instance = nullptr;
}

float vvConfig::getSceneSize() const
{
    return m_sceneSize;
}

int vvConfig::parseStereoMode(const char *modeName, bool *stereo)
{
    bool st = true;

    int stereoMode = vvConfig::ANAGLYPHIC;
    if (modeName)
    {
        if (strcasecmp(modeName, "ANAGLYPHIC") == 0)
            stereoMode = vvConfig::ANAGLYPHIC;
        else if (strcasecmp(modeName, "QUAD_BUFFER") == 0)
            stereoMode = vvConfig::QUAD_BUFFER;
        else if (strcasecmp(modeName, "HORIZONTAL_SPLIT") == 0)
            stereoMode = vvConfig::HORIZONTAL_SPLIT;
        else if (strcasecmp(modeName, "VERTICAL_SPLIT") == 0)
            stereoMode = vvConfig::VERTICAL_SPLIT;
        else if (strcasecmp(modeName, "RIGHT_EYE") == 0)
            stereoMode = vvConfig::RIGHT_EYE;
        else if (strcasecmp(modeName, "RIGHT") == 0)
            stereoMode = vvConfig::RIGHT_EYE;
        else if (strcasecmp(modeName, "LEFT") == 0)
            stereoMode = vvConfig::LEFT_EYE;
        else if (strcasecmp(modeName, "LEFT_EYE") == 0)
            stereoMode = vvConfig::LEFT_EYE;
        else if (strcasecmp(modeName, "STIPPLE") == 0)
            stereoMode = vvConfig::VERTICAL_INTERLACE;
        else if (strcasecmp(modeName, "VERTICAL_INTERLACE") == 0)
            stereoMode = vvConfig::VERTICAL_INTERLACE;
        else if (strcasecmp(modeName, "HORIZONTAL_INTERLACE") == 0)
            stereoMode = vvConfig::HORIZONTAL_INTERLACE;
        else if (strcasecmp(modeName, "CHECKERBOARD") == 0)
            stereoMode = vvConfig::CHECKERBOARD;
        else if (strcasecmp(modeName, "MONO") == 0
                || strcasecmp(modeName, "MIDDLE") == 0
                || strcasecmp(modeName, "NONE") == 0
                || strcasecmp(modeName, "") == 0)
        {
            st = false;
            stereoMode = vvConfig::LEFT_EYE;
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

bool vvConfig::requiresTwoViewpoints(int stereomode)
{

    switch (stereomode) {
    case vvConfig::LEFT_EYE:
    case vvConfig::RIGHT_EYE:
        return false;
    }

    return true;
}

vvConfig::vvConfig()
    : m_useDISPLAY(false)
    , m_useVirtualGL(false)
    , m_orthographic(false)
    , m_useWiiMote(false)
    , m_useWiiNavVisenso(false)
    , m_flatDisplay(false)
    , m_continuousRendering(false)
{

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

    constFrameTime = 0.1f;
    constantFrameRate = false;
    float frameRate = (float)coCoviseConfig::getInt("COVER.ConstantFrameRate", 0);
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
    std::string line = coCoviseConfig::getEntry("separation", "COVER.Stereo");
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
                cerr << "vvConfig sscanf failed stereosep" << endl;
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

    //drawStatistics = coCoviseConfig::isOn("COVER.Statistics", false) ? vvStatsDisplay::VIEWER_STATS : vvStatsDisplay::NO_STATS;
    HMDMode = coCoviseConfig::isOn("mode", std::string("COVER.HMD"), false);
    HMDViewingAngle = coCoviseConfig::getFloat("angle", "COVER.HMD", 60.0f);

    // tracked HMD
    trackedHMD = coCoviseConfig::isOn("tracked", std::string("COVER.HMD"), false);

    if (debugLevel(2))
        fprintf(stderr, "\nnew vvConfig\n");

    m_passiveStereo = false;
    m_flatDisplay = true;
    for (size_t i = 0; i < screens.size(); i++)
    {

        float h, p, r;
        
        char str[200];
        sprintf(str, "COVER.ScreenConfig.Screen:%d", (int)i);
        bool state = vvVIVEConfig::getScreenConfigEntry((int)i, screens[i].name, &hsize, &vsize, &x, &y, &z, &h, &p, &r);
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
                    channels[i].stereoMode = vvConfig::RIGHT_EYE;
                }
                else
                {
                    channels[i].stereoMode = vvConfig::LEFT_EYE;
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

        if (channels[i].stereoMode == vvConfig::VERTICAL_INTERLACE || channels[i].stereoMode == vvConfig::HORIZONTAL_INTERLACE || channels[i].stereoMode == vvConfig::CHECKERBOARD)
        {
            m_stencil = true;
        }

        bool exists = false;
        channels[i].fixedViewer = coCoviseConfig::isOn("fixedViewer", str, false, &exists);
        channels[i].stereoOffset = coCoviseConfig::getFloat("stereoOffset", str, 0.f);
        
        channels[i].PBONum = coCoviseConfig::getInt("PBOIndex", str, -1);
        if(channels[i].PBONum == -1)
        {
            channels[i].viewportNum = coCoviseConfig::getInt("viewportIndex", str, (int)i);
        }
        else
        {
            channels[i].viewportNum = -1;
        }
        channels[i].screenNum = coCoviseConfig::getInt("screenIndex", str, (int)i);
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
                if (numWindows > 0)
                {
                    vp.window = 0;
                }
                else
                {
                    std::cerr << "no ChannelConfig for channel " << i << std::endl;
                    exit(1);
                }
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
                    std::vector<std::string> pbos = split(pbolist,',');
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

vvConfig::~vvConfig()
{
    if (debugLevel(2))
        fprintf(stderr, "delete vvConfig\n");

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
vvConfig::debugLevel(int level) const
{
    if (level <= m_dLevel)
        return true;
    else
        return false;
}

int
vvConfig::getDebugLevel() const
{
    return m_dLevel;
}

void
vvConfig::setDebugLevel(int level)
{
    m_dLevel = level;
}

bool vvConfig::mouseNav() const
{
    return false; // return Input::instance()->hasMouse();
}

bool vvConfig::has6DoFInput() const
{
   /* for (int i = 0; i<Input::instance()->getNumPersons(); ++i)
    {
        const Person *p = Input::instance()->getPerson(i);
        if (p->hasHand(0))
            return true;
        if (p->hasRelative() && p->getRelative()->is6Dof())
            return true;
    }*/
    return false;
}

bool vvConfig::mouseTracking() const
{
    return false; //return !Input::instance()->isTrackingOn() && mouseNav();
}

bool vvConfig::useWiiMote() const
{
    return m_useWiiMote;
}

bool vvConfig::useWiiNavigationVisenso() const
{
    return m_useWiiNavVisenso;
}

bool vvConfig::isMenuModeOn() const
{
    return m_menuModeOn;
}

bool vvConfig::colorSceneInMenuMode() const
{
    return m_coloringSceneInMenuMode;
}

bool vvConfig::haveFlatDisplay() const
{
    return m_flatDisplay;
}

bool vvConfig::useDisplayLists() const
{
    return m_useDisplayLists;
}

bool vvConfig::useVBOs() const
{
    return m_useVBOs;
}

float vvConfig::worldAngle() const
{
    return m_worldAngle;
}

int vvConfig::lockToCPU() const
{
    return m_lockToCPU;
}
int vvConfig::numScreens() const
{
    return (int)screens.size();
}
int vvConfig::numViewports() const
{
    return (int)viewports.size();
}
int vvConfig::numBlendingTextures() const
{
    return (int)blendingTextures.size();
}
int vvConfig::numChannels() const
{
    return (int)channels.size();
}

int vvConfig::numPBOs() const
{
    return (int)PBOs.size();
}

int vvConfig::numWindows() const
{
    return (int)windows.size();
}

bool vvConfig::frozen() const
{
    return m_freeze;
}

void vvConfig::setFrozen(bool state)
{
    m_freeze = state;
}

bool vvConfig::orthographic() const
{
    return m_orthographic;
}

void vvConfig::setOrthographic(bool state)
{
    m_orthographic = state;
}

vvConfig::MonoViews vvConfig::monoView() const
{
    return m_monoView;
}

bool vvConfig::stereoState() const
{
    return m_stereoState;
}

float vvConfig::stereoSeparation() const
{
    return m_stereoSeparation;
}

// get number of requested stencil bits (default = 1)
int vvConfig::numStencilBits() const
{
    if (!m_stencil)
        return 0;
    return m_stencilBits;
}

float vvConfig::nearClip() const
{
    return m_nearClip;
}

float vvConfig::farClip() const
{
    return m_farClip;
}

float vvConfig::getLODScale() const
{
    return m_LODScale;
}

void vvConfig::setLODScale(float s)
{
    m_LODScale = s;
}

bool vvConfig::stencil() const
{
    return m_stencil;
}

int vvConfig::stereoMode() const
{
    return m_stereoMode;
}

void vvConfig::setFrameRate(float fr)
{
    if (fr > 0)
    {
        constantFrameRate = true;
        constFrameTime = 1.0f / fr;
    }
    else
        constantFrameRate = false;
}

float vvConfig::frameRate() const
{
    if (constantFrameRate)
        return 1.0f / constFrameTime;
    else
        return 0.f;
}

bool vvConfig::continuousRendering() const
{
    if (vvMSController::instance()->isCluster())
        return true;

    return m_continuousRendering;
}
