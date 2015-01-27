/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <config/CoviseConfig.h>
#include "coVRNavigationManager.h"
#include "coCoverConfig.h"
#include "coVRConfig.h"
#include "input/input.h"

using std::cerr;
using std::endl;
using namespace covise;
using namespace opencover;

coVRConfig *coVRConfig::instance()
{
    static coVRConfig *singleton = NULL;
    if (!singleton)
        singleton = new coVRConfig;
    return singleton;
}

float coVRConfig::getSceneSize() const
{
    return m_sceneSize;
}

int coVRConfig::parseStereoMode(const char *modeName)
{

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
        else if (strcasecmp(modeName, "MONO") == 0)
        {
            stereoMode = osg::DisplaySettings::LEFT_EYE;
            m_stereoState = false;
            m_stereoSeparation = 0.f;
        }
        else if (strcasecmp(modeName, "NONE") == 0)
            stereoMode = osg::DisplaySettings::ANAGLYPHIC;
        else if (modeName[0] == '\0')
            stereoMode = osg::DisplaySettings::ANAGLYPHIC;
        else
            cerr << "Unknown stereo mode \"" << modeName << "\"" << endl;
    }
    return stereoMode;
}

coVRConfig::coVRConfig()
    : m_useDISPLAY(false)
    , m_orthographic(false)
    , m_mouseNav(true)
    , m_useWiiMote(false)
    , m_useWiiNavVisenso(false)
    , m_flatDisplay(false)
{
    m_numWindows = 0;
    /// path for the viewpoint file: initialized by 1st param() call

    m_dLevel = coCoviseConfig::getInt("COVER.DebugLevel", 0);

    collaborativeOptionsFile = NULL;
    int hsize, vsize, x, y, z;
    m_numScreens = 1;
    m_numPipes = 1;
    m_passiveStereo = false;

    m_mouseNav = coCoviseConfig::isOn("COVER.Input.MouseNav", m_mouseNav);

    constFrameTime = 0.1;
    constantFrameRate = false;
    float frameRate = coCoviseConfig::getInt("COVER.ConstantFrameRate", 0);
    if (frameRate > 0)
    {
        constantFrameRate = true;
        constFrameTime = 1.0f / frameRate;
    }
    m_freeze = coCoviseConfig::isOn("COVER.Freeze", true);
    m_sceneSize = coCoviseConfig::getFloat("COVER.SceneSize", 2000.0);
    m_farClip = coCoviseConfig::getFloat("COVER.Far", 10000000);
    m_nearClip = coCoviseConfig::getFloat("COVER.Near", 10.0f);
    m_numScreens = coCoviseConfig::getInt("COVER.NumScreens", 1);
    m_numWindows = coCoviseConfig::getInt("COVER.NumWindows", m_numScreens);
    m_numPipes = coCoviseConfig::getInt("COVER.NumPipes", 1);
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
    m_stereoState = coCoviseConfig::isOn("COVER.Stereo", false);
    if (!m_stereoState)
        m_stereoSeparation = 0.f;

    m_monoView = MONO_MIDDLE;

    m_useDisplayLists = coCoviseConfig::isOn("COVER.UseDisplayLists", false);
    m_useVBOs = coCoviseConfig::isOn("COVER.UseVertexBufferObjects", !m_useDisplayLists);

    multisample = coCoviseConfig::isOn("COVER.Multisample", false);
    multisampleInvert = coCoviseConfig::isOn(std::string("invert"), std::string("COVER.Multisample"), false);
    multisampleSamples = coCoviseConfig::getInt("numSamples", "COVER.Multisample", 2);
    multisampleSampleBuffers = coCoviseConfig::getInt("numBuffers", "COVER.Multisample", 2);
    multisampleCoverage = coCoviseConfig::getFloat("sampleCoverage", "COVER.Multisample", 1.0);

    std::string msMode = coCoviseConfig::getEntry("mode", "COVER.Multisample", "FASTEST");

    std::string lang = coCoviseConfig::getEntry("value", "COVER.Menu.Language", "ENGLISH");
    if (lang == "GERMAN")
        m_language = GERMAN;
    else
        m_language = ENGLISH;

    m_restrict = coCoviseConfig::isOn("COVER.Restrict", false);
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
    }

    m_LODScale = coCoviseConfig::getFloat("COVER.LODScale", 1.0);
    m_worldAngle = coCoviseConfig::getFloat("COVER.WorldAngle", 0.);

    drawStatistics = coCoviseConfig::isOn("COVER.Statistics", false);
    HMDMode = coCoviseConfig::isOn("mode", std::string("COVER.HMD"), false);
    HMDViewingAngle = coCoviseConfig::getFloat("angle", "COVER.HMD", 60.0f);

    // tracked HMD
    HMDDistance = coCoviseConfig::getFloat("distance", "COVER.HMD", 0.0f);
    trackedHMD = coCoviseConfig::isOn("tracked", std::string("COVER.HMD"), false);

    if (debugLevel(2))
        fprintf(stderr, "\nnew coVRConfig\n");

    //bool isMaster = coVRMSController::instance()->isMaster();

    if (m_numWindows < 1)
    {
        fprintf(stderr, "numWindows <1\n");
        exit(-1);
    }

    if (m_numScreens < 1)
    {
        fprintf(stderr, "numScreens <1\n");
        exit(-1);
    }

    if (m_numPipes < 1)
    {
        fprintf(stderr, "numPipes <1\n");
        exit(-1);
    }

    if (m_numWindows > 50)
    {
        fprintf(stderr, "numWindows >50\n");
        exit(-1);
    }

    if (m_numPipes > 50)
    {
        fprintf(stderr, "numPipes > 50\n");
        exit(-1);
    }

    if (m_numScreens > 50)
    {
        fprintf(stderr, "numScreens > 50\n");
        exit(-1);
    }

    screens = new screenStruct[m_numScreens];
    pipes = new pipeStruct[m_numPipes];
    windows = new windowStruct[m_numWindows];
    windows[0].window = NULL;

    m_passiveStereo = false;
    m_flatDisplay = true;
    for (int i = 0; i < m_numScreens; i++)
    {

        float h, p, r;
        bool state = coCoverConfig::getScreenConfigEntry(i, screens[i].name, &hsize, &vsize, &x, &y, &z, &h, &p, &r);
        if (!state)
        {
            cerr << "Exiting because of erroneous ScreenConfig entry." << endl;
            exit(-1);
        }
        else
        {
            screens[i].hsize = (float)hsize;
            screens[i].vsize = (float)vsize;
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

    for (int i = 0; i < m_numPipes; i++)
    {
        char str[200];
        sprintf(str, "COVER.PipeConfig.Pipe:%d", i);
        pipes[i].x11DisplayNum = coCoviseConfig::getInt("server", str, 0);
        pipes[i].x11ScreenNum = coCoviseConfig::getInt("screen", str, 0);
    }

    for (int i = 0; i < m_numWindows; i++)
    {
        bool state = coCoverConfig::getWindowConfigEntry(i, windows[i].name,
                                                         &windows[i].pipeNum, &windows[i].ox, &windows[i].oy,
                                                         &windows[i].sx, &windows[i].sy, &windows[i].decoration,
                                                         &windows[i].stereo, &windows[i].resize, &windows[i].embedded);
        if (!state)
        {
            cerr << "Exit because of erroneous WindowConfig entry." << endl;
            exit(-1);
        }
    }

    for (int i = 0; i < m_numScreens; i++)
    {
        std::string stereoM;

        char str[200];
        sprintf(str, "COVER.ChannelConfig.Channel:%d", i);
        std::string s = coCoviseConfig::getEntry("comment", str, "NoNameChannel");
        screens[i].name = s;
        screens[i].render = coCoviseConfig::isOn("render", str, true);
        screens[i].window = coCoviseConfig::getInt("windowIndex", str, 0);
        screens[i].viewportXMin = coCoviseConfig::getFloat("left", str, 0);
        screens[i].viewportYMin = coCoviseConfig::getFloat("bottom", str, 0);

        if (screens[i].viewportXMin > 1.0)
        {
            screens[i].viewportXMin = screens[i].viewportXMin / ((float)(windows[screens[i].window].sx));
        }
        if (screens[i].viewportYMin > 1.0)
        {
            screens[i].viewportYMin = screens[i].viewportYMin / ((float)(windows[screens[i].window].sy));
        }

        screens[i].viewportXMax = coCoviseConfig::getFloat("right", str, -1);
        screens[i].viewportYMax = coCoviseConfig::getFloat("top", str, -1);
        if (screens[i].viewportXMax < 0)
        {
            screens[i].viewportXMax = coCoviseConfig::getFloat("width", str, 1024);
            screens[i].viewportYMax = coCoviseConfig::getFloat("height", str, 768);
            if (screens[i].viewportXMax > 1.0)
                screens[i].viewportXMax = screens[i].viewportXMin + (screens[i].viewportXMax / ((float)(windows[screens[i].window].sx)));
            if (screens[i].viewportYMax > 1.0)
                screens[i].viewportYMax = screens[i].viewportYMin + (screens[i].viewportYMax / ((float)(windows[screens[i].window].sy)));
        }
        else
        {
            if (screens[i].viewportXMax > 1.0)
                screens[i].viewportXMax = screens[i].viewportXMax / ((float)(windows[screens[i].window].sx));
            if (screens[i].viewportYMax > 1.0)
                screens[i].viewportYMax = screens[i].viewportYMax / ((float)(windows[screens[i].window].sy));
        }
        stereoM = coCoviseConfig::getEntry("stereoMode", str);
        if (!stereoM.empty())
        {
            screens[i].stereoMode = parseStereoMode(stereoM.c_str());
        }
        else
        {
            if (m_passiveStereo)
            {
                if (i % 2)
                {
                    screens[i].stereoMode = osg::DisplaySettings::RIGHT_EYE;
                }
                else
                {
                    screens[i].stereoMode = osg::DisplaySettings::LEFT_EYE;
                }
            }
            else
            {
                screens[i].stereoMode = m_stereoMode;
            }
        }

        if (screens[i].stereoMode == osg::DisplaySettings::VERTICAL_INTERLACE || screens[i].stereoMode == osg::DisplaySettings::HORIZONTAL_INTERLACE || screens[i].stereoMode == osg::DisplaySettings::CHECKERBOARD)
        {
            m_stencil = true;
        }
    }
}

coVRConfig::~coVRConfig()
{
    if (debugLevel(2))
        fprintf(stderr, "delete coVRConfig\n");
}

bool
coVRConfig::debugLevel(int level) const
{
    if (level <= m_dLevel)
        return true;
    else
        return false;
}

bool coVRConfig::mouseNav() const
{
    return m_mouseNav;
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

int coVRConfig::numScreens() const
{
    return m_numScreens;
}

int coVRConfig::numWindows() const
{
    return m_numWindows;
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
