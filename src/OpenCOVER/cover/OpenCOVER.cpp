/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *	File			OpenCOVER.C				*
 *									*
 *	Description		renderer class				*
 *									*
 *	Author			D. Rainer 				*
 *				Uwe Woessner  				*
 *									*
 *	Date			20.08.07				*
 *				09.07.98 Performer C++ Interface	*
 *									*
 ************************************************************************/
#include <util/common.h>

#include "coVRTui.h"
#ifdef _WIN32
#include <util/XGetopt.h>
#else
#include <sys/types.h>
#include <signal.h>
#endif

#include <gpu/cudaglinterop.h>

#include <net/message.h>
#include <net/covise_socket.h>

#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include <config/CoviseConfig.h>
#ifdef DOTIMING
#include <util/coTimer.h>
#endif
#include "OpenCOVER.h"
#include "coCommandLine.h"
#include <osgUtil/RenderBin>
#include <osg/Group>
#include <osg/Geode>
#include <osgText/Font>
#include <osgText/Text>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osgGA/GUIActionAdapter>

#include <vrbclient/VRBClient.h>

#include "coVRAnimationManager.h"
#include "coVRCollaboration.h"
#include "coVRFileManager.h"
#include "coVRNavigationManager.h"
#include "coVRCommunication.h"
#include "coVRMSController.h"
#include "coVRPartner.h"
#include "coVRPluginList.h"
#include "VRWindow.h"
#include "VRViewer.h"
#include "VRSceneGraph.h"
#include "VRPinboard.h"
#include "coVRLighting.h"
#include "ARToolKit.h"
#include "VRVruiRenderInterface.h"
#include "coHud.h"
#include "coVRShader.h"
#include "coOnscreenDebug.h"
#include "coShutDownHandler.h" // added by Sebastian for singleton shutdown

#include <input/input.h>
#include <input/coMousePointer.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace opencover;
using namespace covise;
VRBClient *opencover::vrbc = NULL;
static char envOsgNotifyLevel[200];
static char envDisplay[1000];

static void usage()
{
    fprintf(stderr, "OpenCOVER\n");
    fprintf(stderr, "       (C) HLRS, University of Stuttgart (2004)\n\n");
    fprintf(stderr, "usage: cover [-v <viewpoints file>] [-s <collaborative config file>] [-h] <data file>\n\n");
    fprintf(stderr, "       -h : print this message\n");
    fprintf(stderr, "       -v : automatically load the indicated viewpoint file\n");
    fprintf(stderr, "       -s : collaborative VR configuration file, used by web interface\n");
}

//Signal handler
static void handler(int signo)
{
    if (cover->debugLevel(3))
        fprintf(stderr,"signal handler got signal %d\n", signo);
#ifndef _WIN32
    switch(signo)
    {
        case SIGTERM:
            coVRMSController::instance()->killClients();
            OpenCOVER::instance()->setExitFlag(true);
            break;
        case SIGTTIN:
        case SIGTTOU:
            break;
    }
#endif
}

static void installSignalHandlers()
{
#ifndef _WIN32
    struct sigaction act = { handler, 0, SA_SIGINFO };
    if (sigaction(SIGTTIN, &act, NULL) != 0)
    {
        fprintf(stderr, "failed to install handler for SIGTTIN: %s\n", strerror(errno));
    }
    if (sigaction(SIGTTOU, &act, NULL) != 0)
    {
        fprintf(stderr, "failed to install handler for SIGTTOU: %s\n", strerror(errno));
    }
#endif
}

void printDefinition()
{
    cout << "Module:      \""
         << "COVER"
         << "\"" << endl;
    cout << "Desc:        \""
         << "VR-Renderer"
         << "\"" << endl;
    cout << "Parameters:   " << 2 << endl;
    cout << "  \"Viewpoints\" \"Browser\" \"./default.vwp *.vwp\" \"Viewpoints\" \"START\"" << endl;
    cout << "  \"Plugins\" \"String\" \"\" \"Additional plugins\" \"START\"" << endl;
    cout << "  \"WindowID\" \"IntScalar\" \"0\" \"window ID to render to\" \"START\"" << endl;
    cout << "OutPorts:     " << 0 << endl;
    cout << "InPorts:     " << 1 << endl;
    cout << "  \""
         << "RenderData"
         << "\" \""
         << "Geometry|UnstructuredGrid|Points|StructuredGrid|Polygons|TriangleStrips|Lines"
         << "\" \""
         << "render geometry"
         << "\" \""
         << "req" << '"' << endl;
}

OpenCOVER *OpenCOVER::instance()
{
    return s_instance;
}

OpenCOVER *OpenCOVER::s_instance = NULL;

OpenCOVER::OpenCOVER(bool forceMpi)
    : m_visPlugin(NULL)
    , m_forceMpi(forceMpi)
{
    initCudaGlInterop();

#ifdef WIN32
    parentWindow = NULL;
#else
    parentWindow = 0;
#endif
}

#ifdef WIN32
OpenCOVER::OpenCOVER(HWND pw)
    : m_visPlugin(NULL)
    , m_forceMpi(false)
{
    initCudaGlInterop();
    parentWindow = pw;
}
#else
OpenCOVER::OpenCOVER(int pw)
    : m_visPlugin(NULL)
    , m_forceMpi(false)
{
    initCudaGlInterop();

    parentWindow = pw;
}
#endif

void OpenCOVER::waitForWindowID()
{
    bool validWindowID = false;
    while (!validWindowID)
    {
#if 0
      //FIXME: visPlugin()->preFrame() ?
      if(VRCoviseConnection::covconn)
      {
         VRCoviseConnection::covconn->update(true);
      }
#endif

        if (parentWindow != 0)
        {
            //sleep(5); // we have to wait, window is in creation
            validWindowID = true;
        }
    }
}

bool OpenCOVER::init()
{
    installSignalHandlers();

#ifdef _WIN32
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(2, 2);
    WSAStartup(wVersionRequested, &wsaData);
	// Require at least 4 processors, otherwise the process could occupy the machine.
	if (OpenThreads::GetNumberOfProcessors() >= 4)
	{
		SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	}
#endif

    m_visPlugin = NULL;
    Socket::initialize();
    s_instance = this;
    ignoreMouseEvents = false;

    /// This must be checked BEFORE windows are opened !!!!
    if (coCommandLine::argc() > 1 && 0 == strcmp(coCommandLine::argv(1), "-d"))
    {
        printDefinition();
    }

    int port = 0;
    char *addr = NULL;
    int myID = 0;
    if ((coCommandLine::argc() >= 5) && (!strcmp(coCommandLine::argv(1), "-c")))
    {
        myID = atoi(coCommandLine::argv(2));
        addr = coCommandLine::argv(3);
        port = atoi(coCommandLine::argv(4));
        coCommandLine::shift(4);
    }

    vrbHost = NULL;
    vrbPort = 0;
    int c = 0;
    std::string collaborativeOptionsFile, viewpointsFile;
    while ((c = getopt(coCommandLine::argc(), coCommandLine::argv(), "hdC:s:v:c:::")) != -1)
    {
        switch (c)
        {
        case 'd':
            printDefinition();
            exit(EXIT_SUCCESS);
            break;
        case 's':
            collaborativeOptionsFile = optarg;
            break;
        case 'v':
            viewpointsFile = optarg;
            break;
        case 'C':
        {
            std::cerr << "Optional Argument: " << optarg << std::endl;

            const char *sepChar = ":";
            char *sep = strstr(optarg, (const char *)sepChar);
            vrbHost = new char[strlen(optarg) - strlen(sep) + 1];
            char *tmpPort = new char[strlen(sep) + 1];
            strncpy(vrbHost, optarg, strlen(optarg) - strlen(sep));
            vrbHost[strlen(optarg) - strlen(sep)] = '\0';
            sep++;
            strncpy(tmpPort, sep, strlen(sep));
            tmpPort[strlen(sep)] = '\0';
            vrbPort = atoi(tmpPort);
            delete[] tmpPort;
            break;
        }
        case 'h':
            usage();
            exit(EXIT_SUCCESS);
        case '?':
            usage();
            exit(EXIT_SUCCESS);
        }
    }

    bool loadFiles = false;
    if (optind < coCommandLine::argc())
    {
        char *c = coCommandLine::argv(optind);
        while (*c != '\0')
        {
            if (!isdigit(*c)) // this is not a port number for a covise connection,
            //thus we load the files given in the command line
            {
                loadFiles = true;
                break;
            }
            c++;
        }
    }

    frameNum = 0;

    new coVRMSController(m_forceMpi, myID, addr, port);
    coVRMSController::instance()->startSlaves();
    coVRMSController::instance()->startupSync();

    collaborativeOptionsFile = coVRMSController::instance()->syncString(collaborativeOptionsFile);
    viewpointsFile = coVRMSController::instance()->syncString(viewpointsFile);

    coVRConfig::instance()->collaborativeOptionsFile = collaborativeOptionsFile;
    coVRConfig::instance()->viewpointsFile = viewpointsFile;

#ifdef _OPENMP
    std::string openmpThreads = coCoviseConfig::getEntry("value", "COVER.OMPThreads", "off");
    if (openmpThreads == "auto")
    {
        switch (omp_get_num_procs())
        {
        case 1:
            omp_set_num_threads(1);
            break;
        default:
            omp_set_num_threads(2);
        }
    }
    else if (openmpThreads == "off")
    {
        omp_set_num_threads(1);
    }
    else
    {
        omp_set_num_threads(coCoviseConfig::getInt("COVER.OMPThreads", 1));
    }
    std::cerr << "Compiled with OpenMP support, using a maximum of " << omp_get_max_threads() << " threads" << std::endl;
    omp_set_nested(true);
#endif

#ifndef _WIN32
    bool useDISPLAY = coCoviseConfig::isOn("COVER.HonourDisplay", false);
#ifdef __linux__
    if (getenv("LD_PRELOAD"))
    {
        if (strstr(getenv("LD_PRELOAD"), "faker.so"))
        {
            useDISPLAY = true;
            cerr << "Apparently running with VirtualGL, using DISPLAY environment variable" << endl;
        }
    }
#endif

    int debugLevel = coCoviseConfig::getInt("COVER.DebugLevel", 0);
    if (useDISPLAY && getenv("DISPLAY") == NULL)
    {
        useDISPLAY = false;
        cerr << "DISPLAY not set" << endl;
    }
    else if (useDISPLAY)
    {
        if (debugLevel > 1)
            cerr << "DISPLAY set to " << getenv("DISPLAY") << endl;
    }
    if (!useDISPLAY)
    {
        bool present = false;
        if (!coCoviseConfig::isOn("useDISPLAY", "COVER.PipeConfig.Pipe:0", false, &present))
        {
            std::string firstPipe = coCoviseConfig::getEntry("server", "COVER.PipeConfig.Pipe:0");
            strcpy(envDisplay, "DISPLAY=:");
            strcat(envDisplay, firstPipe.empty() ? "0" : firstPipe.c_str());
            if (firstPipe.empty())
            {
                cerr << "No PipeConfig for Pipe 0 found, using " << envDisplay << endl;
            }

            // do NOT use cover->debugLevel here : cover is not yet created!
            if (debugLevel > 1)
            {
                fprintf(stderr, "\nUsing '%s' as main Display\n", envDisplay);
            }
            putenv(envDisplay);
        }
    }
#endif
    int fsaa = coCoviseConfig::getInt("COVER.FSAAMode", -1);
    if (fsaa >= 0)
    {
        char *envStr = new char[200];
        sprintf(envStr, "__GL_FSAA_MODE=%d", fsaa);
        putenv(envStr);
    }
    int AnisotropicFiltering = coCoviseConfig::getInt("COVER.AnisotropicFiltering", -1);
    if (AnisotropicFiltering >= 0)
    {
        char *envStr = new char[200];
        sprintf(envStr, "__GL_LOG_MAX_ANISO=%d", AnisotropicFiltering);
        putenv(envStr);
    }
    if (coCoviseConfig::isOn("COVER.SyncToVBlank", false))
        putenv((char *)"__GL_SYNC_TO_VBLANK=1");
    else
        putenv((char *)"__GL_SYNC_TO_VBLANK=0");
    std::string syncDevice = coCoviseConfig::getEntry("device", "COVER.SyncToVBlank");
    if (!syncDevice.empty())
    {
        char *envString = new char[strlen(syncDevice.c_str()) + 60];
        sprintf(envString, "__GL_SYNC_DISPLAY_DEVICE=%s", syncDevice.c_str());
        putenv(envString);
    }

#ifndef _WIN32
    coVRConfig::instance()->m_useDISPLAY = useDISPLAY;
#endif
    cover = new coVRPluginSupport();
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "\nnew OpenCOVER\n");
    }
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "COVISEDIR: %s\n", getenv("COVISEDIR"));
        fprintf(stderr, "COVISE_PATH: %s\n", getenv("COVISE_PATH"));
        fprintf(stderr, "DISPLAY: %s\n", getenv("DISPLAY"));
        fprintf(stderr, "PWD: %s\n", getenv("PWD"));
    }


    exitFlag = false;

    readConfigFile();

    cover->updateTime();
    coVRMSController::instance()->syncTime();

    coVRPluginList::instance();
	Input::instance()->init();
    coVRTui::instance();

    ARToolKit::instance();

    if (cover->debugLevel(4))
        fprintf(stderr, "Calling pfConfig\n");

    osgUtil::RenderBin::setDefaultRenderBinSortMode(osgUtil::RenderBin::SORT_BY_STATE_THEN_FRONT_TO_BACK);

    VRWindow::instance();

    // init channels and view
    VRViewer::instance();

    new VRVruiRenderInterface();

    coVRShaderList::instance()->update();

    // init scene graph
    VRSceneGraph::instance()->init();
    VRViewer::instance()->setSceneData(cover->getScene());

	Input::instance()->update(); // requires scenegraph

    cover->setScale(coCoviseConfig::getFloat("COVER.DefaultScaleFactor", 1.f));

    // initialize communication
    bool loadCovisePlugin = false;
    if (loadFiles == false && coVRConfig::instance()->collaborativeOptionsFile.empty() && coCommandLine::argc() > 3 && vrbHost == NULL)
    {
        loadCovisePlugin = true;
        //fprintf(stderr, "need covise connection\n");

        // if there's an embedded OpenCOVER, then wait for a window ID
        for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
        {
            if (coVRConfig::instance()->windows[i].embedded)
            {
                waitForWindowID();
            }
        }
    }
    else
    {
        //fprintf(stderr, "no covise connection\n");
    }

    loadCovisePlugin = coVRMSController::instance()->syncBool(loadCovisePlugin);
    if (loadCovisePlugin)
    {
        m_visPlugin = coVRPluginList::instance()->addPlugin("CovisePlugin");
    }
    else
    {
        const char *vistlePlugin = getenv("VISTLE_PLUGIN");
        bool loadVistlePlugin = vistlePlugin && (coCommandLine::argc() == 3 || coCommandLine::argc() == 4);
        loadVistlePlugin = coVRMSController::instance()->syncBool(loadVistlePlugin);
        if (loadVistlePlugin)
        {
            loadFiles = false;
            m_visPlugin = coVRPluginList::instance()->addPlugin(vistlePlugin);
            if (!m_visPlugin)
            {
                m_visPlugin = coVRPluginList::instance()->addPlugin("VistlePlugin");
            }
        }
    }

    hud = coHud::instance();

    bool haveWindows = VRWindow::instance()->config();
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::SlaveData sd(sizeof(haveWindows));
        coVRMSController::instance()->readSlaves(&sd);
        for (size_t i=0; i<coVRMSController::instance()->getNumSlaves(); ++i)
        {
            if (!*(bool*)sd.data[i])
                haveWindows = false;
        }
    }
    else
    {
        coVRMSController::instance()->sendMaster(&haveWindows, sizeof(haveWindows));
    }
    haveWindows = coVRMSController::instance()->syncBool(haveWindows);
    if (!haveWindows)
        return false;

    VRViewer::instance()->config();

    string welcomeMessage = coCoviseConfig::getEntry("value", "COVER.WelcomeMessage", "Welcome to OpenCOVER at HLRS");
    hud->setText1(welcomeMessage.c_str());

    hud->setText2("startup");
    // initialize movable screen if there (IWR)
    hud->setText3("Tracking");

    bool showHud = coCoviseConfig::isOn("COVER.SplashScreen", true);
    if (showHud)
    {
        hud->show();
        hud->redraw();
    }

    // Connect to VRBroker, if available
    if (coVRMSController::instance()->isMaster())
    {
        if (vrbHost == NULL)
        {
            hud->setText2("connecting");
            hud->setText3("to VRB");
            hud->redraw();
            vrbc = new VRBClient("COVER", coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave());
            vrbc->connectToServer();
        }
        else
        {
            hud->setText2("connecting(VRB)");
            hud->setText3("AG mode");
            hud->redraw();
            vrbc = new VRBClient("COVER", vrbHost, vrbPort, coVRMSController::instance()->isSlave());
            vrbc->connectToServer();
        }
    }

    // setup Pinboard
    VRPinboard::instance();
    VRPinboard::instance()->configInteraction();
    coVRLighting::instance()->initMenu();

    hud->setText2("loading plugin");

    coVRPluginList::instance()->init();

    //fprintf(stderr,"isMaster %d\n",coVRMSController::instance()->isMaster());
    if (coVRMSController::instance()->isMaster())
    {
        int num = 0;
        if (loadFiles)
        {
            num = coCommandLine::argc() - optind;
            coVRMSController::instance()->sendSlaves(&num, sizeof(int));
            //fprintf(stderr,"NumToLoad %d\n",num);
            for (; optind < coCommandLine::argc(); optind++)
            {
                //fprintf(stderr,"Arg %d : %s",optind, coCommandLine::argv(optind));
                coVRMSController::instance()->loadFile(coCommandLine::argv(optind));
            }
        }
        else
        {

            coVRMSController::instance()->sendSlaves(&num, sizeof(int));
            //fprintf(stderr,"NumToLoad %d\n",num);
        }
    }
    else
    {
        int num = 0;
        coVRMSController::instance()->readMaster(&num, sizeof(int));
        //fprintf(stderr,"NumToLoad %d\n",num);
        for (int i = 0; i < num; i++)
            coVRMSController::instance()->loadFile(NULL);
    }

    ARToolKit::instance()->config(); // setup Rendering Node
    VRSceneGraph::instance()->config();
    coVRAnimationManager::instance();

    coVRTui::instance()->config();

    if (cover->debugLevel(5))
    {
        fprintf(stderr, "\nOpenCOVER::preparing rendering loop\n");
    }
    sum_time = 0;
    frameNum++;

    cover->updateTime();
    coVRMSController::instance()->syncTime();
    if (cover->debugLevel(2))
        cerr << "doneSync" << endl;

    old_fl_time = cover->frameRealTime();

    printFPS = coCoviseConfig::isOn("COVER.FPS", false);

#if 0
   sleep(coVRMSController::instance()->getID());
   std::cerr << "MS id=" << coVRMSController::instance()->getID() << ": pid=" << getpid() << std::endl;
   Input::instance()->printConfig();
   sleep(10);
#endif

    hud->hideLater();

    beginAppTraversal = VRViewer::instance()->elapsedTime();

    coVRShaderList::instance()->init();

    VRViewer::instance()->forceCompile(); // compile all OpenGL objects once after all files have been loaded
    
    coVRPluginList::instance()->init2();

    return true;
}

bool OpenCOVER::initDone()
{
    return (frameNum > 1);
}

void OpenCOVER::loop()
{
    while (!exitFlag && !VRViewer::instance()->done())
    {
        if(VRViewer::instance()->done())
            exitFlag = true;
        exitFlag = coVRMSController::instance()->syncBool(exitFlag);
        if (!exitFlag)
        {
            frame();
        }
    }
}

void OpenCOVER::handleEvents(int type, int state, int code)
{
    switch (type)
    {
    case (osgGA::GUIEventAdapter::SCROLL):
    case (osgGA::GUIEventAdapter::PUSH):
    case (osgGA::GUIEventAdapter::DRAG):
    case (osgGA::GUIEventAdapter::MOVE):
    case (osgGA::GUIEventAdapter::RELEASE):
    case (osgGA::GUIEventAdapter::DOUBLECLICK):
    {
        if (!ignoreMouseEvents)
        {
            Input::instance()->mouse()->handleEvent(type, state, code);
        }
    }
    break;
    case (osgGA::GUIEventAdapter::FRAME):
    {
    }
    break;
    case (osgGA::GUIEventAdapter::KEYDOWN):
    {
        if (!cover->isKeyboardGrabbed())
        {
            //fprintf(stderr, "\nKEYDOWN-Event - state: 0x%04x   code: %d   (0x%04x)\n", state, code, code);

            if ((state & osgGA::GUIEventAdapter::MODKEY_ALT) && (state & osgGA::GUIEventAdapter::MODKEY_SHIFT))
            {
                switch (code)
                {
                case 'T':
                case 't':
                    cerr << "calling: coTabletUI::instance()->close()" << endl;
                    coTabletUI::instance()->close();
                    break;
                }
            }
            else if (state & osgGA::GUIEventAdapter::MODKEY_ALT)
            {
                switch (code)
                {
                case 'b':
                    VRViewer::instance()->separation *= -1;
                    cerr << VRViewer::instance()->separation << endl;
                    break;

                case 'd':
                    coOnscreenDebug::instance()->toggleVisibility();
                    break;

#if 0
               case 'f':
                  cover->windows[0].rs->fullScreen(!cover->windows[0].rs->isFullScreen());
                  break;
#endif
                case 'n':
                    if (coVRMSController::instance()->getID() == 1)
                    {
                        VRViewer::instance()->separation *= -1;
                    }
                    cerr << VRViewer::instance()->separation << endl;
                    break;
                case 'm':
                    if (coVRMSController::instance()->getID() == 2)
                    {
                        VRViewer::instance()->separation *= -1;
                    }
                    cerr << VRViewer::instance()->separation << endl;
                    break;
                case 'z':
                    coVRConfig::instance()->m_worldAngle += 1;
                    cerr << coVRConfig::instance()->worldAngle() << endl;
                    break;
                case 't':
                    cerr << "calling: coTabletUI::instance()->tryConnect()" << endl;
                    coTabletUI::instance()->tryConnect();
                    break;
                case 'x':
                    coVRConfig::instance()->m_worldAngle -= 1;
                    cerr << coVRConfig::instance()->worldAngle() << endl;
                    break;
                }
            }
            else
            {
                switch (code)
                {
                case osgGA::GUIEventAdapter::KEY_Escape:
                case 'q':
                case 'Q':
                    if ((coVRConfig::instance()->numWindows() > 0) && coVRConfig::instance()->windows[0].embedded)
                    {
                        break; // embedded OpenCOVER ignores q
                    }
                    coVRPluginList::instance()->requestQuit(true);
                    // exit COVER, even if COVER has a vrb connection
                    exitFlag = true;
                    break;
                }
            }

            coVRNavigationManager::instance()->keyEvent(type, code, state);
            VRSceneGraph::instance()->keyEvent(type, code, state);
            coVRAnimationManager::instance()->keyEvent(type, code, state);
        }
        coVRPluginList::instance()->key(type, code, state);
    }
    break;
    case (osgGA::GUIEventAdapter::KEYUP):
    {
        if (!cover->isKeyboardGrabbed())
        {
            coVRNavigationManager::instance()->keyEvent(type, code, state);
            VRSceneGraph::instance()->keyEvent(type, code, state);
            coVRAnimationManager::instance()->keyEvent(type, code, state);
        }
        coVRPluginList::instance()->key(type, code, state);
    }
    break;
    case (osgGA::GUIEventAdapter::USER):
    {
        coVRPluginList::instance()->userEvent(state);
    }
    break;

    default:
    {
    }
    break;
    }
}

void OpenCOVER::frame()
{
    // NO MODIFICATION OF SCENEGRAPH DATA PRIOR TO THIS POINT
    //=========================================================
    //cerr << "-- OpenCOVER::frame" << endl;

    cover->updateTime();
    if (frameNum > 2)
    {
        coVRPluginList::instance()->prepareFrame();
    }
    coVRMSController::instance()->syncTime();

    //MARK0("COVER reading input devices");

    VRViewer::instance()->handleEvents(); // handle e.g. mouse events
    Input::instance()->update(); //update all hardware devices

    // wait for all cull and draw threads to complete.
    //
    coVRTui::instance()->update();

    // update window size
    VRWindow::instance()->update();

    coVRAnimationManager::instance()->update();
    // update transformations node according to interaction
    coVRNavigationManager::instance()->update();
    VRSceneGraph::instance()->update();
    coVRCollaboration::instance()->update();

    // update viewer position and channels
    if (Input::instance()->hasHead() && Input::instance()->isHeadValid())
        VRViewer::instance()->updateViewerMat(Input::instance()->getHeadMat());
    VRViewer::instance()->update();

    // print frame rate
    fl_time = cover->frameRealTime();

    sum_time += fl_time - old_fl_time;
    static float maxTime = -1;
    if (maxTime < fl_time - old_fl_time)
        maxTime = fl_time - old_fl_time;

    static int frameCount = 0;
    ++frameCount;
    if (sum_time > 5.0)
    {

        if (!coVRMSController::instance()->isSlave())
        {
            if (printFPS)
            {
                cout << "avg fps: " << frameCount / sum_time << ", min fps: " << 1.0 / maxTime << '\r' << flush;
            }
            coVRTui::instance()->updateFPS(1.0 / (fl_time - old_fl_time));
        }
        sum_time = 0;
        maxTime = -1;
        frameCount = 0;
    }
    old_fl_time = fl_time;

    // copy matrices to plugin support class
    // pointer ray intersection test
    // update update manager =:-|
    cover->update();

    //Remote AR update (send picture if required)
    if (ARToolKit::instance()->remoteAR)
        ARToolKit::instance()->remoteAR->update();

    interactionManager.update();

    if (frameNum > 2)
    {
        double beginTime = VRViewer::instance()->elapsedTime();

        // call preFrame for all plugins
        coVRPluginList::instance()->preFrame();

        if (VRViewer::instance()->getStats() && VRViewer::instance()->getStats()->collectStats("plugin"))
        {
            int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
            double endTime = VRViewer::instance()->elapsedTime();
            VRViewer::instance()->getStats()->setAttribute(fn, "Plugin begin time", beginTime);
            VRViewer::instance()->getStats()->setAttribute(fn, "Plugin end time", endTime);
            VRViewer::instance()->getStats()->setAttribute(fn, "Plugin time taken", endTime - beginTime);
        }
    }

    ARToolKit::instance()->update();

    coVRMSController::instance()->syncApp(frameNum++);

    // NO MODIFICATION OF SCENEGRAPH DATA AFTER THIS POINT

    if (cover->frameRealTime() > Input::instance()->mouse()->eventTime() + 1.5)
    {
        cover->setCursorVisible(false);
    }
    else if (coVRMSController::instance()->isMaster())
    {
        cover->setCursorVisible(coVRConfig::instance()->mouseNav());
    }

    coVRMSController::instance()->syncVRBMessages();

    if (VRViewer::instance()->getStats() && VRViewer::instance()->getStats()->collectStats("opencover"))
    {
        int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
        endAppTraversal = VRViewer::instance()->elapsedTime();
        VRViewer::instance()->getStats()->setAttribute(fn, "opencover begin time", beginAppTraversal);
        VRViewer::instance()->getStats()->setAttribute(fn, "opencover end time", endAppTraversal);
        VRViewer::instance()->getStats()->setAttribute(fn, "opencover time taken", endAppTraversal - beginAppTraversal);
        // update current frames stats
    }
    coVRShaderList::instance()->update();
    VRViewer::instance()->frame();
    beginAppTraversal = VRViewer::instance()->elapsedTime();
    if (frameNum > 2)
        coVRPluginList::instance()->postFrame();

    hud->update();

    //cerr << "OpenCOVER::frame EMD " << frameCount << endl;
}

void OpenCOVER::doneRendering()
{
    coVRMSController::instance()->killClients();

    if (cover->debugLevel(3))
        fprintf(stderr, "OpenCOVER: done with the loop\n");
}

OpenCOVER::~OpenCOVER()
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "\ndelete OpenCOVER\n");
    }
    VRViewer::instance()->stopThreading();
    coVRFileManager::instance()->unloadFile();
    delete coVRPluginList::instance();
    delete coVRTui::instance();
    //delete vrbHost;
    coVRPartnerList::instance()->reset();
    while (coVRPartnerList::instance()->current())
        coVRPartnerList::instance()->remove();
    // da sollte noch mehr geloescht werden

    cover->intersectedNode = NULL;
    delete VRPinboard::instance();
    delete VRSceneGraph::instance();
    delete VRViewer::instance();
    delete VRWindow::instance();

    delete ARToolKit::instance();

    coShutDownHandlerList::instance()->shutAllDown();
    delete coShutDownHandlerList::instance();

    if (cover->debugLevel(2))
    {
        fprintf(stderr, "\nThank you for using COVER!\nBye\n");
    }
    delete Input::instance();
    delete cover;
    cover = NULL;
#ifdef DOTIMING
    coTimer::quit();
#endif
}

void OpenCOVER::setExitFlag(bool flag)
{
    if (cover)
    {
        if (cover && cover->debugLevel(3))
        {
            fprintf(stderr, "OpenCOVER::setExitFlag\n");
        }

        if (flag && vrbc && vrbc->isConnected())
        {
            // do not quit, if we are connected to a vr Broker
            // but close connection to Covise
            //CoviseRender::appmod->getConnectionList()->remove(vrbc->getConnection());
            if (m_visPlugin)
                coVRPluginList::instance()->unload(m_visPlugin);
            m_visPlugin = NULL;
            exitFlag = false;
        }
        else
            exitFlag = flag;
    }
}

void
OpenCOVER::readConfigFile()
{

    std::string line = coCoviseConfig::getEntry("COVER.Notify");
    if (!line.empty())
    {
        sprintf(envOsgNotifyLevel, "OSG_NOTIFY_LEVEL=%s", line.c_str());
        putenv(envOsgNotifyLevel);
        /* ALWAYS
      FATAL
      WARN
      NOTICE
      INFO
      DEBUG_INFO
      DEBUG_FP*/
    }
}

void
OpenCOVER::quitCallback(void * /*sceneGraph*/, buttonSpecCell * /*spec*/)
{
    OpenCOVER::instance()->setExitFlag(true);
    coVRPluginList::instance()->requestQuit(true);
    if (vrbc)
        delete vrbc;
    vrbc = NULL;
    OpenCOVER::instance()->setExitFlag(true);
    // exit COVER, even if COVER has a vrb connection
}

coVRPlugin *
OpenCOVER::visPlugin() const
{

    return m_visPlugin;
}
