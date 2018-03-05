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
#include <util/unixcompat.h>

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
#include "coVRLighting.h"
#include "ARToolKit.h"
#include "coHud.h"
#include "coVRShader.h"
#include "coOnscreenDebug.h"
#include "coShutDownHandler.h" // added by Sebastian for singleton shutdown
#include "QuitDialog.h"
#include "Deletable.h"

#include <input/input.h>
#include <input/coMousePointer.h>

#include "ui/VruiView.h"
#include "ui/TabletView.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Group.h"

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

OpenCOVER::OpenCOVER()
    : m_visPlugin(NULL)
    , m_forceMpi(false)
#ifdef HAS_MPI
    , m_comm(MPI_COMM_WORLD)
#endif
    , m_renderNext(true)
{
    initCudaGlInterop();

#ifdef WIN32
    parentWindow = NULL;
#else
    parentWindow = 0;
#endif
}

#ifdef HAS_MPI
OpenCOVER::OpenCOVER(const MPI_Comm *comm)
    : m_visPlugin(NULL)
    , m_forceMpi(true)
    , m_comm(*comm)
    , m_renderNext(true)
{
    initCudaGlInterop();

#ifdef WIN32
    parentWindow = NULL;
#else
    parentWindow = 0;
#endif
}
#endif


#ifdef WIN32
OpenCOVER::OpenCOVER(HWND pw)
    : m_visPlugin(NULL)
    , m_forceMpi(false)
#ifdef HAS_MPI
    , m_comm(MPI_COMM_WORLD)
#endif
    , m_renderNext(true)
{
    initCudaGlInterop();
    parentWindow = pw;
}
#else
OpenCOVER::OpenCOVER(int pw)
    : m_visPlugin(NULL)
    , m_forceMpi(false)
#ifdef HAS_MPI
    , m_comm(MPI_COMM_WORLD)
#endif
    , m_renderNext(true)
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
        else
        {
            usleep(10000);
        }
    }
}

bool OpenCOVER::run()
{
    int dl = coCoviseConfig::getInt("COVER.DebugLevel", 0);

    if (init())
    {
        if (!coVRConfig::instance()->continuousRendering())
        {
            if (dl >= 1)
            {
                fprintf(stderr, "OpenCOVER: disabling continuous rendering\n");
            }
            VRViewer::instance()->setRunFrameScheme(osgViewer::Viewer::ON_DEMAND);
        }

        if (dl >= 2)
            fprintf(stderr, "OpenCOVER: Entering main loop\n\n");

        loop();

        doneRendering();
        if (dl >= 2)
            fprintf(stderr, "OpenCOVER: Leaving main loop\n\n");
    }
    else
    {
        fprintf(stderr, "OpenCOVER: Start-up failed\n\n");
        return false;
    }

    if (dl >= 1)
        fprintf(stderr, "OpenCOVER: Shutting down\n\n");

    return true;
}

bool OpenCOVER::init()
{
    if (m_initialized)
        return true;

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

    bool useVirtualGL = false;
    if (getenv("VGL_ISACTIVE"))
    {
        useVirtualGL = true;
    }
    coVRConfig::instance()->m_useVirtualGL = useVirtualGL;

#ifdef HAS_MPI
    if (m_forceMpi)
    {
        new coVRMSController(&m_comm);
    }
    else
#endif
    {
        new coVRMSController(myID, addr, port);
    }
    coVRMSController::instance()->startSlaves();
    coVRMSController::instance()->startupSync();

    collaborativeOptionsFile = coVRMSController::instance()->syncString(collaborativeOptionsFile);
    viewpointsFile = coVRMSController::instance()->syncString(viewpointsFile);

    coVRConfig::instance()->collaborativeOptionsFile = collaborativeOptionsFile;
    coVRConfig::instance()->viewpointsFile = viewpointsFile;

#ifdef _OPENMP
    std::string openmpThreads = coCoviseConfig::getEntry("value", "COVER.OMPThreads", "default");
    if (openmpThreads == "default")
    {
    }
    else if (openmpThreads == "auto")
    {
        switch (omp_get_num_procs())
        {
        case 1:
            omp_set_num_threads(1);
            break;
        case 2:
        case 3:
            omp_set_num_threads(2);
            break;
        default:
            omp_set_num_threads(4);
            break;
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
    if (useVirtualGL)
    {
        useDISPLAY = true;
        cerr << "Apparently running with VirtualGL, using DISPLAY environment variable" << endl;
    }

    int debugLevel = coCoviseConfig::getInt("COVER.DebugLevel", 0);
    if (useDISPLAY && getenv("DISPLAY") == NULL)
    {
        useDISPLAY = false;
        cerr << "DISPLAY not set, defaulting to DISPLAY=:0" << endl;
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
        else if (!getenv("DISPLAY"))
        {
            strcpy(envDisplay, "DISPLAY=:0");
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

#if 0
    m_clusterStats = new ui::Button(cover->viewOptionsMenu, "ClusterStats");
    m_clusterStats->setText("Cluster statistics");
    m_clusterStats->setState(coVRMSController::instance()->drawStatistics());
    m_clusterStats->setCallback([](bool state){
        coVRMSController::instance()->setDrawStatistics(state);
    });
#endif

    exitFlag = false;

    readConfigFile();

    cover->updateTime();

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

    coVRAnimationManager::instance();
    coVRShaderList::instance()->update();

    // init scene graph
    VRSceneGraph::instance()->init();
    VRViewer::instance()->setSceneData(cover->getScene());

	Input::instance()->update(); // requires scenegraph

    cover->setScale(coCoviseConfig::getFloat("COVER.DefaultScaleFactor", 1.f));

    bool haveWindows = VRWindow::instance()->config();

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
    hud = coHud::instance();

    loadCovisePlugin = coVRMSController::instance()->syncBool(loadCovisePlugin);
    if (loadCovisePlugin)
    {
        m_visPlugin = coVRPluginList::instance()->addPlugin("COVISE");
        if (!m_visPlugin)
        {
            fprintf(stderr, "failed to load COVISE plugin\n");
            exit(1);
        }
    }
    else
    {
        const char *vistlePlugin = getenv("VISTLE_PLUGIN");
        bool loadVistlePlugin = vistlePlugin;
        loadVistlePlugin = coVRMSController::instance()->syncBool(loadVistlePlugin);
        if (loadVistlePlugin)
        {
            loadFiles = false;
            m_visPlugin = coVRPluginList::instance()->addPlugin("Vistle");
            if (!m_visPlugin)
            {
                fprintf(stderr, "failed to load Vistle plugin\n");
                exit(1);
            }
        }
    }


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

    coVRPluginList::instance()->loadDefault(); // vive and other tracking system plugins have to be loaded before Input is initialized

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

    coVRLighting::instance()->initMenu();

    hud->setText2("loading plugin");

    coVRPluginList::instance()->init();

    ARToolKit::instance()->config(); // setup Rendering Node
    VRSceneGraph::instance()->config();

    coVRTui::instance()->config();

    if (cover->debugLevel(5))
    {
        fprintf(stderr, "\nOpenCOVER::preparing rendering loop\n");
    }
    sum_time = 0;
    frameNum++;

    cover->updateTime();
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

    beginAppTraversal = VRViewer::instance()->elapsedTime();

    coVRShaderList::instance()->init();

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
    hud->hideLater();

    VRViewer::instance()->forceCompile(); // compile all OpenGL objects once after all files have been loaded
    
    coVRPluginList::instance()->init2();

    cover->vruiView = new ui::VruiView;
    cover->ui->addView(cover->vruiView);

    cover->ui->addView(new ui::TabletView(coVRTui::instance()->mainFolder));
    tabletUIs.push_back(coTabletUI::instance());

    auto mapeditorTui = new coTabletUI("localhost", 31803);
    cover->ui->addView(new ui::TabletView("mapeditor", mapeditorTui));
    tabletUIs.push_back(mapeditorTui);

    m_quitGroup = new ui::Group(cover->fileMenu, "QuitGroup");
    m_quitGroup->setText("");
    m_quit = new ui::Action(m_quitGroup, "Quit");
    m_quit->setShortcut("q");
    m_quit->addShortcut("Q");
    m_quit->addShortcut("Esc");
    m_quit->setCallback([this](){
#if 1
        requestQuit();
#else
        auto qd = new QuitDialog;
        qd->show();
#endif
    });
    m_quit->setIcon("application-exit");
    if ((coVRConfig::instance()->numWindows() > 0) && coVRConfig::instance()->windows[0].embedded)
    {
        m_quit->setEnabled(false);
        m_quit->setVisible(false);
    }

    m_initialized = true;
    return true;
}

bool OpenCOVER::initDone()
{
    return (frameNum > 1);
}

class CheckVisitor: public osg::NodeVisitor 
{
 public:
   CheckVisitor()
       : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}

   void apply(osg::Group &group)
   {
       if (group.getNumChildrenRequiringUpdateTraversal() > 0) {
           std::cerr << group.getName() << ": " << group.getNumChildrenRequiringUpdateTraversal() << std::endl;
       }
       traverse(group);
   }

   void apply(osg::Node &node)
   {
       if (!node.getName().empty() || node.getUpdateCallback()) {
           std::cerr << node.getName() << ": " << (node.getUpdateCallback()?"U":".") << std::endl;
       }
       traverse(node);
   }
};

void OpenCOVER::loop()
{
    while (!exitFlag)
    {
        if(VRViewer::instance()->done())
            exitFlag = true;
        exitFlag = coVRMSController::instance()->syncBool(exitFlag);
        if (exitFlag)
        {
            VRViewer::instance()->disableSync();
            frame();
            frame();
        }
        else
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
                    for (auto tui: tabletUIs)
                        tui->close();
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
                    for (auto tui: tabletUIs)
                        tui->tryConnect();
                    break;
                case 'x':
                    coVRConfig::instance()->m_worldAngle -= 1;
                    cerr << coVRConfig::instance()->worldAngle() << endl;
                    break;
                }
            }

            cover->ui->keyEvent(type, state, code);
            coVRNavigationManager::instance()->keyEvent(type, code, state);
            VRSceneGraph::instance()->keyEvent(type, code, state);
        }
        coVRPluginList::instance()->key(type, code, state);
    }
    break;
    case (osgGA::GUIEventAdapter::KEYUP):
    {
        if (!cover->isKeyboardGrabbed())
        {
            cover->ui->keyEvent(type, state, code);
            coVRNavigationManager::instance()->keyEvent(type, code, state);
            VRSceneGraph::instance()->keyEvent(type, code, state);
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

bool OpenCOVER::frame()
{
    // NO MODIFICATION OF SCENEGRAPH DATA PRIOR TO THIS POINT
    //=========================================================
    //cerr << "-- OpenCOVER::frame" << endl;

    DeletionManager::the()->run();

    bool render = false;

    //MARK0("COVER reading input devices");

    cover->updateTime();
    
    // update window size and process events
    VRWindow::instance()->update();
    if (VRViewer::instance()->handleEvents())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of mouse input" << std::endl;
        // handle e.g. mouse events
        render = true;
        m_renderNext = true; // for possible delayed button release
    }
    if (Input::instance()->update())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of input" << std::endl;
        render = true;
    }
    if (Input::instance()->hasRelative() && Input::instance()->isRelativeValid())
    {
        const auto &mat = Input::instance()->getRelativeMat();
        if (!mat.isIdentity())
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because of active relative input" << std::endl;
            render = true;
        }
    }

    // wait for all cull and draw threads to complete.
    //
    coVRTui::instance()->update();

    if (coVRAnimationManager::instance()->update())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of animation" << std::endl;
        render = true;
    }
    // update transformations node according to interaction
    coVRNavigationManager::instance()->update();
    VRSceneGraph::instance()->update();
    coVRCollaboration::instance()->update();

    // update viewer position and channels
    if (Input::instance()->hasHead() && Input::instance()->isHeadValid())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of head tracking" << std::endl;
        render = true;
        VRViewer::instance()->updateViewerMat(Input::instance()->getHeadMat());
    }
    if (VRViewer::instance()->update())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of VRViewer" << std::endl;
        render = true;
    }

    // copy matrices to plugin support class
    // pointer ray intersection test
    // update update manager =:-|
    cover->update();
    for (auto tui: tabletUIs)
    {
        if (tui->update())
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because of tabletUI on " << tui->connectedHost << std::endl;
            render = true;
        }
    }

    //Remote AR update (send picture if required)
    if (ARToolKit::instance()->remoteAR)
        ARToolKit::instance()->remoteAR->update();

    if (interactionManager.update())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of interactionManager" << std::endl;
        render = true;
    }
    if (cover->ui->update())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of ui update" << std::endl;
        render = true;
    }
    if (cover->ui->sync())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of ui sync" << std::endl;
        render = true;
    }

    if (frameNum > 2)
    {
        if (coVRPluginList::instance()->update())
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because of plugins" << std::endl;
            render = true;
        }
    }
    else
    {
        render = true;
    }

    if (!render)
    {
        if (VRViewer::instance()->getRunFrameScheme() == osgViewer::Viewer::ON_DEMAND)
        {
            if (!VRViewer::instance()->checkNeedToDoFrame())
            {
                if (!m_renderNext)
                {
                    usleep(10000);
                    return false;
                }
                m_renderNext = false;
                if (cover->debugLevel(4))
                    std::cerr << "OpenCOVER::frame: rendering because rendering next frame was requested" << std::endl;
            }
            else
            {
                if (cover->debugLevel(4))
                    std::cerr << "OpenCOVER::frame: rendering because checkNeedToDoFrame()==true" << std::endl;
            }
        }
        else
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because getRunFrameScheme()!=ON_DEMAND" << std::endl;
        }
    }

    if (frameNum > 2)
    {
        double beginTime = VRViewer::instance()->elapsedTime();

        // call preFrame for all plugins
        coVRPluginList::instance()->preFrame();

        if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("plugin"))
        {
            int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
            double endTime = VRViewer::instance()->elapsedTime();
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "Plugin begin time", beginTime);
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "Plugin end time", endTime);
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "Plugin time taken", endTime - beginTime);
        }
    }
    ARToolKit::instance()->update();

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

    coVRMSController::instance()->syncApp(frameNum++);

    // NO MODIFICATION OF SCENEGRAPH DATA AFTER THIS POINT

    if (coVRMSController::instance()->isMaster() && cover->frameRealTime() < Input::instance()->mouse()->eventTime() + 1.5)
    {
        cover->setCursorVisible(coVRConfig::instance()->mouseNav());
    }
    else
    {
        cover->setCursorVisible(false);
    }

    coVRMSController::instance()->syncVRBMessages();

    if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("opencover"))
    {
        int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
        endAppTraversal = VRViewer::instance()->elapsedTime();
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "opencover begin time", beginAppTraversal);
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "opencover end time", endAppTraversal);
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "opencover time taken", endAppTraversal - beginAppTraversal);
        // update current frames stats
    }
    coVRShaderList::instance()->update();
    VRViewer::instance()->frame();
    beginAppTraversal = VRViewer::instance()->elapsedTime();
    if (frameNum > 2)
        coVRPluginList::instance()->postFrame();

    hud->update();

    //cerr << "OpenCOVER::frame EMD " << frameCount << endl;
    return render;
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
#if 0
    if (m_visPlugin)
        coVRPluginList::instance()->unload(m_visPlugin);
#endif
    m_visPlugin = NULL;
    coVRFileManager::instance()->unloadFile();
    coVRPluginList::instance()->unloadAllPlugins();
    VRViewer::instance()->stopThreading();
    VRViewer::instance()->setSceneData(NULL);
    //delete vrbHost;
    delete coVRPartnerList::instance();
    delete coVRAnimationManager::instance();
    delete coVRNavigationManager::instance();
    delete coVRCommunication::instance();
    delete ARToolKit::instance();
    delete coVRTui::instance();

    cover->intersectedNode = NULL;
    delete VRSceneGraph::instance();
    delete coVRShaderList::instance();
    delete coVRLighting::instance();
    delete VRViewer::instance();
    delete coVRConfig::instance();
    delete VRWindow::instance();

    delete coVRPluginList::instance();

    coShutDownHandlerList::instance()->shutAllDown();
    delete coShutDownHandlerList::instance();

    for (auto tui: tabletUIs)
        delete tui;
    tabletUIs.clear();

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
#if 0
            if (m_visPlugin)
                coVRPluginList::instance()->unload(m_visPlugin);
#endif
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
OpenCOVER::requestQuit()
{
    setExitFlag(true);
    bool terminateOnCoverQuit = coCoviseConfig::isOn("COVER.TerminateCoviseOnQuit", false);
    if (getenv("COVISE_TERMINATE_ON_QUIT"))
    {
        terminateOnCoverQuit = true;
    }
    if (terminateOnCoverQuit)
        coVRPluginList::instance()->requestQuit(true);
    delete vrbc;
    vrbc = NULL;
    setExitFlag(true);
    // exit COVER, even if COVER has a vrb connection
}

coVRPlugin *
OpenCOVER::visPlugin() const
{

    return m_visPlugin;
}
