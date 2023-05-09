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
#include <clocale>
#include <util/unixcompat.h>

#include <gpu/cudaglinterop.h>

#include <net/message.h>
#include <net/covise_socket.h>
#include <messages/CRB_EXEC.h>

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

#include <vrb/client/SharedStateManager.h>
#include <vrb/client/VRBClient.h>

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
#include "MarkerTracking.h"
#include "coHud.h"
#include "coVRShader.h"
#include "coOnscreenDebug.h"
#include "coShutDownHandler.h" // added by Sebastian for singleton shutdown
#include "QuitDialog.h"
#include "Deletable.h"

#include <input/input.h>
#include <input/coMousePointer.h>
#include <input/deviceDiscovery.h>
#include "ui/VruiView.h"
#include "ui/TabletView.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Group.h"
#include "ui/Manager.h"
#include <OpenConfig/access.h>


#ifdef _OPENMP
#include <omp.h>
#endif

using namespace opencover;
using namespace covise;
static char envOsgNotifyLevel[200];
static char envDisplay[1000];

static void usage()
{
    fprintf(stderr, "OpenCOVER\n");
    fprintf(stderr, "       (C) HLRS, University of Stuttgart (2004)\n\n");
    fprintf(stderr, "usage: cover [-g sessionName] [-C vrbServer:port] [-v <viewpoints file>] [-s <collaborative config file>] [-h] <data file>\n\n");
    fprintf(stderr, "       -h : print this message\n");
    fprintf(stderr, "       -v : automatically load the indicated viewpoint file\n");
    fprintf(stderr, "       -s : collaborative VR configuration file, used by web interface\n");
    fprintf(stderr, "       -C : vrb to connect to in form host:port\n");
    fprintf(stderr, "       -g : Collaborative Session to load\n");
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
OpenCOVER::OpenCOVER(const MPI_Comm *comm, pthread_barrier_t *shmBarrier)
    : m_visPlugin(NULL)
    , m_forceMpi(true)
    , m_comm(*comm)
    , m_shmBarrier(shmBarrier)
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
	// always parse floats with . as separator
    setlocale(LC_NUMERIC, "C");
    int dl = coCoviseConfig::getInt("COVER.DebugLevel", 0);

    if (init())
    {
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

    covise::Socket::initialize();
    setlocale(LC_NUMERIC, "C");

    installSignalHandlers();

#ifdef _WIN32
	// Require at least 4 processors, otherwise the process could occupy the machine.
	if (OpenThreads::GetNumberOfProcessors() >= 4)
	{
		SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	}
#endif

    std::string startCommand = coCoviseConfig::getEntry("COVER.StartCommand");
    if (!startCommand.empty())
    {
        int ret = system(startCommand.c_str());
        if (ret == -1)
        {
            std::cerr << "COVER.StartCommand " << startCommand << " failed: " << strerror(errno) << std::endl;
        }
        else if (ret > 0)
        {
            std::cerr << "COVER.StartCommand " << startCommand << " returned exit code  " << ret << std::endl;
        }
    }

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


    int c = 0;
    std::string collaborativeOptionsFile, viewpointsFile;
    while ((c = getopt(coCommandLine::argc(), coCommandLine::argv(), "hdC:s:v:c:::g:")) != -1)
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

            std::vector<char> vrbHostdata(strlen(optarg) + 1);
            char *vrbHost = vrbHostdata.data();
            strcpy(vrbHost, optarg);
            char *sep = strchr(vrbHost, ':');
            int tcpPort = 0, udpPort = 0;
            if (sep)
            {
                *sep = '\0';
                ++sep;
                tcpPort = atoi(sep);
                sep = strchr(sep + 1, ':');
            }
            if(sep)
            {
                *sep = '\0';
                ++sep;
                udpPort = atoi(sep);
            }
            m_vrbCredentials.reset(new vrb::VrbCredentials(std::string{vrbHost}, tcpPort, udpPort));
            break;
        }
		case 'g':
		{
			m_startSession = optarg;
		}
		break;
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
        new coVRMSController(&m_comm, m_shmBarrier);
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
    coVRConfig::instance()->m_stereoState = coVRMSController::instance()->allReduceOr(coVRConfig::instance()->m_stereoState);

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
    {
        putenv((char *)"__GL_SYNC_TO_VBLANK=1");
        fprintf(stderr,"__GL_SYNC_TO_VBLANK=1\n");
    }
    else
    {
        putenv((char *)"__GL_SYNC_TO_VBLANK=0");
        fprintf(stderr,"__GL_SYNC_TO_VBLANK=0\n");
	}
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

    const char *vistlePlugin = getenv("VISTLE_PLUGIN");
    bool loadVistlePlugin = vistlePlugin;
    m_loadVistlePlugin = coVRMSController::instance()->syncBool(loadVistlePlugin);

	coVRCommunication::instance();
    interactionManager.initializeRemoteLock();
    cover = new coVRPluginSupport();
    coVRCommunication::instance()->init();
    cover->initUI();
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

    MarkerTracking::instance();

    if (cover->debugLevel(4))
        fprintf(stderr, "Calling pfConfig\n");

    osgUtil::RenderBin::setDefaultRenderBinSortMode(osgUtil::RenderBin::SORT_BY_STATE_THEN_FRONT_TO_BACK);

    VRWindow::instance();

    // init channels and view
    VRViewer::instance();

    coVRAnimationManager::instance();
    coVRShaderList::instance();

    // init scene graph
    VRSceneGraph::instance()->init();
    coVRShaderList::instance()->update();
    VRViewer::instance()->setSceneData(cover->getScene());

	Input::instance()->update(); // requires scenegraph

    cover->setScale(coCoviseConfig::getFloat("COVER.DefaultScaleFactor", 1.f));

    bool haveWindows = VRWindow::instance()->config();
    haveWindows = coVRMSController::instance()->allReduceOr(haveWindows);
    if (!haveWindows)
        return false;

    // initialize communication
    bool loadCovisePlugin = false;
    if (!m_loadVistlePlugin && loadFiles == false && coVRConfig::instance()->collaborativeOptionsFile.empty() && coCommandLine::argc() > 3 && m_vrbCredentials == NULL)
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

    cover->vruiView = new ui::VruiView;
    cover->ui->addView(cover->vruiView);

    hud = coHud::instance();
    if (m_loadVistlePlugin)
    {
        loadFiles = false;
        m_visPlugin = coVRPluginList::instance()->addPlugin("Vistle", coVRPluginList::Vis);
        if (!m_visPlugin)
        {
            fprintf(stderr, "failed to load Vistle plugin\n");
            exit(1);
        }
    }
    else
    {
        loadCovisePlugin = coVRMSController::instance()->syncBool(loadCovisePlugin);
        if (loadCovisePlugin)
        {
            m_visPlugin = coVRPluginList::instance()->addPlugin("COVISE", coVRPluginList::Vis);
            if (!m_visPlugin)
            {
                fprintf(stderr, "failed to load COVISE plugin\n");
                exit(1);
            }
        }
    }

    VRViewer::instance()->config();

    hud->setText2("loading plugins");
    hud->redraw();

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

    coVRLighting::instance()->initMenu();

    MarkerTracking::instance()->config(); // setup Rendering Node
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

    m_quitGroup = new ui::Group(cover->fileMenu, "QuitGroup");
    m_quitGroup->setText("");
    m_quit = new ui::Action(m_quitGroup, "Quit");
    m_quit->setShortcut("q");
    m_quit->addShortcut("Q");
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

    auto tab = coVRTui::instance()->mainFolder;
    cover->ui->addView(new ui::TabletView("mainTui", tab));
    tabletUIs.push_back(coTabletUI::instance());
    tabletTabs.push_back(tab);

    auto mapeditorTui = new coTabletUI("localhost", 31803);
    tab = new coTUITabFolder(mapeditorTui, "root");
    cover->ui->addView(new ui::TabletView("mapeditor", tab));
    tabletUIs.push_back(mapeditorTui);
    tabletTabs.push_back(tab);
    for (auto tui: tabletUIs)
    {
        tui->tryConnect();
        tui->update();
    }

    hud->setText2("initialising plugins");
    hud->redraw();

    coVRPluginList::instance()->init();

    hud->redraw();

    // Connect to VRBroker, if available
    if (coVRMSController::instance()->isMaster())
    {
        if (loadCovisePlugin)//use covise session
        {
            auto cmdExec = getExecFromCmdArgs(coCommandLine::instance()->argc(), coCommandLine::instance()->argv());
            std::stringstream ss;
            ss << "covise" << cmdExec.vrbClientIdOfController() << "_" << cmdExec.moduleId();
            m_startSession = ss.str();
            m_vrbCredentials.reset(new vrb::VrbCredentials{cmdExec.vrbCredentials()});
        }
        if (m_vrbCredentials) {
            hud->setText2("connecting(VRB)");
            hud->setText3("AG mode");
            hud->redraw();
        } else {
            hud->setText2("connecting");
            hud->setText3("to VRB");
        }
        hud->redraw();
        startVrbc();
    }

    double loadStart = cover->currentTime();
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
    double loadEnd = cover->currentTime();

    coVRPluginList::instance()->init2();
    double init2End = cover->currentTime();

    if (!coVRConfig::instance()->continuousRendering())
    {
        if (cover->debugLevel(2))
        {
            fprintf(stderr, "OpenCOVER: disabling continuous rendering\n");
        }
        VRViewer::instance()->setRunFrameScheme(osgViewer::Viewer::ON_DEMAND);
    }

    if (cover->viewOptionsMenu) {
        auto cr = new ui::Button(cover->viewOptionsMenu, "ContinuousRendering");
        cr->setText("Continuous rendering");
        cr->setState(coVRConfig::instance()->continuousRendering());
        cr->setCallback([this](bool state){
           if (state)
               VRViewer::instance()->setRunFrameScheme(osgViewer::Viewer::CONTINUOUS);
           else
               VRViewer::instance()->setRunFrameScheme(osgViewer::Viewer::ON_DEMAND);
        });
    }

    VRViewer::instance()->forceCompile(); // compile all OpenGL objects once after all files have been loaded

    VRViewer::instance()->clearWindow=true; // clear the whole window to get rid of white remains that sticked there during startup (Who knows where they are comming from)

    frame();
    double frameEnd = cover->currentTime();
    hud->hideLater();

    Input::instance()->discovery()->init();

    config::Access config;
    config.setErrorHandler(); // make parse errors in configuration files non-fatal

    m_initialized = true;

    if (cover->debugLevel(1))
    {
        std::cerr << std::endl << "INIT TIMES:"
                  << " load " << loadEnd-loadStart << "s"
                  << ", init2 " << init2End-loadEnd << "s"
                  << ", 1st frame " << frameEnd-init2End << "s"
                  << std::endl;
    }

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
    while (true)
    {
        if(VRViewer::instance()->done())
            exitFlag = true;
        exitFlag = coVRMSController::instance()->syncBool(exitFlag);
        if (exitFlag)
            break;
        frame();
    }

    VRViewer::instance()->disableSync();

    std::string exitCommand = coCoviseConfig::getEntry("COVER.ExitCommand");
    if (!exitCommand.empty())
    {
        int ret = system(exitCommand.c_str());
        if (ret == -1)
        {
            std::cerr << "COVER.ExitCommand " << exitCommand << " failed: " << strerror(errno) << std::endl;
        }
        else if (ret > 0)
        {
            std::cerr << "COVER.ExitCommand " << exitCommand << " returned exit code  " << ret << std::endl;
        }
    }

    m_visPlugin = NULL; // prevent any new messages from being sent
    coVRPluginList::instance()->unloadAllPlugins(coVRPluginList::Vis);
    coVRFileManager::instance()->unloadFile();
    frame();
    coVRPluginList::instance()->unloadAllPlugins();
    frame();
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

    bool render = m_renderNext;
    m_renderNext = false;

    //MARK0("COVER reading input devices");

    cover->updateTime();

    if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("frame_rate"))
    {
        auto stats = VRViewer::instance()->getViewerStats();
        int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
        double updateTime = VRViewer::instance()->elapsedTime();
        double deltaUpdateTime = updateTime - lastUpdateTime;
        lastUpdateTime = updateTime;
        stats->setAttribute(fn, "Update duration", deltaUpdateTime);
        stats->setAttribute(fn, "Update rate", 1.0/deltaUpdateTime);
    }


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
    if (coVRCollaboration::instance()->update())
    {
        
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of collaborative action" << std::endl;
        render = true;
    }
    if (vrb::SharedStateManager::instance())
    {
        vrb::SharedStateManager::instance()->frame(cover->frameTime());
    }
    // update viewer position and channels
    if (cover->isViewerGrabbed())
    {
        if (coVRPluginList::instance()->viewerGrabber()->updateViewer())
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because of plugin updated viewer" << std::endl;
            render = true;
        }
    }
    else
    {
        if (Input::instance()->hasHead() && Input::instance()->isHeadValid())
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because of head tracking" << std::endl;
            render = true;
            VRViewer::instance()->updateViewerMat(Input::instance()->getHeadMat());
        }
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
    if (MarkerTracking::instance()->remoteAR)
        MarkerTracking::instance()->remoteAR->update();

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

	double beginPluginTime = VRViewer::instance()->elapsedTime();
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

    if (VRViewer::instance()->getRunFrameScheme() != osgViewer::Viewer::ON_DEMAND)
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because getRunFrameScheme()!=ON_DEMAND" << std::endl;
        render = true;
    }

    if (VRViewer::instance()->checkNeedToDoFrame())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because checkNeedToDoFrame()==true" << std::endl;
        render = true;
    }

    if (m_renderNext)
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because rendering of next frame was requested" << std::endl;
        render = true;
    }

    if (coVRMSController::instance()->syncVRBMessages())
    {
        if (cover->debugLevel(4))
            std::cerr << "OpenCOVER::frame: rendering because of VRB message" << std::endl;
        render = true;
    }

    render = coVRMSController::instance()->syncBool(render);
    if (!render)
    {
        int maxfd = -1;
        fd_set fds;
        FD_ZERO(&fds);
        for (const auto &fd: m_watchedFds) {
            FD_SET(fd, &fds);
            if (maxfd < fd)
                maxfd = fd;
        }
        struct timeval tenms {0, 10000};
        int nready = select(maxfd+1, &fds, &fds, &fds, &tenms);

        if (nready > 0)
        {
            if (cover->debugLevel(4))
                std::cerr << "OpenCOVER::frame: rendering because of filedescriptor activity" << std::endl;
            render = true;
        }
        render = coVRMSController::instance()->syncBool(render);
    }

    if (!render)
    {
        return render;
    }

    if (frameNum > 2)
    {
        double beginPreFrameTime = VRViewer::instance()->elapsedTime();

        // call preFrame for all plugins
        coVRPluginList::instance()->preFrame();

        if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("plugin"))
        {
            int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
            double endTime = VRViewer::instance()->elapsedTime();
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "Plugin begin time", beginPluginTime);
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "Plugin end time", endTime);
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "Plugin time taken", endTime - beginPluginTime);

            VRViewer::instance()->getViewerStats()->setAttribute(fn, "preframe begin time", beginPreFrameTime);
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "preframe end time", endTime);
            VRViewer::instance()->getViewerStats()->setAttribute(fn, "preframe time taken", endTime - beginPreFrameTime);
        }
    }
    MarkerTracking::instance()->update();

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

    coVRShaderList::instance()->update();

    if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("opencover"))
    {
        int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
        endAppTraversal = VRViewer::instance()->elapsedTime();
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "opencover begin time", beginAppTraversal);
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "opencover end time", endAppTraversal);
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "opencover time taken", endAppTraversal - beginAppTraversal);
        // update current frames stats
    }
    VRViewer::instance()->frame();
    beginAppTraversal = VRViewer::instance()->elapsedTime();
    if (frameNum > 2)
        coVRPluginList::instance()->postFrame();

    if (hud->update())
        m_renderNext = true;

    double frameTime = VRViewer::instance()->elapsedTime();
    double frameDuration = frameTime - lastFrameTime;
    lastFrameTime = frameTime;
    frameDurations.push_back(frameDuration);
    if (frameDurations.size() > 20)
        frameDurations.pop_front();

    if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("frame_rate"))
    {
        auto stats = VRViewer::instance()->getViewerStats();
        int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
        double maxDuration = -1.;
        for (auto &d: frameDurations) {
            maxDuration = std::max(maxDuration, d);
        }
        stats->setAttribute(fn, "Max frame duration", maxDuration);
    }

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
    VRViewer::instance()->stopThreading();
    VRViewer::instance()->setSceneData(NULL);
    //delete vrbHost;
    delete coVRAnimationManager::instance();
    delete coVRNavigationManager::instance();
    delete coVRCommunication::instance();
    delete coVRPartnerList::instance();
    delete MarkerTracking::instance();
    delete coVRTui::instance();

    cover->intersectedNode = NULL;
    delete VRSceneGraph::instance();
    delete coVRShaderList::instance();
    delete coVRLighting::instance();
    delete VRViewer::instance();
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
    delete coVRMSController::instance();
    delete coVRConfig::instance();
    delete coCommandLine::instance();
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
    terminateOnCoverQuit = coVRMSController::instance()->syncBool(terminateOnCoverQuit);
    if (terminateOnCoverQuit)
        coVRPluginList::instance()->requestQuit(true);
    m_vrbc.reset(nullptr);
    setExitFlag(true);
    // exit COVER, even if COVER has a vrb connection
}

coVRPlugin *
OpenCOVER::visPlugin() const
{

    return m_visPlugin;
}

size_t OpenCOVER::numTuis() const
{
    return tabletUIs.size();
}

coTabletUI *OpenCOVER::tui(size_t idx) const
{
    assert(tabletTabs.size() == tabletUIs.size());
    if (idx >= tabletUIs.size())
        return nullptr;
    return tabletUIs[idx];
}

coTUITabFolder *OpenCOVER::tuiTab(size_t idx) const
{
    assert(tabletTabs.size() == tabletUIs.size());
    if (idx >= tabletTabs.size())
        return nullptr;

    return tabletTabs[idx];
}

bool OpenCOVER::watchFileDescriptor(int fd)
{
    return m_watchedFds.insert(fd).second;
}

bool OpenCOVER::unwatchFileDescriptor(int fd)
{
    auto it = m_watchedFds.find(fd);
    if (it == m_watchedFds.end())

    m_watchedFds.erase(it);
    return true;
}

const vrb::VRBClient *OpenCOVER::vrbc() const
{
    return m_vrbc.get();
}

vrb::VRBClient *OpenCOVER::vrbc()
{
    return m_vrbc.get();
}

void OpenCOVER::startVrbc()
{
    if (!m_vrbc)
        restartVrbc();
}

void OpenCOVER::restartVrbc()
{
    if (exitFlag) {
        m_vrbc.reset();
        return;
    }

    if (m_loadVistlePlugin) {
        class PluginMessageSender : public covise::MessageSenderInterface {

          public:
            bool sendMessage(const covise::Message *msg) const override {
                if (OpenCOVER::instance()->m_visPlugin)
                    return coVRPluginList::instance()->sendVisMessage(msg);
                return false;
            }

            bool sendMessage(const UdpMessage *msg) const override {
                return false;
            }
        };

        auto sender = new PluginMessageSender();
        if (cover->debugLevel(2))
            std::cerr << "starting VRB client with credentials from Vistle, session=" << m_visPlugin->collaborativeSessionId() << std::endl;
        m_vrbc.reset(new vrb::VRBClient(covise::Program::opencover, sender,
                                        coVRMSController::instance()->isSlave(),
                                        false));
        m_startSession = m_visPlugin->collaborativeSessionId();
    }
    else if (m_vrbCredentials)
    {
        if (cover->debugLevel(2))
            std::cerr << "starting VRB client with credentials from memory" << std::endl;
        m_vrbc.reset(new vrb::VRBClient(covise::Program::opencover, *m_vrbCredentials, coVRMSController::instance()->isSlave(),true));
    }
    else
    {
        if (cover->debugLevel(2))
            std::cerr << "starting VRB client with options from " << coVRConfig::instance()->collaborativeOptionsFile << std::endl;
        m_vrbc.reset(new vrb::VRBClient(covise::Program::opencover, coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave(),true));
    }
    m_vrbc->connectToServer(m_startSession);
}

bool OpenCOVER::useVistle() const
{
    return m_loadVistlePlugin;
}

bool OpenCOVER::isVRBconnected() const
{
    return m_vrbc && m_vrbc->isConnected();
}
