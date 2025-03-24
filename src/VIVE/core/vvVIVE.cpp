/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>

#include "vvTui.h"
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

#include "vvPluginSupport.h"
#include "vvConfig.h"
#include <config/CoviseConfig.h>
#ifdef DOTIMING
#include <util/coTimer.h>
#endif
#include "vvVIVE.h"
#include "vvCommandLine.h"

#include <vrb/client/SharedStateManager.h>
#include <vrb/client/VRBClient.h>

#include <vsg/core/Version.h>
#include <vsg/io/Logger.h>
#include <vsgXchange/all.h>
#include <vsg/io/read.h>
#include <vsg/app/RenderGraph.h>
#include <vsg/app/View.h>
#include <vsg/all.h>

#include "vvPluginList.h"
#include "vvMSController.h"

#include "vvWindows.h"
#include "vvViewer.h"
#include "vvAnimationManager.h"
#include "vvCollaboration.h"
#include "vvFileManager.h"
#include "vvNavigationManager.h"
#include "vvCommunication.h"
#include "vvPartner.h"
#include "vvLighting.h"
#include "vvMarkerTracking.h"
#include "vvHud.h"
#include "vvShader.h"
#include "vvOnscreenDebug.h"
#include "vvShutDownHandler.h" // added by Sebastian for singleton shutdown
#include "vvQuitDialog.h"
#include "vvDeletable.h"

#include <input/input.h>
#include <input/vvMousePointer.h>
#include <input/deviceDiscovery.h>
#include "ui/VruiView.h"
#include "ui/TabletView.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Group.h"
#include "ui/Manager.h"
#include "vvSceneGraph.h"
#include "../OpenConfig/access.h"
#include "vvVIVE.h"


#ifdef _OPENMP
#include <omp.h>
#endif

using namespace vive;
using namespace covise;
static char envOsgNotifyLevel[200];
static char envDisplay[1000];
class vvVive;


static void usage()
{
    fprintf(stderr, "vvVIVE\n");
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
        vsg::info("signal handler got signal", signo," \n");
#ifndef _WIN32
    switch(signo)
    {
        case SIGTERM:
            vvMSController::instance()->killClients();
            vvVIVE::instance()->setExitFlag(true);
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

vvVIVE *vvVIVE::instance()
{
    return s_instance;
}

vvVIVE *vvVIVE::s_instance = NULL;

vvVIVE::vvVIVE()
    : m_visPlugin(NULL)
    , m_forceMpi(false)
#ifdef HAS_MPI
    , m_comm(MPI_COMM_WORLD)
#endif
    , m_renderNext(true)
{

#ifdef WIN32
    parentWindow = NULL;
#else
    parentWindow = 0;
#endif
}

#ifdef HAS_MPI
vvVIVE::vvVIVE(const MPI_Comm *comm, pthread_barrier_t *shmBarrier)
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
vvVIVE::vvVIVE(HWND pw)
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
vvVIVE::vvVIVE(int pw)
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

void vvVIVE::waitForWindowID()
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

bool vvVIVE::run()
{
	// always parse floats with . as separator
    setlocale(LC_NUMERIC, "C");
    int dl = coCoviseConfig::getInt("VIVE.DebugLevel", 0);

    if (init())
    {
        if (dl >= 2)
            fprintf(stderr, "vvVIVE: Entering main loop\n\n");

        loop();

        doneRendering();
        if (dl >= 2)
            fprintf(stderr, "vvVIVE: Leaving main loop\n\n");
    }
    else
    {
        fprintf(stderr, "vvVIVE: Start-up failed\n\n");
        return false;
    }

    return true;
}

bool vvVIVE::init()
{
    if (m_initialized)
        return true;

    covise::Socket::initialize();

    installSignalHandlers();





#ifdef _WIN32

    const auto processor_count = std::thread::hardware_concurrency();
	// Require at least 4 processors, otherwise the process could occupy the machine.
	if (processor_count >= 4)
	{
		SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	}
#endif

    std::string startCommand = coCoviseConfig::getEntry("VIVE.StartCommand");
    if (!startCommand.empty())
    {
        std::cerr << "Running COVER.StartCommand " << startCommand << std::flush;
        int ret = system(startCommand.c_str());
        if (ret == -1)
        {
            std::cerr << " failed: " << strerror(errno) << std::endl;
        }
        else if (ret > 0)
        {
            std::cerr << " returned exit code  " << ret << std::endl;
        }
        else
        {
            std::cerr << "." << ret << std::endl;
        }
    }

    m_visPlugin = NULL;
    s_instance = this;
    ignoreMouseEvents = false;

    /// This must be checked BEFORE windows are opened !!!!
    if (vvCommandLine::argc() > 1 && 0 == strcmp(vvCommandLine::argv(1), "-d"))
    {
        printDefinition();
    }

    int port = 0;
    char *addr = NULL;
    int myID = 0;
    if ((vvCommandLine::argc() >= 5) && (!strcmp(vvCommandLine::argv(1), "-c")))
    {
        myID = atoi(vvCommandLine::argv(2));
        addr = vvCommandLine::argv(3);
        port = atoi(vvCommandLine::argv(4));
        vvCommandLine::shift(4);
    }


    int c = 0;
    std::string collaborativeOptionsFile, viewpointsFile;
    bool saveAndExit = false;
    int saveFormat = 0;
    while ((c = getopt(vvCommandLine::argc(), vvCommandLine::argv(), "SIhdC:s:v:c:::g:")) != -1)
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
        case 'S':
            saveFormat = 0;
            saveAndExit = true;;
            break;
        case 'I':
            saveFormat = 1;
            saveAndExit = true;;
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
            m_vrbCredentials.reset( new vrb::VrbCredentials(std::string{vrbHost}, tcpPort, udpPort));
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
    if (optind < vvCommandLine::argc())
    {
        char *c = vvCommandLine::argv(optind);
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

    
#ifdef HAS_MPI
    if (m_forceMpi)
    {
        new vvMSController(&m_comm, m_shmBarrier);
    }
    else
#endif
    {
        new vvMSController(myID, addr, port);
    }
    vvMSController::instance()->startSlaves();
    vvMSController::instance()->startupBarrier();

    collaborativeOptionsFile = vvMSController::instance()->syncString(collaborativeOptionsFile);
    viewpointsFile = vvMSController::instance()->syncString(viewpointsFile);

    vvConfig::instance()->collaborativeOptionsFile = collaborativeOptionsFile;
    vvConfig::instance()->viewpointsFile = viewpointsFile;

    //vvConfig::instance()->m_stereoState = vvMSController::instance()->allReduceOr(vvConfig::instance()->m_stereoState);

#ifdef _OPENMP
    std::string openmpThreads = coCoviseConfig::getEntry("value", "VIVE.OMPThreads", "default");
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
        omp_set_num_threads(coCoviseConfig::getInt("VIVE.OMPThreads", 1));
    }
    std::cerr << "Compiled with OpenMP support, using a maximum of " << omp_get_max_threads() << " threads" << std::endl;
    omp_set_nested(true);
#endif

#ifndef _WIN32
#ifdef USE_X11
    bool useDISPLAY = coCoviseConfig::isOn("VIVE.HonourDisplay", false);

    int debugLevel = coCoviseConfig::getInt("VIVE.DebugLevel", 0);
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
        if (!coCoviseConfig::isOn("useDISPLAY", "VIVE.PipeConfig.Pipe:0", false, &present))
        {
            std::string firstPipe = coCoviseConfig::getEntry("server", "VIVE.PipeConfig.Pipe:0");
            strcpy(envDisplay, "DISPLAY=:");
            strcat(envDisplay, firstPipe.empty() ? "0" : firstPipe.c_str());
            if (firstPipe.empty())
            {
                cerr << "No PipeConfig for Pipe 0 found, using " << envDisplay << endl;
            }

            // do NOT use vv->debugLevel here : cover is not yet created!
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
    vvConfig::instance()->m_useDISPLAY = useDISPLAY;
#endif
#endif

    const char *vistlePlugin = getenv("VISTLE_PLUGIN");
    bool loadVistlePlugin = vistlePlugin;
    m_loadVistlePlugin = vvMSController::instance()->syncBool(loadVistlePlugin);
    if (m_loadVistlePlugin)
    {
        m_vistlePlugin = vvMSController::instance()->syncString(vistlePlugin);
    }
    
    vvCommunication::instance();
    interactionManager.initializeRemoteLock();
    vv = vvPluginSupport::instance();
    
    vvCommunication::instance()->init();
    vv->initUI();
    if (vv->debugLevel(2))
    {
        fprintf(stderr, "\nnew vvVIVE\n");
    }
    if (vv->debugLevel(3))
    {
        fprintf(stderr, "COVISEDIR: %s\n", getenv("COVISEDIR"));
        fprintf(stderr, "COVISE_PATH: %s\n", getenv("COVISE_PATH"));
        fprintf(stderr, "DISPLAY: %s\n", getenv("DISPLAY"));
        fprintf(stderr, "PWD: %s\n", getenv("PWD"));
    }

#if 0
    m_clusterStats = new ui::Button(vv->viewOptionsMenu, "ClusterStats");
    m_clusterStats->setText("Cluster statistics");
    m_clusterStats->setState(vvMSController::instance()->drawStatistics());
    m_clusterStats->setCallback([](bool state){
        vvMSController::instance()->setDrawStatistics(state);
    });
#endif

    exitFlag = false;

    vvPluginSupport::instance()->updateTime();

   
    vvPluginList::instance();


    	Input::instance()->init();

    auto mainTui = vvTabletUI::instance();
    auto vrtui = new vvTui(nullptr);
    vrtui->config();
    pushTui(mainTui, vrtui);
    auto tab = vrtui->mainFolder;
    vv->ui->addView(new ui::TabletView("mainTui", tab));

    MarkerTracking::instance();

    if (vv->debugLevel(4))
        fprintf(stderr, "Calling pfConfig\n");
    /*
    osgUtil::RenderBin::setDefaultRenderBinSortMode(osgUtil::RenderBin::SORT_BY_STATE_THEN_FRONT_TO_BACK);
    */
    vvWindows::instance();

    // init channels and view
    vvViewer::instance();
    

    vvAnimationManager::instance();
    vvShaderList::instance();

    // init scene graph
    
    vvSceneGraph::instance()->init();
    vvShaderList::instance()->update();
    /*
    auto pointLight = vsg::PointLight::create();
    pointLight->name = "point";
    pointLight->color.set(1.0f, 1.0f, 0.0);
    pointLight->intensity = static_cast<float>(1.0);
    pointLight->position.set(static_cast<float>(0.0), static_cast<float>(-100.0), static_cast<float>(0.0));
    pointLight->radius = 5000;*/

    //vv->getScene()->addChild(pointLight);


	Input::instance()->update(); // requires scenegraph
    
    vv->setScale(coCoviseConfig::getFloat("VIVE.DefaultScaleFactor", 1.f));


    /*std::stringstream str;
    auto loadedScene = vsg::MatrixTransform::create();
    vsg::ref_ptr<vsg::Node> vpb = vsg::read_cast<vsg::Node>("c:\\QSync\\visnas\\Data\\Suedlink\\out\\vpb_DGM1m_FDOP20\\vpb_DGM1m_FDOP20.ive", vvPluginSupport::instance()->options);

    loadedScene->addChild(vpb);
    loadedScene->matrix = vsg::rotate(1.5, 1.0, 0.0, 0.0) * vsg::translate(-518740.0, -5966100.0, 0.0);
    vv->getObjectsRoot()->addChild(loadedScene);
    vvPluginSupport::instance()->setScale(1000);*/

    bool haveWindows = vvWindows::instance()->config();
    haveWindows = vvMSController::instance()->allReduceOr(haveWindows);
    if (!haveWindows)
        return false;
        
    // initialize communication
    bool loadCovisePlugin = false;
    if (!m_loadVistlePlugin && loadFiles == false && vvConfig::instance()->collaborativeOptionsFile.empty() && vvCommandLine::argc() > 3 && m_vrbCredentials == NULL)
    {
        loadCovisePlugin = true;
        //fprintf(stderr, "need covise connection\n");

        // if there's an embedded vvVIVE, then wait for a window ID
        for (int i = 0; i < vvConfig::instance()->numWindows(); i++)
        {
            if (vvConfig::instance()->windows[i].embedded)
            {
                waitForWindowID();
            }
        }
    }
    else
    {
        //fprintf(stderr, "no covise connection\n");
    }
    
    vv->vruiView = new ui::VruiView;
    vv->ui->addView(vv->vruiView);

    hud = vvHud::instance();
    

    vvViewer::instance()->config();

    
    vvShaderList::instance()->init();

    hud->setText2("loading plugins");
    hud->redraw();
    
    if (m_loadVistlePlugin)
    {
        loadFiles = false;
        m_visPlugin = vvPluginList::instance()->addPlugin(m_vistlePlugin.c_str(), vvPluginList::Vis);
        if (!m_visPlugin)
        {
            fprintf(stderr, "failed to load Vistle plugin %s\n", m_vistlePlugin.c_str());
            exit(1);
        }
    }
    else
    {
        loadCovisePlugin = vvMSController::instance()->syncBool(loadCovisePlugin);
        if (loadCovisePlugin)
        {
            m_visPlugin = vvPluginList::instance()->addPlugin("COVISE", vvPluginList::Vis);
            if (!m_visPlugin)
            {
                fprintf(stderr, "failed to load COVISE plugin\n");
                exit(1);
            }
            
            auto mapeditorTui = new vvTabletUI("localhost", 31803);
            auto mapeditorVrTui = new vvTui(mapeditorTui);
            mapeditorVrTui->config();
            pushTui(mapeditorTui, mapeditorVrTui);
            tab = tuiTab(numTuis() - 1);
            vv->ui->addView(new ui::TabletView("mapeditor", tab));
        }
    }
    vvPluginList::instance()->loadDefault(); // vive and other tracking system plugins have to be loaded before Input is initialized

    string welcomeMessage = coCoviseConfig::getEntry("value", "VIVE.WelcomeMessage", "Welcome to vvVIVE at HLRS");
    
    hud->setText1(welcomeMessage.c_str());

    hud->setText2("startup");
    // initialize movable screen if there (IWR)
    hud->setText3("Tracking");

    bool showHud = coCoviseConfig::isOn("VIVE.SplashScreen", true);
    if (showHud)
    {
        hud->show();
        hud->redraw();
    }

    vvLighting::instance()->initMenu();

    MarkerTracking::instance()->config(); // setup Rendering Node

    
    vvSceneGraph::instance()->config();

    if (vv->debugLevel(5))
    {
        fprintf(stderr, "\nvvVIVE::preparing rendering loop\n");
    }
    sum_time = 0;
    frameNum++;

    vv->updateTime();
    if (vv->debugLevel(2))
        cerr << "doneSync" << endl;

    old_fl_time = vv->frameRealTime();

    printFPS = coCoviseConfig::isOn("VIVE.FPS", false);

#if 0
   sleep(vvMSController::instance()->getID());
   std::cerr << "MS id=" << vvMSController::instance()->getID() << ": pid=" << getpid() << std::endl;
   Input::instance()->printConfig();
   sleep(10);
#endif
   
    //beginAppTraversal = vvViewer::instance()->elapsedTime();

    m_quitGroup = new ui::Group(vv->fileMenu, "QuitGroup");
    m_quitGroup->setText("");
    m_quit = new ui::Action(m_quitGroup, "Quit");
    m_quit->setShortcut("q");
    m_quit->addShortcut("Q");
    m_quit->setCallback([this](){
#if 1
        requestQuit();
#else
        auto qd = new vvQuitDialog;
        qd->show();
#endif
    });
    m_quit->setIcon("application-exit");
    if ((vvConfig::instance()->numWindows() > 0) && vvConfig::instance()->windows[0].embedded)
    {
        m_quit->setEnabled(false);
        m_quit->setVisible(false);
    }

    hud->setText2("initialising plugins");
    hud->redraw();
    
    vvPluginList::instance()->init();
    
    hud->redraw();

    for (auto &tui: tabletUIs)
    {
        tui->tryConnect();
        tui->update();
    }
    
    // Connect to VRBroker, if available
    if (vvMSController::instance()->isMaster())
    {
        if (loadCovisePlugin)//use covise session
        {
            auto cmdExec = getExecFromCmdArgs(vvCommandLine::instance()->argc(), vvCommandLine::instance()->argv());
            std::stringstream ss;
            ss << "covise" << cmdExec.vrbClientIdOfController() << "_" << cmdExec.moduleId();
            m_startSession = ss.str();
            m_vrbCredentials.reset( new vrb::VrbCredentials{cmdExec.vrbCredentials()});
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

    double loadStart = vv->currentTime();
    //fprintf(stderr,"isMaster %d\n",vvMSController::instance()->isMaster());
    if (vvMSController::instance()->isMaster())
    {
        int num = 0;
        if (loadFiles)
        {
            num = vvCommandLine::argc() - optind;
            vvMSController::instance()->sendSlaves(&num, sizeof(int));
            //fprintf(stderr,"NumToLoad %d\n",num);
            for (; optind < vvCommandLine::argc(); optind++)
            {
                //fprintf(stderr,"Arg %d : %s",optind, vvCommandLine::argv(optind));
                vvMSController::instance()->loadFile(vvCommandLine::argv(optind));
                if (saveAndExit)
                {
                    std::string saveFile = vvCommandLine::argv(optind);
                    if(saveFormat == 0)
                    saveFile = saveFile.substr(0,saveFile.length() - 3) + "obj";
                    else if(saveFormat == 1)
                        saveFile = saveFile.substr(0, saveFile.length() - 3) + "ive";
                    else
                        saveFile = saveFile.substr(0, saveFile.length() - 3) + "osg";
                    //osgDB::writeNodeFile(*vv->getObjectsRoot(), saveFile.c_str());
                }
            }
            if (saveAndExit)
            {
                exit(0);
            }
        }
        else
        {

            vvMSController::instance()->sendSlaves(&num, sizeof(int));
            //fprintf(stderr,"NumToLoad %d\n",num);
        }
    }
    else
    {
        int num = 0;
        vvMSController::instance()->readMaster(&num, sizeof(int));
        //fprintf(stderr,"NumToLoad %d\n",num);
        for (int i = 0; i < num; i++)
            vvMSController::instance()->loadFile(NULL);
    }
    double loadEnd = vv->currentTime();

    vvPluginList::instance()->init2();
    double init2End = vv->currentTime();
    
    if (!vvConfig::instance()->continuousRendering())
    {
        if (vv->debugLevel(2))
        {
            fprintf(stderr, "vvVIVE: disabling continuous rendering\n");
        }
        //vvViewer::instance()->setRunFrameScheme(osgViewer::Viewer::ON_DEMAND);
    }

    if (vv->viewOptionsMenu) {
        auto cr = new ui::Button(vv->viewOptionsMenu, "ContinuousRendering");
        cr->setText("Continuous rendering");
        cr->setState(vvConfig::instance()->continuousRendering());
        cr->setCallback([this](bool state){
            vvConfig::instance()->setContinuousRendering(state);
        });
    }


    vvViewer::instance()->clearWindow=true; // clear the whole window to get rid of white remains that sticked there during startup (Who knows where they are comming from)
    
    //frame();

    double frameEnd = vv->currentTime();
    
    hud->hideLater();

    Input::instance()->discovery()->init();
    
    config::Access config;
    config.setErrorHandler(); // make parse errors in configuration files non-fatal

    m_initialized = true;

    if (vv->debugLevel(1))
    {
        std::cerr << std::endl << "INIT TIMES:"
                  << " load " << loadEnd-loadStart << "s"
                  << ", init2 " << init2End-loadEnd << "s"
                  << ", 1st frame " << frameEnd-init2End << "s"
                  << std::endl;
    }

    // set up defaults and read command line arguments to override them






// add close handler to respond to the close window button and pressing escape


    vvViewer::instance()->InitialCompile();

    return true;
}

bool vvVIVE::initDone()
{
    return (frameNum > 1);
}
/*
class CheckVisitor: public vsg::NodeVisitor 
{
 public:
   CheckVisitor()
       : vsg::NodeVisitor(vsg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}

   void apply(vsg::Group &group)
   {
       if (group.children.sizeRequiringUpdateTraversal() > 0) {
           std::cerr << group.getName() << ": " << group.children.sizeRequiringUpdateTraversal() << std::endl;
       }
       traverse(group);
   }

   void apply(vsg::Node &node)
   {
       if (!node.getName().empty() || node.getUpdateCallback()) {
           std::cerr << node.getName() << ": " << (node.getUpdateCallback()?"U":".") << std::endl;
       }
       traverse(node);
   }
};*/

void vvVIVE::loop()
{
    while (true)
    {
        //if(vvViewer::instance()->done())
        //    exitFlag = true;
        exitFlag = vvMSController::instance()->syncBool(exitFlag);
        if (exitFlag)
            break;
        frame();
    }
    
    if (vv->debugLevel(1))
        fprintf(stderr, "vvVIVE: Shutting down\n\n");
    
    vvViewer::instance()->disableSync();
    

    std::string exitCommand = coCoviseConfig::getEntry("VIVE.ExitCommand");
    if (!exitCommand.empty())
    {
        int ret = system(exitCommand.c_str());
        if (ret == -1)
        {
            std::cerr << "VIVE.ExitCommand " << exitCommand << " failed: " << strerror(errno) << std::endl;
        }
        else if (ret > 0)
        {
            std::cerr << "VIVE.ExitCommand " << exitCommand << " returned exit code  " << ret << std::endl;
        }
    }

    m_visPlugin = NULL; // prevent any new messages from being sent
    vvPluginList::instance()->unloadAllPlugins(vvPluginList::Vis);
    vvFileManager::instance()->unloadFile();
    frame();
    vvPluginList::instance()->unloadAllPlugins();
    frame();
}


bool vvVIVE::frame()
{
    bool render = m_renderNext;
    m_renderNext = false;



    
    // NO MODIFICATION OF SCENEGRAPH DATA PRIOR TO THIS POINT
    //=========================================================
    //cerr << "-- vvVIVE::frame" << endl;

    DeletionManager::the()->run();


    //MARK0("COVER reading input devices");
    
    vv->updateTime();

    // update window size and process events
    vvWindows::instance()->update();

    vvViewer::instance()->handleEvents();
    
    if (Input::instance()->update())
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of input" << std::endl;
        render = true;
    }
    // copy matrices to plugin support class
    // This must be done right after reading input devices before any use of Head and Hand matrices
    // pointer ray intersection test
    // update update manager =:-|
    vv->update();
    if (Input::instance()->hasRelative() && Input::instance()->isRelativeValid())
    {
        const auto &mat = Input::instance()->getRelativeMat();
        /*if (!mat.isIdentity())
        {
            if (vv->debugLevel(4))
                std::cerr << "vvVIVE::frame: rendering because of active relative input" << std::endl;
            render = true;
        }*/
    }

    // update viewer position and channels
    if (vv->isViewerGrabbed())
    {
        if (vvPluginList::instance()->viewerGrabber()->updateViewer())
        {
            if (vv->debugLevel(4))
                std::cerr << "vvVIVE::frame: rendering because of plugin updated viewer" << std::endl;
            render = true;
        }
    }
    else
    {
        if (Input::instance()->hasHead() && Input::instance()->isHeadValid())
        {
            if (vv->debugLevel(4))
                std::cerr << "vvVIVE::frame: rendering because of head tracking" << std::endl;
            render = true;
            vvViewer::instance()->updateViewerMat(Input::instance()->getHeadMat());
        }
    }
    vvViewer::instance()->vvUpdate();

    // wait for all cull and draw threads to complete.
    //
    for (int i = 0; i < numTuis(); i++)
    {
        vrTui(i)->update();
        tui(i)->update();
    }

    if (vvAnimationManager::instance()->update())
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of animation" << std::endl;
        render = true;
    }
    // update transformations node according to interaction
    vvNavigationManager::instance()->update();
    
    vvSceneGraph::instance()->update();
    
    if (vvCollaboration::instance()->update())
    {
        
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of collaborative action" << std::endl;
        render = true;
    }
    
    if (vrb::SharedStateManager::instance())
    {
        vrb::SharedStateManager::instance()->frame(vv->frameTime());
    }
    for (auto& tui : tabletUIs)
    {
        if (tui->update())
        {
            if (vv->debugLevel(4))
                std::cerr << "vvVIVE::frame: rendering because of tabletUI on " << tui->connectedHost << std::endl;
            render = true;
        }
    }

    //Remote AR update (send picture if required)
    if (MarkerTracking::instance()->remoteAR)
        MarkerTracking::instance()->remoteAR->update();
    

    if (interactionManager.update())
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of interactionManager" << std::endl;
        render = true;
    }
    if (vv->ui->update())
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of ui update" << std::endl;
        render = true;
    }
    if (vv->ui->sync())
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of ui sync" << std::endl;
        render = true;
    }

	//double beginPluginTime = vvViewer::instance()->elapsedTime();
    if (frameNum > 2)
    {
        if (vvPluginList::instance()->update())
        {
            if (vv->debugLevel(4))
                std::cerr << "vvVIVE::frame: rendering because of plugins" << std::endl;
            render = true;
        }
    }
    else
    {
        render = true;
    }

    if (m_renderNext)
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because rendering of next frame was requested" << std::endl;
        render = true;
    }

    if (vvMSController::instance()->syncVRBMessages())
    {
        if (vv->debugLevel(4))
            std::cerr << "vvVIVE::frame: rendering because of VRB message" << std::endl;
        render = true;
    }

    render = vvMSController::instance()->syncBool(render);
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
            if (vv->debugLevel(4))
                std::cerr << "vvVIVE::frame: rendering because of filedescriptor activity" << std::endl;
            render = true;
        }
        render = vvMSController::instance()->syncBool(render);
    }

    if (!render)
    {
        return render;
    }
    if (frameNum > 2)
    {
        //double beginPreFrameTime = vvViewer::instance()->elapsedTime();

        // call preFrame for all plugins
        vvPluginList::instance()->preFrame();
    }
    
    MarkerTracking::instance()->update();
    
    // print frame rate
    fl_time = vv->frameRealTime();

    sum_time += fl_time - old_fl_time;
    static double maxTime = -1;
    if (maxTime < fl_time - old_fl_time)
        maxTime = fl_time - old_fl_time;

    static int frameCount = 0;
    ++frameCount;
    if (sum_time > 5.0)
    {

        if (!vvMSController::instance()->isSlave())
        {
            if (printFPS)
            {
                cout << "avg fps: " << frameCount / sum_time << ", min fps: " << 1.0 / maxTime << '\r' << flush;
            }
            //vvTui::instance()->updateFPS(1.0 / (fl_time - old_fl_time));
        }
        sum_time = 0;
        maxTime = -1;
        frameCount = 0;
    }
    old_fl_time = fl_time;
    
    vvMSController::instance()->barrierApp(frameNum++);
    
    // NO MODIFICATION OF SCENEGRAPH DATA AFTER THIS POINT

    if (vvMSController::instance()->isMaster() && vv->frameRealTime() < Input::instance()->mouse()->eventTime() + 1.5)
    {
        vv->setCursorVisible(vvConfig::instance()->mouseNav());
    }
    else
    {
        vv->setCursorVisible(false);
    }

    vvShaderList::instance()->update();
    
    //beginAppTraversal = vvViewer::instance()->elapsedTime();
    if (frameNum > 2)
        vvPluginList::instance()->postFrame();
    
    if (hud->update())
        m_renderNext = true;

    
    /*double frameTime = vvViewer::instance()->elapsedTime();
    double frameDuration = frameTime - lastFrameTime;
    lastFrameTime = frameTime;
    frameDurations.push_back(frameDuration);
    if (frameDurations.size() > 20)
        frameDurations.pop_front();*/

    //cerr << "vvVIVE::frame EMD " << frameCount << endl;
    return render;
}

void vvVIVE::doneRendering()
{
    
    vvMSController::instance()->killClients();

    if (vv->debugLevel(3))
        fprintf(stderr, "vvVIVE: done with the loop\n");
        
}

vvVIVE::~vvVIVE()
{
    if (vv->debugLevel(2))
    {
        fprintf(stderr, "\ndelete vvVIVE\n");
    }

    vvViewer::instance()->stopThreading();
    
    delete vvAnimationManager::instance();
    delete vvNavigationManager::instance();
    delete vvCommunication::instance();
    delete vvPartnerList::instance();
    delete MarkerTracking::instance();

    //vv->intersectedNode = NULL;
    
    //vvViewer::instance()->unconfig();
    //delete vvSceneGraph::instance();
    
    delete vvShaderList::instance();
    delete vvLighting::instance();
    
    vvViewer::destroy();
    //delete vvWindow::instance();
    
    delete vvPluginList::instance();

    vvShutDownHandlerList::instance()->shutAllDown();
    delete vvShutDownHandlerList::instance();

    while (numTuis() > 0)
    {
        popTui();
    }

    if (vv->debugLevel(2))
    {
        fprintf(stderr, "\nThank you for using COVER!\nBye\n");
    }
    delete Input::instance();
    
    vvPluginSupport::destroy();
    vvMSController::destroy();
    vvConfig::destroy();
    vvCommandLine::destroy();
#ifdef DOTIMING
    coTimer::quit();
#endif

}

void vvVIVE::setExitFlag(bool flag)
{
    
    if (vv)
    {
        if (vv && vv->debugLevel(3))
        {
            fprintf(stderr, "vvVIVE::setExitFlag\n");
        }
        exitFlag = flag;
    }
    
}



void
vvVIVE::requestQuit()
{
    setExitFlag(true);
     bool terminateOnCoverQuit = coCoviseConfig::isOn("VIVE.TerminateCoviseOnQuit", false);
    if (getenv("COVISE_TERMINATE_ON_QUIT"))
    {
        terminateOnCoverQuit = true;
    }
    terminateOnCoverQuit = vvMSController::instance()->syncBool(terminateOnCoverQuit);
    if (terminateOnCoverQuit)
        vvPluginList::instance()->requestQuit(true);
    m_vrbc.reset(nullptr);
    setExitFlag(true);
    // exit COVER, even if COVER has a vrb connection
    
}

vvPlugin *
vvVIVE::visPlugin() const
{

    return m_visPlugin;
}

size_t vvVIVE::numTuis() const
{
    return tabletUIs.size();
}

vvTabletUI *vvVIVE::tui(size_t idx) const
{
    assert(tabletVrTuis.size() == tabletUIs.size());
    if (idx >= tabletUIs.size())
        return nullptr;
    return tabletUIs[idx].get();
}

vvTui *vvVIVE::vrTui(size_t idx) const
{
    assert(tabletVrTuis.size() == tabletUIs.size());
    if (idx >= tabletVrTuis.size())
        return nullptr;
    return tabletVrTuis[idx].get();
}

vvTUITabFolder *vvVIVE::tuiTab(size_t idx) const
{
    auto vrtui = vrTui(idx);
    if (!vrtui)
        return nullptr;
    return vrtui->mainFolder;
}

void vvVIVE::pushTui(vvTabletUI *tui, vvTui *vrTui)
{
    assert(tabletVrTuis.size() == tabletUIs.size());
    tabletUIs.emplace_back(tui);
    tabletVrTuis.emplace_back(vrTui);
}

void vvVIVE::popTui()
{
    assert(tabletVrTuis.size() == tabletUIs.size());
    if (tabletUIs.empty())
        return;
    tabletVrTuis.pop_back();
    tabletUIs.pop_back();
}


bool vvVIVE::watchFileDescriptor(int fd)
{
    return m_watchedFds.insert(fd).second;
}

bool vvVIVE::unwatchFileDescriptor(int fd)
{
    auto it = m_watchedFds.find(fd);
    if (it == m_watchedFds.end())

    m_watchedFds.erase(it);
    return true;
}

const vrb::VRBClient *vvVIVE::vrbc() const
{
    return m_vrbc.get();
}

vrb::VRBClient *vvVIVE::vrbc()
{
    return m_vrbc.get();
}

void vvVIVE::startVrbc()
{
    if (!m_vrbc)
        restartVrbc();
}

void vvVIVE::restartVrbc()
{
    if (exitFlag) {
        m_vrbc.reset();
        return;
    }

    if (m_loadVistlePlugin) {
        class PluginMessageSender : public covise::MessageSenderInterface {

          public:
            bool sendMessage(const covise::Message *msg) const override {
                if (vvVIVE::instance()->m_visPlugin)
                    return vvPluginList::instance()->sendVisMessage(msg);
                return false;
            }

            bool sendMessage(const UdpMessage *msg) const override {
                return false;
            }
        };

        auto sender = new PluginMessageSender();
        if (vv->debugLevel(2))
            std::cerr << "starting VRB client with credentials from Vistle, session=" << m_visPlugin->collaborativeSessionId() << std::endl;
        m_vrbc.reset(new vrb::VRBClient(covise::Program::vive, sender,
            vvMSController::instance()->isSlave(),
            false));
        m_startSession = m_visPlugin->collaborativeSessionId();
    }
    else if (m_vrbCredentials)
    {
        if (vv->debugLevel(2))
            std::cerr << "starting VRB client with credentials from memory" << std::endl;
        m_vrbc.reset(new vrb::VRBClient(covise::Program::vive, *m_vrbCredentials, vvMSController::instance()->isSlave(), true));
    }
    else
    {
        if (vv->debugLevel(2))
            std::cerr << "starting VRB client with options from " << vvConfig::instance()->collaborativeOptionsFile << std::endl;
        m_vrbc.reset(new vrb::VRBClient(covise::Program::vive, vvConfig::instance()->collaborativeOptionsFile.c_str(), vvMSController::instance()->isSlave(), true));
    }
    m_vrbc->connectToServer(m_startSession);
    
}

bool vvVIVE::useVistle() const
{
    return m_loadVistlePlugin;
}

bool vvVIVE::isVRBconnected() const
{
    return m_vrbc && m_vrbc->isConnected();
}


void EventHandler::apply(vsg::KeyPressEvent& keyPress)
{

    int type = (int)KEYDOWN;
    int code = (int)keyPress.keyBase;
    int state = (int)keyPress.keyModifier;
    if (!vv->isKeyboardGrabbed())
    {
        if (keyPress.keyModifier == (vsg::MODKEY_Alt | vsg::MODKEY_Shift))
        {
            switch (keyPress.keyBase)
            {
            case 'T':
            case 't':
                cerr << "calling: vvTabletUI::instance()->close()" << endl;
                for (auto& tui : vvVIVE::instance()->tabletUIs)
                    tui->close();
                break;
            }
        }
        else if (keyPress.keyModifier == vsg::MODKEY_Alt)
        {
            switch (keyPress.keyBase)
            {
            case 'b':
                 vvConfig::instance()->m_stereoSeparation *= -1;
                 cerr << vvConfig::instance()->m_stereoSeparation << endl;
                break;

            case 'd':
                //vvOnscreenDebug::instance()->toggleVisibility();
                break;

#if 0
            case 'f':
                vv->windows[0].rs->fullScreen(!vv->windows[0].rs->isFullScreen());
                break;
#endif
            case 'n':
                if (vvMSController::instance()->getID() == 1)
                {
                    vvConfig::instance()->m_stereoSeparation *= -1;
                }
                cerr << vvConfig::instance()->m_stereoSeparation << endl;
                break;
            case 'm':
                if (vvMSController::instance()->getID() == 2)
                {
                    vvConfig::instance()->m_stereoSeparation *= -1;
                }
                cerr << vvConfig::instance()->m_stereoSeparation << endl;
                break;
            case 'z':
                vvConfig::instance()->m_worldAngle += 1;
                cerr << vvConfig::instance()->worldAngle() << endl;
                break;
            case 't':
                cerr << "calling: vvTabletUI::instance()->tryConnect()" << endl;
                for (auto& tui : vvVIVE::instance()->tabletUIs)
                    tui->tryConnect();
                break;
            case 'x':
                vvConfig::instance()->m_worldAngle -= 1;
                cerr << vvConfig::instance()->worldAngle() << endl;
                break;
            }
        }

        vv->ui->keyEvent(keyPress);
        vvNavigationManager::instance()->keyEvent(keyPress);
        vvSceneGraph::instance()->keyEvent(keyPress);
    }
    vvPluginList::instance()->key(type, code, state);
}
void EventHandler::apply(vsg::KeyReleaseEvent& keyRelease)
{
    int type = (int)KEYUP;
    int code = (int)keyRelease.keyBase;
    int state = (int)keyRelease.keyModifier;
    if (!vv->isKeyboardGrabbed())
    {
        //vv->ui->keyEvent(type, state, code);
    }
    vvPluginList::instance()->key(type, code, state);
}
void EventHandler::apply(vsg::FocusInEvent& focusIn)
{}
void EventHandler::apply(vsg::FocusOutEvent& focusOut)
{}
void EventHandler::apply(vsg::ButtonPressEvent& buttonPress)
{
    Input::instance()->mouse()->handleEvent(buttonPress);
}
void EventHandler::apply(vsg::ButtonReleaseEvent& buttonRelease)
{
    Input::instance()->mouse()->handleEvent(buttonRelease);
}
void EventHandler::apply(vsg::MoveEvent& moveEvent)
{
    Input::instance()->mouse()->handleEvent(moveEvent);
}
void EventHandler::apply(vsg::ScrollWheelEvent& scrollWheel)
{}

void EventHandler::apply(vsg::CloseWindowEvent&)
{
    _vive->setExitFlag(true);
}

void EventHandler::apply(vsg::TerminateEvent&)
{
    _vive->setExitFlag(true);
}
void vvVIVE::handleEvents(int type, int state, int code)
{
    /*
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

    }
    break;
    case (osgGA::GUIEventAdapter::KEYUP):
    {
    }
    break;
    case (osgGA::GUIEventAdapter::USER):
    {
        vvPluginList::instance()->userEvent(state);
    }
    break;

    default:
    {
    }
    break;
    }*/
}
