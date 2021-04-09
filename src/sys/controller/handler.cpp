/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <iostream>
#include <signal.h>
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <boost/program_options.hpp>

#include <comsg/CRB_EXEC.h>
#include <comsg/NEW_UI.h>
#include <comsg/coviseLaunchOptions.h>
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <util/coSignal.h>
#include <util/covise_version.h>
#include <config/coConfig.h>
#include <appl/CoviseBase.h>
#include <util/coTimer.h>

#include "controlProcess.h"
#include "exception.h"
#include "global.h"
#include "handler.h"
#include "list.h"
#include "module.h"
#include "object.h"
#include "port.h"
#include "subProcess.h"
#include "util.h"
#include "renderModule.h"
#include "crb.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>

using namespace covise;
using namespace covise::controller;
namespace po = boost::program_options;
namespace po_style = boost::program_options::command_line_style;

//  dummy message to the Mapeditor. Mapeditor returns this message to
//  the controller. Result: wait_for_msg loop is left and signals can
//  be interpreted.
Message m_dummyMessage;

//  m_quitNow == 1 if a SIGTERM signal was sent to the controller
int m_quitNow = 0;

std::string autosaveFile()
{
    QString file = QDir::homePath();
#ifdef _WIN32
    file += "/COVISE";
#else
    file += "/.covise";
#endif
#if QT_VERSION < 0x040400
    pid_t pid = getpid();
#else
    qint64 pid = QCoreApplication::applicationPid();
#endif
    file += "/autosave-" + QString(Host().getAddress()) + "-" + QString::number(pid) + ".net";
    return file.toStdString();
}

class sigQuitHandler : public coSignalHandler
{
public:
    sigQuitHandler(const HostManager &manager)
        : manager(manager)
    {
    }

private:
    virtual void sigHandler(int sigNo) //  catch SIGTERM
    {
        if (sigNo == SIGTERM)
        {
            m_quitNow = 1;
            try
            {
                manager.getMasterUi().send(&m_dummyMessage);
            }
            catch (const Exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        return;
    }

    virtual const char *sigHandlerName() { return "sigQuitHandler"; }
    const HostManager &manager;
};

void preventBrokenPipe()
{
//  prevent "broken pipe" forever
#ifdef _WIN32

    int err;
    unsigned short wVersionRequested;
    struct WSAData wsaData;
    wVersionRequested = MAKEWORD(1, 1);
    err = WSAStartup(wVersionRequested, &wsaData);

#else

    coSignalHandler emptyHandler;
    coSignal::addSignal(SIGPIPE, emptyHandler);

#endif
}

void initDummyMessage()
{
    m_dummyMessage.type = COVISE_MESSAGE_LAST_DUMMY_MESSAGE; //  initialize dummy message
    m_dummyMessage.data = DataHandle(2);
    memcpy(m_dummyMessage.data.accessData(), " ", 2);
}

void printWelcomeMessage()
{
    //  Say hello
    cerr << endl;
    cerr << "*******************************************************************************" << endl;
    string text = CoviseVersion::shortVersion();
    string text2 = "* COVISE " + text + " starting up, please be patient....                    *";
    cerr << text2 << endl;
    cerr << "*                                                                             *" << endl;
}

/*!
    \class CTRLHandler
    \brief Covise controller main handling   
*/

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

CTRLHandler *CTRLHandler::singleton = nullptr;

CTRLHandler::CTRLHandler(int argc, char *argv[])
    : m_autosavefile(autosaveFile())
{
    std::cerr << "starting covise" << std::endl;
    singleton = this;

    preventBrokenPipe();
    initDummyMessage();

    // signal(SIGTERM, sigHandlerQuit );
    sigQuitHandler quitHandler(m_hostManager);
    coSignal::addSignal(SIGTERM, quitHandler);

    parseCommandLine(argc, argv);

    lookupSiblings();
    printWelcomeMessage();
    // read covise.config
    m_hostManager.getLocalHost().startCrb();
    m_hostManager.getLocalHost().startUI(m_options.uiOptions);

    loadNetworkFile();
    cerr << "* ...done initialization                                                      *" << endl;
    cerr << "*******************************************************************************" << endl;
    loop();
}

CTRLHandler *CTRLHandler::instance()
{
    assert(singleton);
    return singleton;
}

NumRunning &CTRLHandler::numRunning()
{
    return m_numRunning;
}

const UIOptions &CTRLHandler::uiOptions()
{
    return m_options.uiOptions;
}

void CTRLHandler::lookupSiblings()
{
    coConfigEntryStringList list = coConfig::getInstance()->getScopeList("System.Siblings");

    std::list<coConfigEntryString>::iterator listentry = list.begin();
    while (listentry != list.end() && (!list.empty()))
    {
        cerr << "Sibling: " << (*listentry).toStdString() << endl;
        QString value = coConfig::getInstance()->getString("mod1", QString("System.Siblings.") + (*listentry), "");
        QString value2 = coConfig::getInstance()->getString("mod2", QString("System.Siblings.") + (*listentry), "");
        cerr << "Sibling: Entry = " << value.toStdString() << endl;

        siblings.push_back(std::pair<std::string, std::string>(value.toStdString(), value2.toStdString()));
        //mHostList.push_back(value.toStdString());
        listentry++;
    }
}

void CTRLHandler::loop()
{
    while (!m_exit)
    {
        std::unique_ptr<Message> msg;
        if (!m_quitNow)
        {
            msg.reset(CTRLGlobal::getInstance()->controller->wait_for_msg());
            handleMsg(msg);
        }
        else
        {
            return;
        }

    } //  while
}

//!
//! Handle messages from the different covise parts
//!
void CTRLHandler::handleMsg(const std::unique_ptr<Message> &msg)
{

    string copyMessageData;

    //  copy message to a secure place
    if (msg->data.length() > 0)
        copyMessageData = msg->data.data();

    //  Switch across message types

    if (m_quitNow)
        msg->type = COVISE_MESSAGE_QUIT;
    CTRLGlobal *global = CTRLGlobal::getInstance();
    switch (msg->type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
    {
        handleClosedMsg(msg);
        break;
    }

    case COVISE_MESSAGE_QUIT:
        handleQuit(msg);
        break;

    //  FINALL: Module says it has finished
    case COVISE_MESSAGE_FINALL:
        handleFinall(msg, copyMessageData);
        break;

    //  FINISHED : Finish from a Rendermodule
    case COVISE_MESSAGE_FINISHED:
    {
        bool update = false;
        for (auto renderer : m_hostManager.getAllModules<Renderer>())
        {
            auto display = renderer->getDisplay(msg->sender);
            if (display != renderer->end())
            {
                update = renderer->update(display, m_numRunning);
                break;
            }
        }
        if (update && m_numRunning.apps == 0)
        {
            //  send Finished Message to the MapEditor
            //  if no modules are running
            m_hostManager.slaveUpdate();
            if (m_options.quit)
            {
                m_quitNow = 1;
                msg->data.setLength(0);
                copyMessageData.clear();
            }

            Message mapmsg{COVISE_MESSAGE_UI, "FINISHED\n"};
            m_hostManager.sendAll<Userinterface>(mapmsg);
            m_hostManager.sendAll<Renderer>(mapmsg);
        }

        break;
    }

    case COVISE_MESSAGE_PLOT:
    case COVISE_MESSAGE_RENDER_MODULE:
    case COVISE_MESSAGE_RENDER:
    {
        //  forward the Message to the other Renderer
        //
        //   RENDER-Messages should only be exchanged between
        //   Renderer-pairs
        //
        //   Necessary informations: process_id
        //
        for (const Renderer *renderer : m_hostManager.getAllModules<Renderer>())
        {
            auto display = renderer->getDisplay(msg->sender);
            if (display != renderer->end() || renderer->isMirrorOf(msg->sender))
            {
                renderer->send(msg.get());
            }
            else if ((strncmp(msg->data.data(), "VRML", 4) == 0 && renderer->info().name == "VRMLRenderer") || strncmp(msg->data.data(), "GRMSG", 5) == 0)
            {
                renderer->send(msg.get());
            }
        }
        break;
    }

    case COVISE_MESSAGE_PARINFO:
    {
        // send message to all userinterfaces

        Message new_msg{COVISE_MESSAGE_PARINFO, copyMessageData};
        m_hostManager.sendAll<Userinterface>(new_msg);

        // handle message, change parameter
        int iel = 0;
        vector<string> list = splitStringAndRemoveComments(copyMessageData, "\n");
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        const string &portname = list[iel++];
        const string &porttype = list[iel++];
        const string &value = list[iel++];
        try
        {
            m_hostManager.findHost(host).getModule(name, std::stoi(nr)).connectivity().getParam(portname).set_value_list(value);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }

        break;
    }

    //  WARNING : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_WARNING:
    {
        m_hostManager.sendAll<Userinterface>(*msg);
        sendGenericInfoToRenderer("WARNING", *msg);
        break;
    }

    //  INFO  : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_INFO:
    {
        m_hostManager.sendAll<Userinterface>(*msg);
        sendGenericInfoToRenderer("INFO", *msg);
        break;
    }

    //  UPDATE_LOADED_MAPNAME  : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_UPDATE_LOADED_MAPNAME:
    {
        MARK0("UPDATE_LOADED_MAPNAME")
        m_hostManager.sendAll<Userinterface>(*msg);
        break;
    }

    //  REQ_UI : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_REQ_UI:
    {
        m_hostManager.sendAll<Userinterface>(*msg);
        break;
    }

    case COVISE_MESSAGE_COVISE_ERROR:
    {
        int iel = 0;
        vector<string> list = splitStringAndRemoveComments(copyMessageData, "\n");
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        try
        {
            auto &app = m_hostManager.findHost(host).getModule(name, std::stoi(nr));
            int delta = app.overflowOfNextError();
            if (delta)
            {
                if (delta == 1)
                {
                    for (const auto &err : app.errorsSentByModule())
                    {
                        Message warning{COVISE_MESSAGE_WARNING, err};
                        m_hostManager.sendAll<Userinterface>(warning);
                    }

                    string buffer = "Overflow of error messages from " + name + "_" + nr + " module on host " + host + " (last errors in \"Info Messages\")!";
                    Message error(COVISE_MESSAGE_COVISE_ERROR, buffer);
                    m_hostManager.sendAll<Userinterface>(error);
                }
            }
            else
            {
                app.errorsSentByModule().emplace_back("Error log: " + string(msg->data.data()));
                m_hostManager.sendAll<Userinterface>(*msg);
                sendGenericInfoToRenderer("ERROR", *msg);
            }
            //  change Modulestatus to STOP
            app.setStatus(NetModule::Status::stopping);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            cerr << "COVISE_ERROR: did not find module  " << name << "_" << nr << " on " << host << endl;
        }
        break;
    }

    case COVISE_MESSAGE_COVISE_STOP_PIPELINE:
    {

        int iel = 0;
        vector<string> list = splitStringAndRemoveComments(copyMessageData, "\n");
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];

        try
        {
            auto &app = m_hostManager.findHost(host).getModule(name, std::stoi(nr));
            app.setStatus(NetModule::Status::stopping);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
            cerr << "STOP_PIPELINE: did not find module  " << name << "_" << nr << " on " << host << endl;
        }
        break;
    }

    case COVISE_MESSAGE_GENERIC:
    {
        /* nach Keywords und Module auswerten */
        /* 18.6.97
         FORMAT: keyword ACTION DATA
         ---------------------------
         keyword INIT  name,nr,host executable
         create_mod wird aufgerufen und die UIF-Teile gestartet
         keyword APPINFO name,nr,host DATA
         Message wird an APP-Teil weitergeschickt
         keyword UIFINFO name,nr,host DATA
         Message wird an UIF-Teile weitergeschickt
         Hier evtl. erkennnen, von wo die Msg. kam und nur an die anderen
         UIF-Teile schicken.

         */
        int iel = 0;
        vector<string> list = splitStringAndRemoveComments(copyMessageData, "\n");
        string key = list[iel++];
        string action = list[iel++];

        modui *tmpmod = global->modUIList->get(key);

        if (tmpmod == NULL)
        {
            /* Start der UIF-Teile */
            if (action == "INIT")
            {
                print_comment(__LINE__, __FILE__, " GENERIC: Wrong INIT-Message! ");
                print_exit(__LINE__, __FILE__, 1);
            }

            string name = list[iel++];
            string instanz = list[iel++];
            string host = list[iel++];
            string executable = list[iel++];
            string category = list[iel++];
            try
            {
                global->modUIList->create_mod(m_hostManager.findHost(host).getModule(name, std::stoi(instanz)), key, executable);
            }
            catch (const Exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }

        else if (action == "APPINFO")
        {
            /* send Message to the APP-Part */
            tmpmod->sendapp(&*msg);
        }

        else if (action == "UIFINFO")
        {
            /* send Message to the UIF-Parts */
            tmpmod->send_msg(&*msg);
        }
        break;
    }

    //  FEEDBACK : Messages from Renderer sent to a module
    case COVISE_MESSAGE_FEEDBACK:
    {

        int iel = 0;
        vector<string> list = splitStringAndRemoveComments(copyMessageData, "\n");
        const string &name = list[iel++];
        const string &instanz = list[iel++];
        const string &host = list[iel++];

        try
        {
            auto app = m_hostManager.findHost(host).getModule(name, std::stoi(instanz)).send(&*msg);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        break;
    }

    // UI messages
    case COVISE_MESSAGE_UI:
    {
        // handle UNDO message
        // get all operations for last action
        // generate a new message and buffer content
        QString action(copyMessageData.c_str());
        if (action.startsWith("UNDO"))
        {
            if (!m_undoBuffer.isEmpty())
            {
                m_writeUndoBuffer = false;
                QString text = m_undoBuffer.last();
                QByteArray line = text.toLatin1();
                copyMessageData = line.data();
                m_undoBuffer.removeLast();
                Message undo_msg{COVISE_MESSAGE_UI, copyMessageData};
                handleUI(&undo_msg, copyMessageData);
                m_writeUndoBuffer = true;

                if (m_undoBuffer.isEmpty())
                {
                    undo_msg = Message{COVISE_MESSAGE_UI, "UNDO_BUFFER_FALSE"};
                    m_hostManager.sendAll<Userinterface>(undo_msg);
                }
            }
        }

        else
            handleUI(&*msg, copyMessageData);

        break;
    }
    case COVISE_MESSAGE_NEW_UI:
    {
        NEW_UI uimsg{*msg};
        handleNewUi(uimsg);
    }
    break;
    default:
        break;
    } //  end message switch

    if (m_quitNow == 0)
    {
        copyMessageData.clear();
    }
}

//!
//! handle the message COVISE_MESSAGE_EMPTY, COVISE_MESSAGE_CLOSE_SOCKET, COVISE_MESSAGE_SOCKET_CLOSED:
//!
void CTRLHandler::handleClosedMsg(const std::unique_ptr<Message> &msg)
{

    if (msg->conn == NULL)
        return;
    std::string msg_txt;
    sender_type peer_type = (sender_type)msg->conn->get_peer_type();
    int peer_id = msg->conn->get_peer_id();
    CTRLGlobal *global = CTRLGlobal::getInstance();
    //  look which socket is broken
    switch (peer_type)
    {
    case RENDERER:
    case APPLICATIONMODULE:
    {
        auto p_mod = m_hostManager.findModule(peer_id);
        if (!p_mod)
        {
            break;
        }

        auto p_app = p_mod->as<NetModule>();
        if (!p_app)
        {
            std::cerr << "crb or userinterface crashed " << std::endl;
            break;
        }

        bool del_mod = false;
        int instance = 0;
        if (auto p_rend = dynamic_cast<Renderer *>(p_app))
        {
            auto disp = p_rend->getDisplay(peer_id);
            std::stringstream ss;
            ss << "The " << disp->get()->host.userInfo().userName << "@" << disp->get()->host.userInfo().ipAdress << "'s display of the "
               << p_rend->info().name << p_rend->instance() << " crashed !!!";
            msg_txt = ss.str();
            instance = p_rend->instance();
            p_rend->removeDisplay(disp);
            if (p_rend->numDisplays() == 0)
                del_mod = true;
        }
        else
        {
            del_mod = true;
            std::stringstream ss;
            ss << "Module " << p_app->fullName() << "@" << p_app->host.userInfo().ipAdress << " crashed !!!";
            msg_txt = ss.str();
            instance = p_app->instance();
        }
        m_hostManager.sendAll<Userinterface>(Message{COVISE_MESSAGE_COVISE_ERROR, msg_txt});

        // module have to be deleted
        if (del_mod)
        {
            std::stringstream ss;
            ss << "DIED\n"
               << p_app->info().name << "\n"
               << instance << "\n"
               << p_mod->host.userInfo().ipAdress;
            Message msg{COVISE_MESSAGE_UI, ss.str()};
            m_hostManager.sendAll<Userinterface>(msg);

            finishExecuteIfLastRunning(*p_app);
            p_app->setAlive(false);
        }
        break;
    }

    case USERINTERFACE:
    {
        auto mod = m_hostManager.findModule(peer_id);
        if (auto ui = mod->as<Userinterface>())
        {
            cerr << "Map editor crashed" << ui->host.userInfo().userName << "@" << ui->host.userInfo().ipAdress << endl;
            cerr << "Trying to restart session " << endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            ui->restart(m_options.uiOptions);
        }
        break;
    }

    default:
    {
        break;
    }
    }
}

//!
//! parse the commandline & init global states
//!

void CTRLHandler::parseCommandLine(int argc, char **argv)
{
    po::options_description desc("usage");
    desc.add_options()("help,h", "show this message")("iconify,i", "iconify map editor")("maximize,m", "maximize map editor")("quit,q", "quit after execution")("execute,e", "execute on loading")("user,u", po::value<std::string>(&m_options.localUserName), "scan local user")("timer,t", "activate timer")("X11,x", "start X11 user interface")("script", po::value<std::string>(&m_options.uiOptions.pyFile), "start script mode, executing Python script script.py")("CrbScript,s", po::value<std::string>(&m_options.crbLaunchScript), "Starts script with the manual message \"crb ...\" as parameter for every connection")("short,v", "print short version")("version", "print long version")("nogui", "don't start the userinterface when loading and executing a network")("gui", "start userinterface, even in scripting mode (specify as last argument)")("minigui", "start a minimal userinterface when loading and executing a network");

    po::options_description hidden("");
    hidden.add_options()("mapFile", "covise map(.net) or python(.py) file to load");
    po::positional_options_description p;
    p.add("mapFile", 1);
    po::options_description all("");
    all.add(desc).add(hidden);
    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(all).positional(p).style(po_style::unix_style).run(), vm);
        po::notify(vm);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return;
    }

    if (vm.count("help"))
    {
        std::cerr << desc << std::endl;
        exit(0);
    }
    if (vm.count("short"))
    {
        printf("%s\n", CoviseVersion::shortVersion());
        exit(0);
    }
    if (vm.count("version"))
    {
        printf("%s\n", CoviseVersion::longVersion());
        exit(0);
    }
    if (vm.count("timer"))
    {
        coTimer::init("timing", 2000);
    }
    if (vm.count("quit"))
    {
        m_options.quit = true;
    }
    if (vm.count("execute"))
    {
        m_options.executeOnLoad = true;
    }
    if (vm.count("iconify"))
    {
        m_options.uiOptions.iconify = true;
    }
    if (vm.count("maximize"))
    {
        m_options.uiOptions.maximize = true;
    }
    if (vm.count("script") || vm.count("X11"))
    {
        m_options.uiOptions.type = UIOptions::python;
    }
    if (vm.count("nogui"))
    {
        m_options.uiOptions.type = UIOptions::nogui;
        m_options.executeOnLoad = true;
    }
    if (vm.count("gui"))
    {
        m_options.uiOptions.type = UIOptions::gui;
    }
    if (vm.count("minigui"))
    {
        m_options.uiOptions.type = UIOptions::miniGui;
        m_options.executeOnLoad = true;
    }
    if (vm.count("mapFile"))
    {
        auto file = vm["mapFile"].as<std::string>();
        std::cerr << "mapfile: " << file << std::endl;
        if (file.substr(file.find_last_of('.')) == ".net")
        {
            m_options.netFile = file;
        }
        else if (file.substr(file.find_last_of('.')) == ".py")
        {
            m_options.uiOptions.type = UIOptions::python;
            m_options.uiOptions.pyFile = file;
        }
        else
        {
            std::cerr << desc << std::endl;
            std::cerr << "the provided filename has to a COVISE map file (.net) or a python script (.py)" << std::endl;
            exit(-1);
        }
        if (!m_options.isLoaded && m_options.uiOptions.type == UIOptions::miniGui)
        {
            cerr << endl;
            cerr << "***********************  E R R O R   **************************" << endl;
            cerr << "Starting COVISE with a minimal user interface is only possible " << endl;
            cerr << "with a network file (.net) given" << endl;
            cerr << "***************************************************************" << endl;
            exit(5);
        }

        if (!m_options.isLoaded && m_options.uiOptions.type != UIOptions::miniGui)
        {
            cerr << endl;
            cerr << "***********************  E R R O R   **************************" << endl;
            cerr << "Starting COVISE without a user interface is only possible" << endl;
            cerr << "with a network file (.net) given" << endl;
            cerr << "***************************************************************" << endl;
            exit(6);
        }
    }
}

//!
//! if a network file was given in the commandline, try to load the map
//!
void CTRLHandler::loadNetworkFile()
{

    if (m_options.netFile.empty())
        return;
    //  look, if a path for searching is given, otherwise create a directory net
    char *returnPath = NULL;
    FILE *fp = CoviseBase::fopen(m_options.netFile.c_str(), "r", &returnPath);
    if (fp)
        m_globalFilename = m_options.netFile;

    else
    {
        string pathname = getenv("COVISEDIR");
        if (!pathname.empty())
            m_globalFilename = pathname + "/net/" + m_options.netFile;
    }

    m_globalLoadReady = loadNetworkFile(m_globalFilename);
    if (m_globalLoadReady && m_options.executeOnLoad)
    {
        for (NetModule *app : m_hostManager.getAllModules<NetModule>())
        {
            if (!dynamic_cast<const Renderer *>(app) && app->isOnTop())
            {
                app->exec(m_numRunning);
            }
        }
    }
}

//!
//! handle QUIT messages from user interface, opencover or other sources
//!
void CTRLHandler::handleQuit(const std::unique_ptr<Message> &msg)
{

    //  Configurable reaction: Quit complete session on COVER quit?
    bool terminateForCover = false;

    if (msg->send_type == RENDERER)
    {
        bool terminateOnCoverQuit = coCoviseConfig::isOn("COVER.TerminateCoviseOnQuit", false);
        if (getenv("COVISE_TERMINATE_ON_QUIT"))
        {
            terminateOnCoverQuit = true;
        }
        if (terminateOnCoverQuit)
        {
            auto mod = m_hostManager.findModule(msg->conn->get_peer_id());
            if (mod)
            {
                if (auto renderer = dynamic_cast<const Renderer *>(mod))
                {
                    const string &name = renderer->info().name;

                    if (name == "VRRenderer" || name == "OpenCOVER" || name == "COVER" || name == "COVER_VRML")
                        terminateForCover = true;
                }
            }
        }
    }

    const auto &uis = m_hostManager.getAllModules<Userinterface>();
    auto ui = std::find_if(uis.begin(), uis.end(), [&msg](const Userinterface *ui) {
        return ui->processId == msg->sender;
    });

    if (ui != uis.end() || (m_quitNow == 1) || terminateForCover)
    {
        Message msg{COVISE_MESSAGE_NEW_DESK, ""};
        m_hostManager.sendAll<CRBModule>(msg);
        for (NetModule *app : m_hostManager.getAllModules<NetModule>())
        {
            app->setAlive(false);
            CTRLGlobal::getInstance()->modUIList->delete_mod(app->info().name, std::to_string(app->instance()), app->host.userInfo().ipAdress);
        }
        //  must be done for deleting shared memory files all other processes must be closed before
        std::this_thread::sleep_for(std::chrono::seconds(1));
        coTimer::quit();

        if (ui != uis.end() && !m_quitNow && !m_autosavefile.empty())
        {
            QFile autosave(QString(m_autosavefile.c_str()));
            if (autosave.exists())
            {
                if (!autosave.remove())
                {
                    std::cerr << "failed to remove " << m_autosavefile << std::endl;
                }
            }
        }

        m_exit = true;
    }
}

//!
//! handle message after a module has been executed (finished)
//!
void CTRLHandler::handleFinall(const std::unique_ptr<Message> &msg, string copyMessageData)
{

    int iel = 0;
    vector<string> list = splitStringAndRemoveComments(copyMessageData, "\n");
    const string &name = list[iel++];
    const string &instance = list[iel++];
    const string &hostAddress = list[iel++];
    try
    {
        auto &app = m_hostManager.findHost(hostAddress).getModule(name, std::stoi(instance));

        app.errorsSentByModule().clear();

        int noOfParameter = std::stoi(list[iel++]);
        int noOfSaveDataObjects = std::stoi(list[iel++]);
        int noOfReleaseDataObjects = std::stoi(list[iel++]);
        //  read and change Parameter
        for (int i = 1; i <= noOfParameter; i++)
        {
            const string &parameterName = list[iel++];
            iel++; // unused parameter type
            int noOfValues = std::stoi(list[iel++]);

            for (int iv = 1; iv <= noOfValues; iv++)
            {
                // workaround for choice param if only the current index is sent & not the parameter text
                if (list.size() == iel)
                {
                    break;
                }
                const string &parameterValue = list[iel++];
                app.connectivity().getParam(parameterName).set_value(iv, parameterValue);
            }
        }

        //  get the Dataobjects for SAVE
        for (int i = 1; i <= noOfSaveDataObjects; i++)
        {
            const string &dataObjectNames = list[iel++];
            app.set_DO_status(DO_SAVE, dataObjectNames);
        }

        //  get the Dataobjects for RELEASE
        for (int i = 1; i <= noOfReleaseDataObjects; i++)
        {
            const string &dataObjectNames = list[iel++];
            app.set_DO_status(DO_RELEASE, dataObjectNames);
        }

        //  send Message with Output-Parameters to all UIF
        string content = app.get_outparaobj();
        if (!content.empty())
        {
            Message msg{COVISE_MESSAGE_FINISHED, content};
            m_hostManager.sendAll<Userinterface>(msg);
        }

        //  check if one level up is a module that has to be run
        auto stat = app.status();
        app.setExecuting(false);
        if (!app.startModuleWaitingAbove(m_numRunning))
        {
            //  check if the module which has just finished
            //  has not been started again
            if (stat != NetModule::Status::stopping)
            {
                if (app.numRunning() == 0)
                {
                    //  change Modulestatus to Finish
                    app.setExecuting(false);

                    //  start Following Modules
                    app.setStart();
                    app.startModulesUnder(m_numRunning);
                    app.setExecuting(false);
                }

                else
                {
                    app.exec(m_numRunning);
                }
            }
        }

        // send Finished Message to the MapEditor if no modules are running
        m_numRunning.apps--;
        if (m_numRunning.apps == 0)
        {
            if (m_options.quit)
            {
                m_quitNow = 1;
                msg->data.setLength(0);
                copyMessageData.clear();
            }

            Message mapmsg{COVISE_MESSAGE_UI, "FINISHED\n"};
            m_hostManager.sendAll<Userinterface>(mapmsg);
            m_hostManager.sendAll<Renderer>(mapmsg);
        }
    }
    catch (const Exception &e)
    {
        std::cerr << e.what() << '\n';
        cerr << "Module was already deleted but has sent Finished" << endl;
    }
}

//!
//!
//!
void CTRLHandler::addBuffer(const QString &text)
{
    m_undoBuffer.append(text);
    if (m_undoBuffer.size() == 1)
    {
        Message undo_msg{COVISE_MESSAGE_UI, "UNDO_BUFFER_TRUE"};
        m_hostManager.sendAll<Userinterface>(undo_msg);
    }
}

//!
//!  handle all message received from the user interface
//!
void CTRLHandler::handleUI(Message *msg, string copyData)
{

    int iel = 0;
    vector<string> list = splitStringAndRemoveComments(copyData, "\n");

    //  get Message-Keyword
    const string &key = list[iel++];

    if (key == "EXEC")
    {
        //  EXECUTE ON CHANGE
        if (list.size() == 4)
        {
            try
            {
                m_hostManager.findHost(list[iel + 2]).getModule(list[iel], std::stoi(list[iel + 1])).exec(m_numRunning);
            }
            catch (const Exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        else
        {
            for (NetModule *app : m_hostManager.getAllModules<NetModule>())
            {
                if (app->isOnTop() && app->isOriginal() && !dynamic_cast<Renderer *>(app))
                {
                    app->exec(m_numRunning);
                }
            }
        }
    }

    else if (key == "STATUS")
    {
        Userinterface::Status status = list[iel++] == "MASTER" ? Userinterface::Status::Master : Userinterface::Status::Slave;
        auto &master = m_hostManager.getMasterUi();
        master.setStatus(status);
        dynamic_cast<Userinterface &>(*m_hostManager.findModule(m_hostManager.uiState.masterRequestSenderId)).changeStatus(status == Userinterface::Status::Master ? Userinterface::Status::Slave : Userinterface::Status::Master);
        for (Renderer *renderer : m_hostManager.getAllModules<Renderer>())
        {
            renderer->setSenderStatus();
        }
        CTRLGlobal::getInstance()->modUIList->set_new_status();
        sendCollaborativeState();
    }

    else if (key == "INIT" || key == "INIT_DEBUG" || key == "INIT_MEMCHECK" || key == "COPY")
    {
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];

        int posx = std::stoi(list[iel++]);
        int posy = std::stoi(list[iel++]);

        //  entry in net-module-list
        ExecFlag flags = ExecFlag::Normal;
        if (key == "INIT_DEBUG")
            flags = ExecFlag::Debug;
        else if (key == "INIT_MEMCHECK")
            flags = ExecFlag::Memcheck;
        int action = key == "COPY" ? 1 : 0;
        initModuleNode(name, nr, host, posx, posy, "", action, flags);
    }
    else if (key == "GETDESC")
    {
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];

        try
        {
            auto &app = m_hostManager.findHost(host).getModule(name, std::stoi(nr));

            ostringstream buffer;
            buffer << "PARAMDESC\n"
                   << name << "\n"
                   << nr << "\n"
                   << host << "\n";
            buffer << app.connectivity().inputParams.size() << "\n";
            // loop over all input parameters
            for (int i = 0; i < app.connectivity().inputParams.size(); i++)
            {
                buffer << app.connectivity().inputParams[i].get_name() << "\n";
                buffer << app.connectivity().inputParams[i].get_val_list() << "\n";
            }
            Message msg{COVISE_MESSAGE_PARAMDESC, buffer.str()};
            m_hostManager.sendAll<Userinterface>(msg);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << "CTRLHandler.cpp: GETDESC: did not find module: name=" << name << ", nr=" << nr << ", host=" << host << std::endl;
        }
    }

    else if (key == "MIRROR_ALL")
    {
        auto apps = m_hostManager.getAllModules<NetModule>();
        for (auto &host : m_hostManager)
        {
            for (NetModule *app : apps)
            {
                int instance = app->instance() + 1000 * (app->numMirrors() + 1);
                NetModule::MapPosition pos{(app->pos().x + (app->numMirrors() + 1) * 400), app->pos().y};
                try
                {
                    auto &copy = host.second->startApplicationModule(app->info().name, std::to_string(instance), pos.x + (app->numMirrors() + 1) * 400, pos.y, 4, ExecFlag::Normal, app);

                    ostringstream os;
                    os << "COPY2\n"
                       << copy.createBasicModuleDescription()
                       << pos.x << "\n"
                       << pos.y << "\n"
                       << app->createBasicModuleDescription();
                    Message msg{COVISE_MESSAGE_UI, os.str()};
                    m_hostManager.sendAll<Userinterface>(msg);
                    msg = Message{COVISE_MESSAGE_UI, "DESC\n" + copy.createDescription()};
                    m_hostManager.sendAll<Userinterface>(msg);
                }
                catch (const Exception &e)
                {
                    std::cerr << e.what() << std::endl;
                    ostringstream os;
                    os << "Failing to start " << app->fullName() << "@" << host.first << "!!!";
                    Message err{COVISE_MESSAGE_COVISE_ERROR, os.str()};
                    m_hostManager.sendAll<Userinterface>(err);
                }
            }
        }
    }

    else if (key == "REPLACE")
    {
        // read parameter
        string newmod = list[iel++];
        string newinst = list[iel++];
        string newhost = list[iel++];

        int posx = std::stoi(list[iel++]);
        int posy = std::stoi(list[iel++]);

        string oldmod = list[iel++];
        string oldinst = list[iel++];
        string oldhost = list[iel++];

        //  store the current parameters of the module to be replaced
        int numOldInputParams = 0;
        vector<NetModule *> appList;
        vector<string> from_param;

        try
        {
            auto &app = m_hostManager.findHost(oldhost).getModule(oldmod, std::stoi(oldinst));
            appList.push_back(&app);
            std::string buffer = app.get_parameter(controller::Direction::Input, false);
            if (!buffer.empty())
            {
                from_param = splitStringAndRemoveComments(buffer, "\n");
                numOldInputParams = app.connectivity().inputParams.size();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        // get all connections
        getAllConnections();

        //  1.delete old module
        m_writeUndoBuffer = false;
        delModuleNode(appList);
        m_writeUndoBuffer = true;

        //  2. init new module
        // return new unique instance number from controller
        string title;
        const auto newApp = initModuleNode(newmod, newinst, newhost, posx, posy, title, 0, ExecFlag::Normal);
        newinst = std::to_string(newApp->instance());

        // 3. look if parameters can be reused
        int npnew = 0;
        vector<string> to_param;
        if (newApp)
        {
            string buffer = newApp->get_parameter(controller::Direction::Input, false);
            if (!buffer.empty())
            {
                to_param = splitStringAndRemoveComments(buffer, "\n");
                istringstream npl2(to_param[0]);
                npl2 >> npnew;

                if (npnew != 0 && numOldInputParams != 0)
                {
                    for (int j = 1; j < npnew * 6 + 1; j = j + 6)
                    {
                        string paramname = to_param[j]; //  name
                        string type = to_param[j + 1];  //  type

                        for (int i = 1; i < numOldInputParams * 6 + 1; i = i + 6)
                        {
                            if (paramname == from_param[i] && type == from_param[i + 1])
                            {
                                string value = from_param[i + 3];   //  value
                                string imm = from_param[i + 4];     //  IMM
                                string apptype = from_param[i + 5]; //  appearance type
                                sendNewParam(newmod, newinst, newhost, paramname, type, value, apptype, oldhost);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // 4. get interface names of new module
        vector<string> inpInterfaces;
        vector<string> outInterfaces;
        for (const auto &interface : newApp->connectivity().interfaces)
        {
            if (interface->get_direction() == controller::Direction::Output)
                outInterfaces.push_back(interface->get_name());
            else
                inpInterfaces.push_back(interface->get_name());
        }
        //  5. restore connections with new module
        //  loop over all entries in the temporary connections list
        //  replace connection line if port name is the same

        for (int kk = 0; kk < m_nconn; kk++)
        {
            string fname = from_name[kk];
            string fnr = from_inst[kk];
            string fhost = from_host[kk];
            string fport = from_port[kk];

            string tname = to_name[kk];
            string tnr = to_inst[kk];
            string thost = to_host[kk];
            string tport = to_port[kk];

            //  look if port name  is in the list
            //  substitute module if found
            bool change = false;
            if (from_name[kk] == oldmod && from_inst[kk] == oldinst && from_host[kk] == oldhost)
            {
                fname = newmod;
                fnr = newinst;
                fhost = newhost;
                for (int i = 0; i < outInterfaces.size(); i++)
                {
                    if (outInterfaces[i] == fport)
                    {
                        change = true;
                        break;
                    }
                }
            }

            //  look if string is in the list of moved/copied modules
            //  substitute module if found
            if (to_name[kk] == oldmod && to_inst[kk] == oldinst && to_host[kk] == oldhost)
            {
                tname = newmod;
                tnr = newinst;
                thost = newhost;
                for (int i = 0; i < inpInterfaces.size(); i++)
                {
                    if (inpInterfaces[i] == tport)
                    {
                        change = true;
                        break;
                    }
                }
            }

            if (change)
                makeConnection(fname, fnr, fhost, fport, tname, tnr, thost, tport);
        }
    }

    else if (key == "SETCLIPBOARD")
    {

        //  no of modules
        int no = std::stoi(list[iel++]);
        //  allocate memory for module
        vector<NetModule *> moduleList;
        for (int i = 0; i < no; i++)
        {
            const string &name = list[iel++];
            const string &nr = list[iel++];
            const string &host = list[iel++];
            try
            {
                moduleList.push_back(&m_hostManager.findHost(host).getModule(name, std::stoi(nr)));
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                cerr << endl
                     << "---Controller : module : " << name << "_" << nr << "@" << host << " not found !!!\n";
            }
        }

        string buffer = writeClipboard("SETCLIPBOARD", moduleList);

        Message tmpmsg{COVISE_MESSAGE_UI, buffer};
        m_hostManager.sendAll<Userinterface>(tmpmsg);
    }

    else if (key == "GETCLIPBOARD")
    {

        m_clipboardBuffer = copyData;
        int len = (int)key.length() + 1;
        m_clipboardBuffer.erase(0, len);
        m_clipboardReady = recreate(m_clipboardBuffer, CLIPBOARD);
    }

    else if (key == "GETCLIPBOARD_UNDO")
    {

        m_clipboardBuffer = copyData;
        int len = (int)key.length() + 1;
        m_clipboardBuffer.erase(0, len);
        m_clipboardReady = recreate(m_clipboardBuffer, UNDO);
    }

    else if (key == "MOVE2" || key == "COPY2" || key == "MOVE2_DEBUG" || key == "MOVE2_MEMCHECK")
    {
        ExecFlag flags = ExecFlag::Normal;
        if (key == "MOVE2_DEBUG")
        {
            flags = ExecFlag::Debug;
        }
        else if (key == "MOVE2_MEMCHECK")
        {
            flags = ExecFlag::Memcheck;
        }

        //  no of moved/copied modules
        int no = std::stoi(list[iel++]);

        //  MOVE = 2, COPY = 3
        int action = std::stoi(list[iel++]);

        //  allocate memory
        vector<string> oldmod(no), oldinst(no), oldhost(no), oldparam(no), oldtitle(no);
        vector<string> newmod(no), newinst(no), newhost(no), newxpos(no), newypos(no);

        //  get old modules && store some stuff
        vector<NetModule *> moduleList;
        for (int ll = 0; ll < no; ll++)
        {
            newmod[ll] = list[iel++];
            newinst[ll] = list[iel++];
            newhost[ll] = list[iel++];
            newxpos[ll] = list[iel++];
            newypos[ll] = list[iel++];

            oldmod[ll] = list[iel++];
            oldinst[ll] = list[iel++];
            oldhost[ll] = list[iel++];
            try
            {
                NetModule &oldApp = m_hostManager.findHost(oldhost[ll]).getModule(oldmod[ll], std::stoi(oldinst[ll]));
                moduleList.push_back(&oldApp);
                oldtitle[ll] = oldApp.fullName();
                oldparam[ll] = oldApp.get_parameter(controller::Direction::Input, false);
            }
            catch (const Exception &e)
            {
                (void)e; //ignore
            }
        }

        // store the current connection list
        getAllConnections();

        //  1. step
        //  delete old modules if it is a MOVE
        if (action == 2)
        {
            m_writeUndoBuffer = false;
            delModuleNode(moduleList);
            m_writeUndoBuffer = true;
        }

        // send message to UI that loading of a map will be started
        Message tmpmsg{COVISE_MESSAGE_UI, "START_READING\n"};
        m_hostManager.sendAll<Userinterface>(tmpmsg);

        //  2.
        //  start new modules and tell it to the uifs
        //  copy current parameters
        for (int ll = 0; ll < no; ll++)
        {
            const string &name = newmod[ll];
            const string &nr = newinst[ll];
            const string &host = newhost[ll];
            const string &ohost = oldhost[ll];
            const string &title = oldtitle[ll];
            int posx = std::stoi(newxpos[ll]);
            int posy = std::stoi(newypos[ll]);

            const NetModule *app = initModuleNode(name, nr, host, posx, posy, title, action, flags);
            if (!app)
                continue;

            newinst[ll] = std::to_string(app->instance());

            string myparam = oldparam[ll];
            vector<string> parameter = splitStringAndRemoveComments(myparam, "\n");

            int ipl = 0;
            int np = std::stoi(parameter[ipl++]);

            for (int l1 = 0; l1 < np; l1++)
            {
                const string &paramname = parameter[ipl++];
                const string &type = parameter[ipl++];
                const string &description = parameter[ipl++]; //dummy
                const string &value = parameter[ipl++];
                const string &imm = parameter[ipl++]; //dummy
                const string &appearanceType = parameter[ipl++];
                sendNewParam(name, newinst[ll], host, paramname, type, value, appearanceType, ohost);
            }
        }

        //  3.
        //  restore connections with new modules
        //  loop over all entries in the temporary connections list
        bool change = false;

        for (int kk = 0; kk < m_nconn; kk++)
        {
            string fname = from_name[kk];
            string fnr = from_inst[kk];
            string fhost = from_host[kk];

            string tname = to_name[kk];
            string tnr = to_inst[kk];
            string thost = to_host[kk];

            //  look if string is in the list of moved/copied modules
            for (int k2 = 0; k2 < no; k2++)
            {
                //  substitute module if found
                if (from_name[kk] == oldmod[k2] && from_inst[kk] == oldinst[k2] && from_host[kk] == oldhost[k2])
                {
                    fname = newmod[k2];
                    fnr = newinst[k2];
                    fhost = newhost[k2];
                    change = true;
                    break;
                }
            }

            //  don't allow connection to input ports outside the moved/copied modules
            if (action == 1)
                change = false;

            //  look if string is in the list of moved/copied modules
            for (int k2 = 0; k2 < no; k2++)
            {
                //  substitute module if found
                if (to_name[kk] == oldmod[k2] && to_inst[kk] == oldinst[k2] && to_host[kk] == oldhost[k2])
                {
                    tname = newmod[k2];
                    tnr = newinst[k2];
                    thost = newhost[k2];
                    change = true;
                    break;
                }
            }

            if (change)
                makeConnection(fname, fnr, fhost, from_port[kk], tname, tnr, thost, to_port[kk]);

            change = false;
        }

        tmpmsg = Message(COVISE_MESSAGE_UI, "END_READING\nfalse");
        m_hostManager.sendAll<Userinterface>(tmpmsg);
    }

    else if (key == "MASTERREQ")
    {
        const string &hostAddress = list[iel++];
        try
        {
            auto &newMasterHost = m_hostManager.findHost(hostAddress);
            auto &mod = newMasterHost.getProcess(sender_type::USERINTERFACE);
            if (auto ui = dynamic_cast<Userinterface *>(&mod))
            {
                m_hostManager.uiState.masterRequestSenderId = ui->processId;
                m_hostManager.getMasterUi().changeMaster(newMasterHost);
            }
            else
            {
                throw Exception{""};
            }
        }
        catch (const Exception &e)
        {
            string buffer = "MASTER-REQ FAILED: Bad hostAddress (" + string(hostAddress) + ")\n";
            Message tmpmsg{COVISE_MESSAGE_UI, buffer};
            m_hostManager.getMasterUi().send(&tmpmsg);
        }
    }

    else if (key == "USERNAME")
    {
        const string &rhost = list[iel++];
        const string &sender_name = list[iel++];
        const string &sender_nr = list[iel++];
        const string &sender_mod_id = list[iel++];
        string ruser_ind = "covise";
        if (list.size() < iel)
            string ruser_ind = list[iel++];
        try
        {
            const auto &host = m_hostManager.findHost(rhost);
            ruser_ind = host.userInfo().userName;
            Message msg{COVISE_MESSAGE_RENDER, "USERNAME\n" + sender_mod_id + "\n" + ruser_ind + "\n"};
            host.getModule(sender_name, std::stoi(sender_nr)).send(&msg);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "FORCE_MASTER")
    {
        iel++;
        iel++;
        const string &host = list[iel];
        iel++;
        cerr << host << " forced Master status\n"
             << endl;
        try
        {
            auto &master = m_hostManager.getMasterUi();
            auto newMaster = m_hostManager.findHost(host).getProcess(sender_type::USERINTERFACE).as<Userinterface>();
            master.setStatus(Userinterface::Status::Slave);
            newMaster->setStatus(Userinterface::Status::Master);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "DEL_REQ")
    {
        m_hostManager.sendAll<Userinterface>(*msg);

        // get modules
        vector<NetModule *> moduleList;
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        try
        {
            NetModule &app = m_hostManager.findHost(host).getModule(name, std::stoi(nr));
            moduleList.push_back(&app);
            if (app.info().name == "OpenCOVER" &&
                msg->send_type == sender_type::RENDERER &&
                coCoviseConfig::isOn("COVER.TerminateCoviseOnQuit", false))
            {
                resetLists();
                m_exit = true;
                return;
            }
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
            cerr << endl
                 << "---Controller : module : " << name << "_" << nr << "@" << host << " not found !!!\n";
        }
        // quit covise if TerminateCoviseOnQuit is set

        delModuleNode(moduleList);
    }

    else if (key == "DEL")
    {
        m_hostManager.sendAll<Userinterface>(*msg);
        //  no of deleted modules
        int no = std::stoi(list[iel++]);
        // get modules
        vector<NetModule *> moduleList;
        for (int i = 0; i < no; i++)
        {
            const string &name = list[iel++];
            const string &nr = list[iel++];
            const string &host = list[iel++];
            try
            {
                auto &app = m_hostManager.findHost(host).getModule(name, std::stoi(nr));
                moduleList.push_back(&app);
            }
            catch (const Exception &e)
            {
                std::cerr << e.what() << '\n';
                cerr << endl
                     << "---Controller : module : " << name << "_" << nr << "@" << host << " not found !!!\n";
            }
        }
        delModuleNode(moduleList);
    }

    else if (key == "DEL_DIED")
    {
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];

        //  delete Module and its Connections
        try
        {
            auto &h = m_hostManager.findHost(host);
            auto &app = h.getModule(name, std::stoi(nr));
            CTRLGlobal::getInstance()->modUIList->delete_mod(name, nr, host);
            h.removeModule(app, 1);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "MOV")
    {
        m_hostManager.sendAll<Userinterface>(*msg);
        //  no of moved modules
        int no = std::stoi(list[iel++]);

        // get modules && store old positions
        vector<NetModule::MapPosition> old_pos;
        vector<NetModule *> appList;
        for (int i = 0; i < no; i++)
        {
            const string &from_name = list[iel++];
            const string &from_nr = list[iel++];
            const string &from_host = list[iel++];

            int posx = std::stoi(list[iel++]);
            int posy = std::stoi(list[iel++]);
            try
            {
                auto &app = m_hostManager.findHost(from_host).getModule(from_name, std::stoi(from_nr));
                appList.push_back(&app);
                old_pos.push_back(app.pos());
                app.move({posx, posy});
            }
            catch (const Exception &e)
            {
                std::cerr << e.what() << '\n';
                cerr << endl
                     << "---Controller : module : " << from_name << "_" << from_nr << "@" << from_host << " not found !!!\n";
            }
        }
        // write to undo buffer
        if (m_writeUndoBuffer)
        {
            m_qbuffer.clear();
            m_qbuffer << "MOV" << QString::number(appList.size());
            for (int i = 0; i < appList.size(); i++)
            {
                m_qbuffer << appList[i]->createBasicModuleDescription().c_str();
                m_qbuffer << QString::number(old_pos[i].x) << QString::number(old_pos[i].y);
            }
            //qDebug() << "________________________________  " << m_qbuffer;
            addBuffer(m_qbuffer.join("\n"));
        }
    }

    else if (key == "CCONN")
    {
        broadcastMsgToOtherUis(*msg);

        const string &from_name = list[iel++];
        const string &from_nr = list[iel++];
        const string &from_host = list[iel++];
        const string &to_name = list[iel++];
        const string &to_nr = list[iel++];
        const string &to_host = list[iel++];

        try
        {
            auto &from = m_hostManager.findHost(from_host).getModule(from_name, std::stoi(from_nr));
            auto &to = m_hostManager.findHost(to_host).getModule(to_name, std::stoi(to_nr));
            from.to_c_connections.push_back(&to);
            to.from_c_connections.push_back(&from);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "CDEL")
    {
        broadcastMsgToOtherUis(*msg);
        // ?????
    }

    else if (key == "DEPEND")
    {
        broadcastMsgToOtherUis(*msg);

        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        const string &portname = list[iel++];
        const string &type = list[iel++];
        try
        {
            m_hostManager.findHost(host).getModule(name, std::stoi(nr)).connectivity().getInterface<C_interface>(portname).set_demand(type);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "DELETE_LINK")
    {
        m_hostManager.sendAll<Userinterface>(*msg);

        const string &from_name = list[iel++];
        const string &from_nr = list[iel++];
        const string &from_host = list[iel++];
        const string &from_port = list[iel++];
        const string &to_name = list[iel++];
        const string &to_nr = list[iel++];
        const string &to_host = list[iel++];
        const string &to_port = list[iel++];

        //  fetch object
        try
        {
            auto &fromApp = m_hostManager.findHost(from_host).getModule(from_name, std::stoi(from_nr));
            auto &fromInterface = fromApp.connectivity().getInterface<net_interface>(from_port);
            auto fromObj = fromInterface.get_object();

            if (fromObj)
            {
                auto &toApp = m_hostManager.findHost(to_host).getModule(to_name, std::stoi(to_nr));
                //auto &toInterface = toApp.connectivity().getInterface(to_port);
                //delete links
                toApp.delObjectConn(to_port, fromObj);
                fromObj->del_to_connection(to_name, to_nr, to_host, to_port);

                if (m_writeUndoBuffer)
                {
                    m_qbuffer.clear();
                    m_qbuffer << "OBJCONN";
                    m_qbuffer << from_name.c_str() << from_nr.c_str() << from_host.c_str() << from_port.c_str();
                    m_qbuffer << to_name.c_str() << to_nr.c_str() << to_host.c_str() << to_port.c_str();
                    addBuffer(m_qbuffer.join("\n"));
                }
            }
            else
            {
                cerr << "DELETE_LINK: Object not in Objectlist" << endl;
            }
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "OBJCONN")
    {

        m_hostManager.sendAll<Userinterface>(*msg);

        const string &from_name = list[iel++];
        const string &from_nr = list[iel++];
        const string &from_host = list[iel++];
        const string &from_port = list[iel++];

        const string &to_name = list[iel++];
        const string &to_nr = list[iel++];
        const string &to_host = list[iel++];
        const string &to_port = list[iel++];
        connectPorts(from_name, from_nr, from_host, from_port, to_name, to_nr, to_host, to_port);
    }

    else if (key == "OBJ")
    {
        // nothing to do
    }

    else if (key == "PIPELINE_STATE")
    {
        broadcastMsgToOtherUis(*msg);

        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        try
        {
            m_hostManager.findHost(host).getModule(name, std::stoi(nr)).send(&*msg);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key.find("MODULE_TITLE") != -1)
    {
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        const string &title = list[iel++];
        try
        {
            m_hostManager.findHost(host).getModule(name, std::stoi(nr)).setTitle(title);
            ostringstream buffer;
            buffer << "MODULE_TITLE\n"
                   << name << "\n"
                   << nr << "\n"
                   << host << "\n"
                   << title;
            Message tmpmsg{COVISE_MESSAGE_UI, buffer.str()};
            m_hostManager.sendAll<Userinterface>(tmpmsg);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << "CTRLHandler.cpp: MODULE_TITLE: did not find module: name=" << name << ", nr=" << nr << ", host=" << host << std::endl;
        }
    }

    else if (key.find("PARAM") != -1)
    {

        //  store parameter value
        //  send Parameterreplay to Module
        //  send Parameter to all UIF

        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        const string &portname = list[iel++];
        const string &porttype = list[iel++];
        try
        {
            auto &app = m_hostManager.findHost(host).getModule(name, std::stoi(nr));
            app.send(&*msg);
            m_hostManager.sendAll<Userinterface>(*msg);
            auto &param = app.connectivity().getParam(portname);
            if (m_writeUndoBuffer)
            {
                m_qbuffer.clear();
                m_qbuffer << "PARAM" << name.c_str() << nr.c_str() << host.c_str()
                          << portname.c_str() << porttype.c_str()
                          << param.get_val_list().c_str();
                addBuffer(m_qbuffer.join("\n"));
            }
            string value;
            if (iel < list.size()) // otherwise empty string
                value = list[iel++];
            param.set_value_list(value);
            //update siblings
            for (slist::iterator it = siblings.begin(); it != siblings.end(); it++)
            {
                std::string modName = name + "_" + nr;
                if (modName == it->first)
                    modName = it->second;
                else if (modName == it->second)
                    modName = it->first;
                else
                    continue;
                size_t pos = modName.find_last_of("_");
                std::string siblingName = modName.substr(0, pos);
                std::string siblingNr = modName.substr(pos + 1); // TODO send Message to Mapeditors
                m_hostManager.findHost(host).getModule(siblingName, std::stoi(siblingNr)).connectivity().getParam(portname).set_value_list(value);
            }

            //  update param for mirrored modules
            if (app.isOriginal())
            {
                for (NetModule *cpy : app.getMirrors())
                {
                    cpy->connectivity().getParam(portname).set_value_list(value);
                }
            }
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key.find("ADD_PANEL") != -1 || key.find("RM_PANEL") != -1)
    {
        m_hostManager.sendAll<Userinterface>(*msg);
        const string &name = list[iel++];
        const string &nr = list[iel++];
        const string &host = list[iel++];
        const string &param_name = list[iel++];
        const string &add_param = list[iel++];
        try
        {
            m_hostManager.findHost(host).getModule(name, std::stoi(nr)).connectivity().getParam(param_name).set_addvalue(add_param);
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    else if (key == "RENDERER_IMBEDDED_POSSIBLE" || key == "RENDERER_IMBEDDED_ACTIVE" || key == "FILE_SEARCH" || key == "FILE_LOOKUP")
    {

        const string &hostAddress = list[iel++];
        const string &username = list[iel++];

        // send request to crb
        auto crbs = m_hostManager.getAllModules<CRBModule>();
        auto host = std::find_if(m_hostManager.begin(), m_hostManager.end(), [&hostAddress, &username](const HostManager::HostMap::value_type &h) {
            return h.second->userInfo().ipAdress == hostAddress && h.second->userInfo().userName == username;
        });
        if (host != m_hostManager.end())
        {
            host->second->getProcess(sender_type::CRB).send(&*msg);
        }
    }

    else if (key == "HOSTINFO")
    {
        string hostAddress = list[iel++];
        if (!hostAddress.empty())
        {
            ExecType exectype = Config.getexectype(hostAddress);
            int timeout = Config.gettimeout(hostAddress);
            ostringstream buffer;
            buffer << "HOSTINFO\n"
                   << static_cast<int>(exectype) << "\n"
                   << timeout << "\n"
                   << hostAddress;

            Message tmpmsg{COVISE_MESSAGE_UI, buffer.str()};
            m_hostManager.getMasterUi().send(&tmpmsg);
        }

        else
        {
            Message tmpmsg{COVISE_MESSAGE_COVISE_ERROR, "A HOST SHOULD BE SPECIFIED !!!"};
            m_hostManager.sendAll<Userinterface>(tmpmsg);
        }
    }

    else if (key.find("NEW") != -1)
    {
        if (m_writeUndoBuffer)
        {
            string buffer = "GETCLIPBOARD_UNDO\n" + m_hostManager.getHostsInfo() + createApplicationsAndConnectionsData();
            addBuffer(buffer.c_str());
        }

        m_hostManager.sendAll<Userinterface>(*msg);
        resetLists();
    }

    else if (key == "SAVE")
    {
        const string &filename = list[iel++];
        saveCurrentNetworkFile(filename);
    }

    else if (key == "AUTOSAVE")
    {
        saveCurrentNetworkFile(m_autosavefile);
    }

    else if (key == "OPEN")
    {
        // clear undo buffer
        {
            m_undoBuffer.clear();
            Message undo_msg{COVISE_MESSAGE_UI, "UNDO_BUFFER_FALSE"};
            m_hostManager.sendAll<Userinterface>(undo_msg);
        }
        m_globalFilename = list[iel++];

        char *returnPath = NULL;
        FILE *fp = CoviseBase::fopen(m_globalFilename.c_str(), "r", &returnPath);
        if (fp)
        {
            m_globalFilename = returnPath;
            fclose(fp);
        }
        resetLists();
        m_hostManager.sendAll<Userinterface>(*msg);
        m_globalLoadReady = loadNetworkFile(m_globalFilename);
    }

    else if (key == "END_IMM_CB")
    {
        if (m_globalLoadReady)
        {
            if (m_options.uiOptions.iconify)
            {
                Message tmpmsg{COVISE_MESSAGE_UI, "ICONIFY"};
                m_hostManager.sendAll<Userinterface>(tmpmsg);
            }

            if (m_options.executeOnLoad)
            {
                m_options.executeOnLoad = false;
                for (NetModule *app : m_hostManager.getAllModules<NetModule>())
                {
                    if (app->isOnTop())
                        app->exec(m_numRunning);
                }
            }
        }
    }
    //       UI::DEFAULT
    // ----------------------------------------------------------

    else
    {
        m_hostManager.sendAll<Userinterface>(*msg);
    }
}

void CTRLHandler::handleNewUi(const NEW_UI &msg)
{
    switch (msg.type)
    {
    case NEW_UI_TYPE::HandlePartners:
    {
        auto &handlePartnerMsg = msg.unpackOrCast<NEW_UI_HandlePartners>();
        auto errors = m_hostManager.handleAction(handlePartnerMsg);
        assert(errors.size() == handlePartnerMsg.clients.size());
        if (handlePartnerMsg.launchStyle == LaunchStyle::Partner)
        {
            for (size_t i = 0; i < errors.size(); i++)
            {
                if (errors[i])
                {
                    const auto &ui = dynamic_cast<const Userinterface &>(m_hostManager.getHost(handlePartnerMsg.clients[i])->getProcess(sender_type::USERINTERFACE));

                    ui.sendCurrentNetToUI(m_globalFilename);
                    // add displays for the existing renderers on the new partner
                    for (const auto &renderer : m_hostManager.getAllModules<Renderer>())
                    {
                        if (renderer->isOriginal())
                        {
                            renderer->addDisplayAndHandleConnections(ui);
                        }
                    }
                }
            }
        }

        sendCollaborativeState();
    }
    break;
    case NEW_UI_TYPE::RequestAvailablePartners:
    {
        m_hostManager.sendPartnerList();
    }
    default:
        break;
    }
}

void CTRLHandler::sendGenericInfoToRenderer(const std::string &prefix, const Message &msg)
{
    DataHandle newData{msg.data.length() + prefix.size() + 1}; // +1 because of \n
    sprintf(newData.accessData(), "%s\n%s", prefix.c_str(), msg.data.data());
    newData.setLength((int)strlen(newData.data()) + 1);
    Message newMessage;
    newMessage.data = newData;
    newMessage.type = COVISE_MESSAGE_COVISE_ERROR;
    m_hostManager.sendAll<Renderer>(newMessage);
}

void CTRLHandler::broadcastMsgToOtherUis(const Message &msg)
{
    auto uis = m_hostManager.getAllModules<Userinterface>();
    auto ui = std::for_each(uis.begin(), uis.end(), [&msg](const Userinterface *u) {
        if (u->processId != msg.sender)
            u->send(&msg);
    });
}

void CTRLHandler::finishExecuteIfLastRunning(const NetModule &app)
{
    //  the last running module to delete = >finished exec
    if (app.status() != NetModule::Status::Idle && m_numRunning.apps == 1)
    {
        Message mapmsg{COVISE_MESSAGE_UI, "FINISHED\n"};
        m_hostManager.sendAll<Userinterface>(mapmsg);
        m_hostManager.sendAll<Renderer>(mapmsg);
    }
}

void CTRLHandler::saveCurrentNetworkFile(const std::string &filename)
{

#ifdef _WIN32
    std::ios_base::openmode openMode = std::ios::binary;
#else
    std::ios_base::openmode openMode = std::ios_base::out;
#endif
    std::filebuf *pbuf = nullptr;
    ofstream outFile(filename.c_str(), openMode);
    if (!outFile.good())
    {
        string covisepath = getenv("COVISEDIR");
        if (filename == "UNDO.NET" && !covisepath.empty())
        {
            string filestr = covisepath + "/" + filename;
            ofstream outFile2(filestr.c_str(), openMode);
            if (outFile2.good())
            {
                pbuf = outFile2.rdbuf();
            }
        }
    }
    else
        pbuf = outFile.rdbuf();

    if (!pbuf)
    {
        std::cerr << "ERROR: Error saving file " + filename << std::endl;
        Message err_msg(COVISE_MESSAGE_COVISE_ERROR, std::string("Error saving file " + filename));
        m_hostManager.sendAll<Userinterface>(err_msg);
        return;
    }

    if (filename.length() > 3 && filename.substr(filename.length() - 3, 3) == ".py")
    {
        // write a python script

        outFile << "#" << endl
                << "# create global net" << endl
                << "#" << endl
                << "network = net()" << endl;

        // store all modules
        for (const NetModule *app : m_hostManager.getAllModules<NetModule>())
        {
            app->writeScript(outFile);
        }
        // store all connections
        outFile << "#" << endl
                << "# CONNECTIONS" << endl
                << "#" << endl;
        CTRLGlobal::getInstance()->objectList->writeScript(outFile, m_hostManager.getLocalHost().userInfo().ipAdress, m_hostManager.getLocalHost().userInfo().userName);

        // same ending as python files from map_converter
        outFile << "#" << endl
                << "# uncomment the following line if you want your script to be executed after loading" << endl
                << "#" << endl
                << "#runMap()" << endl
                << "#" << endl
                << "# uncomment the following line if you want exit the COVISE-Python interface" << endl
                << "#" << endl
                << "#sys.exit()" << endl;
    }

    else // write a normal .net
    {

        // write content
        // get hosts
        outFile << "#" << NET_FILE_VERSION << endl;
        auto mdata = createApplicationsAndConnectionsData();
        if (!mdata.empty())
        {
            string data = m_hostManager.getHostsInfo() + mdata;
            pbuf->sputn(data.c_str(), data.length());
        }
    }
}

bool CTRLHandler::loadNetworkFile(const std::string &filename)
{
    return load_config(filename, *this, const_cast<const CTRLHandler *>(this)->m_hostManager.getAllModules<Userinterface>());
}

std::string CTRLHandler::createApplicationsAndConnectionsData()
{
    auto apps = m_hostManager.getAllModules<NetModule>();
    if (apps.empty())
        return std::string{};
    stringstream moduleData;
    moduleData << "#numModules\n"
               << apps.size() << "\n";
    for (const NetModule *app : apps)
    {
        moduleData << app->get_module(true);
    }
    // get module descrptions
    // get connections
    const auto &localHost = m_hostManager.getLocalHost();
    moduleData << CTRLGlobal::getInstance()->objectList->get_connections(localHost.userInfo().ipAdress, localHost.userInfo().userName);
    return moduleData.str();
}

//!
//! reset lists when NEW or OPEN was received
//!
void CTRLHandler::resetLists()
{
    auto apps = m_hostManager.getAllModules<NetModule>();
    if (!apps.empty())
    {
        Message msg{COVISE_MESSAGE_NEW_DESK, ""};
        m_hostManager.sendAll<CRBModule>(msg);
    }
    for (NetModule *app : apps)
    {
        //  go through the net_module_list and remove all modules
        //  and connections to modules
        app->setAlive(false);
        finishExecuteIfLastRunning(*app);
        CTRLGlobal::getInstance()->modUIList->delete_mod(app->info().name, std::to_string(app->instance()), app->host.userInfo().ipAdress);
    }
    for (NetModule *app : apps)
    {
        m_hostManager.findHost(app->host.userInfo().ipAdress).removeModule(*app, -1);
    }
    CTRLGlobal::getInstance()->objectList.reset(new object_list{});
    m_numRunning.apps = 0; //  no modules run
    m_numRunning.renderer = 0;
    // reset module counter & global id counter
    m_hostManager.resetModuleInstances();

    NetModule::resetId();
}

//!
//! simulate connection between ports after reading a map or adding a partner
//!
void CTRLHandler::makeConnection(const string &from_mod, const string &from_nr, const string &from_host,
                                 const string &from_port,
                                 const string &to_mod, const string &to_nr, const string &to_host,
                                 const string &to_port)
{
    if (auto connection = connectPorts(from_mod, from_nr, from_host, from_port, to_mod, to_nr, to_host, to_port))
    {
        try
        {
            auto &toApp = m_hostManager.findHost(to_host).getModule(to_mod, std::stoi(to_nr));
            auto &fromObjName = m_hostManager.findHost(from_host).getModule(from_mod, std::stoi(from_nr)).connectivity().getInterface<net_interface>(from_port).get_object()->get_name();
            if (!fromObjName.empty())
            {
                ostringstream oss;
                oss << "OBJCONN\n"
                    << from_mod << "\n"
                    << from_nr << "\n"
                    << from_host << "\n"
                    << from_port << "\n"
                    << to_mod << "\n"
                    << to_nr << "\n"
                    << to_host << "\n"
                    << to_port;

                Message tmp_msg{COVISE_MESSAGE_UI, oss.str()};
                m_hostManager.sendAll<Userinterface>(tmp_msg);
            }
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}

obj_conn *CTRLHandler::connectPorts(const string &from_name, const string &from_nr, const string &from_host,
                                    const string &from_port,
                                    const string &to_name, const string &to_nr, const string &to_host,
                                    const string &to_port)
{
    string fromObjName;
    try
    {
        auto &fromApp = m_hostManager.findHost(from_host).getModule(from_name, std::stoi(from_nr));
        auto &fromInterface = fromApp.connectivity().getInterface<net_interface>(from_port);
        auto fromObj = fromInterface.get_object();
        fromObjName = fromObj->get_name();
        fromObj = CTRLGlobal::getInstance()->objectList->select(fromObjName);
        auto &toApp = m_hostManager.findHost(to_host).getModule(to_name, std::stoi(to_nr));
        auto &toInterface = toApp.connectivity().getInterface<C_interface>(to_port);
        net_interface *toAppInterface = dynamic_cast<net_interface *>(&toInterface);

        obj_conn *connection = nullptr;
        if (!toAppInterface || !toAppInterface->get_conn_state())
        {
            toApp.setObjectConn(to_port, fromObj);
            connection = fromObj->connect_to(&toApp, to_port);
        }
        else
        {
            fprintf(stderr, "attempted duplicate connection to %s_%s:%s\n",
                    to_name.c_str(), to_nr.c_str(), to_name.c_str());
        }
        auto currObjName = fromObj->get_current_name();
        if (currObjName.empty())
            currObjName = fromObjName;
        if (auto renderer = dynamic_cast<Renderer *>(&toApp))
        {
            renderer->send_add_obj(currObjName);
            connection->set_old_name(currObjName);
        }
        return connection;
    }
    catch (const Exception &e)
    {
        std::cerr << e.what() << '\n';
        Message err_msg{COVISE_MESSAGE_WARNING, "Duplicate or connection to non-existing port " + fromObjName + " -> " + to_port + "(" + to_name + "_" + to_nr + "@" + to_host + ") !!!"};
        m_hostManager.sendAll<Userinterface>(err_msg);
        return nullptr;
    }
}

//!
//! delete a module
//!
void CTRLHandler::delModuleNode(const vector<NetModule *> &moduleList)
{

    // write to undo buffer
    if (m_writeUndoBuffer)
    {
        string buffer = writeClipboard("GETCLIPBOARD_UNDO", moduleList, true);
        m_qbuffer.clear();
        m_qbuffer << buffer.c_str();
        addBuffer(m_qbuffer.join("\n"));
    }

    // delete modules
    for (auto app : moduleList)
    {
        finishExecuteIfLastRunning(*app);
        ostringstream os;
        os << "DEL\n"
           << 1 << "\n"
           << app->createBasicModuleDescription();

        Message mapmsg(COVISE_MESSAGE_UI, os.str());
        m_hostManager.sendAll<Userinterface>(mapmsg);

        app->setAlive(0);
        CTRLGlobal::getInstance()->modUIList->delete_mod(app->processId);
        m_hostManager.findHost(app->host.userInfo().ipAdress).removeModule(*app, 0);
    }
}

//!
//! init a module
//!
const NetModule *CTRLHandler::initModuleNode(const string &name, const string &nr, const string &ipAddress,
                                             int posx, int posy, const string &title, int action, ExecFlag flags)
{
    try
    {
        auto &app = m_hostManager.findHost(ipAddress).startApplicationModule(name, nr, posx, posy, 0, flags);
        // send INIT message
        ostringstream os;
        os << "INIT\n"
           << name << "\n"
           << app.instance() << "\n"
           << ipAddress + "\n"
           << posx << "\n"
           << posy;

        Message tmp_msg{COVISE_MESSAGE_UI, os.str()};
        m_hostManager.sendAll<Userinterface>(tmp_msg);

        if (m_writeUndoBuffer)
        {
            m_qbuffer.clear();
            m_qbuffer << "DEL" << QString::number(1) << name.c_str() << QString::number(app.instance()) << ipAddress.c_str();
            //qDebug() << "________________________________  " << m_qbuffer;
            addBuffer(m_qbuffer.join("\n"));
        }

        ostringstream oss;
        oss << "DESC\n";
        oss << app.createDescription();

        tmp_msg = Message{COVISE_MESSAGE_UI, oss.str()};
        m_hostManager.sendAll<Userinterface>(tmp_msg);

        //  2 move mode: keep module title
        //  1 copy mode: add a useful nunber
        //  0 use stored title
        //  -1 dont"t send title
        if (action == 2)
            app.setTitle(title);
        // send TITLE message
        ostringstream osss;
        osss << "MODULE_TITLE\n"
             << name << "\n"
             << app.instance() << "\n"
             << ipAddress << "\n"
             << app.fullName();
        tmp_msg = Message{COVISE_MESSAGE_UI, osss.str()};
        m_hostManager.sendAll<Userinterface>(tmp_msg);
        return &app;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        ostringstream os;
        os << "Failing to start " << name << "@" << ipAddress;
        Message msg(COVISE_MESSAGE_COVISE_ERROR, os.str());
        m_hostManager.sendAll<Userinterface>(msg);
        return nullptr;
    }
}

NetModule *CTRLHandler::findApplication(const std::string &ipAddress, const std::string &name, int instance)
{
    try
    {
        for (auto &app : m_hostManager.findHost(ipAddress))
        {
            if (auto tmpApp = dynamic_cast<NetModule *>(&*app))
            {
                if (tmpApp->instance() == instance && tmpApp->info().name == name)
                {
                    return tmpApp;
                }
            }
        }
        return nullptr;
    }
    catch (const Exception &e)
    {
        std::cerr << e.what() << '\n';
        return nullptr;
    }
}

//!
//! send the new parameter to module and UI
//!
void CTRLHandler::sendNewParam(const string &name, const string &nr, const string &hostAddress,
                               const string &parameterName, const string &parameterType,
                               const string &parameterValue, const string &appType,
                               const string &oldhost, bool init)
{
    try
    {
        auto &application = m_hostManager.findHost(hostAddress).getModule(name, std::stoi(nr));
        const parameter *par = getParameter(application.connectivity().inputParams, parameterName);
        if (!par)
            return;
        string newVal = parameterValue;
        if (parameterType != par->get_type())
        {
            string buffer = "Changed type of parameter " + name + ":" + parameterName + " from " + parameterType + " to " + par->get_type();
            Message err_msg{COVISE_MESSAGE_WARNING, buffer};
            m_hostManager.sendAll<Userinterface>(err_msg);
            // read and convert old color string parameter to rgba value
            // f.e. blue <-> "0. 0. 1. 1."
            if (parameterType == "String" && par->get_type() == "Color")
            {
                if (!m_rgbTextOpen)
                {
                    char *returnPath = NULL;
                    fp = CoviseBase::fopen("share/covise/rgb.txt", "r", &returnPath);
                    if (fp != NULL)
                        m_rgbTextOpen = true;
                }

                newVal = "0. 0. 1. 1.";
                if (m_rgbTextOpen)
                {
                    char line[80];
                    while (fgets(line, sizeof(line), fp) != NULL)
                    {
                        if (line[0] == '!')
                            continue;

                        int count = 0;
                        const int tmax = 15;
                        char *token[tmax];
                        char *tp = strtok(line, " \t");
                        for (count = 0; count < tmax && tp != NULL;)
                        {
                            token[count] = tp;
                            tp = strtok(NULL, " \t");
                            count++;
                        }

                        // get color name
                        // remove \n fron end of string
                        string colorname(token[3]);
                        for (int i = 4; i < count; i++)
                        {
                            colorname.append(" ");
                            colorname.append(token[i]);
                        }
                        int ll = (int)colorname.length();
                        colorname.erase(ll - 1, 1);

                        if (colorname == parameterValue)
                        {
                            ostringstream os;
                            os << atof(token[0]) / 255. << " " << atof(token[1]) / 255. << " " << atof(token[2]) / 255. << " " << 1.;
                            auto newVal = os.str();

                            string buffer = "Changed value of parameter " + name + ":" + parameterName;
                            buffer.append(" from " + parameterValue + " to " + newVal);
                            err_msg = Message{COVISE_MESSAGE_WARNING, buffer};
                            m_hostManager.sendAll<Userinterface>(err_msg);
                            break;
                        }
                    }
                }
            }
        }
        if (parameterType == "Browser")
            newVal = handleBrowserPath(name, nr, hostAddress, oldhost, parameterName, parameterValue);
        auto &p = application.connectivity().getParam(parameterName);
        p.set_value_list(newVal);
        p.set_addvalue(appType);
        // split value parameter
        vector<string> parList = splitStringAndRemoveComments(newVal, " ");

        //  send parameter to modules & UIF
        ostringstream stream;
        stream << (init ? "PARAM_INIT\n" : "PARAM_NEW\n") << name << "\n"
               << nr << "\n"
               << hostAddress << "\n"
               << parameterName << "\n"
               << par->get_type() << "\n"
               << newVal;

        Message msg2{COVISE_MESSAGE_UI, stream.str()};
        m_hostManager.sendAll<Userinterface>(msg2);
        application.send(&msg2);
        //  send ADD_PANEL
        ostringstream ss;
        ss << "ADD_PANEL\n"
           << name << "\n"
           << nr << "\n"
           << hostAddress << "\n"
           << parameterName << "\n"
           << appType << "\n";
        msg2 = Message{COVISE_MESSAGE_UI, ss.str()};
        m_hostManager.sendAll<Userinterface>(msg2);
    }
    catch (const Exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

string CTRLHandler::handleBrowserPath(const string &name, const string &nr, const string &host, const string &oldhost,
                                      const string &parameterName, const string &parameterValue)
{
    string value = parameterValue;

    if (host != oldhost)
    {
        try
        {
            // get Datamanager for old host
            auto crb = m_hostManager.findHost(oldhost).getProcess(sender_type::CRB).as<CRBModule>();
            string path = crb->covisePath;
            string sep = path.substr(0, 1);
            path.erase(0, 1);
            vector<string> pathList = splitStringAndRemoveComments(path, sep);

            for (int i = 0; i < pathList.size(); i++)
            {
                string path = pathList[i];
                int find = (int)value.find(path);
                if (find != std::string::npos)
                {
                    value = value.substr(path.length());
                    while (value.length() > 0 && value[0] == '/')
                        value.erase(0, 1);
                    break;
                }
            }
        }
        catch (const Exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    ostringstream os;
    os << "FILE_LOOKUP\n"
       << host << "\nuser\n"
       << name << "\n"
       << nr << "\n"
       << parameterName << "\n"
                           "\n"
       << value;

    // send request for COVISE_PATH to new datamanager on new host
    try
    {
        auto crb = m_hostManager.findHost(host).getProcess(sender_type::CRB).as<CRBModule>();
        Message msg2{COVISE_MESSAGE_UI, os.str()};
        crb->send(&msg2);
        Message rmsg;
        crb->recv_msg(&rmsg);
        if (rmsg.type == COVISE_MESSAGE_UI)
        {
            vector<string> revList = splitStringAndRemoveComments(rmsg.data.data(), "\n");
            value = revList[7];
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    return value;
}

bool CTRLHandler::recreate(const string &content, readMode mode)
{
    // send message to UI that loading of a map has been started
    Message tmpmsg;
    if (mode == NETWORKMAP)
        tmpmsg = Message(COVISE_MESSAGE_UI, "START_READING\n" + m_globalFilename);
    else
        tmpmsg = Message(COVISE_MESSAGE_UI, "START_READING\n");
    m_hostManager.sendAll<Userinterface>(tmpmsg);

    m_writeUndoBuffer = false;

    string localIp = m_hostManager.getLocalHost().userInfo().ipAdress;
    string localuser = m_hostManager.getLocalHost().userInfo().userName;

    vector<string> mmodList; // list of obsolete modules

    int iel = 0;
    vector<string> list = splitStringAndRemoveComments(content, "\n");

    // read host information
    int numHosts = std::stoi(list[iel++]);

    bool allhosts = true;
    LaunchStyle addPartner = LaunchStyle::Host;

    // add hosts if not already added
    for (int i = 0; i < numHosts; i++)
    {
        string &hostAddress = list[iel++];
        string & username = list[iel++];
        if (hostAddress == "LOCAL")
            hostAddress = localIp;

        vector<string> token = splitStringAndRemoveComments(username, " ");
        if (token[0] == "LUSER")
            username = localuser;

        if (token.size() == 2)
        {
            if (token[1] == "Partner")
                addPartner = LaunchStyle::Partner;
            username = token[0];
        }
        try
        {
            RemoteHost &tmp_host = m_hostManager.findHost(hostAddress);
            try
            {
                tmp_host.getProcess(sender_type::CRB);
            }
            catch (const Exception &)
            {

                NEW_UI_HandlePartners msg{addPartner, 30, std::vector<int>{tmp_host.ID()}};
                handleNewUi(msg);
            }
        }
        catch (const Exception &)
        {
            NEW_UI_RequestNewHost hostRequest{hostAddress.c_str(), username.c_str(), m_hostManager.getVrbClient().getCredentials()};
            sendCoviseMessage(hostRequest, m_hostManager.getMasterUi());
            //allhosts = false;
        }
    }

    //  read all modules
    int numModules = std::stoi(list[iel++]);

    // craete list that contains module name, old number and new number
    vector<string> mnames;
    vector<string> oldInstances;
    vector<string> newInstances;

    string selectionBuffer;
    int ready = 0;

    //  start module if exist and tell it to the uifs
    for (int ll = 0; ll < numModules; ll++)
    {
        const string & name = list[iel++];
        const string & oldInstance = list[iel++];
        string host = list[iel++];
        if (host == "LOCAL")
            host = localIp;
        iel++; // category, not used

        string title = list[iel++];
        if (title.find("TITLE=", 0, 6) != -1)
            title.erase(0, 6);

        int posx = std::stoi(list[iel++]);
        int posy = std::stoi(list[iel++]);

        bool modExist = checkIfModuleAvailable(name, host);
        string newInstance;
        string current = oldInstance;
        if (modExist)
        {
            if (mode == CLIPBOARD)
            {
                current = "-1";
                posx = posx + 10;
                posy = posy + 10;
                m_writeUndoBuffer = true;
            }
            auto app = initModuleNode(name, current, host, posx, posy, title, 2, ExecFlag::Normal);
            if (app)
                newInstance = std::to_string(app->instance());
        }

        else
            mmodList.push_back(name);

        // wrap input ports
        int numInputPorts = std::stoi(list[iel++]);
        iel += numInputPorts * 5;

        // wrap output ports
        int numOutputPorts = std::stoi(list[iel++]);
        iel += numOutputPorts * 5;

        // update  parameter
        int numParameters = std::stoi(list[iel++]);
        for (int l1 = 0; l1 < numParameters; l1++)
        {
            const string & paramname = list[iel++];
            const string & type = list[iel++];
            iel++; //  unused description
            const string & value = list[iel++];
            iel++; //  unused IMM
            const string & apptype = list[iel++];
            if (modExist)
                sendNewParam(name, newInstance, host, paramname, type, value, apptype, host, mode == NETWORKMAP);
        }

        // wrap output parameter
        int numOutParams = std::stoi(list[iel++]);
        iel += numOutParams * 5;

        mnames.push_back(name);
        oldInstances.push_back(oldInstance);
        newInstances.push_back(newInstance);

        // when reading from a clipboard send a select for all pasted modules
        if (mode == CLIPBOARD && modExist)
        {
            ready++;
            selectionBuffer += name + "\n" + newInstance + "\n" + host + "\n";
        }
    }

    // send message for node selction
    if (mode == CLIPBOARD)
    {
        ostringstream os;
        os << ready;
        selectionBuffer = "SELECT_CLIPBOARD\n" + os.str() + "\n" + selectionBuffer;
        tmpmsg = Message{COVISE_MESSAGE_UI, selectionBuffer};
        m_hostManager.sendAll<Userinterface>(tmpmsg);
    }

    //connections
    int numConnections = std::stoi(list[iel++]);
    for (int ll = 0; ll < numConnections; ll++)
    {
        const string & fname = list[iel++];
        string &fnr = list[iel++];
        for (int k = 0; k < numModules; k++)
        {
            if (mnames[k] == fname && oldInstances[k] == fnr)
            {
                fnr = newInstances[k];
                break;
            }
        }
        string &fhost = list[iel++];
        if (fhost == "LOCAL")
            fhost = localIp;
        string fport = list[iel++];
        iel++; // unused data name

        const string &tname = list[iel++];
        string & tnr = list[iel++];
        for (int k = 0; k < numModules; k++)
        {
            if (mnames[k] == tname && oldInstances[k] == tnr)
            {
                tnr = newInstances[k];
                break;
            }
        }
        string &thost = list[iel++];
        if (thost == "LOCAL")
            thost = localIp;
        const string &tport = list[iel++];

        makeConnection(fname, fnr, fhost, fport, tname, tnr, thost, tport);
    }

    if (mode == NETWORKMAP)
        tmpmsg = Message(COVISE_MESSAGE_UI, "END_READING\ntrue");
    else
        tmpmsg = Message(COVISE_MESSAGE_UI, "END_READING\nfalse");
    m_hostManager.sendAll<Userinterface>(tmpmsg);

    mmodList.clear();
    m_writeUndoBuffer = true;

    return true;
}

bool CTRLHandler::checkIfModuleAvailable(const string &modname, const string &modhost)
{
    try
    {
        if (!m_hostManager.findHost(modhost).isModuleAvailable(modname))
            throw Exception{""};
    }
    catch (const Exception &e)
    {
        string data = "Error in load. Module " + modname + " on host " + modhost + " is not available. " + e.what() + "\n";
        Message msg{COVISE_MESSAGE_COVISE_ERROR, data};
        m_hostManager.getMasterUi().send(&msg);
        return false;
    }
    return true;
}

string CTRLHandler::writeClipboard(const string &keyword, const vector<NetModule *> &moduleList, bool all)
{
    // prepare buffer for return to UIF
    string buffer = keyword + "\n";

    // store hosts
    string hostnames = m_hostManager.getHostsInfo();
    buffer = buffer + hostnames;

    // store modules
    ostringstream temp;
    temp << moduleList.size();
    buffer = buffer + temp.str() + "\n";
    for (const NetModule *app : moduleList)
    {
        string erg = app->get_parameter(controller::Direction::Input, false);
        if (!app->isOriginal()) //old code checked of app is also in m_hostManager.getLocalHost<Application>()
        {
            buffer += app->get_module(true);
        }
    }

    // store only connections containing these modules
    // get all connections (used for UNDO of "Delete a node")
    string buffer2;
    int nconn = 0;

    vector<string> connList;
    if (all)
    {
        getAllConnections();

        for (int kk = 0; kk < m_nconn; kk++)
        {
            for (int k2 = 0; k2 < moduleList.size(); k2++)
            {
                auto tmp_mod = moduleList[k2];
                if (from_name[kk] == tmp_mod->info().name &&
                    std::stoi(from_inst[kk]) == tmp_mod->instance() &&
                    from_host[kk] == tmp_mod->host.userInfo().ipAdress)
                {
                    ostringstream erg;
                    erg << from_name[kk] << "\n"
                        << from_inst[kk] << "\n"
                        << from_host[kk] << "\n"
                        << from_port[kk] << "\n\n";
                    erg << to_name[kk] << "\n"
                        << to_inst[kk] << "\n"
                        << to_host[kk] << "\n"
                        << to_port[kk] << "\n";
                    vector<string>::iterator result;
                    result = find(connList.begin(), connList.end(), erg.str());
                    if (result == connList.end())
                    {
                        connList.push_back(erg.str());
                        nconn++;
                    }
                    break;
                }

                if (to_name[kk] == tmp_mod->info().name &&
                    std::stoi(to_inst[kk]) == tmp_mod->instance() &&
                    to_host[kk] == tmp_mod->host.userInfo().ipAdress)
                {
                    ostringstream erg;
                    erg << from_name[kk] << "\n"
                        << from_inst[kk] << "\n"
                        << from_host[kk] << "\n"
                        << from_port[kk] << "\n\n";
                    erg << to_name[kk] << "\n"
                        << to_inst[kk] << "\n"
                        << to_host[kk] << "\n"
                        << to_port[kk] << "\n";
                    vector<string>::iterator result;
                    result = find(connList.begin(), connList.end(), erg.str());
                    if (result == connList.end())
                    {
                        connList.push_back(erg.str());
                        nconn++;
                    }
                    break;
                }
            }
        }
        vector<string>::iterator iter;
        for (iter = connList.begin(); iter != connList.end(); iter++)
            buffer2 = buffer2 + *iter;
    }

    // store only connections inside a group
    else
    {
        object *tmp_obj;
        for (int ll = 0; ll < moduleList.size(); ll++)
        {
            CTRLGlobal::getInstance()->objectList->reset();
            while ((tmp_obj = CTRLGlobal::getInstance()->objectList->next()) != NULL)
            {
                const auto tmp_mod = tmp_obj->get_from().get_mod();
                if (tmp_mod && tmp_mod == moduleList[ll])
                {
                    int i = 0;
                    ostringstream res_str, from_str;

                    from_str << tmp_mod->info().name << "\n"
                             << tmp_mod->instance() << "\n";
                    if (tmp_mod->host.userInfo().ipAdress == m_hostManager.getLocalHost().userInfo().ipAdress)
                        from_str << "LOCAL";
                    else
                        from_str << tmp_mod->host.userInfo().ipAdress;

                    from_str << "\n"
                             << tmp_obj->get_from().get_intf() << "\n";

                    // get all to-connections inside the group
                    for (const auto &conn_tmp : tmp_obj->get_to())
                    {
                        const auto to_mod = conn_tmp.get_mod();
                        auto result = find_if(moduleList.begin(), moduleList.end(), [to_mod](const NetModule *a) { return a == to_mod; });
                        if (result != moduleList.end())
                        {
                            i++;
                            res_str << from_str.str() << "\n"
                                    << to_mod->info().name << "\n"
                                    << to_mod->instance() << "\n";
                            if (to_mod->host.userInfo().ipAdress == m_hostManager.getLocalHost().userInfo().ipAdress)
                                res_str << "LOCAL";
                            else
                                res_str << to_mod->host.userInfo().ipAdress;
                            res_str << "\n"
                                    << conn_tmp.get_mod_intf() << "\n";
                        }
                    }

                    string erg = res_str.str();
                    if (!erg.empty() || i != 0)
                    {
                        buffer2 = buffer2 + erg;
                        nconn = nconn + i;
                    }
                }
            }
        }
    }

    // complete return buffer
    string str;
    ostringstream out;
    out << nconn;
    str = out.str();
    buffer = buffer + str + "\n" + buffer2;

    return buffer;
}

void CTRLHandler::getAllConnections()
{
    string connections = CTRLGlobal::getInstance()->objectList->get_connections("dummy", "dummy");

    if (!connections.empty())
    {
        vector<string> token = splitStringAndRemoveComments(connections, "\n");
        int itl = 0;

        //  rearrange temporary connection list
        // no of connection
        istringstream t1(token[itl]);
        itl++;
        t1 >> m_nconn;

        from_name.clear();
        from_inst.clear();
        from_host.clear();
        from_port.clear();
        to_name.clear();
        to_inst.clear();
        to_host.clear();
        to_port.clear();

        for (int ll = 0; ll < m_nconn; ll++)
        {
            from_name.push_back(token[itl]);
            itl++; //   from_mod
            from_inst.push_back(token[itl]);
            itl++; //   from_inst
            from_host.push_back(token[itl]);
            itl++; //   from_host
            from_port.push_back(token[itl]);
            itl++; //   from_port
            itl++; //   objectname not needed
            to_name.push_back(token[itl]);
            itl++; //   to_mod
            to_inst.push_back(token[itl]);
            itl++; //   to_inst
            to_host.push_back(token[itl]);
            itl++; //   to_host
            to_port.push_back(token[itl]);
            itl++; //   to_port
        }
    }
}

void CTRLHandler::sendCollaborativeState()
{
    std::stringstream buffer;
    int numHosts = 0;
    for (const auto &host : m_hostManager)
    {
        if (host.second->state() == LaunchStyle::Host || host.second->state() == LaunchStyle::Host)
        {
            buffer << (host.second->state() == LaunchStyle::Host ? "COHOST" : "COPARTNER") << "\n"
                   << host.second->userInfo().ipAdress << "\n"
                   << host.second->userInfo().userName;

            try
            {
                auto &ui = host.second->getProcess(sender_type::USERINTERFACE);
                buffer << Userinterface::getStatusName(dynamic_cast<Userinterface &>(ui).status()) << "\n";
            }
            catch (const std::exception &e)
            {
                buffer << "NONE\n";
            }
            ++numHosts;
        }
    }
    stringstream text;
    text << "COLLABORATIVE_STATE\n"
         << numHosts << "\n"
         << buffer.str();
    Message msg{COVISE_MESSAGE_UI, text.str()};
    m_hostManager.sendAll<Userinterface>(msg);
}
