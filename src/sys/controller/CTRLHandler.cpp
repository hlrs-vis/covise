/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <signal.h>
#include <iostream>
#include <string>

#include <covise/covise.h>
#include <util/unixcompat.h>
#include <util/covise_version.h>
#include <util/coTimer.h>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <util/coFileUtil.h>
#include <net/covise_connect.h>
#include <net/tokenbuffer.h>
#include <covise/covise_msg.h>
#include <net/covise_host.h>
#include <appl/CoviseBase.h>

#include <vrbserver/VrbClientList.h>

#include "Token.h"
#include "CTRLHandler.h"
#include "CTRLGlobal.h"
#include "AccessGridDaemon.h"
#include "control_process.h"
#include "control_define.h"
#include "control_def.h"
#include "control_list.h"
#include "control_port.h"
#include "control_modlist.h"
#include "control_object.h"
#include "control_module.h"
#include "control_coviseconfig.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDebug>

using namespace covise;

CTRLHandler *CTRLHandler::singleton = NULL;

//  flag for adding partner from a file
//  if addpartner is true a signal was sent to the controller
bool m_addPartner = false;

//  global variable needed for SIGPWR && SIGTERM
ui_list *userinterfaceList;

//  dummy message to the Mapeditor. Mapeditor returns this message to
//  the controller. Result: wait_for_msg loop is left and signals can
//  be interpreted.
Message m_dummyMessage;

#ifndef __APPLE__
static void sigHandler(int sigNo) //  catch SIGPWR as addpartner signal
{
#if !defined(_WIN32) && !defined(__APPLE__)
    if (sigNo == SIGPWR)
    {
        m_addPartner = true;
        userinterfaceList->send_master(&m_dummyMessage);
    }
#endif
    return;
}
#endif

//  m_quitNow == 1 if a SIGTERM signal was sent to the controller
int m_quitNow = 0;

class sigQuitHandler : public coSignalHandler
{
    virtual void sigHandler(int sigNo) //  catch SIGTERM
    {
        if (sigNo == SIGTERM)
        {
            m_quitNow = 1;
            userinterfaceList->send_master(&m_dummyMessage);
        }
        return;
    }

    virtual const char *sigHandlerName() { return "sigQuitHandler"; }
};

const char *short_usage = "\n usage: covise [--help] [--minigui] [--gui] [--nogui] [network.net] [-a] [-d] [-i] [-m] [-p] [-s] [-t] [-u] [-q] [-x] [-e] [-v] [-V] [--script [script.py]]\n";
const char *long_usage = "\n"
                         "  -a\tscan for AccessGrid daemon\n"
                         "  -d\tscan for daemon\n"
                         "  -i\ticonify map editor\n"
                         "  -m\tmaximize map editor\n"
                         "  -q\tquit after execution\n"
                         "  -e\texecute on loading\n"
                         "  -u\tscan local user\n"
                         "  -t\tactivate timer\n"
                         "  -x\tstart X11 user interface\n"
                         "  --script [script.py]\n\tstart script mode, executing Python script script.py\n"
                         "  -v\tshort version\n"
                         "  -V\tlong version\n"
                         "  --version\n\tlong version\n"
                         "  --nogui\n\tdon't start the userinterface when loading and executing a network\n"
                         "  --gui\n\tstart userinterface, even in scripting mode (specify as last argument)\n"
                         "  --minigui\n\tStart a minimal userinterface when loading and executing a network\n"
                         "  --help\n\tthis help\n";

/*!
    \class CTRLHandler
    \brief Covise controller main handling   
*/

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
CTRLHandler::CTRLHandler(int argc, char *argv[])
    : m_miniGUI(false)
    , m_rgbTextOpen(false)
    , m_numRunning(0)
    , m_numRendererRunning(0)
    , m_accessGridDaemon(NULL)
    , m_globalLoadReady(true)
    , m_clipboardReady(true)
    , m_readConfig(true)
    , m_useGUI(true)
    , m_isLoaded(true)
    , m_executeOnLoad(false)
    , m_iconify(false)
    , m_maximize(false)
    , m_quitAfterExececute(0)
    , m_daemonPort(-1)
    , m_xuif(0)
    , m_startScript(0)
    , m_accessGridDaemonPort(0)
    , m_writeUndoBuffer(true)
    , m_handler(this)
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
{

    singleton = this;


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
    m_autosavefile = file.toStdString();

    m_dummyMessage.type = COVISE_MESSAGE_LAST_DUMMY_MESSAGE; //  initialize dummy message
    m_dummyMessage.data = DataHandle(2);
    memcpy(m_dummyMessage.data.accessData(), " ", 2);
    m_SSLDaemonPort = 0;
    m_SSLClient = NULL;

    // signal(SIGTERM, sigHandlerQuit );
    sigQuitHandler quitHandler;
    coSignal::addSignal(SIGTERM, quitHandler);

    //  parse commandline
    int ierr = parseCommandLine(argc, argv);
    if (ierr != 0)
        return;


    //  needed for SIGPWR && SIGTERM
    userinterfaceList = CTRLGlobal::getInstance()->userinterfaceList;

    coConfigEntryStringList list = coConfig::getInstance()->getScopeList("System.Siblings");

    //std::list<std::pair<std::string,std::string>> siblings;
    QLinkedList<coConfigEntryString>::iterator listentry = list.begin();
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

    //  Say hello
    cerr << endl;
    cerr << "*******************************************************************************" << endl;
    string text = CoviseVersion::shortVersion();
    string text2 = "* COVISE " + text + " starting up, please be patient....                    *";
    cerr << text2 << endl;
    cerr << "*                                                                             *" << endl;

    //  start crb, datamanager and UI
    startCrbUiDm();

    // load a network file
    loadNetworkFile();

    // start Controller main-Loop
    // check for daemons and handler messages
    bool startMainLoop = true;

    while (1)
    {
        Message *msg = new Message;
        if (m_addPartner)
        {
            //  input file stream
            ifstream input(m_filePartnerHost.c_str(), std::ios::in);

            if (!input.fail())
            {
                //  set timeout to 500 seconds
                char partner_host[128], partner_name[128];
                DataHandle partner_msg(400);
                input.getline(partner_host, 128);
                input.getline(partner_name, 128);

                sprintf(partner_msg.accessData(), "ADDPARTNER\n%s\n%s\nNONE\n%d\n500\n \n",
                        partner_host, partner_name, m_startScript ? COVISE_SCRIPT : COVISE_MANUAL);
                partner_msg.setLength(strlen(partner_msg.data()) + 1);
                msg->data = partner_msg;
                msg->type = COVISE_MESSAGE_UI;
                startMainLoop = false;
            }

            else
                startMainLoop = true;
        }

        else
            startMainLoop = true;

        if (!m_quitNow && startMainLoop)
        {
            msg = CTRLGlobal::getInstance()->controller->wait_for_msg();
        }

        handleAndDeleteMsg(msg);

    } //  while

    delete m_accessGridDaemon;
    m_accessGridDaemon = NULL;

    return;
}

CTRLHandler *CTRLHandler::instance()
{
    return singleton;
}

//!
//! Handle messages from the different covise parts
//!
void CTRLHandler::handleAndDeleteMsg(Message *msg)
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
        	m_handler.handleMessage(msg); // make sure VRB gets informed if the socket to a module is closed
			handleClosedMsg(msg);
        break;
    }

    case COVISE_MESSAGE_ACCESSGRID_DAEMON:
        handleAccessGridDaemon(msg);
        break;

    case COVISE_MESSAGE_SSLDAEMON:
        handleSSLDaemon(msg);
        break;

    case COVISE_MESSAGE_QUIT:
		m_handler.handleMessage(msg); // make sure VRB gets informed if the socket to a module is closed
		handleQuit(msg);
        break;

    //  FINALL: Module says it has finished
    case COVISE_MESSAGE_FINALL:
        handleFinall(msg, copyMessageData);
        break;

    //  FINISHED : Finish from a Rendermodule
    case COVISE_MESSAGE_FINISHED:
    {
        CTRLGlobal *global = CTRLGlobal::getInstance();
        if (global->netList->update(msg->sender, global->userinterfaceList))
        {
            if (m_numRunning == 0)
            {
                //  send Finished Message to the MapEditor
                //  if no modules are running
                global->userinterfaceList->slave_update();

                if (m_quitAfterExececute)
                {
                    m_quitNow = 1;
                    msg->data.setLength(0);
                    copyMessageData.clear();
                }

                Message *mapmsg = new Message(COVISE_MESSAGE_UI, "FINISHED\n");
                global->userinterfaceList->send_all(mapmsg);
                global->netList->send_all_renderer(mapmsg);
                delete mapmsg;
            }
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
        global->netList->send_renderer(msg);
        break;
    }

    case COVISE_MESSAGE_PARINFO:
    {
        // send message to all userinterfaces

        Message *new_msg = new Message(COVISE_MESSAGE_PARINFO, copyMessageData);
        global->userinterfaceList->send_all(new_msg);
        delete new_msg;

        // handle message, change parameter
        int iel = 0;
        vector<string> list = splitString(copyMessageData, "\n");
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;
        string portname = list[iel];
        iel++;
        string porttype = list[iel];
        iel++;
        string value = list[iel];
        iel++;

        global->netList->change_param(name, nr, host, portname, value);

        break;
    }

    //  WARNING : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_WARNING:
    {
        global->userinterfaceList->send_all(msg);
        global->netList->send_gen_info_renderer(msg);
        break;
    }

    //  INFO  : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_INFO:
    {
        global->userinterfaceList->send_all(msg);
        global->netList->send_gen_info_renderer(msg);
        break;
    }

    //  UPDATE_LOADED_MAPNAME  : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_UPDATE_LOADED_MAPNAME:
    {
        MARK0("UPDATE_LOADED_MAPNAME")
        global->userinterfaceList->send_all(msg);
        break;
    }

    //  REQ_UI : Messages are simply relayed to all Map-Editors
    case COVISE_MESSAGE_REQ_UI:
    {
        global->userinterfaceList->send_all(msg);
        break;
    }

    case COVISE_MESSAGE_COVISE_ERROR:
    {
        int iel = 0;
        vector<string> list = splitString(copyMessageData, "\n");
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        net_module *module = global->netList->get(name, nr, host);
        if (!module)
        {
            cerr << "COVISE_ERROR: did not find module  " << name << "_" << nr << " on " << host << endl;
            break;
        }

        int delta = module->error_owf();
        if (delta)
        {
            if (delta == 1)
            {
                Message *err_msg;
                for (int i = 0; i < module->m_errlist.size(); i++)
                {
                    err_msg = new Message(COVISE_MESSAGE_WARNING, module->m_errlist[i]);
                    global->userinterfaceList->send_all(err_msg);
                    delete err_msg;
                }

                string buffer = "Overflow of error messages from " + name + "_" + nr + " module on host " + host + " (last errors in \"Info Messages\")!";
                err_msg = new Message(COVISE_MESSAGE_COVISE_ERROR, buffer);
                global->userinterfaceList->send_all(err_msg);
                delete err_msg;
            }
        }

        else
        {
            module->add_error(msg);
            global->userinterfaceList->send_all(msg); //  error-msg sent
            global->netList->send_gen_info_renderer(msg);
        }

        //  change Modulestatus to STOP
        module->set_status(MODULE_STOP);
        break;
    }

    case COVISE_MESSAGE_COVISE_STOP_PIPELINE:
    {

        int iel = 0;
        vector<string> list = splitString(copyMessageData, "\n");
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        net_module *module = global->netList->get(name, nr, host);

        if (!module)
            cerr << "STOP_PIPELINE: did not find module  " << name << "_" << nr << " on " << host << endl;

        else
            module->set_status(MODULE_STOP);

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
        vector<string> list = splitString(copyMessageData, "\n");
        string key = list[iel];
        iel++;
        string action = list[iel];
        iel++;

        modui *tmpmod = global->modUIList->get(key);

        if (tmpmod == NULL)
        {
            /* Start der UIF-Teile */
            if (action == "INIT")
            {
                print_comment(__LINE__, __FILE__, " GENERIC: Wrong INIT-Message! ");
                print_exit(__LINE__, __FILE__, 1);
            }

            string name = list[iel];
            iel++;
            string instanz = list[iel];
            iel++;
            string host = list[iel];
            iel++;
            string executable = list[iel];
            iel++;
            string category = list[iel];
            iel++;

            global->modUIList->create_mod(name, instanz, category, host, key, executable);
        }

        else if (action == "APPINFO")
        {
            /* send Message to the APP-Part */
            tmpmod->sendapp(msg);
        }

        else if (action == "UIFINFO")
        {
            /* send Message to the UIF-Parts */
            tmpmod->send_msg(msg);
        }
        break;
    }

    //  FEEDBACK : Messages from Renderer sent to a module
    case COVISE_MESSAGE_FEEDBACK:
    {

        int iel = 0;
        vector<string> list = splitString(copyMessageData, "\n");
        string name = list[iel];
        iel++;
        string instanz = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        net_module *module = global->netList->get(name, instanz, host);

        if (module)
            module->send_msg(msg);

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
                Message *undo_msg = new Message(COVISE_MESSAGE_UI, copyMessageData);
                handleUI(undo_msg, copyMessageData);
                delete undo_msg;
                m_writeUndoBuffer = true;

                if (m_undoBuffer.isEmpty())
                {
                    Message *undo_msg = new Message(COVISE_MESSAGE_UI, "UNDO_BUFFER_FALSE");
                    global->userinterfaceList->send_all(undo_msg);
                    delete undo_msg;
                }
            }
        }

        else
            handleUI(msg, copyMessageData);

        break;
    }

    default:
        //check if it is a vrb message
        m_handler.handleMessage(msg);
        break;
    } //  end message switch

    if (m_quitNow == 0)
    {
        global->controller->delete_msg(msg);
        msg = NULL;
        copyMessageData.clear();
    }
}

//!
//! handle the message COVISE_MESSAGE_EMPTY, COVISE_MESSAGE_CLOSE_SOCKET, COVISE_MESSAGE_SOCKET_CLOSED:
//!
void CTRLHandler::handleClosedMsg(Message *msg)
{

    char msg_txt[2000];
    if (msg->conn == NULL)
        return;

    sender_type peer_type = (sender_type)msg->conn->get_peer_type();
    int peer_id = msg->conn->get_peer_id();
    CTRLGlobal *global = CTRLGlobal::getInstance();
    //  look which socket is broken
    switch (peer_type)
    {
    case RENDERER:
    case APPLICATIONMODULE:
    {
        net_module *p_mod = global->netList->get_mod(peer_id);
        if (p_mod != NULL)
        {
            string name = p_mod->get_name();
            string nr = p_mod->get_nr();
            string host = p_mod->get_host();

            //  remove the module
            int mod_type = p_mod->is_renderer();
            bool del_mod = false;

            if (mod_type == REND_MOD)
            {
                render_module *p_rend = (render_module *)p_mod;
                display *disp = p_rend->get_display(peer_id);
                sprintf(msg_txt, "The %s@%s display of the %s%s crashed !!!", disp->get_userid().c_str(), disp->get_hostname().c_str(), name.c_str(), nr.c_str());
                p_rend->remove_display(peer_id);
                if (p_rend->get_count() == 0)
                    del_mod = true;
            }

            if (mod_type == NET_MOD)
            {
                del_mod = true;
                sprintf(msg_txt, "Module %s_%s@%s crashed !!!", p_mod->get_name().c_str(), p_mod->get_nr().c_str(), p_mod->get_host().c_str());
            }

            global->userinterfaceList->sendError(msg_txt);

            // module have to be deleted
            if (del_mod)
            {
                sprintf(msg_txt, "DIED\n%s\n%s\n%s\n", name.c_str(), nr.c_str(), host.c_str());
                Message msg{ COVISE_MESSAGE_UI, DataHandle{msg_txt, strlen(msg_txt) + 1, false} };
                global->userinterfaceList->send_all(&msg);

                //  the last running module to delete = >finished exec
                if (p_mod->get_status() != 0 && CTRLHandler::m_numRunning == 1)
                {
                    Message *mapmsg = new Message(COVISE_MESSAGE_UI, "FINISHED\n");
                    global->userinterfaceList->send_all(mapmsg);
                    global->netList->send_all_renderer(mapmsg);
                    delete mapmsg;
                }
                p_mod->set_alive(0);
            }
        }
        break;
    }

    case USERINTERFACE:
    {
        userinterface *p_ui = global->userinterfaceList->get(peer_id);
        if (p_ui != NULL)
        {
            cerr << "Map editor crashed" << p_ui->get_userid() << "@" << p_ui->get_host() << endl;
            cerr << "Trying to restart session " << endl;
            sleep(5);
            p_ui->restart();
        }
        break;
    }

    // delete AccessGrid Daemon
    default:
    {
        if (m_accessGridDaemon && msg->conn == m_accessGridDaemon->conn)
        {
            delete m_accessGridDaemon;
            m_accessGridDaemon = NULL;
        }
        break;
    }
    }
}

//!
//! parse the commandline & init global states
//!
int CTRLHandler::parseCommandLine(int argc, char **argv)
{

    //  add partner manually: start script with crb parameters
    int scan_script_name = 0;

    //  file which contains the host of the partner
    int scan_add_partner_host = 0;

    //  name of local user
    int scan_m_localUser = 0;

    //  port to connect to the local daemon
    int scan_daemon = 0;

    // flag indicating whether secure connection is requested
    bool scan_SSLDaemon = false;

    bool scan_AccessGridDaemon = false;

    for (int i = 1; i < argc; i++)
    {
        if (scan_daemon)
        {
            //  argument: -d [port#] [room name]
            int retval = sscanf(argv[i], "%d", &m_daemonPort);
            if (retval != 1)
            {
                std::cerr << "main: sscanf failed" << std::endl;
                return -1;
            }

            if (++i == argc)
            {
                m_collaborationRoom = "ROOM";
            }

            else
            {
                m_collaborationRoom = argv[i];
            }

            scan_daemon = 0;
        }

        else if (scan_AccessGridDaemon)
        {
            //  argument: -a [port#]
            int retval = sscanf(argv[i], "%d", &m_accessGridDaemonPort);
            if (retval != 1)
            {
                std::cerr << "main: sscanf failed" << std::endl;
                return -1;
            }

            scan_AccessGridDaemon = false;
        }
        else if (scan_SSLDaemon)
        {
            //Do port retrieval for SSL conn
            std::string sport = std::string(argv[i]);
            std::stringstream strm;
            strm.str(sport);
            strm >> m_SSLDaemonPort;
        }

        else if (scan_add_partner_host)
        {
            //  argument: -p [m_filename]
            //     file format: <hostname>\n<username>
            m_filePartnerHost = argv[i];
            scan_add_partner_host = 0;
        }

        else if (scan_script_name)
        {
            //  argument: -s [m_filename]
            //  Starts script with the manual message "crb ..." as parameter for
            //  every connection
            m_scriptName = argv[i];
            scan_script_name = 0;
        }

        else if (scan_m_localUser)
        {
            //  argument: -u [username]
            //  set user name for the host the controller is running on
            m_localUser = argv[i];
            scan_m_localUser = 0;
        }

        else if (argv[i][0] == '-')
        {
            //  options
            switch (argv[i][1])
            {
            case 'r':
            case 'R':
            {
                //Secure connection for Covise Start from CoviseSSLDaemon
                scan_SSLDaemon = true;
                break;
            }
            case 'd':
            case 'D': //  daemon infos
            {
                scan_daemon = 1;
                break;
            }
            case 'a':
            case 'A': //  daemon infos
            {
                scan_AccessGridDaemon = true;
                break;
            }
            case 'i':
            case 'I':
                m_iconify = true;
                break;
            case 'm':
            case 'M':

                m_maximize = true;
                break;
            case 'p':
            case 'P':

//  initialize signal handler
#if !defined(_WIN32) && !defined(__APPLE__)
                signal(SIGPWR, sigHandler);
#endif
                scan_add_partner_host = 1;
                break;

            case 's':
            case 'S':
                m_startScript = 1;
                scan_script_name = 1;
                break;

            case 'v':
                printf("%s\n", CoviseVersion::shortVersion());
                exit(0);
                break;

            case 'V':
                printf("%s\n", CoviseVersion::longVersion());
                exit(0);
                break;

            case 'e':
            case 'E':
                m_executeOnLoad = true;
                break;
            case 't':
            {
                //  activate timing
                coTimer::init("timing", 2000);
                break;
            }
            case 'u':
            case 'U':
            {
                scan_m_localUser = 1;
                break;
            }
            case 'q':
            case 'Q':
            {
                m_quitAfterExececute = 1;
                break;
            }
            case 'x':
                m_xuif = 1;
                break;

            case '-':
                if (strncmp(argv[i], "--script", 8) == 0)
                {
                    m_xuif = 1;
                    m_useGUI = false;
                    if ((argv[i + 1]) && (strncmp(argv[i + 1], "--", 2)))
                    {
                        m_pyFile = argv[i + 1];
                        i++;
                        break;
                    }
                }
                else if (strncmp(argv[i], "--nogui", 7) == 0)
                {
                    m_useGUI = false;
                    m_executeOnLoad = true;
                    break;
                }
                else if (strncmp(argv[i], "--gui", 5) == 0)
                {
                    m_useGUI = true;
                    break;
                }

                else if (strncmp(argv[i], "--minigui", 9) == 0)
                {
                    m_miniGUI = true;
                    m_executeOnLoad = true;
                    break;
                }

                else if (strncmp(argv[i], "--version", 9) == 0)
                {
                    printf("%s\n", CoviseVersion::longVersion());
                    exit(0);
                    break;
                }
                else
                {
                    if (strncmp(argv[i], "--help", 6) != 0)
                        cerr << "Unrecognized Option -" << argv[i][1] << " \n";
                    cerr << short_usage;
                    cerr << long_usage;
                    exit(-1);
                }
                break;

            default:
            {
                cerr << "Unrecognized Option -" << argv[i][1] << " \n";
                cerr << short_usage;
                exit(-1);
                break;
            }
            } //  end options switch
        } //  end options if

        else if (strlen(argv[i]) > 4)
        {
            if (strcmp(argv[i] + strlen(argv[i]) - 4, ".net") == 0)
            {
                m_netfile = argv[i];
                m_isLoaded = true;
            }

            else if (strcmp(argv[i] + strlen(argv[i]) - 3, ".py") == 0 && m_useGUI)
            {
                m_xuif = 1;
                m_pyFile = argv[i];
            }
        }

        else
        {
            cerr << short_usage;
            exit(-1);
        }
    } //  end arguments - for

    if (!m_isLoaded && m_miniGUI)
    {
        cerr << endl;
        cerr << "***********************  E R R O R   **************************" << endl;
        cerr << "Starting COVISE with a minimal user interface is only possible " << endl;
        cerr << "with a network file (.net) given" << endl;
        cerr << "***************************************************************" << endl;
        exit(5);
    }

    if (!m_isLoaded && !m_useGUI)
    {
        cerr << endl;
        cerr << "***********************  E R R O R   **************************" << endl;
        cerr << "Starting COVISE without a user interface is only possible" << endl;
        cerr << "with a network file (.net) given" << endl;
        cerr << "***************************************************************" << endl;
        exit(6);
    }

    return 0;
}

//!
//! start the request broker, the datamanager and the user interface
//!
void CTRLHandler::startCrbUiDm()
{

    // read covise.config
    Config = new ControlConfig;

    //  start crb (covise request broker)
    cerr << "* Starting local request broker...                                            *" << endl;
    CTRLGlobal::getInstance()->userinterfaceList->set_iconify(m_iconify);
    CTRLGlobal::getInstance()->userinterfaceList->set_maximize(m_maximize);

    //  start data manager with local user name ( -u option)
    //  default: standard user name
    int ret = CTRLGlobal::getInstance()->hostList->add_local_host(m_localUser);
    if (ret == 0)
    {
        cerr << "* ...failed  (Increase Timeout for local machine)             *" << endl;
        exit(-1);
    }

    DM_data *local_crb = CTRLGlobal::getInstance()->dataManagerList->get_local();
    AppModule *module = local_crb->get_DM();
    CTRLGlobal::getInstance()->controller->get_shared_memory(module);

    //  start user interface
    cerr << "* Starting user interface....                                                 *" << endl;
    string moduleinfo = CTRLGlobal::getInstance()->moduleList->create_modulelist();

    if (m_xuif == 1)
    {
        CTRLGlobal::getInstance()->userinterfaceList->start_local_xuif(moduleinfo, m_pyFile);
    }
    //else
    {
        if (m_useGUI)
            CTRLGlobal::getInstance()->userinterfaceList->start_local_Mapeditor(moduleinfo);
#ifdef HAVE_GSOAP
        CTRLGlobal::getInstance()->userinterfaceList->start_local_WebService(moduleinfo);
#endif
    }

    //  connect to an AccessGrid Daemon
    if (m_accessGridDaemonPort)
    {
        cerr << "* Connect to AccessGridDaemon on Port = " << m_accessGridDaemonPort << "                                     *" << endl;
        m_accessGridDaemon = new AccessGridDaemon(m_accessGridDaemonPort);
    }
    if (m_SSLDaemonPort)
    {
#ifdef HAVE_OPENSSL
        cerr << "* Connect to SSLDaemon on Port = " << m_SSLDaemonPort << endl;
        Host *hostid = new Host("localhost");
        SSLClientConnection *conn = new SSLClientConnection(hostid, m_SSLDaemonPort, NULL, NULL);

        if (conn->AttachSSLToSocket(conn->getSocket()) == 0)
        {
            cerr << "Attaching socket FD to SSL failed!" << endl;
        }

        cerr << "Waiting for SSL-connect..." << endl;
        if (conn->connect() <= 0)
        {
            cerr << "SSL_Connect failed!" << endl;
        }

        CTRLGlobal::getInstance()->controller->addConnection(conn);
        cerr << "* SSL done! " << endl;
#else
        cerr << "* No SSL support" << endl;
#endif
    }

    cerr << "* ...done initialization                                                      *" << endl;
    cerr << "*******************************************************************************" << endl;
}

//!
//! if a network file was given in the commandline, try to load the map
//!
void CTRLHandler::loadNetworkFile()
{

    if (m_netfile.empty())
        return;

    //  look, if a path for searching is given, otherwise create a directory net
    char *returnPath = NULL;
    FILE *fp = CoviseBase::fopen(m_netfile.c_str(), "r", &returnPath);
    if (fp)
        m_globalFilename = m_netfile;

    else
    {
        string pathname = getenv("COVISEDIR");
        if (!pathname.empty())
            m_globalFilename = pathname + "/net/" + m_netfile;
    }
    CTRLGlobal *global = CTRLGlobal::getInstance();
    m_globalLoadReady = global->netList->load_config(m_globalFilename);
    m_isLoaded = false;

    if (m_globalLoadReady && m_executeOnLoad)
    {
        net_module *netmod;
        m_executeOnLoad = false;
        global->netList->reset();
        while ((netmod = global->netList->next()) != NULL)
        {
            if (netmod->is_on_top())
            {
                netmod->exec_module(global->userinterfaceList);
            }
        }
    }
}

//!
//! handle messages from the AccessGrid Daemon
//!
void CTRLHandler::handleAccessGridDaemon(Message *msg)
{

    if (strncmp(msg->data.data(), "join", 4) == 0)
    {
        const char *hname = msg->data.data() + 5;
        const char *passwd = "sessionpwd";
        const char *user_id = "AG2User";
        const char *c = strchr(hname, ':');

        if (c)
        {
            char *tmp = new char[strlen(c) + 1];
            strcpy(tmp, c);
            *tmp = '\0';
            size_t retval;
            retval = sscanf(tmp + 1, "%d", &m_accessGridDaemon->DaemonPort);
            if (retval != 1)
            {
                cerr << "main: sscanf failed" << endl;
                exit(-1);
            }
            delete[] tmp;
        }

        // set exectype to remote daemon which is 6
        Config->set_exectype(hname, "6");
        string hostname(hname);
        if (CTRLGlobal::getInstance()->userinterfaceList->add_partner(m_globalFilename, hostname, user_id, passwd, m_scriptName))
        {
            if (m_globalLoadReady == false)
            {
                m_globalLoadReady = CTRLGlobal::getInstance()->netList->load_config(m_globalFilename);
            }
        }

        else
        {
            char *msg_tmp = new char[200];
            sprintf(msg_tmp, "ADDPARTNER_FAILED\n%s\n%s\nPassword\n", hostname.c_str(), user_id);

            Message f_msg{ COVISE_MESSAGE_UI , DataHandle{msg_tmp, strlen(msg_tmp) + 1} };
            CTRLGlobal::getInstance()->userinterfaceList->send_master(&f_msg);
        }
    }
}

//!
//! handle QUIT messages from user interface, opencover or other sources
//!
void CTRLHandler::handleQuit(Message *msg)
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
            net_module *p_mod = CTRLGlobal::getInstance()->netList->get_mod(msg->conn->get_peer_id());
            if (p_mod)
            {
                const string name = p_mod->get_name();

                if (name == "VRRenderer" || name == "OpenCOVER" || name == "COVER" || name == "COVER_VRML")
                    terminateForCover = true;
            }
        }
    }

    bool fromUIF = CTRLGlobal::getInstance()->userinterfaceList->testid(msg->sender);

    if ((fromUIF == true)
        || (m_quitNow == 1)
        || terminateForCover)
    {
        CTRLGlobal::getInstance()->netList->reset();
        net_module *p_netmod = CTRLGlobal::getInstance()->netList->next();

        if (p_netmod)
        {
            //  send NEW_DESK to Datamanager
            CTRLGlobal::getInstance()->dataManagerList->new_desk();

            // go through the net_module_list and remove all modules
            // and connections to modules
            while (p_netmod)
            {
                p_netmod->set_alive(0);
                CTRLGlobal::getInstance()->modUIList->delete_mod(p_netmod->get_name(), p_netmod->get_nr(), p_netmod->get_host());
                p_netmod = CTRLGlobal::getInstance()->netList->next();
            }

            CTRLGlobal::getInstance()->netList->reset();
            p_netmod = CTRLGlobal::getInstance()->netList->next();
            while (p_netmod)
            {
                CTRLGlobal::getInstance()->netList->re_move(p_netmod->get_name(), p_netmod->get_nr(), p_netmod->get_host(), -1);
                CTRLGlobal::getInstance()->netList->reset();
                p_netmod = CTRLGlobal::getInstance()->netList->next();
            }
        }

        //  delete modulelist
        CTRLGlobal::getInstance()->moduleList->reset();
        module *mod;
        while ((mod = CTRLGlobal::getInstance()->moduleList->next()) != NULL)
            CTRLGlobal::getInstance()->moduleList->remove(mod);

        //  send quit to all userinterfaces
        CTRLGlobal::getInstance()->userinterfaceList->quit_and_del();

        //  send quit to Datamanager
        CTRLGlobal::getInstance()->dataManagerList->quit();
#ifdef _WIN32
        //  must be done for deleting shared memory files all aother processes must be closed before
        sleep(2);
#endif
        coTimer::quit();

        delete CTRLGlobal::getInstance()->controller;

        if (fromUIF && !m_quitNow && !m_autosavefile.empty())
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

        exit(0);
    }
}

//!
//! handle message after a module has been executed (finished)
//!
void CTRLHandler::handleFinall(Message *msg, string copyMessageData)
{

    int iel = 0;
    vector<string> list = splitString(copyMessageData, "\n");
    string name = list[iel];
    iel++;
    string nr = list[iel];
    iel++;
    string host = list[iel];
    iel++;

    net_module *module = CTRLGlobal::getInstance()->netList->get(name, nr, host);

    if (!module)
    {
        cerr << "Module was already deleted but has sent Finished" << endl;
        return;
    }

    module->empty_errlist();

    int noOfParameter;
    istringstream s1(list[iel]);
    iel++;
    s1 >> noOfParameter;

    int noOfSaveDataObjects;
    istringstream s2(list[iel]);
    iel++;
    s2 >> noOfSaveDataObjects;

    int noOfReleaseDataObjects;
    istringstream s3(list[iel]);
    iel++;
    s3 >> noOfReleaseDataObjects;

    //  read and change Parameter
    for (int i = 1; i <= noOfParameter; i++)
    {
        string parameterName = list[iel];
        iel++;
        iel++; // unused parameter type
        int noOfValues;
        istringstream inStream(list[iel]);
        iel++;
        inStream >> noOfValues;

        for (int iv = 1; iv <= noOfValues; iv++)
        {
            // workaround for choice param if only the current index is sent & not the parameter text
            if (list.size() == iel)
            {
                break;
            }
            string parameterValue = list[iel];
            iel++;
            module->change_param(parameterName, parameterValue, iv, 0);
        }
    }

    //  get the Dataobjects for SAVE
    for (int i = 1; i <= noOfSaveDataObjects; i++)
    {
        string dataObjectNames = list[iel];
        iel++;
        module->set_DO_status(DO_SAVE, dataObjectNames);
    }

    //  get the Dataobjects for RELEASE
    for (int i = 1; i <= noOfReleaseDataObjects; i++)
    {
        string dataObjectNames = list[iel];
        iel++;
        module->set_DO_status(DO_RELEASE, dataObjectNames);
    }

    //  send Message with Output-Parameters to all UIF
    module->send_finish();

    //  check if one level up is a module that has to be run
    int stat = module->get_status();
    module->set_status(MODULE_IDLE);
    if (!module->is_one_waiting_above(CTRLGlobal::getInstance()->userinterfaceList))
    {
        //  check if the module which has just finished
        //  has not been started again
        if (stat != MODULE_STOP)
        {
            if (module->get_num_running() == 0)
            {
                //  change Modulestatus to Finish
                module->set_status(MODULE_IDLE);

                //  start Following Modules
                module->set_start();
                module->start_modules(CTRLGlobal::getInstance()->userinterfaceList);
                module->set_status(MODULE_IDLE);
            }

            else
            {
                module->exec_module(CTRLGlobal::getInstance()->userinterfaceList);
            }
        }
    }

    // send Finished Message to the MapEditor if no modules are running
    m_numRunning--;
    if (m_numRunning == 0)
    {
        if (m_quitAfterExececute)
        {
            m_quitNow = 1;
            msg->data.setLength(0);
            copyMessageData.clear();
        }

        Message *mapmsg = new Message(COVISE_MESSAGE_UI, "FINISHED\n");
        CTRLGlobal::getInstance()->userinterfaceList->send_all(mapmsg);
        CTRLGlobal::getInstance()->netList->send_all_renderer(mapmsg);
        delete mapmsg;
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
        Message *undo_msg = new Message(COVISE_MESSAGE_UI, "UNDO_BUFFER_TRUE");
        CTRLGlobal::getInstance()->userinterfaceList->send_all(undo_msg);
        delete undo_msg;
    }
}

//!
//!  handle all message received from the user interface
//!
void CTRLHandler::handleUI(Message *msg, string copyData)
{
    string remain = copyData;
    int iel = 0;
    vector<string> list = splitString(remain, "\n");

    //  get Message-Keyword
    string key = list[iel];
    iel++;

    //       UI::EXEC
    // ----------------------------------------------------------

    if (key == "EXEC")
    {
        //  EXECUTE ON CHANGE
        if (list.size() == 4)
        {
            net_module *module = CTRLGlobal::getInstance()->netList->get(list[iel], list[iel + 1], list[iel + 2]);
            if (module)
                module->exec_module(CTRLGlobal::getInstance()->userinterfaceList);
        }

        else
        {
            CTRLGlobal::getInstance()->netList->reset();
            net_module *netmod;
            while ((netmod = CTRLGlobal::getInstance()->netList->next()) != NULL)
            {
                if (netmod->is_on_top() && netmod->get_mirror_status() != CPY_MIRR)
                    netmod->exec_module(CTRLGlobal::getInstance()->userinterfaceList);
            }
        }
    }

    //       UI::STATUS
    // ----------------------------------------------------------

    else if (key == "STATUS")
    {
        string status = list[iel];
        iel++;
        CTRLGlobal::getInstance()->userinterfaceList->send_new_status(status);
        CTRLGlobal::getInstance()->netList->set_renderstatus(CTRLGlobal::getInstance()->userinterfaceList);
        CTRLGlobal::getInstance()->modUIList->set_new_status();
        sendCollaborativeState();
    }

    //       UI::INIT
    // ----------------------------------------------------------

    else if (key == "INIT" || key == "INIT_DEBUG" || key == "INIT_MEMCHECK")
    {
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        int posx;
        istringstream s1(list[iel]);
        iel++;
        s1 >> posx;

        int posy;
        istringstream s2(list[iel]);
        iel++;
        s2 >> posy;

        //  entry in net-module-list
        Start::Flags flags = Start::Normal;
        if (key == "INIT_DEBUG")
            flags = Start::Debug;
        else if (key == "INIT_MEMCHECK")
            flags = Start::Memcheck;
        initModuleNode(name, nr, host, posx, posy, "", 0, flags);
    }
    else if (key == "GETDESC")
    {
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        net_module *module = CTRLGlobal::getInstance()->netList->get(name, nr, host);

        if (module)
        {

            ostringstream buffer;
            buffer << "PARAMDESC\n" << name << "\n" << nr << "\n" << host << "\n";
            vector<string> name_list;
            vector<string> type_list;
            vector<string> val_list;
            vector<string> panel_list;
            int n_pc = module->get_inpars_values(&name_list, &type_list, &val_list, &panel_list);

            buffer << n_pc << "\n";
            // loop over all input parameters
            for (int i = 0; i < n_pc; i++)
            {
                buffer << name_list[i]<<"\n";
                buffer << val_list[i] << "\n";
            }
            Message *tmpmsg = new Message(COVISE_MESSAGE_PARAMDESC, buffer.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
            delete tmpmsg;
        }
        else
        {
            std::cerr << "CTRLHandler.cpp: GETDESC: did not find module: name=" << name << ", nr=" << nr << ", host=" << host << std::endl;
        }
    }


    //       UI::COPY  (SYNC)
    // ----------------------------------------------------------

    else if (key == "COPY")
    {
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        int posx;
        istringstream s1(list[iel]);
        iel++;
        s1 >> posx;

        int posy;
        istringstream s2(list[iel]);
        iel++;
        s2 >> posy;

        //  Entry in net-module-list
        string title;
        initModuleNode(name, nr, host, posx, posy, title, 1, Start::Normal);
    }

    //       UI::MIRROR_ALL
    // ----------------------------------------------------------

    else if (key == "MIRROR_ALL")
    {
        CTRLGlobal::getInstance()->hostList->reset();
        CTRLGlobal::getInstance()->hostList->next();
        rhost *r_host;
        while ((r_host = CTRLGlobal::getInstance()->hostList->next()))
        {
            CTRLGlobal::getInstance()->netList->mirror_all(r_host->get_hostname());
        }
    }

    //       UI::REPLACE
    // ----------------------------------------------------------

    else if (key == "REPLACE")
    {
        // read parameter
        string newmod = list[iel];
        iel++; //  new_modulname
        string newinst = list[iel];
        iel++; //  new_instance
        string newhost = list[iel];
        iel++; //  new_host
        int posx;
        istringstream s1(list[iel]);
        iel++;
        s1 >> posx;

        int posy;
        istringstream s2(list[iel]);
        iel++;
        s2 >> posy;

        string oldmod = list[iel];
        iel++;
        string oldinst = list[iel];
        iel++;
        string oldhost = list[iel];
        iel++;

        //  store the current parameters of the module to be replaced
        int npold = 0;
        vector<net_module *> moduleList;
        vector<string> from_param;
        net_module *from_mod = CTRLGlobal::getInstance()->netList->get(oldmod, oldinst, oldhost);
        if (from_mod != NULL)
        {
            moduleList.push_back(from_mod);
            string buffer = from_mod->get_parameter("input", false);
            if (!buffer.empty())
            {
                from_param = splitString(buffer, "\n");
                istringstream npl(from_param[0]);
                npl >> npold;
            }
        }

        // get all connections
        getAllConnections();

        //  1.delete old module
        m_writeUndoBuffer = false;
        delModuleNode(moduleList);
        m_writeUndoBuffer = true;

        //  2. init new module
        // return new unique instance number from controller
        string title;
        int id = initModuleNode(newmod, newinst, newhost, posx, posy, title, 0, Start::Normal);
        newinst = (CTRLGlobal::getInstance()->netList->get(id))->get_nr();

        // 3. look if parameters can be reused
        int npnew = 0;
        vector<string> to_param;
        net_module *n_mod = CTRLGlobal::getInstance()->netList->get(id);
        if (n_mod != NULL)
        {
            string buffer = n_mod->get_parameter("input", false);
            if (!buffer.empty())
            {
                to_param = splitString(buffer, "\n");
                istringstream npl2(to_param[0]);
                npl2 >> npnew;

                if (npnew != 0 && npold != 0)
                {
                    for (int j = 1; j < npnew * 6 + 1; j = j + 6)
                    {
                        string paramname = to_param[j]; //  name
                        string type = to_param[j + 1]; //  type

                        for (int i = 1; i < npold * 6 + 1; i = i + 6)
                        {
                            if (paramname == from_param[i] && type == from_param[i + 1])
                            {
                                string value = from_param[i + 3]; //  value
                                string imm = from_param[i + 4]; //  IMM
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

        module *module = n_mod->get_type();
        module->reset_intflist();
        while (module->next_interface() != 0)
        {
            if (module->get_interfdirection() == "output")
                outInterfaces.push_back(module->get_interfname());
            else
                inpInterfaces.push_back(module->get_interfname());
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

    //       UI::SETCLIPBOARD
    // ----------------------------------------------------------

    else if (key == "SETCLIPBOARD")
    {

        //  no of modules
        string num = list[1];
        iel++;
        istringstream s1(num);
        int no;
        s1 >> no;

        //  allocate memory for module
        vector<net_module *> moduleList;
        for (int i = 0; i < no; i++)
        {
            net_module *from_mod = CTRLGlobal::getInstance()->netList->get(list[iel], list[iel + 1], list[iel + 2]);
            if (from_mod != NULL)
                moduleList.push_back(from_mod);
            else
                cerr << endl << "---Controller : module : " << list[iel] << "_" << list[iel + 1] << "@" << list[iel + 2] << " not found !!!\n";
            iel = iel + 3;
        }

        string buffer = writeClipboard("SETCLIPBOARD", moduleList);

        Message *tmpmsg = new Message(COVISE_MESSAGE_UI, buffer);
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
        delete tmpmsg;
    }

    //             UI::GETCLIPBOARD
    // ----------------------------------------------------------

    else if (key == "GETCLIPBOARD")
    {

        m_clipboardBuffer = copyData;
        int len = (int)key.length() + 1;
        m_clipboardBuffer.erase(0, len);
        m_clipboardReady = recreate(m_clipboardBuffer, CLIPBOARD);
    }

    //             UI::GETCLIPBOARD _UNDO
    // ----------------------------------------------------------

    else if (key == "GETCLIPBOARD_UNDO")
    {

        m_clipboardBuffer = copyData;
        int len = (int)key.length() + 1;
        m_clipboardBuffer.erase(0, len);
        m_clipboardReady = recreate(m_clipboardBuffer, UNDO);
    }

    //       UI::MOVE2/COPY2
    // ----------------------------------------------------------

    else if (key == "MOVE2" || key == "COPY2" || key == "MOVE2_DEBUG" || key == "MOVE2_MEMCHECK")
    {
        Start::Flags flags = Start::Normal;
        if (key == "MOVE2_DEBUG")
        {
            flags = Start::Debug;
        }
        else if (key == "MOVE2_MEMCHECK")
        {
            flags = Start::Memcheck;
        }

        //  no of moved/copied modules
        istringstream s1(list[iel]);
        iel++;
        int no;
        s1 >> no;

        //  MOVE = 2, COPY = 3
        istringstream s2(list[iel]);
        iel++;
        int action;
        s2 >> action;

        //  allocate memory
        vector<string> oldmod(no), oldinst(no), oldhost(no), oldparam(no), oldtitle(no);
        vector<string> newmod(no), newinst(no), newhost(no), newxpos(no), newypos(no);

        //  get old modules && store some stuff
        vector<net_module *> moduleList;
        for (int ll = 0; ll < no; ll++)
        {
            newmod[ll] = list[iel];
            iel++; //  new_mod
            newinst[ll] = list[iel];
            iel++; //  new_inst
            newhost[ll] = list[iel];
            iel++; //  new_host
            newxpos[ll] = list[iel];
            iel++; //  new_xpos
            newypos[ll] = list[iel];
            iel++; //  new_ypos

            oldmod[ll] = list[iel];
            iel++;
            oldinst[ll] = list[iel];
            iel++;
            oldhost[ll] = list[iel];
            iel++;
            net_module *old_mod = CTRLGlobal::getInstance()->netList->get(oldmod[ll], oldinst[ll], oldhost[ll]);
            if (old_mod != NULL)
            {
                moduleList.push_back(old_mod);
                oldtitle[ll] = old_mod->get_title();
                oldparam[ll] = old_mod->get_parameter("input", false);
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
        Message *tmpmsg = new Message(COVISE_MESSAGE_UI, "START_READING\n");
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
        delete tmpmsg;

        //  2.
        //  start new modules and tell it to the uifs
        //  copy current parameters
        for (int ll = 0; ll < no; ll++)
        {
            string name = newmod[ll];
            string nr = newinst[ll];
            string host = newhost[ll];
            string ohost = oldhost[ll];
            string title = oldtitle[ll];
            istringstream p1(newxpos[ll]);
            istringstream p2(newypos[ll]);

            int posx, posy;
            p1 >> posx;
            p2 >> posy;

            int id = initModuleNode(name, nr, host, posx, posy, title, action, flags);
            if (!CTRLGlobal::getInstance()->netList->get(id))
                continue;

            newinst[ll] = (CTRLGlobal::getInstance()->netList->get(id))->get_nr();

            string myparam = oldparam[ll];
            vector<string> parameter = splitString(myparam, "\n");

            int np = 0, ipl = 0;
            istringstream np1(parameter[ipl]);
            ipl++;
            np1 >> np;

            for (int l1 = 0; l1 < np; l1++)
            {
                string paramname = parameter[ipl];
                ipl++; //  name
                string type = parameter[ipl];
                ipl++; //  type
                string desc = parameter[ipl];
                ipl++; //  description
                string value = parameter[ipl];
                ipl++; //  value
                string imm = parameter[ipl];
                ipl++; //  IMM
                string apptype = parameter[ipl];
                ipl++; //  appearance type
                sendNewParam(name, newinst[ll], host, paramname, type, value, apptype, ohost);
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

        tmpmsg = new Message(COVISE_MESSAGE_UI, "END_READING\nfalse");
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
        delete tmpmsg;
    }

    //       UI::MASTER-REQUEST
    // ----------------------------------------------------------

    else if (key == "MASTERREQ")
    {
        string hostname1 = list[iel];
        iel++;

        Host thost(hostname1.c_str());
        Host host(thost.getAddress());
        string hostname = host.getAddress();

        userinterface *sender_ui = CTRLGlobal::getInstance()->userinterfaceList->get(hostname);

        if (sender_ui != 0)
        {
            CTRLGlobal::getInstance()->userinterfaceList->change_master(sender_ui->get_mod_id(), sender_ui->get_userid(), hostname1);
        }

        else
        {
            string buffer = "MASTER-REQ FAILED: Bad Hostname (" + string(hostname) + ")\n";
            Message *tmpmsg = new Message(COVISE_MESSAGE_UI, buffer);
            CTRLGlobal::getInstance()->userinterfaceList->send_master(tmpmsg);
            delete tmpmsg;
        }
    }

    //       UI::USERNAME-REQUEST
    // ----------------------------------------------------------

    else if (key == "USERNAME")
    {
        string rhost = list[iel];
        iel++;
        string sender_name = list[iel];
        iel++;
        string sender_nr = list[iel];
        iel++;
        string sender_mod_id = list[iel];
        iel++;
        string ruser_ind;
        if (list.size() < iel)
            string ruser_ind = list[iel];
        iel++;

        userinterface *rsender_ui = CTRLGlobal::getInstance()->userinterfaceList->get(rhost);

        //  quick solution: just one userinterface in mirror mode with addhost
        if (rsender_ui)
            ruser_ind = rsender_ui->get_userid();

        else
            ruser_ind = "covise";

        net_module *module_tmp = CTRLGlobal::getInstance()->netList->get(sender_name, sender_nr);

        if (module_tmp)
        {
            string buffer = "USERNAME\n" + sender_mod_id + "\n" + ruser_ind + "\n";
            Message *tmp_msg = new Message(COVISE_MESSAGE_RENDER, buffer);
            module_tmp->send_msg(tmp_msg);
            delete tmp_msg;
        }
    }

    //       UI::FORCE-MASTER
    // ----------------------------------------------------------

    else if (key == "FORCE_MASTER")
    {
        iel++;
        iel++;
        string host = list[iel];
        iel++;
        cerr << host << " forced Master status\n" << endl;
        ;
        CTRLGlobal::getInstance()->userinterfaceList->force_master(host);
    }

    //       UI::DEL
    // ----------------------------------------------------------

    else if (key == "DEL_REQ")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        //  no of deleted modules
        int no = 1;

        // get modules
        vector<net_module *> moduleList;
        for (int i = 0; i < no; i++)
        {
            net_module *p_netmod = CTRLGlobal::getInstance()->netList->get(list[iel], list[iel + 1], list[iel + 2]);
            if (p_netmod)
                moduleList.push_back(p_netmod);
            else
                cerr << endl << "---Controller : module : " << list[iel] << "_" << list[iel + 1] << "@" << list[iel + 2] << " not found !!!\n";
            iel = iel + 3;

            // quit covise if TerminateCoviseOnQuit is set
            if (p_netmod->get_name() == "OpenCOVER" && msg->send_type == RENDERER)
            {
                if (coCoviseConfig::isOn("COVER.TerminateCoviseOnQuit", false))
                {
                    CTRLGlobal::getInstance()->netList->reset();
                    net_module *p_netmod = CTRLGlobal::getInstance()->netList->next();

                    if (p_netmod)
                    {
                        //  send NEW_DESK to Datamanager
                        CTRLGlobal::getInstance()->dataManagerList->new_desk();

                        // go through the net_module_list and remove all modules
                        // and connections to modules
                        while (p_netmod)
                        {
                            p_netmod->set_alive(0);
                            CTRLGlobal::getInstance()->modUIList->delete_mod(p_netmod->get_name(), p_netmod->get_nr(), p_netmod->get_host());
                            p_netmod = CTRLGlobal::getInstance()->netList->next();
                        }

                        CTRLGlobal::getInstance()->netList->reset();
                        p_netmod = CTRLGlobal::getInstance()->netList->next();
                        while (p_netmod)
                        {
                            CTRLGlobal::getInstance()->netList->re_move(p_netmod->get_name(), p_netmod->get_nr(), p_netmod->get_host(), -1);
                            CTRLGlobal::getInstance()->netList->reset();
                            p_netmod = CTRLGlobal::getInstance()->netList->next();
                        }
                    }

                    //  delete modulelist
                    CTRLGlobal::getInstance()->moduleList->reset();
                    module *mod;
                    while ((mod = CTRLGlobal::getInstance()->moduleList->next()) != NULL)
                        CTRLGlobal::getInstance()->moduleList->remove(mod);

                    //  send quit to all userinterfaces
                    CTRLGlobal::getInstance()->userinterfaceList->quit_and_del();

                    //  send quit to Datamanager
                    CTRLGlobal::getInstance()->dataManagerList->quit();
#ifdef _WIN32
                    //  must be done for deleting shared memory files all aother processes must be closed before
                    sleep(2);
#endif
                    coTimer::quit();

                    delete CTRLGlobal::getInstance()->controller;

                    exit(0);
                }
            }
        }

        delModuleNode(moduleList);
    }
    else if (key == "DEL")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        //  no of deleted modules
        istringstream s1(list[iel]);
        iel++;
        int no;
        s1 >> no;

        // get modules
        vector<net_module *> moduleList;
        for (int i = 0; i < no; i++)
        {
            net_module *p_netmod = CTRLGlobal::getInstance()->netList->get(list[iel], list[iel + 1], list[iel + 2]);
            if (p_netmod)
                moduleList.push_back(p_netmod);
            else
                cerr << endl << "---Controller : module : " << list[iel] << "_" << list[iel + 1] << "@" << list[iel + 2] << " not found !!!\n";
            iel = iel + 3;
        }

        delModuleNode(moduleList);
    }

    //       UI::DEL_DIED
    // ----------------------------------------------------------

    else if (key == "DEL_DIED")
    {
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;

        //  delete Module and its Connections
        net_module *p_netmod = CTRLGlobal::getInstance()->netList->get(name, nr, host);

        if (p_netmod)
        {
            p_netmod->set_alive(0);
            CTRLGlobal::getInstance()->modUIList->delete_mod(name, nr, host);
            CTRLGlobal::getInstance()->netList->re_move(name, nr, host, 1);
        }
    }

    //       UI::MOV
    // ----------------------------------------------------------

    else if (key == "MOV")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        //  no of moved modules
        istringstream s1(list[iel]);
        iel++;
        int no;
        s1 >> no;

        // get modules && store old positions
        vector<int> old_posx, old_posy;
        vector<net_module *> moduleList;
        for (int i = 0; i < no; i++)
        {
            string from_name = list[iel];
            iel++;
            string from_nr = list[iel];
            iel++;
            string from_host = list[iel];
            iel++;

            int posx, posy;
            istringstream s1(list[iel]);
            iel++;
            s1 >> posx;
            istringstream s2(list[iel]);
            iel++;
            s2 >> posy;

            net_module *p_netmod = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);
            if (p_netmod)
            {
                moduleList.push_back(p_netmod);
                old_posx.push_back(p_netmod->get_x_pos());
                old_posy.push_back(p_netmod->get_y_pos());
                int id = p_netmod->get_nodeid();
                CTRLGlobal::getInstance()->netList->move(id, posx, posy);
            }

            else
                cerr << endl << "---Controller : module : " << from_name << "_" << from_nr << "@" << from_host << " not found !!!\n";
        }

        // write to undo buffer
        if (m_writeUndoBuffer)
        {
            m_qbuffer.clear();
            m_qbuffer << "MOV" << QString::number(moduleList.size());
            for (int i = 0; i < moduleList.size(); i++)
            {
                net_module *p_netmod = moduleList[i];
                m_qbuffer << p_netmod->get_name().c_str() << p_netmod->get_nr().c_str() << p_netmod->get_host().c_str();
                m_qbuffer << QString::number(old_posx[i]) << QString::number(old_posy[i]);
            }
            //qDebug() << "________________________________  " << m_qbuffer;
            addBuffer(m_qbuffer.join("\n"));
        }
    }

    //       UI::CCONN
    // ----------------------------------------------------------

    else if (key == "CCONN")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_slave(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;

        string to_name = list[iel];
        iel++;
        string to_nr = list[iel];
        iel++;
        string to_host = list[iel];
        iel++;

        CTRLGlobal::getInstance()->netList->set_C_conn(from_name, from_nr, from_host,
                                   to_name, to_nr, to_host);
    }

    //       UI::CDEL
    // ----------------------------------------------------------

    else if (key == "CDEL")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_slave(msg);
        // ?????
    }

    //       UI::DEPEND
    // ----------------------------------------------------------

    else if (key == "DEPEND")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_slave(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;

        string portname = list[iel];
        iel++;
        string type = list[iel];
        iel++;

        // select module and change interfacetype
        net_module *module = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);
        if (module != NULL)
            module->set_intf_demand(portname, type);
    }

    //       UI::DELETE_LINK
    // ----------------------------------------------------------

    else if (key == "DELETE_LINK")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;
        string from_port = list[iel];
        iel++;

        string to_name = list[iel];
        iel++;
        string to_nr = list[iel];
        iel++;
        string to_host = list[iel];
        iel++;
        string to_port = list[iel];
        iel++;

        //  fetch object
        net_module *n_mod = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);
        if (n_mod)
        {
            net_interface *nettmp = (net_interface *)n_mod->get_interfacelist()->get(from_port);
            if (nettmp)
            {
                object *obj = nettmp->get_object();
                if (obj == NULL)
                    cerr << "DIDEL: Object not in Objectlist" << endl;

                else
                {
                    CTRLGlobal::getInstance()->netList->del_DI_conn(to_name, to_nr, to_host, to_port, obj);
                    if (m_writeUndoBuffer)
                    {
                        m_qbuffer.clear();
                        m_qbuffer << "OBJCONN";
                        m_qbuffer << from_name.c_str() << from_nr.c_str() << from_host.c_str() << from_port.c_str();
                        m_qbuffer << to_name.c_str() << to_nr.c_str() << to_host.c_str() << to_port.c_str();
                        addBuffer(m_qbuffer.join("\n"));
                    }
                }
            }
        }
    }

    //       UI::OBJCONN
    // ----------------------------------------------------------

    else if (key == "OBJCONN")
    {

        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;
        string from_port = list[iel];
        iel++;

        string to_name = list[iel];
        iel++;
        string to_nr = list[iel];
        iel++;
        string to_host = list[iel];
        iel++;
        string to_port = list[iel];
        iel++;

        net_module *n_mod = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);
        if (n_mod)
        {
            net_interface *nettmp = (net_interface *)n_mod->get_interfacelist()->get(from_port);
            if (nettmp)
            {
                object *obj = nettmp->get_object();
                string object_name = obj->get_name();
                obj = CTRLGlobal::getInstance()->objectList->select(object_name);

                obj_conn *connection;
                connection = (obj_conn *)CTRLGlobal::getInstance()->netList->set_DI_conn(to_name, to_nr, to_host, to_port, obj);
                if (connection == NULL)
                {
                    string text = "Duplicate or connection to non-existing port " + string(object_name) + " -> " + to_port + "(" + to_name + "_" + to_nr + "@" + to_host + ") !!!";
                    Message *err_msg = new Message(COVISE_MESSAGE_WARNING, text);
                    CTRLGlobal::getInstance()->userinterfaceList->send_all(err_msg);
                    delete err_msg;
                }

                else
                {
                    if (!object_name.empty())
                    {
                        n_mod = CTRLGlobal::getInstance()->netList->get(to_name, to_nr, to_host);
                        n_mod->send_add_obj(obj->get_current_name(), connection);

                        if (m_writeUndoBuffer)
                        {
                            m_qbuffer.clear();
                            m_qbuffer << "DELETE_LINK";
                            m_qbuffer << from_name.c_str() << from_nr.c_str() << from_host.c_str() << from_port.c_str();
                            m_qbuffer << to_name.c_str() << to_nr.c_str() << to_host.c_str() << to_port.c_str();
                            //qDebug() << "________________________________  " << m_qbuffer;
                            addBuffer(m_qbuffer.join("\n"));
                        }
                    }
                }
            }
        }
    }

//       UI::PARCONN
// ----------------------------------------------------------
#ifdef PARAM_CONN
    else if (key == "PARCONN")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_slave(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;
        string output_name = list[iel];
        iel++;

        string to_name = list[iel];
        iel++;
        string to_nr = list[iel];
        iel++;
        string to_host = list[iel];
        iel++;
        string input_name = list[iel];
        iel++;

        CTRLGlobal::getInstance()->netList->set_P_conn(from_name, from_nr, from_host, output_name,
                                   to_name, to_nr, to_host, input_name);
    }

    //       UI::PARDEL
    // ----------------------------------------------------------

    else if (key == "PARDEL")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_slave(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;
        string output_name = list[iel];
        iel++;

        string to_name = list[iel];
        iel++;
        string to_nr = list[iel];
        iel++;
        string to_host = list[iel];
        iel++;
        string input_name = list[iel];
        iel++;

      CTRLGlobal::getInstance()->netList->del_P_conn((from_name, from_nr, from_host, output_name,
                                 to_name, to_nr, to_host, input_name);
    }

#endif

    //       UI::OBJ
    // ----------------------------------------------------------
    else if (key == "OBJ")
    {
        // nothing to do
    }

    //       UI::DIED
    // ----------------------------------------------------------

    else if (key == "DIED")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::PARREQ
    // ----------------------------------------------------------
    else if (key == "PARREQ")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::PIPELINE_STATE
    // ----------------------------------------------------------

    else if (key == "PIPELINE_STATE")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_slave(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;

        net_module *module = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);
        if (module != NULL)
            module->send_msg(msg);
    }

    //       UI::MODULE_TITLE
    // ----------------------------------------------------------

    else if (key.find("MODULE_TITLE") != -1)
    {
        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;
        string title = list[iel];
        iel++;

        net_module *module = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);

        if (module)
        {
            module->set_title(title);

            ostringstream buffer;
            buffer << "MODULE_TITLE\n" << from_name << "\n" << from_nr << "\n" << from_host << "\n" << title;
            Message *tmpmsg = new Message(COVISE_MESSAGE_UI, buffer.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
            delete tmpmsg;
        }
        else
        {
            std::cerr << "CTRLHandler.cpp: MODULE_TITLE: did not find module: name=" << from_name << ", nr=" << from_nr << ", host=" << from_host << std::endl;
        }
    }

    //       UI::PARAM
    // ----------------------------------------------------------

    else if (key.find("PARAM") != -1)
    {

        //  store parameter value
        //  send Parameterreplay to Module
        //  send Parameter to all UIF

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;

        net_module *org = CTRLGlobal::getInstance()->netList->get(from_name, from_nr, from_host);
        if (org)
        {
            org->send_msg(msg);
            CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

            string portname = list[iel];
            iel++;
            string porttype = list[iel];
            iel++;

            if (m_writeUndoBuffer)
            {
                string parvalue = org->get_one_param(portname);
                m_qbuffer.clear();
                m_qbuffer << "PARAM" << from_name.c_str() << from_nr.c_str() << from_host.c_str();
                m_qbuffer << portname.c_str() << porttype.c_str() << parvalue.c_str();
                //qDebug() << "________________________________  " << m_qbuffer;
                addBuffer(m_qbuffer.join("\n"));
            }

            string value;
            if (iel < list.size()) // otherwise empty string
                value = list[iel];
            CTRLGlobal::getInstance()->netList->change_param(from_name, from_nr, from_host, portname, value);
            for (slist::iterator it = siblings.begin(); it != siblings.end(); it++)
            {
                std::string modName = from_name + "_" + from_nr;
                if (modName == (*it).first)
                {
                    size_t pos = (*it).second.find_last_of("_");
                    std::string name = (*it).second.substr(0, pos);
                    std::string n = (*it).second.substr(pos + 1); // TODO send Message to Mapeditors
                    CTRLGlobal::getInstance()->netList->change_param(name, n, from_host, portname, value);
                }
                else if (modName == (*it).second)
                {
                    size_t pos = (*it).first.find_last_of("_");
                    std::string name = (*it).first.substr(0, pos);
                    std::string n = (*it).first.substr(pos + 1);
                    CTRLGlobal::getInstance()->netList->change_param(name, n, from_host, portname, value);
                }
            }

            //  update param for mirrored modules
            if (org && org->get_mirror_status() == ORG_MIRR)
            {
                net_module *modul;
                org->reset_mirror_list();
                while ((modul = org->mirror_list_next()) != NULL)
                {
                    CTRLGlobal::getInstance()->netList->change_param(from_name, modul->get_nr(), modul->get_host(),
                                                 portname, value);
                }
            }
        }
    }

    //       UI::PARSTATE
    // ----------------------------------------------------------

    else if (key == "PARSTATE")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::HIDE
    // ----------------------------------------------------------

    else if (key == "HIDE")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::SHOW
    // ----------------------------------------------------------

    else if (key == "SHOW")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::BROWSER_UPDATE
    // ----------------------------------------------------------

    else if (key == "BROWSER_UPDATE")
    {
        //  no longer used
    }

    //       UI::ADD_PANEL/RM_PANEL
    // ----------------------------------------------------------

    else if (key.find("ADD_PANEL") != -1 || key.find("RM_PANEL") != -1)
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        string from_name = list[iel];
        iel++;
        string from_nr = list[iel];
        iel++;
        string from_host = list[iel];
        iel++;
        string param_name = list[iel];
        iel++;
        string add_param = list[iel];
        iel++;

        CTRLGlobal::getInstance()->netList->add_param(from_name, from_nr, from_host, param_name, add_param);
    }

    //       UI::RENDERER_IMBEDDED_POSSIBLE
    // ----------------------------------------------------------

    else if (key == "RENDERER_IMBEDDED_POSSIBLE")
    {

        string hostname = list[iel];
        iel++;
        string username = list[iel];
        iel++;

        // send request to crb
        CTRLGlobal::getInstance()->dataManagerList->get(hostname, username)->get_DM()->send_msg(msg);
    }

    //       UI::RENDERER_IMBEDDED_NOT_ACTIVE
    // ----------------------------------------------------------
    else if (key == "RENDERER_IMBEDDED_ACTIVE")
    {

        string hostname = list[iel];
        iel++;
        string username = list[iel];
        iel++;

        // send request to crb
        CTRLGlobal::getInstance()->dataManagerList->get(hostname, username)->get_DM()->send_msg(msg);
    }

    //       UI::FILE_SEARCH
    // ----------------------------------------------------------

    else if (key == "FILE_SEARCH")
    {

        string hostname = list[iel];
        iel++;
        string username = list[iel];
        iel++;

        // send request to crb
        CTRLGlobal::getInstance()->dataManagerList->get(hostname, username)->get_DM()->send_msg(msg);
    }

    //       UI::FILE_LOOKUP
    // ----------------------------------------------------------

    else if (key == "FILE_LOOKUP")
    {

        string hostname = list[iel];
        iel++;
        string username = list[iel];
        iel++;

        // send request to crb
        CTRLGlobal::getInstance()->dataManagerList->get(hostname, username)->get_DM()->send_msg(msg);
    }

    //       UI::FILE_SEARCH_RESULT
    // ----------------------------------------------------------

    else if (key == "FILE_SEARCH_RESULT")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::FILE_LOOKUP_RESULT
    // ----------------------------------------------------------

    else if (key == "FILE_LOOKUP_RESULT")
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }

    //       UI::HOSTINFO
    // ----------------------------------------------------------

    else if (key == "HOSTINFO")
    {
        string hostname = list[iel];
        iel++;
        if (!hostname.empty())
        {
            int exectype = Config->getexectype(hostname);
            int timeout = Config->gettimeout(hostname);
            ostringstream buffer;
            buffer << "HOSTINFO\n" << exectype << "\n" << timeout << "\n" << hostname;

            Message *tmpmsg = new Message(COVISE_MESSAGE_UI, buffer.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_master(tmpmsg);
            delete tmpmsg;
        }

        else
            CTRLGlobal::getInstance()->userinterfaceList->sendError("A HOST SHOULD BE SPECIFIED !!!");
    }

    //       UI::ADD_HOST
    // ----------------------------------------------------------

    else if (key == "ADD_HOST")
    {
        bool completed = false;

        string hname = list[iel];
        iel++;
        string user_id = list[iel];
        iel++;
        string passwd = list[iel];
        iel++;
        string exectype = list[iel];
        iel++;
        string timeout = list[iel];
        iel++;
        string display;
        if (list.size() < iel)
            display = list[iel];
        iel++;

        string hostname = hname;
        if (!hostname.empty())
        {
            Config->set_exectype(hostname.c_str(), exectype.c_str());

            if (!timeout.empty())
                Config->set_timeout(hostname.c_str(), timeout.c_str());

            if (!display.empty())
                Config->set_display(hostname.c_str(), display.c_str());

            if (m_readConfig == false)
            {
                int ret = CTRLGlobal::getInstance()->userinterfaceList->config_action("", hostname, user_id, passwd);
                if (ret)
                {
                    m_readConfig = CTRLGlobal::getInstance()->userinterfaceList->add_config(m_filename, NULL);
                    if (m_readConfig == true)
                        completed = true;

                    if (m_readConfig && m_isLoaded)
                    {
                        m_globalLoadReady = CTRLGlobal::getInstance()->netList->load_config(m_netfile);
                        m_isLoaded = false;
                    }
                }
            }

            else
            {
                coHostType htype(CO_HOST);
                if (CTRLGlobal::getInstance()->hostList->add_host(hostname, user_id, passwd, "", htype))
                {
                    completed = 1;
                    if (!m_globalLoadReady)
                        m_globalLoadReady = CTRLGlobal::getInstance()->netList->load_config(m_globalFilename);

                    if (!m_clipboardReady)
                        m_clipboardReady = recreate(m_clipboardBuffer, CLIPBOARD);

                    // send infos about hosttype & ui status
                    sendCollaborativeState();
                }

                else
                {
                    string text = "ADDHOST_FAILED\n" + hostname + "\n" + user_id + "\nPassword\n";
                    Message *tmpmsg = new Message(COVISE_MESSAGE_UI, text);
                    CTRLGlobal::getInstance()->userinterfaceList->send_master(tmpmsg);

                    completed = false;
                } //  add_host
            }

            //  m_readConfig
            if (m_globalLoadReady && completed)
            {
                if (m_iconify)
                {
                    Message *tmpmsg = new Message(COVISE_MESSAGE_UI, "ICONIFY");
                    CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
                    delete tmpmsg;
                }

                if (m_maximize)
                {
                    Message *tmpmsg;
                    tmpmsg = new Message(COVISE_MESSAGE_UI, "MAXIMIZE");
                    CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
                    delete tmpmsg;
                }

                if (m_executeOnLoad)
                {
                    m_executeOnLoad = false;
                    CTRLGlobal::getInstance()->netList->reset();
                    net_module *netmod;
                    while ((netmod = CTRLGlobal::getInstance()->netList->next()) != NULL)
                    {
                        if (netmod->is_on_top())
                            netmod->exec_module(CTRLGlobal::getInstance()->userinterfaceList);
                    }
                }
            }
        } // hostname

        else
        {
            string text = "ADDHOST_FAILED\nBad Hostname\n" + user_id + "\nPassword\n";
            Message *tmpmsg = new Message(COVISE_MESSAGE_UI, text);
            CTRLGlobal::getInstance()->userinterfaceList->send_master(tmpmsg);
            delete tmpmsg;
        }
    }

    //       UI::ADD_PARTNER
    // ----------------------------------------------------------

    else if (key == "ADD_PARTNER")
    {
        if (m_addPartner)
            m_addPartner = false;

        string hname = list[iel];
        iel++;
        string user_id = list[iel];
        iel++;
        string passwd = list[iel];
        iel++;
        string exectype = list[iel];
        iel++;
        string timeout = list[iel];
        iel++;
        string display;
        if (list.size() < iel)
            display = list[iel];
        iel++;

        string hostname = hname;
        if (!hostname.empty())
        {
            Config->set_exectype(hostname.c_str(), exectype.c_str());

            if (!timeout.empty())
                Config->set_timeout(hostname.c_str(), timeout.c_str());

            if (!display.empty())
                Config->set_display(hostname.c_str(), display.c_str());

            if (CTRLGlobal::getInstance()->userinterfaceList->add_partner(m_globalFilename, hostname, user_id, passwd, m_scriptName))
            {
                if (!m_globalLoadReady)
                    m_globalLoadReady = CTRLGlobal::getInstance()->netList->load_config(m_globalFilename);

                if (!m_clipboardReady)
                    m_clipboardReady = recreate(m_clipboardBuffer, CLIPBOARD);

                // send infos about hosttype & ui status
                sendCollaborativeState();
            }

            else
            {
                string text = "ADDPARTNER_FAILED\n" + hostname + "\n" + user_id + "\nPassword\n";
                Message *tmpmsg = new Message(COVISE_MESSAGE_UI, text);
                CTRLGlobal::getInstance()->userinterfaceList->send_master(tmpmsg);

                delete tmpmsg;
            }

            m_addPartner = false;
        }

        else
        {
            string text = "ADDPARTNER_FAILED\nBad Hostname\n" + user_id + "\nPassword\n";
            Message *tmpmsg = new Message(COVISE_MESSAGE_UI, text);
            CTRLGlobal::getInstance()->userinterfaceList->send_master(tmpmsg);
            delete tmpmsg;
        }
    }

    //       UI::RMV_HOST
    // ----------------------------------------------------------

    else if (key == "RMV_HOST")
    {
        string w_hostname = list[iel];
        iel++;
        string w_user = list[iel];
        iel++;

        if (!w_hostname.empty())
        {
            string hostname = w_hostname;
            if (!hostname.empty())
            {
                if (!CTRLGlobal::getInstance()->hostList->rmv_host(hostname, w_user))
                {
                    string text = "REMOVING HOST OR PARTNER " + hostname + " FAILED !!!";
                    CTRLGlobal::getInstance()->userinterfaceList->sendError(text);
                }
            }

            else
            {
                string text = "HOSTNAME " + w_hostname + "NOT FOUND !!!";
                CTRLGlobal::getInstance()->userinterfaceList->sendError(text);
            }
        }

        else
            CTRLGlobal::getInstance()->userinterfaceList->sendError("A HOST SHOULD BE SPECIFIED !!!");
    }

    //       UI::RMV_PARTNER
    // ----------------------------------------------------------

    else if (key == "RMV_PARTNER")
    {
        string w_hostname = list[iel];
        iel++;
        string w_user = list[iel];
        iel++;

        if (!w_hostname.empty())
        {
            string hostname = w_hostname;
            if (!hostname.empty())
            {
                if (!CTRLGlobal::getInstance()->userinterfaceList->rmv_partner(hostname, w_user))
                {
                    string text = "REMOVING PARTNER ON HOST " + hostname + " FAILED !!!";
                    CTRLGlobal::getInstance()->userinterfaceList->sendError(text);
                }
            }

            else
            {
                string text = "HOSTNAME " + w_hostname + "NOT FOUND !!!";
                CTRLGlobal::getInstance()->userinterfaceList->sendError(text);
            }
        }

        else
        {
            CTRLGlobal::getInstance()->userinterfaceList->sendError("A HOST SHOULD BE SPECIFIED !!!");
        }
    }

    //       UI::NEW
    // ----------------------------------------------------------

    else if (key.find("NEW") != -1)
    {
        if (m_writeUndoBuffer)
        {
            string buffer = "GETCLIPBOARD_UNDO\n";

            //  get module infos (descrption & parameter values analog map saving)
            DM_data *dm_local = CTRLGlobal::getInstance()->dataManagerList->get_local();
            string localname = dm_local->get_hostname();
            string localuser = dm_local->get_user();

            // store hosts
            string hostnames = CTRLGlobal::getInstance()->hostList->get_hosts(localname, localuser);
            buffer = buffer + hostnames;

            // get module descrptions
            string mdata = CTRLGlobal::getInstance()->netList->get_modules(localname, localuser, true);
            if (!mdata.empty())
            {
                // get connections
                string cdata = CTRLGlobal::getInstance()->objectList->get_connections(localname, localuser);
                buffer = buffer + mdata + cdata;
            }
            addBuffer(buffer.c_str());
        }

        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
        resetLists();
    }

    //       UI::SAVE
    // ----------------------------------------------------------

    else if (key == "SAVE")
    {
        string filename = list[iel];
        iel++;
        CTRLGlobal::getInstance()->netList->save_config(filename);
    }

    //       UI::AUTOSAVE
    // ----------------------------------------------------------

    else if (key == "AUTOSAVE")
    {
        CTRLGlobal::getInstance()->netList->save_config(m_autosavefile);
    }

    //       UI::OPEN
    // ----------------------------------------------------------

    else if (key == "OPEN")
    {
        // clear undo buffer
        {
            m_undoBuffer.clear();
            Message *undo_msg = new Message(COVISE_MESSAGE_UI, "UNDO_BUFFER_FALSE");
            CTRLGlobal::getInstance()->userinterfaceList->send_all(undo_msg);
            delete undo_msg;
        }
        m_globalFilename = list[iel];
        iel++;

        char *returnPath = NULL;
        FILE *fp = CoviseBase::fopen(m_globalFilename.c_str(), "r", &returnPath);
        if (fp)
        {
            m_globalFilename = returnPath;
            fclose(fp);
        }

        resetLists();
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);

        m_globalLoadReady = CTRLGlobal::getInstance()->netList->load_config(m_globalFilename);
    }

    //       UI::END_IMM_CB
    // ----------------------------------------------------------

    else if (key == "END_IMM_CB")
    {
        if (m_globalLoadReady)
        {
            if (m_iconify)
            {
                Message *tmpmsg = new Message(COVISE_MESSAGE_UI, "ICONIFY");
                CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
                delete tmpmsg;
            }

            if (m_executeOnLoad)
            {
                net_module *netmod;
                m_executeOnLoad = false;
                CTRLGlobal::getInstance()->netList->reset();
                while ((netmod = CTRLGlobal::getInstance()->netList->next()) != NULL)
                {
                    if (netmod->is_on_top())
                        netmod->exec_module(CTRLGlobal::getInstance()->userinterfaceList);
                }
            }
        }
    }

    //       UI::DEFAULT
    // ----------------------------------------------------------

    else
    {
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    }
}

//!
//! reset lists when NEW or OPEN was received
//!
void CTRLHandler::resetLists()
{

    // delete all active modules
    CTRLGlobal::getInstance()->netList->reset();
    net_module *p_netmod = CTRLGlobal::getInstance()->netList->next();
    if (p_netmod)
    {
        //  send NEW_DESK to Datamanager
        CTRLGlobal::getInstance()->dataManagerList->new_desk();

        //  go through the net_module_list and remove all modules
        //  and connections to modules
        while (p_netmod)
        {
            p_netmod->set_alive(0);

            //  the last running module to delete = >finished exec
            if (p_netmod->get_status() != 0 && m_numRunning == 1)
            {
                Message *mapmsg = new Message(COVISE_MESSAGE_UI, "FINISHED\n");
                CTRLGlobal::getInstance()->userinterfaceList->send_all(mapmsg);
                CTRLGlobal::getInstance()->netList->send_all_renderer(mapmsg);
                delete mapmsg;
            }

            CTRLGlobal::getInstance()->modUIList->delete_mod(p_netmod->get_name(), p_netmod->get_nr(), p_netmod->get_host());
            p_netmod = CTRLGlobal::getInstance()->netList->next();
        }

        CTRLGlobal::getInstance()->netList->reset();
        p_netmod = CTRLGlobal::getInstance()->netList->next();
        while (p_netmod)
        {
            CTRLGlobal::getInstance()->netList->re_move(p_netmod->get_name(), p_netmod->get_nr(), p_netmod->get_host(), -1);
            CTRLGlobal::getInstance()->netList->reset();
            p_netmod = CTRLGlobal::getInstance()->netList->next();
        }
        m_numRunning = 0; //  no modules run
    }

    // reset module counter & global id counter
    CTRLGlobal::getInstance()->moduleList->reset();
    module *mod = CTRLGlobal::getInstance()->moduleList->next();
    while (mod)
    {
        mod->reset_counter();
        mod = CTRLGlobal::getInstance()->moduleList->next();
    }

    CTRLGlobal::getInstance()->s_nodeID = 0;
}

//!
//! simulate connection between ports after reading a map or adding a partner
//!
void CTRLHandler::makeConnection(const string &from_mod, const string &from_nr, const string &from_host,
                                 const string &from_port,
                                 const string &to_mod, const string &to_nr, const string &to_host,
                                 const string &to_port)
{
    net_interface *nettmp = NULL;
    net_module *n_mod = CTRLGlobal::getInstance()->netList->get(from_mod, from_nr, from_host);
    if (n_mod)
        nettmp = (net_interface *)n_mod->get_interfacelist()->get(from_port);

    if (nettmp)
    {
        object *obj = nettmp->get_object();
        string object_name = obj->get_name();
        obj = CTRLGlobal::getInstance()->objectList->select(object_name);

        obj_conn *connection;
        connection = (obj_conn *)CTRLGlobal::getInstance()->netList->set_DI_conn(to_mod, to_nr, to_host, to_port, obj);
        if (connection == NULL)
        {
            ostringstream os;
            os << "Duplicate or connection to non-existing port " << object_name << " -> " << to_port << "(" << to_mod << "_" << to_nr << "@" << to_host;
            Message *tmp_msg = new Message(COVISE_MESSAGE_WARNING, os.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(tmp_msg);
            delete tmp_msg;
        }
        else
        {
            if (!object_name.empty())
            {
                (CTRLGlobal::getInstance()->netList->get(to_mod, to_nr, to_host))->send_add_obj(object_name, connection);
                ostringstream oss;
                oss << "OBJCONN\n" << from_mod << "\n" << from_nr << "\n" << from_host << "\n" << from_port << "\n";
                oss << to_mod << "\n" << to_nr << "\n" << to_host << "\n" << to_port;

                Message *tmp_msg = new Message(COVISE_MESSAGE_UI, oss.str());
                CTRLGlobal::getInstance()->userinterfaceList->send_all(tmp_msg);
                delete tmp_msg;
            }
        }
    }
}

//!
//! delete a module
//!
void CTRLHandler::delModuleNode(vector<net_module *> moduleList)
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
    for (int i = 0; i < moduleList.size(); i++)
    {
        net_module *p_netmod = moduleList[i];

        //  the last running module to delete = >finished exec
        Message *mapmsg;
        if (p_netmod->get_status() != 0 && m_numRunning == 1)
        {
            mapmsg = new Message(COVISE_MESSAGE_UI, "FINISHED\n");
            CTRLGlobal::getInstance()->userinterfaceList->send_all(mapmsg);
            CTRLGlobal::getInstance()->netList->send_all_renderer(mapmsg);
            delete mapmsg;
        }
        ostringstream os;
        os << "DEL\n" << 1 << "\n" << p_netmod->get_name() << "\n" << p_netmod->get_nr() << "\n" << p_netmod->get_host();

        mapmsg = new Message(COVISE_MESSAGE_UI, os.str());
        CTRLGlobal::getInstance()->userinterfaceList->send_all(mapmsg);
        delete mapmsg;

        int id = p_netmod->get_nodeid();
        p_netmod->set_alive(0);
        CTRLGlobal::getInstance()->modUIList->delete_mod(id);
        CTRLGlobal::getInstance()->netList->re_move(id, 0);
    }
}

//!
//! init a module
//!
int CTRLHandler::initModuleNode(const string &name, const string &nr, const string &host,
                                int posx, int posy, const string &title, int action, Start::Flags flags)
{
    CTRLGlobal::getInstance()->s_nodeID++;
    int s_nodeID = CTRLGlobal::getInstance()->s_nodeID;
    int count = CTRLGlobal::getInstance()->netList->init(s_nodeID, name, nr, host, posx, posy, 0, flags);
    //create vrb client for OpenCOVER
    if (count != 0)
    {
        // send INIT message
        ostringstream os;
        os << "INIT\n" << name << "\n" << count << "\n" << host + "\n" << posx << "\n" << posy;

        Message *tmp_msg = new Message(COVISE_MESSAGE_UI, os.str());
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmp_msg);
        delete tmp_msg;

        if (m_writeUndoBuffer)
        {
            m_qbuffer.clear();
            m_qbuffer << "DEL" << QString::number(1) << name.c_str() << QString::number(count) << host.c_str();
            //qDebug() << "________________________________  " << m_qbuffer;
            addBuffer(m_qbuffer.join("\n"));
        }

        // send DESC message
        net_module *n_mod = CTRLGlobal::getInstance()->netList->get(s_nodeID);
        module *module = n_mod->get_type();
        ostringstream oss;
        oss << "DESC\n";
        if (module)
            oss << module->create_descr();

        tmp_msg = new Message(COVISE_MESSAGE_UI, oss.str());
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmp_msg);
        delete tmp_msg;

        //  2 move mode: keep module title
        //  1 copy mode: add a useful nunber
        //  0 use stored title
        //  -1 dont"t send title
        if (action == 2)
            n_mod->set_title(title);

        // send TITLE message
        ostringstream osss;
        osss << "MODULE_TITLE\n" << name << "\n" << count << "\n" << host << "\n" << n_mod->get_title();
        tmp_msg = new Message(COVISE_MESSAGE_UI, osss.str());
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmp_msg);
        delete tmp_msg;
        return s_nodeID;
    }

    else
    {
        ostringstream os;
        os << "Failing to start " << name << "_" << count << "@" << host;
        string data = os.str();
        CTRLGlobal::getInstance()->userinterfaceList->sendError(data);
        return -1;
    }
}

//!
//! send the new parameter to module and UI
//!
void CTRLHandler::sendNewParam(const string &name, const string &nr, const string &host,
                               const string &parameterName, const string &parameterType,
                               const string &parameterValue, const string &appType,
                               const string &oldhost, bool init)
{

    net_module *tmp_net = CTRLGlobal::getInstance()->netList->get(name, nr, host);
    parameter *par = NULL;
    if (tmp_net)
    {
        module *tmp_mod = tmp_net->get_type();
        if (tmp_mod)
        {
            par = tmp_mod->get_parameter("in", parameterName);
        }
    }

    if (par != NULL)
    {
        string parType = par->get_type();
        {
            string newVal = parameterValue;
            if (parameterType != parType)
            {
                string buffer = "Changed type of parameter " + name + ":" + parameterName;
                buffer.append(" from " + parameterType + " to " + parType);
                Message *err_msg = new Message(COVISE_MESSAGE_WARNING, buffer);
                CTRLGlobal::getInstance()->userinterfaceList->send_all(err_msg);
                delete err_msg;

                // read and convert old color string parameter to rgba value
                // f.e. blue <-> "0. 0. 1. 1."
                if (parameterType == "String" && parType == "Color")
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
                                newVal = os.str();

                                string buffer = "Changed value of parameter " + name + ":" + parameterName;
                                buffer.append(" from " + parameterValue + " to " + newVal);
                                err_msg = new Message(COVISE_MESSAGE_WARNING, buffer);
                                CTRLGlobal::getInstance()->userinterfaceList->send_all(err_msg);
                                delete err_msg;

                                break;
                            }
                        }
                    }
                }
            }

            // strip COVISE_PATH entries from filename
            // and add new one for new host
            if (parameterType == "Browser")
                newVal = handleBrowserPath(name, nr, host, oldhost, parameterName, parameterValue);

            //  change-values
            CTRLGlobal::getInstance()->netList->change_param(name, nr, host, parameterName, newVal);
            CTRLGlobal::getInstance()->netList->add_param(name, nr, host, parameterName, appType);

            // split value parameter
            vector<string> parList = splitString(newVal, " ");

            //  send parameter to modules & UIF
            ostringstream stream;
            stream << (init ? "PARAM_INIT\n" : "PARAM_NEW\n") << name << "\n" << nr << "\n" << host << "\n"
                   << parameterName << "\n" << parType << "\n" << newVal;

            Message *msg2 = new Message(COVISE_MESSAGE_UI, stream.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(msg2);
            tmp_net->send_msg(msg2);
            delete msg2;

            //  send ADD_PANEL
            ostringstream ss;
            ss << "ADD_PANEL\n" << name << "\n" << nr << "\n" << host << "\n" << parameterName << "\n" << appType << "\n";
            msg2 = new Message(COVISE_MESSAGE_UI, ss.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(msg2);
            delete msg2;
        }
    }
}

string CTRLHandler::handleBrowserPath(const string &name, const string &nr, const string &host, const string &oldhost,
                                      const string &parameterName, const string &parameterValue)
{

    string value = parameterValue;

    if (host != oldhost)
    {
        // get Datamanager for old host
        DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(oldhost);
        string path = tmp_data->get_DM()->covise_path;
        string sep = path.substr(0, 1);
        path.erase(0, 1);
        vector<string> pathList = splitString(path, sep);

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

    ostringstream os;
    os << "FILE_LOOKUP\n" << host << "\nuser\n" << name << "\n" << nr << "\n"
       << parameterName << "\n"
                           "\n" << value;

    Message *msg2 = new Message(COVISE_MESSAGE_UI, os.str());

    // send request for COVISE_PATH to new datamanager on new host
    DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(host);
    if (tmp_data != NULL)
    {
        AppModule *module = tmp_data->get_DM();
        if (module != NULL)
        {
            module->send_msg(msg2);
            delete msg2;

            Message *rmsg = new Message;
            module->recv_msg(rmsg);
            if (rmsg->type == COVISE_MESSAGE_UI)
            {
                vector<string> revList = splitString(rmsg->data.data(), "\n");
                value = revList[7];
            }
            delete rmsg;
        }
    }
    return value;
}

vector<string> CTRLHandler::splitString(string text, const string &sep)
{
    vector<string> list;
    int search = 0, first = 0, len = 0;
    ;

    while (1)
    {
        first = (int)text.find(sep, search);
        if (first != -1)
        {
            len = first - search;
            string sub = text.substr(search, len);
            if (sub.empty())
            {
                list.push_back("");
            }
            else
            {
                if (sub[0] != '#')
                    list.push_back(sub);
            }
            search = first + 1;
        }

        else
        {
            if (search < text.length())
            {
                len = (int)text.length() - search;
                string sub = text.substr(search, len);
                if (!sub.empty() && (sub[0] != '#'))
                    list.push_back(sub);
            }
            return list;
        }
    }
}

bool CTRLHandler::recreate(string content, readMode mode)
{
    // send message to UI that loading of a map has been started
    Message *tmpmsg;
    if (mode == NETWORKMAP)
        tmpmsg = new Message(COVISE_MESSAGE_UI, "START_READING\n" + m_globalFilename);
    else
        tmpmsg = new Message(COVISE_MESSAGE_UI, "START_READING\n");
    CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
    delete tmpmsg;

    m_writeUndoBuffer = false;

    DM_data *dm_local = CTRLGlobal::getInstance()->dataManagerList->get_local();
    string localname = dm_local->get_hostname();
    string localuser = dm_local->get_user();

    vector<string> mmodList; // list of obsolete modules

    int iel = 0;
    vector<string> list = splitString(content, "\n");

    // read host information
    istringstream s3(list[iel]);
    iel++;
    int nhosts;
    s3 >> nhosts;

    bool allhosts = true;
    bool addPartner = false;

    // add hosts if not already added
    for (int i = 0; i < nhosts; i++)
    {
        string hostname = list[iel];
        iel++;
        string username = list[iel];
        iel++;
        if (hostname == "LOCAL")
            hostname = localname;

        vector<string> token = splitString(username, " ");
        if (token[0] == "LUSER")
            username = localuser;

        if (token.size() == 2)
        {
            if (token[1] == "Partner")
                addPartner = true;
        }

        rhost *tmp_host = CTRLGlobal::getInstance()->hostList->get(hostname);

        if (tmp_host == NULL)
        {
            string data;
            if (addPartner)
                data = "ADDPARTNER\n" + hostname + "\n" + username + "\nPassword\n";
            else
                data = "ADDHOST\n" + hostname + "\n" + username + "\nPassword\n";
            Message *msg2 = new Message(COVISE_MESSAGE_UI, data);
            userinterface *tmp_ui = CTRLGlobal::getInstance()->userinterfaceList->get_master();
            tmp_ui->send(msg2);
            delete msg2;

            allhosts = false;
        }
    }

    if (!allhosts)
    {
        Message *tmpmsg = new Message(COVISE_MESSAGE_UI, "END_READING\nfalse");
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
        delete tmpmsg;
        return false;
    }

    //  read all modules
    //  no of modules
    istringstream s1(list[iel]);
    iel++;
    int no;
    s1 >> no;

    // craete list that contains module name, old number and new number
    vector<string> mnames;
    vector<string> oldnr;
    vector<string> newnr;

    string selectionBuffer;
    int ready = 0;

    //  start module if exist and tell it to the uifs
    for (int ll = 0; ll < no; ll++)
    {
        string name = list[iel];
        iel++;
        string nr = list[iel];
        iel++;
        string host = list[iel];
        iel++;
        if (host == "LOCAL")
            host = localname;
        iel++; // category, not used

        string title = list[iel];
        iel++;
        if (title.find("TITLE=", 0, 6) != -1)
            title.erase(0, 6);

        istringstream p1(list[iel]);
        iel++;
        istringstream p2(list[iel]);
        iel++;

        int posx, posy;
        p1 >> posx;
        p2 >> posy;

        bool modExist = checkModule(name, host);
        string nrnew;
        string current = nr;
        if (modExist)
        {
            if (mode == CLIPBOARD)
            {
                current = "-1";
                posx = posx + 10;
                posy = posy + 10;
                m_writeUndoBuffer = true;
            }
            int id = initModuleNode(name, current, host, posx, posy, title, 2, Start::Normal);
            if (id != -1)
                nrnew = (CTRLGlobal::getInstance()->netList->get(id))->get_nr();
        }

        else
            mmodList.push_back(name);

        // wrap input ports
        istringstream inp(list[iel]);
        iel++;
        int ninp;
        inp >> ninp;
        iel = iel + ninp * 5;

        // wrap output ports
        istringstream out(list[iel]);
        iel++;
        int nout;
        out >> nout;
        iel = iel + nout * 5;

        // update  parameter
        istringstream para(list[iel]);
        iel++;
        int npara;
        para >> npara;

        for (int l1 = 0; l1 < npara; l1++)
        {
            string paramname = list[iel];
            iel++; //  name
            string type = list[iel];
            iel++; //  type
            iel++; //  unused description
            string value = list[iel];
            iel++; //  value
            // not only set which choice during network load but also choice values
            // this is not ok if choices changed and an old net is loaded but in cases where
            // choices are updated during file load
            // one option could be to only set these Values if  Choice values start with NONE or ---
            // but this should be decided in the module and Mapeditor if(type == "Choice")
            //{
            //   vector<string> choices = splitString(value, " ");
            //   value = choices[0];
            //}
            iel++; //  unused IMM
            string apptype = list[iel];
            iel++; //  appearance type
            if (modExist)
                sendNewParam(name, nrnew, host, paramname, type, value, apptype, host, mode == NETWORKMAP);
        }

        // wrap output parameter
        istringstream pout(list[iel]);
        iel++;
        int npout;
        pout >> npout;
        iel = iel + npout * 5;

        mnames.push_back(name);
        oldnr.push_back(nr);
        newnr.push_back(nrnew);

        // when reading from a clipboard send a select for all pasted modules
        if (mode == CLIPBOARD && modExist)
        {
            ready++;
            selectionBuffer = selectionBuffer + name + "\n" + nrnew + "\n" + host + "\n";
        }
    }

    // send message for node selction
    if (mode == CLIPBOARD)
    {
        ostringstream os;
        os << ready;
        selectionBuffer = "SELECT_CLIPBOARD\n" + os.str() + "\n" + selectionBuffer;
        tmpmsg = new Message(COVISE_MESSAGE_UI, selectionBuffer);
        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
        delete tmpmsg;
    }

    //  no of connections
    istringstream s2(list[iel]);
    iel++;
    int nc;
    s2 >> nc;

    // loop over all connections
    // check if current module number has changed
    for (int ll = 0; ll < nc; ll++)
    {
        string fname = list[iel];
        iel++;
        string fnr = list[iel];
        iel++;
        for (int k = 0; k < no; k++)
        {
            if (mnames[k] == fname && oldnr[k] == fnr)
            {
                fnr = newnr[k];
                break;
            }
        }
        string fhost = list[iel];
        iel++;
        if (fhost == "LOCAL")
            fhost = localname;
        string fport = list[iel];
        iel++;
        iel++; // unused data name

        string tname = list[iel];
        iel++;
        string tnr = list[iel];
        iel++;
        for (int k = 0; k < no; k++)
        {
            if (mnames[k] == tname && oldnr[k] == tnr)
            {
                tnr = newnr[k];
                break;
            }
        }
        string thost = list[iel];
        iel++;
        if (thost == "LOCAL")
            thost = localname;
        string tport = list[iel];
        iel++;

        // check if connection is made to a non existing module
        bool doConnect = true;
        for (int i = 0; i < mmodList.size(); i++)
        {
            if (mmodList[i] == fname || mmodList[i] == tname)
            {
                doConnect = false;
                break;
            }
        }

        if (doConnect)
            makeConnection(fname, fnr, fhost, fport, tname, tnr, thost, tport);
    }

    if (mode == NETWORKMAP)
        tmpmsg = new Message(COVISE_MESSAGE_UI, "END_READING\ntrue");
    else
        tmpmsg = new Message(COVISE_MESSAGE_UI, "END_READING\nfalse");
    CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
    delete tmpmsg;

    mmodList.clear();
    m_writeUndoBuffer = true;

    return true;
}

void covise::CTRLHandler::removeConnection(covise::Connection * conn)
{
    //implement here
}

bool CTRLHandler::checkModule(const string &modname, const string &modhost)
{

    module *tmpmod = NULL;
    bool modfound = false;

    CTRLGlobal::getInstance()->moduleList->reset();
    while ((modfound == false) && ((tmpmod = CTRLGlobal::getInstance()->moduleList->next()) != NULL))
    {
        if (tmpmod->get_name() == modname && tmpmod->get_host() == modhost)
            modfound = true;
    }

    if (!modfound)
    {
        string data = "Error in load. Module " + modname + " on host " + modhost + " is not available. \n";

        Message *msg = new Message(COVISE_MESSAGE_UI, data);
        userinterface *ui = CTRLGlobal::getInstance()->userinterfaceList->get_master();
        ui->send(msg);
        delete msg;
    }

    return modfound;
}

string CTRLHandler::writeClipboard(const string &keyword, vector<net_module *> moduleList, bool all)
{
    // prepare buffer for return to UIF
    string buffer = keyword + "\n";

    //  get module infos (descrption & parameter values analog map saving)
    DM_data *dm_local = CTRLGlobal::getInstance()->dataManagerList->get_local();
    string localname = dm_local->get_hostname();
    string localuser = dm_local->get_user();

    // store hosts
    string hostnames = CTRLGlobal::getInstance()->hostList->get_hosts(localname, localuser);
    buffer = buffer + hostnames;

    // store modules
    ostringstream temp;
    temp << moduleList.size();
    buffer = buffer + temp.str() + "\n";

    net_module *tmp_mod;
    for (int ll = 0; ll < moduleList.size(); ll++)
    {
        net_module *from_mod = moduleList[ll];
        string erg = from_mod->get_parameter("input", false);

        //  store the current parameters of the modules to be moved/copied
        CTRLGlobal::getInstance()->netList->reset();
        while ((tmp_mod = CTRLGlobal::getInstance()->netList->next()) != NULL)
        {
            bool test = tmp_mod->test_copy();
            if (test != true && from_mod == tmp_mod)
            {
                string erg = tmp_mod->get_module(localname, localuser, true);
                buffer = buffer + erg;
            }
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
                net_module *tmp_mod = moduleList[k2];
                if (from_name[kk] == tmp_mod->get_name() && from_inst[kk] == tmp_mod->get_nr() && from_host[kk] == tmp_mod->get_host())
                {
                    ostringstream erg;
                    erg << from_name[kk] << "\n" << from_inst[kk] << "\n" << from_host[kk] << "\n" << from_port[kk] << "\n\n";
                    erg << to_name[kk] << "\n" << to_inst[kk] << "\n" << to_host[kk] << "\n" << to_port[kk] << "\n";
                    vector<string>::iterator result;
                    result = find(connList.begin(), connList.end(), erg.str());
                    if (result == connList.end())
                    {
                        connList.push_back(erg.str());
                        nconn++;
                    }
                    break;
                }

                if (to_name[kk] == tmp_mod->get_name() && to_inst[kk] == tmp_mod->get_nr() && to_host[kk] == tmp_mod->get_host())
                {
                    ostringstream erg;
                    erg << from_name[kk] << "\n" << from_inst[kk] << "\n" << from_host[kk] << "\n" << from_port[kk] << "\n\n";
                    erg << to_name[kk] << "\n" << to_inst[kk] << "\n" << to_host[kk] << "\n" << to_port[kk] << "\n";
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
                net_module *tmp_mod = tmp_obj->get_from()->get_mod();
                if (tmp_mod && tmp_mod == moduleList[ll])
                {
                    int i = 0;
                    ostringstream res_str, from_str;

                    from_str << tmp_mod->get_name() << "\n" << tmp_mod->get_nr() << "\n";
                    if (tmp_mod->get_host() == localname)
                        from_str << "LOCAL";
                    else
                        from_str << tmp_mod->get_host();

                    from_str << "\n" << tmp_obj->get_from()->get_intf() << "\n";

                    // get all to-connections inside the group
                    tmp_obj->get_to()->reset();
                    obj_conn *conn_tmp;
                    while ((conn_tmp = tmp_obj->get_to()->next()) != NULL)
                    {
                        net_module *to_mod = conn_tmp->get_mod();
                        vector<net_module *>::iterator result;
                        result = find(moduleList.begin(), moduleList.end(), to_mod);
                        if (result != moduleList.end())
                        {
                            i++;
                            res_str << from_str.str() << "\n" << to_mod->get_name() << "\n" << to_mod->get_nr() << "\n";
                            if (to_mod->get_host() == localname)
                                res_str << "LOCAL";
                            else
                                res_str << to_mod->get_host();
                            res_str << "\n" << conn_tmp->get_mod_intf() << "\n";
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
        vector<string> token = splitString(connections, "\n");
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
            token[itl];
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

//!
//! handle messages from the SSL-Daemon
//!
void CTRLHandler::handleSSLDaemon(Message *msg)
{

    if (strncmp(msg->data.data(), "join", 4) == 0)
    {
        const char *hname = msg->data.data() + 5;
        const char *passwd = "<empty>";
        const char *user_id = "<empty>";
        const char *c = strchr(hname, ':');

        if (c)
        {
            char *tmp = new char[strlen(c) + 1];
            strcpy(tmp, c);
            *tmp = '\0';
            size_t retval;
            retval = sscanf(tmp + 1, "%d", &m_accessGridDaemon->DaemonPort);
            if (retval != 1)
            {
                cerr << "main: sscanf failed" << endl;
                exit(-1);
            }
            delete[] tmp;
        }

        // set exectype to remote daemon which is 6
        Config->set_exectype(hname, "7");
        string hostname(hname);
        if (CTRLGlobal::getInstance()->userinterfaceList->add_partner(m_globalFilename, hostname, user_id, passwd, m_scriptName))
        {
            if (m_globalLoadReady == false)
            {
                m_globalLoadReady = CTRLGlobal::getInstance()->netList->load_config(m_globalFilename);
            }
        }

        else
        {
            char *msg_tmp = new char[200];
            sprintf(msg_tmp, "ADDPARTNER_FAILED\n%s\n%s\nPassword\n", hostname.c_str(), user_id);

            Message f_msg{ COVISE_MESSAGE_UI , DataHandle{msg_tmp, strlen(msg_tmp) + 1} };
            CTRLGlobal::getInstance()->userinterfaceList->send_master(&f_msg);
        }
    }
}

void CTRLHandler::sendCollaborativeState()
{

    int size = 0;
    string buffer;
    DM_data *p_data;

    CTRLGlobal::getInstance()->dataManagerList->reset();
    while ((p_data = CTRLGlobal::getInstance()->dataManagerList->next()) != NULL)
    {
        string chost = p_data->get_hostname();
        string cuser = p_data->get_user();
        rhost *host = CTRLGlobal::getInstance()->hostList->get(chost, cuser);
        buffer = buffer + host->get_type() + "\n" + chost + "\n" + cuser + "\n";
        userinterface *ui = CTRLGlobal::getInstance()->userinterfaceList->get(chost, cuser);
        if (ui != NULL)
            buffer = buffer + ui->get_status() + "\n";
        else
            buffer = buffer + "NONE\n";
        size++;
    }

    ostringstream s1;
    s1 << size;

    string text = "COLLABORATIVE_STATE\n" + s1.str() + "\n" + buffer;
    Message *msg = new Message(COVISE_MESSAGE_UI, text);
    CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
    delete msg;
}
