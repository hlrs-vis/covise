/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <process.h>
#else
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <net/route.h>
#include <arpa/inet.h>
#endif

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <util/unixcompat.h>
#include <net/concrete_messages.h>

#include "AccessGridDaemon.h"
#include "CTRLHandler.h"
#include "CTRLGlobal.h"
#include "control_process.h"
#include "control_coviseconfig.h"

#define STDIN_MAPPING

using namespace covise;

#if defined(__sgi) || defined(__alpha) || defined(_AIX) || defined(__APPLE__) || defined(__FreeBSD__)
extern "C" int rexec(char **ahost, int inport, char *user, char *passwd,
                     char *cmd, int *fd2p);
#endif

// LOCAL .....
AppModule *Controller::start_datamanager(const string &name)
{
    char chport[10];
    char chid[16];
    int port;

    module_count += 2;
    ServerConnection *conn = new ServerConnection(&port, module_count, CONTROLLER);
    conn->listen();
    if (!conn->is_connected())
        return NULL;

#ifdef _WIN32
    sprintf(chport, "%d", port);
    sprintf(chid, "%d", module_count);
    string win_cmd_line(name);
    win_cmd_line.append(" ");
    win_cmd_line.append(chport);
    win_cmd_line.append(" ");
    win_cmd_line.append(host->getAddress());
    win_cmd_line.append(" ");
    win_cmd_line.append(chid);

    cerr << "* Starting crb with: " << win_cmd_line.c_str() << endl;

    /*
   errno =  0;
   int ret =  spawnlp(P_NOWAIT, name.c_str(), name.c_str(),  chport, host->getAddress(), chid, NULL);
   if(ret <0)
   {
      cerr << "spawnlp returned " << ret  << endl;
      cerr << "errno " << errno  << endl;
      cerr << "spawnlp args " << name << " "<< chport << " " << host->getAddress() << " " << chid << endl;
      cerr << "PATH is not correct or too long" << endl;
   }
   */
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Start the child process.
    if (!CreateProcess(NULL, // No module name (use command line)
                       (LPSTR)win_cmd_line.c_str(), // Command line
                       NULL, // Process handle not inheritable
                       NULL, // Thread handle not inheritable
                       FALSE, // Set handle inheritance to FALSE
                       0, // No creation flags
                       NULL, // Use parent's environment block
                       NULL, // Use parent's starting directory
                       &si, // Pointer to STARTUPINFO structure
                       &pi) // Pointer to PROCESS_INFORMATION structure
        )
    {
        printf("Could not launch %s !\nCreateProcess failed (%d).\n", win_cmd_line.c_str(), GetLastError());
    }
    else
    {
        // Wait until child process exits.
        //WaitForSingleObject( pi.hProcess, INFINITE );
        // Close process and thread handles.
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

#else
    if (fork() == 0)
    {
        sprintf(chport, "%d", port);
        sprintf(chid, "%d", module_count);

        const char *covisedir = getenv("COVISEDIR");
        const char *archsuffix = getenv("ARCHSUFFIX");
        if (covisedir && archsuffix)
        {
            std::string crb(covisedir);
            crb += "/";
            crb += archsuffix;
            crb += "/bin/";
            crb += name.c_str();
            execl(crb.c_str(), name.c_str(), chport, "127.0.0.1", chid, NULL);
        }
        else
            execlp(name.c_str(), name.c_str(), chport, "127.0.0.1", chid, NULL);
        print_error(__LINE__, __FILE__, "exec of data manager failed");
        exit(1);
    }
    else
#endif

    {
        Host thisHost;
        if (conn->acceptOne(CTRLHandler::instance()->Config->gettimeout(thisHost)) < 0)
        {
            delete conn;
            cerr << "* timelimit in accept for crb exceeded!!" << endl;
            return NULL;
        }
        AppModule *mod = new AppModule(conn, module_count - 1, name, host);
        list_of_connections->add(conn);
        mod->set_peer(module_count - 1, DATAMANAGER);
        return mod;
    }
    return NULL;
}

void Controller::addConnection(Connection *conn)
{
    list_of_connections->add(conn);
}

// REXEC......
AppModule *Controller::start_datamanager(Host *rhost, const char *user, const char *passwd, const char *name)
{
    char remote_command[100];
    int port, ret;
    u_short execport;
    struct servent *se;
#ifdef STDIN_MAPPING
    Connection *tmp_conn;
#endif
    char *tmphost;

    module_count += 2;
    ServerConnection *conn = new ServerConnection(&port, module_count - 1, CONTROLLER);
    if (!conn->is_connected())
        return NULL;
    conn->listen();

    sprintf(remote_command, ""
                            "%s %d %s %d"
                            "",
            name, port, host->getAddress(), module_count - 1);
    if (!(se = getservbyname("exec", "tcp")))
    {
        fprintf(stderr, "%s: can't getservbyname()\n", "exec");
        print_error(__LINE__, __FILE__, "rexec of %s failed", remote_command);
        return NULL;
    }
    execport = se->s_port;
#if defined(__APPLE__) || defined(_WIN32) || defined(__FreeBSD__)
    (void)tmphost;
    (void)user;
    (void)passwd;
    print_comment(__LINE__, __FILE__, "rexec is not implemented on Mac OS X");
    ret = -1;
#else
    tmphost = new char[strlen(rhost->getAddress()) + 1];
    strcpy(tmphost, rhost->getAddress());
    ret = rexec(&tmphost, execport, user, passwd, remote_command, 0);
#endif
    if (ret == -1)
    {
        print_comment(__LINE__, __FILE__, "rexec of %s failed", remote_command);
        return NULL;
    }
#ifdef STDIN_MAPPING
    else
    {
        tmp_conn = new Connection(ret);
        tmp_conn->set_peer(module_count - 1, DATAMANAGER);
        list_of_connections->add(tmp_conn);
    }
#endif
    if (conn->acceptOne(CTRLHandler::instance()->Config->gettimeout(*rhost)) < 0)
    {
        delete conn;
        cerr << "* timelimit in accept for crb exceeded!!" << endl;
#ifdef STDIN_MAPPING
        list_of_connections->remove(tmp_conn);
        delete tmp_conn;
#endif
        return NULL;
    }
    AppModule *mod = new AppModule(conn, module_count - 1, name, rhost);
    list_of_connections->add(conn);
    mod->set_peer(module_count - 1, DATAMANAGER);
    return mod;
}

AppModule *Controller::start_datamanager(Host *rhost, const char *user, const char *name,
                                         int exec_type, const char *script_name)
{
    //std::cerr << "Controller::start_datamanager: name=" << name << ", rhost name=" <<  rhost->getName() << ", v4=" << rhost->get_ipv4() << std::endl;
    char chport[10];
    char chid[16];
    int port;
    char *dsp = CTRLHandler::instance()->Config->getDisplayIP(*rhost);

    CTRLGlobal *global = CTRLGlobal::getInstance();

    module_count += 2;
    ServerConnection *conn = new ServerConnection(&port, module_count - 1, CONTROLLER);
    if (!conn->is_connected())
        return NULL;
    conn->listen();
    if (exec_type == COVISE_MANUAL)
    {
        char text[1000];
        snprintf(text, sizeof(text), "please start \"%s %d %s %d\" on %s", name, port,
                 host->getAddress(), module_count, rhost->getName());
        Message *msg = new Message(COVISE_MESSAGE_COVISE_ERROR, text);
        global->userinterfaceList->send_master(msg);
        delete msg;
        fprintf(stderr, "please start \"%s %d %s %d\" on %s", name, port, host->getAddress(),
                module_count, rhost->getName());
    }

    else if (exec_type == COVISE_ACCESSGRID)
    {
        char start_string[200];
        sprintf(start_string, "startCrb %s %d %s %d %s\n", name, port, host->getAddress(), module_count - 1, dsp);
        // open connection to remote Accessgrid Daemon

        SimpleClientConnection *conn = new SimpleClientConnection(rhost, CTRLHandler::instance()->m_accessGridDaemon->DaemonPort, 0);
        if (!conn->is_connected()) // could not open server port
        {
            fprintf(stderr, "Could not connect to Accessgrid Daemon %s, port %d\n", rhost->getAddress(), port);
            fprintf(stdout, "start \" %s %d %s %d \" on %s\n", name, port, host->getAddress(),
                    module_count, rhost->getName());
            delete conn;
            conn = NULL;
        }
        else
        {
            conn->getSocket()->setNonBlocking(true);
            conn->getSocket()->write(start_string, (int)strlen(start_string));
        }
        delete conn;
    }
    else if (exec_type == COVISE_SSLDAEMON)
    {
#ifdef HAVE_OPENSSL
        int remotePort = coCoviseConfig::getInt("port", "System.CoviseDaemon.Server", 31090);

        //Create start string
        char start_string[200];
        sprintf(start_string, "startCrb %s %d %s %d %s\n", name, port, host->getAddress(), module_count - 1, dsp);

        //Use a SSLConnection to start CRB
        //Do SSL handling
        SSLClientConnection *sslconn = new SSLClientConnection(rhost, remotePort, NULL, NULL);

        //Attach SSL to socket
        if (sslconn->AttachSSLToSocket(sslconn->getSocket()) == 0)
        {
            global->userinterfaceList->sendWarning("SSLConnection to RemoteHost failed! Local SSL attach failed");
            delete sslconn;
            sslconn = NULL;
            return NULL;
        }

        //Connect SSL, do handshake
        if (sslconn->connect() < 0)
        {
            global->userinterfaceList->sendWarning("SSLConnection to RemoteHost failed! Connect failed!");
            delete sslconn;
            sslconn = NULL;
            return NULL;
        }

        //Check for connection
        if (!sslconn->is_connected())
        {
            fprintf(stderr, "Could not connect to SSLDaemon %s, port %d\n", rhost->getAddress(), remotePort);
            fprintf(stdout, "start \" %s %d %s %d \" on %s\n", name, remotePort, host->getAddress(),
                    module_count, rhost->getName());
            delete sslconn;
            sslconn = NULL;
        }
        else
        {
            // send command string
            sslconn->send(start_string, (int)strlen(start_string));

            //listen for ACK
            bool bWaitingForAck = true;
            int timeout = 65000;
            while (bWaitingForAck || --timeout <= 0)
            {
                sslconn->check_for_input((float)0.1);
                const char *result = sslconn->readLine();
                //Check for result == NULL here to avoid crashes
                if (result && strncmp(result, "ACK\n", 4))
                {
                    global->userinterfaceList->sendWarning("RemoteHost confirmed command!");
                    bWaitingForAck = false;
                }
            }
            if (bWaitingForAck)
            {
                global->userinterfaceList->sendWarning("Command timed out!");
            }
        }

        //Cleanup everything
        delete sslconn;
        sslconn = NULL;
#else
        fprintf(stderr, "no SSL support, cannot handle exec type \"SSL daemon\"");
#endif
    }

    else if (exec_type == COVISE_SCRIPT)
    {
        char start_string[200];
        sprintf(start_string, "%s %s %d %s %d", script_name, name, port, host->getAddress(), module_count - 1);
        int retval;
        retval = system(start_string);
        if (retval == -1)
        {
            std::cerr << "Controller::start_datamanager: system failed" << std::endl;
            return NULL;
        }
    }

    else if (exec_type == COVISE_REMOTE_DAEMON)
    {

        Host *rdHost = rhost;

        int remport = coCoviseConfig::getInt("port", "System.RemoteDaemon.Server", 31090);

        // verify myID value
        SimpleClientConnection *clientConn = new SimpleClientConnection(rdHost, remport);
        if (!clientConn)
        {
            cerr << "Creation of ClientConnection failed!" << endl;
            return NULL;
        }
        else
        {
            cerr << "ClientConnection created!" << endl;
        }
        if (!(clientConn->is_connected()))
        {
            cerr << "Connection to RemoteDaemon on " << rhost->getAddress() << " failed!" << endl;
            return NULL;
        }
        else
        {
            cerr << "Connection to RemoteDaemon on " << rhost->getAddress() << " established!" << endl;
        }

        char start_string[2000];
        sprintf(start_string, "startFEN %s %d %s %d\n", name, port, host->getAddress(), module_count);
        clientConn->getSocket()->write(start_string, (int)strlen(start_string));

        cerr << "Message sent!" << endl;
        cerr << "Closing connection objects!" << endl;

        delete clientConn;

        cerr << "Leaving Start-Method of coVRSlave " << endl;
    }
    else
    {
        sprintf(chport, "%d", port);
        sprintf(chid, "%d", module_count - 1);

#ifdef _WIN32
        switch (exec_type)
        {
        case COVISE_SSH:
            if (spawnlp(P_NOWAIT, "ssh", "ssh", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL) == -1
                && (errno == ENOENT || errno == ENOEXEC))
            {
                spawnlp(P_NOWAIT, "ssh2", "ssh2", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL);
            }
            break;
        case COVISE_RSH:
            //fclose(stdin);
            spawnlp(P_NOWAIT, "rsh", "rsh", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL);
            break;
        case COVISE_NQS:
            spawnlp(P_NOWAIT, "qsub", "qsub", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL);
            break;
        case COVISE_GLOBUS_GRAM:
            cerr << "globus gram support NYI" << endl;
            break;
        case COVISE_REATTACH:
            cerr << "reattach support NYI" << endl;
            break;
        }
#else
        if (fork() == 0)
        {
            if (user && strlen(user) > 0)
            {
                switch (exec_type)
                {
                case COVISE_SSH:
                    execlp("ssh", "ssh", "-f", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL);
                    break;
                case COVISE_RSH:
//fclose(stdin);
#ifdef __hpux
                    execlp("remsh", "remsh", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL);
#else
                    execlp("rsh", "rsh", rhost->getAddress(), "-l", user, "-n", name, chport, host->getAddress(), chid, dsp, NULL);
#endif
                    break;
                case COVISE_NQS:
                    execlp("qsub", "qsub", rhost->getAddress(), "-l", user, name, chport, host->getAddress(), chid, dsp, NULL);
                    break;
                }
            }
            else
            {
                switch (exec_type)
                {
                case COVISE_SSH:
                    execlp("ssh", "ssh", "-f", rhost->getAddress(), name, chport, host->getAddress(), chid, dsp, NULL);
                    break;
                case COVISE_RSH:
//fclose(stdin);
#ifdef __hpux
                    execlp("remsh", "remsh", rhost->getAddress(), name, chport, host->getAddress(), chid, dsp, i NULL);
#else
                    execlp("rsh", "rsh", rhost->getAddress(), "-n", name, chport, host->getAddress(), chid, dsp, NULL);
#endif
                    break;
                case COVISE_NQS:
                    execlp("qsub", "qsub", rhost->getAddress(), name, chport, host->getAddress(), chid, dsp, NULL);
                    break;
                }
            }
            print_error(__LINE__, __FILE__, "exec of remote command failed");
            _exit(1);
        }
#endif
    }
    if (exec_type == COVISE_MANUAL)
        conn->acceptOne(-1);
    else
    {
        if (conn->acceptOne(CTRLHandler::instance()->Config->gettimeout(*rhost)) < 0)
        {
            cerr << "* timelimit in accept for crb exceeded!!" << endl;
            delete conn;
            return NULL;
        }
    }
    AppModule *mod = new AppModule(conn, module_count - 1, name, rhost);
    list_of_connections->add(conn);
    mod->set_peer(module_count - 1, DATAMANAGER);
    return mod;
}

AppModule* Controller::start_applicationmodule(sender_type peer_type, const char *name, AppModule *dmod, const char *instance, ExecFlag flags, const char *category, const std::vector<std::string> &params){
    module_count++;
    int port;
    ServerConnection *conn = new ServerConnection(&port, module_count, CONTROLLER);
    if (!conn->is_connected())
        return nullptr;
    conn->listen();
    Host localhost("127.0.0.1");
    Host *h = host->getAddress() == dmod->get_host()->getAddress()
                  ? &localhost
                  : host;
    
    const char *displayIp = CTRLHandler::instance()->Config->getDisplayIP((*dmod->get_host()));
    CRB_EXEC crbExec{flags, name, port, h->getAddress(), module_count, instance, dmod->get_host()->getAddress(), dmod->get_host()->getName(), displayIp, category, CTRLHandler::instance()->vrbClientID(), params};
    
    int timeout = 0; // do not timeout
    if (flags != ExecFlag::Debug)
    {
        timeout = CTRLHandler::instance()->Config->gettimeout(*dmod->get_host());
    }
        // start renderer (OPENSG) inside the mapeditor
    auto hostname = dmod->get_host()->getAddress();
    auto mapeditor = CTRLGlobal::getInstance()->userinterfaceList->get(hostname);
    if (mapeditor && confirmIsRenderer(category, dmod))
        sendCoviseMessage(crbExec, *mapeditor);
    else
        sendCoviseMessage(crbExec, *dmod);

    if (conn->acceptOne(timeout) < 0)
    {
        cerr << "* timelimit in accept for module " << name << " exceeded!!" << endl;
        delete conn;
        return NULL;
    }
    AppModule *mod = new AppModule(conn, module_count, name, host);
    list_of_connections->add(conn);
    mod->set_peer(module_count, peer_type);
    return mod;
}


bool Controller::confirmIsRenderer(const char* cat, AppModule* dmod){
    userinterface *mapeditor = NULL;
    if (cat && strcmp("Renderer", cat) == 0)
    {
        string hostname = dmod->get_host()->getAddress();
        mapeditor = CTRLGlobal::getInstance()->userinterfaceList->get(hostname);

        if (mapeditor)
        {
            string text = "QUERY_IMBEDDED_RENDERER\n";
            text.append(name);
            text.append("\n");
            text.append(cat);
            Message *msg2 = new Message(COVISE_MESSAGE_UI, text);
            dmod->send(msg2);
            delete msg2;

            Message rmsg;
            dmod->recv_msg(&rmsg);
            if (rmsg.type == COVISE_MESSAGE_UI)
            {
                if (rmsg.data.data() && strcmp(rmsg.data.data(), "YES") == 0)
                    return true;
            }
        }
    }
    return false;
}

void Controller::get_shared_memory(AppModule *dmod)
{
    Message msg{ COVISE_MESSAGE_GET_SHM_KEY , DataHandle{} };

    print_comment(__LINE__, __FILE__, "in get_shared_memory");
    dmod->send(&msg);
    dmod->recv_msg(&msg);
    if (msg.type == COVISE_MESSAGE_GET_SHM_KEY)
    {
        print_comment(__LINE__, __FILE__, "GET_SHM_KEY: %d: %x, %d length: %d", *(int *)msg.data.data(),
                      ((int *)msg.data.data())[1], ((int *)msg.data.data())[2], msg.data.length());
        shm = new ShmAccess(msg.data.accessData(), 0);
    }
    // data of received message can be deleted
}

void Controller::handle_shm_msg(Message *msg)
{
    int tmpkey, size;

    if (msg->conn->get_sender_id() == 1)
    {
        tmpkey = ((int *)msg->data.data())[0];
        size = ((int *)msg->data.data())[1];
        shm->add_new_segment(tmpkey, size);
        print_comment(__LINE__, __FILE__, "new SharedMemory");
    }
}
