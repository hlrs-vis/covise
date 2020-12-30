/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _MSC_VER
#define POLARITY
#define _WIN32_DCOM
#include <winsock2.h>
#include <iphlpapi.h>

#include "covise.h"
#include <iostream>
using namespace std;
#include <iomanip>
#include <windows.h>
#include <WTypes.h>
#include <rpc.h>
#include <chstring.h>
#include <chstrarr.h>
#include <assert.h>
#include <comdef.h>
#include <wbemcli.h>

#include <winbase.h>
#include <atlbase.h>
#include <wbemidl.h>

#endif

#include "covise_process.h"
#include <util/unixcompat.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <net/concrete_messages.h>
#include <util/coLog.h>
#include <config/CoviseConfig.h>
#ifdef _WIN32
#include <io.h>
#else
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <net/route.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#endif

#ifdef _SX
#include <unistd.h>
extern "C" {
extern int lockf(int, int, long);
}
#endif

#ifdef _WIN32
#include <dmgr/dmgr.h>
#endif

#include "covise.h"

/*
 $Log:  $
Revision 1.7  1994/03/23  18:07:04  zrf30125
Modifications for multiple Shared Memory segments have been finished
(not yet for Cray)

Revision 1.6  94/02/18  14:04:45  zrfg0125
gethostname

Revision 1.5  93/12/02  16:00:35  zrfg0125
gethostname

Revision 1.4  93/11/15  13:57:32  zrfg0125
corrected errors

Revision 1.3  93/10/20  15:14:18  zrhk0125
socket cleanup improved

Revision 1.1  93/09/25  20:48:17  zrhk0125
Initial revision

*/

/***********************************************************************\
 **                                                                     **
 **   Process Class Routines                       Version: 1.0         **
 **                                                                     **
 **                                                                     **
 **   Description  : The infrastructure for general processes           **
 **                  is provided here.                                  **                                 **
 **                                                                     **
 **   Classes      : DataManagerProcess                                 **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#ifdef __alpha
extern "C" {
gethostname(char *, int);
}
#endif

namespace covise
{

#undef DEBUG
Process *Process::this_process = NULL;
FILE *COVISE_time_hdl = NULL;
static FILE *COVISE_debug_hdl = NULL;
//static FILE *COVISE_new_delete_hdl = NULL;
int COVISE_debug_level = 0;
static const char *COVISE_debug_filename = "covise.log";
//extern CoviseTime *covise_time;

// void inline print_nothing(...) {}

// void inline print_nothing(int,char*,void*) {}

/*int main(int argc, char* argv[])
{  //WMI Exec
    execProcessWMI("C:\\src\\uwewin\\covise\\win32\\bin\\Renderer\\COVER.EXE",NULL,NULL,NULL,NULL);
}
*/


#ifdef _MSC_VER
static void cleanSpaces(char *str)
{
    int current = 0;
    int ctr = 0;

    for (; str[ctr] != 0; ++ctr)
    {
        if (!isspace(str[ctr]))
        {
            break;
        }
    }

    for (; str[ctr] != 0; ++ctr)
    {
        str[current] = str[ctr];
        ++current;
    }

    str[current] = 0;

    if (current > 0)
    {
        while (isspace(str[current - 1]))
        {
            str[current - 1] = 0;
            --current;
        }
    }
}

int execProcessWMI(const char *commandLine, const char *wd, const char *inhost, const char *inuser, const char *password)
{ //WMI Exec

#if 1 // New version

    char *host = new char[strlen(inhost) + 1];
    char *user = new char[strlen(inuser) + 1];

    strcpy(host, inhost);
    strcpy(user, inuser);

    cleanSpaces(host);
    cleanSpaces(user);

    cerr << "execProcessWMI: info: starting '"
         << commandLine << "' on '" << host << "' using user '"
         << user << "'" << endl;

    CoInitialize(0);

    HRESULT hr = S_OK;

    bool setUser = (strlen(user) != 0);

    CComPtr<IWbemLocator> spLoc;
    CComPtr<IWbemClassObject> remoteClass;
    CComPtr<IWbemClassObject> remoteMethod;
    CComPtr<IWbemClassObject> remoteInstance;
    CComPtr<IWbemClassObject> outInstance;

    //cerr << "execProcessWMI: info: creating locator " << endl;

    hr = spLoc.CoCreateInstance(CLSID_WbemLocator);

    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: Could not create locator (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    //char * tmpString = new char[(strlen(host)) + 20];
    //sprintf(tmpString, "\\\\%s\\root\\CIMV2", host);
    //CComBSTR bstrNamespace(tmpString);
    //delete[] tmpString;

    ostringstream strNamespace;
    strNamespace << "\\\\" << host << "\\root\\CIMV2";
    string str = strNamespace.str();
    CComBSTR bstrNamespace(str.c_str());

    CComBSTR bstrUser(user);
    CComBSTR bstrPasswd(password);
    CComBSTR bstrClassPath("Win32_Process");
    CComBSTR bstrMethod("Create");
    CComPtr<IWbemServices> spServices;

    //cerr << "execProcessWMI: info: connecting" << endl;

    if (setUser)
    {
        hr = spLoc->ConnectServer(bstrNamespace, bstrUser, bstrPasswd,
                                  0, NULL, 0, 0, &spServices);
    }
    else
    {

        hr = spLoc->ConnectServer(bstrNamespace, NULL, NULL,
                                  0, NULL, 0, 0, &spServices);
    }

    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: Could not connect to server (errno "
             << hex << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    //cerr << "execProcessWMI: info: creating security context" << endl;

    if (setUser)
    {
        cerr << "execProcessWMI: info: setting new security context" << endl;
        SEC_WINNT_AUTH_IDENTITY ident;
        ident.User = (unsigned char *)user;
        ident.UserLength = (unsigned long)strlen(user);
        ident.Domain = (unsigned char *)host;
        ident.DomainLength = (unsigned long)strlen(host);
        ident.Password = (unsigned char *)password;
        ident.PasswordLength = (unsigned long)strlen(password);
        ident.Flags = SEC_WINNT_AUTH_IDENTITY_ANSI;

        hr = CoSetProxyBlanket(spServices,
                               RPC_C_AUTHN_DEFAULT, RPC_C_AUTHZ_NONE,
                               NULL, RPC_C_AUTHN_LEVEL_CALL,
                               RPC_C_IMP_LEVEL_IMPERSONATE, &ident, EOAC_NONE);
    }
    else
    {

        cerr << "execProcessWMI: info: using old security context" << endl;
        hr = CoSetProxyBlanket(spServices,
                               RPC_C_AUTHN_DEFAULT, RPC_C_AUTHZ_NONE,
                               NULL, RPC_C_AUTHN_LEVEL_CALL,
                               RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE);
    }

    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: Creation of proxy blanket failed (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    //cerr << "execProcessWMI: info: getting remote process handles" << endl;

    hr = spServices->GetObject(bstrClassPath, 0, NULL, &remoteClass, 0);
    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: GetObject on Win32_Process failed (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    hr = remoteClass->GetMethod(bstrMethod, 0, &remoteMethod, NULL);
    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: GetMethod on Win32_Process::Create failed (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    hr = remoteMethod->SpawnInstance(0, &remoteInstance);
    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: SpawnInstance on Win32_Process::Create failed (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    CComBSTR bstrCommand = "CommandLine";
    CComBSTR bstrParam = commandLine;
    VARIANT var;
    var.vt = VT_BSTR;
    var.bstrVal = bstrParam;

    hr = remoteMethod->Put(bstrCommand, 0, &var, 0);
    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: Failed setting command line (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }

    //cerr << "execProcessWMI: info: executing" << endl;

    hr = spServices->ExecMethod(bstrClassPath, bstrMethod, 0, NULL,
                                remoteMethod, &outInstance, NULL);
    if (hr != S_OK)
    {
        cerr << "execProcessWMI: err: Could not execute remote process (errno "
             << hr << ")" << endl;
        delete[] host;
        delete[] user;
        return -1;
    }
    else
    {
        cerr << "execProcessWMI: info: WMI call successful" << endl;
        delete[] host;
        delete[] user;
        return 0;
    }

#else // Old Version
    IWbemLocator *pLocator = NULL;
    IWbemServices *pNamespace = 0;
    IWbemClassObject *pClass = NULL;
    IWbemClassObject *pOutInst = NULL;
    IWbemClassObject *pInClass = NULL;
    IWbemClassObject *pInInst = NULL;
    BSTR path;
    if (host)
    {
        int bufSize = strlen(host) + 199;
        char *tmp = new char[bufSize];
        sprintf(tmp, "\\\\%s\\root\\cimv2");
        wchar_t *wbuf = new wchar_t[bufSize];
        mbstowcs(wbuf, tmp, bufSize);
        path = SysAllocString(wbuf);
        delete[] wbuf;
    }
    else
        path = SysAllocString(L"root\\cimv2");
    BSTR ClassPath = SysAllocString(L"Win32_Process");
    BSTR MethodName = SysAllocString(L"Create");
    BSTR User = NULL;
    if (user)
    {
        int bufSize = strlen(user);
        wchar_t *wbuf = new wchar_t[bufSize];
        mbstowcs(wbuf, user, bufSize);
        User = SysAllocString(wbuf);
        delete[] wbuf;
    }
    BSTR Password = NULL;
    if (password)
    {
        int bufSize = strlen(password);
        wchar_t *wbuf = new wchar_t[bufSize];
        mbstowcs(wbuf, password, bufSize);
        Password = SysAllocString(wbuf);
        delete[] wbuf;
    }
    //BSTR Text;

    // Initialize COM and connect to WMI.

    HRESULT hr = CoInitialize(0);
    hr = CoInitializeSecurity(NULL,
                              -1,
                              NULL,
                              NULL,
                              RPC_C_AUTHN_LEVEL_DEFAULT,
                              RPC_C_IMP_LEVEL_IMPERSONATE,
                              NULL,
                              EOAC_NONE,
                              NULL);
    if (hr != S_OK)
        return -1;
    hr = CoCreateInstance(__uuidof(WbemLocator), 0, CLSCTX_INPROC_SERVER,
                          __uuidof(IWbemLocator) /*IID_IWbemLocator*/, (LPVOID *)&pLocator);
    if (hr != S_OK)
        return -1;
    hr = pLocator->ConnectServer(path, User, Password, NULL, WBEM_FLAG_CONNECT_USE_MAX_WAIT, NULL,
                                 NULL, &pNamespace);
    if (hr != S_OK)
        return -1;

    // Get the class object for the method definition.

    hr = pNamespace->GetObject(ClassPath, 0, NULL, &pClass, NULL);
    if (hr != S_OK)
        return -1;

    // Get the input-argument class object and create an instance.

    hr = pClass->GetMethod(MethodName, 0, &pInClass, NULL);
    if (hr != S_OK)
        return -1;
    hr = pInClass->SpawnInstance(0, &pInInst);
    if (hr != S_OK)
        return -1;

    // Set the property.

    BSTR ArgName = SysAllocString(L"CommandLine");
    VARIANT var;
    var.vt = VT_BSTR;
    var.bstrVal = SysAllocString(L"C:\\src\\uwewin\\covise\\win32\\bin\\Renderer\\COVER.EXE");
    hr = pInInst->Put(ArgName, 0, &var, 0);
    VariantClear(&var);
    if (hr != S_OK)
        return -1;

    ArgName = SysAllocString(L"CurrentDirectory");
    var.vt = VT_BSTR;
    var.bstrVal = SysAllocString(L"C:\\src");
    hr = pInInst->Put(ArgName, 0, &var, 0);
    VariantClear(&var);
    if (hr != S_OK)
        return -1;

    // Call the method.

    hr = pNamespace->ExecMethod(ClassPath, MethodName, 0, NULL,
                                pInInst, &pOutInst, NULL);
    if (hr != S_OK)
        return -1;

    // Display the results. Note that the return value is in the
    // property "ReturnValue" and the returned string is in the
    // property "sOutArg".
    /*
       hr = pOutInst->GetObjectText(0, &Text);
      if(hr != S_OK)
         return -1;
       printf("\nThe object text is:\n%S", Text);*/

    // Free up resources.

    SysFreeString(path);
    SysFreeString(ClassPath);
    SysFreeString(MethodName);
    SysFreeString(ArgName);
    //  SysFreeString(Text);
    pClass->Release();
    pInInst->Release();
    pInClass->Release();
    pOutInst->Release();
    pLocator->Release();
    pNamespace->Release();
    CoUninitialize();
    return 0;
#endif
}

#else
int execProcessWMI(const char *, const char *, const char *, const char *, const char *)
{
    return -1;
}
#endif

#ifndef _WIN32
#ifdef COVISE_Signals
#ifndef NO_SIGNALS
//=====================================================================
//
//=====================================================================
void process_signal_handler(int sg, void *)
{
    // int cpid, status;
    switch (sg)
    {
    case SIGPIPE:
        break;
    default:
        delete Process::this_process;
        sleep(1);
        exit(EXIT_FAILURE);
        break;
    }
}
#endif
#endif
#endif
}

using namespace covise;

Process::Process(const char *n, int i, sender_type st)
{
    host = NULL;
    hostid = 0;
    name = n;
    send_type = st;
    id = i;
    init_env();
    covise_hostname = NULL;

#ifdef __linux__
    signal(SIGPIPE, SIG_IGN);
#endif

#ifndef _WIN32
#ifdef COVISE_Signals
#ifndef NO_SIGNALS
    sig_handler.addSignal(SIGPIPE, (void *)process_signal_handler, NULL);
#endif
#endif
#endif
    host = get_covise_host();
    list_of_connections = new ConnectionList;
    msg_queue = new List<Message>();
    //    init_statics();
    this_process = this;
    //    covise_time->mark(__LINE__, "new Process");
}

Process::Process(const char *n, int i, sender_type st, int port)
{
    hostid = 0;
    name = n;
    send_type = st;
    id = i;
    init_env();
    covise_hostname = NULL;

#ifdef __linux__
    signal(SIGPIPE, SIG_IGN);
#endif

#ifndef _WIN32
#ifdef COVISE_Signals
#ifndef NO_SIGNALS
    sig_handler.addSignal(SIGPIPE, (void *)process_signal_handler, NULL);
#endif
#endif
#endif
    host = get_covise_host();
    ServerConnection *open_sock = new ServerConnection(port, id, send_type);
    list_of_connections = new ConnectionList(open_sock);
    msg_queue = new List<Message>();
    //    init_statics();
    this_process = this;
    //    covise_time->mark(__LINE__, "new Process");
}

Process::Process(const char *n, int arc, char *arv[], sender_type st)
{
    auto crbExec = covise::getExecFromCmdArgs(arc, arv);
    if (crbExec.displayIp)
    {
        setenv("DISPLAY", crbExec.displayIp, true);
    }

    id = crbExec.moduleCount;
    name = n;
    send_type = st;
    init_env();
    covise_hostname = crbExec.moduleIp;
    host = new Host(crbExec.moduleHostName);
//char *instance = arv[4];

#ifdef __linux__
    signal(SIGPIPE, SIG_IGN);
#endif

#ifdef COVISE_Signals
#ifndef NO_SIGNALS
    sig_handler.addSignal(SIGPIPE, (void *)process_signal_handler, NULL);
#endif
#endif
    list_of_connections = new ConnectionList;
    msg_queue = new List<Message>();
    //    init_statics();
    this_process = this;
    //   covise_time->mark(__LINE__, "new Process");
}

void Process::init_env()
{
    char *COVISE_time_filename;
    int stp;
    char *tmp_env;
#ifndef _WIN32
    char tmp_fname[100];
    snprintf(tmp_fname, sizeof(tmp_fname), "/tmp/kill_ids_%d", getuid());
    FILE *hdl = fopen(tmp_fname, "a+");
    if (hdl)
    {
        fprintf(hdl, "%d\n", getpid());
        fclose(hdl);
    }
#endif
    COVISE_time_hdl = NULL;
    COVISE_debug_hdl = NULL;

    char *debug_filename = new char[strlen(name) + 40];
    COVISE_debug_filename = debug_filename;
    sprintf(debug_filename, "COVISE_%d_%s", id, name);
    tmp_env = getenv("COVISE_DEBUG_ALL");
    COVISE_debug_level = 0;
    if (tmp_env)
    {
        if (strcmp(tmp_env, "OFF") == 0)
            COVISE_debug_level = 0;
        else
            COVISE_debug_level = atoi(tmp_env);
    }
    else
    {
        char *env_test = new char[strlen(name) + 40];
        sprintf(env_test, "COVISE_DEBUG_%s", name);
        tmp_env = getenv(env_test);
        if (tmp_env)
            COVISE_debug_level = atoi(tmp_env);
        delete[] env_test;
    }

    tmp_env = getenv("COVISE_TIME_ALL");
    if (tmp_env && strcmp(tmp_env, "ON") == 0)
    {
        COVISE_time_filename = new char[strlen(name) + 40];
        sprintf(COVISE_time_filename, "COVISE_TIME_%d_%s", id, name);
        COVISE_time_hdl = fopen(COVISE_time_filename, "w");
    }
    else
    {
        char *env_test = new char[strlen(name) + 40];
        sprintf(env_test, "COVISE_TIME_%s", name);
        tmp_env = getenv(env_test);
        delete[] env_test;
        if (tmp_env && strcmp(tmp_env, "ON") == 0)
        {
            COVISE_time_filename = new char[strlen(name) + 40];
            sprintf(COVISE_time_filename, "COVISE_TIME_%d_%s", id, name);
            COVISE_time_hdl = fopen(COVISE_time_filename, "w");
        }
    }

    tmp_env = getenv("COVISE_NEW_DELETE");
    if (tmp_env && strcmp(tmp_env, "ON") == 0)
    {
        COVISE_debug_hdl = fopen(COVISE_debug_filename, "w");
        //	COVISE_new_delete_hdl = COVISE_debug_hdl;
    }

    stp = 31000;
    tmp_env = getenv("COVISE_PORT");
    if (tmp_env)
    {
        stp = atoi(tmp_env);
    }
    else
    {
        stp = coCoviseConfig::getInt("covisePort", "System.Network", stp);
    }
    Socket::set_start_port(stp);
}

Process::~Process()
{
    static int first = 1;
#ifndef _WIN32
    int pid, i, lock_hdl = -1, ende;
    char tmp_fname[100];
    //char tmp_str[255];
    FILE *hdl;
    int ids[100];
#endif
    if (!first)
        return;
    else
        first = 0;
    print_comment(__LINE__, __FILE__, "in ~Process");
#ifndef _WIN32
    delete list_of_connections;
    pid = getpid();
    sprintf(tmp_fname, "/tmp/kill_ids_lock_%d", getuid());
    lock_hdl = open(tmp_fname, O_RDWR | O_CREAT, 0644);
    //    cerr << "lock_hdl: " << lock_hdl << endl;
    if (lock_hdl == -1)
    {
        print_comment(__LINE__, __FILE__, "can't create Process lockfile");
        perror("can't create Process lockfile");
    }
    else
    {
        if (lockf(lock_hdl, F_LOCK, 0) == -1)
        {
            print_comment(__LINE__, __FILE__, "can't lock Process lockfile");
            perror("can't lock Process lockfile");
        }
        else
        {
            //	cerr << "Process " << getpid() << " locking\n";
        }
    }
    sprintf(tmp_fname, "/tmp/kill_ids_%d", getuid());
    //    print_comment(__LINE__, __FILE__, "reading file");
    hdl = fopen(tmp_fname, "rw");
    i = ende = 0;
    while (i < 100 && !ende)
    {
        ende = (fscanf(hdl, "%d\n", &ids[i]) != 1);
        //	sprintf(tmp_str, "%2d: %d", i, ids[i]);
        //	print_comment(__LINE__, __FILE__, tmp_str);
        if (!ende && ids[i] != pid)
            i++;
    }

    for (int j = 0; j < i; j++)
    {
        fprintf(hdl, "%d\n", ids[j]);
        //	sprintf(tmp_str, "%2d: %d", j, ids[j]);
        //	print_comment(__LINE__, __FILE__, tmp_str);
    }
    fclose(hdl);
    //    cerr << "Process " << getpid() << " unlocking\n";
    if (lock_hdl != -1)
    {
        if (lockf(lock_hdl, F_ULOCK, 0) == -1)
        {
            print_comment(__LINE__, __FILE__, "can't unlock Process lockfile");
            perror("can't unlock Process lockfile");
        }

        close(lock_hdl);
    }
#endif
}

Message *Process::wait_for_msg()
{
    Message *msg;
    Connection *conn;
#ifdef DEBUG
    char tmp_str[255];
#endif

    msg_queue->reset();
    msg = msg_queue->next();
    if (msg)
    {
        msg_queue->remove(msg);
#ifdef DEBUG
        sprintf(tmp_str, "msg %s removed from queue", covise_msg_types_array[msg->type]);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        return msg;
    }
    else
    {
        if (list_of_connections->count() == 0)
        {
            msg = new Message;
            msg->type = COVISE_MESSAGE_SOCKET_CLOSED;
            return msg; // avoid error message, if connection list is empty
        }
        conn = list_of_connections->wait_for_input();
        msg = new Message;
        //	printf("new Message in wait_for_msg: %x\n",msg);
        conn->recv_msg(msg);
        switch (msg->type)
        {
        case COVISE_MESSAGE_SOCKET_CLOSED:
        case COVISE_MESSAGE_CLOSE_SOCKET:
        case COVISE_MESSAGE_EMPTY:
        case COVISE_MESSAGE_STDINOUT_EMPTY:
            list_of_connections->remove(conn);
            print_comment(__LINE__, __FILE__, "Socket Closed");
            //delete conn;
            return msg;
        //break;
        case COVISE_MESSAGE_NEW_SDS:
            handle_shm_msg(msg);
            delete msg;
            return wait_for_msg();
        //break;
        default:
            break;
        }
        return msg;
    }
}

Message *Process::check_queue()
{
    Message *msg;
#ifdef DEBUG
    char tmp_str[255];
#endif

    msg_queue->reset();
    msg = msg_queue->next();
    if (msg)
    {
        msg_queue->remove(msg);
        return msg;
    }
    return NULL;
}

Message *Process::check_for_msg(float time)
{
    Message *msg;
    Connection *conn;
#ifdef DEBUG
    char tmp_str[255];
#endif

    msg_queue->reset();
    msg = msg_queue->next();
    if (msg)
    {
        msg_queue->remove(msg);
#ifdef DEBUG
        sprintf(tmp_str, "msg %s removed from queue", covise_msg_types_array[msg->type]);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        return msg;
    }
    else
    {
        if (list_of_connections->count() == 0)
        {
            msg = new Message;
            msg->type = COVISE_MESSAGE_SOCKET_CLOSED;
            return msg; // avoid error message, if connection list is empty
        }
        conn = list_of_connections->check_for_input(time);
        if (conn)
        {
            msg = new Message;
            conn->recv_msg(msg);
            switch (msg->type)
            {
            case COVISE_MESSAGE_CLOSE_SOCKET:
            case COVISE_MESSAGE_EMPTY:
            case COVISE_MESSAGE_SOCKET_CLOSED:
                list_of_connections->remove(conn);
                print_comment(__LINE__, __FILE__, "Socket Closed");
                //delete conn;
                return msg;
            //break;
            case COVISE_MESSAGE_NEW_SDS:
                handle_shm_msg(msg);
                delete msg;
                return check_for_msg(time);
            //break;
            default:
                break;
            }
            return msg;
        }
        else
            return NULL;
    }
}

Message *Process::wait_for_msg(int covise_msg_type, Connection *conn = 0)
{
    Message *msg;
    Connection *tmpconn;
#ifdef DEBUG
    char tmp_str[255];
#endif

    while (1)
    {
        tmpconn = list_of_connections->wait_for_input();
        msg = new Message;
        tmpconn->recv_msg(msg);
        switch (msg->type)
        {
        case COVISE_MESSAGE_NEW_SDS:
            handle_shm_msg(msg);
            delete msg;
            break;
        case COVISE_MESSAGE_CLOSE_SOCKET:
        case COVISE_MESSAGE_STDINOUT_EMPTY:
        case COVISE_MESSAGE_EMPTY:
        case COVISE_MESSAGE_SOCKET_CLOSED:
            list_of_connections->remove(tmpconn);
            print_comment(__LINE__, __FILE__, "Socket Closed");
        //delete tmpconn;
        default:
            if (conn == 0 || tmpconn == conn)
            {
                if (msg->type == covise_msg_type)
                    return msg;
            }
            msg_queue->add(msg);
#ifdef DEBUG
            sprintf(tmp_str, "msg %s added to queue", covise_msg_types_array[msg->type]);
            print_comment(__LINE__, __FILE__, tmp_str);
#endif
            break;
        }
    }
}

Message *Process::wait_for_msg(int *covise_msg_type, int no,
                               Connection *conn = 0)
{
    Message *msg;
    Connection *tmpconn;
    int i;
#ifdef DEBUG
    char tmp_str[255];
#endif

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "in Process::wait_for_msg, waiting for:");
    for (i = 0; i < no; i++)
    {
        sprintf(tmp_str, "%s", covise_msg_types_array[ec_msg_type[i]]);
        print_comment(__LINE__, __FILE__, tmp_str);
    }
#endif
    while (1)
    {
        tmpconn = list_of_connections->wait_for_input();
        msg = new Message;
        tmpconn->recv_msg(msg);
#ifdef DEBUG
        sprintf(tmp_str, "msg %s received", covise_msg_types_array[msg->type]);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        switch (msg->type)
        {
        case COVISE_MESSAGE_NEW_SDS:
            handle_shm_msg(msg);
            delete msg;
            break;
        case COVISE_MESSAGE_CLOSE_SOCKET:
        case COVISE_MESSAGE_STDINOUT_EMPTY:
        case COVISE_MESSAGE_EMPTY:
        case COVISE_MESSAGE_SOCKET_CLOSED:
            list_of_connections->remove(tmpconn);
            print_comment(__LINE__, __FILE__, "Socket Closed");
        //delete tmpconn;
        default:
            if (conn == 0 || tmpconn == conn)
            {
                for (i = 0; i < no && msg->type != covise_msg_type[i]; i++)
                {
                }
                if (i < no && msg->type == covise_msg_type[i])
                    return msg;
            } // else
            msg_queue->add(msg);
#ifdef DEBUG
            sprintf(tmp_str, "msg %s added to queue", covise_msg_types_array[msg->type]);
            print_comment(__LINE__, __FILE__, tmp_str);
#endif
            break;
        }
    }
}

int OrdinaryProcess::is_connected()
{
    if (!controller)
        return 0;
    return (controller->is_connected());
}

void OrdinaryProcess::send_ctl_msg(const Message *msg)
{
    if ((controller) && (controller->sendMessage(msg) == COVISE_SOCKET_INVALID))
    {
        list_of_connections->remove(controller);
        delete this;
        exit(0);
    }
}

void OrdinaryProcess::send_ctl_msg(TokenBuffer tb)
{
    Message msg(tb);
    msg.type = COVISE_MESSAGE_CO_MODULE;
    send_ctl_msg(&msg);
}

int OrdinaryProcess::get_socket_id(void (*remove_func)(int))
{
    {
        //	fprintf(stderr,"------ in OrdinaryProcess::get_socket_id: controller=%x\n",
        //			controller);
        return controller->get_id(remove_func);
    }
}

void Process::handle_shm_msg(Message *)
{
    print_comment(__LINE__, __FILE__, "handle_shm_msg invoked on Process object");
}

SimpleProcess::SimpleProcess(char *name, int arc, char *arv[])
    : OrdinaryProcess(name, arc, arv, SIMPLEPROCESS)
{
    Host *tmphost = new Host(arv[2]);
    contact_controller(atoi(arv[1]), tmphost);
}

SimpleProcess::~SimpleProcess() // destructor
{
}

void SimpleProcess::contact_controller(int p, Host *h)
{
    controller = new ControllerConnection(h, p, id, send_type);
    list_of_connections->add(controller);
}

namespace
{

// Retrieve information from running network interfaces
#if defined(__linux__)

void getNetAdaptInfo(vector<string> &name, vector<string> &ip_address, vector<string> &mac_address)
{
    char buf[1024];
    int sck;
    int nInterfaces;
    int i;
    struct ifconf ifc;
    struct ifreq *ifr;

    // Get a socket handle
    // sck = socket(AF_INET, SOCK_DGRAM, 0);
    sck = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sck < 0)
    {
        perror("socket");
        // TODO
    }

    // Query available interfaces
    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = buf;
    if (ioctl(sck, SIOCGIFCONF, &ifc) < 0)
    {
        perror("ioctl(SIOCGIFCONF)");
        // TODO
    }

    // Iterate through the list of interfaces
    ifr = ifc.ifc_req;
    nInterfaces = ifc.ifc_len / sizeof(struct ifreq);

#ifdef DEBUG
    cout << endl << "Found " << nInterfaces << " running network interfaces:" << endl;
    cout << endl;
#endif

    for (i = 0; i < nInterfaces; i++)
    {
        struct ifreq *item = &ifr[i];

// Show the device name and IP address
#ifdef DEBUG
        printf("%s: IP %s", item->ifr_name, inet_ntoa(((struct sockaddr_in *)&item->ifr_addr)->sin_addr));
#endif

        ip_address.push_back(inet_ntoa(((struct sockaddr_in *)&item->ifr_addr)->sin_addr));

        // Get the MAC address
        if (ioctl(sck, SIOCGIFHWADDR, item) < 0)
        {
            perror("ioctl(SIOCGIFHWADDR)");
            // TODO
        }

        else
        {
#ifdef DEBUG
            printf(", MAC %02x:%02x:%02x:%02x:%02x:%02x\n",
                   (int)((unsigned char *)item->ifr_hwaddr.sa_data)[0],
                   (int)((unsigned char *)item->ifr_hwaddr.sa_data)[1],
                   (int)((unsigned char *)item->ifr_hwaddr.sa_data)[2],
                   (int)((unsigned char *)item->ifr_hwaddr.sa_data)[3],
                   (int)((unsigned char *)item->ifr_hwaddr.sa_data)[4],
                   (int)((unsigned char *)item->ifr_hwaddr.sa_data)[5]);
#endif
            std::ostringstream mac;
            for (int index = 0; index < 6; index++)
            {
                if (index)
                    mac << ":";
                mac.width(2);
                mac.fill('0');
                mac << std::hex
                    << (int)((unsigned char *)item->ifr_hwaddr.sa_data)[index];
            }
            name.push_back(item->ifr_name);
            // ip_address.push_back(inet_ntoa(((struct sockaddr_in *)&item->ifr_addr)->sin_addr));
            mac_address.push_back(mac.str());
        }

        // Get the broadcast address
        if (ioctl(sck, SIOCGIFBRDADDR, item) >= 0)
        {
#ifdef DEBUG
            printf(", BROADCAST %s", inet_ntoa(((struct sockaddr_in *)&item->ifr_broadaddr)->sin_addr));
            printf("\n");
#endif
        }
    }

    close(sck);
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/sysctl.h>
#include <sys/sockio.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <net/route.h>

void getNetAdaptInfo(vector<string> &name, vector<string> &ip_address, vector<string> &mac_address)
{
    struct ifaddrs *allifa = NULL;
    getifaddrs(&allifa);
    std::map<std::string, std::string> lladdr;
    typedef std::map<std::string, std::string> llmap;
    llmap lladdrs;
    std::map<std::string, bool> inetmap;

    // scan ether addresses of all interfaces
    for (struct ifaddrs *ifa = allifa; ifa; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr->sa_family == AF_LINK)
        {
            const struct sockaddr_dl *sdl = (const struct sockaddr_dl *)ifa->ifa_addr;
            const char *u = LLADDR(sdl);
            if (sdl->sdl_alen >= 6)
            {
                char buffer[1024];
                for (int i = 0; i < 6; i++)
                {
                    sprintf(buffer + 2 * i, "%02x", u[i] & 0xff);
                }
                lladdrs[ifa->ifa_name] = buffer;
            }
        }
    }

    // determine inet addresses of all interfaces
    for (struct ifaddrs *ifa = allifa; ifa; ifa = ifa->ifa_next)
    {
        std::string lladdr;
        llmap::iterator ent = lladdrs.find(ifa->ifa_name);
        if (ent != lladdrs.end())
            lladdr = ent->second;

#ifdef DEBUG
        fprintf(stderr, "NAME: %s (%s)\n", ifa->ifa_name,
                ifa->ifa_flags & IFF_UP ? "up" : "down");
        if (!lladdr.empty())
            fprintf(stderr, "\tMAC %s\n", lladdr.c_str());
#endif

        // create entry for each inet address (if the interface is up)
        bool have_inet = false;
        if (ifa->ifa_addr->sa_family == AF_INET && (ifa->ifa_flags & IFF_UP))
        {
            for (struct ifaddrs *ift = ifa; ift != NULL; ift = ift->ifa_next)
            {
                if (strcmp(ifa->ifa_name, ift->ifa_name) != 0)
                    continue;
                if (!ift->ifa_addr)
                    continue;

                if (ift->ifa_addr->sa_family == AF_INET)
                {
                    struct sockaddr_in *sin = (struct sockaddr_in *)ifa->ifa_addr;
                    if (sin == NULL)
                        continue;

                    inetmap[ifa->ifa_name] = true;

                    name.push_back(ifa->ifa_name);
                    ip_address.push_back(inet_ntoa(sin->sin_addr));
                    mac_address.push_back(lladdr);

#ifdef DEBUG
                    fprintf(stderr, "\tINET %s\n", inet_ntoa(sin->sin_addr));
#endif
                }
            }
        }
    }

    // make sure to have an entry for each interface with link level address
    for (llmap::iterator it = lladdrs.begin(); it != lladdrs.end(); ++it)
    {
        std::string n = it->first;
        std::string lladdr = it->second;
        if (!inetmap[n] && !lladdr.empty())
        {
            name.push_back(n);
            ip_address.push_back("");
            mac_address.push_back(lladdr);
        }
    }
    freeifaddrs(allifa);
}

#elif defined(_WIN32)
#include <windows.h>
#include <wincon.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <Nb30.h>

typedef struct _ASTAT_
{

    ADAPTER_STATUS adapt;
    NAME_BUFFER NameBuff[30];

} ASTAT, *PASTAT;

void getNetAdaptInfo(vector<string> &name, vector<string> &ip_address, vector<string> &mac_address)
{
#ifdef NETBIOS_VERSION
    ASTAT Adapter;

    NCB Ncb;
    UCHAR uRetCode;
    char NetName[50];
    char buffer[6];
    LANA_ENUM lenum;
    int i;

    memset(&Ncb, 0, sizeof(Ncb));
    Ncb.ncb_command = NCBENUM;
    Ncb.ncb_buffer = (UCHAR *)&lenum;
    Ncb.ncb_length = sizeof(lenum);
    uRetCode = Netbios(&Ncb);
//printf( "The NCBENUM return code is: 0x%x \n", uRetCode );

#ifdef DEBUG
    printf("Found %d network interfaces\n", lenum.length);
#endif

    for (i = 0; i < lenum.length; i++)
    {
        memset(&Ncb, 0, sizeof(Ncb));
        Ncb.ncb_command = NCBRESET;
        Ncb.ncb_lana_num = lenum.lana[i];

        uRetCode = Netbios(&Ncb);
        /*printf( "The NCBRESET on LANA %d return code is: 0x%x \n",
              lenum.lana[i], uRetCode );*/

        memset(&Ncb, 0, sizeof(Ncb));
        Ncb.ncb_command = NCBASTAT;
        Ncb.ncb_lana_num = lenum.lana[i];

        strcpy((char *)Ncb.ncb_callname, "*               ");
        Ncb.ncb_buffer = (PUCHAR)&Adapter;
        Ncb.ncb_length = sizeof(Adapter);

        uRetCode = Netbios(&Ncb);
        // printf( "The NCBASTAT on LANA %d return code is: 0x%x \n",
        //        lenum.lana[i], uRetCode );
        if (uRetCode == 0)
        {

#ifdef DEBUG
            printf("The Ethernet Number on LANA %d is: %02x:%02x:%02x:%02x:%02x:%02x\n",
                   lenum.lana[i],
                   Adapter.adapt.adapter_address[0],
                   Adapter.adapt.adapter_address[1],
                   Adapter.adapt.adapter_address[2],
                   Adapter.adapt.adapter_address[3],
                   Adapter.adapt.adapter_address[4],
                   Adapter.adapt.adapter_address[5]);
#endif

            sprintf(buffer, "%02x%02x%02x%02x%02x%02x",
                    Adapter.adapt.adapter_address[0],
                    Adapter.adapt.adapter_address[1],
                    Adapter.adapt.adapter_address[2],
                    Adapter.adapt.adapter_address[3],
                    Adapter.adapt.adapter_address[4],
                    Adapter.adapt.adapter_address[5]);

            mac_address.push_back(buffer);
        }
    }

#else
    // Get the buffer length required for IP_ADAPTER_INFO.
    char buffer[16];
    char description[1024];
    char ip_addr[18];
    ULONG BufferLength = 0;
    BYTE *pBuffer = 0;
    if (ERROR_BUFFER_OVERFLOW == GetAdaptersInfo(0, &BufferLength))
    {
        // Now the BufferLength contains the required buffer length.
        // Allocate necessary buffer.
        pBuffer = new BYTE[BufferLength];
    }
    else
    {
        // Error occurred. handle it accordingly.
    }

    // Get the Adapter Information.
    PIP_ADAPTER_INFO pAdapterInfo = reinterpret_cast<PIP_ADAPTER_INFO>(pBuffer);
    GetAdaptersInfo(pAdapterInfo, &BufferLength);

#ifdef _DEBUG
    printf("Found the following network interfaces:\n");
#endif

    // Iterate the network adapters and print their MAC address.
    while (pAdapterInfo)
    {
#ifdef _DEBUG
        cout << pAdapterInfo->Description << ": ";

        printf("MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
               pAdapterInfo->Address[0],
               pAdapterInfo->Address[1],
               pAdapterInfo->Address[2],
               pAdapterInfo->Address[3],
               pAdapterInfo->Address[4],
               pAdapterInfo->Address[5]);
#endif

        sprintf(buffer, "%02x%02x%02x%02x%02x%02x",
                pAdapterInfo->Address[0],
                pAdapterInfo->Address[1],
                pAdapterInfo->Address[2],
                pAdapterInfo->Address[3],
                pAdapterInfo->Address[4],
                pAdapterInfo->Address[5]);

        sprintf(description, "%s", pAdapterInfo->Description);
        sprintf(ip_addr, "%s", pAdapterInfo->IpAddressList.IpAddress.String);

        name.push_back(description);
        ip_address.push_back(ip_addr);
        mac_address.push_back(buffer);
        pAdapterInfo = pAdapterInfo->Next;
    }

    delete[] pBuffer;

#endif
}

#else

void getNetAdaptInfo(vector<string> &name, vector<string> &ip_address, vector<string> &mac_address)
{
}
#endif
}

char *Process::get_list_of_interfaces()
{
    std::vector<std::string> ifname, ip, mac;
    getNetAdaptInfo(ifname, ip, mac);
    std::string result;
    for (std::vector<std::string>::iterator it = ip.begin();
         it != ip.end();
         ++it)
    {
        result += *it;
        result += "\n";
    }

    char *ret = new char[result.length() + 1];
    strcpy(ret, result.c_str());
    return ret;
}

const char *Process::get_hostname()
{
    return covise_hostname;
}

#ifndef MAXHOSTNAMELEN
#define MAXHOSTNAMELEN 64
#endif

Host *Process::get_covise_host()
{
    if (!host)
        host = new Host();
    return host;
}
