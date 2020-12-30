/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description: Interface class for application modules to the COVISE     **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 **	  25.06.97  V1.1 Harald Nebel, added GENERIC-stuff		  **
\**************************************************************************/

#include "ApplInterface.h"
#include <covise/covise.h>
#include <covise/covise_appproc.h>
#include <covise/Covise_Util.h>
#include <util/coLog.h>
#include <do/coDistributedObject.h>
#include <net/concrete_messages.h>
#if defined(__linux__) || defined(__APPLE__)
#define NODELETE_APPROC
#endif

#if defined TTEST || defined TIMING
#include <sys/time.h>
#include <sys/times.h>

#ifdef __sgi
#define BEST_TIMER CLOCK_SGI_CYCLE
#else
#define BEST_TIMER CLOCK_REALTIME
#endif
#endif

using namespace covise;

// this is our own C++ conformant strcpy routine
inline char *STRDUP(const char *old)
{
    return strcpy(new char[strlen(old) + 1], old);
}

//==========================================================================
// definition of static class elements
//==========================================================================

static int counter = 0;

CtlMessage *Covise::msg = NULL;
char *Covise::reply_port_name = NULL;

char *Covise::modkey = NULL;

CoviseCallback *Covise::portReplyCallbackFunc = NULL;
CoviseCallback *Covise::startCallbackFunc = NULL;
CoviseCallback *Covise::addObjectCallbackFunc = NULL;
FeedbackCallback *Covise::feedbackCallbackFunc = NULL;

CoviseCallback *Covise::genericCallbackFunc = NULL;
CoviseCallback *Covise::syncCallbackFunc = NULL;

CoviseCallback *Covise::afterFinishCallbackFunc = NULL;
CoviseCallback *Covise::pipelineFinishCallbackFunc = NULL;
void *Covise::portReplyUserData = 0L;
void *Covise::portReplyCallbackData = 0L;
void *Covise::startUserData = 0L;
void *Covise::addObjectUserData = 0L;
void *Covise::feedbackUserData = 0L;
void *Covise::startCallbackData = 0L;
void *Covise::addObjectCallbackData = 0L;

void *Covise::genericUserData = 0L;
void *Covise::genericCallbackData = 0L;
char *Covise::genericMessageData = 0L;
void *Covise::syncUserData = 0L;
void *Covise::syncCallbackData = 0L;

void *Covise::afterFinishUserData = 0L;
void *Covise::afterFinishCallbackData = 0L;
void *Covise::pipelineFinishUserData = 0L;
void *Covise::pipelineFinishCallbackData = 0L;
int Covise::pipeline_state_once = 0;
int Covise::renderMode_ = 0;
char *Covise::objNameToAdd_ = NULL;
char *Covise::objNameToDelete_ = NULL;

#ifndef _WIN32
#ifdef COVISE_Signals
static SignalHandler sig_handler;
#endif
#endif

void Covise::generic(Message *applMsg)
{

    // int ntok;
    char *token[MAXTOKENS];
    const char *sep = "\n";
    char *tmp;
    char *tmpp;
    char *tmp2;

    //  for(k=0; k < MAXTOKENS; k++) {
    //   token[k] = NULL;
    //  };

    // hier Switch fuer sync, generic, master, slave
    //        0   1    2     3   4    5           6
    // Format key key2 name, nr, host STATUS|SYNC MASTER|SLAVE DATA
    //
    // keyword: Kennung fuer Module, hier uninteressant
    // key2: hier UIFINFO (APPINFO und INIT kommen nicht im Module vor
    // name,nr,host: beim ersten Mal speichern !

    tmp = new char[strlen(applMsg->data.data()) + 1];
    strcpy(tmp, applMsg->data.data());
    parseMessage(tmp, &token[0], MAXTOKENS, sep); /*ntok =*/
    // check modkey-Variable
    if (modkey == NULL)
    {
        modkey = new char[strlen(token[0]) + 1];
        strcpy(modkey, token[0]);
    };

    if (strcmp(token[5], "SYNC") == 0)
    {
        tmp = new char[strlen(applMsg->data.data()) + 1];
        strcpy(tmp, applMsg->data.data());

        for (int k = 0; k <= 5; k++)
        {
            tmpp = strstr(tmp, sep);
            tmp2 = new char[strlen(tmpp) + 1];
            strcpy(tmp2, tmpp);
            tmp = &tmp2[1];
        };
        genericMessageData = new char[strlen(tmp) + 1];
        strcpy(genericMessageData, tmp);

        doSync(applMsg);
    }
    else if (strcmp(token[5], "STATUS") == 0)
    {
        if (strcmp(token[6], "MASTER") == 0)
        {
            doMasterSwitch();
        }
        else if (strcmp(token[6], "SLAVE") == 0)
        {
            doSlaveSwitch();
        };
    }
    else
    {

        tmp = new char[strlen(applMsg->data.data()) + 1];
        strcpy(tmp, applMsg->data.data());

        for (int k = 0; k <= 4; k++)
        {
            tmpp = strstr(tmp, sep);
            tmp2 = new char[strlen(tmpp) + 1];
            strcpy(tmp2, tmpp);
            tmp = &tmp2[1];
        };
        genericMessageData = new char[strlen(tmp) + 1];
        strcpy(genericMessageData, tmp);

        doGeneric(applMsg);
    };
}

char *Covise::get_generic_message()
{
    char *tmp;

    tmp = new char[strlen(genericMessageData) + 1];
    strcpy(tmp, genericMessageData);
    return tmp;
}

// set the description for the module
void Covise::set_module_description(const char *descr)
{
    module_description = STRDUP(descr);
}

// get the description for the module
const char *Covise::get_module_description()
{
    return module_description;
}

// unspecified port: why ?
/*
void Covise::add_port(enum appl_port_type type, char *name)
{
   if(type == OUTPUT || type == INPUT || type == PARIN || type == PAROUT)
   {
      int i=0;
      while(port_name[i])
         i++;
      port_type[i]=type;
      port_name[i]=name;
      port_default[i]=NULL;
port_datatype[i]=NULL;
port_dependency[i]=NULL;
port_required[i]=1;
port_description[i]=NULL;
port_name[i+1]=NULL;
}
else
{
cerr << "wrong description type in add_port " << name << "\n";
return;
}
}
*/

/*
//////// Bei const char* : erzeuge Kopie
void Covise::add_port(enum appl_port_type type, const char *name, const char *dt, const char *desc)
{
    char *name_=new char [strlen(name)+1]; strcpy(name_,name);
    char *dt_  =new char [strlen(dt  )+1]; strcpy(dt_,dt);
    char *desc_=new char [strlen(desc)+1]; strcpy(desc_,desc);
    add_port(type, name_, dt_, desc_);
}
*/
void Covise::add_port(enum appl_port_type type, const char *name,
                      const char *datatype, const char *descr)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while ((i < MAX_PORTS) && (port_name[i]))
            i++;
        if (i == MAX_PORTS)
        {
            cerr << "\n\007Number of ports limited to "
                 << MAX_PORTS << " in class Covise (ApplInterface.h)\n"
                 << "Covise must be completely recompiled to change this !\n"
                 << "-> ignoring port '" << name << "'" << endl;
            return;
        }
        port_type[i] = type;
        port_name[i] = STRDUP(name);
        port_default[i] = NULL;
        port_datatype[i] = STRDUP(datatype);
        port_dependency[i] = NULL;
        port_required[i] = 1;
        port_description[i] = STRDUP(descr);
        port_name[i + 1] = NULL;
    }
    else
    {
        cerr << "wrong description type in add_port " << name << "\n";
        return;
    }
}

/*
void Covise::set_port_description( char *name,char *descr)
{
    int i=0;
    while(port_name[i])
    {
   if(strcmp(port_name[i],name)==0)
       break;
   i++;
    }
    if(port_name[i]==NULL)
{
cerr << "wrong portname " << name << " in set_port_description\n";
return;
}
port_description[i]=descr;

}
*/

void Covise::set_port_default(const char *name, const char *def)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_default\n";
        return;
    }

    if (port_type[i] != PARIN && port_type[i] != PAROUT)
    {
        cerr << "wrong port type in set_port_default " << name << "\n";
        return;
    }
    port_default[i] = def;
}

void Covise::set_port_dependency(const char *name, char *depPort)
{
    int i = 0;

    /// find the port
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i])
        port_dependency[i] = STRDUP(depPort);
    else
        cerr << "wrong portname " << name << " in set_port_dependency\n";
}

void Covise::set_port_required(const char *name, int isReqired)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_required\n";
        return;
    }
    if (port_type[i] != INPUT_PORT)
    {
        cerr << "wrong port type in set_port_required " << name << "\n";
        return;
    }
    port_required[i] = isReqired;
}

char *Covise::get_description_message()
{
    CharBuffer msg(500);
    msg += "DESC\n";
    msg += get_module();
    msg += '\n';
    msg += get_host();
    msg += '\n';
    if (module_description)
        msg += module_description;
    else
        msg += get_module();
    msg += '\n';

    int i = 0, ninput = 0, noutput = 0, nparin = 0, nparout = 0;

    while (port_name[i])
    {
        switch (port_type[i])
        {
        case INPUT_PORT:
            ninput++;
            break;
        case OUTPUT_PORT:
            noutput++;
            break;
        case PARIN:
            nparin++;
            break;
        case PAROUT:
            nparout++;
            break;
        default:
            break;
        }
        i++;
    }

    msg += ninput; // number of parameters
    msg += '\n';
    msg += noutput;
    msg += '\n';
    msg += nparin;
    msg += '\n';
    msg += nparout;
    msg += '\n';

    i = 0; // INPUT ports
    while (port_name[i])
    {
        if (port_type[i] == INPUT_PORT)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
                cerr << "no datatype for port " << port_name[i] << "\n";
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_required[i])
                msg += "req\n";
            else
                msg += "opt\n";
        }
        i++;
    }

    i = 0; // OUTPUT ports
    while (port_name[i])
    {
        if (port_type[i] == OUTPUT_PORT)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
                cerr << "no datatype for port " << port_name[i] << "\n";
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_dependency[i])
            {
                msg += port_dependency[i];
                msg += '\n';
            }
            else
                msg += "default\n";
        }
        i++;
    }

    i = 0; // PARIN ports
    while (port_name[i])
    {
        if (port_type[i] == PARIN)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
                cerr << "no datatype for port " << port_name[i] << "\n";
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_default[i] == NULL)
                cerr << "no default value for parameter " << port_name[i] << "\n";

            msg += port_default[i];

            msg += '\n';
            msg += "IMM\n";
        }
        i++;
    }

    i = 0; // PAROUT ports
    while (port_name[i])
    {
        if (port_type[i] == PAROUT)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
                cerr << "no datatype for port " << port_name[i] << "\n";
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_default[i] == NULL)
                cerr << "no default value for parameter " << port_name[i] << "\n";
            msg += port_default[i];
            msg += '\n';
        }
        i++;
    }
    return (msg.return_data());
}

//=====================================================================
//
//=====================================================================
void Covise::init(int argc, char *argv[])
{
    if (argc == 2 && 0 == strcmp(argv[1], "-d"))
    {
        printDesc(argv[0]);
        exit(0);
    }
// Initialization of the communciation environment
#ifdef _WIN32
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD(1, 1);

    err = WSAStartup(wVersionRequested, &wsaData);
#endif

    //fprintf(stderr,"-- appmod = new ApplicationProcess\n");
    appmod = new ApplicationProcess(argv[0], argc, argv);
    //fprintf(stderr,"-- appmod = %x\n", appmod);

    //fprintf(stderr,"--socket_id = appmod->get_socket_id\n");
    socket_id = appmod->get_socket_id(Covise::remove_socket);
    //fprintf(stderr,"-- socket_id = %d\n", socket_id);
    auto crbExec = covise::getExecFromCmdArgs(argc, argv);
    h_name = appmod->get_hostname();
    const char *p = strrchr(crbExec.name, '/');
    if (p)
        m_name = p + 1;
    else
        m_name = crbExec.name;
    instance = std::to_string(crbExec.moduleCount);
    print_comment(__LINE__, __FILE__, "Application Module succeeded");

#ifdef DEBUG
    cerr << crbExec << " Application Module succeeded" << endl;
#endif

    init_flag = 1;
#ifndef _WIN32
#ifdef COVISE_Signals
    // Initialization of signal handlers
    sig_handler.addSignal(SIGBUS, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGPIPE, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGTERM, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGSEGV, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGFPE, (void *)signal_handler, NULL);
    init_emergency_message();
#endif
// initialize time for time measurements
//    open_timing();
#endif

    char* d = get_description_message();
    Message message{ COVISE_MESSAGE_PARINFO, DataHandle(d, strlen(d) + 1) }; // should be a real type

    appmod->send_ctl_msg(&message);
}

//=====================================================================
//
//=====================================================================
void Covise::send_stop_pipeline()
{
    if ((get_module() != NULL) && (get_host() != NULL) && (get_instance() != NULL) && (appmod != NULL))
    {
        int size = 1; // final '\0'
        size += (int)strlen(get_module()) + 1;
        size += (int)strlen(get_instance()) + 1;
        size += (int)strlen(get_host()) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, get_module());
        strcat(msgdata, "\n");
        strcat(msgdata, get_instance());
        strcat(msgdata, "\n");
        strcat(msgdata, get_host());

        Message message{ COVISE_MESSAGE_COVISE_STOP_PIPELINE , DataHandle(msgdata, strlen(msgdata) + 1)};
        appmod->send_ctl_msg(&message);

    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send message without get_instance()/init before");

    return;
}

//=====================================================================
//
//=====================================================================
int Covise::send_ctl_message(covise_msg_type type, char *string)
{
    if ((get_module() != NULL) && (get_host() != NULL) && (get_instance() != NULL) && (appmod != NULL))
    {
        //cerr << "MODULE SENDING MESSAGE TO CONTROLLER : " << message->data << endl;
        Message message{ type, DataHandle(string, strlen(string) + 1) };
        appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "Cannot send message without get_instance()/init before");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
int Covise::send_generic_message(const char *keyword, const char *string)
{
    if ((get_module() != NULL) && (get_host() != NULL) && (get_instance() != NULL) && (appmod != NULL))
    {
        int size = 1; // final '\0'
        size += (int)strlen(modkey) + 1;
        size += (int)strlen(keyword) + 1;
        size += (int)strlen(get_module()) + 1;
        size += (int)strlen(get_instance()) + 1;
        size += (int)strlen(get_host()) + 1;
        size += (int)strlen(string) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, modkey);
        strcat(msgdata, "\n");
        strcat(msgdata, keyword);
        strcat(msgdata, "\n");
        strcat(msgdata, get_module());
        strcat(msgdata, "\n");
        strcat(msgdata, get_instance());
        strcat(msgdata, "\n");
        strcat(msgdata, get_host());
        strcat(msgdata, "\n");
        strcat(msgdata, string);

        //cerr << "MODULE SENDING MESSAGE TO CONTROLLER : " << message->data << endl;

        Message message{ COVISE_MESSAGE_GENERIC , DataHandle(msgdata , strlen(msgdata ) + 1)};

        appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "Cannot send message without get_instance()/init before");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
int Covise::send_genericinit_message(const char *mkey, const char *keyword, const char *string)
{
    if ((get_module() != NULL) && (get_host() != NULL) && (get_instance() != NULL) && (appmod != NULL))
    {

        /* copy the modulespecific keyword mkey into modkey */
        if (modkey == NULL)
        {
            modkey = new char[strlen(mkey) + 1];
            strcpy(modkey, mkey);
        };

        int size = 1; // final '\0'
        size += (int)strlen(mkey) + 1;
        size += (int)strlen(keyword) + 1;
        size += (int)strlen(get_module()) + 1;
        size += (int)strlen(get_instance()) + 1;
        size += (int)strlen(get_host()) + 1;
        size += (int)strlen(string) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, mkey);
        strcat(msgdata, "\n");
        strcat(msgdata, keyword);
        strcat(msgdata, "\n");
        strcat(msgdata, get_module());
        strcat(msgdata, "\n");
        strcat(msgdata, get_instance());
        strcat(msgdata, "\n");
        strcat(msgdata, get_host());
        strcat(msgdata, "\n");
        strcat(msgdata, string);

        //cerr << "MODULE SENDING MESSAGE TO CONTROLLER : " << message->data << endl;

        Message message{ COVISE_MESSAGE_GENERIC , DataHandle(msgdata , strlen(msgdata) + 1 )};
        appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "Cannot send message without get_instance()/init before");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
void Covise::cancel_param(const char *name)
{
    if ((get_module() != NULL) && (get_host() != NULL) && (get_instance() != NULL) && (appmod != NULL))
    {
        int size = 1; // final '\0'
        size += (int)strlen(get_module()) + 1;
        size += (int)strlen(get_instance()) + 1;
        size += (int)strlen(get_host()) + 1;
        size += (int)strlen("CANCEL") + 1;
        size += (int)strlen(name) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, get_module());
        strcat(msgdata, "\n");
        strcat(msgdata, get_instance());
        strcat(msgdata, "\n");
        strcat(msgdata, get_host());
        strcat(msgdata, "\n");
        strcat(msgdata, "CANCEL");
        strcat(msgdata, "\n");
        strcat(msgdata, name);

        Message message{ COVISE_MESSAGE_REQ_UI , DataHandle(msgdata , strlen(msgdata) + 1) };
        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send show_param message without get_instance()/init before");
}

//=====================================================================
//
//=====================================================================
void Covise::reopen_param(const char *name)
{
    if ((get_module() != NULL) && (get_host() != NULL) && (get_instance() != NULL) && (appmod != NULL))
    {
        int size = 1; // final '\0'
        size += (int)strlen(get_module()) + 1;
        size += (int)strlen(get_instance()) + 1;
        size += (int)strlen(get_host()) + 1;
        size += (int)strlen("REOPEN") + 1;
        size += (int)strlen(name) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, get_module());
        strcat(msgdata, "\n");
        strcat(msgdata, get_instance());
        strcat(msgdata, "\n");
        strcat(msgdata, get_host());
        strcat(msgdata, "\n");
        strcat(msgdata, "REOPEN");
        strcat(msgdata, "\n");
        strcat(msgdata, name);

        Message message{ COVISE_MESSAGE_REQ_UI , DataHandle(msgdata , strlen(msgdata) + 1) };
        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send show_param message without get_instance()/init before");
}

//=====================================================================
//
//=====================================================================
void Covise::main_loop()
{
    int quit_now;
    while (1)
    {
        applMsg = appmod->wait_for_ctl_msg();
        switch (applMsg->type)
        {
        case COVISE_MESSAGE_INFO:
            doPortReply(applMsg);
            break;
        case COVISE_MESSAGE_UI:
            doParam(applMsg);
            reply_param_name = NULL;
            break;

        case COVISE_MESSAGE_SOCKET_CLOSED:
            exit(0);
            break;

        case COVISE_MESSAGE_QUIT:
            quit_now = doQuit();
#ifndef _WIN32
//          close_timing();
#endif
            if (quit_now)
            {
                print_comment(__LINE__, __FILE__,
                              "Application module: correctly finishing");
                appmod->getConnectionList()->remove(applMsg->conn);
                appmod->getConnectionList()->deleteConnection(applMsg->conn);
                appmod->delete_msg(applMsg);
#ifndef NODELETE_APPROC
                delete appmod;
#endif
                appmod = NULL;
                applMsg = NULL;
                exit(0);
            }
            break;

        case COVISE_MESSAGE_START:
            startCallbackData = (void *)applMsg;
#ifndef _WIN32
// start_timer(0);
#endif
            // fprintf(stderr, "vor doStart\n");
            doStart(applMsg);
// fprintf(stderr, "nach doStart\n");
#ifndef _WIN32
//  stop_timer(0);
//  print_timer(Covise::get_module(),Covise::get_instance(),Covise::get_host(),0,"Time for execution");
#endif
            break;

        case COVISE_MESSAGE_GENERIC:
            generic(applMsg);
            break;

        case COVISE_MESSAGE_FEEDBACK:
            callFeedbackCallback(applMsg);
            break;

        default:
            doCustom(applMsg);
            break;
        }
        appmod->delete_msg(applMsg);
    }
}

//=====================================================================
//
//=====================================================================
void Covise::progress_main_loop()
{
    int quit_now;

    while (1)
    {
        applMsg = appmod->check_for_ctl_msg();
        if (applMsg == NULL)
            doProgress();
        else
        {
            switch (applMsg->type)
            {
            case COVISE_MESSAGE_INFO:
                doPortReply(applMsg);
                break;

            case COVISE_MESSAGE_UI:
                doParam(applMsg);
                reply_param_name = NULL;
                break;

            case COVISE_MESSAGE_SOCKET_CLOSED:
                exit(0);
                break;

            case COVISE_MESSAGE_QUIT:
                quit_now = doQuit();
#ifndef _WIN32
// close_timing();
#endif
                if (quit_now)
                {
                    print_comment(__LINE__, __FILE__, "Application module: correctly finishing");
                    appmod->getConnectionList()->deleteConnection(applMsg->conn);
                    appmod->delete_msg(applMsg);
#ifndef NODELETE_APPROC
                    delete appmod;
#endif
                    appmod = NULL;
                    applMsg = NULL;
                    exit(0);
                }
                break;

            case COVISE_MESSAGE_START:
#ifndef _WIN32
// start_timer(0);
#endif
                doStart(applMsg);
#ifndef _WIN32
// stop_timer(0);
// print_timer(Covise::get_module(),Covise::get_instance(),Covise::get_host(),0,"Time for execution");
#endif
                break;

            case COVISE_MESSAGE_GENERIC:
                generic(applMsg);
                break;

            case COVISE_MESSAGE_FEEDBACK:
                callFeedbackCallback(applMsg);
                break;

            default:
                doCustom(applMsg);
                break;
            }
            appmod->delete_msg(applMsg);
        }
    }
}

//=====================================================================
// do one message
//=====================================================================
void Covise::do_one_event()
{
    int quit_now;
    applMsg = appmod->wait_for_ctl_msg();
    switch (applMsg->type)
    {
    case COVISE_MESSAGE_INFO:
        doPortReply(applMsg);
        break;
    case COVISE_MESSAGE_UI:
        doParam(applMsg);
        reply_param_name = NULL;
        break;

    case COVISE_MESSAGE_SOCKET_CLOSED:
        exit(0);
        break;

    case COVISE_MESSAGE_QUIT:
        quit_now = doQuit();
#ifndef _WIN32
//close_timing();
#endif
        if (quit_now)
        {
            print_comment(__LINE__, __FILE__, "Application module: correctly finishing");
            appmod->getConnectionList()->deleteConnection(applMsg->conn);
            appmod->delete_msg(applMsg);
#ifndef NODELETE_APPROC
            delete appmod;
#endif
            appmod = NULL;
            applMsg = NULL;
            exit(0);
        }
        break;

    case COVISE_MESSAGE_START:
#ifndef _WIN32
// start_timer(0);
#endif
        doStart(applMsg);
#ifndef _WIN32
// stop_timer(0);
// print_timer(Covise::get_module(),Covise::get_instance(),Covise::get_host(),0,"Time for execution");
#endif
        break;

    case COVISE_MESSAGE_FEEDBACK:
        callFeedbackCallback(applMsg);
        break;

    case COVISE_MESSAGE_GENERIC:
        generic(applMsg);
        break;

    default:
        doCustom(applMsg);
        break;
    }
    appmod->delete_msg(applMsg);
}

//=====================================================================
//
//=====================================================================
int Covise::check_and_handle_event(float time)
{
    int quit_now;

    char *token[MAXTOKENS];
    const char *sep = "\n";

    int i;
    //initialize we never need more
    for (i = 0; i < 3; ++i)
        token[i] = NULL;

    applMsg = appmod->check_for_ctl_msg(time);

    if (applMsg == NULL)
        return 0;
    else
    {
        switch (applMsg->type)
        {
        case COVISE_MESSAGE_INFO:
            doPortReply(applMsg);
            //	  cerr << "Covise::check_and_handle_event() : INFO" << endl;
            break;
        case COVISE_MESSAGE_UI:
            doParam(applMsg);
            reply_param_name = NULL;
            //	  cerr << "Covise::check_and_handle_event() : UI" << endl;
            break;

        case COVISE_MESSAGE_SOCKET_CLOSED:
            //	  cerr << "Covise::check_and_handle_event() : SOCKET_CLOSED" << endl;
            exit(0);
            break;

        case COVISE_MESSAGE_QUIT:
            //	  cerr << "Covise::check_and_handle_event() : QUIT" << endl;
            quit_now = doQuit();
#ifndef _WIN32
//close_timing();
#endif
            if (quit_now)
            {
                print_comment(__LINE__, __FILE__, "Application module: correctly finishing");
                appmod->getConnectionList()->deleteConnection(applMsg->conn);
                appmod->delete_msg(applMsg);
#ifndef NODELETE_APPROC
                delete appmod;
#endif
                appmod = NULL;
                applMsg = NULL;
                exit(0);
            }
            break;

        case COVISE_MESSAGE_START:
//	  cerr << "Covise::check_and_handle_event() : START" << endl;
#ifndef _WIN32
// start_timer(0);
#endif
            doStart(applMsg);
#ifndef _WIN32
//  stop_timer(0);
// print_timer(Covise::get_module(),Covise::get_instance(),Covise::get_host(),0,"Time for execution");
#endif
            break;

        case COVISE_MESSAGE_ADD_OBJECT:
        {
            //	  cerr << "Covise::check_and_handle_event() : ADD_OBJECT" << endl;

            parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);

            objNameToAdd_ = token[0];
            objNameToDelete_ = NULL;
            doAddObject();

            sendFinishedMsg();
            break;
        }
        case COVISE_MESSAGE_DELETE_OBJECT:
        {
            //	  cerr << "Covise::check_and_handle_event() : DELETE_OBJECT" << endl;

            parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);

            objNameToAdd_ = NULL;
            objNameToDelete_ = token[0];

            doAddObject();

            sendFinishedMsg();
            break;
        }
        case COVISE_MESSAGE_REPLACE_OBJECT:
        {
            //	  cerr << "Covise::check_and_handle_event() : REPLACE_OBJECT" << endl;

            parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);

            objNameToAdd_ = token[1];
            objNameToDelete_ = token[0];

            doAddObject();

            sendFinishedMsg();
            break;
        }

        case COVISE_MESSAGE_GENERIC:
            //	  cerr << "Covise::check_and_handle_event() : GENERIC" << endl;
            generic(applMsg);
            break;

        case COVISE_MESSAGE_FEEDBACK:
            //	  cerr << "Covise::check_and_handle_event() : FEEDBACK" << endl;
            callFeedbackCallback(applMsg);
            break;

        default:
            //	  cerr << "Covise::check_and_handle_event() : default" << endl;
            doCustom(applMsg);
            break;
        }
        appmod->delete_msg(applMsg);
        return 1;
    } // if
}

//=====================================================================
//
//=====================================================================
void Covise::ReceiveOneMsg()
{
    static int quit_now = 1;

    // Receive one message without sending a FINISHED to the Controller.
    // It is the application module's property to do this by calling
    // Covise::sendFinishedMsg(applMsg)
    //
    // For obvious reasons a copy of the message is made so that new messages
    // can arrive here without the old being overwritten
    //
    if (appmod != NULL)
        applMsg = appmod->wait_for_ctl_msg();
    else
    {
        cerr << "Application process not initialized" << endl;
        return;
    }

    switch (applMsg->type)
    {
    case COVISE_MESSAGE_INFO:
        doPortReply(applMsg);
        break;
    case COVISE_MESSAGE_UI:
        doParam(applMsg);
        reply_param_name = NULL;
        break;

    case COVISE_MESSAGE_SOCKET_CLOSED:
        exit(0);
        break;

    case COVISE_MESSAGE_QUIT:
        quit_now = doQuit();
        if (quit_now)
        {
            print_comment(__LINE__, __FILE__, "Application module: correctly finishing");
            appmod->getConnectionList()->deleteConnection(applMsg->conn);
            appmod->delete_msg(applMsg);
#ifndef NODELETE_APPROC
            delete appmod;
#endif
            appmod = NULL;
            applMsg = NULL;
            exit(0);
        }
        break;

    case COVISE_MESSAGE_START:
#ifndef _WIN32
// start_timer(0);
#endif
        doStartWithoutFinish(applMsg);
        // delete applMsg not here because we are not finished !
        break;

    case COVISE_MESSAGE_FEEDBACK:
        callFeedbackCallback(applMsg);
        break;

    case COVISE_MESSAGE_GENERIC:
        generic(applMsg);
        break;

    default:
        doCustom(applMsg);
        break;
    }
    appmod->delete_msg(applMsg);
}

//=====================================================================
//
//=====================================================================
void Covise::callStartCallback()
{
//#if defined __linux__ || defined _IA64
#ifdef TTEST
    struct tms tinfo;
    clock_t starttime;
    starttime = times(&tinfo);
#endif
    //#else
    //#ifdef TTEST
    //   struct timespec starttime;
    //   clock_gettime(BEST_TIMER,&starttime);
    //#endif
    //#endif

    (*startCallbackFunc)(startUserData, startCallbackData);

//#if defined __linux__ || defined _IA64
#ifdef TTEST
    clock_t diff, endtime;
    endtime = times(&tinfo);
    diff = endtime - starttime;
    cerr << diff / (double)CLK_TCK << " (s) for " << get_module() << "_" << get_instance() << endl;
#endif
    /*#else
   #ifdef TTEST
      struct timespec endtime;
      clock_gettime(BEST_TIMER,&endtime);
      double time_needed =          (endtime.tv_sec  - starttime.tv_sec )
                           + 1e-9 * (endtime.tv_nsec - starttime.tv_nsec);

      cerr << time_needed  << " (s) for " <<  get_module() << "_" << get_instance() << endl;
   #endif
   #endif*/
}

void
Covise::callAddObjectCallback()
{
    cerr << "Covise::callAddObjectCallback() called" << endl;

#ifdef TTEST
    struct tms tinfo;
    clock_t starttime;
    starttime = times(&tinfo);
#endif

    (*addObjectCallbackFunc)(addObjectUserData, addObjectCallbackData);

//#if defined __linux__ || defined _IA64
#ifdef TTEST
    clock_t diff, endtime;
    endtime = times(&tinfo);
    diff = endtime - starttime;
    cerr << diff / (double)CLK_TCK << " (s) for " << get_module() << "_" << get_instance() << endl;
#endif
}

//=====================================================================
//
//=====================================================================
void Covise::callFeedbackCallback(Message *msg)
{
    if (!msg)
        return;
    int sLen = (int)strlen(msg->data.data()) + 1;
    if (feedbackCallbackFunc)
        (*feedbackCallbackFunc)(feedbackUserData, msg->data.length() - sLen, msg->data.data() + sLen);
}

//=====================================================================
//
//=====================================================================
void Covise::callGenericCallback()
{
    (*genericCallbackFunc)(genericUserData, genericCallbackData);
}

//=====================================================================
//
//=====================================================================
void Covise::callSyncCallback()
{
    (*syncCallbackFunc)(syncUserData, syncCallbackData);
}

//=====================================================================
//
//=====================================================================
void Covise::callPortReplyCallback()
{
    (*portReplyCallbackFunc)(portReplyUserData, portReplyCallbackData);
}

//=====================================================================
//
//=====================================================================
void Covise::callAfterFinishCallback()
{
    (*afterFinishCallbackFunc)(afterFinishUserData, afterFinishCallbackData);
}

//=====================================================================
//
//=====================================================================
void Covise::callPipelineFinishCallback()
{
    (*pipelineFinishCallbackFunc)(pipelineFinishUserData, pipelineFinishCallbackData);
}

//=====================================================================
//
//=====================================================================
void Covise::set_start_callback(CoviseCallback *f, void *data)
{
    startCallbackFunc = f;
    startUserData = data;
    startCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void
Covise::set_add_object_callback(CoviseCallback *f, void *data)
{
    addObjectCallbackFunc = f;
    addObjectUserData = data;
    addObjectCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::set_feedback_callback(FeedbackCallback *f, void *data)
{
    feedbackCallbackFunc = f;
    feedbackUserData = data;
}

//=====================================================================
//
//=====================================================================
void Covise::set_generic_callback(CoviseCallback *f, void *data)
{
    genericCallbackFunc = f;
    genericUserData = data;
    genericCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::set_sync_callback(CoviseCallback *f, void *data)
{
    syncCallbackFunc = f;
    syncUserData = data;
    syncCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::set_port_callback(CoviseCallback *f, void *data)
{
    portReplyCallbackFunc = f;
    portReplyUserData = data;
    portReplyCallbackData = (void *)NULL;

    //cerr << "MODULE: I have set the callback to" << portReplyCallbackFunc << endl;
}

//=====================================================================
//
//=====================================================================
void Covise::set_after_finish_callback(CoviseCallback *f, void *data)
{
    afterFinishCallbackFunc = f;
    afterFinishUserData = data;
    afterFinishCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::set_pipeline_finish_callback(CoviseCallback *f, void *userData)
{

    pipelineFinishCallbackFunc = f;
    pipelineFinishUserData = userData;
    pipelineFinishCallbackData = (void *)NULL;

    if (pipeline_state_once == 0)
    {
        // inform Mapeditor that want to be called every time
        // a pipeline exection is finished

        if (appmod != NULL)
        {

            // build and send message
            Covise::send_ui_message("PIPELINE_STATE", "");

            // make sure this is done only once
            pipeline_state_once = 1;
        }
    }
}

//=====================================================================
//
//=====================================================================
void Covise::remove_start_callback(void)
{
    startCallbackFunc = (CoviseCallback *)NULL;
    startUserData = (void *)NULL;
    startCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::remove_feedback_callback(void)
{
    feedbackCallbackFunc = NULL;
    feedbackUserData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::remove_generic_callback(void)
{
    genericCallbackFunc = (CoviseCallback *)NULL;
    genericUserData = (void *)NULL;
    genericCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::remove_sync_callback(void)
{
    syncCallbackFunc = (CoviseCallback *)NULL;
    syncUserData = (void *)NULL;
    syncCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::remove_after_finish_callback(void)
{
    afterFinishCallbackFunc = (CoviseCallback *)NULL;
    afterFinishUserData = (void *)NULL;
    afterFinishCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void Covise::remove_pipeline_finish_callback(void)
{
    pipelineFinishCallbackFunc = (CoviseCallback *)NULL;
    pipelineFinishUserData = (void *)NULL;
    pipelineFinishCallbackData = (void *)NULL;
}

#ifdef COVISE_USE_X11

//=====================================================================
//
//=====================================================================
void Covise::socketCommunicationCB(XtPointer, int *, XtInputId *)
{

    do_one_event();
}
#endif
#ifndef _WIN32
//extern CoviseTime *covise_time;
#endif

//=====================================================================
//
//=====================================================================
void Covise::doStart(Message *m)
{
    if (!m)
        return;

    //    tmpptr = new char[100];
    //    sprintf(tmpptr, "module %s starts", ap->get_name());
    //    covise_time->mark(__LINE__, tmpptr);

    msg = new CtlMessage(m);

    //cerr << msg->data.data() << endl;

    // call back the function provided by the user
    if (startCallbackFunc != NULL)
        callStartCallback();

    msg->create_finall_message();

    //  cerr << "Sending message to controller :" << msg->data.data() << endl;

    appmod->send_ctl_msg((Message *)msg);

    delete msg;

    // call back the after finish function provided by the user
    if (afterFinishCallbackFunc != NULL)
    {
        // cerr << "Calling afterFinishCallback now" << endl;
        afterFinishCallbackData = NULL;
        callAfterFinishCallback();
    }
}

void
Covise::doAddObject()
{

    if ((objNameToAdd_ == NULL) && (objNameToDelete_ == NULL))
        return;

    // call back the function provided by the user
    if (addObjectCallbackFunc != NULL)
        (*addObjectCallbackFunc)(addObjectUserData, addObjectCallbackData);
}

void
Covise::sendFinishedMsg()
{
    char buf[100];

    if (appmod != NULL)
    {

        const char *key = "";
        strcpy(buf, key);
        strcat(buf, "\n");
        Message message{ COVISE_MESSAGE_FINISHED, DataHandle(buf, strlen(buf) + 1) };

        appmod->send_ctl_msg(&message);
        // print_comment( __LINE__ , __FILE__ , "sended finished message" );
    }
}

//=====================================================================
//
//=====================================================================
void Covise::doStartWithoutFinish(Message *m)
{

#ifdef TIMING
    char *tmpptr;
    tmpptr = new char[100];
    sprintf(tmpptr, "module %s starts", appmod->get_name());
//covise_time->mark(__LINE__, tmpptr);
#endif

    msg = new CtlMessage(m);

    //cerr << "[" << counter << "] CtlMessage has been newed *************" << endl;

    // call back the function provided by the user
    if (startCallbackFunc != NULL)
    {
        startCallbackData = (void *)msg;
        callStartCallback();
    }
}

//=====================================================================
//
//=====================================================================
void Covise::sendFinishedMsg(void *Msg)
{
    if (!Msg)
        return;

    if (appmod != NULL)
    {

        CtlMessage *msg = (CtlMessage *)Msg;

        msg->create_finall_message();

#ifdef TIMING
        char *tmpptr;
        tmpptr = new char[100];

        sprintf(tmpptr, "module %s is finished", appmod->get_name());
//covise_time->mark(__LINE__, tmpptr);
//covise_time->print();
#endif

        appmod->send_ctl_msg((Message *)msg);

        delete msg;

        // cerr << "[" << counter << "] CtlMessage has been deleted ############# " << endl;
        counter++;

#ifndef _WIN32
// stop_timer(0);
// print_timer(Covise::get_module(),Covise::get_instance(),Covise::get_host(),0,"Time for execution");
#endif
    }
}

//=====================================================================
//
//=====================================================================
void Covise::partobjects_initialized(void)
{
    if (!msg)
        return;

    if (appmod != NULL)
    {
        msg->create_finpart_message();
        msg->conn->sendMessage(msg);
    }
}

//=====================================================================
//
//=====================================================================
void Covise::doGeneric(Message *m)
{
    // call back the function provided by th user
    if (genericCallbackFunc != NULL)
    {
        genericCallbackData = (void *)m;
        callGenericCallback();
    }
}

//=====================================================================
//
//=====================================================================
void Covise::doSync(Message *m)
{
    // call back the function provided by th user
    if (syncCallbackFunc != NULL)
    {
        syncCallbackData = (void *)m;
        callSyncCallback();
    }
}

//=====================================================================
//
//=====================================================================
void Covise::doPortReply(Message *m)
{
    char *datacopy;

    if (reply_buffer) // delete old reply_buffer
    {
        delete[] reply_buffer;
        reply_buffer = NULL;
    }

    datacopy = new char[strlen(m->data.data()) + 1];
    strcpy(datacopy, m->data.data());

    char *p = datacopy;
    reply_port_name = strsep(&p, "\n");

    // call back the function provided by the user
    if (portReplyCallbackFunc != NULL)
    {
        portReplyCallbackData = (void *)m;
        callPortReplyCallback();
    }

    delete[] datacopy;
}

//=====================================================================
//
//=====================================================================
void Covise::doPipelineFinish()
{

    // call back the function provided by the user

    if (pipelineFinishCallbackFunc != NULL)
    {
        pipelineFinishCallbackData = NULL;
        callPipelineFinishCallback();
    }
}

//=====================================================================
//
//=====================================================================
int Covise::deleteConnection()
{
    if (appmod != NULL)
    {

#ifndef NODELETE_APPROC
        delete appmod;
#endif
        appmod = NULL;
        return 1;
    }
    else
        return 0;
}

void Covise::addInteractor(coDistributedObject *obj, const char *name, const char *value)
{
    char *buf = new char[strlen(name) + strlen(value) + strlen(Covise::get_module()) + strlen(Covise::get_host()) + 200];
    sprintf(buf, "X%s\n%s\n%s\n%s\n%s", Covise::get_module(), Covise::get_instance(), Covise::get_host(), name, value);
    obj->addAttribute("INTERACTOR", buf);
    delete[] buf;
}

//=====================================================================
//
//=====================================================================
int Covise::doQuit()
{
    int dont_quit;

    char data[4];

    // call back the function provided by the user
    dont_quit = 0;
    if (quitInfoCallbackFunc != NULL)
    {
        quitInfoCallbackData = (void *)&dont_quit;
        callQuitInfoCallback();
        dont_quit = *((int *)quitInfoCallbackData);
        if (dont_quit != 0)
            return 0;
        else
            return 1;
    }
    else if (quitCallbackFunc != NULL)
    {
        callQuitCallback();
        return 1;
    }

    sprintf(&data[0], "%s\n", " ");

    // build and send message
    Covise::send_message(COVISE_MESSAGE_QUIT, data);

    return 1;
}

//=====================================================================
//
//=====================================================================
void Covise::doParam(Message *m)
{

    if (reply_buffer) // delete old reply_buffer
    {
        delete[] reply_buffer;
        reply_buffer = NULL;
    }
    //cerr << " do Param " << endl  << m->data << endl;
    char *buf = new char[strlen(m->data.data()) + 1];
    char *p = buf;
    strcpy(p, m->data.data());

    reply_keyword = strsep(&p, "\n");
    if (strcmp(reply_keyword, "GETDESC") == 0)
    {
        char * d = get_description_message();
        Message message{ COVISE_MESSAGE_PARINFO , DataHandle(d, strlen(d) + 1)}; // should be a real type

        appmod->send_ctl_msg(&message);
    }
    else if ((strcmp(reply_keyword, "INEXEC") != 0) && (strcmp(reply_keyword, "FINISHED") != 0))
    {
        // See if its a pipeline state message...
        if (strcmp("PIPELINE_STATE", reply_keyword) == 0)
        {
            doPipelineFinish();
            // we're done
            delete[] p;
            return;
        }

        // skip next 3 parts of the message
        strsep(&p, "\n");
        strsep(&p, "\n");
        strsep(&p, "\n");

        reply_param_name = strsep(&p, "\n");
        reply_param_type = strsep(&p, "\n");

        // only PARAM messages
        if (strncmp("PARAM", reply_keyword, 5) == 0)
        {
            tokenlist.clear();
            string value = strsep(&p, "\n");
            if (!strcmp(reply_param_type, "FloatScalar") || !strcmp(reply_param_type, "FloatSlider") || !strcmp(reply_param_type, "FloatVector") || !strcmp(reply_param_type, "IntScalar") || !strcmp(reply_param_type, "IntSlider") || !strcmp(reply_param_type, "IntVector") || !strcmp(reply_param_type, "Color") || !strcmp(reply_param_type, "ColormapChoice") || !strcmp(reply_param_type, "Material") || !strcmp(reply_param_type, "Colormap"))
            {
                while (value[0] == ' ') // remove leading spaces
                    value.erase(0, 1);

                //cerr << " doParam ... get value   " << value << endl;

                int search = 0, first = 0;
                while (1)
                {
                    first = (int)value.find(" ", search);
                    if (first != -1)
                    {
                        string tmp = value.substr(search, first - search);
                        tokenlist.push_back(tmp);
                        search = first + 1;
                    }
                    else
                    {
                        if (search < value.length())
                        {
                            int ll = (int)value.length() - search;
                            string sub = value.substr(search, ll);
                            if (!sub.empty())
                                tokenlist.push_back(sub);
                        }
                        break;
                    }
                }

                no_of_reply_tokens = (int)tokenlist.size();
            }
            else if (!strcmp(reply_param_type, "Browser"))
            {
                no_of_reply_tokens = 1;
                tokenlist.push_back(value);
            }
            else
            {
                no_of_reply_tokens = 1;
                tokenlist.push_back(value);
            }
            reply_buffer = new const char *[no_of_reply_tokens];
            for (int i = 0; i < no_of_reply_tokens; i++)
            {
                reply_buffer[i] = tokenlist[i].c_str();
            }
        }

        // anything else
        else
        {
            no_of_reply_tokens = atoi(strsep(&p, "\n"));
            reply_buffer = new const char *[no_of_reply_tokens];
            for (int i = 0; i < no_of_reply_tokens; i++)
                reply_buffer[i] = strsep(&p, "\n");
        }
    }

    // call back the function provided by the user
    if ((paramCallbackFunc != NULL) && ((strncmp("PARAM", reply_keyword, 5) == 0) || !strncmp("MODULE_TITLE", reply_keyword, 12)))
    {
        bool inMapLoading = !strcmp(reply_keyword, "PARAM_INIT");
        paramCallbackData = (void *)m;
        callParamCallback(inMapLoading);
    }
    delete[] buf;
}
