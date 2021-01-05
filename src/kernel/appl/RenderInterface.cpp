/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Interface class for renderer modules to the COVISE        **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C)1995 RUS                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:   11.09.95  V1.0                                                 **
 ** Author: Dirk Rantzau                                                   **
\**************************************************************************/
#include "RenderInterface.h"
#include <covise/covise.h>
#include <covise/covise_appproc.h>
#include <util/coTimer.h>
#include <covise/Covise_Util.h>
#include <util/coLog.h>
#include <do/coDistributedObject.h>
#include <net/dataHandle.h>
#include <comsg/CRB_EXEC.h>

#ifdef _WIN32
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#endif

using namespace covise;

//==========================================================================
// definition of static class elements
//==========================================================================

// static int counter = 0;

int CoviseRender::replaceObject = 0;
char *CoviseRender::object_name = NULL;
char *CoviseRender::render_keyword = NULL;
char *CoviseRender::render_data = NULL;
CoviseCallback *CoviseRender::renderCallbackFunc = NULL;
voidFuncintvoidpDef *CoviseRender::renderModuleCallbackFunc = NULL;
CoviseCallback *CoviseRender::addObjectCallbackFunc = NULL;
CoviseCallback *CoviseRender::coviseErrorCallbackFunc = NULL;
CoviseCallback *CoviseRender::deleteObjectCallbackFunc = NULL;
void *CoviseRender::renderUserData = 0L;
void *CoviseRender::addObjectUserData = 0L;
void *CoviseRender::coviseErrorUserData = 0L;
void *CoviseRender::deleteObjectUserData = 0L;
void *CoviseRender::renderCallbackData = 0L;
void *CoviseRender::addObjectCallbackData = 0L;
void *CoviseRender::coviseErrorCallbackData = 0L;
void *CoviseRender::deleteObjectCallbackData = 0L;

#ifdef COVISE_Signals
#ifdef __linux__
#define SignalHandler CO_SignalHandler
#endif
static SignalHandler sig_handler;
#endif

/////////////////////////////////////////////////////////////////////////
// This struct stores name replacements: Whenever an empty object
// should replace another object, its name is saved here

struct Repl
{
    char *oldName, *newName;
    struct Repl *next;
    Repl(const char *oldN, const char *newN)
    {
        oldName = strcpy(new char[strlen(oldN) + 1], oldN);
        newName = strcpy(new char[strlen(newN) + 1], newN);
        next = NULL;
    }
};

static Repl dummy("", "");
static Repl *coReplace = &dummy;

// enter a replacement into list, replace old replacement if exist
/*static void setReplace(char  *oldName, char  *newName)
{
   Repl *repl = coReplace->next;   // 1st is dummy
   Repl *orep = coReplace;
   while (repl && strcmp(repl->newName,oldName)!=0)
   {
      orep = repl;
      repl = repl->next;
   }
   if (repl)   // replace more than the first time
   {
delete [] repl->newName;
repl->newName = strcpy ( new char [strlen(newName) + 1], newName);
}
else
orep->next = new Repl( oldName, newName);

cerr << "\n\nReplace list:" << endl;
repl = coReplace->next;
while (repl)
{
cerr << "   " << repl->oldName << "  -->  " << repl->newName << endl;
repl = repl->next;
}
cerr << "\n" << endl;
}*/

// check replacement: if exist: perform and remove from list
static void doReplace(char *&name)
{
    char *buffer;
    Repl *repl = coReplace->next; // 1st is dummy
    Repl *orep = coReplace;
    while (repl && strcmp(repl->newName, name) != 0)
    {
        orep = repl;
        repl = repl->next;
    }
    if (repl)
    {
        //cerr << "Using old " << repl->oldName << " instead " << object << endl;
        buffer = new char[strlen(repl->oldName) + 1]; // leaks memory!!!
        strcpy(buffer, repl->oldName);
        delete[] repl -> oldName;
        delete[] repl -> newName;
        orep->next = repl->next;
        delete repl;
        name = buffer;
    }
}

/////////////////////////////////////////////////////////////////////////

void CoviseRender::set_module_description(const char *descr)
{
    module_description = (char *)descr;
}

void CoviseRender::remove_ports()
{

    port_name[0] = 0;
}

void CoviseRender::add_port(enum appl_port_type type, const char *name)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while (port_name[i])
            i++;
        port_type[i] = type;
        port_name[i] = (char *)name;
        port_default[i] = NULL;
        port_datatype[i] = NULL;
        port_dependency[i] = NULL;
        port_required[i] = 1;
        port_description[i] = NULL;
        port_name[i + 1] = NULL;
    }
    else
    {
        cerr << "wrong description type in add_port " << name << "\n";
        return;
    }
}

void CoviseRender::add_port(enum appl_port_type type, const char *name, const char *dt, const char *descr)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while (port_name[i])
            i++;
        port_type[i] = type;
        port_name[i] = (char *)name;
        port_default[i] = NULL;
        port_datatype[i] = (char *)dt;
        port_dependency[i] = NULL;
        port_required[i] = 1;
        port_description[i] = (char *)descr;
        port_name[i + 1] = NULL;
    }
    else
    {
        cerr << "wrong description type in add_port " << name << "\n";
        return;
    }
}

void CoviseRender::set_port_description(char *name, char *descr)
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
        cerr << "wrong portname " << name << " in set_port_description\n";
        return;
    }
    port_description[i] = descr;
}

void CoviseRender::set_port_default(const char *name, const char *def)
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
    port_default[i] = (char *)def;
}

void CoviseRender::set_port_datatype(char *name, char *dt)
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
        cerr << "wrong portname " << name << " in set_port_datatype\n";
        return;
    }
    port_datatype[i] = dt;
}

void CoviseRender::set_port_required(char *name, int req)
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
    port_required[i] = req;
}

char *CoviseRender::get_description_message()
{
    CharBuffer msg(400);
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
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
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
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
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
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_default[i] == NULL)
            {
                cerr << "no default value for parameter " << port_name[i] << "\n";
                msg += "";
            }
            else
            {
                msg += port_default[i];
            }
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
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_default[i] == NULL)
            {
                cerr << "no default value for parameter " << port_name[i] << "\n";
            }
            msg += port_default[i];
            msg += '\n';
        }
        i++;
    }

    return msg.return_data();
#if 0
   char *buf=msg;
   msg.keep_data();
   return(buf);
#endif
}

#ifdef COVISE_USE_X11
//=====================================================================
//
//=====================================================================
void CoviseRender::init(int argc, char *argv[], XtAppContext appcon)
{

    //cerr << "(module: RenderInterface) CoviseRender::init(..) called"
    //	   << endl;

    if ((argc < 7) || (argc > 8))
    {
        if (argc == 2 && 0 == strcmp(argv[1], "-d"))
            printDesc(argv[0]);
        else
            cerr << "+++Render Module with inappropriate arguments called\n";
        exit(0);
    }

    // Initialization of the communciation environment

    appmod = new ApplicationProcess(argv[0], argc, argv, RENDERER);
    socket_id = appmod->get_socket_id(CoviseRender::remove_socket);
    h_name = (char *)appmod->get_hostname();
    m_name = argv[0];
    instance = argv[4];
    appContext = appcon;

    print_comment(__LINE__, __FILE__, "Xt Render Module succeeded");
#ifdef DEBUG
    cerr << argv[0] << "Render Module succeeded" << endl;
#endif
    init_flag = 1;

#ifdef COVISE_USE_X11
    //
    // add X input
    //
    X_id = XtAppAddInput(appContext, socket_id,
                         XtPointer(XtInputReadMask),
                         (XtInputCallbackProc)CoviseRender::socketCommunicationCB,
                         NULL);
#endif

#ifdef COVISE_Signals
    // Initialization of signal handlers
    sig_handler.addSignal(SIGBUS, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGPIPE, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGTERM, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGSEGV, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGFPE, (void *)signal_handler, NULL);
#endif
    init_emergency_message();
    Message *message = new Message();

    message->data = get_description_message();
    message->type = COVISE_MESSAGE_PARINFO; // should be a real type
    message->length = strlen(message->data) + 1;

    appmod->send_ctl_msg(message);
    delete[] message -> data;
    delete message;
}

#else
//=====================================================================
//
//=====================================================================
void CoviseRender::init(int argc, char *argv[])
{
    //cerr << "(module: RenderInterface) CoviseRender::init(..2..) called"
    //	   << endl;
    if (argc == 2 && 0 == strcmp(argv[1], "-d"))
        printDesc(argv[0]);


    // Initialization of the communciation environment

    appmod = new ApplicationProcess(argv[0], argc, argv, RENDERER);
    socket_id = appmod->get_socket_id(CoviseRender::remove_socket);
    h_name = (char *)appmod->get_hostname();
    auto crbExec = covise::getExecFromCmdArgs(argc, argv);
    m_name =crbExec.name;
    instance = crbExec.moduleId;

    print_comment(__LINE__, __FILE__, "Renderer Module succeeded");

#ifdef DEBUG
    cerr << argv[0] << "Render Module succeeded" << endl;
#endif

    init_flag = 1;

#ifdef COVISE_Signals
    // Initialization of signal handlers
    sig_handler.addSignal(SIGBUS, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGPIPE, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGTERM, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGSEGV, (void *)signal_handler, NULL);
    sig_handler.addSignal(SIGFPE, (void *)signal_handler, NULL);
#endif
    init_emergency_message();

    char * d = get_description_message();
    Message message{ COVISE_MESSAGE_PARINFO, DataHandle(d, strlen(d) + 1) }; // should be a real type
    if (appmod)
        appmod->send_ctl_msg(&message);
}
#endif

//=====================================================================
//
//=====================================================================
int CoviseRender::send_render_message(const char *keyword, const char *string)
{
    int size;

    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {



        size = 1; // final '\0'
        size += (int)strlen(keyword) + 1;
        size += (int)strlen(string) + 1;

        Message message{ COVISE_MESSAGE_RENDER , DataHandle(size)};

        strcpy(&(message.data.accessData()[0]), keyword);
        strcat(message.data.accessData(), "\n");
        strcat(message.data.accessData(), string);

        //cerr << "RENDER SENDING MESSAGE TO SLAVES : " << message->data << endl;
        message.data.setLength(strlen(message.data.data()) + 1);

        if (appmod)
            appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "Cannot send message without instance/init before");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
int CoviseRender::send_render_binmessage(const char *keyword, const char *data, int len)
{
    int size;

    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {


        size = (int)strlen(keyword) + 2;
        size += len;

        Message message{ COVISE_MESSAGE_RENDER , DataHandle(size)};
        message.data.accessData()[0] = 0;
        strcpy(&message.data.accessData()[1], keyword);
        memcpy(&message.data.accessData()[strlen(keyword) + 2], data, len);
        //cerr << "RENDER SENDING MESSAGE TO SLAVES : " << message->data << endl;

        if (appmod)
            appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "Cannot send message without instance/init before");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::send_quit_message()
{
    if (init_flag == 0)
    {
        print_comment(__LINE__, __FILE__, "CoviseRender::send_message : init not called before");
        return;
    }

    char *string = new char[strlen("QUIT") + 1];
    strcpy(string, "QUIT");
    CoviseRender::send_message(COVISE_MESSAGE_QUIT, string);
}

//=====================================================================
//
//=====================================================================
void CoviseRender::main_loop()
{
    while (1)
    {
        if (appmod)
            applMsg = appmod->wait_for_ctl_msg();
        handleControllerMessage();
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::progress_main_loop()
{
    while (1)
    {
        if (appmod)
            applMsg = appmod->check_for_ctl_msg();
        if (applMsg == NULL)
            doProgress();
        else
            handleControllerMessage();
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::do_one_event()
{
    if (appmod)
        applMsg = appmod->check_for_ctl_msg(); // infinit wait
    handleControllerMessage(); // use global applMsg
}

int CoviseRender::check_and_handle_event(float time)
{
	covise::Message* msg = check_event(time);
	
	if (!msg)
        return 0;

	handle_event(msg);

    return 1;
}
covise::Message* CoviseRender::check_event(float time)
{

	if (!appmod)
		return nullptr;

	covise::Message* msg = appmod->check_for_ctl_msg(time);
	return msg;
}
void CoviseRender::handle_event(covise::Message *msg)
{
	if (!msg)
	{
		return;
	}
	applMsg = msg;
	handleControllerMessage(); /// use global applMsg
	applMsg = nullptr;
}

//=====================================================================
// time=0.0 means infinit time
//=====================================================================
void CoviseRender::handleControllerMessage()
{
    MARK0("COVER handling controller message");
    int quit_now;
    //int ntok;
    char *token[MAXTOKENS];
    const char *sep = "\n";

    token[0] = NULL;
    token[1] = NULL;
    object_name = NULL;

    switch (applMsg->type)
    {
    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_UI:
        print_comment(__LINE__, __FILE__, "CoviseRender::check_and_handle_event applMsg->type = UI\n");
        doParam(applMsg);
        reply_param_name = NULL;
        break;

    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_QUIT:
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
        print_comment(__LINE__, __FILE__, "Renderer pid %d is requested to exit", getpid());
        quit_now = doQuit();
        if (appmod)
            appmod->delete_msg(applMsg);
        delete appmod;
        appmod = NULL;
        applMsg = NULL;
        if (quit_now) // otherwise module exits later itself
        {
            print_comment(__LINE__, __FILE__, "Render module: correctly finishing");
            exit(0);
        }
        break;

    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_RENDER:
        if (applMsg->data.data()[0] != 0)
        {
            char *buffer = new char[strlen(applMsg->data.data()) + 1];
            strcpy(buffer, applMsg->data.data());
            parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);
            if (strcmp(token[0], "MASTER") == 0)
            {
                doMasterSwitch();
            }
            else if (strcmp(token[0], "SLAVE") == 0)
            {
                doSlaveSwitch();
            }
            else
            {
                char *pos = strchr(buffer, '\n');
                strcpy(&applMsg->data.accessData()[strlen(token[0]) + 1], pos + 1);
                doRender(token[0], &applMsg->data.accessData()[strlen(token[0]) + 1]);
            }
        }
        else
        {
            doRender(&applMsg->data.accessData()[1], &applMsg->data.accessData()[strlen(&applMsg->data.accessData()[1]) + 2]);
        }
        break;

    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_RENDER_MODULE:
        doRenderModule(applMsg->data);
        break;

    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_ADD_OBJECT:
    {
        addObjectCallbackData = (void *)applMsg;
        parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);

        // AW: try to create the object, so we know whether it is empty
        MARK0("COVER ADD_OBJECT get object shm address");

        const coDistributedObject *dataObj = coDistributedObject::createFromShm(coObjInfo(token[0]));
        // only add existing objects: ignore NULL objects here
        //if (data_obj)
        doAddObject(dataObj, token[0]);

        sendFinishedMsg();
        break;
    }

    case COVISE_MESSAGE_COVISE_ERROR:
    {
        coviseErrorCallbackData = (void *)applMsg->data.data();
        doCoviseError(applMsg->data.data());
        break;
    }

    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_DELETE_OBJECT:
    {
        MARK0("COVER DELETE_OBJECT");

        deleteObjectCallbackData = (void *)applMsg;
        parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);
        replaceObject = false;
        doReplace(token[0]); // aw: name may be replaced: NULL obj handling
        doDeleteObject(token[0]);
        sendFinishedMsg();
        break;
    }

    ///////////////////////////////////////////////////////////////////////
    case COVISE_MESSAGE_REPLACE_OBJECT:
    {
        parseMessage(applMsg->data.accessData(), &token[0], MAXTOKENS, sep);
        if (token[0] == NULL || token[1] == NULL)
            print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
        else
        {
            MARK0("COVER REPLACE_OBJECT get shm address");

            // AW: try to create the object, so we know whether it is empty
            const coDistributedObject *data_obj = coDistributedObject::createFromShm(coObjInfo(token[1]));

            // only add existing objects:  NULL objects go to replacement table
            //if (data_obj)
            //{
            // delete and add
            deleteObjectCallbackData = (void *)applMsg;
            replaceObject = true;
            doReplace(token[0]); // aw: name may be replaced: NULL obj handling
            doDeleteObject(token[0]);
            addObjectCallbackData = (void *)applMsg;
            doAddObject(data_obj, token[1]);
            //}
            //else
            // setReplace(token[0],token[1]);
        }
        sendFinishedMsg();
        break;
    }

    ///////////////////////////////////////////////////////////////////////
    default:
        doCustom(applMsg);
        break;

    } // Message type switch

    if (applMsg && appmod)
    {
        appmod->delete_msg(applMsg);
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::callAddObjectCallback()
{
    (*addObjectCallbackFunc)(addObjectUserData, addObjectCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseRender::callCoviseErrorCallback()
{
    (*coviseErrorCallbackFunc)(coviseErrorUserData, coviseErrorCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseRender::callDeleteObjectCallback()
{
    (*deleteObjectCallbackFunc)(deleteObjectUserData, deleteObjectCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseRender::callRenderCallback()
{
    (*renderCallbackFunc)(renderUserData, renderCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseRender::set_add_object_callback(CoviseCallback *f, void *data)
{
    addObjectCallbackFunc = f;
    addObjectUserData = data;
    addObjectCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::set_covise_error_callback(CoviseCallback *f, void *data)
{
    coviseErrorCallbackFunc = f;
    coviseErrorUserData = data;
    coviseErrorCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::set_delete_object_callback(CoviseCallback *f, void *data)
{
    deleteObjectCallbackFunc = f;
    deleteObjectUserData = data;
    deleteObjectCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::set_render_callback(CoviseCallback *f, void *data)
{
    renderCallbackFunc = f;
    renderUserData = data;
    renderCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::set_render_module_callback(voidFuncintvoidpDef *f)
{
    renderModuleCallbackFunc = f;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::remove_add_object_callback(void)
{
    addObjectCallbackFunc = (CoviseCallback *)NULL;
    addObjectUserData = (void *)NULL;
    addObjectCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::remove_covise_error_callback(void)
{
    coviseErrorCallbackFunc = (CoviseCallback *)NULL;
    coviseErrorUserData = (void *)NULL;
    coviseErrorCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::remove_delete_object_callback(void)
{
    deleteObjectCallbackFunc = (CoviseCallback *)NULL;
    deleteObjectUserData = (void *)NULL;
    deleteObjectCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::remove_render_callback(void)
{
    renderCallbackFunc = (CoviseCallback *)NULL;
    renderUserData = (void *)NULL;
    renderCallbackData = (void *)NULL;
}

#ifdef COVISE_USE_X11

//=====================================================================
//
//=====================================================================
void CoviseRender::socketCommunicationCB(XtPointer, int *, XtInputId *)
{

    do_one_event();
}
#endif

//extern CoviseTime *covise_time;

//=====================================================================
//
//=====================================================================
void CoviseRender::doRender(char *key, char *data)
{

    // call back the function provided by the user
    if (renderCallbackFunc != NULL)
    {
        render_keyword = key;
        render_data = data;
        callRenderCallback();
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::doRenderModule(const DataHandle& dh)
{

    // call back the function provided by the user
    if (renderModuleCallbackFunc != NULL)
    {
        (*renderModuleCallbackFunc)(dh);
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::sendFinishedMsg()
{
    if (appmod != NULL)
    {

   
        const char *key = "";
        char* buf = new char[100];
        strcpy(buf, key);
        strcat(buf, "\n");
        Message msg{ COVISE_MESSAGE_FINISHED, DataHandle(buf, strlen(buf) + 1) };

        appmod->send_ctl_msg(&msg);
        // print_comment( __LINE__ , __FILE__ , "sended finished message" );
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::doAddObject(const coDistributedObject *obj, char *name)
{
    MARK0("COVER ADD_OBJECT adding object to the scenegraph");

    // call back the function provided by the user
    if (addObjectCallbackFunc != NULL)
    {
        struct
        {
            const coDistributedObject *obj;
            char *name;
        } cbData;
        cbData.name = name;
        cbData.obj = obj;
        object_name = name;
        addObjectCallbackData = (void *)&cbData;
        callAddObjectCallback();
    }
    MARK0("done");
}

//=====================================================================
//
//=====================================================================
void CoviseRender::doCoviseError(const char *error)
{
    // call back the function provided by the user
    if (coviseErrorCallbackFunc != NULL)
    {
        coviseErrorCallbackData = (void *)error;
        callCoviseErrorCallback();
    }
}

//=====================================================================
//
//=====================================================================
void CoviseRender::doDeleteObject(char *name)
{
    MARK0("COVER DELETE_OBJECT from scenegraph");

    // call back the function provided by the user
    if (deleteObjectCallbackFunc != NULL)
    {
        object_name = name;
        deleteObjectCallbackData = (void *)name;
        callDeleteObjectCallback();
    }
    MARK0("done");
}

//=====================================================================
// because a repace doesn't exist any more
//=====================================================================

int CoviseRender::isReplace()
{
    return replaceObject;
}

//=====================================================================
//
//=====================================================================
int CoviseRender::deleteConnection()
{
    if (appmod != NULL)
    {

        delete appmod;
        appmod = NULL;
        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseRender::doQuit()
{
    int dont_quit;

    // inform controller that we are going to exit
    /********************************************************************
      char data[4];
      sprintf(&data[0], "%s\n"," ");
      CoviseRender::send_message(QUIT,data);
   ********************************************************************/
    // call back the cleanup functions provided by the user
    // there are two versions:
    //
    // [1] quitInfoCallback : user decides if exit is called afterwards
    // [2] quitCallback     : exit is called afterwards
    //
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

    return 1;
}

//=====================================================================
//
//=====================================================================
void CoviseRender::doParam(Message *m)
{
    char *datacopy;

    if (reply_buffer) // delete old reply_buffer
    {
        delete[] reply_buffer;
        reply_buffer = NULL;
    }

    datacopy = new char[strlen(m->data.data()) + 1];
    strcpy(datacopy, m->data.data());

    char *p = m->data.accessData();
    reply_keyword = strsep(&p, "\n");

    if ((strcmp(reply_keyword, "INEXEC") == 0) || (strcmp(reply_keyword, "FINISHED") == 0))
    {
        doRender(reply_keyword, p);
    }
    else
    {

        // skip next 3 parts of the message
        strsep(&p, "\n");
        strsep(&p, "\n");
        strsep(&p, "\n");

        reply_param_name = strsep(&p, "\n");
        reply_param_type = strsep(&p, "\n");

        // only PARAM messages
        if (strncmp("PARAM", reply_keyword, 5) == 0 && p != NULL) // p is NULL for PARAM_INIT messages--> value is undefined
        {
            string value = strsep(&p, "\n");
            if (value.length() > 0)
            {
                while (value[0] == ' ') // remove leading spaces
                    value.erase(0, 1);

                //cerr << " doParam ... get value   " << value << endl;

                int search = 0, first = 0, i = 0;
                tokenlist.clear();
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
                reply_buffer = new const char *[no_of_reply_tokens];
                for (i = 0; i < no_of_reply_tokens; i++)
                {
                    reply_buffer[i] = tokenlist[i].c_str();
                    //if(strcmp(reply_param_type, "Browser") == 0)
                    //cerr << "........... reply_buffer:" << i << "  " << reply_buffer[i]  <<endl;
                }
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
    delete[] datacopy;
}
