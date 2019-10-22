/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPL_INTERFACE_H
#define _APPL_INTERFACE_H

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

#include "CoviseBase.h"

namespace covise
{
class DataHandle;
typedef void(voidFuncintvoidpDef)(const DataHandle &dh);

//=====================================================================
//
//=====================================================================
class APPLEXPORT CoviseRender : public CoviseBase
{
private:
    static int replaceObject; //

    // Communication Data
    static char *render_keyword;
    static char *render_data;
    static char *object_name;

    // private member funcs
    static void doParam(Message *m);
    static void doRender(char *key, char *data);
    static void doRenderModule(const DataHandle &dh);
    static void doAddObject(const coDistributedObject *, char *name);
    static void doCoviseError(const char *error);
    static void doDeleteObject(char *name);
    static int doQuit();

    //
    // callback stuff
    //
    static voidFuncintvoidpDef *renderModuleCallbackFunc;
    static CoviseCallback *renderCallbackFunc;
    static void *renderUserData;
    static void *renderCallbackData;
    static CoviseCallback *addObjectCallbackFunc;
    static void *addObjectUserData;
    static void *addObjectCallbackData;
    static CoviseCallback *coviseErrorCallbackFunc;
    static void *coviseErrorUserData;
    static void *coviseErrorCallbackData;
    static CoviseCallback *deleteObjectCallbackFunc;
    static void *deleteObjectUserData;
    static void *deleteObjectCallbackData;

public:
    //
    // description stuff
    //
    static void set_module_description(const char *);
    static void add_port(enum appl_port_type, const char *);
    static void add_port(enum appl_port_type, const char *, const char *, const char *);
    static void remove_ports();
    static void set_port_description(char *, char *);
    static void set_port_default(const char *, const char *);
    static void set_port_datatype(char *, char *);
    static void set_port_dependency(char *, char *);
    static void set_port_required(char *, int);
    static char *get_description_message();
//
// Callback and control
//
#ifdef COVISE_USE_X11
    static void init(int argc, char *argv[], XtAppContext appContext);
#else
    static void init(int argc, char *argv[]);
#endif
    static void reset()
    {
        CoviseRender::port_name[0] = NULL;
    };

    static void handleControllerMessage();

    static void sendFinishedMsg();
    static int check_and_handle_event(float time = 0.0);
	static covise::Message* check_event(float time = 0.0);
	static void handle_event(covise::Message* msg);
    static void do_one_event();
    static void ReceiveOneMsg();
    static int deleteConnection();
    static void main_loop();
    static void progress_main_loop();
    static void set_render_callback(CoviseCallback *f, void *userData);
    static void set_render_module_callback(voidFuncintvoidpDef *f);
    static void set_add_object_callback(CoviseCallback *f, void *userData);
    static void set_covise_error_callback(CoviseCallback *f, void *userData);
    static void set_delete_object_callback(CoviseCallback *f, void *userData);
    static void remove_render_callback();
    static void remove_add_object_callback();
    static void remove_covise_error_callback();
    static void remove_delete_object_callback();
    static int isReplace();

    //
    // Object name handling
    //
    static char *get_object_name()
    {
        return object_name;
    }
    //
    // Render message components retrieving
    //
    static char *get_render_keyword()
    {
        return render_keyword;
    }
    static char *get_render_data()
    {
        return render_data;
    }
    //
    // Render message setting
    //
    static void set_applMsg(Message *new_msg)
    {
        applMsg = new_msg;
    };
    //
    // Render message retrieving
    //
    static Message *get_applMsg()
    {
        return applMsg;
    }

    //
    // Messages to COVISE environment
    //
    static int send_render_message(const char *keyword, const char *string);
    static int send_render_binmessage(const char *keyword, const char *data, int len);
    static void send_quit_message();
    //
    // The callback functions called in callXXX  has to be
    // implemented by the user
    //
    static void callRenderCallback(void);
    static void callAddObjectCallback(void);
    static void callCoviseErrorCallback(void);
    static void callDeleteObjectCallback(void);
};
}
#endif
