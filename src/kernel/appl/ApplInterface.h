/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPL_INTERFACE_H
#define _APPL_INTERFACE_H

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
 **   	 25.6.97    V1.1 Harald Nebel, added GENERIC-stuff                                                                    **
\**************************************************************************/

#include "CoviseBase.h"

#include <covise/covise_msg.h>

#define COMPAT452

namespace covise
{

//=====================================================================
// Covise callbacks
//=====================================================================
typedef void FeedbackCallback(void *userData, int len, const char *data);

//=====================================================================
//
//=====================================================================
class APPLEXPORT Covise : public CoviseBase
{
private:
    // Communication Data
    static CtlMessage *msg;
    static char *reply_port_name;

    static char *modkey;

    static int renderMode_;

    static int pipeline_state_once;

    // private member funcs
    static void doParam(Message *m);
    static void doPortReply(Message *m);
    static void doPipelineFinish();

    static void generic(Message *applMsg);

    static void doStartWithoutFinish(Message *m);
    static void doStart(Message *m);
    static void doAddObject();
    static void doGeneric(Message *m);
    static void doSync(Message *m);
    static int doQuit();

    //
    // callback stuff
    //
    static FeedbackCallback *feedbackCallbackFunc;
    static void *feedbackUserData;

    static CoviseCallback *startCallbackFunc;
    static void *startUserData;
    static void *startCallbackData;

    static CoviseCallback *addObjectCallbackFunc;
    static void *addObjectUserData;
    static void *addObjectCallbackData;

    static CoviseCallback *genericCallbackFunc;
    static void *genericUserData;
    static void *genericCallbackData;
    static char *genericMessageData;

    static CoviseCallback *syncCallbackFunc;
    static void *syncUserData;
    static void *syncCallbackData;

    static CoviseCallback *portReplyCallbackFunc;
    static void *portReplyUserData;
    static void *portReplyCallbackData;
    static CoviseCallback *afterFinishCallbackFunc;
    static void *afterFinishUserData;
    static void *afterFinishCallbackData;
    static CoviseCallback *pipelineFinishCallbackFunc;
    static void *pipelineFinishUserData;
    static void *pipelineFinishCallbackData;

    static char *objNameToAdd_;
    static char *objNameToDelete_;

public:
    /// set/get the module description
    static void set_module_description(const char *desc);
    static const char *get_module_description();

    /// create a new port
    static void add_port(enum appl_port_type type, const char *name,
                         const char *datatype, const char *description);

    /// set a port's default value
    static void set_port_default(const char *name, const char *defVal);

    /// declare this (input) port dependent from another (output) port
    static void set_port_dependency(const char *port, char *depPort);

    /// tell that this port is (not) required
    static void set_port_required(const char *name, int isReqired);

    static char *get_description_message();


    /// check whether port is connected - use in compute() CB only
    static int is_port_connected(const char *name)
    {
        return msg ? msg->is_port_connected(name) : 0;
    }

    static char *getObjNameToAdd()
    {
        return objNameToAdd_;
    };
    static char *getObjNameToDelete()
    {
        return objNameToDelete_;
    };

/*  old stuff: not supported any further
      static void add_port(enum appl_port_type, char *,char *, char *);
      static void set_port_description(char *, char *);
      static void set_port_datatype(char *, char *);
      static void set_port_required(char *, int);
      static void add_port(enum appl_port_type type, const char *name);
      */

//
// Callback and control
//
#ifdef COVISE_USE_X11
    static void init(int argc, char *argv[], XtAppContext appContext);
#else
    static void init(int argc, char *argv[]);
#endif
    static void sendFinishedMsg(void *msg);
    static int check_and_handle_event(float time = 0.0);
    static void do_one_event();
    static void ReceiveOneMsg();
    static int deleteConnection();
    static void main_loop();
    static void progress_main_loop();
    static void set_start_callback(CoviseCallback *f, void *userData);
    static void set_add_object_callback(CoviseCallback *f, void *userData);

    static void set_feedback_callback(FeedbackCallback *f, void *userData);

    static void set_generic_callback(CoviseCallback *f, void *userData);
    static void set_sync_callback(CoviseCallback *f, void *userData);

    static void set_port_callback(CoviseCallback *f, void *userData);
    static void set_after_finish_callback(CoviseCallback *f, void *userData);
    static void set_pipeline_finish_callback(CoviseCallback *f, void *userData);
    static void remove_start_callback();
    static void remove_feedback_callback();

    static void remove_generic_callback();
    static void remove_sync_callback();

    static void remove_after_finish_callback();
    static void remove_pipeline_finish_callback();

    static char *get_reply_port_name()
    {
        return reply_port_name;
    }
    //
    // Object name handling
    //
    static char *get_object_name(const char *name)
    {
        return msg ? msg->get_object_name(name) : NULL;
    }

    static char *getObjectType(const char *name)
    {
        return msg ? msg->getObjectType(name) : NULL;
    }

    //
    // Parameter setting, retrieving and updating
    //
    static int get_scalar_param(const char *name, long *val)
    {
        return msg ? msg->get_scalar_param(name, val) : 0;
    }

    static int get_scalar_param(const char *name, float *val)
    {
        return msg ? msg->get_scalar_param(name, val) : 0;
    }

    static int get_vector_param(const char *name, int pos, long *list)
    {
        return msg ? msg->get_vector_param(name, pos, list) : 0;
    }

    static int get_vector_param(const char *name, int pos, float *list)
    {
        return msg ? msg->get_vector_param(name, pos, list) : 0;
    }

    static int get_string_param(const char *name, char **string)
    {
        return msg ? msg->get_string_param(name, string) : 0;
    }

    static int get_browser_param(const char *name, char **file)
    {
        return msg ? msg->get_browser_param(name, file) : 0;
    }

    static int get_boolean_param(const char *name, int *b)
    {
        return msg ? msg->get_boolean_param(name, b) : 0;
    }

    static int get_slider_param(const char *name, long *min, long *max, long *val)
    {
        return msg ? msg->get_slider_param(name, min, max, val) : 0;
    }

    static int get_slider_param(const char *name, float *min, float *max, float *val)
    {
        return msg ? msg->get_slider_param(name, min, max, val) : 0;
    }

    static int get_text_param(const char *name, char ***text, int *linenum)
    {
        return msg ? msg->get_text_param(name, text, linenum) : 0;
    }

    static int get_timer_param(const char *name, long *start, long *delta, long *state)
    {
        return msg ? msg->get_timer_param(name, start, delta, state) : 0;
    }

    static int get_passwd_param(const char *name, char **host, char **user,
                                char **passwd)
    {
        return msg ? msg->get_passwd_param(name, host, user, passwd) : 0;
    }

    static int get_choice_param(const char *name, int *pos)
    {
        return msg ? msg->get_choice_param(name, pos) : 0;
    }

    static int get_choice_param(const char *name, char **string)
    {
        return msg ? msg->get_choice_param(name, string) : 0;
    }

    static int get_material_param(const char *name, char **string)
    {
        return msg ? msg->get_material_param(name, string) : 0;
    }

    static int get_colormapchoice_param(const char *name, int *pos)
    {
        return msg ? msg->get_colormapchoice_param(name, pos) : 0;
    }

    static int get_colormapchoice_param(const char *name, char **string)
    {
        return msg ? msg->get_colormapchoice_param(name, string) : 0;
    }

    static int get_colormap_param(const char *name, float *min, float *max, int *len, colormap_type *type)
    {
        return msg ? msg->get_colormap_param(name, min, max, len, type) : 0;
    }

    static int get_color_param(const char *name, float *r, float *g, float *b, float *a)
    {
        return msg ? msg->get_color_param(name, r, g, b, a) : 0;
    }

    static int set_scalar_param(const char *name, long val)
    {
        return msg ? msg->set_scalar_param(name, val) : 0;
    }

    static int set_scalar_param(const char *name, float val)
    {
        return msg ? msg->set_scalar_param(name, val) : 0;
    }

    static int set_vector_param(const char *name, int num, long *list)
    {
        return msg ? msg->set_vector_param(name, num, list) : 0;
    }

    static int set_vector_param(const char *name, int num, float *list)
    {
        return msg ? msg->set_vector_param(name, num, list) : 0;
    }

    static int set_string_param(const char *name, char *string)
    {
        return msg ? msg->set_string_param(name, string) : 0;
    }

    static int set_boolean_param(const char *name, int val)
    {
        return msg ? msg->set_boolean_param(name, val) : 0;
    }

    static int set_slider_param(const char *name, long min, long max, long val)
    {
        return msg ? msg->set_slider_param(name, min, max, val) : 0;
    }

    static int set_slider_param(const char *name, float min, float max, float val)
    {
        return msg ? msg->set_slider_param(name, min, max, val) : 0;
    }

    static int set_choice_param(const char *name, int len, char **list, int pos)
    {
        return msg ? msg->set_choice_param(name, len, list, pos) : 0;
    }

    static int set_browser_param(const char *name, char *file, char *wildcard)
    {
        return msg ? msg->set_browser_param(name, file, wildcard) : 0;
    }

    static int set_text_param(const char *name, char *text, int linenum)
    {
        return msg ? msg->set_text_param(name, text, linenum) : 0;
    }

    static int set_timer_param(const char *name, int start, int delta, int state)
    {
        return msg ? msg->set_timer_param(name, start, delta, state) : 0;
    }

    static int set_passwd_param(const char *name, char *host, char *user, char *passwd)
    {
        return msg ? msg->set_passwd_param(name, host, user, passwd) : 0;
    }

    static int set_save_object(char *name)
    {
        return msg ? msg->set_save_object(name) : 0;
    }

    static int set_release_object(char *name)
    {
        return msg ? msg->set_release_object(name) : 0;
    }

    static void cancel_param(const char *name);
    static void reopen_param(const char *name);

    //
    // Messages to COVISE environment
    //
    static int send_generic_message(const char *keyword, const char *string);
    static int send_genericinit_message(const char *mkey, const char *keyword, const char *string);
    static char *get_generic_message();

    static void send_stop_pipeline();
    static int send_ctl_message(covise_msg_type type, char *msg_string);
    static void sendFinishedMsg();

    //
    // The callback functions called in callXXX  has to be
    // implemented by the user
    //
    static void callFeedbackCallback(Message *);
    static void callStartCallback(void);

    static void callAddObjectCallback(void);

    static void callGenericCallback(void);
    static void callSyncCallback(void);

    static void callPortReplyCallback(void);
    static void callAfterFinishCallback(void);
    static void callPipelineFinishCallback(void);
    //
    // Partitioned object support
    //
    static void partobjects_initialized(void);

    // Add Interactor for feedback
    static void addInteractor(coDistributedObject *obj, const char *name, const char *value);
};
}
#endif
