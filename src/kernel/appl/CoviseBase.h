/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_BASE_H
#define COVISE_BASE_H

#if !defined(__linux__) && !defined(_WIN32)
#define COVISE_Signals
#endif

#include <covise/covise.h>
#include <util/coTypes.h>
#include <string>

#ifndef _WIN32
#include <dirent.h>
#endif

#ifdef COVISE_USE_X11
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>
#endif

#ifdef COVISE_Signals
#include <covise/covise_signal.h>
#endif

#include <covise/covise_msg.h>

namespace covise
{

class ApplicationProcess;
class coDistributedObject;

const int MAXTOKENS = 10;

enum appl_port_type
{
    DESCRIPTION = 0, //0
    INPUT_PORT, //1
    OUTPUT_PORT, //2
    PARIN, //3
    PAROUT //4
};

struct TColormapChoice
{
    string mapName;
    vector<float> mapValues;
};

/*
 * Covise callbacks
 */
typedef void CoviseCallback(void *userData, void *callbackData);
typedef void CoviseParamCallback(bool inMapLoading, void *userData, void *callbackData);

class APPLEXPORT CoviseBase
{
public:
    CoviseBase();

    enum
    {
        MAX_PORTS = 4096
    };
    enum
    {
        ADD_OBJ,
        DEL_OBJ
    };

protected:
    // Error flag
    static int error;

    // Communication data
    static char *reply_keyword;
    static char *reply_param_name;
    static char *reply_param_type;
    static const char **reply_buffer;
    static int no_of_reply_tokens;
    static vector<string> tokenlist;
    static Message *applMsg;
    static int socket_id;
    static std::string h_name;
    static std::string m_name;
    static std::string instance;
    static Message emergency_message;
    static char *emergency_data;
    static const char *module_description;
    static bool master;

    static char *port_name[MAX_PORTS];
    static char *port_description[MAX_PORTS];
    static enum appl_port_type port_type[MAX_PORTS];
    static const char *port_default[MAX_PORTS];
    static char *port_datatype[MAX_PORTS];
    static char *port_dependency[MAX_PORTS];
    static int port_required[MAX_PORTS];

#ifdef COVISE_USE_X11
    static XtInputId X_id;
    static XtAppContext appContext;
#endif

    // Flag for initialization
    static int init_flag;

    static char *feedback_info;

    // protected member funcs
    static void remove_socket(int sock_id);

    static void doProgress();
    static void doCustom(Message *m);
    static void doMasterSwitch();
    static void doSlaveSwitch();
//static    int  doQuit();
#ifndef _WIN32
#ifdef COVISE_Signals
    static void signal_handler(int signo, void *);
#endif
#endif
    static void init_emergency_message(void);
#ifdef COVISE_USE_X11
    static void socketCommunicationCB(XtPointer client_data, int *source, XtInputId *id);
#endif

    // callback stuff
    static CoviseCallback *progressCallbackFunc;
    static void *progressUserData;
    static void *progressCallbackData;
    static CoviseCallback *quitCallbackFunc;
    static void *quitUserData;
    static void *quitCallbackData;
    static CoviseCallback *quitInfoCallbackFunc;
    static void *quitInfoUserData;
    static void *quitInfoCallbackData;
    static CoviseCallback *customCallbackFunc;
    static void *customUserData;
    static void *customCallbackData;
    static CoviseCallback *masterSwitchCallbackFunc;
    static void *masterSwitchUserData;
    static CoviseParamCallback *paramCallbackFunc;
    static void *paramUserData;
    static void *paramCallbackData;
    static void callProgressCallback(void);
    static void callQuitCallback(void);
    static void callQuitInfoCallback(void);
    static void callCustomCallback(void);
    static void callMasterSwitchCallback(void);
    static void callParamCallback(bool inMapLoading);

public:
    static int parseMessage(char *line, char *token[], int tmax, const char *sep);

    static void set_progress_callback(CoviseCallback *f, void *userData);
    static void set_quit_callback(CoviseCallback *f, void *userData);
    static void set_quit_info_callback(CoviseCallback *f, void *userData);
    static void set_custom_callback(CoviseCallback *f, void *userData);
    static void set_param_callback(CoviseParamCallback *f, void *userData);
    static void remove_progress_callback();
    static void remove_quit_callback();
    static void remove_quit_info_callback();
    static void remove_custom_callback();
    static void remove_param_callback();

    static void set_master_switch_callback(CoviseCallback *f, void *userData);
    static void remove_master_switch_callback();

    static bool is_master()
    {
        return master;
    };

    static int request_param(const char *name);
    static void enable_param(const char *name);
    static void disable_param(const char *name);
    static void hide_param(const char *name);
    static void show_param(const char *name);
    static void cancel_param(const char *name);
    static void reopen_param(const char *name);

    static int get_reply_passwd(const char **host, const char **user, const char **passwd);
    static int get_reply_cli(const char **command);
    static int get_reply_int_slider(long *min, long *max, long *val);
    static int get_reply_float_slider(float *min, float *max, float *val);
    static int get_reply_int_vector(int pos, long *val);
    static int get_reply_float_vector(int pos, float *val);
	static int get_reply_int_scalar(long *val);
	static int get_reply_int64_scalar(int64_t *val);
    static int get_reply_float_scalar(float *val);
    static int get_reply_boolean(int *);
    static int get_reply_string(const char **string);
    static int get_reply_browser(const char **file);
    static int get_reply_timer(long *start, long *delta, long *state);
    static int get_reply_arrayset(const char **msg_buf);
    static int get_reply_choice(int *selected_pos);
    static int get_reply_choice(int pos, string *choiceLabels);
    static int get_reply_colormapchoice(int *selected_pos);
    static int get_reply_colormapchoice(TColormapChoice *values);
    static int get_reply_material(string *value);
    static int get_reply_colormap(float *min, float *max, int *len, colormap_type *type);
    static int get_reply_colormap(int pos, float *r, float *g, float *b, float *a, float *x);
    static int get_reply_color(float *r, float *g, float *b, float *a);

    static int update_slider_param(const char *name, long min, long max, long val);
    static int update_slider_param(const char *name, float min, float max, float val);
    static int update_scalar_param(const char *name, long val);
    static int update_scalar_param(const char *name, float val);
    static int update_vector_param(const char *name, int num, long *list);
    static int update_vector_param(const char *name, int num, float *list);
    static int update_string_param(const char *name, char *string);
    static int update_text_param(const char *name, char *text, int linenum);
    static int update_boolean_param(const char *name, int val);
    static int update_choice_param(const char *name, int len, const char *const *list, int pos);
    static int update_timer_param(const char *name, long start, long delta, long state);
    static int update_colormap_param(const char *name, float min, float max, const string &cmap);
    static int update_material_param(const char *name, const string &value);
    static int update_colormapchoice_param(const char *name, int num, int pos, TColormapChoice *list);
    static int update_color_param(const char *name, float r, float g, float b, float a);

    static int get_socket_id();
    static const char *get_instance()
    {
        if (instance.empty())
        {
            return nullptr;
        }
        return instance.c_str();
    }
    static const char *get_host()
    {
        if (h_name.empty())
        {
            return nullptr;
        }
        return h_name.c_str();
    }
    static const char *get_module()
    {
        if (m_name.empty())
        {
            return nullptr;
        }
        return m_name.c_str();
    }
    static const char *get_reply_param_name()
    {
        return reply_param_name;
    }

    //
    // Messages to COVISE environment
    //

    /// send warning messages - printf style
    static void sendWarning(const char *fmt, ...)
#ifdef __GNUC__
        __attribute__((format(printf, 1, 2)))
#endif
        ;

    /// send warning messages - printf style
    static void sendError(const char *fmt, ...)
#ifdef __GNUC__
        __attribute__((format(printf, 1, 2)))
#endif
        ;

    /// send warning messages - printf style
    static void sendInfo(const char *string, ...)
#ifdef __GNUC__
        __attribute__((format(printf, 1, 2)))
#endif
        ;
    static int send_message(covise_msg_type type, const char *msg_string);
    static int send_ui_message(const char *keyword, const char *string);

    static int send_feedback_message(const char *keyword, const char *string);
    static void set_feedback_info(const char *string);
    static const char *get_feedback_info();
    static char get_feedback_type();
    static void send_quit_request();
    static void log_message(int line, const char *file, const char *comment);

    static ApplicationProcess *appmod;

    //
    // File Open Utilities by Uwe Woessner
    //
    // define const char * functions as well...
    static FILE *fopen(const char *file, const char *mode, char **returnPath = NULL);
    static int open(const char *file, int mode);
#if !defined(_WIN32) && !defined(__alpha) && !defined(_AIX)
    static DIR *opendir(const char *file);
#endif
    static void getname(char *buf, const char *file, const char *addpath = NULL);
    static bool getnameinpath(char *buf, const char *file, const char *path);

    // Add Interactor for feedback
    static void addInteractor(coDistributedObject *obj, const char *name, const char *value);

    // Print out description for -d option
    static void printDesc(const char *callname);
};
}
#endif
