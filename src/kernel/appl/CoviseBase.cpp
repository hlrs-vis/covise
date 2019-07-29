/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CoviseBase.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <covise/covise_appproc.h>
#include <util/coLog.h>
#include <util/unixcompat.h>

#ifdef _WIN32
#include <direct.h>
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif

using namespace covise;

ApplicationProcess *CoviseBase::appmod = NULL;

Message *CoviseBase::applMsg = NULL;
const char *CoviseBase::m_name = NULL;
const char *CoviseBase::h_name = NULL;
const char *CoviseBase::instance = NULL;
int CoviseBase::error = 0;

Message CoviseBase::emergency_message;
char *CoviseBase::emergency_data;

const char *CoviseBase::module_description = NULL;
int CoviseBase::socket_id = 0;

vector<string> CoviseBase::tokenlist;
const char **CoviseBase::reply_buffer = NULL;
char *CoviseBase::reply_param_name = NULL;
char *CoviseBase::reply_keyword = NULL;
char *CoviseBase::reply_param_type = NULL;
int CoviseBase::no_of_reply_tokens = 0;

char *CoviseBase::port_name[CoviseBase::MAX_PORTS] = { NULL, NULL };
char *CoviseBase::port_description[CoviseBase::MAX_PORTS];
char *CoviseBase::port_datatype[CoviseBase::MAX_PORTS];
char *CoviseBase::port_dependency[CoviseBase::MAX_PORTS];
const char *CoviseBase::port_default[CoviseBase::MAX_PORTS];
int CoviseBase::port_required[CoviseBase::MAX_PORTS];
enum appl_port_type CoviseBase::port_type[CoviseBase::MAX_PORTS];
int CoviseBase::init_flag = 0;

#ifdef COVISE_USE_X11
XtAppContext CoviseBase::appContext = NULL;
XtInputId CoviseBase::X_id = (XtInputId)0;
#endif

CoviseCallback *CoviseBase::progressCallbackFunc = NULL;
CoviseCallback *CoviseBase::masterSwitchCallbackFunc = NULL;
CoviseCallback *CoviseBase::quitInfoCallbackFunc = NULL;
CoviseCallback *CoviseBase::customCallbackFunc = NULL;
void *CoviseBase::progressUserData = 0L;
void *CoviseBase::quitCallbackData = 0L;
void *CoviseBase::quitUserData = 0L;
CoviseCallback *CoviseBase::quitCallbackFunc = NULL;
void *CoviseBase::quitInfoCallbackData = 0L;
void *CoviseBase::quitInfoUserData = 0L;
void *CoviseBase::progressCallbackData = 0L;
void *CoviseBase::customUserData = 0L;
void *CoviseBase::customCallbackData = 0L;
void *CoviseBase::masterSwitchUserData = 0L;
CoviseParamCallback *CoviseBase::paramCallbackFunc = NULL;
void *CoviseBase::paramUserData = 0L;
void *CoviseBase::paramCallbackData = 0L;

bool CoviseBase::master = false;
char *CoviseBase::feedback_info = NULL;

CoviseBase::CoviseBase()
{
    setlocale(LC_NUMERIC, "C");
}

void CoviseBase::log_message(int line, const char *file, const char *comment)
{
    print_comment(line, file, "%s", comment);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::init_emergency_message(void)
{
    size_t size;
    emergency_data = new char[500];
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        strcpy(emergency_data, "DIED");
        strcat(emergency_data, "\n");
        strcat(emergency_data, m_name);
        strcat(emergency_data, "\n");
        strcat(emergency_data, instance);
        strcat(emergency_data, "\n");
        strcat(emergency_data, h_name);
        strcat(emergency_data, "\n");
        strcat(emergency_data, "A severe error has occured! The module has exited.");
        strcat(emergency_data, "\n");
        strcat(emergency_data, "You may try to restart the module in the Mapeditor.");
        size = strlen(emergency_data) + 1;
        emergency_message.data = DataHandle(emergency_data, size);
        emergency_message.type = COVISE_MESSAGE_UI;
    }
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_socket(int)
{
#ifdef COVISE_USE_X11
    XtRemoveInput(CoviseBase::X_id);
#else
// should be done by application module itself
#endif
}

//=====================================================================
//
//=====================================================================
void CoviseBase::doProgress()
{
    if (progressCallbackFunc != NULL)
        callProgressCallback();
}

//=====================================================================
//
//=====================================================================
void CoviseBase::callProgressCallback()
{
    (*progressCallbackFunc)(progressUserData, progressCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::callQuitCallback()
{
    (*quitCallbackFunc)(quitUserData, quitCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::callQuitInfoCallback()
{
    (*quitInfoCallbackFunc)(quitInfoUserData, quitInfoCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::callCustomCallback()
{
    (*customCallbackFunc)(customUserData, customCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::callMasterSwitchCallback()
{
    (*masterSwitchCallbackFunc)(masterSwitchUserData, NULL);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::callParamCallback(bool inMapLoading)
{
    (*paramCallbackFunc)(inMapLoading, paramUserData, paramCallbackData);
}

//=====================================================================
//
//=====================================================================
void CoviseBase::set_progress_callback(CoviseCallback *f, void *data)
{
    progressCallbackFunc = f;
    progressUserData = data;
    progressCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::set_quit_callback(CoviseCallback *f, void *data)
{
    quitCallbackFunc = f;
    quitUserData = data;
    quitCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::set_quit_info_callback(CoviseCallback *f, void *data)
{
    quitInfoCallbackFunc = f;
    quitInfoUserData = data;
    quitInfoCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::set_custom_callback(CoviseCallback *f, void *data)
{
    customCallbackFunc = f;
    customUserData = data;
    customCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::set_master_switch_callback(CoviseCallback *f, void *data)
{
    masterSwitchUserData = data;
    masterSwitchCallbackFunc = f;
    master = 0;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::set_param_callback(CoviseParamCallback *f, void *data)
{
    paramCallbackFunc = f;
    paramUserData = data;
    paramCallbackData = (void *)NULL;

    //cerr << "MODULE: I have set the callback to" << paramCallbackFunc << endl;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_progress_callback(void)
{
    progressCallbackFunc = (CoviseCallback *)NULL;
    progressUserData = (void *)NULL;
    progressCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_quit_callback(void)
{
    quitCallbackFunc = (CoviseCallback *)NULL;
    quitUserData = (void *)NULL;
    quitCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_quit_info_callback(void)
{
    quitInfoCallbackFunc = (CoviseCallback *)NULL;
    quitInfoUserData = (void *)NULL;
    quitInfoCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_custom_callback(void)
{
    customCallbackFunc = (CoviseCallback *)NULL;
    customUserData = (void *)NULL;
    customCallbackData = (void *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_master_switch_callback(void)
{
    masterSwitchCallbackFunc = (CoviseCallback *)NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::remove_param_callback(void)
{
    paramCallbackFunc = NULL;
    paramUserData = NULL;
    paramCallbackData = NULL;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::doCustom(Message *m)
{

    // call back the function provided by the user
    if (customCallbackFunc != NULL)
    {
        customCallbackData = (void *)m;
        callCustomCallback();
    }
}

//=====================================================================
//
//=====================================================================
void CoviseBase::doMasterSwitch()
{

    master = 1;

    // call back the function provided by the user
    if (masterSwitchCallbackFunc != NULL)
    {
        callMasterSwitchCallback();
    }
}

//=====================================================================
//
//=====================================================================
void CoviseBase::doSlaveSwitch()
{

    master = 0;

    // call back the function provided by the user
    if (masterSwitchCallbackFunc != NULL)
    {
        callMasterSwitchCallback();
    }
}

static bool is_file(const char *file)
{
#ifdef _WIN32
    struct _stat statbuf;
    int ret = _stat(file, &statbuf);
#else
    struct stat statbuf;
    int ret = stat(file, &statbuf);
#endif
    if (ret < 0)
    {
        return false;
    }

    return !S_ISDIR(statbuf.st_mode);
}

//=====================================================================
//
//=====================================================================
FILE *CoviseBase::fopen(const char *file, const char *mode, char **returnPath)
{
    static char buf[800];
    char *dirname, *covisepath;
    FILE *fp = NULL;

    if (returnPath)
        *returnPath = buf;

#ifdef _WIN32
    if (file[0] == '/' && file[1] && file[2] == ':')
        file++;
#endif

    if (is_file(file) || *mode == 'a' || *mode == 'w')
        fp = ::fopen(file, mode);

    if (fp != NULL || file[0] == '/'
#ifdef _WIN32
        || (file[0] && file[1] == ':')
#endif
            )
    {
        strcpy(buf, "");
        if (file[0] != '/'
#ifdef _WIN32
            && !(file[0] && file[1] == ':')
#endif
                )
        {
            char *ret = getcwd(buf, sizeof(buf));
            if (ret == NULL)
            {
                // do something useful here
            }
            strncat(buf, "/", sizeof(buf) - strlen(buf) - 1);
        }
        strncat(buf, file, sizeof(buf) - strlen(buf) - 1);
        return (fp);
    }

    if ((covisepath = getenv("COVISE_PATH")) == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: COVISE_PATH not defined!\n");
        return NULL;
    };

    char *pathbuf = new char[strlen(covisepath) + 1];
    strcpy(pathbuf, covisepath);
#ifdef _WIN32
    dirname = strtok(pathbuf, ";");
#else
    dirname = strtok(pathbuf, ":");
#endif

    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        if (is_file(buf) || *mode == 'a' || *mode == 'w')
            fp = ::fopen(buf, mode);

        if (fp != NULL)
        {
            delete[] pathbuf;
            return (fp);
        }

#ifdef CHECK_COVISE_PATH_UP
        for (int i = strlen(dirname) - 2; i > 0; i--)
        {
            if ((dirname[i] == '/') || (dirname[i] == '\\'))
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        if (is_file(buf) || *mode == 'a' || *mode == 'w')
            fp = ::fopen(buf, mode);
        if (fp != NULL)
        {
            delete[] pathbuf;
            return (fp);
        }
#endif
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }

    delete[] pathbuf;
    if (returnPath)
        *returnPath = NULL;
    return NULL;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::open(const char *file, int mode)
{
    char buf[800], *dirname, *covisepath;
    int fd, errno_save;

#ifdef _WIN32
    mode |= _O_BINARY; // make sure that file is not treated as text (no special handling of CTLR-Z, ...)
#endif

    fd = ::open(file, mode, 0660);
    if (fd >= 0)
        return (fd);
    errno_save = errno;

    if ((covisepath = getenv("COVISE_PATH")) == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: COVISE_PATH not defined!\n");
        return (-1);
    };

    char *pathbuf = new char[strlen(covisepath) + 1];
    strcpy(pathbuf, covisepath);
#ifdef _WIN32
    dirname = strtok(pathbuf, ";");
#else
    dirname = strtok(pathbuf, ":");
#endif
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        fd = ::open(buf, mode, 0660);
        if (fd >= 0)
        {
            delete[] pathbuf;
            return (fd);
        }
        else if (errno != 2) // errno==2: no such file
        {
            errno_save = errno;
        }
#ifdef CHECK_COVISE_PATH_UP
        for (int i = strlen(dirname) - 2; i > 0; i--)
        {
            if ((dirname[i] == '/') || (dirname[i] == '\\'))
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fd = ::open(buf, mode, 0660);
        if (fd >= 0)
        {
            delete[] pathbuf;
            return (fd);
        }
        else if (errno != 2) // errno==2: no such file
        {
            errno_save = errno;
        }
#endif
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }
    errno = errno_save;
    delete[] pathbuf;
    return (-1);
}

#if !defined(_WIN32) && !defined(__alpha) && !defined(_AIX)
//=====================================================================
//
//=====================================================================
DIR *CoviseBase::opendir(const char *file)
{
    char buf[800], *dirname, *covisepath;
    DIR *fp;

    fp = ::opendir(file);
    if (fp != NULL)
        return (fp);

    if ((covisepath = getenv("COVISE_PATH")) == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: COVISE_PATH not defined!\n");
        return NULL;
    };

    char *pathbuf = new char[strlen(covisepath) + 1];
    strcpy(pathbuf, covisepath);
#ifdef _WIN32
    dirname = strtok(pathbuf, ";");
#else
    dirname = strtok(pathbuf, ":");
#endif
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::opendir(buf);
        if (fp != NULL)
        {
            delete[] pathbuf;
            return (fp);
        }
#ifdef CHECK_COVISE_PATH_UP
        for (int i = strlen(dirname) - 2; i > 0; i--)
        {
            if ((dirname[i] == '/') || (dirname[i] == '\\'))
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::opendir(buf);
        if (fp != NULL)
        {
            delete[] pathbuf;
            return (fp);
        }
#endif
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }
    delete[] pathbuf;
    return (NULL);
}
#endif
//=====================================================================
//
//=====================================================================
void CoviseBase::getname(char *buf, const char *file, const char *addpath)
{
    FILE *fp = ::fopen(file, "r");
    if (fp != NULL)
    {
        strcpy(buf, file);
        fclose(fp);
        return;
    }

    if (addpath && getnameinpath(buf, file, addpath))
    {
        return;
    }

    const char *covisepath = getenv("COVISE_PATH");
    if (!covisepath)
    {
        print_comment(__LINE__, __FILE__, "ERROR: COVISE_PATH not defined!\n");
        print_exit(__LINE__, __FILE__, 1);
    }

    getnameinpath(buf, file, covisepath);
}

bool CoviseBase::getnameinpath(char *buf, const char *file, const char *path)
{
    char *pathbuf = new char[strlen(path) + 1];
    strcpy(pathbuf, path);
#ifdef _WIN32
    const char *dirname = strtok(pathbuf, ";");
#else
    const char *dirname = strtok(pathbuf, ":");
#endif
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        FILE *fp = ::fopen(buf, "r");
        if (fp != NULL)
        {
            delete[] pathbuf;
            fclose(fp);
            return true;
        }
#ifdef CHECK_COVISE_PATH_UP
        for (int i = strlen(dirname) - 2; i > 0; i--)
        {
            if ((dirname[i] == '/') || (dirname[i] == '\\'))
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::fopen(buf, "r");
        if (fp != NULL)
        {
            delete[] pathbuf;
            fclose(fp);
            return true;
        }
#endif
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }
    delete[] pathbuf;
    buf[0] = '\0';

    return false;
}

void CoviseBase::printDesc(const char *callname)
{
    // strip leading path from module name
    const char *modName = strrchr(callname, '/');
    if (modName)
        modName++;
    else
        modName = callname;

    cout << "Module:      \"" << modName << "\"" << endl;
    cout << "Desc:        \"" << module_description << "\"" << endl;

    int i, numItems;

    // count parameters
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == PARIN)
            numItems++;
    cout << "Parameters:   " << numItems << endl;

    // print parameters
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == PARIN)
        {
            cout << "  \"" << port_name[i]
                 << "\" \"" << port_datatype[i]
                 << "\" \"" << port_default[i]
                 << "\" \"" << port_description[i]
                 << "\" \"IMM\"" << endl;
        }

    // count OutPorts
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == OUTPUT_PORT)
            numItems++;
    cout << "OutPorts:     " << numItems << endl;

    // print outPorts
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == OUTPUT_PORT)
        {
            char *dependency;
            if (port_dependency[i])
            {
                dependency = new char[1 + strlen(port_dependency[i])];
                strcpy(dependency, port_dependency[i]);
            }
            else
            {
                dependency = new char[10];
                strcpy(dependency, "default");
            }
            cout << "  \"" << port_name[i]
                 << "\" \"" << port_datatype[i]
                 << "\" \"" << port_description[i]
                 << "\" \"" << dependency << '"' << endl;
        }

    // count InPorts
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == INPUT_PORT)
            numItems++;
    cout << "InPorts:      " << numItems << endl;

    // print InPorts
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == INPUT_PORT)
        {
            char *required = new char[10];
            if (port_required[i] == 0)
            {
                strcpy(required, "opt");
            }
            else
            {
                strcpy(required, "req");
            }

            cout << "  \"" << port_name[i]
                 << "\" \"" << port_datatype[i]
                 << "\" \"" << port_description[i]
                 << "\" \"" << required << '"' << endl;
        }
}

//=========================================================================
// parse the message string
//=========================================================================
int CoviseBase::parseMessage(char *line, char *token[], int tmax, const char *sep)
{
    char *tp;
    int count;

    count = 0;
    tp = strsep(&line, sep);
    for (count = 0; count < tmax && tp != NULL;)
    {
        token[count] = tp;
        tp = strsep(&line, sep);
        count++;
    }
    token[count] = NULL;
    return count;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::sendError(const char *fmt, ...)
{
    if (!fmt || !fmt[0])
        fmt = "appl: empty message";
    char *text = new char[strlen(fmt) + 500];

    va_list args;
    va_start(args, fmt);
    int messageSize = vsnprintf(text, strlen(fmt) + 500, fmt, args);
	va_end(args);
	if (messageSize>(strlen(fmt) + 500))
	{
		delete[] text;
		text = new char[strlen(fmt) + messageSize];
		va_start(args, fmt);
		vsnprintf(text, (strlen(fmt)+messageSize), fmt, args);
		va_end(args);
	}

    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen(text) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, text);

        Message message;
        message.type = COVISE_MESSAGE_COVISE_ERROR;
        message.data = DataHandle(msgdata, (int)strlen(msgdata) + 1);
        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send message without instance/init before");

    delete[] text;

    return;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::sendWarning(const char *fmt, ...)
{
    if (!fmt || !fmt[0])
        fmt = "appl: empty message";
    char *text = new char[strlen(fmt) + 500];

    va_list args;
    va_start(args, fmt);
    int messageSize = vsnprintf(text, strlen(fmt) + 500, fmt, args);
	va_end(args);
	if (messageSize>(strlen(fmt) + 500))
	{
		delete[] text;
		text = new char[strlen(fmt) + messageSize];
		va_start(args, fmt);
		vsnprintf(text, (strlen(fmt)+messageSize), fmt, args);
		va_end(args);
	}

    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += 10;
        size += strlen(h_name) + 1;
        size += strlen(text) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, text);

        Message message;
        message.type = COVISE_MESSAGE_WARNING;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send message without instance/init before");

    delete[] text;

    return;
}

//=====================================================================
//
//=====================================================================
void CoviseBase::sendInfo(const char *fmt, ...)
{
    if (!fmt || !fmt[0])
        fmt = "appl: empty message";
    char *text = new char[strlen(fmt) + 500];

    va_list args;
    va_start(args, fmt);
    int messageSize = vsnprintf(text, strlen(fmt) + 500, fmt, args);
	va_end(args);
	if (messageSize>(strlen(fmt) + 500))
	{
		delete[] text;
		text = new char[strlen(fmt) + messageSize];
		va_start(args, fmt);
		vsnprintf(text, (strlen(fmt)+messageSize), fmt, args);
		va_end(args);
	}

    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += 10;
        size += strlen(h_name) + 1;
        size += strlen(text) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, text);

        //cerr << "MODULE Sending INFO to mapeditor : " << message->data << endl;

        Message message;
        message.type = COVISE_MESSAGE_INFO;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send message without instance/init before");

    delete[] text;

    return;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::send_ui_message(const char *keyword, const char *string)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(keyword) + 1;
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen(string) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, keyword);
        strcat(msgdata, "\n");
        strcat(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, string);

        //cerr << "MODULE SENDING MESSAGE TO UI : " << message->data << endl;
        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

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
int CoviseBase::send_message(covise_msg_type type, const char *string)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen(string) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, string);

        //cerr << "MODULE SENDING MESSAGE TO CONTROLLER : " << message->data << endl;
        Message message;
        message.type = type;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

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
void CoviseBase::send_quit_request()
{
    if (init_flag == 0)
    {
        print_comment(__LINE__, __FILE__, "CoviseBase::send_message : init not called before");
        return;
    }
    char *string = new char[strlen("QUIT") + 1];
    strcpy(string, "QUIT");
    CoviseBase::send_message(COVISE_MESSAGE_REQ_UI, string);
}

//=====================================================================
//                     FEEDBACK
//=====================================================================
char CoviseBase::get_feedback_type()
{
    if (feedback_info)
        return (feedback_info[0]);
    else
        return (0);
}

void CoviseBase::set_feedback_info(const char *string)
{
    delete[] feedback_info;
    feedback_info = new char[strlen(string) + 1];
    strcpy(feedback_info, string);
}

const char *CoviseBase::get_feedback_info()
{
    return feedback_info;
}

//=====================================================================
//

//=====================================================================
//
//=====================================================================
int CoviseBase::send_feedback_message(const char *keyword, const char *string)
{
    if (feedback_info && appmod)
    {
        size_t size = 1; // final '\0'
        size += strlen(keyword) + 1;
        size += strlen(feedback_info + 1);
        size += strlen(string);

        char *msgdata = new char[size];
        strcpy(msgdata, keyword);
        strcat(msgdata, "\n");
        strcat(msgdata, feedback_info + 1);
        strcat(msgdata, string);

        //cerr << "MODULE SENDING MESSAGE TO UI : " << message->data << endl;

        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        CoviseBase::sendInfo("Sorry, No Object supports Feedback");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
int CoviseBase::request_param(const char *param_name)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen("PARREQ") + 1;
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen(param_name) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, "PARREQ");
        strcat(msgdata, "\n");
        strcat(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, param_name);

        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "Cannot send request_param message without instance/init before");
        return 0;
    }
}

//=====================================================================
//
//=====================================================================
void CoviseBase::enable_param(const char *name)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen("PARSTATE") + 1;
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen(name) + 1;
        size += strlen("TRUE") + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, "PARSTATE");
        strcat(msgdata, "\n");
        strcat(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");

        strcat(msgdata, name);
        strcat(msgdata, "\n");
        strcat(msgdata, "TRUE\n");

        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send enable_param message without instance/init before");
}

//=====================================================================
//
//=====================================================================
void CoviseBase::disable_param(const char *name)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen("PARSTATE") + 1;
        size += strlen(name) + 1;
        size += strlen("FALSE") + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, "PARSTATE");
        strcat(msgdata, "\n");
        strcat(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");

        strcat(msgdata, name);
        strcat(msgdata, "\n");
        strcat(msgdata, "FALSE\n");

        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send disable_param message without instance/init before");
}

//=====================================================================
//
//=====================================================================
void CoviseBase::hide_param(const char *name)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL)
        && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen("HIDE") + 1;
        size += strlen(name) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, "HIDE");
        strcat(msgdata, "\n");
        strcat(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");
        strcat(msgdata, name);

        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);;

        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send hide_param message without instance/init before");
}

//=====================================================================
//
//=====================================================================
void CoviseBase::show_param(const char *name)
{
    if ((m_name != NULL) && (h_name != NULL) && (instance != NULL) && (appmod != NULL))
    {
        size_t size = 1; // final '\0'
        size += strlen(m_name) + 1;
        size += strlen(instance) + 1;
        size += strlen(h_name) + 1;
        size += strlen("SHOW") + 1;
        size += strlen(name) + 1;

        char *msgdata = new char[size];
        strcpy(msgdata, "SHOW");
        strcat(msgdata, "\n");
        strcat(msgdata, m_name);
        strcat(msgdata, "\n");
        strcat(msgdata, instance);
        strcat(msgdata, "\n");
        strcat(msgdata, h_name);
        strcat(msgdata, "\n");

        strcat(msgdata, name);

        Message message;
        message.type = COVISE_MESSAGE_UI;
        message.data = DataHandle(msgdata, strlen(msgdata) + 1);

        appmod->send_ctl_msg(&message);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send show_param message without instance/init before");
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_socket_id()
{

    if (appmod != NULL)
        return appmod->get_socket_id(CoviseBase::remove_socket);
    else
        return -1;
}

#ifndef _WIN32
#ifdef COVISE_Signals
//=====================================================================
//
//=====================================================================
void CoviseBase::signal_handler(int signo, void *)
{

    if (signo != SIGPIPE)
    {
        // try to inform controller and mapeditor about our problem
        appmod->send_ctl_msg(&emergency_message);
#ifndef NODELETE_APPROC
        delete appmod;
#endif
        appmod = NULL;
        sleep(1);
    }
    else
        print_comment(__LINE__, __FILE__, "Cannot send message without instance/init before");

    exit(-1);
}
#endif
#endif

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_passwd(const char **host, const char **user, const char **passwd)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 3)
            return 0;

        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        if (NULL == reply_buffer[1])
        {
            return 0;
        }
        if (NULL == reply_buffer[2])
        {
            return 0;
        }
        *host = reply_buffer[0];
        *user = reply_buffer[1];
        *passwd = reply_buffer[2];
        return 3;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_colormapchoice(int *nmap)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens == 1)
        {
            if (NULL == reply_buffer[0])
            {
                return 0;
            }
            *nmap = atoi(reply_buffer[0]); // only one number
            return 0;
        }
        if (NULL == reply_buffer[0] || NULL == reply_buffer[1])
        {
            return 0;
        }
        *nmap = atoi(reply_buffer[0]);
        return atoi(reply_buffer[1]);
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_colormapchoice(TColormapChoice *values)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 2)
            return 0;
        if (!reply_buffer[0] || !reply_buffer[1])
            return 0;
        int nmap = (int)atoi(reply_buffer[1]);
        int ie = 2;
        for (int i = 0; i < nmap; i++)
        {
            if (!reply_buffer[ie])
                return 0;
            values[i].mapName = reply_buffer[ie];
            ie++;
            if (!reply_buffer[ie])
                return 0;
            int ns = (int)atoi(reply_buffer[ie]);
            ie++;
            for (int j = 0; j < ns * 5; j++)
            {
                if (!reply_buffer[ie])
                    return 0;
                values[i].mapValues.push_back((float)atof(reply_buffer[ie]));
                ie++;
            }
        }
        return nmap;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_choice(int *selected_pos)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        *selected_pos = atoi(reply_buffer[0]);
        return no_of_reply_tokens - 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_choice(int pos, string *label)
{
    if (appmod != NULL && get_reply_param_name() != NULL && no_of_reply_tokens > pos + 1)
    {
        if (reply_buffer[pos + 1] == 0)
        {
            return 0;
        }
        *label = reply_buffer[pos + 1];
        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_cli(const char **command)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        *command = reply_buffer[0];

        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_int_slider(long *min, long *max, long *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 3)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        if (NULL == reply_buffer[1])
        {
            return 0;
        }
        if (NULL == reply_buffer[2])
        {
            return 0;
        }
        *min = atol(reply_buffer[0]);
        *max = atol(reply_buffer[1]);
        *val = atol(reply_buffer[2]);

        return 3;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_float_slider(float *min, float *max, float *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 3)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        if (NULL == reply_buffer[1])
        {
            return 0;
        }
        if (NULL == reply_buffer[2])
        {
            return 0;
        }
        *min = (float)atof(reply_buffer[0]);
        *max = (float)atof(reply_buffer[1]);
        *val = (float)atof(reply_buffer[2]);
        return 3;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_int_vector(int pos, long *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {

        if (pos >= 0 && pos < no_of_reply_tokens)
        {
            *val = atol(reply_buffer[pos]);
            return 1;
        }
        else
            return 0;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_float_vector(int pos, float *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (pos >= 0 && pos < no_of_reply_tokens)
        {
            if (NULL == reply_buffer[pos])
            {
                return 0;
            }
            *val = (float)atof(reply_buffer[pos]);
            return 1;
        }
        else
            return 0;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_int_scalar(long *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        *val = atol(reply_buffer[0]);

        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_int64_scalar(int64_t *val)
{

	if (appmod != NULL && get_reply_param_name() != NULL)
	{
		if (no_of_reply_tokens < 1)
			return 0;
		if (NULL == reply_buffer[0])
		{
			return 0;
		}
#ifdef _WIN32
		*val = _atoi64(reply_buffer[0]);
#else
		*val = atoll(reply_buffer[0]);
#endif

		return 1;
	}
	else
		return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_float_scalar(float *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        *val = (float)atof(reply_buffer[0]);

        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_boolean(int *val)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }

        if (strcmp(reply_buffer[0], "TRUE") == 0)
            *val = 1;
        else if (strcmp(reply_buffer[0], "FALSE") == 0)
            *val = 0;
        else if (strcmp(reply_buffer[0], "1") == 0)
            *val = 1;
        else if (strcmp(reply_buffer[0], "0") == 0)
            *val = 0;
        else
            return 0;
        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_material(string *value)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens == 0)
            return 0;

        ostringstream os;
        for (int i = 0; i < no_of_reply_tokens; i++)
            os << " " << reply_buffer[i];

        *value = os.str();
        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_string(const char **string)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;

        if (reply_buffer == NULL || NULL == reply_buffer[0])
        {
            return 0;
        }
        *string = reply_buffer[0];

        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_browser(const char **file)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }

        *file = reply_buffer[0];

        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_timer(long *start, long *delta, long *state)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 3)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        if (NULL == reply_buffer[1])
        {
            return 0;
        }
        if (NULL == reply_buffer[2])
        {
            return 0;
        }
        *start = atol(reply_buffer[0]);
        *delta = atol(reply_buffer[1]);
        *state = atol(reply_buffer[2]);

        return 3;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_arrayset(const char **buf)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 1)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        *buf = reply_buffer[0];
        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_color(float *r, float *g, float *b, float *a)
{

    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 4)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        if (NULL == reply_buffer[1])
        {
            return 0;
        }
        if (NULL == reply_buffer[2])
        {
            return 0;
        }
        if (NULL == reply_buffer[3])
        {
            return 0;
        }
        *r = (float)atof(reply_buffer[0]);
        *g = (float)atof(reply_buffer[1]);
        *b = (float)atof(reply_buffer[2]);
        *a = (float)atof(reply_buffer[3]);
        return 4;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_colormap(float *min, float *max, int *len, colormap_type *type)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        if (no_of_reply_tokens < 4)
            return 0;
        if (NULL == reply_buffer[0])
        {
            return 0;
        }
        if (NULL == reply_buffer[1])
        {
            return 0;
        }
        if (NULL == reply_buffer[2])
        {
            return 0;
        }
        if (NULL == reply_buffer[3])
        {
            return 0;
        }
        *min = (float)atof(reply_buffer[0]);
        *max = (float)atof(reply_buffer[1]);
        if (!strcmp(reply_buffer[2], "RGBAX"))
        {
            *type = RGBAX;
            *len = atoi(reply_buffer[3]);
        }
        else if (!strcmp(reply_buffer[2], "VIRVO"))
        {
            *type = VIRVO;
        }
        else
        {
            cerr << "invalid colormap type " << reply_buffer[2] << endl;
            *type = RGBAX;
        }
        return 1;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::get_reply_colormap(int pos, float *r, float *g, float *b, float *a, float *x)
{
    if (appmod != NULL && get_reply_param_name() != NULL)
    {
        int ie = pos * 5;
        if (ie >= 0 && ie + 4 < no_of_reply_tokens)
        {
            *r = (float)atof(reply_buffer[ie + 4]);
            *g = (float)atof(reply_buffer[ie + 5]);
            *b = (float)atof(reply_buffer[ie + 6]);
            *a = (float)atof(reply_buffer[ie + 7]);
            *x = (float)atof(reply_buffer[ie + 8]);
            //cerr << "Color [" << pos << "] = " << *r << " " << *g << " " << *b << " " << *a << endl;
            return 1;
        }
        else
            return 0;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_colormapchoice_param(const char *pname, int num, int pos, TColormapChoice *list)
{

    if (appmod != NULL)
    {
        ostringstream os;
        os << pname << "\nColormapChoice\n" << pos << " " << num;

        for (int i = 0; i < num; i++)
        {
            size_t ll = list[i].mapValues.size();
            os << " " << list[i].mapName << " " << ll / 5;

            for (size_t j = 0; j < ll; j++)
                os << " " << list[i].mapValues[j];
        }

        string buffer = os.str();
        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_colormap_param(const char *pname, float min, float max, const string &cmap)
{

    if (appmod != NULL)
    {
        ostringstream os;
        os << pname << "\nColormap\n" << min << " " << max << " " << cmap;

        string buffer = os.str();
        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

int CoviseBase::update_choice_param(const char *pname, int num, const char *const *list, int pos)
{
    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nChoice\n" << pos;

        for (int i = 0; i < num; i++)
            os << " " << list[i];

        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

int CoviseBase::update_material_param(const char *pname, const string &value)
{

    if (appmod != NULL)
    {
        ostringstream os;
        os << pname << "\nMaterial\n" << value;
        string buffer = os.str();
        cerr << "::::::::::   " << buffer << endl;
        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());
        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_color_param(const char *pname, float r, float g, float b, float a)
{

    if (appmod != NULL)
    {
        ostringstream os;
        os << pname << "\nColor\n" << r << " " << g << " " << b << " " << a;

        string buffer = os.str();
        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_slider_param(const char *pname, long min, long max, long val)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nIntSlider\n" << min << " " << max << " " << val;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_slider_param(const char *pname, float min, float max, float val)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nFloatSlider\n" << min << " " << max << " " << val;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_string_param(const char *pname, char *stext)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nString\n" << stext;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return -1;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_text_param(const char *pname, char *text, int linenum)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nText\n" << linenum << " " << text;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_boolean_param(const char *pname, int val)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nBoolean\n" << val;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_scalar_param(const char *pname, long val)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nIntScalar\n" << val;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_scalar_param(const char *pname, float val)
{
    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nFloatScalar\n" << val;
        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_vector_param(const char *pname, int num, long *list)
{

    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nIntVector\n";

        for (int i = 0; i < num; i++)
            os << *(list + i) << " ";

        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_vector_param(const char *pname, int num, float *list)
{
    if (appmod != NULL)
    {
        ostringstream os;
        string buffer;

        os << pname << "\nFloatVector\n";

        for (int i = 0; i < num; i++)
            os << *(list + i) << " ";

        buffer = os.str();

        send_message(COVISE_MESSAGE_PARINFO, (char *)buffer.c_str());

        return 1;
    }

    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int CoviseBase::update_timer_param(const char *pname, long start, long delta, long state)
{
    char numbuf[64];
    char *data;
    size_t size;

    if (appmod != NULL)
    {

        int parnum = 1;
        int no_of_tokens = 3; // start , delta , state

        size = 1; // final '\0'
        size += 2; // number of params
        size += strlen(pname) + 1;
        size += strlen("Timer") + 1;
        size += 5; // number of parameter tokens
        size += 64; // start
        size += 64; // delta
        size += 64; // state

        data = new char[size];

        sprintf(data, "%d\n", parnum);
        strcat(data, pname);
        strcat(data, "\n");
        strcat(data, "Timer");
        strcat(data, "\n");
        sprintf(numbuf, "%d\n", no_of_tokens);
        strcat(data, numbuf);
        sprintf(numbuf, "%ld\n", start);
        strcat(data, numbuf);
        sprintf(numbuf, "%ld\n", delta);
        strcat(data, numbuf);
        sprintf(numbuf, "%ld\n", state);
        strcat(data, numbuf);

        // build and send message
        send_message(COVISE_MESSAGE_PARINFO, data);

        // delete data
        delete[] data;
        return 1;
    }
    else
        return 0;
}
