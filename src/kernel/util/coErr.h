/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_ERR_H_
#define _CO_ERR_H_

#include "coExport.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>

// 25.02.98

/**
 * Class to handle errors, warnings, messages. Static functions
 * only, no non-static members
 */
namespace covise
{

class UTILEXPORT coErr
{
public:
    static bool init(void);
    static void destroy(void);

    static void setLogLevels(int lev_console, int lev_file);

    // Methods to log error-, warning- and info-messages. Usage like printf().
    // the error-level is set with the environment-variables YAC_DEBUG_FILE and YAC_DEBUG_CONSOLE
    // level set to 0 : nothing gets logged
    //              1 : only errors
    //              2 : errors + warnings
    //              3 : errors + warnings + messages
    static void error(const char *msg, ...);
    static void warning(const char *msg, ...);
    static void info(const char *msg, ...);

    // for use with the macros (current file-name, file-line and the message will get logged)
    // don't call the following three methods explicitly use LOGERROR, LOGWARNING and LOGINFO instead
    static void fl_error(const char *msg, ...);
    static void fl_warning(const char *msg, ...);
    static void fl_info(const char *msg, ...);

    // ------ old methods ------
    // print error message
    static void error(int line, const char *fname, const char *comment);
    // print warning message
    static void warning(int line, const char *fname, const char *comment);
    // print message
    static void comment(int line, const char *fname, const char *comment);
    // print message with 1 int parameter
    static void comment(int line, const char *fname, const char *comment, int i)
    {
        char buf[1024];
        sprintf(buf, comment, i);
        coErr::comment(line, fname, buf);
    }
    // print message with 1 string parameter
    static void comment(int line, const char *fname, const char *comment, const char *st)
    {
        char buf[1024];
        sprintf(buf, comment, st);
        coErr::comment(line, fname, st);
    }

private:
    // we don't want a copy-constructor
    coErr(const coErr &);
    // we don't want an assignment operator
    coErr &operator=(const coErr &);

    // file to write data to
    static FILE *m_file;
    // file name to use
    //  static char *m_filename;
    // file logging level: 99: don't know
    //                      0: nothing
    //                      1: only errors
    //                      2: errors + warnings
    //                      3: errors + warnings + messages
    static int m_file_lev;

    // level to copy messages on stderr (default NO
    static int m_console_lev;
};
}
// ----- BC: Helper-Function & Macro for Debugging
// NOTE:    For debug messages you may use DBGOUT which behaves like printf (with variable parameter list).
//          Message output goes to the debug output window on windows and stderr on linux
//          Output will only be generated in Debug-Builds (if _DEBUG is defined).
//
// WARNING: Since this macro is fragile you should take care when using "if ... else ...". Use brackets { } !!!

extern std::string UTILEXPORT __MY_FILE__;
extern int UTILEXPORT __MY_LINE__;
UTILEXPORT void bcHelperDebug(const char *msg, ...);

// DBGOUT - print a debug message only if _DEBUG is defined !
#ifdef _DEBUG
// in DEBUG-Builds use the following
#define DBGOUT              \
    __MY_FILE__ = __FILE__; \
    __MY_LINE__ = __LINE__; \
    bcHelperDebug
#else
// otherwise replace with "nothing"
//inline  void __dummy_dbgout(const char */*crap*/=NULL, ... )  {}
inline void __dummy_dbgout(const char *, ...)
{
}
//inline  void __dummy_dbgout(std::string &, ... )  {}
#define DBGOUT 1 ? (void)0 : __dummy_dbgout
#endif // _DEBUG

// ----- BC: Macros for logging
// see Note & Warning above
#ifdef YAC_DONT_LOG
// replace with "nothing"
//inline  void __dummy_logout(const char *crap=NULL, ... )  {}
inline void __dummy_logout(const char *, ...)
{
}
#define LOGERROR 1 ? (void)0 : __dummy_logout
#define LOGWARNING 1 ? (void)0 : __dummy_logout
#define LOGINFO 1 ? (void)0 : __dummy_logout
#else
// Usually do the following
#define LOGERROR            \
    __MY_FILE__ = __FILE__; \
    __MY_LINE__ = __LINE__; \
    covise::coErr::fl_error
#define LOGWARNING          \
    __MY_FILE__ = __FILE__; \
    __MY_LINE__ = __LINE__; \
    covise::coErr::fl_warning
#define LOGINFO             \
    __MY_FILE__ = __FILE__; \
    __MY_LINE__ = __LINE__; \
    covise::coErr::fl_info
#endif // YAC_DONT_LOG

#endif
