/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coErr.h"
#include <iostream>
#include <stdlib.h>
//#include <qfileinfo.h>

#ifndef _WIN32
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <errno.h>
#include <process.h>
#include <direct.h>
#endif
#include <cstring>

using namespace covise;

// -------------------------------------------
// static init values
//
std::string __MY_FILE__;
int __MY_LINE__;
FILE *coErr::m_file = NULL;
//char *coErr::m_filename     = NULL;
int coErr::m_console_lev = 99; // don't know yet
int coErr::m_file_lev = 99; // don't know yet

void bcHelperDebug(const char *msg, ...)
{
    char buf[1024];
    char final[1024];
    //QFileInfo fi(__MY_FILE__);
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);
    //qDebug("%s:%i - %s", (fi.fileName()).latin1(), __MY_LINE__, buf);
    sprintf(final, "%s:%i # %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
#ifdef WIN32
    OutputDebugString(final);
#else
    fprintf(stderr, "%s", final);
#endif
}

bool coErr::init(void)
{
    const char *env;
    // did we setup yet ? if no, do.

    if (!m_file)
    {
        /// setup levels
        if (m_file_lev == 99)
        {
            env = getenv("YAC_DEBUG_FILE");
            if (env)
            {
                m_file_lev = atoi(env);
            }
            else
            {
                m_file_lev = 0;
            }
        }
        if (m_console_lev == 99)
        {
            env = getenv("YAC_DEBUG_CONSOLE");
            if (env)
            {
                m_console_lev = atoi(env);
            }
            else
            {
                m_console_lev = 0;
            }
        }
        /// setup file if needed
        if (m_file_lev > 0)
        {
            // prepare path & filename
            char log_fname[256] = "";
            char tmpbuf[256];
#ifdef WIN32
            // windows specific
            env = getenv("TEMP");
            if (!env)
                env = getenv("TMP");
            if (env)
            {
                sprintf(log_fname, "%s\\yaclogs\\", env);
                if (_mkdir(log_fname) == -1)
                {
                    if (errno == ENOENT) // OOPS: TEMP set to an invalid path. use current path then
                    {
                        DBGOUT("coErr::init() - TEMP-Path (%s) invalid !", env);
                        memset(log_fname, 0, sizeof(log_fname));
                    }
                }
            }
            else
            {
                DBGOUT("coErr::Init() - Neither TMP nor TEMP is set!");
                memset(log_fname, 0, sizeof(log_fname));
            }
            sprintf(tmpbuf, "yac.pid=%d", _getpid());
            strcat(log_fname, tmpbuf);
#else
            // unix specific
            if (mkdir("/tmp/yaclogs/", S_IRWXU) == -1)
            {
                if (errno != EEXIST)
                {
                    DBGOUT("coErr::init() - Could not create directory /tmp/yaclogs (errno: %d)!", errno);
                    // just use current path
                    memset(log_fname, 0, sizeof(log_fname));
                }
                else
                {
                    DBGOUT("coErr::init() - Path /tmp/yaclogs exists!");
                }
            }
            sprintf(tmpbuf, "/tmp/yaclogs/yac.pid=%d", getpid());
            strcat(log_fname, tmpbuf);
#endif
            // open the logfile
            m_file = fopen(log_fname, "w");
            if (!m_file)
            {
                DBGOUT("coErr::init() - Could not open >%s< for logging !", log_fname);
                return false; // bail out
            }
            // ok. now install an atexit handler to close the logfile properly when the program is closed

            if (atexit(coErr::destroy))
            {
                DBGOUT("coErr::init() ... atexit() failed !");
            }

        } // endif m_file_lev > 0
    } // endif !m_file
    return true;
}

void coErr::destroy(void)
{
    // TODO: make that threadsafe
    if (!m_file)
    {
        DBGOUT("coErr::destroy() - Logger already destroyed !");
        return;
    }
    // logfile open
    setLogLevels(0, 0);
    fclose(m_file);
    m_file = NULL;
}

void coErr::setLogLevels(int lev_console, int lev_file)
{
    m_console_lev = lev_console;
    m_file_lev = lev_file;
}

void coErr::error(const char *msg, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);

    init();
    if (m_file && (m_file_lev > 0))
    {
        fprintf(m_file, "Error: %s\n", buf);
        fflush(m_file);
    }
    if (m_console_lev > 0)
    {
        char tmpstr[1024];
        sprintf(tmpstr, "Error: %s\n", buf);
#ifdef WIN32
        OutputDebugString(tmpstr);
#else
        fprintf(stderr, "%s", tmpstr);
#endif
    }
}

void coErr::warning(const char *msg, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);

    init();
    if (m_file && (m_file_lev > 1))
    {
        fprintf(m_file, "Warning: %s\n", buf);
        fflush(m_file);
    }
    if (m_console_lev > 1)
    {
        char tmpstr[1024];
        sprintf(tmpstr, "Warning: %s\n", buf);
#ifdef WIN32
        OutputDebugString(tmpstr);
#else
        fprintf(stderr, "%s", tmpstr);
#endif
    }
}

void coErr::info(const char *msg, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);

    init();
    if (m_file && (m_file_lev > 2))
    {
        fprintf(m_file, "Info: %s\n", buf);
        fflush(m_file);
    }
    if (m_console_lev > 2)
    {
        char tmpstr[1024];
        sprintf(tmpstr, "Info: %s\n", buf);
#ifdef WIN32
        OutputDebugString(tmpstr);
#else
        fprintf(stderr, "%s", tmpstr);
#endif
    }
}

void coErr::fl_error(const char *msg, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);

    init();
    if (m_file && (m_file_lev > 0))
    {
        fprintf(m_file, "%s:%i # Error: %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
        fflush(m_file);
    }
    if (m_console_lev > 0)
    {
        char tmpstr[1024];
        sprintf(tmpstr, "%s:%i # Error: %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
#ifdef WIN32
        OutputDebugString(tmpstr);
#else
        fprintf(stderr, "%s", tmpstr);
#endif
    }
}

void coErr::fl_warning(const char *msg, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);

    init();
    if (m_file && (m_file_lev > 1))
    {
        fprintf(m_file, "%s:%i # Warning: %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
        fflush(m_file);
    }
    if (m_console_lev > 1)
    {
        char tmpstr[1024];
        sprintf(tmpstr, "%s:%i # Warning: %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
#ifdef WIN32
        OutputDebugString(tmpstr);
#else
        fprintf(stderr, "%s", tmpstr);
#endif
    }
}

void coErr::fl_info(const char *msg, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, msg);
    vsprintf(buf, msg, ap);
    va_end(ap);

    init();
    if (m_file && (m_file_lev > 2))
    {
        fprintf(m_file, "%s:%i # Info: %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
        fflush(m_file);
    }
    if (m_console_lev > 2)
    {
        char tmpstr[1024];
        sprintf(tmpstr, "%s:%i # Info: %s\n", __MY_FILE__.c_str(), __MY_LINE__, buf);
#ifdef WIN32
        OutputDebugString(tmpstr);
#else
        fprintf(stderr, "%s", tmpstr);
#endif
    }
}

// --- corrected "old"-methods ----
void coErr::error(int line, const char *fname, const char *comment)
{
    init();
    if (m_file && (m_file_lev > 0))
    {
        fprintf(m_file, "%s:%i # Error: %s\n", fname, line, comment);
        fflush(m_file);
    }
    if (m_console_lev > 0)
    {
        __MY_FILE__ = fname;
        __MY_LINE__ = line;
        bcHelperDebug("Error: %s", comment);
    }
}

//////////////////////////////////////////////////////////////////////////

void coErr::warning(int line, const char *fname, const char *comment)
{
    init();
    if (m_file && (m_file_lev > 1))
    {
        fprintf(m_file, "%s:%i # Warning: %s\n", fname, line, comment);
        fflush(m_file);
    }
    if (m_console_lev > 1)
    {
        __MY_FILE__ = fname;
        __MY_LINE__ = line;
        bcHelperDebug("Warning: %s", comment);
    }
}

//////////////////////////////////////////////////////////////////////////

void coErr::comment(int line, const char *fname, const char *comment)
{
    init();
    if (m_file && (m_file_lev > 2))
    {
        fprintf(m_file, "%s:%i # Info: %s\n", fname, line, comment);
        fflush(m_file);
    }
    if (m_console_lev > 2)
    {
        __MY_FILE__ = fname;
        __MY_LINE__ = line;
        bcHelperDebug("Info: %s", comment);
    }
}
