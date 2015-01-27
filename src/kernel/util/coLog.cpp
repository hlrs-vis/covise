/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coLog.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <iostream>

#undef DEBUG

namespace covise
{

FILE *COVISE_time_hdl = NULL;
static FILE *COVISE_debug_hdl = NULL;
int COVISE_debug_level = 0;
static const char *COVISE_debug_filename = "covise.log";

void print_comment(int line, const char *filename, const char *fmt, ...)
{
    char header[1000];
    sprintf(header, "%s:%d: ", filename, line);
    const size_t len = strlen(fmt) + strlen(filename) + 1000;
    char *text = new char[len + strlen(header)];
    strcpy(text, header);
    va_list args;
    va_start(args, fmt);
    vsnprintf(text + strlen(header), len, fmt, args);
    va_end(args);

    if (COVISE_debug_level >= 4)
    {
        std::cerr << text << std::endl;
        if ((COVISE_debug_hdl == NULL) && COVISE_debug_filename != NULL)
            COVISE_debug_hdl = fopen(COVISE_debug_filename, "w");

        if (COVISE_debug_hdl != NULL)
        {
            fprintf(COVISE_debug_hdl, "%s", text);
            fflush(COVISE_debug_hdl);
        }
    }
    delete[] text;
}

#if 0
void print_comment(int line, const char *filename, const char *comment, int level)
{
   if(level < COVISE_debug_level)
   {
      if(COVISE_debug_hdl == NULL)
         COVISE_debug_hdl = fopen(COVISE_debug_filename,"w");
      if(COVISE_debug_hdl)
      {
         fprintf(COVISE_debug_hdl,"(%d)\t\"%s\" in line %4d in %s\n",
            level, comment, line, filename);
         fflush(COVISE_debug_hdl);
      }
   }
}
#endif

void print_exit(int line, const char *filename, int how)
{
    static int first = 1;

    if (!first)
        return;
    else
        first = 0;
#ifndef _WIN32
//delete Process::this_process;
#endif
    if (how != 0)
    {
        if (COVISE_debug_level > 8)
        {
            if (COVISE_debug_hdl == NULL)
                COVISE_debug_hdl = fopen(COVISE_debug_filename, "w");
            if (COVISE_debug_hdl)
            {
                fprintf(COVISE_debug_hdl, "exit due to failure in line %4d in %s\n",
                        line, filename);
                fclose(COVISE_debug_hdl);
            }
        }
        fprintf(stderr, "exit due to failure in line %4d in %s\n", line, filename);
        //abort();
        exit(0);
    }
    else
    {
        exit(0);
    }
}

void print_exit(int line, const char *filename, const char *why, int how)
{
    static int first = 1;

    if (!first)
        return;
    else
        first = 0;
#ifndef _WIN32
//delete Process::this_process;
#endif
    if (how != 0)
    {
        if (COVISE_debug_hdl)
        {
            fprintf(COVISE_debug_hdl, "exit due to failure in line %4d in %s: %s\n",
                    line, filename, why);
            fclose(COVISE_debug_hdl);
        }
        fprintf(stderr, "exit due to failure in line %4d in %s: %s\n", line, filename, why);
        //abort();
        exit(0);
    }
    else
    {
        exit(0);
    }
}

void print_error(int line, const char *filename, const char *fmt, ...)
{
    char header[1000];
    sprintf(header, "%s:%d - error: ", filename, line);
    const size_t len = strlen(fmt) + strlen(filename) + 1000;
    char *text = new char[len + strlen(header)];
    strcpy(text, header);
    va_list args;
    va_start(args, fmt);
    vsnprintf(text + strlen(header), len, fmt, args);
    va_end(args);

    if ((COVISE_debug_hdl == NULL) && COVISE_debug_filename != NULL)
        COVISE_debug_hdl = fopen(COVISE_debug_filename, "w");

    if (COVISE_debug_hdl != NULL)
    {
        fprintf(COVISE_debug_hdl, "%s", text);
        fflush(COVISE_debug_hdl);
    }
    std::cerr << text << std::endl;
    delete[] text;
}

void print_time(const char *comment)
{
    if (COVISE_time_hdl == NULL)
        return;

    fprintf(COVISE_time_hdl, "%s\n", comment);
    //    fflush(COVISE_debug_hdl);
}
}
