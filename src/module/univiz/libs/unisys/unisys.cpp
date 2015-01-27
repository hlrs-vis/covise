/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Unification Library for Modular Visualization Systems
//
// System
//
// CGL ETH Zuerich
// Filip Sadlo 2006 - 2007

#include <stdio.h>
#include <stdarg.h>
#include "unisys.h"

#define SUPPRESS_IMPLEMENTATION_WARNINGS 1
#ifdef WIN32
#pragma warning(disable : 4996 4244)
#endif

void UniSys::moduleStatus(const char *str, int perc)
{
#ifdef AVS
    char w[4096]; // ###
    strcpy(w, str);
    AVSmodule_status(w, perc);
#endif

#ifdef COVISE
    // TODO
    //printf("ERROR: moduleStatus not yet implemented\n");
    printf("%s, %d%% done             \r", str, perc);
#endif

#ifdef VTK
    vtkAlg->UpdateProgress(perc / 100.0);
    vtkAlg->SetProgressText(str);
#endif
}

void UniSys::info(const char *str, ...)
{
    char buf[4096]; // ###
    va_list argList;
    va_start(argList, str);
    vsprintf(buf, str, argList);

#ifdef AVS
    printf("%s\n", buf);
#endif

#ifdef COVISE
#ifdef COVISE5
    covModule->send_info(buf);
#else
    covModule->sendInfo("%s", buf);
#endif
    printf("%s\n", buf);
#endif

#ifdef VTK
    //printf("WARNING: info not yet implemented, string=%s\n", str);
    printf("%s\n", buf);
#endif

    va_end(argList);
}

void UniSys::warning(const char *str, ...)
{
    char buf[4096]; // ###
    va_list argList;
    va_start(argList, str);
    vsprintf(buf, str, argList);

#ifdef AVS
    AVSwarning(buf);
#endif

#ifdef COVISE
#ifdef COVISE5
    covModule->send_warning(buf);
#else
    covModule->sendWarning("%s", buf);
#endif
#endif

#ifdef VTK
    //printf("WARNING: warning not yet implemented, string=%s\n", str);
    //printf("warning: %s\n", buf);
    //vtkWarningMacro(buf);
    char buf2[4096]; // ###
    sprintf(buf2, "warning: %s", buf);
    vtkOutputWindowDisplayWarningText(buf2);
#endif
}

void UniSys::error(const char *str, ...)
{
    char buf[4096]; // ###
    va_list argList;
    va_start(argList, str);
    vsprintf(buf, str, argList);

#ifdef AVS
    AVSerror(buf);
#endif

#ifdef COVISE
#ifdef COVISE5
    covModule->send_error(buf);
#else
    covModule->sendError("%s", buf);
#endif
#endif

#ifdef VTK
    //printf("WARNING: error not yet implemented, string=%s\n", str);
    //printf("error: %s\n", buf);
    //vtkErrorMacro(buf);
    char buf2[4096]; // ###
    sprintf(buf2, "ERROR: %s", buf);
    vtkOutputWindowDisplayWarningText(buf2);
#endif
}

#ifndef COVISE
bool UniSys::inputChanged(const char *name, int connection)
#else
bool UniSys::inputChanged(const char *, int)
#endif
{
#ifdef AVS
    char w[4096]; // ###
    strcpy(w, name);
    return AVSinput_changed(w, connection);
#endif

#ifdef COVISE
// TODO
#if !SUPPRESS_IMPLEMENTATION_WARNINGS
    printf("WARNING: inputChanged not yet implemented\n");
#endif
    return true; // #### not ok for some cases?
#endif

#ifdef VTK
#if !SUPPRESS_IMPLEMENTATION_WARNINGS
    printf("WARNING: inputChanged not yet implemented\n");
#endif
    return true; // #### not ok for some cases?
#endif
}

#ifndef COVISE
bool UniSys::parameterChanged(const char *name)
#else
bool UniSys::parameterChanged(const char *)
#endif
{
#ifdef AVS
    char w[4096]; // ###
    strcpy(w, name);
    return AVSparameter_changed(w);
#endif

#ifdef COVISE
// TODO
#if !SUPPRESS_IMPLEMENTATION_WARNINGS
    printf("WARNING: parameterChanged not yet implemented\n");
#endif
    return true; // #### not ok for some cases?
#endif

#ifdef VTK
#if !SUPPRESS_IMPLEMENTATION_WARNINGS
    printf("WARNING: inputChanged not yet implemented\n");
#endif
    return true; // #### not ok for some cases?
#endif
}
