/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          error.cpp  -  error handling
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __sgi
#include <cstdio>
#include <cstdarg>
#else
#include <stdio.h>
#include <stdarg.h>
#endif
#include <iostream>

#include "error.h"
#include "main.h"

#ifdef COVISE
#include "coviseInterface.h"
#endif
// ***************************************************************************
// throw exception
// ***************************************************************************

void ErrorFunction(char *pstr, ...)
{
    char buffer[200];
    va_list argptr;

    va_start(argptr, pstr);
    vsprintf(buffer, pstr, argptr);
    va_end(argptr);

    throw TException(buffer);
}

// ***************************************************************************
// show warning
// ***************************************************************************

void WarningFunction(char *pstr, ...)
{
    char buffer[200];
    va_list argptr;

    va_start(argptr, pstr);
    vsprintf(buffer, pstr, argptr);
    va_end(argptr);
#ifdef COVISE
    covise.warning(buffer);
#endif
    *perr << buffer << endl;
}

// ***************************************************************************
// error class
// ***************************************************************************

// constructor

TException::TException(char *pstr, ...)
{
    char buffer[200];
    va_list argptr;

    va_start(argptr, pstr);
    vsprintf(buffer, pstr, argptr);
    va_end(argptr);
    strcpy(Description, buffer);
}

// display error message

void TException::Display()
{
#ifdef COVISE
    covise.error(Description);
#endif
    *perr << Description << endl;
}
