/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------------------------------------------
//   $Id: FC_Base.h,v 1.9 2000/02/07 09:12:04 goesslet Exp $
//----------------------------------------------------------------------
//
//  $Log: FC_Base.h,v $
//  Revision 1.9  2000/02/07 09:12:04  goesslet
//  removed handler from FC_Base
//
//  Revision 1.8  2000/01/18 08:30:52  HUIBERP
//  added windows.h, MAXFLOAT, MINFLOAT for OS_WINDOWS
//
//  Revision 1.7  1999/12/15 08:54:07  goesslet
//  added PLATFORM dependent includes
//
//  Revision 1.6  1999/12/14 16:37:24  kickingf
//  Modification for linux port
//
//  Revision 1.5  1999/12/01 10:04:52  goesslet
//  bugfixes for ibm
//
//  Revision 1.4  1999/11/25 09:34:55  goesslet
//  wrong implemention of IsKindOf
//  use of define classname instead of function ClassName()
//
//  Revision 1.3  1999/11/23 09:58:25  goesslet
//  programmName and programmVersion now available in FC_Base
//
//  Revision 1.2  1999/11/09 17:39:47  goesslet
//  1. New define FC_Message which can handle arguments
//     used for cout. If DEBUG is not define no output is done.
//  2. New functions ActivateHandler and DeactivateHandler
//     instead of macro FIRE_BASE_HANDLER because the macro was
//     not working when the constructor was not inlined.
//
//  Revision 1.1  1999/11/08 13:24:23  goesslet
//  initial version
//
//----------------------------------------------------------------------

#ifndef _FC_BASE_H_
#define _FC_BASE_H_

#include <math.h>

#ifdef OS_HPUX
#include </usr/include/sys/param.h>
#undef MAXINT
#endif

#ifdef OS_WINDOWS
#include <windows.h>
#include <float.h>
#define M_PI 3.1415926535897932385E0
#include <strstrea.h>
#define MAXFLOAT FLT_MAX
#define MINFLOAT FLT_MIN
#else
#include <values.h>
#include <strstream.h>
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <iomanip.h>

#ifdef DEBUG
#define FC_ASSERT(EX) \
    if (!EX)          \
    cout << "FC_ASSERT failed: " << #EX << ", file: " << __FILE__ << ", line: " << __LINE__ << endl

#define FC_MESSAGE(EX) cout << "FC_MESSAGE: " << EX
#else
#define FC_ASSERT(EX) ((void)0)
#define FC_MESSAGE(EX) ((void)0)
#endif

class FC_Base
{
protected:
    static char programmName[100];
    static char programmVersion[100];
    static char programmIdStr[100];

public:
    virtual const char *ClassName() const
    {
        return "FC_Base";
    }
    virtual int IsKindOf(const char *nameIn) const
    {
        return (strcmp(nameIn, ClassName()) == 0);
    }

    // set name and version of programm for writing header
    static void SetProgramm(const char *pName, const char *pVersion);
    static const char *ProgrammName()
    {
        return programmName;
    }
    static const char *ProgrammVersion()
    {
        return programmVersion;
    }
};

#define IMPLEMENT_FIRE_BASE(className, subClass)                                \
public:                                                                         \
    virtual const char *ClassName() const                                       \
    {                                                                           \
        return #className;                                                      \
    }                                                                           \
    virtual int IsKindOf(const char *nameIn) const                              \
    {                                                                           \
        return (strcmp(nameIn, #className) == 0 || subClass::IsKindOf(nameIn)); \
    }                                                                           \
                                                                                \
private:
// min and max for all types
template <class T1>
inline T1 Max(const T1 &x, const T1 &y)
{
    return (x > y) ? x : y;
}

template <class T1>
inline T1 Min(const T1 &x, const T1 &y)
{
    return (x < y) ? x : y;
}
#endif
