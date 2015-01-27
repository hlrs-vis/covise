/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//
//
#pragma once

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <windows.h>
#include <process.h>
#include <dmusici.h>
#include <stdio.h>
#include <tchar.h>
#include <atlbase.h>

#include "loader.h"
#include "common.h"
#include "resource.h"

#pragma hdrstop
