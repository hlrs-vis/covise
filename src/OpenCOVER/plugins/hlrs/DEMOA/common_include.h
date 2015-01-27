/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __COMMON_INCLUDE__HHH
#define __COMMON_INCLUDE__HHH

#ifdef WIN32
#pragma warning(disable : 4786)
typedef __int64 longint;
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Selten benutzte Teile der Windows-Header nicht einbinden
#endif

#ifndef FUNCTION_DLL
#ifdef FUNCTION_EXPORT
#define FUNCTION_DLL __declspec(dllexport)
#else
#define FUNCTION_DLL __declspec(dllimport)
#endif
#endif
#else
#define FUNCTION_DLL
typedef long long longint;
#endif

#include <cstddef>
#include <cstring>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>

// empty dummy function to be used where no action is needed (e.g. body of a
// method that is left empty intentionally)
inline void nooperation()
{
    // do nothing
}

// calculates size of an array, passing raw pointer results in compilation error
#define ARRAY_SIZE(e) (sizeof(deduceArraySize_(e)))
template <typename T, std::size_t N>
char(&deduceArraySize_(const T(&)[N]))[N];

#endif // __COMMON_INCLUDE__HHH
