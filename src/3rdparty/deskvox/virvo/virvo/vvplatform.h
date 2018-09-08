// Virvo - Virtual Reality Volume Rendering
// Contact: Stefan Zellmann, zellmans@uni-koeln.de
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

#ifndef VV_PLATFORM_H
#define VV_PLATFORM_H

#ifdef _WIN32

#ifndef WIN32
#define WIN32 1
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

// Min windows version: WinXP.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

// MinGW-w64 needs these three includes before windows.h
#include <stdlib.h>
#include <intrin.h>
#include <malloc.h>

#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <process.h>
#include <time.h>

#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX

#else

#include <unistd.h>
#include <dirent.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#endif
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
