/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//
//libvtrans
//
//A library for translating Visenso applications.
//
//Visenso GmbH
//2012
//
//$Id: vtransexport.h 3468 2013-03-26 09:28:40Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once
#ifndef __VTRANSEXPORT_H__
#define __VTRANSEXPORT_H__

//--------------------------------------------------------------------------
// Wrapper for nvtrans lib's c++ interface
// to be marshalled in Unity3D.
// For deeper insight into vtrans functionality and usage
// see it's documentation.
//--------------------------------------------------------------------------
//
#ifdef WIN32
#define VTRANS_EXPORT __declspec(dllexport)
#else
#define VTRANS_EXPORT
#endif

extern "C" VTRANS_EXPORT const char *translate_path(const char *locale, const char *path);

//--------------------------------------------------------------------------

extern "C" VTRANS_EXPORT const char *translate(const char *translatorType, const char *pathToDictionary, const char *dictionaryDomain, const char *locale, const char *message);

//--------------------------------------------------------------------------

#endif //__VTRANSEXPORT_H__
