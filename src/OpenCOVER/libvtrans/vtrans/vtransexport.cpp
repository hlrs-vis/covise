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
//$Id: vtransexport.cpp 3469 2013-03-26 09:40:39Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <iostream>
#include <cstring>

#include "vtransexport.h"

#include "vtrans.h"
#include "PathTranslator.h"

static const unsigned int buffLen = 8192;

static char buff[buffLen];
static char buff_path[buffLen];

//--------------------------------------------------------------------------

extern "C" VTRANS_EXPORT const char *translate_path(const char *locale, const char *path)
{
    std::cout << "Path: " << locale << std::endl;
    strncpy(buff_path, vtrans::PathTranslator::TranslatePath(std::string(locale), std::string(path)).c_str(), buffLen);
    buff_path[buffLen - 1] = 0;
    return buff_path;
}

//--------------------------------------------------------------------------

extern "C" VTRANS_EXPORT const char *translate(const char *translatorType, const char *pathToDictionary, const char *dictionaryDomain, const char *locale, const char *message)
{
    strncpy(buff, vtrans::VTrans::translate(std::string(translatorType), std::string(pathToDictionary), std::string(dictionaryDomain), std::string(locale), std::string(message)).c_str(), buffLen);
    buff[buffLen - 1] = 0;
    return buff;
}

//--------------------------------------------------------------------------
