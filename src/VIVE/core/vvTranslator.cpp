/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//
// OpenCOVER kernel
//
//
// Visenso GmbH
// 2012
//
// $Id$
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <algorithm>
#include <cstdlib>

#if 0
#include <vtrans/vtrans.h>
#endif
#include "config/CoviseConfig.h"
#include "vvTranslator.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vvTranslator::coTranslate(const std::string &msg)
{
    std::string retMsg = msg;

#if 0
    char *covisedir = getenv("COVISEDIR");
    if (covisedir)
    {
        std::string coviseDir(covisedir);
        //yes, there could be a semicolon in it!
        retMsg = vtrans::VTrans::translate(covise::coCoviseConfig::getEntry("value", "VIVE.Localization.TranslatorType", ""),
                                           coviseDir + std::string("/") + covise::coCoviseConfig::getEntry("value", "VIVE.Localization.LocalePath", ""),
                                           covise::coCoviseConfig::getEntry("value", "VIVE.Localization.ModuleDomain", ""),
                                           covise::coCoviseConfig::getEntry("value", "VIVE.Localization.LanguageLocale", ""),
                                           retMsg);
    }
    else
    {
        retMsg = vtrans::VTrans::translate(covise::coCoviseConfig::getEntry("value", "VIVE.Localization.TranslatorType", ""),
                                           covise::coCoviseConfig::getEntry("value", "VIVE.Localization.LocalePath", ""),
                                           covise::coCoviseConfig::getEntry("value", "VIVE.Localization.ModuleDomain", ""),
                                           covise::coCoviseConfig::getEntry("value", "VIVE.Localization.LanguageLocale", ""),
                                           retMsg);
    }
#endif

    return retMsg;
}

std::string vvTranslator::translatePath(const std::string &path)
{
    std::string localizedPath = path;
#if 0
    localizedPath = vtrans::PathTranslator::TranslatePath(
        coCoviseConfig::getEntry("value", "VIVE.Localization.LocalePrefix", ".") + "\\"
        + coCoviseConfig::getEntry("value", "VIVE.Localization.LanguageLocale", ""),
        path);
#endif
    return localizedPath;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
