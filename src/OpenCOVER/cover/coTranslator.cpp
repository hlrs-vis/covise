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

#include <vtrans/vtrans.h>
#include "config/CoviseConfig.h"
#include "coTranslator.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string coTranslator::coTranslate(const std::string &msg)
{
    std::string retMsg = msg;

    char *covisepath = getenv("COVISE_PATH");
    if (covisepath)
    {
        std::string covisePath(covisepath);
        //yes, there could be a semicolon in it!
        covisePath.erase(std::remove(covisePath.begin(), covisePath.end(), ';'), covisePath.end());
        retMsg = vtrans::VTrans::translate(covise::coCoviseConfig::getEntry("value", "COVER.Localization.TranslatorType", ""),
                                           covisePath + std::string("/") + covise::coCoviseConfig::getEntry("value", "COVER.Localization.LocalePath", ""),
                                           covise::coCoviseConfig::getEntry("value", "COVER.Localization.ModuleDomain", ""),
                                           covise::coCoviseConfig::getEntry("value", "COVER.Localization.LanguageLocale", ""),
                                           retMsg);
    }
    else
    {
        retMsg = vtrans::VTrans::translate(covise::coCoviseConfig::getEntry("value", "COVER.Localization.TranslatorType", ""),
                                           covise::coCoviseConfig::getEntry("value", "COVER.Localization.LocalePath", ""),
                                           covise::coCoviseConfig::getEntry("value", "COVER.Localization.ModuleDomain", ""),
                                           covise::coCoviseConfig::getEntry("value", "COVER.Localization.LanguageLocale", ""),
                                           retMsg);
    }

    return retMsg;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
