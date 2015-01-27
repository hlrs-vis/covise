/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//Visenso GmbH
//2012
//
//$Id: BoostTranslator.cpp 2318 2012-07-25 09:04:07Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <string>
#include <sstream>

#include "BoostTranslator.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

vtrans::BoostTranslator::BoostTranslator()
    : generator_()
    , locale_()
{
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

vtrans::BoostTranslator::BoostTranslator(const std::string &path, const std::string &domain, const std::string &language)
    : generator_()
    , locale_()
{
    configureLanguage(path, domain, language);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

vtrans::BoostTranslator::~BoostTranslator()
{
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::BoostTranslator::translate(const std::string &msg) const
{
    std::stringstream strstream;
    strstream.imbue(locale_);
    strstream << boost::locale::translate(msg);
    return strstream.str();
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

/*
std::string vtrans::BoostTranslator::translateIfPresent(const std::string& msg) const
{
    std::string retStr;
    return retStr;
}
*/

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

void vtrans::BoostTranslator::configureLanguage(const std::string &path, const std::string &domain, const std::string &language)
{
    generator_.clear_cache();
    generator_.clear_paths();
    generator_.clear_domains();
    generator_.add_messages_path(path);
    generator_.add_messages_domain(domain);
    locale_ = generator_(language);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::BoostTranslator::toUpper(const std::string &msg) const
{
    return boost::locale::to_upper(msg, locale_);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::BoostTranslator::toLower(const std::string &msg) const
{
    return boost::locale::to_lower(msg, locale_);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
