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
//$Id: PathTranslator.cpp 2413 2012-08-07 13:46:06Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <boost/algorithm/string/classification.hpp>

#ifdef BOOST_SYSTEM_FOUND
#include <boost/filesystem.hpp>
#else
#include <fstream>
#endif //BOOST_SYSTEM_FOUND

#include "PathTranslator.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

vtrans::PathTranslator::PathTranslator(const std::string &locale)
    : winSeparator_('\\')
    , posixSeparator_('/')
    , prefix_()
{
    prefix_ = PathTranslator::StripLocale(locale);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::PathTranslator::StripLocale(const std::string &locale) const
{
    std::string retString = locale;
    size_t pos, separatorPos;

    pos = retString.find_first_of("=");
    if (pos != std::string::npos)
    {
        retString = retString.substr(0, pos);
    }

    pos = retString.find_first_of("@");
    if (pos != std::string::npos)
    {
        retString = retString.substr(0, pos);
    }

    pos = retString.find_last_of(".");

    std::string ws;
    ws = winSeparator_;
    std::replace_if(retString.begin(), retString.end(), boost::is_any_of(ws), posixSeparator_);

    separatorPos = retString.rfind("./");

    if (pos != std::string::npos && pos != separatorPos)
    {
        retString = retString.substr(0, pos);
    }

    return retString;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::PathTranslator::TranslatePath(const std::string &path) const
{
    std::string retPath = path;

    if (prefix_.empty())
    {
        return path;
    }

    //translate all path separarors to posix
    std::string ws;
    ws = winSeparator_;
    std::replace_if(retPath.begin(), retPath.end(), boost::is_any_of(ws), posixSeparator_);

    //insert the prefix
    size_t pos = retPath.find_last_of(posixSeparator_);
    if (pos != std::string::npos)
    {
        retPath.insert(pos, posixSeparator_ + prefix_);
    }

#ifdef _WINDOWS
    //convert all back
    std::string ps;
    ps = posixSeparator_;
    std::replace_if(retPath.begin(), retPath.end(), boost::is_any_of(ps), winSeparator_);
#endif

    //check wether the file exist
    if (!vtrans::PathTranslator::FileExists(retPath))
    {
        //fall back
        return path;
    }

    return retPath;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

bool vtrans::PathTranslator::FileExists(const std::string &path)
{
#ifdef BOOST_SYSTEM_FOUND
    return boost::filesystem::exists(path);
#else
    std::ifstream ifile(path.c_str());
    return !ifile.fail();
#endif //BOOST_SYSTEM_FOUND
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::PathTranslator::TranslatePath(const std::string &locale, const std::string &path)
{
    PathTranslator pt(locale);
    return pt.TranslatePath(path);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
